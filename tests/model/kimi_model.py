#!/usr/bin/env python3
"""tests/model/kimi_model.py: the executable reading of docs/models/kimi.md.

Kimi K3 = transformers AutoTokenizer(trust_remote_code=True) -> tokenization_kimi.TikTokenTokenizer -> tiktoken.
Every rule of the doc is implemented here in the most direct way, from the model's own files (never from the
oracle's objects), so that tests/model/run_kimi_fuzz.py can diff it against the real thing:

  section 3  the files: tiktoken.model (base64 token, rank), tokenizer_config.json (added_tokens_decoder, the
             special-token attributes), tokenization_kimi.py (its constants, checked verbatim)
  section 4  classes: probed through tiktoken's own regex engine (the 0.14.0 wheel) over every scalar value,
             cached in build/kimi/kimi_probes.pkl; the doc's class claims are asserted on every scalar
  section 5  the scanner: rules K1-K7 over atoms and classes (leftmost-first semantics written out)
  section 6  bpe: the whole piece if it is a token, else lowest rank of the concatenation, leftmost on ties
  section 7  tiktoken's special matching inside one chunk
  section 2  the wrapper's chunking (400,000 chars, then runs of <= 25,000 str.isspace / non-isspace chars),
             the trie split of the serving path, and the three modes

    from kimi_model import Model
    m = Model()                       # needs tiktoken==0.14.0 importable for the probes (once; then cached)
    m.encode(text, "serving")         # "serving" (toks ALL), "direct" (NONSPECIAL), "none" (NONE)
    m.pieces(segment_text)            # byte (start, end) pieces of one regex segment (rules K1-K7)
"""
from __future__ import annotations

import base64
import heapq
import json
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE_PATH = os.environ.get("TOKS_KIMI_PROBES", os.path.join(ROOT, "build", "kimi", "kimi_probes.pkl"))
KIMI_DIR = os.path.expanduser(os.environ.get("TOKS_KIMI_DIR", "~/.cache/toks/kimik3"))

# ---------------------------------------------------------------- section 3: the wrapper, checked verbatim
# tokenization_kimi.py must contain these lines exactly (docs/models/kimi.md section 3): the model's pattern
# alternatives, the reserved-special count and the two chunking constants. Anything else is another wrapper.
WRAPPER_LINES = [
    '    num_reserved_special_tokens = 256',
    '        r"""[\\p{Han}]+""",',
    '        r"""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)?""",',
    '        r"""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)?""",',
    '        r"""\\p{N}{1,3}""",',
    '        r""" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*""",',
    '        r"""\\s*[\\r\\n]+""",',
    '        r"""\\s+(?!\\S)""",',
    '        r"""\\s+""",',
    '        TIKTOKEN_MAX_ENCODE_CHARS = 400_000',
    '        MAX_NO_WHITESPACES_CHARS = 25_000',
]
CHUNK_CHARS = 400_000
RUN_CHARS = 25_000
N_RESERVED = 256

# python 3.12's str.isspace() (docs/models/kimi.md section 2.3): bidi class WS, B or S, or category Zs.
ISSPACE = frozenset([*range(0x09, 0x0E), *range(0x1C, 0x21), 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                     0x2028, 0x2029, 0x202F, 0x205F, 0x3000])

# base classes (layout.h TOKS_C_*)
P, L, N, WS, NL = 0, 1, 2, 3, 4


def read_ranks(path):
    """tiktoken.model per docs/models/kimi.md section 3.1 (the strict reading toks does): one 'token rank'
    line per token, canonical base64, ranks distinct; the reference's own reader is looser (section 3.1)."""
    ranks = {}
    with open(path, "rb") as f:
        data = f.read()
    seen = set()
    for ln in data.split(b"\n"):
        if not ln:
            continue
        t, r = ln.split(b" ")
        b = base64.b64decode(t, validate=True)
        if base64.b64encode(b) != t or not r.isdigit() or b in ranks or int(r) in seen:
            raise ValueError(f"tiktoken.model line {ln!r}")
        ranks[b] = int(r)
        seen.add(int(r))
    return ranks


def read_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_wrapper(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for w in WRAPPER_LINES:
        if w not in lines:
            raise ValueError(f"tokenization_kimi.py lacks the line {w!r}")


# ---------------------------------------------------------------- section 4: classes through tiktoken
PROBES = {
    # the pattern's own sets, as written in it
    "H": r"[\p{Han}]",
    "UP": r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
    "LO": r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]",
    "PFX": r"[^\r\n\p{L}\p{N}]",
    "PUN": r"[^\s\p{L}\p{N}]",
    "N": r"\p{N}",
    "S": r"\s",
    "NOTS": r"\S",
    # their parts (the doc's derivation)
    "L": r"\p{L}", "M": r"\p{M}", "Lu": r"\p{Lu}", "Ll": r"\p{Ll}", "Lt": r"\p{Lt}", "Lm": r"\p{Lm}",
    "Lo": r"\p{Lo}",
    # the (?i) folds of the contraction letters
    **{"fold_" + c: "(?i:%s)" % c for c in "stmdrevl"},
}
CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
SCALARS = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]


def _tiktoken_matches(pat, text):
    """the concatenated text of every match of pat in text, through tiktoken's own engine: an Encoding whose
    tokens are the 256 bytes (rank = byte) returns the matched bytes; the text between matches is dropped."""
    import tiktoken
    enc = tiktoken.Encoding("probe", pat_str=pat, mergeable_ranks={bytes([b]): b for b in range(256)},
                            special_tokens={})
    return bytes(enc.encode_ordinary(text)).decode("utf-8")


def run_probes(force=False):
    if os.path.exists(PROBE_PATH) and not force:
        with open(PROBE_PATH, "rb") as f:
            return pickle.load(f)
    import tiktoken
    pr = {"tiktoken": tiktoken.__version__}
    text = "".join(map(chr, SCALARS))
    for name, pat in PROBES.items():
        pr[name] = frozenset(map(ord, _tiktoken_matches(pat, text)))
        print(f"probe {name:7s} {len(pr[name]):7d}", file=sys.stderr, flush=True)
    # the contraction group, every scalar X in 'X 'Xe 'Xl (NUL-separated: no match spans two items)
    for ctx, tail in (("c1", ""), ("ce", "e"), ("cl", "l")):
        items = ["'" + chr(c) + tail for c in SCALARS if c not in (0x27, 0)]
        got = _tiktoken_matches(CONTR, "\x00".join(items))
        pr["contr_" + ctx] = frozenset(m[0] for m in [s for s in got.split("'") if s])
    os.makedirs(os.path.dirname(PROBE_PATH), exist_ok=True)
    with open(PROBE_PATH, "wb") as f:
        pickle.dump(pr, f)
    return pr


class Classes:
    """per code point: base class and the flags the rules read (docs/models/kimi.md section 4)."""

    def __init__(self, pr):
        self.pr = pr
        Ls, Ns, Ss, Hs = pr["L"], pr["N"], pr["S"], pr["H"]
        base = bytearray(0x110000)
        for cp in range(0x110000):
            if cp in Ls:
                base[cp] = L
            elif cp in Ns:
                base[cp] = N
            elif cp in (0x0A, 0x0D):
                base[cp] = NL
            elif cp in Ss:
                base[cp] = WS
            else:
                base[cp] = P
        self.base = base
        self.up, self.lo, self.han = pr["UP"], pr["LO"], Hs
        self.fold = {c: pr["fold_" + c] for c in "stmdrevl"}

    def check(self):
        """the doc's class claims, on every scalar (raises on the first false one)."""
        pr, b = self.pr, self.base
        S = set(SCALARS)
        lu, ll, lt, lm, lo, m = pr["Lu"], pr["Ll"], pr["Lt"], pr["Lm"], pr["Lo"], pr["M"]
        claims = {
            "L == Lu|Ll|Lt|Lm|Lo": pr["L"] == (lu | ll | lt | lm | lo),
            "UP == (Lu|Lt|Lm|Lo|M) - Han": pr["UP"] == (lu | lt | lm | lo | m) - pr["H"],
            "LO == (Ll|Lm|Lo|M) - Han": pr["LO"] == (ll | lm | lo | m) - pr["H"],
            "PFX == base in {P, WS}": pr["PFX"] == {c for c in S if b[c] in (P, WS)},
            "PUN == base P": pr["PUN"] == {c for c in S if b[c] == P},
            "\\S == not \\s": pr["NOTS"] == S - pr["S"],
            "L, N, \\s disjoint": not (pr["L"] & pr["N"]) and not (pr["S"] & (pr["L"] | pr["N"])),
            "CR LF in \\s": {0x0A, 0x0D} <= pr["S"],
            "\\s has 25 scalars": len(pr["S"]) == 25,
            "fold s = {s, S, U+017F}": pr["fold_s"] == {0x73, 0x53, 0x17F},
            "other folds ascii pairs": all(pr["fold_" + c] == {ord(c), ord(c.upper())} for c in "tmdrevl"),
            "contraction letters": pr["contr_c1"] == set("sStTmMdD\u017f") and
                                   pr["contr_ce"] == set("sStTmMdD\u017frRvV") and
                                   pr["contr_cl"] == set("sStTmMdD\u017flL"),
        }
        bad = [k for k, v in claims.items() if not v]
        if bad:
            raise AssertionError("class claims fail: " + ", ".join(bad))
        return claims

    def fold_s(self, cp):
        return cp in self.fold["s"]


# ---------------------------------------------------------------- section 5: the scanner (rules K1-K7)
class Scanner:
    def __init__(self, cls: Classes):
        self.c = cls

    def pieces(self, s: str):
        """the pieces tiktoken's regex gives one segment s (str), as byte (start, end) offsets."""
        cps = [ord(ch) for ch in s]
        n = len(cps)
        offs = [0] * (n + 1)
        for i, cp in enumerate(cps):
            offs[i + 1] = offs[i] + (1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4)
        base, up, lo, han = self.c.base, self.c.up, self.c.lo, self.c.han
        fold = self.c.fold

        def B(j):
            return base[cps[j]] if j < n else None

        def is_up(j):
            return j < n and cps[j] in up

        def is_lo(j):
            return j < n and cps[j] in lo

        def u(sx):                      # end of the maximal UP run from sx
            while sx < n and cps[sx] in up:
                sx += 1
            return sx

        def lo_run(sx):                 # end of the maximal LO run from sx
            while sx < n and cps[sx] in lo:
                sx += 1
            return sx

        def K(e):                       # the contraction suffix at e: its end
            if e < n and cps[e] == 0x27 and e + 1 < n:
                c1 = cps[e + 1]
                if c1 in fold["s"] or c1 in fold["t"] or c1 in fold["m"] or c1 in fold["d"]:
                    return e + 2
                if e + 2 < n:
                    c2 = cps[e + 2]
                    if (c1 in fold["r"] and c2 in fold["e"]) or (c1 in fold["v"] and c2 in fold["e"]) or \
                       (c1 in fold["l"] and c2 in fold["l"]):
                        return e + 3
            return e

        def L1(sx):                     # [UP]*[LO]+ K?  from sx, or None
            if sx >= n:
                return None
            k = u(sx)
            if is_lo(k):
                return K(lo_run(k))
            for q in range(k - 1, sx - 1, -1):
                if cps[q] in lo:
                    return K(q + 1)
            return None

        def L2(sx):                     # [UP]+[LO]* K?  from sx, or None
            if not is_up(sx):
                return None
            return K(lo_run(u(sx)))

        out = []
        i = 0
        while i < n:
            c = cps[i]
            b = base[c]
            e = None
            # K1 Han
            if c in han:
                e = i
                while e < n and cps[e] in han:
                    e += 1
            # K2 letters (the two case alternatives, prefix taken first)
            if e is None and b in (P, WS):
                for f in (lambda: L1(i + 1), lambda: L1(i), lambda: L2(i + 1)):
                    e = f()
                    if e is not None:
                        break
            elif e is None and b == L:
                e = L1(i)
                if e is None:
                    e = L2(i)
            # K3 digits
            if e is None and b == N:
                e = i
                while e < n and e < i + 3 and base[cps[e]] == N:
                    e += 1
            # K4 punctuation, tail [\r\n]*
            if e is None and (b == P or (c == 0x20 and B(i + 1) == P)):
                e = i if b == P else i + 1
                while e < n and base[cps[e]] == P:
                    e += 1
                while e < n and base[cps[e]] == NL:
                    e += 1
            # K5-K7 whitespace
            if e is None and b in (WS, NL):
                k = i
                while k < n and base[cps[k]] in (WS, NL):
                    k += 1
                t = k - 1
                while t >= i and base[cps[t]] != NL:
                    t -= 1
                if t >= i:
                    e = t + 1                   # K5
                elif k == n:
                    e = k                       # K6 at the segment end
                elif k - i >= 2:
                    e = k - 1                   # K6 gives one atom back
                else:
                    e = k                       # K7
            if e is None or e <= i:
                raise AssertionError(f"no rule at {i} of {s!r}")
            out.append((offs[i], offs[e]))
            i = e
        return out


# ---------------------------------------------------------------- the model
class Model:
    def __init__(self, d=KIMI_DIR, probes=None):
        check_wrapper(os.path.join(d, "tokenization_kimi.py"))
        self.ranks = read_ranks(os.path.join(d, "tiktoken.model"))
        cfg = read_config(os.path.join(d, "tokenizer_config.json"))
        nb = len(self.ranks)
        if sorted(self.ranks.values()) != list(range(nb)):
            raise ValueError("ranks are not 0..n-1 (the wrapper numbers specials from len(ranks))")
        dec = {int(k): v["content"] for k, v in cfg.get("added_tokens_decoder", {}).items()}
        # section 3.2: the 256 tiktoken specials, named from added_tokens_decoder, else <|reserved_token_i|>
        self.specials = {dec.get(i, f"<|reserved_token_{i}|>"): i for i in range(nb, nb + N_RESERVED)}
        if len(self.specials) != N_RESERVED:
            raise ValueError("two specials share a name")
        # section 3.3: the trie (transformers' added tokens): added_tokens_decoder, plus the special-token
        # attributes the config names that are not in it (they would get the next free id)
        self.trie = {v: k for k, v in dec.items()}
        attrs = [cfg.get(a) for a in ("bos_token", "eos_token", "unk_token", "pad_token")]
        attrs += list(cfg.get("additional_special_tokens") or [])
        for a in attrs:
            if a is not None and a not in self.trie:
                raise ValueError(f"special-token attribute {a!r} outside added_tokens_decoder (unmodelled)")
        for name, i in self.trie.items():
            if self.specials.get(name) != i:
                raise ValueError(f"added token {name!r} is not the tiktoken special of id {i}")
        self._check_no_overlap(self.specials)
        self._check_strings()
        # literal searches only (names never overlap, so any leftmost search finds the same occurrences)
        self._special_re = re.compile("|".join(map(re.escape, self.specials)))
        self._trie_re = re.compile("|".join(map(re.escape, self.trie)))
        self.cls = Classes(probes if probes is not None else run_probes())
        self.scanner = Scanner(self.cls)
        self.id2bytes = {v: k for k, v in self.ranks.items()}
        for name, i in self.specials.items():
            self.id2bytes[i] = name.encode("utf-8")

    # -- section 3: facts the doc relies on, checked on the files
    @staticmethod
    def _check_no_overlap(names):
        """no name occurs inside another and no proper suffix of one is a proper prefix of another: every
        occurrence of every name is disjoint from every other, so leftmost scanning finds them all."""
        ns = list(names)
        for a in ns:
            for b in ns:
                if a != b and a in b:
                    raise ValueError(f"special {a!r} inside {b!r}")
                if a != b:
                    for k in range(1, min(len(a), len(b))):
                        if a[-k:] == b[:k]:
                            raise ValueError(f"specials {a!r} and {b!r} overlap")

    def _check_strings(self):
        """the serving path's id -> string -> id round trip is the identity (docs/models/kimi.md 2.2): the
        wrapper's decoder strings (bytes_to_unicode of each token's bytes) are distinct over all 163,840 ids."""
        bc = _bytes_char()
        seen = {}
        items = list(self.ranks.items()) + [(k.encode("utf-8"), v) for k, v in self.specials.items()]
        for b, i in items:
            s = "".join(bc[x] for x in b)
            if s in seen:
                raise ValueError(f"ids {seen[s]} and {i} share the string {s!r}")
            seen[s] = i

    # -- section 6: bpe
    def bpe(self, piece: bytes):
        """the whole piece if it is a token; else start from its bytes and repeatedly merge the adjacent pair
        whose concatenation has the lowest rank, the leftmost on ties, until no adjacent pair concatenates to
        a token. (A heap of (rank, start) with stale entries skipped picks exactly that pair each round.)"""
        r = self.ranks
        if piece in r:
            return [r[piece]]
        n = len(piece)
        end = list(range(1, n + 1))             # end[i]: end of the part starting at i (i alive)
        prev = list(range(-1, n - 1))           # prev[i]: start of the part before the one at i
        alive = [True] * n
        heap = []

        def push(i):
            j = end[i]
            if j < n:
                rk = r.get(piece[i:end[j]])
                if rk is not None:
                    heapq.heappush(heap, (rk, i, end[j]))

        for i in range(n - 1):
            push(i)
        while heap:
            rk, i, k = heapq.heappop(heap)
            if not alive[i] or end[i] >= n or end[end[i]] != k:
                continue                        # stale: the pair at i is no longer (i, end[i], k)
            j = end[i]
            alive[j] = False
            end[i] = k
            if k < n:
                prev[k] = i
            if prev[i] >= 0:
                push(prev[i])
            push(i)
        out, i = [], 0
        while i < n:
            out.append(r[piece[i:end[i]]])
            i = end[i]
        return out

    def pieces(self, s):
        return self.scanner.pieces(s)

    # -- section 7: one chunk through tiktoken (specials, then regex pieces, then bpe)
    def tiktoken_chunk(self, s: str, allow: bool):
        ids = []
        names = self.specials if allow else {}
        pos = 0
        while True:
            best = None
            if names:                               # leftmost occurrence (occurrences never overlap)
                m = self._special_re.search(s, pos)
                best = (m.start(), m.group()) if m else None
            end = best[0] if best else len(s)
            seg = s[pos:end]
            b = seg.encode("utf-8")
            for (x, y) in self.scanner.pieces(seg):
                ids.extend(self.bpe(b[x:y]))
            if best is None:
                return ids
            ids.append(names[best[1]])
            pos = best[0] + len(best[1])

    # -- section 2: the wrapper's chunking
    @staticmethod
    def cuts(s: str):
        """the substrings _encode_text_piece hands tiktoken: 400,000-char chunks, each cut where a run of
        same-isspace chars passes 25,000 (the 25,001st char starts a new substring)."""
        out = []
        for c0 in range(0, len(s), CHUNK_CHARS):
            ch = s[c0:c0 + CHUNK_CHARS]
            start, run, prev = 0, 0, (ord(ch[0]) in ISSPACE) if ch else False
            for i, x in enumerate(ch):
                sp = ord(x) in ISSPACE
                if sp != prev:
                    run, prev = 1, sp
                else:
                    run += 1
                    if run > RUN_CHARS:
                        out.append(ch[start:i])
                        start, run = i, 1
            out.append(ch[start:])
        if not s:
            out.append("")
        return out

    def wrapper(self, s: str, allow: bool):
        ids = []
        for sub in self.cuts(s):
            ids.extend(self.tiktoken_chunk(sub, allow))
        return ids

    def trie_split(self, s: str):
        """[(is_token, text)]: s cut at every occurrence of a trie token (they never overlap)."""
        out, pos = [], 0
        while True:
            m = self._trie_re.search(s, pos)
            best = (m.start(), m.group()) if m else None
            if best is None:
                if pos < len(s):
                    out.append((False, s[pos:]))
                return out
            if best[0] > pos:
                out.append((False, s[pos:best[0]]))
            out.append((True, best[1]))
            pos = best[0] + len(best[1])

    def encode(self, s: str, mode: str = "serving"):
        if mode == "direct":
            return self.wrapper(s, True)
        if mode == "none":
            return self.wrapper(s, False)
        if mode == "serving":
            ids = []
            for is_tok, t in self.trie_split(s):
                ids.extend([self.trie[t]] if is_tok else self.wrapper(t, True))
            return ids
        if mode == "tiktoken":
            return self.tiktoken_chunk(s, True)
        if mode == "ordinary":
            return self.tiktoken_chunk(s, False)
        raise ValueError(mode)

    # -- section 8: decode (no kwargs: tiktoken's bytes, utf-8 with replacement)
    def decode(self, ids):
        return b"".join(self.id2bytes[i] for i in ids).decode("utf-8", "replace")


def _bytes_char():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


if __name__ == "__main__":
    pr = run_probes(force="--force" in sys.argv)
    cl = Classes(pr)
    for k in cl.check():
        print("ok  ", k)
    print("Han", len(pr["H"]), "UP", len(pr["UP"]), "LO", len(pr["LO"]), "L", len(pr["L"]), "N", len(pr["N"]),
          "M", len(pr["M"]), "\\s", len(pr["S"]), "PFX", len(pr["PFX"]), "PUN", len(pr["PUN"]))
