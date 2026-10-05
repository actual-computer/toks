"""tests/model/unigram_model.py: the executable reading of docs/algorithms/unigram.md.

A straightforward python transcription of what hf tokenizers 0.23.2 does for the Unigram family (sentencepiece
unigram models converted to tokenizer.json): added-token extraction (both phases, lstrip / rstrip), the
normalizers these files use (Precompiled charsmap walked grapheme by grapheme, Replace, Strip), the
pre-tokenizers (WhitespaceSplit, Metaspace), the Unigram Viterbi (f64 scores parsed the way serde_json parses
them, unk penalty, fuse_unk, byte_fallback), TemplateProcessing, truncation, and the decoders (Metaspace,
Replace, ByteFallback, Fuse). Every rule here has a section in the doc; hf 0.23.2 stays the definition and
tests/unigram/run_diff.py measures the distance (0 mismatches is the bar).

Valid utf-8 only (python str); the byte-input rules of SPEC §3.3 live in the doc and the c twin's tests.
Stdlib only, so `uv run` needs nothing but tokenizers==0.23.2 for the differential.
"""

from __future__ import annotations

import base64
import json
import os
import re
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHEME_TABLE = os.path.join(HERE, "..", "data", "unigram", "grapheme17.txt")

# the 25 White_Space code points: rust char::is_whitespace and rust regex \s (hf's WhitespaceSplit, Strip,
# and the added-token lstrip / rstrip regexes \s*$ and ^\s*)
WHITE_SPACE = frozenset([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                         0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


class Unsupported(Exception):
    """a feature this model (and toks) does not run; the message names it"""


class HfError(Exception):
    """a case where hf 0.23.2 raises (MissingUnkId) or panics (charsmap slicing); the model raises too"""


# ------------------------------------------------------------------------------------------------ §2 scores
_POW10 = [float(f"1e{k}") for k in range(309)]
_U64_MAX = (1 << 64) - 1


def serde_f64(text: str) -> float:
    """serde_json 1.0.151 without float_roundtrip (hf's build): the decimal significand accumulates in a u64
    (digits past the first overflow are dropped, integer ones still count in the exponent), then
    f = (significand as f64) and ONE multiply or divide by 10^|e| (e < -308: divide by 1e308 first, repeatedly).
    Not correctly rounded: bge-m3 has 65,856 scores 1 ulp away from strtod's."""
    s = text
    i = 0
    neg = False
    if s[i] == "-":
        neg = True
        i += 1
    sig = 0
    exp = 0
    n = len(s)
    # integer part
    overflow = False
    while i < n and s[i].isdigit():
        d = ord(s[i]) - 48
        if not overflow and sig * 10 + d > _U64_MAX:
            overflow = True
        if overflow:
            exp += 1
        else:
            sig = sig * 10 + d
        i += 1
    if i < n and s[i] == ".":
        i += 1
        while i < n and s[i].isdigit():
            d = ord(s[i]) - 48
            if not overflow and sig * 10 + d > _U64_MAX:
                overflow = True
            if not overflow:
                sig = sig * 10 + d
                exp -= 1
            i += 1
    if i < n and s[i] in "eE":
        i += 1
        eneg = False
        if s[i] in "+-":
            eneg = s[i] == "-"
            i += 1
        e = 0
        while i < n and s[i].isdigit():
            e = e * 10 + ord(s[i]) - 48
            if e > 2**31 - 1:
                # parse_exponent_overflow: +inf is an error for a nonzero significand, else 0.0
                if sig != 0 and not eneg:
                    raise HfError("number out of range")
                return -0.0 if neg else 0.0
            i += 1
        exp = exp - e if eneg else exp + e
    if text.lstrip("-").isdigit() and not overflow:
        # an integer literal: ParserNumber::U64 / I64, converted by serde's f64 visitor (correctly rounded)
        return -float(sig) if neg else float(sig)
    f = float(sig)
    while True:  # bounded: exp rises by 308 per round
        if abs(exp) <= 308:
            if exp >= 0:
                f = f * _POW10[exp]
                if f == float("inf"):
                    raise HfError("number out of range")
            else:
                f = f / _POW10[-exp]
            break
        if f == 0.0:
            break
        if exp >= 0:
            raise HfError("number out of range")
        f = f / 1e308
        exp += 308
    return -f if neg else f


# --------------------------------------------------------------------------------------------- §3 graphemes
class Graphemes:
    """extended grapheme clusters, unicode-segmentation 1.13.3 (Unicode 17.0.0) exactly: GB3-GB13 and GB999,
    with GB9c's InCB rule. Tables: tests/data/unigram/grapheme17.txt (tests/unigram/gen_grapheme.py)."""

    def __init__(self, path: str = GRAPHEME_TABLE):
        self.ranges = []  # (lo, hi, cat)
        self.incb_extend = []
        self.linkers = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                if p[0] == "C":
                    self.ranges.append((int(p[1], 16), int(p[2], 16), p[3]))
                elif p[0] == "E":
                    self.incb_extend.append((int(p[1], 16), int(p[2], 16)))
                elif p[0] == "K":
                    self.linkers.add(int(p[1], 16))
        self._los = [r[0] for r in self.ranges]
        self._elos = [r[0] for r in self.incb_extend]
        self._cache = {}

    def cat(self, cp: int) -> str:
        if cp <= 0x7E:  # the crate special-cases ascii except DEL
            if cp >= 0x20:
                return "Any"
            return "LF" if cp == 0x0A else "CR" if cp == 0x0D else "Control"
        c = self._cache.get(cp)
        if c is None:
            import bisect
            k = bisect.bisect_right(self._los, cp) - 1
            c = "Any"
            if k >= 0 and self.ranges[k][0] <= cp <= self.ranges[k][1]:
                c = self.ranges[k][2]
            self._cache[cp] = c
        return c

    def is_incb_extend(self, cp: int) -> bool:
        import bisect
        k = bisect.bisect_right(self._elos, cp) - 1
        return k >= 0 and self.incb_extend[k][0] <= cp <= self.incb_extend[k][1]

    def is_boundary(self, cps, i: int) -> bool:
        """boundary between cps[i-1] and cps[i] (0 < i < len); the crate's check_pair and its look-backs"""
        b = self.cat(cps[i - 1])
        a = self.cat(cps[i])
        if b == "CR" and a == "LF":
            return False  # GB3
        if b in ("Control", "CR", "LF"):
            return True  # GB4
        if a in ("Control", "CR", "LF"):
            return True  # GB5
        if b == "L" and a in ("L", "V", "LV", "LVT"):
            return False  # GB6
        if b in ("LV", "V") and a in ("V", "T"):
            return False  # GB7
        if b in ("LVT", "T") and a == "T":
            return False  # GB8
        if a in ("Extend", "ZWJ"):
            return False  # GB9
        if a == "SpacingMark":
            return False  # GB9a
        if b == "Prepend":
            return False  # GB9b
        if a == "InCB_Consonant":  # GB9c: Consonant [Extend Linker]* Linker [Extend Linker]* x Consonant
            linkers = 0
            j = i - 1
            while j >= 0:  # bounded by i
                c = cps[j]
                if c in self.linkers:
                    linkers += 1
                elif self.is_incb_extend(c):
                    pass
                else:
                    return not (linkers > 0 and self.cat(c) == "InCB_Consonant")
                j -= 1
            return True
        if b == "ZWJ" and a == "Extended_Pictographic":  # GB11: ExtPict Extend* ZWJ x ExtPict
            j = i - 2
            while j >= 0:  # bounded by i
                c = self.cat(cps[j])
                if c == "Extend":
                    j -= 1
                    continue
                return c != "Extended_Pictographic"
            return True
        if b == "Regional_Indicator" and a == "Regional_Indicator":  # GB12 / GB13
            n = 0
            j = i - 1
            while j >= 0 and self.cat(cps[j]) == "Regional_Indicator":  # bounded by i
                n += 1
                j -= 1
            return n % 2 == 0
        return True  # GB999

    def split(self, s: str):
        cps = [ord(c) for c in s]
        out = []
        start = 0
        for i in range(1, len(cps)):  # bounded by len(s)
            if self.is_boundary(cps, i):
                out.append(s[start:i])
                start = i
        if cps:
            out.append(s[start:])
        return out


_GRAPHEMES = None


def graphemes() -> Graphemes:
    global _GRAPHEMES
    if _GRAPHEMES is None:
        _GRAPHEMES = Graphemes()
    return _GRAPHEMES


# ------------------------------------------------------------------------------------- §3 precompiled charsmap
class Charsmap:
    """spm_precompiled 0.1.4: [u32 le trie bytes][double-array units][normalized strings, NUL-terminated]"""

    def __init__(self, blob: bytes):
        if len(blob) < 4:
            raise Unsupported("precompiled charsmap: short blob")
        size = struct.unpack_from("<I", blob, 0)[0]
        n = size // 4
        if 4 + 4 * n > len(blob):
            raise Unsupported("precompiled charsmap: trie past the blob")
        self.array = list(struct.unpack_from(f"<{n}I", blob, 4))
        self.normalized = blob[4 + 4 * n:]  # hf: rest after trie_size/4 units (a trie_size % 4 tail stays here)
        try:
            self.normalized.decode("utf-8")
        except UnicodeDecodeError:
            raise Unsupported("precompiled charsmap: normalized strings are not utf-8")  # hf refuses at load
        self._memo = {}

    def transform(self, key: bytes):
        """value of the FIRST (shortest) key that is a prefix of `key`, or None. The walk stops at a NUL byte."""
        r = self._memo.get(key)
        if r is not None or key in self._memo:
            return r
        a = self.array
        node = 0
        unit = a[node]
        node ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
        res = None
        for c in key:  # bounded by len(key)
            if c == 0:
                break
            node ^= c
            unit = a[node]  # IndexError where hf panics
            if (unit & ((1 << 31) | 0xFF)) != c:
                break
            node ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
            if (unit >> 8) & 1:
                res = a[node] & ((1 << 31) - 1)
                break
        if res is not None:
            nb = self.normalized
            if res > len(nb):
                raise HfError("charsmap value offset past the strings (hf panics)")
            end = nb.find(b"\x00", res)
            if end < 0:
                end = len(nb)
            try:
                res = nb[res:end].decode("utf-8")
            except UnicodeDecodeError:
                raise HfError("charsmap value not on a char boundary (hf panics)")
        self._memo[key] = res
        return res

    def normalize(self, s: str) -> str:
        out = []
        for g in graphemes().split(s):
            gb = g.encode("utf-8")
            if len(gb) < 6:
                v = self.transform(gb)
                if v is not None:
                    out.append(v)  # the whole grapheme, whatever followed the matched prefix
                    continue
            for c in g:
                v = self.transform(c.encode("utf-8"))
                out.append(c if v is None else v)
        return "".join(out)

    def entries(self):
        """every (key bytes, value offset) of the double array, by an explicit-stack walk over labels 1..255"""
        a = self.array
        out = []
        root = a[0]
        stack = [(0 ^ ((root >> 10) << ((root & (1 << 9)) >> 6)), b"")]
        while stack:  # bounded: each node is pushed once per parent edge (a tree)
            node, key = stack.pop()
            for c in range(1, 256):
                p = node ^ c
                if p >= len(a):
                    continue
                u = a[p]
                if (u & ((1 << 31) | 0xFF)) != c:
                    continue
                nxt = p ^ ((u >> 10) << ((u & (1 << 9)) >> 6))
                k = key + bytes([c])
                if (u >> 8) & 1 and nxt < len(a):
                    out.append((k, a[nxt] & ((1 << 31) - 1)))
                if len(k) < 64:
                    stack.append((nxt, k))
        return out

    def live_keys(self):
        """keys hf can ever use: single chars, and multi-char keys of <= 5 bytes without a shorter key prefix"""
        ks = {k for k, _ in self.entries()}
        live = []
        for k in ks:
            try:
                s = k.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(s) == 1:
                live.append(s)
            elif len(k) <= 5 and not any(s[:i].encode("utf-8") in ks for i in range(1, len(s))):
                live.append(s)
        return sorted(live)


# ------------------------------------------------------------------------------------------ §4 other normalizers
def _replace_fn(pattern: dict, content: str):
    if "String" in pattern:
        lit = pattern["String"]
        if lit == "":
            raise Unsupported("normalizer Replace with an empty pattern")
        return lambda s: s.replace(lit, content)  # onig on the escaped literal: leftmost, non-overlapping
    rx = pattern["Regex"]
    if rx == " {2,}":
        return lambda s: re.sub(" {2,}", content, s)
    if rx == "(?<!\\n)^":
        # onig ruby syntax: ^ matches at 0 and after every \n; the look-behind leaves position 0 only
        return lambda s: content + s if s else s
    if len(rx) == 1 and rx not in "\\^$.|?*+()[]{}":
        return lambda s: s.replace(rx, content)
    raise Unsupported(f"normalizer Replace regex {rx!r}")


def _strip_fn(left: bool, right: bool):
    def f(s: str) -> str:
        a, b = 0, len(s)
        if left:
            while a < b and ord(s[a]) in WHITE_SPACE:
                a += 1
        if right:
            while b > a and ord(s[b - 1]) in WHITE_SPACE:
                b -= 1
        return s[a:b]
    return f


def make_normalizer(nd, found=None):
    if nd is None:
        return None
    t = nd.get("type")
    if t == "Sequence":
        fs = [make_normalizer(x, found) for x in nd["normalizers"]]
        fs = [f for f in fs if f is not None]

        def seq(s: str) -> str:
            for f in fs:
                s = f(s)
            return s
        return seq
    if t == "Precompiled":
        blob = base64.b64decode(nd["precompiled_charsmap"])
        if not blob:
            return None  # hf: an empty charsmap normalizes nothing
        cm = Charsmap(blob)
        if found is not None:
            found.append(cm)
        return cm.normalize
    if t == "Replace":
        return _replace_fn(nd["pattern"], nd["content"])
    if t == "Strip":
        return _strip_fn(bool(nd.get("strip_left", True)), bool(nd.get("strip_right", True)))
    raise Unsupported(f"normalizer {t}")


# -------------------------------------------------------------------------------------------- §5 pre-tokenizers
def _whitespace_split(s: str):
    out = []
    cur = []
    for c in s:
        if ord(c) in WHITE_SPACE:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


class Metaspace:
    def __init__(self, d: dict):
        self.rep = d.get("replacement", "\u2581")
        scheme = d.get("prepend_scheme")
        aps = d.get("add_prefix_space")
        if scheme is None:
            scheme = "always"
        if aps is False:
            if scheme != "never":
                raise Unsupported("Metaspace add_prefix_space does not match prepend_scheme")  # hf refuses
            scheme = "never"
        self.scheme = scheme
        self.split = d.get("split", True)
        if len(self.rep) != 1:
            raise Unsupported("Metaspace replacement must be one char")

    def run(self, s: str, at_original_start: bool):
        s = s.replace(" ", self.rep)
        if self.scheme == "always" or (self.scheme == "first" and at_original_start):
            if not s.startswith(self.rep):
                s = self.rep + s
        if not self.split:
            return [s] if s else []
        out = []
        start = 0
        for i, c in enumerate(s):  # MergedWithNext on the replacement char
            if c == self.rep and i > start:
                out.append(s[start:i])
                start = i
        if start < len(s):
            out.append(s[start:])
        return out


def make_pretok(pd):
    """a list of steps: ('ws',) or ('meta', Metaspace)"""
    if pd is None:
        return []
    t = pd.get("type")
    if t == "Sequence":
        steps = []
        for x in pd["pretokenizers"]:
            steps.extend(make_pretok(x))
        return steps
    if t == "WhitespaceSplit":
        return [("ws",)]
    if t == "Metaspace":
        return [("meta", Metaspace(pd))]
    raise Unsupported(f"pre_tokenizer {t}")


# ----------------------------------------------------------------------------------------------- §6 the model
class Unigram:
    def __init__(self, md: dict, md_scores_text):
        vocab = md["vocab"]
        self.pieces = [p for p, _ in vocab]
        self.scores = [serde_f64(t) for t in md_scores_text]
        self.unk_id = md.get("unk_id")
        self.byte_fallback = bool(md.get("byte_fallback", False))
        if self.unk_id is not None and (not vocab or self.unk_id >= len(vocab)):
            raise Unsupported("Unigram unk_id outside the vocabulary")  # hf refuses at load
        self.tok2id = {}
        for i, p in enumerate(self.pieces):
            self.tok2id[p] = i  # duplicates: the last id wins (hf's HashMap insert)
        self.min_score = min(self.scores) if self.scores else float("inf")
        self.unk_score = self.min_score - 10.0
        self.maxlen = max((len(p) for p in self.pieces), default=0)
        # every prefix of every piece: the common-prefix search stops where no piece continues (a trie walk)
        self.prefixes = set()
        for p in self.pieces:
            for k in range(1, len(p) + 1):
                self.prefixes.add(p[:k])
        self._cache = {}  # piece -> ids; hf caches too, the cache never changes ids

    def viterbi(self, piece: str):
        """hf Unigram::encode_optimized: list of (start, end, id) on the best path, before unk fusion"""
        n = len(piece)
        best = [0.0] * (n + 1)
        frm = [None] * (n + 1)
        ids = [0] * (n + 1)
        tok2id = self.tok2id
        scores = self.scores
        prefixes = self.prefixes
        for s in range(n):  # bounded by n; every char start in order
            base = best[s]
            single = False
            e = s + 1
            while e <= n:  # common prefix search, shortest first; bounded by n
                sub = piece[s:e]
                if sub not in prefixes:
                    break
                i = tok2id.get(sub)
                if i is not None:
                    cand = scores[i] + base
                    if frm[e] is None or cand > best[e]:
                        best[e] = cand
                        frm[e] = s
                        ids[e] = i
                    if e == s + 1:
                        single = True
                e += 1
            if not single:
                e = s + 1
                cand = self.unk_score + base
                if frm[e] is None or cand > best[e]:
                    if self.unk_id is None:
                        raise HfError("MissingUnkId")
                    best[e] = cand
                    frm[e] = s
                    ids[e] = self.unk_id
        path = []
        e = n
        while e > 0:  # bounded: e strictly decreases
            s = frm[e]
            path.append((s, e, ids[e]))
            e = s
        path.reverse()
        return path

    def tokenize(self, piece: str):
        if not piece:
            return []
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        if len(self._cache) > 200_000:
            self._cache.clear()
        out = self._tokenize(piece)
        self._cache[piece] = out
        return out

    def _tokenize(self, piece: str):
        path = self.viterbi(piece)
        strings = []
        k = 0
        while k < len(path):  # fuse_unk: consecutive unk nodes become one string
            s, e, i = path[k]
            if self.unk_id is not None and i == self.unk_id:
                j = k
                while j + 1 < len(path) and path[j + 1][2] == self.unk_id:
                    j += 1
                strings.append(piece[s:path[j][1]])
                k = j + 1
            else:
                strings.append(piece[s:e])
                k += 1
        out = []
        for st in strings:
            i = self.tok2id.get(st)
            if i is not None:
                out.append(i)
                continue
            if self.byte_fallback:
                bt = [self.tok2id.get(f"<0x{b:02X}>") for b in st.encode("utf-8")]
                if all(x is not None for x in bt):
                    out.extend(bt)
                    continue
            if self.unk_id is None:
                raise HfError("MissingUnkId")
            out.append(self.unk_id)
        return out


# ------------------------------------------------------------------------------------------- §7 added tokens
class AddedTok:
    __slots__ = ("content", "id", "special", "lstrip", "rstrip", "single_word", "normalized", "pattern")


def _leading_ws(s: str, i: int) -> int:
    j = i
    while j < len(s) and ord(s[j]) in WHITE_SPACE:
        j += 1
    return j


def _trailing_ws_start(s: str, i: int) -> int:
    j = i
    while j > 0 and ord(s[j - 1]) in WHITE_SPACE:
        j -= 1
    return j


def find_matches(text: str, toks, drop_specials: bool):
    """hf AddedVocabulary::find_matches: daachorse leftmost-longest, then the per-match policy.
    returns [(id or None, a, b)] covering text (char offsets)"""
    if not toks:
        return [(None, 0, len(text))] if text else []
    by_first = {}
    for t in toks:
        by_first.setdefault(t.pattern[0], []).append(t)
    for v in by_first.values():
        v.sort(key=lambda t: -len(t.pattern))
    out = []
    start_offset = 0
    i = 0
    n = len(text)
    while i < n:  # bounded: i strictly increases
        cand = by_first.get(text[i])
        m = None
        if cand:
            for t in cand:
                if text.startswith(t.pattern, i):
                    m = t
                    break
        if m is None:
            i += 1
            continue
        a, b = i, i + len(m.pattern)
        i = b  # resume at the raw match end
        if drop_specials and m.special:
            continue
        if m.single_word:
            raise Unsupported("added token single_word")  # no census unigram tokenizer uses it
        if m.lstrip:
            a = max(_trailing_ws_start(text, a), start_offset)
        if m.rstrip:
            b = _leading_ws(text, b)
        if start_offset < a:
            out.append((None, start_offset, a))
        out.append((m.id, a, b))
        start_offset = b
    if start_offset != n:
        out.append((None, start_offset, n))
    return out


# ------------------------------------------------------------------------------------------------ the pipeline
class Tokenizer:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        tj = json.loads(raw)
        tjs = json.loads(raw, parse_float=str, parse_int=str)
        md = tj["model"]
        if md.get("type", "Unigram") != "Unigram":
            raise Unsupported(f"model {md.get('type')}")
        self.model = Unigram(md, [s for _, s in tjs["model"]["vocab"]])
        found = []
        self.normalizer = make_normalizer(tj.get("normalizer"), found)
        self.charsmap = found[0] if found else None
        self.pretok = make_pretok(tj.get("pre_tokenizer"))
        self._added(tj.get("added_tokens") or [])
        self._post(tj.get("post_processor"))
        self.truncation = tj.get("truncation")
        pad = tj.get("padding")
        if pad is not None and pad.get("strategy") != "BatchLongest":
            raise Unsupported("padding Fixed")  # BatchLongest is a no-op for one sequence
        if pad is not None and pad.get("pad_to_multiple_of"):
            raise Unsupported("padding pad_to_multiple_of")
        self.decoder = tj.get("decoder")

    # hf AddedVocabulary::add_tokens: the json id is ignored; model vocab id, else the next id from vocab_size
    def _added(self, lst):
        nvocab = len(self.model.pieces)
        next_id = nvocab
        by_content = {}
        self.added = []
        for d in lst:  # file order
            c = d["content"]
            if c == "":
                continue
            if c in by_content:
                tid = by_content[c]
            elif c in self.model.tok2id:
                tid = self.model.tok2id[c]
            else:
                tid = next_id
                next_id += 1
            by_content[c] = tid
            t = AddedTok()
            t.content = c
            t.id = tid
            t.special = bool(d.get("special", False))
            t.lstrip = bool(d.get("lstrip", False))
            t.rstrip = bool(d.get("rstrip", False))
            t.single_word = bool(d.get("single_word", False))
            t.normalized = bool(d.get("normalized", not t.special))
            t.pattern = c
            if t.normalized and self.normalizer is not None:
                t.pattern = self.normalizer(c)  # hf's normalized_cache (matching form)
            self.added = [x for x in self.added if x.id != tid] + [t]
        self.id2added = {t.id: t for t in self.added}
        self.specials = {t.content for t in self.added if t.special}
        self.p0 = [t for t in self.added if not t.normalized and t.pattern]
        self.p1 = [t for t in self.added if t.normalized and t.pattern]

    def _post(self, pp):
        self.prefix, self.suffix = [], []
        if pp is None:
            return
        if pp.get("type") != "TemplateProcessing":
            raise Unsupported(f"post_processor {pp.get('type')}")
        side = self.prefix
        for item in pp["single"]:
            if "Sequence" in item:
                side = self.suffix
                continue
            name = item["SpecialToken"]["id"]
            side.extend(int(x) for x in pp["special_tokens"][name]["ids"])

    # §5: the pre-tokenizer chain on one normalized segment
    def pre_tokenize(self, s: str, at_original_start: bool):
        splits = [(s, at_original_start)]
        for k, step in enumerate(self.pretok):
            nxt = []
            for (x, first) in splits:
                if step[0] == "ws":
                    nxt.extend((p, False) for p in _whitespace_split(x))
                else:
                    if step[1].scheme == "first" and k > 0:
                        # the split's original offset after an earlier splitter: an spm case, not modelled here
                        raise Unsupported("Metaspace prepend_scheme=first after another pre-tokenizer")
                    nxt.extend((p, False) for p in step[1].run(x, first))
            splits = [(x, f) for (x, f) in nxt if x]
        return [x for x, _ in splits]

    def encode(self, text: str, add_special_tokens: bool = True, truncate: bool = True, drop_specials: bool = False):
        ids = []
        for (tid, a, b) in find_matches(text, self.p0, drop_specials):
            if tid is not None:
                ids.append(tid)
                continue
            seg = text[a:b]
            if self.normalizer is not None:
                seg = self.normalizer(seg)
            if not seg:
                continue
            for (tid1, a1, b1) in find_matches(seg, self.p1, drop_specials):
                if tid1 is not None:
                    ids.append(tid1)
                    continue
                for piece in self.pre_tokenize(seg[a1:b1], a == 0 and a1 == 0):
                    ids.extend(self.model.tokenize(piece))
        if truncate and self.truncation is not None:
            n_added = len(self.prefix) + len(self.suffix) if add_special_tokens else 0
            ml = int(self.truncation["max_length"]) - n_added
            if ml < 0:
                raise HfError("truncation max_length below the added tokens")
            if self.truncation.get("direction", "Right") == "Right":
                ids = ids[:ml]
            else:
                ids = ids[len(ids) - ml:] if len(ids) > ml else ids
        if add_special_tokens:
            ids = self.prefix + ids + self.suffix
        return ids

    # §8 decode
    def id_to_token(self, i: int):
        t = self.id2added.get(i)
        if t is not None:
            return t.pattern if t.normalized and t.pattern != t.content else t.content
        if 0 <= i < len(self.model.pieces):
            return self.model.pieces[i]
        return None

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        toks = []
        for i in ids:
            t = self.id_to_token(i)
            if t is None:
                continue
            if skip_special_tokens and t in self.specials:
                continue
            toks.append(t)
        if self.decoder is None:
            return " ".join(toks)
        return "".join(decode_chain(self.decoder, toks))


def _byte_token(t: str):
    if len(t.encode("utf-8")) == 6 and t.startswith("<0x") and t.endswith(">"):
        mid = t[3:5]
        if len(mid.encode("utf-8")) != 2:
            return None
        try:
            if mid[0] == "+":  # rust u8::from_str_radix takes one leading '+'
                v = int(mid[1], 16)
            else:
                v = int(mid, 16) if all(c in "0123456789abcdefABCDEF" for c in mid) else None
        except ValueError:
            return None
        return v
    return None


def decode_chain(dd: dict, toks):
    t = dd.get("type")
    if t == "Sequence":
        for d in dd["decoders"]:
            toks = decode_chain(d, toks)
        return toks
    if t == "Metaspace":
        m = Metaspace(dd)
        out = []
        for k, tok in enumerate(toks):
            out.append("".join(("" if (k == 0 and m.scheme != "never") else " ") if c == m.rep else c for c in tok))
        return out
    if t == "Replace":
        pat = dd["pattern"]
        content = dd["content"]
        if "String" in pat:
            return [x.replace(pat["String"], content) for x in toks]
        rx = pat["Regex"]
        if rx == "(?<!\\n)^ ":
            return [content + x[1:] if x.startswith(" ") else x for x in toks]
        if len(rx) == 1 and rx not in "\\^$.|?*+()[]{}":
            return [x.replace(rx, content) for x in toks]
        raise Unsupported(f"decoder Replace regex {rx!r}")
    if t == "ByteFallback":
        out = []
        pend = []
        for tok in toks:
            b = _byte_token(tok)
            if b is not None:
                pend.append(b)
                continue
            if pend:
                try:
                    out.append(bytes(pend).decode("utf-8"))
                except UnicodeDecodeError:
                    out.extend(["\ufffd"] * len(pend))
                pend = []
            out.append(tok)
        if pend:
            try:
                out.append(bytes(pend).decode("utf-8"))
            except UnicodeDecodeError:
                out.extend(["\ufffd"] * len(pend))
        return out
    if t == "Fuse":
        return ["".join(toks)]
    raise Unsupported(f"decoder {t}")
