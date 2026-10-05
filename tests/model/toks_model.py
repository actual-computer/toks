#!/usr/bin/env python3
"""Review harness: model of docs/kernels.md (the contract merged in commit 9fa466c) vs hf tokenizers 0.23.2.

Everything the doc specifies is implemented here in the most straightforward way:
 - §2/§3: atoms, classes (probed through hf's own onig regex, never unicodedata),
   the cl100k template rules A1-A7 parameterised exactly as layout.h TOKS_TP_*.
 - §4: added-token find + driver policy, in two flavours: 'hf' mirrors
   added_vocabulary.rs find_matches exactly (match per start position, resume at
   start+1); 'doc' mirrors kernels.md §4 as written (resume at raw m_end after
   every match, dropped or accepted).
 - §5: bpe = linear lowest-rank-leftmost merging, ignore_merges via model vocab,
   duplicate merge pairs keep the last.
Ground truth for classes comes from probes through hf Split (the same onig
engine the real patterns run under). Any mismatch of the 'probed' model vs hf
is a rule defect in the doc; deltas between probed classes and the doc's claimed
classes are class defects.
"""
import json, os, pickle, random, re, sys, unicodedata
from itertools import product

import tokenizers
from tokenizers import Tokenizer, Regex, pre_tokenizers, models

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_PATH = os.path.join(HERE, "probes.pkl")

# ---------------------------------------------------------------- patterns
GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
CL100K_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
# kernels.md's construction of the qwen 3.5 pattern (verified against hub file at runtime):
QWEN35_PATTERN_EXPECTED = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# layout.h TOKS_TP_* parameterisation of each variant
VARIANTS = {
    "gpt2":   dict(contr="CS", lprefix_any=False, digits="SP_RUN", punct_nl=False, ws_nl=False, marks=False),
    "cl100k": dict(contr="CI", lprefix_any=True,  digits="1_3",    punct_nl=True,  ws_nl=True,  marks=False),
    "qwen2":  dict(contr="CI", lprefix_any=True,  digits="1",      punct_nl=True,  ws_nl=True,  marks=False),
    "qwen35": dict(contr="CI", lprefix_any=True,  digits="1",      punct_nl=True,  ws_nl=True,  marks=True),
}
P,L,N,WS,NL = 0,1,2,3,4   # base classes

UNICODE_WS = {0x09,0x0A,0x0B,0x0C,0x0D,0x20,0x85,0xA0,0x1680,*range(0x2000,0x200B),0x2028,0x2029,0x202F,0x205F,0x3000}
ALPHABET = ["a","A","z","5","7","s","S","t","T","m","M","d","D","r","R","e","E","v","V","l","L",
            "'"," ","\t","\r","\n","\v","\f","!",".","-","_","\x00",
            "\xa0","\u2028","\u3000","\u0085","\u202f","é","中","ק","\u0301","\u017f","\u212a",
            "\u0660","\u2460","\uff10","\U0001F600","\u200d"]

# ---------------------------------------------------------------- probes
def _batch_members(pat, cps):
    """Probe membership through hf's own onig: chars NOT covered by any gap of
    Split(pattern, removed) are exactly the matched chars. The python binding
    returns CHARACTER offsets, so arithmetic is in joined-string chars: char i
    of the batch occupies [2i, 2i+1), separators sit between."""
    sp = pre_tokenizers.Split(Regex(pat), behavior="removed")
    res = set()
    B = 2048
    for st in range(0, len(cps), B):
        sub = cps[st:st+B]
        s = "\x00".join(map(chr, sub))
        gaps = sp.pre_tokenize_str(s) or []
        cov = [(a, b) for (_t, (a, b)) in gaps]
        gi = 0
        ncov = len(cov)
        for idx in range(len(sub)):
            s0, s1 = 2 * idx, 2 * idx + 1
            while gi < ncov and cov[gi][1] <= s0:
                gi += 1
            covered = gi < ncov and cov[gi][0] <= s0 and s1 <= cov[gi][1]
            if not covered:
                res.add(sub[idx])
    return res

def run_probes(force=False):
    if os.path.exists(PROBE_PATH) and not force:
        with open(PROBE_PATH, "rb") as f:
            return pickle.load(f)
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    pr = {}
    for name, pat in [("L", r"\p{L}"), ("N", r"\p{N}"), ("M", r"\p{M}"), ("s", r"\s"),
                      ("fold_s", r"(?i:s)"), ("fold_t", r"(?i:t)"), ("fold_m", r"(?i:m)"),
                      ("fold_d", r"(?i:d)"), ("fold_r", r"(?i:r)"), ("fold_e", r"(?i:e)"),
                      ("fold_v", r"(?i:v)"), ("fold_l", r"(?i:l)")]:
        pr[name] = _batch_members(pat, cps)
        print(f"probe {name}: {len(pr[name])} cps", file=sys.stderr, flush=True)
    with open(PROBE_PATH, "wb") as f:
        pickle.dump(pr, f)
    return pr

PROBE = None
_CLS = {}

def cls_array(variant, ws_mode="probed"):
    """base class per code point for a template variant; ws_mode='doc_ws' uses the
    25-cp unicode whitespace set the doc's notes assume."""
    key = (variant, ws_mode)
    if key in _CLS:
        return _CLS[key]
    P0 = VARIANTS[variant]
    Lset, Nset, Mset = PROBE["L"], PROBE["N"], PROBE["M"]
    Sset = PROBE["s"] if ws_mode == "probed" else UNICODE_WS
    marks = P0["marks"]
    arr = bytearray(0x110000)
    la, na, sa = Lset, Nset, Sset
    for cp in range(0x110000):
        if cp in la or (marks and cp in Mset):
            arr[cp] = L
        elif cp in na:
            arr[cp] = N
        elif cp == 0x0A or cp == 0x0D:
            arr[cp] = NL
        elif cp in sa:
            arr[cp] = WS
        else:
            arr[cp] = P
    _CLS[key] = arr
    return arr

def fold_sets():
    return {k: PROBE["fold_" + k] for k in "stmdrevl"}

# ---------------------------------------------------------------- §3 K3 model
def _atoms(s):
    cps = [ord(ch) for ch in s]
    boffs = []
    o = 0
    for cp in cps:
        boffs.append(o)
        o += 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
    return cps, boffs, o

def k3_scan(s, variant, cls=None, ws_mode="probed"):
    """kernels.md §3 A1-A7. Returns list of (start,end) byte offsets."""
    if cls is None:
        cls = cls_array(variant, ws_mode)
    P0 = VARIANTS[variant]
    F = fold_sets()
    CI1 = F["s"] | F["t"] | F["m"] | F["d"]
    cps, boffs, tot = _atoms(s)
    n = len(cps)
    pieces = []
    i = 0
    while i < n:
        c = cps[i]
        b = cls[c]
        e = None
        if P0["contr"] and c == 0x27 and i + 1 < n:
            c1 = cps[i + 1]
            if P0["contr"] == "CS":
                if c1 in (0x73, 0x74, 0x6D, 0x64):
                    e = i + 2
                elif i + 2 < n and (c1, cps[i + 2]) in ((0x72, 0x65), (0x76, 0x65), (0x6C, 0x6C)):
                    e = i + 3
            else:
                if c1 in CI1:
                    e = i + 2
                elif i + 2 < n:
                    c2 = cps[i + 2]
                    if (c1 in F["r"] and c2 in F["e"]) or (c1 in F["v"] and c2 in F["e"]) \
                       or (c1 in F["l"] and c2 in F["l"]):
                        e = i + 3
        if e is None:
            if b == L:
                e = i
                while e < n and cls[cps[e]] == L:
                    e += 1
            elif i + 1 < n and cls[cps[i + 1]] == L and \
                 ((P0["lprefix_any"] and b in (P, WS)) or ((not P0["lprefix_any"]) and c == 0x20)):
                e = i + 1
                while e < n and cls[cps[e]] == L:
                    e += 1
            elif b == N:
                if P0["digits"] == "1_3":
                    e = min(i + 3, n)
                    while e > i and cls[cps[e - 1]] != N:
                        e -= 1
                    j = i
                    while j < n and cls[cps[j]] == N:
                        j += 1
                    e = min(i + 3, j)
                elif P0["digits"] == "1":
                    e = i + 1
                else:
                    e = i
                    while e < n and cls[cps[e]] == N:
                        e += 1
            elif P0["digits"] == "SP_RUN" and c == 0x20 and i + 1 < n and cls[cps[i + 1]] == N:
                e = i + 1
                while e < n and cls[cps[e]] == N:
                    e += 1
            elif b == P or (c == 0x20 and i + 1 < n and cls[cps[i + 1]] == P):
                j = i if b == P else i + 1
                e = j
                while e < n and cls[cps[e]] == P:
                    e += 1
                if P0["punct_nl"]:
                    while e < n and cls[cps[e]] == NL:
                        e += 1
            elif b in (WS, NL):
                k = i
                while k < n and cls[cps[k]] in (WS, NL):
                    k += 1
                if P0["ws_nl"]:
                    t = k - 1
                    while t >= i and cls[cps[t]] != NL:
                        t -= 1
                    if t >= i:
                        e = t + 1
                if e is None:
                    if k == n:
                        e = k
                    elif k - i >= 2:
                        e = k - 1
                    else:
                        e = k
            else:
                raise AssertionError("atom with no alternative")
        pieces.append((boffs[i], boffs[e] if e < n else tot))
        i = e
    return pieces

def _c2b(s):
    """char index -> byte offset table; the binding's offsets are CHARACTER offsets."""
    t = [0] * (len(s) + 1)
    o = 0
    for i, ch in enumerate(s):
        t[i] = o
        o += 4 if ord(ch) >= 0x10000 else (3 if ord(ch) >= 0x800 else (2 if ord(ch) >= 0x80 else 1))
    t[len(s)] = o
    return t

_HF_SPLIT = {}
def hf_pieces(variant, s, pattern_override=None):
    """hf pre-tokenizer pieces as (start,end) BYTE offsets of s."""
    pat = pattern_override or {"gpt2": GPT2_PATTERN, "cl100k": CL100K_PATTERN,
                                "qwen2": QWEN2_PATTERN, "qwen35": QWEN35_PATTERN_EXPECTED}[variant]
    t = _c2b(s)
    if variant == "gpt2" and pattern_override is None:
        bl = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)
        return [(t[a], t[b]) for (_t2, (a, b)) in bl.pre_tokenize_str(s)]
    sp = _HF_SPLIT.get(pat)
    if sp is None:
        sp = pre_tokenizers.Split(Regex(pat), behavior="isolated", invert=False)
        _HF_SPLIT[pat] = sp
    out = []
    for tok, (a, b) in sp.pre_tokenize_str(s):
        assert s[a:b] == tok, (repr(s), a, b, repr(tok))  # convention guard: char offsets
        out.append((t[a], t[b]))
    return out

# ---------------------------------------------------------------- §5 bpe model
def bytes_char():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTES_CHAR = bytes_char()

class Bpe:
    def __init__(self, model_dict):
        self.vocab = model_dict["vocab"]
        self.merges_list = model_dict["merges"]
        self.ignore_merges = bool(model_dict.get("ignore_merges", False))
        self.mm = {}
        for rank, (a, b) in enumerate(self.merges_list):
            ia, ib = self.vocab[a], self.vocab[b]
            self.mm[(ia, ib)] = (rank, self.vocab[a + b])   # duplicate pairs: last wins
        self.byte2id = {}
        for b in range(256):
            t = self.vocab.get(BYTES_CHAR[b])
            if t is None:
                raise ValueError("missing byte token")
            self.byte2id[b] = t

    def tokenize_bytes(self, piece: bytes):
        if self.ignore_merges:
            ms = "".join(BYTES_CHAR[b] for b in piece)
            if ms in self.vocab:
                return [self.vocab[ms]]
        ids = [self.byte2id[b] for b in piece]
        mm = self.mm
        while len(ids) > 1:
            best_rank = None
            best_pos = -1
            for p in range(len(ids) - 1):
                r = mm.get((ids[p], ids[p + 1]))
                if r is not None and (best_rank is None or r[0] < best_rank):
                    best_rank, best_pos = r[0], p
            if best_rank is None:
                break
            new_id = mm[(ids[best_pos], ids[best_pos + 1])][1]
            ids[best_pos:best_pos + 2] = [new_id]
        return ids

def bpe_only_tokenizer(tokenizer_json_path):
    with open(tokenizer_json_path) as f:
        tj = json.load(f)
    m = dict(tj["model"])
    m.setdefault("type", "BPE")
    if m.get("merges") and isinstance(m["merges"][0], str):
        m["merges"] = [s.split(" ") for s in m["merges"]]
    tj2 = {"version": tj.get("version", "1.0"), "truncation": None, "padding": None,
           "added_tokens": [], "normalizer": None, "pre_tokenizer": None,
           "post_processor": None, "decoder": tj.get("decoder"), "model": m}
    return Tokenizer.from_str(json.dumps(tj2)), Bpe(m)

# ------------------------------------------------- rust-regex \w / \s (for §4)
def load_word_set():
    """rust regex crate \w = [\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]"""
    cache = os.path.join(HERE, "wordset.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    import urllib.request
    txt = urllib.request.urlopen(
        "https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt", timeout=60).read().decode()
    alpha, joinc = set(), set()
    cur = None
    for line in txt.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("#") or ";" not in line:
            # property blocks look like "# Alphabetic (A)" comments before ranges; parse differently
            pass
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        rng, prop = parts[0], parts[1]
        if ".." in rng:
            a, b = rng.split("..")
            targets = range(int(a, 16), int(b, 16) + 1)
        else:
            targets = [int(rng, 16)]
        if prop == "Alphabetic":
            alpha.update(targets)
        elif prop == "Join_Control":
            joinc.update(targets)
    word = set(c for c in alpha)
    for cp in range(0x110000):
        if cp in word or cp in joinc:
            continue
        cat = unicodedata.category(chr(cp)) if not (0xD800 <= cp <= 0xDFFF) else ""
        if cat in ("Mn", "Mc", "Me", "Nd", "Pc"):
            word.add(cp)
    word |= joinc
    with open(cache, "wb") as f:
        pickle.dump(word, f)
    return word

RUST_WS = UNICODE_WS  # rust regex \s = \p{White_Space} (the 25 cps)

# ---------------------------------------------------------------- §4 model
class AddTok:
    __slots__ = ("content", "norm", "id", "special", "single_word", "lstrip", "rstrip", "normalized")
    def __init__(self, d, normalizer=None):
        self.content = d["content"]
        self.id = d["id"]
        self.special = bool(d.get("special", False))
        self.single_word = bool(d.get("single_word", False))
        self.lstrip = bool(d.get("lstrip", False))
        self.rstrip = bool(d.get("rstrip", False))
        self.normalized = bool(d.get("normalized", not self.special))
        # hf matches the normalized form (normalized_cache) but checks special-ness
        # and strips against the ORIGINAL content
        self.norm = self.content
        if self.normalized and normalizer:
            c = normalizer(self.content)
            if c != self.content:
                self.norm = c
    def bytes(self):
        return self.norm.encode("utf-8")

def _iter_matches(text, by_first, resume):
    """leftmost-longest at each start position. resume: 'start+1' (hf/daachorse
    observation) or 'end' (kernels.md §4 as written)."""
    n = len(text)
    s = 0
    while s < n:
        lst = by_first.get(text[s])
        m = None
        if lst:
            for tb, info in lst:
                if text.startswith(tb, s):
                    m = (s, s + len(tb), info)
                    break
        if m is not None:
            yield m
            s = s + 1 if resume == "start+1" else m[1]
        else:
            s += 1

def added_split(text, tokens, drop_specials, word, resume="end", specials=None):
    """Mirrors hf added_vocabulary.rs find_matches. resume: 'end' (daachorse
    find_iter semantics, which kernels.md §4 specifies) or 'start+1' (control)."""
    if specials is None:
        specials = {t.content for t in tokens if t.special}
    by_first = {}
    for t in tokens:
        by_first.setdefault(t.bytes()[0], []).append((t.bytes(), t))
    for k in by_first:
        by_first[k].sort(key=lambda x: -len(x[0]))
    splits = []
    start_offset = 0
    n = len(text)
    for (s, e, tok) in _iter_matches(text, by_first, resume):
        start, stop = s, e
        if drop_specials and tok.content in specials:
            continue
        if tok.single_word:
            pre = text[:start]
            suf = text[stop:]
            start_ok = start == 0 or not _ends_with_word(pre, word)
            stop_ok = stop == n or not _starts_with_word(suf, word)
            if not (start_ok and stop_ok):
                continue
        if tok.lstrip:
            newstart = _space_leftmost_at_end(text[:start])
            start = max(newstart, start_offset)
        if tok.rstrip:
            stop = stop + _space_rightmost_at_start(text[stop:])
        if start_offset < start:
            splits.append(("gap", start_offset, start))
        splits.append(("tok", start, stop, tok.id))
        start_offset = stop
    if start_offset != n:
        splits.append(("gap", start_offset, n))
    return splits

def _ends_with_word(b, word):
    if not b:
        return False
    # last char (utf-8 decode of final sequence)
    i = len(b) - 1
    if b[i] < 0x80:
        cp = b[i]
    else:
        start = i
        while start > 0 and 0x80 <= b[start] < 0xC0:
            start -= 1
        cp = _decode_one(b[start:])
    return cp in word

def _starts_with_word(b, word):
    if not b:
        return False
    return _decode_one(b) in word

def _decode_one(b):
    return ord(b.decode("utf-8", "ignore")[:1] or "\x00") if b else 0

def _space_leftmost_at_end(b):
    # leftmost match of \s*$ == start of the trailing unicode-whitespace run
    j = len(b)
    while j > 0:
        k = _char_start(b, j)
        if _decode_one(b[k:j]) in RUST_WS:
            j = k
        else:
            break
    return j

def _char_start(b, j):
    k = j - 1
    while k > 0 and 0x80 <= b[k] < 0xC0:
        k -= 1
    return k

def _space_rightmost_at_start(b):
    i = 0
    while i < len(b):
        j = _next_char(b, i)
        if _decode_one(b[i:j]) in RUST_WS:
            i = j
        else:
            break
    return i

def _char_len(b, i):
    c = b[i]
    if c < 0x80:
        return 1
    if c < 0xE0:
        return 2
    if c < 0xF0:
        return 3
    return 4

def _next_char(b, i):
    return i + _char_len(b, i)

# ------------------------------------------------------------- normalization
def make_normalizer(norm_dict):
    if norm_dict is None:
        return None
    t = norm_dict.get("type")
    if t is None:
        return None
    if t == "NFC":
        return lambda s: unicodedata.normalize("NFC", s)
    if t == "NFD":
        return lambda s: unicodedata.normalize("NFD", s)
    if t == "Lowercase":
        return lambda s: s.lower()
    raise ValueError(f"normalizer {t} not modelled")

# ------------------------------------------------------- full pipeline model
class Pipe:
    """Model of the whole encode pipeline per kernels.md + hf mirror."""
    def __init__(self, tj, word, resume="end"):
        self.tj = tj
        self.word = word
        self.resume = resume
        self.normalizer = make_normalizer(tj.get("normalizer"))
        self.bpe = Bpe(tj["model"])
        added = tj.get("added_tokens") or []
        self.p0 = [AddTok(d, self.normalizer) for d in added if not d.get("normalized", not d.get("special", False))]
        self.p1 = [AddTok(d, self.normalizer) for d in added if d.get("normalized", not d.get("special", False))]
        pt = tj.get("pre_tokenizer")
        self.variant = None
        self.bl_prefix = False
        self._parse_pretok(pt)

    def _parse_pretok(self, pt):
        if pt is None:
            return
        seq = pt.get("pretokenizers") if pt.get("type") == "Sequence" else [pt]
        for p in seq:
            t = p.get("type")
            if t == "ByteLevel":
                if p.get("use_regex", True):
                    self.variant = "gpt2"
                self.bl_prefix = bool(p.get("add_prefix_space", True))
            elif t == "Split":
                pat = p["pattern"]["Regex"] if isinstance(p["pattern"], dict) else p["pattern"]
                if pat == GPT2_PATTERN:
                    self.variant = "gpt2"
                elif pat == CL100K_PATTERN:
                    self.variant = "cl100k"
                elif pat == QWEN2_PATTERN:
                    self.variant = "qwen2"
                elif pat == QWEN35_PATTERN_EXPECTED:
                    self.variant = "qwen35"
                else:
                    m = re.match(r"\(\\p\{N\}\|\?", "")
                    raise ValueError(f"unmodelled pattern {pat!r}")
            else:
                raise ValueError(f"unmodelled pretok {t}")

    def encode(self, s, drop_specials=False):
        self._raw = text = s.encode("utf-8")
        ids = []
        for seg in added_split(text, self.p0, drop_specials, self.word, resume=self.resume):
            if seg[0] == "tok":
                ids.append(seg[3])
            else:
                ids.extend(self._encode_gap(seg[1], seg[2], drop_specials))
        return ids

    def _encode_gap(self, a, b, drop_specials):
        s = self._raw[a:b].decode("utf-8")
        if self.normalizer:
            s = self.normalizer(s)
        ids = []
        for seg1 in added_split(s.encode("utf-8"), self.p1, drop_specials, self.word, resume=self.resume):
            if seg1[0] == "tok":
                ids.append(seg1[3])
            else:
                ids.extend(self._encode_text_seg(s, seg1[1], seg1[2]))
        return ids

    def _encode_text_seg(self, s, a, b):
        piece_s = s[a:b] if isinstance(a, int) else s
        # a,b are byte offsets into the (normalized) segment
        raw = s.encode("utf-8")[a:b]
        if self.bl_prefix and not raw.startswith(b" "):
            raw = b" " + raw
        if self.variant is None:
            return self.bpe.tokenize_bytes(raw)
        cps, boffs, tot = _atoms(raw.decode("utf-8"))
        cls = cls_array(self.variant)
        pieces = k3_scan(raw.decode("utf-8"), self.variant, cls)
        ids = []
        st = 0
        for (x, y) in pieces:
            ids.extend(self.bpe.tokenize_bytes(raw[x:y]))
        return ids

def load_tj(path):
    with open(path) as f:
        return json.load(f)

def hf_tokenizer(path):
    return Tokenizer.from_file(path)

def set_encode_special(tk, val):
    if hasattr(tk, "encode_special_tokens"):
        try:
            tk.encode_special_tokens = val
            return "property"
        except Exception:
            pass
    if hasattr(tk, "_tokenizer"):
        inner = tk._tokenizer
        for name in ("set_encode_special_tokens", "encode_special_tokens"):
            fn = getattr(inner, name, None)
            if callable(fn):
                try:
                    r = fn(val)
                    if r is None or r is False or r is True:
                        return f"inner.{name}"
                except TypeError:
                    try:
                        fn(val)
                        return f"inner.{name}"
                    except Exception:
                        pass
    raise RuntimeError("no encode_special_tokens API found: " + repr([a for a in dir(tk) if 'special' in a.lower()]))

# ---------------------------------------------------------------- generators
WS_CPS = [0x09,0x0A,0x0B,0x0C,0x0D,0x20,0x85,0xA0,0x1680,*range(0x2000,0x200B),0x2028,0x2029,0x202F,0x205F,0x3000]
MARKS = [0x300,0x301,0x308,0x361,0x897,0x1AC1,0x3099,0x20D0,0xFE0F]
NONASCII_DIGITS = [*range(0x660,0x66A),0x2460,0x2461,0xFF10,0xFF11,0xB2,0xB3,0xB9,0x2160]
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" + "".join(
    map(chr,[0xE9,0x4E2D,0x6587,0x0410,0xAC00,0x3042,0x1E9E,0x05D0,0x0628,0x017F,0x212A]))
PUNCTS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "".join(map(chr,[0xA1,0x2190,0x2500,0x3001,0xFF01]))
EMOJI = [chr(c) for c in [0x1F600,0x1F44D,0x2764,0x200D,0xFE0F,0x1F1E6]]
APOS_FOLLOW = "stmdrevlSTMDREVL" + chr(0x17F)

def _rand_ws(rng):
    return chr(rng.choice(WS_CPS))

def gen_case(rng, profile=None):
    pr = profile or rng.choice(["mixed","mixed","ws","crlf","apos","digits","marks","punct","emoji","tiny"])
    if pr == "tiny":
        return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 3)))
    if pr == "ws":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append("".join(_rand_ws(rng) for _ in range(rng.randint(1, 6))))
            out.append(rng.choice(["x", "1", "!", "中", ""]))
        return "".join(out)
    if pr == "crlf":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["\r","\n","\r\n","\n\r","\t"," "]) * rng.randint(1,4))
            if rng.random() < 0.7:
                out.append(rng.choice(["x","!","1",""]))
        return "".join(out)
    if pr == "apos":
        n = rng.randint(1, 6)
        out = []
        for _ in range(n):
            out.append(rng.choice(["", " ", "\t", "a", "1", "!"]))
            out.append("'")
            k = rng.randint(1, 3)
            for _ in range(k):
                r = rng.random()
                if r < 0.75:
                    out.append(rng.choice(APOS_FOLLOW))
                else:
                    out.append(rng.choice(ALPHABET))
        return "".join(out)
    if pr == "digits":
        pool = "0123456789" + "".join(map(chr, NONASCII_DIGITS))
        out = []
        for _ in range(rng.randint(1, 5)):
            out.append(rng.choice(["", " ", "\t", "a", "!"]))
            out.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 9))))
        return "".join(out)
    if pr == "marks":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append(rng.choice(["a", "中", "e", "1", "!", " "]))
            out.append("".join(chr(rng.choice(MARKS)) for _ in range(rng.randint(1, 3))))
        return "".join(out)
    if pr == "punct":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "\t", "a", "1"]))
            out.append("".join(rng.choice(PUNCTS + "'") for _ in range(rng.randint(1, 5))))
            if rng.random() < 0.4:
                out.append(rng.choice(["\r\n", "\n", "\r", "\r\n\r\n"]))
        return "".join(out)
    if pr == "emoji":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append("".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 4))))
            out.append(rng.choice(["", " ", "a", "!"]))
        return "".join(out)
    # mixed
    n = rng.randint(0, 200)
    out = []
    while len(out) < n:
        r = rng.random()
        if r < 0.22:
            out.append(rng.choice(LETTERS) * rng.randint(1, 4))
        elif r < 0.34:
            out.append(rng.choice("0123456789" + "".join(map(chr, NONASCII_DIGITS))) * rng.randint(1, 4))
        elif r < 0.48:
            out.append(rng.choice(PUNCTS) * rng.randint(1, 3))
        elif r < 0.60:
            out.append(_rand_ws(rng) * rng.randint(1, 3))
        elif r < 0.68:
            out.append(chr(rng.choice(MARKS)))
        elif r < 0.78:
            out.append("'")
            out.append(rng.choice(APOS_FOLLOW))
        elif r < 0.88:
            out.append(rng.choice(EMOJI))
        else:
            out.append(rng.choice(ALPHABET))
    return "".join(out)[:n]

def gen_long(rng):
    kind = rng.choice(["letters", "digits", "punct", "ws", "mixed", "base64"])
    n = rng.randint(60, 1000)
    if kind == "letters":
        return "".join(rng.choice(LETTERS) for _ in range(n))
    if kind == "digits":
        return "".join(rng.choice("0123456789" + "".join(map(chr, NONASCII_DIGITS))) for _ in range(n))
    if kind == "punct":
        return "".join(rng.choice(PUNCTS) for _ in range(n))
    if kind == "ws":
        return "".join(_rand_ws(rng) for _ in range(n))
    if kind == "base64":
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") for _ in range(n))
    return gen_case(rng, "mixed")[:n]
