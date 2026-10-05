#!/usr/bin/env python3
"""dsv3 template model: docs/templates/dsv3.md §2-§3 transcribed literally, vs hf tokenizers 0.23.2.

  - classes come from hf's own regex engine (oniguruma), never python's unicodedata: every scalar value is run
    through pre_tokenizers.Split(Regex(set), 'removed') for every set the chain uses (\\p{L} \\p{M} \\p{N} \\p{P}
    \\p{S} \\s, the brackets [\\p{L}\\p{M}] [\\p{P}\\p{S}] [^\\r\\n\\p{L}\\p{P}\\p{S}], alternative 1's punctuation
    bracket, [A-Za-z], [\\r\\n], split 2's CJK bracket) plus the complement set X; the doc's §2 claims are
    asserted on every scalar (build_classes).
  - k3_scan_atoms() is §3: regions (N chunks, CJK edges), D0-D7 in the order the doc gives, over kernels.md §2
    atoms (an invalid byte is a P atom without the CJK bit), so it also runs on ill-formed bytes (k3_scan_bytes).
    The doc's side claims (D7's give-back needs >= 2 atoms; a CJK region holds only L, P and X atoms; every atom
    is taken by some rule) are asserted on every piece.
  - hf_pieces() is the oracle: the pinned tokenizer.json's own pre_tokenizer (Tokenizer.from_file(...).
    pre_tokenizer.pre_tokenize_str: the three Splits and ByteLevel); its offsets are character offsets, converted
    to bytes; every piece string is checked against the byte-level image of its slice and the pieces must tile
    the text. seq3_pieces() builds the three Splits from this file's constants (no ByteLevel): the runner
    compares both on a sample (the constants are the file's, ByteLevel(use_regex=false) splits nothing).
  - generators: class-aware random strings (gen_case / gen_long), exhaustive strings over a representative
    alphabet (REP, REP_SMALL), vocabulary-token concatenations, real-text segments (segments_of).

Used by tests/model/run_dsv3_fuzz.py (the proof runs) and tests/k3/dsv3_gen.py (cases for the c twin).
Probe cache: build/dsv3_probes.pkl (TOKS_DSV3_PROBES overrides); the tokenizer: ~/.cache/toks/tokenizers/dsv3
(TOKS_DSV3_TOKENIZER overrides; its sha256 is checked).
"""
import hashlib
import json
import os
import pickle
import random
import sys

from tokenizers import Regex, Tokenizer, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBE_PATH = os.environ.get("TOKS_DSV3_PROBES", os.path.join(REPO, "build", "dsv3_probes.pkl"))
TOKENIZER = os.environ.get("TOKS_DSV3_TOKENIZER", os.path.expanduser("~/.cache/toks/tokenizers/dsv3"))
TOKENIZER_SHA = "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41"   # deepseek-ai/DeepSeek-V3

# ---------------------------------------------------------------- the chain (docs/templates/dsv3.md §1)
SPLIT1 = "\\p{N}{1,3}"
SPLIT2 = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"          # the literal characters, as the file has them
SPLIT3 = ("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
          "|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
          "| ?[\\p{P}\\p{S}]+[\r\n]*"
          "|\\s*[\r\n]+"
          "|\\s+(?!\\S)"
          "|\\s+")
APUNCT = SPLIT3[:SPLIT3.index("[A-Za-z]")]                         # alternative 1's punctuation bracket
CJK_BRACKET = SPLIT2[:-1]
CJK_RANGES = [(0x3040, 0x30FF), (0x4E00, 0x9FA5)]                  # §2: 309F + 1 = 30A0, one range


def check_tokenizer_file():
    """-> report lines; raises when the pinned file disagrees with its sha256 or with the constants above."""
    with open(TOKENIZER, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != TOKENIZER_SHA:
        raise SystemExit("dsv3_model: %s sha256 %s != pin %s" % (TOKENIZER, got, TOKENIZER_SHA))
    tj = json.loads(raw)
    pt = tj["pre_tokenizer"]
    steps = pt["pretokenizers"] if pt.get("type") == "Sequence" else None
    want = [SPLIT1, SPLIT2, SPLIT3]
    if steps is None or len(steps) != 4:
        raise SystemExit("dsv3_model: the pre_tokenizer is not a 4-step Sequence")
    for k in range(3):
        s = steps[k]
        if (s.get("type") != "Split" or s.get("pattern") != {"Regex": want[k]} or s.get("behavior") != "Isolated"
                or s.get("invert") is not False):
            raise SystemExit("dsv3_model: Split %d differs from the doc's: %r" % (k + 1, s))
    bl = steps[3]
    if bl.get("type") != "ByteLevel" or bl.get("add_prefix_space") is not False or bl.get("use_regex") is not False:
        raise SystemExit("dsv3_model: the last step is not ByteLevel(add_prefix_space=false, use_regex=false)")
    nm = tj.get("normalizer")
    if nm != {"type": "Sequence", "normalizers": []}:
        raise SystemExit("dsv3_model: the normalizer is not the empty Sequence: %r" % nm)
    return ["tokenizer %s: sha256 ok; Sequence[Split x3 (the doc's strings, Isolated, invert false), "
            "ByteLevel(add_prefix_space false, use_regex false)]; normalizer Sequence[]" % TOKENIZER]


# ---------------------------------------------------------------- probes through hf's onig
SETS = [
    ("L", "\\p{L}"), ("M", "\\p{M}"), ("N", "\\p{N}"), ("P", "\\p{P}"), ("S", "\\p{S}"), ("WS", "\\s"),
    ("LM", "[\\p{L}\\p{M}]"), ("PS", "[\\p{P}\\p{S}]"), ("PFX", "[^\r\n\\p{L}\\p{P}\\p{S}]"),
    ("APUNCT", APUNCT), ("ALPHA", "[A-Za-z]"), ("NLS", "[\r\n]"), ("CJK", CJK_BRACKET),
    ("X", "[^\\p{L}\\p{M}\\p{N}\\p{P}\\p{S}\\s]"),
]
BATCH = 0x8000


def scalars():
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


def _kept(sp, s):
    keep = bytearray(len(s))
    for _, (a, b) in sp.pre_tokenize_str(s):
        keep[a:b] = b"\1" * (b - a)
    return keep


def probe_set(pat, cps):
    """set of code points the one-char regex matches (a match is removed: not kept == matched)."""
    sp = pre_tokenizers.Split(Regex(pat), "removed")
    hit = set()
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        keep = _kept(sp, "".join(map(chr, part)))
        for k, cp in enumerate(part):
            if not keep[k]:
                hit.add(cp)
    return hit


def run_probes(force=False, log=sys.stderr):
    if os.path.exists(PROBE_PATH) and not force:
        with open(PROBE_PATH, "rb") as f:
            pr = pickle.load(f)
        if set(pr) >= {name for name, _ in SETS}:
            return pr
    cps = scalars()
    pr = {}
    for name, pat in SETS:
        pr[name] = probe_set(pat, cps)
        print("probe %-6s %7d code points  %s" % (name, len(pr[name]), pat.encode("unicode_escape").decode()),
              file=log, flush=True)
    os.makedirs(os.path.dirname(PROBE_PATH), exist_ok=True)
    with open(PROBE_PATH, "wb") as f:
        pickle.dump(pr, f)
    return pr


# class byte (docs/templates/dsv3.md §2; proposed layout.h values): base | CJK
P, L, N, WS, NL, X = 0, 1, 2, 3, 4, 5
CJK = 0x80
BASE_NAMES = {P: "P", L: "L", N: "N", WS: "WS", NL: "NL", X: "X"}

PROBE = None
CLS = None          # bytearray(0x110000): the class byte of every code point, from the probes
FACTS = None        # report lines


def build_classes(pr):
    """class bytes from the probes, then the doc's §2 claims asserted over every scalar."""
    sc = scalars()
    cls = bytearray(0x110000)
    Ls, Ms, Ns, Ps, Ss, WSs = pr["L"], pr["M"], pr["N"], pr["P"], pr["S"], pr["WS"]
    cjk = pr["CJK"]
    for cp in sc:
        if cp in Ls or cp in Ms:
            b = L
        elif cp in Ns:
            b = N
        elif cp in (0x0A, 0x0D):
            b = NL
        elif cp in WSs:
            b = WS
        elif cp in Ps or cp in Ss:
            b = P
        else:
            b = X
        if cp in cjk:
            b |= CJK
        cls[cp] = b
    bad = []

    def claim(ok, what):
        if not ok:
            bad.append(what)

    props = [("L", Ls), ("M", Ms), ("N", Ns), ("P", Ps), ("S", Ss), ("\\s", WSs)]
    for a in range(len(props)):
        for b2 in range(a + 1, len(props)):
            inter = props[a][1] & props[b2][1]
            claim(not inter, "%s and %s share %s" % (props[a][0], props[b2][0], sorted(inter)[:5]))
    allsc = set(sc)
    union = Ls | Ms | Ns | Ps | Ss | WSs
    claim(pr["LM"] == Ls | Ms, "[\\p{L}\\p{M}] == L | M")
    claim(pr["PS"] == Ps | Ss, "[\\p{P}\\p{S}] == P | S")
    claim(pr["X"] == allsc - union, "X == the complement of L|M|N|P|S|\\s")
    claim(pr["NLS"] == {0x0A, 0x0D} and {0x0A, 0x0D} <= WSs, "[\\r\\n] == {CR, LF}, both \\s")
    claim(pr["PFX"] == allsc - (Ls | Ps | Ss | {0x0A, 0x0D}), "PFX == not L, P, S, CR, LF")
    pfx_kinds = {cp for cp in sc if (cls[cp] & 7) in (N, WS, X)} | Ms
    claim(pr["PFX"] == pfx_kinds, "PFX == base in {N, WS, X} or a mark")
    apunct = {cp for cp in range(0x80) if (cls[cp] & 7) == P}
    claim(pr["APUNCT"] == apunct and len(apunct) == 32, "alt-1 bracket == the 32 ascii P atoms")
    alpha = {cp for cp in range(0x80) if (cls[cp] & 7) == L}
    claim(pr["ALPHA"] == alpha and alpha == set(range(0x41, 0x5B)) | set(range(0x61, 0x7B)),
          "[A-Za-z] == the ascii L atoms")
    want_cjk = {cp for a, b2 in CJK_RANGES for cp in range(a, b2 + 1)}
    claim(cjk == want_cjk, "the CJK bracket == U+3040..30FF, U+4E00..9FA5")
    claim(not (cjk & (Ns | WSs)), "no CJK code point is N or \\s")
    claim(all(cls[cp] & 7 in (L, P, X) for cp in cjk), "CJK atoms are L, P or X")
    if bad:
        raise SystemExit("dsv3_model: class claims fail: %s" % "; ".join(bad[:20]))
    facts = []
    counts = {}
    for cp in sc:
        counts[cls[cp]] = counts.get(cls[cp], 0) + 1
    facts.append("kinds over the scalars: " + ", ".join(
        "%s%s %d" % (BASE_NAMES[k & 7], "|CJK" if k & CJK else "", v) for k, v in sorted(counts.items())))
    odd = sorted(cp for cp in cjk if cls[cp] & 7 != L)
    facts.append("CJK atoms not L: " + ", ".join("U+%04X %s" % (cp, BASE_NAMES[cls[cp] & 7]) for cp in odd))
    facts.append("sets: " + ", ".join("%s %d" % (name, len(pr[name])) for name, _ in SETS))
    # stage2 size of a two-level table of these class bytes (blocks of 256, deduplicated)
    blocks = {bytes(cls[hi * 256:(hi + 1) * 256]) for hi in range(0x1100)}
    facts.append("class table: %d distinct 256-code-point blocks (%d B stage2 + 8704 B stage1 + 128 B ascii)"
                 % (len(blocks), 256 * len(blocks)))
    return cls, facts


def init(force=False):
    global PROBE, CLS, FACTS
    if CLS is None or force:
        PROBE = run_probes(force)
        CLS, FACTS = build_classes(PROBE)
    return CLS


# ---------------------------------------------------------------- atoms (kernels.md §2)
def atoms_of_str(s):
    """[(cp, byte length)] of a str (every atom a scalar)."""
    return [(ord(ch), 1 if ord(ch) < 0x80 else 2 if ord(ch) < 0x800 else 3 if ord(ch) < 0x10000 else 4) for ch in s]


def atoms_of_bytes(b):
    """[(cp, k)] per kernels.md §2: a well-formed utf-8 sequence (unicode 3.9 table 3-7) is one atom, every byte of
    an ill-formed one an atom of its own (cp = -1)."""
    out = []
    i, n = 0, len(b)
    while i < n:
        x = b[i]
        k = 0
        if x < 0x80:
            k = 1
        elif 0xC2 <= x <= 0xDF:
            k = 2 if i + 1 < n and 0x80 <= b[i + 1] <= 0xBF else 0
        elif 0xE0 <= x <= 0xEF:
            lo = 0xA0 if x == 0xE0 else 0x80
            hi = 0x9F if x == 0xED else 0xBF
            k = 3 if i + 2 < n and lo <= b[i + 1] <= hi and 0x80 <= b[i + 2] <= 0xBF else 0
        elif 0xF0 <= x <= 0xF4:
            lo = 0x90 if x == 0xF0 else 0x80
            hi = 0x8F if x == 0xF4 else 0xBF
            k = 4 if (i + 3 < n and lo <= b[i + 1] <= hi and 0x80 <= b[i + 2] <= 0xBF
                      and 0x80 <= b[i + 3] <= 0xBF) else 0
        if k == 0:
            out.append((-1, 1))
            i += 1
        else:
            out.append((ord(b[i:i + k].decode("utf-8")), k))
            i += k
    return out


# ---------------------------------------------------------------- §3: the template rules
def k3_scan_atoms(atoms):
    """docs/templates/dsv3.md §3 over atoms [(cp, k)] (cp -1: an invalid byte, a P atom without the CJK bit).
    -> [(start, end)] in bytes."""
    assert CLS is not None, "init() first"
    n = len(atoms)
    cps = [a[0] for a in atoms]
    kd = [CLS[cp] if cp >= 0 else P for cp in cps]
    off = [0] * (n + 1)
    for j in range(n):
        off[j + 1] = off[j] + atoms[j][1]

    def base(j):
        return kd[j] & 7 if j < n else None

    def ascii_atom(j, b):                  # an ascii atom of base b
        return j < n and 0 <= cps[j] < 0x80 and kd[j] & 7 == b

    pieces = []
    i = 0
    while i < n:
        c = kd[i]
        b = c & 7
        r = c & CJK                        # the region's CJK value
        if r:
            assert b in (L, P, X), "doc: a CJK region holds only L, P and X atoms"

        def inreg(j, want):                # an in-region atom of base `want`: kind want | r
            return j < n and kd[j] == (want | r)

        def run(j, want):                  # end of the maximal run of in-region `want` atoms from j
            while inreg(j, want):
                j += 1
            return j

        e = None
        if b == N:                                                   # D0 digits
            e = i + 1
            while e < n and e < i + 3 and base(e) == N:
                e += 1
        elif ascii_atom(i, P) and ascii_atom(i + 1, L):              # D1 ascii word
            e = i + 1
            while ascii_atom(e, L):
                e += 1
        elif b == L:                                                 # D2 letters
            e = run(i, L)
        elif b in (WS, X) and inreg(i + 1, L):                       # D2 with a WS / X prefix
            e = run(i + 1, L)
        if e is None:                                                # D3 punctuation
            j = i if b == P else (i + 1 if cps[i] == 0x20 and inreg(i + 1, P) else None)
            if j is not None:
                e = run(run(j, P), NL)                               # the P run, then the in-region NL tail
        if e is None and b in (WS, NL):                              # D4-D6 whitespace
            k = i
            while k < n and base(k) in (WS, NL):
                k += 1
            t = k - 1
            while t >= i and base(t) != NL:
                t -= 1
            region_end = k == n or base(k) == N or (kd[k] & CJK) != r
            if t >= i:
                e = t + 1                  # D4: through the last NL of the run
            elif region_end:
                e = k                      # D5: the run ends its region
            elif k - i >= 2:
                e = k - 1                  # D5: give back the last atom
            else:
                e = k                      # D6
        if e is None:                                                # D7 gap
            assert b == X, "doc: every atom is taken by some rule"
            g = run(i, X)
            if inreg(g, L):
                assert g - i >= 2, "doc: D7 gives back only from a run of >= 2 X atoms"
                e = g - 1
            else:
                e = g
        pieces.append((off[i], off[e]))
        i = e
    return pieces


def k3_scan(s):
    return k3_scan_atoms(atoms_of_str(s))


def k3_scan_bytes(b):
    return k3_scan_atoms(atoms_of_bytes(b))


# ---------------------------------------------------------------- the oracle
def _bytes_to_unicode():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    k = 0
    for x in range(256):
        if x not in bs:
            bs.append(x)
            cs.append(256 + k)
            k += 1
    return {x: chr(c) for x, c in zip(bs, cs)}


B2U = _bytes_to_unicode()


def c2b(s):
    t = [0] * (len(s) + 1)
    o = 0
    for i, ch in enumerate(s):
        t[i] = o
        cp = ord(ch)
        o += 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
    t[len(s)] = o
    return t


_PT = None
_SEQ3 = None


def hf_pieces(s):
    """the pinned file's own pre-tokenizer: byte offsets of its pieces (asserted to tile s)."""
    global _PT
    if _PT is None:
        _PT = Tokenizer.from_file(TOKENIZER).pre_tokenizer
    t = c2b(s)
    out = []
    prev = 0
    for tok, (a, b) in _PT.pre_tokenize_str(s):
        assert a == prev and b > a, (repr(s), a, b)
        assert tok == "".join(B2U[x] for x in s[a:b].encode("utf-8")), (repr(s), a, b, repr(tok))
        out.append((t[a], t[b]))
        prev = b
    assert prev == len(s), (repr(s), prev)
    return out


def seq3_pieces(s):
    """the three Splits built from this file's constants, without ByteLevel."""
    global _SEQ3
    if _SEQ3 is None:
        _SEQ3 = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(p), "isolated", invert=False)
                                         for p in (SPLIT1, SPLIT2, SPLIT3)])
    t = c2b(s)
    return [(t[a], t[b]) for _tok, (a, b) in _SEQ3.pre_tokenize_str(s)]


# ---------------------------------------------------------------- generators
def _chars(*cps):
    return [chr(c) for c in cps]


ASCII_L = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTERS = _chars(0xE9, 0xDF, 0xC0, 0x3C9, 0x416, 0x436, 0x5D0, 0x627, 0x915, 0xE01, 0xAC00, 0x10D0, 0x13A0,
                 0x1E9E, 0x2B0, 0x2BC, 0x1C5, 0xAA, 0x17F, 0x1D400, 0x20000, 0x3400, 0xF900, 0x3005, 0x3006,
                 0x31F0, 0xFF66, 0xFF21, 0xFF41, 0x9FA6, 0x4DB5, 0x16B40, 0x105C0, 0x1100)
MARKS = _chars(0x300, 0x301, 0x308, 0x93E, 0x94D, 0xE31, 0x20DD, 0xFE0F, 0xE0100, 0x897, 0x5B0, 0x64B, 0x1AC1,
               0x302A, 0x1DC0, 0xE48)
DIGITS = list("0123456789") + _chars(0x660, 0x663, 0x6F0, 0x966, 0xB2, 0xB9, 0xBD, 0x2160, 0x2167, 0x2460,
                                     0xFF10, 0x1D7CE, 0x3007, 0x3021, 0x10D40)
WS_CPS = [0x09, 0x0B, 0x0C, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F,
          0x3000]
WSS = _chars(*WS_CPS)
NLS = ["\r", "\n"]
APUNCTS = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
PUNCTS = _chars(0xA1, 0xAB, 0xBB, 0xA9, 0xB4, 0x2019, 0x201C, 0x2014, 0x2026, 0x20AC, 0x2122, 0x2190, 0x2500,
                0x3001, 0x3002, 0x300C, 0xFF01, 0xFF04, 0xFFE5, 0x1F600, 0x1F44D, 0x1F3FB, 0x1F468, 0x2764,
                0x2E80, 0x4DFF, 0x303F, 0xFFFD, 0x5F, 0x60)
OTHERS = _chars(0x00, 0x01, 0x08, 0x0E, 0x1F, 0x7F, 0x80, 0x84, 0x86, 0x9F, 0xAD, 0x200B, 0x200D, 0x200F,
                0x2060, 0xFEFF, 0xE000, 0xF8FF, 0xF0000, 0x10FFFD, 0xE0001, 0x378, 0xFFFF, 0x10FFFF, 0x2FFFF)
CJK_L = _chars(0x4E00, 0x4E2D, 0x6587, 0x65E5, 0x672C, 0x8A9E, 0x9FA5, 0x3041, 0x3042, 0x306E, 0x3093, 0x3096,
               0x309D, 0x309E, 0x309F, 0x30A1, 0x30AB, 0x30C6, 0x30FA, 0x30FC, 0x30FE, 0x30FF, 0x3099, 0x309A)
CJK_P = _chars(0x309B, 0x309C, 0x30A0, 0x30FB)
CJK_X = _chars(0x3040, 0x3097, 0x3098)
CJK_ALL = CJK_L * 3 + CJK_P + CJK_X
JA_WORDS = ["日本語", "テキスト", "の", "は", "です", "カタカナ", "ひらがな", "東京", "中文", "文字", "ガード",
            "ー", "・", "゛", "゠", "ヷヸヹ", "ゝゞ", "ヽヾ", "\u3040", "\u3097\u3098"]
SEPS = [" ", "", "", "\t", "\n", "\r\n", "!", "'", ".", ",", "1", "\u3000", "\u00a0", " \n", "\u0301", "\x00",
        "\U0001F600", "-", "  ", "・", "中", "\u3040", "«", "_"]

# one representative per behaviour, for the exhaustive generators
REP = ["a", "\u00e9", "\u0301", "\u4e2d", "\u3099", "\u30fb", "\u3040", "1", "\u0663", " ", "\t", "\u3000", "\r",
       "\n", "!", "\u00ab", "\U0001F600", "\x00", "\u00ad", "Z"]
REP_SMALL = ["a", "\u0301", "\u4e2d", "\u30fb", "\u3040", "1", " ", "\n", "!", "\u00ab", "\x00", "\u3000"]


def _word(rng):
    k = rng.random()
    if k < 0.45:
        return "".join(rng.choice(ASCII_L) for _ in range(rng.randint(1, 8)))
    if k < 0.65:
        return "".join(rng.choice(ASCII_L + LETTERS + MARKS) for _ in range(rng.randint(1, 6)))
    if k < 0.85:
        return "".join(rng.choice(JA_WORDS + CJK_ALL) for _ in range(rng.randint(1, 4)))
    return "".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 7)))


def gen_case(rng, profile=None):
    pr = profile or rng.choice(["words", "words", "words", "mixed", "mixed", "cjk", "cjk", "apunct", "punct", "ws",
                                "gaps", "digits", "marks", "tiny", "tiny"])
    if pr == "tiny":
        return "".join(rng.choice(REP) for _ in range(rng.randint(0, 6)))
    if pr == "words":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append(rng.choice(SEPS) if rng.random() < 0.8 else rng.choice(PUNCTS + WSS + OTHERS + APUNCTS))
            out.append(_word(rng))
        return "".join(out)
    if pr == "cjk":
        out = []
        for _ in range(rng.randint(1, 10)):
            x = rng.random()
            if x < 0.5:
                out.append(rng.choice(JA_WORDS))
            elif x < 0.75:
                out.append("".join(rng.choice(CJK_ALL) for _ in range(rng.randint(1, 4))))
            else:
                out.append(rng.choice(SEPS + DIGITS + PUNCTS + WSS + NLS + OTHERS + ASCII_L + MARKS))
        return "".join(out)
    if pr == "apunct":                    # D1 against D3: ascii punctuation before ascii and other letters
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "  ", "\t", "a", "!", "'", ".", "\n", "\u00e9", "1", "\x00", "\u3000"]))
            out.append("".join(rng.choice(APUNCTS) for _ in range(rng.randint(1, 3))))
            out.append("".join(rng.choice(ASCII_L if rng.random() < 0.7 else LETTERS + MARKS + CJK_L)
                               for _ in range(rng.randint(0, 4))))
        return "".join(out)
    if pr == "punct":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "\t", "a", "1", "\u0301", "\u4e2d", "\x00"]))
            out.append("".join(rng.choice(PUNCTS + APUNCTS + CJK_P) for _ in range(rng.randint(1, 5))))
            if rng.random() < 0.6:
                out.append("".join(rng.choice(["\r", "\n", "\r\n", " ", "\u30fb"]) for _ in range(rng.randint(1, 4))))
        return "".join(out)
    if pr == "ws":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append("".join(rng.choice(WSS + NLS) for _ in range(rng.randint(1, 6))))
            out.append(rng.choice(["x", "1", "!", "\u4e2d", "\u0301", "", "\x00", "\u30fb", "\u3040", "\u00ab",
                                   "12", "\u3099"]))
        return "".join(out)
    if pr == "gaps":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append("".join(rng.choice(OTHERS + CJK_X) for _ in range(rng.randint(1, 4))))
            out.append(rng.choice(["", "a", "\u0301", "\u4e2d", "\u30ab", " ", "!", "1", "\n", "\u00e9", "\u3099",
                                   " a", "\u30fb"]))
        return "".join(out)
    if pr == "digits":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "  ", "\t", "a", "!", "'", "\u4e2d", "\u3000", "\x00", "\u0301", "."]))
            out.append("".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 10))))
        return "".join(out)
    if pr == "marks":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append(rng.choice(["a", "!", " ", "\t", "'", "1", "", "\n", "\u4e2d", "\x00", "\u30ab", "\u00ab"]))
            out.append("".join(rng.choice(MARKS + ["\u3099", "\u309a"]) for _ in range(rng.randint(1, 3))))
        return "".join(out)
    n = rng.randint(0, 120)                                       # mixed: runs from every pool
    out = []
    pools = [ASCII_L, LETTERS, MARKS, DIGITS, WSS, NLS, APUNCTS, PUNCTS, OTHERS, CJK_L, CJK_P, CJK_X, REP]
    while len(out) < n:
        out.append(rng.choice(rng.choice(pools)) * rng.randint(1, 3))
    return "".join(out)


def gen_long(rng):
    kind = rng.choice(["words", "cjk", "letters", "punct", "ws", "digits", "mixed", "base64", "code"])
    n = rng.randint(60, 1000)
    if kind == "words":
        out = []
        while sum(map(len, out)) < n:
            out.append(rng.choice(SEPS))
            out.append(_word(rng))
        return "".join(out)[:n]
    if kind == "cjk":
        return gen_case(rng, "cjk") * (1 + n // 40)
    if kind == "letters":
        return "".join(rng.choice(ASCII_L + LETTERS + MARKS + CJK_L) for _ in range(n))
    if kind == "punct":
        return "".join(rng.choice(APUNCTS + PUNCTS + CJK_P + NLS + [" "]) for _ in range(n))
    if kind == "ws":
        return "".join(rng.choice(WSS + NLS) for _ in range(n))
    if kind == "digits":
        return "".join(rng.choice(DIGITS) for _ in range(n))
    if kind == "base64":
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") for _ in range(n))
    if kind == "code":
        out = []
        while sum(map(len, out)) < n:
            out.append(rng.choice(["def ", "x.y", "(a, b)", " = ", "self._x", "\n    ", "#include <a.h>\n", "->",
                                   "'s'", "\"str\"", "0x1F", "1e-9", "  # 注释\n", "//", "/*", "*/", "::", "{}", "[]",
                                   "__init__", "\t", "x+=1;", "$var", "@dec", "%d", "&&", "||", "!=", "`cmd`"]))
        return "".join(out)[:n]
    return gen_case(rng, "mixed")[:n]


def segments_of(text, rng, max_chunk=16384):
    """real-text segments: every line, every paragraph, and random chunks of 1..max_chunk chars."""
    for ln in text.splitlines(keepends=True):
        yield ln
    for para in text.split("\n\n"):
        if para:
            yield para
    i = 0
    while i < len(text):
        k = int(2 ** rng.uniform(0, 14))
        k = max(1, min(k, max_chunk))
        yield text[i:i + k]
        i += k


_VOCAB = None


def vocab_strings():
    """the pinned file's model vocabulary as text: each token's bytes (byte-level alphabet), when utf-8."""
    global _VOCAB
    if _VOCAB is None:
        with open(TOKENIZER, "rb") as f:
            tj = json.loads(f.read())
        inv = {u: x for x, u in B2U.items()}
        out = []
        for tok in tj["model"]["vocab"]:
            try:
                out.append(bytes(inv[ch] for ch in tok).decode("utf-8"))
            except (KeyError, UnicodeDecodeError):
                pass
        _VOCAB = out
    return _VOCAB
