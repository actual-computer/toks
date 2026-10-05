#!/usr/bin/env python3
"""o200k template model: docs/templates/o200k.md §2-§3 transcribed literally, vs hf tokenizers 0.23.2.

  - classes come from hf's own regex engine (oniguruma), never python's unicodedata: every scalar value is run
    through pre_tokenizers.Split(Regex(set), 'removed') for the pattern's five sets (UP, LO, PFX, P, \\p{N}),
    \\p{L}, \\p{M}, \\s and the eight (?i) letter folds, and the contraction group is probed over every scalar in
    'X / 'Xe / 'Xl contexts (tools/gen/unicode.py's method). The doc's §2 claims are asserted on the probes.
  - k3_scan() is §3: L1 / L2 / K, O1-O6, in the order the doc gives, evaluated generically (L1(i) is computed,
    not assumed); the doc's two side claims ("L1(i) applies exactly when c is a mark", "L2(i) never decides")
    are asserted on every piece.
  - hf_pieces() is the oracle: Split(Regex(pattern), 'isolated') (or 'removed' + invert, which the doc says is
    the same); the binding returns character offsets, converted to byte offsets with an asserted slice check.
  - generators: class-aware random strings (gen_case / gen_long), exhaustive strings over a representative
    alphabet (exhaustive), and real-text segments (segments_of).

Used by tests/model/run_o200k_fuzz.py (the proof runs) and tests/k3/o200k_gen.py (cases for the c twin).
Probe cache: build/o200k_probes.pkl (TOKS_O200K_PROBES overrides); build/ survives tools/remote.sh syncs.
"""
import hashlib
import json
import os
import pickle
import random
import sys

from tokenizers import Regex, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBE_PATH = os.environ.get("TOKS_O200K_PROBES", os.path.join(REPO, "build", "o200k_probes.pkl"))

# ---------------------------------------------------------------- patterns (docs/templates/o200k.md §1)
_CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
_UPG = r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"
_LOG = r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]"
_PFX = r"[^\r\n\p{L}\p{N}]?"
_TAIL = r"| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
O200K_PATTERN = (_PFX + _UPG + "*" + _LOG + "+" + _CONTR + "|" + _PFX + _UPG + "+" + _LOG + "*" + _CONTR
                 + r"|\p{N}{1,3}" + _TAIL)
NEMO_PATTERN = _PFX + _UPG + "*" + _LOG + "+|" + _PFX + _UPG + "+" + _LOG + "*" + r"|\p{N}" + _TAIL

# kimi (moonshotai/Kimi-K3, tokenization_kimi.py lines 54-63, a tiktoken pat_str): docs/templates/o200k.md §6
_NOHAN = "&&[^\\p{Han}]]"
KIMI_PATTERN = ("[\\p{Han}]+|" + _PFX + _UPG[:-1] + _NOHAN + "*" + _LOG[:-1] + _NOHAN + "+" + _CONTR + "|"
                + _PFX + _UPG[:-1] + _NOHAN + "+" + _LOG[:-1] + _NOHAN + "*" + _CONTR
                + r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
KIMI_FILE = (os.path.expanduser("~/.cache/toks/tokenizers/kimik3_tokenization_kimi.py"),
             "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944")

# tmpl_params: layout.h TOKS_TP_CONTR_CI = 0x02, TOKS_TP_DIGITS_1 = 0x08 (DIGITS_1_3 = 0), and (§6)
# TOKS_TP_HAN = 0x80, TOKS_TP_NO_SLASH = 0x100. engine: the oracle (hf's onig, or tiktoken's fancy-regex).
VARIANTS = {
    "o200k": dict(pattern=O200K_PATTERN, contr=True, digits3=True, han=False, slash=True, params=0x02, vi=0,
                  engine="hf"),
    "nemo": dict(pattern=NEMO_PATTERN, contr=False, digits3=False, han=False, slash=True, params=0x08, vi=1,
                 engine="hf"),
    "kimi": dict(pattern=KIMI_PATTERN, contr=True, digits3=True, han=True, slash=False, params=0x182, vi=2,
                 engine="tiktoken"),
}

# the real files the strings were read from (docs/templates/o200k.md §1); checked when present locally
PATTERN_FILES = [
    ("o200k", os.path.expanduser("~/.cache/toks/tokenizers/o200k"),
     "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"),
    ("nemo", os.path.expanduser("~/.cache/toks/tokenizers/nemotron3-4b"),
     "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"),
    ("nemo", os.path.expanduser("~/.cache/toks/tokenizers/mistral-nemo"),
     "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db"),
]


def check_pattern_files():
    """-> list of report lines; raises when a present file disagrees with its pin or with the pattern."""
    out = []
    for variant, path, sha in PATTERN_FILES:
        if not os.path.exists(path):
            out.append("pattern file %s: absent (not checked)" % path)
            continue
        with open(path, "rb") as f:
            raw = f.read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha:
            raise SystemExit("pattern file %s: sha256 %s != pin %s" % (path, got, sha))
        tj = json.loads(raw)
        pts = tj["pre_tokenizer"]["pretokenizers"]
        pat = pts[0]["pattern"]["Regex"]
        if pat != VARIANTS[variant]["pattern"]:
            raise SystemExit("pattern file %s: pattern differs from the %s string" % (path, variant))
        out.append("pattern file %s: sha256 ok, %s pattern identical, behavior %s invert %s"
                   % (path, variant, pts[0]["behavior"], pts[0]["invert"]))
    path, sha = KIMI_FILE
    if os.path.exists(path):
        import re
        raw = open(path, "rb").read()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit("kimi file %s: sha256 != pin %s" % (path, sha))
        m = re.search(r'pat_str = "\|"\.join\(\[(.*?)\]\)', raw.decode("utf-8"), re.S)
        if "|".join(re.findall(r'r"""(.*?)"""', m.group(1), re.S)) != KIMI_PATTERN:
            raise SystemExit("kimi file %s: pat_str differs from KIMI_PATTERN" % path)
        out.append("pattern file %s: sha256 ok, kimi pat_str identical" % path)
    else:
        out.append("pattern file %s: absent (not checked)" % path)
    return out


# ---------------------------------------------------------------- probes through hf's onig
SETS = [
    ("L", r"\p{L}"), ("N", r"\p{N}"), ("M", r"\p{M}"), ("S", r"\s"),
    ("UP", _UPG), ("LO", _LOG), ("PFX", r"[^\r\n\p{L}\p{N}]"), ("P", r"[^\s\p{L}\p{N}]"),
] + [("fold_" + ch, "(?i:%s)" % ch) for ch in "stmdrevl"]
CONTRACTION = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
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


def probe_contraction(cps, tail):
    """code points X for which 'X<tail> lets the contraction group consume X (items NUL-separated)."""
    sp = pre_tokenizers.Split(Regex(CONTRACTION), "removed")
    item = 2 + len(tail) + 1
    hit = set()
    for i in range(0, len(cps), 0x1000):
        part = cps[i:i + 0x1000]
        keep = _kept(sp, "".join("'" + chr(cp) + tail + "\0" for cp in part))
        for k, cp in enumerate(part):
            if not keep[k * item + 1]:
                hit.add(cp)
    return hit


def run_probes(force=False, log=sys.stderr):
    if os.path.exists(PROBE_PATH) and not force:
        with open(PROBE_PATH, "rb") as f:
            return pickle.load(f)
    cps = scalars()
    pr = {}
    for name, pat in SETS:
        pr[name] = probe_set(pat, cps)
        print("probe %-7s %7d code points  %s" % (name, len(pr[name]), pat), file=log, flush=True)
    for name, tail in (("contr_2", ""), ("contr_e", "e"), ("contr_l", "l")):
        pr[name] = probe_contraction(cps, tail)
        print("probe %-7s %7d code points  'X%s" % (name, len(pr[name]), tail), file=log, flush=True)
    os.makedirs(os.path.dirname(PROBE_PATH), exist_ok=True)
    with open(PROBE_PATH, "wb") as f:
        pickle.dump(pr, f)
    return pr


# class byte: layout.h values (base | flags)
P, L, N, WS, NL = 0, 1, 2, 3, 4
UPPER, LOWER, MARK, FOLD_S = 0x08, 0x10, 0x20, 0x40

PROBE = None
CLS = None          # bytearray(0x110000): the class byte of every code point, from the probes
FOLDS = None        # dict letter -> set of code points (?i) matches


def build_classes(pr):
    """class bytes from the probes, then the doc's §2 claims asserted over every scalar."""
    cls = bytearray(0x110000)
    Ls, Ns, Ms, Ss, UPs, LOs = pr["L"], pr["N"], pr["M"], pr["S"], pr["UP"], pr["LO"]
    fold_s_nonascii = {cp for cp in pr["fold_s"] if cp >= 0x80}
    for cp in scalars():
        if cp in Ls:
            b = L
        elif cp in Ns:
            b = N
        elif cp in (0x0A, 0x0D):
            b = NL
        elif cp in Ss:
            b = WS
        else:
            b = P
        if cp in UPs:
            b |= UPPER
        if cp in LOs:
            b |= LOWER
        if cp in Ms:
            b |= MARK
        if cp in fold_s_nonascii:
            b |= FOLD_S
        cls[cp] = b
    # §2: the five sets are exactly class-byte tests; marks are base P in both groups; no other overlap
    sc = scalars()
    bad = []
    for cp in sc:
        c = cls[cp]
        base = c & 7
        checks = (
            ((cp in pr["PFX"]) == (base in (P, WS)), "PFX == base in {P, WS}"),
            ((cp in pr["P"]) == (base == P), "P == base P"),
            ((cp in Ss) == (base in (WS, NL)), "\\s == base in {WS, NL}"),
            ((cp in UPs or cp in LOs) == (cp in Ls or cp in Ms), "UP | LO == L | M"),
            (not (cp in Ms) or (cp in UPs and cp in LOs and base == P), "M => UP & LO & base P"),
            (not (cp in Ns and (cp in Ls or cp in Ss)), "N disjoint from L, \\s"),
            (not (cp in Ls and cp in Ss), "L disjoint from \\s"),
        )
        for ok, what in checks:
            if not ok:
                bad.append((cp, what))
    if bad:
        raise SystemExit("o200k_model: class claims fail: %s" % ", ".join("U+%04X %s" % b for b in bad[:20]))
    folds = {ch: pr["fold_" + ch] for ch in "stmdrevl"}
    expect = {ch: {ord(ch), ord(ch.upper())} for ch in "stmdrevl"}
    expect["s"] |= {0x17F}
    if folds != expect:
        raise SystemExit("o200k_model: (?i) letter folds are not {x, X} (+ U+017F for s): %r" % {
            k: sorted(v) for k, v in folds.items()})
    exp2 = folds["s"] | folds["t"] | folds["m"] | folds["d"]
    for name, exp in (("contr_2", exp2), ("contr_e", exp2 | folds["r"] | folds["v"]),
                      ("contr_l", exp2 | folds["l"])):
        if pr[name] != exp:
            raise SystemExit("o200k_model: contraction probe %s: extra %r missing %r"
                             % (name, sorted(pr[name] - exp), sorted(exp - pr[name])))
    return cls, folds


def init(force=False):
    global PROBE, CLS, FOLDS
    if CLS is None or force:
        PROBE = run_probes(force)
        CLS, FOLDS = build_classes(PROBE)
    return CLS


# ---------------------------------------------------------------- kimi: probes through tiktoken's engine
# tiktoken compiles pat_str with fancy-regex, which hands classes to the regex crate (regex-syntax): class
# membership is probed through tiktoken itself. A one-char pattern over a vocabulary holding every scalar as
# one token: tiktoken encodes only the regex matches (gaps are skipped), so the tokens are the members.
HAN = 0x80                                    # TOKS_C_HAN (layout.h; docs/templates/o200k.md §6)
KIMI_PROBE_PATH = os.environ.get("TOKS_KIMI_PROBES", os.path.join(REPO, "build", "kimi_probes.pkl"))
KIMI_SETS = [
    ("HAN", r"[\p{Han}]"), ("L", r"\p{L}"), ("N", r"\p{N}"), ("M", r"\p{M}"), ("S", r"\s"),
    ("UP", _UPG), ("LO", _LOG), ("UPK", _UPG[:-1] + _NOHAN), ("LOK", _LOG[:-1] + _NOHAN),
    ("PFX", r"[^\r\n\p{L}\p{N}]"), ("P", r"[^\s\p{L}\p{N}]"),
] + [("fold_" + ch, "(?i:%s)" % ch) for ch in "stmdrevl"]
# \p{Han} as tiktoken resolves it: regex-syntax 0.8.11's Script=Han (unicode_tables/script.rs, Unicode 16.0.0; a
# bare \p{Han} is the Script property there, unicode.rs canonical_binary, not Script_Extensions). The one copy is
# src/gen/han_ranges.c (tools/gen/han.py, from tiktoken's own engine; classes.c builds TOKS_C_HAN from it), read
# here and asserted equal to the probe.
def _read_han_ranges():
    import re
    with open(os.path.join(REPO, "src", "gen", "han_ranges.c"), encoding="utf-8") as f:
        src = f.read()
    rs = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"\{ 0x([0-9A-F]+)u, 0x([0-9A-F]+)u \}", src)]
    if len(rs) != 22 or sum(b - a + 1 for a, b in rs) != 99030:
        raise SystemExit("o200k_model: src/gen/han_ranges.c is not the 22-range, 99,030-code-point table")
    return rs


HAN_RANGES = _read_han_ranges()


def tiktoken_probe_set(pat, cps):
    import tiktoken
    ranks = {chr(cp).encode("utf-8"): i for i, cp in enumerate(cps)}
    enc = tiktoken.Encoding("probe", pat_str=pat, mergeable_ranks=ranks, special_tokens={})
    hit = set()
    for i in range(0, len(cps), BATCH):
        for t in enc.encode_ordinary("".join(map(chr, cps[i:i + BATCH]))):
            hit.add(ord(enc.decode_single_token_bytes(t).decode("utf-8")))
    return hit


def run_kimi_probes(force=False, log=sys.stderr):
    if os.path.exists(KIMI_PROBE_PATH) and not force:
        with open(KIMI_PROBE_PATH, "rb") as f:
            return pickle.load(f)
    import importlib.metadata as md
    cps = scalars()
    pr = {"tiktoken": md.version("tiktoken")}
    for name, pat in KIMI_SETS:
        pr[name] = tiktoken_probe_set(pat, cps)
        print("kimi probe %-7s %7d code points  %s" % (name, len(pr[name]), pat), file=log, flush=True)
    os.makedirs(os.path.dirname(KIMI_PROBE_PATH), exist_ok=True)
    with open(KIMI_PROBE_PATH, "wb") as f:
        pickle.dump(pr, f)
    return pr


KPROBE = None
CLS_KIMI = None


def build_kimi_classes(pr):
    """the class byte as tiktoken's engine sees each scalar, plus HAN; §6's claims asserted."""
    cls = bytearray(0x110000)
    han = pr["HAN"]
    want_han = {cp for a, b in HAN_RANGES for cp in range(a, b + 1)}
    if han != want_han:
        raise SystemExit("kimi: tiktoken's \\p{Han} != regex-syntax 0.8.11 Script=Han: extra %r missing %r"
                         % (sorted(han - want_han)[:10], sorted(want_han - han)[:10]))
    fold_s_nonascii = {cp for cp in pr["fold_s"] if cp >= 0x80}
    for cp in scalars():
        b = L if cp in pr["L"] else N if cp in pr["N"] else NL if cp in (0x0A, 0x0D) else \
            WS if cp in pr["S"] else P
        b |= (UPPER if cp in pr["UP"] else 0) | (LOWER if cp in pr["LO"] else 0) | (MARK if cp in pr["M"] else 0)
        b |= (FOLD_S if cp in fold_s_nonascii else 0) | (HAN if cp in han else 0)
        cls[cp] = b
    bad = []
    for cp in scalars():
        c = cls[cp]
        base = c & 7
        for ok, what in (
                ((cp in pr["UPK"]) == ((cp in pr["UP"]) and cp not in han), "UPK == UP - HAN"),
                ((cp in pr["LOK"]) == ((cp in pr["LO"]) and cp not in han), "LOK == LO - HAN"),
                ((cp in pr["PFX"]) == (base in (P, WS)), "PFX == base in {P, WS}"),
                ((cp in pr["P"]) == (base == P), "P == base P"),
                ((cp in pr["S"]) == (base in (WS, NL)), "\\s == base in {WS, NL}"),
                (not (cp in pr["M"]) or (cp in pr["UP"] and cp in pr["LO"] and base == P), "M => UP & LO & P")):
            if not ok:
                bad.append((cp, what))
    if bad:
        raise SystemExit("kimi: class claims fail: %s" % ", ".join("U+%04X %s" % b for b in bad[:20]))
    expect = {ch: {ord(ch), ord(ch.upper())} for ch in "stmdrevl"}
    expect["s"] |= {0x17F}
    got = {ch: pr["fold_" + ch] for ch in "stmdrevl"}
    if got != expect:
        raise SystemExit("kimi: (?i) letter folds are not {x, X} (+ U+017F): %r" % {k: sorted(v) for k, v in got.items()})
    return cls


def init_kimi(force=False):
    global KPROBE, CLS_KIMI
    init()                                            # FOLDS and the onig tables (for comparisons)
    if CLS_KIMI is None or force:
        KPROBE = run_kimi_probes(force)
        CLS_KIMI = build_kimi_classes(KPROBE)
    return CLS_KIMI


KIMI_MAXB = 64                                        # the substring-vocabulary oracle's string bound


def tiktoken_pieces(strings):
    """tiktoken's own pieces for kimi's pattern: with every substring of every string in the vocabulary, each
    regex piece is one token (tiktoken looks a piece up whole before any merge), so decoding the tokens gives
    the pieces. Sound for any string; the vocabulary is O(len^2), hence KIMI_MAXB."""
    import tiktoken
    ranks = {bytes([b]): b for b in range(256)}
    for s in strings:
        raw = s.encode("utf-8")
        assert len(raw) <= KIMI_MAXB, len(raw)
        for i in range(len(raw)):
            for j in range(i + 1, len(raw) + 1):
                if raw[i:j] not in ranks:
                    ranks[raw[i:j]] = len(ranks)
    enc = tiktoken.Encoding("kimi-oracle", pat_str=KIMI_PATTERN, mergeable_ranks=ranks, special_tokens={})
    out = []
    for s in strings:
        o, ps = 0, []
        for t in enc.encode_ordinary(s):
            n = len(enc.decode_single_token_bytes(t))
            ps.append((o, o + n))
            o += n
        out.append(ps)
    return out


def oracle_batch(variant, strings):
    if VARIANTS[variant]["engine"] == "tiktoken":
        return tiktoken_pieces(strings)
    return [hf_pieces(variant, s) for s in strings]


# ---------------------------------------------------------------- §3: the template rules
def _boffs(cps):
    t = [0] * (len(cps) + 1)
    o = 0
    for i, cp in enumerate(cps):
        t[i] = o
        o += 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
    t[len(cps)] = o
    return t


def k3_scan(s, variant):
    """docs/templates/o200k.md §3 on the atoms of s (a str: every atom a scalar). -> [(start, end)] bytes."""
    V = VARIANTS[variant]
    contr, digits3, han, slash = V["contr"], V["digits3"], V["han"], V["slash"]
    F = FOLDS
    two = F["s"] | F["t"] | F["m"] | F["d"]
    cps = [ord(ch) for ch in s]
    n = len(cps)
    table = CLS_KIMI if V["engine"] == "tiktoken" else CLS
    cl = [table[cp] for cp in cps]
    if han:                            # §6: HAN atoms leave both case groups
        cl = [c & ~(UPPER | LOWER) if c & HAN else c for c in cl]
    tail = (0x0D, 0x0A, 0x2F) if slash else (0x0D, 0x0A)
    boff = _boffs(cps)

    def base(j):                       # absent atoms are in no set
        return cl[j] & 7 if j < n else -1

    def up(j):
        return j < n and (cl[j] & UPPER) != 0

    def lo(j):
        return j < n and (cl[j] & LOWER) != 0

    def K(e):                          # the contraction suffix at e, in atoms
        if not contr or e >= n or cps[e] != 0x27 or e + 1 >= n:
            return 0
        c1 = cps[e + 1]
        if c1 in two:
            return 2
        if e + 2 < n:
            c2 = cps[e + 2]
            if (c1 in F["r"] and c2 in F["e"]) or (c1 in F["v"] and c2 in F["e"]) or \
               (c1 in F["l"] and c2 in F["l"]):
                return 3
        return 0

    def u_of(s0):
        u = s0
        while up(u):
            u += 1
        return u

    def L1(s0):
        u = u_of(s0)
        if lo(u):
            e = u
            while lo(e):
                e += 1
            return e + K(e)
        j = u - 1
        while j >= s0 and not lo(j):
            j -= 1
        if j >= s0:
            return j + 1 + K(j + 1)
        return None

    def L2(s0):
        if not up(s0):
            return None
        e = u_of(s0)
        while lo(e):
            e += 1
        return e + K(e)

    pieces = []
    i = 0
    while i < n:
        c = cps[i]
        b = base(i)
        e = None
        # O0 Han (§6)
        if han and (cl[i] & HAN):
            e = i + 1
            while e < n and (cl[e] & HAN):
                e += 1
        # O1 letters
        elif b in (P, WS):
            e = L1(i + 1)
            if e is None:
                e = L1(i)
                assert (e is not None) == ((cl[i] & MARK) != 0 and not (cl[i] & HAN)), \
                    "doc: L1(i) applies exactly for a (non-Han) mark"
            if e is None:
                e = L2(i + 1)
            if e is None:
                assert L2(i) is None, "doc: L2(i) never decides"
        elif b == L:
            e = L1(i)
            if e is None:
                e = L2(i)
            assert e is not None, "doc: an L atom is UP or LO"
        # O2 digits
        if e is None and b == N:
            e = i + 1
            if digits3:
                while e < n and e < i + 3 and base(e) == N:
                    e += 1
        # O3 punctuation
        if e is None:
            j = i if b == P else (i + 1 if c == 0x20 and base(i + 1) == P else None)
            if j is not None:
                e = j
                while base(e) == P:
                    e += 1
                while e < n and cps[e] in tail:
                    e += 1
        # O4-O6 whitespace
        if e is None:
            assert b in (WS, NL), "every atom is matched by some alternative"
            k = i
            while base(k) in (WS, NL):
                k += 1
            t = k - 1
            while t >= i and base(t) != NL:
                t -= 1
            if t >= i:
                e = t + 1                  # O4: through the last NL of the run
            elif k == n:
                e = k                      # O5: the run ends the segment
            elif k - i >= 2:
                e = k - 1                  # O5: give back the last atom
            else:
                e = k                      # O6
        pieces.append((boff[i], boff[e]))
        i = e
    return pieces


# ---------------------------------------------------------------- the oracle
_SPLITS = {}


def c2b(s):
    t = [0] * (len(s) + 1)
    o = 0
    for i, ch in enumerate(s):
        t[i] = o
        cp = ord(ch)
        o += 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4
    t[len(s)] = o
    return t


def hf_pieces(variant, s, removed_invert=False):
    key = (variant, removed_invert)
    sp = _SPLITS.get(key)
    if sp is None:
        pat = Regex(VARIANTS[variant]["pattern"])
        sp = (pre_tokenizers.Split(pat, behavior="removed", invert=True) if removed_invert
              else pre_tokenizers.Split(pat, behavior="isolated", invert=False))
        _SPLITS[key] = sp
    t = c2b(s)
    out = []
    for tok, (a, b) in sp.pre_tokenize_str(s):
        assert s[a:b] == tok, (repr(s), a, b, repr(tok))   # the binding's offsets are character offsets
        out.append((t[a], t[b]))
    return out


# ---------------------------------------------------------------- generators
def _chars(*cps):
    return [chr(c) for c in cps]


UPPER_ONLY = list("ABCDEKLMRSTVZ") + _chars(0xC0, 0x3A9, 0x416, 0x531, 0x10A0, 0xFF21, 0x1E9E, 0x212A,
                                             0x1D400, 0x1C4, 0x1C5, 0x1C8, 0x1F88, 0x1E921)
LOWER_ONLY = list("abcdeklmrstvxz") + _chars(0xDF, 0xE9, 0x3C9, 0x436, 0x17F, 0xFB00, 0xFF41, 0x1D41A, 0x1C6,
                                              0x149, 0x10D0, 0x1E943)
BOTH = _chars(0x2B0, 0x2C6, 0x30FC, 0x1D43, 0xA717, 0x16B40, 0x4E2D, 0x6587, 0x5D0, 0x627, 0x3042, 0xAC00,
              0xE01, 0x915, 0x20000, 0xAA, 0xBA, 0x1BB, 0x1C0, 0x13A0, 0x1100)
MARKS = _chars(0x300, 0x301, 0x308, 0x361, 0x93E, 0x94D, 0xE31, 0x20DD, 0xFE0F, 0xE0100, 0x897, 0x1AC1,
               0x3099, 0x5B0, 0x64B, 0x1F3B)
DIGITS = list("0123456789") + _chars(0x660, 0x661, 0xB2, 0xB9, 0xBD, 0x2160, 0x2460, 0xFF10, 0x1D7CE, 0x6F0,
                                     0x966)
WS_CPS = [0x09, 0x0B, 0x0C, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F,
          0x3000]
WSS = _chars(*WS_CPS)
NLS = ["\r", "\n"]
PUNCTS = list("!\"#$%&()*+,-.:;<=>?@[\\]^_`{|}~/'") + _chars(
    0xA1, 0xA9, 0x2019, 0x2190, 0x2500, 0x3001, 0xFF01, 0x1F600, 0x1F44D, 0x1F3FB, 0x200D, 0xAD, 0xFEFF,
    0xE000, 0xFFFF, 0x10FFFF, 0x0, 0x1, 0x7F, 0x80, 0x378, 0x1F1E6, 0xFFFD)
CONTR_FOLLOW = list("stmdrevlSTMDREVL") + _chars(0x17F, 0x212A)
SUFFIXES = ["'s", "'S", "'t", "'T", "'m", "'M", "'d", "'D", "'re", "'RE", "'rE", "'ve", "'Ve", "'ll", "'LL",
            "'lL", "'\u017f", "'\u212a", "'x", "'", "'r", "'v", "'l", "'e", "''s", "'s's", "'ll'"]
SEPS = [" ", "", "", "\t", "\n", "\r\n", "!", "/", "'", ".", ",", "1", "\u3000", "\u00a0", " \n", "\u0301",
        "\U0001F600", "-", "  "]

# one representative per behaviour, for the exhaustive generator
REP = ["A", "a", "\u01c5", "\u02b0", "\u4e2d", "\u0301", "5", " ", "\t", "\u3000", "\r", "\n", "!", "/", "'",
       "s", "S", "l", "e", "r", "\u017f", "\U0001F600", "\u00e9", "\u0660"]
REP_SMALL = ["A", "a", "\u4e2d", "\u0301", "5", " ", "\n", "!", "/", "'", "s", "l"]


def _word(rng):
    """a case-structured word: upper / lower / both runs, marks, an optional suffix."""
    out = []
    for _ in range(rng.randint(1, 4)):
        kind = rng.random()
        pool = UPPER_ONLY if kind < 0.3 else LOWER_ONLY if kind < 0.65 else BOTH if kind < 0.9 else MARKS
        out.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 5))))
    if rng.random() < 0.35:
        out.append(rng.choice(SUFFIXES))
    return "".join(out)


def gen_case(rng, profile=None):
    pr = profile or rng.choice(["words", "words", "words", "mixed", "mixed", "marks", "apos", "punct", "ws",
                                "crlf", "digits", "tiny", "tiny"])
    if pr == "tiny":
        return "".join(rng.choice(REP) for _ in range(rng.randint(0, 5)))
    if pr == "words":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append(rng.choice(SEPS) if rng.random() < 0.8 else rng.choice(PUNCTS + WSS))
            out.append(_word(rng))
        return "".join(out)
    if pr == "marks":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append(rng.choice(["a", "A", "\u4e2d", "!", " ", "\t", "'", "1", "", "\n", "/", "\u02b0"]))
            out.append("".join(rng.choice(MARKS) for _ in range(rng.randint(1, 3))))
            if rng.random() < 0.5:
                out.append(rng.choice(UPPER_ONLY + LOWER_ONLY + BOTH + SUFFIXES + PUNCTS))
        return "".join(out)
    if pr == "apos":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "\t", "a", "A", "\u4e2d", "1", "!", "\u0301", "ab", "AB", "aB"]))
            out.append("'")
            for _ in range(rng.randint(0, 3)):
                out.append(rng.choice(CONTR_FOLLOW) if rng.random() < 0.75 else rng.choice(REP))
        return "".join(out)
    if pr == "punct":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["", " ", "\t", "a", "1", "A", "\u0301"]))
            out.append("".join(rng.choice(PUNCTS) for _ in range(rng.randint(1, 5))))
            if rng.random() < 0.6:
                out.append("".join(rng.choice(["\r", "\n", "/", "\r\n", "//"]) for _ in range(rng.randint(1, 4))))
            if rng.random() < 0.3:
                out.append(rng.choice(["!", "a", " ", "/", "\u0301", "1"]))
        return "".join(out)
    if pr == "ws":
        out = []
        for _ in range(rng.randint(1, 8)):
            out.append("".join(rng.choice(WSS + NLS) for _ in range(rng.randint(1, 6))))
            out.append(rng.choice(["x", "X", "1", "!", "\u4e2d", "\u0301", "", "/", "'s"]))
        return "".join(out)
    if pr == "crlf":
        out = []
        for _ in range(rng.randint(1, 6)):
            out.append(rng.choice(["\r", "\n", "\r\n", "\n\r", "\t", " ", "/"]) * rng.randint(1, 4))
            if rng.random() < 0.7:
                out.append(rng.choice(["x", "!", "1", "", "!/", "\u0301"]))
        return "".join(out)
    if pr == "digits":
        out = []
        for _ in range(rng.randint(1, 5)):
            out.append(rng.choice(["", " ", "\t", "a", "!", "A", "'s"]))
            out.append("".join(rng.choice(DIGITS) for _ in range(rng.randint(1, 9))))
        return "".join(out)
    # mixed: runs from every pool
    n = rng.randint(0, 120)
    out = []
    pools = [UPPER_ONLY, LOWER_ONLY, BOTH, MARKS, DIGITS, WSS, NLS, PUNCTS, CONTR_FOLLOW, SUFFIXES, REP]
    while len(out) < n:
        out.append(rng.choice(rng.choice(pools)) * rng.randint(1, 3))
    return "".join(out)


def gen_long(rng):
    kind = rng.choice(["words", "letters", "punct", "ws", "digits", "mixed", "base64", "paths"])
    n = rng.randint(60, 1000)
    if kind == "words":
        out = []
        while sum(map(len, out)) < n:
            out.append(rng.choice(SEPS))
            out.append(_word(rng))
        return "".join(out)[:n]
    if kind == "letters":
        pool = UPPER_ONLY + LOWER_ONLY + BOTH + MARKS
        return "".join(rng.choice(pool) for _ in range(n))
    if kind == "punct":
        return "".join(rng.choice(PUNCTS + NLS + ["/"]) for _ in range(n))
    if kind == "ws":
        return "".join(rng.choice(WSS + NLS) for _ in range(n))
    if kind == "digits":
        return "".join(rng.choice(DIGITS) for _ in range(n))
    if kind == "base64":
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") for _ in range(n))
    if kind == "paths":
        out = []
        while sum(map(len, out)) < n:
            out.append(rng.choice(["https://", "/", "//", "./", "../", "\n/", "\r\n//", "a/b", "Foo/", "x.y",
                                   "?q=1&r=2", " ", "\n", "C:\\", "'s/", "*/\n", "/*", "\u4e2d/"]))
        return "".join(out)[:n]
    return gen_case(rng, "mixed")[:n]


def exhaustive(alphabet, length):
    """every string of exactly `length` atoms over the alphabet (itertools.product order)."""
    import itertools
    for t in itertools.product(alphabet, repeat=length):
        yield "".join(t)


def segments_of(text, rng, max_chunk=16384):
    """real-text segments: every line, every paragraph, and random chunks of 1..max_chunk chars."""
    lines = text.splitlines(keepends=True)
    for ln in lines:
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



# kimi: Han atoms of every base class (Lo, Lm, Nl, So, Mc, Po) and the scripts around them
HAN_POOL = _chars(0x4E2D, 0x6587, 0x65E5, 0x672C, 0x8A9E, 0x3005, 0x303B, 0x3007, 0x3021, 0x2E80, 0x2F00,
                  0x16FF0, 0x16FE2, 0x16FE3, 0x20000, 0xF900, 0x3400)
NEAR_HAN = _chars(0x3001, 0x3002, 0xFF0C, 0xFF01, 0x300C, 0x300D, 0x30FC, 0x306E, 0x30C6, 0xD55C, 0x3000, 0x302A,
                  0x3099, 0xFF21, 0xFF41)
REP_KIMI = REP + ["\u4e2d", "\u3007", "\u2e80", "\u3005", "\U00016ff0", "\u3001"]


def gen_kimi(rng):
    """a string of at most KIMI_MAXB utf-8 bytes mixing Han atoms into the o200k profiles."""
    pools = [HAN_POOL, HAN_POOL, NEAR_HAN, UPPER_ONLY, LOWER_ONLY, BOTH, MARKS, DIGITS, WSS, NLS, PUNCTS,
             CONTR_FOLLOW, SUFFIXES, REP]
    out, size = [], 0
    for _ in range(rng.randint(0, 24)):
        x = rng.choice(rng.choice(pools)) * rng.randint(1, 3)
        b = len(x.encode("utf-8"))
        if size + b > KIMI_MAXB:
            break
        out.append(x)
        size += b
    return "".join(out)


def kimi_real_slices(text, rng, n):
    """n random slices of real text, each at most KIMI_MAXB utf-8 bytes, cut at char boundaries."""
    for _ in range(n):
        if not text:
            return
        i = rng.randrange(len(text))
        k = rng.randint(1, KIMI_MAXB)
        piece = text[i:i + k]
        while len(piece.encode("utf-8")) > KIMI_MAXB:
            piece = piece[:-1]
        yield piece
