# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""toks normalizer tables (NFC, NFKC, NFD, NFKD, StripAccents), generated from the exact data hf tokenizers 0.23.2 runs,
verified through hf itself.

hf `normalizers.NFC` is `NormalizedString::nfc` = `str.nfc()` of the crate unicode-normalization-alignments
0.1.12 (bindings/python/Cargo.lock of tokenizers v0.23.2, checksum pinned below). Its tables.rs embeds
UNICODE_VERSION = (9, 0, 0): Unicode 9.0 decompositions, combining classes and composition pairs, NOT the
Unicode version of python's unicodedata or of oniguruma (16.0, docs/unicode.md).

This script
  1. downloads that crate from crates.io, checks the sha256 Cargo.lock pins, and parses tables.rs: the
     canonical combining classes, the full canonical and compatibility decompositions, the composition pairs
     (BMP table + the astral match) and the combining marks (General_Category=Mark: hf's StripAccents);
  2. models the crate's algorithm exactly (decompose.rs + recompose.rs: ascii passthrough, hangul
     arithmetic, stable sort of every non-starter run by ccc, the composee / last_ccc / buffer state machine);
  3. verifies the model against hf's own NFC and NFD (tokenizers==0.23.2) on: every scalar value alone (NFC
     and NFD); every scalar between a ccc-240 and a ccc-1 mark (exactly the non-starters reorder); every
     non-starter against a representative of every combining class in both orders (the class order); every
     (composition first x composition second) pair; every hangul L x V, LV x T (T_BASE included) and LVT x T;
     every scalar after every composition first's partner set is left to the lab-host differential
     (tests/norm/); NFKC, NFKD and StripAccents (and NFKD + StripAccents, albert's chain) on every scalar
     alone, every compatibility decomposition between marks. Any disagreement stops the run;
  4. asserts the facts the C code relies on (see the header it writes) and emits src/gen/norm_nfc.{c,h} plus
     the hf golden vectors tests/norm/nfc_golden.txt that `make test` checks.

Usage:
    uv run tools/gen/norm.py            (on macOS: uv run --python 3.12 tools/gen/norm.py)
"""

import hashlib
import io
import re
import struct
import sys
import tarfile
import urllib.request
from pathlib import Path

import tokenizers
from tokenizers import normalizers

REPO = Path(__file__).resolve().parents[2]

CRATE = "unicode-normalization-alignments"
CRATE_VER = "0.1.12"
CRATE_URL = "https://static.crates.io/crates/%s/%s-%s.crate" % (CRATE, CRATE, CRATE_VER)
CRATE_SHA256 = "43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de"  # tokenizers v0.23.2 Cargo.lock

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

# info word / pool entry layout (mirrored in the emitted header)
DOFF_BITS = 13
DLEN_SHIFT = 13
NB = 1 << 16
NBK = 1 << 17            # the NB of NFKC
KX = 1 << 18             # a compatibility decomposition (toks_nfc_kinfo)
MARK = 1 << 19           # General_Category=Mark (unicode 9.0): hf's StripAccents removes it
SECOND = 1 << 21
CLS_SHIFT = 24
KLEN_SHIFT = 16          # kinfo word: pool offset | length << 16
COMP_LOG2 = 11
FIB64 = 0x9E3779B97F4A7C15
HOT_BYTE = 0xCC  # lead bytes >= 0xCC start code points >= U+0300
HOT_BYTE_K = 0xC2  # NFKC: lead bytes >= 0xC2 start code points >= U+0080

SEP = "\n"  # probe separator: a starter that is in no composition pair and no decomposition


# --------------------------------------------------------------------------------------------- crate tables
def fetch_crate():
    with urllib.request.urlopen(CRATE_URL, timeout=60) as r:
        blob = r.read()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != CRATE_SHA256:
        raise SystemExit("norm.py: %s sha256 %s != pinned %s" % (CRATE_URL, sha, CRATE_SHA256))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        member = "%s-%s/src/tables.rs" % (CRATE, CRATE_VER)
        return tf.extractfile(member).read().decode("utf-8")


def rust_block(src, name):
    """the text of `pub(crate) const NAME: ... = &[ ... ];`"""
    m = re.search(r"pub\(crate\) const %s: [^=]*= &\[\n(.*?)\n\];" % name, src, re.S)
    if not m:
        raise SystemExit("norm.py: table %s not found in tables.rs" % name)
    return m.group(1)


def uchars(s):
    return [int(h, 16) for h in re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'", s)]


def parse_tables(src):
    ver = re.search(r"UNICODE_VERSION: \(u64, u64, u64\) = \((\d+), (\d+), (\d+)\);", src)
    version = tuple(int(x) for x in ver.groups())
    if version != (9, 0, 0):
        raise SystemExit("norm.py: crate UNICODE_VERSION %r, expected (9, 0, 0)" % (version,))

    ccc = {}
    for h in re.findall(r"0x([0-9A-Fa-f]+),", rust_block(src, "CANONICAL_COMBINING_CLASS_KV")):
        v = int(h, 16)
        ccc[v >> 8] = v & 0xFF
    decomp = {}
    for line in rust_block(src, "CANONICAL_DECOMPOSED_KV").splitlines():
        m = re.match(r"\s*\(0x([0-9a-fA-F]+), &\[(.*)\]\),$", line)
        if not m:
            raise SystemExit("norm.py: bad decomposition line %r" % line)
        decomp[int(m.group(1), 16)] = uchars(m.group(2))
    comp = {}
    for line in rust_block(src, "COMPOSITION_TABLE_KV").splitlines():
        m = re.match(r"\s*\(0x([0-9A-Fa-f]+), '\\u\{([0-9A-Fa-f]+)\}'\),$", line)
        if not m:
            raise SystemExit("norm.py: bad composition line %r" % line)
        k = int(m.group(1), 16)
        comp[(k >> 16, k & 0xFFFF)] = int(m.group(2), 16)
    astral = re.search(r"fn composition_table_astral\(c1: char, c2: char\) -> Option<char> \{(.*?)\n\}", src, re.S)
    n_astral = 0
    for a, b, c in re.findall(r"\('\\u\{([0-9A-F]+)\}', '\\u\{([0-9A-F]+)\}'\) => Some\('\\u\{([0-9A-F]+)\}'\)",
                              astral.group(1)):
        comp[(int(a, 16), int(b, 16))] = int(c, 16)
        n_astral += 1
    kdecomp = {}
    for line in rust_block(src, "COMPATIBILITY_DECOMPOSED_KV").splitlines():
        m = re.match(r"\s*\(0x([0-9a-fA-F]+), &\[(.*)\]\),$", line)
        if not m:
            raise SystemExit("norm.py: bad compatibility decomposition line %r" % line)
        kdecomp[int(m.group(1), 16)] = uchars(m.group(2))
    marks = {int(h, 16) for h in re.findall(r"0x([0-9A-Fa-f]+),", rust_block(src, "COMBINING_MARK_KV"))}
    counts = (len(ccc), len(decomp), len(comp), n_astral, len(kdecomp), len(marks))
    if counts != (814, 2060, 940, 12, 3678, 2097):
        raise SystemExit("norm.py: unexpected table sizes ccc/decomp/comp/astral/kdecomp/marks %r" % (counts,))
    return ccc, decomp, comp, kdecomp, marks


# ------------------------------------------------------------------------------- the crate's algorithm
class Model:
    """exact transcription of unicode-normalization-alignments 0.1.12 nfd() / nfc() over code points."""

    def __init__(self, ccc, decomp, comp, kdecomp):
        self.ccc, self.decomp, self.comp, self.kdecomp = ccc, decomp, comp, kdecomp

    def decompose(self, c, compat=False):
        if c <= 0x7F:
            return [c]
        if S_BASE <= c < S_BASE + S_COUNT:
            s = c - S_BASE
            out = [L_BASE + s // N_COUNT, V_BASE + (s % N_COUNT) // T_COUNT]
            if s % T_COUNT:
                out.append(T_BASE + s % T_COUNT)
            return out
        if compat and c in self.kdecomp:                  # decompose_compatible: the compat table first
            return self.kdecomp[c]
        return self.decomp.get(c, [c])

    def compose(self, a, b):
        if L_BASE <= a < L_BASE + L_COUNT and V_BASE <= b < V_BASE + V_COUNT:
            return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT
        if S_BASE <= a < S_BASE + S_COUNT and T_BASE + 1 <= b < T_BASE + T_COUNT and (a - S_BASE) % T_COUNT == 0:
            return a + (b - T_BASE)
        return self.comp.get((a, b))

    def nfd(self, cps, compat=False):
        out, run = [], []
        for c in cps:
            for d in self.decompose(c, compat):
                if self.ccc.get(d, 0) == 0:
                    run.sort(key=lambda x: self.ccc.get(x, 0))  # stable
                    out.extend(run)
                    run = []
                    out.append(d)
                else:
                    run.append(d)
        run.sort(key=lambda x: self.ccc.get(x, 0))
        out.extend(run)
        return out

    def nfkd(self, cps):
        return self.nfd(cps, True)

    def nfkc(self, cps):
        return self.nfc(cps, True)

    def nfc(self, cps, compat=False):
        out, buf = [], []
        composee, last = None, None
        for ch in self.nfd(cps, compat):
            cls = self.ccc.get(ch, 0)
            if composee is None:
                if cls != 0:
                    out.append(ch)
                    continue
                composee = ch
                continue
            if last is None:
                r = self.compose(composee, ch)
                if r is not None:
                    composee = r
                    continue
                if cls == 0:
                    out.append(composee)
                    composee = ch
                    continue
                buf.append(ch)
                last = cls
            elif last >= cls:
                if cls == 0:
                    out.append(composee)
                    out.extend(buf)
                    buf = []
                    composee, last = ch, None
                    continue
                buf.append(ch)
                last = cls
            else:
                r = self.compose(composee, ch)
                if r is not None:
                    composee = r
                    continue
                buf.append(ch)
                last = cls
        if composee is not None:
            out.append(composee)
        out.extend(buf)
        return out


# ------------------------------------------------------------------------------------------- hf probes
def s_of(cps):
    return "".join(map(chr, cps))


def check(name, norm, model_fn, items, batch=20000):
    """items: lists of code points. hf over SEP-joined batches == the model per item; stops on a mismatch."""
    n = 0
    for i in range(0, len(items), batch):
        part = items[i:i + batch]
        got = norm.normalize_str(SEP.join(s_of(x) for x in part))
        want = SEP.join(s_of(model_fn(x)) for x in part)
        if got != want:
            g = got.split(SEP)
            for k, x in enumerate(part):
                w = s_of(model_fn(x))
                if k >= len(g) or g[k] != w:
                    raise SystemExit("norm.py: %s: hf %r != model %r for input %s"
                                     % (name, g[k] if k < len(g) else None, w,
                                        " ".join("U+%04X" % c for c in x)))
            raise SystemExit("norm.py: %s: batch %d differs (separator interaction?)" % (name, i))
        n += len(part)
    print("  %-34s %9d cases, 0 mismatches" % (name, n))
    return n


def u8len(c):
    return 1 if c < 0x80 else 2 if c < 0x800 else 3 if c < 0x10000 else 4


def main():
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("norm.py: tokenizers %s != 0.23.2 (the pinned oracle)" % tokenizers.__version__)
    print("oracle: tokenizers %s; data: %s %s (%s)" % (tokenizers.__version__, CRATE, CRATE_VER, CRATE_URL))
    ccc, decomp, comp, kdecomp, marks = parse_tables(fetch_crate())
    model = Model(ccc, decomp, comp, kdecomp)
    print("crate tables: unicode 9.0.0, %d non-zero ccc, %d decompositions, %d composition pairs"
          % (len(ccc), len(decomp), len(comp)))

    scalars = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    hf_nfc, hf_nfd, hf_nfkc, hf_nfkd = normalizers.NFC(), normalizers.NFD(), normalizers.NFKC(), normalizers.NFKD()
    hf_strip = normalizers.StripAccents()
    hf_kd_strip = normalizers.Sequence([normalizers.NFKD(), normalizers.StripAccents()])
    assert all(SEP not in s_of(d) for d in decomp.values()) and all(ord(SEP) not in k for k in comp)

    # ---- the sets the tables are built from
    seconds = {b for (_a, b) in comp} | set(range(V_BASE, V_BASE + V_COUNT)) | set(range(T_BASE + 1, T_BASE + T_COUNT))
    firsts = sorted({a for (a, _b) in comp})
    classes = sorted(set(ccc.values()))
    assert 0 not in classes and len(classes) <= 63, classes
    rank = {v: i + 1 for i, v in enumerate(classes)}
    rep = {}
    for c in sorted(ccc):
        rep.setdefault(ccc[c], c)
    hi = rep[max(classes)]
    lo = rep[min(classes)]
    print("classes: %d non-zero ccc values (%d..%d); %d seconds, %d firsts"
          % (len(classes), min(classes), max(classes), len(seconds), len(firsts)))

    # ---- verification through hf
    print("verifying the model against hf:")
    total = 0
    alone = [[c] for c in scalars]
    total += check("NFC, every scalar alone", hf_nfc, model.nfc, alone)
    total += check("NFD, every scalar alone", hf_nfd, model.nfd, alone)
    total += check("NFD, x hi(240) c lo(1), every c", hf_nfd, model.nfd, [[0x78, hi, c, lo] for c in scalars])
    nonstarters = sorted(ccc)
    order = [[0x78, rep[k], c] for c in nonstarters for k in classes]
    order += [[0x78, c, rep[k]] for c in nonstarters for k in classes]
    total += check("NFD, class order (both ways)", hf_nfd, model.nfd, order)
    total += check("NFC, every first x every second", hf_nfc, model.nfc,
                   [[a, b] for a in firsts for b in sorted(seconds)])
    hangul = [[L_BASE + l, V_BASE + v] for l in range(L_COUNT) for v in range(V_COUNT)]
    hangul += [[S_BASE + s, T_BASE + t] for s in range(0, S_COUNT) for t in range(T_COUNT)]
    total += check("NFC, hangul L+V, LV+T, LVT+T", hf_nfc, model.nfc, hangul)
    total += check("NFC, composition pairs + mark", hf_nfc, model.nfc,
                   [[a, b, m] for (a, b) in sorted(comp) for m in (0x0301, 0x0316, 0x0334, 0x0345)])
    total += check("NFKC, every scalar alone", hf_nfkc, model.nfkc, alone)
    total += check("NFKD, every scalar alone", hf_nfkd, model.nfkd, alone)
    kx_marks = [[0x61, c, m1, m2] for c in sorted(kdecomp) for (m1, m2) in ((0x0301, 0x0316), (0x0308, 0x0301))]
    total += check("NFKC, compat char between marks", hf_nfkc, model.nfkc, kx_marks)
    total += check("NFKD, compat char between marks", hf_nfkd, model.nfkd, kx_marks)
    total += check("StripAccents, every scalar alone", hf_strip, lambda x: [c for c in x if c not in marks], alone)
    total += check("NFKD + StripAccents, every scalar", hf_kd_strip,
                   lambda x: [c for c in model.nfkd(x) if c not in marks], alone)
    print("  total %d cases against hf, 0 mismatches" % total)

    # ---- per code point facts
    info = [0] * 0x110000
    pool, pool_at = [], {}
    max_dlen, max_ratio_num, max_ratio_cp = 0, (0, 1), 0
    n_nb = n_nbk = 0
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        cls = rank[ccc[c]] if c in ccc else 0
        w = (cls << CLS_SHIFT) | (SECOND if c in seconds else 0)
        d = decomp.get(c)
        if d is not None:
            assert len(d) >= 1 and all(x not in decomp and not S_BASE <= x < S_BASE + S_COUNT for x in d), c
            key = tuple(d)
            if key not in pool_at:
                pool_at[key] = len(pool)
                for x in d:
                    pool.append(x | (SECOND if x in seconds else 0) | ((rank[ccc[x]] if x in ccc else 0) << CLS_SHIFT))
            w |= pool_at[key] | (len(d) << DLEN_SHIFT)
            max_dlen = max(max_dlen, len(d))
        full = model.decompose(c)
        nb_bytes = sum(u8len(x) for x in full)
        if nb_bytes * max_ratio_num[1] > max_ratio_num[0] * u8len(c):
            max_ratio_num, max_ratio_cp = (nb_bytes, u8len(c)), c
        # B: a boundary before c (its first decomposed char is a starter that never composes backwards: any
        # run before it ends, nothing before it composes with it or past it) that NFC leaves unchanged alone.
        first = full[0]
        boundary = ccc.get(first, 0) == 0 and first not in seconds and model.nfc([c]) == [c]
        if not boundary:
            w |= NB
            n_nb += 1
        kfirst = model.decompose(c, True)[0]                 # the same for NFKC
        if not (ccc.get(kfirst, 0) == 0 and kfirst not in seconds and model.nfkc([c]) == [c]):
            w |= NBK
            n_nbk += 1
        w |= (KX if c in kdecomp else 0) | (MARK if c in marks else 0)
        info[c] = w
    assert len(pool) < (1 << DOFF_BITS) and max_dlen < 8

    # ---- compatibility decompositions: after the canonical ones in the pool, found through toks_nfc_kinfo
    kinfo = [0] * 0x110000
    max_klen, max_kratio, max_kratio_cp = 0, (0, 1), 0
    for c in sorted(kdecomp):
        d = kdecomp[c]
        assert d != decomp.get(c, [c]) and all(x not in kdecomp and x not in decomp for x in d), c   # full, final
        assert not S_BASE <= c < S_BASE + S_COUNT and c > 0x7F, c
        key = tuple(d)
        if key not in pool_at:
            pool_at[key] = len(pool)
            for x in d:
                pool.append(x | (SECOND if x in seconds else 0) | ((rank[ccc[x]] if x in ccc else 0) << CLS_SHIFT))
        kinfo[c] = pool_at[key] | (len(d) << KLEN_SHIFT)
        max_klen = max(max_klen, len(d))
        b = sum(u8len(x) for x in d)
        if b * max_kratio[1] > max_kratio[0] * u8len(c):
            max_kratio, max_kratio_cp = (b, u8len(c)), c
    assert len(pool) < (1 << KLEN_SHIFT)

    # facts the C code relies on
    for c in range(0x300):
        assert not info[c] & NB, "U+%04X below U+0300 needs NFC work" % c
    for c in range(0xA0):
        assert not info[c] & NBK, "U+%04X below U+00A0 needs NFKC work" % c
    assert info[0xA0] & NBK and HOT_BYTE_K == 0xC2
    assert max_dlen == 4, max_dlen
    assert max_ratio_num[0] == 3 * max_ratio_num[1], (max_ratio_num, hex(max_ratio_cp))
    assert max_klen == 18 and max_kratio[0] == 11 * max_kratio[1], (max_klen, max_kratio, hex(max_kratio_cp))
    for (a, b), c in comp.items():
        assert u8len(c) <= u8len(a) + u8len(b), (a, b, c)        # composing never lengthens the utf-8
        assert ccc.get(a, 0) == 0 and ccc.get(c, 0) == 0, (a, b, c)
        assert not S_BASE <= a < S_BASE + S_COUNT and not L_BASE <= a < L_BASE + L_COUNT, (a, b)
        full = model.decompose(c)
        assert len(full) <= 4
    for c in range(S_BASE, S_BASE + S_COUNT):
        assert not info[c] & NB and model.nfc([c]) == [c]
    # a composee absorbs at most 3 marks: each absorption adds one char to its full decomposition (<= 4)
    print("facts: every cp < U+0300 is a boundary NFC leaves unchanged (K2 hot byte 0x%02X); max full "
          "decomposition 4 chars; worst utf-8 expansion 3x (U+%04X); %d NB code points; pool %d entries"
          % (HOT_BYTE, max_ratio_cp, n_nb, len(pool)))
    print("compat: %d decompositions, max %d chars, worst utf-8 expansion 11x (U+%04X); %d NBK code points (hot "
          "byte 0x%02X); %d marks" % (len(kdecomp), max_klen, max_kratio_cp, n_nbk, HOT_BYTE_K, len(marks)))

    # ---- composition hash (u64 slots: (a << 21 | b) << 21 | composite; 0 empty; fibonacci, linear probing)
    size = 1 << COMP_LOG2
    slots = [0] * size
    maxprobe = 0
    for (a, b), c in sorted(comp.items()):
        key = (a << 21) | b
        h = ((key * FIB64) & (2 ** 64 - 1)) >> (64 - COMP_LOG2)
        for p in range(size):
            if slots[(h + p) & (size - 1)] == 0:
                slots[(h + p) & (size - 1)] = (key << 21) | c
                maxprobe = max(maxprobe, p + 1)
                break
    assert all(s < 2 ** 63 for s in slots)

    # ---- two-level table: stage1 u8 block id per 256 code points, stage2 u32 info, blocks deduplicated
    blocks, index, stage1 = [], {}, []
    for hi_ in range(0x1100):
        blk = tuple(info[hi_ * 256:(hi_ + 1) * 256])
        if blk not in index:
            index[blk] = len(blocks)
            blocks.append(blk)
        stage1.append(index[blk])
    assert stage1[0] == 0 and len(blocks) <= 256
    stage2 = [w for b in blocks for w in b]
    kblocks, kindex, kstage1 = [], {}, []
    for hi_ in range(0x1100):
        blk = tuple(kinfo[hi_ * 256:(hi_ + 1) * 256])
        if blk not in kindex:
            kindex[blk] = len(kblocks)
            kblocks.append(blk)
        kstage1.append(kindex[blk])
    assert kstage1[0] == 0 and len(kblocks) <= 256
    kstage2 = [w for b in kblocks for w in b]
    # round trip
    for c in range(0x110000):
        assert stage2[(stage1[c >> 8] << 8) | (c & 0xFF)] == info[c]
        assert kstage2[(kstage1[c >> 8] << 8) | (c & 0xFF)] == kinfo[c]
    cls_ccc = [0] + classes
    # ---- the scan's per-code-point classes (norm.c clean_run): bit (c & 63) of word (c >> 6) of form f (0 NFC, 1
    # NFKC) is set when the BMP code point c is a boundary the form leaves unchanged (NB / NBK clear); surrogates never
    fast = [0] * 2048
    for c in range(0x10000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        if not info[c] & NB:
            fast[c >> 6] |= 1 << (c & 63)
        if not info[c] & NBK:
            fast[1024 + (c >> 6)] |= 1 << (c & 63)
    assert all(fast[c >> 6] >> (c & 63) & 1 for c in range(0x300))           # bytes < 0xCC: cold for NFC
    assert all(fast[1024 + (c >> 6)] >> (c & 63) & 1 for c in range(0xA0))   # bytes < 0xC2: cold for NFKC

    data = (struct.pack("<I", len(blocks)) + bytes(stage1) + struct.pack("<%dI" % len(stage2), *stage2)
            + struct.pack("<%dI" % len(pool), *pool) + struct.pack("<%dQ" % size, *slots) + bytes(cls_ccc)
            + bytes(kstage1) + struct.pack("<%dI" % len(kstage2), *kstage2) + struct.pack("<2048Q", *fast))
    sha = hashlib.sha256(data).hexdigest()
    print("tables: %d blocks (stage1 %d B, stage2 %d B), pool %d B, comp %d B (max probe %d); data sha256 %s"
          % (len(blocks), len(stage1), 4 * len(stage2), 4 * len(pool), 8 * size, maxprobe, sha))

    p = dict(sha=sha, n_blocks=len(blocks), pool_len=len(pool), comp_log2=COMP_LOG2, maxprobe=maxprobe,
             n_cls=len(classes), n_nb=n_nb, n_pairs=len(comp), n_dec=len(decomp), n_ccc=len(ccc),
             crate=CRATE, ver=CRATE_VER, crate_sha=CRATE_SHA256, hot=HOT_BYTE, ratio_cp=max_ratio_cp,
             hotk=HOT_BYTE_K, n_kdec=len(kdecomp), n_nbk=n_nbk, n_marks=len(marks), kratio_cp=max_kratio_cp,
             n_kblocks=len(kblocks))
    h_text = H_TEXT % p
    c_text = C_TEXT % dict(p, stage1=fmt(stage1, "%d", 24), stage2=fmt(stage2, "0x%08X", 8),
                           pool=fmt(pool, "0x%08X", 8), comp=fmt(slots, "0x%016XULL", 4),
                           cls_ccc=fmt(cls_ccc, "%d", 24), kstage1=fmt(kstage1, "%d", 24),
                           kstage2=fmt(kstage2, "0x%08X", 8), fast=fmt(fast, "0x%016XULL", 4))
    out = {REPO / "src/gen/norm_nfc.h": h_text, REPO / "src/gen/norm_nfc.c": c_text,
           REPO / "tests/norm/nfc_golden.txt": golden(model, hf_nfc, comp, ccc, classes, rep, scalars),
           REPO / "tests/norm/nfkc_golden.txt": kgolden(model, hf_nfkc, hf_nfkd, kdecomp)}
    for path, text in out.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote %s (%d bytes) sha256 %s" % (path.relative_to(REPO), path.stat().st_size,
                                                 hashlib.sha256(path.read_bytes()).hexdigest()))


def fmt(vals, f, per):
    return "\n".join("    " + ", ".join(f % v for v in vals[i:i + per]) + "," for i in range(0, len(vals), per))


def golden(model, hf_nfc, comp, ccc, classes, rep, scalars):
    """hf NFC vectors for `make test`: every scalar NFC changes alone, every composition pair, the class order
    between representatives, hangul, and composition chains. one case per line: hex(in) hex(out)."""
    cases = [[c] for c in scalars if model.nfc([c]) != [c]]
    cases += [[a, b] for (a, b) in sorted(comp)]
    for k1 in classes:
        for k2 in classes:
            cases.append([0x61, rep[k1], rep[k2]])
    cases += [[L_BASE + l, V_BASE + v, T_BASE + t] for l in (0, 18) for v in (0, 20) for t in (0, 1, 27)]
    cases += [[S_BASE, T_BASE + 1], [S_BASE + 1, T_BASE + 1], [S_BASE, T_BASE], [S_BASE + 27, T_BASE + 2]]
    cases += [[0x3B1, 0x313, 0x300, 0x345], [0x3B1, 0x345, 0x300, 0x313], [0x41, 0x30A, 0x301],
              [0x61, 0x323, 0x302], [0x61, 0x302, 0x323], [0x1E08, 0x323], [0x65, 0x301, 0x323]]
    lines = []
    for x in cases:
        s = s_of(x)
        got = hf_nfc.normalize_str(s)
        assert got == s_of(model.nfc(x)), x
        lines.append("%s %s" % (s.encode("utf-8").hex(), got.encode("utf-8").hex()))
    head = ("# hf tokenizers 0.23.2 normalizers.NFC vectors (generated by tools/gen/norm.py, %d cases):\n"
            "# hex(utf-8 input) hex(utf-8 NFC output)\n" % len(lines))
    return head + "\n".join(lines) + "\n"


def kgolden(model, hf_nfkc, hf_nfkd, kdecomp):
    """hf NFKC / NFKD vectors for `make test`: every compatibility decomposition alone and after 'a' between two
    marks, and a few chains. one case per line: hex(in) hex(NFKC out) hex(NFKD out)."""
    cases = [[c] for c in sorted(kdecomp)] + [[0x61, c, 0x0301, 0x0316] for c in sorted(kdecomp)]
    cases += [[0x1E9B, 0x0323], [0x0385, 0x0316], [0xFB01, 0x0308], [0x1100, 0xFFC2], [0x212B, 0x0301], [0xAC01]]
    lines = []
    for x in cases:
        s = s_of(x)
        k, d = hf_nfkc.normalize_str(s), hf_nfkd.normalize_str(s)
        assert k == s_of(model.nfkc(x)) and d == s_of(model.nfkd(x)), x
        lines.append("%s %s %s" % (s.encode("utf-8").hex(), k.encode("utf-8").hex(), d.encode("utf-8").hex()))
    head = ("# hf tokenizers 0.23.2 normalizers.NFKC / NFKD vectors (generated by tools/gen/norm.py, %d cases):\n"
            "# hex(utf-8 input) hex(utf-8 NFKC output) hex(utf-8 NFKD output)\n" % len(lines))
    return head + "\n".join(lines) + "\n"


H_TEXT = """/* toks: NFC / NFKC data exactly as hf tokenizers 0.23.2 runs it: crate %(crate)s %(ver)s
 * (sha256 %(crate_sha)s, pinned by tokenizers v0.23.2's Cargo.lock), whose tables.rs is UNICODE 9.0.0.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with: uv run tools/gen/norm.py
 * The generator parses the crate's own tables and verifies its model of the crate against hf's NFC and NFD on
 * every scalar alone, every scalar between two marks, the class order, every composition first x second,
 * all of hangul (docs: src/core/norm.h).
 *
 * Facts asserted by the generator (the C code relies on them):
 *   - every code point below U+0300 is a boundary NFC leaves unchanged: a byte < 0x%(hot)02X never needs work
 *   - a full canonical decomposition has <= 4 chars; its utf-8 is <= 3x the char's (U+%(ratio_cp)04X is 3x)
 *   - composing never lengthens the utf-8 (len(c) <= len(a) + len(b) for every pair); firsts are starters
 *   - hangul syllables are boundaries; L+V and LV+T compose by arithmetic (no table entries)
 *   - NFKC: every code point below U+00A0 is a boundary it leaves unchanged (a byte < 0x%(hotk)02X never needs
 *     work); a full compatibility decomposition has <= 18 chars, its utf-8 <= 11x the char's (U+%(kratio_cp)04X)
 * counts: %(n_ccc)d non-starters in %(n_cls)d classes, %(n_dec)d decompositions, %(n_pairs)d pairs, %(n_nb)d NB code points;
 *   %(n_kdec)d compatibility decompositions, %(n_nbk)d NBK code points, %(n_marks)d marks
 * table data sha256 (n_blocks u32 || stage1 || stage2 || pool || comp || cls_ccc || kstage1 || kstage2 || fast, LE):
 *   %(sha)s
 */
#ifndef TOKS_NORM_NFC_H
#define TOKS_NORM_NFC_H

#include <stdint.h>

/* info word of a code point, toks_nfc_info(cp):
 *   bits  0..12  offset of its full canonical decomposition in toks_nfc_pool (when dlen > 0)
 *   bits 13..15  dlen: decomposition length 1..4, or 0 when the char is its own decomposition
 *   bit  16      NB: not a boundary NFC leaves unchanged (a char with NB clear has a starter first decomposed
 *                char that composes with nothing before it, and NFC maps it to itself)
 *   bit  17      NBK: the same for NFKC
 *   bit  18      KX: the char has a compatibility decomposition (toks_nfc_kinfo: offset | length << 16 in the pool)
 *   bit  19      MARK: General_Category=Mark (hf's StripAccents removes it)
 *   bit  21      SECOND: the char may compose with a preceding char (the second of a pair, hangul V or T)
 *   bits 24..29  cls: dense rank of its canonical combining class (0 = starter; ccc = toks_nfc_cls_ccc[cls])
 * pool entry: bits 0..20 the code point, bit 21 SECOND, bits 24..29 cls (as above). */
#define TOKS_NFC_DOFF_MASK    0x1FFFu
#define TOKS_NFC_DLEN_SHIFT   13
#define TOKS_NFC_DLEN_MASK    0x7u
#define TOKS_NFC_NB           0x00010000u
#define TOKS_NFC_NBK          0x00020000u
#define TOKS_NFC_KX           0x00040000u
#define TOKS_NFC_MARK         0x00080000u
#define TOKS_NFC_SECOND       0x00200000u
#define TOKS_NFC_CLS_SHIFT    24
#define TOKS_NFC_CLS_MASK     0x3Fu
#define TOKS_NFC_CP_MASK      0x001FFFFFu

#define TOKS_NFC_HOT_BYTE     0x%(hot)02Xu  /* the K2 predicate: no byte >= this, no NFC work */
#define TOKS_NFKC_HOT_BYTE    0x%(hotk)02Xu  /* ... no NFKC work */
#define TOKS_NFC_MAX_DECOMP   4             /* chars in a full canonical decomposition */
#define TOKS_NFC_MAX_KDECOMP  18            /* chars in a full compatibility decomposition */
#define TOKS_NFC_EXPANSION    3             /* utf-8 bytes out per byte in, worst case (proof: norm.h) */
#define TOKS_NFKC_EXPANSION   11            /* the same with compatibility decompositions */
#define TOKS_NFC_N_CLS        %(n_cls)d            /* non-zero combining classes (cls 1..N_CLS) */

#define TOKS_NFC_N_BLOCKS     %(n_blocks)d
#define TOKS_NFC_N_KBLOCKS    %(n_kblocks)d
#define TOKS_NFC_POOL_LEN     %(pool_len)d
#define TOKS_NFC_COMP_LOG2    %(comp_log2)d
#define TOKS_NFC_COMP_MAXPROBE %(maxprobe)d          /* longest linear probe of any present pair */
#define TOKS_NFC_FIB64        0x9E3779B97F4A7C15ull
#define TOKS_NFC_DATA_SHA256  "%(sha)s"

extern const uint8_t  toks_nfc_stage1[0x1100];
extern const uint32_t toks_nfc_stage2[TOKS_NFC_N_BLOCKS * 256];
extern const uint32_t toks_nfc_pool[TOKS_NFC_POOL_LEN];
/* composition pairs: slot = (a << 21 | b) << 21 | composite, 0 empty; home slot
 * ((a << 21 | b) * TOKS_NFC_FIB64) >> (64 - TOKS_NFC_COMP_LOG2), linear probing */
extern const uint64_t toks_nfc_comp[1u << TOKS_NFC_COMP_LOG2];
extern const uint8_t  toks_nfc_cls_ccc[TOKS_NFC_N_CLS + 1];
extern const uint8_t  toks_nfc_kstage1[0x1100];
extern const uint32_t toks_nfc_kstage2[TOKS_NFC_N_KBLOCKS * 256];
/* the scan's per-code-point classes: bit (cp & 63) of toks_nfc_fast[f * 1024 + (cp >> 6)] is set when the BMP
 * code point cp is a boundary the form f (0 NFC, 1 NFKC) leaves unchanged (NB / NBK clear); never a surrogate */
extern const uint64_t toks_nfc_fast[2048];

/* info word of a scalar value (cp <= U+10FFFF) */
static inline uint32_t toks_nfc_info(uint32_t cp)
{
    return toks_nfc_stage2[((uint32_t)toks_nfc_stage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

/* its full compatibility decomposition: pool offset | length << 16, 0 when it has none (KX clear) */
static inline uint32_t toks_nfc_kinfo(uint32_t cp)
{
    return toks_nfc_kstage2[((uint32_t)toks_nfc_kstage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_NORM_NFC_H */
"""

C_TEXT = """/* toks: NFC tables of %(crate)s %(ver)s (unicode 9.0.0), the data hf tokenizers 0.23.2 runs.
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with: uv run tools/gen/norm.py
 * table data sha256: %(sha)s
 */

#include "norm_nfc.h"

/* stage 1: block id per 256 code points; block 0 is U+0000..U+00FF */
const uint8_t toks_nfc_stage1[0x1100] = {
%(stage1)s
};

/* stage 2: %(n_blocks)d deduplicated blocks of 256 info words */
const uint32_t toks_nfc_stage2[TOKS_NFC_N_BLOCKS * 256] = {
%(stage2)s
};

/* full canonical decompositions, then the compatibility ones, deduplicated */
const uint32_t toks_nfc_pool[TOKS_NFC_POOL_LEN] = {
%(pool)s
};

const uint64_t toks_nfc_comp[1u << TOKS_NFC_COMP_LOG2] = {
%(comp)s
};

/* cls -> canonical combining class */
const uint8_t toks_nfc_cls_ccc[TOKS_NFC_N_CLS + 1] = {
%(cls_ccc)s
};

/* compatibility decompositions: block id per 256 code points, then %(n_kblocks)d blocks of kinfo words */
const uint8_t toks_nfc_kstage1[0x1100] = {
%(kstage1)s
};

const uint32_t toks_nfc_kstage2[TOKS_NFC_N_KBLOCKS * 256] = {
%(kstage2)s
};

/* the scan's per-code-point classes, NFC then NFKC (norm_nfc.h) */
const uint64_t toks_nfc_fast[2048] = {
%(fast)s
};
"""

if __name__ == "__main__":
    main()
