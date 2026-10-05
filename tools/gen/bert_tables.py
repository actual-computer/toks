# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""toks BERT normalizer / pre-tokenizer unicode tables, derived from the exact sources hf tokenizers 0.23.2 runs and
verified through hf itself on every unicode scalar value.

hf's BertNormalizer and BertPreTokenizer (tokenizers 0.23.2, bindings/python/Cargo.lock) read four sources:

  unicode_categories 0.1.1   is_other (Cc | Cf | Co: clean_text's removal), is_punctuation (Pc Pd Pe Pf Pi Po Ps:
                             BertPreTokenizer's split), is_mark_nonspacing (Mn: strip_accents' filter). Its tables
                             are per-char lists generated from UnicodeData.txt: they equal Unicode 8.0.0 exactly.
  unicode-normalization-alignments 0.1.12
                             NFD (strip_accents): canonical decompositions + combining classes, UNICODE_VERSION 9.0.0.
  rust std (the wheel's compiler)
                             char::to_lowercase (lowercase) and char::is_whitespace (clean_text's map, the
                             pre-tokenizer's split): Unicode 17.0.0 (simple lowercase + SpecialCasing's
                             unconditional mappings; White_Space from PropList).
  hard-coded                 is_chinese_char's eight ranges; char::is_ascii_punctuation.

This script
  1. downloads both crates from crates.io (sha256 = the Cargo.lock checksums) and the three UCD 17.0.0 files from
     unicode.org (sha256 pinned below) and parses the tables;
  2. probes hf (tokenizers==0.23.2) over EVERY scalar value for each normalizer step alone (clean_text,
     handle_chinese_chars, strip_accents, lowercase), the full cased and uncased BertNormalizer, the NFD normalizer,
     and the classes of BertPreTokenizer, WhitespaceSplit and Whitespace; plus every ordered pair of the 83
     non-starters that survive strip_accents (canonical reordering), with and without a removed char or a ccc-0
     mark between them. Any disagreement with the derived tables stops the run;
  3. writes tests/data/wordpiece/unicode.json (the python model's data) and src/gen/bert_tables.{c,h} (the c twin's
     two-level tables), deterministic, and prints their sha256.

Usage:
    uv run tools/gen/bert_tables.py            # on macOS: uv run --python 3.12 ... works too
"""

import collections
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from tokenizers import normalizers, pre_tokenizers

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "build" / "gen-cache"

CRATES = {
    "unicode_categories": ("0.1.1", "39ec24b3121d976906ece63c9daad25b85969647682eee313cb5779fdd69e14e"),
    "unicode-normalization-alignments": ("0.1.12", "43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de"),
}
UCD = {
    "UnicodeData.txt": "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c",
    "SpecialCasing.txt": "efc25faf19de21b92c1194c111c932e03d2a5eaf18194e33f1156e96de4c9588",
    "PropList.txt": "130dcddcaadaf071008bdfce1e7743e04fdfbc910886f017d9f9ac931d8c64dd",
}
UCD_VER = "17.0.0"

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT

CJK_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
              (0x2B920, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]
ASCII_PUNCT = [ord(c) for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"]

# class bits of the c table (mirrored in the emitted header)
BC_REMOVE, BC_WS, BC_CJK, BC_PUNCT, BC_DECOMP, BC_LOWER, BC_NS, BC_MN = 1, 2, 4, 8, 16, 32, 64, 128

SCALARS = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]


def die(msg):
    raise SystemExit("bert_tables.py: " + msg)


def fetch(url, sha, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha:
        return p.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "toks-bert-tables"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    got = hashlib.sha256(blob).hexdigest()
    if got != sha:
        die("%s sha256 %s != pinned %s" % (url, got, sha))
    p.write_bytes(blob)
    return blob


def crate_file(name, member):
    ver, sha = CRATES[name]
    blob = fetch("https://static.crates.io/crates/%s/%s-%s.crate" % (name, name, ver), sha, "%s-%s.crate" % (name, ver))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        return tf.extractfile("%s-%s/%s" % (name, ver, member)).read().decode("utf-8")


def ucd_file(name):
    return fetch("https://www.unicode.org/Public/%s/ucd/%s" % (UCD_VER, name), UCD[name],
                 "ucd%s-%s" % (UCD_VER, name)).decode("utf-8")


# ------------------------------------------------------------------------------------------------ the sources
def load_sources():
    uc = crate_file("unicode_categories", "src/tables.rs")
    lib = crate_file("unicode_categories", "src/lib.rs")

    def clist(name):
        m = re.search(r"pub static %s : &'static \[char\] = &\[(.*?)\];" % name, uc, re.S)
        if not m:
            die("unicode_categories table %s not found" % name)
        return [int(h, 16) for h in re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'", m.group(1))]

    # is_other_private_use: three explicit ranges, then the table (its six entries are the range ends)
    if "'\\u{E000}'...'\\u{F8FF}' => true" not in lib or "'\\u{100000}'...'\\u{10FFFD}' => true" not in lib:
        die("unicode_categories is_other_private_use is not the expected match")
    src = {
        "Cc": set(clist("OTHER_CONTROL")),
        "Cf": set(clist("OTHER_FORMAT")),
        "Co": set(clist("OTHER_PRIVATE_USE")) | set(range(0xE000, 0xF900)) | set(range(0xF0000, 0xFFFFE))
        | set(range(0x100000, 0x10FFFE)),
        "Mn": set(clist("MARK_NONSPACING")),
        "P": set(),
    }
    for n in ("PUNCTUATION_CONNECTOR", "PUNCTUATION_DASH", "PUNCTUATION_CLOSE", "PUNCTUATION_FINAL_QUOTE",
              "PUNCTUATION_INITIAL_QUOTE", "PUNCTUATION_OTHER", "PUNCTUATION_OPEN"):
        src["P"] |= set(clist(n))

    nt = crate_file("unicode-normalization-alignments", "src/tables.rs")
    ver = re.search(r"UNICODE_VERSION: \(u64, u64, u64\) = \((\d+), (\d+), (\d+)\);", nt)
    if tuple(int(x) for x in ver.groups()) != (9, 0, 0):
        die("unicode-normalization-alignments UNICODE_VERSION %r" % (ver.groups(),))

    def rust_block(name):
        m = re.search(r"pub\(crate\) const %s: [^=]*= &\[\n(.*?)\n\];" % name, nt, re.S)
        if not m:
            die("normalization table %s not found" % name)
        return m.group(1)

    ccc = {}
    for h in re.findall(r"0x([0-9A-Fa-f]+),", rust_block("CANONICAL_COMBINING_CLASS_KV")):
        v = int(h, 16)
        ccc[v >> 8] = v & 0xFF
    dec = {}
    for line in rust_block("CANONICAL_DECOMPOSED_KV").splitlines():
        m = re.match(r"\s*\(0x([0-9a-fA-F]+), &\[(.*)\]\),$", line)
        if not m:
            die("bad decomposition line %r" % line)
        dec[int(m.group(1), 16)] = [int(h, 16) for h in re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'", m.group(2))]
    if (len(ccc), len(dec)) != (814, 2060):
        die("unexpected normalization table sizes %r" % ((len(ccc), len(dec)),))
    src["ccc"] = ccc
    src["dec"] = dec
    # is_combining_mark (General_Category=Mark, 9.0): the standalone StripAccents normalizer's filter
    m = re.search(r"pub\(crate\) const COMBINING_MARK_KV: &\[u32\] = &\[(.*?)\];", nt, re.S)
    if not m:
        die("COMBINING_MARK_KV not found")
    src["M9"] = set(int(h, 16) for h in re.findall(r"0x([0-9A-Fa-f]+)", m.group(1)))

    simple = {}
    for line in ucd_file("UnicodeData.txt").splitlines():
        f = line.split(";")
        if f[13]:
            simple[int(f[0], 16)] = [int(f[13], 16)]
    special = {}
    for line in ucd_file("SpecialCasing.txt").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        f = [x.strip() for x in line.split(";")]
        if len(f) >= 5 and f[4]:
            continue                                   # conditional mappings: rust's char::to_lowercase skips them
        special[int(f[0], 16)] = [int(x, 16) for x in f[1].split()]
    lower = {}
    for c in SCALARS:
        m = special.get(c, simple.get(c, [c]))
        if m != [c]:
            lower[c] = m
    ws = set()
    for line in ucd_file("PropList.txt").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        r, prop = [x.strip() for x in line.split(";")]
        if prop == "White_Space":
            a, _, b = r.partition("..")
            ws |= set(range(int(a, 16), int(b or a, 16) + 1))
    src["lower"] = lower
    src["ws"] = ws
    return src


# ------------------------------------------------------------------------------------------- the derived model
class Derived:
    def __init__(self, s):
        self.s = s
        self.cjk = set()
        for a, b in CJK_RANGES:
            self.cjk |= set(range(a, b + 1))
        self.punct = set(ASCII_PUNCT) | s["P"]
        self.other = s["Cc"] | s["Cf"] | s["Co"]
        self.remove = ({0, 0xFFFD} | self.other) - {9, 10, 13}
        self.wsb = s["ws"] | {9, 10, 13}            # bert's is_whitespace (\t \n \r are White_Space anyway)

    def decompose(self, c):
        if S_BASE <= c < S_BASE + S_COUNT:
            k = c - S_BASE
            out = [L_BASE + k // N_COUNT, V_BASE + (k % N_COUNT) // T_COUNT]
            if k % T_COUNT:
                out.append(T_BASE + k % T_COUNT)
            return out
        if c in self.s["dec"]:
            out = []
            for x in self.s["dec"][c]:
                out.extend(self.decompose(x))
            return out
        return [c]

    def nfd(self, cps):
        out = []
        for c in cps:
            out.extend(self.decompose(c))
        res, i, ccc = [], 0, self.s["ccc"]
        while i < len(out):                           # stable sort of every maximal non-starter run
            if ccc.get(out[i], 0) == 0:
                res.append(out[i])
                i += 1
                continue
            j = i
            while j < len(out) and ccc.get(out[j], 0) != 0:
                j += 1
            res.extend(sorted(out[i:j], key=lambda x: ccc.get(x, 0)))
            i = j
        return res

    def clean(self, cps):
        return [32 if c in self.wsb else c for c in cps if c not in self.remove]

    def chinese(self, cps):
        out = []
        for c in cps:
            out.extend([32, c, 32] if c in self.cjk else [c])
        return out

    def strip(self, cps):
        return [c for c in self.nfd(cps) if c not in self.s["Mn"]]

    def lower(self, cps):
        out = []
        for c in cps:
            out.extend(self.s["lower"].get(c, [c]))
        return out

    def bert(self, cps, clean, chinese, strip, lower):
        if clean:
            cps = self.clean(cps)
        if chinese:
            cps = self.chinese(cps)
        if strip:
            cps = self.strip(cps)
        if lower:
            cps = self.lower(cps)
        return cps


# -------------------------------------------------------------------------------------------------- the probes
def s_of(cps):
    return "".join(map(chr, cps))


def probe_norm(norm, sep, cps, model, label):
    """every cp in cps through hf's normalizer, batched with a separator the normalizer leaves alone."""
    bad = []
    sepc = ord(sep)
    if sepc in cps:
        if norm.normalize_str(sep) != s_of(model([sepc])):
            bad.append(sepc)
        cps = [c for c in cps if c != sepc]
    for i in range(0, len(cps), 0x4000):
        b = cps[i:i + 0x4000]
        got = norm.normalize_str(sep.join(chr(c) for c in b)).split(sep)
        if len(got) != len(b):
            die("%s: separator U+%04X did not survive" % (label, sepc))
        for c, g in zip(b, got):
            if g != s_of(model([c])):
                bad.append(c)
    if bad:
        die("%s: %d mismatches, first %s" % (label, len(bad), ["U+%04X" % c for c in bad[:10]]))
    print("  %-34s 0 mismatches over %d scalars" % (label, len(cps) + 1), file=sys.stderr)


def probe_classes(pt, label):
    """class of every scalar under a pre-tokenizer: 'w' removed (split on), 'p' isolated, 'o' kept in the word."""
    cls = {}
    cps = [c for c in SCALARS if c != ord("Z")]
    for i in range(0, len(cps), 0x4000):
        b = cps[i:i + 0x4000]
        pieces = [p for p, _ in pt.pre_tokenize_str(" ".join("Z" + chr(c) + "Z" for c in b))]
        k = 0
        for c in b:
            ch = chr(c)
            if pieces[k] == "Z" + ch + "Z":
                cls[c] = "o"
                k += 1
            elif pieces[k] == "Z" and pieces[k + 1] == ch and pieces[k + 2] == "Z":
                cls[c] = "p"
                k += 3
            elif pieces[k] == "Z" and pieces[k + 1] == "Z":
                cls[c] = "w"
                k += 2
            else:
                die("%s: unparsed U+%04X %r" % (label, c, pieces[k:k + 4]))
        if k != len(pieces):
            die("%s: trailing pieces" % label)
    cls[ord("Z")] = "o"
    return cls


def ranges(cps):
    out = []
    for c in sorted(cps):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def main():
    t0 = time.time()
    src = load_sources()
    D = Derived(src)
    print("sources: Cc %d Cf %d Co %d Mn %d P %d | ccc %d dec %d M9 %d | lower %d ws %d" % (
        len(src["Cc"]), len(src["Cf"]), len(src["Co"]), len(src["Mn"]), len(src["P"]), len(src["ccc"]),
        len(src["dec"]), len(src["M9"]), len(src["lower"]), len(src["ws"])), file=sys.stderr)

    BN = normalizers.BertNormalizer
    sc = list(SCALARS)
    probe_norm(BN(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False), "\u4e00", sc,
               lambda x: D.bert(x, 1, 0, 0, 0), "BertNormalizer clean_text")
    probe_norm(BN(clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=False), "\x00", sc,
               lambda x: D.bert(x, 0, 1, 0, 0), "BertNormalizer handle_chinese_chars")
    probe_norm(BN(clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False), "\x00", sc,
               lambda x: D.bert(x, 0, 0, 1, 0), "BertNormalizer strip_accents")
    probe_norm(BN(clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=True), "\x00", sc,
               lambda x: D.bert(x, 0, 0, 0, 1), "BertNormalizer lowercase")
    probe_norm(BN(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True), "\u3007", sc,
               lambda x: D.bert(x, 1, 1, 1, 1), "BertNormalizer uncased (defaults)")
    probe_norm(BN(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=False), "\u3007", sc,
               lambda x: D.bert(x, 1, 1, 0, 0), "BertNormalizer cased")
    probe_norm(BN(clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=True), "\x00", sc,
               lambda x: D.bert(x, 0, 0, 1, 1), "BertNormalizer strip+lower")
    probe_norm(normalizers.NFD(), "\x00", sc, D.nfd, "NFD")
    probe_norm(normalizers.Lowercase(), "\x00", sc, D.lower, "Lowercase")
    probe_norm(normalizers.StripAccents(), "\x00", sc, lambda x: [c for c in x if c not in src["M9"]], "StripAccents")

    # canonical reordering of the non-starters strip_accents keeps, across chars and removed chars
    keep_ns = sorted(c for c, v in src["ccc"].items() if v and c not in src["Mn"])
    for c in SCALARS:                    # the scanner relies on it: no non-starter run crosses a piece boundary
        if src["ccc"].get(c, 0) and (c in src["ws"] or c in D.punct or c in D.cjk or c in D.remove):
            die("U+%04X has a combining class and is a boundary / removed char" % c)
    mn_ns = sorted(c for c, v in src["ccc"].items() if v and c in src["Mn"])[::37]
    mn_0 = sorted(c for c in src["Mn"] if not src["ccc"].get(c))[::97]
    strip = BN(clean_text=True, handle_chinese_chars=False, strip_accents=True, lowercase=False)
    cases = []
    for x in keep_ns:
        for y in keep_ns:
            cases.append([0x61, x, y])
            cases.append([0x61, x, 0x200B, y])          # a removed char: the run continues across it
            for z in mn_ns[:3]:
                cases.append([0x61, x, z, y])           # a stripped non-starter inside the run
            for z in mn_0[:2]:
                cases.append([0x61, x, z, y])           # a stripped STARTER (ccc 0): two runs
    sep = "\u3007"
    for i in range(0, len(cases), 4000):
        b = cases[i:i + 4000]
        got = strip.normalize_str(sep.join(s_of(c) for c in b)).split(sep)
        for c, g in zip(b, got):
            if g != s_of(D.bert(c, 1, 0, 1, 0)):
                die("reorder mismatch on %s" % ["U+%04X" % x for x in c])
    print("  %-34s 0 mismatches over %d sequences" % ("strip_accents reordering", len(cases)), file=sys.stderr)

    bert_pre = probe_classes(pre_tokenizers.BertPreTokenizer(), "BertPreTokenizer")
    for c in SCALARS:
        want = "w" if c in src["ws"] else "p" if c in D.punct else "o"
        if bert_pre[c] != want:
            die("BertPreTokenizer class of U+%04X: hf %s, derived %s" % (c, bert_pre[c], want))
    print("  %-34s 0 mismatches over %d scalars" % ("BertPreTokenizer classes", len(SCALARS)), file=sys.stderr)
    wsplit = probe_classes(pre_tokenizers.WhitespaceSplit(), "WhitespaceSplit")
    for c in SCALARS:
        if (wsplit[c] == "w") != (c in src["ws"]) or wsplit[c] == "p":
            die("WhitespaceSplit class of U+%04X" % c)
    print("  %-34s 0 mismatches over %d scalars" % ("WhitespaceSplit classes", len(SCALARS)), file=sys.stderr)
    wre = probe_classes(pre_tokenizers.Whitespace(), "Whitespace")
    word = {c for c in SCALARS if wre[c] == "o"}
    for c in SCALARS:
        if (wre[c] == "w") != (c in src["ws"]):
            die("Whitespace \\s class of U+%04X" % c)
    print("  %-34s probed (\\w = %d scalars, \\s = White_Space)" % ("Whitespace classes", len(word)), file=sys.stderr)

    # -------------------------------------------------------------------------------------------- outputs
    keep_ccc = {c: src["ccc"][c] for c in keep_ns}
    data = {
        "_comment": "BERT normalizer / pre-tokenizer unicode data of hf tokenizers 0.23.2, written by "
                    "tools/gen/bert_tables.py and verified against hf on every scalar value. Ranges are inclusive.",
        "sources": {
            "unicode_categories": "%s sha256 %s (Unicode 8.0.0 tables)" % CRATES["unicode_categories"],
            "unicode-normalization-alignments": "%s sha256 %s (Unicode 9.0.0)" % CRATES["unicode-normalization-alignments"],
            "ucd": {k: "%s %s" % (UCD_VER, v) for k, v in UCD.items()},
        },
        "remove": ranges(D.remove),
        "whitespace": ranges(src["ws"]),
        "cjk": [list(r) for r in CJK_RANGES],
        "punct": ranges(D.punct),
        "mn": ranges(src["Mn"]),
        "mark9": ranges(src["M9"]),
        "ccc": sorted([c, v] for c, v in src["ccc"].items()),
        "decomp": sorted([c, d] for c, d in src["dec"].items()),
        "lower": sorted([c, m] for c, m in src["lower"].items()),
        "word": ranges(word),
    }
    jpath = REPO / "tests" / "data" / "wordpiece" / "unicode.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jtext = json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
    jpath.write_text(jtext)

    # ---- c tables
    cls = [0] * 0x110000
    full_dec = {}
    for c in SCALARS:
        v = 0
        if c in D.remove:
            v |= BC_REMOVE
        if c in D.wsb:
            v |= BC_WS
        if c in D.cjk:
            v |= BC_CJK
        if c in D.punct:
            v |= BC_PUNCT
        d = D.decompose(c)
        if d != [c]:
            v |= BC_DECOMP
            if not (S_BASE <= c < S_BASE + S_COUNT):
                full_dec[c] = d
        if c in src["lower"]:
            v |= BC_LOWER
        if src["ccc"].get(c, 0):
            v |= BC_NS
        if c in src["Mn"]:
            v |= BC_MN
        cls[c] = v
    max_dec = max(len(d) for d in full_dec.values())
    max_low = max(len(m) for m in src["lower"].values())
    # map entries: every cp with a (non-hangul) decomposition or a lowercase mapping
    mapped = sorted(set(full_dec) | set(src["lower"]))
    pool, entries = [], [(0, 0, 0, 0)]               # entry 0 = none
    for c in mapped:
        d = full_dec.get(c, [])
        lo = src["lower"].get(c, [])
        doff = len(pool)
        pool.extend(d)
        loff = len(pool)
        pool.extend(lo)
        entries.append((doff, len(d), loff, len(lo)))
    if len(entries) >= 0x10000 or len(pool) >= 0x10000:
        die("map index overflow")
    mapidx = [0] * 0x110000
    for i, c in enumerate(mapped):
        mapidx[c] = i + 1

    def two_level(vals, width):
        blocks, index, stage1 = [], {}, []
        for hi in range(0x1100):
            blk = tuple(vals[hi << 8:(hi + 1) << 8])
            if blk not in index:
                index[blk] = len(blocks)
                blocks.append(blk)
            stage1.append(index[blk])
        return stage1, blocks

    c_s1, c_blocks = two_level(cls, 8)
    m_s1, m_blocks = two_level(mapidx, 16)

    # worst-case growth of one input char (utf-8 bytes out / in) over every flag combination
    def u8len(cps):
        return sum(len(chr(x).encode("utf-8")) for x in cps)
    growth_num, growth_den, growth_cp = 1, 1, 0x41      # ascii never grows
    for c in SCALARS:
        if not (cls[c] & (BC_CJK | BC_DECOMP | BC_LOWER)):
            continue
        n_in = len(chr(c).encode("utf-8"))
        for f in range(16):
            o = u8len(D.bert([c], f & 1, f & 2, f & 4, f & 8))
            if o * growth_den > growth_num * n_in:
                growth_num, growth_den, growth_cp = o, n_in, c
    print("worst growth: U+%04X %d -> %d bytes" % (growth_cp, growth_den, growth_num), file=sys.stderr)

    blob = bytearray()
    blob += len(c_blocks).to_bytes(4, "little") + len(m_blocks).to_bytes(4, "little")
    for v in c_s1:
        blob += v.to_bytes(2, "little")
    for b in c_blocks:
        blob += bytes(b)
    for v in m_s1:
        blob += v.to_bytes(2, "little")
    for b in m_blocks:
        for v in b:
            blob += v.to_bytes(2, "little")
    for e in entries:
        blob += e[0].to_bytes(2, "little") + bytes([e[1]]) + bytes([0]) + e[2].to_bytes(2, "little") + bytes([e[3]]) + bytes([0])
    for v in pool:
        blob += v.to_bytes(4, "little")
    for c in keep_ns:
        blob += c.to_bytes(4, "little") + keep_ccc[c].to_bytes(4, "little")
    dsha = hashlib.sha256(bytes(blob)).hexdigest()

    def rows(vals, per, fmt):
        out = []
        for i in range(0, len(vals), per):
            out.append("    " + ", ".join(fmt % v for v in vals[i:i + per]) + ",")
        return "\n".join(out)

    h = """/* toks: the unicode tables of hf tokenizers 0.23.2's BertNormalizer and BertPreTokenizer.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:  uv run tools/gen/bert_tables.py
 * Sources (pinned by sha256 in the generator): unicode_categories 0.1.1 (Unicode 8.0.0: Cc Cf Co, P*, Mn),
 * unicode-normalization-alignments 0.1.12 (Unicode 9.0.0 NFD), rust std's char::to_lowercase / is_whitespace
 * (Unicode %(ucd)s), is_chinese_char's ranges. Verified against hf on every scalar value (docs/algorithms/wordpiece.md).
 *
 * class byte of a code point (two-level: toks_bert_cls_s2[toks_bert_cls_s1[cp >> 8] << 8 | (cp & 0xFF)]):
 */
#ifndef TOKS_BERT_TABLES_H
#define TOKS_BERT_TABLES_H

#include <stdint.h>

#define TOKS_BC_REMOVE  0x01u  /* clean_text removes it: U+0000, U+FFFD, Cc Cf Co (8.0) except \\t \\n \\r */
#define TOKS_BC_WS      0x02u  /* White_Space (17.0): clean_text maps it to ' '; the pre-tokenizers split on it */
#define TOKS_BC_CJK     0x04u  /* is_chinese_char: handle_chinese_chars puts ' ' before and after it */
#define TOKS_BC_PUNCT   0x08u  /* BertPreTokenizer isolates it: ascii punctuation or P* (8.0) */
#define TOKS_BC_DECOMP  0x10u  /* NFD (9.0) changes it (hangul syllables: arithmetic, no map entry) */
#define TOKS_BC_LOWER   0x20u  /* char::to_lowercase (17.0) changes it */
#define TOKS_BC_NS      0x40u  /* canonical combining class > 0 (9.0): a non-starter */
#define TOKS_BC_MN      0x80u  /* Mn (8.0): strip_accents removes it after NFD */

#define TOKS_BERT_CLS_NBLOCKS  %(ncb)d
#define TOKS_BERT_MAP_NBLOCKS  %(nmb)d
#define TOKS_BERT_MAP_N        %(nent)d   /* map entries, entry 0 = none */
#define TOKS_BERT_POOL_N       %(npool)d
#define TOKS_BERT_KEEPNS_N     %(nkeep)d   /* non-starters strip_accents keeps (ccc > 0, not Mn 8.0) */
#define TOKS_BERT_MAX_DECOMP   %(maxdec)d    /* longest NFD expansion of one code point (non-hangul) */
#define TOKS_BERT_MAX_LOWER    %(maxlow)d    /* longest to_lowercase expansion (U+0130) */
/* the normalizer's output never exceeds GROWTH_NUM / GROWTH_DEN times its input bytes (worst: U+%(gcp)04X) */
#define TOKS_BERT_GROWTH_NUM   %(gnum)d
#define TOKS_BERT_GROWTH_DEN   %(gden)d
#define TOKS_BERT_DATA_SHA256  "%(dsha)s"

typedef struct toks_bert_map {
    uint16_t dec_off;   /* NFD decomposition: pool[dec_off .. dec_off + dec_len) */
    uint8_t  dec_len;
    uint8_t  rsv0;
    uint16_t low_off;   /* to_lowercase: pool[low_off .. low_off + low_len) */
    uint8_t  low_len;
    uint8_t  rsv1;
} toks_bert_map;

extern const uint16_t      toks_bert_cls_s1[0x1100];
extern const uint8_t       toks_bert_cls_s2[TOKS_BERT_CLS_NBLOCKS * 256];
extern const uint16_t      toks_bert_map_s1[0x1100];
extern const uint16_t      toks_bert_map_s2[TOKS_BERT_MAP_NBLOCKS * 256];
extern const toks_bert_map toks_bert_maps[TOKS_BERT_MAP_N];
extern const uint32_t      toks_bert_pool[TOKS_BERT_POOL_N];
extern const uint32_t      toks_bert_keepns[TOKS_BERT_KEEPNS_N][2];   /* (cp, ccc), ascending cp */

static inline uint8_t toks_bert_cls(uint32_t cp)
{
    if (cp > 0x10FFFFu) { return 0; }
    return toks_bert_cls_s2[((uint32_t)toks_bert_cls_s1[cp >> 8] << 8) | (cp & 0xFFu)];
}

static inline uint16_t toks_bert_map_index(uint32_t cp)
{
    if (cp > 0x10FFFFu) { return 0; }
    return toks_bert_map_s2[((uint32_t)toks_bert_map_s1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_BERT_TABLES_H */
""" % dict(ucd=UCD_VER, ncb=len(c_blocks), nmb=len(m_blocks), nent=len(entries), npool=len(pool), nkeep=len(keep_ns),
           maxdec=max_dec, maxlow=max_low, gcp=growth_cp, gnum=growth_num, gden=growth_den, dsha=dsha)
    c = ["/* GENERATED by tools/gen/bert_tables.py -- DO NOT EDIT. data sha256 %s */" % dsha,
         '#include "bert_tables.h"', "",
         "const uint16_t toks_bert_cls_s1[0x1100] = {", rows(c_s1, 16, "%d"), "};", "",
         "const uint8_t toks_bert_cls_s2[TOKS_BERT_CLS_NBLOCKS * 256] = {",
         rows([v for b in c_blocks for v in b], 32, "%d"), "};", "",
         "const uint16_t toks_bert_map_s1[0x1100] = {", rows(m_s1, 16, "%d"), "};", "",
         "const uint16_t toks_bert_map_s2[TOKS_BERT_MAP_NBLOCKS * 256] = {",
         rows([v for b in m_blocks for v in b], 16, "%d"), "};", "",
         "const toks_bert_map toks_bert_maps[TOKS_BERT_MAP_N] = {",
         "\n".join("    {%d, %d, 0, %d, %d, 0}," % e for e in entries), "};", "",
         "const uint32_t toks_bert_pool[TOKS_BERT_POOL_N] = {", rows(pool, 12, "0x%04X"), "};", "",
         "const uint32_t toks_bert_keepns[TOKS_BERT_KEEPNS_N][2] = {",
         "\n".join("    {0x%04X, %d}," % (cp, keep_ccc[cp]) for cp in keep_ns), "};", ""]
    (REPO / "src" / "gen" / "bert_tables.h").write_text(h)
    (REPO / "src" / "gen" / "bert_tables.c").write_text("\n".join(c))
    for p in ("tests/data/wordpiece/unicode.json", "src/gen/bert_tables.h", "src/gen/bert_tables.c"):
        print("%s sha256 %s" % (p, hashlib.sha256((REPO / p).read_bytes()).hexdigest()))
    print("data sha256 %s; cls blocks %d, map blocks %d, map entries %d, pool %d, keep-ns %d; %.1f s" % (
        dsha, len(c_blocks), len(m_blocks), len(entries), len(pool), len(keep_ns), time.time() - t0))


if __name__ == "__main__":
    main()
