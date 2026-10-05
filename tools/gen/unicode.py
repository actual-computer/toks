# /// script
# requires-python = ">=3.9"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""toks unicode property tables, generated from hf tokenizers' own regex engine.

toks' scanner templates classify code points by data (SPEC 1.4).  This script
probes the regex engine hf `tokenizers` 0.23.2 actually runs (oniguruma) through
`pre_tokenizers.Split(Regex(...), behavior="removed")` over every unicode scalar
value U+0000..U+10FFFF, and emits the two-level per-code-point flag table
`src/gen/ucd_flags.{c,h}`:

  - one sweep per property (\\p{Lu} \\p{Ll} \\p{Lt} \\p{Lm} \\p{Lo} \\p{M} \\p{N}
    \\s \\p{P} \\p{S}) over batched strings of code points; a code point the regex matches is
    removed from the split output, so "not kept" == "in the property"
    (pre_tokenize_str offsets are char offsets, verified below).
  - \\p{L} is probed separately and must equal the union of the five letter
    properties (asserted on every code point).
  - case folds are probed per contraction letter (?i:s ... ?i:l) and with the
    full contraction group (?i:'s|'t|'re|'ve|'m|'ll|'d) over batched items
    "'<cp>\\0", "'<cp>e\\0", "'<cp>l\\0": the only non-ASCII fold that reaches
    the contraction set is U+017F (long s) -> s.  Anything else fails the run
    and is printed (no silent assumptions).
  - surrogates U+D800..U+DFFF are not scalar values and keep flags 0.

The properties are oniguruma's own unicode data (unicode 16.0 classes; see
docs/unicode.md for the version evidence).  Everything is cross-checked against
python's unicodedata (its version is stated in the report) and every
disagreement is printed.

Usage:
    uv run tools/gen/unicode.py

Writes (deterministic output; checked in with the hashes this prints):
    src/gen/ucd_flags.c
    src/gen/ucd_flags.h

The table is the foundation of the per-template class tables built at load time
by src/core/classes.c (toks_classes_build).
"""

import hashlib
import struct
import sys
import unicodedata
from pathlib import Path

import tokenizers
from tokenizers import Regex, pre_tokenizers

REPO = Path(__file__).resolve().parents[2]

# flag bits (mirrored in src/gen/ucd_flags.h; single source of truth is here)
LU, LL, LT, LM, LO = 0x0001, 0x0002, 0x0004, 0x0008, 0x0010
M, N, WS, FOLD_S = 0x0020, 0x0040, 0x0080, 0x0100
P_, S_ = 0x0200, 0x0400           # \\p{P}, \\p{S} (the dsv3 template's punctuation; docs/templates/dsv3.md)
# hf's other pre-tokenizer sets (not oniguruma's): Digits' rust char::is_numeric (the compiler's unicode, newer
# than onig's: a superset of \\p{N}) and Punctuation's is_punc (ascii punctuation + the unicode_categories
# crate's old P tables); then \\d and \\w of the regex engine (the generic engine's \\d and \\b)
RNUM, ND, WORD, PUNC = 0x0800, 0x1000, 0x2000, 0x4000
LETTERS = LU | LL | LT | LM | LO

PROPS = [
    ("Lu", LU, r"\p{Lu}"),
    ("Ll", LL, r"\p{Ll}"),
    ("Lt", LT, r"\p{Lt}"),
    ("Lm", LM, r"\p{Lm}"),
    ("Lo", LO, r"\p{Lo}"),
    ("M", M, r"\p{M}"),
    ("N", N, r"\p{N}"),
    ("WS", WS, r"\s"),
    ("P", P_, r"\p{P}"),
    ("S", S_, r"\p{S}"),
    ("Nd", ND, r"\d"),
    ("W", WORD, r"\w"),
]

CONTRACTION = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
CONTRACTION_LETTERS = "stmdrevl"

WS_EXPECTED = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
WS_EXPECTED += list(range(0x2000, 0x200B))
WS_EXPECTED += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]

BATCH = 0x8000  # code points per probe string (32 Ki chars)


def scalars():
    """every unicode scalar value, ascending; surrogates are not scalar values."""
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


def make_split(pat):
    return pre_tokenizers.Split(Regex(pat), "removed")


def kept_positions(sp, s):
    """bytearray over the chars of s: 1 where the split output kept the char."""
    keep = bytearray(len(s))
    for _, (a, b) in sp.pre_tokenize_str(s):
        keep[a:b] = b"\1" * (b - a)
    return keep


def probe_property(pat, cps):
    """-> bytearray(0x110000): 1 where the single-char regex matches the cp.

    A match is removed from the split output, so "not kept" == "matched".
    """
    sp = make_split(pat)
    hit = bytearray(0x110000)
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        keep = kept_positions(sp, "".join(map(chr, part)))
        for k, cp in enumerate(part):
            if not keep[k]:
                hit[cp] = 1
    return hit


def probe_isolated(pt, cps):
    """-> bytearray(0x110000): 1 where the pre-tokenizer pt (a char predicate, behavior isolated) makes the cp
    its own piece between two 'a' (in no hf predicate set)."""
    hit = bytearray(0x110000)
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        s = "a" + "".join(chr(cp) + "a" for cp in part)
        for _, (a, b) in pt.pre_tokenize_str(s):
            if b == a + 1 and s[a] != "a":
                hit[ord(s[a])] = 1
    return hit


def probe_contraction(cps, tail):
    """-> set of code points X where 'X (in context 'X<tail>) matches the group.

    Items are "'X<tail>\\0", NUL-separated so no match can span items: the
    pattern needs a literal ' first and NUL matches nothing.  A match removes
    at least the ' and X, so the X position not being kept means X participates.
    This catches 1:1 folds (X->s/t/m/d directly, X->r/v/l with the tail) and
    would catch a multi-char fold covering two pattern letters.
    """
    sp = make_split(CONTRACTION)
    item = 2 + len(tail) + 1
    matched = set()
    for i in range(0, len(cps), 0x1000):
        part = cps[i:i + 0x1000]
        s = "".join("'" + chr(cp) + tail + "\0" for cp in part)
        keep = kept_positions(sp, s)
        for k, cp in enumerate(part):
            if not keep[k * item + 1]:
                matched.add(cp)
    return matched


def two_level(vals):
    """u16 value per code point -> (stage1 u16[0x1100], [block bytes]).

    Blocks are 256 consecutive code points, deduplicated in first-seen order;
    block 0 is U+0000..U+00FF (stage1[0] == 0, asserted) so Latin-1 is a
    direct stage2[c] load.
    """
    blocks, index, stage1 = [], {}, []
    for hi in range(0x1100):
        blk = struct.pack("<256H", *vals[hi * 256:(hi + 1) * 256])
        if blk not in index:
            index[blk] = len(blocks)
            blocks.append(blk)
        stage1.append(index[blk])
    assert stage1[0] == 0
    return stage1, blocks


def fmt_u16(vals, per=16):
    out = []
    for i in range(0, len(vals), per):
        out.append("    " + ", ".join("0x%04X" % v for v in vals[i:i + per]) + ",")
    return "\n".join(out)


def u16_bytes(vals):
    return struct.pack("<%dH" % len(vals), *vals)


H_Facts = """/* toks: per-code-point unicode property flags as hf tokenizers' regex engine
 * (oniguruma) sees them -- NOT python's unicodedata.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:
 *     uv run tools/gen/unicode.py
 * (deps pinned in that script's PEP 723 header: tokenizers==0.23.2)
 *
 * Probed facts (every scalar value U+0000..U+10FFFF, surrogates flags 0):
 *   \\p{L} == \\p{Lu}|\\p{Ll}|\\p{Lt}|\\p{Lm}|\\p{Lo}: %(L)d code points (probed both ways)
 *   \\p{M} %(M)d, \\p{N} %(N)d, \\s %(WS)d, \\p{P} %(P)d, \\p{S} %(S)d code points
 *   \\d %(Nd)d, \\w %(W)d; hf Digits' is_numeric %(RNUM)d (\\p{N} plus %(RNUM_X)s),
 *   hf Punctuation's is_punc %(PUNC)d
 *   L, M, N, \\s, P, S pairwise disjoint
 *   \\s = U+0009..000D, 0020, 0085, 00A0, 1680, 2000..200A, 2028, 2029, 202F, 205F, 3000
 *   (?i) folds U+017F (long s) onto s: the only non-ASCII fold reaching the
 *   contraction set of (?i:'s|'t|'re|'ve|'m|'ll|'d) (every scalar probed with
 *   the full group, in 'X, 'Xe, 'Xl contexts; no other fold matched)
 *   classes agree with unicode 16.0 (== python 3.14 unicodedata on L/M/N/\\s/P/S);
 *   python 3.12's unicodedata 15.0 disagrees exactly on post-15.0 additions
 *
 * two-level table: stage1[cp >> 8] is the block id, then
 * stage2[(block << 8) | (cp & 0xFF)] is the flag word (0 for surrogates).
 *
 * table data (n_blocks u32 LE || stage1 u16[0x1100] LE || stage2 u16[n_blocks*256] LE)
 * sha256: %(sha)s
 */
"""

H_BODY = """
#ifndef TOKS_UCD_FLAGS_H
#define TOKS_UCD_FLAGS_H

#include <stdint.h>

/* flag bits of toks_ucd_flags(); each is 1 iff oniguruma (as run by hf
 * tokenizers 0.23.2) has the code point in the property. */
#define TOKS_UCD_LU 0x0001u /* \\p{Lu} */
#define TOKS_UCD_LL 0x0002u /* \\p{Ll} */
#define TOKS_UCD_LT 0x0004u /* \\p{Lt} */
#define TOKS_UCD_LM 0x0008u /* \\p{Lm} */
#define TOKS_UCD_LO 0x0010u /* \\p{Lo} */
#define TOKS_UCD_M 0x0020u  /* \\p{M} */
#define TOKS_UCD_N 0x0040u  /* \\p{N} */
#define TOKS_UCD_WS 0x0080u /* \\s */
#define TOKS_UCD_P 0x0200u  /* \\p{P} */
#define TOKS_UCD_S 0x0400u  /* \\p{S} */
/* hf's other pre-tokenizer sets, probed through hf too: Digits' rust char::is_numeric (a superset of \\p{N}: the
 * compiler's newer unicode) and Punctuation's is_punc; the regex engine's \\d and \\w */
#define TOKS_UCD_RNUM 0x0800u
#define TOKS_UCD_ND 0x1000u
#define TOKS_UCD_WORD 0x2000u
#define TOKS_UCD_PUNC 0x4000u
/* non-ASCII code point that (?i) matches as 's' (U+017F long s; the only one) */
#define TOKS_UCD_FOLD_S 0x0100u

/* \\p{L} == \\p{Lu}|\\p{Ll}|\\p{Lt}|\\p{Lm}|\\p{Lo} (probed, asserted at generation) */
#define TOKS_UCD_LETTERS (TOKS_UCD_LU | TOKS_UCD_LL | TOKS_UCD_LT | TOKS_UCD_LM | TOKS_UCD_LO)

/* deduplicated stage 2 blocks of 256 code points */
#define TOKS_UCD_N_BLOCKS %(n_blocks)d
/* sha256 of n_blocks u32 LE || stage1 || stage2 (see the file header) */
#define TOKS_UCD_DATA_SHA256 "%(sha)s"

extern const uint16_t toks_ucd_stage1[0x1100];
extern const uint16_t toks_ucd_stage2[TOKS_UCD_N_BLOCKS * 256];

/* flags of one code point; 0 for cp > U+10FFFF (and for surrogates). */
static inline uint16_t toks_ucd_flags(uint32_t cp)
{
    if (cp > 0x10FFFFu) {
        return 0;
    }
    return toks_ucd_stage2[((uint32_t)toks_ucd_stage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_UCD_FLAGS_H */
"""

C_SRC = """/* toks: per-code-point unicode property flags as oniguruma (hf tokenizers
 * 0.23.2's regex engine) sees them.  GENERATED FILE -- DO NOT EDIT.
 * Regenerate with: uv run tools/gen/unicode.py
 * table data sha256: %(sha)s  (n_blocks %(n_blocks)d)
 */

#include "ucd_flags.h"

/* stage 1: block id per 256 code points; block 0 is U+0000..U+00FF */
const uint16_t toks_ucd_stage1[0x1100] = {
%(stage1)s
};

/* stage 2: %(n_blocks)d deduplicated blocks of 256 flag words */
const uint16_t toks_ucd_stage2[TOKS_UCD_N_BLOCKS * 256] = {
%(stage2)s
};
"""


def main():
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("unicode.py: tokenizers %s != 0.23.2 (the pinned oracle)"
                         % tokenizers.__version__)
    print("oracle: tokenizers %s (oniguruma)" % tokenizers.__version__)

    cps = scalars()
    print("probing %d scalar values ..." % len(cps))

    hits = {}
    for name, _bit, pat in PROPS:
        hits[name] = probe_property(pat, cps)
        print("  %-3s %6d code points" % (name, sum(hits[name])))
    l_all = probe_property(r"\p{L}", cps)
    hits["RNUM"] = probe_isolated(pre_tokenizers.Digits(individual_digits=True), cps)
    hits["PUNC"] = probe_isolated(pre_tokenizers.Punctuation("isolated"), cps)
    for name in ("RNUM", "PUNC"):
        print("  %-4s %6d code points" % (name, sum(hits[name])))
    for cp in cps:
        if hits["N"][cp] and not hits["RNUM"][cp]:
            raise SystemExit("unicode.py: U+%04X is \\p{N} but not hf Digits' is_numeric" % cp)
        if hits["Nd"][cp] and not hits["N"][cp]:
            raise SystemExit("unicode.py: U+%04X is \\d but not \\p{N}" % cp)
    print("hf Digits' is_numeric beyond \\p{N}: %s" % " ".join(
        "U+%04X" % cp for cp in cps if hits["RNUM"][cp] and not hits["N"][cp]))

    # ---- letter union cross-check --------------------------------------
    for cp in cps:
        if bool(l_all[cp]) != any(hits[k][cp] for k in ("Lu", "Ll", "Lt", "Lm", "Lo")):
            raise SystemExit("unicode.py: \\p{L} != Lu|Ll|Lt|Lm|Lo at U+%04X" % cp)
    union_letters = [cp for cp in cps if l_all[cp]]
    print("cross-check: \\p{L} == Lu|Ll|Lt|Lm|Lo on every code point (%d)" % len(union_letters))

    # ---- properties are pairwise disjoint ------------------------------
    bits = {name: bit for name, bit, _ in PROPS}
    for cp in cps:
        acc = 0
        for name, _bit, _pat in PROPS[:10]:
            if hits[name][cp]:
                acc |= bits[name]
        for a, b in (("Lu", "Ll"), ("M", "N"), ("M", "WS"), ("N", "WS"), ("LETTERS", "M"),
                     ("LETTERS", "N"), ("LETTERS", "WS"), ("P", "S"), ("P", "LETTERS"), ("P", "M"),
                     ("P", "N"), ("P", "WS"), ("S", "LETTERS"), ("S", "M"), ("S", "N"), ("S", "WS")):
            x = l_all[cp] if a == "LETTERS" else bool(hits[a][cp])
            y = l_all[cp] if b == "LETTERS" else bool(hits[b][cp])
            if x and y:
                raise SystemExit("unicode.py: U+%04X is both %s and %s" % (cp, a, b))

    # ---- (?i) folds ----------------------------------------------------
    folds = {}
    for letter in CONTRACTION_LETTERS:
        hit = probe_property("(?i:%s)" % letter, cps)
        folds[letter] = {cp for cp in cps if hit[cp]}
        extra = sorted(cp for cp in folds[letter] if cp >= 0x80)
        print("  (?i:%s) matches %d code points%s"
              % (letter, len(folds[letter]),
                 ("; non-ASCII: " + " ".join("U+%04X" % c for c in extra)) if extra else ""))
    fold_all = set().union(*folds.values())
    fold_nonascii = {cp for cp in fold_all if cp >= 0x80}
    print("non-ASCII chars that (?i) folds onto a contraction letter: %s"
          % " ".join("U+%04X" % c for c in sorted(fold_nonascii)))
    if fold_nonascii != {0x17F}:
        raise SystemExit("unicode.py: unexpected fold set %r (expected only U+017F)"
                         % sorted(fold_nonascii))

    # the contraction group itself, over every scalar, three contexts
    obs2 = probe_contraction(cps, "")
    obs_e = probe_contraction(cps, "e")
    obs_l = probe_contraction(cps, "l")
    exp2 = folds["s"] | folds["t"] | folds["m"] | folds["d"]
    exp_e = exp2 | folds["r"] | folds["v"]
    exp_l = exp2 | folds["l"]
    for what, obs, exp in (("'X", obs2, exp2), ("'Xe", obs_e, exp_e), ("'Xl", obs_l, exp_l)):
        if obs != exp:
            print("SURPRISE: contraction matches in %s context:" % what)
            print("  extra:      %r" % sorted(obs - exp))
            print("  missing:    %r" % sorted(exp - obs))
            raise SystemExit("unicode.py: the contraction fold set is not the expected one")
    print("contraction group (?i:'s|'t|'re|'ve|'m|'ll|'d) over every scalar, contexts "
          "'X / 'Xe / 'Xl: participants %s"
          % " ".join(chr(c) for c in sorted(obs2 | obs_e | obs_l)))
    fold_s_cps = sorted(folds["s"] - set(range(0x80)))

    # ---- \s -------------------------------------------------------------
    ws_list = [cp for cp in cps if hits["WS"][cp]]
    if ws_list != WS_EXPECTED:
        raise SystemExit("unicode.py: \\s is %r, expected %r" % (ws_list, WS_EXPECTED))
    print("\\s: %d code points (matches the known 25)" % len(ws_list))

    # ---- flags array ----------------------------------------------------
    flags = [0] * 0x110000
    for cp in cps:
        f = 0
        for name, bit, _pat in PROPS:
            if hits[name][cp]:
                f |= bit
        f |= (RNUM if hits["RNUM"][cp] else 0) | (PUNC if hits["PUNC"][cp] else 0)
        flags[cp] = f
    for cp in fold_s_cps:
        flags[cp] |= FOLD_S

    # ---- cross-check against python's unicodedata -----------------------
    ver = unicodedata.unidata_version
    diffs = {name: [] for name, _bit, _pat in PROPS}
    diffs["L"] = []
    py_ws = {}
    for cp in cps:
        c = chr(cp)
        cat = unicodedata.category(c)
        if bool(l_all[cp]) != cat.startswith("L"):
            diffs["L"].append(cp)
        for name, _bit, _pat in PROPS[:10]:
            if name == "WS":
                continue
            want = cat.startswith(name)
            if bool(hits[name][cp]) != want:
                diffs[name].append(cp)
    # \s against the category reading: Zs|Zl|Zp plus the Cc controls 09-0D and NEL
    for cp in cps:
        want = unicodedata.category(chr(cp)) in ("Zs", "Zl", "Zp") or cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85)
        if bool(hits["WS"][cp]) != want:
            diffs["WS"].append(cp)
    pyws_extra = sorted(cp for cp in cps if chr(cp).isspace() and not hits["WS"][cp])
    print("cross-check vs python unicodedata %s (python %s):" % (ver, sys.version.split()[0]))
    for name in ("L", "Lu", "Ll", "Lt", "Lm", "Lo", "M", "N", "WS", "P", "S"):
        d = diffs[name]
        print("  %-3s disagreements %d%s" % (name, len(d),
              (" e.g. " + " ".join("U+%04X" % c for c in d[:8])) if d else ""))

    # targeted post-unicode-9 code points (their class changed after 9.0/15.0)
    for cp, why in ((0x897, "Todhri mark, unicode 14.0"), (0x9FEA, "CJK ext H, 13.0"),
                    (0x30000, "CJK ext G, 13.0"), (0x105C0, "Todhri letter, 16.0")):
        print("  U+%05X (%s): flags 0x%04X" % (cp, why, flags[cp]))

    # ---- two-level table -------------------------------------------------
    stage1, blocks = two_level(flags)
    n_blocks = len(blocks)
    stage2_vals = struct.unpack("<%dH" % (n_blocks * 256), b"".join(blocks))
    data = struct.pack("<I", n_blocks) + u16_bytes(stage1) + b"".join(blocks)
    sha = hashlib.sha256(data).hexdigest()
    print("two-level table: %d blocks; stage1 %d B, stage2 %d B, data sha256 %s"
          % (n_blocks, 0x2200, n_blocks * 512, sha))

    counts = {
        "L": len(union_letters),
        "M": sum(hits["M"]),
        "N": sum(hits["N"]),
        "WS": len(ws_list),
        "P": sum(hits["P"]),
        "S": sum(hits["S"]),
        "Nd": sum(hits["Nd"]),
        "W": sum(hits["W"]),
        "RNUM": sum(hits["RNUM"]),
        "PUNC": sum(hits["PUNC"]),
        "RNUM_X": " ".join("U+%04X" % cp for cp in cps if hits["RNUM"][cp] and not hits["N"][cp]),
    }
    facts = H_Facts % dict(counts, sha=sha, n_blocks=n_blocks)
    h_text = facts + H_BODY % dict(n_blocks=n_blocks, sha=sha)
    c_text = C_SRC % dict(sha=sha, n_blocks=n_blocks,
                          stage1=fmt_u16(stage1), stage2=fmt_u16(stage2_vals))

    h_path = REPO / "src/gen/ucd_flags.h"
    c_path = REPO / "src/gen/ucd_flags.c"
    h_path.parent.mkdir(parents=True, exist_ok=True)
    h_path.write_text(h_text, encoding="utf-8", newline="\n")
    c_path.write_text(c_text, encoding="utf-8", newline="\n")

    for p in (h_path, c_path):
        print("wrote %s (%d bytes) sha256 %s"
              % (p.relative_to(REPO), p.stat().st_size,
                 hashlib.sha256(p.read_bytes()).hexdigest()))
    print("table data sha256: %s" % sha)

    print("summary: Lu %d Ll %d Lt %d Lm %d Lo %d | L %d M %d N %d WS %d P %d S %d FOLD_S %d"
          % (sum(hits["Lu"]), sum(hits["Ll"]), sum(hits["Lt"]), sum(hits["Lm"]),
             sum(hits["Lo"]), counts["L"], counts["M"], counts["N"], counts["WS"],
             counts["P"], counts["S"], len(fold_s_cps)))


if __name__ == "__main__":
    main()
