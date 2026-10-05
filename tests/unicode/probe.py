# /// script
# requires-python = ">=3.9"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Fresh hf probe for tests/unicode/main.c: ground truth over every scalar value.

Probes hf tokenizers 0.23.2's regex engine (oniguruma) exactly as
tools/gen/unicode.py does (verified to agree with the generated table), and
prints one line per code point U+0000..U+10FFFF:

    cp base0 base1 attr0 attr1

where baseN is the base class (0 P, 1 L, 2 N, 3 WS, 4 NL) and attrN the
attribute bits (UPPER 8, LOWER 16, MARK 32, FOLD_S 64) that
toks_classes_build(class_flags = N) must produce.  Output is deterministic.

Usage:  uv run tests/unicode/probe.py > gt.txt   (then feed to ./main)
"""

import sys

import tokenizers
from tokenizers import Regex, pre_tokenizers

LU, LL, LT, LM, LO = 0x0001, 0x0002, 0x0004, 0x0008, 0x0010
M, N, WS = 0x0020, 0x0040, 0x0080
FOLD_S = 0x0100
LETTERS = LU | LL | LT | LM | LO

WS_EXPECTED = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
WS_EXPECTED += list(range(0x2000, 0x200B))
WS_EXPECTED += [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]

BATCH = 0x8000


def scalars():
    return [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


def probe_property(pat, cps):
    sp = pre_tokenizers.Split(Regex(pat), "removed")
    hit = bytearray(0x110000)
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        s = "".join(map(chr, part))
        keep = bytearray(len(s))
        for _, (a, b) in sp.pre_tokenize_str(s):
            keep[a:b] = b"\1" * (b - a)
        for k, cp in enumerate(part):
            if not keep[k]:
                hit[cp] = 1
    return hit


def class_bytes(f, cp, marks_are_letters):
    """flags -> (base, attr) using the classes.h layout."""
    if f & LETTERS or (marks_are_letters and f & M):
        base = 1
    elif f & N:
        base = 2
    elif cp in (0x0A, 0x0D):
        base = 4
    elif f & WS:
        base = 3
    else:
        base = 0
    attr = 0
    if f & (LU | LT | LM | LO | M):
        attr |= 8
    if f & (LL | LM | LO | M):
        attr |= 16
    if f & M:
        attr |= 32
    if f & FOLD_S:
        attr |= 64
    return base, attr


def main():
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("probe.py: tokenizers %s != 0.23.2" % tokenizers.__version__)
    cps = scalars()
    hits = {name: probe_property(p, cps) for name, p in [
        ("Lu", r"\p{Lu}"), ("Ll", r"\p{Ll}"), ("Lt", r"\p{Lt}"), ("Lm", r"\p{Lm}"),
        ("Lo", r"\p{Lo}"), ("M", r"\p{M}"), ("N", r"\p{N}"), ("WS", r"\s")]}
    ws_got = [cp for cp in cps if hits["WS"][cp]]
    if ws_got != WS_EXPECTED:
        raise SystemExit("probe.py: \\s is not the known 25: %r" % ws_got)
    fold_s = probe_property(r"(?i:s)", cps)
    fold_hits = [cp for cp in cps if fold_s[cp] and cp >= 0x80]
    if fold_hits != [0x17F]:
        raise SystemExit("probe.py: (?i:s) non-ASCII hits %r" % fold_hits)

    out = []
    for cp in range(0x110000):
        f = 0
        if not 0xD800 <= cp <= 0xDFFF:
            for name in ("Lu", "Ll", "Lt", "Lm", "Lo", "M", "N", "WS"):
                if hits[name][cp]:
                    f |= {"Lu": LU, "Ll": LL, "Lt": LT, "Lm": LM, "Lo": LO,
                          "M": M, "N": N, "WS": WS}[name]
            if fold_s[cp] and cp >= 0x80:
                f |= FOLD_S
        b0, a0 = class_bytes(f, cp, False)
        b1, a1 = class_bytes(f, cp, True)
        out.append("%d %d %d %d %d" % (cp, b0, b1, a0, a1))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
