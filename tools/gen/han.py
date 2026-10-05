# /// script
# requires-python = ">=3.9"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""toks: the \\p{Han} range table of the o200k template's kimi variant, from tiktoken's own regex engine.

moonshotai/Kimi-K3 encodes with tiktoken (docs/templates/o200k.md §6). tiktoken compiles its pat_str with
fancy-regex, which hands character classes to the regex crate; there a bare script name is the Script property
(regex-syntax 0.8.11 unicode.rs canonical_binary -> Script, not Script_Extensions), so U+3001 is not Han. This
script asks tiktoken itself: an Encoding whose pattern is `[\\p{Han}]` and whose vocabulary is every scalar value
returns one token per member (tiktoken drops unmatched text), over all of U+0000..U+10FFFF but the surrogates.
The members become ascending, disjoint ranges, written as

    src/gen/han_ranges.h   TOKS_HAN_N, TOKS_HAN_CODEPOINTS, TOKS_HAN_SHA256, extern toks_han_ranges
    src/gen/han_ranges.c   the table

with the sha256 of the table (each range as two u32 little-endian: lo, hi) in both, which tests check.
classes.c sets TOKS_C_HAN from this table under TOKS_CLASSES_HAN; tests/model/o200k_model.py and the c tests
read the same table: it is the one copy.

Usage (any lab host; ~1 minute of tiktoken):
    uv run tools/gen/han.py

Expected today (tiktoken 0.14.0 = regex-syntax 0.8.11, Unicode 16.0.0): 22 ranges, 99,030 code points; the
script refuses to write anything else without --accept-new (a new tiktoken must be re-proven, not absorbed).
"""

import hashlib
import struct
import sys
from pathlib import Path

import tiktoken

REPO = Path(__file__).resolve().parents[2]
EXPECT_RANGES, EXPECT_CPS = 22, 99030
BATCH = 4096


def probe():
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    ranks = {chr(c).encode("utf-8"): i for i, c in enumerate(cps)}
    enc = tiktoken.Encoding("toks-han-probe", pat_str=r"[\p{Han}]", mergeable_ranks=ranks, special_tokens={})
    han = set()
    for i in range(0, len(cps), BATCH):
        for t in enc.encode_ordinary("".join(map(chr, cps[i:i + BATCH]))):
            han.add(ord(enc.decode_single_token_bytes(t).decode("utf-8")))
    return han


def ranges_of(cps):
    out = []
    for c in sorted(cps):
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [tuple(r) for r in out]


H_SRC = """/* toks: \\p{Han} as tiktoken %(ver)s resolves it (the kimi variant of the o200k template,
 * docs/templates/o200k.md §6): regex-syntax's Script=Han, not Script_Extensions (U+3001 is not Han).
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:
 *     uv run tools/gen/han.py
 * (probes tiktoken's own engine over every scalar value; deps pinned in that script's PEP 723 header)
 *
 * %(n)d ranges, %(cps)d code points, ascending and disjoint.
 * table sha256 (each range as u32 LE lo, u32 LE hi): %(sha)s
 */

#ifndef TOKS_HAN_RANGES_H
#define TOKS_HAN_RANGES_H

#include <stdint.h>

#define TOKS_HAN_N %(n)du
#define TOKS_HAN_CODEPOINTS %(cps)du
#define TOKS_HAN_SHA256 "%(sha)s"

extern const uint32_t toks_han_ranges[TOKS_HAN_N][2];

#endif /* TOKS_HAN_RANGES_H */
"""

C_SRC = """/* toks: \\p{Han} as tiktoken %(ver)s resolves it.  GENERATED FILE -- DO NOT EDIT.
 * Regenerate with: uv run tools/gen/han.py
 * table sha256: %(sha)s  (%(n)d ranges, %(cps)d code points)
 */

#include "han_ranges.h"

const uint32_t toks_han_ranges[TOKS_HAN_N][2] = {
%(rows)s
};
"""


def main():
    accept_new = "--accept-new" in sys.argv[1:]
    han = probe()
    rs = ranges_of(han)
    n_cps = sum(hi - lo + 1 for lo, hi in rs)
    print("tiktoken %s \\p{Han}: %d code points in %d ranges" % (tiktoken.__version__, n_cps, len(rs)))
    print("U+3001 %s, U+3005 %s, U+3006 %s, U+3007 %s" % tuple(
        "in" if c in han else "out" for c in (0x3001, 0x3005, 0x3006, 0x3007)))
    if (len(rs), n_cps) != (EXPECT_RANGES, EXPECT_CPS) and not accept_new:
        raise SystemExit("han.py: %d ranges / %d code points, expected %d / %d (tiktoken changed? re-prove the "
                         "kimi variant, then --accept-new)" % (len(rs), n_cps, EXPECT_RANGES, EXPECT_CPS))
    data = b"".join(struct.pack("<II", lo, hi) for lo, hi in rs)
    sha = hashlib.sha256(data).hexdigest()
    rows = []
    for i in range(0, len(rs), 4):
        rows.append("    " + " ".join("{ 0x%05Xu, 0x%05Xu }," % r for r in rs[i:i + 4]))
    args = dict(ver=tiktoken.__version__, n=len(rs), cps=n_cps, sha=sha, rows="\n".join(rows))
    for path, text in ((REPO / "src/gen/han_ranges.h", H_SRC % args), (REPO / "src/gen/han_ranges.c", C_SRC % args)):
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote %s (%d bytes) sha256 %s" % (path.relative_to(REPO), path.stat().st_size,
                                                  hashlib.sha256(path.read_bytes()).hexdigest()))
    print("table sha256: %s" % sha)


if __name__ == "__main__":
    main()
