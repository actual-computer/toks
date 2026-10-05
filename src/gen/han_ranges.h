/* toks: \p{Han} as tiktoken 0.14.0 resolves it (the kimi variant of the o200k template,
 * docs/templates/o200k.md §6): regex-syntax's Script=Han, not Script_Extensions (U+3001 is not Han).
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:
 *     uv run tools/gen/han.py
 * (probes tiktoken's own engine over every scalar value; deps pinned in that script's PEP 723 header)
 *
 * 22 ranges, 99030 code points, ascending and disjoint.
 * table sha256 (each range as u32 LE lo, u32 LE hi): a03ac5c876a98de2a9f2db44e9fe709999d3c3ee1f451f093455d8a63a2c45a9
 */

#ifndef TOKS_HAN_RANGES_H
#define TOKS_HAN_RANGES_H

#include <stdint.h>

#define TOKS_HAN_N 22u
#define TOKS_HAN_CODEPOINTS 99030u
#define TOKS_HAN_SHA256 "a03ac5c876a98de2a9f2db44e9fe709999d3c3ee1f451f093455d8a63a2c45a9"

extern const uint32_t toks_han_ranges[TOKS_HAN_N][2];

#endif /* TOKS_HAN_RANGES_H */
