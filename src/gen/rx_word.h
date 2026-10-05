/* toks: the \w of hf tokenizers 0.23.2's added-token single_word check (added_vocabulary.rs
 * ends_with_word / starts_with_word: the rust regex crate's unicode \w), as probed through hf itself.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate on a lab host with:
 *     uv run --with tokenizers==0.23.2 python tests/data/breadth/probe_hf.py word --emit
 *
 * 144667 code points in 796 ranges, probed on both sides of a match over every
 * scalar value. lstrip / rstrip's \s probed the same way: exactly the 25 code points of segment.c.
 * data (lo u32 LE || hi u32 LE per range) sha256: 74ad8de6f2e9dddce09ec6c5547ee750ff65b1d5ca5a9dbc923731127985b712
 */
#ifndef TOKS_RX_WORD_H
#define TOKS_RX_WORD_H

#include <stdint.h>

#define TOKS_RX_WORD_N 796u
#define TOKS_RX_WORD_SHA256 "74ad8de6f2e9dddce09ec6c5547ee750ff65b1d5ca5a9dbc923731127985b712"

/* sorted, disjoint, non-adjacent [lo, hi] ranges */
extern const uint32_t toks_rx_word[TOKS_RX_WORD_N][2];

#endif /* TOKS_RX_WORD_H */
