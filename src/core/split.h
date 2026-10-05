/* split.h: what the core gives toks_par and the incremental layer beyond the public abi (docs/notes/c-core.md
 * §split.h.1) */
#ifndef TOKS_SPLIT_H
#define TOKS_SPLIT_H

#include "core.h"

/* rationale: docs/notes/c-core.md §split.h.2 */
void toks_pp_ids(const toks_ctx *ctx, uint32_t flags, const uint32_t **pre, uint32_t *n_pre,
                 const uint32_t **suf, uint32_t *n_suf);

/* the certified-cut rule families (docs/split.md) */
#define TOKS_CUT_NONE    0u
#define TOKS_CUT_CL100K  1u    /* byte-level bpe, the cl100k template: kernels.md R1-R5 */
#define TOKS_CUT_O200K   2u    /* byte-level bpe, the o200k template: docs/split.md §3 */
#define TOKS_CUT_DSV3    3u    /* byte-level bpe, deepseek's three Splits: docs/split.md §4 */
#define TOKS_CUT_SPM     4u    /* sentencepiece-style bpe: K7's certified word cuts (spm_bpe.md §5.5) */
#define TOKS_CUT_WP      5u    /* wordpiece: after ascii whitespace */
#define TOKS_CUT_UNI     6u    /* unigram: at a lone U+0020 between simple ascii chars */

/* a call's rules: the family certifying cuts for (ctx, flags' mode) and how far a rule reads */
typedef struct toks_cuts {
    uint32_t family;       /* TOKS_CUT_*; NONE: no cut is ever certified (truncation, padding, kimi's cuts, ...) */
    uint32_t rsv;
    uint64_t maxlen;       /* the longest added token the mode recognizes (0: none) */
    uint64_t win;          /* toks_cut_ok(c) reads only x[c - win, c + win) */
} toks_cuts;

/* 1 and *k filled when ctx has certified cuts under flags, else 0 (k->family NONE) */
int toks_cuts_of(const toks_ctx *ctx, uint32_t flags, toks_cuts *k);

/* 1 when 0 < c < len is a certified cut of x[0, len) (SPEC §5.2): encoding x[0, c) and x[c, len) (the second with
 * TOKS_CONTINUATION, no post-processing) gives x's ids. Depends only on x[c - k->win, c + k->win) and on len when the
 * window reaches it: a cut with c + win <= len stays certified for every text that extends x. */
int toks_cut_ok(const toks_ctx *ctx, const toks_cuts *k, const uint8_t *x, uint64_t len, uint64_t c);

#endif /* TOKS_SPLIT_H */
