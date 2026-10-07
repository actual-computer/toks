/*
 * tests/proof/prelude.h: force-included before every translation unit of an eva run (tests/proof/eva.sh).
 * Frama-C's own libc headers give memcpy / memset / memcmp / malloc their ACSL contracts (eva checks each
 * call against them: SPEC §14.1's external contracts); __fc_builtin.h gives the nondeterminism.
 */
#ifndef TOKS_PROOF_PRELUDE_H
#define TOKS_PROOF_PRELUDE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "__fc_builtin.h"

#define PROOF_SRC_MAX (256ull << 20)       /* SPEC §8.3: a source of at most 256 MiB */

/* ---- the arena ------------------------------------------------------------------------------------------
 * The allocator under its contract (assume-guarantee; docs/proof.md: core.h's toks_ar_alloc meets it): NULL
 * when the arena cannot hold n more bytes, else a block of n bytes inside [a->base, a->base + a->len), aligned
 * to align, disjoint from every block returned before. Its clients are
 * analyzed with each block as its own exact-size allocation: an access past a block's end is an alarm even
 * where the real arena would hand it the next block's bytes (stricter than the arena itself). The entries give
 * the arena no base (entries.c), so core.h's allocator is never what runs here. */
#define toks_ar_alloc toks_ar_alloc_core
#include "core.h"
#undef toks_ar_alloc
static inline void *toks_ar_alloc(toks_arena *a, uint64_t n, uint64_t align)
{
    (void)align;
    if (a->pos > a->len || n > a->len - a->pos) { return NULL; }
    void *p = malloc((size_t)n);          /* eva: may be NULL (exhaustion is modeled as possible anywhere) */
    if (p != NULL) { a->pos += n; }
    return p;
}

#endif
