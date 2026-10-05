/* compile.h: toks_config -> toks_ctx (every table the kernels read except the bpe tables). */
#ifndef TOKS_COMPILE_H
#define TOKS_COMPILE_H

#include "core.h"

/* rationale: docs/notes/c-core.md §compile.h.1 */
int64_t toks_compile(const struct toks_config *cfg, struct toks_ctx *ctx, toks_arena *parse_ar);

/* rationale: docs/notes/c-core.md §compile.h.2 */
int64_t toks_compile_cls_flags(const toks_tables *t);

/* rationale: docs/notes/c-core.md §compile.h.3 */
const char *toks_tmpl_invalid(const toks_tables *t);

/* K1's index over t->add_entries / add_bytes / add_n (kernels.md §4): shuf [64], idx [2 * 65536 + (2 << TOKS_K1_H4_BITS)]
 * (index2, then h4), single [512], cand [toks_added_cand_words(n2, n4, b4)] (n2 / n4 the tokens of >= 2 / >= 4 bytes, b4
 * the latter's bytes): index2's and h4's lists, then the radix trees of h4's long buckets; sets every add_* table
 * pointer and add_phases */
uint64_t toks_added_cand_words(uint64_t n2, uint64_t n4, uint64_t b4);
void toks_added_index(toks_tables *t, uint8_t *shuf, uint64_t *idx, uint32_t *single, uint32_t *cand);

/* toks_encode_bound's terms for a built context (the proof: compile.c above toks_bound_terms_of): r = num / den ids per
 * input byte and g ids, then which term sets r and its inputs, for the tests */
enum { TOKS_BOUND_TEXT = 0, TOKS_BOUND_PHASE0 = 1, TOKS_BOUND_PHASE1 = 2 };
typedef struct toks_bound_terms {
    uint32_t num, den, g;
    uint32_t x, xd, f;      /* x / xd normalized units per input byte, f ids per unit */
    uint32_t p0, p1;        /* prefix ids the unit after a phase-0 / phase-1 match can add */
    uint32_t l0, l1;        /* the shortest phase-0 match in input bytes, phase-1 in units (0: no such token) */
    uint32_t e0, e1;        /* their entries in t.add_entries */
    uint32_t pfirst;        /* prefix ids of the first unit (in g) */
    uint32_t why;           /* TOKS_BOUND_*: the term that sets r */
} toks_bound_terms;
void toks_bound_terms_of(const struct toks_ctx *c, toks_bound_terms *b);

#endif /* TOKS_COMPILE_H */
