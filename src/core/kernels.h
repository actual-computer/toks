/* kernels.h: the kernel prototypes of every tier and the driver's call sites (docs/notes/c-core.md §kernels.h.1) */
#ifndef TOKS_KERNELS_H
#define TOKS_KERNELS_H

#include "core.h"

#ifndef TOKS_HAVE_K1_NEON
#define TOKS_HAVE_K1_NEON 0
#endif
#ifndef TOKS_HAVE_K3_CL100K_NEON
#define TOKS_HAVE_K3_CL100K_NEON 0
#endif
#ifndef TOKS_HAVE_K3_O200K_NEON
#define TOKS_HAVE_K3_O200K_NEON 0
#endif
#ifndef TOKS_HAVE_K3_DSV3_NEON
#define TOKS_HAVE_K3_DSV3_NEON 0
#endif
#ifndef TOKS_HAVE_K5_NEON
#define TOKS_HAVE_K5_NEON 0
#endif
#ifndef TOKS_HAVE_K1_AVX2
#define TOKS_HAVE_K1_AVX2 0
#endif
#ifndef TOKS_HAVE_K3_CL100K_AVX2
#define TOKS_HAVE_K3_CL100K_AVX2 0
#endif
#ifndef TOKS_HAVE_K3_O200K_AVX2
#define TOKS_HAVE_K3_O200K_AVX2 0
#endif
#ifndef TOKS_HAVE_K3_DSV3_AVX2
#define TOKS_HAVE_K3_DSV3_AVX2 0
#endif
#ifndef TOKS_HAVE_K5_AVX2
#define TOKS_HAVE_K5_AVX2 0
#endif
#ifndef TOKS_HAVE_K1_AVX512
#define TOKS_HAVE_K1_AVX512 0
#endif
#ifndef TOKS_HAVE_K3_CL100K_AVX512
#define TOKS_HAVE_K3_CL100K_AVX512 0
#endif
#ifndef TOKS_HAVE_K3_O200K_AVX512
#define TOKS_HAVE_K3_O200K_AVX512 0
#endif
#ifndef TOKS_HAVE_K3_DSV3_AVX512
#define TOKS_HAVE_K3_DSV3_AVX512 0
#endif
#ifndef TOKS_HAVE_K5_AVX512
#define TOKS_HAVE_K5_AVX512 0
#endif
#ifndef TOKS_HAVE_K7_SPM_NEON
#define TOKS_HAVE_K7_SPM_NEON 0
#endif
#ifndef TOKS_HAVE_K7_SPM_AVX2
#define TOKS_HAVE_K7_SPM_AVX2 0
#endif

/* a tier is offered when at least one of its own kernels is built (load.c) */
#define TOKS_HAVE_NEON   (TOKS_HAVE_K1_NEON | TOKS_HAVE_K3_CL100K_NEON | TOKS_HAVE_K3_O200K_NEON | TOKS_HAVE_K3_DSV3_NEON | \
                          TOKS_HAVE_K5_NEON | TOKS_HAVE_K7_SPM_NEON)
#define TOKS_HAVE_AVX2   (TOKS_HAVE_K1_AVX2 | TOKS_HAVE_K3_CL100K_AVX2 | TOKS_HAVE_K3_O200K_AVX2 | TOKS_HAVE_K3_DSV3_AVX2 | \
                          TOKS_HAVE_K5_AVX2 | TOKS_HAVE_K7_SPM_AVX2)
#define TOKS_HAVE_AVX512 (TOKS_HAVE_K1_AVX512 | TOKS_HAVE_K3_CL100K_AVX512 | TOKS_HAVE_K3_O200K_AVX512 | \
                          TOKS_HAVE_K3_DSV3_AVX512 | TOKS_HAVE_K5_AVX512)

/* rationale: docs/notes/c-core.md §kernels.h.2 */
#define TOKS_RUN(K, tier)                                                                             \
    ((tier) == TOKS_TIER_NEON && TOKS_HAVE_##K##_NEON ? TOKS_TIER_NEON :                              \
     (tier) == TOKS_TIER_AVX512 && TOKS_HAVE_##K##_AVX512 ? TOKS_TIER_AVX512 :                        \
     ((tier) == TOKS_TIER_AVX512 || (tier) == TOKS_TIER_AVX2) && TOKS_HAVE_##K##_AVX2 ? TOKS_TIER_AVX2 : \
     TOKS_TIER_SCALAR)

/* ---- prototypes (K6 is called by K5 only; its merge loop by spm's model too) ------------------------- */
uint64_t toks_k1_added_find_c(const toks_tables *t, toks_k1_args *a);
uint64_t toks_k1_added_find_neon(const toks_tables *t, toks_k1_args *a);
uint64_t toks_k1_added_find_avx2(const toks_tables *t, toks_k1_args *a);
uint64_t toks_k1_added_find_avx512(const toks_tables *t, toks_k1_args *a);

uint64_t toks_k3_scan_cl100k_c(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_cl100k_neon(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_cl100k_avx2(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_cl100k_avx512(const toks_tables *t, toks_k3_args *a);

uint64_t toks_k3_scan_o200k_c(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_o200k_neon(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_o200k_avx2(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_o200k_avx512(const toks_tables *t, toks_k3_args *a);

uint64_t toks_k3_scan_dsv3_c(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_dsv3_neon(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_dsv3_avx2(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_dsv3_avx512(const toks_tables *t, toks_k3_args *a);

uint64_t toks_k5_encode_c(const toks_tables *t, toks_k5_args *a);
uint64_t toks_k5_encode_neon(const toks_tables *t, toks_k5_args *a);
uint64_t toks_k5_encode_avx2(const toks_tables *t, toks_k5_args *a);
uint64_t toks_k5_encode_avx512(const toks_tables *t, toks_k5_args *a);

uint64_t toks_k6_bpe_c(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k6_bpe_neon(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k6_bpe_avx2(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k6_bpe_avx512(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k6_merge_neon(const toks_tables *t, toks_k6_args *a);    /* given symbols (spm's model): k6_*.S */
uint64_t toks_k6_merge_avx2(const toks_tables *t, toks_k6_args *a);

/* ---- the K3 c twins' readers (kernels.md §2-3; k3_c.c, k3_o200k_c.c) ---------------------------------- */

/* rationale: docs/notes/c-core.md §kernels.h.3 */
static inline uint8_t toks_k3_atom(const toks_tables *t, const uint8_t *p, uint64_t avail, uint32_t *cp,
                                   uint32_t *k, uint8_t strip)
{
    uint32_t n = toks_utf8_len(p, avail);
    uint8_t c;
    *cp = p[0];
    *k = n != 0u ? n : 1u;
    if (n <= 1u) { return n == 0u ? (uint8_t)TOKS_C_P : t->cls_ascii[p[0]]; }
    *cp = toks_cp_decode(p, n);
    c = toks_cls_cp(t, *cp);
    return (c & TOKS_C_HAN) != 0u ? (uint8_t)(c & ~strip) : c;
}

/* the end of the maximal run of atoms of base class b from `from` (an atom start; from itself when the atom
 * there is another class). Bound: len - from, >= 1 byte per atom. */
static inline uint64_t toks_k3_run(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t from,
                                   uint8_t b, uint8_t strip)
{
    uint64_t e = from;
    while (e < len) {                                   /* bound: len - e (>= 1 byte per atom) */
        uint32_t cp, k;
        if ((toks_k3_atom(t, text + e, len - e, &cp, &k, strip) & TOKS_C_BASE_MASK) != b) { break; }
        e += k;
    }
    return e;
}

/* \p{N}{1,3} from p1, the second atom of a piece that starts with an N atom: the end of up to two more */
static inline uint64_t toks_k3_n13(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t p1,
                                   uint8_t strip)
{
    uint64_t e = p1;
    for (uint32_t c = 1u; c < 3u && e < len; c++) {    /* bound: 3 atoms */
        uint32_t cp, k;
        if ((toks_k3_atom(t, text + e, len - e, &cp, &k, strip) & TOKS_C_BASE_MASK) != TOKS_C_N) { break; }
        e += k;
    }
    return e;
}

/* A9 (TOKS_TP_NL_CUT): a match of (?:\r?\n)+ starts at q (0 < q < len), so the regex's input ends there */
static inline int toks_k3_nlcut(const uint8_t *x, uint64_t len, uint64_t q)
{
    return x[q - 1u] != 0x0Au && (x[q] == 0x0Au ? x[q - 1u] != 0x0Du : x[q] == 0x0Du && q + 1u < len && x[q + 1u] == 0x0Au);
}

/* rationale: docs/notes/c-core.md §kernels.h.4; p: cl100k's DIGIT_CUT (A8), NL_CUT (A9) and GB_SP (A10) bits, and
 * bit 7: an atom with class bit 7 after the run ends the input too (dsv3's D5: its CJK bit) */
static inline uint64_t toks_k3_ws(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos,
                                  int ws_nl, uint8_t strip, uint32_t p)
{
    uint64_t e = pos, last_nl = 0u, last_start = pos;
    uint8_t c = 0u, b = 0u;                             /* the class and base of the atom after R (when e < len) */
    while (e < len) {                                   /* bound: len - e (>= 1 byte per atom) */
        uint32_t cp, k;
        if ((p & TOKS_TP_NL_CUT) != 0u && e > pos && toks_k3_nlcut(text, len, e)) { len = e; break; }   /* A9 */
        c = toks_k3_atom(t, text + e, len - e, &cp, &k, strip);
        b = (uint8_t)(c & TOKS_C_BASE_MASK);
        if (b != TOKS_C_WS && b != TOKS_C_NL) { break; }
        if (b == TOKS_C_NL) { last_nl = e + k; }
        last_start = e;
        e += k;
    }
    if (ws_nl && last_nl != 0u) { return last_nl; }
    return e == len || last_start == pos || ((p & TOKS_TP_DIGIT_CUT) != 0u && b == TOKS_C_N) || (c & p & 0x80u) != 0u ||
           ((p & TOKS_TP_GB_SP) != 0u && text[last_start] != 0x20u) ? e : last_start;
}

/* ---- the call sites: TOKS_RUN picks the version, each case is a direct call ------------------- */

/* rationale: docs/notes/c-core.md §kernels.h.5 */
#if defined(__aarch64__) || defined(_M_ARM64)
#define TOKS_K3_MIN_A(tw) ((tw) == toks_k3_scan_dsv3_c ? 16u : 7u)
#define TOKS_K3_MIN_N(tw) ((tw) == toks_k3_scan_dsv3_c ? 32u : 24u)
#else
#define TOKS_K3_MIN_A(tw) ((tw) == toks_k3_scan_dsv3_c ? 7u : 4u)
#define TOKS_K3_MIN_N(tw) 16u
#endif
typedef uint64_t (*toks_k3_fn)(const toks_tables *, toks_k3_args *);
static inline __attribute__((always_inline)) int toks_k3_short(toks_k3_fn twin, const toks_tables *t, toks_k3_args *a,
                                                               uint64_t *n)
{
    uint64_t r = a->len - a->pos;
    const uint8_t *p = a->text + a->pos;
    if (r >= TOKS_K3_MIN_N(twin)) { return 0; }
    if (r == 1u || (r - 2u < 3u && p[0] >= 0xC0u && toks_utf8_len(p, r) == r)) {
        toks_st32(a->ends, (uint32_t)a->len);       /* one atom is one piece (cap >= 1) */
        a->pos = a->len;
        a->n = 1u;
        *n = 1u;
        return 1;
    }
    if (r >= TOKS_K3_MIN_A(twin)) {             /* the non-ascii bytes of p[0, 8) (r < 8: of p[0, 4), p[r - 4, r)) */
        uint64_t w = (uint64_t)toks_ld32(p) | (uint64_t)toks_ld32(p + (r < 8u ? r - 4u : 4u)) << 32;
        if (((w >> 7 & 0x0101010101010101u) * 0x0101010101010101u) >> 56 < 4u) { return 0; }
    }
    *n = twin(t, a);
    return 1;
}

static inline __attribute__((always_inline))
uint64_t toks_k3_parts(toks_k3_fn part, toks_k3_fn twin, const toks_tables *t, toks_k3_args *a)
{
    uint64_t rsv = a->rsv, n = part(t, a), m, cap;
    if (a->rsv != 0u) {
        toks_k3_args b = *a;
        b.ends += n;
        b.cap -= n;
        while (b.rsv != 0u && b.cap != 0u && b.pos < b.len) {   /* bound: pieces (each round of twins writes >= 1) */
            if (b.pos < b.rsv) {                    /* the stretch: the twin, a piece a call */
                cap = b.cap;
                b.cap = 1u;
                n += twin(t, &b);
                b.ends++;
                b.cap = cap - 1u;
            } else if (toks_k3_short(twin, t, &b, &m)) {
                n += m;                             /* a short rest: as at the call site, no part entry */
                break;
            } else {
                m = part(t, &b);
                n += m;
                b.ends += m;
                b.cap -= m;
            }
        }
        a->n = n;
        a->pos = b.pos;
    }
    a->rsv = rsv;
    return n;
}

static inline __attribute__((always_inline))
uint64_t toks_k3_tier(toks_k3_fn part, toks_k3_fn twin, const toks_tables *t, toks_k3_args *a)
{
    uint64_t n;
    return toks_k3_short(twin, t, a, &n) ? n : toks_k3_parts(part, twin, t, a);
}

static inline uint64_t toks_k1(const toks_tables *t, toks_k1_args *a, uint32_t tier)
{
    switch (TOKS_RUN(K1, tier)) {
#if TOKS_HAVE_K1_NEON
    case TOKS_TIER_NEON: return toks_k1_added_find_neon(t, a);
#endif
#if TOKS_HAVE_K1_AVX2
    case TOKS_TIER_AVX2: return toks_k1_added_find_avx2(t, a);
#endif
#if TOKS_HAVE_K1_AVX512
    case TOKS_TIER_AVX512: return toks_k1_added_find_avx512(t, a);
#endif
    default: return toks_k1_added_find_c(t, a);
    }
}

static inline uint64_t toks_k3(const toks_tables *t, toks_k3_args *a, uint32_t tier)
{
    uint64_t n;
    if (toks_k3_short(toks_k3_scan_cl100k_c, t, a, &n)) { return n; }   /* every tier */
    switch (TOKS_RUN(K3_CL100K, tier)) {
#if TOKS_HAVE_K3_CL100K_NEON
    case TOKS_TIER_NEON: return toks_k3_parts(toks_k3_scan_cl100k_neon, toks_k3_scan_cl100k_c, t, a);
#endif
#if TOKS_HAVE_K3_CL100K_AVX2
    case TOKS_TIER_AVX2: return toks_k3_parts(toks_k3_scan_cl100k_avx2, toks_k3_scan_cl100k_c, t, a);
#endif
#if TOKS_HAVE_K3_CL100K_AVX512
    case TOKS_TIER_AVX512: return toks_k3_parts(toks_k3_scan_cl100k_avx512, toks_k3_scan_cl100k_c, t, a);
#endif
    default: return toks_k3_scan_cl100k_c(t, a);
    }
}

/* K3 of the o200k template (docs/templates/o200k.md); api.c's run_text picks it for t->tmpl == TOKS_TMPL_O200K */
static inline uint64_t toks_k3_o200k(const toks_tables *t, toks_k3_args *a, uint32_t tier)
{
    uint64_t n;
    if (toks_k3_short(toks_k3_scan_o200k_c, t, a, &n)) { return n; }   /* every tier */
    switch (TOKS_RUN(K3_O200K, tier)) {
#if TOKS_HAVE_K3_O200K_NEON
    case TOKS_TIER_NEON: return toks_k3_parts(toks_k3_scan_o200k_neon, toks_k3_scan_o200k_c, t, a);
#endif
#if TOKS_HAVE_K3_O200K_AVX2
    case TOKS_TIER_AVX2: return toks_k3_parts(toks_k3_scan_o200k_avx2, toks_k3_scan_o200k_c, t, a);
#endif
#if TOKS_HAVE_K3_O200K_AVX512
    case TOKS_TIER_AVX512: return toks_k3_parts(toks_k3_scan_o200k_avx512, toks_k3_scan_o200k_c, t, a);
#endif
    default: return toks_k3_scan_o200k_c(t, a);
    }
}

/* K3 of the dsv3 template (docs/templates/dsv3.md); api.c's run_text picks it for t->tmpl == TOKS_TMPL_DSV3 */
static inline uint64_t toks_k3_dsv3(const toks_tables *t, toks_k3_args *a, uint32_t tier)
{
    uint64_t n;
    if (toks_k3_short(toks_k3_scan_dsv3_c, t, a, &n)) { return n; }   /* every tier */
    switch (TOKS_RUN(K3_DSV3, tier)) {
#if TOKS_HAVE_K3_DSV3_NEON
    case TOKS_TIER_NEON: return toks_k3_parts(toks_k3_scan_dsv3_neon, toks_k3_scan_dsv3_c, t, a);
#endif
#if TOKS_HAVE_K3_DSV3_AVX2
    case TOKS_TIER_AVX2: return toks_k3_parts(toks_k3_scan_dsv3_avx2, toks_k3_scan_dsv3_c, t, a);
#endif
#if TOKS_HAVE_K3_DSV3_AVX512
    case TOKS_TIER_AVX512: return toks_k3_parts(toks_k3_scan_dsv3_avx512, toks_k3_scan_dsv3_c, t, a);
#endif
    default: return toks_k3_scan_dsv3_c(t, a);
    }
}

static inline uint64_t toks_k5(const toks_tables *t, toks_k5_args *a, uint32_t tier)
{
    switch (TOKS_RUN(K5, tier)) {
#if TOKS_HAVE_K5_NEON
    case TOKS_TIER_NEON: return toks_k5_encode_neon(t, a);
#endif
#if TOKS_HAVE_K5_AVX2
    case TOKS_TIER_AVX2: return toks_k5_encode_avx2(t, a);
#endif
#if TOKS_HAVE_K5_AVX512
    case TOKS_TIER_AVX512: return toks_k5_encode_avx512(t, a);
#endif
    default: return toks_k5_encode_c(t, a);
    }
}

#endif /* TOKS_KERNELS_H */
