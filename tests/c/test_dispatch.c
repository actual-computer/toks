/*
 * test_dispatch.c: kernels.h's tier rule (TOKS_RUN) and its call sites, with stand-in kernels.
 *
 * The rule: a context's tier runs each kernel's version for that tier when it is built, else the best
 * version built for a lower tier the cpu also runs (avx512 -> avx2 on x86-64), else the c twin; never a
 * higher tier's, never another isa's. TOKS_RUN is checked against a restatement of that rule on all 8
 * sets of built versions and every tier value; then toks_k1 / toks_k3 / toks_k5 run with three different
 * sets (the flags below replace the build's) and must reach the stand-in the rule names: K5 = {avx2}
 * is the state master ships for x86-64 (the avx512 tier must run the avx2 K5), K1 = {neon, avx512} has
 * no avx2 version (the avx2 tier runs the c twin), K3 = {avx2, avx512} has both (each tier its own).
 * Nothing from libtoks is linked: every kernel the three call sites reference is defined here.
 */
#undef TOKS_HAVE_K1_NEON
#undef TOKS_HAVE_K1_AVX2
#undef TOKS_HAVE_K1_AVX512
#undef TOKS_HAVE_K3_CL100K_NEON
#undef TOKS_HAVE_K3_CL100K_AVX2
#undef TOKS_HAVE_K3_CL100K_AVX512
#undef TOKS_HAVE_K5_NEON
#undef TOKS_HAVE_K5_AVX2
#undef TOKS_HAVE_K5_AVX512
#define TOKS_HAVE_K1_NEON 1
#define TOKS_HAVE_K1_AVX2 0
#define TOKS_HAVE_K1_AVX512 1
#define TOKS_HAVE_K3_CL100K_NEON 0
#define TOKS_HAVE_K3_CL100K_AVX2 1
#define TOKS_HAVE_K3_CL100K_AVX512 1
#define TOKS_HAVE_K5_NEON 0
#define TOKS_HAVE_K5_AVX2 1
#define TOKS_HAVE_K5_AVX512 0
#include "../../src/core/kernels.h"

#include <stdio.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* the rule, restated: set bit 0 = a neon version is built, bit 1 = avx2, bit 2 = avx512 */
static uint32_t rule(uint32_t set, uint32_t tier)
{
    if (tier == TOKS_TIER_NEON) { return (set & 1u) ? TOKS_TIER_NEON : TOKS_TIER_SCALAR; }
    if (tier == TOKS_TIER_AVX512 && (set & 4u)) { return TOKS_TIER_AVX512; }
    if ((tier == TOKS_TIER_AVX512 || tier == TOKS_TIER_AVX2) && (set & 2u)) { return TOKS_TIER_AVX2; }
    return TOKS_TIER_SCALAR;
}

/* eight pseudo-kernels S0..S7: Sn has the versions of set n */
#define TOKS_HAVE_S0_NEON 0
#define TOKS_HAVE_S0_AVX2 0
#define TOKS_HAVE_S0_AVX512 0
#define TOKS_HAVE_S1_NEON 1
#define TOKS_HAVE_S1_AVX2 0
#define TOKS_HAVE_S1_AVX512 0
#define TOKS_HAVE_S2_NEON 0
#define TOKS_HAVE_S2_AVX2 1
#define TOKS_HAVE_S2_AVX512 0
#define TOKS_HAVE_S3_NEON 1
#define TOKS_HAVE_S3_AVX2 1
#define TOKS_HAVE_S3_AVX512 0
#define TOKS_HAVE_S4_NEON 0
#define TOKS_HAVE_S4_AVX2 0
#define TOKS_HAVE_S4_AVX512 1
#define TOKS_HAVE_S5_NEON 1
#define TOKS_HAVE_S5_AVX2 0
#define TOKS_HAVE_S5_AVX512 1
#define TOKS_HAVE_S6_NEON 0
#define TOKS_HAVE_S6_AVX2 1
#define TOKS_HAVE_S6_AVX512 1
#define TOKS_HAVE_S7_NEON 1
#define TOKS_HAVE_S7_AVX2 1
#define TOKS_HAVE_S7_AVX512 1

/* ---- stand-in kernels: record which kernel and which version ran ---- */
static uint32_t ran_kernel, ran_tier;

#define STAND_IN(fn, args_t, k, v)                                      \
    uint64_t fn(const toks_tables *t, args_t *a)                        \
    {                                                                   \
        (void)t; (void)a; ran_kernel = (k); ran_tier = (v); return (k); \
    }
STAND_IN(toks_k1_added_find_c, toks_k1_args, 1u, TOKS_TIER_SCALAR)
STAND_IN(toks_k1_added_find_neon, toks_k1_args, 1u, TOKS_TIER_NEON)
STAND_IN(toks_k1_added_find_avx512, toks_k1_args, 1u, TOKS_TIER_AVX512)
STAND_IN(toks_k3_scan_cl100k_c, toks_k3_args, 3u, TOKS_TIER_SCALAR)
STAND_IN(toks_k3_scan_cl100k_avx2, toks_k3_args, 3u, TOKS_TIER_AVX2)
STAND_IN(toks_k3_scan_cl100k_avx512, toks_k3_args, 3u, TOKS_TIER_AVX512)
STAND_IN(toks_k5_encode_c, toks_k5_args, 5u, TOKS_TIER_SCALAR)
STAND_IN(toks_k5_encode_avx2, toks_k5_args, 5u, TOKS_TIER_AVX2)

int main(void)
{
    /* every context tier, plus values no context holds (AUTO, unknown) */
    static const uint32_t TIERS[] = { TOKS_TIER_SCALAR, TOKS_TIER_NEON, TOKS_TIER_AVX2, TOKS_TIER_AVX512,
                                      TOKS_TIER_AUTO, 9u };
    static const char *NAME[] = { "auto", "scalar", "neon", "avx2", "avx512" };
    for (size_t i = 0; i < sizeof TIERS / sizeof TIERS[0]; i++) {
        uint32_t t = TIERS[i];
        uint32_t got[8] = { TOKS_RUN(S0, t), TOKS_RUN(S1, t), TOKS_RUN(S2, t), TOKS_RUN(S3, t),
                            TOKS_RUN(S4, t), TOKS_RUN(S5, t), TOKS_RUN(S6, t), TOKS_RUN(S7, t) };
        for (uint32_t s = 0; s < 8u; s++) {
            CHECK(got[s] == rule(s, t), "TOKS_RUN(set %u, tier %u) = %u, want %u", s, t, got[s], rule(s, t));
        }
        toks_k1_args a1;
        static const uint8_t text3[64];
        uint32_t ends3[8];
        toks_k3_args a3 = { text3, 64u, 0u, ends3, 8u, 0u, 0u, 0u };   /* asm K3 runs through toks_k3_tier */
        toks_k5_args a5;
        ran_kernel = ran_tier = 99u;
        CHECK(toks_k1(NULL, &a1, t) == 1u && ran_kernel == 1u && ran_tier == rule(5u, t),
              "toks_k1 at tier %u ran version %u, want %u", t, ran_tier, rule(5u, t));
        ran_kernel = ran_tier = 99u;
        CHECK(toks_k3(NULL, &a3, t) == 3u && ran_kernel == 3u && ran_tier == rule(6u, t),
              "toks_k3 at tier %u ran version %u, want %u", t, ran_tier, rule(6u, t));
        ran_kernel = ran_tier = 99u;
        CHECK(toks_k5(NULL, &a5, t) == 5u && ran_kernel == 5u && ran_tier == rule(2u, t),
              "toks_k5 at tier %u ran version %u, want %u", t, ran_tier, rule(2u, t));
        if (t >= TOKS_TIER_SCALAR && t <= TOKS_TIER_AVX512) {
            printf("tier %-6s: k1 %s, k3 %s, k5 %s\n", NAME[t], NAME[rule(5u, t)], NAME[rule(6u, t)], NAME[rule(2u, t)]);
        }
    }
    /* the aggregate the loader offers a tier by: one of the tier's own kernels is built */
    CHECK(TOKS_HAVE_NEON == 1 && TOKS_HAVE_AVX2 == 1 && TOKS_HAVE_AVX512 == 1, "TOKS_HAVE_<TIER> aggregates");
    printf("test_dispatch: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
