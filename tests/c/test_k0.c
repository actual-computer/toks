/* tests/c/test_k0.c: the selftest kernels on every tier this machine runs, through the geometry suite of
 * SPEC §14.3 (every length 0..255 plus block multiples, end flush against a guard page, start flush plus
 * every alignment 0..63) and through the abi checker. Tiers the machine lacks are skipped and said so. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "../../src/platform/cpu.h"
#include "../common/guard.h"

#if defined(TOKS_ARCH_ARM64)
uint64_t toks_k0_sum_neon(const uint8_t *p, uint64_t n);
uint64_t toks_k0_frame_neon(const uint8_t *p, uint64_t n);
#else
uint64_t toks_k0_sum_avx2(const uint8_t *p, uint64_t n);
uint64_t toks_k0_sum_avx512(const uint8_t *p, uint64_t n);
uint64_t toks_k0_frame_x86(const uint8_t *p, uint64_t n);
#endif

typedef uint64_t (*k0_fn)(const uint8_t *, uint64_t);   /* test code: function pointers are fine here */

typedef struct { const char *name; k0_fn fn; int ok; } tier;

static uint64_t ref_sum(const uint8_t *p, size_t n)
{
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) s += p[i];
    return s;
}

static int failures;

static void check(const tier *t, const uint8_t *p, size_t n, const char *what)
{
    uint64_t want = ref_sum(p, n);
    uint64_t got = t->fn(p, n);
    uint64_t rep = 0;
    uint64_t got2 = toks_abicheck_call((const void *)t->fn, (uint64_t)(uintptr_t)p, (uint64_t)n, &rep);
    if (got != want || got2 != want || rep != 0) {
        if (failures < 20)
            fprintf(stderr, "FAIL %s %s n=%zu: want %" PRIu64 " got %" PRIu64 " / %" PRIu64 " abi report %#" PRIx64 "\n",
                    t->name, what, n, want, got, got2, rep);
        failures++;
    }
}

int main(void)
{
    uint64_t f = toks_cpu_features();
#if defined(TOKS_ARCH_ARM64)
    tier tiers[] = {
        {"neon", toks_k0_sum_neon, TOKS_CPU_HAS(f, TOKS_FEAT_NEON_TIER)},
        {"frame", toks_k0_frame_neon, 1},
    };
#else
    tier tiers[] = {
        {"avx2", toks_k0_sum_avx2, TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER)},
        {"avx512", toks_k0_sum_avx512, TOKS_CPU_HAS(f, TOKS_FEAT_AVX512_TIER)},
        {"frame", toks_k0_frame_x86, 1},
    };
#endif
    size_t ntiers = sizeof tiers / sizeof tiers[0];
    printf("cpu features %#" PRIx64 ", page %zu\n", f, guard_page_size());
    static const size_t extra[] = {256, 320, 511, 512, 4096, 4096 + 63, 65536 + 17};
    uint64_t seed = 0x746f6b73u;
    size_t cases = 0;
    for (size_t ti = 0; ti < ntiers; ti++) {
        const tier *t = &tiers[ti];
        if (!t->ok) { printf("skip %s: not supported by this cpu\n", t->name); continue; }
        for (size_t k = 0; k < 256 + sizeof extra / sizeof extra[0]; k++) {
            size_t n = k < 256 ? k : extra[k - 256];
            guard_buf g = {0};
            uint8_t *p = guard_alloc(&g, n, GUARD_END, 0);
            if (!p) { fprintf(stderr, "guard_alloc failed\n"); return 2; }
            for (size_t i = 0; i < n; i++) p[i] = (uint8_t)guard_rng(&seed);
            check(t, p, n, "end-flush");
            cases++;
            guard_free(&g);
            for (size_t off = 0; off < 64; off++) {
                guard_buf h = {0};
                uint8_t *q = guard_alloc(&h, n, GUARD_START, off);
                if (!q) { fprintf(stderr, "guard_alloc failed\n"); return 2; }
                memset(q - off, 0xEE, off);       /* poison before q: an under-read changes the sum */
                for (size_t i = 0; i < n; i++) q[i] = (uint8_t)guard_rng(&seed);
                check(t, q, n, off ? "start+align" : "start-flush");
                cases++;
                guard_free(&h);
            }
        }
        printf("tier %s: ok\n", t->name);
    }
    printf("%zu cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
