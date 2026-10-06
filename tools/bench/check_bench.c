/* tools/bench/check_bench.c: the segment memo's hit-path verify, alone (kernels.md §7 "the segment memo"; check.c).
 * The library's keyed check (toks_memo_check: the cpu's carry-less multiply; toks_memo_check_with 0: the portable one)
 * against memcmp of the bytes a record held before the check:
 *   alone   one segment in L1, the key hot: best of 5 runs of a fixed count of calls, ns a call and GB/s
 *   replay  <segments> segments of random printable text through records laid out like api.c's (64-aligned: a 32-byte
 *           head, then the bytes (before) or the 16-byte check (now), then the ids), the ids copied out: best of
 *           <passes> passes, ns a segment (the text and the records sit where their size puts them: 2 MB of text and
 *           3.9 MB of records before, 1.9 MB now, at the defaults)
 *
 *   check_bench <tokenizer> [segment bytes (4096)] [ids a segment (975)] [segments (480)] [passes (30)]
 *       e.g. taskset -c 8 ./build/check-bench ~/.cache/toks/tokenizers/llama3
 *   build (as tools/bench/e2e_commits.sh builds e2e.c): $CC -std=c17 -O3 -mcpu=native $(CPPFLAGS) -o build/check-bench
 *       tools/bench/check_bench.c build/<platform>/libtoks.a -lpthread
 */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"
#include "core.h"                                   /* toks_memo_check, toks_memo_check_with: check.c */
#include "cpu.h"
#if defined(TOKS_ARCH_ARM64)
#  define CLMUL TOKS_ARM64_PMULL                    /* the cpu's carry-less multiply, as check.c asks for it */
#else
#  define CLMUL TOKS_X86_PCLMUL
#endif

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t rnd_s = 0x746F6B73u;
static uint64_t rnd(void)                           /* splitmix64 */
{
    uint64_t z = (rnd_s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static volatile uint64_t sink;

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: check_bench <tokenizer> [seg] [ids] [segments] [passes]\n"); return 2; }
    uint64_t seg = argc > 2 ? strtoull(argv[2], NULL, 10) : 4096u, k = argc > 3 ? strtoull(argv[3], NULL, 10) : 975u;
    uint64_t nseg = argc > 4 ? strtoull(argv[4], NULL, 10) : 480u, passes = argc > 5 ? strtoull(argv[5], NULL, 10) : 30u;
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, argv[1], NULL);
    if (r != 0 || ctx->memo_keyed == 0u) { fprintf(stderr, "toks_load: %lld (keyed %u)\n", (long long)r, ctx ? ctx->memo_keyed : 0u); return 1; }
    uint64_t textn = seg * nseg, pad = (seg + 7u) & ~7ull;
    uint64_t rec_old = (32u + pad + 4u * k + 63u) & ~63ull, rec_new = (48u + 4u * k + 63u) & ~63ull;   /* api.c memo_need */
    uint8_t *text = malloc((size_t)textn), *ring_old = malloc((size_t)(rec_old * nseg)), *ring_new = malloc((size_t)(rec_new * nseg));
    uint32_t *out = malloc((size_t)(4u * k * nseg));
    if (text == NULL || ring_old == NULL || ring_new == NULL || out == NULL) { return 1; }
    for (uint64_t i = 0; i < textn; i++) { text[i] = (uint8_t)(32u + rnd() % 95u); }
    for (uint64_t s = 0; s < nseg; s++) {           /* both rings: a head, the bytes or the check, the ids */
        uint8_t *a = ring_old + rec_old * s, *b = ring_new + rec_new * s;
        uint64_t c[2];
        memset(a, 0, 32u), memset(b, 0, 32u);
        memcpy(a + 32u, text + seg * s, (size_t)seg);
        toks_memo_check(ctx, text + seg * s, seg, c);
        memcpy(b + 32u, c, 16u);
        for (uint64_t i = 0; i < k; i++) {
            uint32_t v = (uint32_t)(s * k + i);
            memcpy(a + 32u + pad + 4u * i, &v, 4u), memcpy(b + 48u + 4u * i, &v, 4u);
        }
    }
    printf("CHECK_BENCH seg %" PRIu64 " ids %" PRIu64 " segments %" PRIu64 " text %.2f MB records %.2f MB (bytes) %.2f MB "
           "(check) passes %" PRIu64 " clmul %d\n", seg, k, nseg, (double)textn / 1e6, (double)(rec_old * nseg) / 1e6,
           (double)(rec_new * nseg) / 1e6, passes, TOKS_CPU_HAS(ctx->cpu_features, CLMUL) ? 1 : 0);

    /* alone: one L1 segment */
    for (int f = 0; f < 3; f++) {                   /* memcmp, check, check portable */
        uint64_t calls = f == 2 ? 2000u : 50000u, c[2] = { 0u, 0u };
        double best = 1e30;
        for (int t = 0; t < 5; t++) {
            double t0 = now();
            for (uint64_t i = 0; i < calls; i++) {
                if (f == 0) {
                    c[0] += (uint64_t)memcmp(text, ring_old + 32u, (size_t)seg);
                } else {
                    toks_memo_check_with(ctx, f == 1 ? TOKS_CPU_HAS(ctx->cpu_features, CLMUL) : 0, text, seg, c);
                }
                __asm__ volatile("" ::: "memory");
            }
            double dt = now() - t0;
            best = dt < best ? dt : best;
        }
        sink += c[0];
        printf("alone  %-15s %8.1f ns  %6.2f GB/s\n", f == 0 ? "memcmp" : f == 1 ? "check" : "check-portable",
               best / (double)calls * 1e9, (double)(seg * calls) / best / 1e9);
    }
    /* replay: every segment hits, its ids copied out */
    for (int f = 0; f < 2; f++) {
        double best = 1e30;
        uint64_t bad = 0u;
        for (uint64_t p = 0; p < passes; p++) {
            double t0 = now();
            for (uint64_t s = 0; s < nseg; s++) {
                const uint8_t *g = text + seg * s;
                if (f == 0) {
                    const uint8_t *a = ring_old + rec_old * s;
                    if (memcmp(a + 32u, g, (size_t)seg) != 0) { bad++; continue; }
                    memcpy(out + k * s, a + 32u + pad, (size_t)(4u * k));
                } else {
                    const uint8_t *b = ring_new + rec_new * s;
                    uint64_t c[2], w[2];
                    toks_memo_check(ctx, g, seg, c);
                    memcpy(w, b + 32u, 16u);
                    if (c[0] != w[0] || c[1] != w[1]) { bad++; continue; }
                    memcpy(out + k * s, b + 48u, (size_t)(4u * k));
                }
            }
            __asm__ volatile("" ::: "memory");
            double dt = now() - t0;
            best = dt < best ? dt : best;
        }
        sink += out[k * nseg - 1u];
        printf("replay %-15s %8.1f ns a segment  %8.1f MB/s  bad %" PRIu64 "\n", f == 0 ? "memcmp+copy" : "check+copy",
               best / (double)nseg * 1e9, (double)textn / best / 1e6, bad);
    }
    toks_unload(ctx);
    free(text), free(ring_old), free(ring_new), free(out);
    return 0;
}
