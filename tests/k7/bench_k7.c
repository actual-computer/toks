/*
 * tests/k7/bench_k7.c: K7 alone (the rent rule's isolated cell, SPEC §10.5): the spm scan of a corpus, unit by
 * unit (chunks cut as tools/bench/e2e.c cuts them; 0 = the whole text as one unit), window by window as
 * toks_spm_encode walks it: the c twin alone vs the tier's asm part (+ the twin on the windows it leaves). Word
 * starts and one-char steps are checked equal outside the timers; abba rounds, the best of each side.
 *
 *   bench_k7 <tokenizer> <chunk_bytes|0> <rounds> <file>...
 */
#define _POSIX_C_SOURCE 200809L
#include "core.h"
#include "kernels.h"
#include "spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if TOKS_HAVE_K7_SPM_NEON
#define PART toks_k7_spm_neon
#elif TOKS_HAVE_K7_SPM_AVX2
#define PART toks_k7_spm_avx2
#endif

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint32_t ONE[64];
static uint64_t taken;

/* one unit: a digest of its word starts (and of one[] at the one-char steps when check) */
static uint64_t unit(const toks_spm *s, const uint8_t *t, uint64_t n, int part, int check)
{
    toks_k7_args a = { t, n, 0u, 0x001FFFFFu | (255u << 21), ONE, 0u, 1u, 0u };
    uint64_t h = 0, g = 0;                                  /* word starts, one-char steps (windows differ) */
    while (a.pos < n) {
        uint64_t base = a.pos, m = 0;
#ifdef PART
        if (part) { m = PART(s, &a); taken += a.pos != base; }
#else
        (void)part;
#endif
        if (a.pos == base) { m = toks_k7_spm_c(s, &a); }
        if (!check) {
            h += m;
            continue;
        }
        for (uint64_t b = m; b != 0u; b &= b - 1u) { h = h * 0x9E3779B97F4A7C15ull + base + (uint64_t)__builtin_ctzll(b); }
        {
            for (uint64_t i = base; i < a.pos; i++) {
                if (t[i] >= 0xC2u && toks_utf8_len(t + i, n - i) >= 2u) { g = g * 31u + ONE[i - base] + i; }
            }
        }
    }
    return h ^ (g * 0xC2B2AE3D27D4EB4Full);
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: bench_k7 <tokenizer> <chunk|0> <rounds> <file>...\n"); return 2; }
    uint64_t chunk = strtoull(argv[2], NULL, 10);
    int rounds = atoi(argv[3]);
    uint8_t *buf = NULL;
    uint64_t n = 0;
    for (int i = 4; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (f == NULL) { perror(argv[i]); return 1; }
        fseek(f, 0, SEEK_END);
        long m = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf = realloc(buf, n + (uint64_t)m + 1);
        if (fread(buf + n, 1, (size_t)m, f) != (size_t)m) { return 1; }
        fclose(f);
        n += (uint64_t)m;
    }
    uint64_t *beg = malloc((n / (chunk ? chunk : n) + 2) * 8), *len = malloc((n / (chunk ? chunk : n) + 2) * 8), nc = 0;
    for (uint64_t p = 0; p < n;) {
        uint64_t e = n;
        if (chunk != 0 && p + chunk < n) {
            e = p + chunk;
            while (e < n && buf[e - 1] != '\n') { e++; }
        }
        beg[nc] = p;
        len[nc++] = e - p;
        p = e;
    }
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, argv[1], NULL) != 0 || ctx->spm == NULL) { fprintf(stderr, "not an spm tokenizer\n"); return 1; }
    const toks_spm *s = ctx->spm;
    uint64_t h0 = 0, h1 = 0;
    for (uint64_t k = 0; k < nc; k++) {
        h0 += unit(s, buf + beg[k], len[k], 0, 1);
        h1 += unit(s, buf + beg[k], len[k], 1, 1);
    }
    if (h0 != h1) { printf("K7 MISMATCH twin %016llx part %016llx\n", (unsigned long long)h0, (unsigned long long)h1); }
    uint64_t best[2] = { UINT64_MAX, UINT64_MAX }, sink = 0;
    for (int r = 0; r < rounds; r++) {
        for (int q = 0; q < 4; q++) {                       /* abba */
            int side = (q == 1 || q == 2);
            uint64_t a = now_ns();
            for (uint64_t k = 0; k < nc; k++) { sink += unit(s, buf + beg[k], len[k], side, 0); }
            uint64_t d = now_ns() - a;
            if (d < best[side]) { best[side] = d; }
        }
    }
    taken = 0;
    for (uint64_t k = 0; k < nc; k++) { (void)unit(s, buf + beg[k], len[k], 1, 0); }
    printf("K7 chunk %llu bytes %llu units %llu: twin %.1f MB/s (%.3f ns/B), part %.1f MB/s (%.3f ns/B), part/twin %.2fx;"
           " windows in the part %llu (sink %llx)\n", (unsigned long long)chunk, (unsigned long long)n,
           (unsigned long long)nc, (double)n / (double)best[0] * 1e3, (double)best[0] / (double)n,
           (double)n / (double)best[1] * 1e3, (double)best[1] / (double)n, (double)best[0] / (double)best[1],
           (unsigned long long)taken, (unsigned long long)(sink & 0xFFFF));
    toks_unload(ctx);
    return h0 != h1;
}
