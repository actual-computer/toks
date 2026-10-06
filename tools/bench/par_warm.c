/* tools/bench/par_warm.c: toks_par's default scratch, first pass and warm, one participant (the caller: n_threads 1),
 * scratch flags 0, the corpus cut like tools/bench/e2e.c's chunks (>= CHUNK bytes, to the next '\n') and encoded as one
 * toks_par_encode_batch per pass. Each rep: a fresh pool, its first pass (the scratch is made on the first unit:
 * toks_plat_arena + toks_scratch_init), then WARM passes on the same pool (the scratch keeps its caches). Prints the
 * best first and the best warm MB/s over the reps, an fnv digest of the ids (every pass of the run must give the same
 * digest, else it fails), and thp_kb (AnonHugePages of the process, linux) read while the last pool lives.
 * The digest is checked only across this run's own passes: for an A/B of two libraries, the evidence that both give
 * the same ids is that their digests are equal (compare them in the log). A quick look for A/B pairs, not a cell of
 * the measurement protocol; built by hand against a library:
 *   clang -std=c17 -O3 -march=native -Iinclude -o build/par_warm tools/bench/par_warm.c build/<os>-<isa>/libtoks.a -lpthread
 *   par_warm TOKENIZER CHUNK REPS WARM FILE...
 */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint64_t fnv(const uint32_t *p, uint64_t n, uint64_t h)
{
    for (uint64_t i = 0; i < n; i++) { h = (h ^ p[i]) * 0x100000001b3ull; }
    return h;
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: par_warm TOKENIZER CHUNK REPS WARM FILE...\n"); return 2; }
    uint64_t chunk = strtoull(argv[2], NULL, 10);
    int reps = atoi(argv[3]), warm = atoi(argv[4]);
    uint8_t *buf = NULL;
    uint64_t n = 0;
    for (int i = 5; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (f == NULL) { perror(argv[i]); free(buf); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *nb = realloc(buf, (size_t)(n + (uint64_t)sz));
        if (nb == NULL || fread(nb + n, 1, (size_t)sz, f) != (size_t)sz) {
            fprintf(stderr, "read %s\n", argv[i]);
            free(nb != NULL ? nb : buf);
            fclose(f);
            return 1;
        }
        buf = nb;
        n += (uint64_t)sz;
        fclose(f);
    }
    uint64_t cap = n / (chunk ? chunk : n) + 2, ni = 0, p = 0;
    toks_par_item *it = calloc((size_t)cap, sizeof *it);
    uint32_t *ids = malloc((size_t)(n + 4u * cap) * 4u);
    if (it == NULL || ids == NULL) { fprintf(stderr, "out of memory\n"); return 1; }
    uint64_t off = 0;
    while (p < n) {
        uint64_t e = n;
        if (chunk != 0 && p + chunk < n) {
            e = p + chunk;
            while (e < n && buf[e - 1] != '\n') { e++; }
        }
        it[ni].text = buf + p;
        it[ni].len = e - p;
        it[ni].out = ids + off;
        it[ni].cap = e - p + 4u;
        off += e - p + 4u;
        ni++;
        p = e;
    }
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, argv[1], NULL) != 0) { fprintf(stderr, "load %s\n", argv[1]); return 1; }
    uint64_t best_first = UINT64_MAX, best_warm = UINT64_MAX, dig0 = 0;
    long thp = -1;
    for (int r = 0; r < reps; r++) {
        toks_par *par = NULL;
        if (toks_par_create(&par, ctx, 1u, 0u) != 0) { fprintf(stderr, "toks_par_create\n"); return 1; }
        for (int w = 0; w <= warm; w++) {
            uint64_t a = now_ns();
            if (toks_par_encode_batch(par, it, ni, 0u) != 0) { fprintf(stderr, "batch\n"); return 1; }
            uint64_t t = now_ns() - a, dig = 0xcbf29ce484222325ull;
            for (uint64_t k = 0; k < ni; k++) {
                if (it[k].n < 0) { fprintf(stderr, "item %llu: %lld\n", (unsigned long long)k, (long long)it[k].n); return 1; }
                dig = fnv(it[k].out, (uint64_t)it[k].n, dig);
            }
            if (dig0 == 0) { dig0 = dig; }
            if (dig != dig0) { fprintf(stderr, "ids differ between passes\n"); return 1; }
            if (w == 0 && t < best_first) { best_first = t; }
            if (w > 0 && t < best_warm) { best_warm = t; }
        }
        if (r + 1 == reps) {                            /* the pool's scratch still mapped */
            FILE *f = fopen("/proc/self/smaps_rollup", "r");
            char ln[256];
            while (f != NULL && fgets(ln, sizeof ln, f) != NULL) {
                if (strncmp(ln, "AnonHugePages:", 14) == 0) { thp = atol(ln + 14); }
            }
            if (f != NULL) { fclose(f); }
        }
        toks_par_destroy(par);
    }
    printf("PAR tool=toks_par threads=1 chunk=%llu bytes=%llu items=%llu digest=%016llx first_s=%.6f warm_s=%.6f "
           "first_mbs=%.1f warm_mbs=%.1f thp_kb=%ld\n", (unsigned long long)chunk, (unsigned long long)n,
           (unsigned long long)ni, (unsigned long long)dig0, (double)best_first * 1e-9, (double)best_warm * 1e-9,
           (double)n / ((double)best_first * 1e-9) / 1e6, (double)n / ((double)best_warm * 1e-9) / 1e6, thp);
    toks_unload(ctx);
    free(ids);
    free(it);
    free(buf);
    return 0;
}
