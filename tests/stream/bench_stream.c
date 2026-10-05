/*
 * bench_stream.c: batch decode and stream push speed through the public abi only (include/toks.h), so the
 * same file times any revision of the library (a library without stream decode reports it unsupported).
 *
 *   clang -std=c17 -O2 -Iinclude tests/stream/bench_stream.c build/<os>-<isa>/libtoks.a -o build/bench_stream
 *   build/bench_stream <tokenizer.json> <text file> [reps]
 *
 * The text is encoded once (mode ALL, no post-processor); then, best of reps, each rep timing in turn:
 * toks_decode of every id, pushes of one id (the serving path: one push per generated token, out
 * advancing through one buffer), pushes of 16 ids. Every output is checked equal to the batch decode
 * outside the timer. MB/s counts decoded bytes. Not part of make test: run it on a lab host (pin it:
 * taskset -c 5 on a gb10 host is an X925 core) and record the load.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* one stream decode of ids[0, n) in pushes of `step` ids into out; returns the bytes, or a negative code */
static int64_t stream_all(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint64_t step, uint8_t *out,
                          uint64_t cap)
{
    toks_stream st;
    toks_stream_init(ctx, &st, 0u);
    uint64_t pos = 0;
    for (uint64_t i = 0; i < n; i += step) {
        uint64_t m = n - i < step ? n - i : step;
        int64_t r = toks_stream_push(ctx, &st, ids + i, m, out + pos, cap - pos);
        if (r < 0) { return r; }
        pos += (uint64_t)r;
    }
    int64_t r = toks_stream_flush(ctx, &st, out + pos, cap - pos);
    return r < 0 ? r : (int64_t)(pos + (uint64_t)r);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: bench_stream <tokenizer.json> <text> [reps]\n"); return 2; }
    int reps = argc >= 4 ? atoi(argv[3]) : 15;
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, argv[1], NULL) != 0) { fprintf(stderr, "load %s failed\n", argv[1]); return 2; }
    FILE *f = fopen(argv[2], "rb");
    if (f == NULL) { fprintf(stderr, "open %s failed\n", argv[2]); return 2; }
    fseek(f, 0, SEEK_END);
    uint64_t len = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *text = (uint8_t *)malloc(len + 1u);
    if (text == NULL || fread(text, 1, (size_t)len, f) != len) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);
    uint64_t sb = toks_scratch_bytes(ctx, len, 0u);
    void *scr = malloc((size_t)sb);
    uint32_t *ids = (uint32_t *)malloc(4u * (size_t)(len + 64u));
    if (scr == NULL || ids == NULL || toks_scratch_init(ctx, scr, sb, 0u) != 0) { fprintf(stderr, "scratch\n"); return 2; }
    int64_t n = toks_encode(ctx, text, len, TOKS_NO_POSTPROCESS, ids, len + 64u, scr);
    if (n <= 0) { fprintf(stderr, "encode failed: %" PRId64 "\n", n); return 2; }
    uint64_t cap = 3u * len + 4096u;               /* >= 3 x the ids' bytes, which are the text's bytes */
    uint8_t *ref = (uint8_t *)malloc((size_t)cap), *out = (uint8_t *)malloc((size_t)cap);
    int64_t nref = toks_decode(ctx, ids, (uint64_t)n, 0u, ref, cap);
    if (nref < 0 || (uint64_t)nref > cap) { fprintf(stderr, "decode failed: %" PRId64 "\n", nref); return 2; }

    double best[3] = { 1e30, 1e30, 1e30 };
    int stream_ok = 1;
    for (int rep = 0; rep < reps; rep++) {
        for (int v = 0; v < 3; v++) {
            if (v > 0 && !stream_ok) { continue; }
            memset(out, 0, 4096);
            double t0 = now_s();
            int64_t k = v == 0 ? toks_decode(ctx, ids, (uint64_t)n, 0u, out, cap)
                               : stream_all(ctx, ids, (uint64_t)n, v == 1 ? 1u : 16u, out, cap);
            double t = now_s() - t0;
            if (v > 0 && k == TOKS_E_UNSUPPORTED) { stream_ok = 0; continue; }
            if (k != nref || memcmp(out, ref, (size_t)nref) != 0) {
                fprintf(stderr, "variant %d: output differs from the batch decode (%" PRId64 " bytes)\n", v, k);
                return 1;
            }
            if (t < best[v]) { best[v] = t; }
        }
    }
    printf("%s %s: %" PRIu64 " text bytes, %" PRId64 " ids, %" PRId64 " decoded bytes, best of %d\n", argv[1], argv[2],
           len, n, nref, reps);
    static const char *NAME[3] = { "toks_decode", "stream push 1 id", "stream push 16 ids" };
    for (int v = 0; v < 3; v++) {
        if (v > 0 && !stream_ok) { printf("  %-20s unsupported in this library\n", NAME[v]); continue; }
        printf("  %-20s %9.1f MB/s  %7.2f ns/id\n", NAME[v], (double)nref / best[v] / 1e6, best[v] * 1e9 / (double)n);
    }
    free(ref); free(out); free(ids); free(scr); free(text);
    toks_unload(ctx);
    return 0;
}
