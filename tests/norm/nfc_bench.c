/*
 * nfc_bench.c: single-thread MB/s of the NFC paths over a record stream from tests/norm/gen.py (u32 n_in, in,
 * u32 n_out, hf NFC(in)); every record is one segment, as one text gap of the driver. Exactness is checked
 * outside the timer (SPEC §12.4). Not part of `make test`; run on a lab host, pinned (taskset -c N), with
 * `uptime` recorded.
 *
 *   nfc_bench FILE [REPS] [TOKENIZER_JSON]   -> min / median / max MB/s over REPS timed passes (default 31) of
 *
 *   scan     toks_nfc_scan over each segment: what the driver pays on already-NFC text (it reads, writes nothing)
 *   nfc      toks_norm (NFC) of each whole segment into a buffer (the normalizer's own speed; every byte written)
 *   with TOKENIZER_JSON (a tokenizer whose normalizer is NFC alone, e.g. Qwen 3.8):
 *   plan     the driver's stretch plan (norm.h) over each segment: bytes it normalizes into scratch per input byte
 *   encode   toks_encode mode NONE of each segment, normalizer NFC, and the same file with "normalizer": null
 *            (ids compared where the segment is already NFC: they must be equal)
 */
#define _POSIX_C_SOURCE 199309L
#include "toks.h"
#include "../../src/core/norm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct rec { uint8_t *in, *want; uint32_t n_in, n_out; } rec;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static size_t n, total, maxin;
static rec *r;
static int reps;
static uint8_t *out;
static uint32_t *ids;
static volatile uint64_t sink;

static void report(const char *what, double *mbs)
{
    qsort(mbs, (size_t)reps, sizeof *mbs, cmp_d);
    printf("  %-22s MB/s over %d passes: min %8.1f  median %8.1f  max %8.1f\n", what, reps, mbs[0], mbs[reps / 2],
           mbs[reps - 1]);
}

static void bench_scan(double *mbs)
{
    for (int k = 0; k < reps; k++) {
        double t0 = now();
        for (size_t i = 0; i < n; i++) {
            uint64_t e;
            sink += toks_nfc_scan(TOKS_NS_NFC, r[i].in, r[i].n_in, 0, &e);
        }
        mbs[k] = (double)total / (now() - t0) / 1e6;
    }
    report("scan (read only)", mbs);
}

static void bench_nfc(double *mbs)
{
    for (int k = 0; k < reps; k++) {
        double t0 = now();
        for (size_t i = 0; i < n; i++) {
            sink += (uint64_t)toks_norm(TOKS_NS_NFC, r[i].in, r[i].n_in, out, 3 * (uint64_t)r[i].n_in);
        }
        mbs[k] = (double)total / (now() - t0) / 1e6;
    }
    report("nfc (whole, copied)", mbs);
}

static void bench_encode(const char *what, toks_ctx *ctx, void *scr, double *mbs)
{
    for (int k = 0; k < reps; k++) {
        double t0 = now();
        for (size_t i = 0; i < n; i++) {
            sink += (uint64_t)toks_encode(ctx, r[i].in, r[i].n_in, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS, ids,
                                          3 * (uint64_t)maxin + 8, scr);
        }
        mbs[k] = (double)total / (now() - t0) / 1e6;
    }
    report(what, mbs);
}

static uint8_t *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long m = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)m + 1);
    if (fread(b, 1, (size_t)m, f) != (size_t)m) { fclose(f); return NULL; }
    fclose(f);
    b[m] = 0;
    *len = (size_t)m;
    return b;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: nfc_bench FILE [REPS] [TOKENIZER_JSON]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    reps = argc > 2 ? atoi(argv[2]) : 31;
    size_t cap = 1024;
    r = malloc(cap * sizeof *r);
    uint8_t hdr[4];
    while (fread(hdr, 1, 4, f) == 4) {
        if (n == cap) { cap *= 2; r = realloc(r, cap * sizeof *r); }
        r[n].n_in = (uint32_t)hdr[0] | (uint32_t)hdr[1] << 8 | (uint32_t)hdr[2] << 16 | (uint32_t)hdr[3] << 24;
        r[n].in = malloc(r[n].n_in + 1);
        if (fread(r[n].in, 1, r[n].n_in, f) != r[n].n_in || fread(hdr, 1, 4, f) != 4) { fprintf(stderr, "truncated\n"); return 2; }
        r[n].n_out = (uint32_t)hdr[0] | (uint32_t)hdr[1] << 8 | (uint32_t)hdr[2] << 16 | (uint32_t)hdr[3] << 24;
        r[n].want = malloc(r[n].n_out + 1);
        if (fread(r[n].want, 1, r[n].n_out, f) != r[n].n_out) { fprintf(stderr, "truncated\n"); return 2; }
        total += r[n].n_in;
        if (r[n].n_in > maxin) { maxin = r[n].n_in; }
        n++;
    }
    fclose(f);
    out = malloc(3 * maxin + 64);
    ids = malloc(4 * (3 * maxin + 8));
    uint64_t changed = 0, scan_changed = 0;
    for (size_t i = 0; i < n; i++) {                       /* exactness, outside the timer */
        int64_t k = toks_norm(TOKS_NS_NFC, r[i].in, r[i].n_in, out, 3 * (uint64_t)r[i].n_in);
        if (k != (int64_t)r[i].n_out || memcmp(out, r[i].want, r[i].n_out) != 0) {
            fprintf(stderr, "record %zu: output differs from hf\n", i);
            return 1;
        }
        uint64_t e;
        int same = r[i].n_out == r[i].n_in && memcmp(r[i].in, r[i].want, r[i].n_in) == 0;
        int unchanged = toks_nfc_scan(TOKS_NS_NFC, r[i].in, r[i].n_in, 0, &e) == r[i].n_in;
        if (unchanged && !same) { fprintf(stderr, "record %zu: the scan misses a change\n", i); return 1; }
        scan_changed += !unchanged;
        changed += !same;
    }
    printf("%s: %zu segments, %zu bytes; exact vs hf; %llu segments changed, %llu with a changed run found by the scan\n",
           argv[1], n, total, (unsigned long long)changed, (unsigned long long)scan_changed);
    double *mbs = malloc((size_t)reps * sizeof *mbs);
    bench_scan(mbs);
    bench_nfc(mbs);
    if (argc > 3) {
        size_t jl;
        uint8_t *j = slurp(argv[3], &jl);
        if (!j) { return 2; }
        toks_ctx *on = NULL, *off = NULL;
        if (toks_load_mem_copy(&on, j, jl, NULL) != 0) { fprintf(stderr, "load %s\n", argv[3]); return 2; }
        char *a = strstr((char *)j, "\"normalizer\": {"), *b = a ? strchr(a, '}') : NULL;
        if (!b) { fprintf(stderr, "no normalizer object\n"); return 2; }
        size_t pre = (size_t)(a - (char *)j), post = (size_t)(b + 1 - (char *)j);
        uint8_t *j2 = malloc(jl + 32);
        memcpy(j2, j, pre);
        memcpy(j2 + pre, "\"normalizer\": null", 18);
        memcpy(j2 + pre + 18, j + post, jl - post);
        if (toks_load_mem_copy(&off, j2, pre + 18 + jl - post, NULL) != 0) { fprintf(stderr, "load without NFC\n"); return 2; }
        uint64_t sb_on = toks_scratch_bytes(on, maxin, 0), sb_off = toks_scratch_bytes(off, maxin, 0);
        void *s_on = malloc(sb_on), *s_off = malloc(sb_off);
        toks_scratch_init(on, s_on, sb_on, 0);
        toks_scratch_init(off, s_off, sb_off, 0);
        const toks_tables *tb = &((const struct toks_ctx *)on)->t;
        uint64_t mat = 0, wrote = 0, stretches = 0, zc = 0;
        uint32_t *ids2 = malloc(4 * (3 * maxin + 8));
        for (size_t i = 0; i < n; i++) {                   /* the plan the driver runs, outside the timer */
            toks_nfc_plan pl;
            toks_nfc_plan_begin(&pl, TOKS_NS_NFC, r[i].in, r[i].n_in);
            zc += pl.d0 == r[i].n_in;
            uint64_t s, e;
            while (toks_nfc_plan_next(&pl, tb, r[i].in, r[i].n_in, &s, &e)) {
                mat += e - s;
                wrote += (uint64_t)toks_norm(TOKS_NS_NFC, r[i].in + s, e - s, out, 3 * (e - s));
                stretches++;
            }
            if (pl.d0 == r[i].n_in && pl.r == 0) {         /* already NFC: NFC on and off give the same ids */
                int64_t x = toks_encode(on, r[i].in, r[i].n_in, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS, ids, 3 * maxin + 8, s_on);
                int64_t y = toks_encode(off, r[i].in, r[i].n_in, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS, ids2, 3 * maxin + 8, s_off);
                if (x < 0 || x != y || memcmp(ids, ids2, (size_t)x * 4) != 0) { fprintf(stderr, "record %zu: ids differ\n", i); return 1; }
            }
        }
        printf("  plan: %llu of %zu segments in place (zero copies); %llu stretches; bytes normalized into scratch"
               " per input byte: %.6f read, %.6f written\n", (unsigned long long)zc, n, (unsigned long long)stretches,
               (double)mat / (double)total, (double)wrote / (double)total);
        bench_encode("encode, NFC (Qwen 3.8)", on, s_on, mbs);
        bench_encode("encode, normalizer null", off, s_off, mbs);
    }
    return (int)(sink & 0);
}
