/*
 * tests/spm/check.c: sentencepiece-style bpe against hf tokenizers 0.23.2 at scale, end to end (not part of
 * `make test`; tests/spm/run.sh builds it). Loads the tokenizer.json given as argv[1] with toks_load_mem_copy,
 * then reads tests/spm/gen.py's records from stdin:
 *   1 a gap          toks_encode (mode NONE, no pp; TOKS_CONTINUATION for at_start 0), and the same gap through
 *                    toks_spm_encode on whole pieces without the cache (§5.5: the same ids)
 *   2 a piece        toks_spm_model
 *   3 decode         toks_decode (skip special or not)
 *   4 a failing gap  toks_encode must fail
 *   5 the file       toks_encode mode ALL with post-processing, or NONSPECIAL without
 * Prints the counts; exit 0 iff every record matched.
 */
#include "core.h"
#include "spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_all(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n + 1u);
    if (b == NULL || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

static int rd(void *p, size_t n) { return n == 0u || fread(p, 1, n, stdin) == n; }

static uint32_t rd32(void)
{
    uint8_t b[4];
    if (!rd(b, 4)) { fprintf(stderr, "truncated record\n"); exit(2); }
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
}

#define MAXB (1u << 20)

static void hexdump(const uint8_t *p, uint64_t n)
{
    for (uint64_t i = 0; i < n && i < 200u; i++) { printf("%02x", p[i]); }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: check tokenizer.json < cases\n"); return 2; }
    uint64_t len;
    uint8_t *data = read_all(argv[1], &len);
    if (data == NULL) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    toks_diag dg;
    memset(&dg, 0, sizeof dg);
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &dg;
    toks_ctx *ctx = NULL;
    int64_t r = toks_load_mem_copy(&ctx, data, len, &o);
    free(data);
    if (r != 0) { printf("LOAD %s: %lld (%s)\n", argv[1], (long long)r, dg.what); return 3; }
    uint64_t sb = toks_scratch_bytes(ctx, MAXB, 0);
    uint8_t *scr = (uint8_t *)malloc((size_t)sb);
    if (scr == NULL || toks_scratch_init(ctx, scr, sb, 0) != 0) { fprintf(stderr, "scratch\n"); return 2; }
    uint8_t *text = (uint8_t *)malloc(MAXB);
    uint32_t *want = (uint32_t *)malloc(MAXB * 4u + 64u), *got = (uint32_t *)malloc(MAXB * 4u + 64u);
    uint8_t *dec = (uint8_t *)malloc(MAXB * 4u), *wdec = (uint8_t *)malloc(MAXB * 4u);
    uint8_t *work = (uint8_t *)malloc(TOKS_SPM_WORK_BYTES(MAXB + 1u) + 64u);
    uint64_t n_seg = 0, n_piece = 0, n_dec = 0, n_err = 0, n_full = 0, bad = 0;
    int kind;
    while ((kind = getchar()) != EOF) {
        if (kind == 1 || kind == 4) {
            int at_start = getchar();
            uint32_t n = rd32();
            if (n > MAXB || !rd(text, n)) { fprintf(stderr, "bad record\n"); return 2; }
            uint32_t fl = TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS | (at_start ? 0u : TOKS_CONTINUATION);
            if (kind == 4) {
                n_err++;
                int64_t k = toks_encode(ctx, text, n, fl, got, MAXB, scr);
                if (k >= 0 && bad++ < 10) { printf("MISMATCH fail-gap n=%u: toks gave %lld ids\n", n, (long long)k); }
                continue;
            }
            uint32_t k = rd32();
            if (k > MAXB || !rd(want, (size_t)k * 4u)) { fprintf(stderr, "bad record\n"); return 2; }
            n_seg++;
            int64_t g = toks_encode(ctx, text, n, fl, got, MAXB, scr);
            if (g != (int64_t)k || memcmp(got, want, (size_t)k * 4u) != 0) {
                if (bad++ < 10) {
                    printf("MISMATCH gap at_start=%d n=%u: toks %lld ids, hf %u; text ", at_start, n, (long long)g, k);
                    hexdump(text, n);
                }
            }
            uint64_t g2 = toks_spm_encode(&ctx->t, ctx->spm, text, n, at_start, got, MAXB, 0, NULL, 0u, 0u, work,
                                          TOKS_SPM_NOCUTS | TOKS_SPM_NOCACHE);
            if (g2 != (uint64_t)k || memcmp(got, want, (size_t)k * 4u) != 0) {
                if (bad++ < 10) {
                    printf("MISMATCH whole-pieces at_start=%d n=%u: toks %llu ids, hf %u; text ", at_start, n,
                           (unsigned long long)g2, k);
                    hexdump(text, n);
                }
            }
        } else if (kind == 5) {
            int how = getchar();
            uint32_t n = rd32();
            if (n > MAXB || !rd(text, n)) { fprintf(stderr, "bad record\n"); return 2; }
            uint32_t k = rd32();
            if (k > MAXB || !rd(want, (size_t)k * 4u)) { fprintf(stderr, "bad record\n"); return 2; }
            n_full++;
            uint32_t fl = how == 0 ? TOKS_ADDED_ALL : (TOKS_ADDED_NONSPECIAL | TOKS_NO_POSTPROCESS);
            int64_t g = toks_encode(ctx, text, n, fl, got, MAXB, scr);
            if (g != (int64_t)k || memcmp(got, want, (size_t)k * 4u) != 0) {
                if (bad++ < 10) {
                    printf("MISMATCH full how=%d n=%u: toks %lld ids, hf %u; text ", how, n, (long long)g, k);
                    hexdump(text, n);
                }
            }
        } else if (kind == 2) {
            uint32_t n = rd32();
            if (n > MAXB || !rd(text, n)) { fprintf(stderr, "bad record\n"); return 2; }
            uint32_t k = rd32();
            if (k > MAXB || !rd(want, (size_t)k * 4u)) { fprintf(stderr, "bad record\n"); return 2; }
            n_piece++;
            uint64_t g = toks_spm_model(&ctx->t, ctx->spm, text, n, got, work);
            if (g != (uint64_t)k || memcmp(got, want, (size_t)k * 4u) != 0) {
                if (bad++ < 10) { printf("MISMATCH piece n=%u: toks %llu ids, hf %u; piece ", n, (unsigned long long)g, k); hexdump(text, n); }
            }
        } else if (kind == 3) {
            int skip = getchar();
            uint32_t k = rd32();
            if (k > MAXB || !rd(want, (size_t)k * 4u)) { fprintf(stderr, "bad record\n"); return 2; }
            uint32_t n = rd32();
            if (n > MAXB * 4u || !rd(wdec, n)) { fprintf(stderr, "bad record\n"); return 2; }
            n_dec++;
            int64_t g = toks_decode(ctx, want, k, skip ? TOKS_SKIP_SPECIAL : 0u, dec, MAXB * 4u);
            if (g != (int64_t)n || memcmp(dec, wdec, n) != 0) {
                if (bad++ < 10) { printf("MISMATCH decode skip=%d k=%u: toks %lld bytes, hf %u\n", skip, k, (long long)g, n); }
            }
        } else {
            fprintf(stderr, "unknown record %d\n", kind);
            return 2;
        }
    }
    printf("RESULT %s: gaps %llu (x2: certified words + cache, whole pieces) full-file %llu pieces %llu decodes %llu "
           "failing-gaps %llu mismatches %llu\n",
           argv[1], (unsigned long long)n_seg, (unsigned long long)n_full, (unsigned long long)n_piece,
           (unsigned long long)n_dec, (unsigned long long)n_err, (unsigned long long)bad);
    toks_unload(ctx);
    return bad != 0;
}
