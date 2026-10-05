/*
 * tests/k3/check.c: differential checker of the K3 c twin against hf tokenizers 0.23.2 at scale.
 * NOT part of `make test` (it reads a case stream produced by tests/k3/gen.py); tests/k3/run.sh builds and runs it:
 *
 *   tests/k3/run.sh 300000 30000 1     (gen.py's stream to build/k3cases.bin, then build/k3check on it)
 *
 * Reads records: u8 variant, u32 len, text bytes, u32 n_ends, u32 pairs (a, b). For each, runs the
 * kernel (cap 4096) and compares ends exactly; also re-runs with cap 1 and resume to check the
 * tiling equals the one-shot result. Exit 0 iff every case matched.
 */
/* kernels.h -> core.h redeclares memcpy/memset/memcmp (freestanding core): it must come before
 * <string.h>, whose secure _chk macros would collide with the declarations. */
#include "../../src/core/classes.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"

/* the tier under test: the c twin by default, an asm tier with -DK3_SCAN=toks_k3_scan_cl100k_<tier> */
#ifndef K3_SCAN
#define K3_SCAN toks_k3_scan_cl100k_c
#endif
uint64_t K3_SCAN(const toks_tables *t, toks_k3_args *a);
/* the tier's whole kernel: an asm part runs with the c twin's stretches (kernels.h toks_k3_tier) */
static uint64_t k3_scan(const toks_tables *t, toks_k3_args *a)
{
    return toks_k3_tier(K3_SCAN, toks_k3_scan_cl100k_c, t, a);
}

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the variants in tests/k3/gen.py order; flags identical to tests/c/test_k3.c */
typedef struct { uint32_t params; uint32_t cls_flags; } hf_variant;

static const hf_variant V[] = {
    { TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN, 0u },
    { TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, 0u },
    { TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, 0u },
    { TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL,
      TOKS_CLASSES_MARKS_ARE_LETTERS },
};

#define MAXEND 4096

static uint8_t cls_buf[2][128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CTAB[2];
static toks_tables TT[4];

static void build(void)
{
    static const uint32_t flags[2] = { 0u, TOKS_CLASSES_MARKS_ARE_LETTERS };
    for (int i = 0; i < 2; i++) {
        int used = toks_classes_build(flags[i], cls_buf[i], sizeof cls_buf[i], &CTAB[i]);
        if (used <= 0) { fprintf(stderr, "classes build failed\n"); exit(2); }
    }
    for (int v = 0; v < 4; v++) {
        toks_tables *t = &TT[v];
        memset(t, 0, sizeof *t);
        t->magic = TOKS_TABLES_MAGIC;
        t->version = TOKS_TABLES_VERSION;
        t->tmpl = TOKS_TMPL_CL100K;
        t->tmpl_params = V[v].params;
        int i = V[v].cls_flags != 0u;
        t->cls_ascii = CTAB[i].ascii;
        t->cls_stage1 = CTAB[i].stage1;
        t->cls_stage2 = CTAB[i].stage2;
        t->cls_nblocks = CTAB[i].n_blocks;
        int64_t cf = toks_compile_cls_flags(t);           /* TOKS_TF_CJK_L, as the compiler sets it */
        if (cf < 0) { fprintf(stderr, "class tables break kernels.md §2\n"); exit(2); }
        t->flags = (uint32_t)cf;
    }
}

static int rd_u8(FILE *f, uint8_t *v) { return fread(v, 1, 1, f) == 1 ? 0 : -1; }
static int rd_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1 ? 0 : -1; }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s cases.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    build();
    static uint8_t text[1 << 24];
    static uint32_t want[MAXEND], got[MAXEND], acc[MAXEND];
    uint64_t cases = 0, bad = 0;
    uint8_t vi;
    while (rd_u8(f, &vi) == 0) {
        uint32_t len, n_pairs;
        if (vi > 3 || rd_u32(f, &len) != 0 || len > sizeof text || fread(text, 1, len, f) != len) {
            fprintf(stderr, "corrupt case stream at case %" PRIu64 "\n", cases);
            return 2;
        }
        if (rd_u32(f, &n_pairs) != 0 || n_pairs > MAXEND ||
            fread(want, 8, (size_t)n_pairs, f) != n_pairs) {   /* (start, end) u32 pairs */
            fprintf(stderr, "corrupt ends at case %" PRIu64 "\n", cases);
            return 2;
        }
        cases++;
        /* hf's pieces must tile [0, len): piece i is [want[2i], want[2i+1]) */
        uint32_t n_ends = 0;
        int tiles = 1;
        for (uint32_t i = 0; i < n_pairs; i++) {               /* bound: n_pairs <= MAXEND */
            if (want[2 * i] != (i == 0 ? 0u : want[2 * i - 1]) ||
                want[2 * i] >= want[2 * i + 1] || want[2 * i + 1] > len) { tiles = 0; }
            want[i] = want[2 * i + 1];
            n_ends = i + 1;
        }
        if (n_pairs != 0 && want[n_pairs - 1] != len) { tiles = 0; }
        if (!tiles) {
            fprintf(stderr, "BAD ORACLE case %" PRIu64 ": hf pieces do not tile [0, len)\n", cases);
            bad++;
            continue;
        }
        /* one call, big cap: piece ends must be exactly hf's */
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        a.text = text; a.len = len; a.pos = 0; a.ends = got; a.cap = MAXEND;
        uint64_t n = k3_scan(&TT[vi], &a);
        if (n != n_ends || (n_ends != 0 && memcmp(got, want, (size_t)n_ends * 4) != 0)) {
            if (bad < 10) {
                fprintf(stderr, "MISMATCH case %" PRIu64 " variant %u len %u: n %" PRIu64
                        " want %u\n", cases, vi, len, n, n_ends);
            }
            bad++;
            continue;
        }
        /* cap-1 resume tiling must equal the one-shot ends (already verified vs hf above) */
        if (len != 0) {
            uint64_t pos = 0, an = 0;
            while (pos < len) {                         /* bound: pieces (each advances >= 1) */
                memset(&a, 0, sizeof a);
                a.text = text; a.len = len; a.pos = pos; a.ends = acc + an; a.cap = 1;
                uint64_t c = k3_scan(&TT[vi], &a);
                if (c != 1 || a.pos <= pos) {
                    fprintf(stderr, "RESUME FAIL case %" PRIu64 " variant %u pos %" PRIu64 "\n",
                            cases, vi, pos);
                    bad++;
                    break;
                }
                an++;
                pos = a.pos;
            }
            if (an != n_ends || (n_ends != 0 && memcmp(acc, want, (size_t)n_ends * 4) != 0)) {
                if (bad < 10) {
                    fprintf(stderr, "RESUME MISMATCH case %" PRIu64 " variant %u: %" PRIu64
                            " pieces want %u\n", cases, vi, an, n_ends);
                }
                bad++;
            }
        }
    }
    printf("test_k3_hf: %" PRIu64 " cases, %" PRIu64 " mismatches\n", cases, bad);
    return bad ? 1 : 0;
}
