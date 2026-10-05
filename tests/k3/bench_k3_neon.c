/*
 * bench_k3_neon.c: K3 throughput, the c twin vs the neon tier, on the same input, one core.
 * NOT part of `make test`; a lab-host tool (maintainer doctrine: no benchmarks on the control-plane mac):
 *
 *   clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core \
 *     -Isrc/platform -Isrc/asm/arm64 -o build/bench_k3_neon tests/k3/bench_k3_neon.c \
 *     src/core/classes.c src/core/k3_c.c src/gen/ucd_flags.c src/asm/arm64/k3_cl100k_neon.S
 *   taskset -c 3 ./build/bench_k3_neon [-m MiB] [-r abba-blocks] [-c cap] [-v variant] name=spec ...
 *
 * Each `name=spec` is one input; spec is a '+'-joined list of sources, each a text file or
 * `cases:<path>[@MiB]` (the texts of a tests/k3/gen.py case stream, concatenated, at most MiB of
 * them). The sources are
 * concatenated and the result repeated to exactly -m MiB (default 64): one segment, as the driver
 * would hand K3 a long gap. Every kernel is driven the driver's way: cap -c pieces per call
 * (default TOKS_CHUNK_PIECES = 256), resume at a.pos until the segment is done.
 *
 * fuzz mode (seconds to minutes; heavy runs belong on a lab host):
 *   ./build/bench_k3_neon fuzz [-x] [-p] ex LMAX [NSYM]     every string of <= LMAX atoms over the
 *                                                          first NSYM of the alphabet below
 *   ./build/bench_k3_neon fuzz [-x] [-p] rnd N MAXLEN SEED  N random strings (runs up to 80 atoms)
 *   ./build/bench_k3_neon fuzz [-x] [-p] byt N MAXLEN SEED  N random byte strings with utf-8 shape
 * compares the neon tier with the c twin on every case: all four variants (-x adds four other TOKS_TP_*
 * combinations, -p four perturbed class tables that break the block algebra's preconditions), one call
 * with a big cap, then cap 1 2 3 7 with resume. Exit 0 iff no mismatch.
 *
 * Before timing, both kernels scan the whole input once with an unbounded cap and their piece ends
 * are compared (a bench that times wrong output is not a bench). Timing follows SPEC §12.5's shape:
 * -r paired abba blocks (default 15: c, neon, neon, c -> 30 paired samples), each sample one full pass
 * over the input. Reported: the median MB/s (10^6 bytes) of each kernel, the speedup = mean of the
 * per-pair ratios with its 95% bootstrap interval (2000 resamples), the best pass of each kernel, and
 * the load average (/proc/loadavg; -1 where there is none).
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c17 on linux */
#include "../../src/core/classes.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t toks_k3_scan_cl100k_neon(const toks_tables *t, toks_k3_args *a);

typedef uint64_t (*k3_fn)(const toks_tables *t, toks_k3_args *a);

static const struct { const char *name; uint32_t params; uint32_t cls_flags; } V[] = {
    { "gpt2",   TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN, 0u },
    { "cl100k", TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, 0u },
    { "qwen2",  TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL
                | TOKS_TP_WS_NL, 0u },
    { "qwen35", TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL
                | TOKS_TP_WS_NL, TOKS_CLASSES_MARKS_ARE_LETTERS },
};
#define NV 4

static uint8_t cls_buf[2][128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CTAB[2];
static toks_tables TT[NV];

static void build(void)
{
    static const uint32_t flags[2] = { 0u, TOKS_CLASSES_MARKS_ARE_LETTERS };
    for (int i = 0; i < 2; i++) {
        if (toks_classes_build(flags[i], cls_buf[i], sizeof cls_buf[i], &CTAB[i]) <= 0) {
            fprintf(stderr, "classes build failed\n");
            exit(2);
        }
    }
    for (int v = 0; v < NV; v++) {
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
    }
}

/* ---------------------------------------------------------------- input */

typedef struct { uint8_t *p; size_t n, cap; } buf;

static void put(buf *b, const uint8_t *p, size_t n)
{
    if (b->n + n > b->cap) {
        size_t c = b->cap ? b->cap : (1u << 20);
        while (c < b->n + n) { c *= 2; }
        b->p = realloc(b->p, c);
        if (!b->p) { fprintf(stderr, "out of memory\n"); exit(2); }
        b->cap = c;
    }
    memcpy(b->p + b->n, p, n);
    b->n += n;
}

static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)sz + 1u);
    if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read %s failed\n", path); exit(2); }
    fclose(f);
    *n = (size_t)sz;
    return p;
}

/* the texts of a gen.py case stream: u8 variant, u32 len, text, u32 n, n x (u32, u32) */
static void add_cases(buf *b, const char *path, size_t limit)
{
    size_t n;
    uint8_t *p = slurp(path, &n);
    size_t i = 0;
    while (i + 5u <= n && b->n < limit) {
        uint32_t len, cnt;
        memcpy(&len, p + i + 1u, 4);
        i += 5u;
        if (i + len + 4u > n) { break; }
        put(b, p + i, len);
        i += len;
        memcpy(&cnt, p + i, 4);
        i += 4u + 8u * (size_t)cnt;
    }
    free(p);
}

static void add_spec(buf *b, char *spec, size_t limit)
{
    for (char *tok = strtok(spec, "+"); tok; tok = strtok(NULL, "+")) {
        if (strncmp(tok, "cases:", 6) == 0) {             /* cases:<path>[@<MiB>] */
            char *at = strchr(tok + 6, '@');
            size_t lim = limit;
            if (at) { *at = '\0'; lim = b->n + ((size_t)atol(at + 1) << 20); }
            add_cases(b, tok + 6, lim < limit ? lim : limit);
        } else {
            size_t n;
            uint8_t *p = slurp(tok, &n);
            put(b, p, n);
            free(p);
        }
    }
}

/* ---------------------------------------------------------------- timing */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double loadavg(void)
{
    double l = -1.0;
    FILE *f = fopen("/proc/loadavg", "r");
    if (f) {
        if (fscanf(f, "%lf", &l) != 1) { l = -1.0; }
        fclose(f);
    }
    return l;
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(const double *v, int n)
{
    double *t = malloc((size_t)n * sizeof *t);
    if (!t) { exit(2); }
    memcpy(t, v, (size_t)n * sizeof *t);
    qsort(t, (size_t)n, sizeof *t, cmp_dbl);
    double m = (n & 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
    free(t);
    return m;
}

/* mean of r[0, n) and its 95% percentile bootstrap interval (2000 resamples, fixed seed) */
static void boot(const double *r, int n, double *mean, double *lo, double *hi)
{
    enum { NB = 2000 };
    static double m[NB];
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    double s0 = 0.0;
    for (int i = 0; i < n; i++) { s0 += r[i]; }
    *mean = s0 / n;
    for (int b = 0; b < NB; b++) {                     /* bound: NB resamples x n */
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            sum += r[seed % (uint64_t)n];
        }
        m[b] = sum / n;
    }
    qsort(m, NB, sizeof m[0], cmp_dbl);
    *lo = m[NB * 25 / 1000];
    *hi = m[NB * 975 / 1000 - 1];
}

static uint32_t chunk_ends[1u << 16];

/* the driver's loop: cap pieces per call, resume at a.pos; returns pieces */
static uint64_t drive(k3_fn k, const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t cap)
{
    uint64_t pos = 0, total = 0;
    while (pos < len) {
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        a.text = text; a.len = len; a.pos = pos; a.ends = chunk_ends; a.cap = cap;
        uint64_t n = k(t, &a);
        if (n == 0u || a.pos <= pos) { fprintf(stderr, "kernel made no progress at %" PRIu64 "\n", pos); exit(3); }
        total += n;
        pos = a.pos;
    }
    return total;
}


/* ---------------------------------------------------------------- fuzz mode */

static toks_tables FT[12];
static int NFT;

static void fuzz_tables(int extra, int perturb)
{
    static const uint32_t X[4] = {
        TOKS_TP_CONTR_NONE | TOKS_TP_DIGITS_RUN,
        TOKS_TP_CONTR_CS | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1_3 | TOKS_TP_WS_NL,
        TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_SP_RUN | TOKS_TP_PUNCT_NL,
        TOKS_TP_CONTR_NONE | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_SP_RUN | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL,
    };
    static uint8_t asc_a[128], asc_b[128];
    static uint8_t big[0x1100 * 256];
    NFT = 0;
    for (int v = 0; v < NV; v++) { FT[NFT++] = TT[v]; }
    if (extra) {                                  /* other parameter combinations of the template */
        for (int v = 0; v < 4; v++) { FT[NFT] = TT[(v & 1) ? 3 : 0]; FT[NFT++].tmpl_params = X[v]; }
    }
    if (perturb) {                                /* tables outside the block algebra's preconditions */
        memcpy(asc_a, CTAB[0].ascii, 128);
        asc_a[0x20] = TOKS_C_P;                   /* U+0020 not WS: scalar-only mode */
        memcpy(asc_b, CTAB[0].ascii, 128);
        asc_b['!'] = 5;                           /* an ascii base class > NL: scalar-only mode */
        memcpy(big, CTAB[0].stage2, (size_t)CTAB[0].n_blocks * 256u);
        static const uint32_t cps[4] = { 0xE9, 0x4E2D, 0x1F600, 0xA0 };
        for (int i = 0; i < 4; i++) {             /* non-ascii base classes > NL: the block cuts */
            uint32_t b1 = CTAB[0].stage1[cps[i] >> 8];
            big[b1 * 256u + (cps[i] & 0xFFu)] = (uint8_t)(5 + i % 3);
        }
        for (int v = 0; v < 4; v++) {
            FT[NFT] = TT[v];
            if (v == 0) { FT[NFT].cls_ascii = asc_a; }
            else if (v == 1) { FT[NFT].cls_ascii = asc_b; }
            else { FT[NFT].cls_stage2 = big; }
            NFT++;
        }
    }
}

/* alphabet: utf-8 snippets */
static const char *const SYM[] = {
    "a", " ", "'", "1", "\n", "!", "s", "\t", "\r", "e", "l", "r", "\xC3\xA9", "\xE2\x80\x9C",
    "\xC2\xA0", "\xD9\xA3", "\xC5\xBF", "\x80", "\xC3", "\xE3\x80\x80", "\xCC\x81", "\xF0\x9F\x98\x80",
    "S", "v", "E", "L", "x", "2", "\xEF\xBC\x91", "\xE2\x80\xA8", "\xF0\x9F", "\xED\xA0\x80", ".",
    "\xE4\xB8\xAD", "\xE3\x81\x82", "\xE3\x80\x82", "\xEF\xBC\x8C", "\xD0\x96", "\xE0\xA4\x95",
    "\xE0\x9F\xBF", "\xED\x9F\xBF", "\xEE\x80\x80", "\xF0\x90\x80\x80", "\xF4\x8F\xBF\xBF", "\xF4\x90\x80\x80",
};
#define NSYM (sizeof SYM / sizeof SYM[0])

static uint64_t fz_cases, fz_bad;
static uint32_t fz_want[1u << 14], fz_got[1u << 14], fz_acc[1u << 14];

static void fz_check(const uint8_t *s, uint64_t len)
{
    for (int v = 0; v < NFT; v++) {
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        a.text = s; a.len = len; a.ends = fz_want; a.cap = 1u << 14;
        uint64_t rn = toks_k3_scan_cl100k_c(&FT[v], &a);
        memset(&a, 0, sizeof a);
        a.text = s; a.len = len; a.ends = fz_got; a.cap = 1u << 14;
        uint64_t kn = toks_k3_scan_cl100k_neon(&FT[v], &a);
        fz_cases++;
        if (kn != rn || memcmp(fz_want, fz_got, (size_t)rn * 4u) != 0 || (rn != 0u && a.pos != fz_want[rn - 1u])) {
            if (fz_bad++ < 10) {
                fprintf(stderr, "MISMATCH table %d len %" PRIu64 " text:", v, len);
                for (uint64_t i = 0; i < len; i++) { fprintf(stderr, " %02X", s[i]); }
                fprintf(stderr, "\n  c   :");
                for (uint64_t i = 0; i < rn; i++) { fprintf(stderr, " %u", fz_want[i]); }
                fprintf(stderr, "\n  neon:");
                for (uint64_t i = 0; i < kn; i++) { fprintf(stderr, " %u", fz_got[i]); }
                fprintf(stderr, "\n");
            }
            continue;
        }
        static const uint64_t caps[4] = { 1, 2, 3, 7 };
        for (int ci = 0; ci < 4; ci++) {
            uint64_t pos = 0, an = 0;
            while (pos < len) {                       /* bound: pieces (each call advances) */
                memset(&a, 0, sizeof a);
                a.text = s; a.len = len; a.pos = pos; a.ends = fz_acc + an; a.cap = caps[ci];
                uint64_t c = toks_k3_scan_cl100k_neon(&FT[v], &a);
                if (c == 0u || c > caps[ci] || a.pos <= pos || a.n != c) {
                    if (fz_bad++ < 10) { fprintf(stderr, "RESUME FAIL table %d cap %" PRIu64 "\n", v, caps[ci]); }
                    return;
                }
                an += c;
                pos = a.pos;
            }
            if (an != rn || memcmp(fz_acc, fz_want, (size_t)rn * 4u) != 0) {
                if (fz_bad++ < 10) {
                    fprintf(stderr, "RESUME MISMATCH table %d cap %" PRIu64 " len %" PRIu64 "\n", v, caps[ci], len);
                }
                return;
            }
        }
    }
}

static uint64_t fz_rng(uint64_t *x) { *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; return *x; }

static int fuzz_main(int argc, char **argv)
{
    int extra = 0, perturb = 0, ai = 1;
    for (; ai < argc && argv[ai][0] == '-'; ai++) {
        if (strcmp(argv[ai], "-x") == 0) { extra = 1; }
        else if (strcmp(argv[ai], "-p") == 0) { perturb = 1; }
        else { fprintf(stderr, "unknown fuzz option %s\n", argv[ai]); return 2; }
    }
    if (ai + 1 >= argc) { fprintf(stderr, "usage: fuzz [-x] [-p] ex LMAX [NSYM] | rnd|byt N MAXLEN SEED\n"); return 2; }
    build();
    fuzz_tables(extra, perturb);
    static uint8_t buf[1u << 16];
    const char *mode = argv[ai];
    if (strcmp(mode, "ex") == 0) {
        int lmax = atoi(argv[ai + 1]);
        int ns = ai + 2 < argc ? atoi(argv[ai + 2]) : (int)NSYM;
        if (lmax < 0 || lmax > 16 || ns < 1 || ns > (int)NSYM) { fprintf(stderr, "bad ex arguments\n"); return 2; }
        for (int l = 0; l <= lmax; l++) {             /* bound: ns^l strings per length */
            int idx[16] = { 0 };
            for (;;) {
                uint64_t len = 0;
                for (int i = 0; i < l; i++) {
                    size_t k = strlen(SYM[idx[i]]);
                    memcpy(buf + len, SYM[idx[i]], k);
                    len += k;
                }
                fz_check(buf, len);
                int i = 0;
                while (i < l && ++idx[i] == ns) { idx[i] = 0; i++; }
                if (i == l) { break; }
            }
        }
    } else if ((strcmp(mode, "rnd") == 0 || strcmp(mode, "byt") == 0) && ai + 3 < argc) {
        uint64_t n = strtoull(argv[ai + 1], NULL, 10), maxl = strtoull(argv[ai + 2], NULL, 10);
        uint64_t seed = strtoull(argv[ai + 3], NULL, 10) | 1u;
        int byt = mode[0] == 'b';
        static const uint8_t asc[] = "a s'1\n!\t\re.SLx2";
        for (uint64_t c = 0; c < n; c++) {             /* bound: n cases */
            uint64_t l = fz_rng(&seed) % (maxl + 1u), len = 0;
            uint64_t shape = fz_rng(&seed) % 4u;
            while (len < l && len + 96u < sizeof buf) {
                if (byt) {                             /* ascii, stray bytes, leads + 0-4 tail bytes */
                    uint64_t r = fz_rng(&seed) % 16u;
                    if (r < 6u) { buf[len++] = asc[fz_rng(&seed) % (sizeof asc - 1u)]; }
                    else if (r < 8u) { buf[len++] = (uint8_t)(0x80u + fz_rng(&seed) % 0x80u); }
                    else {
                        buf[len++] = (uint8_t)(0xC0u + fz_rng(&seed) % 0x40u);
                        uint64_t m = fz_rng(&seed) % 5u;
                        for (uint64_t j = 0; j < m; j++) {
                            buf[len++] = fz_rng(&seed) % 8u < 6u ? (uint8_t)(0x80u + fz_rng(&seed) % 0x40u)
                                                                   : (uint8_t)(fz_rng(&seed) & 0xFFu);
                        }
                    }
                } else {                               /* atoms, some repeated into long runs */
                    const char *sy = SYM[fz_rng(&seed) % (shape == 0u ? 12u : NSYM)];
                    uint64_t rep = shape == 3u ? 1u + fz_rng(&seed) % 80u
                                               : (fz_rng(&seed) % 8u == 0u ? 1u + fz_rng(&seed) % 20u : 1u);
                    size_t k = strlen(sy);
                    for (uint64_t r = 0; r < rep && len + k + 96u < sizeof buf; r++) {
                        memcpy(buf + len, sy, k);
                        len += k;
                    }
                }
            }
            fz_check(buf, len);
        }
    } else {
        fprintf(stderr, "usage: fuzz [-x] [-p] ex LMAX [NSYM] | rnd|byt N MAXLEN SEED\n");
        return 2;
    }
    printf("fuzz: %" PRIu64 " cases, %" PRIu64 " mismatches\n", fz_cases, fz_bad);
    return fz_bad ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "fuzz") == 0) { return fuzz_main(argc - 1, argv + 1); }
    size_t mib = 64;
    int reps = 15, only = -1;
    uint64_t cap = TOKS_CHUNK_PIECES;
    int ai = 1;
    for (; ai < argc && argv[ai][0] == '-'; ai += 2) {
        if (ai + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[ai]); return 2; }
        if (strcmp(argv[ai], "-m") == 0) { mib = (size_t)atol(argv[ai + 1]); }
        else if (strcmp(argv[ai], "-r") == 0) { reps = atoi(argv[ai + 1]); }
        else if (strcmp(argv[ai], "-c") == 0) { cap = (uint64_t)atol(argv[ai + 1]); }
        else if (strcmp(argv[ai], "-v") == 0) { only = atoi(argv[ai + 1]); }
        else { fprintf(stderr, "unknown option %s\n", argv[ai]); return 2; }
    }
    if (ai >= argc || cap == 0u || cap > (1u << 16) || reps < 1 || mib == 0u || mib > 512u) {
        fprintf(stderr, "usage: %s [-m MiB] [-r abba-blocks] [-c cap<=65536] [-v variant] name=spec ...\n", argv[0]);
        return 2;
    }
    build();
    const size_t want = mib << 20;
    uint32_t *ends_c = malloc(want * sizeof(uint32_t));
    uint32_t *ends_n = malloc(want * sizeof(uint32_t));
    if (!ends_c || !ends_n) { fprintf(stderr, "out of memory\n"); return 2; }
    printf("bench_k3_neon: %zu MiB per input, cap %" PRIu64 ", %d abba blocks (%d paired samples), load %.2f\n",
           mib, cap, reps, 2 * reps, loadavg());
    printf("%-8s %-7s %9s %8s %9s %7s %15s %8s %9s %6s\n", "input", "variant", "pieces", "c MB/s", "neon MB/s",
           "speedup", "95% interval", "c best", "neon best", "load");
    int bad = 0;
    for (; ai < argc; ai++) {
        char *eq = strchr(argv[ai], '=');
        if (!eq) { fprintf(stderr, "input must be name=spec: %s\n", argv[ai]); return 2; }
        *eq = '\0';
        const char *name = argv[ai];
        buf src = { 0 };
        add_spec(&src, eq + 1, want);
        if (src.n == 0u) { fprintf(stderr, "%s: empty input\n", name); return 2; }
        uint8_t *text = malloc(want);
        if (!text) { fprintf(stderr, "out of memory\n"); return 2; }
        for (size_t i = 0; i < want; i += src.n) {      /* bound: want / src.n repetitions */
            memcpy(text + i, src.p, (want - i < src.n) ? want - i : src.n);
        }
        free(src.p);
        for (int v = 0; v < NV; v++) {
            if (only >= 0 && v != only) { continue; }
            const toks_tables *t = &TT[v];
            /* correctness first: one unbounded-cap pass each, ends compared */
            toks_k3_args a;
            memset(&a, 0, sizeof a);
            a.text = text; a.len = want; a.pos = 0; a.ends = ends_c; a.cap = want;
            uint64_t nc = toks_k3_scan_cl100k_c(t, &a);
            memset(&a, 0, sizeof a);
            a.text = text; a.len = want; a.pos = 0; a.ends = ends_n; a.cap = want;
            uint64_t nn = toks_k3_scan_cl100k_neon(t, &a);
            if (nn != nc || memcmp(ends_c, ends_n, (size_t)nc * sizeof(uint32_t)) != 0) {
                uint64_t i = 0;
                while (i < nc && i < nn && ends_c[i] == ends_n[i]) { i++; }
                fprintf(stderr, "%s %s: MISMATCH c %" PRIu64 " pieces, neon %" PRIu64 ", first diff #%" PRIu64
                        " (c %u neon %u)\n", name, V[v].name, nc, nn, i,
                        i < nc ? ends_c[i] : 0u, i < nn ? ends_n[i] : 0u);
                bad++;
                continue;
            }
            int np = 2 * reps;
            double *sc = malloc((size_t)np * sizeof *sc), *sn = malloc((size_t)np * sizeof *sn);
            double *ratio = malloc((size_t)np * sizeof *ratio);
            if (!sc || !sn || !ratio) { fprintf(stderr, "out of memory\n"); return 2; }
            for (int r = 0; r < reps; r++) {             /* bound: reps abba blocks */
                double tc[2], tn[2];
                for (int k = 0; k < 4; k++) {            /* a b b a: c neon neon c */
                    int is_c = (k == 0 || k == 3);
                    double t0 = now();
                    uint64_t pk = drive(is_c ? toks_k3_scan_cl100k_c : toks_k3_scan_cl100k_neon, t, text, want, cap);
                    double t1 = now();
                    if (pk != nc) { fprintf(stderr, "piece count changed under cap\n"); return 3; }
                    if (is_c) { tc[k == 3] = t1 - t0; } else { tn[k == 2] = t1 - t0; }
                }
                for (int j = 0; j < 2; j++) {
                    sc[2 * r + j] = (double)want / tc[j] / 1e6;
                    sn[2 * r + j] = (double)want / tn[j] / 1e6;
                    ratio[2 * r + j] = tc[j] / tn[j];
                }
            }
            double bc = 0.0, bn = 0.0;
            for (int i = 0; i < np; i++) {
                if (sc[i] > bc) { bc = sc[i]; }
                if (sn[i] > bn) { bn = sn[i]; }
            }
            double mean, lo, hi;
            boot(ratio, np, &mean, &lo, &hi);
            printf("%-8s %-7s %9" PRIu64 " %8.1f %9.1f %6.2fx  [%5.2f, %5.2f] %8.1f %9.1f %6.2f\n", name, V[v].name, nc,
                   median(sc, np), median(sn, np), mean, lo, hi, bc, bn, loadavg());
            fflush(stdout);
            free(sc); free(sn); free(ratio);
        }
        free(text);
    }
    free(ends_c);
    free(ends_n);
    return bad ? 1 : 0;
}
