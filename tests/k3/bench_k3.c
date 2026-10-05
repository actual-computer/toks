/*
 * tests/k3/bench_k3.c: speed of a K3 tier against its c twin, one thread, with SPEC §12.5 statistics.
 * Not part of `make test`; built by hand on the tier's host against the library as shipped (`make lib`:
 * the c twin at -O3, the asm tiers of the isa):
 *
 *   make -j lib
 *   clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core \
 *     -Isrc/platform -DK3_SCAN=toks_k3_scan_cl100k_avx2 -o build/bench_k3_avx2 tests/k3/bench_k3.c \
 *     build/linux-x86_64/libtoks.a
 *
 *   bench_k3 file  R VARIANT FILE...    each FILE one segment
 *   bench_k3 segs  R VARIANT FILE L...  FILE cut into segments of about L bytes (at character starts; the
 *                                       first 4 MB of FILE when L < 64, else the first 32 MB)
 *   bench_k3 cases R FILE.bin           a tests/k3/gen.py stream, each text its own segment under its own
 *                                       variant: all, split by "hf has a piece >= 64 B", and by length
 *
 * VARIANT is gpt2, cl100k, qwen2, qwen35 (tests/c/test_k3.c's) or digits (digits-gpt2, A8); with -DK3_O200K the template is o200k
 * (twin toks_k3_scan_o200k_c) and VARIANT o200k, nemo or kimi (docs/templates/o200k.md; no cases mode). Segments are scanned the driver's way:
 * cap TOKS_CHUNK_PIECES, resumed at a.pos, every call through the call sites' short rule first (kernels.h
 * toks_k3_short, one copy for every tier): c = the scalar tier (the twin behind the short rule), A = K3_SCAN's.
 * Before timing, every tier's ends are compared with the twin's on
 * exactly the timed segments (a mismatch voids the cell). Timing: a warm-up pass of each, then R rounds of
 * c, A, A, c (abba). Reported per cell: each tier's MB/s (10^6 B/s) at its median round time;
 * "A/c x r [lo, hi]": A's speed over the twin's, the median of the R paired per-round ratios with its 95%
 * bootstrap interval (2000 resamples of the rounds); ns per call; and bytes moved per input byte,
 * (n + 4 pieces) / n: the text read once, the ends written once. -DK3_SCAN_B=<symbol> adds a second build
 * of the tier (e.g. the previous kernel assembled under another name) to the same rounds, c A B B A c, and
 * prints B/c and A/B too. /proc/loadavg at start and end (SPEC §12.8: pin with taskset, record load).
 */
#define _POSIX_C_SOURCE 200809L                /* clock_gettime under -std=c17 */
#include "../../src/core/classes.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"

#ifndef K3_SCAN
#error "build with -DK3_SCAN=toks_k3_scan_<tmpl>_<tier>"
#endif
uint64_t K3_SCAN(const toks_tables *t, toks_k3_args *a);
#ifdef K3_SCAN_B                                       /* a second build of the tier: A/B in the same rounds */
uint64_t K3_SCAN_B(const toks_tables *t, toks_k3_args *a);
#define NT 3
#else
#define NT 2
#endif
#define STR2(x) #x
#define STR(x) STR2(x)

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint64_t (*k3_fn)(const toks_tables *, toks_k3_args *);
static k3_fn TIER[3];
static const char *TNAME[3];

#if defined(K3_DSV3)                                   /* dsv3 (docs/templates/dsv3.md): one variant, its own tables */
#define K3_TWIN toks_k3_scan_dsv3_c
#define NV 1
static const char *VNAME[4] = { "dsv3", "-", "-", "-" };
static const uint32_t VPARAM[4] = { 0, 0, 0, 0 };
#elif defined(K3_O200K)
#define K3_TWIN toks_k3_scan_o200k_c
#define NV 3
static const char *VNAME[4] = { "o200k", "nemo", "kimi", "-" };
static const uint32_t VPARAM[4] = {
    TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1_3, TOKS_TP_CONTR_NONE | TOKS_TP_DIGITS_1,
    TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1_3 | TOKS_TP_HAN | TOKS_TP_NO_SLASH, 0,
};
#else
#define K3_TWIN toks_k3_scan_cl100k_c
#define NV 5
static const char *VNAME[5] = { "gpt2", "cl100k", "qwen2", "qwen35", "digits" };
static const uint32_t VPARAM[5] = {
    TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN,
    TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL,
    TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL,
    TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL,
    TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_1 | TOKS_TP_DIGIT_CUT,
};
#endif
/* A (and B): the tier's whole kernel, its asm part with the c twin's stretches (kernels.h toks_k3_tier), inlined
 * into the pass as the driver inlines it; -DK3_B_DIRECT calls B alone, with no short rule (a kernel built before
 * the part / twin split) */
static uint64_t whole_a(const toks_tables *t, toks_k3_args *a) { return toks_k3_tier(K3_SCAN, K3_TWIN, t, a); }
#ifdef K3_SCAN_B
#ifdef K3_B_DIRECT
#define B_CALL(t, a) K3_SCAN_B(t, a)
#define B_SHORT 0
#else
#define B_CALL(t, a) toks_k3_parts(K3_SCAN_B, K3_TWIN, t, a)
#define B_SHORT 1
#endif
static uint64_t whole_b(const toks_tables *t, toks_k3_args *a) { return B_SHORT ? toks_k3_tier(K3_SCAN_B, K3_TWIN, t, a) : B_CALL(t, a); }
#else
#define B_CALL(t, a) 0
#define B_SHORT 1
#endif
static uint8_t cls_buf[2][128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CT[2];
static toks_tables TT[5];

static void build(void)
{
    for (int i = 0; i < 2; i++) {
#if defined(K3_DSV3)
        uint32_t cf0 = TOKS_CLASSES_DSV3, cf1 = TOKS_CLASSES_DSV3;
#elif defined(K3_O200K)
        uint32_t cf0 = 0u, cf1 = TOKS_CLASSES_HAN;      /* kimi's tables */
#else
        uint32_t cf0 = 0u, cf1 = TOKS_CLASSES_MARKS_ARE_LETTERS;
#endif
        if (toks_classes_build(i ? cf1 : cf0, cls_buf[i], sizeof cls_buf[i], &CT[i]) <= 0) {
            fprintf(stderr, "classes build failed\n");
            exit(2);
        }
    }
    for (int v = 0; v < NV; v++) {
        toks_tables *t = &TT[v];
#ifdef K3_O200K
        int i = v == 2;                                /* kimi: the Han tables */
#else
        int i = v == 3;                                /* qwen35 folds marks into letters */
#endif
        memset(t, 0, sizeof *t);
        t->magic = TOKS_TABLES_MAGIC;
        t->version = TOKS_TABLES_VERSION;
#if defined(K3_DSV3)
        t->tmpl = TOKS_TMPL_DSV3;
#elif defined(K3_O200K)
        t->tmpl = TOKS_TMPL_O200K;
#else
        t->tmpl = TOKS_TMPL_CL100K;
#endif
        t->tmpl_params = VPARAM[v];
        t->cls_ascii = CT[i].ascii;
        t->cls_stage1 = CT[i].stage1;
        t->cls_stage2 = CT[i].stage2;
        t->cls_nblocks = CT[i].n_blocks;
        int64_t cf = toks_compile_cls_flags(t);           /* the TOKS_TF_CJK_* flags, as the compiler sets them */
        if (cf < 0) { fprintf(stderr, "class tables break kernels.md §2\n"); exit(2); }
        t->flags = (uint32_t)cf;
    }
}

typedef struct { const uint8_t *p; uint64_t len; uint8_t v; } seg;
typedef struct { seg *s; uint64_t n, cap, bytes; } segset;

static void add(segset *ss, const uint8_t *p, uint64_t len, uint8_t v)
{
    if (ss->n == ss->cap) {
        ss->cap = ss->cap ? 2 * ss->cap : 1024;
        ss->s = realloc(ss->s, ss->cap * sizeof *ss->s);
        if (!ss->s) { fprintf(stderr, "oom\n"); exit(2); }
    }
    ss->s[ss->n].p = p;
    ss->s[ss->n].len = len;
    ss->s[ss->n].v = v;
    ss->n++;
    ss->bytes += len;
}

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(2); }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

static uint32_t ENDS[TOKS_CHUNK_PIECES];
static volatile uint64_t sink;

/* one pass over the set, the driver's way, tier k; returns pieces, *calls the K3 calls */
static uint64_t pass(int k, const segset *ss, uint64_t *calls)
{
    uint64_t pieces = 0, nc = 0;
    for (uint64_t i = 0; i < ss->n; i++) {             /* bound: segments */
        const seg *s = &ss->s[i];
        uint64_t pos = 0;
        while (pos < s->len) {                         /* bound: pieces of the segment */
            toks_k3_args a;
            a.text = s->p; a.len = s->len; a.pos = pos; a.ends = ENDS; a.cap = TOKS_CHUNK_PIECES;
            a.n = 0; a.flags = 0; a.rsv = 0;
            const toks_tables *t = &TT[s->v];
            uint64_t n;
            if ((k < 2 || B_SHORT) && toks_k3_short(K3_TWIN, t, &a, &n)) {
                pieces += n;                           /* the call sites' short rule: the same code for every tier */
            } else {
                pieces += k == 0 ? K3_TWIN(t, &a) : k == 1 ? toks_k3_parts(K3_SCAN, K3_TWIN, t, &a) : B_CALL(t, &a);
            }
            pos = a.pos;
            nc++;
        }
    }
    sink += pieces;
    if (calls) { *calls = nc; }
    return pieces;
}

/* exactness outside the timer: every call of every tier on the set, ends and pos equal to the c twin's */
static int same(const segset *ss)
{
    static uint32_t e0[TOKS_CHUNK_PIECES], e1[TOKS_CHUNK_PIECES];
    for (uint64_t i = 0; i < ss->n; i++) {
        const seg *s = &ss->s[i];
        uint64_t pos = 0;
        while (pos < s->len) {
            toks_k3_args a, b;
            memset(&a, 0, sizeof a);
            a.text = s->p; a.len = s->len; a.pos = pos; a.ends = e0; a.cap = TOKS_CHUNK_PIECES;
            uint64_t n0 = K3_TWIN(&TT[s->v], &a);
            for (int k = 1; k < NT; k++) {
                b = a;
                b.pos = pos; b.ends = e1; b.n = 0;
                uint64_t n1 = TIER[k](&TT[s->v], &b);
                if (n0 == 0 || n0 != n1 || a.pos != b.pos || memcmp(e0, e1, (size_t)n0 * 4) != 0) {
                    fprintf(stderr, "DIFFER %s segment %" PRIu64 " at pos %" PRIu64 "\n", TNAME[k], i, pos);
                    return 0;
                }
            }
            pos = a.pos;
        }
    }
    return 1;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void load(const char *when)
{
    FILE *f = fopen("/proc/loadavg", "r");
    char b[128] = "n/a";
    if (f) { if (!fgets(b, sizeof b, f)) { strcpy(b, "n/a"); } fclose(f); }
    b[strcspn(b, "\n")] = 0;
    printf("# load %s: %s\n", when, b);
}

static uint64_t xs = 88172645463325252ull;
static uint64_t xr(void) { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return xs; }
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double median(const double *v, int n)
{
    double *w = malloc((size_t)n * sizeof *w);
    if (!w) { exit(2); }
    memcpy(w, v, (size_t)n * sizeof *w);
    qsort(w, (size_t)n, sizeof *w, dcmp);
    double m = n & 1 ? w[n / 2] : 0.5 * (w[n / 2 - 1] + w[n / 2]);
    free(w);
    return m;
}
/* 95% bootstrap interval of the median of v */
static void boot(const double *v, int n, double *lo, double *hi)
{
    enum { B = 2000 };
    double *m = malloc(B * sizeof *m), *w = malloc((size_t)n * sizeof *w);
    if (!m || !w) { exit(2); }
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < n; i++) { w[i] = v[xr() % (uint64_t)n]; }
        m[b] = median(w, n);
    }
    qsort(m, B, sizeof *m, dcmp);
    *lo = m[(int)(0.025 * B)];
    *hi = m[(int)(0.975 * B)];
    free(m);
    free(w);
}

/* one cell: exactness, warm-up, R rounds of c A A c (c A B B A c with B); one line */
static int race(const char *label, const segset *ss, int R)
{
    if (!same(ss)) { printf("%-40s MISMATCH: run void\n", label); return 1; }
    uint64_t calls, pieces = pass(0, ss, &calls);                        /* warm-up */
    for (int k = 1; k < NT; k++) { pass(k, ss, NULL); }
    double *t[3], *q = calloc((size_t)R, sizeof(double));
    for (int k = 0; k < NT; k++) { t[k] = calloc((size_t)R, sizeof(double)); if (!t[k]) { exit(2); } }
    if (!q) { exit(2); }
    for (int r = 0; r < R; r++) {                      /* bound: R rounds */
        for (int j = 0; j < 2 * NT; j++) {
            int k = j < NT ? j : 2 * NT - 1 - j;
            double t0 = now();
            pass(k, ss, NULL);
            t[k][r] += now() - t0;
        }
    }
    double mb = (double)ss->bytes / 1e6, m[3];
    for (int k = 0; k < NT; k++) { m[k] = median(t[k], R) / 2; }
    printf("%-40s %9.2f MB %9" PRIu64 " segs %9" PRIu64 " calls | MB/s", label, mb, ss->n, calls);
    for (int k = 0; k < NT; k++) { printf(" %s %.1f", TNAME[k], mb / m[k]); }
    printf(" |");
    for (int k = 1; k <= NT; k++) {                    /* c/A, c/B, then A over B */
        int num = k < NT ? 0 : 2, den = k < NT ? k : 1;
        if (k == NT && NT < 3) { break; }
        for (int r = 0; r < R; r++) { q[r] = t[num][r] / t[den][r]; }
        double lo, hi;
        boot(q, R, &lo, &hi);
        printf(" %s/%s x%.3f [%.3f, %.3f]", TNAME[den], TNAME[num], median(q, R), lo, hi);
    }
    printf(" | ns/call");
    for (int k = 0; k < NT; k++) { printf(" %s %.1f", TNAME[k], m[k] * 1e9 / (double)calls); }
    printf(" | %.3f B moved/B\n", ss->bytes ? (double)(ss->bytes + 4 * pieces) / (double)ss->bytes : 0.0);
    fflush(stdout);
    for (int k = 0; k < NT; k++) { free(t[k]); }
    free(q);
    return 0;
}

static int vidx(const char *s)
{
    for (int v = 0; v < NV; v++) { if (!strcmp(s, VNAME[v])) { return v; } }
    fprintf(stderr, "unknown variant %s\n", s);
    exit(2);
}

/* share of bytes in pieces >= 64 B (the c twin), for the label */
static double long_share(const segset *ss)
{
    static uint32_t e[1 << 16];
    uint64_t lb = 0;
    for (uint64_t i = 0; i < ss->n; i++) {
        const seg *s = &ss->s[i];
        uint64_t pos = 0;
        while (pos < s->len) {
            toks_k3_args a;
            memset(&a, 0, sizeof a);
            a.text = s->p; a.len = s->len; a.pos = pos; a.ends = e; a.cap = 1 << 16;
            uint64_t n = K3_TWIN(&TT[s->v], &a);
            for (uint64_t j = 0; j < n; j++) { lb += e[j] - pos >= 64 ? e[j] - pos : 0; pos = e[j]; }
        }
    }
    return ss->bytes ? 100.0 * (double)lb / (double)ss->bytes : 0.0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s file R VARIANT FILE... | segs R VARIANT FILE L... | cases R FILE.bin\n", argv[0]);
        return 2;
    }
    build();
    TIER[0] = K3_TWIN;
    TNAME[0] = "c";
    TIER[1] = whole_a;
    TNAME[1] = "A";
#ifdef K3_SCAN_B
    TIER[2] = whole_b;
    TNAME[2] = "B";
    printf("# A = %s, B = %s\n", STR(K3_SCAN), STR(K3_SCAN_B));
#else
    printf("# A = %s\n", STR(K3_SCAN));
#endif
    int R = atoi(argv[2]), bad = 0;
    if (R < 1) { R = 1; }
    load("start");
    char label[128];
    if (!strcmp(argv[1], "file")) {
        int v = vidx(argv[3]);
        for (int i = 4; i < argc; i++) {
            uint64_t len;
            uint8_t *p = slurp(argv[i], &len);
            segset ss = { 0 };
            add(&ss, p, len, (uint8_t)v);
            const char *b = strrchr(argv[i], '/');
            snprintf(label, sizeof label, "%s %s (long %.1f%%)", b ? b + 1 : argv[i], VNAME[v], long_share(&ss));
            bad |= race(label, &ss, R);
            free(ss.s);
            free(p);
        }
    } else if (!strcmp(argv[1], "segs") && argc >= 6) {
        int v = vidx(argv[3]);
        uint64_t len;
        uint8_t *p = slurp(argv[4], &len);
        for (int i = 5; i < argc; i++) {
            uint64_t L = strtoull(argv[i], NULL, 10), off = 0;
            uint64_t limit = L < 64 ? (4u << 20) : (32u << 20);
            if (limit > len) { limit = len; }
            segset ss = { 0 };
            while (L > 0 && off < limit) {             /* bound: limit / L segments */
                uint64_t e = off + L < limit ? off + L : limit;
                while (e > off + 1 && e < limit && (p[e] & 0xC0) == 0x80) { e--; }   /* a character start */
                add(&ss, p + off, e - off, (uint8_t)v);
                off = e;
            }
            const char *b = strrchr(argv[4], '/');
            snprintf(label, sizeof label, "%s L=%" PRIu64 " %s", b ? b + 1 : argv[4], L, VNAME[v]);
            bad |= race(label, &ss, R);
            free(ss.s);
        }
        free(p);
#if !defined(K3_O200K) && !defined(K3_DSV3)
    } else if (!strcmp(argv[1], "cases")) {
        uint64_t clen, off = 0;
        uint8_t *cs = slurp(argv[3], &clen);
        segset all = { 0 }, bk[6] = { { 0 } }, haslong = { 0 }, nolong = { 0 };
        static const uint64_t edge[6] = { 16, 64, 256, 1024, 4096, ~0ull };
        static const char *bn[6] = { "len<16", "16-63", "64-255", "256-1023", "1024-4095", ">=4096" };
        while (off + 9 <= clen) {                      /* bound: records */
            uint8_t vi = cs[off];
            uint32_t len, np;
            memcpy(&len, cs + off + 1, 4);
            if (vi > 3 || off + 9 + (uint64_t)len > clen) { fprintf(stderr, "corrupt case stream\n"); return 2; }
            memcpy(&np, cs + off + 5 + len, 4);
            const uint8_t *p = cs + off + 5;
            add(&all, p, len, vi);
            int b = 0;
            while (len >= edge[b]) { b++; }
            add(&bk[b], p, len, vi);
            int lg = 0;                                /* hf's split holds a piece >= 64 B ((start, end) u32) */
            for (uint32_t j = 0; j < np; j++) {
                uint32_t s0, e0;
                memcpy(&s0, cs + off + 9 + len + 8 * (uint64_t)j, 4);
                memcpy(&e0, cs + off + 13 + len + 8 * (uint64_t)j, 4);
                lg |= e0 - s0 >= 64;
            }
            add(lg ? &haslong : &nolong, p, len, vi);
            off += 9 + (uint64_t)len + 8 * (uint64_t)np;
        }
        snprintf(label, sizeof label, "cases all (long %.1f%%)", long_share(&all));
        bad |= race(label, &all, R);
        snprintf(label, sizeof label, "cases without a piece >= 64 (long %.1f%%)", long_share(&nolong));
        bad |= race(label, &nolong, R);
        snprintf(label, sizeof label, "cases with a piece >= 64 (long %.1f%%)", long_share(&haslong));
        bad |= race(label, &haslong, R);
        for (int b = 0; b < 6; b++) {
            if (!bk[b].n) { continue; }
            snprintf(label, sizeof label, "cases %s (long %.1f%%)", bn[b], long_share(&bk[b]));
            bad |= race(label, &bk[b], R);
        }
#endif
    } else {
        fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }
    load("end");
    return bad;
}
