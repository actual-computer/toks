/*
 * tests/bpe/bench_x86.c: K6 and K5, c twin vs asm (avx2; avx512 when built with -DBENCH_AVX512), one
 * core, real-text pieces. NOT part of `make test`: the speed receipts for SPEC §10.5's rent rule,
 * each kernel reported on its own.
 *
 *   bench_x86 PIECES.bin [REPS [lens|k6]]   (REPS passes per cell, default 5; run under taskset -c <core>)
 *   bench_x86 PIECES.bin selfcheck          (the checks below, then exit)
 *
 * PIECES.bin is tests/bpe/gen.py's stream format (the model header, then records u32 len, bytes, u32 n,
 * n ids) holding the pre-tokenizer pieces of real text IN TEXT ORDER with hf 0.23.2's ids, made with
 * gen.py's own model and record code:
 *
 *   uv run --with tokenizers==0.23.2 python - TOKENIZER_JSON TEXT... > PIECES.bin <<'EOF'
 *   import struct, sys; sys.path.insert(0, "tests/bpe"); import gen
 *   path = sys.argv[1]; gen.init(path); out = sys.stdout.buffer; out.write(gen.model_header(path))
 *   for tp in sys.argv[2:]:
 *       s = open(tp, "rb").read().decode("utf-8", "replace")
 *       for p, _ in gen.TK.pre_tokenizer.pre_tokenize_str(s):
 *           r = gen.raw(p)
 *           if r: out.write(gen.record(r))
 *   out.write(struct.pack("<I", 0xFFFFFFFF))
 *   EOF
 *
 * The tables are built from the stream's model with toks_bpe_build (tests/bpe/check.c shows the real load
 * path builds the same model for every pinned tokenizer it accepts). Build (from the repo root):
 *
 *   BUILD_DIR=build/bench make -s lib
 *   clang -std=c17 -O3 -fno-strict-aliasing -fwrapv -fcf-protection=full -Wall -Wextra -Werror -Iinclude \
 *     -Isrc/core -Isrc/platform -Itests/common -o build/bench/bench_x86 tests/bpe/bench_x86.c \
 *     tests/common/guard.c tests/common/abicheck_x86_64.S build/bench/libtoks.a
 *
 * selfcheck, on the stream's real tables (what test_bpe's small models cannot reach), the asm against the c
 * twin, the asm called through tests/common's abi checker (every callee-saved register canaried):
 *   K6 and K5 geometry   segments of real pieces, every length 0..255, the text flush against a guard page
 *                        at its end, and at its start for every alignment 0..63; out (room len + 4 ids for
 *                        K5, len for K6) and work (TOKS_BPE_WORK_BYTES, rounded to 64) flush at their ends;
 *                        K5 cold then warm on a 16-bucket cache: ids, counters and the cache bytes compared.
 *   vhash collisions     ignore_merges only: for vocabulary tokens of >= 5 bytes, a different string of the
 *                        same length and the same vocab hash (crc32c is linear: a kernel vector of the
 *                        map from the first 40 bits to h), which must not come out as that token.
 *
 * Cells (best of REPS passes; the tiers alternate ABBA pass by pass):
 *   K6 all      every piece through K6 on its own, in text order, ids appended to one buffer.
 *   K6 first    the first occurrence of each distinct piece, in text order: what K6 sees behind an
 *               unbounded cache (the cold miss stream). With `lens`, also by piece length.
 *   K5 tier     the text as one segment in chunks of 256 pieces (the driver's K3 cap), the driver's
 *               dynamic cache (core.h TOKS_CACHE_BUCKETS), cold (zeroed before each pass) and warm (left by
 *               the previous pass): k5_c + k6_c against k5_<asm> + k6_<asm>, what a tier switch buys.
 *   K5 alone    the same, both K5 tiers calling the asm K6 (k5_c.c compiled into this file against
 *               toks_k6_bpe_avx2): K5's own share of the difference.
 *   K5 hot      K5 alone over the first 16 chunks (4096 pieces) 64 times, an 8192-bucket cache warmed
 *               first: (nearly) every piece a cache or static hit, the hit path's own cost.
 *   every K5 cell above runs each chunk as a FRESH scratch (counters 0 in: static table first, static answers
 *   never fill; kernels.md §6), e2e's cold 4 KiB state; the "wo" cells run the same with the counters starting
 *   at TOKS_K5_WARM, a WARM scratch (cache first, static answers fill), e2e's pass / warm / whole-file states:
 *   K5 wo cold (cache zeroed first), K5 wo warm (left by the previous pass), K5 wo hot.
 * -DBENCH_EXTRA_K5=<symbol> adds a tier "x" whose K5 is another build of k5_avx2.S under that name (A/B of two
 * asm versions in the same run: assemble the other source with -Dtoks_k5_encode_avx2=<symbol>); -DBENCH_EXTRA_K5B
 * a third, tier "y".
 * Every pass's ids (and K5's counters) are checked against hf's ids outside the timer (SPEC §12.4); a
 * pass that differs voids the run (exit 1). K6 alone is held to hf's ids except where kernels.md §6's K5 / K6
 * CONTRACT lets it differ (tests/bpe/check.c's rule, counted on the header line: under ignore_merges with
 * TOKS_TF_PROBE_LONG, a 2..15-byte model token its own merges do not rebuild, answered by K5's words); there
 * every tier must give the c twin's merges. Load averages are printed before and after.
 */
#define _POSIX_C_SOURCE 200809L    /* clock_gettime under -std=c17 */
#include "../../src/core/bpe.h"
#include "../../src/core/kernels.h"
#include "../common/guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t toks_k6_bpe_avx2(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k5_encode_avx2(const toks_tables *t, toks_k5_args *a);
#if defined(BENCH_AVX512)
uint64_t toks_k6_bpe_avx512(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k5_encode_avx512(const toks_tables *t, toks_k5_args *a);
#endif
#if defined(BENCH_EXTRA_K6)      /* tuning: another K6 build under this name, A/B in the same run */
uint64_t BENCH_EXTRA_K6(const toks_tables *t, toks_k6_args *a);
#endif
#if defined(BENCH_EXTRA_K5)      /* tuning: other K5 builds under these names, A/B in the same run */
#  if defined(BENCH_EXTRA_K6)
#    error "BENCH_EXTRA_K5 and BENCH_EXTRA_K6 both name their tier x: build one at a time"
#  endif
uint64_t BENCH_EXTRA_K5(const toks_tables *t, toks_k5_args *a);
#endif
#if defined(BENCH_EXTRA_K5B)
uint64_t BENCH_EXTRA_K5B(const toks_tables *t, toks_k5_args *a);
#endif

/* K5's c twin again, calling the avx2 K6: the "K5 alone" cells */
#define toks_k5_encode_c toks_k5_encode_c_on_avx2
#define toks_k6_bpe_c toks_k6_bpe_avx2
#include "../../src/core/k5_c.c"
#undef toks_k5_encode_c
#undef toks_k6_bpe_c

typedef uint64_t (*k6_fn)(const toks_tables *, toks_k6_args *);   /* bench code: pointers are fine here */
typedef uint64_t (*k5_fn)(const toks_tables *, toks_k5_args *);

#define CHUNK    256u
#define NLEN     8
#define NK5      8
#define MAXCELLS (2 + NLEN + NK5)
#define HOT_CHUNKS 16u
#define HOT_REPS   64u

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

static void *xalloc(uint64_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(uint64_t)63);
    if (p == NULL) { fprintf(stderr, "out of memory (%" PRIu64 " bytes)\n", n); exit(2); }
    return p;
}

static int rd(void *p, size_t n, FILE *f) { return fread(p, 1, n, f) == n; }

static void loadavg(const char *when)
{
    char buf[128] = "?";
    FILE *f = fopen("/proc/loadavg", "r");
    if (f != NULL) {
        if (fgets(buf, sizeof buf, f) == NULL) { buf[0] = 0; }
        fclose(f);
    }
    buf[strcspn(buf, "\n")] = 0;
    printf("load %s: %s\n", when, buf);
}

/* ---- the run's data ---- */
static toks_tables T;
static uint8_t *TEXT;                 /* every piece, in order */
static uint32_t *ENDS;                /* piece ends in TEXT */
static uint32_t *WANT;                /* hf's ids, concatenated */
static uint64_t *WOFF;                /* piece i's ids are WANT[WOFF[i], WOFF[i + 1]) */
static uint32_t *K6W;                 /* K6 alone's answer per piece (hf's, but see k6_reference), concatenated */
static uint64_t *K6O;                 /* piece i's K6 ids are K6W[K6O[i], K6O[i + 1]) */
static uint64_t NP, NB, NW, MAXLEN;
static uint32_t *OUT;                 /* NW + 4 ids */
static uint8_t *WORK;

static uint64_t pstart(uint64_t i) { return i == 0 ? 0 : ENDS[i - 1]; }

/* a list of pieces for the K6 cells, with hf's ids for exactly those pieces */
typedef struct { const char *name; uint32_t *idx; uint64_t n, bytes, nw; uint32_t *want; } subset;

static void subset_finish(subset *ss)
{
    ss->nw = 0;
    ss->bytes = 0;
    for (uint64_t k = 0; k < ss->n; k++) { uint64_t i = ss->idx[k]; ss->nw += K6O[i + 1] - K6O[i]; ss->bytes += ENDS[i] - pstart(i); }
    ss->want = xalloc(4 * ss->nw + 4);
    uint64_t w = 0;
    for (uint64_t k = 0; k < ss->n; k++) {
        uint64_t i = ss->idx[k];
        memcpy(ss->want + w, K6W + K6O[i], 4 * (K6O[i + 1] - K6O[i]));
        w += K6O[i + 1] - K6O[i];
    }
}

/* K6 alone's answer for every piece, outside the timers: hf's, except where kernels.md §6's K5 / K6 CONTRACT lets
 * them differ (tests/bpe/check.c's rule): under ignore_merges with TOKS_TF_PROBE_LONG, a piece of 2..15 bytes that is
 * a model token whose own merges do not rebuild it (hf: [that token], K6 alone: the merges) and whose static words
 * entry is hf's answer (K5 answers it before K6). There the c twin's merges are K6's answer. Any other difference
 * from hf is a K6 bug: the run stops. Returns the pieces answered by the words, or -1. */
static int64_t k6_reference(int ignore_merges)
{
    K6W = xalloc(4 * (NB + 4));                            /* K6 writes at most len ids a piece */
    K6O = xalloc(8 * (NP + 1));
    uint64_t w = 0;
    int64_t by_words = 0;
    for (uint64_t i = 0; i < NP; i++) {
        uint64_t s = pstart(i), len = ENDS[i] - s, n = WOFF[i + 1] - WOFF[i];
        const uint32_t *want = WANT + WOFF[i];
        K6O[i] = w;
        toks_k6_args a = { TEXT + s, len, K6W + w, WORK, TOKS_BPE_WORK_BYTES(MAXLEN), 0, 0, 0 };
        uint64_t ng = toks_k6_bpe_c(&T, &a);
        if (ng == n && memcmp(K6W + w, want, 4 * n) == 0) { w += n; continue; }
        int ok = 0;
        if (ignore_merges && (T.flags & TOKS_TF_PROBE_LONG) != 0u && len >= 2 && len <= TOKS_KEY_MAXLEN && n == 1) {
            bpe_key k = bpe_key_at(TEXT + s, len, 0, len);
            const uint8_t *v = bpe_words_probe(T.words, T.words_mask, bpe_key_hash(k), k);
            uint32_t wv[4], vid = 0;
            ok = bpe_vhash_find(&T, TEXT + s, len, &vid) && vid == want[0] && v != NULL && bpe_val_put(v, wv) == 1 &&
                 wv[0] == vid;
        }
        if (!ok) { fprintf(stderr, "piece %" PRIu64 ": the c twin's K6 differs from hf outside the contract\n", i); return -1; }
        by_words++;
        w += ng;
    }
    K6O[NP] = w;
    return by_words;
}

static double run_k6(k6_fn f, const subset *ss, const char *what, int *ok)
{
    double t0 = now_ns();
    uint64_t n = 0;
    for (uint64_t k = 0; k < ss->n; k++) {
        uint64_t i = ss->idx[k], s = pstart(i);
        toks_k6_args a = { TEXT + s, ENDS[i] - s, OUT + n, WORK, TOKS_BPE_WORK_BYTES(MAXLEN), 0, 0, 0 };
        n += f(&T, &a);
    }
    double dt = now_ns() - t0;
    if (n != ss->nw || memcmp(OUT, ss->want, 4 * n) != 0) {
        fprintf(stderr, "%s, %s: ids differ from hf\n", what, ss->name);
        *ok = 0;
    }
    return dt;
}

/* one pass of K5 over the text in chunks, each chunk's counters starting at age (0: a fresh scratch, TOKS_K5_WARM:
 * a warm one); ctr: static, cache, miss (this pass's own) */
static double run_k5(k5_fn f, uint8_t *cache, uint64_t mask, uint64_t age, const char *what, int *ok, uint64_t ctr[3])
{
    ctr[0] = ctr[1] = ctr[2] = 0;
    double t0 = now_ns();
    uint64_t n = 0;
    for (uint64_t i = 0; i < NP; i += CHUNK) {
        uint64_t k = NP - i < CHUNK ? NP - i : CHUNK, start = pstart(i);
        toks_k5_args a = { TEXT, NB, ENDS + i, k, start, OUT + n, NW + 4 - n, 0, cache, mask, WORK,
                           TOKS_BPE_WORK_BYTES(MAXLEN), age, 0, 0, 0, NULL };
        n += f(&T, &a);
        ctr[0] += a.hits_static - age; ctr[1] += a.hits_cache; ctr[2] += a.misses;
    }
    double dt = now_ns() - t0;
    if (n != NW || memcmp(OUT, WANT, 4 * NW) != 0 || ctr[0] + ctr[1] + ctr[2] != NP) {
        fprintf(stderr, "%s, K5: ids differ from hf\n", what);
        *ok = 0;
    }
    return dt;
}

/* K5 hot: the first HOT_CHUNKS chunks, HOT_REPS times, on a warmed cache; age and ctr as run_k5 (summed) */
static double run_k5_hot(k5_fn f, uint8_t *cache, uint64_t mask, uint64_t age, const char *what, int *ok, uint64_t ctr[3])
{
    uint64_t np = NP < HOT_CHUNKS * CHUNK ? NP : HOT_CHUNKS * CHUNK, nw = WOFF[np];
    ctr[0] = ctr[1] = ctr[2] = 0;
    double t0 = now_ns();
    for (uint32_t r = 0; r < HOT_REPS; r++) {
        uint64_t n = 0;
        for (uint64_t i = 0; i < np; i += CHUNK) {
            uint64_t k = np - i < CHUNK ? np - i : CHUNK, start = pstart(i);
            toks_k5_args a = { TEXT, NB, ENDS + i, k, start, OUT + n, NW + 4 - n, 0, cache, mask, WORK,
                               TOKS_BPE_WORK_BYTES(MAXLEN), age, 0, 0, 0, NULL };
            n += f(&T, &a);
            ctr[0] += a.hits_static - age; ctr[1] += a.hits_cache; ctr[2] += a.misses;
        }
        if (n != nw) { *ok = 0; }
    }
    double dt = now_ns() - t0;
    if (memcmp(OUT, WANT, 4 * nw) != 0) {
        fprintf(stderr, "%s, K5 hot: ids differ from hf\n", what);
        *ok = 0;
    }
    return dt;
}

typedef struct { const char *name; k6_fn k6; k5_fn k5; } tier;

static void row(const char *cell, const char *name, double best, double base, uint64_t np, uint64_t nb)
{
    printf("%-14s %-7s %9.2f ms %8.2f Mpieces/s %8.1f MB/s", cell, name, best / 1e6, (double)np / best * 1e3,
           (double)nb / best * 1e3);
    if (base > 0) { printf("   x%.3f vs c (%+.1f%%)", base / best, 100.0 * (base / best - 1.0)); }
    printf("\n");
}

/* ---- selfcheck ---- */
static uint64_t SC_N, SC_BAD;

static void sc_fail(const char *what, uint64_t len, int where, size_t off)
{
    if (SC_BAD++ < 20) {
        fprintf(stderr, "SELFCHECK FAIL %s: %" PRIu64 " bytes, %s flush, offset %zu\n", what, len,
                where == GUARD_END ? "end" : "start", off);
    }
}

/* K6 on src[0, len): the c twin on plain buffers, the asm through the abi checker on guarded ones */
static void sc_k6(k6_fn asm_k6, const uint8_t *src, uint64_t len, int where, size_t off)
{
    static uint32_t want[256];
    static _Alignas(64) uint8_t w0[TOKS_BPE_WORK_BYTES(256)];
    toks_k6_args a0 = { src, len, want, w0, sizeof w0, 0, 0, 0 };
    uint64_t n0 = toks_k6_bpe_c(&T, &a0);
    guard_buf gp = {0}, go = {0}, gw = {0};
    uint64_t wb = TOKS_BPE_WORK_BYTES(len), wb64 = (wb + 63) & ~(uint64_t)63;
    uint8_t *p = guard_alloc(&gp, len, where, where == GUARD_START ? off : 0);
    uint32_t *o = (uint32_t *)(void *)guard_alloc(&go, 4 * len, GUARD_END, 0);
    uint8_t *w = guard_alloc(&gw, wb64, GUARD_END, 0);
    if (p == NULL || o == NULL || w == NULL) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    memcpy(p, src, len);
    toks_k6_args a = { p, len, o, w, wb, 99, 99, 0 };
    uint64_t rep = 0;
    uint64_t n = toks_abicheck_call((const void *)asm_k6, (uint64_t)(uintptr_t)&T, (uint64_t)(uintptr_t)&a, &rep);
    SC_N++;
    if (rep != 0 || n != n0 || a.n_out != a0.n_out || a.merges != a0.merges || memcmp(o, want, 4 * n0) != 0) {
        sc_fail("K6", len, where, off);
    }
    guard_free(&gp);
    guard_free(&go);
    guard_free(&gw);
}

/* K5 over pieces ends[0, n) of src[0, len), cold then warm on a 16-bucket cache, as a fresh scratch and as a warm
 * one (the in counters start at 0 / at TOKS_K5_WARM: kernels.md §6): the c twin on plain buffers, the asm through
 * the abi checker on guarded ones; ids, counters and the cache bytes */
static void sc_k5(k5_fn asm_k5, const uint8_t *src, uint64_t len, const uint32_t *ends, uint64_t n, uint64_t maxp,
                  int where, size_t off)
{
    static uint32_t want[256 + 4];
    static _Alignas(64) uint8_t c0[16 * 64], c1[16 * 64], w0[TOKS_BPE_WORK_BYTES(256)];
    memset(c0, 0, sizeof c0);
    memset(c1, 0, sizeof c1);
    guard_buf gt = {0}, go = {0}, gw = {0};
    uint64_t wb = TOKS_BPE_WORK_BYTES(maxp), wb64 = (wb + 63) & ~(uint64_t)63;
    uint8_t *t = guard_alloc(&gt, len, where, where == GUARD_START ? off : 0);
    uint32_t *o = (uint32_t *)(void *)guard_alloc(&go, 4 * (len + 4), GUARD_END, 0);
    uint8_t *w = guard_alloc(&gw, wb64, GUARD_END, 0);
    if (t == NULL || o == NULL || w == NULL) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    memcpy(t, src, len);
    for (int pass = 0; pass < 4; pass++) {
        uint64_t age = pass >= 2 ? (uint64_t)TOKS_K5_WARM : 0u;
        if (pass == 2) { memset(c0, 0, sizeof c0); memset(c1, 0, sizeof c1); }
        toks_k5_args a0 = { src, len, ends, n, 0, want, len + 4, 0, c0, 15, w0, sizeof w0, age, 0, 0, 0x2A5A5Au, NULL };
        uint64_t n0 = toks_k5_encode_c(&T, &a0);
        toks_k5_args a;
        memset(&a, 0xA5, sizeof a);                     /* the counters accumulate: they start at age, 0, 0 */
        a.hits_static = age;
        a.hits_cache = a.misses = 0;
        a.text = t; a.len = len; a.ends = ends; a.n = n; a.start = 0; a.out = o; a.room = len + 4;
        a.cache = c1; a.cache_mask = 15; a.work = w; a.work_bytes = wb; a.cache_tag = a0.cache_tag; a.lcache = NULL;
        uint64_t rep = 0;
        uint64_t r = toks_abicheck_call((const void *)asm_k5, (uint64_t)(uintptr_t)&T, (uint64_t)(uintptr_t)&a, &rep);
        SC_N++;
        if (rep != 0 || r != n0 || a.n_out != a0.n_out || a.hits_static != a0.hits_static || a.hits_cache != a0.hits_cache
            || a.misses != a0.misses || memcmp(o, want, 4 * n0) != 0 || memcmp(c0, c1, sizeof c0) != 0) {
            sc_fail(pass == 0 ? "K5 cold" : pass == 1 ? "K5 warm" : pass == 2 ? "K5 cold (warm scratch)" : "K5 warm (warm scratch)",
                    len, where, off);
        }
    }
    guard_free(&gt);
    guard_free(&go);
    guard_free(&gw);
}

/* s2 = s1 with its first 40 bits changed by a nonzero vector the vocab hash cannot see (crc32c is linear in
 * the bytes for a fixed length): same length, same h, different bytes. 0 when none was found. */
static int sc_collide(const uint8_t *s1, uint32_t len, uint8_t *s2)
{
    uint8_t t[256];
    memcpy(t, s1, len);
    uint32_t h0 = bpe_vhash_h(t, len), col[40];
    uint64_t comb[40];
    for (uint32_t i = 0; i < 40; i++) {
        t[i >> 3] ^= (uint8_t)(1u << (i & 7));
        col[i] = bpe_vhash_h(t, len) ^ h0;
        t[i >> 3] ^= (uint8_t)(1u << (i & 7));
        comb[i] = (uint64_t)1 << i;
    }
    uint32_t r = 0;
    for (uint32_t b = 0; b < 32; b++) {                     /* gaussian elimination over gf(2) */
        uint32_t q = r;
        while (q < 40 && ((col[q] >> b) & 1u) == 0) { q++; }
        if (q == 40) { continue; }
        uint32_t tc = col[q]; col[q] = col[r]; col[r] = tc;
        uint64_t tm = comb[q]; comb[q] = comb[r]; comb[r] = tm;
        for (uint32_t j = 0; j < 40; j++) {
            if (j != r && ((col[j] >> b) & 1u) != 0) { col[j] ^= col[r]; comb[j] ^= comb[r]; }
        }
        r++;
    }
    if (r >= 40) { return 0; }
    memcpy(s2, s1, len);
    for (uint32_t i = 0; i < 40; i++) {
        if ((comb[r] >> i) & 1u) { s2[i >> 3] ^= (uint8_t)(1u << (i & 7)); }
    }
    return bpe_vhash_h(s2, len) == h0 && memcmp(s1, s2, len) != 0;
}

static int selfcheck(uint32_t n_vocab)
{
    static uint8_t seg[256];
    static uint32_t ends[256];
    uint64_t geo = 0, coll = 0, k6c = 0;
    for (uint64_t len = 0; len < 256; len++) {
        uint64_t i = (len * 7919u) % NP, total = 0, n = 0, maxp = 1;
        while (total < len) {                               /* real pieces; the last one cut to fit */
            uint64_t s = pstart(i), l = ENDS[i] - s;
            if (total + l > len) { l = len - total; }
            memcpy(seg + total, TEXT + s, l);
            total += l;
            ends[n++] = (uint32_t)total;
            maxp = l > maxp ? l : maxp;
            i = (i + 1) % NP;
        }
        sc_k5(toks_k5_encode_avx2, seg, len, ends, n, maxp, GUARD_END, 0);
        if (len > 0) { sc_k6(toks_k6_bpe_avx2, seg, len, GUARD_END, 0); k6c++; }
        for (size_t off = 0; off < 64; off++) {
            sc_k5(toks_k5_encode_avx2, seg, len, ends, n, maxp, GUARD_START, off);
            if (len > 0) { sc_k6(toks_k6_bpe_avx2, seg, len, GUARD_START, off); k6c++; }
        }
        geo++;
    }
    if ((T.flags & TOKS_TF_IGNORE_MERGES) != 0) {
        for (uint32_t id = 0; id < n_vocab && coll < 3000; id++) {
            uint32_t l = T.tok_off[id + 1] - T.tok_off[id];
            uint8_t s2[256];
            if (l < 5 || l > 255 || !sc_collide(T.tok_bytes + T.tok_off[id], l, s2)) { continue; }
            sc_k6(toks_k6_bpe_avx2, s2, l, GUARD_END, 0);
            coll++;
        }
    }
    printf("selfcheck: %" PRIu64 " lengths x (end flush + 64 start alignments): %" PRIu64 " K6 and %" PRIu64
           " K5 calls; %" PRIu64 " vhash collisions; %" PRIu64 " checks, %" PRIu64 " failures\n",
           geo, k6c, 2 * geo * 65, coll, SC_N, SC_BAD);
    return SC_BAD == 0 ? 0 : 1;
}

static uint64_t fnv(const uint8_t *p, uint64_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < n; i++) { h = (h ^ p[i]) * 0x100000001b3ull; }
    return h;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: bench_x86 PIECES.bin [REPS [lens|k6]] | PIECES.bin selfcheck\n"); return 2; }
    int self = argc > 2 && strcmp(argv[2], "selfcheck") == 0;
    int reps = argc > 2 && !self ? atoi(argv[2]) : 5;
    int lens = argc > 3 && strcmp(argv[3], "lens") == 0;
    int k6only = argc > 3 && strcmp(argv[3], "k6") == 0;          /* the K6 cells only (tuning runs) */
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    /* ---- the model (gen.py's header) -> tables ---- */
    char magic[8];
    uint32_t hdr[4];
    if (!rd(magic, 8, f) || memcmp(magic, "TOKSBPE1", 8) != 0 || !rd(hdr, sizeof hdr, f)) { fprintf(stderr, "bad stream\n"); return 2; }
    uint32_t n_vocab = hdr[0], n_ids = hdr[1], n_merges = hdr[2];
    uint32_t *tok_off = xalloc(4 * ((uint64_t)n_ids + 1));
    uint64_t cap = 1u << 20, o = 0;
    uint8_t *tok_bytes = malloc(cap);
    for (uint32_t id = 0; id < n_ids; id++) {
        uint32_t l;
        if (!rd(&l, 4, f)) { fprintf(stderr, "truncated model\n"); return 2; }
        while (o + l > cap) { cap *= 2; tok_bytes = realloc(tok_bytes, cap); }
        if (tok_bytes == NULL || !rd(tok_bytes + o, l, f)) { fprintf(stderr, "truncated model\n"); return 2; }
        tok_off[id] = (uint32_t)o;
        o += l;
    }
    tok_off[n_ids] = (uint32_t)o;
    uint32_t *trip = xalloc(12 * (uint64_t)n_merges + 4), *ml = xalloc(4 * (uint64_t)n_merges + 4);
    uint32_t *mr = xalloc(4 * (uint64_t)n_merges + 4), *mo = xalloc(4 * (uint64_t)n_merges + 4);
    if (!rd(trip, 12 * (size_t)n_merges, f)) { fprintf(stderr, "truncated merges\n"); return 2; }
    for (uint32_t i = 0; i < n_merges; i++) { ml[i] = trip[3 * i]; mr[i] = trip[3 * i + 1]; mo[i] = trip[3 * i + 2]; }
    toks_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_vocab = n_vocab;
    cfg.n_merges = n_merges;
    cfg.m_left_id = ml;
    cfg.m_right_id = mr;
    cfg.m_out_id = mo;
    cfg.ignore_merges = (uint8_t)hdr[3];
    memset(&T, 0, sizeof T);
    T.n_ids = n_ids;
    T.tok_off = tok_off;
    T.tok_bytes = tok_bytes;
    uint64_t need = toks_bpe_tables_bytes(&cfg);
    toks_arena ar = { xalloc(need), need, 0 };
    if (toks_bpe_build(&T, &ar, &cfg) != 0) { fprintf(stderr, "toks_bpe_build failed\n"); return 1; }

    /* ---- the pieces ---- */
    uint64_t tcap = 1u << 24, wcap = 1u << 22, ecap = 1u << 20;
    TEXT = malloc(tcap);
    WANT = malloc(4 * wcap);
    ENDS = malloc(4 * ecap);
    WOFF = malloc(8 * (ecap + 1));
    for (;;) {
        uint32_t len, n;
        if (!rd(&len, 4, f)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (len == UINT32_MAX) { break; }
        while (NB + len > tcap) { tcap *= 2; TEXT = realloc(TEXT, tcap); }
        if (TEXT == NULL || len == 0 || !rd(TEXT + NB, len, f) || !rd(&n, 4, f)) { fprintf(stderr, "bad record\n"); return 2; }
        while (NW + n > wcap) { wcap *= 2; WANT = realloc(WANT, 4 * wcap); }
        if (NP == ecap) { ecap *= 2; ENDS = realloc(ENDS, 4 * ecap); WOFF = realloc(WOFF, 8 * (ecap + 1)); }
        if (WANT == NULL || ENDS == NULL || WOFF == NULL || !rd(WANT + NW, 4 * (size_t)n, f)) { fprintf(stderr, "bad record\n"); return 2; }
        WOFF[NP] = NW;
        NB += len;
        NW += n;
        ENDS[NP++] = (uint32_t)NB;
        MAXLEN = len > MAXLEN ? len : MAXLEN;
    }
    WOFF[NP] = NW;
    fclose(f);
    if (NB > UINT32_MAX) { fprintf(stderr, "text over 4 GiB\n"); return 2; }
    OUT = xalloc(4 * (NW + 4));
    WORK = xalloc(TOKS_BPE_WORK_BYTES(MAXLEN));
    uint8_t *cache = xalloc(TOKS_CACHE_BUCKETS * 64u), *hot = xalloc(8192u * 64u);
    if (self) { return selfcheck(n_vocab); }
    int64_t by_words = k6_reference(cfg.ignore_merges != 0u);
    if (by_words < 0) { return 1; }

    /* ---- the K6 piece lists: all, first occurrences, first occurrences by length ---- */
    static const uint64_t lmax[NLEN] = { 1, 2, 4, 8, 16, 32, 128, UINT64_MAX };
    static const char *lname[NLEN] = { "K6 len 1", "K6 len 2", "K6 len 3-4", "K6 len 5-8", "K6 len 9-16",
                                       "K6 len 17-32", "K6 len 33-128", "K6 len >128" };
    subset sub[2 + NLEN];
    uint32_t nsub = 2 + (lens ? NLEN : 0);
    for (uint32_t c = 0; c < nsub; c++) {
        sub[c].name = c == 0 ? "K6 all" : c == 1 ? "K6 first" : lname[c - 2];
        sub[c].idx = xalloc(4 * NP + 4);
        sub[c].n = 0;
    }
    uint64_t hs = 1;
    while (hs < 2 * NP) { hs <<= 1; }
    uint32_t *seen = calloc(hs, 4);                          /* piece index + 1, 0 empty */
    if (seen == NULL) { fprintf(stderr, "out of memory\n"); return 2; }
    for (uint64_t i = 0; i < NP; i++) {
        sub[0].idx[sub[0].n++] = (uint32_t)i;
        uint64_t s = pstart(i), l = ENDS[i] - s, h = fnv(TEXT + s, l) & (hs - 1);
        int dup = 0;
        while (seen[h] != 0) {
            uint64_t j = seen[h] - 1, sj = pstart(j);
            if (ENDS[j] - sj == l && memcmp(TEXT + sj, TEXT + s, l) == 0) { dup = 1; break; }
            h = (h + 1) & (hs - 1);
        }
        if (dup) { continue; }
        seen[h] = (uint32_t)i + 1;
        sub[1].idx[sub[1].n++] = (uint32_t)i;
        for (uint32_t c = 0; lens && c < NLEN; c++) {
            if (l <= lmax[c]) { sub[2 + c].idx[sub[2 + c].n++] = (uint32_t)i; break; }
        }
    }
    for (uint32_t c = 0; c < nsub; c++) { subset_finish(&sub[c]); }

    printf("%s: n_vocab %u, merges %u, flags %#x; %" PRIu64 " pieces (%" PRIu64 " distinct), %" PRIu64 " bytes, %"
           PRIu64 " ids, longest %" PRIu64 "; K6 alone checked against hf, %" PRId64 " pieces by K5's words (the"
           " whole-piece rule)\n", argv[1], n_vocab, n_merges, T.flags, NP, sub[1].n, NB, NW, MAXLEN, by_words);
    loadavg("before");

    tier tiers[] = {
        { "c", toks_k6_bpe_c, toks_k5_encode_c },
        { "avx2", toks_k6_bpe_avx2, toks_k5_encode_avx2 },
#if defined(BENCH_AVX512)
        { "avx512", toks_k6_bpe_avx512, toks_k5_encode_avx512 },
#endif
#if defined(BENCH_EXTRA_K6)
        { "x", BENCH_EXTRA_K6, toks_k5_encode_avx2 },
#endif
#if defined(BENCH_EXTRA_K5)
        { "x", toks_k6_bpe_avx2, BENCH_EXTRA_K5 },
#endif
#if defined(BENCH_EXTRA_K5B)
        { "y", toks_k6_bpe_avx2, BENCH_EXTRA_K5B },
#endif
    };
    uint32_t nt = (uint32_t)(sizeof tiers / sizeof tiers[0]);
    int ok = 1;
    double best[6][MAXCELLS];
    uint64_t ctr[6][MAXCELLS][3];
    for (uint32_t i = 0; i < 6; i++) { for (uint32_t c = 0; c < MAXCELLS; c++) { best[i][c] = 1e30; } }
    uint32_t k5c = nsub;          /* cells k5c + 0..7: K5 tier cold / warm, alone cold / warm, hot, wo cold / warm / hot */
    const uint64_t W = TOKS_K5_WARM;

    for (int r = 0; r < reps; r++) {
        for (uint32_t j = 0; j < nt; j++) {
            uint32_t i = (r & 1) ? nt - 1 - j : j;                   /* ABBA */
            const tier *tt = &tiers[i];
            k5_fn alone = i == 0 ? toks_k5_encode_c_on_avx2 : tt->k5;
            double d;
            uint64_t c3[3];
            for (uint32_t c = 0; c < nsub; c++) {
                d = run_k6(tt->k6, &sub[c], tt->name, &ok);
                if (d < best[i][c]) { best[i][c] = d; }
            }
            if (k6only) { continue; }
            for (uint32_t c = 0; c < 2; c++) {                      /* tier, alone */
                k5_fn k5 = c == 0 ? tt->k5 : alone;
                memset(cache, 0, TOKS_CACHE_BUCKETS * 64u);
                d = run_k5(k5, cache, TOKS_CACHE_MASK, 0, tt->name, &ok, c3);       /* cold */
                if (d < best[i][k5c + 2 * c]) { best[i][k5c + 2 * c] = d; memcpy(ctr[i][k5c + 2 * c], c3, sizeof c3); }
                d = run_k5(k5, cache, TOKS_CACHE_MASK, 0, tt->name, &ok, c3);       /* warm */
                if (d < best[i][k5c + 2 * c + 1]) { best[i][k5c + 2 * c + 1] = d; memcpy(ctr[i][k5c + 2 * c + 1], c3, sizeof c3); }
            }
            memset(hot, 0, 8192u * 64u);
            (void)run_k5_hot(alone, hot, 8191u, 0, tt->name, &ok, c3);
            d = run_k5_hot(alone, hot, 8191u, 0, tt->name, &ok, c3);
            if (d < best[i][k5c + 4]) { best[i][k5c + 4] = d; memcpy(ctr[i][k5c + 4], c3, sizeof c3); }
            memset(cache, 0, TOKS_CACHE_BUCKETS * 64u);                 /* the warm order (a warm scratch) */
            d = run_k5(alone, cache, TOKS_CACHE_MASK, W, tt->name, &ok, c3);
            if (d < best[i][k5c + 5]) { best[i][k5c + 5] = d; memcpy(ctr[i][k5c + 5], c3, sizeof c3); }
            d = run_k5(alone, cache, TOKS_CACHE_MASK, W, tt->name, &ok, c3);
            if (d < best[i][k5c + 6]) { best[i][k5c + 6] = d; memcpy(ctr[i][k5c + 6], c3, sizeof c3); }
            memset(hot, 0, 8192u * 64u);
            (void)run_k5_hot(alone, hot, 8191u, W, tt->name, &ok, c3);
            d = run_k5_hot(alone, hot, 8191u, W, tt->name, &ok, c3);
            if (d < best[i][k5c + 7]) { best[i][k5c + 7] = d; memcpy(ctr[i][k5c + 7], c3, sizeof c3); }
        }
    }
    static const char *k5cells[NK5] = { "K5 tier cold", "K5 tier warm", "K5 alone cold", "K5 alone warm", "K5 hot",
                                        "K5 wo cold", "K5 wo warm", "K5 wo hot" };
    uint64_t hot_np = NP < HOT_CHUNKS * CHUNK ? NP : HOT_CHUNKS * CHUNK;
    printf("best of %d passes, ids checked against hf after every pass: %s\n", reps, ok ? "all equal" : "DIFFER");
    for (uint32_t c = 0; c < nsub; c++) {
        for (uint32_t i = 0; i < nt; i++) {
            row(sub[c].name, tiers[i].name, best[i][c], i == 0 ? 0.0 : best[0][c], sub[c].n, sub[c].bytes);
        }
    }
    for (uint32_t c = 0; c < NK5 && !k6only; c++) {
        int ishot = c == 4 || c == 7;
        uint64_t np = ishot ? HOT_REPS * hot_np : NP, nb = ishot ? HOT_REPS * (uint64_t)ENDS[hot_np - 1] : NB;
        for (uint32_t i = 0; i < nt; i++) { row(k5cells[c], tiers[i].name, best[i][k5c + c], i == 0 ? 0.0 : best[0][k5c + c], np, nb); }
        int same = 1;
        for (uint32_t i = 1; i < nt; i++) { same &= memcmp(ctr[0][k5c + c], ctr[i][k5c + c], sizeof ctr[0][0]) == 0; }
        printf("%-14s counters static %" PRIu64 " / cache %" PRIu64 " / miss %" PRIu64 "%s\n", "", ctr[0][k5c + c][0],
               ctr[0][k5c + c][1], ctr[0][k5c + c][2], same ? " (every tier)" : " (TIERS DIFFER)");
        ok &= same;
    }
    loadavg("after");
    printf("RESULT %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
