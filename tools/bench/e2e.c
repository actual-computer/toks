/* tools/bench/e2e.c: single-thread toks_encode over a corpus cut into chunks. Four cache states, an
 * exactness digest, the B_stream floor of the same footprint, the core clock, the caches' counters, and every
 * rep's seconds per state (for paired comparisons; e2e.sh's table takes the best of reps: a quick look, no
 * intervals).
 *
 *   e2e <tokenizer> <chunk_bytes|0> <reps> <file>...
 *
 * chunks: the files are concatenated; chunk_bytes > 0 cuts after the first '\n' at or past each
 * chunk_bytes boundary (never inside a utf-8 char); 0 encodes the whole text in one call. flags 0 (hf's
 * default call: added tokens recognized, post-processor applied). One output buffer, reused by every
 * call (the caller's array for that call). OTHER text: E2E_WARM_ON="<file> ..." (e2e.sh passes every bench
 * corpus file outside the measured corpus), cut at WARM_CHUNK (4096) bytes whatever the cell's chunk: a whole
 * cell's warm-up is then 4096-byte calls, not one ~18 MB call, so the scratch stays sized by the measured text
 * (en whole: ~72 MiB, not ~650 MiB).
 * states (kernels.md §6: the dynamic piece cache lives in the scratch; so does the segment memo with E2E_MEMO_MIB),
 * cpu-cache residency declared with each (SPEC §12.3: a dimension apart from the semantic caches):
 *   cold  the scratch is initialized before EVERY call, outside the timer, and each call is timed alone:
 *         the dynamic cache is empty at the start of every call (SPEC §12.3 cold, for m1a). The cold reps
 *         run first, back to back, after the exactness passes: every cold pass starts with this text's lines
 *         in the cpu caches (never after another state's sweep of the whole cache, which cost 6-12% cold and
 *         more with a bigger cache)
 *   pass  with OTHER text: init, one untimed pass over it, init again, then the timed pass: an empty piece
 *         cache that fills as the pass goes (one scratch per thread over a stream of documents), the cpu
 *         caches holding the other text's lines, not this text's (pass_after=other). Without OTHER text: init,
 *         the timed pass right after the previous state's pass over the same text (pass_after=same: the
 *         5f71528 logs' pass, called pass-same)
 *   warm  the pass right after the pass: an exact replay (not a SPEC state). Each tool replays from its own
 *         cache budget: toks' default is a 2 MiB piece cache and a 4 MiB segment memo (E2E_CACHE_MIB /
 *         E2E_MEMO_MIB change them),
 *         gigatoken's 512 MiB per worker plus, for sentencepiece, its unit memo: warm vs warm at the defaults is
 *         an UNMATCHED comparison (gigatoken-bench's GIGA_CACHE_MIB matches a budget)
 *   lang  with OTHER text only: init, one untimed pass over it, then the timed pass (the scratch keeps what the
 *         other text left). e2e.sh's other text is the other corpora (lang-x, tools/bench/tokv1.sh's rule), not
 *         same-language text: not SPEC §12.3's warm-lang bar until same-language corpora are pinned
 *   warmo with OTHER text only: init, an untimed pass over this text, an untimed
 *         pass over the OTHER text, then the timed pass over this text, no init between: the serving replay (a
 *         system prompt seen again after other requests), the 2nd sight of the text after a stream of other text.
 *         warm stays the back-to-back replay (SPEC T8's); warmo is reported beside it. Not SPEC §12.3's
 *         warm-lang (memo off, shortcut caches warmed on a disjoint corpus)
 *   coldo with OTHER text only: cold, each rep after an untimed
 *         pass over the OTHER text: the cold reps above run back to back, so their cpu caches hold this text's
 *         lines and the tables it touched (cpu-cache-hot); coldo's hold the other text's. The coldo reps run after
 *         the cold reps
 *   each rep after the cold and coldo reps runs pass, warm (and lang, warmo) in that order.
 * keys: <state>_s = the best rep, <state>_reps_s = every rep in order (comma-separated), pass_after = other | same.
 * CTR line (outside every timer): piece classes (toks_pieces: pieces, one-byte, over 15 bytes) and per state
 *   (cold summed over its calls; the last rep) K5's counters from the scratch header, hits_static / hits_cache /
 *   misses (kernels.md §6), lc = the long cache's arena bytes appended / generations started, memo = the segment
 *   memo's ring bytes written / hits; thp_kb = the process's AnonHugePages (linux).
 * E2E_CACHE_MIB=n: the scratch flags TOKS_SCRATCH_CACHE_MIB(n) (the piece caches' size; same ids); E2E_MEMO_MIB=m:
 * TOKS_SCRATCH_MEMO_MIB(m) too (the segment memo, SPEC §6: warm and the pass of repeated text replay; without it the
 * default 4 MiB, m = 0 none); the E2E line's memo_mib is the scratch's. E2E_HUGE=1: the
 * scratch from toks_plat_arena (2 MiB-aligned, huge pages where the platform gives them), else malloc.
 * exact: the id stream of all chunks in order (u32 little-endian) -> count + sha-256, once from fresh
 *   scratch and once warm (must agree: cache contents never change outputs, T2); e2e_ref.py computes hf's
 *   on the same chunks. E2E_IDS_OUT=<file> also writes the stream (e2e_ref.py --ids finds a first diff).
 * floor: SPEC §12.9 for this cell: per call, read the chunk's bytes and write its id count of u32s
 *   into the same output buffer; best of three variants (vector xor-read + plain stores, memcpy-read +
 *   memset, vector read + non-temporal stores) and of reps. x_floor = T_toks / T_floor.
 * ghz: a chain of dependent 1-cycle adds timed before and after (the clock the cell actually ran at).
 * -DE2E_STAGES (the e2e-stages binary only, with tools/bench/stages.c and the renamed library objects
 *   e2e.sh builds): per-stage timers and counters over pass-state passes (E2E_STATE=cold: cold passes, the
 *   scratch initialized before every call outside the timers), then a classifying pass in the same state.
 *   With -DTB_COUNT too (e2e-count): one pass with every library load / store / copy counted instead. */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"
#include "core.h"                                   /* toks_scr_header: K5's counters (kernels.md §6), the memo's */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#ifdef E2E_STAGES
void tb_reset(void);
void tb_snapshot(void);
void tb_classify_on(int on);
void tb_print(FILE *f, int live, double total_s, double ghz);
#endif
#ifdef TB_COUNT
void tb_count_regions(const void *ctx, const void *text, uint64_t n, const void *ids, uint64_t ids_bytes,
                      const void *scr, uint64_t sb, const void *stack);
void tb_count_on(int on);
void tb_count_print(FILE *f);
#endif

static uint64_t now_ns(void)
{
#if defined(__APPLE__)
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) { mach_timebase_info(&tb); }
    return mach_absolute_time() * tb.numer / tb.denom;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
}

/* the core clock: 8 dependent adds per iteration (1-cycle latency on every core we run on) */
static double ghz(void)
{
    uint64_t x = 0, iters = 25000000;
    uint64_t a = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
#if defined(__aarch64__)
        __asm__ volatile("add %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\t"
                         "add %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1" : "+r"(x));
#elif defined(__x86_64__)
        __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                         "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
#endif
    }
    uint64_t d = now_ns() - a;
    return (double)(8 * iters) / (double)d;
}

typedef struct cell {
    const toks_ctx *ctx;
    const uint8_t *buf;
    uint64_t n, nc, sb, idcap;
    uint64_t *beg, *len, *tc;   /* chunk start, length, ids */
    void *scr;
    uint32_t *ids;
} cell;

static uint32_t SFL;                               /* toks_scratch_* flags: E2E_CACHE_MIB */
#define WARM_CHUNK 4096u                           /* the OTHER text's chunk, whatever the cell's (header) */

/* the scratch's running counters (outside every timer): K5's hits_static, hits_cache, misses (kernels.md §6), the
 * long cache's arena fill and generation, the memo's ring bytes written and hits (kernels.md §7) */
enum { C_ST, C_CA, C_MI, C_LPOS, C_LGEN, C_MPOS, C_MHIT, C_N };
static uint64_t CTR[6][C_N];                       /* per state: cold, pass, warm, lang, warmo, coldo (cold, coldo: */
                                                   /* summed over calls) */
static void ctr_read(const void *scr, uint64_t v[C_N])
{
    toks_scratch *h = toks_scr_header((void *)(uintptr_t)scr);
    v[C_ST] = h->hits_static, v[C_CA] = h->hits_cache, v[C_MI] = h->misses;
    v[C_LPOS] = h->long_pos, v[C_LGEN] = h->long_gen;
    toks_scr_memo_ctr(h, &v[C_MPOS], &v[C_MHIT]);   /* core.h names the fields: a moved one is a compile error here */
}
#ifndef E2E_STAGES
static void ctr_delta(uint64_t d[C_N], const uint64_t a[C_N], const uint64_t b[C_N])   /* b - a; fills as of b */
{
    for (int i = 0; i < C_N; i++) { d[i] = b[i] - a[i]; }
    d[C_LPOS] = b[C_LGEN] == a[C_LGEN] ? b[C_LPOS] - a[C_LPOS] : b[C_LPOS];   /* a new generation restarts the arena */
    d[C_LGEN] = b[C_LGEN] - a[C_LGEN];
}
#endif

/* the files concatenated (+1 byte of slack), their length in *n */
static uint8_t *read_all(char *const *files, int nf, uint64_t *n_out)
{
    uint8_t *buf = NULL;
    uint64_t n = 0;
    for (int i = 0; i < nf; i++) {
        FILE *f = fopen(files[i], "rb");
        if (f == NULL) { perror(files[i]); exit(1); }
        fseek(f, 0, SEEK_END);
        long m = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf = realloc(buf, n + (uint64_t)m + 1);
        if (fread(buf + n, 1, (size_t)m, f) != (size_t)m) { fprintf(stderr, "short read %s\n", files[i]); exit(1); }
        fclose(f);
        n += (uint64_t)m;
    }
    *n_out = n;
    return buf;
}

/* c->buf[0, c->n) cut into chunks: after the first '\n' at or past each chunk boundary (0: one chunk); the longest */
static uint64_t cut(cell *c, uint64_t chunk)
{
    uint64_t nmax = chunk ? c->n / chunk + 2 : 2, maxlen = 0, p = 0;
    c->beg = malloc(nmax * sizeof(uint64_t));
    c->len = malloc(nmax * sizeof(uint64_t));
    c->tc = malloc(nmax * sizeof(uint64_t));
    while (p < c->n) {
        uint64_t e = c->n;
        if (chunk != 0 && p + chunk < c->n) {
            e = p + chunk;
            while (e < c->n && c->buf[e - 1] != '\n') { e++; }
        }
        c->beg[c->nc] = p;
        c->len[c->nc] = e - p;
        if (e - p > maxlen) { maxlen = e - p; }
        c->nc++;
        p = e;
    }
    return maxlen;
}

static int64_t enc(const cell *c, uint64_t k)
{
    int64_t m = toks_encode(c->ctx, c->buf + c->beg[k], c->len[k], 0, c->ids, c->idcap, c->scr);
    if (m < 0) { fprintf(stderr, "toks_encode: %lld at chunk %llu\n", (long long)m, (unsigned long long)k); exit(1); }
    return m;
}

static void init(const cell *c)
{
    if (toks_scratch_init(c->ctx, c->scr, c->sb, SFL) != 0) { fprintf(stderr, "toks_scratch_init\n"); exit(1); }
}

static uint64_t pass_run(const cell *c)           /* one timed pass, scratch as it is */
{
    uint64_t a = now_ns();
    for (uint64_t k = 0; k < c->nc; k++) { (void)enc(c, k); }
    return now_ns() - a;
}

static uint64_t pass_cold(const cell *c, uint64_t *ctr)   /* init before every call, outside the timer; ctr sums */
{                                                          /* the calls' counts */
    uint64_t t = 0, z[C_N], v[C_N];
    memset(ctr, 0, C_N * sizeof ctr[0]);
    for (uint64_t k = 0; k < c->nc; k++) {
        init(c);
        ctr_read(c->scr, z);
        uint64_t a = now_ns();
        (void)enc(c, k);
        t += now_ns() - a;
        ctr_read(c->scr, v);                        /* this call's counts (outside the timer) */
        for (int i = 0; i < C_N; i++) { ctr[i] += i == C_LGEN ? 0u : v[i] - z[i]; }
    }
    return t;
}

#ifndef E2E_STAGES

/* ---- floor variants: read len[k] bytes, write tc[k] u32 into ids (per call) ---------------------- */
static volatile uint64_t sink;

static void floor_xor(const cell *c)
{
    uint64_t acc = 0;
    for (uint64_t k = 0; k < c->nc; k++) {
        const uint8_t *p = c->buf + c->beg[k];
        uint64_t n = c->len[k], x = 0, i = 0;
        for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); x ^= w; }
        for (; i < n; i++) { x ^= p[i]; }
        uint32_t v = (uint32_t)x, *o = c->ids;
        for (uint64_t j = 0; j < c->tc[k]; j++) { o[j] = v + (uint32_t)j; }
        acc += x;
    }
    sink = acc;
}

static void floor_memcpy(const cell *c)
{
    static uint8_t l1[4096];
    uint64_t acc = 0;
    for (uint64_t k = 0; k < c->nc; k++) {
        const uint8_t *p = c->buf + c->beg[k];
        for (uint64_t o = 0; o < c->len[k]; o += sizeof l1) {
            uint64_t m = c->len[k] - o < sizeof l1 ? c->len[k] - o : sizeof l1;
            memcpy(l1, p + o, m);
        }
        memset(c->ids, l1[0] | 1, c->tc[k] * 4);
        acc += l1[0];
    }
    sink = acc;
}

static void floor_nt(const cell *c)
{
    uint64_t acc = 0;
    for (uint64_t k = 0; k < c->nc; k++) {
        const uint8_t *p = c->buf + c->beg[k];
        uint64_t n = c->len[k], x = 0, i = 0;
        for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); x ^= w; }
        for (; i < n; i++) { x ^= p[i]; }
        uint32_t v = (uint32_t)x, *o = c->ids;
        for (uint64_t j = 0; j < c->tc[k]; j++) { __builtin_nontemporal_store(v + (uint32_t)j, o + j); }
        acc += x;
    }
    sink = acc;
}

static uint64_t best_of(void (*f)(const cell *), const cell *c, int reps)
{
    uint64_t b = UINT64_MAX;
    for (int r = 0; r < reps; r++) {
        uint64_t a = now_ns();
        f(c);
        uint64_t d = now_ns() - a;
        if (d < b) { b = d; }
    }
    return b;
}
#endif

/* the id stream of every chunk in order, from the scratch as it is: count + sha-256 (+ file) */
static uint64_t digest(const cell *c, uint8_t sha[32], const char *path)
{
    uint64_t cap = c->n + 64 * c->nc + 64, tot = 0;
    uint32_t *all = malloc(cap * 4);
    if (all == NULL) { fprintf(stderr, "malloc\n"); exit(1); }
    for (uint64_t k = 0; k < c->nc; k++) {
        int64_t m = toks_encode(c->ctx, c->buf + c->beg[k], c->len[k], 0, all + tot, cap - tot, c->scr);
        if (m < 0 || (uint64_t)m > cap - tot) { fprintf(stderr, "digest: encode %lld\n", (long long)m); exit(1); }
        c->tc[k] = (uint64_t)m;
        tot += (uint64_t)m;
    }
    toks_sha256((const uint8_t *)all, tot * 4, sha);       /* little-endian hosts only (arm64, x86-64) */
    if (path != NULL) {
        FILE *f = fopen(path, "wb");
        if (f == NULL || fwrite(all, 4, tot, f) != tot) { perror(path); exit(1); }
        fclose(f);
    }
    free(all);
    return tot;
}

static const char *tier_name(uint32_t t)
{
    switch (t) {
    case TOKS_TIER_SCALAR: return "scalar";
    case TOKS_TIER_NEON:   return "neon";
    case TOKS_TIER_AVX2:   return "avx2";
    case TOKS_TIER_AVX512: return "avx512";
    default:               return "?";
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: e2e <tokenizer> <chunk_bytes|0> <reps> <file>...\n");
        return 2;
    }
    uint64_t chunk = strtoull(argv[2], NULL, 10);
    int reps = atoi(argv[3]);
    SFL = getenv("E2E_CACHE_MIB") ? TOKS_SCRATCH_CACHE_MIB(strtoul(getenv("E2E_CACHE_MIB"), NULL, 10)) : 0u;
    SFL |= getenv("E2E_MEMO_MIB") ? TOKS_SCRATCH_MEMO_MIB(strtoul(getenv("E2E_MEMO_MIB"), NULL, 10)) : 0u;
    cell c, w;                                         /* w: the E2E_WARM_ON text (lang), w.nc == 0 without it */
    memset(&c, 0, sizeof c);
    memset(&w, 0, sizeof w);
    uint64_t n = 0;
    uint8_t *buf = read_all(argv + 4, argc - 4, &n);
    c.buf = buf;
    c.n = n;
    uint64_t maxlen = cut(&c, chunk);
    if (getenv("E2E_WARM_ON") != NULL && getenv("E2E_WARM_ON")[0] != 0) {
        char *list = strdup(getenv("E2E_WARM_ON")), *files[64];
        int nf = 0;
        for (char *f = strtok(list, " "); f != NULL && nf < 64; f = strtok(NULL, " ")) { files[nf++] = f; }
        w.buf = read_all(files, nf, &w.n);
        uint64_t wl = cut(&w, WARM_CHUNK);
        if (wl > maxlen) { maxlen = wl; }
    }

    toks_ctx *ctx = NULL;
    uint64_t t0 = now_ns();
    int64_t r = toks_load(&ctx, argv[1], NULL);
    double tload = (double)(now_ns() - t0) * 1e-9;
    if (r != 0) { fprintf(stderr, "toks_load: %lld\n", (long long)r); return 1; }
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    (void)toks_get_info(ctx, &info);
    c.ctx = ctx;
    c.sb = toks_scratch_bytes(ctx, maxlen, SFL);
    c.scr = getenv("E2E_HUGE") != NULL && getenv("E2E_HUGE")[0] == '1' ? toks_plat_arena(c.sb) : malloc(c.sb);
    c.idcap = maxlen + 64;
    c.ids = malloc(c.idcap * sizeof(uint32_t));
    w.ctx = ctx;
    w.sb = c.sb;
    w.scr = c.scr;
    w.idcap = c.idcap;
    w.ids = c.ids;

    /* exactness first (fills tc for the floor), cold and warm */
    uint8_t sha[32], sha_w[32];
    init(&c);
    uint64_t memo_b;                                    /* the scratch's memo: SFL's, or the default */
    (void)toks_scr_memo(toks_scr_header(c.scr), &memo_b);
    uint64_t tot = digest(&c, sha, getenv("E2E_IDS_OUT"));
    uint64_t tot_w = digest(&c, sha_w, NULL);              /* the cache is full of this text now */
    if (tot_w != tot || memcmp(sha, sha_w, 32) != 0) { fprintf(stderr, "e2e: warm ids differ from cold ids\n"); return 1; }
    char hex[65];
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", sha[i]); }

    double g0 = ghz();
#ifndef E2E_STAGES
    /* piece classes (toks_pieces, untimed): K5's one-byte pieces are hits_static without a probe; pieces over 15
     * bytes never reach the short cache */
    uint64_t np = 0, n1 = 0, nl = 0;
    for (uint64_t k = 0; k < c.nc; k++) {
        int64_t m = toks_pieces(ctx, c.buf + c.beg[k], c.len[k], 0, c.ids, c.idcap, c.scr);
        if (m < 0 || (uint64_t)m > c.idcap) { fprintf(stderr, "toks_pieces: %lld\n", (long long)m); return 1; }
        uint64_t s = 0;
        for (int64_t i = 0; i < m; i++) {
            uint64_t l = c.ids[i] - s;
            n1 += l == 1u, nl += l > 15u;
            s = c.ids[i];
        }
        np += (uint64_t)m;
    }
    enum { COLD, PASS, WARM, LANG, WARMO, COLDO, NST };
    static const char *const ST[NST] = { "cold", "pass", "warm", "lang", "warmo", "coldo" };
    int other = w.nc != 0, nst = other ? NST : LANG;   /* pass after other text, lang, warmo, coldo: E2E_WARM_ON */
    uint64_t *tm = calloc((size_t)NST * (size_t)(reps > 0 ? reps : 1), sizeof(uint64_t)), best[NST];
    if (tm == NULL) { fprintf(stderr, "malloc\n"); return 1; }
#define TM(st, rep) tm[(size_t)(st) * (size_t)reps + (size_t)(rep)]
    for (int rep = 0; rep < reps; rep++) {             /* cold reps back to back: a cold pass never follows a pass */
        TM(COLD, rep) = pass_cold(&c, CTR[COLD]);       /* that swept the whole cache (another state's) */
    }
    for (int rep = 0; other && rep < reps; rep++) {    /* coldo: each cold rep after an untimed pass over the */
        init(&c);                                       /* OTHER text (the cpu caches hold its lines, not this */
        for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }   /* text's: a new request after others) */
        TM(COLDO, rep) = pass_cold(&c, CTR[COLDO]);
    }
    for (int rep = 0; rep < reps; rep++) {
        uint64_t v0[C_N], v1[C_N], v2[C_N];
        if (other) {                                    /* pass: the cpu caches hold the other text's lines */
            init(&c);
            for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }
        }
        init(&c);
        ctr_read(c.scr, v0);
        TM(PASS, rep) = pass_run(&c);
        ctr_read(c.scr, v1);
        ctr_delta(CTR[1], v0, v1);
        TM(WARM, rep) = pass_run(&c);
        ctr_read(c.scr, v2);
        ctr_delta(CTR[2], v1, v2);
        if (other) {                                    /* lang: warmed on the other text */
            init(&c);
            for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }
            ctr_read(c.scr, v0);
            TM(LANG, rep) = pass_run(&c);
            ctr_read(c.scr, v1);
            ctr_delta(CTR[3], v0, v1);
            init(&c);                                   /* warmo: this text, then the other text, then this text */
            (void)pass_run(&c);                         /* again on the same scratch: the 2nd sight after other */
            for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }   /* requests (a system prompt seen again) */
            ctr_read(c.scr, v0);
            TM(WARMO, rep) = pass_run(&c);
            ctr_read(c.scr, v1);
            ctr_delta(CTR[4], v0, v1);
        }
    }
    for (int st = 0; st < NST; st++) {
        best[st] = st < nst ? UINT64_MAX : 0;
        for (int rep = 0; st < nst && rep < reps; rep++) { if (TM(st, rep) < best[st]) { best[st] = TM(st, rep); } }
    }
    uint64_t fx = best_of(floor_xor, &c, reps), fm = best_of(floor_memcpy, &c, reps), fn = best_of(floor_nt, &c, reps);
    uint64_t fl = fx;
    const char *fv = "xor";
    if (fm < fl) { fl = fm; fv = "memcpy"; }
    if (fn < fl) { fl = fn; fv = "nt"; }
    double g1 = ghz();
    double sc = (double)best[COLD] * 1e-9, sp = (double)best[PASS] * 1e-9, sw = (double)best[WARM] * 1e-9;
    double sf = (double)fl * 1e-9;
    printf("toks  chunk %7llu  cold %7.1f MB/s %6.2f Mtok/s %9.0f ns/call | pass %7.1f MB/s | warm %7.1f MB/s"
           " | floor %8.1f MB/s (%s) x_floor %6.0f | %s\n",
           (unsigned long long)chunk, (double)n / sc / 1e6, (double)tot / sc / 1e6, sc / (double)c.nc * 1e9,
           (double)n / sp / 1e6, (double)n / sw / 1e6, (double)n / sf / 1e6, fv, sp / sf, tier_name(info.tier));
    printf("E2E tool=toks tier=%s chunk=%llu bytes=%llu calls=%llu ids=%llu sha=%.16s cold_s=%.6f pass_s=%.6f"
           " warm_s=%.6f lang_s=%.6f warmo_s=%.6f coldo_s=%.6f floor_s=%.6f floor_v=%s floor_xor_s=%.6f floor_memcpy_s=%.6f"
           " floor_nt_s=%.6f"
           " ghz0=%.3f ghz1=%.3f load_ms=%.0f scratch_mib=%.2f cache_mib=%u memo_mib=%u reps=%d pass_after=%s",
           tier_name(info.tier), (unsigned long long)chunk, (unsigned long long)n, (unsigned long long)c.nc,
           (unsigned long long)tot, hex, sc, sp, sw, (double)best[LANG] * 1e-9, (double)best[WARMO] * 1e-9,
           (double)best[COLDO] * 1e-9, sf, fv,
           (double)fx * 1e-9,
           (double)fm * 1e-9, (double)fn * 1e-9, g0, g1, tload * 1e3, (double)c.sb / 1048576.0,
           (unsigned)((SFL >> 12) & 0xFFu), (unsigned)(memo_b >> 20), reps, other ? "other" : "same");
    for (int st = 0; st < nst; st++) {
        printf(" %s_reps_s=", ST[st]);
        for (int rep = 0; rep < reps; rep++) { printf("%s%.6f", rep ? "," : "", (double)TM(st, rep) * 1e-9); }
    }
    printf("\n");
#undef TM
    free(tm);
    printf("CTR pieces=%llu one=%llu over15=%llu", (unsigned long long)np, (unsigned long long)n1, (unsigned long long)nl);
    {
        FILE *f = fopen("/proc/self/smaps_rollup", "r");   /* linux: the process's huge-paged anonymous memory */
        char ln[256];
        while (f != NULL && fgets(ln, sizeof ln, f) != NULL) {
            if (strncmp(ln, "AnonHugePages:", 14) == 0) { printf(" thp_kb=%lld", atoll(ln + 14)); }
        }
        if (f != NULL) { fclose(f); }
    }
    for (int st = 0; st < nst; st++) {
        const uint64_t *v = CTR[st];
        printf(" %s=%llu/%llu/%llu lc=%llu/%llu memo=%llu/%llu", ST[st], (unsigned long long)v[C_ST],
               (unsigned long long)v[C_CA], (unsigned long long)v[C_MI], (unsigned long long)v[C_LPOS],
               (unsigned long long)v[C_LGEN], (unsigned long long)v[C_MPOS], (unsigned long long)v[C_MHIT]);
    }
    printf("\n");
#else
#ifdef TB_COUNT
    /* the counting build: one pass with every library load / store / libc copy counted (E2E_STATE=cold: the
     * scratch initialized before every call, uncounted) */
    int cold = getenv("E2E_STATE") != NULL && strcmp(getenv("E2E_STATE"), "cold") == 0;
    init(&c);
    tb_count_regions(ctx, c.buf, c.n, c.ids, c.idcap * 4, c.scr, c.sb, &c);
    tb_count_on(1);
    tb_count_on(0);
    for (uint64_t k = 0; k < c.nc; k++) {
        if (cold) { init(&c); }
        tb_count_on(2);
        (void)enc(&c, k);
        tb_count_on(0);
    }
    printf("COUNT state=%s tier=%s chunk=%llu bytes=%llu calls=%llu ids=%llu sha=%.16s", cold ? "cold" : "pass",
           tier_name(info.tier),
           (unsigned long long)chunk, (unsigned long long)n, (unsigned long long)c.nc, (unsigned long long)tot, hex);
    tb_count_print(stdout);
    (void)g0;
    (void)reps;
    (void)tload;
    (void)pass_run;
    (void)pass_cold;
#else
    /* stage breakdown of one state: reps passes, the fastest one's counters; then one classifying pass (untimed
     * classification of every K6 call and every piece). With OTHER text (E2E_WARM_ON), every
     * timed pass and the classifying pass follow an untimed pass over it on the same scratch and a re-init, as
     * e2e's pass does (pass_after=other; E2E_STATE=cold is then e2e's coldo); without it, each follows the previous
     * pass over the same text (pass_after=same: cpu-cache hot). The OTHER pass runs before tb_reset: untimed */
    int cold = getenv("E2E_STATE") != NULL && strcmp(getenv("E2E_STATE"), "cold") == 0;
    const char *stn = cold ? (w.nc ? "coldo" : "cold") : "pass", *after = w.nc ? "other" : "same";
    uint64_t best = UINT64_MAX;
    for (int rep = 0; rep < reps; rep++) {
        if (w.nc) {
            init(&c);
            for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }
        }
        init(&c);
        tb_reset();
        uint64_t d = cold ? pass_cold(&c, CTR[0]) : pass_run(&c);
        if (d < best) { best = d; tb_snapshot(); }        /* the fastest rep's timers and counters */
    }
    double g1 = ghz();
    printf("STAGES state=%s pass_after=%s tier=%s chunk=%llu bytes=%llu calls=%llu ids=%llu sha=%.16s ghz0=%.3f ghz1=%.3f"
           " load_ms=%.0f", stn, after, tier_name(info.tier), (unsigned long long)chunk, (unsigned long long)n,
           (unsigned long long)c.nc, (unsigned long long)tot, hex, g0, g1, tload * 1e3);
    tb_print(stdout, 0, (double)best * 1e-9, g0);
    if (w.nc) {
        init(&c);
        for (uint64_t k = 0; k < w.nc; k++) { (void)enc(&w, k); }
    }
    init(&c);
    tb_reset();
    tb_classify_on(1);
    (void)(cold ? pass_cold(&c, CTR[0]) : pass_run(&c));
    tb_classify_on(0);
    printf("CLASSIFY state=%s pass_after=%s tier=%s chunk=%llu", stn, after, tier_name(info.tier), (unsigned long long)chunk);
    tb_print(stdout, 1, -1.0, g0);
#endif
#endif
    toks_unload(ctx);
    return 0;
}
