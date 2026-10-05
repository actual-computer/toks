/* tools/bench/stages.c: per-stage timers and counters for the e2e-stages bench binary. Bench-only: the
 * library sources are compiled a second time for that binary (tools/bench/e2e.sh) with the kernel
 * symbols renamed at their call sites -- api.c and segment.c call tb_k1_* / tb_k3_* / tb_k3o_* / tb_k5_*,
 * k5_long.c calls tb_k6_* -- and the wrappers here time the real kernels. The shipping library and its sources
 * carry no instrumentation.
 *
 * Timers: rdtscp (x86-64; tsc ticks calibrated against CLOCK_MONOTONIC) or isb + cntvct_el0 (arm64;
 * cntfrq_el0). One timer pair per kernel call; its cost (measured: ovh) is subtracted per call. The
 * driver's share is the rest: total - K1 - K3 - K5 (K5 includes K6). K6 is timed in every tier (k5_long.c's
 * K6 calls are renamed; every K5 tier calls K6 through it). K5's own time (minus K6) is split over its piece
 * classes by a least-squares fit over the rounds (x = 1-byte, static 2..15 B, cache hits, short misses, long
 * pieces, 1 per round; y = the round's K5 ticks minus its K6 ticks): coefficients in ns per piece.
 * Classifying pass (untimed): every piece of every K3 round (distinct pieces, length histogram) and
 * every K6 call (why the shortcut layer missed it).
 * -DTB_COUNT (the e2e-count binary): the library's c is compiled with -fsanitize-coverage=func,trace-loads,
 * trace-stores and memcpy / memset / memcmp renamed to tb_cmem*: every load, store and libc copy of the
 * library is counted here, by stage (driver, K1, K3, K5 own, K6) and by memory region (text, ids out, the
 * scratch's dynamic cache / ends / bpe work / bounce, each bpe table: words, vhash, mt (merge buckets), pf (pair
 * filter), bytepair, premerge, apm, rank2id, byte2id; the other tables, the stack, other), in bytes, in cache-line
 * changes (an access to another 64-byte line than the stage's previous access to that region) and in distinct
 * lines (every line the stage touched in the counted calls once: what a pass after other text may have to fetch;
 * not for the other tables, the stack, other). Fields: <stage>_<region>=read/written/changes/distinct.
 * Asm kernels are not instrumented: their traffic is the counted traffic of the c twin they replace only
 * where the algorithm is the same. */
#define _POSIX_C_SOURCE 200809L
#include "bpe.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

static inline uint64_t tick(void)
{
#if defined(__x86_64__)
    unsigned aux;
    return __rdtscp(&aux);
#else
    uint64_t v;
    __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
#endif
}

#if defined(__x86_64__)
static uint64_t mono_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
#endif

static double tick_hz(void)
{
    static double hz;
    if (hz == 0.0) {
#if defined(__aarch64__)
        uint64_t f;
        __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
        hz = (double)f;
#else
        uint64_t a = mono_ns(), c0 = tick(), b;
        do { b = mono_ns(); } while (b - a < 50000000ull);
        hz = (double)(tick() - c0) / ((double)(b - a) * 1e-9);
#endif
    }
    return hz;
}

static double ovh_ticks(void)                     /* cost of one timer pair, in ticks */
{
    static double o = -1.0;
    if (o < 0.0) {
        uint64_t s = 0;
        for (int i = 0; i < 1000000; i++) { uint64_t a = tick(); s += tick() - a; }
        o = (double)s / 1e6;
    }
    return o;
}

/* ---- counters ----------------------------------------------------------------------------------- */
#define NH 10   /* length buckets: 1, 2, 3, 4-5, 6-7, 8-11, 12-15, 16-31, 32-63, 64+ */
static unsigned hb(uint64_t l)
{
    return l <= 3 ? (unsigned)(l - 1) : l <= 5 ? 3u : l <= 7 ? 4u : l <= 11 ? 5u : l <= 15 ? 6u : l <= 31 ? 7u : l <= 63 ? 8u : 9u;
}

typedef struct tb_t {
    uint64_t k1_calls, k1_ticks, k1_bytes;
    uint64_t k3_calls, k3_ticks, k3_bytes, k3_pieces;
    uint64_t k5_calls, k5_ticks, k5_pieces, k5_ids, hits_static, hits_cache, misses;
    uint64_t k6_calls, k6_ticks, k6_bytes, k6_ids, k6_merges, k6_hist[NH];
    uint64_t k6s_calls, k6s_ticks, k3o_calls;               /* K6 on pieces <= 15 B; o200k K3 rounds */
    uint64_t c_one, c_static, c_cache, c_mshort, c_long;    /* K5's pieces by class */
    double xtx[6][6], xty[6], yy;                           /* the K5 fit's normal equations */
    /* classifying pass */
    uint64_t pieces, piece_hist[NH], distinct, distinct_long, distinct_vocab, distinct_multi;
    uint64_t m_long, m_vocab_unplaced, m_vocab_gt4, m_multi_first, m_multi_again, m_multi_gt4;
} tb_t;

static tb_t live, snap;
static int classify;
static const toks_tables *tt;

void tb_reset(void) { memset(&live, 0, sizeof live); }
void tb_snapshot(void) { snap = live; }

/* a set of 64-bit piece hashes (open addressing); collisions over ~1e6 keys: ~1e-8, ignored */
typedef struct hset { uint64_t *s; uint64_t mask, n; } hset;
static hset seen_piece, seen_k6;

static uint64_t hash_bytes(const uint8_t *p, uint64_t n)
{
    uint64_t h = 0xcbf29ce484222325ull ^ (n * 0x9e3779b97f4a7c15ull);
    for (uint64_t i = 0; i < n; i++) { h = (h ^ p[i]) * 0x100000001b3ull; }
    h ^= h >> 29;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 32;
    return h | 1u;                                 /* 0 = empty */
}

static int hset_add(hset *h, uint64_t k)           /* 1 when new */
{
    if (h->s == NULL) {
        h->mask = (1ull << 22) - 1;
        h->s = calloc(h->mask + 1, 8);
        if (h->s == NULL) { fprintf(stderr, "stages: calloc\n"); exit(1); }
    }
    if (h->n * 2 > h->mask) { fprintf(stderr, "stages: piece set full\n"); exit(1); }
    for (uint64_t i = k & h->mask;; i = (i + 1) & h->mask) {
        if (h->s[i] == k) { return 0; }
        if (h->s[i] == 0) { h->s[i] = k; h->n++; return 1; }
    }
}

void tb_classify_on(int on)
{
    classify = on;
    if (on) {
        if (seen_piece.s) { memset(seen_piece.s, 0, (seen_piece.mask + 1) * 8); seen_piece.n = 0; }
        if (seen_k6.s) { memset(seen_k6.s, 0, (seen_k6.mask + 1) * 8); seen_k6.n = 0; }
    }
}

static int is_vocab(const uint8_t *p, uint64_t n)
{
    uint32_t id;
    return bpe_vhash_find(tt, p, n, &id);
}

static void classify_round(const toks_k3_args *a, uint64_t p0, uint64_t n)
{
    uint64_t s = p0;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t e = a->ends[i], l = e - s;
        const uint8_t *p = a->text + s;
        live.pieces++;
        live.piece_hist[hb(l)]++;
        if (hset_add(&seen_piece, hash_bytes(p, l))) {
            live.distinct++;
            if (l > TOKS_KEY_MAXLEN) { live.distinct_long++; }
            else if (l >= 2 && is_vocab(p, l)) { live.distinct_vocab++; }
            else if (l >= 2) { live.distinct_multi++; }
        }
        s = e;
    }
}

static FILE *miss_log;                            /* TB_MISSES=<file>: every K6 call of the classifying pass */

static void classify_k6(const uint8_t *p, uint64_t l, uint64_t n_out, const uint32_t *ids)
{
    if (miss_log == NULL && getenv("TB_MISSES") != NULL) { miss_log = fopen(getenv("TB_MISSES"), "w"); }
    if (miss_log != NULL) {                       /* len n bytes(hex) ids */
        fprintf(miss_log, "%llu %llu ", (unsigned long long)l, (unsigned long long)n_out);
        for (uint64_t i = 0; i < l; i++) { fprintf(miss_log, "%02x", p[i]); }
        for (uint64_t i = 0; i < n_out; i++) { uint32_t v; memcpy(&v, ids + i, 4); fprintf(miss_log, " %u", v); }
        fputc('\n', miss_log);
    }
    if (l > TOKS_KEY_MAXLEN) { live.m_long++; return; }
    if (is_vocab(p, l)) {
        if (n_out > 4) { live.m_vocab_gt4++; } else { live.m_vocab_unplaced++; }
        return;
    }
    if (n_out > 4) { live.m_multi_gt4++; return; }
    if (hset_add(&seen_k6, hash_bytes(p, l))) { live.m_multi_first++; } else { live.m_multi_again++; }
}

#ifdef TB_COUNT
#include "core.h"
enum { S_DRV, S_K1, S_K3, S_K5, S_K6, NS };
/* the bpe tables one region each (the words table, vhash, the merge table's buckets and pair filter, bytepair,
 * premerge, apm, rank2id, byte2id); "tables" is every other table byte (the compiled tables, the ctx) */
enum { R_TEXT, R_IDS, R_CACHE, R_ENDS, R_WORK, R_BOUNCE, R_WORDS, R_VHASH, R_MT, R_PF, R_BP, R_PM, R_APM, R_R2I, R_B2I,
       R_TABLES, R_STACK, R_OTHER, NR };
static const char *const SN[NS] = { "drv", "k1", "k3", "k5", "k6" };
static const char *const RN[NR] = { "text", "ids", "cache", "ends", "work", "bounce", "words", "vhash", "mt", "pf",
                                    "bytepair", "premerge", "apm", "rank2id", "byte2id", "tables", "stack", "other" };
static int cur = S_DRV, counting;
static uint64_t cb[NS][NR][2], cl[NS][NR], cpy[NS], dl[NS][NR];
static uintptr_t lastl[NS][NR];
/* address ranges, first match wins (the bpe tables before the arenas that hold them); one range per region except
 * "tables" (three: the compiled tables, the bpe arena's rest, the ctx); dist: a bit per line per stage (one-range
 * regions only), so dl counts the distinct lines a stage touches in the counted calls */
static struct { uintptr_t lo, hi; int r; uint8_t *dist[NS]; } rg[NR + 2];
static int nrg;

static int cls(uintptr_t a, int *g)
{
    for (int i = 0; i < nrg; i++) { if (a >= rg[i].lo && a < rg[i].hi) { *g = i; return rg[i].r; } }
    *g = -1;
    return R_OTHER;
}
static void acct(uintptr_t a, uint64_t n, int w)
{
    if (!counting) { return; }
    int g, r = cls(a, &g);
    cb[cur][r][w] += n;
    uint8_t *d = g >= 0 ? rg[g].dist[cur] : NULL;
    for (uintptr_t l = a >> 6, e = (a + n - 1) >> 6; l <= e; l++) {
        if (l != lastl[cur][r]) { cl[cur][r]++; lastl[cur][r] = l; }
        if (d != NULL) {
            uintptr_t i = l - (rg[g].lo >> 6);
            if (!(d[i >> 3] & (1u << (i & 7u)))) { d[i >> 3] |= (uint8_t)(1u << (i & 7u)); dl[cur][r]++; }
        }
    }
}
#define LD(n, T) void __sanitizer_cov_load##n(T *a); void __sanitizer_cov_load##n(T *a) { acct((uintptr_t)a, n, 0); }
#define ST(n, T) void __sanitizer_cov_store##n(T *a); void __sanitizer_cov_store##n(T *a) { acct((uintptr_t)a, n, 1); }
LD(1, uint8_t) LD(2, uint16_t) LD(4, uint32_t) LD(8, uint64_t) LD(16, __int128)
ST(1, uint8_t) ST(2, uint16_t) ST(4, uint32_t) ST(8, uint64_t) ST(16, __int128)

void *tb_cmemcpy(void *d, const void *s, size_t n);
void *tb_cmemcpy(void *d, const void *s, size_t n)
{
    if (n) { acct((uintptr_t)s, n, 0); acct((uintptr_t)d, n, 1); if (counting) { cpy[cur] += n; } }
    return memcpy(d, s, n);
}
void *tb_cmemset(void *d, int c, size_t n);
void *tb_cmemset(void *d, int c, size_t n)
{
    if (n) { acct((uintptr_t)d, n, 1); }
    return memset(d, c, n);
}
int tb_cmemcmp(const void *a, const void *b, size_t n);
int tb_cmemcmp(const void *a, const void *b, size_t n)
{
    if (n) { acct((uintptr_t)a, n, 0); acct((uintptr_t)b, n, 0); }
    return memcmp(a, b, n);
}

void tb_count_regions(const void *ctxv, const void *text, uint64_t n, const void *ids, uint64_t ids_bytes,
                      const void *scr, uint64_t sb, const void *stack);
void tb_count_regions(const void *ctxv, const void *text, uint64_t n, const void *ids, uint64_t ids_bytes,
                      const void *scr, uint64_t sb, const void *stack)
{
    const struct toks_ctx *ctx = ctxv;
    const toks_tables *t = &ctx->t;
    uintptr_t base = (uintptr_t)scr;
    const toks_scratch *h = toks_scr_header((void *)(uintptr_t)scr);   /* header, ends, caches, work, bounce */
    nrg = 0;
#define RG(region, a, b) do { rg[nrg].lo = (uintptr_t)(a); rg[nrg].hi = (uintptr_t)(b); rg[nrg].r = (region); nrg++; } while (0)
    RG(R_TEXT, text, (uintptr_t)text + n);
    RG(R_IDS, ids, (uintptr_t)ids + ids_bytes);
    RG(R_ENDS, (uintptr_t)h + TOKS_SCR_HDR, base + h->off_cache);
    RG(R_CACHE, base + h->off_cache, base + h->off_work);
    RG(R_WORK, base + h->off_work, base + h->off_bounce);
    RG(R_BOUNCE, base + h->off_bounce, base + sb);
    /* each bpe table up to its size, or up to the next table when the size is not kept (premerge, apm) */
    struct { int r; uintptr_t a; uint64_t len; } bt[] = {
        { R_WORDS, (uintptr_t)t->words, t->words != NULL ? (t->words_mask + 1u) * TOKS_BUCKET : 0u },
        { R_VHASH, (uintptr_t)t->vhash, t->vhash != NULL ? (t->vhash_mask + 1u) * 8u + 1024u : 0u },
        { R_MT, (uintptr_t)t->merge_slots, t->merge_slots != NULL ? (t->merge_mask + 1u) * 64u : 0u },
        { R_PF, (uintptr_t)t->pairf, t->pairf != NULL ? (t->merge_mask + 1u) * 4u : 0u },
        { R_BP, (uintptr_t)t->bytepair, t->bytepair != NULL ? 65536u * 4u : 0u },
        { R_PM, (uintptr_t)t->premerge, UINT64_MAX },
        { R_APM, (uintptr_t)t->apm, UINT64_MAX },
        { R_R2I, (uintptr_t)t->rank2id, t->rank2id != NULL ? (uint64_t)t->n_merges * 4u : 0u },
        { R_B2I, (uintptr_t)t->byte2id, t->byte2id != NULL ? 256u * 4u : 0u },
    };
    uintptr_t bpe_end = (uintptr_t)ctx->mem_bpe + ctx->mem_bpe_len;
    const size_t nbt = sizeof bt / sizeof bt[0];
    for (size_t i = 0; i < nbt; i++) {
        if (bt[i].a == 0u || bt[i].len == 0u) { continue; }
        uintptr_t e = bt[i].len == UINT64_MAX ? bpe_end : bt[i].a + bt[i].len;
        for (size_t j = 0; j < nbt; j++) { if (bt[j].a > bt[i].a && bt[j].a < e) { e = bt[j].a; } }   /* the next table */
        RG(bt[i].r, bt[i].a, e);
    }
    RG(R_STACK, (uintptr_t)stack - (8u << 20), (uintptr_t)stack + 4096u);
    int one = nrg;                                     /* regions below: no distinct-line count */
    RG(R_TABLES, ctx->mem_tables, (uintptr_t)ctx->mem_tables + ctx->mem_tables_len);
    RG(R_TABLES, ctx->mem_bpe, bpe_end);
    RG(R_TABLES, ctx, (uintptr_t)ctx + sizeof *ctx);
#undef RG
    for (int i = 0; i < one; i++) {
        uint64_t lines = ((rg[i].hi + 63u) >> 6) - (rg[i].lo >> 6);
        for (int st = 0; st < NS; st++) {
            free(rg[i].dist[st]);
            rg[i].dist[st] = rg[i].r == R_STACK ? NULL : calloc((size_t)(lines + 7u) / 8u, 1);
        }
    }
}

void tb_count_on(int on);
void tb_count_on(int on)                          /* 1: reset and count, 2: count on (resume), 0: off */
{
    if (on == 1) {
        memset(cb, 0, sizeof cb); memset(cl, 0, sizeof cl); memset(cpy, 0, sizeof cpy); memset(lastl, 0, sizeof lastl);
        memset(dl, 0, sizeof dl);
        for (int i = 0; i < nrg; i++) {
            uint64_t lines = ((rg[i].hi + 63u) >> 6) - (rg[i].lo >> 6);
            for (int st = 0; st < NS; st++) { if (rg[i].dist[st] != NULL) { memset(rg[i].dist[st], 0, (size_t)(lines + 7u) / 8u); } }
        }
    }
    counting = on;
}

void tb_count_print(FILE *f);
void tb_count_print(FILE *f)
{
    for (int st = 0; st < NS; st++) {
        for (int r = 0; r < NR; r++) {
            if (cb[st][r][0] | cb[st][r][1]) {          /* bytes read / written / line changes / distinct lines */
                fprintf(f, " %s_%s=%llu/%llu/%llu/%llu", SN[st], RN[r], (unsigned long long)cb[st][r][0],
                        (unsigned long long)cb[st][r][1], (unsigned long long)cl[st][r], (unsigned long long)dl[st][r]);
            }
        }
        fprintf(f, " %s_copy=%llu", SN[st], (unsigned long long)cpy[st]);
    }
    fprintf(f, "\n");
}
#define STAGE_IN(s) int sv_ = cur; cur = (s)
#define STAGE_OUT() cur = sv_
#else
#define STAGE_IN(s) do { } while (0)
#define STAGE_OUT() do { } while (0)
#endif

/* ---- wrappers ------------------------------------------------------------------------------------ */
#define K1_WRAP(tier)                                                                       \
    uint64_t tb_k1_##tier(const toks_tables *t, toks_k1_args *a);                          \
    uint64_t tb_k1_##tier(const toks_tables *t, toks_k1_args *a)                           \
    {                                                                                       \
        STAGE_IN(S_K1);                                                                     \
        uint64_t p0 = a->pos, c0 = tick();                                                  \
        uint64_t r = toks_k1_added_find_##tier(t, a);                                       \
        live.k1_ticks += tick() - c0;                                                       \
        STAGE_OUT();                                                                        \
        live.k1_calls++;                                                                    \
        live.k1_bytes += a->next > p0 ? a->next - p0 : 0u;                                  \
        return r;                                                                           \
    }
#define K3_WRAP(name, tmpl, tier)                                                           \
    uint64_t tb_##name##_##tier(const toks_tables *t, toks_k3_args *a);                    \
    uint64_t tb_##name##_##tier(const toks_tables *t, toks_k3_args *a)                     \
    {                                                                                       \
        tt = t;                                                                             \
        STAGE_IN(S_K3);                                                                     \
        uint64_t p0 = a->pos, c0 = tick();                                                  \
        uint64_t r = toks_k3_scan_##tmpl##_##tier(t, a);                                    \
        live.k3_ticks += tick() - c0;                                                       \
        STAGE_OUT();                                                                        \
        live.k3_calls++;                                                                    \
        live.k3_bytes += a->pos - p0;                                                       \
        live.k3_pieces += r;                                                                \
        if (classify) { classify_round(a, p0, r); }                                         \
        return r;                                                                           \
    }
/* one round into the K5 fit: x = its piece classes, y = K5's own ticks */
static void k5_fit(const toks_k5_args *a, uint64_t hs, uint64_t hc, uint64_t ms, double y)
{
    uint64_t n1 = 0, nl = 0, s = a->start;
    for (uint64_t i = 0; i < a->n; i++) {
        uint64_t l = a->ends[i] - s;
        n1 += l == 1u;
        nl += l > TOKS_KEY_MAXLEN;
        s = a->ends[i];
    }
    double x[6] = { (double)n1, (double)(hs - n1), (double)hc, (double)(ms - nl), (double)nl, 1.0 };
    live.c_one += n1; live.c_static += hs - n1; live.c_cache += hc; live.c_mshort += ms - nl; live.c_long += nl;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) { live.xtx[i][j] += x[i] * x[j]; }
        live.xty[i] += x[i] * y;
    }
    live.yy += y * y;
}

#define K5_WRAP(tier)                                                                       \
    uint64_t tb_k5_##tier(const toks_tables *t, toks_k5_args *a);                          \
    uint64_t tb_k5_##tier(const toks_tables *t, toks_k5_args *a)                           \
    {                                                                                       \
        tt = t;                                                                             \
        uint64_t hs = a->hits_static, hc = a->hits_cache, ms = a->misses;                   \
        uint64_t k6t = live.k6_ticks, k6c = live.k6_calls;                                  \
        STAGE_IN(S_K5);                                                                     \
        uint64_t c0 = tick();                                                               \
        uint64_t r = toks_k5_encode_##tier(t, a);                                           \
        uint64_t dt = tick() - c0;                                                          \
        STAGE_OUT();                                                                        \
        live.k5_ticks += dt;                                                                \
        live.k5_calls++;                                                                    \
        live.k5_pieces += a->n;                                                             \
        live.k5_ids += r;                                                                   \
        hs = a->hits_static - hs;  /* the counters are in-out (since init): this round's */ \
        hc = a->hits_cache - hc;                                                            \
        ms = a->misses - ms;                                                                \
        live.hits_static += hs;                                                             \
        live.hits_cache += hc;                                                              \
        live.misses += ms;                                                                  \
        double o = ovh_ticks();                                                             \
        k5_fit(a, hs, hc, ms, (double)dt - o - (double)(live.k6_ticks - k6t) -              \
                                  2.0 * o * (double)(live.k6_calls - k6c));                 \
        return r;                                                                           \
    }

#define K6_WRAP(tier)                                                                       \
    uint64_t tb_k6_##tier(const toks_tables *t, toks_k6_args *a);                          \
    uint64_t tb_k6_##tier(const toks_tables *t, toks_k6_args *a)                           \
    {                                                                                       \
        STAGE_IN(S_K6);                                                                     \
        uint64_t c0 = tick();                                                               \
        uint64_t r = toks_k6_bpe_##tier(t, a);                                              \
        uint64_t dt = tick() - c0;                                                          \
        STAGE_OUT();                                                                        \
        live.k6_ticks += dt;                                                                \
        live.k6_calls++;                                                                    \
        if (a->len <= TOKS_KEY_MAXLEN) { live.k6s_ticks += dt; live.k6s_calls++; }          \
        live.k6_bytes += a->len;                                                            \
        live.k6_ids += r;                                                                   \
        live.k6_merges += a->merges;                                                        \
        live.k6_hist[hb(a->len)]++;                                                         \
        if (classify) { tt = t; classify_k6(a->piece, a->len, r, a->out); }                 \
        return r;                                                                           \
    }

K1_WRAP(c)
K3_WRAP(k3, cl100k, c)
K3_WRAP(k3o, o200k, c)
K5_WRAP(c)
K6_WRAP(c)
#if TOKS_HAVE_K1_NEON
K1_WRAP(neon)
#endif
#if TOKS_HAVE_K3_CL100K_NEON
K3_WRAP(k3, cl100k, neon)
#endif
#if TOKS_HAVE_K3_O200K_NEON
K3_WRAP(k3o, o200k, neon)
#endif
#if TOKS_HAVE_K5_NEON
K5_WRAP(neon)
K6_WRAP(neon)
#endif
#if TOKS_HAVE_K1_AVX2
K1_WRAP(avx2)
#endif
#if TOKS_HAVE_K3_CL100K_AVX2
K3_WRAP(k3, cl100k, avx2)
#endif
#if TOKS_HAVE_K3_O200K_AVX2
K3_WRAP(k3o, o200k, avx2)
#endif
#if TOKS_HAVE_K5_AVX2
K5_WRAP(avx2)
K6_WRAP(avx2)
#endif

/* ---- report --------------------------------------------------------------------------------------- */
static void words_stats(uint64_t *eligible, uint64_t *placed, uint64_t *slots)
{
    *eligible = *placed = *slots = 0;
    if (tt == NULL) { return; }
    for (uint32_t id = 0; id < tt->n_ids; id++) {
        uint32_t o = tt->tok_off[id], l = tt->tok_off[id + 1] - o, got = 0;
        if (l >= 2 && l <= TOKS_KEY_MAXLEN && bpe_vhash_find(tt, tt->tok_bytes + o, l, &got) && got == id) { (*eligible)++; }
    }
    if (tt->words == NULL) { return; }
    *slots = (tt->words_mask + 1) * 2;
    for (uint64_t b = 0; b <= tt->words_mask; b++) {
        for (int w = 0; w < 2; w++) { *placed += tt->words[b * TOKS_BUCKET + (uint64_t)w * 16 + 15] != 0; }
    }
}

void tb_print(FILE *f, int use_live, double total_s, double ghz)
{
    const tb_t *s = use_live ? &live : &snap;
    double hz = tick_hz(), o = ovh_ticks();
    if (total_s >= 0.0) {
        double k1 = ((double)s->k1_ticks - o * (double)s->k1_calls) / hz;
        double k3 = ((double)s->k3_ticks - o * (double)s->k3_calls) / hz;
        double k6 = ((double)s->k6_ticks - o * (double)s->k6_calls) / hz;
        /* K5's own: minus K6 and minus the K6 wrapper's whole timer pair, which sits inside K5's window */
        double k5 = ((double)s->k5_ticks - o * (double)s->k5_calls) / hz - k6 - 2.0 * o * (double)s->k6_calls / hz;
        double drv = total_s - k1 - k3 - k5 - k6;
        /* the K5 fit: solve xtx b = xty (gaussian elimination, partial pivoting) */
        double m[6][7], b[6];
        for (int i = 0; i < 6; i++) { for (int j = 0; j < 6; j++) { m[i][j] = s->xtx[i][j]; } m[i][6] = s->xty[i]; }
        for (int c = 0; c < 6; c++) {
            int pr = c;
            for (int r = c + 1; r < 6; r++) { if (fabs(m[r][c]) > fabs(m[pr][c])) { pr = r; } }
            for (int j = 0; j < 7; j++) { double x = m[c][j]; m[c][j] = m[pr][j]; m[pr][j] = x; }
            for (int r = 0; r < 6; r++) {
                if (r == c || m[c][c] == 0.0) { continue; }
                double f = m[r][c] / m[c][c];
                for (int j = c; j < 7; j++) { m[r][j] -= f * m[c][j]; }
            }
        }
        for (int i = 0; i < 6; i++) { b[i] = m[i][i] != 0.0 ? m[i][6] / m[i][i] : 0.0; }
        double sse = s->yy, ny = s->xtx[5][5], my = s->xty[5] / (ny > 0 ? ny : 1), sst = s->yy - ny * my * my;
        for (int i = 0; i < 6; i++) { sse -= b[i] * s->xty[i]; }
        fprintf(f, " k5fit_ns=%.2f,%.2f,%.2f,%.2f,%.2f,%.1f k5fit_r2=%.3f k5_classes=%llu,%llu,%llu,%llu,%llu"
                   " k6short_calls=%llu k6short_s=%.6f",
                b[0] / hz * 1e9, b[1] / hz * 1e9, b[2] / hz * 1e9, b[3] / hz * 1e9, b[4] / hz * 1e9, b[5] / hz * 1e9,
                sst > 0 ? 1.0 - sse / sst : 0.0, (unsigned long long)s->c_one, (unsigned long long)s->c_static,
                (unsigned long long)s->c_cache, (unsigned long long)s->c_mshort, (unsigned long long)s->c_long,
                (unsigned long long)s->k6s_calls, ((double)s->k6s_ticks - o * (double)s->k6s_calls) / hz);
        fprintf(f, " total_s=%.6f k1_s=%.6f k3_s=%.6f k5own_s=%.6f k6_s=%.6f driver_s=%.6f tick_hz=%.0f ovh_ticks=%.1f"
                   " ghz=%.3f", total_s, k1, k3, k5, k6, drv, hz, o, ghz);
        fprintf(f, " k1_calls=%llu k1_bytes=%llu k3_calls=%llu k3_bytes=%llu k3_pieces=%llu k5_calls=%llu k5_pieces=%llu"
                   " k5_ids=%llu hits_static=%llu hits_cache=%llu misses=%llu k6_calls=%llu k6_bytes=%llu k6_ids=%llu"
                   " k6_merges=%llu k6_hist=",
                (unsigned long long)s->k1_calls, (unsigned long long)s->k1_bytes, (unsigned long long)s->k3_calls,
                (unsigned long long)s->k3_bytes, (unsigned long long)s->k3_pieces, (unsigned long long)s->k5_calls,
                (unsigned long long)s->k5_pieces, (unsigned long long)s->k5_ids, (unsigned long long)s->hits_static,
                (unsigned long long)s->hits_cache, (unsigned long long)s->misses, (unsigned long long)s->k6_calls,
                (unsigned long long)s->k6_bytes, (unsigned long long)s->k6_ids, (unsigned long long)s->k6_merges);
        for (int i = 0; i < NH; i++) { fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)s->k6_hist[i]); }
        fprintf(f, "\n");
        return;
    }
    uint64_t el, pl, sl;
    words_stats(&el, &pl, &sl);
    fprintf(f, " pieces=%llu distinct=%llu distinct_long=%llu distinct_vocab=%llu distinct_multi=%llu"
               " m_long=%llu m_vocab_unplaced=%llu m_vocab_gt4=%llu m_multi_first=%llu m_multi_again=%llu"
               " m_multi_gt4=%llu hits_static=%llu hits_cache=%llu misses=%llu words_eligible=%llu words_placed=%llu"
               " words_slots=%llu cache_entries=%llu piece_hist=",
            (unsigned long long)s->pieces, (unsigned long long)s->distinct, (unsigned long long)s->distinct_long,
            (unsigned long long)s->distinct_vocab, (unsigned long long)s->distinct_multi,
            (unsigned long long)s->m_long, (unsigned long long)s->m_vocab_unplaced, (unsigned long long)s->m_vocab_gt4,
            (unsigned long long)s->m_multi_first, (unsigned long long)s->m_multi_again,
            (unsigned long long)s->m_multi_gt4, (unsigned long long)s->hits_static,
            (unsigned long long)s->hits_cache, (unsigned long long)s->misses, (unsigned long long)el,
            (unsigned long long)pl, (unsigned long long)sl, (unsigned long long)TOKS_CACHE_BUCKETS * 2u);
    for (int i = 0; i < NH; i++) { fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)s->piece_hist[i]); }
    fprintf(f, " k6_hist=");
    for (int i = 0; i < NH; i++) { fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)s->k6_hist[i]); }
    fprintf(f, "\n");
}
