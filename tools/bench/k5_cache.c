/* tools/bench/k5_cache.c: K5's dynamic cache (kernels.md §6: 32,768 64-byte buckets x 2 ways = 2 MiB) as a model,
 * exact on the real piece stream of tools/bench/e2e.c's calls, so a fill or placement rule is measured in hits and in
 * cache lines touched before anything is built. Not a SPEC §12 cell: it counts, it does not time.
 *
 *   k5_cache <tokenizer> <chunk> <measured file>... -- <other-text file>...
 *
 * Calls: the measured files concatenated and cut like e2e.c (after the first '\n' at or past each chunk boundary; 0 =
 * one call), the other text cut at 4096 (e2e.c's WARM_CHUNK). A call's pieces come from toks_pieces in K3 rounds of
 * TOKS_CHUNK_PIECES. Per piece, k5_c.c's semantics: a one-byte piece is a static hit; a piece of 2..15 bytes is keyed
 * (bpe_key_at, bpe_key_hash) and on a WARM scratch (>= TOKS_K5_WARM pieces since init when its round starts) probes the
 * cache, then the static words table, then runs K6, a static answer filling the cache; on a FRESH one the static table
 * first, then the cache, then K6, a static answer not filling; a K6 answer of <= 4 ids fills; a longer piece runs K6 (a
 * miss) and is never cached. States as e2e.c runs them: cold (init before every call), pass (init, the measured text),
 * warm (the measured text again), lang (init, the other text, the measured text).
 * Rules (b1 = h & mask, b2 = rotr32(h, 16) & mask, the static table's second choice; a bucket is full when its way 1
 * is live; a fill is the shift fill: way 0 to way 1, the new entry to way 0):
 *   today   b1 only (the library)
 *   2shft   every answer two-choice: a fill goes to b1 while b1 is not full, else to b2 while b2 is not, else to b1; a
 *           lookup reads b2 only while b1 is full (a key is in b2 only then)
 *   2k6sf   only K6 answers two-choice: a static answer fills b1; a lookup reads b1, the static table, then b2 only while
 *           b1 has spilled (a bit set on b1 when one of its keys went to b2, carried while b1 stays full: exact), then K6
 *   2k6b32  2k6sf whose answers spill only while the fills since init are under 32,768: the rule kernels.md §6 names
 *           as the build if two-choice is ever revisited
 *   fa64k   one fully associative LRU of 65,536 entries: capacity alone
 *   inf     no eviction: compulsory and uncacheable misses only (the floor)
 * Output per rule and state: hits_static / hits_cache / misses (K5's counters, e2e.c's CTR line), and t2: the lines a
 * lookup or a fill reads beyond b1. Validated: today's counters equal the library's (e2e.c's CTR) in 96 of 96 (cell,
 * state) triples (llama3 / gpt2 / qwen38 x en code ml zh x 4096 + whole), and 2shft's and 2k6sf's equal those of K5's c
 * twin built with each rule in 96 of 96 (12 cells x 4 states at 4096; kernels.md §6).
 * Build: tools/bench/k5_cache.sh, or cc -std=c17 -O2 -Iinclude -Isrc/core -Isrc/platform -o build/k5_cache
 *        tools/bench/k5_cache.c build/<os>-<isa>/libtoks.a -lpthread */
#define _POSIX_C_SOURCE 200809L
#include "bpe.h"
#include "kernels.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K_ONE, K_LONG, K_STATIC, K_K6, K_K6BIG };       /* one byte; > 15 bytes; static entry; K6 <= 4 ids; > 4 ids */
typedef struct piece { uint64_t lo, hi; uint32_t h, kind; } piece;
typedef struct calls { uint64_t nc, np; uint64_t *cb; uint8_t *round; piece *ps; } calls;   /* cb[c]: call c's first */

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
        buf = realloc(buf, n + (uint64_t)m + 64);
        if (buf == NULL || fread(buf + n, 1, (size_t)m, f) != (size_t)m) { fprintf(stderr, "read %s\n", files[i]); exit(1); }
        fclose(f);
        n += (uint64_t)m;
    }
    *n_out = n;
    return buf;
}

/* K6's id count per key, computed once (open addressing on the key) */
#define KCAP (1u << 22)
static uint64_t *KL, *KH;
static uint32_t *KN;
static uint32_t k6_count(const toks_ctx *ctx, const uint8_t *p, uint64_t len, bpe_key k, uint8_t *work, uint64_t wb,
                         uint32_t *ids)
{
    uint64_t j = ((k.lo * 0x9E3779B97F4A7C15ull) ^ (k.hi * 0xC2B2AE3D27D4EB4Full)) >> 20 & (KCAP - 1u);
    while (KN[j] != 0u) {
        if (KL[j] == k.lo && KH[j] == k.hi) { return KN[j]; }
        j = (j + 1u) & (KCAP - 1u);
    }
    toks_k6_args a = { p, len, ids, work, wb, 0u, 0u, 0u };
    uint32_t n = (uint32_t)toks_k6_bpe_c(&ctx->t, &a);
    KL[j] = k.lo, KH[j] = k.hi, KN[j] = n;
    return n;
}

static void load_calls(calls *cs, const toks_ctx *ctx, char **files, int nf, uint64_t chunk)
{
    uint64_t n;
    uint8_t *buf = read_all(files, nf, &n);
    memset(cs, 0, sizeof *cs);
    uint64_t sb = toks_scratch_bytes(ctx, n, 0);
    void *scr = malloc(sb);
    uint32_t *ends = malloc((n + 64) * 4), *ids = malloc((n + 64) * 4);
    uint64_t wb = TOKS_BPE_WORK_BYTES(n) + 64;
    uint8_t *work = malloc(wb);
    cs->ps = malloc((n + 64) * sizeof(piece));
    cs->cb = malloc((n / (chunk != 0u ? chunk : n + 1u) + 4) * sizeof(uint64_t));
    cs->round = calloc(n + 64, 1);
    if (scr == NULL || ends == NULL || ids == NULL || work == NULL || cs->ps == NULL || cs->cb == NULL || cs->round == NULL ||
        toks_scratch_init(ctx, scr, sb, 0) != 0) {
        fprintf(stderr, "alloc / init\n");
        exit(1);
    }
    for (uint64_t p = 0; p < n;) {
        uint64_t e = n;
        if (chunk != 0u && p + chunk < n) {
            e = p + chunk;
            while (e < n && buf[e - 1] != '\n') { e++; }
        }
        int64_t np = toks_pieces(ctx, buf + p, e - p, 0, ends, n + 64, scr);
        if (np < 0) { fprintf(stderr, "toks_pieces: %lld\n", (long long)np); exit(1); }
        cs->cb[cs->nc++] = cs->np;
        uint64_t prev = 0;
        for (uint64_t i = 0; i < (uint64_t)np; i++) {
            piece *q = &cs->ps[cs->np];
            uint64_t s = prev, len = ends[i] - prev;
            prev = ends[i];
            cs->round[cs->np++] = i % TOKS_CHUNK_PIECES == 0u;   /* a K3 round starts here */
            q->lo = q->hi = 0, q->h = 0;
            if (len == 1u) { q->kind = K_ONE; continue; }
            if (len > (uint64_t)TOKS_KEY_MAXLEN) { q->kind = K_LONG; continue; }
            bpe_key k = bpe_key_at(buf + p, e - p, s, len);
            q->lo = k.lo, q->hi = k.hi, q->h = bpe_key_hash(k);
            if (ctx->t.words != NULL && bpe_w3_probe(ctx->t.words, ctx->t.words_mask, q->h, k) != NULL) {
                q->kind = K_STATIC;
            } else {
                q->kind = k6_count(ctx, buf + p + s, len, k, work, wb, ids) <= 4u ? K_K6 : K_K6BIG;
            }
        }
        p = e;
    }
    cs->cb[cs->nc] = cs->np;
    free(work), free(ids), free(ends), free(scr), free(buf);
}

/* the cache: buckets x 2 ways of (key, epoch); key 0 / 0 is empty (a real key's byte 15 is its length, >= 2) */
#define NB ((uint32_t)TOKS_CACHE_BUCKETS)
typedef struct way { uint64_t lo, hi; uint32_t ep; } way;
static way C[NB][2];
static uint32_t EP, SPILL[NB];                         /* SPILL[b]: the epoch in which b last spilled into a b2 */
static uint64_t SEEN, FILLS;                           /* pieces and fills since init */
enum { R_TODAY, R_2SHFT, R_2K6SF, R_2K6B32, R_FA, R_INF, NRULE };
static const char *const RN[NRULE] = { "today", "2shft", "2k6sf", "2k6b32", "fa64k", "inf" };
static int R;
typedef struct cnt { uint64_t st, ca, mi, t2; } cnt;

/* fa64k / inf: an exact LRU (65,536 entries) / an unbounded set over keys, open addressing with backward-shift
 * deletion and a doubly linked recency list */
#define FCAP (1u << 22)
static uint64_t *FL, *FH, *LP, *LN, FN, LHEAD = UINT64_MAX, LTAIL = UINT64_MAX;
static uint32_t *FE;
static uint64_t f_home(uint64_t lo, uint64_t hi) { return ((lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full)) >> 20 & (FCAP - 1u); }
static int64_t f_find(uint64_t lo, uint64_t hi)
{
    for (uint64_t j = f_home(lo, hi);; j = (j + 1u) & (FCAP - 1u)) {
        if (FL[j] == 0 && FH[j] == 0) { return -(int64_t)j - 1; }
        if (FL[j] == lo && FH[j] == hi) { return (int64_t)j; }
    }
}
static void l_unlink(uint64_t j)
{
    if (LP[j] != UINT64_MAX) { LN[LP[j]] = LN[j]; } else { LHEAD = LN[j]; }
    if (LN[j] != UINT64_MAX) { LP[LN[j]] = LP[j]; } else { LTAIL = LP[j]; }
}
static void l_push(uint64_t j)
{
    LP[j] = LTAIL, LN[j] = UINT64_MAX;
    if (LTAIL != UINT64_MAX) { LN[LTAIL] = j; } else { LHEAD = j; }
    LTAIL = j;
}
static void f_del(uint64_t j)
{
    l_unlink(j);
    FL[j] = FH[j] = 0;
    for (uint64_t i = (j + 1u) & (FCAP - 1u); FL[i] != 0 || FH[i] != 0; i = (i + 1u) & (FCAP - 1u)) {
        uint64_t k = f_home(FL[i], FH[i]);
        if ((i > j && (k <= j || k > i)) || (i < j && k <= j && k > i)) {
            FL[j] = FL[i], FH[j] = FH[i], FE[j] = FE[i], LP[j] = LP[i], LN[j] = LN[i];
            if (LP[j] != UINT64_MAX) { LN[LP[j]] = j; } else { LHEAD = j; }
            if (LN[j] != UINT64_MAX) { LP[LN[j]] = j; } else { LTAIL = j; }
            FL[i] = FH[i] = 0;
            j = i;
        }
    }
}
static int f_get(const piece *q)
{
    int64_t j = f_find(q->lo, q->hi);
    if (j < 0 || FE[j] != EP) { return 0; }
    l_unlink((uint64_t)j), l_push((uint64_t)j);
    return 1;
}
static void f_fill(const piece *q)
{
    int64_t j = f_find(q->lo, q->hi);
    if (j >= 0) {
        FE[j] = EP;
        l_unlink((uint64_t)j), l_push((uint64_t)j);
        return;
    }
    if (R == R_FA && FN >= 65536u) {
        f_del(LHEAD);
        FN--;
        j = f_find(q->lo, q->hi);
    }
    uint64_t s = (uint64_t)(-j - 1);
    FL[s] = q->lo, FH[s] = q->hi, FE[s] = EP;
    l_push(s);
    FN++;
}

static uint32_t b2_of(uint32_t h) { return ((h >> 16) | (h << 16)) & (NB - 1u); }
static int live(uint32_t b, int w) { return C[b][w].ep == EP && (C[b][w].lo | C[b][w].hi) != 0; }
static int full(uint32_t b) { return live(b, 1); }
static int has(uint32_t b, const piece *q)            /* bpe_cache_get: the first way holding the key decides */
{
    for (int w = 0; w < 2; w++) {
        if (C[b][w].lo == q->lo && C[b][w].hi == q->hi) { return C[b][w].ep == EP; }
    }
    return 0;
}
static void shift_fill(uint32_t b, const piece *q)    /* bpe_cache_fill */
{
    C[b][1] = C[b][0];
    C[b][0].lo = q->lo, C[b][0].hi = q->hi, C[b][0].ep = EP;
}
static void fill2(const piece *q, cnt *k, int spill_ok)   /* b1 while not full, else b2 while not full, else b1 */
{
    uint32_t b1 = q->h & (NB - 1u), b2 = b2_of(q->h);
    FILLS++;
    if (!full(b1) || !spill_ok || b2 == b1) { shift_fill(b1, q); return; }
    k->t2++;                                           /* is b2 full? */
    if (full(b2)) { shift_fill(b1, q); return; }
    shift_fill(b2, q);
    SPILL[b1] = EP;
}

/* b2's probe, when the rule reads it for this piece's b1: 2shft while b1 is full, 2k6sf / 2k6b32 while b1 has spilled */
static int get_b2(const piece *q, cnt *k)
{
    uint32_t b1 = q->h & (NB - 1u);
    if (!full(b1) || (R != R_2SHFT && SPILL[b1] != EP)) { return 0; }
    k->t2++;
    return has(b2_of(q->h), q);
}
static void fill_static(const piece *q, cnt *k)       /* a warm scratch's static answer */
{
    if (R == R_2SHFT) { fill2(q, k, 1); return; }
    FILLS++;
    shift_fill(q->h & (NB - 1u), q);
}
static void fill_k6(const piece *q, cnt *k)           /* a K6 answer of <= 4 ids */
{
    if (R == R_TODAY) {
        FILLS++;
        shift_fill(q->h & (NB - 1u), q);
        return;
    }
    fill2(q, k, R != R_2K6B32 || FILLS < 32768u);
}

static void run_call(const calls *cs, uint64_t c, cnt *k)
{
    int warm = 0;
    for (uint64_t i = cs->cb[c]; i < cs->cb[c + 1]; i++) {
        if (cs->round[i]) { warm = SEEN >= (uint64_t)TOKS_K5_WARM; }
        SEEN++;
        const piece *q = &cs->ps[i];
        int st = q->kind == K_STATIC;
        if (q->kind == K_ONE) { k->st++; continue; }
        if (q->kind == K_LONG) { k->mi++; continue; }
        if (R == R_FA || R == R_INF) {                 /* the set models: k5_c.c's order, one place per key */
            if (warm && f_get(q)) { k->ca++; continue; }
            if (st) {
                k->st++;
                if (warm) { f_fill(q); }
                continue;
            }
            if (!warm && f_get(q)) { k->ca++; continue; }
            k->mi++;
            if (q->kind == K_K6) { f_fill(q); }
            continue;
        }
        uint32_t b1 = q->h & (NB - 1u);
        if (warm) {                                    /* the cache (2shft: both buckets), the static table, 2k6: b2 */
            if (has(b1, q) || (R == R_2SHFT && get_b2(q, k))) { k->ca++; continue; }
            if (st) {
                k->st++;
                fill_static(q, k);
                continue;
            }
            if (R != R_TODAY && R != R_2SHFT && get_b2(q, k)) { k->ca++; continue; }
        } else {                                       /* the static table, then the cache */
            if (st) { k->st++; continue; }
            if (has(b1, q) || (R != R_TODAY && get_b2(q, k))) { k->ca++; continue; }
        }
        k->mi++;
        if (q->kind == K_K6) { fill_k6(q, k); }
    }
}

static void init_scr(void)
{
    EP++;                                              /* toks_scratch_init: every entry stale at once */
    SEEN = FILLS = 0;
    if (R == R_FA || R == R_INF) { memset(FL, 0, (size_t)FCAP * 8), memset(FH, 0, (size_t)FCAP * 8), FN = 0, LHEAD = LTAIL = UINT64_MAX; }
}

int main(int argc, char **argv)
{
    int sep = 0;
    for (int i = 3; i < argc; i++) { if (strcmp(argv[i], "--") == 0) { sep = i; } }
    if (argc < 4 || sep == 0) { fprintf(stderr, "usage: k5_cache <tokenizer> <chunk> <measured>... -- <other>...\n"); return 2; }
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, argv[1], NULL) != 0) { fprintf(stderr, "load %s\n", argv[1]); return 1; }
    KL = calloc(KCAP, 8), KH = calloc(KCAP, 8), KN = calloc(KCAP, 4);
    FL = calloc(FCAP, 8), FH = calloc(FCAP, 8), FE = calloc(FCAP, 4), LP = malloc((size_t)FCAP * 8), LN = malloc((size_t)FCAP * 8);
    if (KL == NULL || KH == NULL || KN == NULL || FL == NULL || FH == NULL || FE == NULL || LP == NULL || LN == NULL) { return 1; }
    calls m, o;
    load_calls(&m, ctx, argv + 3, sep - 3, strtoull(argv[2], NULL, 10));
    load_calls(&o, ctx, argv + sep + 1, argc - sep - 1, 4096u);
    uint64_t kind[5] = { 0 };
    for (uint64_t i = 0; i < m.np; i++) { kind[m.ps[i].kind]++; }
    printf("K5CACHE tok=%s chunk=%s calls=%" PRIu64 " pieces=%" PRIu64 " one=%" PRIu64 " long=%" PRIu64 " static=%" PRIu64
           " k6=%" PRIu64 " k6big=%" PRIu64 " other_calls=%" PRIu64 " other_pieces=%" PRIu64 "\n", argv[1], argv[2], m.nc,
           m.np, kind[K_ONE], kind[K_LONG], kind[K_STATIC], kind[K_K6], kind[K_K6BIG], o.nc, o.np);
    for (R = 0; R < NRULE; R++) {
        cnt st[4] = { { 0 } }, scrap = { 0 };
        memset(C, 0, sizeof C), memset(SPILL, 0, sizeof SPILL);
        EP = 1;
        for (uint64_t c = 0; c < m.nc; c++) { init_scr(); run_call(&m, c, &st[0]); }   /* cold: init before every call */
        init_scr();
        for (uint64_t c = 0; c < o.nc; c++) { run_call(&o, c, &scrap); }   /* pass: after other text (then init) */
        init_scr();
        for (uint64_t c = 0; c < m.nc; c++) { run_call(&m, c, &st[1]); }
        for (uint64_t c = 0; c < m.nc; c++) { run_call(&m, c, &st[2]); }   /* warm */
        init_scr();
        for (uint64_t c = 0; c < o.nc; c++) { run_call(&o, c, &scrap); }
        for (uint64_t c = 0; c < m.nc; c++) { run_call(&m, c, &st[3]); }   /* lang */
        printf("RULE %-6s", RN[R]);
        static const char *const sn[4] = { "cold", "pass", "warm", "lang" };
        for (int s = 0; s < 4; s++) {
            printf(" | %s %" PRIu64 "/%" PRIu64 "/%" PRIu64 " t2 %" PRIu64, sn[s], st[s].st, st[s].ca, st[s].mi, st[s].t2);
        }
        printf("\n");
        fflush(stdout);
    }
    toks_unload(ctx);
    return 0;
}
