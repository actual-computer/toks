/*
 * test_alloc.c: the load path's memory, through this program's own platform allocator (T11: E17, PA26, G4b, G9).
 * toks_plat_alloc / toks_plat_free / toks_plat_arena / toks_plat_arena_free / toks_plat_hint_huge are defined here,
 * so they win the link over mem.o (test_api.c's toks_cpu_features stand-in wins over cpu.o the same way). They
 * count every request, can fail the k-th one (or every arena request), and keep a table of the live blocks: an
 * arena is zeroed, page-aligned memory as mem.c's is, and freeing one with another length than it was given fails.
 *  - TOKS_E_NOMEM, "load-time allocation failed" (E17): each fixture (byte-level, the generic engine, spm, unigram;
 *    gpt2 and a wordpiece model from the tokenizer cache when there) loaded with its k-th request failing, for every k up to the requests a
 *    clean load makes, through toks_load_mem_copy and (the first fixture) toks_load: TOKS_E_NOMEM with diag.code
 *    the same and *out NULL, and no block left live (nothing leaked, nothing half-built); or 0, when the request
 *    had a fallback, and then the context encodes the probe texts as the clean one does and unload frees it all.
 *  - "a context is read-only after load", "a failure leaves the context unchanged" (G4b, G9): the blocks live after
 *    a load are the context's; they are hashed before and after a battery of every entry point on it (encode and
 *    pieces in every mode and cap, split_points, encode_bound, decode, token, token_to_id and id_flags on every id,
 *    stream init / push / flush / hold, get_info, scratch_bytes / init), the failing calls included (mode 3, unknown
 *    bits, NULL with a length, an id beyond the table, a short scratch or out, cap 0): unchanged. The battery makes
 *    no request (toks.h: after load there is no allocation).
 *  - toks_par, "TOKS_E_NOMEM when a worker's scratch cannot grow" (PA26): a pool on gpt2style whose arena requests
 *    fail after create: toks_par_encode of 1 MiB returns TOKS_E_NOMEM; with the allocator back the same call
 *    equals toks_encode.
 * Run from the repository root (make test does).
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "toks.h"
#include "guard.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* ================================================================ the stand-in allocator */

#define MAX_LIVE 8192u
typedef struct { void *p; uint64_t n; int arena; } blk;
static blk live[MAX_LIVE];
static uint32_t n_live;
static uint64_t n_req;              /* requests (alloc + arena) since the last reset */
static uint64_t fail_at;            /* the request that fails (1-based); 0: none */
static int fail_arenas;             /* every arena request fails */
static long bad_frees;              /* a free of a block not live, or an arena freed with another length */
static atomic_flag lock_flag = ATOMIC_FLAG_INIT;

static void lock(void) { while (atomic_flag_test_and_set_explicit(&lock_flag, memory_order_acquire)) { } }
static void unlock(void) { atomic_flag_clear_explicit(&lock_flag, memory_order_release); }

/* 1 when this request is to fail (counted either way) */
static int take(int arena)
{
    n_req++;
    return n_req == fail_at || (arena && fail_arenas);
}

static void track(void *p, uint64_t n, int arena)
{
    if (n_live < MAX_LIVE) { live[n_live++] = (blk){ p, n, arena }; } else { bad_frees++; }
}

static void untrack(void *p, uint64_t n, int arena)
{
    for (uint32_t i = 0; i < n_live; i++) {                 /* bound: n_live */
        if (live[i].p != p) { continue; }
        if (live[i].arena != arena || (arena && live[i].n != n)) { bad_frees++; }
        live[i] = live[--n_live];
        return;
    }
    bad_frees++;
}

void *toks_plat_alloc(uint64_t n);
void toks_plat_free(void *p, uint64_t n);
uint8_t *toks_plat_arena(uint64_t n);
void toks_plat_arena_free(uint8_t *p, uint64_t n);
void toks_plat_hint_huge(void *p, uint64_t n);

void *toks_plat_alloc(uint64_t n)
{
    if (n == 0u || n > (uint64_t)SIZE_MAX) { return NULL; }
    lock();
    void *p = take(0) ? NULL : malloc((size_t)n);
    if (p != NULL) { track(p, n, 0); }
    unlock();
    return p;
}

void toks_plat_free(void *p, uint64_t n)
{
    if (p == NULL) { return; }
    lock();
    untrack(p, n, 0);
    unlock();
    free(p);
}

uint8_t *toks_plat_arena(uint64_t n)
{
    if (n == 0u || n > (uint64_t)SIZE_MAX - 8192u) { return NULL; }
    size_t z = ((size_t)n + 4095u) & ~(size_t)4095u;       /* whole pages, as mem.c maps them */
    lock();
    uint8_t *p = take(1) ? NULL : (uint8_t *)guard_aligned_alloc(4096u, z);
    if (p != NULL) { memset(p, 0, z); track(p, n, 1); }    /* zeroed, as fresh pages are */
    unlock();
    return p;
}

void toks_plat_arena_free(uint8_t *p, uint64_t n)
{
    if (p == NULL) { return; }
    lock();
    untrack(p, n, 1);
    unlock();
    guard_aligned_free(p);
}

void toks_plat_hint_huge(void *p, uint64_t n) { (void)p; (void)n; }

static void reset(uint64_t k) { lock(); n_req = 0; fail_at = k; fail_arenas = 0; unlock(); }

/* ================================================================ helpers */

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = n > 0 ? (uint8_t *)malloc((size_t)n) : NULL;
    if (b == NULL || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

static const char *const PROBE[] = {
    "Hello, world! The quick brown fox jumps over the lazy dog. 12345 + 678 = 13023.",
    "caf\xc3\xa9 na\xc3\xafve \xe4\xb8\xad\xe6\x96\x87 \xf0\x9f\x98\x80 \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n\n\tx  y",
    "<|endoftext|>a<s></s> <unk>[CLS] [SEP]\xe2\x96\x81\xe2\x96\x81 \xff\xfe\x80 end",
};
#define N_PROBE (sizeof PROBE / sizeof PROBE[0])
#define MAXIDS 1024u

/* the probe texts' ids under flags 0 into out[N_PROBE][MAXIDS], counts into n; 0 when every encode succeeded */
static int probe(const toks_ctx *c, uint32_t out[N_PROBE][MAXIDS], int64_t n[N_PROBE])
{
    uint64_t sb = toks_scratch_bytes(c, 4096u, 0u);
    void *scr = malloc((size_t)sb);
    int bad = scr == NULL || toks_scratch_init(c, scr, sb, 0u) != 0;
    for (size_t i = 0; !bad && i < N_PROBE; i++) {
        n[i] = toks_encode(c, PROBE[i], strlen(PROBE[i]), 0u, out[i], MAXIDS, scr);
        bad |= n[i] < 0 || n[i] > (int64_t)MAXIDS;
    }
    free(scr);
    return bad;
}

/* FNV-1a over every live block, in table order (the table is not reordered while nothing is freed) */
static uint64_t hash_live(uint32_t upto)
{
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < upto; i++) {
        const uint8_t *p = (const uint8_t *)live[i].p;
        for (uint64_t k = 0; k < live[i].n; k++) { h = (h ^ p[k]) * 1099511628211ull; }
        h = (h ^ live[i].n) * 1099511628211ull;
    }
    return h;
}

/* ================================================================ E17: every request failing in turn */

static void sweep(const char *label, const uint8_t *src, uint64_t len, const char *path)
{
    uint32_t ref[N_PROBE][MAXIDS], got[N_PROBE][MAXIDS];
    int64_t nref[N_PROBE], ngot[N_PROBE];
    toks_ctx *c = NULL;
    reset(0);
    int64_t r = path != NULL ? toks_load(&c, path, NULL) : toks_load_mem_copy(&c, src, len, NULL);
    uint64_t reqs = n_req;
    CHECK(r == 0 && c != NULL && reqs > 0u, "%s: clean load %" PRId64 ", %" PRIu64 " requests", label, r, reqs);
    if (c == NULL) { return; }
    CHECK(probe(c, ref, nref) == 0, "%s: probe", label);
    toks_unload(c);
    CHECK(n_live == 0u && bad_frees == 0, "%s: unload left %u blocks, %ld bad frees", label, n_live, bad_frees);
    uint32_t nomem = 0, fallback = 0;
    for (uint64_t k = 1; k <= reqs; k++) {                  /* bound: the clean load's requests */
        toks_diag d;
        memset(&d, 0, sizeof d);
        toks_load_opts o = { sizeof o, 0, 0, 0, &d };
        c = (toks_ctx *)(uintptr_t)1;
        reset(k);
        r = path != NULL ? toks_load(&c, path, &o) : toks_load_mem_copy(&c, src, len, &o);
        reset(0);
        if (r == TOKS_E_NOMEM) {
            nomem++;
            CHECK(c == NULL && d.code == TOKS_E_NOMEM && n_live == 0u && bad_frees == 0,
                  "%s: request %" PRIu64 " failing: *out %p, diag %" PRId64 " (%s), %u blocks left", label, k, (void *)c, d.code,
                  d.what, n_live);
        } else if (r == 0) {                                /* a request with a fallback: the same context */
            fallback++;
            CHECK(c != NULL && probe(c, got, ngot) == 0 && memcmp(ngot, nref, sizeof nref) == 0, "%s: request %" PRIu64
                  " failing, loaded: the probe differs", label, k);
            for (size_t i = 0; c != NULL && i < N_PROBE && ngot[i] == nref[i]; i++) {
                CHECK(memcmp(got[i], ref[i], (size_t)nref[i] * 4u) == 0, "%s: request %" PRIu64 ": probe %zu's ids", label, k, i);
            }
            toks_unload(c);
            CHECK(n_live == 0u && bad_frees == 0, "%s: request %" PRIu64 ": unload left %u blocks", label, k, n_live);
        } else {
            CHECK(0, "%s: request %" PRIu64 " failing: %" PRId64 " (%s), want TOKS_E_NOMEM", label, k, r, d.what);
            toks_unload(r == 0 ? c : NULL);
        }
        n_live = 0, bad_frees = 0;                          /* one failure is reported once */
    }
    printf("  %-28s %3" PRIu64 " requests: %u TOKS_E_NOMEM, %u loaded by a fallback\n", label, reqs, nomem, fallback);
}

/* ================================================================ G4b, G9: the context after every entry point */

static void battery(const char *label, const toks_ctx *c)
{
    toks_info in;
    memset(&in, 0, sizeof in);
    in.size = (uint32_t)sizeof in;
    CHECK(toks_get_info(c, &in) == 0, "%s: get_info", label);
    in.size = 7u;
    CHECK(toks_get_info(c, &in) == TOKS_E_ARG, "%s: get_info size 7", label);
    uint64_t sb = toks_scratch_bytes(c, 4096u, 0u);
    uint8_t *scr = (uint8_t *)malloc((size_t)sb), *small = (uint8_t *)malloc(256u);
    uint32_t *ids = (uint32_t *)malloc(4u * 8192u);
    uint8_t *bytes = (uint8_t *)malloc(65536u), hold[1024];
    uint64_t offs[16];
    if (scr == NULL || small == NULL || ids == NULL || bytes == NULL) { CHECK(0, "malloc"); free(scr); free(small); free(ids); free(bytes); return; }
    CHECK(toks_scratch_init(c, scr, sb, 1u << 21) == TOKS_E_ARG, "%s: scratch_init bit 21", label);
    CHECK(toks_scratch_init(c, small, 256u, 0u) != 0, "%s: scratch_init short", label);
    CHECK(toks_scratch_init(c, scr, sb, 0u) == 0, "%s: scratch_init", label);
    uint64_t seed = 7u;
    char rnd[512];
    for (size_t i = 0; i < sizeof rnd; i++) { rnd[i] = (char)(guard_rng(&seed) & 0xFFu); }
    for (size_t t = 0; t <= N_PROBE; t++) {
        const char *s = t < N_PROBE ? PROBE[t] : rnd;
        uint64_t len = t < N_PROBE ? strlen(PROBE[t]) : sizeof rnd;
        static const uint32_t FL[] = { 0u, 1u, 2u, 4u, 8u, 5u, 14u };
        for (size_t f = 0; f < sizeof FL / sizeof FL[0]; f++) {
            int64_t n = toks_encode(c, s, len, FL[f], ids, 8192u, scr);
            CHECK(n >= 0, "%s: encode %zu flags %u: %" PRId64, label, t, FL[f], n);
            (void)toks_encode(c, s, len, FL[f], NULL, 0u, scr);
            (void)toks_encode(c, s, len, FL[f], ids, 1u, scr);
            (void)toks_pieces(c, s, len, FL[f], ids + 4096, 4096u, scr);
            (void)toks_split_points(c, s, len, FL[f], 4u, offs, 16u, NULL);
            if (n > 0) {
                (void)toks_decode(c, ids, (uint64_t)n, 0u, bytes, 65536u);
                (void)toks_decode(c, ids, (uint64_t)n, TOKS_SKIP_SPECIAL, bytes, 3u);
                toks_stream st;
                toks_stream_init(c, &st, 0u);
                (void)toks_stream_push(c, &st, ids, (uint64_t)n, bytes, 65536u);
                (void)toks_stream_flush(c, &st, bytes, 65536u);
                toks_stream_init(c, &st, 0u);
                (void)toks_stream_hold(c, &st, hold, sizeof hold);
                (void)toks_stream_push(c, &st, ids, (uint64_t)n, bytes, 0u);         /* TOKS_E_CAP where it needs room */
                (void)toks_stream_push(c, &st, ids, (uint64_t)n, bytes, 65536u);
                (void)toks_stream_hold(c, &st, NULL, 0u);
                (void)toks_stream_flush(c, &st, bytes, 65536u);
            }
        }
        /* the failing calls */
        CHECK(toks_encode(c, s, len, 3u, ids, 8192u, scr) == TOKS_E_ARG, "%s: mode 3", label);
        CHECK(toks_encode(c, s, len, 64u, ids, 8192u, scr) == TOKS_E_ARG, "%s: bit 6", label);
        CHECK(toks_encode(c, NULL, 5u, 0u, ids, 8192u, scr) == TOKS_E_ARG, "%s: text NULL", label);
        CHECK(toks_encode(c, s, len, 0u, NULL, 5u, scr) == TOKS_E_ARG, "%s: out NULL", label);
        CHECK(toks_encode(c, s, len, 0u, ids, 8192u, NULL) == TOKS_E_SCRATCH, "%s: scratch NULL", label);
        CHECK(toks_encode(c, s, len, 0u, ids, 8192u, small) == TOKS_E_SCRATCH, "%s: scratch never initialized", label);
        CHECK(toks_pieces(c, s, len, 3u, ids, 8192u, scr) == TOKS_E_ARG, "%s: pieces mode 3", label);
        CHECK(toks_split_points(c, s, len, 3u, 4u, offs, 16u, NULL) == TOKS_E_ARG, "%s: split mode 3", label);
        CHECK(toks_split_points(c, NULL, 5u, 0u, 4u, offs, 16u, NULL) == TOKS_E_ARG, "%s: split text NULL", label);
    }
    uint32_t bad = in.n_ids;
    CHECK(toks_decode(c, &bad, 1u, 0u, bytes, 64u) == TOKS_E_ID, "%s: decode id n_ids", label);
    CHECK(toks_decode(c, ids, 1u, 4u, bytes, 64u) == TOKS_E_ARG, "%s: decode flags 4", label);
    CHECK(toks_decode(c, NULL, 1u, 0u, bytes, 64u) == TOKS_E_ARG, "%s: decode ids NULL", label);
    CHECK(toks_id_flags(c, bad) == TOKS_E_ID, "%s: id_flags n_ids", label);
    CHECK(toks_token_to_id(c, NULL, 3u) == TOKS_E_ARG, "%s: token_to_id NULL", label);
    for (uint32_t id = 0; id < in.n_ids; id++) {            /* bound: n_ids */
        uint64_t tl = 0;
        const uint8_t *tp = toks_token(c, id, &tl);
        (void)toks_id_flags(c, id);
        if (tp != NULL) { (void)toks_token_to_id(c, tp, tl); }
        if (tp != NULL && tl > 1u) { (void)toks_token_to_id(c, tp, tl - 1u); }
    }
    for (uint64_t len = 0; len < 64u; len++) { (void)toks_encode_bound(c, len << 20); }
    (void)toks_stream_bound(c, 100u);
    toks_stream st;
    toks_stream_init(c, &st, 2u);                           /* unknown flag: push and flush refuse */
    CHECK(toks_stream_push(c, &st, ids, 1u, bytes, 64u) == TOKS_E_ARG, "%s: push after init flags 2", label);
    free(scr); free(small); free(ids); free(bytes);
}

static void readonly(const char *label, const uint8_t *src, uint64_t len)
{
    toks_ctx *c = NULL;
    reset(0);
    CHECK(toks_load_mem_copy(&c, src, len, NULL) == 0 && c != NULL, "%s: load", label);
    if (c == NULL) { return; }
    uint32_t blocks = n_live;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < blocks; i++) { bytes += live[i].n; }
    uint64_t h0 = hash_live(blocks), r0 = n_req;
    battery(label, c);
    uint64_t h1 = hash_live(blocks);
    CHECK(n_live == blocks && n_req == r0, "%s: the battery made %" PRIu64 " requests, %u blocks live (was %u)", label, n_req - r0,
          n_live, blocks);
    CHECK(h1 == h0, "%s: the context's %u blocks (%" PRIu64 " bytes) changed during the battery", label, blocks, bytes);
    printf("  %-28s %u blocks, %" PRIu64 " bytes: unchanged by every entry point, no request after load\n", label, blocks, bytes);
    toks_unload(c);
    CHECK(n_live == 0u && bad_frees == 0, "%s: unload left %u blocks", label, n_live);
}

/* ================================================================ PA26: a worker's scratch that cannot grow */

static void par_nomem(const uint8_t *src, uint64_t len)
{
    toks_ctx *c = NULL;
    reset(0);
    CHECK(toks_load_mem_copy(&c, src, len, NULL) == 0 && c != NULL, "par: load");
    if (c == NULL) { return; }
    uint64_t tl = 1u << 20;
    char *text = (char *)malloc((size_t)tl);
    uint32_t *a = (uint32_t *)malloc((size_t)tl * 4u), *b = (uint32_t *)malloc((size_t)tl * 4u);
    uint64_t sb = toks_scratch_bytes(c, tl, 0u);
    void *scr = malloc((size_t)sb);
    toks_par *p = NULL;
    if (text == NULL || a == NULL || b == NULL || scr == NULL || toks_scratch_init(c, scr, sb, 0u) != 0 ||
        toks_par_create(&p, c, 4u, 0u) != 0) {
        CHECK(0, "par: setup");
    } else {
        for (uint64_t i = 0; i < tl; i++) { text[i] = PROBE[0][i % strlen(PROBE[0])]; }
        int64_t want = toks_encode(c, text, tl, 0u, a, tl, scr);
        lock(); fail_arenas = 1; unlock();
        int64_t r = toks_par_encode(p, text, tl, 0u, b, tl);
        lock(); fail_arenas = 0; unlock();
        CHECK(r == TOKS_E_NOMEM, "par: toks_par_encode of 1 MiB with no arena to grow into: %" PRId64, r);
        r = toks_par_encode(p, text, tl, 0u, b, tl);
        CHECK(want > 0 && r == want && memcmp(a, b, (size_t)want * 4u) == 0, "par: the same call with memory: %" PRId64 ", serial %"
              PRId64, r, want);
        printf("  %-28s toks_par_encode: TOKS_E_NOMEM without memory, then equal to serial (%" PRId64 " ids)\n", "par (gpt2style)", want);
    }
    toks_par_destroy(p);
    free(text); free(a); free(b); free(scr);
    toks_unload(c);
    CHECK(n_live == 0u && bad_frees == 0, "par: destroy and unload left %u blocks", n_live);
}

int main(void)
{
    static const char *const FIX[] = {
        "tests/data/compile/gpt2style.json", "tests/data/compile/nosplit.json", "tests/data/spm/llamalike.json",
        "tests/data/unigram/bound_bf_meta.json",
    };
    static const char *const CACHED[] = { "gpt2", "wp-minilm-l6" };   /* byte-level with real tables; wordpiece */
    enum { NF = sizeof FIX / sizeof FIX[0], NC = sizeof CACHED / sizeof CACHED[0], N = NF + NC };
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    char path[N][1024];
    const char *name[N];
    uint8_t *src[N];
    uint64_t len[N];
    for (size_t i = 0; i < N; i++) {
        if (i < NF) {
            snprintf(path[i], sizeof path[i], "%s", FIX[i]);
            name[i] = strrchr(FIX[i], '/') + 1;
        } else if (root != NULL && root[0] != 0) {
            snprintf(path[i], sizeof path[i], "%s/%s", root, CACHED[i - NF]);
            name[i] = CACHED[i - NF];
        } else {
            snprintf(path[i], sizeof path[i], "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", CACHED[i - NF]);
            name[i] = CACHED[i - NF];
        }
        len[i] = 0;
        src[i] = slurp(path[i], &len[i]);
        if (i < NF) { CHECK(src[i] != NULL, "%s (run from the source root)", path[i]); }
        else if (src[i] == NULL) { printf("SKIP %s: %s missing\n", name[i], path[i]); }
    }
    printf("TOKS_E_NOMEM: each request of a load failing in turn\n");
    if (src[0] != NULL) { sweep("gpt2style.json (toks_load)", NULL, 0u, FIX[0]); }
    for (size_t i = 0; i < N; i++) {
        if (src[i] != NULL) { sweep(name[i], src[i], len[i], NULL); }
    }
    printf("read-only after load\n");
    for (size_t i = 0; i < N; i++) {
        if (src[i] != NULL) { readonly(name[i], src[i], len[i]); }
    }
    printf("toks_par\n");
    if (src[0] != NULL) { par_nomem(src[0], len[0]); }
    for (size_t i = 0; i < N; i++) { free(src[i]); }
    printf("test_alloc: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
