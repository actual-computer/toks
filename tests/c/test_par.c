/*
 * test_par.c: toks_par (SPEC §0.3, §2.4, §4.10; src/par/par.c). It has no semantics of its own (T2), so every
 * result is compared with the serial toks_encode call on the same bytes, flags and capacity:
 *
 *   - toks_par_encode on inputs from empty to past one unit's 1 MiB limit (word text that cuts everywhere,
 *     added tokens, CJK and letter runs that cut rarely or never), every mode, post-processing on and off,
 *     capacities 0 (NULL out), short, exact and roomy, with canaries past cap; pools of 1, 2, 3, 8 and the
 *     default number of participants, each both as the pool decides (the cost model) and with every
 *     participant forced (env TOKS_PAR_EAGER=1 at create: the parallel paths on every input that has units);
 *   - toks_par_encode_batch: items of mixed lengths and capacities (0 with NULL out, short, exact, roomy),
 *     invalid items (their toks_encode argument errors), big items that the batch splits at cuts, canaries
 *     past every item's cap;
 *   - the policy: a call under 16 KiB never takes a second participant, no call takes more than the pool's
 *     cap, the default pool is a couple (<= 4), toks_par_get_info's fields and argument errors;
 *   - two caller threads sharing one pool, calls separated by sleeps longer than the spin (workers park and
 *     are woken), create / destroy cycles, the argument errors.
 * Tokenizers: the fixtures (tests/data/compile, tests/data/spm) and, when present, the real gpt2, llama3,
 * GLM 5.3, Qwen 3.8 and Gemma 4 files. Run under -fsanitize=thread and =address on the hosts.
 *
 *   test_par [scale]     scale 1 (make test): ~1.3 MiB inputs; larger scales add bigger inputs and more rounds
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <pthread.h>
#  include <time.h>
#endif

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rng = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static void sleep_ms(unsigned ms)
{
#if defined(_WIN32)
    Sleep(ms);
#else
    struct timespec ts = { (time_t)(ms / 1000u), (long)(ms % 1000u) * 1000000L };
    nanosleep(&ts, NULL);
#endif
}

static void set_eager(int on)            /* toks_par_create reads TOKS_PAR_EAGER */
{
#if defined(_WIN32)
    SetEnvironmentVariableA("TOKS_PAR_EAGER", on ? "1" : NULL);
#else
    if (on) { setenv("TOKS_PAR_EAGER", "1", 1); } else { unsetenv("TOKS_PAR_EAGER"); }
#endif
}

static toks_par *pool(toks_ctx *ctx, uint32_t n, uint32_t flags, int eager)
{
    toks_par *p = NULL;
    set_eager(eager);
    int64_t r = toks_par_create(&p, ctx, n, flags);
    set_eager(0);
    CHECK(r == 0 && p != NULL, "create %u eager %d: %" PRId64, n, eager, r);
    return p;
}

static toks_par_info info(toks_par *p)
{
    toks_par_info in;
    memset(&in, 0, sizeof in);
    in.size = sizeof in;
    CHECK(toks_par_get_info(p, &in) == 0, "get_info");
    return in;
}

static uint8_t *slurp(const char *path, uint64_t *len)
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

static toks_ctx *load(const char *name)
{
    char path[1024];
    if (strchr(name, '/') != NULL) { snprintf(path, sizeof path, "%s", name); }
    else {
        const char *root = getenv("TOKS_TOKENIZER_CACHE");
        if (root != NULL && root[0] != 0) { snprintf(path, sizeof path, "%s/%s", root, name); }
        else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    }
    uint64_t n;
    uint8_t *b = slurp(path, &n);
    if (b == NULL) { printf("SKIP %s: %s not found\n", name, path); return NULL; }
    toks_ctx *ctx = NULL;
    int64_t r = toks_load_mem_copy(&ctx, b, n, NULL);
    free(b);
    if (r != 0) { printf("SKIP %s: does not load here (%" PRId64 ")\n", name, r); return NULL; }
    return ctx;
}

/* ================================================================ inputs */

static const char *const WORDS[] = {
    "the", "of", "and", "token", "parallel", "split", "Hello", "world", "it's", "we'll", "x", "1234567",
    "3.14", "foo_bar", "(a, b)", "{", "}", ";", "->", "==", "\xc3\xa9t\xc3\xa9", "na\xc3\xafve", "\xd0\xbc\xd0\xb8\xd1\x80",
    "\xe4\xb8\xad\xe6\x96\x87", "\xe3\x80\x82", "e\xcc\x81", "\xf0\x9f\x98\x80", "<|a|>", "@", "<|endoftext|>",
    "<|im_start|>", "<|eot_id|>", "<bos>", "<|turn>", "\xe2\x96\x81", "  ", "\t", "\xff",
};
#define N_WORDS (sizeof WORDS / sizeof WORDS[0])

/* n bytes of text of a kind: 0 words and separators (cuts everywhere), 1 a long CJK / letter run with rare
 * breaks (few cuts), 2 one letter run (no cut at all) */
static uint64_t gen(uint8_t *t, uint64_t n, int kind)
{
    uint64_t k = 0;
    while (k < n) {
        const char *w;
        if (kind == 2) { w = "a"; }
        else if (kind == 1) { w = (rnd() % 997u == 0u) ? "\xe3\x80\x82 " : "\xe4\xb8\xad"; }
        else {
            uint64_t r = rnd() % 100u;
            w = r < 55u ? WORDS[rnd() % N_WORDS] : r < 85u ? " " : r < 93u ? "\n" : r < 97u ? ", " : "\n\n";
        }
        size_t l = strlen(w);
        if (k + l > n) { break; }
        memcpy(t + k, w, l);
        k += l;
    }
    return k;
}

/* ================================================================ one-big-input checks */

typedef struct ref { uint64_t n; uint32_t *ids; } ref;

static ref serial(toks_ctx *ctx, void *scr, const uint8_t *t, uint64_t len, uint32_t flags)
{
    ref r;
    r.ids = (uint32_t *)malloc(4u * (len + 80u));
    int64_t n = toks_encode(ctx, t, len, flags, r.ids, len + 80u, scr);
    CHECK(n >= 0, "serial encode %" PRId64, n);
    r.n = n < 0 ? 0u : (uint64_t)n;
    return r;
}

#define CANARY 0xA5A5A5A5u

/* toks_par_encode at capacity cap against the serial ids: the same count, the same prefix, nothing past cap;
 * the participants it took: <= the pool's cap, 1 under 16 KiB */
static void doc_check(toks_par *p, const char *what, const uint8_t *t, uint64_t len, uint32_t flags, ref *r, uint64_t cap)
{
    uint32_t *o = (uint32_t *)malloc(4u * (cap + 64u));
    for (uint64_t i = 0; i < cap + 64u; i++) { o[i] = CANARY; }
    int64_t n = toks_par_encode(p, t, len, flags, cap != 0u ? o : NULL, cap);
    int same = n == (int64_t)r->n;
    uint64_t m = r->n < cap ? r->n : cap;
    for (uint64_t i = 0; same && i < m; i++) { same = o[i] == r->ids[i]; }
    int clean = 1;
    for (uint64_t i = cap; i < cap + 64u; i++) { clean &= o[i] == CANARY; }
    CHECK(same && clean, "%s: len %" PRIu64 " flags %u cap %" PRIu64 ": n %" PRId64 " want %" PRIu64 " same %d clean %d",
          what, len, flags, cap, n, r->n, same, clean);
    toks_par_info in = info(p);
    CHECK(in.last >= 1u && in.last <= in.threads && (len >= (16u << 10) || in.last == 1u),
          "%s: len %" PRIu64 " took %u of %u participants", what, len, in.last, in.threads);
    free(o);
}

static const uint32_t FLAGS[6] = {
    0u, TOKS_NO_POSTPROCESS, TOKS_ADDED_NONSPECIAL, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS,
    TOKS_ADDED_NONSPECIAL | TOKS_CONTINUATION, TOKS_ADDED_NONE,
};

static void docs(const char *label, toks_ctx *ctx, int scale, int real)
{
    /* inputs and their serial ids once; then every pool on them */
    uint64_t sizes[] = { 0u, 1u, 100u, 40000u, 300000u, 1300000u, 3000000u };
    uint32_t nsz = scale > 1 ? 7u : (real ? 5u : 6u);
    enum { NF = 2 };
    uint8_t *t[7];
    uint64_t len[7];
    uint32_t fl[7][NF];
    ref r[7][NF];
    uint64_t sb = toks_scratch_bytes(ctx, 3100000u, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch");
    uint64_t with_cuts = 0, runs = 0, wide = 0;
    for (uint32_t si = 0; si < nsz; si++) {
        int kind = (si >= 3u && rnd() % 4u == 0u) ? (int)(1u + rnd() % 2u) : 0;
        t[si] = (uint8_t *)malloc(sizes[si] + 1u);
        len[si] = gen(t[si], sizes[si], kind);
        for (int fi = 0; fi < NF; fi++) {
            fl[si][fi] = FLAGS[rnd() % 6u];
            r[si][fi] = serial(ctx, scr, t[si], len[si], fl[si][fi]);
            uint64_t offs[4];
            if (toks_split_points(ctx, t[si], len[si], fl[si][fi], 32u, offs, 4u, NULL) > 0) { with_cuts++; }
        }
    }
    uint32_t pools[] = { 1u, 2u, 3u, 8u, 0u };
    for (uint32_t pi = 0; pi < 10u; pi++) {
        int eager = pi % 2u == 0u;
        toks_par *p = pool(ctx, pools[pi / 2u], 0u, eager);
        if (p == NULL) { continue; }
        for (uint32_t si = 0; si < nsz; si++) {
            for (int fi = 0; fi < NF; fi++) {
                uint64_t n = r[si][fi].n, caps[4] = { 0u, n / 2u, n, n + 40u };
                for (int ci = 0; ci < 4; ci++) {
                    if (ci != 3 && len[si] > 200000u && (rnd() % 3u) != 0u) { continue; }
                    doc_check(p, label, t[si], len[si], fl[si][fi], &r[si][fi], caps[ci]);
                    wide += info(p).last > 1u;
                    runs++;
                }
            }
            if (pi == 4u && si == 3u) { sleep_ms(30u); }   /* longer than the spin: the workers sleep, then wake */
        }
        toks_par_destroy(p);
    }
    printf("  %-14s toks_par_encode: %" PRIu64 " calls equal serial, %" PRIu64 " of them parallel (%" PRIu64
           " of %u inputs x flags have cuts)\n", label, runs, wide, with_cuts, nsz * NF);
    for (uint32_t si = 0; si < nsz; si++) {
        free(t[si]);
        for (int fi = 0; fi < NF; fi++) { free(r[si][fi].ids); }
    }
    free(scr);
}

/* ================================================================ batches */

static void batches(const char *label, toks_ctx *ctx, int scale)
{
    uint64_t sb = toks_scratch_bytes(ctx, 400000u, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch");
    uint32_t n_items = 300u * (uint32_t)scale;
    toks_par_item *it = (toks_par_item *)calloc(n_items, sizeof *it);
    uint8_t **txt = (uint8_t **)calloc(n_items, sizeof *txt);
    ref *rf = (ref *)calloc(n_items, sizeof *rf);
    uint32_t **buf = (uint32_t **)calloc(n_items, sizeof *buf);
    uint64_t *cap = (uint64_t *)calloc(n_items, 8u);
    uint32_t pools[] = { 1u, 4u, 0u };
    uint64_t wide = 0;
    for (int round = 0; round < 6; round++) {
        uint32_t flags = FLAGS[round % 6];
        for (uint32_t i = 0; i < n_items; i++) {
            uint64_t r = rnd() % 30u;          /* 1 in 30 big enough for the batch to split it at its cuts */
            uint64_t want = r == 0u ? 100000u + rnd() % 300000u : r < 4u ? rnd() % 60000u : rnd() % 3000u;
            txt[i] = (uint8_t *)malloc(want + 1u);
            uint64_t len = gen(txt[i], want, rnd() % 10u == 0u ? 1 : 0);
            rf[i] = serial(ctx, scr, txt[i], len, flags);
            uint64_t c[5] = { 0u, rf[i].n / 2u, rf[i].n, rf[i].n + 9u, len + 80u };
            cap[i] = c[rnd() % 5u];
            buf[i] = (uint32_t *)malloc(4u * (cap[i] + 64u));
            it[i].text = txt[i];
            it[i].len = len;
            it[i].cap = cap[i];
            it[i].out = cap[i] != 0u ? buf[i] : NULL;
            it[i].n = 12345;
            if (rnd() % 50u == 0u) { it[i].text = NULL; it[i].len = 5u; }              /* TOKS_E_ARG */
            if (rnd() % 50u == 0u) { it[i].len = (1ull << 29) + 1u; }                  /* TOKS_E_LIMIT */
        }
        toks_par *p = pool(ctx, pools[round % 3], (round & 1u) ? TOKS_SCRATCH_CACHE_MIB(4) : 0u, round < 3);
        for (int rep = 0; rep < 2 && p != NULL; rep++) {
            for (uint32_t i = 0; i < n_items; i++) {
                for (uint64_t k = 0; k < cap[i] + 64u; k++) { buf[i][k] = CANARY; }
                it[i].n = 12345;
            }
            uint64_t from = rep == 0 ? 0u : n_items / 3u;            /* the second pass: a sub-array */
            CHECK(toks_par_encode_batch(p, it + from, n_items - from, flags) == 0, "batch");
            toks_par_info in = info(p);
            CHECK(in.last >= 1u && in.last <= in.threads, "batch took %u of %u", in.last, in.threads);
            wide += in.last > 1u;
            for (uint32_t i = (uint32_t)from; i < n_items; i++) {
                int64_t want;
                if (it[i].text == NULL && it[i].len != 0u) { want = TOKS_E_ARG; }
                else if (it[i].len > (1ull << 29)) { want = TOKS_E_LIMIT; }
                else { want = (int64_t)rf[i].n; }
                int same = it[i].n == want;
                uint64_t m = want <= 0 ? 0u : ((uint64_t)want < cap[i] ? (uint64_t)want : cap[i]);
                for (uint64_t k = 0; same && want >= 0 && k < m; k++) { same = buf[i][k] == rf[i].ids[k]; }
                for (uint64_t k = cap[i]; same && k < cap[i] + 64u; k++) { same = buf[i][k] == CANARY; }
                CHECK(same, "%s batch item %u: n %" PRId64 " want %" PRId64 " cap %" PRIu64, label, i, it[i].n, want, cap[i]);
            }
        }
        toks_par_destroy(p);
        for (uint32_t i = 0; i < n_items; i++) { free(txt[i]); free(rf[i].ids); free(buf[i]); }
    }
    printf("  %-14s toks_par_encode_batch: 6 rounds x %u items x 2 passes equal serial, %" PRIu64 " passes parallel\n",
           label, n_items, wide);
    free(it); free(txt); free(rf); free(buf); free(cap); free(scr);
}

/* ================================================================ shared pool, policy, errors */

typedef struct shared { toks_par *p; const uint8_t *t; uint64_t len; ref *r; int bad; } shared;

#if !defined(_WIN32)
static void *caller(void *arg)
{
    shared *s = (shared *)arg;
    uint32_t *o = (uint32_t *)malloc(4u * (s->r->n + 8u));
    for (int k = 0; k < 6; k++) {
        int64_t n = toks_par_encode(s->p, s->t, s->len, 0u, o, s->r->n + 8u);
        if (n != (int64_t)s->r->n || memcmp(o, s->r->ids, (size_t)s->r->n * 4u) != 0) { s->bad++; }
    }
    free(o);
    return NULL;
}
#endif

static void shared_pool(toks_ctx *ctx)
{
    uint64_t sb = toks_scratch_bytes(ctx, 400000u, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch");
    uint8_t *t = (uint8_t *)malloc(400000u);
    uint64_t len = gen(t, 400000u, 0);
    ref r = serial(ctx, scr, t, len, 0u);
    toks_par *p = pool(ctx, 4u, 0u, 1);
#if !defined(_WIN32)
    shared a = { p, t, len, &r, 0 }, b = { p, t, len, &r, 0 };
    pthread_t ta, tb;
    int ra = pthread_create(&ta, NULL, caller, &a);
    int rb = pthread_create(&tb, NULL, caller, &b);
    CHECK(ra == 0 && rb == 0, "threads");
    if (ra == 0) { pthread_join(ta, NULL); }
    if (rb == 0) { pthread_join(tb, NULL); }
    CHECK(a.bad == 0 && b.bad == 0, "two callers on one pool: %d / %d bad", a.bad, b.bad);
#endif
    uint32_t *o = (uint32_t *)malloc(4u * (r.n + 8u));
    for (int k = 0; k < 3; k++) {             /* parked workers wake for each call */
        sleep_ms(20u);
        CHECK(toks_par_encode(p, t, len, 0u, o, r.n + 8u) == (int64_t)r.n && memcmp(o, r.ids, (size_t)r.n * 4u) == 0,
              "after a sleep");
        toks_par_info e = info(p);
        CHECK(e.last == e.threads, "an eager pool of %u on 400 KB took %u", e.threads, e.last);
    }
    toks_par_item one = { t, len, o, r.n + 8u, 0 };
    CHECK(toks_par_encode(NULL, t, len, 0u, o, 8u) == TOKS_E_ARG, "pool NULL");
    CHECK(toks_par_encode(p, t, len, 3u, o, 8u) == TOKS_E_ARG, "mode 3");
    CHECK(toks_par_encode(p, t, len, 64u, o, 8u) == TOKS_E_ARG, "unknown flag");
    CHECK(toks_par_encode(p, NULL, len, 0u, o, 8u) == TOKS_E_ARG, "text NULL");
    CHECK(toks_par_encode(p, t, len, 0u, NULL, 8u) == TOKS_E_ARG, "out NULL");
    CHECK(toks_par_encode(p, t, (1ull << 29) + 1u, 0u, o, 8u) == TOKS_E_LIMIT, "limit");
    CHECK(toks_par_encode_batch(NULL, &one, 1u, 0u) == TOKS_E_ARG, "batch pool NULL");
    CHECK(toks_par_encode_batch(p, NULL, 1u, 0u) == TOKS_E_ARG, "batch items NULL");
    CHECK(toks_par_encode_batch(p, &one, 1u, 3u) == TOKS_E_ARG, "batch mode 3");
    {   /* a refused batch touches no item (toks.h): n and out as they were, for mode 3, unknown bits and a NULL pool */
        static const uint32_t BAD[] = { 3u, 16u, 64u, 1u << 31 };
        for (size_t b = 0; b <= sizeof BAD / sizeof BAD[0]; b++) {   /* bound: 5 calls */
            for (uint32_t k = 0; k < 8u; k++) { o[k] = 0xC0DE0000u + k; }
            toks_par_item two[2] = { { t, len, o, 4u, -12345 }, { t, 7u, o + 4, 4u, -54321 } };
            int64_t rb = b < sizeof BAD / sizeof BAD[0] ? toks_par_encode_batch(p, two, 2u, BAD[b])
                                                         : toks_par_encode_batch(NULL, two, 2u, 0u);
            int same = two[0].n == -12345 && two[1].n == -54321;
            for (uint32_t k = 0; k < 8u; k++) { same &= o[k] == 0xC0DE0000u + k; }
            CHECK(rb == TOKS_E_ARG && same, "batch %s 0x%x: %" PRId64 ", items untouched %d",
                  b < sizeof BAD / sizeof BAD[0] ? "flags" : "pool NULL", b < sizeof BAD / sizeof BAD[0] ? BAD[b] : 0u, rb, same);
        }
    }
    CHECK(toks_par_encode_batch(p, NULL, 0u, 0u) == 0, "batch empty");
    CHECK(toks_par_encode_batch(p, &one, 1u, 0u) == 0 && one.n == (int64_t)r.n, "batch of one");
    toks_par_info in;
    memset(&in, 0, sizeof in);
    in.size = sizeof in - 1u;
    CHECK(toks_par_get_info(p, &in) == TOKS_E_ARG, "get_info short size");
    in.size = sizeof in;
    CHECK(toks_par_get_info(NULL, &in) == TOKS_E_ARG && toks_par_get_info(p, NULL) == TOKS_E_ARG, "get_info NULL");
    toks_par_destroy(p);

    /* the policy, as the pool decides: under 16 KiB nobody else; the default pool is a couple; a big call
     * spreads only after the model measured it is worth it, and never past the cap */
    p = pool(ctx, 0u, 0u, 0);
    in = info(p);
    CHECK(in.threads >= 1u && in.threads <= 4u && in.fast >= 1u, "default pool: %u threads, %u fast", in.threads, in.fast);
    CHECK(in.ns_per_mib > 0u && in.wake_ns > 0u && in.join_ns > 0u, "a model before the first call");
    CHECK(in.threads == 1u ? in.min_bytes == 0u : in.min_bytes >= (16u << 10), "min_bytes %" PRIu64, in.min_bytes);
    for (int k = 0; k < 4; k++) {
        CHECK(toks_par_encode(p, t, 16000u, 0u, o, r.n + 8u) >= 0 && info(p).last == 1u, "16000 B took %u", info(p).last);
        CHECK(toks_par_encode(p, t, len, 0u, o, r.n + 8u) == (int64_t)r.n && info(p).last <= in.threads, "400 KB");
    }
    toks_par_destroy(p);
    toks_par *q = (toks_par *)1;
    CHECK(toks_par_create(NULL, ctx, 2u, 0u) == TOKS_E_ARG, "create out NULL");
    CHECK(toks_par_create(&q, NULL, 2u, 0u) == TOKS_E_ARG && q == NULL, "create ctx NULL");
    CHECK(toks_par_create(&q, ctx, 5000u, 0u) == TOKS_E_ARG && q == NULL, "create too many");
    CHECK(toks_par_create(&q, ctx, 2u, TOKS_SCRATCH_CACHE_MIB(3)) == TOKS_E_ARG && q == NULL, "create cache 3 MiB");
    q = (toks_par *)1;
    CHECK(toks_par_create(&q, ctx, 2u, 1u << 21) == TOKS_E_ARG && q == NULL, "create scratch flag bit 21");
    q = (toks_par *)1;
    CHECK(toks_par_create(&q, ctx, 2u, 1u << 31) == TOKS_E_ARG && q == NULL, "create scratch flag bit 31");
    toks_par_destroy(NULL);
    static const uint32_t NT[] = { 1u, 2u, 3u, 8u };   /* "at most n_threads participants, the caller included" */
    for (size_t k = 0; k < sizeof NT / sizeof NT[0]; k++) {
        q = pool(ctx, NT[k], 0u, 0);
        if (q == NULL) { continue; }
        toks_par_info qi = info(q);
        CHECK(qi.threads >= 1u && qi.threads <= NT[k], "a pool of at most %u: %u threads", NT[k], qi.threads);
        CHECK(toks_par_encode(q, t, len, 0u, o, r.n + 8u) == (int64_t)r.n && info(q).last <= qi.threads,
              "a pool of at most %u: the call took %u", NT[k], info(q).last);
        toks_par_destroy(q);
    }
    for (int k = 0; k < 20; k++) {            /* create / destroy cycles, some with one call */
        q = pool(ctx, 1u + (uint32_t)(k % 7), 0u, k & 1);
        if (k % 3 == 0) { CHECK(toks_par_encode(q, t, 50000u, 0u, NULL, 0u) >= 0, "cycle call"); }
        toks_par_destroy(q);
    }
    free(o); free(r.ids); free(t); free(scr);
}

int main(int argc, char **argv)
{
    int scale = argc > 1 ? atoi(argv[1]) : 1;
    if (scale < 1) { scale = 1; }
    static const char *const FIX[] = {
        "tests/data/compile/llama3style.json", "tests/data/compile/gpt2style.json",
        "tests/data/compile/qwen35style.json", "tests/data/compile/nosplit.json",
        "tests/data/spm/gemma4like.json", "tests/data/spm/mistrallike.json", "tests/data/spm/meta_always_split.json",
        "tests/data/spm/llamalike.json",
    };
    for (size_t k = 0; k < sizeof FIX / sizeof FIX[0]; k++) {
        toks_ctx *c = load(FIX[k]);
        if (c == NULL) { continue; }
        const char *label = strrchr(FIX[k], '/') + 1;
        docs(label, c, scale, 0);
        if (k < 2 || k == 4) { batches(label, c, scale); }
        if (k == 0) { shared_pool(c); }
        toks_unload(c);
    }
    static const char *const REAL[] = { "gpt2", "llama3", "glm53", "qwen38", "gemma4" };
    for (size_t k = 0; k < sizeof REAL / sizeof REAL[0]; k++) {
        toks_ctx *c = load(REAL[k]);
        if (c == NULL) { continue; }
        docs(REAL[k], c, scale, 1);
        if (scale > 1 || k == 1) { batches(REAL[k], c, scale); }
        toks_unload(c);
    }
    printf("test_par: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
