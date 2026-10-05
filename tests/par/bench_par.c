/*
 * bench_par.c: toks_par through the public abi (include/toks.h): one big input and batches of documents,
 * serial toks_encode against pools of k participants, every result checked equal to the serial ids outside the
 * timer (a mismatch exits 1). Not part of make test: run it on a lab host, pinned, and record the load.
 *
 *   clang -std=c17 -O2 -Iinclude tests/par/bench_par.c build/<os>-<isa>/libtoks.a -pthread -o build/bench_par
 *   taskset -c <cpus> build/bench_par <tokenizer> <text> doc   <MiB,..> <k,..> [reps]
 *   taskset -c <cpus> build/bench_par <tokenizer> <text> batch <doc bytes> <MiB,..> <k,..> [reps]
 *   taskset -c <cpus> build/bench_par <tokenizer> <text> sweep <k> [reps] [gap us] [same|fresh]
 *
 *   doc    S MiB of text as one input (0.25 = 256 KiB), toks_par_encode on a pool of n = k (k = 0: serial
 *          toks_encode). A size the file does not hold prints skip=text_short.
 *   batch  S MiB cut into documents of about <doc bytes> (after the first '\n' at or past each boundary, as
 *          tools/bench/e2e.c cuts chunks), toks_par_encode_batch on a pool of n = k (k = 0: a serial loop of
 *          toks_encode on one scratch)
 *   sweep  one input of 4 KiB .. 4 MiB: serial, the pool's own choice (n = k) and every participant forced
 *          (env TOKS_PAR_EAGER=1, n = k and n = 2): where parallel starts to win on this host. gap: the caller
 *          sleeps that long before every timed call (sporadic calls find the workers asleep; 0: back to back).
 *          same: every call encodes the file's first bytes (repeated text: one core's caches hold it all, a
 *          split's participants each only their units; the worst case for going wide); fresh: every call new
 *          text, as a server sees it (each variant its own windows, interleaved over the file: no variant meets
 *          text another one warmed; compare medians)
 *
 * windows: the file holds K = len(file) / S disjoint windows of S bytes, [j S, (j + 1) S).
 * states per cell (MB/s of input; ids checked in every state, every rep):
 *   first  a fresh pool (created outside the timer), its first call, on window 0: scratches, staging, the threads'
 *          first touches included (serial: a fresh scratch's first call)
 *   pass   reps samples of new text on a warm pool, the serving state: sample i is a fresh pool (scratch) that
 *          encodes window i + 1 mod K untimed, then window i mod K timed: every timed byte is new to its pool and
 *          consecutive samples meet different windows. pass_ms / pass_mbps = the median sample. A file of fewer
 *          than two windows (K < 2) warms on its last S bytes instead, which overlap the timed text: pass_seen =
 *          the overlap's share of it (> 0: the row is HALF-WARM, its pass is no pass)
 *   warm   the last sample's text again on its pool: best and median of reps
 * every serial reference is computed before the first timed call (no timed call follows a reference encode).
 * Output lines start with PAR and carry key=value pairs; fnv_nopp hashes the ids without the post-processor (one
 * flat stream for a batch) of window 0, pass_fnv_nopp the same of every pass sample's window, in order, as
 * tools/bench/gigatoken/par.rs prints them (the comparator's ids are checked on every timed text). The caller's out
 * arrays are touched before any timer (they are the caller's: toks writes ids into memory it is given).
 * Built against a toks.h with TOKS_PAR_HAS_INFO, a pool's lines also report toks_par_get_info (participants
 * used, the measured model).
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

#define FNV0 0xCBF29CE484222325ull

static uint64_t fnv_more(uint64_t h, const uint32_t *ids, uint64_t n)   /* fnv-1a-64 over ids, continued */
{
    for (uint64_t i = 0; i < n; i++) { h = (h ^ ids[i]) * 0x100000001B3ull; }
    return h;
}

static uint64_t fnv(const uint32_t *ids, uint64_t n) { return fnv_more(FNV0, ids, n); }

static int parse_list(const char *s, double *v, int max)
{
    int n = 0;
    while (*s && n < max) {
        char *e;
        v[n++] = strtod(s, &e);
        if (e == s) { break; }
        s = (*e == ',') ? e + 1 : e;
    }
    return n;
}

static const char *base_name(const char *p)
{
    const char *b = strrchr(p, '/');
    return (b != NULL && b[1] != 0) ? b + 1 : p;
}

static toks_ctx *ctx;
static const uint8_t *text;
static uint64_t text_len;

/* the pass state's windows (the header): K, sample i's timed text and its warm-up text, and the share of the timed
 * text the warm-up held (> 0 only when K < 2: HALF-WARM) */
static uint64_t windows(uint64_t len) { return len != 0u ? text_len / len : 0u; }
static const uint8_t *pass_text(uint64_t len, int i)
{
    uint64_t k = windows(len);
    return k >= 2u ? text + (uint64_t)i % k * len : text;
}
static const uint8_t *pass_warm(uint64_t len, int i)
{
    uint64_t k = windows(len);
    return k >= 2u ? text + ((uint64_t)i + 1u) % k * len : text + text_len - len;
}
static double seen(uint64_t len)
{
    return windows(len) >= 2u ? 0.0 : (double)(len * 2u - text_len) / (double)len;
}
static const char *tok_name;
static int reps = 7;

#define MAX_REPS 64

static double median(const double *v, int n)       /* of v[0, n) (unchanged) */
{
    double s[MAX_REPS];
    memcpy(s, v, (size_t)n * sizeof s[0]);
    qsort(s, (size_t)n, sizeof s[0], cmpd);
    return s[n / 2];
}

static void reps_key(const char *key, const double *v, int n)   /* " key=v0,v1,..." in ms */
{
    printf(" %s=", key);
    for (int i = 0; i < n; i++) { printf("%s%.3f", i ? "," : "", v[i] * 1e3); }
}

static void info_line(toks_par *p)
{
#if defined(TOKS_PAR_HAS_INFO)
    toks_par_info in;
    memset(&in, 0, sizeof in);
    in.size = sizeof in;
    if (toks_par_get_info(p, &in) == 0) {
        printf(" used=%u threads=%u fast=%u ns_per_mib=%" PRIu64 " wake_ns=%" PRIu64 " join_ns=%" PRIu64
               " min_bytes=%" PRIu64, in.last, in.threads, in.fast, in.ns_per_mib, in.wake_ns, in.join_ns, in.min_bytes);
    }
#else
    (void)p;
#endif
}

static void setenv_eager(int on)
{
#if defined(_WIN32)
    _putenv(on ? "TOKS_PAR_EAGER=1" : "TOKS_PAR_EAGER=");
#else
    if (on) { setenv("TOKS_PAR_EAGER", "1", 1); } else { unsetenv("TOKS_PAR_EAGER"); }
#endif
}

static toks_par *pool(uint32_t k)
{
    toks_par *p = NULL;
    if (toks_par_create(&p, ctx, k, 0u) != 0) { fprintf(stderr, "toks_par_create %u failed\n", k); exit(2); }
    return p;
}

/* ---- one big input --------------------------------------------------------------------------------------- */

typedef struct ref { const uint8_t *t; int64_t n; uint64_t h, h_nopp; } ref;   /* a text's serial ids: count + fnv */

/* the reference of every text a cell times (window 0 and each pass sample's window), all before any timed call */
static int doc_refs(ref *r, uint64_t len, int nr, uint32_t *buf, uint64_t cap, void *scr)
{
    int m = 0;
    for (int i = -1; i < nr; i++) {                   /* -1: window 0 (first) */
        const uint8_t *t = i < 0 ? text : pass_text(len, i);
        int j = 0;
        while (j < m && r[j].t != t) { j++; }
        if (j < m) { continue; }
        int64_t n = toks_encode(ctx, t, len, TOKS_NO_POSTPROCESS, buf, cap, scr);
        r[m].h_nopp = fnv(buf, n > 0 ? (uint64_t)n : 0u);   /* the comparators' ids (no post-processor) */
        r[m].n = toks_encode(ctx, t, len, 0u, buf, cap, scr);
        r[m].h = fnv(buf, r[m].n > 0 ? (uint64_t)r[m].n : 0u);
        r[m++].t = t;
    }
    return m;
}

static const ref *ref_of(const ref *r, int m, const uint8_t *t)
{
    for (int j = 0; j < m; j++) { if (r[j].t == t) { return &r[j]; } }
    fprintf(stderr, "no reference\n");
    exit(2);
}

static void same(const char *what, uint32_t k, int64_t n, const uint32_t *out, const ref *r)
{
    if (n != r->n || fnv(out, n > 0 ? (uint64_t)n : 0u) != r->h) {
        fprintf(stderr, "k %u %s: par != serial (n %" PRId64 " want %" PRId64 ")\n", k, what, n, r->n);
        exit(1);
    }
}

static double doc_once(toks_par *p, void *scr, const uint8_t *t, uint64_t len, uint32_t *out, uint64_t cap, int64_t *n)
{
    memset(out, 0, 256);
    double t0 = now_s();
    *n = p != NULL ? toks_par_encode(p, t, len, 0u, out, cap) : toks_encode(ctx, t, len, 0u, out, cap, scr);
    return now_s() - t0;
}

static void doc_cell(double mib, uint32_t k)
{
    uint64_t len = (uint64_t)(mib * 1048576.0);
    if (len == 0u || len > text_len) {
        printf("PAR mode=doc tok=%s bytes=%" PRIu64 " k=%u skip=text_short text_bytes=%" PRIu64 "\n", tok_name, len, k,
               text_len);
        fflush(stdout);
        return;
    }
    uint64_t cap = len + 64u;
    uint32_t *out = (uint32_t *)malloc((size_t)cap * 4u), *buf = (uint32_t *)malloc((size_t)cap * 4u);
    uint64_t sb = toks_scratch_bytes(ctx, len, 0u);
    void *scr = malloc((size_t)sb);
    if (out == NULL || buf == NULL || scr == NULL || toks_scratch_init(ctx, scr, sb, 0u) != 0) { fprintf(stderr, "alloc\n"); exit(2); }
    memset(out, 0, (size_t)cap * 4u);
    memset(buf, 0, (size_t)cap * 4u);
    int nr = reps < MAX_REPS ? reps : MAX_REPS;
    ref r[MAX_REPS + 1];
    int m = doc_refs(r, len, nr, buf, cap, scr);
    double first, ps[MAX_REPS], w[MAX_REPS];
    int64_t n;
    toks_par *p = NULL;
    if (k == 0u) {
        toks_scratch_init(ctx, scr, sb, 0u);              /* first: a fresh scratch's first call */
    } else {
        p = pool(k);
    }
    first = doc_once(p, scr, text, len, out, cap, &n);
    same("first", k, n, out, &r[0]);
    for (int i = 0; i < nr; i++) {                    /* pass: a fresh pool (scratch) warmed on other text */
        if (k == 0u) {
            toks_scratch_init(ctx, scr, sb, 0u);
        } else {
            toks_par_destroy(p);
            p = pool(k);
        }
        doc_once(p, scr, pass_warm(len, i), len, out, cap, &n);
        ps[i] = doc_once(p, scr, pass_text(len, i), len, out, cap, &n);
        same("pass", k, n, out, ref_of(r, m, pass_text(len, i)));
    }
    const ref *rw = ref_of(r, m, pass_text(len, nr - 1));
    for (int i = 0; i < nr; i++) {                    /* warm: the last sample's text again */
        w[i] = doc_once(p, scr, rw->t, len, out, cap, &n);
        same("warm", k, n, out, rw);
    }
    double pm = median(ps, nr), wm = median(w, nr);
    qsort(w, (size_t)nr, sizeof w[0], cmpd);
    printf("PAR mode=doc tok=%s bytes=%" PRIu64 " k=%u ids=%" PRId64 " fnv=%016" PRIx64 " fnv_nopp=%016" PRIx64
           " first_ms=%.3f pass_ms=%.3f warm_best_ms=%.3f warm_med_ms=%.3f first_mbps=%.1f pass_mbps=%.1f"
           " warm_mbps=%.1f pass_seen=%.2f windows=%" PRIu64,
           tok_name, len, k, r[0].n, r[0].h, r[0].h_nopp, first * 1e3, pm * 1e3, w[0] * 1e3, wm * 1e3,
           (double)len / first / 1e6, (double)len / pm / 1e6, (double)len / w[0] / 1e6, seen(len), windows(len));
    reps_key("pass_reps_ms", ps, nr);
    printf(" pass_fnv_nopp=");
    for (int i = 0; i < nr; i++) { printf("%s%016" PRIx64, i ? "," : "", ref_of(r, m, pass_text(len, i))->h_nopp); }
    if (p != NULL) { info_line(p); }
    printf("\n");
    fflush(stdout);
    toks_par_destroy(p);
    free(out); free(buf); free(scr);
}

/* ---- batches of documents -------------------------------------------------------------------------------- */

typedef struct docs {
    const uint8_t *t;                    /* the text they cut, len bytes */
    toks_par_item *it;
    uint64_t n;
    uint32_t *buf;
    int64_t *want;                       /* the serial results: count and fnv per document, the flat nopp fnv */
    uint64_t *h, maxl, ids, h_nopp;
} docs;

static docs cut_docs(const uint8_t *t, uint64_t len, uint64_t doc)
{
    docs d;
    memset(&d, 0, sizeof d);
    d.t = t;
    uint64_t cap_items = len / (doc > 0u ? doc : 1u) + 2u, ids = 0;
    d.it = (toks_par_item *)calloc((size_t)cap_items, sizeof *d.it);
    for (uint64_t p = 0; p < len && d.n < cap_items;) {
        uint64_t e = len;
        if (p + doc < len) {
            e = p + doc;
            while (e < len && t[e - 1u] != '\n') { e++; }
        }
        d.it[d.n].text = t + p;
        d.it[d.n].len = e - p;
        if (e - p > d.maxl) { d.maxl = e - p; }
        ids += e - p + 64u;
        d.n++;
        p = e;
    }
    d.buf = (uint32_t *)malloc((size_t)ids * 4u);
    d.want = (int64_t *)calloc((size_t)d.n, sizeof *d.want);
    d.h = (uint64_t *)calloc((size_t)d.n, sizeof *d.h);
    if (d.it == NULL || d.buf == NULL || d.want == NULL || d.h == NULL) { fprintf(stderr, "alloc\n"); exit(2); }
    memset(d.buf, 0, (size_t)ids * 4u);
    uint64_t o = 0;
    for (uint64_t i = 0; i < d.n; i++) {
        d.it[i].out = d.buf + o;
        d.it[i].cap = d.it[i].len + 64u;
        o += d.it[i].len + 64u;
    }
    return d;
}

static void docs_free(docs *d) { free(d->it); free(d->buf); free(d->want); free(d->h); }

static double batch_once(toks_par *p, void *scr, docs *d)
{
    for (uint64_t i = 0; i < d->n; i++) { d->it[i].n = -12345; }
    double t0 = now_s();
    if (p == NULL) {
        for (uint64_t i = 0; i < d->n; i++) {
            d->it[i].n = toks_encode(ctx, d->it[i].text, d->it[i].len, 0u, d->it[i].out, d->it[i].cap, scr);
        }
    } else if (toks_par_encode_batch(p, d->it, d->n, 0u) != 0) {
        fprintf(stderr, "toks_par_encode_batch failed\n");
        exit(1);
    }
    return now_s() - t0;
}

/* the serial results of d (the reference: no timed call follows it) */
static void batch_ref(docs *d, void *scr)
{
    d->h_nopp = FNV0;
    for (uint64_t i = 0; i < d->n; i++) {                 /* the comparators' ids (no post-processor), flat */
        int64_t m = toks_encode(ctx, d->it[i].text, d->it[i].len, TOKS_NO_POSTPROCESS, d->it[i].out, d->it[i].cap, scr);
        d->h_nopp = fnv_more(d->h_nopp, d->it[i].out, m > 0 ? (uint64_t)m : 0u);
    }
    batch_once(NULL, scr, d);
    for (uint64_t i = 0; i < d->n; i++) {
        d->want[i] = d->it[i].n;
        d->h[i] = d->it[i].n >= 0 ? fnv(d->it[i].out, (uint64_t)d->it[i].n) : 0u;
        d->ids += d->it[i].n >= 0 ? (uint64_t)d->it[i].n : 0u;
    }
}

static void batch_check(docs *d, const char *what, uint32_t k)
{
    for (uint64_t i = 0; i < d->n; i++) {
        if (d->it[i].n != d->want[i] || fnv(d->it[i].out, (uint64_t)d->want[i]) != d->h[i]) {
            fprintf(stderr, "k %u %s: document %" PRIu64 " par != serial\n", k, what, i);
            exit(1);
        }
    }
}

static docs *docs_of(docs *ws, int m, const uint8_t *t)
{
    for (int j = 0; j < m; j++) { if (ws[j].t == t) { return &ws[j]; } }
    fprintf(stderr, "no documents\n");
    exit(2);
}

static void batch_cell(uint64_t doc, double mib, uint32_t k)
{
    uint64_t len = (uint64_t)(mib * 1048576.0);
    if (len == 0u || len > text_len) {
        printf("PAR mode=batch tok=%s doc=%" PRIu64 " bytes=%" PRIu64 " k=%u skip=text_short text_bytes=%" PRIu64 "\n",
               tok_name, doc, len, k, text_len);
        fflush(stdout);
        return;
    }
    int nr = reps < MAX_REPS ? reps : MAX_REPS, m = 0;
    docs ws[2 * MAX_REPS + 1];                        /* every text the cell encodes, cut once */
    uint64_t maxl = 0;
    for (int i = -1; i < nr; i++) {                   /* -1: window 0 (first) */
        const uint8_t *ts[2] = { i < 0 ? text : pass_text(len, i), i < 0 ? text : pass_warm(len, i) };
        for (int s = 0; s < 2; s++) {
            int j = 0;
            while (j < m && ws[j].t != ts[s]) { j++; }
            if (j < m) { continue; }
            ws[m] = cut_docs(ts[s], len, doc);
            if (ws[m].maxl > maxl) { maxl = ws[m].maxl; }
            m++;
        }
    }
    uint64_t sb = toks_scratch_bytes(ctx, maxl, 0u);
    void *scr = malloc((size_t)sb);
    if (scr == NULL || toks_scratch_init(ctx, scr, sb, 0u) != 0) { fprintf(stderr, "alloc\n"); exit(2); }
    for (int j = 0; j < m; j++) { batch_ref(&ws[j], scr); }
    docs *d0 = docs_of(ws, m, text);
    double first, ps[MAX_REPS], w[MAX_REPS];
    toks_par *p = NULL;
    if (k == 0u) { toks_scratch_init(ctx, scr, sb, 0u); } else { p = pool(k); }
    first = batch_once(p, scr, d0);
    batch_check(d0, "first", k);
    for (int i = 0; i < nr; i++) {                    /* pass: a fresh pool (scratch) warmed on other text */
        if (k == 0u) {
            toks_scratch_init(ctx, scr, sb, 0u);
        } else {
            toks_par_destroy(p);
            p = pool(k);
        }
        docs *d = docs_of(ws, m, pass_text(len, i));
        batch_once(p, scr, docs_of(ws, m, pass_warm(len, i)));
        ps[i] = batch_once(p, scr, d);
        batch_check(d, "pass", k);
    }
    docs *dw = docs_of(ws, m, pass_text(len, nr - 1));
    for (int i = 0; i < nr; i++) {                    /* warm: the last sample's documents again */
        w[i] = batch_once(p, scr, dw);
        batch_check(dw, "warm", k);
    }
    double pm = median(ps, nr), wm = median(w, nr);
    qsort(w, (size_t)nr, sizeof w[0], cmpd);
    printf("PAR mode=batch tok=%s doc=%" PRIu64 " docs=%" PRIu64 " bytes=%" PRIu64 " k=%u ids=%" PRIu64
           " fnv_nopp=%016" PRIx64 " first_ms=%.3f pass_ms=%.3f warm_best_ms=%.3f warm_med_ms=%.3f first_mbps=%.1f"
           " pass_mbps=%.1f warm_mbps=%.1f pass_seen=%.2f windows=%" PRIu64,
           tok_name, doc, d0->n, len, k, d0->ids, d0->h_nopp, first * 1e3, pm * 1e3, w[0] * 1e3, wm * 1e3,
           (double)len / first / 1e6, (double)len / pm / 1e6, (double)len / w[0] / 1e6, seen(len), windows(len));
    reps_key("pass_reps_ms", ps, nr);
    printf(" pass_fnv_nopp=");
    for (int i = 0; i < nr; i++) { printf("%s%016" PRIx64, i ? "," : "", docs_of(ws, m, pass_text(len, i))->h_nopp); }
    if (p != NULL) { info_line(p); }
    printf("\n");
    fflush(stdout);
    toks_par_destroy(p);
    for (int j = 0; j < m; j++) { docs_free(&ws[j]); }
    free(scr);
}

/* ---- where parallel starts to win -------------------------------------------------------------------------- */

static unsigned gap_us;                  /* sweep: idle time before every timed call (0: back to back) */
static int fresh;                        /* sweep: every timed call on new text (else the same text every time) */

static void idle(void)
{
    if (gap_us == 0u) { return; }
    struct timespec ts = { (time_t)(gap_us / 1000000u), (long)(gap_us % 1000000u) * 1000L };
    nanosleep(&ts, NULL);
}

/* the sweep's text for variant v's rep i (i = -1: the warm-up) of a len-byte call: the file's start, or with
 * fresh a window of its own: variant v takes windows 1 + 4 i + v (each variant meets its text first, and the
 * four variants' windows interleave over the file), the warm-up window 0 */
static const uint8_t *sweep_text(uint64_t len, int v, int i)
{
    if (!fresh) { return text; }
    uint64_t room = text_len - len, w = i < 0 ? 0u : 1u + 4u * (uint64_t)i + (uint64_t)v;
    return text + (room == 0u ? 0u : (w * len) % room);
}

typedef struct href { int64_t n; uint64_t h; } href;   /* a reference: the count and the ids' fnv */

static double best_of(toks_par *p, void *scr, int v, uint64_t len, uint32_t *out, uint64_t cap, const href *r,
                      double *med)
{
    double w[MAX_REPS];
    int nr = reps < MAX_REPS ? reps : MAX_REPS;
    for (int i = 0; i < nr; i++) {
        int64_t n;
        idle();
        w[i] = doc_once(p, scr, sweep_text(len, v, i), len, out, cap, &n);
        const href *x = &r[fresh ? 4 * i + v : 0];
        if (n != x->n || fnv(out, n > 0 ? (uint64_t)n : 0u) != x->h) {
            fprintf(stderr, "sweep variant %d rep %d: par != serial\n", v, i);
            exit(1);
        }
    }
    qsort(w, (size_t)nr, sizeof w[0], cmpd);
    *med = w[nr / 2];
    return w[0];
}

static void sweep(uint32_t k)
{
    uint64_t maxlen = 4u << 20;
    if (maxlen > text_len) { maxlen = text_len; }
    uint64_t cap = maxlen + 64u;
    int nr = reps < MAX_REPS ? reps : MAX_REPS, nref = fresh ? 4 * nr : 1;
    uint32_t *out = (uint32_t *)malloc((size_t)cap * 4u), *rid = (uint32_t *)malloc((size_t)cap * 4u);
    href r[4 * MAX_REPS];
    uint64_t sb = toks_scratch_bytes(ctx, maxlen, 0u);
    void *scr = malloc((size_t)sb), *rscr = malloc((size_t)sb);   /* timed serial calls; the references */
    if (out == NULL || rid == NULL || scr == NULL || rscr == NULL || toks_scratch_init(ctx, scr, sb, 0u) != 0 ||
        toks_scratch_init(ctx, rscr, sb, 0u) != 0) {
        fprintf(stderr, "alloc\n");
        exit(2);
    }
    memset(out, 0, (size_t)cap * 4u);
    memset(rid, 0, (size_t)cap * 4u);
    setenv_eager(0);
    toks_par *a = pool(k);
    setenv_eager(1);
    toks_par *e = pool(k), *e2 = pool(2u);
    setenv_eager(0);
    for (uint64_t len = 4096u; len <= maxlen; len *= 2u) {
        int64_t n;
        for (int j = 0; j < nref; j++) {                    /* outside every timer; j = 4 i + v */
            r[j].n = toks_encode(ctx, sweep_text(len, j % 4, j / 4), len, 0u, rid, cap, rscr);
            r[j].h = fnv(rid, r[j].n > 0 ? (uint64_t)r[j].n : 0u);
        }
        doc_once(NULL, scr, sweep_text(len, 0, -1), len, out, cap, &n);   /* every variant meets the warm-up */
        doc_once(a, NULL, sweep_text(len, 1, -1), len, out, cap, &n);     /* text before its timer */
        doc_once(e, NULL, sweep_text(len, 2, -1), len, out, cap, &n);
        doc_once(e2, NULL, sweep_text(len, 3, -1), len, out, cap, &n);
        double ms, ma, me, me2;
        double bs = best_of(NULL, scr, 0, len, out, cap, r, &ms);
        double ba = best_of(a, NULL, 1, len, out, cap, r, &ma);
        printf("PAR mode=sweep tok=%s bytes=%" PRIu64 " k=%u gap_us=%u fresh=%d serial_us=%.1f auto_us=%.1f auto_x=%.2f",
               tok_name, len, k, gap_us, fresh, bs * 1e6, ba * 1e6, bs / ba);
        info_line(a);
        double be = best_of(e, NULL, 2, len, out, cap, r, &me);
        double be2 = best_of(e2, NULL, 3, len, out, cap, r, &me2);
        printf(" eager_us=%.1f eager_x=%.2f eager2_us=%.1f eager2_x=%.2f med_serial_us=%.1f med_auto_us=%.1f"
               " med_eager_us=%.1f med_eager2_us=%.1f\n",
               be * 1e6, bs / be, be2 * 1e6, bs / be2, ms * 1e6, ma * 1e6, me * 1e6, me2 * 1e6);
        fflush(stdout);
    }
    toks_par_destroy(a);
    toks_par_destroy(e);
    toks_par_destroy(e2);
    free(out); free(rid); free(scr); free(rscr);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: bench_par <tokenizer> <text> doc <MiB,..> <k,..> [reps]\n"
                        "       bench_par <tokenizer> <text> batch <doc bytes> <MiB,..> <k,..> [reps]\n"
                        "       bench_par <tokenizer> <text> sweep <k> [reps] [gap us] [same|fresh]\n");
        return 2;
    }
    if (toks_load(&ctx, argv[1], NULL) != 0) { fprintf(stderr, "load %s failed\n", argv[1]); return 2; }
    tok_name = base_name(argv[1]);
    FILE *f = fopen(argv[2], "rb");
    if (f == NULL) { fprintf(stderr, "open %s failed\n", argv[2]); return 2; }
    fseek(f, 0, SEEK_END);
    text_len = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)text_len + 1u);
    if (b == NULL || fread(b, 1, (size_t)text_len, f) != text_len) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);
    text = b;
    const char *mode = argv[3];
    double sz[32], ks[32];
    if (strcmp(mode, "doc") == 0 && argc >= 6) {
        int ns = parse_list(argv[4], sz, 32), nk = parse_list(argv[5], ks, 32);
        if (argc > 6) { reps = atoi(argv[6]) < 1 ? 1 : atoi(argv[6]); }
        for (int i = 0; i < ns; i++) {
            for (int j = 0; j < nk; j++) { doc_cell(sz[i], (uint32_t)ks[j]); }
        }
    } else if (strcmp(mode, "batch") == 0 && argc >= 7) {
        uint64_t doc = (uint64_t)atoll(argv[4]);
        int ns = parse_list(argv[5], sz, 32), nk = parse_list(argv[6], ks, 32);
        if (argc > 7) { reps = atoi(argv[7]) < 1 ? 1 : atoi(argv[7]); }
        for (int i = 0; i < ns; i++) {
            for (int j = 0; j < nk; j++) { batch_cell(doc, sz[i], (uint32_t)ks[j]); }
        }
    } else if (strcmp(mode, "sweep") == 0) {
        if (argc > 5) { reps = atoi(argv[5]) < 1 ? 1 : atoi(argv[5]); }
        if (argc > 6) { gap_us = (unsigned)atoi(argv[6]); }
        if (argc > 7) { fresh = strcmp(argv[7], "fresh") == 0; }
        sweep((uint32_t)atoi(argv[4]));
    } else {
        fprintf(stderr, "bad mode or arguments\n");
        return 2;
    }
    free(b);
    toks_unload(ctx);
    return 0;
}
