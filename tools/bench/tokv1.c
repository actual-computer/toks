/* tools/bench/tokv1.c: toks vs tok v1, e's incumbent tokenizer (host/tok.asm on x86-64, host/a64/tok.S on
 * arm64, packed by e's tools/tokpack), in one process on one pinned core: SPEC §12.6's incumbent comparator
 * and T8's "incumbent" row. Built and run by tools/bench/tokv1.sh only: an e checkout (E_DIR) is an optional
 * bench input like gigatoken, never a build or test dependency of toks. Linux only (tok v1 is elf / gnu as).
 *
 *   tokv1 <tokenizer.json> <tok.bin> <chunk_bytes|0|replay> <reps> <file>...
 *
 * the call: toks_encode(flags 0) vs tok_encode(TOK_ADDED_ALL), hf's default encode on both (added tokens
 * recognized, post-processor applied; qwen38 and glm53 add no template ids).
 * chunks: as tools/bench/e2e.c: the files concatenated, cut after the first '\n' at or past each chunk_bytes
 * boundary (0 = the whole text in one call). replay: one conversation of 24 turns (user and assistant text cut
 * from the files in order, 1-4 KiB each, qwen's chat markup) encoded as its 24 growing prompts, in order, on
 * one scratch (SPEC §12.7's C3 shape, the affinity case): tok v1's segment memo answers the earlier turns.
 * sides: tokv1 (2 MiB piece cache + WORDS + 4 MiB segment memo), toks (the 2 MiB piece cache, memo off:
 * TOKS_SCRATCH_MEMO_MIB(0), as flags 0 was before the memo default), toks32 (TOKS_SCRATCH_CACHE_MIB(32), memo off:
 * what a long-lived worker scratch gets) and toksm (TOKS_SCRATCH_MEMO_MIB(4): toks's 4 MiB segment memo, SPEC §6,
 * the default since 0.3.0 and what e's workers get; TOKV1_MEMO_MIB=m: m MiB); every scratch 2 MiB-aligned.
 * states, each self-contained per side; the sides run A B C in even reps, C B A in odd ones (paired):
 *   cold  the scratch initialized before EVERY call, outside the timer (SPEC §12.3 cold)
 *   pass  initialized once, then every chunk in order (one worker scratch over a stream of documents)
 *   lang  TOKV1_WARM_ON="<file> ...": initialized, one untimed pass over those files (same chunking), then the
 *         timed pass: e's "new prompt in a warm worker" (e docs/host/tok.md §7: warmed on other text only)
 *   warm  initialized, one untimed pass over this text, then the timed pass (tok v1 and toksm: the memo answers
 *         every segment it recorded: a replay, reported, not a piece-cache comparison)
 * memo: tok v1's piece-cache counters (scratch header hits + misses) count the pieces it encoded; memo_frac =
 *   1 - pieces(state) / pieces(cold) is the share its segment memo answered. cold is 0 by construction.
 * exact, outside every timer: every chunk's ids from every side, fresh and warm, compared id for id against
 *   tokv1's; one difference voids the cell ("EXACT NO", exit 1).
 * stats: per side and state the best and the median over reps; per state the paired speedup tokv1 time / toks
 *   time of each rep, its median and a 95% bootstrap interval of that median (> 1: toks is faster). */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"
#include "tok.h"   /* e's tests/host/tok.h: the C mirror of tok v1's api */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void toks_sha256(const uint8_t *data, uint64_t len, uint8_t out[32]);   /* src/core/alloc.c (internal) */

#define NSIDE 4
#define NSTATE 4
enum { COLD, PASS, LANG, WARM };
static const char *const STATE[NSTATE] = { "cold", "pass", "lang", "warm" };
static const char *const SIDE[NSIDE] = { "tokv1", "toks", "toks32", "toksm" };

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* the core clock: 8 dependent adds per iteration (1-cycle latency on every core we run on) */
static double ghz(void)
{
    uint64_t x = 0, iters = 25000000, a = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
#if defined(__aarch64__)
        __asm__ volatile("add %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\t"
                         "add %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1\n\tadd %0, %0, #1" : "+r"(x));
#else
        __asm__ volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
                         "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
#endif
    }
    return (double)(8 * iters) / (double)(now_ns() - a);
}

static void die(const char *m)
{
    fprintf(stderr, "tokv1: %s\n", m);
    exit(1);
}

typedef struct text {                      /* calls: [beg, beg + len) of buf */
    uint8_t *buf;
    uint64_t n, nc, maxlen;
    uint64_t *beg, *len;
} text;

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
        if (buf == NULL || fread(buf + n, 1, (size_t)m, f) != (size_t)m) { die("read"); }
        fclose(f);
        n += (uint64_t)m;
    }
    *n_out = n;
    return buf;
}

static void add_call(text *t, uint64_t b, uint64_t l)
{
    t->beg = realloc(t->beg, (t->nc + 1) * sizeof(uint64_t));
    t->len = realloc(t->len, (t->nc + 1) * sizeof(uint64_t));
    if (t->beg == NULL || t->len == NULL) { die("malloc"); }
    t->beg[t->nc] = b;
    t->len[t->nc] = l;
    t->nc++;
    if (l > t->maxlen) { t->maxlen = l; }
}

/* the end of a cut of at least `want` bytes from p: after the first '\n' at or past p + want (or n) */
static uint64_t cut_end(const uint8_t *s, uint64_t n, uint64_t p, uint64_t want)
{
    if (want == 0 || p + want >= n) { return n; }
    uint64_t e = p + want;
    while (e < n && s[e - 1] != '\n') { e++; }
    return e;
}

static void cut(text *t, uint64_t chunk)
{
    for (uint64_t p = 0; p < t->n;) {
        uint64_t e = cut_end(t->buf, t->n, p, chunk);
        add_call(t, p, e - p);
        p = e;
    }
}

/* replay: prompt k = turns 0..k-1 + "<|im_start|>user\n" U_k "<|im_end|>\n<|im_start|>assistant\n", turn j =
 * "<|im_start|>user\n" U_j "<|im_end|>\n<|im_start|>assistant\n" A_j "<|im_end|>\n"; U_j, A_j the next 1-4 KiB of
 * src cut after a '\n'. The 24 prompts are stored back to back, one call each. */
#define TURNS 24
static void replay_build(text *t, const uint8_t *src, uint64_t n)
{
    static const char US[] = "<|im_start|>user\n", MID[] = "<|im_end|>\n<|im_start|>assistant\n",
                      END[] = "<|im_end|>\n";
    uint64_t ub[TURNS], ul[TURNS], ab[TURNS], al[TURNS], p = 0, conv = 0;
    for (int j = 0; j < TURNS; j++) {
        uint64_t e = cut_end(src, n, p, 1024u << (j % 3));
        ub[j] = p, ul[j] = e - p, p = e;
        e = cut_end(src, n, p, 4096u >> (j % 3));
        ab[j] = p, al[j] = e - p, p = e;
        if (al[j] == 0 || ul[j] == 0) { die("replay: the files are too short for 24 turns"); }
    }
    uint64_t cap = 0;
    for (int k = 0; k < TURNS; k++) { cap += 2 * (p + 64 * TURNS); }
    t->buf = malloc(cap);
    if (t->buf == NULL) { die("malloc"); }
    uint8_t *conv_buf = malloc(p + 64 * TURNS);       /* the conversation so far (complete turns) */
    if (conv_buf == NULL) { die("malloc"); }
    uint64_t at = 0;
#define PUT(dst, off, s, l) (memcpy((dst) + (off), (s), (l)), (off) += (l))
    for (int k = 0; k < TURNS; k++) {
        uint64_t b = at;
        PUT(t->buf, at, conv_buf, conv);
        PUT(t->buf, at, US, sizeof US - 1);
        PUT(t->buf, at, src + ub[k], ul[k]);
        PUT(t->buf, at, MID, sizeof MID - 1);
        add_call(t, b, at - b);
        PUT(conv_buf, conv, US, sizeof US - 1);         /* turn k joins the conversation */
        PUT(conv_buf, conv, src + ub[k], ul[k]);
        PUT(conv_buf, conv, MID, sizeof MID - 1);
        PUT(conv_buf, conv, src + ab[k], al[k]);
        PUT(conv_buf, conv, END, sizeof END - 1);
    }
#undef PUT
    t->n = at;
    free(conv_buf);
}

typedef struct side {
    int v1;                                /* 1: tok v1 */
    uint32_t flags;                        /* toks scratch flags */
    uint64_t sb;
    void *scr;
} side;

static tok_ctx VCTX;
static toks_ctx *TCTX;
static uint32_t *IDS;                      /* one output buffer for every timed call */
static uint64_t IDCAP;

static void init(const side *s)
{
    int64_t r = s->v1 ? tok_scratch_init(s->scr, s->sb) : toks_scratch_init(TCTX, s->scr, s->sb, s->flags);
    if (r != 0) { die("scratch init"); }
}

static inline int64_t enc(const side *s, const text *t, uint64_t k, uint32_t *out, uint64_t cap)
{
    const uint8_t *p = t->buf + t->beg[k];
    int64_t m = s->v1 ? tok_encode(&VCTX, p, t->len[k], out, cap, s->scr, TOK_ADDED_ALL)
                      : toks_encode(TCTX, p, t->len[k], 0, out, cap, s->scr);
    if (m < 0 || (uint64_t)m > cap) { fprintf(stderr, "tokv1: %s encode %lld at call %llu\n", s->v1 ? "tok v1" : "toks",
                                              (long long)m, (unsigned long long)k); exit(1); }
    return m;
}

static uint64_t pieces(const side *s)      /* tok v1's piece-cache counters (hits + misses) */
{
    const tok_scratch_hdr *h = s->scr;
    return s->v1 ? h->hits + h->misses : 0;
}

static uint64_t run(const side *s, const text *t)   /* one timed pass, the scratch as it is */
{
    uint64_t a = now_ns();
    for (uint64_t k = 0; k < t->nc; k++) { (void)enc(s, t, k, IDS, IDCAP); }
    return now_ns() - a;
}

/* one state on one side: its time; *pc the tok v1 pieces encoded in the timed part */
static uint64_t state(const side *s, int st, const text *t, const text *w, uint64_t *pc)
{
    uint64_t d = 0, p0;
    if (st == COLD) {
        *pc = 0;
        for (uint64_t k = 0; k < t->nc; k++) {
            init(s);
            p0 = pieces(s);
            uint64_t a = now_ns();
            (void)enc(s, t, k, IDS, IDCAP);
            d += now_ns() - a;
            *pc += pieces(s) - p0;
        }
        return d;
    }
    init(s);
    if (st == LANG) { (void)run(s, w); }
    if (st == WARM) { (void)run(s, t); }
    p0 = pieces(s);
    d = run(s, t);
    *pc = pieces(s) - p0;
    return d;
}

/* every call's ids into all[], from the scratch as it is; the total */
static uint64_t ids_all(const side *s, const text *t, uint32_t *all, uint64_t cap, uint64_t *cnt)
{
    uint64_t tot = 0;
    for (uint64_t k = 0; k < t->nc; k++) {
        int64_t m = enc(s, t, k, all + tot, cap - tot);
        cnt[k] = (uint64_t)m;
        tot += (uint64_t)m;
    }
    return tot;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median_d(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* the median of r[0..n) and a 95% percentile-bootstrap interval of it (2000 resamples, fixed seed) */
static void boot(const double *r, int n, double *med, double *lo, double *hi)
{
    double *v = malloc((size_t)n * sizeof(double)), m[2000];
    uint64_t x = 0x9E3779B97F4A7C15ull;
    if (v == NULL) { die("malloc"); }
    memcpy(v, r, (size_t)n * sizeof(double));
    *med = median_d(v, n);
    for (int b = 0; b < 2000; b++) {
        for (int i = 0; i < n; i++) {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            v[i] = r[x % (uint64_t)n];
        }
        m[b] = median_d(v, n);
    }
    qsort(m, 2000, sizeof(double), cmp_dbl);
    *lo = m[49];
    *hi = m[1950];
    free(v);
}

int main(int argc, char **argv)
{
    if (argc < 6) { die("usage: tokv1 <tokenizer.json> <tok.bin> <chunk_bytes|0|replay> <reps> <file>..."); }
    int replay = strcmp(argv[3], "replay") == 0, reps = atoi(argv[4]);
    uint64_t chunk = replay ? 4096 : strtoull(argv[3], NULL, 10);
    if (reps < 1 || reps > 1000) { die("reps"); }
    text c, w;
    memset(&c, 0, sizeof c);
    memset(&w, 0, sizeof w);
    uint64_t n;
    uint8_t *src = read_all(argv + 5, argc - 5, &n);
    if (replay) {
        replay_build(&c, src, n);
    } else {
        c.buf = src, c.n = n;
        cut(&c, chunk);
    }
    const char *wo = getenv("TOKV1_WARM_ON");
    if (wo != NULL && wo[0] != 0) {
        char *list = strdup(wo), *files[256];
        int nf = 0;
        for (char *f = strtok(list, " "); f != NULL && nf < 256; f = strtok(NULL, " ")) { files[nf++] = f; }
        w.buf = read_all(files, nf, &w.n);
        cut(&w, chunk);
    }
    uint64_t maxlen = c.maxlen > w.maxlen ? c.maxlen : w.maxlen;

    uint64_t t0 = now_ns();
    if (toks_load(&TCTX, argv[1], NULL) != 0) { die("toks_load"); }
    double load_toks = (double)(now_ns() - t0) * 1e-6;
    t0 = now_ns();
    if (tok_load(&VCTX, argv[2]) != 0) { die("tok_load"); }
    double load_v1 = (double)(now_ns() - t0) * 1e-6;
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    (void)toks_get_info(TCTX, &info);

    const char *mm = getenv("TOKV1_MEMO_MIB");          /* toksm's memo size (default 4 MiB, tok v1's) */
    side s[NSIDE] = { { 1, 0, 0, NULL }, { 0, TOKS_SCRATCH_MEMO_MIB(0), 0, NULL },          /* toks / toks32: memo off */
                      { 0, TOKS_SCRATCH_CACHE_MIB(32) | TOKS_SCRATCH_MEMO_MIB(0), 0, NULL },
                      { 0, TOKS_SCRATCH_MEMO_MIB(mm != NULL && mm[0] != 0 ? strtoul(mm, NULL, 10) : 4u), 0, NULL } };
    for (int i = 0; i < NSIDE; i++) {
        s[i].sb = s[i].v1 ? tok_scratch_bytes(maxlen) : toks_scratch_bytes(TCTX, maxlen, s[i].flags);
        uint64_t al = (s[i].sb + (2u << 20) - 1) & ~(uint64_t)((2u << 20) - 1);
        if (posix_memalign(&s[i].scr, 2u << 20, al) != 0) { die("scratch alloc"); }
    }
    IDCAP = 3 * maxlen + 64;                            /* tok v1's bound: n <= 3 len (NFC growth) */
    IDS = malloc(IDCAP * sizeof(uint32_t));

    /* exact: every side's id stream, fresh and then warm, == tok v1's fresh one */
    uint64_t cap = 3 * c.n + 64 * c.nc + 64, *cnt0 = malloc(c.nc * 8), *cnt = malloc(c.nc * 8);
    uint32_t *ref = malloc(cap * 4), *got = malloc(cap * 4);
    if (IDS == NULL || cnt0 == NULL || cnt == NULL || ref == NULL || got == NULL) { die("malloc"); }
    init(&s[0]);
    uint64_t tot = ids_all(&s[0], &c, ref, cap, cnt0), bad = 0;
    for (int i = 0; i < NSIDE; i++) {
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 0) { init(&s[i]); }
            uint64_t t = ids_all(&s[i], &c, got, cap, cnt);
            if (t != tot || memcmp(got, ref, tot * 4) != 0) {
                uint64_t a = 0, k = 0;
                while (k < c.nc && cnt[k] == cnt0[k] && memcmp(got + a, ref + a, cnt[k] * 4) == 0) { a += cnt[k++]; }
                fprintf(stderr, "tokv1: %s (%s) differs from tok v1 at call %llu of %llu (ids %llu vs %llu)\n", SIDE[i],
                        pass ? "warm" : "fresh", (unsigned long long)k, (unsigned long long)c.nc,
                        (unsigned long long)(k < c.nc ? cnt[k] : 0), (unsigned long long)(k < c.nc ? cnt0[k] : 0));
                bad++;
            }
        }
    }
    uint8_t sha[32];
    char hex[65];
    toks_sha256((const uint8_t *)ref, tot * 4, sha);
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", sha[i]); }
    printf("TOKV1 mode=%s bytes=%llu calls=%llu maxlen=%llu ids=%llu sha=%.16s exact=%s tier=%u warm_on_bytes=%llu"
           " load_ms_toks=%.1f load_ms_tokv1=%.1f scratch_mib=%.1f/%.1f/%.1f/%.1f reps=%d\n",
           replay ? "replay" : argv[3], (unsigned long long)c.n, (unsigned long long)c.nc, (unsigned long long)maxlen,
           (unsigned long long)tot, hex, bad || tot == 0 ? "NO" : "yes", info.tier,   /* no ids compared: not exact */ (unsigned long long)w.n, load_toks, load_v1,
           (double)s[0].sb / 1048576.0, (double)s[1].sb / 1048576.0, (double)s[2].sb / 1048576.0,
           (double)s[3].sb / 1048576.0, reps);
    if (bad) { printf("EXACT NO: cell void\n"); return 1; }

    /* timing: per rep, per state, the sides in alternating order */
    uint64_t *tm = malloc(sizeof(uint64_t) * NSTATE * NSIDE * (size_t)reps), pc[NSTATE] = { 0 };
    if (tm == NULL) { die("malloc"); }
#define TM(st, i, r) tm[((st) * NSIDE + (i)) * (size_t)reps + (r)]
    double g0 = ghz();
    for (int r = 0; r < reps; r++) {
        for (int st = 0; st < NSTATE; st++) {
            if (st == LANG && w.nc == 0) { continue; }
            for (int j = 0; j < NSIDE; j++) {
                int i = r % 2 ? NSIDE - 1 - j : j;
                uint64_t p;
                TM(st, i, r) = state(&s[i], st, &c, &w, &p);
                if (i == 0) { pc[st] = p; }
            }
        }
    }
    double g1 = ghz();
    printf("CLOCK ghz0=%.3f ghz1=%.3f\n", g0, g1);
    for (int st = 0; st < NSTATE; st++) {
        if (st == LANG && w.nc == 0) { continue; }
        double memo = pc[COLD] ? 1.0 - (double)pc[st] / (double)pc[COLD] : 0.0;
        double v1med = 0;
        for (int i = 0; i < NSIDE; i++) {
            uint64_t *v = malloc(sizeof(uint64_t) * (size_t)reps);
            for (int r = 0; r < reps; r++) { v[r] = TM(st, i, r); }
            qsort(v, (size_t)reps, sizeof(uint64_t), cmp_u64);
            double best = (double)v[0] * 1e-9, med = (double)v[reps / 2] * 1e-9;
            if (i == 0) { v1med = med; }
            printf("TV1 state=%s side=%s best_s=%.6f med_s=%.6f mbs_best=%.1f mbs_med=%.1f", STATE[st], SIDE[i], best, med,
                   (double)c.n / best * 1e-6, (double)c.n / med * 1e-6);
            if (i == 0) {
                printf(" pieces=%llu memo_frac=%.3f\n", (unsigned long long)pc[st], memo);
            } else {
                double *ratio = malloc(sizeof(double) * (size_t)reps), m, lo, hi;
                for (int r = 0; r < reps; r++) { ratio[r] = (double)TM(st, 0, r) / (double)TM(st, i, r); }
                boot(ratio, reps, &m, &lo, &hi);
                printf(" x_med=%.3f x_lo=%.3f x_hi=%.3f x_of_medians=%.3f\n", m, lo, hi, v1med / med);
                free(ratio);
            }
            free(v);
        }
    }
    tok_unload(&VCTX);
    toks_unload(TCTX);
    return 0;
}
