/*
 * test_stall.c: the stall regression of docs/hardening.md (SPEC §7.3, T7): adversarial input classes, each
 * aimed at a path (whitespace / strip-token floods for the added-token walk, one giant piece for K6's long path
 * and spm's whole-segment words, '<' floods for K1, combining marks for the normalizers, random CJK / bytes),
 * timed through toks.h. A screen times each class at 8, 32 and 128 KiB, the best of 3 cold calls per size (the
 * scratch initialized outside the timer). A class whose last screen step grows x8 or more for 4x the bytes is a
 * growth suspect, and one over its path's bound a bound suspect; the screen alone fails nothing. A suspect is measured
 * carefully, and the verdict rests on that: 5 rounds over its sizes, each round one batch per size, each batch the
 * mean of as many cold calls as make >= 5 ms of encoding, the best batch per size (a neighbour that comes and goes
 * hits every size of a round, not one size's five batches). A class fails when
 *   - it grows superlinearly: x8 or more for 4x the bytes from 128 to 512 KiB, then x12 or more from 512 KiB to
 *     2 MiB, measured only after the first step read x8. That is a quadratic's curve: a n + b n^2 whose first step
 *     reads x8 (b n = a / 2 at 128 KiB) reads x12 at the second, and x16 as n grows. Linear is x4; K6's long-piece
 *     heap, whose working set grows 32 B a byte, read up to x11.6 then x8.8 on a CI arm64 runner, and linear classes
 *     up to x8.3 then x8.7 on a contended x86 one (docs/hardening.md §2: every class forced through both steps). The
 *     steps start at 128 KiB, where a call takes milliseconds. A step of x8 or more into a call of over 8 us a byte
 *     ends the walk: a stall without the next sizes (no class costs a tenth of that on any runner measured);
 *   - it costs more than its path's bound x the pseudo-en text's ns per byte at 128 KiB (pinned tokenizers only;
 *     each bound is ~3x the worst class measured, docs/hardening.md §2: it catches a new stall, not noise).
 * What the growth rule gives up: a quadratic term under half the linear cost at 128 KiB, a power law between n^1.5
 * and n^1.79 (x8 to x12 per 4x at both steps, which the heap imitates on that runner), and a stall bounded by a
 * block (docs/hardening.md §2). The fixtures always run; a pinned tokenizer missing from $TOKS_TOKENIZER_CACHE
 * (~/.cache/toks/tokenizers) is a SKIP. Every suspect prints its screen and careful times.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <time.h>
#endif

static uint64_t now_ns(void)
{
#if defined(_WIN32)
    LARGE_INTEGER c, f;              /* the UCRT has no clock_gettime */
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    uint64_t n = (uint64_t)c.QuadPart, hz = (uint64_t)f.QuadPart;
    return n / hz * 1000000000ull + n % hz * 1000000000ull / hz;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
}

static uint64_t rng(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t utf8(uint32_t cp, uint8_t *o)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) { o[0] = (uint8_t)(0xC0u | (cp >> 6)); o[1] = (uint8_t)(0x80u | (cp & 0x3Fu)); return 2; }
    o[0] = (uint8_t)(0xE0u | (cp >> 12)); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 3;
}

enum { EN, FIX, LET, DIG, CJK, BYTES, MARKS, CUT };
typedef struct cls { const char *name; int kind; const char *unit; } cls;
static const cls CLS[] = {
    { "en", EN, "" },              { "ws_space", FIX, " " },  { "ws_nl", FIX, "\n" },   { "ws_tab", FIX, "\t" },
    { "ws_ideo", FIX, "\xe3\x80\x80" }, { "ws_sp_nl", FIX, " \n" }, { "lt", FIX, "<" }, { "punct", FIX, "!" },
    { "letters", FIX, "a" },       { "letters_rand", LET, "" }, { "digits_rand", DIG, "" }, { "cjk_rand", CJK, "" },
    { "bytes_rand", BYTES, "" },   { "marks", MARKS, "" },    { "meta", FIX, "\xe2\x96\x81" }, { "added_cut", CUT, "" },
};
#define NCLS (sizeof CLS / sizeof CLS[0])

/* pseudo-en: the reference text. A cold call over real English hits K5's tables on most words and misses on the
 * rare tail, so tiling one paragraph (every word repeated within the call) would make en look far too fast: words
 * are drawn Zipf-like from common English words, one in eight a fresh random word (the rare tail), with sentence
 * punctuation, capitals and line breaks */
static const char *const WORDS[] = {
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it", "with", "as", "his", "on", "be", "at",
    "by", "I", "this", "had", "not", "are", "but", "from", "or", "have", "an", "they", "which", "one", "you", "were",
    "her", "all", "she", "there", "would", "their", "we", "him", "been", "has", "when", "who", "will", "more", "no",
    "if", "out", "so", "said", "what", "up", "its", "about", "into", "than", "them", "can", "only", "other", "new",
    "some", "could", "time", "these", "two", "may", "then", "do", "first", "any", "my", "now", "such", "like", "our",
    "over", "man", "me", "even", "most", "made", "after", "also", "did", "many", "before", "must", "through", "back",
    "years", "where", "much", "your", "way", "well", "down", "should", "because", "each", "just", "those", "people",
    "how", "too", "little", "state", "good", "very", "make", "world", "still", "own", "see", "men", "work", "long",
    "get", "here", "between", "both", "life", "being", "under", "never", "day", "same", "another", "know", "while",
    "last", "might", "us", "great", "old", "year", "off", "come", "since", "against", "go", "came", "right", "used",
    "take", "three", "himself", "few", "house", "use", "during", "without", "again", "place", "around", "however",
    "home", "small", "found", "thought", "went", "say", "part", "once", "general", "high", "upon", "school", "every",
    "don't", "does", "got", "united", "left", "number", "course", "war", "until", "always", "away", "something",
    "fact", "though", "water", "less", "public", "put", "think", "almost", "hand", "enough", "far", "took", "head",
};

static uint64_t gen_en(uint8_t *b, uint64_t n, uint64_t *s)
{
    uint64_t i = 0, col = 0;
    int cap = 1;
    const uint64_t nw = sizeof WORDS / sizeof WORDS[0];
    while (i + 24u <= n) {
        char w[24];
        uint64_t k;
        if (rng(s) % 8u == 0u) {                                         /* the rare tail */
            k = 3u + rng(s) % 9u;
            for (uint64_t j = 0; j < k; j++) { w[j] = (char)("etaoinshrdlucmfwypvbgk"[rng(s) % 22u]); }
        } else {                                                         /* Zipf-like: rank ~ nw^u */
            uint64_t r = rng(s) % 1000u, idx = (r * r * r / 1000000u) * nw / 1000u;
            k = strlen(WORDS[idx]);
            memcpy(w, WORDS[idx], k);
        }
        if (cap && w[0] >= 'a' && w[0] <= 'z') { w[0] = (char)(w[0] - 32); }
        cap = 0;
        memcpy(b + i, w, k);
        i += k;
        col += k;
        uint64_t p = rng(s) % 20u;
        if (p == 0u) { b[i++] = '.'; cap = 1; }
        else if (p == 1u) { b[i++] = ','; }
        if (col > 72u) { b[i++] = '\n'; col = 0; } else { b[i++] = ' '; col++; }
    }
    return i;
}

/* the markup tokens of a context (<...> / [...]), for added_cut */
typedef struct lits { const uint8_t *p[64]; uint64_t n[64]; uint32_t k; } lits;

static void gen(const cls *c, const lits *l, uint8_t *b, uint64_t n)
{
    uint64_t s = 0x5EEDull + n, i = 0;
    uint8_t u[4];
    while (i < n) {
        uint64_t k = 0;
        switch (c->kind) {
        case EN: i += gen_en(b + i, n - i, &s); k = n - i; break;
        case FIX: k = strlen(c->unit); break;
        case LET: u[0] = (uint8_t)('a' + rng(&s) % 26u); k = 1; break;
        case DIG: u[0] = (uint8_t)('0' + rng(&s) % 10u); k = 1; break;
        case CJK: k = utf8(0x4E00u + (uint32_t)(rng(&s) % 0x5200u), u); break;
        case BYTES: u[0] = (uint8_t)rng(&s); k = 1; break;
        case MARKS: k = i == 0 ? utf8('a', u) : utf8(0x0301u, u); break;
        default: {                                                      /* CUT */
            if (l->k == 0u) { u[0] = '<'; k = 1; break; }
            uint32_t j = (uint32_t)(rng(&s) % l->k);
            k = 1u + rng(&s) % (l->n[j] - 1u);
            if (i + k > n) { k = n - i; }
            memcpy(b + i, l->p[j], k);
            i += k;
            continue;
        }
        }
        if (i + k > n || c->kind == EN) { break; }
        if (c->kind == FIX) { memcpy(b + i, c->unit, k); }
        else { memcpy(b + i, u, k); }
        i += k;
    }
    while (i < n) { b[i++] = 'a'; }
}

static int failures, checks, remeasured;

/* one cold call (the scratch initialized outside the timer): ns, or -1 */
static double cold_ns(const toks_ctx *ctx, const uint8_t *x, uint64_t n, void *scr, uint64_t sb, uint32_t *out)
{
    if (toks_scratch_init(ctx, scr, sb, 0u) != 0) { return -1.0; }
    uint64_t t0 = now_ns();
    int64_t m = toks_encode(ctx, x, n, 0u, out, 3u * n + 64u, scr);
    double t = (double)(now_ns() - t0);
    return m < 0 ? -1.0 : t;
}

/* ns per cold call, the best of 3 (the screen) */
static double screen_ns(const toks_ctx *ctx, const uint8_t *x, uint64_t n, void *scr, uint64_t sb, uint32_t *out)
{
    double best = 1e30;
    for (int i = 0; i < 3; i++) {                           /* bound: 3 */
        double t = cold_ns(ctx, x, n, scr, sb, out);
        if (t < 0.0) { return -1.0; }
        if (t < best) { best = t; }
    }
    return best;
}

/* the sizes: the screen's 8 / 32 / 128 KiB (0-2), the verdict's 128 / 512 / 2048 KiB (2-4). A size runs in the scratch
 * of its group (up to 128 KiB, 512 KiB, 2 MiB), allocated when first needed */
#define NSZ 5
static const uint64_t SZ[NSZ] = { 8192u, 32768u, 131072u, 524288u, 2097152u };
typedef struct scrs { void *p[3]; uint64_t n[3]; } scrs;
static void *scr_for(const toks_ctx *ctx, scrs *s, uint64_t n, uint64_t *sb)
{
    int k = n <= SZ[2] ? 0 : n <= SZ[3] ? 1 : 2;
    if (s->p[k] == NULL) {
        s->n[k] = toks_scratch_bytes(ctx, SZ[2 + k], 0u);
        s->p[k] = malloc((size_t)s->n[k]);
    }
    *sb = s->n[k];
    return s->p[k];
}

/* the screen: best of 3 cold calls at 8, 32 and 128 KiB into t (0: not run), the steps into g. 128 KiB is not run
 * after a step of x8 or more into a 32 KiB call of over 0.1 s (3 us a byte: the verdict takes it from 128 KiB). 0, or
 * -1 when an encode failed */
static int screen(const cls *c, const lits *l, const toks_ctx *ctx, uint8_t *x, scrs *s, uint32_t *out, double t[3],
                  double g[2])
{
    uint64_t sb = 0;
    t[0] = t[1] = t[2] = 0.0;
    g[0] = g[1] = 0.0;
    for (int i = 0; i < 3; i++) {                           /* bound: 3 */
        gen(c, l, x, SZ[i]);
        void *scr = scr_for(ctx, s, SZ[i], &sb);
        if (scr == NULL || (t[i] = screen_ns(ctx, x, SZ[i], scr, sb, out)) <= 0.0) { return -1; }
        if (i > 0) { g[i - 1] = t[i] / t[i - 1]; }
        if (i == 1 && g[0] >= 8.0 && t[1] > 1e8) { break; }
    }
    return 0;
}

/* the measurement a verdict rests on, over n sizes sz[0, n), into t (0: not run): 5 rounds, each round one batch per
 * size (the mean of r cold calls, r so a batch times >= 5 ms), the best batch per size; a neighbour that comes and
 * goes hits every size of a round, not one size's batches. A size whose first call takes over 5 s is that one call
 * (its rounds would take minutes). A size before the last whose first call costs over 8 us a byte is that one call
 * and ends the list: the sizes after it are not run (no class costs a tenth of that on any runner measured,
 * docs/hardening.md §2; a quadratic's next call would cost 16 times as much). 0, or -1 when an encode failed */
static int careful(const cls *c, const lits *l, const toks_ctx *ctx, uint8_t *x, scrs *s, uint32_t *out,
                   const uint64_t *sz, int n, double *t)
{
    uint32_t r[3];
    uint64_t sb = 0;
    for (int i = 0; i < n; i++) { t[i] = 0.0; }
    for (int i = 0; i < n; i++) {                           /* bound: 3; r from one call per size */
        gen(c, l, x, sz[i]);
        void *scr = scr_for(ctx, s, sz[i], &sb);
        double t1 = scr != NULL ? cold_ns(ctx, x, sz[i], scr, sb, out) : -1.0;
        if (t1 < 0.0) { return -1; }
        int cut = i < n - 1 && t1 > 8e3 * (double)sz[i];   /* ns: 8 us a byte */
        r[i] = t1 > 5e9 || cut ? 0u : t1 < 5e6 ? (uint32_t)(5e6 / (t1 > 1e3 ? t1 : 1e3)) + 1u : 1u;   /* <= 5,001 */
        t[i] = r[i] == 0u ? t1 : 1e30;
        if (cut) { n = i + 1; }
    }
    for (int round = 0; round < 5; round++) {               /* bound: 5 rounds x n sizes */
        for (int i = 0; i < n; i++) {
            if (r[i] == 0u) { continue; }
            gen(c, l, x, sz[i]);
            void *scr = scr_for(ctx, s, sz[i], &sb);
            double sum = 0.0;
            for (uint32_t k = 0; k < r[i]; k++) {           /* bound: r[i] <= 5,001 */
                double tk = cold_ns(ctx, x, sz[i], scr, sb, out);
                if (tk < 0.0) { return -1; }
                sum += tk;
            }
            if (sum / r[i] < t[i]) { t[i] = sum / r[i]; }
        }
    }
    return 0;
}

/* a growth suspect's verdict: 128 -> 512 KiB measured carefully, and 512 KiB -> 2 MiB only when that step read x8 or
 * more, each step in its own interleaved measurement; v the times at 128 and 512 KiB, then 512 KiB and 2 MiB (0: not
 * run), gv the two steps. A 2 MiB call of over 5 s is one call, outside the rounds, so its step is taken against the
 * best 512 KiB time of both measurements (a call alone is only ever slowed by noise; no linear class costs 2.4 us a
 * byte at 2 MiB on any runner measured). 1 when the first step read x8 or more and the second x12 or more (a n + b n^2 whose first
 * step reads x8 has b n = a / 2 at 128 KiB and 2a at 512 KiB, so its second step reads (4 + 32) / 3 = x12), or when a
 * call that a step of x8 or more led to ended the walk (careful: over 8 us a byte); 0 when not; -1 when an encode
 * failed */
static int growth(const cls *c, const lits *l, const toks_ctx *ctx, uint8_t *x, scrs *s, uint32_t *out, double v[4],
                  double gv[2])
{
    v[2] = v[3] = gv[0] = gv[1] = 0.0;
    if (careful(c, l, ctx, x, s, out, SZ + 2, 2, v) != 0) { return -1; }
    if (v[1] == 0.0) { return 1; }                          /* the 128 KiB call ended the walk */
    gv[0] = v[1] / v[0];
    if (gv[0] < 8.0) { return 0; }
    if (careful(c, l, ctx, x, s, out, SZ + 3, 2, v + 2) != 0) { return -1; }
    if (v[3] == 0.0) { return 1; }                          /* the 512 KiB call ended the walk */
    if (v[3] > 5e9 && v[1] < v[2]) { v[2] = v[1]; }         /* 2 MiB one call, not in the rounds: against the best */
    gv[1] = v[3] / v[2];                                    /* 512 KiB of both measurements */
    return gv[1] >= 12.0;
}

/* every class on one tokenizer; bound 0 = the growth check only */
static void run(const char *name, const char *path, double bound)
{
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, path, NULL) != 0) {
        printf("  %-18s SKIP (not loaded: %s)\n", name, path);
        return;
    }
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    toks_get_info(ctx, &info);
    lits l;
    memset(&l, 0, sizeof l);
    for (uint32_t id = 0; id < info.n_ids && l.k < 64u; id++) {
        uint64_t k = 0;
        const uint8_t *p = toks_token(ctx, id, &k);
        if (p != NULL && k >= 3u && k <= 64u && ((p[0] == '<' && p[k - 1] == '>') || (p[0] == '[' && p[k - 1] == ']'))) {
            l.p[l.k] = p;
            l.n[l.k++] = k;
        }
    }
    scrs s;
    memset(&s, 0, sizeof s);
    uint8_t *x = (uint8_t *)malloc(SZ[NSZ - 1]);
    uint32_t *out = (uint32_t *)malloc((3u * SZ[NSZ - 1] + 64u) * 4u);
    double en = 0.0, en_careful = 0.0, worst_g = 0.0, worst_x = 0.0;
    const char *wg = "-", *wx = "-";
    for (size_t c = 0; c < NCLS; c++) {
        double t[3], g[2], v[4] = { 0.0, 0.0, 0.0, 0.0 }, gv[2] = { 0.0, 0.0 };
        checks++;
        int bad = screen(&CLS[c], &l, ctx, x, &s, out, t, g) != 0, stall = 0;
        if (!bad) {
            double gs = t[2] > 0.0 ? g[1] : g[0];                   /* the screen's last step */
            double nsb = t[2] > 0.0 ? t[2] / (double)SZ[2] : t[1] / (double)SZ[1];
            if (CLS[c].kind == EN) { en = nsb; }
            if (gs > worst_g) { worst_g = gs; wg = CLS[c].name; }
            if (en > 0.0 && nsb / en > worst_x) { worst_x = nsb / en; wx = CLS[c].name; }
            int grow = gs >= 8.0, over = bound > 0.0 && en > 0.0 && nsb > bound * en;
            if (grow || over) {
                /* past the screen: the verdict rests on the careful measurement, growth from 128 KiB (growth()),
                 * the bound at 128 KiB against the careful en's */
                remeasured++;
                if (grow) { bad = (stall = growth(&CLS[c], &l, ctx, x, &s, out, v, gv)) < 0; }
                else { bad = careful(&CLS[c], &l, ctx, x, &s, out, SZ + 2, 1, v) != 0; }
                if (!bad && bound > 0.0 && en_careful == 0.0 && CLS[c].kind != EN) {
                    double te[1];
                    if (careful(&CLS[0], &l, ctx, x, &s, out, SZ + 2, 1, te) == 0) { en_careful = te[0] / (double)SZ[2]; }
                }
            }
            if (!bad && (grow || over)) {                           /* the receipt of every class past the screen */
                printf("  suspect %s %s: screen %.2f / %.2f / %.2f ms at 8 / 32 / 128 KiB (x%.1f, x%.1f); careful %.2f / "
                       "%.2f ms at 128 / 512 KiB (x%.1f), %.2f / %.2f ms at 512 / 2048 KiB (x%.1f) (0: not run)\n", name,
                       CLS[c].name, t[0] / 1e6, t[1] / 1e6, t[2] / 1e6, g[0], g[1], v[0] / 1e6, v[1] / 1e6, gv[0],
                       v[2] / 1e6, v[3] / 1e6, gv[1]);
                if (stall) {
                    if (gv[1] >= 12.0) {
                        printf("  FAIL %s %s: superlinear, x%.1f then x%.1f per 4x the bytes from 128 KiB to 512 KiB to "
                               "2 MiB\n", name, CLS[c].name, gv[0], gv[1]);
                    } else {
                        printf("  FAIL %s %s: superlinear, x%.1f per 4x the bytes into %.2f s at %s KiB, over 8 us a "
                               "byte: a stall without the next sizes\n", name, CLS[c].name, gv[0] > 0.0 ? gv[0] : gs,
                               (gv[0] > 0.0 ? v[2] : v[0]) / 1e9, gv[0] > 0.0 ? "512" : "128");
                    }
                    failures++;
                }
                nsb = v[0] / (double)SZ[2];
                if (over && en_careful > 0.0 && nsb > bound * en_careful) {
                    printf("  FAIL %s %s: %.1f ns/B = %.1fx en (%.1f ns/B) at 128 KiB, bound %.0fx\n", name, CLS[c].name,
                           nsb, nsb / en_careful, en_careful, bound);
                    failures++;
                }
            }
        }
        if (bad) {
            printf("  FAIL %s %s: encode failed\n", name, CLS[c].name);
            failures++;
        }
    }
    printf("  %-18s en %5.1f ns/B; worst growth x%.1f per 4x bytes (%s); worst %.1fx en (%s)%s\n", name, en, worst_g, wg,
           worst_x, wx, bound > 0.0 ? "" : " [growth only]");
    free(out);
    free(x);
    for (int k = 0; k < 3; k++) { free(s.p[k]); }
    toks_unload(ctx);
}

int main(void)
{
    char pin[1024];
    const char *dir = getenv("TOKS_TOKENIZER_CACHE"), *home = getenv("HOME");
    setvbuf(stdout, NULL, _IONBF, 0);       /* every line out at once, even if the job is killed (MSVC has no _IOLBF) */
    printf("test_stall: adversarial classes at 8 / 32 / 128 KiB; a suspect at 128 / 512 / 2048 KiB (docs/hardening.md)\n");
    static const struct { const char *name, *path; } FIX[] = {
        { "ws_lstrip", "tests/data/hardening/ws_lstrip.json" },     { "ws_rstrip", "tests/data/hardening/ws_rstrip.json" },
        { "ws_lrstrip", "tests/data/hardening/ws_lrstrip.json" },   { "llama3style", "tests/data/compile/llama3style.json" },
        { "gemma4like", "tests/data/spm/gemma4like.json" },          { "llamalike", "tests/data/spm/llamalike.json" },
    };
    for (size_t i = 0; i < sizeof FIX / sizeof FIX[0]; i++) { run(FIX[i].name, FIX[i].path, 0.0); }
    /* bounds per path: ~3x the worst class measured at 64 KiB (docs/hardening.md §2) */
    static const struct { const char *name; double bound; } PIN[] = {
        { "llama3", 60.0 }, { "o200k", 60.0 }, { "qwen38", 60.0 }, { "dsv3", 60.0 }, { "gpt2", 60.0 },
        { "gemma4", 40.0 }, { "mistral-v0.3", 40.0 }, { "uni_bgem3", 20.0 }, { "uni_t5base", 20.0 },
        { "wp-bert-uncased", 20.0 },
    };
    for (size_t i = 0; i < sizeof PIN / sizeof PIN[0]; i++) {
        if (dir != NULL) { snprintf(pin, sizeof pin, "%s/%s", dir, PIN[i].name); }
        else { snprintf(pin, sizeof pin, "%s/.cache/toks/tokenizers/%s", home != NULL ? home : ".", PIN[i].name); }
        run(PIN[i].name, pin, PIN[i].bound);
    }
    printf("test_stall: %d checks, %d failures (%d re-measured carefully)\n", checks, failures, remeasured);
    return failures != 0;
}
