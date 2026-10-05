/*
 * test_stall.c: the stall regression of docs/hardening.md (SPEC §7.3, T7): adversarial input classes, each
 * aimed at a path (whitespace / strip-token floods for the added-token walk, one giant piece for K6's long path
 * and spm's whole-segment words, '<' floods for K1, combining marks for the normalizers, random CJK / bytes),
 * timed through toks.h at 8, 32 and 128 KiB, best of 3 cold calls each (the scratch initialized outside the
 * timer). A class fails when
 *   - its time grows superlinearly over both steps: x8 or more for 4x the bytes twice (exponent >= 1.5; linear
 *     is x4, K6's heap n log n ~x4.5, the quadratic walks this test was written for x16); one step alone is a
 *     cache boundary, not a stall. A first step of x8 whose call already takes > 0.1 s fails at once;
 *   - it costs more than its path's bound x the pseudo-en text's ns per byte at 128 KiB (pinned tokenizers only;
 *     each bound is ~3x the worst class measured, docs/hardening.md §2: it catches a new stall, not noise).
 * The fixtures always run; a pinned tokenizer missing from $TOKS_TOKENIZER_CACHE (~/.cache/toks/tokenizers) is a
 * SKIP. A few seconds on the Mac.
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

static int failures, checks;

/* best of 3 cold calls: ns per call */
static double call_ns(const toks_ctx *ctx, const uint8_t *x, uint64_t n, void *scr, uint64_t sb, uint32_t *out)
{
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        if (toks_scratch_init(ctx, scr, sb, 0u) != 0) { return -1.0; }
        uint64_t t0 = now_ns();
        int64_t m = toks_encode(ctx, x, n, 0u, out, 3u * n + 64u, scr);
        double t = (double)(now_ns() - t0);
        if (m < 0) { return -1.0; }
        if (t < best) { best = t; }
    }
    return best;
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
    const uint64_t sz[3] = { 8192u, 32768u, 131072u };
    uint64_t sb = toks_scratch_bytes(ctx, sz[2], 0u);
    void *scr = malloc(sb);
    uint8_t *x = (uint8_t *)malloc(sz[2]);
    uint32_t *out = (uint32_t *)malloc((3u * sz[2] + 64u) * 4u);
    double en = 0.0, worst_g = 0.0, worst_x = 0.0;
    const char *wg = "-", *wx = "-";
    for (size_t c = 0; c < NCLS; c++) {
        double t[3] = { 0.0, 0.0, 0.0 }, g[2] = { 0.0, 0.0 };
        int bad = 0;
        for (int i = 0; i < 3 && !bad; i++) {
            gen(&CLS[c], &l, x, sz[i]);
            t[i] = call_ns(ctx, x, sz[i], scr, sb, out);
            if (t[i] <= 0.0) { bad = 1; break; }
            if (i > 0) { g[i - 1] = t[i] / t[i - 1]; }
            if (i == 1 && g[0] >= 8.0 && t[1] > 1e8) { break; }   /* already a stall: do not wait for the next size */
        }
        checks++;
        if (bad) {
            printf("  FAIL %s %s: encode failed\n", name, CLS[c].name);
            failures++;
            continue;
        }
        double nsb = t[2] > 0.0 ? t[2] / (double)sz[2] : t[1] / (double)sz[1];
        if (CLS[c].kind == EN) { en = nsb; }
        double gm = g[0] < g[1] ? g[0] : g[1];                         /* sustained growth: the smaller step */
        if (t[2] == 0.0) { gm = g[0]; }
        if (gm > worst_g) { worst_g = gm; wg = CLS[c].name; }
        if (en > 0.0 && nsb / en > worst_x) { worst_x = nsb / en; wx = CLS[c].name; }
        if (gm >= 8.0) {
            printf("  FAIL %s %s: superlinear, x%.1f per 4x the bytes (%.1f / %.1f ms at 8 / 32 KiB%s)\n", name, CLS[c].name, gm,
                   t[0] / 1e6, t[1] / 1e6, t[2] > 0.0 ? ", and again to 128 KiB" : "; 128 KiB not run");
            failures++;
        }
        if (bound > 0.0 && en > 0.0 && nsb > bound * en) {
            printf("  FAIL %s %s: %.1f ns/B = %.1fx en (%.1f ns/B) at 128 KiB, bound %.0fx\n", name, CLS[c].name, nsb, nsb / en,
                   en, bound);
            failures++;
        }
    }
    printf("  %-18s en %5.1f ns/B; worst growth x%.1f per 4x bytes (%s); worst %.1fx en (%s)%s\n", name, en, worst_g, wg,
           worst_x, wx, bound > 0.0 ? "" : " [growth only]");
    free(out);
    free(x);
    free(scr);
    toks_unload(ctx);
}

int main(void)
{
    char pin[1024];
    const char *dir = getenv("TOKS_TOKENIZER_CACHE"), *home = getenv("HOME");
    printf("test_stall: adversarial classes at 8 / 32 / 128 KiB (docs/hardening.md)\n");
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
    printf("test_stall: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
