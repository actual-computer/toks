/*
 * test_split.c: toks_split_points (SPEC §4.5, §5; docs/split.md).
 *
 *   semantics   argument errors, n_want <= 1, cap, a family without rules (TOKS_TMPL_NONE) gets no cut, and
 *               the selection (nearest certified cut to len * i / n_want, lower on a tie, within D, skipped
 *               when none, equal neighbours collapsed, strictly increasing) against a transcription over the
 *               set of every certified position (n_want = len gives exactly that set: D = 0).
 *   exactness   §5.3's evidence: every string of up to L symbols over class-representative alphabets (each
 *               template variant: gpt-2, llama 3 / cl100k, qwen 2, qwen 3.5; added-token alphabets for the
 *               fixtures with tokens), every mode, every certified cut: the whole input's pieces have a
 *               boundary there, the prefix's pieces are the whole input's prefix, the suffix's pieces (with
 *               TOKS_CONTINUATION) the shifted rest, and the ids concatenate to the whole input's ids
 *               (no post-processing); then all of a string's cuts at once (adjacent cuts included). Random
 *               long strings over the same symbols the same way. Every family with rules (docs/split.md): the
 *               cl100k template variants, o200k / nemo (with NFC too), deepseek's three Splits, sentencepiece-style
 *               bpe, wordpiece, unigram. The real gpt2 / llama3 / GLM 5.3 / gpt-oss / nemotron / llama 4 / minimax /
 *               deepseek / bert / bge-m3 / t5 files ($TOKS_TOKENIZER_CACHE or ~/.cache/toks/tokenizers) too when
 *               present (SKIP when not); files whose rules are refused (kimi's cuts, truncation) get no cut.
 *   reads       each real file plans a 32 MiB text whose pages beyond D + 512 bytes of a target are no-access.
 *
 *   test_split [L [random [seed]]]     defaults 3 / 300 / 1 (make test); the evidence runs pass L = 5 or 6.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rng = 1;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

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

/* a tokenizer file ("tests/..." paths are the fixtures, bare names the pinned real files), with up to two
 * literal rewrites (template variants, an NFC normalizer) applied first. */
static toks_ctx *load(const char *name, const char *f1, const char *t1, const char *f2, const char *t2)
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
    const char *from[2] = { f1, f2 }, *to[2] = { t1, t2 };
    for (int k = 0; k < 2; k++) {
        if (from[k] == NULL) { continue; }
        b[n] = 0;
        char *at = strstr((char *)b, from[k]);
        if (at == NULL) { free(b); CHECK(0, "%s: no %s to rewrite", name, from[k]); return NULL; }
        size_t lf = strlen(from[k]), lt = strlen(to[k]), pre = (size_t)(at - (char *)b);
        uint8_t *c = (uint8_t *)malloc((size_t)n - lf + lt + 1u);
        memcpy(c, b, pre);
        memcpy(c + pre, to[k], lt);
        memcpy(c + pre + lt, b + pre + lf, (size_t)n - pre - lf);
        free(b);
        b = c;
        n = n - lf + lt;
    }
    toks_ctx *ctx = NULL;
    int64_t r = toks_load_mem_copy(&ctx, b, n, NULL);
    free(b);
    if (r != 0) { printf("SKIP %s: does not load here (%" PRId64 ")\n", name, r); return NULL; }
    return ctx;
}

/* ================================================================ the exactness check of one string */

typedef struct run {
    toks_ctx *ctx;
    void *scr;
    uint64_t *offs;
    int pieces;                              /* 0: ids only (spm's pieces are not its words; wordpiece and
                                                unigram drop whitespace from the normalized stream) */
    uint64_t strings, cuts, bad;
} run;

#define MAXB 8192u

static void report(run *R, const char *what, uint64_t c, uint32_t mode, const uint8_t *x, uint64_t n)
{
    R->bad++;
    if (R->bad > 12) { return; }
    fprintf(stderr, "FAIL %s at %" PRIu64 " mode %u in:", what, c, mode);
    for (uint64_t q = 0; q < n && q < 200u; q++) { fprintf(stderr, " %02x", x[q]); }
    fputc('\n', stderr);
}

/* 1 when the cuts of x under mode keep pieces (in the normalized stream) and ids, alone and all together. */
static int check_string(run *R, const uint8_t *x, uint64_t n, uint32_t mode)
{
    toks_ctx *ctx = R->ctx;
    uint32_t fl = mode | TOKS_NO_POSTPROCESS;
    int64_t nc = toks_split_points(ctx, x, n, mode, (uint32_t)(n == 0 ? 1 : n), R->offs, n, NULL);
    if (nc < 0) { CHECK(0, "split_points %" PRId64, nc); return 0; }
    R->strings++;
    if (nc == 0) { return 1; }
    static uint32_t E[MAXB * 3u], I[MAXB], P[MAXB * 3u], Q[MAXB * 3u];
    int64_t ne = R->pieces ? toks_pieces(ctx, x, n, fl, E, MAXB * 3u, R->scr) : 0;
    int64_t ni = toks_encode(ctx, x, n, fl, I, MAXB, R->scr);
    if (ne < 0 || ni < 0 || ne > (int64_t)MAXB * 3 || ni > (int64_t)MAXB) { CHECK(0, "whole encode"); return 0; }
    int ok = 1;
    for (int64_t k = 0; k < nc; k++) {
        uint64_t c = R->offs[k];
        if (n > 64u && rnd() % (uint64_t)nc >= 8u) { continue; }   /* long strings: ~8 cuts alone, all at once below */
        R->cuts++;
        if (R->pieces) {
            int64_t nl = toks_pieces(ctx, x, c, fl, P, MAXB * 3u, R->scr);
            int64_t nr = toks_pieces(ctx, x + c, n - c, fl | TOKS_CONTINUATION, Q, MAXB * 3u, R->scr);
            uint64_t cn = nl > 0 ? P[nl - 1] : 0u;         /* the cut in the normalized stream */
            int good = nl > 0 && nr > 0 && nl + nr == ne && memcmp(P, E, (size_t)nl * 4u) == 0 && E[nl - 1] == cn;
            for (int64_t q = 0; good && q < nr; q++) { good = (uint64_t)Q[q] + cn == E[nl + q]; }
            if (!good) { ok = 0; report(R, "pieces", c, mode, x, n); }
        }
        int64_t il = toks_encode(ctx, x, c, fl, P, MAXB, R->scr);
        int64_t ir = il < 0 ? -1 : toks_encode(ctx, x + c, n - c, fl | TOKS_CONTINUATION, P + il, MAXB - (uint64_t)il, R->scr);
        if (!(il >= 0 && ir >= 0 && il + ir == ni && memcmp(P, I, (size_t)ni * 4u) == 0)) {
            ok = 0;
            report(R, "ids", c, mode, x, n);
        }
    }
    uint64_t at = 0, got = 0;                  /* every cut at once */
    for (int64_t k = 0; k <= nc; k++) {
        uint64_t e = k < nc ? R->offs[k] : n;
        int64_t r = toks_encode(ctx, x + at, e - at, fl | (k > 0 ? TOKS_CONTINUATION : 0u), P + got, MAXB - got, R->scr);
        if (r < 0 || got + (uint64_t)r > MAXB) { got = MAXB + 1u; break; }
        got += (uint64_t)r;
        at = e;
    }
    if (got != (uint64_t)ni || memcmp(P, I, (size_t)ni * 4u) != 0) { ok = 0; report(R, "all cuts at once", 0, mode, x, n); }
    return ok;
}

static const uint32_t MODES[3] = { TOKS_ADDED_ALL, TOKS_ADDED_NONSPECIAL, TOKS_ADDED_NONE };

/* every string of up to L symbols, every mode. */
static void enumerate(run *R, const char *const *sym, uint32_t ns, uint32_t L)
{
    uint32_t idx[16];
    uint8_t x[MAXB];
    for (uint32_t len = 0; len <= L; len++) {
        memset(idx, 0, sizeof idx);
        for (;;) {
            uint64_t n = 0;
            for (uint32_t k = 0; k < len; k++) {
                size_t sl = strlen(sym[idx[k]]);
                memcpy(x + n, sym[idx[k]], sl);
                n += sl;
            }
            for (int m = 0; m < 3; m++) { check_string(R, x, n, MODES[m]); }
            uint32_t k = 0;                    /* next tuple */
            while (k < len && ++idx[k] == ns) { idx[k] = 0; k++; }
            if (k == len) { break; }
        }
    }
}

/* random strings of 1..maxb bytes over the symbols (spaces and newlines weighted up: more cuts). */
static void randoms(run *R, const char *const *sym, uint32_t ns, uint32_t count, uint64_t maxb)
{
    uint8_t *x = (uint8_t *)malloc(MAXB);
    for (uint32_t t = 0; t < count; t++) {
        uint64_t want = 1u + rnd() % maxb, n = 0;
        while (n < want) {
            const char *s = (rnd() % 4u == 0u) ? " " : sym[rnd() % ns];
            size_t sl = strlen(s);
            if (n + sl > MAXB / 2u) { break; }
            memcpy(x + n, s, sl);
            n += sl;
        }
        check_string(R, x, n, MODES[rnd() % 3u]);
    }
    free(x);
}

/* ================================================================ the selection against a transcription */

static void selection(toks_ctx *ctx, void *scr, const char *const *sym, uint32_t ns)
{
    (void)scr;
    uint8_t *x = (uint8_t *)malloc(1u << 16);
    uint64_t *all = (uint64_t *)malloc(8u << 16), *got = (uint64_t *)malloc(8u * 4096u);
    uint8_t *is = (uint8_t *)malloc(1u << 16);
    for (int t = 0; t < 400; t++) {
        uint64_t n = 0, want = 2u + rnd() % ((t & 7) == 0 ? 60000u : 300u);
        while (n < want) {
            const char *s = (rnd() % 3u == 0u) ? " " : sym[rnd() % ns];
            size_t sl = strlen(s);
            if (n + sl > (1u << 16) - 8u) { break; }
            memcpy(x + n, s, sl);
            n += sl;
        }
        uint32_t mode = MODES[rnd() % 3u];
        int64_t na = toks_split_points(ctx, x, n, mode, (uint32_t)n, all, n, NULL);
        CHECK(na >= 0, "all: %" PRId64, na);
        memset(is, 0, (size_t)n + 1u);
        for (int64_t k = 0; k < na; k++) {
            CHECK(all[k] > 0 && all[k] < n && (k == 0 || all[k] > all[k - 1]), "all: order");
            is[all[k]] = 1;
        }
        for (int q = 0; q < 6; q++) {
            uint32_t nw = (q == 5) ? (uint32_t)(n + 1u + rnd() % 9u) : 2u + (uint32_t)(rnd() % 70u);
            uint64_t cap = (q == 4) ? rnd() % 5u : 4096u;
            int64_t nc = toks_split_points(ctx, x, n, mode, nw, got, cap, NULL);
            /* the transcription */
            uint64_t w = (nw > n) ? n : nw, D = n / (4u * w), prev = 0, e = 0;
            if (D > 4096u) { D = 4096u; }
            int same = nc >= 0;
            for (uint64_t i = 1; i < w && e < cap; i++) {
                uint64_t tg = n * i / w, best = 0;
                for (uint64_t d = 0; d <= D && best == 0; d++) {
                    if (d < tg && is[tg - d]) { best = tg - d; }
                    else if (d > 0 && tg + d < n && is[tg + d]) { best = tg + d; }
                }
                if (best > prev) {
                    if (same && ((int64_t)e >= nc || got[e] != best)) { same = 0; }
                    e++;
                    prev = best;
                }
            }
            CHECK(same && (int64_t)e == nc, "selection: n %" PRIu64 " n_want %u cap %" PRIu64 ": got %" PRId64 " want %" PRIu64,
                  n, nw, cap, nc, e);
        }
    }
    free(x); free(all); free(got); free(is);
}

static void api(toks_ctx *ctx, toks_ctx *nosplit)
{
    uint64_t o[8];
    const char *t = "hello world again";
    uint64_t n = strlen(t);
    CHECK(toks_split_points(NULL, t, n, 0, 4, o, 8, NULL) == TOKS_E_ARG, "ctx NULL");
    CHECK(toks_split_points(ctx, t, n, 3, 4, o, 8, NULL) == TOKS_E_ARG, "mode 3");
    CHECK(toks_split_points(ctx, t, n, 16, 4, o, 8, NULL) == TOKS_E_ARG, "unknown flag");
    CHECK(toks_split_points(ctx, NULL, n, 0, 4, o, 8, NULL) == TOKS_E_ARG, "text NULL");
    CHECK(toks_split_points(ctx, NULL, 0, 0, 4, o, 8, NULL) == 0, "empty");
    CHECK(toks_split_points(ctx, t, n, 0, 4, NULL, 8, NULL) == TOKS_E_ARG, "offs NULL");
    CHECK(toks_split_points(ctx, t, n, 0, 4, NULL, 0, NULL) == 0, "cap 0");
    CHECK(toks_split_points(ctx, t, (1ull << 29) + 1u, 0, 4, o, 8, NULL) == TOKS_E_LIMIT, "limit");
    CHECK(toks_split_points(ctx, t, n, 0, 0, o, 8, NULL) == 0, "n_want 0");
    CHECK(toks_split_points(ctx, t, n, 0, 1, o, 8, NULL) == 0, "n_want 1");
    CHECK(toks_split_points(ctx, t, 1, 0, 4, o, 8, NULL) == 0, "len 1");
    CHECK(toks_split_points(ctx, t, n, 0, 2, o, 8, NULL) == 0, "target 8, D = 17 / 8 = 2: o|_ at 5 is too far");
    CHECK(toks_split_points(ctx, "aaaa bbbb", 9, TOKS_CONTINUATION | TOKS_NO_POSTPROCESS, 2, o, 8, NULL) == 1 &&
          o[0] == 4, "aaaa|_bbbb");
    CHECK(toks_split_points(ctx, "aaa bbbbb", 9, 0, 2, o, 8, NULL) == 1 && o[0] == 3, "target 4, D 1: 3");
    CHECK(toks_split_points(ctx, "aaaaa bbb", 9, 0, 2, o, 8, NULL) == 1 && o[0] == 5, "target 4, D 1: 5");
    CHECK(toks_split_points(ctx, "aaa!a bbb", 9, 0, 2, o, 8, NULL) == 1 && o[0] == 3, "a tie: the lower");
    CHECK(toks_split_points(ctx, t, n, 0, 3, o, 1, NULL) == 1, "cap 1");
    CHECK(toks_split_points(nosplit, t, n, 0, (uint32_t)n, o, 8, NULL) == 0, "TOKS_TMPL_NONE: no cut");
}

/* ================================================================ the alphabets */

/* class representatives: L (ascii, contraction letters s r e S, U+017F folding to s, 2- and 3-byte), N (ascii,
 * U+0663), P (apostrophe, '!', '<', invalid bytes: a lone lead, a stray continuation, a truncated
 * sequence), WS (space, tab, U+3000), NL (LF, CR), M (U+0301: P, or L where marks fold into letters). */
static const char *const CLS[] = {
    "a", "s", "r", "e", "S", "\xc5\xbf", "\xc3\xa9", "\xe4\xb8\xad", "1", "\xd9\xa3",
    "'", "!", "<", "\xff", "\x80", "\xe4\xb8", " ", "\t", "\xe3\x80\x80", "\n", "\r", "\xcc\x81",
};
#define N_CLS (sizeof CLS / sizeof CLS[0])

/* NFC: atoms that compose with what precedes them (U+0301 U+030A U+0307 U+0323, V and T jamo), that
 * decompose (U+212B, U+0344), their bases (e A G, L jamo, an LV syllable) and plain class representatives. */
static const char *const NFC_CLS[] = {
    "a", "e", "A", "G", "s", "'", "1", "!", " ", "\n", "\xff", "\xe4\xb8\xad", "\xc3\xa9",
    "\xcc\x81", "\xcc\x8a", "\xcc\x87", "\xcc\xa3", "\xe1\x84\x80", "\xe1\x85\xa1", "\xe1\x86\xa8", "\xea\xb0\x80",
    "\xe2\x84\xab", "\xcd\x84",
};
#define N_NFC_CLS (sizeof NFC_CLS / sizeof NFC_CLS[0])

/* sentencepiece-style bpe: space and a literal U+2581 (one char-table entry), the gemma "> </" junction,
 * chars with and without vocab entries (byte fallback: an emoji, an invalid byte), tab, newline. */
static const char *const SPM_CLS[] = {
    "a", "b", "e", " ", "\xe2\x96\x81", ">", "<", "/", "p", "\t", "\n", "\xc3\xa9", "\xe4\xb8\xad", "1", "!",
    "\xff", "\xf0\x9f\x98\x80",
};
#define N_SPM_CLS (sizeof SPM_CLS / sizeof SPM_CLS[0])

/* llama3style's added tokens (tests/data/compile): whole, overlapping and partial occurrences; '@' one byte;
 * two spaces (phase 0); phase 1: tab, "é", "«ñ»", " hi there " (strip-like whitespace inside a token). */
static const char *const TOK_L3[] = {
    "a", " ", "\t", "\n", "!", "@", "<|a|>", "<|a", "b|>", "|>", "<|x", "y|>", "\xc3\xa9",
    "\xc2\xab\xc3\xb1\xc2\xbb", "\xc2\xab", " hi", " there", "1",
};
#define N_TOK_L3 (sizeof TOK_L3 / sizeof TOK_L3[0])

static const char *const TOK_GPT2[] = {
    "a", " ", "\n", "!", "1", "<|endoftext|>", "<|endof", "text|>", "<|", "|>",
};
#define N_TOK_GPT2 (sizeof TOK_GPT2 / sizeof TOK_GPT2[0])

/* the real files' tokens (each file recognizes its own; the others are text) and their fragments */
static const char *const TOK_REAL[] = {
    "a", " ", "\n", "!", "1", "\xc3\xa9", "<|endoftext|>", "<|begin_of_text|>", "<|eot_id|>", "<|im_start|>",
    "<|im_end|>", "<|user|>", "[gMASK]", "<sop>", "<|", "|>", "\t", "<bos>", "<|turn>", "<turn|>", "<|\"|>",
    "<think>", "</think>", "\xe2\x96\x81", ">",
};
#define N_TOK_REAL (sizeof TOK_REAL / sizeof TOK_REAL[0])

/* o200k: upper-only (A S T), lower-only (a s t e r l d m), both (Lm U+02B0, Lo 中), Lt U+01C5, a mark, U+017F folding
 * to s, digits (ascii, U+0663), P (' ! / < and invalid bytes), WS (space, tab, U+3000), NL (LF, CR) */
static const char *const O2_CLS[] = {
    "a", "s", "t", "e", "r", "l", "A", "S", "T", "\xca\xb0", "\xe4\xb8\xad", "\xc7\x85", "\xcc\x81", "\xc5\xbf", "1",
    "\xd9\xa3", "'", "!", "/", "<", "\xff", "\xe4\xb8", " ", "\t", "\xe3\x80\x80", "\n", "\r",
};
#define N_O2_CLS (sizeof O2_CLS / sizeof O2_CLS[0])

/* o200k + NFC (minimax): composing marks and their bases beside the case classes */
static const char *const O2_NFC_CLS[] = {
    "a", "e", "A", "s", "\xca\xb0", "\xe4\xb8\xad", "'", "1", "!", "/", " ", "\t", "\n", "\xff", "\xc3\xa9", "\xcc\x81",
    "\xcc\x8a", "\xcc\xa3", "\xe1\x84\x80", "\xe1\x85\xa1", "\xea\xb0\x80", "\xe2\x84\xab",
};
#define N_O2_NFC_CLS (sizeof O2_NFC_CLS / sizeof O2_NFC_CLS[0])

/* deepseek v3: ascii and other letters, a mark (L there), CJK (Han, katakana), digits, ascii and other P / S, WS,
 * NL, X atoms (U+200B format, a C0 control) and invalid bytes */
static const char *const DS_CLS[] = {
    "a", "B", "\xc3\xa9", "\xcc\x81", "\xe4\xb8\xad", "\xe3\x82\xa2", "1", "\xd9\xa3", "!", "'", "<", "\xc2\xab",
    "\xe2\x82\xac", "\xff", " ", "\t", "\xe3\x80\x80", "\n", "\r", "\xe2\x80\x8b", "\x01",
};
#define N_DS_CLS (sizeof DS_CLS / sizeof DS_CLS[0])

/* wordpiece: word bytes (folded or not), punctuation, ascii and other whitespace, a removed control, accents and
 * CJK (the normalizer's work), the continuation prefix, invalid bytes */
static const char *const WP_CLS[] = {
    "a", "B", "un", "##", "!", ",", "[", " ", "\t", "\n", "\r", "\xc3\xa9", "e\xcc\x81", "\xe4\xb8\xad", "\x01", "1",
    "\xe3\x80\x80", "\xc2\xa0", "\xff",
};
#define N_WP_CLS (sizeof WP_CLS / sizeof WP_CLS[0])

/* unigram: simple ascii, U+0020 (runs collapse), tab / newline / NBSP / U+3000 (the charsmap's spaces), U+2581,
 * fullwidth A, a mark, CJK, invalid bytes */
static const char *const UNI_CLS[] = {
    "a", "b", "Th", "1", ".", "!", " ", "\t", "\n", "\xc2\xa0", "\xe3\x80\x80", "\xe2\x96\x81", "\xef\xbc\xa1",
    "\xcc\x81", "\xc3\xa9", "\xe4\xb8\xad", "\xff",
};
#define N_UNI_CLS (sizeof UNI_CLS / sizeof UNI_CLS[0])

/* the real files' tokens beyond TOK_REAL: gpt-oss / llama 4 / nemotron / deepseek / bert / xlm-r */
static const char *const TOK_REAL2[] = {
    "a", " ", "\n", "!", "1", "<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|eot|>", "<|header_start|>",
    "<|begin_of_text|>", "<SPECIAL_10>", ("<\xef\xbd\x9c" "User\xef\xbd\x9c>"), "[CLS]", "[SEP]", "[MASK]", "<s>",
    "</s>", "<mask>", "<|", "|>", "[", "\t",
};
#define N_TOK_REAL2 (sizeof TOK_REAL2 / sizeof TOK_REAL2[0])

/* the template strings as they appear in a tokenizer.json (o200k.md §1), for literal rewrites of llama3style */
#define RE_LLAMA3 "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"
#define RE_O200K "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"
#define RE_NEMO "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"

static void family(const char *label, toks_ctx *ctx, const char *const *cls, uint32_t ncls,
                   const char *const *tok, uint32_t ntok, uint32_t L, uint32_t nrand)
{
    if (ctx == NULL) { return; }
    run R;
    memset(&R, 0, sizeof R);
    R.ctx = ctx;
    toks_info info;
    info.size = (uint32_t)sizeof info;
    CHECK(toks_get_info(ctx, &info) == 0, "info");
    R.pieces = info.algorithm == TOKS_ALGO_BPE_BYTELEVEL;   /* spm / wordpiece / unigram: ids only (§5.2) */
    uint64_t sb = toks_scratch_bytes(ctx, MAXB, 0);
    R.scr = malloc((size_t)sb);
    R.offs = (uint64_t *)malloc(8u * MAXB);
    CHECK(toks_scratch_init(ctx, R.scr, sb, 0) == 0, "scratch");
    enumerate(&R, cls, ncls, L);
    uint64_t s1 = R.strings, c1 = R.cuts;
    if (tok != NULL) { enumerate(&R, tok, ntok, L > 1 ? L - 1u : L); }
    randoms(&R, cls, ncls, nrand, 3000u);
    if (tok != NULL) { randoms(&R, tok, ntok, nrand, 3000u); }
    selection(ctx, R.scr, cls, ncls);
    printf("  %-14s L=%u: classes %" PRIu64 " string-modes / %" PRIu64 " cuts; tokens + random %" PRIu64
           " / %" PRIu64 "; mismatches %" PRIu64 "\n", label, L, s1, c1, R.strings - s1, R.cuts - c1, R.bad);
    CHECK(R.bad == 0, "%s: %" PRIu64 " mismatching cuts", label, R.bad);
    free(R.scr);
    free(R.offs);
}

#define NFC_FROM "\"normalizer\": null"
#define NFC_TO   "\"normalizer\": {\"type\": \"NFC\"}"

/* toks.h: "Reads at most n_want x (2D + W) bytes, W per tokenizer": a 32 MiB text whose only accessible pages are
 * those within D + 512 bytes of a target (W is twice the longest recognized added token + 16 a side, docs/split.md:
 * under 512 a side for every file here), the rest no-access. Text with cuts near every target, then letters with no
 * cut at all (every target's whole window read, D each side): split_points never faults, and every cut is within D */
static void reads_bounded(const char *name, toks_ctx *ctx)
{
    if (ctx == NULL) { return; }
    const uint64_t len = 32ull << 20;
    const uint32_t nw = 8u;
    uint64_t d = len / (4u * nw);
    if (d > 4096u) { d = 4096u; }
    const uint64_t side = d + 512u;
    static const char *const FILL[2] = { "The quick brown fox jumps over the lazy dog, 1234 times. ", "abcdefghij" };
    for (int f = 0; f < 2; f++) {                           /* bound: 2 fills */
        guard_buf g;
        uint8_t *x = guard_map(&g, (size_t)len, 0);
        CHECK(x != NULL, "%s: guard_map 32 MiB", name);
        if (x == NULL) { return; }
        size_t pg = guard_page_size(), fl = strlen(FILL[f]);
        for (uint32_t i = 1u; i < nw; i++) {                /* bound: nw - 1 targets */
            uint64_t tg = len * i / nw, a = (tg - side) & ~(uint64_t)(pg - 1u), b = (tg + side + pg - 1u) & ~(uint64_t)(pg - 1u);
            CHECK(guard_open(&g, (size_t)a, (size_t)(b - a)) == 0, "%s: guard_open", name);
            for (uint64_t j = a; j < b; j++) { x[j] = (uint8_t)FILL[f][j % fl]; }   /* bound: the window's pages */
        }
        uint64_t offs[16];
        int64_t c = toks_split_points(ctx, x, len, 0u, nw, offs, 16u, NULL);
        int near = c >= 0;
        for (int64_t k = 0; k < c; k++) {                   /* bound: c <= 7 cuts */
            int ok = 0;
            for (uint32_t i = 1u; i < nw; i++) { uint64_t tg = len * i / nw; ok |= offs[k] + d >= tg && offs[k] <= tg + d; }
            near &= ok;
        }
        CHECK(near && (f == 1 || c >= 0), "%s: %s: %" PRId64 " cuts, each within D of a target", name, f ? "letters" : "text", c);
        if (f == 0) { printf("  %-14s reads within D + 512 of 7 targets in 32 MiB: %" PRId64 " cuts", name, c); }
        else { printf(", %" PRId64 " in letters\n", c); }
        guard_free(&g);
    }
}

int main(int argc, char **argv)
{
    uint32_t L = argc > 1 ? (uint32_t)atoi(argv[1]) : 3u;
    uint32_t nrand = argc > 2 ? (uint32_t)atoi(argv[2]) : 60u;
    rng = argc > 3 ? (uint64_t)strtoull(argv[3], NULL, 10) | 1u : 1u;
    if (L > 8u) { L = 8u; }
    uint32_t Ls = L > 1u ? L - 1u : L;           /* the fixtures with more symbols, and the real files */

    toks_ctx *l3s = load("tests/data/compile/llama3style.json", NULL, NULL, NULL, NULL);
    toks_ctx *nos = load("tests/data/compile/nosplit.json", NULL, NULL, NULL, NULL);
    if (l3s != NULL && nos != NULL) { api(l3s, nos); }

    static const struct { const char *label, *path, *f1, *t1, *f2, *t2; int kind; } F[] = {
        { "gpt2style", "tests/data/compile/gpt2style.json", NULL, NULL, NULL, NULL, 1 },
        { "llama3style", "tests/data/compile/llama3style.json", NULL, NULL, NULL, NULL, 2 },
        { "qwen2style", "tests/data/compile/llama3style.json", "\\\\p{N}{1,3}", "\\\\p{N}", NULL, NULL, 2 },
        { "qwen35style", "tests/data/compile/qwen35style.json", NULL, NULL, NULL, NULL, 0 },
        { "nosplit", "tests/data/compile/nosplit.json", NULL, NULL, NULL, NULL, 2 },
        { "gpt2style+nfc", "tests/data/compile/gpt2style.json", NFC_FROM, NFC_TO, NULL, NULL, 3 },
        { "llama3+nfc", "tests/data/compile/llama3style.json", NFC_FROM, NFC_TO, NULL, NULL, 3 },
        { "qwen35+nfc", "tests/data/compile/qwen35style.json", NFC_FROM, NFC_TO, NULL, NULL, 3 },
        { "spm gemma4like", "tests/data/spm/gemma4like.json", NULL, NULL, NULL, NULL, 4 },
        { "spm mistrallike", "tests/data/spm/mistrallike.json", NULL, NULL, NULL, NULL, 4 },
        { "spm always_split", "tests/data/spm/meta_always_split.json", NULL, NULL, NULL, NULL, 4 },
        { "spm never_split", "tests/data/spm/meta_never_split.json", NULL, NULL, NULL, NULL, 4 },
        { "spm llamalike", "tests/data/spm/llamalike.json", NULL, NULL, NULL, NULL, 4 },
        { "spm ignore_merges", "tests/data/spm/ignore_merges.json", NULL, NULL, NULL, NULL, 4 },
        { "spm unk_fused", "tests/data/spm/unk_fused.json", NULL, NULL, NULL, NULL, 4 },
        { "spm space_vocab", "tests/data/spm/space_in_vocab.json", NULL, NULL, NULL, NULL, 4 },
        { "spm replace", "tests/data/spm/replace_chain.json", NULL, NULL, NULL, NULL, 4 },
        { "o200kstyle", "tests/data/compile/llama3style.json", RE_LLAMA3, RE_O200K, NULL, NULL, 5 },
        { "nemostyle", "tests/data/compile/llama3style.json", RE_LLAMA3, RE_NEMO, NULL, NULL, 5 },
        { "o200k+nfc", "tests/data/compile/llama3style.json", RE_LLAMA3, RE_O200K, NFC_FROM, NFC_TO, 6 },
        { "dsv3style", "tests/data/compile/dsv3style.json", NULL, NULL, NULL, NULL, 7 },
    };
    for (size_t k = 0; k < sizeof F / sizeof F[0]; k++) {
        toks_ctx *c = load(F[k].path, F[k].f1, F[k].t1, F[k].f2, F[k].t2);
        switch (F[k].kind) {
        case 0: family(F[k].label, c, CLS, (uint32_t)N_CLS, NULL, 0, L, nrand); break;
        case 1: family(F[k].label, c, CLS, (uint32_t)N_CLS, TOK_GPT2, (uint32_t)N_TOK_GPT2, L, nrand); break;
        case 2: family(F[k].label, c, CLS, (uint32_t)N_CLS, TOK_L3, (uint32_t)N_TOK_L3, k == 4 ? Ls : L, nrand); break;
        case 3: family(F[k].label, c, NFC_CLS, (uint32_t)N_NFC_CLS, TOK_L3, (uint32_t)N_TOK_L3, L, nrand); break;
        case 4: family(F[k].label, c, SPM_CLS, (uint32_t)N_SPM_CLS, NULL, 0, L, nrand); break;
        case 5: family(F[k].label, c, O2_CLS, (uint32_t)N_O2_CLS, TOK_L3, (uint32_t)N_TOK_L3, L, nrand); break;
        case 6: family(F[k].label, c, O2_NFC_CLS, (uint32_t)N_O2_NFC_CLS, TOK_L3, (uint32_t)N_TOK_L3, L, nrand); break;
        default: family(F[k].label, c, DS_CLS, (uint32_t)N_DS_CLS, TOK_L3, (uint32_t)N_TOK_L3, L, nrand); break;
        }
        toks_unload(c);
    }

    static const struct { const char *name; int kind; } RF[] = {
        { "gpt2", 0 }, { "llama3", 0 }, { "glm53", 0 }, { "qwen3", 1 }, { "qwen38", 1 }, { "gemma4", 2 },
        { "gemma4-base", 2 }, { "mistral-v0.3", 2 }, { "llama2", 2 }, { "o200k", 3 }, { "nemotron3-4b", 3 },
        { "llama4", 3 }, { "minimaxm2", 4 }, { "dsv3", 5 }, { "dsv4", 5 }, { "wp-bert-base-uncased", 6 },
        { "wp-labse", 6 }, { "wp-minilm-l6", 6 }, { "uni_bgem3", 7 }, { "uni_t5base", 7 }, { "uni_flant5", 7 },
        { "uni_me5small", 7 }, { "uni_ruri3", 7 },
    };
    static const struct { const char *const *cls; size_t n; } KC[] = {
        { CLS, N_CLS }, { NFC_CLS, N_NFC_CLS }, { SPM_CLS, N_SPM_CLS }, { O2_CLS, N_O2_CLS }, { O2_NFC_CLS, N_O2_NFC_CLS },
        { DS_CLS, N_DS_CLS }, { WP_CLS, N_WP_CLS }, { UNI_CLS, N_UNI_CLS },
    };
    for (size_t k = 0; k < sizeof RF / sizeof RF[0]; k++) {
        toks_ctx *c = load(RF[k].name, NULL, NULL, NULL, NULL);
        int kd = RF[k].kind;
        family(RF[k].name, c, KC[kd].cls, (uint32_t)KC[kd].n, kd < 3 ? TOK_REAL : TOK_REAL2,
               (uint32_t)(kd < 3 ? N_TOK_REAL : N_TOK_REAL2), Ls, nrand);
        reads_bounded(RF[k].name, c);
        toks_unload(c);
    }
    toks_unload(l3s);
    toks_unload(nos);
    printf("test_split: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
