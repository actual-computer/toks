/*
 * test_norm_driver.c: the driver's NFC path (api.c run_gap executing the stretch plan of norm.c) on the four
 * template fixtures of tests/data/compile with their normalizer set to NFC, and on the real Qwen 3.8 file when
 * it is cached (~/.cache/toks/tokenizers/qwen38 or $TOKS_TOKENIZER_CACHE/qwen38).
 *
 * The property: the pieces and ids of a text T equal those of N = NFC(T), normalized whole by toks_nfc (the
 * normalizer test_norm checks against hf). N is already NFC, so it is scanned in place; T goes through the plan's
 * stretches. Equality checks that every cut is a K3 restart point (pieces) and an NFC boundary (bytes) and that
 * the offsets add up. Texts are drawn from atoms built to sit on both sides of restart points and changed runs:
 * letters, digits and punctuation of every base class, whitespace runs, CRLF, contractions, invalid bytes,
 * combining marks, decomposed sequences, singletons (three of them to ascii), hangul jamo, exclusions, mark runs
 * longer than TOKS_NFC_SCAN_RUN. Mode ALL joins parts with added tokens and normalizes each part alone (hf: every
 * gap is normalized on its own); llama3style's normalized: true tokens take the phase-1 path. No atom is U+0307 or
 * 'G': NFC would make llama3style's phase-0 token "ĠĠ" out of them, which hf (matching raw text) does not see.
 * Also: toks_nfc_scan(T) finds no changed run exactly when N == T (up to runs longer than TOKS_NFC_SCAN_RUN, which
 * it reports as changed), and an already-NFC text leaves every byte of the scratch's norm region untouched (zero
 * copies: maintainer doctrine, speed).
 */
#include "toks.h"
#include "core.h"
#include "norm.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* unchanged atoms of every base class, and invalid bytes */
static const char *const PLAIN[] = {
    "a", "b", "s", "t", "r", "e", "l", "v", "m", "d", "S", "T", "x", "Z",
    "\xc3\xa9", "\xd0\xb6", "\xe4\xb8\xad", "\xd5\xa1", "\xd7\xa9", "\xe0\xb8\x81", "\xea\xb0\x80", "\xc5\xbf",
    "0", "7", "\xd9\xa3", "\xc2\xb2", "\xe2\x85\xab",
    ".", ",", "!", "'", "\"", "-", "(", "<", "=", "|", "?",
    "\xe3\x80\x82", "\xef\xbc\x8c", "\xe2\x82\xac", "\xf0\x9f\x98\x80", "\xe2\x9c\x93",
    " ", " ", " ", " ", "  ", "\t", "\n", "\r\n", "\n\n", " \n", "\xc2\xa0", "\xe3\x80\x80", "\xe2\x80\xa8",
    "'s", "'re", "'ll", "'S", "'ve", "'\xc5\xbf",
    "\xff", "\xc3", "\x80", "\xe2\x82", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xc0\xaf",
};
#define N_PLAIN ((uint32_t)(sizeof PLAIN / sizeof PLAIN[0]))

/* atoms NFC changes, or that change with their neighbours */
static const char *const HOT[] = {
    "\xcc\x81", "\xcc\x80", "\xcc\xa3", "\xcc\x88", "\xcc\xa7", "\xcc\x8a", "\xcc\xb8", "\xcc\x83",
    "\xcd\x84", "\xcd\x80", "\xe0\xa4\xbc", "\xe0\xae\xbe", "\xe3\x82\x99", "\xd6\xb4",
    "e\xcc\x81", "A\xcc\x8a", "a\xcc\xa3\xcc\x82", "\xe1\x84\x80\xe1\x85\xa1", "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8",
    "\xea\xb0\x80\xe1\x86\xa8", "\xe3\x81\x8b\xe3\x82\x99", "\xe1\x85\xa1", "\xe1\x86\xa8",
    "\xe2\x84\xab", "\xe2\x84\xa6", "\xe2\x84\xaa", "\xcd\xbe", "\xe1\xbf\xaf", "\xe2\x80\x80", "\xe2\x80\x81",
    "\xef\xa4\x80", "\xe0\xa5\x98", "\xf0\x9d\x85\x9e", "\xf0\x9d\x85\xa0", "\xe2\xab\x9c",
    "\xc3\xa9\xcc\xa3", "\xe1\xba\xb9\xcc\x81", "n\xcc\x83", "<\xcc\xb8", "=\xcc\xb8", "\xe0\xaf\x86\xe0\xae\xbe",
};
#define N_HOT ((uint32_t)(sizeof HOT / sizeof HOT[0]))

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(uint32_t n)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)((rng >> 11) % n);
}

#define MAXT 65536u

/* a random text: atoms, hot with probability hot/1000, sometimes a mark run past TOKS_NFC_SCAN_RUN */
static uint64_t gen(uint8_t *t, uint32_t atoms, uint32_t hot)
{
    uint64_t n = 0;
    for (uint32_t i = 0; i < atoms; i++) {                              /* bound: atoms */
        const char *a;
        if (rnd(400u) == 0u) {                  /* a long mark run behind a letter: shuffled, or already NFC */
            uint32_t k = 30u + rnd(60u), same = rnd(2u);
            if (n + 1u + 2u * k > MAXT - 64u) { break; }
            t[n++] = same ? 'x' : 'o';          /* x + U+0301...: SECOND marks that compose with nothing */
            for (uint32_t j = 0; j < k; j++) { memcpy(t + n, HOT[same ? 0u : rnd(8u)], 2); n += 2u; }   /* bound: k */
            continue;
        }
        a = rnd(1000u) < hot ? HOT[rnd(N_HOT)] : PLAIN[rnd(N_PLAIN)];
        uint64_t k = strlen(a);
        if (n + k > MAXT - 64u) { break; }
        memcpy(t + n, a, (size_t)k);
        n += k;
    }
    return n;
}

static uint64_t nfc_whole(const uint8_t *t, uint64_t n, uint8_t *out)
{
    int64_t m = toks_norm(TOKS_NS_NFC, t, n, out, TOKS_NORM_BOUND(TOKS_NS_NFC, n));
    CHECK(m >= 0, "toks_norm %" PRId64, m);
    return m < 0 ? 0u : (uint64_t)m;
}

/* the scratch's norm region (core.h layout: the header first, at the first 64-aligned address; api.c run_gap) */
static uint8_t *norm_region(void *scr, uint64_t *bytes)
{
    toks_scratch *h = toks_scr_header(scr);                 /* the header (core.h: the scratch's first region) */
    *bytes = toks_scr_norm(h->max_len, TOKS_NFC_X);
    return (uint8_t *)scr + h->off_bounce + toks_scr_bounce(toks_scr_tmax(h->max_len, TOKS_NFC_X));
}

typedef struct bufs {
    uint8_t *t, *n;
    uint32_t *a, *b;
    void *scr;
    long texts, changed, stretches2, zero_copy, conservative;
    uint64_t in_bytes, mat_bytes;
} bufs;

/* pieces and ids of t[0, tn) and n[0, nn) agree under flags */
static void same(const toks_ctx *ctx, bufs *B, const uint8_t *t, uint64_t tn, const uint8_t *n, uint64_t nn,
                 uint32_t flags, const char *what)
{
    uint64_t cap = 3u * MAXT;
    int64_t x = toks_pieces(ctx, t, tn, flags & 3u, B->a, cap, B->scr);
    int64_t y = toks_pieces(ctx, n, nn, flags & 3u, B->b, cap, B->scr);
    int ok = x >= 0 && x == y && memcmp(B->a, B->b, (size_t)x * 4u) == 0;
    CHECK(ok, "%s pieces flags %u: %" PRId64 " vs %" PRId64 " (text %" PRIu64 " -> %" PRIu64 " bytes)", what, flags, x,
          y, tn, nn);
    if (!ok && failures < 3) {
        fprintf(stderr, "  text:");
        for (uint64_t i = 0; i < tn && i < 200u; i++) { fprintf(stderr, " %02x", t[i]); }
        fprintf(stderr, "\n");
        for (int64_t i = 0; i < x && i < y; i++) {
            if (B->a[i] != B->b[i]) { fprintf(stderr, "  first diff at piece %" PRId64 ": %u vs %u\n", i, B->a[i], B->b[i]); break; }
        }
    }
    x = toks_encode(ctx, t, tn, flags | TOKS_NO_POSTPROCESS, B->a, cap, B->scr);
    y = toks_encode(ctx, n, nn, flags | TOKS_NO_POSTPROCESS, B->b, cap, B->scr);
    CHECK(x >= 0 && x == y && memcmp(B->a, B->b, (size_t)x * 4u) == 0, "%s encode flags %u: %" PRId64 " vs %" PRId64,
          what, flags, x, y);
}

/* mode NONE: the whole text is one gap */
static void one_none(const toks_ctx *ctx, const toks_tables *tb, bufs *B, uint64_t tn, const char *what)
{
    uint64_t nn = nfc_whole(B->t, tn, B->n);
    uint64_t re, s = toks_nfc_scan(TOKS_NS_NFC, B->t, tn, 0u, &re);
    int eq = nn == tn && memcmp(B->t, B->n, (size_t)tn) == 0;
    CHECK(s == tn ? eq : (!eq || re - s > TOKS_NFC_SCAN_RUN), "%s scan: changed run at %" PRIu64 " (eq %d)", what, s, eq);
    if (!eq) { B->changed++; }
    if (s != tn && eq) { B->conservative++; }
    /* the plan, as the driver runs it: stretches and bytes materialized */
    toks_nfc_plan pl;
    toks_nfc_plan_begin(&pl, TOKS_NS_NFC, B->t, tn);
    uint64_t a, b, prev = 0;
    int ns = 0;
    while (toks_nfc_plan_next(&pl, tb, B->t, tn, &a, &b)) {             /* bound: <= tn stretches */
        CHECK(a >= prev && b > a && b <= tn, "%s plan: [%" PRIu64 ", %" PRIu64 ") after %" PRIu64, what, a, b, prev);
        prev = b;
        ns++;
        B->mat_bytes += b - a;
    }
    B->in_bytes += tn;
    if (ns >= 2) { B->stretches2++; }
    same(ctx, B, B->t, tn, B->n, nn, TOKS_ADDED_NONE, what);
    /* an already-NFC text copies nothing: the norm region keeps its sentinel */
    uint64_t rn, rs = toks_nfc_scan(TOKS_NS_NFC, B->n, nn, 0u, &rn);
    CHECK(rs == nn || rn - rs > TOKS_NFC_SCAN_RUN, "%s: NFC(T) not NFC at %" PRIu64, what, rs);
    if (rs == nn && nn > 0u) {
        uint64_t rb;
        uint8_t *r = norm_region(B->scr, &rb);
        memset(r, 0xA5, (size_t)rb);
        int64_t k = toks_encode(ctx, B->n, nn, TOKS_ADDED_NONE, B->a, 3u * MAXT, B->scr);
        uint64_t i = 0;
        while (i < rb && r[i] == 0xA5u) { i++; }                        /* bound: rb */
        CHECK(k >= 0 && i == rb, "%s: already-NFC text wrote the norm region at %" PRIu64, what, i);
        B->zero_copy++;
    }
}

/* mode ALL / NONSPECIAL: parts joined by added tokens, each part normalized alone */
static void one_tokens(const toks_ctx *ctx, bufs *B, const char *const *toks, uint32_t ntok, uint32_t hot,
                       const char *what)
{
    uint64_t tn = 0, nn = 0;
    uint32_t parts = 1u + rnd(5u);
    for (uint32_t p = 0; p < parts; p++) {                              /* bound: 5 */
        uint64_t k = gen(B->t + tn, rnd(12u), hot);
        nn += nfc_whole(B->t + tn, k, B->n + nn);
        tn += k;
        if (p + 1u < parts) {
            const char *a = toks[rnd(ntok)];
            uint64_t l = strlen(a);
            memcpy(B->t + tn, a, (size_t)l);
            memcpy(B->n + nn, a, (size_t)l);
            tn += l;
            nn += l;
        }
    }
    same(ctx, B, B->t, tn, B->n, nn, TOKS_ADDED_ALL, what);
    same(ctx, B, B->t, tn, B->n, nn, TOKS_ADDED_NONSPECIAL, what);
}

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n + 64u);
    if (b == NULL || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

/* loads path with "normalizer": null replaced by NFC (nfc_edit), or as it is */
static toks_ctx *load(const char *path, int nfc_edit)
{
    uint64_t n = 0;
    uint8_t *j = slurp(path, &n);
    if (j == NULL) { return NULL; }
    static const char from[] = "\"normalizer\": null", to[] = "\"normalizer\": {\"type\": \"NFC\"}";
    uint8_t *k = (uint8_t *)malloc((size_t)n + sizeof to);
    uint64_t m = n;
    if (nfc_edit) {
        j[n] = 0;
        char *at = strstr((char *)j, from);
        CHECK(at != NULL, "%s: no null normalizer", path);
        if (at == NULL) { free(j); free(k); return NULL; }
        uint64_t pre = (uint64_t)(at - (char *)j), fl = sizeof from - 1u, tl = sizeof to - 1u;
        memcpy(k, j, (size_t)pre);
        memcpy(k + pre, to, (size_t)tl);
        memcpy(k + pre + tl, j + pre + fl, (size_t)(n - pre - fl));
        m = n - fl + tl;
    } else {
        memcpy(k, j, (size_t)n);
    }
    toks_ctx *ctx = NULL;
    int64_t r = toks_load_mem_copy(&ctx, k, m, NULL);
    CHECK(r == 0 && ctx != NULL, "%s: load %" PRId64, path, r);
    free(j);
    free(k);
    return r == 0 ? ctx : NULL;
}

static void run_fixture(const char *name, toks_ctx *ctx, const char *const *toks, uint32_t ntok, long texts)
{
    if (ctx == NULL) { return; }
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = (uint32_t)sizeof info;
    bufs B;
    memset(&B, 0, sizeof B);
    B.t = (uint8_t *)malloc(MAXT);
    B.n = (uint8_t *)malloc(TOKS_NORM_BOUND(TOKS_NS_NFC, MAXT));
    B.a = (uint32_t *)malloc(4u * 3u * MAXT);
    B.b = (uint32_t *)malloc(4u * 3u * MAXT);
    uint64_t sb = toks_scratch_bytes(ctx, TOKS_NORM_BOUND(TOKS_NS_NFC, MAXT), 0u);
    B.scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, B.scr, sb, 0u) == 0, "%s scratch", name);
    const toks_tables *tb = &((const struct toks_ctx *)ctx)->t;
    static const uint32_t HOTS[] = { 0u, 0u, 5u, 20u, 100u, 500u };
    for (long i = 0; i < texts; i++) {                                  /* bound: texts */
        uint32_t atoms = rnd(50u) == 0u ? 300u + rnd(3000u) : rnd(40u);
        uint64_t tn = gen(B.t, atoms, HOTS[rnd(6u)]);
        one_none(ctx, tb, &B, tn, name);
        if (ntok != 0u && (i & 3) == 0) { one_tokens(ctx, &B, toks, ntok, HOTS[rnd(6u)], name); }
    }
    printf("  %-12s %ld texts: %ld changed, %ld with >= 2 stretches, %ld already-NFC zero-copy checks, %ld long runs"
           " reported changed; %.3f bytes materialized per input byte\n", name, texts, B.changed, B.stretches2,
           B.zero_copy, B.conservative, B.in_bytes ? (double)B.mat_bytes / (double)B.in_bytes : 0.0);
    free(B.t); free(B.n); free(B.a); free(B.b); free(B.scr);
    toks_unload(ctx);
}

int main(int argc, char **argv)
{
    long scale = argc > 1 ? atol(argv[1]) : 1;          /* make test: 1 (seconds); lab-host runs: more */
    if (scale < 1) { scale = 1; }
    static const char *const T_GPT2[] = { "<|endoftext|>" };
    static const char *const T_LLAMA[] = { "<|begin|>", "<|end|>", "<|a|>", "hello", "@", "\xc2\xab\xc3\xb1\xc2\xbb", " hi there ",
                                           "\xc3\xa9", "\t" };
    static const char *const T_QWEN[] = { "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>" };
    run_fixture("gpt2style", load("tests/data/compile/gpt2style.json", 1), T_GPT2, 1u, 3000 * scale);
    run_fixture("llama3style", load("tests/data/compile/llama3style.json", 1), T_LLAMA, 9u, 3000 * scale);
    run_fixture("qwen35style", load("tests/data/compile/qwen35style.json", 1), NULL, 0u, 3000 * scale);
    run_fixture("nosplit", load("tests/data/compile/nosplit.json", 1), NULL, 0u, 1500 * scale);

    char path[1024];
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    if (root != NULL && root[0] != 0) { snprintf(path, sizeof path, "%s/qwen38", root); }
    else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/qwen38", getenv("HOME") ? getenv("HOME") : "."); }
    FILE *f = fopen(path, "rb");
    if (f != NULL) {
        fclose(f);
        run_fixture("qwen38", load(path, 0), T_QWEN, 5u, 1500 * scale);
    } else {
        printf("  qwen38       SKIP (no %s)\n", path);
    }
    printf("test_norm_driver: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
