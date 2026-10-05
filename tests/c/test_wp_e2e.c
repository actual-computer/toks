/*
 * test_wp_e2e.c: include/toks.h end to end on six pinned WordPiece tokenizer.json files (test_wp_e2e.py lists them)
 * against hf tokenizers 0.23.2's own outputs (test_wp_e2e.inc): encode in modes ALL / NONSPECIAL / NONE with and
 * without the post-processor (truncation, the template and padding included), pieces in the normalized stream,
 * every capacity on each case with a sentinel past it, scratch sizes at the boundary, and texts flush against a
 * guard page.
 *
 * The files live under $TOKS_TOKENIZER_CACHE or ~/.cache/toks/tokenizers (tests/wordpiece/fetch.py). A missing
 * file, or one whose sha-256 is not the pin's, prints SKIP and fails nothing.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "toks.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_wp_e2e.inc"

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static const char *path_of(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    static char buf[1024];
    if (root != NULL && root[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", root, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
}

/* the expected values of case c at index i (a trailing pad run is stored as a count) */
static uint32_t want_at(const wpe_case *c, uint32_t i)
{
    return i < c->stored ? WPE_IDS[c->off + i] : WPE_PIN[c->pin].pad_id;
}

static int64_t call(const toks_ctx *ctx, const wpe_case *c, const void *text, uint64_t len, uint32_t *out,
                    uint64_t cap, void *scr)
{
    return c->op == 0 ? toks_encode(ctx, text, len, c->flags, out, cap, scr)
                      : toks_pieces(ctx, text, len, c->flags, out, cap, scr);
}

static void run_pin(uint32_t p)
{
    const wpe_pin *pin = &WPE_PIN[p];
    toks_ctx *ctx = NULL;
    toks_diag dg;
    toks_load_opts o = { sizeof o, 0, 0, 0, &dg };
    int64_t r = toks_load(&ctx, path_of(pin->name), &o);
    if (r == TOKS_E_OPEN) { printf("SKIP %s (%s absent)\n", pin->name, path_of(pin->name)); return; }
    CHECK(r == 0, "%s: load %lld (%s)", pin->name, (long long)r, dg.what);
    if (r != 0) { return; }
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    CHECK(toks_get_info(ctx, &info) == 0 && info.algorithm == TOKS_ALGO_WORDPIECE, "%s: info", pin->name);
    char hex[65];
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", info.source_sha256[i]); }
    if (strcmp(hex, pin->sha256) != 0) {
        printf("SKIP %s (sha256 %s, the pin is %s)\n", pin->name, hex, pin->sha256);
        toks_unload(ctx);
        return;
    }
    uint64_t max_len = 0;
    for (size_t t = 0; t < sizeof WPE_TEXT_LEN / sizeof WPE_TEXT_LEN[0]; t++) {
        if (WPE_TEXT_LEN[t] > max_len) { max_len = WPE_TEXT_LEN[t]; }
    }
    uint64_t sb = toks_scratch_bytes(ctx, max_len, 0);
    void *scr = malloc(sb);
    CHECK(scr != NULL && toks_scratch_init(ctx, scr, sb, 0) == 0, "%s: scratch", pin->name);
    uint32_t *out = (uint32_t *)malloc(4u * 4096u);
    uint64_t n_cases = 0;
    for (size_t i = 0; i < sizeof WPE_CASE / sizeof WPE_CASE[0]; i++) {
        const wpe_case *c = &WPE_CASE[i];
        if (c->pin != p) { continue; }
        n_cases++;
        const char *text = WPE_TEXT[c->text];
        uint64_t len = WPE_TEXT_LEN[c->text];
        /* every capacity in 0 .. n + 1 for short results, a spread for long ones; a sentinel right past cap */
        uint32_t caps[24];
        uint32_t nc = 0;
        if (c->n <= 18u) {
            for (uint32_t k = 0; k <= c->n + 1u; k++) { caps[nc++] = k; }
        } else {
            uint32_t pick[] = { 0, 1, 2, c->n / 3u, c->n / 2u, c->n - 2u, c->n - 1u, c->n, c->n + 1u, c->n + 64u };
            for (uint32_t k = 0; k < sizeof pick / sizeof pick[0]; k++) { caps[nc++] = pick[k]; }
        }
        for (uint32_t k = 0; k < nc; k++) {
            uint64_t cap = caps[k];
            if (cap + 1u > 4096u) { continue; }
            for (uint64_t j = 0; j <= cap; j++) { out[j] = 0xA5A5A5A5u; }
            int64_t n = call(ctx, c, text, len, out, cap, scr);
            int ok = n == (int64_t)c->n && out[cap] == 0xA5A5A5A5u;
            uint64_t w = cap < c->n ? cap : c->n;
            for (uint64_t j = 0; ok && j < w; j++) { ok = out[j] == want_at(c, (uint32_t)j); }
            if (!ok) {
                CHECK(0, "%s text %u %s flags %u cap %llu: got %lld, want %u", pin->name, c->text,
                      c->op ? "pieces" : "encode", c->flags, (unsigned long long)cap, (long long)n, c->n);
                break;
            }
            checks++;
        }
        /* the same call with the text flush against a guard page, either side */
        for (int where = 0; where < 2; where++) {
            guard_buf g;
            uint8_t *gt = guard_alloc(&g, len ? len : 1, where == 0 ? GUARD_END : GUARD_START, 0);
            if (gt == NULL) { CHECK(0, "guard_alloc"); break; }
            if (where == 0 && len == 0) { gt += 1; }
            memcpy(gt, text, len);
            int64_t n = call(ctx, c, gt, len, out, 4096, scr);
            int ok = n == (int64_t)c->n;
            for (uint64_t j = 0; ok && j < c->n && j < 4096u; j++) { ok = out[j] == want_at(c, (uint32_t)j); }
            CHECK(ok, "%s text %u flags %u: guard %d", pin->name, c->text, c->flags, where);
            guard_free(&g);
        }
    }
    /* a scratch laid out for len - 1 bytes refuses a len-byte text; for len it runs */
    uint64_t L = WPE_TEXT_LEN[8];
    uint64_t sb2 = toks_scratch_bytes(ctx, L, 0);
    void *s2 = malloc(sb2);
    CHECK(toks_scratch_init(ctx, s2, sb2, 0) == 0 && toks_encode(ctx, WPE_TEXT[8], L, 0, out, 4096, s2) >= 0,
          "%s: exact scratch", pin->name);
    CHECK(toks_scratch_init(ctx, s2, toks_scratch_bytes(ctx, L - 1u, 0), 0) == 0 &&
          toks_encode(ctx, WPE_TEXT[8], L, 0, out, 4096, s2) == TOKS_E_SCRATCH, "%s: short scratch", pin->name);
    free(s2);
    printf("%-16s ok: %llu cases (n_ids %u, %u added)\n", pin->name, (unsigned long long)n_cases, info.n_ids, info.n_added);
    free(out);
    free(scr);
    toks_unload(ctx);
}

int main(void)
{
    for (uint32_t p = 0; p < sizeof WPE_PIN / sizeof WPE_PIN[0]; p++) { run_pin(p); }
    printf("test_wp_e2e: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
