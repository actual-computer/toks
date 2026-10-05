/* test_cuts.c: run_cuts (api.c; docs/notes/c-core.md §api.c.3, docs/models/kimi.md §2.3) against the per-code-point
 * walk it speeds up, copied below verbatim from api.c @ d11f9d1: the same cuts on every text. api.c is compiled into
 * this test (as test_memo_hash does) and run on a zeroed context with only the cut limits set: no template, so
 * run_piece emits each cut piece's end and the ends are the cuts. Small limits make every edge frequent: a word of 8
 * ascii atoms across a run's limit or a chunk's end, runs of each str.isspace() class, multi-byte spaces and
 * non-spaces, ill-formed bytes (one non-space atom each), segments right at the skip's min(cut_run, cut_chunk) bytes;
 * then kimi's own 25,000 / 400,000 on long texts. */
#include "../../src/core/api.c"

#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 20) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* the walk at d11f9d1 (no skip, no word path): the ends of the cut pieces */
static uint64_t ref_cuts(const uint8_t *t, uint64_t len, uint32_t cut_chunk, uint32_t cut_run, uint32_t *out)
{
    uint64_t pos = 0u, n = 0u;
    uint32_t in_chunk = 0u, run = 0u;
    int sp_run = 0;
    while (pos < len) {
        uint32_t k = toks_utf8_len(t + pos, len - pos);
        uint32_t cp = 0xFFFFFFFFu;
        if (k == 1u) {
            cp = t[pos];
        } else if (k > 1u) {
            cp = toks_cp_decode(t + pos, k);
        } else {
            k = 1u;
        }
        int sp = py_isspace(cp);
        if (in_chunk == cut_chunk) {
            out[n++] = (uint32_t)pos;
            in_chunk = 0u;
        }
        if (in_chunk == 0u || sp != sp_run) {
            run = 1u;
            sp_run = sp;
        } else if (++run > cut_run) {
            out[n++] = (uint32_t)pos;
            run = 1u;
        }
        in_chunk++;
        pos += k;
    }
    if (len != 0u) { out[n++] = (uint32_t)len; }
    return n;
}

static const struct { const char *s; uint32_t n; } ATOM[] = {
    { "a", 1 }, { "x", 1 }, { "!", 1 }, { "1", 1 }, { "\x7f", 1 }, { "\x01", 1 }, { "\x0e", 1 }, { "\x1b", 1 },
    { "\x08", 1 }, { "\x21", 1 }, { "\0", 1 },
    { " ", 1 }, { "\t", 1 }, { "\n", 1 }, { "\r", 1 }, { "\x0b", 1 }, { "\x0c", 1 }, { "\x1c", 1 }, { "\x1d", 1 },
    { "\x1e", 1 }, { "\x1f", 1 },
    { "\xc2\x85", 2 }, { "\xc2\xa0", 2 }, { "\xe1\x9a\x80", 3 }, { "\xe2\x80\x80", 3 }, { "\xe2\x80\x8a", 3 },
    { "\xe2\x80\xa8", 3 }, { "\xe2\x80\xa9", 3 }, { "\xe2\x80\xaf", 3 }, { "\xe2\x81\x9f", 3 }, { "\xe3\x80\x80", 3 },
    { "\xc3\xa9", 2 }, { "\xe4\xb8\xad", 3 }, { "\xe2\x82\xac", 3 }, { "\xf0\x9f\x98\x80", 4 }, { "\xc2\xa1", 2 },
    { "\xe1\x9a\x81", 3 }, { "\xe2\x80\x8b", 3 }, { "\xe3\x80\x81", 3 },
    { "\x80", 1 }, { "\xbf", 1 }, { "\xc0", 1 }, { "\xc1", 1 }, { "\xf5", 1 }, { "\xff", 1 }, { "\xe4\xb8", 2 },
    { "\xed\xa0\x80", 3 }, { "\xe0\x80\x80", 3 }, { "\xf4\x90\x80\x80", 4 }, { "\xc2", 1 },
};
#define N_ATOM (sizeof ATOM / sizeof ATOM[0])
#define N_ASCII 21u                                     /* ATOM[0 .. 21): one-byte, 11 non-space then 10 spaces */

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(uint32_t n)
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)((rng >> 11) % n);
}

/* a text of about want bytes: runs of one atom, mostly short, often ascii, some around the run limit */
static uint64_t gen(uint8_t *t, uint64_t cap, uint64_t want, uint32_t cut_run)
{
    uint64_t n = 0u;
    while (n < want) {
        uint32_t a = rnd(4u) != 0u ? rnd(N_ASCII) : rnd((uint32_t)N_ATOM);
        uint32_t r = rnd(8u) == 0u ? (cut_run > 9u ? cut_run - 9u : 0u) + rnd(20u) : 1u + rnd(rnd(4u) == 0u ? 24u : 3u);
        for (uint32_t i = 0u; i < r && n + ATOM[a].n <= cap; i++) {   /* bound: r */
            memcpy(t + n, ATOM[a].s, ATOM[a].n);
            n += ATOM[a].n;
        }
        if (n + 4u > cap) { break; }
    }
    return n;
}

static uint32_t pieces[1u << 20], want_cuts[1u << 20];

static void one(toks_ctx *ctx, toks_scratch *h, const uint8_t *t, uint64_t len)
{
    emit e = { pieces, sizeof pieces / 4u, 0u, UINT64_MAX };
    run_cuts(ctx, h, t, len, 0u, 0, &e, 0, 1);
    uint64_t m = ref_cuts(t, len, ctx->cut_chunk, ctx->cut_run, want_cuts);
    int same = e.n == m && memcmp(pieces, want_cuts, (size_t)m * 4u) == 0;
    uint64_t i = 0u;
    while (same == 0 && i < m && i < e.n && pieces[i] == want_cuts[i]) { i++; }   /* bound: m */
    CHECK(same, "len %" PRIu64 " chunk %u run %u: %" PRIu64 " cuts, want %" PRIu64 "; first diff #%" PRIu64 ": %u, want %u",
          len, ctx->cut_chunk, ctx->cut_run, e.n, m, i, i < e.n ? pieces[i] : 0u, i < m ? want_cuts[i] : 0u);
}

int main(void)
{
    static toks_ctx ctx;                               /* zero: no template, no added tokens, no spm, no generic */
    static uint64_t scr[(TOKS_SCR_HDR + 64u) / 8u];
    toks_scratch *h = (toks_scratch *)(void *)scr;     /* run_text's ends[0] lands after its header */
    guard_buf gb, gs;                                  /* texts flush against a no-access page: end, or start */
    uint8_t *g = guard_alloc(&gb, 1u << 20, GUARD_END, 0u), *s = guard_alloc(&gs, 1u << 20, GUARD_START, 0u);
    uint8_t *t = (uint8_t *)malloc(1u << 20);
    if (g == NULL || s == NULL || t == NULL) { return 1; }
    uint64_t flip = 0u;
#define ONE(len) do { uint8_t *p_ = (flip++ & 1u) != 0u ? s : g + (1u << 20) - (len); \
                      memcpy(p_, t, (size_t)(len)); one(&ctx, h, p_, (len)); } while (0)
    for (uint32_t it = 0u; it < 200000u; it++) {       /* bound: 200,000 small texts with small limits */
        ctx.cut_run = 1u + rnd(48u);
        ctx.cut_chunk = rnd(4u) == 0u ? 1u + rnd(400u) : 9u + rnd(4000u);
        uint32_t lim = ctx.cut_run < ctx.cut_chunk ? ctx.cut_run : ctx.cut_chunk;
        uint64_t want = rnd(3u) == 0u ? (uint64_t)lim + rnd(5u) - (lim >= 2u ? 2u : lim) : rnd(2000u);   /* at the skip */
        uint64_t len = gen(t, want + 8u, want, ctx.cut_run);
        if (rnd(3u) == 0u && len > want) { len = want; }   /* cut anywhere, inside an atom too */
        ONE(len);
    }
    ctx.cut_run = 25000u;                              /* kimi's own limits on long texts */
    ctx.cut_chunk = 400000u;
    for (uint32_t it = 0u; it < 24u; it++) {           /* bound: 24 */
        uint64_t len = gen(t, 1u << 20, 380000u + rnd(600000u), ctx.cut_run);
        ONE(len);
    }
    uint64_t n = 0u;                                   /* one class past 25,000 in 8-byte words, both edges */
    for (uint32_t i = 0u; i < 60011u; i++) { t[n++] = (uint8_t)" a"[(i / 25003u) & 1u]; }
    ONE(n);
    guard_free(&gb);
    guard_free(&gs);
    free(t);
    printf("test_cuts: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
