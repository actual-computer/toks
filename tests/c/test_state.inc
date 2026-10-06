/*
 * test_state.inc: SPEC T2 (state independence) as an executable matrix; the body of test_state.c (the library
 * as built) and test_state_hash.c (the degenerate-hash build: api.c and k5_long.c compiled in with TOKS_TEST_DEGEN,
 * so every piece cache -- K5's short cache, the long cache, spm's word cache -- is ONE bucket, the long cache's and
 * the memo's hashes are constants, and only exact compares keep the ids right).
 *
 * Every call of every cell is compared with the COLD result: the scalar tier on a scratch without a memo
 * (TOKS_SCRATCH_MEMO_MIB(0)) whose header was cleared before the call (init then takes its zeroing path), with room
 * for every id. Cells, run per tier (the scalar context and the AUTO one, both from one source):
 *   fresh     flags 0 (the default: a 4 MiB memo), the scratch re-initialized before every call (the O(1) epoch bump)
 *   warm      one scratch without a memo (K5's cache alone), every case in order (K5 turns warm after TOKS_K5_WARM
 *             pieces)
 *   warm32    one scratch, TOKS_SCRATCH_CACHE_MIB(32) (+ the long-piece cache once warm) without a memo, cases + long
 *             pieces
 *   memo      flags 0, the default scratch (its 4 MiB memo): every case, then conversations (growing prompts of turns
 *             >= 256 B between added tokens), every prompt twice; the memo's hits are reported
 *   rebind    the scratch bound to another tokenizer every 8th case (every 1024th in the second half, so it is warm
 *             in between), which encodes the same bytes there, and bound back; prompts likewise
 *   migrate   two memo scratches with their own histories; conversations whose turns move between them
 *   errors    per case a call with cap below the count (the exact prefix, nothing written at or past cap, any out
 *             alignment), now and then a failing call (unknown flags, another context, a text longer than the
 *             scratch holds, NULL out with cap, a text over TOKS_MAX_TEXT), then the same call with room
 *   wrap      the epoch and the long cache's generation driven over their wraps, the long arena wrapped (its fill
 *             position set near its end, and by text), a 1 MiB memo ring lapped; then every epoch bit (22) and
 *             every generation bit (32) flipped after ANOTHER tokenizer's entries on the same bytes
 *   mode      one scratch (memo + 32 MiB), every case and prompt in a shuffled order (flags, modes, pieces mixed)
 *   tiermix   one scratch shared by the scalar and the auto context (one source, one binding), alternating
 *   par       toks_par (pools with flags 0, the default memo, and with no memo + 32 MiB): batches of every text, big
 *             inputs, prompts, full and short caps, vs serial
 * Without arguments: the make-test suite (fixtures + the pinned files present, built-in texts, seconds). With
 * --target: one target of the host matrix (tests/state/run.sh), texts from tests/state/texts.py.
 */
#include "toks.h"
#include "core.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef ST_NAME
#  define ST_NAME "test_state"
#endif

/* ---- cells, counts --------------------------------------------------------------------------------------- */

enum { C_FRESH, C_WARM, C_WARM32, C_MEMO, C_REBIND, C_MIGRATE, C_ERRORS, C_WRAP, C_MODE, C_TIERMIX, C_PAR, C_N };
static const char *const CELL[C_N] = { "fresh", "warm", "warm32", "memo", "rebind", "migrate", "errors", "wrap",
                                       "mode", "tiermix", "par" };
static const char *const TIER[3] = { "scalar", "auto", "mix" };

static uint64_t st_cnt[3][C_N], st_bad[3][C_N], st_hits[3][C_N];
static double st_sec[3][C_N];
static long st_fail;                       /* every mismatch or failed expectation */
static int st_verbose = 1;
static int st_cell, st_tier;               /* what is running (for the messages) */
static const char *st_tname = "";

#define SCHECK(cond, ...) do { if (!(cond)) { st_fail++; st_bad[st_tier][st_cell]++; if (st_fail < 60) { \
    fprintf(stderr, "FAIL %s %s %s: ", st_tname, CELL[st_cell], TIER[st_tier]); fprintf(stderr, __VA_ARGS__); \
    fputc('\n', stderr); } } } while (0)

static uint64_t st_rng = 0x2545F4914F6CDD1Dull;
static uint32_t st_rand(void)
{
    st_rng ^= st_rng << 13;
    st_rng ^= st_rng >> 7;
    st_rng ^= st_rng << 17;
    return (uint32_t)(st_rng >> 16);
}

static void *st_alloc(uint64_t n)
{
    void *p = malloc((size_t)(n != 0u ? n : 1u));
    if (p == NULL) { fprintf(stderr, ST_NAME ": out of memory (%" PRIu64 " bytes)\n", n); exit(2); }
    return p;
}

/* a growable byte buffer */
typedef struct st_buf { uint8_t *p; uint64_t n, cap; } st_buf;
static void st_put(st_buf *b, const void *p, uint64_t n)
{
    if (b->n + n > b->cap) {
        uint64_t c = b->cap != 0u ? b->cap : 4096u;
        while (c < b->n + n) { c *= 2u; }
        uint8_t *q = (uint8_t *)st_alloc(c);
        if (b->n != 0u) { memcpy(q, b->p, (size_t)b->n); }
        free(b->p);
        b->p = q;
        b->cap = c;
    }
    if (n != 0u) { memcpy(b->p + b->n, p, (size_t)n); }
    b->n += n;
}

/* ---- cases and their cold results ---------------------------------------------------------------------- */

typedef struct st_case {
    uint64_t off;          /* the text: st_txt.p + off */
    uint32_t len, flags;
    uint32_t op;           /* 0 encode, 1 pieces */
    uint32_t rsv;
    int64_t  ref_n;        /* the cold result: its count and, at st_ref.p + 4 ref_off, its ids / ends */
    uint64_t ref_off;
} st_case;

typedef struct st_set { st_case *c; uint64_t n, cap; } st_set;

static st_buf st_txt, st_ref;              /* the texts of a target, the cold results (u32s) */
static st_set S_CASE, S_PROMPT, S_BIG, S_LONG;
static uint32_t *st_out, *st_out2;         /* call outputs */
static uint64_t st_out_cap;                /* ids each holds */
static uint8_t *st_R;                      /* the reference scratch (zeroed before every cold call) */
static uint64_t st_Rb, st_Rlen;
static uint64_t st_maxlen;                 /* the longest case / prompt (cell scratches hold it) */

static uint64_t st_text(const uint8_t *p, uint64_t len)
{
    uint64_t off = st_txt.n;
    st_put(&st_txt, p, len);
    return off;
}

static void st_add(st_set *s, uint64_t off, uint64_t len, uint32_t flags, uint32_t op)
{
    if (s->n == s->cap) {
        s->cap = s->cap != 0u ? 2u * s->cap : 1024u;
        st_case *c = (st_case *)st_alloc(s->cap * sizeof *c);
        if (s->n != 0u) { memcpy(c, s->c, (size_t)s->n * sizeof *c); }
        free(s->c);
        s->c = c;
    }
    st_case *c = &s->c[s->n++];
    memset(c, 0, sizeof *c);
    c->off = off;
    c->len = (uint32_t)len;
    c->flags = flags;
    c->op = op;
}

static const uint8_t *st_at(const st_case *c) { return st_txt.p + c->off; }

static int64_t st_call(const toks_ctx *ctx, const uint8_t *p, uint64_t len, uint32_t flags, uint32_t op,
                       uint32_t *out, uint64_t cap, void *scr)
{
    return op != 0u ? toks_pieces(ctx, p, len, flags, out, cap, scr) : toks_encode(ctx, p, len, flags, out, cap, scr);
}

static void st_grow_out(uint64_t need)
{
    if (need <= st_out_cap) { return; }
    uint64_t c = st_out_cap != 0u ? st_out_cap : 65536u;
    while (c < need) { c *= 2u; }
    free(st_out);
    free(st_out2);
    st_out = (uint32_t *)st_alloc((c + 64u) * 4u);
    st_out2 = (uint32_t *)st_alloc((c + 64u) * 4u);
    st_out_cap = c;
}

/* the cold result of one call (ids at st_out): the scalar context on the reference scratch (no memo), header cleared
 * first */
static int64_t st_cold(const toks_ctx *ctx, const uint8_t *p, uint64_t len, uint32_t flags, uint32_t op)
{
    for (;;) {
        memset(toks_scr_header(st_R), 0, TOKS_SCR_HDR);
        int64_t r = toks_scratch_init(ctx, st_R, st_Rb, TOKS_SCRATCH_MEMO_MIB(0));
        if (r != 0) { fprintf(stderr, ST_NAME ": reference init %" PRId64 "\n", r); exit(2); }
        int64_t n = st_call(ctx, p, len, flags, op, st_out, st_out_cap, st_R);
        if (n < 0 || (uint64_t)n <= st_out_cap) { return n; }
        st_grow_out((uint64_t)n);
    }
}

static void st_refs(const toks_ctx *ctx, st_set *s)
{
    for (uint64_t i = 0; i < s->n; i++) {
        st_case *c = &s->c[i];
        c->ref_n = st_cold(ctx, st_at(c), c->len, c->flags, c->op);
        c->ref_off = st_ref.n / 4u;
        if (c->ref_n > 0) { st_put(&st_ref, st_out, 4u * (uint64_t)c->ref_n); }
        if (c->ref_n < 0) {
            fprintf(stderr, "FAIL %s: the cold call returned %" PRId64 " (len %u flags %u op %u)\n", st_tname, c->ref_n,
                    c->len, c->flags, c->op);
            st_fail++;
        }
    }
}

static const uint32_t *st_refp(const st_case *c) { return (const uint32_t *)(const void *)st_ref.p + c->ref_off; }

static void st_hex(const uint8_t *p, uint64_t len)
{
    uint64_t k = len < 256u ? len : 256u;
    for (uint64_t i = 0; i < k; i++) { fprintf(stderr, "%02x", p[i]); }
    if (k < len) { fprintf(stderr, "... (%" PRIu64 " bytes)", len); }
    fputc('\n', stderr);
}

/* got (count n, its first min(cap, n) entries at out) against c's cold result */
static int st_cmp(const st_case *c, int64_t n, const uint32_t *out, uint64_t cap, const char *what)
{
    st_cnt[st_tier][st_cell]++;
    uint64_t k = c->ref_n < 0 ? 0u : (uint64_t)c->ref_n;
    if (k > cap) { k = cap; }
    if (n == c->ref_n && (k == 0u || memcmp(out, st_refp(c), (size_t)(4u * k)) == 0)) { return 1; }
    uint64_t d = 0;
    if (n >= 0) {
        uint64_t m = (uint64_t)n < k ? (uint64_t)n : k;
        while (d < m && out[d] == st_refp(c)[d]) { d++; }
    }
    SCHECK(0, "%s: %s len %u flags %u cap %" PRIu64 ": got %" PRId64 ", cold %" PRId64 " (first difference at %" PRIu64 ")",
           what, c->op ? "pieces" : "encode", c->len, c->flags, cap, n, c->ref_n, d);
    if (st_fail < 60) {
        fprintf(stderr, "  text ");
        st_hex(st_at(c), c->len);
    }
    return 0;
}

/* one call of c on scr through ctx, room for every id, compared */
static int st_do(const toks_ctx *ctx, void *scr, const st_case *c, const char *what)
{
    int64_t n = st_call(ctx, st_at(c), c->len, c->flags, c->op, st_out, st_out_cap, scr);
    return st_cmp(c, n, st_out, st_out_cap, what);
}

/* ---- scratches ----------------------------------------------------------------------------------------- */

typedef struct st_scr { uint8_t *mem, *p; uint64_t bytes; uint32_t flags; } st_scr;

/* a scratch for texts of st_maxlen bytes under each of the contexts given, at an odd offset (any alignment) */
static void st_scr_new(st_scr *s, uint32_t flags, const toks_ctx *a, const toks_ctx *b, uint64_t maxlen)
{
    uint64_t n = toks_scratch_bytes(a, maxlen, flags);
    if (b != NULL && toks_scratch_bytes(b, maxlen, flags) > n) { n = toks_scratch_bytes(b, maxlen, flags); }
    s->mem = (uint8_t *)st_alloc(n + 64u);
    s->p = s->mem + 1u + st_rand() % 63u;
    s->bytes = n;
    s->flags = flags;
}

static void st_scr_init(st_scr *s, const toks_ctx *ctx)
{
    int64_t r = toks_scratch_init(ctx, s->p, s->bytes, s->flags);
    SCHECK(r == 0, "toks_scratch_init returned %" PRId64, r);
}

static void st_scr_free(st_scr *s) { free(s->mem); }

static uint64_t st_memo_hits(st_scr *s)
{
    uint64_t pos, h;
    toks_scr_memo_ctr(toks_scr_header(s->p), &pos, &h);
    return h;
}

/* ---- a target ------------------------------------------------------------------------------------------ */

typedef struct st_tok {
    const char *name;
    toks_ctx *s, *a;                       /* the scalar and the AUTO context of one source */
    uint32_t tier_a;
} st_tok;

#define ST_LIT 48
static uint8_t st_lit[ST_LIT][256];        /* the target's added-token contents (phase 0 first) */
static uint32_t st_lit_len[ST_LIT], st_nlit;

static void st_literals(const toks_ctx *ctx)
{
    const toks_tables *t = &ctx->t;
    st_nlit = 0;
    for (uint32_t ph = 0; ph < 2u; ph++) {
        uint64_t n = t->add_n, step = n > 32u ? n / 24u : 1u;
        for (uint64_t i = 0; i < n && st_nlit < ST_LIT; i++) {     /* the first 16 of a phase, then a spread */
            if (i >= 16u && i % step != 0u) { continue; }
            const toks_added_entry *e = &t->add_entries[i];
            if (e->phase != ph || e->len == 0u || e->len > 255u) { continue; }
            memcpy(st_lit[st_nlit], t->add_bytes + e->off, e->len);
            st_lit_len[st_nlit++] = e->len;
        }
    }
}

/* a text from 1..k fragments of the pool, plus the literals */
static const char *const FRAG[] = {
    "the", " the", " memo", " token", "izer", ",", ".", " ", "  ", "   ", "\n", "\n\n", "\r\n", "\t", "'s", "'ll", " 42",
    "7", "2026", " 3.14159", " caf\xc3\xa9", " e\xcc\x81t\xc3\xa9", " \xe4\xb8\xad\xe6\x96\x87", "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",
    "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4", " \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82", " \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7",
    " \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87", "\xf0\x9f\x98\x80", "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb",
    "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", " zebra", "Quick", " BROWN", "!?", " (x)", "def f(x):\n    return x",
    "{\"k\": [1, 2]}", "<", "|", ">", "<|", "|>", "</", "[", "]", "\xff", "\x80", "\xc0\xaf", "\xed\xa0\x80", "a\xcc", "\x00",
    "\xe2\x96\x81", "\xc2\xa0", "\xe3\x80\x80", "\xe2\x80\xa8", "\xcd\x84", "\xf0\x9d\x85\xa0", "ab\xcc\x81\xcc\x96", "    ",
    "        return", "https://example.com/a?b=c", "x_y_z", "AbCdEf", "\xe2\x82\xac", "\xc3\x9f"
};
#define NFRAG (sizeof FRAG / sizeof FRAG[0])

static uint64_t st_gen(uint8_t *p, uint32_t k, int lits)
{
    uint64_t n = 0;
    for (uint32_t i = 0; i < k; i++) {
        if (lits && st_nlit != 0u && st_rand() % 4u == 0u) {
            uint32_t j = st_rand() % st_nlit;
            memcpy(p + n, st_lit[j], st_lit_len[j]);
            n += st_lit_len[j];
        } else {
            const char *f = FRAG[st_rand() % NFRAG];
            size_t l = f[0] == 0 ? 1u : strlen(f);       /* "\x00" is one NUL byte */
            memcpy(p + n, f, l);
            n += l;
        }
    }
    return n;
}

/* distinct long pieces (K5's long cache): words of 16..40 random letters (or digits / CJK now and then) */
static uint64_t st_gen_long(uint8_t *p, uint64_t want)
{
    uint64_t n = 0;
    while (n < want) {
        uint32_t w = 16u + st_rand() % 25u, kind = st_rand() % 8u;
        for (uint32_t i = 0; i < w; i++) {
            if (kind == 0u) { memcpy(p + n, "\xe4\xb8\x80", 3); p[n + 2] = (uint8_t)(0x80u + st_rand() % 64u); n += 3; }
            else { p[n++] = (uint8_t)(kind == 1u ? '0' + st_rand() % 10u : 'a' + st_rand() % 26u); }
        }
        p[n++] = (uint8_t)(st_rand() % 16u == 0u ? '\n' : ' ');
    }
    return n;
}

static const uint32_t ENC_FLAGS[] = { 0u, 1u, 2u, 4u, 5u, 6u, 12u };      /* modes x post-processing, continuation */
static const uint32_t PIECE_FLAGS[] = { 0u, 1u, 2u, 8u };
#define N_ENC (sizeof ENC_FLAGS / sizeof ENC_FLAGS[0])
#define N_PIECE (sizeof PIECE_FLAGS / sizeof PIECE_FLAGS[0])

static void st_cases_of(uint64_t off, uint64_t len)
{
    for (uint32_t f = 0; f < N_ENC; f++) { st_add(&S_CASE, off, len, ENC_FLAGS[f], 0u); }
    for (uint32_t f = 0; f < N_PIECE; f++) { st_add(&S_CASE, off, len, PIECE_FLAGS[f], 1u); }
}

/* the specials of tests/parity's g_special, from this target's literals: each alone, doubled, after '<', minus its
 * last byte; prefix pairs nested; random strings of literals and fragments */
static void st_special_texts(uint32_t nrand)
{
    uint8_t b[4096];
    for (uint32_t i = 0; i < st_nlit; i++) {
        uint32_t l = st_lit_len[i];
        st_cases_of(st_text(st_lit[i], l), l);
        memcpy(b, st_lit[i], l);
        memcpy(b + l, st_lit[i], l);
        st_cases_of(st_text(b, 2u * l), 2u * l);
        b[0] = '<';
        memcpy(b + 1, st_lit[i], l);
        st_cases_of(st_text(b, l + 1u), l + 1u);
        if (l > 1u) { st_cases_of(st_text(st_lit[i], l - 1u), l - 1u); }
        for (uint32_t j = 0; j < st_nlit && j < 24u; j++) {
            uint32_t m = st_lit_len[j];
            if (i == j || (m <= l ? memcmp(st_lit[i], st_lit[j], m) : memcmp(st_lit[i], st_lit[j], l)) != 0) { continue; }
            uint64_t n = 0;                                  /* b + x + a, a + b, " a b " */
            memcpy(b + n, st_lit[j], m), n += m;
            b[n++] = 'x';
            memcpy(b + n, st_lit[i], l), n += l;
            memcpy(b + n, st_lit[i], l), n += l;
            memcpy(b + n, st_lit[j], m), n += m;
            b[n++] = ' ';
            st_cases_of(st_text(b, n), n);
        }
    }
    for (uint32_t r = 0; r < nrand; r++) {
        uint64_t n = st_gen(b, 1u + st_rand() % 12u, 1);
        st_cases_of(st_text(b, n), n);
    }
}

/* conversations (SPEC §6's evidence: a conversation re-sent turn by turn): turns of >= 256 bytes cut from the
 * cases' texts, each first alone (a segment at the start of a call), then prompt j = every turn i <= j between two
 * literals (the same bytes after a token: another document state). One conversation keeps its flags (a server's
 * calls do; the memo keeps one slot per segment's bytes, whatever the key), the conversations cycle over them */
#define ST_TURNS 24
static void st_conversations(uint32_t nconv, uint32_t turn_max)
{
    static const uint32_t PF[] = { 0u, 4u, 0u, 1u, 2u, 12u, 0u, 5u };
    const uint8_t *la = st_nlit > 0u ? st_lit[0] : (const uint8_t *)"\n\n";
    const uint8_t *lb = st_nlit > 1u ? st_lit[1] : la;
    uint32_t na = st_nlit > 0u ? st_lit_len[0] : 2u, nb = st_nlit > 1u ? st_lit_len[1] : na;
    uint64_t src = 0;
    uint8_t *turn[ST_TURNS];
    uint64_t tl[ST_TURNS];
    st_buf pr = { NULL, 0, 0 };
    for (uint32_t c = 0; c < nconv; c++) {
        for (uint32_t i = 0; i < ST_TURNS; i++) {
            uint64_t want = i % 5u == 4u ? 40u + st_rand() % 150u : 256u + st_rand() % turn_max;
            turn[i] = (uint8_t *)st_alloc(want + 4096u);
            tl[i] = 0;
            for (uint64_t g = 0; tl[i] < want && g < 64u * S_CASE.n; g++) {   /* texts in order, flags 0 copies */
                const st_case *k = &S_CASE.c[src++ % S_CASE.n];
                if (k->op != 0u || k->flags != 0u || k->len > 4000u || (st_rand() & 3u) != 0u) { continue; }
                memcpy(turn[i] + tl[i], st_at(k), k->len);
                tl[i] += k->len;
            }
        }
        for (uint32_t i = 0; i < ST_TURNS; i++) { st_add(&S_PROMPT, st_text(turn[i], tl[i]), tl[i], 0u, 0u); }
        for (uint32_t j = 0; j < ST_TURNS; j++) {
            pr.n = 0;
            for (uint32_t i = 0; i <= j; i++) {
                st_put(&pr, i % 2u ? lb : la, i % 2u ? nb : na);
                st_put(&pr, turn[i], tl[i]);
                st_put(&pr, lb, nb);
            }
            st_add(&S_PROMPT, st_text(pr.p, pr.n), pr.n, PF[c % 8u], 0u);
            if (pr.n > st_maxlen) { st_maxlen = pr.n; }
        }
        for (uint32_t i = 0; i < ST_TURNS; i++) { free(turn[i]); }
    }
    free(pr.p);
}

/* ---- the cells ----------------------------------------------------------------------------------------- */

static void cell_fresh(const toks_ctx *ctx, const toks_ctx *other)
{
    st_scr f;
    st_scr_new(&f, 0u, ctx, other, st_maxlen);
    for (uint64_t i = 0; i < S_CASE.n; i++) {
        st_scr_init(&f, ctx);
        (void)st_do(ctx, f.p, &S_CASE.c[i], "fresh");
    }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) {
        st_scr_init(&f, ctx);
        (void)st_do(ctx, f.p, &S_PROMPT.c[i], "fresh prompt");
    }
    st_scr_free(&f);
}

static void cell_warm(const toks_ctx *ctx, uint32_t flags)
{
    st_scr w;
    st_scr_new(&w, flags, ctx, NULL, st_maxlen);
    st_scr_init(&w, ctx);
    for (uint64_t i = 0; i < S_CASE.n; i++) { (void)st_do(ctx, w.p, &S_CASE.c[i], "warm"); }
    for (uint32_t r = 0; r < 2u; r++) {                 /* long pieces: the second pass hits the long cache */
        for (uint64_t i = 0; i < S_LONG.n; i++) { (void)st_do(ctx, w.p, &S_LONG.c[i], "long pieces"); }
    }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) { (void)st_do(ctx, w.p, &S_PROMPT.c[i], "warm prompt"); }
    for (uint64_t i = 0; i < S_CASE.n; i++) { (void)st_do(ctx, w.p, &S_CASE.c[i], "warm, second pass"); }
    st_hits[st_tier][st_cell] += toks_scr_header(w.p)->hits_cache;     /* K5's cache hits since init */
    st_scr_free(&w);
}

static void cell_memo(const toks_ctx *ctx)
{
    st_scr m;
    st_scr_new(&m, 0u, ctx, NULL, st_maxlen);           /* the default: a 4 MiB memo */
    st_scr_init(&m, ctx);
    for (uint64_t i = 0; i < S_CASE.n; i++) { (void)st_do(ctx, m.p, &S_CASE.c[i], "memo"); }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) {
        (void)st_do(ctx, m.p, &S_PROMPT.c[i], "memo prompt");
        (void)st_do(ctx, m.p, &S_PROMPT.c[i], "memo prompt again");
    }
    st_hits[st_tier][st_cell] += st_memo_hits(&m);
    st_scr_free(&m);
}

/* the other tokenizer encodes the same bytes on the scratch (its own cold result checked), then back */
static void st_visit(const toks_ctx *ctx, const toks_ctx *other, st_scr *b, const st_case *c)
{
    st_scr_init(b, other);
    int64_t want = st_cold(other, st_at(c), c->len, c->flags, c->op);
    uint64_t k = want > 0 ? (uint64_t)want : 0u;
    memcpy(st_out2, st_out, (size_t)(4u * k));
    int64_t n = st_call(other, st_at(c), c->len, c->flags, c->op, st_out, st_out_cap, b->p);
    st_cnt[st_tier][st_cell]++;
    SCHECK(n == want && memcmp(st_out, st_out2, (size_t)(4u * k)) == 0, "the other tokenizer on the rebound scratch: %"
           PRId64 " vs cold %" PRId64 " (len %u flags %u)", n, want, c->len, c->flags);
    st_scr_init(b, ctx);
}

static void cell_rebind(const toks_ctx *ctx, const toks_ctx *other)
{
    st_scr b;
    st_scr_new(&b, TOKS_SCRATCH_MEMO_MIB(1) | TOKS_SCRATCH_CACHE_MIB(4), ctx, other, st_maxlen);
    st_scr_init(&b, ctx);
    for (uint64_t i = 0; i < S_CASE.n; i++) {
        (void)st_do(ctx, b.p, &S_CASE.c[i], "rebind");
        uint64_t period = i < S_CASE.n / 2u ? 8u : 1024u;
        if (other != NULL && i % period == period - 1u) { st_visit(ctx, other, &b, &S_CASE.c[i]); }
    }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) {
        (void)st_do(ctx, b.p, &S_PROMPT.c[i], "rebind prompt");
        if (other != NULL && i % 3u == 2u) {
            st_visit(ctx, other, &b, &S_PROMPT.c[i]);
            (void)st_do(ctx, b.p, &S_PROMPT.c[i], "rebind prompt after the other");
        }
    }
    st_scr_free(&b);
}

static void cell_migrate(const toks_ctx *ctx)
{
    st_scr a, b;
    st_scr_new(&a, TOKS_SCRATCH_MEMO_MIB(4), ctx, NULL, st_maxlen);
    st_scr_new(&b, TOKS_SCRATCH_MEMO_MIB(4) | TOKS_SCRATCH_CACHE_MIB(4), ctx, NULL, st_maxlen);
    st_scr_init(&a, ctx);
    st_scr_init(&b, ctx);
    for (uint64_t i = 0; i < S_CASE.n; i++) {          /* two histories: a in order, b backwards */
        (void)st_do(ctx, a.p, &S_CASE.c[i], "migrate a");
        (void)st_do(ctx, b.p, &S_CASE.c[S_CASE.n - 1u - i], "migrate b");
    }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) {        /* turns 0..11 on a, 12..23 on b, then all on a */
        uint64_t j = i % ST_TURNS;
        (void)st_do(ctx, j < ST_TURNS / 2u ? a.p : b.p, &S_PROMPT.c[i], "migrating");
    }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) { (void)st_do(ctx, a.p, &S_PROMPT.c[i], "migrated back"); }
    st_hits[st_tier][st_cell] += st_memo_hits(&a) + st_memo_hits(&b);
    st_scr_free(&a);
    st_scr_free(&b);
}

/* a call with cap below c's count: the count, the exact prefix, nothing written at or past cap; out at offset mis
 * bytes from 4-byte alignment */
static void st_short(const toks_ctx *ctx, void *scr, const st_case *c, uint64_t cap, uint32_t mis)
{
    uint8_t *raw = (uint8_t *)st_out2;
    uint8_t *o = raw + mis;
    memset(raw, 0xA5, (size_t)(4u * (cap + 16u) + 8u));
    int64_t n = st_call(ctx, st_at(c), c->len, c->flags, c->op, cap != 0u ? (uint32_t *)(void *)o : NULL, cap, scr);
    if (cap != 0u) { memcpy(st_out, o, (size_t)(4u * cap)); }
    (void)st_cmp(c, n, st_out, cap, "short cap");
    for (uint64_t i = 4u * cap + mis; i < 4u * (cap + 16u); i++) {
        if (raw[i] != 0xA5u) { SCHECK(0, "cap %" PRIu64 ": byte %" PRIu64 " past out[cap] written", cap, i - mis); break; }
    }
}

static void cell_errors(const toks_ctx *ctx, const toks_ctx *other)
{
    st_scr e;
    st_scr_new(&e, TOKS_SCRATCH_MEMO_MIB(4) | TOKS_SCRATCH_CACHE_MIB(4), ctx, NULL, st_maxlen);
    st_scr_init(&e, ctx);
    uint64_t room = toks_scr_header(e.p)->max_len;
    uint8_t *big = (uint8_t *)st_alloc(room + 2u);
    memset(big, 'a', (size_t)room + 2u);
    for (uint64_t i = 0; i < S_CASE.n + S_PROMPT.n; i++) {
        const st_case *c = i < S_CASE.n ? &S_CASE.c[i] : &S_PROMPT.c[i - S_CASE.n];
        uint64_t k = c->ref_n > 0 ? (uint64_t)c->ref_n : 0u;
        if (k > 0u) {
            uint64_t cap = i % 3u == 0u ? k - 1u : i % 3u == 1u ? k / 2u : 0u;
            st_short(ctx, e.p, c, cap, (uint32_t)(i % 4u));
        }
        if (i % 4u == 0u) {                               /* a failing call: the documented code, nothing changed */
            int64_t r, want;
            switch ((i / 4u) % 5u) {
            case 0: r = toks_encode(ctx, st_at(c), c->len, c->flags | 64u, st_out, st_out_cap, e.p); want = TOKS_E_ARG; break;
            case 1: r = other != NULL ? st_call(other, st_at(c), c->len, c->flags, c->op, st_out, st_out_cap, e.p) : TOKS_E_SCRATCH;
                    want = TOKS_E_SCRATCH; break;
            case 2: r = st_call(ctx, big, room + 1u, c->flags, c->op, st_out, st_out_cap, e.p); want = TOKS_E_SCRATCH; break;
            case 3: r = st_call(ctx, st_at(c), c->len, c->flags, c->op, NULL, 5u, e.p); want = TOKS_E_ARG; break;
            default: r = st_call(ctx, st_at(c), TOKS_MAX_TEXT + 1u, c->flags, c->op, st_out, st_out_cap, e.p); want = TOKS_E_LIMIT; break;
            }
            st_cnt[st_tier][st_cell]++;
            SCHECK(r == want, "failing call %" PRIu64 " returned %" PRId64 ", not %" PRId64, (i / 4u) % 5u, r, want);
        }
        (void)st_do(ctx, e.p, c, "after a short or failing call");
    }
    free(big);
    st_scr_free(&e);
}

/* cases [from, to) on x, re-initialized every 16 cases (the epoch and generation advance) and counted warm (K5's
 * counters set as if TOKS_K5_WARM pieces had passed, so the long cache takes part at once) */
static void st_wrap_run(const toks_ctx *ctx, st_scr *x, uint64_t from, uint64_t to, const char *what)
{
    for (uint64_t i = from; i < to && i < S_CASE.n; i++) {
        if (i % 16u == 15u) {
            st_scr_init(x, ctx);
            toks_scr_header(x->p)->hits_static = TOKS_K5_WARM;
        }
        (void)st_do(ctx, x->p, &S_CASE.c[i], what);
    }
}

static char st_what[64];

/* a few prompts, a long-piece input and cases on x, compared (check) or not; the scratch counts as warm (K5's
 * counters set as if TOKS_K5_WARM pieces had passed), so the long cache takes part from the first call */
static void st_probe(const toks_ctx *ctx, st_scr *x, int check)
{
    toks_scr_header(x->p)->hits_static = TOKS_K5_WARM;
    for (uint64_t i = 0; i < S_PROMPT.n && i < 3u; i++) {
        if (check) { (void)st_do(ctx, x->p, &S_PROMPT.c[i], st_what); }
        else { (void)st_call(ctx, st_at(&S_PROMPT.c[i]), S_PROMPT.c[i].len, S_PROMPT.c[i].flags, 0u, st_out, st_out_cap, x->p); }
    }
    for (uint64_t i = 0; i < S_LONG.n && i < 1u; i++) {
        if (check) { (void)st_do(ctx, x->p, &S_LONG.c[i], st_what); }
        else { (void)st_call(ctx, st_at(&S_LONG.c[i]), S_LONG.c[i].len, S_LONG.c[i].flags, 0u, st_out, st_out_cap, x->p); }
    }
    for (uint64_t i = 0; i < S_CASE.n && i < 128u; i++) {
        const st_case *c = &S_CASE.c[i];
        if (check) { (void)st_do(ctx, x->p, c, st_what); }
        else { (void)st_call(ctx, st_at(c), c->len, c->flags, c->op, st_out, st_out_cap, x->p); }
    }
}

static void cell_wrap(const toks_ctx *ctx, const toks_ctx *other)
{
    st_scr x;
    st_scr_new(&x, TOKS_SCRATCH_MEMO_MIB(1) | TOKS_SCRATCH_CACHE_MIB(4), ctx, other, st_maxlen);
    toks_scratch *h = toks_scr_header(x.p);
    uint64_t q = S_CASE.n / 4u;
    /* the epoch, then the long generation, over their ends: the other tokenizer's entries sit at epoch (generation)
     * 1, where a counter that wrapped without the zeroing path would come back */
    for (int part = 0; part < 2; part++) {
        memset(h, 0, TOKS_SCR_HDR);
        st_scr_init(&x, other != NULL ? other : ctx);     /* the zeroing path: epoch 1, generation 1 */
        if (other != NULL) { st_probe(other, &x, 0); }
        if (part == 0) { h->epoch = TOKS_TAG_MAX - 4u; } else { h->long_gen = 0xFFFFFFFFu - 4u; }
        st_scr_init(&x, ctx);
        st_wrap_run(ctx, &x, (uint64_t)part * q, (uint64_t)(part + 1) * q, part == 0 ? "epoch wrap" : "generation wrap");
    }
    /* the long arena wrapped (its fill set near its end before each round): generation FFFFFFFE -> FFFFFFFF -> 1 (the
     * buckets emptied) -> 2 -> 3, the other tokenizer's long entries at generation 1 again; the memo ring likewise
     * lapped (its write position set 2 KiB before a lap's end) */
    memset(h, 0, TOKS_SCR_HDR);
    st_scr_init(&x, other != NULL ? other : ctx);
    if (other != NULL) { st_probe(other, &x, 0); }
    h->long_gen = 0xFFFFFFFDu;
    st_scr_init(&x, ctx);
    for (uint32_t r = 0; r < 4u; r++) {
        uint64_t mb;
        toks_memo_head *m = (toks_memo_head *)(void *)toks_scr_memo(h, &mb);   /* NULL: no memo (wordpiece, unigram) */
        if (m != NULL) {
            uint64_t ring = mb - 64u - (mb >> 5);
            m->pos += 2u * ring - m->pos % ring - 2048u;  /* forward, 2 KiB before a lap's end, and the lap */
            m->lap = m->pos - m->pos % ring;              /* ending there: a record that does not fit starts */
            m->run = TOKS_MEMO_DRY - 1u;                  /* the next one (kernels.md §7) */
        }
        h->hits_static = TOKS_K5_WARM;                    /* warm: the long cache takes part */
        if (h->off_long != 0u) { h->long_pos = 3u * ((uint64_t)h->cache_mib << 17) - 96u; }
        for (uint64_t i = 0; i < S_LONG.n; i++) { (void)st_do(ctx, x.p, &S_LONG.c[i], "arena wrap"); }
        for (uint64_t i = 0; i < S_PROMPT.n; i++) { (void)st_do(ctx, x.p, &S_PROMPT.c[i], "ring wrap"); }
        for (uint64_t i = 2u * q + r * q / 4u; i < 2u * q + (r + 1u) * q / 4u && i < S_CASE.n; i++) {
            (void)st_do(ctx, x.p, &S_CASE.c[i], "after the wraps");
        }
    }
    /* every epoch bit, then every generation bit: another tokenizer's entries on the same bytes, then this one at an
     * epoch (generation) that differs from theirs in that bit only. Both are new values (the counters only grow
     * between the zeroing paths that empty everything; an older value would name entries of earlier bindings),
     * so each bit starts from the zeroing path (the header's epoch at TOKS_TAG_MAX) */
    if (other != NULL) {
        for (uint32_t b = 0; b < 22u + 32u; b++) {
            h->epoch = TOKS_TAG_MAX;
            st_scr_init(&x, other);                       /* zeroed: epoch 1, generation 1 */
            int ep = b < 22u;
            uint64_t bit = 1ull << (ep ? b : b - 22u), g = 2u;
            if (!ep && h->off_long == 0u) { continue; }
            if (g & bit) { g = 2u * bit; }               /* the least value > 1 with the bit clear */
            if (ep) { h->epoch = g - 1u; } else { h->long_gen = g - 1u; }
            st_scr_init(&x, other);                       /* the other tokenizer at g */
            st_probe(other, &x, 0);
            if (g + bit >= (ep ? (uint64_t)TOKS_TAG_MAX : 0xFFFFFFFFull)) { continue; }
            if (ep) { h->epoch = g + bit - 1u; } else { h->long_gen = g + bit - 1u; }
            snprintf(st_what, sizeof st_what, "%s bit %u after the other's entries", ep ? "epoch" : "generation",
                     ep ? b : b - 22u);
            st_scr_init(&x, ctx);                         /* this one at g + bit */
            st_probe(ctx, &x, 1);
        }
    }
    st_scr_free(&x);
}

static void cell_mode(const toks_ctx *ctx)
{
    st_scr y;
    st_scr_new(&y, TOKS_SCRATCH_MEMO_MIB(4) | TOKS_SCRATCH_CACHE_MIB(32), ctx, NULL, st_maxlen);
    st_scr_init(&y, ctx);
    uint64_t n = S_CASE.n + 2u * S_PROMPT.n;
    uint64_t *ord = (uint64_t *)st_alloc(n * 8u);
    for (uint64_t i = 0; i < n; i++) { ord[i] = i; }
    for (uint64_t i = n; i > 1u; i--) {                  /* a seeded shuffle */
        uint64_t j = ((uint64_t)st_rand() << 16 ^ st_rand()) % i, t = ord[i - 1u];
        ord[i - 1u] = ord[j];
        ord[j] = t;
    }
    for (uint64_t i = 0; i < n; i++) {
        uint64_t k = ord[i];
        (void)st_do(ctx, y.p, k < S_CASE.n ? &S_CASE.c[k] : &S_PROMPT.c[(k - S_CASE.n) % S_PROMPT.n], "shuffled");
    }
    st_hits[st_tier][st_cell] += st_memo_hits(&y);
    free(ord);
    st_scr_free(&y);
}

static void cell_tiermix(const st_tok *T)
{
    st_scr z;
    st_scr_new(&z, TOKS_SCRATCH_MEMO_MIB(4) | TOKS_SCRATCH_CACHE_MIB(32), T->s, NULL, st_maxlen);
    st_scr_init(&z, T->s);
    for (uint64_t i = 0; i < S_CASE.n; i++) {
        (void)st_do((st_rand() & 1u) ? T->a : T->s, z.p, &S_CASE.c[i], "tiers mixed");
    }
    for (uint64_t i = 0; i < S_LONG.n; i++) { (void)st_do((i & 1u) ? T->a : T->s, z.p, &S_LONG.c[i], "tiers mixed long"); }
    for (uint64_t i = 0; i < S_PROMPT.n; i++) {
        (void)st_do((i & 1u) ? T->a : T->s, z.p, &S_PROMPT.c[i], "tiers mixed prompt");
        (void)st_do((i & 1u) ? T->s : T->a, z.p, &S_PROMPT.c[i], "tiers mixed prompt again");
    }
    st_hits[2][C_TIERMIX] += st_memo_hits(&z);
    st_scr_free(&z);
}

static uint32_t st_threads = 2u;

/* toks_par: every encode case of one flags value as a batch (items with exact and short caps), the prompts and
 * the big inputs as one input each, alternating two pools (scratch flags 0: the default memo; no memo + 32 MiB) */
static void cell_par(const toks_ctx *ctx)
{
    toks_par *p[2] = { NULL, NULL };
    int64_t r0 = toks_par_create(&p[0], ctx, st_threads, 0u);
    int64_t r1 = toks_par_create(&p[1], ctx, st_threads, TOKS_SCRATCH_MEMO_MIB(0) | TOKS_SCRATCH_CACHE_MIB(32));
    SCHECK(r0 == 0 && r1 == 0, "toks_par_create %" PRId64 " %" PRId64, r0, r1);
    if (r0 != 0 || r1 != 0) { toks_par_destroy(p[0]); toks_par_destroy(p[1]); return; }
    enum { CHUNK = 512 };
    toks_par_item it[CHUNK];
    const st_case *ic[CHUNK];
    uint64_t slot[CHUNK + 1];
    for (uint32_t f = 0; f < N_ENC; f++) {
        uint32_t k = 0, round = 0;
        slot[0] = 0;
        for (uint64_t i = 0; i <= S_CASE.n; i++) {
            const st_case *c = i < S_CASE.n ? &S_CASE.c[i] : NULL;
            if (c != NULL && (c->op != 0u || c->flags != ENC_FLAGS[f] || c->ref_n < 0)) { continue; }
            uint64_t need = c != NULL ? (uint64_t)c->ref_n + 4u : 0u;
            if (c == NULL || k == CHUNK || (k > 0u && slot[k] + need > st_out_cap)) {
                if (k > 0u) {
                    for (uint32_t j = 0; j < k; j++) { it[j].out = st_out + slot[j]; }
                    int64_t r = toks_par_encode_batch(p[round & 1u], it, k, ENC_FLAGS[f]);
                    SCHECK(r == 0, "toks_par_encode_batch returned %" PRId64, r);
                    for (uint32_t j = 0; j < k; j++) { (void)st_cmp(ic[j], it[j].n, it[j].out, it[j].cap, "par batch"); }
                    round++;
                }
                k = 0;
                slot[0] = 0;
                if (c == NULL) { break; }
            }
            ic[k] = c;
            it[k].text = st_at(c);
            it[k].len = c->len;
            it[k].cap = (k % 7u == 3u && c->ref_n > 1) ? (uint64_t)c->ref_n / 2u : need;   /* now and then short */
            it[k].n = -1;
            slot[k + 1u] = slot[k] + need;
            k++;
        }
    }
    for (uint32_t s = 0; s < 2u; s++) {
        for (uint64_t i = 0; i < S_BIG.n + S_PROMPT.n; i++) {
            const st_case *c = i < S_BIG.n ? &S_BIG.c[i] : &S_PROMPT.c[i - S_BIG.n];
            if (c->ref_n < 0) { continue; }
            uint64_t cap = (s == 1u && i % 2u == 1u) ? (uint64_t)c->ref_n / 2u : (uint64_t)c->ref_n + 4u;
            st_grow_out(cap + 4u);
            int64_t n = toks_par_encode(p[(i + s) & 1u], st_at(c), c->len, c->flags, st_out, cap);
            (void)st_cmp(c, n, st_out, cap, i < S_BIG.n ? "par big input" : "par prompt");
        }
    }
    toks_par_destroy(p[0]);
    toks_par_destroy(p[1]);
}

/* ---- one target: its texts, the cold results, every cell on both tiers --------------------------------- */

typedef struct st_texts { const uint8_t *p; uint64_t n; } st_texts;   /* records of u32 length + bytes */

static void st_target(st_tok *T, const toks_ctx *other, const st_texts *tx, uint64_t stride, uint32_t nspecial,
                      uint32_t nconv, uint32_t nbig, uint64_t big_len, uint64_t long_len)
{
    st_tname = T->name;
    st_txt.n = 0;
    st_ref.n = 0;
    S_CASE.n = S_PROMPT.n = S_BIG.n = S_LONG.n = 0;
    st_maxlen = 0;
    st_rng = 0x2545F4914F6CDD1Dull;
    st_literals(T->s);
    static uint8_t b[1u << 17];
    if (tx != NULL) {                                     /* the parity quick set's texts (tests/state/texts.py) */
        uint64_t at = 0, k = 0;
        while (at + 4u <= tx->n) {
            uint32_t l;
            memcpy(&l, tx->p + at, 4);
            if (at + 4u + l > tx->n) { break; }
            if (k++ % stride == 0u) { st_cases_of(st_text(tx->p + at + 4u, l), l); }
            at += 4u + l;
        }
    } else {                                              /* built-in: fragments, long runs */
        for (uint32_t i = 0; i < 64u; i++) {
            uint64_t n = st_gen(b, 1u + st_rand() % (i % 8u == 7u ? 300u : 30u), i % 4u == 0u);
            st_cases_of(st_text(b, n), n);
        }
        static const char *const RUN[] = { "a", " ", "\n", "=", "\xe4\xb8\xad", "e\xcc\x81", "0", "\t", "\xff" };
        for (uint32_t i = 0; i < sizeof RUN / sizeof RUN[0]; i++) {
            uint64_t n = 0, l = strlen(RUN[i]);
            while (n + l <= 400u) { memcpy(b + n, RUN[i], l); n += l; }
            st_cases_of(st_text(b, n), n);
        }
    }
    st_special_texts(nspecial);
    for (uint64_t i = 0; i < S_CASE.n; i++) { if (S_CASE.c[i].len > st_maxlen) { st_maxlen = S_CASE.c[i].len; } }
    st_conversations(nconv, 2000u);
    uint8_t *lb = (uint8_t *)st_alloc(long_len + 64u);
    for (uint64_t at = 0; at < long_len; at += 16384u) {  /* distinct long pieces, 16 KiB inputs */
        uint64_t n = st_gen_long(lb, 16384u - 64u);
        st_add(&S_LONG, st_text(lb, n), n, (at / 16384u) % 3u == 2u ? 4u : 0u, 0u);
        if (n > st_maxlen) { st_maxlen = n; }
    }
    free(lb);
    for (uint32_t i = 0; i < nbig; i++) {                 /* big inputs for toks_par: the case texts joined */
        st_buf bb = { NULL, 0, 0 };
        uint64_t want = big_len >> (i % 3u), j = st_rand();
        while (bb.n < want && S_CASE.n != 0u) {
            const st_case *c = &S_CASE.c[j++ % S_CASE.n];
            if (c->op == 0u && c->flags == 0u) { st_put(&bb, st_at(c), c->len); }
        }
        static const uint32_t BF[] = { 0u, 4u, 6u, 1u };
        st_add(&S_BIG, st_text(bb.p, bb.n), bb.n, BF[i % 4u], 0u);
        free(bb.p);
    }
    uint64_t rmax = st_maxlen;
    for (uint64_t i = 0; i < S_BIG.n; i++) { if (S_BIG.c[i].len > rmax) { rmax = S_BIG.c[i].len; } }
    if (rmax > st_Rlen) {                                 /* the reference scratch holds everything */
        free(st_R);
        st_Rlen = rmax;
        st_Rb = toks_scratch_bytes(NULL, rmax, TOKS_SCRATCH_MEMO_MIB(0));
        st_R = (uint8_t *)st_alloc(st_Rb);
    }
    st_grow_out(4u * rmax + 4096u);
    st_refs(T->s, &S_CASE);
    st_refs(T->s, &S_PROMPT);
    st_refs(T->s, &S_LONG);
    st_refs(T->s, &S_BIG);

    memset(st_cnt, 0, sizeof st_cnt);
    memset(st_bad, 0, sizeof st_bad);
    memset(st_hits, 0, sizeof st_hits);
    memset(st_sec, 0, sizeof st_sec);
    for (st_tier = 0; st_tier < 3; st_tier++) {
        const toks_ctx *ctx = st_tier == 0 ? T->s : T->a;
        for (st_cell = st_tier == 2 ? C_TIERMIX : 0; st_cell < (st_tier == 2 ? C_TIERMIX + 1 : C_N); st_cell++) {
            clock_t t0 = clock();
            switch (st_cell) {
            case C_FRESH: cell_fresh(ctx, other); break;
            case C_WARM: cell_warm(ctx, TOKS_SCRATCH_MEMO_MIB(0)); break;
            case C_WARM32: cell_warm(ctx, TOKS_SCRATCH_CACHE_MIB(32) | TOKS_SCRATCH_MEMO_MIB(0)); break;
            case C_MEMO: cell_memo(ctx); break;
            case C_REBIND: cell_rebind(ctx, other); break;
            case C_MIGRATE: cell_migrate(ctx); break;
            case C_ERRORS: cell_errors(ctx, other); break;
            case C_WRAP: cell_wrap(ctx, other); break;
            case C_MODE: cell_mode(ctx); break;
            case C_TIERMIX: if (st_tier == 2) { cell_tiermix(T); } break;
            default: cell_par(ctx); break;
            }
            st_sec[st_tier][st_cell] += (double)(clock() - t0) / CLOCKS_PER_SEC;
        }
    }
    st_tier = 0;

    uint64_t tot = 0, bad = 0;
    for (int t = 0; t < 3; t++) {
        for (int c = 0; c < C_N; c++) {
            if (st_cnt[t][c] == 0u) { continue; }
            tot += st_cnt[t][c];
            bad += st_bad[t][c];
            if (st_verbose > 1) {
                printf("cell %s %s %s %" PRIu64 " %" PRIu64 " hits=%" PRIu64 " cpu=%.2fs\n", T->name, CELL[c],
                       t == 1 ? (T->tier_a == TOKS_TIER_NEON ? "neon" : T->tier_a == TOKS_TIER_AVX2 ? "avx2"
                                 : T->tier_a == TOKS_TIER_AVX512 ? "avx512" : "auto=scalar") : TIER[t],
                       st_cnt[t][c], st_bad[t][c], st_hits[t][c], st_sec[t][c]);
            }
        }
    }
    double sec = 0;
    for (int t = 0; t < 3; t++) { for (int c = 0; c < C_N; c++) { sec += st_sec[t][c]; } }
    printf("  %-24s %7" PRIu64 " texts, %7" PRIu64 " cases + %3" PRIu64 " prompts + %2" PRIu64 " big + %2" PRIu64
           " long: %9" PRIu64 " calls compared, %" PRIu64 " mismatches (cells %.1f s)\n", T->name,
           S_CASE.n / (N_ENC + N_PIECE), S_CASE.n, S_PROMPT.n, S_BIG.n, S_LONG.n, tot, bad, sec);
    fflush(stdout);
}

/* ---- loading ------------------------------------------------------------------------------------------- */

static int st_load(st_tok *T, const char *name)
{
    char path[1400];
    if (name[0] == '@') {
        const char *root = getenv("TOKS_TOKENIZER_CACHE");
        if (root != NULL && root[0] != 0) { snprintf(path, sizeof path, "%s/%s", root, name + 1); }
        else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name + 1); }
    } else {
        snprintf(path, sizeof path, "%s", name);
    }
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.tier = TOKS_TIER_SCALAR;
    T->name = name;
    T->s = T->a = NULL;
    if (toks_load(&T->s, path, &o) != 0) { return 0; }
    o.tier = TOKS_TIER_AUTO;
    if (toks_load(&T->a, path, &o) != 0) { toks_unload(T->s); T->s = NULL; return 0; }
    toks_info in;
    memset(&in, 0, sizeof in);
    in.size = (uint32_t)sizeof in;
    T->tier_a = toks_get_info(T->a, &in) == 0 ? in.tier : 0u;
    return 1;
}

static uint8_t *st_slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = (uint8_t *)st_alloc((uint64_t)(n > 0 ? n : 1));
    if (n > 0 && fread(d, 1, (size_t)n, f) != (size_t)n) { free(d); fclose(f); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return d;
}

/* the make-test suite: fixtures of the bpe paths (byte-level with added-token options, NFC + dropped bytes, spm,
 * truncation) + the pinned files present (cl100k, NFC, spm with and without Metaspace "first", o200k, wordpiece,
 * unigram), each with the other tokenizer its scratch is rebound to: one whose cache entries have its own format
 * (another vocabulary on the same path). The host matrix (tests/state/run.sh) runs every ledger target. */
static const char *const ST_TOKS[][2] = {
    { "tests/data/compile/llama3style.json", "tests/data/breadth/added_opts.json" },
    { "tests/data/breadth/added_opts.json", "tests/data/compile/llama3style.json" },
    { "tests/data/breadth/drop_nfc.json", "tests/data/compile/llama3style.json" },
    { "tests/data/spm/gemma4like.json", "tests/data/spm/mistrallike.json" },
    { "tests/data/spm/mistrallike.json", "tests/data/spm/gemma4like.json" },
    { "tests/data/breadth/trunc_left.json", "tests/data/compile/llama3style.json" },
    { "@llama3", "@qwen38" },
    { "@gemma4", "@mistral-v0.3" },
    { "@mistral-v0.3", "@gemma4" },
#ifndef ST_DEGEN
    { "@qwen38", "@llama3" },
    { "@o200k", "@llama3" },
    { "@wp-bert-uncased", "@wp-minilm-l6" },
    { "@uni_t5base", "@uni_bgem3" },
#endif
};

int main(int argc, char **argv)
{
    const char *target = NULL, *other = NULL, *texts = NULL;
    uint64_t stride = 1;
    uint32_t nspecial = 4000u, nconv = 6u, nbig = 4u;
    uint64_t big_len = 1u << 20, long_len = 1u << 20;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (strcmp(a, "--target") == 0) { target = v; i++; }
        else if (strcmp(a, "--other") == 0) { other = v; i++; }
        else if (strcmp(a, "--texts") == 0) { texts = v; i++; }
        else if (strcmp(a, "--stride") == 0) { stride = strtoull(v, NULL, 10); i++; }
        else if (strcmp(a, "--threads") == 0) { st_threads = (uint32_t)strtoul(v, NULL, 10); i++; }
        else if (strcmp(a, "--conv") == 0) { nconv = (uint32_t)strtoul(v, NULL, 10); i++; }
        else if (strcmp(a, "-v") == 0) { st_verbose = 2; }
        else { fprintf(stderr, "usage: %s [--target T --other U --texts F --stride k --threads k --conv k -v]\n", argv[0]); return 2; }
    }
    if (stride == 0u) { stride = 1u; }
    if (target != NULL) {                                 /* one target of the host matrix (tests/state/run.sh) */
        st_tok T, U;
        st_texts tx = { NULL, 0 };
        uint8_t *tb = texts != NULL ? st_slurp(texts, &tx.n) : NULL;
        tx.p = tb;
        if (texts != NULL && tb == NULL) { fprintf(stderr, ST_NAME ": cannot read %s\n", texts); return 2; }
        if (!st_load(&T, target)) { printf("%s: %s does not load\n", ST_NAME, target); return 2; }
        int ou = other != NULL && st_load(&U, other);
        st_verbose = 2;
        st_target(&T, ou ? U.a : NULL, tb != NULL ? &tx : NULL, stride, nspecial, nconv, nbig, big_len, long_len);
        toks_unload(T.s);
        toks_unload(T.a);
        if (ou) { toks_unload(U.s); toks_unload(U.a); }
        free(tb);
    } else {
        const uint32_t nt = (uint32_t)(sizeof ST_TOKS / sizeof ST_TOKS[0]);
        uint32_t ran = 0;
        for (uint32_t i = 0; i < nt; i++) {
            st_tok T, U;
            if (!st_load(&T, ST_TOKS[i][0])) {
                if (ST_TOKS[i][0][0] != '@') { fprintf(stderr, "FAIL %s: not loadable\n", ST_TOKS[i][0]); st_fail++; }
                else { printf("  SKIP %s (not present)\n", ST_TOKS[i][0]); }
                continue;
            }
            int ou = st_load(&U, ST_TOKS[i][1]);
            if (!ou) { printf("  (%s: %s not present, no rebinding)\n", ST_TOKS[i][0], ST_TOKS[i][1]); }
            st_target(&T, ou ? U.a : NULL, NULL, 1u, 48u, 1u, 2u, 48u << 10, 64u << 10);
            toks_unload(T.s);
            toks_unload(T.a);
            if (ou) { toks_unload(U.s); toks_unload(U.a); }
            ran++;
        }
        printf("%s: %u tokenizers, %d cells x 2 tiers, %ld failures\n", ST_NAME, ran, (int)C_N, st_fail);
    }
    free(st_txt.p);
    free(st_ref.p);
    free(S_CASE.c);
    free(S_PROMPT.c);
    free(S_BIG.c);
    free(S_LONG.c);
    free(st_out);
    free(st_out2);
    free(st_R);
    return st_fail != 0;
}
