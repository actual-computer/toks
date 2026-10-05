/*
 * test_memo.inc: the body of test_memo.c and test_memo_hash.c (hand-written; included, so the Makefile's *.inc rule
 * rebuilds both). The segment memo (SPEC §6, kernels.md §7 "memo"): every encode on a scratch with
 * a memo (TOKS_SCRATCH_MEMO_MIB(m), and the default's 4 MiB) equals the same encode on a scratch without one
 * (TOKS_SCRATCH_MEMO_MIB(0)), over §6's evidence list: conversations
 * replayed turn by turn (hits counted: the memo must answer), eviction (more segments than slots) and ring wrap,
 * changed bytes at one address, identical bytes at another, caps too small for a segment's ids (no record), the
 * call's mode and flags changing on one scratch, the scratch rebound to another tokenizer, a conversation migrating
 * across scratches. Fixtures from tests/data (always) and the pinned llama3 / qwen38 / gemma4 when present.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"
#include "core.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int mt_failures;
static long mt_checks;

#define MCHECK(cond, ...) do { mt_checks++; if (!(cond)) { mt_failures++; if (mt_failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

#define MT_MAX (1u << 21)                  /* the longest prompt (one call laps the 1 MiB memo's ring) */
#define MEMO_MIN_T 256u                     /* api.c's MEMO_MIN: the shortest segment the memo records */

typedef struct mt_tok {
    const char *name;                       /* a path, or a pinned name ("@llama3") */
    int added;                              /* seps[0..1] are added tokens: a replayed conversation must hit */
    const char *seps[6];                    /* added-token contents (and plain text) that cut segments */
} mt_tok;

static const mt_tok MT_TOKS[] = {
    { "tests/data/compile/llama3style.json", 1, { "<|begin|>", "<|end|>", "<|a|>", "@", "hello", "<|x" } },
    { "tests/data/breadth/added_opts.json", 0, { "<mask>", "<|end|>", "<|both|>", "[W]", "  ", "\t" } },
    { "tests/data/breadth/drop_nfc.json", 1, { "<|endoftext|>", "<|x|>", "\n", "", "", "" } },
    { "tests/data/spm/llamalike.json", 0, { "\n", "", "", "", "", "" } },
    { "tests/data/spm/gemma4like.json", 0, { "\n", "", "", "", "", "" } },
    { "tests/data/spm/mistrallike.json", 0, { "\n", "", "", "", "", "" } },   /* Metaspace first: the doc state */
    { "tests/data/breadth/trunc_pad.json", 0, { "<s>", "</s>", "<mask>", "\n", "", "" } },   /* normalized tokens, pad */
    { "tests/data/breadth/trunc_left.json", 0, { "<pad>", "\n", "", "", "", "" } },          /* truncation */
    { "@llama3", 1, { "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "", "" } },
    { "@qwen38", 1, { "<|im_start|>", "<|im_end|>", "\n", "", "", "" } },
    { "@gemma4", 1, { "<|turn>", "<turn|>", "\n", "", "", "" } },
};

static uint64_t mt_rng = 0x9E3779B97F4A7C15ull;
static uint32_t mt_rand(void)
{
    mt_rng ^= mt_rng << 13;
    mt_rng ^= mt_rng >> 7;
    mt_rng ^= mt_rng << 17;
    return (uint32_t)(mt_rng >> 16);
}

/* a random segment of about want bytes: words, spaces, punctuation, digits, newlines, accents (precomposed and
 * combining), CJK, an emoji, now and then an ill-formed byte */
static uint64_t mt_segment(uint8_t *p, uint64_t want)
{
    static const char *const W[] = { "the", " memo", " token", "izer", ",", ".", " ", "  ", "\n", "\n\n", "'s", " 42",
                                     "7", " caf\xc3\xa9", " e\xcc\x81t\xc3\xa9", " \xe4\xb8\xad\xe6\x96\x87",
                                     "\xf0\x9f\x98\x80", " zebra", "Quick", " BROWN", "!?", "\t", " (x)", "\xff" };
    uint64_t n = 0;
    while (n < want) {                      /* bound: want / 1 */
        const char *w = W[mt_rand() % (sizeof W / sizeof W[0])];
        size_t l = strlen(w);
        memcpy(p + n, w, l);
        n += l;
    }
    return n;
}

typedef struct mt_conv {
    uint8_t *seg[32];
    uint64_t seg_len[32];
    uint32_t n;
} mt_conv;

/* prompt j of a conversation: for every turn i <= j, sep a, segment i, sep b (seps of the tokenizer) */
static uint64_t mt_prompt(const mt_tok *tk, const mt_conv *cv, uint32_t j, uint8_t *out)
{
    uint64_t n = 0;
    for (uint32_t i = 0; i <= j; i++) {     /* bound: 32 turns */
        const char *a = tk->seps[i % 2], *b = tk->seps[2 + i % 2] != NULL && tk->seps[2 + i % 2][0] ? tk->seps[2 + i % 2] : a;
        memcpy(out + n, a, strlen(a));
        n += strlen(a);
        memcpy(out + n, cv->seg[i], cv->seg_len[i]);
        n += cv->seg_len[i];
        memcpy(out + n, b, strlen(b));
        n += strlen(b);
    }
    return n;
}

static uint64_t mt_hits(void *scr)
{
    uint64_t pos, hits;
    toks_scr_memo_ctr(toks_scr_header(scr), &pos, &hits);
    return hits;
}

/* encode text on the memo scratch and on the reference scratch: equal results (count, ids within cap) */
static int64_t mt_same(const toks_ctx *ctx, const uint8_t *text, uint64_t len, uint32_t flags, uint64_t cap, void *mscr,
                       void *rscr, const char *what)
{
    static uint32_t a[MT_MAX + 64], b[MT_MAX + 64];
    int64_t na = toks_encode(ctx, text, len, flags, a, cap, mscr);
    int64_t nb = toks_encode(ctx, text, len, flags, b, MT_MAX + 64, rscr);
    uint64_t k = (uint64_t)(na < 0 ? 0 : na) < cap ? (uint64_t)(na < 0 ? 0 : na) : cap;
    MCHECK(na == nb && na >= 0 && memcmp(a, b, 4u * k) == 0, "%s: memo %" PRId64 " ids vs %" PRId64 " (flags %u, cap %" PRIu64 ")",
           what, na, nb, flags, cap);
    return na;
}

static void mt_run(const mt_tok *tk, const toks_ctx *ctx, const toks_ctx *other)
{
    static uint8_t text[MT_MAX], text2[MT_MAX];
    /* the memo scratches hold every budget below (the default's 4 MiB the largest); the reference has no memo */
    uint64_t sb = toks_scratch_bytes(ctx, MT_MAX, 0u), rb = toks_scratch_bytes(ctx, MT_MAX, TOKS_SCRATCH_MEMO_MIB(0));
    uint64_t ob = toks_scratch_bytes(other, MT_MAX, 0u), orb = toks_scratch_bytes(other, MT_MAX, TOKS_SCRATCH_MEMO_MIB(0));
    sb = ob > sb ? ob : sb;
    rb = orb > rb ? orb : rb;
    void *ms = malloc((size_t)sb), *ms2 = malloc((size_t)sb), *rs = malloc((size_t)rb);
    if (ms == NULL || ms2 == NULL || rs == NULL) { fprintf(stderr, "malloc\n"); exit(2); }
    MCHECK(toks_scratch_init(ctx, ms, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0 &&
           toks_scratch_init(ctx, rs, rb, TOKS_SCRATCH_MEMO_MIB(0)) == 0, "%s: init", tk->name);
    mt_conv cv;
    cv.n = 24;
    for (uint32_t i = 0; i < cv.n; i++) {   /* bound: 24 turns, most over 256 bytes */
        cv.seg[i] = (uint8_t *)malloc(4096);
        if (cv.seg[i] == NULL) { exit(2); }
        cv.seg_len[i] = mt_segment(cv.seg[i], i % 5 == 4 ? 40u + mt_rand() % 150u : 260u + mt_rand() % 2000u);
    }
    /* a conversation, prompt by prompt: the memo answers the earlier turns */
    for (uint32_t j = 0; j < cv.n; j++) {
        uint64_t n = mt_prompt(tk, &cv, j, text);
        (void)mt_same(ctx, text, n, 0u, MT_MAX + 64, ms, rs, tk->name);
    }
    uint64_t h1 = mt_hits(ms);
#ifndef MT_DEGEN                             /* one slot for everything: hits are not expected */
    MCHECK(h1 >= 100u || tk->added == 0, "%s: the memo answered only %" PRIu64 " segments of a 24-turn conversation",
           tk->name, h1);
#else
    (void)h1;
#endif
    /* the default scratch (flags 0): a 4 MiB memo, which answers the same conversation alike */
    {
        uint64_t mb = 0;
        MCHECK(toks_scratch_init(ctx, ms2, sb, 0u) == 0 && toks_scr_memo(toks_scr_header(ms2), &mb) != NULL &&
               mb == (uint64_t)TOKS_MEMO_MIB << 20, "%s: the default scratch's memo: %" PRIu64 " bytes", tk->name, mb);
        for (uint32_t j = 0; j < cv.n; j++) {
            uint64_t n = mt_prompt(tk, &cv, j, text);
            (void)mt_same(ctx, text, n, 0u, MT_MAX + 64, ms2, rs, "default");
        }
#ifndef MT_DEGEN
        MCHECK(mt_hits(ms2) >= h1, "%s: the default memo answered %" PRIu64 " segments, a 1 MiB one %" PRIu64, tk->name,
               mt_hits(ms2), h1);
#endif
    }
    /* identical bytes at another address: answered; changed bytes at one address: never the stale record */
    uint64_t n = mt_prompt(tk, &cv, cv.n - 1u, text);
    memcpy(text2, text, n);
    (void)mt_same(ctx, text2, n, 0u, MT_MAX + 64, ms, rs, "another address");
#ifndef MT_DEGEN
    MCHECK(mt_hits(ms) > h1 || tk->added == 0, "%s: identical bytes at another address not answered", tk->name);
#endif
    for (uint32_t r = 0; r < 64; r++) {     /* bound: 64 edits */
        uint64_t at = mt_rand() % n;
        uint8_t old = text2[at];
        text2[at] = (uint8_t)(old ^ (1u + mt_rand() % 255u));
        (void)mt_same(ctx, text2, n, 0u, MT_MAX + 64, ms, rs, "changed byte");
        text2[at] = old;
    }
    /* caps too small for a segment's ids: the prefix is exact, nothing half-recorded answers later */
    for (uint64_t cap = 1; cap < 4000; cap = cap * 3 + 7) { (void)mt_same(ctx, text, n, 0u, cap, ms, rs, "small cap"); }
    (void)mt_same(ctx, text, n, 0u, MT_MAX + 64, ms, rs, "after small caps");
    /* a call returning n > cap publishes nothing (SPEC §4.4): on a fresh scratch, its text answers nothing next */
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init cap", tk->name);
    int64_t full = mt_same(ctx, text, n, 0u, MT_MAX + 64, ms2, rs, "cap full");
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init cap 2", tk->name);
    (void)mt_same(ctx, text, n, 0u, full > 1 ? (uint64_t)full - 1u : 0u, ms2, rs, "cap n - 1");
    (void)mt_same(ctx, text, n, 0u, MT_MAX + 64, ms2, rs, "after cap n - 1");
#ifndef MT_DEGEN
    MCHECK(mt_hits(ms2) == 0u, "%s: a call returning n > cap published %" PRIu64 " records", tk->name, mt_hits(ms2));
    (void)mt_same(ctx, text, n, 0u, MT_MAX + 64, ms2, rs, "published");
    MCHECK(mt_hits(ms2) > 0u || tk->added == 0, "%s: a call returning n <= cap published nothing", tk->name);
#endif
    /* truncation (trunc_left.json): a call can return n <= cap although a segment's ids passed cap; that segment is
     * not recorded (only its first ids reached out), so it never answers where more of its ids are seen */
    {
        uint64_t m = 0;
        const char *a = tk->seps[0], *b = a, *x = "Quick brown zebras, 42 of them!";
        memcpy(text2 + m, a, strlen(a)), m += strlen(a);
        memcpy(text2 + m, x, strlen(x)), m += strlen(x);
        memcpy(text2 + m, b, strlen(b)), m += strlen(b);
        memcpy(text2 + m, cv.seg[5], cv.seg_len[5]), m += cv.seg_len[5];
        MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init trunc", tk->name);
        int64_t na = toks_encode(ctx, text2, m, 0u, NULL, 0u, rs);
        (void)mt_same(ctx, text2, m, 0u, na > 0 ? (uint64_t)na : 0u, ms2, rs, "trunc a");
        m = 0;
        memcpy(text2 + m, a, strlen(a)), m += strlen(a);
        memcpy(text2 + m, cv.seg[5], cv.seg_len[5]), m += cv.seg_len[5];
        (void)mt_same(ctx, text2, m, 0u, MT_MAX + 64, ms2, rs, "trunc b");
    }
    /* the mode and the other flags on one scratch */
    static const uint32_t F[] = { TOKS_ADDED_NONE, TOKS_ADDED_NONSPECIAL, 0u, TOKS_NO_POSTPROCESS, TOKS_CONTINUATION,
                                  TOKS_ADDED_NONE | TOKS_CONTINUATION, TOKS_ADDED_NONSPECIAL | TOKS_NO_POSTPROCESS };
    for (uint32_t r = 0; r < 3; r++) {
        for (uint32_t f = 0; f < sizeof F / sizeof F[0]; f++) { (void)mt_same(ctx, text, n, F[f], MT_MAX + 64, ms, rs, "flags"); }
    }
    /* the document state: a segment at the start of a call and the same bytes continuing a document */
    for (uint32_t r = 0; r < 4; r++) {
        (void)mt_same(ctx, cv.seg[1], cv.seg_len[1], (r & 1u) != 0u ? TOKS_CONTINUATION : 0u, MT_MAX + 64, ms, rs, "doc state");
    }
    /* a recorded segment's prefixes as segments of their own, longest first (one slot for all of them in the
     * degenerate build: the slot then holds a longer record whose first bytes are the probe's) */
    for (uint64_t cut = cv.seg_len[1] - 1u; cut >= 256u; cut -= 97u) {
        uint64_t m = 0;
        const char *a = tk->seps[0];
        memcpy(text2 + m, a, strlen(a));
        m += strlen(a);
        memcpy(text2 + m, cv.seg[1], cut);
        m += cut;
        memcpy(text2 + m, a, strlen(a));
        m += strlen(a);
        (void)mt_same(ctx, text2, m, 0u, MT_MAX + 64, ms, rs, "prefix");
    }
    /* eviction and ring wrap: 1 MiB, 512 sets of two slots; thousands of distinct segments, earlier ones replayed */
    static uint8_t *segs[3000];
    static uint64_t sl[3000];
    for (uint32_t i = 0; i < 3000; i++) {   /* bound: 3000 segments */
        segs[i] = (uint8_t *)malloc(1200);
        if (segs[i] == NULL) { exit(2); }
        sl[i] = mt_segment(segs[i], 256u + mt_rand() % 700u);
        uint64_t m = 0;
        for (uint32_t k = 0; k < 3; k++) {  /* this one and two earlier ones, between separators */
            uint32_t s = k == 0 ? i : mt_rand() % (i + 1u);
            const char *a = tk->seps[k % 2];
            memcpy(text + m, a, strlen(a));
            m += strlen(a);
            memcpy(text + m, segs[s], sl[s]);
            m += sl[s];
        }
        (void)mt_same(ctx, text, m, 0u, MT_MAX + 64, ms, rs, "eviction / wrap");
    }
    /* one call whose records lap the ring (1.6 MiB of distinct segments): none published, every id exact */
    {
        uint64_t m = 0;
        MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init big", tk->name);
        for (uint32_t i = 0; i < 3000 && m < (MT_MAX - 4096u); i++) {     /* bound: 3000 segments */
            const char *a = tk->seps[i % 2];
            memcpy(text + m, a, strlen(a));
            m += strlen(a);
            memcpy(text + m, segs[i], sl[i]);
            m += sl[i];
        }
        for (uint32_t k = 0; k < 3; k++) { (void)mt_same(ctx, text, m, 0u, MT_MAX + 64, ms2, rs, "one call laps"); }
    }
    /* the full lap (kernels.md §7), one segment per call: text A, 1.5 times the segments one lap holds (counted on a
     * first pass, so each tokenizer's text takes the same shape whatever its ids per byte), seen four times with 300 refused
     * first sights before the fourth, keeps the lap it filled and is answered on its third and fourth sights alike (a
     * lapping ring: nothing on the fourth); then text C, seen three times: its refused second sights start the next
     * lap, and its third sight is answered; on another scratch, after A's third sight, text B's first sights:
     * TOKS_MEMO_DRY refused records start one */
    {
        uint64_t hs[5], lap0, na = 0, ring = (1u << 20) - 64u - (1u << 15);
        MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init size", tk->name);
        const toks_memo_head *sz = (const toks_memo_head *)(void *)toks_scr_memo(toks_scr_header(ms2), &(uint64_t){ 0 });
        uint64_t n1 = 0, last = 0;
        while (na < 1200u && (n1 == 0 || na < n1 + n1 / 2u)) {   /* bound: 1200; n1: the segments the first lap holds */
            (void)mt_same(ctx, segs[na], sl[na], 0u, MT_MAX + 64, ms2, rs, "size");
            if (n1 == 0 && na > 0u && sz->pos == last) { n1 = na; }
            last = sz->pos;
            na++;
        }
        (void)ring;
        void *sc[2] = { ms2, ms };
        for (uint32_t w = 0; w < 2; w++) {
            MCHECK(toks_scratch_init(ctx, sc[w], sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init lap %u", tk->name, w);
            toks_memo_head *mh = (toks_memo_head *)(void *)toks_scr_memo(toks_scr_header(sc[w]), &(uint64_t){ 0 });
            for (uint32_t r = 0; r < 4u - w; r++) {
                if (r == 3) {                       /* 300 first sights in between: refused, their marks beside A's */
                    for (uint32_t i = 2400; i < 2700; i++) { (void)mt_same(ctx, segs[i], sl[i], 0u, MT_MAX + 64, sc[w], rs, "marks"); }
                    hs[2] = mt_hits(sc[w]);
                }
                for (uint32_t i = 0; i < na; i++) { (void)mt_same(ctx, segs[i], sl[i], 0u, MT_MAX + 64, sc[w], rs, "full lap"); }
                hs[r] = mt_hits(sc[w]);
            }
            lap0 = mh->lap;
            int one = mh->probes >= (4u - w) * na - 100u;   /* every call one segment (added_opts' added tokens cut them) */
            if (w == 0) {
#ifndef MT_DEGEN
                MCHECK(!one || (hs[2] - hs[1] >= na / 4u && hs[3] - hs[2] >= hs[2] - hs[1]),
                       "%s: a full lap: third sight %" PRIu64 " hits, fourth %" PRIu64, tk->name, hs[2] - hs[1], hs[3] - hs[2]);
#endif
                for (uint32_t r = 0; r < 3; r++) {
                    for (uint32_t i = 2700; i < 2760; i++) { (void)mt_same(ctx, segs[i], sl[i], 0u, MT_MAX + 64, sc[w], rs, "next lap"); }
                    hs[4] = mt_hits(sc[w]);
                    if (r == 1) { hs[3] = hs[4]; }
                }
#ifndef MT_DEGEN
                MCHECK(!one || (mh->lap != lap0 && hs[4] - hs[3] >= 30u), "%s: second sights: lap %" PRIu64 " (was %" PRIu64 "), %" PRIu64
                       " hits on the third", tk->name, mh->lap, lap0, hs[4] - hs[3]);
#endif
            } else {
                for (uint32_t i = 1200; i < 2400; i++) { (void)mt_same(ctx, segs[i], sl[i], 0u, MT_MAX + 64, sc[w], rs, "dry"); }
#ifndef MT_DEGEN
                MCHECK(!one || mh->lap != lap0, "%s: %u refused records: lap %" PRIu64 " (was %" PRIu64 ")", tk->name, TOKS_MEMO_DRY,
                       mh->lap, lap0);
#endif
            }
            (void)one, (void)lap0;
        }
        MCHECK(toks_scratch_init(ctx, ms, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init back", tk->name);
        for (uint32_t j = 0; j < cv.n; j++) {   /* the conversation again on ms, for the cells below */
            uint64_t m = mt_prompt(tk, &cv, j, text);
            (void)mt_same(ctx, text, m, 0u, MT_MAX + 64, ms, rs, "again");
        }
    }
    /* a conversation migrating to a fresh scratch half way */
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init 2", tk->name);
    for (uint32_t j = 0; j < cv.n; j++) {
        uint64_t m = mt_prompt(tk, &cv, j, text);
        (void)mt_same(ctx, text, m, 0u, MT_MAX + 64, j < cv.n / 2 ? ms : ms2, rs, "migrating");
    }
    /* the 2-way set (kernels.md §7), one segment per call: A, B, A, B, C, B, C, A. In the degenerate-hash build every
     * segment has one hash, so all three share a set: A and B are both answered, C takes the older record's slot (A's),
     * then B and C are answered and A is not; in the library build each has its own set and all five replays answer */
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init 2-way", tk->name);
    {
        const toks_memo_head *mh = (const toks_memo_head *)(void *)toks_scr_memo(toks_scr_header(ms2), &(uint64_t){ 0 });
        uint32_t ix[3], nx = 0;
        for (uint32_t i = 0; i < cv.n && nx < 3u; i++) {    /* bound: 24 turns; three of distinct lengths >= 256 */
            int dup = cv.seg_len[i] < MEMO_MIN_T;
            for (uint32_t j = 0; j < nx; j++) { dup |= cv.seg_len[ix[j]] == cv.seg_len[i]; }
            if (!dup) { ix[nx++] = i; }
        }
        static const uint32_t order[8] = { 0, 1, 0, 1, 2, 1, 2, 0 };
        uint64_t hw[8] = { 0 };
        for (uint32_t k = 0; k < 8u && nx == 3u; k++) {
            (void)mt_same(ctx, cv.seg[ix[order[k]]], cv.seg_len[ix[order[k]]], 0u, MT_MAX + 64, ms2, rs, "2-way");
            hw[k] = mt_hits(ms2);
        }
        int one = nx == 3u && mh->probes == 8u;       /* every call one segment */
#ifdef MT_DEGEN
        MCHECK(!one || (hw[3] == 2u && hw[6] - hw[4] == 2u && hw[7] == hw[6]), "%s: one set of two: A B answered %" PRIu64
               ", then B C %" PRIu64 ", then A %" PRIu64 " (not)", tk->name, hw[3], hw[6] - hw[4], hw[7] - hw[6]);
#else
        MCHECK(!one || hw[7] == 5u, "%s: three segments: %" PRIu64 " of 5 replays answered", tk->name, hw[7]);
#endif
    }
    /* a record lapped by the write position: a longer segment that begins with its bytes lands on it a lap later
     * (the lap simulated: the header's write position and lap start set one ring on); its slot must not answer */
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init lap", tk->name);
    {
        uint64_t mb, m = 0, pos;
        uint8_t *mm = toks_scr_memo(toks_scr_header(ms2), &mb);
        const char *a = tk->seps[0];
        for (uint32_t k = 0; k < 3; k++) {          /* S; S + tail one lap later; S again */
            m = 0;
            memcpy(text2 + m, a, strlen(a));
            m += strlen(a);
            memcpy(text2 + m, cv.seg[3], cv.seg_len[3]);
            m += cv.seg_len[3];
            if (k == 1) {
                memcpy(text2 + m, cv.seg[4], cv.seg_len[4]);
                m += cv.seg_len[4];
                pos = mb - 64u - 32u * (mb >> 10);
                ((toks_memo_head *)(void *)mm)->pos = pos, ((toks_memo_head *)(void *)mm)->lap = pos;
            }
            memcpy(text2 + m, a, strlen(a));
            m += strlen(a);
            (void)mt_same(ctx, text2, m, 0u, MT_MAX + 64, ms2, rs, "lapped");
        }
    }
    /* the first record of a fresh scratch, then the scratch rebound to another tokenizer: not answered there */
    MCHECK(toks_scratch_init(ctx, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "%s: init 3", tk->name);
    (void)mt_same(ctx, cv.seg[2], cv.seg_len[2], 0u, MT_MAX + 64, ms2, rs, "before rebind");
    MCHECK(toks_scratch_init(other, ms2, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0 &&
           toks_scratch_init(other, rs, rb, TOKS_SCRATCH_MEMO_MIB(0)) == 0, "%s: rebind 2", tk->name);
    (void)mt_same(other, cv.seg[2], cv.seg_len[2], 0u, MT_MAX + 64, ms2, rs, "after rebind");
    MCHECK(toks_scratch_init(ctx, rs, rb, TOKS_SCRATCH_MEMO_MIB(0)) == 0, "%s: reference back", tk->name);
    /* the scratch rebound to another tokenizer: nothing of this one answers there */
    MCHECK(toks_scratch_init(other, ms, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0 &&
           toks_scratch_init(other, rs, rb, TOKS_SCRATCH_MEMO_MIB(0)) == 0, "%s: rebind", tk->name);
    MCHECK(mt_hits(ms) == 0u, "%s: rebinding kept the memo's counter", tk->name);
    for (uint32_t j = 0; j < cv.n; j++) {
        uint64_t m = mt_prompt(tk, &cv, j, text);
        (void)mt_same(other, text, m, 0u, MT_MAX + 64, ms, rs, "rebound");
    }
    for (uint32_t i = 0; i < cv.n; i++) { free(cv.seg[i]); }
    for (uint32_t i = 0; i < 3000; i++) { free(segs[i]); }
    free(ms);
    free(ms2);
    free(rs);
}

static toks_ctx *mt_load(const char *name)
{
    char path[1024];
    if (name[0] == '@') {
        const char *root = getenv("TOKS_TOKENIZER_CACHE");
        if (root != NULL && root[0] != 0) { snprintf(path, sizeof path, "%s/%s", root, name + 1); }
        else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name + 1); }
    } else {
        snprintf(path, sizeof path, "%s", name);
    }
    toks_ctx *ctx = NULL;
    return toks_load(&ctx, path, NULL) == 0 ? ctx : NULL;
}

int main(void)
{
    const uint32_t nt = (uint32_t)(sizeof MT_TOKS / sizeof MT_TOKS[0]);
    toks_ctx *ctx[sizeof MT_TOKS / sizeof MT_TOKS[0]];
    uint32_t ran = 0;
    for (uint32_t i = 0; i < nt; i++) {
        ctx[i] = mt_load(MT_TOKS[i].name);
        MCHECK(ctx[i] != NULL || MT_TOKS[i].name[0] == '@', "%s: not loadable", MT_TOKS[i].name);
    }
    for (uint32_t i = 0; i < nt; i++) {
        if (ctx[i] == NULL) { printf("  SKIP %s (not present)\n", MT_TOKS[i].name); continue; }
        mt_run(&MT_TOKS[i], ctx[i], ctx[(i + 1u) % nt] != NULL ? ctx[(i + 1u) % nt] : ctx[0]);
        ran++;
    }
    for (uint32_t i = 0; i < nt; i++) { toks_unload(ctx[i]); }
    printf("%s: %u tokenizers, %ld checks, %d failures\n", MT_NAME, ran, mt_checks, mt_failures);
    return mt_failures != 0;
}
