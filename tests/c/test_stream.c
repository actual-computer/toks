/*
 * test_stream.c: stream decode (SPEC §3.4, §4.7; src/core/stream.c) against its rule, read again here
 * independently of the core, and against toks_decode (batch).
 *
 * The rule: B = the bytes of the ids pushed since init / flush that decode keeps; after every push the
 * concatenated output equals lossy(B[0, |B| - held(B))), held(B) = the longest suffix that starts with a
 * lead byte and is a proper prefix of a well-formed sequence; flush emits the rest of lossy(B). The
 * reference here decides well-formedness by code-point ranges (not unicode table 3-7 as the core does),
 * computes lossy by maximal subparts from that, and B from toks_token.
 *
 * 1. hand cases on a hand-built context (ids 0..255 = the byte, plus split, invalid, special,
 *    non-special, string-less and long tokens): each push's bytes written out by hand.
 * 2. the api: argument errors, uninitialized / foreign / corrupted states, unknown init flags,
 *    TOKS_E_ID / TOKS_E_LIMIT / TOKS_E_CAP / TOKS_E_UNSUPPORTED leave st unchanged, the bound.
 * 3. stream == batch over every partition, for the hand context and gpt2, llama3 and GLM 5.3
 *    (~/.cache/toks/tokenizers or $TOKS_TOKENIZER_CACHE; a missing file prints SKIP): exhaustive
 *    partitions of short id windows, random partitions of long sequences (empty pushes included), a
 *    flush in the middle; at every push: capacities below its need (TOKS_E_CAP, st unchanged, out[0,
 *    cap) the exact prefix), the push with its output flush against a guard page, an empty push, the
 *    bound; the per-push bytes against the reference; push_1 ++ ... ++ flush == toks_decode.
 * 4. the caller's hold (toks_stream_hold), spm chains with ByteFallback: every sequence of 3. again
 *    through a hold of exactly its longest held run (the ones over 44 bytes included), the hold
 *    untouched by a failed push; one byte less is TOKS_E_LIMIT with st and the hold unchanged, and
 *    growing the hold by the push's n and pushing the same ids again recovers; the 12 x U+13000 run
 *    (48 byte tokens) and 300 x U+13000 through st's own bytes, a 1 KiB hold and the recovery; the
 *    hold's api (moves back and forth, overlapping moves of a run of distinct bytes checked byte for byte, the
 *    move back into st's own bytes with a run in flight against a stream that never left them, the 44 / 45
 *    edge of that move, a corrupted record, a run too long to move); on a ByteLevel or WordPiece stream the
 *    hold is a no-op: 0 and st unchanged.
 * Speed: tests/stream/bench_stream.c.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
#include "spm.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rng = 0x5EED5EED12345678ull;

/* ================================================================ the reference */

/* the sequence length a lead announces, 0 for a byte that is never a lead */
static uint32_t ref_need(uint8_t c)
{
    if (c < 0x80u) { return 1u; }
    if (c >= 0xC0u && c < 0xE0u) { return 2u; }
    if (c >= 0xE0u && c < 0xF0u) { return 3u; }
    if (c >= 0xF0u && c < 0xF8u) { return 4u; }
    return 0u;
}

/* 1 when b[0, m) (m >= 1) is a prefix of a well-formed utf-8 sequence: the code points it can still become
 * meet the scalars of its length (overlongs, surrogates and anything above U+10FFFF excluded). */
static int ref_prefix_ok(const uint8_t *b, uint64_t m)
{
    uint32_t need = ref_need(b[0]);
    if (need == 0u || m > need) { return 0; }
    if (need == 1u) { return 1; }
    uint32_t cp = (uint32_t)b[0] & (need == 2u ? 0x1Fu : need == 3u ? 0x0Fu : 0x07u);
    for (uint64_t i = 1; i < m; i++) {
        if ((b[i] & 0xC0u) != 0x80u) { return 0; }
        cp = (cp << 6) | (b[i] & 0x3Fu);
    }
    uint32_t rest = need - (uint32_t)m;
    uint32_t lo = cp << (6u * rest), hi = lo | ((1u << (6u * rest)) - 1u);
    uint32_t vmin = need == 2u ? 0x80u : need == 3u ? 0x800u : 0x10000u;
    uint32_t vmax = need == 2u ? 0x7FFu : need == 3u ? 0xFFFFu : 0x10FFFFu;
    if (hi < vmin || lo > vmax) { return 0; }
    if (need == 3u && lo >= 0xD800u && hi <= 0xDFFFu) { return 0; }
    return 1;
}

/* lossy(b[0, n)) into out (may be NULL); ob (may be NULL) gets, at every unit start, the bytes output
 * before it, -1 elsewhere, and ob[n] = the total. A unit: a whole sequence, or the longest prefix of one
 * (a maximal subpart, one U+FFFD), or a byte that starts nothing (one U+FFFD). */
static uint64_t ref_units(const uint8_t *b, uint64_t n, uint8_t *out, int64_t *ob)
{
    uint64_t i = 0, o = 0;
    if (ob != NULL) { for (uint64_t k = 0; k <= n; k++) { ob[k] = -1; } }
    while (i < n) {
        if (ob != NULL) { ob[i] = (int64_t)o; }
        uint64_t m = 0;
        while (i + m < n && m < 4u && ref_prefix_ok(b + i, m + 1u)) { m++; }
        if (m >= 1u && m == ref_need(b[i])) {
            if (out != NULL) { memcpy(out + o, b + i, m); }
            o += m;
            i += m;
        } else {
            if (out != NULL) { memcpy(out + o, "\xef\xbf\xbd", 3); }
            o += 3u;
            i += m ? m : 1u;
        }
    }
    if (ob != NULL) { ob[n] = (int64_t)o; }
    return o;
}

/* held(b[0, n)): the suffix from the nearest lead byte among the last three, when it is a proper prefix
 * of a well-formed sequence; else 0 */
static uint64_t ref_held(const uint8_t *b, uint64_t n)
{
    for (uint64_t t = 1; t <= 3u && t <= n; t++) {
        uint8_t c = b[n - t];
        if (c < 0xC2u || c > 0xF4u) { continue; }
        return (t < ref_need(c) && ref_prefix_ok(b + n - t, t)) ? t : 0u;
    }
    return 0u;
}

static int special_bit(const toks_ctx *ctx, uint32_t id)
{
    return ctx->special_ids != NULL && ((ctx->special_ids[id >> 5] >> (id & 31u)) & 1u) != 0u;
}

/* the bytes of the ids decode keeps, from toks_token; b may be NULL (count only) */
static uint64_t kept_bytes(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *b)
{
    uint64_t m = 0;
    for (uint64_t i = 0; i < n; i++) {
        if ((flags & TOKS_SKIP_SPECIAL) != 0u && special_bit(ctx, ids[i])) { continue; }
        uint64_t len = 0;
        const uint8_t *p = toks_token(ctx, ids[i], &len);
        if (p != NULL) {
            if (b != NULL) { memcpy(b + m, p, len); }
            m += len;
        }
    }
    return m;
}

typedef struct ref {
    uint8_t  *R;     /* lossy of every kept byte: the batch decode */
    uint64_t  nR;
    uint64_t *L;     /* L[i]: the bytes the rule has emitted after the first i ids, i = 0..n */
    uint64_t  hold;  /* spm: the longest open run (bytes) the rule holds after some prefix */
} ref;

static void ref_free(ref *r) { free(r->R); free(r->L); }

static void ref_build(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, ref *r)
{
    r->hold = 0;
    uint64_t nb = kept_bytes(ctx, ids, n, flags, NULL);
    uint8_t *B = (uint8_t *)malloc(nb + 1u);
    uint64_t *off = (uint64_t *)malloc((n + 1u) * 8u);
    int64_t *ob = (int64_t *)malloc((nb + 1u) * 8u);
    r->R = (uint8_t *)malloc(3u * nb + 4u);
    r->L = (uint64_t *)malloc((n + 1u) * 8u);
    off[0] = 0;
    for (uint64_t i = 0; i < n; i++) { off[i + 1u] = off[i] + kept_bytes(ctx, ids + i, 1u, flags, B + off[i]); }
    r->nR = ref_units(B, nb, r->R, ob);
    uint8_t *tmp = (uint8_t *)malloc(3u * nb + 4u);
    for (uint64_t i = 0; i <= n; i++) {
        uint64_t c = off[i] - ref_held(B, off[i]);
        CHECK(ob[c] >= 0, "reference: the safe cut %" PRIu64 " is not a unit start", c);
        r->L[i] = ob[c] >= 0 ? (uint64_t)ob[c] : 0u;
        /* the definition itself, on a sample of prefixes (all of them for short sequences): the
         * lossy of the prefix up to its held tail is the batch decode's prefix */
        if (n <= 64u || i % 61u == 0u || i == n) {
            uint64_t k = ref_units(B, c, tmp, NULL);
            CHECK(k == r->L[i] && memcmp(tmp, r->R, (size_t)k) == 0, "reference: prefix %" PRIu64, i);
        }
    }
    free(tmp);
    free(ob);
    free(off);
    free(B);
}

/* sentencepiece-style chains (spm.h): the rule from toks_decode (spm_c.c's batch) on prefixes. After i
 * ids the stream has emitted decode(ids[0, cut)), cut = i, or, when the kept strings end in a run of byte
 * tokens (ByteFallback) whose bytes are still a well-formed utf-8 prefix, the position after the last kept
 * string before that run: such a run's output (its bytes, or a U+FFFD per byte) waits on its end. */
static int ref_hexd(uint8_t c)
{
    return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
}

static int ref_byte_token(const uint8_t *p, uint64_t k)     /* "<0x" h h ">", Rust's from_str_radix takes "+h" */
{
    if (p == NULL || k != 6u || memcmp(p, "<0x", 3) != 0 || p[5] != '>') { return -1; }
    if (p[3] == '+') { return ref_hexd(p[4]); }
    return (ref_hexd(p[3]) < 0 || ref_hexd(p[4]) < 0) ? -1 : ref_hexd(p[3]) * 16 + ref_hexd(p[4]);
}

/* the byte of id's "<0xHH>" string, else -1. toks_token runs before len is read: in one call's argument list
 * (ref_byte_token(toks_token(.., &len), len)) the order is unspecified, and the windows abi reads len first */
static int ref_byte_id(const toks_ctx *ctx, uint32_t id)
{
    uint64_t len = 0;
    const uint8_t *p = toks_token(ctx, id, &len);
    return ref_byte_token(p, len);
}

/* 1 when b[0, n) is whole well-formed sequences, then at most one proper prefix of one */
static int ref_valid_so_far(const uint8_t *b, uint64_t n)
{
    uint64_t i = 0;
    while (i < n) {
        uint64_t m = 0;
        while (i + m < n && m < 4u && ref_prefix_ok(b + i, m + 1u)) { m++; }
        if (m == 0u) { return 0; }
        if (m == ref_need(b[i])) { i += m; continue; }
        return i + m == n;                               /* a prefix cut by the end, or a broken one */
    }
    return 1;
}

static int spm_bf(const toks_ctx *ctx)
{
    for (uint32_t i = 0; i < ctx->spm->n_dec; i++) { if (ctx->spm->dec[i].kind == TOKS_SPM_D_BYTE_FALLBACK) { return 1; } }
    return 0;
}

static int spm_kept(const toks_ctx *ctx, uint32_t id, uint32_t flags)
{
    if ((flags & TOKS_SKIP_SPECIAL) != 0u && special_bit(ctx, id)) { return 0; }
    return !(ctx->spm->holes != NULL && ((ctx->spm->holes[id >> 5] >> (id & 31u)) & 1u));
}

static void ref_build_spm(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, ref *r)
{
    int64_t total = toks_decode(ctx, ids, n, flags, NULL, 0u);
    CHECK(total >= 0, "spm reference: decode %" PRId64, total);
    r->nR = total > 0 ? (uint64_t)total : 0u;
    r->R = (uint8_t *)malloc(r->nR + 1u);
    r->L = (uint64_t *)malloc((n + 1u) * 8u);
    toks_decode(ctx, ids, n, flags, r->R, r->nR);
    int runs = ctx->spm->has_decoder && spm_bf(ctx);
    uint8_t *tmp = (uint8_t *)malloc(r->nR + 1u), *run = (uint8_t *)malloc(n + 1u);
    r->hold = 0;
    for (uint64_t i = 0; i <= n; i++) {
        uint64_t cut = i, m = 0, j = i;
        if (runs) {
            while (j > 0) {                              /* back over the trailing run of kept byte tokens */
                uint32_t id = ids[j - 1u];
                if (!spm_kept(ctx, id, flags)) { j--; continue; }
                int b = ref_byte_id(ctx, id);
                if (b < 0) { break; }
                run[m++] = (uint8_t)b;
                j--;
            }
            if (m != 0u) {
                for (uint64_t x = 0; x < m / 2u; x++) { uint8_t t = run[x]; run[x] = run[m - 1u - x]; run[m - 1u - x] = t; }
                if (ref_valid_so_far(run, m)) { cut = j; if (m > r->hold) { r->hold = m; } }
            }
        }
        int64_t k = toks_decode(ctx, ids, cut, flags, tmp, r->nR);
        CHECK(k >= 0 && (uint64_t)k <= r->nR && memcmp(tmp, r->R, (size_t)k) == 0, "spm reference: prefix %" PRIu64, i);
        r->L[i] = k >= 0 ? (uint64_t)k : 0u;
    }
    free(run);
    free(tmp);
}

/* ================================================================ the walker */

static uint8_t *G_END;            /* end of a guard buffer: out = G_END - cap is flush against a guard page */
static uint64_t G_SIZE;

static uint8_t *gout(uint64_t cap) { return cap != 0u ? G_END - cap : NULL; }

static uint8_t *G_HOLD;           /* a caller's hold the walks attach (toks_stream_hold), NULL: st's own bytes */
static uint64_t G_HOLD_CAP;

/* st as toks_stream_init leaves it, with G_HOLD attached when it is set */
static void stream_fresh(const toks_ctx *ctx, toks_stream *st, uint32_t flags)
{
    toks_stream_init(ctx, st, flags);
    if (G_HOLD != NULL) { CHECK(toks_stream_hold(ctx, st, G_HOLD, G_HOLD_CAP) == 0, "a hold on a fresh stream"); }
}
static uint64_t hold_bound(void) { return G_HOLD != NULL ? 3u * G_HOLD_CAP : 0u; }   /* toks.h: + 3 cap */

/* one partition, cuts[0] = 0 < ... <= cuts[k] = n (equal cuts = empty pushes), through st (in its initial
 * state; back in it after the flush). every_cap: every capacity below each push's need, else three. */
static void walk(const toks_ctx *ctx, toks_stream *st, const uint32_t *ids, uint64_t n, uint32_t flags,
                 const ref *r, const uint64_t *cuts, uint64_t ncuts, int every_cap)
{
    toks_stream st2, fresh;
    stream_fresh(ctx, &fresh, flags);
    CHECK(memcmp(st, &fresh, sizeof fresh) == 0, "walk: st not in its initial state");
    uint8_t *hsnap = G_HOLD != NULL ? (uint8_t *)malloc(G_HOLD_CAP) : NULL;
    for (uint64_t k = 0; k + 1u < ncuts; k++) {
        uint64_t a = cuts[k], b = cuts[k + 1u], need = r->L[b] - r->L[a];
        const uint8_t *exp = r->R + r->L[a];
        if (need > 0u) {
            uint64_t tries = every_cap ? need : 3u;
            if (hsnap != NULL) { memcpy(hsnap, G_HOLD, (size_t)G_HOLD_CAP); }
            for (uint64_t t = 0; t < tries; t++) {
                uint64_t cap = every_cap ? t : t == 0u ? 0u : t == 1u ? need - 1u : guard_rng(&rng) % need;
                memcpy(&st2, st, sizeof st2);
                uint8_t *o = gout(cap);
                int64_t res = toks_stream_push(ctx, &st2, ids + a, b - a, o, cap);
                CHECK(res == TOKS_E_CAP && memcmp(&st2, st, sizeof st2) == 0, "E_CAP: push [%" PRIu64 ", %" PRIu64
                      ") cap %" PRIu64 " < %" PRIu64 ": %" PRId64, a, b, cap, need, res);
                CHECK(cap == 0u || memcmp(o, exp, (size_t)cap) == 0, "E_CAP: out[0, cap) not the exact prefix");
            }
            CHECK(hsnap == NULL || memcmp(hsnap, G_HOLD, (size_t)G_HOLD_CAP) == 0, "E_CAP wrote the hold");
        }
        uint64_t bound = toks_stream_bound(ctx, b - a) + hold_bound();
        CHECK(need <= bound, "bound: push of %" PRIu64 " ids needs %" PRIu64 " > %" PRIu64, b - a, need, bound);
        uint64_t cap = (guard_rng(&rng) & 3u) == 0u && bound < G_SIZE ? bound : need;
        uint8_t *o = gout(cap);
        const uint32_t *ip = (b > a || (guard_rng(&rng) & 1u)) ? ids + a : NULL;
        int64_t res = toks_stream_push(ctx, st, ip, b - a, o, cap);
        CHECK(res == (int64_t)need && (need == 0u || memcmp(o, exp, (size_t)need) == 0),
              "push [%" PRIu64 ", %" PRIu64 ") flags %u cap %" PRIu64 ": got %" PRId64 ", want %" PRIu64,
              a, b, flags, cap, res, need);
        memcpy(&st2, st, sizeof st2);
        CHECK(toks_stream_push(ctx, st, NULL, 0u, NULL, 0u) == 0 && memcmp(&st2, st, sizeof st2) == 0,
              "an empty push changed the stream");
    }
    free(hsnap);
    uint64_t need = r->nR - r->L[n];
    CHECK(need <= toks_stream_bound(ctx, 0u) + hold_bound(), "flush: %" PRIu64 " bytes", need);
    for (uint64_t cap = 0; cap < need; cap++) {
        memcpy(&st2, st, sizeof st2);
        CHECK(toks_stream_flush(ctx, &st2, gout(cap), cap) == TOKS_E_CAP && memcmp(&st2, st, sizeof st2) == 0 &&
              (cap == 0u || memcmp(gout(cap), r->R + r->L[n], (size_t)cap) == 0), "flush E_CAP at cap %" PRIu64, cap);
    }
    int64_t res = toks_stream_flush(ctx, st, gout(need), need);
    CHECK(res == (int64_t)need && (need == 0u || memcmp(gout(need), r->R + r->L[n], (size_t)need) == 0),
          "flush: %" PRId64 ", want %" PRIu64, res, need);
    CHECK(memcmp(st, &fresh, sizeof fresh) == 0, "flush: st not back in its initial state");
}

static uint64_t g_partitions, g_pushes, g_unholdable, g_held, g_limits;

/* the walks of check_seq for ids[0, n) and its reference, through G_HOLD (else st's own 44 bytes) */
static void seq_walks(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, const ref *r, uint64_t exh,
                      int every_cap, int rand_parts)
{
    uint64_t lim = G_HOLD != NULL ? G_HOLD_CAP : 44u;
    uint64_t *cuts = (uint64_t *)malloc((2u * n + 4u) * 8u);
    toks_stream st;
    stream_fresh(ctx, &st, flags);
    if (n <= exh) {
        uint64_t parts = n ? 1ull << (n - 1u) : 1u;
        for (uint64_t mask = 0; mask < parts; mask++) {
            uint64_t nc = 0;
            cuts[nc++] = 0;
            for (uint64_t i = 1; i < n; i++) { if ((mask >> (i - 1u)) & 1u) { cuts[nc++] = i; } }
            cuts[nc++] = n;
            walk(ctx, &st, ids, n, flags, r, cuts, nc, every_cap);
            g_partitions++;
            g_pushes += nc - 1u;
        }
    }
    for (int p = 0; p < rand_parts; p++) {
        uint64_t nc = 0, i = 0;
        cuts[nc++] = 0;
        while (i < n) {
            uint64_t x = guard_rng(&rng) % 8u;
            uint64_t len = x < 3u ? 1u : x < 5u ? guard_rng(&rng) % 4u : x < 7u ? guard_rng(&rng) % 17u
                                                                           : guard_rng(&rng) % 65u;
            i = (i + len < n) ? i + len : n;
            cuts[nc++] = i;
        }
        walk(ctx, &st, ids, n, flags, r, cuts, nc, every_cap);
        g_partitions++;
        g_pushes += nc - 1u;
    }

    /* a flush in the middle: [0, s) then [s, n) through the same stream, each against its own reference */
    if (n >= 2u && rand_parts > 0) {
        uint64_t s = 1u + guard_rng(&rng) % (n - 1u);
        for (int half = 0; half < 2; half++) {
            const uint32_t *hp = half ? ids + s : ids;
            uint64_t hn = half ? n - s : s, nc = 0, i = 0;
            ref h;
            if (ctx->spm != NULL) { ref_build_spm(ctx, hp, hn, flags, &h); } else { ref_build(ctx, hp, hn, flags, &h); }
            if (h.hold > lim) { ref_free(&h); break; }
            cuts[nc++] = 0;
            while (i < hn) { i += 1u + guard_rng(&rng) % 9u; if (i > hn) { i = hn; } cuts[nc++] = i; }
            walk(ctx, &st, hp, hn, flags, &h, cuts, nc, every_cap);
            ref_free(&h);
        }
    }
    free(cuts);
}

/* ids[0, n) one id per push, starting with a hold of `cap` bytes (0: st's own 44), each TOKS_E_LIMIT answered as
 * toks.h says (a hold of the current size plus the push's n: `twice` doubles it instead, the amortized way) and
 * the same id pushed again: every push's bytes against the reference, the flush, st and the hold unchanged by
 * the refusal. Returns the refusals. */
static uint64_t grow_walk(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, const ref *r,
                          uint64_t cap, int twice, const char *name)
{
    uint8_t *h = cap != 0u ? (uint8_t *)malloc(cap) : NULL, *snap = (uint8_t *)malloc(cap + 1u);
    uint64_t lim = cap != 0u ? cap : 44u, limits = 0, oc = 0;
    uint8_t *o = NULL;
    toks_stream st, st2;
    toks_stream_init(ctx, &st, flags);
    CHECK(toks_stream_hold(ctx, &st, h, cap) == 0, "%s: grow_walk: the first hold", name);
    for (uint64_t i = 0; i <= n; i++) {                   /* n pushes, then the flush */
        uint64_t need = i < n ? r->L[i + 1u] - r->L[i] : r->nR - r->L[n];
        uint64_t want = toks_stream_bound(ctx, i < n ? 1u : 0u) + 3u * lim;
        if (want > oc) { free(o); oc = want; o = (uint8_t *)malloc(oc); }
        if (i == n) {
            int64_t f = toks_stream_flush(ctx, &st, o, oc);
            CHECK(f == (int64_t)need && (need == 0u || memcmp(o, r->R + r->L[n], (size_t)need) == 0),
                  "%s: grow_walk: flush %" PRId64 ", want %" PRIu64, name, f, need);
            break;
        }
        memcpy(&st2, &st, sizeof st);
        if (h != NULL) { memcpy(snap, h, (size_t)lim); }
        int64_t res = toks_stream_push(ctx, &st, ids + i, 1u, o, oc);
        if (res == TOKS_E_LIMIT) {
            limits++;
            CHECK(memcmp(&st, &st2, sizeof st) == 0 && (h == NULL || memcmp(snap, h, (size_t)lim) == 0),
                  "%s: TOKS_E_LIMIT changed st or the hold (push %" PRIu64 ")", name, i);
            uint64_t ncap = twice ? 2u * lim : lim + 1u;  /* lim + n, n = 1 */
            uint8_t *nh = (uint8_t *)malloc(ncap);
            int64_t held = toks_stream_hold(ctx, &st, nh, ncap);
            CHECK(held == (int64_t)lim, "%s: the move holds %" PRId64 ", want %" PRIu64, name, held, lim);
            free(h);                                      /* the old hold: free once the call returned */
            h = nh, lim = ncap;
            free(snap);
            snap = (uint8_t *)malloc(lim + 1u);
            want = toks_stream_bound(ctx, 1u) + 3u * lim;
            if (want > oc) { free(o); oc = want; o = (uint8_t *)malloc(oc); }
            res = toks_stream_push(ctx, &st, ids + i, 1u, o, oc);
        }
        CHECK(res == (int64_t)need && (need == 0u || memcmp(o, r->R + r->L[i], (size_t)need) == 0),
              "%s: grow_walk: push %" PRIu64 ": %" PRId64 ", want %" PRIu64, name, i, res, need);
    }
    free(o);
    free(snap);
    free(h);
    g_limits += limits;
    return limits;
}

/* stream == batch for ids[0, n): the reference against toks_decode, then every partition (n <= exh) or
 * `rand_parts` random ones, and a flush in the middle; for an spm chain whose rule holds a run, the same
 * again through a caller's hold of exactly its longest run, and one id per push through a hold one byte
 * short of it (one TOKS_E_LIMIT, then the recovery). */
static void check_seq(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint64_t exh,
                      int every_cap, int rand_parts)
{
    ref r;
    if (ctx->spm != NULL) { ref_build_spm(ctx, ids, n, flags, &r); } else { ref_build(ctx, ids, n, flags, &r); }
    uint8_t *dec = (uint8_t *)malloc(r.nR + 1u);
    int64_t res = toks_decode(ctx, ids, n, flags, dec, r.nR);
    CHECK(res == (int64_t)r.nR && memcmp(dec, r.R, (size_t)r.nR) == 0,
          "toks_decode != the reference (%" PRId64 " vs %" PRIu64 " bytes)", res, r.nR);
    free(dec);
    if (r.hold > 44u) {                  /* st's own bytes refuse it (TOKS_E_LIMIT, test_ctx checks that) */
        g_unholdable++;
    } else {
        seq_walks(ctx, ids, n, flags, &r, exh, every_cap, rand_parts);
    }
    if (ctx->spm != NULL && r.hold != 0u) {
        G_HOLD_CAP = r.hold;
        G_HOLD = (uint8_t *)malloc(r.hold);
        seq_walks(ctx, ids, n, flags, &r, exh, every_cap, rand_parts);
        free(G_HOLD);
        G_HOLD = NULL;
        G_HOLD_CAP = 0u;
        g_held++;
        if (r.hold >= 2u && rand_parts > 0) {
            CHECK(grow_walk(ctx, ids, n, flags, &r, r.hold - 1u, 0, ctx->name) == 1u, "%s: a hold one byte short of"
                  " the longest run (%" PRIu64 ") refuses once", ctx->name, r.hold);
        }
    }
    ref_free(&r);
}

/* exhaustive partitions of `count` random windows of `w` ids of a long sequence */
static void check_windows(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint64_t w,
                          int count, int every_cap)
{
    for (int k = 0; k < count && n >= w; k++) {
        uint64_t at = guard_rng(&rng) % (n - w + 1u);
        check_seq(ctx, ids + at, w, flags, w, every_cap, 0);
    }
}

/* ================================================================ the hand context */

static const struct { const char *s; uint32_t len; } HX[] = {
#define S(x) { x, (uint32_t)sizeof(x) - 1u }
    S("<|s|>"),                    /* 256 special */
    S("<|n|>"),                    /* 257 added, not special */
    S("\xe2\x82\xac"),             /* 258 euro sign */
    S("\xe2\x82"),                 /* 259 its first two bytes */
    S(""),                         /* 260 no string */
    S("\xf0\x9f"),                 /* 261 half an emoji */
    S("\x98\x80"),                 /* 262 its second half */
    S("x\xe2"),                    /* 263 ascii, then a lead */
    S("\x82\xacy"),                /* 264 the rest of that euro sign, then ascii */
    S("\xed\xa0\x80"),             /* 265 an encoded surrogate */
    S("\xc0\xaf"),                 /* 266 an overlong slash */
    S("\xf4\x90\x80\x80"),         /* 267 above U+10FFFF */
    S("abc\xf0\x9f\x98"),          /* 268 ascii, then three quarters of an emoji */
    S("\xe0\xa0"),                 /* 269 a valid prefix with E0's narrow second byte */
    S("\xe0\x80"),                 /* 270 E0 with a second byte out of its range */
    S("The quick brown fox jumps over the lazy "),   /* 271: 40 ascii bytes */
    S("caf\xc3\xa9 \xe6\xbc\xa2\xe5\xad\x97 \xf0\x9f\x98"),   /* 272 valid chars, then a partial */
    S("abcdefgh\x80"),             /* 273 a non-ascii last byte past the first word */
    S("\x80" "bcdefghij"),         /* 274 a non-ascii first byte */
    S("\xe6\xbc\xa2"),             /* 275 one cjk char */
    S("\xe6\xbc\xa2\xe5\xad\x97\xe6\xbc\xa2\xe5\xad\x97\xe6\xbc\xa2" "a"),  /* 276: 16 valid bytes */
    S("\xe6\xbc\xa2\xe5\xad\x97\xe6\xbc\xa2\xe5\xad\x97\xe6\xbc\xa2" "ab"), /* 277: 17 valid bytes */
    S("\xf0\x9f\x98\x80"),         /* 278 one emoji */
    S("ab"), S("abc"), S("abcd"), S("abcde"), S("abcdef"), S("abcdefg"), S("abcdefgh"), S("abcdefghi"),
    S("abcdefghij"), S("abcdefghijk"), S("abcdefghijkl"), S("abcdefghijklm"), S("abcdefghijklmn"),
    S("abcdefghijklmno"), S("abcdefghijklmnop"), S("abcdefghijklmnopq"), S("abcdefghijklmnopqr"),
    S("abcdefghijklmnopqrs"), S("abcdefghijklmnopqrst"),   /* 279..297: ascii of 2..20 bytes */
    S("zz"),                       /* 298: the last bytes of the table */
#undef S
};
#define H_N (256u + (uint32_t)(sizeof HX / sizeof HX[0]))
enum { H_SPECIAL = 256, H_ADDED, H_EUR, H_E282, H_NIL, H_F09F, H_9880, H_XE2, H_82ACY, H_SURR, H_OVERLONG,
       H_BIG, H_ABC3Q, H_E0A0, H_E080, H_LONG, H_MIXED, H_ASC80, H_80ASC, H_CJK, H_CJK16, H_CJK17, H_EMOJI,
       H_ASCII2 };

static toks_ctx HC;
static uint32_t H_OFF[H_N + 1u];
static uint8_t  H_BYTES[1024];
static uint32_t H_SPEC[(H_N + 31u) / 32u];

static void build_hand(void)
{
    memset(&HC, 0, sizeof HC);
    uint32_t o = 0;
    for (uint32_t b = 0; b < 256u; b++) { H_OFF[b] = o; H_BYTES[o++] = (uint8_t)b; }
    for (uint32_t i = 256u; i < H_N; i++) {
        H_OFF[i] = o;
        memcpy(H_BYTES + o, HX[i - 256u].s, HX[i - 256u].len);
        o += HX[i - 256u].len;
    }
    H_OFF[H_N] = o;
    H_SPEC[H_SPECIAL >> 5] |= 1u << (H_SPECIAL & 31u);
    HC.t.magic = TOKS_TABLES_MAGIC;
    HC.t.version = TOKS_TABLES_VERSION;
    HC.t.algo = TOKS_ALGO_BPE_BYTELEVEL;
    HC.t.n_ids = H_N;
    HC.t.tok_off = H_OFF;
    HC.t.tok_bytes = H_BYTES;
    HC.tier = TOKS_TIER_SCALAR;
    HC.dec_byte_level = 1u;
    HC.special_ids = H_SPEC;
    HC.identity = 0x5354524541D00001ull;
    if (toks_dec_build(&HC) != 0) { fprintf(stderr, "toks_dec_build\n"); exit(2); }
    memcpy(HC.name, "hand", 5);
}

static int hexb(const char *h, uint8_t *b)
{
    int n = 0;
    while (*h) {
        if (*h == ' ') { h++; continue; }
        unsigned v;
        if (sscanf(h, "%2x", &v) != 1) { break; }
        b[n++] = (uint8_t)v;
        h += 2;
    }
    return n;
}

#define FLUSH 0xFFFFFFFFu        /* a push of this one id stands for a flush in the middle */

typedef struct hcase {
    uint32_t flags;
    int np;                      /* pushes (FLUSH counts as one) */
    struct { int n; uint32_t ids[4]; } push[6];
    const char *want[7];         /* per push, then the final flush */
} hcase;

#define SK TOKS_SKIP_SPECIAL
#define P1(a) { 1, { a } }
#define P2(a, b) { 2, { a, b } }
#define P3(a, b, c) { 3, { a, b, c } }
#define P0 { 0, { 0 } }
#define FFD "ef bf bd "
static const hcase HCASES[] = {
    { 0, 3, { P1(0xE2), P1(0x82), P1(0xAC) }, { "", "", "e2 82 ac", "" } },
    { 0, 2, { P1(H_E282), P1(0xAC) }, { "", "e2 82 ac", "" } },
    { 0, 2, { P1(0xE2), P2(0x82, 0xAC) }, { "", "e2 82 ac", "" } },
    { 0, 2, { P1(H_F09F), P1(H_9880) }, { "", "f0 9f 98 80", "" } },
    { 0, 4, { P1(0xF0), P1(0x9F), P1(0x98), P1(0x80) }, { "", "", "", "f0 9f 98 80", "" } },
    { 0, 1, { P1(H_E282) }, { "", FFD } },
    { 0, 2, { P1(H_E282), P1(0x41) }, { "", FFD "41", "" } },
    { 0, 2, { P1(H_E282), P1(H_EUR) }, { "", FFD "e2 82 ac", "" } },
    { 0, 2, { P1(H_E282), P1(H_E282) }, { "", FFD, FFD } },
    { 0, 1, { P1(0x80) }, { FFD, "" } },
    { 0, 2, { P1(0xFF), P1(0xFE) }, { FFD, FFD, "" } },
    { 0, 2, { P1(H_XE2), P1(H_82ACY) }, { "78", "e2 82 ac 79", "" } },
    { 0, 1, { P1(H_SURR) }, { FFD FFD FFD, "" } },
    { 0, 2, { P1(0xED), P2(0xA0, 0x80) }, { "", FFD FFD FFD, "" } },
    { 0, 1, { P1(H_OVERLONG) }, { FFD FFD, "" } },
    { 0, 1, { P1(H_BIG) }, { FFD FFD FFD FFD, "" } },
    { 0, 2, { P1(H_E0A0), P1(0x80) }, { "", "e0 a0 80", "" } },
    { 0, 1, { P1(H_E0A0) }, { "", FFD } },
    { 0, 1, { P1(H_E080) }, { FFD FFD, "" } },
    { 0, 3, { P1(H_E282), P1(H_SPECIAL), P1(0xAC) }, { "", FFD "3c 7c 73 7c 3e", FFD, "" } },
    { SK, 3, { P1(H_E282), P1(H_SPECIAL), P1(0xAC) }, { "", "", "e2 82 ac", "" } },
    { SK, 1, { P1(H_ADDED) }, { "3c 7c 6e 7c 3e", "" } },
    { 0, 2, { P1(H_NIL), P2(0x62, H_NIL) }, { "", "62", "" } },
    { 0, 6, { P0, P1(H_E282), P0, P0, P1(0xAC), P0 }, { "", "", "", "", "e2 82 ac", "", "" } },
    { 0, 2, { P1(H_ABC3Q), P1(0x80) }, { "61 62 63", "f0 9f 98 80", "" } },
    { 0, 1, { P1(H_ABC3Q) }, { "61 62 63", FFD } },
    { 0, 3, { P1(H_E282), P1(FLUSH), P1(0xAC) }, { "", FFD, FFD, "" } },
    { 0, 4, { P1(0xF0), P1(0x9F), P1(0x98), P1(0x41) }, { "", "", "", FFD "41", "" } },
    { 0, 2, { P1(0xF0), P1(0x80) }, { "", FFD FFD, "" } },
    { 0, 3, { P1(0xC2), P1(0xC2), P1(0x80) }, { "", FFD, "c2 80", "" } },
    { 0, 1, { P1(H_MIXED) }, { "63 61 66 c3 a9 20 e6 bc a2 e5 ad 97 20", FFD } },
    { 0, 2, { P1(H_MIXED), P1(0x80) }, { "63 61 66 c3 a9 20 e6 bc a2 e5 ad 97 20", "f0 9f 98 80", "" } },
    { 0, 1, { P1(H_ASC80) }, { "61 62 63 64 65 66 67 68 " FFD, "" } },
    { 0, 1, { P1(H_80ASC) }, { FFD "62 63 64 65 66 67 68 69 6a", "" } },
    { SK, 1, { P1(H_SPECIAL) }, { "", "" } },
    { SK, 3, { P1(H_E282), P1(H_SPECIAL), P2(H_SPECIAL, 0xAC) }, { "", "", "e2 82 ac", "" } },
    { 0, 2, { P3(H_E282, 0xAC, H_E282), P1(0xAC) }, { "e2 82 ac", "e2 82 ac", "" } },
    { 0, 4, { P1(0xF4), P1(0x8F), P1(0xBF), P1(0xBF) }, { "", "", "", "f4 8f bf bf", "" } },
    { 0, 2, { P1(0xF4), P1(0x90) }, { "", FFD FFD, "" } },
    { 0, 2, { P1(0xC2), P1(H_SPECIAL) }, { "", FFD "3c 7c 73 7c 3e", "" } },
    { 0, 3, { P1(H_CJK), P1(H_CJK16), P1(H_CJK17) },
      { "e6 bc a2", "e6 bc a2 e5 ad 97 e6 bc a2 e5 ad 97 e6 bc a2 61",
        "e6 bc a2 e5 ad 97 e6 bc a2 e5 ad 97 e6 bc a2 61 62", "" } },
    { 0, 2, { P1(0xE6), P2(0xBC, H_EMOJI) }, { "", FFD "f0 9f 98 80", "" } },
};

static void test_hand(void)
{
    uint8_t want[256], got[256];
    for (size_t c = 0; c < sizeof HCASES / sizeof HCASES[0]; c++) {
        const hcase *h = &HCASES[c];
        toks_stream st, st2;
        toks_stream_init(&HC, &st, h->flags);
        for (int k = 0; k <= h->np; k++) {
            int nw = hexb(h->want[k], want);
            int flush = k == h->np || (h->push[k].n == 1 && h->push[k].ids[0] == FLUSH);
            for (int cap = 0; cap < nw; cap++) {           /* every capacity below the need */
                memcpy(&st2, &st, sizeof st);
                uint8_t *o = gout((uint64_t)cap);
                int64_t r = flush ? toks_stream_flush(&HC, &st2, o, (uint64_t)cap)
                                  : toks_stream_push(&HC, &st2, h->push[k].ids, (uint64_t)h->push[k].n, o, (uint64_t)cap);
                CHECK(r == TOKS_E_CAP && memcmp(&st, &st2, sizeof st) == 0 && (cap == 0 || memcmp(o, want, (size_t)cap) == 0),
                      "hand %zu push %d cap %d: %" PRId64, c, k, cap, r);
            }
            int64_t r = flush ? toks_stream_flush(&HC, &st, got, sizeof got)
                              : toks_stream_push(&HC, &st, h->push[k].ids, (uint64_t)h->push[k].n, got, sizeof got);
            CHECK(r == nw && memcmp(got, want, (size_t)nw) == 0, "hand %zu push %d: got %" PRId64 " bytes, want %d",
                  c, k, r, nw);
        }
        /* and every partition of the same ids (no flush in the middle) against the reference */
        uint32_t ids[24];
        uint64_t n = 0;
        int mid = 0;
        for (int k = 0; k < h->np; k++) {
            for (int i = 0; i < h->push[k].n; i++) { if (h->push[k].ids[i] == FLUSH) { mid = 1; } ids[n++] = h->push[k].ids[i]; }
        }
        if (!mid) { check_seq(&HC, ids, n, h->flags, 16u, 1, 4); }
    }

    /* the bound is reached: a held lead broken by the longest single-id decode (271, 40 bytes) */
    toks_stream st;
    toks_stream_init(&HC, &st, 0u);
    uint32_t e2 = 0xE2, lng = H_LONG;
    CHECK(toks_stream_push(&HC, &st, &e2, 1u, got, sizeof got) == 0, "bound: hold");
    CHECK(toks_stream_push(&HC, &st, &lng, 1u, got, sizeof got) == (int64_t)toks_stream_bound(&HC, 1u) &&
          toks_stream_bound(&HC, 1u) == 43u, "bound(1) = %" PRIu64 " is reached", toks_stream_bound(&HC, 1u));

    /* toks_decode at every capacity, out flush against a guard page: out[0, min(cap, n)) exact, nothing
     * written past cap (the fast path's 16-byte moves only where whole moves fit) */
    for (int s = 0; s < 300; s++) {
        uint32_t q[48];
        uint64_t n = guard_rng(&rng) % 48u;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t x = guard_rng(&rng) % 4u;
            q[i] = x < 2u ? (uint32_t)(256u + guard_rng(&rng) % (H_N - 256u)) : (uint32_t)(guard_rng(&rng) % 256u);
        }
        uint32_t fl = (uint32_t)(guard_rng(&rng) & 1u);
        ref r;
        ref_build(&HC, q, n, fl, &r);
        for (uint64_t cap = 0; cap <= r.nR + 1u; cap++) {
            int64_t res = toks_decode(&HC, q, n, fl, gout(cap), cap);
            uint64_t k = cap < r.nR ? cap : r.nR;
            CHECK(res == (int64_t)r.nR && (k == 0u || memcmp(gout(cap), r.R, (size_t)k) == 0),
                  "decode cap %" PRIu64 " of %" PRIu64, cap, r.nR);
        }
        ref_free(&r);
    }

    /* random sequences over the hand vocabulary: every partition of short windows, random partitions */
    uint32_t seq[400];
    for (int s = 0; s < 60; s++) {
        uint64_t n = 20u + guard_rng(&rng) % 380u;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t x = guard_rng(&rng) % 10u;
            seq[i] = x < 4u ? (uint32_t)(256u + guard_rng(&rng) % (H_N - 256u))
                   : x < 8u ? (uint32_t)(0x80u + guard_rng(&rng) % 0x80u) : (uint32_t)(guard_rng(&rng) % 0x80u);
        }
        for (uint32_t fl = 0; fl < 2u; fl++) {
            check_seq(&HC, seq, n, fl, 0u, 0, 6);
            check_windows(&HC, seq, n, fl, 9u, 4, (s & 7) == 0);
        }
    }
}

/* ================================================================ the api */

static void test_api(void)
{
    toks_stream st, st2, z;
    uint8_t out[64], hold[64];
    uint32_t ids[4] = { H_E282, 0xAC, H_N, 'a' };
    memset(&z, 0, sizeof z);

    toks_stream_init(&HC, &st, 0u);
    CHECK(toks_stream_push(NULL, &st, ids, 1u, out, 8u) == TOKS_E_ARG, "push ctx NULL");
    CHECK(toks_stream_push(&HC, NULL, ids, 1u, out, 8u) == TOKS_E_ARG, "push st NULL");
    CHECK(toks_stream_push(&HC, &st, NULL, 1u, out, 8u) == TOKS_E_ARG, "push ids NULL");
    CHECK(toks_stream_push(&HC, &st, ids, 1u, NULL, 8u) == TOKS_E_ARG, "push out NULL");
    CHECK(toks_stream_flush(NULL, &st, out, 8u) == TOKS_E_ARG, "flush ctx NULL");
    CHECK(toks_stream_flush(&HC, NULL, out, 8u) == TOKS_E_ARG, "flush st NULL");
    CHECK(toks_stream_flush(&HC, &st, NULL, 8u) == TOKS_E_ARG, "flush out NULL");
    CHECK(toks_stream_push(&HC, &st, NULL, 0u, NULL, 0u) == 0, "push NULL with 0");
    CHECK(toks_stream_flush(&HC, &st, NULL, 0u) == 0, "flush NULL with 0");
    toks_stream_init(&HC, NULL, 0u);                      /* no crash */

    /* states push and flush refuse, unchanged */
    memset(&st, 0, sizeof st);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == TOKS_E_ARG && memcmp(&st, &z, sizeof z) == 0, "zeroed state");
    memset(&st, 0xA5, sizeof st);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == TOKS_E_ARG, "garbage state");
    toks_stream_init(NULL, &st, 0u);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == TOKS_E_ARG, "init with ctx NULL");
    toks_stream_init(&HC, &st, 2u);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == TOKS_E_ARG &&
          toks_stream_flush(&HC, &st, out, 8u) == TOKS_E_ARG, "init with unknown flags");
    toks_ctx other = HC;
    other.identity ^= 0x100u;
    toks_stream_init(&other, &st, 0u);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == TOKS_E_ARG, "a stream of another context");
    toks_stream_init(&HC, &st, 0u);
    CHECK(toks_stream_push(&HC, &st, ids, 1u, out, 8u) == 0, "hold e2 82");
    struct { uint32_t at; uint8_t val, at2, val2; } bad[] = {    /* the layout: stream.c's sst */
        { 16u, 4u, 0u, 0u },         /* np = 4 */
        { 20u, 0x41u, 0u, 0u },      /* buf[0] not a lead */
        { 20u, 0xE0u, 21u, 0x80u },  /* np = 2: E0 80 is no prefix */
        { 16u, 3u, 0u, 0u },         /* np = 3 >= the need of E2 */
        { 8u, 4u, 0u, 0u },          /* unknown flag bits */
        { 18u, 1u, 0u, 0u },         /* rsv */
        { 17u, 1u, 0u, 0u },         /* an spm mode in a ByteLevel stream */
        { 12u, 1u, 0u, 0u },         /* a Strip count in a ByteLevel stream */
        { 3u, 0x10u, 0u, 0u },       /* the tag */
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        memcpy(&st2, &st, sizeof st);
        ((uint8_t *)&st2)[bad[i].at] ^= (bad[i].at == 3u) ? bad[i].val : 0u;
        if (bad[i].at != 3u) { ((uint8_t *)&st2)[bad[i].at] = bad[i].val; }
        if (bad[i].at2 != 0u) { ((uint8_t *)&st2)[bad[i].at2] = bad[i].val2; }
        toks_stream st3 = st2;
        CHECK(toks_stream_push(&HC, &st2, ids + 1, 1u, out, 8u) == TOKS_E_ARG && memcmp(&st2, &st3, sizeof st2) == 0,
              "corrupted state %zu", i);
        CHECK(toks_stream_flush(&HC, &st2, out, 8u) == TOKS_E_ARG, "corrupted state %zu (flush)", i);
    }

    /* errors leave st unchanged */
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_push(&HC, &st, ids + 1, 3u, out, sizeof out) == TOKS_E_ID && memcmp(&st, &st2, sizeof st) == 0, "E_ID");
    CHECK(toks_stream_push(&HC, &st, ids, (uint64_t)TOKS_MAX_TEXT + 1u, out, sizeof out) == TOKS_E_LIMIT &&
          memcmp(&st, &st2, sizeof st) == 0, "E_LIMIT");
    toks_ctx nobl = HC;
    nobl.dec_byte_level = 0u;
    CHECK(toks_stream_push(&nobl, &st, ids + 1, 1u, out, 8u) == TOKS_E_UNSUPPORTED &&
          toks_stream_flush(&nobl, &st, out, 8u) == TOKS_E_UNSUPPORTED && memcmp(&st, &st2, sizeof st) == 0,
          "a decoder other than ByteLevel");
    /* toks_stream_hold on a ByteLevel stream, e2 82 held: a no-op, 0 and st unchanged (the bytes stay in st), with
     * a hold and with NULL / 0; a decoder the stream does not know is TOKS_E_UNSUPPORTED as push's; then the
     * third byte completes the char below */
    CHECK(toks_stream_hold(&HC, &st, hold, sizeof hold) == 0 && memcmp(&st, &st2, sizeof st) == 0, "hold: a ByteLevel no-op");
    CHECK(toks_stream_hold(&HC, &st, NULL, 0u) == 0 && memcmp(&st, &st2, sizeof st) == 0, "hold NULL / 0: a ByteLevel no-op");
    CHECK(toks_stream_hold(&nobl, &st, hold, sizeof hold) == TOKS_E_UNSUPPORTED && memcmp(&st, &st2, sizeof st) == 0,
          "hold: a decoder other than ByteLevel");
    CHECK(toks_stream_flush(&HC, &st, out, 2u) == TOKS_E_CAP && memcmp(&st, &st2, sizeof st) == 0, "flush E_CAP");
    CHECK(toks_stream_push(&HC, &st, ids + 1, 1u, out, 2u) == TOKS_E_CAP && memcmp(&st, &st2, sizeof st) == 0 &&
          out[0] == 0xE2 && out[1] == 0x82, "push E_CAP: out[0, cap) the exact prefix");
    CHECK(toks_stream_push(&HC, &st, ids + 1, 1u, out, 3u) == 3 && memcmp(out, "\xe2\x82\xac", 3) == 0, "then 3 fit");
    CHECK(toks_stream_flush(&HC, &st, NULL, 0u) == 0, "nothing held");

    /* the bound's weight counts a held tail's U+FFFD: a table whose longest single-id decode is
     * "abc" + U+FFFD (a token ending in three quarters of an emoji) */
    {
        static uint32_t moff[258];
        static uint8_t mbytes[256 + 6 + 64];
        for (uint32_t b = 0; b < 256u; b++) { moff[b] = b; mbytes[b] = (uint8_t)b; }
        memcpy(mbytes + 256, "abc\xf0\x9f\x98", 6);
        moff[256] = 256u;
        moff[257] = 262u;
        toks_ctx mini = HC;
        mini.t.n_ids = 257u;
        mini.t.tok_off = moff;
        mini.t.tok_bytes = mbytes;
        CHECK(toks_dec_build(&mini) == 0 && mini.dec_max == 6u && toks_stream_bound(&mini, 1u) == 9u,
              "bound weight with a held tail: dec_max %u", mini.dec_max);
        toks_dec_free(&mini);
    }

    /* the bound */
    CHECK(toks_stream_bound(NULL, 5u) == 0u, "bound ctx NULL");
    CHECK(toks_stream_bound(&HC, 0u) == 3u, "bound(0) covers flush");
    CHECK(toks_stream_bound(&HC, 1000u) == 1000u * 40u + 3u, "bound(1000)");
    CHECK(toks_stream_bound(&HC, UINT64_MAX) == UINT64_MAX &&
          toks_stream_bound(&HC, (UINT64_MAX - 3u) / 40u + 1u) == UINT64_MAX, "bound saturates");
    CHECK(toks_stream_bound(&HC, (UINT64_MAX - 3u) / 40u) == (UINT64_MAX - 3u) / 40u * 40u + 3u, "bound at the edge");
}

/* the hold on a WordPiece stream (nothing is ever held, st keeps only its flags): a no-op, 0 and st unchanged, on a
 * fresh stream and with a string kept; the rest of the ids then stream as toks_decode gives them. The tokenizer is
 * a four-string WordPiece inline (test_wp.c's shape), so no file is needed. */
static void test_hold_wp(void)
{
    static const char json[] =
        "{\"version\":\"1.0\",\"added_tokens\":[],\"normalizer\":{\"type\":\"BertNormalizer\",\"clean_text\":true,"
        "\"handle_chinese_chars\":true,\"strip_accents\":null,\"lowercase\":true},\"pre_tokenizer\":{\"type\":"
        "\"BertPreTokenizer\"},\"post_processor\":null,\"decoder\":{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true},"
        "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
        "\"max_input_chars_per_word\":100,\"vocab\":{\"[UNK]\":0,\"a\":1,\"##b\":2,\"c\":3}}}";
    toks_ctx *ctx = NULL;
    toks_diag dg;
    memset(&dg, 0, sizeof dg);
    toks_load_opts lo = { sizeof lo, 0, 0, 0, &dg };
    int64_t r = toks_load_mem_copy(&ctx, json, sizeof json - 1u, &lo);
    CHECK(r == 0, "wordpiece: load %" PRId64 " (%s)", r, dg.what);
    if (r != 0) { return; }
    uint32_t ids[3] = { 1u, 2u, 3u };                     /* a ##b c */
    toks_stream st, st2;
    uint8_t h[64], o[64], d[64];
    toks_stream_init(ctx, &st, 0u);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_hold(ctx, &st, h, sizeof h) == 0 && memcmp(&st, &st2, sizeof st) == 0, "wordpiece: hold on a fresh stream");
    int64_t p = toks_stream_push(ctx, &st, ids, 1u, o, sizeof o);
    CHECK(p == 1 && o[0] == 'a', "wordpiece: the first string: %" PRId64, p);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_hold(ctx, &st, h, sizeof h) == 0 && memcmp(&st, &st2, sizeof st) == 0, "wordpiece: hold is a no-op");
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == 0 && memcmp(&st, &st2, sizeof st) == 0, "wordpiece: NULL / 0 is a no-op");
    int64_t q = toks_stream_push(ctx, &st, ids + 1, 2u, o + 1, sizeof o - 1u);
    int64_t f = q >= 0 ? toks_stream_flush(ctx, &st, o + 1 + q, sizeof o - 1u - (uint64_t)q) : -1;
    int64_t dn = toks_decode(ctx, ids, 3u, 0u, d, sizeof d);
    CHECK(q > 0 && f == 0 && 1 + q == dn && memcmp(o, d, (size_t)dn) == 0,
          "wordpiece: the rest as decode gives it: %" PRId64 " + %" PRId64 " bytes, decode %" PRId64, q, f, dn);
    toks_unload(ctx);
}

/* ================================================================ the caller's hold: its api */

/* on an spm context with ByteFallback and a decoder; ba = the byte token of 'A' */
static void test_hold_api(const toks_ctx *ctx, const char *name, uint32_t ba)
{
    toks_stream st, st2, fresh;
    uint8_t h1[64], h2[128], snap[128], o[1024], d[1024];
    uint32_t run[80];
    for (int i = 0; i < 80; i++) { run[i] = ba; }
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_hold(NULL, &st, h1, 8u) == TOKS_E_ARG && toks_stream_hold(ctx, NULL, h1, 8u) == TOKS_E_ARG &&
          toks_stream_hold(ctx, &st, NULL, 8u) == TOKS_E_ARG, "%s: hold: argument errors", name);
    memset(&st2, 0, sizeof st2);
    CHECK(toks_stream_hold(ctx, &st2, h1, sizeof h1) == TOKS_E_ARG, "%s: hold: an uninitialized stream", name);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == 0 && memcmp(&st, &st2, sizeof st) == 0, "%s: NULL / 0 on a fresh stream", name);
    CHECK(toks_stream_push(ctx, &st, run, 30u, o, sizeof o) == 0, "%s: 30 run bytes in st's own", name);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_hold(ctx, &st, h1, 29u) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0,
          "%s: a hold below the held bytes", name);
    memset(h1, 0, sizeof h1);
    CHECK(toks_stream_hold(ctx, &st, h1, sizeof h1) == 30 && h1[0] == 'A' && h1[29] == 'A' && h1[30] == 0,
          "%s: the 30 bytes moved into the hold", name);
    CHECK(toks_stream_push(ctx, &st, run, 30u, o, sizeof o) == 0, "%s: 60 run bytes in a 64-byte hold", name);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0,
          "%s: 60 bytes do not go back into st's 44", name);
    memcpy(snap, h1, sizeof h1);
    CHECK(toks_stream_push(ctx, &st, run, 5u, o, sizeof o) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0 &&
          memcmp(snap, h1, sizeof h1) == 0, "%s: 65 > 64: TOKS_E_LIMIT, st and the hold unchanged", name);
    CHECK(toks_stream_hold(ctx, &st, h2, 64u + 5u) == 60, "%s: grown by the push's n", name);
    memset(h1, 0xEE, sizeof h1);                          /* the old hold is the caller's again */
    CHECK(toks_stream_push(ctx, &st, run, 5u, o, sizeof o) == 0, "%s: the same 5 ids again", name);
    CHECK(toks_stream_hold(ctx, &st, h2 + 1, sizeof h2 - 1u) == 65 && toks_stream_hold(ctx, &st, h2, sizeof h2) == 65 &&
          toks_stream_hold(ctx, &st, h2 + 7, 66u) == 65 && toks_stream_hold(ctx, &st, h2, 65u) == 65,
          "%s: overlapping moves both ways, the same buffer", name);
    CHECK(h2[0] == 'A' && h2[64] == 'A', "%s: the bytes survive the moves", name);
    /* a corrupted state: the hold's record in st (stream.c sst_x at byte 20: p, cap, n, chk, need lo hi, zeros) */
    static const struct { uint32_t at; uint8_t x; } bad[] = { { 20u, 1u }, { 27u, 0x80u }, { 28u, 1u }, { 34u, 1u },
                                                              { 38u, 1u }, { 44u, 1u }, { 51u, 0x40u }, { 52u, 4u },
                                                              { 60u, 1u }, { 63u, 1u }, { 17u, 0x10u } };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        memcpy(&st2, &st, sizeof st);
        ((uint8_t *)&st2)[bad[i].at] ^= bad[i].x;
        toks_stream st3 = st2;
        CHECK(toks_stream_push(ctx, &st2, run, 1u, o, sizeof o) == TOKS_E_ARG && toks_stream_flush(ctx, &st2, o, sizeof o) ==
              TOKS_E_ARG && toks_stream_hold(ctx, &st2, h1, sizeof h1) == TOKS_E_ARG && memcmp(&st2, &st3, sizeof st2) == 0,
              "%s: a corrupted hold record %zu", name, i);
    }
    int64_t f = toks_stream_flush(ctx, &st, o, sizeof o), dn = toks_decode(ctx, run, 65u, 0u, d, sizeof d);
    CHECK(f == dn && f > 0 && memcmp(o, d, (size_t)f) == 0, "%s: the flush of 65 held bytes", name);
    toks_stream_init(ctx, &fresh, 0u);
    CHECK(toks_stream_hold(ctx, &fresh, h2, 65u) == 0 && memcmp(&st, &fresh, sizeof st) == 0, "%s: a flush keeps the hold", name);
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == 0, "%s: NULL / 0 after the flush", name);
    toks_stream_init(ctx, &fresh, 0u);
    CHECK(memcmp(&st, &fresh, sizeof st) == 0, "%s: NULL / 0 is st's own bytes again", name);
    /* push and flush on a copy of st share its hold (toks.h): the copy's push writes it */
    CHECK(toks_stream_hold(ctx, &st, h2, sizeof h2) == 0 && toks_stream_push(ctx, &st, run, 50u, o, sizeof o) == 0,
          "%s: 50 held", name);
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_flush(ctx, &st2, o, sizeof o) == toks_decode(ctx, run, 50u, 0u, d, sizeof d) &&
          toks_stream_flush(ctx, &st, o, sizeof o) == toks_decode(ctx, run, 50u, 0u, d, sizeof d), "%s: two flushes", name);
}

/* the hold's moves with a run of 65 DISTINCT byte tokens (0x30..0x70): a copy over the overlap in the wrong direction
 * smears one byte across the rest, which a run of one repeated byte cannot show. 1. test_hold_api's overlapping moves
 * again (and a 2-byte overlap at the buffer's end), the hold's bytes checked after each, then the flush against
 * toks_decode. 2. the move back into st's own bytes with a run in flight: 30 bytes moved out, 10 more pushed, 40
 * back; st then equals a stream that never left its own bytes, byte for byte; exactly 44 go back, 45 are
 * TOKS_E_LIMIT with st and the hold unchanged. 3. a hold left behind (moved out of, or before toks_stream_init) is
 * neither read nor written again. */
static void test_hold_moves(const toks_ctx *ctx, const char *name, const uint32_t *bid)
{
    enum { N = 65 };
    uint32_t run[N];
    uint8_t txt[N], h1[64], h2[128], snap[64], o[1024], d[1024];
    for (int i = 0; i < N; i++) {
        txt[i] = (uint8_t)(0x30 + i);
        if (bid[txt[i]] == UINT32_MAX) { return; }
        run[i] = bid[txt[i]];
    }
    toks_stream st, st2, own;
    /* 1. the overlapping moves */
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_hold(ctx, &st, h2, sizeof h2) == 0 && toks_stream_push(ctx, &st, run, N, o, sizeof o) == 0 &&
          memcmp(h2, txt, N) == 0, "%s: %d distinct run bytes in a hold", name, N);
    static const struct { uint32_t off, cap; } mv[] = { { 1u, 127u }, { 0u, 128u }, { 7u, 66u }, { 0u, 65u },
                                                        { 63u, 65u }, { 0u, 128u } };
    for (size_t i = 0; i < sizeof mv / sizeof mv[0]; i++) {
        int64_t r = toks_stream_hold(ctx, &st, h2 + mv[i].off, mv[i].cap);
        CHECK(r == N && memcmp(h2 + mv[i].off, txt, N) == 0, "%s: move %zu to h2 + %u: %" PRId64 ", the bytes %s", name, i,
              mv[i].off, r, memcmp(h2 + mv[i].off, txt, N) == 0 ? "intact" : "damaged");
    }
    int64_t f = toks_stream_flush(ctx, &st, o, sizeof o), dn = toks_decode(ctx, run, N, 0u, d, sizeof d);
    CHECK(f == dn && f > 0 && memcmp(o, d, (size_t)f) == 0, "%s: the flush after the moves equals decode", name);
    /* 2. back into st's own bytes with a run in flight */
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_push(ctx, &st, run, 30u, o, sizeof o) == 0 && toks_stream_hold(ctx, &st, h1, sizeof h1) == 30 &&
          toks_stream_push(ctx, &st, run + 30, 10u, o, sizeof o) == 0, "%s: 30 bytes moved out, 10 more pushed", name);
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == 40, "%s: 40 bytes back into st's own", name);
    toks_stream_init(ctx, &own, 0u);
    CHECK(toks_stream_push(ctx, &own, run, 40u, o, sizeof o) == 0 && memcmp(&st, &own, sizeof st) == 0,
          "%s: st equals the stream that never left its own bytes", name);
    f = toks_stream_flush(ctx, &st, o, sizeof o), dn = toks_decode(ctx, run, 40u, 0u, d, sizeof d);
    CHECK(f == dn && f > 0 && memcmp(o, d, (size_t)f) == 0, "%s: the flush of the 40 bytes equals decode", name);
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_hold(ctx, &st, h1, sizeof h1) == 0 && toks_stream_push(ctx, &st, run, 44u, o, sizeof o) == 0 &&
          toks_stream_hold(ctx, &st, NULL, 0u) == 44, "%s: exactly 44 bytes go back into st's own", name);
    toks_stream_init(ctx, &own, 0u);
    CHECK(toks_stream_push(ctx, &own, run, 44u, o, sizeof o) == 0 && memcmp(&st, &own, sizeof st) == 0,
          "%s: the 44 bytes back: st equals the stream that never left its own bytes", name);
    CHECK(toks_stream_hold(ctx, &st, h1, sizeof h1) == 44 && toks_stream_push(ctx, &st, run + 44, 1u, o, sizeof o) == 0,
          "%s: 45 held", name);
    memcpy(&st2, &st, sizeof st);
    memcpy(snap, h1, sizeof h1);
    CHECK(toks_stream_hold(ctx, &st, NULL, 0u) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0 &&
          memcmp(snap, h1, sizeof h1) == 0, "%s: 45 bytes do not go back: TOKS_E_LIMIT, st and the hold unchanged", name);
    f = toks_stream_flush(ctx, &st, o, sizeof o), dn = toks_decode(ctx, run, 45u, 0u, d, sizeof d);
    CHECK(f == dn && f > 0 && memcmp(o, d, (size_t)f) == 0, "%s: the flush of the 45 bytes equals decode", name);
    /* 3. a hold left behind is neither read nor written: after a move to hb the caller reuses ha (filled with a
     * canary), after toks_stream_init on st it reuses hb; the pushes and the flush that follow give decode's bytes,
     * leave the old buffer as the caller filled it, and the init'ed st equals a fresh stream */
    uint8_t ha[128], hb[128], canary[128];
    for (int i = 0; i < 128; i++) { canary[i] = (uint8_t)(0xC3 ^ i); }
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_hold(ctx, &st, ha, sizeof ha) == 0 && toks_stream_push(ctx, &st, run, 20u, o, sizeof o) == 0 &&
          toks_stream_hold(ctx, &st, hb, sizeof hb) == 20, "%s: 20 bytes held in ha, moved to hb", name);
    memcpy(ha, canary, sizeof ha);
    int64_t p1 = toks_stream_push(ctx, &st, run + 20, N - 20u, o, sizeof o);
    f = p1 >= 0 ? toks_stream_flush(ctx, &st, o + p1, sizeof o - (size_t)p1) : -1;
    dn = toks_decode(ctx, run, N, 0u, d, sizeof d);
    CHECK(p1 >= 0 && f >= 0 && p1 + f == dn && memcmp(o, d, (size_t)dn) == 0 && memcmp(ha, canary, sizeof ha) == 0,
          "%s: after the move: %" PRId64 " + %" PRId64 " bytes, decode %" PRId64 ", ha %s", name, p1, f, dn,
          memcmp(ha, canary, sizeof ha) == 0 ? "untouched" : "touched");
    toks_stream_init(ctx, &st, 0u);
    memcpy(hb, canary, sizeof hb);
    toks_stream_init(ctx, &own, 0u);
    CHECK(memcmp(&st, &own, sizeof st) == 0, "%s: toks_stream_init after a hold: st equals a fresh stream", name);
    p1 = toks_stream_push(ctx, &st, run, 30u, o, sizeof o);
    f = p1 >= 0 ? toks_stream_flush(ctx, &st, o + p1, sizeof o - (size_t)p1) : -1;
    dn = toks_decode(ctx, run, 30u, 0u, d, sizeof d);
    CHECK(p1 >= 0 && f >= 0 && p1 + f == dn && memcmp(o, d, (size_t)dn) == 0 && memcmp(hb, canary, sizeof hb) == 0,
          "%s: after init: %" PRId64 " + %" PRId64 " bytes, decode %" PRId64 ", hb %s", name, p1, f, dn,
          memcmp(hb, canary, sizeof hb) == 0 ? "untouched" : "touched");
}

/* the maintainer's receipt: U+13000 (F0 93 80 80, 4 byte tokens) twelve times is one valid 48-byte run, over st's own 44.
 * st's own bytes refuse the 45th; a 1 KiB hold takes all of it; the recovery from st's own (four refusals, each
 * answered with a hold one byte larger); 300 of them (1200 bytes) through a 1 KiB hold, doubled once; the same 12
 * with a stray 0x80 after them (the run invalid as a whole: 49 U+FFFD) */
static void test_u13000(const toks_ctx *ctx, const char *name, const uint32_t *bid)
{
    if (bid[0xF0] == UINT32_MAX || bid[0x93] == UINT32_MAX || bid[0x80] == UINT32_MAX) { return; }
    uint64_t n = 1200u;
    uint32_t *u = (uint32_t *)malloc(4u * (n + 1u));
    for (uint64_t c = 0; c < n / 4u; c++) {
        u[4u * c] = bid[0xF0], u[4u * c + 1u] = bid[0x93], u[4u * c + 2u] = bid[0x80], u[4u * c + 3u] = bid[0x80];
    }
    uint8_t txt[48], o[256];
    for (int c = 0; c < 12; c++) { memcpy(txt + 4 * c, "\xF0\x93\x80\x80", 4); }
    uint64_t sb = toks_scratch_bytes(ctx, 48u, 0u);
    void *scr = malloc(sb);
    uint32_t e[128];
    int64_t ne = toks_scratch_init(ctx, scr, sb, 0u) == 0 ? toks_encode(ctx, txt, 48u, TOKS_NO_POSTPROCESS, e, 128u, scr) : -1;
    int byfb = ne >= 48 && memcmp(e + ne - 48, u, 48u * 4u) == 0;
    free(scr);
    toks_stream st, st2;
    toks_stream_init(ctx, &st, 0u);
    for (int i = 0; i < 44; i++) { CHECK(toks_stream_push(ctx, &st, u + i, 1u, o, sizeof o) == 0, "%s: U+13000 push %d", name, i); }
    memcpy(&st2, &st, sizeof st);
    CHECK(toks_stream_push(ctx, &st, u + 44, 1u, o, sizeof o) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0,
          "%s: st's own 44 bytes refuse the 45th of 12 x U+13000", name);
    ref r;
    ref_build_spm(ctx, u, 48u, 0u, &r);
    CHECK(r.hold == 48u && r.nR == 48u && memcmp(r.R, txt, 48) == 0, "%s: decode of 12 x U+13000", name);
    uint64_t l1k = grow_walk(ctx, u, 48u, 0u, &r, 1024u, 0, name), lown = grow_walk(ctx, u, 48u, 0u, &r, 0u, 0, name);
    CHECK(l1k == 0u && lown == 4u, "%s: 12 x U+13000: %" PRIu64 " refusals through 1 KiB, %" PRIu64 " from st's own",
          name, l1k, lown);
    ref_free(&r);
    ref_build_spm(ctx, u, n, 0u, &r);
    CHECK(r.hold == n && r.nR == n, "%s: decode of 300 x U+13000", name);
    uint64_t l300 = grow_walk(ctx, u, n, 0u, &r, 1024u, 1, name);
    CHECK(l300 == 1u, "%s: 300 x U+13000 through a 1 KiB hold: %" PRIu64 " refusals, want 1", name, l300);
    ref_free(&r);
    u[48] = bid[0x80];
    ref_build_spm(ctx, u, 49u, 0u, &r);
    CHECK(r.nR == 49u * 3u && r.L[48] == 0u && r.L[49] == 49u * 3u, "%s: 12 x U+13000 then 0x80: 49 U+FFFD", name);
    CHECK(grow_walk(ctx, u, 49u, 0u, &r, 1024u, 0, name) == 0u, "%s: the invalid tail through 1 KiB", name);
    ref_free(&r);
    printf("  %-8s U+13000 x 12 %s byte fallback: st's own refuses the 45th byte, 1 KiB holds it, 4 refusals from st's own"
           " (each + 1 byte), 300 x through 1 KiB: 1 refusal (doubled)\n", name, byfb ? "encodes by" : "is NOT encoded by");
    free(u);
}

/* ================================================================ real tokenizers */

static char *path_of(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    static char buf[1024];
    if (root != NULL && root[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", root, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
}

static const char *TEXTS[] = {
    "Hello, world! The quick brown fox jumps over the lazy dog. 1234567890 -- naive cafe\xcc\x81 \xc3\xa9t\xc3\xa9.\n",
    "\xe6\xbc\xa2\xe5\xad\x97\xe4\xbb\xae\xe5\x90\x8d\xe4\xba\xa4\xe3\x81\x98\xe3\x82\x8a\xe6\x96\x87\xe3\x80\x82"
    "\xe4\xb8\xad\xe6\x96\x87\xe6\x96\x87\xe6\x9c\xac\xef\xbc\x8c\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 "
    "\xed\x85\x8d\xec\x8a\xa4\xed\x8a\xb8.\n",
    "\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7 family \xf0\x9f\x8f\xb3\xef\xb8\x8f"
    "\xe2\x80\x8d\xf0\x9f\x8c\x88 \xf0\x9f\x98\x80\xf0\x9f\x98\x83 \xf0\x9f\x87\xaf\xf0\x9f\x87\xb5 "
    "\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd\n",
    "Z\xcc\xa4\xcd\x94\xcd\xa7\xcc\x91\xcc\x93" "a\xcc\x88\xcd\x96\xcc\xad l\xcd\xae\xcc\x92 g\xcc\x8c\xcc\x97 "
    "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd7\xa9\xd7\x9c\xd7\x95\xd7\x9d \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8"
    "\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 \xe0\xb8\xaa\xe0\xb8\xa7\xe0\xb8\xb1\xe0\xb8\xaa\xe0\xb8\x94\xe0\xb8\xb5\n",
    "abc\xff\xfe\x80" "def\xe2\x82 \xed\xa0\x80 \xc0\xaf \xf0\x9f\x98 \xf4\x90\x80\x80 end\n",
    "def f(x):\n\treturn x ** 2  # comment\n\n    if (a && b) { return; }\n",
    "<|endoftext|> <|begin_of_text|><|start_header_id|>user<|end_header_id|> hi<|eot_id|> "
    "[gMASK]<sop><|user|>\nhi<|assistant|> <think>ok</think>\n",
};

static uint32_t *encode_all(toks_ctx *ctx, uint64_t *n_out)
{
    char text[4096];
    uint64_t len = 0;
    for (size_t i = 0; i < sizeof TEXTS / sizeof TEXTS[0]; i++) {
        size_t k = strlen(TEXTS[i]);
        memcpy(text + len, TEXTS[i], k);
        len += k;
    }
    uint64_t sb = toks_scratch_bytes(ctx, len, 0u);
    void *scr = malloc(sb);
    uint32_t *ids = (uint32_t *)malloc(4u * (len + 64u));
    int64_t n = -1;
    if (scr != NULL && ids != NULL && toks_scratch_init(ctx, scr, sb, 0u) == 0) {
        n = toks_encode(ctx, text, len, 0u, ids, len + 64u, scr);
    }
    CHECK(n > 0 && (uint64_t)n <= len + 64u, "encode: %" PRId64, n);
    free(scr);
    *n_out = n > 0 ? (uint64_t)n : 0u;
    return ids;
}

static void test_ctx(const char *name, const char *path)
{
    toks_ctx *ctx = NULL;
    FILE *f = fopen(path, "rb");
    if (f == NULL) { printf("  %-7s SKIP: %s missing\n", name, path); return; }
    fclose(f);
    int64_t r = toks_load(&ctx, path, NULL);
    CHECK(r == 0, "%s: load %" PRId64, name, r);
    if (r != 0) { return; }
    toks_info info;
    info.size = (uint32_t)sizeof info;
    toks_get_info(ctx, &info);
    uint32_t nv = info.n_ids;

    /* the bound's weight against the reference: the longest single-id decode */
    uint64_t dmax = 0;
    uint32_t *partial = (uint32_t *)malloc(4u * nv), *high = (uint32_t *)malloc(4u * 256u), *spec = (uint32_t *)malloc(4u * nv);
    uint32_t n_partial = 0, n_high = 0, n_spec = 0;
    uint8_t *tmp = (uint8_t *)malloc(3u * 65536u);
    for (uint32_t id = 0; id < nv; id++) {
        uint64_t len = 0;
        const uint8_t *p = toks_token(ctx, id, &len);
        uint64_t d = p != NULL ? ref_units(p, len, tmp, NULL) : 0u;
        if (d > dmax) { dmax = d; }
        if (p != NULL && (d != len || memcmp(tmp, p, (size_t)len) != 0)) { partial[n_partial++] = id; }  /* not well-formed alone */
        if (p != NULL && len == 1u && p[0] >= 0x80u) { high[n_high++] = id; }
        if (special_bit(ctx, id) || id >= nv - 300u) { spec[n_spec++] = id; }
    }
    int is_spm = ctx->spm != NULL;
    if (is_spm) {                    /* byte tokens stand in for the partial tokens, the specials as they are */
        n_partial = n_high = 0;
        for (uint32_t id = 0; id < nv; id++) {
            if (ref_byte_id(ctx, id) >= 0) { partial[n_partial++] = id; }
        }
        CHECK(toks_stream_bound(ctx, 1u) == ctx->dec_max + 132u && toks_stream_bound(ctx, 0u) == 132u &&
              ctx->dec_max >= 3u, "%s: spm bound(1) = %" PRIu64, name, toks_stream_bound(ctx, 1u));
    } else {
        CHECK(toks_stream_bound(ctx, 1u) == dmax + 3u && toks_stream_bound(ctx, 0u) == 3u,
              "%s: bound(1) = %" PRIu64 ", want %" PRIu64 " + 3", name, toks_stream_bound(ctx, 1u), dmax);
    }

    /* the fast path's table: well-formed alone (the reference) and short: its length and its bytes,
     * zero-padded, in its slot; long: 0xFE when every 16-byte load stays inside tok_bytes; else 0xFF */
    uint32_t n_clean = 0;
    uint64_t tend = ctx->t.tok_off[nv];
    CHECK(!is_spm || (ctx->dec_len == NULL && ctx->dec_slot == NULL), "%s: an spm context has no slots", name);
    for (uint32_t id = 0; id < nv && !is_spm; id++) {
        uint64_t o = ctx->t.tok_off[id], k = ctx->t.tok_off[id + 1u] - o;
        int wf = ref_units(ctx->t.tok_bytes + o, k, tmp, NULL) == k && memcmp(tmp, ctx->t.tok_bytes + o, (size_t)k) == 0;
        uint32_t want = !wf ? 0xFFu : k <= 16u ? (uint32_t)k : o + ((k + 15u) & ~(uint64_t)15u) <= tend ? 0xFEu : 0xFFu;
        uint32_t got = ctx->dec_len != NULL ? ctx->dec_len[id] : 0x100u;
        uint8_t slot[16] = { 0 };
        if (want <= 16u) { memcpy(slot, ctx->t.tok_bytes + o, (size_t)k); }
        if (got != want || (want <= 16u && memcmp(slot, ctx->dec_slot + 16u * id, 16) != 0)) {
            CHECK(0, "%s: dec_len / slot of id %u: %u, want %u", name, id, got, want);
            break;
        }
        n_clean += want != 0xFFu;
    }

    uint64_t n_text = 0;
    uint32_t *text = encode_all(ctx, &n_text);
    uint64_t n_rand = 2000u;
    uint32_t *rnd = (uint32_t *)malloc(4u * n_rand);
    for (uint64_t i = 0; i < n_rand; i++) {
        uint64_t x = guard_rng(&rng) % 10u;
        rnd[i] = x < 3u && n_partial ? partial[guard_rng(&rng) % n_partial]
               : x < 5u && n_high ? high[guard_rng(&rng) % n_high]
               : x < 6u && n_spec ? spec[guard_rng(&rng) % n_spec] : (uint32_t)(guard_rng(&rng) % nv);
    }
    uint64_t p0 = g_partitions;
    for (uint32_t fl = 0; fl < 2u; fl++) {
        check_seq(ctx, text, n_text, fl, 0u, 0, 12);
        check_seq(ctx, rnd, n_rand, fl, 0u, 0, 12);
        check_windows(ctx, text, n_text, fl, 10u, 12, 0);
        check_windows(ctx, rnd, n_rand, fl, 10u, 12, 0);
        check_windows(ctx, rnd, n_rand, fl, 6u, 6, 1);
    }
    if (!is_spm) {                   /* the same through the byte-wise path only (no slots) */
        toks_ctx slow = *ctx;
        slow.dec_slot = NULL;
        slow.dec_len = NULL;
        check_seq(&slow, rnd, n_rand, 0u, 0u, 0, 4);
        check_seq(&slow, text, n_text, TOKS_SKIP_SPECIAL, 0u, 0, 4);
        printf("  %-8s ok: %u ids (%u fast, %u not well-formed alone, %u high bytes), bound(1) %" PRIu64 ", %" PRIu64
               " text ids, %" PRIu64 " partitions\n", name, nv, n_clean, n_partial, n_high, toks_stream_bound(ctx, 1u),
               n_text, g_partitions - p0);
    } else {
        /* a byte run the stream cannot hold: SST_RUN (44) bytes of a valid run are held, one more is
         * TOKS_E_LIMIT with st unchanged; the flush gives what decode gives for those 44 ids */
        uint32_t ba = UINT32_MAX;
        for (uint32_t x = 0; x < n_partial && ba == UINT32_MAX; x++) {
            if (ref_byte_id(ctx, partial[x]) == 'A') { ba = partial[x]; }
        }
        if (ba != UINT32_MAX && spm_bf(ctx) && ctx->spm->has_decoder) {
            uint32_t run[45];
            for (int x = 0; x < 45; x++) { run[x] = ba; }
            toks_stream st, st2;
            uint8_t o1[256], o2[256];
            toks_stream_init(ctx, &st, 0u);
            CHECK(toks_stream_push(ctx, &st, run, 44u, o1, sizeof o1) == 0, "%s: 44 run bytes held", name);
            memcpy(&st2, &st, sizeof st);
            CHECK(toks_stream_push(ctx, &st, run, 1u, o1, sizeof o1) == TOKS_E_LIMIT && memcmp(&st, &st2, sizeof st) == 0,
                  "%s: the 45th run byte is TOKS_E_LIMIT", name);
            int64_t f1 = toks_stream_flush(ctx, &st, o1, sizeof o1), d1 = toks_decode(ctx, run, 44u, 0u, o2, sizeof o2);
            CHECK(f1 == d1 && f1 > 0 && memcmp(o1, o2, (size_t)f1) == 0, "%s: the held run's flush", name);
            toks_stream_init(ctx, &st, 0u);
            CHECK(toks_stream_push(ctx, &st, run, 45u, o1, sizeof o1) == TOKS_E_LIMIT, "%s: 45 run bytes in one push", name);
            /* a run ended inside its push is never held: 45 byte tokens then a string in one push */
            uint32_t ended[46];
            memcpy(ended, run, sizeof run);
            ended[45] = UINT32_MAX;                      /* a string from the text: not a byte token, not special */
            for (uint64_t x = 0; x < n_text && ended[45] == UINT32_MAX; x++) {
                uint64_t len = 0;
                const uint8_t *q = toks_token(ctx, text[x], &len);
                if (q != NULL && len > 0u && ref_byte_token(q, len) < 0 && !special_bit(ctx, text[x])) { ended[45] = text[x]; }
            }
            toks_stream_init(ctx, &st, 0u);
            int64_t p1 = toks_stream_push(ctx, &st, ended, 46u, o1, sizeof o1);
            int64_t f2 = toks_stream_flush(ctx, &st, o1 + (p1 > 0 ? p1 : 0), sizeof o1 - (size_t)(p1 > 0 ? p1 : 0));
            int64_t d2 = toks_decode(ctx, ended, 46u, 0u, o2, sizeof o2);
            CHECK(ended[45] == UINT32_MAX || (p1 >= 0 && f2 >= 0 && p1 + f2 == d2 && memcmp(o1, o2, (size_t)d2) == 0),
                  "%s: a 45-byte run ended in its push", name);
            test_hold_api(ctx, name, ba);
            uint32_t bid[256];
            for (int x = 0; x < 256; x++) { bid[x] = UINT32_MAX; }
            for (uint32_t x = 0; x < n_partial; x++) {
                int bv = ref_byte_id(ctx, partial[x]);
                if (bv >= 0 && bid[bv] == UINT32_MAX) { bid[bv] = partial[x]; }
            }
            test_hold_moves(ctx, name, bid);
            test_u13000(ctx, name, bid);
        }
        printf("  %-8s ok: %u ids (spm, %u byte tokens), bound(1) %" PRIu64 ", %" PRIu64 " text ids, %" PRIu64
               " partitions\n", name, nv, n_partial, toks_stream_bound(ctx, 1u), n_text, g_partitions - p0);
    }

    /* a stream of one tokenizer is refused by another (identities differ) */
    toks_stream st;
    toks_stream_init(ctx, &st, 0u);
    CHECK(toks_stream_push(&HC, &st, NULL, 0u, NULL, 0u) == TOKS_E_ARG, "%s: a stream of another context", name);
    free(tmp);
    free(text);
    free(rnd);
    free(partial);
    free(high);
    free(spec);
    toks_unload(ctx);
}

int main(void)
{
    guard_buf g;
    G_SIZE = 1u << 20;
    uint8_t *gb = guard_alloc(&g, (size_t)G_SIZE, GUARD_END, 0);
    if (gb == NULL) { fprintf(stderr, "guard_alloc failed\n"); return 2; }
    G_END = gb + G_SIZE;
    build_hand();
    test_hand();
    test_api();
    test_hold_wp();
    uint8_t *slot = HC.dec_slot, *dlen = HC.dec_len;      /* again through the byte-wise path only */
    HC.dec_slot = NULL;
    HC.dec_len = NULL;
    test_hand();
    HC.dec_slot = slot;
    HC.dec_len = dlen;
    printf("  hand    ok: %u ids, bound(1) %" PRIu64 ", %zu hand cases\n", H_N, toks_stream_bound(&HC, 1u),
           sizeof HCASES / sizeof HCASES[0]);
    static const char *REAL[] = { "gpt2", "llama3", "glm53",
                                  "qwen38",        /* ByteLevel after NFC */
                                  "gemma4",        /* spm: Replace, ByteFallback, Fuse */
                                  "mistral-v0.3",  /* spm: Replace, ByteFallback, Fuse, Strip */
                                  "llama2",        /* spm: the same chain, normalized=true specials */
    };
    for (size_t i = 0; i < sizeof REAL / sizeof REAL[0]; i++) { test_ctx(REAL[i], path_of(REAL[i])); }
    /* the spm fixtures (tests/data/spm): Metaspace (always, never), no decoder with id holes, Strip, unk */
    static const char *FIX[] = { "gemma4like", "llamalike", "mistrallike", "meta_always_split", "meta_never_split",
                                 "holes_nodec", "unk_fused", "unk_unfused", "replace_chain", "space_in_vocab" };
    for (size_t i = 0; i < sizeof FIX / sizeof FIX[0]; i++) {
        char fp[256];
        snprintf(fp, sizeof fp, "tests/data/spm/%s.json", FIX[i]);
        test_ctx(FIX[i], fp);
    }
    toks_dec_free(&HC);
    guard_free(&g);
    printf("test_stream: %ld checks, %" PRIu64 " partitions, %" PRIu64 " pushes (%" PRIu64 " spm sequences with an open run"
           " over 44 bytes: a caller's hold only; %" PRIu64 " through an exact hold, %" PRIu64 " TOKS_E_LIMIT recoveries),"
           " %d failures\n", checks, g_partitions, g_pushes, g_unholdable, g_held, g_limits, failures);
    return failures != 0;
}
