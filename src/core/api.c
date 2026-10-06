/* api.c: the per-call driver of include/toks.h (docs/notes/c-core.md §api.c.1) */
#include "kernels.h"
#include "cpu.h"
#include "norm.h"
#include "../gen/norm_nfc.h"
#include "spm.h"
#include "bpe.h"
#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
#    include <sanitizer/msan_interface.h>
#    define SCR_DEFINED(p, n) __msan_unpoison((p), (n))   /* MSan builds only (SEAM-H2): see toks_scratch_init */
#  endif
#endif
#ifndef SCR_DEFINED
#  define SCR_DEFINED(p, n) ((void)0)
#endif

/* ---- scratch (the layout and the formula: core.h) -------------------------------------------------- */

/* the norm region's factor (core.h): a normalizing tokenizer's (norm.h), 3 for one that drops bytes (ctx->has_drop:
 * the region holds a piece with its dropped bytes removed, run_drop), else 0: none */
static uint32_t scr_x(const toks_ctx *ctx)
{
    return ctx->nfc != 0u ? TOKS_NORM_X(ctx->nfc) : ctx->has_drop != 0u ? TOKS_NFC_X : 0u;
}

/* the memo's bytes (core.h): wordpiece and unigram never use one, so their scratch has none */
static uint64_t scr_memo(const toks_ctx *ctx, uint32_t flags)
{
    return ctx->wp != NULL || ctx->uni != NULL ? 0u : toks_scr_memo_bytes(flags);
}

uint64_t toks_scratch_bytes(const toks_ctx *ctx, uint64_t max_len, uint32_t flags)
{
    uint64_t n = toks_scr_cache_mib(flags);             /* a cache size init refuses: the default */
    n = toks_scr_cache_ok(n) ? n : 0u;
    if (max_len > TOKS_MAX_TEXT) { max_len = TOKS_MAX_TEXT; }
    /* no ctx: the largest layout (the norm region, the memo, and wordpiece's pieces + copies) */
    return ctx != NULL ? toks_scratch_size(max_len, scr_x(ctx), n) + scr_memo(ctx, flags) + ctx->scr_extra
                       : toks_scratch_size(max_len, TOKS_NFKC_X, n) + toks_scr_memo_bytes(flags) + TOKS_WP_SCR_EXTRA;
}

int64_t toks_scratch_init(const toks_ctx *ctx, void *scr, uint64_t bytes, uint32_t flags)
{
    if (ctx == NULL || (scr == NULL && bytes != 0u)) { return TOKS_E_ARG; }
    uint64_t n = toks_scr_cache_mib(flags);             /* the piece caches (core.h) */
    if ((flags & ~(TOKS_SCRATCH_MEMO_MIB(0xFFFu) | TOKS_SCRATCH_CACHE_MIB(0xFFu))) != 0u || !toks_scr_cache_ok(n)) {
        return TOKS_E_ARG;
    }
    uint64_t extra = ctx->scr_extra + scr_memo(ctx, flags);   /* wordpiece's pieces + copies, the memo */
    if (bytes < toks_scratch_size(0u, scr_x(ctx), n) + extra) { return TOKS_E_SCRATCH; }   /* also scr == NULL */

    /* the longest text the buffer holds: the largest L <= TOKS_MAX_TEXT with size(L) <= bytes */
    uint64_t lo = 0u, hi = TOKS_MAX_TEXT;
    while (lo < hi) {                                   /* bound: 30 halvings of [0, 2^29] */
        uint64_t mid = lo + ((hi - lo + 1u) >> 1);
        if (toks_scratch_size(mid, scr_x(ctx), n) + extra <= bytes) { lo = mid; } else { hi = mid - 1u; }
    }
    extra = ctx->scr_extra;
    uint64_t base = (uint64_t)(uintptr_t)scr;
    uint64_t a = toks_align64(base) - base;              /* 0..63 */
    uint8_t *s0 = (uint8_t *)scr;
    toks_scratch *h = (toks_scratch *)(void *)(s0 + a);
    /* the caches (kernels.md §7): when scr already holds a scratch toks laid out with these flags, they hold only
     * entries tagged with epochs (generations) up to the header's: the next ones empty them in O(1). Else zeroed. */
    uint64_t epoch = 0u, gen = 0u, memo = scr_memo(ctx, flags);
    SCR_DEFINED(h, sizeof *h);    /* the test below reads a caller's buffer on purpose: sound (garbage that passes it
                                     names this buffer, whose caches then hold only older epochs), never UB (no trap
                                     values), and MSan is told so */
    if (h->magic == TOKS_SCRATCH_MAGIC && h->base == base && h->bytes == bytes && h->off_cache == a + TOKS_SCR_FIXED &&
        h->cache_mib == n && h->off_work == h->off_cache + toks_scr_caches(n) + memo && h->epoch >= 1u &&
        h->epoch < (uint64_t)TOKS_TAG_MAX && (n == 0u || (h->long_gen >= 1u && h->long_gen < 0xFFFFFFFFu)) &&
        (h->off_long != 0u) == (n != 0u && ctx->spm == NULL)) {   /* the long region: K5's, or spm buckets */
        epoch = h->epoch + 1u;
        gen = h->long_gen + 1u;
    }
    memset(h, 0, sizeof *h);                             /* the counters line starts at 0 */
    h->magic = TOKS_SCRATCH_MAGIC;
    h->identity = ctx->identity;
    h->base = base;
    h->bytes = bytes;
    h->max_len = lo;
    h->off_cache = a + TOKS_SCR_FIXED;
    h->off_work = h->off_cache + toks_scr_caches(n) + memo;     /* the memo between the caches and work */
    h->off_bounce = h->off_work + toks_scr_work(toks_scr_tmax(lo, scr_x(ctx))) + extra;
    h->epoch = epoch != 0u ? epoch : 1u;
    h->cache_mib = n;
    h->off_long = n != 0u && ctx->spm == NULL ? h->off_cache + toks_scr_short(n) : 0u;   /* spm: one word cache */
    h->long_gen = n != 0u ? (epoch != 0u ? gen : 1u) : 0u;
    toks_scr_carved(ctx, h);                            /* the guard build's regions (core.h toks_scr_p) */
    if (epoch == 0u && n != 0u) { toks_plat_hint_huge(s0 + h->off_cache, toks_scr_caches(n)); }   /* before the touch */
    if (epoch == 0u) {
        toks_scr_zero(h, s0 + h->off_cache, h->off_long != 0u ? toks_scr_short(n) + toks_scr_long_buckets(n) : toks_scr_caches(n));
    }
    if (memo != 0u) {                                   /* its header (first init: and slots; the ring is read only
                                                           through them); else the epoch empties it */
        toks_scr_zero(h, s0 + h->off_cache + toks_scr_caches(n), epoch == 0u ? 64u + memo / 32u : 64u);
    }
    return 0;
}

/* the header of a scratch bound to ctx and laid out for texts of len bytes, else NULL. Only the
 * header's line 0 (the 64 bytes at the first 64-aligned address) is read before the binding is proven,
 * so any buffer of at least 127 bytes that init never bound is refused without a read past its end. */
static toks_scratch *scratch_get(const toks_ctx *ctx, void *scr, uint64_t len)
{
    if (scr == NULL) { return NULL; }
    uint64_t base = (uint64_t)(uintptr_t)scr;
    toks_scratch *h = toks_scr_header(scr);
    if (h->magic != TOKS_SCRATCH_MAGIC || h->identity != ctx->identity || h->base != base ||
        len > h->max_len) {
        return NULL;
    }
    return h;
}

/* ---- encode / pieces (kernels.md §7 steps 1-5) ----------------------------------------------------- */

/* the output cursor: counts everything, stores what fits in [0, cap). */
typedef toks_emit emit;                 /* core.h; its limit is wordpiece's and unigram's, emit1 never reads it */

static void emit1(emit *e, uint32_t v)
{
    if (e->n < e->cap) { toks_st32(e->out + e->n, v); }
    e->n++;
}

/* K5 over the pieces text[start, ends[0]), [ends[0], ends[1]), ... (n >= 1) into out or the bounce (kernels.md §7) */
static void k5_run(const toks_ctx *ctx, toks_scratch *h, const uint8_t *text, uint64_t len, const uint32_t *ends,
                   uint64_t n, uint64_t start, emit *e)
{
    uint8_t *s = (uint8_t *)(uintptr_t)h->base;
    uint64_t left = (e->n < e->cap) ? e->cap - e->n : 0u;
    int direct = left >= (ends[n - 1u] - start) + 4u;
    uint32_t *bounce = (uint32_t *)(void *)toks_scr_p(h, s + h->off_bounce);
    toks_k5_args k;
    k.text = text;
    k.len = len;
    k.ends = ends;
    k.n = n;
    k.start = start;
    k.out = direct ? e->out + e->n : bounce;
    k.room = direct ? left : toks_scr_tmax(h->max_len, scr_x(ctx)) + 4u;   /* the bounce (core.h) */
    k.n_out = 0u;
    /* a warm scratch (K5's test, kernels.md §6) gets the whole short cache and the long one, a fresh one 2 MiB */
    int warm = h->hits_static + h->hits_cache + h->misses >= TOKS_K5_WARM;
    k.cache = toks_scr_p(h, s + h->off_cache);
    k.cache_mask = TOKS_TEST_DEGEN((warm ? toks_scr_short(h->cache_mib) : TOKS_CACHE_BYTES) / TOKS_BUCKET - 1u);
    k.work = toks_scr_p(h, s + h->off_work);
    k.work_bytes = h->off_bounce - h->off_work - ctx->scr_extra;   /* the generic engine's lists follow (gen.c) */
    k.hits_static = h->hits_static;             /* in-out: K5 adds this round's counts */
    k.hits_cache = h->hits_cache;
    k.misses = h->misses;
    k.cache_tag = h->epoch;                     /* entries of earlier epochs are misses (kernels.md §7) */
    uint64_t nb = toks_scr_long_buckets(h->cache_mib);  /* the long cache: pointers and sizes from the header */
    toks_lcache lc = { toks_scr_p(h, s + h->off_long), toks_scr_p(h, s + h->off_long + nb), TOKS_TEST_DEGEN(nb / TOKS_BUCKET - 1u),
                       toks_scr_long_arena(h->cache_mib), h->long_pos, h->long_gen, 0u, 0u };
    k.lcache = h->off_long != 0u && warm ? &lc : NULL;
    uint64_t m = toks_k5(&ctx->t, &k, ctx->tier);
    h->hits_static = k.hits_static;
    h->hits_cache = k.hits_cache;
    h->misses = k.misses;
    h->long_pos = lc.pos;
    h->long_gen = lc.gen;
    if (!direct && left != 0u) {
        toks_cpy(e->out + e->n, bounce, (left < m ? left : m) * 4u);   /* out: any alignment */
    }
    e->n += m;
}

static int dropped(const toks_ctx *ctx, uint8_t c) { return (ctx->drop[c >> 3] >> (c & 7u)) & 1u; }

static int any_dropped(const toks_ctx *ctx, const uint8_t *p, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) {                  /* bound: n */
        if (dropped(ctx, p[i])) { return 1; }
    }
    return 0;
}

/* a K3 round seg[pos, ends[n - 1]) holding a byte the vocab lacks (ctx->has_drop; config.c): clean pieces through
 * K5 as usual, a dirty one compacted (the norm region, or in place) into one K5 piece (kernels.md §7 "Dropped") */
static void run_drop(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t pos, const uint32_t *ends,
                     uint64_t n, emit *e)
{
    uint8_t *norm = toks_scr_p(h, (uint8_t *)(uintptr_t)h->base + h->off_bounce + toks_scr_bounce(toks_scr_tmax(h->max_len, scr_x(ctx))));
    int in_norm = seg >= norm && seg < norm + toks_scr_norm(h->max_len, scr_x(ctx));
    uint64_t i = 0u, st = pos;
    while (i < n) {                                     /* bound: n pieces, >= 1 per pass */
        uint64_t j = i, a = st;
        while (j < n && !any_dropped(ctx, seg + a, ends[j] - a)) { a = ends[j]; j++; }   /* bound: n */
        if (j > i) {
            k5_run(ctx, h, seg, ends[j - 1u], ends + i, j - i, st, e);
            st = ends[j - 1u];
        }
        if (j == n) { break; }
        uint8_t *dst = in_norm ? (uint8_t *)(uintptr_t)(seg + st) : norm;
        uint32_t m = 0u, unk = 0u;
        for (uint64_t k = st; k < ends[j]; k++) {      /* bound: the piece */
            if (!dropped(ctx, seg[k])) {
                dst[m++] = seg[k];
                unk = 0u;
            } else if (ctx->drop_unk != 0u) {           /* hf's unk symbol: no merge crosses it (config.c) */
                if (m != 0u) { k5_run(ctx, h, dst, m, &m, 1u, 0u, e); }
                if (unk == 0u || ctx->drop_fuse == 0u) { emit1(e, ctx->drop_unk - 1u); }
                m = 0u;
                unk = 1u;
            }
        }
        if (m != 0u) { k5_run(ctx, h, dst, m, &m, 1u, 0u, e); }
        st = ends[j];
        i = j + 1u;
    }
}

void toks_round(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len, uint64_t pos,
                const uint32_t *ends, uint64_t n, uint64_t base, emit *e, int ids)
{
    if (ids == 0) {
        for (uint64_t i = 0u; i < n; i++) { emit1(e, (uint32_t)(base + ends[i])); }      /* bound: n */
    } else if (ctx->has_drop != 0u && any_dropped(ctx, seg + pos, ends[n - 1u] - pos)) {
        run_drop(ctx, h, seg, pos, ends, n, e);         /* rare: the round holds a byte the vocab lacks */
    } else {
        k5_run(ctx, h, seg, len, ends, n, pos, e);
    }
}

/* one text segment seg[0, len) at text offset base: K3 rounds of <= TOKS_CHUNK_PIECES pieces, each followed by K5
 * (ids, k5_run) or by the round's piece ends (pieces) (kernels.md §7 step 3); a chain no template compiles: gen.c */
static void run_text(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len,
                     uint64_t base, emit *e, int ids, int at_start)
{
    uint8_t *s = (uint8_t *)(uintptr_t)h->base;
    uint32_t *ends = (uint32_t *)(void *)toks_scr_p(h, (uint8_t *)h + TOKS_SCR_HDR);
    if (ctx->spm != NULL) {             /* sentencepiece-style bpe: one call per text unit (spm.h) */
        if (ids != 0) {
            e->n = toks_spm_encode(&ctx->t, ctx->spm, seg, len, at_start & 1, e->out, e->cap, e->n, toks_scr_p(h, s + h->off_cache),
                                   TOKS_TEST_DEGEN(toks_scr_caches(h->cache_mib) / TOKS_BUCKET - 1u), h->epoch,
                                   toks_scr_p(h, s + h->off_work), ((uint32_t)at_start & TOKS_SPM_NOPFX) | TOKS_SPM_TIER(ctx->tier));
        } else {
            e->n = toks_spm_pieces(ctx->spm, seg, len, at_start & 1, base, e->out, e->cap, e->n);
        }
        return;
    }
    if (ctx->gen != NULL) {
        toks_gen_run(ctx, h, seg, len, base, e, ids);
        return;
    }
    uint64_t pos = 0u;
    uint32_t tier = (ctx->t.flags & TOKS_TF_TWIN) != 0u ? TOKS_TIER_SCALAR : ctx->tier;   /* layout.h TOKS_TF_TWIN */
    while (pos < len) {                     /* bound: len; every round covers >= 1 byte (K3 contract) */
        uint64_t n, end;
        if (ctx->t.tmpl != TOKS_TMPL_NONE) {   /* cl100k, o200k or dsv3: the template's K3 */
            toks_k3_args k;
            k.text = seg;
            k.len = len;
            k.pos = pos;
            k.ends = ends;
            k.cap = (uint64_t)TOKS_CHUNK_PIECES;
            k.n = 0u;
            k.flags = 0u;
            k.rsv = 0u;
            n = (ctx->t.tmpl == TOKS_TMPL_O200K) ? toks_k3_o200k(&ctx->t, &k, tier)
              : (ctx->t.tmpl == TOKS_TMPL_DSV3)  ? toks_k3_dsv3(&ctx->t, &k, tier)
                                                 : toks_k3(&ctx->t, &k, tier);
            end = k.pos;
            if (n == 0u || end <= pos || end > len) { return; }   /* unreachable for a correct K3 */
        } else {                            /* TOKS_TMPL_NONE: the segment is one piece */
            ends[0] = (uint32_t)len;
            n = 1u;
            end = len;
        }
        toks_round(ctx, h, seg, len, pos, ends, n, base, e, ids);
        pos = end;
    }
}

/* phase-1 tokens (none in mode NONE) split seg[0, n) into units for K3/K5 (docs/notes/c-core.md §api.c.2) */
static void run_units(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t n, uint64_t nb,
                      uint32_t mode, emit *e, int ids, int at_start)
{
    if (mode == TOKS_ADDED_NONE || (ctx->t.add_phases & 2u) == 0u) {   /* no phase-1 token can match: one unit */
        run_text(ctx, h, seg, n, nb, e, ids, at_start);
        return;
    }
    toks_seg_iter it1;
    toks_seg_out v;
    toks_seg_begin(&it1, &ctx->t, mode, 1u, ctx->tier, seg, n);
    int pfx = ctx->spm != NULL && ctx->spm->pfx_mode == TOKS_SPM_PFX_GAP;   /* Prepend: the gap's first unit (spm.h) */
    while (toks_seg_next(&it1, &v)) {                   /* bound: <= 2 n + 1 units */
        if (v.kind == TOKS_SEG_TOKEN) {
            if (pfx && v.start == 0u && (seg[0] == 0x20u || toks_meta_at(seg, n, 0u))) {   /* segment.c §5: a lone ▁ */
                if (ids) { run_text(ctx, h, (const uint8_t *)"\xE2\x96\x81", 3u, nb, e, ids, (int)TOKS_SPM_NOPFX); }
                else { emit1(e, (uint32_t)(nb + (seg[0] == 0x20u ? 1u : 3u))); }   /* hf: aligned to that char */
            }
            emit1(e, ids ? v.id : (uint32_t)(nb + v.end));
        } else {
            run_text(ctx, h, seg + v.start, v.end - v.start, nb + v.start, e, ids,
                     (at_start && v.start == 0u) | (pfx && v.start != 0u ? (int)TOKS_SPM_NOPFX : 0));
        }
    }
}

/* a text gap g[0, n) whose normalized form starts at offset nb of the normalized stream (kernels.md §7 step 2 and
 * "NFC gaps"): its normalized length, or toks_norm's error. at_start: the gap begins at input offset 0 of a call
 * without TOKS_CONTINUATION (sentencepiece-style bpe's prefix goes to the unit at that offset, spm.h). A gap without a
 * hot byte of the form (K2) is left alone without a plan (toks_nfc_scan's first step); a memo hit never gets here. */
static int64_t run_gap(const toks_ctx *ctx, toks_scratch *h, const uint8_t *g, uint64_t n, uint64_t nb,
                       uint32_t mode, emit *e, int ids, int at_start)
{
    toks_nfc_plan pl;
    pl.d0 = n;
    uint8_t hb = (ctx->nfc & TOKS_NS_COMPAT) != 0u ? TOKS_NFKC_HOT_BYTE : TOKS_NFC_HOT_BYTE;   /* gen/norm_nfc.h */
    if (ctx->nfc != 0u && toks_k2(g, 0u, n, hb) < n) { toks_nfc_plan_begin(&pl, ctx->nfc, g, n); }
    if (pl.d0 == n) {                                   /* nothing to normalize: in place, zero copies */
        run_units(ctx, h, g, n, nb, mode, e, ids, at_start);
        return (int64_t)n;
    }
    uint8_t *buf = toks_scr_p(h, (uint8_t *)(uintptr_t)h->base + h->off_bounce + toks_scr_bounce(toks_scr_tmax(h->max_len, scr_x(ctx))));
    int64_t m;
    if ((mode != TOKS_ADDED_NONE && (ctx->t.add_phases & 2u) != 0u) || ctx->gen != NULL) {
        m = toks_norm(ctx->nfc, g, n, buf, TOKS_NORM_BOUND(ctx->nfc, n));   /* phase-1 tokens or a generic chain: all */
        if (m < 0) { return m; }                        /* unreachable: the call meets toks_norm's preconditions */
        run_units(ctx, h, buf, (uint64_t)m, nb, mode, e, ids, at_start);
        return m;
    }
    uint64_t r = 0u, o = nb, s, t;                      /* g[0, r) is done, its normalized form ends at o */
    while (toks_nfc_plan_next(&pl, &ctx->t, g, n, &s, &t)) {    /* bound: every stretch takes >= 1 changed run */
        if (s > r) {                                    /* in place */
            run_text(ctx, h, g + r, s - r, o, e, ids, at_start && r == 0u);
            o += s - r;
        }
        m = toks_norm(ctx->nfc, g + s, t - s, buf, TOKS_NORM_BOUND(ctx->nfc, t - s));   /* [s, t): the norm region */
        if (m < 0) { return m; }                        /* unreachable */
        run_text(ctx, h, buf, (uint64_t)m, o, e, ids, at_start && s == 0u);
        o += (uint64_t)m;
        r = t;
    }
    if (r < n) {                                        /* in place */
        run_text(ctx, h, g + r, n - r, o, e, ids, at_start && r == 0u);
        o += n - r;
    }
    return (int64_t)(o - nb);
}

/* ---- a tiktoken wrapper's cuts (config.h cut_run; docs/models/kimi.md §2.3) ------------------------ */

/* python's str.isspace() (python 3.10 .. 3.14: 29 code points): the pattern's \s plus U+001C..U+001F. */
static int py_isspace(uint32_t c)
{
    if (c < 0x80u) { return (c >= 0x09u && c <= 0x0Du) || (c >= 0x1Cu && c <= 0x20u); }
    return c == 0x85u || c == 0xA0u || c == 0x1680u || (c >= 0x2000u && c <= 0x200Au) || c == 0x2028u ||
           c == 0x2029u || c == 0x202Fu || c == 0x205Fu || c == 0x3000u;
}

/* run_cuts' walk over 8 ascii atoms w (each byte < 0x80, so no sum below carries into the next byte) inside a chunk:
 * 1 with the run and its class moved past them, or 0 when a run could pass cut_run inside (one by one, then; a run
 * that starts inside is at most 8 long, so cut_run >= 8 cuts nothing there) */
static inline int run8(uint64_t w, uint32_t *run, int *sp_run, uint32_t cut_run)
{
    uint64_t s = (((w + 0x7777777777777777ull) & ~(w + 0x7272727272727272ull)) |      /* py_isspace: 09-0D, 1C-20 */
                  ((w + 0x6464646464646464ull) & ~(w + 0x5F5F5F5F5F5F5F5Full))) & 0x8080808080808080ull;
    uint64_t o = (*sp_run != 0 ? ~s : s) & 0x8080808080808080ull;                    /* the bytes that end the run */
    if (cut_run < 8u || *run + (o != 0u ? (uint32_t)__builtin_ctzll(o) >> 3 : 8u) > cut_run) { return 0; }
    if (o == 0u) {
        *run += 8u;
        return 1;
    }
    *sp_run = (int)(s >> 63);                           /* it ends inside: the last byte's class, and its stretch */
    o = (*sp_run != 0 ? ~s : s) & 0x8080808080808080ull;
    *run = o != 0u ? (uint32_t)__builtin_clzll(o) >> 3 : 8u;
    return 1;
}

/* one cut piece seg[0, len) at text offset base, encoded alone as tiktoken.encode(piece) is: with specials
 * (modes ALL and NONSPECIAL) the phase-1 names split it first, every one allowed (allowed_special="all"),
 * and the pattern runs on each gap; without (NONE) the pattern runs on all of it. */
static void run_piece(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len, uint64_t base,
                      int specials, emit *e, int ids, int at_start)
{
    if (specials == 0 || (ctx->t.add_phases & 2u) == 0u) {
        run_text(ctx, h, seg, len, base, e, ids, at_start);
        return;
    }
    toks_seg_iter it;
    toks_seg_out v;
    toks_seg_begin(&it, &ctx->t, TOKS_ADDED_ALL, 1u, ctx->tier, seg, len);
    while (toks_seg_next(&it, &v)) {                    /* bound: <= 2 len + 1 units */
        if (v.kind == TOKS_SEG_TOKEN) {
            emit1(e, ids ? v.id : (uint32_t)(base + v.end));
        } else {
            run_text(ctx, h, seg + v.start, v.end - v.start, base + v.start, e, ids, at_start && v.start == 0u);
        }
    }
}

/* t[0, len) at text offset base, cut where the wrapper cuts (docs/notes/c-core.md §api.c.3) */
static void run_cuts(const toks_ctx *ctx, toks_scratch *h, const uint8_t *t, uint64_t len, uint64_t base,
                     int specials, emit *e, int ids, int at_start)
{
    uint64_t piece = 0u, pos = len <= ctx->cut_run && len <= ctx->cut_chunk ? len : 0u;   /* too short to cut: no walk */
    uint32_t in_chunk = 0u, run = 0u;
    int sp_run = 0;
    while (pos < len) {                                 /* bound: len (>= 1 byte per atom) */
        uint32_t k = toks_utf8_len(t + pos, len - pos), cp = t[pos];
        if (k == 1u) {
            if (len - pos >= 8u && in_chunk + 8u <= ctx->cut_chunk) {   /* 8 ascii atoms at once, no chunk edge inside */
                uint64_t w = bpe_load_le(t + pos, 8u);
                if ((w & 0x8080808080808080ull) == 0u && run8(w, &run, &sp_run, ctx->cut_run)) {
                    in_chunk += 8u;
                    pos += 8u;
                    continue;
                }
            }
        } else if (k > 1u) {                            /* only leads C2, E1, E2, E3 begin one of the 29 spaces */
            cp = cp == 0xC2u || cp - 0xE1u < 3u ? toks_cp_decode(t + pos, k) : 0xFFFFFFFFu;
        } else {
            k = 1u;
            cp = 0xFFFFFFFFu;                           /* an ill-formed byte: no space */
        }
        int sp = py_isspace(cp);
        if (in_chunk == ctx->cut_chunk) {               /* the next chunk starts at this code point */
            run_piece(ctx, h, t + piece, pos - piece, base + piece, specials, e, ids, at_start && piece == 0u);
            piece = pos;
            in_chunk = 0u;
        }
        if (in_chunk == 0u || sp != sp_run) {
            run = 1u;
            sp_run = sp;
        } else if (++run > ctx->cut_run) {              /* a cut before this code point */
            run_piece(ctx, h, t + piece, pos - piece, base + piece, specials, e, ids, at_start && piece == 0u);
            piece = pos;
            run = 1u;
        }
        in_chunk++;
        pos += k;
    }
    if (len != 0u) { run_piece(ctx, h, t + piece, len - piece, base + piece, specials, e, ids, at_start && piece == 0u); }
}

/* the segment memo (SPEC §6, kernels.md §7 "the segment memo"; docs/notes/c-core.md §api.c.4) */
#define MEMO_MIN 256u                                   /* the shortest segment recorded */
typedef struct memo_rec { uint64_t hash, pos; uint32_t key, n_ids, len, epoch; } memo_rec;   /* a slot; a record's head */
static inline uint64_t memo_ring(uint64_t mb) { return mb - 64u - (mb >> 5); }
static inline uint64_t memo_need(uint64_t n, uint64_t k) { return (32u + ((n + 7u) & ~7ull) + 4u * k + 63u) & ~63ull; }
static inline memo_rec *memo_at(uint8_t *m, uint64_t mb, uint64_t p) { return (memo_rec *)(void *)(m + 64u + (mb >> 5) + p); }
/* the set of a hash: two slots in one 64-byte line, mb >> 11 sets (a probe reads the line either way) */
static inline memo_rec *memo_set(uint8_t *m, uint64_t mb, uint64_t hash)
{
    return (memo_rec *)(void *)(m + 64u + 64u * (((hash >> 32) * (mb >> 11)) >> 32));
}

/* the slot of set st that a segment's mark (rec 0) or record (rec 1) takes: the one holding its hash, key and length
 * (for a mark: unless that is a live record, another segment's whose bytes differ), else an empty or dead one (another
 * epoch's, or a record the write position pos lapped), else the older mark, else for a record the older live record;
 * NULL when a mark would replace a live record (kernels.md §7) */
static memo_rec *memo_way(memo_rec *st, uint64_t hash, uint32_t key, uint64_t n, uint32_t epoch, uint64_t pos,
                          uint64_t ring, int rec)
{
    int mark[2];
    for (uint32_t w = 0; w < 2u; w++) {                 /* bound: 2 ways */
        if (st[w].hash == hash && st[w].key == key && st[w].len == n &&
            (rec != 0 || st[w].n_ids == UINT32_MAX || st[w].epoch != epoch)) {
            return &st[w];
        }
    }
    for (uint32_t w = 0; w < 2u; w++) {                 /* bound: 2 ways */
        const memo_rec *x = &st[w];
        mark[w] = x->n_ids == UINT32_MAX;
        if (x->len == 0u || x->epoch != epoch ||
            (!mark[w] && (pos - x->pos > ring || x->pos % ring + memo_need(x->len, x->n_ids) > ring))) {
            return &st[w];
        }
    }
    if (mark[0] != mark[1]) { return mark[0] ? &st[0] : &st[1]; }
    if (mark[0] == 0 && rec == 0) { return NULL; }
    return st[0].pos <= st[1].pos ? &st[0] : &st[1];
}

/* the keyed 64-bit hash of g[0, n) (n >= MEMO_MIN): its length and three 64-byte windows (the start, the middle and
 * the end) through eight multiply chains, O(1) per segment. It only locates: a hit compares every byte, so segments
 * that share the windows and the length cost a compare, never ids (SPEC §6; the header counts them: differ) */
static uint64_t memo_hash(const uint8_t *g, uint64_t n, uint64_t seed)
{
    uint64_t x[8], w[8];
    const uint8_t *at[3] = { g, g + n / 2u - 32u, g + n - 64u };
    for (uint64_t j = 0; j < 8u; j++) { x[j] = seed ^ (n + j * TOKS_FIB64); }
    for (uint64_t k = 0; k < 3u; k++) {                 /* bound: 3 windows */
        memcpy(w, at[k], 64);
        for (uint64_t j = 0; j < 8u; j++) { x[j] = toks_mix64(x[j], w[j]); }
    }
    for (uint64_t j = 1; j < 8u; j++) { x[0] = toks_mix64(x[0], x[j]); }
    return TOKS_TEST_DEGEN(x[0]);                       /* core.h: SPEC §6's degenerate-hash build */
}

/* m answers g[0, n) under (key, hash): its ids into e, 1; else 0, or 2 for a mark (*mk: its slot). A hit needs a slot
 * of the set with the epoch, hash, key and length, its record not lapped by the write position, and every byte equal;
 * nothing outside m is read. A hit ends the lap's run of refused records and moves vpos as a record would */
static int memo_get(uint8_t *m, uint64_t mb, uint32_t epoch, const uint8_t *g, uint64_t n, uint32_t key, uint64_t hash,
                    emit *e, memo_rec **mk)
{
    uint64_t ring = memo_ring(mb);
    toks_memo_head *hd = (toks_memo_head *)(void *)m;
    memo_rec *st = memo_set(m, mb, hash), *s = NULL;
    hd->probes++;
    for (uint32_t w = 0; w < 2u && s == NULL; w++) {    /* bound: 2 ways */
        if (st[w].hash == hash && st[w].epoch == epoch && st[w].key == key && st[w].len == n) { s = &st[w]; }
    }
    if (s == NULL) { return 0; }
    if (s->n_ids == UINT32_MAX) {                       /* a mark: a sight not recorded */
        *mk = s;
        return 2;
    }
    if (hd->pos - s->pos > ring || s->pos % ring + memo_need(n, s->n_ids) > ring) { return 0; }
    if (memcmp(memo_at(m, mb, s->pos % ring) + 1, g, (size_t)n) != 0) {
        hd->differ++;
        return 0;
    }
    uint64_t left = e->n < e->cap ? e->cap - e->n : 0u, k = s->n_ids;
    if (left != 0u) { toks_cpy(e->out + e->n, (uint8_t *)(memo_at(m, mb, s->pos % ring) + 1) + ((n + 7u) & ~7ull),
                               4u * (left < k ? left : k)); }
    e->n += k;
    hd->hits++, hd->drought = 0u, hd->vpos += memo_need(n, k), hd->run = 0u;
    return 1;
}

/* write g[0, n)'s record with its k ids (need bytes, at most half the ring) at the write position, unpublished; run_seg
 * starts each lap at the ring's start, so a record never crosses its end */
static void memo_put(uint8_t *m, uint64_t mb, const uint8_t *g, uint64_t n, uint32_t key, uint64_t hash,
                     const uint32_t *ids, uint64_t k, uint64_t need)
{
    toks_memo_head *hd = (toks_memo_head *)(void *)m;
    memo_rec *r = memo_at(m, mb, hd->pos % memo_ring(mb));
    r->hash = hash, r->key = key, r->n_ids = (uint32_t)k, r->len = (uint32_t)n;
    memcpy(r + 1, g, (size_t)n);
    toks_cpy((uint8_t *)(r + 1) + ((n + 7u) & ~7ull), ids, 4u * k);
    hd->pos += need, hd->drought += need, hd->vpos += need;
}

/* publish the records written since position p, each slot's length last; none if they lapped each other */
static void memo_publish(uint8_t *m, uint64_t mb, uint32_t epoch, uint64_t p)
{
    uint64_t ring = memo_ring(mb), pos = ((const toks_memo_head *)(const void *)m)->pos;
    while (pos - p <= ring && p < pos) {                /* bound: the records the call wrote */
        memo_rec *r = memo_at(m, mb, p % ring);
        if (r->len == 0u) {
            p += ring - p % ring;
            continue;
        }
        memo_rec *s = memo_way(memo_set(m, mb, r->hash), r->hash, r->key, r->len, epoch, pos, ring, 1);
        *s = *r;
        s->pos = p, s->epoch = epoch, s->len = 0u;
        s->len = r->len;
        p += memo_need(r->len, r->n_ids);
    }
}

/* one segment g[0, n) (SPEC §6: a phase-0 unit, or the whole text) at nb: run_gap's ids, or with cut run_cuts' (its
 * specials); through the memo when the scratch has one, the call encodes ids and n >= MEMO_MIN (a hit returns n: only
 * pieces read the normalized length) */
static int64_t run_seg(const toks_ctx *ctx, toks_scratch *h, const uint8_t *g, uint64_t n, uint64_t nb,
                       uint32_t flags, uint32_t mode, emit *e, int ids, int at_start, int cut)
{
    uint64_t mb = 0u, hash = 0u, n0 = e->n;
    uint8_t *m = ids != 0 && n >= MEMO_MIN ? toks_scr_memo(h, &mb) : NULL;
    uint32_t key = flags | (uint32_t)at_start << 8 | (uint32_t)cut << 9 | mode << 10;
    int got = 0;
    memo_rec *mk = NULL;                                /* got 2: the mark's slot */
    if (m != NULL) {
        hash = memo_hash(g, n, h->identity);              /* the ctx's: a text collides alike in every run */
        got = memo_get(m, mb, (uint32_t)h->epoch, g, n, key, hash, e, &mk);
        if (got == 1) { return (int64_t)n; }
    }
    int64_t r = (int64_t)n;
    if (cut != 0) {
        run_cuts(ctx, h, g, n, nb, mode != TOKS_ADDED_NONE, e, ids, at_start);
    } else {
        r = run_gap(ctx, h, g, n, nb, mode, e, ids, at_start);
    }
    if (m != NULL && r >= 0 && e->n <= e->cap) {       /* all its ids in out: record it, or mark it */
        toks_memo_head *hd = (toks_memo_head *)(void *)m;
        uint64_t ring = memo_ring(mb), need = memo_need(n, e->n - n0), vpos = hd->vpos;
        /* admission: open until a ring of records passes without a hit; then a second sight; never a record over half
         * the ring */
        int put = (hd->drought < ring || got == 2) && need <= ring / 2u;
        if (put && hd->pos - hd->lap + need > ring) {  /* a full lap keeps what it holds (kernels.md §7) */
            hd->run += got == 2 && vpos - mk->pos <= ring ? TOKS_MEMO_DRY / TOKS_MEMO_GHOSTS : 1u;
            if (hd->run < TOKS_MEMO_DRY) {
                put = 0;
                hd->vpos += need;
            } else {                                    /* the next lap, from the ring's start */
                if (hd->pos % ring != 0u) {
                    memo_at(m, mb, hd->pos % ring)->len = 0u;      /* the publish walk skips the lap's tail */
                    hd->pos += ring - hd->pos % ring;
                }
                hd->lap = hd->pos, hd->run = 0u;
            }
        }
        memo_rec *s = put ? NULL : memo_way(memo_set(m, mb, hash), hash, key, n, (uint32_t)h->epoch, hd->pos, ring, 0);
        if (put) {
            memo_put(m, mb, g, n, key, hash, e->out + n0, e->n - n0, need);
        } else if (s != NULL) {                         /* a mark, never over another segment's live record */
            s->hash = hash, s->key = key, s->len = (uint32_t)n, s->n_ids = UINT32_MAX, s->epoch = (uint32_t)h->epoch;
            s->pos = vpos;
        }
    }
    return r;
}

/* the unit walk shared by encode (ids = 1) and pieces (ids = 0). */
static int64_t run(const toks_ctx *ctx, const void *text_v, uint64_t len, uint32_t flags,
                   uint32_t *out, uint64_t cap, void *scr, int ids)
{
    if (ctx == NULL) { return TOKS_E_ARG; }
    if ((flags & ~(TOKS_ADDED_MASK | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION | TOKS_NO_TRUNCATE | TOKS_NO_PAD)) != 0u ||
        (flags & TOKS_ADDED_MASK) == TOKS_ADDED_MASK) {
        return TOKS_E_ARG;
    }
    if ((text_v == NULL && len != 0u) || (out == NULL && cap != 0u)) { return TOKS_E_ARG; }
    if (len > TOKS_MAX_TEXT) { return TOKS_E_LIMIT; }
    toks_scratch *h = scratch_get(ctx, scr, len);
    if (h == NULL) { return TOKS_E_SCRATCH; }
    if (ctx->wp != NULL) { return toks_wp_run(ctx, h, (const uint8_t *)text_v, len, flags, out, cap, ids); }
    if (ctx->uni != NULL) {                                  /* uni_api.c (docs/algorithms/unigram.md) */
        return toks_uni_run(ctx, h, (const uint8_t *)text_v, len, flags, out, cap, ids);
    }

    /* TOKS_CONTINUATION: no cl100k-family pattern has start-of-input behaviour; sentencepiece-style bpe's
     * Metaspace prepend_scheme first does (the unit at input offset 0 gets the prefix, spm.h). */
    const uint8_t *text = (const uint8_t *)text_v;
    int cont = (flags & TOKS_CONTINUATION) != 0u;
    uint32_t mode = flags & TOKS_ADDED_MASK;
    int pp = ids != 0 && (flags & TOKS_NO_POSTPROCESS) == 0u;
    emit e = { out, cap, 0u, UINT64_MAX };
    uint64_t mb = 0u;
    uint8_t *memo = ids != 0 ? toks_scr_memo(h, &mb) : NULL;
    uint64_t pos0 = memo != NULL ? ((const toks_memo_head *)(const void *)memo)->pos : 0u;   /* at the call's start */

    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_prefix; i++) { emit1(&e, ctx->pp_ids[i]); }  /* bound: 64 */
    }
    if (ctx->cut_run != 0u) {                           /* a tiktoken wrapper (config.h cut_run) */
        if (mode == TOKS_ADDED_ALL && (ctx->t.add_phases & 1u) != 0u) {
            toks_seg_iter it;                           /* transformers' trie, then each gap cut */
            toks_seg_out u;
            toks_seg_begin(&it, &ctx->t, TOKS_ADDED_ALL, 0u, ctx->tier, text, len);
            while (toks_seg_next(&it, &u)) {            /* bound: <= 2 len + 1 units */
                if (u.kind == TOKS_SEG_TOKEN) {
                    emit1(&e, ids ? u.id : (uint32_t)u.end);
                } else {                                /* specials (mode ALL) */
                    (void)run_seg(ctx, h, text + u.start, u.end - u.start, u.start, flags, mode, &e, ids,
                                  !cont && u.start == 0u, 1);
                }
            }
        } else {                                        /* NONSPECIAL, NONE: the whole text cut */
            (void)run_seg(ctx, h, text, len, 0u, flags, mode, &e, ids, !cont, 1);
        }
    } else if (mode == TOKS_ADDED_NONE || ctx->t.add_n == 0u ||
               (mode == TOKS_ADDED_NONSPECIAL && ctx->n_nonspecial == 0u)) {
        int64_t r = run_seg(ctx, h, text, len, 0u, flags, TOKS_ADDED_NONE, &e, ids, !cont, 0);   /* no token can match */
        if (r < 0) { return r; }
    } else {
        /* (re, nb): the furthest unit end in the input, its normalized offset (docs/notes/c-core.md §api.c.5) */
        uint64_t nb = 0u, re = 0u;
        toks_seg_iter it;
        toks_seg_out u;
        toks_seg_begin(&it, &ctx->t, mode, 0u, ctx->tier, text, len);
        while (toks_seg_next(&it, &u)) {                /* bound: <= 2 len + 1 units */
            uint64_t s_nb = nb - ((re > u.start) ? re - u.start : 0u), e_nb;
            if (u.kind == TOKS_SEG_TOKEN) {             /* phase-0 tokens are not normalized (hf) */
                e_nb = s_nb + (u.end - u.start);
                emit1(&e, ids ? u.id : (uint32_t)e_nb);
            } else {
                int64_t r = run_seg(ctx, h, text + u.start, u.end - u.start, s_nb, flags, mode, &e, ids,
                                    !cont && u.start == 0u, 0);
                if (r < 0) { return r; }
                e_nb = s_nb + (uint64_t)r;
            }
            if (u.end >= re) { re = u.end; nb = e_nb; }
        }
    }
    /* hf encode(): the text's ids cut to max_length minus the template's (Right; config.c trunc_check), whole
     * documents, then padded; a call opts out of either (TOKS_NO_TRUNCATE, TOKS_NO_PAD: no_truncation, no_padding) */
    if (ids != 0 && !cont && ctx->o.trunc_on && (flags & TOKS_NO_TRUNCATE) == 0u) {
        uint64_t lim = (uint64_t)ctx->o.trunc_max - (pp ? ctx->n_pp_suffix : 0u);
        if (e.n > lim) { e.n = lim; }
    }
    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_suffix; i++) {          /* bound: 64 */
            emit1(&e, ctx->pp_ids[ctx->n_pp_prefix + i]);
        }
    }
    if (ids != 0 && !cont && ctx->o.pad_on && (flags & TOKS_NO_PAD) == 0u) { toks_pad(&ctx->o, &e); }
    if (memo != NULL && e.n <= cap) { memo_publish(memo, mb, (uint32_t)h->epoch, pos0); }
    return (int64_t)e.n;
}

int64_t toks_encode(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                    uint32_t *out, uint64_t cap, void *scr)
{
    return run(ctx, text, len, flags, out, cap, scr, 1);
}

int64_t toks_pieces(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                    uint32_t *ends, uint64_t cap, void *scr)
{
    return run(ctx, text, len, flags, ends, cap, scr, 0);
}

/* the template run() emits around a text (pp_ids: prefix then suffix; compile.c), its type ids beside it */
int64_t toks_template(const toks_ctx *ctx, uint32_t *ids, uint32_t *type_ids, uint64_t cap, uint32_t *n_prefix)
{
    if (ctx == NULL || (ids == NULL && cap != 0u)) { return TOKS_E_ARG; }
    uint32_t n = ctx->n_pp_prefix + ctx->n_pp_suffix;     /* <= 64 (config.c) */
    if (ids != NULL && cap < n) { return TOKS_E_CAP; }
    for (uint32_t i = 0u; ids != NULL && i < n; i++) {    /* bound: 64; ids, type_ids: a caller's arrays (core.h) */
        toks_st32(ids + i, ctx->pp_ids[i]);
        if (type_ids != NULL) { toks_st32(type_ids + i, ctx->pp_type[i]); }
    }
    if (n_prefix != NULL) { *n_prefix = ctx->n_pp_prefix; }
    return (int64_t)n;
}

/* ceil(r len) + g with the context's r = bound_num / bound_den and g (compile.c toks_bound_terms_of has the proof),
 * then padding's target (toks_pad: Fixed's length or the count, rounded up to the multiple); saturates
 * (docs/notes/c-core.md §api.c.7) */
uint64_t toks_encode_bound(const toks_ctx *ctx, uint64_t len)
{
    if (ctx == NULL) { return 0u; }
    uint64_t num = ctx->bound_num, den = ctx->bound_den, q = len / den;
    if (q > (UINT64_MAX - ctx->bound_g - num) / num) { return UINT64_MAX; }
    uint64_t b = num * q + (num * (len % den) + den - 1u) / den + ctx->bound_g;
    if (ctx->o.pad_on) {
        uint64_t t = ctx->o.pad_fixed ? ctx->o.pad_len : b, m = ctx->o.pad_multiple;
        if (m != 0u && t % m != 0u) { t = t > UINT64_MAX - m ? UINT64_MAX : t + (m - t % m); }
        b = t > b ? t : b;
    }
    return b;
}

/* ---- decode (SPEC §3.4): hf's ByteLevel decoder (docs/notes/c-core.md §api.c.6) --------------------- */

int64_t toks_decode(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags,
                    uint8_t *out, uint64_t cap)
{
    if (ctx == NULL || (ids == NULL && n != 0u) || (out == NULL && cap != 0u)) { return TOKS_E_ARG; }
    if ((flags & ~(TOKS_SKIP_SPECIAL | TOKS_DECODE_RAW)) != 0u) { return TOKS_E_ARG; }
    for (uint64_t i = 0u; i < n; i++) {                 /* bound: n (nothing is written on TOKS_E_ID) */
        if (toks_ld32(ids + i) >= ctx->t.n_ids) { return TOKS_E_ID; }
    }
    if (ctx->wp != NULL) { return toks_wp_decode(ctx, ids, n, flags, out, cap); }   /* wordpiece.md §8: always utf-8 */
    if (ctx->uni != NULL) { return toks_uni_dec(ctx, ids, n, flags, out, cap); }     /* unigram.md §8 */
    if (ctx->spm != NULL) {             /* sentencepiece-style bpe: its decoder chain (spm.h) */
        return toks_spm_decode(&ctx->t, ctx->spm, ctx->special_ids, flags, ids, n, out, cap);
    }
    if ((flags & TOKS_DECODE_RAW) != 0u) {              /* ByteLevel: the ids' bytes, the lossy pass skipped */
        const uint32_t *spec = (flags & TOKS_SKIP_SPECIAL) != 0u ? ctx->special_ids : NULL;
        toks_dsink d = { out, cap, 0u };
        for (uint64_t i = 0u; i < n; i++) {             /* bound: n */
            uint32_t id = toks_ld32(ids + i), o = ctx->t.tok_off[id];
            uint64_t k = (uint64_t)ctx->t.tok_off[id + 1u] - o;
            if (k != 0u && (spec == NULL || toks_bit(spec, id) == 0u)) { toks_dput(&d, ctx->t.tok_bytes + o, k); }
        }
        return (int64_t)d.n;
    }
    toks_lossy d;
    memset(&d, 0, sizeof d);
    d.out = out;
    d.cap = cap;
    (void)toks_lossy_ids(ctx, &d, ids, n, flags);       /* 0: every id was checked above */
    toks_lossy_end(&d);                                 /* an incomplete sequence at the end: one U+FFFD */
    return (int64_t)d.n;
}

const uint8_t *toks_token(const toks_ctx *ctx, uint32_t id, uint64_t *len)
{
    const uint8_t *p = NULL;
    uint64_t k = 0u;
    if (ctx != NULL && id < ctx->t.n_ids) {
        k = (uint64_t)(ctx->t.tok_off[id + 1u] - ctx->t.tok_off[id]);
        if (k != 0u) { p = ctx->t.tok_bytes + ctx->t.tok_off[id]; }
    }
    if (len != NULL) { *len = k; }
    return p;
}

/* ---- stream decode: stream.c ---------------------------------------------------------------------- */

/* ---- info ------------------------------------------------------------------------------------------ */

#define INFO_SIZE_03 offsetof(toks_info, trunc_on)       /* abi 0.3's toks_info: the fields before trunc_on */
_Static_assert(INFO_SIZE_03 == 184u, "abi 0.3's toks_info is 184 bytes");

int64_t toks_get_info(const toks_ctx *ctx, toks_info *o)
{
    if (ctx == NULL || o == NULL || (o->size != (uint32_t)sizeof(toks_info) && o->size != (uint32_t)INFO_SIZE_03)) {
        return TOKS_E_ARG;
    }
    toks_info i;                                        /* written whole, then o->size bytes of it: a 0.3 caller's
                                                           struct ends at trunc_on */
    memset(&i, 0, sizeof i);
    i.size = o->size;
    i.abi_major = TOKS_ABI_MAJOR;
    i.abi_minor = TOKS_ABI_MINOR;
    i.algorithm = ctx->t.algo;
    i.tier = ctx->tier;
    i.n_ids = ctx->t.n_ids;
    i.n_added = (uint32_t)ctx->t.add_n;
    i.paths = (ctx->gen != NULL ? 0u : TOKS_PATH_SCAN) | TOKS_PATH_NORMALIZE;   /* the generic engine is no fast path */
    i.cpu_features = ctx->cpu_features;                 /* read at load: no syscall here (SPEC §4.1) */
    i.max_text = TOKS_MAX_TEXT;
    i.control_isolation = 0u;                           /* not certified yet (SPEC §3.6) */
    memcpy(i.source_sha256, ctx->source_sha256, 32u);   /* image_sha256 stays 0: no .toks image */
    memcpy(i.name, ctx->name, sizeof i.name);
    i.name[sizeof i.name - 1u] = 0;
    i.trunc_on = ctx->o.trunc_on;                       /* config.c read_trunc_pad: hf's truncation and padding, */
    i.trunc_max = ctx->o.trunc_max;                     /* zero where the file has none */
    i.trunc_stride = ctx->o.trunc_stride;
    i.pad_on = ctx->o.pad_file;
    i.pad_fixed = ctx->o.pad_fixed;
    i.pad_id = ctx->o.pad_id;
    i.pad_type_id = ctx->o.pad_type_id;
    i.pad_len = ctx->o.pad_len;
    i.pad_multiple = ctx->o.pad_multiple;
    i.pad_left = ctx->o.pad_left;
    i.n_template_prefix = ctx->n_pp_prefix;
    i.n_template_suffix = ctx->n_pp_suffix;
    i.seq_type_id = ctx->pp_seq_type;
    memcpy(o, &i, (size_t)o->size);
    return 0;
}
