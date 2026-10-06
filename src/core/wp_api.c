/* wp_api.c: include/toks.h's encode (docs/notes/c-core.md §wp_api.c.1) */
#include "core.h"
#include "wp.h"

/* ---- the walk ----------------------------------------------------------------------------------------- */
typedef struct wpw {
    const toks_ctx *ctx;
    const toks_wp_tables *t;
    toks_wp_piece *pieces;    /* TOKS_CHUNK_PIECES */
    uint8_t *mat;
    uint64_t mat_cap;
    uint8_t *norm;            /* a normalized gap */
    uint64_t norm_cap;
    uint32_t *bounce;
    uint64_t bounce_n;
    uint8_t *cache;           /* the scratch's piece cache (greedy answers, wp.c) */
    uint64_t cache_mask;
    uint64_t tw;
    uint32_t mode;
    int ids;
} wpw;

/* one scan round over g[pos, n) (scan flags fl); *end = where the next round starts */
static uint64_t scan(const wpw *w, const uint8_t *g, uint64_t n, uint64_t pos, uint64_t fl, const toks_emit *e,
                     uint64_t *end)
{
    toks_wp_scan_args a;
    memset(&a, 0, sizeof a);
    uint64_t cap = TOKS_CHUNK_PIECES;
    if (e->lim != UINT64_MAX && e->lim - e->n + 2u < cap) { cap = e->lim - e->n + 2u; }   /* >= 3: a piece is >= 1 id */
    a.text = g;
    a.len = n;
    a.pos = pos;
    a.pieces = w->pieces;
    a.cap = cap;
    a.mat = w->mat;
    a.mat_cap = w->mat_cap;
    a.flags = fl;
    uint64_t np = toks_wp_scan_c(w->t, &a);
    *end = a.pos;
    return np;
}

/* the ids of one round's np pieces (bytes in g[0, gn) / the copy buffer); bound >= the ids they can give */
static void model(const wpw *w, const uint8_t *g, uint64_t gn, uint64_t np, toks_emit *e, uint64_t bound)
{
    uint64_t left = (e->n < e->cap) ? e->cap - e->n : 0u;
    int direct = left >= bound;
    toks_wp_encode_args k;
    memset(&k, 0, sizeof k);
    k.text = g;
    k.mat = w->mat;
    k.pieces = w->pieces;
    k.n = np;
    k.out = direct ? e->out + e->n : w->bounce;
    k.room = direct ? left : w->bounce_n;
    k.text_len = gn;
    k.mat_len = w->mat_cap;
    k.cache = w->cache;
    k.cache_mask = w->cache_mask;
    k.tw = w->tw;
    uint64_t m = toks_wp_encode_c(w->t, &k);
    if (!direct && left != 0u) { toks_cpy(e->out + e->n, w->bounce, (left < m ? left : m) * 4u); }   /* out: any alignment */
    e->n += m;
    if (e->n > e->lim) { e->n = e->lim; }          /* ids past the limit sit at indices the call may overwrite */
}

/* the in-place path over one gap g[0, n) */
static void text_inplace(const wpw *w, const uint8_t *g, uint64_t n, toks_emit *e)
{
    uint64_t pos = 0;
    while (pos < n && e->n < e->lim) {              /* bound: n (a round consumes >= 1 byte or writes a piece) */
        uint64_t end;
        uint64_t np = scan(w, g, n, pos, ~0ull, e, &end);
        if (np == 0u && end <= pos) { return; }     /* unreachable: mat_cap >= toks_wp_mat_min */
        if (np != 0u) { model(w, g, n, np, e, end - pos); } /* ids <= chars <= raw bytes (§12.5) */
        pos = end;
    }
}

/* the pre-tokenizer + model over normalized bytes g[0, n) at normalized-stream offset nb */
static void text_norm(const wpw *w, const uint8_t *g, uint64_t n, toks_emit *e, uint64_t nb)
{
    uint64_t pos = 0;
    while (pos < n && e->n < e->lim) {              /* bound: n */
        uint64_t end;
        uint64_t np = scan(w, g, n, pos, 0u, e, &end);
        if (np == 0u && end <= pos) { return; }     /* unreachable */
        if (w->ids == 0) {
            for (uint64_t k = 0; k < np; k++) {     /* bound: np; in place: off indexes g */
                toks_put(e, (uint32_t)(nb + w->pieces[k].off + w->pieces[k].len));
            }
        } else if (np != 0u) {
            model(w, g, n, np, e, end - pos);       /* ids <= chars <= normalized bytes */
        }
        pos = end;
    }
}

/* the normalized path over raw g[0, n): normalize, phase 1, pre-tokenizer + model */
static void gap_norm(const wpw *w, const uint8_t *g, uint64_t n, toks_emit *e, int match, uint64_t *nbase)
{
    int64_t r = toks_norm(w->t->flags, g, n, w->norm, w->norm_cap);
    uint64_t nl = (uint64_t)r;
    if (r < 0) { return; }                          /* unreachable: 3 bytes per byte (§12.5) */
    const toks_ctx *ctx = w->ctx;
    if (!match || (ctx->t.add_phases & 2u) == 0u) {
        text_norm(w, w->norm, nl, e, *nbase);
        *nbase += nl;
        return;
    }
    toks_seg_iter it;
    toks_seg_out v;
    toks_seg_begin(&it, &ctx->t, w->mode, 1u, ctx->tier, w->norm, nl);
    while (e->n < e->lim && toks_seg_next(&it, &v)) {      /* bound: <= 2 nl + 1 units */
        if (v.kind == TOKS_SEG_TOKEN) {
            toks_put(e, w->ids ? v.id : (uint32_t)(*nbase + v.end));
        } else {
            text_norm(w, w->norm + v.start, v.end - v.start, e, *nbase + v.start);
        }
    }
    *nbase += nl;
}

/* rationale: docs/notes/c-core.md §wp_api.c.2 */
static void gap(const wpw *w, const uint8_t *g, uint64_t n, toks_emit *e, int norm, int match, uint64_t *nbase)
{
    if (!norm) {
        text_inplace(w, g, n, e);
        return;
    }
    if (w->ids == 0 || w->ctx->o.wp_win == 0u) {
        gap_norm(w, g, n, e, match, nbase);
        return;
    }
    uint64_t r = 0;
    while (r < n && e->n < e->lim) {                /* bound: n (a window covers >= 1 byte) */
        uint64_t t = n;
        if (n - r > TOKS_WP_WIN) {
            t = r + TOKS_WP_WIN;
            while (t < n && g[t] != 0x20u && g[t] != 0x09u && g[t] != 0x0Au && g[t] != 0x0Du) { t++; }  /* bound: n */
        }
        gap_norm(w, g + r, t - r, e, match, nbase);
        r = t;
    }
}

int64_t toks_wp_run(const toks_ctx *ctx, toks_scratch *h, const uint8_t *text, uint64_t len, uint32_t flags,
                    uint32_t *out, uint64_t cap, int ids)
{
    uint8_t *s = (uint8_t *)(uintptr_t)h->base;
    uint8_t *wk = s + h->off_work;                  /* 64-aligned (core.h) */
    uint64_t ml = h->max_len;
    wpw w;
    w.ctx = ctx;
    w.t = ctx->wp;
    w.pieces = (toks_wp_piece *)(void *)wk;
    w.mat = wk + 16u * (uint64_t)TOKS_CHUNK_PIECES;
    w.mat_cap = ctx->wp_mat_cap;
    w.norm = w.mat + w.mat_cap;
    w.norm_cap = toks_align64(3u * ml + 64u);
    w.bounce = (uint32_t *)(void *)(s + h->off_bounce);
    w.bounce_n = ml + 4u;
    w.cache = ids != 0 ? s + h->off_cache : NULL;          /* greedy answers (wp.c), epoch-tagged (kernels.md §7) */
    w.cache_mask = TOKS_TEST_DEGEN(toks_scr_short(h->cache_mib) / TOKS_BUCKET - 1u);
    w.tw = toks_tag_word(h->epoch);
    w.mode = flags & TOKS_ADDED_MASK;
    w.ids = ids;

    int pp = ids != 0 && (flags & TOKS_NO_POSTPROCESS) == 0u;
    int doc = ids != 0 && (flags & TOKS_CONTINUATION) == 0u;   /* truncation, padding: whole-document steps */
    toks_emit e = { out, cap, 0u, UINT64_MAX };
    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_prefix; i++) { toks_put(&e, ctx->pp_ids[i]); }   /* bound: 64 */
    }
    if (doc && ctx->o.trunc_on && (flags & TOKS_NO_TRUNCATE) == 0u) {
        uint64_t n_added = pp ? (uint64_t)ctx->n_pp_prefix + ctx->n_pp_suffix : 0u;
        e.lim = e.n + (ctx->o.trunc_max - n_added);   /* config.c: trunc_max >= n_added */
    }

    uint32_t mode = w.mode;
    int match = !(mode == TOKS_ADDED_NONE || ctx->t.add_n == 0u ||
                  (mode == TOKS_ADDED_NONSPECIAL && ctx->n_nonspecial == 0u));
    int norm = ids == 0 || (match && (ctx->t.add_phases & 2u) != 0u);
    uint64_t nbase = 0u;                            /* pieces: the normalized stream's offset */
    if (!match) {
        gap(&w, text, len, &e, norm, 0, &nbase);
    } else {
        toks_seg_iter it;
        toks_seg_out u;
        toks_seg_begin(&it, &ctx->t, mode, 0u, ctx->tier, text, len);
        while (e.n < e.lim && toks_seg_next(&it, &u)) {    /* bound: <= 2 len + 1 units */
            if (u.kind == TOKS_SEG_TOKEN) {
                nbase += u.end - u.start;           /* a token's bytes are not normalized */
                toks_put(&e, ids ? u.id : (uint32_t)nbase);
                continue;
            }
            gap(&w, text + u.start, u.end - u.start, &e, norm, 1, &nbase);
        }
    }
    e.lim = UINT64_MAX;
    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_suffix; i++) { toks_put(&e, ctx->pp_ids[ctx->n_pp_prefix + i]); }  /* bound: 64 */
    }
    if (doc && ctx->o.pad_on && (flags & TOKS_NO_PAD) == 0u) { toks_pad(&ctx->o, &e); }
    return (int64_t)e.n;
}

/* ---- decode (§8) ------------------------------------------------------------------------------------- */
/* hf's cleanup, in its order: each str::replace runs over the whole token before the next one */
#define CL_N 11u
static const uint8_t CL_PAT[CL_N][8] = { " .", " ?", " !", " ,", " ' ", " n't", " 'm", " do not", " 's", " 've", " 're" };
static const uint8_t CL_PL[CL_N] = { 2, 2, 2, 2, 3, 4, 3, 7, 3, 4, 4 };
static const uint8_t CL_REP[CL_N][8] = { ".", "?", "!", ",", "'", "n't", "'m", " don't", "'s", "'ve", "'re" };
static const uint8_t CL_RL[CL_N] = { 1, 1, 1, 1, 1, 3, 2, 6, 2, 3, 3 };

typedef struct clst { uint8_t b[8]; uint32_t n; } clst;

/* one byte into replace pass q (leftmost-first, non-overlapping: str::replace): what the pass can no longer
 * change is appended to o */
static void cl_push(clst *s, uint32_t q, uint8_t c, uint8_t *o, uint32_t *no)
{
    s->b[s->n++] = c;
    for (uint32_t it = 0; it < 9u && s->n != 0u; it++) {      /* bound: 8 bytes held, one dropped per pass */
        if (s->n == CL_PL[q] && memcmp(s->b, CL_PAT[q], s->n) == 0) {
            memcpy(o + *no, CL_REP[q], CL_RL[q]);
            *no += CL_RL[q];
            s->n = 0;
            return;
        }
        if (memcmp(s->b, CL_PAT[q], s->n) == 0) { return; }   /* a proper prefix of the pattern: wait */
        o[(*no)++] = s->b[0];
        s->n--;
        for (uint32_t k = 0; k < s->n; k++) { s->b[k] = s->b[k + 1u]; }   /* bound: 7 */
    }
}

/* bytes src[0, n) (n <= 80) through passes q0 .. CL_N - 1, then out */
static void cl_feed(clst *st, uint32_t q0, const uint8_t *src, uint32_t n, toks_dsink *d)
{
    uint8_t a[160], b[160];                         /* a pass adds <= 7 bytes to its input: 80 + 77 < 160 */
    uint32_t na = n;
    memcpy(a, src, n);
    for (uint32_t q = q0; q < CL_N; q++) {          /* bound: 11 passes */
        uint32_t nb = 0;
        for (uint32_t i = 0; i < na; i++) { cl_push(&st[q], q, a[i], b, &nb); }   /* bound: na < 160 */
        memcpy(a, b, nb);
        na = nb;
    }
    toks_dput(d, a, na);
}

static void cleanup(toks_dsink *d, int sp, const uint8_t *p, uint64_t l)
{
    uint64_t spaces = 0;
    for (uint64_t i = 0; i < l; i++) { spaces += p[i] == ' '; }     /* bound: l */
    if (spaces == 0u) {
        if (!sp) { toks_dput(d, p, l); return; }
        /* one space, at 0: the first pattern in pass order that matches there replaces it, and no space is
           left for any later pass (" ' " and " do not" need a second space) */
        for (uint32_t q = 0; q < CL_N; q++) {       /* bound: 11 */
            uint32_t k = CL_PL[q] - 1u;
            if (q == 4u || q == 7u || l < k || memcmp(p, CL_PAT[q] + 1, k) != 0) { continue; }
            toks_dput(d, CL_REP[q], CL_RL[q]);
            toks_dput(d, p + k, l - k);
            return;
        }
        toks_dput(d, (const uint8_t *)" ", 1u);
        toks_dput(d, p, l);
        return;
    }
    clst st[CL_N];
    memset(st, 0, sizeof st);
    if (sp) { cl_feed(st, 0u, (const uint8_t *)" ", 1u, d); }
    for (uint64_t i = 0; i < l; i++) { cl_feed(st, 0u, p + i, 1u, d); }    /* bound: l */
    for (uint32_t q = 0; q < CL_N; q++) {           /* bound: 11: each pass's held bytes, through the rest */
        uint8_t tmp[8];
        uint32_t m = st[q].n;
        memcpy(tmp, st[q].b, m);
        st[q].n = 0;
        if (q + 1u < CL_N) { cl_feed(st, q + 1u, tmp, m, d); } else { toks_dput(d, tmp, m); }
    }
}

int64_t toks_wp_decode(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *out,
                       uint64_t cap)
{
    uint64_t k = 0u;
    return toks_wp_decode_k(ctx, ids, n, flags, out, cap, &k);
}

/* rationale: docs/notes/c-core.md §wp_api.c.3 */
int64_t toks_wp_decode_k(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *out,
                         uint64_t cap, uint64_t *kept)
{
    toks_dsink d = { out, cap, 0u };
    const uint32_t *off = ctx->t.tok_off;
    uint64_t k = *kept;                             /* tokens kept so far (hf's index i) */
    for (uint64_t i = 0u; i < n; i++) {             /* bound: n */
        uint32_t id = toks_ld32(ids + i);           /* the caller's ids need no alignment (toks.h) */
        if ((flags & TOKS_SKIP_SPECIAL) != 0u && ctx->special_ids != NULL &&
            toks_bit(ctx->special_ids, id) != 0u) {
            continue;                               /* compile.c: hf's rule, by the token's string */
        }
        const uint8_t *p = ctx->t.tok_bytes + off[id];
        uint64_t l = (uint64_t)(off[id + 1u] - off[id]);
        if (ctx->o.dec_wordpiece == 0u) {             /* no decoder: hf joins with " " */
            if (k != 0u) { toks_dput(&d, (const uint8_t *)" ", 1u); }
            toks_dput(&d, p, l);
            k++;
            continue;
        }
        int sp = 0;
        if (k != 0u) {
            uint64_t pl = ctx->o.dec_prefix_len;
            if (l >= pl && memcmp(p, ctx->o.dec_prefix, (size_t)pl) == 0) { p += pl; l -= pl; } else { sp = 1; }
        }
        k++;
        if (ctx->o.dec_cleanup == 0u) {
            if (sp) { toks_dput(&d, (const uint8_t *)" ", 1u); }
            toks_dput(&d, p, l);
            continue;
        }
        cleanup(&d, sp, p, l);
    }
    *kept = k;
    return (int64_t)d.n;
}
