/* uni_api.c: the Unigram family behind include/toks.h (docs/notes/c-core.md §uni_api.c.1) */
#include "kernels.h"
#include "norm.h"
#include "unigram.h"

/* hf's steps before Strip and the charsmap (config.c uni_norm_one): the Replace(String)s in order (in place in r: no
 * longer), then the engine's form into p (<= pre_x n bytes); the text the rest reads (t: no steps), its length in *n */
static const uint8_t *uni_pre(const toks_uni_cfg *c, const uint8_t *t, uint64_t *n, uint8_t *r, uint8_t *p)
{
    const uint8_t *s = t;
    uint64_t m = *n;
    for (uint32_t k = 0; k < c->rep_n; k++) {                /* bound: 4 */
        uint64_t o = 0u, pl = c->rep_pl[k];
        for (uint64_t i = 0u; i < m;) {                      /* bound: m; o <= i */
            if (s[i] == c->rep_p[k][0] && m - i >= pl && memcmp(s + i, c->rep_p[k], (size_t)pl) == 0) {
                memcpy(r + o, c->rep_c[k], c->rep_cl[k]);
                o += c->rep_cl[k];
                i += pl;
            } else {
                r[o++] = s[i++];
            }
        }
        s = r;
        m = o;
    }
    if (c->form != 0u) {
        m = (uint64_t)toks_norm(c->form, s, m, p, TOKS_NORM_BOUND(c->form, m));   /* cap = the bound: never fails */
        s = p;
    }
    *n = m;
    return s;
}

int64_t toks_uni_load(struct toks_ctx *c, const struct toks_config *cfg, const char **why)
{
    const toks_uni *u = NULL;
    uint8_t *mem = NULL;
    uint64_t ml = 0u;
    *why = "Unigram tables";
    int64_t r = toks_uni_build(cfg->uni, &u, &mem, &ml, why);
    if (r != 0) { return r; }
    c->uni = u;
    c->dc = (toks_dchain){ u->dec, u->n_dec, u->cfg.dec != TOKS_UNI_DEC_NONE, u->cfg.dec == TOKS_UNI_DEC_BFRF, 1u, NULL };
    c->mem_uni = mem;
    c->mem_uni_len = ml;
    /* rationale: docs/notes/c-core.md §uni_api.c.2 */
    for (uint32_t i = 0u; i < cfg->n_added; i++) {           /* bound: n_added */
        const toks_cfg_added *a = &cfg->added[i];
        if (!a->normalized) { continue; }
        uint32_t sp = u->cfg.has_charsmap ? toks_pc_char(&u->pc, 0x2581u) : 0u;
        if (u->cfg.rep_n != 0u || u->cfg.form != 0u) {
            *why = "normalized added tokens under Replace / NF* / StripAccents / Lowercase steps (Unigram)";
            return TOKS_E_UNSUPPORTED;
        }
        if (!u->cfg.has_charsmap || u->cfg.meta_prefix || u->cfg.meta_replace || sp == 0u ||
            TOKS_PC_LEN(sp) != 1u || u->pc.pool[TOKS_PC_OFF(sp)] != 0x20u) {
            *why = "normalized added tokens with a chain that keeps U+2581 (Unigram)";
            return TOKS_E_UNSUPPORTED;
        }
        uint8_t buf[4096];
        int64_t m = toks_uni_normalize(u, a->content, a->len, 1, buf, sizeof buf);
        if (m != (int64_t)a->len || memcmp(buf, a->content, a->len) != 0) {
            *why = "a normalized added token that the normalizer changes (Unigram)";
            return TOKS_E_UNSUPPORTED;
        }
    }
    return 0;
}

/* rationale: docs/notes/c-core.md §uni_api.c.3 */
int64_t toks_uni_run(const struct toks_ctx *ctx, const toks_scratch *h, const uint8_t *text, uint64_t len,
                     uint32_t flags, uint32_t *out, uint64_t cap, int ids)
{
    const toks_uni *u = ctx->uni;
    uint8_t *work = toks_scr_at(h, h->off_work);
    uint64_t wbytes = (h->off_bounce - h->off_work) + toks_scr_bounce(toks_scr_tmax(h->max_len, ctx->nfc != 0u ?
                      TOKS_NORM_X(ctx->nfc) : 0u));
    uint64_t one = toks_uni_area(u, len);
    if (3u * one > wbytes) { return TOKS_E_SCRATCH; }        /* unreachable (above) */
    uint32_t mode = flags & TOKS_ADDED_MASK;
    int pp = ids != 0 && (flags & TOKS_NO_POSTPROCESS) == 0u;
    int cont = (flags & TOKS_CONTINUATION) != 0u;
    toks_uni_call c;
    memset(&c, 0, sizeof c);
    c.e = (toks_emit){ out, cap, 0u, UINT64_MAX };
    c.pieces = (ids == 0);
    c.work = work;
    c.work_bytes = 2u * one;
    if (ids != 0) {                                          /* the piece cache (unigram.c uni_cached) */
        c.cache = toks_scr_at(h, h->off_cache);
        c.cache_mask = TOKS_TEST_DEGEN(toks_scr_short(h->cache_mib) / TOKS_BUCKET - 1u);
        c.tw = toks_tag_word(h->epoch);
    }
    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_prefix; i++) { toks_put(&c.e, ctx->pp_ids[i]); }   /* bound: 64 */
    }
    /* hf encode(): the model's ids cut to max_length minus the template's ids (add_special_tokens), Right;
     * the room past the cut is never written */
    if (ids != 0 && !cont && ctx->o.trunc_on != 0u) {          /* one document: a TOKS_CONTINUATION part keeps all */
        c.e.lim = c.e.n + (uint64_t)ctx->o.trunc_max - (pp ? (uint64_t)(ctx->n_pp_prefix + ctx->n_pp_suffix) : 0u);
    }
    int64_t r = 0;
    /* the steps before the charsmap (uni_pre): r (scratch) and their output p past the two areas the segment uses */
    uint8_t *pb = work + 2u * one, *rb = u->cfg.form != 0u ? work + one : pb;
    if (mode == TOKS_ADDED_NONE || ctx->t.add_n == 0u || (mode == TOKS_ADDED_NONSPECIAL && ctx->n_nonspecial == 0u)) {
        uint64_t n = len;
        const uint8_t *x = uni_pre(&u->cfg, text, &n, rb, pb);
        r = toks_uni_encode_segment(u, x, n, 0, !cont, &c);
    } else {
        int phase1 = (ctx->t.add_phases & 2u) != 0u;
        uint64_t nb = 0u;                                    /* the normalized stream's offset of the next unit */
        toks_seg_iter it;
        toks_seg_out g;
        toks_seg_begin(&it, &ctx->t, mode, 0u, ctx->tier, text, len);
        while (r >= 0 && c.e.n < c.e.lim && toks_seg_next(&it, &g)) {   /* bound: <= 2 len + 1 units */
            if (g.kind == TOKS_SEG_TOKEN) {                  /* phase-0 tokens are not normalized (hf) */
                nb += g.end - g.start;
                toks_put(&c.e, ids ? g.id : (uint32_t)nb);
                continue;
            }
            uint64_t gl = g.end - g.start;
            const uint8_t *gt = uni_pre(&u->cfg, text + g.start, &gl, rb, pb);
            int gap_start = !(cont && g.start == 0u);
            if (!phase1) {
                c.nbase = nb;
                c.work = work;
                c.work_bytes = 2u * one;
                r = toks_uni_encode_segment(u, gt, gl, 0, gap_start, &c);
                if (r >= 0) { nb += (uint64_t)r; }
                continue;
            }
            /* phase 1: the gap's normalized form in the first area, its tokens, the rest read there in place */
            int64_t m = toks_uni_normalize(u, gt, gl, gap_start, work, one);
            if (m < 0) { r = TOKS_E_SCRATCH; break; }        /* unreachable: one >= work_x gl */
            toks_seg_iter it1;
            toks_seg_out v;
            toks_seg_begin(&it1, &ctx->t, mode, 1u, ctx->tier, work, (uint64_t)m);
            while (r >= 0 && c.e.n < c.e.lim && toks_seg_next(&it1, &v)) {   /* bound: <= 2 m + 1 units */
                if (v.kind == TOKS_SEG_TOKEN) { toks_put(&c.e, ids ? v.id : (uint32_t)(nb + v.end)); continue; }
                c.nbase = nb + v.start;
                c.work = work + one;
                c.work_bytes = 2u * one;
                r = toks_uni_encode_segment(u, work + v.start, v.end - v.start, 1, 1, &c);
            }
            nb += (uint64_t)m;
        }
    }
    if (r < 0) { return r; }
    c.e.lim = UINT64_MAX;
    if (pp) {
        for (uint32_t i = 0u; i < ctx->n_pp_suffix; i++) { toks_put(&c.e, ctx->pp_ids[ctx->n_pp_prefix + i]); }   /* bound: 64 */
    }
    return (int64_t)c.e.n;
}
