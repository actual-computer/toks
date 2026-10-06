/* split.c: exact split planning (docs/notes/c-core.md §split.c.1; every family's rules and proofs: docs/split.md) */
#include "kernels.h"
#include "norm.h"
#include "spm.h"
#include "split.h"
#include "unigram.h"
#include "wp.h"

#define TOKS_SPLIT_D 4096u

/* the atom holding byte i of x[0, len) (kernels.md §2): its start, *k its length. A lead or ascii byte is
 * always an atom start, so the atom is decided by at most 3 bytes back and the sequence's own bytes. */
static uint64_t atom_holding(const uint8_t *x, uint64_t len, uint64_t i, uint32_t *k)
{
    uint64_t s = i;
    for (uint32_t back = 0u; back < 3u && s > 0u && (x[s] & 0xC0u) == 0x80u; back++) { s--; }   /* bound: 3 */
    uint32_t n = toks_utf8_len(x + s, len - s);
    if (n != 0u && s + n > i) {
        *k = n;
        return s;
    }
    *k = 1u;                                              /* ascii, or an invalid byte: one atom */
    return i;
}

/* base class of the atom p[0, k) (an invalid atom is P); *cp its code point (the byte when invalid). */
static uint8_t atom_base(const toks_tables *t, const uint8_t *p, uint32_t k, uint32_t *cp)
{
    if (k == 1u) {
        *cp = p[0];
        return (p[0] < 0x80u) ? (uint8_t)(t->cls_ascii[p[0]] & TOKS_C_BASE_MASK) : (uint8_t)TOKS_C_P;
    }
    *cp = toks_cp_decode(p, k);
    return (uint8_t)(toks_cls_cp(t, *cp) & TOKS_C_BASE_MASK);
}

/* the scan conjunct at c (0 < c < len): 1 when certified; [*lo, *hi) = the atoms the rule read. */
static int scan_ok(const toks_tables *t, const uint8_t *x, uint64_t len, uint64_t c, uint64_t *lo, uint64_t *hi)
{
    uint32_t ka, kb, kz, cpa, cpb, cpz;
    uint64_t sa = atom_holding(x, len, c - 1u, &ka);
    if (sa + ka != c) { return 0; }                       /* c is inside an atom */
    kb = toks_utf8_len(x + c, len - c);
    if (kb == 0u) { kb = 1u; }
    uint8_t A = atom_base(t, x + sa, ka, &cpa);
    uint8_t B = atom_base(t, x + c, kb, &cpb);
    *lo = sa;
    *hi = c + kb;

    uint32_t pr = t->tmpl_params;
    int lpany = (pr & TOKS_TP_LPREFIX_ANY) != 0u;
    uint32_t digits = pr & TOKS_TP_DIGITS_MASK;
    int ws_nl = (pr & TOKS_TP_WS_NL) != 0u;
    int sp = (cpa == 0x20u);                              /* only U+0020 is the ' ?' of A2 / A3 / A4 */

    if (A == TOKS_C_L) { return B != TOKS_C_L; }                                         /* R1 */
    if (A == TOKS_C_N) { return B != TOKS_C_N || digits == TOKS_TP_DIGITS_1; }           /* R2 */
    if (A == TOKS_C_P) {                                                                 /* R3 */
        if (B == TOKS_C_P) { return 0; }
        if (B == TOKS_C_NL) { return (pr & TOKS_TP_PUNCT_NL) == 0u; }
        if (B != TOKS_C_L) { return 1; }                  /* N or WS */
        int apos = (pr & TOKS_TP_CONTR_MASK) != TOKS_TP_CONTR_NONE && cpa == 0x27u;
        if (!lpany && !apos) { return 1; }
        if (sa == 0u) { return 0; }
        uint64_t sz = atom_holding(x, len, sa - 1u, &kz);
        *lo = sz;
        return atom_base(t, x + sz, kz, &cpz) == TOKS_C_P;   /* a continues z's P run, so starts no piece */
    }
    if (B == TOKS_C_WS || B == TOKS_C_NL) { return 0; }   /* a and b are one whitespace run */
    if (A == TOKS_C_NL && ws_nl) { return 1; }                                           /* R4 */
    /* R5: a WS, or NL without WS_NL */
    if (B == TOKS_C_L && (lpany || sp)) { return 0; }
    if (sp && (B == TOKS_C_P || (B == TOKS_C_N && digits == TOKS_TP_DIGITS_SP_RUN))) { return 0; }
    if (sa == 0u) { return 1; }
    uint64_t sz = atom_holding(x, len, sa - 1u, &kz);
    *lo = sz;
    uint8_t Z = atom_base(t, x + sz, kz, &cpz);
    return Z != TOKS_C_WS && Z != TOKS_C_NL;
}

/* a whitespace atom a ending at c (atoms read: z before it): a one-atom run (docs/split.md O5 / D5) */
static int lone_ws(const toks_tables *t, const uint8_t *x, uint64_t len, uint64_t sa, uint64_t *lo, int no_x)
{
    if (sa == 0u) { return 1; }
    uint32_t kz, cpz;
    uint64_t sz = atom_holding(x, len, sa - 1u, &kz);
    *lo = sz;
    uint8_t Z = toks_k3_atom(t, x + sz, len - sz, &cpz, &kz, 0u) & TOKS_C_BASE_MASK;
    return Z != TOKS_C_WS && Z != TOKS_C_NL && !(no_x && Z == TOKS_C_X);
}

/* the o200k scan conjunct (docs/split.md §3: rules O1-O5 over a, the atom ending at c, b at c, z before a) */
static int scan_o200k(const toks_tables *t, const uint8_t *x, uint64_t len, uint64_t c, uint64_t *lo, uint64_t *hi)
{
    uint32_t ka, kb, cpa, cpb;
    uint64_t sa = atom_holding(x, len, c - 1u, &ka);
    if (sa + ka != c) { return 0; }
    uint8_t ca = toks_k3_atom(t, x + sa, len - sa, &cpa, &ka, 0u), cb = toks_k3_atom(t, x + c, len - c, &cpb, &kb, 0u);
    uint8_t A = ca & TOKS_C_BASE_MASK, B = cb & TOKS_C_BASE_MASK, cs = TOKS_C_UPPER | TOKS_C_LOWER;
    *lo = sa;
    *hi = c + kb;
    if (A == TOKS_C_L) { return (cb & cs) == 0u && cpb != 0x27u; }                       /* O1 */
    if (A == TOKS_C_N) {                                                                 /* O2 */
        return B != TOKS_C_N || (t->tmpl_params & TOKS_TP_DIGITS_MASK) == TOKS_TP_DIGITS_1;
    }
    if (A == TOKS_C_P) { return (ca & cs) == 0u && (B == TOKS_C_N || B == TOKS_C_WS); }   /* O3 */
    uint32_t tail = (t->tmpl_params & TOKS_TP_NO_SLASH) != 0u ? 0x0Du : 0x2Fu;
    if (A == TOKS_C_NL) { return B != TOKS_C_WS && B != TOKS_C_NL && cpb != tail; }       /* O4 */
    if (A != TOKS_C_WS || B == TOKS_C_WS || B == TOKS_C_NL || (cb & cs) != 0u) { return 0; }   /* O5 */
    if (cpa == 0x20u && B == TOKS_C_P) { return 0; }
    return lone_ws(t, x, len, sa, lo, 0);
}

/* the deepseek v3 conjunct (docs/split.md §4: D1-D5; no cut beside an X atom, an unmatched gap) */
static int scan_dsv3(const toks_tables *t, const uint8_t *x, uint64_t len, uint64_t c, uint64_t *lo, uint64_t *hi)
{
    uint32_t ka, kb, cpa, cpb;
    uint64_t sa = atom_holding(x, len, c - 1u, &ka);
    if (sa + ka != c) { return 0; }
    uint8_t A = toks_k3_atom(t, x + sa, len - sa, &cpa, &ka, 0u) & TOKS_C_BASE_MASK;
    uint8_t B = toks_k3_atom(t, x + c, len - c, &cpb, &kb, 0u) & TOKS_C_BASE_MASK;
    *lo = sa;
    *hi = c + kb;
    if (A == TOKS_C_X || B == TOKS_C_X) { return 0; }
    if (A == TOKS_C_L) { return B != TOKS_C_L; }                                         /* D1 */
    if (A == TOKS_C_N) { return B != TOKS_C_N; }                                         /* D2 */
    if (A == TOKS_C_P) { return B == TOKS_C_N || B == TOKS_C_WS; }                       /* D3 */
    if (A == TOKS_C_NL) { return B != TOKS_C_WS && B != TOKS_C_NL; }                     /* D4 */
    if (B == TOKS_C_WS || B == TOKS_C_NL || B == TOKS_C_L || (cpa == 0x20u && B == TOKS_C_P)) { return 0; }   /* D5 */
    return lone_ws(t, x, len, sa, lo, 1);
}

/* wordpiece (docs/split.md §6): an ascii whitespace byte before c, an ascii word or punctuation byte at c */
static int cut_wp(const toks_wp_tables *w, const uint8_t *x, uint64_t c, uint64_t *lo, uint64_t *hi)
{
    uint8_t p = x[c - 1u], q = x[c];
    if (p >= 0x80u || q >= 0x80u || w->ascii_cls[p] != TOKS_WPA_SPLIT) { return 0; }
    *lo = c - 1u;
    *hi = c + 1u;
    return w->ascii_cls[q] == TOKS_WPA_WORD || w->ascii_cls[q] == TOKS_WPA_FOLD || w->ascii_cls[q] == TOKS_WPA_PUNCT;
}

/* unigram (docs/split.md §7): c at a lone U+0020 between two simple ascii chars (unigram.h simple[]) */
static int cut_uni(const toks_uni *u, const uint8_t *x, uint64_t len, uint64_t c, uint64_t *lo, uint64_t *hi)
{
    if (c + 1u >= len || x[c] != 0x20u || x[c - 1u] >= 0x80u || x[c + 1u] >= 0x80u) { return 0; }
    *lo = c - 1u;
    *hi = c + 2u;
    return u->simple[x[c - 1u]] == 1u && u->simple[x[c + 1u]] == 1u;
}

/* the added-token conjunct: 1 when no occurrence of any added token (both phases, every token: a special
 * dropped under NONSPECIAL still moves hf's cursor) overlaps [lo, hi). maxlen = the longest token. */
static int tokens_clear(const toks_tables *t, const uint8_t *x, uint64_t len, uint64_t lo, uint64_t hi,
                        uint64_t maxlen)
{
    uint64_t s = (lo + 1u > maxlen) ? lo + 1u - maxlen : 0u;  /* a token starting before s ends by lo */
    for (; s < hi; s++) {                                 /* bound: maxlen - 1 + (hi - lo) */
        uint8_t b0 = x[s];
        for (uint64_t ph = 0u; ph < 2u; ph++) {           /* bound: 2 phases */
            if ((t->add_phases & (1ull << ph)) == 0u) { continue; }
            const uint8_t *sh = t->add_shufti + ph * 32u;
            if ((sh[b0 & 0x0Fu] & sh[16u + (b0 >> 4)]) == 0u) { continue; }
            if (s >= lo && t->add_single[ph * 256u + b0] != 0u) { return 0; }
            if (s + 1u >= len) { continue; }
            uint64_t ent = t->add_index[ph * 65536u + ((uint64_t)b0 | ((uint64_t)x[s + 1u] << 8))];
            const uint32_t *cand = t->add_cand + (ent & 0xFFFFFFFFull);
            for (uint64_t j = 0u; j < (ent >> 32); j++) { /* bound: the bucket, longest first */
                const toks_added_entry *e = &t->add_entries[cand[j]];
                if (s + e->len <= lo) { break; }          /* this one and every later one end by lo */
                if (s + e->len > len) { continue; }
                if (memcmp(t->add_bytes + e->off, x + s, (size_t)e->len) == 0) { return 0; }
            }
        }
    }
    return 1;
}

/* the NFC conjunct: every atom of [lo, hi) and the atom at hi is a boundary atom (norm.h). */
static int nfc_clear(uint32_t f, const uint8_t *x, uint64_t len, uint64_t lo, uint64_t hi)
{
    for (uint64_t i = lo; i <= hi && i < len;) {          /* bound: <= 4 atoms */
        /* an ascii atom is a boundary (no composition has an ascii second char); toks_nfc_boundary decodes
         * 2..4-byte atoms only (norm.c, reported in the team chat) */
        if (x[i] >= 0x80u && !toks_nfc_boundary(f, x, len, i)) { return 0; }
        uint32_t k = toks_utf8_len(x + i, len - i);
        i += (k != 0u) ? k : 1u;
    }
    return 1;
}

/* the unigram chains with certified cuts (docs/split.md §7): a pre-tokenizer that cuts at U+0020 (WhitespaceSplit,
 * or Metaspace with split), no inserted prefix / U+0020 -> U+2581 Replace, a charsmap mapping U+0020 to itself, no
 * Replace(String) holding U+0020 */
static int uni_rules(const toks_uni *u)
{
    const toks_uni_cfg *g = &u->cfg;
    if (g->meta_prefix || g->meta_replace || !(g->ws_split || (g->metaspace && g->meta_split))) { return 0; }
    if (g->has_charsmap && u->aent[0x20] != 0u) { return 0; }
    for (uint32_t k = 0u; k < g->rep_n; k++) {            /* bound: 4 */
        for (uint32_t i = 0u; i < 4u; i++) {              /* bound: 4 */
            if ((i < g->rep_pl[k] && g->rep_p[k][i] == 0x20u) || (i < g->rep_cl[k] && g->rep_c[k][i] == 0x20u)) { return 0; }
        }
    }
    return 1;
}

int toks_cuts_of(const toks_ctx *ctx, uint32_t flags, toks_cuts *k)
{
    memset(k, 0, sizeof *k);
    if ((ctx->o.trunc_on && (flags & TOKS_NO_TRUNCATE) == 0u) || (ctx->o.pad_on && (flags & TOKS_NO_PAD) == 0u) ||
        ctx->cut_run != 0u || ctx->gen != NULL) {        /* whole-document steps the call applies; kimi's cuts */
        return 0;
    }
    const toks_tables *t = &ctx->t;
    uint32_t mode = flags & TOKS_ADDED_MASK, fam = TOKS_CUT_NONE;
    int tokens = mode != TOKS_ADDED_NONE && t->add_n != 0u && !(mode == TOKS_ADDED_NONSPECIAL && ctx->n_nonspecial == 0u);
    int phase1 = tokens && (t->add_phases & 2u) != 0u;   /* tokens matched on normalized text */
    const toks_spm *sp = ctx->spm;
    if (ctx->wp != NULL) {
        fam = phase1 ? TOKS_CUT_NONE : TOKS_CUT_WP;
    } else if (ctx->uni != NULL) {
        fam = phase1 || !uni_rules(ctx->uni) ? TOKS_CUT_NONE : TOKS_CUT_UNI;
    } else if (sp != NULL) {
        fam = sp->pfx_mode == TOKS_SPM_PFX_GAP || (sp->pairs == NULL && !sp->ms_split) ? TOKS_CUT_NONE : TOKS_CUT_SPM;
    } else if (t->tmpl == TOKS_TMPL_CL100K) {             /* no rules yet for the cut-first parameters (P20 / P21 / P25) */
        fam = (t->tmpl_params & (TOKS_TP_DIGIT_CUT | TOKS_TP_NL_CUT | TOKS_TP_GB_SP)) != 0u ? TOKS_CUT_NONE : TOKS_CUT_CL100K;
    } else if (t->tmpl == TOKS_TMPL_O200K) {
        fam = (t->tmpl_params & TOKS_TP_HAN) != 0u ? TOKS_CUT_NONE : TOKS_CUT_O200K;
    } else if (t->tmpl == TOKS_TMPL_DSV3) {
        fam = TOKS_CUT_DSV3;
    }
    if (fam == TOKS_CUT_NONE || (phase1 && ctx->nfc != 0u)) { return 0; }   /* phase 1 matches NFC'd text */
    if (tokens) {
        for (uint64_t i = 0u; i < t->add_n; i++) {        /* bound: add_n */
            if (t->add_entries[i].len > k->maxlen) { k->maxlen = t->add_entries[i].len; }
        }
    }
    k->family = fam;
    k->win = k->maxlen + 16u;                             /* atoms of <= 4 bytes: z a | b + the token reach */
    return 1;
}

int toks_cut_ok(const toks_ctx *ctx, const toks_cuts *k, const uint8_t *x, uint64_t len, uint64_t c)
{
    uint64_t lo = c, hi = c;
    int ok;
    switch (k->family) {
    case TOKS_CUT_SPM: {
        uint32_t ka;
        lo = atom_holding(x, len, c - 1u, &ka);
        ok = lo + ka == c && toks_spm_cut(ctx->spm, x, len, lo, c);
        uint32_t kb = toks_utf8_len(x + c, len - c);
        hi = c + (kb != 0u ? kb : 1u);
        break;
    }
    case TOKS_CUT_WP: ok = cut_wp(ctx->wp, x, c, &lo, &hi); break;
    case TOKS_CUT_UNI: ok = cut_uni(ctx->uni, x, len, c, &lo, &hi); break;
    case TOKS_CUT_O200K: ok = scan_o200k(&ctx->t, x, len, c, &lo, &hi); break;
    case TOKS_CUT_DSV3: ok = scan_dsv3(&ctx->t, x, len, c, &lo, &hi); break;
    case TOKS_CUT_CL100K: ok = scan_ok(&ctx->t, x, len, c, &lo, &hi); break;
    default: return 0;
    }
    if (!ok || (ctx->nfc != 0u && !nfc_clear(ctx->nfc, x, len, lo, hi))) { return 0; }
    return k->maxlen == 0u || tokens_clear(&ctx->t, x, len, lo, hi, k->maxlen);
}

int64_t toks_split_points(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                          uint32_t n_want, uint64_t *offs, uint64_t cap, void *scr)
{
    (void)scr;                                            /* reserved: no family needs scratch to plan */
    if (ctx == NULL) { return TOKS_E_ARG; }
    if ((flags & ~(TOKS_ADDED_MASK | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION | TOKS_NO_TRUNCATE | TOKS_NO_PAD)) != 0u ||
        (flags & TOKS_ADDED_MASK) == TOKS_ADDED_MASK) {
        return TOKS_E_ARG;
    }
    if ((text == NULL && len != 0u) || (offs == NULL && cap != 0u)) { return TOKS_E_ARG; }
    if (len > TOKS_MAX_TEXT) { return TOKS_E_LIMIT; }
    toks_cuts k;
    if (n_want <= 1u || len < 2u || cap == 0u || !toks_cuts_of(ctx, flags, &k)) { return 0; }   /* no rule: none */

    const uint8_t *x = (const uint8_t *)text;
    /* more targets than bytes hit every position, as n_want = len does (D = 0 either way) */
    uint64_t nw = ((uint64_t)n_want > len) ? len : (uint64_t)n_want;
    uint64_t d_max = len / (4u * nw);
    if (d_max > TOKS_SPLIT_D) { d_max = TOKS_SPLIT_D; }

    uint64_t c = 0u, prev = 0u;
    for (uint64_t i = 1u; i < nw && c < cap; i++) {      /* bound: nw - 1 <= len targets */
        uint64_t tg = len * i / nw;                       /* in [1, len): nw <= len; len * i < 2^58 */
        uint64_t best = 0u;
        for (uint64_t d = 0u; d <= d_max && best == 0u; d++) {   /* bound: d_max + 1 rings */
            for (uint32_t side = 0u; side < 2u; side++) { /* bound: 2: below first (ties go lower) */
                if (side == 1u && d == 0u) { break; }
                if (side == 0u ? d >= tg : tg + d >= len) { continue; }
                uint64_t p = side == 0u ? tg - d : tg + d;
                if (toks_cut_ok(ctx, &k, x, len, p)) {
                    best = p;
                    break;
                }
            }
        }
        if (best > prev) {                                /* 0 = none; == prev: the previous target's cut */
            toks_st64((uint8_t *)(void *)offs + 8u * c, best);   /* the caller's offs may be misaligned (toks.h) */
            c++;
            prev = best;
        }
    }
    return (int64_t)c;
}

void toks_pp_ids(const toks_ctx *ctx, uint32_t flags, const uint32_t **pre, uint32_t *n_pre,
                 const uint32_t **suf, uint32_t *n_suf)
{
    int on = (flags & TOKS_NO_POSTPROCESS) == 0u;
    *pre = ctx->pp_ids;
    *n_pre = on ? ctx->n_pp_prefix : 0u;
    *suf = ctx->pp_ids + ctx->n_pp_prefix;
    *n_suf = on ? ctx->n_pp_suffix : 0u;
}
