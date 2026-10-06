/* compile.c: toks_config -> toks_ctx (docs/notes/c-core.md §compile.c.1) */
#include "core.h"
#include "classes.h"
#include "compile.h"
#include "norm.h"
#include "spm.h"
#include "unigram.h"
#include "wp.h"

static uint64_t up64(uint64_t v) { return (v + 63u) & ~(uint64_t)63u; }

/* rationale: docs/notes/c-core.md §compile.c.2 */
static uint8_t ascii_class(uint32_t b)
{
    if (b - 0x41u < 26u) { return (uint8_t)(TOKS_C_L | TOKS_C_UPPER); }
    if (b - 0x61u < 26u) { return (uint8_t)(TOKS_C_L | TOKS_C_LOWER); }
    if (b - 0x30u < 10u) { return TOKS_C_N; }
    if (b == 0x0Au || b == 0x0Du) { return TOKS_C_NL; }
    if (b == 0x09u || b == 0x0Bu || b == 0x0Cu || b == 0x20u) { return TOKS_C_WS; }
    return TOKS_C_P;
}

int64_t toks_compile_cls_flags(const toks_tables *t)
{
    int d = t->tmpl == TOKS_TMPL_DSV3;                      /* dsv3's byte: letters L, the controls outside \s X */
    for (uint32_t b = 0; b < 128u; b++) {                   /* bound: 128 ascii bytes */
        uint8_t want = ascii_class(b);
        if (d && (want & TOKS_C_BASE_MASK) == TOKS_C_L) { want = TOKS_C_L; }
        if (d && (b < 9u || b - 14u < 18u || b == 127u)) { want = TOKS_C_X; }
        if (t->cls_ascii[b] != want) { return -1; }
    }
    for (uint32_t cp = 0x4E00u; d && cp < 0x9F00u; cp++) {  /* bound: 20736 code points */
        if (toks_cls_cp(t, cp) != (uint8_t)(TOKS_C_L | TOKS_C_CJK)) { return 0; }
    }
    if (d) { return TOKS_TF_CJK_D; }
    static const uint32_t lo[2] = { 0x4E00u, 0xAC00u }, hi[2] = { 0xA400u, 0xD700u };
    const uint8_t keep = (uint8_t)(TOKS_C_BASE_MASK | TOKS_C_FOLD_S), both = (uint8_t)(TOKS_C_L | TOKS_C_UPPER | TOKS_C_LOWER);
    int64_t f = TOKS_TF_CJK_L | TOKS_TF_CJK_B;
    for (uint32_t r = 0; r < 2u && f != 0; r++) {           /* bound: 2 ranges */
        for (uint32_t cp = lo[r]; cp < hi[r] && f != 0; cp++) {  /* bound: 22016 + 11008 code points */
            uint8_t c = toks_cls_cp(t, cp);
            if ((uint8_t)(c & keep) != TOKS_C_L) { f &= ~(int64_t)TOKS_TF_CJK_L; }
            if (c != both) { f &= ~(int64_t)TOKS_TF_CJK_B; }
        }
    }
    return f;
}

/* rationale: docs/notes/c-core.md §compile.c.3 */
static uint32_t added_bytes(const toks_config *cfg, const toks_cfg_added *a, uint8_t *out)
{
    if (cfg->algo == TOKS_ALGO_WORDPIECE) {                 /* hf decode prints the form (wordpiece.md §8 D1) */
        if (out != NULL) { memcpy(out, a->form, a->form_len); }
        return a->form_len;
    }
    if (cfg->algo != TOKS_ALGO_BPE_SPM && cfg->algo != TOKS_ALGO_UNIGRAM) { return toks_token_bytes(a->content, a->len, out); }
    uint32_t p = a->pfx != 0u ? 3u : 0u;                    /* hf prints a normalized token's form: "▁" + content */
    if (out != NULL) { memcpy(out, "\xE2\x96\x81", p); memcpy(out + p, a->content, a->len); }
    return p + a->len;
}

/* rationale: docs/notes/c-core.md §compile.c.4 */
static uint32_t vocab_bytes(const toks_config *cfg, uint32_t id, uint8_t *out)
{
    if (cfg->algo != TOKS_ALGO_BPE_SPM && cfg->algo != TOKS_ALGO_WORDPIECE && cfg->algo != TOKS_ALGO_UNIGRAM) {
        return toks_token_bytes(cfg->vocab[id], cfg->vocab_len[id], out);
    }
    if (out != NULL && cfg->vocab_len[id] != 0u) { memcpy(out, cfg->vocab[id], cfg->vocab_len[id]); }
    return cfg->vocab_len[id];
}

/* the string hf decode prints for an id that has added tokens is theirs, never the vocab's (the added map is read
 * first, docs/algorithms/spm_bpe.md §8.1). Two tokens share an id where an spm vocab has holes (hf numbers a new
 * token from vocab.len(), which can be a vocab id); a reader may list one token twice (tiktoken.c: both phases).
 * The last listed holds the id, except that a stored form (spm's pfx ▁ + content: hf's normalized_cache, never
 * cleared) outlives a later token without one. Until the lengths are summed, off[id + 1] names that token: TOK_OWN |
 * TOK_FORM when it has a form | its index (a length is < 2^28: the sources are <= 256 MiB; an index < 2^16). */
#define TOK_OWN  0x80000000u
#define TOK_FORM 0x40000000u

/* entry i's bucket in index2 (h4 = 0: phase << 16 | b0 | b1 << 8) or h4 (slot << 1 | phase); -1: not in it */
static int64_t add_key(const toks_tables *t, uint32_t i, int h4)
{
    const toks_added_entry *x = &t->add_entries[i];
    const uint8_t *c = t->add_bytes + x->off;
    if (x->len < (h4 ? 4u : 2u)) { return -1; }
    return h4 ? (int64_t)(((uint64_t)toks_k1_h4(c) << 1) | x->phase)
              : (int64_t)((uint32_t)x->phase << 16 | (uint32_t)c[0] | (uint32_t)c[1] << 8);
}

/* one index: counts -> prefix sums from cand[pos] -> candidates longest first (ties in entry order, the count
 * rebuilt in the high half as a bucket fills). Returns the next free cand slot. */
static uint64_t add_fill(const toks_tables *t, uint64_t *idx, uint64_t nb, uint32_t *cand, uint64_t pos, int h4)
{
    memset(idx, 0, nb * 8u);
    for (uint32_t i = 0; i < t->add_n; i++) {               /* bound: add_n */
        int64_t k = add_key(t, i, h4);
        if (k >= 0) { idx[k] += 1u; }
    }
    for (uint64_t s = 0; s < nb; s++) {                     /* bound: nb slots */
        uint64_t c = idx[s];
        idx[s] = pos;
        pos += c;
    }
    for (uint32_t L = TOKS_MAX_ADDED_BYTES; L >= 2u; L--) { /* bound: 254 lengths */
        for (uint32_t i = 0; i < t->add_n; i++) {           /* bound: add_n */
            int64_t k = add_key(t, i, h4);
            if (k < 0 || t->add_entries[i].len != L) { continue; }
            cand[(uint32_t)idx[k] + (idx[k] >> 32)] = i;
            idx[k] += 1ull << 32;
        }
    }
    return pos;
}

/* ---- K1's long h4 buckets as radix trees (kernels.md §4): the node bound is 2n - 1 per bucket of n tokens (every node
 * ends a token or has >= 2 children), labels and first bytes take at most the tokens' bytes + one per node */
uint64_t toks_added_cand_words(uint64_t n2, uint64_t n4, uint64_t b4)
{
    uint64_t nodes = n4 > TOKS_K1_LIST ? 2u * n4 : 0u;
    return 2u * n2 + (nodes != 0u ? 4u * nodes + (b4 + nodes + 35u) / 4u + 4u : 0u);
}

static const uint8_t *rx_s(const toks_tables *t, uint32_t e) { return t->add_bytes + t->add_entries[e].off; }

static int rx_cmp(const toks_tables *t, uint32_t a, uint32_t b)       /* by bytes, a prefix first */
{
    uint32_t la = t->add_entries[a].len, lb = t->add_entries[b].len;
    int c = memcmp(rx_s(t, a), rx_s(t, b), la < lb ? la : lb);
    return c != 0 ? c : (int)(la > lb) - (int)(la < lb);
}

static void rx_sift(const toks_tables *t, uint32_t *c, uint32_t k, uint32_t n)
{
    for (uint32_t j = 2u * k + 1u; j < n; k = j, j = 2u * k + 1u) {   /* bound: log2 n levels */
        if (j + 1u < n && rx_cmp(t, c[j + 1u], c[j]) > 0) { j++; }
        if (rx_cmp(t, c[k], c[j]) >= 0) { return; }
        uint32_t x = c[k]; c[k] = c[j]; c[j] = x;
    }
}

typedef struct { uint8_t *base; uint32_t node, lab; } rx_cur;          /* byte offsets from base (add_cand) */

/* the node at `at` of the sorted entries c[lo, hi), which share [0, d): its label is their common bytes from d (first vs
 * last), the token ending there (the first: a prefix sorts first), then one child per next byte */
static void rx_node(const toks_tables *t, const uint32_t *c, uint32_t lo, uint32_t hi, uint32_t d, rx_cur *r, uint32_t at)
{
    const uint8_t *f = rx_s(t, c[lo]), *g = rx_s(t, c[hi - 1u]);
    uint32_t lf = t->add_entries[c[lo]].len, lg = t->add_entries[c[hi - 1u]].len, D = d, nch = 0;
    while (D < lf && D < lg && f[D] == g[D]) { D++; }     /* bound: 255 bytes */
    uint32_t ent = lf == D ? c[lo] + 1u : 0u, s = lo + (ent != 0u), ch = r->node, lab = r->lab;
    for (uint32_t i = s; i < hi; i++) { nch += i == s || rx_s(t, c[i])[D] != rx_s(t, c[i - 1u])[D]; }   /* bound: hi */
    r->node += 16u * nch;
    r->lab += (D - d) + nch;
    memcpy(r->base + lab, f + d, D - d);
    uint8_t *nd = r->base + at;
    uint16_t ln = (uint16_t)(D - d), nc = (uint16_t)nch;
    toks_st32(nd + RX_CH, ch);
    toks_st32(nd + RX_LAB, lab);
    memcpy(nd + RX_LLEN, &ln, 2);
    memcpy(nd + RX_NCH, &nc, 2);
    toks_st32(nd + RX_ENT, ent);
    for (uint32_t i = s, k = 0; i < hi; k++) {              /* bound: nch children */
        uint32_t j = i + 1u;
        while (j < hi && rx_s(t, c[j])[D] == rx_s(t, c[i])[D]) { j++; }   /* bound: hi */
        r->base[lab + (D - d) + k] = rx_s(t, c[i])[D];
        rx_node(t, c, i, j, D, r, ch + 16u * k);
        i = j;
    }
}

void toks_added_index(toks_tables *t, uint8_t *shuf, uint64_t *idx, uint32_t *single, uint32_t *cand)
{
    memset(shuf, 0, 64u);
    memset(single, 0, 2u * 256u * 4u);
    t->add_phases = 0;
    for (uint32_t i = 0; i < t->add_n; i++) {               /* bound: add_n */
        const toks_added_entry *x = &t->add_entries[i];
        uint8_t b0 = t->add_bytes[x->off];
        uint8_t bit = (uint8_t)(1u << ((uint32_t)(b0 >> 4) & 7u));
        shuf[x->phase * 32u + (b0 & 15u)] |= bit;           /* lo-nibble table */
        shuf[x->phase * 32u + 16u + (b0 >> 4)] |= bit;      /* hi-nibble table */
        if (x->len == 1u) { single[x->phase * 256u + b0] = i + 1u; }
        t->add_phases |= 1ull << x->phase;
    }
    uint64_t used = add_fill(t, idx + 2u * 65536u, 2u << TOKS_K1_H4_BITS, cand, add_fill(t, idx, 2u * 65536u, cand, 0, 0), 1);
    uint64_t n4 = 0;
    for (uint32_t i = 0; i < t->add_n; i++) { n4 += t->add_entries[i].len >= 4u; }   /* bound: add_n */
    uint8_t *base = (uint8_t *)cand;                        /* nodes from a 16-aligned offset, then the labels */
    rx_cur r = { base, (uint32_t)((((uintptr_t)(cand + used) + 15u) & ~(uintptr_t)15u) - (uintptr_t)base), 0 };
    r.lab = r.node + 16u * (uint32_t)(n4 > TOKS_K1_LIST ? 2u * n4 : 0u);
    for (uint64_t s = 0; s < (2u << TOKS_K1_H4_BITS); s++) {   /* bound: 2048 slots; a long bucket: its radix tree */
        uint64_t *h = &idx[2u * 65536u + s];
        uint32_t n = (uint32_t)(*h >> 32), *c = cand + (uint32_t)*h;
        if (n <= TOKS_K1_LIST) { continue; }
        for (uint32_t k = n / 2u; k-- > 0u;) { rx_sift(t, c, k, n); }   /* heapsort by bytes, bound: n */
        for (uint32_t m = n; m-- > 1u;) {                    /* bound: n */
            uint32_t x = c[0]; c[0] = c[m]; c[m] = x;
            rx_sift(t, c, 0, m);
        }
        uint32_t root = r.node;
        r.node += 16u;
        rx_node(t, c, 0, n, 0, &r, root);
        *h = (uint64_t)root | (uint64_t)n << 32 | 1ull << 63;
    }
    t->add_shufti = shuf;
    t->add_index = idx;
    t->add_single = single;
    t->add_cand = cand;
}

/* the added-token entries (grouped by phase, in file order) and their bytes, then the index (layout.h; docs/kernels.md
 * §4 is how K1 reads it). An entry a later token shadowed stays out (hf's tries hold one token an id, the last listed:
 * toks_compile's byte per entry, in idx until the index is built). */
static void build_added(toks_tables *t, const toks_config *cfg, toks_added_entry *ent, uint8_t *ab, uint8_t *shuf,
                        uint64_t *idx, uint32_t *single, uint32_t *cand)
{
    const uint8_t *shadowed = (const uint8_t *)(const void *)idx;
    uint32_t e = 0;
    uint64_t off = 0;
    for (uint32_t p = 0; p < 2u; p++) {                     /* bound: 2 phases (grouped by phase) */
        for (uint32_t i = 0; i < cfg->n_added; i++) {       /* bound: n_added */
            const toks_cfg_added *a = &cfg->added[i];
            if ((uint32_t)a->normalized != p || shadowed[i] != 0u) { continue; }
            /* phase 1 matches the normalized form (wordpiece lane: config.c sets form; bpe: none) */
            const uint8_t *mb = (p != 0u && a->form != NULL) ? a->form : a->content;
            uint32_t ml = (p != 0u && a->form != NULL) ? a->form_len : a->len;
            memset(&ent[e], 0, sizeof(ent[e]));
            ent[e].off = (uint32_t)off;
            ent[e].len = (uint16_t)ml;
            ent[e].flags = (uint8_t)((a->special != 0u ? TOKS_AF_SPECIAL : 0u) | (p != 0u ? TOKS_AF_NORMALIZED : 0u) |
                                     (a->lstrip != 0u ? TOKS_AF_LSTRIP : 0u) | (a->rstrip != 0u ? TOKS_AF_RSTRIP : 0u) |
                                     (a->single_word != 0u ? TOKS_AF_SINGLE_WORD : 0u) | (a->pfx != 0u ? TOKS_AF_PFX : 0u));
            ent[e].phase = (uint8_t)p;
            ent[e].id = a->id;
            memcpy(ab + off, mb, ml);
            off += ml;
            e++;
        }
    }
    t->add_entries = ent;
    t->add_bytes = ab;
    t->add_n = e;
    toks_added_index(t, shuf, idx, single, cand);
}

int64_t toks_compile(const struct toks_config *cfg, struct toks_ctx *ctx, toks_arena *parse_ar)
{
    (void)parse_ar;
    uint32_t n_ids = cfg->n_ids, n_vocab = cfg->n_vocab;
    int wp = cfg->algo == TOKS_ALGO_WORDPIECE;              /* token strings are text, not alphabet images */

    /* ---- sizes ---------------------------------------------------------------------------------------- */
    uint64_t tb = 0;                                        /* token bytes */
    for (uint32_t id = 0; id < n_vocab; id++) {             /* bound: n_vocab */
        tb += vocab_bytes(cfg, id, NULL);                    /* decode-only ids: raw utf-8 */
    }
    uint64_t ab = 0;                                        /* added contents (phase 1: forms) */
    uint32_t n_long = 0, n4 = 0;
    uint64_t b4 = 0;
    for (uint32_t i = 0; i < cfg->n_added; i++) {           /* bound: n_added */
        const toks_cfg_added *a = &cfg->added[i];
        uint32_t ml = (a->normalized != 0u && a->form != NULL) ? a->form_len : a->len;   /* what K1 matches */
        tb += added_bytes(cfg, a, NULL);                    /* a bound: the id's token's or the vocab's, below */
        ab += ml;
        n_long += (ml >= 2u) ? 1u : 0u;
        n4 += (ml >= 4u) ? 1u : 0u;
        b4 += (ml >= 4u) ? ml : 0u;
    }
    if (tb > 0xFFFFFFFFu) { return TOKS_E_LIMIT; }

    uint8_t *ctmp = NULL;                                   /* class tables: built, measured, copied */
    uint64_t ctmp_len = 0, cls = 0;
    toks_class_tables ct;
    memset(&ct, 0, sizeof(ct));
    if (cfg->pattern != NULL) {
        uint32_t cf = cfg->pattern->class_flags;           /* the pattern table's (config.c) */
        ctmp_len = toks_classes_bytes(cf);
        ctmp = (uint8_t *)toks_plat_alloc(ctmp_len);
        if (ctmp == NULL) { return TOKS_E_NOMEM; }
        int used = toks_classes_build(cf, ctmp, ctmp_len, &ct);
        if (used <= 0) { toks_plat_free(ctmp, ctmp_len); return TOKS_E_NOMEM; }
        cls = (uint64_t)used;
    }

    uint64_t o_bytes = up64(((uint64_t)n_ids + 1u) * 4u);
    uint64_t o_cls   = o_bytes + up64(tb);
    uint64_t o_spec  = o_cls + up64(cls);
    uint64_t o_ent   = o_spec + up64(((uint64_t)n_ids + 63u) / 64u * 8u);
    uint64_t o_ab    = o_ent + up64((uint64_t)cfg->n_added * sizeof(toks_added_entry));
    uint64_t o_shuf  = o_ab + up64(ab);
    uint64_t o_idx   = o_shuf + 64u;
    uint64_t o_one   = o_idx + 2u * 65536u * 8u + (16u << TOKS_K1_H4_BITS);  /* index2, then h4 */
    uint64_t o_cand  = o_one + 2u * 256u * 4u;
    uint64_t total   = (cfg->n_added != 0u) ? o_cand + up64(4u * toks_added_cand_words(n_long, n4, b4)) : o_ent;

    uint8_t *mem = toks_plat_arena(total);                  /* 2 MiB-aligned, huge-page advised */
    if (mem == NULL) {
        if (ctmp != NULL) { toks_plat_free(ctmp, ctmp_len); }
        return TOKS_E_NOMEM;
    }
    uint8_t *base = mem;

    /* ---- fill ----------------------------------------------------------------------------------------- */
    toks_tables *t = &ctx->t;
    memset(t, 0, sizeof(*t));
    t->magic = TOKS_TABLES_MAGIC;
    t->version = TOKS_TABLES_VERSION;
    t->algo = (cfg->algo == TOKS_ALGO_BPE_SPM || cfg->algo == TOKS_ALGO_WORDPIECE || cfg->algo == TOKS_ALGO_UNIGRAM)
                  ? cfg->algo : TOKS_ALGO_BPE_BYTELEVEL;
    t->tmpl = (cfg->pattern != NULL) ? cfg->pattern->tmpl : TOKS_TMPL_NONE;
    t->tmpl_params = (cfg->pattern != NULL) ? cfg->pattern->params : 0u;
    t->flags = (cfg->ignore_merges != 0u) ? TOKS_TF_IGNORE_MERGES : 0u;
    t->n_ids = n_ids;

    uint32_t *off = (uint32_t *)toks_tab(base, 0u, ((uint64_t)n_ids + 1u) * 4u, TOKS_X_TOK_OFF);   /* lengths, then sums */
    uint8_t *bytes = (uint8_t *)toks_tab(base, o_bytes, tb, TOKS_X_TOK_BYTES);
    uint8_t *idx = cfg->n_added != 0u ? (uint8_t *)toks_tab(base, o_idx, o_one - o_idx, TOKS_X_ADD_INDEX) : NULL;
    memset(off, 0, ((size_t)n_ids + 1u) * 4u);
    for (uint32_t id = 0; id < n_vocab; id++) {             /* bound: n_vocab */
        off[id + 1u] = vocab_bytes(cfg, id, NULL);
    }
    uint8_t *shadowed = idx;                                /* build_added's: 1 = a later token took the id */
    if (shadowed != NULL) { memset(shadowed, 0, cfg->n_added); }
    for (uint32_t i = 0; i < cfg->n_added; i++) {           /* bound: n_added; the token that holds the id (TOK_OWN) */
        const toks_cfg_added *a = &cfg->added[i];
        uint32_t *o = &off[a->id + 1u], form = a->pfx != 0u ? TOK_FORM : 0u, k = *o & 0xFFFFu;
        if ((*o & TOK_OWN) != 0u && (cfg->added[k].len != a->len || memcmp(cfg->added[k].content, a->content, a->len) != 0)) {
            shadowed[k] = 1u;                               /* two contents at most an id: one numbered, one found */
        }
        if ((*o & TOK_OWN) == 0u || form != 0u || (*o & TOK_FORM) == 0u) { *o = TOK_OWN | form | i; }
    }
    for (uint32_t id = 0, at = 0; id < n_ids; id++) {       /* bound: n_ids; each id's bytes once, at = off[id] */
        uint32_t v = off[id + 1u];
        if ((v & TOK_OWN) != 0u) { at += added_bytes(cfg, &cfg->added[v & 0xFFFFu], bytes + at); }
        else if (id < n_vocab) { at += vocab_bytes(cfg, id, bytes + at); }
        off[id + 1u] = at;
    }
    t->tok_off = off;
    t->tok_bytes = (const uint8_t *)toks_tab_fit(bytes, off[n_ids], TOKS_X_TOK_BYTES);   /* tb was a bound */

    if (ctmp != NULL) {                                     /* same offsets inside the block */
        uint8_t *ca = (uint8_t *)toks_tab(base, o_cls + (uint64_t)(ct.ascii - ctmp), 128u, TOKS_X_CLS_ASCII);
        uint8_t *c1 = (uint8_t *)toks_tab(base, o_cls + (uint64_t)((const uint8_t *)ct.stage1 - ctmp), 0x1100u * 2u,
                                          TOKS_X_CLS_STAGE1);
        uint8_t *c2 = (uint8_t *)toks_tab(base, o_cls + (uint64_t)(ct.stage2 - ctmp), (uint64_t)ct.n_blocks * 256u,
                                          TOKS_X_CLS_STAGE2);
        memcpy(ca, ct.ascii, 128u);
        memcpy(c1, ct.stage1, 0x1100u * 2u);
        memcpy(c2, ct.stage2, (size_t)ct.n_blocks * 256u);
        t->cls_ascii = ca;
        t->cls_stage1 = (const uint16_t *)(const void *)c1;
        t->cls_stage2 = c2;
        t->cls_nblocks = ct.n_blocks;
        toks_plat_free(ctmp, ctmp_len);
        int64_t cf = toks_compile_cls_flags(t);            /* kernels.md §2: the K3 tiers' preconditions, else the twin */
        t->flags |= cf < 0 ? TOKS_TF_TWIN : (uint32_t)cf;
    }

    uint64_t spec_bytes = ((uint64_t)n_ids + 63u) / 64u * 8u;
    uint8_t *spec = (uint8_t *)toks_tab(base, o_spec, spec_bytes, TOKS_X_SPECIAL);   /* bit id = byte id >> 3, bit id & 7 */
    memset(spec, 0, (size_t)spec_bytes);
    int str = wp || cfg->algo == TOKS_ALGO_BPE_SPM || cfg->algo == TOKS_ALGO_UNIGRAM;   /* tok_bytes are strings */
    for (uint32_t i = 0; i < cfg->n_added; i++) {           /* bound: n_added */
        const toks_cfg_added *a = &cfg->added[i];
        uint32_t id = a->id, o = off[id], sl = off[id + 1u] - o;
        int skip = a->special != 0u;
        if (str && (sl != a->len || memcmp(bytes + o, a->content, sl) != 0)) {
            /* hf skips by the id's STRING (spm_bpe.md §8.1, wordpiece.md §8 D2), the tok_bytes decode reads: one
               other than the content (a normalized form: wordpiece's, spm's ▁ + content) is skipped iff it equals
               some special content (n_added^2 at load at most). Byte-level's is always the content (config.c). */
            skip = 0;
            for (uint32_t j = 0; j < cfg->n_added && !skip; j++) {   /* bound: n_added */
                const toks_cfg_added *b = &cfg->added[j];
                skip = b->special != 0u && b->len == sl && memcmp(b->content, bytes + o, sl) == 0;
            }
        }
        if (skip) { spec[id >> 3] |= (uint8_t)(1u << (id & 7u)); }
    }
    void *spec_words = spec;
    ctx->special_ids = spec_words;

    if (cfg->n_added != 0u) {
        build_added(t, cfg, (toks_added_entry *)toks_tab(base, o_ent, (uint64_t)cfg->n_added * sizeof(toks_added_entry),
                                                         TOKS_X_ADD_ENTRIES),
                    (uint8_t *)toks_tab(base, o_ab, ab, TOKS_X_ADD_BYTES), (uint8_t *)toks_tab(base, o_shuf, 64u, TOKS_X_ADD_SHUFTI),
                    (uint64_t *)(void *)idx, (uint32_t *)toks_tab(base, o_one, o_cand - o_one, TOKS_X_ADD_SINGLE),
                    (uint32_t *)toks_tab(base, o_cand, 4u * toks_added_cand_words(n_long, n4, b4), TOKS_X_ADD_CAND));
    }
    toks_tab_seal(mem, total);

    /* the single template: prefix ids (before $A), then suffix ids. config.c bounds it to 64. */
    uint32_t k = 0;
    memset(ctx->pp_ids, 0, sizeof(ctx->pp_ids));
    ctx->n_pp_prefix = 0u;
    for (uint32_t i = 0; i < cfg->n_pp_single; i++) {       /* bound: n_pp_single <= 64 */
        if (cfg->pp_single[i].kind == TOKS_PPS_TOK) { ctx->pp_ids[k++] = cfg->pp_single[i].id; }
        else { ctx->n_pp_prefix = k; }                      /* $A: config.c guarantees exactly one */
    }
    ctx->n_pp_suffix = k - ctx->n_pp_prefix;

    ctx->dec_byte_level = cfg->dec_byte_level;
    ctx->nfc = cfg->nfc;
    ctx->cut_chunk = cfg->cut_chunk;
    ctx->cut_run = cfg->cut_run;
    ctx->has_drop = cfg->has_drop;
    memcpy(ctx->drop, cfg->drop, sizeof(ctx->drop));
    ctx->drop_unk = cfg->drop_unk;
    ctx->drop_fuse = cfg->drop_fuse;
    ctx->o = cfg->o;                                        /* encode options, the WordPiece decoder */
    ctx->mem_tables = mem;
    ctx->mem_tables_len = total;
    return 0;
}

/* ---- the template invariants (compile.h) ---------------------------------------------------------------- */

const char *toks_tmpl_invalid(const toks_tables *t)
{
    uint32_t p = t->tmpl_params;
    int han = 0, dsv3 = 0;
    if (t->tmpl == TOKS_TMPL_NONE) {
        return (p == 0u) ? NULL : "template parameters: TOKS_TMPL_NONE takes none";
    } else if (t->tmpl == TOKS_TMPL_CL100K) {
        if ((p & ~(uint32_t)(0x7Fu | TOKS_TP_DIGIT_CUT | TOKS_TP_NL_CUT | TOKS_TP_GB_SP)) != 0u || (p & TOKS_TP_CONTR_MASK) == TOKS_TP_CONTR_MASK ||
            ((p & TOKS_TP_DIGIT_CUT) != 0u && (p & TOKS_TP_DIGITS_MASK) != TOKS_TP_DIGITS_1 &&
             (p & TOKS_TP_DIGITS_MASK) != TOKS_TP_DIGITS_1_3)) {
            return "template parameters: undefined for cl100k (kernels.md §3)";
        }
    } else if (t->tmpl == TOKS_TMPL_O200K) {
        if ((p & ~(uint32_t)(TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1 | TOKS_TP_HAN | TOKS_TP_NO_SLASH)) != 0u) {
            return "template parameters: undefined for o200k (docs/templates/o200k.md §1, §6)";
        }
        han = (p & TOKS_TP_HAN) != 0u;
    } else if (t->tmpl == TOKS_TMPL_DSV3) {
        if (p != 0u) { return "template parameters: dsv3 takes none (docs/templates/dsv3.md)"; }
        dsv3 = 1;
    } else {
        return "template: unknown";
    }
    if (t->cls_ascii == NULL || t->cls_stage1 == NULL || t->cls_stage2 == NULL || t->cls_nblocks == 0u ||
        t->cls_nblocks > 0x1100u) {
        return "class tables: missing";
    }
    for (uint32_t i = 0; i < 0x1100u; i++) {                /* bound: 0x1100 blocks of 256 code points */
        if ((uint64_t)t->cls_stage1[i] >= t->cls_nblocks) { return "class tables: block index out of range"; }
    }
    uint8_t any = 0;
    for (uint32_t i = 0; i < 128u; i++) { any = (uint8_t)(any | t->cls_ascii[i]); }       /* bound: 128 */
    if ((any & TOKS_C_HAN) != 0u) { return "class tables: Han bit on an ascii byte"; }
    for (uint64_t i = 0; i < t->cls_nblocks * 256u; i++) {  /* bound: n_blocks * 256 <= 0x110000 */
        any = (uint8_t)(any | t->cls_stage2[i]);
    }
    if (dsv3 != 0) {                                        /* the dsv3 byte (classes.h): bit 7 is CJK there */
        if ((toks_cls_cp(t, 0x4E00u) & TOKS_C_CJK) == 0u || (t->cls_ascii[0] & TOKS_C_BASE_MASK) != TOKS_C_X) {
            return "class tables: dsv3 without its classes (TOKS_CLASSES_DSV3)";
        }
        return NULL;
    }
    if ((t->cls_ascii[0] & TOKS_C_BASE_MASK) == TOKS_C_X) { return "class tables: dsv3's outside dsv3"; }
    if (han != 0 && (toks_cls_cp(t, 0x4E00u) & TOKS_C_HAN) == 0u) {
        return "class tables: TOKS_TP_HAN without the Han bit (TOKS_CLASSES_HAN)";
    }
    if (han == 0 && (any & TOKS_C_HAN) != 0u) {
        return "class tables: the Han bit without TOKS_TP_HAN";
    }
    if (t->tmpl == TOKS_TMPL_O200K && (toks_cls_cp(t, 0x0301u) & TOKS_C_BASE_MASK) != TOKS_C_P) {
        return "class tables: o200k with marks folded into letters";
    }
    return NULL;
}

/* ---- toks_encode_bound's r and g (include/toks.h "capacity"), set once per context at load ------------------------
 *
 * Claim: for every text of len bytes and every flags, toks_encode returns at most ceil(r len) + g ids, where
 *   r = max(x f, (1 + p0) / l0, x (1 + p1) / l1)   (a token term only where such tokens exist)
 *   g = the template's ids + pfirst
 * and a context that pads (o.pad_on) at most its padded length of that (api.c). Every id is (a) a template id, at
 * most n_pp_prefix + n_pp_suffix per call; (b) an added token's, one per match; (c) a prefix: the ▁ a
 * sentencepiece-style model gives a text unit; (d) a text id, from the model on the normalized text. Truncation only
 * drops ids. A unit is a phase-0 gap or a split of one between phase-1 matches: every unit but the first follows a
 * match.
 *
 * (d) per family: x normalized units per input byte, f ids per unit.
 *   byte-level bpe (gpt2, llama3, o200k, qwen, kimi; the generic engine; a tiktoken wrapper's cuts): a unit is a
 *     normalized byte, f = 1: a token is >= 1 byte and merges only join; a byte the vocab lacks gives one unk or none
 *     (has_drop). x = 1 without a normalizer; with c->nfc, TOKS_NORM_X: 3 for NFC (U+1D160: 4 bytes become 12), 11
 *     for NFKC (U+FDFA: 3 become 33), the generator's scan of these tables (TOKS_NFC_EXPANSION, static-asserted in
 *     norm.c), so load scans nothing.
 *   sentencepiece-style bpe: a unit is an input byte, x = f = 1. The fold (spm_bpe.md §6.1) refuses an image
 *     phi(c) != c or a prefix that is not a one-char vocab string, so a mapped char is one symbol; any other char is
 *     one symbol, its own k bytes' byte-fallback ids, one unk or nothing; an invalid byte one <0xHH> or unk.
 *   wordpiece: a unit is a normalized char, f = 1: a subword is >= 1 char, [UNK] (and max_input_chars_per_word) one id
 *     per word. x = 1: BertNormalizer writes at most a char per input byte (lowercase: U+0130's 2 bytes become 2
 *     chars; strip_accents: NFD then the marks out, a Hangul syllable's 3 bytes become 3 jamo; the spaces of
 *     handle_chinese_chars and clean_text are no ids).
 *   unigram: a unit is a normalized char, x = pre_x cx: the steps' TOKS_NORM_X (bytes) times cx, the charsmap's most
 *     chars written per key byte (pc_chars_x, the one scan of the map at load: nmt_nfkc's U+FDFA, 18 chars for 3
 *     bytes, so 6). A piece is >= 1 char, a char with none one unk, f = 1. Under byte fallback such a char is its
 *     utf-8 bytes instead: units are then bytes, U+2581 counted as the one byte it replaces (unigram.md §10.3), x =
 *     pre_x work_x (toks_pc_build's max_expand), and f = 3 when ▁ is no piece (its 3 bytes in 1 unit), else 1.
 *
 * (b), (c): a prefix is p ids: spm's prefix char is a one-char vocab string, 1; unigram's ▁ is 1 as a piece, else 3
 * (byte fallback) or 1 (unk). Charge each match its id and the prefix of the unit after it: p0 for a phase-0 match
 * (spm GAP and ALWAYS; unigram's Metaspace always, and its Replace ^ -> ▁, whose ▁ is one more normalized unit per
 * gap that a phase-1 match may take: max(p, 1 + p1)), p1 for a phase-1 match (spm ALWAYS; unigram Metaspace always;
 * spm GAP gives a unit after a phase-1 match no prefix). A phase-0 match covers >= l0 input bytes (its strip runs
 * only add), a phase-1 match >= l1 units: its form's chars (spm's ▁ + content form may take the zero-width prefix:
 * one less). So a byte costs at most (1 + p0) / l0 ids inside a phase-0 match and x max(f, (1 + p1) / l1) elsewhere;
 * the first unit's prefix, pfirst, has no match to charge and is in g.
 *
 * The term that binds, the census (tests/c/test_bound.c prints every context's): x f everywhere. The token terms bind
 * on none of the cached tokenizers: the closest are llama2's <s> under Prepend (p0 = 1, l0 = 3: 2 / 3) and gemma's
 * one-byte added tokens, which get no prefix (p0 = 0, l0 = 1: 1). r = 1 where nothing expands, and then g is
 * reached (an input of bytes that are one id each). */
static void bound_max(toks_bound_terms *b, uint32_t num, uint32_t den, uint32_t why)
{
    if ((uint64_t)num * b->den > (uint64_t)b->num * den) { b->num = num; b->den = den; b->why = why; }
}

/* 1 when the unigram model has U+2581 as a one-char piece: its trie path (remapped: the byte 0x20) ends a piece */
static int uni_has_meta(const toks_uni *u)
{
    static const uint8_t M[3] = { 0xE2u, 0x96u, 0x81u };
    uint32_t node = 0u, nb = u->cell[0].base, n = u->remap ? 1u : 3u;
    for (uint32_t k = 0u; k < n; k++) {                     /* bound: 3 bytes */
        uint32_t t = (nb & TOKS_UNI_BASE) + (u->remap ? 0x20u : M[k]);   /* < da_len + 256: padded (unigram.c) */
        if (u->cell[t].check != (int32_t)node) { return 0; }
        node = t;
        nb = u->cell[t].base;
    }
    return (nb & TOKS_UNI_TERM) != 0u;
}

static uint32_t utf8_chars(const uint8_t *s, uint32_t n)
{
    uint32_t k = 0u;
    for (uint32_t i = 0u; i < n; i++) { k += (s[i] & 0xC0u) != 0x80u; }   /* bound: n */
    return k;
}

/* the charsmap's scan: the most chars a key's value writes per byte of the key, as *num / *den >= 1 (U+FDFA's 18
 * chars for 3 bytes in nmt_nfkc); one pass over its live single-char entries (the blocks stage1 does not send to the
 * all-identity block 0) and its multi-char keys */
static void pc_chars_x(const toks_pc *pc, uint32_t *num, uint32_t *den)
{
    *num = 1u;
    *den = 1u;
    for (uint32_t hi = 0u; hi < 0x1100u; hi++) {            /* bound: 0x1100 blocks of 256 code points */
        uint32_t blk = pc->stage1[hi];
        if (blk == 0u) { continue; }
        uint32_t kb = hi < 0x08u ? (hi == 0u ? 1u : 2u) : hi < 0x100u ? 3u : 4u;   /* utf-8 bytes of the block's chars */
        for (uint32_t lo = 0u; lo < 256u; lo++) {           /* bound: 256 */
            uint32_t e = pc->stage2[blk * 256u + lo];
            if (e == 0u) { continue; }
            uint32_t c = utf8_chars(pc->pool + TOKS_PC_OFF(e), TOKS_PC_LEN(e)), k = hi == 0u && lo >= 0x80u ? 2u : kb;
            if ((uint64_t)c * *den > (uint64_t)*num * k) { *num = c; *den = k; }
        }
    }
    for (uint64_t i = 0u; pc->mk_mask != 0u && i <= pc->mk_mask; i++) {   /* bound: the multi-char key table */
        uint64_t key = pc->mk_key[i];
        if (key == 0u) { continue; }
        uint32_t e = pc->mk_val[i], c = utf8_chars(pc->pool + TOKS_PC_OFF(e), TOKS_PC_LEN(e)), k = (uint32_t)(key >> 56);
        if (k != 0u && (uint64_t)c * *den > (uint64_t)*num * k) { *num = c; *den = k; }
    }
}

void toks_bound_terms_of(const toks_ctx *c, toks_bound_terms *b)
{
    memset(b, 0, sizeof *b);
    b->x = 1u;
    b->xd = 1u;
    b->f = 1u;
    if (c->spm != NULL) {
        uint32_t m = c->spm->pfx_mode;
        b->p0 = (m == TOKS_SPM_PFX_GAP || m == TOKS_SPM_PFX_ALWAYS) ? 1u : 0u;
        b->p1 = (m == TOKS_SPM_PFX_ALWAYS) ? 1u : 0u;
        b->pfirst = (m != TOKS_SPM_PFX_NONE) ? 1u : 0u;
    } else if (c->uni != NULL) {
        const toks_uni *u = c->uni;
        int meta = uni_has_meta(u);
        uint32_t p = meta ? 1u : (u->byte_fallback ? 3u : 1u);
        uint32_t ms = (u->cfg.metaspace && u->cfg.meta_prepend) ? p : 0u;
        b->x = u->pre_x * u->work_x;                        /* bytes; without byte fallback the ids are chars: */
        if (!u->byte_fallback && u->cfg.has_charsmap) {     /* the steps' bytes times the map's chars per byte */
            uint32_t cn, cd;
            pc_chars_x(&u->pc, &cn, &cd);
            b->x = u->pre_x * cn;
            b->xd = cd;
        }
        b->f = (u->byte_fallback && !meta) ? 3u : 1u;
        b->p1 = ms;
        b->p0 = ms + (u->cfg.meta_prefix ? (p > 1u + ms ? p : 1u + ms) : 0u);
        b->pfirst = b->p0;
    } else if (c->wp == NULL) {
        b->x = c->nfc != 0u ? TOKS_NORM_X(c->nfc) : 1u;
    }
    b->num = b->x * b->f;
    b->den = b->xd;
    b->why = TOKS_BOUND_TEXT;
    for (uint32_t i = 0u; i < c->t.add_n; i++) {             /* bound: add_n */
        const toks_added_entry *a = &c->t.add_entries[i];
        uint32_t l = a->len;
        if (a->phase != 0u) {                               /* its form's chars, the zero-width prefix off */
            l = utf8_chars(c->t.add_bytes + a->off, a->len);
            l -= (c->spm != NULL && (a->flags & TOKS_AF_PFX) != 0u && l > 1u) ? 1u : 0u;
            if (b->l1 == 0u || l < b->l1) { b->l1 = l; b->e1 = i; }
        } else if (b->l0 == 0u || l < b->l0) {
            b->l0 = l;
            b->e0 = i;
        }
    }
    if (b->l0 != 0u) { bound_max(b, 1u + b->p0, b->l0, TOKS_BOUND_PHASE0); }
    if (b->l1 != 0u) { bound_max(b, b->x * (1u + b->p1), b->xd * b->l1, TOKS_BOUND_PHASE1); }
    b->g = c->n_pp_prefix + c->n_pp_suffix + b->pfirst;
}
