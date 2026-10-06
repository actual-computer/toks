/* unigram.c: the Unigram model's c twin (docs/notes/c-core.md §unigram.c.1) */
#include "norm.h"
#include "unigram.h"
#include "bpe.h"
#include "da.h"

/* rationale: docs/notes/c-core.md §unigram.c.2 */
extern const double toks_pow10[309];               /* src/gen/pow10.c (tools/gen/pow10.py) */

static int is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

/* a score's json number lexeme as serde_json 1.0.151 reads it (unigram.md §2): 0, or -1 when serde refuses it */
static int parse_score(const uint8_t *t, uint32_t n, double *out)
{
    uint32_t i = 0u;
    int neg = 0;
    if (i < n && t[i] == '-') { neg = 1; i++; }
    uint64_t sig = 0u;
    int64_t exp = 0;
    int ovf = 0, is_int = 1;
    while (i < n && is_digit(t[i])) {                        /* bound: n */
        uint64_t d = (uint64_t)(t[i] - '0');
        if (!ovf && (sig > (UINT64_MAX - d) / 10u)) { ovf = 1; }
        if (ovf) { exp++; } else { sig = sig * 10u + d; }   /* parse_long_integer: later digits count only */
        i++;
    }
    if (i < n && t[i] == '.') {
        is_int = 0;
        i++;
        while (i < n && is_digit(t[i])) {                    /* bound: n */
            uint64_t d = (uint64_t)(t[i] - '0');
            if (!ovf && (sig > (UINT64_MAX - d) / 10u)) { ovf = 1; }
            if (!ovf) { sig = sig * 10u + d; exp--; }       /* parse_decimal_overflow drops the rest */
            i++;
        }
    }
    if (i < n && (t[i] == 'e' || t[i] == 'E')) {
        is_int = 0;
        i++;
        int eneg = 0;
        if (i < n && (t[i] == '+' || t[i] == '-')) { eneg = (t[i] == '-'); i++; }
        int64_t e = 0;
        while (i < n && is_digit(t[i])) {                    /* bound: n */
            e = e * 10 + (int64_t)(t[i] - '0');
            if (e > 2147483647) {                            /* parse_exponent_overflow */
                if (sig != 0u && !eneg) { return -1; }
                *out = neg ? -0.0 : 0.0;
                return 0;
            }
            i++;
        }
        exp = eneg ? exp - e : exp + e;                      /* saturating i32 in serde; |exp| stays far below */
    }
    if (is_int && !ovf) {                                    /* ParserNumber::U64 / I64: `as f64` */
        double f = (double)sig;
        *out = neg ? -f : f;
        return 0;
    }
    double f = (double)sig;                                  /* f64_from_parts */
    for (uint32_t k = 0u; k < 16u; k++) {                    /* bound: |exp| < 2^31 needs < 8 rounds of 308 */
        int64_t ae = exp < 0 ? -exp : exp;
        if (ae <= 308) {
            if (exp >= 0) {
                f = f * toks_pow10[ae];
                if (f > 1.7976931348623157e308) { return -1; }   /* infinite */
            } else {
                f = f / toks_pow10[ae];
            }
            break;
        }
        if (f == 0.0) { break; }
        if (exp >= 0) { return -1; }
        f = f / 1e308;
        exp += 308;
    }
    *out = neg ? -f : f;
    return 0;
}

/* rationale: docs/notes/c-core.md §unigram.c.3 */
#define UNI_MAX_PIECE 127u            /* back-pointer bytes keep bit 7 for the path mark */
#define UNI_FREE      (-1)

/* the remapped bytes of a piece (U+2581 -> 0x20 when remap) into dst; returns their count, or -1 when the
 * piece holds a raw 0x20 under remap (unreachable, dropped) */
static int64_t remap_piece(const uint8_t *s, uint32_t n, int remap, uint8_t *dst)
{
    uint32_t i = 0u;
    int64_t m = 0;
    while (i < n) {                                          /* bound: n */
        if (remap && s[i] == 0x20u) { return -1; }
        if (remap && i + 2u < n && s[i] == 0xE2u && s[i + 1u] == 0x96u && s[i + 2u] == 0x81u) {
            if (dst != NULL) { dst[m] = 0x20u; }
            m++;
            i += 3u;
            continue;
        }
        if (dst != NULL) { dst[m] = s[i]; }
        m++;
        i++;
    }
    return m;
}

static int64_t fail_why(const char **why, const char *w, int64_t code) { *why = w; return code; }

/* the decoder chain as spm.h ops: Metaspace; Replace('▁' -> ' ') > ByteFallback > Fuse; ByteFallback > Replace >
 * Fuse > Strip(' ', 1) (hf's Replace('(?<!\n)^ ' -> '') on the fused string); none: no ops */
static uint32_t dec_ops(const toks_uni_cfg *c, toks_spm_op *op)
{
    memset(op, 0, 4u * sizeof *op);
    memcpy(op[0].a.b, "\xE2\x96\x81", 3);
    op[0].a.n = 3u;
    op[0].kind = TOKS_SPM_D_METASPACE;
    op[0].scheme = c->dec_prepend ? TOKS_SPM_PS_ALWAYS : TOKS_SPM_PS_NEVER;
    if (c->dec != TOKS_UNI_DEC_RBF && c->dec != TOKS_UNI_DEC_BFRF) { return c->dec == TOKS_UNI_DEC_META ? 1u : 0u; }
    uint32_t r = (c->dec == TOKS_UNI_DEC_RBF) ? 0u : 1u;     /* the Replace: first, or after ByteFallback */
    op[r] = op[0];
    op[r].kind = TOKS_SPM_D_REPLACE;
    op[r].b.b[0] = ' ';
    op[r].b.n = 1u;
    op[1u - r].kind = TOKS_SPM_D_BYTE_FALLBACK;
    op[2].kind = TOKS_SPM_D_FUSE;
    op[3].kind = TOKS_SPM_D_STRIP;
    op[3].a.b[0] = ' ';
    op[3].a.n = 1u;
    op[3].start = 1u;
    return 3u + r;
}

/* rationale: docs/notes/c-core.md §unigram.c.4 */
/* the byte at position i of the piece (0x20 for the virtual first symbol) */
static inline uint8_t pbyte(int virt, const uint8_t *p, uint64_t i)
{
    return (virt && i == 0u) ? (uint8_t)0x20u : p[i - (uint64_t)(virt != 0)];
}

/* the exact trie entry for positions [s, e) of the piece, or -1 */
static int32_t exact_id(const toks_uni *u, int virt, const uint8_t *p, uint64_t s, uint64_t e)
{
    uint32_t node = 0u;
    for (uint64_t j = s; j < e; j++) {                       /* bound: e - s */
        uint32_t t = (u->cell[node].base & TOKS_UNI_BASE) + pbyte(virt, p, j);   /* < da_len + 256: padded */
        if (u->cell[t].check != (int32_t)node) { return -1; }
        node = t;
    }
    return u->term[node];
}

/* an unknown (fused unk) string [s, e): its own id when it is a piece, else <0xXX> per byte of its hf form
 * (all present), else unk_id */
static void emit_unknown(const toks_uni *u, int virt, const uint8_t *p, uint64_t s, uint64_t end, toks_emit *e)
{
    int32_t id = exact_id(u, virt, p, s, end);
    if (id >= 0) { toks_put(e, (uint32_t)id); return; }
    if (u->byte_fallback) {
        int all = 1;
        for (uint64_t j = s; j < end && all; j++) {            /* bound: end - s */
            uint8_t c = pbyte(virt, p, j);
            if (u->remap && c == 0x20u) {
                all = u->byte_id[0xE2] >= 0 && u->byte_id[0x96] >= 0 && u->byte_id[0x81] >= 0;
            } else {
                all = u->byte_id[c] >= 0;
            }
        }
        if (all) {
            for (uint64_t j = s; j < end; j++) {               /* bound: end - s */
                uint8_t c = pbyte(virt, p, j);
                if (u->remap && c == 0x20u) {
                    toks_put(e, (uint32_t)u->byte_id[0xE2]);
                    toks_put(e, (uint32_t)u->byte_id[0x96]);
                    toks_put(e, (uint32_t)u->byte_id[0x81]);
                } else {
                    toks_put(e, (uint32_t)u->byte_id[c]);
                }
            }
            return;
        }
    }
    toks_put(e, u->unk_id);
}

/* the model on one piece: a virtual leading U+2581 when virt, then p[0, len) (remapped bytes); delta: len + 2 bytes.
 * best[] is a ring of 128 (an edge spans <= max_piece <= 127 bytes): position e lives in best[e & 127]; a piece of
 * < UNI_BID bytes keeps each end's id too (bid), so the backtrack does not walk the trie again */
#define UNI_BID 512u
static void uni_piece(const toks_uni *u, int virt, const uint8_t *p, uint64_t len, toks_emit *em, uint8_t *delta)
{
    virt = (virt != 0);
    uint64_t L = len + (uint64_t)virt;
    if (L == 0u) { return; }
    double best[UNI_MAX_PIECE + 1u];
    int32_t bid[UNI_BID];
    const toks_uni_cell *cell = u->cell;
    const double *score = u->score;
    const int32_t *term = u->term;
    int keep = L < UNI_BID;
    memset(delta, 0, (size_t)(L + 1u));
    best[0] = 0.0;
    uint64_t s = 0u;
    while (s < L) {                                          /* bound: L (s advances one atom) */
        uint32_t mb = 1u;
        if (!(virt && s == 0u)) {
            uint64_t i = s - (uint64_t)virt;
            mb = toks_utf8_len(p + i, len - i);
            if (mb == 0u) { mb = 1u; }                       /* an invalid atom: one byte (§9) */
        }
        double bs = best[s & UNI_MAX_PIECE];
        int single = 0;
        uint32_t node = 0u, nb = cell[0].base;
        uint64_t lim = L - s < (uint64_t)u->max_piece ? L - s : (uint64_t)u->max_piece;
        int v0 = virt && s == 0u;                            /* the walk starts at the virtual ▁ */
        const uint8_t *q = v0 ? p : p + (s - (uint64_t)virt);  /* else byte k of the walk is q[k] */
        for (uint64_t k = 0u; k < lim; k++) {                /* bound: max_piece; pieces by increasing length */
            uint32_t by = v0 ? (k == 0u ? 0x20u : q[k - 1u]) : q[k];
            uint32_t t = (nb & TOKS_UNI_BASE) + by;          /* < da_len + 256: padded */
            const toks_uni_cell *c = &cell[t];
            if (c->check != (int32_t)node) { break; }
            node = t;
            nb = c->base;
            if (!(nb & TOKS_UNI_TERM)) { continue; }
            uint64_t e = s + k + 1u;
            double cand = score[t] + bs;
            if (delta[e] == 0u || cand > best[e & UNI_MAX_PIECE]) {   /* strictly greater: the first seen keeps ties */
                best[e & UNI_MAX_PIECE] = cand;
                delta[e] = (uint8_t)(k + 1u);
                if (keep) { bid[e] = term[t]; }
            }
            if (k + 1u == mb) { single = 1; }
        }
        if (!single) {                                       /* the unk edge, after every piece from s */
            uint64_t e = s + mb;
            double cand = u->unk_score + bs;
            if (delta[e] == 0u || cand > best[e & UNI_MAX_PIECE]) {
                best[e & UNI_MAX_PIECE] = cand;
                delta[e] = (uint8_t)mb;
                if (keep) { bid[e] = -1; }                   /* one char without its own piece: not a piece */
            }
        }
        s += mb;
    }
    /* mark the path ends (bit 7), then emit spans left to right, fusing unk spans */
    uint64_t e = L;
    while (e > 0u) {                                         /* bound: L (e strictly decreases) */
        uint32_t d = delta[e] & 0x7Fu;
        delta[e] = (uint8_t)(delta[e] | 0x80u);
        e -= d;
    }
    uint64_t ps = 0u, us = 0u;
    int in_unk = 0;
    for (uint64_t x = 1u; x <= L; x++) {                     /* bound: L */
        if ((delta[x] & 0x80u) == 0u) { continue; }
        int32_t id = keep ? bid[x] : exact_id(u, virt, p, ps, x);
        if (id < 0 || (uint32_t)id == u->unk_id) {           /* an unk edge, or the unk piece itself: hf fuses
                                                              * by id (unigram.md §6.3) */
            if (!in_unk) { in_unk = 1; us = ps; }
        } else {
            if (in_unk) { emit_unknown(u, virt, p, us, ps, em); in_unk = 0; }
            toks_put(em, (uint32_t)id);
        }
        ps = x;
    }
    if (in_unk) { emit_unknown(u, virt, p, us, L, em); }
}

/* ---- the piece cache (unigram.md §10.3) ---------------------------------------------------------------- */

/* a piece's key: its bytes (<= 15, read in place from b[0, avail)) and len in byte 15, bit 7 there the virtual
 * leading U+2581 (the canonical form of a remapped string that starts with 0x20) */
static inline bpe_key uni_key(int virt, const uint8_t *b, uint64_t l, uint64_t avail)
{
    bpe_key k = { 0u, 0u };
    if (l != 0u) { k = bpe_key_at(b, avail, 0u, l); }
    if (virt) { k.hi |= UINT64_C(0x80) << 56; }
    return k;
}

/* ids[0, n) of a piece the model answered, 1 <= n <= 4, as a val with tag word tw */
static inline void uni_val(uint32_t val[4], const uint32_t *ids, uint64_t n, uint64_t tw)
{
    bpe_val_pack_tag(val, ids, (uint32_t)n, tw);
}

/* the static piece table (toks_uni.words): every remapped piece whose canonical key fits, in descending score order
 * (the most probable first), valued by the model itself; 0, or -1 when its temporary memory is refused */
typedef struct uni_cand { double score; uint32_t key; uint32_t rsv; } uni_cand;

static void cand_sift(uni_cand *c, uint64_t root, uint64_t end)   /* a min-heap on score: sorts descending */
{
    while (2u * root + 1u < end) {                           /* bound: log2(end) */
        uint64_t m = 2u * root + 1u;
        if (m + 1u < end && c[m + 1u].score < c[m].score) { m++; }
        if (!(c[m].score < c[root].score)) { return; }
        uni_cand t = c[root]; c[root] = c[m]; c[m] = t;
        root = m;
    }
}

static int uni_words(toks_uni *u, uint8_t *words, uint64_t wb, const uint8_t *kb, const ukey *keys, uint64_t nk,
                     const double *score)
{
    uint64_t nc = 0u;
    uni_cand *c = (uni_cand *)toks_plat_alloc((nk + 1u) * sizeof(uni_cand));
    if (c == NULL) { return -1; }
    for (uint64_t i = 0u; i < nk; i++) {                     /* bound: nk */
        if (keys[i].len - (uint32_t)(u->remap && kb[keys[i].off] == 0x20u) > (uint32_t)TOKS_KEY_MAXLEN) { continue; }
        c[nc].score = score[keys[i].id];
        c[nc].key = (uint32_t)i;
        nc++;
    }
    for (uint64_t i = nc / 2u; i > 0u; i--) { cand_sift(c, i - 1u, nc); }   /* bound: nc / 2 */
    for (uint64_t e = nc; e > 1u; e--) {                     /* bound: nc - 1: the least score to the back */
        uni_cand t = c[0]; c[0] = c[e - 1u]; c[e - 1u] = t;
        cand_sift(c, 0u, e - 1u);
    }
    memset(words, 0, (size_t)(wb * TOKS_BUCKET));
    uint8_t delta[TOKS_KEY_MAXLEN + 3u], bytes[TOKS_KEY_MAXLEN + 16u];
    uint32_t ids[32], val[4];
    for (uint64_t j = 0u; j < nc; j++) {                     /* bound: nc */
        const ukey *k = &keys[c[j].key];
        int virt = u->remap && kb[k->off] == 0x20u;
        uint64_t l = k->len - (uint32_t)virt;
        memset(bytes, 0, sizeof bytes);
        memcpy(bytes, kb + k->off + (uint32_t)virt, (size_t)l);
        toks_emit e = { ids, 32u, 0u, UINT64_MAX };
        uni_piece(u, virt, bytes, l, &e, delta);
        if (e.n == 0u || e.n > 4u) { continue; }
        bpe_key key = uni_key(virt, bytes, l, sizeof bytes);
        uni_val(val, ids, e.n, 0u);
        (void)bpe_words_put(words, wb - 1u, toks_spm_whash(key.lo, key.hi), key, val);   /* full buckets: left out */
    }
    toks_plat_free(c, (nk + 1u) * sizeof(uni_cand));
    u->words = words;
    u->words_mask = wb - 1u;
    return 0;
}

int64_t toks_uni_build(const toks_uni_src *src, const toks_uni **uo, uint8_t **memo, uint64_t *mem_leno,
                       const char **why)
{
    *uo = NULL;
    *memo = NULL;
    *mem_leno = 0u;
    if (src->n == 0u) { return fail_why(why, "Unigram vocab empty", TOKS_E_FORMAT); }
    if (src->unk_id < 0) { return fail_why(why, "Unigram without unk_id (hf: MissingUnkId on unknown chars)", TOKS_E_UNSUPPORTED); }
    if ((uint64_t)src->unk_id >= src->n) { return fail_why(why, "Unigram unk_id outside the vocab", TOKS_E_FORMAT); }
    int remap = (src->cfg.metaspace != 0u || src->cfg.meta_replace != 0u);

    /* ---- scores, remapped keys --------------------------------------------------------------------- */
    uint64_t kb_len = 0u;
    for (uint32_t i = 0u; i < src->n; i++) { kb_len += src->piece_len[i]; }   /* bound: n */
    uint64_t tmp_len = kb_len + 64u + (uint64_t)src->n * (sizeof(ukey) + 8u) + 64u;
    uint8_t *tmp = (uint8_t *)toks_plat_alloc(tmp_len);
    if (tmp == NULL) { return fail_why(why, "Unigram build memory", TOKS_E_NOMEM); }
    uint8_t *kb = tmp;
    ukey *keys = (ukey *)(void *)(tmp + ((kb_len + 63u) & ~(uint64_t)63u));
    double *score = (double *)(void *)((uint8_t *)keys + (uint64_t)src->n * sizeof(ukey));
    int64_t ret = 0;
    uint64_t nk = 0u, kpos = 0u, n_dropped = 0u;
    uint32_t max_piece = 0u;
    double min_score = 0.0;
    int32_t byte_id[256];
    for (uint32_t b = 0u; b < 256u; b++) { byte_id[b] = -1; }   /* bound: 256 */
    for (uint32_t i = 0u; i < src->n; i++) {                 /* bound: n */
        if (parse_score(src->score_txt[i], src->score_len[i], &score[i]) != 0) {
            ret = fail_why(why, "Unigram score out of range (serde refuses it)", TOKS_E_FORMAT);
            goto out;
        }
        if (i == 0u || score[i] < min_score) { min_score = score[i]; }
        const uint8_t *s = src->piece[i];
        uint32_t l = src->piece_len[i];
        if (l == 6u && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {   /* hf: format!("<0x{:02X}>") */
            int hi = (s[3] >= '0' && s[3] <= '9') ? s[3] - '0' : (s[3] >= 'A' && s[3] <= 'F') ? s[3] - 'A' + 10 : -1;
            int lo = (s[4] >= '0' && s[4] <= '9') ? s[4] - '0' : (s[4] >= 'A' && s[4] <= 'F') ? s[4] - 'A' + 10 : -1;
            if (hi >= 0 && lo >= 0) { byte_id[hi * 16 + lo] = (int32_t)i; }
        }
        int64_t m = remap_piece(s, l, remap, kb + kpos);
        if (m < 0) { n_dropped++; continue; }
        if (m == 0) { continue; }                            /* the empty piece: the trie never yields it */
        if ((uint64_t)m > UNI_MAX_PIECE) {
            ret = fail_why(why, "Unigram piece longer than 127 bytes", TOKS_E_UNSUPPORTED);
            goto out;
        }
        if ((uint32_t)m > max_piece) { max_piece = (uint32_t)m; }
        keys[nk].off = kpos;
        keys[nk].len = (uint32_t)m;
        keys[nk].id = i;
        nk++;
        kpos += (uint64_t)m;
    }
    int32_t *arr = NULL;
    uint64_t cap = 0u, da_len = 0u;
    const char *dwhy = NULL;
    int64_t dr = toks_da_build(kb, keys, &nk, &arr, &cap, &da_len, &dwhy);
    if (dr != 0) { ret = fail_why(why, dr == TOKS_E_NOMEM ? "Unigram trie memory" : "Unigram trie too large", dr); goto out; }
    const int32_t *base = arr, *check = arr + cap, *term = arr + 2u * cap;
    uint64_t da_bytes = cap * 12u;
    uint8_t *da = (uint8_t *)arr;

    /* ---- the block -------------------------------------------------------------------------------- */
    uint64_t pc_bytes = 0u;
    if (src->cfg.has_charsmap) {
        pc_bytes = toks_pc_bytes(src->charsmap, src->charsmap_len, why);
        if (pc_bytes == 0u) { toks_plat_free(da, da_bytes); ret = TOKS_E_UNSUPPORTED; goto out; }
    }
    uint64_t n_cand = 0u;                                    /* static table keys: canonical length <= 15 */
    for (uint64_t i = 0u; i < nk; i++) {                     /* bound: nk */
        n_cand += keys[i].len - (uint32_t)(remap && kb[keys[i].off] == 0x20u) <= (uint32_t)TOKS_KEY_MAXLEN;
    }
    uint64_t wb = n_cand != 0u ? bpe_pow2(n_cand / 2u + 1u) : 0u;
    uint64_t o_u = 0u;
    uint64_t o_cell = (sizeof(toks_uni) + 63u) & ~(uint64_t)63u;
    uint64_t o_score = o_cell + (((da_len + 256u) * sizeof(toks_uni_cell) + 63u) & ~(uint64_t)63u);
    uint64_t o_term = o_score + ((da_len * 8u + 63u) & ~(uint64_t)63u);
    uint64_t o_pc = o_term + ((da_len * 4u + 63u) & ~(uint64_t)63u);
    uint64_t o_words = o_pc + ((pc_bytes + 64u + 63u) & ~(uint64_t)63u);
    uint64_t total = o_words + wb * TOKS_BUCKET + 64u;
    uint8_t *mem = toks_plat_arena(total);
    if (mem == NULL) { toks_plat_free(da, da_bytes); ret = fail_why(why, "Unigram tables", TOKS_E_NOMEM); goto out; }
    toks_uni *u = (toks_uni *)toks_tab(mem, o_u, sizeof(toks_uni), TOKS_X_UNI);
    memset(u, 0, sizeof *u);
    u->cfg = src->cfg;
    u->n_dec = dec_ops(&src->cfg, u->dec);
    toks_uni_cell *cl = (toks_uni_cell *)toks_tab(mem, o_cell, (da_len + 256u) * sizeof(toks_uni_cell), TOKS_X_UNI_CELLS);
    double *sc = (double *)toks_tab(mem, o_score, da_len * 8u, TOKS_X_UNI_SCORE);
    int32_t *tm = (int32_t *)toks_tab(mem, o_term, da_len * 4u, TOKS_X_UNI_TERM);
    for (uint64_t x = 0u; x < da_len + 256u; x++) {          /* bound: da_len + 256 (the cells a step can reach) */
        int in = x < da_len, t = in ? term[x] : -1;
        cl[x].base = in ? ((uint32_t)base[x] | (t >= 0 ? TOKS_UNI_TERM : 0u)) : 0u;
        cl[x].check = in ? check[x] : UNI_FREE;
        if (in) {
            sc[x] = t >= 0 ? score[t] : 0.0;
            tm[x] = t;
        }
    }
    toks_plat_free(da, da_bytes);
    u->cell = cl;
    u->score = sc;
    u->term = tm;
    u->da_len = (uint32_t)da_len;
    u->n_vocab = src->n;
    u->unk_score = min_score - 10.0;
    u->unk_id = (uint32_t)src->unk_id;
    u->max_piece = max_piece;
    for (uint32_t b = 0u; b < 256u; b++) { u->byte_id[b] = byte_id[b]; }   /* bound: 256 */
    u->byte_fallback = (uint8_t)(src->byte_fallback != 0u);
    u->remap = (uint8_t)remap;
    u->n_dropped = (uint32_t)n_dropped;
    u->work_x = 1u;
    u->pre_x = src->cfg.form != 0u ? TOKS_NORM_X(src->cfg.form) : 1u;
    if (src->cfg.has_charsmap) {
        toks_arena par = { mem + o_pc, pc_bytes + 64u, 0u };
        int64_t r = toks_pc_build(&u->pc, src->charsmap, src->charsmap_len, &par, why);
        if (r != 0) { toks_tab_free(mem, total); ret = r; goto out; }
        u->work_x = u->pc.max_expand;
        if (u->work_x > 11u) {                               /* 3 work areas of 11 len fit the scratch (uni_api.c) */
            toks_tab_free(mem, total);
            ret = fail_why(why, "precompiled charsmap expands a byte more than 11x (nmt_nfkc: 11)", TOKS_E_UNSUPPORTED);
            goto out;
        }
    }
    for (uint32_t b = 0u; b < 128u; b++) {                   /* bound: 128 (unigram.md §10.3) */
        u->aent[b] = src->cfg.has_charsmap ? toks_pc_char(&u->pc, b) : 0u;
        u->simple[b] = (uint8_t)(b != 0x20u && !(src->cfg.ws_split && toks_is_regex_ws(b)) &&
                                 !(src->cfg.has_charsmap && (u->aent[b] != 0u || toks_gc_class(b) != 0u)));
    }
    const toks_uni_cfg *cf = &src->cfg;
    u->fast = (uint8_t)(cf->has_charsmap && cf->metaspace && cf->meta_split && cf->meta_prepend && !cf->meta_prefix &&
                        !cf->meta_replace && remap);
    for (uint32_t b = 0u; b < 128u && u->fast; b++) {        /* bound: 128 */
        uint32_t e = u->aent[b];
        int sp = (b == 0x20u && e == 0u) ||
                 (e != 0u && TOKS_PC_LEN(e) == 1u && u->pc.pool[TOKS_PC_OFF(e)] == 0x20u);
        u->acls[b] = (uint8_t)(u->simple[b] ? 1u : sp ? 2u : 0u);
    }
    if (wb != 0u && uni_words(u, (uint8_t *)toks_tab(mem, o_words, wb * TOKS_BUCKET, TOKS_X_WORDS), wb, kb, keys, nk, score) != 0) {
        toks_tab_free(mem, total);
        ret = fail_why(why, "Unigram build memory", TOKS_E_NOMEM);
        goto out;
    }
    toks_tab_seal(mem, total);
    *uo = u;
    *memo = mem;
    *mem_leno = total;
out:
    toks_plat_free(tmp, tmp_len);
    return ret;
}

uint64_t toks_uni_area(const toks_uni *u, uint64_t len)
{
    return ((uint64_t)u->work_x * u->pre_x * len + 3u + 63u) & ~(uint64_t)63u;
}

/* rationale: docs/notes/c-core.md §unigram.c.5 */
#define NOSRC UINT64_MAX
#define UNI_Q 16u                 /* pieces whose cache probes overlap (unigram.md §10.3) */
#define UNI_RESOLVE_IDS (3u * (TOKS_KEY_MAXLEN + 1u))   /* ids of one queued piece (uni_resolve): 3 a byte, 16 bytes */

typedef struct seg {
    const toks_uni *u;
    const uint8_t  *text;      /* the segment (in-place pieces point into it) */
    uint64_t        tlen;      /* its length: in-place keys never read past it */
    toks_uni_call  *c;         /* the output: ids, or piece end offsets */
    uint8_t        *mbuf;      /* materialized piece bytes */
    uint64_t        mcap;
    uint8_t        *delta;
    uint64_t        npos;      /* bytes of normalized text so far (hf's normalized string, U+2581 as 3 bytes) */
    uint64_t        p_nend;    /* npos after the open piece's last char (its toks_pieces end) */
    /* normalizer stream */
    int             seen;      /* a normalized char reached the pre-tokenizers */
    int             prev_space;
    /* pre-tokenizers */
    int             in_word;
    int             split_start;
    int             started;
    /* the open piece */
    int             p_open, p_virt, p_mat;
    uint64_t        p_s, p_e;  /* in-place span [p_s, p_e) of text */
    uint64_t        p_ml;      /* materialized length */
    int             err;
    toks_dsink     *sink;      /* toks_uni_normalize: the normalized chars go here, not to the pre-tokenizers */
    /* in-place pieces of <= 15 bytes waiting for their cache lines (prefetched when queued), answered in order */
    uint32_t        qn;
    struct { const uint8_t *b; uint64_t l; int virt; uint32_t h; bpe_key k; } q[UNI_Q];
} seg;

/* a val's ids (count in val 0) to e: one 16-byte store when 4 ids fit below cap and the count below lim */
static inline void emit_val(toks_emit *e, const uint8_t *v)
{
    uint32_t w[4];
    memcpy(w, v, 16);
    uint64_t m = w[0] >> TOKS_VAL_COUNT_SHIFT;
    if (e->n + 4u <= e->cap && e->n + m <= e->lim) {
        e->n += bpe_val_put(v, e->out + e->n);              /* out: a caller array, any alignment (core.h) */
        return;
    }
    for (uint64_t j = 0; j < m; j++) { toks_put(e, w[j] & TOKS_ID_MASK); }   /* bound: 4 */
}

/* the model on one piece of key k (hash h) behind the piece cache: the scratch's dynamic cache, then the static table
 * (a hit fills the cache: K5's warm rule), then the Viterbi on b[0, l) (<= 4 ids fill the cache); exact, the key is
 * the whole input (unigram.md §10.3) */
static void uni_resolve(const toks_uni *u, toks_uni_call *c, int virt, const uint8_t *b, uint64_t l, bpe_key k,
                        uint32_t h, uint8_t *delta)
{
    uint8_t *bucket = c->cache != NULL ? c->cache + ((uint64_t)h & c->cache_mask) * TOKS_BUCKET : NULL;
    if (bucket != NULL) {
        const uint8_t *v = bpe_cache_get(bucket, k, c->tw);
        if (v != NULL) { emit_val(&c->e, v); return; }
    }
    uint32_t val[4];
    if (u->words != NULL) {
        const uint8_t *v = bpe_words_probe(u->words, u->words_mask, h, k);
        if (v != NULL) {
            if (bucket != NULL) {
                memcpy(val, v, 16);
                val[2] = (val[2] & TOKS_ID_MASK) | (uint32_t)c->tw;
                val[3] = (val[3] & TOKS_ID_MASK) | (uint32_t)(c->tw >> 32);
                bpe_cache_fill(bucket, k, val);
            }
            emit_val(&c->e, v);
            return;
        }
    }
    /* the piece is <= TOKS_KEY_MAXLEN bytes plus the virtual U+2581 (piece_close), and uni_piece emits at most 3 ids a
     * byte (emit_unknown: byte fallback under remap spells a space as <0xE2><0x96><0x81>): 48. A model that emits more
     * than the buffer holds (none known: this bound is read off the code) is answered again, straight into the
     * caller's emitter, uncached, so the ids stay exact. Found by the T6 campaign (UBSan, unigram.c:571, 2026-10-05):
     * the buffer was 32 for a claimed bound of 18, and a unigram file with byte fallback, the ' ' -> U+2581 Replace
     * and no U+2581 piece under 12 emits 33 ids for ten spaces (tests/fuzz/regress/load_json/ubsan-uni-resolve-*). */
    uint32_t ids[UNI_RESOLVE_IDS];
    toks_emit t = { ids, UNI_RESOLVE_IDS, 0u, UINT64_MAX };
    uni_piece(u, virt, b, l, &t, delta);
    if (t.n > UNI_RESOLVE_IDS) { uni_piece(u, virt, b, l, &c->e, delta); return; }
    if (bucket != NULL && t.n >= 1u && t.n <= 4u) {
        uni_val(val, ids, t.n, c->tw);
        bpe_cache_fill(bucket, k, val);
    }
    for (uint64_t j = 0; j < t.n; j++) { toks_put(&c->e, ids[j]); }   /* bound: UNI_RESOLVE_IDS */
}

/* the queued pieces, in order */
static void q_flush(seg *g)
{
    for (uint32_t j = 0; j < g->qn; j++) {                   /* bound: UNI_Q */
        uni_resolve(g->u, g->c, g->q[j].virt, g->q[j].b, g->q[j].l, g->q[j].k, g->q[j].h, g->delta);
    }
    g->qn = 0u;
}

/* the model on the closed piece: an in-place one of <= 15 bytes is queued with its lines prefetched (the probes of
 * UNI_Q pieces overlap), any other is answered now, after the queue */
static void piece_close(seg *g)
{
    if (!g->p_open) { return; }
    uint64_t l = g->p_mat ? g->p_ml : g->p_e - g->p_s;
    if (g->p_virt || l > 0u) {
        toks_uni_call *c = g->c;
        const toks_uni *u = g->u;
        if (c->pieces) {
            toks_put(&c->e, (uint32_t)(c->nbase + g->p_nend));
        } else if (l > (uint64_t)TOKS_KEY_MAXLEN || (c->cache == NULL && u->words == NULL)) {
            q_flush(g);
            uni_piece(u, g->p_virt, g->p_mat ? g->mbuf : g->text + g->p_s, l, &c->e, g->delta);
        } else if (g->p_mat) {
            q_flush(g);
            bpe_key k = uni_key(g->p_virt, g->mbuf, l, g->mcap);
            uni_resolve(u, c, g->p_virt, g->mbuf, l, k, toks_spm_whash(k.lo, k.hi), g->delta);
        } else {
            bpe_key k = uni_key(g->p_virt, g->text + g->p_s, l, g->tlen - g->p_s);
            uint32_t h = toks_spm_whash(k.lo, k.hi);
            if (c->cache != NULL) { __builtin_prefetch(c->cache + ((uint64_t)h & c->cache_mask) * TOKS_BUCKET); }
            if (u->words != NULL) { __builtin_prefetch(u->words + ((uint64_t)h & u->words_mask) * TOKS_BUCKET); }
            uint32_t j = g->qn++;
            g->q[j].b = g->text + g->p_s;
            g->q[j].l = l;
            g->q[j].virt = g->p_virt;
            g->q[j].h = h;
            g->q[j].k = k;
            if (g->qn == UNI_Q) { q_flush(g); }
        }
    }
    g->p_open = 0;
    g->p_virt = 0;
    g->p_mat = 0;
    g->p_ml = 0u;
    g->p_s = g->p_e = 0u;
}

static void piece_open(seg *g, int virt)
{
    g->p_open = 1;
    g->p_virt = virt;
    g->p_mat = 0;
    g->p_ml = 0u;
    g->p_s = g->p_e = 0u;
}

/* appends one normalized char: its piece bytes b[0, k) (V already 0x20), in place at src when the text
 * holds exactly these bytes there */
static void piece_add(seg *g, const uint8_t *b, uint32_t k, uint64_t src, int in_place)
{
    if (!g->p_open) { piece_open(g, 0); }
    g->p_nend = g->npos;
    if (!g->p_virt && !g->p_mat && g->p_e == g->p_s && k == 1u && b[0] == 0x20u && g->u->remap) {
        g->p_virt = 1;                                       /* a leading V is the virtual symbol */
        return;
    }
    if (!g->p_mat && in_place) {
        if (g->p_e == g->p_s) { g->p_s = src; g->p_e = src + k; return; }
        if (src == g->p_e) { g->p_e += k; return; }
    }
    if (!g->p_mat) {                                         /* switch to the work copy */
        uint64_t l = g->p_e - g->p_s;
        if (l > g->mcap) { g->err = 1; return; }            /* unreachable: mcap >= work_x * len */
        if (l > 0u) { memcpy(g->mbuf, g->text + g->p_s, (size_t)l); }
        g->p_ml = l;
        g->p_mat = 1;
    }
    if (g->p_ml + k > g->mcap) { g->err = 1; return; }
    memcpy(g->mbuf + g->p_ml, b, k);
    g->p_ml += k;
}

static const uint8_t V_BYTE[1] = { 0x20u };

/* Metaspace on one normalized char (§5): x is V (U+0020 or U+2581) or not */
static void meta_char(seg *g, const uint8_t *b, uint32_t k, uint64_t src, int in_place, int is_v)
{
    const toks_uni_cfg *c = &g->u->cfg;
    if (g->split_start) {
        g->split_start = 0;
        if (is_v) { piece_open(g, 1); g->p_nend = g->npos; return; }
        piece_open(g, c->meta_prepend ? 1 : 0);
        piece_add(g, b, k, src, in_place);
        return;
    }
    if (is_v && c->meta_split) {
        piece_close(g);
        piece_open(g, 1);
        g->p_nend = g->npos;
        return;
    }
    if (is_v) { piece_add(g, V_BYTE, 1u, src, in_place && k == 1u); return; }
    piece_add(g, b, k, src, in_place);
}

/* the pre-tokenizer stage on one normalized char */
static void pre_char(seg *g, uint32_t cp, const uint8_t *b, uint32_t k, uint64_t src, int in_place)
{
    const toks_uni_cfg *c = &g->u->cfg;
    int is_v = (cp == 0x20u || cp == 0x2581u);
    if (c->ws_split) {
        if (cp != 0xFFFFFFFFu && toks_is_regex_ws(cp)) {     /* White_Space: ends the word, removed */
            if (g->in_word) { piece_close(g); g->in_word = 0; }
            return;
        }
        if (!g->in_word) { g->in_word = 1; g->split_start = 1; }
        if (c->metaspace) { meta_char(g, b, k, src, in_place, is_v); }
        else { piece_add(g, b, k, src, in_place); }
        return;
    }
    if (c->metaspace) {
        if (!g->started) { g->started = 1; g->split_start = 1; }
        meta_char(g, b, k, src, in_place, is_v);
        return;
    }
    /* no pre-tokenizer: the segment is one piece; U+0020 can only be here after Replace ' ' -> V */
    if (is_v && g->u->remap) { piece_add(g, V_BYTE, 1u, src, in_place && k == 1u); return; }
    piece_add(g, b, k, src, in_place);
}

/* the normalizer stages after the charsmap (collapse, prefix, replace) on one normalized char */
static void norm_char(seg *g, uint32_t cp, const uint8_t *b, uint32_t k, uint32_t nlen, uint64_t src, int in_place)
{
    const toks_uni_cfg *c = &g->u->cfg;
    if (c->collapse) {
        if (cp == 0x20u) {
            if (g->prev_space) { return; }
            g->prev_space = 1;
        } else {
            g->prev_space = 0;
        }
    }
    if (g->sink != NULL) {                                   /* (no Metaspace symbol: uni_api.c refuses it there) */
        uint8_t u[4];
        toks_dput(g->sink, cp == 0xFFFFFFFFu ? b : u, cp == 0xFFFFFFFFu ? 1u : toks_utf8_put(u, cp));
        return;
    }
    if (c->meta_prefix && !g->seen) {
        g->seen = 1;
        g->npos += 3u;
        pre_char(g, 0x2581u, V_BYTE, 1u, NOSRC, 0);          /* '▁' before the first char */
    }
    g->seen = 1;
    if (c->meta_replace && cp == 0x20u) {
        g->npos += 3u;
        pre_char(g, 0x2581u, b, k, src, in_place);           /* the in-place 0x20 is the remapped V */
        return;
    }
    g->npos += nlen;
    pre_char(g, cp, b, k, src, in_place);
}

/* the chars of a charsmap value (pool bytes, valid utf-8) */
static void emit_value(seg *g, uint32_t ent)
{
    const toks_pc *pc = &g->u->pc;
    const uint8_t *v = pc->pool + TOKS_PC_OFF(ent);
    uint32_t vl = TOKS_PC_LEN(ent);
    uint32_t i = 0u;
    while (i < vl) {                                         /* bound: vl */
        uint32_t k = toks_utf8_len(v + i, vl - i);
        if (k == 0u) { k = 1u; }
        uint32_t cp = (k == 1u) ? (uint32_t)v[i] : toks_cp_decode(v + i, k);
        if (cp == 0x2581u && g->u->remap) { norm_char(g, cp, V_BYTE, 1u, 3u, NOSRC, 0); }
        else { norm_char(g, cp, v + i, k, k, NOSRC, 0); }
        i += k;
    }
}

/* an atom of the text at i: its code point (0xFFFFFFFF for an invalid byte) and length */
static uint32_t atom(const uint8_t *t, uint64_t i, uint64_t len, uint32_t *k)
{
    uint32_t l = toks_utf8_len(t + i, len - i);
    if (l == 0u) { *k = 1u; return 0xFFFFFFFFu; }
    *k = l;
    return (l == 1u) ? (uint32_t)t[i] : toks_cp_decode(t + i, l);
}

/* one char of the text, unchanged by the charsmap (or no charsmap): to the next stages, in place */
static void ident_char(seg *g, uint32_t cp, uint64_t i, uint32_t k)
{
    const uint8_t *b = g->text + i;
    if (cp == 0x2581u && g->u->remap) { norm_char(g, cp, V_BYTE, 1u, 3u, i, 0); return; }   /* rewritten */
    norm_char(g, cp, b, k, k, i, 1);
}

/* §3.3 on the grapheme [gs, ge) of nchars atoms */
static void flush_grapheme(seg *g, uint64_t gs, uint64_t ge, uint32_t nchars)
{
    const toks_pc *pc = &g->u->pc;
    uint32_t k1 = 0u;
    uint32_t cp1 = atom(g->text, gs, ge, &k1);
    if (ge - gs < 6u) {
        uint32_t e1 = (cp1 != 0xFFFFFFFFu) ? toks_pc_char(pc, cp1) : 0u;
        if (e1 != 0u) { emit_value(g, e1); return; }         /* the whole grapheme */
        uint64_t pe = gs + k1;
        for (uint32_t c = 1u; c < nchars; c++) {             /* bound: nchars <= 5 */
            uint32_t k = 0u;
            (void)atom(g->text, pe, ge, &k);
            pe += k;
            uint32_t em = toks_pc_multi(pc, g->text + gs, (uint32_t)(pe - gs));
            if (em != 0u) { emit_value(g, em); return; }
        }
    }
    uint64_t i = gs;
    while (i < ge) {                                         /* bound: ge - gs */
        uint32_t k = 0u;
        uint32_t cp = atom(g->text, i, ge, &k);
        uint32_t e = (cp != 0xFFFFFFFFu) ? toks_pc_char(pc, cp) : 0u;
        if (e != 0u) { emit_value(g, e); } else { ident_char(g, cp, i, k); }
        i += k;
    }
}

static uint8_t atom_class(uint32_t cp)
{
    return (cp == 0xFFFFFFFFu) ? (uint8_t)TOKS_GC_CONTROL : toks_gc_class(cp);
}

/* strip: [*a, *b) shrinks over White_Space chars at either end (an invalid byte stops it) */
static void strip_span(const toks_uni_cfg *c, int gap_start, const uint8_t *t, uint64_t *a, uint64_t *b)
{
    if (c->strip_left && gap_start) {
        while (*a < *b) {                                    /* bound: b - a */
            uint32_t k = 0u;
            uint32_t cp = atom(t, *a, *b, &k);
            if (cp == 0xFFFFFFFFu || !toks_is_regex_ws(cp)) { break; }
            *a += k;
        }
    }
    if (c->strip_right) {
        while (*b > *a) {                                    /* bound: b - a */
            uint64_t s = *b - 1u;
            uint32_t back = 0u;
            while (s > *a && (t[s] & 0xC0u) == 0x80u && back < 3u) { s--; back++; }   /* bound: 3 */
            uint32_t k = 0u;
            uint32_t cp = atom(t, s, *b, &k);
            if (cp == 0xFFFFFFFFu || s + k != *b || !toks_is_regex_ws(cp)) { break; }
            *b = s;
        }
    }
}

/* the walk's fast path (unigram.md §10.3): a char that maps to itself, is a grapheme cluster on its own when a simple
 * char follows (class Any without InCB bits), and only extends the open piece (not U+0020, not U+2581, not
 * White_Space under WhitespaceSplit) */
static inline int simple(const toks_uni *u, uint32_t cp)
{
    if (cp < 0x80u) { return u->simple[cp]; }
    if (cp == 0xFFFFFFFFu || cp == 0x2581u || (u->cfg.ws_split && toks_is_regex_ws(cp))) { return 0; }
    return !u->cfg.has_charsmap || (toks_pc_char(&u->pc, cp) == 0u && toks_gc_class(cp) == 0u);
}

/* simple chars text[s, e) after the first char of their run went through ident_char: what norm_char .. piece_add
 * do for each of them, at once (they only extend the open piece; the sink takes them as they are) */
static void bulk(seg *g, uint64_t s, uint64_t e)
{
    if (e <= s) { return; }
    if (g->sink != NULL) { toks_dput(g->sink, g->text + s, e - s); return; }
    g->npos += e - s;
    if (!g->p_mat && g->p_e == s && g->p_e != g->p_s) { g->p_e = e; g->p_nend = g->npos; return; }
    while (s < e) {                                          /* bound: e - s (the copy path, a piece at most) */
        uint32_t n = (e - s) > 0xFFFFu ? 0xFFFFu : (uint32_t)(e - s);
        piece_add(g, g->text + s, n, s, 1);
        s += n;
    }
}

/* ascii chars from text[i] while no grapheme is pending: with a charsmap, one followed by an ascii char (but CR, which
 * a LF joins) is a grapheme of its own (GB4, GB5, GB999), so its entry decides it alone: a run of simple ones in bulk,
 * any other through its value or itself. Returns where it stopped (the slow path's char). */
static uint64_t ascii_run(seg *g, uint64_t i, uint64_t b, int cm)
{
    const toks_uni *u = g->u;
    const uint8_t *t = g->text;
    while (i < b) {                                          /* bound: b - i */
        uint32_t c = t[i];
        if (c >= 0x80u || (cm && (c == 0x0Du || (i + 1u < b && t[i + 1u] >= 0x80u)))) { break; }
        if (u->simple[c]) {
            uint64_t j = i + 1u;
            while (j < b && t[j] < 0x80u && u->simple[t[j]]) { j++; }   /* bound: b - i */
            uint64_t e = (cm && j < b && t[j] >= 0x80u) ? j - 1u : j;  /* > i: t[i + 1] is ascii (or the end) */
            ident_char(g, c, i, 1u);
            bulk(g, i + 1u, e);
            i = e;
            continue;
        }
        uint32_t ent = cm ? u->aent[c] : 0u;
        if (ent != 0u) { emit_value(g, ent); } else { ident_char(g, c, i, 1u); }
        i++;
    }
    return i;
}

/* ascii_run for the census's chain (toks_uni.fast: a charsmap, then [collapse] [WhitespaceSplit] Metaspace(always,
 * split), ids only): the stages of a simple run and of a space written out on the seg's state, the same steps as
 * norm_char .. piece_add take for them (a CR LF is one grapheme: the CR's value, here " "). Stops where ascii_run
 * would, and at any other char. */
static uint64_t ascii_fast(seg *g, uint64_t i, uint64_t b)
{
    const toks_uni *u = g->u;
    const uint8_t *t = g->text;
    const int collapse = u->cfg.collapse, ws = u->cfg.ws_split;
    while (i < b) {                                          /* bound: b - i */
        uint32_t ch = t[i];
        if (ch >= 0x80u) { break; }
        uint32_t k = u->acls[ch];
        if (k == 1u) {                                       /* a simple run: they join the open piece */
            uint64_t j = i + 1u;
            while (j < b && t[j] < 0x80u && u->simple[t[j]]) { j++; }   /* bound: b - i */
            uint64_t e = (j < b && t[j] >= 0x80u) ? j - 1u : j;          /* the last one can join a mark */
            if (e == i) { break; }
            g->prev_space = 0;
            g->seen = 1;
            g->npos += e - i;
            if (ws) {
                if (!g->in_word) { g->in_word = 1; g->split_start = 1; }
            } else if (!g->started) {
                g->started = 1;
                g->split_start = 1;
            }
            if (g->split_start) { g->split_start = 0; piece_open(g, 1); }
            if (g->p_open && !g->p_mat && g->p_e == g->p_s) {
                g->p_s = i;
                g->p_e = e;
                g->p_nend = g->npos;
            } else if (g->p_open && !g->p_mat && g->p_e == i) {
                g->p_e = e;
                g->p_nend = g->npos;
            } else {                                         /* the copy path: piece_add (a piece at most 64 KiB here) */
                g->npos -= e - i;
                bulk(g, i, e);
            }
            i = e;
            continue;
        }
        if (k != 2u) { break; }
        uint64_t n = (ch == 0x0Du && i + 1u < b && t[i + 1u] == 0x0Au) ? 2u : 1u;   /* GB3: CR LF, one value */
        if (i + n < b && t[i + n] >= 0x80u) { break; }      /* a mark can join it */
        i += n;
        if (collapse && g->prev_space) { continue; }         /* ' {2,}' -> ' ' */
        g->prev_space = collapse;
        g->seen = 1;
        g->npos += 1u;
        if (ws) {                                            /* White_Space: ends the word, removed */
            if (g->in_word) { piece_close(g); g->in_word = 0; }
            continue;
        }
        if (!g->started) { g->started = 1; }                 /* split_start's first V: the same as below */
        g->split_start = 0;
        piece_close(g);
        piece_open(g, 1);
        g->p_nend = g->npos;
    }
    return i;
}

/* the normalizer over g->text[0, len): Strip, then the charsmap by grapheme (each char when there is none), each
 * char on to norm_char's stages; ascii by ascii_run, a run of other simple chars as one (bulk) but its last char,
 * which can still join a mark after it */
static void walk(seg *g, uint64_t len, int gap_start)
{
    const toks_uni *u = g->u;
    toks_gc_state gs;
    toks_gc_init(&gs);
    uint64_t a = 0u, b = len;
    strip_span(&u->cfg, gap_start, g->text, &a, &b);
    const uint8_t *t = g->text;
    int cm = u->cfg.has_charsmap;
    uint64_t i = a, g0 = a;
    uint32_t nch = 0u;
    while (i < b) {                                          /* bound: b - a */
        if (nch == 0u && t[i] < 0x80u) {
            uint64_t i0 = i;
            i = (u->fast && g->sink == NULL && !g->c->pieces) ? ascii_fast(g, i, b) : i;
            i = ascii_run(g, i, b, cm);
            if (i != i0) {                                   /* the automaton after an ascii char: its class alone */
                toks_gc_init(&gs);
                (void)toks_gc_step(&gs, toks_gc_class(t[i - 1u]));
                g0 = i;
                if (i >= b) { break; }
            }
        }
        uint32_t k = 0u;
        uint32_t cp = atom(t, i, b, &k);
        if (cm && toks_gc_step(&gs, atom_class(cp)) && nch > 0u) {
            flush_grapheme(g, g0, i, nch);
            g0 = i;
            nch = 0u;
        }
        if (nch == 0u && simple(u, cp)) {                    /* a grapheme starts here (cm) */
            uint64_t j = i + k, last = i;
            while (j < b) {                                  /* bound: b - i */
                uint32_t kj = 0u;
                if (!simple(u, atom(t, j, b, &kj))) { break; }
                last = j;
                j += kj;
            }
            if (!cm) { last = j; }                           /* no graphemes: the whole run */
            if (last > i) {                                  /* the char at i ends its grapheme: a simple one follows */
                ident_char(g, cp, i, k);
                bulk(g, i + k, last);                        /* each followed by a simple char too */
                i = last;                                    /* (cm) gs is Any's, as after each of them */
                g0 = i;
                continue;
            }
        }
        if (!cm) { ident_char(g, cp, i, k); } else { nch++; }
        i += k;
    }
    if (cm && nch > 0u) { flush_grapheme(g, g0, b, nch); }
}

int64_t toks_uni_encode_segment(const toks_uni *u, const uint8_t *text, uint64_t len, int pre_normalized,
                                int gap_start, toks_uni_call *c)
{
    /* the piece copy and the back-pointers: work_x len bytes each (a normalized text needs len) */
    uint64_t one = ((uint64_t)(pre_normalized ? 1u : u->work_x) * len + 3u + 63u) & ~(uint64_t)63u;   /* text: past pre */
    if (c->work_bytes < 2u * one) { return TOKS_E_SCRATCH; }   /* unreachable: toks_uni_run sizes it */
    seg g;
    memset(&g, 0, sizeof g);
    g.u = u;
    g.text = text;
    g.tlen = len;
    g.c = c;
    g.mbuf = c->work;
    g.mcap = one;
    g.delta = c->work + one;
    g.seen = !gap_start;                                     /* TOKS_CONTINUATION: no '(?<!\n)^' prefix */
    if (pre_normalized) {                                    /* text is the normalized gap */
        uint64_t i = 0u;
        while (i < len) {                                    /* bound: len */
            uint32_t k = 0u;
            uint32_t cp = atom(text, i, len, &k);
            g.npos += k;
            if (cp == 0x2581u && u->remap) { pre_char(&g, cp, V_BYTE, 1u, i, 0); }
            else { pre_char(&g, cp, text + i, k, i, 1); }
            i += k;
        }
    } else {
        walk(&g, len, gap_start);
    }
    piece_close(&g);
    q_flush(&g);
    return g.err ? TOKS_E_SCRATCH : (int64_t)g.npos;
}

int64_t toks_uni_normalize(const toks_uni *u, const uint8_t *text, uint64_t len, int gap_start, uint8_t *dst,
                           uint64_t cap)
{
    toks_dsink d = { dst, cap, 0u };
    seg g;
    memset(&g, 0, sizeof g);
    g.u = u;
    g.text = text;
    g.sink = &d;
    walk(&g, len, gap_start);
    return d.n > cap ? -1 : (int64_t)d.n;
}
