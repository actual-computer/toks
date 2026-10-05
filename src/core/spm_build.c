/* spm_build.c: toks_spm_config -> the merge tables + toks_spm (docs/notes/c-core.md §spm_build.c.1) */
#include "spm.h"
#include "bpe.h"

static uint64_t up64(uint64_t v) { return (v + 63u) & ~(uint64_t)63u; }

static int hexval(uint8_t c)
{
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

static int64_t fail(toks_err *err, int64_t code, const char *what)
{
    err->code = code;
    err->what = what;
    return code;
}

/* the code point of a one-char string, or -1 */
static int64_t one_cp(const uint8_t *s, uint32_t n)
{
    if (s == NULL || n == 0u) { return -1; }
    uint32_t k = toks_utf8_len(s, n);
    if (k == 0u || k != n) { return -1; }
    return (k == 1u) ? (int64_t)s[0] : (int64_t)toks_cp_decode(s, k);
}

/* ---- the fold (doc §6.1) -------------------------------------------------------------------------------- */

#define MAXMAP (TOKS_SPM_MAX_OPS + 1u)

typedef struct fold {
    uint32_t src[MAXMAP], dst[MAXMAP], n;   /* phi on its sources (identity elsewhere) */
    int64_t  pfx;                           /* the prefix char (normalizer Prepend, then mapped), or -1 */
    int64_t  repl;                          /* the Metaspace replacement, or -1 */
    uint32_t mode;                          /* TOKS_SPM_PFX_* */
} fold;

static uint32_t phi(const fold *f, uint32_t c)
{
    for (uint32_t i = 0; i < f->n; i++) {                   /* bound: MAXMAP */
        if (f->src[i] == c) { return f->dst[i]; }
    }
    return c;
}

/* compose a one char -> one char substitution a -> b after the steps so far */
static void fold_step(fold *f, uint32_t a, uint32_t b)
{
    int seen = 0;
    for (uint32_t i = 0; i < f->n; i++) {                   /* bound: MAXMAP */
        if (f->dst[i] == a) { f->dst[i] = b; }
        if (f->src[i] == a) { seen = 1; }
    }
    if (f->pfx == (int64_t)a) { f->pfx = b; }
    if (!seen && a != b) {                                  /* n < MAXMAP: one source per step at most */
        f->src[f->n] = a;
        f->dst[f->n] = b;
        f->n++;
    }
}

static int64_t fold_text(const toks_spm_text *tx, fold *f, toks_err *err)
{
    memset(f, 0, sizeof(*f));
    f->pfx = -1;
    f->repl = -1;
    f->mode = TOKS_SPM_PFX_NONE;
    for (uint32_t i = 0; i < tx->n_norm; i++) {             /* bound: TOKS_SPM_MAX_OPS */
        const toks_spm_op *op = &tx->norm[i];
        if (op->kind == TOKS_SPM_N_PREPEND) {
            if (op->a.n == 0u) { continue; }                /* prepends nothing */
            int64_t c = one_cp(op->a.b, op->a.n);
            if (c < 0) { return fail(err, TOKS_E_UNSUPPORTED, "normalizer Prepend of more than one char (the tables hold a one-symbol prefix)"); }
            if (f->pfx >= 0) { return fail(err, TOKS_E_UNSUPPORTED, "normalizer: two Prepend steps"); }
            f->pfx = c;
            f->mode = TOKS_SPM_PFX_GAP;
            continue;
        }
        int64_t a = one_cp(op->a.b, op->a.n), b = one_cp(op->b.b, op->b.n);
        if (a < 0 || b < 0) { return fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace other than one char -> one char (not absorbed by the tables)"); }
        fold_step(f, (uint32_t)a, (uint32_t)b);
    }
    if (tx->metaspace) {
        if (tx->ms_split && f->pfx >= 0) {                  /* a piece of the prefix alone: zero width */
            return fail(err, TOKS_E_UNSUPPORTED, "Metaspace split after a Prepend prefix");
        }
        int64_t r = one_cp(tx->ms_repl.b, tx->ms_repl.n);
        if (r < 0) { return fail(err, TOKS_E_FORMAT, "Metaspace replacement"); }
        fold_step(f, ' ', (uint32_t)r);
        f->repl = r;
        if (tx->ms_scheme != TOKS_SPM_PS_NEVER) {
            if (f->pfx >= 0) {                              /* the unit starts with the Prepend char */
                if (f->pfx != r) { return fail(err, TOKS_E_UNSUPPORTED, "Metaspace prefix after a Prepend prefix"); }
            } else {
                f->mode = (tx->ms_scheme == TOKS_SPM_PS_ALWAYS) ? TOKS_SPM_PFX_ALWAYS : TOKS_SPM_PFX_FIRST;
            }
        }
    }
    return 0;
}

/* doc §6.4: c is unreachable when the text model maps it away and nothing maps to it */
static int unreachable(const fold *f, uint32_t c)
{
    int is_src = 0;
    for (uint32_t i = 0; i < f->n; i++) {                   /* bound: MAXMAP */
        if (f->dst[i] == c) { return 0; }
        if (f->src[i] == c) { is_src = 1; }
    }
    return is_src && f->pfx != (int64_t)c && f->repl != (int64_t)c;
}

/* doc §6.6: a vocab string's input form (an image char that a one-byte input char maps to becomes that byte: "▁" ->
 * " "), when it has 1..15 bytes; else 0 */
static uint32_t input_form(const fold *f, const uint8_t *v, uint32_t n, uint8_t out[15])
{
    uint32_t o = 0;
    for (uint32_t i = 0, k = 1; i < n; i += k) {            /* bound: n */
        k = toks_utf8_len(v + i, n - i);
        uint32_t cp = k == 1u ? v[i] : k != 0u ? toks_cp_decode(v + i, k) : 0u, src = cp;
        for (uint32_t j = 0; j < f->n && k > 1u; j++) {     /* bound: MAXMAP */
            if (f->dst[j] == cp && f->src[j] < 0x80u) { src = f->src[j]; }
        }
        uint32_t w = src != cp ? 1u : k;
        if (k == 0u || o + w > 15u) { return 0u; }
        if (src != cp) { out[o] = (uint8_t)src; } else { memcpy(out + o, v + i, k); }
        o += w;
    }
    return o;
}

/* doc §5.7: the risk classes (bpe_apm_class) of the input bytes that can end (last) / start a string whose last /
 * first char is cp: the char itself unless the fold maps it away, and every input char the fold maps to it */
static uint32_t pre_classes(const fold *f, uint32_t cp, int last)
{
    uint32_t m = 0u, fixed = 1u;
    for (uint32_t i = 0; i <= f->n; i++) {                  /* bound: MAXMAP + 1 (the last turn: cp itself) */
        uint32_t c = i < f->n ? f->src[i] : cp;
        if (i < f->n && c == cp) { fixed = 0u; }
        if (i < f->n ? f->dst[i] != cp : !fixed) { continue; }
        m |= 1u << bpe_apm_class(c < 0x80u ? c : last ? 0x80u : c < 0x800u ? 0xC0u : 0xE0u);
    }
    return m;
}

/* doc §5.7: the ascii pair premerge (kernels.md 5.1 over spm's symbols) into apm (layout.h TOKS_APM_*, keyed by two
 * ascii input bytes); 0 when it does not apply (no unk token: a char can vanish; a merge joins a byte-fallback or unk
 * id: their strings say nothing of the bytes) */
static int spm_apm(const toks_tables *t, const toks_spm *s, const toks_spm_config *cfg, const fold *f, uint8_t *apm)
{
    uint32_t nv = cfg->n_ids, *m = s->unk_id != TOKS_SPM_NONE ? (uint32_t *)toks_plat_alloc(17u * (uint64_t)nv) : NULL;
    if (m == NULL) { return 0; }
    memset(m, 0, 17u * (uint64_t)nv);
    uint32_t *lm = m, *fm = m + nv, *rb = m + 2u * nv, *ra = m + 3u * nv;
    uint8_t *fl = (uint8_t *)(void *)(m + 4u * nv);         /* 1: a byte-fallback or unk id, 2: an ascii byte's image */
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        const uint8_t *v = cfg->vocab[id];
        uint32_t n = cfg->vocab_len[id], k0 = v != NULL && n != 0u ? toks_utf8_len(v, n) : 0u, j = n;
        if (k0 == 0u) { continue; }
        while (j > 1u && (v[j - 1u] & 0xC0u) == 0x80u) { j--; }   /* bound: 3 (the last char's start + 1) */
        lm[id] = pre_classes(f, n - j + 1u > 1u ? toks_cp_decode(v + j - 1u, n - j + 1u) : v[n - 1u], 1);
        fm[id] = pre_classes(f, k0 > 1u ? toks_cp_decode(v, k0) : v[0], 0);
    }
    for (uint32_t b = 0; b < 256u; b++) {                   /* bound: 256 */
        uint32_t id = t->byte2id[b];
        if (id < nv) { lm[id] = fm[id] = 1u << bpe_apm_class(b); fl[id] = 1u; }
        if (b < 128u && (s->ascii[b] & TOKS_SPM_E_ID) < nv) { fl[s->ascii[b] & TOKS_SPM_E_ID] |= 2u; }
    }
    lm[s->unk_id] = fm[s->unk_id] = UINT32_MAX;
    fl[s->unk_id] |= 1u;
    int ok = 1;
    for (uint32_t r = 0; r < cfg->n_merges && ok; r++) { ok = !(fl[cfg->m_left[r]] & 1u) && !(fl[cfg->m_right[r]] & 1u); }   /* bound: n_merges */
    memset(apm + TOKS_APM_PAIRS, 0xFF, TOKS_APM_BYTES - TOKS_APM_PAIRS);
    for (uint32_t b = 0; b < 256u; b++) { apm[b] = (uint8_t)bpe_apm_class(b); }   /* bound: 256 */
    bpe_mt mt = bpe_mt_of(t);
    for (uint32_t r = 0; r < cfg->n_merges && ok; r++) {    /* bound: n_merges; every merge in rank order */
        uint32_t a = cfg->m_left[r], b = cfg->m_right[r], o = cfg->m_out[r];
        rb[b] |= lm[a];                                     /* (x, b): x before a left edge b, prio <= r */
        ra[a] |= fm[b];
        if ((fl[a] & fl[b] & 2u) == 0u || bpe_mt_find(&mt, bpe_pair_key(a, b)) != r) { continue; }
        uint32_t v[4] = { o, ra[b] | ra[o], rb[a] | rb[o], 0u };   /* B = (a, b) -> o at R = r: after, before */
        for (uint32_t x = 0; x < 128u; x++) {               /* bound: 128 x 128 (an image has few ascii bytes) */
            for (uint32_t y = 0; y < 128u && (s->ascii[x] & TOKS_SPM_E_ID) == a; y++) {
                if ((s->ascii[y] & TOKS_SPM_E_ID) == b) { memcpy(apm + TOKS_APM_PAIRS + 16u * (x << 7 | y), v, 16); }
            }
        }
    }
    toks_plat_free(m, 17u * (uint64_t)nv);
    return ok;
}

/* ---- geometry ----------------------------------------------------------------------------------------- */

static uint64_t merge_buckets(uint64_t n)                   /* buckets of 8, load <= 0.5 */
{
    uint64_t b = 2u;
    while (b * 4u < n) { b <<= 1; }                         /* bound: 22 doublings (n < 2^22) */
    return b;
}

/* the pair set: open addressing, home = (key * FIB64 >> 32) & mask. 1 when the key was new. */
static int pair_insert(uint64_t *slots, uint64_t mask, uint64_t key)
{
    uint64_t i = (key * TOKS_FIB64 >> 32) & mask;
    for (uint64_t k = 0; k <= mask; k++) {                  /* bound: the slots (load <= 0.5) */
        if (slots[i] == key) { return 0; }
        if (slots[i] == UINT64_MAX) { slots[i] = key; return 1; }
        i = (i + 1u) & mask;
    }
    return 0;
}

static int pair_has(const uint64_t *slots, uint64_t mask, uint64_t key)
{
    uint64_t i = (key * TOKS_FIB64 >> 32) & mask;
    for (uint64_t k = 0; k <= mask; k++) {                  /* bound: the slots */
        if (slots[i] == UINT64_MAX) { return 0; }
        if (slots[i] == key) { return 1; }
        i = (i + 1u) & mask;
    }
    return 0;
}

/* raw[cp]: the id of the one-char vocab string cp (TOKS_SPM_E_NOID: none), through the stage tables */
static uint32_t raw_id(const uint16_t *st1, const uint32_t *raw2, uint32_t cp)
{
    if (cp >= 0x110000u) { return TOKS_SPM_E_NOID; }
    return raw2[(uint32_t)st1[cp >> 8] * 256u + (cp & 0xFFu)];
}

/* the small index of an image char */
static uint32_t small_index(uint32_t img)
{
    return img < 128u ? img : (img == 0x2581u ? 128u : TOKS_SPM_E_SI_NONE);
}

/* the pairs of adjacent chars inside s[0, n) whose both chars are one-char vocab strings, keyed by ids */
static uint64_t add_pairs(uint64_t *slots, uint64_t mask, const uint16_t *st1, const uint32_t *raw2,
                          const uint8_t *s, uint32_t n, uint32_t *left_bits)
{
    uint64_t added = 0;
    uint32_t i = 0, prev = TOKS_SPM_E_NOID;
    while (i < n) {                                         /* bound: n (i advances >= 1) */
        uint32_t k = toks_utf8_len(s + i, n - i);
        if (k == 0u) { return added; }
        uint32_t cp = (k == 1u) ? (uint32_t)s[i] : toks_cp_decode(s + i, k);
        uint32_t id = raw_id(st1, raw2, cp);
        if (prev != TOKS_SPM_E_NOID && id != TOKS_SPM_E_NOID) {
            if (pair_insert(slots, mask, ((uint64_t)prev << 21) | id)) {
                added++;
                left_bits[prev >> 5] |= 1u << (prev & 31u);
            }
        }
        prev = id;
        i += k;
    }
    return added;
}

/* doc §6.8: phase A (a temporary block: raw char table, byte2id, reachable ids, spanned pairs), then B (the block) */
int64_t toks_spm_build(toks_tables *t, const toks_spm_config *cfg, uint8_t **mem_out, uint64_t *mem_len_out,
                       const toks_spm **out, toks_err *err)
{
    *mem_out = NULL;
    *mem_len_out = 0;
    *out = NULL;
    uint32_t n_ids = t->n_ids;                              /* compile.c's: vocab and added ids */
    uint32_t nv = cfg->n_ids;                               /* the model vocab's id range */
    if (n_ids >= TOKS_MAX_IDS || nv > n_ids || cfg->n_merges >= (1u << TOKS_PRIO_BITS)) {
        return fail(err, TOKS_E_LIMIT, "ids or merges beyond the table widths");
    }
    fold f;
    int64_t r = fold_text(&cfg->text, &f, err);
    if (r != 0) { return r; }
    int identity = (f.n == 0u && f.mode == TOKS_SPM_PFX_NONE);
    int im = (cfg->sflags & TOKS_SPM_IGNORE_MERGES) != 0u;
    if (im && !identity) {
        return fail(err, TOKS_E_UNSUPPORTED, "ignore_merges with a Replace / Prepend / Metaspace text model");
    }

    /* ---- geometry known before the vocab is read ------------------------------------------------------ */
    uint8_t blk_used[0x1100];                               /* which cp >> 8 blocks hold an entry */
    memset(blk_used, 0, sizeof(blk_used));
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        int64_t cp = one_cp(cfg->vocab[id], cfg->vocab_len[id]);
        if (cp >= 0) { blk_used[cp >> 8] = 1u; }
    }
    for (uint32_t i = 0; i < f.n; i++) { blk_used[f.src[i] >> 8] = 1u; }   /* bound: MAXMAP */
    blk_used[0] = 1u;                                       /* ASCII's block always exists (the ascii[] copy) */
    uint32_t n_blocks = 1;                                  /* block 0: all NOID */
    for (uint32_t b = 0; b < 0x1100u; b++) { n_blocks += blk_used[b]; }     /* bound: 0x1100 */
    uint64_t st2_bytes = (uint64_t)n_blocks * 256u * 4u;
    uint64_t bits = ((uint64_t)n_ids + 31u) / 32u * 4u;
    uint64_t ub = 0;                                        /* the vocab's bytes: >= its distinct pairs */
    for (uint32_t id = 0; id < nv; id++) { ub += cfg->vocab_len[id]; }      /* bound: nv */
    uint64_t tps = im ? 0u : bpe_pow2(2u * ub + 2u);

    /* ---- phase A ----------------------------------------------------------------------------------------- */
    uint64_t a_st1 = 0, a_st2 = up64(0x1100u * 2u), a_b2id = a_st2 + up64(st2_bytes);
    uint64_t a_reach = a_b2id + up64(256u * 4u), a_left = a_reach + up64(bits), a_pairs = a_left + up64(bits);
    uint64_t a_total = a_pairs + tps * 8u;
    uint8_t *tmp = (uint8_t *)toks_plat_alloc(a_total);
    if (tmp == NULL) { return fail(err, TOKS_E_NOMEM, "spm build"); }
    memset(tmp, 0, (size_t)a_total);
    uint16_t *st1 = (uint16_t *)(void *)(tmp + a_st1);
    uint32_t *st2 = (uint32_t *)(void *)(tmp + a_st2);
    uint32_t *b2id = (uint32_t *)(void *)(tmp + a_b2id);
    uint32_t *reach = (uint32_t *)(void *)(tmp + a_reach);
    uint32_t *left = (uint32_t *)(void *)(tmp + a_left);
    uint64_t *tpairs = (uint64_t *)(void *)(tmp + a_pairs);
    const char *why = NULL;

    for (uint32_t b = 0; b < 256u; b++) { b2id[b] = TOKS_SPM_NONE; }        /* bound: 256 */
    uint32_t n_bytes = 0;
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        const uint8_t *v = cfg->vocab[id];
        if (v == NULL || cfg->vocab_len[id] != 6u || v[0] != '<' || v[1] != '0' || v[2] != 'x' || v[5] != '>') { continue; }
        int hi = hexval(v[3]), lo = hexval(v[4]);           /* hf: exactly "<0x%02X>", upper-case digits */
        if (hi >= 0 && lo >= 0 && !(v[3] >= 'a' && v[3] <= 'f') && !(v[4] >= 'a' && v[4] <= 'f')) {
            if (b2id[hi * 16 + lo] == TOKS_SPM_NONE) { n_bytes++; }
            b2id[hi * 16 + lo] = id;
        }
    }
    if ((cfg->sflags & TOKS_SPM_UNK_ERROR) && !((cfg->sflags & TOKS_SPM_BYTE_FALLBACK) && n_bytes == 256u)) {
        why = "model unk_token missing from the vocab (hf fails on a char without a token)";
    }

    /* the raw char table: the id of every one-char vocab string */
    for (uint64_t i = 0; i < (uint64_t)n_blocks * 256u; i++) { st2[i] = TOKS_SPM_E_NOID; }   /* bound: blocks */
    uint32_t next_blk = 1;
    for (uint32_t b = 0; b < 0x1100u; b++) {                /* bound: 0x1100 */
        if (blk_used[b] != 0u) { st1[b] = (uint16_t)next_blk++; }
    }
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        int64_t cp = one_cp(cfg->vocab[id], cfg->vocab_len[id]);
        if (cp >= 0) { st2[(uint32_t)st1[cp >> 8] * 256u + ((uint32_t)cp & 0xFFu)] = id; }
    }
    /* the images and the prefix must be vocab chars: a substituted char then always has an id */
    for (uint32_t i = 0; i < f.n && why == NULL; i++) {     /* bound: MAXMAP */
        if (raw_id(st1, st2, f.dst[i]) == TOKS_SPM_E_NOID) { why = "text model substitutes a char that is not a vocab char"; }
    }
    int64_t pc = (f.mode == TOKS_SPM_PFX_GAP) ? f.pfx : (f.mode != TOKS_SPM_PFX_NONE ? f.repl : -1);
    if (why == NULL && pc >= 0 && raw_id(st1, st2, (uint32_t)pc) == TOKS_SPM_E_NOID) { why = "text model prefix char is not a vocab char"; }
    uint32_t id_repl = (f.repl >= 0) ? raw_id(st1, st2, (uint32_t)f.repl) : TOKS_SPM_NONE;
    if (why == NULL && f.repl >= 0 && id_repl == TOKS_SPM_E_NOID) { why = "Metaspace replacement is not a vocab char"; }
    if (why != NULL) {
        toks_plat_free(tmp, a_total);
        return fail(err, TOKS_E_UNSUPPORTED, why);
    }

    /* reachable ids (doc §6.4): images of reachable chars, every byte token, unk; then merges to a fixpoint */
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        int64_t cp = one_cp(cfg->vocab[id], cfg->vocab_len[id]);
        if (cp >= 0 && !unreachable(&f, (uint32_t)cp)) { reach[id >> 5] |= 1u << (id & 31u); }
    }
    for (uint32_t b = 0; b < 256u; b++) {                   /* bound: 256 */
        if (b2id[b] != TOKS_SPM_NONE) { reach[b2id[b] >> 5] |= 1u << (b2id[b] & 31u); }
    }
    if (cfg->unk_id != TOKS_SPM_NONE) { reach[cfg->unk_id >> 5] |= 1u << (cfg->unk_id & 31u); }
    for (uint32_t pass = 0; pass <= cfg->n_merges; pass++) {   /* bound: n_merges + 1 passes (each adds an id or ends) */
        uint32_t grew = 0;
        for (uint32_t m = 0; m < cfg->n_merges; m++) {      /* bound: n_merges */
            uint32_t l = cfg->m_left[m], rr = cfg->m_right[m], o = cfg->m_out[m];
            if (((reach[l >> 5] >> (l & 31u)) & 1u) && ((reach[rr >> 5] >> (rr & 31u)) & 1u) &&
                !((reach[o >> 5] >> (o & 31u)) & 1u)) {
                reach[o >> 5] |= 1u << (o & 31u);
                grew = 1;
            }
        }
        if (!grew) { break; }
    }

    /* the spanned pairs of reachable strings (doc §5.5), keyed by ids, counted */
    uint64_t np = 0;
    if (tps != 0u) {
        memset(tpairs, 0xFF, (size_t)(tps * 8u));
        for (uint32_t id = 0; id < nv; id++) {              /* bound: nv */
            if (cfg->vocab[id] == NULL || !((reach[id >> 5] >> (id & 31u)) & 1u)) { continue; }
            np += add_pairs(tpairs, tps - 1u, st1, st2, cfg->vocab[id], cfg->vocab_len[id], left);
        }
    }

    /* ---- phase B: the block ------------------------------------------------------------------------------ */
    int holes_any = 0;
    for (uint32_t id = 0; id < n_ids && !holes_any; id++) { /* bound: n_ids */
        int vocab = id < nv && cfg->vocab[id] != NULL;
        if (!vocab && t->tok_off[id + 1u] == t->tok_off[id]) { holes_any = 1; }
    }
    uint64_t mb = merge_buckets(cfg->n_merges);
    uint64_t vs = im ? bpe_pow2(2u * (uint64_t)cfg->n_strings + 2u) : 0u;
    uint64_t ps = im ? 0u : bpe_pow2(4u * np + 2u);        /* load <= 1/4: K7's two slots mostly decide */
    uint64_t o_b2id  = up64(sizeof(toks_spm));
    uint64_t o_st1   = o_b2id + up64(256u * 4u);
    uint64_t o_st2   = o_st1 + up64(0x1100u * 2u);
    uint64_t o_slots = o_st2 + up64(st2_bytes);
    uint64_t o_r2id  = o_slots + mb * 64u;
    uint64_t o_vh    = o_r2id + up64((uint64_t)cfg->n_merges * 4u + 4u);
    uint64_t o_pairs = o_vh + vs * 8u;
    uint64_t o_holes = o_pairs + up64((ps + 1u) * 8u);
    uint64_t o_pf    = o_holes + (holes_any ? up64(bits) : 0u);
    uint64_t n_words = 0u;                                  /* the static word table: K5's sizing (bpe_build.c) */
    uint8_t form[15];
    for (uint32_t id = 0; id < nv; id++) { n_words += cfg->vocab[id] != NULL && input_form(&f, cfg->vocab[id], cfg->vocab_len[id], form) != 0u; }   /* bound: nv */
    uint64_t wb = n_words != 0u ? bpe_pow2(n_words / 2u + 1u) : 0u;
    uint64_t o_words = up64(o_pf + mb * 4u);
    uint64_t o_apm   = o_words + wb * TOKS_BUCKET;
    uint64_t o_ab8   = up64(o_apm + TOKS_APM_BYTES);
    uint64_t total   = o_ab8 + (ps != 0u ? 65536u : 0u);
    uint8_t *mem = toks_plat_arena(total);
    if (mem == NULL) {
        toks_plat_free(tmp, a_total);
        return fail(err, TOKS_E_NOMEM, "spm tables");
    }
    memset(mem, 0, (size_t)total);
    toks_spm *s = (toks_spm *)(void *)mem;
    s->unk_id = cfg->unk_id;
    s->sflags = cfg->sflags;
    s->n_blocks = n_blocks;
    s->n_dec = cfg->n_dec;
    s->has_decoder = cfg->has_decoder;
    memcpy(s->dec, cfg->dec, sizeof(s->dec));
    s->ms_split = cfg->text.metaspace ? cfg->text.ms_split : 0u;
    s->pfx_mode = f.mode;
    s->id_repl = id_repl;
    memcpy(mem + o_b2id, b2id, 256u * 4u);
    t->byte2id = (const uint32_t *)(const void *)(mem + o_b2id);

    /* the merge table (layout.h): reachable merges, the last listing of a pair sets its rank */
    uint32_t *r2id = (uint32_t *)(void *)(mem + o_r2id);
    for (uint32_t m = 0; m < cfg->n_merges; m++) { r2id[m] = cfg->m_out[m]; }   /* bound: n_merges */
    s->n_dropped = toks_merge_slots(t, (uint64_t *)(void *)(mem + o_slots), (uint64_t *)(void *)(mem + o_pf), mb, cfg->m_left,
                                    cfg->m_right, cfg->n_merges, reach);
    t->rank2id = r2id;
    t->flags = im ? TOKS_TF_IGNORE_MERGES : 0u;

    /* the vocab hash (ignore_merges; the text model is the identity, every string is reachable) */
    if (vs != 0u) {
        uint64_t *vh = (uint64_t *)(void *)(mem + o_vh);
        memset(vh, 0xFF, (size_t)(vs * 8u));
        for (uint32_t id = 0; id < nv; id++) {              /* bound: nv */
            if (cfg->vocab[id] == NULL) { continue; }
            uint32_t h = bpe_vhash_h(cfg->vocab[id], cfg->vocab_len[id]);
            uint64_t i = (uint64_t)h & (vs - 1u);
            while (vh[i] != UINT64_MAX) { i = (i + 1u) & (vs - 1u); }   /* bound: at most half full */
            vh[i] = ((uint64_t)id << 32) | h;
        }
        t->vhash = vh;
        t->vhash_mask = vs - 1u;
    }

    /* the pair set at load <= 0.5 */
    if (ps != 0u) {
        uint64_t *pairs = (uint64_t *)(void *)(mem + o_pairs);
        memset(pairs, 0xFF, (size_t)((ps + 1u) * 8u));
        for (uint64_t i = 0; i < tps; i++) {                /* bound: tps */
            if (tpairs[i] != UINT64_MAX) { (void)pair_insert(pairs, ps - 1u, tpairs[i]); }
        }
        pairs[ps] = pairs[0];                               /* the mirror: slots i, i + 1 without a wrap */
        s->pairs = pairs;
        s->pairs_mask = ps - 1u;
        s->n_pairs = np;
    }

    /* fold the char table (doc §6.2): every input cp gets its image's id, small index and PAIRED */
    uint16_t *fst1 = (uint16_t *)(void *)(mem + o_st1);
    uint32_t *fst2 = (uint32_t *)(void *)(mem + o_st2);
    memcpy(fst1, st1, 0x1100u * 2u);
    for (uint32_t lo = 0; lo < 256u; lo++) {                /* bound: 256: block 0, every unused block's */
        fst2[lo] = TOKS_SPM_E_NOID | (TOKS_SPM_E_SI_NONE << TOKS_SPM_E_SI_SHIFT);
    }
    for (uint32_t b = 0; b < 0x1100u; b++) {                /* bound: 0x1100 */
        if (fst1[b] == 0u) { continue; }
        for (uint32_t lo = 0; lo < 256u; lo++) {            /* bound: 256 */
            uint32_t cp = (b << 8) | lo;
            uint32_t img = phi(&f, cp);
            uint32_t id = raw_id(st1, st2, img);            /* the image's id (the raw table is A's) */
            uint32_t *e = &fst2[(uint32_t)fst1[b] * 256u + lo];
            if (id == TOKS_SPM_E_NOID) { *e = TOKS_SPM_E_NOID | (TOKS_SPM_E_SI_NONE << TOKS_SPM_E_SI_SHIFT); continue; }
            *e = id | (small_index(img) << TOKS_SPM_E_SI_SHIFT) | (((left[id >> 5] >> (id & 31u)) & 1u) ? TOKS_SPM_E_PAIRED : 0u) |
                 (small_index(img) == TOKS_SPM_E_SI_NONE && !(s->ms_split && id == id_repl) ? TOKS_SPM_E_PLAIN : 0u);
        }
    }
    for (uint32_t c = 0; c < 128u; c++) { s->ascii[c] = fst2[(uint32_t)fst1[0] * 256u + c]; }   /* bound: 128 */
    s->stage1 = fst1;
    s->stage2 = fst2;
    if (pc >= 0) {                                          /* the prefix is already an image: its raw id */
        uint32_t id = raw_id(st1, st2, (uint32_t)pc);
        s->pfx_entry = id | (small_index((uint32_t)pc) << TOKS_SPM_E_SI_SHIFT) |
                       (((left[id >> 5] >> (id & 31u)) & 1u) ? TOKS_SPM_E_PAIRED : 0u);
    }

    /* the cut bitmap over the small alphabet's images: 1 = a certified cut between them */
    if (s->pairs != NULL) {
        for (uint32_t x = 0; x < TOKS_SPM_SMALL; x++) {     /* bound: 129 */
            uint32_t ix = raw_id(st1, st2, x < 128u ? x : 0x2581u);
            for (uint32_t y = 0; y < TOKS_SPM_SMALL; y++) { /* bound: 129 */
                uint32_t iy = raw_id(st1, st2, y < 128u ? y : 0x2581u);
                if (ix == TOKS_SPM_E_NOID || iy == TOKS_SPM_E_NOID) { continue; }
                if (pair_has(s->pairs, s->pairs_mask, ((uint64_t)ix << 21) | iy)) { continue; }
                uint32_t bit = x * TOKS_SPM_SMALL + y;
                s->cut[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
            }
        }
        /* the scan's byte-pair table (spm_c.c): for two ASCII input bytes, its whole decision in one bit */
        uint8_t *ab8 = mem + o_ab8;                         /* the same as bytes (zero elsewhere: the arena) */
        for (uint32_t x = 0; x < 128u; x++) {               /* bound: 128 */
            for (uint32_t y = 0; y < 128u; y++) {           /* bound: 128 */
                uint32_t ey = s->ascii[y], bit = x | (y << 8);
                if ((s->ms_split && (ey & TOKS_SPM_E_ID) == s->id_repl) || toks_spm_cut_between(s, s->ascii[x], ey)) {
                    s->cut_ab[bit >> 5] |= 1u << (bit & 31u);
                    ab8[bit] = 1u;
                }
            }
        }
        s->cut_ab8 = ab8;
    }

    /* holes: ids with neither a vocab string nor an added token's string */
    if (holes_any) {
        uint32_t *h = (uint32_t *)(void *)(mem + o_holes);
        for (uint32_t id = 0; id < n_ids; id++) {           /* bound: n_ids */
            int vocab = id < nv && cfg->vocab[id] != NULL;
            if (!vocab && t->tok_off[id + 1u] == t->tok_off[id]) { h[id >> 5] |= 1u << (id & 31u); }
        }
        s->holes = h;
    }
    t->apm = spm_apm(t, s, cfg, &f, mem + o_apm) ? mem + o_apm : NULL;   /* doc §5.7 */

    /* the static word table (doc §6.6): each vocab string's input form, valued by the model itself, in id order */
    _Alignas(8) uint8_t work[TOKS_SPM_WORK_BYTES(16) + 8u];
    uint32_t ids[16], val[4];
    t->words = wb != 0u ? mem + o_words : NULL;
    t->words_mask = wb - 1u;
    for (uint32_t id = 0, placed = 0; id < nv && placed * 100u < wb * 170u; id++) {   /* bound: nv; load <= 0.85 */
        uint32_t l = cfg->vocab[id] != NULL ? input_form(&f, cfg->vocab[id], cfg->vocab_len[id], form) : 0u;
        uint64_t m = l != 0u ? toks_spm_model(t, s, form, l, ids, work) : 0u;
        if (m == 0u || m > 4u) { continue; }
        bpe_key k = bpe_key_at(form, l, 0u, l);
        bpe_val_pack_tag(val, ids, (uint32_t)m, 0u);
        placed += (uint32_t)bpe_words_put(mem + o_words, wb - 1u, toks_spm_whash(k.lo, k.hi), k, val);
    }
    toks_plat_free(tmp, a_total);
    *mem_out = mem;
    *mem_len_out = total;
    *out = s;
    return 0;
}
