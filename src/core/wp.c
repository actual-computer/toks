/* wp.c: the WordPiece model of toks (docs/algorithms/wordpiece.md §5, §12.3-12.4) (docs/notes/c-core.md §wp.c.1) */
#include "wp.h"
#include "config.h"
#include "bpe.h"
#include "da.h"

#define WP_SMAX 64u                                     /* prefix states kept per start: keys up to 512 bytes */

uint8_t toks_wp_ascii_class(uint32_t f, uint32_t b);   /* wp_scan.c */

/* ---- lookup ------------------------------------------------------------------------------------------------ */

static int wp_eq(const uint8_t *key, const uint8_t *p, uint64_t n, int fold)
{
    uint64_t i = 0;
    while (n - i >= 8u) {                               /* bound: n / 8 */
        uint64_t a = wp_load(key + i, 8), b = wp_load(p + i, 8);
        if ((fold ? wp_fold(b) : b) != a) { return 0; }
        i += 8u;
    }
    if (i < n) {
        uint64_t a = wp_load(key + i, n - i), b = wp_load(p + i, n - i);
        if ((fold ? wp_fold(b) : b) != a) { return 0; }
    }
    return 1;
}

/* the id of key p[0, n) in a table, or -1 */
static int64_t wp_find(const toks_wp_tables *t, const toks_wp_entry *tab, uint64_t mask, uint32_t h,
                       const uint8_t *p, uint64_t n, int fold)
{
    if (tab == NULL) { return -1; }
    uint64_t i = (uint64_t)h & mask;
    for (uint64_t k = 0; k <= mask; k++) {              /* bound: the table (load <= 1/2 ends it far sooner) */
        const toks_wp_entry *e = &tab[i];
        if (e->len == 0u) { return -1; }
        if (e->h == h && e->len == n && wp_eq(t->keys + e->off, p, n, fold)) { return (int64_t)e->id; }
        i = (i + 1u) & mask;
    }
    return -1;
}

/* ---- the model --------------------------------------------------------------------------------------------- */

uint64_t toks_wp_piece_ids(const toks_wp_tables *t, const uint8_t *p, uint64_t len, uint32_t flags, uint32_t *out,
                           uint64_t *probes)
{
    if (flags & (TOKS_WPP_OVER | TOKS_WPP_INVALID) || len == 0u) {
        toks_st32(out, t->unk_id);                      /* out may be the caller's array: any alignment */
        return 1;
    }
    int fold = (t->flags & TOKS_WPF_LOWER) != 0u;
    if (t->wcell != NULL) {                             /* W2 on the tries: the longest key from each start */
        uint64_t n = 0, s = 0;
        while (s < len) {                               /* bound: len (s strictly increases) */
            const toks_wp_cell *cell = s == 0u ? t->wcell : t->ccell;
            const int32_t *term = s == 0u ? t->wterm : t->cterm;
            uint32_t node = 0u, nb = cell[0].base;
            int32_t hit = -1;
            uint64_t e = s;
            for (uint64_t i = s; i < len; i++) {        /* bound: len - s */
                uint32_t c = p[i];
                if (fold && c - 0x41u < 26u) { c |= 0x20u; }
                uint32_t x = (nb & TOKS_WP_BASE) + c;   /* < cells + 256: padded */
                if (cell[x].check != (int32_t)node) { break; }
                node = x;
                nb = cell[x].base;
                if (nb & TOKS_WP_TERM) { hit = term[x]; e = i + 1u; }
            }
            *probes += 1u;
            if (hit < 0) {                              /* one unmatched position: the whole piece is [unk] */
                toks_st32(out, t->unk_id);
                return 1;
            }
            toks_st32(out + n++, (uint32_t)hit);
            s = e;
        }
        return n;
    }
    /* W2's first candidate: the whole piece (a certified shortcut, wordpiece.md §12.4) */
    if (len <= t->word_maxlen) {
        int64_t id = wp_find(t, t->word, t->word_mask, wp_hash(p, len, fold), p, len, fold);
        *probes += 1u;
        if (id >= 0) {
            toks_st32(out, (uint32_t)id);
            return 1;
        }
    }
    uint64_t n = 0;
    uint64_t s = 0;
    uint32_t st[WP_SMAX + 1u];
    while (s < len) {                                   /* bound: len (s strictly increases) */
        const toks_wp_entry *tab = s == 0u ? t->word : t->cont;
        uint64_t mask = s == 0u ? t->word_mask : t->cont_mask;
        uint64_t maxlen = s == 0u ? t->word_maxlen : t->cont_maxlen;
        uint64_t hi = len - s < maxlen ? len - s : maxlen;      /* longer candidates cannot match */
        /* prefix states: st[m] = the crc after the first m 8-byte words of p[s, ...) */
        uint64_t nst = hi / 8u < WP_SMAX ? hi / 8u : WP_SMAX;
        st[0] = TOKS_WP_HSEED;
        for (uint64_t m = 0; m < nst; m++) {            /* bound: WP_SMAX */
            uint64_t w = wp_load(p + s + 8u * m, 8);
            st[m + 1u] = toks_crc32c_u64(st[m], fold ? wp_fold(w) : w);
        }
        int64_t hit = -1;
        uint64_t e = s + hi;
        for (; e > s; e--) {                            /* bound: hi candidates */
            if (e < len && (p[e] & 0xC0u) == 0x80u) { continue; }   /* not a char boundary */
            if (s == 0u && e == len) { continue; }       /* the whole piece: probed above */
            uint64_t k = e - s;
            uint32_t h;
            if (k / 8u <= nst) {
                uint64_t m = k / 8u;
                h = st[m];
                if (k % 8u) {
                    uint64_t w = wp_load(p + s + 8u * m, k % 8u);
                    h = toks_crc32c_u64(h, fold ? wp_fold(w) : w);
                }
                h = wp_hash_end(h, k);
            } else {
                h = wp_hash(p + s, k, fold);
            }
            *probes += 1u;
            hit = wp_find(t, tab, mask, h, p + s, k, fold);
            if (hit >= 0) { break; }
        }
        if (hit < 0) {                                  /* one unmatched position: the whole piece is [unk] */
            toks_st32(out, t->unk_id);
            return 1;
        }
        toks_st32(out + n++, (uint32_t)hit);
        s = e;
    }
    return n;
}

/* ---- the answer tables: whole words (t->wtab) and the scratch's cache (wordpiece.md §12.7) ------------------- */

/* a piece's key, p[0, len) (1..15 bytes, read in place from avail) under the lowercase fold: W2 only ever compares
 * fold(p), and byte 15 (len < 16) is no letter */
static inline bpe_key wp_key(const uint8_t *p, uint64_t len, uint64_t avail, int fold)
{
    bpe_key k = bpe_key_at(p, avail, 0u, len);
    if (fold) {
        k.lo = wp_fold(k.lo);
        k.hi = wp_fold(k.hi);
    }
    return k;
}

static inline uint32_t wp_whash(bpe_key k)                 /* one multiply (spm.h toks_spm_whash) */
{
    return (uint32_t)(((k.lo ^ (k.hi * TOKS_FIB64)) * 0xD6E8FEB86659FD93ull) >> 32);
}

#define WP_BATCH 64u                                    /* pieces whose keys are hashed and lines prefetched first */

uint64_t toks_wp_encode_c(const toks_wp_tables *t, toks_wp_encode_args *a)
{
    uint64_t n = 0, hits = 0, misses = 0, probes = 0;
    int fold = (t->flags & TOKS_WPF_LOWER) != 0u;
    bpe_key kq[WP_BATCH];
    uint32_t hq[WP_BATCH];
    for (uint64_t i = 0; i < a->n; i++) {               /* bound: the pieces */
        if (t->wtab != NULL && i % WP_BATCH == 0u) {    /* the next batch's keys and lines (memory-level parallelism) */
            for (uint64_t j = i; j < a->n && j < i + WP_BATCH; j++) {   /* bound: WP_BATCH */
                const toks_wp_piece *pj = &a->pieces[j];
                if (pj->len - 1u >= (uint64_t)TOKS_KEY_MAXLEN || (pj->flags & (TOKS_WPP_OVER | TOKS_WPP_INVALID))) {
                    continue;
                }
                int mj = (pj->flags & TOKS_WPP_MAT) != 0u;
                uint64_t lim = mj ? a->mat_len : a->text_len;   /* 0: unknown, read the piece only */
                kq[j - i] = wp_key((mj ? a->mat : a->text) + pj->off, pj->len,
                                   lim > pj->off + pj->len ? lim - pj->off : pj->len, fold);
                hq[j - i] = wp_whash(kq[j - i]);
                __builtin_prefetch(t->wtab + ((uint64_t)hq[j - i] & t->wtab_mask) * TOKS_BUCKET);
            }
        }
        const toks_wp_piece *pc = &a->pieces[i];
        int mat = (pc->flags & TOKS_WPP_MAT) != 0u;
        const uint8_t *p = (mat ? a->mat : a->text) + pc->off;
        uint64_t len = pc->len;
        if (t->wtab != NULL && len - 1u < (uint64_t)TOKS_KEY_MAXLEN && !(pc->flags & (TOKS_WPP_OVER | TOKS_WPP_INVALID))) {
            bpe_key k = kq[i % WP_BATCH];
            uint32_t h = hq[i % WP_BATCH];
            const uint8_t *v = bpe_words_probe(t->wtab, t->wtab_mask, h, k);
            if (v != NULL) {                            /* a vocab string: W2's first candidate */
                uint32_t id;
                memcpy(&id, v, 4);
                toks_st32(a->out + n, id & TOKS_ID_MASK);   /* out may be the caller's array: any alignment */
                n++;
                hits++;
                probes++;
                continue;
            }
            uint8_t *bucket = a->cache != NULL ? a->cache + ((uint64_t)h & a->cache_mask) * TOKS_BUCKET : NULL;
            if (bucket != NULL && (v = bpe_cache_get(bucket, k, a->tw)) != NULL) {   /* a greedy answer seen before */
                uint32_t w[4];
                memcpy(w, v, 16);
                uint32_t m = w[0] >> TOKS_VAL_COUNT_SHIFT;
                for (uint32_t j = 0; j < m; j++) { toks_st32(a->out + n + j, w[j] & TOKS_ID_MASK); }   /* bound: 4 */
                n += m;
                misses++;
                continue;
            }
            uint64_t m = toks_wp_piece_ids(t, p, len, pc->flags, a->out + n, &probes);
            if (bucket != NULL && m <= 4u) {             /* the scratch's cache: K5's policy (kernels.md §6) */
                uint32_t val[4];
                bpe_val_pack_tag(val, a->out + n, (uint32_t)m, a->tw);
                bpe_cache_fill(bucket, k, val);
            }
            n += m;
            misses++;
            continue;
        }
        uint64_t before = probes;
        uint64_t k = toks_wp_piece_ids(t, p, len, pc->flags, a->out + n, &probes);
        if (probes - before <= 1u && k == 1u) { hits++; } else { misses++; }
        n += k;
    }
    a->n_out = n;
    a->hits = hits;
    a->misses = misses;
    a->probes = probes;
    return n;
}

/* ---- build ------------------------------------------------------------------------------------------------- */

static uint64_t pow2_at_least(uint64_t x)
{
    uint64_t p = 16;
    while (p < x) { p <<= 1; }                          /* bound: 64 */
    return p;
}

/* buckets of the whole-word table: K5's sizing (two buckets of two ways, pow2(n / 2 + 1)) */
static uint64_t wtab_buckets(const toks_wp_vocab *v)
{
    uint64_t n = 0;
    for (uint32_t i = 0; i < v->n; i++) { n += v->len[i] - 1u < (uint32_t)TOKS_KEY_MAXLEN; }   /* bound: the vocab */
    return n != 0u ? bpe_pow2(n / 2u + 1u) : 0u;
}

uint64_t toks_wp_tables_bytes(const toks_wp_vocab *v)
{
    uint64_t keys = 0;
    for (uint32_t i = 0; i < v->n; i++) { keys += v->len[i]; }  /* bound: the vocab */
    uint64_t slots = pow2_at_least(2u * (uint64_t)v->n + 2u);
    return 2u * slots * sizeof(toks_wp_entry) + keys + wtab_buckets(v) * TOKS_BUCKET + 4u * 64u;
}

/* inserts (h, key, id); a key already present takes the new id (json map semantics: the last value wins) */
static void wp_insert(toks_wp_entry *tab, uint64_t mask, const uint8_t *keys, uint32_t h, uint32_t off, uint32_t len,
                      uint32_t id)
{
    uint64_t i = (uint64_t)h & mask;
    for (uint64_t k = 0; k <= mask; k++) {              /* bound: the table (never full: load <= 1/2) */
        toks_wp_entry *e = &tab[i];
        if (e->len == 0u) {
            e->h = h; e->id = id; e->off = off; e->len = len;
            return;
        }
        if (e->h == h && e->len == len && memcmp(keys + e->off, keys + off, len) == 0) {
            e->id = id;
            return;
        }
        i = (i + 1u) & mask;
    }
}

int64_t toks_wp_build(toks_wp_tables *t, toks_arena *ar, const toks_wp_vocab *v, const toks_wp_params *p,
                      toks_err *err)
{
    memset(t, 0, sizeof *t);
    err->code = 0;
    err->what = NULL;
    if (p->max_chars > TOKS_WP_MAX_CHARS) {
        err->code = TOKS_E_UNSUPPORTED;
        err->what = "WordPiece max_input_chars_per_word above 1024";
        return TOKS_E_UNSUPPORTED;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < v->n; i++) {               /* bound: the vocab */
        if (v->len[i] > TOKS_WP_MAX_KEY) {
            err->code = TOKS_E_LIMIT;
            err->what = "WordPiece vocab string over 65535 bytes";
            return TOKS_E_LIMIT;
        }
        total += v->len[i];
    }
    if (total >= (1ull << 32)) {
        err->code = TOKS_E_LIMIT;
        err->what = "WordPiece vocab over 4 GiB";
        return TOKS_E_LIMIT;
    }
    uint8_t *keys = toks_tab_ar(ar, total, 64u, TOKS_X_WP_KEYS);
    (void)toks_ar_alloc(ar, 8u, 1u);                    /* the 8 bytes the arena keeps after the keys (placement) */
    uint64_t slots = pow2_at_least(2u * (uint64_t)v->n + 2u);
    toks_wp_entry *word = toks_tab_ar(ar, slots * sizeof(toks_wp_entry), 64u, TOKS_X_WP_ENTRIES);
    toks_wp_entry *cont = toks_tab_ar(ar, slots * sizeof(toks_wp_entry), 64u, TOKS_X_WP_ENTRIES);
    if (keys == NULL || word == NULL || cont == NULL) {
        err->code = TOKS_E_NOMEM;
        err->what = "WordPiece tables";
        return TOKS_E_NOMEM;
    }
    memset(word, 0, slots * sizeof(toks_wp_entry));
    memset(cont, 0, slots * sizeof(toks_wp_entry));
    uint64_t off = 0;
    uint32_t wmax = 0, cmax = 0;
    int unk_found = 0;
    uint32_t unk_id = 0;
    for (uint32_t i = 0; i < v->n; i++) {               /* bound: the vocab */
        uint32_t n = v->len[i];
        if (n == 0u) { continue; }                      /* "" is never a candidate */
        memcpy(keys + off, v->str[i], n);
        uint32_t ko = (uint32_t)off;
        off += n;
        wp_insert(word, slots - 1u, keys, wp_hash(keys + ko, n, 0), ko, n, v->id[i]);
        if (n > wmax) { wmax = n; }
        if (n == p->unk_len && memcmp(keys + ko, p->unk, n) == 0) {
            unk_found = 1;
            unk_id = v->id[i];                          /* the last duplicate wins, as in the table */
        }
        if (n > p->prefix_len && memcmp(keys + ko, p->prefix, p->prefix_len) == 0) {
            uint32_t cn = n - p->prefix_len;
            wp_insert(cont, slots - 1u, keys, wp_hash(keys + ko + p->prefix_len, cn, 0), ko + p->prefix_len, cn,
                      v->id[i]);
            if (cn > cmax) { cmax = cn; }
        }
    }
    if (!unk_found) {
        err->code = TOKS_E_UNSUPPORTED;
        err->what = "WordPiece unk_token not in vocab";
        return TOKS_E_UNSUPPORTED;
    }
    t->flags = p->flags;
    t->max_chars = p->max_chars;
    t->unk_id = unk_id;
    t->word_maxlen = wmax;
    t->cont_maxlen = cmax;
    t->prefix_len = p->prefix_len;
    for (uint32_t b = 0; b < 128u; b++) {               /* bound: 128 */
        t->ascii_cls[b] = toks_wp_ascii_class(p->flags, b);
        t->ascii_cls0[b] = toks_wp_ascii_class(0u, b);
    }
    t->word = word;
    t->word_mask = slots - 1u;
    t->cont = cont;
    t->cont_mask = slots - 1u;
    t->keys = keys;
    /* the whole-word table: every vocab string of 1..15 bytes (under lowercase none with A-Z: fold(p) never holds
     * one), its id the word table's (the json's last value), each key once; a full pair of buckets leaves it out */
    uint64_t wb = wtab_buckets(v);
    uint8_t *wt = wb != 0u ? toks_tab_ar(ar, wb * TOKS_BUCKET, 64u, TOKS_X_WORDS) : NULL;
    if (wb != 0u && wt == NULL) {
        err->code = TOKS_E_NOMEM;
        err->what = "WordPiece tables";
        return TOKS_E_NOMEM;
    }
    if (wt != NULL) {
        memset(wt, 0, wb * TOKS_BUCKET);
        int fold = (p->flags & TOKS_WPF_LOWER) != 0u;
        for (uint32_t i = 0; i < v->n; i++) {           /* bound: the vocab */
            uint32_t n = v->len[i];
            if (n - 1u >= (uint32_t)TOKS_KEY_MAXLEN) { continue; }
            int upper = 0;
            for (uint32_t j = 0; j < n; j++) { upper |= v->str[i][j] - 0x41u < 26u; }   /* bound: 15 */
            if (fold && upper) { continue; }
            uint8_t kb[16];
            bpe_key_make(kb, v->str[i], n);
            bpe_key k;
            memcpy(&k.lo, kb, 8);
            memcpy(&k.hi, kb + 8, 8);
            uint32_t h = wp_whash(k);
            if (bpe_words_probe(wt, wb - 1u, h, k) != NULL) { continue; }   /* a repeated string: once */
            int64_t id = wp_find(t, word, slots - 1u, wp_hash(v->str[i], n, 0), v->str[i], n, 0);
            uint32_t val[4] = { (1u << TOKS_VAL_COUNT_SHIFT) | ((uint32_t)id & TOKS_ID_MASK), 0u, 0u, 0u };
            (void)bpe_words_put(wt, wb - 1u, h, k, val);
        }
        t->wtab = wt;
        t->wtab_mask = wb - 1u;
    }
    return 0;
}

/* ---- the tries (W2's longest match) --------------------------------------------------------------------------- */

/* the trie of the vocab strings (which 0), or of the prefixed ones without the prefix (1): base | check | term columns
 * (da.h; term = the vocab index, the last of equal strings: json map order); *nk keys */
static int64_t wp_trie(const toks_wp_vocab *v, const toks_wp_params *p, int which, int32_t **arr, uint64_t *cap,
                       uint64_t *da_len)
{
    uint64_t kbytes = 0;
    for (uint32_t i = 0; i < v->n; i++) { kbytes += v->len[i]; }   /* bound: the vocab */
    uint64_t kb_len = (kbytes + 64u + 63u) & ~(uint64_t)63u, keys_len = ((uint64_t)v->n + 1u) * sizeof(ukey);
    uint8_t *kb = (uint8_t *)toks_plat_alloc(kb_len + keys_len);
    if (kb == NULL) { return TOKS_E_NOMEM; }
    ukey *keys = (ukey *)(void *)(kb + kb_len);
    uint64_t nk = 0, off = 0;
    for (uint32_t i = 0; i < v->n; i++) {               /* bound: the vocab */
        uint32_t n = v->len[i], skip = 0;
        if (which == 1) {
            if (n <= p->prefix_len || memcmp(v->str[i], p->prefix, p->prefix_len) != 0) { continue; }
            skip = p->prefix_len;
        }
        if (n - skip == 0u) { continue; }               /* "" is never a candidate */
        memcpy(kb + off, v->str[i] + skip, n - skip);
        keys[nk].off = off;
        keys[nk].len = n - skip;
        keys[nk].id = i;
        nk++;
        off += n - skip;
    }
    const char *why = NULL;
    int64_t r = nk != 0u ? toks_da_build(kb, keys, &nk, arr, cap, da_len, &why) : 0;
    if (nk == 0u) { *arr = NULL; *cap = 0u; *da_len = 0u; }
    toks_plat_free(kb, kb_len + keys_len);
    return r;
}

int64_t toks_wp_tries(toks_wp_tables *t, toks_arena *ar, const toks_wp_vocab *v, const toks_wp_params *p,
                      uint64_t *bytes)
{
    *bytes = 0u;
    int32_t *arr[2] = { NULL, NULL };
    uint64_t cap[2] = { 0u, 0u }, dl[2] = { 0u, 0u };
    int64_t r = 0;
    for (int w = 0; w < 2 && r == 0; w++) { r = wp_trie(v, p, w, &arr[w], &cap[w], &dl[w]); }   /* bound: 2 */
    for (int w = 0; w < 2; w++) {                       /* bound: 2: cells (+ 256 a step can reach) and terms */
        *bytes += (dl[w] + 256u) * sizeof(toks_wp_cell) + dl[w] * 4u + 128u;
    }
    if (r == 0 && ar != NULL && dl[0] != 0u && dl[1] != 0u) {
        toks_wp_cell *cl[2];
        int32_t *tm[2];
        for (int w = 0; w < 2; w++) {                   /* bound: 2 */
            cl[w] = (toks_wp_cell *)toks_tab_ar(ar, (dl[w] + 256u) * sizeof(toks_wp_cell), 64u, TOKS_X_WP_CELLS);
            tm[w] = (int32_t *)toks_tab_ar(ar, dl[w] * 4u, 64u, TOKS_X_WP_TERM);
            if (cl[w] == NULL || tm[w] == NULL) { r = TOKS_E_NOMEM; break; }
            const int32_t *base = arr[w], *check = arr[w] + cap[w], *term = arr[w] + 2u * cap[w];
            for (uint64_t x = 0; x < dl[w] + 256u; x++) {   /* bound: the cells */
                int in = x < dl[w];
                int32_t id = in && term[x] >= 0 ? (int32_t)v->id[term[x]] : -1;
                cl[w][x].base = in ? ((uint32_t)base[x] | (id >= 0 ? TOKS_WP_TERM : 0u)) : 0u;
                cl[w][x].check = in ? check[x] : TOKS_DA_FREE;
                if (in) { tm[w][x] = id; }
            }
        }
        if (r == 0) {
            t->wcell = cl[0];
            t->ccell = cl[1];
            t->wterm = tm[0];
            t->cterm = tm[1];
        }
    }
    for (int w = 0; w < 2; w++) {                       /* bound: 2 */
        if (arr[w] != NULL) { toks_plat_free(arr[w], cap[w] * 12u); }
    }
    return r;
}

/* ---- the driver's hookup (load.c) --------------------------------------------------------------------------- */

int64_t toks_wp_ctx_build(const struct toks_config *cfg, toks_arena *par, uint8_t **mem, uint64_t *mem_len,
                          const toks_wp_tables **out, toks_err *err)
{
    uint32_t *ids = (uint32_t *)toks_ar_alloc(par, 4u * (uint64_t)cfg->n_vocab + 8u, 8u);
    toks_wp_tables *t = (toks_wp_tables *)toks_ar_alloc(par, sizeof(toks_wp_tables), 64u);
    if (ids == NULL || t == NULL) { return toks_fail(err, TOKS_E_NOMEM, "wordpiece vocab ids"); }
    for (uint32_t i = 0; i < cfg->n_vocab; i++) { ids[i] = i; }   /* bound: n_vocab */
    toks_wp_vocab v = { cfg->vocab, cfg->vocab_len, ids, cfg->n_vocab };
    toks_wp_params p = { cfg->wp_flags, cfg->wp_max_chars, cfg->wp_unk, cfg->wp_unk_len, cfg->wp_prefix,
                         cfg->wp_prefix_len };
    uint64_t tb = 0u;
    int64_t tr = toks_wp_tries(NULL, NULL, &v, &p, &tb);  /* their size first (a load-time build, then again) */
    if (tr != 0) { tb = 0u; }
    uint64_t nb = toks_wp_tables_bytes(&v) + tb + sizeof(toks_wp_tables) + 128u;
    uint8_t *m = toks_plat_arena(nb);
    if (m == NULL) { return toks_fail(err, TOKS_E_NOMEM, "wordpiece tables"); }
    *mem = m;
    *mem_len = nb;
    toks_arena ar = { m, nb, 0 };
    toks_wp_tables *dst = (toks_wp_tables *)toks_tab_ar(&ar, sizeof(toks_wp_tables), 64u, TOKS_X_WP);
    if (dst == NULL) { return toks_fail(err, TOKS_E_NOMEM, "wordpiece tables"); }
    int64_t r = toks_wp_build(t, &ar, &v, &p, err);
    if (r != 0) { return r; }
    if (tb != 0u && toks_wp_tries(t, &ar, &v, &p, &tb) != 0) {   /* no tries: the candidates' hash probes */
        t->wcell = t->ccell = NULL;
        t->wterm = t->cterm = NULL;
    }
    memcpy(dst, t, sizeof *t);
    toks_tab_seal(m, nb);
    *out = dst;
    return 0;
}
