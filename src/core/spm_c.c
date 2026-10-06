/* spm_c.c: sentencepiece-style bpe over the folded tables (docs/notes/c-core.md §spm_c.c.1) */
#include "spm.h"
#include "bpe.h"
#include "kernels.h"

#define DEAD     0xFFFFFFFFu
#define NOENTRY  (TOKS_SPM_E_NOID | (TOKS_SPM_E_SI_NONE << TOKS_SPM_E_SI_SHIFT))

/* the entry of the char at p[0, avail) and its byte length (an ill-formed byte: 1, no entry) */
static inline uint32_t entry_at(const toks_spm *s, const uint8_t *p, uint64_t avail, uint32_t *k)
{
    uint8_t b = p[0];
    if (b < 0x80u) { *k = 1u; return s->ascii[b]; }
    uint32_t n = toks_utf8_len(p, avail);
    if (n == 0u) { *k = 1u; return NOENTRY; }
    *k = n;
    uint32_t cp = toks_cp_decode(p, n);
    return s->stage2[(uint32_t)s->stage1[cp >> 8] * 256u + (cp & 0xFFu)];
}

/* the ascii byte pairs p[j] p[j + 1], j = 0..7, through the byte-pair table (spm.h cut_ab): bit j = a word
 * starts at p[j + 1]. Eight independent bit tests, no branch. */
static inline uint64_t cut8(const uint32_t *ab, const uint8_t *p)
{
    uint64_t m = 0;
    for (uint32_t j = 0; j < 8u; j++) {                     /* bound: 8 */
        uint32_t x = (uint32_t)p[j] | ((uint32_t)p[j + 1u] << 8);
        m |= (uint64_t)((ab[x >> 5] >> (x & 31u)) & 1u) << j;
    }
    return m;
}

/* the split planner's question (spm.h; why: docs/algorithms/spm_bpe.md §5.6); a: the start of the char ending at c */
int toks_spm_cut(const toks_spm *s, const uint8_t *text, uint64_t len, uint64_t a, uint64_t c)
{
    uint32_t k;
    uint32_t ea = entry_at(s, text + a, len - a, &k), eb = entry_at(s, text + c, len - c, &k);
    int repl = (eb & TOKS_SPM_E_ID) == s->id_repl && s->id_repl != TOKS_SPM_NONE;
    if (s->pfx_mode == TOKS_SPM_PFX_GAP || (s->pfx_mode == TOKS_SPM_PFX_ALWAYS && !repl)) { return 0; }
    if (s->ms_split && repl) { return 1; }
    return s->pairs != NULL && toks_spm_cut_between(s, ea, eb);
}

/* §6.3: does this unit get the virtual prefix symbol? (first: the first char's entry) */
static inline int has_prefix(const toks_spm *s, uint32_t first, int at_start)
{
    int m = (int)s->pfx_mode, repl = (first & TOKS_SPM_E_ID) == s->id_repl;
    return m == TOKS_SPM_PFX_GAP || (!repl && (m == TOKS_SPM_PFX_ALWAYS || (m == TOKS_SPM_PFX_FIRST && at_start)));
}

/* ---- §5.2 symbols --------------------------------------------------------------------------------------- */

/* the initial ids of [prefix] p[0, len) into sym (room len + 1); returns the count */
static uint64_t symbols(const toks_tables *t, const toks_spm *s, const uint8_t *p, uint64_t len, int wp, uint32_t *sym)
{
    uint64_t n = 0, i = 0;
    uint32_t pending = TOKS_SPM_NONE;
    if (wp) { sym[n++] = s->pfx_entry & TOKS_SPM_E_ID; }
    while (i < len) {                                       /* bound: len (i advances >= 1) */
        uint32_t k, w[3];
        if (t->apm != NULL && i + 1u < len && (p[i] | p[i + 1u]) < 0x80u && (i > 0u || !wp)) {   /* §5.7 */
            memcpy(w, t->apm + TOKS_APM_PAIRS + 16u * ((uint32_t)p[i] << 7 | p[i + 1u]), 12);
            uint32_t bf = i > 0u ? 1u << t->apm[p[i - 1u]] : 0u, af = i + 2u < len ? 1u << t->apm[p[i + 2u]] : 0u;
            if (w[0] != UINT32_MAX && ((w[1] & af) | (w[2] & bf)) == 0u) {
                if (pending != TOKS_SPM_NONE) { sym[n++] = pending; pending = TOKS_SPM_NONE; }
                sym[n++] = w[0];
                i += 2u;
                continue;
            }
        }
        uint32_t id = entry_at(s, p + i, len - i, &k) & TOKS_SPM_E_ID;
        if (id != TOKS_SPM_E_NOID) {                        /* a: a vocab char (the image's) */
            if (pending != TOKS_SPM_NONE) { sym[n++] = pending; pending = TOKS_SPM_NONE; }
            sym[n++] = id;
            i += k;
            continue;
        }
        /* not a vocab char: its image is itself (substituted chars always have ids, spm_build.c) */
        if (s->sflags & TOKS_SPM_BYTE_FALLBACK) {           /* b: every byte has <0xHH>, all or nothing */
            uint32_t all = 1;
            for (uint32_t j = 0; j < k; j++) {              /* bound: 4 */
                if (t->byte2id[p[i + j]] == TOKS_SPM_NONE) { all = 0; }
            }
            if (all) {
                if (pending != TOKS_SPM_NONE) { sym[n++] = pending; pending = TOKS_SPM_NONE; }
                for (uint32_t j = 0; j < k; j++) { sym[n++] = t->byte2id[p[i + j]]; }   /* bound: 4 */
                i += k;
                continue;
            }
        }
        if (s->unk_id != TOKS_SPM_NONE) {                   /* c: unk (fused when the last was unk too) */
            if (pending == TOKS_SPM_NONE || !(s->sflags & TOKS_SPM_FUSE_UNK)) {
                if (pending != TOKS_SPM_NONE) { sym[n++] = pending; }
                pending = s->unk_id;
            }
        }
        i += k;                                             /* d: no unk token: the char vanishes */
    }
    if (pending != TOKS_SPM_NONE) { sym[n++] = pending; }
    return n;
}

/* ---- §5.3 the merge loop: K6's (k6_c.c toks_k6_merge, lowest rank then leftmost) on the word's symbols ---- */

/* the model on one word: [prefix] p[0, len) -> ids at work (8-aligned, TOKS_SPM_WORK_BYTES(len + 1)); returns the
 * count. The symbols go where K6's merge keeps its ids (work + 16 (len + 1)); the result over its dead heap. */
static uint64_t model(const toks_tables *t, const toks_spm *s, const uint8_t *p, uint64_t len, int wp, uint8_t *work)
{
    uint32_t id, *ids = (uint32_t *)(void *)work;
    if ((t->flags & TOKS_TF_IGNORE_MERGES) && !wp && len != 0u && bpe_vhash_find(t, p, len, &id)) {   /* §5.1 */
        ids[0] = id;
        return 1;
    }
#if TOKS_HAVE_K6_NEON || TOKS_HAVE_K6_AVX2
    if ((t->flags & TOKS_TF_ASM_MERGE) != 0u && len < 128u) {   /* n <= 128 symbols: K6's asm merge loop */
        _Alignas(64) uint32_t sym[192], kw[(TOKS_BPE_WORK_BYTES(128) + 3u) / 4u];   /* sym: 6 n bytes */
        uint64_t n = symbols(t, s, p, len, wp, sym);
        if (n < 2u) { ids[0] = sym[0]; return n; }
        toks_k6_args a = { (const uint8_t *)(void *)sym, n, ids, (uint8_t *)(void *)kw, sizeof kw, 0u, 0u, 0u };
#if TOKS_HAVE_K6_NEON
        return toks_k6_merge_neon(t, &a);
#else
        return toks_k6_merge_avx2(t, &a);
#endif
    }
#endif
    uint64_t n = symbols(t, s, p, len, wp, (uint32_t *)(void *)(work + 16u * (len + 1u)));
    return n == 0u ? 0u : toks_k6_merge(t, work, len + 1u, n, ids);
}

uint64_t toks_spm_model(const toks_tables *t, const toks_spm *s, const uint8_t *p, uint64_t len, uint32_t *out,
                        uint8_t *work)
{
    uint8_t *a = work + ((8u - ((uintptr_t)work & 7u)) & 7u);
    uint64_t m = model(t, s, p, len, 0, a);
    memcpy(out, a, (size_t)(m * 4u));
    return m;
}

/* ---- one word through the cache ------------------------------------------------------------------------- */

/* the word cache (spm_bpe.md §6.6; docs/notes/c-core.md §spm_c.c.2) */
#define SPM_WIDE_LEN   30u
#define SPM_WIDE_IDS   7u                                   /* the val's 3-bit count */
#define SPM_LOW56      UINT64_C(0x00FFFFFFFFFFFFFF)

static inline uint64_t spm_bucket(bpe_key k, uint64_t cm) { return toks_spm_whash(k.lo, k.hi) & cm; }

typedef struct wkey {
    uint64_t q[4];
} wkey;

/* W of a word of 1..30 bytes at p (bytes run to end; sk: its short key when len <= 15) */
static wkey wide_key(const uint8_t *p, uint64_t len, const uint8_t *end, int wp, bpe_key sk)
{
    wkey w;
    uint64_t mark = (uint64_t)(0x40u | (wp ? 0x20u : 0u)) << 56, tail = 0u;
    w.q[2] = 0u;
    if (len > (uint64_t)TOKS_KEY_MAXLEN) {                  /* bytes 0..14, then 15..len - 1 */
        uint64_t avail = (uint64_t)(end - p);
        bpe_key b = bpe_key_at(p + 15, avail - 15u, 0u, len - 15u);
        sk = bpe_key_at(p, avail, 0u, 15u);
        w.q[2] = b.lo;
        tail = b.hi & SPM_LOW56;
    }
    w.q[0] = sk.lo;
    w.q[1] = (sk.hi & SPM_LOW56) | mark;
    w.q[3] = tail | ((uint64_t)(0x40u | len) << 56);
    return w;
}

static inline uint64_t wide_bucket(const wkey *w, uint64_t cm)
{
    uint64_t x = (w->q[0] ^ (w->q[1] * TOKS_FIB64)) * 0xD6E8FEB86659FD93ull;
    uint64_t y = (w->q[2] ^ (w->q[3] * TOKS_FIB64)) * 0x9E3779B97F4A7C15ull;
    return ((x ^ y) >> 32) & cm;
}

static inline uint64_t put_ids(uint32_t *out, uint64_t cap, uint64_t n, const uint32_t *ids, uint64_t m)
{
    for (uint64_t j = 0; j < m; j++) {                      /* bound: m */
        if (n + j < cap) { toks_st32(out + n + j, ids[j]); }   /* out: a caller array (core.h) */
    }
    return n + m;
}

/* a wide entry's ids (vals at v: 32 bytes) to out[n..): two 16-byte stores when the room rule allows (toks_encode
 * may overwrite up to cap), else per id; returns the new count */
static inline uint64_t put_wide(const uint8_t *v, uint32_t *out, uint64_t cap, uint64_t n)
{
    uint32_t val[8];
    memcpy(val, v, 32);
    uint32_t m = bpe_val_count(val);
    for (uint32_t j = 0; j < 8u; j++) { val[j] &= TOKS_ID_MASK; }   /* bound: 8 */
    if (n + 8u <= cap) {
        toks_cpy(out + n, val, 32u);                        /* out: a caller array, any alignment (core.h) */
        return n + m;
    }
    for (uint32_t j = 0; j < m; j++) {                      /* bound: 7 */
        if (n + j < cap) { toks_st32(out + n + j, val[j]); }
    }
    return n + m;
}

/* a word the inline probe did not answer (no short entry: a wide entry, the model, or no cache): out of line, so
 * the scan's flush keeps only the short probe. bucket: the word's short bucket (NULL: no cache or len > 15). */
static __attribute__((noinline)) uint64_t word_slow(const toks_tables *t, const toks_spm *s, const uint8_t *p, uint64_t len,
                                                    const uint8_t *end, int wp, uint32_t *out, uint64_t cap, uint64_t n,
                                                    uint8_t *cache, uint64_t cm, uint8_t *bucket, bpe_key k, uint64_t tw,
                                                    uint8_t *work)
{
    wkey W;
    uint8_t *wb = NULL;
    if (cache != NULL && len <= SPM_WIDE_LEN && len != 0u) {
        W = wide_key(p, len, end, wp, k);
        wb = bucket != NULL ? bucket : cache + wide_bucket(&W, cm) * TOKS_BUCKET;
        uint64_t q[4], g;
        memcpy(q, wb, 32);
        memcpy(&g, wb + 40, 8);
        if (q[0] == W.q[0] && q[1] == W.q[1] && q[2] == W.q[2] && q[3] == W.q[3] && (g & TOKS_TAG_MASK64) == tw) {
            return put_wide(wb + 32, out, cap, n);
        }
    }
    if (bucket != NULL && t->words != NULL) {                /* the static word table (spm_bpe.md §6.6) */
        const uint8_t *v = bpe_words_probe(t->words, t->words_mask, toks_spm_whash(k.lo, k.hi), k);
        if (v != NULL) {
            uint32_t val[4], sid[4];
            memcpy(val, v, 16);
            for (uint32_t j = 0; j < 4u; j++) { sid[j] = val[j] & TOKS_ID_MASK; }   /* bound: 4 */
            bpe_val_pack_tag(val, sid, bpe_val_count(val), tw);
            bpe_cache_fill(bucket, k, val);                 /* K5's warm rule: a static answer fills the cache */
            /* bit 63: a static answer (a line the window batch fetches early); n < 2^63 */
            return put_ids(out, cap, n, sid, bpe_val_count(val)) | (UINT64_C(1) << 63);
        }
    }
    uint64_t m = model(t, s, p, len, wp, work);
    const uint32_t *ids = (const uint32_t *)(const void *)work;
    if (bucket != NULL && m >= 1u && m <= 4u) {             /* short */
        uint32_t val[4];
        bpe_val_pack_tag(val, ids, (uint32_t)m, tw);
        bpe_cache_fill(bucket, k, val);                     /* K5's policy */
    } else if (wb != NULL && m >= 1u && m <= SPM_WIDE_IDS) {   /* wide: the whole bucket */
        uint32_t val[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        for (uint64_t j = 0; j < m; j++) { val[j] = ids[j] & TOKS_ID_MASK; }   /* bound: 7 */
        val[0] |= (uint32_t)m << TOKS_VAL_COUNT_SHIFT;
        val[2] |= (uint32_t)tw;
        val[3] |= (uint32_t)(tw >> 32);
        memcpy(wb, W.q, 32);
        memcpy(wb + 32, val, 32);
    }
    return put_ids(out, cap, n, ids, m);
}

/* the word p[0, len) of a unit whose bytes run to end (the key is read in place, never past end); tw: the scratch
 * epoch's tag word (layout.h): an entry of another epoch is a miss, a fill writes it (kernels.md §6-7) */
/* a short word's key: its bytes read in place (never past end), len in byte 15, bit 7 there the prefix */
static inline bpe_key skey(const uint8_t *p, uint64_t len, const uint8_t *end, int wp)
{
    bpe_key k = { 0u, 0u };
    if (len != 0u) { k = bpe_key_at(p, (uint64_t)(end - p), 0u, len); }
    if (wp) { k.hi |= UINT64_C(0x80) << 56; }
    return k;
}

/* the word p[0, len) of a unit whose bytes run to end, k its key when len <= 15 (read in place, never past end); tw:
 * the scratch epoch's tag word (layout.h): an entry of another epoch is a miss, a fill writes it (kernels.md §6-7).
 * *miss counts the words the static word table answered (toks_spm_encode batches the next window) */
static inline __attribute__((always_inline)) uint64_t word_k(const toks_tables *t, const toks_spm *s, const uint8_t *p,
                                                             uint64_t len, const uint8_t *end, int wp, uint32_t *out,
                                                             uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cm,
                                                             uint64_t tw, uint8_t *work, uint32_t *miss, bpe_key k)
{
    uint8_t *bucket = NULL;
    if (cache != NULL && len <= (uint64_t)TOKS_KEY_MAXLEN) {
        bucket = cache + spm_bucket(k, cm) * TOKS_BUCKET;
        const uint8_t *v = bpe_cache_get(bucket, k, tw);
        if (v != NULL) {
            if (n + 4u <= cap) { return n + bpe_val_put(v, out + n); }   /* 4 ids in one store, count added */
            uint32_t val[4];
            memcpy(val, v, 16);
            uint32_t m = bpe_val_count(val);                /* <= 4: a short entry */
            for (uint32_t j = 0; j < m; j++) {              /* bound: 4 */
                if (n + j < cap) { toks_st32(out + n + j, val[j] & TOKS_ID_MASK); }
            }
            return n + m;
        }
    }
    uint64_t r = word_slow(t, s, p, len, end, wp, out, cap, n, cache, cm, bucket, k, tw, work);
    *miss += (uint32_t)(r >> 63);
    return r & ~(UINT64_C(1) << 63);
}

static inline __attribute__((always_inline)) uint64_t word(const toks_tables *t, const toks_spm *s, const uint8_t *p,
                                                           uint64_t len, const uint8_t *end, int wp, uint32_t *out,
                                                           uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cm,
                                                           uint64_t tw, uint8_t *work, uint32_t *miss)
{
    bpe_key k = { 0u, 0u };
    if (cache != NULL && len <= (uint64_t)TOKS_KEY_MAXLEN) { k = skey(p, len, end, wp); }
    return word_k(t, s, p, len, end, wp, out, cap, n, cache, cm, tw, work, miss, k);
}

/* a window's words (the set bits of m end them; the first starts at *wsp) batched: their keys and both lines (the
 * word cache's, the static word table's) asked for first, then each word in order, so the misses overlap
 * (spm_bpe.md 6.6). Out of line: toks_spm_encode enters it only for a window after one with a word-cache miss. */
static __attribute__((noinline)) uint64_t window_batch(const toks_tables *t, const toks_spm *s, const uint8_t *text,
                                                       uint64_t len, uint64_t base, uint64_t m, uint64_t x,
                                                       const uint32_t *one, uint64_t *wsp, int *wpp, uint32_t *out,
                                                       uint64_t cap, uint64_t n, uint8_t *c, uint64_t cm, uint64_t tw,
                                                       uint8_t *w, uint32_t *miss)
{
    uint64_t ws = *wsp;
    int wp = *wpp;
    uint32_t ms = 0;
    while (m != 0u) {                                       /* bound: 2 rounds of <= 32 words (64 set bits) */
        uint64_t wst[32];
        bpe_key wk[32];
        uint32_t nw = 0, one_w = 0;                         /* one_w: bit j = word j is one vocab char (its id) */
        while (m != 0u && nw < 32u) {                       /* bound: 32 */
            uint64_t at = base + (uint64_t)__builtin_ctzll(m);
            m &= m - 1u;
            uint64_t wl = at - ws;
            wst[nw] = ws;
            /* a word of one 2-4 byte vocab char is its id (docs/notes/c-core.md §spm_c.c.3) */
            uint32_t o = (x != 0u && text[ws] >= 0xC0u && ws >= base) ? one[ws - base] : 0u;
            if ((o >> 24) == wl && !(wp && nw == 0u) && (o & TOKS_SPM_E_ID) != TOKS_SPM_E_NOID) {
                one_w |= 1u << nw;
                wk[nw].lo = o & TOKS_SPM_E_ID;
            } else if (wl <= (uint64_t)TOKS_KEY_MAXLEN) {
                wk[nw] = skey(text + ws, wl, text + len, wp && nw == 0u);
                uint32_t h = toks_spm_whash(wk[nw].lo, wk[nw].hi);
                __builtin_prefetch(c + ((uint64_t)h & cm) * TOKS_BUCKET);
                if (t->words != NULL) { __builtin_prefetch(t->words + ((uint64_t)h & t->words_mask) * TOKS_BUCKET); }
            }
            nw++;
            ws = at;
        }
        for (uint32_t j = 0; j < nw; j++) {                 /* bound: 32 */
            uint64_t a0 = wst[j], e = j + 1u < nw ? wst[j + 1u] : ws;
            if ((one_w >> j) & 1u) {
                if (n < cap) { toks_st32(out + n, (uint32_t)wk[j].lo); }   /* out: a caller array (core.h) */
                n++;
            } else {
                n = word_k(t, s, text + a0, e - a0, text + len, wp, out, cap, n, c, cm, tw, w, &ms, wk[j]);
            }
            wp = 0;
        }
    }
    *wsp = ws;
    *wpp = wp;
    *miss = ms;
    return n;
}

/* ---- a text unit ---------------------------------------------------------------------------------------- */

/* K7's c twin (spm_bpe.md §5.5 "the compiled rule", kernels.md §8): a window's word starts as a bit mask (8 ASCII
 * bytes per step through cut_ab, any other char by the per-char rule); the unit's first char gets one too */
__attribute__((always_inline)) uint64_t toks_k7_spm_c(const toks_spm *s, toks_k7_args *a)
{
    const uint8_t *text = a->text;
    uint64_t len = a->len, i = a->pos, base = i, m = 0, cuts = a->flags & 1u;
    uint32_t prev = (uint32_t)a->prev, mb = 0, k, *one = a->one;
    while (i < len && i - base <= 56u) {                    /* bound: 57 steps (i advances >= 1) */
        if (cuts && i != 0u && len - i >= 8u && text[i - 1u] < 0x80u) {
            uint64_t v;
            memcpy(&v, text + i, 8);
            if ((v & UINT64_C(0x8080808080808080)) == 0u) {   /* text[i - 1, i + 8) all ASCII */
                m |= cut8(s->cut_ab, text + i - 1u) << (i - base);
                i += 8u;
                continue;
            }
        }
        if (i != 0u && text[i - 1u] < 0x80u) { prev = s->ascii[text[i - 1u]]; }   /* that byte is the char */
        uint32_t e = entry_at(s, text + i, len - i, &k);
        one[i - base] = (e & TOKS_SPM_E_ID) | (k << 24);
        mb |= k >> 1;
        int cut = (s->ms_split && (e & TOKS_SPM_E_ID) == s->id_repl) || (cuts && toks_spm_cut_between(s, prev, e));
        m |= (uint64_t)cut << (i - base);
        prev = e;
        i += k;
    }
    a->pos = i;
    a->prev = prev;
    a->mb = mb;
    return m;
}

#if TOKS_HAVE_K7_SPM_NEON
#define K7_PART(s, a) ((flags >> 8) == TOKS_TIER_NEON ? toks_k7_spm_neon(s, a) : 0u)
#elif TOKS_HAVE_K7_SPM_AVX2
#define K7_PART(s, a) ((flags >> 8) - TOKS_TIER_AVX2 <= 1u ? toks_k7_spm_avx2(s, a) : 0u)
#else
#define K7_PART(s, a) 0u
#endif

/* the scan (K7) collects a window's word starts first, then one word per set bit */
uint64_t toks_spm_encode(const toks_tables *t, const toks_spm *s, const uint8_t *text, uint64_t len, int at_start,
                         uint32_t *out, uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cm, uint64_t tag,
                         uint8_t *work, uint32_t flags)
{
    if (len == 0u) { return n; }
    uint8_t *w = work + ((8u - ((uintptr_t)work & 7u)) & 7u);
    uint8_t *c = (flags & TOKS_SPM_NOCACHE) ? NULL : cache;
    uint64_t tw = toks_tag_word(tag);
    uint32_t k;
    int pre = has_prefix(s, entry_at(s, text, len, &k), at_start) && !(flags & TOKS_SPM_NOPFX);
    uint64_t ws = 0;
    int wp = pre;
    uint32_t one[64];                                       /* per-char steps: id | length << 24 at i - base */
    uint32_t miss = 0;                                      /* the window's words the static word table answered */
    toks_k7_args ka = { text, len, 0u, pre ? s->pfx_entry : NOENTRY, one, 0u, s->pairs != NULL && !(flags & TOKS_SPM_NOCUTS),
                        0u };
    while (ka.pos < len) {                                  /* bound: len / 57 + 1 windows (each but the last >= 57 bytes) */
        uint64_t base = ka.pos, m = K7_PART(s, &ka), x;
        if (ka.pos == base) { m = toks_k7_spm_c(s, &ka); }  /* the scalar tier, or a window the asm part leaves */
        if (base == 0u && !pre) { m &= ~UINT64_C(1); }      /* the unit's first char starts its first word */
        x = ka.mb;
        if (miss >= 3u && c != NULL) {                      /* the last window had static-table answers: batched */
            n = window_batch(t, s, text, len, base, m, x, one, &ws, &wp, out, cap, n, c, cm, tw, w, &miss);
            continue;
        }
        miss = 0u;
        while (m != 0u && (x == 0u || c == NULL)) {         /* bound: 64 (one word per set bit) */
            uint64_t at = base + (uint64_t)__builtin_ctzll(m);
            m &= m - 1u;
            n = word(t, s, text + ws, at - ws, text + len, wp, out, cap, n, c, cm, tw, w, &miss);
            ws = at;
            wp = 0;
        }
        while (m != 0u) {                                   /* bound: 64 (one word per set bit) */
            uint64_t at = base + (uint64_t)__builtin_ctzll(m);
            m &= m - 1u;
            /* a word of one 2-4 byte vocab char is its id (docs/notes/c-core.md §spm_c.c.3) */
            uint32_t o = (text[ws] >= 0xC0u && ws >= base) ? one[ws - base] : 0u;
            if ((o >> 24) == at - ws && !wp && (o & TOKS_SPM_E_ID) != TOKS_SPM_E_NOID) {
                if (n < cap) { toks_st32(out + n, o & TOKS_SPM_E_ID); }   /* out: a caller array (core.h) */
                n++;
            } else {
                n = word(t, s, text + ws, at - ws, text + len, wp, out, cap, n, c, cm, tw, w, &miss);
            }
            ws = at;
            wp = 0;
        }
    }
    return word(t, s, text + ws, len - ws, text + len, wp, out, cap, n, c, cm, tw, w, &miss);
}

uint64_t toks_spm_pieces(const toks_spm *s, const uint8_t *text, uint64_t len, int at_start, uint64_t base,
                         uint32_t *out, uint64_t cap, uint64_t n)
{
    if (len == 0u) { return n; }
    if (s->ms_split) {
        uint32_t k;
        uint32_t e = entry_at(s, text, len, &k);
        int pre = has_prefix(s, e, at_start);
        uint64_t i = 0;
        while (i < len) {                                   /* bound: len (i advances >= 1) */
            if (i != 0u) { e = entry_at(s, text + i, len - i, &k); }
            if ((e & TOKS_SPM_E_ID) == s->id_repl && (i != 0u || pre)) {
                if (n < cap) { toks_st32(out + n, (uint32_t)(base + i)); }
                n++;
            }
            i += k;
        }
    }
    if (n < cap) { toks_st32(out + n, (uint32_t)(base + len)); }
    return n + 1u;
}

/* ---- §8 decode, streaming ------------------------------------------------------------------------------- */

typedef struct dec {
    uint8_t     *out;
    uint64_t     cap, total;
    uint32_t     strip_left;            /* Strip: leading content chars still to drop */
    toks_spm_str strip;                 /* Strip's content char */
    int          raw;                   /* TOKS_DECODE_RAW: a run's bytes where decode's U+FFFD would stand */
} dec;

static const uint8_t FFFD[3] = { 0xEFu, 0xBFu, 0xBDu };

static void put_raw(dec *d, const uint8_t *p, uint64_t k)
{
    if (k == 0u) { return; }
    if (d->total < d->cap) {
        uint64_t room = d->cap - d->total;
        memcpy(d->out + d->total, p, (size_t)(room < k ? room : k));
    }
    d->total += k;
}

/* p[0, k) holds whole chars: Strip drops leading content chars first */
static void put(dec *d, const uint8_t *p, uint64_t k)
{
    while (d->strip_left != 0u && k != 0u) {               /* bound: Strip start */
        if (k >= d->strip.n && memcmp(p, d->strip.b, d->strip.n) == 0) {
            p += d->strip.n;
            k -= d->strip.n;
            d->strip_left--;
        } else {
            d->strip_left = 0;
        }
    }
    put_raw(d, p, k);
}

/* one byte of a run that is not utf-8: decode's U+FFFD, or under TOKS_DECODE_RAW the byte where that U+FFFD stands
 * (a Strip that would drop the U+FFFD drops it, else the strip ends there as at the U+FFFD) */
static void put_bad(dec *d, uint8_t b)
{
    if (!d->raw) { put(d, FFFD, 3u); return; }
    if (d->strip_left != 0u) {
        if (d->strip.n == 3u && memcmp(d->strip.b, FFFD, 3) == 0) { d->strip_left--; return; }
        d->strip_left = 0u;
    }
    put_raw(d, &b, 1u);
}

/* a token's string through the per-token step (Replace a -> b, or Metaspace: repl -> " ", dropped in
 * the first string unless the scheme is never) */
static void put_token(dec *d, const toks_spm_op *op, const uint8_t *p, uint64_t k, uint64_t index)
{
    if (op == NULL) { put(d, p, k); return; }
    static const uint8_t SP = ' ';
    const uint8_t *to = op->b.b;
    uint64_t to_n = op->b.n;
    if (op->kind == TOKS_SPM_D_METASPACE) {
        to = &SP;
        to_n = (index == 0u && op->scheme != TOKS_SPM_PS_NEVER) ? 0u : 1u;
    }
    uint64_t i = 0, from = 0;
    while (i + op->a.n <= k) {                              /* bound: k (i advances >= 1) */
        if (p[i] == op->a.b[0] && memcmp(p + i, op->a.b, op->a.n) == 0) {
            put(d, p + from, i - from);
            put(d, to, to_n);
            i += op->a.n;
            from = i;
        } else {
            i++;
        }
    }
    put(d, p + from, k - from);
}

/* hf ByteFallback's token test: "<0x" h h ">", h h = u8::from_str_radix(.., 16) (which takes a '+') */
static inline int hexd(uint8_t c)
{
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

static inline int byte_token(const uint8_t *p, uint64_t k)
{
    if (k != 6u || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>') { return -1; }
    if (p[3] == '+') { return hexd(p[4]); }
    int hi = hexd(p[3]), lo = hexd(p[4]);
    return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
}

static inline int kept(const toks_spm *s, const uint32_t *special, int skip, uint32_t id)
{
    if (skip && special != NULL && ((special[id >> 5] >> (id & 31u)) & 1u)) { return 0; }
    if (s->holes != NULL && ((s->holes[id >> 5] >> (id & 31u)) & 1u)) { return 0; }
    return 1;
}

/* the byte run ids[from, to) (kept ids only): one string when its bytes are utf-8, else U+FFFD each (put_bad) */
static void flush_run(dec *d, const toks_tables *t, const toks_spm *s, const uint32_t *special, int skip,
                      const uint32_t *ids, uint64_t from, uint64_t to, uint64_t nbytes, int valid)
{
    if (!valid && !d->raw) {
        for (uint64_t j = 0; j < nbytes; j++) { put(d, FFFD, 3u); }   /* bound: the run */
        return;
    }
    uint8_t ch[4];
    uint32_t have = 0, need = 0;
    for (uint64_t j = from; j < to; j++) {                  /* bound: to - from */
        uint32_t id = toks_ld32(ids + j);                   /* ids: a caller array (core.h) */
        if (!kept(s, special, skip, id)) { continue; }
        uint32_t o = t->tok_off[id];
        uint8_t b = (uint8_t)byte_token(t->tok_bytes + o, t->tok_off[id + 1u] - o);
        if (!valid) { put_bad(d, b); continue; }            /* TOKS_DECODE_RAW: the run's bytes as they are */
        if (have == 0u) { need = (b < 0x80u) ? 1u : (b < 0xE0u) ? 2u : (b < 0xF0u) ? 3u : 4u; }
        ch[have++] = b;
        if (have == need) { put(d, ch, have); have = 0; }   /* whole chars (the run is valid utf-8) */
    }
}

int64_t toks_spm_decode(const toks_tables *t, const toks_spm *s, const uint32_t *special, uint32_t flags,
                        const uint32_t *ids, uint64_t n, uint8_t *out, uint64_t cap)
{
    dec d;
    memset(&d, 0, sizeof d);
    d.out = out;
    d.cap = cap;
    d.raw = (flags & TOKS_DECODE_RAW) != 0u;
    int skip = (flags & TOKS_SKIP_SPECIAL) != 0u;
    const toks_spm_op *per_token = NULL;
    int bf = 0;
    for (uint32_t i = 0; i < s->n_dec; i++) {               /* bound: TOKS_SPM_MAX_OPS (order: spm_config.c) */
        const toks_spm_op *op = &s->dec[i];
        if (op->kind == TOKS_SPM_D_REPLACE || op->kind == TOKS_SPM_D_METASPACE) { per_token = op; }
        else if (op->kind == TOKS_SPM_D_BYTE_FALLBACK) { bf = 1; }
        else if (op->kind == TOKS_SPM_D_STRIP) { d.strip_left = op->start; d.strip = op->a; }
    }
    uint64_t index = 0;                                     /* kept strings so far */
    uint64_t run_from = 0, run_n = 0;
    uint32_t need = 0, got = 0;                             /* the run's utf-8 state */
    uint8_t lo = 0x80u, hi = 0xBFu;
    int valid = 1;
    for (uint64_t i = 0; i < n; i++) {                      /* bound: n */
        uint32_t id = toks_ld32(ids + i);                   /* ids: a caller array (core.h) */
        if (!kept(s, special, skip, id)) { continue; }
        uint32_t o = t->tok_off[id];
        const uint8_t *p = t->tok_bytes + o;
        uint64_t k = t->tok_off[id + 1u] - o;
        if (!s->has_decoder) {                              /* hf joins the strings with " " */
            if (index != 0u) { put_raw(&d, (const uint8_t *)" ", 1u); }
            put_raw(&d, p, k);
            index++;
            continue;
        }
        int b = bf ? byte_token(p, k) : -1;
        if (b >= 0) {                                       /* a byte of the current run */
            if (run_n == 0u) { run_from = i; valid = 1; need = 0; got = 0; }
            run_n++;
            index++;
            if (!valid) { continue; }
            uint8_t c = (uint8_t)b;
            if (need == 0u) {
                if (c < 0x80u) { continue; }
                if (c < 0xC2u || c > 0xF4u) { valid = 0; continue; }
                need = (c < 0xE0u) ? 1u : (c < 0xF0u) ? 2u : 3u;
                got = 0;
                lo = (c == 0xE0u) ? 0xA0u : (c == 0xF0u) ? 0x90u : 0x80u;
                hi = (c == 0xEDu) ? 0x9Fu : (c == 0xF4u) ? 0x8Fu : 0xBFu;
                continue;
            }
            if (c < (got == 0u ? lo : 0x80u) || c > (got == 0u ? hi : 0xBFu)) { valid = 0; continue; }
            got++;
            if (got == need) { need = 0; }
            continue;
        }
        if (run_n != 0u) {
            flush_run(&d, t, s, special, skip, ids, run_from, i, run_n, valid && need == 0u);
            run_n = 0;
        }
        put_token(&d, per_token, p, k, index);
        index++;
    }
    if (run_n != 0u) { flush_run(&d, t, s, special, skip, ids, run_from, n, run_n, valid && need == 0u); }
    return (int64_t)d.total;
}
