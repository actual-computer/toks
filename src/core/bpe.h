/* bpe.h: the byte-level bpe tables (layout.h) built from a toks_config, and the probe helpers shared by bpe_build.c
 * and the c twins, pure functions of (t, arguments). The tables' conventions (raw bytes, raw ranks): kernels.md §5. */
#ifndef TOKS_BPE_H
#define TOKS_BPE_H

#include "core.h"

/* arena bytes toks_bpe_build allocates at most, whatever the arena's base alignment (kernels.md §5 "Tables") */
uint64_t toks_bpe_tables_bytes(const toks_config *cfg);

/* the tables of bpe_build.c's header from cfg's merges (id triples in raw rank order): 0, TOKS_E_FORMAT (a byte
 * token missing, a merge id outside the model vocabulary), TOKS_E_UNSUPPORTED (too many ids / merges), TOKS_E_NOMEM */
int64_t toks_bpe_build(toks_tables *t, toks_arena *ar, const toks_config *cfg);

/* the merge table (layout.h) in slots (nb buckets, a power of two >= 2, 8 nb > the merges kept) and its pair filter in
 * pf (nb / 2 words): merges i < n in raw order, a pair met again taking the later rank (hf's MergeMap); with reach !=
 * NULL only merges whose operands both have their bit set. Sets merge_slots / -mask / -shift / -maxprobe, pairf and
 * n_merges; returns the merges left out. */
uint64_t toks_merge_slots(toks_tables *t, uint64_t *slots, uint64_t *pf, uint64_t nb, const uint32_t *ml, const uint32_t *mr,
                          uint32_t n, const uint32_t *reach);

static inline uint64_t bpe_pow2(uint64_t n)                /* the least power of two >= n (n <= 2^40 here) */
{
    uint64_t p = 1u;
    while (p < n) { p <<= 1; }                             /* bound: 41 doublings */
    return p;
}

/* ---- shared probe helpers (layout.h formats) ----------------------------------------------------- */

#define BPE_PRIO_MASK   ((UINT64_C(1) << TOKS_PRIO_BITS) - 1u)   /* u64, so ~BPE_PRIO_MASK keeps the key */
#define BPE_EMPTY_SLOT  UINT64_MAX

/* merge key = left << 21 | right (both ids < 2^21, layout.h) */
static inline uint64_t bpe_pair_key(uint32_t left, uint32_t right)
{
    return ((uint64_t)left << 21) | (uint64_t)right;
}

/* the merge table (layout.h) in locals: a caller's stores into its symbol arrays never make the compiler reload t */
typedef struct bpe_mt {
    const uint64_t *slots;
    uint64_t        shift, mask, maxprobe;
    const uint32_t *r2i;                                   /* NULL under TOKS_TF_IDS_AS_RANK */
    const uint64_t *pf;                                    /* the pair filter, NULL: none */
} bpe_mt;

static inline bpe_mt bpe_mt_of(const toks_tables *t)
{
    bpe_mt mt = { t->merge_slots, t->merge_shift, t->merge_mask, t->merge_maxprobe,
                  (t->flags & TOKS_TF_IDS_AS_RANK) != 0u ? NULL : t->rank2id, t->pairf };
    return mt;
}

/* the pair filter (layout.h TT_PAIRF): one word per two merge-table buckets holding two bits of each key homed in
 * them, bits 20-25 and 26-31 of the hash (below any bucket index's). *w = the key's word; its bits returned (a key
 * without both set has no merge) */
static inline uint64_t bpe_pf_bits(uint64_t key, uint64_t shift, uint64_t *w)
{
    uint64_t h = key * TOKS_FIB64;
    *w = h >> shift >> 1;
    return (UINT64_C(1) << ((h >> 20) & 63u)) | (UINT64_C(1) << ((h >> 26) & 63u));
}

/* the prio of the pair key, or TOKS_PRIO_NONE: the home bucket (key * FIB64) >> merge_shift, then the following ones
 * (wrapping), at most merge_maxprobe; an empty slot ends the probe */
static inline uint32_t bpe_mt_find(const bpe_mt *mt, uint64_t key)
{
    uint64_t b = ((key * TOKS_FIB64) >> mt->shift) & mt->mask, w = 0u, m = mt->pf != NULL ? bpe_pf_bits(key, mt->shift, &w) : 0u;
    if ((mt->pf != NULL ? mt->pf[w] & m : m) != m) { return TOKS_PRIO_NONE; }   /* filtered out */
    for (uint64_t i = 0; i < mt->maxprobe; i++) {          /* bound: merge_maxprobe */
        const uint64_t *s = mt->slots + b * 8u;
        for (uint32_t j = 0; j < 8u; j++) {                /* bound: 8 (one bucket) */
            if (s[j] == BPE_EMPTY_SLOT) { return TOKS_PRIO_NONE; }
            if ((s[j] >> TOKS_PRIO_BITS) == key) { return (uint32_t)(s[j] & BPE_PRIO_MASK); }
        }
        b = (b + 1u) & mt->mask;
    }
    return TOKS_PRIO_NONE;
}

static inline uint32_t bpe_mt_id(const bpe_mt *mt, uint32_t prio) { return mt->r2i == NULL ? prio : mt->r2i[prio]; }

/* p[0, n) as a little-endian word, 1 <= n <= 8, reading only those n bytes (two overlapping loads) */
static inline uint64_t bpe_load_le(const uint8_t *p, uint64_t n)
{
    if (n >= 4u) {
        uint32_t x, y;
        memcpy(&x, p, 4);
        memcpy(&y, p + n - 4u, 4);
        return (uint64_t)x | ((uint64_t)y << (8u * (n - 4u)));
    }
    if (n >= 2u) {
        uint16_t x, y;
        memcpy(&x, p, 2);
        memcpy(&y, p + n - 2u, 2);
        return (uint64_t)x | ((uint64_t)y << (8u * (n - 2u)));
    }
    return (uint64_t)p[0];
}

/* the vocab hash (layout.h): TOKS_VSEED ^ len, then crc32c_u64 over p's little-endian words, the last zero-padded */
static inline uint32_t bpe_vhash_h(const uint8_t *p, uint64_t len)
{
    uint32_t h = TOKS_VSEED ^ (uint32_t)len;
    uint64_t i = 0;
    for (; len - i >= 8u; i += 8u) { h = toks_crc32c_u64(h, bpe_load_le(p + i, 8u)); }   /* bound: len / 8 */
    if (i < len) { h = toks_crc32c_u64(h, bpe_load_le(p + i, len - i)); }
    return h;
}

/* whole-piece model-vocabulary lookup (layout.h): a hit needs equal h and equal bytes */
static inline int bpe_vhash_find(const toks_tables *t, const uint8_t *p, uint64_t len, uint32_t *id_out)
{
    if (t->vhash == NULL) { return 0; }
    uint32_t h = bpe_vhash_h(p, len);
    uint64_t i = (uint64_t)h & t->vhash_mask;
    for (uint64_t k = 0; k <= t->vhash_mask; k++) {      /* bound: the slots (load < 0.5 ends it sooner) */
        uint64_t slot = t->vhash[i];
        if (slot == BPE_EMPTY_SLOT) { return 0; }
        if ((uint32_t)slot == h) {
            uint32_t id = (uint32_t)(slot >> 32);
            uint32_t o0 = t->tok_off[id];
            if ((uint64_t)t->tok_off[id + 1u] - o0 == len && memcmp(t->tok_bytes + o0, p, len) == 0) {
                *id_out = id;
                return 1;
            }
        }
        i = (i + 1u) & t->vhash_mask;
    }
    return 0;
}

/* ---- premerge (layout.h; kernels.md §5.1): K6's initial symbol for a 2- or 3-byte char ------------- */

/* a neighbour byte's risk class in the ascii pair table (layout.h TOKS_APM_*): letters (case-folded), space, digits,
 * other ascii, continuation bytes, the two lead ranges */
static inline uint32_t bpe_apm_class(uint32_t b)
{
    uint32_t l = (b | 0x20u) - 0x61u;
    return l < 26u ? l : b == 0x20u ? 26u : b - 0x30u < 10u ? 27u : b < 0x80u ? 28u : b < 0xC0u ? 29u : b < 0xE0u ? 30u : 31u;
}

/* the risk bit of the byte before a char / of the byte after it (layout.h) */
static inline uint32_t bpe_pm_lbit(uint32_t b)
{
    return b < 0x80u ? (b == 0x20u ? 56u : 57u) : b < 0xC0u ? 40u + (b & 15u) : 63u;
}

static inline uint32_t bpe_pm_rbit(uint32_t b)
{
    return b < 0x80u ? 58u : b < 0xC0u ? 63u : b < 0xE0u ? 59u : b < 0xF0u ? 24u + (b & 15u) : 60u;
}

/* the char pattern at b[0, avail): a lead C0..DF and one continuation byte (2), or a lead E0..EF and two (3);
 * its stage indexes. 0 when b[0] starts neither (the byte stays a byte). */
static inline uint32_t bpe_pm_index(const uint8_t *b, uint64_t avail, uint32_t *s1, uint32_t *s2)
{
    uint32_t b0 = b[0];
    if (b0 >= 0xC0u && b0 < 0xE0u && avail >= 2u && (b[1] & 0xC0u) == 0x80u) {
        *s1 = b0 - 0xC0u;
        *s2 = b[1] & 63u;
        return 2u;
    }
    if (b0 >= 0xE0u && b0 < 0xF0u && avail >= 3u && (b[1] & 0xC0u) == 0x80u && (b[2] & 0xC0u) == 0x80u) {
        *s1 = 32u + (((b0 & 15u) << 6) | (b[1] & 63u));
        *s2 = b[2] & 63u;
        return 3u;
    }
    return 0u;
}

/* K6's initial symbol at p[i] of the piece p[0, len) under the premerge table pm: the char's token (*k = its
 * length) when it has an entry and neither neighbour byte is risky for it, else UINT32_MAX (*k = 1). Reads
 * only inside the piece. */
static inline uint32_t bpe_pm_at(const uint8_t *pm, const uint8_t *p, uint64_t i, uint64_t len, uint32_t *k)
{
    uint32_t s1 = 0u, s2 = 0u, n = bpe_pm_index(p + i, len - i, &s1, &s2);
    *k = 1u;
    if (n == 0u) { return UINT32_MAX; }
    uint32_t off = ((const uint32_t *)(const void *)pm)[s1];
    uint64_t e = *(const uint64_t *)(const void *)(pm + off + 8u * s2);
    uint32_t lb = i > 0u ? bpe_pm_lbit(p[i - 1u]) : (uint32_t)TOKS_PM_NEVER;
    uint32_t rb = i + n < len ? bpe_pm_rbit(p[i + n]) : (uint32_t)TOKS_PM_NEVER;
    if (e == 0u || (((e >> lb) | (e >> rb)) & 1u) != 0u) { return UINT32_MAX; }
    *k = n;
    return (uint32_t)e & TOKS_ID_MASK;
}

/* K6's merge loop (k6_c.c, kernels.md §5) on given initial symbols: work + 16 cap holds sym[0, n) (1 <= n <= cap;
 * work 8-aligned, 32 cap bytes); the ids go to out (work itself may be out); returns their count */
uint64_t toks_k6_merge(const toks_tables *t, uint8_t *work, uint64_t cap, uint64_t n, uint32_t *out);

/* the long-piece cache's per-call descriptor (kernels.md §6 "Long pieces"), built by the driver from the header */
typedef struct toks_lcache {
    uint8_t *buckets;
    uint8_t *arena;
    uint64_t mask;           /* bucket mask (nb a power of two) */
    uint64_t arena_bytes;
    uint64_t pos;            /* in/out: the arena's fill in this generation */
    uint64_t gen;            /* in/out: 1 .. 2^32 - 1 */
    uint64_t hits, misses;   /* in/out: counters, K5 adds its calls' */
} toks_lcache;
_Static_assert(sizeof(toks_lcache) == 64, "toks_lcache size");

/* K5's K6 call (K6's args, out room >= len + 4): K6, or with lc the long-piece cache in front of it */
uint64_t toks_k5_long_c(const toks_tables *t, toks_k6_args *a, toks_lcache *lc);
uint64_t toks_k5_long_neon(const toks_tables *t, toks_k6_args *a, toks_lcache *lc);
uint64_t toks_k5_long_avx2(const toks_tables *t, toks_k6_args *a, toks_lcache *lc);

/* ---- shortcut entries (the static words table and the dynamic cache) ---------------------------- */

/* key (layout.h) in the builder's form; K5 reads its keys in place (bpe_key_at) */
static inline void bpe_key_make(uint8_t key[16], const uint8_t *p, uint32_t len)
{
    memset(key, 0, 16);
    memcpy(key, p, len);
    key[15] = (uint8_t)len;
}

/* a key as its two little-endian words (layout.h): lo = bytes 0-7, hi = bytes 8-15 */
typedef struct bpe_key {
    uint64_t lo, hi;
} bpe_key;

/* the key of text[s, s + n), 2 <= n <= 15, read in place, never a byte at or past tlen (kernels.md §6 "Key") */
static inline bpe_key bpe_key_at(const uint8_t *text, uint64_t tlen, uint64_t s, uint64_t n)
{
    bpe_key k;
    const uint8_t *p = text + s;
    if (s + 16u <= tlen) {
        memcpy(&k.lo, p, 8);
        memcpy(&k.hi, p + 8, 8);
        k.lo &= n >= 8u ? UINT64_MAX : (UINT64_C(1) << (8u * n)) - 1u;
        k.hi &= n > 8u ? (UINT64_C(1) << (8u * (n - 8u))) - 1u : 0u;
    } else if (n > 8u) {
        memcpy(&k.lo, p, 8);
        k.hi = bpe_load_le(p + 8, n - 8u);
    } else {
        k.lo = bpe_load_le(p, n);
        k.hi = 0u;
    }
    k.hi |= (uint64_t)n << 56;                             /* byte 15 = len */
    return k;
}

/* layout.h's key hash (= toks_key_hash(TOKS_HSEED, key)), once per piece */
static inline uint32_t bpe_key_hash(bpe_key k)
{
    return toks_crc32c_u64(toks_crc32c_u64(TOKS_HSEED, k.lo), k.hi);
}

/* the value of key k in a bucket (layout.h), or NULL; an empty way never matches (a key's byte 15 is >= 1) */
static inline const uint8_t *bpe_bucket_get(const uint8_t *b, bpe_key k)
{
    uint64_t w[4];
    memcpy(w, b, 32);
    if (w[0] == k.lo && w[1] == k.hi) { return b + 32; }
    if (w[2] == k.lo && w[3] == k.hi) { return b + 48; }
    return NULL;
}

/* the dynamic cache's lookup: the way holding k (way 0 first), a hit when its tag bits are tw (layout.h) */
static inline const uint8_t *bpe_cache_get(const uint8_t *b, bpe_key k, uint64_t tw)
{
    const uint8_t *v = bpe_bucket_get(b, k);
    if (v == NULL) { return NULL; }
    uint64_t g;
    memcpy(&g, v + 8, 8);
    return (g & TOKS_TAG_MASK64) == tw ? v : NULL;
}

/* val: ids[0] = count << 29 | id0; ids[1..3] = the next ids (0 beyond count). */
static inline uint32_t bpe_val_count(const uint32_t *v)
{
    return v[0] >> TOKS_VAL_COUNT_SHIFT;
}

/* idv[0..n) (1 <= n <= 4, any alignment) as a val, with tag bits tw (0 for static entries; layout.h) */
static inline void bpe_val_pack_tag(uint32_t val[4], const uint32_t *idv, uint32_t n, uint64_t tw)
{
    val[0] = (n << TOKS_VAL_COUNT_SHIFT) | (toks_ld32(idv) & TOKS_ID_MASK);
    val[1] = n > 1u ? (toks_ld32(idv + 1) & TOKS_ID_MASK) : 0u;
    val[2] = (n > 2u ? (toks_ld32(idv + 2) & TOKS_ID_MASK) : 0u) | (uint32_t)tw;
    val[3] = (n > 3u ? (toks_ld32(idv + 3) & TOKS_ID_MASK) : 0u) | (uint32_t)(tw >> 32);
}

/* a val to out: 4 masked ids in one 16-byte store (K5's room rule; any alignment); returns the count */
static inline uint64_t bpe_val_put(const uint8_t *v, void *out)
{
    uint32_t w[4];
    memcpy(w, v, 16);
    uint64_t count = (uint64_t)(w[0] >> TOKS_VAL_COUNT_SHIFT);
    for (int i = 0; i < 4; i++) { w[i] &= TOKS_ID_MASK; }      /* bound: 4 */
    memcpy(out, w, 16);
    return count;
}

/* a dynamic-cache fill (kernels.md §6): old way 0 -> way 1, the new entry -> way 0 */
static inline void bpe_cache_fill(uint8_t *bucket, bpe_key k, const uint32_t val[4])
{
    memcpy(bucket + 16, bucket, 16);
    memcpy(bucket + 48, bucket + 32, 16);
    memcpy(bucket, &k.lo, 8);
    memcpy(bucket + 8, &k.hi, 8);
    memcpy(bucket + 32, val, 16);
}

/* a static entry (val's tag bits 0) into the words table: the first empty way of bucket h, then of rotr32(h, 16)
 * (layout.h); 0 when both are full (the key is left out: a shortcut, never a requirement) */
static inline int bpe_words_put(uint8_t *words, uint64_t mask, uint32_t h, bpe_key k, const uint32_t val[4])
{
    for (uint32_t which = 0; which < 2u; which++) {          /* bound: 2 buckets */
        uint8_t *b = words + ((uint64_t)(which == 0u ? h : (h >> 16) | (h << 16)) & mask) * TOKS_BUCKET;
        for (uint32_t way = 0; way < 2u; way++, b += 16) {  /* bound: 2 ways */
            if (b[15] != 0u) { continue; }                  /* byte 15 = len, 0: empty */
            memcpy(b, &k.lo, 8);
            memcpy(b + 8, &k.hi, 8);
            memcpy(b + 32, val, 16);
            return 1;
        }
    }
    return 0;
}

/* the static words table's value of key k (hash h), or NULL: bucket h, then rotr32(h, 16) (layout.h) */
static inline const uint8_t *bpe_words_probe(const uint8_t *words, uint64_t mask, uint32_t h, bpe_key k)
{
    const uint8_t *v = bpe_bucket_get(words + ((uint64_t)h & mask) * TOKS_BUCKET, k);
    if (v == NULL) { v = bpe_bucket_get(words + ((uint64_t)((h >> 16) | (h << 16)) & mask) * TOKS_BUCKET, k); }
    return v;
}

#endif /* TOKS_BPE_H */
