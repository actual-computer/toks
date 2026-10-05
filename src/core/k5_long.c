/* k5_long.c: the long-piece cache behind K5's K6 call, one c file for every tier (kernels.md §6 "Long pieces") */
#include "bpe.h"
#include "kernels.h"

static inline uint32_t lc_hash(const uint8_t *p, uint64_t n)          /* one multiply per 8 bytes, loads inside p */
{
    uint64_t h = 0x9E3779B97F4A7C15ull ^ n, i = 0;
    for (; i + 8u <= n; i += 8u) { h = toks_mix64(h, bpe_load_le(p + i, 8u)); }   /* bound: n / 8 */
    if (i < n) { h = toks_mix64(h, bpe_load_le(p + i, n - i)); }
    return TOKS_TEST_DEGEN((uint32_t)(h ^ (h >> 29)));
}

static inline int lc_same(const uint8_t *a, const uint8_t *b, uint64_t n)             /* every byte of [0, n) */
{
    uint64_t i = 0;
    for (; i + 8u <= n; i += 8u) {                                     /* bound: n / 8 */
        if (bpe_load_le(a + i, 8u) != bpe_load_le(b + i, 8u)) { return 0; }
    }
    return i == n || bpe_load_le(a + i, n - i) == bpe_load_le(b + i, n - i);
}

static inline uint64_t lc_ids_at(uint64_t len) { return (4u + len + 3u) & ~(uint64_t)3u; }      /* in the entry */
static inline uint64_t lc_bytes(uint64_t len, uint64_t n) { return (lc_ids_at(len) + 4u * n + 27u) & ~(uint64_t)15u; }

/* a hit (generation, hash, length, every byte; inside the filled arena, n <= len): its ids at out, 16 bytes at a
 * time (K5's room is len + 4), their count; else 0 */
static inline __attribute__((always_inline)) uint64_t lc_get(toks_lcache *lc, toks_k6_args *a, const uint32_t *s,
                                                             uint32_t h)
{
    uint64_t len = a->len, hl = (uint64_t)h | (len << 32), g;
    for (uint32_t w = 0; w < 4u; w++, s += 4) {                        /* bound: 4 slots */
        memcpy(&g, s, 8);                                              /* the slot's hash and length at once */
        if (g != hl || s[3] != (uint32_t)lc->gen || s[2] + lc_bytes(len, 0u) > lc->pos) { continue; }
        const uint8_t *e = lc->arena + s[2];
        uint32_t n;
        memcpy(&n, e, 4);
        if (n == 0u || n > len || s[2] + lc_bytes(len, n) > lc->pos || !lc_same(e + 4, a->piece, len)) { continue; }
        for (uint64_t j = 0; j < 4u * (uint64_t)n; j += 16u) {          /* bound: n / 4 */
            memcpy((uint8_t *)(void *)a->out + j, e + lc_ids_at(len) + j, 16);
        }
        a->n_out = n;
        a->merges = 0u;
        lc->hits += 1u;
        return n;
    }
    return 0u;
}

/* K6's n ids at out: append (n, bytes, ids), 64-aligned when it fits a line, its slot first (the last drops); a full
 * arena starts the next generation */
static void lc_put(toks_lcache *lc, const toks_k6_args *a, uint32_t *s, uint32_t h, uint64_t n)
{
    uint64_t len = a->len, need = lc_bytes(len, n), pos = lc->pos;
    if (need > lc->arena_bytes) { return; }
    if (need <= 64u && (pos & 63u) + need > 64u) { pos = (pos + 63u) & ~(uint64_t)63u; }
    if (pos > lc->arena_bytes - need) {
        lc->gen = lc->gen >= 0xFFFFFFFFu ? 1u : lc->gen + 1u;
        if (lc->gen == 1u) { memset(lc->buckets, 0, (size_t)((lc->mask + 1u) * TOKS_BUCKET)); }
        pos = 0u;
    }
    uint8_t *e = lc->arena + pos;
    uint32_t n32 = (uint32_t)n;
    memcpy(e, &n32, 4);
    for (uint64_t i = 0; i < len; i++) { e[4 + i] = a->piece[i]; }    /* bound: len */
    for (uint64_t j = 0; j < 4u * n; j += 16u) {                       /* bound: n / 4 (the entry has the room) */
        memcpy(e + lc_ids_at(len) + j, (const uint8_t *)(const void *)a->out + j, 16);
    }
    memcpy(s + 12, s + 8, 16);
    memcpy(s + 8, s + 4, 16);
    memcpy(s + 4, s, 16);
    s[0] = h;
    s[1] = (uint32_t)len;
    s[2] = (uint32_t)pos;
    s[3] = (uint32_t)lc->gen;
    lc->pos = pos + need;
}

/* lc NULL, or a piece of < 5 bytes (<= 4 ids: the short cache's): K6 itself */
#define K5_LONG(tier)                                                                                 \
    uint64_t toks_k5_long_##tier(const toks_tables *t, toks_k6_args *a, toks_lcache *lc)               \
    {                                                                                                 \
        if (lc == NULL || a->len < 5u) { return toks_k6_bpe_##tier(t, a); }                           \
        uint32_t h = lc_hash(a->piece, a->len);                                                       \
        uint32_t *s = (uint32_t *)(void *)(lc->buckets + ((uint64_t)h & lc->mask) * TOKS_BUCKET);     \
        uint64_t n = lc_get(lc, a, s, h);                                                             \
        if (n != 0u) { return n; }                                                                    \
        n = toks_k6_bpe_##tier(t, a);                                                                 \
        lc->misses += 1u;                                                                             \
        if ((a->len > 15u || n > 4u) && n != 0u) { lc_put(lc, a, s, h, n); }                          \
        return n;                                                                                     \
    }

K5_LONG(c)
#if TOKS_HAVE_K5_NEON                                       /* the asm K5s call these (and K6 is there with them) */
K5_LONG(neon)
#endif
#if TOKS_HAVE_K5_AVX2
K5_LONG(avx2)
#endif
