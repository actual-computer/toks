/* k6_c.c: the c twin of K6 and toks_k6_merge (docs/notes/c-core.md §k6_c.c.1) */
#include "bpe.h"

#define K6_END   UINT32_MAX
#define K6_NONE  UINT32_MAX          /* == TOKS_PRIO_NONE */
#define K6_SHORT 32u                 /* up to this many symbols take the heap-free path */
#define K6_SHIFT 5                   /* key = prio << K6_SHIFT | pos, pos < K6_SHORT */

_Static_assert((1u << K6_SHIFT) == K6_SHORT && TOKS_PRIO_BITS + K6_SHIFT < 32, "k6 short key packing");
_Static_assert(K6_NONE == TOKS_PRIO_NONE, "k6 none");

/* the initial symbols of p[0, len) into id[0, n), the round-one prio of pair (i, i + 1) into pr[i]
 * (pr[n - 1] = NONE); returns n */
static uint32_t k6_init(const toks_tables *t, const bpe_mt *mt, const uint8_t *p, uint64_t len, uint32_t *id,
                        uint32_t *pr)
{
    const uint8_t *pm = t->premerge, *apm = len - 4u <= TOKS_KEY_MAXLEN - 4u ? t->apm : NULL;
    const uint32_t *b2i = t->byte2id, *bp = t->bytepair;
    uint32_t n = 0u, pb = 0u, pbyte = 0u;                  /* pbyte: symbol n - 1 is the byte pb */
    for (uint64_t i = 0; i < len;) {                       /* bound: len */
        uint32_t b = p[i], k = 1u, s = UINT32_MAX;
        if (pm != NULL && b >= 0xC0u) { s = bpe_pm_at(pm, p, i, len, &k); }
        if (apm != NULL && b < 0x80u && i + 2u <= len && p[i + 1u] < 0x80u) {   /* an ascii pair (layout.h TOKS_APM_*) */
            uint32_t w[3], bf = i > 0u ? 1u << apm[p[i - 1u]] : 0u, af = i + 2u < len ? 1u << apm[p[i + 2u]] : 0u;
            memcpy(w, apm + TOKS_APM_PAIRS + 16u * (b << 7 | p[i + 1u]), 12);
            if (w[0] != UINT32_MAX && ((w[1] & af) | (w[2] & bf)) == 0u) { s = w[0]; k = 2u; }
        }
        uint32_t byte = s == UINT32_MAX;
        if (byte) { s = b2i[b]; }
        if (n > 0u) { pr[n - 1u] = byte && pbyte ? bp[(pb << 8) | b] : bpe_mt_find(mt, bpe_pair_key(id[n - 1u], s)); }
        id[n] = s;
        pb = b;
        pbyte = byte;
        n += 1u;
        i += k;
    }
    pr[n - 1u] = K6_NONE;
    return n;
}

/* ---- short: fixed arrays, min scan ------------------------------------------------------------ */

/* id[0, n) and key[0, n) (key[i] = prio << K6_SHIFT | i, or NONE) in place: merge to the end, ids to out */
static uint64_t k6_short(const bpe_mt *mt, uint32_t *id, uint32_t *key, uint32_t n, uint32_t *out)
{
    uint8_t nx[K6_SHORT], pv[K6_SHORT];                    /* nx: n = none; pv: 0xFF = none */
    for (uint32_t i = 0; i < n; i++) {                     /* bound: n */
        nx[i] = (uint8_t)(i + 1u);
        pv[i] = (uint8_t)(i - 1u);
    }
    for (;;) {                                             /* bound: n - 1 merges, then one more scan */
        uint32_t k = K6_NONE;
        for (uint32_t i = 0; i < n; i++) { k = key[i] < k ? key[i] : k; }   /* bound: n */
        if (k == K6_NONE) { break; }
        uint32_t pos = k & (K6_SHORT - 1u), nid = bpe_mt_id(mt, k >> K6_SHIFT), r = nx[pos], rn = nx[r], l = pv[pos];
        id[pos] = nid;
        key[r] = K6_NONE;
        nx[pos] = (uint8_t)rn;
        key[pos] = K6_NONE;
        if (rn < n) {
            pv[rn] = (uint8_t)pos;
            uint32_t q = bpe_mt_find(mt, bpe_pair_key(nid, id[rn]));
            if (q != K6_NONE) { key[pos] = (q << K6_SHIFT) | pos; }
        }
        if (l != 0xFFu) {
            uint32_t q = bpe_mt_find(mt, bpe_pair_key(id[l], nid));
            key[l] = q == K6_NONE ? K6_NONE : (q << K6_SHIFT) | l;
        }
    }
    uint64_t m = 0u;
    for (uint32_t i = 0u; i < n; i = nx[i]) {              /* bound: n (the alive chain from 0) */
        toks_st32(out + m, id[i]);                         /* out: a caller array (core.h) */
        m += 1u;
    }
    return m;
}

static void k6_keys(uint32_t *key, const uint32_t *pr, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) { key[i] = pr[i] == K6_NONE ? K6_NONE : (pr[i] << K6_SHIFT) | i; }   /* bound: n */
}

/* ---- long: heap of (prio, pos), current-prio stale test -------------------------------------- */

static void heap_push(uint64_t *heap, uint64_t *n, uint64_t e)
{
    uint64_t i = *n;
    *n += 1u;
    while (i > 0u) {                                       /* bound: log2(n) + 1 */
        uint64_t p = (i - 1u) >> 1;
        if (heap[p] <= e) { break; }
        heap[i] = heap[p];
        i = p;
    }
    heap[i] = e;
}

static void heap_sift(uint64_t *heap, uint64_t n, uint64_t i, uint64_t e)   /* place e at i, sifting down */
{
    for (;;) {                                             /* bound: log2(n) + 1 */
        uint64_t c = 2u * i + 1u;
        if (c >= n) { break; }
        if (c + 1u < n && heap[c + 1u] < heap[c]) { c += 1u; }
        if (heap[c] >= e) { break; }
        heap[i] = heap[c];
        i = c;
    }
    heap[i] = e;
}

/* symbols ids[0, n) and their round-one prios cur[0, n) are in place (the work-area layout above) */
static uint64_t k6_long(const bpe_mt *mt, uint8_t *work, uint64_t len, uint32_t n, uint32_t *out)
{
    uint64_t *heap = (uint64_t *)(void *)work;
    uint32_t *ids = (uint32_t *)(void *)(heap + 2u * len);
    uint32_t *prev = ids + len, *nxt = prev + len, *cur = nxt + len;
    uint64_t hn = 0u;
    for (uint32_t i = 0; i < n; i++) {                     /* bound: n */
        prev[i] = i - 1u;                                  /* K6_END at 0 */
        nxt[i] = i + 1u == n ? K6_END : i + 1u;
        if (cur[i] != K6_NONE) { heap[hn++] = ((uint64_t)cur[i] << 32) | i; }
    }
    for (uint64_t i = hn / 2u; i > 0u; i--) { heap_sift(heap, hn, i - 1u, heap[i - 1u]); }   /* bound: hn / 2 */

    while (hn > 0u) {                                      /* bound: pops <= pushes <= 3 (n - 1) */
        uint64_t e = heap[0];
        hn -= 1u;
        heap_sift(heap, hn, 0u, heap[hn]);
        uint32_t pos = (uint32_t)e, prio = (uint32_t)(e >> 32);
        if (cur[pos] != prio) { continue; }                /* stale: the pair at pos changed */
        uint32_t nid = bpe_mt_id(mt, prio), r = nxt[pos], rn = nxt[r], l = prev[pos];
        ids[pos] = nid;
        cur[r] = K6_NONE;
        nxt[pos] = rn;
        cur[pos] = K6_NONE;
        if (rn != K6_END) {
            prev[rn] = pos;
            uint32_t q = bpe_mt_find(mt, bpe_pair_key(nid, ids[rn]));
            cur[pos] = q;
            if (q != K6_NONE) { heap_push(heap, &hn, ((uint64_t)q << 32) | pos); }
        }
        if (l != K6_END) {
            uint32_t q = bpe_mt_find(mt, bpe_pair_key(ids[l], nid));
            cur[l] = q;
            if (q != K6_NONE) { heap_push(heap, &hn, ((uint64_t)q << 32) | l); }
        }
    }

    uint64_t m = 0u;
    for (uint32_t i = 0u; i != K6_END; i = nxt[i]) {       /* bound: n (the alive chain from 0) */
        toks_st32(out + m, ids[i]);                        /* out: a caller array (core.h) */
        m += 1u;
    }
    return m;
}

uint64_t toks_k6_merge(const toks_tables *t, uint8_t *work, uint64_t cap, uint64_t n, uint32_t *out)
{
    bpe_mt mt = bpe_mt_of(t);
    uint32_t *ids = (uint32_t *)(void *)(work + 16u * cap), *cur = ids + 3u * cap;
    for (uint64_t i = 0; i + 1u < n; i++) { cur[i] = bpe_mt_find(&mt, bpe_pair_key(ids[i], ids[i + 1u])); }   /* bound: n - 1 */
    cur[n - 1u] = K6_NONE;
    if (n > K6_SHORT) { return k6_long(&mt, work, cap, (uint32_t)n, out); }
    uint32_t id[K6_SHORT], key[K6_SHORT];
    memcpy(id, ids, 4u * n);
    k6_keys(key, cur, (uint32_t)n);
    return k6_short(&mt, id, key, (uint32_t)n, out);
}

uint64_t toks_k6_bpe_c(const toks_tables *t, toks_k6_args *a)
{
    const uint8_t *p = a->piece;
    uint64_t len = a->len;
    uint32_t *out = a->out;
    a->n_out = 0u;
    a->merges = 0u;
    if (len == 0u) { return 0u; }

    const uint32_t f = t->flags;
    if ((f & TOKS_TF_IGNORE_MERGES) != 0u && t->vhash != NULL   /* no probe when K5's words answer it, no vhash token */
        && ((f & TOKS_TF_PROBE_LONG) == 0u                      /* ends in a non-ascii byte, or none that starts so */
            || (len > TOKS_KEY_MAXLEN && ((f & TOKS_TF_PROBE_ASCII) == 0u || p[len - 1u] < 0x80u)))   /* is longer */
        && len <= toks_ld32((const uint32_t *)(const void *)(t->vhash + t->vhash_mask + 1u) + p[0])) {
        uint32_t id = 0u;
        if (bpe_vhash_find(t, p, len, &id)) {
            toks_st32(out, id);
            a->n_out = 1u;
            return 1u;
        }
    }

    bpe_mt mt = bpe_mt_of(t);
    uint32_t id[K6_SHORT], key[K6_SHORT];
    uint64_t n;
    if (len <= K6_SHORT) {                                 /* bytes: round one from bytepair[] */
        const uint32_t *b2i = t->byte2id, *bp = t->bytepair;
        uint32_t any = p[len - 1u];                        /* the OR of the bytes: < 0xC0 when no lead */
        for (uint32_t i = 0; i + 1u < len; i++) {          /* bound: len - 1 */
            uint32_t q = bp[((uint32_t)p[i] << 8) | p[i + 1u]];
            any |= p[i];
            id[i] = b2i[p[i]];
            key[i] = q == K6_NONE ? K6_NONE : (q << K6_SHIFT) | i;
        }
        id[len - 1u] = b2i[p[len - 1u]];
        key[len - 1u] = K6_NONE;
        uint32_t ns = (uint32_t)len;
        if ((any >= 0xC0u && t->premerge != NULL) || (len - 4u <= TOKS_KEY_MAXLEN - 4u && t->apm != NULL)) {   /* premerge */
            uint32_t pr[K6_SHORT];
            ns = k6_init(t, &mt, p, len, id, pr);
            k6_keys(key, pr, ns);
        }
        n = k6_short(&mt, id, key, ns, out);
    } else {
        uint32_t *ids = (uint32_t *)(void *)(a->work + 16u * len), *cur = ids + 3u * len;
        uint32_t ns = k6_init(t, &mt, p, len, ids, cur);
        if (ns <= K6_SHORT) {
            memcpy(id, ids, 4u * ns);
            k6_keys(key, cur, ns);
            n = k6_short(&mt, id, key, ns, out);
        } else {
            n = k6_long(&mt, a->work, len, ns, out);
        }
    }
    a->n_out = n;
    a->merges = len - n;
    return n;
}
