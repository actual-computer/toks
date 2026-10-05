/* da.h: the double-array trie builder shared by unigram.c (the Viterbi's lattice) and wp.c (WordPiece's longest
 * match): sort the keys, keep the last id of equal ones, place the nodes depth first (a path's cells sit close), darts'
 * next_check_pos heuristic keeping the fit search linear (docs/notes/c-core.md §unigram.c.3). Load time only. */
#ifndef TOKS_DA_H
#define TOKS_DA_H

#include "core.h"

#define TOKS_DA_FREE (-1)

typedef struct ukey { uint64_t off; uint32_t len; uint32_t id; } ukey;   /* a key: kb[off, off + len), its id */

static inline int key_less(const uint8_t *kb, const ukey *a, const ukey *b)
{
    uint32_t m = a->len < b->len ? a->len : b->len;
    int c = (m != 0u) ? memcmp(kb + a->off, kb + b->off, m) : 0;
    if (c != 0) { return c < 0; }
    if (a->len != b->len) { return a->len < b->len; }
    return a->id < b->id;
}

/* heap sort of keys by (bytes, id): no recursion, no callback (SPEC §9). bound: n log n sift steps */
static inline void sift(const uint8_t *kb, ukey *k, uint64_t root, uint64_t end)
{
    while (2u * root + 1u < end) {                           /* bound: log2(end) */
        uint64_t c = 2u * root + 1u;
        if (c + 1u < end && key_less(kb, &k[c], &k[c + 1u])) { c++; }
        if (!key_less(kb, &k[root], &k[c])) { return; }
        ukey t = k[root]; k[root] = k[c]; k[c] = t;
        root = c;
    }
}

static inline void heap_sort(const uint8_t *kb, ukey *k, uint64_t n)
{
    if (n < 2u) { return; }
    for (uint64_t i = n / 2u; i > 0u; i--) { sift(kb, k, i - 1u, n); }   /* bound: n / 2 */
    for (uint64_t e = n - 1u; e > 0u; e--) {                  /* bound: n - 1 */
        ukey t = k[0]; k[0] = k[e]; k[e] = t;
        sift(kb, k, 0u, e);
    }
}

typedef struct qent { uint32_t slot, lo, hi, depth; } qent;

/* the double array's columns base | check | term in one block of 3 cap cells: cells [from, cap) set free */
static inline void da_init(int32_t *arr, uint64_t cap, uint64_t from)
{
    for (uint64_t i = from; i < cap; i++) {                  /* bound: cap */
        arr[i] = 0;
        arr[cap + i] = TOKS_DA_FREE;
        arr[2u * cap + i] = -1;
    }
}

/* grows the block to at least need cells (load time: doubling, so log2(2^31 / 512) rounds at most) */
static inline int da_grow(int32_t **arr, uint64_t *cap, uint64_t need)
{
    uint64_t oc = *cap, nc = oc;
    while (nc < need) { nc *= 2u; }                          /* bound: 23 doublings from 512 to 2^31 */
    if (nc == oc) { return 0; }
    if (nc > 0x7FFFFFFFu) { return -1; }
    int32_t *na = (int32_t *)toks_plat_alloc(nc * 12u);
    if (na == NULL) { return -1; }
    for (uint32_t c = 0u; c < 3u; c++) {                     /* bound: 3 columns */
        memcpy(na + c * nc, *arr + c * oc, (size_t)oc * 4u);
    }
    da_init(na, nc, oc);
    toks_plat_free(*arr, oc * 12u);
    *arr = na;
    *cap = nc;
    return 0;
}

/* the trie of keys[0, *nk) (bytes in kb; sorted here, equal keys keep the last id: *nk shrinks): *arr = base | check |
 * term columns of *cap cells each (toks_plat_alloc'd, the caller frees *cap * 12 bytes), cells [0, *da_len) used,
 * root 0 (check[0] = -2), child = base[node] + byte valid iff check[child] == node, term[node] = the id ending there
 * or -1. 0, TOKS_E_NOMEM or TOKS_E_LIMIT (*why names it). */
static inline int64_t toks_da_build(const uint8_t *kb, ukey *keys, uint64_t *nkp, int32_t **arrp, uint64_t *capp,
                             uint64_t *da_lenp, const char **why)
{
    uint64_t nk = *nkp;
    heap_sort(kb, keys, nk);
    uint64_t w = 0u;
    for (uint64_t i = 0u; i < nk; i++) {                     /* bound: nk */
        if (w > 0u && keys[w - 1u].len == keys[i].len &&
            memcmp(kb + keys[w - 1u].off, kb + keys[i].off, keys[i].len) == 0) {
            keys[w - 1u] = keys[i];                          /* sorted by id within equal bytes: last wins */
            continue;
        }
        keys[w++] = keys[i];
    }
    nk = w;
    *nkp = nk;
    /* trie nodes: 1 + sum of (len - lcp with the previous key) */
    uint64_t nodes = 1u;
    for (uint64_t i = 0u; i < nk; i++) {                     /* bound: nk */
        uint32_t l = 0u;
        if (i > 0u) {
            uint32_t m = keys[i].len < keys[i - 1u].len ? keys[i].len : keys[i - 1u].len;
            while (l < m && kb[keys[i].off + l] == kb[keys[i - 1u].off + l]) { l++; }   /* bound: the key */
        }
        nodes += keys[i].len - l;
    }
    uint64_t cap = 2u * nodes + 512u;
    if (cap > 0x7FFFFFFFu) { *why = "trie too large"; return TOKS_E_LIMIT; }
    uint64_t q_bytes = nodes * sizeof(qent) + 64u;
    qent *q = (qent *)toks_plat_alloc(q_bytes);
    int32_t *arr = (int32_t *)toks_plat_alloc(cap * 12u);
    if (q == NULL || arr == NULL) {
        if (q != NULL) { toks_plat_free(q, q_bytes); }
        if (arr != NULL) { toks_plat_free(arr, cap * 12u); }
        *why = "trie memory";
        return TOKS_E_NOMEM;
    }
    da_init(arr, cap, 0u);
    arr[cap] = -2;                                            /* check[0]: the root is never anyone's child */
    uint64_t qt = 0u, ncp = 1u, da_len = 1u;
    q[qt].slot = 0u; q[qt].lo = 0u; q[qt].hi = (uint32_t)nk; q[qt].depth = 0u; qt++;
    while (qt > 0u) {                                         /* bound: nodes (each node is stacked once) */
        qent e = q[--qt];                                     /* depth first: a node's subtree is placed next */
        uint32_t lo = e.lo, hi = e.hi, d = e.depth;
        if (lo < hi && keys[lo].len == d) { arr[2u * cap + e.slot] = (int32_t)keys[lo].id; lo++; }
        uint32_t lab[256], llo[256], lhi[256], nl = 0u;
        uint32_t j = lo;
        while (j < hi) {                                      /* bound: hi - lo */
            uint8_t c = kb[keys[j].off + d];
            uint32_t k = j + 1u;
            while (k < hi && kb[keys[k].off + d] == c) { k++; }   /* bound: hi - j */
            lab[nl] = c; llo[nl] = j; lhi[nl] = k; nl++;
            j = k;
        }
        if (nl == 0u) { continue; }
        /* the first free cell at or after max(lab[0] + 1, ncp) whose base fits every label (darts 0.32) */
        uint64_t pos = ((uint64_t)lab[0] + 1u > ncp ? (uint64_t)lab[0] + 1u : ncp) - 1u;
        uint64_t nonzero = 0u, b = 0u;
        int first = 1;
        for (;;) {                                            /* bound: cells scanned < final cap <= 2^31 */
            pos++;
            if (pos + 257u >= cap) {
                if (da_grow(&arr, &cap, pos + 258u) != 0) {
                    toks_plat_free(q, q_bytes);
                    toks_plat_free(arr, cap * 12u);
                    *why = "trie over 2^31 cells";
                    return TOKS_E_LIMIT;
                }
            }
            if (arr[cap + pos] != TOKS_DA_FREE) { nonzero++; continue; }
            if (first) { ncp = pos; first = 0; }
            b = pos - lab[0];                                 /* >= 1: pos >= lab[0] + 1 */
            uint32_t ok = 1u;
            for (uint32_t x = 1u; x < nl; x++) {              /* bound: 256 */
                if (arr[cap + b + lab[x]] != TOKS_DA_FREE) { ok = 0u; break; }
            }
            if (ok) { break; }
        }
        if (nonzero * 20u >= (pos - ncp + 1u) * 19u) { ncp = pos; }   /* >= 95% of the scan was full */
        arr[e.slot] = (int32_t)b;
        for (uint32_t x = 0u; x < nl; x++) {                  /* bound: 256 */
            uint64_t sl = b + lab[x];
            arr[cap + sl] = (int32_t)e.slot;
            q[qt].slot = (uint32_t)sl; q[qt].lo = llo[x]; q[qt].hi = lhi[x]; q[qt].depth = d + 1u; qt++;
            if (sl + 1u > da_len) { da_len = sl + 1u; }
        }
    }
    toks_plat_free(q, q_bytes);
    *arrp = arr;
    *capp = cap;
    *da_lenp = da_len;
    return 0;
}

#endif /* TOKS_DA_H */
