/*
 * tests/k1/k1_tab.h: added-token tables for the K1 tests, built by src/core/compile.c's own index builder
 * (toks_added_index; layout.h TT_ADD_*, docs/kernels.md §4), and a reference K1 written from the prose (every
 * token of the phase tried at every position, no index): leftmost, then longest. Header-only; included by
 * tests/c/test_k1.c and tests/k1/bench_k1.c.
 *
 * Contents are unique across both phases and 1..TOKS_MAX_ADDED_BYTES long (config.c refuses anything else),
 * so the reference needs no tie rule. A token set with no tokens leaves every add_* pointer NULL, as the
 * compiler does.
 */
#ifndef TOKS_TEST_K1_TAB_H
#define TOKS_TEST_K1_TAB_H

/* kernels.h -> core.h declares memcpy / memset / memcmp itself: it comes before <string.h> */
#include "../../src/core/kernels.h"
#include "../../src/core/compile.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct k1_tok { const uint8_t *s; uint32_t len; uint8_t phase; uint8_t special; } k1_tok;

typedef struct k1_tab {
    toks_tables t;
    toks_added_entry *ent;
    uint8_t *ab;
    uint64_t *idx;     /* [2][65536], then h4 [2 << TOKS_K1_H4_BITS] */
    uint32_t *cand;    /* [toks_added_cand_words]: index2's and h4's lists, the radix trees of h4's long buckets */
    uint8_t shuf[64];
    uint32_t single[512];
} k1_tab;

static void k1_tab_free(k1_tab *k)
{
    free(k->ent);
    free(k->ab);
    free(k->idx);
    free(k->cand);
    memset(k, 0, sizeof *k);
}

/* entries grouped by phase in input order (compile.c build_added), then compile.c's own index builder
 * (toks_added_index: shufti, single, index2, h4). 0 on success. */
static int k1_tab_build(k1_tab *k, const k1_tok *toks, uint32_t n)
{
    memset(k, 0, sizeof *k);
    if (n == 0u) { return 0; }                              /* no added tokens: NULL tables, no phases */
    uint64_t total = 0, n2 = 0, n4 = 0, b4 = 0;
    for (uint32_t i = 0; i < n; i++) {
        total += toks[i].len;
        n2 += toks[i].len >= 2u;
        n4 += toks[i].len >= 4u;
        b4 += toks[i].len >= 4u ? toks[i].len : 0u;
    }
    k->ent = (toks_added_entry *)calloc(n, sizeof(toks_added_entry));
    k->ab = (uint8_t *)malloc(total ? total : 1u);
    k->idx = (uint64_t *)calloc(2u * 65536u + (2u << TOKS_K1_H4_BITS), 8u);
    k->cand = (uint32_t *)calloc(toks_added_cand_words(n2, n4, b4) + 1u, 4u);
    if (!k->ent || !k->ab || !k->idx || !k->cand) { k1_tab_free(k); return -1; }
    uint32_t e = 0;
    uint64_t off = 0;
    for (uint32_t p = 0; p < 2u; p++) {
        for (uint32_t i = 0; i < n; i++) {
            const k1_tok *a = &toks[i];
            if (a->phase != p) { continue; }
            k->ent[e].off = (uint32_t)off;
            k->ent[e].len = (uint16_t)a->len;
            k->ent[e].flags = (uint8_t)((a->special ? TOKS_AF_SPECIAL : 0u) | (p ? TOKS_AF_NORMALIZED : 0u));
            k->ent[e].phase = (uint8_t)p;
            k->ent[e].id = 1000u + i;
            memcpy(k->ab + off, a->s, a->len);
            off += a->len;
            e++;
        }
    }
    k->t.magic = TOKS_TABLES_MAGIC;
    k->t.version = TOKS_TABLES_VERSION;
    k->t.add_entries = k->ent;
    k->t.add_bytes = k->ab;
    k->t.add_n = e;
    toks_added_index(&k->t, k->shuf, k->idx, k->single, k->cand);
    return 0;
}

/* the prose (kernels.md §4): the smallest m_start >= pos where a token of the phase occurs entirely inside
 * [0, len), the longest one there; none: m_start = m_end = len, m_entry 0. Returns m_start. */
static uint64_t k1_ref(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t phase,
                       uint64_t *m_end, uint64_t *m_entry)
{
    *m_end = len;
    *m_entry = 0;
    if (phase > 1u || t->add_n == 0u) { return len; }
    for (uint64_t i = pos; i < len; i++) {
        uint64_t best = 0, be = 0;
        for (uint64_t e = 0; e < t->add_n; e++) {
            const toks_added_entry *x = &t->add_entries[e];
            if (x->phase != phase || x->len <= best || i + x->len > len) { continue; }
            if (memcmp(t->add_bytes + x->off, text + i, x->len) == 0) { best = x->len; be = e; }
        }
        if (best != 0u) { *m_end = i + best; *m_entry = be; return i; }
    }
    return len;
}

/* K1's batch by the prose: up to cap k1_ref matches, each from the previous one's end; *next = the last end at
 * n == cap, else len. Returns n. */
static uint64_t k1_ref_batch(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t phase,
                             uint64_t cap, toks_k1_match *m, uint64_t *next)
{
    uint64_t n = 0, e, x;
    *next = len;
    while (n < cap) {
        uint64_t s = k1_ref(t, text, len, pos, phase, &e, &x);
        if (s >= len) { return n; }
        m[n++] = (toks_k1_match){ (uint32_t)s, (uint32_t)e, (uint32_t)x, 0u };
        pos = e;
    }
    *next = pos;
    return n;
}

#endif /* TOKS_TEST_K1_TAB_H */
