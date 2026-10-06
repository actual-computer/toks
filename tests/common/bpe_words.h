/* tests/common/bpe_words.h: test and tool helpers over bpe.h's static words table (not used by the library) */
#ifndef TOKS_TEST_BPE_WORDS_H
#define TOKS_TEST_BPE_WORDS_H
#include "bpe.h"

/* the static words table's candidate buckets for a key hash h: h, then rotr32(h, 16) */
static inline uint64_t bpe_words_bucket(const toks_tables *t, uint32_t h, uint32_t which)
{
    uint32_t x = which == 0u ? h : ((h >> 16) | (h << 16));
    return (uint64_t)x & t->words_mask;
}

/* the value of a key in the builder's form, copied out as a 16-byte val (count << 29 | id0, id1, 0, 0): 1 found,
 * 0 absent */
static inline int bpe_words_find(const toks_tables *t, const uint8_t key[16], uint32_t val[4])
{
    bpe_key k;
    memcpy(&k.lo, key, 8);
    memcpy(&k.hi, key + 8, 8);
    const uint8_t *v = t->words == NULL ? NULL : bpe_w3_probe(t->words, t->words_mask, bpe_key_hash(k), k);
    if (v == NULL) { return 0; }
    uint32_t ids[2];
    uint64_t n = bpe_w3_val(v, ids);
    bpe_val_pack_tag(val, ids, (uint32_t)n, 0u);
    return 1;
}

/* way w of words bucket b: its key (*key) and ids; the count, 0 for an empty way */
static inline uint64_t bpe_words_way(const uint8_t *b, uint32_t w, const uint8_t **key, uint32_t ids[2])
{
    *key = b + 16u * w;
    if (b[16u * w + 15u] == 0u) { return 0u; }
    return bpe_w3_val(b + TOKS_W3_VAL + 5u * w, ids);
}
#endif
