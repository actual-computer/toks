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

/* the value of a key in the builder's form, copied out: 1 found, 0 absent */
static inline int bpe_words_find(const toks_tables *t, const uint8_t key[16], uint32_t val[4])
{
    bpe_key k;
    memcpy(&k.lo, key, 8);
    memcpy(&k.hi, key + 8, 8);
    const uint8_t *v = t->words == NULL ? NULL : bpe_words_probe(t->words, t->words_mask, bpe_key_hash(k), k);
    if (v == NULL) { return 0; }
    memcpy(val, v, 16);
    return 1;
}
#endif
