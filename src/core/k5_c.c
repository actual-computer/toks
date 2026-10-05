/* k5_c.c: the c twin of K5, encode pieces; the asm tiers are byte-identical to it. Semantics (the shortcuts, order
 * and fill by scratch age, tags, out[0, room) with room >= bytes + 4, in-out counters): kernels.md §6. */
#include "bpe.h"
#include "kernels.h"

uint64_t toks_k5_encode_c(const toks_tables *t, toks_k5_args *a)
{
    /* locals: stores to out and the cache may alias *a and *t (-fno-strict-aliasing) */
    const uint8_t *text = a->text;
    const uint32_t *ends = a->ends;
    const uint64_t tlen = a->len, np = a->n, cmask = a->cache_mask, wbytes = a->work_bytes;
    const uint64_t tw = toks_tag_word(a->cache_tag);
    uint32_t *out = a->out;
    uint8_t *cache = a->cache, *work = a->work;
    const uint32_t *byte2id = t->byte2id;
    const uint8_t *words = t->words;
    const uint64_t wmask = t->words_mask;
    uint64_t n_out = 0u, hits_static = 0u, hits_cache = 0u, misses = 0u;
    uint64_t s = a->start;
    /* the scratch's age: warm = cache first, static answers fill too; fresh = static first (kernels.md §6) */
    const int warm = a->hits_static + a->hits_cache + a->misses >= (uint64_t)TOKS_K5_WARM;
    for (uint64_t i = 0; i < np; i++) {                    /* bound: a->n pieces */
        uint64_t ps = s, len = (uint64_t)ends[i] - s;
        uint32_t *o = out + n_out;
        s = ends[i];

        if (len == 1u) {
            toks_st32(o, byte2id[text[ps]]);
            n_out += 1u;
            hits_static += 1u;
            continue;
        }
        uint8_t *bucket = NULL;
        bpe_key k = { 0u, 0u };
        if (len <= (uint64_t)TOKS_KEY_MAXLEN) {
            k = bpe_key_at(text, tlen, ps, len);
            uint32_t h = bpe_key_hash(k);
            if (cache != NULL) { bucket = cache + ((uint64_t)h & cmask) * TOKS_BUCKET; }
            const uint8_t *v = (warm && bucket != NULL) ? bpe_cache_get(bucket, k, tw) : NULL;
            if (v != NULL) {
                n_out += bpe_val_put(v, o);
                hits_cache += 1u;
                continue;
            }
            v = words != NULL ? bpe_words_probe(words, wmask, h, k) : NULL;
            if (v != NULL) {
                n_out += bpe_val_put(v, o);
                hits_static += 1u;
                if (warm && bucket != NULL) {              /* a warm scratch: the static answer fills too */
                    uint32_t val[4];
                    memcpy(val, v, 16);
                    val[2] |= (uint32_t)tw;                /* static vals hold 0 in the tag bits */
                    val[3] |= (uint32_t)(tw >> 32);
                    bpe_cache_fill(bucket, k, val);
                }
                continue;
            }
            v = (!warm && bucket != NULL) ? bpe_cache_get(bucket, k, tw) : NULL;
            if (v != NULL) {
                n_out += bpe_val_put(v, o);
                hits_cache += 1u;
                continue;
            }
        }
        toks_k6_args k6 = { text + ps, len, o, work, wbytes, 0u, 0u, 0u };
        uint64_t n = toks_k5_long_c(t, &k6, (toks_lcache *)a->lcache);   /* K6, or the long cache (k5_long.c) */
        misses += 1u;
        if (bucket != NULL && n <= 4u) {                   /* bucket != NULL: len <= 15, cache on */
            uint32_t val[4];
            bpe_val_pack_tag(val, o, (uint32_t)n, tw);
            bpe_cache_fill(bucket, k, val);
        }
        n_out += n;
    }
    a->n_out = n_out;
    a->hits_static += hits_static;
    a->hits_cache += hits_cache;
    a->misses += misses;
    return n_out;
}
