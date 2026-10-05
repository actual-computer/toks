/* k1_c.c: K1's c twin, added-token find (kernels.md §4), docs/notes/c-core.md §k1_c.c.1 */
#include "kernels.h"

/* the longest token of a long h4 bucket at s[0, avail): a walk down its radix tree from the root at base + off (kernels.md
 * §4); 0 when none, else its length and *entry */
static uint64_t k1_radix(const uint8_t *base, uint32_t off, const uint8_t *s, uint64_t avail, uint64_t *entry)
{
    uint64_t d = 0, best = 0;
    for (;;) {                                          /* bound: avail + 1 nodes (a child's label has >= 1 byte) */
        const uint8_t *nd = base + off;
        uint16_t ln, nc;
        memcpy(&ln, nd + RX_LLEN, 2);
        memcpy(&nc, nd + RX_NCH, 2);
        uint32_t lab = toks_ld32(nd + RX_LAB), ent = toks_ld32(nd + RX_ENT), k = 0;
        if (ln > avail - d || memcmp(base + lab, s + d, ln) != 0) { break; }
        d += ln;
        if (ent != 0u) { best = d; *entry = ent - 1u; }
        if (d == avail) { break; }
        while (k < nc && base[lab + ln + k] != s[d]) { k++; }   /* bound: nc */
        if (k == nc) { break; }
        off = toks_ld32(nd + RX_CH) + 16u * k;
    }
    return best;
}

uint64_t toks_k1_added_find_c(const toks_tables *t, toks_k1_args *a)
{
    const uint8_t *text = a->text;
    uint64_t len = a->len, pos = a->pos;
    uint64_t phase = a->phase, n = 0;

    a->n = 0; a->next = len;
    if (phase > 1u || (t->add_phases & (1ull << phase)) == 0u) {
        return 0;      /* no tokens in this phase (the index pointers may be NULL) */
    }

    const toks_added_entry *ents = t->add_entries;
    const uint8_t *ab = t->add_bytes;
    const uint8_t *shufti_lo = t->add_shufti + phase * 32u;        /* lo-nibble [16] */
    const uint8_t *shufti_hi = t->add_shufti + phase * 32u + 16u;  /* hi-nibble [16] */
    const uint64_t *index2 = t->add_index + phase * 65536u;
    const uint32_t *single = t->add_single + phase * 256u;

    for (uint64_t i = pos; i < len; i++) {             /* bound: len - pos (i++) */
        uint8_t b0 = text[i];
        if ((shufti_lo[b0 & 0x0Fu] & shufti_hi[b0 >> 4]) == 0u) { continue; }
        uint64_t best_len = 0, best_entry = 0;
        if (len - i >= 4u) {                            /* >= 4 bytes: the h4 bucket of the first four, longest first */
            uint64_t ent = t->add_index[2u * 65536u + ((uint64_t)toks_k1_h4(text + i) << 1 | phase)];
            if ((ent >> 63) != 0u) {                    /* a long bucket: its radix tree */
                best_len = k1_radix((const uint8_t *)t->add_cand, (uint32_t)ent, text + i, len - i, &best_entry);
                ent = 0u;
            }
            const uint32_t *cand = t->add_cand + (ent & 0xFFFFFFFFull);
            for (uint64_t c = 0; c < ent >> 32; c++) {  /* bound: the bucket; the first exact match is the longest */
                const toks_added_entry *x = &ents[cand[c]];
                if (x->len <= len - i && memcmp(ab + x->off, text + i, x->len) == 0) {
                    best_len = x->len;
                    best_entry = cand[c];
                    break;
                }
            }
        }
        if (best_len == 0u && len - i >= 2u) {          /* 2 or 3 bytes: index2's bucket from its end, shortest first */
            uint64_t ent = index2[(uint64_t)b0 | ((uint64_t)text[i + 1u] << 8)];
            const uint32_t *cand = t->add_cand + (ent & 0xFFFFFFFFull);
            for (uint64_t c = ent >> 32; c > 0u && ents[cand[c - 1u]].len < 4u; c--) {   /* bound: the bucket */
                const toks_added_entry *x = &ents[cand[c - 1u]];
                if (x->len <= len - i && memcmp(ab + x->off, text + i, x->len) == 0) {
                    best_len = x->len;                  /* the longest so far */
                    best_entry = cand[c - 1u];
                }
            }
        }
        if (best_len == 0u && single[b0] != 0u) {
            best_len = 1;
            best_entry = (uint64_t)(single[b0] - 1u);
        }
        if (best_len != 0u) {
            a->m[n++] = (toks_k1_match){ (uint32_t)i, (uint32_t)(i + best_len), (uint32_t)best_entry, 0u };
            i += best_len - 1u;                         /* the next search starts at the match's end */
            if (n >= a->cap) { a->next = i + 1u; break; }
        }
        /* else a false positive of the shufti mask: continue scanning */
    }
    a->n = n;
    return n;
}
