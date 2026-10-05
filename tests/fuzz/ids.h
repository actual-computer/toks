/* tests/fuzz/ids.h: the id harnesses' input format (fuzz_decode.c, fuzz_stream.c). A 4-byte header, then u32 ids
 * (little-endian):
 *   d[0]     the pinned tokenizer (fuzz.h fz_pin)
 *   d[1]     bit 0 TOKS_SKIP_SPECIAL
 *   d[2..3]  sel (capacity, partitions); its top bits come from the ids' hash
 * An id word with bit 31 set maps into [0, n_ids); a clear one into [0, n_ids] (so n_ids itself, out of range,
 * now and then). */
#ifndef TOKS_FUZZ_IDS_H
#define TOKS_FUZZ_IDS_H

#include "check.h"

typedef struct fz_ids { fz_tok *t; uint32_t flags, *v; uint64_t n, sel; } fz_ids;

FZ_FN int fz_ids_open(fz_ids *q, const uint8_t *d, size_t n)
{
    if (n < 4u) { return 0; }
    q->t = fz_pin(d[0]);
    if (q->t == NULL) { return 0; }
    q->flags = d[1] & 1u;
    q->n = (n - 4u) / 4u;
    if (q->n > 4096u) { q->n = 4096u; }
    q->sel = (uint64_t)d[2] | (uint64_t)d[3] << 8 | fz_hash(d + 4, n - 4u) << 16;
    q->v = (uint32_t *)fz_alloc(q->n * 4u);
    for (uint64_t i = 0; i < q->n; i++) {
        uint32_t w = (uint32_t)d[4 + 4 * i] | (uint32_t)d[5 + 4 * i] << 8 | (uint32_t)d[6 + 4 * i] << 16 | (uint32_t)d[7 + 4 * i] << 24;
        q->v[i] = (w >> 31) ? (w & 0x7FFFFFFFu) % q->t->info.n_ids : w % (q->t->info.n_ids + 1u);
    }
    return 1;
}

#endif /* TOKS_FUZZ_IDS_H */
