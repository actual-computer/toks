/* tests/fuzz/fuzz_decode.c: libFuzzer harness for toks_decode on arbitrary id sequences over the pinned set (ids.h
 * has the input format; check.h fz_check_decode the oracles: TOKS_E_ID / TOKS_E_ARG with nothing written, tiers,
 * the capacity rule, valid utf-8, from_utf8_lossy of the token bytes for byte-level files), and the vocabulary
 * lookups on the same ids (fz_check_vocab: toks_id_flags, toks_token_to_id of each id's string and of variants of
 * it). docs/fuzz.md. */
#include "ids.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n)
{
    fz_ids q;
    if (!fz_ids_open(&q, d, n)) { return 0; }
    fz_check_decode(q.t, q.v, q.n, q.flags, q.sel);
    fz_check_vocab(q.t, q.v, q.n < 64u ? q.n : 64u, NULL, 0u, q.sel >> 6);
    free(q.v);
    return 0;
}
