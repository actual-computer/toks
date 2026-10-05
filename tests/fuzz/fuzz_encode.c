/* tests/fuzz/fuzz_encode.c: libFuzzer harness for toks_encode on arbitrary bytes and on the class-aware generator's
 * texts, every flag combination, over the pinned set (all four algorithms; gen.h has the input format, check.h the
 * oracles: tiers, scratch states, rebinding, capacity, decode / stream / roundtrip of the ids). docs/fuzz.md. */
#include "gen.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n)
{
    fz_text tx;
    if (!fz_text_open(&tx, d, n)) { return 0; }
    fz_check_text(tx.t, 0, tx.x, tx.len, tx.flags, tx.sel);
    fz_text_close(&tx);
    return 0;
}

size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed) { return fz_text_mutate(d, n, max, seed); }
