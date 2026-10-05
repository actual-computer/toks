/* tests/fuzz/fuzz_par.c: libFuzzer harness for toks_par (SPEC §2.4, §5): toks_par_encode == toks_encode, and
 * every toks_split_points cut exact (the parts, TOKS_CONTINUATION after the first, concatenate to the whole's ids
 * without post-processing), over the pinned set; gen.h's input format and generator. docs/fuzz.md. */
#include "gen.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n)
{
    fz_text tx;
    if (!fz_text_open(&tx, d, n)) { return 0; }
    fz_check_par(tx.t, tx.x, tx.len, tx.flags, tx.sel);
    fz_text_close(&tx);
    return 0;
}

size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed) { return fz_text_mutate(d, n, max, seed); }
