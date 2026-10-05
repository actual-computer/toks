/* tests/fuzz/fuzz_load.c: libFuzzer harness for toks_load_mem_copy / toks_load on arbitrary bytes (SPEC T6, T9;
 * load.h has the oracles: one verdict for both tiers, documented refusals, the battery on every accepted file).
 * libFuzzer's own mutations + tests/fuzz/tokenizer.dict; seeds from tests/fuzz/seeds.py. docs/fuzz.md. */
#include "load.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n) { return fz_load_one(d, n); }
