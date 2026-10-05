/* tests/fuzz/fuzz_load_json.c: libFuzzer harness for the readers and the compiler on structurally valid hostile
 * tokenizer files (SPEC T6, T9): fuzz_load.c's oracles (load.h) with jsonmut.h's structure-preserving mutator and
 * crossover instead of byte mutations, and fresh synthetic tokenizers (synth.h) when an input does not parse. */
#include "jsonmut.h"
#include "load.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);
size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed);
size_t LLVMFuzzerCustomCrossOver(const uint8_t *d1, size_t n1, const uint8_t *d2, size_t n2, uint8_t *out, size_t max,
                                 unsigned int seed);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n) { return fz_load_one(d, n); }

size_t LLVMFuzzerCustomMutator(uint8_t *d, size_t n, size_t max, unsigned int seed) { return jm_mutate(d, n, max, seed, NULL, 0u); }

size_t LLVMFuzzerCustomCrossOver(const uint8_t *d1, size_t n1, const uint8_t *d2, size_t n2, uint8_t *out, size_t max,
                                 unsigned int seed)
{
    if (n1 > max) { n1 = max; }
    memcpy(out, d1, n1);
    return jm_mutate(out, n1, max, seed, d2, n2);
}
