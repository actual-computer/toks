/* tests/fuzz/arprobe.h: the arena probe of the fuzz build (docs/fuzz.md §5.7). tests/fuzz/Makefile force-includes
 * it (-include) into the src/core objects of build/fuzz-*, never into the library's own build: every toks_ar_alloc
 * call reports its site, the arena's bound (len), its position and the request (n, align) to fz_ar_probe
 * (arprobe.c), which keeps per site the calls, the refusals and the least slack, bound - (aligned start + n), seen.
 * `tests/fuzz/run.sh probe` replays the corpora and prints the table. */
#ifndef TOKS_FUZZ_ARPROBE_H
#define TOKS_FUZZ_ARPROBE_H

#include "core.h"

void fz_ar_probe(const char *file, uint32_t line, uint64_t len, uint64_t pos, uint64_t n, uint64_t align, int got);

static inline void *fz_ar_alloc_at(toks_arena *a, uint64_t n, uint64_t align, const char *file, uint32_t line)
{
    uint64_t len = a->len, pos = a->pos;
    void *r = toks_ar_alloc(a, n, align);
    fz_ar_probe(file, line, len, pos, n, align, r != NULL);
    return r;
}

/* every later use in the object is the probed one (the definition above keeps the real call) */
#define toks_ar_alloc(a, n, al) fz_ar_alloc_at((a), (n), (al), __FILE__, (uint32_t)__LINE__)

#endif
