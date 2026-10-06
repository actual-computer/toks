/* tests/common/guard.h: guard-page buffers for kernel geometry tests (SPEC §14.3) and the abi checker. */
#ifndef TOKS_TEST_GUARD_H
#define TOKS_TEST_GUARD_H

#include <stddef.h>
#include <stdint.h>

typedef struct guard_buf {
    void  *map;
    size_t map_len;
} guard_buf;

enum { GUARD_END = 0,     /* [p, p + n) ends exactly where a no-access page begins */
       GUARD_START = 1 }; /* the byte before p + off is no-access (off = 0) or accessible padding (off > 0) */

size_t   guard_page_size(void);
/* GUARD_END: returns p with p + n flush against a no-access page (off must be 0).
 * GUARD_START: returns p = (first byte after a no-access page) + off. NULL on failure. */
uint8_t *guard_alloc(guard_buf *g, size_t n, int where, size_t off);
/* n bytes of address space, never touched here: no access (readable 0: any read faults) or read-only zero pages
 * (readable 1: only the pages read are ever backed). For calls that must refuse a length before reading a byte,
 * or read a few pages of a huge input. NULL on failure; free with guard_free. */
uint8_t *guard_map(guard_buf *g, size_t n, int readable);
/* [off, off + n) of a guard_map'd region (widened to whole pages) made readable and writable; 0 on success */
int      guard_open(guard_buf *g, size_t off, size_t n);
void     guard_free(guard_buf *g);

/* abi checker (tests/common/abicheck_<isa>.S): calls fn(a0, a1) with canaries in every callee-saved
 * register; *report gets a bit per clobbered register (0 = clean). Returns fn's result. */
uint64_t toks_abicheck_call(const void *fn, uint64_t a0, uint64_t a1, uint64_t *report);

#if defined(TOKS_GUARD)
/* the guard geometry (make test-guard, docs/testing.md; tests/c/test_guard.c): the tables mapped so far, the moves
 * toks_tab_fit made (each leaves its bound-sized table mapped beside the exact one), each table's first byte and extent
 * (its bytes and its declared pad), the table holding q (0: none), and the regions of a scratch whose header is h, as
 * the geometry placed them */
size_t   guard_tabs(void);
size_t   guard_fits(void);
int      guard_tab(size_t i, const uint8_t **p, uint64_t *n);
int      guard_tab_of(const void *q, const uint8_t **p, uint64_t *n);
uint32_t guard_regions(const void *h, const uint8_t **p, uint64_t *n, uint32_t max);
#endif

/* deterministic test bytes */
static inline uint64_t guard_rng(uint64_t *s)
{
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 33;
}

/* aligned test buffers: c11 aligned_alloc + free, or the ucrt's _aligned_malloc + _aligned_free (msvc's c
 * runtime has no aligned_alloc). Free what guard_aligned_alloc returned with guard_aligned_free only. */
#if defined(_WIN32)
#  include <malloc.h>
static inline void *guard_aligned_alloc(size_t align, size_t n) { return _aligned_malloc(n, align); }
static inline void guard_aligned_free(void *p) { _aligned_free(p); }
#else
#  include <stdlib.h>
static inline void *guard_aligned_alloc(size_t align, size_t n) { return aligned_alloc(align, n); }
static inline void guard_aligned_free(void *p) { free(p); }
#endif

#endif
