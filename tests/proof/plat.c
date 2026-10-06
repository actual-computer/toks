/*
 * tests/proof/plat.c: the platform layer as the proof package assumes it (SPEC §14.1: external contracts).
 * src/platform is outside the analyzed core; these models give each toks_plat_* function exactly the
 * contract core.h states, with every allowed outcome (allocation failure included):
 *   toks_plat_alloc(n)      NULL, or a fresh block of n bytes (malloc: may fail; n == 0 -> NULL)
 *   toks_plat_arena(n)      NULL, or a fresh block of n bytes (zeroed: mmap'd anonymous memory)
 *   toks_plat_*free         releases it
 *   toks_plat_hint_huge     no effect on memory contents (advice only)
 *   toks_plat_getenv        -1 (unset), -2 (longer than cap - 1, buf[0] = 0), or the length n < cap of a
 *                           NUL-terminated string of unknown bytes in buf (core.h, docs/notes/c-core.md §core.h.5)
 *   toks_plat_read_file     TOKS_E_OPEN / TOKS_E_LIMIT / TOKS_E_NOMEM (out NULL, *is_dir unknown), or 0 with a
 *                           fresh block of 1..PROOF_SRC_MAX unknown bytes
 *   toks_plat_dir_lookup    TOKS_E_OPEN, or 0 with a NUL-terminated path in buf[0, cap)
 *   toks_cpu_features       any bit set
 */
#include "prelude.h"
#include "core.h"
#include "cpu.h"

void *toks_plat_alloc(uint64_t n)
{
    if (n == 0u) { return NULL; }
    return malloc((size_t)n);
}

void toks_plat_free(void *p, uint64_t n)
{
    (void)n;
    free(p);
}

uint8_t *toks_plat_arena(uint64_t n)
{
    if (n == 0u) { return NULL; }
    uint8_t *p = (uint8_t *)malloc((size_t)n);
    if (p != NULL) { memset(p, 0, (size_t)n); }
    return p;
}

void toks_plat_arena_free(uint8_t *p, uint64_t n)
{
    (void)n;
    free(p);
}

void toks_plat_hint_huge(void *p, uint64_t n)
{
    (void)p;
    (void)n;
}

int64_t toks_plat_getenv(const char *name, char *buf, uint64_t cap)
{
    (void)name;
    if (cap == 0u) { return -2; }
    int r = Frama_C_interval(0, 2);
    if (r == 0) { return -1; }
    if (r == 1) { buf[0] = 0; return -2; }
    uint64_t n = Frama_C_unsigned_long_long_interval(1u, cap - 1u);
    Frama_C_make_unknown(buf, (size_t)n);
    buf[n] = 0;
    return (int64_t)n;
}

int64_t toks_plat_read_file(const char *path, uint8_t **out, uint64_t *len, int *is_dir)
{
    (void)path;
    *out = NULL; *len = 0; *is_dir = Frama_C_nondet(0, 1);
    int r = Frama_C_interval(0, 3);
    if (r == 1) { return TOKS_E_OPEN; }
    if (r == 2) { return TOKS_E_LIMIT; }
    if (r == 3) { return TOKS_E_NOMEM; }
    *is_dir = 0;
    uint64_t n = Frama_C_unsigned_long_long_interval(1u, PROOF_SRC_MAX);
    uint8_t *buf = (uint8_t *)malloc((size_t)n);
    if (buf == NULL) { return TOKS_E_NOMEM; }
    Frama_C_make_unknown((char *)buf, (size_t)n);
    *out = buf; *len = n;
    return 0;
}

int64_t toks_plat_dir_lookup(const char *dir, char *buf, uint64_t cap)
{
    (void)dir;
    if (cap == 0u || Frama_C_nondet(0, 1)) { return TOKS_E_OPEN; }
    Frama_C_make_unknown(buf, (size_t)cap - 1u);
    buf[cap - 1u] = 0;
    return 0;
}

uint64_t toks_cpu_features(void)
{
    uint64_t f = Frama_C_unsigned_long_long_interval(0u, UINT64_MAX);
    return f;
}
