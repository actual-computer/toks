/* mem.c: platform load-time memory (docs/notes/c-core.md §mem.c.1) */
#if !defined(_WIN32)
#  define _DEFAULT_SOURCE 1     /* madvise, MAP_ANONYMOUS: glibc hides them under -std=c17 */
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"

#if defined(_WIN32)

#include <windows.h>

void *toks_plat_alloc(uint64_t n)
{
    if (n == 0u) { return NULL; }
    return HeapAlloc(GetProcessHeap(), 0, (SIZE_T)n);
}

void toks_plat_free(void *p, uint64_t n)
{
    (void)n;
    if (p == NULL) { return; }
    HeapFree(GetProcessHeap(), 0, p);
}

/* VirtualAlloc2 (windows 10 1803+, looked up at run time) with a 2 MiB alignment requirement; without it, a
 * reservation of n + 2 MiB with only its aligned n bytes committed */
typedef void *(WINAPI *va2_fn)(HANDLE, void *, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER *, ULONG);

uint8_t *toks_plat_arena(uint64_t n)
{
    const SIZE_T hp = 0x200000u;
    if (n == 0u || n > (uint64_t)(SIZE_MAX - hp)) { return NULL; }
    va2_fn va2 = (va2_fn)(void *)GetProcAddress(GetModuleHandleA("kernelbase.dll"), "VirtualAlloc2");
    MEM_ADDRESS_REQUIREMENTS r = { NULL, NULL, hp };
    MEM_EXTENDED_PARAMETER x = { .Type = MemExtendedParameterAddressRequirements, .Pointer = &r };
    uint8_t *p = NULL;
    if (va2 != NULL) { p = (uint8_t *)va2(NULL, NULL, (SIZE_T)n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &x, 1); }
    if (p != NULL) { return p; }
    uint8_t *b = (uint8_t *)VirtualAlloc(NULL, (SIZE_T)n + hp, MEM_RESERVE, PAGE_NOACCESS);
    if (b == NULL) { return NULL; }
    p = b + ((hp - ((uintptr_t)b & (hp - 1u))) & (hp - 1u));
    if (VirtualAlloc(p, (SIZE_T)n, MEM_COMMIT, PAGE_READWRITE) != NULL) { return p; }
    VirtualFree(b, 0, MEM_RELEASE);
    return NULL;
}

/* the whole reservation p lies in (the fallback's begins below p) */
void toks_plat_arena_free(uint8_t *p, uint64_t n)
{
    MEMORY_BASIC_INFORMATION m;
    (void)n;
    if (p != NULL && VirtualQuery(p, &m, sizeof m) != 0) { VirtualFree(m.AllocationBase, 0, MEM_RELEASE); }
}

#else /* posix */

#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

void *toks_plat_alloc(uint64_t n)
{
    if (n == 0u) { return NULL; }
    size_t z = (size_t)n;
    if ((uint64_t)z != n) { return NULL; }
    return malloc(z);
}

void toks_plat_free(void *p, uint64_t n)
{
    (void)n;
    free(p);                                  /* free(NULL) does nothing */
}

/* the page size (munmap takes whole pages: a trim or free at a page-unaligned address fails) */
static size_t page_bytes(void)
{
    long sc = sysconf(_SC_PAGESIZE);
    return sc > 0 ? (size_t)sc : (size_t)4096u;
}

/* one private anonymous mapping of n bytes rounded up to whole pages, at a 2 MiB-aligned address: mapped with
 * 2 MiB of slack, then the head and the tail of the slack unmapped (whole pages, kernels.md §7) */
uint8_t *toks_plat_arena(uint64_t n)
{
    size_t pg = page_bytes();
    size_t hp = 0x200000u;                    /* 2 MiB */
    if (pg > hp) { hp = pg; }                 /* a huge page is never < a page */
    if (n == 0u || n > (uint64_t)SIZE_MAX - 2u * hp) { return NULL; }
    size_t z = ((size_t)n + pg - 1u) & ~(pg - 1u);
    size_t total = z + hp;
    void *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { return NULL; }
    uint8_t *base = (uint8_t *)m;
    size_t head = (size_t)((hp - ((uintptr_t)base & (hp - 1u))) & (hp - 1u));   /* to the next 2 MiB */
    uint8_t *p = base + head;
    if (head != 0u && munmap(base, head) != 0) { munmap(base, total); return NULL; }
    if (munmap(p + z, hp - head) != 0) { munmap(p, z + (hp - head)); return NULL; }   /* the tail */
#if defined(MADV_HUGEPAGE)
    if (madvise(p, z, MADV_HUGEPAGE) != 0) { /* best effort */ }
#endif
    return p;
}

/* n: the length toks_plat_arena was given (the mapping is its whole pages) */
void toks_plat_arena_free(uint8_t *p, uint64_t n)
{
    if (p == NULL) { return; }
    size_t pg = page_bytes();
    if (munmap(p, ((size_t)n + pg - 1u) & ~(pg - 1u)) != 0) { /* a range toks_plat_arena mapped: cannot fail */ }
}

#endif

/* huge-page advice for [p, p + n)'s 2 MiB-aligned interior before its first touch (none on windows) */
void toks_plat_hint_huge(void *p, uint64_t n)
{
#if defined(MADV_HUGEPAGE)
    uintptr_t a = ((uintptr_t)p + 0x1FFFFFu) & ~(uintptr_t)0x1FFFFFu;
    uintptr_t b = ((uintptr_t)p + (uintptr_t)n) & ~(uintptr_t)0x1FFFFFu;
    if (b > a && madvise((void *)a, (size_t)(b - a), MADV_HUGEPAGE) != 0) { /* best effort */ }
#else
    (void)p;
    (void)n;
#endif
}
