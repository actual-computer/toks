/* tests/common/guard.c: guard-page buffers (posix mmap / mprotect, windows VirtualAlloc / VirtualProtect). */
#if !defined(_WIN32)
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#endif
#include "guard.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif

size_t guard_page_size(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
#else
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t)p : 4096u;
#endif
}

uint8_t *guard_alloc(guard_buf *g, size_t n, int where, size_t off)
{
    size_t pg = guard_page_size();
    size_t body = ((n + off + pg - 1) / pg) * pg;
    if (body == 0) body = pg;
    size_t total = body + 2 * pg;               /* no-access page, body, no-access page */
    uint8_t *base;
#if defined(_WIN32)
    base = (uint8_t *)VirtualAlloc(NULL, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!base) return NULL;
    DWORD old;
    if (!VirtualProtect(base, pg, PAGE_NOACCESS, &old) ||
        !VirtualProtect(base + pg + body, pg, PAGE_NOACCESS, &old)) {
        VirtualFree(base, 0, MEM_RELEASE);
        return NULL;
    }
#else
    void *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) return NULL;
    base = (uint8_t *)m;
    if (mprotect(base, pg, PROT_NONE) != 0 || mprotect(base + pg + body, pg, PROT_NONE) != 0) {
        munmap(m, total);
        return NULL;
    }
#endif
    g->map = base;
    g->map_len = total;
    if (where == GUARD_END) return base + pg + body - n;
    return base + pg + off;
}

uint8_t *guard_map(guard_buf *g, size_t n, int readable)
{
    if (n == 0) return NULL;
#if defined(_WIN32)
    void *m = readable ? VirtualAlloc(NULL, n, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY)
                       : VirtualAlloc(NULL, n, MEM_RESERVE, PAGE_NOACCESS);
    if (!m) return NULL;
#else
    void *m = mmap(NULL, n, readable ? PROT_READ : PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (m == MAP_FAILED) return NULL;
#endif
    g->map = m;
    g->map_len = n;
    return (uint8_t *)m;
}

int guard_open(guard_buf *g, size_t off, size_t n)
{
    size_t pg = guard_page_size();
    size_t a = off & ~(pg - 1), b = (off + n + pg - 1) & ~(pg - 1);
    size_t end = (g->map_len + pg - 1) & ~(pg - 1);
    if (g->map == NULL || n == 0 || b > end) return -1;
#if defined(_WIN32)
    return VirtualAlloc((uint8_t *)g->map + a, b - a, MEM_COMMIT, PAGE_READWRITE) != NULL ? 0 : -1;
#else
    return mprotect((uint8_t *)g->map + a, b - a, PROT_READ | PROT_WRITE) == 0 ? 0 : -1;
#endif
}

void guard_free(guard_buf *g)
{
    if (!g->map) return;
#if defined(_WIN32)
    VirtualFree(g->map, 0, MEM_RELEASE);
#else
    munmap(g->map, g->map_len);
#endif
    g->map = NULL;
}
