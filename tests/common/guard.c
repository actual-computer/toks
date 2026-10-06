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

#if defined(TOKS_GUARD)
/* ---- the guard geometry (docs/testing.md): the hooks the guard build's library calls (core.h) -------------------------
 * Every table a builder takes (toks_tab, toks_tab_ar) and every region toks_scratch_init carves is a mapping of its own:
 * a no-access page, the body, a no-access page, with the table or region flush against the page after it (TOKS_GUARD 1)
 * or the page before it (TOKS_GUARD 2). A table's mapping is kept under its owner (the block or arena it was carved
 * from) and unmapped when toks_plat_arena_free frees that block (toks_guard_release); a scratch's under its header,
 * kept while the same buffer is laid out the same way again, unmapped when another init or a free covers the header. */
#if defined(_WIN32)
#  error "the guard build (TOKS_GUARD) is posix only"
#endif
#include "core.h"
#include "norm.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GS_MAX   12u                       /* regions a scratch is carved into, at most */
#define GS_MAGIC 0x3144524155474B54ull     /* "TKGUARD1": the region table in the caller's buffer */

typedef struct gtab { const void *owner; uint8_t *map; size_t len; const uint8_t *p; uint64_t n; const uint8_t *at; } gtab;
typedef struct gblk { const void *p; uint64_t n; } gblk;
typedef struct gscr { const toks_scratch *h; uint32_t n; uint64_t off[GS_MAX], len[GS_MAX]; uint8_t *p[GS_MAX];
                      uint8_t *map[GS_MAX]; size_t mlen[GS_MAX]; } gscr;
/* the copy toks_guard_scr_at reads, in the caller's buffer right after the header (the ends array's place in
 * production: the guard build maps ends with the other regions) */
typedef struct gscr_tab { uint64_t magic; uint32_t n, rsv; struct { uint64_t off, len; uint8_t *p; } r[GS_MAX]; } gscr_tab;
_Static_assert(sizeof(gscr_tab) <= 4u * TOKS_CHUNK_PIECES, "the region table fits where ends lives in production");

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static gtab *g_tab;
static size_t g_ntab, g_captab;
static gscr *g_scr;
static size_t g_nscr, g_capscr;
static gblk *g_blk;                        /* the blocks mem.c's toks_plat_arena mapped: the ones a seal may close */
static size_t g_nblk, g_capblk;
static uint8_t *g_poison;                  /* 64 MiB of no-access address space: what an offset in no region maps to */

static void gfail(const char *what)
{
    fprintf(stderr, "guard: %s\n", what);
    abort();
}

static void *grow(void *a, size_t *cap, size_t each, size_t first)   /* under g_mu */
{
    *cap = *cap ? 2 * *cap : first;
    a = realloc(a, *cap * each);
    if (!a) gfail("out of memory");
    return a;
}

/* n bytes, start aligned to align, on their own pages: flush against the no-access page after them (1) or before (2) */
static uint8_t *gmap(size_t n, size_t align, uint8_t **map, size_t *len)
{
    size_t pg = guard_page_size();
    size_t body = n == 0 ? pg : (n + pg - 1) / pg * pg;
    uint8_t *b = (uint8_t *)mmap(NULL, body + 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if ((void *)b == MAP_FAILED) gfail("mmap failed");
    if (mprotect(b, pg, PROT_NONE) != 0 || mprotect(b + pg + body, pg, PROT_NONE) != 0) gfail("mprotect failed");
    *map = b;
    *len = body + 2 * pg;
#if TOKS_GUARD == 1
    return (uint8_t *)(((uintptr_t)(b + pg + body) - n) & ~((uintptr_t)align - 1u));
#else
    (void)align;
    return b + pg;
#endif
}

/* a table of n bytes (its pad included) the shipped build places at at (NULL: a copy, toks_guard_fit's) */
void *toks_guard_tab(const void *owner, const void *at, uint64_t n, uint64_t align)
{
    gtab t = { owner, NULL, 0, NULL, n, (const uint8_t *)at };
    uint8_t *p = gmap((size_t)n, (size_t)align, &t.map, &t.len);
    t.p = p;
    pthread_mutex_lock(&g_mu);
    if (g_ntab == g_captab) g_tab = (gtab *)grow(g_tab, &g_captab, sizeof *g_tab, 256);
    g_tab[g_ntab++] = t;
    pthread_mutex_unlock(&g_mu);
    return p;
}

void toks_guard_block(const void *block, uint64_t n)
{
    pthread_mutex_lock(&g_mu);
    if (g_nblk == g_capblk) g_blk = (gblk *)grow(g_blk, &g_capblk, sizeof *g_blk, 256);
    g_blk[g_nblk++] = (gblk){ block, n };
    pthread_mutex_unlock(&g_mu);
}

static int gat_cmp(const void *a, const void *b)
{
    const gtab *x = *(const gtab *const *)a, *y = *(const gtab *const *)b;
    return (x->at > y->at) - (x->at < y->at);
}

/* a block its builder has carved. First its shipped placement: every table carved from it lies inside it and no two
 * overlap (a hand-computed offset that runs past the block or into the next table fails here, though each table has
 * its own pages in this build). Then no access from here on, so a table pointer that did not come from toks_tab
 * faults; only a block mem.c mapped (a test's stand-in allocator gives heap memory, which must stay as it is). */
void toks_guard_seal(void *block, uint64_t n)
{
    size_t pg = guard_page_size(), k = 0;
    const uint8_t *lo = (const uint8_t *)block, *hi = lo + n;
    int ours = 0;
    pthread_mutex_lock(&g_mu);
    const gtab **v = (const gtab **)malloc((g_ntab + 1) * sizeof *v);
    if (!v) gfail("out of memory");
    for (size_t i = 0; i < g_ntab; i++) {
        const gtab *t = &g_tab[i];
        if ((const uint8_t *)t->owner < lo || (const uint8_t *)t->owner >= hi || t->at == NULL) continue;
        if (t->at < lo || t->n > (uint64_t)(hi - t->at)) {
            fprintf(stderr, "guard: a table of %llu bytes at offset %lld runs past its block of %llu\n", (unsigned long long)t->n,
                    (long long)(t->at - lo), (unsigned long long)n);
            abort();
        }
        v[k++] = t;
    }
    qsort(v, k, sizeof *v, gat_cmp);
    for (size_t i = 1; i < k; i++) {
        if (v[i - 1]->at + v[i - 1]->n > v[i]->at) {
            fprintf(stderr, "guard: tables overlap in their block: [%lld, +%llu) and [%lld, +%llu)\n", (long long)(v[i - 1]->at - lo),
                    (unsigned long long)v[i - 1]->n, (long long)(v[i]->at - lo), (unsigned long long)v[i]->n);
            abort();
        }
    }
    free(v);
    for (size_t i = 0; i < g_nblk && !ours; i++) ours = g_blk[i].p == block && g_blk[i].n == n;
    pthread_mutex_unlock(&g_mu);
    if (ours && mprotect(block, ((size_t)n + pg - 1) / pg * pg, PROT_NONE) != 0) gfail("seal failed");
}

/* the block the table starting at p was carved from (decode frees its slots' block through it) */
uint8_t *toks_guard_owner(const void *p)
{
    const void *o = NULL;
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_ntab && o == NULL; i++) o = g_tab[i].p == (const uint8_t *)p ? g_tab[i].owner : NULL;
    pthread_mutex_unlock(&g_mu);
    return (uint8_t *)(uintptr_t)(o != NULL ? o : p);
}

/* p's table, its first n bytes moved to a table of exactly n (the same owner): its end is the contents' end */
const void *toks_guard_fit(const void *p, uint64_t n, uint64_t align)
{
    const void *owner = NULL;
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_ntab && owner == NULL; i++) owner = g_tab[i].p == (const uint8_t *)p ? g_tab[i].owner : NULL;
    pthread_mutex_unlock(&g_mu);
    if (owner == NULL) return p;                       /* not a guard table (a test's own): as it is */
    uint8_t *q = (uint8_t *)toks_guard_tab(owner, NULL, n, align);
    memcpy(q, p, (size_t)n);
    return q;
}

static void gscr_drop(size_t i)            /* under g_mu */
{
    for (uint32_t r = 0; r < g_scr[i].n; r++) munmap(g_scr[i].map[r], g_scr[i].mlen[r]);
    g_scr[i] = g_scr[--g_nscr];
}

void toks_guard_release(const void *block, uint64_t n)
{
    uintptr_t lo = (uintptr_t)block, hi = lo + (uintptr_t)n;
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_ntab;) {
        uintptr_t o = (uintptr_t)g_tab[i].owner;
        if (o >= lo && o < hi) { munmap(g_tab[i].map, g_tab[i].len); g_tab[i] = g_tab[--g_ntab]; } else i++;
    }
    for (size_t i = 0; i < g_nscr;) {
        uintptr_t o = (uintptr_t)g_scr[i].h;
        if (o >= lo && o < hi) gscr_drop(i); else i++;
    }
    for (size_t i = 0; i < g_nblk; i++) {
        if (g_blk[i].p == block) { g_blk[i] = g_blk[--g_nblk]; break; }
    }
    pthread_mutex_unlock(&g_mu);
}

/* the regions toks_scratch_init carved (api.c, core.h's layout), each as its readers use it: wordpiece's work holds its
 * pieces and copies after toks_scr_work (wp_api.c: one region), unigram's runs through extra and the bounce
 * (uni_api.c: one region), the generic engine's lists are extra (gen.c); the bounce is the ids K5 may write
 * (tmax + 4), the norm region x max_len bytes */
static uint32_t gscr_regions(const toks_ctx *ctx, const toks_scratch *h, uint32_t x, uint64_t *off, uint64_t *len,
                             uint64_t *al)
{
    uint32_t k = 0;
#define GS_ADD(o, l, a) do { if ((l) != 0) { off[k] = (o); len[k] = (l); al[k] = (a); k++; } } while (0)
    uint64_t c = h->off_cache, n = h->cache_mib, caches = toks_scr_caches(n), tmax = toks_scr_tmax(h->max_len, x);
    uint64_t bounce = 4u * (tmax + 4u), nb = toks_scr_long_buckets(n);
    GS_ADD(c - 4u * TOKS_CHUNK_PIECES, 4u * TOKS_CHUNK_PIECES, 4u);
    if (h->off_long != 0) {
        GS_ADD(c, toks_scr_short(n), 64u);
        GS_ADD(h->off_long, nb, 64u);
        GS_ADD(h->off_long + nb, toks_scr_long_arena(n), 64u);
    } else {
        GS_ADD(c, caches, 64u);
    }
    GS_ADD(c + caches, h->off_work - c - caches, 64u);                       /* the memo */
    if (ctx->uni != NULL) {
        GS_ADD(h->off_work, h->off_bounce - h->off_work + toks_scr_bounce(tmax), 64u);
    } else {
        uint64_t extra = ctx->wp != NULL ? 0u : ctx->scr_extra;
        GS_ADD(h->off_work, h->off_bounce - h->off_work - extra, 64u);
        GS_ADD(h->off_bounce - extra, extra, 8u);
        GS_ADD(h->off_bounce, bounce, 4u);
    }
    if (x != 0) GS_ADD(h->off_bounce + toks_scr_bounce(tmax), toks_scr_norm(h->max_len, x) != 0 ? x * h->max_len : 0u, 1u);
#undef GS_ADD
    return k;
}

void toks_guard_scr(const toks_ctx *ctx, toks_scratch *h)
{
    uint32_t x = ctx->nfc != 0u ? TOKS_NORM_X(ctx->nfc) : ctx->has_drop != 0u ? TOKS_NFC_X : 0u;   /* api.c scr_x */
    uint64_t off[GS_MAX], len[GS_MAX], al[GS_MAX];
    uint32_t n = gscr_regions(ctx, h, x, off, len, al);
    uintptr_t lo = (uintptr_t)h->base, hi = lo + (uintptr_t)h->bytes;
    pthread_mutex_lock(&g_mu);
    if (g_poison == NULL) {
        g_poison = (uint8_t *)mmap(NULL, 64u << 20, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if ((void *)g_poison == MAP_FAILED) gfail("mmap failed");
    }
    gscr *s = NULL;
    for (size_t i = 0; i < g_nscr;) {      /* a scratch whose header this buffer covers is gone (or is this one) */
        uintptr_t o = (uintptr_t)g_scr[i].h;
        if (o < lo || o >= hi) { i++; continue; }
        if (g_scr[i].h == h && g_scr[i].n == n && memcmp(g_scr[i].off, off, n * 8u) == 0 &&
            memcmp(g_scr[i].len, len, n * 8u) == 0) { s = &g_scr[i]; break; }   /* laid out the same: kept */
        gscr_drop(i);
    }
    if (s == NULL) {
        if (g_nscr == g_capscr) g_scr = (gscr *)grow(g_scr, &g_capscr, sizeof *g_scr, 64);
        s = &g_scr[g_nscr++];
        s->h = h;
        s->n = n;
        for (uint32_t r = 0; r < n; r++) {
            s->off[r] = off[r];
            s->len[r] = len[r];
            s->p[r] = gmap((size_t)len[r], (size_t)al[r], &s->map[r], &s->mlen[r]);
            size_t z = len[r] < (1u << 20) ? (size_t)len[r] : (size_t)1 << 20, t4 = len[r] < 4096u ? (size_t)len[r] : 4096u;
            memset(s->p[r], 0xA5, z);                           /* a caller's buffer is not zeroed: init owes every */
            memset(s->p[r] + len[r] - t4, 0xA5, t4);            /* zero (the head and the tail: the rest stays unbacked) */
        }
    }
    gscr_tab *t = (gscr_tab *)(void *)((uint8_t *)h + TOKS_SCR_HDR);
    t->magic = GS_MAGIC;
    t->n = n;
    for (uint32_t r = 0; r < n; r++) {
        t->r[r].off = s->off[r];
        t->r[r].len = s->len[r];
        t->r[r].p = s->p[r];
    }
    pthread_mutex_unlock(&g_mu);
}

uint8_t *toks_guard_scr_at(const toks_scratch *h, uint64_t off)
{
    const gscr_tab *t = (const gscr_tab *)(const void *)((const uint8_t *)h + TOKS_SCR_HDR);
    if (t->magic != GS_MAGIC) return (uint8_t *)(uintptr_t)(h->base + off);   /* a white-box test's own header */
    for (uint32_t r = 0; r < t->n; r++) {
        if (off - t->r[r].off < t->r[r].len) return t->r[r].p + (off - t->r[r].off);
    }
    for (uint32_t r = 0; r < t->n; r++) {
        if (off == t->r[r].off + t->r[r].len) return t->r[r].p + t->r[r].len;   /* one past a region's end */
    }
    return g_poison;                       /* in no region: a pointer formed, never to be used */
}

/* [off, off + n) of the layout zeroed, region by region (init zeroes the short cache and the long buckets in one go) */
void toks_guard_scr_zero(const toks_scratch *h, uint64_t off, uint64_t n)
{
    const gscr_tab *t = (const gscr_tab *)(const void *)((const uint8_t *)h + TOKS_SCR_HDR);
    if (t->magic != GS_MAGIC) { memset((uint8_t *)(uintptr_t)(h->base + off), 0, (size_t)n); return; }
    uint64_t done = 0;
    for (uint32_t r = 0; r < t->n; r++) {
        uint64_t a = off > t->r[r].off ? off : t->r[r].off, e = t->r[r].off + t->r[r].len;
        uint64_t b = off + n < e ? off + n : e;
        if (a < b) { memset(t->r[r].p + (a - t->r[r].off), 0, (size_t)(b - a)); done += b - a; }
    }
    if (done != n) gfail("a scratch zeroed past its regions");
}

/* ---- for tests/c/test_guard.c: the tables and a scratch's regions as the geometry placed them ---------------- */
size_t guard_tabs(void)
{
    pthread_mutex_lock(&g_mu);
    size_t n = g_ntab;
    pthread_mutex_unlock(&g_mu);
    return n;
}

int guard_tab(size_t i, const uint8_t **p, uint64_t *n)
{
    pthread_mutex_lock(&g_mu);
    int ok = i < g_ntab;
    if (ok) { *p = g_tab[i].p; *n = g_tab[i].n; }
    pthread_mutex_unlock(&g_mu);
    return ok;
}

int guard_tab_of(const void *q, const uint8_t **p, uint64_t *n)
{
    int ok = 0;
    pthread_mutex_lock(&g_mu);
    for (size_t i = 0; i < g_ntab && !ok; i++) {
        ok = (const uint8_t *)q == g_tab[i].p || (uintptr_t)q - (uintptr_t)g_tab[i].p < g_tab[i].n;
        if (ok) { *p = g_tab[i].p; *n = g_tab[i].n; }
    }
    pthread_mutex_unlock(&g_mu);
    return ok;
}

uint32_t guard_regions(const void *h, const uint8_t **p, uint64_t *n, uint32_t max)
{
    const gscr_tab *t = (const gscr_tab *)(const void *)((const uint8_t *)h + TOKS_SCR_HDR);
    uint32_t k = 0;
    for (; t->magic == GS_MAGIC && k < t->n && k < max; k++) { p[k] = t->r[k].p; n[k] = t->r[k].len; }
    return k;
}
#endif
