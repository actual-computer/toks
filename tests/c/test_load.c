/*
 * test_load.c: the context lifecycle through the real load path (toks_load_mem_copy).
 *  - address space: 2000 load + unload cycles of tests/data/compile/nosplit.json (two arenas, of 1472 and
 *    288,260 bytes: neither a page multiple) and 2000 toks_plat_arena + toks_plat_arena_free cycles of odd
 *    sizes leave the process's virtual size flat (VmSize on linux, task_info on macos, reserved + committed
 *    address space from GlobalMemoryStatusEx on windows). Every arena goes
 *    back whole: mem.c maps, trims and frees whole pages. Before, the tail trim at a page-unaligned
 *    address failed and up to 2 MiB per arena stayed mapped (+3.7 GiB over 2000 loads on the mac).
 *  - every arena is 2 MiB-aligned and writable over its whole length.
 *  - info: cpu_features is the word the tier was picked from at load (toks_cpu_features, called here by
 *    the test; the library calls it once, at load).
 *  - entropy: toks_plat_entropy fills exactly what it is given (a key flush against a guard page, odd address;
 *    n = 0 writes nothing), draws differ, about half the bits are set, a 1 MiB draw repeats no 256-byte block.
 *  - regress: the load fuzzers' repros (tests/fuzz/regress/expect.txt: file, the code it loads with). Small
 *    files used to run the parse arena out (TOKS_E_NOMEM): the generic engine's compile memory was taken from it.
 *  - diag: toks_diag.what is NUL-terminated within its 248 bytes (toks.h): a 600-byte path that does not open gives
 *    TOKS_E_OPEN with the first 247 bytes of the path, and nothing past the struct is written (it ends at a guard page).
 *  - limits and arguments (toks.h's limits, toks_load_opts, toks_load): rsv 1 and data NULL with a length are
 *    TOKS_E_ARG; a source of 256 MiB + 1 is TOKS_E_LIMIT before a byte of it is read (a no-access mapping); a text of
 *    exactly 2^29 bytes passes the length check (encode and pieces then want a bigger scratch, split_points plans it
 *    over read-only zero pages); tokenizer.json at each limit loads and one past it is TOKS_E_LIMIT: a 65535-byte
 *    vocabulary token (a unigram piece of 65536 is refused too; unigram's own limit: 127 loads, 128 is -3), a 255-byte
 *    added token (matched by encode), and a vocabulary id of 2^21 - 1 is refused, dense or with holes; a Mistral
 *    tekken.json is not read (its directory has no model, the file itself no "model"). The json fixtures are
 *    tests/data/compile/gpt2style.json, tests/data/spm/holes_added.json and tests/data/unigram/bound_bf_meta.json with
 *    one entry spliced in.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
#include "cpu.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lines.inc"
#if defined(__APPLE__)
#  include <mach/mach.h>
#elif defined(_WIN32)
#  include <windows.h>
#endif

static int failures;
static long checks;

/* under AddressSanitizer: no quarantine, else the allocator's held-back frees (~280 MiB over the cycles)
 * would read as address-space growth */
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
const char *__asan_default_options(void);
const char *__asan_default_options(void) { return "quarantine_size_mb=0"; }
#  endif
#endif

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* the process's virtual size in bytes, 0 when this os has no probe here */
static uint64_t vsize(void)
{
#if defined(__APPLE__)
    mach_task_basic_info_data_t i;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&i, &n) != KERN_SUCCESS) { return 0; }
    return (uint64_t)i.virtual_size;
#elif defined(__linux__)
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) { return 0; }
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof line, f) != NULL) {
        if (strncmp(line, "VmSize:", 7) == 0) { kb = strtoull(line + 7, NULL, 10); break; }
    }
    fclose(f);
    return kb * 1024u;
#elif defined(_WIN32)
    MEMORYSTATUSEX m = { .dwLength = (DWORD)sizeof m };   /* reserved + committed: the arena's slack shows */
    return GlobalMemoryStatusEx(&m) ? (uint64_t)(m.ullTotalVirtual - m.ullAvailVirtual) : 0u;
#else
    return 0;
#endif
}

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n + 1u);
    if (b == NULL || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

#define CYCLES 2000
#define SLACK  (16ull << 20)    /* allowed growth: malloc's own caches; the leak was ~2-4 MiB per cycle */

static void test_cycles(void)
{
    uint64_t len = 0;
    uint8_t *json = slurp("tests/data/compile/nosplit.json", &len);
    CHECK(json != NULL, "tests/data/compile/nosplit.json (run from the source root)");
    if (json == NULL) { return; }
    for (int i = 0; i < 8; i++) {                       /* warm malloc up before the first sample */
        toks_ctx *c = NULL;
        CHECK(toks_load_mem_copy(&c, json, len, NULL) == 0 && c != NULL, "load");
        toks_unload(c);
    }
    uint64_t v0 = vsize();
    int ok = 1;
    for (int i = 0; i < CYCLES && ok; i++) {
        toks_ctx *c = NULL;
        ok = toks_load_mem_copy(&c, json, len, NULL) == 0 && c != NULL;
        toks_unload(c);
    }
    uint64_t v1 = vsize();
    CHECK(ok, "a load failed during the cycles");
    if (v0 == 0u) {
        printf("SKIP address space: no virtual-size probe on this os\n");
    } else {
        CHECK(v1 <= v0 + SLACK, "%d load/unload cycles grew the address space by %.1f MiB", CYCLES,
              (double)(int64_t)(v1 - v0) / 1048576.0);
        printf("%d load/unload cycles: address space %+.1f MiB\n", CYCLES, (double)(int64_t)(v1 - v0) / 1048576.0);
    }
    free(json);

    /* the arenas themselves: odd sizes, 2 MiB-aligned, writable to the last byte, returned whole */
    static const uint64_t N[] = { 1u, 4095u, 4097u, 16385u, 65537u, (2u << 20) - 1u, (2u << 20) + 1u, (3u << 20) + 123u };
    v0 = vsize();
    for (int i = 0; i < CYCLES; i++) {
        uint64_t n = N[i % (int)(sizeof N / sizeof N[0])];
        uint8_t *p = toks_plat_arena(n);
        CHECK(p != NULL && ((uintptr_t)p & 0x1FFFFFu) == 0u, "arena(%" PRIu64 ") = %p", n, (void *)p);
        if (p == NULL) { break; }
        p[0] = 1;
        p[n - 1u] = 2;
        toks_plat_arena_free(p, n);
    }
    v1 = vsize();
    if (v0 != 0u) {
        CHECK(v1 <= v0 + SLACK, "%d arena cycles grew the address space by %.1f MiB", CYCLES,
              (double)(int64_t)(v1 - v0) / 1048576.0);
        printf("%d arena cycles: address space %+.1f MiB\n", CYCLES, (double)(int64_t)(v1 - v0) / 1048576.0);
    }
    CHECK(toks_plat_arena(0u) == NULL, "arena(0)");
    toks_plat_arena_free(NULL, 0u);
}

static void test_info(void)
{
    uint64_t len = 0;
    uint8_t *json = slurp("tests/data/compile/llama3style.json", &len);
    CHECK(json != NULL, "tests/data/compile/llama3style.json");
    if (json == NULL) { return; }
    toks_ctx *c = NULL;
    CHECK(toks_load_mem_copy(&c, json, len, NULL) == 0 && c != NULL, "load llama3style");
    if (c != NULL) {
        toks_info info;
        memset(&info, 0, sizeof info);
        info.size = (uint32_t)sizeof info;
        CHECK(toks_get_info(c, &info) == 0 && info.cpu_features == toks_cpu_features(),
              "info.cpu_features %#" PRIx64 " is not the load-time word %#" PRIx64, info.cpu_features, toks_cpu_features());
        toks_unload(c);
    }
    free(json);
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* toks_plat_entropy, the os's randomness (file.c): a 4,144-byte key (the memo check's) flush against a guard page,
 * from an odd address; n = 0 writes nothing; two draws differ; about half the bits are set (33,152: 16,576 +- 546,
 * six sigma); a 1 MiB draw (4,096 of macos's 256-byte calls) repeats no 256-byte block (each block's first 8 bytes) */
static void test_entropy(void)
{
    enum { KEY = 4144, BIG = 1 << 20, BLK = 256 };
    guard_buf g;
    uint8_t *p = guard_alloc(&g, KEY + 1u, GUARD_END, 0);
    static uint8_t a[KEY];
    CHECK(p != NULL, "guard_alloc");
    if (p == NULL) { return; }
    memset(p, 0x5A, KEY + 1u);
    CHECK(toks_plat_entropy(p + 1, 0u) == 0, "entropy n = 0");
    int kept = 1;
    for (int i = 0; i <= KEY; i++) { kept &= p[i] == 0x5A; }
    CHECK(kept, "entropy n = 0 wrote a byte");
    CHECK(toks_plat_entropy(p + 1, KEY) == 0 && p[0] == 0x5A, "entropy: a %d-byte key at an odd address", KEY);
    memcpy(a, p + 1, KEY);
    CHECK(toks_plat_entropy(p + 1, KEY) == 0 && memcmp(a, p + 1, KEY) != 0, "entropy: two draws are equal");
    long ones = 0;
    for (int i = 0; i < KEY; i++) { ones += __builtin_popcount(a[i]); }
    CHECK(ones > 16576 - 546 && ones < 16576 + 546, "entropy: %ld of 33152 bits set", ones);
    guard_free(&g);
    uint8_t *m = (uint8_t *)malloc(BIG);
    uint64_t *k = (uint64_t *)malloc((BIG / BLK) * sizeof *k);
    CHECK(m != NULL && k != NULL && toks_plat_entropy(m, BIG) == 0, "entropy: 1 MiB");
    if (m != NULL && k != NULL) {
        for (int i = 0; i < BIG / BLK; i++) { memcpy(&k[i], m + (size_t)i * BLK, 8u); }
        qsort(k, BIG / BLK, sizeof *k, cmp_u64);
        int rep = 0;
        for (int i = 1; i < BIG / BLK; i++) { rep += k[i] == k[i - 1]; }
        CHECK(rep == 0, "entropy: 1 MiB repeats %d of its 256-byte blocks", rep);
    }
    free(m);
    free(k);
    printf("entropy: a %d-byte key, two draws, %ld of 33152 bits set; 1 MiB without a repeated block\n", KEY, ones);
}

static void test_regress(void)
{
    uint64_t len = 0;
    uint8_t *list = slurp("tests/fuzz/regress/expect.txt", &len);
    CHECK(list != NULL, "tests/fuzz/regress/expect.txt (run from the source root)");
    if (list != NULL && !td_text("tests/fuzz/regress/expect.txt", list, len)) { failures++; free(list); return; }
    char path[256];
    long want = 0;
    int n = 0;
    for (uint64_t i = 0; list != NULL && i < len;) {                   /* one line a pass */
        char *line = (char *)list + i;
        uint64_t e = i;
        while (e < len && list[e] != '\n') { e++; }
        list[e] = 0;                                                   /* slurp leaves a byte past the end */
        i = e + 1u;
        if (line[0] == '#' || sscanf(line, "%255s %ld", path, &want) != 2) { continue; }
        uint64_t fl = 0;
        uint8_t *f = slurp(path, &fl);
        toks_ctx *ctx = NULL;
        toks_diag dg;
        toks_load_opts o = { sizeof o, 0, 0, 0, &dg };
        int64_t r = f != NULL ? toks_load_mem_copy(&ctx, f, fl, &o) : 1;
        CHECK(r == want && r != TOKS_E_NOMEM, "%s: %" PRId64 " (%s), want %ld", path, r, r < 0 ? dg.what : "", want);
        toks_unload(ctx);
        free(f);
        n++;
    }
    CHECK(n >= 19, "regress: %d repros", n);
    free(list);
}

/* a reason longer than toks_diag.what: cut at 247 bytes and terminated, the struct flush against a no-access page */
static void test_diag(void)
{
    guard_buf g;
    toks_diag *d = (toks_diag *)(void *)guard_alloc(&g, sizeof(toks_diag), GUARD_END, 0);
    CHECK(d != NULL, "guard_alloc");
    if (d == NULL) { return; }
    char path[600];
    memset(path, 'p', sizeof path);
    memcpy(path, "missing-dir/", 12);
    path[sizeof path - 1u] = 0;
    memset(d, 0x5A, sizeof *d);
    toks_load_opts o = { sizeof o, 0, 0, 0, d };
    toks_ctx *c = (toks_ctx *)(uintptr_t)1;
    int64_t r = toks_load(&c, path, &o);
    size_t k = 0;
    while (k < sizeof d->what && d->what[k] != 0) { k++; }      /* bound: 248 */
    CHECK(r == TOKS_E_OPEN && c == NULL && d->code == TOKS_E_OPEN && k == sizeof d->what - 1u &&
          memcmp(d->what, path, k) == 0, "a 599-byte path: %" PRId64 ", diag.code %" PRId64 ", what %zu bytes, want 247",
          r, d->code, k);
    guard_free(&g);
}

/* a unigram file whose only U+2581 handling is the prefix Replace '(?<!\n)^' -> U+2581 is refused at load (the
 * prefix would encode as a plain space; found by the bound review, 2026-10-05); with ' ' -> U+2581 beside it
 * (llm-jp's shape) it loads. */
static void test_refuse_unigram_prefix(void)
{
    static const struct { const char *path; int64_t want; const char *what; } F[] = {
        { "tests/data/unigram/refuse_prefix_only.json", TOKS_E_UNSUPPORTED, "without ' ' -> '" },
        { "tests/data/unigram/accept_prefix_space.json", 0, NULL },
    };
    for (uint32_t i = 0; i < 2u; i++) {                             /* bound: 2 */
        uint64_t len = 0;
        uint8_t *json = slurp(F[i].path, &len);
        CHECK(json != NULL, "%s (run from the source root)", F[i].path);
        if (json == NULL) { continue; }
        toks_diag dg; memset(&dg, 0, sizeof dg);
        toks_load_opts o; memset(&o, 0, sizeof o);
        o.size = (uint32_t)sizeof o; o.diag = &dg;
        toks_ctx *c = NULL;
        int64_t r = toks_load_mem_copy(&c, json, len, &o);
        CHECK(r == F[i].want, "%s: %" PRId64 " (%s), want %" PRId64, F[i].path, r, r < 0 ? dg.what : "", F[i].want);
        if (F[i].what != NULL) { CHECK(r < 0 && strstr(dg.what, F[i].what) != NULL, "%s: diag '%s'", F[i].path, dg.what); }
        toks_unload(c);
        free(json);
    }
}

/* uni_resolve's id buffer (unigram.c): a queued piece is <= 15 bytes plus the virtual U+2581 and byte fallback under
 * the ' ' -> U+2581 remap spells each space as three ids, so 15 spaces are 48 ids, the most one piece gives. The T6
 * campaign found the buffer at 32 (UBSan unigram.c:571, 2026-10-05): 11 spaces gave 33 ids whose 33rd was garbage on
 * a release build. The fixture is the minimized model (unk, the three U+2581 bytes, 'a'; the llm-jp prefix + space
 * Replace, byte_fallback), hf-tokenizers 0.23.2 loads it, and its ids are a rule, checked against hf for every n
 * in 0..20: n spaces are (1 2 3) x (n + 1) for n >= 1 and nothing for n = 0; 'a' then n spaces are 1 2 3 4 then
 * (1 2 3) x n; n spaces then 'a' are (1 2 3) x (n + 1) then 4. n = 16..20 take the direct path (a piece over 15
 * bytes), 15 fills the buffer exactly. */
static void test_uni_resolve_ids(void)
{
    uint64_t len = 0;
    uint8_t *json = slurp("tests/data/unigram/byte_fallback_spaces.json", &len);
    CHECK(json != NULL, "tests/data/unigram/byte_fallback_spaces.json (run from the source root)");
    if (json == NULL) { return; }
    toks_diag dg; memset(&dg, 0, sizeof dg);
    toks_load_opts o; memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o; o.diag = &dg;
    toks_ctx *c = NULL;
    int64_t r = toks_load_mem_copy(&c, json, len, &o);
    CHECK(r == 0, "byte_fallback_spaces: %" PRId64 " (%s)", r, r < 0 ? dg.what : "");
    uint64_t sb = c != NULL ? toks_scratch_bytes(c, 64u, 0u) : 0u;
    void *scr = c != NULL ? malloc(sb) : NULL;
    CHECK(c == NULL || (scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0), "a scratch");
    for (uint32_t shape = 0; c != NULL && scr != NULL && shape < 3u; shape++) {       /* bound: 3 */
        for (uint32_t n = 0; n <= 20u; n++) {                                        /* bound: 21 */
            char text[24];
            uint32_t want[80], nw = 0, tl = 0;
            if (shape == 1u) { text[tl++] = 'a'; want[nw++] = 1u; want[nw++] = 2u; want[nw++] = 3u; want[nw++] = 4u; }
            for (uint32_t j = 0; j < n; j++) { text[tl++] = ' '; }
            if (shape == 2u) { text[tl++] = 'a'; }
            uint32_t runs = shape == 1u ? n : n + 1u;                                 /* U+2581s spelled as bytes */
            if (tl == 0u) { runs = 0u; }                                              /* the empty text: nothing */
            for (uint32_t j = 0; j < runs; j++) { want[nw++] = 1u; want[nw++] = 2u; want[nw++] = 3u; }
            if (shape == 2u) { want[nw++] = 4u; }
            uint32_t ids[80];
            memset(ids, 0xAA, sizeof ids);
            int64_t ne = toks_encode(c, text, tl, 0u, ids, 80u, scr);
            uint32_t bad = 0;
            while (ne == (int64_t)nw && bad < nw && ids[bad] == want[bad]) { bad++; }     /* bound: nw */
            CHECK(ne == (int64_t)nw && bad == nw, "shape %u, %u spaces: %" PRId64 " ids, want %u (hf); ids differ at %u",
                  shape, n, ne, nw, bad);
            CHECK(ne < 0 || (uint64_t)ne <= toks_encode_bound(c, tl), "shape %u, %u spaces: the bound", shape, n);
        }
    }
    free(scr);
    toks_unload(c);
    free(json);
}

/* base with `ins` written at the one occurrence of anchor plus skip bytes (a NUL-terminated copy; NULL if anchor is
 * not there exactly once). pad: spaces appended after the json (its parse arena grows with the source: 32 B a byte) */
static char *splice(const char *base, const char *anchor, size_t skip, const char *ins, size_t pad, uint64_t *len)
{
    const char *a = strstr(base, anchor);
    if (a == NULL || strstr(a + 1, anchor) != NULL || skip > strlen(anchor)) { return NULL; }
    size_t pre = (size_t)(a - base) + skip, il = strlen(ins), rest = strlen(base) - pre;
    char *o = (char *)malloc(pre + il + rest + pad + 1u);
    if (o == NULL) { return NULL; }
    memcpy(o, base, pre);
    memcpy(o + pre, ins, il);
    memcpy(o + pre + il, base + pre, rest);
    memset(o + pre + il + rest, ' ', pad);
    o[pre + il + rest + pad] = 0;
    *len = pre + il + rest + pad;
    return o;
}

/* loads base with ins spliced in after anchor (or skip bytes into it): the load's code and its diag; the context
 * goes to *keep when keep is not NULL, else it is unloaded */
static int64_t load_spliced(const char *base, const char *anchor, size_t skip, const char *ins, size_t pad, toks_ctx **keep,
                            toks_diag *d)
{
    uint64_t len = 0;
    char *j = splice(base, anchor, skip, ins, pad, &len);
    CHECK(j != NULL, "splice at %s", anchor);
    if (j == NULL) { return 1; }
    memset(d, 0, sizeof *d);
    toks_load_opts o = { sizeof o, 0, 0, 0, d };
    toks_ctx *c = (toks_ctx *)(uintptr_t)1;
    int64_t r = toks_load_mem_copy(&c, j, len, &o);
    CHECK((r == 0) == (c != NULL), "load: %" PRId64 " with a context %p", r, (void *)c);
    if (keep != NULL && r == 0) { *keep = c; } else { toks_unload(r == 0 ? c : NULL); }
    free(j);
    return r;
}

static char *repeat(char ch, size_t n, const char *pre, const char *post)
{
    size_t a = strlen(pre), b = strlen(post);
    char *s = (char *)malloc(a + n + b + 1u);
    if (s == NULL) { return NULL; }
    memcpy(s, pre, a);
    memset(s + a, ch, n);
    memcpy(s + a + n, post, b + 1u);
    return s;
}

static void test_limits(void)
{
    toks_diag d;
    toks_load_opts o = { sizeof o, 0, 0, 0, &d };
    toks_ctx *c = (toks_ctx *)(uintptr_t)1;
    uint64_t glen = 0, hlen = 0, ulen = 0;
    char *gpt2 = (char *)slurp("tests/data/compile/gpt2style.json", &glen);
    char *holes = (char *)slurp("tests/data/spm/holes_added.json", &hlen);
    char *uni = (char *)slurp("tests/data/unigram/bound_bf_meta.json", &ulen);
    CHECK(gpt2 != NULL && holes != NULL && uni != NULL, "the json fixtures (run from the source root)");
    if (gpt2 == NULL || holes == NULL || uni == NULL) { free(gpt2); free(holes); free(uni); return; }
    gpt2[glen] = 0, holes[hlen] = 0, uni[ulen] = 0;          /* slurp leaves a byte past the end */

    /* arguments: rsv, and data NULL with a length (toks.h: TOKS_E_ARG), *out NULL, diag set */
    o.rsv = 1u;
    memset(&d, 0, sizeof d);
    CHECK(toks_load_mem_copy(&c, gpt2, glen, &o) == TOKS_E_ARG && c == NULL && d.code == TOKS_E_ARG, "rsv 1");
    o.rsv = 0u;
    c = (toks_ctx *)(uintptr_t)1;
    memset(&d, 0, sizeof d);
    CHECK(toks_load_mem_copy(&c, NULL, 5u, &o) == TOKS_E_ARG && c == NULL && d.code == TOKS_E_ARG, "data NULL, len 5");

    /* Mistral tekken.json is not read: its directory holds no model toks reads, the file no tokenizer.json model */
    c = (toks_ctx *)(uintptr_t)1;
    int64_t r = toks_load(&c, "tests/data/tekken", &o);
    CHECK(r == TOKS_E_OPEN && c == NULL, "a directory with only tekken.json: %" PRId64 " (%s)", r, d.what);
    c = (toks_ctx *)(uintptr_t)1;
    r = toks_load(&c, "tests/data/tekken/tekken.json", &o);
    CHECK(r == TOKS_E_FORMAT && c == NULL && d.code == TOKS_E_FORMAT, "tekken.json: %" PRId64 " (%s)", r, d.what);

    /* TOKS_MAX_SOURCE_BYTES: 256 MiB + 1 refused before a byte is read (the mapping has no access) */
    guard_buf gm;
    uint8_t *big = guard_map(&gm, (size_t)TOKS_MAX_SOURCE_BYTES + 1u, 0);
    CHECK(big != NULL, "guard_map 256 MiB + 1");
    if (big != NULL) {
        c = (toks_ctx *)(uintptr_t)1;
        memset(&d, 0, sizeof d);
        r = toks_load_mem_copy(&c, big, TOKS_MAX_SOURCE_BYTES + 1u, &o);
        CHECK(r == TOKS_E_LIMIT && c == NULL && d.code == TOKS_E_LIMIT, "a source of 256 MiB + 1: %" PRId64, r);
        guard_free(&gm);
    }

    /* TOKS_MAX_TEXT: exactly 2^29 bytes pass the length check. encode and pieces on a scratch for 64 bytes say
     * TOKS_E_SCRATCH (the text is longer than the scratch, checked after the length), reading no text byte (no
     * access); split_points plans it over read-only zero pages (it reads at most n_want x (2D + W) bytes) */
    toks_ctx *g = NULL;
    CHECK(toks_load_mem_copy(&g, gpt2, glen, NULL) == 0 && g != NULL, "load gpt2style");
    uint8_t *t29 = guard_map(&gm, (size_t)TOKS_MAX_TEXT + 1u, 0);
    uint64_t sb = g != NULL ? toks_scratch_bytes(g, 64u, 0u) : 0u;
    void *scr = sb != 0u ? malloc((size_t)sb) : NULL;
    CHECK(t29 != NULL && scr != NULL && toks_scratch_init(g, scr, sb, 0u) == 0, "a 2^29 + 1 mapping, a scratch");
    if (g != NULL && t29 != NULL && scr != NULL) {
        uint32_t ids[4];
        uint64_t offs[4];
        CHECK(toks_encode(g, t29, TOKS_MAX_TEXT, 0u, ids, 4u, scr) == TOKS_E_SCRATCH, "encode 2^29: the scratch");
        CHECK(toks_pieces(g, t29, TOKS_MAX_TEXT, 0u, ids, 4u, scr) == TOKS_E_SCRATCH, "pieces 2^29: the scratch");
        CHECK(toks_encode(g, t29, TOKS_MAX_TEXT + 1u, 0u, ids, 4u, scr) == TOKS_E_LIMIT, "encode 2^29 + 1");
        CHECK(toks_pieces(g, t29, TOKS_MAX_TEXT + 1u, 0u, ids, 4u, scr) == TOKS_E_LIMIT, "pieces 2^29 + 1");
        CHECK(toks_split_points(g, t29, TOKS_MAX_TEXT + 1u, 0u, 2u, offs, 4u, NULL) == TOKS_E_LIMIT, "split_points 2^29 + 1");
        guard_free(&gm);
        t29 = guard_map(&gm, (size_t)TOKS_MAX_TEXT, 1);
        CHECK(t29 != NULL, "a readable 2^29 mapping");
        r = t29 != NULL ? toks_split_points(g, t29, TOKS_MAX_TEXT, 0u, 2u, offs, 4u, NULL) : 0;
        CHECK(r >= 0 && r <= 1, "split_points over 2^29 zero bytes: %" PRId64, r);
    }
    if (t29 != NULL) { guard_free(&gm); }
    free(scr);

    /* TOKS_MAX_TOKEN_BYTES: a 65535-byte vocabulary token loads (id 262: gpt2style's vocab is 0..261) and is found;
     * 65536 is TOKS_E_LIMIT, as a unigram piece of 65536 bytes is (last in its array) */
    static const char VOC[] = "\"vocab\": {", UNI[] = "]], \"byte_fallback\"";
    for (uint32_t n = 65535u; n <= 65536u; n++) {           /* bound: 2 lengths */
        char *tok = repeat('a', n, "\"", "\": 262, ");
        char *piece = repeat('q', n, ", [\"", "\", -20.0]");
        toks_ctx *k = NULL;
        uint64_t tl = 0;
        r = tok != NULL ? load_spliced(gpt2, VOC, sizeof VOC - 1u, tok, 0u, &k, &d) : 1;
        if (n == 65535u) {
            const uint8_t *ts = k != NULL ? toks_token(k, 262u, &tl) : NULL;
            CHECK(r == 0 && ts != NULL && tl == n && ts[0] == 'a' && ts[n - 1u] == 'a' && toks_token_to_id(k, tok + 1, n) == 262,
                  "a 65535-byte vocab token: %" PRId64 " (%s), %" PRIu64 " bytes", r, d.what, tl);
        } else {
            CHECK(r == TOKS_E_LIMIT && strstr(d.what, "TOKS_MAX_TOKEN_BYTES") != NULL, "a 65536-byte vocab token: %" PRId64 " (%s)",
                  r, d.what);
        }
        toks_unload(k);
        if (n == 65536u) {                                  /* unigram refuses a piece over 127 bytes (-3, named): */
            r = piece != NULL ? load_spliced(uni, UNI, 1u, piece, 0u, NULL, &d) : 1;   /* only the limit's side here */
            CHECK(r == TOKS_E_LIMIT && strstr(d.what, "TOKS_MAX_TOKEN_BYTES") != NULL, "a 65536-byte unigram piece: %" PRId64 " (%s)",
                  r, d.what);
        }
        free(tok);
        free(piece);
    }

    /* toks.h's "unigram pieces: 127, -3 above": a 127-byte piece loads (id 274), a 128-byte one is TOKS_E_UNSUPPORTED */
    for (uint32_t n = 127u; n <= 128u; n++) {               /* bound: 2 lengths */
        char *piece = repeat('q', n, ", [\"", "\", -20.0]");
        toks_ctx *k = NULL;
        uint64_t tl = 0;
        r = piece != NULL ? load_spliced(uni, UNI, 1u, piece, 0u, &k, &d) : 1;
        const uint8_t *ps = (r == 0 && k != NULL) ? toks_token(k, 274u, &tl) : NULL;
        if (n == 127u) {
            CHECK(r == 0 && ps != NULL && tl == n, "a 127-byte unigram piece: %" PRId64 " (%s), %" PRIu64 " bytes", r, d.what, tl);
        } else {
            CHECK(r == TOKS_E_UNSUPPORTED && strstr(d.what, "127 bytes") != NULL, "a 128-byte unigram piece: %" PRId64 " (%s)", r,
                  d.what);
        }
        toks_unload(k);
        free(piece);
    }

    /* TOKS_MAX_ADDED_BYTES: a 255-byte added token loads (id 262) and encode matches it; 256 is TOKS_E_LIMIT */
    for (uint32_t n = 255u; n <= 256u; n++) {
        char *add = repeat('q', n - 2u, "{\"id\": 262, \"content\": \"<", ">\", \"single_word\": false, \"lstrip\": false, "
                           "\"rstrip\": false, \"normalized\": false, \"special\": false}, ");
        toks_ctx *k = NULL;
        static const char ADD[] = "\"added_tokens\": [";
        r = add != NULL ? load_spliced(gpt2, ADD, sizeof ADD - 1u, add, 0u, &k, &d) : 1;
        if (n == 255u) {
            char *text = repeat('q', n - 2u, "x <", ">y");
            uint32_t ids[16];
            int64_t ne = (k != NULL && text != NULL) ? toks_encode(k, text, strlen(text), 0u, ids, 16u, NULL) : -100;
            uint64_t sbk = k != NULL ? toks_scratch_bytes(k, 512u, 0u) : 0u;
            void *sk = sbk != 0u ? malloc((size_t)sbk) : NULL;
            if (sk != NULL && toks_scratch_init(k, sk, sbk, 0u) == 0 && text != NULL) {
                ne = toks_encode(k, text, strlen(text), 0u, ids, 16u, sk);
            }
            int found = 0;
            for (int64_t i = 0; i < ne && i < 16; i++) { found += ids[i] == 262u; }
            CHECK(r == 0 && found == 1 && (toks_id_flags(k, 262u) & TOKS_ID_ADDED) != 0,
                  "a 255-byte added token: %" PRId64 " (%s), encode %" PRId64 " ids, %d matches", r, d.what, ne, found);
            free(sk);
            free(text);
        } else {
            CHECK(r == TOKS_E_LIMIT && strstr(d.what, "TOKS_MAX_ADDED_BYTES") != NULL, "a 256-byte added token: %" PRId64 " (%s)", r, d.what);
        }
        toks_unload(k);
        free(add);
    }

    /* TOKS_MAX_IDS: every id is < 2^21 - 1. A vocabulary id of 2^21 - 1 is TOKS_E_LIMIT, in a dense vocab (gpt2style)
     * and in one with holes (holes_added: sentencepiece-style; 1 MiB of trailing spaces gives its parse arena room) */
    r = load_spliced(gpt2, VOC, sizeof VOC - 1u, "\"zz\": 2097151, ", 0u, NULL, &d);
    CHECK(r == TOKS_E_LIMIT && strstr(d.what, "model.vocab id >= TOKS_MAX_IDS") != NULL, "vocab id 2^21 - 1: %" PRId64 " (%s)", r, d.what);
    r = load_spliced(holes, VOC, sizeof VOC - 1u, "\"zz\": 2097151, ", 1u << 20, NULL, &d);
    CHECK(r == TOKS_E_LIMIT && strstr(d.what, "model.vocab id >= TOKS_MAX_IDS") != NULL, "holes, vocab id 2^21 - 1: %" PRId64 " (%s)",
          r, d.what);
    toks_unload(g);
    free(gpt2);
    free(holes);
    free(uni);
}

int main(void)
{
    test_cycles();
    test_info();
    test_entropy();
    test_regress();
    test_refuse_unigram_prefix();
    test_uni_resolve_ids();
    test_diag();
    test_limits();
    printf("test_load: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
