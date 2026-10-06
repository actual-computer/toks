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
 *  - regress: the load fuzzers' repros (tests/fuzz/regress/expect.txt: file, the code it loads with). Small
 *    files used to run the parse arena out (TOKS_E_NOMEM): the generic engine's compile memory was taken from it.
 *  - diag: toks_diag.what is NUL-terminated within its 248 bytes (toks.h): a 600-byte path that does not open gives
 *    TOKS_E_OPEN with the first 247 bytes of the path, and nothing past the struct is written (it ends at a guard page).
 *  - hf's shape: what hf tokenizers 0.23.2 refuses before a model reads the file (a top-level key outside its nine, a
 *    version other than "1.0", a declared field given twice, an enum object of other than one key, a post-processor
 *    every variant refuses) is TOKS_E_FORMAT naming the object; what hf reads last-wins loads, and a post-processor a
 *    duplicate turns into hf's next variant gives hf's ids (tests/data/hfshape, checked against hf by its gen.py).
 *  - limits and arguments (toks.h's limits, toks_load_opts, toks_load): rsv 1 and data NULL with a length are
 *    TOKS_E_ARG; a source of 256 MiB + 1 is TOKS_E_LIMIT before a byte of it is read (a no-access mapping); a text of
 *    exactly 2^29 bytes passes the length check (encode and pieces then want a bigger scratch, split_points plans it
 *    over read-only zero pages); tokenizer.json at each limit loads and one past it is TOKS_E_LIMIT: a 65535-byte
 *    vocabulary token (a unigram piece of 65536 is refused too; unigram's own limit: 127 loads, 128 is -3), a 255-byte
 *    added token (matched by encode), and a vocabulary id of 2^21 - 1 is refused, dense or with holes; a Mistral
 *    tekken.json is not read (its directory has no model, the file itself no "model"). The json fixtures are
 *    tests/data/compile/gpt2style.json, tests/data/spm/holes_added.json and tests/data/unigram/bound_bf_meta.json with
 *    one entry spliced in.
 *  - arena huge pages (linux): a 4 MiB toks_plat_arena read first (as toks_scratch_init reads a scratch's header
 *    before it writes), then written, is all huge pages, where the host gives a written-first madvised mapping huge
 *    pages at all (else SKIP: THP off, or no huge page free). It can fail only where a write fault on the huge zero
 *    page splits the frame: linux 5.8 through 6.12 (mm/huge_memory.c's do_huge_pmd_wp_page; up to 5.7 and from 6.13
 *    the write fault allocates a huge page instead), and only with use_zero_page 1 (else the read fault itself
 *    allocates one). There a read first, without the arena's own first write, leaves 2 MiB small until khugepaged
 *    collapses it (max_ptes_none permitting), and its scan (every 10 s by default) does not come within the test's
 *    milliseconds. A short count is retried once (a huge page can fail to allocate at that instant on a shared
 *    host); the line names the kernel.
 *  - scratch frames (linux, where the arena check's written-first probe got huge pages, else SKIP): a scratch made as
 *    par.c's scratch_fit makes a participant's first one (toks_plat_arena, then toks_scratch_init with flags 0, sized
 *    for a 1 MiB text: PAR_SCR_MIN) and one encode of 1 MiB of text through it; every whole 2 MiB frame the encode
 *    touched (mincore: a resident page) is a huge page, AnonHugePages of its mapping = 2048 kB x the touched frames,
 *    exactly (a frame read first through the huge zero page counts as touched and not huge). SKIP when the scratch's
 *    mapping is merged with a neighbour (nothing to attribute); a short count is retried once, as the arena check.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#  define _DEFAULT_SOURCE 1     /* linux: mmap's MAP_ANONYMOUS and madvise under -std=c17 */
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
#elif defined(__linux__)
#  include <sys/mman.h>
#  include <sys/utsname.h>
#  include <unistd.h>
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

/* what hf tokenizers 0.23.2 refuses before any model reads the file, toks refuses as well (config.c hf_refuses): a
 * top-level key outside hf's nine, a version other than the string "1.0", a declared field given twice in an
 * added_tokens entry, truncation or padding, padding's strategy of other than one key, and a post-processor every
 * variant of hf's untagged enum refuses (Roberta and Bert: sep and cls once each; ByteLevel and Sequence: their fields
 * once, their own "type" once; Template: its fields once, one-key pieces with id and type_id once, special_tokens
 * entries with id, ids and tokens once), a Sequence element included. What hf reads last-wins loads (a repeated
 * top-level key, a repeated special_tokens key, an undeclared field given twice, no version at all), and where a
 * duplicate makes one variant refuse, hf takes the next in its order and so does toks (config.c pp_kind): 'a' gives
 * hf's ids. A template naming a special token its special_tokens map lacks: hf loads the file and panics on every
 * encode that adds special tokens, toks refuses it at load and says so. tests/data/hfshape/gen.py writes the
 * fixtures and checks each against hf (--check: the verdict, the post-processor hf builds and its ids for 'a'). */
static void test_hf_shape(void)
{
#define HFS(f) "tests/data/hfshape/" f
#define PPR "post_processor: every variant hf tries refuses it"
    static const struct { const char *path; int64_t want; const char *what; uint32_t n; uint32_t ids[3]; } F[] = {
        { HFS("refuse_top_key.json"), TOKS_E_FORMAT, "a top-level key other than version", 0u, { 0 } },
        { HFS("refuse_version_1_1.json"), TOKS_E_FORMAT, "version is not the string \"1.0\"", 0u, { 0 } },
        { HFS("refuse_version_number.json"), TOKS_E_FORMAT, "version is not the string \"1.0\"", 0u, { 0 } },
        { HFS("refuse_added_twice.json"), TOKS_E_FORMAT, "an added_tokens entry gives 'special' twice", 0u, { 0 } },
        { HFS("refuse_truncation_twice.json"), TOKS_E_FORMAT, "truncation gives 'max_length' twice", 0u, { 0 } },
        { HFS("refuse_padding_twice.json"), TOKS_E_FORMAT, "padding gives 'direction' twice", 0u, { 0 } },
        { HFS("refuse_padding_strategy.json"), TOKS_E_FORMAT, "padding: strategy is an object of other than one key", 0u, { 0 } },
        { HFS("refuse_post_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_bytelevel_type_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_roberta_sep_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_bert_cls_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_template_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_template_piece.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_template_piece_field.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_special_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_sequence_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_sequence_type_twice.json"), TOKS_E_FORMAT, PPR, 0u, { 0 } },
        { HFS("refuse_sequence_element.json"), TOKS_E_FORMAT, "post_processor.processors: an element every variant", 0u, { 0 } },
        { HFS("accept_top_twice.json"), 0, NULL, 1u, { 97u } },
        { HFS("accept_special_key_twice.json"), 0, NULL, 2u, { 97u, 261u } },
        { HFS("accept_added_unknown.json"), 0, NULL, 1u, { 97u } },
        { HFS("accept_no_version.json"), 0, NULL, 1u, { 97u } },
        { HFS("accept_template.json"), 0, NULL, 2u, { 97u, 261u } },
        { HFS("accept_roberta_trim_twice.json"), 0, NULL, 3u, { 261u, 97u, 261u } },   /* hf: BertProcessing */
        { HFS("accept_template_sep_twice.json"), 0, NULL, 2u, { 97u, 261u } },         /* hf: TemplateProcessing */
        { HFS("accept_bytelevel_sep_twice.json"), 0, NULL, 1u, { 97u } },              /* hf: ByteLevel */
        { HFS("accept_bytelevel_template.json"), 0, NULL, 2u, { 97u, 261u } },         /* hf: TemplateProcessing */
        { HFS("accept_sequence_template_twice.json"), 0, NULL, 1u, { 97u } },          /* hf: Sequence */
        { HFS("panic_template_missing.json"), TOKS_E_FORMAT, "panics on every encode that adds special tokens", 0u, { 0 } },
    };
#undef PPR
#undef HFS
    for (uint32_t i = 0; i < sizeof F / sizeof F[0]; i++) {         /* bound: 29 */
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
        if (r == 0 && F[i].n != 0u) {                               /* hf's ids for 'a', its post-processor included */
            uint32_t ids[8];
            uint64_t sb = toks_scratch_bytes(c, 64u, 0u);
            void *scr = sb != 0u ? malloc((size_t)sb) : NULL;
            int64_t ne = (scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0) ? toks_encode(c, "a", 1u, 0u, ids, 8u, scr) : -100;
            free(scr);
            uint32_t k = 0;
            while (ne == (int64_t)F[i].n && k < F[i].n && ids[k] == F[i].ids[k]) { k++; }   /* bound: 3 */
            CHECK(k == F[i].n && ne == (int64_t)F[i].n, "%s: 'a' gives %" PRId64 " ids, want %u (hf), differing at %u",
                  F[i].path, ne, F[i].n, k);
        }
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
    /* the last id, 2^21 - 2, loads (L4c): spm_build.c refused n_ids == TOKS_MAX_IDS, one id early. It maps to its bytes
     * and back, and toks_info's n_ids is TOKS_MAX_IDS */
    toks_ctx *hc = NULL;
    r = load_spliced(holes, VOC, sizeof VOC - 1u, "\"zz\": 2097150, ", 1u << 20, &hc, &d);
    CHECK(r == 0, "holes, vocab id 2^21 - 2: %" PRId64 " (%s)", r, r < 0 ? d.what : "");
    if (r == 0) {
        uint64_t zl = 0;
        const uint8_t *zp = toks_token(hc, 2097150u, &zl);
        toks_info hi;
        memset(&hi, 0, sizeof hi);
        hi.size = (uint32_t)sizeof hi;
        CHECK(zp != NULL && zl == 2u && memcmp(zp, "zz", 2) == 0 && toks_token_to_id(hc, "zz", 2u) == 2097150 &&
              toks_get_info(hc, &hi) == 0 && hi.n_ids == TOKS_MAX_IDS, "holes, id 2^21 - 2: token %s, n_ids %u",
              zp != NULL ? "found" : "NULL", hi.n_ids);
        toks_unload(hc);
    }
    toks_unload(g);
    free(gpt2);
    free(holes);
    free(uni);
}

#if defined(__linux__)
/* the kB of AnonHugePages in the mapping holding p (/proc/self/smaps), -1 when not found */
static long huge_kb(const void *p)
{
    FILE *f = fopen("/proc/self/smaps", "r");
    char ln[512];
    int in = 0;
    long kb = -1;
    unsigned long a, b;
    while (f != NULL && fgets(ln, sizeof ln, f) != NULL) {
        if (sscanf(ln, "%lx-%lx ", &a, &b) == 2 && strchr(ln, '-') != NULL && strchr(ln, '-') < strchr(ln, ' ')) {
            in = (uintptr_t)p >= a && (uintptr_t)p < b;
        } else if (in && strncmp(ln, "AnonHugePages:", 14) == 0) {
            kb = atol(ln + 14);
        }
    }
    if (f != NULL) { fclose(f); }
    return kb;
}

/* a fresh 4 MiB arena read first (as toks_scratch_init's binding check), then written: its kB on huge pages, -2
 * when it could not be had */
static long arena_read_first_kb(size_t n)
{
    uint8_t *a = toks_plat_arena(n);
    if (a == NULL) { return -2; }
    volatile uint8_t r = a[0];
    (void)r;
    memset(a, 1, n);
    long k = huge_kb(a);
    toks_plat_arena_free(a, n);
    return k;
}

/* the arena's first 2 MiB frame is a huge page even when its user's first access is a read: the file header says
 * where this can fail and where it cannot. 1 when the host gives a written-first madvised mapping huge pages */
static int test_arena_huge(void)
{
    const size_t n = (size_t)4u << 20, hp = (size_t)2u << 20;
    const long full = (long)(n >> 10);
    struct utsname u;
    const char *rel = uname(&u) == 0 ? u.release : "?";
    long zp = -1;
    FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/use_zero_page", "r");
    if (f != NULL) { if (fscanf(f, "%ld", &zp) != 1) { zp = -1; } fclose(f); }
    uint8_t *m = mmap(NULL, n + hp, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { printf("SKIP arena huge pages: mmap failed\n"); return 0; }
    uint8_t *q = (uint8_t *)(((uintptr_t)m + hp - 1u) & ~(uintptr_t)(hp - 1u));
    long kq = madvise(q, n, MADV_HUGEPAGE) == 0 ? (memset(q, 1, n), huge_kb(q)) : -1;
    munmap(m, n + hp);
    if (kq < full) {
        printf("SKIP arena huge pages: linux %s gave a written-first madvised 4 MiB %ld kB of them\n", rel, kq);
        return 0;
    }
    long ka = arena_read_first_kb(n);
    int retried = ka >= 0 && ka < full;
    if (retried) { ka = arena_read_first_kb(n); }      /* once: a huge page can fail to allocate at that instant */
    CHECK(ka != -2, "arena(4 MiB)");
    CHECK(ka >= full, "arena(4 MiB) read first, then written: %ld kB on huge pages, want %ld (linux %s, use_zero_page %ld%s)",
          ka, full, rel, zp, retried ? ", retried once" : "");
    printf("arena huge pages: %ld of %ld kB after a read first (linux %s, use_zero_page %ld, the written-first probe %ld kB%s)\n",
           ka, full, rel, zp, kq, retried ? ", retried once" : "");
    return 1;
}

/* toks_par's scratch as par.c's scratch_fit makes it (toks_plat_arena, toks_scratch_init) for a 1 MiB text, after
 * one encode: n_frames of its n_whole whole 2 MiB frames touched (mincore: a resident page), and the kB of
 * AnonHugePages of its mapping; -1 when the mapping is not the arena's alone (merged with a neighbour: nothing to
 * attribute) */
static long scratch_frames_kb(const toks_ctx *ctx, const uint8_t *text, uint64_t len, uint32_t *ids, long *n_frames,
                              long *n_whole)
{
    const uint64_t hp = 2u << 20, bytes = toks_scratch_bytes(ctx, len, 0u);
    uint8_t *m = toks_plat_arena(bytes);
    long kb = -1;
    *n_frames = 0, *n_whole = (long)(bytes / hp);
    if (m == NULL || toks_scratch_init(ctx, m, bytes, 0u) != 0 || toks_encode(ctx, text, len, 0u, ids, len + 16u, m) <= 0) {
        toks_plat_arena_free(m, bytes);
        return -2;
    }
    unsigned char vec[512];
    for (uint64_t at = 0; at + hp <= bytes; at += hp) {          /* bound: bytes / 2 MiB frames */
        int touched = 0;
        if (mincore(m + at, (size_t)hp, vec) == 0) {
            for (size_t i = 0; !touched && i < sizeof vec; i++) { touched = (vec[i] & 1u) != 0u; }   /* bound: 512 */
        }
        *n_frames += touched;
    }
    FILE *f = fopen("/proc/self/smaps", "r");
    char ln[512];
    int in = 0;
    unsigned long a, b, pg = (unsigned long)sysconf(_SC_PAGESIZE);
    unsigned long end = (unsigned long)(uintptr_t)m + (unsigned long)((bytes + pg - 1u) & ~(uint64_t)(pg - 1u));
    while (f != NULL && fgets(ln, sizeof ln, f) != NULL) {
        if (sscanf(ln, "%lx-%lx ", &a, &b) == 2 && strchr(ln, '-') != NULL && strchr(ln, '-') < strchr(ln, ' ')) {
            in = (uintptr_t)m >= a && (uintptr_t)m < b;
            if (in && (a != (unsigned long)(uintptr_t)m || b != end)) { in = 0, kb = -1; break; }
        } else if (in && strncmp(ln, "AnonHugePages:", 14) == 0) {
            kb = atol(ln + 14);
        }
    }
    if (f != NULL) { fclose(f); }
    toks_plat_arena_free(m, bytes);
    return kb;
}

/* every whole frame of a fresh toks_par scratch that one encode touches is a huge page: the arena's own first write
 * keeps frame 0 (test_arena_huge), and this pins that the scratch writes each later frame before it reads it, which
 * a splitting kernel (the file header) needs as much */
static void test_scratch_frames(void)
{
    const uint64_t len = 1u << 20;                    /* par.c PAR_SCR_MIN: a participant's first scratch */
    toks_ctx *ctx = NULL;
    uint8_t *text = malloc((size_t)len);
    uint32_t *ids = malloc(4u * (size_t)(len + 16u));
    CHECK(text != NULL && ids != NULL && toks_load(&ctx, "tests/data/compile/llama3style.json", NULL) == 0, "scratch frames: setup");
    if (ctx == NULL || text == NULL || ids == NULL) { free(text), free(ids); return; }
    static const char *const W[] = { "the ", "of ", "and ", "tokenizer ", "frame ", "huge ", "page, ", "write\n", "read. ",
                                     "1984 ", "x86 ", "(arena) ", "  ", "zero ", "\xC3\xA9t\xC3\xA9 " };
    uint64_t s = 0x746F6B73u;
    for (uint64_t i = 0; i < len;) {                  /* words in a fixed pseudo-random order */
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const char *w = W[(s >> 33) % (sizeof W / sizeof W[0])];
        for (size_t j = 0; w[j] != '\0' && i < len; j++) { text[i++] = (uint8_t)w[j]; }
    }
    long frames = 0, whole = 0, kb = scratch_frames_kb(ctx, text, len, ids, &frames, &whole);
    int retried = kb >= 0 && kb < 2048 * frames;
    if (retried) { kb = scratch_frames_kb(ctx, text, len, ids, &frames, &whole); }   /* once, as test_arena_huge */
    CHECK(kb != -2, "scratch frames: the scratch and its encode");
    if (kb == -1) {
        printf("SKIP scratch frames: the scratch's mapping is merged with another\n");
    } else if (kb >= 0) {
        CHECK(kb == 2048 * frames, "a fresh scratch after one encode: %ld kB on huge pages, want %ld (%ld touched frames%s)",
              kb, 2048 * frames, frames, retried ? ", retried once" : "");
        printf("scratch frames: %ld of the %ld frames one encode touched on huge pages (%ld whole frames)%s\n", kb / 2048,
               frames, whole, retried ? ", retried once" : "");
    }
    toks_unload(ctx);
    free(text), free(ids);
}
#endif

int main(void)
{
    test_cycles();
    test_info();
    test_regress();
    test_refuse_unigram_prefix();
    test_hf_shape();
    test_uni_resolve_ids();
    test_diag();
    test_limits();
#if defined(__linux__)
    if (test_arena_huge()) {
        test_scratch_frames();
    } else {
        printf("SKIP scratch frames: no huge pages here (above)\n");
    }
#endif
    printf("test_load: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
