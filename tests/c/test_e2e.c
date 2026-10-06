/*
 * test_e2e.c: include/toks.h end to end on the real gpt2, llama3, GLM 5.3, Qwen 3.8 (NFC), gpt-oss (o200k) and
 * Mistral-Nemo (o200k's nemo variant) tokenizer.json files (the pins in test_e2e.py) against hf tokenizers
 * 0.23.2's own outputs (test_e2e.inc, written
 * by test_e2e.py): encode in every mode with and without the post-processor, every capacity on each
 * case, count-only calls, pieces in every mode (hf's piece ends in the normalized stream), decode with and
 * without TOKS_SKIP_SPECIAL; then
 * loading from a model directory and from memory, the scratch binding across contexts, info, forced
 * tiers and the load errors. A Qwen-1 model (qwen.tiktoken and its two files) by its directory and by its ranks file.
 *
 * The files live under $TOKS_TOKENIZER_CACHE or ~/.cache/toks/tokenizers (python3 tools/ci/fetch_tokenizers.py
 * fetches them). A missing file prints SKIP with its path and fails nothing.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "toks.h"
#include "core.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#  include <sys/stat.h>
#  include <unistd.h>
#else
#  define WIN32_LEAN_AND_MEAN                       /* no dlgs.h: its scr1..scr8 / lst1 / edt1 macros eat locals */
#  include <windows.h>
#  define setenv(k, v, o) _putenv_s((k), (v))      /* the ucrt's; "" removes the variable */
#  define unsetenv(k) _putenv_s((k), "")
#endif

#include "test_e2e.inc"

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static char *path_of(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    static char buf[1024];
    if (root != NULL && root[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", root, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
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

static const uint32_t *find_ids(const char *tok, uint32_t text, uint32_t flags, uint32_t *n)
{
    for (int k = 0; k < E2E_N; k++) {
        if (strcmp(E2E[k].tok, tok) == 0 && E2E[k].text == text && E2E[k].flags == flags) {
            *n = E2E[k].n;
            return E2E[k].v;
        }
    }
    *n = 0;
    return NULL;
}

static const uint32_t *find_pieces(const char *tok, uint32_t text, uint32_t flags, uint32_t *n)
{
    for (int k = 0; k < E2E_N_PCS; k++) {
        if (strcmp(E2E_PCS[k].tok, tok) == 0 && E2E_PCS[k].text == text && E2E_PCS[k].flags == flags) {
            *n = E2E_PCS[k].n;
            return E2E_PCS[k].v;
        }
    }
    *n = 0;
    return NULL;
}

static void run_cases(const char *name, toks_ctx *ctx, void *scr)
{
    toks_info ci;
    memset(&ci, 0, sizeof ci);
    ci.size = (uint32_t)sizeof ci;
    int ws_drop = toks_get_info(ctx, &ci) == 0 && ci.algorithm == TOKS_ALGO_UNIGRAM;
    uint32_t *out = (uint32_t *)malloc(4u * 8192u);
    for (int k = 0; k < E2E_N; k++) {
        if (strcmp(E2E[k].tok, name) != 0) { continue; }
        const char *t = E2E_TEXT[E2E[k].text];
        uint64_t len = E2E_LEN[E2E[k].text], n = E2E[k].n;
        int64_t r = toks_encode(ctx, t, len, E2E[k].flags, out, 8192u, scr);
        int ok = r == (int64_t)n && memcmp(out, E2E[k].v, (size_t)n * 4u) == 0;
        CHECK(ok, "%s encode text %u flags %u: got %" PRId64 " ids, want %" PRIu64, name, E2E[k].text, E2E[k].flags, r, n);
        if (!ok && r > 0) {
            for (uint64_t i = 0; i < n && i < (uint64_t)r; i++) {
                if (out[i] != E2E[k].v[i]) {
                    fprintf(stderr, "  first diff at [%" PRIu64 "]: got %u want %u\n", i, out[i], E2E[k].v[i]);
                    break;
                }
            }
        }
        /* every capacity below n + 2: the count stays, the prefix is exact */
        for (uint64_t cap = 0; cap <= n + 1u; cap += (n > 64u ? 1u + n / 16u : 1u)) {
            memset(out, 0xEE, 4u * (cap + 1u));
            r = toks_encode(ctx, t, len, E2E[k].flags, cap ? out : NULL, cap, scr);
            uint64_t m = cap < n ? cap : n;
            CHECK(r == (int64_t)n && memcmp(out, E2E[k].v, (size_t)m * 4u) == 0 && (cap > n || out[cap] == 0xEEEEEEEEu),
                  "%s encode text %u flags %u cap %" PRIu64 ": %" PRId64, name, E2E[k].text, E2E[k].flags, cap, r);
        }
        if ((E2E[k].flags & 4u) != 0u) { continue; }
        /* pieces: hf's piece ends in the normalized stream, exactly (strictly increasing, tiling [0, nlen); nlen ==
         * len unless the tokenizer's normalizer changed the text), at most one per id */
        uint32_t pn = 0;
        const uint32_t *pw = find_pieces(name, E2E[k].text, E2E[k].flags & 3u, &pn);
        r = toks_pieces(ctx, t, len, E2E[k].flags & 3u, out, 8192u, scr);
        uint64_t nlen = E2E[k].nlen;
        /* a Unigram's WhitespaceSplit / Metaspace drop whitespace: its pieces end at or before nlen */
        int tiled = r >= 0 && (nlen == 0u ? r == 0 : (r > 0 && (ws_drop ? out[r - 1] <= nlen : out[r - 1] == nlen)));
        for (int64_t i = 1; tiled && i < r; i++) { tiled = out[i] > out[i - 1]; }
        /* at most one piece per id, unless the file truncates (uni_pmminilm: max_length 128 cuts the ids, not the
         * pieces) */
        CHECK(tiled && (ws_drop || (uint64_t)r <= n), "%s pieces text %u flags %u: %" PRId64, name, E2E[k].text,
              E2E[k].flags, r);
        int same = pw != NULL && r == (int64_t)pn && memcmp(out, pw, (size_t)pn * 4u) == 0;
        CHECK(same, "%s pieces text %u flags %u: %" PRId64 " ends, want %u", name, E2E[k].text, E2E[k].flags, r, pn);
        if (!same && pw != NULL && r > 0) {
            for (uint32_t i = 0; i < pn && i < (uint64_t)r; i++) {
                if (out[i] != pw[i]) { fprintf(stderr, "  first diff at [%u]: got %u want %u\n", i, out[i], pw[i]); break; }
            }
        }
    }
    /* a scratch of exactly toks_scratch_bytes(len) for each text, at alignment k mod 64: the regions hold what
     * the text needs (for NFC tokenizers its normalized form, up to 3x longer: core.h tmax) */
    for (int k = 0; k < E2E_N; k++) {
        if (strcmp(E2E[k].tok, name) != 0) { continue; }
        const char *t = E2E_TEXT[E2E[k].text];
        uint64_t len = E2E_LEN[E2E[k].text], n = E2E[k].n;
        uint64_t sb = toks_scratch_bytes(ctx, len, 0u);
        uint8_t *mem = (uint8_t *)malloc((size_t)sb + 64u);
        void *tight = mem + (k & 63);
        int64_t r = mem != NULL && toks_scratch_init(ctx, tight, sb, 0u) == 0
                        ? toks_encode(ctx, t, len, E2E[k].flags, out, 8192u, tight) : -100;
        CHECK(r == (int64_t)n && memcmp(out, E2E[k].v, (size_t)n * 4u) == 0,
              "%s tight scratch (%" PRIu64 " bytes) text %u flags %u: got %" PRId64 " ids, want %" PRIu64, name, sb,
              E2E[k].text, E2E[k].flags, r, n);
        if (mem != NULL) {                              /* half the room: K5 writes through the bounce */
            r = toks_encode(ctx, t, len, E2E[k].flags, out, n / 2u, tight);
            CHECK(r == (int64_t)n && memcmp(out, E2E[k].v, (size_t)(n / 2u) * 4u) == 0,
                  "%s tight scratch cap %" PRIu64 " text %u flags %u: got %" PRId64, name, n / 2u, E2E[k].text,
                  E2E[k].flags, r);
        }
        if ((E2E[k].flags & 4u) == 0u && mem != NULL) {
            r = toks_pieces(ctx, t, len, E2E[k].flags & 3u, out, 8192u, tight);
            CHECK(r > 0 ? (ws_drop ? out[r - 1] <= E2E[k].nlen : out[r - 1] == E2E[k].nlen) : (ws_drop || E2E[k].nlen == 0u),
                  "%s tight scratch pieces text %u flags %u: %" PRId64, name, E2E[k].text, E2E[k].flags, r);
        }
        free(mem);
    }
    uint8_t *dec = (uint8_t *)malloc(65536u);
    for (int k = 0; k < E2E_N_DEC; k++) {
        if (strcmp(E2E_DEC[k].tok, name) != 0) { continue; }
        uint32_t n;
        const uint32_t *ids = find_ids(name, E2E_DEC[k].text, 0u, &n);
        int64_t r = toks_decode(ctx, ids, n, E2E_DEC[k].skip ? TOKS_SKIP_SPECIAL : 0u, dec, 65536u);
        CHECK(r == (int64_t)E2E_DEC[k].n && memcmp(dec, E2E_DEC[k].s, E2E_DEC[k].n) == 0,
              "%s decode text %u skip %u: %" PRId64 " bytes, want %u", name, E2E_DEC[k].text, E2E_DEC[k].skip, r, E2E_DEC[k].n);
    }
    free(dec);
    free(out);
}

static void test_one(const char *name, uint32_t n_ids, uint32_t n_added, const char *sha_hex)
{
    const char *path = path_of(name);
    uint64_t flen = 0;
    uint8_t *file = slurp(path, &flen);
    if (file == NULL) { printf("SKIP %s: %s not found\n", name, path); return; }

    toks_diag diag;
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &diag;
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, path, &o);
    CHECK(r == 0 && ctx != NULL && diag.code == 0, "%s load: %" PRId64 " (%s)", name, r, diag.what);
    if (ctx == NULL) { free(file); return; }

    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = (uint32_t)sizeof info;
    CHECK(toks_get_info(ctx, &info) == 0, "info");
    char hex[65];
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", info.source_sha256[i]); }
    uint32_t algo = strncmp(name, "uni_", 4) == 0 ? TOKS_ALGO_UNIGRAM : TOKS_ALGO_BPE_BYTELEVEL;   /* the pins' names */
    CHECK(info.n_ids == n_ids && info.n_added == n_added && info.algorithm == algo &&
          strcmp(info.name, name) == 0 && strcmp(hex, sha_hex) == 0,
          "%s info: n_ids %u n_added %u name %s sha %s", name, info.n_ids, info.n_added, info.name, hex);
    /* 0.3: every normalizer runs on the compiled path, and no tokenizer is certified for SPEC §3.6 (toks.h's
     * TOKS_PATH_NORMALIZE and control_isolation: pinned, so a change to either is a decision, not a drift) */
    CHECK((info.paths & TOKS_PATH_NORMALIZE) != 0u && info.control_isolation == 0u && info.rsv == 0u,
          "%s info: paths %#x, control_isolation %u", name, info.paths, info.control_isolation);

    uint64_t maxlen = 0;
    for (int i = 0; i < E2E_N_TEXTS; i++) { if (E2E_LEN[i] > maxlen) { maxlen = E2E_LEN[i]; } }
    uint64_t sb = toks_scratch_bytes(ctx, maxlen, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch");
    run_cases(name, ctx, scr);
    if (strcmp(name, "uni_pmminilm") == 0) {           /* truncation (max_length 128) cuts a whole document only */
        static char lt[3000];
        uint32_t *o2 = (uint32_t *)malloc(4u * 4096u);
        uint64_t lb = toks_scratch_bytes(ctx, sizeof lt, 0u);
        void *ls = malloc((size_t)lb);
        for (int i = 0; i < (int)sizeof lt; i++) { lt[i] = "ab "[i % 3]; }
        int64_t whole = -1, part = -1;
        if (o2 != NULL && ls != NULL && toks_scratch_init(ctx, ls, lb, 0u) == 0) {
            whole = toks_encode(ctx, lt, sizeof lt, TOKS_NO_POSTPROCESS, o2, 4096u, ls);
            part = toks_encode(ctx, lt, sizeof lt, TOKS_NO_POSTPROCESS | TOKS_CONTINUATION, o2, 4096u, ls);
        }
        CHECK(whole == 128 && part >= 1000, "%s: truncation %" PRId64 " ids, a continuation part %" PRId64, name, whole, part);
        free(ls);
        free(o2);
    }
    /* the same cases through a scratch with the long-piece cache (TOKS_SCRATCH_CACHE_MIB): warm after the first
     * few calls (the capacity loop encodes every text many times), so repeated long pieces come from it */
    uint64_t sb4 = toks_scratch_bytes(ctx, maxlen, TOKS_SCRATCH_CACHE_MIB(4));
    void *scr4 = malloc((size_t)sb4);
    CHECK(scr4 != NULL && toks_scratch_init(ctx, scr4, sb4, TOKS_SCRATCH_CACHE_MIB(4)) == 0, "scratch, cache MiB 4");
    if (scr4 != NULL) {
        run_cases(name, ctx, scr4);
        const toks_scratch *h4 = toks_scr_header(scr4);
        CHECK(algo != TOKS_ALGO_BPE_BYTELEVEL || h4->long_pos != 0u, "%s: the long cache stored nothing", name);
    }
    free(scr4);

    /* the same file from memory: same ids, and the scratch stays bound (same tables) */
    toks_ctx *ctx2 = NULL;
    CHECK(toks_load_mem_copy(&ctx2, file, flen, NULL) == 0 && ctx2 != NULL, "%s load_mem_copy", name);
    if (ctx2 != NULL) {
        uint32_t n, out[64];
        const uint32_t *want = find_ids(name, 2u, 0u, &n);
        CHECK(toks_encode(ctx2, E2E_TEXT[2], E2E_LEN[2], 0u, out, 64u, scr) == (int64_t)n &&
              memcmp(out, want, n * 4u) == 0, "%s encode through the copy", name);
        toks_unload(ctx2);
    }

    /* a model directory holding tokenizer.json */
#if !defined(_WIN32)
    char dir[512], link[600];
    snprintf(dir, sizeof dir, "%s/toks-e2e-XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (mkdtemp(dir) != NULL) {
        snprintf(link, sizeof link, "%s/tokenizer.json", dir);
        CHECK(symlink(path, link) == 0, "symlink");
        toks_ctx *ctx3 = NULL;
        CHECK(toks_load(&ctx3, dir, NULL) == 0 && ctx3 != NULL, "%s load from a directory", name);
        toks_info i3;
        memset(&i3, 0, sizeof i3);
        i3.size = (uint32_t)sizeof i3;
        CHECK(ctx3 != NULL && toks_get_info(ctx3, &i3) == 0 && strcmp(i3.name, dir + strlen(dir) - 15) == 0,
              "%s directory name %s", name, i3.name);
        toks_unload(ctx3);
        unlink(link);
        rmdir(dir);
    }
    if (strcmp(name, "gpt2") == 0) {   /* a directory name of 100 bytes: info.name is its first 63, terminated */
        char longdir[700], name100[101], l2[800];
        memset(name100, 'n', 100u);
        memcpy(name100, "toks-e2e-", 9u);
        name100[100] = 0;
        snprintf(longdir, sizeof longdir, "%s/%s", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", name100);
        snprintf(l2, sizeof l2, "%s/tokenizer.json", longdir);
        unlink(l2);                                    /* a crashed run's leftovers */
        rmdir(longdir);
        if (mkdir(longdir, 0700) == 0) {
            CHECK(symlink(path, l2) == 0, "symlink");
            toks_ctx *c5 = NULL;
            toks_info i5;
            memset(&i5, 0, sizeof i5);
            i5.size = (uint32_t)sizeof i5;
            CHECK(toks_load(&c5, longdir, NULL) == 0 && c5 != NULL && toks_get_info(c5, &i5) == 0 && strlen(i5.name) == 63u &&
                  memcmp(i5.name, name100, 63u) == 0, "%s a 100-byte directory name: %zu bytes", name, strlen(i5.name));
            toks_unload(c5);
            unlink(l2);
            rmdir(longdir);
        } else {
            CHECK(0, "mkdir %s", longdir);
        }
    }
#else
    char dir[MAX_PATH + 32], copy[MAX_PATH + 64], tmp[MAX_PATH + 1];   /* a copy: symlinks need a privilege */
    DWORD tl = GetTempPathA(sizeof tmp, tmp);
    snprintf(dir, sizeof dir, "%stoks-e2e-%06lu", (tl > 0 && tl < sizeof tmp) ? tmp : ".\\",
             (unsigned long)(GetCurrentProcessId() % 1000000u));
    if (CreateDirectoryA(dir, NULL)) {
        snprintf(copy, sizeof copy, "%s\\tokenizer.json", dir);
        CHECK(CopyFileA(path, copy, FALSE) != 0, "copy into a model directory");
        toks_ctx *ctx3 = NULL;
        CHECK(toks_load(&ctx3, dir, NULL) == 0 && ctx3 != NULL, "%s load from a directory", name);
        toks_info i3;
        memset(&i3, 0, sizeof i3);
        i3.size = (uint32_t)sizeof i3;
        CHECK(ctx3 != NULL && toks_get_info(ctx3, &i3) == 0 && strcmp(i3.name, dir + strlen(dir) - 15) == 0,
              "%s directory name %s", name, i3.name);
        toks_unload(ctx3);
        DeleteFileA(copy);
        RemoveDirectoryA(dir);
    } else {
        CHECK(0, "CreateDirectoryA %s", dir);
    }
#endif

    /* tiers: scalar always loads; a tier this machine or build lacks is TOKS_E_TIER */
    toks_ctx *c4 = NULL;
    o.tier = TOKS_TIER_SCALAR;
    CHECK(toks_load(&c4, path, &o) == 0 && c4 != NULL, "%s forced scalar", name);
    info.size = (uint32_t)sizeof info;
    CHECK(c4 != NULL && toks_get_info(c4, &info) == 0 && info.tier == TOKS_TIER_SCALAR, "scalar tier");
    toks_unload(c4);
#if defined(__aarch64__) || defined(_M_ARM64)
    o.tier = TOKS_TIER_AVX512;
#else
    o.tier = TOKS_TIER_NEON;
#endif
    c4 = NULL;
    CHECK(toks_load(&c4, path, &o) == TOKS_E_TIER && c4 == NULL && diag.code == TOKS_E_TIER, "%s foreign tier", name);
    o.tier = TOKS_TIER_AUTO;

    free(scr);
    free(file);
    toks_unload(ctx);
}

/* a Qwen-1 model (toks.h: "a tiktoken model's ... qwen.tiktoken (its other files beside it), or a model directory
 * holding one of them"): the cache's qwen1-72b.* (tools/corpora/fetch_tokenizers.py's pins) copied into a model
 * directory under their model names, loaded by the directory and by its qwen.tiktoken; the ids are tiktoken 0.14.0's
 * through the wrapper's own constants (tests/parity/qwen1_check.py's reference) */
static int copy_file(const char *from, const char *to)
{
    uint64_t n = 0;
    uint8_t *b = slurp(from, &n);
    FILE *f = b != NULL ? fopen(to, "wb") : NULL;
    int ok = f != NULL && fwrite(b, 1, (size_t)n, f) == (size_t)n;
    if (f != NULL && fclose(f) != 0) { ok = 0; }
    free(b);
    return ok;
}

static void test_qwen1(void)
{
    static const char *const SRC[3] = { "qwen1-72b.tiktoken", "qwen1-72b_tokenizer_config.json", "qwen1-72b_tokenization_qwen.py" };
    static const char *const DST[3] = { "qwen.tiktoken", "tokenizer_config.json", "tokenization_qwen.py" };
    static const char T0[] = "Hello, world! caf\xc3\xa9 \xe4\xb8\xad\xe6\x96\x87<|im_start|>user<|endoftext|>";
    static const char T1[] = "e\xcc\x81 1234567";                       /* NFC first: e + U+0301 is one char */
    static const uint32_t W0[] = { 9707, 11, 1879, 0, 51950, 72858, 16744, 151644, 872, 151643 };
    static const uint32_t W1[] = { 963, 220, 16, 17, 18, 19, 20, 21, 22 };
    char src[3][1100];
    for (int i = 0; i < 3; i++) {
        snprintf(src[i], sizeof src[i], "%s", path_of(SRC[i]));
        FILE *f = fopen(src[i], "rb");
        if (f == NULL) { printf("SKIP qwen1: %s missing (tools/corpora/fetch_tokenizers.py)\n", src[i]); return; }
        fclose(f);
    }
    char dir[600], dst[3][700], file[700];
#if !defined(_WIN32)
    snprintf(dir, sizeof dir, "%s/toks-qwen1-XXXXXX", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    int made = mkdtemp(dir) != NULL;
#else
    char tmp[MAX_PATH + 1];
    DWORD tl = GetTempPathA(sizeof tmp, tmp);
    snprintf(dir, sizeof dir, "%stoks-qwen1-%06lu", (tl > 0 && tl < sizeof tmp) ? tmp : ".\\",
             (unsigned long)(GetCurrentProcessId() % 1000000u));
    int made = CreateDirectoryA(dir, NULL) != 0;
#endif
    CHECK(made, "qwen1: a model directory under %s", dir);
    if (!made) { return; }
    int copied = 1;
    for (int i = 0; i < 3; i++) {
        snprintf(dst[i], sizeof dst[i], "%s/%s", dir, DST[i]);
        copied &= copy_file(src[i], dst[i]);
    }
    CHECK(copied, "qwen1: copies into %s", dir);
    snprintf(file, sizeof file, "%s", dst[0]);
    const char *paths[2] = { dir, file };
    const char *base = strrchr(dir, '/');
#if defined(_WIN32)
    if (base == NULL) { base = strrchr(dir, '\\'); }
#endif
    base = base != NULL ? base + 1 : dir;
    for (int i = 0; copied && i < 2; i++) {
        toks_diag d;
        memset(&d, 0, sizeof d);
        toks_load_opts o = { sizeof o, 0, 0, 0, &d };
        toks_ctx *c = NULL;
        int64_t r = toks_load(&c, paths[i], &o);
        CHECK(r == 0 && c != NULL, "qwen1: toks_load(%s): %" PRId64 " %s", paths[i], r, d.what);
        if (c == NULL) { continue; }
        toks_info in;
        memset(&in, 0, sizeof in);
        in.size = (uint32_t)sizeof in;
        CHECK(toks_get_info(c, &in) == 0 && in.n_ids == 151851u && in.algorithm == TOKS_ALGO_BPE_BYTELEVEL &&
              strcmp(in.name, base) == 0, "qwen1 (%s): n_ids %u, algorithm %u, name %s", paths[i], in.n_ids, in.algorithm, in.name);
        uint64_t sb = toks_scratch_bytes(c, 256u, 0u);
        void *scr = malloc((size_t)sb);
        uint32_t out[32];
        CHECK(scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0, "qwen1: scratch");
        int64_t n0 = scr != NULL ? toks_encode(c, T0, sizeof T0 - 1u, 0u, out, 32u, scr) : -100;
        CHECK(n0 == (int64_t)(sizeof W0 / 4u) && memcmp(out, W0, sizeof W0) == 0, "qwen1 (%s): ids of T0 (%" PRId64 ")", paths[i], n0);
        int64_t n1 = scr != NULL ? toks_encode(c, T1, sizeof T1 - 1u, 0u, out, 32u, scr) : -100;
        CHECK(n1 == (int64_t)(sizeof W1 / 4u) && memcmp(out, W1, sizeof W1) == 0, "qwen1 (%s): ids of T1 (%" PRId64 ")", paths[i], n1);
        free(scr);
        toks_unload(c);
    }
    for (int i = 0; i < 3; i++) { remove(dst[i]); }
#if !defined(_WIN32)
    rmdir(dir);
#else
    RemoveDirectoryA(dir);
#endif
    printf("qwen1: %s and its qwen.tiktoken load (tiktoken 0.14.0's ids)\n", dir);
}

int main(void)
{
    test_qwen1();
    test_one("gpt2", 50257u, 1u, "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6");
    test_one("llama3", 128256u, 256u, "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b");
    test_one("glm53", 154856u, 36u, "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d");
    test_one("dsv3", 128815u, 818u, "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41");
    test_one("qwen38", 248077u, 33u, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3");
    test_one("o200k", 200019u, 21u, "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3");
    test_one("mistral-nemo", 131072u, 1000u, "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db");
    /* one file per family of the 2026-10-04 critical targets (test_e2e.py's PINS) */
    test_one("nemotron3-4b", 131072u, 1000u, "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7");
    test_one("llama4", 201135u, 1135u, "172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4");
    test_one("minimaxm2", 200054u, 54u, "757622126525aeeb131756849d93298070ff3f0319c455ec8c5bb0f6b1cebbe8");
    test_one("dsv4", 129280u, 1283u, "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf");
    /* a byte-level vocab missing some byte chars (dropped, docs/breadth.md §4) + decode-only strings + lstrip */
    test_one("modernbert", 50368u, 116u, "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1");
    test_one("uni_pmminilm", 250002u, 5u, "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8");
    test_one("uni_llmjp4", 196608u, 352u, "a35f390c6489427e4dcac7d957f34d9dacd7332972c990f4b4eb6b724c8873f4");

    /* a scratch bound to one tokenizer is refused by another */
    toks_ctx *a = NULL, *b = NULL;
    if (toks_load(&a, path_of("gpt2"), NULL) == 0 && toks_load(&b, path_of("llama3"), NULL) == 0) {
        uint64_t sb = toks_scratch_bytes(a, 64u, 0u);
        void *scr = malloc((size_t)sb);
        uint32_t out[8];
        CHECK(toks_scratch_init(a, scr, sb, 0u) == 0 && toks_encode(a, "hi", 2u, 0u, out, 8u, scr) > 0 &&
              toks_encode(b, "hi", 2u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "scratch bound to gpt2 refused by llama3");
        free(scr);
    }
    toks_unload(a);
    toks_unload(b);

    /* load errors */
    toks_ctx *c = (toks_ctx *)(uintptr_t)1;
    toks_load_opts o;
    toks_diag diag;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &diag;
    CHECK(toks_load(&c, "/nonexistent/toks/tokenizer.json", &o) == TOKS_E_OPEN && c == NULL && diag.code == TOKS_E_OPEN, "open");
    CHECK(toks_load(NULL, "x", NULL) == TOKS_E_ARG, "out NULL");
    CHECK(toks_load(&c, NULL, NULL) == TOKS_E_ARG, "path NULL");
    o.size = 8u;
    CHECK(toks_load(&c, path_of("gpt2"), &o) == TOKS_E_ARG, "opts size");
    {   /* a struct of another size is refused before diag is read: diag lies past a 16-byte caller block */
        uint32_t *small = (uint32_t *)malloc(16u);
        CHECK(small != NULL, "malloc");
        small[0] = 16u; small[1] = 0u; small[2] = 0u; small[3] = 0u;
        CHECK(toks_load(&c, path_of("gpt2"), (const toks_load_opts *)(void *)small) == TOKS_E_ARG, "opts size 16, no diag read");
        CHECK(toks_load_mem_copy(&c, "x", 1u, (const toks_load_opts *)(void *)small) == TOKS_E_ARG, "opts size 16 (mem_copy)");
        free(small);
    }
    o.size = (uint32_t)sizeof o;
    o.flags = 1u;
    CHECK(toks_load(&c, path_of("gpt2"), &o) == TOKS_E_ARG, "load flag 1 (no load flags are defined)");
    o.flags = 0x80u;
    CHECK(toks_load(&c, path_of("gpt2"), &o) == TOKS_E_ARG, "unknown load flag");
    o.flags = 0u;
    CHECK(toks_load_mem_copy(&c, "{", 1u, &o) == TOKS_E_FORMAT && diag.code == TOKS_E_FORMAT, "bad json");
    CHECK(toks_load_mem_copy(&c, NULL, 0u, NULL) == TOKS_E_FORMAT, "empty");
    uint64_t glen = 0;
    uint8_t *g = slurp(path_of("gpt2"), &glen);        /* the env checks need a loadable file */
    int have = g != NULL;
    free(g);
    if (have) {
        setenv("TOKS_TIER", "bogus", 1);
        CHECK(toks_load(&c, path_of("gpt2"), &o) == TOKS_E_TIER, "TOKS_TIER=bogus");
        setenv("TOKS_TIER", "scalar", 1);
        if (toks_load(&c, path_of("gpt2"), &o) == 0) {
            toks_info info;
            memset(&info, 0, sizeof info);
            info.size = (uint32_t)sizeof info;
            CHECK(toks_get_info(c, &info) == 0 && info.tier == TOKS_TIER_SCALAR, "TOKS_TIER=scalar");
            toks_unload(c);
        }
        unsetenv("TOKS_TIER");
    }
    toks_unload(NULL);

    printf("test_e2e: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
