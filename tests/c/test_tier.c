/*
 * test_tier.c: kernel dispatch through the real library, on tests/data/compile/llama3style.json (a cl100k
 * template, a post-processor and added tokens of both phases, so K1, K3, K5 and K6 all run):
 *  - wiring: every kernel source this build assembles (TOKS_ASM_SOURCES, from the Makefile) has its
 *    dispatch flag (kernels.h) set, so no kernel is assembled and never called; k6_<tier>.S needs the
 *    k5_<tier>.S that calls it;
 *  - selection: TOKS_TIER_AUTO picks the highest tier that has one of its own kernels in the build and
 *    runs on this cpu, and toks_info reports it; a forced tier (opts and the TOKS_TIER variable) loads
 *    exactly when AUTO could pick it, scalar always, else TOKS_E_TIER;
 *  - results: every tier that loads encodes and splits exactly like scalar (the asm kernels against the
 *    c twins through the driver: chunk rounds, the bounce, count-only calls, added tokens, every mode).
 *  - the feature bits that bind no tier (cpu.h: pclmul, pmull): set only where the instruction runs, its 1000
 *    products equal to a portable carry-less multiply; set on every avx2 cpu and every apple arm64 cpu (all have
 *    it), so a detection that never sets it fails there.
 * Inside a tier each kernel falls back to its best lower version (TOKS_RUN): test_dispatch.c.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "../../src/core/kernels.h"
#include "cpu.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static const char *TIER_NAME[] = { "auto", "scalar", "neon", "avx2", "avx512" };

/* ================================================================ wiring */

/* the dispatched kernels: source name (src/asm/<isa>/<name>.S), its flag, its tier */
typedef struct { const char *name; int have; const char *flag; uint32_t tier; } wired;
#define W(n, F, T) { n, F, #F, T }
static const wired KERNELS[] = {
    W("k1_neon", TOKS_HAVE_K1_NEON, TOKS_TIER_NEON),
    W("k3_cl100k_neon", TOKS_HAVE_K3_CL100K_NEON, TOKS_TIER_NEON),
    W("k3_o200k_neon", TOKS_HAVE_K3_O200K_NEON, TOKS_TIER_NEON),
    W("k3_dsv3_neon", TOKS_HAVE_K3_DSV3_NEON, TOKS_TIER_NEON),
    W("k5_neon", TOKS_HAVE_K5_NEON, TOKS_TIER_NEON),
    W("k7_spm_neon", TOKS_HAVE_K7_SPM_NEON, TOKS_TIER_NEON),
    W("k1_avx2", TOKS_HAVE_K1_AVX2, TOKS_TIER_AVX2),
    W("k3_cl100k_avx2", TOKS_HAVE_K3_CL100K_AVX2, TOKS_TIER_AVX2),
    W("k3_o200k_avx2", TOKS_HAVE_K3_O200K_AVX2, TOKS_TIER_AVX2),
    W("k3_dsv3_avx2", TOKS_HAVE_K3_DSV3_AVX2, TOKS_TIER_AVX2),
    W("k5_avx2", TOKS_HAVE_K5_AVX2, TOKS_TIER_AVX2),
    W("k7_spm_avx2", TOKS_HAVE_K7_SPM_AVX2, TOKS_TIER_AVX2),
    W("k1_avx512", TOKS_HAVE_K1_AVX512, TOKS_TIER_AVX512),
    W("k3_cl100k_avx512", TOKS_HAVE_K3_CL100K_AVX512, TOKS_TIER_AVX512),
    W("k3_o200k_avx512", TOKS_HAVE_K3_O200K_AVX512, TOKS_TIER_AVX512),
    W("k3_dsv3_avx512", TOKS_HAVE_K3_DSV3_AVX512, TOKS_TIER_AVX512),
    W("k5_avx512", TOKS_HAVE_K5_AVX512, TOKS_TIER_AVX512),
};
#define N_KERNELS (sizeof KERNELS / sizeof KERNELS[0])

static int built[5];             /* per tier: one of its own dispatched kernels is in the build */
static char src[64][48];         /* the build's asm sources, without .S */
static int n_src;

static int has_src(const char *name)
{
    for (int i = 0; i < n_src; i++) { if (strcmp(src[i], name) == 0) { return 1; } }
    return 0;
}

static void check_wiring(void)
{
#if defined(TOKS_ASM_SOURCES)
    const char *s = TOKS_ASM_SOURCES;
    while (*s != 0 && n_src < 64) {
        while (*s == ' ') { s++; }
        size_t k = 0;
        while (s[k] != 0 && s[k] != ' ') { k++; }
        if (k > 2 && k < sizeof src[0] + 2 && s[k - 2] == '.' && s[k - 1] == 'S') {
            memcpy(src[n_src], s, k - 2);
            src[n_src][k - 2] = 0;
            n_src++;
        }
        s += k;
    }
    printf("asm sources (%d):%s%s\n", n_src, TOKS_ASM_SOURCES[0] ? " " : "", TOKS_ASM_SOURCES);
    for (int i = 0; i < n_src; i++) {
        const char *n = src[i];
        if (strcmp(n, "selftest") == 0) { continue; }          /* the macro-layer selftest (test_k0) */
        if (strncmp(n, "k6_", 3) == 0) {                        /* K6 runs inside its tier's K5 */
            char k5[48];
            snprintf(k5, sizeof k5, "k5_%s", n + 3);
            CHECK(has_src(k5), "%s.S is assembled but no %s.S calls it", n, k5);
            continue;
        }
        size_t j = 0;
        while (j < N_KERNELS && strcmp(KERNELS[j].name, n) != 0) { j++; }
        CHECK(j < N_KERNELS, "%s.S is assembled but nothing dispatches it: give it a flag and a case in kernels.h "
              "(and a line in this table)", n);
        if (j == N_KERNELS) { continue; }
        CHECK(KERNELS[j].have == 1, "%s.S is in the build but %s is 0: the kernel would never run", n, KERNELS[j].flag);
        built[KERNELS[j].tier] = 1;
    }
    for (size_t j = 0; j < N_KERNELS; j++) {
        CHECK(!KERNELS[j].have || has_src(KERNELS[j].name), "%s set without %s.S", KERNELS[j].flag, KERNELS[j].name);
    }
#else
    printf("SKIP wiring: this build passes no TOKS_ASM_SOURCES (the Makefile does); tiers from the flags\n");
    for (size_t j = 0; j < N_KERNELS; j++) { if (KERNELS[j].have) { built[KERNELS[j].tier] = 1; } }
#endif
    CHECK(built[TOKS_TIER_NEON] == TOKS_HAVE_NEON && built[TOKS_TIER_AVX2] == TOKS_HAVE_AVX2 &&
          built[TOKS_TIER_AVX512] == TOKS_HAVE_AVX512, "the TOKS_HAVE_<TIER> aggregates (kernels.h) miss a kernel");
}

/* ================================================================ texts */

static const char *FRAG[] = {
    "<|begin|>", "<|end|>", "<|a|>", "<|ab|>", "<|abc|>", "<|x", "<|xy|>", "@", "hello", "  ", "\xab\xf1\xbb",
    " hi there ", "<|", "|>", "<|ab", "Hello", " world", "don't", "'s", "'LL", "123", "4567", "8", " ", "\n",
    "\r\n", "\t", "  \n\n", "!?", "...", " (x)", "\xc3\xa9", "\xc3\xb1", "\xe4\xbd\xa0\xe5\xa5\xbd", "\xf0\x9f\x98\x80",
    "\xff", "\xe2\x82", "\xc3", "x", "y", "z", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "                                                                                ", "0123456789012345678901234567",
};
#define N_FRAG (sizeof FRAG / sizeof FRAG[0])

static uint64_t rng = 0x243F6A8885A308D3ull;

static uint64_t gen_text(uint8_t *buf, uint64_t want)
{
    uint64_t n = 0u;
    while (n < want) {
        const char *f = FRAG[guard_rng(&rng) % N_FRAG];
        size_t k = strlen(f);
        if (n + k > want) { k = (size_t)(want - n); }
        memcpy(buf + n, f, k);
        n += k;
    }
    return n;
}

/* ================================================================ contexts */

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

static toks_ctx *load(const uint8_t *json, uint64_t len, uint32_t tier, int64_t *r, uint32_t *got)
{
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.tier = tier;
    toks_ctx *c = NULL;
    *r = toks_load_mem_copy(&c, json, len, &o);
    *got = 0u;
    if (*r == 0 && c != NULL) {
        toks_info info;
        memset(&info, 0, sizeof info);
        info.size = (uint32_t)sizeof info;
        if (toks_get_info(c, &info) == 0) { *got = info.tier; }
    }
    return c;
}

#define MAXLEN 70000u

/* c (a tier) against s (scalar): encode at cap n, n / 2 and 0, and pieces, on the same texts */
static void compare(toks_ctx *c, toks_ctx *s, const char *what)
{
    uint64_t sb = toks_scratch_bytes(c, MAXLEN, 0u);
    void *scr_c = malloc((size_t)sb), *scr_s = malloc((size_t)sb);
    uint8_t *text = (uint8_t *)malloc(MAXLEN);
    uint32_t *a = (uint32_t *)malloc(4u * (MAXLEN + 16u)), *b = (uint32_t *)malloc(4u * (MAXLEN + 16u));
    CHECK(scr_c != NULL && scr_s != NULL && text != NULL && a != NULL && b != NULL, "malloc");
    CHECK(toks_scratch_init(c, scr_c, sb, 0u) == 0 && toks_scratch_init(s, scr_s, sb, 0u) == 0, "scratch init");
    static const uint32_t FLAGS[] = { 0u, 1u, 2u, 4u, 5u, 6u };
    int bad = 0;
    for (int r = 0; r < 400 && !bad; r++) {
        uint64_t want = r < 397 ? guard_rng(&rng) % (r < 300 ? 300u : 3000u) : (r == 397 ? 20000u : MAXLEN);
        uint64_t len = gen_text(text, want);
        uint32_t fl = FLAGS[guard_rng(&rng) % (sizeof FLAGS / sizeof FLAGS[0])];
        int64_t ns = toks_encode(s, text, len, fl, b, MAXLEN + 16u, scr_s);
        int64_t nc = toks_encode(c, text, len, fl, a, MAXLEN + 16u, scr_c);
        CHECK(ns >= 0 && nc == ns && memcmp(a, b, (size_t)ns * 4u) == 0, "%s: encode len %" PRIu64 " flags %u: %" PRId64
              " ids, scalar %" PRId64, what, len, fl, nc, ns);
        if (ns < 0 || nc != ns) { bad = 1; continue; }
        uint64_t half = (uint64_t)ns / 2u;
        memset(a, 0xEE, (size_t)(half + 1u) * 4u);
        CHECK(toks_encode(c, text, len, fl, half ? a : NULL, half, scr_c) == ns &&
              memcmp(a, b, (size_t)half * 4u) == 0 && a[half] == 0xEEEEEEEEu, "%s: encode cap %" PRIu64, what, half);
        CHECK(toks_encode(c, text, len, fl, NULL, 0u, scr_c) == ns, "%s: count-only encode", what);
        ns = toks_pieces(s, text, len, fl & 3u, b, MAXLEN + 16u, scr_s);
        nc = toks_pieces(c, text, len, fl & 3u, a, MAXLEN + 16u, scr_c);
        CHECK(ns >= 0 && nc == ns && memcmp(a, b, (size_t)ns * 4u) == 0, "%s: pieces len %" PRIu64 " mode %u", what, len, fl & 3u);
    }
    free(a); free(b); free(text); free(scr_c); free(scr_s);
}

static void set_tier_env(const char *v)
{
#if defined(_WIN32)
    char buf[64];
    snprintf(buf, sizeof buf, "TOKS_TIER=%s", v != NULL ? v : "");
    _putenv(buf);
#else
    if (v == NULL) { unsetenv("TOKS_TIER"); } else { setenv("TOKS_TIER", v, 1); }
#endif
}

/* ================================================================ bits that bind no tier */

/* a 64 x 64 -> 128 carry-less product, portable */
static void clmul_ref(uint64_t a, uint64_t b, uint64_t r[2])
{
    r[0] = r[1] = 0u;
    for (unsigned i = 0u; i < 64u; i++) {
        if ((b >> i) & 1u) { r[0] ^= a << i; r[1] ^= i != 0u ? a >> (64u - i) : 0u; }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#  include <wmmintrin.h>
#  define CLMUL_BIT TOKS_X86_PCLMUL
#  define CLMUL_NAME "pclmulqdq"
__attribute__((target("pclmul"))) static void clmul_hw(uint64_t a, uint64_t b, uint64_t r[2])
{
    __m128i p = _mm_clmulepi64_si128(_mm_set_epi64x(0, (long long)a), _mm_set_epi64x(0, (long long)b), 0x00);
    _mm_storeu_si128((__m128i *)(void *)r, p);
}
#else
#  include <arm_neon.h>
#  define CLMUL_BIT TOKS_ARM64_PMULL
#  define CLMUL_NAME "pmull"
__attribute__((target("aes"))) static void clmul_hw(uint64_t a, uint64_t b, uint64_t r[2])
{
    uint64x2_t p = vreinterpretq_u64_p128(vmull_p64((poly64_t)a, (poly64_t)b));
    r[0] = vgetq_lane_u64(p, 0);
    r[1] = vgetq_lane_u64(p, 1);
}
#endif

static uint64_t mix64(uint64_t *s)                 /* splitmix64: operands over all 64 bits */
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* the carry-less multiply bit is set only where the instruction runs and gives the product (a wrong bit would
 * trap here, or a wrong encoding differ); clear, nothing is executed */
static void check_clmul(uint64_t f)
{
    int set = (f & CLMUL_BIT) != 0u;
#if defined(__x86_64__) || defined(_M_X64)
    CHECK(set || !TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER), "an avx2 cpu without pclmulqdq: the detection is wrong");
#elif defined(__APPLE__)
    CHECK(set, "an apple arm64 cpu without pmull: the detection is wrong");   /* every apple arm64 cpu has it */
#endif
    uint64_t s = 0x243F6A8885A308D3ull;
    int same = 0;
    for (int i = 0; set && i < 1000; i++) {
        uint64_t a = i == 0 ? ~0ull : mix64(&s), b = i == 0 ? ~0ull : mix64(&s), w[2], g[2];
        clmul_ref(a, b, w);
        clmul_hw(a, b, g);
        CHECK(w[0] == g[0] && w[1] == g[1], CLMUL_NAME " of %#" PRIx64 " and %#" PRIx64 ": %#" PRIx64 ":%#" PRIx64
              ", want %#" PRIx64 ":%#" PRIx64, a, b, g[1], g[0], w[1], w[0]);
        same += w[0] == g[0] && w[1] == g[1];
    }
    printf("carry-less multiply (" CLMUL_NAME "): %s\n", set ? (same == 1000 ? "set, 1000 products equal the portable one" :
           "set, products DIFFER") : "clear");
}

int main(void)
{
    check_wiring();
    check_clmul(toks_cpu_features());

    uint64_t len = 0;
    uint8_t *json = slurp("tests/data/compile/llama3style.json", &len);
    CHECK(json != NULL, "tests/data/compile/llama3style.json (run from the source root)");
    if (json == NULL) { return 1; }

    uint64_t f = toks_cpu_features();
    int cpu[5] = { 0, 1, TOKS_CPU_HAS(f, TOKS_FEAT_NEON_TIER), TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER),
                   TOKS_CPU_HAS(f, TOKS_FEAT_AVX512_TIER) };
    uint32_t want = TOKS_TIER_SCALAR;
    for (uint32_t t = TOKS_TIER_NEON; t <= TOKS_TIER_AVX512; t++) { if (built[t] && cpu[t]) { want = t; } }

    set_tier_env(NULL);                                     /* AUTO decides, not the caller's environment */
    int64_t r;
    uint32_t got;
    toks_ctx *sc = load(json, len, TOKS_TIER_SCALAR, &r, &got);
    CHECK(r == 0 && sc != NULL && got == TOKS_TIER_SCALAR, "scalar loads: %" PRId64 ", tier %u", r, got);
    if (sc == NULL) { return 1; }
    toks_ctx *au = load(json, len, TOKS_TIER_AUTO, &r, &got);
    CHECK(r == 0 && au != NULL && got == want, "AUTO picked %s, want %s (cpu features %#" PRIx64 ")",
          got <= 4u ? TIER_NAME[got] : "?", TIER_NAME[want], f);
    printf("tier auto = %s (cpu features %#" PRIx64 "; built:%s%s%s)\n", got <= 4u ? TIER_NAME[got] : "?", f,
           built[2] ? " neon" : "", built[3] ? " avx2" : "", built[4] ? " avx512" : "");
    if (au != NULL) { compare(au, sc, "auto"); toks_unload(au); }

    for (uint32_t t = TOKS_TIER_SCALAR; t <= TOKS_TIER_AVX512; t++) {
        int ok = t == TOKS_TIER_SCALAR || (built[t] && cpu[t]);
        toks_ctx *c = load(json, len, t, &r, &got);
        CHECK(ok ? (r == 0 && c != NULL && got == t) : (r == TOKS_E_TIER && c == NULL),
              "forced %s: %" PRId64 " tier %u, want %s", TIER_NAME[t], r, got, ok ? "it" : "TOKS_E_TIER");
        if (c != NULL && t != TOKS_TIER_SCALAR) { compare(c, sc, TIER_NAME[t]); }
        toks_unload(c);
        /* the same through the variable, which AUTO honours */
        set_tier_env(TIER_NAME[t]);
        c = load(json, len, TOKS_TIER_AUTO, &r, &got);
        CHECK(ok ? (r == 0 && got == t) : r == TOKS_E_TIER, "TOKS_TIER=%s: %" PRId64 " tier %u", TIER_NAME[t], r, got);
        toks_unload(c);
        set_tier_env(NULL);
    }
    toks_unload(sc);
    free(json);
    printf("test_tier: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
