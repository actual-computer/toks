/*
 * test_norm.c: NFC (src/core/norm.c, SPEC §2.6) under `make test`.
 *
 *   1. hf 0.23.2 golden vectors (tests/norm/nfc_golden.txt, written by tools/gen/norm.py): every case whole,
 *      behind 0..129 boundary bytes (so the 64-byte K2 block edge falls at every position of it), and cut in
 *      two at every boundary atom (the parts normalize alone).
 *   2. every scalar value alone: its golden output when listed there, else itself.
 *   3. a reference: the crate's algorithm restated over materialized arrays (hangul decomposed as the crate
 *      does, linear-search pair table, insertion sort per run), invalid bytes as barriers (SPEC §3.3),
 *      against the normalizer on generated strings over an adversarial alphabet, whole and in random chunks,
 *      plus long combining runs.
 *   4. hand cases: invalid bytes beside combining marks, the 3x bound, api errors; toks_nfc_boundary on every
 *      byte and scalar at the end of a guarded buffer (ascii atoms once read past the input).
 *   5. K2 against a byte loop, and guard-page geometry (every length 0..255 x start 0..63, the buffer flush
 *      against a no-access page at its end and at its start) for K2 and the normalizer.
 *   6. NFKC / NFKD: hf 0.23.2 vectors (tests/norm/nfkc_golden.txt), whole and behind 0..129 boundary bytes; every
 *      scalar alone under every form a reader accepts stays within TOKS_NORM_X(f) bytes per input byte (the
 *      scratch's factor: the forms never lengthen across chars, so this bounds every text).
 * The outer oracle is hf at scale: tests/norm/ (run on lab hosts).
 */
/* norm.h -> core.h redeclares memcpy/memset/memcmp (freestanding core): it must come before <string.h>. */
#include "../../src/core/norm.h"
#include "../../src/gen/norm_nfc.h"
#include "../common/guard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lines.inc"

static uint64_t n_checks, n_fail;

#define CHECK(cond, ...) do { n_checks++; if (!(cond)) { n_fail++; if (n_fail <= 20) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static void hexs(const uint8_t *p, size_t n, char *dst, size_t cap)
{
    size_t o = 0;
    for (size_t i = 0; i < n && o + 3 < cap; i++) { o += (size_t)snprintf(dst + o, cap - o, "%02x", p[i]); }
    dst[o] = 0;
}

/* ------------------------------------------------------------------------------------------- the normalizer */

static size_t nfc_whole(const uint8_t *s, size_t n, uint8_t *out, size_t cap)
{
    int64_t r = toks_norm(TOKS_NS_NFC, s, n, out, cap);
    if (r < 0) { fprintf(stderr, "toks_norm failed (%lld)\n", (long long)r); exit(2); }
    return (size_t)r;
}

static int atom_start(const uint8_t *s, size_t n, size_t e)
{
    size_t i = 0;
    while (i < e) {
        uint32_t k = toks_utf8_len(s + i, n - i);
        i += k ? k : 1;
    }
    return i == e;
}

/* NFC of the parts of s cut at those of ends[0..k) (non-decreasing) that start a boundary atom: the property the
 * driver's stretches and toks_split_points rest on (a boundary atom splits the text into independent parts) */
static size_t nfc_chunked(const uint8_t *s, size_t n, const size_t *ends, size_t k, uint8_t *out, size_t cap)
{
    size_t o = 0, from = 0;
    for (size_t i = 0; i <= k; i++) {
        size_t e = i < k ? ends[i] : n;
        if (e < n && (e <= from || !atom_start(s, n, e) || !toks_nfc_boundary(TOKS_NS_NFC, s, n, e))) { continue; }
        o += nfc_whole(s + from, e - from, out + o, cap - o);
        from = e;
    }
    return o;
}

/* --------------------------------------------------------------------------------------------- reference */

#define R_BAD 0x80000000u

static uint32_t (*PAIRS)[3];   /* a, b, composite: sorted by (a, b) */
static size_t N_PAIRS;

static int pair_cmp(const void *x, const void *y)
{
    const uint32_t *p = x, *q = y;
    return p[0] != q[0] ? (p[0] < q[0] ? -1 : 1) : p[1] != q[1] ? (p[1] < q[1] ? -1 : 1) : 0;
}

static void ref_init(void)
{
    size_t size = (size_t)1 << TOKS_NFC_COMP_LOG2;
    PAIRS = calloc(size, sizeof *PAIRS);
    for (size_t i = 0; i < size; i++) {
        uint64_t s = toks_nfc_comp[i];
        if (s == 0) { continue; }
        PAIRS[N_PAIRS][0] = (uint32_t)(s >> 42);
        PAIRS[N_PAIRS][1] = (uint32_t)((s >> 21) & 0x1FFFFF);
        PAIRS[N_PAIRS][2] = (uint32_t)(s & 0x1FFFFF);
        N_PAIRS++;
    }
    qsort(PAIRS, N_PAIRS, sizeof *PAIRS, pair_cmp);
}

static uint32_t ref_ccc(uint32_t c)
{
    if (c & R_BAD) { return 0; }
    return toks_nfc_cls_ccc[(toks_nfc_info(c) >> TOKS_NFC_CLS_SHIFT) & TOKS_NFC_CLS_MASK];
}

/* the crate's compose(): hangul arithmetic, then the pair table */
static uint32_t ref_compose(uint32_t a, uint32_t b)
{
    if ((a | b) & R_BAD) { return 0; }
    if (a >= 0x1100 && a < 0x1100 + 19 && b >= 0x1161 && b < 0x1161 + 21) {
        return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    }
    if (a >= 0xAC00 && a < 0xAC00 + 11172 && b > 0x11A7 && b < 0x11A7 + 28 && (a - 0xAC00) % 28 == 0) {
        return a + (b - 0x11A7);
    }
    uint32_t key[3] = { a, b, 0 };
    uint32_t (*hit)[3] = bsearch(key, PAIRS, N_PAIRS, sizeof *PAIRS, pair_cmp);
    return hit ? (*hit)[2] : 0;
}

/* the crate's decompose(): ascii, hangul arithmetic, the full decomposition table */
static size_t ref_decompose(uint32_t c, uint32_t *d)
{
    if (c <= 0x7F) { d[0] = c; return 1; }
    if (c >= 0xAC00 && c < 0xAC00 + 11172) {
        uint32_t s = c - 0xAC00;
        d[0] = 0x1100 + s / 588;
        d[1] = 0x1161 + (s % 588) / 28;
        if (s % 28) { d[2] = 0x11A7 + s % 28; return 3; }
        return 2;
    }
    uint32_t w = toks_nfc_info(c), dl = (w >> TOKS_NFC_DLEN_SHIFT) & TOKS_NFC_DLEN_MASK;
    if (dl == 0) { d[0] = c; return 1; }
    for (uint32_t j = 0; j < dl; j++) { d[j] = toks_nfc_pool[(w & TOKS_NFC_DOFF_MASK) + j] & TOKS_NFC_CP_MASK; }
    return dl;
}

/* utf-8 per unicode table 3-7, restated: the length of the well-formed sequence at s[0..n), or 0 */
static size_t ref_u8len(const uint8_t *s, size_t n, uint32_t *cp)
{
    uint8_t b = s[0];
    size_t k;
    uint32_t c, lo = 0x80, hi = 0xBF;
    if (b < 0x80) { *cp = b; return 1; }
    else if (b >= 0xC2 && b <= 0xDF) { k = 2; c = b & 0x1F; }
    else if (b >= 0xE0 && b <= 0xEF) { k = 3; c = b & 0x0F; if (b == 0xE0) lo = 0xA0; if (b == 0xED) hi = 0x9F; }
    else if (b >= 0xF0 && b <= 0xF4) { k = 4; c = b & 0x07; if (b == 0xF0) lo = 0x90; if (b == 0xF4) hi = 0x8F; }
    else { return 0; }
    if (n < k || s[1] < lo || s[1] > hi) { return 0; }
    c = (c << 6) | (s[1] & 0x3F);
    for (size_t i = 2; i < k; i++) {
        if (s[i] < 0x80 || s[i] > 0xBF) { return 0; }
        c = (c << 6) | (s[i] & 0x3F);
    }
    *cp = c;
    return k;
}

static size_t ref_put(uint8_t *o, uint32_t c)
{
    if (c & R_BAD) { o[0] = (uint8_t)c; return 1; }
    if (c < 0x80) { o[0] = (uint8_t)c; return 1; }
    if (c < 0x800) { o[0] = (uint8_t)(0xC0 | c >> 6); o[1] = (uint8_t)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) {
        o[0] = (uint8_t)(0xE0 | c >> 12); o[1] = (uint8_t)(0x80 | ((c >> 6) & 0x3F)); o[2] = (uint8_t)(0x80 | (c & 0x3F));
        return 3;
    }
    o[0] = (uint8_t)(0xF0 | c >> 18); o[1] = (uint8_t)(0x80 | ((c >> 12) & 0x3F));
    o[2] = (uint8_t)(0x80 | ((c >> 6) & 0x3F)); o[3] = (uint8_t)(0x80 | (c & 0x3F));
    return 4;
}

static size_t ref_nfc(const uint8_t *s, size_t n, uint8_t *out)
{
    uint32_t *D = malloc(sizeof(uint32_t) * (4 * n + 4)), *O = malloc(sizeof(uint32_t) * (4 * n + 4));
    uint32_t *buf = malloc(sizeof(uint32_t) * (4 * n + 4));
    size_t nd = 0, no = 0, nb = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        size_t k = ref_u8len(s + i, n - i, &cp);
        if (k == 0) { D[nd++] = R_BAD | s[i]; i++; continue; }
        nd += ref_decompose(cp, D + nd);
        i += k;
    }
    for (size_t i = 0; i < nd;) {                     /* canonical ordering: stable sort of each run */
        if (ref_ccc(D[i]) == 0) { i++; continue; }
        size_t j = i;
        while (j < nd && ref_ccc(D[j]) != 0) { j++; }
        for (size_t a = i + 1; a < j; a++) {
            uint32_t x = D[a];
            size_t b = a;
            while (b > i && ref_ccc(D[b - 1]) > ref_ccc(x)) { D[b] = D[b - 1]; b--; }
            D[b] = x;
        }
        i = j;
    }
    int have = 0;
    long last = -1;
    uint32_t comp = 0;
    for (size_t i = 0; i < nd; i++) {                 /* recompose.rs, state for state */
        uint32_t ch = D[i];
        long cls = (long)ref_ccc(ch);
        if (!have) {
            if (cls != 0) { O[no++] = ch; continue; }
            comp = ch; have = 1;
            continue;
        }
        if (last < 0) {
            uint32_t r = ref_compose(comp, ch);
            if (r) { comp = r; continue; }
            if (cls == 0) { O[no++] = comp; comp = ch; continue; }
            buf[nb++] = ch; last = cls;
            continue;
        }
        if (last >= cls) {
            if (cls == 0) {
                O[no++] = comp;
                for (size_t j = 0; j < nb; j++) { O[no++] = buf[j]; }
                nb = 0; comp = ch; last = -1;
                continue;
            }
            buf[nb++] = ch; last = cls;
            continue;
        }
        uint32_t r = ref_compose(comp, ch);
        if (r) { comp = r; continue; }
        buf[nb++] = ch; last = cls;
    }
    if (have) { O[no++] = comp; }
    for (size_t j = 0; j < nb; j++) { O[no++] = buf[j]; }
    size_t o = 0;
    for (size_t i = 0; i < no; i++) { o += ref_put(out + o, O[i]); }
    free(D); free(O); free(buf);
    return o;
}

/* -------------------------------------------------------------------------------------------- golden file */

typedef struct gcase { uint8_t *in, *out; size_t nin, nout; } gcase;
static gcase *G;
static size_t NG;

static size_t unhex(const char *h, uint8_t *dst)
{
    size_t n = 0;
    while (h[0] && h[1] && h[0] != ' ' && h[0] != '\n') {
        unsigned v;
        sscanf(h, "%2x", &v);
        dst[n++] = (uint8_t)v;
        h += 2;
    }
    return n;
}

static void load_golden(void)
{
    FILE *f = fopen("tests/norm/nfc_golden.txt", "rb");
    if (!f) { fprintf(stderr, "test_norm: run from the repo root (tests/norm/nfc_golden.txt)\n"); exit(2); }
    char line[4096];
    size_t cap = 8192;
    G = malloc(cap * sizeof *G);
    td_at at = { "tests/norm/nfc_golden.txt", 0 };
    while (fgets(line, sizeof line, f)) {
        at.line++;
        if (!td_line(&at, line, sizeof line)) { n_fail++; break; }
        if (line[0] == '#') { continue; }
        char *sp = strchr(line, ' ');
        if (!sp) { n_fail += !td_bad(&at, line + strlen(line) - 1, "'<hex> <hex>' expected"); break; }
        uint8_t a[2048], b[2048];
        size_t na = unhex(line, a), nb = unhex(sp + 1, b);
        if (NG == cap) { cap *= 2; G = realloc(G, cap * sizeof *G); }
        G[NG].in = malloc(na + 1); memcpy(G[NG].in, a, na); G[NG].nin = na;
        G[NG].out = malloc(nb + 1); memcpy(G[NG].out, b, nb); G[NG].nout = nb;
        NG++;
    }
    fclose(f);
}

static void test_golden(void)
{
    uint8_t in[512], out[2048];
    char h1[600], h2[600];
    size_t ends[2];
    for (size_t g = 0; g < NG; g++) {
        const gcase *c = &G[g];
        for (size_t k = 0; k < 130; k++) {             /* K2 block edges at every position */
            memset(in, '#', k);
            memcpy(in + k, c->in, c->nin);
            size_t n = nfc_whole(in, k + c->nin, out, sizeof out);
            int ok = n == k + c->nout && memcmp(out + k, c->out, c->nout) == 0;
            if (!ok) { hexs(c->in, c->nin, h1, sizeof h1); hexs(out + k, n - k, h2, sizeof h2); }
            CHECK(ok, "golden %s at offset %zu: got %s", h1, k, h2);
            if (k != 0 && k != 1 && k != 63 && k != 64) { continue; }
            for (size_t e = 0; e <= k + c->nin; e++) {   /* two calls: every chunk end */
                ends[0] = e;
                n = nfc_chunked(in, k + c->nin, ends, 1, out, sizeof out);
                ok = n == k + c->nout && memcmp(out + k, c->out, c->nout) == 0;
                CHECK(ok, "golden case %zu split at %zu (offset %zu)", g, e, k);
            }
        }
    }
}

static void test_every_scalar(void)
{
    uint8_t **exp = calloc(0x110000, sizeof *exp);
    size_t *nexp = calloc(0x110000, sizeof *nexp);
    for (size_t g = 0; g < NG; g++) {
        uint32_t cp;
        if (ref_u8len(G[g].in, G[g].nin, &cp) == G[g].nin) { exp[cp] = G[g].out; nexp[cp] = G[g].nout; }
    }
    uint8_t in[4], out[16];
    size_t listed = 0;
    for (uint32_t cp = 0; cp < 0x110000; cp++) {
        if (cp >= 0xD800 && cp <= 0xDFFF) { continue; }
        size_t n = ref_put(in, cp), m = nfc_whole(in, n, out, sizeof out);
        if (exp[cp]) {
            listed++;
            CHECK(m == nexp[cp] && memcmp(out, exp[cp], m) == 0, "U+%04X alone", cp);
        } else {
            CHECK(m == n && memcmp(out, in, n) == 0, "U+%04X alone changed", cp);
        }
    }
    CHECK(listed > 1000, "golden lists %zu changing scalars", listed);
    free(exp); free(nexp);
}

/* ------------------------------------------------------------------------------------- generated strings */

/* an adversarial alphabet: starters, every kind of mark and second, firsts, hangul, singletons, exclusions,
 * non-starter decompositions, astral pairs, cold latin-1, and invalid byte sequences (as raw strings). */
static const uint32_t ALPHA_CP[] = {
    0x00, 0x20, 0x23, 0x3C, 0x3D, 0x3E, 0x41, 0x43, 0x45, 0x55, 0x61, 0x65, 0x6F, 0x75, 0x78,
    0xC0, 0xC5, 0xC7, 0xDC, 0xE9, 0xDF, 0xA0, 0x17F, 0x2B9,
    0x300, 0x301, 0x302, 0x303, 0x304, 0x306, 0x307, 0x308, 0x30A, 0x30C, 0x313, 0x314, 0x316, 0x31B, 0x323,
    0x327, 0x328, 0x334, 0x338, 0x340, 0x341, 0x342, 0x343, 0x344, 0x345, 0x35C, 0x360, 0x374, 0x37E, 0x387,
    0x390, 0x391, 0x3B1, 0x3B9, 0x3C9, 0x1F00, 0x1F71, 0x1F82, 0x1FB6,
    0x5B0, 0x5BC, 0x5C1, 0x5E9, 0xFB2C, 0xFB1D, 0x64E, 0x650, 0x654, 0x627, 0x64A,
    0x915, 0x928, 0x930, 0x933, 0x93C, 0x94D, 0x952, 0x958, 0x9C7, 0x9BE, 0x9D7,
    0xB47, 0xB3E, 0xB56, 0xB57, 0xCC6, 0xCD5, 0xCD6, 0xCC2, 0xDD9, 0xDCA, 0xDCF, 0xDDF,
    0xE38, 0xE48, 0xF71, 0xF72, 0xF73, 0xF74, 0xF75, 0xF80, 0xF81, 0x1025, 0x102E, 0x1B05, 0x1B35,
    0x1100, 0x1112, 0x1161, 0x1175, 0x11A7, 0x11A8, 0x11C2, 0xAC00, 0xAC01, 0xD788, 0xD7A3,
    0x1E08, 0x1E9B, 0x1EA0, 0x1EC7, 0x2000, 0x2126, 0x212B, 0x2190, 0x219A, 0x226E, 0x2ADC,
    0x304B, 0x3099, 0x309A, 0x30AB, 0x302A, 0x302E, 0xF900, 0xFA10,
    0x11099, 0x110BA, 0x1109A, 0x11131, 0x11127, 0x11347, 0x1133E, 0x11357, 0x114B9, 0x114B0, 0x114BD,
    0x115B8, 0x115AF, 0x1D157, 0x1D158, 0x1D15E, 0x1D160, 0x1D165, 0x1D16E, 0x1D1BB, 0x2F800, 0x10FFFF,
};
static const char *const ALPHA_BAD[] = {
    "\x80", "\xBF", "\xC0", "\xC1", "\xC3", "\xCC", "\xE2", "\xE2\x82", "\xED\xA0\x80", "\xE0\x80\x80",
    "\xF0\x80\x80\x80", "\xF4\x90\x80\x80", "\xF5", "\xFF", "\xF0\x9D\x85",
};
#define N_CP  (sizeof ALPHA_CP / sizeof ALPHA_CP[0])
#define N_BAD (sizeof ALPHA_BAD / sizeof ALPHA_BAD[0])

static size_t gen(uint64_t *rng, uint8_t *s, size_t items)
{
    size_t n = 0;
    for (size_t i = 0; i < items; i++) {
        uint64_t r = guard_rng(rng) % 100;
        if (r < 8) {
            const char *b = ALPHA_BAD[guard_rng(rng) % N_BAD];
            size_t k = strlen(b);
            memcpy(s + n, b, k);
            n += k;
        } else {
            n += ref_put(s + n, ALPHA_CP[guard_rng(rng) % N_CP]);
        }
    }
    return n;
}

static void test_generated(void)
{
    uint64_t rng = 0x5eed;
    uint8_t s[2048], a[8192], b[8192];
    size_t ends[8];
    char h1[600], h2[600], h3[600];
    for (int it = 0; it < 300000; it++) {
        size_t n = gen(&rng, s, (size_t)(guard_rng(&rng) % 24));
        size_t na = ref_nfc(s, n, a);
        size_t nb = nfc_whole(s, n, b, sizeof b);
        int ok = na == nb && memcmp(a, b, na) == 0;
        if (!ok) { hexs(s, n, h1, sizeof h1); hexs(a, na, h2, sizeof h2); hexs(b, nb, h3, sizeof h3); }
        CHECK(ok, "generated %s: reference %s, toks %s", h1, h2, h3);
        CHECK(nb <= TOKS_NORM_BOUND(TOKS_NS_NFC, n), "bound");
        size_t k = (size_t)(guard_rng(&rng) % 8);
        for (size_t i = 0; i < k; i++) { ends[i] = n ? (size_t)(guard_rng(&rng) % (n + 1)) : 0; }
        for (size_t i = 1; i < k; i++) {             /* sort the ends */
            for (size_t j = i; j > 0 && ends[j - 1] > ends[j]; j--) { size_t t = ends[j]; ends[j] = ends[j - 1]; ends[j - 1] = t; }
        }
        nb = nfc_chunked(s, n, ends, k, b, sizeof b);
        CHECK(na == nb && memcmp(a, b, na) == 0, "generated case %d in %zu chunks", it, k + 1);
    }
}

/* long combining runs: a starter (or none) and up to 3000 marks of mixed classes, seconds among them */
static void test_long_runs(void)
{
    static const uint32_t MARKS[] = { 0x300, 0x301, 0x308, 0x30A, 0x316, 0x323, 0x327, 0x328, 0x334, 0x345,
                                      0x313, 0x314, 0x342, 0x93C, 0x94D, 0x64E, 0x5B0, 0xE38, 0xF71, 0xF72,
                                      0x1D165, 0x1D16E, 0x302A, 0x3099, 0x344, 0x343, 0xF73 };
    static const uint32_t BASES[] = { 0x61, 0x41, 0x3B1, 0x55, 0x1F00, 0x915, 0x304B, 0xAC00, 0x1100 };
    uint64_t rng = 77;
    size_t cap = 3 * 4 * 4000 + 64;
    uint8_t *s = malloc(4 * 4000 + 16), *a = malloc(cap), *b = malloc(cap);
    for (int it = 0; it < 60; it++) {
        size_t m = (size_t)(guard_rng(&rng) % 3000) + 33, n = 0;
        if (it % 5) { n += ref_put(s, BASES[guard_rng(&rng) % (sizeof BASES / sizeof BASES[0])]); }
        int one = it % 3 == 0;                         /* one class only, or mixed */
        uint32_t fixed = MARKS[guard_rng(&rng) % (sizeof MARKS / sizeof MARKS[0])];
        for (size_t i = 0; i < m; i++) {
            n += ref_put(s + n, one ? fixed : MARKS[guard_rng(&rng) % (sizeof MARKS / sizeof MARKS[0])]);
        }
        if (it % 2) { n += ref_put(s + n, 0x62); }
        size_t na = ref_nfc(s, n, a), nb = nfc_whole(s, n, b, cap);
        CHECK(na == nb && memcmp(a, b, na) == 0, "long run %d (%zu marks)", it, m);
        size_t ends[3] = { n / 3, n / 2, n - 1 };
        nb = nfc_chunked(s, n, ends, 3, b, cap);
        CHECK(na == nb && memcmp(a, b, na) == 0, "long run %d chunked", it);
    }
    free(s); free(a); free(b);
}

/* ------------------------------------------------------------------------------------------- hand cases */

static void expect(const char *in, size_t nin, const char *want, size_t nwant, const char *what)
{
    uint8_t out[256];
    size_t n = nfc_whole((const uint8_t *)in, nin, out, sizeof out);
    CHECK(n == nwant && memcmp(out, want, n) == 0, "%s", what);
    uint8_t r[256];
    size_t nr = ref_nfc((const uint8_t *)in, nin, r);
    CHECK(nr == nwant && memcmp(r, want, nr) == 0, "%s (reference)", what);
}
#define EXPECT(in, want, what) expect(in, sizeof(in) - 1, want, sizeof(want) - 1, what)

static void test_hand(void)
{
    EXPECT("A\xCC\x8A", "\xC3\x85", "A + ring composes");
    EXPECT("A\xFF\xCC\x8A", "A\xFF\xCC\x8A", "an invalid byte between base and mark: no composition");
    EXPECT("A\xCC\x8A\xFF", "\xC3\x85\xFF", "mark then invalid byte");
    EXPECT("\xFF\xCC\x8A", "\xFF\xCC\x8A", "invalid byte then mark");
    EXPECT("e\xCC\x81\xFF\xCC\xA3", "\xC3\xA9\xFF\xCC\xA3", "the barrier ends the run: no reorder across it");
    EXPECT("e\xCC\x81\xCC\xA3", "\xE1\xBA\xB9\xCC\x81", "e + acute + dot below -> e-dot-below + acute");
    EXPECT("\xCC\x81\xCC\x96", "\xCC\x96\xCC\x81", "leading marks are reordered");
    EXPECT("\xCC\x81\x80\xCC\x96", "\xCC\x81\x80\xCC\x96", "a lone continuation byte is a barrier");
    EXPECT("\xE2\x82\xCC\x8A", "\xE2\x82\xCC\x8A", "a truncated sequence is two barriers");
    EXPECT("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81", "hangul L V T");
    EXPECT("\xEA\xB0\x80\xE1\x86\xA7", "\xEA\xB0\x80\xE1\x86\xA7", "LV + T_BASE does not compose");
    EXPECT("\xEA\xB0\x81\xE1\x86\xA8", "\xEA\xB0\x81\xE1\x86\xA8", "LVT + T does not compose");
    EXPECT("\xE2\x84\xAB", "\xC3\x85", "angstrom sign singleton");
    EXPECT("\xE0\xA5\x98", "\xE0\xA4\x95\xE0\xA4\xBC", "U+0958 is excluded from composition");
    EXPECT("\xCD\x84", "\xCC\x88\xCC\x81", "U+0344, a non-starter decomposition");
    EXPECT("\xF0\x9D\x85\xA0", "\xF0\x9D\x85\x98\xF0\x9D\x85\xA5\xF0\x9D\x85\xAE", "U+1D160: 3x");
    EXPECT("\xCE\xB1\xCC\x93\xCC\x80\xCD\x85", "\xE1\xBE\x82", "alpha + psili + grave + ypogegrammeni: 3 absorbed");
    EXPECT("\xC3\x80\xCC\x96", "\xC3\x80\xCC\x96", "precomposed + lower-class mark stays");
    EXPECT("", "", "empty");
    /* the 3x bound is reached */
    uint8_t s[400], o[1200];
    for (int i = 0; i < 100; i++) { memcpy(s + 4 * i, "\xF0\x9D\x85\xA0", 4); }
    size_t n = nfc_whole(s, 400, o, sizeof o);
    CHECK(n == 1200 && n == TOKS_NORM_BOUND(TOKS_NS_NFC, 400), "U+1D160 x 100 writes exactly 3n");
    /* api errors */
    CHECK(toks_norm(TOKS_NS_NFC, s, 400, o, 1199) == TOKS_E_CAP, "cap below the bound");
    CHECK(toks_norm(TOKS_NS_NFC, NULL, 0, NULL, 0) == 0, "empty with NULLs");
    /* the scan: what the driver copies */
    uint64_t re;
    CHECK(toks_nfc_scan(TOKS_NS_NFC, (const uint8_t *)"abc", 3, 0, &re) == 3, "ascii: no changed run");
    CHECK(toks_nfc_scan(TOKS_NS_NFC, (const uint8_t *)"\xC3\xA9\xCC\x96", 4, 0, &re) == 4,
          "a run already in NFC (boundary + an in-order mark that composes with nothing): the quick check");
    CHECK(toks_nfc_scan(TOKS_NS_NFC, (const uint8_t *)"\xC3\xA9\xCC\x81", 4, 0, &re) == 4,
          "a normalized run that maps to itself (the mark is a composition second)");
    CHECK(toks_nfc_scan(TOKS_NS_NFC, (const uint8_t *)"xA\xCC\x8A", 4, 0, &re) == 1 && re == 4, "changed: the run [1, 4)");
}

/* --------------------------------------------------------------------------------------------- K2, geometry */

static uint64_t k2(const uint8_t *p, uint64_t pos, uint64_t len) { return toks_k2(p, pos, len, TOKS_NFC_HOT_BYTE); }

static void test_k2_and_geometry(void)
{
    uint64_t rng = 99;
    uint8_t buf[600];
    for (int it = 0; it < 200000; it++) {
        size_t n = (size_t)(guard_rng(&rng) % 300), pos = n ? (size_t)(guard_rng(&rng) % (n + 1)) : 0;
        for (size_t i = 0; i < n; i++) { buf[i] = (uint8_t)(guard_rng(&rng) % 0xCC); }
        if (n && guard_rng(&rng) % 2) { buf[guard_rng(&rng) % n] = (uint8_t)(0xCC + guard_rng(&rng) % 0x34); }
        size_t want = pos;
        while (want < n && buf[want] < 0xCC) { want++; }
        CHECK(k2(buf, pos, n) == want, "k2 n=%zu pos=%zu", n, pos);
    }
    /* guard pages: every length 0..255 at every start 0..63, flush against no-access memory at either end */
    guard_buf g;
    uint8_t src[256], tmp[4 * 256 + 16], out[3 * 256], want[3 * 256];
    for (size_t n = 0; n < 256; n++) {
        size_t m = gen(&rng, tmp, n);
        if (m > n) { m = n; }
        memcpy(src, tmp, m);
        while (m < n) { src[m++] = 'a'; }
        size_t nw = ref_nfc(src, n, want);
        for (int where = 0; where < 2; where++) {
            for (size_t off = 0; off < 64; off++) {
                if (where == GUARD_END && off != 0) { break; }
                uint8_t *p = guard_alloc(&g, n, where, where == GUARD_START ? off : 0);
                if (!p) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
                memcpy(p, src, n);
                size_t h = 0;
                while (h < n && p[h] < 0xCC) { h++; }
                CHECK(k2(p, 0, n) == h, "k2 guard n=%zu off=%zu", n, off);
                size_t no = nfc_whole(p, n, out, sizeof out);
                CHECK(no == nw && memcmp(out, want, nw) == 0, "nfc guard n=%zu where=%d off=%zu", n, where, off);
                guard_free(&g);
            }
        }
    }
}

/* ------------------------------------------------------------------------------------ toks_nfc_boundary */

/* toks_nfc_boundary(f, text, len, i): 1 for an invalid byte or a code point with NB clear (the split planners ask it
 * about any atom: split.c and norm.c's y_base). Regression: an ascii atom used to go through the 4-byte
 * decode, reading text[i + 1 .. i + 3] past len and indexing toks_nfc_stage1 out of bounds (UBSan, guard pages). */
static void test_boundary(void)
{
    guard_buf g;
    uint8_t *p = guard_alloc(&g, 4, GUARD_END, 0);      /* p[0, 4) is flush against a no-access page */
    if (!p) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    for (uint32_t b = 0; b < 0x80u; b++) {
        CHECK((toks_nfc_info(b) & TOKS_NFC_NB) == 0, "ascii %02x has NB clear in the table", b);
        p[0] = (uint8_t)b; p[1] = 0xBF; p[2] = 0xBF; p[3] = 0xBF;   /* a 4-byte decode lands past stage1 */
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, p, 4, 0) == 1, "ascii %02x before continuation bytes", b);
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, p, 1, 0) == 1, "ascii %02x with len 1 inside a longer buffer", b);
    }
    for (uint32_t b = 0; b < 0x80u; b++) {
        p[3] = (uint8_t)b;                              /* the last byte before the guard page */
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, p + 3, 1, 0) == 1, "ascii %02x at the end of the input", b);
    }
    for (uint32_t b = 0x80; b < 0x100u; b++) {           /* alone: invalid (a truncated lead or a stray byte) */
        p[3] = (uint8_t)b;
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, p + 3, 1, 0) == 1, "lone byte %02x is invalid: a boundary", b);
    }
    p[2] = 0xE2; p[3] = 0x82;                           /* a truncated 3-byte sequence at the end */
    CHECK(toks_nfc_boundary(TOKS_NS_NFC, p + 2, 2, 0) == 1, "truncated E2 82 is invalid: a boundary");
    uint64_t nb = 0;
    uint8_t in[12], o1[64], o2[64];
    for (uint32_t cp = 0x80; cp < 0x110000; cp++) {
        if (cp >= 0xD800 && cp <= 0xDFFF) { continue; }
        uint8_t e[4];
        size_t k = ref_put(e, cp);
        memcpy(p + 4 - k, e, k);
        int want = (toks_nfc_info(cp) & TOKS_NFC_NB) == 0;
        nb += (uint64_t)(want == 0);
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, p + 4 - k, k, 0) == want, "U+%04X at the end of the input", cp);
        in[0] = 'a'; memcpy(in + 1, e, k);
        CHECK(toks_nfc_boundary(TOKS_NS_NFC, in, 1 + k, 1) == want, "U+%04X at i = 1", cp);
        if (want) {                                     /* what a boundary means: a split before it is exact */
            static const char *const pre[] = { "e", "\xE1\x84\x80", "\xEA\xB0\x80", "a\xCC\x81" };
            for (size_t j = 0; j < sizeof pre / sizeof pre[0]; j++) {
                size_t np = strlen(pre[j]);
                memcpy(in, pre[j], np); memcpy(in + np, e, k);
                size_t n1 = nfc_whole(in, np + k, o1, sizeof o1);
                size_t na = nfc_whole(in, np, o2, sizeof o2);
                size_t nc = nfc_whole(e, k, o2 + na, sizeof o2 - na);
                CHECK(n1 == na + nc && memcmp(o1, o2, n1) == 0, "split before boundary U+%04X after %s", cp, pre[j]);
            }
        }
    }
    CHECK(nb > 1000, "%llu scalars are not boundaries", (unsigned long long)nb);
    guard_free(&g);
}

/* ------------------------------------------------------------------------------------------ NFKC / NFKD */

static void test_compat(void)
{
    FILE *f = fopen("tests/norm/nfkc_golden.txt", "rb");
    CHECK(f != NULL, "tests/norm/nfkc_golden.txt");
    char line[4096];
    uint8_t in[512], want[2][2048], out[8192];
    size_t ncase = 0;
    td_at at = { "tests/norm/nfkc_golden.txt", 0 };
    while (f != NULL && fgets(line, sizeof line, f)) {
        at.line++;
        if (!td_line(&at, line, sizeof line)) { n_fail++; break; }
        if (line[0] == '#') { continue; }
        char *s1 = strchr(line, ' '), *s2 = s1 ? strchr(s1 + 1, ' ') : NULL;
        if (!s2) { n_fail += !td_bad(&at, line + strlen(line) - 1, "'<hex> <hex> <hex>' expected"); break; }
        size_t pre = ncase++ % 130, nin = unhex(line, in + pre), nw[2] = { unhex(s1 + 1, want[0]), unhex(s2 + 1, want[1]) };
        memset(in, '#', pre);
        for (int k = 0; k < 2; k++) {
            int64_t m = toks_norm(k == 0 ? TOKS_NS_NFKC : TOKS_NS_NFKD, in, pre + nin, out, sizeof out);
            CHECK(m == (int64_t)(pre + nw[k]) && memcmp(out + pre, want[k], nw[k]) == 0, "%s case %zu: %.40s",
                  k == 0 ? "NFKC" : "NFKD", ncase, line);
        }
    }
    CHECK(ncase > 7000, "nfkc_golden lists %zu cases", ncase);
    if (f != NULL) { fclose(f); }
    static const uint32_t FORMS[] = { TOKS_NS_NFC, TOKS_NS_NFKC, TOKS_NS_CANON, TOKS_NS_NFKD, TOKS_NS_STRIP_M, TOKS_NS_LOWER,
        TOKS_NS_STRIP_M | TOKS_NS_LOWER, TOKS_NS_CANON | TOKS_NS_STRIP_M, TOKS_NS_CANON | TOKS_NS_LOWER,
        TOKS_NS_CANON | TOKS_NS_STRIP_M | TOKS_NS_LOWER, TOKS_NS_NFKD | TOKS_NS_STRIP_M, TOKS_NS_NFKD | TOKS_NS_LOWER,
        TOKS_NS_NFKD | TOKS_NS_STRIP_M | TOKS_NS_LOWER };
    for (size_t k = 0; k < sizeof FORMS / sizeof FORMS[0]; k++) {
        for (uint32_t cp = 0; cp < 0x110000; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) { continue; }
            size_t n = ref_put(in, cp);
            int64_t m = toks_norm(FORMS[k], in, n, out, sizeof out);
            if (m < 0 || (uint64_t)m > TOKS_NORM_X(FORMS[k]) * n) { CHECK(0, "form 0x%x U+%04X: %lld bytes", FORMS[k], cp, (long long)m); }
        }
        n_checks++;
    }
    /* albert's chain on a few cases (hf: NFKD, StripAccents, Lowercase) */
    static const char *const A[][2] = { { "\xC3\x89t\xC3\xA9", "ete" }, { "\xEF\xAC\x81", "fi" }, { "\xE2\x84\xAB", "a" },
        { "\xC4\xB0", "i" }, { "\xE1\xBE\x8A", "\xCE\xB1" }, { "\xEF\xBC\xA1\xCC\x81", "a" } };
    for (size_t k = 0; k < sizeof A / sizeof A[0]; k++) {
        int64_t m = toks_norm(TOKS_NS_NFKD | TOKS_NS_STRIP_M | TOKS_NS_LOWER, (const uint8_t *)A[k][0], strlen(A[k][0]), out,
                              sizeof out);
        CHECK(m == (int64_t)strlen(A[k][1]) && memcmp(out, A[k][1], (size_t)m) == 0, "albert chain case %zu", k);
    }
}

/* toks_nfc_fast (norm_nfc.h, the scan's clean_run): bit cp of form f is set exactly when the BMP code point cp is not
 * a surrogate and its info has NB (NFC) / NBK (NFKC) clear: the generator's table against the generator's info */
static void test_fast(void)
{
    for (uint32_t cp = 0; cp < 0x10000u; cp++) {
        uint32_t w = toks_nfc_info(cp), sur = cp >= 0xD800u && cp <= 0xDFFFu;
        uint32_t b0 = (uint32_t)(toks_nfc_fast[cp >> 6] >> (cp & 63u)) & 1u;
        uint32_t b1 = (uint32_t)(toks_nfc_fast[1024u + (cp >> 6)] >> (cp & 63u)) & 1u;
        CHECK(b0 == (uint32_t)(!sur && (w & TOKS_NFC_NB) == 0u), "toks_nfc_fast NFC U+%04X: %u", cp, b0);
        CHECK(b1 == (uint32_t)(!sur && (w & TOKS_NFC_NBK) == 0u), "toks_nfc_fast NFKC U+%04X: %u", cp, b1);
    }
}

int main(void)
{
    ref_init();
    load_golden();
    test_hand();
    test_boundary();
    test_fast();
    test_golden();
    test_every_scalar();
    test_generated();
    test_long_runs();
    test_k2_and_geometry();
    test_compat();
    printf("test_norm: %zu golden cases, %llu checks, %llu failures\n", NG, (unsigned long long)n_checks,
           (unsigned long long)n_fail);
    return n_fail != 0;
}
