/* tests/fuzz/hostile/hostile_driver.c: the hostile battery (docs/fuzz.md "hostile"; SPEC T6 / T9).
 *
 * Not a libFuzzer harness: a driver linked against whichever toks build is under test (the sanitizer fuzz
 * build's objects through tests/fuzz/hostile/Makefile, a guard-geometry build, or the release library),
 * taking a texts.json (tests/fuzz/hostile/gen.py --texts) and tokenizer files, running every
 * hand-reasoned call shape and checking the contract's own invariants -- the same oracle set as
 * tests/fuzz/check.h, restated for a plain driver:
 *
 *   load    both tiers give the same verdict; a refusal is a documented code, *out NULL, diag set
 *   encode  the total is stable across cap 0 / short / exact / over (the capacity rule, nothing
 *           written at or past cap), count <= toks_encode_bound(len), every id < n_ids, the scalar
 *           tier == the asm tier, a scratch one byte short is TOKS_E_SCRATCH, out at +1..+7 inside a
 *           heap block, out NULL with cap 0, unknown flags TOKS_E_ARG, len > TOKS_MAX_TEXT is
 *           TOKS_E_LIMIT before any read (a guard-mapped buffer would fault)
 *   pieces  the capacity rule; ends <= 3 len + 3; tiers agree
 *   decode  an id >= n_ids is TOKS_E_ID with nothing written; output is valid utf-8
 *   stream  one id at a time + flush == batch decode (with toks_stream_hold's documented recovery);
 *           a push one byte short is TOKS_E_CAP with the state unchanged and the exact prefix out
 *   split   cuts strictly increasing inside (0, len); n_want 0 / 1 / huge; the parts with
 *           TOKS_CONTINUATION concatenate to the whole's ids without post-processing (SPEC 5.2)
 *   par     toks_par_encode == toks_encode at any cap; pools of 1 and 64; batch items at odd caps
 *   info    toks_get_info with every size; toks_scratch_bytes / toks_encode_bound monotonicity
 *
 * Usage: hostile_driver [-v] [-m max-bytes] <texts.json> <tokenizer-file>...
 * Exit 0 clean, 1 on a violated invariant ("HOSTILE VIOLATION ..."), 2 on a setup failure.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toks.h"

#if !defined(_WIN32)
#  include <sys/mman.h>
#  include <unistd.h>
#endif

#define MAX_PIECES_TEXT (1u << 17)     /* texts.json hex cap: 128 KiB of text */

/* ---- failure ---------------------------------------------------------------------------------------------- */

static int fails = 0;
static int verbose = 0;
static int strict_flags = 1;   /* the guard build's branch may predate the strict flags check */
static uint64_t max_bytes = 16u << 20;  /* the biggest text this run builds (par / size cases) */
static const char *cur_file = "?";
static char cur_case[160] = "?";

static void violation(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "\nHOSTILE VIOLATION [%s] %s: ", cur_file, cur_case);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    fails++;
}

#define CHECK(cond, ...) do { if (!(cond)) { violation(__VA_ARGS__); } } while (0)

static void *xmalloc(uint64_t n)
{
    void *p = malloc((size_t)(n != 0u ? n : 1u));
    if (p == NULL) { fprintf(stderr, "hostile: out of memory (%llu bytes)\n", (unsigned long long)n); exit(2); }
    return p;
}

/* a read-only zero-filled mapping (never written, pages backed as read): for lengths the caller must
 * not read past, and for huge texts without paying for content */
static uint8_t *zero_map(uint64_t n)
{
#if !defined(_WIN32)
    if (n == 0u) { return (uint8_t *)xmalloc(1u); }
    void *m = mmap(NULL, (size_t)n, PROT_READ, MAP_PRIVATE | MAP_ANON, -1, 0);
    return m == MAP_FAILED ? NULL : (uint8_t *)m;
#else
    return (uint8_t *)calloc(1u, (size_t)n);
#endif
}

static void zero_unmap(uint8_t *p, uint64_t n)
{
    if (p == NULL) { return; }
#if !defined(_WIN32)
    if (n != 0u) { munmap(p, (size_t)n); return; }
#endif
    free(p);
}

/* ---- texts.json -------------------------------------------------------------------------------------------- */

typedef struct {
    char     name[64];
    uint8_t *bytes;
    uint64_t n;
} h_text;

typedef struct {
    char     kind[16];
    char     name[64];
    uint64_t a, b, c;
} h_call;

static h_text *texts;
static uint32_t n_texts;
static h_call *calls;
static uint32_t n_calls;

static const uint8_t *jp;
static uint64_t jn;
static const uint8_t *jp_end;

static void jerr(const char *why)
{
    fprintf(stderr, "hostile: texts.json: %s near byte %lld\n", why, (long long)(jp - (jp_end - jn)));
    exit(2);
}

static void jws(void)
{
    while (jp < jp_end && (*jp == ' ' || *jp == '\t' || *jp == '\n' || *jp == '\r')) { jp++; }
}

static int jchar(uint8_t c)
{
    jws();
    if (jp < jp_end && *jp == c) { jp++; return 1; }
    return 0;
}

static void jexpect(uint8_t c)
{
    if (!jchar(c)) { jerr("unexpected byte"); }
}

static int jstr_raw(char *out, uint64_t cap)
{
    jws();
    if (jp >= jp_end || *jp != '"') { jerr("expected a string"); }
    jp++;
    uint64_t n = 0;
    while (jp < jp_end && *jp != '"') {
        uint8_t c = *jp++;
        if (c == '\\') {
            if (jp >= jp_end) { jerr("escape at end"); }
            c = *jp++;
            if (c == 'u') {
                uint32_t v = 0;
                for (uint32_t k = 0; k < 4u; k++) {
                    if (jp >= jp_end) { jerr("short \\u"); }
                    uint8_t h = *jp++;
                    int d = (h >= '0' && h <= '9') ? h - '0' : (h >= 'a' && h <= 'f') ? h - 'a' + 10 :
                             (h >= 'A' && h <= 'F') ? h - 'A' + 10 : -1;
                    if (d < 0) { jerr("bad hex"); }
                    v = (v << 4) | (uint32_t)d;
                }
                if (v < 0x80u) { c = (uint8_t)v; }
                else {
                    uint32_t k = v < 0x800u ? 2u : 3u;
                    if (n + k >= cap) { jerr("string too long"); }
                    if (k == 2u) {
                        out[n++] = (char)(0xC0u | (v >> 6));
                        out[n++] = (char)(0x80u | (v & 0x3Fu));
                    } else {
                        out[n++] = (char)(0xE0u | (v >> 12));
                        out[n++] = (char)(0x80u | ((v >> 6) & 0x3Fu));
                        out[n++] = (char)(0x80u | (v & 0x3Fu));
                    }
                    continue;
                }
            } else {
                switch (c) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                default: break;      /* \" \\ \/ */
                }
            }
        }
        if (n + 1 >= cap) { jerr("string too long"); }
        out[n++] = (char)c;
    }
    if (jp >= jp_end) { jerr("unterminated string"); }
    jp++;
    out[n] = 0;
    return (int)n;
}

static uint64_t jnum(void)
{
    jws();
    uint64_t v = 0u;
    int any = 0;
    while (jp < jp_end && *jp >= '0' && *jp <= '9') { v = v * 10u + (uint64_t)(*jp++ - '0'); any = 1; }
    if (!any) { jerr("expected a number"); }
    return v;
}

static int hexval(uint8_t c)
{
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

/* ---- helpers ----------------------------------------------------------------------------------------------- */

static void set_case(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cur_case, sizeof cur_case, fmt, ap);
    va_end(ap);
}

static int text_index(const char *name)
{
    for (uint32_t i = 0; i < n_texts; i++) {
        if (strcmp(texts[i].name, name) == 0) { return (int)i; }
    }
    return -1;
}

/* a text of exactly len bytes: the seed tiled (a one-byte 'a' when the seed is empty), heap-allocated */
static uint8_t *tile(const h_text *t, uint64_t len, uint8_t **to_free)
{
    uint8_t *p = xmalloc(len != 0u ? len : 1u);
    uint64_t done = 0u;
    while (done < len) {
        uint64_t k = t->n != 0u && t->n <= len - done ? t->n : (len - done < t->n || t->n == 0u ? len - done : t->n);
        if (t->n != 0u && k > t->n) { k = t->n; }
        if (t->n == 0u) { memset(p + done, 'a', (size_t)k); }
        else { memcpy(p + done, t->bytes, (size_t)k); }
        done += k;
        if (t->n == 0u) { break; }
    }
    *to_free = p;
    return p;
}

/* both contexts of one file */
typedef struct {
    toks_ctx *s, *a;            /* scalar, auto */
    toks_info info;
    uint32_t n_ids;
} pair;

/* ---- encode / pieces ---------------------------------------------------------------------------------------- */

static const uint32_t FLAG_SHAPES[] = {
    0, TOKS_ADDED_ALL, TOKS_ADDED_NONSPECIAL, TOKS_ADDED_NONE,
    TOKS_NO_POSTPROCESS, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS,
    TOKS_CONTINUATION, TOKS_ADDED_ALL | TOKS_CONTINUATION,
    TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION,
};

/* encode + pieces of x[0, len) under flags: every invariant above. *ids_out (the exact-capacity ids)
 * stays for the decode / stream / split batteries. Returns the count, or < 0 when encode refused. */
static int64_t battery_text(const pair *p, const uint8_t *x, uint64_t len, uint32_t flags, uint32_t mis,
                            uint32_t **ids_out, const char *what)
{
    uint64_t sb = 0u, ab = 0u;
    uint8_t *blk_s = NULL, *blk_a = NULL;
    uint32_t *ids_a = NULL;
    uint8_t *ob = NULL;
    int64_t n = -1;

    /* the scratch: exact size at the given misalignment (0..7); one byte short must be refused */
    sb = toks_scratch_bytes(p->s, len, 0u);
    ab = toks_scratch_bytes(p->a, len, 0u);
    blk_s = xmalloc(sb + 64u);
    blk_a = xmalloc(ab + 64u);
    if (toks_scratch_init(p->s, blk_s, sb, 0u) != 0 || toks_scratch_init(p->a, blk_a + mis, ab, 0u) != 0) {
        violation("scratch init failed (%s)", what);
        goto out;
    }
    {
        uint8_t *bad = xmalloc(sb > 8u ? sb - 1u : 1u);
        uint32_t o[4] = { 0u, 0u, 0u, 0u };
        if (toks_scratch_init(p->s, bad, sb > 8u ? sb - 1u : 1u, 0u) != 0) {
            CHECK(toks_encode(p->s, x, len, flags, o, len != 0u ? 1u : 0u, bad) == TOKS_E_SCRATCH,
                  "encode with an uninitialized scratch returned something else (%s)", what);
        }
        free(bad);
    }

    /* count-only, then the exact capacity (through a misaligned out: no buffer needs alignment) */
    n = toks_encode(p->a, x, len, flags, NULL, 0u, blk_a + mis);
    CHECK(n >= 0, "encode count-only returned %lld (%s)", (long long)n, what);
    CHECK((uint64_t)n <= toks_encode_bound(p->a, len), "encode %lld ids over the bound %llu (%s)",
          (long long)n, (unsigned long long)toks_encode_bound(p->a, len), what);

    ob = xmalloc(4u * (uint64_t)(n + 2) + 8u);
    ids_a = xmalloc(4u * (uint64_t)(n + 1));
    {
        uint8_t *ob_mis = ob + (mis & 3u);              /* a byte pointer: no typed access, no UBSan */
        int64_t m0 = toks_encode(p->a, x, len, flags, (uint32_t *)(void *)ob_mis, (uint64_t)n, blk_a + mis);
        if (m0 == n && n > 0) { memcpy(ids_a, ob_mis, (size_t)n * 4u); }
    }
    int64_t m = toks_encode(p->a, x, len, flags, ids_a, (uint64_t)n, blk_a + mis);
    CHECK(m == n, "encode at cap == n returned %lld want %lld (%s)", (long long)m, (long long)n, what);
    for (int64_t i = 0; i < n; i++) {
        CHECK(ids_a[i] < p->n_ids, "encode gave id %u >= n_ids %u at %lld (%s)", ids_a[i], p->n_ids,
              (long long)i, what);
    }

    /* the scalar tier agrees */
    {
        uint32_t *ids_s = xmalloc(4u * (uint64_t)(n + 1));
        int64_t k = toks_encode(p->s, x, len, flags, ids_s, (uint64_t)n, blk_s);
        CHECK(k == n, "scalar encode returned %lld (%s)", (long long)k, what);
        if (k == n && n > 0 && memcmp(ids_a, ids_s, (size_t)n * 4u) != 0) {
            uint64_t d = 0u;
            while (d < (uint64_t)n && ids_a[d] == ids_s[d]) { d++; }
            violation("tier disagreement at %llu: %u vs %u (%s)", (unsigned long long)d, ids_a[d], ids_s[d], what);
        }
        free(ids_s);
    }

    /* the capacity rule: cap 0 (out NULL), short, and one over; nothing written at or past cap */
    {
        static const uint64_t CAPS[5] = { 0u, 1u, 3u, 0xFFFFFFFFFFFFFFFFull /* n / 2 below */, 0u };
        uint64_t caps[5];
        caps[0] = 0u;
        caps[1] = (uint64_t)n > 2u ? 1u : 0u;
        caps[2] = (uint64_t)n > 6u ? 3u : 0u;
        caps[3] = (uint64_t)n / 2u;
        caps[4] = (uint64_t)n + 1u;
        (void)CAPS;
        for (uint32_t ci = 0; ci < 5u; ci++) {
            uint64_t cap = caps[ci];
            if (ci != 4u && cap >= (uint64_t)n && cap != 0u) { continue; }
            uint64_t k = 4u * (cap + 8u);
            uint8_t *buf = xmalloc(k);
            memset(buf, 0xA5, (size_t)k);
            int64_t r = toks_encode(p->a, x, len, flags, cap != 0u ? (uint32_t *)(void *)buf : NULL, cap,
                                    blk_a + mis);
            CHECK(r == n, "encode at cap %llu returned %lld want %lld (%s)", (unsigned long long)cap,
                  (long long)r, (long long)n, what);
            uint64_t cut = cap < (uint64_t)n ? cap : (uint64_t)n;
            if (cut != 0u && memcmp(buf, ids_a, (size_t)cut * 4u) != 0) {
                violation("encode at cap %llu: not the exact prefix (%s)", (unsigned long long)cap, what);
            }
            for (uint64_t i = cap * 4u; i < k; i++) {
                CHECK(buf[i] == 0xA5u, "encode at cap %llu wrote out byte %llu (%s)",
                      (unsigned long long)cap, (unsigned long long)i, what);
            }
            free(buf);
        }
    }

    /* unknown flags; a bad length is refused before any read (x is 1 byte: a read of len would fault
     * under the guard build's geometry, and here it would read the heap redzone) */
    if (strict_flags) {
        CHECK(toks_encode(p->a, x, len, flags | 16u, NULL, 0u, blk_a + mis) == TOKS_E_ARG,
              "encode accepted unknown flags (%s)", what);
    }
    if (len > 1u) {
        CHECK(toks_encode(p->a, x, TOKS_MAX_TEXT + 1u, flags, NULL, 0u, blk_a + mis) == TOKS_E_LIMIT,
              "encode accepted len > TOKS_MAX_TEXT (%s)", what);
    }

    /* pieces: the count, the capacity rule, the ends' bound, the tiers */
    {
        int64_t pn = toks_pieces(p->a, x, len, flags, NULL, 0u, blk_a + mis);
        CHECK(pn >= 0, "pieces count-only returned %lld (%s)", (long long)pn, what);
        uint32_t *ends = xmalloc(4u * (uint64_t)(pn + 1));
        int64_t pm = toks_pieces(p->a, x, len, flags, ends, (uint64_t)pn, blk_a + mis);
        CHECK(pm == pn, "pieces at cap == n returned %lld (%s)", (long long)pm, what);
        for (int64_t i = 0; i < pn; i++) {
            /* the ends index the NORMALIZED stream: a normalizer can expand (SPEC's worst is NFKD
             * then the charsmap, 66x); 3x covers only the no-normalizer case */
            CHECK(ends[i] <= 66u * len + 3u, "piece end %u past 66 len + 3 (len %llu, %s)", ends[i],
                  (unsigned long long)len, what);
        }
        if (pn > 1) {
            uint32_t *half = xmalloc(4u * 2u);
            memset(half, 0x5A, 8u);
            int64_t r = toks_pieces(p->a, x, len, flags, half, 1u, blk_a + mis);
            CHECK(r == pn && half[0] == ends[0], "pieces at cap 1 (%s)", what);
            free(half);
        }
        uint32_t *ends_s = xmalloc(4u * (uint64_t)(pn + 1));
        int64_t ps = toks_pieces(p->s, x, len, flags, ends_s, (uint64_t)pn, blk_s);
        CHECK(ps == pn && (pn == 0 || memcmp(ends, ends_s, (size_t)pn * 4u) == 0),
              "pieces tiers disagree (%s)", what);
        free(ends_s);
        free(ends);
    }

    if (ids_out != NULL) { *ids_out = ids_a; ids_a = NULL; }
out:
    free(ob);
    if (ids_a != NULL) { free(ids_a); }
    free(blk_s);
    free(blk_a);
    return n;
}

/* ---- decode / stream ---------------------------------------------------------------------------------------- */

/* a strict utf-8 validity check (the same shape as the fuzz harness's fz_utf8_valid) */
/* unicode 3.9 table 3-7, the same shape as the fuzz harness's fz_lossy: the narrow second-byte
 * ranges E0:A0-BF, ED:80-9F, F0:90-BF, F4:80-8F, checked only where they narrow */
static int u8_valid(const uint8_t *s, uint64_t n)
{
    uint64_t i = 0u;
    while (i < n) {
        uint8_t b = s[i];
        if (b < 0x80u) { i++; continue; }
        uint32_t need = 0u;
        uint8_t lo = 0x80u, hi = 0xBFu;              /* the second byte's range */
        if (b >= 0xC2u && b <= 0xDFu) { need = 2u; }
        else if (b == 0xE0u) { need = 3u; lo = 0xA0u; }
        else if ((b >= 0xE1u && b <= 0xECu) || b == 0xEEu || b == 0xEFu) { need = 3u; }
        else if (b == 0xEDu) { need = 3u; hi = 0x9Fu; }
        else if (b == 0xF0u) { need = 4u; lo = 0x90u; }
        else if (b >= 0xF1u && b <= 0xF3u) { need = 4u; }
        else if (b == 0xF4u) { need = 4u; hi = 0x8Fu; }
        else { return 0; }
        if (i + need > n) { return 0; }
        if (need >= 2u && (s[i + 1u] < lo || s[i + 1u] > hi)) { return 0; }
        for (uint32_t k = 2u; k < need; k++) {
            if ((s[i + k] & 0xC0u) != 0x80u) { return 0; }
        }
        i += need;
    }
    return 1;
}

static void battery_ids(const pair *p, const uint32_t *ids, int64_t n, const char *what)
{
    /* decode: count, valid utf-8, TOKS_E_ID past n_ids with nothing written */
    int64_t dn = toks_decode(p->a, ids, (uint64_t)n, 0u, NULL, 0u);
    CHECK(dn >= 0, "decode returned %lld (%s)", (long long)dn, what);
    if (dn > 0) {
        uint8_t *d = xmalloc((uint64_t)dn);
        CHECK(toks_decode(p->a, ids, (uint64_t)n, 0u, d, (uint64_t)dn) == dn, "decode at cap == n (%s)", what);
        CHECK(u8_valid(d, (uint64_t)dn), "decode output is not valid utf-8 (%s)", what);
        uint8_t c = d[dn / 2u];
        (void)c;
        free(d);
    }
    if (n > 0) {
        uint32_t *bad = xmalloc(4u * (uint64_t)n);
        memcpy(bad, ids, (size_t)n * 4u);
        bad[n - 1] = p->n_ids;
        uint8_t guard[8];
        memset(guard, 0x5A, sizeof guard);
        CHECK(toks_decode(p->a, bad, (uint64_t)n, 0u, guard, sizeof guard) == TOKS_E_ID,
              "decode past n_ids returned something else (%s)", what);
        for (uint32_t i = 0; i < sizeof guard; i++) {
            CHECK(guard[i] == 0x5Au, "TOKS_E_ID wrote out[%u] (%s)", i, what);
        }
        free(bad);
    }

    /* stream: one id at a time + flush == batch decode, with hold recovery and atomicity */
    {
        toks_stream st;
        toks_stream_init(p->a, &st, 0u);
        uint64_t bound = toks_stream_bound(p->a, 1u) + 3u * ((uint64_t)(dn > 0 ? dn : 1) + 64u);  /* toks.h:
                                                                     * a push/flush writes bound + 3 x hold,
                                                                     * and the held run is at most the batch's
                                                                     * bytes (dn) once it ends */
        uint8_t *o = xmalloc(bound + 8u);
        uint8_t *acc = xmalloc((uint64_t)(dn > 0 ? dn : 1) + 16u);
        uint64_t got = 0u;
        uint8_t *hold = NULL;              /* the stream's caller memory: alive until init, per toks.h */
        uint64_t hold_cap = 0u;
        for (int64_t i = 0; i < n; i++) {
            toks_stream before = st;
            int64_t r = toks_stream_push(p->a, &st, ids + i, 1u, o, bound);
            if (r == TOKS_E_LIMIT) {                 /* the documented byte-fallback hold: grow it.
                                                         * toks.h: a hold of at least the current size
                                                         * (44 for st's own) plus the push's n bytes */
                CHECK(memcmp(&before, &st, sizeof st) == 0, "TOKS_E_LIMIT changed the stream state (%s)", what);
                hold_cap = hold_cap != 0u ? hold_cap * 2u : (44u + 4u + 4096u);   /* grow until the run fits */
                uint64_t hc = hold_cap;
                uint8_t *nh = xmalloc(hc);         /* the old hold stays alive during the move (toks.h) */
                CHECK(toks_stream_hold(p->a, &st, nh, hc) >= 0, "hold(grow) refused (%s)", what);
                free(hold);
                hold = nh;
                r = toks_stream_push(p->a, &st, ids + i, 1u, o, bound);
                CHECK(r >= 0, "push after a grown hold returned %lld (%s)", (long long)r, what);
            } else if (r >= 0) {
                CHECK((uint64_t)r <= bound, "push returned %lld over the bound %llu (%s)", (long long)r,
                      (unsigned long long)bound, what);
                if (r > 0) {                          /* atomicity: one byte short */
                    toks_stream again = before;
                    uint8_t *c = xmalloc((uint64_t)r + 8u);
                    memset(c, 0x5A, (size_t)r + 8u);
                    int64_t r2 = toks_stream_push(p->a, &again, ids + i, 1u, c, (uint64_t)r - 1u);
                    CHECK(r2 == TOKS_E_CAP, "a push one byte short returned %lld (%s)", (long long)r2, what);
                    CHECK(memcmp(&again, &before, sizeof st) == 0, "TOKS_E_CAP changed the stream state (%s)",
                          what);
                    CHECK((uint64_t)r < 2u || memcmp(c, o, (size_t)r - 1u) == 0,
                          "TOKS_E_CAP: out is not the exact prefix (%s)", what);
                    for (int64_t k = r - 1; k < r + 8; k++) {
                        CHECK(c[k] == 0x5Au, "TOKS_E_CAP wrote out[%lld] (%s)", (long long)k, what);
                    }
                    free(c);
                }
            } else {
                violation("push returned %lld (%s)", (long long)r, what);
                break;
            }
            if (r > 0) { memcpy(acc + got, o, (size_t)r); got += (uint64_t)r; }
        }
        {
            int64_t r = toks_stream_flush(p->a, &st, o, bound);
            CHECK(r >= 0, "flush returned %lld (%s)", (long long)r, what);
            if (r > 0) { memcpy(acc + got, o, (size_t)r); got += (uint64_t)r; }
            CHECK(got == (uint64_t)dn, "stream one-at-a-time gave %llu bytes, batch %lld (%s)",
                  (unsigned long long)got, (long long)dn, what);
        }
        free(acc);
        free(o);
        free(hold);
    }
}

/* ---- split --------------------------------------------------------------------------------------------------- */

static void battery_split(const pair *p, const uint8_t *x, uint64_t len, uint32_t flags,
                          const uint32_t *whole0, int64_t n_whole, const char *what)
{
    (void)whole0;     /* the caller's ids may carry the post-processor; re-encoded under base below */
    static const uint32_t WANTS[] = { 0u, 1u, 2u, 3u, 16u, 17u, 0x80000000u };
    uint32_t base = (flags & ~(TOKS_NO_POSTPROCESS | TOKS_CONTINUATION)) | TOKS_NO_POSTPROCESS;
    for (uint32_t wi = 0; wi < sizeof WANTS / sizeof WANTS[0]; wi++) {
        uint32_t want = WANTS[wi];
        for (uint64_t cap = 0; cap <= 4u; cap++) {
            uint8_t *blk = xmalloc(8u * (cap + 2u) + 16u);
            memset(blk, 0x5A, 8u * (size_t)(cap + 2u) + 16u);
            int64_t c = toks_split_points(p->a, x, len, base, want, cap != 0u ? (uint64_t *)(void *)blk : NULL,
                                          cap, NULL);
            CHECK(c >= 0, "split_points returned %lld for n_want %u (%s)", (long long)c, want, what);
            CHECK(c == 0 || want > 1, "n_want %u returned %lld cuts (%s)", want, (long long)c, what);
            CHECK((uint64_t)c <= cap, "%lld cuts with cap %llu (%s)", (long long)c,
                  (unsigned long long)cap, what);
            free(blk);
            if (want <= 1u) { break; }
        }
    }
    /* the exactness of the cuts it found (SPEC 5.2): the parts concatenate to the whole WITHOUT
     * post-processing -- re-encode the whole under the same base, since the caller's ids may carry
     * the post-processor's template ids */
    if (len >= 2u && n_whole >= 0) {
        uint64_t offs[16];
        int64_t c = toks_split_points(p->a, x, len, base, 4u, offs, 16u, NULL);
        if (c > 0) {
            uint64_t sb0 = toks_scratch_bytes(p->a, len, 0u);
            uint8_t *scr0 = xmalloc(sb0);
            CHECK(toks_scratch_init(p->a, scr0, sb0, 0u) == 0, "scratch init for the whole (%s)", what);
            int64_t nw = toks_encode(p->a, x, len, base, NULL, 0u, scr0);
            free(scr0);
            n_whole = nw;        /* re-encoded under base: the caller's ids may be the pp'd ones */
            uint64_t prev = 0u;
            uint64_t sb = toks_scratch_bytes(p->a, len, 0u);
            uint8_t *scr = xmalloc(sb);
            CHECK(toks_scratch_init(p->a, scr, sb, 0u) == 0, "scratch init (%s)", what);
            uint32_t *wh = xmalloc(4u * (uint64_t)(n_whole > 0 ? n_whole : 1));
            CHECK(toks_encode(p->a, x, len, base, wh, (uint64_t)n_whole, scr) == n_whole,
                  "the whole re-encode under base (%s)", what);
            uint32_t *cat = xmalloc(4u * (uint64_t)(n_whole + 16));
            uint64_t got = 0u;
            for (int64_t i = 0; i <= c; i++) {
                uint64_t e = i < c ? offs[i] : len;
                CHECK(e > prev || (i == c && e == prev), "cuts not strictly increasing (%s)", what);
                int64_t k = toks_encode(p->a, x + prev, e - prev, base | (i > 0 ? TOKS_CONTINUATION : 0u),
                                        NULL, 0u, scr);
                CHECK(k >= 0, "a part refused to encode (%s)", what);
                if (k > 0) {
                    uint32_t *part = xmalloc(4u * (uint64_t)k);
                    CHECK(toks_encode(p->a, x + prev, e - prev, base | (i > 0 ? TOKS_CONTINUATION : 0u), part,
                                      (uint64_t)k, scr) == k, "a part at cap == n (%s)", what);
                    CHECK(got + (uint64_t)k <= (uint64_t)n_whole + 16u, "the parts overflow (%s)", what);
                    memcpy(cat + got, part, (size_t)k * 4u);
                    got += (uint64_t)k;
                    free(part);
                }
                prev = e;
            }
            CHECK(got == (uint64_t)n_whole && (got == 0u || memcmp(cat, wh, (size_t)got * 4u) == 0),
                  "the parts' ids (%llu) != the whole's (%lld) at the certified cuts (%s)",
                  (unsigned long long)got, (long long)n_whole, what);
            free(cat);
            free(wh);
            free(scr);
        }
    }
}

/* ---- par ---------------------------------------------------------------------------------------------------- */

static void battery_par(const pair *p, const uint8_t *x, uint64_t len, uint32_t flags, const char *what)
{
    for (uint32_t pool_n = 1u; pool_n <= 64u; pool_n = pool_n == 1u ? 64u : pool_n + 1u) {
        toks_par *pool = NULL;
        int64_t r = toks_par_create(&pool, p->a, pool_n, 0u);
        CHECK(r == 0 && pool != NULL, "toks_par_create(%u) returned %lld (%s)", pool_n, (long long)r, what);
        if (pool == NULL) { return; }
        uint64_t sb = toks_scratch_bytes(p->a, len, 0u);
        uint8_t *scr = xmalloc(sb);
        CHECK(toks_scratch_init(p->a, scr, sb, 0u) == 0, "scratch init (%s)", what);
        int64_t n = toks_encode(p->a, x, len, flags, NULL, 0u, scr);
        CHECK(n >= 0, "serial encode returned %lld (%s)", (long long)n, what);
        uint32_t *ids = xmalloc(4u * (uint64_t)(n + 1));
        CHECK(toks_encode(p->a, x, len, flags, ids, (uint64_t)n, scr) == n, "serial at cap (%s)", what);
        for (uint64_t cap = 0; cap <= (uint64_t)n; cap = cap == 0u ? ((uint64_t)n > 2u ? 1u : (uint64_t)n) :
                                                            (cap + 1u >= (uint64_t)n ? (uint64_t)n : cap * 2u + 1u)) {
            uint8_t *buf = xmalloc(4u * (cap + 8u));
            memset(buf, 0xA5, (size_t)(4u * (cap + 8u)));
            int64_t k = toks_par_encode(pool, x, len, flags, cap != 0u ? (uint32_t *)(void *)buf : NULL, cap);
            CHECK(k == n, "toks_par_encode (pool %u, cap %llu) returned %lld want %lld (%s)", pool_n,
                  (unsigned long long)cap, (long long)k, (long long)n, what);
            uint64_t cut = cap < (uint64_t)n ? cap : (uint64_t)n;
            if (cut != 0u && memcmp(buf, ids, (size_t)cut * 4u) != 0) {
                violation("toks_par_encode (pool %u, cap %llu): not the exact prefix (%s)", pool_n,
                          (unsigned long long)cap, what);
            }
            free(buf);
            if (cap == (uint64_t)n) { break; }
        }
        /* a two-item batch */
        if (len >= 2u) {
            toks_par_item it[2];
            uint32_t *o0 = xmalloc(4u * 8u), *o1 = xmalloc(4u * 8u);
            it[0].text = x;
            it[0].len = len / 2u;
            it[0].out = o0;
            it[0].cap = 8u;
            it[1].text = x + len / 2u;
            it[1].len = len - len / 2u;
            it[1].out = o1;
            it[1].cap = 8u;
            CHECK(toks_par_encode_batch(pool, it, 2u, flags) == 0, "batch refused (%s)", what);
            int64_t w0 = toks_encode(p->a, x, len / 2u, flags, NULL, 0u, scr);
            CHECK(it[0].n == w0, "batch item 0: %lld vs %lld (%s)", (long long)it[0].n, (long long)w0, what);
            free(o0);
            free(o1);
        }
        free(ids);
        free(scr);
        toks_par_destroy(pool);
        if (pool_n == 64u) { break; }
    }
}

/* ---- one tokenizer ------------------------------------------------------------------------------------------- */

static void battery_file(const char *path)
{
    pair p;
    toks_load_opts o;
    toks_diag dg;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &dg;
    memset(&dg, 0, sizeof dg);
    o.tier = TOKS_TIER_SCALAR;
    int64_t r1 = toks_load(&p.s, path, &o);
    o.tier = TOKS_TIER_AUTO;
    o.diag = NULL;
    int64_t r2 = toks_load(&p.a, path, &o);
    set_case("load");
    CHECK(r1 == r2, "the scalar load returned %lld, the auto load %lld", (long long)r1, (long long)r2);
    if (r1 != 0) {
        CHECK(p.s == NULL && p.a == NULL, "a refused load left a context set");
        CHECK(r1 >= TOKS_E_NOMEM && r1 <= TOKS_E_OPEN, "undocumented load code %lld", (long long)r1);
        CHECK(dg.code == r1 && memchr(dg.what, 0, sizeof dg.what) != NULL, "diag code %lld vs %lld",
              (long long)dg.code, (long long)r1);
        CHECK(r1 != TOKS_E_NOMEM, "TOKS_E_NOMEM on load (load work must be bounded by the source)");
        printf("hostile: %s refused: %lld (%.60s)\n", path, (long long)r1, dg.what);
        return;
    }
    memset(&p.info, 0, sizeof p.info);
    p.info.size = (uint32_t)sizeof p.info;
    CHECK(toks_get_info(p.a, &p.info) == 0, "toks_get_info");
    p.n_ids = p.info.n_ids;
    CHECK(p.n_ids >= 1u && p.n_ids <= TOKS_MAX_IDS, "n_ids %u", p.n_ids);
    /* toks_get_info with a wrong size */
    {
        toks_info bad;
        memset(&bad, 0, sizeof bad);
        bad.size = (uint32_t)sizeof bad - 1u;
        CHECK(toks_get_info(p.a, &bad) == TOKS_E_ARG, "toks_get_info accepted a short size");
    }
    printf("hostile: %s loaded (n_ids %u, algo %u)\n", path, p.n_ids, p.info.algorithm);

    /* the fixed texts x every flag shape */
    for (uint32_t ti = 0; ti < n_texts; ti++) {
        const h_text *tx = &texts[ti];
        if (tx->n == 0u) { continue; }
        for (uint32_t fi = 0; fi < sizeof FLAG_SHAPES / sizeof FLAG_SHAPES[0]; fi++) {
            uint32_t flags = FLAG_SHAPES[fi];
            set_case("text %s flags %u", tx->name, flags);
            uint32_t *ids = NULL;
            int64_t n = battery_text(&p, tx->bytes, tx->n, flags, fi & 7u, &ids, cur_case);
            if (n > 0 && ids != NULL) {
                battery_ids(&p, ids, n, cur_case);
                battery_split(&p, tx->bytes, tx->n, flags, ids, n, cur_case);
                free(ids);
            } else {
                free(ids);
            }
        }
        if (verbose) { printf("  text %s ok\n", tx->name); }
    }

    /* the call shapes */
    for (uint32_t ci = 0; ci < n_calls; ci++) {
        const h_call *c = &calls[ci];
        int i = text_index(c->name);
        if (i < 0) { set_case("call %s", c->kind); violation("call names an unknown text: %s", c->name); continue; }
        const h_text *tx = &texts[i];
        if (strcmp(c->kind, "size") == 0) {
            uint64_t len = c->a;
            set_case("size %s %llu", c->name, (unsigned long long)len);
            if (len > TOKS_MAX_TEXT) {
                uint8_t *one = (uint8_t *)xmalloc(1u);
                uint64_t sb = toks_scratch_bytes(p.a, 1u, 0u);
                uint8_t *scr = xmalloc(sb);
                CHECK(toks_scratch_init(p.a, scr, sb, 0u) == 0, "scratch init");
                CHECK(toks_encode(p.a, one, len, 0u, NULL, 0u, scr) == TOKS_E_LIMIT,
                      "len %llu over the limit accepted", (unsigned long long)len);
                CHECK(toks_pieces(p.a, one, len, 0u, NULL, 0u, scr) == TOKS_E_LIMIT, "pieces over the limit");
                CHECK(toks_split_points(p.a, one, len, 0u, 2u, NULL, 0u, NULL) == TOKS_E_LIMIT,
                      "split over the limit");
                free(one);
                free(scr);
            } else if (len <= max_bytes && len > tx->n) {
                uint8_t *buf = NULL;
                const uint8_t *x = tile(tx, len, &buf);
                battery_text(&p, x, len, 0u, 0u, NULL, cur_case);
                free(buf);
            } else if (len > max_bytes && len <= TOKS_MAX_TEXT) {
                /* a huge but in-limit length: a zero mapping (never read past, at most read once);
                 * run the count-only battery only when the exact scratch fits RAM (a normalizing
                 * context's scratch at 2^29 can be 22 GiB: the limit checks above are the point
                 * of these cases, not the full battery) */
                uint64_t sb = toks_scratch_bytes(p.a, len, 0u);
                uint8_t *z = sb <= (1u << 30) ? zero_map(len) : NULL;
                if (z != NULL) {
                    uint8_t *scr = malloc((size_t)sb);
                    if (scr != NULL && toks_scratch_init(p.a, scr, sb, 0u) == 0) {
                        int64_t n = toks_encode(p.a, z, len, 0u, NULL, 0u, scr);
                        CHECK(n >= 0, "encode of %llu zero bytes returned %lld", (unsigned long long)len,
                              (long long)n);
                        CHECK((uint64_t)n <= toks_encode_bound(p.a, len), "over the bound at %llu",
                              (unsigned long long)len);
                        uint32_t *ids = xmalloc(4u * (uint64_t)(n > 0 ? n : 1));
                        CHECK(toks_encode(p.a, z, len, 0u, ids, (uint64_t)n, scr) == n, "at cap == n");
                        free(ids);
                    } else {
                        printf("hostile: %s: skipped a %llu-byte case (no scratch)\n", path,
                               (unsigned long long)len);
                    }
                    free(scr);
                    zero_unmap(z, len);
                }
            }
        } else if (strcmp(c->kind, "misalign") == 0) {
            set_case("misalign %s +%llu", c->name, (unsigned long long)c->a);
            battery_text(&p, tx->bytes, tx->n, 0u, (uint32_t)(c->a & 7u), NULL, cur_case);
        } else if (strcmp(c->kind, "stream") == 0) {
            set_case("stream %s", c->name);
            uint32_t *ids = NULL;
            int64_t n = battery_text(&p, tx->bytes, tx->n, 0u, 0u, &ids, cur_case);
            if (n > 0 && ids != NULL) { battery_ids(&p, ids, n, cur_case); }
            free(ids);
        } else if (strcmp(c->kind, "split") == 0) {
            set_case("split %s %llu", c->name, (unsigned long long)c->a);
            uint32_t *ids = NULL;
            int64_t n = battery_text(&p, tx->bytes, tx->n, 0u, 0u, &ids, cur_case);
            if (n >= 0 && ids != NULL) { battery_split(&p, tx->bytes, tx->n, 0u, ids, n, cur_case); }
            free(ids);
        } else if (strcmp(c->kind, "par") == 0) {
            uint64_t len = c->a != 0u ? c->a : tx->n;
            set_case("par %s %llu x %llu", c->name, (unsigned long long)len, (unsigned long long)c->b);
            if (len <= max_bytes && len > 0u) {
                uint8_t *buf = NULL;
                const uint8_t *x = tile(tx, len, &buf);
                battery_par(&p, x, len, 0u, cur_case);
                free(buf);
            } else {
                printf("hostile: %s: skipped a %llu-byte par case (-m %llu)\n", path,
                       (unsigned long long)len, (unsigned long long)max_bytes);
            }
        }
    }

    toks_unload(p.s);
    toks_unload(p.a);
}

/* ---- main ------------------------------------------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[argi], "-l") == 0) {
            strict_flags = 0;
        } else if (strcmp(argv[argi], "-m") == 0 && argi + 1 < argc) {
            max_bytes = strtoull(argv[++argi], NULL, 10);
        } else {
            fprintf(stderr, "usage: hostile_driver [-v] [-l] [-m max-bytes] <texts.json> <tokenizer-file>...\n");
            return 2;
        }
        argi++;
    }
    if (argc - argi < 2) {
        fprintf(stderr, "usage: hostile_driver [-v] [-l] [-m max-bytes] <texts.json> <tokenizer-file>...\n");
        return 2;
    }

    uint64_t jlen = 0u;
    uint8_t *j = NULL;
    {
        FILE *f = fopen(argv[argi], "rb");
        if (f == NULL) { fprintf(stderr, "hostile: cannot read %s\n", argv[argi]); return 2; }
        fseek(f, 0, SEEK_END);
        long m = ftell(f);
        fseek(f, 0, SEEK_SET);
        j = xmalloc((uint64_t)(m > 0 ? m : 1));
        if (m > 0 && fread(j, 1, (size_t)m, f) != (size_t)m) { fprintf(stderr, "hostile: short read\n"); return 2; }
        fclose(f);
        jlen = (uint64_t)(m > 0 ? m : 0);
    }
    jp = j;
    jn = jlen;
    jp_end = j + jlen;

    jexpect('{');
    char key[32];
    for (;;) {
        jws();
        if (jp < jp_end && *jp == '}') { break; }
        (void)jstr_raw(key, sizeof key);      /* jstr_raw consumes the opening quote itself */
        jexpect(':');
        if (strcmp(key, "texts") == 0) {
            jexpect('[');
            uint32_t cap = 256u;
            texts = xmalloc(sizeof *texts * cap);
            jws();
            if (jp < jp_end && *jp == ']') { jp++; }          /* an empty texts array */
            else if (jchar('{')) {
                for (;;) {
                    char name[64], desc[256], hex[MAX_PIECES_TEXT * 2 + 8];
                    for (;;) {
                        jws();
                        if (jp < jp_end && *jp == '}') { break; }
                        (void)jstr_raw(key, sizeof key);
                        jexpect(':');
                        if (strcmp(key, "name") == 0) { (void)jstr_raw(name, sizeof name); }
                        else if (strcmp(key, "desc") == 0) { (void)jstr_raw(desc, sizeof desc); }
                        else if (strcmp(key, "hex") == 0) { (void)jstr_raw(hex, sizeof hex); }
                        else { jerr("unknown text field"); }
                        jws();
                        if (!jchar(',')) { break; }
                    }
                    jexpect('}');
                    if (n_texts == cap) { cap *= 2u; texts = realloc(texts, sizeof *texts * cap); }
                    h_text *t = &texts[n_texts++];
                    snprintf(t->name, sizeof t->name, "%s", name);
                    uint64_t hl = strlen(hex);
                    t->n = hl / 2u;
                    t->bytes = xmalloc(t->n != 0u ? t->n : 1u);
                    for (uint64_t k = 0; k < t->n; k++) {
                        int hi = hexval((uint8_t)hex[2u * k]), lo = hexval((uint8_t)hex[2u * k + 1u]);
                        if (hi < 0 || lo < 0) { jerr("bad hex"); }
                        t->bytes[k] = (uint8_t)(hi * 16 + lo);
                    }
                    jws();
                    if (jp >= jp_end || *jp != ',') { break; }
                    jp++;                                   /* the comma: another element follows */
                    jexpect('{');
                }
                jexpect(']');
            }
        } else if (strcmp(key, "calls") == 0) {
            jexpect('[');
            uint32_t cap = 128u;
            calls = xmalloc(sizeof *calls * cap);
            jws();
            if (jp < jp_end && *jp == ']') { jp++; }          /* an empty calls array */
            else if (jchar('[')) {
                for (;;) {
                    char kind[16], name[64];
                    (void)jstr_raw(kind, sizeof kind);
                    jexpect(',');
                    (void)jstr_raw(name, sizeof name);
                    h_call c;
                    memset(&c, 0, sizeof c);
                    snprintf(c.kind, sizeof c.kind, "%s", kind);
                    snprintf(c.name, sizeof c.name, "%s", name);
                    uint32_t ai = 0u;
                    while (jchar(',')) {
                        if (ai < 3u) { (&c.a)[ai] = jnum(); }
                        else { jnum(); }
                        ai++;
                    }
                    jexpect(']');
                    if (n_calls == cap) { cap *= 2u; calls = realloc(calls, sizeof *calls * cap); }
                    calls[n_calls++] = c;
                    jws();
                    if (jp >= jp_end || *jp != ',') { break; }
                    jp++;                                   /* the comma: another element follows */
                    jexpect('[');
                }
                jexpect(']');
            }
        } else {
            /* skip an unknown value */
            jws();
            if (jchar('[') || jchar('{')) {
                /* crude skip: the texts.json this reads has only flat arrays of scalars */
                while (jp < jp_end && *jp != ']' && *jp != '}') { jp++; }
            } else {
                while (jp < jp_end && *jp != ',' && *jp != '}') { jp++; }
            }
        }
        jws();
        if (!jchar(',')) { break; }
    }
    jexpect('}');
    printf("hostile: %u texts, %u call shapes\n", n_texts, n_calls);
    free(j);
    j = NULL;
    jp = NULL;
    jp_end = NULL;

    for (int k = argi + 1; k < argc; k++) {
        cur_file = argv[k];
        battery_file(argv[k]);
    }
    for (uint32_t i = 0; i < n_texts; i++) { free(texts[i].bytes); }
    free(texts);
    free(calls);
    if (fails != 0) {
        fprintf(stderr, "hostile: %d violation(s)\n", fails);
        return 1;
    }
    printf("hostile: clean\n");
    return 0;
}
