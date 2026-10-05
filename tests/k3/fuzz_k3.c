/*
 * tests/k3/fuzz_k3.c: differential fuzz of a K3 tier against its c twin (docs/kernels.md §1-3, SPEC
 * §14.3). Salvaged from an unmerged avx-512 bench (its fuzz and stats modes), made tier-generic.
 * Not part of `make test`; built by hand on a host of the tier's isa, against the library as shipped:
 *
 *   make -j lib
 *   clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core \
 *     -Isrc/platform -Itests/common -DK3_SCAN=toks_k3_scan_cl100k_avx2 -o build/fuzz_k3_avx2 \
 *     tests/k3/fuzz_k3.c tests/common/guard.c tests/common/abicheck_x86_64.S build/linux-x86_64/libtoks.a
 *
 * With -DK3_O200K the template is o200k (docs/templates/o200k.md; the twin toks_k3_scan_o200k_c): its 16
 * parameter combinations (CONTR_CI, DIGITS_1, HAN, NO_SLASH), plain or Han class tables (the Han ones under
 * TOKS_TP_HAN), mutated the same way.
 *
 *   fuzz_k3 fuzz ITERS SEED [FILE]   random TOKS_TP_* (all 96 combinations); class tables std, marks folded,
 *                                    or mutated within the contract (non-ascii classes only: any base 0-4,
 *                                    any flag, FOLD_S included; CJK / hangul blocks made non-uniform or
 *                                    recoloured), toks_tables.flags as the compiler sets them
 *                                    (toks_compile_cls_flags; TOKS_TF_CJK_L withheld now and then); random
 *                                    texts (pool fragments, runs, long runs of one class's members, random
 *                                    bytes, words, slices of FILE) up to 64 KiB; the text flush against a
 *                                    guard page at its end or its start, or plain at a random alignment; a
 *                                    random start piece and cap, sometimes resumed to the end
 *   fuzz_k3 geo SEED                 SPEC §14.3: every length 0..255 x 8 contents, the text's end flush
 *                                    against a guard page, its start flush, and plain at every alignment
 *                                    0..63; every start piece, caps 1 / random / big
 *   fuzz_k3 stats FILE...            piece-length profile of each FILE (cl100k, the c twin)
 *
 * Every call: ends[] sits flush against a no-access page at exactly cap entries (a store past ends[cap - 1]
 * faults; tiers may leave slack inside [n, cap), kernels.md §1), the 8 entries before ends[0] are checked
 * unchanged, a's input fields are checked unchanged, n / a->n / a->pos / the ends against the c twin, and
 * every 8th call runs through the abi checker (callee-saved registers kept). Exit 1 on any failure; with
 * FUZZ_K3_DUMP=<path> a failing text over 400 bytes is written there whole.
 */
#include "../../src/core/classes.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"
#include "guard.h"

#ifndef K3_SCAN
#error "build with -DK3_SCAN=toks_k3_scan_<tmpl>_<tier>"
#endif
uint64_t K3_SCAN(const toks_tables *t, toks_k3_args *a);
static int abi_on;                                     /* this call's parts go through the abi checker */
static uint64_t abi_rep;

/* the tier's part (K3_SCAN), which kernels.h's toks_k3_tier runs with the c twin's stretches */
static uint64_t part(const toks_tables *t, toks_k3_args *a)
{
    uint64_t rep = 0, n;
    if (!abi_on) { return K3_SCAN(t, a); }
    n = toks_abicheck_call((const void *)K3_SCAN, (uint64_t)(uintptr_t)t, (uint64_t)(uintptr_t)a, &rep);
    abi_rep |= rep;
    return n;
}
#if defined(K3_DSV3)                                   /* dsv3 (docs/templates/dsv3.md): its own tables only */
#define K3_TWIN toks_k3_scan_dsv3_c
#define K3_TMPL TOKS_TMPL_DSV3
#define K3_CLASSES0 TOKS_CLASSES_DSV3
#define K3_CLASSES1 TOKS_CLASSES_DSV3
#elif defined(K3_O200K)
#define K3_TWIN toks_k3_scan_o200k_c
#define K3_TMPL TOKS_TMPL_O200K
#define K3_CLASSES1 TOKS_CLASSES_HAN                   /* CTAB[1]: kimi's tables */
#else
#define K3_TWIN toks_k3_scan_cl100k_c
#define K3_TMPL TOKS_TMPL_CL100K
#define K3_CLASSES1 TOKS_CLASSES_MARKS_ARE_LETTERS     /* CTAB[1]: marks folded */
#endif
#ifndef K3_CLASSES0
#define K3_CLASSES0 0u
#endif
#define STR2(x) #x
#define STR(x) STR2(x)

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------------------------ tables */

static uint8_t cls_buf[2][128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CTAB[2];
static uint8_t m_ascii[128];
static uint16_t m_st1[0x1100];
static uint8_t *m_st2;
static uint64_t m_nb;

static uint64_t rs = 1;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint64_t rn(uint64_t n) { return n ? rnd() % n : 0; }

static void build_classes(void)
{
    static const uint32_t flags[2] = { K3_CLASSES0, K3_CLASSES1 };
    for (int i = 0; i < 2; i++) {
        if (toks_classes_build(flags[i], cls_buf[i], sizeof cls_buf[i], &CTAB[i]) <= 0) {
            fprintf(stderr, "classes build failed\n");
            exit(2);
        }
    }
    uint64_t nb = CTAB[0].n_blocks > CTAB[1].n_blocks ? CTAB[0].n_blocks : CTAB[1].n_blocks;
    m_st2 = malloc((size_t)(nb + 8) * 256);
    if (!m_st2) { exit(2); }
}

/* code points the generators emit (mutation targets) */
static const uint32_t HOT[] = {
    0xE9, 0xDF, 0x430, 0x3B1, 0x4E2D, 0x6587, 0xD55C, 0xAC00, 0x3042, 0x30A2, 0x3001, 0x3002, 0xFF0C, 0x3000,
    0xA0, 0x85, 0x2028, 0x2003, 0x301, 0x660, 0x661, 0xFF10, 0xFF11, 0xB2, 0x2460, 0x1D7CE, 0x1F600, 0x17F,
    0x212A, 0xFFFD, 0xE000, 0xD7FF, 0x9FFF, 0x4E00, 0xA3FF, 0xAC01, 0xD6FF, 0xD7A3, 0x800, 0xFFFF, 0x2019,
    0x201C, 0xAB, 0xBB, 0x966, 0x915, 0x94D, 0x10FFFF, 0x1C5, 0x2B0, 0x2E80, 0x3005, 0x3007, 0x41, 0x61,
    0xAD, 0x200B, 0x200D, 0x3040, 0x3099, 0x309B, 0x30A0, 0x30FB, 0x9FA5, 0x9FA6,
};
#define NHOT (sizeof HOT / sizeof HOT[0])

/* t for params p; kind 0 std, 1 marks folded, 2 mutated within the contract (non-ascii classes only); the
 * flags as the compiler sets them (TOKS_TF_CJK_L when the CJK / hangul ranges are base L without FOLD_S) */
static void tables_cls(toks_tables *t, uint32_t p, int kind);
static void tables(toks_tables *t, uint32_t p, int kind)
{
#if defined(K3_O200K) && !defined(K3_DSV3)
    if (kind < 2) { kind = (p & TOKS_TP_HAN) != 0u; }  /* the Han tables go with TOKS_TP_HAN */
#endif
    tables_cls(t, p, kind);
    int64_t cf = toks_compile_cls_flags(t);
    if (cf < 0) { fprintf(stderr, "harness: class tables break kernels.md §2\n"); exit(2); }
    t->flags = rn(4) ? (uint32_t)cf : 0u;               /* the flag permits, never obliges: sometimes withheld */
}
static void tables_cls(toks_tables *t, uint32_t p, int kind)
{
    memset(t, 0, sizeof *t);
    t->magic = TOKS_TABLES_MAGIC;
    t->version = TOKS_TABLES_VERSION;
    t->tmpl = K3_TMPL;
    t->tmpl_params = p;
    if (kind < 2) {
        t->cls_ascii = CTAB[kind].ascii;
        t->cls_stage1 = CTAB[kind].stage1;
        t->cls_stage2 = CTAB[kind].stage2;
        t->cls_nblocks = CTAB[kind].n_blocks;
        return;
    }
#if defined(K3_O200K) && !defined(K3_DSV3)
    const toks_class_tables *b = &CTAB[(p & TOKS_TP_HAN) != 0u];
#else
    const toks_class_tables *b = &CTAB[rnd() & 1];
#endif
    memcpy(m_ascii, b->ascii, 128);
    memcpy(m_st1, b->stage1, sizeof m_st1);
    memcpy(m_st2, b->stage2, (size_t)b->n_blocks * 256);
    m_nb = b->n_blocks;
    uint32_t nmut = 1 + (uint32_t)rn(6);
    for (uint32_t i = 0; i < nmut && m_nb < b->n_blocks + 8; i++) {     /* bound: 6 mutations */
#ifdef K3_DSV3
        uint8_t cls = (uint8_t)(rn(6) | (rn(2) << 7));                    /* a dsv3 kind: base P..X | CJK */
#else
        uint8_t cls = (uint8_t)rn(5);                                     /* base P L N WS NL */
        if (rn(3) == 0) { cls |= (uint8_t)(rn(16) << 3); }                /* UPPER LOWER MARK FOLD_S */
#endif
        uint32_t what = (uint32_t)rn(3);
        if (what == 0) {                                /* a hot code point (its block copied first) */
            uint32_t cp = HOT[rn(NHOT)];
            memcpy(m_st2 + m_nb * 256, m_st2 + (size_t)m_st1[cp >> 8] * 256, 256);
            m_st1[cp >> 8] = (uint16_t)m_nb;
            m_st2[m_nb * 256 + (cp & 0xFF)] = cls;
            m_nb++;
        } else if (what == 1) {                         /* a CJK ideograph / hangul block made non-uniform */
            static const uint32_t UNI[] = { 0x4E00, 0x4E2D, 0x6587, 0x9FFF, 0xA000, 0xA3FF, 0xAC00, 0xAC01, 0xD55C,
                                            0xD6FF };
            uint32_t cp = UNI[rn(sizeof UNI / sizeof UNI[0])];
            uint32_t hi = rn(4) ? cp >> 8 : rn(2) ? 0x4E + (uint32_t)rn(0xA4 - 0x4E) : 0xAC + (uint32_t)rn(0xD7 - 0xAC);
            memcpy(m_st2 + m_nb * 256, m_st2 + (size_t)m_st1[hi] * 256, 256);
            m_st1[hi] = (uint16_t)m_nb;
            m_st2[m_nb * 256 + (hi == cp >> 8 ? (cp & 0xFF) : rn(256))] = cls;
            m_nb++;
        } else {                                        /* the block U+4E00 maps to, recoloured */
            memset(m_st2 + (size_t)m_st1[0x4E] * 256, cls, 256);
        }
    }
    t->cls_ascii = m_ascii;
    t->cls_stage1 = m_st1;
    t->cls_stage2 = m_st2;
    t->cls_nblocks = m_nb;
}

static uint32_t rand_params(void)
{
#if defined(K3_DSV3)
    return (uint32_t)rnd();                            /* dsv3 has no parameters: every bit ignored */
#elif defined(K3_O200K)
    return (rn(4) ? TOKS_TP_CONTR_CI : 0u) | (rn(3) ? 0u : TOKS_TP_DIGITS_1) | (rn(3) ? 0u : TOKS_TP_HAN) |
           (rn(3) ? 0u : TOKS_TP_NO_SLASH);
#else
    uint32_t p = (uint32_t)rn(3) | (rn(2) ? TOKS_TP_LPREFIX_ANY : 0u) | (uint32_t)(rn(4) << 3) |
                 (rn(2) ? TOKS_TP_PUNCT_NL : 0u) | (rn(2) ? TOKS_TP_WS_NL : 0u) | (rn(3) ? 0u : TOKS_TP_NL_CUT);
    uint32_t d = p & TOKS_TP_DIGITS_MASK;           /* A8 with \p{N} (digits-gpt2) or \p{N}{1,3} (MiniCPM5) */
    return (d == TOKS_TP_DIGITS_1 || d == TOKS_TP_DIGITS_1_3) && rn(2) ? p | TOKS_TP_DIGIT_CUT : p;
#endif
}

/* ------------------------------------------------------------------------------------------- texts */

static const char *POOL[] = {
    "a", "b", "Z", "x", "s", "t", "m", "d", "r", "e", "v", "l", "S", "T", "M", "D", "R", "E", "V", "L",
    "'", "'", "'", " ", " ", " ", "  ", "\t", "\n", "\r", "\r\n", "\n\r", "\v", "\f",
    "0", "1", "7", "12", "345", "6789", "!", "?", ".", ",", "-", "=", "\"", "(", ")", "@", "#", "_", "/",
    "\xC3\xA9", "\xC3\x9F", "\xE4\xB8\xAD", "\xE6\x96\x87", "\xED\x95\x9C", "\xEA\xB0\x80", "\xCC\x81",
    "\xC2\xA0", "\xF0\x9F\x98\x80", "\xC5\xBF", "\xE2\x84\xAA", "\xD9\xA1", "\xD9\xA0", "\xEF\xBC\x91",
    "\xEF\xBC\x90", "\xC2\xB2", "\xE2\x91\xA0", "\xE3\x80\x80", "\xE2\x80\x83", "\xC2\x85", "\xE2\x80\xA8",
    "\xE2\x80\x99", "\xE2\x80\x9C", "\xC2\xAB", "\xC2\xBB", "\xD0\xB0", "\xCE\xB1", "\xE3\x81\x82",
    "\xE3\x82\xA2", "\xE3\x80\x81", "\xE3\x80\x82", "\xEF\xBC\x8C", "\xF0\x9D\x9F\x8E", "\xE0\xA5\xA6",
    "\xE0\xA4\x95\xE0\xA5\x8D", "\xEF\xBF\xBD", "\xEE\x80\x80", "\xED\x9F\xBF", "\xE9\xBF\xBF", "\xE4\xB8\x80",
    "\xEA\x8F\xBF", "\xEA\x80\x80", "\xED\x9B\xBF", "\xED\x9E\xA3", "\xE0\xA0\x80", "\xEF\xBF\xBF", "\xF4\x8F\xBF\xBF",
    "\x80", "\xBF", "\xFF", "\xFE", "\xC3", "\xE4\xB8", "\xE0\x80", "\xE0\x80\x80", "\xE0\x9F\xBF",
    "\xED\xA0\x80", "\xED\xBF\xBF", "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xC0\xAF", "\xC1\xBF",
    "\xF0\x80\x80\x80", "\xF0\x8F\xBF\xBF", "\xF0\x9F\x98", "\xF0\x9F", "\xE4", "\xF0",
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "'S", "'RE", "'Ll", "'lL", "'Ve", "'D", "'\xC5\xBF",
    ("'\xC5\xBF" "x"), "camelCase", "HTMLParser", "ABCdEF", "it'sx", "they'r", "DON'T", "'llama", "!\xCC\x81",
    " \xCC\x81's", "\xCC\x81\xCC\x81", "!!\xCC\x81", "!\n/", "!\n//\n/x", "a/b", "\xC7\x85", "a\xC7\x85" "b",
    "\xCA\xB0" "A", "\xE4\xB8\xAD" "AB!", "A\xE4\xB8\xAD" "B", "\xE4\xB8\xAD" "A's", "\xE2\xBA\x80", "\xE3\x80\x87",
    "\xE3\x80\x85", "A", "B", "Q", "it's\xE4\xB8\xAD" "Ab", "\xE0\xA4\x95\xE0\xA4\xBF", "x\xCC\x81Y",
    "!\n/!\xCC\x81\r\n/", "!\n//\xCC\x81\n", "a!\xCC\x81\n/x",
    "\x01", "\x1F", "\x7F", "\x08", "\xC2\xAD", "\xE2\x80\x8B", "\xE2\x80\x8D", "\xEF\xBB\xBF", "\xE3\x82\x9B",
    "\xE3\x83\xBB", "\xE3\x82\xA0", "\xE3\x81\x80", "\xE3\x82\x99", "\xE9\xBE\xA5", "\xE9\xBE\xA6", "(ab\xC3\xA9",
    "(a\xCC\x81", "x(Ab\xC3\x9F", " \xE4\xB8\xAD", "  \xE4\xB8\xAD", "  1", "\t\t7", "\xE3\x82\xA2\xE3\x82\x99x",
};
#define NPOOL (sizeof POOL / sizeof POOL[0])
static const char *WORDS[] = {
    "the", " quick", " brown", " fox", "'s", " don't", "  ", "\n", " 123", "4567", "!!", " ...", "\t",
    "\r\n", " \n ", "x", " I'm", " we'll", "    ", "\n\n", "\xE2\x80\x99s", " caf\xC3\xA9",
    "\xE4\xB8\xAD\xE6\x96\x87", " Hello", ",", ".", " =", " ==", "\xE3\x81\x82\xE3\x81\x84",
};
#define NWORDS (sizeof WORDS / sizeof WORDS[0])

/* class runs (members of one base class, ascii and not, 1-4 bytes; the L pool holds U+017F, which cuts) */
static const char *RUNS[4][16] = {
    { "a", "Z", "q", "\xC3\xA9", "\xC3\x9F", "\xD0\xB0", "\xCE\xB1", "\xE4\xB8\xAD", "\xE6\x96\x87", "\xED\x95\x9C",
      "\xEA\xB0\x80", "\xE3\x81\x82", "\xE3\x82\xA2", "\xCC\x81", "\xD0\x96", "\xF0\x9D\x90\x80" },
    { "0", "1", "9", "5", "\xD9\xA0", "\xD9\xA1", "\xEF\xBC\x90", "\xC2\xB2", "\xE2\x91\xA0", "\xF0\x9D\x9F\x8E",
      "\xE0\xA5\xA6", "7", "3", "\xEF\xBC\x91", "\xC2\xB9", "2" },
    { "!", "-", "=", ".", "\xE2\x80\xA6", "\xE2\x80\x94", "\xE3\x80\x82", "\xE3\x80\x81", "\xE2\x94\x80", "\xFF", "\x80",
      "\xF0\x9F\x98\x80", "\xE2\x82\xAC", "#", "*", "\xC2\xAB" },
    { " ", " ", "\t", "\n", "\r", "\r\n", "\v", "\f", "\xC2\xA0", "\xE3\x80\x80", "\xE2\x80\x83", "\xC2\x85",
      "\xE2\x80\xA8", " ", "\n", "  " },
};

static uint8_t *REAL;
static uint64_t REAL_LEN;

static uint64_t put(uint8_t *b, uint64_t len, uint64_t max, const char *s)
{
    size_t n = strlen(s);
    if (len + n > max) { return len; }
    memcpy(b + len, s, n);
    return len + n;
}

/* a text of up to max bytes; mode < 0 picks one */
static uint64_t gen_text(uint8_t *buf, uint64_t max, int mode)
{
    uint64_t len = 0, target = rn(4) == 0 ? rn(max + 1) : rn(330);
    if (target > max) { target = max; }
    if (mode < 0) { mode = (int)rn(REAL ? 7 : 6); }
    while (len < target) {                             /* bound: target bytes (each round adds >= 1 or stops) */
        uint64_t before = len;
        if (mode == 0) {                               /* runs of one fragment, any length */
            const char *s = POOL[rn(NPOOL)];
            uint64_t r = 1 + rn(rn(4) == 0 ? 200 : 40);
            for (uint64_t k = 0; k < r; k++) { len = put(buf, len, target, s); }   /* bound: r */
            if (rn(2)) { len = put(buf, len, target, POOL[rn(NPOOL)]); }
        } else if (mode == 1) {                        /* random bytes, half of them ascii */
            uint64_t x = rnd();
            buf[len++] = (uint8_t)(x & 1 ? x >> 8 : (x >> 8) & 0x7F);
        } else if (mode == 2 || mode == 3) {           /* words */
            len = put(buf, len, target, WORDS[rn(NWORDS)]);
        } else if (mode == 4) {                        /* pool fragments */
            len = put(buf, len, target, POOL[rn(NPOOL)]);
        } else if (mode == 5) {                        /* a long class run, now and then another atom */
            uint64_t c = rn(4), r = 20 + rn(rn(3) ? 120 : 700);
            for (uint64_t k = 0; k < r; k++) {         /* bound: r atoms */
                len = put(buf, len, target, rn(24) ? RUNS[c][rn(16)] : POOL[rn(NPOOL)]);
            }
            if (rn(2)) { len = put(buf, len, target, POOL[rn(NPOOL)]); }
        } else {                                       /* a slice of FILE */
            uint64_t off = rn(REAL_LEN), n = target - len;
            if (off + n > REAL_LEN) { n = REAL_LEN - off; }
            memcpy(buf + len, REAL + off, (size_t)n);
            len += n;
        }
        if (len == before && mode != 1) { break; }
    }
    return len;
}

/* --------------------------------------------------------------------------------------------- check */

#define MAXT (1 << 16)
#define MAXCAP 8192
static uint32_t *ENDS_G;                               /* ENDS_G + MAXCAP is flush against a no-access page */
static uint32_t want[MAXT + 1];
static uint64_t calls, fails, abi_bad;
static int mutated;

static void report(const char *what, const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t start,
                   uint64_t cap, uint64_t n, uint64_t en, uint64_t first)
{
    if (fails < 8) {
        fprintf(stderr, "FAIL %s: params %02x tables %s len %" PRIu64 " start %" PRIu64 " cap %" PRIu64 " n %"
                PRIu64 " want %" PRIu64 " first difference #%" PRIu64 "\n", what, t->tmpl_params,
                t->cls_stage2 == CTAB[0].stage2 ? "std" : t->cls_stage2 == CTAB[1].stage2 ? "tables 1" : "mutated",
                len, start, cap, n, en, first);
        if (len <= 400) {
            fprintf(stderr, "  text:");
            for (uint64_t i = 0; i < len; i++) { fprintf(stderr, " %02x", text[i]); }
            fprintf(stderr, "\n");
        } else if (getenv("FUZZ_K3_DUMP") != NULL) {   /* a longer text whole, for a replay */
            FILE *f = fopen(getenv("FUZZ_K3_DUMP"), "wb");
            if (f != NULL) { fwrite(text, 1, (size_t)len, f); fclose(f); }
        }
    }
    fails++;
}

/* one call from want's start piece si with cap; nw = the twin's pieces of the whole text */
static int one(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t si, uint64_t nw, uint64_t cap)
{
    uint64_t start = si ? want[si - 1] : 0;
    if (cap == 0 || cap > MAXCAP) { fprintf(stderr, "harness: cap %" PRIu64 " outside 1..%d\n", cap, MAXCAP); exit(2); }
    uint32_t *ends = ENDS_G + MAXCAP - cap;            /* ends[cap] is the first no-access byte */
    uint32_t *pre = ends - 8 >= ENDS_G ? ends - 8 : ENDS_G;
    for (uint32_t *x = pre; x < ends; x++) { *x = 0xA5A5A5A5u; }
    toks_k3_args a, b;
    memset(&a, 0, sizeof a);
    a.text = text; a.len = len; a.pos = start; a.ends = ends; a.cap = cap; a.n = 0xDEADu; a.flags = 0;
    a.rsv = 0x5A5Au;
    b = a;
    uint64_t n, rep = 0;
    calls++;
    abi_on = calls % 8 == 0;                           /* through the abi checker: callee-saved kept */
    abi_rep = 0;
    n = toks_k3_tier(part, K3_TWIN, t, &a);
    rep = abi_rep;
    if (rep != 0) {
        fprintf(stderr, "ABI: callee-saved registers clobbered (report %#" PRIx64 ")\n", rep);
        abi_bad++;
    }
    uint64_t en = nw - si < cap ? nw - si : cap, first = 0;
    while (first < n && first < en && ends[first] == want[si + first]) { first++; }
    int pre_ok = 1;
    for (uint32_t *x = pre; x < ends; x++) { pre_ok &= *x == 0xA5A5A5A5u; }
    if (n != en || a.n != n || first != n || a.pos != (n ? (uint64_t)ends[n - 1] : start) || !pre_ok ||
        a.text != b.text || a.len != b.len || a.ends != b.ends || a.cap != b.cap || a.flags != b.flags ||
        a.rsv != b.rsv || rep != 0) {
        report(pre_ok ? "call" : "write before ends[0]", t, text, len, start, cap, n, en, first);
        return 0;
    }
    return 1;
}

static const uint64_t CAPS[] = { 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257 };
#define NCAPS (sizeof CAPS / sizeof CAPS[0])

/* the twin's pieces of the whole text into want; then calls from `starts` start pieces (all when < 0) */
static void check_text(const toks_tables *t, const uint8_t *text, uint64_t len, int starts)
{
    toks_k3_args a;
    memset(&a, 0, sizeof a);
    a.text = text; a.len = len; a.pos = 0; a.ends = want; a.cap = MAXT + 1;
    uint64_t nw = K3_TWIN(t, &a);
    uint64_t big = nw + 1 + rn(3);
    if (big > MAXCAP) { big = MAXCAP; }
    uint64_t ns = starts < 0 || nw + 1 <= (uint64_t)starts ? nw + 1 : (uint64_t)starts;
    for (uint64_t k = 0; k < ns; k++) {                /* bound: nw + 1 starts */
        uint64_t si = ns == nw + 1 ? k : (k == 0 ? 0 : rn(nw + 1));
        if (!one(t, text, len, si, nw, big)) { return; }
        if (si == nw) { continue; }                    /* pos = len: n = 0 at every cap */
        uint64_t cap = rn(3) ? CAPS[rn(NCAPS)] : 1 + rn(MAXCAP);
        if (!one(t, text, len, si, nw, cap)) { return; }
        if (rn(4) == 0) {                              /* resume to the end */
            for (uint64_t s = si; s < nw; s += cap) {  /* bound: nw / cap calls */
                if (!one(t, text, len, s, nw, cap)) { return; }
            }
        }
    }
}

/* the text placed: 0 end flush against a guard page, 1 start flush, 2 plain at alignment al (< 64) */
static void place_and_check(const toks_tables *t, const uint8_t *src, uint64_t len, int where, uint64_t al,
                            int starts)
{
    static uint8_t plain[MAXT + 128] __attribute__((aligned(64)));
    guard_buf g = { 0 };
    uint8_t *p = plain + al;
    if (where < 2) {
        p = guard_alloc(&g, (size_t)len, where == 0 ? GUARD_END : GUARD_START, 0);
        if (!p) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    }
    memcpy(p, src, (size_t)len);
    check_text(t, p, len, starts);
    if (where < 2) { guard_free(&g); }
}

static int fuzz(uint64_t iters)
{
    static uint8_t text[MAXT];
    for (uint64_t it = 0; it < iters; it++) {          /* bound: iters */
        toks_tables t;
        uint64_t r = rn(10);
        tables(&t, rand_params(), r < 4 ? 0 : r < 7 ? 1 : 2);
        mutated += r >= 7;
        uint64_t len = gen_text(text, rn(8) == 0 ? MAXT : 4096, -1);
        place_and_check(&t, text, len, (int)rn(3), rn(64), 2 + (int)rn(6));
    }
    return 0;
}

static int geo(void)
{
    enum { NC = 8, CL = 256 };
    static uint8_t content[NC][CL + 8];
    for (int c = 0; c < NC; c++) {                     /* 8 contents, each 256 bytes long */
        uint64_t l = 0;
        while (l < CL) {                               /* bound: CL bytes */
            uint64_t before = l;
            l += gen_text(content[c] + l, CL - l, c % 5);
            if (l == before) { content[c][l++] = (uint8_t)('a' + c); }
        }
    }
    for (uint64_t len = 0; len <= 255; len++) {        /* bound: 256 lengths */
        for (int c = 0; c < NC; c++) {                 /* bound: NC contents */
            for (uint64_t w = 0; w < 66; w++) {        /* end flush, start flush, alignments 0..63 */
                toks_tables t;
                tables(&t, rand_params(), (int)rn(3));
                place_and_check(&t, content[c], len, w < 2 ? (int)w : 2, w < 2 ? 0 : w - 2, w < 2 ? -1 : 3);
            }
        }
    }
    return 0;
}

/* --------------------------------------------------------------------------------------------- stats */

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read %s failed\n", path); exit(2); }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

/* piece lengths of FILE under cl100k (the c twin): bytes in pieces of 1-8, 9-16, 17-32, 33-63, 64+ */
static void stats(const char *path)
{
    toks_tables t;
    tables(&t, TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, 0);
    uint64_t len, pos = 0, hist[5] = { 0 }, np = 0;
    uint8_t *text = slurp(path, &len);
    while (pos < len) {                                /* bound: pieces */
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        a.text = text; a.len = len; a.pos = pos; a.ends = want; a.cap = MAXT;
        uint64_t n = K3_TWIN(&t, &a);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t l = want[i] - pos;
            hist[l <= 8 ? 0 : l <= 16 ? 1 : l <= 32 ? 2 : l < 64 ? 3 : 4] += l;
            pos = want[i];
        }
        np += n;
    }
    double d = len ? (double)len / 100.0 : 1.0;
    printf("%s: %" PRIu64 " bytes, %" PRIu64 " pieces; bytes in pieces of 1-8 %.1f%%, 9-16 %.1f%%, 17-32 %.1f%%, "
           "33-63 %.1f%%, 64+ %.1f%%\n", path, len, np, (double)hist[0] / d, (double)hist[1] / d,
           (double)hist[2] / d, (double)hist[3] / d, (double)hist[4] / d);
    free(text);
}

int main(int argc, char **argv)
{
    build_classes();
    if (argc >= 3 && strcmp(argv[1], "stats") == 0) {
        for (int i = 2; i < argc; i++) { stats(argv[i]); }
        return 0;
    }
    guard_buf eg = { 0 };
    uint8_t *e = guard_alloc(&eg, MAXCAP * 4, GUARD_END, 0);
    if (!e) { fprintf(stderr, "guard_alloc (ends) failed\n"); return 2; }
    ENDS_G = (uint32_t *)(void *)e;
    if (argc >= 2 && strcmp(argv[1], "fuzz") == 0) {
        uint64_t iters = argc > 2 ? strtoull(argv[2], NULL, 10) : 100000;
        rs = (argc > 3 ? strtoull(argv[3], NULL, 10) : 1) | 1;
        if (argc > 4) { REAL = slurp(argv[4], &REAL_LEN); }
        fuzz(iters);
        printf("fuzz_k3 %s fuzz: %" PRIu64 " cases (%d with mutated tables), %" PRIu64 " calls, %" PRIu64
               " failures, %" PRIu64 " abi\n", STR(K3_SCAN), iters, mutated, calls, fails, abi_bad);
    } else if (argc >= 3 && strcmp(argv[1], "geo") == 0) {
        rs = strtoull(argv[2], NULL, 10) | 1;
        geo();
        printf("fuzz_k3 %s geo: lengths 0..255 x 8 contents x (end flush, start flush, alignments 0..63), %"
               PRIu64 " calls, %" PRIu64 " failures, %" PRIu64 " abi\n", STR(K3_SCAN), calls, fails, abi_bad);
    } else {
        fprintf(stderr, "usage: %s fuzz [ITERS [SEED [FILE]]] | geo SEED | stats FILE...\n", argv[0]);
        return 2;
    }
    guard_free(&eg);
    return fails || abi_bad ? 1 : 0;
}
