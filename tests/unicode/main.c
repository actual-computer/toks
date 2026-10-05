/* toks: unicode class-table test driver (tests/unicode/).
 *
 * Reads ground truth from stdin (produced by probe.py, a fresh probe of hf
 * tokenizers 0.23.2's regex engine), builds the class tables for both
 * class_flags settings, and checks:
 *   - every scalar value U+0000..U+10FFFF, both class_flags settings
 *   - cp > 0x10FFFF -> P; surrogates -> P (flags 0)
 *   - the 25 \\s code points, CR/LF vs U+0085/U+2028/U+2029, U+017F
 *   - determinism: two builds are bit-for-bit identical
 *   - toks_class_of fast paths agree with the direct stage2 read
 *   - toks_classes_bytes / E_CAP / E_FLAGS / E_ARG contracts
 *
 * Standalone (not part of `make test`):
 *   clang -std=c17 -O2 -Wall -Wextra -Wconversion -Wsign-conversion -Werror \
 *     -I src -I . tests/unicode/main.c src/core/classes.c src/gen/ucd_flags.c \
 *     -o build/unicode_test
 *   uv run tests/unicode/probe.py | ./build/unicode_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/classes.h"
#include "gen/ucd_flags.h"

#define N_CP 0x110000

static unsigned char gt_base[2][N_CP];  /* expected base class, both flags */
static unsigned char gt_attr[2][N_CP];  /* expected attribute bits */

static int failures;

static void check(int cond, const char *fmt, unsigned long cp)
{
    if (!cond) {
        failures++;
        if (failures <= 20) {
            printf("FAIL cp=U+%04lX %s\n", cp, fmt);
        }
    }
}

int main(void)
{
    unsigned long v;
    uint8_t *buf0, *buf1, *buf2;
    uint64_t cap;
    toks_class_tables t0, t1, t2;
    unsigned long cp;
    int r;

    /* ---- read ground truth: lines "cp base0 base1 attr0 attr1" ---- */
    for (cp = 0; cp < N_CP; cp++) {
        unsigned long b0, b1, a0, a1;
        if (scanf("%lu %lu %lu %lu %lu", &v, &b0, &b1, &a0, &a1) != 5) {
            printf("FAIL truncated ground truth at cp=%lu\n", cp);
            return 1;
        }
        if (v != cp) {
            printf("FAIL ground truth out of order: %lu != %lu\n", v, cp);
            return 1;
        }
        gt_base[0][cp] = (unsigned char)b0;
        gt_base[1][cp] = (unsigned char)b1;
        gt_attr[0][cp] = (unsigned char)a0;
        gt_attr[1][cp] = (unsigned char)a1;
    }

    cap = toks_classes_bytes(0u);
    check(cap == 128u + 0x2200u + (uint64_t)0x1100u * 256u, "bytes formula", 0);
    check(toks_classes_bytes(1u) == cap, "bytes formula flags=1", 0);
    check(toks_classes_bytes(0xDEADBEu) == 0u, "bytes flags garbage", 0);

    buf0 = malloc((size_t)cap);
    buf1 = malloc((size_t)cap);
    buf2 = malloc((size_t)cap);
    if (!buf0 || !buf1 || !buf2) {
        printf("FAIL alloc\n");
        return 1;
    }

    /* ---- contracts ---- */
    check(toks_classes_build(0u, buf0, cap, 0) == TOKS_CLASSES_E_ARG,
          "E_ARG on NULL out", 0);
    check(toks_classes_build(0u, 0, cap, &t0) == TOKS_CLASSES_E_CAP,
          "E_CAP on NULL buf", 0);
    check(toks_classes_build(0x2u, buf0, cap, &t0) == TOKS_CLASSES_E_FLAGS,
          "E_FLAGS on unknown bit", 0);
    check(toks_classes_build(0u, buf0, cap - 1u, &t0) == TOKS_CLASSES_E_CAP,
          "E_CAP on short buf", 0);

    r = toks_classes_build(0u, buf0, cap, &t0);
    check(r > 0, "build flags=0", 0);
    r = toks_classes_build(1u, buf1, cap, &t1);
    check(r > 0, "build flags=1", 0);

    /* determinism: same flags, different buffer */
    check(toks_classes_build(1u, buf2, cap, &t2) > 0, "rebuild flags=1", 0);
    check(t2.n_blocks == t1.n_blocks, "n_blocks stable", 0);
    check(memcmp(buf1, buf2, (size_t)r) == 0, "bit-for-bit deterministic", 0);

    /* stage1[0] == 0: Latin-1 direct load */
    check(t0.stage1[0] == 0, "stage1[0] == 0", 0);
    check(t1.stage1[0] == 0, "stage1[0] == 0 (flags=1)", 0);

    /* ---- every scalar value, both flags ---- */
    for (cp = 0; cp < N_CP; cp++) {
        int fi;
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            continue; /* surrogates: flags 0 -> P, checked below via table */
        }
        for (fi = 0; fi < 2; fi++) {
            const toks_class_tables *t = fi ? &t1 : &t0;
            uint8_t got = toks_class_of(t, (uint32_t)cp);
            uint8_t want = (uint8_t)(gt_base[fi][cp] | gt_attr[fi][cp]);
            if (got != want) {
                char msg[64];
                snprintf(msg, sizeof msg, "class mismatch flags=%d got=0x%02X want=0x%02X",
                         fi, got, want);
                check(0, msg, cp);
            }
        }
    }

    /* surrogates: flags 0 -> class P, both settings */
    for (cp = 0xD800; cp <= 0xDFFF; cp++) {
        check(toks_class_of(&t0, (uint32_t)cp) == TOKS_CLS_P, "surrogate -> P", cp);
        check(toks_class_of(&t1, (uint32_t)cp) == TOKS_CLS_P, "surrogate -> P (f1)", cp);
        check(toks_ucd_flags((uint32_t)cp) == 0, "surrogate flags 0", cp);
    }

    /* ---- out of range: cp > 0x10FFFF ---- */
    {
        uint32_t oor[] = {0x110000u, 0x110001u, 0xFFFFFFFFu, 0x80000000u, 0x10FFFFu};
        size_t i;
        for (i = 0; i < sizeof oor / sizeof oor[0]; i++) {
            uint32_t c = oor[i];
            uint8_t want = c == 0x10FFFFu ? toks_class_of(&t0, c) : TOKS_CLS_P;
            check(toks_class_of(&t0, c) == want, "cp out of range -> P", c);
            check(toks_ucd_flags(c) == 0, "ucd_flags out of range -> 0", c);
        }
    }

    /* ---- targeted checks ---- */
    /* the 25 \s code points: WS (or NL for CR/LF), both settings */
    {
        unsigned long ws25[] = {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680,
                                 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
                                 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028,
                                 0x2029, 0x202F, 0x205F, 0x3000};
        size_t i;
        for (i = 0; i < sizeof ws25 / sizeof ws25[0]; i++) {
            unsigned long c = ws25[i];
            uint8_t base = (uint8_t)(toks_class_of(&t0, (uint32_t)c) & 7u);
            check(base == TOKS_CLS_NL || base == TOKS_CLS_WS, "\\s member is WS/NL", c);
        }
        check((toks_class_of(&t0, 0x0A) & 7u) == TOKS_CLS_NL, "LF is NL", 0xA);
        check((toks_class_of(&t0, 0x0D) & 7u) == TOKS_CLS_NL, "CR is NL", 0xD);
        check((toks_class_of(&t0, 0x85) & 7u) == TOKS_CLS_WS, "U+0085 is WS not NL", 0x85);
        check((toks_class_of(&t0, 0x2028) & 7u) == TOKS_CLS_WS, "U+2028 is WS not NL", 0x2028);
        check((toks_class_of(&t0, 0x2029) & 7u) == TOKS_CLS_WS, "U+2029 is WS not NL", 0x2029);
        check((toks_class_of(&t0, 0x180E) & 7u) == TOKS_CLS_P, "U+180E is P (not \\s)", 0x180E);
        check((toks_class_of(&t0, 0x200B) & 7u) == TOKS_CLS_P, "U+200B is P (not \\s)", 0x200B);
        check((toks_class_of(&t0, 0xFEFF) & 7u) == TOKS_CLS_P, "U+FEFF is P (not \\s)", 0xFEFF);
        /* CR/LF must also have no letter bits; WS chars are not letters */
        check((toks_class_of(&t0, 0x0A) & TOKS_CLS_UPPER) == 0, "LF no UPPER", 0xA);
    }
    {
        /* ſ: FOLD_S bit set, base class L (it is Ll), in both settings */
        uint8_t c0 = toks_class_of(&t0, 0x17F);
        uint8_t c1 = toks_class_of(&t1, 0x17F);
        check((c0 & TOKS_CLS_FOLD_S) != 0, "U+017F has FOLD_S", 0x17F);
        check((c0 & 7u) == TOKS_CLS_L, "U+017F base L", 0x17F);
        check((c0 & TOKS_CLS_LOWER) != 0, "U+017F has LOWER", 0x17F);
        check(c0 == c1, "U+017F same in both settings", 0x17F);
        /* no other code point carries FOLD_S */
        for (cp = 0x80; cp < N_CP; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) continue;
            if ((toks_class_of(&t0, (uint32_t)cp) & TOKS_CLS_FOLD_S) != 0) {
                check(cp == 0x17F, "only U+017F has FOLD_S", cp);
            }
        }
    }
    {
        /* post-unicode-9 code points: classified by onig (16.0 data) */
        unsigned long late[] = {0x897, 0x9FEA, 0x30000, 0x105C0};
        size_t i;
        for (i = 0; i < sizeof late / sizeof late[0]; i++) {
            unsigned long c = late[i];
            if (c == 0x897) {
                /* Todhri mark (14.0): MARK in both, P unless marks are letters */
                check((toks_class_of(&t0, (uint32_t)c) & 7u) == TOKS_CLS_P,
                      "U+0897 base P when f=0", c);
                check((toks_class_of(&t1, (uint32_t)c) & 7u) == TOKS_CLS_L,
                      "U+0897 base L when f=1", c);
            } else {
                /* late letters (13.0 / 16.0 additions) */
                check((toks_class_of(&t0, (uint32_t)c) & 7u) == TOKS_CLS_L,
                      "late letter is a letter", c);
            }
        }
        /* a mark: MARK bit in both; base L only when MARKS_ARE_LETTERS */
        check((toks_class_of(&t0, 0x300) & TOKS_CLS_MARK) != 0, "U+0300 MARK bit", 0x300);
        check((toks_class_of(&t0, 0x300) & 7u) == TOKS_CLS_P, "U+0300 base P when f=0", 0x300);
        check((toks_class_of(&t1, 0x300) & 7u) == TOKS_CLS_L, "U+0300 base L when f=1", 0x300);
        check((toks_class_of(&t1, 0x300) & TOKS_CLS_MARK) != 0, "U+0300 MARK when f=1", 0x300);
        /* marks keep UPPER/LOWER (they are in both case groups) */
        check((toks_class_of(&t1, 0x300) & TOKS_CLS_UPPER) != 0, "U+0300 UPPER", 0x300);
        check((toks_class_of(&t1, 0x300) & TOKS_CLS_LOWER) != 0, "U+0300 LOWER", 0x300);
        /* U+0897: mark added in 14.0 (Todhri) */
        check((toks_class_of(&t0, 0x897) & TOKS_CLS_MARK) != 0, "U+0897 MARK", 0x897);
        check((toks_class_of(&t1, 0x897) & 7u) == TOKS_CLS_L, "U+0897 base L when f=1", 0x897);
        /* ascii sanity: 'a' L|LOWER, 'A' L|UPPER, '0' N, ' ' WS, '.' P */
        check(toks_class_of(&t0, 'a') == (TOKS_CLS_L | TOKS_CLS_LOWER), "'a' class", 'a');
        check(toks_class_of(&t0, 'A') == (TOKS_CLS_L | TOKS_CLS_UPPER), "'A' class", 'A');
        check(toks_class_of(&t0, '0') == TOKS_CLS_N, "'0' class", '0');
        check(toks_class_of(&t0, ' ') == TOKS_CLS_WS, "' ' class", ' ');
        check(toks_class_of(&t0, '.') == TOKS_CLS_P, "'.' class", '.');
        /* titlecase: U+01C5 (Dž) is Lt -> UPPER, base L */
        check((toks_class_of(&t0, 0x1C5) & TOKS_CLS_UPPER) != 0, "U+01C5 (Lt) UPPER", 0x1C5);
        check((toks_class_of(&t0, 0x1C5) & 7u) == TOKS_CLS_L, "U+01C5 base L", 0x1C5);
    }

    /* ---- ascii table == stage2 block 0 prefix == toks_class_of ---- */
    for (cp = 0; cp < 128u; cp++) {
        uint8_t a = t0.ascii[cp];
        uint8_t b = toks_class_of(&t0, (uint32_t)cp);
        uint8_t c = t0.stage2[t0.stage1[0] * 256u + cp];
        check(a == b && b == c, "ascii/stage2/class_of agree", cp);
    }

    /* ---- MARKS_ARE_LETTERS only ever moves M chars P -> L ---- */
    for (cp = 0; cp < N_CP; cp++) {
        uint8_t a = toks_class_of(&t0, (uint32_t)cp);
        uint8_t b = toks_class_of(&t1, (uint32_t)cp);
        if ((a & TOKS_CLS_MARK) != 0) {
            check(a == (uint8_t)(b & ~0x07u), "M: only base differs", cp);
        } else if (cp < 0x80u && (uint32_t)cp != 0x0Au && (uint32_t)cp != 0x0Du) {
            /* non-marks identical (CR/LF handled: both NL in both) */
        }
        if ((a & TOKS_CLS_MARK) == 0) {
            check(a == b, "non-M identical across settings", cp);
        }
    }

    if (failures == 0) {
        printf("unicode test: OK (%d scalar values x 2 settings, contracts, "
               "targeted checks; n_blocks %lu / %lu)\n",
               N_CP - 2048, (unsigned long)t0.n_blocks, (unsigned long)t1.n_blocks);
        return 0;
    }
    printf("unicode test: %d FAILURES\n", failures);
    return 1;
}
