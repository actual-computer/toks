/* rationale: docs/notes/c-core.md §classes.c.1 */

#include "classes.h"
#include "../gen/ucd_flags.h"
#include "../gen/han_ranges.h"

#define TOKS_CLS_ASCII_LEN 128u
#define TOKS_CLS_STAGE1_LEN 0x1100u /* 256-cp blocks over U+0000..U+10FFFF */
#define TOKS_CLS_STAGE1_BYTES (2u * TOKS_CLS_STAGE1_LEN)
#define TOKS_CLS_BLOCK 256u

/* rationale: docs/notes/c-core.md §classes.c.2 */
#define TOKS_DSV3_CJK_N 2u
static const uint32_t TOKS_DSV3_CJK[TOKS_DSV3_CJK_N][2] = { { 0x3040u, 0x30FFu }, { 0x4E00u, 0x9FA5u } };
/* bloom's  ?[^(\s|[.,!?…。，、।۔،])]+ : the literals its class leaves out besides \s (probed through hf) */
static const uint32_t TOKS_BLOOM_E[14] = { '!', '(', ')', ',', '.', '?', '|', 0x060Cu, 0x06D4u, 0x0964u, 0x2026u,
                                           0x3001u, 0x3002u, 0xFF0Cu };

/* flags of one code point -> its class byte (layout: classes.h).
 * base class first, then the attribute bits; bit 7 (HAN) is the caller's. */
static uint8_t toks_class_from_flags(uint16_t f, uint32_t cp, int marks_are_letters)
{
    uint8_t b;

    if ((f & TOKS_UCD_LETTERS) != 0 ||
        (marks_are_letters != 0 && (f & TOKS_UCD_M) != 0)) {
        b = TOKS_CLS_L;
    } else if ((f & TOKS_UCD_N) != 0) {
        b = TOKS_CLS_N;
    } else if (cp == 0x0Au || cp == 0x0Du) {
        b = TOKS_CLS_NL; /* CR and LF: base class NL, not WS */
    } else if ((f & TOKS_UCD_WS) != 0) {
        b = TOKS_CLS_WS;
    } else {
        b = TOKS_CLS_P;
    }
    if ((f & (TOKS_UCD_LU | TOKS_UCD_LT | TOKS_UCD_LM | TOKS_UCD_LO | TOKS_UCD_M)) != 0) {
        b |= TOKS_CLS_UPPER;
    }
    if ((f & (TOKS_UCD_LL | TOKS_UCD_LM | TOKS_UCD_LO | TOKS_UCD_M)) != 0) {
        b |= TOKS_CLS_LOWER;
    }
    if ((f & TOKS_UCD_M) != 0) {
        b |= TOKS_CLS_MARK;
    }
    if ((f & TOKS_UCD_FOLD_S) != 0) {
        b |= TOKS_CLS_FOLD_S;
    }
    return b;
}

/* rationale: docs/notes/c-core.md §classes.c.3 */
static uint8_t toks_class_dsv3(uint16_t f, uint32_t cp)
{
    uint8_t b;
    uint32_t r;

    if ((f & (TOKS_UCD_LETTERS | TOKS_UCD_M)) != 0) {
        b = TOKS_CLS_L;
    } else if ((f & TOKS_UCD_N) != 0) {
        b = TOKS_CLS_N;
    } else if (cp == 0x0Au || cp == 0x0Du) {
        b = TOKS_CLS_NL;
    } else if ((f & TOKS_UCD_WS) != 0) {
        b = TOKS_CLS_WS;
    } else if ((f & (TOKS_UCD_P | TOKS_UCD_S)) != 0) {
        b = TOKS_CLS_P;
    } else {
        b = TOKS_CLS_X;
    }
    for (r = 0; r < TOKS_DSV3_CJK_N; r++) {             /* bound: 2 ranges */
        if (cp >= TOKS_DSV3_CJK[r][0] && cp <= TOKS_DSV3_CJK[r][1]) {
            b = (uint8_t)(b | TOKS_CLS_CJK);
        }
    }
    return b;
}

/* first differing position + 1, 0 if equal (avoids memcmp; bounded 256) */
static uint32_t toks_block_differs(const uint8_t *a, const uint8_t *b)
{
    uint32_t i;
    for (i = 0; i < TOKS_CLS_BLOCK; i++) {
        if (a[i] != b[i]) {
            return i + 1u;
        }
    }
    return 0u;
}

uint64_t toks_classes_bytes(uint32_t class_flags)
{
    if ((class_flags & ~TOKS_CLASSES_KNOWN) != 0u) {
        return 0u; /* unknown bits: build would refuse */
    }
    return (uint64_t)TOKS_CLS_ASCII_LEN + TOKS_CLS_STAGE1_BYTES +
           (uint64_t)TOKS_CLS_STAGE1_LEN * TOKS_CLS_BLOCK;
}

int toks_classes_build(uint32_t class_flags, uint8_t *buf, uint64_t buf_len,
                       toks_class_tables *out)
{
    uint16_t *s1;
    uint8_t *s2;
    uint64_t s2_off;
    uint32_t hi;
    uint32_t hr; /* Han cursor: the first range not wholly below the code point */
    int marks_are_letters;
    int han;
    int dsv3, bloom;
    uint16_t rnum;

    if (out == 0) {
        return TOKS_CLASSES_E_ARG;
    }
    if ((class_flags & ~TOKS_CLASSES_KNOWN) != 0u) {
        return TOKS_CLASSES_E_FLAGS;
    }
    marks_are_letters = (class_flags & TOKS_CLASSES_MARKS_ARE_LETTERS) != 0;
    han = (class_flags & TOKS_CLASSES_HAN) != 0;
    hr = 0u;
    dsv3 = (class_flags & TOKS_CLASSES_DSV3) != 0;
    bloom = (class_flags & TOKS_CLASSES_BLOOM) != 0;
    rnum = (class_flags & TOKS_CLASSES_DIGITS) != 0u ? TOKS_UCD_RNUM : 0u;
    if (buf == 0 || buf_len < toks_classes_bytes(class_flags)) {
        return TOKS_CLASSES_E_CAP;
    }

    s1 = (uint16_t *)(void *)(buf + TOKS_CLS_ASCII_LEN);
    s2 = buf + TOKS_CLS_ASCII_LEN + TOKS_CLS_STAGE1_BYTES;
    s2_off = 0;

    /* over all 0x1100 blocks, in code point order */
    for (hi = 0; hi < TOKS_CLS_STAGE1_LEN; hi++) {
        uint8_t row[TOKS_CLS_BLOCK];
        uint32_t j;
        uint64_t k;
        int found;

        for (j = 0; j < TOKS_CLS_BLOCK; j++) {
            uint32_t cp = hi * TOKS_CLS_BLOCK + j;
            uint16_t f = toks_ucd_flags(cp);
            f = (uint16_t)((f & rnum) != 0u ? f | TOKS_UCD_N : f);   /* the 13 beyond \p{N} have no other flag */
            row[j] = dsv3 != 0 ? toks_class_dsv3(f, cp) : toks_class_from_flags(f, cp, marks_are_letters);
            if (bloom != 0) {                                    /* \s and the literals stay WS / NL, all else P */
                uint8_t b = (uint8_t)(row[j] & 7u);
                row[j] = (b == TOKS_CLS_WS || b == TOKS_CLS_NL) ? b : TOKS_CLS_P;
                for (uint32_t e = 0; cp < 0x10000u && e < 14u; e++) { row[j] = cp == TOKS_BLOOM_E[e] ? TOKS_CLS_WS : row[j]; }
            }
            if (han != 0) {
                /* rationale: docs/notes/c-core.md §classes.c.4 */
                while (hr < TOKS_HAN_N && cp > toks_han_ranges[hr][1]) {
                    hr++;
                }
                if (hr < TOKS_HAN_N && cp >= toks_han_ranges[hr][0]) {
                    row[j] = (uint8_t)(row[j] | TOKS_CLS_HAN);
                }
            }
        }

        /* dedup: scan the written blocks, at most 0x1100 of them */
        found = 0;
        for (k = 0; k < s2_off; k += TOKS_CLS_BLOCK) {
            if (toks_block_differs(s2 + k, row) == 0u) {
                s1[hi] = (uint16_t)(k / TOKS_CLS_BLOCK);
                found = 1;
                break;
            }
        }
        if (found == 0) {
            for (j = 0; j < TOKS_CLS_BLOCK; j++) {
                s2[s2_off + j] = row[j];
            }
            s1[hi] = (uint16_t)(s2_off / TOKS_CLS_BLOCK);
            s2_off += TOKS_CLS_BLOCK;
        }
    }

    /* ASCII table = block 0's first 128 bytes (stage1[0] == 0 by construction:
     * block 0 is written before any other, so it is block 0). */
    for (hi = 0; hi < TOKS_CLS_ASCII_LEN; hi++) {
        buf[hi] = s2[hi];
    }

    out->ascii = buf;
    out->stage1 = s1;
    out->stage2 = s2;
    out->n_blocks = (uint32_t)(s2_off / TOKS_CLS_BLOCK);
    out->_pad = 0u;
    return (int)(TOKS_CLS_ASCII_LEN + TOKS_CLS_STAGE1_BYTES + s2_off);
}
