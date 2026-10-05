/* rationale: docs/notes/c-core.md §classes.h.1 */
#ifndef TOKS_CLASSES_H
#define TOKS_CLASSES_H

#include <stdint.h>

#ifndef TOKS_CLS_P
#define TOKS_CLS_P 0x00u /* not L, N or \s */
#define TOKS_CLS_L 0x01u /* letter (marks too under MARKS_ARE_LETTERS) */
#define TOKS_CLS_N 0x02u /* \p{N} */
#define TOKS_CLS_WS 0x03u /* \s except CR and LF */
#define TOKS_CLS_NL 0x04u /* CR or LF */
#define TOKS_CLS_UPPER 0x08u /* Lu, Lt, Lm, Lo, M */
#define TOKS_CLS_LOWER 0x10u /* Ll, Lm, Lo, M */
#define TOKS_CLS_MARK 0x20u /* \p{M} */
#define TOKS_CLS_FOLD_S 0x40u /* U+017F: (?i) matches it as 's' */
#endif
#ifndef TOKS_CLS_HAN
#define TOKS_CLS_HAN 0x80u /* \p{Han} (Script=Han), under TOKS_CLASSES_HAN only */
#endif
#ifndef TOKS_CLS_X
#define TOKS_CLS_X 0x05u /* dsv3 tables: not L, M, N, P, S or \s */
#define TOKS_CLS_CJK 0x80u /* dsv3 tables: deepseek v3's literal CJK ranges */
#endif

/* rationale: docs/notes/c-core.md §classes.h.2 */
#define TOKS_CLASSES_MARKS_ARE_LETTERS 0x00000001u
#define TOKS_CLASSES_HAN 0x00000002u
#define TOKS_CLASSES_DSV3 0x00000008u
#define TOKS_CLASSES_DIGITS 0x00000020u /* hf Digits' is_numeric is base N (\p{N} + 13 newer code points) */
#define TOKS_CLASSES_BLOOM 0x00000040u /* bloom (P21): \s and its class's 14 literals WS / NL, all else P (kernels.md §3) */
#define TOKS_CLASSES_KNOWN (TOKS_CLASSES_MARKS_ARE_LETTERS | TOKS_CLASSES_HAN | TOKS_CLASSES_DSV3 | TOKS_CLASSES_DIGITS | \
                            TOKS_CLASSES_BLOOM)

/* error returns of toks_classes_build (mapped onto the toks.h codes when the
 * abi header lands: TOKS_E_CAP / TOKS_E_UNSUPPORTED / TOKS_E_ARG) */
#define TOKS_CLASSES_E_CAP (-1) /* buf too small for the worst case */
#define TOKS_CLASSES_E_FLAGS (-2) /* unknown class_flags bit */
#define TOKS_CLASSES_E_ARG (-3) /* out == NULL */

typedef struct {
    const uint8_t *ascii; /* [128]: class per ASCII byte */
    const uint16_t *stage1; /* [0x1100]: block index per 256 code points */
    const uint8_t *stage2; /* [n_blocks * 256]: class per code point */
    uint32_t n_blocks;
    uint32_t _pad;
} toks_class_tables;

/* rationale: docs/notes/c-core.md §classes.h.3 */
uint64_t toks_classes_bytes(uint32_t class_flags);

/* rationale: docs/notes/c-core.md §classes.h.4 */
int toks_classes_build(uint32_t class_flags, uint8_t *buf, uint64_t buf_len,
                       toks_class_tables *out);

/* class of one code point; cp > U+10FFFF is class P (0). */
static inline uint8_t toks_class_of(const toks_class_tables *t, uint32_t cp)
{
    if (cp > 0x10FFFFu) {
        return TOKS_CLS_P;
    }
    if (cp < 0x80u) {
        return t->ascii[cp];
    }
    return t->stage2[((uint32_t)t->stage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_CLASSES_H */
