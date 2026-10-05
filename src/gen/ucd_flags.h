/* toks: per-code-point unicode property flags as hf tokenizers' regex engine
 * (oniguruma) sees them -- NOT python's unicodedata.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:
 *     uv run tools/gen/unicode.py
 * (deps pinned in that script's PEP 723 header: tokenizers==0.23.2)
 *
 * Probed facts (every scalar value U+0000..U+10FFFF, surrogates flags 0):
 *   \p{L} == \p{Lu}|\p{Ll}|\p{Lt}|\p{Lm}|\p{Lo}: 141028 code points (probed both ways)
 *   \p{M} 2501, \p{N} 1911, \s 25, \p{P} 855, \p{S} 8514 code points
 *   \d 760, \w 144671; hf Digits' is_numeric 1924 (\p{N} plus U+11DE0 U+11DE1 U+11DE2 U+11DE3 U+11DE4 U+11DE5 U+11DE6 U+11DE7 U+11DE8 U+11DE9 U+16FF4 U+16FF5 U+16FF6),
 *   hf Punctuation's is_punc 726
 *   L, M, N, \s, P, S pairwise disjoint
 *   \s = U+0009..000D, 0020, 0085, 00A0, 1680, 2000..200A, 2028, 2029, 202F, 205F, 3000
 *   (?i) folds U+017F (long s) onto s: the only non-ASCII fold reaching the
 *   contraction set of (?i:'s|'t|'re|'ve|'m|'ll|'d) (every scalar probed with
 *   the full group, in 'X, 'Xe, 'Xl contexts; no other fold matched)
 *   classes agree with unicode 16.0 (== python 3.14 unicodedata on L/M/N/\s/P/S);
 *   python 3.12's unicodedata 15.0 disagrees exactly on post-15.0 additions
 *
 * two-level table: stage1[cp >> 8] is the block id, then
 * stage2[(block << 8) | (cp & 0xFF)] is the flag word (0 for surrogates).
 *
 * table data (n_blocks u32 LE || stage1 u16[0x1100] LE || stage2 u16[n_blocks*256] LE)
 * sha256: 14ade6123d97ec55c668bd2ec93f44bce842ab8a22342658de6311d68f94abff
 */

#ifndef TOKS_UCD_FLAGS_H
#define TOKS_UCD_FLAGS_H

#include <stdint.h>

/* flag bits of toks_ucd_flags(); each is 1 iff oniguruma (as run by hf
 * tokenizers 0.23.2) has the code point in the property. */
#define TOKS_UCD_LU 0x0001u /* \p{Lu} */
#define TOKS_UCD_LL 0x0002u /* \p{Ll} */
#define TOKS_UCD_LT 0x0004u /* \p{Lt} */
#define TOKS_UCD_LM 0x0008u /* \p{Lm} */
#define TOKS_UCD_LO 0x0010u /* \p{Lo} */
#define TOKS_UCD_M 0x0020u  /* \p{M} */
#define TOKS_UCD_N 0x0040u  /* \p{N} */
#define TOKS_UCD_WS 0x0080u /* \s */
#define TOKS_UCD_P 0x0200u  /* \p{P} */
#define TOKS_UCD_S 0x0400u  /* \p{S} */
/* hf's other pre-tokenizer sets, probed through hf too: Digits' rust char::is_numeric (a superset of \p{N}: the
 * compiler's newer unicode) and Punctuation's is_punc; the regex engine's \d and \w */
#define TOKS_UCD_RNUM 0x0800u
#define TOKS_UCD_ND 0x1000u
#define TOKS_UCD_WORD 0x2000u
#define TOKS_UCD_PUNC 0x4000u
/* non-ASCII code point that (?i) matches as 's' (U+017F long s; the only one) */
#define TOKS_UCD_FOLD_S 0x0100u

/* \p{L} == \p{Lu}|\p{Ll}|\p{Lt}|\p{Lm}|\p{Lo} (probed, asserted at generation) */
#define TOKS_UCD_LETTERS (TOKS_UCD_LU | TOKS_UCD_LL | TOKS_UCD_LT | TOKS_UCD_LM | TOKS_UCD_LO)

/* deduplicated stage 2 blocks of 256 code points */
#define TOKS_UCD_N_BLOCKS 155
/* sha256 of n_blocks u32 LE || stage1 || stage2 (see the file header) */
#define TOKS_UCD_DATA_SHA256 "14ade6123d97ec55c668bd2ec93f44bce842ab8a22342658de6311d68f94abff"

extern const uint16_t toks_ucd_stage1[0x1100];
extern const uint16_t toks_ucd_stage2[TOKS_UCD_N_BLOCKS * 256];

/* flags of one code point; 0 for cp > U+10FFFF (and for surrogates). */
static inline uint16_t toks_ucd_flags(uint32_t cp)
{
    if (cp > 0x10FFFFu) {
        return 0;
    }
    return toks_ucd_stage2[((uint32_t)toks_ucd_stage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_UCD_FLAGS_H */
