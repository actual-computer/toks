/* toks: NFC / NFKC data exactly as hf tokenizers 0.23.2 runs it: crate unicode-normalization-alignments 0.1.12
 * (sha256 43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de, pinned by tokenizers v0.23.2's Cargo.lock), whose tables.rs is UNICODE 9.0.0.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with: uv run tools/gen/norm.py
 * The generator parses the crate's own tables and verifies its model of the crate against hf's NFC and NFD on
 * every scalar alone, every scalar between two marks, the class order, every composition first x second,
 * all of hangul (docs: src/core/norm.h).
 *
 * Facts asserted by the generator (the C code relies on them):
 *   - every code point below U+0300 is a boundary NFC leaves unchanged: a byte < 0xCC never needs work
 *   - a full canonical decomposition has <= 4 chars; its utf-8 is <= 3x the char's (U+0390 is 3x)
 *   - composing never lengthens the utf-8 (len(c) <= len(a) + len(b) for every pair); firsts are starters
 *   - hangul syllables are boundaries; L+V and LV+T compose by arithmetic (no table entries)
 *   - NFKC: every code point below U+00A0 is a boundary it leaves unchanged (a byte < 0xC2 never needs
 *     work); a full compatibility decomposition has <= 18 chars, its utf-8 <= 11x the char's (U+FDFA)
 * counts: 814 non-starters in 54 classes, 2060 decompositions, 940 pairs, 2001 NB code points;
 *   3678 compatibility decompositions, 5675 NBK code points, 2097 marks
 * table data sha256 (n_blocks u32 || stage1 || stage2 || pool || comp || cls_ccc || kstage1 || kstage2 || fast, LE):
 *   084588bfbf11505c62d2379b23ed97ded4986e2254ae3de94e891ab6a8e8179b
 */
#ifndef TOKS_NORM_NFC_H
#define TOKS_NORM_NFC_H

#include <stdint.h>

/* info word of a code point, toks_nfc_info(cp):
 *   bits  0..12  offset of its full canonical decomposition in toks_nfc_pool (when dlen > 0)
 *   bits 13..15  dlen: decomposition length 1..4, or 0 when the char is its own decomposition
 *   bit  16      NB: not a boundary NFC leaves unchanged (a char with NB clear has a starter first decomposed
 *                char that composes with nothing before it, and NFC maps it to itself)
 *   bit  17      NBK: the same for NFKC
 *   bit  18      KX: the char has a compatibility decomposition (toks_nfc_kinfo: offset | length << 16 in the pool)
 *   bit  19      MARK: General_Category=Mark (hf's StripAccents removes it)
 *   bit  21      SECOND: the char may compose with a preceding char (the second of a pair, hangul V or T)
 *   bits 24..29  cls: dense rank of its canonical combining class (0 = starter; ccc = toks_nfc_cls_ccc[cls])
 * pool entry: bits 0..20 the code point, bit 21 SECOND, bits 24..29 cls (as above). */
#define TOKS_NFC_DOFF_MASK    0x1FFFu
#define TOKS_NFC_DLEN_SHIFT   13
#define TOKS_NFC_DLEN_MASK    0x7u
#define TOKS_NFC_NB           0x00010000u
#define TOKS_NFC_NBK          0x00020000u
#define TOKS_NFC_KX           0x00040000u
#define TOKS_NFC_MARK         0x00080000u
#define TOKS_NFC_SECOND       0x00200000u
#define TOKS_NFC_CLS_SHIFT    24
#define TOKS_NFC_CLS_MASK     0x3Fu
#define TOKS_NFC_CP_MASK      0x001FFFFFu

#define TOKS_NFC_HOT_BYTE     0xCCu  /* the K2 predicate: no byte >= this, no NFC work */
#define TOKS_NFKC_HOT_BYTE    0xC2u  /* ... no NFKC work */
#define TOKS_NFC_MAX_DECOMP   4             /* chars in a full canonical decomposition */
#define TOKS_NFC_MAX_KDECOMP  18            /* chars in a full compatibility decomposition */
#define TOKS_NFC_EXPANSION    3             /* utf-8 bytes out per byte in, worst case (proof: norm.h) */
#define TOKS_NFKC_EXPANSION   11            /* the same with compatibility decompositions */
#define TOKS_NFC_N_CLS        54            /* non-zero combining classes (cls 1..N_CLS) */

#define TOKS_NFC_N_BLOCKS     89
#define TOKS_NFC_N_KBLOCKS    41
#define TOKS_NFC_POOL_LEN     6619
#define TOKS_NFC_COMP_LOG2    11
#define TOKS_NFC_COMP_MAXPROBE 7          /* longest linear probe of any present pair */
#define TOKS_NFC_FIB64        0x9E3779B97F4A7C15ull
#define TOKS_NFC_DATA_SHA256  "084588bfbf11505c62d2379b23ed97ded4986e2254ae3de94e891ab6a8e8179b"

extern const uint8_t  toks_nfc_stage1[0x1100];
extern const uint32_t toks_nfc_stage2[TOKS_NFC_N_BLOCKS * 256];
extern const uint32_t toks_nfc_pool[TOKS_NFC_POOL_LEN];
/* composition pairs: slot = (a << 21 | b) << 21 | composite, 0 empty; home slot
 * ((a << 21 | b) * TOKS_NFC_FIB64) >> (64 - TOKS_NFC_COMP_LOG2), linear probing */
extern const uint64_t toks_nfc_comp[1u << TOKS_NFC_COMP_LOG2];
extern const uint8_t  toks_nfc_cls_ccc[TOKS_NFC_N_CLS + 1];
extern const uint8_t  toks_nfc_kstage1[0x1100];
extern const uint32_t toks_nfc_kstage2[TOKS_NFC_N_KBLOCKS * 256];
/* the scan's per-code-point classes: bit (cp & 63) of toks_nfc_fast[f * 1024 + (cp >> 6)] is set when the BMP
 * code point cp is a boundary the form f (0 NFC, 1 NFKC) leaves unchanged (NB / NBK clear); never a surrogate */
extern const uint64_t toks_nfc_fast[2048];

/* info word of a scalar value (cp <= U+10FFFF) */
static inline uint32_t toks_nfc_info(uint32_t cp)
{
    return toks_nfc_stage2[((uint32_t)toks_nfc_stage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

/* its full compatibility decomposition: pool offset | length << 16, 0 when it has none (KX clear) */
static inline uint32_t toks_nfc_kinfo(uint32_t cp)
{
    return toks_nfc_kstage2[((uint32_t)toks_nfc_kstage1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_NORM_NFC_H */
