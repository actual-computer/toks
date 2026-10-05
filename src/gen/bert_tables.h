/* toks: the unicode tables of hf tokenizers 0.23.2's BertNormalizer and BertPreTokenizer.
 *
 * GENERATED FILE -- DO NOT EDIT.  Regenerate with:  uv run tools/gen/bert_tables.py
 * Sources (pinned by sha256 in the generator): unicode_categories 0.1.1 (Unicode 8.0.0: Cc Cf Co, P*, Mn),
 * unicode-normalization-alignments 0.1.12 (Unicode 9.0.0 NFD), rust std's char::to_lowercase / is_whitespace
 * (Unicode 17.0.0), is_chinese_char's ranges. Verified against hf on every scalar value (docs/algorithms/wordpiece.md).
 *
 * class byte of a code point (two-level: toks_bert_cls_s2[toks_bert_cls_s1[cp >> 8] << 8 | (cp & 0xFF)]):
 */
#ifndef TOKS_BERT_TABLES_H
#define TOKS_BERT_TABLES_H

#include <stdint.h>

#define TOKS_BC_REMOVE  0x01u  /* clean_text removes it: U+0000, U+FFFD, Cc Cf Co (8.0) except \t \n \r */
#define TOKS_BC_WS      0x02u  /* White_Space (17.0): clean_text maps it to ' '; the pre-tokenizers split on it */
#define TOKS_BC_CJK     0x04u  /* is_chinese_char: handle_chinese_chars puts ' ' before and after it */
#define TOKS_BC_PUNCT   0x08u  /* BertPreTokenizer isolates it: ascii punctuation or P* (8.0) */
#define TOKS_BC_DECOMP  0x10u  /* NFD (9.0) changes it (hangul syllables: arithmetic, no map entry) */
#define TOKS_BC_LOWER   0x20u  /* char::to_lowercase (17.0) changes it */
#define TOKS_BC_NS      0x40u  /* canonical combining class > 0 (9.0): a non-starter */
#define TOKS_BC_MN      0x80u  /* Mn (8.0): strip_accents removes it after NFD */

#define TOKS_BERT_CLS_NBLOCKS  101
#define TOKS_BERT_MAP_NBLOCKS  50
#define TOKS_BERT_MAP_N        3169   /* map entries, entry 0 = none */
#define TOKS_BERT_POOL_N       4893
#define TOKS_BERT_KEEPNS_N     83   /* non-starters strip_accents keeps (ccc > 0, not Mn 8.0) */
#define TOKS_BERT_MAX_DECOMP   4    /* longest NFD expansion of one code point (non-hangul) */
#define TOKS_BERT_MAX_LOWER    2    /* longest to_lowercase expansion (U+0130) */
/* the normalizer's output never exceeds GROWTH_NUM / GROWTH_DEN times its input bytes (worst: U+AC01) */
#define TOKS_BERT_GROWTH_NUM   9
#define TOKS_BERT_GROWTH_DEN   3
#define TOKS_BERT_DATA_SHA256  "27561609130ea82cbf3b6aaa98b3f5f4fc05bc3e284ae32e7a2c17196ee76a74"

typedef struct toks_bert_map {
    uint16_t dec_off;   /* NFD decomposition: pool[dec_off .. dec_off + dec_len) */
    uint8_t  dec_len;
    uint8_t  rsv0;
    uint16_t low_off;   /* to_lowercase: pool[low_off .. low_off + low_len) */
    uint8_t  low_len;
    uint8_t  rsv1;
} toks_bert_map;

extern const uint16_t      toks_bert_cls_s1[0x1100];
extern const uint8_t       toks_bert_cls_s2[TOKS_BERT_CLS_NBLOCKS * 256];
extern const uint16_t      toks_bert_map_s1[0x1100];
extern const uint16_t      toks_bert_map_s2[TOKS_BERT_MAP_NBLOCKS * 256];
extern const toks_bert_map toks_bert_maps[TOKS_BERT_MAP_N];
extern const uint32_t      toks_bert_pool[TOKS_BERT_POOL_N];
extern const uint32_t      toks_bert_keepns[TOKS_BERT_KEEPNS_N][2];   /* (cp, ccc), ascending cp */

static inline uint8_t toks_bert_cls(uint32_t cp)
{
    if (cp > 0x10FFFFu) { return 0; }
    return toks_bert_cls_s2[((uint32_t)toks_bert_cls_s1[cp >> 8] << 8) | (cp & 0xFFu)];
}

static inline uint16_t toks_bert_map_index(uint32_t cp)
{
    if (cp > 0x10FFFFu) { return 0; }
    return toks_bert_map_s2[((uint32_t)toks_bert_map_s1[cp >> 8] << 8) | (cp & 0xFFu)];
}

#endif /* TOKS_BERT_TABLES_H */
