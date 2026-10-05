/* norm.h: the normalizer, one engine whose steps are data (docs/notes/c-core.md §norm.h.1) */
#ifndef TOKS_NORM_H
#define TOKS_NORM_H

#include <stdint.h>

#include "core.h"

/* the steps of a form, applied to every char in this order */
#define TOKS_NS_CLEAN     0x01u   /* BertNormalizer clean_text: controls removed, White_Space -> ' ' */
#define TOKS_NS_CJK       0x02u   /* handle_chinese_chars: ' ' before and after each CJK char */
#define TOKS_NS_STRIP_MN  0x04u   /* strip_accents: canonical decomposition, then Mn (unicode 8.0) removed */
#define TOKS_NS_LOWER     0x08u   /* char::to_lowercase */
#define TOKS_NS_CANON     0x10u   /* canonical decomposition, non-starters reordered */
#define TOKS_NS_COMPOSE   0x20u   /* canonical composition */
#define TOKS_NS_COMPAT    0x40u   /* compatibility decomposition (else canonical), non-starters reordered */
#define TOKS_NS_STRIP_M   0x80u   /* StripAccents: General_Category=Mark (unicode 9.0) removed */
#define TOKS_NS_NFC       (TOKS_NS_CANON | TOKS_NS_COMPOSE)
#define TOKS_NS_NFKC      (TOKS_NS_NFC | TOKS_NS_COMPAT)
#define TOKS_NS_NFKD      (TOKS_NS_CANON | TOKS_NS_COMPAT)

/* output bytes per input byte of a form, worst case (core.h's scratch factors; norm.h §1) */
#define TOKS_NORM_X(f)          (((f) & TOKS_NS_COMPAT) != 0u ? TOKS_NFKC_X : TOKS_NFC_X)
#define TOKS_NORM_BOUND(f, n)   ((uint64_t)TOKS_NORM_X(f) * (uint64_t)(n))

/* rationale: docs/notes/c-core.md §norm.h.2 */
uint64_t toks_k2(const uint8_t *text, uint64_t pos, uint64_t len, uint8_t hot);

/* rationale: docs/notes/c-core.md §norm.h.3 */
int64_t toks_norm(uint32_t steps, const uint8_t *text, uint64_t len, uint8_t *out, uint64_t cap);

/* rationale: docs/notes/c-core.md §norm.h.4 */
#define TOKS_NORM_MAX_OUT 8u
#define TOKS_NORM_GHOST   0x00200000u
uint32_t toks_norm_char(uint32_t steps, uint32_t cp, uint32_t o[TOKS_NORM_MAX_OUT]);

/* ---- the read-only scan: the driver's zero-copy path (maintainer doctrine, speed) --------------------------------- */

#define TOKS_NFC_SCAN_RUN 64u   /* longest run the scan normalizes on its stack to compare */

/* rationale: docs/notes/c-core.md §norm.h.5 */
uint64_t toks_nfc_scan(uint32_t f, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t *run_end);

/* rationale: docs/notes/c-core.md §norm.h.6 */
int toks_nfc_boundary(uint32_t f, const uint8_t *text, uint64_t len, uint64_t i);

/* rationale: docs/notes/c-core.md §norm.h.7 */
typedef struct toks_nfc_plan {
    uint64_t r;        /* the end of the last stretch */
    uint64_t d0, d1;   /* the next changed run [d0, d1); d0 == n: none */
    uint32_t f;        /* the form: NFC or NFKC */
} toks_nfc_plan;

void toks_nfc_plan_begin(toks_nfc_plan *p, uint32_t f, const uint8_t *g, uint64_t n);
int toks_nfc_plan_next(toks_nfc_plan *p, const toks_tables *t, const uint8_t *g, uint64_t n, uint64_t *s,
                       uint64_t *e);

#endif /* TOKS_NORM_H */
