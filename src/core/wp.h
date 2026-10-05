/* wp.h: the WordPiece / BERT text model of toks (docs/notes/c-core.md §wp.h.1) */
#ifndef TOKS_WP_H
#define TOKS_WP_H

#include "core.h"
#include "norm.h"
#include "../gen/bert_tables.h"

/* ---- tokenizer flags (toks_wp_tables.flags): the BertNormalizer's steps, norm.h's own bits -------------- */
#define TOKS_WPF_CLEAN      TOKS_NS_CLEAN      /* clean_text */
#define TOKS_WPF_CHINESE    TOKS_NS_CJK        /* handle_chinese_chars */
#define TOKS_WPF_STRIP      TOKS_NS_STRIP_MN   /* strip_accents (null already resolved to lowercase) */
#define TOKS_WPF_LOWER      TOKS_NS_LOWER      /* lowercase */

/* ---- ascii classes (toks_wp_tables.ascii_cls[b], b < 0x80), built from the flags ------------------------ */
#define TOKS_WPA_WORD       0u      /* joins the piece, unchanged */
#define TOKS_WPA_FOLD       1u      /* A-Z under lowercase: joins the piece, folded at lookup */
#define TOKS_WPA_SPLIT      2u      /* White_Space: ends the piece */
#define TOKS_WPA_PUNCT      3u      /* ascii punctuation: a piece of its own */
#define TOKS_WPA_REMOVE     4u      /* clean_text drops it (0x00-0x08, \v, \f, 0x0E-0x1F, 0x7F) */

/* ---- pieces (toks_wp_scan_c output; 16 bytes) ------------------------------------------------------------ */
#define TOKS_WPP_MAT        0x01u   /* off indexes the mat buffer (the normalized piece), else the text */
#define TOKS_WPP_OVER       0x02u   /* more than max_input_chars_per_word chars: the model answers [unk] */
#define TOKS_WPP_INVALID    0x04u   /* one ill-formed utf-8 byte (SPEC §3.3): [unk] */

typedef struct toks_wp_piece {
    uint32_t off;      /* byte offset of the piece's bytes (text or mat) */
    uint32_t len;      /* bytes */
    uint32_t end;      /* raw end in the text (a resume position once this piece is consumed) */
    uint32_t flags;    /* TOKS_WPP_* */
} toks_wp_piece;
#define WPP_OFF   0x0
#define WPP_LEN   0x4
#define WPP_END   0x8
#define WPP_FLAGS 0xC
#define WPP_SIZE  0x10

/* rationale: docs/notes/c-core.md §wp.h.2 */
typedef struct toks_wp_entry {
    uint32_t h;
    uint32_t id;
    uint32_t off;
    uint32_t len;
} toks_wp_entry;

#define TOKS_WP_HSEED       0x7F4A7C15u
#define TOKS_WP_MAX_KEY     65535u           /* bytes of one vocab string (SPEC §8.3: tokens <= 65,535 B) */

typedef struct toks_wp_tables {
    uint32_t flags;            /* TOKS_WPF_* */
    uint32_t max_chars;        /* max_input_chars_per_word (<= TOKS_WP_MAX_CHARS) */
    uint32_t unk_id;           /* vocab[unk_token]; toks_wp_build refuses a vocab without it */
    uint32_t word_maxlen;      /* longest key in word (bytes) */
    uint32_t cont_maxlen;      /* longest key in cont (bytes, prefix excluded) */
    uint32_t prefix_len;       /* bytes of continuing_subword_prefix */
    uint8_t  ascii_cls[128];   /* TOKS_WPA_* */
    uint8_t  ascii_cls0[128];  /* the same under scan flags 0 (the pre-tokenizer alone over normalized text) */
    const toks_wp_entry *word; /* every vocab string, keyed by itself */
    uint64_t word_mask;        /* entries - 1 */
    const toks_wp_entry *cont; /* every vocab string that starts with the prefix, keyed by the rest */
    uint64_t cont_mask;
    const uint8_t *keys;       /* key bytes (cont keys point inside word keys) */
    /* the whole-word answers (wordpiece.md §12.7) in K5's words-table format (bpe.h: 64-byte buckets, 2 ways x 2
     * buckets, a 16-byte key = the bytes + len in byte 15): every vocab string of 1..15 bytes a piece can be (under
     * lowercase: none with A-Z) -> its id; one line answers a piece. NULL: none */
    const uint8_t *wtab;
    uint64_t wtab_mask;
    /* W2's longest match (wordpiece.md §12.7): double-array tries (da.h) over the vocab strings (wcell, start 0) and
     * over the prefixed ones without the prefix (ccell, start > 0); cell = base (bit 31: a key ends here, its id in
     * the term array) + check; walked over the folded bytes. NULL: none (the hash probes of the candidates) */
    const struct toks_wp_cell *wcell, *ccell;
    const int32_t *wterm, *cterm;
} toks_wp_tables;

#define TOKS_WP_TERM 0x80000000u
#define TOKS_WP_BASE 0x7FFFFFFFu
typedef struct toks_wp_cell {
    uint32_t base;
    int32_t  check;
} toks_wp_cell;


/* ---- K-scan: pieces of text[pos, len) ------------------------------------------------------------------- */
typedef struct toks_wp_scan_args {
    const uint8_t *text;       /* the segment (a phase-0 gap, or the whole text) */
    uint64_t len;
    uint64_t pos;              /* in: a position where no piece is open (0 or an earlier return's pos); out */
    toks_wp_piece *pieces;     /* out: pieces[0, n) */
    uint64_t cap;              /* >= 1 */
    uint64_t n;
    uint8_t *mat;              /* materialized pieces: mat[0, mat_len) */
    uint64_t mat_cap;          /* >= toks_wp_mat_min(t) */
    uint64_t mat_len;          /* in: 0 (or where to continue); out */
    uint64_t flags;            /* TOKS_WPF_* override mask: the bits of t->flags the scan applies (all: ~0; the
                                  pre-tokenizer alone over already-normalized text: 0) */
    uint64_t rsv[6];
} toks_wp_scan_args;

/* mat bytes a scan needs to make progress: one maximal materialized piece (max_chars + 1 chars, each at most
 * TOKS_BERT_MAX_DECOMP * TOKS_BERT_MAX_LOWER chars of 4 bytes) plus a reorder area of the same size. */
uint64_t toks_wp_mat_min(const toks_wp_tables *t);

/* returns n. Stops at the end of the segment, when cap pieces are written, or before a piece the mat buffer
 * might not hold; a.pos is then where the next call resumes. */
uint64_t toks_wp_scan_c(const toks_wp_tables *t, toks_wp_scan_args *a);

/* ---- K9: ids of pieces ---------------------------------------------------------------------------------- */
typedef struct toks_wp_encode_args {
    const uint8_t *text;       /* the scan's text */
    const uint8_t *mat;        /* the scan's mat buffer */
    const toks_wp_piece *pieces;
    uint64_t n;
    uint32_t *out;             /* ids */
    uint64_t room;             /* >= the sum over pieces of (chars of the piece, at least 1) */
    uint64_t n_out;            /* out */
    uint64_t hits;             /* out: pieces answered by the whole-piece probe */
    uint64_t misses;           /* out: pieces that ran the greedy match */
    uint64_t probes;           /* out: hash probes of the greedy match (SPEC §7.1 counter) */
    uint64_t text_len;         /* text's bytes (in-place keys never read past them; 0: unknown, keys read only the
                                  piece's own bytes) */
    uint64_t mat_len;          /* mat's bytes (0: unknown) */
    uint8_t *cache;            /* the scratch's piece cache (kernels.md §6-7 format, epoch-tagged; greedy answers of
                                  <= 4 ids), NULL: none */
    uint64_t cache_mask;       /* buckets - 1 */
    uint64_t tw;               /* the epoch's tag word (layout.h toks_tag_word) */
    uint64_t rsv[1];
} toks_wp_encode_args;

uint64_t toks_wp_encode_c(const toks_wp_tables *t, toks_wp_encode_args *a);

/* the ids of one piece (bytes p[0, len), nch chars, TOKS_WPP_* flags), written at out (room >= nch, >= 1);
 * returns the count. The building block of toks_wp_encode_c, exposed for tests and the driver's tail cases. */
uint64_t toks_wp_piece_ids(const toks_wp_tables *t, const uint8_t *p, uint64_t len, uint32_t flags, uint32_t *out,
                           uint64_t *probes);

/* ---- utf-8 (unicode §3.9 table 3-7; an ill-formed byte is an atom of its own, SPEC §3.3) ---------------- */
#define TOKS_WP_INVALID_CP  0xFFFFFFFFu

/* decodes the atom at p[0, n) (n >= 1): returns its length (1..4) and its code point, or 1 and
 * TOKS_WP_INVALID_CP for an ill-formed byte. Reads p[0, min(n, 4)) only. */
static inline uint32_t wp_utf8(const uint8_t *p, uint64_t n, uint32_t *cp)
{
    uint32_t b0 = p[0];
    if (b0 < 0x80u) { *cp = b0; return 1; }
    *cp = TOKS_WP_INVALID_CP;
    if (b0 < 0xC2u || b0 > 0xF4u || n < 2u) { return 1; }
    uint32_t b1 = p[1];
    if (b0 < 0xE0u) {
        if ((b1 & 0xC0u) != 0x80u) { return 1; }
        *cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
        return 2;
    }
    uint32_t lo = 0x80u, hi = 0xBFu;
    if (b0 == 0xE0u) { lo = 0xA0u; }
    if (b0 == 0xEDu) { hi = 0x9Fu; }
    if (b0 == 0xF0u) { lo = 0x90u; }
    if (b0 == 0xF4u) { hi = 0x8Fu; }
    if (b1 < lo || b1 > hi || n < 3u) { return 1; }
    uint32_t b2 = p[2];
    if ((b2 & 0xC0u) != 0x80u) { return 1; }
    if (b0 < 0xF0u) {
        *cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        return 3;
    }
    if (n < 4u) { return 1; }
    uint32_t b3 = p[3];
    if ((b3 & 0xC0u) != 0x80u) { return 1; }
    *cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) | ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
    return 4;
}

/* ---- load ------------------------------------------------------------------------------------------------- */
typedef struct toks_wp_vocab {
    const uint8_t *const *str; /* [n] utf-8 strings (any order) */
    const uint32_t *len;
    const uint32_t *id;
    uint32_t n;
} toks_wp_vocab;

typedef struct toks_wp_params {
    uint32_t flags;                     /* TOKS_WPF_* */
    uint32_t max_chars;
    const uint8_t *unk; uint32_t unk_len;
    const uint8_t *prefix; uint32_t prefix_len;
} toks_wp_params;

/* arena bytes toks_wp_build needs at most. */
uint64_t toks_wp_tables_bytes(const toks_wp_vocab *v);
/* the tries of v under p into tables t already built by toks_wp_build: *bytes = the arena bytes they take (ar NULL:
 * only that), else placed in ar. 0 or TOKS_E_NOMEM / TOKS_E_LIMIT. */
int64_t toks_wp_tries(toks_wp_tables *t, toks_arena *ar, const toks_wp_vocab *v, const toks_wp_params *p,
                      uint64_t *bytes);

/* rationale: docs/notes/c-core.md §wp.h.3 */
int64_t toks_wp_build(toks_wp_tables *t, toks_arena *ar, const toks_wp_vocab *v, const toks_wp_params *p,
                      toks_err *err);

/* ---- the driver's hookup (config.c reads the file; load.c calls toks_wp_ctx_build) ----------------------- */
#define TOKS_WP_MAX_CHARS   1024u   /* max_input_chars_per_word toks runs: the scan's copy buffer is sized by it */
#define TOKS_WP_DEC_PREFIX  16u     /* bytes of the WordPiece decoder's prefix */

struct toks_config;
/* the model tables of cfg in a new toks_plat_arena block (*mem, *mem_len: the caller frees it); *out in it */
int64_t toks_wp_ctx_build(const struct toks_config *cfg, toks_arena *par, uint8_t **mem, uint64_t *mem_len,
                          const toks_wp_tables **out, toks_err *err);

/* rationale: docs/notes/c-core.md §wp.h.4 */
static inline uint64_t wp_fold(uint64_t w)
{
    uint64_t x = w & 0x7F7F7F7F7F7F7F7Full;
    uint64_t m = (x + 0x3F3F3F3F3F3F3F3Full) & ~(x + 0x2525252525252525ull) & ~w & 0x8080808080808080ull;
    return w | (m >> 2);                        /* A-Z (and only A-Z) gain 0x20 */
}

static inline uint64_t wp_load(const uint8_t *p, uint64_t n)    /* n in 1..8 bytes, zero-padded */
{
    uint64_t w = 0;
    if (n >= 8u) {
        memcpy(&w, p, 8);
        return w;
    }
    for (uint64_t k = 0; k < n; k++) {          /* bound: 7 */
        w |= (uint64_t)p[k] << (8u * k);
    }
    return w;
}

static inline uint32_t wp_hash_end(uint32_t h, uint64_t len)
{
    h = toks_crc32c_u32(h, (uint32_t)len);
    h ^= h >> 16;                               /* fmix32: crc32c is linear in the key bits; the probe index */
    h *= 0x85EBCA6Bu;                           /* must not be (keys differing in one byte would share a few */
    h ^= h >> 13;                               /* slots and lengthen every probe through them) */
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}

static inline uint32_t wp_hash(const uint8_t *p, uint64_t len, int fold)
{
    uint32_t h = TOKS_WP_HSEED;
    uint64_t i = 0;
    while (len - i >= 8u) {                     /* bound: len / 8 */
        uint64_t w = wp_load(p + i, 8);
        h = toks_crc32c_u64(h, fold ? wp_fold(w) : w);
        i += 8u;
    }
    if (i < len) {
        uint64_t w = wp_load(p + i, len - i);
        h = toks_crc32c_u64(h, fold ? wp_fold(w) : w);
    }
    return wp_hash_end(h, len);
}

#endif /* TOKS_WP_H */
