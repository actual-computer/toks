/* spm.h: sentencepiece-style bpe (docs/notes/c-core.md §spm.h.1) */
#ifndef TOKS_SPM_H
#define TOKS_SPM_H

#include "core.h"
#include "json.h"

#define TOKS_SPM_NONE          0xFFFFFFFFu   /* no id */
#define TOKS_SPM_MAX_OPS       8u            /* normalizer / decoder steps */
#define TOKS_SPM_MAX_STR       16u           /* bytes of a pattern / content / prepend / replacement */

/* sflags */
#define TOKS_SPM_BYTE_FALLBACK 0x01u
#define TOKS_SPM_FUSE_UNK      0x02u
#define TOKS_SPM_UNK_ERROR     0x04u         /* unk_token set but not a model-vocab string: needing it fails */
#define TOKS_SPM_IGNORE_MERGES 0x08u

/* normalizer steps (doc §3) */
enum { TOKS_SPM_N_PREPEND = 1, TOKS_SPM_N_REPLACE = 2 };
/* decoder steps (doc §8) */
enum { TOKS_SPM_D_REPLACE = 1, TOKS_SPM_D_BYTE_FALLBACK = 2, TOKS_SPM_D_FUSE = 3, TOKS_SPM_D_STRIP = 4,
       TOKS_SPM_D_METASPACE = 5 };
/* Metaspace prepend schemes */
enum { TOKS_SPM_PS_ALWAYS = 0, TOKS_SPM_PS_FIRST = 1, TOKS_SPM_PS_NEVER = 2 };

typedef struct toks_spm_str {                /* a short literal (utf-8) */
    uint8_t  b[TOKS_SPM_MAX_STR];
    uint32_t n;
} toks_spm_str;

typedef struct toks_spm_op {                 /* one normalizer or decoder step */
    uint32_t     kind;
    uint32_t     start, stop;                /* Strip */
    uint32_t     scheme;                     /* Metaspace (decoder): TOKS_SPM_PS_* */
    toks_spm_str a, b;                       /* Prepend: a. Replace: a -> b. Strip: a (one char). Metaspace: a */
} toks_spm_op;

/* the text model as written in the file: normalizer steps, then at most one Metaspace (a Split
 * pre-tokenizer is accepted only where it provably never cuts, doc §1.1 / §11). */
typedef struct toks_spm_text {
    toks_spm_op  norm[TOKS_SPM_MAX_OPS];
    uint32_t     n_norm;
    uint32_t     metaspace;                  /* 0: no Metaspace pre-tokenizer: a split is one piece */
    uint32_t     ms_scheme;                  /* TOKS_SPM_PS_* */
    uint32_t     ms_split;                   /* 1: cut before every replacement char */
    toks_spm_str ms_repl;                    /* the replacement char */
} toks_spm_text;

typedef struct toks_spm_config {
    /* model vocab by id: raw utf-8; ids may have holes (len 0, str NULL); n_ids = max id + 1 */
    const uint8_t *const *vocab;
    const uint32_t       *vocab_len;
    uint32_t              n_ids;
    uint32_t              n_strings;         /* hf's vocab.len(): where added-only ids start */
    const uint32_t       *vslot;             /* the string index: open addressing, slot = id + 1 */
    uint64_t              vmask;
    /* merges in rank order (legacy "#version" lines dropped), as ids; duplicates kept (last wins later) */
    const uint32_t       *m_left, *m_right, *m_out;
    uint32_t              n_merges;
    uint32_t              unk_id;            /* TOKS_SPM_NONE when unk_token is null or not in the vocab */
    uint32_t              sflags;            /* TOKS_SPM_* */
    toks_spm_text         text;
    toks_spm_op           dec[TOKS_SPM_MAX_OPS];
    uint32_t              n_dec;
    uint32_t              has_decoder;       /* 0: hf joins the strings with " " */
} toks_spm_config;

/* rationale: docs/notes/c-core.md §spm.h.2 */
#define TOKS_SPM_E_ID          0x001FFFFFu
#define TOKS_SPM_E_NOID        0x001FFFFFu
#define TOKS_SPM_E_SI_SHIFT    21u
#define TOKS_SPM_E_SI_NONE     255u
#define TOKS_SPM_E_PAIRED      0x40000000u
#define TOKS_SPM_E_PLAIN       0x80000000u   /* K7's fast pair test: an id, not small, not the Metaspace split image */
#define TOKS_SPM_SMALL         129u          /* small alphabet: ASCII + U+2581 */

/* where the virtual prefix symbol goes (doc §6.3) */
enum { TOKS_SPM_PFX_NONE = 0,                /* never */
       TOKS_SPM_PFX_GAP = 1,                 /* normalizer Prepend: every non-empty gap */
       TOKS_SPM_PFX_ALWAYS = 2,              /* Metaspace always: a unit whose first image is not the repl */
       TOKS_SPM_PFX_FIRST = 3 };             /* Metaspace first: the same, for the unit at input offset 0 */

typedef struct toks_spm {
    uint32_t        ascii[128];              /* the entries of code points 0..127 */
    const uint16_t *stage1;                  /* [0x1100]: block of cp >> 8 (block 0: code points 0..255) */
    const uint32_t *stage2;                  /* [n_blocks * 256] entries */
    const uint64_t *pairs;                   /* the spanned-pair set, keys id_x << 21 | id_y (empty = all ones), load
                                                <= 1/4, then one more slot = slot 0 (K7 reads two at once); NULL: no
                                                certified cut (ignore_merges) */
    uint64_t        pairs_mask;
    uint64_t        n_pairs;
    const uint32_t *holes;                   /* n_ids bits: ids with no string at all (decode skips them); NULL
                                                when there are none */
    uint8_t         cut[(TOKS_SPM_SMALL * TOKS_SPM_SMALL + 7u) / 8u];   /* bit x * 129 + y: a cut between x y */
    uint32_t        unk_id;
    uint32_t        sflags;
    uint32_t        pfx_mode;                /* TOKS_SPM_PFX_* */
    uint32_t        pfx_entry;               /* the prefix char's entry */
    uint32_t        id_repl;                 /* Metaspace: the replacement char's id (TOKS_SPM_NONE: none) */
    uint32_t        ms_split;                /* Metaspace split: a piece starts at every image == repl */
    uint32_t        n_blocks;
    uint32_t        n_dec;
    uint32_t        has_decoder;
    uint32_t        rsv;
    uint64_t        n_dropped;               /* merges dropped as unreachable (doc §6.5) */
    toks_spm_op     dec[TOKS_SPM_MAX_OPS];
    uint32_t        cut_ab[1024];            /* bit x | y << 8 for input bytes x, y < 128, x right before y: a word
                                                starts at y (Metaspace split or a §5.5 cut: the scan's whole decision
                                                for an ASCII pair); all zero without a pair set */
    const uint8_t  *cut_ab8;                 /* [65536]: byte x | y << 8 = cut_ab's bit (0 when x or y >= 128): K7's
                                                ascii pairs, one load from two text bytes; NULL without a pair set */
} toks_spm;
/* K7, the scan (layout.h, kernels.md §8; test_k7.c checks the SPM_* offsets): the c twin (spm_c.c), the asm parts */
uint64_t toks_k7_spm_c(const toks_spm *s, toks_k7_args *a);
uint64_t toks_k7_spm_neon(const toks_spm *s, toks_k7_args *a);
uint64_t toks_k7_spm_avx2(const toks_spm *s, toks_k7_args *a);

/* §5.5's pair set: is id_x << 21 | id_y spanned? One exit test per slot, the answer computed, not branched on */
static inline int toks_spm_spanned(const toks_spm *s, uint64_t key)
{
    uint64_t mask = s->pairs_mask, i = (key * TOKS_FIB64 >> 32) & mask;
    for (uint64_t k = 0; k <= mask; k++) {                  /* bound: the slots (load <= 0.5 ends it sooner) */
        uint64_t v = s->pairs[i];
        if (v == key || v == UINT64_MAX) { return v == key; }
        i = (i + 1u) & mask;
    }
    return 0;
}

/* §5.5 on two entries: a certified cut between adjacent images a | b (both vocab chars, no reachable string
 * spans them). Needs the pair set (s->pairs != NULL). */
static inline int toks_spm_cut_between(const toks_spm *s, uint32_t a, uint32_t b)
{
    uint32_t ia = a & TOKS_SPM_E_ID, ib = b & TOKS_SPM_E_ID;
    if (ia == TOKS_SPM_E_NOID || ib == TOKS_SPM_E_NOID) { return 0; }
    uint32_t xa = (a >> TOKS_SPM_E_SI_SHIFT) & 0xFFu, xb = (b >> TOKS_SPM_E_SI_SHIFT) & 0xFFu;
    if (xa != TOKS_SPM_E_SI_NONE && xb != TOKS_SPM_E_SI_NONE) {   /* ASCII and U+2581: one bit */
        uint32_t bit = xa * TOKS_SPM_SMALL + xb;
        return (s->cut[bit >> 3] >> (bit & 7u)) & 1u;
    }
    if (!(a & TOKS_SPM_E_PAIRED)) { return 1; }
    return !toks_spm_spanned(s, ((uint64_t)ia << 21) | ib);
}

/* rationale: docs/notes/c-core.md §spm.h.3 */
int64_t toks_spm_build(toks_tables *t, const toks_spm_config *cfg, uint8_t **mem, uint64_t *mem_len,
                       const toks_spm **out, toks_err *err);

/* work bytes for a word of up to n symbols (sym, next, prev: 12 B each; a heap of <= 2n entries of 8 B;
 * the result ids: 4 B) plus alignment: within the driver's TOKS_BPE_WORK_BYTES(n) */
#define TOKS_SPM_WORK_BYTES(n)  (64u + 32u * (uint64_t)(n))

/* encode flags (tests): whole pieces instead of certified words; no cache; TOKS_SPM_TIER(t): K7's tier (0: the twin) */
#define TOKS_SPM_NOCUTS        1u
#define TOKS_SPM_NOCACHE       2u
#define TOKS_SPM_NOPFX         4u            /* a unit after a phase-1 token inside its gap: no Prepend (segment.c) */
#define TOKS_SPM_TIER(t)       ((uint32_t)(t) << 8)

/* a word key's hash (layout.h key words lo, hi): the word cache's bucket and the static word table (one multiply) */
static inline uint32_t toks_spm_whash(uint64_t lo, uint64_t hi)
{
    return (uint32_t)(((lo ^ (hi * TOKS_FIB64)) * 0xD6E8FEB86659FD93ull) >> 32);
}

/* rationale: docs/notes/c-core.md §spm.h.4 */
uint64_t toks_spm_encode(const toks_tables *t, const toks_spm *s, const uint8_t *text, uint64_t len, int at_start,
                         uint32_t *out, uint64_t cap, uint64_t n, uint8_t *cache, uint64_t cache_mask, uint64_t tag,
                         uint8_t *work, uint32_t flags);

/* rationale: docs/notes/c-core.md §spm.h.5 */
int toks_spm_cut(const toks_spm *s, const uint8_t *text, uint64_t len, uint64_t a, uint64_t c);

/* the pre-tokenizer pieces of one text unit (Metaspace split points, else the unit) as end offsets
 * base + end into out[k] for n <= k < cap; returns the new count. */
uint64_t toks_spm_pieces(const toks_spm *s, const uint8_t *text, uint64_t len, int at_start, uint64_t base,
                         uint32_t *out, uint64_t cap, uint64_t n);

/* the model alone on a piece of image chars p[0, len) (hf Model::tokenize; tests): ids to out (room
 * len + 1), returns the count. work: TOKS_SPM_WORK_BYTES(len + 1) + 8 bytes. */
uint64_t toks_spm_model(const toks_tables *t, const toks_spm *s, const uint8_t *p, uint64_t len, uint32_t *out,
                        uint8_t *work);

/* hf decode (doc §8), streaming: skip special ids (bit set in special) under TOKS_SKIP_SPECIAL; TOKS_DECODE_RAW writes
 * an invalid byte run's bytes where decode writes its U+FFFD (toks.h); ids >= n_ids are the caller's to refuse. Returns
 * the byte count (bytes beyond cap are counted, not written). */
int64_t toks_spm_decode(const toks_tables *t, const toks_spm *s, const uint32_t *special, uint32_t flags,
                        const uint32_t *ids, uint64_t n, uint8_t *out, uint64_t cap);

#endif /* TOKS_SPM_H */
