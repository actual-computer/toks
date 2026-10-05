/* unigram.h: the Unigram family's tables and c twins (docs/notes/c-core.md §unigram.h.1) */
#ifndef TOKS_UNIGRAM_H
#define TOKS_UNIGRAM_H

#include "core.h"
#include "spm.h"

/* ---- grapheme classes (src/gen/grapheme17.c) ------------------------------------------------------------ */

enum {                                  /* the crate's GraphemeCat, low 4 bits of a class byte */
    TOKS_GC_ANY = 0, TOKS_GC_CR, TOKS_GC_CONTROL, TOKS_GC_EXTEND, TOKS_GC_EXTPICT, TOKS_GC_INCB_CONSONANT,
    TOKS_GC_L, TOKS_GC_LF, TOKS_GC_LV, TOKS_GC_LVT, TOKS_GC_PREPEND, TOKS_GC_RI, TOKS_GC_SPACINGMARK,
    TOKS_GC_T, TOKS_GC_V, TOKS_GC_ZWJ
};
#define TOKS_GC_CAT_MASK     0x0Fu
#define TOKS_GC_INCB_EXTEND  0x10u      /* InCB=Extend */
#define TOKS_GC_INCB_LINKER  0x20u      /* InCB=Linker */
#define TOKS_GC_INVALID      0x40u      /* an invalid atom (SPEC §3.3): its own grapheme, like a control */

extern const uint16_t TOKS_GC_STAGE1[0x1100];   /* cp >> 8 -> block */
extern const uint8_t  TOKS_GC_STAGE2[];         /* block * 256 + (cp & 0xFF) -> class */

static inline uint8_t toks_gc_class(uint32_t cp)
{
    return TOKS_GC_STAGE2[(uint32_t)TOKS_GC_STAGE1[cp >> 8] * 256u + (cp & 0xFFu)];
}

/* the boundary automaton: state after the chars consumed so far (all fields fit in one word) */
typedef struct toks_gc_state {
    uint8_t prev;        /* class of the previous atom; 0xFF at the start of the text */
    uint8_t ri_odd;      /* an odd number of regional indicators ends the text so far */
    uint8_t ep_run;      /* the text ends with ExtPict Extend* */
    uint8_t ep_zwj;      /* the text ends with ExtPict Extend* ZWJ */
    uint8_t incb_cons;   /* the text ends with InCB-Consonant [InCB-Linker InCB-Extend]* */
    uint8_t incb_link;   /* ... and that run holds a linker */
    uint8_t rsv[2];
} toks_gc_state;

static inline void toks_gc_init(toks_gc_state *s) { memset(s, 0, sizeof *s); s->prev = 0xFFu; }

/* 1 when a grapheme boundary lies before an atom of class cls (unigram.md §3.3: GB3-GB13, GB999); then
 * consumes it. */
int toks_gc_step(toks_gc_state *s, uint8_t cls);

/* ---- the compiled charsmap ------------------------------------------------------------------------------ */

/* entry of a key: kind in the top 2 bits, then the value's offset (18 bits) and length (12 bits) in pool */
#define TOKS_PC_KIND(e)      ((e) >> 30)
#define TOKS_PC_OFF(e)       (((e) >> 12) & 0x3FFFFu)
#define TOKS_PC_LEN(e)       ((e) & 0xFFFu)
enum { TOKS_PC_IDENT = 0, TOKS_PC_VALUE = 1 };   /* entry 0: not a key (the char maps to itself) */

typedef struct toks_pc {
    uint16_t       *stage1;       /* cp >> 8 -> block of stage2 (0: the all-identity block) */
    uint32_t       *stage2;       /* block * 256 + (cp & 0xFF) -> entry (0 = identity) */
    uint32_t        n_blocks;
    uint64_t       *mk_key;       /* live multi-char keys: bytes packed little-endian, length in bits 56-63 */
    uint32_t       *mk_val;       /* their entries */
    uint32_t        mk_mask;      /* open addressing, power-of-two size - 1; key 0 = empty */
    uint8_t        *pool;         /* replacement strings */
    uint32_t        pool_len;
    uint32_t        max_expand;   /* ceil(max value bytes / key bytes): the materialization bound per byte */
    uint32_t        n_keys;       /* keys in the double array */
    uint32_t        n_live_single, n_live_multi;
} toks_pc;

/* bytes of arena toks_pc_build needs for this blob (0 when the blob is refused; *why names the reason) */
uint64_t toks_pc_bytes(const uint8_t *blob, uint64_t len, const char **why);
/* compiles the raw (base64-decoded) charsmap; 0 or TOKS_E_UNSUPPORTED with *why set */
int64_t  toks_pc_build(toks_pc *pc, const uint8_t *blob, uint64_t len, toks_arena *ar, const char **why);

/* entry for one char (0 = identity) */
static inline uint32_t toks_pc_char(const toks_pc *pc, uint32_t cp)
{
    if (cp > 0x10FFFFu) { return 0; }
    return pc->stage2[(uint32_t)pc->stage1[cp >> 8] * 256u + (cp & 0xFFu)];
}
/* entry of a live multi-char key equal to key[0, n) (2 <= chars, n <= 5), else 0 */
uint32_t toks_pc_multi(const toks_pc *pc, const uint8_t *key, uint32_t n);

/* ---- the model ------------------------------------------------------------------------------------------- */

enum {                                   /* decoder chains of the census (unigram.md §8) */
    TOKS_UNI_DEC_NONE = 0,               /* no decoder: tokens joined with ' ' */
    TOKS_UNI_DEC_META = 1,               /* Metaspace */
    TOKS_UNI_DEC_RBF  = 2,               /* Replace('▁' -> ' ') > ByteFallback > Fuse (ruri) */
    TOKS_UNI_DEC_BFRF = 3                /* ByteFallback > Replace('▁' -> ' ') > Fuse > Replace('(?<!\n)^ ' -> '') */
};

typedef struct toks_uni_cfg {
    /* normalizer chain in hf's order, each optional: Replace(String)s, NF*, StripAccents, Lowercase (the last three:
     * cfg->nfc's form), Strip, Precompiled, Replace(' {2,}' -> ' '), Replace('(?<!\n)^' -> '▁'), Replace(' ' -> '▁') */
    uint8_t  strip_left, strip_right;
    uint8_t  has_charsmap;
    uint8_t  collapse;
    uint8_t  meta_prefix;
    uint8_t  meta_replace;
    /* pre-tokenizers: [WhitespaceSplit] [Metaspace] */
    uint8_t  ws_split;
    uint8_t  metaspace;
    uint8_t  meta_prepend;    /* prepend_scheme always (never = 0; first is refused) */
    uint8_t  meta_split;
    uint8_t  dec;             /* TOKS_UNI_DEC_* */
    uint8_t  dec_prepend;     /* the Metaspace decoder's scheme is not never */
    uint8_t  rep_n;           /* Replace(String pattern -> content no longer) before everything, in order (albert) */
    uint8_t  rep_pl[4], rep_cl[4], rep_p[4][4], rep_c[4][4];
    uint8_t  form;            /* NF*, StripAccents, Lowercase: the engine's steps (norm.h), 0: none */
    uint8_t  rsv[2];
} toks_uni_cfg;

/* a double-array trie cell: child = (base & TOKS_UNI_BASE) + byte, valid iff check[child] == node; base's top bit
 * set: a piece ends at this node (its score and id in the node-indexed arrays). One 8-byte load per trie step. */
#define TOKS_UNI_TERM 0x80000000u
#define TOKS_UNI_BASE 0x7FFFFFFFu
typedef struct toks_uni_cell {
    uint32_t base;
    int32_t  check;
} toks_uni_cell;

typedef struct toks_uni {
    toks_uni_cfg cfg;
    toks_pc      pc;          /* when cfg.has_charsmap */
    /* double-array trie over the remapped pieces (root = 0, laid out depth first: a path's cells sit close);
     * term[node] = the piece id ending there (hf's token_to_ids: the last duplicate), or -1 */
    const toks_uni_cell *cell;
    const double  *score;     /* per node: the score of the piece ending there (hf's f64, serde_json's parse) */
    const int32_t *term;
    uint32_t     da_len;
    uint32_t     n_vocab;
    double       unk_score;   /* min over every score - 10.0 */
    /* the static piece table (bpe.h bucket format, 2 ways x 2 buckets): a piece's canonical key (its remapped bytes
     * <= 15 with a leading 0x20 as the virtual bit, unigram.md §10.3) -> its model ids (<= 4), most probable pieces
     * first up to load 0.85; NULL: none */
    const uint8_t *words;
    uint64_t     words_mask;
    uint8_t      simple[128]; /* the walk's fast path: 1 = an ascii char that is its own grapheme, maps to itself and
                                 only extends the open piece (unigram.md §10.3) */
    uint32_t     aent[128];   /* the charsmap entry of each ascii char (0: itself) */
    uint8_t      acls[128];   /* the walk's ascii fast path (fast): 1 a simple char, 2 a space (U+0020 or a char the
                                 charsmap maps to exactly " "), 0 any other */
    uint8_t      fast;        /* charsmap + Metaspace(always, split) and nothing that writes U+2581 (census rows 1-7) */
    uint8_t      rsv2[7];
    uint32_t     unk_id;      /* present (a model without unk_id is refused) */
    uint32_t     max_piece;   /* longest remapped piece in bytes (<= 127) */
    int32_t      byte_id[256];/* <0xXX> ids for byte fallback, -1 when missing */
    uint8_t      byte_fallback;
    uint8_t      remap;       /* U+2581 in pieces is the byte 0x20 (the chain leaves no other 0x20) */
    uint8_t      rsv[2];
    uint32_t     work_x;      /* bytes of normalized text per input byte, worst case (charsmap expansion) */
    uint32_t     pre_x;       /* the same for the steps before (cfg.form's TOKS_NORM_X, 1: none) */
    uint32_t     n_dropped;   /* pieces holding a raw 0x20: unreachable under remap, kept out of the trie */
    uint32_t     n_dec;       /* the decoder chain as spm.h ops (stream.c runs it, unigram.md §8) */
    toks_spm_op  dec[4];
} toks_uni;

typedef struct toks_uni_src {
    uint32_t              n;          /* vocab size */
    const uint8_t *const *piece;      /* piece bytes (utf-8, distinct) */
    const uint32_t       *piece_len;
    const uint8_t *const *score_txt;  /* json number lexeme of each score */
    const uint32_t       *score_len;
    int64_t               unk_id;     /* -1: none (refused) */
    uint32_t              byte_fallback;
    const uint8_t        *charsmap;   /* decoded blob, or NULL */
    uint64_t              charsmap_len;
    toks_uni_cfg          cfg;
} toks_uni_src;

/* builds the model's tables in one toks_plat_arena block (*mem, *mem_len; *u points into it).
 * 0, TOKS_E_UNSUPPORTED (*why names the feature) or TOKS_E_NOMEM. */
int64_t toks_uni_build(const toks_uni_src *src, const toks_uni **u, uint8_t **mem, uint64_t *mem_len,
                       const char **why);

/* one work area for a text of len bytes (its layout and bound: docs/notes/c-core.md §unigram.h.2) */
uint64_t toks_uni_area(const toks_uni *u, uint64_t len);

/* rationale: docs/notes/c-core.md §unigram.h.3 */
int64_t toks_uni_normalize(const toks_uni *u, const uint8_t *text, uint64_t len, int gap_start, uint8_t *dst,
                           uint64_t cap);

/* rationale: docs/notes/c-core.md §unigram.h.4 */
typedef struct toks_uni_call {
    toks_emit e;
    uint64_t  nbase;
    int       pieces;
    int       rsv;
    uint8_t  *work;
    uint64_t  work_bytes;
    uint8_t  *cache;          /* the scratch's piece cache (kernels.md §6-7 format, epoch-tagged), NULL: none */
    uint64_t  cache_mask;     /* buckets - 1 */
    uint64_t  tw;             /* the epoch's tag word (layout.h toks_tag_word) */
} toks_uni_call;

/* rationale: docs/notes/c-core.md §unigram.h.5 */
int64_t toks_uni_encode_segment(const toks_uni *u, const uint8_t *text, uint64_t len, int pre_normalized,
                                int gap_start, toks_uni_call *c);

#endif /* TOKS_UNIGRAM_H */
