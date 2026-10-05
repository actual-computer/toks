/* rationale: docs/notes/c-core.md §config.h.1 */
#ifndef TOKS_CONFIG_H
#define TOKS_CONFIG_H

#include "json.h"

/* --- template params (TOKS_TP_*) + pattern table ------------------------------------------------------- */

typedef struct toks_pattern {
    const char *regex;       /* static: the exact pattern string (of a Split chain: its last Split's) */
    uint32_t    params;      /* TOKS_TP_* */
    uint32_t    tmpl;        /* TOKS_TMPL_CL100K, TOKS_TMPL_O200K or TOKS_TMPL_DSV3 */
    uint32_t    class_flags; /* TOKS_CLASSES_* (classes.h) its class tables are built with */
} toks_pattern;

/* known patterns, exact-string matched: cl100k family (kernels.md §3) [0] = gpt-2, [3] = the marks-folded
 * qwen 3.5 variant; o200k family (docs/templates/o200k.md) [4] = o200k (gpt-oss, gpt-4o), [5] = nemo. */
extern const toks_pattern TOKS_PATTERNS[];
extern const uint32_t     TOKS_PATTERNS_N;

/* the o200k template's kimi variant (docs/templates/o200k.md §6, docs/models/kimi.md §5): tiktoken input only
 * (src/core/tiktoken.c names it), never matched against a tokenizer.json regex. */
extern const toks_pattern TOKS_PATTERN_KIMI;

/* digits-gpt2: [Digits(individual_digits), ByteLevel(use_regex)] (SmolLM; kernels.md §3 A8) */
extern const toks_pattern TOKS_PATTERN_DIGITS;
extern const toks_pattern TOKS_PATTERN_P16;
extern const toks_pattern TOKS_PATTERN_P20;

/* rationale: docs/notes/c-core.md §config.h.2 */
extern const char *const  TOKS_DSV3_SPLITS[3];
extern const toks_pattern TOKS_PATTERN_DSV3;

/* --- added tokens --------------------------------------------------------------------------------------- */

/* rationale: docs/notes/c-core.md §config.h.3 */
typedef struct toks_cfg_added {
    const uint8_t *content;
    uint32_t       len;        /* 1 .. TOKS_MAX_ADDED_BYTES */
    uint32_t       id;
    uint8_t        special;    /* 0/1 */
    uint8_t        normalized; /* 0/1: 1 = matched on the normalized text (phase 1; with cuts: see cut_run) */
    uint8_t        lstrip;     /* 0/1: hf's lstrip / rstrip / single_word (segment.c applies them) */
    uint8_t        rstrip;
    uint8_t        single_word;
    uint8_t        pfx;        /* 1: phase 1 under Prepend ▁ + Replace ' ' -> ▁ (spm): ▁ + content (segment.c) */
    uint8_t        rsv[2];
    const uint8_t *form;       /* wordpiece: what phase 1 matches -- the content run through the file's
                                  normalizer when normalized (hf's normalized_cache), else the content */
    uint32_t       form_len;
    uint32_t       rsv2;
} toks_cfg_added;

/* --- post-processor pieces ------------------------------------------------------------------------------ */

enum { TOKS_PPS_SEQ = 0, TOKS_PPS_TOK = 1 };

typedef struct toks_pp_piece {
    uint32_t kind;      /* TOKS_PPS_* */
    uint32_t id;        /* TOKS_PPS_TOK: one id of a template special token (its ids, expanded) */
} toks_pp_piece;

/* --- the config ----------------------------------------------------------------------------------------- */

typedef struct toks_config {
    /* rationale: docs/notes/c-core.md §config.h.4 */
    const uint8_t *const *vocab;    /* [n_vocab] pointers into parse memory */
    const uint32_t       *vocab_len;
    uint32_t              n_vocab;
    uint32_t              n_vocab_raw;
    /* byte-level: the bytes whose alphabet char the vocab lacks (bit b & 7 of drop[b >> 3]); has_drop when any.
     * hf drops them inside the model (docs/breadth.md §4); config.c accepts that only where it is exact. */
    uint8_t               drop[32];
    uint32_t              has_drop;
    uint32_t              drop_unk, drop_fuse;   /* with unk_token: its id + 1 (an unk per dropped byte, or per run under
                                                    fuse_unk), else 0: hf skips the byte */

    /* merges in raw rank order (duplicates kept; the bpe builder applies hf's last-wins): the ids of
     * each merge's left, right and merged token. */
    const uint32_t       *m_left_id;
    const uint32_t       *m_right_id;
    const uint32_t       *m_out_id;
    uint32_t              n_merges;

    uint8_t   ignore_merges;
    uint8_t   dec_byte_level;      /* the decoder is ByteLevel (m1a refuses every other decoder) */
    uint8_t   nfc;                 /* the normalizer's form (norm.h, 0: none): byte-level NFC / NFKC on every text gap
                                      before phase 1 and K3 (api.c run_gap: unchanged text in place, changed stretches
                                      into scratch); unigram's steps before its charsmap (uni_api.c) */
    uint8_t   ids_as_rank;         /* 1: a merge's priority is its merged id, whatever the merge order
                                    * (tiktoken: every split of a token has the token's rank; the
                                    * tiktoken reader sets it). 0: hf's raw merge rank (bpe_build.c
                                    * still picks ids-as-rank when the merged ids rise strictly). */
    uint8_t   rsv;

    const toks_pattern *pattern;   /* NULL: no split (ByteLevel use_regex=false: one piece per segment) */
    const struct toks_gen *gen;    /* else, a chain no template compiles: gen.c's program (parse memory), its size */
    uint64_t gen_bytes;

    /* rationale: docs/notes/c-core.md §config.h.5 */
    uint32_t cut_chunk;
    uint32_t cut_run;

    toks_cfg_added *added;         /* [n_added], parse arena, file order */
    uint32_t        n_added;
    uint32_t        n_ids;         /* max(n_vocab, the largest added id + 1): every id is below it */

    toks_pp_piece *pp_single;      /* the single-sequence template, in order; NULL when none */
    uint32_t       n_pp_single;    /* <= 64; exactly one TOKS_PPS_SEQ piece when > 0 */

    const char *name;              /* static basename of the source, or NULL */

    /* rationale: docs/notes/c-core.md §config.h.6 */
    uint32_t                      algo;
    uint32_t                      n_strings;
    const struct toks_spm_config *spm;

    /* ---- wordpiece lane (docs/algorithms/wordpiece.md): appended ------------------------------------- */
    uint32_t wp_flags;             /* WordPiece: TOKS_WPF_* (BertNormalizer, strip_accents null resolved) */
    uint32_t wp_max_chars;         /* max_input_chars_per_word (<= TOKS_WP_MAX_CHARS) */
    uint32_t wp_unk_id;            /* vocab[unk_token] (config refuses a vocab without it) */
    const uint8_t *wp_unk;         /* unk_token */
    uint32_t wp_unk_len;
    uint32_t wp_prefix_len;        /* continuing_subword_prefix */
    const uint8_t *wp_prefix;
    toks_opts o;                   /* truncation, padding, the WordPiece decoder (core.h) */

    /* ---- unigram (docs/algorithms/unigram.md): appended. uni holds the model, normalizer, pre-tokenizer and
     * decoder; truncation is o.trunc_on / o.trunc_max, as for every algorithm. */
    const struct toks_uni_src *uni;
} toks_config;

/* the readers' string index (config.c; tiktoken.c): open addressing over s[e] / len[e], slot = entry + 1, at most
 * half full; find: the entry whose string is a || b (b may be empty), or -1; add: 0, or 1 when already there */
typedef struct toks_sidx {
    const uint8_t *const *s;
    const uint32_t       *len;
    uint32_t             *slot;
    uint64_t              mask;
} toks_sidx;
int64_t toks_sidx_init(toks_sidx *x, toks_arena *ar, const uint8_t *const *s, const uint32_t *len, uint64_t n);
int64_t toks_sidx_find(const toks_sidx *x, const uint8_t *a, uint32_t al, const uint8_t *b, uint32_t bl);
int     toks_sidx_add(toks_sidx *x, uint32_t e);

/* rationale: docs/notes/c-core.md §config.h.7 */
int64_t toks_config_parse(const uint8_t *data, uint64_t len, toks_arena *ar, toks_config *cfg,
                          toks_err *err);

/* rationale: docs/notes/c-core.md §config.h.8 */
uint64_t toks_gen_max_bytes(void);                         /* gen.c: its largest program, in the parse arena */
static inline uint64_t toks_config_arena_bound(uint64_t len) { return 32u * len + 65536u + toks_gen_max_bytes(); }

/* rationale: docs/notes/c-core.md §config.h.9 */
int64_t toks_alpha_bytes(const uint8_t *s, uint32_t n, uint8_t *out);

/* rationale: docs/notes/c-core.md §config.h.10 */
uint32_t toks_token_bytes(const uint8_t *s, uint32_t n, uint8_t *out);

#endif /* TOKS_CONFIG_H */
