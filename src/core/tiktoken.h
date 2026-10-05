/* tiktoken.h: tiktoken-format models -> toks_config (docs/notes/c-core.md §tiktoken.h.1) */
#ifndef TOKS_TIKTOKEN_H
#define TOKS_TIKTOKEN_H

#include "core.h"

#define TOKS_TIKTOKEN_RESERVED   256u       /* the kimi wrapper's num_reserved_special_tokens */

typedef struct toks_tiktoken_info {
    uint32_t n_ranks;              /* tokens in the ranks file */
    uint32_t first_special;        /* the wrapper's first special id: len(ranks) */
    uint32_t chunk_chars;          /* 400,000: the wrapper encodes text[i : i + chunk_chars] chunk by chunk */
    uint32_t run_chars;            /* 25,000: a run of same-str.isspace() chars is cut after this many */
    uint32_t n_trie;               /* specials transformers' trie splits on first (the serving path) */
    uint32_t n_named;              /* specials decode(skip_special_tokens=True) drops (cfg->added .special) */
    uint8_t  trie[TOKS_TIKTOKEN_RESERVED / 8u];   /* bit i: special first_special + i is a trie token */
} toks_tiktoken_info;

/* parse-arena bytes that always suffice for toks_tiktoken_parse over files of these sizes. */
uint64_t toks_tiktoken_arena_bound(uint64_t ranks_len, uint64_t config_len, uint64_t wrapper_len);

/* rationale: docs/notes/c-core.md §tiktoken.h.2 */
int64_t toks_tiktoken_ranks(const uint8_t *data, uint64_t len, toks_arena *ar, toks_config *cfg, toks_err *err);

/* after toks_tiktoken_ranks on cfg: the kimi wrapper (config + code) -> cfg->added (the 256 specials, file
 * order = id order), cfg->n_ids, and *info. Same returns. */
int64_t toks_tiktoken_kimi(const uint8_t *config, uint64_t config_len, const uint8_t *wrapper, uint64_t wrapper_len,
                           toks_arena *ar, toks_config *cfg, toks_tiktoken_info *info, toks_err *err);

/* both, then the pre-tokenizer (see the header comment): cfg ready for toks_compile. Same returns. */
int64_t toks_tiktoken_parse(const uint8_t *ranks, uint64_t ranks_len, const uint8_t *config, uint64_t config_len,
                            const uint8_t *wrapper, uint64_t wrapper_len, toks_arena *ar, toks_config *cfg,
                            toks_tiktoken_info *info, toks_err *err);

/* 1 when data looks like a ranks file rather than json: its first byte is a base64 character. */
int toks_tiktoken_sniff(const uint8_t *data, uint64_t len);

#endif /* TOKS_TIKTOKEN_H */
