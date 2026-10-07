/*
 * tests/proof/entries.c: the eva entry points (SPEC §14.1, docs/proof.md). Each eva_<name> establishes one
 * entry's preconditions (SPEC §4.1, the §8.3 limits) for every input inside them -- unknown contents
 * (Frama_C_make_unknown), every length up to the limit -- and calls the entry once.
 */
#include "prelude.h"
#include "core.h"

/* ---- input buffers ---------------------------------------------------------------------------------------
 * A source of len bytes, len in [1, 256 MiB] (SPEC §8.3), unknown contents, placed inside one static
 * block of the maximal size as EVA_PLACE says (tests/proof/eva.sh runs both placements):
 *   EVA_PLACE == 0  flush end:   [blk + MAX - len, blk + MAX): end is the block's end, so a read at or past
 *                                the source's end is outside the block -- an alarm whatever len is
 *   EVA_PLACE == 1  flush start: [blk, blk + len): a read before the source's start is outside the block
 * Together they bound every read of the source to [data, data + len); each alone checks one side (the other
 * side's stray reads would stay inside the block). */
#ifndef EVA_PLACE
#define EVA_PLACE 0
#endif

static uint8_t proof_src_blk[PROOF_SRC_MAX];

static uint8_t *proof_src(uint64_t *len_out)
{
    uint64_t len = Frama_C_unsigned_long_long_interval(1u, PROOF_SRC_MAX);
    Frama_C_make_unknown((char *)proof_src_blk, (size_t)PROOF_SRC_MAX);
    *len_out = len;
    return EVA_PLACE == 0 ? proof_src_blk + (PROOF_SRC_MAX - len) : proof_src_blk;
}

/* ---- readers --------------------------------------------------------------------------------------- */

/* toks_json_parse over a source and the parse arena load.c gives it (toks_config_arena_bound). */
void eva_json(void)
{
    uint64_t len;
    const uint8_t *data = proof_src(&len);
    uint64_t alen = toks_config_arena_bound(len);
    toks_arena ar = { NULL, alen, 0u };       /* the model allocator never touches base (prelude.h) */
    jv *root = NULL;
    int64_t r = toks_json_parse(data, len, &ar, &root);
    if (r == 0) {
        jv *m = toks_jv_get(root, "model");
        (void)m;
    }
}

/* toks_config_parse: the json reader plus config.c over a source (load.c's call). */
void eva_config(void)
{
    uint64_t len;
    const uint8_t *data = proof_src(&len);
    uint64_t alen = toks_config_arena_bound(len);
    toks_arena ar = { NULL, alen, 0u };
    toks_config cfg;
    toks_err err;
    int64_t r = toks_config_parse(data, len, &ar, &cfg, &err);
    (void)r;
}
