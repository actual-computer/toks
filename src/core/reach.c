/* reach.c: SPEC §3.6, structural control-token isolation: no added token's id comes out of plain text under
 * TOKS_ADDED_NONE, proven per tokenizer at load (docs/algorithms/reach.md). toks_info.control_isolation reports it. */
#include "core.h"
#include "config.h"

/* the last splitting step alone as a generic program (docs/algorithms/reach.md §3), compiled into mem[0, cap); NULL
 * when there is no such step to test pieces against (every text a piece, a step without a proof, no memory) */
static const struct toks_gen *piece_prog(const toks_config *cfg, uint8_t *mem, uint64_t cap)
{
    if ((cfg->reach_kind != TOKS_REACH_SPLIT && cfg->reach_kind != TOKS_REACH_GPT2) || mem == NULL) { return NULL; }
    toks_gen_spec sp[2];
    uint32_t k = 0u;
    memset(sp, 0, sizeof sp);
    if (cfg->reach_bytes) { sp[k++].kind = TOKS_GS_BYTELEVEL; }         /* the split runs on the byte domain */
    if (cfg->reach_kind == TOKS_REACH_GPT2) {
        sp[k++] = (toks_gen_spec){ TOKS_GS_BYTELEVEL, TOKS_GB_ISOLATED, 0u, 1u, NULL, 0u };
    } else {
        sp[k++] = (toks_gen_spec){ TOKS_GS_SPLIT, TOKS_GB_ISOLATED, 0u, cfg->reach_lit, cfg->reach_s, cfg->reach_len };
    }
    toks_arena ar = { mem, cap, 0u };
    const struct toks_gen *g = NULL;
    uint64_t bytes = 0u;
    toks_err err = { 0, NULL };
    return toks_gen_compile(sp, k, &ar, &g, &bytes, &err) == 0 ? g : NULL;
}

/* rationale: docs/algorithms/reach.md §2. 1 when every added token's id is proven never to come out of
 * TOKS_ADDED_NONE's encode of any text; 0 when one may (or the model is not byte-level bpe: not analyzed). */
uint32_t toks_reach_certify(const toks_config *cfg)
{
    if (cfg->algo != TOKS_ALGO_BPE_BYTELEVEL || cfg->spm != NULL || cfg->uni != NULL) { return 0u; }
    uint64_t words = ((uint64_t)cfg->n_vocab + 63u) / 64u + 1u;
    uint64_t *made = (uint64_t *)toks_plat_alloc(8u * words);           /* the ids a merge makes */
    if (made == NULL) { return 0u; }
    memset(made, 0, (size_t)(8u * words));
    for (uint32_t m = 0u; m < cfg->n_merges; m++) {                     /* bound: n_merges */
        uint32_t o = cfg->m_out_id[m];
        if (o < cfg->n_vocab) { made[o >> 6] |= (uint64_t)1u << (o & 63u); }
    }
    uint32_t ok = 1u, whole = (uint32_t)(cfg->ignore_merges || cfg->ids_as_rank);   /* a whole piece in the vocab is
                                                                                       its id (hf ignore_merges,
                                                                                       tiktoken's ranks) */
    uint8_t buf[TOKS_MAX_ADDED_BYTES * 4u + 4u];
    uint64_t cap = toks_gen_max_bytes() + 64u;
    uint8_t *mem = NULL;                                                /* the piece test's program, once */
    const struct toks_gen *g = NULL;
    for (uint32_t i = 0u; ok && i < cfg->n_added; i++) {                /* bound: n_added */
        uint32_t id = cfg->added[i].id;
        if (cfg->drop_unk != 0u && cfg->drop_unk - 1u == id) { ok = 0u; break; }   /* the unk of a dropped byte */
        if (id >= cfg->n_vocab || cfg->vocab_len[id] == 0u) { continue; }   /* not the model's: bpe never emits it */
        int64_t m = toks_alpha_bytes(cfg->vocab[id], cfg->vocab_len[id], NULL);
        if (m < 0) { continue; }                                        /* decode-only: bpe never forms it */
        if (m <= 1 || ((made[id >> 6] >> (id & 63u)) & 1u) != 0u) { ok = 0u; break; }   /* one byte, or merged */
        if (!whole) { continue; }                                       /* only merges make a longer token */
        if (cfg->has_drop || (uint64_t)m > sizeof buf) { ok = 0u; break; }   /* a dropped byte can sit inside a piece */
        if (mem == NULL) {
            mem = (uint8_t *)toks_plat_alloc(cap);
            g = piece_prog(cfg, mem, cap);
        }
        (void)toks_alpha_bytes(cfg->vocab[id], cfg->vocab_len[id], buf);
        if (g == NULL || toks_gen_piece(g, buf, (uint64_t)m)) { ok = 0u; }   /* the piece's own lookup (§3) */
    }
    if (mem != NULL) { toks_plat_free(mem, cap); }
    toks_plat_free(made, 8u * words);
    return ok;
}
