/*
 * tests/common/wp_json.h: a tokenizer.json's WordPiece vocab and parameters through the library's own reader
 * (toks_config_parse), for the tests that build the model tables themselves (tests/c/test_wp.c,
 * tests/wordpiece/check.c). ids are the vocab's (dense, as config.c requires).
 */
#ifndef TOKS_TEST_WP_JSON_H
#define TOKS_TEST_WP_JSON_H

#include "../../src/core/config.h"
#include "../../src/core/wp.h"

static int64_t wp_from_json(const uint8_t *src, uint64_t len, toks_arena *ar, toks_wp_vocab *v, toks_wp_params *p,
                            toks_err *err)
{
    toks_config cfg;
    int64_t r = toks_config_parse(src, len, ar, &cfg, err);
    if (r != 0) { return r; }
    if (cfg.algo != TOKS_ALGO_WORDPIECE) {
        err->code = TOKS_E_UNSUPPORTED;
        err->what = "model type (WordPiece)";
        return TOKS_E_UNSUPPORTED;
    }
    uint32_t *ids = (uint32_t *)toks_ar_alloc(ar, 4u * (uint64_t)cfg.n_vocab + 8u, 8u);
    if (ids == NULL) { return TOKS_E_NOMEM; }
    for (uint32_t i = 0; i < cfg.n_vocab; i++) { ids[i] = i; }
    v->str = cfg.vocab;
    v->len = cfg.vocab_len;
    v->id = ids;
    v->n = cfg.n_vocab;
    p->flags = cfg.wp_flags;
    p->max_chars = cfg.wp_max_chars;
    p->unk = cfg.wp_unk;
    p->unk_len = cfg.wp_unk_len;
    p->prefix = cfg.wp_prefix;
    p->prefix_len = cfg.wp_prefix_len;
    return 0;
}

#endif /* TOKS_TEST_WP_JSON_H */
