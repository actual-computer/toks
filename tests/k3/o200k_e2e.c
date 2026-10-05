/*
 * o200k_e2e.c: a tokenizer.json with the o200k pattern through the real load path, then text -> ids
 * against hf tokenizers 0.23.2. NOT part of `make test`; tests/k3/o200k_e2e.sh builds and feeds it:
 *
 *   uv run ... python tests/k3/o200k_gen.py e2e TOKENIZER [...] | ./o200k_e2e TOKENIZER
 *
 * Load: toks_config_parse + toks_compile + toks_bpe_build (as toks_load will), then the compiled
 * template must be TOKS_TMPL_O200K with CONTR_CI | DIGITS_1_3 and class tables built with class_flags 0.
 * Records: u32 len, text, u32 n_ids, ids -- hf's Tokenizer.encode(text, add_special_tokens=False) with
 * encode_special_tokens=True (every gpt-oss added token is special, so hf runs exactly pre-tokenizer +
 * bpe on the whole text). Per record: K3 o200k (cap TOKS_CHUNK_PIECES, the driver's chunking) then K5
 * over each chunk, ids concatenated, compared with hf's. Exit 0 iff all equal.
 */
#include "../../src/core/bpe.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXREC (1u << 23)

static void *xalloc(uint64_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(uint64_t)63);
    if (p == NULL) { fprintf(stderr, "out of memory (%" PRIu64 " bytes)\n", n); exit(2); }
    return p;
}

static int rd(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: o200k_e2e <tokenizer.json> < records\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    uint64_t flen = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *file = xalloc(flen);
    if (fread(file, 1, flen, f) != flen) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);

    uint64_t plen = toks_config_arena_bound(flen);
    toks_arena par = { xalloc(plen), plen, 0 };
    toks_config cfg;
    toks_err err = { 0, NULL };
    struct toks_ctx *ctx = calloc(1, sizeof *ctx);
    int64_t r = toks_config_parse(file, flen, &par, &cfg, &err);
    if (r != 0) { printf("load %s: toks_config_parse refused: %" PRId64 " %s\n", argv[1], r, err.what ? err.what : ""); return 1; }
    r = toks_compile(&cfg, ctx, &par);
    if (r != 0) { printf("load %s: toks_compile refused: %" PRId64 "\n", argv[1], r); return 1; }
    uint64_t need = toks_bpe_tables_bytes(&cfg);
    toks_arena bar = { xalloc(need), need, 0 };
    r = toks_bpe_build(&ctx->t, &bar, &cfg);
    if (r != 0) { printf("load %s: toks_bpe_build: %" PRId64 "\n", argv[1], r); return 1; }
    const toks_tables *t = &ctx->t;
    int tmpl_ok = t->tmpl == TOKS_TMPL_O200K && t->tmpl_params == (TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1_3) &&
                  (toks_cls_cp(t, 0x0301u) & (TOKS_C_BASE_MASK | TOKS_C_UPPER | TOKS_C_LOWER | TOKS_C_MARK)) ==
                      (TOKS_C_P | TOKS_C_UPPER | TOKS_C_LOWER | TOKS_C_MARK);
    printf("load %s: n_ids %u, merges %u, flags %#x, tmpl %u, tmpl_params %#x, U+0301 class %#x: %s\n", argv[1],
           t->n_ids, t->n_merges, t->flags, t->tmpl, t->tmpl_params, toks_cls_cp(t, 0x0301u),
           tmpl_ok ? "the o200k template" : "NOT the o200k template");
    if (!tmpl_ok) { return 1; }

    uint8_t *text = xalloc(MAXREC);
    uint32_t *want = xalloc(4u * (uint64_t)MAXREC + 16u), *got = xalloc(4u * (uint64_t)MAXREC + 64u);
    uint32_t ends[TOKS_CHUNK_PIECES];
    uint64_t work_bytes = TOKS_BPE_WORK_BYTES(MAXREC);
    uint8_t *work = xalloc(work_bytes);
    uint8_t *cache = xalloc(TOKS_CACHE_BUCKETS * 64u);
    memset(cache, 0, TOKS_CACHE_BUCKETS * 64u);
    uint64_t cases = 0, bad = 0, bytes = 0, pieces = 0, nids = 0;
    uint32_t len;
    while (rd(&len, 4)) {
        uint32_t n;
        if (len > MAXREC || !rd(text, len) || !rd(&n, 4) || n > MAXREC || !rd(want, 4u * (size_t)n)) {
            fprintf(stderr, "corrupt record stream at case %" PRIu64 "\n", cases);
            return 2;
        }
        cases++;
        bytes += len;
        nids += n;
        uint64_t pos = 0, no = 0;
        while (pos < len) {                             /* bound: chunks (each advances >= 1 piece) */
            toks_k3_args k3 = { text, len, pos, ends, TOKS_CHUNK_PIECES, 0, 0, 0 };
            uint64_t np = toks_k3_scan_o200k_c(t, &k3);
            uint64_t start = pos, longest = 0, prev = pos;
            for (uint64_t i = 0; i < np; i++) {        /* bound: np */
                longest = ends[i] - prev > longest ? ends[i] - prev : longest;
                prev = ends[i];
            }
            toks_k5_args k5 = { text, len, ends, np, start, got + no, (k3.pos - start) + 4u, 0, cache,
                                TOKS_CACHE_MASK, work, TOKS_BPE_WORK_BYTES(longest), 0, 0, 0, 0, NULL };
            no += toks_k5_encode_c(t, &k5);
            pieces += np;
            pos = k3.pos;
        }
        if (no != n || memcmp(got, want, 4u * (size_t)n) != 0) {
            if (bad < 10) {
                fprintf(stderr, "MISMATCH case %" PRIu64 " len %u: %" PRIu64 " ids, hf %u; text", cases, len, no, n);
                for (uint32_t i = 0; i < len && i < 48u; i++) { fprintf(stderr, " %02X", text[i]); }
                fprintf(stderr, "\n");
            }
            bad++;
        }
    }
    printf("o200k_e2e: %" PRIu64 " texts (%" PRIu64 " bytes, %" PRIu64 " pieces, %" PRIu64 " ids), %" PRIu64
           " mismatches\n", cases, bytes, pieces, nids, bad);
    return bad ? 1 : 0;
}
