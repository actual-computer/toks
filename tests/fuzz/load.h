/*
 * tests/fuzz/load.h: the load harnesses' body (fuzz_load.c: arbitrary bytes; fuzz_load_json.c: structurally
 * valid hostile tokenizer files from jsonmut.h), SPEC T6 / T9.
 *
 * fz_load_one(bytes): toks_load_mem_copy at TOKS_TIER_SCALAR and AUTO (1 in 8 inputs through toks_load on a
 * file instead, 1 in 64 as a model directory holding it): both tiers must give the same verdict; a refusal is a
 * documented code with *out NULL and diag naming it (fuzz.h fz_tok_open). An accepted file is a synthetic
 * tokenizer (SPEC §14.4): toks_info sane, then the text battery of check.h on probe texts, slices of the
 * file itself and concatenations of its own token strings (encode with every oracle, pieces, now and then
 * toks_par and the split cuts), and decode + stream of random ids, all through the same oracles as the pinned
 * tokenizers (no FZ_RT assumption).
 */
#ifndef TOKS_FUZZ_LOAD_H
#define TOKS_FUZZ_LOAD_H

#include "check.h"

#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *const FZ_PROBE[] = {
    "Hello, world! It's 2026.", " leading", "trailing ", "\t\n\r\n x", "e\xcc\x81 caf\xc3\xa9 \xe2\x84\xab",
    "\xe4\xb8\xad\xe6\x96\x87 \xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd8\xa7\xd9\x84", "1234567 3.14", "<|endoftext|>",
    "\xe2\x96\x81\xe2\x96\x81\x61 b", "\xff\xfe\x80 x", "  \n\n  def f(x):\n    return x\n", "[CLS] ##ing <s> </s>",
};

/* the directory a file load writes into, $TMPDIR/toks-fuzz-<pid>: made before each such load and removed after it,
 * so a run leaves nothing behind (a crash in between leaves it, holding the input, beside the finding) */
FZ_FN const char *fz_tmp_dir(void)
{
    static char d[512];
    if (d[0] == 0) {
        const char *t = getenv("TMPDIR");
        snprintf(d, sizeof d, "%s/toks-fuzz-%ld", t != NULL ? t : "/tmp", (long)getpid());
    }
    return d;
}

FZ_FN int fz_write_file(const char *path, const uint8_t *d, uint64_t n)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) { return 0; }
    int ok = fwrite(d, 1, (size_t)n, f) == (size_t)n;
    return fclose(f) == 0 && ok;
}

/* the battery on an accepted tokenizer */
FZ_FN void fz_battery(fz_tok *t, const uint8_t *src, uint64_t n, uint64_t seed)
{
    uint64_t s = seed;
    for (int k = 0; k < 3; k++) {
        fz_bytes x = { NULL, 0, 0 };
        uint64_t kind = fz_below(&s, 4u);
        if (kind == 0u) {                               /* a probe text */
            const char *p = FZ_PROBE[fz_below(&s, FZ_N(FZ_PROBE))];
            fz_bytes_put(&x, p, strlen(p));
        } else if (kind == 1u && n != 0u) {             /* a slice of the file itself */
            uint64_t a = fz_below(&s, n), l = 1u + fz_below(&s, n - a < 512u ? n - a : 512u);
            fz_bytes_put(&x, src + a, l);
        } else {                                        /* its own token strings, glued */
            uint64_t m = 1u + fz_below(&s, 24u);
            for (uint64_t i = 0; i < m; i++) {
                uint64_t l = 0;
                const uint8_t *p = toks_token(t->a, (uint32_t)fz_below(&s, t->info.n_ids), &l);
                if (p != NULL && l <= 256u) { fz_bytes_put(&x, p, l); }
                if (fz_below(&s, 4u) == 0u) { fz_bytes_put(&x, " ", 1u); }
            }
        }
        uint32_t flags = (uint32_t)fz_below(&s, 3u) | (fz_below(&s, 2u) ? TOKS_NO_POSTPROCESS : 0u) |
                         (fz_below(&s, 4u) == 0u ? TOKS_CONTINUATION : 0u);
        uint8_t *blk = (uint8_t *)fz_alloc(x.n);                 /* the text flush with its heap block */
        if (x.n != 0u) { memcpy(blk, x.p, (size_t)x.n); }
        fz_check_text(t, (int)(k == 2), blk, x.n, flags, fz_rng(&s));
        if (fz_below(&s, 16u) == 0u) { fz_check_par(t, blk, x.n, flags, fz_rng(&s)); }
        free(blk);
        fz_bytes_free(&x);
    }
    uint64_t m = fz_below(&s, 40u);                     /* decode + stream of random ids (now and then one too big) */
    uint32_t *ids = (uint32_t *)fz_alloc((m + 1u) * 4u);
    for (uint64_t i = 0; i < m; i++) {
        ids[i] = (uint32_t)fz_below(&s, fz_below(&s, 32u) == 0u ? (uint64_t)t->info.n_ids + 2u : t->info.n_ids);
    }
    fz_check_decode(t, ids, m, (uint32_t)fz_below(&s, 2u), fz_rng(&s));
    fz_check_stream(t, ids, m, (uint32_t)fz_below(&s, 2u), fz_rng(&s));
    free(ids);
}

FZ_FN int fz_load_one(const uint8_t *d, size_t n)
{
    fz_tok t;
    uint64_t h = fz_hash(d, n);
    int64_t r;
    fz_what = "loaded file";
    if ((h & 7u) == 0u) {                               /* through a file (or a model directory) */
        char path[640];
        int dir = ((h >> 3) & 7u) == 0u;
        const char *td = fz_tmp_dir();
        if (mkdir(td, 0700) != 0 && errno != EEXIST) { return 0; }
        snprintf(path, sizeof path, "%s/%s", td, dir ? "tokenizer.json" : "in.json");
        if (!fz_write_file(path, d, n)) { unlink(path); rmdir(td); return 0; }
        r = fz_tok_open(&t, "loaded file", dir ? td : path, NULL, 0u);
        unlink(path);
        rmdir(td);
    } else {
        r = fz_tok_open(&t, "loaded bytes", NULL, d, n);
    }
    if (r != 0) {
        FZ_CHECK(r != TOKS_E_NOMEM || n > (16u << 20), "TOKS_E_NOMEM on a %zu-byte source", n);
        return 0;
    }
    FZ_CHECK(t.info.algorithm >= TOKS_ALGO_BPE_BYTELEVEL && t.info.algorithm <= TOKS_ALGO_WORDPIECE,
             "algorithm %u", t.info.algorithm);
    fz_battery(&t, d, n, h);
    fz_tok_close(&t);
    return 0;
}

#endif /* TOKS_FUZZ_LOAD_H */
