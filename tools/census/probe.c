/*
 * tools/census/probe.c: what toks's own load path says about each census tokenizer today (docs/coverage.md).
 * The verdict is the public loader's: toks_load(path) on a tokenizer.json file, or on a model directory (a
 * tiktoken model is three files; coverage.py writes each repo's files under their own names), so every step
 * src/core/load.c runs counts: the config parse (or the tiktoken reader), compile, the split-parameter check, the
 * algorithm's tables (byte-level bpe, sentencepiece-style bpe, wordpiece, unigram), the decode tables and the
 * tier. For a file, toks_config_parse alone names the config stage, and toks_load_mem_copy on the same bytes must
 * agree with toks_load. One line per input:
 *
 *   <path> TAB <stage> TAB <code> TAB <what>      stage 0 = loads (what = "ok", or "ok generic" when the
 *                                                 pre-tokenizer runs on the generic engine: no compiled fast path,
 *                                                 SPEC 1.4), 1 = the config parse refused it (what = its
 *                                                 reason), 2 = a later load step refused it, or a model directory
 *                                                 was refused (what = toks_load's diag), 8 = toks_load,
 *                                                 toks_load_mem_copy and the config stage disagree, 9 = unreadable
 *
 * Census tooling, not part of the library. Build on a lab host (maintainer doctrine: no builds on the control mac):
 *   make lib && $CC -std=c17 -O2 -Iinclude -Isrc/core -Isrc/platform tools/census/probe.c \
 *       build/<os>-<isa>/libtoks.a -lpthread -o build/census-probe
 */
#include "toks.h"
#include "core.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* toks_load (toks_load_mem_copy when data != NULL) -> its code; its diag (or "ok") in what[248] */
static int64_t load(const char *path, const uint8_t *data, uint64_t len, char *what)
{
    toks_diag dg;
    memset(&dg, 0, sizeof(dg));
    toks_load_opts o;
    memset(&o, 0, sizeof(o));
    o.size = (uint32_t)sizeof(o);
    o.diag = &dg;
    toks_ctx *c = NULL;
    int64_t r = (data != NULL) ? toks_load_mem_copy(&c, data, len, &o) : toks_load(&c, path, &o);
    toks_info inf;
    memset(&inf, 0, sizeof(inf));
    inf.size = (uint32_t)sizeof(inf);
    int generic = (r == 0 && toks_get_info(c, &inf) == 0 && (inf.paths & TOKS_PATH_SCAN) == 0u);
    toks_unload(c);
    memcpy(what, r != 0 ? dg.what : generic ? "ok generic" : "ok", r != 0 ? sizeof dg.what : generic ? 11u : 3u);
    what[sizeof dg.what - 1u] = 0;
    for (char *p = what; *p != 0; p++) {                 /* one line per input */
        if (*p == '\t' || *p == '\n' || *p == '\r') { *p = ' '; }
    }
    return r;
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const char *path = argv[i];
        char what[248], what2[248];
        int64_t r = load(path, NULL, 0u, what);
        uint8_t *data = NULL;
        uint64_t len = 0u;
        int is_dir = 0;
        if (toks_plat_read_file(path, &data, &len, &is_dir) != 0) {
            if (is_dir) { printf("%s\t%d\t%lld\t%s\n", path, r == 0 ? 0 : 2, (long long)r, what); }
            else { printf("%s\t9\t%lld\tunreadable\n", path, (long long)r); }
            continue;
        }
        /* the config stage alone, then the same bytes through toks_load_mem_copy */
        uint64_t cap = toks_config_arena_bound(len);
        uint8_t *ar = malloc((size_t)cap);
        toks_config cfg;
        toks_err err = { 0, NULL };
        toks_arena a = { ar, cap, 0 };
        int64_t pr = (ar == NULL) ? TOKS_E_NOMEM : toks_config_parse(data, len, &a, &cfg, &err);
        int64_t lr = load(path, data, len, what2);
        int stage = (r == 0) ? 0 : (pr != 0) ? 1 : 2;
        if (lr != r || (pr != 0 && pr != r)) {
            printf("%s\t8\t%lld\ttoks_load %lld (%s), toks_load_mem_copy %lld (%s), config parse %lld (%s)\n", path,
                   (long long)r, (long long)r, what, (long long)lr, what2, (long long)pr,
                   (err.what != NULL) ? err.what : "-");
        } else {
            printf("%s\t%d\t%lld\t%s\n", path, stage, (long long)r, (stage == 1 && err.what != NULL) ? err.what : what);
        }
        free(ar);
        toks_plat_free(data, len);
    }
    return 0;
}
