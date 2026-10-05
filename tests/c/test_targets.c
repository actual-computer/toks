/* tests/c/test_targets.c: the critical targets (SPEC §1.1 (d)): what this build does with each pinned file.
 *
 * tests/data/targets/ledger.txt holds one line per target: name, cache file, sha256, expectation:
 *   compiles            toks_config_parse + toks_compile accept the file
 *   refused <text>      refused, and the reason contains <text>
 *   no-reader <text>    toks has no reader yet for what the model ships (printed, not loaded)
 * A file missing from $TOKS_TOKENIZER_CACHE (default ~/.cache/toks/tokenizers) is a SKIP, printed with the fetch
 * command; a file whose sha256 differs from the pin is a failure. A change that moves a target updates its line.
 * A <stem>.tiktoken file goes through the tiktoken reader with its companions <stem>_tokenizer_config.json and
 * <stem>_tokenization_kimi.py from the same directory (their pins: tools/corpora/fetch_tokenizers.py). */
#include "core.h"
#include "config.h"
#include "compile.h"
#include "tiktoken.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lines.inc"

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = (n > 0) ? malloc((size_t)n) : NULL;
    if (d == NULL || fread(d, 1, (size_t)n, f) != (size_t)n) { free(d); fclose(f); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return d;
}

int main(void)
{
    const char *dir = getenv("TOKS_TOKENIZER_CACHE");
    char home[1024];
    if (dir == NULL && getenv("HOME") != NULL) {
        snprintf(home, sizeof(home), "%s/.cache/toks/tokenizers", getenv("HOME"));
        dir = home;
    }
    FILE *lf = fopen("tests/data/targets/ledger.txt", "rb");
    if (lf == NULL || dir == NULL) { printf("test_targets: no ledger or no cache dir\n"); return 1; }
    char line[1024];
    int targets = 0, skipped = 0, fails = 0;
    td_at at = { "tests/data/targets/ledger.txt", 0 };
    while (fgets(line, sizeof(line), lf) != NULL) {
        at.line++;
        if (!td_line(&at, line, sizeof line)) { fails++; break; }
        if (line[0] == '#' || line[0] == '\n') { continue; }
        char name[64], file[256], sha[80], kind[32];
        int used = 0;
        if (sscanf(line, "%63s %255s %79s %31s %n", name, file, sha, kind, &used) < 4) {
            printf("  bad ledger line: %s", line);
            fails++;
            continue;
        }
        char *want = line + used;
        want[strcspn(want, "\n")] = '\0';
        targets++;
        if (strcmp(kind, "no-reader") == 0) { printf("  %-13s no reader yet (%s)\n", name, want); continue; }
        char path[1400];
        snprintf(path, sizeof(path), "%s/%s", dir, file);
        uint64_t len = 0;
        uint8_t *data = slurp(path, &len);
        if (data == NULL) {
            printf("  %-13s SKIP: %s missing (uv run tools/corpora/fetch_tokenizers.py --targets)\n", name, path);
            skipped++;
            continue;
        }
        uint8_t d[32];
        char hex[65];
        toks_sha256(data, len, d);
        for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", d[i]); }
        if (strcmp(hex, sha) != 0) {
            printf("  %-13s FAIL: %s is not the pinned file (sha256 %s)\n", name, path, hex);
            fails++;
            free(data);
            continue;
        }
        uint64_t cap = toks_config_arena_bound(len);
        uint8_t *comp[2] = { NULL, NULL };                  /* a tiktoken target's companions */
        uint64_t clen[2] = { 0, 0 };
        size_t fl = strlen(file);
        int tik = fl > 9 && strcmp(file + fl - 9, ".tiktoken") == 0;
        if (tik) {
            static const char *const SUF[3] = { "_tokenizer_config.json", "_tokenization_kimi.py", "_tokenization_qwen.py" };
            for (int k = 0; k < 3; k++) {                   /* the config, then kimi's wrapper or Qwen-1's */
                char cp[1500];
                snprintf(cp, sizeof(cp), "%s/%.*s%s", dir, (int)(fl - 9), file, SUF[k]);
                if (k < 2 || comp[1] == NULL) { comp[k < 2 ? k : 1] = slurp(cp, &clen[k < 2 ? k : 1]); }
            }
            cap = toks_tiktoken_arena_bound(len, clen[0], clen[1]);
        }
        uint8_t *mem = malloc(cap);
        toks_arena ar = { mem, cap, 0 };
        toks_config cfg;
        toks_err err = { 0, NULL };
        toks_ctx ctx;
        memset(&cfg, 0, sizeof(cfg));
        memset(&ctx, 0, sizeof(ctx));
        int64_t r = TOKS_E_NOMEM;
        if (mem != NULL && tik) {
            toks_tiktoken_info info;
            r = (comp[0] != NULL && comp[1] != NULL)
                ? toks_tiktoken_parse(data, len, comp[0], clen[0], comp[1], clen[1], &ar, &cfg, &info, &err)
                : TOKS_E_OPEN;
            if (r == TOKS_E_OPEN && err.what == NULL) { err.what = "companion files missing (fetch_tokenizers.py --targets)"; }
        } else if (mem != NULL) {
            r = toks_config_parse(data, len, &ar, &cfg, &err);
        }
        free(comp[0]);
        free(comp[1]);
        int stage = 1;
        if (r == 0) {
            stage = 2;
            r = toks_compile(&cfg, &ctx, &ar);
        }
        int ok = (strcmp(kind, "compiles") == 0) ? (r == 0)
               : (strcmp(kind, "refused") == 0 && r != 0 && stage == 1 && err.what != NULL && strstr(err.what, want) != NULL);
        if (r == 0) {
            printf("  %-13s %s: compiles (vocab %u, merges %u, added %u)\n", name, ok ? "ok" : "FAIL", cfg.n_vocab, cfg.n_merges,
                   cfg.n_added);
        } else {
            printf("  %-13s %s: refused at %s: %lld \"%s\"\n", name, ok ? "ok" : "FAIL", stage == 1 ? "config" : "compile", (long long)r,
                   err.what != NULL ? err.what : "");
        }
        if (!ok) {
            printf("            the ledger expects: %s %s (if your change moved this target, update tests/data/targets/ledger.txt)\n",
                   kind, want);
            fails++;
        }
        if (ctx.mem_tables != NULL) { toks_plat_arena_free(ctx.mem_tables, ctx.mem_tables_len); }
        free(mem);
        free(data);
    }
    fclose(lf);
    printf("test_targets: %d targets, %d skipped, %d failures\n", targets, skipped, fails);
    return fails != 0;
}
