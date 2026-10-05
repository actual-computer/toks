/*
 * test_kimi.c: Kimi K3 (moonshotai/Kimi-K3: tiktoken.model + tokenizer_config.json + tokenization_kimi.py)
 * end to end through include/toks.h against the model's own tokenizer (test_kimi.inc, written by
 * test_kimi.py from tests/parity/oracle_tiktoken.py): ids in ALL (serving), NONSPECIAL (direct) and NONE (none)
 * on the notes of docs/models/kimi.md and on texts at the wrapper's cuts (names straddling the 25,000-run cut
 * and the 400,000-code-point chunk edge, long runs of each str.isspace() class, the 25,000 / 25,001-byte edge of
 * run_cuts' skipped walk), decode of the serving ids with
 * and without TOKS_SKIP_SPECIAL, pieces tiling every text. Every capacity below the count keeps the count and
 * the prefix. The files come from $TOKS_KIMI_DIR or ~/.cache/toks/kimik3 (tests/parity/oracle_tiktoken.py
 * --fetch); absent, the test prints SKIP and fails nothing. tests/parity/run_kimi.py is the parity at scale.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_kimi.inc"

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static void sha_hex(const void *p, uint64_t n, char hex[65])
{
    uint8_t d[32];
    toks_sha256((const uint8_t *)p, n, d);
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", d[i]); }
}

static uint8_t *text_of(int c, uint64_t *len)
{
    uint64_t n = 0;
    for (uint32_t i = 0; i < KC[c].n; i++) { n += (uint64_t)KP[KC[c].first + i].len * KP[KC[c].first + i].rep; }
    uint8_t *t = (uint8_t *)malloc((size_t)n + 1u);
    uint64_t o = 0;
    for (uint32_t i = 0; i < KC[c].n; i++) {
        for (uint32_t r = 0; r < KP[KC[c].first + i].rep; r++) {
            memcpy(t + o, KP[KC[c].first + i].s, KP[KC[c].first + i].len);
            o += KP[KC[c].first + i].len;
        }
    }
    *len = n;
    return t;
}

int main(void)
{
    const char *dir = getenv("TOKS_KIMI_DIR");
    char buf[1024];
    if (dir == NULL || dir[0] == 0) {
        snprintf(buf, sizeof buf, "%s/.cache/toks/kimik3", getenv("HOME") ? getenv("HOME") : ".");
        dir = buf;
    }
    char model[1100];
    snprintf(model, sizeof model, "%s/tiktoken.model", dir);
    FILE *f = fopen(model, "rb");
    if (f == NULL) {
        printf("SKIP test_kimi: %s missing (tests/parity/oracle_tiktoken.py --fetch)\n", model);
        return 0;
    }
    fclose(f);
    toks_ctx *ctx = NULL;
    toks_diag diag;
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &diag;
    int64_t r = toks_load(&ctx, dir, &o);
    CHECK(r == 0 && ctx != NULL, "load %s: %" PRId64 " %s", dir, r, diag.what);
    if (ctx == NULL) { printf("test_kimi: %ld checks, %d failures\n", checks, failures); return 1; }

    uint64_t maxlen = 0;
    for (int c = 0; c < KIMI_N; c++) {
        uint64_t n;
        free(text_of(c, &n));
        if (n > maxlen) { maxlen = n; }
    }
    uint64_t sb = toks_scratch_bytes(ctx, maxlen, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch");
    static const uint32_t FLAGS[3] = { TOKS_ADDED_ALL, TOKS_ADDED_NONSPECIAL, TOKS_ADDED_NONE };
    uint64_t total_ids = 0;
    for (int c = 0; c < KIMI_N; c++) {
        uint64_t len;
        uint8_t *t = text_of(c, &len);
        uint64_t cap = len + 16u;
        uint32_t *ids = (uint32_t *)malloc((size_t)cap * 4u);
        uint32_t *ids0 = (uint32_t *)malloc((size_t)cap * 4u);
        int64_t n0 = 0;
        for (int m = 0; m < 3; m++) {
            char hex[65];
            int64_t n = toks_encode(ctx, t, len, FLAGS[m], ids, cap, scr);
            sha_hex(ids, n > 0 ? (uint64_t)n * 4u : 0u, hex);
            CHECK(n == (int64_t)KE[3 * c + m].n && strcmp(hex, KE[3 * c + m].sha) == 0,
                  "case %d mode %u (%" PRIu64 " bytes): %" PRId64 " ids, want %u", c, FLAGS[m], len, n, KE[3 * c + m].n);
            if (m == 0) { memcpy(ids0, ids, (size_t)(n > 0 ? n : 0) * 4u); n0 = n; }
            total_ids += (uint64_t)(n > 0 ? n : 0);
            /* a short capacity: the count stays, the prefix is exact */
            if (n > 2) {
                uint64_t half = (uint64_t)n / 2u;
                uint32_t *p = (uint32_t *)malloc((size_t)(half + 1u) * 4u);
                p[half] = 0xEEEEEEEEu;
                int64_t k = toks_encode(ctx, t, len, FLAGS[m], p, half, scr);
                CHECK(k == n && memcmp(p, ids, (size_t)half * 4u) == 0 && p[half] == 0xEEEEEEEEu, "case %d mode %u cap %" PRIu64,
                      c, FLAGS[m], half);
                free(p);
            }
            /* pieces: strictly increasing ends tiling [0, len) */
            int64_t pn = toks_pieces(ctx, t, len, FLAGS[m], ids, cap, scr);
            int tiled = pn >= 0 && (len == 0u ? pn == 0 : (pn > 0 && ids[pn - 1] == len));
            for (int64_t i = 1; tiled && i < pn; i++) { tiled = ids[i] > ids[i - 1]; }
            CHECK(tiled, "case %d mode %u pieces: %" PRId64, c, FLAGS[m], pn);
        }
        for (int skip = 0; skip < 2; skip++) {
            uint64_t dcap = 64u * (uint64_t)(n0 > 0 ? n0 : 1) + 256u;
            uint8_t *d = (uint8_t *)malloc((size_t)dcap);
            int64_t k = toks_decode(ctx, ids0, (uint64_t)(n0 > 0 ? n0 : 0), skip ? TOKS_SKIP_SPECIAL : 0u, d, dcap);
            if (k > (int64_t)dcap) {                    /* the count runs on past cap: decode again into room */
                free(d);
                dcap = (uint64_t)k;
                d = (uint8_t *)malloc((size_t)dcap);
                k = toks_decode(ctx, ids0, (uint64_t)n0, skip ? TOKS_SKIP_SPECIAL : 0u, d, dcap);
            }
            char hex[65];
            sha_hex(d, k > 0 ? (uint64_t)k : 0u, hex);
            CHECK(k == (int64_t)KD[2 * c + skip].n && strcmp(hex, KD[2 * c + skip].sha) == 0, "case %d decode skip %d: %" PRId64
                  " bytes, want %u", c, skip, k, KD[2 * c + skip].n);
            free(d);
        }
        free(ids0);
        free(ids);
        free(t);
    }
    free(scr);
    toks_unload(ctx);
    printf("test_kimi: %d texts (%" PRIu64 " ids over 3 modes), %ld checks, %d failures\n", KIMI_N, total_ids, checks,
           failures);
    return failures ? 1 : 0;
}
