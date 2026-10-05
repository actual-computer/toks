/*
 * test_misalign.c: SPEC §4.1 / toks.h, "no alignment is required of any buffer": encode, pieces and decode
 * with out, ids and the scratch at every misalignment 1..3, and split_points with offs at 1..7, give exactly the
 * aligned results (a fuzz finding,
 * fixed in commit d494771). The core reaches caller arrays only through core.h's
 * toks_ld32 / toks_st32. Under `make test` this checks the results on the build's tiers; the c paths' typed
 * accesses are what UBSan checks (on a lab host or the mac, seconds; no asm, so the c twins run):
 *
 *   clang -std=c17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=all -fno-strict-aliasing -fwrapv \
 *       -Iinclude -Isrc/core -Isrc/platform -o build/misalign_ubsan tests/c/test_misalign.c \
 *       src/core/?*.c src/platform/?*.c src/gen/?*.c && ./build/misalign_ubsan
 *
 * The tokenizers (gpt2, and the wordpiece / unigram emitters: wp-bert-uncased, uni_t5base) are under
 * $TOKS_TOKENIZER_CACHE or ~/.cache/toks/tokenizers (as test_e2e); a missing one: SKIP. The wordpiece model wrote ids
 * through a typed pointer into a misaligned out until fuzz finding 2 (UBSan, wp.c / wp_api.c).
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static int run(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    char path[1024];
    if (root != NULL && root[0] != 0) { snprintf(path, sizeof path, "%s/%s", root, name); }
    else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, path, NULL) != 0) {
        printf("SKIP test_misalign: %s not loadable\n", path);
        return 0;
    }
    static const char text[] = "Hello, world! It's a misaligned buffer: \xe4\xb8\xad\xe6\x96\x87. <|endoftext|> again, "
                               "and a longer tail so that more than one K5 call and the bounce both run: 0123456789";
    const uint64_t len = sizeof text - 1u;
    uint64_t sb = toks_scratch_bytes(ctx, len, 0u);
    uint8_t *scr_raw = (uint8_t *)malloc((size_t)sb + 64u);
    uint8_t *out_raw = (uint8_t *)malloc(4u * 512u + 64u);
    uint32_t want[512], wpieces[512];
    uint8_t wdec[1024], dec[1024];
    if (scr_raw == NULL || out_raw == NULL) { fprintf(stderr, "malloc\n"); return 2; }

    /* the aligned reference */
    uint8_t *scr = scr_raw;
    CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "aligned scratch");
    int64_t n = toks_encode(ctx, text, len, 0u, want, 512u, scr);
    int64_t np = toks_pieces(ctx, text, len, 0u, wpieces, 512u, scr);
    int64_t nd = toks_decode(ctx, want, (uint64_t)(n > 0 ? n : 0), 0u, wdec, sizeof wdec);
    CHECK(n > 8 && np > 8 && nd > 0 && (strcmp(name, "gpt2") != 0 || nd == (int64_t)len),   /* gpt2 round-trips */
          "%s aligned: %" PRId64 " ids, %" PRId64 " pieces, %" PRId64 " bytes", name, n, np, nd);

    for (uint32_t mis = 1u; mis < 4u; mis++) {                 /* bound: 3 misalignments */
        scr = scr_raw + mis;
        CHECK(toks_scratch_init(ctx, scr, sb, 0u) == 0, "scratch +%u", mis);
        uint8_t *o = out_raw + mis;                            /* out / ids at an odd address */
        for (uint64_t cap = 0u; cap <= (uint64_t)n + 1u; cap += (cap < 8u ? 1u : 7u)) {   /* direct and bounced */
            memset(o, 0xEE, 4u * 512u);
            int64_t r = toks_encode(ctx, text, len, 0u, (uint32_t *)(void *)o, cap, scr);
            uint64_t m = cap < (uint64_t)n ? cap : (uint64_t)n;
            CHECK(r == n && memcmp(o, want, (size_t)m * 4u) == 0, "encode, out +%u, cap %" PRIu64 ": %" PRId64, mis, cap, r);
        }
        int64_t rp = toks_pieces(ctx, text, len, 0u, (uint32_t *)(void *)o, 512u, scr);
        CHECK(rp == np && memcmp(o, wpieces, (size_t)np * 4u) == 0, "pieces, ends +%u", mis);
        memcpy(o, want, (size_t)n * 4u);                       /* ids at an odd address */
        int64_t rd = toks_decode(ctx, (const uint32_t *)(const void *)o, (uint64_t)n, 0u, dec, sizeof dec);
        CHECK(rd == nd && memcmp(dec, wdec, (size_t)nd) == 0, "decode, ids +%u", mis);
    }
    {   /* toks_split_points: offs is a uint64_t array, so every misalignment 1..7 (the T11 audit's gap: split.c
         * stored through the typed pointer until toks_st64) */
        char big[4096];
        uint64_t bl = 0u;
        while (bl + len < sizeof big) { memcpy(big + bl, text, len); bl += len; }   /* bound: sizeof big / len */
        uint64_t woffs[16];
        int64_t ns = toks_split_points(ctx, big, bl, 0u, 8u, woffs, 16u, scr);
        CHECK(ns >= 0, "split_points aligned: %" PRId64, ns);
        uint8_t *offs_raw = (uint8_t *)malloc(16u * 8u + 64u);
        CHECK(offs_raw != NULL, "malloc offs");
        for (uint32_t mis = 1u; mis < 8u; mis++) {             /* bound: 7 misalignments */
            uint8_t *o = offs_raw + ((64u - ((uintptr_t)offs_raw & 63u)) & 63u) + mis;
            memset(o, 0xA5, 16u * 8u);
            int64_t rs = toks_split_points(ctx, big, bl, 0u, 8u, (uint64_t *)(void *)o, 16u, scr);
            CHECK(rs == ns && memcmp(o, woffs, (size_t)(ns > 0 ? ns : 0) * 8u) == 0, "split_points, offs +%u: %" PRId64, mis, rs);
        }
        free(offs_raw);
    }
    free(out_raw);
    free(scr_raw);
    toks_unload(ctx);
    return 0;
}

int main(void)
{
    static const char *const NAMES[] = { "gpt2", "wp-bert-uncased", "uni_t5base" };
    for (uint32_t i = 0; i < 3u; i++) {                        /* bound: 3 tokenizers */
        if (run(NAMES[i]) != 0) { return 2; }
    }
    printf("test_misalign: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
