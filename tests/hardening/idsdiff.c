/* tests/hardening/idsdiff.c: a differential digest for semantics-preserving fixes (docs/hardening.md). Encodes N
 * random texts drawn from an alphabet of whitespace (all 25 regex \s chars, cut multi-byte ones), the tokenizer's
 * markup tokens and any literals given on the command line (whole, cut, doubled), letters, digits, CJK, marks and
 * invalid bytes, under every added-token mode with and without post-processing, and prints one fnv-1a digest of the
 * ids (and of toks_pieces' ends) per text. Build it against two libraries (before / after a fix) and diff the
 * outputs: any line that differs is a text whose ids moved.
 *
 *   idsdiff <tokenizer> <n> <seed> [literal...]       literals may use \n \t \xHH escapes
 */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint64_t fnv(uint64_t h, const void *p, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    for (uint64_t i = 0; i < n; i++) { h = (h ^ b[i]) * 0x100000001B3ull; }
    return h;
}

static const char *const WS[] = {
    "\t", "\n", "\v", "\f", "\r", " ", "\xc2\x85", "\xc2\xa0", "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
    "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86", "\xe2\x80\x87", "\xe2\x80\x88",
    "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8", "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80",
    "\xe3\x80", "\xe2\x80", "\xc2", "\x1c", "\x1f", "  ", "\n\n", " \n", "\r\n",
};
static const char *const OTHER[] = {
    "a", "b", "Z", "hello", "1", "42", "<", ">", "|", "<|", "|>", "'s", ".", "\xe4\xb8\xad", "\xcc\x81", "\xff", "\x80",
    "\xe2\x96\x81", "e\xcc\x81", "\xf0\x9f\x98\x80", "_", "##", "[", "]",
};

static uint64_t unescape(const char *s, uint8_t *o)
{
    uint64_t n = 0;
    while (*s) {
        if (s[0] == '\\' && s[1] == 'n') { o[n++] = '\n'; s += 2; }
        else if (s[0] == '\\' && s[1] == 't') { o[n++] = '\t'; s += 2; }
        else if (s[0] == '\\' && s[1] == 'x' && s[2] && s[3]) { char h[3] = { s[2], s[3], 0 }; o[n++] = (uint8_t)strtoul(h, NULL, 16); s += 4; }
        else { o[n++] = (uint8_t)*s++; }
    }
    return n;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: idsdiff <tokenizer> <n> <seed> [literal...]\n"); return 2; }
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, argv[1], NULL);
    if (r != 0) { printf("load %lld\n", (long long)r); return 1; }
    uint64_t n = strtoull(argv[2], NULL, 10), seed = strtoull(argv[3], NULL, 10);
    toks_info info;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    toks_get_info(ctx, &info);
    /* literals: the command line's, then markup tokens of the vocabulary */
    uint8_t *lit[512];
    uint64_t litn[512];
    uint32_t nl = 0;
    for (int a = 4; a < argc && nl < 512u; a++) {
        lit[nl] = (uint8_t *)malloc(strlen(argv[a]) + 1u);
        litn[nl] = unescape(argv[a], lit[nl]);
        nl++;
    }
    for (uint32_t id = 0; id < info.n_ids && nl < 512u; id++) {
        uint64_t k = 0;
        const uint8_t *p = toks_token(ctx, id, &k);
        if (p != NULL && k >= 3u && k <= 64u && ((p[0] == '<' && p[k - 1] == '>') || (p[0] == '[' && p[k - 1] == ']'))) {
            lit[nl] = (uint8_t *)(uintptr_t)p;
            litn[nl] = k;
            nl++;
        }
    }
    uint64_t maxlen = 4096u, sb = toks_scratch_bytes(ctx, maxlen, 0u);
    void *scr = malloc(sb);
    uint8_t *text = (uint8_t *)malloc(maxlen);
    uint32_t *out = (uint32_t *)malloc(3u * maxlen * 4u + 64u);
    if (toks_scratch_init(ctx, scr, sb, 0u) != 0) { return 1; }
    uint64_t s = seed;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t len = 0, target = 1u + rng(&s) % (rng(&s) % 8u == 0u ? 4000u : 80u);
        while (len < target) {
            uint64_t k = rng(&s) % 16u;
            const uint8_t *p;
            uint64_t m;
            if (k < 7u) { p = (const uint8_t *)WS[rng(&s) % (sizeof WS / sizeof WS[0])]; m = strlen((const char *)p); }
            else if (k < 11u && nl != 0u) {
                uint32_t j = (uint32_t)(rng(&s) % nl);
                p = lit[j];
                m = litn[j];
                if (rng(&s) % 4u == 0u && m > 1u) { m = 1u + rng(&s) % (m - 1u); }   /* cut short */
            } else { p = (const uint8_t *)OTHER[rng(&s) % (sizeof OTHER / sizeof OTHER[0])]; m = strlen((const char *)p); }
            uint64_t rep = rng(&s) % 8u == 0u ? 1u + rng(&s) % 40u : 1u;
            for (uint64_t q = 0; q < rep && len + m <= maxlen; q++) { memcpy(text + len, p, m); len += m; }
            if (len + m > maxlen) { break; }
        }
        uint64_t h = 0xCBF29CE484222325ull;
        for (uint32_t f = 0; f < 6u; f++) {
            uint32_t flags = (f % 3u) | (f >= 3u ? TOKS_NO_POSTPROCESS : 0u);
            int64_t c = toks_encode(ctx, text, len, flags, out, 3u * maxlen, scr);
            h = fnv(h, &c, 8);
            if (c > 0) { h = fnv(h, out, (uint64_t)c * 4u); }
            int64_t e = toks_pieces(ctx, text, len, flags, out, 3u * maxlen, scr);
            h = fnv(h, &e, 8);
            if (e > 0) { h = fnv(h, out, (uint64_t)e * 4u); }
        }
        printf("%llu %016llx\n", (unsigned long long)i, (unsigned long long)h);
    }
    toks_unload(ctx);
    return 0;
}
