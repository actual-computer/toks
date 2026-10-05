/*
 * nfc_check.c: the NFC normalizer (src/core/norm.c) against hf tokenizers 0.23.2 at scale. Not part of
 * `make test`: tests/norm/run.sh pipes tests/norm/gen.py's records into it on a lab host.
 *
 * stdin: records of u32 n_in, in[n_in], u32 n_out, out[n_out] (little-endian; out = hf NFC(in)).
 * Each record is normalized (1) whole, (2) in parts cut at boundary atoms (every 1..300 bytes or later,
 * bytes apart, then the flush), (3) whole from a copy at an odd address behind a random prefix of '#' bytes
 * (so K2's 64-byte blocks fall at other positions); each result must equal hf's bytes exactly.
 * argv[1] names the family for the report. Exit 0 iff 0 mismatches.
 */
#include "../../src/core/norm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rng = rng * 6364136223846793005ull + 1442695040888963407ull; return rng >> 33; }

static int read_u32(uint32_t *v)
{
    uint8_t b[4];
    if (fread(b, 1, 4, stdin) != 4) { return 0; }
    *v = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 1;
}

static void report(const char *fam, uint64_t rec, const char *how, const uint8_t *in, size_t n_in,
                   const uint8_t *want, size_t n_want, const uint8_t *got, size_t n_got)
{
    size_t d = 0;
    while (d < n_want && d < n_got && want[d] == got[d]) { d++; }
    fprintf(stderr, "MISMATCH %s record %llu (%s): in %zu bytes, hf %zu, toks %zu, first difference at out %zu\n",
            fam, (unsigned long long)rec, how, n_in, n_want, n_got, d);
    size_t a = d > 24 ? d - 24 : 0;
    fprintf(stderr, "  hf  :");
    for (size_t i = a; i < d + 24 && i < n_want; i++) { fprintf(stderr, " %02x", want[i]); }
    fprintf(stderr, "\n  toks:");
    for (size_t i = a; i < d + 24 && i < n_got; i++) { fprintf(stderr, " %02x", got[i]); }
    fprintf(stderr, "\n  in (from %zu):", a);
    for (size_t i = a; i < a + 64 && i < n_in; i++) { fprintf(stderr, " %02x", in[i]); }
    fputc('\n', stderr);
}

int main(int argc, char **argv)
{
    const char *fam = argc > 1 ? argv[1] : "?";
    uint64_t recs = 0, bytes_in = 0, bad = 0;
    size_t cap_in = 1 << 20, cap_out = 3u << 20;
    uint8_t *in = malloc(cap_in + 128), *want = malloc(cap_out), *got = malloc(3 * cap_in + 64);
    uint32_t n_in, n_out;
    while (read_u32(&n_in)) {
        if (n_in > cap_in) {
            while (cap_in < n_in) { cap_in *= 2; }
            in = realloc(in, cap_in + 128);
            got = realloc(got, 3 * cap_in + 64);
        }
        if (fread(in, 1, n_in, stdin) != n_in || !read_u32(&n_out)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (n_out > cap_out) { while (cap_out < n_out) { cap_out *= 2; } want = realloc(want, cap_out); }
        if (fread(want, 1, n_out, stdin) != n_out) { fprintf(stderr, "truncated stream\n"); return 2; }
        recs++;
        bytes_in += n_in;

        int64_t r = toks_norm(TOKS_NS_NFC, in, n_in, got, 3 * (uint64_t)n_in);
        if (r != (int64_t)n_out || memcmp(got, want, n_out) != 0) {
            if (bad++ < 10) { report(fam, recs, "whole", in, n_in, want, n_out, got, r < 0 ? 0 : (size_t)r); }
            continue;
        }
        size_t o = 0, e = 0, from = 0;
        int ok = 1;
        for (;;) {                                     /* parts cut at random boundary atoms, each alone */
            e += 1 + (size_t)(rnd() % 300);
            if (e > n_in) { e = n_in; }
            while (e < n_in && ((in[e] & 0xC0) == 0x80 || !toks_nfc_boundary(TOKS_NS_NFC, in, n_in, e))) { e++; }
            r = toks_norm(TOKS_NS_NFC, in + from, e - from, got + o, 3 * (uint64_t)(e - from));
            if (r < 0) { ok = 0; break; }
            o += (size_t)r;
            from = e;
            if (e == n_in) { break; }
        }
        if (!ok || o != n_out || memcmp(got, want, n_out) != 0) {
            if (bad++ < 10) { report(fam, recs, "chunked", in, n_in, want, n_out, got, o); }
            continue;
        }
        size_t pad = (size_t)(rnd() % 64);             /* odd address, K2 blocks shifted */
        uint8_t *shifted = malloc(n_in + pad + 1);
        memset(shifted + 1, '#', pad);
        memcpy(shifted + 1 + pad, in, n_in);
        uint8_t *g2 = malloc(3 * (n_in + pad) + 64);
        r = toks_norm(TOKS_NS_NFC, shifted + 1, n_in + pad, g2, 3 * (uint64_t)(n_in + pad));
        if (r != (int64_t)(n_out + pad) || memcmp(g2 + pad, want, n_out) != 0) {
            if (bad++ < 10) { report(fam, recs, "shifted", in, n_in, want, n_out, g2 + pad, r < (int64_t)pad ? 0 : (size_t)r - pad); }
        }
        free(shifted);
        free(g2);
    }
    printf("nfc_check %s: %llu records, %llu input bytes, %llu mismatches\n", fam, (unsigned long long)recs,
           (unsigned long long)bytes_in, (unsigned long long)bad);
    return bad != 0;
}
