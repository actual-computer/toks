/*
 * tests/unigram/e2e_check.c: the toks side of the Unigram end-to-end differential. Reads tests/unigram/e2e_gen.py's
 * stream (hf 0.23.2's ids and decodes) and runs every case through include/toks.h: toks_load, toks_encode in the
 * case's flags, toks_decode. Exit 0 iff nothing differs. NOT part of `make test`; built by hand against the library
 * (`make lib` and `mkdir -p build/unigram-e2e` first), then fed tests/unigram/e2e_gen.py's stream:
 *
 *   clang -std=c17 -O2 -Iinclude -o build/unigram-e2e/e2e_check tests/unigram/e2e_check.c build/<os>-<isa>/libtoks.a
 *   e2e_check <tokenizer.json> < stream
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toks.h"

static int rd(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }

static void esc(const uint8_t *s, uint32_t n)
{
    for (uint32_t i = 0; i < n && i < 400; i++) {
        uint8_t c = s[i];
        if (c >= 0x20 && c < 0x7F && c != '\\') { fputc(c, stderr); }
        else { fprintf(stderr, "\\x%02x", c); }
    }
    if (n > 400) { fprintf(stderr, "..."); }
}

static void ids(const char *what, const uint32_t *v, uint64_t n)
{
    fprintf(stderr, "  %s (%" PRIu64 "):", what, n);
    for (uint64_t i = 0; i < n && i < 48; i++) { fprintf(stderr, " %u", v[i]); }
    fprintf(stderr, "%s\n", n > 48 ? " ..." : "");
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: e2e_check <tokenizer.json> < stream\n"); return 2; }
    char magic[8];
    if (!rd(magic, 8) || memcmp(magic, "TKUE2E01", 8) != 0) { fprintf(stderr, "bad stream\n"); return 2; }
    toks_ctx *ctx = NULL;
    toks_diag dg;
    memset(&dg, 0, sizeof dg);
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &dg;
    int64_t r = toks_load(&ctx, argv[1], &o);
    if (r != 0) { fprintf(stderr, "toks_load: %" PRId64 " %s\n", r, dg.what); return 2; }
    uint64_t max_len = 1u << 20;
    uint64_t sb = toks_scratch_bytes(ctx, max_len, 0);
    void *scr = malloc(sb);
    if (scr == NULL || toks_scratch_init(ctx, scr, sb, 0) != 0) { fprintf(stderr, "scratch\n"); return 2; }
    uint64_t tcap = 1u << 16, icap = 1u << 16, dcap = 1u << 20;
    uint8_t *text = malloc(tcap);
    uint32_t *want = malloc(4u * icap), *got = malloc(4u * icap);
    uint8_t *dwant = malloc(dcap), *dgot = malloc(dcap);
    uint64_t n_enc = 0, n_dec = 0, n_pc = 0, bad = 0;
    uint32_t hdr[3];
    while (rd(hdr, sizeof hdr)) {
        uint32_t op = hdr[0], fl = hdr[1], tl = hdr[2], n = 0;
        if (tl > tcap) { tcap = tl; text = realloc(text, tcap); }
        if (tl && !rd(text, tl)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (!rd(&n, 4)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (n > icap) { icap = n; want = realloc(want, 4u * icap); got = realloc(got, 4u * icap); }
        if (n && !rd(want, 4u * (size_t)n)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (op == 0 || op == 2) {
            if (op == 0) { n_enc++; } else { n_pc++; }
            if (tl > max_len) { fprintf(stderr, "text over %" PRIu64 " bytes\n", max_len); return 2; }
            int64_t m = (op == 0) ? toks_encode(ctx, text, tl, fl, got, icap, scr) : toks_pieces(ctx, text, tl, fl, got, icap, scr);
            if (m != (int64_t)n || (n && memcmp(got, want, 4u * (size_t)n) != 0)) {
                if (bad < 12) {
                    fprintf(stderr, "%s MISMATCH flags %u text \"", op == 0 ? "ENCODE" : "PIECES", fl);
                    esc(text, tl);
                    fprintf(stderr, "\"\n");
                    ids("hf", want, n);
                    if (m >= 0) { ids("toks", got, (uint64_t)m < icap ? (uint64_t)m : icap); }
                    else { fprintf(stderr, "  toks error %" PRId64 "\n", m); }
                }
                bad++;
            }
        } else {
            n_dec++;
            uint32_t dl = 0;
            if (!rd(&dl, 4)) { fprintf(stderr, "truncated stream\n"); return 2; }
            if (dl > dcap) { dcap = dl; dwant = realloc(dwant, dcap); dgot = realloc(dgot, dcap); }
            if (dl && !rd(dwant, dl)) { fprintf(stderr, "truncated stream\n"); return 2; }
            int64_t m = toks_decode(ctx, want, n, fl ? TOKS_SKIP_SPECIAL : 0u, dgot, dcap);
            if (m != (int64_t)dl || (dl && memcmp(dgot, dwant, dl) != 0)) {
                if (bad < 12) {
                    fprintf(stderr, "DECODE MISMATCH skip %u\n", fl);
                    ids("ids", want, n);
                    fprintf(stderr, "  hf   \"");
                    esc(dwant, dl);
                    fprintf(stderr, "\"\n  toks \"");
                    if (m >= 0) { esc(dgot, (uint32_t)((uint64_t)m < dcap ? (uint64_t)m : dcap)); }
                    fprintf(stderr, "\" (%" PRId64 ")\n", m);
                }
                bad++;
            }
        }
    }
    printf("e2e %s: %" PRIu64 " encodes, %" PRIu64 " pieces, %" PRIu64 " decodes, %" PRIu64 " mismatches\n", argv[1],
           n_enc, n_pc, n_dec, bad);
    toks_unload(ctx);
    return bad ? 1 : 0;
}
