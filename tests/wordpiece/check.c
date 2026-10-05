/*
 * tests/wordpiece/check.c: the wordpiece differential's c side. NOT part of `make test`: tests/wordpiece/run.sh
 * builds it and feeds it tests/wordpiece/gen.py's stream (hf 0.23.2's normalized text, pieces and model ids).
 *
 *   check <tokenizer.json>  < cases.bin
 *
 * The tokenizer.json goes through toks_config_parse (tests/common/wp_json.h) + toks_wp_build (the load path the driver
 * will call). Then for every case:
 *   norm     toks_norm(the BertNormalizer's steps, text) == hf normalizer(text)
 *   scan     toks_wp_scan_c(text) in one call: the pieces, ascii-folded for an uncased tokenizer, == hf's
 *            pre-tokenizer words (a piece flagged OVER: hf's word has more than max_input_chars_per_word chars)
 *   chunks   the same scan in calls of cap 2..9 pieces with a mat buffer of exactly toks_wp_mat_min: the same
 *            pieces
 *   pretok   toks_wp_scan_c over hf's normalized text with the flags masked to 0 (the pre-tokenizer alone)
 *   ids      toks_wp_encode_c over the scan's pieces == hf encode(text).ids of the model alone
 * Exit 0 iff nothing differs. Counts on stdout.
 */
#include "../../src/core/wp.h"
#include "../../src/core/json.h"
#include "../common/wp_json.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 65536u

static void *xalloc(uint64_t n)
{
    void *p = malloc(n ? n : 1);
    if (p == NULL) { fprintf(stderr, "out of memory\n"); exit(2); }
    return p;
}

static int rd(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }

static uint8_t *rd_bytes(uint32_t *n)
{
    if (!rd(n, 4)) { return NULL; }
    if (*n == 0xFFFFFFFFu) { return NULL; }
    uint8_t *b = xalloc(*n + 1u);
    if (*n && !rd(b, *n)) { fprintf(stderr, "truncated stream\n"); exit(2); }
    return b;
}

static void esc(const char *what, const uint8_t *p, uint64_t n)
{
    fprintf(stderr, "  %s:", what);
    for (uint64_t i = 0; i < n && i < 160; i++) {
        if (p[i] >= 0x20 && p[i] < 0x7F) { fputc(p[i], stderr); } else { fprintf(stderr, "\\x%02x", p[i]); }
    }
    fputc('\n', stderr);
}

static uint64_t nchars(const uint8_t *p, uint64_t n)
{
    uint64_t c = 0;
    for (uint64_t i = 0; i < n; i++) { if ((p[i] & 0xC0u) != 0x80u) { c++; } }
    return c;
}

/* the piece's normalized bytes (ascii-folded when the tokenizer lowercases) */
static uint64_t piece_bytes(const toks_wp_tables *t, const uint8_t *text, const uint8_t *mat, const toks_wp_piece *pc,
                            uint8_t *o)
{
    const uint8_t *b = (pc->flags & TOKS_WPP_MAT) ? mat : text;
    for (uint32_t i = 0; i < pc->len; i++) {
        uint8_t x = b[pc->off + i];
        if ((t->flags & TOKS_WPF_LOWER) && x >= 'A' && x <= 'Z') { x = (uint8_t)(x + 32); }
        o[i] = x;
    }
    return pc->len;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: check <tokenizer.json> < cases.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long fl = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *src = xalloc((uint64_t)fl);
    if (fread(src, 1, (size_t)fl, f) != (size_t)fl) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);
    toks_arena ar = { xalloc(64u * (uint64_t)fl + (1u << 20)), 64u * (uint64_t)fl + (1u << 20), 0 };
    toks_wp_vocab v;
    toks_wp_params p;
    toks_err err;
    int64_t rc = wp_from_json(src, (uint64_t)fl, &ar, &v, &p, &err);
    if (rc != 0) { fprintf(stderr, "wp_from_json: %" PRId64 " %s\n", rc, err.what ? err.what : ""); return 3; }
    uint64_t tb = toks_wp_tables_bytes(&v);
    toks_arena tar = { xalloc(tb), tb, 0 };
    toks_wp_tables t;
    rc = toks_wp_build(&t, &tar, &v, &p, &err);
    if (rc != 0) { fprintf(stderr, "toks_wp_build: %" PRId64 " %s\n", rc, err.what ? err.what : ""); return 3; }
    printf("loaded %s: vocab %u, flags 0x%x, max_chars %u, unk %u, word_maxlen %u, cont_maxlen %u, table bytes %"
           PRIu64 " (used %" PRIu64 ")\n", argv[1], v.n, t.flags, t.max_chars, t.unk_id, t.word_maxlen,
           t.cont_maxlen, tb, tar.pos);

    char magic[4];
    if (!rd(magic, 4) || memcmp(magic, "WPC1", 4) != 0) { fprintf(stderr, "bad stream\n"); return 2; }
    uint64_t mat_min = toks_wp_mat_min(&t);
    toks_wp_piece *pieces = xalloc(MAXP * sizeof(toks_wp_piece));
    toks_wp_piece *pieces2 = xalloc(MAXP * sizeof(toks_wp_piece));
    uint64_t n_case = 0, n_bad = 0, n_pieces = 0, n_mat = 0, n_ids = 0, hits = 0, misses = 0;
    uint64_t bad_norm = 0, bad_scan = 0, bad_chunk = 0, bad_pre = 0, bad_ids = 0;
    for (;;) {
        uint32_t tl, nl;
        uint8_t *text = rd_bytes(&tl);
        if (text == NULL) { break; }
        uint8_t *norm = rd_bytes(&nl);
        uint32_t nw;
        if (!rd(&nw, 4)) { fprintf(stderr, "truncated\n"); return 2; }
        uint8_t **w = xalloc((uint64_t)nw * sizeof(*w) + 8u);
        uint32_t *wl = xalloc((uint64_t)nw * 4u + 8u);
        for (uint32_t k = 0; k < nw; k++) { w[k] = rd_bytes(&wl[k]); }
        uint32_t ni;
        if (!rd(&ni, 4)) { fprintf(stderr, "truncated\n"); return 2; }
        uint32_t *ids = xalloc((uint64_t)ni * 4u + 8u);
        if (ni && !rd(ids, (size_t)ni * 4u)) { fprintf(stderr, "truncated\n"); return 2; }
        n_case++;
        int bad = 0;

        /* norm */
        uint64_t cap = 3u * (uint64_t)tl + 16u;
        uint8_t *nb = xalloc(cap);
        int64_t r = toks_norm(t.flags, text, tl, nb, cap);
        uint64_t got = r < 0 ? 0u : (uint64_t)r;
        if (got != nl || memcmp(nb, norm, nl) != 0) {
            bad_norm++; bad = 1;
            if (bad_norm <= 5) { esc("NORM text", text, tl); esc("hf  ", norm, nl); esc("toks", nb, got); }
        }

        /* scan, one call */
        uint64_t mcap = 3u * (uint64_t)tl + mat_min + 64u;
        uint8_t *mat = xalloc(mcap);
        toks_wp_scan_args sa;
        memset(&sa, 0, sizeof sa);
        sa.text = text; sa.len = tl; sa.pos = 0; sa.pieces = pieces; sa.cap = MAXP; sa.mat = mat; sa.mat_cap = mcap;
        sa.flags = ~(uint64_t)0;
        uint64_t np = toks_wp_scan_c(&t, &sa);
        if (sa.pos != tl) { fprintf(stderr, "scan stopped at %" PRIu64 " of %u\n", sa.pos, tl); return 4; }
        uint8_t *pb = xalloc(4u * (uint64_t)tl + mat_min + 16u);
        int scan_ok = np == nw;
        for (uint64_t k = 0; scan_ok && k < np; k++) {
            if (pieces[k].flags & TOKS_WPP_OVER) {
                if (nchars(w[k], wl[k]) <= t.max_chars) { scan_ok = 0; }
                continue;
            }
            uint64_t bl = piece_bytes(&t, text, mat, &pieces[k], pb);
            if (bl != wl[k] || memcmp(pb, w[k], bl) != 0) { scan_ok = 0; }
            if (pieces[k].flags & TOKS_WPP_MAT) { n_mat++; }
        }
        if (!scan_ok) {
            bad_scan++; bad = 1;
            if (bad_scan <= 5) {
                esc("SCAN text", text, tl);
                fprintf(stderr, "  hf %u words, toks %" PRIu64 " pieces\n", nw, np);
                int shown = 0;
                for (uint64_t k = 0; k < np && k < nw && shown < 12; k++) {
                    uint64_t bl = piece_bytes(&t, text, mat, &pieces[k], pb);
                    int same = (pieces[k].flags & TOKS_WPP_OVER) ? nchars(w[k], wl[k]) > t.max_chars
                                                                 : bl == wl[k] && memcmp(pb, w[k], bl) == 0;
                    if (same) { continue; }
                    shown++;
                    fprintf(stderr, "   piece %" PRIu64 " flags %u", k, pieces[k].flags);
                    esc("", pb, bl);
                    esc("   hf", w[k], wl[k]);
                }
            }
        }
        n_pieces += np;

        /* scan in chunks with a minimal mat buffer */
        uint64_t chunk_cap = 2u + n_case % 8u;
        uint8_t *mat2 = xalloc(mat_min);
        uint64_t pos = 0, k2 = 0;
        int chunk_ok = 1;
        for (uint64_t round = 0; pos < tl || round == 0; round++) {
            if (round > 4u * (uint64_t)tl + 8u) { chunk_ok = 0; break; }
            toks_wp_scan_args c;
            memset(&c, 0, sizeof c);
            c.text = text; c.len = tl; c.pos = pos; c.pieces = pieces2; c.cap = chunk_cap; c.mat = mat2;
            c.mat_cap = mat_min; c.flags = ~(uint64_t)0;
            uint64_t m = toks_wp_scan_c(&t, &c);
            for (uint64_t j = 0; j < m; j++, k2++) {
                if (k2 >= np) { chunk_ok = 0; break; }
                uint8_t q1[4096], q2[4096];
                if (pieces2[j].len > sizeof q1 || pieces[k2].len > sizeof q2) { continue; }
                uint64_t l1 = piece_bytes(&t, text, mat2, &pieces2[j], q1);
                uint64_t l2 = piece_bytes(&t, text, mat, &pieces[k2], q2);
                if ((pieces2[j].flags ^ pieces[k2].flags) & (TOKS_WPP_OVER | TOKS_WPP_INVALID)) { chunk_ok = 0; }
                if (!(pieces[k2].flags & TOKS_WPP_OVER) && (l1 != l2 || memcmp(q1, q2, l1) != 0)) { chunk_ok = 0; }
            }
            if (c.pos == pos && m == 0u && pos < tl) { chunk_ok = 0; break; }
            pos = c.pos;
            if (tl == 0u) { break; }
        }
        if (k2 != np) { chunk_ok = 0; }
        if (!chunk_ok) {
            bad_chunk++; bad = 1;
            if (bad_chunk <= 5) { esc("CHUNK text", text, tl); }
        }

        /* the pre-tokenizer alone over hf's normalized text */
        toks_wp_scan_args pa;
        memset(&pa, 0, sizeof pa);
        uint64_t mcap3 = 3u * (uint64_t)nl + mat_min + 64u;
        uint8_t *mat3 = xalloc(mcap3);
        pa.text = norm; pa.len = nl; pa.pieces = pieces2; pa.cap = MAXP; pa.mat = mat3; pa.mat_cap = mcap3;
        pa.flags = 0;
        uint64_t np3 = toks_wp_scan_c(&t, &pa);
        int pre_ok = np3 == nw;
        for (uint64_t k = 0; pre_ok && k < np3; k++) {
            if (pieces2[k].flags & TOKS_WPP_OVER) { continue; }
            uint64_t bl = piece_bytes(&t, norm, mat3, &pieces2[k], pb);
            if (bl != wl[k] || memcmp(pb, w[k], bl) != 0) { pre_ok = 0; }
        }
        if (!pre_ok) {
            bad_pre++; bad = 1;
            if (bad_pre <= 5) { esc("PRETOK norm", norm, nl); }
        }

        /* ids */
        uint32_t *out = xalloc(4u * ((uint64_t)tl + np) + 16u);
        toks_wp_encode_args ea;
        memset(&ea, 0, sizeof ea);
        ea.text = text; ea.mat = mat; ea.pieces = pieces; ea.n = np; ea.out = out; ea.room = (uint64_t)tl + np;
        uint64_t no = toks_wp_encode_c(&t, &ea);
        hits += ea.hits;
        misses += ea.misses;
        n_ids += no;
        if (no != ni || (ni && memcmp(out, ids, (size_t)ni * 4u) != 0)) {
            bad_ids++; bad = 1;
            if (bad_ids <= 5) {
                esc("IDS text", text, tl);
                fprintf(stderr, "  hf  ");
                for (uint32_t k = 0; k < ni && k < 24; k++) { fprintf(stderr, " %u", ids[k]); }
                fprintf(stderr, "\n  toks");
                for (uint64_t k = 0; k < no && k < 24; k++) { fprintf(stderr, " %u", out[k]); }
                fprintf(stderr, "\n");
            }
        }
        n_bad += (uint64_t)bad;
        free(out); free(mat3); free(mat2); free(pb); free(mat); free(nb);
        for (uint32_t k = 0; k < nw; k++) { free(w[k]); }
        free(w); free(wl); free(ids); free(text); free(norm);
    }
    printf("cases %" PRIu64 ": pieces %" PRIu64 " (copied %" PRIu64 "), ids %" PRIu64 ", whole-piece hits %" PRIu64
           ", greedy %" PRIu64 "; mismatches: norm %" PRIu64 " scan %" PRIu64 " chunks %" PRIu64 " pretok %" PRIu64
           " ids %" PRIu64 " (cases %" PRIu64 ")\n", n_case, n_pieces, n_mat, n_ids, hits, misses, bad_norm,
           bad_scan, bad_chunk, bad_pre, bad_ids, n_bad);
    return n_bad ? 1 : 0;
}
