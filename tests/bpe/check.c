/*
 * tests/bpe/check.c: the bpe differential's c side. NOT part of `make test`: tests/bpe/run.sh builds it
 * and feeds it tests/bpe/gen.py's stream (pieces + the ids hf 0.23.2's BPE model gives for them).
 *
 *   check <tokenizer.json>  < cases.bin
 *
 * The stream starts with the model as python derives it from the file (gen.py: raw bytes per id, merges
 * as vocab[a], vocab[b], vocab[a + b] triples, ignore_merges). The file itself then goes through the
 * real load path, toks_config_parse + toks_compile, whose merge ids, n_ids and token bytes must equal
 * the stream's; toks_bpe_build runs on that ctx->t and cfg in an arena of exactly
 * toks_bpe_tables_bytes(cfg). (When config / compile refuse the file the tables come from the stream's
 * model and the log says so; when they load a model that differs from the stream's -- dsv3's, granite's
 * and modernbert's today -- its words table is still built and re-certified as below, and the run fails
 * there without a piece compared.) Then for every record: K6 on the piece against hf's ids (under
 * ignore_merges and TOKS_TF_PROBE_LONG, K6 leaves the whole-piece rule of a piece of 2..15 bytes to
 * K5's words: there the words entry answers it); and K5 over batches of consecutive pieces (<= 256
 * pieces, <= 64 KiB) with a 1024-bucket cache that lives for the whole run, against the batch's
 * concatenated ids. Every words entry is re-certified first: against K6, under ignore_merges against
 * the whole-piece rule (its key is a model token: [that id]), and where K5 reads it (in its second
 * bucket only behind its first bucket's spill bit; no other bit in a meta byte). Exit 0 iff nothing
 * differs.
 */
#include "../../src/core/bpe.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* the tier under test: the c twins by default; for an asm tier, build this file with
 * -DK6_BPE=toks_k6_bpe_<tier> -DK5_ENCODE=toks_k5_encode_<tier> plus its .S files */
#ifndef K6_BPE
#define K6_BPE toks_k6_bpe_c
#endif
#ifndef K5_ENCODE
#define K5_ENCODE toks_k5_encode_c
#endif
uint64_t K6_BPE(const toks_tables *t, toks_k6_args *a);
uint64_t K5_ENCODE(const toks_tables *t, toks_k5_args *a);

#define BATCH_PIECES 256u
#define BATCH_BYTES  65536u

static double now_ms(void) { return 1000.0 * (double)clock() / CLOCKS_PER_SEC; }

static void *xalloc(uint64_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(uint64_t)63);
    if (p == NULL) { fprintf(stderr, "out of memory (%" PRIu64 " bytes)\n", n); exit(2); }
    return p;
}

static int rd(void *p, size_t n) { return fread(p, 1, n, stdin) == n; }

static void hexdump(const char *what, const uint8_t *p, uint64_t n)
{
    fprintf(stderr, "  %s (%" PRIu64 " bytes):", what, n);
    for (uint64_t i = 0; i < n && i < 48; i++) { fprintf(stderr, " %02x", p[i]); }
    fprintf(stderr, "%s\n", n > 48 ? " ..." : "");
}

static void ids(const char *what, const uint32_t *v, uint64_t n)
{
    fprintf(stderr, "  %s (%" PRIu64 "):", what, n);
    for (uint64_t i = 0; i < n && i < 32; i++) { fprintf(stderr, " %u", v[i]); }
    fprintf(stderr, "%s\n", n > 32 ? " ..." : "");
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: check <tokenizer.json> < cases.bin\n"); return 2; }

    /* ---- the model, as toks_load hands it to toks_bpe_build ---- */
    char magic[8];
    uint32_t hdr[4];
    if (!rd(magic, 8) || memcmp(magic, "TOKSBPE1", 8) != 0 || !rd(hdr, sizeof hdr)) { fprintf(stderr, "bad stream header\n"); return 2; }
    uint32_t n_vocab = hdr[0], n_ids = hdr[1], n_merges = hdr[2], n_rawless = 0, differ = 0;
    uint32_t *tok_off = xalloc(4 * ((uint64_t)n_ids + 1));
    uint64_t cap = 1u << 20, o = 0;
    uint8_t *tok_bytes = malloc(cap);
    for (uint32_t id = 0; id < n_ids; id++) {
        uint32_t l;
        if (!rd(&l, 4)) { fprintf(stderr, "truncated model\n"); return 2; }
        while (o + l > cap) { cap *= 2; tok_bytes = realloc(tok_bytes, cap); }
        if (tok_bytes == NULL || !rd(tok_bytes + o, l)) { fprintf(stderr, "truncated model\n"); return 2; }
        tok_off[id] = (uint32_t)o;
        o += l;
    }
    tok_off[n_ids] = (uint32_t)o;
    uint32_t *trip = xalloc(12 * (uint64_t)n_merges + 4), *ml = xalloc(4 * (uint64_t)n_merges + 4);
    uint32_t *mr = xalloc(4 * (uint64_t)n_merges + 4), *mo = xalloc(4 * (uint64_t)n_merges + 4);
    if (!rd(trip, 12 * (size_t)n_merges)) { fprintf(stderr, "truncated merges\n"); return 2; }
    for (uint32_t i = 0; i < n_merges; i++) { ml[i] = trip[3 * i]; mr[i] = trip[3 * i + 1]; mo[i] = trip[3 * i + 2]; }
    toks_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.n_vocab = n_vocab;
    cfg.n_merges = n_merges;
    cfg.m_left_id = ml;
    cfg.m_right_id = mr;
    cfg.m_out_id = mo;
    cfg.ignore_merges = (uint8_t)hdr[3];
    double t0;
    int64_t r;
    toks_tables t;
    memset(&t, 0, sizeof t);
    t.n_ids = n_ids;
    t.tok_off = tok_off;
    t.tok_bytes = tok_bytes;

    /* ---- the real load path on the same file ---- */
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    uint64_t flen = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *file = xalloc(flen);
    if (fread(file, 1, flen, f) != flen) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(f);
    toks_config rc;
    toks_err err;
    uint64_t plen = 16 * flen + (64u << 20);
    toks_arena par = { xalloc(plen), plen, 0 };
    struct toks_ctx *ctx = calloc(1, sizeof *ctx);
    t0 = now_ms();
    r = toks_config_parse(file, flen, &par, &rc, &err);
    if (r == 0) { r = toks_compile(&rc, ctx, &par); err.what = r == 0 ? NULL : "toks_compile"; }
    double t_load = now_ms() - t0;
    if (r != 0) {
        printf("load path: refused (%" PRId64 " %s); tables from the stream's model\n", r, err.what ? err.what : "");
    } else {
        int same = rc.n_vocab == n_vocab && rc.n_merges == n_merges && ctx->t.n_ids == n_ids
                   && rc.ignore_merges == (uint8_t)hdr[3] && memcmp(rc.m_left_id, ml, 4 * (size_t)n_merges) == 0
                   && memcmp(rc.m_right_id, mr, 4 * (size_t)n_merges) == 0 && memcmp(rc.m_out_id, mo, 4 * (size_t)n_merges) == 0;
        for (uint32_t id = 0; same && id < n_vocab; id++) {
            uint32_t a = ctx->t.tok_off[id], l = ctx->t.tok_off[id + 1] - a;
            same = l == tok_off[id + 1] - tok_off[id] && memcmp(ctx->t.tok_bytes + a, tok_bytes + tok_off[id], l) == 0;
        }
        printf("load path: toks_config_parse + toks_compile %.0f ms, merge ids / n_ids / token bytes %s the stream's model\n",
               t_load, same ? "equal" : "DIFFER FROM");
        differ = !same;                                    /* its words table is still certified below */
        t = ctx->t;
        cfg = rc;
    }
    uint64_t need = toks_bpe_tables_bytes(&cfg);
    toks_arena bar = { xalloc(need), need, 0 };
    t0 = now_ms();
    r = toks_bpe_build(&t, &bar, &cfg);
    double t_build = now_ms() - t0;
    for (uint32_t id = 0; id < cfg.n_vocab; id++) { n_rawless += t.tok_off[id + 1] == t.tok_off[id]; }
    if (r != 0) { fprintf(stderr, "toks_bpe_build: %" PRId64 " in an arena of exactly %" PRIu64 " bytes\n", r, need); return 1; }
    printf("load %s: n_vocab %u, n_ids %u, merges %u, rawless %u, flags %#x; bpe build %.0f ms; "
           "arena %" PRIu64 " of exactly %" PRIu64 " bytes\n", argv[1], cfg.n_vocab, t.n_ids, cfg.n_merges, n_rawless,
           t.flags, t_build, bar.pos, need);

    /* ---- the words table, re-certified ---- */
    static _Alignas(64) uint8_t w6[TOKS_BPE_WORK_BYTES(15)];
    uint64_t placed = 0, bad_words = 0, bad_spill = 0, bad_meta = 0, cands = 0, spills = 0, away = 0;
    for (uint64_t bu = 0; t.words != NULL && bu <= t.words_mask; bu++) {
        const uint8_t *b = t.words + bu * TOKS_BUCKET;
        bad_meta += (b[TOKS_W3_META] & ~TOKS_W3_SPILL) != 0;  /* the meta byte holds the spill bit only */
        spills += (b[TOKS_W3_META] & TOKS_W3_SPILL) != 0;     /* a miss here reads the second bucket too */
        for (uint32_t way = 0; way < 3; way++) {
            const uint8_t *key = b + way * 16;
            uint32_t got[2], want[15];
            if (key[15] == 0) { continue; }
            uint64_t gn = bpe_w3_val(b + TOKS_W3_VAL + 5 * way, got);
            placed++;
            /* an entry in its second bucket: its first bucket's spill bit is set, or K5 never reads it */
            bpe_key k;
            memcpy(&k.lo, key, 8);
            memcpy(&k.hi, key + 8, 8);
            uint64_t home = (uint64_t)bpe_key_hash(k) & t.words_mask;
            away += home != bu;
            bad_spill += home != bu && (t.words[home * TOKS_BUCKET + TOKS_W3_META] & TOKS_W3_SPILL) == 0;
            toks_k6_args a = { key, key[15], want, w6, sizeof w6, 0, 0, 0 };
            uint64_t n = 1;
            if (cfg.ignore_merges == 0u || !bpe_vhash_find(&t, key, key[15], want)) { n = K6_BPE(&t, &a); }
            if (gn != n || got[0] != want[0] || (n == 2 && got[1] != want[1])) { bad_words++; }
        }
    }
    for (uint32_t id = 0; id < cfg.n_vocab; id++) {
        uint32_t l = t.tok_off[id + 1] - t.tok_off[id], out[15];
        if (l < 2 || l > 15) { continue; }
        toks_k6_args a = { t.tok_bytes + t.tok_off[id], l, out, w6, sizeof w6, 0, 0, 0 };
        cands += bpe_w3_fits(out, K6_BPE(&t, &a));
    }
    printf("words: %" PRIu64 " entries of %" PRIu64 " candidates (%" PRIu64 " buckets, %" PRIu64 " with the spill bit; %"
           PRIu64 " entries in their second), %" PRIu64 " not K6's answer, %" PRIu64 " in a second bucket without the"
           " first's spill bit, %" PRIu64 " meta bytes with other bits\n",
           placed, cands, t.words_mask + 1, spills, away, bad_words, bad_spill, bad_meta);
    bad_words += bad_spill + bad_meta;                    /* each fails the run below */
    if (differ) {
        printf("RESULT %s: FAIL (the load path's model differs from the stream's: no piece compared)\n", argv[1]);
        return 1;
    }

    /* ---- the stream ---- */
    uint64_t piece_cap = 1u << 20, work_cap = TOKS_BPE_WORK_BYTES(1u << 20);
    uint8_t *piece = xalloc(piece_cap), *work = xalloc(work_cap);
    uint32_t *want = xalloc(4 * piece_cap), *got = xalloc(4 * piece_cap);
    uint8_t *btext = xalloc(BATCH_BYTES + piece_cap);
    uint32_t *bwant = xalloc(4 * (BATCH_BYTES + piece_cap)), *bout = xalloc(4 * (BATCH_BYTES + piece_cap + 4));
    uint32_t bends[BATCH_PIECES];
    uint8_t *cache = xalloc(TOKS_CACHE_BUCKETS * 64);
    memset(cache, 0, TOKS_CACHE_BUCKETS * 64);
    uint64_t pieces = 0, bytes = 0, k6_bad = 0, k5_bad = 0, batches = 0, bn = 0, blen = 0, bnw = 0, maxlen = 0, by_words = 0;
    uint64_t hs = 0, hc = 0, ms = 0, merges = 0;
    double t_k6 = 0, t_k5 = 0;
    for (;;) {
        uint32_t len = 0, n = 0;
        int more = rd(&len, 4) && len != UINT32_MAX;
        if (more) {
            if (len == 0 || len > piece_cap) { fprintf(stderr, "bad record length %u\n", len); return 2; }
            if (!rd(piece, len) || !rd(&n, 4) || n > len || !rd(want, 4 * (size_t)n)) { fprintf(stderr, "truncated stream\n"); return 2; }
            toks_k6_args a = { piece, len, got, work, TOKS_BPE_WORK_BYTES(len), 0, 0, 0 };
            double c0 = now_ms();
            uint64_t ng = K6_BPE(&t, &a);
            t_k6 += now_ms() - c0;
            merges += a.merges;
            int bad = ng != n || memcmp(got, want, 4 * (size_t)n) != 0;
            if (bad && cfg.ignore_merges != 0u && (t.flags & TOKS_TF_PROBE_LONG) != 0u && len >= 2 && len <= TOKS_KEY_MAXLEN) {
                /* only a model token whose own merges do not rebuild it (hf: [that token], K6 alone: the merges),
                 * and only when K5's words, which answer it before K6, hold hf's answer (kernels.md §6 CONTRACT) */
                bpe_key k = bpe_key_at(piece, len, 0, len);
                const uint8_t *v = bpe_w3_probe(t.words, t.words_mask, bpe_key_hash(k), k);
                uint32_t wv[2], vid = 0;
                bad = n != 1 || !bpe_vhash_find(&t, piece, len, &vid) || vid != want[0] || v == NULL || bpe_w3_val(v, wv) != 1 ||
                      wv[0] != vid;
                by_words += bad ? 0 : 1;
            }
            if (bad) {
                if (k6_bad++ < 10) {
                    fprintf(stderr, "K6 MISMATCH (piece %" PRIu64 ")\n", pieces);
                    hexdump("piece", piece, len);
                    ids("hf  ", want, n);
                    ids("toks", got, ng);
                }
            }
            pieces++;
            bytes += len;
            maxlen = len > maxlen ? len : maxlen;
        }
        /* K5 over the batch when it is full, when this piece would overflow it, or at the end */
        if (bn > 0 && (!more || bn == BATCH_PIECES || blen + len > BATCH_BYTES)) {
            uint64_t wb = TOKS_BPE_WORK_BYTES(maxlen);
            toks_k5_args a = { btext, blen, bends, bn, 0, bout, blen + 4, 0, cache, TOKS_CACHE_MASK, work, wb, 0, 0, 0, 0, NULL };
            double c0 = now_ms();
            uint64_t no = K5_ENCODE(&t, &a);
            t_k5 += now_ms() - c0;
            hs += a.hits_static; hc += a.hits_cache; ms += a.misses;
            if (no != bnw || memcmp(bout, bwant, 4 * bnw) != 0 || a.hits_static + a.hits_cache + a.misses != bn) {
                if (k5_bad++ < 10) { fprintf(stderr, "K5 MISMATCH (batch %" PRIu64 ", %" PRIu64 " pieces)\n", batches, bn); }
            }
            batches++;
            bn = blen = bnw = 0;
        }
        if (!more) { break; }
        memcpy(btext + blen, piece, len);
        memcpy(bwant + bnw, want, 4 * (size_t)n);
        blen += len;
        bnw += n;
        bends[bn++] = (uint32_t)blen;
    }
    printf("K6: %" PRIu64 " pieces (%" PRIu64 " bytes, longest %" PRIu64 ", %" PRIu64 " merges; %" PRIu64 " by K5's words, "
           "the whole-piece rule), %" PRIu64 " mismatches; %.0f ms\n", pieces, bytes, maxlen, merges, by_words, k6_bad, t_k6);
    printf("K5: %" PRIu64 " batches, %" PRIu64 " mismatches; static %" PRIu64 " / cache %" PRIu64 " / miss %" PRIu64 "; %.0f ms\n",
           batches, k5_bad, hs, hc, ms, t_k5);
    int ok = k6_bad == 0 && k5_bad == 0 && bad_words == 0 && pieces > 0;
    printf("RESULT %s: %s\n", argv[1], ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
