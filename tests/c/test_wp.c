/*
 * tests/c/test_wp.c: unit tests of the WordPiece / BERT c twins (src/core/norm.c BERT steps, wp_scan.c, wp.c; the reader is config.c).
 * Runs in `make test` in well under a second; the differential against hf on real tokenizers is
 * tests/wordpiece/run.sh (lab hosts).
 *
 *   golden     toks_norm (the BertNormalizer's steps) on hf 0.23.2's own outputs (tests/data/wordpiece/norm_golden.inc, 5 flag sets)
 *   fold       wp_fold (SWAR) == the per-byte rule on every byte value in every lane, and on random words
 *   utf8       wp_utf8 == unicode table 3-7 on every 1-, 2- and 3-byte sequence and random 4-byte ones
 *   model      a small vocabulary: greedy longest match, [unk] rules, copy-on-write pieces, holes, punctuation
 *              made by strip_accents, invalid bytes, chinese chars, max_input_chars_per_word
 *   chunks     every cap and the minimal mat buffer give the one-call pieces
 *   geometry   scans and lookups on buffers flush against a guard page at either end (no access outside)
 *   load       toks_config_parse (tests/common/wp_json.h) + toks_wp_build on small tokenizer.json texts, and their
 *              refusals; toks_wp_build's last-wins rule for a repeated string on a hand-built vocab
 *   bounds     wordpiece.md §12.5 on every scalar and flag set: a char normalizes to at most 3 bytes per byte and to
 *              at most as many chars as it has bytes (the driver's buffers and id bounds rest on both)
 *   decode     toks_decode through toks.h on six small WordPiece tokenizers whose strings stress the prefix rule
 *              and hf's cleanup passes, against hf 0.23.2's own strings (tests/data/wordpiece/decode_golden.inc)
 */
#include "../../src/core/wp.h"
#include "../../src/core/json.h"
#include "../common/wp_json.h"
#include "toks.h"
#include "guard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                            fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* ---- golden normalizer vectors ------------------------------------------------------------------------------ */
#define S(x) (const uint8_t *)(x), (sizeof(x) - 1u)
typedef struct { uint32_t flags; const uint8_t *in; size_t in_len; const uint8_t *out; size_t out_len; } golden;
static const golden GOLDEN[] = {
#include "../data/wordpiece/norm_golden.inc"
};
#include "../data/wordpiece/decode_golden.inc"

static void test_golden(void)
{
    uint8_t buf[512];
    for (size_t i = 0; i < sizeof GOLDEN / sizeof GOLDEN[0]; i++) {
        const golden *g = &GOLDEN[i];
        int64_t n = toks_norm(g->flags, g->in, g->in_len, buf, sizeof buf);
        CHECK(n == (int64_t)g->out_len && memcmp(buf, g->out, g->out_len) == 0, "golden %zu (flags 0x%x): got %lld bytes, "
              "want %zu", i, g->flags, (long long)n, g->out_len);
    }
    /* the capacity contract: the bound (3 bytes a byte) or TOKS_E_CAP, nothing written */
    CHECK(toks_norm(0xFu, S("Hello World"), buf, 32) == TOKS_E_CAP && toks_norm(0xFu, S("Hello World"), buf, 33) == 11,
          "normalize cap");
}

/* ---- fold ----------------------------------------------------------------------------------------------- */
static void test_fold(void)
{
    for (uint32_t b = 0; b < 256u; b++) {
        for (uint32_t lane = 0; lane < 8u; lane++) {
            uint64_t w = (uint64_t)b << (8u * lane);
            uint64_t want = (uint64_t)((b >= 0x41u && b <= 0x5Au) ? b + 32u : b) << (8u * lane);
            CHECK(wp_fold(w) == want, "fold byte 0x%02x lane %u", b, lane);
        }
    }
    uint64_t s = 1;
    for (int k = 0; k < 200000; k++) {
        uint64_t w = guard_rng(&s) << 31 ^ guard_rng(&s);
        uint64_t want = 0;
        for (uint32_t lane = 0; lane < 8u; lane++) {
            uint64_t b = (w >> (8u * lane)) & 0xFFu;
            if (b >= 0x41u && b <= 0x5Au) { b += 32u; }
            want |= b << (8u * lane);
        }
        if (wp_fold(w) != want) { CHECK(0, "fold word %016llx", (unsigned long long)w); break; }
    }
}

/* ---- utf-8 ---------------------------------------------------------------------------------------------- */
static uint32_t ref_utf8(const uint8_t *p, uint64_t n, uint32_t *cp)
{
    /* the well-formed table 3-7 rows, written out */
    static const uint8_t rows[9][5] = {
        {0x00, 0x7F, 0, 0, 1}, {0xC2, 0xDF, 0x80, 0xBF, 2}, {0xE0, 0xE0, 0xA0, 0xBF, 3}, {0xE1, 0xEC, 0x80, 0xBF, 3},
        {0xED, 0xED, 0x80, 0x9F, 3}, {0xEE, 0xEF, 0x80, 0xBF, 3}, {0xF0, 0xF0, 0x90, 0xBF, 4},
        {0xF1, 0xF3, 0x80, 0xBF, 4}, {0xF4, 0xF4, 0x80, 0x8F, 4}};
    for (int r = 0; r < 9; r++) {
        if (p[0] < rows[r][0] || p[0] > rows[r][1]) { continue; }
        uint32_t len = rows[r][4];
        if (len == 1u) { *cp = p[0]; return 1; }
        if (n < len || p[1] < rows[r][2] || p[1] > rows[r][3]) { break; }
        for (uint32_t k = 2; k < len; k++) { if (p[k] < 0x80u || p[k] > 0xBFu) { *cp = TOKS_WP_INVALID_CP; return 1; } }
        uint32_t c = p[0] & (len == 2u ? 0x1Fu : len == 3u ? 0x0Fu : 0x07u);
        for (uint32_t k = 1; k < len; k++) { c = c << 6 | (p[k] & 0x3Fu); }
        *cp = c;
        return len;
    }
    *cp = TOKS_WP_INVALID_CP;
    return 1;
}

static void test_utf8(void)
{
    uint8_t p[4] = { 0, 0, 0, 0 };
    uint64_t bad = 0;
    for (uint32_t a = 0; a < 0x1000000u; a++) {         /* every 1-, 2- and 3-byte sequence, cut at every length */
        p[0] = (uint8_t)(a >> 16); p[1] = (uint8_t)(a >> 8); p[2] = (uint8_t)a; p[3] = 0x80;
        uint64_t n_lo = (a & 0xFFFFu) == 0u ? 1u : (a & 0xFFu) == 0u ? 2u : 3u;
        for (uint64_t n = n_lo; n <= 4; n++) {
            uint32_t c1, c2;
            uint32_t l1 = wp_utf8(p, n, &c1), l2 = ref_utf8(p, n, &c2);
            if (l1 != l2 || c1 != c2) { bad++; }
        }
    }
    uint64_t s = 7;
    for (int k = 0; k < 2000000; k++) {
        uint64_t r = guard_rng(&s);
        p[0] = (uint8_t)(0xF0u + (r & 7u)); p[1] = (uint8_t)(r >> 3); p[2] = (uint8_t)(r >> 11);
        p[3] = (uint8_t)(r >> 19);
        uint32_t c1, c2;
        if (wp_utf8(p, 4, &c1) != ref_utf8(p, 4, &c2) || c1 != c2) { bad++; }
    }
    CHECK(bad == 0, "utf8: %llu disagreements with table 3-7", (unsigned long long)bad);
    uint8_t q[4];
    for (uint32_t cp = 0; cp < 0x110000u; cp++) {      /* put/decode round trip on every scalar */
        if (cp >= 0xD800u && cp < 0xE000u) { continue; }
        uint32_t l = toks_utf8_put(q, cp), back;
        if (wp_utf8(q, l, &back) != l || back != cp) { CHECK(0, "utf8 round trip U+%04X", cp); break; }
    }
}

/* ---- a small vocabulary ----------------------------------------------------------------------------------- */
static const char *VOCAB[] = { "[UNK]", "un", "##aff", "##able", "a", "##b", "hello", "world", ",", "!", "##s",
                               "e", "cafe", "`", "x", "=", "b", "##", "hel", "##lo", "caf\xc3\xa9", "\xe4\xb8\xad" };

typedef struct fixture {
    toks_wp_tables t;
    toks_arena ar;
} fixture;

static void build(fixture *fx, uint32_t flags, uint32_t max_chars, int with_unk)
{
    static const uint8_t *str[64];
    static uint32_t len[64], id[64];
    uint32_t n = 0;
    for (uint32_t i = 0; i < sizeof VOCAB / sizeof VOCAB[0]; i++) {
        if (!with_unk && i == 0) { continue; }
        str[n] = (const uint8_t *)VOCAB[i];
        len[n] = (uint32_t)strlen(VOCAB[i]);
        id[n] = i;
        n++;
    }
    toks_wp_vocab v = { str, len, id, n };
    toks_wp_params p = { flags, max_chars, (const uint8_t *)"[UNK]", 5, (const uint8_t *)"##", 2 };
    uint64_t nb = toks_wp_tables_bytes(&v);
    fx->ar.base = malloc(nb);
    fx->ar.len = nb;
    fx->ar.pos = 0;
    toks_err err;
    int64_t rc = toks_wp_build(&fx->t, &fx->ar, &v, &p, &err);
    CHECK(with_unk ? rc == 0 : rc == TOKS_E_UNSUPPORTED, "build rc %lld", (long long)rc);
}

static void drop(fixture *fx) { free(fx->ar.base); }   /* the arena build() took (LeakSanitizer) */

/* scan + encode of s; returns the id count, ids in out */
static uint64_t encode(const fixture *fx, const uint8_t *s, uint64_t n, uint32_t *out, uint64_t room,
                       uint32_t *n_mat)
{
    toks_wp_piece pieces[256];
    uint8_t mat[4096];
    toks_wp_scan_args a;
    memset(&a, 0, sizeof a);
    a.text = s; a.len = n; a.pieces = pieces; a.cap = 256; a.mat = mat; a.mat_cap = sizeof mat; a.flags = ~0ull;
    uint64_t np = toks_wp_scan_c(&fx->t, &a);
    CHECK(a.pos == n, "scan did not finish");
    uint32_t m = 0;
    for (uint64_t k = 0; k < np; k++) { if (pieces[k].flags & TOKS_WPP_MAT) { m++; } }
    if (n_mat) { *n_mat = m; }
    toks_wp_encode_args e;
    memset(&e, 0, sizeof e);
    e.text = s; e.mat = mat; e.pieces = pieces; e.n = np; e.out = out; e.room = room;
    return toks_wp_encode_c(&fx->t, &e);
}

static void expect(const fixture *fx, const char *s, const uint32_t *want, uint64_t nw, int want_mat)
{
    uint32_t out[256];
    uint32_t n_mat = 0;
    uint64_t n = encode(fx, (const uint8_t *)s, strlen(s), out, 256, &n_mat);
    int ok = n == nw;
    for (uint64_t k = 0; ok && k < n; k++) { ok = out[k] == want[k]; }
    CHECK(ok, "ids of \"%s\": got %llu ids (first %u)", s, (unsigned long long)n, n ? out[0] : 0u);
    if (want_mat >= 0) { CHECK((int)n_mat == want_mat, "\"%s\": %u pieces copied, want %d", s, n_mat, want_mat); }
}

#define E(fx, s, m, ...) do { static const uint32_t w_[] = { __VA_ARGS__ }; \
                              expect(fx, s, w_, sizeof w_ / sizeof w_[0], m); } while (0)

static void test_model(void)
{
    fixture unc, cas, tiny, nounk;
    build(&unc, TOKS_WPF_CLEAN | TOKS_WPF_CHINESE | TOKS_WPF_STRIP | TOKS_WPF_LOWER, 100, 1);
    build(&cas, TOKS_WPF_CLEAN | TOKS_WPF_CHINESE, 100, 1);
    build(&tiny, TOKS_WPF_CLEAN | TOKS_WPF_CHINESE | TOKS_WPF_STRIP | TOKS_WPF_LOWER, 3, 1);
    build(&nounk, 0, 100, 0);

    E(&unc, "unaffable", 0, 1, 2, 3);                   /* un ##aff ##able: greedy longest prefix */
    E(&unc, "Hello, World!", 0, 6, 8, 7, 9);            /* ascii folded in the register: nothing copied */
    E(&unc, "HELLO", 0, 6);
    E(&cas, "HELLO", 0, 0);                             /* cased: no fold, not in the vocab, no prefix: [unk] */
    E(&unc, "ab", 0, 4, 5);
    E(&unc, "abc", 0, 0);                               /* one unmatched position: the whole piece is [unk] */
    E(&unc, "hello", 0, 6);                             /* whole-piece hit beats hel ##lo */
    E(&unc, "Caf\xc3\xa9", 1, 12);                      /* strip_accents: the piece is copied, "cafe" */
    E(&cas, "caf\xc3\xa9", 0, 20);                      /* cased keeps the accent, in place */
    E(&unc, "x\xe2\x89\xa0" "b", 1, 14, 15, 16);        /* U+2260 -> "=" (punctuation made by N3) */
    E(&cas, "x\xe2\x89\xa0" "b", 0, 0);                 /* cased: one piece "x≠b", not in the vocab */
    E(&unc, "a\xff" "b", 0, 4, 0, 16);                  /* an invalid byte is a piece of its own: [unk] */
    E(&unc, "a\xe4\xb8\xad" "b", 0, 4, 21, 16);         /* chinese char isolated, in place */
    E(&unc, "hel\x01lo", 1, 6);                         /* a control char removed inside the word: copied */
    E(&unc, "hel\xe2\x80\x8blo", 1, 6);                 /* U+200B likewise */
    E(&unc, "hello\xe2\x80\x8b world", 0, 6, 7);        /* a hole at the end of the word: still in place */
    E(&unc, "\xe2\x80\x8bhello", 0, 6);                 /* a hole before the word: the piece starts after it */
    E(&unc, "e\xcc\x81", 0, 11);                        /* e + U+0301: the mark is a hole, "e" in place */
    E(&unc, "a\x0b" "b", 1, 4, 5);                      /* \v is removed (Cc beats White_Space): "ab" */
    E(&cas, "\xc2\x85" "a", 0, 4);                      /* U+0085 removed */
    E(&unc, "a\xc2\xa0" "b", 0, 4, 16);                 /* U+00A0 splits */
    E(&unc, "##", 0, 0, 0);                             /* "#" is punctuation: two pieces "#", [unk] each */
    E(&tiny, "hello", 0, 0);                            /* 5 chars > 3 */
    E(&tiny, "hel", 0, 18);
    E(&tiny, "caf\xc3\xa9x", 0, 0);                     /* over: nothing copied (the content is irrelevant) */
    E(&tiny, "!", 0, 9);
    expect(&unc, "", NULL, 0, 0);
    drop(&unc);
    drop(&cas);
    drop(&tiny);
    drop(&nounk);
}

/* ---- chunks: every cap, the minimal mat buffer -------------------------------------------------------------- */
static void test_chunks(void)
{
    fixture fx;
    build(&fx, TOKS_WPF_CLEAN | TOKS_WPF_CHINESE | TOKS_WPF_STRIP | TOKS_WPF_LOWER, 100, 1);
    static const char *texts[] = {
        "Hello, World! Caf\xc3\xa9 au lait x\xe2\x89\xa0y \xe4\xb8\xad\xe6\x96\x87 hel\x01lo \xff end",
        ("\xc3\xa9\xc3\xa9\xc3\xa9 \xc3\x89 a\xf0\x9e\x80\x80\xf0\x9e\xa5\x8a" "b ,,,!!! \xea\xb0\x80\xea\xb0\x81"),
        "a", "", "   ", "\xe2\x80\x8b\xe2\x80\x8b"};
    for (size_t ti = 0; ti < sizeof texts / sizeof texts[0]; ti++) {
        const uint8_t *s = (const uint8_t *)texts[ti];
        uint64_t n = strlen(texts[ti]);
        uint32_t ref[512];
        uint64_t nref = encode(&fx, s, n, ref, 512, NULL);
        for (uint64_t cap = 2; cap <= 9; cap++) {
            uint64_t mat_min = toks_wp_mat_min(&fx.t);
            uint8_t *mat = malloc(mat_min);
            toks_wp_piece pieces[16];
            uint32_t got[512];
            uint64_t ng = 0, pos = 0;
            for (int round = 0; round < 1000; round++) {
                toks_wp_scan_args a;
                memset(&a, 0, sizeof a);
                a.text = s; a.len = n; a.pos = pos; a.pieces = pieces; a.cap = cap; a.mat = mat; a.mat_cap = mat_min;
                a.flags = ~0ull;
                uint64_t np = toks_wp_scan_c(&fx.t, &a);
                toks_wp_encode_args e;
                memset(&e, 0, sizeof e);
                e.text = s; e.mat = mat; e.pieces = pieces; e.n = np; e.out = got + ng; e.room = 512 - ng;
                ng += toks_wp_encode_c(&fx.t, &e);
                pos = a.pos;
                if (pos >= n) { break; }
            }
            int ok = ng == nref && memcmp(got, ref, ng * 4u) == 0;
            CHECK(ok, "chunks: text %zu cap %llu: %llu ids, want %llu", ti, (unsigned long long)cap,
                  (unsigned long long)ng, (unsigned long long)nref);
            free(mat);
        }
    }
    drop(&fx);
}

/* ---- geometry: guard pages at either end -------------------------------------------------------------------- */
static void test_geometry(void)
{
    fixture fx;
    build(&fx, TOKS_WPF_CLEAN | TOKS_WPF_CHINESE | TOKS_WPF_STRIP | TOKS_WPF_LOWER, 100, 1);
    static const uint8_t alpha[] = "aAbB,! \t\x01\xc3\xa9\xe4\xb8\xad\xf0\x9e\x80\x80\xe2\x80\x8b\xff\xe2\x89\xa0#";
    uint64_t s = 3;
    for (size_t len = 0; len <= 96; len++) {
        for (int where = 0; where < 2; where++) {
            for (int rep = 0; rep < 8; rep++) {
                guard_buf g;
                uint8_t *p = guard_alloc(&g, len ? len : 1, where == 0 ? GUARD_END : GUARD_START, 0);
                if (p == NULL) { CHECK(0, "guard_alloc"); return; }
                if (where == 0 && len == 0) { p += 1; }
                for (size_t i = 0; i < len; i++) { p[i] = alpha[guard_rng(&s) % (sizeof alpha - 1u)]; }
                uint32_t out[512];
                (void)encode(&fx, p, len, out, 512, NULL);
                uint8_t nb[512];
                (void)toks_norm(fx.t.flags, p, len, nb, sizeof nb);
                uint64_t probes = 0;
                if (len) { (void)toks_wp_piece_ids(&fx.t, p, len, 0, out, &probes); }
                guard_free(&g);
            }
        }
    }
    drop(&fx);
}

/* ---- load ------------------------------------------------------------------------------------------------- */
static int64_t load_json(const char *js, toks_wp_tables *t, toks_err *err)
{
    static uint8_t arena[1 << 20], arena2[1 << 20];
    toks_arena ar = { arena, sizeof arena, 0 }, ar2 = { arena2, sizeof arena2, 0 };
    toks_wp_vocab v;
    toks_wp_params p;
    int64_t rc = wp_from_json((const uint8_t *)js, strlen(js), &ar, &v, &p, err);
    if (rc != 0) { return rc; }
    return toks_wp_build(t, &ar2, &v, &p, err);
}

static void test_load(void)
{
    toks_wp_tables t;
    toks_err err;
    const char *ok = "{\"normalizer\":{\"type\":\"BertNormalizer\",\"clean_text\":true,\"handle_chinese_chars\":true,"
                     "\"strip_accents\":null,\"lowercase\":true},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
                     "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
                     "\"max_input_chars_per_word\":100,\"vocab\":{\"[UNK]\":0,\"a\":1,\"##b\":2,\"c\":3}}}";
    CHECK(load_json(ok, &t, &err) == 0, "load ok: %s", err.what ? err.what : "");
    CHECK(t.flags == 0xFu, "strip_accents null follows lowercase true: flags 0x%x", t.flags);
    uint32_t out[4];
    uint64_t probes = 0;
    {                                       /* toks_wp_build: a repeated string keeps its LAST id (json map) */
        static uint8_t arena3[1 << 16];
        toks_arena ar3 = { arena3, sizeof arena3, 0 };
        static const uint8_t *const ds[4] = { (const uint8_t *)"[UNK]", (const uint8_t *)"a", (const uint8_t *)"##b",
                                              (const uint8_t *)"a" };
        static const uint32_t dl[4] = { 5, 1, 3, 1 }, di[4] = { 0, 1, 2, 3 };
        toks_wp_vocab dv = { ds, dl, di, 4 };
        toks_wp_params dp = { 0xFu, 100, (const uint8_t *)"[UNK]", 5, (const uint8_t *)"##", 2 };
        toks_wp_tables td;
        CHECK(toks_wp_build(&td, &ar3, &dv, &dp, &err) == 0 &&
              toks_wp_piece_ids(&td, (const uint8_t *)"ab", 2, 0, out, &probes) == 2 && out[0] == 3 && out[1] == 2,
              "duplicate key keeps the last id");
    }
    const char *cased = "{\"normalizer\":{\"type\":\"BertNormalizer\",\"clean_text\":true,\"handle_chinese_chars\":true,"
                        "\"strip_accents\":null,\"lowercase\":false},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
                        "\"model\":{\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
                        "\"max_input_chars_per_word\":100,\"vocab\":{\"[UNK]\":0}}}";
    CHECK(load_json(cased, &t, &err) == 0 && t.flags == 0x3u, "cased: strip follows lowercase false");
    const char *nounk = "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"model\":{\"type\":"
                        "\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
                        "\"max_input_chars_per_word\":100,\"vocab\":{\"a\":0}}}";
    CHECK(load_json(nounk, &t, &err) == TOKS_E_UNSUPPORTED && strstr(err.what, "unk_token") != NULL,
          "unk_token not in the vocab is refused");
    const char *ws = "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Whitespace\"},\"model\":{\"type\":\"WordPiece\","
                     "\"unk_token\":\"a\",\"continuing_subword_prefix\":\"##\",\"max_input_chars_per_word\":100,"
                     "\"vocab\":{\"a\":0}}}";
    CHECK(load_json(ws, &t, &err) == TOKS_E_UNSUPPORTED && strstr(err.what, "pre_tokenizer") != NULL,
          "Whitespace pre-tokenizer refused (c twin)");
    const char *seq = "{\"normalizer\":{\"type\":\"Lowercase\"},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},"
                      "\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"a\",\"continuing_subword_prefix\":\"##\","
                      "\"max_input_chars_per_word\":100,\"vocab\":{\"a\":0}}}";
    CHECK(load_json(seq, &t, &err) == TOKS_E_UNSUPPORTED && strstr(err.what, "normalizer") != NULL,
          "Lowercase normalizer refused (c twin)");
    const char *bpe = "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"model\":{\"type\":\"BPE\","
                      "\"vocab\":{}}}";
    CHECK(load_json(bpe, &t, &err) == TOKS_E_UNSUPPORTED, "BPE model refused");
    const char *big = "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"model\":{\"type\":"
                      "\"WordPiece\",\"unk_token\":\"a\",\"continuing_subword_prefix\":\"##\","
                      "\"max_input_chars_per_word\":2000000,\"vocab\":{\"a\":0}}}";
    CHECK(load_json(big, &t, &err) == TOKS_E_UNSUPPORTED, "max_input_chars_per_word above 2^20 refused");
}

/* ---- bounds (wordpiece.md §12.5) ------------------------------------------------------------------------- */
static void test_bounds(void)
{
    static const uint32_t FL[4] = { TOKS_WPF_CHINESE, TOKS_WPF_CHINESE | TOKS_WPF_STRIP, TOKS_WPF_CHINESE | TOKS_WPF_LOWER,
                                    TOKS_WPF_CHINESE | TOKS_WPF_STRIP | TOKS_WPF_LOWER };   /* CLEAN only shrinks */
    uint64_t bad_bytes = 0, bad_chars = 0, worst = 0;
    for (uint32_t cp = 0; cp < 0x110000u; cp++) {       /* every scalar */
        if (cp >= 0xD800u && cp < 0xE000u) { continue; }
        uint8_t in[4], out[64];
        uint32_t l = toks_utf8_put(in, cp);
        for (uint32_t k = 0; k < 4u; k++) {
            uint64_t n = (uint64_t)toks_norm(FL[k], in, l, out, sizeof out);
            if (n > 3u * l) { bad_bytes++; }
            if (n * 1000u / l > worst) { worst = n * 1000u / l; }
            uint32_t o[TOKS_NORM_MAX_OUT];
            uint32_t m = toks_norm_char(FL[k], cp, o), chars = 0;
            for (uint32_t i = 0; i < m; i++) { chars += o[i] != TOKS_NORM_GHOST; }
            if (chars > l) { bad_chars++; }
        }
    }
    CHECK(bad_bytes == 0 && bad_chars == 0, "bounds: %llu chars over 3 bytes per byte, %llu with more chars than bytes",
          (unsigned long long)bad_bytes, (unsigned long long)bad_chars);
    CHECK(worst == 3000u, "bounds: the worst ratio is %llu/1000 (Hangul under strip_accents is 3)", (unsigned long long)worst);
}

/* ---- decode (toks.h) ------------------------------------------------------------------------------------- */
static void test_decode(void)
{
    uint32_t nt = (uint32_t)(sizeof DEC_JSON / sizeof DEC_JSON[0]);
    toks_ctx *ctx[8] = { 0 };
    for (uint32_t v = 0; v < nt && v < 8u; v++) {
        toks_diag dg;
        toks_load_opts o = { sizeof o, 0, 0, 0, &dg };
        int64_t r = toks_load_mem_copy(&ctx[v], DEC_JSON[v], strlen(DEC_JSON[v]), &o);
        CHECK(r == 0, "decode tokenizer %u: load %lld (%s)", v, (long long)r, dg.what);
    }
    uint64_t bad = 0;
    for (size_t i = 0; i < sizeof DEC_CASES / sizeof DEC_CASES[0]; i++) {
        const dec_case *c = &DEC_CASES[i];
        if (ctx[c->v] == NULL) { continue; }
        uint8_t out[512];
        int64_t n = toks_decode(ctx[c->v], c->ids, c->n, c->flags, out, sizeof out);
        int ok = n == (int64_t)c->want_len && memcmp(out, c->want, c->want_len) == 0;
        uint8_t small[7];                               /* the capacity rule: the total, and the prefix that fits */
        int64_t n2 = toks_decode(ctx[c->v], c->ids, c->n, c->flags, small, sizeof small);
        ok = ok && n2 == n && memcmp(small, c->want, c->want_len < 7u ? c->want_len : 7u) == 0;
        if (!ok && ++bad <= 5) {
            fprintf(stderr, "decode case %zu (tokenizer %u, flags %u): got %lld bytes \"%.*s\", want \"%s\"\n", i, c->v,
                    c->flags, (long long)n, (int)(n > 0 && n < 512 ? n : 0), (const char *)out, c->want);
        }
    }
    CHECK(bad == 0, "decode: %llu of %zu cases differ from hf", (unsigned long long)bad, sizeof DEC_CASES / sizeof DEC_CASES[0]);
    for (uint32_t v = 0; v < nt && v < 8u; v++) { toks_unload(ctx[v]); }
}

int main(void)
{
    test_golden();
    test_fold();
    test_utf8();
    test_model();
    test_chunks();
    test_geometry();
    test_load();
    test_bounds();
    test_decode();
    if (fails) {
        fprintf(stderr, "test_wp: %d FAILED\n", fails);
        return 1;
    }
    printf("test_wp: ok (%zu normalizer and %zu decode vectors from hf 0.23.2; bounds on every scalar)\n",
           sizeof GOLDEN / sizeof GOLDEN[0], sizeof DEC_CASES / sizeof DEC_CASES[0]);
    return 0;
}
