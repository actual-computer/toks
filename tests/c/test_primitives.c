/*
 * test_primitives.c: the parity primitives of include/toks.h against the references that define them, hf tokenizers
 * 0.23.2 and tiktoken 0.14.0 (tests/c/test_primitives.py writes test_primitives.inc from them; python/toks_oracle/
 * primitives.py states each reference):
 *  1. TOKS_NO_TRUNCATE / TOKS_NO_PAD: on every file that truncates or pads, five texts under eight flag sets give hf's
 *     ids after no_truncation() / no_padding() (each, both, neither; with and without post-processing); on every other
 *     file the two flags change nothing; toks_pieces takes them and is unchanged; a file that truncates or pads has
 *     no certified cut under flags 0 and, once the call opts out of both, its parts concatenate to the whole.
 *  2. toks_template: ids, type ids and prefix count as hf's post_process() of the empty encoding, the text's type id
 *     (toks_info's seq_type_id), encode("") with both flags equal to hf's; the sizing call, TOKS_E_CAP writing
 *     nothing, NULL type_ids / n_prefix, the argument errors.
 *  3. toks_added: count and digest of (id, flags, content) equal hf's get_added_tokens_decoder() (Kimi K3: its
 *     tiktoken specials, special where transformers lists them); every id toks_id_flags calls ADDED is listed once,
 *     with the same bits but SPECIAL (whose rule differs only for a content listed both ways); its content finds the
 *     id; the errors.
 *  4. TOKS_DECODE_RAW: tiktoken decode_bytes (kimik3), hf's ByteLevel bytes before its lossy step, hf's ByteFallback
 *     decode with each U+FFFD of a run that is not utf-8 back to its byte; on every file, its texts' ids and random
 *     ids: a raw result that is valid utf-8 is decode's, a byte-level one is the tokens' bytes; every cap; the flags.
 *  5. toks_info: the fields equal hf's truncation / padding dicts and the template; abi 0.3's size (184) gets the
 *     fields before trunc_on and nothing past them; other sizes are TOKS_E_ARG.
 * Files: the tests/data fixtures and the cache's (~/.cache/toks/tokenizers or $TOKS_TOKENIZER_CACHE; Kimi K3 in
 * ~/.cache/toks/kimik3 or $TOKS_KIMI_DIR). An absent file, or one whose sha-256 is not the pinned one, prints SKIP.
 */
#include "toks.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 60) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

typedef struct pr_enc { uint32_t text, flags, n; uint64_t digest; } pr_enc;
typedef struct pr_raw { const uint32_t *ids; uint32_t n, flags, len; uint64_t digest; } pr_raw;
typedef struct pr_file {
    const char *name;
    int src;                       /* 1: a tests/data path, 0: the cache's file of that name, 2: the Kimi K3 directory */
    const char *sha;               /* the file's sha-256 (hex; "" for kimik3: oracle_tiktoken.py pins its files) */
    uint32_t n_tmpl, n_prefix, seq_type;
    const uint32_t *tmpl_ids, *tmpl_types;
    uint32_t info[10];             /* trunc_on, trunc_max, trunc_stride, pad_on, pad_fixed, pad_id, pad_type_id, pad_len,
                                      pad_multiple, pad_left */
    uint32_t n_added;
    uint64_t added_digest;         /* FNV-1a 64 of (id, flags without TOKS_ID_BYTE, length, content) in id order */
    const char *text4;             /* a text with the file's added tokens */
    uint32_t empty_n;
    uint64_t empty_digest;         /* hf's encode("") without truncation or padding */
    const pr_enc *enc;
    uint32_t n_enc;
    const pr_raw *raw;
    uint32_t n_raw;
} pr_file;

#include "test_primitives.inc"

/* ---- helpers -------------------------------------------------------------------------------------------------- */

static uint64_t fnv(uint64_t h, const void *p, uint64_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    for (uint64_t i = 0; i < n; i++) { h = (h ^ b[i]) * 0x100000001B3ull; }
    return h;
}
#define FNV0 0xCBF29CE484222325ull

static uint64_t fnv_u32(uint64_t h, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    return fnv(h, b, 4);
}

static uint64_t ids_digest(const uint32_t *ids, uint64_t n)
{
    uint64_t h = FNV0;
    for (uint64_t i = 0; i < n; i++) { h = fnv_u32(h, ids[i]); }
    return h;
}

/* strict utf-8 (unicode §3.9 table 3-7), written out here independent of the core's */
static int utf8_ok(const uint8_t *p, uint64_t n)
{
    uint64_t i = 0;
    while (i < n) {
        uint8_t c = p[i];
        uint32_t k = c < 0x80u ? 1u : c < 0xC2u ? 0u : c < 0xE0u ? 2u : c < 0xF0u ? 3u : c < 0xF5u ? 4u : 0u;
        if (k == 0u || i + k > n) { return 0; }
        uint8_t lo = (c == 0xE0u) ? 0xA0u : (c == 0xF0u) ? 0x90u : 0x80u;
        uint8_t hi = (c == 0xEDu) ? 0x9Fu : (c == 0xF4u) ? 0x8Fu : 0xBFu;
        for (uint32_t j = 1; j < k; j++) {
            uint8_t d = p[i + j];
            if (d < (j == 1u ? lo : 0x80u) || d > (j == 1u ? hi : 0xBFu)) { return 0; }
        }
        i += k;
    }
    return 1;
}

static uint64_t rng = 0x5EED0F7015C0FFEEull;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static char *path_of(const pr_file *f)
{
    static char buf[1024];
    const char *env = getenv(f->src == 2 ? "TOKS_KIMI_DIR" : "TOKS_TOKENIZER_CACHE");
    const char *home = getenv("HOME") ? getenv("HOME") : ".";
    if (f->src == 1) { snprintf(buf, sizeof buf, "%s", f->name); }
    else if (f->src == 2) {
        if (env != NULL && env[0] != 0) { snprintf(buf, sizeof buf, "%s", env); }
        else { snprintf(buf, sizeof buf, "%s/.cache/toks/kimik3", home); }
    } else if (env != NULL && env[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", env, f->name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", home, f->name); }
    return buf;
}

/* the texts: T0 "", T1, T2, T3 = T2 x PR_T3_REPEAT, T4 the file's */
static char *T3;
static uint64_t T3_LEN;
static const char *text_of(const pr_file *f, uint32_t t, uint64_t *len)
{
    const char *s = t == 0u ? "" : t == 1u ? PR_T1 : t == 2u ? PR_T2 : t == 3u ? T3 : f->text4;
    *len = t == 3u ? T3_LEN : (uint64_t)strlen(s);
    return s;
}

typedef struct run { toks_ctx *c; void *scr; uint32_t *out; uint64_t cap; uint8_t *bytes; uint64_t bcap; } run;

static int64_t enc(run *r, const char *s, uint64_t len, uint32_t flags)
{
    return toks_encode(r->c, s, len, flags, r->out, r->cap, r->scr);
}

/* totals */
static uint64_t g_files, g_absent, g_other, g_refused, g_enc, g_inv, g_tmpl, g_added, g_special_rule, g_raw, g_raw_inv,
    g_raw_bad, g_cuts;

/* ---- 1. TOKS_NO_TRUNCATE / TOKS_NO_PAD --------------------------------------------------------------------------- */

static void check_flags(const pr_file *f, run *r)
{
    for (uint32_t k = 0; k < f->n_enc; k++) {               /* hf's ids after no_truncation() / no_padding() */
        const pr_enc *e = &f->enc[k];
        uint64_t len;
        const char *s = text_of(f, e->text, &len);
        int64_t n = enc(r, s, len, e->flags);
        CHECK(n == (int64_t)e->n && (n < 0 || (uint64_t)n > r->cap || ids_digest(r->out, (uint64_t)n) == e->digest),
              "%s: text %u flags %u: %" PRId64 " ids, hf %u", f->name, e->text, e->flags, n, e->n);
        g_enc++;
    }
    uint64_t *offs = (uint64_t *)malloc(8u * 16u);
    uint32_t *w = (uint32_t *)malloc(4u * (size_t)r->cap);
    for (uint32_t t = 0; t < PR_N_TEXTS + 1u; t++) {         /* nothing else moves: pieces, and encode where hf has no case */
        uint64_t len;
        const char *s = text_of(f, t, &len);
        for (uint32_t pp = 0; pp < 2u; pp++) {
            uint32_t base = pp ? TOKS_NO_POSTPROCESS : 0u;
            int64_t a = toks_pieces(r->c, s, len, base, r->out, r->cap, r->scr);
            int64_t b = toks_pieces(r->c, s, len, base | TOKS_NO_TRUNCATE | TOKS_NO_PAD, w, r->cap, r->scr);
            CHECK(a >= 0 && a == b && (uint64_t)a <= r->cap && memcmp(r->out, w, 4u * (size_t)(a > 0 ? a : 0)) == 0,
                  "%s: pieces of text %u with the two flags: %" PRId64 " vs %" PRId64, f->name, t, b, a);
            if (f->info[0] != 0u || f->info[3] != 0u) { continue; }
            int64_t n0 = enc(r, s, len, base);
            memcpy(w, r->out, 4u * (size_t)(n0 > 0 ? n0 : 0));
            for (uint32_t x = 1; x < 4u; x++) {
                int64_t n = enc(r, s, len, base | (x << 4));
                CHECK(n == n0 && memcmp(w, r->out, 4u * (size_t)(n0 > 0 ? n0 : 0)) == 0,
                      "%s: text %u flags %u: %" PRId64 " ids, flags %u %" PRId64, f->name, t, base | (x << 4), n, base, n0);
                g_inv++;
            }
        }
    }
    /* cuts: none while the file's truncation or padding applies to the call; with both turned off, the family's, and
     * the parts concatenate to the whole (toks.h toks_split_points) */
    int applies = f->info[0] != 0u || (f->info[3] != 0u && (f->info[4] != 0u || f->info[8] != 0u));
    if (applies) {
        CHECK(toks_split_points(r->c, T3, T3_LEN, 0u, 8u, offs, 7u, NULL) == 0, "%s: a cut under the file's truncation /"
              " padding", f->name);
        int64_t c = toks_split_points(r->c, T3, T3_LEN, TOKS_NO_TRUNCATE | TOKS_NO_PAD, 8u, offs, 7u, NULL);
        CHECK(c >= 0, "%s: split_points with both flags: %" PRId64, f->name, c);
        if (c > 0) {
            uint32_t fl = TOKS_NO_TRUNCATE | TOKS_NO_PAD | TOKS_NO_POSTPROCESS;
            int64_t whole = enc(r, T3, T3_LEN, fl);
            uint64_t at = 0, m = 0;
            int ok = whole >= 0 && (uint64_t)whole <= r->cap;
            if (ok) { memcpy(w, r->out, 4u * (size_t)whole); }
            for (int64_t j = 0; ok && j <= c; j++) {
                uint64_t end = j < c ? offs[j] : T3_LEN;
                int64_t n = enc(r, T3 + at, end - at, fl | (j > 0 ? TOKS_CONTINUATION : 0u));
                ok = n >= 0 && m + (uint64_t)n <= (uint64_t)whole && memcmp(w + m, r->out, 4u * (size_t)n) == 0;
                m += n > 0 ? (uint64_t)n : 0u;
                at = end;
            }
            CHECK(ok && m == (uint64_t)whole, "%s: %" PRId64 " cuts with both flags: the parts are not the whole", f->name, c);
            g_cuts += (uint64_t)c;
        }
    }
    free(offs);
    free(w);
}

/* ---- 2. toks_template ---------------------------------------------------------------------------------------- */

static void check_template(const pr_file *f, run *r)
{
    uint32_t np = 77u, ids[72], ty[72];
    CHECK(toks_template(r->c, NULL, NULL, 0u, &np) == (int64_t)f->n_tmpl && np == f->n_prefix,
          "%s: template size %" PRId64 " prefix %u, hf %u / %u", f->name, toks_template(r->c, NULL, NULL, 0u, NULL), np,
          f->n_tmpl, f->n_prefix);
    memset(ids, 0xAB, sizeof ids);
    memset(ty, 0xCD, sizeof ty);
    np = 77u;
    int64_t n = toks_template(r->c, ids, ty, f->n_tmpl, &np);
    CHECK(n == (int64_t)f->n_tmpl && np == f->n_prefix && memcmp(ids, f->tmpl_ids, 4u * f->n_tmpl) == 0 &&
          memcmp(ty, f->tmpl_types, 4u * f->n_tmpl) == 0 && ids[f->n_tmpl] == 0xABABABABu && ty[f->n_tmpl] == 0xCDCDCDCDu,
          "%s: template ids / type ids", f->name);
    if (f->n_tmpl > 0u) {                                   /* a cap below the count: TOKS_E_CAP, nothing written */
        uint32_t ids2[72], ty2[72], np2 = 77u;
        memset(ids2, 0x5A, sizeof ids2);
        memset(ty2, 0x5A, sizeof ty2);
        CHECK(toks_template(r->c, ids2, ty2, f->n_tmpl - 1u, &np2) == TOKS_E_CAP && np2 == 77u && ids2[0] == 0x5A5A5A5Au &&
              ty2[0] == 0x5A5A5A5Au, "%s: template cap %u", f->name, f->n_tmpl - 1u);
    }
    memset(ids, 0, sizeof ids);
    CHECK(toks_template(r->c, ids, NULL, 64u, NULL) == (int64_t)f->n_tmpl && memcmp(ids, f->tmpl_ids, 4u * f->n_tmpl) == 0,
          "%s: template without type ids / prefix", f->name);
    CHECK(toks_template(NULL, ids, ty, 64u, &np) == TOKS_E_ARG && toks_template(r->c, NULL, ty, 1u, &np) == TOKS_E_ARG &&
          toks_template(r->c, NULL, NULL, 0u, NULL) == (int64_t)f->n_tmpl, "%s: template errors", f->name);
    /* encode("") with both flags: hf's encode("") without truncation or padding (the template, the card says) */
    int64_t e = enc(r, "", 0u, TOKS_NO_TRUNCATE | TOKS_NO_PAD);
    CHECK(e == (int64_t)f->empty_n && ids_digest(r->out, (uint64_t)(e > 0 ? e : 0)) == f->empty_digest,
          "%s: encode(\"\") %" PRId64 " ids, hf %u", f->name, e, f->empty_n);
    g_tmpl++;
}

/* ---- 3. toks_added --------------------------------------------------------------------------------------------- */

static void check_added(const pr_file *f, run *r, uint32_t n_ids)
{
    uint64_t h = FNV0;
    uint32_t i = 0, prev = 0, differ = 0;
    for (;; i++) {
        const void *content = NULL;
        uint64_t len = 0;
        uint32_t id = 0;
        int64_t fl = toks_added(r->c, i, &content, &len, &id);
        if (fl == TOKS_E_ARG || i > n_ids) { break; }
        CHECK(fl >= 0 && (fl & TOKS_ID_ADDED) != 0 && (fl & ~(int64_t)0x7F) == 0 && (i == 0u || id > prev) && id < n_ids,
              "%s: added %u: flags %" PRId64 " id %u", f->name, i, fl, id);
        int64_t idf = toks_id_flags(r->c, id);
        CHECK((idf & ~(int64_t)TOKS_ID_SPECIAL) == (fl & ~(int64_t)TOKS_ID_SPECIAL), "%s: id %u: toks_added %" PRId64
              ", toks_id_flags %" PRId64, f->name, id, fl, idf);
        differ += (idf & TOKS_ID_SPECIAL) != (fl & TOKS_ID_SPECIAL);
        CHECK(toks_token_to_id(r->c, content, len) == (int64_t)id, "%s: added %u's content -> %" PRId64 ", its id %u",
              f->name, i, toks_token_to_id(r->c, content, len), id);
        h = fnv(fnv_u32(fnv_u32(fnv_u32(h, id), (uint32_t)fl & ~TOKS_ID_BYTE), (uint32_t)len), content, len);
        prev = id;
    }
    CHECK(i == f->n_added && h == f->added_digest, "%s: %u added tokens (digest %016" PRIx64 "), hf %u (%016" PRIx64 ")",
          f->name, i, h, f->n_added, f->added_digest);
    uint32_t n_flag = 0;                                    /* every ADDED id is listed */
    for (uint32_t id = 0; id < n_ids; id++) { n_flag += (toks_id_flags(r->c, id) & TOKS_ID_ADDED) != 0; }
    CHECK(n_flag == i, "%s: %u ids are ADDED, %u listed", f->name, n_flag, i);
    const void *cp = NULL;
    uint64_t ln = 7u;
    uint32_t id = 7u;
    CHECK(toks_added(NULL, 0u, &cp, &ln, &id) == TOKS_E_ARG && toks_added(r->c, i, &cp, &ln, &id) == TOKS_E_ARG &&
          toks_added(r->c, UINT32_MAX, &cp, &ln, &id) == TOKS_E_ARG && cp == NULL && ln == 7u && id == 7u,
          "%s: added errors", f->name);
    CHECK(i == 0u || toks_added(r->c, 0u, NULL, NULL, NULL) > 0, "%s: added with NULL outs", f->name);
    g_added += i;
    g_special_rule += differ;
}

/* ---- 4. TOKS_DECODE_RAW ---------------------------------------------------------------------------------------- */

/* decode under RAW against the rule: valid utf-8 is decode's; byte-level is the kept tokens' bytes */
static uint8_t *room(run *r, uint64_t need)
{
    if (need > r->bcap) {
        free(r->bytes);
        r->bcap = need + need / 2u + 64u;
        r->bytes = (uint8_t *)malloc((size_t)r->bcap);
    }
    return r->bytes;
}

static void raw_rule(const pr_file *f, run *r, const uint32_t *ids, uint64_t n, uint32_t algo, const char *what)
{
    for (uint32_t skip = 0; skip < 2u; skip++) {
        int64_t a = toks_decode(r->c, ids, n, TOKS_DECODE_RAW | skip, NULL, 0u);
        int64_t b = toks_decode(r->c, ids, n, skip, NULL, 0u);
        CHECK(a >= 0 && b >= 0, "%s: %s: decode %" PRId64 " / %" PRId64, f->name, what, a, b);
        if (a < 0 || b < 0) { return; }
        uint8_t *x = room(r, (uint64_t)a + (uint64_t)b + 1u), *y = x + a;
        CHECK(toks_decode(r->c, ids, n, TOKS_DECODE_RAW | skip, x, (uint64_t)a) == a &&
              toks_decode(r->c, ids, n, skip, y, (uint64_t)b) == b, "%s: %s: decode again", f->name, what);
        int valid = utf8_ok(x, (uint64_t)a);
        CHECK(!valid || (a == b && memcmp(x, y, (size_t)a) == 0),
              "%s: %s (skip %u): a raw decode that is utf-8 is not decode's", f->name, what, skip);
        CHECK(valid || algo != TOKS_ALGO_WORDPIECE, "%s: %s: wordpiece raw not utf-8", f->name, what);
        g_raw_bad += !valid;
        if (algo == TOKS_ALGO_BPE_BYTELEVEL && skip == 0u) {   /* the tokens' bytes, every one */
            uint64_t m = 0;
            int same = 1;
            for (uint64_t i = 0; i < n && same; i++) {
                uint64_t k = 0;
                const uint8_t *p = toks_token(r->c, ids[i], &k);
                same = m + k <= (uint64_t)a && (k == 0u || memcmp(x + m, p, (size_t)k) == 0);
                m += k;
            }
            CHECK(same && m == (uint64_t)a, "%s: %s: byte-level raw is not the tokens' bytes", f->name, what);
        }
        g_raw_inv++;
    }
}

static void check_raw(const pr_file *f, run *r, uint32_t n_ids, uint32_t algo)
{
    for (uint32_t k = 0; k < f->n_raw; k++) {               /* the references */
        const pr_raw *x = &f->raw[k];
        uint8_t *o = room(r, (uint64_t)x->len + 64u);
        int64_t a = toks_decode(r->c, x->ids, x->n, x->flags, o, r->bcap);
        CHECK(a == (int64_t)x->len && fnv(FNV0, o, (uint64_t)(a > 0 ? a : 0)) == x->digest,
              "%s: raw case %u: %" PRId64 " bytes, reference %u", f->name, k, a, x->len);
        g_raw++;
    }
    for (uint32_t t = 0; t < PR_N_TEXTS + 1u; t++) {         /* the texts' ids */
        uint64_t len;
        const char *s = text_of(f, t, &len);
        int64_t n = enc(r, s, len, TOKS_NO_TRUNCATE | TOKS_NO_PAD);
        if (n > 0 && (uint64_t)n <= r->cap) {
            char what[32];
            snprintf(what, sizeof what, "text %u", t);
            raw_rule(f, r, r->out, (uint64_t)n, algo, what);
        }
    }
    uint32_t ids[24];
    for (uint32_t k = 0; k < 64u && n_ids > 0u; k++) {      /* random ids */
        uint32_t n = 1u + (uint32_t)(next() % 24u);
        for (uint32_t i = 0; i < n; i++) { ids[i] = (uint32_t)(next() % n_ids); }
        raw_rule(f, r, ids, n, algo, "random ids");
    }
    /* every cap: the exact prefix, nothing at or past cap (one random sequence) */
    for (uint32_t i = 0; i < 24u; i++) { ids[i] = (uint32_t)(next() % (n_ids ? n_ids : 1u)); }
    int64_t full = toks_decode(r->c, ids, 24u, TOKS_DECODE_RAW, NULL, 0u);
    uint8_t *want = full >= 0 ? room(r, 2u * (uint64_t)full + 4u) : NULL, *o = want != NULL ? want + full + 2 : NULL;
    CHECK(full >= 0 && toks_decode(r->c, ids, 24u, TOKS_DECODE_RAW, want, (uint64_t)full) == full, "%s: raw 24 ids",
          f->name);
    for (int64_t cap = 0; o != NULL && cap <= full + 1; cap++) {
        memset(o, 0xEE, (size_t)full + 2u);
        int64_t g = toks_decode(r->c, ids, 24u, TOKS_DECODE_RAW, cap != 0 ? o : NULL, (uint64_t)cap);
        uint64_t w = (uint64_t)(cap < full ? cap : full);
        CHECK(g == full && memcmp(o, want, (size_t)w) == 0 && (cap > full || o[cap] == 0xEEu),
              "%s: raw decode cap %" PRId64 ": %" PRId64 " (want %" PRId64 ")", f->name, cap, g, full);
    }
    uint8_t one[8];
    CHECK(toks_decode(r->c, ids, 1u, 4u, one, sizeof one) == TOKS_E_ARG &&
          toks_decode(r->c, ids, 1u, TOKS_DECODE_RAW | 8u, one, sizeof one) == TOKS_E_ARG, "%s: decode flags", f->name);
    uint32_t bad = n_ids;
    CHECK(toks_decode(r->c, &bad, 1u, TOKS_DECODE_RAW, one, sizeof one) == TOKS_E_ID, "%s: raw id n_ids", f->name);
}

/* ---- 5. toks_info -------------------------------------------------------------------------------------------- */

static void check_info(const pr_file *f, run *r, const toks_info *in)
{
    const uint32_t got[10] = { in->trunc_on, in->trunc_max, in->trunc_stride, in->pad_on, in->pad_fixed, in->pad_id,
                               in->pad_type_id, in->pad_len, in->pad_multiple, in->pad_left };
    static const char *const NAME[10] = { "trunc_on", "trunc_max", "trunc_stride", "pad_on", "pad_fixed", "pad_id",
                                          "pad_type_id", "pad_len", "pad_multiple", "pad_left" };
    for (int k = 0; k < 10; k++) {
        CHECK(got[k] == f->info[k], "%s: info %s %u, hf %u", f->name, NAME[k], got[k], f->info[k]);
    }
    CHECK(in->n_template_prefix == f->n_prefix && in->n_template_suffix == f->n_tmpl - f->n_prefix &&
          in->seq_type_id == f->seq_type && in->rsv2 == 0u && in->size == (uint32_t)sizeof(toks_info) &&
          in->abi_minor == TOKS_ABI_MINOR, "%s: info template %u / %u, seq type %u", f->name, in->n_template_prefix,
          in->n_template_suffix, in->seq_type_id);
    toks_info old[2];                                       /* abi 0.3's size: the fields before trunc_on, no more */
    memset(old, 0xA5, sizeof old);
    old[0].size = (uint32_t)offsetof(toks_info, trunc_on);
    CHECK(toks_get_info(r->c, &old[0]) == 0 && old[0].size == 184u &&
          memcmp((const uint8_t *)&old[0] + 4, (const uint8_t *)in + 4, 180u) == 0 &&
          ((const uint8_t *)&old[0])[184] == 0xA5u && ((const uint8_t *)&old[0])[239] == 0xA5u, "%s: info size 184",
          f->name);
    static const uint32_t BAD[] = { 0u, 4u, 183u, 185u, 239u, 241u, 4096u };
    for (size_t k = 0; k < sizeof BAD / sizeof BAD[0]; k++) {
        toks_info x;
        memset(&x, 0, sizeof x);
        x.size = BAD[k];
        CHECK(toks_get_info(r->c, &x) == TOKS_E_ARG, "%s: info size %u", f->name, BAD[k]);
    }
}

/* ---- one file ------------------------------------------------------------------------------------------------- */

static void test_file(const pr_file *f)
{
    const char *path = path_of(f);
    if (f->src != 2) {
        FILE *fp = fopen(path, "rb");
        if (fp == NULL) { g_absent++; printf("  %-34s SKIP: %s absent\n", f->name, path); return; }
        fclose(fp);
    }
    toks_diag diag;
    memset(&diag, 0, sizeof diag);
    toks_load_opts lo = { (uint32_t)sizeof(toks_load_opts), TOKS_TIER_AUTO, 0u, 0u, &diag };
    run r;
    memset(&r, 0, sizeof r);
    int64_t lr = toks_load(&r.c, path, &lo);
    if (lr != 0 && f->src == 2) { g_absent++; printf("  %-34s SKIP: %s absent (%s)\n", f->name, path, diag.what); return; }
    if (lr != 0) { g_refused++; printf("  %-34s refused at load: %s (no primitive to check)\n", f->name, diag.what); return; }
    toks_info in;
    memset(&in, 0, sizeof in);
    in.size = (uint32_t)sizeof in;
    CHECK(toks_get_info(r.c, &in) == 0, "%s: info", f->name);
    char hex[65];
    for (int i = 0; i < 32; i++) { snprintf(hex + 2 * i, 3, "%02x", in.source_sha256[i]); }
    if (f->sha[0] != 0 && strcmp(hex, f->sha) != 0) {
        g_other++;
        printf("  %-34s SKIP: not the pinned file (sha-256 %.16s..., pinned %.16s...)\n", f->name, hex, f->sha);
        toks_unload(r.c);
        return;
    }
    uint64_t maxlen = T3_LEN + 4096u;
    uint64_t sb = toks_scratch_bytes(r.c, maxlen, 0u);
    r.scr = malloc((size_t)sb);
    CHECK(r.scr != NULL && toks_scratch_init(r.c, r.scr, sb, 0u) == 0, "%s: scratch", f->name);
    r.cap = toks_encode_bound(r.c, maxlen);
    r.out = (uint32_t *)malloc(4u * (size_t)r.cap);
    check_info(f, &r, &in);
    check_template(f, &r);
    check_added(f, &r, in.n_ids);
    check_flags(f, &r);
    check_raw(f, &r, in.n_ids, in.algorithm);
    printf("  %-34s ok: template %u (prefix %u, text type %u), %u added, %u hf encode cases, %u raw references; "
           "trunc %u / %u, pad %u (fixed %u, %u, multiple %u, %s)\n", f->name, f->n_tmpl, f->n_prefix, f->seq_type,
           f->n_added, f->n_enc, f->n_raw, in.trunc_on, in.trunc_max, in.pad_on, in.pad_fixed, in.pad_len, in.pad_multiple,
           in.pad_left ? "left" : "right");
    g_files++;
    free(r.bytes);
    free(r.out);
    free(r.scr);
    toks_unload(r.c);
}

int main(void)
{
    uint64_t l2 = (uint64_t)strlen(PR_T2);
    T3_LEN = l2 * PR_T3_REPEAT;
    T3 = (char *)malloc((size_t)T3_LEN + 1u);
    for (uint32_t i = 0; i < PR_T3_REPEAT; i++) { memcpy(T3 + l2 * i, PR_T2, (size_t)l2); }
    T3[T3_LEN] = 0;
    for (uint32_t i = 0; i < PR_N_FILES; i++) { test_file(&PR_FILE[i]); }
    printf("test_primitives: %ld checks, %" PRIu64 " files (%" PRIu64 " absent, %" PRIu64 " other bytes, %" PRIu64
           " refused at load); hf encode cases %" PRIu64 ", flag no-ops %" PRIu64 ", cuts %" PRIu64 "; templates %" PRIu64
           "; added %" PRIu64 " (%" PRIu64 " with toks_id_flags' SPECIAL rule apart); raw references %" PRIu64
           ", raw rule %" PRIu64 " (%" PRIu64 " not utf-8); %d failures\n", checks, g_files, g_absent, g_other, g_refused,
           g_enc, g_inv, g_cuts, g_tmpl, g_added, g_special_rule, g_raw, g_raw_inv, g_raw_bad, failures);
    free(T3);
    return failures != 0;
}
