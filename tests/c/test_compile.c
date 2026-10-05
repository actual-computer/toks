/*
 * test_compile.c: config.c + compile.c (toks_config_parse, toks_compile).
 *   - hand fixtures tests/data/compile/<name>.json against <name>.expect (gen.py wrote both and checked them
 *     against hf tokenizers 0.23.2: ids, specials, template ids, each added token's decode bytes);
 *   - refuse.txt: every file there is refused with its code and a diag naming the feature;
 *   - the pinned real files ($TOKS_TOKENIZER_CACHE, default ~/.cache/toks/tokenizers): gpt2, llama3, qwen3 and
 *     qwen38 (both NFC) compiled and checked; each one skipped when absent;
 *   - the normalizer flag: a fixture with {"type":"NFC"} sets cfg.nfc, one with NFD / NFKC is refused by name;
 *   - invariants on everything compiled: token bytes round-trip through an independent byte-level alphabet,
 *     the special bitmap, the added-token index layout, and K1 (toks_k1_added_find_c) against a brute-force
 *     leftmost-longest finder on every token and on random texts;
 *   - hostile smoke: every truncation and random byte mutations of a fixture never crash, and whatever is
 *     still accepted passes the invariants.
 * Run from the repository root (make test does).
 */
#include "core.h"
#include "classes.h"
#include "compile.h"
#include "kernels.h"
#include "norm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "lines.inc"

static int fails;
static uint64_t checks;
#define CHECK(c, ...)                                                                                      \
    do {                                                                                                   \
        checks++;                                                                                          \
        if (!(c)) {                                                                                        \
            if (++fails <= 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
        }                                                                                                  \
    } while (0)

/* ---- an independent byte-level alphabet (gpt-2 bytes_to_unicode) ------------------------------------- */

static uint32_t B2U[256];

static void alphabet_init(void)
{
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256u; b++) {
        int self = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || b >= 0xAEu;
        B2U[b] = self ? b : 0x100u + n++;
    }
}

static uint32_t put_utf8(uint32_t cp, uint8_t *o)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) { o[0] = (uint8_t)(0xC0u | cp >> 6); o[1] = (uint8_t)(0x80u | (cp & 0x3Fu)); return 2; }
    o[0] = (uint8_t)(0xE0u | cp >> 12); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 3;
}

/* the bytes hf's ByteLevel decoder gives one token string: the alphabet image when every char maps,
 * else the string's own utf-8. out room >= n. */
static uint32_t decode_bytes(const uint8_t *s, uint32_t n, uint8_t *out)
{
    uint32_t i = 0, m = 0;
    while (i < n) {
        uint32_t k = toks_utf8_len(s + i, n - i);
        if (k == 0u) { break; }
        uint32_t cp = (k == 1u) ? s[i] : toks_cp_decode(s + i, k);
        uint32_t b = 0;
        while (b < 256u && B2U[b] != cp) { b++; }
        if (b == 256u) { break; }
        out[m++] = (uint8_t)b;
        i += k;
    }
    if (i == n) { return m; }
    memcpy(out, s, n);
    return n;
}

/* ---- loading ----------------------------------------------------------------------------------------- */

typedef struct loaded {
    toks_config     cfg;
    struct toks_ctx ctx;
    toks_arena      ar;
    toks_err        err;
    int64_t         r;
    int             stage;      /* 0 ok, 1 config refused, 2 compile refused */
    double          ms_parse, ms_compile;
} loaded;

static double now_ms(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void load(loaded *L, const uint8_t *data, uint64_t len, uint8_t *arena, uint64_t arena_len)
{
    memset(L, 0, sizeof(*L));
    L->ar.base = arena;
    L->ar.len = arena_len;
    double t0 = now_ms();
    L->r = toks_config_parse(data, len, &L->ar, &L->cfg, &L->err);
    double t1 = now_ms();
    L->ms_parse = t1 - t0;
    if (L->r != 0) { L->stage = 1; return; }
    L->r = toks_compile(&L->cfg, &L->ctx, &L->ar);
    L->ms_compile = now_ms() - t1;
    if (L->r != 0) { L->stage = 2; }
    CHECK(L->ar.pos <= toks_config_arena_bound(len), "parse arena %llu > bound %llu", (unsigned long long)L->ar.pos,
          (unsigned long long)toks_config_arena_bound(len));
}

static void unload(loaded *L);

/* the arena bound holds for the shapes that cost json.c and config.c the most per source byte */
static void arena_bound(void)
{
    enum { N = 60000 };
    uint64_t cap_s = 16u * N + 4096u;
    char *s = malloc(cap_s);
    for (int shape = 0; shape < 4; shape++) {
        uint64_t n = 0;
        n += (uint64_t)snprintf(s + n, cap_s - n, shape == 3 ? "{\"model\":{\"merges\":[],\"vocab\":{" : "{\"a\":%s", shape == 2 ? "{" : "[");
        for (uint32_t i = 0; i < N; i++) {
            if (shape == 0) { n += (uint64_t)snprintf(s + n, cap_s - n, "%s0", i ? "," : ""); }
            else if (shape == 1) { n += (uint64_t)snprintf(s + n, cap_s - n, "%s\"\"", i ? "," : ""); }
            else if (shape == 2) { n += (uint64_t)snprintf(s + n, cap_s - n, "%s\"\":0", i ? "," : ""); }
            else { n += (uint64_t)snprintf(s + n, cap_s - n, "%s\"%c%c%c\":%u", i ? "," : "", 'A' + i % 26, 'A' + i / 26 % 26, 'A' + i / 676 % 26, i); }
        }
        n += (uint64_t)snprintf(s + n, cap_s - n, shape == 3 ? "}}}" : (shape == 2 ? "}}" : "]}"));
        uint64_t cap = toks_config_arena_bound(n);
        uint8_t *ar = malloc(cap);
        loaded L;
        load(&L, (const uint8_t *)s, n, ar, cap);
        CHECK(L.r == TOKS_E_FORMAT || L.r == TOKS_E_UNSUPPORTED, "arena shape %d: %lld %s", shape, (long long)L.r,
              L.err.what ? L.err.what : "");
        printf("arena bound: shape %d, %llu bytes of json -> %.1fx of the source used (bound 32x)\n", shape,
               (unsigned long long)n, (double)L.ar.pos / (double)n);
        unload(&L);
        free(ar);
    }
    free(s);
}

static void unload(loaded *L)
{
    if (L->ctx.mem_tables != NULL) { toks_plat_arena_free(L->ctx.mem_tables, L->ctx.mem_tables_len); }
    L->ctx.mem_tables = NULL;
}

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)n + 1u);
    if (p == NULL || fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(p); return NULL; }
    fclose(f);
    p[n] = 0;
    *len = (uint64_t)n;
    return p;
}

/* ---- invariants on a compiled context ---------------------------------------------------------------- */

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static void ref_find(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t phase,
                     uint64_t *ms, uint64_t *me, uint64_t *ment)
{
    for (uint64_t i = pos; i < len; i++) {
        uint64_t best = 0, be = 0;
        for (uint64_t e = 0; e < t->add_n; e++) {
            const toks_added_entry *x = &t->add_entries[e];
            if (x->phase != phase || x->len <= best || i + x->len > len) { continue; }
            if (memcmp(t->add_bytes + x->off, text + i, x->len) == 0) { best = x->len; be = e; }
        }
        if (best != 0u) { *ms = i; *me = i + best; *ment = be; return; }
    }
    *ms = len; *me = len; *ment = 0;
}

static void k1_vs_ref(const char *name, const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t phase)
{
    uint64_t pos = 0;
    while (pos <= len) {
        toks_k1_match m = { 0, 0, 0, 0 };
        toks_k1_args a;
        memset(&a, 0, sizeof(a));
        a.text = text; a.len = len; a.pos = pos; a.phase = phase; a.m = &m; a.cap = 1;
        uint64_t r = toks_k1_added_find_c(t, &a);
        uint64_t ms, me, ment;
        ref_find(t, text, len, pos, phase, &ms, &me, &ment);
        uint64_t gs = r ? m.start : len, ge = r ? m.end : len;
        CHECK(r == (ms < len) && a.next == me && gs == ms && ge == me && (ms == len || m.entry == ment),
              "%s k1 phase %llu pos %llu: got [%llu,%llu) e%llu want [%llu,%llu) e%llu", name,
              (unsigned long long)phase, (unsigned long long)pos, (unsigned long long)gs, (unsigned long long)ge,
              (unsigned long long)m.entry, (unsigned long long)ms, (unsigned long long)me, (unsigned long long)ment);
        if (ms == len) { break; }
        pos = (rnd() & 3u) == 0u ? ms + 1u : me;   /* resume at the raw end, sometimes inside the match */
    }
}

static void check_k1(const char *name, const toks_tables *t, uint32_t rounds)
{
    if (t->add_n == 0u) { return; }
    uint8_t text[600];
    for (uint64_t e = 0; e < t->add_n; e++) {                 /* every token alone: found as itself */
        const toks_added_entry *x = &t->add_entries[e];
        toks_k1_match m = { 9, 9, 9, 9 };
        toks_k1_args a;
        memset(&a, 0, sizeof(a));
        a.text = t->add_bytes + x->off; a.len = x->len; a.phase = x->phase; a.m = &m; a.cap = 1;
        uint64_t r = toks_k1_added_find_c(t, &a);
        CHECK(r == 1u && m.start == 0u && m.end == x->len && m.entry == e, "%s k1 entry %llu alone", name,
              (unsigned long long)e);
    }
    for (uint32_t r = 0; r < rounds; r++) {                   /* random texts of tokens, pieces, noise */
        uint64_t len = 0;
        while (len < 300u) {
            const toks_added_entry *x = &t->add_entries[rnd() % t->add_n];
            uint64_t k = rnd() % 4u;
            if (k == 0u) { memcpy(text + len, t->add_bytes + x->off, x->len); len += x->len; }
            else if (k == 1u) { uint64_t n = 1u + rnd() % x->len; memcpy(text + len, t->add_bytes + x->off, n); len += n; }
            else if (k == 2u) { text[len++] = t->add_bytes[x->off]; }
            else { text[len++] = (uint8_t)rnd(); }
        }
        k1_vs_ref(name, t, text, len, rnd() & 1u);
    }
}

static void check_index(const char *name, const toks_config *c, const toks_tables *t)
{
    if (c->n_added == 0u) {
        CHECK(t->add_n == 0u && t->add_phases == 0u && t->add_entries == NULL && t->add_index == NULL &&
              t->add_shufti == NULL && t->add_single == NULL && t->add_cand == NULL && t->add_bytes == NULL,
              "%s: no added tokens, add_* must be empty", name);
        return;
    }
    CHECK(t->add_n == c->n_added, "%s add_n %llu", name, (unsigned long long)t->add_n);
    uint32_t e = 0;
    uint64_t phases = 0, n_long = 0;
    for (uint32_t p = 0; p < 2u; p++) {
        for (uint32_t i = 0; i < c->n_added && e < t->add_n; i++) {
            const toks_cfg_added *a = &c->added[i];
            if (a->normalized != p) { continue; }
            const toks_added_entry *x = &t->add_entries[e];
            const uint8_t *s = t->add_bytes + x->off;
            CHECK(x->phase == p && x->id == a->id && x->len == a->len && memcmp(s, a->content, a->len) == 0,
                  "%s entry %u (grouped by phase, file order)", name, e);
            CHECK(((x->flags & TOKS_AF_SPECIAL) != 0) == (a->special != 0) &&
                  ((x->flags & TOKS_AF_NORMALIZED) != 0) == (p == 1u) &&
                  (x->flags & (TOKS_AF_LSTRIP | TOKS_AF_RSTRIP | TOKS_AF_SINGLE_WORD)) == 0, "%s entry %u flags", name, e);
            CHECK((t->add_shufti[p * 32u + (s[0] & 15u)] & t->add_shufti[p * 32u + 16u + (s[0] >> 4)]) != 0,
                  "%s entry %u: shufti false negative", name, e);
            if (x->len == 1u) {
                CHECK(t->add_single[p * 256u + s[0]] == e + 1u, "%s entry %u add_single", name, e);
            } else {
                uint64_t slot = t->add_index[p * 65536u + (s[0] | (uint32_t)s[1] << 8)];
                uint32_t seen = 0;
                for (uint64_t j = 0; j < (slot >> 32); j++) { seen += t->add_cand[(uint32_t)slot + j] == e; }
                CHECK(seen == 1u, "%s entry %u listed %u times in its bucket", name, e, seen);
                n_long++;
            }
            phases |= 1ull << p;
            e++;
        }
    }
    CHECK(t->add_phases == phases, "%s add_phases", name);
    uint64_t listed = 0;
    for (uint32_t s = 0; s < 2u * 65536u; s++) {
        uint64_t slot = t->add_index[s];
        for (uint64_t j = 0; j < (slot >> 32); j++) {
            const toks_added_entry *x = &t->add_entries[t->add_cand[(uint32_t)slot + j]];
            const uint8_t *b = t->add_bytes + x->off;
            CHECK(x->phase == (s >> 16) && x->len >= 2u && (b[0] | (uint32_t)b[1] << 8) == (s & 0xFFFFu),
                  "%s bucket %u holds a foreign entry", name, s);
            if (j > 0u) {
                CHECK(t->add_entries[t->add_cand[(uint32_t)slot + j - 1u]].len >= x->len,
                      "%s bucket %u not longest first", name, s);
            }
        }
        listed += slot >> 32;
    }
    CHECK(listed == n_long, "%s buckets list %llu entries, want %llu", name, (unsigned long long)listed,
          (unsigned long long)n_long);
    for (uint32_t s = 0; s < 512u; s++) {
        uint32_t v = t->add_single[s];
        if (v != 0u) {
            const toks_added_entry *x = &t->add_entries[v - 1u];
            CHECK(x->len == 1u && x->phase == s / 256u && t->add_bytes[x->off] == s % 256u, "%s add_single[%u]", name, s);
        }
    }
}

/* 1 when the pattern's class tables fold \p{M} into letters (the qwen 3.5 entry's class_flags; dsv3's byte) */
static uint32_t marks_folded(const toks_config *c)
{
    return (c->pattern != NULL &&
            (c->pattern->class_flags & (TOKS_CLASSES_MARKS_ARE_LETTERS | TOKS_CLASSES_DSV3)) != 0u) ? 1u : 0u;
}

static void check_ctx(const char *name, const loaded *L, uint32_t k1_rounds)
{
    const toks_config *c = &L->cfg;
    const toks_tables *t = &L->ctx.t;
    CHECK(t->magic == TOKS_TABLES_MAGIC && t->version == TOKS_TABLES_VERSION && t->algo == TOKS_ALGO_BPE_BYTELEVEL,
          "%s header", name);
    CHECK(t->n_ids == c->n_ids && c->n_ids >= c->n_vocab && c->n_ids < TOKS_MAX_IDS, "%s n_ids %u", name, t->n_ids);
    CHECK(((t->flags & TOKS_TF_IGNORE_MERGES) != 0) == (c->ignore_merges != 0) && t->n_merges == 0u &&
          t->byte2id == NULL && t->vhash == NULL && t->words == NULL, "%s flags / bpe fields", name);
    CHECK(L->ctx.dec_byte_level == 1u, "%s dec_byte_level", name);
    if (c->pattern != NULL) {
        CHECK(t->tmpl == c->pattern->tmpl && t->tmpl_params == c->pattern->params, "%s template", name);
        CHECK(toks_tmpl_invalid(t) == NULL, "%s template invariants: %s", name, toks_tmpl_invalid(t));
        CHECK(t->cls_ascii != NULL && t->cls_stage1 != NULL && t->cls_stage2 != NULL && t->cls_nblocks > 0u,
              "%s class tables", name);
        if (t->cls_ascii != NULL) {
            CHECK((t->cls_ascii['a'] & 7u) == TOKS_C_L && (t->cls_ascii['Z'] & 7u) == TOKS_C_L &&
                  (t->cls_ascii['7'] & 7u) == TOKS_C_N && (t->cls_ascii[' '] & 7u) == TOKS_C_WS &&
                  (t->cls_ascii['\t'] & 7u) == TOKS_C_WS && (t->cls_ascii['\n'] & 7u) == TOKS_C_NL &&
                  (t->cls_ascii['\r'] & 7u) == TOKS_C_NL && (t->cls_ascii['!'] & 7u) == TOKS_C_P &&
                  (t->cls_ascii['\''] & 7u) == TOKS_C_P, "%s ascii classes", name);
            CHECK((toks_cls_cp(t, 0xE9u) & 7u) == TOKS_C_L && (toks_cls_cp(t, 0x3000u) & 7u) == TOKS_C_WS &&
                  (toks_cls_cp(t, 0x0661u) & 7u) == TOKS_C_N, "%s classes", name);
            if (t->tmpl == TOKS_TMPL_DSV3) {           /* docs/templates/dsv3.md §2: the byte is base | CJK only */
                CHECK(toks_cls_cp(t, 0x0301u) == TOKS_C_L && toks_cls_cp(t, 0x017Fu) == TOKS_C_L &&
                      toks_cls_cp(t, 0x3042u) == (TOKS_C_L | TOKS_C_CJK) && toks_cls_cp(t, 0x30FBu) == (TOKS_C_P | TOKS_C_CJK) &&
                      toks_cls_cp(t, 0x00ADu) == TOKS_C_X && t->cls_ascii[0] == TOKS_C_X && t->cls_ascii['$'] == TOKS_C_P &&
                      (t->flags & (TOKS_TF_CJK_L | TOKS_TF_CJK_D)) == TOKS_TF_CJK_D, "%s dsv3 classes", name);
            } else {
                CHECK((toks_cls_cp(t, 0x017Fu) & TOKS_C_FOLD_S) != 0 &&
                      (toks_cls_cp(t, 0x0301u) & 7u) == (marks_folded(c) ? TOKS_C_L : TOKS_C_P) &&
                      (toks_cls_cp(t, 0x0301u) & TOKS_C_MARK) != 0, "%s classes (marks folded %u)", name, marks_folded(c));
                CHECK((t->flags & TOKS_TF_CJK_L) != 0u && toks_compile_cls_flags(t) == (TOKS_TF_CJK_L | TOKS_TF_CJK_B),
                      "%s TOKS_TF_CJK_L", name);
            }
        }
    } else {
        CHECK(t->tmpl == TOKS_TMPL_NONE && t->tmpl_params == 0u && t->cls_ascii == NULL &&
              (t->flags & TOKS_TF_CJK_L) == 0u, "%s no template", name);
    }

    /* token bytes: vocab ids round-trip to their strings; added-only ids hold their decode bytes */
    int32_t *owner = malloc(((size_t)t->n_ids + 1u) * sizeof(int32_t));
    uint8_t *spec_want = calloc((size_t)t->n_ids + 1u, 1);
    for (uint32_t id = 0; id < t->n_ids; id++) { owner[id] = -1; }
    for (uint32_t i = 0; i < c->n_added; i++) {
        owner[c->added[i].id] = (int32_t)i;
        if (c->added[i].special) { spec_want[c->added[i].id] = 1; }
    }
    CHECK(t->tok_off[0] == 0u, "%s tok_off[0]", name);
    uint8_t buf[4 * 65536];
    for (uint32_t id = 0; id < t->n_ids; id++) {
        uint32_t o = t->tok_off[id], n = t->tok_off[id + 1u] - o;
        if (t->tok_off[id + 1u] < o) { CHECK(0, "%s tok_off not monotonic at %u", name, id); break; }
        if (id < c->n_vocab && toks_alpha_bytes(c->vocab[id], c->vocab_len[id], NULL) < 0) {
            /* a string outside the alphabet: its decode is its own utf-8 (hf's ByteLevel decoder) */
            CHECK(n == c->vocab_len[id] && memcmp(t->tok_bytes + o, c->vocab[id], n) == 0, "%s id %u (not an alphabet "
                  "string) is not its utf-8", name, id);
        } else if (id < c->n_vocab) {
            uint32_t m = 0;
            for (uint32_t k = 0; k < n; k++) { m += put_utf8(B2U[t->tok_bytes[o + k]], buf + m); }
            CHECK(m == c->vocab_len[id] && memcmp(buf, c->vocab[id], m) == 0, "%s id %u does not round-trip", name, id);
        } else if (owner[id] >= 0) {
            const toks_cfg_added *a = &c->added[owner[id]];
            uint32_t m = decode_bytes(a->content, a->len, buf);
            CHECK(m == n && memcmp(buf, t->tok_bytes + o, n) == 0, "%s added id %u bytes", name, id);
        } else {
            CHECK(n == 0u, "%s id %u has no token but %u bytes", name, id, n);
        }
        const uint8_t *bm = (const uint8_t *)(const void *)L->ctx.special_ids;
        CHECK(((bm[id >> 3] >> (id & 7u)) & 1u) == spec_want[id], "%s special bit %u", name, id);
    }
    free(owner);
    free(spec_want);

    check_index(name, c, t);
    check_k1(name, t, k1_rounds);

    uint32_t k = 0;                                           /* pp_ids: prefix then suffix */
    for (uint32_t i = 0; i < c->n_pp_single; i++) {
        if (c->pp_single[i].kind == TOKS_PPS_TOK) {
            CHECK(L->ctx.pp_ids[k] == c->pp_single[i].id, "%s pp_ids[%u]", name, k);
            k++;
        }
    }
    for (; k < 64u; k++) { CHECK(L->ctx.pp_ids[k] == 0u, "%s pp_ids[%u] beyond the template", name, k); }
}

/* ---- the hand fixtures and their expectations -------------------------------------------------------- */

static uint32_t hexbytes(const char *h, uint8_t *out)
{
    uint32_t n = 0;
    while (h[0] != 0 && h[1] != 0 && h[0] != ' ' && h[0] != '\n') {
        unsigned v;
        if (sscanf(h, "%2x", &v) != 1) { break; }
        out[n++] = (uint8_t)v;
        h += 2;
    }
    return n;
}

static uint32_t ids_of(const char *s, uint32_t *ids)
{
    uint32_t n = 0;
    char *end;
    for (;;) {
        unsigned long v = strtoul(s, &end, 0);
        if (end == s || n >= 64u) { return n; }
        ids[n++] = (uint32_t)v;
        s = end;
    }
}

static void fixture(const char *name, uint8_t *arena, uint64_t arena_len)
{
    char path[512];
    uint64_t len = 0;
    snprintf(path, sizeof(path), "tests/data/compile/%s.json", name);
    uint8_t *data = slurp(path, &len);
    snprintf(path, sizeof(path), "tests/data/compile/%s.expect", name);
    FILE *f = fopen(path, "rb");
    CHECK(data != NULL && f != NULL, "%s: fixture files missing (run from the repository root)", name);
    if (data == NULL || f == NULL) { free(data); if (f) { fclose(f); } return; }
    loaded L;
    load(&L, data, len, arena, arena_len);
    CHECK(L.r == 0, "%s refused: %lld %s", name, (long long)L.r, L.err.what ? L.err.what : "");
    if (L.r == 0) {
        check_ctx(name, &L, 400);
        const toks_tables *t = &L.ctx.t;
        char line[8192];
        uint32_t e = 0, pp_want[128], n_pp = 0;
        td_at at = { path, 0 };
        while (fgets(line, sizeof(line), f) != NULL) {
            at.line++;
            if (!td_line(&at, line, sizeof line)) { fails++; break; }
            char key[32];
            int at = 0;
            if (sscanf(line, "%31s%n", key, &at) != 1) { continue; }
            const char *rest = line + at;
            uint32_t ids[64];
            uint32_t n = ids_of(rest, ids);
            if (strcmp(key, "n_vocab") == 0) { CHECK(L.cfg.n_vocab == ids[0], "%s n_vocab %u", name, L.cfg.n_vocab); }
            else if (strcmp(key, "n_ids") == 0) { CHECK(t->n_ids == ids[0], "%s n_ids %u want %u", name, t->n_ids, ids[0]); }
            else if (strcmp(key, "tmpl") == 0) { CHECK(t->tmpl == ids[0] && t->tmpl_params == ids[1], "%s tmpl %u %#x", name, t->tmpl, t->tmpl_params); }
            else if (strcmp(key, "marks") == 0) { CHECK(marks_folded(&L.cfg) == ids[0], "%s marks", name); }
            else if (strcmp(key, "ignore_merges") == 0) { CHECK(((t->flags & TOKS_TF_IGNORE_MERGES) != 0) == (ids[0] != 0), "%s ignore_merges", name); }
            else if (strcmp(key, "prefix") == 0 || strcmp(key, "suffix") == 0) {
                for (uint32_t i = 0; i < n; i++) { pp_want[n_pp++] = ids[i]; }
            } else if (strcmp(key, "special") == 0) {
                const uint8_t *bm = (const uint8_t *)(const void *)L.ctx.special_ids;
                uint32_t set = 0;
                for (uint32_t id = 0; id < t->n_ids; id++) { set += (bm[id >> 3] >> (id & 7u)) & 1u; }
                CHECK(set == n, "%s special bits %u want %u", name, set, n);
                for (uint32_t i = 0; i < n; i++) { CHECK((bm[ids[i] >> 3] >> (ids[i] & 7u)) & 1u, "%s special %u", name, ids[i]); }
            } else if (strcmp(key, "added") == 0) {
                unsigned id, ph, sp;
                char ch[600], bh[600];
                uint8_t cb[300], bb[300];
                if (sscanf(rest, "%u %u %u %599s %599s", &id, &ph, &sp, ch, bh) != 5 || e >= t->add_n) {
                    CHECK(0, "%s expect line: %s", name, line);
                    continue;
                }
                uint32_t cn = hexbytes(ch, cb), bn = hexbytes(bh, bb);
                const toks_added_entry *x = &t->add_entries[e];
                CHECK(x->id == id && x->phase == ph && ((x->flags & TOKS_AF_SPECIAL) != 0) == (sp != 0) && x->len == cn &&
                      memcmp(t->add_bytes + x->off, cb, cn) == 0, "%s entry %u is not hf's id %u", name, e, id);
                uint32_t o = t->tok_off[id];
                CHECK(t->tok_off[id + 1u] - o == bn && memcmp(t->tok_bytes + o, bb, bn) == 0, "%s id %u decode bytes", name, id);
                e++;
            }
        }
        CHECK(e == t->add_n, "%s %u added lines for %llu entries", name, e, (unsigned long long)t->add_n);
        for (uint32_t i = 0; i < n_pp; i++) { CHECK(L.ctx.pp_ids[i] == pp_want[i], "%s template id %u", name, i); }
        CHECK(n_pp == 64u || L.ctx.pp_ids[n_pp] == 0u, "%s template length", name);
        printf("fixture %-12s ok: n_ids %u, %llu added, template %u/%#x\n", name, t->n_ids,
               (unsigned long long)t->add_n, t->tmpl, t->tmpl_params);
    }
    unload(&L);
    fclose(f);
    free(data);
}

/* the normalizer: NFC / NFKC parse and set cfg.nfc's form (src/core/norm.h runs it), inside Sequences too (config.c
 * flattens them as hf applies them; NFC then NFKC is NFKC); NFD / NFKD are refused by name. Variants of gpt2style.json,
 * the normalizer changed. */
static void normalizer_flag(uint8_t *arena, uint64_t arena_len)
{
    static const struct { const char *norm; int64_t r; uint8_t nfc; const char *what; } V[] = {
        { "null", 0, 0u, NULL },
        { "{\"type\": \"NFC\"}", 0, TOKS_NS_NFC, NULL },
        { "{\"type\": \"NFD\"}", TOKS_E_UNSUPPORTED, 0u, "normalizer NFD" },
        { "{\"type\": \"NFKC\"}", 0, TOKS_NS_NFKC, NULL },
        { "{\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"NFC\"}]}", 0, TOKS_NS_NFC, NULL },
        { "{\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"NFC\"}, {\"type\": \"NFKC\"}]}", 0, TOKS_NS_NFKC, NULL },
        { "{\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"Sequence\", \"normalizers\": [{\"type\": \"NFKD\"}]}]}",
          TOKS_E_UNSUPPORTED, 0u, "normalizer NFKD" },
    };
    static const char KEY[] = "\"normalizer\": null";
    uint64_t len = 0;
    uint8_t *base = slurp("tests/data/compile/gpt2style.json", &len);
    CHECK(base != NULL, "gpt2style.json missing (run from the repository root)");
    if (base == NULL) { return; }
    uint64_t at = 0;
    while (at + sizeof(KEY) - 1 <= len && memcmp(base + at, KEY, sizeof(KEY) - 1) != 0) { at++; }
    CHECK(at + sizeof(KEY) - 1 <= len, "gpt2style.json has no null normalizer");
    char *buf = malloc(len + 256);
    for (uint32_t i = 0; i < sizeof(V) / sizeof(V[0]) && at + sizeof(KEY) - 1 <= len; i++) {
        int n = snprintf(buf, len + 256, "%.*s\"normalizer\": %s%.*s", (int)at, (const char *)base, V[i].norm,
                         (int)(len - at - (sizeof(KEY) - 1)), (const char *)base + at + sizeof(KEY) - 1);
        loaded L;
        load(&L, (const uint8_t *)buf, (uint64_t)n, arena, arena_len);
        if (V[i].r == 0) {
            CHECK(L.r == 0 && L.cfg.nfc == V[i].nfc, "normalizer %s: got %lld nfc %u", V[i].norm, (long long)L.r,
                  L.cfg.nfc);
        } else {
            CHECK(L.r == V[i].r && L.stage == 1 && L.err.what != NULL && strstr(L.err.what, V[i].what) != NULL,
                  "normalizer %s: want refused naming '%s', got %lld '%s'", V[i].norm, V[i].what, (long long)L.r,
                  L.err.what ? L.err.what : "");
        }
        unload(&L);
    }
    printf("normalizer: NFC / NFKC accepted (cfg.nfc, also inside Sequences), NFD / NFKD refused by name\n");
    free(buf);
    free(base);
}

static void refusals(uint8_t *arena, uint64_t arena_len)
{
    uint64_t len = 0;
    uint8_t *all = slurp("tests/data/compile/refuse.txt", &len);
    CHECK(all != NULL, "refuse.txt missing");
    if (all == NULL) { return; }
    if (!td_text("tests/data/compile/refuse.txt", all, len)) { fails++; free(all); return; }
    uint32_t n = 0;
    char *p = (char *)all;
    td_at at = { "tests/data/compile/refuse.txt", 1 };
    while (*p != 0) {                                   /* bound: the file, two lines a pass */
        char *nl = strchr(p, '\n');
        if (nl == NULL) { break; }
        *nl = 0;
        char *json = nl + 1;
        char *nl2 = strchr(json, '\n');
        if (nl2 == NULL) { fails += !td_bad(&at, nl, "a json line after the '<code> <name>' line"); break; }
        *nl2 = 0;
        char *sp = strchr(p, ' ');
        if (sp == NULL || (p[0] != '-' && (p[0] < '0' || p[0] > '9'))) {
            fails += !td_bad(&at, sp != NULL ? p : nl, "'<code> <name>' expected");
            break;
        }
        long code = strtol(p, NULL, 10);
        const char *what = sp + 1;
        at.line += 2;
        loaded L;
        load(&L, (const uint8_t *)json, (uint64_t)(nl2 - json), arena, arena_len);
        CHECK(L.r == code && L.stage == 1 && L.err.what != NULL && strstr(L.err.what, what) != NULL,
              "refuse '%s': got %lld '%s'", what, (long long)L.r, L.err.what ? L.err.what : "");
        unload(&L);
        n++;
        p = nl2 + 1;
    }
    printf("refusals: %u files, each refused by name\n", n);
    free(all);
}

/* ---- the pinned real files --------------------------------------------------------------------------- */

static void real(const char *dir, const char *name)
{
    char path[1024];
    uint64_t len = 0;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    uint8_t *data = slurp(path, &len);
    if (data == NULL) { printf("real %-7s skipped (%s absent)\n", name, path); return; }
    uint64_t arena_len = len * 8u + (16u << 20);
    uint8_t *arena = malloc(arena_len);
    loaded L;
    load(&L, data, len, arena, arena_len);
    const toks_tables *t = &L.ctx.t;
    {
        CHECK(L.r == 0, "%s refused: %lld %s", name, (long long)L.r, L.err.what ? L.err.what : "");
        if (L.r == 0) {
            check_ctx(name, &L, 2000);
            const uint8_t *bm = (const uint8_t *)(const void *)L.ctx.special_ids;
            uint32_t n_special = 0;
            for (uint32_t id = 0; id < t->n_ids; id++) { n_special += (bm[id >> 3] >> (id & 7u)) & 1u; }
            int qwen = strcmp(name, "qwen3") == 0 || strcmp(name, "qwen38") == 0;
            CHECK(L.cfg.nfc == (qwen ? TOKS_NS_NFC : 0u), "%s: nfc flag %u", name, L.cfg.nfc);
            if (strcmp(name, "gpt2") == 0) {
                CHECK(L.cfg.n_vocab == 50257u && t->n_ids == 50257u && t->tmpl == TOKS_TMPL_CL100K &&
                      t->tmpl_params == (TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN) && t->flags == (TOKS_TF_CJK_L | TOKS_TF_CJK_B),
                      "gpt2 shape");
                CHECK(t->add_n == 1u && t->add_phases == 2u && t->add_entries[0].id == 50256u &&
                      t->add_entries[0].phase == 1u && n_special == 1u && (bm[50256 >> 3] >> (50256 & 7)) & 1u,
                      "gpt2 <|endoftext|>: one special, normalized (phase 1), id 50256");
                CHECK(L.cfg.n_pp_single == 0u && L.ctx.pp_ids[0] == 0u, "gpt2 has no template");
            } else if (strcmp(name, "dsv3") == 0) {
                /* deepseek-ai/DeepSeek-V3: the three-Split chain, ids 0..2 outside the alphabet and added specials,
                 * 818 added (804 special; the 14 tool tokens normalized: phase 1), a ByteLevel post-processor */
                CHECK(L.cfg.n_vocab == 128000u && t->n_ids == 128815u && t->tmpl == TOKS_TMPL_DSV3 && t->tmpl_params == 0u &&
                      t->flags == TOKS_TF_CJK_D && L.cfg.n_vocab_raw == 3u && toks_alpha_bytes(L.cfg.vocab[0], L.cfg.vocab_len[0], NULL) < 0 &&
                      toks_alpha_bytes(L.cfg.vocab[3], L.cfg.vocab_len[3], NULL) >= 0, "dsv3 shape");
                CHECK(t->add_n == 818u && t->add_phases == 3u && n_special == 804u && L.cfg.n_pp_single == 0u,
                      "dsv3: 818 added tokens, 804 special, both phases, no template");
                CHECK(t->tok_off[1] - t->tok_off[0] == 29u && memcmp(t->tok_bytes + t->tok_off[0],
                      "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>", 29) == 0,
                      "dsv3: id 0 decodes to its own utf-8");
            } else if (qwen) {
                /* qwen3: Qwen3-0.6B (the qwen 2 pattern), qwen38: Qwen3.8-27B (the qwen 3.5 pattern, marks folded);
                 * both NFC, ignore_merges false, every added token non-normalized (phase 0), no template */
                int q38 = strcmp(name, "qwen38") == 0;
                uint32_t nv = q38 ? 248044u : 151643u, na = q38 ? 33u : 26u, ns = q38 ? 21u : 14u;
                CHECK(L.cfg.n_vocab == nv && t->n_ids == nv + na && t->tmpl == TOKS_TMPL_CL100K &&
                      t->tmpl_params == (TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 |
                                         TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL) &&
                      t->flags == (TOKS_TF_CJK_L | TOKS_TF_CJK_B) && marks_folded(&L.cfg) == (q38 ? 1u : 0u), "%s shape",
                      name);
                CHECK(t->add_n == na && t->add_phases == 1u && n_special == ns, "%s: %u added in phase 0, %u special",
                      name, na, ns);
                CHECK(L.cfg.n_pp_single == 0u && L.ctx.pp_ids[0] == 0u, "%s has no template", name);
            } else {
                CHECK(L.cfg.n_vocab == 128000u && t->n_ids == 128256u && t->tmpl == TOKS_TMPL_CL100K &&
                      t->tmpl_params == (TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1_3 |
                                         TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL) &&
                      t->flags == (TOKS_TF_IGNORE_MERGES | TOKS_TF_CJK_L | TOKS_TF_CJK_B), "llama3 shape");
                CHECK(t->add_n == 256u && t->add_phases == 1u && n_special == 256u, "llama3: 256 specials in phase 0");
                for (uint32_t e = 0; e < 256u && e < t->add_n; e++) {
                    CHECK(t->add_entries[e].id == 128000u + e && t->add_entries[e].phase == 0u &&
                          (t->add_entries[e].flags & TOKS_AF_SPECIAL) != 0, "llama3 entry %u", e);
                }
                CHECK(L.ctx.pp_ids[0] == 128000u && L.ctx.pp_ids[1] == 0u && L.cfg.n_pp_single == 2u &&
                      L.cfg.pp_single[0].kind == TOKS_PPS_TOK && L.cfg.pp_single[1].kind == TOKS_PPS_SEQ,
                      "llama3 template: <|begin_of_text|> $A");
            }
            printf("real %-7s ok: n_ids %u, %llu added (phases %llu), parse %.0f ms, compile %.0f ms, "
                   "parse arena %.1f MiB (file %.1f MiB), tables %.2f MiB\n",
                   name, t->n_ids, (unsigned long long)t->add_n, (unsigned long long)t->add_phases, L.ms_parse,
                   L.ms_compile, (double)L.ar.pos / 1048576.0, (double)len / 1048576.0,
                   (double)L.ctx.mem_tables_len / 1048576.0);
        }
    }
    unload(&L);
    free(arena);
    free(data);
}

/* ---- hostile smoke ------------------------------------------------------------------------------------ */

static void hostile(uint8_t *arena, uint64_t arena_len)
{
    uint64_t len = 0;
    uint8_t *src = slurp("tests/data/compile/llama3style.json", &len);
    if (src == NULL) { CHECK(0, "hostile: fixture missing"); return; }
    uint8_t *m = malloc(len + 1u);
    uint32_t accepted = 0, refused = 0;
    static const uint8_t nasty[] = { '"', '\\', '{', '}', '[', ']', ',', ':', '0', '9', '-', 'e', '.', 't', 'n',
                                     ' ', 0x00, 0x80, 0xC0, 0xFF, 'u', 'x' };
    for (uint64_t cut = 0; cut + 2u < len; cut++) {           /* every truncation but the trailing newline */
        loaded L;
        load(&L, src, cut, arena, arena_len);
        CHECK(L.r != 0, "truncated at %llu accepted", (unsigned long long)cut);
        unload(&L);
    }
    for (uint32_t i = 0; i < 4000u; i++) {                    /* random byte mutations: refuse or stay sane */
        memcpy(m, src, len);
        uint32_t k = 1u + (uint32_t)(rnd() % 3u);
        for (uint32_t j = 0; j < k; j++) {
            m[rnd() % len] = (rnd() & 1u) ? nasty[rnd() % sizeof(nasty)] : (uint8_t)rnd();
        }
        loaded L;
        load(&L, m, len, arena, arena_len);
        if (L.r == 0) { accepted++; check_ctx("mutant", &L, 20); } else { refused++; }
        CHECK(L.r == 0 || L.r == TOKS_E_FORMAT || L.r == TOKS_E_UNSUPPORTED || L.r == TOKS_E_LIMIT,
              "mutant: unexpected code %lld", (long long)L.r);
        unload(&L);
    }
    printf("hostile: %llu truncations refused; 4000 mutants: %u accepted (invariants hold), %u refused\n",
           (unsigned long long)(len - 2u), accepted, refused);
    free(m);
    free(src);
}

/* toks_compile_cls_flags on built and hand-broken class tables (docs/kernels.md §2's preconditions) */
static void cls_flags(void)
{
    static uint8_t buf[2][128 + 2 * 0x1100 + 0x1100 * 256], m_ascii[128], m_st2[0x1100 * 256 + 256];
    static uint16_t m_st1[0x1100];
    for (uint32_t k = 0; k < 2u; k++) {                         /* plain and marks folded */
        toks_class_tables ct;
        CHECK(toks_classes_build(k ? TOKS_CLASSES_MARKS_ARE_LETTERS : 0u, buf[k], sizeof buf[k], &ct) > 0, "classes");
        toks_tables t;
        memset(&t, 0, sizeof t);
        t.cls_ascii = ct.ascii; t.cls_stage1 = ct.stage1; t.cls_stage2 = ct.stage2; t.cls_nblocks = ct.n_blocks;
        const int64_t LB = TOKS_TF_CJK_L | TOKS_TF_CJK_B;
        CHECK(toks_compile_cls_flags(&t) == LB, "cls_flags: built tables (marks %u)", k);
        /* one change per case on copies: {ascii byte or code point, new class, expected} */
        static const struct { uint32_t at; uint8_t cls; int64_t want; } C[] = {
            { 's', TOKS_C_P, -1 }, { ' ', TOKS_C_P, -1 }, { '\n', TOKS_C_WS, -1 }, { '7', TOKS_C_L, -1 },
            { 'x', TOKS_C_L | TOKS_C_LOWER | TOKS_C_FOLD_S, -1 }, { '!', TOKS_C_P | TOKS_C_MARK, -1 },
            { 'a', TOKS_C_L | TOKS_C_UPPER, -1 }, { 'A', TOKS_C_L, -1 }, { 'z', TOKS_C_L | TOKS_C_UPPER | TOKS_C_LOWER, -1 },
            { 0x4E00u + 0x80u, TOKS_C_P, 0 }, { 0x9FFFu + 0x80u, TOKS_C_N, 0 }, { 0xA3FFu + 0x80u, TOKS_C_P, 0 },
            { 0xD6FFu + 0x80u, TOKS_C_L | TOKS_C_FOLD_S, 0 }, { 0xAC00u + 0x80u, TOKS_C_WS, 0 },
            { 0xA400u + 0x80u, TOKS_C_P, LB }, { 0xD700u + 0x80u, TOKS_C_P, LB }, { 0x4DFFu + 0x80u, TOKS_C_N, LB },
            { 0x6587u + 0x80u, TOKS_C_L | TOKS_C_UPPER, TOKS_TF_CJK_L },
            { 0x4E2Du + 0x80u, TOKS_C_L | TOKS_C_UPPER | TOKS_C_LOWER | TOKS_C_MARK, TOKS_TF_CJK_L },
            { 0xAC01u + 0x80u, TOKS_C_L | TOKS_C_UPPER | TOKS_C_LOWER | TOKS_C_HAN, TOKS_TF_CJK_L },
        };
        for (uint32_t i = 0; i < sizeof C / sizeof C[0]; i++) {    /* code points are stored + 0x80 */
            toks_tables m = t;
            memcpy(m_ascii, ct.ascii, 128);
            memcpy(m_st1, ct.stage1, sizeof m_st1);
            memcpy(m_st2, ct.stage2, (size_t)ct.n_blocks * 256u);
            if (C[i].at < 0x80u) {
                m_ascii[C[i].at] = C[i].cls;
            } else {
                uint32_t cp = C[i].at - 0x80u, nb = ct.n_blocks;       /* the code point's block, copied */
                memcpy(m_st2 + (size_t)nb * 256u, m_st2 + (size_t)m_st1[cp >> 8] * 256u, 256);
                m_st1[cp >> 8] = (uint16_t)nb;
                m_st2[(size_t)nb * 256u + (cp & 0xFFu)] = C[i].cls;
            }
            m.cls_ascii = m_ascii; m.cls_stage1 = m_st1; m.cls_stage2 = m_st2;
            CHECK(toks_compile_cls_flags(&m) == C[i].want, "cls_flags case %u (marks %u): got %lld want %lld", i, k,
                  (long long)toks_compile_cls_flags(&m), (long long)C[i].want);
        }
    }
}

int main(void)
{
    alphabet_init();
    cls_flags();
    for (uint32_t b = 0; b < 256u; b++) {                     /* the core's alphabet agrees with ours */
        CHECK(toks_char_byte(B2U[b]) == (int32_t)b, "toks_char_byte(U+%04X) != %u", B2U[b], b);
    }
    for (uint32_t cp = 0; cp < 0x400u; cp++) {
        uint32_t b = 0;
        while (b < 256u && B2U[b] != cp) { b++; }
        CHECK(toks_char_byte(cp) == (b < 256u ? (int32_t)b : -1), "toks_char_byte(U+%04X)", cp);
    }
    uint64_t arena_len = 64u << 20;
    uint8_t *arena = malloc(arena_len);
    static const char *const fixtures[] = { "gpt2style", "llama3style", "qwen35style", "nosplit", "nonalpha", "dsv3style" };
    for (uint32_t i = 0; i < 6u; i++) { fixture(fixtures[i], arena, arena_len); }
    refusals(arena, arena_len);
    normalizer_flag(arena, arena_len);
    hostile(arena, arena_len);
    arena_bound();
    free(arena);

    const char *dir = getenv("TOKS_TOKENIZER_CACHE");
    char home[1024];
    if (dir == NULL && getenv("HOME") != NULL) {
        snprintf(home, sizeof(home), "%s/.cache/toks/tokenizers", getenv("HOME"));
        dir = home;
    }
    if (dir != NULL) {
        real(dir, "gpt2");
        real(dir, "llama3");
        real(dir, "qwen3");
        real(dir, "dsv3");
        real(dir, "qwen38");
    }
    printf("test_compile: %llu checks, %d failures\n", (unsigned long long)checks, fails);
    return fails != 0;
}
