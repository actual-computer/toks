/*
 * test_tiktoken.c: the tiktoken reader (src/core/tiktoken.c; docs/models/kimi.md section 3).
 *   - hand ranks files: vocabulary strings against an independent byte-level alphabet, holes, the exact merge
 *     list (brute force: every split whose halves are both tokens), the flags; compiled and bpe-built:
 *     TOKS_TF_IDS_AS_RANK with every pair's priority == its merged id and no rank2id (with cfg->ids_as_rank
 *     cleared the builder falls back to raw ranks + rank2id: the check that fails without the flag), and K6
 *     == a brute-force tiktoken byte_pair_merge on every string of 1..6 letters over the hand alphabet;
 *   - hostile ranks files: each one refused with its code and reason; every truncation and 3,000 seeded
 *     mutations of a valid file never crash, and what is still accepted is self-consistent;
 *   - the kimi wrapper and config: every WRAPPER line missing, the pattern out of order, re-indented, and
 *     each config rule (keys, class, auto_map, ids, strips, attributes, overlapping names, flags);
 *   - the real files ($TOKS_TOKENIZER_CACHE, default ~/.cache/toks/tokenizers: kimik3.tiktoken,
 *     kimik3_tokenizer_config.json, kimik3_tokenization_kimi.py; SKIP when absent): counts and sha-256s
 *     against tests/data/kimi/expect.txt (tiktoken 0.14.0's own reader), toks_tiktoken_parse's refusal naming
 *     the template, the tables, and K6 on tests/data/kimi/bpe_cases.txt == tiktoken's ids.
 * Run from the repository root (make test does).
 */
#include "core.h"
#include "bpe.h"
#include "compile.h"
#include "kernels.h"
#include "tiktoken.h"
#include "classes.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* ---- helpers ------------------------------------------------------------------------------------------ */

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *d = malloc(n > 0 ? (size_t)n : 1u);
    if (d == NULL || (n > 0 && fread(d, 1, (size_t)n, f) != (size_t)n)) { free(d); fclose(f); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return d;
}

static void b64(const uint8_t *p, size_t n, char *out)      /* out room >= 4 * ceil(n / 3) + 1 */
{
    static const char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out[o++] = A[v >> 18 & 63];
        out[o++] = A[v >> 12 & 63];
        out[o++] = i + 1 < n ? A[v >> 6 & 63] : '=';
        out[o++] = i + 2 < n ? A[v & 63] : '=';
    }
    out[o] = 0;
}

static uint32_t B2U[256];

static void alphabet_init(void)
{
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256u; b++) {
        int self = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || b >= 0xAEu;
        B2U[b] = self ? b : 0x100u + n++;
    }
}

static size_t alpha_image(const uint8_t *p, size_t n, uint8_t *out)   /* the byte-level string of p */
{
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t c = B2U[p[i]];
        if (c < 0x80u) { out[o++] = (uint8_t)c; }
        else { out[o++] = (uint8_t)(0xC0u | c >> 6); out[o++] = (uint8_t)(0x80u | (c & 0x3Fu)); }
    }
    return o;
}

typedef struct buf { char *p; size_t n, cap; } buf;

static void bput(buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void bputs(buf *b, const char *s) { bput(b, s, strlen(s)); }

/* ---- the hand vocabulary ------------------------------------------------------------------------------ */

typedef struct htok { const char *s; uint32_t rank; } htok;

/* one-byte tokens at rank (7 b + 3) mod 256 (a permutation: byte2id must come from the file), then these;
 * "xyz" has no split (only the whole-piece lookup reaches it); abc's rank is below ab's (inverted). */
static const htok HAND[] = {
    { "bc", 256 }, { "abc", 257 }, { "ab", 258 }, { "aa", 259 }, { "aaa", 260 }, { "aaaa", 261 }, { "xyz", 262 },
    { "ca", 263 }, { "bca", 264 }, { "cab", 265 }, { "abca", 266 }, { "zz", 300 },
};
#define N_HAND (sizeof HAND / sizeof HAND[0])

static uint32_t byte_rank(uint32_t b) { return (7u * b + 3u) & 255u; }

/* the ranks file of the hand vocabulary; dense = 1 leaves out the rank-300 token (no gap). crlf: CR LF
 * line ends; blank: an empty line in the middle. */
static char *hand_file(int dense, int crlf, int blank)
{
    buf b = { 0 };
    char line[256], e[64];
    for (uint32_t x = 0; x < 256u; x++) {
        uint8_t one = (uint8_t)x;
        b64(&one, 1, e);
        snprintf(line, sizeof line, "%s %u%s", e, byte_rank(x), crlf ? "\r\n" : "\n");
        bputs(&b, line);
        if (blank && x == 100u) { bputs(&b, crlf ? "\r\n" : "\n"); }
    }
    for (size_t i = 0; i < N_HAND; i++) {
        if (dense && HAND[i].rank == 300u) { continue; }
        b64((const uint8_t *)HAND[i].s, strlen(HAND[i].s), e);
        snprintf(line, sizeof line, "%s %u%s", e, HAND[i].rank, crlf ? "\r\n" : "\n");
        bputs(&b, line);
    }
    return b.p;
}

/* the token of rank r in the hand vocabulary (NULL: a hole) */
static const uint8_t *hand_tok(uint32_t r, size_t *n, uint8_t *one)
{
    if (r < 256u) {
        for (uint32_t x = 0; x < 256u; x++) {
            if (byte_rank(x) == r) { *one = (uint8_t)x; *n = 1; return one; }
        }
    }
    for (size_t i = 0; i < N_HAND; i++) {
        if (HAND[i].rank == r) { *n = strlen(HAND[i].s); return (const uint8_t *)HAND[i].s; }
    }
    return NULL;
}

static int64_t hand_rank_of(const uint8_t *p, size_t n)
{
    if (n == 1) { return byte_rank(p[0]); }
    for (size_t i = 0; i < N_HAND; i++) {
        if (strlen(HAND[i].s) == n && memcmp(HAND[i].s, p, n) == 0) { return HAND[i].rank; }
    }
    return -1;
}

/* tiktoken's byte_pair_encode, written out: the whole piece if it is a token, else merge the adjacent pair
 * whose concatenation has the lowest rank, the leftmost on ties. returns the id count. */
static uint32_t brute_bpe(const uint8_t *p, size_t n, uint32_t *ids, int whole)   /* whole 0: the merges alone */
{
    int64_t w = whole ? hand_rank_of(p, n) : -1;
    if (w >= 0) { ids[0] = (uint32_t)w; return 1; }
    size_t bd[64];
    size_t nb = n + 1;
    for (size_t i = 0; i <= n; i++) { bd[i] = i; }
    while (nb > 2) {
        int64_t best = -1;
        size_t at = 0;
        for (size_t k = 0; k + 2 < nb; k++) {
            int64_t r = hand_rank_of(p + bd[k], bd[k + 2] - bd[k]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; at = k; }
        }
        if (best < 0) { break; }
        memmove(&bd[at + 1], &bd[at + 2], (nb - at - 2) * sizeof bd[0]);
        nb--;
    }
    for (size_t k = 0; k + 1 < nb; k++) { ids[k] = (uint32_t)hand_rank_of(p + bd[k], bd[k + 1] - bd[k]); }
    return (uint32_t)(nb - 1);
}

/* ---- building -------------------------------------------------------------------------------------- */

typedef struct built {
    toks_ctx ctx;
    uint8_t *bpe_mem;
    int64_t r;
} built;

static void build_tables(built *b, const toks_config *cfg, toks_arena *par)
{
    memset(b, 0, sizeof *b);
    b->r = toks_compile(cfg, &b->ctx, par);
    if (b->r != 0) { return; }
    uint64_t n = toks_bpe_tables_bytes(cfg);
    b->bpe_mem = malloc((size_t)n + 64u);
    toks_arena bar = { b->bpe_mem, n + 64u, 0 };
    b->r = toks_bpe_build(&b->ctx.t, &bar, cfg);
}

static void free_tables(built *b)
{
    if (b->ctx.mem_tables != NULL) { toks_plat_arena_free(b->ctx.mem_tables, b->ctx.mem_tables_len); }
    free(b->bpe_mem);
    memset(b, 0, sizeof *b);
}

static uint64_t k6(const toks_tables *t, const uint8_t *p, uint64_t n, uint32_t *out)
{
    static uint8_t work[TOKS_BPE_WORK_BYTES(4096) + 64];
    uint8_t *w = (uint8_t *)(((uintptr_t)work + 63u) & ~(uintptr_t)63u);
    toks_k6_args a = { p, n, out, w, TOKS_BPE_WORK_BYTES(4096), 0, 0, 0 };
    return toks_k6_bpe_c(t, &a);
}

/* a piece as K5 runs it: under TOKS_TF_PROBE_LONG a piece of 2..15 bytes that K5's static words answer never
 * reaches K6, which leaves the whole-piece rule to them for pieces that short (layout.h); every other piece is K6's */
static uint64_t k5k6(const toks_tables *t, const uint8_t *p, uint64_t n, uint32_t *out)
{
    if ((t->flags & TOKS_TF_PROBE_LONG) != 0u && n >= 2u && n <= TOKS_KEY_MAXLEN) {
        bpe_key k = bpe_key_at(p, n, 0u, n);
        const uint8_t *v = bpe_w3_probe(t->words, t->words_mask, bpe_key_hash(k), k);
        if (v != NULL) {
            uint32_t ids[2];
            uint64_t c = bpe_w3_val(v, ids);
            out[0] = ids[0];
            out[1] = ids[1];
            return c;
        }
    }
    return k6(t, p, n, out);
}

/* every merge's halves concatenate to its merged token; every split of every token with both halves tokens
 * is a merge (counted); vocabulary strings are alphabet images. */
static void consistent(const toks_config *cfg, const char *what)
{
    uint8_t x[65536 * 2], y[65536];
    for (uint32_t i = 0; i < cfg->n_merges; i++) {
        uint32_t l = cfg->m_left_id[i], r = cfg->m_right_id[i], o = cfg->m_out_id[i];
        CHECK(l < cfg->n_vocab && r < cfg->n_vocab && o < cfg->n_vocab, "%s: merge %u ids", what, i);
        if (l >= cfg->n_vocab || r >= cfg->n_vocab || o >= cfg->n_vocab) { return; }
        int64_t a = toks_alpha_bytes(cfg->vocab[l], cfg->vocab_len[l], x);
        int64_t b = toks_alpha_bytes(cfg->vocab[r], cfg->vocab_len[r], x + (a > 0 ? a : 0));
        int64_t c = toks_alpha_bytes(cfg->vocab[o], cfg->vocab_len[o], y);
        CHECK(a > 0 && b > 0 && c == a + b && memcmp(x, y, (size_t)c) == 0, "%s: merge %u is not left + right", what, i);
    }
}

/* ---- hand ranks files -------------------------------------------------------------------------------- */

/* toks_tiktoken_parse's output (src/core/tiktoken.h): the kimi pattern, the wrapper's cuts, and the specials
 * as config.h's two matchers -- phase 0 the trie's tokens, phase 1 all 256 names, each in id order. */
static void check_phases(const toks_config *cfg, const toks_tiktoken_info *info, const char *what)
{
    CHECK(cfg->pattern == &TOKS_PATTERN_KIMI && cfg->pattern->tmpl == TOKS_TMPL_O200K && cfg->pattern->params == 0x182u &&
          cfg->pattern->class_flags == TOKS_CLASSES_HAN, "%s: the kimi pattern", what);
    CHECK(cfg->cut_chunk == 400000u && cfg->cut_run == 25000u, "%s: cuts %u / %u", what, cfg->cut_chunk, cfg->cut_run);
    CHECK(cfg->n_added == info->n_trie + TOKS_TIKTOKEN_RESERVED, "%s: %u entries for %u trie tokens", what, cfg->n_added,
          info->n_trie);
    uint32_t k = 0;
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED && k < cfg->n_added; i++) {
        if (((info->trie[i >> 3] >> (i & 7u)) & 1u) == 0u) { continue; }
        CHECK(cfg->added[k].normalized == 0u && cfg->added[k].id == info->first_special + i, "%s: trie entry %u", what, k);
        k++;
    }
    CHECK(k == info->n_trie, "%s: %u trie entries", what, k);
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED && k + i < cfg->n_added; i++) {
        const toks_cfg_added *a = &cfg->added[k + i];
        CHECK(a->normalized == 1u && a->id == info->first_special + i, "%s: name entry %u", what, i);
    }
}

static void test_hand(void)
{
    for (int variant = 0; variant < 3; variant++) {          /* plain, crlf, blank line */
        char *f = hand_file(0, variant == 1, variant == 2);
        uint64_t len = strlen(f);
        uint64_t cap = toks_tiktoken_arena_bound(len, 0, 0);
        uint8_t *mem = malloc((size_t)cap);
        toks_arena ar = { mem, cap, 0 };
        toks_config cfg;
        toks_err err = { 0, NULL };
        int64_t r = toks_tiktoken_ranks((const uint8_t *)f, len, &ar, &cfg, &err);
        CHECK(r == 0, "hand %d: %" PRId64 " %s", variant, r, err.what ? err.what : "");
        if (r != 0) { free(mem); free(f); continue; }
        CHECK(cfg.n_vocab == 301u && cfg.n_ids == 301u, "hand: n_vocab %u", cfg.n_vocab);
        CHECK(cfg.ignore_merges == 1u && cfg.ids_as_rank == 1u && cfg.dec_byte_level == 1u && cfg.pattern == NULL,
              "hand: flags");
        /* vocabulary strings: the alphabet image of each rank's token, holes empty */
        for (uint32_t id = 0; id < cfg.n_vocab; id++) {
            uint8_t one, img[64];
            size_t n = 0;
            const uint8_t *t = hand_tok(id, &n, &one);
            size_t m = t ? alpha_image(t, n, img) : 0;
            CHECK(cfg.vocab_len[id] == m && (m == 0 || memcmp(cfg.vocab[id], img, m) == 0), "hand: vocab string of id %u", id);
        }
        /* merges: exactly the brute-force list, by merged id then split position */
        uint32_t want = 0, pos = 0;
        int order_ok = 1;
        for (uint32_t id = 0; id < cfg.n_vocab; id++) {
            uint8_t one;
            size_t n = 0;
            const uint8_t *t = hand_tok(id, &n, &one);
            for (size_t k = 1; t != NULL && k < n; k++) {
                int64_t a = hand_rank_of(t, k), b = hand_rank_of(t + k, n - k);
                if (a < 0 || b < 0) { continue; }
                want++;
                if (pos < cfg.n_merges && !(cfg.m_left_id[pos] == (uint32_t)a && cfg.m_right_id[pos] == (uint32_t)b &&
                                           cfg.m_out_id[pos] == id)) { order_ok = 0; }
                pos++;
            }
        }
        CHECK(cfg.n_merges == want && order_ok, "hand: %u merges (want %u), order %d", cfg.n_merges, want, order_ok);
        consistent(&cfg, "hand");

        /* tables: ids-as-rank, priority = merged id, no rank2id; then K6 against brute force */
        built B;
        build_tables(&B, &cfg, &ar);
        CHECK(B.r == 0, "hand: compile / bpe build %" PRId64, B.r);
        if (B.r == 0 && variant == 0) {
            const toks_tables *t = &B.ctx.t;
            CHECK((t->flags & TOKS_TF_IDS_AS_RANK) != 0u && t->rank2id == NULL, "hand: not ids-as-rank");
            CHECK((t->flags & TOKS_TF_IGNORE_MERGES) != 0u, "hand: ignore_merges");
            for (uint32_t i = 0; i < cfg.n_merges; i++) {
                const bpe_mt mt = bpe_mt_of(t);
                uint32_t prio = bpe_mt_find(&mt, bpe_pair_key(cfg.m_left_id[i], cfg.m_right_id[i]));
                int found = prio != TOKS_PRIO_NONE;
                CHECK(found && prio == cfg.m_out_id[i], "hand: merge %u priority %u != merged id %u", i, prio, cfg.m_out_id[i]);
            }
            static const char AL[] = "abcxyz";
            uint8_t s[8];
            uint32_t got[16], exp[16];
            uint64_t n_cases = 0, bad = 0;
            for (uint32_t L = 1; L <= 6; L++) {
                uint32_t total = 1;
                for (uint32_t k = 0; k < L; k++) { total *= 6u; }
                for (uint32_t c = 0; c < total; c++) {
                    uint32_t v = c;
                    for (uint32_t k = 0; k < L; k++) { s[k] = (uint8_t)AL[v % 6u]; v /= 6u; }
                    uint64_t ng = k5k6(t, s, L, got);
                    uint32_t ne = brute_bpe(s, L, exp, 1);
                    n_cases++;
                    if (ng != ne || memcmp(got, exp, ne * 4u) != 0) {
                        if (bad++ < 5) { CHECK(0, "hand: K5 / K6 of '%.*s' differ from tiktoken's rule", (int)L, (const char *)s); }
                    }
                    ng = k6(t, s, L, got);             /* K6's own answer: the merges alone this short (PROBE_LONG) */
                    ne = brute_bpe(s, L, exp, (t->flags & TOKS_TF_PROBE_LONG) == 0u);
                    if (ng != ne || memcmp(got, exp, ne * 4u) != 0) {
                        if (bad++ < 5) { CHECK(0, "hand: K6 of '%.*s' is not its own answer", (int)L, (const char *)s); }
                    }
                }
            }
            CHECK((t->flags & TOKS_TF_PROBE_LONG) != 0u, "hand: K5's words do not hold every token that needs the rule");
            CHECK(bad == 0, "hand: K5 / K6 != brute force on %" PRIu64 " of %" PRIu64 " strings", bad, n_cases);
            printf("  hand: %u tokens, %u merges, K5 / K6 == tiktoken's rule on %" PRIu64 " strings\n", cfg.n_vocab, cfg.n_merges, n_cases);
        }
        free_tables(&B);
        /* the flag is what makes the builder take merged ids: without it, the repeated merged ids send it to
         * raw ranks + rank2id (the priorities then are list positions, not tiktoken's ranks) */
        if (variant == 0) {
            cfg.ids_as_rank = 0u;
            build_tables(&B, &cfg, &ar);
            CHECK(B.r == 0 && (B.ctx.t.flags & TOKS_TF_IDS_AS_RANK) == 0u && B.ctx.t.rank2id != NULL,
                  "hand: without cfg.ids_as_rank the builder should fall back to rank2id");
            free_tables(&B);
        }
        free(mem);
        free(f);
    }
}

/* ---- hostile ranks files -------------------------------------------------------------------------- */

static int64_t ranks_of(const char *f, uint64_t len, const char **why, toks_config *cfg_out)
{
    uint64_t cap = toks_tiktoken_arena_bound(len, 0, 0);
    uint8_t *mem = malloc((size_t)cap);
    toks_arena ar = { mem, cap, 0 };
    toks_config cfg;
    toks_err err = { 0, NULL };
    int64_t r = toks_tiktoken_ranks((const uint8_t *)f, len, &ar, &cfg, &err);
    *why = err.what;
    if (r == 0) {
        consistent(&cfg, "accepted hostile file");
        CHECK(ar.pos <= cap, "arena overrun");
    }
    if (cfg_out != NULL) { *cfg_out = cfg; }
    free(mem);
    return r;
}

static void test_hostile(void)
{
    char *good = hand_file(1, 0, 0);
    size_t gl = strlen(good);
    static const struct { const char *tail; int64_t code; const char *why; } BAD[] = {
        { "QQ== 5\n", TOKS_E_FORMAT, "rank appears twice" },          /* rank 5 is a byte's */
        { "YWI= 400\n", TOKS_E_FORMAT, "token appears twice" },       /* "ab" again */
        { "eHg=  400\n", TOKS_E_FORMAT, "one space" },
        { "eHg=\t400\n", TOKS_E_FORMAT, "without the space" },
        { "eHg 400\n", TOKS_E_FORMAT, "canonical base64" },
        { "eHh= 400\n", TOKS_E_FORMAT, "canonical base64" },          /* unused bits set (one '=') */
        { "eR== 400\n", TOKS_E_FORMAT, "canonical base64" },          /* unused bits set (two '=') */
        { "eH!= 400\n", TOKS_E_FORMAT, "canonical base64" },
        { "==== 400\n", TOKS_E_FORMAT, "canonical base64" },
        { " 400\n", TOKS_E_FORMAT, "canonical base64" },
        { "eHg= 0400\n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= -4\n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= +4\n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= 4x\n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= \n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= 2097151\n", TOKS_E_FORMAT, "canonical decimal" },     /* TOKS_MAX_IDS */
        { "eHg= 12345678\n", TOKS_E_FORMAT, "canonical decimal" },
        { "eHg= 400\r\r\n", TOKS_E_FORMAT, "LF line ends" },
        { "eHg= 4\r00\n", TOKS_E_FORMAT, "LF line ends" },
    };
    for (size_t i = 0; i < sizeof BAD / sizeof BAD[0]; i++) {
        buf b = { 0 };
        bput(&b, good, gl);
        bputs(&b, BAD[i].tail);
        const char *why = NULL;
        int64_t r = ranks_of(b.p, b.n, &why, NULL);
        CHECK(r == BAD[i].code && why != NULL && strstr(why, BAD[i].why) != NULL, "hostile %zu: %" PRId64 " \"%s\" (want %s)",
              i, r, why ? why : "", BAD[i].why);
        free(b.p);
    }
    const char *why = NULL;
    CHECK(ranks_of("", 0, &why, NULL) == TOKS_E_FORMAT && strstr(why, "empty") != NULL, "empty file");
    CHECK(ranks_of("\n\n\r\n", 4, &why, NULL) == TOKS_E_FORMAT && strstr(why, "no tokens") != NULL, "no tokens");
    /* a byte missing: drop the line of byte 'q' */
    {
        buf b = { 0 };
        char e[8], skip[32];
        uint8_t q = 'q';
        b64(&q, 1, e);
        snprintf(skip, sizeof skip, "%s %u\n", e, byte_rank('q'));
        char *at = strstr(good, skip);
        bput(&b, good, (size_t)(at - good));
        bputs(&b, at + strlen(skip));
        CHECK(ranks_of(b.p, b.n, &why, NULL) == TOKS_E_UNSUPPORTED && strstr(why, "one-byte") != NULL, "missing byte: %s", why ? why : "");
        free(b.p);
    }
    /* a token above TOKS_MAX_TOKEN_BYTES */
    {
        buf b = { 0 };
        bput(&b, good, gl);
        size_t n = TOKS_MAX_TOKEN_BYTES + 1u;
        uint8_t *big = malloc(n);
        char *e = malloc(4 * (n / 3 + 1) + 8);
        memset(big, 'a', n);
        b64(big, n, e);
        bputs(&b, e);
        bputs(&b, " 400\n");
        CHECK(ranks_of(b.p, b.n, &why, NULL) == TOKS_E_FORMAT && strstr(why, "TOKS_MAX_TOKEN_BYTES") != NULL, "long token");
        free(big);
        free(e);
        free(b.p);
    }
    /* no final LF, CR LF everywhere, blank lines: accepted */
    toks_config cfg;
    CHECK(ranks_of(good, gl - 1, &why, &cfg) == 0, "no final LF: %s", why ? why : "");
    char *crlf = hand_file(1, 1, 1);
    CHECK(ranks_of(crlf, strlen(crlf), &why, &cfg) == 0 && cfg.n_vocab == 267u, "crlf + blank lines: %s", why ? why : "");
    free(crlf);

    /* every truncation: refused or accepted, never a crash; accepted ones are self-consistent */
    uint64_t acc = 0;
    for (size_t n = 0; n <= gl; n++) {
        if (ranks_of(good, n, &why, NULL) == 0) { acc++; }
    }
    CHECK(acc >= 1, "truncations: the whole file must be accepted");
    /* seeded mutations: 1..4 random bytes replaced */
    uint64_t seed = 0x9E3779B97F4A7C15ull, macc = 0;
    char *m = malloc(gl + 1);
    for (int it = 0; it < 3000; it++) {
        memcpy(m, good, gl + 1);
        uint32_t k = 1u + (uint32_t)(seed >> 60) % 4u;
        for (uint32_t j = 0; j < k; j++) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            size_t at = (size_t)((seed >> 33) % gl);
            static const char POOL[] = "AZaz09+/= \n\r\t-x";
            m[at] = (seed & 1u) ? POOL[(seed >> 8) % (sizeof POOL - 1)] : (char)(seed >> 16);
        }
        if (ranks_of(m, gl, &why, NULL) == 0) { macc++; }
    }
    free(m);
    printf("  hostile: %zu named refusals, %zu truncations (%" PRIu64 " accepted), 3000 mutations (%" PRIu64 " accepted)\n",
           sizeof BAD / sizeof BAD[0] + 4, gl + 1, acc, macc);
    free(good);
}

/* ---- the kimi wrapper and config ------------------------------------------------------------------- */

static const char *WR[] = {
    "    num_reserved_special_tokens = 256",
    "    pat_str = \"|\".join([",
    "        r\"\"\"[\\p{Han}]+\"\"\",",
    "        r\"\"\"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?\"\"\",",
    "        r\"\"\"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?\"\"\",",
    "        r\"\"\"\\p{N}{1,3}\"\"\",",
    "        r\"\"\" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\"\"\",",
    "        r\"\"\"\\s*[\\r\\n]+\"\"\",",
    "        r\"\"\"\\s+(?!\\S)\"\"\",",
    "        r\"\"\"\\s+\"\"\",",
    "    ])",
    "        TIKTOKEN_MAX_ENCODE_CHARS = 400_000",
    "        MAX_NO_WHITESPACES_CHARS = 25_000",
    "        current_slice_is_space = s[0].isspace() if len(s) > 0 else False",
    "                if current_slice_len > max_consecutive_slice_len:",
    "                            allowed_special=\"all\",",
    "                            disallowed_special=(),",
};
#define N_WR (sizeof WR / sizeof WR[0])

/* the wrapper text without line `skip` (-1: none), with lines a and b swapped (a < 0: no swap), and every
 * line re-indented by `indent` extra spaces */
static char *wrapper_text(int skip, int a, int b, int indent)
{
    buf o = { 0 };
    bputs(&o, "import tiktoken\n\nclass TikTokenTokenizer(PreTrainedTokenizer):\n\n");
    for (int i = 0; i < (int)N_WR; i++) {
        if (i == skip) { continue; }
        int j = (i == a) ? b : (i == b) ? a : i;
        if (a < 0) { j = i; }
        for (int k = 0; k < indent; k++) { bputs(&o, " "); }
        bputs(&o, WR[j]);
        bputs(&o, "\n");
        if (i == 6) { bputs(&o, "\n"); }                      /* a blank line inside the pattern is fine */
    }
    return o.p;
}

static const char *CFG_OK =
    "{\"added_tokens_decoder\": {"
    "\"267\": {\"content\": \"[BOS]\", \"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},"
    "\"268\": {\"content\": \"[EOS]\", \"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},"
    "\"269\": {\"content\": \"<|open|>\", \"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": false},"
    "\"521\": {\"content\": \"[UNK]\", \"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},"
    "\"522\": {\"content\": \"[PAD]\", \"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true}},"
    "\"additional_special_tokens\": [\"[EOS]\"], \"bos_token\": \"[BOS]\", \"clean_up_tokenization_spaces\": false,"
    "\"eos_token\": {\"content\": \"[EOS]\", \"__type\": \"AddedToken\"}, \"extra_special_tokens\": {}, \"model_max_length\": 1000000,"
    "\"pad_token\": \"[PAD]\", \"tokenizer_class\": \"TikTokenTokenizer\", \"unk_token\": \"[UNK]\","
    "\"auto_map\": {\"AutoTokenizer\": [\"tokenization_kimi.TikTokenTokenizer\", null]}}";

/* CFG_OK with the first occurrence of `from` replaced by `to` */
static char *cfg_with(const char *from, const char *to)
{
    const char *at = strstr(CFG_OK, from);
    buf o = { 0 };
    if (at == NULL) { bputs(&o, "{}"); return o.p; }
    bput(&o, CFG_OK, (size_t)(at - CFG_OK));
    bputs(&o, to);
    bputs(&o, at + strlen(from));
    return o.p;
}

static int64_t kimi_of(const char *ranks, const char *conf, const char *wrap, const char **why, toks_config *cfg,
                       toks_tiktoken_info *info, uint8_t **mem_out)
{
    uint64_t rl = strlen(ranks), cl = strlen(conf), wl = strlen(wrap);
    uint64_t cap = toks_tiktoken_arena_bound(rl, cl, wl);
    uint8_t *mem = malloc((size_t)cap);
    toks_arena ar = { mem, cap, 0 };
    toks_err err = { 0, NULL };
    int64_t r = toks_tiktoken_ranks((const uint8_t *)ranks, rl, &ar, cfg, &err);
    if (r == 0) { r = toks_tiktoken_kimi((const uint8_t *)conf, cl, (const uint8_t *)wrap, wl, &ar, cfg, info, &err); }
    *why = err.what;
    if (mem_out != NULL) { *mem_out = mem; } else { free(mem); }
    return r;
}

static void test_kimi_rules(void)
{
    char *ranks = hand_file(1, 0, 0);
    char *wrap = wrapper_text(-1, -1, -1, 0);
    const char *why = NULL;
    toks_config cfg;
    toks_tiktoken_info info;
    uint8_t *mem = NULL;
    int64_t r = kimi_of(ranks, CFG_OK, wrap, &why, &cfg, &info, &mem);
    CHECK(r == 0, "kimi hand: %" PRId64 " %s", r, why ? why : "");
    if (r == 0) {
        CHECK(cfg.n_added == 256u && cfg.n_ids == 267u + 256u && info.first_special == 267u && info.n_ranks == 267u,
              "kimi hand: specials %u n_ids %u", cfg.n_added, cfg.n_ids);
        CHECK(info.chunk_chars == 400000u && info.run_chars == 25000u, "kimi hand: chunking constants");
        CHECK(info.n_trie == 5u && info.n_named == 4u, "kimi hand: trie %u named %u", info.n_trie, info.n_named);
        CHECK(cfg.added[2].len == 8 && memcmp(cfg.added[2].content, "<|open|>", 8) == 0 && cfg.added[2].special == 0u &&
              cfg.added[2].id == 269u, "kimi hand: added_tokens_decoder name");
        CHECK(cfg.added[3].len == 22 && memcmp(cfg.added[3].content, "<|reserved_token_270|>", 22) == 0, "kimi hand: reserved name");
        CHECK(cfg.added[0].special == 1u && cfg.added[1].special == 1u && cfg.added[254].special == 1u &&
              cfg.added[255].special == 1u && cfg.added[3].special == 0u, "kimi hand: named specials");
        CHECK((info.trie[0] & 0x07u) == 0x07u && (info.trie[0] & 0x08u) == 0u && (info.trie[31] & 0xC0u) == 0xC0u, "kimi hand: trie bits");
        /* the whole config compiles with these specials (the 256 added tokens in K1's index) */
        uint64_t cap = 1u << 20;
        uint8_t *pm = malloc(cap);
        toks_arena par = { pm, cap, 0 };
        built B;
        build_tables(&B, &cfg, &par);
        CHECK(B.r == 0 && B.ctx.t.add_n == 256u && B.ctx.t.n_ids == 523u, "kimi hand: compile %" PRId64, B.r);
        if (B.r == 0) {
            uint64_t n = 0;
            const uint8_t *p = B.ctx.t.tok_bytes + B.ctx.t.tok_off[269];
            n = B.ctx.t.tok_off[270] - B.ctx.t.tok_off[269];
            CHECK(n == 8 && memcmp(p, "<|open|>", 8) == 0, "kimi hand: a special's decode bytes");
        }
        free_tables(&B);
        free(pm);
    }
    free(mem);

    /* the parse entry point: the kimi pattern, the wrapper's cuts, the specials as the two matchers */
    {
        uint64_t rl = strlen(ranks), cl = strlen(CFG_OK), wl = strlen(wrap);
        uint64_t cap = toks_tiktoken_arena_bound(rl, cl, wl);
        uint8_t *m2 = malloc((size_t)cap);
        toks_arena ar = { m2, cap, 0 };
        toks_err err = { 0, NULL };
        r = toks_tiktoken_parse((const uint8_t *)ranks, rl, (const uint8_t *)CFG_OK, cl, (const uint8_t *)wrap, wl, &ar, &cfg,
                                &info, &err);
        CHECK(r == 0, "parse: %" PRId64 " %s", r, err.what ? err.what : "");
        if (r == 0) { check_phases(&cfg, &info, "parse"); }
        free(m2);
    }

    /* the wrapper: each line missing is named; the pattern out of order; re-indented is the same file */
    for (int i = 0; i < (int)N_WR; i++) {
        if (i == 1 || i == 10) { continue; }                  /* the join lines are not required */
        char *w = wrapper_text(i, -1, -1, 0);
        r = kimi_of(ranks, CFG_OK, w, &why, &cfg, &info, NULL);
        CHECK(r == TOKS_E_UNSUPPORTED && why != NULL && strstr(why, "tokenization_kimi.py") != NULL, "wrapper without line %d: %" PRId64 " %s",
              i, r, why ? why : "");
        free(w);
    }
    char *w = wrapper_text(-1, 4, 5, 0);
    r = kimi_of(ranks, CFG_OK, w, &why, &cfg, &info, NULL);
    CHECK(r == TOKS_E_UNSUPPORTED && why != NULL && strstr(why, "in order") != NULL, "pattern out of order: %s", why ? why : "");
    free(w);
    w = wrapper_text(-1, -1, -1, 3);
    CHECK(kimi_of(ranks, CFG_OK, w, &why, &cfg, &info, NULL) == 0, "re-indented wrapper: %s", why ? why : "");
    free(w);
    CHECK(kimi_of(ranks, CFG_OK, "", &why, &cfg, &info, NULL) == TOKS_E_UNSUPPORTED, "empty wrapper");

    /* the config rules */
    static const struct { const char *from, *to; int64_t code; const char *why; } CF[] = {
        { "\"bos_token\": \"[BOS]\",", "", TOKS_E_FORMAT, "bos / eos / unk / pad" },
        { "\"bos_token\": \"[BOS]\"", "\"bos_token\": \"[NOPE]\"", TOKS_E_FORMAT, "bos / eos / unk / pad" },
        { "\"267\":", "\"266\":", TOKS_E_UNSUPPORTED, "outside the wrapper's 256" },
        { "\"267\":", "\"0267\":", TOKS_E_FORMAT, "not an id" },
        { "\"lstrip\": false, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},\"268\"",
          "\"lstrip\": true, \"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},\"268\"",
          TOKS_E_UNSUPPORTED, "lstrip" },
        { "\"normalized\": false, \"rstrip\": false, \"single_word\": false, \"special\": true},\"268\"",
          "\"normalized\": true, \"rstrip\": false, \"single_word\": false, \"special\": true},\"268\"",
          TOKS_E_UNSUPPORTED, "normalized" },
        { "\"rstrip\": false, \"single_word\": false, \"special\": true},\"268\"",
          "\"rstrip\": false, \"special\": true},\"268\"", TOKS_E_UNSUPPORTED, "single_word" },
        { "\"eos_token\": {\"content\": \"[EOS]\",", "\"eos_token\": {\"content\": \"[EOS]\", \"rstrip\": true,",
          TOKS_E_FORMAT, "bos / eos / unk / pad" },
        { "\"model_max_length\"", "\"split_special_tokens\": true, \"model_max_length\"", TOKS_E_UNSUPPORTED, "does not know" },
        { "\"TikTokenTokenizer\",", "\"PreTrainedTokenizerFast\",", TOKS_E_UNSUPPORTED, "tokenizer_class" },
        { "tokenization_kimi.TikTokenTokenizer", "tokenization_other.TikTokenTokenizer", TOKS_E_UNSUPPORTED, "auto_map" },
        { "\"extra_special_tokens\": {}", "\"extra_special_tokens\": {\"x\": \"[BOS]\"}", TOKS_E_UNSUPPORTED, "named extra special" },
        { "\"model_max_length\"", "\"backend\": \"tokenizers\", \"model_max_length\"", TOKS_E_UNSUPPORTED, "backend" },
        { "\"model_max_length\"", "\"model_specific_special_tokens\": {\"x\": \"[BOS]\"}, \"model_max_length\"", TOKS_E_UNSUPPORTED, "named extra special" },
        { "<|open|>", "<|op\\u00e9n|>", TOKS_E_UNSUPPORTED, "printable ascii" },
        { "<|open|>", "[BOS]x", TOKS_E_UNSUPPORTED, "overlap" },
        { "<|open|>", "S]<|", TOKS_E_UNSUPPORTED, "overlap" },        /* "[BOS]" ends in S] */
        { "<|open|>", "[BOS]", TOKS_E_UNSUPPORTED, "overlap" },       /* a repeated name */
        { "\"additional_special_tokens\": [\"[EOS]\"]", "\"additional_special_tokens\": [\"hello\"]", TOKS_E_UNSUPPORTED, "not one of the 256" },
        { "{\"added_tokens_decoder\"", "[{\"added_tokens_decoder\"", TOKS_E_FORMAT, "json" },
    };
    for (size_t i = 0; i < sizeof CF / sizeof CF[0]; i++) {
        char *c = cfg_with(CF[i].from, CF[i].to);
        r = kimi_of(ranks, c, wrap, &why, &cfg, &info, NULL);
        CHECK(r == CF[i].code && why != NULL && strstr(why, CF[i].why) != NULL, "config rule %zu: %" PRId64 " \"%s\" (want %s)", i, r,
              why ? why : "", CF[i].why);
        free(c);
    }
    /* transformers 5's save_pretrained keys (docs/models/kimi.md §3.5): inert ones load; extra_special_tokens as a
     * list names specials as additional_special_tokens does; the decoder's special flag never decides (all_special_ids
     * are the attributes' names: nvidia/Kimi-K2.5-NVFP4's [PAD] is special:true, pad_token [EOS]) */
    {
        CHECK(kimi_of(ranks, CFG_OK, wrap, &why, &cfg, &info, NULL) == 0, "CFG_OK");
        uint32_t base = info.n_named;
        static const struct { const char *from, *to; uint32_t named; } OK5[] = {
            { "\"model_max_length\"", "\"backend\": \"custom\", \"is_local\": true, \"local_files_only\": false, "
              "\"tool_parser_type\": \"kimi_k3\", \"processor_class\": \"P\", \"padding_side\": \"left\", "
              "\"model_specific_special_tokens\": {}, \"model_max_length\"", 0u },
            { "\"extra_special_tokens\": {}", "\"extra_special_tokens\": [\"<|open|>\"]", 1u },
            { "\"special\": false", "\"special\": true", 0u },
            { "\"additional_special_tokens\": [\"[EOS]\"]", "\"additional_special_tokens\": [\"<|open|>\"]", 1u },
        };
        for (size_t i = 0; i < sizeof OK5 / sizeof OK5[0]; i++) {
            char *c = cfg_with(OK5[i].from, OK5[i].to);
            r = kimi_of(ranks, c, wrap, &why, &cfg, &info, NULL);
            CHECK(r == 0 && info.n_named == base + OK5[i].named, "transformers 5 config %zu: %" PRId64 " %s, %u named", i, r,
                  why ? why : "", info.n_named);
            free(c);
        }
    }
    /* an attribute missing from added_tokens_decoder still joins the trie (transformers adds it with the
     * id get_vocab gives its name: checked against transformers 5.18.0 on a modified K3 config) */
    {
        char *c2 = cfg_with("\"additional_special_tokens\": [\"[EOS]\"]",
                            "\"additional_special_tokens\": [\"[EOS]\", \"<|reserved_token_300|>\"]");
        r = kimi_of(ranks, c2, wrap, &why, &cfg, &info, NULL);
        CHECK(r == 0 && info.n_trie == 6u && info.n_named == 5u && ((info.trie[(300 - 267) >> 3] >> ((300 - 267) & 7)) & 1u),
              "attribute outside the decoder: %" PRId64 " trie %u named %u", r, info.n_trie, info.n_named);
        free(c2);
    }
    /* reserved names come from the ids: a decoder name may take an unused reserved name's place, but a
     * duplicate of a generated name overlaps */
    char *c = cfg_with("<|open|>", "<|reserved_token_300|>");
    CHECK(kimi_of(ranks, c, wrap, &why, &cfg, &info, NULL) == TOKS_E_UNSUPPORTED, "a decoder name equal to a generated one");
    free(c);
    /* ranks with a gap: the wrapper's specials would collide or leave holes */
    char *gap = hand_file(0, 0, 0);
    CHECK(kimi_of(gap, CFG_OK, wrap, &why, &cfg, &info, NULL) == TOKS_E_UNSUPPORTED && strstr(why, "0..n-1") != NULL, "gap: %s",
          why ? why : "");
    free(gap);
    printf("  kimi rules: %zu wrapper lines, %zu config rules\n", N_WR, sizeof CF / sizeof CF[0]);
    free(wrap);
    free(ranks);
}

/* ---- the real files ---------------------------------------------------------------------------------- */

static int hexval(int c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

static void sha_hex(const uint8_t *d, uint64_t n, char out[65])
{
    uint8_t h[32];
    toks_sha256(d, n, h);
    for (int i = 0; i < 32; i++) { snprintf(out + 2 * i, 3, "%02x", h[i]); }
}

static const char *expect_get(const char *exp, const char *key, char *val, size_t cap)
{
    size_t kl = strlen(key);
    const char *p = exp;
    while ((p = strstr(p, key)) != NULL) {
        if ((p == exp || p[-1] == '\n') && p[kl] == ' ') {
            const char *e = strchr(p + kl + 1, '\n');
            size_t n = e ? (size_t)(e - p - kl - 1) : strlen(p + kl + 1);
            if (n >= cap) { n = cap - 1; }
            memcpy(val, p + kl + 1, n);
            val[n] = 0;
            return val;
        }
        p += kl;
    }
    return NULL;
}

static void test_real(void)
{
    const char *dir = getenv("TOKS_TOKENIZER_CACHE");
    char home[1024], path[1200];
    if (dir == NULL && getenv("HOME") != NULL) {
        snprintf(home, sizeof home, "%s/.cache/toks/tokenizers", getenv("HOME"));
        dir = home;
    }
    uint64_t rl = 0, cl = 0, wl = 0, el = 0;
    snprintf(path, sizeof path, "%s/kimik3.tiktoken", dir);
    uint8_t *ranks = slurp(path, &rl);
    snprintf(path, sizeof path, "%s/kimik3_tokenizer_config.json", dir);
    uint8_t *conf = slurp(path, &cl);
    snprintf(path, sizeof path, "%s/kimik3_tokenization_kimi.py", dir);
    uint8_t *wrap = slurp(path, &wl);
    uint8_t *exp = slurp("tests/data/kimi/expect.txt", &el);
    if (ranks == NULL || conf == NULL || wrap == NULL || exp == NULL) {
        printf("  real: SKIP (kimik3.* missing in %s: uv run tools/corpora/fetch_tokenizers.py --targets)\n", dir);
        free(ranks); free(conf); free(wrap); free(exp);
        return;
    }
    char *ex = realloc(exp, el + 1);
    ex[el] = 0;
    if (!td_text("tests/data/kimi/expect.txt", (const uint8_t *)ex, el)) {
        fails++;
        free(ranks); free(conf); free(wrap); free(ex);
        return;
    }
    uint64_t cap = toks_tiktoken_arena_bound(rl, cl, wl);
    uint8_t *mem = malloc((size_t)cap);
    toks_arena ar = { mem, cap, 0 };
    toks_config cfg;
    toks_tiktoken_info info;
    toks_err err = { 0, NULL };
    int64_t r = toks_tiktoken_parse(ranks, rl, conf, cl, wrap, wl, &ar, &cfg, &info, &err);
    CHECK(r == 0, "real parse: %" PRId64 " %s", r, err.what ? err.what : "");
    if (r == 0) {
        check_phases(&cfg, &info, "real parse");
        CHECK(info.n_trie == 16u && cfg.n_added == 16u + TOKS_TIKTOKEN_RESERVED, "real parse: %u trie, %u entries",
              info.n_trie, cfg.n_added);
    }
    ar.pos = 0;
    r = toks_tiktoken_ranks(ranks, rl, &ar, &cfg, &err);
    if (r == 0) { r = toks_tiktoken_kimi(conf, cl, wrap, wl, &ar, &cfg, &info, &err); }
    CHECK(r == 0, "real: %" PRId64 " %s", r, err.what ? err.what : "");
    if (r != 0) { free(mem); free(ranks); free(conf); free(wrap); free(ex); return; }

    char v[512], h[65];
    CHECK(expect_get(ex, "ranks", v, sizeof v) && strtoul(v, NULL, 10) == info.n_ranks && cfg.n_vocab == info.n_ranks,
          "real: ranks %u", info.n_ranks);
    CHECK(expect_get(ex, "merges", v, sizeof v) && strtoul(v, NULL, 10) == cfg.n_merges, "real: merges %u", cfg.n_merges);
    CHECK(expect_get(ex, "n_ids", v, sizeof v) && strtoul(v, NULL, 10) == cfg.n_ids, "real: n_ids %u", cfg.n_ids);
    /* the token bytes (alphabet strings decoded back) and the merge list, hashed as gen.py hashes tiktoken's */
    {
        uint8_t *tb = malloc(rl * 2 + 16), tmp[65536];
        uint64_t n = 0;
        for (uint32_t id = 0; id < cfg.n_vocab; id++) {
            int64_t m = toks_alpha_bytes(cfg.vocab[id], cfg.vocab_len[id], tmp);
            uint32_t ml = (uint32_t)m;
            memcpy(tb + n, &ml, 4);
            memcpy(tb + n + 4, tmp, (size_t)m);
            n += 4u + (uint64_t)m;
        }
        sha_hex(tb, n, h);
        CHECK(expect_get(ex, "tokens_sha256", v, sizeof v) && strcmp(v, h) == 0, "real: token bytes sha256 %s", h);
        free(tb);
        uint8_t *mb = malloc((size_t)cfg.n_merges * 12u + 1);
        for (uint32_t i = 0; i < cfg.n_merges; i++) {
            memcpy(mb + 12u * i, &cfg.m_left_id[i], 4);
            memcpy(mb + 12u * i + 4, &cfg.m_right_id[i], 4);
            memcpy(mb + 12u * i + 8, &cfg.m_out_id[i], 4);
        }
        sha_hex(mb, (uint64_t)cfg.n_merges * 12u, h);
        CHECK(expect_get(ex, "merges_sha256", v, sizeof v) && strcmp(v, h) == 0, "real: merges sha256 %s", h);
        free(mb);
        buf sb = { 0 };
        for (uint32_t i = 0; i < cfg.n_added; i++) {
            uint32_t ln = cfg.added[i].len;
            bput(&sb, (const char *)&ln, 4);
            bput(&sb, (const char *)cfg.added[i].content, ln);
        }
        sha_hex((const uint8_t *)sb.p, sb.n, h);
        CHECK(expect_get(ex, "specials_sha256", v, sizeof v) && strcmp(v, h) == 0, "real: specials sha256 %s", h);
        free(sb.p);
    }
    {   /* the trie ids and the named (decode-skipped) ids */
        buf t1 = { 0 }, t2 = { 0 };
        char num[16];
        for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {
            if ((info.trie[i >> 3] >> (i & 7u)) & 1u) { snprintf(num, sizeof num, "%s%u", t1.n ? " " : "", info.first_special + i); bputs(&t1, num); }
            if (cfg.added[i].special) { snprintf(num, sizeof num, "%s%u", t2.n ? " " : "", cfg.added[i].id); bputs(&t2, num); }
        }
        CHECK(expect_get(ex, "trie", v, sizeof v) && strcmp(v, t1.p) == 0, "real: trie ids %s", t1.p);
        CHECK(expect_get(ex, "named", v, sizeof v) && strcmp(v, t2.p) == 0, "real: named ids %s", t2.p);
        free(t1.p);
        free(t2.p);
    }
    consistent(&cfg, "real");

    /* the tables, and K6 on tiktoken's own pieces */
    built B;
    build_tables(&B, &cfg, &ar);
    CHECK(B.r == 0, "real: compile / bpe build %" PRId64, B.r);
    uint64_t pieces = 0, bad = 0;
    if (B.r == 0) {
        const toks_tables *t = &B.ctx.t;
        CHECK((t->flags & TOKS_TF_IDS_AS_RANK) != 0u && t->rank2id == NULL && (t->flags & TOKS_TF_IGNORE_MERGES) != 0u,
              "real: flags %#x", t->flags);
        uint64_t prio_bad = 0;
        for (uint32_t i = 0; i < cfg.n_merges; i++) {
            const bpe_mt mt = bpe_mt_of(t);
            if (bpe_mt_find(&mt, bpe_pair_key(cfg.m_left_id[i], cfg.m_right_id[i])) != cfg.m_out_id[i]) { prio_bad++; }
        }
        CHECK(prio_bad == 0, "real: %" PRIu64 " merges whose priority is not the merged id", prio_bad);
        uint64_t bl = 0;
        uint8_t *bc = slurp("tests/data/kimi/bpe_cases.txt", &bl);
        CHECK(bc != NULL, "tests/data/kimi/bpe_cases.txt missing");
        if (bc != NULL && !td_text("tests/data/kimi/bpe_cases.txt", bc, bl)) { fails++; free(bc); bc = NULL; }
        char *line = (char *)bc;
        uint8_t piece[4096];
        uint32_t got[4096];
        td_at at = { "tests/data/kimi/bpe_cases.txt", 0 };
        while (bc != NULL && line < (char *)bc + bl) {
            char *nl = memchr(line, '\n', (size_t)((char *)bc + bl - line));
            if (nl == NULL) { break; }
            *nl = 0;
            at.line++;
            if (line[0] != '#') {
                char *sp = strchr(line, ' ');
                if (sp == NULL || sp == line || (size_t)(sp - line) % 2u != 0u || (size_t)(sp - line) / 2u > sizeof piece) {
                    fails += !td_bad(&at, sp != NULL ? sp : nl, "'<hex bytes> <ids>' expected");
                    break;
                }
                size_t hn = (size_t)(sp - line), n = hn / 2;
                for (size_t i = 0; i < n; i++) { piece[i] = (uint8_t)(hexval(line[2 * i]) << 4 | hexval(line[2 * i + 1])); }
                uint64_t ng = k5k6(t, piece, n, got);
                char *q = sp + 1;
                uint64_t k = 0;
                int ok = 1;
                while (*q) {                                    /* bound: the line (td_num moves q or fails) */
                    unsigned long id = 0;
                    if (!td_num(&at, &q, &id)) { fails++; ok = 0; break; }
                    if (k >= ng || got[k] != id) { ok = 0; }
                    k++;
                    while (*q == ' ') { q++; }
                }
                if (k != ng) { ok = 0; }
                pieces++;
                if (!ok && bad++ < 5) { CHECK(0, "real: K5 / K6 of piece %.*s differ from tiktoken", (int)(hn > 80 ? 80 : hn), line); }
            }
            line = nl + 1;
        }
        CHECK(bad == 0, "real: K6 != tiktoken on %" PRIu64 " of %" PRIu64 " pieces", bad, pieces);
        free(bc);
    }
    free_tables(&B);
    /* without the flag: the builder falls back to raw ranks and rank2id (the tables differ) */
    cfg.ids_as_rank = 0u;
    build_tables(&B, &cfg, &ar);
    CHECK(B.r == 0 && (B.ctx.t.flags & TOKS_TF_IDS_AS_RANK) == 0u && B.ctx.t.rank2id != NULL,
          "real: without cfg.ids_as_rank the builder must fall back to rank2id");
    free_tables(&B);
    printf("  real: kimi k3 %u ranks, %u merges, %u ids, trie %u, named %u; K5 / K6 == tiktoken on %" PRIu64 " pieces\n",
           info.n_ranks, cfg.n_merges, cfg.n_ids, info.n_trie, info.n_named, pieces);
    free(mem);
    free(ranks);
    free(conf);
    free(wrap);
    free(ex);
}

/* $TOKS_KIMI_DIRS (colon-separated model directories: tiktoken.model, tokenizer_config.json,
 * tokenization_kimi.py) are read too: the kimi family check (k2, k2.5, mirrors), printed. */
static void test_dirs(void)
{
    const char *env = getenv("TOKS_KIMI_DIRS");
    if (env == NULL || env[0] == 0) { return; }
    char list[4096];
    snprintf(list, sizeof list, "%s", env);
    for (char *d = strtok(list, ":"); d != NULL; d = strtok(NULL, ":")) {
        char p[1200];
        uint64_t rl = 0, cl = 0, wl = 0;
        snprintf(p, sizeof p, "%s/tiktoken.model", d);
        uint8_t *ranks = slurp(p, &rl);
        snprintf(p, sizeof p, "%s/tokenizer_config.json", d);
        uint8_t *conf = slurp(p, &cl);
        snprintf(p, sizeof p, "%s/tokenization_kimi.py", d);
        uint8_t *wrap = slurp(p, &wl);
        CHECK(ranks && conf && wrap, "%s: files missing", d);
        if (ranks && conf && wrap) {
            uint64_t cap = toks_tiktoken_arena_bound(rl, cl, wl);
            uint8_t *mem = malloc((size_t)cap);
            toks_arena ar = { mem, cap, 0 };
            toks_config cfg;
            toks_tiktoken_info info;
            toks_err err = { 0, NULL };
            int64_t r = toks_tiktoken_ranks(ranks, rl, &ar, &cfg, &err);
            if (r == 0) { r = toks_tiktoken_kimi(conf, cl, wrap, wl, &ar, &cfg, &info, &err); }
            CHECK(r == 0, "%s: %" PRId64 " %s", d, r, err.what ? err.what : "");
            if (r == 0) {
                printf("  dir %s: %u ranks, %u merges, trie %u, named %u\n", d, info.n_ranks, cfg.n_merges, info.n_trie, info.n_named);
            }
            free(mem);
        }
        free(ranks);
        free(conf);
        free(wrap);
    }
}

/* toks_load reaches the reader: a model directory with tiktoken.model (no tokenizer.json), the
 * tiktoken.model path itself, and a ranks file without its companions. ~/.cache/toks/kimik3 holds the
 * hub files under their own names (tests/parity/oracle_tiktoken.py --fetch); SKIP when absent. */
static void test_load(void)
{
    const char *home = getenv("HOME");
    char dir[1024], file[1100], cache[1100];
    snprintf(dir, sizeof dir, "%s", getenv("TOKS_KIMI_DIR") ? getenv("TOKS_KIMI_DIR") : "");
    if (dir[0] == 0 && home != NULL) { snprintf(dir, sizeof dir, "%s/.cache/toks/kimik3", home); }
    snprintf(file, sizeof file, "%s/tiktoken.model", dir);
    FILE *f = fopen(file, "rb");
    if (f == NULL) {
        printf("  load: SKIP (%s missing: tests/parity/oracle_tiktoken.py --fetch)\n", file);
        return;
    }
    fclose(f);
    toks_diag d;
    toks_load_opts o = { sizeof(toks_load_opts), 0, 0, 0, &d };
    toks_ctx *c = NULL;
    const char *paths[2] = { dir, file };
    for (int i = 0; i < 2; i++) {
        memset(&d, 0, sizeof d);
        c = NULL;
        int64_t r = toks_load(&c, paths[i], &o);
        CHECK(r == 0 && c != NULL, "toks_load(%s): %" PRId64 " %s", paths[i], r, d.what);
        toks_info in;
        memset(&in, 0, sizeof in);
        in.size = (uint32_t)sizeof in;
        CHECK(c != NULL && toks_get_info(c, &in) == 0 && in.n_ids == 163840u, "toks_load(%s): n_ids %u", paths[i], in.n_ids);
        toks_unload(c);
        c = NULL;
    }
    snprintf(cache, sizeof cache, "%s/.cache/toks/tokenizers/kimik3.tiktoken", home != NULL ? home : "");
    f = fopen(cache, "rb");
    if (f != NULL) {
        fclose(f);
        memset(&d, 0, sizeof d);
        int64_t r = toks_load(&c, cache, &o);
        CHECK(r == TOKS_E_OPEN && strstr(d.what, "beside it") != NULL, "toks_load(ranks without companions): %" PRId64 " %s", r, d.what);
    }
    uint64_t len = 0;
    uint8_t *ranks = slurp(file, &len);
    memset(&d, 0, sizeof d);
    CHECK(ranks != NULL && toks_load_mem_copy(&c, ranks, len, &o) == TOKS_E_UNSUPPORTED && strstr(d.what, "three files") != NULL,
          "toks_load_mem_copy(ranks): %s", d.what);
    free(ranks);
    printf("  load: %s and its tiktoken.model load (the kimi template, the wrapper's cuts)\n", dir);
}

int main(void)
{
    alphabet_init();
    test_load();
    test_hand();
    test_hostile();
    test_kimi_rules();
    test_real();
    test_dirs();
    printf("test_tiktoken: %" PRIu64 " checks, %d failures\n", checks, fails);
    return fails != 0;
}
