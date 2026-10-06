/*
 * test_vocab.c: the vocabulary lookups (include/toks.h toks_token_to_id, toks_id_flags; src/core/vocab.c) against
 * their rule, read again here from a sorted copy of every key (independent of vocab.c's index):
 *
 * 1. every id's string, and near it (one byte longer, one byte shorter, its last byte changed), gets exactly the
 *    rule's answer: the id of an added content of those bytes (every listing in the file, read with
 *    toks_config_parse; K1's entries for a source too big to parse again here or not json), else of the ids with
 *    that string the one the file writes as those very bytes, else the later one, else TOKS_E_ID; the api's errors.
 * 2. flags: SPECIAL only with ADDED, every id K1 can match is ADDED, BYTE exactly for an id that stands for one raw
 *    byte (a <0xHH> string of a chain with ByteFallback, a byte-level id of one byte); ids hf's files pin (gpt2,
 *    llama2, llama3, the spm fixtures).
 * Real files from ~/.cache/toks/tokenizers or $TOKS_TOKENIZER_CACHE (a missing one prints SKIP), the fixtures from
 * tests/data. hf's own answers for every cached file: tests/parity/vocab_sweep.py.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
#include "spm.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static uint64_t rng = 0x70C5AB0C0FFEE123ull;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static char *path_of(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    static char buf[1024];
    if (root != NULL && root[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", root, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
}

static int has_bf(const toks_ctx *c)
{
    for (uint32_t i = 0; c->dc.on && i < c->dc.n_dec; i++) { if (c->dc.dec[i].kind == TOKS_SPM_D_BYTE_FALLBACK) { return 1; } }
    return 0;
}

static int byte_level(const toks_ctx *c) { return !c->dc.on && c->wp == NULL && c->dec_byte_level != 0u; }

/* ---- the reference: every key the rule knows, sorted ---------------------------------------------------------- */

typedef struct rkey { const uint8_t *p; uint32_t n, id; uint8_t content, written; } rkey;
typedef struct refx { rkey *k; uint64_t n; uint8_t *src, *ar; int cfg; } refx;

static int rkey_cmp(const void *x, const void *y)            /* bytes, then contents first, then the id */
{
    const rkey *a = (const rkey *)x, *b = (const rkey *)y;
    int c = memcmp(a->p, b->p, a->n < b->n ? a->n : b->n);
    if (c != 0) { return c; }
    if (a->n != b->n) { return a->n < b->n ? -1 : 1; }
    if (a->content != b->content) { return a->content ? -1 : 1; }
    return a->id < b->id ? -1 : a->id > b->id;
}

/* the keys of c: the file's added contents (every listing) and written vocabulary strings when it is a json small
 * enough to parse again here (4 MiB: the parse arena is 32 x the source), else K1's entries and no written forms */
static void refx_build(refx *R, const toks_ctx *c, const char *path)
{
    memset(R, 0, sizeof *R);
    toks_config cfg;
    memset(&cfg, 0, sizeof cfg);
    FILE *f = fopen(path, "rb");
    long len = -1;
    if (f != NULL && fseek(f, 0, SEEK_END) == 0) { len = ftell(f); }
    if (f != NULL && len > 0 && len <= (4l << 20) && fseek(f, 0, SEEK_SET) == 0) {
        R->src = (uint8_t *)malloc((size_t)len);
        uint64_t cap = toks_config_arena_bound((uint64_t)len);
        R->ar = (uint8_t *)malloc((size_t)cap);
        toks_arena ar = { R->ar, cap, 0u };
        toks_err err = { 0, NULL };
        R->cfg = R->src != NULL && R->ar != NULL && fread(R->src, 1, (size_t)len, f) == (size_t)len &&
                 toks_config_parse(R->src, (uint64_t)len, &ar, &cfg, &err) == 0;
    }
    if (f != NULL) { fclose(f); }
    uint64_t na = R->cfg ? cfg.n_added : c->t.add_n;
    R->k = (rkey *)malloc((na + c->t.n_ids + 1u) * sizeof(rkey));
    for (uint64_t i = 0; i < na; i++) {
        rkey *k = &R->k[R->n++];
        if (R->cfg) { k->p = cfg.added[i].content, k->n = cfg.added[i].len, k->id = cfg.added[i].id; }
        else {
            const toks_added_entry *x = &c->t.add_entries[i];
            k->p = c->t.add_bytes + x->off, k->n = x->len, k->id = x->id;
        }
        k->content = 1, k->written = 0;
    }
    for (uint32_t id = 0; id < c->t.n_ids; id++) {
        uint64_t n = 0;
        const uint8_t *s = toks_token(c, id, &n);
        if (s == NULL) { continue; }
        rkey *k = &R->k[R->n++];
        k->p = s, k->n = (uint32_t)n, k->id = id, k->content = 0;
        k->written = R->cfg && id < cfg.n_vocab && cfg.vocab_len[id] == n && memcmp(cfg.vocab[id], s, (size_t)n) == 0;
    }
    qsort(R->k, (size_t)R->n, sizeof(rkey), rkey_cmp);
}

static void refx_free(refx *R) { free(R->k); free(R->src); free(R->ar); }

/* the rule's answer for m[0, n): an added content's id, else of the ids with that string the one written as those
 * bytes (unless a later one is too), else the later, else TOKS_E_ID */
static int64_t refx_lookup(const refx *R, const uint8_t *m, uint64_t n)
{
    if (n == 0u) { return TOKS_E_ID; }
    rkey q = { m, (uint32_t)n, 0u, 1u, 0u };
    uint64_t lo = 0, hi = R->n;
    while (lo < hi) {                                        /* the first key >= m (contents sort first) */
        uint64_t mid = lo + (hi - lo) / 2u;
        if (rkey_cmp(&R->k[mid], &q) < 0) { lo = mid + 1u; } else { hi = mid; }
    }
    int64_t best = TOKS_E_ID;
    int bw = 0;
    for (uint64_t i = lo; i < R->n && R->k[i].n == n && memcmp(R->k[i].p, m, (size_t)n) == 0; i++) {
        if (R->k[i].content) { return R->k[i].id; }          /* the first listing (the file's ids agree, hf's rule) */
        if (best < 0 || !(bw && !R->k[i].written)) { best = R->k[i].id; bw = R->k[i].written; }   /* ids ascending */
    }
    return best;
}

static uint64_t g_ids, g_rt, g_won, g_miss, g_added, g_special, g_byte;

static void test_ctx(const char *name, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {                                         /* a directory (kimi's) opens elsewhere: try the load */
        toks_ctx *t = NULL;
        if (toks_load(&t, path, NULL) != 0) { printf("  %-12s SKIP: %s missing\n", name, path); return; }
        toks_unload(t);
    } else {
        fclose(f);
    }
    toks_ctx *c = NULL;
    int64_t lr = toks_load(&c, path, NULL);
    CHECK(lr == 0, "%s: load %" PRId64, name, lr);
    if (lr != 0) { return; }
    uint32_t nv = c->t.n_ids, bf = (uint32_t)has_bf(c), bl = (uint32_t)byte_level(c);
    uint64_t rt = 0, won = 0, miss = 0, na = 0, ns = 0, nb = 0;
    uint8_t buf[TOKS_MAX_TOKEN_BYTES + 2u];
    refx R;
    refx_build(&R, c, path);
    uint8_t *added = (uint8_t *)calloc(nv ? nv : 1u, 1u);    /* the ids an added content holds */
    for (uint64_t i = 0; added != NULL && i < R.n; i++) { if (R.k[i].content && R.k[i].id < nv) { added[R.k[i].id] = 1u; } }
    for (uint32_t id = 0; id < nv; id++) {
        uint64_t n = 0;
        const uint8_t *s = toks_token(c, id, &n);
        int64_t fl = toks_id_flags(c, id);
        CHECK(fl >= 0 && (fl & ~(int64_t)7) == 0 && ((fl & TOKS_ID_SPECIAL) == 0 || (fl & TOKS_ID_ADDED) != 0),
              "%s: flags of %u: %" PRId64, name, id, fl);
        int byte = s != NULL && ((bf && toks_byte_token(s, n) >= 0) || (bl && n == 1u));
        CHECK(((fl & TOKS_ID_BYTE) != 0) == byte, "%s: BYTE of %u", name, id);
        na += (fl & TOKS_ID_ADDED) != 0, ns += (fl & TOKS_ID_SPECIAL) != 0, nb += (fl & TOKS_ID_BYTE) != 0;
        if (s == NULL) {                                     /* toks.h: 0 for an id with no string (no content holds it) */
            CHECK(n == 0u && (fl == 0 || (added != NULL && added[id])), "%s: id %u has no string, flags %" PRId64, name, id, fl);
            continue;
        }
        int64_t r = toks_token_to_id(c, s, n), want = refx_lookup(&R, s, n);
        CHECK(r == want, "%s: id %u's string -> %" PRId64 ", the rule's %" PRId64, name, id, r, want);
        if (r == (int64_t)id) { rt++; } else { won++; }
        /* misses near it: one byte longer, shorter, the last byte changed (sampled on big vocabularies) */
        if (nv > 4096u && (next() % 2048u) != 0u) { continue; }
        for (int v = 0; v < 3; v++) {
            uint64_t m = n;
            memcpy(buf, s, (size_t)n);
            if (v == 0) { buf[n] = (uint8_t)next(); m = n + 1u; }
            else if (v == 1) { if (n < 2u) { continue; } m = n - 1u; }
            else { buf[n - 1u] ^= (uint8_t)(1u + next() % 255u); }
            int64_t want = refx_lookup(&R, buf, m), got = toks_token_to_id(c, buf, m);
            CHECK(got == want, "%s: near id %u (v %d) -> %" PRId64 ", the rule's %" PRId64, name, id, v, got, want);
            miss += want == TOKS_E_ID;
        }
    }
    /* every token K1 can match is an added id */
    for (uint64_t i = 0; i < c->t.add_n; i++) {
        uint32_t id = c->t.add_entries[i].id;
        CHECK((toks_id_flags(c, id) & TOKS_ID_ADDED) != 0, "%s: K1's entry %" PRIu64 " (id %u) not ADDED", name, i, id);
        const toks_added_entry *x = &c->t.add_entries[i];
        if (x->phase == 0u) {                                /* a raw content: its own key */
            int64_t r = toks_token_to_id(c, c->t.add_bytes + x->off, x->len);
            CHECK(r == (int64_t)id || (r >= 0 && (toks_id_flags(c, (uint32_t)r) & TOKS_ID_ADDED) != 0),
                  "%s: K1 content %" PRIu64 " -> %" PRId64 ", its id %u", name, i, r, id);
        }
    }
    /* the api */
    CHECK(toks_token_to_id(NULL, "a", 1u) == TOKS_E_ARG && toks_token_to_id(c, NULL, 1u) == TOKS_E_ARG &&
          toks_token_to_id(c, NULL, 0u) == TOKS_E_ID && toks_token_to_id(c, "a", 0u) == TOKS_E_ID &&
          toks_id_flags(NULL, 0u) == TOKS_E_ARG && toks_id_flags(c, nv) == TOKS_E_ID &&
          toks_id_flags(c, UINT32_MAX) == TOKS_E_ID, "%s: api errors", name);
    memset(buf, 'z', sizeof buf);
    CHECK(toks_token_to_id(c, buf, TOKS_MAX_TOKEN_BYTES + 1u) == TOKS_E_ID, "%s: a string over the longest token", name);
    printf("  %-14s ok: %u ids, %" PRIu64 " round-trip, %" PRIu64 " won by another id, %" PRIu64 " near misses,"
           " %" PRIu64 " added (%" PRIu64 " special), %" PRIu64 " byte; index %.2f B/id; contents from %s\n", name, nv,
           rt, won, miss, na, ns, nb, (double)c->mem_voc_len / (double)(nv ? nv : 1u), R.cfg ? "the file" : "K1");
    refx_free(&R);
    free(added);
    g_ids += nv, g_rt += rt, g_won += won, g_miss += miss, g_added += na, g_special += ns, g_byte += nb;
    toks_unload(c);
}

/* ids hf 0.23.2 gives these strings (and the flags its added_tokens_decoder gives the ids) */
typedef struct pin { const char *s; int64_t id; int64_t flags; } pin;

static void test_pins(const char *name, const char *path, const pin *p, size_t n)
{
    toks_ctx *c = NULL;
    if (toks_load(&c, path, NULL) != 0) { printf("  %-14s SKIP: %s\n", name, path); return; }
    for (size_t i = 0; i < n; i++) {
        int64_t r = toks_token_to_id(c, p[i].s, strlen(p[i].s));
        CHECK(r == p[i].id, "%s: \"%s\" -> %" PRId64 ", want %" PRId64, name, p[i].s, r, p[i].id);
        if (p[i].id >= 0 && p[i].flags >= 0) {
            CHECK(toks_id_flags(c, (uint32_t)p[i].id) == p[i].flags, "%s: flags of %" PRId64 ": %" PRId64 ", want %"
                  PRId64, name, p[i].id, toks_id_flags(c, (uint32_t)p[i].id), p[i].flags);
        }
    }
    printf("  %-14s pins ok: %zu strings\n", name, n);
    toks_unload(c);
}

#define AS TOKS_ID_ADDED | TOKS_ID_SPECIAL

/* toks.h's "then the later one", which no file toks loads reaches (a byte-level string outside the alphabet is
 * written as its own bytes, a piece's string is its key, a content wins first), on a hand-built context through
 * vocab.c's own builder: ids 1 and 2 decode to "zz" and the file spells them "Z1" and "Z2", so neither is written as
 * those bytes: the later, 2; ids 3 and 4 decode to "qq", 3 written so: 3, though 4 is later */
static void test_later_by_hand(void)
{
    static const char *const STR[] = { "a", "zz", "zz", "qq", "qq", "b" };
    static const char *const WROTE[] = { "a", "Z1", "Z2", "qq", "Q4", "b" };
    enum { N = 6 };
    uint32_t off[N + 1u], vlen[N];
    uint8_t bytes[16];
    const uint8_t *vocab[N];
    uint32_t o = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t k = (uint32_t)strlen(STR[i]);
        off[i] = o;
        memcpy(bytes + o, STR[i], k);
        o += k;
        vocab[i] = (const uint8_t *)WROTE[i];
        vlen[i] = (uint32_t)strlen(WROTE[i]);
    }
    off[N] = o;
    toks_ctx c;
    memset(&c, 0, sizeof c);
    c.t.n_ids = N, c.t.tok_off = off, c.t.tok_bytes = bytes;
    toks_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.vocab = vocab, cfg.vocab_len = vlen, cfg.n_vocab = N;
    CHECK(toks_vocab_build(&c, &cfg) == 0, "by hand: the index");
    int64_t zz = toks_token_to_id(&c, "zz", 2u), qq = toks_token_to_id(&c, "qq", 2u);
    CHECK(zz == 2 && qq == 3 && toks_token_to_id(&c, "a", 1u) == 0 && toks_token_to_id(&c, "b", 1u) == 5,
          "by hand: \"zz\" -> %" PRId64 " (want 2, the later), \"qq\" -> %" PRId64 " (want 3, written so)", zz, qq);
    printf("  %-14s ok: two ids of one string, neither written so: the later\n", "by hand");
    toks_plat_arena_free(c.mem_voc, c.mem_voc_len);
}

/* toks.h's TOKS_ID_BYTE "(or hf's <0x+F>)": tests/data/vocab/byte_plus.json spells llamalike's byte token 0F as
 * "<0x+F>" (id 16). hf 0.23.2 decodes it as the byte 0F (u8::from_str_radix takes the '+'), and its byte fallback
 * cannot reach it ("a\x0fb" encodes to unk there, hf: 269 257 0 258); toks: a byte id (its string is the piece as
 * written, as every sentencepiece piece's), decode 0F */
static void test_byte_plus(void)
{
    toks_ctx *c = NULL;
    if (toks_load(&c, "tests/data/vocab/byte_plus.json", NULL) != 0) { CHECK(0, "byte_plus.json (run from the source root)"); return; }
    uint64_t n = 0;
    const uint8_t *t = toks_token(c, 16u, &n);
    uint32_t ids[2] = { 16u, 17u }, out[16];
    uint8_t d[16];
    int64_t nd = toks_decode(c, ids, 2u, 0u, d, sizeof d);
    uint64_t sb = toks_scratch_bytes(c, 64u, 0u);
    void *scr = malloc((size_t)sb);
    int64_t ne = (scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0) ? toks_encode(c, "a\x0f" "b", 3u, 0u, out, 16u, scr) : -1;
    static const uint32_t HF[] = { 269u, 257u, 0u, 258u };
    CHECK(toks_id_flags(c, 16u) == TOKS_ID_BYTE && t != NULL && n == 6u && memcmp(t, "<0x+F>", 6u) == 0 &&
          toks_token_to_id(c, "<0x+F>", 6u) == 16 && nd == 2 && d[0] == 0x0Fu && d[1] == 0x10u,
          "byte_plus: id 16 \"<0x+F>\": flags %" PRId64 ", %" PRIu64 " bytes, decode %" PRId64, toks_id_flags(c, 16u), n, nd);
    CHECK(ne == 4 && memcmp(out, HF, sizeof HF) == 0, "byte_plus: \"a\\x0fb\" encodes to %" PRId64 " ids, hf 269 257 0 258", ne);
    printf("  %-14s ok: \"<0x+F>\" is the byte 0F (flags, decode); encode is hf's\n", "byte_plus");
    free(scr);
    toks_unload(c);
}

int main(void)
{
    static const pin GPT2[] = {
        { "<|endoftext|>", 50256, AS }, { "hello", 31373, 0 }, { " hello", 23748, 0 }, { "\xc4\xa0hello", TOKS_E_ID, -1 },
        { "!", 0, TOKS_ID_BYTE },
        /* the footgun toks.h names: hf's spelling "\xc2\xa2" (the byte A2's alphabet char) as bytes is another id */
        { "\xa2", 95, TOKS_ID_BYTE }, { "\xc2\xa2", 44359, 0 },
    };
    static const pin LLAMA2[] = {
        { "<s>", 1, AS }, { "\xe2\x96\x81<s>", 1, AS }, { "</s>", 2, AS }, { "<unk>", 0, AS }, { "<0x41>", 68, TOKS_ID_BYTE },
        { "<0x00>", 3, TOKS_ID_BYTE }, { "\xe2\x96\x81hello", 22172, 0 }, { " hello", TOKS_E_ID, -1 },
    };
    static const pin LLAMA3[] = {
        { "<|begin_of_text|>", 128000, AS }, { "<|eot_id|>", 128009, AS }, { " hello", 24748, 0 }, { "hello", 15339, 0 },
    };
    /* hf's answers on the fixtures where tokens share ids (holes_added: a later token holds the id, a stale form
     * outlives it; norm_specials: an added "▁</s>" and </s>'s normalized form are one string) */
    static const pin HOLES[] = {
        { "<A>", 39, AS }, { "<B>", 40, AS }, { "<C>", 41, TOKS_ID_ADDED }, { "<D>", 42, TOKS_ID_ADDED },
        { "hell", 42, TOKS_ID_ADDED }, { "\xe2\x96\x81<D>", 42, -1 }, { "<E>", 43, AS }, { "hello", 44, TOKS_ID_ADDED },
        { "<F>", 44, TOKS_ID_ADDED }, { "<G>", 45, TOKS_ID_ADDED }, { "<H>", 46, AS }, { "\xe2\x96\x81w", 46, AS },
        { "<I>", 47, TOKS_ID_ADDED }, { "<JJJJJJJJJJ>", 48, AS }, { "or", 48, AS },
    };
    static const pin NORMSP[] = {
        { "<unk>", 0, AS }, { "<s>", 1, AS }, { "</s>", 2, AS }, { "<pad>", 297, AS }, { "\xe2\x96\x81</s>", 298, AS },
    };
    /* dup_added (tests/data/breadth): "<b>" and "<c>" are each listed special, then again not special. Our special bit
     * is any listing's (hf's special_tokens_set, which decode's skip reads): 296 and 297 are ADDED | SPECIAL. hf's
     * added_tokens_decoder keeps the last listing instead, special False for both: the documented divergence. */
    static const pin DUP[] = {
        { "<a>", 295, AS }, { "<b>", 296, AS }, { "<c>", 297, AS }, { "c>", 298, TOKS_ID_ADDED }, { "<d>", 299, TOKS_ID_ADDED },
        { "  ", 300, TOKS_ID_ADDED },
    };
    /* written_tie (tests/data/vocab/gen.py): a raw U+200D (295) and the alphabet's "\u00e2\u0122\u012f" (296) both
     * decode to E2 80 8D; hf's token_to_id gives 295 for that text, the id the file writes as those bytes, not the
     * later one */
    static const pin TIE[] = { { "\xe2\x80\x8d", 295, 0 }, { "\xc3\xa2\xc4\xa2\xc4\xaf", TOKS_E_ID, -1 } };
    /* content_first (tests/data/vocab/gen.py): the content "\u2581<q>" (299) and "<q>" normalized (300, its string
     * "\u2581<q>"): hf's token_to_id gives 299 for that text, the content, not the later id of the same bytes */
    static const pin CF[] = { { "\xe2\x96\x81<q>", 299, AS }, { "<q>", 300, AS } };
    test_pins("dup_added", "tests/data/breadth/dup_added.json", DUP, sizeof DUP / sizeof DUP[0]);
    test_pins("written_tie", "tests/data/vocab/written_tie.json", TIE, sizeof TIE / sizeof TIE[0]);
    test_pins("content_first", "tests/data/vocab/content_first.json", CF, sizeof CF / sizeof CF[0]);
    test_later_by_hand();
    test_byte_plus();
    test_pins("holes_added", "tests/data/spm/holes_added.json", HOLES, sizeof HOLES / sizeof HOLES[0]);
    test_pins("norm_specials", "tests/data/spm/norm_specials.json", NORMSP, sizeof NORMSP / sizeof NORMSP[0]);
    test_pins("gpt2", path_of("gpt2"), GPT2, sizeof GPT2 / sizeof GPT2[0]);
    test_pins("llama2", path_of("llama2"), LLAMA2, sizeof LLAMA2 / sizeof LLAMA2[0]);
    test_pins("llama3", path_of("llama3"), LLAMA3, sizeof LLAMA3 / sizeof LLAMA3[0]);
    static const char *REAL[] = { "gpt2", "llama3", "o200k", "qwen38", "glm53", "gemma4", "mistral-v0.3", "llama2", "llama1",
                                  "dsv3", "wp-bert-uncased", "uni_t5base", "uni_albert" };
    for (size_t i = 0; i < sizeof REAL / sizeof REAL[0]; i++) { test_ctx(REAL[i], path_of(REAL[i])); }
    char kimi[1024];
    snprintf(kimi, sizeof kimi, "%s/.cache/toks/kimik3", getenv("HOME") ? getenv("HOME") : ".");
    test_ctx("kimik3", kimi);
    static const char *FIX[] = { "spm/gemma4like.json", "spm/llamalike.json", "spm/mistrallike.json", "spm/holes_added.json",
                                 "spm/holes_nodec.json", "spm/norm_specials.json", "spm/unk_fused.json", "spm/space_in_vocab.json",
                                 "spm/meta_always_split.json", "breadth/dup_added.json", "vocab/written_tie.json",
                                 "vocab/content_first.json", "vocab/byte_plus.json" };
    for (size_t i = 0; i < sizeof FIX / sizeof FIX[0]; i++) {
        char fp[256];
        snprintf(fp, sizeof fp, "tests/data/%s", FIX[i]);
        test_ctx(FIX[i], fp);
    }
    printf("test_vocab: %ld checks, %" PRIu64 " ids, %" PRIu64 " round-trip, %" PRIu64 " won by another id, %" PRIu64
           " near misses, %" PRIu64 " added, %" PRIu64 " special, %" PRIu64 " byte, %d failures\n", checks, g_ids, g_rt,
           g_won, g_miss, g_added, g_special, g_byte, failures);
    return failures != 0;
}
