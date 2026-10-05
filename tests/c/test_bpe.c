/*
 * test_bpe.c: the bpe gate under `make test` (SPEC §2.7, kernels.md §5-6): the table
 * builder (toks_bpe_build) and the K6 / K5 c twins, on hand-built models whose tables are filled
 * the way the compile step fills them (n_ids, tok_off, tok_bytes = each token's raw bytes).
 *
 * The oracle is ref_bpe: hf's rule restated from the raw merge list (merge the adjacent pair whose
 * LAST raw rank is lowest, leftmost on ties, until none merges; ignore_merges = a whole-piece
 * model-vocabulary hit), never from the built tables. It is tests/model/toks_model.py's rule, which
 * matched hf 0.23.2 over 1.2M pieces; tests/bpe/ diffs K6 against hf itself on gpt-2 and llama 3.
 *
 *   hand     SPEC §2.7's trap on both prio paths; equal-prio leftmost ties; vocabulary strings
 *            unreachable by merges; ignore_merges on and off; added-only tokens (never an
 *            ignore_merges hit); tokens without a raw form; duplicate merges (the last wins);
 *            merged ids not rising with rank (rank2id); one-byte pieces; zero merges; build errors;
 *            a 1 MiB piece (n log n, timed).
 *   random   models with shuffled ids, duplicate merges, several merges to one string, merges ranked
 *            before the merges that build their parts, unreachable / rawless / added-only tokens:
 *            K6 against ref on every piece, guard pages on the piece, out and work.
 *   tables   byte2id, merge table, bytepair and vhash against the merge list; the merge table's
 *            layout invariant (layout.h: buckets fill front to back, a key's probe path is full up to
 *            it, inside merge_maxprobe: the asm probes stop at the first empty slot); the certified words
 *            table: every entry is the reference's answer for its key, and a candidate (a model token
 *            of 2..15 bytes with <= 4 ids) is left out only when both of its buckets are full or the
 *            0.85 load cap is reached.
 *   K5       against ref over random piece streams (no cache, a 1-bucket cache, 16 buckets; cold and
 *            warm), out room = bytes + 4 flush against a guard page; counters (added to what a held);
 *            the fill rule (static and K6 answers both enter the cache; a hit fills nothing); the tag
 *            (an entry answers only under the tag it was filled with); the in-place key
 *            (every piece of every segment of 1..48 bytes, flush against a guard page at either end).
 *   arena    every build runs in an arena of exactly toks_bpe_tables_bytes(cfg) bytes, base 64-aligned
 *            and base + 8.
 */
#include "../../src/core/bpe.h"
#include "bpe_words.h"
#include "../../src/core/kernels.h"
#include "../common/guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* the tier under test: the c twins by default; make test also builds test_bpe_<tier> for every asm tier with
 * K5 / K6 kernels (the Makefile's BPE_TIERS): -DK6_BPE=toks_k6_bpe_<tier> -DK5_ENCODE=toks_k5_encode_<tier>,
 * TEST_BPE_TIER its name and TEST_BPE_FEAT the cpu features it needs (a cpu without them skips) */
#ifndef K6_BPE
#define K6_BPE toks_k6_bpe_c
#endif
#ifndef K5_ENCODE
#define K5_ENCODE toks_k5_encode_c
#endif
uint64_t K6_BPE(const toks_tables *t, toks_k6_args *a);
uint64_t K5_ENCODE(const toks_tables *t, toks_k5_args *a);
#ifdef TEST_BPE_FEAT
#include "cpu.h"
#define TEST_BPE_NAME "test_bpe_" TEST_BPE_TIER
#else
#define TEST_BPE_NAME "test_bpe"
#endif

static uint64_t checks, failures;
#define CHECK(cond, ...) do {                                                        \
        checks++;                                                                    \
        if (!(cond)) {                                                               \
            if (failures++ < 40) {                                                   \
                fprintf(stderr, "FAIL test_bpe.c:%d: ", __LINE__);                   \
                fprintf(stderr, __VA_ARGS__);                                        \
                fprintf(stderr, "\n");                                               \
            }                                                                        \
        }                                                                            \
    } while (0)

/* ======================================================================================== the model */

#define MAXTOK   1024
#define MAXBYTES (1u << 16)
#define MAXMERGE 2048
#define BIGLEN   (1u << 20)

typedef struct model {
    uint32_t n_ids, n_vocab;                 /* ids >= n_vocab are added-only */
    uint32_t off[MAXTOK + 1];
    uint8_t  bytes[MAXBYTES];
    uint32_t ml[MAXMERGE], mr[MAXMERGE], mo[MAXMERGE], nm;
    int      ignore_merges;
    toks_tables t;
    uint8_t *mem;
    uint64_t arena_need, arena_used;
    uint32_t b2i[256];                       /* the reference's byte tokens */
} model;

static model M;
static int32_t RANK[MAXTOK][MAXTOK];         /* the reference's pair -> LAST raw rank, -1 = none */
static uint64_t arena_shift;                 /* alternates 0 / 8: the arena base alignment */

static void m_reset(model *m)
{
    for (uint32_t i = 0; i < m->nm; i++) {
        if (m->ml[i] < MAXTOK && m->mr[i] < MAXTOK) { RANK[m->ml[i]][m->mr[i]] = -1; }
    }
    guard_aligned_free(m->mem);
    memset(m, 0, sizeof *m);
}

static uint32_t m_add(model *m, const void *s, uint32_t len)
{
    uint32_t id = m->n_ids++;
    memcpy(m->bytes + m->off[id], s, len);
    m->off[id + 1] = m->off[id] + len;
    return id;
}

static void m_add_bytes(model *m)
{
    for (uint32_t b = 0; b < 256; b++) { uint8_t c = (uint8_t)b; m_add(m, &c, 1); }
}

static uint32_t m_len(const model *m, uint32_t id) { return m->off[id + 1] - m->off[id]; }

static int64_t m_find(const model *m, const void *s, uint64_t len, uint32_t lim)
{
    for (uint32_t id = 0; id < lim; id++) {
        if (m_len(m, id) == len && memcmp(m->bytes + m->off[id], s, len) == 0) { return id; }
    }
    return -1;
}

static uint32_t m_id(const model *m, const char *s)
{
    int64_t id = m_find(m, s, strlen(s), m->n_vocab ? m->n_vocab : m->n_ids);
    if (id < 0) { fprintf(stderr, "test bug: no token '%s'\n", s); exit(2); }
    return (uint32_t)id;
}

static void m_merge(model *m, const char *l, const char *r)
{
    char o[64];
    snprintf(o, sizeof o, "%s%s", l, r);
    m->ml[m->nm] = m_id(m, l);
    m->mr[m->nm] = m_id(m, r);
    m->mo[m->nm] = m_id(m, o);
    m->nm++;
}

/* the model tokens' byte-level alphabet strings (gpt-2's bytes_to_unicode, utf-8), as a real config has
 * them: toks_bpe_tables_bytes sizes the premerge table from them */
static uint8_t ALPHA_S[2 * MAXBYTES];
static const uint8_t *ALPHA_P[MAXTOK];
static uint32_t ALPHA_L[MAXTOK];

static uint32_t byte_char(uint32_t b)
{
    if ((b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || b >= 0xAEu) { return b; }
    uint32_t n = 0;
    for (uint32_t c = 0; c < b; c++) { n += !((c >= 0x21u && c <= 0x7Eu) || (c >= 0xA1u && c <= 0xACu) || c >= 0xAEu); }
    return 256u + n;
}

static void m_config(const model *m, toks_config *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    uint32_t o = 0;
    for (uint32_t id = 0; id < m->n_vocab && id < MAXTOK; id++) {
        ALPHA_P[id] = ALPHA_S + o;
        for (uint32_t i = m->off[id]; i < m->off[id + 1]; i++) {
            uint32_t c = byte_char(m->bytes[i]);
            if (c < 0x80u) { ALPHA_S[o++] = (uint8_t)c; } else { ALPHA_S[o++] = (uint8_t)(0xC0u | (c >> 6)); ALPHA_S[o++] = (uint8_t)(0x80u | (c & 63u)); }
        }
        ALPHA_L[id] = (uint32_t)(ALPHA_S + o - ALPHA_P[id]);
    }
    cfg->vocab = ALPHA_P;
    cfg->vocab_len = ALPHA_L;
    cfg->n_vocab = m->n_vocab;
    cfg->n_merges = m->nm;
    cfg->m_left_id = m->ml;
    cfg->m_right_id = m->mr;
    cfg->m_out_id = m->mo;
    cfg->ignore_merges = (uint8_t)m->ignore_merges;
}

/* builds in an arena of exactly toks_bpe_tables_bytes(cfg) (or `size` when not UINT64_MAX) */
static int64_t m_build_sized(model *m, uint64_t size)
{
    if (m->n_vocab == 0) { m->n_vocab = m->n_ids; }
    toks_config cfg;
    m_config(m, &cfg);
    memset(&m->t, 0, sizeof m->t);
    m->t.n_ids = m->n_ids;
    m->t.tok_off = m->off;
    m->t.tok_bytes = m->bytes;
    m->arena_need = toks_bpe_tables_bytes(&cfg);
    uint64_t len = size == UINT64_MAX ? m->arena_need : size;
    guard_aligned_free(m->mem);
    m->mem = guard_aligned_alloc(64, (len + 128 + 63) & ~(uint64_t)63);
    if (m->mem == NULL) { fprintf(stderr, "out of memory\n"); exit(2); }
    toks_arena ar = { m->mem + arena_shift, len, 0 };
    arena_shift ^= 8u;
    int64_t r = toks_bpe_build(&m->t, &ar, &cfg);
    m->arena_used = ar.pos;
    for (uint32_t b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        int64_t id = m_find(m, &c, 1, m->n_vocab);
        m->b2i[b] = id < 0 ? 0u : (uint32_t)id;
    }
    for (uint32_t i = 0; i < m->nm; i++) {
        if (m->ml[i] < MAXTOK && m->mr[i] < MAXTOK) { RANK[m->ml[i]][m->mr[i]] = (int32_t)i; }   /* last wins */
    }
    return r;
}

static int64_t m_build(model *m)
{
    int64_t r = m_build_sized(m, UINT64_MAX);
    CHECK(r == 0, "build failed %" PRId64 " in an arena of exactly %" PRIu64 " bytes", r, m->arena_need);
    return r;
}

/* ===================================================================================== the oracle */

static uint64_t ref_merges(const model *m, const uint8_t *p, uint64_t len, uint32_t *out);

static uint64_t ref_bpe(const model *m, const uint8_t *p, uint64_t len, uint32_t *out)
{
    if (m->ignore_merges) {
        int64_t id = m_find(m, p, len, m->n_vocab);
        if (id >= 0) { out[0] = (uint32_t)id; return 1; }
    }
    return ref_merges(m, p, len, out);
}

/* K6's own answer: ref's, but the merges alone for a piece K6 does not look up whole (layout.h TOKS_TF_PROBE_*):
 * K5's words answer every token of 2..15 bytes that needs the whole-piece rule, and under TOKS_TF_PROBE_ASCII
 * every token over 15 bytes ending in a non-ascii byte is its own merges' answer (check_tables) */
static uint64_t ref_k6(const model *m, const uint8_t *p, uint64_t len, uint32_t *out)
{
    uint32_t f = m->t.flags;
    if ((f & TOKS_TF_PROBE_LONG) == 0u || (len > TOKS_KEY_MAXLEN && ((f & TOKS_TF_PROBE_ASCII) == 0u || p[len - 1] < 0x80u))) {
        return ref_bpe(m, p, len, out);
    }
    return ref_merges(m, p, len, out);
}

/* the merges alone: the lowest rank, leftmost on ties, until none merges */
static uint64_t ref_merges(const model *m, const uint8_t *p, uint64_t len, uint32_t *out)
{
    for (uint64_t i = 0; i < len; i++) { out[i] = m->b2i[p[i]]; }
    uint64_t n = len;
    for (;;) {
        int32_t best = -1;
        uint64_t at = 0;
        for (uint64_t i = 0; i + 1 < n; i++) {
            int32_t k = RANK[out[i]][out[i + 1]];
            if (k >= 0 && (best < 0 || k < best)) { best = k; at = i; }
        }
        if (best < 0) { break; }
        out[at] = m->mo[best];
        memmove(out + at + 1, out + at + 2, (n - at - 2) * 4);
        n--;
    }
    return n;
}

/* ===================================================================================== the kernels */

static uint8_t *WORK;            /* 64-aligned, TOKS_BPE_WORK_BYTES(BIGLEN) */
static uint32_t *OUTB, *REFB;    /* BIGLEN ids each */

static uint64_t run_k6(const model *m, const uint8_t *p, uint64_t len, uint32_t *out, uint64_t *merges)
{
    toks_k6_args a = { p, len, out, WORK, TOKS_BPE_WORK_BYTES(len), 99u, 99u, 0u };
    uint64_t n = K6_BPE(&m->t, &a);
    CHECK(n == a.n_out, "k6 returned %" PRIu64 " but n_out %" PRIu64, n, a.n_out);
    if (merges != NULL) { *merges = a.merges; }
    return n;
}

static int same_ids(const uint32_t *a, uint64_t na, const uint32_t *b, uint64_t nb)
{
    return na == nb && memcmp(a, b, na * 4) == 0;
}

static void show(const char *what, const uint32_t *v, uint64_t n)
{
    fprintf(stderr, "    %s (%" PRIu64 "):", what, n);
    for (uint64_t i = 0; i < n && i < 24; i++) { fprintf(stderr, " %u", v[i]); }
    fprintf(stderr, "%s\n", n > 24 ? " ..." : "");
}

/* K6 on a piece against ref (K6's own answer: ref_k6) */
static void k6_vs_ref(const model *m, const uint8_t *p, uint64_t len, const char *what)
{
    uint64_t ng = run_k6(m, p, len, OUTB, NULL);
    uint64_t nr = ref_k6(m, p, len, REFB);
    int ok = same_ids(OUTB, ng, REFB, nr);
    CHECK(ok, "%s: k6 != ref on a %" PRIu64 "-byte piece (ignore_merges %d, flags %#x)", what, len,
          m->ignore_merges, m->t.flags);
    if (!ok && failures < 10) { show("k6 ", OUTB, ng); show("ref", REFB, nr); }
}

/* the cache tag run_k5 passes (layout.h K5_CACHE_TAG; test_k5_tags moves it like toks_scratch_init does), and the
 * in counters' start: K5WARM picks a warm scratch (>= TOKS_K5_WARM pieces since init: cache first, static answers
 * fill) or a fresh one (static table first, K6 answers fill only) */
static uint64_t K5TAG = 0x2A5A5Au;
static int K5WARM = 1;
static toks_lcache *K5LC;                    /* the long-piece cache run_k5 passes (k5_long.c), or NULL */

/* one K5 call over pieces ends[0..n) of text */
static uint64_t run_k5(const model *m, const uint8_t *text, uint64_t text_len, const uint32_t *ends, uint64_t n,
                       uint64_t start, uint32_t *out, uint64_t room, uint8_t *cache, uint64_t cache_mask,
                       uint8_t *work, uint64_t work_bytes, uint64_t ctr[3])
{
    toks_k5_args a;
    memset(&a, 0xA5, sizeof a);              /* the counters are in-out: K5 adds this call's counts */
    a.text = text; a.len = text_len; a.ends = ends; a.n = n; a.start = start; a.out = out; a.room = room;
    a.cache = cache; a.cache_mask = cache_mask; a.work = work; a.work_bytes = work_bytes; a.cache_tag = K5TAG;
    a.lcache = K5LC;
    if (!K5WARM) { a.hits_static = a.hits_cache = a.misses = (TOKS_K5_WARM - 1u) / 3u; }   /* sum < TOKS_K5_WARM */
    uint64_t c0 = a.hits_static;             /* 0xA5A5... (or the fresh start): what the three held */
    uint64_t r = K5_ENCODE(&m->t, &a);
    CHECK(r == a.n_out, "k5 returned %" PRIu64 " but n_out %" PRIu64, r, a.n_out);
    ctr[0] = a.hits_static - c0; ctr[1] = a.hits_cache - c0; ctr[2] = a.misses - c0;
    CHECK(ctr[0] <= n && ctr[1] <= n && ctr[2] <= n && ctr[0] + ctr[1] + ctr[2] == n,
          "k5 counters +%" PRIu64 " +%" PRIu64 " +%" PRIu64 " != %" PRIu64 " pieces", ctr[0], ctr[1], ctr[2], n);
    return r;
}

/* ===================================================================================== hand cases */

static uint32_t words_count(const model *m, const char *s);

/* ref on `piece` equals spec ("a|bc": token strings) and K6 its own answer (ref_k6: spec unless K5's words answer
 * the piece); K5 says spec cold and warm, in a warm and in a fresh scratch: a K6 answer of <= 4 ids enters the
 * cache in pass 1 and is a cache hit in pass 2; a static answer does the same in a warm scratch and stays a static
 * hit in a fresh one (pass 0 runs without a cache) */
static void expect(const model *m, const char *piece, const char *spec)
{
    uint32_t want[64];
    uint64_t nw = 0;
    for (const char *s = spec; *s != 0;) {
        const char *e = strchr(s, '|');
        size_t l = e != NULL ? (size_t)(e - s) : strlen(s);
        int64_t id = m_find(m, s, l, m->n_vocab);
        if (id < 0) { fprintf(stderr, "test bug: spec token '%.*s'\n", (int)l, s); exit(2); }
        want[nw++] = (uint32_t)id;
        s += l + (e != NULL ? 1 : 0);
    }
    const uint8_t *p = (const uint8_t *)piece;
    uint64_t len = strlen(piece);
    uint32_t got[64], ref[64], own[64], k5o[64 + 4];
    uint64_t ng = run_k6(m, p, len, got, NULL), nr = ref_bpe(m, p, len, ref), no = ref_k6(m, p, len, own);
    CHECK(same_ids(got, ng, own, no), "'%s' (ignore_merges %d, flags %#x): k6 is not its own answer (%s)", piece,
          m->ignore_merges, m->t.flags, spec);
    CHECK(same_ids(ref, nr, want, nw), "'%s' (ignore_merges %d): ref is not %s", piece, m->ignore_merges, spec);
    uint32_t end = (uint32_t)len;
    uint64_t ctr[3];
    int in_words = len >= 2 && len <= 15 && words_count(m, piece) != 0u;
    int keep = K5WARM;
    for (int warm = 0; warm < 2; warm++) {
        _Alignas(64) uint8_t cache[4 * 64];
        memset(cache, 0, sizeof cache);
        K5WARM = warm;
        for (int pass = 0; pass < 3; pass++) {
            uint8_t *c = pass == 0 ? NULL : cache;
            uint64_t n = run_k5(m, p, len, &end, 1, 0, k5o, len + 4, c, 3, WORK, TOKS_BPE_WORK_BYTES(len), ctr);
            CHECK(same_ids(k5o, n, want, nw), "'%s': k5 (pass %d, warm %d) is not %s", piece, pass, warm, spec);
            if (len >= 2 && len <= 15 && (in_words || nw <= 4)) {
                int want_c = in_words ? (pass == 2 && warm ? 1 : 0) : (pass == 2 ? 1 : 2);
                CHECK(ctr[want_c] == 1, "'%s' (pass %d, warm %d): not a %s", piece, pass, warm,
                      want_c == 1 ? "cache hit" : (want_c == 0 ? "static hit" : "K6 run"));
            }
        }
    }
    K5WARM = keep;
}

static uint32_t words_count(const model *m, const char *s)   /* the certified entry's count, 0 when absent */
{
    uint8_t key[16];
    uint32_t val[4];
    bpe_key_make(key, (const uint8_t *)s, (uint32_t)strlen(s));
    return bpe_words_find(&m->t, key, val) ? bpe_val_count(val) : 0u;
}

static void test_trap(void)
{
    /* SPEC §2.7: vocabulary {a, b, c, ab, bc, abc}, merges b+c, a+b, ab+c in that rank order: "abc" is
     * [a, bc], not [abc]. order 0: merged ids rise with rank (ids as rank); order 1: bc (rank 0) has a
     * higher id than ab (rank 1), so the rank2id path. */
    for (int order = 0; order < 2; order++) {
        for (int im = 0; im < 2; im++) {
            m_reset(&M);
            m_add_bytes(&M);
            if (order == 0) { m_add(&M, "bc", 2); m_add(&M, "ab", 2); }
            else            { m_add(&M, "ab", 2); m_add(&M, "bc", 2); }
            m_add(&M, "abc", 3);
            m_merge(&M, "b", "c");
            m_merge(&M, "a", "b");
            m_merge(&M, "ab", "c");
            M.ignore_merges = im;
            if (m_build(&M) != 0) { continue; }
            CHECK(((M.t.flags & TOKS_TF_IDS_AS_RANK) != 0) == (order == 0), "trap order %d: ids-as-rank flag %#x", order, M.t.flags);
            CHECK((M.t.rank2id == NULL) == (order == 0), "trap order %d: rank2id", order);
            uint32_t vid = UINT32_MAX;                 /* without ignore_merges no token is over 15 B: no probe; */
            uint32_t pf = TOKS_TF_IGNORE_MERGES | TOKS_TF_PROBE_LONG | TOKS_TF_PROBE_ASCII;   /* with it, abc */
            CHECK((M.t.flags & pf) == (im ? pf : 0u), "trap: probe flags %#x, im %d", M.t.flags, im);   /* (a|bc) is */
                                                                                            /* a words entry */
            CHECK(bpe_vhash_find(&M.t, (const uint8_t *)"abc", 3, &vid) == (im != 0), "trap: abc in vhash, im %d", im);
            CHECK(bpe_vhash_find(&M.t, (const uint8_t *)"bc", 2, &vid) == (im != 0), "trap: bc in vhash, im %d", im);
            CHECK(M.t.n_merges == 3, "trap: n_merges %u", M.t.n_merges);
            expect(&M, "abc", im ? "abc" : "a|bc");
            expect(&M, "abcabc", "a|bc|a|bc");
            expect(&M, "abcc", "a|bc|c");
            expect(&M, "ab", "ab");
            expect(&M, "bc", "bc");
            expect(&M, "cab", "c|ab");
            CHECK(words_count(&M, "abc") == (im ? 1u : 2u), "trap: the words entry for abc is not K6's answer");
        }
    }
}

static void test_ties(void)
{
    for (int im = 0; im < 2; im++) {
        m_reset(&M);
        m_add_bytes(&M);
        m_add(&M, "aa", 2);
        m_add(&M, "ab", 2);
        m_add(&M, "aaaa", 4);
        m_merge(&M, "a", "a");
        m_merge(&M, "a", "b");
        m_merge(&M, "aa", "aa");
        M.ignore_merges = im;
        if (m_build(&M) != 0) { continue; }
        expect(&M, "aaa", "aa|a");                 /* the leftmost of two equal-prio overlapping pairs */
        expect(&M, "aaab", "aa|ab");               /* rightmost would give a|aa|b */
        expect(&M, "aaaaa", "aaaa|a");
        expect(&M, "aaaaaa", "aaaa|aa");
        expect(&M, "aaaaaaa", "aaaa|aa|a");
        expect(&M, "baaa", "b|aa|a");
        expect(&M, "abab", "ab|ab");
        expect(&M, "aaaa", "aaaa");
    }
}

static void test_unreachable(void)
{
    for (int im = 0; im < 2; im++) {
        m_reset(&M);
        m_add_bytes(&M);
        m_add(&M, "xyz", 3);                       /* in the vocabulary, no merge builds it */
        m_add(&M, "", 0);                          /* a vocabulary entry without a raw form */
        m_add(&M, "pq", 2);
        m_merge(&M, "p", "q");
        M.n_vocab = M.n_ids;
        m_add(&M, "qq", 2);                        /* added-only tokens: never an ignore_merges hit */
        m_add(&M, "xyz", 3);
        m_add(&M, "", 0);
        M.ignore_merges = im;
        if (m_build(&M) != 0) { continue; }
        expect(&M, "xyz", im ? "xyz" : "x|y|z");
        expect(&M, "xyzxyz", "x|y|z|x|y|z");
        expect(&M, "qq", "q|q");
        expect(&M, "pq", "pq");
        expect(&M, "pqq", "pq|q");
        CHECK(words_count(&M, "xyz") == (im ? 1u : 3u), "unreachable: the words entry for xyz is not K6's answer");
        uint32_t id = 0;
        int vf = bpe_vhash_find(&M.t, (const uint8_t *)"xyz", 3, &id);
        CHECK(im ? vf && id == m_id(&M, "xyz") : !vf, "vhash: xyz (bpe x|y|z) in it %d, ignore_merges %d", vf, im);
        CHECK(!bpe_vhash_find(&M.t, (const uint8_t *)"qq", 2, &id), "vhash holds an added-only token");
    }
}

/* the whole-piece probe without ignore_merges reaches only certified tokens over 15 bytes: P = p x 8, Q = q x 8;
 * b+Q ranks before P+b, so Z = P b Q (17 B, a vocabulary token) is [P, bQ], while PQ (16 B) is its own bpe. With
 * q a non-ascii byte, Z ends in one: under ignore_merges K6 then probes such pieces too (no TOKS_TF_PROBE_ASCII) */
static void test_long_probe(void)
{
    for (int na = 0; na < 2; na++) {
        char q = na ? '\xF1' : 'q', P[9], Q[9], X[10], Y[10], Z[18], PQ[17], zs[24], zb[24], zsp[24];
        char q1[2] = { q, 0 }, q2[3] = { q, q, 0 }, q4[5] = { q, q, q, q, 0 };
        memset(P, 'p', 8); memset(Q, q, 8); P[8] = Q[8] = 0;
        snprintf(X, sizeof X, "%sb", P); snprintf(Y, sizeof Y, "b%s", Q); snprintf(Z, sizeof Z, "%sb%s", P, Q);
        snprintf(PQ, sizeof PQ, "%s%s", P, Q);
        snprintf(zs, sizeof zs, "%s|%s", P, Y); snprintf(zb, sizeof zb, "%sb", Z); snprintf(zsp, sizeof zsp, "%s|%s|b", P, Y);
        for (int im = 0; im < 2; im++) {
            m_reset(&M);
            m_add_bytes(&M);
            m_add(&M, "pp", 2); m_add(&M, "pppp", 4); m_add(&M, P, 8);
            m_add(&M, q2, 2); m_add(&M, q4, 4); m_add(&M, Q, 8);
            m_add(&M, Y, 9); m_add(&M, X, 9); m_add(&M, Z, 17); m_add(&M, PQ, 16);
            m_merge(&M, "p", "p"); m_merge(&M, "pp", "pp"); m_merge(&M, "pppp", "pppp");
            m_merge(&M, q1, q1); m_merge(&M, q2, q2); m_merge(&M, q4, q4);
            m_merge(&M, "b", Q); m_merge(&M, P, "b"); m_merge(&M, X, Q); m_merge(&M, P, Q);
            M.ignore_merges = im;
            if (m_build(&M) != 0) { continue; }
            uint32_t id = 0, pf = TOKS_TF_IGNORE_MERGES | TOKS_TF_PROBE_LONG | TOKS_TF_PROBE_ASCII;
            CHECK((M.t.flags & pf) == (im ? (na ? pf & ~TOKS_TF_PROBE_ASCII : pf) : na ? 0u : pf),
                  "long probe: flags %#x, im %d, non-ascii q %d", M.t.flags, im, na);
            CHECK(bpe_vhash_find(&M.t, (const uint8_t *)Z, 17, &id) == (im != 0), "long probe: Z in vhash, im %d", im);
            int pq = bpe_vhash_find(&M.t, (const uint8_t *)PQ, 16, &id);
            CHECK(pq == (im || !na) && (!pq || id == m_id(&M, PQ)), "long probe: PQ in vhash %d, im %d, non-ascii q %d", pq, im, na);
            CHECK(bpe_vhash_find(&M.t, (const uint8_t *)P, 8, &id) == (im != 0), "long probe: P (8 B) in vhash, im %d", im);
            expect(&M, Z, im ? Z : zs);
            expect(&M, PQ, PQ);
            expect(&M, zb, zsp);
            expect(&M, P, P);
        }
    }
}

static void check_tables(const model *m, uint64_t *seed);

/* ignore_merges with the words table at its load cap (510 keys: 256 buckets, 435 entries): the 30 tokens no merge
 * builds come last in id order, so the fill reaches them past the cap and seats each in a free way, in the place
 * of an entry that moves to its other bucket, or of one whose own bytes bpe back to it (bpe_build.c words_seat; this
 * table: 19, 5 and 6 of them); K6 then leaves the whole-piece rule of pieces of 2..15 bytes to K5's words */
static void test_seat(void)
{
    static const char AL[] = "abcdefghijklmnopqrstuv";
    uint64_t seed = 0x73656174u;
    m_reset(&M);
    m_add_bytes(&M);
    char s[3] = { 0 }, l[2] = { 0 }, r[2] = { 0 };
    for (uint32_t i = 0; i < 480; i++) {                   /* 480 two-letter tokens, each one merge of two bytes */
        s[0] = l[0] = AL[i / 22]; s[1] = r[0] = AL[i % 22];
        m_add(&M, s, 2);
        m_merge(&M, l, r);
    }
    char un[30][4];
    for (uint32_t i = 0; i < 30; i++) {                    /* 30 three-letter tokens no merge builds: xy|w */
        un[i][0] = AL[i % 22]; un[i][1] = AL[(i / 22) * 5 + 1]; un[i][2] = 'w'; un[i][3] = 0;
        m_add(&M, un[i], 3);
    }
    M.ignore_merges = 1;
    if (m_build(&M) != 0) { CHECK(0, "seat: build"); return; }
    CHECK((M.t.flags & TOKS_TF_PROBE_LONG) != 0, "seat: flags %#x", M.t.flags);
    check_tables(&M, &seed);
    for (uint32_t i = 0; i < 30; i++) { expect(&M, un[i], un[i]); }
}

static void test_duplicates(void)
{
    /* a+b, b+c, a+b: the pair a+b keeps rank 2, so "abc" is [a, bc]; without the duplicate it is [ab, c].
     * order 0: the survivors' merged ids are bc > ab (rank2id); order 1: bc < ab (ids as rank). */
    for (int order = 0; order < 2; order++) {
        for (int dup = 0; dup < 2; dup++) {
            m_reset(&M);
            m_add_bytes(&M);
            if (order == 0) { m_add(&M, "ab", 2); m_add(&M, "bc", 2); }
            else            { m_add(&M, "bc", 2); m_add(&M, "ab", 2); }
            m_merge(&M, "a", "b");
            m_merge(&M, "b", "c");
            if (dup) { m_merge(&M, "a", "b"); }
            if (m_build(&M) != 0) { continue; }
            int as_rank = (M.t.flags & TOKS_TF_IDS_AS_RANK) != 0;
            CHECK(as_rank == (dup ? order == 1 : order == 0), "duplicates order %d dup %d: ids-as-rank %d", order, dup, as_rank);
            expect(&M, "abc", dup ? "a|bc" : "ab|c");
            expect(&M, "abcabc", dup ? "a|bc|a|bc" : "ab|c|ab|c");
        }
    }
    /* several merges to one string: abc from ab+c and from a+bc (merged ids repeat: rank2id) */
    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2); m_add(&M, "bc", 2); m_add(&M, "abc", 3);
    m_merge(&M, "b", "c"); m_merge(&M, "a", "b"); m_merge(&M, "a", "bc"); m_merge(&M, "ab", "c");
    if (m_build(&M) == 0) {
        CHECK((M.t.flags & TOKS_TF_IDS_AS_RANK) == 0, "repeated merged ids must take the rank2id path");
        expect(&M, "abc", "abc");
        expect(&M, "abcbc", "abc|bc");
    }
    /* merged ids that never fall but repeat (ab, bc, abc, abc) are not strictly rising either */
    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2); m_add(&M, "bc", 2); m_add(&M, "abc", 3);
    m_merge(&M, "a", "b"); m_merge(&M, "b", "c"); m_merge(&M, "ab", "c"); m_merge(&M, "a", "bc");
    if (m_build(&M) == 0) {
        CHECK((M.t.flags & TOKS_TF_IDS_AS_RANK) == 0, "equal merged ids must take the rank2id path");
        expect(&M, "abc", "abc");
    }
}

/* an arena of exactly toks_bpe_tables_bytes when the estimate has almost no rounding slack: 1023
 * model tokens (2 n + 1 = 2047 vhash slots -> 2048; 767 two-byte keys -> 512 words buckets, as for
 * 1023) and 767 merges on the rank2id path (merged ids fall with rank) */
static void test_arena_tight(void)
{
    m_reset(&M);
    m_add_bytes(&M);
    for (uint32_t i = 0; i < 767; i++) { uint8_t s[2] = { (uint8_t)(i / 26 + 'A'), (uint8_t)(i % 26 + 'a') }; m_add(&M, s, 2); }
    for (uint32_t i = 767; i-- > 0;) {
        M.ml[M.nm] = (uint32_t)(i / 26 + 'A');              /* m_add_bytes: byte b is id b */
        M.mr[M.nm] = (uint32_t)(i % 26 + 'a');
        M.mo[M.nm] = 256 + i;
        M.nm++;
    }
    if (m_build(&M) != 0) { return; }
    CHECK((M.t.flags & TOKS_TF_IDS_AS_RANK) == 0 && M.n_vocab == 1023, "tight model: not on the rank2id path");
    CHECK(M.arena_need - M.arena_used < 1024, "tight model: %" PRIu64 " of %" PRIu64 " arena bytes used", M.arena_used, M.arena_need);
    expect(&M, "Aa", "Aa");
    expect(&M, "AaBb", "Aa|Bb");
}

static void test_one_byte_and_zero_merges(void)
{
    for (int im = 0; im < 2; im++) {
        m_reset(&M);
        /* byte tokens in a scrambled id order, as real vocabularies have them */
        for (uint32_t i = 0; i < 256; i++) { uint8_t c = (uint8_t)(i * 167u + 13u); m_add(&M, &c, 1); }
        M.ignore_merges = im;
        if (m_build(&M) != 0) { continue; }
        CHECK(M.t.n_merges == 0 && (M.t.flags & TOKS_TF_IDS_AS_RANK) != 0 && M.t.rank2id == NULL, "zero merges: flags %#x", M.t.flags);
        expect(&M, "hello", "h|e|l|l|o");
        uint8_t text[256];
        uint32_t ends[256], out[256 + 4];
        for (uint32_t b = 0; b < 256; b++) {
            text[b] = (uint8_t)b;
            ends[b] = b + 1;
            uint32_t got;
            uint64_t n = run_k6(&M, text + b, 1, &got, NULL);
            CHECK(n == 1 && got == M.b2i[b] && M.t.byte2id[b] == M.b2i[b], "one-byte piece %u", b);
        }
        uint64_t ctr[3];
        uint64_t n = run_k5(&M, text, 256, ends, 256, 0, out, 256 + 4, NULL, 0, WORK, TOKS_BPE_WORK_BYTES(1), ctr);
        CHECK(n == 256 && same_ids(out, 256, M.b2i, 256) && ctr[0] == 256, "k5 over 256 one-byte pieces");
    }
}

static void test_build_errors(void)
{
    m_reset(&M);
    for (uint32_t b = 1; b < 256; b++) { uint8_t c = (uint8_t)b; m_add(&M, &c, 1); }   /* no token for byte 0 */
    CHECK(m_build_sized(&M, UINT64_MAX) == TOKS_E_FORMAT, "a missing byte token must fail the build");

    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2);
    M.n_vocab = M.n_ids - 1;                   /* "ab" is added-only: a merge may not produce it */
    M.ml[0] = m_id(&M, "a"); M.mr[0] = m_id(&M, "b"); M.mo[0] = M.n_ids - 1; M.nm = 1;
    CHECK(m_build_sized(&M, UINT64_MAX) == TOKS_E_FORMAT, "a merge outside the model vocabulary must fail");

    m_reset(&M);
    m_add_bytes(&M);
    CHECK(m_build_sized(&M, 0) == TOKS_E_NOMEM, "an empty arena must fail with TOKS_E_NOMEM");
    CHECK(m_build_sized(&M, 200000) == TOKS_E_NOMEM, "a short arena must fail with TOKS_E_NOMEM");
    M.n_vocab = M.n_ids + 1;
    CHECK(m_build_sized(&M, UINT64_MAX) == TOKS_E_FORMAT, "n_vocab > n_ids must fail");
}

static void test_long_piece(void)
{
    /* "a" x 2^20 under merges a+a, aa+aa, ... up to a^1024: every round merges leftmost-first, so the
     * answer is 1024 x a^1024 after 2^20 - 1024 merges. A quadratic K6 would take hours. */
    m_reset(&M);
    m_add_bytes(&M);
    static uint8_t big[BIGLEN];
    memset(big, 'a', sizeof big);
    for (uint32_t k = 2; k <= 1024; k *= 2) { m_add(&M, big, k); }
    for (uint32_t k = 1; k < 1024; k *= 2) {
        M.ml[M.nm] = M.mr[M.nm] = (uint32_t)m_find(&M, big, k, M.n_ids);
        M.mo[M.nm] = (uint32_t)m_find(&M, big, 2 * k, M.n_ids);
        M.nm++;
    }
    if (m_build(&M) != 0) { return; }
    uint32_t top = (uint32_t)m_find(&M, big, 1024, M.n_ids);
    uint64_t merges = 0;
    clock_t c0 = clock();
    uint64_t n = run_k6(&M, big, BIGLEN, OUTB, &merges);
    double ms = 1000.0 * (double)(clock() - c0) / CLOCKS_PER_SEC;
    int ok = n == 1024 && merges == BIGLEN - 1024;
    for (uint64_t i = 0; ok && i < n; i++) { ok = OUTB[i] == top; }
    CHECK(ok, "1 MiB of 'a': %" PRIu64 " ids after %" PRIu64 " merges", n, merges);
    CHECK(ms < 10000.0, "1 MiB piece took %.0f ms (quadratic?)", ms);
    printf("1 MiB piece: %" PRIu64 " ids, %" PRIu64 " merges, %.1f ms\n", n, merges, ms);
}

/* the dynamic cache's counters and fill rule: new entry to way 0, old way 0 to way 1 */
/* runs (kernels.md 5, k6_neon.S's fast path): a piece whose bytes after the first are one byte c, when neither
 * (first, c) nor (c, c) merges, is its byte ids. Eight models -- c c merges or not, '\n' c merges or not,
 * ignore_merges or not; three spaces a token no merge makes -- every c, first bytes c and nine others, 2..40 bytes,
 * each also with one byte after the first off (somewhere; the one before the last): K6 against ref_k6, the piece
 * flush against a no-access page at its end and then at its start, the ids out and the work area flush against one
 * at their end. */
static void test_runs(void)
{
    enum { RL = 40 };
    static const uint8_t XS[] = { 0x00, '\n', ' ', '\t', 'a', '-', 0x80, 0xC3, 0xE4 };
    const uint64_t wmax = (TOKS_BPE_WORK_BYTES(RL) + 63) & ~(uint64_t)63;
    guard_buf gp = {0}, gq = {0}, go = {0}, gw = {0};
    uint8_t *pe = guard_alloc(&gp, RL, GUARD_END, 0), *ps = guard_alloc(&gq, RL, GUARD_START, 0);
    uint8_t *oe = guard_alloc(&go, 4u * RL, GUARD_END, 0), *we = guard_alloc(&gw, wmax, GUARD_END, 0);
    if (!pe || !ps || !oe || !we) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    for (int v = 0; v < 8; v++) {
        m_reset(&M);
        m_add_bytes(&M);
        m_add(&M, "  ", 2);
        m_add(&M, "   ", 3);
        m_add(&M, "    ", 4);
        m_add(&M, "\n ", 2);
        m_add(&M, "--", 2);
        if (v & 1) { m_merge(&M, " ", " "); m_merge(&M, "  ", "  "); m_merge(&M, "-", "-"); }
        if (v & 2) { m_merge(&M, "\n", " "); }
        M.ignore_merges = (v & 4) != 0;
        if (m_build(&M) != 0) { continue; }
        uint8_t p[RL];
        for (uint32_t c = 0; c < 256; c++) {
            for (size_t xi = 0; xi <= sizeof XS; xi++) {
                for (uint64_t len = 2; len <= RL; len++) {
                    for (int off = 0; off < 3; off++) {   /* a run; one byte off somewhere; the one before the last off */
                        p[0] = xi == sizeof XS ? (uint8_t)c : XS[xi];
                        memset(p + 1, (int)c, len - 1);
                        if (off == 1) { p[1u + (c * 13u + (uint32_t)xi * 5u + len * 3u) % (len - 1u)] = (uint8_t)(c + 1u); }
                        if (off == 2) { p[len > 2 ? len - 2 : 1] = (uint8_t)(c + 1u); }
                        uint64_t nr = ref_k6(&M, p, len, REFB), wb = (TOKS_BPE_WORK_BYTES(len) + 63) & ~(uint64_t)63;
                        uint32_t *out = (uint32_t *)(void *)(oe + 4u * (RL - len));
                        for (int k = 0; k < 2; k++) {
                            uint8_t *q = k == 0 ? pe + (RL - len) : ps;
                            memcpy(q, p, len);
                            toks_k6_args a = { q, len, out, we + (wmax - wb), TOKS_BPE_WORK_BYTES(len), 0u, 0u, 0u };
                            uint64_t n = K6_BPE(&M.t, &a);
                            int ok = n == a.n_out && same_ids(out, n, REFB, nr);
                            CHECK(ok, "%s: k6 != ref on a %" PRIu64 "-byte piece %02x %02x.. (model %d, flush %s)",
                                  off != 0 ? "not a run" : "run", len, p[0], c, v, k == 0 ? "end" : "start");
                        }
                    }
                }
            }
        }
    }
    guard_free(&gp); guard_free(&gq); guard_free(&go); guard_free(&gw);
}

static void test_k5_cache_mode(int warm)
{
    K5WARM = warm;
    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2); m_add(&M, "bc", 2); m_add(&M, "abc", 3);
    m_merge(&M, "b", "c"); m_merge(&M, "a", "b"); m_merge(&M, "ab", "c");
    if (m_build(&M) != 0) { return; }
    static const char *P[] = { "abc", "abc", "ca", "ca", "x", "abcabcabcabcabcabc", "abcabcabcabcabcabc" };
    uint8_t text[256];
    uint32_t ends[16], out[256], want[256];
    uint64_t len = 0, nw = 0;
    for (uint32_t i = 0; i < 7; i++) {
        uint64_t l = strlen(P[i]);
        memcpy(text + len, P[i], l);
        nw += ref_bpe(&M, text + len, l, want + nw);
        len += l;
        ends[i] = (uint32_t)len;
    }
    _Alignas(64) uint8_t cache[4 * 64];
    memset(cache, 0, sizeof cache);
    uint64_t ctr[3];
    uint64_t n = run_k5(&M, text, len, ends, 7, 0, out, len + 4, cache, 3, WORK, TOKS_BPE_WORK_BYTES(18), ctr);
    CHECK(same_ids(out, n, want, nw), "k5 cache case: ids");
    /* warm: abc a static hit that fills the cache, then a cache hit; fresh: abc static twice; ca a K6 run, then a
     * cache hit; x static (one byte); the 18-byte pieces K6 runs */
    CHECK(ctr[0] == (warm ? 2u : 3u) && ctr[1] == (warm ? 2u : 1u) && ctr[2] == 3,
          "k5 cache case (warm %d): static/cache/miss %" PRIu64 "/%" PRIu64 "/%" PRIu64, warm, ctr[0], ctr[1], ctr[2]);
    n = run_k5(&M, text, len, ends, 7, 0, out, len + 4, NULL, 0, WORK, TOKS_BPE_WORK_BYTES(18), ctr);
    CHECK(same_ids(out, n, want, nw), "k5 no-cache case: ids");
    CHECK(ctr[0] == 3 && ctr[1] == 0 && ctr[2] == 4, "k5 no cache: static/cache/miss %" PRIu64 "/%" PRIu64 "/%" PRIu64 " != 3/0/4",
          ctr[0], ctr[1], ctr[2]);

    /* one bucket: ca, cb, cc fill (way 0 = cc, way 1 = cb); cb hits way 1 and fills nothing; ca was
     * dropped: a K6 run that refills (way 0 = ca, way 1 = cc); ab is a static entry: warm, answered from the
     * table and filled (way 0 = ab, way 1 = ca), then a cache hit; fresh, answered from the table twice and
     * never filled (the bucket stays ca, cc); ca then hits */
    static const char *Q[] = { "ca", "cb", "cc", "cb", "ca", "ab", "ab", "ca" };
    static const uint64_t hitw[] = { 0, 0, 0, 1, 0, 0, 1, 1 }, hitf[] = { 0, 0, 0, 1, 0, 0, 0, 1 };
    static const uint64_t staw[] = { 0, 0, 0, 0, 0, 1, 0, 0 }, staf[] = { 0, 0, 0, 0, 0, 1, 1, 0 };
    static const char *const W0w[] = { "ca", "cb", "cc", "cc", "ca", "ab", "ab", "ab" };
    static const char *const W1w[] = { NULL, "ca", "cb", "cb", "cc", "ca", "ca", "ca" };
    static const char *const W0f[] = { "ca", "cb", "cc", "cc", "ca", "ca", "ca", "ca" };
    static const char *const W1f[] = { NULL, "ca", "cb", "cb", "cc", "cc", "cc", "cc" };
    const uint64_t *hit = warm ? hitw : hitf, *sta = warm ? staw : staf;
    const char *const *W0 = warm ? W0w : W0f, *const *W1 = warm ? W1w : W1f;
    _Alignas(64) uint8_t one[64];
    memset(one, 0, sizeof one);
    for (uint32_t i = 0; i < 8; i++) {
        uint32_t e = 2;
        n = run_k5(&M, (const uint8_t *)Q[i], 2, &e, 1, 0, out, 6, one, 0, WORK, TOKS_BPE_WORK_BYTES(2), ctr);
        uint64_t nr = ref_bpe(&M, (const uint8_t *)Q[i], 2, want);
        CHECK(same_ids(out, n, want, nr) && ctr[1] == hit[i] && ctr[0] == sta[i],
              "fill rule step %u (%s, warm %d): static/cache hits %" PRIu64 "/%" PRIu64, i, Q[i], warm, ctr[0], ctr[1]);
        if (W1[i] != NULL) {
            const char *w0 = W0[i], *w1 = W1[i];
            uint8_t k0[16], k1[16];
            uint32_t v0[4], v1[4], ids[4];
            bpe_key_make(k0, (const uint8_t *)w0, 2);
            bpe_key_make(k1, (const uint8_t *)w1, 2);
            bpe_val_pack_tag(v0, ids, (uint32_t)ref_bpe(&M, (const uint8_t *)w0, 2, ids), toks_tag_word(K5TAG));
            bpe_val_pack_tag(v1, ids, (uint32_t)ref_bpe(&M, (const uint8_t *)w1, 2, ids), toks_tag_word(K5TAG));
            CHECK(memcmp(one, k0, 16) == 0 && memcmp(one + 16, k1, 16) == 0 && memcmp(one + 32, v0, 16) == 0
                  && memcmp(one + 48, v1, 16) == 0, "fill rule step %u (warm %d): way 0 = %s, way 1 = %s", i, warm, w0, w1);
        }
    }
}

static void test_k5_cache(void)
{
    int keep = K5WARM;
    test_k5_cache_mode(1);
    test_k5_cache_mode(0);
    K5WARM = keep;
}

/* the long-piece cache (k5_long.c): a warm K5 with an lcache answers repeated long pieces (> 15 bytes) and
 * repeated non-ascii pieces of > 4 ids from it, exactly; a full arena starts the next generation; whatever the
 * cache memory holds (slots past the filled arena, counts above the length, other bytes), nothing else answers */
static void test_k5_long(void)
{
    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2); m_add(&M, "bc", 2); m_add(&M, "abc", 3);
    m_merge(&M, "b", "c"); m_merge(&M, "a", "b"); m_merge(&M, "ab", "c");
    if (m_build(&M) != 0) { return; }
    static const char *P[] = { "abcabcabcabcabcabc", "xyzxyzxyzxyzxyzxyzxyz", "abcabcabcabcabcabc",
                               "\xc3\xa9\xc3\xa9\xc3\xa9", "\xc3\xa9\xc3\xa9\xc3\xa9", "ab" };
    uint8_t text[256];
    uint32_t ends[8], out[256], want[256];
    uint64_t len = 0, nw = 0;
    for (uint32_t i = 0; i < 6; i++) {
        uint64_t l = strlen(P[i]);
        memcpy(text + len, P[i], l);
        nw += ref_bpe(&M, text + len, l, want + nw);
        len += l;
        ends[i] = (uint32_t)len;
    }
    static _Alignas(64) uint8_t cache[4 * 64], lb[4 * 64], la[1024];
    memset(cache, 0, sizeof cache);
    memset(lb, 0, sizeof lb);
    toks_lcache lc = { lb, la, 3, sizeof la, 0, 7, 0, 0 };
    uint64_t ctr[3], n;
    K5LC = &lc;
    for (int pass = 0; pass < 2; pass++) {       /* pass 0: 3 stored, 2 hits (pieces 2, 4); pass 1: 5 hits */
        n = run_k5(&M, text, len, ends, 6, 0, out, len + 4, cache, 3, WORK, TOKS_BPE_WORK_BYTES(21), ctr);
        CHECK(same_ids(out, n, want, nw), "k5 long cache pass %d: ids", pass);
        CHECK(lc.hits == (pass ? 7u : 2u) && lc.misses == 3u && lc.gen == 7u,
              "long cache pass %d: hits %" PRIu64 " misses %" PRIu64 " gen %" PRIu64, pass, lc.hits, lc.misses, lc.gen);
    }
    uint64_t pos = lc.pos;
    for (int bad = 0; bad < 3; bad++) {          /* every slot past the fill, every count huge, every copy changed */
        for (uint32_t w = 0; w < 16; w++) {
            uint32_t *s = (uint32_t *)(void *)lb + 4u * w;
            if (s[3] != 7u) { continue; }
            if (bad == 0) { s[2] += (uint32_t)pos; }
            if (bad == 1) { memcpy(la + s[2], &(uint32_t){ 1000u }, 4); }
            if (bad == 2) { la[s[2] + 4u] ^= 1u; }
        }
        uint64_t h0 = lc.hits;
        lc.pos = pos;
        n = run_k5(&M, text, len, ends, 6, 0, out, len + 4, cache, 3, WORK, TOKS_BPE_WORK_BYTES(21), ctr);
        CHECK(same_ids(out, n, want, nw), "k5 long cache, bad memory %d: ids", bad);
        CHECK(lc.hits - h0 == 2u, "long cache, bad memory %d: %" PRIu64 " hits (only this call's new entries)", bad,
              lc.hits - h0);
        memset(lb, 0, sizeof lb);                /* the next case starts from an empty cache */
        lc.pos = 0;
        n = run_k5(&M, text, len, ends, 6, 0, out, len + 4, cache, 3, WORK, TOKS_BPE_WORK_BYTES(21), ctr);
        pos = lc.pos;
    }
    lc.arena_bytes = 128;                        /* the 21-byte piece needs 128: each store starts a generation */
    lc.pos = 0;
    memset(lb, 0, sizeof lb);
    n = run_k5(&M, text, len, ends, 6, 0, out, len + 4, cache, 3, WORK, TOKS_BPE_WORK_BYTES(21), ctr);
    CHECK(same_ids(out, n, want, nw), "k5 long cache, small arena: ids");
    CHECK(lc.gen > 7u, "long cache, small arena: gen %" PRIu64 " never moved", lc.gen);
    K5LC = NULL;
}

/* the cache tag (layout.h, kernels.md §7): an entry answers only under the tag it was filled with. One bucket:
 * ca fills under tag A and hits under A; under B it is a miss that refills way 0 (way 1 keeps the A entry); back
 * under A, way 0 (the B entry) is chosen by key and its tag says miss: a K6 run, refilled under A; only the low
 * TOKS_TAG_BITS bits of the tag count; a tag's bits never reach out (emit masks them) */
static void test_k5_tags(void)
{
    m_reset(&M);
    m_add_bytes(&M);
    m_add(&M, "ab", 2); m_add(&M, "bc", 2); m_add(&M, "abc", 3);
    m_merge(&M, "b", "c"); m_merge(&M, "a", "b"); m_merge(&M, "ab", "c");
    if (m_build(&M) != 0) { return; }
    static const uint64_t TAG[] = { 1, 1, 2, 2, 1, 1 + (1ull << TOKS_TAG_BITS), TOKS_TAG_MAX, TOKS_TAG_MAX, 0x2A5A5A };
    static const uint64_t hit[] = { 0, 1, 0, 1, 0, 1, 0, 1, 0 };
    _Alignas(64) uint8_t one[64];
    memset(one, 0, sizeof one);
    uint64_t keep = K5TAG;
    for (uint32_t i = 0; i < sizeof TAG / sizeof TAG[0]; i++) {
        uint32_t e = 2, out[8], want[8];
        uint64_t ctr[3];
        K5TAG = TAG[i];
        uint64_t n = run_k5(&M, (const uint8_t *)"ca", 2, &e, 1, 0, out, 6, one, 0, WORK, TOKS_BPE_WORK_BYTES(2), ctr);
        uint64_t nr = ref_bpe(&M, (const uint8_t *)"ca", 2, want);
        CHECK(same_ids(out, n, want, nr) && ctr[1] == hit[i] && ctr[2] == 1u - hit[i],
              "tag step %u (tag %#" PRIx64 "): ids or cache hits %" PRIu64 " / misses %" PRIu64, i, TAG[i], ctr[1], ctr[2]);
        uint64_t g;
        memcpy(&g, one + 40, 8);                 /* way 0's val, high 8 bytes: the tag of the entry last filled */
        CHECK((g & TOKS_TAG_MASK64) == toks_tag_word(TAG[i]) && (g & ~TOKS_TAG_MASK64) == 0u,
              "tag step %u: way 0's tag bits %#" PRIx64 " for tag %#" PRIx64, i, g, TAG[i]);
    }
    CHECK(toks_tag_word(TOKS_TAG_MAX) == TOKS_TAG_MASK64 && toks_tag_word(0) == 0u &&
          toks_tag_word(1ull << TOKS_TAG_BITS) == 0u, "toks_tag_word's bits");
    K5TAG = keep;
}

/* the in-place key: for every segment of 1..48 bytes, flush against a guard page at its end and then
 * at its start, every piece [s, s + n) of 2..15 bytes inside it reads to exactly bpe_key_make's key
 * (a byte read past the segment's end or before its start faults) */
static void test_key_in_place(void)
{
    uint64_t seed = 0x6b657921u;
    uint8_t src[48];
    for (int where = 0; where < 2; where++) {
        for (uint64_t tlen = 1; tlen <= 48u; tlen++) {
            guard_buf g = {0};
            uint8_t *text = guard_alloc(&g, (size_t)tlen, where == 0 ? GUARD_END : GUARD_START, 0);
            if (text == NULL) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
            for (uint64_t i = 0; i < tlen; i++) { src[i] = (uint8_t)(1u + guard_rng(&seed) % 255u); }
            memcpy(text, src, (size_t)tlen);
            for (uint64_t s = 0; s < tlen; s++) {
                for (uint64_t n = 2; n <= 15u && s + n <= tlen; n++) {
                    uint8_t key[16];
                    bpe_key_make(key, src + s, (uint32_t)n);
                    bpe_key k = bpe_key_at(text, tlen, s, n);
                    uint64_t lo, hi;
                    memcpy(&lo, key, 8);
                    memcpy(&hi, key + 8, 8);
                    CHECK(k.lo == lo && k.hi == hi && bpe_key_hash(k) == toks_key_hash(TOKS_HSEED, key),
                          "in-place key: segment %" PRIu64 " piece [%" PRIu64 ", +%" PRIu64 ") (%s guard)", tlen, s, n,
                          where == 0 ? "end" : "start");
                }
            }
            guard_free(&g);
        }
    }
}

/* ===================================================================================== random models */

static uint64_t rnd(uint64_t *s, uint64_t n) { return guard_rng(s) % n; }

enum {
    G_SHUFFLE_ALL = 1, G_SHUFFLE_TOP = 2, G_DUPS = 4, G_MULTI = 8, G_REORDER = 16, G_EXTRAS = 32, G_IGNORE = 64, G_CHARS = 128,
};

typedef struct { uint8_t s[16]; uint32_t len; } gstr;
static gstr S[MAXTOK];
static uint32_t NS, PM[MAXTOK], INV[MAXTOK];
static const uint8_t ALPHA[] = { 'a', 'b', 'c', 'd', ' ', 0xC3, 0xA9, 0xE4, 0xB8, 0xAD };   /* e-acute, U+4E2D */

static int64_t s_find(const uint8_t *s, uint32_t len)
{
    for (uint32_t i = 0; i < NS; i++) {
        if (S[i].len == len && memcmp(S[i].s, s, len) == 0) { return i; }
    }
    return -1;
}

static void gen_model(model *m, uint64_t *seed, uint32_t mode)
{
    m_reset(m);
    NS = 0;
    for (uint32_t b = 0; b < 256; b++) { S[NS].s[0] = (uint8_t)b; S[NS].len = 1; NS++; }
    uint32_t pool[MAXTOK], np = 0;
    for (uint32_t i = 0; i < sizeof ALPHA; i++) { pool[np++] = ALPHA[i]; }
    uint32_t L[MAXMERGE], R[MAXMERGE], O[MAXMERGE], nm = 0;
    if (mode & G_CHARS) {                       /* e-acute and U+4E2D as tokens from the start: premerge candidates */
        static const uint8_t CH[3][3] = { { 0xC3, 0xA9, 0 }, { 0xE4, 0xB8, 0 }, { 0xE4, 0xB8, 0xAD } };
        for (uint32_t c = 0; c < 3; c++) {
            uint32_t len = CH[c][2] ? 3 : 2;
            memcpy(S[NS].s, CH[c], len);
            S[NS].len = len;
            L[nm] = c == 2 ? NS - 1 : CH[c][0]; R[nm] = CH[c][len - 1]; O[nm] = NS; nm++;
            pool[np++] = NS++;
        }
    }
    uint32_t target = 10 + (uint32_t)rnd(seed, 300);
    for (uint32_t attempt = 0; attempt < 8 * target && nm < target; attempt++) {
        uint32_t l = pool[rnd(seed, np)], r = pool[rnd(seed, np)];
        if (S[l].len + S[r].len > 12) { continue; }
        uint8_t o[16];
        memcpy(o, S[l].s, S[l].len);
        memcpy(o + S[l].len, S[r].s, S[r].len);
        int64_t f = s_find(o, S[l].len + S[r].len);
        if (f >= 0) {
            int known = 0;
            for (uint32_t i = 0; i < nm; i++) { known |= L[i] == l && R[i] == r; }
            if (known ? !(mode & G_DUPS) : !(mode & G_MULTI)) { continue; }
            L[nm] = l; R[nm] = r; O[nm] = (uint32_t)f; nm++;
            continue;
        }
        memcpy(S[NS].s, o, 16);
        S[NS].len = S[l].len + S[r].len;
        pool[np++] = NS;
        L[nm] = l; R[nm] = r; O[nm] = NS; nm++;
        NS++;
    }
    if (mode & G_DUPS) {                        /* exact repeats of earlier merges, anywhere later */
        uint32_t k = nm / 3;
        for (uint32_t i = 0; i < k && nm < MAXMERGE; i++) {
            uint32_t j = (uint32_t)rnd(seed, nm);
            L[nm] = L[j]; R[nm] = R[j]; O[nm] = O[j]; nm++;
        }
    }
    if (mode & G_REORDER) {                     /* a merge may be ranked before those that build its parts */
        for (uint32_t i = 0; i < nm / 4 + 1 && nm > 1; i++) {
            uint32_t x = (uint32_t)rnd(seed, nm), y = (uint32_t)rnd(seed, nm), t;
            t = L[x]; L[x] = L[y]; L[y] = t;
            t = R[x]; R[x] = R[y]; R[y] = t;
            t = O[x]; O[x] = O[y]; O[y] = t;
        }
    }
    uint32_t n_extra_empty = 0;
    if (mode & G_EXTRAS) {                      /* strings no merge builds, and entries without a raw form */
        uint32_t k = 1 + (uint32_t)rnd(seed, 20);
        for (uint32_t i = 0; i < k; i++) {
            uint8_t s[16];
            uint32_t len = 2 + (uint32_t)rnd(seed, 7);
            for (uint32_t j = 0; j < len; j++) { s[j] = ALPHA[rnd(seed, sizeof ALPHA)]; }
            if (s_find(s, len) >= 0) { continue; }
            memcpy(S[NS].s, s, len);
            S[NS].len = len;
            NS++;
        }
        n_extra_empty = (uint32_t)rnd(seed, 3);
        for (uint32_t i = 0; i < n_extra_empty; i++) { S[NS].len = 0; NS++; }
    }
    for (uint32_t i = 0; i < NS; i++) { PM[i] = i; }
    uint32_t lo = (mode & G_SHUFFLE_ALL) ? 0 : 256, shuffle = (mode & (G_SHUFFLE_ALL | G_SHUFFLE_TOP)) != 0;
    for (uint32_t i = NS; shuffle && i > lo + 1; i--) {
        uint32_t j = lo + (uint32_t)rnd(seed, i - lo), t = PM[i - 1];
        PM[i - 1] = PM[j]; PM[j] = t;
    }
    for (uint32_t i = 0; i < NS; i++) { INV[PM[i]] = i; }
    for (uint32_t id = 0; id < NS; id++) { m_add(m, S[INV[id]].s, S[INV[id]].len); }
    for (uint32_t i = 0; i < nm; i++) { m->ml[i] = PM[L[i]]; m->mr[i] = PM[R[i]]; m->mo[i] = PM[O[i]]; }
    m->nm = nm;
    m->n_vocab = m->n_ids;
    if (mode & G_EXTRAS) {                      /* added-only tokens, one of them a model string */
        uint32_t k = (uint32_t)rnd(seed, 4);
        for (uint32_t i = 0; i < k; i++) {
            uint8_t s[16];
            uint32_t len = 2 + (uint32_t)rnd(seed, 5);
            for (uint32_t j = 0; j < len; j++) { s[j] = ALPHA[rnd(seed, sizeof ALPHA)]; }
            m_add(m, s, len);
        }
        uint32_t any = 256 + (uint32_t)rnd(seed, NS - 256 + 1) - 1;
        if (any < NS && S[any].len > 0) { m_add(m, S[any].s, S[any].len); }
    }
    m->ignore_merges = (mode & G_IGNORE) != 0;
}

/* a test piece for model m into p (returns its length) */
static uint64_t gen_piece(const model *m, uint64_t *seed, uint8_t *p, uint64_t maxlen)
{
    uint64_t kind = rnd(seed, 10), len = 0;
    if (kind <= 3) {
        len = 1 + rnd(seed, kind == 3 ? 200 : 24);
        for (uint64_t i = 0; i < len; i++) { p[i] = ALPHA[rnd(seed, sizeof ALPHA)]; }
    } else if (kind <= 6) {                     /* 1..3 whole tokens (any ids: added-only included) */
        uint64_t k = kind - 3;
        for (uint64_t j = 0; j < k; j++) {
            uint32_t id = (uint32_t)rnd(seed, m->n_ids);
            memcpy(p + len, m->bytes + m->off[id], m_len(m, id));
            len += m_len(m, id);
        }
        if (len == 0) { p[0] = 'a'; len = 1; }
    } else if (kind == 7) {
        len = 1 + rnd(seed, 16);
        for (uint64_t i = 0; i < len; i++) { p[i] = (uint8_t)rnd(seed, 256); }
    } else {
        len = 1 + rnd(seed, 3);
        for (uint64_t i = 0; i < len; i++) { p[i] = ALPHA[rnd(seed, 3)]; }
    }
    return len < maxlen ? len : maxlen;
}

static int expect_as_rank(const model *m)       /* the survivors' merged ids rise strictly in raw order */
{
    int have = 0;
    uint32_t prev = 0;
    for (uint32_t i = 0; i < m->nm; i++) {
        if (RANK[m->ml[i]][m->mr[i]] != (int32_t)i) { continue; }
        if (have && m->mo[i] <= prev) { return 0; }
        prev = m->mo[i];
        have = 1;
    }
    return 1;
}

static void check_tables(const model *m, uint64_t *seed)
{
    const toks_tables *t = &m->t;
    const bpe_mt mt = bpe_mt_of(t);
    int as_rank = expect_as_rank(m);
    CHECK(((t->flags & TOKS_TF_IDS_AS_RANK) != 0) == as_rank, "ids-as-rank flag %#x, expected %d", t->flags, as_rank);
    CHECK(as_rank ? t->rank2id == NULL : t->rank2id != NULL, "rank2id presence");
    CHECK(t->n_merges == m->nm, "n_merges %u != %u", t->n_merges, m->nm);
    CHECK(m->arena_used <= m->arena_need, "arena overrun");
    for (uint32_t b = 0; b < 256; b++) { CHECK(t->byte2id[b] == m->b2i[b], "byte2id[%u]", b); }
    for (uint32_t i = 0; i < m->nm; i++) {
        uint32_t pr = 0, k = (uint32_t)RANK[m->ml[i]][m->mr[i]];
        int found = ((pr = bpe_mt_find(&mt, bpe_pair_key(m->ml[i], m->mr[i]))) != TOKS_PRIO_NONE);
        int prio_ok = found && (as_rank ? pr == m->mo[k] : pr == k);
        CHECK(prio_ok && (as_rank || t->rank2id[pr] == m->mo[k]), "merge %u: prio %u (last rank %u)", i, pr, k);
    }
    /* layout.h's merge-table invariant, which the asm probes rely on (they stop at the first empty slot):
     * every bucket fills front to back, and a key sits d < merge_maxprobe buckets past its home with every
     * bucket in between full */
    uint64_t live = 0;
    for (uint64_t b = 0; b <= t->merge_mask; b++) {
        const uint64_t *sl = t->merge_slots + b * 8u;
        int gap = 0;
        for (uint32_t j = 0; j < 8u; j++) {
            if (sl[j] == BPE_EMPTY_SLOT) { gap = 1; continue; }
            CHECK(!gap, "merge bucket %" PRIu64 ": slot %u holds a key after an empty slot", b, j);
            live++;
            uint64_t key = sl[j] >> TOKS_PRIO_BITS;
            uint64_t home = ((key * TOKS_FIB64) >> t->merge_shift) & t->merge_mask;
            uint64_t d = (b - home) & t->merge_mask;
            CHECK(d < t->merge_maxprobe, "merge key %" PRIu64 " buckets past its home, maxprobe %" PRIu64, d, t->merge_maxprobe);
            for (uint64_t k = 0; k < d; k++) {
                CHECK(t->merge_slots[((home + k) & t->merge_mask) * 8u + 7u] != BPE_EMPTY_SLOT,
                      "merge key %" PRIu64 " buckets past its home, but bucket home + %" PRIu64 " has room", d, k);
            }
        }
    }
    CHECK(live <= m->nm, "%" PRIu64 " merge slots for %u merges", live, m->nm);
    for (uint32_t i = 0; i < 2000; i++) {
        uint32_t x = (uint32_t)rnd(seed, m->n_vocab), y = (uint32_t)rnd(seed, m->n_vocab);
        if (RANK[x][y] < 0) { CHECK(bpe_mt_find(&mt, bpe_pair_key(x, y)) == TOKS_PRIO_NONE, "absent pair %u+%u found", x, y); }
    }
    for (uint32_t bp = 0; bp < 65536; bp++) {
        int32_t k = RANK[m->b2i[bp >> 8]][m->b2i[bp & 255]];
        uint32_t want = k < 0 ? TOKS_PRIO_NONE : as_rank ? m->mo[k] : (uint32_t)k;
        if (t->bytepair[bp] != want) { CHECK(0, "bytepair[%#x] = %u, want %u", bp, t->bytepair[bp], want); break; }
    }
    for (uint32_t id = 0; id < m->n_ids; id++) {
        uint32_t got = UINT32_MAX;
        int64_t want = m_len(m, id) == 0 ? -1 : m_find(m, m->bytes + m->off[id], m_len(m, id), m->n_vocab);
        if (want >= 0 && !m->ignore_merges) {                /* only the tokens over 15 B whose bytes bpe back to them */
            uint64_t nr = ref_bpe(m, m->bytes + m->off[want], m_len(m, (uint32_t)want), REFB);
            uint32_t hi = 0;
            for (uint32_t i = 0; i < m_len(m, (uint32_t)want); i++) { hi |= m->bytes[m->off[want] + i]; }
            want = m_len(m, (uint32_t)want) > TOKS_KEY_MAXLEN && hi < 0x80 && nr == 1 && REFB[0] == (uint32_t)want ? want : -1;
        }
        int f = m_len(m, id) > 0 && bpe_vhash_find(t, m->bytes + m->off[id], m_len(m, id), &got);
        CHECK(want < 0 ? !f : (f && got == (uint32_t)want), "vhash on id %u: %d/%u, want %" PRId64, id, f, got, want);
    }
    /* the certified words table */
    uint64_t slots = (t->words_mask + 1) * 2, placed = 0;
    uint32_t ref[TOKS_KEY_MAXLEN];
    for (uint64_t bu = 0; t->words != NULL && bu <= t->words_mask; bu++) {
        for (uint32_t way = 0; way < 2; way++) {
            const uint8_t *key = t->words + bu * TOKS_BUCKET + way * 16;
            const uint32_t *val = (const uint32_t *)(const void *)(t->words + bu * TOKS_BUCKET + 32 + way * 16);
            uint32_t len = key[15];
            if (len == 0) { continue; }
            placed++;
            uint8_t pad = 0;
            for (uint32_t j = len; j < 15; j++) { pad |= key[j]; }
            uint32_t h = toks_key_hash(TOKS_HSEED, key);
            CHECK(len >= 2 && len <= 15 && pad == 0 && (bpe_words_bucket(t, h, 0) == bu || bpe_words_bucket(t, h, 1) == bu),
                  "words entry malformed or misplaced (bucket %" PRIu64 ")", bu);
            uint64_t nr = ref_bpe(m, key, len, ref);
            uint32_t got[4];
            (void)bpe_val_put((const uint8_t *)val, got);
            CHECK(nr >= 1 && nr <= 4 && bpe_val_count(val) == nr && memcmp(got, ref, nr * 4) == 0,
                  "words entry (%u bytes) is not the reference's answer", len);
        }
    }
    /* under ignore_merges every token of 2..15 bytes whose merges alone are not itself (K6's own answer without the
     * probe) is an entry, seated past the load cap when the fill reached it there: then TOKS_TF_PROBE_LONG */
    uint64_t cap = slots * 85 / 100, need = 0, seated = 0;
    for (uint32_t id = 0; t->words != NULL && m->ignore_merges && id < m->n_vocab; id++) {
        uint32_t len = m_len(m, id);
        if (len < 2 || len > 15 || (ref_merges(m, m->bytes + m->off[id], len, ref) == 1 && ref[0] == id)) { continue; }
        uint8_t key[16];
        uint32_t val[4];
        bpe_key_make(key, m->bytes + m->off[id], len);
        need++;
        seated += bpe_words_find(t, key, val) && bpe_val_count(val) == 1 && (val[0] & TOKS_ID_MASK) == id;
    }
    CHECK(((t->flags & TOKS_TF_PROBE_LONG) != 0) == (t->words != NULL && (m->ignore_merges ? seated == need : t->vhash != NULL &&
          (t->flags & TOKS_TF_IGNORE_MERGES) != 0)), "words: probe-long flag %#x, %" PRIu64 " of %" PRIu64 " tokens that need the "
          "whole-piece rule seated", t->flags, seated, need);
    CHECK(t->words == NULL || placed <= cap + seated, "words load %" PRIu64 "/%" PRIu64 " above 0.85", placed, slots);
    int ascii = 1;                                         /* TOKS_TF_PROBE_ASCII: with PROBE_LONG, without */
    for (uint32_t id = 0; m->ignore_merges && id < m->n_vocab; id++) {   /* ignore_merges, or when no token over */
        uint32_t len = m_len(m, id);                       /* 15 bytes ending in a non-ascii byte needs the rule */
        uint32_t big[256];
        if (len <= 15 || len > 256 || m->bytes[m->off[id] + len - 1] < 0x80) { continue; }
        ascii &= ref_merges(m, m->bytes + m->off[id], len, big) == 1 && big[0] == id;
    }
    CHECK(((t->flags & TOKS_TF_PROBE_ASCII) != 0) == ((t->flags & TOKS_TF_PROBE_LONG) != 0 && ascii),
          "words: probe-ascii flag %#x", t->flags);
    for (uint32_t id = 0; t->words != NULL && id < m->n_vocab; id++) {
        uint32_t len = m_len(m, id);
        if (len < 2 || len > 15 || ref_bpe(m, m->bytes + m->off[id], len, ref) > 4) { continue; }
        uint8_t key[16];
        uint32_t val[4];
        bpe_key_make(key, m->bytes + m->off[id], len);
        if (bpe_words_find(t, key, val)) { continue; }
        uint32_t h = toks_key_hash(TOKS_HSEED, key);
        const uint8_t *b0 = t->words + bpe_words_bucket(t, h, 0) * TOKS_BUCKET;
        const uint8_t *b1 = t->words + bpe_words_bucket(t, h, 1) * TOKS_BUCKET;
        int full = b0[15] && b0[31] && b1[15] && b1[31];
        CHECK(full || placed >= cap, "words left out token %u with room in its buckets", id);
    }
}

/* K6 against ref with the piece, out and work flush against guard pages */
static void k6_guarded(const model *m, const uint8_t *src, uint64_t len)
{
    guard_buf gp = {0}, gq = {0}, go = {0}, gw = {0};
    uint64_t wb = TOKS_BPE_WORK_BYTES(len), wb64 = (wb + 63) & ~(uint64_t)63;
    uint8_t *pe = guard_alloc(&gp, len, GUARD_END, 0);
    uint8_t *ps = guard_alloc(&gq, len, GUARD_START, 0);
    uint32_t *out = (uint32_t *)(void *)guard_alloc(&go, len * 4, GUARD_END, 0);
    uint8_t *work = guard_alloc(&gw, wb64, GUARD_END, 0);
    if (!pe || !ps || !out || !work) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    memcpy(pe, src, len);
    memcpy(ps, src, len);
    uint64_t nr = ref_bpe(m, src, len, REFB);
    for (int k = 0; k < 2; k++) {
        toks_k6_args a = { k == 0 ? pe : ps, len, out, work, wb, 0, 0, 0 };
        uint64_t n = K6_BPE(&m->t, &a);
        CHECK(same_ids(out, n, REFB, nr), "guarded k6 (%s flush, %" PRIu64 " bytes) != ref", k == 0 ? "end" : "start", len);
    }
    guard_free(&gp); guard_free(&gq); guard_free(&go); guard_free(&gw);
}

/* K5 over a stream of pieces, against the concatenated references */
static uint64_t LCHITS;                      /* long-cache hits over every k5_stream */

static void k5_stream(const model *m, uint64_t *seed)
{
    static uint8_t buf[256 * 320];
    static uint32_t ends[256], want[256 * 320];
    uint64_t np = 8 + rnd(seed, 200), len = 0, nw = 0, maxlen = 1;
    for (uint64_t i = 0; i < np; i++) {
        uint64_t l = gen_piece(m, seed, buf + len, 300);
        nw += ref_bpe(m, buf + len, l, want + nw);
        len += l;
        ends[i] = (uint32_t)len;
        maxlen = l > maxlen ? l : maxlen;
    }
    guard_buf gt = {0}, go = {0}, gw = {0};
    uint64_t room = len + 4, wb = TOKS_BPE_WORK_BYTES(maxlen);
    uint8_t *text = guard_alloc(&gt, len, GUARD_END, 0);
    uint32_t *out = (uint32_t *)(void *)guard_alloc(&go, room * 4, GUARD_END, 0);
    uint8_t *work = guard_alloc(&gw, (wb + 63) & ~(uint64_t)63, GUARD_END, 0);
    if (!text || !out || !work) { fprintf(stderr, "guard_alloc failed\n"); exit(2); }
    memcpy(text, buf, len);
    static _Alignas(64) uint8_t cache[16 * 64], lb[4 * 64], la[4096];
    static const uint64_t masks[] = { UINT64_MAX, 0, 15, 15 };
    toks_lcache lc = { lb, la, 3, 1024u + 16u * rnd(seed, 192), 0, 1, 0, 0 };   /* c = 3: a long cache too, small */
    memset(lb, 0, sizeof lb);
    for (uint32_t c = 0; c < 4; c++) {
        K5LC = c == 3 ? &lc : NULL;
        memset(cache, 0, sizeof cache);
        uint8_t *cp = masks[c] == UINT64_MAX ? NULL : cache;
        uint64_t mask = masks[c] == UINT64_MAX ? 0 : masks[c];
        for (int pass = 0; pass < 2; pass++) {
            uint64_t ctr[3];
            uint64_t n = run_k5(m, text, len, ends, np, 0, out, room, cp, mask, work, wb, ctr);
            int ok = same_ids(out, n, want, nw);
            CHECK(ok, "k5 stream (%" PRIu64 " pieces, cache %u, pass %d) != ref", np, c, pass);
            CHECK(cp != NULL || ctr[1] == 0, "k5 counted cache hits without a cache");
        }
        /* resume from a middle piece: start = the previous end */
        uint64_t k = np / 2, skip = 0, nr = 0;
        for (uint64_t i = 0; i < np; i++) {
            uint64_t s = i == 0 ? 0 : ends[i - 1];
            uint64_t l = ends[i] - s;
            uint64_t cnt = ref_bpe(m, buf + s, l, REFB);
            if (i < k) { skip += cnt; } else { nr += cnt; }
        }
        uint64_t ctr[3];
        uint64_t n = run_k5(m, text, len, ends + k, np - k, k == 0 ? 0 : ends[k - 1], out, room, cp, mask, work, wb, ctr);
        CHECK(same_ids(out, n, want + skip, nr), "k5 resumed at piece %" PRIu64 " != ref", k);
    }
    K5LC = NULL;
    LCHITS += lc.hits;
    guard_free(&gt); guard_free(&go); guard_free(&gw);
}

/* a run of U+4E2D / e-acute chars with now and then an ascii or a loose ALPHA byte: premerge side by side */
static uint64_t gen_chars(uint64_t *seed, uint8_t *p, uint64_t maxlen)
{
    static const char *const U[] = { "\xE4\xB8\xAD", "\xC3\xA9", "a", " " };
    uint64_t len = 0, n = 1 + rnd(seed, 40);
    for (uint64_t i = 0; i < n && len + 3 <= maxlen; i++) {
        uint64_t r = rnd(seed, 20);
        const char *u = U[r < 10 ? 0 : r < 17 ? 1 : r < 19 ? r - 15 : 0];
        if (r == 19) { p[len++] = ALPHA[rnd(seed, sizeof ALPHA)]; continue; }
        memcpy(p + len, u, strlen(u));
        len += strlen(u);
    }
    return len;
}

#if TOKS_HAVE_K6_NEON
#define K6_MERGE toks_k6_merge_neon
#elif TOKS_HAVE_K6_AVX2
#define K6_MERGE toks_k6_merge_avx2
#endif
#ifdef K6_MERGE
/* K6's asm merge loop on given symbols (spm's model) against the c twin's toks_k6_merge: byte symbols of ALPHA and
 * random vocab ids, 2..128 of them */
static void merge_vs_c(const model *m, uint64_t *seed)
{
    _Alignas(64) static uint8_t w1[TOKS_BPE_WORK_BYTES(128)], w2[TOKS_BPE_WORK_BYTES(128)];
    static uint32_t sym[192], o1[129], o2[129];
    for (uint32_t i = 0; i < 100; i++) {
        uint64_t n = 2 + rnd(seed, i < 50 ? 14 : 127);
        for (uint64_t j = 0; j < n; j++) {
            sym[j] = rnd(seed, 10) < 7 ? m->t.byte2id[ALPHA[rnd(seed, sizeof ALPHA)]] : (uint32_t)rnd(seed, m->n_vocab);
        }
        memcpy(w1 + 16 * n, sym, 4 * n);
        uint64_t c1 = toks_k6_merge(&m->t, w1, n, n, o1);
        toks_k6_args a = { (const uint8_t *)(void *)sym, n, o2, w2, sizeof w2, 0, 0, 0 };
        uint64_t c2 = K6_MERGE(&m->t, &a);
        CHECK(c1 == c2 && memcmp(o1, o2, 4 * c1) == 0 && a.n_out == c2 && a.merges == n - c2,
              "asm merge on %" PRIu64 " symbols: %" PRIu64 " ids != the c twin's %" PRIu64, n, c2, c1);
    }
}
#endif

static void test_random(void)
{
    uint64_t seed = 0x62706521u;
    uint64_t pieces = 0, ids_rank2id = 0, with_pm = 0, pm_chars = 0;
    static uint8_t p[2048];
    for (uint32_t round = 0; round < 240; round++) {
        uint32_t mode = (uint32_t)rnd(&seed, 256);
        if (round % 4 == 0) { mode &= G_EXTRAS | G_IGNORE | G_CHARS; }   /* clean merges: must take the ids-as-rank path */
        gen_model(&M, &seed, mode);
        if (m_build(&M) != 0) { continue; }
        if (round % 4 == 0) { CHECK((M.t.flags & TOKS_TF_IDS_AS_RANK) != 0, "clean model %u: not ids-as-rank", round); }
        ids_rank2id += (M.t.flags & TOKS_TF_IDS_AS_RANK) == 0;
        if (M.t.premerge != NULL) {
            with_pm++;
            for (uint32_t id = 0; id < M.n_vocab; id++) {
                uint32_t s1, s2, l = m_len(&M, id);
                if ((l == 2 || l == 3) && bpe_pm_index(M.bytes + M.off[id], l, &s1, &s2) == l) {
                    const uint32_t *st1 = (const uint32_t *)(const void *)M.t.premerge;
                    pm_chars += *(const uint64_t *)(const void *)(M.t.premerge + st1[s1] + 8 * s2) != 0;
                }
            }
        }
        check_tables(&M, &seed);
        for (uint32_t i = 0; i < 300; i++) {
            uint64_t len = gen_piece(&M, &seed, p, sizeof p);
            k6_vs_ref(&M, p, len, "random");
            pieces++;
        }
        for (uint32_t i = 0; i < 100; i++) {                 /* char runs: premerged chars side by side */
            uint64_t len = gen_chars(&seed, p, 120);
            k6_vs_ref(&M, p, len, "chars");
            pieces++;
        }
        for (uint32_t i = 0; i < 6; i++) {
            uint64_t len = 1 + rnd(&seed, 70);
            for (uint64_t j = 0; j < len; j++) { p[j] = ALPHA[rnd(&seed, sizeof ALPHA)]; }
            k6_guarded(&M, p, len);
        }
        uint64_t len = 600 + rnd(&seed, 500);
        for (uint64_t j = 0; j < len; j++) { p[j] = ALPHA[rnd(&seed, 4)]; }
        k6_vs_ref(&M, p, len, "random long");
        k5_stream(&M, &seed);
#ifdef K6_MERGE
        merge_vs_c(&M, &seed);
#endif
    }
    printf("random: 240 models (%" PRIu64 " on the rank2id path, %" PRIu64 " with a premerge table: %" PRIu64
           " chars), %" PRIu64 " pieces, %" PRIu64 " long-cache hits\n", ids_rank2id, with_pm, pm_chars, pieces, LCHITS);
    CHECK(LCHITS > 0, "k5 streams: the long cache never answered");
}

int main(void)
{
#ifdef TEST_BPE_FEAT
    if (!TOKS_CPU_HAS(toks_cpu_features(), TEST_BPE_FEAT)) {
        printf("SKIP %s: this cpu lacks the %s tier\n", TEST_BPE_NAME, TEST_BPE_TIER);
        return 0;
    }
#endif
    memset(RANK, 0xFF, sizeof RANK);
    WORK = guard_aligned_alloc(64, (TOKS_BPE_WORK_BYTES(BIGLEN) + 63) & ~(uint64_t)63);
    OUTB = malloc(4u * BIGLEN);
    REFB = malloc(4u * BIGLEN);
    if (!WORK || !OUTB || !REFB) { fprintf(stderr, "out of memory\n"); return 2; }
    test_trap();
    test_ties();
    test_unreachable();
    test_long_probe();
    test_seat();
    test_duplicates();
    test_one_byte_and_zero_merges();
    test_build_errors();
    test_arena_tight();
    test_k5_cache();
    test_k5_tags();
    test_k5_long();
    test_key_in_place();
    test_long_piece();
    test_runs();
    test_random();
    m_reset(&M);
    printf("%s: %" PRIu64 " checks, %" PRIu64 " failures\n", TEST_BPE_NAME, checks, failures);
    return failures == 0 ? 0 : 1;
}
