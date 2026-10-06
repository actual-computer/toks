/*
 * test_api.c: the per-call driver (api.c: scratch, encode, pieces, decode, token, info) over a
 * hand-built context: real class tables (classes.c), the real K1 and K3 c twins, 256 byte tokens
 * (id = byte), added tokens of both phases, a post-processor prefix and suffix.
 *
 * K5 is a stand-in on purpose (the real one runs in test_e2e.c): bpe with no merges (one id per
 * byte, what K6 computes on merge-free tables), checking every precondition the driver owes K5
 * (room >= bytes + 4, work >= TOKS_BPE_WORK_BYTES(longest piece), 64-aligned cache and work) and
 * writing out[n_out, n_out + 68), out[room - 1] and the work area, so a region the driver misplaces
 * faults on the guard pages below. It wins the link over k5_c.o, which nothing else here pulls.
 *
 * Checked: K5's counters adding up in the scratch header's counters line; the scratch formula and
 * layout (exactly toks_scratch_bytes bytes, flush against a guard
 * page at either end, every start alignment); every capacity 0..n+1 of encode and pieces with the
 * output's end flush against a guard page; modes, post-processing, chunk rounds past
 * TOKS_CHUNK_PIECES, the bounce; decode against hf 0.23.2's own outputs (lossy utf-8) plus a
 * differential reference; argument and scratch-binding errors; token; info; sha-256; the crc32c
 * table against layout.h's bitwise definition; the byte-level alphabet against gpt-2's; one compiled file
 * (tests/data/compile/gpt2style.json): its normalized=true <|endoftext|> dropped by TOKS_SKIP_SPECIAL, batch and stream.
 */
#include "../../src/core/kernels.h"
#include "../../src/core/bpe.h"
#include "../../src/core/compile.h"
#include "classes.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* ================================================================ the stand-in K5 */

static uint64_t k5_calls, k5_pieces, k5_tag, k5_mask;
static toks_lcache k5_lc;                               /* the stand-in's last lcache (k5_lc.buckets NULL: none) */

uint64_t toks_k5_encode_c(const toks_tables *t, toks_k5_args *a)
{
    k5_calls++;
    uint64_t s = a->start, bytes = 0u, longest = 0u;
    for (uint64_t i = 0; i < a->n; i++) {
        uint64_t e = a->ends[i];
        CHECK(e > s && e <= a->len, "k5: piece [%" PRIu64 ", %" PRIu64 ") outside the segment", s, e);
        if (e - s > longest) { longest = e - s; }
        bytes += e - s;
        s = e;
    }
    CHECK(a->room >= bytes + 4u, "k5: room %" PRIu64 " < bytes %" PRIu64 " + 4", a->room, bytes);
    CHECK(a->work_bytes >= TOKS_BPE_WORK_BYTES(longest), "k5: work %" PRIu64 " < need for %" PRIu64,
          a->work_bytes, longest);
    CHECK(((uintptr_t)a->cache & 63u) == 0 && ((uintptr_t)a->work & 63u) == 0, "k5: cache/work not 64-aligned");
    k5_mask = a->cache_mask;                            /* TOKS_CACHE_MASK, or TOKS_SCRATCH_CACHE_MIB's (test_cache_mib) */
    memset(&k5_lc, 0, sizeof k5_lc);
    if (a->lcache != NULL) {
        k5_lc = *(const toks_lcache *)a->lcache;
        k5_lc.buckets[0] = k5_lc.buckets[0];            /* both ends of the long cache */
        k5_lc.arena[k5_lc.arena_bytes - 1u] = k5_lc.arena[k5_lc.arena_bytes - 1u];
    }
    /* touch what K5 may write: the work K6 needs, the work area's last byte, both cache ends,
     * out[n_out, n_out + 68) and out[room - 1] (the farthest id the driver allows) */
    memset(a->work, 0xA5, (size_t)TOKS_BPE_WORK_BYTES(longest));
    a->work[a->work_bytes - 1u] = 0xA5;
    volatile uint8_t *c = a->cache;
    c[0] = c[0];
    c[(a->cache_mask + 1u) * 64u - 1u] = c[(a->cache_mask + 1u) * 64u - 1u];
    uint64_t n_out = 0u;
    for (uint64_t j = a->start; j < s; j++) { a->out[n_out++] = t->byte2id[a->text[j]]; }
    for (uint64_t k = n_out; k < a->room && k < n_out + 68u; k++) { a->out[k] = 0xDEADBEEFu; }
    a->out[a->room - 1u] = 0xDEADBEEFu;
    a->n_out = n_out;
    a->misses += a->n;                                  /* the counters accumulate (kernels.md §6) */
    k5_pieces += a->n;
    k5_tag = a->cache_tag;                              /* the scratch's epoch (kernels.md §7) */
    return n_out;
}

/* ================================================================ the stand-in cpu probe */

/* toks_get_info reports the feature word load stored and never probes the cpu again (SPEC §4.1:
 * no syscall after load; on macos the probe is three sysctlbyname calls). This stand-in wins the
 * link over cpu.o and counts its callers. */
static uint64_t cpu_calls;

uint64_t toks_cpu_features(void)
{
    cpu_calls++;
    return 0u;
}

/* ================================================================ the context */

#define N_IDS 262u
#define ID_EOT 256u      /* "<|eot|>"  special, phase 0 (raw) */
#define ID_X   257u      /* "<|x|>"    non-special, phase 0 */
#define ID_EUR 258u      /* "\xe2\x82\xac" (decode only) */
#define ID_E2  259u      /* "\xe2\x82" (decode only) */
#define ID_NIL 260u      /* no string */
#define ID_XYZ 261u      /* "xyz"      non-special, phase 1 (normalized) */

typedef struct { const char *s; uint32_t id; uint8_t flags; uint8_t phase; } add_tok;
static const add_tok ADDED[] = {
    { "<|eot|>", ID_EOT, TOKS_AF_SPECIAL, 0 },
    { "<|x|>",   ID_X,   0,               0 },
    { "xyz",     ID_XYZ, TOKS_AF_NORMALIZED, 1 },
};
#define N_ADDED (sizeof ADDED / sizeof ADDED[0])

static toks_ctx CTX;
static uint32_t BYTE2ID[256];
static uint32_t TOK_OFF[N_IDS + 1u];
static uint8_t  TOK_BYTES[512];
static toks_added_entry ENTRIES[8];
static uint8_t  ADD_BYTES[64];
static uint8_t  SHUFTI[2][32];
static uint64_t INDEX2[2u * 65536u + (2u << TOKS_K1_H4_BITS)];   /* index2, then h4 */
static uint32_t SINGLE[2][256];
static uint32_t CAND[16];
static uint32_t SPECIAL[(N_IDS + 31u) / 32u];

static void build_ctx(void)
{
    memset(&CTX, 0, sizeof CTX);
    toks_tables *t = &CTX.t;
    t->magic = TOKS_TABLES_MAGIC;
    t->version = TOKS_TABLES_VERSION;
    t->algo = TOKS_ALGO_BPE_BYTELEVEL;
    t->tmpl = TOKS_TMPL_CL100K;
    t->tmpl_params = TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1_3 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL;
    t->n_ids = N_IDS;

    uint64_t cb = toks_classes_bytes(0u);
    uint8_t *cls = (uint8_t *)malloc((size_t)cb);
    toks_class_tables ct;
    if (cls == NULL || toks_classes_build(0u, cls, cb, &ct) <= 0) { fprintf(stderr, "classes\n"); exit(2); }
    t->cls_ascii = ct.ascii;
    t->cls_stage1 = ct.stage1;
    t->cls_stage2 = ct.stage2;
    t->cls_nblocks = ct.n_blocks;

    /* token bytes: ids 0..255 = the byte; then the added and decode-only tokens */
    uint32_t o = 0u;
    for (uint32_t b = 0; b < 256u; b++) { BYTE2ID[b] = b; TOK_OFF[b] = o; TOK_BYTES[o++] = (uint8_t)b; }
    const char *extra[N_IDS - 256u] = { "<|eot|>", "<|x|>", "\xe2\x82\xac", "\xe2\x82", "", "xyz" };
    for (uint32_t i = 256u; i < N_IDS; i++) {
        TOK_OFF[i] = o;
        size_t k = strlen(extra[i - 256u]);
        memcpy(TOK_BYTES + o, extra[i - 256u], k);
        o += (uint32_t)k;
    }
    TOK_OFF[N_IDS] = o;
    t->byte2id = BYTE2ID;
    t->tok_off = TOK_OFF;
    t->tok_bytes = TOK_BYTES;

    /* the added entries, then K1's index (kernels.md §4) */
    uint32_t pool = 0u;
    for (uint32_t i = 0; i < N_ADDED; i++) {
        uint32_t k = (uint32_t)strlen(ADDED[i].s), p = ADDED[i].phase;
        ENTRIES[i].off = pool;
        ENTRIES[i].len = (uint16_t)k;
        ENTRIES[i].flags = ADDED[i].flags;
        ENTRIES[i].phase = (uint8_t)p;
        ENTRIES[i].id = ADDED[i].id;
        memcpy(ADD_BYTES + pool, ADDED[i].s, k);
        pool += k;
        if ((ADDED[i].flags & TOKS_AF_SPECIAL) != 0u) { SPECIAL[ADDED[i].id >> 5] |= 1u << (ADDED[i].id & 31u); }
        else { CTX.n_nonspecial++; }
    }
    t->add_entries = ENTRIES;
    t->add_bytes = ADD_BYTES;
    t->add_n = N_ADDED;
    toks_added_index(t, &SHUFTI[0][0], INDEX2, &SINGLE[0][0], CAND);   /* the compiler's own builder */

    CTX.tier = TOKS_TIER_SCALAR;
    CTX.dec_byte_level = 1u;
    CTX.pp_ids[0] = ID_EOT;        /* prefix */
    CTX.pp_ids[1] = ID_X;          /* suffix */
    CTX.n_pp_prefix = 1u;
    CTX.n_pp_suffix = 1u;
    CTX.special_ids = SPECIAL;
    CTX.identity = 0x1234567890ABCDEFull;
    CTX.cpu_features = 0x5EED0000F00Dull;
    memcpy(CTX.name, "hand", 5);
}

/* ================================================================ the reference */

typedef struct { uint32_t *v; uint64_t n, cap; } vec;

static void push(vec *x, uint32_t v)
{
    if (x->n == x->cap) {
        x->cap = x->cap ? 2u * x->cap : 64u;
        x->v = (uint32_t *)realloc(x->v, (size_t)x->cap * 4u);
        if (x->v == NULL) { exit(2); }
    }
    x->v[x->n++] = v;
}

/* leftmost-longest among the phase's tokens at or after pos: the start (len when none) */
static uint64_t ref_find(const uint8_t *t, uint64_t len, uint64_t pos, uint32_t phase, uint32_t *mlen, uint32_t *mi)
{
    for (uint64_t i = pos; i < len; i++) {
        uint32_t best = 0u;
        for (uint32_t k = 0; k < N_ADDED; k++) {
            uint32_t tl = (uint32_t)strlen(ADDED[k].s);
            if (ADDED[k].phase == phase && i + tl <= len && tl > best && memcmp(t + i, ADDED[k].s, tl) == 0) {
                best = tl;
                *mi = k;
            }
        }
        if (best != 0u) { *mlen = best; return i; }
    }
    return len;
}

/* a text segment: one id per byte (the stand-in K5), or the K3 c twin's piece ends */
static void ref_text(const uint8_t *t, uint64_t base, uint64_t len, int ids, vec *out)
{
    if (len == 0u) { return; }
    if (ids) { for (uint64_t i = 0; i < len; i++) { push(out, t[i]); } return; }
    uint32_t *ends = (uint32_t *)malloc((size_t)len * 4u);
    toks_k3_args k;
    memset(&k, 0, sizeof k);
    k.text = t; k.len = len; k.pos = 0; k.ends = ends; k.cap = len;
    uint64_t n = toks_k3_scan_cl100k_c(&CTX.t, &k);
    for (uint64_t i = 0; i < n; i++) { push(out, (uint32_t)(base + ends[i])); }
    free(ends);
}

static void ref_phase(const uint8_t *t, uint64_t base, uint64_t len, uint32_t mode, uint32_t phase, int ids, vec *out)
{
    uint64_t pos = 0u, prev = 0u;
    for (;;) {
        uint32_t ml = 0u, mi = 0u;
        uint64_t m = ref_find(t, len, pos, phase, &ml, &mi);
        if (m >= len) { break; }
        pos = m + ml;                                       /* resume at the raw end, always */
        if (mode == TOKS_ADDED_NONSPECIAL && (ADDED[mi].flags & TOKS_AF_SPECIAL) != 0u) { continue; }
        if (phase == 0u) { ref_phase(t + prev, base + prev, m - prev, mode, 1u, ids, out); }
        else { ref_text(t + prev, base + prev, m - prev, ids, out); }
        push(out, ids ? ADDED[mi].id : (uint32_t)(base + m + ml));
        prev = m + ml;
    }
    if (phase == 0u) { ref_phase(t + prev, base + prev, len - prev, mode, 1u, ids, out); }
    else { ref_text(t + prev, base + prev, len - prev, ids, out); }
}

static void ref_run(const uint8_t *t, uint64_t len, uint32_t flags, int ids, vec *out)
{
    out->n = 0u;
    uint32_t mode = flags & TOKS_ADDED_MASK;
    int pp = ids && (flags & TOKS_NO_POSTPROCESS) == 0u;
    if (pp) { push(out, ID_EOT); }
    if (mode == TOKS_ADDED_NONE) { ref_text(t, 0u, len, ids, out); }
    else { ref_phase(t, 0u, len, mode, 0u, ids, out); }
    if (pp) { push(out, ID_X); }
}

/* ================================================================ texts */

static const char *FRAG[] = {
    "<|eot|>", "<|x|>", "xyz", "<|", "|>", "<|eot", "x", "y", "z", " ", "  ", "\n", "\r\n", "\t",
    "a", "Hello", "world", "123", "4", "56789", "'s", "'T", "!", "?!", ".", "\xc3\xa9", "\xe4\xbd\xa0",
    "\xf0\x9f\x98\x80", "\xff", "\xe2\x82", "\xc3", " \n ", "don't", "IT'S",
};
#define N_FRAG (sizeof FRAG / sizeof FRAG[0])

static uint64_t rng = 0x9E3779B97F4A7C15ull;

static uint64_t gen_text(uint8_t *buf, uint64_t want)
{
    uint64_t n = 0u;
    while (n < want) {
        const char *f = FRAG[guard_rng(&rng) % N_FRAG];
        size_t k = strlen(f);
        if (n + k > want) { k = (size_t)(want - n); }
        memcpy(buf + n, f, k);
        n += k;
    }
    return n;
}

/* ================================================================ scratch */

static void test_formula(void)
{
    static const uint64_t L[] = { 0, 1, 2, 63, 64, 65, 255, 256, 1000, 4096, 65539, 1u << 20, TOKS_MAX_TEXT };
    uint64_t prev = 0u;
    for (size_t i = 0; i < sizeof L / sizeof L[0]; i++) {
        uint64_t b = toks_scratch_bytes(&CTX, L[i], TOKS_SCRATCH_MEMO_MIB(0));
        /* independent restatement: alignment slack + header + ends + cache + work + bounce */
        uint64_t need = 63u + 128u + 4u * TOKS_CHUNK_PIECES + (uint64_t)TOKS_CACHE_BUCKETS * 64u
                      + 256u + 32u * L[i] + 4u * (L[i] + 4u);
        CHECK(b >= need, "formula(%" PRIu64 ") = %" PRIu64 " < %" PRIu64, L[i], b, need);
        CHECK(b <= need + 3u * 64u, "formula(%" PRIu64 ") = %" PRIu64 " wastes > 192 B", L[i], b);
        CHECK(b >= prev, "formula not monotone at %" PRIu64, L[i]);
        /* the segment memo on top: 4 MiB by default (flags 0), m MiB with TOKS_SCRATCH_MEMO_MIB(m); a 0.2.0 binary's
         * TOKS_SCRATCH_MEMO_MIB(m) had no bit 20: m > 0 means m MiB still, its m = 0 is flags 0 */
        CHECK(toks_scratch_bytes(&CTX, L[i], 0u) == b + (4u << 20) &&
              toks_scratch_bytes(&CTX, L[i], TOKS_SCRATCH_MEMO_MIB(4)) == b + (4u << 20) &&
              toks_scratch_bytes(&CTX, L[i], TOKS_SCRATCH_MEMO_MIB(1)) == b + (1u << 20) &&
              toks_scratch_bytes(&CTX, L[i], TOKS_SCRATCH_MEMO_SET) == b &&
              toks_scratch_bytes(&CTX, L[i], 3u) == b + (3u << 20), "formula(%" PRIu64 "): the memo's bytes", L[i]);
        prev = b;
    }
    CHECK(toks_scratch_bytes(&CTX, TOKS_MAX_TEXT + 1u, 0u) == toks_scratch_bytes(&CTX, TOKS_MAX_TEXT, 0u), "clamp");
    /* wordpiece and unigram never use a memo: none at any size (their tables are never read here) */
    toks_ctx wp = CTX, uni = CTX;
    wp.wp = (const struct toks_wp_tables *)(const void *)&CTX;
    uni.uni = (const struct toks_uni *)(const void *)&CTX;
    uint64_t b0 = toks_scratch_bytes(&CTX, 1000u, TOKS_SCRATCH_MEMO_MIB(0));
    CHECK(toks_scratch_bytes(&wp, 1000u, 0u) == b0 && toks_scratch_bytes(&wp, 1000u, TOKS_SCRATCH_MEMO_MIB(8)) == b0 &&
          toks_scratch_bytes(&uni, 1000u, 0u) == b0 && toks_scratch_bytes(&uni, 1000u, TOKS_SCRATCH_MEMO_MIB(8)) == b0,
          "wordpiece / unigram: a memo in the formula");
    CHECK(toks_scratch_bytes(NULL, 1000u, 0u) == toks_scratch_bytes(NULL, 1000u, TOKS_SCRATCH_MEMO_MIB(0)) + (4u << 20),
          "no ctx: the largest layout has the default memo");
}

/* the header of an initialized scratch (core.h: first, at the first 64-aligned address) */
static toks_scratch *hdr_of(void *scr)
{
    uintptr_t base = (uintptr_t)scr;
    return (toks_scratch *)(void *)((uint8_t *)scr + (((base + 63u) & ~(uintptr_t)63u) - base));
}

/* the regions of an initialized scratch: 64-aligned, in order, disjoint, inside [scr, scr + bytes); memo: the memo's
 * bytes its flags give (between the cache and work) */
static void check_layout(void *scr, uint64_t bytes, uint64_t want_len, uint32_t flags, uint64_t memo)
{
    uintptr_t base = (uintptr_t)scr;
    uint64_t a = ((base + 63u) & ~(uintptr_t)63u) - base;
    uint64_t c = a + 128u + 4u * TOKS_CHUNK_PIECES;    /* the cache follows the header (binding + counters) and the ends */
    toks_scratch *h = hdr_of(scr);
    CHECK(h->magic == TOKS_SCRATCH_MAGIC && h->base == base && h->bytes == bytes, "header");
    CHECK(h->max_len >= want_len, "max_len %" PRIu64 " < %" PRIu64, h->max_len, want_len);
    CHECK(toks_scratch_bytes(&CTX, h->max_len + 1u, flags) > bytes || h->max_len == TOKS_MAX_TEXT,
          "max_len %" PRIu64 " is not the largest that fits %" PRIu64, h->max_len, bytes);
    CHECK(h->off_cache == c && ((base + h->off_work) & 63u) == 0 && ((base + h->off_bounce) & 63u) == 0,
          "regions not 64-aligned");
    uint64_t mb;
    uint8_t *m = toks_scr_memo(h, &mb);
    CHECK(mb == memo && h->off_work == c + TOKS_CACHE_BYTES + memo && (memo == 0u || m == toks_scr_at(h, c + TOKS_CACHE_BYTES)),
          "the memo: %" PRIu64 " bytes, want %" PRIu64 " between the cache and work", mb, memo);
    CHECK(h->off_bounce >= h->off_work + TOKS_BPE_WORK_BYTES(h->max_len), "bounce overlaps work");
    CHECK(h->off_bounce + 4u * (h->max_len + 4u) <= bytes, "bounce past the end");
    for (uint64_t i = 0; i < TOKS_CACHE_BYTES; i += 512u) {
        CHECK(toks_scr_at(h, c)[i] == 0, "cache not zeroed");
    }
    /* touch every byte of every region: a region past the end faults on the guard page */
    memset(toks_scr_at(h, c), 0x5A, (size_t)TOKS_CACHE_BYTES);
    if (memo != 0u) { memset(toks_scr_at(h, c + TOKS_CACHE_BYTES), 0x5A, (size_t)memo); }
    memset(toks_scr_ends(h), 0x5A, 4u * TOKS_CHUNK_PIECES);
    memset(toks_scr_at(h, h->off_work), 0x5A, (size_t)(h->off_bounce - h->off_work));
    memset(toks_scr_at(h, h->off_bounce), 0x5A, (size_t)(4u * (h->max_len + 4u)));
}

/* encode + pieces of an exactly-max_len text through this scratch, cap 0 (every round bounces)
 * and cap n (straight into out) */
static void run_through(void *scr, uint64_t max_len)
{
    toks_scratch *h = hdr_of(scr);
    uint64_t m0 = h->misses, p0 = k5_pieces;
    uint8_t *text = (uint8_t *)malloc((size_t)max_len + 1u);
    uint64_t len = gen_text(text, max_len);
    vec want = { 0, 0, 0 };
    ref_run(text, len, 0u, 1, &want);
    int64_t c0 = toks_encode(&CTX, text, len, 0u, NULL, 0u, scr);
    CHECK(c0 == (int64_t)want.n, "count-only encode: %" PRId64 " want %" PRIu64 " (len %" PRIu64 ")", c0, want.n, len);
    uint32_t *out = (uint32_t *)malloc((size_t)want.n * 4u + 4u);
    int64_t n = toks_encode(&CTX, text, len, 0u, out, want.n, scr);
    CHECK(n == (int64_t)want.n && memcmp(out, want.v, (size_t)want.n * 4u) == 0, "encode through scratch");
    ref_run(text, len, 0u, 0, &want);
    CHECK(toks_pieces(&CTX, text, len, 0u, NULL, 0u, scr) == (int64_t)want.n, "count-only pieces");
    CHECK(h->misses - m0 == k5_pieces - p0 && h->hits_static == 0u && h->hits_cache == 0u,
          "K5's counters in the scratch: misses +%" PRIu64 " for %" PRIu64 " pieces", h->misses - m0, k5_pieces - p0);
    free(out);
    free(want.v);
    free(text);
}

static void test_scratch(void)
{
    static const uint64_t L[] = { 0, 1, 5, 63, 64, 255, 256, 257, 1000, 5000 };
    static const uint32_t SF[2] = { 0u, TOKS_SCRATCH_MEMO_MIB(0) };   /* the default (a 4 MiB memo), no memo */
    for (size_t i = 0; i < 2u * (sizeof L / sizeof L[0]); i++) {
        uint32_t sf = SF[i & 1u];
        uint64_t len = L[i >> 1], memo = sf == 0u ? 4u << 20 : 0u;
        uint64_t exact = toks_scratch_bytes(&CTX, len, sf);
        /* the end flush against a guard page; extra bytes move the start's alignment */
        for (uint64_t extra = 0; extra < 64u; extra += (extra < 4u ? 1u : 13u)) {
            guard_buf g;
            uint8_t *scr = guard_alloc(&g, (size_t)(exact + extra), GUARD_END, 0);
            CHECK(scr != NULL, "guard_alloc");
            if (scr == NULL) { continue; }
            CHECK(toks_scratch_init(&CTX, scr, exact + extra, sf) == 0, "init %" PRIu64 "+%" PRIu64, len, extra);
            run_through(scr, len);
            check_layout(scr, exact + extra, len, sf, memo);
            guard_free(&g);
        }
        /* the start flush against (or off) a guard page: every alignment 0..63 */
        for (size_t mis = 0; mis < 64u; mis += 9u) {
            guard_buf g;
            uint8_t *scr = guard_alloc(&g, (size_t)exact, GUARD_START, mis);
            CHECK(scr != NULL, "guard_alloc");
            if (scr == NULL) { continue; }
            CHECK(toks_scratch_init(&CTX, scr, exact, sf) == 0, "init start+%zu", mis);
            run_through(scr, len);
            check_layout(scr, exact, len, sf, memo);
            guard_free(&g);
        }
    }

    /* a 2 MiB-aligned buffer: the header at its start, the ends and the cache right after; init clears
     * the counters and the cache whatever the buffer held */
    {
        uint64_t b = (toks_scratch_bytes(&CTX, 1000u, 0u) + 0x1FFFFFu) & ~(uint64_t)0x1FFFFFu;
        void *scr = guard_aligned_alloc(0x200000u, (size_t)b);
        if (scr != NULL) { memset(scr, 0xA5, (size_t)b); }
        CHECK(scr != NULL && toks_scratch_init(&CTX, scr, b, 0u) == 0, "init 2 MiB-aligned");
        toks_scratch *h = hdr_of(scr);
        CHECK((void *)h == scr && h->off_cache == TOKS_SCR_FIXED, "header + ends not first, the cache not right after");
        CHECK(h->hits_static == 0u && h->hits_cache == 0u && h->misses == 0u, "init left K5's counters");
        run_through(scr, 1000u);
        check_layout(scr, b, 1000u, 0u, 4u << 20);
        guard_aligned_free(scr);
    }

    /* never-initialized buffers, garbage inside, the end flush against a guard page, every start
     * alignment: refused with TOKS_E_SCRATCH (toks.h), and nothing past the end is read (before, the
     * header sat behind the 64 KiB cache, so a call read past the end of any buffer below ~64 KiB) */
    static const uint64_t SMALL[] = { 127, 128, 200, 4096, 65536 };
    for (size_t i = 0; i < sizeof SMALL / sizeof SMALL[0]; i++) {
        for (uint64_t extra = 0; extra < 64u; extra++) {
            uint64_t n = SMALL[i] + extra;
            guard_buf g;
            uint8_t *p = guard_alloc(&g, (size_t)n, GUARD_END, 0);
            CHECK(p != NULL, "guard_alloc");
            if (p == NULL) { continue; }
            for (uint64_t k = 0; k < n; k++) { p[k] = (uint8_t)guard_rng(&rng); }
            uint32_t o[4];
            CHECK(toks_encode(&CTX, "abc", 3u, 0u, o, 4u, p) == TOKS_E_SCRATCH &&
                  toks_pieces(&CTX, "abc", 3u, 0u, o, 4u, p) == TOKS_E_SCRATCH, "uninitialized %" PRIu64 " B", n);
            guard_free(&g);
        }
    }

    /* errors and binding */
    uint64_t b = toks_scratch_bytes(&CTX, 100u, 0u);
    uint8_t *m = (uint8_t *)calloc(1, (size_t)b * 2u + 128u);
    uint8_t *scr = m + 1;
    uint32_t out[8];
    const uint8_t text[200] = { 'a' };
    CHECK(toks_scratch_init(NULL, scr, b, 0u) == TOKS_E_ARG, "init ctx NULL");
    CHECK(toks_scratch_init(&CTX, NULL, b, 0u) == TOKS_E_ARG, "init scr NULL");
    CHECK(toks_scratch_init(&CTX, NULL, 0u, 0u) == TOKS_E_SCRATCH, "init NULL, 0");
    CHECK(toks_scratch_init(&CTX, scr, toks_scratch_bytes(&CTX, 0u, 0u) - 1u, 0u) == TOKS_E_SCRATCH, "init short");
    CHECK(toks_scratch_init(&CTX, scr, b, 0x1000u) == TOKS_E_ARG, "init flags");
    CHECK(toks_encode(&CTX, text, 1u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "uninitialized");
    CHECK(toks_encode(&CTX, text, 1u, 0u, out, 8u, NULL) == TOKS_E_SCRATCH, "NULL scratch");
    uint64_t b0 = toks_scratch_bytes(&CTX, 100u, TOKS_SCRATCH_MEMO_MIB(0));
    CHECK(toks_scratch_init(&CTX, scr, b0, 0u) == TOKS_E_SCRATCH && b == b0 + (4u << 20) &&
          toks_scratch_init(&CTX, scr, b0, TOKS_SCRATCH_MEMO_MIB(0)) == 0, "the default memo's 4 MiB");
    CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0, "init");
    uint64_t ml = hdr_of(scr)->max_len;
    CHECK(ml >= 100u && ml < 150u, "max_len %" PRIu64, ml);
    CHECK(toks_encode(&CTX, text, ml, 0u, out, 8u, scr) == (int64_t)ml + 2, "len == max_len");
    CHECK(toks_encode(&CTX, text, ml + 1u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "len > max_len");
    toks_ctx other = CTX;
    other.identity ^= 1u;
    CHECK(toks_encode(&other, text, 1u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "another ctx");
    CHECK(toks_pieces(&other, text, 1u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "another ctx (pieces)");
    memmove(m + 64, scr, (size_t)b);                    /* a moved scratch is not bound */
    CHECK(toks_encode(&CTX, text, 1u, 0u, out, 8u, m + 64) == TOKS_E_SCRATCH, "moved");
    CHECK(toks_scratch_init(&CTX, m + 64, b, 0u) == 0 && toks_encode(&CTX, text, 1u, 0u, out, 8u, m + 64) == 3,
          "rebound");
    /* argument errors come before the scratch */
    CHECK(toks_encode(NULL, text, 1u, 0u, out, 8u, scr) == TOKS_E_ARG, "ctx NULL");
    CHECK(toks_encode(&CTX, NULL, 1u, 0u, out, 8u, scr) == TOKS_E_ARG, "text NULL");
    CHECK(toks_encode(&CTX, NULL, 0u, 0u, out, 8u, m + 64) == 2, "text NULL, len 0");
    CHECK(toks_encode(&CTX, text, 1u, 0u, NULL, 1u, scr) == TOKS_E_ARG, "out NULL");
    CHECK(toks_encode(&CTX, text, 1u, 64u, out, 8u, scr) == TOKS_E_ARG, "unknown flag");
    CHECK(toks_encode(&CTX, text, 1u, 3u, out, 8u, scr) == TOKS_E_ARG, "mode 3");
    CHECK(toks_encode(&CTX, text, TOKS_MAX_TEXT + 1u, 0u, out, 8u, scr) == TOKS_E_LIMIT, "limit");
    CHECK(toks_pieces(&CTX, text, 1u, 64u, out, 8u, scr) == TOKS_E_ARG, "pieces unknown flag");
    CHECK(toks_encode(&CTX, text, 1u, TOKS_CONTINUATION, out, 8u, m + 64) == 3, "continuation accepted");
    free(m);
}

/* toks_scratch_init on a buffer that already holds a scratch (kernels.md §7): O(1), the cache region untouched,
 * the next epoch, which K5 gets as its cache_tag, also when the scratch moves to another context; any other
 * memory (another size, a broken magic, a header naming another address), and the last epoch: the cache zeroed
 * and epoch 1 */
static void test_rebind(void)
{
    uint64_t b = toks_scratch_bytes(&CTX, 300u, 0u);
    uint8_t *m = (uint8_t *)malloc((size_t)b + 128u);
    CHECK(m != NULL, "malloc");
    if (m == NULL) { return; }
    memset(m, 0xA5, (size_t)b + 128u);
    uint8_t *scr = m + 3;
    uint32_t out[8];
    CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0, "rebind: init");
    toks_scratch *h = hdr_of(scr);
    uint8_t *cache = toks_scr_at(h, h->off_cache);
    CHECK(h->epoch == 1u && cache[0] == 0u && cache[TOKS_CACHE_BYTES / 2u] == 0u && cache[TOKS_CACHE_BYTES - 1u] == 0u,
          "first init: epoch %" PRIu64 ", the cache not zeroed", h->epoch);
    CHECK(toks_encode(&CTX, "ab", 2u, 0u, out, 8u, scr) == 4 && k5_tag == 1u, "K5's tag %" PRIu64 " != epoch 1", k5_tag);
    for (uint64_t e = 2u; e < 6u; e++) {
        memset(cache, 0x5A, (size_t)TOKS_CACHE_BYTES);  /* stands for the entries K5 filled */
        h->hits_cache = 7u;
        CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0 && h->epoch == e && h->hits_cache == 0u && cache[0] == 0x5Au &&
              cache[TOKS_CACHE_BYTES - 1u] == 0x5Au, "re-init %" PRIu64 ": not O(1) (epoch %" PRIu64 ")", e, h->epoch);
        CHECK(toks_encode(&CTX, "ab", 2u, 0u, out, 8u, scr) == 4 && k5_tag == e, "K5's tag %" PRIu64 " != epoch", k5_tag);
    }
    toks_ctx other = CTX;
    other.identity ^= 2u;
    CHECK(toks_scratch_init(&other, scr, b, 0u) == 0 && h->epoch == 6u && h->identity == other.identity &&
          cache[0] == 0x5Au, "another context: not O(1)");
    CHECK(toks_encode(&CTX, "ab", 2u, 0u, out, 8u, scr) == TOKS_E_SCRATCH, "the old context still bound");
    h->epoch = TOKS_TAG_MAX;                            /* the last epoch: zeroed, epoch 1 */
    CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0 && h->epoch == 1u && cache[0] == 0u &&
          cache[TOKS_CACHE_BYTES - 1u] == 0u, "after the last epoch: not zeroed");
    for (int how = 0; how < 4; how++) {                 /* headers that do not name this scratch */
        CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0, "rebind: init");
        memset(cache, 0x5A, (size_t)TOKS_CACHE_BYTES);
        uint64_t bytes = b;
        if (how == 0) { h->magic ^= 1u; }
        if (how == 1) { h->base ^= 64u; }
        if (how == 2) { h->epoch = 0u; }
        if (how == 3) { bytes = b - 1u; }
        CHECK(toks_scratch_init(&CTX, scr, bytes, 0u) == 0 && h->epoch == 1u && cache[0] == 0u &&
              cache[TOKS_CACHE_BYTES - 1u] == 0u, "header case %d: the cache not zeroed", how);
    }
    free(m);
}

/* TOKS_SCRATCH_CACHE_MIB(n): n a power of two 4..128 (else TOKS_E_ARG) gives n / 2 MiB of short cache and n / 2 of
 * long cache (core.h); re-init with the same flags is O(1) for both (the long cache's generation moves), other
 * flags lay it out again; K5 gets the long cache only once the scratch is warm (TOKS_K5_WARM pieces) */
static void test_cache_mib(void)
{
    uint64_t b0 = toks_scratch_bytes(&CTX, 300u, 0u), b = toks_scratch_bytes(&CTX, 300u, TOKS_SCRATCH_CACHE_MIB(8));
    CHECK(b == b0 + (6u << 20) && toks_scratch_bytes(&CTX, 300u, TOKS_SCRATCH_CACHE_MIB(128)) == b0 + (126u << 20) &&
          toks_scratch_bytes(&CTX, 300u, TOKS_SCRATCH_CACHE_MIB(3)) == b0, "cache MiB: scratch bytes");
    uint8_t *m = (uint8_t *)malloc((size_t)b + 64u);
    CHECK(m != NULL, "malloc");
    if (m == NULL) { return; }
    uint8_t *scr = m + 5;
    static const uint32_t bad[] = { 1, 2, 3, 5, 6, 12, 129, 192, 255 };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_CACHE_MIB(bad[i])) == TOKS_E_ARG, "cache MiB %u accepted", bad[i]);
    }
    CHECK(toks_scratch_init(&CTX, scr, b, 1u << 21) == TOKS_E_ARG, "flag bit 21 accepted");
    memset(m, 0xA5, (size_t)b + 64u);
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_CACHE_MIB(8)) == 0, "cache MiB 8: init");
    toks_scratch *h = hdr_of(scr);
    uint8_t *lb = toks_scr_at(h, h->off_long);
    CHECK(h->cache_mib == 8u && h->off_long == h->off_cache + (4u << 20) && h->off_work == h->off_cache + (12u << 20) &&
          h->long_gen == 1u && h->long_pos == 0u && lb[0] == 0u && lb[(1u << 20) - 1u] == 0u && h->max_len >= 300u,
          "cache MiB 8: layout (the caches' 8 MiB, then the default memo's 4)");
    uint32_t out[8];
    CHECK(toks_encode(&CTX, "ab", 2u, 0u, out, 8u, scr) == 4 && k5_mask == TOKS_CACHE_MASK && k5_lc.buckets == NULL,
          "cache MiB 8, fresh: mask %" PRIu64 " (the first 2 MiB), no long cache", k5_mask);
    h->misses = TOKS_K5_WARM;                           /* warm */
    h->long_pos = 96u;
    CHECK(toks_encode(&CTX, "ab", 2u, 0u, out, 8u, scr) == 4 && k5_mask == (4u << 20) / 64u - 1u &&
          k5_lc.buckets == lb && k5_lc.arena == toks_scr_at(h, h->off_long + (1u << 20)) &&
          k5_lc.mask == (1u << 20) / 64u - 1u && k5_lc.arena_bytes == (3u << 20) && k5_lc.pos == 96u && k5_lc.gen == 1u,
          "cache MiB 8, warm: the long cache K5 got");
    memset(lb, 0x5A, 1u << 20);                         /* stands for the slots K5 filled */
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_CACHE_MIB(8)) == 0 && h->epoch == 2u && h->long_gen == 2u &&
          h->long_pos == 0u && lb[0] == 0x5Au, "cache MiB 8: re-init not O(1)");
    h->long_gen = 0xFFFFFFFFu;                          /* the last generation: zeroed */
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_CACHE_MIB(8)) == 0 && h->epoch == 1u && h->long_gen == 1u &&
          lb[0] == 0u, "cache MiB 8: the last generation not zeroed");
    memset(lb, 0x5A, 1u << 20);
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_CACHE_MIB(4)) == 0 && h->epoch == 1u && h->cache_mib == 4u &&
          h->off_long == h->off_cache + (2u << 20) && toks_scr_at(h, h->off_long)[0] == 0u, "cache MiB 8 -> 4: not laid out again");
    CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0 && h->epoch == 1u && h->cache_mib == 0u && h->off_long == 0u &&
          h->long_gen == 0u, "cache MiB 4 -> default");
    free(m);
}

/* the segment memo's budget (toks.h): flags 0 gives 4 MiB, TOKS_SCRATCH_MEMO_MIB(m) m MiB, TOKS_SCRATCH_MEMO_MIB(0)
 * (TOKS_SCRATCH_MEMO_SET alone) none, a 0.2.0 binary's raw m (no bit 20) m MiB; a first init zeroes its header and
 * slots, a re-init with the same budget is O(1) (the epoch moves), another budget lays the scratch out again;
 * wordpiece and unigram get none; a replay of a segment >= 256 B is the memo's on the default scratch (no K5 call)
 * and K5's on a scratch without one */
static void test_memo_budget(void)
{
    uint64_t b = toks_scratch_bytes(&CTX, 600u, 0u), mb = 1u, pos, hits;
    uint8_t *m = (uint8_t *)malloc((size_t)b + 64u);
    CHECK(m != NULL, "malloc");
    if (m == NULL) { return; }
    memset(m, 0xA5, (size_t)b + 64u);
    uint8_t *scr = m + 7;
    toks_scratch *h = hdr_of(scr);
    CHECK(toks_scratch_init(&CTX, scr, b, 0u) == 0 && toks_scr_memo(h, &mb) == toks_scr_at(h, h->off_cache + TOKS_CACHE_BYTES) &&
          mb == (4u << 20) && h->max_len >= 600u, "flags 0: a memo of %" PRIu64 " bytes", mb);
    uint8_t *mm = toks_scr_at(h, h->off_cache + TOKS_CACHE_BYTES);
    int zero = 1;
    for (uint64_t i = 0; i < 64u + (4u << 20) / 32u; i++) { zero &= mm[i] == 0u; }
    CHECK(zero && mm[64u + (4u << 20) / 32u] == 0xA5u, "flags 0, first init: the memo's header and slots, and no more");
    mm[64] = 0x5A;                                      /* stands for a slot a call wrote */
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_MEMO_MIB(4)) == 0 && h->epoch == 2u && mm[64] == 0x5Au &&
          toks_scratch_init(&CTX, scr, b, 4u) == 0 && h->epoch == 3u && mm[64] == 0x5Au,
          "flags 0 -> TOKS_SCRATCH_MEMO_MIB(4) -> a 0.2.0 binary's: not the same budget (O(1))");
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_MEMO_MIB(0)) == 0 && h->epoch == 1u &&
          toks_scr_memo(h, &mb) == NULL && mb == 0u && h->off_work == h->off_cache + TOKS_CACHE_BYTES,
          "TOKS_SCRATCH_MEMO_MIB(0): a memo of %" PRIu64 " bytes", mb);
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_MEMO_SET) == 0 && h->epoch == 2u && toks_scr_memo(h, &mb) == NULL,
          "TOKS_SCRATCH_MEMO_SET alone: a memo");
    CHECK(toks_scratch_init(&CTX, scr, b, TOKS_SCRATCH_MEMO_MIB(2)) == 0 && h->epoch == 1u &&
          toks_scr_memo(h, &mb) != NULL && mb == (2u << 20), "TOKS_SCRATCH_MEMO_MIB(2): %" PRIu64 " bytes", mb);
    toks_ctx wp = CTX, uni = CTX;                       /* their tables are never read by init */
    wp.wp = (const struct toks_wp_tables *)(const void *)&CTX;
    uni.uni = (const struct toks_uni *)(const void *)&CTX;
    CHECK(toks_scratch_init(&wp, scr, b, 0u) == 0 && toks_scr_memo(h, &mb) == NULL &&
          toks_scratch_init(&uni, scr, b, TOKS_SCRATCH_MEMO_MIB(4)) == 0 && toks_scr_memo(h, &mb) == NULL,
          "wordpiece / unigram: a memo");
    uint8_t text[600];
    for (size_t i = 0; i < sizeof text; i++) { text[i] = (uint8_t)('a' + i % 23u); }   /* one segment, no added token */
    static uint32_t o1[700], o2[700];
    for (int f = 0; f < 2; f++) {
        CHECK(toks_scratch_init(&CTX, scr, b, f == 0 ? 0u : TOKS_SCRATCH_MEMO_MIB(0)) == 0, "replay: init");
        int64_t n1 = toks_encode(&CTX, text, sizeof text, 0u, o1, 700u, scr);
        uint64_t c1 = k5_calls;
        int64_t n2 = toks_encode(&CTX, text, sizeof text, 0u, o2, 700u, scr);
        toks_scr_memo_ctr(h, &pos, &hits);
        CHECK(n1 == 602 && n2 == n1 && memcmp(o1, o2, 4u * 602u) == 0, "replay %d: %" PRId64 " then %" PRId64 " ids", f, n1, n2);
        CHECK(f == 0 ? (hits == 1u && pos != 0u && k5_calls == c1) : (hits == 0u && pos == 0u && k5_calls > c1),
              "replay %d: %" PRIu64 " memo hits, %" PRIu64 " record bytes, %" PRIu64 " K5 calls", f, hits, pos, k5_calls - c1);
    }
    free(m);
}

/* ================================================================ encode / pieces */

static uint32_t *g_out_end;      /* one big guard buffer: out = end - cap is flush against the guard */
static guard_buf g_out;

static int one_cap(const uint8_t *text, uint64_t len, uint32_t flags, int ids, void *scr, const vec *want, uint64_t cap)
{
    uint64_t n = want->n;
    uint32_t *out = cap ? g_out_end - cap : NULL;
    int64_t r = ids ? toks_encode(&CTX, text, len, flags, out, cap, scr)
                    : toks_pieces(&CTX, text, len, flags, out, cap, scr);
    uint64_t k = cap < n ? cap : n;
    int ok = r == (int64_t)n && (k == 0u || memcmp(out, want->v, (size_t)k * 4u) == 0);
    CHECK(ok, "%s flags %u len %" PRIu64 " cap %" PRIu64 ": got %" PRId64 " want %" PRIu64,
          ids ? "encode" : "pieces", flags, len, cap, r, n);
    return ok;
}

/* every capacity 0..n+1 (all_caps), else 0..3, n-3..n+1 and 40 random ones; out's end is flush
 * against a guard page, so a write at or past cap faults */
static void sweep(const uint8_t *text, uint64_t len, uint32_t flags, int ids, void *scr, vec *want, int all_caps)
{
    ref_run(text, len, flags, ids, want);
    uint64_t n = want->n;
    if (all_caps || n < 16u) {
        for (uint64_t cap = 0; cap <= n + 1u; cap++) { if (!one_cap(text, len, flags, ids, scr, want, cap)) { return; } }
        return;
    }
    for (uint64_t cap = 0; cap < 4u; cap++) { if (!one_cap(text, len, flags, ids, scr, want, cap)) { return; } }
    for (uint64_t cap = n - 3u; cap <= n + 1u; cap++) { if (!one_cap(text, len, flags, ids, scr, want, cap)) { return; } }
    for (int i = 0; i < 40; i++) {
        if (!one_cap(text, len, flags, ids, scr, want, 4u + guard_rng(&rng) % (n - 8u))) { return; }
    }
}

static void test_encode(void)
{
    uint64_t maxlen = 70000u;
    uint64_t sb = toks_scratch_bytes(&CTX, maxlen, 0u);
    void *scr = malloc((size_t)sb);
    CHECK(toks_scratch_init(&CTX, scr, sb, 0u) == 0, "init");
    uint8_t *g = guard_alloc(&g_out, 4u * 200000u, GUARD_END, 0);
    g_out_end = (uint32_t *)(void *)(g + 4u * 200000u);
    static const uint32_t FLAGS[] = { 0, 1, 2, 4, 5, 6, 8 };
    vec want = { 0, 0, 0 };

    static const char *FIXED[] = {
        "", "a", "Hello world", "<|eot|>", "a<|eot|>b", "<|eot|><|eot|>", "<|x|>xyz<|eot|>", "xyzxyz",
        "<|eo<|eot|>", "<|eot|", "x<|x|>y", "  \n\n  hi 123 4567 89", "don't I'M 'll", "h\xc3\xa9llo w\xc3\xb6rld \xe4\xbd\xa0 \xf0\x9f\x98\x80",
        "\xff\xfe abc \xe2\x82", "\r\n\r\n", "\t\t x", "<|x|><|eot|>xyz<|x|>", "xy<|eot|>z", "x<|x|>yz",
    };
    for (size_t i = 0; i < sizeof FIXED / sizeof FIXED[0]; i++) {
        for (size_t f = 0; f < sizeof FLAGS / sizeof FLAGS[0]; f++) {
            sweep((const uint8_t *)FIXED[i], strlen(FIXED[i]), FLAGS[f], 1, scr, &want, 1);
            sweep((const uint8_t *)FIXED[i], strlen(FIXED[i]), FLAGS[f], 0, scr, &want, 1);
        }
    }
    uint8_t *text = (uint8_t *)malloc((size_t)maxlen);
    for (int r = 0; r < 300; r++) {
        uint64_t len = gen_text(text, guard_rng(&rng) % 300u);
        uint32_t fl = FLAGS[guard_rng(&rng) % (sizeof FLAGS / sizeof FLAGS[0])];
        sweep(text, len, fl, 1, scr, &want, 1);
        sweep(text, len, fl, 0, scr, &want, 1);
    }
    /* many K3 rounds: thousands of pieces, tokens between them */
    static const uint64_t BIG[] = { 3000, 20000, 70000 };
    for (size_t i = 0; i < sizeof BIG / sizeof BIG[0]; i++) {
        uint64_t len = gen_text(text, BIG[i]);
        for (size_t f = 0; f < 3u; f++) {
            sweep(text, len, FLAGS[f], 1, scr, &want, 0);
            sweep(text, len, FLAGS[f], 0, scr, &want, 0);
        }
    }
    free(text);
    free(want.v);
    free(scr);
    guard_free(&g_out);
}

/* ================================================================ decode */

/* the maximal-subpart reference (unicode §3.9 / rust from_utf8_lossy), written differently from
 * api.c: an ill-formed position emits U+FFFD for the longest prefix of a well-formed sequence. */
static uint64_t ref_lossy(const uint8_t *b, uint64_t n, uint8_t *out)
{
    uint64_t i = 0, o = 0;
    while (i < n) {
        uint32_t w = toks_utf8_len(b + i, n - i);
        if (w != 0u) { memcpy(out + o, b + i, w); o += w; i += w; continue; }
        uint32_t need = b[i] >= 0xF0 ? 4u : b[i] >= 0xE0 ? 3u : 2u, k = 1;
        for (uint32_t m = 2; m < need && i + m <= n; m++) {
            uint8_t pad[4] = { 0x80, 0x80, 0x80, 0x80 };
            memcpy(pad, b + i, m);
            if (b[i] >= 0xC2 && b[i] <= 0xF4 && toks_utf8_len(pad, 4) == need) { k = m; } else { break; }
        }
        memcpy(out + o, "\xef\xbf\xbd", 3); o += 3; i += k;
    }
    return o;
}

static int hexb(const char *h, uint8_t *b)
{
    int n = 0;
    while (*h) {
        if (*h == ' ') { h++; continue; }
        unsigned v;
        sscanf(h, "%2x", &v);
        b[n++] = (uint8_t)v;
        h += 2;
    }
    return n;
}

static void test_decode(void)
{
    /* hf 0.23.2: Tokenizer.decode of gpt2's single-byte tokens, one per input byte (left: the bytes, right: what
     * hf returned); here id = byte */
    static const char *HF[][2] = {
        { "E2 82 AC", "E2 82 AC" }, { "E2 82", "EF BF BD" }, { "E2 82 41", "EF BF BD 41" },
        { "E0 80", "EF BF BD EF BF BD" }, { "ED A0 80", "EF BF BD EF BF BD EF BF BD" },
        { "F0 90 80", "EF BF BD" }, { "F0 90 80 41", "EF BF BD 41" },
        { "F4 90 80 80", "EF BF BD EF BF BD EF BF BD EF BF BD" }, { "C0 AF", "EF BF BD EF BF BD" },
        { "80", "EF BF BD" }, { "FF", "EF BF BD" }, { "F0 9F 98 80", "F0 9F 98 80" },
        { "61 F0 9F 98", "61 EF BF BD" }, { "C2", "EF BF BD" }, { "C2 41", "EF BF BD 41" },
        { "E1 80 E1 80 80", "EF BF BD E1 80 80" }, { "F1 80 80 E1 80 C0", "EF BF BD EF BF BD EF BF BD" },
        { "EF BF BD", "EF BF BD" }, { "F5 80", "EF BF BD EF BF BD" }, { "E0 A0 80", "E0 A0 80" },
        { "ED 9F BF", "ED 9F BF" }, { "F4 8F BF BF", "F4 8F BF BF" }, { "", "" },
        { "41 80 80 42", "41 EF BF BD EF BF BD 42" }, { "E2 28 A1", "EF BF BD 28 EF BF BD" },
        { "F0 28 8C BC", "EF BF BD 28 EF BF BD EF BF BD" },
        { "F8 A1 A1 A1 A1", "EF BF BD EF BF BD EF BF BD EF BF BD EF BF BD" },
    };
    uint8_t in[64], want[64], got[64], ref[256];
    uint32_t ids[64];
    for (size_t c = 0; c < sizeof HF / sizeof HF[0]; c++) {
        int ni = hexb(HF[c][0], in), nw = hexb(HF[c][1], want);
        for (int i = 0; i < ni; i++) { ids[i] = in[i]; }
        int64_t r = toks_decode(&CTX, ids, (uint64_t)ni, 0u, got, sizeof got);
        CHECK(r == nw && memcmp(got, want, (size_t)nw) == 0, "decode hf vector %s", HF[c][0]);
        CHECK(ref_lossy(in, (uint64_t)ni, ref) == (uint64_t)nw && memcmp(ref, want, (size_t)nw) == 0,
              "reference on hf vector %s", HF[c][0]);
    }
    /* multi-byte tokens, sequences across token ends, specials, holes */
    struct { uint32_t ids[6]; int n; uint32_t flags; const char *want; } T[] = {
        { { ID_EUR }, 1, 0, "\xe2\x82\xac" },
        { { ID_E2, 0xAC }, 2, 0, "\xe2\x82\xac" },
        { { ID_E2 }, 1, 0, "\xef\xbf\xbd" },
        { { ID_E2, 'A' }, 2, 0, "\xef\xbf\xbd" "A" },
        { { ID_E2, ID_E2 }, 2, 0, "\xef\xbf\xbd\xef\xbf\xbd" },
        { { ID_EOT, 'A', ID_X }, 3, 0, "<|eot|>A<|x|>" },
        { { ID_EOT, 'A', ID_X }, 3, TOKS_SKIP_SPECIAL, "A<|x|>" },
        { { ID_E2, ID_EOT, 0xAC }, 3, TOKS_SKIP_SPECIAL, "\xe2\x82\xac" },
        { { ID_E2, ID_EOT, 0xAC }, 3, 0, "\xef\xbf\xbd<|eot|>\xef\xbf\xbd" },
        { { ID_NIL, 'b', ID_NIL }, 3, 0, "b" },
        { { ID_XYZ }, 1, TOKS_SKIP_SPECIAL, "xyz" },
    };
    for (size_t c = 0; c < sizeof T / sizeof T[0]; c++) {
        size_t nw = strlen(T[c].want);
        int64_t r = toks_decode(&CTX, T[c].ids, (uint64_t)T[c].n, T[c].flags, got, sizeof got);
        CHECK(r == (int64_t)nw && memcmp(got, T[c].want, nw) == 0, "decode case %zu", c);
    }
    /* errors: nothing written */
    uint32_t bad[3] = { 'a', N_IDS, 'b' };
    memset(got, 0x77, sizeof got);
    CHECK(toks_decode(&CTX, bad, 3u, 0u, got, sizeof got) == TOKS_E_ID && got[0] == 0x77, "E_ID");
    CHECK(toks_decode(NULL, bad, 1u, 0u, got, 8u) == TOKS_E_ARG, "decode ctx NULL");
    CHECK(toks_decode(&CTX, NULL, 1u, 0u, got, 8u) == TOKS_E_ARG, "decode ids NULL");
    CHECK(toks_decode(&CTX, bad, 1u, 0u, NULL, 8u) == TOKS_E_ARG, "decode out NULL");
    CHECK(toks_decode(&CTX, bad, 1u, 4u, got, 8u) == TOKS_E_ARG, "decode flags");
    CHECK(toks_decode(&CTX, NULL, 0u, 0u, NULL, 0u) == 0, "decode empty");

    /* random id sequences: the reference over the concatenation, every capacity, guard-flushed */
    guard_buf g;
    uint8_t *gb = guard_alloc(&g, 4096u, GUARD_END, 0);
    uint8_t *gend = gb + 4096u;
    uint8_t cat[1024];
    for (int r = 0; r < 3000; r++) {
        uint32_t n = (uint32_t)(guard_rng(&rng) % 24u), fl = (uint32_t)(guard_rng(&rng) & 1u);
        uint64_t cl = 0u;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t x = guard_rng(&rng) % 10u;
            ids[i] = x < 6u ? (uint32_t)(0x80u + guard_rng(&rng) % 0x80u)
                   : x < 8u ? (uint32_t)(guard_rng(&rng) % 0x80u) : (uint32_t)(256u + guard_rng(&rng) % 6u);
            if (fl && ids[i] == ID_EOT) { continue; }
            uint32_t a = TOK_OFF[ids[i]], e = TOK_OFF[ids[i] + 1u];
            memcpy(cat + cl, TOK_BYTES + a, e - a);
            cl += e - a;
        }
        uint64_t nw = ref_lossy(cat, cl, ref);
        for (uint64_t cap = 0; cap <= nw + 1u; cap++) {
            uint8_t *o = cap ? gend - cap : NULL;
            int64_t res = toks_decode(&CTX, ids, n, fl, o, cap);
            uint64_t k = cap < nw ? cap : nw;
            CHECK(res == (int64_t)nw && (k == 0u || memcmp(o, ref, (size_t)k) == 0), "decode random %d cap %" PRIu64, r, cap);
        }
    }
    guard_free(&g);
}

/* the byte-level control of hf's by-string skip rule (docs/algorithms/spm_bpe.md §8.1) through compile.c, not the
 * hand context: gpt-2's <|endoftext|> is normalized=true with no normalizer, so its string is its content and
 * TOKS_SKIP_SPECIAL drops it, batch and stream (tests/data/compile/gpt2style.json, id 261; hf 0.23.2's strings) */
static void test_decode_compiled(void)
{
    static uint8_t js[1 << 16];
    FILE *f = fopen("tests/data/compile/gpt2style.json", "rb");
    size_t n = f != NULL ? fread(js, 1, sizeof js, f) : 0u;
    if (f != NULL) { fclose(f); }
    toks_ctx *ctx = NULL;
    CHECK(n > 0u && n < sizeof js && toks_load_mem_copy(&ctx, js, n, NULL) == 0, "gpt2style.json: read and load");
    if (ctx == NULL) { return; }
    static const struct { uint32_t ids[3]; uint64_t n; uint32_t flags; const char *want; } T[] = {
        { { 261 }, 1, 0, "<|endoftext|>" }, { { 261 }, 1, TOKS_SKIP_SPECIAL, "" },
        { { 259, 261, 65 }, 3, 0, "hello<|endoftext|>A" }, { { 259, 261, 65 }, 3, TOKS_SKIP_SPECIAL, "helloA" },
    };
    uint8_t got[64];
    for (size_t c = 0; c < sizeof T / sizeof T[0]; c++) {
        int64_t nw = (int64_t)strlen(T[c].want);
        CHECK(toks_decode(ctx, T[c].ids, T[c].n, T[c].flags, got, sizeof got) == nw && memcmp(got, T[c].want, (size_t)nw) == 0,
              "gpt2style decode case %zu", c);
        toks_stream st;                                 /* one id a push, then the flush */
        toks_stream_init(ctx, &st, T[c].flags);
        int64_t w = 0;
        for (uint64_t i = 0; i < T[c].n && w >= 0; i++) {
            int64_t r = toks_stream_push(ctx, &st, T[c].ids + i, 1u, got + w, sizeof got - (uint64_t)w);
            w = r < 0 ? r : w + r;
        }
        int64_t r = w < 0 ? w : toks_stream_flush(ctx, &st, got + w, sizeof got - (uint64_t)w);
        CHECK(r >= 0 && w + r == nw && memcmp(got, T[c].want, (size_t)nw) == 0, "gpt2style stream case %zu", c);
    }
    toks_unload(ctx);
}

/* ================================================================ token, info, stubs, sha-256 */

static void test_misc(void)
{
    uint64_t len = 99u;
    const uint8_t *p = toks_token(&CTX, ID_EUR, &len);
    CHECK(p != NULL && len == 3u && memcmp(p, "\xe2\x82\xac", 3) == 0, "token");
    CHECK(toks_token(&CTX, 'q', &len) != NULL && len == 1u, "byte token");
    CHECK(toks_token(&CTX, ID_NIL, &len) == NULL && len == 0u, "token without a string");
    CHECK(toks_token(&CTX, N_IDS, &len) == NULL && len == 0u, "token beyond the table");
    CHECK(toks_token(NULL, 1u, &len) == NULL && len == 0u, "token ctx NULL");
    CHECK(toks_token(&CTX, 'q', NULL) != NULL, "token len NULL");

    toks_info info;
    memset(&info, 0xFF, sizeof info);
    info.size = 4u;
    CHECK(toks_get_info(&CTX, &info) == TOKS_E_ARG, "info size");
    info.size = (uint32_t)sizeof info;
    CHECK(toks_get_info(&CTX, &info) == 0, "info");
    CHECK(info.abi_major == TOKS_ABI_MAJOR && info.n_ids == N_IDS && info.n_added == N_ADDED &&
          info.tier == TOKS_TIER_SCALAR && info.algorithm == TOKS_ALGO_BPE_BYTELEVEL &&
          info.max_text == TOKS_MAX_TEXT && strcmp(info.name, "hand") == 0 && info.image_sha256[0] == 0, "info fields");
    CHECK(info.cpu_features == CTX.cpu_features && cpu_calls == 0u,
          "info.cpu_features %#" PRIx64 " (want the load-time word), %" PRIu64 " cpu probes after load (want 0)",
          info.cpu_features, cpu_calls);

    uint64_t offs[4];
    CHECK(toks_split_points(&CTX, "a", 1u, 0u, 2u, offs, 4u, NULL) == 0, "split: one byte has no cut (test_split)");
    /* stream decode: tests/c/test_stream.c */

    static const struct { const char *in; const char *hex; } SHA[] = {
        { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    };
    for (size_t i = 0; i < sizeof SHA / sizeof SHA[0]; i++) {
        uint8_t d[32];
        char hex[65];
        toks_sha256((const uint8_t *)SHA[i].in, strlen(SHA[i].in), d);
        for (int k = 0; k < 32; k++) { snprintf(hex + 2 * k, 3, "%02x", d[k]); }
        CHECK(strcmp(hex, SHA[i].hex) == 0, "sha256(%s) = %s", SHA[i].in, hex);
    }
    /* the crc32c table every table hash uses (alloc.c) against layout.h's bitwise definition: the
     * asm tiers hash with the instruction, so a wrong entry splits c-built tables from asm probes */
    for (uint32_t i = 0; i < 256u; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) { c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u))); }
        CHECK(TOKS_CRC32C_TAB[i] == c, "crc32c table[%u] = %08x, want %08x", i, TOKS_CRC32C_TAB[i], c);
    }
    uint64_t seed = 7u;
    for (int i = 0; i < 100000; i++) {
        uint32_t crc = (uint32_t)guard_rng(&seed);
        uint64_t v = (guard_rng(&seed) << 33) ^ (guard_rng(&seed) << 9) ^ guard_rng(&seed);
        CHECK(toks_crc32c_u64(crc, v) == toks_crc32c_u64_ref(crc, v), "crc32c_u64(%08x, %016" PRIx64 ")", crc, v);
    }

    /* the byte-level alphabet against gpt-2's bytes_to_unicode built the python way: the printable
     * ranges map to themselves, every other byte, in byte order, to 256 + n (compile's token bytes,
     * and so decode, go through toks_char_byte) */
    uint32_t cs[256];
    int keep[256] = { 0 };
    for (uint32_t b = '!'; b <= '~'; b++) { keep[b] = 1; }
    for (uint32_t b = 0xA1u; b <= 0xACu; b++) { keep[b] = 1; }
    for (uint32_t b = 0xAEu; b <= 0xFFu; b++) { keep[b] = 1; }
    for (uint32_t b = 0, n = 0; b < 256u; b++) { cs[b] = keep[b] ? b : 256u + n++; }
    for (uint32_t b = 0; b < 256u; b++) {
        CHECK(toks_char_byte(cs[b]) == (int32_t)b, "char_byte(U+%04X) = %d, want %u", cs[b], toks_char_byte(cs[b]), b);
    }
    CHECK(cs[0x20] == 0x120u && cs[0x01] == 0x101u && cs[0xAD] == 0x143u, "the table itself");
    for (uint32_t cp = 0; cp < 0x400u; cp++) {
        int is = 0;
        for (uint32_t b = 0; b < 256u; b++) { is |= cs[b] == cp; }
        CHECK(is || toks_char_byte(cp) == -1, "char_byte(U+%04X) should be -1", cp);
    }

    uint8_t s1[32] = { 1 }, s2[32] = { 2 };
    CHECK(toks_ctx_identity(s1) != toks_ctx_identity(s2) && toks_ctx_identity(s1) != 0u, "identity");
}

int main(void)
{
    build_ctx();
    test_formula();
    test_scratch();
    test_rebind();
    test_cache_mib();
    test_memo_budget();
    test_encode();
    test_decode();
    test_misc();
    test_decode_compiled();                             /* a load: after test_misc's count of cpu probes */
    printf("test_api: %ld checks, %d failures (k5 stand-in calls %" PRIu64 ")\n", checks, failures, k5_calls);
    return failures != 0;
}
