/* tools/bench/k5_sweep.c: K5 alone, one thread, over the pieces of a corpus: the dynamic cache's fill rule x
 * size x probe order (kernels.md §6), cold and warm, with K5's hit counters. The quick look that sizes
 * TOKS_CACHE_BUCKETS (kernels.md §7); not a SPEC §12 cell (best of reps, no abba pairs, no intervals).
 *
 *   k5_sweep <tokenizer> <label> <reps> <log2 buckets lo> <log2 buckets hi> <file>...
 *
 * Each file is cut in two at the first newline past its middle: the first halves, concatenated, are the
 * warm-up text W; the second halves the measured text M (same languages, disjoint text: SPEC §12.3's
 * warm-lang). K3 (the c twin) cuts each into pieces once, untimed. Every pass runs K5 over all of a text's
 * pieces in TOKS_CHUNK_PIECES rounds, as the driver does, every id written into one array (the driver's
 * direct path). Variants (same tables, same pieces, one process, compiled alike):
 *   master   7b0d938's toks_k5_encode_c, verbatim (counters added, not stored): the key built by memset +
 *            memcpy, hashed twice on a cache miss; a static answer fills the cache
 *   k6only   PR #52's rule: the in-place key, one hash, the cache filled by K6 answers only
 *   static1  k6only with the static table probed before the cache (the order is free under that rule)
 *   lib      the library's toks_k5_encode_c: the in-place key, one hash, static and K6 answers both fill
 *   asm      the library's asm K5 of this isa (toks_k5_encode_neon / _avx2, with its asm K6): the rent rule's
 *            pair for lib, paired in the same process on the same core (SPEC §10.5)
 * States per (variant, size), all timing a pass over M: cold = the cache zeroed; warm = right after the cold
 * pass (M again: an upper bound); lang = the cache zeroed, then an untimed pass over W (warm-lang). Each rep
 * runs every (size, variant) in those states; the best of reps per state is printed, with the counters of
 * the last rep. Every pass over M must write the ids of the first (count + fnv-1a), else the bench exits 1.
 * HUGE=1 (linux): the cache buffer is 2 MiB-aligned and madvise(MADV_HUGEPAGE)'d before its first touch.
 * Build (the library's own flags, so every variant compiles alike):
 *   cc -std=c17 -O3 -fno-strict-aliasing -fwrapv -fPIC -Iinclude -Isrc/core -Isrc/platform \
 *      -o build/k5_sweep tools/bench/k5_sweep.c build/<os>-<isa>/libtoks.a
 */
#if defined(__linux__)
#define _DEFAULT_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L
#include "bpe.h"
#include "../../tests/common/bpe_words.h"
#include "kernels.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

static uint64_t now_ns(void)
{
#if defined(__APPLE__)
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) { mach_timebase_info(&tb); }
    return mach_absolute_time() * tb.numer / tb.denom;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
}

/* ---- master (7b0d938): src/core/k5_c.c and bpe.h's bpe_words_find, renamed; its static lookup now reads today's
 * three-way words bucket (tests/common/bpe_words.h's bpe_words_find, the same 16-byte val out) ------------------- */

static int m_words_find(const toks_tables *t, const uint8_t key[16], uint32_t val[4])
{
    return bpe_words_find(t, key, val);
}

/* the words table's answer for key k (hash h) as a 16-byte val (count << 29 | id0, id1, 0, 0): 1 found, 0 absent */
static int words_val16(const uint8_t *words, uint64_t wmask, uint32_t h, bpe_key k, uint32_t val[4])
{
    const uint8_t *v = words != NULL ? bpe_w3_probe(words, wmask, h, k) : NULL;
    if (v == NULL) { return 0; }
    uint32_t ids[2];
    uint64_t n = bpe_w3_val(v, ids);
    bpe_val_pack_tag(val, ids, (uint32_t)n, 0u);
    return 1;
}

static void m_cache_fill(uint8_t *bucket, const uint8_t key[16], const uint32_t val[4])
{
    memcpy(bucket + 16, bucket, 16);
    memcpy(bucket + 48, bucket + 32, 16);
    memcpy(bucket, key, 16);
    memcpy(bucket + 32, val, 16);
}

static uint64_t k5_master(const toks_tables *t, toks_k5_args *a)
{
    uint64_t n_out = 0u, hits_static = 0u, hits_cache = 0u, misses = 0u;
    uint64_t s = a->start;
    for (uint64_t i = 0; i < a->n; i++) {
        const uint8_t *piece = a->text + s;
        uint64_t len = (uint64_t)a->ends[i] - s;
        uint32_t *o = a->out + n_out;
        s = a->ends[i];

        if (len == 1u) {
            o[0] = t->byte2id[piece[0]];
            n_out += 1u;
            hits_static += 1u;
            continue;
        }
        uint8_t key[16];
        uint32_t val[4];
        uint8_t *bucket = NULL;
        if (len <= (uint64_t)TOKS_KEY_MAXLEN) {
            bpe_key_make(key, piece, (uint32_t)len);
            if (a->cache != NULL) {
                bucket = a->cache + ((uint64_t)toks_key_hash(TOKS_HSEED, key) & a->cache_mask) * TOKS_BUCKET;
                uint32_t way = memcmp(bucket, key, 16) == 0 ? 0u : (memcmp(bucket + 16, key, 16) == 0 ? 1u : 2u);
                if (way < 2u) {
                    memcpy(val, bucket + 32u + way * 16u, 16);
                    (void)bpe_val_put((const uint8_t *)val, o);
                    n_out += bpe_val_count(val);
                    hits_cache += 1u;
                    continue;
                }
            }
            if (m_words_find(t, key, val)) {
                (void)bpe_val_put((const uint8_t *)val, o);
                n_out += bpe_val_count(val);
                hits_static += 1u;
                if (bucket != NULL) { m_cache_fill(bucket, key, val); }
                continue;
            }
        }
        toks_k6_args k6 = { piece, len, o, a->work, a->work_bytes, 0u, 0u, 0u };
        uint64_t n = toks_k6_bpe_c(t, &k6);
        misses += 1u;
        if (bucket != NULL && n <= 4u) {
            bpe_val_pack_tag(val, o, (uint32_t)n, 0u);
            m_cache_fill(bucket, key, val);
        }
        n_out += n;
    }
    a->n_out = n_out;
    a->hits_static += hits_static;
    a->hits_cache += hits_cache;
    a->misses += misses;
    return n_out;
}

/* ---- the new K5's body (src/core/k5_c.c) with the fill rule and probe order as parameters ------------------ */

static void cache_fill(uint8_t *bucket, bpe_key k, const uint32_t val[4])
{
    memcpy(bucket + 16, bucket, 16);
    memcpy(bucket + 48, bucket + 32, 16);
    memcpy(bucket, &k.lo, 8);
    memcpy(bucket + 8, &k.hi, 8);
    memcpy(bucket + 32, val, 16);
}

static inline uint64_t k5_body(const toks_tables *t, toks_k5_args *a, int fill_static, int static_first)
{
    const uint64_t tw = toks_tag_word(a->cache_tag);
    const uint8_t *text = a->text;
    const uint32_t *ends = a->ends;
    const uint64_t tlen = a->len, np = a->n, cmask = a->cache_mask, wbytes = a->work_bytes;
    uint32_t *out = a->out;
    uint8_t *cache = a->cache, *work = a->work;
    const uint32_t *byte2id = t->byte2id;
    const uint8_t *words = t->words;
    const uint64_t wmask = t->words_mask;
    uint64_t n_out = 0u, hits_static = 0u, hits_cache = 0u, misses = 0u;
    uint64_t s = a->start;
    for (uint64_t i = 0; i < np; i++) {
        uint64_t ps = s, len = (uint64_t)ends[i] - s;
        uint32_t *o = out + n_out;
        s = ends[i];
        if (len == 1u) {
            o[0] = byte2id[text[ps]];
            n_out += 1u;
            hits_static += 1u;
            continue;
        }
        uint8_t *bucket = NULL;
        bpe_key k = { 0u, 0u };
        if (len <= (uint64_t)TOKS_KEY_MAXLEN) {
            k = bpe_key_at(text, tlen, ps, len);
            uint32_t h = bpe_key_hash(k);
            if (cache != NULL) { bucket = cache + ((uint64_t)h & cmask) * TOKS_BUCKET; }
            if (static_first) {
                uint32_t sv[4];
                if (words_val16(words, wmask, h, k, sv)) { n_out += bpe_val_put((const uint8_t *)sv, o); hits_static += 1u; continue; }
                if (bucket != NULL) {
                    const uint8_t *v = bpe_cache_get(bucket, k, tw);
                    if (v != NULL) { n_out += bpe_val_put(v, o); hits_cache += 1u; continue; }
                }
            } else {
                if (bucket != NULL) {
                    const uint8_t *v = bpe_cache_get(bucket, k, tw);
                    if (v != NULL) { n_out += bpe_val_put(v, o); hits_cache += 1u; continue; }
                }
                uint32_t sv[4];
                if (words_val16(words, wmask, h, k, sv)) {
                    n_out += bpe_val_put((const uint8_t *)sv, o);
                    hits_static += 1u;
                    if (fill_static && bucket != NULL) {
                        sv[2] |= (uint32_t)tw;
                        sv[3] |= (uint32_t)(tw >> 32);
                        cache_fill(bucket, k, sv);
                    }
                    continue;
                }
            }
        }
        toks_k6_args k6 = { text + ps, len, o, work, wbytes, 0u, 0u, 0u };
        uint64_t n = toks_k6_bpe_c(t, &k6);
        misses += 1u;
        if (bucket != NULL && n <= 4u) {
            uint32_t val[4];
            bpe_val_pack_tag(val, o, (uint32_t)n, tw);
            cache_fill(bucket, k, val);
        }
        n_out += n;
    }
    a->n_out = n_out;
    a->hits_static += hits_static;
    a->hits_cache += hits_cache;
    a->misses += misses;
    return n_out;
}

static uint64_t k5_k6only(const toks_tables *t, toks_k5_args *a) { return k5_body(t, a, 0, 0); }
static uint64_t k5_static1(const toks_tables *t, toks_k5_args *a) { return k5_body(t, a, 0, 1); }

/* ---- the sweep ---------------------------------------------------------------------------------------------- */

typedef uint64_t (*k5fn)(const toks_tables *, toks_k5_args *);
#if defined(__aarch64__)
#define NV 5
static const char *const VNAME[NV] = { "master", "k6only", "static1", "lib", "asm" };
static const k5fn VFN[NV] = { k5_master, k5_k6only, k5_static1, toks_k5_encode_c, toks_k5_encode_neon };
#elif defined(__x86_64__)
#define NV 5
static const char *const VNAME[NV] = { "master", "k6only", "static1", "lib", "asm" };
static const k5fn VFN[NV] = { k5_master, k5_k6only, k5_static1, toks_k5_encode_c, toks_k5_encode_avx2 };
#else
#define NV 4
static const char *const VNAME[NV] = { "master", "k6only", "static1", "lib" };
static const k5fn VFN[NV] = { k5_master, k5_k6only, k5_static1, toks_k5_encode_c };
#endif

typedef struct corpus {
    const uint8_t *text;
    uint64_t len, np, longest;
    uint32_t *ends, *out;
    uint8_t *work;
    uint64_t work_bytes;
} corpus;

static uint64_t fnv(const uint32_t *v, uint64_t n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < n; i++) { h = (h ^ v[i]) * 0x100000001b3ull; }
    return h;
}

/* one pass of K5 over every piece; returns ns; ctr = the counters' increments; *ids = ids written */
static uint64_t pass(k5fn f, const toks_tables *t, const corpus *c, uint8_t *cache, uint64_t mask, uint64_t ctr[3],
                     uint64_t *ids)
{
    toks_k5_args a;
    memset(&a, 0, sizeof a);
    uint64_t start = 0, n_out = 0;
    uint64_t t0 = now_ns();
    for (uint64_t i = 0; i < c->np; i += TOKS_CHUNK_PIECES) {
        uint64_t n = c->np - i < TOKS_CHUNK_PIECES ? c->np - i : TOKS_CHUNK_PIECES;
        uint64_t end = c->ends[i + n - 1];
        a.text = c->text;
        a.len = c->len;
        a.ends = c->ends + i;
        a.n = n;
        a.start = start;
        a.out = c->out + n_out;
        a.room = end - start + 4u;
        a.n_out = 0;
        a.cache = cache;
        a.cache_mask = mask;
        a.work = c->work;
        a.work_bytes = c->work_bytes;
        a.cache_tag = 1u;
        n_out += f(t, &a);
        start = end;
    }
    uint64_t dt = now_ns() - t0;
    ctr[0] = a.hits_static;
    ctr[1] = a.hits_cache;
    ctr[2] = a.misses;
    *ids = n_out;
    return dt;
}

/* K3 rounds over text[0, len), untimed: c's pieces; *n1 / *nlong count the 1-byte and over-15-byte ones */
static int cut_pieces(const toks_tables *t, corpus *c, const uint8_t *text, uint64_t len, uint64_t *n1, uint64_t *nlong)
{
    memset(c, 0, sizeof *c);
    c->text = text;
    c->len = len;
    c->ends = malloc((size_t)(len + 1) * 4);
    c->out = malloc((size_t)(len + 8) * 4);
    if (c->ends == NULL || c->out == NULL) { fprintf(stderr, "malloc\n"); return 1; }
    uint64_t pos = 0, prev = 0;
    while (pos < len) {
        toks_k3_args k;
        memset(&k, 0, sizeof k);
        k.text = text;
        k.len = len;
        k.pos = pos;
        k.ends = c->ends + c->np;
        k.cap = TOKS_CHUNK_PIECES;
        uint64_t r = toks_k3_scan_cl100k_c(t, &k);
        if (r == 0 || k.pos <= pos) { fprintf(stderr, "K3 made no progress\n"); return 1; }
        c->np += r;
        pos = k.pos;
    }
    for (uint64_t i = 0; i < c->np; i++) {
        uint64_t l = c->ends[i] - prev;
        prev = c->ends[i];
        if (l > c->longest) { c->longest = l; }
        if (n1 != NULL) { *n1 += l == 1; }
        if (nlong != NULL) { *nlong += l > TOKS_KEY_MAXLEN; }
    }
    c->work_bytes = (TOKS_BPE_WORK_BYTES(c->longest) + 63u) & ~(uint64_t)63u;
    return 0;
}

static uint8_t *cache_alloc(uint64_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 2u << 20, (size_t)bytes) != 0) { return NULL; }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    const char *hg = getenv("HUGE");
    if (hg != NULL && hg[0] == '1') { (void)madvise(p, (size_t)bytes, MADV_HUGEPAGE); }
#endif
    memset(p, 0, (size_t)bytes);
    return (uint8_t *)p;
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: k5_sweep <tokenizer> <label> <reps> <log2 buckets lo> <log2 buckets hi> <file>...\n");
        return 2;
    }
    const char *label = argv[2];
    int reps = atoi(argv[3]);
    unsigned lo = (unsigned)atoi(argv[4]), hi = (unsigned)atoi(argv[5]);
    if (reps < 1 || lo < 1 || hi > 24 || lo > hi) { fprintf(stderr, "bad arguments\n"); return 2; }
    uint8_t *wbuf = NULL, *buf = NULL;
    uint64_t wn = 0, n = 0;
    for (int i = 6; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb");
        if (f == NULL) { perror(argv[i]); return 1; }
        fseek(f, 0, SEEK_END);
        uint64_t m = (uint64_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        uint8_t *fb = malloc((size_t)m + 1);
        if (fb == NULL || fread(fb, 1, (size_t)m, f) != (size_t)m) { fprintf(stderr, "read %s\n", argv[i]); return 1; }
        fclose(f);
        uint64_t cut = m / 2;
        while (cut < m && fb[cut] != '\n') { cut++; }
        cut = cut < m ? cut + 1 : 0;                         /* no newline past the middle: all of it is M */
        wbuf = realloc(wbuf, (size_t)(wn + cut + 16));
        buf = realloc(buf, (size_t)(n + (m - cut) + 16));
        if (wbuf == NULL || buf == NULL) { fprintf(stderr, "malloc\n"); return 1; }
        memcpy(wbuf + wn, fb, (size_t)cut);
        memcpy(buf + n, fb + cut, (size_t)(m - cut));
        wn += cut;
        n += m - cut;
        free(fb);
    }
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, argv[1], NULL) != 0) { fprintf(stderr, "toks_load %s\n", argv[1]); return 1; }
    const toks_tables *t = &ctx->t;
    if (t->tmpl != TOKS_TMPL_CL100K) { fprintf(stderr, "k5_sweep: cl100k-template tokenizers only\n"); return 1; }

    corpus c, cw;
    uint64_t n1 = 0, nlong = 0;
    if (cut_pieces(t, &c, buf, n, &n1, &nlong) != 0 || cut_pieces(t, &cw, wbuf, wn, NULL, NULL) != 0) { return 1; }
    uint64_t wb = c.work_bytes > cw.work_bytes ? c.work_bytes : cw.work_bytes;
    void *w = NULL;
    if (posix_memalign(&w, 64, (size_t)wb) != 0) { fprintf(stderr, "work\n"); return 1; }
    c.work = cw.work = w;
    c.work_bytes = cw.work_bytes = wb;
    uint8_t *cache = cache_alloc((uint64_t)TOKS_BUCKET << hi);
    if (cache == NULL) { fprintf(stderr, "cache\n"); return 1; }

    uint64_t ns = hi - lo + 1;
    uint64_t *best = calloc(ns * NV * 3, 8), *ctrs = calloc(ns * NV * 3 * 3, 8), *zero_ns = calloc(ns, 8);
    for (uint64_t i = 0; i < ns * NV * 3; i++) { best[i] = UINT64_MAX; }
    for (uint64_t i = 0; i < ns; i++) { zero_ns[i] = UINT64_MAX; }
    uint64_t ref_ids = 0, ref_fnv = 0;
    int have_ref = 0;
    for (int rep = 0; rep < reps; rep++) {
        for (uint64_t si = 0; si < ns; si++) {
            uint64_t buckets = 1ull << (lo + si), bytes = buckets * TOKS_BUCKET;
            for (int v = 0; v < NV; v++) {
                uint64_t z0 = now_ns();
                memset(cache, 0, (size_t)bytes);             /* toks_scratch_init's cost for this size */
                uint64_t zd = now_ns() - z0;
                if (zd < zero_ns[si]) { zero_ns[si] = zd; }
                for (int st = 0; st < 3; st++) {            /* cold, warm, lang */
                    uint64_t ctr[3], ids;
                    if (st == 2) {
                        memset(cache, 0, (size_t)bytes);
                        (void)pass(VFN[v], t, &cw, cache, buckets - 1u, ctr, &ids);
                    }
                    uint64_t d = pass(VFN[v], t, &c, cache, buckets - 1u, ctr, &ids);
                    uint64_t h = fnv(c.out, ids);
                    if (!have_ref) { ref_ids = ids; ref_fnv = h; have_ref = 1; }
                    if (ids != ref_ids || h != ref_fnv) {
                        fprintf(stderr, "k5_sweep: %s at %" PRIu64 " buckets (state %d) wrote other ids\n", VNAME[v], buckets, st);
                        return 1;
                    }
                    uint64_t at = (si * NV + (uint64_t)v) * 3 + (uint64_t)st;
                    if (d < best[at]) { best[at] = d; }
                    memcpy(ctrs + at * 3, ctr, sizeof ctr);
                }
            }
        }
    }
    printf("K5SWEEP label=%s bytes=%" PRIu64 " warm_bytes=%" PRIu64 " pieces=%" PRIu64 " one_byte=%.4f over15=%.4f ids=%" PRIu64
           " fnv=%016" PRIx64 " reps=%d huge=%s\n", label, n, wn, c.np, (double)n1 / (double)c.np, (double)nlong / (double)c.np,
           ref_ids, ref_fnv, reps, getenv("HUGE") ? getenv("HUGE") : "0");
    for (uint64_t si = 0; si < ns; si++) {
        uint64_t buckets = 1ull << (lo + si);
        for (int v = 0; v < NV; v++) {
            uint64_t at = (si * NV + (uint64_t)v) * 3;
            const uint64_t *k0 = ctrs + at * 3, *k1 = ctrs + (at + 1) * 3, *k2 = ctrs + (at + 2) * 3;
            double np = (double)c.np;
            printf("K5 label=%s variant=%s buckets=%" PRIu64 " kib=%" PRIu64 " cold_mbs=%.1f warm_mbs=%.1f lang_mbs=%.1f"
                   " cold_static=%.4f cold_cache=%.4f cold_k6=%.4f warm_static=%.4f warm_cache=%.4f warm_k6=%.4f"
                   " lang_static=%.4f lang_cache=%.4f lang_k6=%.4f zero_us=%.1f\n",
                   label, VNAME[v], buckets, buckets * TOKS_BUCKET / 1024, (double)n / ((double)best[at] * 1e-9) / 1e6,
                   (double)n / ((double)best[at + 1] * 1e-9) / 1e6, (double)n / ((double)best[at + 2] * 1e-9) / 1e6,
                   (double)k0[0] / np, (double)k0[1] / np, (double)k0[2] / np, (double)k1[0] / np, (double)k1[1] / np,
                   (double)k1[2] / np, (double)k2[0] / np, (double)k2[1] / np, (double)k2[2] / np, (double)zero_ns[si] / 1e3);
        }
    }
    toks_unload(ctx);
    return 0;
}
