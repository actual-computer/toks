/*
 * tests/bpe/bench_neon.c: K6 and K5, the c twins against the neon tier, on real-text pieces (the rent rule,
 * SPEC §10.5: an asm kernel ships only if >= 10% faster than its c twin). NOT part of `make test`.
 *
 *   bench_neon <pieces.bin> [rounds]
 *
 * pieces.bin is tests/bpe/gen.py's stream format (tests/bpe/check.c reads the same): the model header, then one
 * record per pre-tokenizer piece of the text IN TEXT ORDER (repeats included) with hf 0.23.2's ids for it. Made
 * from tests/bpe/gen.py's own functions (repo root, a pinned tokenizer):
 *
 *   uv run --with tokenizers==0.23.2 python - TOKENIZER_JSON TEXT... > pieces.bin <<'EOF'
 *   import sys; sys.path.insert(0, "tests/bpe"); import gen
 *   tok, texts = sys.argv[1], sys.argv[2:]; gen.init(tok); out, memo = sys.stdout.buffer, {}
 *   out.write(gen.model_header(tok))
 *   for tp in texts:
 *       for p, _ in gen.TK.pre_tokenizer.pre_tokenize_str(open(tp, "rb").read().decode("utf-8", "replace")):
 *           r = gen.raw(p)
 *           if r:
 *               if r not in memo: memo[r] = gen.record(r)
 *               out.write(memo[r])
 *   out.write(b"\xff\xff\xff\xff")
 *   EOF
 *
 * Build against the library (the c twins at the library's flags), e.g. on a lab host:
 *   BUILD_DIR=build/bench make -s -j8 lib && clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra \
 *     -Werror -Iinclude -Isrc/core -Isrc/platform -Itests/common -o build/bench/bench_neon tests/bpe/bench_neon.c \
 *     build/bench/libtoks.a
 *   taskset -c 3 build/bench/bench_neon pieces.bin
 *
 * The tables come from the stream's model through toks_bpe_build (what toks_load builds; check.c proves the
 * stream's model equals toks_config_parse + toks_compile's on every file they accept).
 *
 * Exactness, outside the timers (SPEC §12.4), both tiers: K6 on every piece == hf's ids, n_out and merges
 * equal, except where kernels.md §6's K5 / K6 CONTRACT lets K6 alone differ (tests/bpe/check.c's rule, counted on
 * the exact K6 line: under ignore_merges with TOKS_TF_PROBE_LONG, a 2..15-byte model token its own merges do not
 * rebuild, answered by K5's words; there both tiers must give the c twin's merges); K5 over the whole text in
 * TOKS_CHUNK_PIECES calls, cold then warm: the ids == hf's, and per call n_out, hits_static, hits_cache, misses
 * equal, and the two caches byte-identical after each pass.
 *
 * Cells, each kernel timed `rounds` times (default 5) in abba order against its twin, best and median:
 *   K6 all      every piece, in text order (no shortcut in front)
 *   K6 miss     the pieces K5 hands K6 without a dynamic cache: > 15 bytes, or 2..15 bytes not in words
 *   K5 cold     the whole text in TOKS_CHUNK_PIECES calls, the dynamic cache (TOKS_CACHE_BUCKETS) zeroed first
 *   K5 warm     the next pass over the same text (the cache as the cold pass left it)
 * Ids land in one contiguous uint32 array (SPEC §12.2). Rates are pieces/s and MB/s of piece bytes.
 *
 * Bytes moved per input byte (maintainer doctrine, speed) are a model, printed next to the rates: perf counters are not
 * readable on the lab hosts. Every count in it is exact (the run's counters, merges and lengths); the model is
 * which memory each count touches. Both tiers read the piece in place (no key copy, no piece copy):
 *   K6   the piece (1 B/B), its merge state (short: 4 * L16 + 16 len, long: 32 len, the work area: L1),
 *        merge-table lines (<= 2 probes of one 64-byte bucket per merge), bytepair entries (one 4-byte read
 *        per byte pair: <= one line each)
 *   K5   the text (1 B/B), ends (4 B/piece), ids out (4 B/id), one cache line per piece of 2..15 bytes, the
 *        words table's two lines per cache miss among those (neon reads both), plus K6 on the misses
 */
#define _POSIX_C_SOURCE 200809L
#include "../../src/core/bpe.h"
#include "bpe_words.h"
#include "../../src/core/kernels.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t toks_k6_bpe_neon(const toks_tables *t, toks_k6_args *a);
uint64_t toks_k5_encode_neon(const toks_tables *t, toks_k5_args *a);

typedef uint64_t (*k6_fn)(const toks_tables *, toks_k6_args *);
typedef uint64_t (*k5_fn)(const toks_tables *, toks_k5_args *);

static void *xalloc(uint64_t n)
{
    void *p = aligned_alloc(64, (n + 63) & ~(uint64_t)63);
    if (p == NULL) { fprintf(stderr, "out of memory (%" PRIu64 " bytes)\n", n); exit(2); }
    return p;
}

static FILE *IN;
static int rd(void *p, size_t n) { return fread(p, 1, n, IN) == n; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void loadavg(const char *when)
{
    FILE *f = fopen("/proc/loadavg", "r");
    char buf[128] = "n/a";
    if (f != NULL) {
        if (fgets(buf, sizeof buf, f) == NULL) { strcpy(buf, "n/a"); }
        fclose(f);
        buf[strcspn(buf, "\n")] = 0;
    }
    printf("load %s: %s\n", when, buf);
}

/* ---- the run ---- */
static toks_tables T;
static uint8_t *TEXT, *WORK, *CACHE[2];
static uint32_t *ENDS, *WANT, *OUT, *K6OUT, *MISS;
static uint64_t *WOFF, NP, TLEN, NW, NMISS, MISSBYTES, MAXLEN;
static uint32_t *MRG;                                   /* merges per piece (K6) */
static uint64_t NPROBE, K5HC[2];                        /* pieces of 2..15 bytes; neon cache hits cold / warm */
static int bad;

static uint64_t k6_state(uint64_t len)                  /* K6's merge state for a piece of len bytes */
{
    return len <= 128 ? 4 * ((len + 15) & ~(uint64_t)15) + 16 * len : 32 * len;
}

/* the K6 traffic model over the pieces idx[0, n) (idx NULL: every piece), per input byte */
static void k6_model(const char *cell, const uint32_t *idx, uint64_t n)
{
    uint64_t b = 0, st = 0, mg = 0, bp = 0;
    for (uint64_t j = 0; j < n; j++) {
        uint64_t i = idx != NULL ? idx[j] : j, len = ENDS[i] - (i == 0 ? 0 : ENDS[i - 1]);
        b += len; st += k6_state(len); mg += MRG[i]; bp += len - 1;
    }
    double d = (double)b;
    printf("%-8s bytes moved per input byte (model): piece 1.00 (in place) + merge state %.1f + merge-table "
           "lines <= %.1f (%.2f merges/B) + bytepair lines <= %.1f\n", cell, (double)st / d, 128.0 * (double)mg / d,
           (double)mg / d, 64.0 * (double)bp / d);
}

static void k5_model(const char *cell, uint64_t hc)
{
    double d = (double)TLEN, words = T.words != NULL ? 128.0 * (double)(NPROBE - hc) : 0.0;
    printf("%-8s bytes moved per input byte (model): text 1.00 (in place) + ends %.2f + ids %.2f + cache lines %.1f"
           " + words lines %.1f = %.1f, plus K6 on the misses\n", cell, 4.0 * (double)NP / d, 4.0 * (double)NW / d,
           64.0 * (double)NPROBE / d, words / d, 1.0 + (4.0 * (double)(NP + NW) + 64.0 * (double)NPROBE + words) / d);
}

static uint64_t k6_run(k6_fn k6, const uint32_t *idx, uint64_t n)    /* idx NULL: every piece */
{
    uint64_t sum = 0;
    for (uint64_t j = 0; j < n; j++) {
        uint64_t i = idx != NULL ? idx[j] : j, s = i == 0 ? 0 : ENDS[i - 1];
        toks_k6_args a = { TEXT + s, ENDS[i] - s, K6OUT, WORK, TOKS_BPE_WORK_BYTES(MAXLEN), 0, 0, 0 };
        sum += k6(&T, &a);
    }
    return sum;
}

typedef struct { uint64_t n_out, hs, hc, ms; } k5_tot;

/* K5 over the whole text, TOKS_CHUNK_PIECES per call; log (when not NULL) gets every call's output fields. age:
 * the in hits_static of every call (0: a fresh scratch, TOKS_K5_WARM: a warm one, kernels.md §6) */
static k5_tot k5_run(k5_fn k5, uint8_t *cache, uint64_t *log, uint64_t age)
{
    k5_tot r = { 0, 0, 0, 0 };
    for (uint64_t c = 0; c < NP; c += TOKS_CHUNK_PIECES) {
        uint64_t n = NP - c < TOKS_CHUNK_PIECES ? NP - c : TOKS_CHUNK_PIECES, start = c == 0 ? 0 : ENDS[c - 1];
        toks_k5_args a = { TEXT, TLEN, ENDS + c, n, start, OUT + r.n_out, TLEN - start + 4, 0, cache,
                           TOKS_CACHE_MASK, WORK, TOKS_BPE_WORK_BYTES(MAXLEN), age, 0, 0, 1, NULL };
        uint64_t no = k5(&T, &a);
        a.hits_static -= age;
        if (log != NULL) {
            uint64_t *l = log + 5 * (c / TOKS_CHUNK_PIECES);
            l[0] = no; l[1] = a.n_out; l[2] = a.hits_static; l[3] = a.hits_cache; l[4] = a.misses;
        }
        r.n_out += a.n_out; r.hs += a.hits_static; r.hc += a.hits_cache; r.ms += a.misses;
    }
    return r;
}

/* ---- exactness ---- */
static int IGNORE_MERGES;                               /* the model's ignore_merges (the stream header) */

/* kernels.md §6's K5 / K6 CONTRACT (tests/bpe/check.c's rule): K6 alone may differ from hf only on a piece of
 * 2..15 bytes that is a model token whose own merges do not rebuild it (hf: [that token], K6 under
 * TOKS_TF_PROBE_LONG: the merges) and whose static words entry is hf's answer (K5 answers it before K6) */
static int by_words(uint64_t s, uint64_t len, const uint32_t *want, uint64_t nw)
{
    if (!IGNORE_MERGES || (T.flags & TOKS_TF_PROBE_LONG) == 0u || len < 2 || len > TOKS_KEY_MAXLEN || nw != 1) {
        return 0;
    }
    bpe_key k = bpe_key_at(TEXT + s, (uint32_t)len, 0, (uint32_t)len);
    const uint8_t *v = bpe_w3_probe(T.words, T.words_mask, bpe_key_hash(k), k);
    uint32_t wv[2], vid = 0;
    return bpe_vhash_find(&T, TEXT + s, (uint32_t)len, &vid) && vid == want[0] && v != NULL && bpe_w3_val(v, wv) == 1
           && wv[0] == vid;
}

static void check_k6(void)
{
    uint32_t *o2 = xalloc(4 * (MAXLEN + 4));
    uint64_t k6_bad = 0, k6_words = 0;
    for (uint64_t i = 0; i < NP; i++) {
        uint64_t s = i == 0 ? 0 : ENDS[i - 1], len = ENDS[i] - s, nw = WOFF[i + 1] - WOFF[i];
        toks_k6_args a = { TEXT + s, len, K6OUT, WORK, TOKS_BPE_WORK_BYTES(len), 7, 7, 7 };
        toks_k6_args b = { TEXT + s, len, o2, WORK, TOKS_BPE_WORK_BYTES(len), 9, 9, 7 };
        uint64_t na = toks_k6_bpe_c(&T, &a), nb = toks_k6_bpe_neon(&T, &b);
        const uint32_t *want = WANT + WOFF[i];
        if (!(na == nw && memcmp(K6OUT, want, 4 * nw) == 0) && by_words(s, len, want, nw)) {
            want = K6OUT;                               /* the c twin's merges: the neon tier must give them too */
            nw = na;
            k6_words++;
        }
        int ok = na == nw && nb == nw && a.n_out == na && b.n_out == nb && a.merges == b.merges && b.rsv == 7
                 && memcmp(K6OUT, want, 4 * nw) == 0 && memcmp(o2, want, 4 * nw) == 0;
        if (!ok && k6_bad++ < 10) { fprintf(stderr, "K6 MISMATCH at piece %" PRIu64 " (%" PRIu64 " bytes)\n", i, len); }
        MRG[i] = (uint32_t)b.merges;
    }
    printf("exact K6: %" PRIu64 " pieces, c and neon == hf (%" PRIu64 " by K5's words: the merges, the CONTRACT), n_out"
           " / merges equal: %s\n", NP, k6_words, k6_bad ? "FAIL" : "ok");
    bad |= k6_bad != 0;
    free(o2);
}

static void check_k5(void)
{
    uint64_t calls = (NP + TOKS_CHUNK_PIECES - 1) / TOKS_CHUNK_PIECES;
    uint64_t *lc = xalloc(40 * calls), *ln = xalloc(40 * calls);
    for (int pass = 0; pass < 4; pass++) {
        uint64_t age = pass >= 2 ? (uint64_t)TOKS_K5_WARM : 0u;   /* passes 2, 3: a warm scratch */
        if (pass == 0 || pass == 2) {
            memset(CACHE[0], 0, TOKS_CACHE_BUCKETS * 64);
            memset(CACHE[1], 0, TOKS_CACHE_BUCKETS * 64);
        }
        k5_tot rc = k5_run(toks_k5_encode_c, CACHE[0], lc, age);
        int ids_c = rc.n_out == NW && memcmp(OUT, WANT, 4 * NW) == 0;
        k5_tot rn = k5_run(toks_k5_encode_neon, CACHE[1], ln, age);
        int ids_n = rn.n_out == NW && memcmp(OUT, WANT, 4 * NW) == 0;
        int fields = memcmp(lc, ln, 40 * calls) == 0, caches = memcmp(CACHE[0], CACHE[1], TOKS_CACHE_BUCKETS * 64) == 0;
        printf("exact K5 %s: %" PRIu64 " calls, ids c %s neon %s, output fields %s, caches %s; static %" PRIu64
               " / cache %" PRIu64 " / miss %" PRIu64 "\n", (pass & 1) ? (pass > 1 ? "warm (warm scratch)" : "warm")
               : (pass > 1 ? "cold (warm scratch)" : "cold"), calls, ids_c ? "ok" : "FAIL",
               ids_n ? "ok" : "FAIL", fields ? "equal" : "DIFFER", caches ? "equal" : "DIFFER", rn.hs, rn.hc, rn.ms);
        bad |= !(ids_c && ids_n && fields && caches);
        if (pass < 2) { K5HC[pass] = rn.hc; }
    }
    free(lc);
    free(ln);
}

/* ---- timing ---- */
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void report(const char *cell, double *tc, double *tn, int rounds, uint64_t pieces, uint64_t bytes)
{
    qsort(tc, (size_t)rounds, sizeof *tc, cmp_d);
    qsort(tn, (size_t)rounds, sizeof *tn, cmp_d);
    double bc = tc[0], bn = tn[0], mc = tc[rounds / 2], mn = tn[rounds / 2];
    printf("%-8s c    best %8.2f ms  %7.2f Mpieces/s  %8.1f MB/s   median %8.2f ms\n", cell, 1e3 * bc,
           1e-6 * (double)pieces / bc, 1e-6 * (double)bytes / bc, 1e3 * mc);
    printf("%-8s neon best %8.2f ms  %7.2f Mpieces/s  %8.1f MB/s   median %8.2f ms   c/neon %.3fx (median %.3fx)\n",
           cell, 1e3 * bn, 1e-6 * (double)pieces / bn, 1e-6 * (double)bytes / bn, 1e3 * mn, bc / bn, mc / mn);
}

#define MAXR 64

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) { fprintf(stderr, "usage: bench_neon <pieces.bin> [rounds]\n"); return 2; }
    int rounds = argc == 3 ? atoi(argv[2]) : 5;
    if (rounds < 1 || rounds > MAXR) { fprintf(stderr, "rounds 1..%d\n", MAXR); return 2; }
    IN = fopen(argv[1], "rb");
    if (IN == NULL) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    /* ---- the model (gen.py's header), tables as toks_load builds them ---- */
    char magic[8];
    uint32_t hdr[4];
    if (!rd(magic, 8) || memcmp(magic, "TOKSBPE1", 8) != 0 || !rd(hdr, sizeof hdr)) { fprintf(stderr, "bad stream header\n"); return 2; }
    uint32_t n_vocab = hdr[0], n_ids = hdr[1], n_merges = hdr[2];
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
    IGNORE_MERGES = hdr[3] != 0u;
    T.n_ids = n_ids;
    T.tok_off = tok_off;
    T.tok_bytes = tok_bytes;
    uint64_t need = toks_bpe_tables_bytes(&cfg);
    toks_arena ar = { xalloc(need), need, 0 };
    int64_t r = toks_bpe_build(&T, &ar, &cfg);
    if (r != 0) { fprintf(stderr, "toks_bpe_build: %" PRId64 "\n", r); return 1; }

    /* ---- the pieces, in text order ---- */
    uint64_t tcap = 1u << 24, wcap = 1u << 22, pcap = 1u << 20;
    TEXT = malloc(tcap); WANT = malloc(4 * wcap); ENDS = malloc(4 * pcap); WOFF = malloc(8 * (pcap + 1));
    WOFF[0] = 0;
    for (;;) {
        uint32_t len = 0, n = 0;
        if (!rd(&len, 4)) { fprintf(stderr, "truncated stream\n"); return 2; }
        if (len == UINT32_MAX) { break; }
        if (len == 0) { fprintf(stderr, "empty piece\n"); return 2; }
        while (TLEN + len > tcap) { tcap *= 2; TEXT = realloc(TEXT, tcap); }
        if (NP + 1 >= pcap) { pcap *= 2; ENDS = realloc(ENDS, 4 * pcap); WOFF = realloc(WOFF, 8 * (pcap + 1)); }
        if (TEXT == NULL || ENDS == NULL || WOFF == NULL || !rd(TEXT + TLEN, len) || !rd(&n, 4) || n > len) {
            fprintf(stderr, "truncated stream\n");
            return 2;
        }
        while (NW + n > wcap) { wcap *= 2; WANT = realloc(WANT, 4 * wcap); }
        if (WANT == NULL || !rd(WANT + NW, 4 * (size_t)n)) { fprintf(stderr, "truncated stream\n"); return 2; }
        TLEN += len;
        NW += n;
        MAXLEN = len > MAXLEN ? len : MAXLEN;
        ENDS[NP++] = (uint32_t)TLEN;
        WOFF[NP] = NW;
    }
    fclose(IN);
    if (NP == 0) { fprintf(stderr, "no pieces\n"); return 2; }
    WORK = xalloc(TOKS_BPE_WORK_BYTES(MAXLEN));
    OUT = xalloc(4 * (TLEN + 4));
    K6OUT = xalloc(4 * (MAXLEN + 4));
    CACHE[0] = xalloc(TOKS_CACHE_BUCKETS * 64);
    CACHE[1] = xalloc(TOKS_CACHE_BUCKETS * 64);
    MISS = xalloc(4 * NP);
    MRG = xalloc(4 * NP);
    uint64_t one = 0;
    for (uint64_t i = 0; i < NP; i++) {
        uint64_t s = i == 0 ? 0 : ENDS[i - 1], len = ENDS[i] - s;
        uint8_t key[16];
        uint32_t val[4];
        one += len == 1;
        NPROBE += len >= 2 && len <= TOKS_KEY_MAXLEN;
        if (len == 1) { continue; }
        if (len <= TOKS_KEY_MAXLEN) {
            bpe_key_make(key, TEXT + s, (uint32_t)len);
            if (bpe_words_find(&T, key, val)) { continue; }
        }
        MISS[NMISS++] = (uint32_t)i;
        MISSBYTES += len;
    }
    printf("%s: n_vocab %u, merges %u, flags %#x (%s%s); %" PRIu64 " pieces, %" PRIu64 " bytes (%.2f B/piece, "
           "longest %" PRIu64 "), %" PRIu64 " ids; one-byte %" PRIu64 ", static misses %" PRIu64 " (%" PRIu64
           " bytes); cache %u buckets\n", argv[1], n_vocab, n_merges, T.flags,
           (T.flags & TOKS_TF_IDS_AS_RANK) ? "ids as rank" : "rank2id", (T.flags & TOKS_TF_IGNORE_MERGES) ? ", ignore_merges" : "",
           NP, TLEN, (double)TLEN / (double)NP, MAXLEN, NW, one, NMISS, MISSBYTES, TOKS_CACHE_BUCKETS);
    loadavg("start");

    check_k6();
    check_k5();
    if (bad) { printf("RESULT %s: FAIL (no timing)\n", argv[1]); return 1; }

    /* ---- timing: abba rounds ---- */
    double t[4][2][MAXR];
    for (int rd_ = 0; rd_ < rounds; rd_++) {
        for (int k = 0; k < 2; k++) {
            int tier = (rd_ & 1) ^ k;                           /* abba: c neon, neon c, ... */
            k6_fn k6 = tier ? toks_k6_bpe_neon : toks_k6_bpe_c;
            k5_fn k5 = tier ? toks_k5_encode_neon : toks_k5_encode_c;
            double t0 = now_s();
            k6_run(k6, NULL, NP);
            double t1 = now_s();
            k6_run(k6, MISS, NMISS);
            double t2 = now_s();
            memset(CACHE[tier], 0, TOKS_CACHE_BUCKETS * 64);
            double t3 = now_s();
            k5_run(k5, CACHE[tier], NULL, 0u);
            double t4 = now_s();
            k5_run(k5, CACHE[tier], NULL, 0u);                  /* warm: the cache as the cold pass left it */
            double t5 = now_s();
            t[0][tier][rd_] = t1 - t0;
            t[1][tier][rd_] = t2 - t1;
            t[2][tier][rd_] = t4 - t3;
            t[3][tier][rd_] = t5 - t4;
        }
    }
    report("K6 all", t[0][0], t[0][1], rounds, NP, TLEN);
    k6_model("K6 all", NULL, NP);
    report("K6 miss", t[1][0], t[1][1], rounds, NMISS, MISSBYTES);
    k6_model("K6 miss", MISS, NMISS);
    report("K5 cold", t[2][0], t[2][1], rounds, NP, TLEN);
    k5_model("K5 cold", K5HC[0]);
    report("K5 warm", t[3][0], t[3][1], rounds, NP, TLEN);
    k5_model("K5 warm", K5HC[1]);
    loadavg("end");
    printf("RESULT %s: ok\n", argv[1]);
    return 0;
}
