/*
 * tests/k1/bench_k1.c: K1 (added-token find), an asm tier against the c twin: a differential fuzz and a
 * paired benchmark. A lab-host tool, not part of `make test` (maintainer doctrine: nothing heavy on the control-plane mac).
 *
 *   make lib    (the c twin as shipped, -O3; with the tier's .S in src/asm/<isa>/ the library has the tier too)
 *   clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
 *     -o build/bench_k1 tests/k1/bench_k1.c tests/common/guard.c tests/common/abicheck_<isa>.S \
 *     build/<os>-<isa>/libtoks.a          [-DK1_TIER=<symbol>: default the isa's tier, neon / avx2]
 *
 *   bench_k1 fuzz [-s seed] [-n calls] [-t dir]     random token sets of six families plus the real sets of
 *       gpt2 / llama3 / glm53 / gemma4 / qwen38 / qwen3 from dir (default $TOKS_TOKENIZER_CACHE, else
 *       ~/.cache/toks/tokenizers; a missing file is skipped and said so); texts built from token contents,
 *       prefixes, one-byte mutations, first bytes, shufti false positives, prose filler and random bytes, cut
 *       anywhere (tokens ending exactly at len, tokens running past it); placed with their end flush against a
 *       no-access page or their start at a page start + 0..63, the bytes around them poisoned with token bytes;
 *       the driver's resume loop (pos = next after a full batch) plus random pos in [0, len + 2], phases 0, 1, 2
 *       and 2^63, a random cap per call (1..20). Every call: n, next and m[0, n) == the c twin's, m[n, cap) not
 *       written, the inputs kept; every 8th call through
 *       toks_abicheck_call (callee-saved registers incl. x18 / v8-v15, win64 rdi rsi xmm6-15); a sample against
 *       the prose reference (k1_tab.h). Exit 0 iff no difference.
 *   bench_k1 bench [-p pairs] [-c chunk] [-m MiB] [-b cap] tok=PATH name=FILE ...   K1 the driver's way over
 *       each file (cut into chunks of -c bytes, default 4096, 0 = whole; every phase that has tokens: pos = 0,
 *       then pos = next after each full batch of -b matches, default 16 = TOKS_SEG_BATCH, until one is not
 *       full), c twin and tier alternating (c t t c), -p pairs (default 30)
 *       of passes over -m MiB (default 64, the file repeated); outputs compared outside the timer; reports
 *       median MB/s of each, the mean per-pair speedup with its 95% bootstrap interval, matches per MiB and
 *       the load average before / after.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c17 on linux */
#include "k1_tab.h"
#include "../common/guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef K1_TIER
#  if defined(__aarch64__) || defined(_M_ARM64)
#    define K1_TIER toks_k1_added_find_neon
#  else
#    define K1_TIER toks_k1_added_find_avx2
#  endif
#endif
uint64_t K1_TIER(const toks_tables *t, toks_k1_args *a);
#define STR2(x) #x
#define STR(x) STR2(x)

/* ------------------------------------------------------------------------------------------- rng */

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint64_t below(uint64_t n) { return n ? rnd() % n : 0u; }

/* --------------------------------------------------------------------------------- token sets */

#define MAXT 1024
static uint8_t pool[MAXT * 256];
static k1_tok toks[MAXT];

static int seen(uint32_t n, const uint8_t *s, uint32_t len)
{
    for (uint32_t i = 0; i < n; i++) {
        if (toks[i].len == len && memcmp(toks[i].s, s, len) == 0) { return 1; }
    }
    return 0;
}

/* one random token of a family into buf; returns its length (1..255) */
static uint32_t gen_token(int fam, uint8_t *buf)
{
    static const char *pre[] = { "<|", "<", "[", "<|im_", "</", "<\xEF\xBD\x9C", "<<", "\xE2\x96\x81" };
    static const char *suf[] = { "|>", ">", "]", "\xEF\xBD\x9C>", "", ">>" };
    static const uint8_t tiny[] = { 'a', 'b', '<', '|', '>' };
    static const uint8_t fpb[] = { 0x00, 0x3C, 0xBC, 0x7C, 0xFC, 0x80, 0xC3, 0xFF, 'a', 0x0C, 0x8C };
    uint32_t n = 0;
    switch (fam) {
    case 0: {                                   /* chat-template shapes with shared prefixes */
        const char *p = pre[below(8)], *s = suf[below(6)];
        memcpy(buf, p, strlen(p)); n = (uint32_t)strlen(p);
        uint32_t body = (uint32_t)below(14);
        for (uint32_t i = 0; i < body; i++) { buf[n++] = "abcdefghijklmnopqrstuvwxyz_0123456789"[below(37)]; }
        memcpy(buf + n, s, strlen(s)); n += (uint32_t)strlen(s);
        break;
    }
    case 1:                                     /* a tiny alphabet: nesting, overlaps, one-byte tokens */
        n = 1u + (uint32_t)below(5);
        for (uint32_t i = 0; i < n; i++) { buf[i] = tiny[below(5)]; }
        break;
    case 2:                                     /* first bytes that share shufti bits: false positives */
        n = 1u + (uint32_t)below(8);
        for (uint32_t i = 0; i < n; i++) { buf[i] = fpb[below(sizeof fpb)]; }
        break;
    case 3: {                                   /* long tokens with a shared prefix and late mismatches */
        uint32_t pl = 1u + (uint32_t)below(40);
        for (uint32_t i = 0; i < pl; i++) { buf[i] = (uint8_t)('<' + (i % 3u)); }
        n = pl + (uint32_t)below(256u - pl);
        for (uint32_t i = pl; i < n; i++) { buf[i] = (uint8_t)("xy"[below(2)]); }
        break;
    }
    case 4:                                     /* llama-3-like reserved tokens */
        n = (uint32_t)snprintf((char *)buf, 64, "<|reserved_special_token_%u|>", (unsigned)below(300));
        break;
    case 5:                                     /* nested runs of one byte (gemma's whitespace tokens): radix chains */
        n = 2u + (uint32_t)below(40);
        memset(buf, " \n\t="[below(4)], n);
        break;
    case 6:                                     /* DeepSeek-like <｜name｜> tokens sharing their first four bytes */
        n = (uint32_t)snprintf((char *)buf, 80, "<\xEF\xBD\x9C%s\xE2\x96\x81%u\xEF\xBD\x9C>",
                               (const char *[]){ "place", "User", "tool", "fim", "end" }[below(5)], (unsigned)below(200));
        break;
    default:                                    /* any bytes, any length */
        n = 1u + (uint32_t)(below(4) ? below(12) : below(255));
        for (uint32_t i = 0; i < n; i++) { buf[i] = (uint8_t)rnd(); }
        break;
    }
    return n;
}

/* a random token set: n tokens, unique contents, phases all-0 / all-1 / mixed; 0 tokens sometimes */
static uint32_t gen_set(void)
{
    uint32_t want = (uint32_t)(below(10) == 0 ? 0 : below(4) == 0 ? 1u + below(600) : 1u + below(40));
    int mix = (int)below(8), pmode = (int)below(3);
    uint32_t n = 0;
    uint64_t used = 0;
    for (uint32_t tries = 0; n < want && tries < 4u * want + 16u; tries++) {
        int fam = mix < 8 && below(3) ? mix : (int)below(8);
        uint8_t *b = pool + used;
        uint32_t len = gen_token(fam, b);
        if (len == 0u || len > TOKS_MAX_ADDED_BYTES || seen(n, b, len)) { continue; }
        toks[n].s = b;
        toks[n].len = len;
        toks[n].phase = (uint8_t)(pmode == 2 ? below(2) : (uint64_t)pmode);
        toks[n].special = (uint8_t)below(2);
        used += len;
        n++;
    }
    return n;
}

/* ------------------------------------------------------------------------------- the checker */

typedef struct {
    const toks_tables *t;
    const uint8_t *tok[2][MAXT];   /* contents per phase (for text generation) */
    uint32_t tlen[2][MAXT];
    uint32_t nt[2];
    uint8_t cand[256];             /* bytes the phase's shufti passes (true and false positives) */
    uint32_t ncand[2];
    uint8_t cbytes[2][256];
} view;

static void make_view(view *v, const toks_tables *t)
{
    memset(v, 0, sizeof *v);
    v->t = t;
    for (uint64_t e = 0; e < t->add_n && e < 4096u; e++) {
        const toks_added_entry *x = &t->add_entries[e];
        uint32_t p = x->phase;
        if (v->nt[p] < MAXT) {
            v->tok[p][v->nt[p]] = t->add_bytes + x->off;
            v->tlen[p][v->nt[p]] = x->len;
            v->nt[p]++;
        }
    }
    for (uint32_t p = 0; p < 2u; p++) {
        if (((t->add_phases >> p) & 1u) == 0u) { continue; }
        const uint8_t *lo = t->add_shufti + p * 32u, *hi = lo + 16;
        for (uint32_t b = 0; b < 256u; b++) {
            if ((lo[b & 15u] & hi[b >> 4]) != 0u) { v->cbytes[p][v->ncand[p]++] = (uint8_t)b; }
        }
    }
}

static uint64_t gen_text(const view *v, uint32_t ph, uint8_t *out, uint64_t L)
{
    static const char filler[] = " etaoinshrdlu \n,.ETAOIN";
    uint64_t n = 0, sparse = below(3);       /* token-dense text, rare tokens (1/30), very rare (1/500) */
    uint32_t o = ph ^ 1u;
    while (n < L) {
        uint64_t r = below(100);
        if (sparse != 0u && below(sparse == 1u ? 30u : 500u) != 0u) { r = 66u + below(26u); }
        uint8_t tmp[300];
        uint64_t k = 0;
        if (r < 20 && v->nt[ph]) {
            uint64_t j = below(v->nt[ph]);
            k = v->tlen[ph][j]; memcpy(tmp, v->tok[ph][j], k);
        } else if (r < 25 && v->nt[o]) {
            uint64_t j = below(v->nt[o]);
            k = v->tlen[o][j]; memcpy(tmp, v->tok[o][j], k);
        } else if (r < 40 && v->nt[ph]) {
            uint64_t j = below(v->nt[ph]);
            k = v->tlen[ph][j] > 1u ? 1u + below(v->tlen[ph][j] - 1u) : 1u;
            memcpy(tmp, v->tok[ph][j], k);
        } else if (r < 50 && v->nt[ph]) {
            uint64_t j = below(v->nt[ph]);
            k = v->tlen[ph][j]; memcpy(tmp, v->tok[ph][j], k);
            tmp[below(k)] ^= (uint8_t)(1u + below(255));
        } else if (r < 58 && v->nt[ph]) {
            uint64_t j = below(v->nt[ph]);
            k = 1u + below(4);
            memset(tmp, v->tok[ph][j][0], k);
        } else if (r < 66 && v->ncand[ph]) {
            k = 1u + below(3);
            for (uint64_t i = 0; i < k; i++) { tmp[i] = v->cbytes[ph][below(v->ncand[ph])]; }
        } else if (r < 92) {
            k = 1u + below(below(4) ? 12 : 200);
            for (uint64_t i = 0; i < k; i++) { tmp[i] = (uint8_t)filler[below(sizeof filler - 1u)]; }
        } else {
            k = 1u + below(16);
            for (uint64_t i = 0; i < k; i++) { tmp[i] = (uint8_t)rnd(); }
        }
        if (k > L - n) { k = L - n; }
        memcpy(out + n, tmp, k);
        n += k;
    }
    return n;
}

static uint64_t calls, abi_calls, ref_calls, matches, bad;

static void report_bad(const char *what, const toks_k1_args *in, uint64_t r1, const toks_k1_args *a1,
                       uint64_t r2, const toks_k1_args *a2)
{
    if (bad++ < 12) {
        uint64_t i = 0;
        while (i < r1 && i < r2 && i < in->cap && memcmp(&a1->m[i], &a2->m[i], sizeof a1->m[0]) == 0) { i++; }
        fprintf(stderr, "MISMATCH %s: len %" PRIu64 " pos %" PRIu64 " phase %" PRIu64 " cap %" PRIu64 " | c %" PRIu64
                " n %" PRIu64 " next %" PRIu64 " | other %" PRIu64 " n %" PRIu64 " next %" PRIu64 " | match %" PRIu64
                ": [%u, %u) e%u vs [%u, %u) e%u\n", what, in->len, in->pos, in->phase, in->cap, r1, a1->n, a1->next,
                r2, a2->n, a2->next, i, i < in->cap ? a1->m[i].start : 0u, i < in->cap ? a1->m[i].end : 0u,
                i < in->cap ? a1->m[i].entry : 0u, i < in->cap ? a2->m[i].start : 0u, i < in->cap ? a2->m[i].end : 0u,
                i < in->cap ? a2->m[i].entry : 0u);
        fprintf(stderr, "  text:");
        uint64_t s = in->pos > 8u ? in->pos - 8u : 0u;
        for (uint64_t i = s; i < in->len && i < s + 48u; i++) { fprintf(stderr, " %02x", in->text[i]); }
        fprintf(stderr, "\n");
    }
}

/* one call of both with a random cap, compared; returns the c twin's next (the resume point) when its batch is
 * full, else len + 1 (done) */
static uint64_t check(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t phase)
{
    static toks_k1_match ma[22], mb[22], mr[22];
    toks_k1_args a, b;
    memset(&a, 0xA5, sizeof a);
    memset(ma, 0xA5, sizeof ma);
    memset(mb, 0xA5, sizeof mb);
    a.text = text; a.len = len; a.pos = pos; a.phase = phase; a.cap = 1u + below(below(4) ? 20u : 3u);
    b = a;
    a.m = ma;
    b.m = mb;
    toks_k1_args in = a;
    uint64_t r1 = toks_k1_added_find_c(t, &a);
    uint64_t r2;
    if ((calls++ & 7u) == 7u) {
        uint64_t rep = 0;
        r2 = toks_abicheck_call((const void *)K1_TIER, (uint64_t)(uintptr_t)t, (uint64_t)(uintptr_t)&b, &rep);
        abi_calls++;
        if (rep != 0u) {
            fprintf(stderr, "ABI: callee-saved clobbered, report %#" PRIx64 "\n", rep);
            report_bad("abi", &in, r1, &a, r2, &b);
        }
    } else {
        r2 = K1_TIER(t, &b);
    }
    b.m = ma;                                   /* every other field must match */
    if (r1 != r2 || memcmp(&a, &b, sizeof a) != 0 || memcmp(ma, mb, sizeof ma) != 0) {
        b.m = mb;
        report_bad("tier", &in, r1, &a, r2, &b);
    }
    if ((calls & 15u) == 0u && t->add_n <= 64u && len <= 4096u) {
        toks_k1_args c = a;
        c.m = mr;
        uint64_t r3 = k1_ref_batch(t, text, len, pos, phase, a.cap, mr, &c.next);
        c.n = r3;
        ref_calls++;
        if (r3 != r1 || c.next != a.next || (r1 != 0u && memcmp(ma, mr, (size_t)r1 * sizeof ma[0]) != 0)) {
            report_bad("c twin vs reference", &in, r1, &a, r3, &c);
        }
    }
    matches += r1;
    return r1 == a.cap ? a.next : len + 1u;
}

#define MAXLEN 20000u
static guard_buf GE, GS;
static uint8_t *ge, *gs;   /* GUARD_END buffer: [ge, ge + MAXLEN) flush against the guard; GUARD_START: gs */
static uint8_t textbuf[MAXLEN];

static void poison(const view *v, uint32_t ph, uint8_t *p, uint64_t n)
{
    const uint8_t *s = v->nt[ph] ? v->tok[ph][0] : (const uint8_t *)"<|x|>";
    uint64_t sl = v->nt[ph] ? v->tlen[ph][0] : 5u;
    for (uint64_t i = 0; i < n; i++) { p[i] = s[i % sl]; }
}

static void run_text(const view *v, uint32_t ph, uint64_t L)
{
    uint64_t len = gen_text(v, ph, textbuf, L);
    uint8_t *p;
    if (below(2)) {
        p = ge + MAXLEN - len;
        uint64_t before = MAXLEN - len < 64u ? MAXLEN - len : 64u;
        poison(v, ph, p - before, before);
    } else {
        p = gs + below(64);
        poison(v, ph, gs, (uint64_t)(p - gs));
        uint64_t after = MAXLEN + 64u - (uint64_t)(p - gs) - len;
        poison(v, ph, p + len, after < 64u ? after : 64u);
    }
    memcpy(p, textbuf, len);
    const toks_tables *t = v->t;
    uint64_t phases[4] = { ph, ph ^ 1u, 2u, 1ull << 63 };
    for (int k = 0; k < (below(8) ? 2 : 4); k++) {
        uint64_t pos = 0;
        for (uint64_t guard = 0; guard <= len + 1u; guard++) {      /* the driver's resume loop */
            uint64_t e = check(t, p, len, pos, phases[k]);
            if (e > len) { break; }
            pos = e;
        }
        for (int r = 0; r < 4; r++) { check(t, p, len, below(len + 3u), phases[k]); }
    }
}

static uint64_t pick_len(void)
{
    uint64_t r = below(100);
    if (r < 50) { return below(141); }
    if (r < 80) { return below(301); }
    if (r < 95) { return 300u + below(1800); }
    return below(MAXLEN + 1u);
}

static int fuzz_main(int argc, char **argv)
{
    uint64_t want = 1000000, seed = 1;
    const char *dir = getenv("TOKS_TOKENIZER_CACHE");
    char dbuf[1024];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) { seed = strtoull(argv[++i], NULL, 0); }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) { want = strtoull(argv[++i], NULL, 0); }
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) { dir = argv[++i]; }
    }
    if (dir == NULL) {
        snprintf(dbuf, sizeof dbuf, "%s/.cache/toks/tokenizers", getenv("HOME") ? getenv("HOME") : ".");
        dir = dbuf;
    }
    rs = seed * 0x9E3779B97F4A7C15ull + 0x1234567ull;
    if (rs == 0u) { rs = 1u; }
    ge = guard_alloc(&GE, MAXLEN, GUARD_END, 0);
    gs = guard_alloc(&GS, MAXLEN + 64u, GUARD_START, 0);
    if (!ge || !gs) { fprintf(stderr, "guard_alloc failed\n"); return 2; }

    /* real sets: loaded once, revisited every 8th round */
    static const char *names[] = { "gpt2", "llama3", "glm53", "gemma4", "qwen38", "qwen3", "dsv3", "dsv4", "gemma3",
                                   "nemotron3-omni", "llama4" };   /* the last five: long h4 buckets (radix trees) */
    enum { NREAL = sizeof names / sizeof names[0] };
    toks_ctx *ctx[NREAL] = { 0 };
    int nreal = 0, real[NREAL];
    for (int i = 0; i < NREAL; i++) {
        char path[1200];
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        if (toks_load(&ctx[i], path, NULL) == 0) {
            const toks_tables *t = &((const struct toks_ctx *)ctx[i])->t;
            printf("real %s: %" PRIu64 " added tokens, phases %#" PRIx64 "\n", names[i], t->add_n, t->add_phases);
            real[nreal++] = i;
        } else {
            printf("SKIP real %s: not loadable from %s\n", names[i], dir);
        }
    }
    static view v;
    uint64_t rounds = 0;
    while (calls < want) {
        k1_tab tab;
        int use_real = nreal > 0 && (rounds & 7u) == 7u;
        if (use_real) {
            make_view(&v, &((const struct toks_ctx *)ctx[real[below((uint64_t)nreal)]])->t);
        } else {
            uint32_t n = gen_set();
            if (k1_tab_build(&tab, toks, n) != 0) { fprintf(stderr, "k1_tab_build failed\n"); return 2; }
            make_view(&v, &tab.t);
        }
        for (int k = 0; k < 24; k++) { run_text(&v, (uint32_t)below(2), pick_len()); }
        if (!use_real) { k1_tab_free(&tab); }
        rounds++;
    }
    for (int i = 0; i < NREAL; i++) { if (ctx[i]) { toks_unload(ctx[i]); } }
    guard_free(&GE);
    guard_free(&GS);
    printf("fuzz %s seed %" PRIu64 ": %" PRIu64 " calls (%" PRIu64 " through the abi checker, %" PRIu64
           " vs the reference), %" PRIu64 " rounds, %" PRIu64 " matches, %" PRIu64 " differences\n",
           STR(K1_TIER), seed, calls, abi_calls, ref_calls, rounds, matches, bad);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------------------------- bench */

typedef uint64_t (*k1_fn)(const toks_tables *t, toks_k1_args *a);

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double loadavg(void)
{
    double l = -1.0;
    FILE *f = fopen("/proc/loadavg", "r");
    if (f) { if (fscanf(f, "%lf", &l) != 1) { l = -1.0; } fclose(f); }
    return l;
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static uint8_t *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc(sz > 0 ? (size_t)sz : 1u);
    *n = (b && sz > 0) ? fread(b, 1, (size_t)sz, f) : 0u;
    fclose(f);
    return b;
}

static uint64_t bcap = 16;   /* -b: matches per call */

/* one pass the driver's way; folds every output into a checksum */
static uint64_t pass(k1_fn k, const toks_tables *t, const uint8_t *text, uint64_t n, uint64_t chunk,
                     uint64_t *nmatch)
{
    static toks_k1_match mm[256];
    uint64_t sum = 0, m = 0;
    for (uint64_t s = 0; s < n; s += chunk) {
        uint64_t len = n - s < chunk ? n - s : chunk;
        for (uint64_t ph = 0; ph < 2u; ph++) {
            if (((t->add_phases >> ph) & 1u) == 0u) { continue; }
            toks_k1_args a;
            a.text = text + s; a.len = len; a.pos = 0; a.phase = ph; a.m = mm; a.cap = bcap;
            for (;;) {
                uint64_t r = k(t, &a);
                for (uint64_t i = 0; i < r; i++) { sum = sum * 31u + mm[i].start + mm[i].end * 7u + mm[i].entry; }
                m += r;
                if (r < bcap) { break; }
                a.pos = a.next;
            }
        }
    }
    *nmatch = m;
    return sum;
}

static int bench_main(int argc, char **argv)
{
    uint64_t chunk = 4096, mib = 64;
    int pairs = 30;
    toks_ctx *ctx = NULL;
    const char *tokname = "?";
    int ninputs = 0;
    printf("bench_k1: c twin vs %s\n", STR(K1_TIER));
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) { pairs = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "-c") && i + 1 < argc) { chunk = strtoull(argv[++i], NULL, 0); continue; }
        if (!strcmp(argv[i], "-m") && i + 1 < argc) { mib = strtoull(argv[++i], NULL, 0); continue; }
        if (!strcmp(argv[i], "-b") && i + 1 < argc) {
            bcap = strtoull(argv[++i], NULL, 0);
            if (bcap < 1u || bcap > 256u) { fprintf(stderr, "-b: 1..256\n"); return 2; }
            continue;
        }
        char *eq = strchr(argv[i], '=');
        if (!eq) { fprintf(stderr, "bad argument %s\n", argv[i]); return 2; }
        *eq = 0;
        const char *name = argv[i], *path = eq + 1;
        if (!strcmp(name, "tok")) {
            if (ctx) { toks_unload(ctx); ctx = NULL; }
            if (toks_load(&ctx, path, NULL) != 0) { fprintf(stderr, "cannot load %s\n", path); return 2; }
            tokname = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
            continue;
        }
        if (!ctx) { fprintf(stderr, "tok=PATH first\n"); return 2; }
        const toks_tables *t = &((const struct toks_ctx *)ctx)->t;
        size_t fn = 0;
        uint8_t *f = slurp(path, &fn);
        if (!f || fn == 0u) { fprintf(stderr, "cannot read %s\n", path); return 2; }
        uint64_t n = mib << 20;
        uint8_t *text = (uint8_t *)malloc(n);
        for (uint64_t o = 0; o < n; o += fn) { memcpy(text + o, f, n - o < fn ? n - o : fn); }
        uint64_t ch = chunk ? chunk : n;
        uint64_t mc = 0, mt = 0;
        uint64_t sc = pass(toks_k1_added_find_c, t, text, n, ch, &mc);
        uint64_t st = pass(K1_TIER, t, text, n, ch, &mt);
        if (sc != st || mc != mt) {
            fprintf(stderr, "%s %s: outputs differ (c %" PRIu64 " matches, tier %" PRIu64 ")\n", tokname, name, mc, mt);
            return 1;
        }
        double l0 = loadavg();
        double *rc = (double *)malloc(sizeof(double) * (size_t)pairs * 2u), *rt = rc + pairs;
        double *ratio = (double *)malloc(sizeof(double) * (size_t)pairs);
        uint64_t sink = 0, dummy;
        for (int w = 0; w < 2; w++) { sink += pass(toks_k1_added_find_c, t, text, n, ch, &dummy); sink += pass(K1_TIER, t, text, n, ch, &dummy); }
        for (int r = 0; r < pairs; r++) {
            double t0, t1, xc, xt;
            if ((r & 1) == 0) {
                t0 = now(); sink += pass(toks_k1_added_find_c, t, text, n, ch, &dummy); t1 = now(); xc = t1 - t0;
                t0 = now(); sink += pass(K1_TIER, t, text, n, ch, &dummy); t1 = now(); xt = t1 - t0;
            } else {
                t0 = now(); sink += pass(K1_TIER, t, text, n, ch, &dummy); t1 = now(); xt = t1 - t0;
                t0 = now(); sink += pass(toks_k1_added_find_c, t, text, n, ch, &dummy); t1 = now(); xc = t1 - t0;
            }
            rc[r] = (double)n / xc / 1e6;
            rt[r] = (double)n / xt / 1e6;
            ratio[r] = xc / xt;
        }
        double l1 = loadavg();
        /* bootstrap the mean ratio */
        double mean = 0;
        for (int r = 0; r < pairs; r++) { mean += ratio[r]; }
        mean /= pairs;
        enum { NB = 2000 };
        static double bm[NB];
        uint64_t bs = 12345;
        for (int b = 0; b < NB; b++) {
            double s = 0;
            for (int r = 0; r < pairs; r++) {
                bs ^= bs << 13; bs ^= bs >> 7; bs ^= bs << 17;
                s += ratio[bs % (uint64_t)pairs];
            }
            bm[b] = s / pairs;
        }
        qsort(bm, NB, sizeof(double), cmp_dbl);
        qsort(rc, (size_t)pairs, sizeof(double), cmp_dbl);
        qsort(rt, (size_t)pairs, sizeof(double), cmp_dbl);
        printf("%-8s %-10s chunk %5" PRIu64 " cap %3" PRIu64 " %3" PRIu64 " MiB: c %8.1f MB/s  tier %8.1f MB/s  "
               "speedup %.3f [%.3f, %.3f]  matches/MiB %.1f  pairs %d  load %.2f -> %.2f  (sink %" PRIu64 ")\n",
               tokname, name, chunk, bcap, mib, rc[pairs / 2], rt[pairs / 2], mean, bm[NB / 40], bm[NB - NB / 40 - 1],
               (double)mc / (double)mib, pairs, l0, l1, sink & 1u);
        fflush(stdout);
        free(rc); free(ratio); free(text); free(f);
        ninputs++;
    }
    if (ctx) { toks_unload(ctx); }
    return ninputs ? 0 : 2;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "fuzz")) { return fuzz_main(argc, argv); }
    if (argc >= 2 && !strcmp(argv[1], "bench")) { return bench_main(argc, argv); }
    fprintf(stderr, "usage: bench_k1 fuzz [-s seed] [-n calls] [-t dir] | bench [-p pairs] [-c chunk] [-m MiB] "
            "[-b cap] tok=PATH name=FILE ...\n");
    return 2;
}
