/* tools/bench/spm_stages.c: the sentencepiece-style bpe path's stage profile (gemma 4 etc.), one thread.
 *
 *   spm_stages <tokenizer> <chunk_bytes|0> <reps> <file>...       (env SP_CACHE_MIB: TOKS_SCRATCH_CACHE_MIB)
 *
 * Linked against one of the library variants tools/bench/spm_stages.sh builds from spm_stages.py's patches of
 * src/core/spm_c.c (the shipping sources stay untouched): ship (as shipped), prof (counters + a timer around
 * toks_spm_encode and around every model call), scan (word() stubbed: the scan + flush loops alone), drv
 * (toks_spm_encode stubbed: the driver alone). Chunks as tools/bench/e2e.c cuts them. Per state, best of reps:
 *   cold  scratch initialized before every call (outside the timer)   pass  once per pass
 *   warm  the pass right after a pass over the same text
 * prints one line per state: wall ns, MB/s, and (prof) the counters of the best rep. */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

/* the prof variant's counters (spm_stages.py): 0 enc ticks, 1 enc calls, 2 model ticks, 3 model calls, 4 wide
 * hits, 5 short hits, 6 word() calls, 7 word() bytes, 8 one-char words, 9 8-byte ascii steps, 10 per-char steps,
 * 11 model calls on words > 15 B, 12 per-char steps with a pair-set probe, 13 static word-table hits */
__attribute__((weak)) uint64_t sp_cnt[16];

/* the words variant's record (spm_stages.py): while sp_on, each word as "len wp hex\n", each unit start as "U\n" */
static FILE *sp_out;
static int sp_on;
void sp_word(const uint8_t *p, uint64_t len, int wp);
void sp_unit(void);
void sp_word(const uint8_t *p, uint64_t len, int wp)
{
    if (!sp_on || sp_out == NULL) { return; }
    fprintf(sp_out, "%llu %d ", (unsigned long long)len, wp != 0);
    for (uint64_t i = 0; i < len; i++) { fprintf(sp_out, "%02x", p[i]); }   /* bound: len */
    fputc('\n', sp_out);
}
void sp_unit(void)
{
    if (sp_on && sp_out != NULL) { fputs("U\n", sp_out); }
}

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static inline uint64_t tick(void)
{
#if defined(__x86_64__)
    unsigned aux;
    return __rdtscp(&aux);
#else
    uint64_t v;
    __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
#endif
}

static double tick_ns(void)                         /* ns per tick */
{
#if defined(__aarch64__)
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return 1e9 / (double)f;
#else
    uint64_t a = now_ns(), ta = tick();
    while (now_ns() - a < 200000000ull) { }
    uint64_t b = now_ns(), tb = tick();
    return (double)(b - a) / (double)(tb - ta);
#endif
}

static double tick_pair(void)                       /* ticks of one back-to-back timer pair (min of many) */
{
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < 100000; i++) {
        uint64_t a = tick(), b = tick();
        if (b - a < best) { best = b - a; }
    }
    return (double)best;
}

typedef struct cell {
    const toks_ctx *ctx;
    const uint8_t *buf;
    uint64_t n, nc, sb, idcap;
    uint64_t *beg, *len;
    void *scr;
    uint32_t *ids;
} cell;

static uint32_t SFL;

static uint8_t *read_all(char *const *files, int nf, uint64_t *n_out)
{
    uint8_t *buf = NULL;
    uint64_t n = 0;
    for (int i = 0; i < nf; i++) {
        FILE *f = fopen(files[i], "rb");
        if (f == NULL) { perror(files[i]); exit(1); }
        fseek(f, 0, SEEK_END);
        long m = ftell(f);
        fseek(f, 0, SEEK_SET);
        buf = realloc(buf, n + (uint64_t)m + 1);
        if (fread(buf + n, 1, (size_t)m, f) != (size_t)m) { fprintf(stderr, "short read %s\n", files[i]); exit(1); }
        fclose(f);
        n += (uint64_t)m;
    }
    *n_out = n;
    return buf;
}

static uint64_t cut(cell *c, uint64_t chunk)        /* tools/bench/e2e.c's cut */
{
    uint64_t nmax = chunk ? c->n / chunk + 2 : 2, maxlen = 0, p = 0;
    c->beg = malloc(nmax * sizeof(uint64_t));
    c->len = malloc(nmax * sizeof(uint64_t));
    while (p < c->n) {
        uint64_t e = c->n;
        if (chunk != 0 && p + chunk < c->n) {
            e = p + chunk;
            while (e < c->n && c->buf[e - 1] != '\n') { e++; }
        }
        c->beg[c->nc] = p;
        c->len[c->nc] = e - p;
        if (e - p > maxlen) { maxlen = e - p; }
        c->nc++;
        p = e;
    }
    return maxlen;
}

static void enc(const cell *c, uint64_t k)
{
    int64_t m = toks_encode(c->ctx, c->buf + c->beg[k], c->len[k], 0, c->ids, c->idcap, c->scr);
    if (m < 0) { fprintf(stderr, "toks_encode: %lld at chunk %llu\n", (long long)m, (unsigned long long)k); exit(1); }
}

static void init(const cell *c)
{
    if (toks_scratch_init(c->ctx, c->scr, c->sb, SFL) != 0) { fprintf(stderr, "toks_scratch_init\n"); exit(1); }
}

/* the ids of every call in state s (0 cold, 1 pass, 2 warm), untimed: FNV-1a 64 over each call's count and ids, so
 * two builds' runs prove ids equality per state (ids=) */
static uint64_t digest(const cell *c, int s)
{
    uint64_t h = 0xcbf29ce484222325ull;
    init(c);
    if (s == 2) {
        for (uint64_t k = 0; k < c->nc; k++) { enc(c, k); }
    }
    for (uint64_t k = 0; k < c->nc; k++) {
        if (s == 0) { init(c); }
        int64_t m = toks_encode(c->ctx, c->buf + c->beg[k], c->len[k], 0, c->ids, c->idcap, c->scr);
        if (m < 0) { fprintf(stderr, "toks_encode: %lld at chunk %llu\n", (long long)m, (unsigned long long)k); exit(1); }
        h = (h ^ (uint64_t)m) * 0x100000001b3ull;
        for (int64_t j = 0; j < m; j++) { h = (h ^ c->ids[j]) * 0x100000001b3ull; }
    }
    return h;
}

static uint64_t pass_run(const cell *c)
{
    uint64_t a = now_ns();
    for (uint64_t k = 0; k < c->nc; k++) { enc(c, k); }
    return now_ns() - a;
}

static uint64_t pass_cold(const cell *c)
{
    uint64_t t = 0;
    for (uint64_t k = 0; k < c->nc; k++) {
        init(c);
        uint64_t a = now_ns();
        enc(c, k);
        t += now_ns() - a;
    }
    return t;
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: spm_stages <tokenizer> <chunk_bytes|0> <reps> <file>...\n"); return 2; }
    uint64_t chunk = strtoull(argv[2], NULL, 10);
    int reps = atoi(argv[3]);
    SFL = getenv("SP_CACHE_MIB") ? TOKS_SCRATCH_CACHE_MIB(strtoul(getenv("SP_CACHE_MIB"), NULL, 10)) : 0u;
    const char *var = getenv("SP_VARIANT") ? getenv("SP_VARIANT") : "?";
    cell c;
    memset(&c, 0, sizeof c);
    c.buf = read_all(argv + 4, argc - 4, &c.n);
    uint64_t maxlen = cut(&c, chunk);
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, argv[1], NULL);
    if (r != 0) { fprintf(stderr, "toks_load: %lld\n", (long long)r); return 1; }
    c.ctx = ctx;
    c.sb = toks_scratch_bytes(ctx, maxlen, SFL);
    c.scr = malloc(c.sb);
    c.idcap = maxlen + 64;
    c.ids = malloc(c.idcap * sizeof(uint32_t));
    double tns = tick_ns(), pair = tick_pair();
    if (strcmp(var, "words") == 0) {                /* one pass from a fresh init, recorded; nothing timed */
        sp_out = getenv("SP_WORDS") != NULL ? fopen(getenv("SP_WORDS"), "w") : NULL;
        if (sp_out == NULL) { fprintf(stderr, "words: set SP_WORDS to a writable path\n"); return 1; }
        init(&c);
        sp_on = 1;
        (void)pass_run(&c);
        sp_on = 0;
        fclose(sp_out);
        printf("SP variant=words chunk=%llu bytes=%llu calls=%llu file=%s\n", (unsigned long long)chunk,
               (unsigned long long)c.n, (unsigned long long)c.nc, getenv("SP_WORDS"));
        (void)tns;
        (void)pair;
        return 0;
    }
    const char *st[3] = { "cold", "pass", "warm" };
    for (int s = 0; s < 3; s++) {
        uint64_t best = UINT64_MAX, snap[16];
        memset(snap, 0, sizeof snap);
        for (int rep = 0; rep < reps; rep++) {
            uint64_t d;
            init(&c);
            if (s == 2) { (void)pass_run(&c); }
            memset(sp_cnt, 0, sizeof sp_cnt);
            d = s == 0 ? pass_cold(&c) : pass_run(&c);
            if (d < best) { best = d; memcpy(snap, sp_cnt, sizeof snap); }
        }
        double mod_ns = ((double)snap[2] - pair * (double)snap[3]) * tns, enc_ns = ((double)snap[0] - pair * (double)snap[1]) * tns;
        printf("SP variant=%s state=%s chunk=%llu bytes=%llu calls=%llu ns=%llu mbs=%.1f enc_ns=%.0f model_ns=%.0f"
               " enc_calls=%llu model_calls=%llu wide_hits=%llu short_hits=%llu words=%llu word_bytes=%llu one_char=%llu"
               " steps8=%llu char_steps=%llu model_long=%llu model_gt30=%llu model_gt7ids=%llu pair_probes=%llu static_hits=%llu"
               " pair_ticks=%.1f cache_mib=%u"
               " ids=%016llx\n",
               var, st[s], (unsigned long long)chunk, (unsigned long long)c.n, (unsigned long long)c.nc,
               (unsigned long long)best, (double)c.n / ((double)best * 1e-9) / 1e6, enc_ns, mod_ns,
               (unsigned long long)snap[1], (unsigned long long)snap[3], (unsigned long long)snap[4],
               (unsigned long long)snap[5], (unsigned long long)snap[6], (unsigned long long)snap[7],
               (unsigned long long)snap[8], (unsigned long long)snap[9], (unsigned long long)snap[10],
               (unsigned long long)snap[11], (unsigned long long)snap[14], (unsigned long long)snap[15],
               (unsigned long long)snap[12], (unsigned long long)snap[13], pair,
               (unsigned)(SFL >> 12), (unsigned long long)digest(&c, s));
        fflush(stdout);
    }
    toks_unload(ctx);
    return 0;
}
