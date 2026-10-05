/*
 * tests/fuzz/fuzz.h: what the libFuzzer harnesses share (docs/fuzz.md; SPEC §14.2, §14.4, T6, T9).
 *
 * Black box: every harness drives toks through include/toks.h only, and every oracle is an invariant of the
 * contract or a second toks path that must agree, never hf (the parity suites own hf):
 *   - tiers: each tokenizer loaded twice, TOKS_TIER_SCALAR (the c twins, ASan / UBSan instrumented) and AUTO
 *     (the asm tier of this machine): encode, pieces and decode must agree on every input;
 *   - state: cold scratch of exactly toks_scratch_bytes(len) at any start alignment, a scratch warm across
 *     every earlier input (2 MiB and TOKS_SCRATCH_CACHE_MIB(32)), the same scratch rebound to another context
 *     and back: one answer (T2);
 *   - capacity: any cap returns the same total and an exact prefix, nothing written at or past cap; count-only;
 *   - decode: TOKS_E_ID / TOKS_E_ARG with nothing written; byte-level files without a normalizer: exactly an
 *     independent from_utf8_lossy (unicode §3.9 table 3-7) of the toks_token bytes; always valid utf-8;
 *     decode(encode(x)) == lossy(x) where the file says it must (FZ_RT);
 *   - stream == batch decode over random partitions; a push below its need is TOKS_E_CAP and leaves st as is;
 *   - toks_par_encode == toks_encode; every toks_split_points cut is exact: the parts (TOKS_CONTINUATION after
 *     the first) concatenate to the whole input's ids without post-processing (SPEC §5.2).
 * A violated invariant prints what it saw and aborts (libFuzzer saves the input as a crash).
 * Every function is static: a harness is one translation unit.
 */
#ifndef TOKS_FUZZ_H
#define TOKS_FUZZ_H

#define _POSIX_C_SOURCE 200809L

#include "toks.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FZ_FN static __attribute__((unused))
#define FZ_N(a) ((uint64_t)(sizeof(a) / sizeof((a)[0])))
#define FZ_MAX_TEXT (1u << 16)

/* ---- failure, rng, buffers ----------------------------------------------------------------------------- */

static const char *fz_what = "";                /* the tokenizer under test, for the report */

FZ_FN void fz_fail(const char *file, int line, const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "\nFUZZ INVARIANT VIOLATED %s:%d [%s]: ", file, line, fz_what);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

#define FZ_CHECK(cond, ...) do { if (!(cond)) { fz_fail(__FILE__, __LINE__, __VA_ARGS__); } } while (0)

FZ_FN void *fz_alloc(uint64_t n)
{
    void *p = malloc(n != 0u ? (size_t)n : 1u);
    FZ_CHECK(p != NULL, "malloc(%" PRIu64 ")", n);
    return p;
}

/* splitmix64: choices derived from the input, so every run of an input is the same run */
FZ_FN uint64_t fz_rng(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

FZ_FN uint64_t fz_below(uint64_t *s, uint64_t n) { return n == 0u ? 0u : fz_rng(s) % n; }

FZ_FN uint64_t fz_hash(const uint8_t *d, uint64_t n)          /* fnv-1a 64 */
{
    uint64_t h = 0xCBF29CE484222325ull;
    for (uint64_t i = 0; i < n; i++) { h = (h ^ d[i]) * 0x100000001B3ull; }
    return h;
}

typedef struct fz_bytes { uint8_t *p; uint64_t n, cap; } fz_bytes;

FZ_FN void fz_bytes_put(fz_bytes *b, const void *src, uint64_t k)
{
    if (b->n + k > b->cap) {
        uint64_t c = b->cap != 0u ? b->cap : 256u;
        while (c < b->n + k) { c *= 2u; }
        b->p = (uint8_t *)realloc(b->p, (size_t)c);
        FZ_CHECK(b->p != NULL, "realloc");
        b->cap = c;
    }
    if (k != 0u) { memcpy(b->p + b->n, src, (size_t)k); }
    b->n += k;
}

FZ_FN void fz_bytes_free(fz_bytes *b) { free(b->p); b->p = NULL; b->n = b->cap = 0u; }

FZ_FN void fz_hexdump(const char *what, const uint8_t *p, uint64_t n)
{
    fprintf(stderr, "%s (%" PRIu64 " bytes):", what, n);
    for (uint64_t i = 0; i < n && i < 96u; i++) { fprintf(stderr, " %02x", p[i]); }
    fprintf(stderr, n > 96u ? " ...\n" : "\n");
}

/* the first index where two id arrays differ (or the shorter length) */
FZ_FN uint64_t fz_first_diff(const uint32_t *a, uint64_t na, const uint32_t *b, uint64_t nb)
{
    uint64_t i = 0;
    while (i < na && i < nb && a[i] == b[i]) { i++; }
    return i;
}

#define FZ_SAME_IDS(a, na, b, nb, what) do { \
    if ((na) != (nb) || ((na) != 0u && memcmp((a), (b), (size_t)(na) * 4u) != 0)) { \
        uint64_t fz_i_ = fz_first_diff((a), (na), (b), (nb)); \
        fz_fail(__FILE__, __LINE__, "%s: ids differ at %" PRIu64 " (n %" PRIu64 " vs %" PRIu64 ", %u vs %u)", (what), \
                fz_i_, (uint64_t)(na), (uint64_t)(nb), fz_i_ < (na) ? (a)[fz_i_] : 0u, fz_i_ < (nb) ? (b)[fz_i_] : 0u); \
    } } while (0)

/* ---- utf-8: an independent from_utf8_lossy (one U+FFFD per maximal subpart, unicode §3.9 table 3-7) ---- */

FZ_FN uint64_t fz_lossy(const uint8_t *s, uint64_t n, uint8_t *out)      /* out: room 3n */
{
    uint64_t i = 0, o = 0;
    while (i < n) {
        uint8_t b = s[i];
        if (b < 0x80u) { out[o++] = b; i++; continue; }
        uint32_t need = 0;
        uint8_t lo = 0x80u, hi = 0xBFu;
        if (b >= 0xC2u && b <= 0xDFu) { need = 2u; }
        else if (b == 0xE0u) { need = 3u; lo = 0xA0u; }
        else if ((b >= 0xE1u && b <= 0xECu) || b == 0xEEu || b == 0xEFu) { need = 3u; }
        else if (b == 0xEDu) { need = 3u; hi = 0x9Fu; }
        else if (b == 0xF0u) { need = 4u; lo = 0x90u; }
        else if (b >= 0xF1u && b <= 0xF3u) { need = 4u; }
        else if (b == 0xF4u) { need = 4u; hi = 0x8Fu; }
        uint32_t k = 1u;
        while (need != 0u && k < need && i + k < n) {
            uint8_t c = s[i + k];
            if (c < (k == 1u ? lo : 0x80u) || c > (k == 1u ? hi : 0xBFu)) { break; }
            k++;
        }
        if (need != 0u && k == need) {
            memcpy(out + o, s + i, need);
            o += need;
            i += need;
        } else {
            out[o++] = 0xEFu; out[o++] = 0xBFu; out[o++] = 0xBDu;
            i += k;
        }
    }
    return o;
}

FZ_FN int fz_utf8_valid(const uint8_t *s, uint64_t n)
{
    uint8_t *tmp = (uint8_t *)fz_alloc(3u * n + 1u);
    uint64_t m = fz_lossy(s, n, tmp);
    int ok = m == n && (n == 0u || memcmp(tmp, s, (size_t)n) == 0);
    free(tmp);
    return ok;
}

FZ_FN uint32_t fz_utf8_put(uint32_t cp, uint8_t *o)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1u; }
    if (cp < 0x800u) { o[0] = (uint8_t)(0xC0u | (cp >> 6)); o[1] = (uint8_t)(0x80u | (cp & 0x3Fu)); return 2u; }
    if (cp < 0x10000u) {
        o[0] = (uint8_t)(0xE0u | (cp >> 12)); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3u;
    }
    o[0] = (uint8_t)(0xF0u | (cp >> 18)); o[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); o[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4u;
}

/* ---- a tokenizer under test: both tiers, warm scratches, what the oracles may assume --------------------- */

/* FZ_RT: byte-level bpe, no normalizer, every byte in the alphabet: decode(ids, 0) is from_utf8_lossy of the ids'
 * toks_token bytes, and decode(encode(x, NONE | NO_POSTPROCESS)) is from_utf8_lossy(x) */
#define FZ_RT 1u
#define FZ_NLIT 256u              /* literals a tokenizer gives the generator: the first 32, then spread over ids */

typedef struct fz_tok {
    const char *name;
    uint32_t    oracles;          /* FZ_* */
    toks_ctx   *s, *a;            /* TOKS_TIER_SCALAR, TOKS_TIER_AUTO */
    toks_info   info;
    void       *warm, *warm32;    /* scratches warm across inputs (flags 0 / TOKS_SCRATCH_CACHE_MIB(32)) */
    uint64_t    warm_len, warm32_len, warm_max;
    toks_par   *par, *par1;       /* pools of 3 over the auto context, made on first use: par eager (env
                                     TOKS_PAR_EAGER=1 at create: every call of >= 16 KiB goes wide), par1 by
                                     the cost model */
    fz_bytes    lit;              /* markup-like tokens (NUL-separated) for the generator */
    uint32_t    nlit, lit_off[FZ_NLIT];
} fz_tok;

FZ_FN uint8_t *fz_slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long m = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = (uint8_t *)fz_alloc((uint64_t)(m > 0 ? m : 1));
    size_t r = m > 0 ? fread(p, 1, (size_t)m, f) : 0u;
    fclose(f);
    if (m < 0 || r != (size_t)m) { free(p); return NULL; }
    *len = (uint64_t)m;
    return p;
}

/* warm scratches sized for texts of up to `need` bytes (grown, then kept: warm across inputs) */
FZ_FN void fz_warm_fit(fz_tok *t, uint64_t need)
{
    if (t->warm != NULL && need <= t->warm_max) { return; }
    uint64_t m = need < 4096u ? 4096u : need;
    free(t->warm);
    free(t->warm32);
    t->warm_len = toks_scratch_bytes(t->a, m, 0u);
    t->warm32_len = toks_scratch_bytes(t->a, m, TOKS_SCRATCH_CACHE_MIB(32));
    t->warm = fz_alloc(t->warm_len);
    t->warm32 = fz_alloc(t->warm32_len);
    FZ_CHECK(toks_scratch_init(t->a, t->warm, t->warm_len, 0u) == 0, "warm scratch init");
    FZ_CHECK(toks_scratch_init(t->a, t->warm32, t->warm32_len, TOKS_SCRATCH_CACHE_MIB(32)) == 0, "warm32 init");
    t->warm_max = m;
}

/* both contexts of t from a file path, or from src bytes when path is NULL; 0, else the load's code (both tiers
 * must agree on it, and a failed load leaves *out NULL) */
FZ_FN int64_t fz_tok_open(fz_tok *t, const char *name, const char *path, const uint8_t *src, uint64_t n)
{
    toks_load_opts o;
    toks_diag dg;
    memset(t, 0, sizeof *t);
    t->name = name;
    memset(&o, 0, sizeof o);
    memset(&dg, 0, sizeof dg);
    o.size = sizeof o;
    o.tier = TOKS_TIER_SCALAR;
    o.diag = &dg;
    int64_t r = path != NULL ? toks_load(&t->s, path, &o) : toks_load_mem_copy(&t->s, src, n, &o);
    o.tier = TOKS_TIER_AUTO;
    o.diag = NULL;
    int64_t r2 = path != NULL ? toks_load(&t->a, path, &o) : toks_load_mem_copy(&t->a, src, n, &o);
    FZ_CHECK(r == r2, "the scalar load returns %" PRId64 ", the auto load %" PRId64, r, r2);
    if (r != 0) {
        FZ_CHECK(t->s == NULL && t->a == NULL, "a failed load (%" PRId64 ") left *out set", r);
        FZ_CHECK(r >= TOKS_E_NOMEM && r <= TOKS_E_OPEN, "undocumented load code %" PRId64, r);
        FZ_CHECK(dg.code == r && memchr(dg.what, 0, sizeof dg.what) != NULL, "diag code %" PRId64 " vs %" PRId64, dg.code, r);
        return r;
    }
    FZ_CHECK(t->s != NULL && t->a != NULL, "a successful load gave no context");
    memset(&t->info, 0, sizeof t->info);
    t->info.size = sizeof t->info;
    FZ_CHECK(toks_get_info(t->a, &t->info) == 0, "toks_get_info");
    FZ_CHECK(t->info.n_ids >= 1u && t->info.n_ids <= TOKS_MAX_IDS, "n_ids %u", t->info.n_ids);
    /* markup-like tokens (<...>, [...]): the first 32 and the rest sampled evenly over the ids, so families that
     * share long prefixes (K1's radix-tree buckets: DeepSeek place holders, <unused..>, reserved tokens) give
     * the generator siblings to cut, overlap and glue */
    uint64_t cand = 0;
    for (int pass = 0; pass < 2; pass++) {
        uint64_t j = 0, step = cand > 32u ? (cand - 32u + (FZ_NLIT - 33u)) / (FZ_NLIT - 32u) : 1u;
        for (uint32_t id = 0; id < t->info.n_ids && t->nlit < FZ_NLIT; id++) {
            uint64_t k = 0;
            const uint8_t *p = toks_token(t->a, id, &k);
            if (p == NULL || k < 3u || k > 64u || !((p[0] == '<' && p[k - 1] == '>') || (p[0] == '[' && p[k - 1] == ']'))) {
                continue;
            }
            if (pass == 0) { cand++; continue; }
            if (j++ >= 32u && (j - 33u) % step != 0u) { continue; }
            t->lit_off[t->nlit++] = (uint32_t)t->lit.n;
            fz_bytes_put(&t->lit, p, k);
            fz_bytes_put(&t->lit, "", 1u);
        }
    }
    return 0;
}

FZ_FN void fz_tok_close(fz_tok *t)
{
    toks_par_destroy(t->par);
    toks_par_destroy(t->par1);
    toks_unload(t->s);
    toks_unload(t->a);
    free(t->warm);
    free(t->warm32);
    fz_bytes_free(&t->lit);
    memset(t, 0, sizeof *t);
}

FZ_FN const char *fz_tok_lit(const fz_tok *t, uint64_t k)   /* the k-th literal (mod count), NULL if none */
{
    if (t == NULL || t->nlit == 0u) { return NULL; }
    return (const char *)t->lit.p + t->lit_off[k % t->nlit];
}

/* ---- the pinned set: every algorithm, normalizer and template family toks ships (loaded on first use) ---- */

#define FZ_NPIN 16u
static fz_tok FZ_PIN[FZ_NPIN];
static struct { const char *name; uint32_t oracles; } FZ_PIN_DEF[FZ_NPIN] = {
    { "gpt2", FZ_RT },         { "llama3", FZ_RT },       { "qwen38", 0 },          { "glm53", FZ_RT },
    { "o200k", FZ_RT },        { "dsv3", FZ_RT },         { "gemma4", 0 },          { "mistral-v0.3", 0 },
    { "wp-bert-uncased", 0 },  { "uni_t5base", 0 },       { "uni_bgem3", 0 },       { "minimaxm2", 0 },
    { "nemotron3-4b", FZ_RT }, { "dg-smollm2", FZ_RT },   { "pythia", 0 },          { "tinyllama", 0 },
};
static int fz_pin_missing[FZ_NPIN];
static uint32_t fz_npin;              /* the pins in use: FZ_NPIN, or the list in env TOKS_FUZZ_PINS */

/* a targeted campaign (docs/fuzz.md §3): env TOKS_FUZZ_PINS = "name[:rt],..." (at most 16 files of the tokenizer
 * cache) replaces the pinned set, e.g. the K1 radix-tree families "dsv3:rt,dsv4:rt,gemma3,llama4:rt"; ":rt" claims
 * FZ_RT (byte-level, no normalizer, every byte in the alphabet; checked at load as for the default set) */
/* at exit: the pinned contexts and their pools, so LeakSanitizer's exit check sees them freed (a pool keeps its
 * plan arrays in malloc'd blocks referenced only from its mmap'd arena, which LSan does not scan) */
static void fz_pins_close(void)
{
    for (uint32_t i = 0; i < FZ_NPIN; i++) {
        if (FZ_PIN[i].a != NULL || FZ_PIN[i].s != NULL) { fz_tok_close(&FZ_PIN[i]); }
    }
}

FZ_FN void fz_pins_init(void)
{
    static char buf[1024];
    if (fz_npin != 0u) { return; }
    fz_npin = FZ_NPIN;
    atexit(fz_pins_close);
    const char *e = getenv("TOKS_FUZZ_PINS");
    if (e == NULL || e[0] == 0 || strlen(e) >= sizeof buf) { return; }
    memcpy(buf, e, strlen(e) + 1u);
    uint32_t n = 0;
    for (char *tok = strtok(buf, ","); tok != NULL && n < FZ_NPIN; tok = strtok(NULL, ",")) {
        char *c = strchr(tok, ':');
        uint32_t o = 0;
        if (c != NULL) { *c = 0; o = strcmp(c + 1, "rt") == 0 ? FZ_RT : 0u; }
        FZ_PIN_DEF[n].name = tok;
        FZ_PIN_DEF[n].oracles = o;
        n++;
    }
    if (n != 0u) { fz_npin = n; }
}

FZ_FN const char *fz_tok_path(const char *name)
{
    static char buf[1024];
    const char *dir = getenv("TOKS_TOKENIZER_CACHE"), *home = getenv("HOME");
    if (dir != NULL) { snprintf(buf, sizeof buf, "%s/%s", dir, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", home != NULL ? home : ".", name); }
    return buf;
}

/* pinned tokenizer i, loaded on first use; NULL when its file is missing or refused */
FZ_FN fz_tok *fz_pin_get(uint32_t i)
{
    if (fz_pin_missing[i]) { return NULL; }
    if (FZ_PIN[i].a == NULL) {
        int64_t r = fz_tok_open(&FZ_PIN[i], FZ_PIN_DEF[i].name, fz_tok_path(FZ_PIN_DEF[i].name), NULL, 0u);
        if (r != 0) {
            fprintf(stderr, "fuzz: %s not loaded (%" PRId64 "), skipped\n", FZ_PIN_DEF[i].name, r);
            fz_pin_missing[i] = 1;
            return NULL;
        }
        FZ_PIN[i].oracles = FZ_PIN_DEF[i].oracles;
        if (FZ_PIN[i].oracles & FZ_RT) {                /* FZ_RT needs every byte in the alphabet: none dropped */
            uint64_t sb = toks_scratch_bytes(FZ_PIN[i].a, 1u, 0u);
            void *scr = fz_alloc(sb);
            FZ_CHECK(toks_scratch_init(FZ_PIN[i].a, scr, sb, 0u) == 0, "scratch init");
            for (uint32_t b = 0; b < 256u; b++) {
                uint8_t c = (uint8_t)b;
                if (toks_encode(FZ_PIN[i].a, &c, 1u, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS, NULL, 0u, scr) <= 0) {
                    fprintf(stderr, "fuzz: %s drops byte %02x: no roundtrip oracle\n", FZ_PIN[i].name, b);
                    FZ_PIN[i].oracles &= ~FZ_RT;
                    break;
                }
            }
            free(scr);
        }
    }
    return &FZ_PIN[i];
}

/* the pinned tokenizer selected by b mod the pins in use (a missing file falls through to the next); NULL: none */
FZ_FN fz_tok *fz_pin(uint32_t b)
{
    fz_pins_init();
    for (uint32_t k = 0; k < fz_npin; k++) {
        fz_tok *t = fz_pin_get((b + k) % fz_npin);
        if (t != NULL) {
            fz_what = t->name;
            return t;
        }
    }
    return NULL;
}

/* a pinned tokenizer of another source than t (the rebinding checks): a loaded one first, else gpt2 */
FZ_FN fz_tok *fz_other(const fz_tok *t, uint32_t k)
{
    fz_pins_init();
    for (uint32_t j = 0; j <= fz_npin; j++) {
        fz_tok *o = j < fz_npin ? &FZ_PIN[(k + j) % fz_npin] : fz_pin_get(0);
        if (o != NULL && o->a != NULL && o != t && memcmp(o->info.source_sha256, t->info.source_sha256, 32) != 0) {
            return o;
        }
    }
    return NULL;
}

/* ---- calls ---------------------------------------------------------------------------------------------- */

/* a fresh scratch for ctx of exactly toks_scratch_bytes(len, flags) bytes at offset align of its block */
typedef struct fz_scr { uint8_t *block; void *p; uint64_t len; } fz_scr;

FZ_FN void fz_scr_new(fz_scr *s, const toks_ctx *ctx, uint64_t len, uint32_t flags, uint32_t align)
{
    s->len = toks_scratch_bytes(ctx, len, flags);
    s->block = (uint8_t *)fz_alloc(s->len + 64u);
    s->p = s->block + (align & 63u);
    FZ_CHECK(toks_scratch_init(ctx, s->p, s->len, flags) == 0, "scratch init (%" PRIu64 " bytes)", s->len);
}

FZ_FN void fz_scr_free(fz_scr *s) { free(s->block); s->block = NULL; }

/* every caller id array the harnesses hand to toks sits fz_mis (0-3) bytes past a heap block's start: SPEC §4.1
 * and toks.h promise "no alignment required" (UBSan's alignment check sees any typed access). The harness itself
 * touches such arrays only through byte pointers and memcpy (clang assumes alignment for a typed one). */
static uint32_t fz_mis;

/* n u32 at fz_mis bytes into a fresh block (a copy of src when non-NULL): *block to free, the array as bytes */
FZ_FN uint8_t *fz_mis_ids(uint8_t **block, const uint32_t *src, uint64_t n)
{
    *block = (uint8_t *)fz_alloc(n * 4u + 4u);
    uint8_t *p = *block + fz_mis;
    if (src != NULL && n != 0u) { memcpy(p, src, (size_t)n * 4u); }
    return p;
}

#define FZ_U32P(b) ((uint32_t *)(void *)(b))

/* encode (pieces = 0) or pieces (pieces = 1) through a misaligned caller array; an aligned copy in *out; the count */
FZ_FN uint64_t fz_run(int pieces, const toks_ctx *ctx, const uint8_t *x, uint64_t len, uint32_t flags, void *scr,
                      uint32_t **out)
{
    int64_t n = pieces ? toks_pieces(ctx, x, len, flags, NULL, 0u, scr) : toks_encode(ctx, x, len, flags, NULL, 0u, scr);
    FZ_CHECK(n >= 0, "%s count-only returned %" PRId64 " (len %" PRIu64 ", flags %u)", pieces ? "pieces" : "encode", n,
             len, flags);
    uint8_t *blk, *mb = fz_mis_ids(&blk, NULL, (uint64_t)n);
    int64_t m = pieces ? toks_pieces(ctx, x, len, flags, FZ_U32P(mb), (uint64_t)n, scr)
                       : toks_encode(ctx, x, len, flags, FZ_U32P(mb), (uint64_t)n, scr);
    FZ_CHECK(m == n, "%s returned %" PRId64 " with cap = its count %" PRId64, pieces ? "pieces" : "encode", m, n);
    uint32_t *o = (uint32_t *)fz_alloc((uint64_t)n * 4u);
    if (n != 0) { memcpy(o, mb, (size_t)n * 4u); }
    free(blk);
    *out = o;
    return (uint64_t)n;
}

/* the capacity rule at one cap: the same total, an exact prefix, nothing written at or past cap */
FZ_FN void fz_check_cap(int pieces, const toks_ctx *ctx, const uint8_t *x, uint64_t len, uint32_t flags, void *scr,
                        const uint32_t *ref, uint64_t n, uint64_t cap)
{
    uint8_t *blk, *mb = fz_mis_ids(&blk, NULL, cap + 8u);
    memset(mb, 0xA5, (size_t)(cap + 8u) * 4u);
    int64_t m = pieces ? toks_pieces(ctx, x, len, flags, FZ_U32P(mb), cap, scr)
                       : toks_encode(ctx, x, len, flags, FZ_U32P(mb), cap, scr);
    FZ_CHECK(m == (int64_t)n, "cap %" PRIu64 ": %" PRId64 " instead of %" PRIu64, cap, m, n);
    uint64_t k = cap < n ? cap : n;
    FZ_CHECK(k == 0u || memcmp(mb, ref, (size_t)k * 4u) == 0, "cap %" PRIu64 ": not the exact prefix", cap);
    for (uint64_t i = cap * 4u; i < (cap + 8u) * 4u; i++) {
        FZ_CHECK(mb[i] == 0xA5u, "cap %" PRIu64 ": wrote byte %" PRIu64 " of out", cap, i);
    }
    free(blk);
}

/* decode (ids through a misaligned caller array) into an exact-size buffer: its length (or the negative code);
 * *out allocated (length + 1) or NULL */
FZ_FN int64_t fz_decode(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t **out)
{
    uint8_t *blk, *mb = fz_mis_ids(&blk, ids, n);
    int64_t m = toks_decode(ctx, FZ_U32P(mb), n, flags, NULL, 0u);
    *out = NULL;
    if (m >= 0) {
        uint8_t *o = (uint8_t *)fz_alloc((uint64_t)m + 1u);
        int64_t m2 = toks_decode(ctx, FZ_U32P(mb), n, flags, o, (uint64_t)m);
        FZ_CHECK(m2 == m, "decode returned %" PRId64 " with cap = its count %" PRId64, m2, m);
        *out = o;
    }
    free(blk);
    return m;
}

#endif /* TOKS_FUZZ_H */
