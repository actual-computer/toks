/*
 * test_k7.c: K7, the spm scan (layout.h; spm_c.c's c twin, src/asm/<isa>/k7_spm_<tier>.S): the tier's part against
 * the twin on random text units over every tests/data/spm fixture and the pinned spm files (gemma 4, mistral v0.3,
 * phi-3 when present): ascii runs, CJK, Cyrillic, Devanagari (E0 leads), Hangul (ED), emoji (F0), U+2581, vocab
 * strings (spanned pairs) and invalid utf-8 (lone continuations, C0 C1, overlongs, surrogates, F5+, cut chars). Both
 * walk a unit window by window as toks_spm_encode does (a window the part leaves goes to the twin); per position:
 * the same word starts, the same one-char steps at every 2-4 byte char, mb iff the window holds one. Every unit
 * ends flush against a no-access page; units of 1..700 bytes; every call of the part goes through the abi checker
 * (callee-saved registers: win64 rdi rsi xmm6-15, arm64 x18-x28 v8-v15). Run from the repository root (make test
 * does).
 */
#include "core.h"
#include "kernels.h"
#include "spm.h"
#include "guard.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
static uint64_t checks, parts;
#define CHECK(c, ...)                                                                                      \
    do {                                                                                                   \
        checks++;                                                                                          \
        if (!(c)) {                                                                                        \
            if (++fails <= 30) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
        }                                                                                                  \
    } while (0)

/* the toks_spm fields the asm parts read at layout.h's SPM_* offsets (ascii[] at 0) */
_Static_assert(offsetof(toks_spm, ascii) == 0 && offsetof(toks_spm, stage1) == SPM_STAGE1 &&
               offsetof(toks_spm, stage2) == SPM_STAGE1 + 8 && offsetof(toks_spm, pairs) == SPM_PAIRS &&
               offsetof(toks_spm, pairs_mask) == SPM_PAIRS + 8 && offsetof(toks_spm, cut) == SPM_CUT &&
               offsetof(toks_spm, id_repl) == SPM_ID_REPL && offsetof(toks_spm, ms_split) == SPM_MS_SPLIT &&
               offsetof(toks_spm, cut_ab) == SPM_CUT_AB && offsetof(toks_spm, cut_ab8) == SPM_CUT_AB8,
               "toks_spm vs layout.h SPM_*");

typedef uint64_t (*k7_fn)(const toks_spm *, toks_k7_args *);
#ifdef K7_MODEL
uint64_t k7_model(const toks_spm *s, toks_k7_args *a);
static k7_fn PART = k7_model;
#elif TOKS_HAVE_K7_SPM_NEON
static k7_fn PART = toks_k7_spm_neon;
#elif TOKS_HAVE_K7_SPM_AVX2
static k7_fn PART = toks_k7_spm_avx2;
#else
static k7_fn PART = NULL;
#endif

#define MAXU 704u

/* a unit's word starts (bit per position) and one-char steps, walked like toks_spm_encode with fn's windows; fn
 * runs through the abi checker (tests/common/abicheck_<isa>.S): *abi ORs its reports (a callee-saved register
 * the part did not preserve; bits 16.. are win64's xmm6-15, which sysv hosts cannot see clobbered) */
static void walk(const toks_spm *s, const uint8_t *t, uint64_t n, k7_fn fn, uint8_t *ws, uint32_t *one_at, int *bad_mb,
                 uint64_t *abi)
{
    uint32_t one[64];
    toks_k7_args a = { t, n, 0u, 0x001FFFFFu | (255u << 21), one, 0u, 1u, 0u };
    memset(ws, 0, n);
    while (a.pos < n) {
        uint64_t base = a.pos, m = 0, rep = 0;
        memset(one, 0xEE, sizeof one);
        if (fn != NULL) {
            m = toks_abicheck_call((const void *)fn, (uint64_t)(uintptr_t)s, (uint64_t)(uintptr_t)&a, &rep);
            *abi |= rep;
            parts += a.pos != base;
            for (uint64_t i = base; i < a.pos; i += toks_utf8_len(t + i, n - i)) {   /* a part takes valid utf-8 only */
                if (toks_utf8_len(t + i, n - i) == 0u) { *bad_mb = 3; break; }
            }
        }
        if (a.pos == base) { m = toks_k7_spm_c(s, &a); }
        int multi = 0;
        for (uint64_t i = base; i < a.pos; i++) {
            if ((m >> (i - base)) & 1u) { ws[i] = 1; }
            if (t[i] >= 0xC0u) {
                uint32_t k = toks_utf8_len(t + i, n - i);
                if (k >= 2u) { one_at[i] = one[i - base]; multi = 1; }
            }
        }
        if ((a.mb != 0u) != multi) { *bad_mb = 1; }
        if (a.pos - base < 64u && (m >> (a.pos - base)) != 0u) { *bad_mb = 2; }   /* no bits past the window */
    }
}

static uint64_t rng;
static uint32_t rnd(uint32_t n) { return (uint32_t)(guard_rng(&rng) % n); }

static uint64_t put_cp(uint8_t *p, uint32_t cp)
{
    if (cp < 0x80u) { p[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) { p[0] = (uint8_t)(0xC0u | cp >> 6); p[1] = (uint8_t)(0x80u | (cp & 0x3Fu)); return 2; }
    if (cp < 0x10000u) {
        p[0] = (uint8_t)(0xE0u | cp >> 12); p[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); p[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    p[0] = (uint8_t)(0xF0u | cp >> 18); p[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    p[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); p[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4;
}

/* random unit text: runs of one kind, so windows are pure ascii, pure CJK, mixed, ... */
static uint64_t gen(const toks_ctx *ctx, uint8_t *p, uint64_t n, int bad)
{
    static const char ASC[] = "aaeeiiooutnsrhlldcmwfgypb  ,.;:'\"!?-()[]{}<>/\\_=+*&^%$#@~`|0123456789\n\n\t\r";
    static const uint8_t INV[][4] = { { 0x80 }, { 0xBF }, { 0xC0, 0x80 }, { 0xC1, 0xBF }, { 0xE0, 0x80, 0x80 },
                                      { 0xED, 0xA0, 0x80 }, { 0xF0, 0x80, 0x80, 0x80 }, { 0xF4, 0x90, 0x80, 0x80 },
                                      { 0xF5, 0x80, 0x80, 0x80 }, { 0xFF }, { 0xE4, 0xB8 }, { 0xF0, 0x9F, 0x98 },
                                      { 0xF9, 0x80, 0x80, 0x80 }, { 0xFC, 0x80, 0x80, 0x80 }, { 0xC2 } };
    static const uint8_t INVN[] = { 1, 1, 2, 2, 3, 3, 4, 4, 4, 1, 2, 3, 4, 4, 1 };
    uint64_t i = 0;
    while (i + 8u < n) {
        uint32_t kind = rnd(bad ? 9u : 8u), run = 1u + rnd(24u);
        for (uint32_t r = 0; r < run && i + 8u < n; r++) {
            switch (kind) {
            case 0: case 1: p[i++] = (uint8_t)ASC[rnd((uint32_t)sizeof ASC - 1u)]; break;
            case 2: i += put_cp(p + i, 0x4E00u + rnd(0x5200u)); break;                       /* CJK */
            case 3: i += put_cp(p + i, rnd(4u) ? 0x0400u + rnd(0x60u) : 0x0370u + rnd(0x90u)); break;
            case 4: i += put_cp(p + i, rnd(2u) ? 0x0900u + rnd(0x80u) : 0xAC00u + rnd(0x2BA4u)); break;
            case 5: i += put_cp(p + i, rnd(3u) ? 0x2581u : rnd(2u) ? 0x1F600u + rnd(0x50u) : 0x2018u + rnd(8u)); break;
            case 6: { uint32_t cp = 0x80u + rnd(0x10FF80u); i += put_cp(p + i, (cp >> 11) == 0x1Bu ? 0xE000u : cp); break; }
            case 7: {                                                                        /* a vocab string */
                uint64_t tl = 0;
                const uint8_t *tb = toks_token(ctx, rnd(4000u) + (rnd(2u) ? 0u : rnd(200000u)), &tl);
                if (tb != NULL && tl < 8u) { memcpy(p + i, tb, (size_t)tl); i += tl; }
                break;
            }
            default: { uint32_t k = rnd((uint32_t)sizeof INVN); memcpy(p + i, INV[k], INVN[k]); i += INVN[k]; }
            }
        }
    }
    while (i < n) { p[i++] = (uint8_t)ASC[rnd(20u)]; }
    return n;
}

static void run(const char *name, const toks_ctx *ctx, int units)
{
    const toks_spm *s = ctx->spm;
    if (s == NULL || s->pairs == NULL) { return; }
    static uint8_t buf[MAXU + 8], ws_t[MAXU], ws_p[MAXU];
    static uint32_t one_t[MAXU], one_p[MAXU];
    guard_buf g;
    uint64_t c0 = checks;
    for (int u = 0; u < units; u++) {
        uint64_t n = 1u + rnd(MAXU - 1u);
        gen(ctx, buf, n, u % 3 == 2);
        uint8_t *t = guard_alloc(&g, (size_t)n, GUARD_END, 0);
        memcpy(t, buf, (size_t)n);
        int bt = 0, bp = 0;
        uint64_t abi = 0;
        memset(one_t, 0, sizeof one_t);
        memset(one_p, 0, sizeof one_p);
        walk(s, t, n, NULL, ws_t, one_t, &bt, &abi);
        walk(s, t, n, PART, ws_p, one_p, &bp, &abi);
        CHECK(bt == 0 && bp == 0, "%s unit %d: mb / window bits (%d %d)", name, u, bt, bp);
        CHECK(abi == 0u, "%s unit %d: the part did not preserve callee-saved registers (abicheck report %#llx)", name,
              u, (unsigned long long)abi);
        for (uint64_t i = 0; i < n; i++) {
            CHECK(ws_t[i] == ws_p[i], "%s unit %d len %llu: word start at %llu: twin %d part %d", name, u,
                  (unsigned long long)n, (unsigned long long)i, ws_t[i], ws_p[i]);
            CHECK(one_t[i] == one_p[i], "%s unit %d: one at %llu: twin %08x part %08x", name, u, (unsigned long long)i,
                  one_t[i], one_p[i]);
        }
        guard_free(&g);
    }
    printf("%s: %llu checks\n", name, (unsigned long long)(checks - c0));
}

int main(int argc, char **argv)
{
    rng = argc > 1 ? strtoull(argv[1], NULL, 10) : 7u;
    int units = argc > 2 ? atoi(argv[2]) : 300;
    if (PART == NULL) { printf("test_k7: no asm part in this build (SKIP)\n"); return 0; }
    static const char *FIX[] = { "gemma4like", "llamalike", "mistrallike", "meta_always_split", "meta_never_split",
                                 "merge_order", "replace_chain", "no_unk_no_bf", "holes_nodec" };
    char path[512];
    for (size_t f = 0; f < sizeof FIX / sizeof FIX[0]; f++) {
        snprintf(path, sizeof path, "tests/data/spm/%s.json", FIX[f]);
        toks_ctx *ctx = NULL;
        if (toks_load(&ctx, path, NULL) != 0) { printf("%s: load failed\n", path); fails++; continue; }
        run(FIX[f], ctx, units);
        toks_unload(ctx);
    }
    static const char *PIN[] = { "gemma4", "mistral-v0.3", "phi3" };
    const char *home = getenv("HOME");
    for (size_t f = 0; f < sizeof PIN / sizeof PIN[0]; f++) {
        snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", home ? home : ".", PIN[f]);
        toks_ctx *ctx = NULL;
        if (toks_load(&ctx, path, NULL) != 0) { printf("%s: SKIP (not here)\n", PIN[f]); continue; }
        run(PIN[f], ctx, units * 4);
        toks_unload(ctx);
    }
    printf("test_k7: %llu checks, %d failures (%llu windows in the part)\n", (unsigned long long)checks, fails,
           (unsigned long long)parts);
    return fails != 0;
}
