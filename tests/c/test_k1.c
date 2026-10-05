/*
 * test_k1.c: K1, added-token find (docs/kernels.md §4), on the c twin and on the asm tier the build has
 * (TOKS_HAVE_K1_NEON / TOKS_HAVE_K1_AVX2; skipped, and said so, when the cpu lacks it), against a reference written from the
 * prose (tests/k1/k1_tab.h: every token of the phase tried at every position, leftmost then longest; no index)
 * and against each other:
 *   - hand cases: no tokens at all (NULL tables), a phase with no tokens, phase 2 and 2^63, pos >= len, nested
 *     tokens and shared prefixes (the longest wins at the leftmost start), one-byte tokens, a token cut by len,
 *     a token ending exactly at len, shufti false positives, the driver's resume loop (batches of cap 1, 2, 3,
 *     16 and 64 in turn: n, next, the matches, nothing written past them);
 *   - geometry (SPEC §14.3): every length 0..255 (and a few longer ones) with the text's end flush against a
 *     no-access page and with its start at a page start + 0..63, the bytes beside it poisoned with token bytes,
 *     both phases, the resume loop and every pos 0..len + 1 (flush placements) or five (offset placements);
 *   - every 8th asm call through toks_abicheck_call (callee-saved registers incl. x18 and v8-v15 / win64 rdi rsi
 *     xmm6-15).
 * tests/k1/bench_k1.c is the long differential fuzz and the benchmark (lab hosts).
 */
#include "../k1/k1_tab.h"
#include "../../src/platform/cpu.h"
#include "../common/guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

typedef uint64_t (*k1_fn)(const toks_tables *t, toks_k1_args *a);   /* test code: pointers are fine */

typedef struct { const char *name; k1_fn fn; int ok; int abi; } tier;

static int failures;
static uint64_t checks, abi_calls;

/* one call per tier with cap 1, 2, 3, 16 or 64 in turn, each compared with the reference batch (n, next, m[0, n),
 * m[n, cap + 2) untouched, inputs kept); every 8th asm call through the abi checker */
static void check(const tier *tiers, size_t nt, const toks_tables *t, const char *label, const uint8_t *text,
                  uint64_t len, uint64_t pos, uint64_t phase, uint64_t *resume)
{
    static const uint64_t CAPS[5] = { 1u, 2u, 3u, 16u, 64u };
    uint64_t cap = CAPS[checks % 5u], want_next;
    toks_k1_match want[64], got[66];
    uint64_t want_n = k1_ref_batch(t, text, len, pos, phase, cap, want, &want_next);
    for (size_t k = 0; k < nt; k++) {
        if (!tiers[k].ok) { continue; }
        toks_k1_args a;
        memset(&a, 0x5A, sizeof a);
        memset(got, 0x5A, (size_t)(cap + 2u) * sizeof got[0]);
        a.text = text; a.len = len; a.pos = pos; a.phase = phase; a.m = got; a.cap = cap;
        uint64_t r, rep = 0;
        if (tiers[k].abi && (checks & 7u) == 7u) {
            r = toks_abicheck_call((const void *)tiers[k].fn, (uint64_t)(uintptr_t)t, (uint64_t)(uintptr_t)&a, &rep);
            abi_calls++;
        } else {
            r = tiers[k].fn(t, &a);
        }
        checks++;
        int bad = r != want_n || a.n != want_n || a.next != want_next || rep != 0u || a.text != text || a.len != len ||
                  a.pos != pos || a.phase != phase || a.m != got || a.cap != cap ||
                  (want_n != 0u && memcmp(got, want, (size_t)want_n * sizeof got[0]) != 0);
        for (size_t i = (size_t)want_n * sizeof got[0]; i < (size_t)(cap + 2u) * sizeof got[0]; i++) {
            bad |= ((const uint8_t *)got)[i] != 0x5Au;
        }
        if (bad) {
            if (failures < 20) {
                uint64_t i = 0;
                while (i < want_n && i < a.n && i < cap && memcmp(&got[i], &want[i], sizeof got[0]) == 0) { i++; }
                fprintf(stderr, "FAIL %s %s len %" PRIu64 " pos %" PRIu64 " phase %" PRIu64 " cap %" PRIu64 ": want n %"
                        PRIu64 " next %" PRIu64 ", got %" PRIu64 " / n %" PRIu64 " next %" PRIu64 " abi %#" PRIx64
                        "; match %" PRIu64 ": want [%u, %u) e%u, got [%u, %u) e%u rsv %#x\n", tiers[k].name, label, len,
                        pos, phase, cap, want_n, want_next, r, a.n, a.next, rep, i, i < want_n ? want[i].start : 0u,
                        i < want_n ? want[i].end : 0u, i < want_n ? want[i].entry : 0u, i < cap ? got[i].start : 0u,
                        i < cap ? got[i].end : 0u, i < cap ? got[i].entry : 0u, i < cap ? got[i].rsv : 0u);
            }
            failures++;
        }
    }
    if (resume) { *resume = want_n == cap ? want_next : len + 1u; }
}

/* the driver's loop (pos = next after each full batch), then every pos in [0, len + 1] (every_pos) or five */
static void sweep(const tier *tiers, size_t nt, const toks_tables *t, const char *label, const uint8_t *text,
                  uint64_t len, int every_pos, uint64_t salt)
{
    for (uint64_t ph = 0; ph < 2u; ph++) {
        uint64_t pos = 0;
        for (uint64_t n = 0; n <= len + 1u && pos <= len; n++) {        /* bound: len + 2 matches */
            check(tiers, nt, t, label, text, len, pos, ph, &pos);
        }
        if (every_pos) {
            for (uint64_t p = 0; p <= len + 1u; p++) { check(tiers, nt, t, label, text, len, p, ph, NULL); }
        } else {
            const uint64_t some[5] = { 1u, salt % (len + 1u), len / 2u, len - salt % (len + 1u), len - 1u };
            for (int k = 0; k < 5; k++) { check(tiers, nt, t, label, text, len, some[k], ph, NULL); }
        }
    }
}

static void s(const tier *tiers, size_t nt, const toks_tables *t, const char *label, const char *str)
{
    sweep(tiers, nt, t, label, (const uint8_t *)str, (uint64_t)strlen(str), 1, 0);
}

/* the fixed token set: nested prefixes, a one-byte token, a 200-byte token, shufti false positives */
static uint8_t longtok[200];
static const char *const P0[] = { "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<", "<|", "<|im", "[gMASK]",
                                  "\xBC\xBC", "ab", "abc", "\xFF" };
static const char *const P1[] = { "\n\n", "x", "xyz", "<s>", "\xE2\x96\x81", "</s>", "\xC3" };

/* long h4 buckets (more than TOKS_K1_LIST tokens share a bucket: radix trees, kernels.md §4): llama 3's reserved tokens,
 * DeepSeek's <｜...｜> names, nested runs (gemma's whitespace tokens), a 200-byte member, a phase-1 bucket; texts of
 * whole tokens, their prefixes and runs, at every pos, then the geometry sweep over lengths 0..300 */
static void radix_cases(const tier *tiers, size_t nt)
{
    static uint8_t pool[16384];
    static k1_tok tk[256];
    uint32_t n = 0, used = 0;
#define RX_ADD(ph, ...) do { int w_ = snprintf((char *)pool + used, 64, __VA_ARGS__); \
        tk[n].s = pool + used; tk[n].len = (uint32_t)w_; tk[n].phase = (ph); tk[n].special = 1; used += (uint32_t)w_; n++; } while (0)
    for (unsigned i = 0; i < 40u; i++) { RX_ADD(0, "<|reserved_special_token_%u|>", i); }
    RX_ADD(0, "<|re");
    RX_ADD(0, "<|res");
    for (unsigned i = 2; i < 32u; i++) { RX_ADD(0, "%*s", (int)i, ""); }
    RX_ADD(0, "<\xEF\xBD\x9CUser\xEF\xBD\x9C>");
    RX_ADD(0, "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>");
    RX_ADD(0, "<\xEF\xBD\x9C" "begin\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>");
    RX_ADD(0, "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>");
    for (unsigned i = 0; i < 30u; i++) { RX_ADD(0, "<\xEF\xBD\x9Cplace\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81%u\xEF\xBD\x9C>", i); }
    for (unsigned i = 4; i < 13u; i++) { RX_ADD(0, "%.*s", (int)i, "aaaaaaaaaaaaaaaa"); }
    RX_ADD(0, "aaaab");
    RX_ADD(0, "aaaabc");
    RX_ADD(0, "aaaaab");
    for (unsigned i = 0; i < 25u; i++) { RX_ADD(1, "<SPECIAL_%u>", i); }
    for (unsigned i = 4; i < 13u; i++) { RX_ADD(1, "%.*s", (int)i, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"); }
    RX_ADD(1, "\n\n");
#undef RX_ADD
    tk[n].s = pool + used;                                  /* a 200-byte member of the "<|re" bucket */
    memcpy(pool + used, "<|re", 4);
    for (uint32_t i = 4; i < 200u; i++) { pool[used + i] = (uint8_t)('a' + i % 5u); }
    tk[n].len = 200; tk[n].phase = 0; tk[n].special = 0; used += 200; n++;
    k1_tab rx;
    if (k1_tab_build(&rx, tk, n)) { fprintf(stderr, "table build failed\n"); failures++; return; }
    uint64_t radix = 0;
    for (uint32_t s = 0; s < (2u << TOKS_K1_H4_BITS); s++) { radix += rx.t.add_index[2u * 65536u + s] >> 63; }
    if (radix < 5u) { fprintf(stderr, "FAIL radix: %" PRIu64 " long buckets, want >= 5\n", radix); failures++; }
    const toks_tables *T = &rx.t;
    s(tiers, nt, T, "rx reserved", "a<|reserved_special_token_7|>b<|reserved_special_token_39|><|reserved_special_token_40|>");
    s(tiers, nt, T, "rx reserved cut", "<|reserved_special_token_1<|reserved_special_token_12|<|res<|re<|r");
    s(tiers, nt, T, "rx spaces", "x  y   z                                   w                               .         ");
    s(tiers, nt, T, "rx deepseek", "<\xEF\xBD\x9CUser\xEF\xBD\x9C>hi<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>yo<\xEF\xBD\x9C"
                                    "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C><\xEF\xBD\x9CUs<\xEF\xBD\x9C");
    s(tiers, nt, T, "rx placeholders", "<\xEF\xBD\x9Cplace\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81" "29\xEF\xBD\x9C>"
                                        "<\xEF\xBD\x9Cplace\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81" "30\xEF\xBD\x9C>"
                                        "<\xEF\xBD\x9Cplace\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81");
    s(tiers, nt, T, "rx nested", "aaaaaaaaaaaaaaaaaaaaaaaaaabaaaabcaaaaabaaaaa");
    s(tiers, nt, T, "rx phase 1", "<SPECIAL_3><SPECIAL_24><SPECIAL_25><SPECIAL_\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    {
        static uint8_t buf[300];
        memset(buf, '.', sizeof buf);
        memcpy(buf + 40, tk[n - 1u].s, 200);
        sweep(tiers, nt, T, "rx 200", buf, 300, 0, 3);
        sweep(tiers, nt, T, "rx 200 cut", buf, 239, 0, 5);
        buf[40 + 199] ^= 1u;
        sweep(tiers, nt, T, "rx 200 last byte", buf, 300, 0, 7);
    }
    static const char *const frag[] = { "<|reserved_special_token_", "<|reserved_special_token_1", "1|>", "|>", "<|re", "<|res",
        "<\xEF\xBD\x9C", "User\xEF\xBD\x9C>", "<\xEF\xBD\x9Cplace\xE2\x96\x81holder\xE2\x96\x81no\xE2\x96\x81", "7\xEF\xBD\x9C>",
        "aaaa", "aa", "b", "c", "<SPECIAL_", "9>", "\n\n\n", "\n", " ", "  ", "     ", "x" };
    uint64_t seed = 0x72785eedu;
    static uint8_t text[512];
    for (uint64_t len = 0; len <= 300u; len++) {          /* bound: 301 lengths */
        uint64_t m = 0;
        while (m < len) {
            const char *fr = frag[guard_rng(&seed) % (sizeof frag / sizeof frag[0])];
            for (size_t i = 0; fr[i] != 0 && m < len; i++) { text[m++] = (uint8_t)fr[i]; }
        }
        for (int where = 0; where < 3; where++) {
            guard_buf g = { 0 };
            uint8_t *p = guard_alloc(&g, (size_t)len + 64u, where == 0 ? GUARD_END : GUARD_START, (size_t)where);
            if (!p) { fprintf(stderr, "guard_alloc failed\n"); failures++; return; }
            if (where == 0) { p += 64; }
            memcpy(p, text, (size_t)len);
            sweep(tiers, nt, T, where == 0 ? "rx end flush" : "rx start", p, len, where < 2, len * 7u + 3u);
            guard_free(&g);
        }
    }
    k1_tab_free(&rx);
}

int main(void)
{
    uint64_t f = toks_cpu_features();
    tier tiers[] = {
        { "c", toks_k1_added_find_c, 1, 0 },
#if TOKS_HAVE_K1_NEON
        { "neon", toks_k1_added_find_neon, TOKS_CPU_HAS(f, TOKS_FEAT_NEON_TIER), 1 },
#endif
#if TOKS_HAVE_K1_AVX2
        { "avx2", toks_k1_added_find_avx2, TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER), 1 },
#endif
    };
    size_t nt = sizeof tiers / sizeof tiers[0];
    (void)f;
    for (size_t k = 0; k < nt; k++) {
        printf("tier %s: %s\n", tiers[k].name, tiers[k].ok ? "run" : "skip (not supported by this cpu)");
    }
    if (nt == 1u) { printf("no K1 asm tier in this build (TOKS_HAVE_K1_<TIER> unset)\n"); }

    /* tables: the fixed set, a set with phase 1 empty, and no tokens at all */
    memcpy(longtok, "<|", 2);
    for (size_t i = 2; i < sizeof longtok; i++) { longtok[i] = (uint8_t)('a' + i % 7u); }
    k1_tok tk[32];
    uint32_t n = 0;
    for (size_t i = 0; i < sizeof P0 / sizeof P0[0]; i++) {
        tk[n].s = (const uint8_t *)P0[i]; tk[n].len = (uint32_t)strlen(P0[i]); tk[n].phase = 0; tk[n].special = 1;
        n++;
    }
    tk[n].s = longtok; tk[n].len = sizeof longtok; tk[n].phase = 0; tk[n].special = 0;
    n++;
    uint32_t n0 = n;
    for (size_t i = 0; i < sizeof P1 / sizeof P1[0]; i++) {
        tk[n].s = (const uint8_t *)P1[i]; tk[n].len = (uint32_t)strlen(P1[i]); tk[n].phase = 1; tk[n].special = 0;
        n++;
    }
    k1_tab full, only0, none;
    if (k1_tab_build(&full, tk, n) || k1_tab_build(&only0, tk, n0) || k1_tab_build(&none, tk, 0)) {
        fprintf(stderr, "table build failed\n");
        return 2;
    }
    const toks_tables *T = &full.t;

    /* ---- hand cases */
    s(tiers, nt, &none.t, "no tokens", "<|endoftext|> abc x");
    s(tiers, nt, &only0.t, "phase 1 empty", "<|endoftext|>\n\nx");
    for (int k = 0; k < 2; k++) {
        static const uint64_t big[2] = { 2u, 1ull << 63 };
        const char *str = "<|im_end|>";
        check(tiers, nt, T, "phase out of range", (const uint8_t *)str, strlen(str), 0, big[k], NULL);
    }
    {
        const char *str = "ab<|im_end|>";
        for (uint64_t p = 12; p < 16; p++) { check(tiers, nt, T, "pos >= len", (const uint8_t *)str, 12, p, 0, NULL); }
        check(tiers, nt, T, "pos huge", (const uint8_t *)str, 12, ~0ull, 0, NULL);
        check(tiers, nt, T, "len 0", (const uint8_t *)str, 0, 0, 0, NULL);
    }
    s(tiers, nt, T, "alone", "<|endoftext|>");
    s(tiers, nt, T, "at end", "hello <|endoftext|>");
    s(tiers, nt, T, "cut by len", "hello <|endoftext|");
    s(tiers, nt, T, "nested longest", "<|im_start|>");
    s(tiers, nt, T, "nested middle", "<|im_x");
    s(tiers, nt, T, "nested short", "<|x");
    s(tiers, nt, T, "one byte", "<x");
    s(tiers, nt, T, "leftmost", "a<|im_end|>");
    s(tiers, nt, T, "overlap", "<|im<|im_end|>abcab");
    s(tiers, nt, T, "false positive", "\xBC\x3C\xBC\xBC\x7C\xFC\xFF");
    s(tiers, nt, T, "brackets", "x[gMASK][gMASK[gMAS]");
    s(tiers, nt, T, "phase 1", "a\n\nb xyz x\xE2\x96\x81\xE2\x96 </s></s");
    s(tiers, nt, T, "prose", "The quick brown fox jumps over the lazy dog, again and again and again.");
    {   /* the 200-byte token: whole, cut, at every offset of a 260-byte text */
        static uint8_t buf[600];
        memset(buf, '.', sizeof buf);
        for (uint64_t at = 0; at + sizeof longtok <= 300; at += 7) {
            memcpy(buf + at, longtok, sizeof longtok);
            sweep(tiers, nt, T, "long token", buf, 300, 0, at);
            sweep(tiers, nt, T, "long token cut", buf, at + sizeof longtok - 1u, 0, at);
            sweep(tiers, nt, T, "long token at end", buf, at + sizeof longtok, 0, at);
            buf[at + 150] ^= 1u;
            sweep(tiers, nt, T, "long token late mismatch", buf, 300, 0, at);
            memset(buf + at, '.', sizeof longtok);
        }
    }

    /* ---- geometry: lengths 0..255 plus a few, end flush / start flush + 0..63, every pos */
    static const uint64_t extra[] = { 256, 320, 511, 1000, 4096 + 17 };
    static const char *const frag[] = { "<|endoftext|>", "<|im_start|>", "<|im", "<", "<|", "[gMASK]", "[gM",
                                        "\xBC", "\xBC\xBC", "ab", "abc", "\xFF", "\n\n", "x", "xyz", "<s>",
                                        "\xE2\x96\x81", "\xE2\x96", "</s>", "\xC3", "hello ", "the ", ", ", "e" };
    uint64_t seed = 0x6b31u;
    static uint8_t text[4096 + 64];
    for (size_t li = 0; li < 256u + sizeof extra / sizeof extra[0]; li++) {
        uint64_t len = li < 256u ? li : extra[li - 256u];
        uint64_t m = 0;
        while (m < len) {
            const char *fr = frag[guard_rng(&seed) % (sizeof frag / sizeof frag[0])];
            size_t fl = strlen(fr);
            for (size_t i = 0; i < fl && m < len; i++) { text[m++] = (uint8_t)fr[i]; }
            if (guard_rng(&seed) % 3u == 0u) {
                for (uint64_t r = guard_rng(&seed) % 70u; r > 0u && m < len; r--) { text[m++] = '.'; }
            }
        }
        for (int where = 0; where < 65; where++) {
            guard_buf g = { 0 };
            uint64_t off = where == 0 ? 0u : (uint64_t)(where - 1);
            uint8_t *p = guard_alloc(&g, (size_t)len + 64u, where == 0 ? GUARD_END : GUARD_START, (size_t)off);
            if (!p) { fprintf(stderr, "guard_alloc failed\n"); return 2; }
            if (where == 0) {
                p += 64;                                   /* [p, p + len) ends at the guard page */
                for (int i = 1; i <= 64; i++) { p[-i] = "<|im_start|>"[(i - 1) % 12]; }
            } else {
                for (uint64_t i = 1; i <= off; i++) { p[-(int64_t)i] = "<|im_start|>"[(i - 1) % 12u]; }
                for (uint64_t i = 0; i < 64u; i++) { p[len + i] = "|endoftext|>"[i % 12u]; }
            }
            memcpy(p, text, (size_t)len);
            sweep(tiers, nt, T, where == 0 ? "end flush" : "start + off", p, len, where < 2, off * 7u + 3u);
            guard_free(&g);
        }
    }

    radix_cases(tiers, nt);
    printf("test_k1: %" PRIu64 " checks (%" PRIu64 " through the abi checker), %d failures\n", checks, abi_calls,
           failures);
    k1_tab_free(&full);
    k1_tab_free(&only0);
    k1_tab_free(&none);
    return failures ? 1 : 0;
}
