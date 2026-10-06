/* test_check.c: the segment memo's keyed check (src/core/check.c; SPEC §6, kernels.md §7 "the segment memo").
 * - known answers: under a fixed key, the check of sixteen texts (0 .. 12,345 bytes: a group's edges, a block's
 *   edges, several blocks) equals an independent reference's (a python transcription of the definition in check.c's
 *   header: bit-serial carry-less products, python integers mod 2^127 - 1), through the portable carry-less multiply
 *   and, where the cpu has one, PMULL / PCLMULQDQ;
 * - the cpu's path equals the portable one on random texts of every length 0 .. 4,500 and on 64 longer ones;
 * - one flipped bit anywhere in a 4,196-byte text changes the check (sampled: every byte, a rotating bit);
 * - a context without a key (the os gave no randomness at load: memo_keyed 0) has no memo and encodes as one.
 * The loaded context's key is replaced by the fixed one (the test owns the context). */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toks.h"
#include "core.h"
#include "cpu.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; if (failures < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } } while (0)

static uint64_t sm_x;
static uint64_t sm_next(void)
{
    uint64_t z = (sm_x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void fill(uint8_t *p, uint64_t n, uint64_t seed)    /* the reference's text(n, seed): little-endian words */
{
    sm_x = seed;
    for (uint64_t i = 0; i < n; i += 8u) {
        uint64_t w = sm_next();
        for (uint64_t j = 0; j < 8u && i + j < n; j++) { p[i + j] = (uint8_t)(w >> (8u * j)); }
    }
}

static const struct { uint64_t n, lo, hi; } KAT[] = {     /* check_ref.py, key = splitmix64 from 0x746F6B73 */
    {     0, 0x0000000000000000ull, 0x0000000000000000ull },
    {     1, 0x3929D468E090B73Bull, 0x138C32256F8C4BBCull },
    {     7, 0xC0CBB3358DD5B479ull, 0x0E64AD92EC8EBAF7ull },
    {     8, 0x50478A72AD74F675ull, 0x09E758533EB2C120ull },
    {    31, 0xEFE8D12C4921A811ull, 0x4844774BA76AB816ull },
    {    32, 0xE1AC1E984B31C25Eull, 0x235BEEF77210388Dull },
    {    33, 0x50B55F1D3A9EB018ull, 0x5324301612AF9FB0ull },
    {   255, 0x4159FA0EC1954D78ull, 0x4358F7F53CD29F84ull },
    {   256, 0x5AA146E49AF30843ull, 0x2D03576A85CD6E42ull },
    {  1000, 0x06A9EA876DD6A17Eull, 0x0759A9D6057231BBull },
    {  4095, 0x3653221E1E9FF736ull, 0x5F8F8E8E187D3AFEull },
    {  4096, 0xCECFA39EB6BD9E9Aull, 0x0043C72BA737F261ull },
    {  4097, 0x93FC8D1EA3C2C696ull, 0x185973D755958164ull },
    {  8191, 0xC195F811E9AD7865ull, 0x7E7A377A4B820CE0ull },
    {  8192, 0xF5660441336C783Full, 0x601620D839E19617ull },
    { 12345, 0xD85F10FC3C3B2BFFull, 0x7CA5C59CCCC74A4Dull },
};

int main(void)
{
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, "tests/data/compile/llama3style.json", NULL) != 0) { printf("test_check: load failed\n"); return 1; }
#if defined(TOKS_ARCH_ARM64)
    int hw = TOKS_CPU_HAS(ctx->cpu_features, TOKS_ARM64_PMULL);
#else
    int hw = TOKS_CPU_HAS(ctx->cpu_features, TOKS_X86_PCLMUL);
#endif
    sm_x = 0x746F6B73u;
    for (uint32_t i = 0; i < TOKS_MEMO_KEY_W; i++) { ctx->memo_key[i] = sm_next(); }
    toks_memo_rpow(ctx);

    static uint8_t t[1u << 16], u[1u << 16];
    uint64_t c0[2], c1[2], cases = 0;
    for (size_t i = 0; i < sizeof KAT / sizeof KAT[0]; i++) {
        fill(t, KAT[i].n, 0x5EEDu + KAT[i].n);
        toks_memo_check_with(ctx, 0, t, KAT[i].n, c0);
        CHECK(c0[0] == KAT[i].lo && c0[1] == KAT[i].hi, "kat n %" PRIu64 ": portable %016" PRIx64 "%016" PRIx64, KAT[i].n,
              c0[1], c0[0]);
        if (hw) {
            toks_memo_check_with(ctx, 1, t, KAT[i].n, c1);
            CHECK(c1[0] == KAT[i].lo && c1[1] == KAT[i].hi, "kat n %" PRIu64 ": cpu %016" PRIx64 "%016" PRIx64, KAT[i].n,
                  c1[1], c1[0]);
        }
        cases++;
    }
    if (hw) {
        fill(t, sizeof t, 1u);
        for (uint64_t n = 0; n <= 4500u; n++) {          /* every length, at a moving offset */
            toks_memo_check_with(ctx, 0, t + n % 61u, n, c0), toks_memo_check_with(ctx, 1, t + n % 61u, n, c1);
            CHECK(c0[0] == c1[0] && c0[1] == c1[1], "length %" PRIu64 ": the cpu's path differs from the portable one", n);
            cases++;
        }
        for (uint64_t r = 0; r < 64u; r++) {
            uint64_t n = 4500u + sm_next() % (sizeof t - 4600u);
            toks_memo_check_with(ctx, 0, t + r, n, c0), toks_memo_check_with(ctx, 1, t + r, n, c1);
            CHECK(c0[0] == c1[0] && c0[1] == c1[1], "length %" PRIu64 ": the cpu's path differs from the portable one", n);
            cases++;
        }
    }
    fill(t, 4196u, 7u);
    toks_memo_check(ctx, t, 4196u, c0);
    for (uint64_t i = 0; i < 4196u; i++) {
        memcpy(u, t, 4196u);
        u[i] ^= (uint8_t)(1u << (i % 8u));
        toks_memo_check(ctx, u, 4196u, c1);
        CHECK(c0[0] != c1[0] || c0[1] != c1[1], "byte %" PRIu64 " flipped: same check", i);
        cases++;
    }
    /* a context whose key the os did not give (load.c: memo_keyed 0) has no memo: its scratch is sized and laid out
     * as with TOKS_SCRATCH_MEMO_MIB(0), and encodes as one */
    {
        uint64_t with = toks_scratch_bytes(ctx, 4096u, 0u), none = toks_scratch_bytes(ctx, 4096u, TOKS_SCRATCH_MEMO_MIB(0));
        CHECK(with > none, "a keyed context's default scratch holds a memo (%" PRIu64 " vs %" PRIu64 ")", with, none);
        ctx->memo_keyed = 0u;
        uint64_t b = toks_scratch_bytes(ctx, 4096u, 0u), mb = 1u;
        CHECK(b == none, "no key: the default scratch is the memo-off one (%" PRIu64 " vs %" PRIu64 ")", b, none);
        void *scr = malloc((size_t)b);
        static uint32_t ids[8192];
        CHECK(scr != NULL && toks_scratch_init(ctx, scr, b, 0u) == 0 && toks_scr_memo(toks_scr_header(scr), &mb) == NULL &&
              mb == 0u, "no key: init lays out no memo");
        fill(t, 4000u, 9u);
        for (uint64_t i = 0; i < 4000u; i++) { t[i] = (uint8_t)('a' + t[i] % 26u); }
        int64_t n0 = scr != NULL ? toks_encode(ctx, t, 4000u, 0u, ids, 8192u, scr) : -1;
        int64_t n1 = scr != NULL ? toks_encode(ctx, t, 4000u, 0u, ids, 8192u, scr) : -1;
        CHECK(n0 > 0 && n0 == n1, "no key: encode works (%" PRId64 ", %" PRId64 ")", n0, n1);
        free(scr);
        ctx->memo_keyed = 1u;
        cases += 4u;
    }
    printf("test_check: %" PRIu64 " cases, the cpu's carry-less multiply %s, %d failures\n", cases,
           hw ? "compared with the portable one" : "absent (portable only)", failures);
    toks_unload(ctx);
    return failures != 0;
}
