/* tools/bench/check_bench.c: a keyed check against memcmp on the segment memo's hit path (kernels.md §7 "the segment
 * memo", decision 18: measured and not taken). The check is the one toks/memo-check (#17, shelved) built, copied here
 * so the negative result can be measured again on any core:
 *   CLNH (carry-less NH, Lemire and Kaser) over 4 KiB blocks of 64-bit words in two Toeplitz passes, then a
 *   polynomial modulo 2^127 - 1 over the passes' sums and the length (a block's four steps as one 256-bit sum of
 *   independent products under r^1..r^5, reduced once); 2^-121 for two different 4 KiB segments chosen without the
 *   key. PMULL / PCLMULQDQ when the compiler targets them (-mcpu=native / -march=native), and a portable carry-less
 *   multiply (BearSSL's ghash_ctmul64) either way. It checks its own value first: #17's sixteen known answers
 *   (tests/c/test_check.c there, from an independent python reference) on both paths.
 * Measured, against memcmp of the bytes a record holds:
 *   alone   one segment in L1, the key hot: best of 5 runs of a fixed count of calls, ns a call and GB/s
 *   replay  <segments> segments of random printable text through records laid out as api.c's (64-aligned: a 32-byte
 *           head, then the bytes (memcmp) or the 16-byte check, then the ids), the ids copied out: best of <passes>
 *           passes, ns a segment (2 MB of text, 3.9 MB of records with the bytes and 1.9 MB with the check, at the
 *           defaults: where they sit is what the replay measures)
 *
 *   check_bench [segment bytes (4096)] [ids a segment (975)] [segments (480)] [passes (30)]
 *       build: $CC -std=c17 -O3 -mcpu=native (x86: -march=native) -o build/check-bench tools/bench/check_bench.c
 *       e.g. taskset -c 8 ./build/check-bench
 */
#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__) && defined(__ARM_FEATURE_AES)
#  include <arm_neon.h>
#  define HW 1
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__PCLMUL__)
#  include <emmintrin.h>
#  include <wmmintrin.h>
#  define HW 1
#else
#  define HW 0
#endif

#define BLOCK 4096u                                 /* a check block: 4 KiB of 64-bit words */
#define KEY_W (BLOCK / 8u + 6u)                     /* a block's words + one group (CLNH), 2 (the polynomial) */

typedef unsigned __int128 u128;
#define P127 (((u128)1 << 127) - 1u)

static uint64_t key[KEY_W];
static u128 rp[5];                                  /* r^1..r^5 */

static u128 p127_mul(u128 a, u128 b)                /* a b mod p (a, b < 2^127) */
{
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    u128 t00 = (u128)a0 * b0, mid = (u128)a0 * b1 + (u128)a1 * b0, t11 = (u128)a1 * b1;
    u128 s1 = (t00 >> 64) + (uint64_t)mid;
    u128 s2 = (s1 >> 64) + (mid >> 64) + (uint64_t)t11;
    uint64_t l1 = (uint64_t)s1, l2 = (uint64_t)s2, l3 = (uint64_t)((s2 >> 64) + (t11 >> 64));
    u128 x = (((u128)(l1 & 0x7FFFFFFFFFFFFFFFull) << 64) | (uint64_t)t00) +
             (((u128)l3 << 65) | ((u128)l2 << 1) | (l1 >> 63));
    x = (x & P127) + (x >> 127);
    return x >= P127 ? x - P127 : x;
}

typedef struct { u128 lo, hi; } u256;
static void acc128(u256 *a, u128 x, u128 y)         /* a += x y */
{
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64), y0 = (uint64_t)y, y1 = (uint64_t)(y >> 64);
    u128 p00 = (u128)x0 * y0, p01 = (u128)x0 * y1, p10 = (u128)x1 * y0, p11 = (u128)x1 * y1;
    u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    u128 lo = (mid << 64) | (uint64_t)p00, hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64), n = a->lo + lo;
    a->hi += hi + (n < lo), a->lo = n;
}
static void acc64(u256 *a, uint64_t x, u128 y)      /* a += x y, x < 2^64 */
{
    u128 p0 = (u128)x * (uint64_t)y, p1 = (u128)x * (uint64_t)(y >> 64);
    u128 mid = (p0 >> 64) + (uint64_t)p1;
    u128 lo = (mid << 64) | (uint64_t)p0, hi = (p1 >> 64) + (mid >> 64), n = a->lo + lo;
    a->hi += hi + (n < lo), a->lo = n;
}
static u128 p127_red(u256 a)                        /* a mod p, a < 2^255 */
{
    u128 x = (a.lo & P127) + (a.lo >> 127) + ((a.hi << 1) & P127) + (a.hi >> 126);
    x = (x & P127) + (x >> 127);
    return x >= P127 ? x - P127 : x;
}
static u128 p127_block(u128 h, const uint64_t s[4]) /* h r^4 + s0 r^3 + s1 r^2 + s2 r + s3 */
{
    u256 a = { s[3], 0u };
    acc128(&a, h, rp[3]), acc64(&a, s[0], rp[2]), acc64(&a, s[1], rp[1]), acc64(&a, s[2], rp[0]);
    return p127_red(a);
}
static u128 p127_last(u128 h, const uint64_t s[4], uint64_t n)   /* (the block) r + n */
{
    u256 a = { n, 0u };
    acc128(&a, h, rp[4]), acc64(&a, s[0], rp[3]), acc64(&a, s[1], rp[2]), acc64(&a, s[2], rp[1]), acc64(&a, s[3], rp[0]);
    return p127_red(a);
}

static uint64_t bmul64(uint64_t x, uint64_t y)      /* the low half of the carry-less product, constant time */
{
    const uint64_t m0 = 0x1111111111111111ull, m1 = m0 << 1, m2 = m0 << 2, m3 = m0 << 3;
    uint64_t x0 = x & m0, x1 = x & m1, x2 = x & m2, x3 = x & m3;
    uint64_t y0 = y & m0, y1 = y & m1, y2 = y & m2, y3 = y & m3;
    uint64_t z0 = (x0 * y0) ^ (x1 * y3) ^ (x2 * y2) ^ (x3 * y1);
    uint64_t z1 = (x0 * y1) ^ (x1 * y0) ^ (x2 * y3) ^ (x3 * y2);
    uint64_t z2 = (x0 * y2) ^ (x1 * y1) ^ (x2 * y0) ^ (x3 * y3);
    uint64_t z3 = (x0 * y3) ^ (x1 * y2) ^ (x2 * y1) ^ (x3 * y0);
    return (z0 & m0) | (z1 & m1) | (z2 & m2) | (z3 & m3);
}
static uint64_t rev64(uint64_t x)
{
    x = ((x & 0x5555555555555555ull) << 1) | ((x >> 1) & 0x5555555555555555ull);
    x = ((x & 0x3333333333333333ull) << 2) | ((x >> 2) & 0x3333333333333333ull);
    x = ((x & 0x0F0F0F0F0F0F0F0Full) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0Full);
    x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFull);
    return (x << 32) | (x >> 32);
}
static void clmul_xor(uint64_t x, uint64_t y, uint64_t s[2])
{
    s[0] ^= bmul64(x, y);
    s[1] ^= rev64(bmul64(rev64(x), rev64(y))) >> 1;
}
/* level 1 over n32 groups of q under k, XORed into s: s[0..1] pass 0's 128-bit sum, s[2..3] pass 1's */
static void clnh_c(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    for (uint64_t i = 0; i < n32; i++) {
        uint64_t m[4];
        memcpy(m, q + 32u * i, 32);
        const uint64_t *a = k + 4u * i, *b = a + 4u;
        clmul_xor(m[0] ^ a[0], m[2] ^ a[2], s), clmul_xor(m[1] ^ a[1], m[3] ^ a[3], s);
        clmul_xor(m[0] ^ b[0], m[2] ^ b[2], s + 2), clmul_xor(m[1] ^ b[1], m[3] ^ b[3], s + 2);
    }
}
#if HW && defined(__aarch64__)
#  define CLNH_GROUP(ml, mh, k_l, k_h, n_l, n_h, a0, a1, b0, b1) do { \
        uint64x2_t x_ = veorq_u64(ml, k_l), y_ = veorq_u64(mh, k_h); \
        a0 = veorq_u64(a0, vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(x_, 0), vgetq_lane_u64(y_, 0)))); \
        a1 = veorq_u64(a1, vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(x_), vreinterpretq_p64_u64(y_)))); \
        x_ = veorq_u64(ml, n_l), y_ = veorq_u64(mh, n_h); \
        b0 = veorq_u64(b0, vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(x_, 0), vgetq_lane_u64(y_, 0)))); \
        b1 = veorq_u64(b1, vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(x_), vreinterpretq_p64_u64(y_)))); \
    } while (0)
static void clnh_hw(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    uint64x2_t a0 = vld1q_u64(s), b0 = vld1q_u64(s + 2), z = vdupq_n_u64(0), a1 = z, b1 = z, c0 = z, c1 = z, d0 = z, d1 = z;
    uint64x2_t kl = vld1q_u64(k), kh = vld1q_u64(k + 2);
    uint64_t i = 0;
    for (; i + 2u <= n32; i += 2u) {
        uint64x2_t ml = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i)), mh = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 16u));
        uint64x2_t pl = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 32u)), ph = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 48u));
        uint64x2_t nl = vld1q_u64(k + 4u * i + 4u), nh = vld1q_u64(k + 4u * i + 6u);
        uint64x2_t ol = vld1q_u64(k + 4u * i + 8u), oh = vld1q_u64(k + 4u * i + 10u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
        CLNH_GROUP(pl, ph, nl, nh, ol, oh, c0, c1, d0, d1);
        kl = ol, kh = oh;
    }
    if (i < n32) {
        uint64x2_t ml = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i)), mh = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 16u));
        uint64x2_t nl = vld1q_u64(k + 4u * i + 4u), nh = vld1q_u64(k + 4u * i + 6u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
    }
    vst1q_u64(s, veorq_u64(veorq_u64(a0, a1), veorq_u64(c0, c1)));
    vst1q_u64(s + 2, veorq_u64(veorq_u64(b0, b1), veorq_u64(d0, d1)));
}
#elif HW
#  define CLNH_GROUP(ml, mh, k_l, k_h, n_l, n_h, a0, a1, b0, b1) do { \
        __m128i x_ = _mm_xor_si128(ml, k_l), y_ = _mm_xor_si128(mh, k_h); \
        a0 = _mm_xor_si128(a0, _mm_clmulepi64_si128(x_, y_, 0x00)); \
        a1 = _mm_xor_si128(a1, _mm_clmulepi64_si128(x_, y_, 0x11)); \
        x_ = _mm_xor_si128(ml, n_l), y_ = _mm_xor_si128(mh, n_h); \
        b0 = _mm_xor_si128(b0, _mm_clmulepi64_si128(x_, y_, 0x00)); \
        b1 = _mm_xor_si128(b1, _mm_clmulepi64_si128(x_, y_, 0x11)); \
    } while (0)
#  define LD(p) _mm_loadu_si128((const __m128i *)(const void *)(p))
static void clnh_hw(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    __m128i a0 = LD(s), b0 = LD(s + 2), z = _mm_setzero_si128(), a1 = z, b1 = z, c0 = z, c1 = z, d0 = z, d1 = z;
    __m128i kl = LD(k), kh = LD(k + 2);
    uint64_t i = 0;
    for (; i + 2u <= n32; i += 2u) {
        __m128i ml = LD(q + 32u * i), mh = LD(q + 32u * i + 16u), pl = LD(q + 32u * i + 32u), ph = LD(q + 32u * i + 48u);
        __m128i nl = LD(k + 4u * i + 4u), nh = LD(k + 4u * i + 6u), ol = LD(k + 4u * i + 8u), oh = LD(k + 4u * i + 10u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
        CLNH_GROUP(pl, ph, nl, nh, ol, oh, c0, c1, d0, d1);
        kl = ol, kh = oh;
    }
    if (i < n32) {
        __m128i ml = LD(q + 32u * i), mh = LD(q + 32u * i + 16u), nl = LD(k + 4u * i + 4u), nh = LD(k + 4u * i + 6u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
    }
    _mm_storeu_si128((__m128i *)(void *)s, _mm_xor_si128(_mm_xor_si128(a0, a1), _mm_xor_si128(c0, c1)));
    _mm_storeu_si128((__m128i *)(void *)(s + 2), _mm_xor_si128(_mm_xor_si128(b0, b1), _mm_xor_si128(d0, d1)));
}
#endif

static void check(int hw, const uint8_t *g, uint64_t n, uint64_t c[2])
{
    u128 h = 0u;
    for (uint64_t at = 0; at < n; at += BLOCK) {
        uint64_t b = n - at < BLOCK ? n - at : BLOCK, s[4] = { 0u, 0u, 0u, 0u };
#if HW
        if (hw) { clnh_hw(key, g + at, b / 32u, s); } else { clnh_c(key, g + at, b / 32u, s); }
#else
        (void)hw, clnh_c(key, g + at, b / 32u, s);
#endif
        if (b % 32u != 0u) {                        /* the last bytes, zero-padded to a group */
            uint8_t t[32] = { 0 };
            memcpy(t, g + at + (b & ~31ull), (size_t)(b % 32u));
#if HW
            if (hw) { clnh_hw(key + 4u * (b / 32u), t, 1u, s); } else { clnh_c(key + 4u * (b / 32u), t, 1u, s); }
#else
            clnh_c(key + 4u * (b / 32u), t, 1u, s);
#endif
        }
        h = at + BLOCK < n ? p127_block(h, s) : p127_last(h, s, n);
    }
    c[0] = (uint64_t)h, c[1] = (uint64_t)(h >> 64);
}

static uint64_t sm_x;
static uint64_t sm_next(void)                       /* splitmix64 */
{
    uint64_t z = (sm_x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void key_set(uint64_t seed)                   /* the key words, then r < 2^126 and its powers */
{
    sm_x = seed;
    for (uint32_t i = 0; i < KEY_W; i++) { key[i] = sm_next(); }
    u128 r = (((u128)key[KEY_W - 1u] << 64) | key[KEY_W - 2u]) >> 2, x = r;
    for (int i = 0; i < 5; i++) { rp[i] = x, x = p127_mul(x, r); }
}
static void fill(uint8_t *p, uint64_t n, uint64_t seed)    /* test_check.c's text(n, seed) */
{
    sm_x = seed;
    for (uint64_t i = 0; i < n; i += 8u) {
        uint64_t w = sm_next();
        for (uint64_t j = 0; j < 8u && i + j < n; j++) { p[i + j] = (uint8_t)(w >> (8u * j)); }
    }
}
static const struct { uint64_t n, lo, hi; } KAT[] = {     /* #17's tests/c/test_check.c, key splitmix64 from "toks" */
    { 0, 0x0000000000000000ull, 0x0000000000000000ull }, { 1, 0x3929D468E090B73Bull, 0x138C32256F8C4BBCull },
    { 7, 0xC0CBB3358DD5B479ull, 0x0E64AD92EC8EBAF7ull }, { 8, 0x50478A72AD74F675ull, 0x09E758533EB2C120ull },
    { 31, 0xEFE8D12C4921A811ull, 0x4844774BA76AB816ull }, { 32, 0xE1AC1E984B31C25Eull, 0x235BEEF77210388Dull },
    { 33, 0x50B55F1D3A9EB018ull, 0x5324301612AF9FB0ull }, { 255, 0x4159FA0EC1954D78ull, 0x4358F7F53CD29F84ull },
    { 256, 0x5AA146E49AF30843ull, 0x2D03576A85CD6E42ull }, { 1000, 0x06A9EA876DD6A17Eull, 0x0759A9D6057231BBull },
    { 4095, 0x3653221E1E9FF736ull, 0x5F8F8E8E187D3AFEull }, { 4096, 0xCECFA39EB6BD9E9Aull, 0x0043C72BA737F261ull },
    { 4097, 0x93FC8D1EA3C2C696ull, 0x185973D755958164ull }, { 8191, 0xC195F811E9AD7865ull, 0x7E7A377A4B820CE0ull },
    { 8192, 0xF5660441336C783Full, 0x601620D839E19617ull }, { 12345, 0xD85F10FC3C3B2BFFull, 0x7CA5C59CCCC74A4Dull },
};

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static volatile uint64_t sink;

int main(int argc, char **argv)
{
    uint64_t seg = argc > 1 ? strtoull(argv[1], NULL, 10) : 4096u, k = argc > 2 ? strtoull(argv[2], NULL, 10) : 975u;
    uint64_t nseg = argc > 3 ? strtoull(argv[3], NULL, 10) : 480u, passes = argc > 4 ? strtoull(argv[4], NULL, 10) : 30u;
    static uint8_t t[12345];
    int bad = 0;
    if (seg < 64u || k == 0u || nseg == 0u || passes == 0u) {   /* the hash windows and the sinks read a segment's ends */
        printf("usage: check-bench [segment bytes >= 64] [ids] [segments] [passes] (each >= 1)\n");
        return 2;
    }
    key_set(0x746F6B73u);
    for (size_t i = 0; i < sizeof KAT / sizeof KAT[0]; i++) {
        uint64_t c[2];
        fill(t, KAT[i].n, 0x5EEDu + KAT[i].n);
        for (int hw = 0; hw <= HW; hw++) {
            check(hw, t, KAT[i].n, c);
            if (c[0] != KAT[i].lo || c[1] != KAT[i].hi) { printf("KAT MISMATCH n %" PRIu64 " hw %d\n", KAT[i].n, hw), bad = 1; }
        }
    }
    if (bad) { return 1; }
    sm_x = 1u;
    uint64_t textn = seg * nseg, pad = (seg + 7u) & ~7ull;
    uint64_t rec_b = (32u + pad + 4u * k + 63u) & ~63ull, rec_c = (48u + 4u * k + 63u) & ~63ull;   /* api.c memo_need */
    uint8_t *text = malloc((size_t)textn), *ring_b = malloc((size_t)(rec_b * nseg)), *ring_c = malloc((size_t)(rec_c * nseg));
    uint32_t *out = malloc((size_t)(4u * k * nseg));
    if (text == NULL || ring_b == NULL || ring_c == NULL || out == NULL) { return 1; }
    for (uint64_t i = 0; i < textn; i++) { text[i] = (uint8_t)(32u + sm_next() % 95u); }
    __asm__ volatile("" : : "r"(text), "r"(ring_b), "r"(ring_c), "r"(out) : "memory");   /* they escape: the
                                                       compiler may not hoist a compare or a check out of a timed loop */
    for (uint64_t s = 0; s < nseg; s++) {           /* both rings: a head, the bytes or the check, the ids */
        uint8_t *a = ring_b + rec_b * s, *b = ring_c + rec_c * s;
        uint64_t c[2];
        memset(a, 0, 32u), memset(b, 0, 32u);
        memcpy(a + 32u, text + seg * s, (size_t)seg);
        check(HW, text + seg * s, seg, c);
        memcpy(b + 32u, c, 16u);
        for (uint64_t i = 0; i < k; i++) {
            uint32_t v = (uint32_t)(s * k + i);
            memcpy(a + 32u + pad + 4u * i, &v, 4u), memcpy(b + 48u + 4u * i, &v, 4u);
        }
    }
    printf("CHECK_BENCH kat 16/16 (portable%s) seg %" PRIu64 " ids %" PRIu64 " segments %" PRIu64 " text %.2f MB records "
           "%.2f MB (bytes) %.2f MB (check) passes %" PRIu64 " clmul %d\n", HW ? ", cpu" : "", seg, k, nseg,
           (double)textn / 1e6, (double)(rec_b * nseg) / 1e6, (double)(rec_c * nseg) / 1e6, passes, HW);
    for (int f = 0; f < 3; f++) {                   /* alone: memcmp, the check, the check's portable multiply */
        uint64_t calls = f == 2 ? 2000u : 50000u, c[2] = { 0u, 0u };
        double best = 1e30;
        for (int r = 0; r < 5; r++) {
            double t0 = now();
            for (uint64_t i = 0; i < calls; i++) {
                if (f == 0) {
                    c[0] += (uint64_t)memcmp(text, ring_b + 32u, (size_t)seg);
                } else {
                    check(f == 1 ? HW : 0, text, seg, c);
                }
                __asm__ volatile("" ::: "memory");
            }
            double dt = now() - t0;
            best = dt < best ? dt : best;
        }
        sink += c[0];
        printf("alone  %-15s %8.1f ns  %6.2f GB/s\n", f == 0 ? "memcmp" : f == 1 ? "check" : "check-portable",
               best / (double)calls * 1e9, (double)(seg * calls) / best / 1e9);
    }
    for (int f = 0; f < 2; f++) {                   /* replay: every segment hits, its ids copied out */
        double best = 1e30;
        uint64_t miss = 0u;
        for (uint64_t p = 0; p < passes; p++) {
            double t0 = now();
            for (uint64_t s = 0; s < nseg; s++) {
                const uint8_t *g = text + seg * s;
                if (f == 0) {
                    const uint8_t *a = ring_b + rec_b * s;
                    if (memcmp(a + 32u, g, (size_t)seg) != 0) { miss++; continue; }
                    memcpy(out + k * s, a + 32u + pad, (size_t)(4u * k));
                } else {
                    const uint8_t *b = ring_c + rec_c * s;
                    uint64_t c[2], w[2];
                    check(HW, g, seg, c);
                    memcpy(w, b + 32u, 16u);
                    if (c[0] != w[0] || c[1] != w[1]) { miss++; continue; }
                    memcpy(out + k * s, b + 48u, (size_t)(4u * k));
                }
            }
            __asm__ volatile("" ::: "memory");
            double dt = now() - t0;
            best = dt < best ? dt : best;
        }
        sink += out[k * nseg - 1u];
        printf("replay %-15s %8.1f ns a segment  %8.1f MB/s  miss %" PRIu64 "\n", f == 0 ? "memcmp+copy" : "check+copy",
               best / (double)nseg * 1e9, (double)textn / best / 1e6, miss);
    }
    free(text), free(ring_b), free(ring_c), free(out);
    return 0;
}
