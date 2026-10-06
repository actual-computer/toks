/* check.c: the segment memo's keyed check (SPEC §6; kernels.md §7 "the segment memo"; docs/notes/c-core.md §check.c.1)
 *
 * A memo record keeps its segment's 16-byte check and its ids, not its bytes; a hit needs the probe's check to equal
 * the record's. The check is a universal hash keyed by the context's secret (ctx->memo_key, drawn at load), so two
 * different segments of equal length chosen without the key share it with probability <= 2^-128 + (4 b + 1) / 2^126
 * (b = the 4 KiB blocks of the segment: 2^-121 at 4 KiB, 2^-107 at 2^29 bytes).
 *
 * Level 1, CLNH (carry-less NH; Lemire & Kaser, "Faster 64-bit universal hashing using carry-less multiplications"):
 * per 4 KiB block of 64-bit words, cut in 32-byte groups m[4i..4i+3], two passes keyed one group apart (Toeplitz):
 *     pass t: XOR over i, j in {0, 1} of (m[4i + j] ^ k[4i + j + 4t]) clmul (m[4i + j + 2] ^ k[4i + j + 2 + 4t])
 * a 128-bit sum per pass. A clmul by a nonzero polynomial is injective, so for two different blocks the last differing
 * pair's free key word (one group fresher in each pass) makes each pass collide with probability <= 2^-64. The last
 * partial group of a segment is zero-padded (records only meet segments of the same length).
 * Level 2: a polynomial modulo p = 2^127 - 1 in the key r < 2^126 over the passes' sums as 64-bit halves, then the
 * length: <= (4 b + 1) / 2^126 for two different level-1 sequences.
 *
 * Three ways to the same value: PMULL (arm64) and PCLMULQDQ (x86-64) where the cpu has them (ctx->cpu_features), else a
 * portable carry-less multiply (integer multiplies with holes, BearSSL's ghash_ctmul64). Never by tier: a forced scalar
 * tier computes what the shipping tier does on the same cpu, and so writes the same records (T2). */
#include "core.h"
#include "cpu.h"
#if defined(TOKS_ARCH_ARM64)
#  include <arm_neon.h>
#else
#  include <emmintrin.h>
#  include <wmmintrin.h>
#endif

typedef unsigned __int128 u128;
#define P127 (((u128)1 << 127) - 1u)

/* a b mod p (a, b < 2^127) */
static u128 p127_mul(u128 a, u128 b)
{
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64), b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    u128 t00 = (u128)a0 * b0, mid = (u128)a0 * b1 + (u128)a1 * b0, t11 = (u128)a1 * b1;
    u128 s1 = (t00 >> 64) + (uint64_t)mid;
    u128 s2 = (s1 >> 64) + (mid >> 64) + (uint64_t)t11;
    uint64_t l1 = (uint64_t)s1, l2 = (uint64_t)s2, l3 = (uint64_t)((s2 >> 64) + (t11 >> 64));
    u128 x = (((u128)(l1 & 0x7FFFFFFFFFFFFFFFull) << 64) | (uint64_t)t00) +
             (((u128)l3 << 65) | ((u128)l2 << 1) | (l1 >> 63));          /* 2^127 = 1 mod p */
    x = (x & P127) + (x >> 127);
    return x >= P127 ? x - P127 : x;
}

/* a block's four sums in one step, the value of four steps h = h r + c mod p: the products are independent, summed in
 * 256 bits and reduced once (r^1..r^5 from load: ctx->memo_rpow) */
typedef struct { u128 lo, hi; } u256;
static void acc128(u256 *a, u128 x, u128 y)             /* a += x y */
{
    uint64_t x0 = (uint64_t)x, x1 = (uint64_t)(x >> 64), y0 = (uint64_t)y, y1 = (uint64_t)(y >> 64);
    u128 p00 = (u128)x0 * y0, p01 = (u128)x0 * y1, p10 = (u128)x1 * y0, p11 = (u128)x1 * y1;
    u128 mid = (p00 >> 64) + (uint64_t)p01 + (uint64_t)p10;
    u128 lo = (mid << 64) | (uint64_t)p00, hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64), n = a->lo + lo;
    a->hi += hi + (n < lo), a->lo = n;
}
static void acc64(u256 *a, uint64_t x, u128 y)          /* a += x y, x < 2^64 */
{
    u128 p0 = (u128)x * (uint64_t)y, p1 = (u128)x * (uint64_t)(y >> 64);
    u128 mid = (p0 >> 64) + (uint64_t)p1;
    u128 lo = (mid << 64) | (uint64_t)p0, hi = (p1 >> 64) + (mid >> 64), n = a->lo + lo;
    a->hi += hi + (n < lo), a->lo = n;
}
static u128 p127_red(u256 a)                            /* a mod p, a < 2^255: 2^127 = 1 mod p */
{
    u128 x = (a.lo & P127) + (a.lo >> 127) + ((a.hi << 1) & P127) + (a.hi >> 126);
    x = (x & P127) + (x >> 127);
    return x >= P127 ? x - P127 : x;
}
/* a block that is not the segment's last: h r^4 + s0 r^3 + s1 r^2 + s2 r + s3; the last one takes the length too:
 * (that) r + n = h r^5 + s0 r^4 + s1 r^3 + s2 r^2 + s3 r + n (r[i] = r^(i + 1)) */
static u128 p127_block(u128 h, const u128 r[5], const uint64_t s[4])
{
    u256 a = { s[3], 0u };
    acc128(&a, h, r[3]), acc64(&a, s[0], r[2]), acc64(&a, s[1], r[1]), acc64(&a, s[2], r[0]);
    return p127_red(a);
}
static u128 p127_last(u128 h, const u128 r[5], const uint64_t s[4], uint64_t n)
{
    u256 a = { n, 0u };
    acc128(&a, h, r[4]), acc64(&a, s[0], r[3]), acc64(&a, s[1], r[2]), acc64(&a, s[2], r[1]), acc64(&a, s[3], r[0]);
    return p127_red(a);
}

/* the low 64 bits of the carry-less product: four interleaved bit planes, so an integer multiply's carries land only
 * in the bits each plane's mask drops (16 set bits a plane: a column sums to at most 16, whose carry passes bit 63) */
static uint64_t bmul64(uint64_t x, uint64_t y)
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

/* x clmul y into s[0] (low) and s[1] (high), XORed in: the high half is the reversed operands' low half, reversed */
static void clmul_xor(uint64_t x, uint64_t y, uint64_t s[2])
{
    s[0] ^= bmul64(x, y);
    s[1] ^= rev64(bmul64(rev64(x), rev64(y))) >> 1;
}

/* level 1 over n32 whole groups of q under the key k (the block's words from the first group's), XORed into s: s[0..1]
 * pass 0's 128-bit sum, s[2..3] pass 1's */
static void clnh_c(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    for (uint64_t i = 0; i < n32; i++) {               /* bound: 128 groups a block */
        uint64_t m[4];
        memcpy(m, q + 32u * i, 32);
        const uint64_t *a = k + 4u * i, *b = a + 4u;
        clmul_xor(m[0] ^ a[0], m[2] ^ a[2], s), clmul_xor(m[1] ^ a[1], m[3] ^ a[3], s);
        clmul_xor(m[0] ^ b[0], m[2] ^ b[2], s + 2), clmul_xor(m[1] ^ b[1], m[3] ^ b[3], s + 2);
    }
}

#if defined(TOKS_ARCH_ARM64)
/* one group's two passes into (a, b): (m ^ k) clmul (m' ^ k') for each lane pair, pass 0 under k, pass 1 under n */
#  define CLNH_GROUP(ml, mh, k_l, k_h, n_l, n_h, a0, a1, b0, b1) do { \
        uint64x2_t x_ = veorq_u64(ml, k_l), y_ = veorq_u64(mh, k_h); \
        a0 = veorq_u64(a0, vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(x_, 0), vgetq_lane_u64(y_, 0)))); \
        a1 = veorq_u64(a1, vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(x_), vreinterpretq_p64_u64(y_)))); \
        x_ = veorq_u64(ml, n_l), y_ = veorq_u64(mh, n_h); \
        b0 = veorq_u64(b0, vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(x_, 0), vgetq_lane_u64(y_, 0)))); \
        b1 = veorq_u64(b1, vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(x_), vreinterpretq_p64_u64(y_)))); \
    } while (0)
__attribute__((target("aes")))                          /* PMULL: the cpu has it (toks_memo_check checks) */
static void clnh_hw(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    uint64x2_t a0 = vld1q_u64(s), b0 = vld1q_u64(s + 2), z = vdupq_n_u64(0), a1 = z, b1 = z, c0 = z, c1 = z, d0 = z, d1 = z;
    uint64x2_t kl = vld1q_u64(k), kh = vld1q_u64(k + 2);
    uint64_t i = 0;
    for (; i + 2u <= n32; i += 2u) {                    /* bound: 64 steps a block; two groups a step */
        uint64x2_t ml = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i)), mh = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 16u));
        uint64x2_t pl = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 32u)), ph = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 48u));
        uint64x2_t nl = vld1q_u64(k + 4u * i + 4u), nh = vld1q_u64(k + 4u * i + 6u);
        uint64x2_t ol = vld1q_u64(k + 4u * i + 8u), oh = vld1q_u64(k + 4u * i + 10u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
        CLNH_GROUP(pl, ph, nl, nh, ol, oh, c0, c1, d0, d1);
        kl = ol, kh = oh;
    }
    if (i < n32) {                                      /* an odd last group */
        uint64x2_t ml = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i)), mh = vreinterpretq_u64_u8(vld1q_u8(q + 32u * i + 16u));
        uint64x2_t nl = vld1q_u64(k + 4u * i + 4u), nh = vld1q_u64(k + 4u * i + 6u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
    }
    vst1q_u64(s, veorq_u64(veorq_u64(a0, a1), veorq_u64(c0, c1)));
    vst1q_u64(s + 2, veorq_u64(veorq_u64(b0, b1), veorq_u64(d0, d1)));
}
#  define CLMUL_FEATURE TOKS_ARM64_PMULL
#else
#  define CLNH_GROUP(ml, mh, k_l, k_h, n_l, n_h, a0, a1, b0, b1) do { \
        __m128i x_ = _mm_xor_si128(ml, k_l), y_ = _mm_xor_si128(mh, k_h); \
        a0 = _mm_xor_si128(a0, _mm_clmulepi64_si128(x_, y_, 0x00)); \
        a1 = _mm_xor_si128(a1, _mm_clmulepi64_si128(x_, y_, 0x11)); \
        x_ = _mm_xor_si128(ml, n_l), y_ = _mm_xor_si128(mh, n_h); \
        b0 = _mm_xor_si128(b0, _mm_clmulepi64_si128(x_, y_, 0x00)); \
        b1 = _mm_xor_si128(b1, _mm_clmulepi64_si128(x_, y_, 0x11)); \
    } while (0)
#  define LD(p) _mm_loadu_si128((const __m128i *)(const void *)(p))
__attribute__((target("pclmul")))                       /* PCLMULQDQ: the cpu has it (toks_memo_check checks) */
static void clnh_hw(const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    __m128i a0 = LD(s), b0 = LD(s + 2), z = _mm_setzero_si128(), a1 = z, b1 = z, c0 = z, c1 = z, d0 = z, d1 = z;
    __m128i kl = LD(k), kh = LD(k + 2);
    uint64_t i = 0;
    for (; i + 2u <= n32; i += 2u) {                    /* bound: 64 steps a block; two groups a step */
        __m128i ml = LD(q + 32u * i), mh = LD(q + 32u * i + 16u), pl = LD(q + 32u * i + 32u), ph = LD(q + 32u * i + 48u);
        __m128i nl = LD(k + 4u * i + 4u), nh = LD(k + 4u * i + 6u), ol = LD(k + 4u * i + 8u), oh = LD(k + 4u * i + 10u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
        CLNH_GROUP(pl, ph, nl, nh, ol, oh, c0, c1, d0, d1);
        kl = ol, kh = oh;
    }
    if (i < n32) {                                      /* an odd last group */
        __m128i ml = LD(q + 32u * i), mh = LD(q + 32u * i + 16u), nl = LD(k + 4u * i + 4u), nh = LD(k + 4u * i + 6u);
        CLNH_GROUP(ml, mh, kl, kh, nl, nh, a0, a1, b0, b1);
    }
    _mm_storeu_si128((__m128i *)(void *)s, _mm_xor_si128(_mm_xor_si128(a0, a1), _mm_xor_si128(c0, c1)));
    _mm_storeu_si128((__m128i *)(void *)(s + 2), _mm_xor_si128(_mm_xor_si128(b0, b1), _mm_xor_si128(d0, d1)));
}
#  undef LD
#  define CLMUL_FEATURE TOKS_X86_PCLMUL
#endif

/* level 1 over n32 groups at q: the cpu's carry-less multiply, else the portable one (hw = 0: tests compare both) */
static void clnh(int hw, const uint64_t *k, const uint8_t *q, uint64_t n32, uint64_t s[4])
{
    if (hw != 0) {
        clnh_hw(k, q, n32, s);
    } else {
        clnh_c(k, q, n32, s);
    }
}

void toks_memo_check_with(const toks_ctx *ctx, int hw, const uint8_t *g, uint64_t n, uint64_t c[2])
{
    const uint64_t *k = ctx->memo_key, *w = ctx->memo_rpow;
    const u128 rp[5] = { ((u128)w[1] << 64) | w[0], ((u128)w[3] << 64) | w[2], ((u128)w[5] << 64) | w[4],
                         ((u128)w[7] << 64) | w[6], ((u128)w[9] << 64) | w[8] };
    u128 h = 0u;
    for (uint64_t at = 0; at < n; at += TOKS_MEMO_BLOCK) {     /* bound: n / 4 KiB + 1 blocks */
        uint64_t b = n - at < TOKS_MEMO_BLOCK ? n - at : TOKS_MEMO_BLOCK, s[4] = { 0u, 0u, 0u, 0u };
        clnh(hw, k, g + at, b / 32u, s);
        if (b % 32u != 0u) {                            /* the segment's last bytes, zero-padded to a group */
            uint8_t t[32] = { 0 };
            memcpy(t, g + at + (b & ~31ull), (size_t)(b % 32u));
            clnh(hw, k + 4u * (b / 32u), t, 1u, s);
        }
        h = at + TOKS_MEMO_BLOCK < n ? p127_block(h, rp, s) : p127_last(h, rp, s, n);
    }
    c[0] = (uint64_t)h, c[1] = (uint64_t)(h >> 64);
}

void toks_memo_check(const toks_ctx *ctx, const uint8_t *g, uint64_t n, uint64_t c[2])
{
    toks_memo_check_with(ctx, TOKS_CPU_HAS(ctx->cpu_features, CLMUL_FEATURE), g, n, c);
}

/* the context's key (load.c, once): the os's randomness (toks_plat_entropy), secret, so no one can choose two segments
 * that share a check; 0, or -1 when the os gives none: the key is zeroed and the context has no memo (api.c scr_memo
 * lays out none; run and run_seg take none even from a scratch a keyed context of the same tokenizer laid out) */
int toks_memo_keygen(toks_ctx *c)
{
    if (toks_plat_entropy(c->memo_key, sizeof c->memo_key) != 0) {
        memset(c->memo_key, 0, sizeof c->memo_key), memset(c->memo_rpow, 0, sizeof c->memo_rpow);   /* nothing partial */
        return -1;
    }
    toks_memo_rpow(c);
    return 0;
}

/* the polynomial's key r < 2^126 (the key's last two words) and its powers r^1..r^5 into ctx->memo_rpow (load) */
void toks_memo_rpow(toks_ctx *c)
{
    const uint64_t *k = c->memo_key;
    u128 r = (((u128)k[TOKS_MEMO_KEY_W - 1u] << 64) | k[TOKS_MEMO_KEY_W - 2u]) >> 2, x = r;
    for (uint32_t i = 0; i < 5u; i++) {                 /* bound: 5 powers */
        c->memo_rpow[2u * i] = (uint64_t)x, c->memo_rpow[2u * i + 1u] = (uint64_t)(x >> 64);
        x = p127_mul(x, r);
    }
}
