/* rationale: docs/notes/c-core.md §cpu.c.1 */
#include "cpu.h"

#if defined(TOKS_ARCH_X86_64)

static void cpuid2(uint32_t leaf, uint32_t sub, uint32_t r[4])
{
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(sub));
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
}

static uint64_t xgetbv0(void)
{
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0u));
    return ((uint64_t)hi << 32) | lo;
}

uint64_t toks_cpu_features(void)
{
    uint64_t f = 0;
    uint32_t r[4];
    cpuid2(0u, 0u, r);
    uint32_t max_leaf = r[0];
    cpuid2(0x80000000u, 0u, r);
    uint32_t max_ext = r[0];
    if (max_leaf < 1u) return 0;

    cpuid2(1u, 0u, r);
    if (r[2] & (1u << 1))  f |= TOKS_X86_PCLMUL;           /* xmm state: always enabled on x86-64 */
    if (r[2] & (1u << 20)) f |= TOKS_X86_SSE42;
    if (r[2] & (1u << 23)) f |= TOKS_X86_POPCNT;
    int osxsave = (r[2] & (1u << 27)) != 0;
    int avx = (r[2] & (1u << 28)) != 0;
    uint64_t xcr0 = osxsave ? xgetbv0() : 0;
    int ymm_ok = avx && (xcr0 & 0x6u) == 0x6u;              /* sse + avx state enabled */
    int zmm_ok = ymm_ok && (xcr0 & 0xE0u) == 0xE0u;        /* opmask + zmm_hi256 + hi16_zmm */

    if (max_leaf >= 7u) {
        cpuid2(7u, 0u, r);
        if (r[1] & (1u << 3)) f |= TOKS_X86_BMI1;
        if (r[1] & (1u << 8)) f |= TOKS_X86_BMI2;
        if (ymm_ok && (r[1] & (1u << 5))) f |= TOKS_X86_AVX2;
        if (zmm_ok) {
            if (r[1] & (1u << 16)) f |= TOKS_X86_AVX512F;
            if (r[1] & (1u << 30)) f |= TOKS_X86_AVX512BW;
            if (r[1] & (1u << 31)) f |= TOKS_X86_AVX512VL;
            if (r[2] & (1u << 1))  f |= TOKS_X86_AVX512VBMI;
            if (r[2] & (1u << 6))  f |= TOKS_X86_AVX512VBMI2;
        }
    }
    if (max_ext >= 0x80000001u) {
        cpuid2(0x80000001u, 0u, r);
        if (r[2] & (1u << 5)) f |= TOKS_X86_LZCNT;
    }
    return f;
}

#elif defined(TOKS_ARCH_ARM64) && defined(_WIN32)

#include <windows.h>
#ifndef PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE
#  define PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE 31
#endif
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#  define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif
#ifndef PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE
#  define PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE 34
#endif
#ifndef PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE
#  define PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE 30
#endif

uint64_t toks_cpu_features(void)
{
    uint64_t f = TOKS_ARM64_NEON;                /* required by the windows arm64 abi */
    if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE)) f |= TOKS_ARM64_CRC32;
    if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE))   f |= TOKS_ARM64_DOTPROD;
    if (IsProcessorFeaturePresent(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE)) f |= TOKS_ARM64_LSE;
    if (IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE))  f |= TOKS_ARM64_PMULL;   /* aes + pmull */
    return f;
}

#elif defined(TOKS_ARCH_ARM64) && defined(__APPLE__)

#include <stddef.h>
#include <sys/sysctl.h>

static int has(const char *name)
{
    int v = 0;
    size_t len = sizeof v;
    return sysctlbyname(name, &v, &len, NULL, 0) == 0 && v != 0;
}

uint64_t toks_cpu_features(void)
{
    uint64_t f = TOKS_ARM64_NEON;                /* every apple arm64 cpu */
    if (has("hw.optional.armv8_crc32"))      f |= TOKS_ARM64_CRC32;
    if (has("hw.optional.arm.FEAT_DotProd")) f |= TOKS_ARM64_DOTPROD;
    if (has("hw.optional.arm.FEAT_LSE"))     f |= TOKS_ARM64_LSE;
    if (has("hw.optional.arm.FEAT_PMULL"))   f |= TOKS_ARM64_PMULL;
    return f;
}

#elif defined(TOKS_ARCH_ARM64)  /* linux */

#include <sys/auxv.h>
#ifndef HWCAP_ASIMD
#  define HWCAP_ASIMD   (1ul << 1)
#endif
#ifndef HWCAP_PMULL
#  define HWCAP_PMULL   (1ul << 4)
#endif
#ifndef HWCAP_CRC32
#  define HWCAP_CRC32   (1ul << 7)
#endif
#ifndef HWCAP_ATOMICS
#  define HWCAP_ATOMICS (1ul << 8)
#endif
#ifndef HWCAP_ASIMDDP
#  define HWCAP_ASIMDDP (1ul << 20)
#endif
#ifndef HWCAP_SVE
#  define HWCAP_SVE     (1ul << 22)
#endif

uint64_t toks_cpu_features(void)
{
    uint64_t f = 0;
    unsigned long h = getauxval(AT_HWCAP);
    if (h & HWCAP_ASIMD)   f |= TOKS_ARM64_NEON;
    if (h & HWCAP_PMULL)   f |= TOKS_ARM64_PMULL;
    if (h & HWCAP_CRC32)   f |= TOKS_ARM64_CRC32;
    if (h & HWCAP_ASIMDDP) f |= TOKS_ARM64_DOTPROD;
    if (h & HWCAP_ATOMICS) f |= TOKS_ARM64_LSE;
    if (h & HWCAP_SVE)     f |= TOKS_ARM64_SVE;
    return f;
}

#endif
