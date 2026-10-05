/* rationale: docs/notes/c-core.md §cpu.h.1 */
#ifndef TOKS_CPU_H
#define TOKS_CPU_H

#include <stdint.h>

/* x86-64 bits */
#define TOKS_X86_SSE42        (1ull << 0)
#define TOKS_X86_POPCNT       (1ull << 1)
#define TOKS_X86_AVX2         (1ull << 2)
#define TOKS_X86_BMI1         (1ull << 3)
#define TOKS_X86_BMI2         (1ull << 4)
#define TOKS_X86_LZCNT        (1ull << 5)
#define TOKS_X86_AVX512F      (1ull << 6)
#define TOKS_X86_AVX512BW     (1ull << 7)
#define TOKS_X86_AVX512VL     (1ull << 8)
#define TOKS_X86_AVX512VBMI   (1ull << 9)
#define TOKS_X86_AVX512VBMI2  (1ull << 10)

/* arm64 bits */
#define TOKS_ARM64_NEON       (1ull << 16)
#define TOKS_ARM64_CRC32      (1ull << 17)
#define TOKS_ARM64_DOTPROD    (1ull << 18)
#define TOKS_ARM64_LSE        (1ull << 19)
#define TOKS_ARM64_SVE        (1ull << 20)

#if defined(__x86_64__) || defined(_M_X64)
#  define TOKS_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define TOKS_ARCH_ARM64 1
#else
#  error "toks: unsupported architecture"
#endif

#define TOKS_FEAT_AVX2_TIER   (TOKS_X86_AVX2 | TOKS_X86_BMI1 | TOKS_X86_BMI2 | TOKS_X86_LZCNT | \
                               TOKS_X86_POPCNT | TOKS_X86_SSE42)
#define TOKS_FEAT_AVX512_TIER (TOKS_FEAT_AVX2_TIER | TOKS_X86_AVX512F | TOKS_X86_AVX512BW | \
                               TOKS_X86_AVX512VL | TOKS_X86_AVX512VBMI)
#define TOKS_FEAT_NEON_TIER   (TOKS_ARM64_NEON | TOKS_ARM64_CRC32)

#define TOKS_CPU_HAS(f, set)  (((f) & (set)) == (set))

/* the features of the cpu this runs on (cpuid / os state; no cache, no global written). */
uint64_t toks_cpu_features(void);

#endif /* TOKS_CPU_H */
