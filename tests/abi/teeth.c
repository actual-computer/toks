/* tests/abi/teeth.c: the canary harness's teeth (tests/common/abicheck_x86_64.S). Each fixture of
 * tests/abi/bad_x86_64.S breaks one rule of the abi (or none); the harness's report must name exactly the
 * registers that rule protects on this platform: sysv rbx rbp r12-r15 (rsp); win64 also rdi, rsi and xmm6-15,
 * all 128 bits. Not in `make test` (x86-64 only); built by hand from the source root:
 *
 *   clang -std=c17 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform -Isrc/asm/x86_64 \
 *     -Itests/common -o teeth tests/abi/teeth.c tests/abi/bad_x86_64.S tests/common/abicheck_x86_64.S \
 *     src/platform/cpu.c
 *
 * (macos x86-64: add -arch x86_64 and run under rosetta.) Exit 0 iff every report is exact.
 */
#include <inttypes.h>
#include <stdio.h>

#include "cpu.h"
#include "guard.h"

uint64_t abibad_none(uint64_t, uint64_t);
uint64_t abibad_rbx(uint64_t, uint64_t);
uint64_t abibad_r15(uint64_t, uint64_t);
uint64_t abibad_rdi(uint64_t, uint64_t);
uint64_t abibad_rsi(uint64_t, uint64_t);
uint64_t abibad_ymm6(uint64_t, uint64_t);
uint64_t abibad_xmm7_hi(uint64_t, uint64_t);
uint64_t abibad_save64(uint64_t, uint64_t);
uint64_t abibad_zmm15(uint64_t, uint64_t);
uint64_t abibad_good_frame(uint64_t, uint64_t);

#if defined(_WIN32)
#  define W(b) (1ull << (b))   /* callee-saved on win64 only */
#else
#  define W(b) 0ull
#endif

int main(void)
{
    uint64_t f = toks_cpu_features();
    int avx2 = (f & TOKS_X86_AVX2) != 0, avx512 = (f & TOKS_X86_AVX512F) != 0;
#if defined(__APPLE__)
    avx2 = 1;   /* rosetta 2 runs avx2 code while its cpuid reports no avx */
#endif
    struct { const char *name; uint64_t (*fn)(uint64_t, uint64_t); uint64_t want; int run; } T[] = {
        { "none",       abibad_none,       0, 1 },
        { "rbx",        abibad_rbx,        1ull << 0, 1 },
        { "r15",        abibad_r15,        1ull << 5, 1 },
        { "rdi",        abibad_rdi,        W(6), 1 },
        { "rsi",        abibad_rsi,        W(7), 1 },
        { "ymm6",       abibad_ymm6,       W(16), avx2 },
        { "xmm7 hi",    abibad_xmm7_hi,    W(17), 1 },
        { "save64 x8",  abibad_save64,     W(18), 1 },
        { "zmm15",      abibad_zmm15,      W(25), avx512 },
        { "good frame", abibad_good_frame, 0, avx2 },
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
        if (!T[i].run) { printf("  %-10s skipped (cpu)\n", T[i].name); continue; }
        uint64_t rep = 0;
        uint64_t r = toks_abicheck_call((const void *)T[i].fn, 1, 2, &rep);
        int ok = rep == T[i].want && r == 7;
        printf("  %-10s report %#010" PRIx64 " want %#010" PRIx64 " %s\n", T[i].name, rep, T[i].want, ok ? "ok" : "FAIL");
        bad += !ok;
    }
    printf("teeth: %d failures\n", bad);
    return bad != 0;
}
