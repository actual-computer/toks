/*
 * tests/k3/abi_k3.c: every K3 call of tests/c/test_k3.c or tests/k3/check.c through the callee-saved canary
 * harness (tests/common/abicheck_<isa>.S), so the hand checks and the hf differential double as an abi check
 * of an asm tier on the platform's own abi (sysv: rbx rbp r12-r15; win64: + rdi rsi xmm6-15, all 128 bits).
 * Built by hand next to the test it wraps, e.g. for the avx2 tier:
 *
 *   clang ... -DK3_SCAN=toks_k3_scan_abi -DK3_TIER=toks_k3_scan_cl100k_avx2 -o test_k3_abi \
 *     tests/c/test_k3.c tests/k3/abi_k3.c tests/common/guard.c tests/common/abicheck_x86_64.S <library>
 *
 * The first clobber stops the run (exit 3) with the harness's report decoded; a clean run prints the call count.
 */
#include "../../src/core/layout.h"
#include "../common/guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef K3_TIER
#error "build with -DK3_TIER=<the kernel symbol under test>"
#endif

uint64_t K3_TIER(const toks_tables *t, toks_k3_args *a);
uint64_t toks_k3_scan_abi(const toks_tables *t, toks_k3_args *a);

static uint64_t calls;

static void summary(void)
{
    printf("abicheck: %" PRIu64 " calls through toks_abicheck_call, 0 clobbered\n", calls);
}

uint64_t toks_k3_scan_abi(const toks_tables *t, toks_k3_args *a)
{
    static const char *const names[32] = {     /* x86-64 / arm64 (the bit layouts of the two harnesses) */
        "rbx/x19", "rbp/x20", "r12/x21", "r13/x22", "r14/x23", "r15/x24", "rdi/x25", "rsi/x26",
        "x27", "x28", "", "", "", "", "", "",
        "xmm6/d8", "xmm7/d9", "xmm8/d10", "xmm9/d11", "xmm10/d12", "xmm11/d13", "xmm12/d14", "xmm13/d15",
        "xmm14", "xmm15", "", "", "", "", "x18", "rsp/sp",
    };
    uint64_t report = 0;
    if (calls++ == 0) { atexit(summary); }
    uint64_t n = toks_abicheck_call((const void *)K3_TIER, (uint64_t)(uintptr_t)t, (uint64_t)(uintptr_t)a, &report);
    if (report != 0) {
        fprintf(stderr, "ABI: call %" PRIu64 " (pos %" PRIu64 " len %" PRIu64 " cap %" PRIu64 ") clobbered:", calls,
                a->pos, a->len, a->cap);
        for (int b = 0; b < 32; b++) {
            if ((report >> b) & 1u) { fprintf(stderr, " %s(bit %d)", names[b], b); }
        }
        fprintf(stderr, "; report %#" PRIx64 "\n", report);
        fflush(stdout);
        fflush(stderr);
        _Exit(3);
    }
    return n;
}
