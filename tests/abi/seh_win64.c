/* tests/abi/seh_win64.c (windows x64 only): the SEH unwind data the asm.h macros emit (.seh_pushreg,
 * .seh_stackalloc, .seh_savexmm). Each kernel is called through toks_abicheck_call (canaries in every
 * callee-saved register, xmm6-15 whole) and made to fault after its prologue; a vectored handler unwinds the
 * kernel's frame with the os unwinder (RtlLookupFunctionEntry + RtlVirtualUnwind) and resumes the caller at its
 * return address as if the kernel had returned 0xFA17. The harness's report then names every register, and
 * rsp, that the unwind codes restore wrongly.
 *   toks_k0_frame_x86 (selftest.S)   PROLOGUE 6, 10, 16: clobbers every callee-saved register, faults on its
 *                                     first byte (all ten .seh_savexmm slots in play)
 *   K3 avx2 / avx512 (-DSEH_K3)       PROLOGUE 6, 0, LOCALS: fault on the first block load (text in a
 *                                     no-access page)
 * Built by hand after tools\win\build.cmd, from the source root:
 *   clang -std=c17 -O2 -Wall -Wextra -Werror -Iinclude -Isrc\core -Isrc\platform -Itests\common [-DSEH_K3] ^
 *     -o seh_win64.exe tests\abi\seh_win64.c tests\common\abicheck_x86_64.S build\windows-x86_64\libtoks.lib
 */
#include <windows.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "classes.h"
#include "kernels.h"
#include "layout.h"
#include "cpu.h"
#include "guard.h"

uint64_t toks_k0_frame_x86(const uint8_t *p, uint64_t n);

static uint8_t *page;           /* PAGE_NOACCESS */
static size_t page_len;
static volatile LONG faults;
static volatile DWORD64 fault_fn, fault_off;

static LONG CALLBACK veh(EXCEPTION_POINTERS *ep)
{
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) { return EXCEPTION_CONTINUE_SEARCH; }
    uintptr_t addr = (uintptr_t)er->ExceptionInformation[1];
    if (addr < (uintptr_t)page || addr >= (uintptr_t)page + page_len) { return EXCEPTION_CONTINUE_SEARCH; }
    CONTEXT *c = ep->ContextRecord;
    DWORD64 base = 0;
    PRUNTIME_FUNCTION rf = RtlLookupFunctionEntry(c->Rip, &base, NULL);
    if (rf == NULL) { return EXCEPTION_CONTINUE_SEARCH; }
    fault_fn = base + rf->BeginAddress;
    fault_off = c->Rip - fault_fn;
    PVOID hd = NULL;
    DWORD64 ef = 0;
    RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, c->Rip, rf, c, &hd, &ef, NULL);   /* c: the caller's state */
    c->Rax = 0xFA17;
    InterlockedIncrement(&faults);
    return EXCEPTION_CONTINUE_EXECUTION;
}

static int run(const char *name, const void *fn, uint64_t a0, uint64_t a1)
{
    uint64_t rep = 0;
    LONG f0 = faults;
    uint64_t r = toks_abicheck_call(fn, a0, a1, &rep);
    int ok = faults == f0 + 1 && r == 0xFA17 && rep == 0 && fault_fn == (DWORD64)(uintptr_t)fn;
    printf("  %-26s fault at +%#" PRIx64 " in %s, unwound: rax %#" PRIx64 ", report %#" PRIx64 " %s\n", name,
           (uint64_t)fault_off, fault_fn == (DWORD64)(uintptr_t)fn ? "the kernel" : "ANOTHER FUNCTION", r, rep,
           ok ? "ok" : "FAIL");
    return !ok;
}

#if defined(SEH_K3)
static int run_k3(const char *name, const void *fn)
{
    static uint8_t cls_buf[128 + 2 * 0x1100 + 0x1100 * 256];
    static uint32_t ends[4096];
    toks_class_tables ct;
    if (toks_classes_build(0u, cls_buf, sizeof cls_buf, &ct) <= 0) { fprintf(stderr, "classes\n"); return 1; }
    toks_tables t;
    memset(&t, 0, sizeof t);
    t.magic = TOKS_TABLES_MAGIC;
    t.version = TOKS_TABLES_VERSION;
    t.tmpl = TOKS_TMPL_CL100K;
    t.tmpl_params = TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL;
    t.cls_ascii = ct.ascii;
    t.cls_stage1 = ct.stage1;
    t.cls_stage2 = ct.stage2;
    t.cls_nblocks = ct.n_blocks;
    toks_k3_args a;
    memset(&a, 0, sizeof a);
    a.text = page; a.len = page_len; a.pos = 0; a.ends = ends; a.cap = 4096;
    return run(name, fn, (uint64_t)(uintptr_t)&t, (uint64_t)(uintptr_t)&a);
}
#endif

int main(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    page_len = 2 * (size_t)si.dwPageSize;
    page = (uint8_t *)VirtualAlloc(NULL, page_len, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    if (page == NULL || AddVectoredExceptionHandler(1, veh) == NULL) { fprintf(stderr, "setup failed\n"); return 2; }
    int bad = run("toks_k0_frame_x86", (const void *)toks_k0_frame_x86, (uint64_t)(uintptr_t)page, 1);
#if defined(SEH_K3)
    uint64_t f = toks_cpu_features();
    if (TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER)) {
        bad += run_k3("toks_k3_scan_cl100k_avx2", (const void *)toks_k3_scan_cl100k_avx2);
    }
    if (TOKS_CPU_HAS(f, TOKS_FEAT_AVX512_TIER)) {
        bad += run_k3("toks_k3_scan_cl100k_avx512", (const void *)toks_k3_scan_cl100k_avx512);
    }
#endif
    printf("seh_win64: %ld faults unwound, %d failures\n", (long)faults, bad);
    return bad != 0;
}
