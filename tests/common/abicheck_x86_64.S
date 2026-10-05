/* tests/common/abicheck_x86_64.S
 * uint64_t toks_abicheck_call(const void *fn, uint64_t a0, uint64_t a1, uint64_t *report)
 * Calls fn(a0, a1) with canaries in every callee-saved register and reports, in *report, a bit per register
 * the callee failed to preserve: rbx 0, rbp 1, r12 2, r13 3, r14 4, r15 5, and on win64 rdi 6, rsi 7 and
 * xmm6..xmm15 -> bits 16..25 (all 128 bits: a canary in each qword, so a 64-bit save/restore is caught);
 * bit 31 for rsp. Returns fn's rax. Test-only (indirect call).
 * Frame: win64 needs 32 bytes of home space for the callee; the trampoline's own frame exceeds the 256-byte
 * kernel limit on win64, so it is laid out by hand here rather than with PROLOGUE.
 */
#include "../../src/asm/x86_64/asm.h"

#define CANARY(k) (0x0c0de00000000000 + (k))

.macro CHK reg, k, bit
        mov     r11, CANARY(\k)
        cmp     \reg, r11
        setne   cl
        movzx   ecx, cl
        shl     rcx, \bit
        or      r10, rcx
.endm

.macro CHKX n, bit
        movq    rdx, xmm\n
        mov     r11, CANARY(0x100 + \n)
        xor     rdx, r11
        pshufd  xmm0, xmm\n, 0xEE                /* the high qword */
        movq    r9, xmm0
        mov     r11, CANARY(0x200 + \n)
        xor     r9, r11
        or      rdx, r9
        setnz   cl
        movzx   ecx, cl
        shl     rcx, \bit
        or      r10, rcx
.endm

#if defined(TOKS_WIN64)
#  define FRAME_ALLOC 232      /* 32 home + 8 fn + 8 report + 8 rsp + 8 pad + 160 xmm + 8 align */
#  define XMM_BASE    64
#else
#  define FRAME_ALLOC 56       /* 8 fn + 8 report + 8 rsp + pad to keep rsp 16-aligned at the call */
#endif
#define L_FN      32
#define L_REPORT  40
#define L_RSP     48

FUNC toks_abicheck_call
        TOKS_PUSH rbx
        TOKS_PUSH rbp
        TOKS_PUSH r12
        TOKS_PUSH r13
        TOKS_PUSH r14
        TOKS_PUSH r15
#if defined(TOKS_WIN64)
        TOKS_PUSH rdi
        TOKS_PUSH rsi
#endif
        sub     rsp, FRAME_ALLOC
#if defined(TOKS_COFF)
        .seh_stackalloc FRAME_ALLOC
        movaps  [rsp + XMM_BASE +   0], xmm6
        .seh_savexmm xmm6, XMM_BASE + 0
        movaps  [rsp + XMM_BASE +  16], xmm7
        .seh_savexmm xmm7, XMM_BASE + 16
        movaps  [rsp + XMM_BASE +  32], xmm8
        .seh_savexmm xmm8, XMM_BASE + 32
        movaps  [rsp + XMM_BASE +  48], xmm9
        .seh_savexmm xmm9, XMM_BASE + 48
        movaps  [rsp + XMM_BASE +  64], xmm10
        .seh_savexmm xmm10, XMM_BASE + 64
        movaps  [rsp + XMM_BASE +  80], xmm11
        .seh_savexmm xmm11, XMM_BASE + 80
        movaps  [rsp + XMM_BASE +  96], xmm12
        .seh_savexmm xmm12, XMM_BASE + 96
        movaps  [rsp + XMM_BASE + 112], xmm13
        .seh_savexmm xmm13, XMM_BASE + 112
        movaps  [rsp + XMM_BASE + 128], xmm14
        .seh_savexmm xmm14, XMM_BASE + 128
        movaps  [rsp + XMM_BASE + 144], xmm15
        .seh_savexmm xmm15, XMM_BASE + 144
        .seh_endprologue
#else
        .cfi_adjust_cfa_offset FRAME_ALLOC
#endif
        mov     [rsp + L_FN], ARG0
        mov     [rsp + L_REPORT], ARG3
        mov     [rsp + L_RSP], rsp
        mov     ARG0, ARG1
        mov     ARG1, ARG2
        mov     rbx, CANARY(0)
        mov     rbp, CANARY(1)
        mov     r12, CANARY(2)
        mov     r13, CANARY(3)
        mov     r14, CANARY(4)
        mov     r15, CANARY(5)
#if defined(TOKS_WIN64)
        mov     rdi, CANARY(6)
        mov     rsi, CANARY(7)
  .irp n, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        mov     rax, CANARY(0x100 + \n)
        movq    xmm\n, rax
        mov     rax, CANARY(0x200 + \n)
        movq    xmm0, rax
        punpcklqdq xmm\n, xmm0                  /* high qword: its own canary */
  .endr
#endif
        call    qword ptr [rsp + L_FN]
        xor     r10d, r10d
        CHK     rbx, 0, 0
        CHK     rbp, 1, 1
        CHK     r12, 2, 2
        CHK     r13, 3, 3
        CHK     r14, 4, 4
        CHK     r15, 5, 5
#if defined(TOKS_WIN64)
        CHK     rdi, 6, 6
        CHK     rsi, 7, 7
        CHKX    6, 16
        CHKX    7, 17
        CHKX    8, 18
        CHKX    9, 19
        CHKX    10, 20
        CHKX    11, 21
        CHKX    12, 22
        CHKX    13, 23
        CHKX    14, 24
        CHKX    15, 25
#endif
        mov     r11, [rsp + L_RSP]
        cmp     rsp, r11
        setne   cl
        movzx   ecx, cl
        shl     rcx, 31
        or      r10, rcx
        mov     r11, [rsp + L_REPORT]
        mov     [r11], r10
#if defined(TOKS_WIN64)
        movaps  xmm6,  [rsp + XMM_BASE +   0]
        movaps  xmm7,  [rsp + XMM_BASE +  16]
        movaps  xmm8,  [rsp + XMM_BASE +  32]
        movaps  xmm9,  [rsp + XMM_BASE +  48]
        movaps  xmm10, [rsp + XMM_BASE +  64]
        movaps  xmm11, [rsp + XMM_BASE +  80]
        movaps  xmm12, [rsp + XMM_BASE +  96]
        movaps  xmm13, [rsp + XMM_BASE + 112]
        movaps  xmm14, [rsp + XMM_BASE + 128]
        movaps  xmm15, [rsp + XMM_BASE + 144]
#endif
        add     rsp, FRAME_ALLOC
#if defined(TOKS_WIN64)
        pop     rsi
        pop     rdi
#endif
        pop     r15
        pop     r14
        pop     r13
        pop     r12
        pop     rbp
        pop     rbx
        ret
ENDFUNC toks_abicheck_call
