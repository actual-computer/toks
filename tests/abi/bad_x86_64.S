/* tests/abi/bad_x86_64.S: fixtures for tests/abi/teeth.c, uint64_t f(uint64_t, uint64_t) returning 7, each
 * breaking one rule of the abi (or none). One source for elf, mach-o and coff through src/asm/x86_64/asm.h. */
#include "asm.h"

FUNC abibad_none
        LEAF
        mov     eax, 7
        ret
ENDFUNC abibad_none

FUNC abibad_rbx
        LEAF
        xor     ebx, ebx
        mov     eax, 7
        ret
ENDFUNC abibad_rbx

FUNC abibad_r15
        LEAF
        xor     r15d, r15d
        mov     eax, 7
        ret
ENDFUNC abibad_r15

FUNC abibad_rdi
        LEAF
        xor     edi, edi
        mov     eax, 7
        ret
ENDFUNC abibad_rdi

FUNC abibad_rsi
        LEAF
        xor     esi, esi
        mov     eax, 7
        ret
ENDFUNC abibad_rsi

/* a vex write of ymm6 with nothing saved: what a kernel using ymm6-15 under PROLOGUE n, 0 would do */
FUNC abibad_ymm6
        LEAF
        vpcmpeqb ymm6, ymm6, ymm6
        vzeroupper
        mov     eax, 7
        ret
ENDFUNC abibad_ymm6

/* only the high qword of xmm7 changes */
FUNC abibad_xmm7_hi
        LEAF
        mov     rax, -1
        movq    xmm0, rax
        movlhps xmm7, xmm0
        mov     eax, 7
        ret
ENDFUNC abibad_xmm7_hi

/* xmm8 saved and restored with 64-bit moves: its high qword is lost */
FUNC abibad_save64
        PROLOGUE 0, 0, 16
        movq    qword ptr [rsp], xmm8
        pcmpeqd xmm8, xmm8
        movq    xmm8, qword ptr [rsp]
        mov     eax, 7
        EPILOGUE 0, 0, 16
ENDFUNC abibad_save64

/* an evex write of zmm15 (avx-512 only) */
FUNC abibad_zmm15
        LEAF
        vpternlogd zmm15, zmm15, zmm15, 0xff
        vzeroupper
        mov     eax, 7
        ret
ENDFUNC abibad_zmm15

/* PROLOGUE 6, 10: every callee-saved register clobbered (ymm6-15 whole) and restored by the macros */
FUNC abibad_good_frame
        PROLOGUE 6, 10, 16
        mov     rbx, -1
        mov     rbp, -1
        mov     r12, -1
        mov     r13, -1
        mov     r14, -1
        mov     r15, -1
        mov     rdi, -1
        mov     rsi, -1
  .irp n, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        vpcmpeqb ymm\n, ymm\n, ymm\n
  .endr
        vzeroupper
        mov     eax, 7
        EPILOGUE 6, 10, 16
ENDFUNC abibad_good_frame
