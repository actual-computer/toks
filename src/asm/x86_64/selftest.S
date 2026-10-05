/* selftest kernels (tiers avx2, avx512): prove the macro layer, the build and the geometry / abi harness.
 *   uint64_t toks_k0_sum_avx2(const uint8_t *p, uint64_t n)     sum of the bytes mod 2^64; a true leaf
 *   uint64_t toks_k0_sum_avx512(const uint8_t *p, uint64_t n)   the same, avx-512 (f, bw, vl)
 *   uint64_t toks_k0_frame_x86(const uint8_t *p, uint64_t n)    the same sum through a full frame that
 *                                                               clobbers every callee-saved register
 */
#include "asm.h"

FUNC toks_k0_sum_avx2
        LEAF
        mov     r8, ARG0
        mov     r9, ARG1
        xor     eax, eax
        mov     r10, r9
        shr     r10, 6                          /* full 64-byte blocks */
        jz      2f
        vpxor   ymm0, ymm0, ymm0
        vpxor   ymm1, ymm1, ymm1
1:      vpsadbw ymm2, ymm0, ymmword ptr [r8]
        vpsadbw ymm3, ymm0, ymmword ptr [r8 + 32]
        vpaddq  ymm1, ymm1, ymm2
        vpaddq  ymm1, ymm1, ymm3
        add     r8, 64
        dec     r10
        jnz     1b
        vextracti128 xmm2, ymm1, 1
        vpaddq  xmm1, xmm1, xmm2
        vpshufd xmm2, xmm1, 0x4e
        vpaddq  xmm1, xmm1, xmm2
        vmovq   rax, xmm1
        vzeroupper
2:      and     r9, 63                          /* scalar tail, inside [p, p + n) */
        jz      4f
3:      movzx   ecx, byte ptr [r8]
        add     rax, rcx
        inc     r8
        dec     r9
        jnz     3b
4:      ret
ENDFUNC toks_k0_sum_avx2

FUNC toks_k0_sum_avx512
        LEAF
        mov     r8, ARG0
        mov     r9, ARG1
        xor     eax, eax
        mov     r10, r9
        shr     r10, 6
        jz      2f
        vpxord  zmm16, zmm16, zmm16
        vpxord  zmm17, zmm17, zmm17
1:      vpsadbw zmm18, zmm16, zmmword ptr [r8]
        vpaddq  zmm17, zmm17, zmm18
        add     r8, 64
        dec     r10
        jnz     1b
        vextracti64x4 ymm18, zmm17, 1
        vpaddq  ymm17, ymm17, ymm18
        vextracti32x4 xmm18, ymm17, 1
        vpaddq  xmm17, xmm17, xmm18
        vpshufd xmm18, xmm17, 0x4e
        vpaddq  xmm17, xmm17, xmm18
        vmovq   rax, xmm17
        vzeroupper
2:      and     r9, 63
        jz      4f
3:      movzx   ecx, byte ptr [r8]
        add     rax, rcx
        inc     r8
        dec     r9
        jnz     3b
4:      ret
ENDFUNC toks_k0_sum_avx512

FUNC toks_k0_frame_x86
        PROLOGUE 6, 10, 16
        mov     r12, ARG0
        mov     r13, ARG1
        mov     qword ptr [rsp], r13            /* a local */
        mov     rbx, -1
        mov     rbp, -1
        mov     r14, -1
        mov     r15, -1
        mov     rdi, -1
        mov     rsi, -1
        pcmpeqd xmm6, xmm6
        pcmpeqd xmm7, xmm7
        pcmpeqd xmm8, xmm8
        pcmpeqd xmm9, xmm9
        pcmpeqd xmm10, xmm10
        pcmpeqd xmm11, xmm11
        pcmpeqd xmm12, xmm12
        pcmpeqd xmm13, xmm13
        pcmpeqd xmm14, xmm14
        pcmpeqd xmm15, xmm15
        xor     ebx, ebx
        mov     r13, qword ptr [rsp]
        test    r13, r13
        jz      2f
1:      movzx   ebp, byte ptr [r12]
        add     rbx, rbp
        inc     r12
        dec     r13
        jnz     1b
2:      mov     rax, rbx
        EPILOGUE 6, 10, 16
ENDFUNC toks_k0_frame_x86
