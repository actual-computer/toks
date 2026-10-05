/* selftest kernels (tier neon): prove the macro layer, the build and the geometry / abi harness.
 *   uint64_t toks_k0_sum_neon(const uint8_t *p, uint64_t n)    sum of the bytes mod 2^64; a true leaf
 *   uint64_t toks_k0_frame_neon(const uint8_t *p, uint64_t n)  the same sum through a full frame that
 *                                                              clobbers every callee-saved register
 */
#include "asm.h"

FUNC toks_k0_sum_neon
        LEAF
        mov     x2, #0
        lsr     x3, x1, #6                      /* full 64-byte blocks */
        cbz     x3, 2f
        movi    v4.2d, #0
1:      ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64
        uaddlp  v0.8h, v0.16b
        uadalp  v0.8h, v1.16b
        uadalp  v0.8h, v2.16b
        uadalp  v0.8h, v3.16b                   /* each u16 lane <= 8 * 255 */
        uaddlp  v0.4s, v0.8h
        uadalp  v4.2d, v0.4s
        subs    x3, x3, #1
        b.ne    1b
        addp    d4, v4.2d
        fmov    x2, d4
2:      ands    x3, x1, #63                     /* scalar tail, inside [p, p + n) */
        b.eq    4f
3:      ldrb    w4, [x0], #1
        add     x2, x2, x4
        subs    x3, x3, #1
        b.ne    3b
4:      mov     x0, x2
        ret
ENDFUNC toks_k0_sum_neon

FUNC toks_k0_frame_neon
        PROLOGUE 5, 4, 32
        str     x0, [sp, #LOCALS]
        str     x1, [sp, #LOCALS + 8]
        mov     x19, #-1
        mov     x20, #-1
        mov     x21, #-1
        mov     x22, #-1
        mov     x23, #-1
        mov     x24, #-1
        mov     x25, #-1
        mov     x26, #-1
        mov     x27, #-1
        mov     x28, #-1
        movi    v8.2d, #0xffffffffffffffff
        movi    v9.2d, #0xffffffffffffffff
        movi    v10.2d, #0xffffffffffffffff
        movi    v11.2d, #0xffffffffffffffff
        movi    v12.2d, #0xffffffffffffffff
        movi    v13.2d, #0xffffffffffffffff
        movi    v14.2d, #0xffffffffffffffff
        movi    v15.2d, #0xffffffffffffffff
        ldr     x19, [sp, #LOCALS]              /* walk with callee-saved registers */
        ldr     x20, [sp, #LOCALS + 8]
        mov     x21, #0
        cbz     x20, 2f
1:      ldrb    w22, [x19], #1
        add     x21, x21, x22
        subs    x20, x20, #1
        b.ne    1b
2:      mov     x0, x21
        EPILOGUE 5, 4, 32
ENDFUNC toks_k0_frame_neon
