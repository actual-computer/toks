/* tests/common/abicheck_arm64.S
 * uint64_t toks_abicheck_call(const void *fn, uint64_t a0, uint64_t a1, uint64_t *report)
 * Calls fn(a0, a1) with canaries in every callee-saved register and reports, in *report, a bit per register
 * the callee failed to preserve: bit k for x(19+k) (k = 0..9), bit 16+k for d(8+k) (k = 0..7), bit 30 for
 * x18 (never touched by toks; holds the platform's value on macos / windows), bit 31 for sp.
 * Returns fn's x0. Test-only code (the indirect call is not allowed in the library).
 */
#include "../../src/asm/arm64/asm.h"

FUNC toks_abicheck_call
        PROLOGUE 5, 4, 32
        str     x3, [sp, #LOCALS]
        mov     x9, sp
        str     x9, [sp, #LOCALS + 8]
        str     x18, [sp, #LOCALS + 16]
        mov     x16, x0
        mov     x0, x1
        mov     x1, x2
  .irp r, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
        movz    x\r, #\r
        movk    x\r, #0xc0de, lsl #48
  .endr
  .irp r, 8, 9, 10, 11, 12, 13, 14, 15
        movz    x9, #\r
        movk    x9, #0xd00d, lsl #48
        fmov    d\r, x9
  .endr
        blr     x16
        mov     x9, #0
  .irp r, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
        movz    x10, #\r
        movk    x10, #0xc0de, lsl #48
        cmp     x\r, x10
        cset    x11, ne
        orr     x9, x9, x11, lsl #(\r - 19)
  .endr
  .irp r, 8, 9, 10, 11, 12, 13, 14, 15
        movz    x10, #\r
        movk    x10, #0xd00d, lsl #48
        fmov    x12, d\r
        cmp     x12, x10
        cset    x11, ne
        orr     x9, x9, x11, lsl #(16 + \r - 8)
  .endr
        ldr     x10, [sp, #LOCALS + 16]
        cmp     x18, x10
        cset    x11, ne
        orr     x9, x9, x11, lsl #30
        ldr     x10, [sp, #LOCALS + 8]
        mov     x12, sp
        cmp     x12, x10
        cset    x11, ne
        orr     x9, x9, x11, lsl #31
        ldr     x10, [sp, #LOCALS]
        str     x9, [x10]
        EPILOGUE 5, 4, 32
ENDFUNC toks_abicheck_call
