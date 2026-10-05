/* tests/asm/arm64/labels.S: the arm64 macro layer's assemble test (make asmcheck, part of make test), assembled
 * for mach-o, elf and coff: a LOCAL label right before EPILOGUE in a PROLOGUE function (an .L label there is a
 * symbol on mach-o and broke the frame's cfi: "invalid CFI advance_loc expression"), LOCAL labels in a leaf,
 * and crc32c (tier neon's crc32: elf and coff assemble it only after asm.h's .arch_extension crc).
 * Never linked, never run. */
#include "asm.h"

FUNC toks_asmcheck_frame
        PROLOGUE 5, 0, 96
        cbz     x0, LOCAL(done)
        mov     x0, #0
LOCAL(done):
        EPILOGUE 5, 0, 96
ENDFUNC toks_asmcheck_frame

FUNC toks_asmcheck_leaf
        LEAF
        mov     w2, #0
LOCAL(loop):
        cbz     x1, LOCAL(out)
        ldrb    w3, [x0], #1
        crc32cb w2, w2, w3
        sub     x1, x1, #1
        b       LOCAL(loop)
LOCAL(out):
        crc32cx w0, w2, x1
        ret
ENDFUNC toks_asmcheck_leaf
