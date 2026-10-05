/* tests/asm/x86_64/labels.S: the x86-64 macro layer's assemble test (make asmcheck, part of make test), assembled
 * for mach-o, elf and coff: a LOCAL label right before EPILOGUE in a PROLOGUE function (an .L label there is a
 * symbol on mach-o and broke the frame's cfi: "invalid CFI advance_loc expression") and LOCAL labels in a leaf.
 * Never linked, never run. */
#include "asm.h"

FUNC toks_asmcheck_frame
        PROLOGUE 2, 0, 16
        test    ARG0, ARG0
        jz      LOCAL(done)
        xor     eax, eax
LOCAL(done):
        EPILOGUE 2, 0, 16
ENDFUNC toks_asmcheck_frame

FUNC toks_asmcheck_leaf
        LEAF
        xor     eax, eax
LOCAL(loop):
        test    ARG1, ARG1
        jz      LOCAL(out)
        add     rax, ARG1
        dec     ARG1
        jmp     LOCAL(loop)
LOCAL(out):
        ret
ENDFUNC toks_asmcheck_leaf
