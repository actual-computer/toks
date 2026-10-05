/* src/asm/x86_64/asm.h: the x86-64 macro layer (SPEC §10, §11). #include it from every .S in
 * src/asm/x86_64. One source assembles for mach-o (macos, sysv), elf (linux, sysv) and coff (windows, win64).
 * Syntax: .intel_syntax noprefix (set here).
 *
 * every function
 *     FUNC name          global, hidden (internal to libtoks), 16-byte aligned, endbr64, unwind start
 *     LEAF  |  PROLOGUE ngpr, nxmm, locals    exactly one of these right after FUNC
 *     ...                body
 *     ret   |  EPILOGUE ngpr, nxmm, locals    (EPILOGUE restores and returns; code may follow it)
 *     ENDFUNC name
 *
 * arguments ARG0..ARG3 (and ARG0d..ARG3d): rdi rsi rdx rcx on sysv, rcx rdx r8 r9 on win64. Kernels take
 *           two arguments (t, a); copy them where you want them first thing.
 * leaf      may use rax, rcx, rdx, r8-r11 (volatile on both abis; mind that ARG0/ARG1 live in two of them),
 *           ARG0/ARG1, ymm0-5 and zmm0-5 / zmm16-31; nothing on the stack.
 * framed    PROLOGUE pushes the first ngpr of rbx, rbp, r12, r13, r14, r15 (ngpr = 0..6) and, on win64,
 *           always rdi and rsi first, so after PROLOGUE every gpr but rsp is usable on both abis (on sysv
 *           rdi/rsi still hold the arguments until you move them). On win64 it also saves xmm6.. (nxmm =
 *           0..10; ignored on sysv where all vector registers are volatile), so ymm/zmm 6-15 are usable
 *           too. `locals` bytes (a multiple of 16) are at [rsp + 0, rsp + locals), 16-byte aligned.
 *           Frame (pushes + allocation) <= 256 bytes. Unwind info (cfi on elf/mach-o, seh on coff) comes
 *           from the same numbers; EPILOGUE must repeat them.
 * vzeroupper before every return from code that touched ymm/zmm upper state (the macros do not).
 * addresses rip-relative only: lea r, [rip + SYM(name)]. No absolute relocations.
 * labels    LOCAL(name) for a branch target private to the object (never a bare .L or L label: below)
 * data      RODATA, then `DATA name` labels a local (not exported) symbol; TEXT returns to code.
 */
#ifndef TOKS_X86_64_ASM_H
#define TOKS_X86_64_ASM_H

#if defined(__APPLE__)
#  define TOKS_MACHO 1
#elif defined(_WIN32)
#  define TOKS_COFF 1
#  define TOKS_WIN64 1
#else
#  define TOKS_ELF 1
#endif

#if defined(TOKS_MACHO)
#  define SYM(x) _##x
#else
#  define SYM(x) x
#endif

/* LOCAL(name): a label private to the object file (`jz LOCAL(done)` ... `LOCAL(done):`). Mach-O's
 * assembler-local prefix is L: an .L label there is a real symbol, and one inside a PROLOGUE function
 * breaks the frame's cfi ("invalid CFI advance_loc expression" at EPILOGUE). ELF and COFF use .L (an L
 * label is a symbol on COFF). Numeric labels (1:, 1b, 1f) are local everywhere too. */
#if defined(TOKS_MACHO)
#  define LOCAL(x) L##x
#else
#  define LOCAL(x) .L##x
#endif

#if defined(TOKS_WIN64)
#  define ARG0 rcx
#  define ARG1 rdx
#  define ARG2 r8
#  define ARG3 r9
#  define ARG0d ecx
#  define ARG1d edx
#  define ARG2d r8d
#  define ARG3d r9d
#else
#  define ARG0 rdi
#  define ARG1 rsi
#  define ARG2 rdx
#  define ARG3 rcx
#  define ARG0d edi
#  define ARG1d esi
#  define ARG2d edx
#  define ARG3d ecx
#endif

        .intel_syntax noprefix

#if defined(TOKS_ELF)
/* GNU_PROPERTY_X86_FEATURE_1_AND: IBT | SHSTK (every FUNC starts with endbr64; returns are plain). */
        .pushsection .note.gnu.property, "a", @note
        .p2align 3
        .long   4
        .long   16
        .long   5
        .asciz  "GNU"
        .long   0xc0000002
        .long   4
        .long   3
        .long   0
        .popsection
        .pushsection .note.GNU-stack, "", @progbits
        .popsection
#endif

.macro TEXT
        .text
.endm

.macro RODATA
#if defined(TOKS_MACHO)
        .section __TEXT,__const
#elif defined(TOKS_COFF)
        .section .rdata, "dr"
#else
        .section .rodata
#endif
.endm

.macro DATA name
#if defined(TOKS_MACHO)
_\name:
#else
\name:
#endif
.endm

.macro FUNC name
        .text
#if defined(TOKS_MACHO)
        .globl  _\name
#else
        .globl  \name
#endif
#if defined(TOKS_ELF)
        .hidden \name
        .type   \name, @function
#elif defined(TOKS_MACHO)
        .private_extern _\name
#else
        .def    \name
        .scl    2
        .type   32
        .endef
#endif
        .p2align 4
#if defined(TOKS_MACHO)
_\name:
#else
\name:
#endif
#if defined(TOKS_COFF)
        .seh_proc \name
#else
        .cfi_startproc
#endif
        endbr64
.endm

.macro ENDFUNC name
#if defined(TOKS_COFF)
        .seh_endproc
#else
        .cfi_endproc
#endif
#if defined(TOKS_ELF)
        .size   \name, . - \name
#endif
.endm

.macro LEAF
#if defined(TOKS_COFF)
        .seh_endprologue
#endif
.endm

.macro TOKS_PUSH r
        push    \r
#if defined(TOKS_COFF)
        .seh_pushreg \r
#else
        .cfi_adjust_cfa_offset 8
        .cfi_rel_offset \r, 0
#endif
.endm

.macro TOKS_POP r
        pop     \r
#if !defined(TOKS_COFF)
        .cfi_adjust_cfa_offset -8
        .cfi_restore \r
#endif
.endm

.macro TOKS_XMM_SAVE n, off
        movaps  [rsp + \off], xmm\n
        .seh_savexmm xmm\n, \off
.endm

.macro TOKS_XMM_LOAD n, off
        movaps  xmm\n, [rsp + \off]
.endm

/* pushes, vector-save bytes and the allocation for (ngpr, nxmm, locals) */
.macro TOKS_FRAME_CALC ngpr, nxmm, locals
  .if (\ngpr < 0) || (\ngpr > 6) || (\nxmm < 0) || (\nxmm > 10) || ((\locals) % 16)
        .error "toks: PROLOGUE ngpr 0..6, nxmm 0..10, locals a multiple of 16"
  .endif
#if defined(TOKS_WIN64)
  .set TOKS_PUSHES, (\ngpr) + 2
  .set TOKS_XMMB, 16*(\nxmm)
#else
  .set TOKS_PUSHES, (\ngpr)
  .set TOKS_XMMB, 0
#endif
  /* entry rsp = 8 (mod 16); after the pushes rsp = 8 + 8*pushes (mod 16); pad so rsp ends 16-aligned */
  .set TOKS_ALLOC, (\locals) + TOKS_XMMB + (((TOKS_PUSHES + 1) % 2) * 8)
  .if (8*TOKS_PUSHES + TOKS_ALLOC) > 256
        .error "toks: frame > 256 bytes (SPEC §10.1)"
  .endif
.endm

.macro PROLOGUE ngpr, nxmm, locals
        TOKS_FRAME_CALC \ngpr, \nxmm, \locals
#if defined(TOKS_WIN64)
        TOKS_PUSH rdi
        TOKS_PUSH rsi
#endif
  .if \ngpr >= 1
        TOKS_PUSH rbx
  .endif
  .if \ngpr >= 2
        TOKS_PUSH rbp
  .endif
  .if \ngpr >= 3
        TOKS_PUSH r12
  .endif
  .if \ngpr >= 4
        TOKS_PUSH r13
  .endif
  .if \ngpr >= 5
        TOKS_PUSH r14
  .endif
  .if \ngpr >= 6
        TOKS_PUSH r15
  .endif
  .if TOKS_ALLOC > 0
        sub     rsp, TOKS_ALLOC
#if defined(TOKS_COFF)
        .seh_stackalloc TOKS_ALLOC
#else
        .cfi_adjust_cfa_offset TOKS_ALLOC
#endif
  .endif
#if defined(TOKS_WIN64)
  .if \nxmm >= 1
        TOKS_XMM_SAVE 6, (\locals)
  .endif
  .if \nxmm >= 2
        TOKS_XMM_SAVE 7, (\locals) + 16
  .endif
  .if \nxmm >= 3
        TOKS_XMM_SAVE 8, (\locals) + 32
  .endif
  .if \nxmm >= 4
        TOKS_XMM_SAVE 9, (\locals) + 48
  .endif
  .if \nxmm >= 5
        TOKS_XMM_SAVE 10, (\locals) + 64
  .endif
  .if \nxmm >= 6
        TOKS_XMM_SAVE 11, (\locals) + 80
  .endif
  .if \nxmm >= 7
        TOKS_XMM_SAVE 12, (\locals) + 96
  .endif
  .if \nxmm >= 8
        TOKS_XMM_SAVE 13, (\locals) + 112
  .endif
  .if \nxmm >= 9
        TOKS_XMM_SAVE 14, (\locals) + 128
  .endif
  .if \nxmm >= 10
        TOKS_XMM_SAVE 15, (\locals) + 144
  .endif
        .seh_endprologue
#endif
.endm

.macro EPILOGUE ngpr, nxmm, locals
        TOKS_FRAME_CALC \ngpr, \nxmm, \locals
#if !defined(TOKS_COFF)
        .cfi_remember_state
#endif
#if defined(TOKS_WIN64)
  .if \nxmm >= 10
        TOKS_XMM_LOAD 15, (\locals) + 144
  .endif
  .if \nxmm >= 9
        TOKS_XMM_LOAD 14, (\locals) + 128
  .endif
  .if \nxmm >= 8
        TOKS_XMM_LOAD 13, (\locals) + 112
  .endif
  .if \nxmm >= 7
        TOKS_XMM_LOAD 12, (\locals) + 96
  .endif
  .if \nxmm >= 6
        TOKS_XMM_LOAD 11, (\locals) + 80
  .endif
  .if \nxmm >= 5
        TOKS_XMM_LOAD 10, (\locals) + 64
  .endif
  .if \nxmm >= 4
        TOKS_XMM_LOAD 9, (\locals) + 48
  .endif
  .if \nxmm >= 3
        TOKS_XMM_LOAD 8, (\locals) + 32
  .endif
  .if \nxmm >= 2
        TOKS_XMM_LOAD 7, (\locals) + 16
  .endif
  .if \nxmm >= 1
        TOKS_XMM_LOAD 6, (\locals)
  .endif
#endif
  .if TOKS_ALLOC > 0
        add     rsp, TOKS_ALLOC
#if !defined(TOKS_COFF)
        .cfi_adjust_cfa_offset -TOKS_ALLOC
#endif
  .endif
  .if \ngpr >= 6
        TOKS_POP r15
  .endif
  .if \ngpr >= 5
        TOKS_POP r14
  .endif
  .if \ngpr >= 4
        TOKS_POP r13
  .endif
  .if \ngpr >= 3
        TOKS_POP r12
  .endif
  .if \ngpr >= 2
        TOKS_POP rbp
  .endif
  .if \ngpr >= 1
        TOKS_POP rbx
  .endif
#if defined(TOKS_WIN64)
        TOKS_POP rsi
        TOKS_POP rdi
#endif
        ret
#if !defined(TOKS_COFF)
        .cfi_restore_state
#endif
.endm

#endif /* TOKS_X86_64_ASM_H */
