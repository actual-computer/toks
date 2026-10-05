/* src/asm/arm64/asm.h: the arm64 macro layer (SPEC §10, §11). #include it from every .S in src/asm/arm64.
 * One source assembles for mach-o (macos), elf (linux) and coff (windows).
 *
 * every function
 *     FUNC name          global, hidden (internal to libtoks), 16-byte aligned, landing pad (bti c on elf),
 *                        unwind start
 *     LEAF  |  PROLOGUE nx, nd, locals      exactly one of these right after FUNC
 *     ...                body
 *     ret   |  EPILOGUE nx, nd, locals      (EPILOGUE restores and returns; code may follow it)
 *     ENDFUNC name
 *
 * leaf      uses only x0-x17, v0-v7, v16-v31 and no stack.
 * framed    PROLOGUE saves x29/x30, nx pairs of x19..x28 (nx = 0..5: x19-x20, x21-x22, ...) and nd pairs of
 *           d8..d15 (nd = 0..4), and reserves `locals` bytes (a multiple of 16) at [sp, #LOCALS] where
 *           LOCALS = 16 + 16*nx + 16*nd. Frame <= 256 bytes. The unwind info (cfi on elf/mach-o, seh on
 *           coff) is generated from the same three numbers; EPILOGUE must repeat them.
 * never     x18 (the platform register on macos and windows), anything below sp (no red zone), indirect
 *           calls, address-taken functions.
 * addresses ADRP_ADD xd, SYM(name)   (page + low 12 bits on every format; no absolute relocations)
 * labels    LOCAL(name) for a branch target private to the object (never a bare .L or L label: below)
 * data      RODATA, then `DATA name` labels a local (not exported) symbol; TEXT returns to code.
 */
#ifndef TOKS_ARM64_ASM_H
#define TOKS_ARM64_ASM_H

#if defined(__APPLE__)
#  define TOKS_MACHO 1
#elif defined(_WIN32)
#  define TOKS_COFF 1
#else
#  define TOKS_ELF 1
#endif

#if defined(TOKS_MACHO)
#  define SYM(x) _##x
#else
#  define SYM(x) x
#endif

/* LOCAL(name): a label private to the object file (`b.eq LOCAL(done)` ... `LOCAL(done):`). Mach-O's
 * assembler-local prefix is L: an .L label there is a real symbol, and one inside a PROLOGUE function
 * breaks the frame's cfi ("invalid CFI advance_loc expression" at EPILOGUE). ELF and COFF use .L (an L
 * label is a symbol on COFF). Numeric labels (1:, 1b, 1f) are local everywhere too. */
#if defined(TOKS_MACHO)
#  define LOCAL(x) L##x
#else
#  define LOCAL(x) .L##x
#endif

/* tier neon = armv8.0-a + neon + crc32 (SPEC §11, cpu.h): clang's linux and windows arm64 default is
 * armv8.0-a without crc, where crc32c* do not assemble; mach-o's default cpu has it. */
        .arch_extension crc

#if defined(TOKS_ELF)
/* GNU_PROPERTY_AARCH64_FEATURE_1_AND: BTI (every FUNC starts with bti c). PAC is not used. */
        .pushsection .note.gnu.property, "a", %note
        .p2align 3
        .long   4
        .long   16
        .long   5
        .asciz  "GNU"
        .long   0xc0000000
        .long   4
        .long   1
        .long   0
        .popsection
        .pushsection .note.GNU-stack, "", %progbits
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

.macro ADRP_ADD xd, sym
#if defined(TOKS_MACHO)
        adrp    \xd, \sym@PAGE
        add     \xd, \xd, \sym@PAGEOFF
#else
        adrp    \xd, \sym
        add     \xd, \xd, :lo12:\sym
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
        .type   \name, %function
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
#if defined(TOKS_ELF)
        bti     c
#endif
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

/* one saved register pair: x pair k (k = 0..4 -> x19+2k) or d pair k (k = 0..3 -> d8+2k) at sp + off */
.macro TOKS_SAVE_PAIR kind, r1, r2, off, frame
        stp     \kind\r1, \kind\r2, [sp, #\off]
#if defined(TOKS_COFF)
  .ifc \kind, x
        .seh_save_regp x\r1, \off
  .else
        .seh_save_fregp d\r1, \off
  .endif
#else
        .cfi_offset \kind\r1, \off - \frame
        .cfi_offset \kind\r2, \off + 8 - \frame
#endif
.endm

.macro TOKS_LOAD_PAIR kind, r1, r2, off
        ldp     \kind\r1, \kind\r2, [sp, #\off]
#if defined(TOKS_COFF)
  .ifc \kind, x
        .seh_save_regp x\r1, \off
  .else
        .seh_save_fregp d\r1, \off
  .endif
#endif
.endm

.macro PROLOGUE nx, nd, locals
  .if (\nx < 0) || (\nx > 5) || (\nd < 0) || (\nd > 4) || ((\locals) % 16)
        .error "toks: PROLOGUE nx 0..5, nd 0..4, locals a multiple of 16"
  .endif
  .set TOKS_FRAME, 16 + 16*(\nx) + 16*(\nd) + (\locals)
  .set LOCALS, 16 + 16*(\nx) + 16*(\nd)
  .if TOKS_FRAME > 256
        .error "toks: frame > 256 bytes (SPEC §10.1)"
  .endif
        stp     x29, x30, [sp, #-TOKS_FRAME]!
#if defined(TOKS_COFF)
        .seh_save_fplr_x TOKS_FRAME
#else
        .cfi_def_cfa_offset TOKS_FRAME
        .cfi_offset x29, -TOKS_FRAME
        .cfi_offset x30, 8 - TOKS_FRAME
#endif
  .if \nx >= 1
        TOKS_SAVE_PAIR x, 19, 20, 16, TOKS_FRAME
  .endif
  .if \nx >= 2
        TOKS_SAVE_PAIR x, 21, 22, 32, TOKS_FRAME
  .endif
  .if \nx >= 3
        TOKS_SAVE_PAIR x, 23, 24, 48, TOKS_FRAME
  .endif
  .if \nx >= 4
        TOKS_SAVE_PAIR x, 25, 26, 64, TOKS_FRAME
  .endif
  .if \nx >= 5
        TOKS_SAVE_PAIR x, 27, 28, 80, TOKS_FRAME
  .endif
  .if \nd >= 1
        TOKS_SAVE_PAIR d, 8, 9, 16 + 16*(\nx), TOKS_FRAME
  .endif
  .if \nd >= 2
        TOKS_SAVE_PAIR d, 10, 11, 32 + 16*(\nx), TOKS_FRAME
  .endif
  .if \nd >= 3
        TOKS_SAVE_PAIR d, 12, 13, 48 + 16*(\nx), TOKS_FRAME
  .endif
  .if \nd >= 4
        TOKS_SAVE_PAIR d, 14, 15, 64 + 16*(\nx), TOKS_FRAME
  .endif
#if defined(TOKS_COFF)
        .seh_endprologue
#endif
.endm

.macro EPILOGUE nx, nd, locals
  .set TOKS_FRAME, 16 + 16*(\nx) + 16*(\nd) + (\locals)
#if defined(TOKS_COFF)
        .seh_startepilogue
#else
        .cfi_remember_state
#endif
  .if \nd >= 4
        TOKS_LOAD_PAIR d, 14, 15, 64 + 16*(\nx)
  .endif
  .if \nd >= 3
        TOKS_LOAD_PAIR d, 12, 13, 48 + 16*(\nx)
  .endif
  .if \nd >= 2
        TOKS_LOAD_PAIR d, 10, 11, 32 + 16*(\nx)
  .endif
  .if \nd >= 1
        TOKS_LOAD_PAIR d, 8, 9, 16 + 16*(\nx)
  .endif
  .if \nx >= 5
        TOKS_LOAD_PAIR x, 27, 28, 80
  .endif
  .if \nx >= 4
        TOKS_LOAD_PAIR x, 25, 26, 64
  .endif
  .if \nx >= 3
        TOKS_LOAD_PAIR x, 23, 24, 48
  .endif
  .if \nx >= 2
        TOKS_LOAD_PAIR x, 21, 22, 32
  .endif
  .if \nx >= 1
        TOKS_LOAD_PAIR x, 19, 20, 16
  .endif
        ldp     x29, x30, [sp], #TOKS_FRAME
#if defined(TOKS_COFF)
        .seh_save_fplr_x TOKS_FRAME
        .seh_endepilogue
        ret
#else
        .cfi_def_cfa_offset 0
        ret
        .cfi_restore_state
#endif
.endm

#endif /* TOKS_ARM64_ASM_H */
