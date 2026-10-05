/*
 * src/asm/arm64/k3_neon.inc: the neon K3 both templates assemble from (docs/kernels.md §3.1; o200k:
 * docs/templates/o200k.md §4): k3_cl100k_neon.S sets K3_O200K 0, k3_o200k_neon.S sets it 1, k3_dsv3_neon.S sets
 * it and K3_DSV3 (o200k's algebra on dsv3's tables, docs/templates/dsv3.md §7). A transcription of
 * src/asm/x86_64/k3_avx2.inc (the same blocks, algebra, decided range and stretches left to the c twin; the
 * same labels and frame slots), with the #20 review's neon parts: ld4 + a 128-entry code table (tbl / tbx) + the
 * sri / shrn movemask for the class masks, and the byte-LUT emission.
 *
 * Classes: each ascii byte's code (k3n2_code, codes in bits 4-7) gives four planes: b0 letters, b1 lower /
 * digit / NL / '/', b2 \s, b3 ' ' / ' / '/'; with NA (the byte's top bit): L = b0, U = b0 & ~b1, W lower
 * = b0 & b1, N = b1 & ~(b0 | b2 | b3), W (\s) = b2, NL = b2 & b1, SP = b2 & b3, AP = b3 & ~(b1 | b2),
 * SL = b3 & b1. The ascii classes are hard-coded (kernels.md §2's precondition, checked by the compiler).
 *
 * Registers: x19 a, x20 text, x21 len, x22 pos, x23 the next ends slot, x24 the end of ends (cap), x26
 * cls_stage1, x27 cls_stage2, x28 tmpl_params | flags << 32. The avx2 body's rax rcx rdx rbp rsi rdi r8 r9
 * r10 r11 are x0-x9; x10-x17 are temps. ATOM (a bl subroutine): in x13 = an offset < len, out w14 = its atom's
 * class, x15 = its length; clobbers x12 x16 x17 x30. v0-v3 the block (ld4 lanes: lane j byte i is block byte
 * 4i + j), v16-v23 the code table, v24 0x40; the rest temps. Frame: PROLOGUE 5, 0, LOCALS_N (x19-x28 saved),
 * the slots at [sp, #96 + S_x].
 */
#include "asm.h"
#include "layout.h"
#ifndef K3_DSV3
#define K3_DSV3 0
#endif

#define L(x) LOCAL(x)
#if defined(TOKS_COFF)
#define LOOP_ALIGN                      /* windows arm64: SEH needs a function length the assembler can compute */
#else
#define LOOP_ALIGN .p2align 5           /* the per-atom loops' heads (kernels.md §3.1) */
#endif
#define FR 96                   /* the slots' base above sp: 16 (x29 x30) + 5 pairs */
#define S_ENDS0  0              /* ends */
#define S_UNTIL  8              /* the pieces before until are the twin's (through a cut atom); 0: none */
#define S_E      16             /* block: bytes in play */
#define S_CAND   24             /* cl100k block: ' at piece starts */
#define S_P1     32             /* atom pass: the non-ascii bytes (L(na_give_up)), scratch */
#define S_CONT   40             /* block: the bytes after the first of each non-ascii atom */
#define S_NWN    48             /* block: a non-W byte follows */
#define S_NNA    56             /* block: non-ascii N atoms */
#define S_WNA    64             /* block: non-ascii WS atoms */
#if K3_O200K
#define S_TAILS  72             /* block: the '/' the last mark round's tails took */
#define S_AU     S_CAND         /* block: upper-only atoms (A-Z, Lu, Lt) */
#define S_AW     80             /* block: lower-only atoms (a-z, Ll) */
#define S_AB     88             /* block: base-L atoms in both groups (Lm, Lo) */
#define S_AM     96             /* block: marks (base P, both groups); from the tail step on, C (every case atom) */
#define S_AH     104            /* block: Han letters (TOKS_TP_HAN); dsv3: its CJK letters */
#if K3_DSV3
#define S_DM     112            /* dsv3 block: the ascii bytes 20-7E */
#define S_DX     120            /* dsv3 block: the ascii X bytes 00-08 0E-1F 7F */
#define S_DA     128            /* dsv3 block: the ascii letters */
#define LOCALS_N 144
#define K3_TF_CJK TOKS_TF_CJK_D         /* U+4E00..U+9EFF atoms: Han letters */
#define S_CJK    S_AH
#else
#define LOCALS_N 112
#define K3_TF_CJK TOKS_TF_CJK_B
#define S_CJK    S_AB
#endif
#else
#define S_LF     72             /* cl100k block (NL_CUT): the LF bytes, then the cuts C (kernels.md §3 A9); 0 without */
#define LOCALS_N 80
#define K3_TF_CJK TOKS_TF_CJK_L
#endif
#define SLOT(x) [sp, #(FR + (x))]

/* BZHI d, s, n, t: d = s with the bits at and above n (0..64) cleared (t a temp) */
.macro BZHI d, s, n, t
        mov     \t, #-1
        lsl     \t, \t, \n
        tst     \n, #64
        csel    \t, xzr, \t, ne
        bic     \d, \s, \t
.endm

/* TZ d, s: trailing zeros (64 for 0) */
.macro TZ d, s
        rbit    \d, \s
        clz     \d, \d
.endm

/* BLSR d, s, t: d = s & (s - 1) */
.macro BLSR d, s, t
        sub     \t, \s, #1
        and     \d, \s, \t
.endm

/* MM xd, a, b, c, d: the top bits of the four ld4 lanes' bytes as one natural-order u64 (the #20 review's
 * sri / shrn movemask; register numbers); destroys b, c, d */
.macro MM xd, a, b, c, d
        sri     v\b\().16b, v\a\().16b, #1
        sri     v\d\().16b, v\c\().16b, #1
        sri     v\d\().16b, v\b\().16b, #2
        sri     v\d\().16b, v\d\().16b, #4
        shrn    v\d\().8b, v\d\().8h, #4
        fmov    \xd, d\d
.endm

/* PLANE xd, sh: plane bit (7 - sh) of the codes v4-v7 as a mask */
.macro PLANE xd, sh
        shl     v25.16b, v4.16b, #\sh
        shl     v26.16b, v5.16b, #\sh
        shl     v27.16b, v6.16b, #\sh
        shl     v28.16b, v7.16b, #\sh
        MM      \xd, 25, 26, 27, 28
.endm

/* RANGE xd, lo, n: the bytes b of v0-v3 with b - lo < n (unsigned) as a mask; v29 / v30 hold lo / n splats */
.macro RANGE xd, lo, n
        movi    v29.16b, #\lo
        movi    v30.16b, #\n
        sub     v25.16b, v0.16b, v29.16b
        sub     v26.16b, v1.16b, v29.16b
        sub     v27.16b, v2.16b, v29.16b
        sub     v28.16b, v3.16b, v29.16b
        cmhi    v25.16b, v30.16b, v25.16b
        cmhi    v26.16b, v30.16b, v26.16b
        cmhi    v27.16b, v30.16b, v27.16b
        cmhi    v28.16b, v30.16b, v28.16b
        MM      \xd, 25, 26, 27, 28
.endm

/* CLASSIFY: the 64 bytes in v0-v3 (ld4 lanes) into x0 = non-ascii, x4 = L (o200k: '/', with A-Z and a-z
 * in S_AU and S_AW), x5 = N, x6 = W (\s), x7 = NL, x8 = ' ', x9 = '. Clobbers x10-x13, v4-v7, v25-v28 */
.macro CLASSIFY
        tbl     v4.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v0.16b
        tbl     v5.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v1.16b
        tbl     v6.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v2.16b
        tbl     v7.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v3.16b
        sub     v25.16b, v0.16b, v24.16b
        sub     v26.16b, v1.16b, v24.16b
        sub     v27.16b, v2.16b, v24.16b
        sub     v28.16b, v3.16b, v24.16b
        tbx     v4.16b, {v20.16b, v21.16b, v22.16b, v23.16b}, v25.16b
        tbx     v5.16b, {v20.16b, v21.16b, v22.16b, v23.16b}, v26.16b
        tbx     v6.16b, {v20.16b, v21.16b, v22.16b, v23.16b}, v27.16b
        tbx     v7.16b, {v20.16b, v21.16b, v22.16b, v23.16b}, v28.16b
        PLANE   x11, 3                          /* b0 */
        PLANE   x12, 2                          /* b1 */
        PLANE   x13, 1                          /* b2 */
        PLANE   x10, 0                          /* b3 */
        mov     v26.16b, v1.16b
        mov     v27.16b, v2.16b
        mov     v28.16b, v3.16b
        MM      x0, 0, 26, 27, 28               /* non-ascii */
        and     x7, x13, x12                    /* NL */
        and     x8, x13, x10                    /* ' ' */
        orr     x9, x12, x13
        bic     x9, x10, x9                     /* ' */
        orr     x5, x11, x13
        orr     x5, x5, x10
        bic     x5, x12, x5                     /* N */
        mov     x6, x13                         /* W */
#if K3_O200K
        bic     x4, x11, x12
        str     x4, SLOT(S_AU)                  /* A-Z */
        and     x4, x11, x12
        str     x4, SLOT(S_AW)                  /* a-z */
        and     x4, x10, x12                    /* '/' */
#if K3_DSV3
        RANGE   x10, 0x20, 0x5F                 /* 20-7E */
        str     x10, SLOT(S_DM)
        orr     x10, x10, x0
        orr     x10, x10, x6
        mvn     x10, x10
        str     x10, SLOT(S_DX)                 /* the ascii X bytes */
#endif
#else
        mov     x4, x11                         /* L */
        tst     x28, #TOKS_TP_NL_CUT
        b.eq    96f
        movi    v29.16b, #0x0A
        cmeq    v25.16b, v0.16b, v29.16b
        cmeq    v26.16b, v1.16b, v29.16b
        cmeq    v27.16b, v2.16b, v29.16b
        cmeq    v28.16b, v3.16b, v29.16b
        MM      x10, 25, 26, 27, 28
        str     x10, SLOT(S_LF)                 /* LF (A9) */
96:
#endif
.endm

/* EMIT: an end x22 + j for each set bit j of x0, ascending, at x23 (advanced). With room for the count + 8,
 * each mask byte stores its 8 table indices + base as two q registers and advances by its popcount (no
 * data-dependent branch; the slack past the count lies inside ends[0, cap)); otherwise one end per bit while
 * room lasts, and L(exit) when ends are left over. Clobbers x0-x3, x10, v26-v31. */
.macro EMIT
        fmov    d29, x0
        cnt     v29.8b, v29.8b
        addv    b28, v29.8b
        fmov    w1, s28
        add     x2, x23, x1, lsl #2
        add     x2, x2, #32
        cmp     x2, x24
        b.hi    87f
        ADRP_ADD x10, SYM(k3n2_bitidx)
        dup     v30.4s, w22
        movi    v31.4s, #8
        .irp    k, 0, 1, 2, 3, 4, 5, 6, 7
        ubfx    x3, x0, #(8 * \k), #8
        ldr     d27, [x10, x3, lsl #3]
        ushll   v27.8h, v27.8b, #0
        uxtl2   v26.4s, v27.8h
        uxtl    v27.4s, v27.4h
        add     v27.4s, v27.4s, v30.4s
        add     v26.4s, v26.4s, v30.4s
        stp     q27, q26, [x23]
        umov    w3, v29.b[\k]
        add     x23, x23, x3, lsl #2
        add     v30.4s, v30.4s, v31.4s
        .endr
        b       89f
87:     cbz     x0, 89f
88:     TZ      x3, x0
        add     w3, w3, w22
        str     w3, [x23], #4
        BLSR    x0, x0, x2
        cbz     x0, 89f
        cmp     x23, x24
        b.lo    88b
        b       L(exit)
89:
.endm

/* BITIDX b: the indices of b's set bits ascending, padded to 8 bytes with 0 */
.macro BITIDX b
        .set    k3n2_n, 0
        .irp    i, 0, 1, 2, 3, 4, 5, 6, 7
        .if     ((\b) >> \i) & 1
        .byte   \i
        .set    k3n2_n, k3n2_n + 1
        .endif
        .endr
        .rept   8 - k3n2_n
        .byte   0
        .endr
.endm

/* CODE b: the class code of ascii byte b (bits 4-7: b0 letter, b1, b2 \s, b3) */
.macro CODE b
  .if ((\b) >= 0x41 && (\b) <= 0x5A)
        .byte   0x10
  .elseif ((\b) >= 0x61 && (\b) <= 0x7A)
        .byte   0x30
  .elseif ((\b) >= 0x30 && (\b) <= 0x39)
        .byte   0x20
  .elseif ((\b) == 0x0A || (\b) == 0x0D)
        .byte   0x60
  .elseif ((\b) == 0x09 || (\b) == 0x0B || (\b) == 0x0C)
        .byte   0x40
  .elseif (\b) == 0x20
        .byte   0xC0
  .elseif (\b) == 0x27
        .byte   0x80
  .elseif (\b) == 0x2F
        .byte   0xA0
  .else
        .byte   0
  .endif
.endm

/* K3O2_SLOT c: the slot (FR + S_x) of class byte c in the o200k atom pass, 0xFF: cut (see k3_avx2.inc) */
.macro K3O2_SLOT c
  .if ((\c) & 0x40) != 0
        .byte   0xFF
  .elseif ((\c) & 0x80) != 0
    .if ((\c) & 7) == 1
        .byte   FR + S_AH
    .else
        .byte   0xFF
    .endif
  .elseif ((\c) & 7) == 1
    .if ((\c) & 0x18) == 0x08
        .byte   FR + S_AU
    .elseif ((\c) & 0x18) == 0x10
        .byte   FR + S_AW
    .elseif ((\c) & 0x18) == 0x18
        .byte   FR + S_AB
    .else
#if K3_DSV3
        .byte   FR + S_AB                       /* dsv3: letters have no case group */
#else
        .byte   0xFF
#endif
    .endif
  .elseif ((\c) & 7) == 0
    .if ((\c) & 0x18) == 0
        .byte   FR + S_NWN
    .elseif ((\c) & 0x18) == 0x18
        .byte   FR + S_AM
    .else
        .byte   0xFF
    .endif
  .elseif ((\c) & 0x1F) == 2
        .byte   FR + S_NNA
  .elseif ((\c) & 0x1F) == 3
        .byte   FR + S_WNA
  .else
        .byte   0xFF
  .endif
.endm

        RODATA
        .p2align 4
DATA k3n2_code                                  /* [b]: the class code of ascii byte b */
        .set    k3n2_b, 0
        .rept   128
        CODE    k3n2_b
        .set    k3n2_b, k3n2_b + 1
        .endr
DATA k3n2_bitidx                                /* [b]: the indices of b's set bits, ascending, 0-padded */
        .set    k3n2_b, 0
        .rept   256
        BITIDX  k3n2_b
        .set    k3n2_b, k3n2_b + 1
        .endr
DATA k3n2_contr                                 /* [c1]: 0x80 for s t m d; e for r v; l for l */
        .fill   0x64, 1, 0
        .byte   0x80
        .fill   0x6C - 0x65, 1, 0
        .byte   0x6C, 0x80
        .fill   0x72 - 0x6E, 1, 0
        .byte   0x65, 0x80, 0x80, 0, 0x65
        .fill   0x100 - 0x77, 1, 0
DATA k3n2_shuf                                  /* tbl windows for L(short): 16 x 0xFF, 0..15, 16 x 0xFF */
        .fill   16, 1, 0xFF
        .byte   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        .fill   16, 1, 0xFF
DATA k3n2_plut                                  /* the page pass's nibble tables: class lo / hi -> slot bits */
#if K3_DSV3                                     /* 2 AB 3 AH 5 N 6 WS 7 P */
        .byte   0x80, 0x0C, 0x20, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        .byte   0xE4, 0, 0, 0, 0, 0, 0, 0, 0x08, 0, 0, 0, 0, 0, 0, 0
#elif K3_O200K                                  /* 0 AU 1 AW 2 AB 3 AH 4 AM 5 N 6 WS 7 P (K3O2_SLOT's sets) */
        .byte   0x80, 0x0A, 0x20, 0x40, 0, 0, 0, 0, 0x10, 0x0D, 0, 0, 0, 0, 0, 0
        .byte   0xE1, 0x16, 0xE1, 0x16, 0, 0, 0, 0, 0x08, 0x08, 0x08, 0x08, 0, 0, 0, 0
#else                                           /* 0 L 5 N 6 WS 7 P, no FOLD_S (CLASS_CP's) */
        .byte   0x80, 0x01, 0x20, 0x40, 0, 0, 0, 0, 0x80, 0x01, 0x20, 0x40, 0, 0, 0, 0
        .byte   0xE1, 0xE1, 0xE1, 0xE1, 0, 0, 0, 0, 0xE1, 0xE1, 0xE1, 0xE1, 0, 0, 0, 0
#endif
#if K3_O200K
DATA k3o2_slot                                  /* [class]: its slot in the atom pass */
        .set    k3o2_c, 0
        .rept   256
        K3O2_SLOT k3o2_c
        .set    k3o2_c, k3o2_c + 1
        .endr
#endif
        TEXT
/* CLASS_CP (cl100k): w10 = code point -> its base (FOLD_S: L(na_give_up)); flags from cmp w10, #TOKS_C_L */
.macro CLASS_CP
        lsr     w11, w10, #8
        ldrh    w11, [x26, x11, lsl #1]
        and     w10, w10, #0xFF
        add     w10, w10, w11, lsl #8
        ldrb    w10, [x27, x10]
        tbnz    w10, #6, L(na_give_up)
        and     w10, w10, #TOKS_C_BASE_MASK
        cmp     w10, #TOKS_C_L
.endm

/* CLASS_SLOT (o200k): w10 = code point -> x1 = its class's slot (FR + S_x; x0 = the slot table); a class
 * that cuts the block goes to L(na_give_up) */
.macro CLASS_SLOT
        lsr     w11, w10, #8
        ldrh    w11, [x26, x11, lsl #1]
        and     w10, w10, #0xFF
        add     w10, w10, w11, lsl #8
        ldrb    w10, [x27, x10]
        ldrb    w1, [x0, x10]
        cmp     w1, #0xFF
        b.eq    L(na_give_up)
.endm

/* F_OTHER next (cl100k): a WS atom (bytes x6) joins W, an N atom joins N; any other class goes to L(na_give_up) */
.macro F_OTHER next
        cmp     w10, #TOKS_C_WS
        b.eq    77f
        cmp     w10, #TOKS_C_N
        b.ne    L(na_give_up)
        ldr     x11, SLOT(S_NNA)
        orr     x11, x11, x6
        str     x11, SLOT(S_NNA)
        b       \next
77:     ldr     x11, SLOT(S_WNA)
        orr     x11, x11, x6
        str     x11, SLOT(S_WNA)
        b       \next
.endm

/* CJKT d, t: d = 0xFF where t (cp >> 8 of a would-be 3-byte lead) is in 4E-A3 or AC-D6 (dsv3: 4E-9E).
 * Clobbers v26-v28 */
.macro CJKT d, t
#if K3_DSV3
        movi    v26.16b, #0x4E
        sub     v27.16b, \t\().16b, v26.16b
        movi    v26.16b, #0x51
        cmhi    \d\().16b, v26.16b, v27.16b
#else
        movi    v26.16b, #0x4E
        sub     v27.16b, \t\().16b, v26.16b
        movi    v26.16b, #0x56
        cmhi    v27.16b, v26.16b, v27.16b
        movi    v26.16b, #0xAC
        sub     v28.16b, \t\().16b, v26.16b
        movi    v26.16b, #0x2B
        cmhi    v28.16b, v26.16b, v28.16b
        orr     \d\().16b, v27.16b, v28.16b
#endif
.endm

/* CJKL d, b, n: d = CJKT((b << 4) | ((n >> 2) & 0x0F)) (v29 = 0x0F) */
.macro CJKL d, b, n
        shl     v25.16b, \b\().16b, #4
        ushr    v26.16b, \n\().16b, #2
        and     v26.16b, v26.16b, v29.16b
        orr     v25.16b, v25.16b, v26.16b
        CJKT    \d, v25
.endm

/* PAGES, the page pass (kernels.md §3.1; out of line, entered without CJK leads): when a dense block keeps 10+
 * leads for the per-atom loops and its first is a 2-byte or E0 / E1 lead (the scripts whose letters share few
 * 64-code-point pages; kana, symbols, hangul spread), up to 4 times the atoms of the lowest lead's page (its
 * 2-byte lead, or 3-byte lead and first continuation), when 4+ and not E2-EF, take their classes from that page
 * of cls_stage2 with one tbl over the block's last bytes; k3n2_plut's two nibble tables give each class its slot bits (0: the class cuts,
 * L(na_give_up)), a mask per slot present. Leads left take the loops. In: x7 / x8 the 2- / 3-byte leads, x3 the
 * window shift. Clobbers x9-x17 x25, v7 v16-v31 (the code table and v24 are reloaded). */
.macro PGLANE j, m, c                           /* v\c |= the page's class at each of its atoms' last bytes */
        and     v26.16b, v\j\().16b, v24.16b
        orn     v26.16b, v26.16b, v\m\().16b
        tbl     v26.16b, {v16.16b, v17.16b, v18.16b, v19.16b}, v26.16b
        orr     v\c\().16b, v\c\().16b, v26.16b
.endm
.macro PGCODE c                                 /* v\c: class -> slot bits */
        and     v27.16b, v\c\().16b, v24.16b
        ushr    v28.16b, v\c\().16b, #4
        tbl     v27.16b, {v25.16b}, v27.16b
        tbl     v28.16b, {v26.16b}, v28.16b
        and     v\c\().16b, v27.16b, v28.16b
.endm
.macro PGSLOT k, slot                           /* the atoms whose classes have slot bit k: into the slot (-1: x4) */
        tbz     w9, #\k, 81f
        movi    v31.16b, #(1 << \k)
        cmtst   v27.16b, v20.16b, v31.16b
        cmtst   v28.16b, v21.16b, v31.16b
        cmtst   v29.16b, v22.16b, v31.16b
        cmtst   v30.16b, v23.16b, v31.16b
        MM      x10, 27, 28, 29, 30
        lsr     x10, x10, x3
        and     x10, x10, x16                   /* their last bytes */
        and     x11, x10, x17
        orr     x10, x10, x10, lsr #1
        orr     x10, x10, x11, lsr #2           /* their bytes */
  .if \slot < 0
        orr     x4, x4, x10
  .else
        ldr     x11, SLOT(\slot)
        orr     x11, x11, x10
        str     x11, SLOT(\slot)
  .endif
81:
.endm
.macro PAGES
        orr     x9, x7, x8
        cbz     x9, 79f
        TZ      x10, x9
        add     x11, x20, x22
        ldrb    w11, [x11, x10]
        cmp     w11, #0xE2
        b.hs    79f                             /* the first lead's page is a spread one: the loops */
        fmov    d25, x9
        cnt     v25.8b, v25.8b
        addv    b25, v25.8b
        fmov    w10, s25
        cmp     w10, #10
        b.lo    79f                             /* under 10 leads: the loops */
        lsl     x25, x7, #1
        orr     x25, x25, x8, lsl #2            /* the leads' atoms' last bytes */
        lsl     x17, x8, #2                     /* ... of the 3-byte ones */
        mov     x16, #0                         /* the pages' last bytes */
        mov     w15, #4                         /* pages left */
        mov     w12, #0                         /* 1: give up */
        movi    v31.16b, #0
        ext     v7.16b, v31.16b, v3.16b, #15    /* ld4 lane 0's byte before (lane 3's, one lane on) */
        ext     v31.16b, v31.16b, v2.16b, #15   /* and two before */
71:     orr     x9, x7, x8
        cbz     x9, 75f
        TZ      x10, x9
        add     x11, x20, x22
        add     x11, x11, x10
        ldrb    w13, [x11, #1]                  /* b1 */
        ldrb    w11, [x11]                      /* b0 */
        cmp     w11, #0xE2
        b.hs    75f                             /* a spread page: the loops take the rest */
        dup     v25.16b, w11
        lsr     x14, x7, x10
        tbz     x14, #0, 72f
        ubfx    w14, w11, #2, #3                /* a 2-byte page: cp >> 8 = b0 >> 2 & 7, (cp >> 6) & 3 = b0 & 3 */
        ldrh    w14, [x26, x14, lsl #1]
        and     w11, w11, #3
        orr     w14, w11, w14, lsl #2
        cmeq    v27.16b, v7.16b, v25.16b
        cmeq    v28.16b, v0.16b, v25.16b
        cmeq    v29.16b, v1.16b, v25.16b
        cmeq    v30.16b, v2.16b, v25.16b
        b       73f
72:     ubfiz   w14, w11, #6, #4
        bfxil   w14, w13, #0, #6                /* a 3-byte page: cp >> 6 */
        cmp     w14, #0x20
        b.lo    74f                             /* E0 80-9F: overlong */
        lsr     w11, w14, #5
        cmp     w11, #0x1B
        b.eq    74f                             /* ED A0-BF: surrogate */
        lsr     w11, w14, #2
        ldrh    w11, [x26, x11, lsl #1]
        and     w14, w14, #3
        orr     w14, w14, w11, lsl #2
        dup     v26.16b, w13
        cmeq    v30.16b, v1.16b, v25.16b        /* b0 two bytes before, b1 one */
        cmeq    v29.16b, v2.16b, v26.16b
        and     v30.16b, v30.16b, v29.16b
        cmeq    v29.16b, v0.16b, v25.16b
        cmeq    v28.16b, v1.16b, v26.16b
        and     v29.16b, v29.16b, v28.16b
        cmeq    v28.16b, v7.16b, v25.16b
        cmeq    v27.16b, v0.16b, v26.16b
        and     v28.16b, v28.16b, v27.16b
        cmeq    v27.16b, v31.16b, v25.16b
        cmeq    v25.16b, v7.16b, v26.16b
        and     v27.16b, v27.16b, v25.16b
73:     mov     v25.16b, v28.16b
        mov     v26.16b, v30.16b
        MM      x10, 27, 25, 29, 26
        lsr     x10, x10, x3
        and     x10, x10, x25                   /* the page's last bytes */
        fmov    d26, x10
        cnt     v26.8b, v26.8b
        addv    b26, v26.8b
        fmov    w11, s26
        cmp     w11, #4
        b.lo    75f                             /* under 4 atoms: the loops take the rest */
        cbnz    x16, 77f
        movi    v20.16b, #0                     /* the first page: the classes at the pages' last bytes */
        movi    v21.16b, #0
        movi    v22.16b, #0
        movi    v23.16b, #0
        movi    v24.16b, #0x3F
77:     add     x14, x27, x14, lsl #6
        ld1     {v16.16b, v17.16b, v18.16b, v19.16b}, [x14]   /* the page's 64 classes */
        PGLANE  0, 27, 20
        PGLANE  1, 28, 21
        PGLANE  2, 29, 22
        PGLANE  3, 30, 23
        orr     x16, x16, x10
        bic     x7, x7, x10, lsr #1
        bic     x8, x8, x10, lsr #2
        subs    w15, w15, #1
        b.ne    71b
75:     cbz     x16, 79f                        /* no page taken: v16-v24 untouched */
        ADRP_ADD x10, SYM(k3n2_plut)
        ld1     {v25.16b, v26.16b}, [x10]
        movi    v24.16b, #0x0F
        PGCODE  20
        PGCODE  21
        PGCODE  22
        PGCODE  23
        cmeq    v27.16b, v20.16b, #0
        cmeq    v28.16b, v21.16b, #0
        cmeq    v29.16b, v22.16b, #0
        cmeq    v30.16b, v23.16b, #0
        MM      x10, 27, 28, 29, 30
        lsr     x10, x10, x3
        tst     x10, x16
        b.ne    74f                             /* a class the blocks do not take */
        orr     v27.16b, v20.16b, v21.16b
        orr     v27.16b, v27.16b, v22.16b
        orr     v27.16b, v27.16b, v23.16b
        ext     v28.16b, v27.16b, v27.16b, #8
        orr     v27.16b, v27.16b, v28.16b
        fmov    x9, d27
        orr     x9, x9, x9, lsr #32
        orr     x9, x9, x9, lsr #16
        orr     x9, x9, x9, lsr #8              /* the slot bits present */
#if K3_O200K
        PGSLOT  0, S_AU
        PGSLOT  1, S_AW
        PGSLOT  2, S_AB
        PGSLOT  3, S_AH
        PGSLOT  4, S_AM
#else
        PGSLOT  0, -1
#endif
        PGSLOT  5, S_NNA
        PGSLOT  6, S_WNA
        b       76f
74:     mov     w12, #1
        cbz     x16, L(na_give_up)              /* v16-v24 untouched */
76:     ADRP_ADD x10, SYM(k3n2_code)
        ld1     {v16.16b, v17.16b, v18.16b, v19.16b}, [x10], #64
        ld1     {v20.16b, v21.16b, v22.16b, v23.16b}, [x10]
        movi    v24.16b, #0x40
        cbnz    w12, L(na_give_up)
79:     b       L(pg_ret)
.endm

/* SPILL: the masks the atom pass does not touch (N W NL ' ' ') wait in v4-v6 */
.macro SPILL
        mov     v4.d[0], x5
        mov     v4.d[1], x6
        mov     v5.d[0], x7
        mov     v5.d[1], x8
        mov     v6.d[0], x9
.endm

FUNC K3_FN
        PROLOGUE 5, 0, LOCALS_N
        mov     x19, x1                         /* a */
        ldr     x20, [x19, #K3_TEXT]
        ldr     x21, [x19, #K3_LEN]
        ldr     x22, [x19, #K3_POS]
        ldr     x23, [x19, #K3_ENDS]
        str     x23, SLOT(S_ENDS0)
        ldr     x24, [x19, #K3_CAP]
        add     x24, x23, x24, lsl #2
#if K3_DSV3
        ldr     w28, [x0, #TT_FLAGS]
        lsl     x28, x28, #32
        orr     x28, x28, #TOKS_TP_NO_SLASH     /* dsv3.md §4: o200k's variant without contractions */
#else
        ldr     x28, [x0, #TT_TMPL_PARAMS]      /* tmpl_params, flags (adjacent u32) */
#endif
        ldr     x26, [x0, #TT_CLS_STAGE1]
        ldr     x27, [x0, #TT_CLS_STAGE2]
        str     xzr, SLOT(S_UNTIL)
#if !K3_O200K
        str     xzr, SLOT(S_LF)                 /* C stays 0 without NL_CUT */
#endif
        mov     x0, x21
        add     x1, x22, #4
        cmp     x1, x21
        b.hi    L(slow)                         /* under 4 bytes: the twin's (kernels.h checks it first) */
        ADRP_ADD x10, SYM(k3n2_code)
        ld1     {v16.16b, v17.16b, v18.16b, v19.16b}, [x10], #64
        ld1     {v20.16b, v21.16b, v22.16b, v23.16b}, [x10]
        movi    v24.16b, #0x40
L(main):
        cmp     x22, x21
        b.hs    L(exit)
        cmp     x23, x24
        b.hs    L(exit)
        ldr     x0, SLOT(S_UNTIL)
        cmp     x22, x0
        b.lo    L(slow)                         /* the pieces through a cut atom */

/* ---- a block at pos -------------------------------------------------------------------------------- */
L(block):
        sub     x0, x21, x22                    /* rem */
        add     x8, x20, x22                    /* window */
        mov     x3, #0                          /* masks >> x3 */
        mov     x2, #64                         /* e */
        cmp     x0, #64
        b.hs    1f
        mov     x2, x0
        cmp     x21, #64
        b.lo    L(short)
        add     x8, x20, x21
        sub     x8, x8, #64                     /* the segment's last 64 bytes, masks shifted down */
        mov     x3, #64
        sub     x3, x3, x0
1:      ld4     {v0.16b, v1.16b, v2.16b, v3.16b}, [x8]
L(classify):
        CLASSIFY
        cbz     x3, 2f
        lsr     x0, x0, x3
        lsr     x4, x4, x3
        lsr     x5, x5, x3
        lsr     x6, x6, x3
        lsr     x7, x7, x3
        lsr     x8, x8, x3
        lsr     x9, x9, x3
#if K3_O200K
        ldr     x1, SLOT(S_AU)
        lsr     x1, x1, x3
        str     x1, SLOT(S_AU)
        ldr     x1, SLOT(S_AW)
        lsr     x1, x1, x3
        str     x1, SLOT(S_AW)
#endif
#if K3_DSV3
        ldr     x1, SLOT(S_DM)
        lsr     x1, x1, x3
        str     x1, SLOT(S_DM)
        ldr     x1, SLOT(S_DX)
        lsr     x1, x1, x3
        str     x1, SLOT(S_DX)
#elif !K3_O200K
        ldr     x1, SLOT(S_LF)
        lsr     x1, x1, x3
        str     x1, SLOT(S_LF)
#endif
2:      str     xzr, SLOT(S_CONT)
#if K3_O200K
        str     xzr, SLOT(S_AB)
        str     xzr, SLOT(S_AM)
        str     xzr, SLOT(S_AH)
#endif
        mov     x1, x2                          /* the cut: none */
        cbnz    x0, L(blk_na)
L(blk_cut):                                     /* x1: the first atom the blocks leave to the twin */
#if K3_DSV3
        ldr     x0, SLOT(S_DX)
        TZ      x0, x0
        cmp     x0, x1
        csel    x1, x0, x1, lo                  /* an ascii X atom (D7's gaps are the twin's) */
#endif
        ldr     x0, SLOT(S_UNTIL)
        add     x3, x22, x1
        add     x3, x3, #1
        cmp     x1, x2
        csel    x2, x1, x2, lo                  /* e = the cut */
        csel    x0, x3, x0, lo                  /* and the twin's through the piece holding it */
        str     x0, SLOT(S_UNTIL)
        cbz     x2, L(stuck)                    /* the block starts with that atom */
        str     x2, SLOT(S_E)
        mov     x0, #-1
        BZHI    x0, x0, x2, x10                 /* V */
        and     x7, x7, x0
        bic     x1, x0, x6
#if K3_DSV3
        bic     x1, x1, x5
        ldr     x10, SLOT(S_AH)
        bic     x1, x1, x10                     /* D5: an N or CJK atom ends the region, no give-back before it */
#elif !K3_O200K
        tst     x28, #TOKS_TP_DIGIT_CUT
        csel    x10, x5, xzr, ne
        bic     x1, x1, x10                     /* A8: an N atom ends the regex's input, no give-back before it */
#endif
        lsr     x1, x1, #1
        str     x1, SLOT(S_NWN)                 /* a non-W byte follows */
#if K3_O200K
        /* o200k.md §3 over the block (k3_avx2.inc's terms). ' and N wait in S_NNA and S_WNA. */
        str     x9, SLOT(S_NNA)
        str     x5, SLOT(S_WNA)
        ldr     x3, SLOT(S_AU)
        and     x3, x3, x0
        str     x3, SLOT(S_AU)                  /* U */
        ldr     x1, SLOT(S_AW)
        and     x1, x1, x0
        str     x1, SLOT(S_AW)                  /* W */
        orr     x3, x3, x1
        ldr     x1, SLOT(S_AB)
        orr     x3, x3, x1
        and     x3, x3, x0                      /* letters (base L, Han aside) */
#if K3_DSV3
        ldr     x1, SLOT(S_AU)
        ldr     x10, SLOT(S_AW)
        orr     x1, x1, x10
        str     x1, SLOT(S_DA)                  /* the ascii letters (D1) */
        str     xzr, SLOT(S_AU)
        str     xzr, SLOT(S_AW)                 /* no case groups: a letter run is whole */
#endif
        ldr     x1, SLOT(S_AH)
        and     x1, x1, x0
        str     x1, SLOT(S_AH)                  /* H */
        orr     x1, x1, x3
        orr     x1, x1, x5
        orr     x1, x1, x6
        bic     x2, x0, x1                      /* P */
        tst     x28, #TOKS_TP_NO_SLASH
        csel    x4, xzr, x4, ne
        orr     x4, x4, x7                      /* X: the tail's bytes, CR LF / (NO_SLASH: CR LF) */
        ldr     x1, SLOT(S_AM)
        ands    x1, x1, x0                      /* marks */
        b.ne    L(o2_marks)
        lsl     x1, x2, #1
        and     x1, x1, x7                      /* no marks: a P run that a CR or LF follows is O3's */
L(o2_tail):                                     /* x1: the tails' first bytes; x2 P, x3 C (every case atom) */
        add     x0, x4, x1
        bic     x0, x4, x0
        orr     x0, x0, x1                      /* the tails: the X run from each */
        bic     x2, x2, x0                      /* P* */
        bic     x6, x6, x0                      /* W* */
        str     x3, SLOT(S_AM)                  /* C */
        orr     x1, x2, x8
        lsl     x1, x1, #1
        bic     x4, x2, x1                      /* P* run starts not after ' ' */
        ldr     x5, SLOT(S_CONT)
        lsl     x0, x2, #1
        bic     x0, x0, x5
        orr     x0, x0, x8, lsl #1
        and     x0, x0, x2
        add     x0, x0, x2
        bic     x0, x0, x2                      /* E */
        bic     x1, x6, x7                      /* S */
#if K3_DSV3
        ldr     x10, SLOT(S_DM)
        and     x10, x10, x2
        lsl     x10, x10, #1
        ldr     x11, SLOT(S_DA)
        and     x10, x10, x11                   /* D1: an ascii P atom before an ascii letter */
        lsl     x1, x1, #1                      /* D2: WS before a letter (X atoms cut) */
        orr     x1, x1, x10
        bic     x1, x1, x0                      /* pre */
        bic     x10, x10, x0                    /* the D1 runs' first letters */
        add     x10, x10, x11
        bic     x10, x10, x11                   /* the byte after each D1 run's ascii letters */
        bic     x11, x3, x11
        and     x10, x10, x11
        orr     x4, x4, x10                     /* a non-ascii letter there starts a piece (D1 takes A-Z a-z) */
#else
        orr     x1, x1, x2
        lsl     x1, x1, #1
        bic     x1, x1, x0                      /* pre */
#endif
        orr     x1, x1, x3, lsl #1
        bic     x0, x3, x1
        orr     x4, x4, x0                      /* case run starts no prefix takes */
        orr     x4, x4, #1                      /* B */
        ldr     x5, SLOT(S_AU)
        ldr     x0, SLOT(S_AW)
        orr     x1, x5, x0
        bic     x1, x3, x1                      /* B' */
        lsl     x0, x0, #1
        and     x2, x0, x1
        add     x2, x2, x1
        bic     x2, x2, x1                      /* the byte after each B' run that follows a W */
        bic     x0, x0, x1
        orr     x0, x0, x2                      /* G */
        and     x0, x0, x5
        orr     x4, x4, x0                      /* U & G: the two-state walk's boundaries */
        lsl     x1, x1, #1
        and     x1, x1, x5
        bics    x1, x1, x0                      /* an upper-only run after B', not in G: L1's backtrack */
        b.ne    L(o2_bt)
L(o2_btd):
        ldr     x0, SLOT(S_AH)
        bic     x0, x0, x0, lsl #1
        orr     x4, x4, x0                      /* Han run starts (O0) */
        ldr     x5, SLOT(S_WNA)                 /* N */
        bic     x3, x6, x7                      /* S */
        b       L(digits)
L(o2_bt):                                       /* x1: the candidates; x5 U, x4 B */
        ldr     x2, SLOT(S_AM)                  /* C */
        ldr     x9, SLOT(S_E)
1:      TZ      x3, x1
        BLSR    x1, x1, x10
        lsr     x0, x5, x3
        mvn     x0, x0
        TZ      x0, x0
        add     x0, x0, x3                      /* the end of x's upper-only run */
        cmp     x0, x9
        b.lo    2f
        add     x0, x22, x9
        cmp     x0, x21
        b.ne    3f                              /* it reaches e: the next block decides */
        b       4f
2:      lsr     x10, x2, x0
        tbnz    x10, #0, 3f                     /* the case run goes on */
4:      mov     x10, #1
        lsl     x10, x10, x3
        orr     x4, x4, x10
3:      cbnz    x1, 1b
        b       L(o2_btd)
L(o2_marks):                                    /* x1: the marks; x2 P, x4 X, x3 letters. A mark is a case atom
                                                 * unless O3 took its P run first */
        str     x1, SLOT(S_AM)
        str     x2, SLOT(S_P1)                  /* P */
        str     xzr, SLOT(S_TAILS)              /* the '/' last round's tails took */
L(o2_mloop):
        orr     x0, x3, x1
        ldr     x5, SLOT(S_CONT)
        orr     x0, x0, x5
        lsr     x0, x0, #1
        bic     x0, x2, x0
        bic     x0, x0, x1                      /* the last bytes of non-mark P atoms no case atom follows */
        .rept   3
        and     x9, x0, x5
        orr     x0, x0, x9, lsr #1              /* whole atoms */
        .endr
        lsl     x9, x8, #1
        and     x9, x9, x2
        bic     x9, x9, x1
        orr     x0, x0, x9                      /* and a non-mark P atom after ' ': O3 starts */
        add     x9, x2, x0
        bic     x9, x2, x9
        orr     x0, x0, x9                      /* Q: each P run from its first O3 start on */
        bic     x1, x1, x0                      /* the marks before Q: case atoms */
        lsl     x9, x0, #1
        and     x9, x9, x7                      /* a CR or LF right after Q starts a tail */
        add     x0, x4, x9
        bic     x0, x4, x0
        orr     x0, x0, x9
        ldr     x10, SLOT(S_P1)
        and     x0, x0, x10                     /* the '/' the tails take */
        ldr     x11, SLOT(S_TAILS)
        cmp     x0, x11
        b.eq    1f                              /* as last round: the rounds settle from the left */
        str     x0, SLOT(S_TAILS)
        bic     x2, x10, x0                     /* P without them: Q again */
        ldr     x1, SLOT(S_AM)
        b       L(o2_mloop)
1:      orr     x3, x3, x1                      /* C */
        bic     x2, x2, x1
        mov     x1, x9
        b       L(o2_tail)
#else
        tst     x28, #TOKS_TP_NL_CUT
        b.eq    1f
        /* A9: the cuts C, where a (\r?\n)+ run starts: an LF not after a CR / LF, a CR before an LF not after an LF */
        ldr     x10, SLOT(S_LF)                 /* LF */
        bic     x11, x10, x7, lsl #1            /* LF & ~(NL << 1) */
        bic     x12, x7, x10
        bic     x12, x12, x10, lsl #1           /* CR & ~(LF << 1) */
        and     x12, x12, x10, lsr #1           /* ... & (LF >> 1) */
        orr     x11, x11, x12
        and     x11, x11, x0
        str     x11, SLOT(S_LF)                 /* C (in play) */
1:      orr     x2, x4, x5
        orr     x2, x2, x6
        bic     x0, x0, x2                      /* P */
        orr     x2, x0, x8
        lsl     x2, x2, #1
        bic     x2, x0, x2                      /* Pst: P run starts not after ' ' */
        and     x9, x9, x2
        str     x9, SLOT(S_CAND)                /* ' at piece starts */
        bic     x3, x6, x7                      /* S */
        /* letters: L run starts, minus those an admitted prefix atom before them takes */
        tst     x28, #TOKS_TP_LPREFIX_ANY
        b.eq    5f
        /* a P atom prefixes when its P run is that one atom and not after ' ' */
        ldr     x1, SLOT(S_CONT)
        lsl     x9, x0, #1
        bic     x1, x9, x1
        orr     x1, x1, x8, lsl #1
        and     x1, x1, x0
        add     x1, x1, x0
        bic     x1, x1, x0                      /* E */
        orr     x9, x3, x0
        lsl     x9, x9, #1
        bic     x9, x9, x1                      /* pre = ((S | P) << 1) & ~E */
        b       6f
5:      lsl     x9, x8, #1                      /* pre = SP << 1 */
6:      orr     x9, x9, x4, lsl #1
        bic     x4, x4, x9
        orr     x4, x4, x2
        orr     x4, x4, #1                      /* B */
        ldr     x10, SLOT(S_LF)
        orr     x4, x4, x10                     /* A9: every cut starts a piece */
        tst     x28, #TOKS_TP_PUNCT_NL
        b.eq    L(digits)
        bic     x10, x7, x10                    /* the CR/LF bytes but the cuts (A9: a tail stops before one) */
        lsl     x0, x0, #1
        and     x0, x0, x10                     /* CR/LF right after P */
        add     x0, x0, x10
        bic     x0, x10, x0                     /* the CR/LF runs they start (absorbed by A4) */
        bic     x6, x6, x0                      /* W* */
#endif
L(digits):                                      /* digits */
        bic     x0, x5, x5, lsl #1              /* N run starts */
        and     x2, x28, #TOKS_TP_DIGITS_MASK
        cbz     x2, L(d13)
        cmp     x2, #TOKS_TP_DIGITS_1
        b.eq    L(d1)
        cmp     x2, #TOKS_TP_DIGITS_SP_RUN
        b.eq    L(dsp)
        orr     x4, x4, x0
        b       L(ddone)
L(d1):  ldr     x2, SLOT(S_CONT)
        bic     x2, x5, x2                      /* each N atom's first byte */
        orr     x4, x4, x2
        b       L(ddone)
L(dsp): bic     x0, x0, x8, lsl #1
        orr     x4, x4, x0
        b       L(ddone)
L(d13): orr     x4, x4, x0
        ldr     x2, SLOT(S_CONT)
        tst     x2, x5
        b.ne    L(d13a)                         /* an N atom of 2+ bytes: count atoms */
        and     x2, x5, x5, lsl #1
        and     x2, x2, x2, lsl #1              /* N with two N before it */
7:      lsl     x0, x0, #3
        ands    x0, x0, x2
        b.eq    L(ddone)
        orr     x4, x4, x0
        b       7b
L(d13a):                                        /* each run's 1st, 4th, 7th ... N atom (x0: run starts) */
        bic     x2, x5, x2                      /* N atom starts */
4:      TZ      x9, x2
        BLSR    x2, x2, x10
        lsr     x10, x0, x9
        tbnz    x10, #0, 5f
        subs    w1, w1, #1
        b.ne    6f
5:      mov     x10, #1
        lsl     x10, x10, x9
        orr     x4, x4, x10
        mov     w1, #3
6:      cbnz    x2, 4b
L(ddone):
        ldr     x1, SLOT(S_NWN)
        bic     x0, x6, x6, lsl #1
        orr     x4, x4, x0                      /* W* run starts */
#if !K3_O200K
        tst     x28, #TOKS_TP_WS_NL
        b.eq    L(w_plain)
#endif
        lsl     x0, x7, #1
        and     x0, x0, x3                      /* C: S right after NL */
        add     x2, x3, x0
        bic     x2, x2, x3                      /* the byte after each such S run */
#if !K3_O200K
        ldr     x10, SLOT(S_LF)
        bic     x2, x2, x10                     /* ... a NL but no cut (A9: then that S run is its stretch's tail) */
#endif
        ands    x2, x2, x7
        b.ne    L(w_fix)
L(w_c): orr     x4, x4, x0
        and     x1, x1, x3                      /* A6 give-back: the last atom of the run */
        b       L(w_gb)
#if !K3_O200K
L(w_plain):
        and     x1, x1, x6
#endif
L(w_gb):                                        /* (its last byte, moved back to the atom's first) */
        ldr     x2, SLOT(S_CONT)
        cbz     x2, 9f
        .rept   3
        and     x0, x1, x2
        bic     x1, x1, x2
        orr     x1, x1, x0, lsr #1
        .endr
9:      orr     x4, x4, x1
        b       L(w_done)
L(w_fix):                                       /* drop the C bit of each S run that ends at a NL */
        mvn     x8, x3
8:      TZ      x5, x2
        BLSR    x2, x2, x10
        BZHI    x9, x8, x5, x10
        clz     x9, x9
        neg     x9, x9
        add     x9, x9, #64
        mov     x10, #1
        lsl     x10, x10, x9
        bic     x0, x0, x10
        cbnz    x2, 8b
        b       L(w_c)
L(w_done):
#if K3_O200K
        /* the suffix K (CONTR_CI): a ' right after a case atom ends that letter piece, and K takes ' and one
         * or two letters with it: their starts go, and a case atom right after it starts a piece. A suffix
         * the block cannot decide stops the decided range at its ' (S_NNA: the limit, 64 none) */
        ldr     x0, SLOT(S_NNA)                 /* ' */
        ldr     x1, SLOT(S_AM)                  /* C */
        and     x0, x0, x1, lsl #1
        mov     x10, #64
        str     x10, SLOT(S_NNA)
        cbz     x0, L(c_done)
        tst     x28, #TOKS_TP_CONTR_MASK
        b.eq    L(c_done)
        ldr     x2, SLOT(S_E)
        ADRP_ADD x3, SYM(k3n2_contr)
        str     xzr, SLOT(S_WNA)                /* the last suffix's end */
L(k_loop):
        TZ      x1, x0                          /* a */
        BLSR    x0, x0, x10
        ldr     x10, SLOT(S_WNA)
        cmp     x1, x10
        b.eq    L(k_next)                       /* after a suffix: that ' starts a piece (K comes once) */
        add     x8, x22, x1
        add     x8, x8, #1                      /* c' */
        cmp     x8, x21
        b.hs    L(k_next)
        ldrb    w9, [x20, x8]
        tbnz    w9, #7, L(k_na)
        orr     w9, w9, #0x20
        ldrb    w9, [x3, x9]                    /* 0x80: s t m d; e l: the second letter of 're 've 'll */
        cbz     w9, L(k_next)
        add     x7, x1, #2                      /* z */
        tbnz    w9, #7, L(k_hit)
        add     x8, x22, x7
        cmp     x8, x21
        b.hs    L(k_next)
        ldrb    w5, [x20, x8]
        orr     w5, w5, #0x20
        cmp     w5, w9
        b.ne    L(k_next)
        add     x7, x7, #1
L(k_hit):
        str     x7, SLOT(S_WNA)
        sub     x8, x7, x1
        mov     x5, #1
        lsl     x5, x5, x8
        sub     x5, x5, #1
        lsl     x5, x5, x1
        bic     x4, x4, x5                      /* no start at ' or the suffix's letters */
        cmp     x7, x2
        b.hs    L(k_far)
        ldr     x8, SLOT(S_AM)
        lsr     x8, x8, x7
        tbz     x8, #0, L(k_next)               /* z in no case group: a start already */
        ldr     x8, SLOT(S_AU)
        ldr     x5, SLOT(S_AW)
        orr     x8, x8, x5
        lsr     x8, x8, x7
        tbz     x8, #0, L(k_defer)              /* both groups: the walk from z is the next block's */
        mov     x8, #1
        lsl     x8, x8, x7
        orr     x4, x4, x8                      /* upper- or lower-only: the walk restarts at z */
        b       L(k_next)
L(k_far):                                       /* z at or past e */
        add     x8, x22, x7
        cmp     x8, x21
        b.hs    L(k_next)
        ldrb    w8, [x20, x8]
        tbz     w8, #7, L(k_next)               /* ascii: whatever starts there lies past the decided range */
        b       L(k_defer)
L(k_na):                                        /* c' not ascii: a suffix only with FOLD_S, which cuts blocks */
        add     x8, x1, #1
        cmp     x8, x2
        b.lo    L(k_next)
L(k_defer):
        str     x1, SLOT(S_NNA)                 /* ascending: the first is the limit */
        b       L(c_done)
L(k_next):
        cbnz    x0, L(k_loop)
#else
        /* A1 contractions at the ' piece starts */
        and     x3, x28, #TOKS_TP_CONTR_MASK
        cbz     x3, L(c_done)
        ldr     x0, SLOT(S_CAND)
        cbz     x0, L(c_done)
        mov     w7, #0
        cmp     x3, #TOKS_TP_CONTR_CI
        b.ne    1f
        mov     w7, #0x20                       /* (?i) on ascii */
1:      ldr     x2, SLOT(S_E)
        ADRP_ADD x3, SYM(k3n2_contr)
L(c_loop):
        TZ      x1, x0                          /* i */
        BLSR    x0, x0, x10
        add     x5, x1, #1
        cmp     x5, x2
        b.hs    L(c_next)                       /* ' at e - 1: deferred with its run, or final */
        add     x8, x22, x1
        add     x8, x8, #2                      /* c2's offset */
        sub     x10, x8, #1
        ldrb    w9, [x20, x10]
        orr     w9, w9, w7
        ldrb    w9, [x3, x9]                    /* t: 0x80 one-letter, or the second letter wanted, or 0 */
        mov     w1, #0                          /* c2: 0 past len */
        cmp     x8, x21
        b.hs    2f
        ldrb    w1, [x20, x8]
        orr     w1, w1, w7
2:      cmp     w9, w1
        cset    w8, eq
        tst     w9, #0x7F
        cset    w1, ne
        and     w8, w8, w1                      /* m3: 're 've 'll */
        lsr     w9, w9, #7                      /* m1: 's 't 'm 'd */
        add     w1, w9, w8
        add     w8, w9, w8, lsl #1
        lsl     w8, w8, #1
        lsl     x1, x1, x5                      /* c1 is inside the piece */
        lsl     x8, x8, x5                      /* the start after it (past bit 63: dropped) */
        bic     x4, x4, x1
        orr     x4, x4, x8
L(c_next):
        cbnz    x0, L(c_loop)
#endif
L(c_done):
        ldr     x2, SLOT(S_E)
#if K3_O200K
        ldr     x10, SLOT(S_NNA)
        cmp     x10, #64
        b.lo    L(c_part)                       /* a deferred suffix: the segment's end waits too */
#endif
        add     x0, x22, x2
        cmp     x0, x21
        b.eq    L(final)
L(c_part):
        /* u + 1 = e - max(r - 1, 0), r = the length of the W* run ending at byte e - 1 */
        mov     x1, #64
        sub     x1, x1, x2
        lsl     x0, x6, x1
        mvn     x0, x0
        clz     x0, x0
        subs    x0, x0, #1
        csel    x0, xzr, x0, lo
        sub     x1, x2, x0
#if K3_O200K
        ldr     x0, SLOT(S_NNA)
        cmp     x1, x0
        csel    x1, x0, x1, hi                  /* not past a deferred suffix's ' */
#endif
        BZHI    x0, x4, x1, x10                 /* B in [0, u] */
        clz     x1, x0
        eor     x1, x1, #63                     /* q */
        cbz     x1, L(stuck)
        add     x7, x22, x1                     /* the next block */
        bic     x0, x0, #1
        EMIT
        mov     x22, x7
        b       L(main)
L(final):                                       /* the block ends the segment: B in (0, e), then len */
        mov     x0, #-1
        BZHI    x0, x0, x2, x10
        and     x0, x0, x4
        bic     x0, x0, #1
        EMIT
        cmp     x23, x24
        b.hs    L(exit)
        str     w21, [x23], #4
        mov     x22, x21
        b       L(exit)
/* non-ascii bytes in the block (x0): each atom's class goes to every byte of it; cont marks the bytes after an
 * atom's first. The first atom of a class the template leaves to the twin cuts the block. The masks the pass
 * does not touch wait in v4-v6. */
L(blk_na):
        str     x2, SLOT(S_E)
        fmov    d29, x0
        cnt     v29.8b, v29.8b
        addv    b29, v29.8b
        fmov    w1, s29
        cmp     w1, #8
        b.lo    L(na_sparse)
        /* a dense block under the CJK flag: for every byte, the cp >> 8 it would give as the lead of a 3-byte
         * atom, tested against the CJK ideograph / Yi (4E-A3) and hangul (AC-D6) blocks; S_NWN holds the mask
         * until the leads are known */
        tst     x28, #(K3_TF_CJK << 32)
        b.eq    2f
        movi    v29.16b, #0x0F
        movi    v30.16b, #0
        ext     v30.16b, v0.16b, v30.16b, #1    /* lane 3's next byte: lane 0, one on */
        CJKL    v4, v0, v1
        CJKL    v5, v1, v2
        CJKL    v6, v2, v3
        CJKL    v7, v3, v30
        MM      x1, 4, 5, 6, 7
        lsr     x1, x1, x3
        str     x1, SLOT(S_NWN)
2:      SPILL
        /* fast path: every non-ascii byte is a 2- or 3-byte lead or a continuation byte some lead calls for;
         * then only E0/ED leads need a range check. A lead whose atom would cross e shrinks e to it */
        RANGE   x6, 0x80, 0x40                  /* 80-BF */
        RANGE   x7, 0xC2, 0x1E                  /* C2-DF */
        RANGE   x8, 0xE0, 0x10                  /* E0-EF */
        lsr     x6, x6, x3
        lsr     x7, x7, x3
        lsr     x8, x8, x3
        ldr     x2, SLOT(S_E)
        sub     x1, x2, #1
        mov     x10, #1
        lsl     x10, x10, x1                    /* bit e - 1 */
        orr     x9, x10, x10, lsr #1            /* bits e - 2, e - 1 */
        and     x10, x10, x7
        and     x9, x9, x8
        orr     x10, x10, x9                    /* leads whose atoms would cross e */
        cbz     x10, 1f
        TZ      x2, x10
        str     x2, SLOT(S_E)                   /* e = the first of them */
        BZHI    x0, x0, x2, x10
        BZHI    x6, x6, x2, x10
        BZHI    x7, x7, x2, x10
        BZHI    x8, x8, x2, x10
1:      orr     x1, x7, x8
        lsl     x9, x1, #1
        orr     x9, x9, x8, lsl #2
        cmp     x9, x6
        b.ne    L(na_slow)                      /* the continuation bytes are not exactly the leads' */
        orr     x1, x1, x6
        cmp     x1, x0
        b.ne    L(na_slow)                      /* other non-ascii bytes (4-byte leads, C0 C1 F5-FF) */
        str     x0, SLOT(S_P1)                  /* kept for L(na_give_up) */
        str     x6, SLOT(S_CONT)
        mov     x2, #0                          /* leads of 3-byte letter atoms */
        tst     x28, #(K3_TF_CJK << 32)
        b.eq    L(uni_done)
        ldr     x2, SLOT(S_NWN)
        and     x2, x2, x8                      /* 3-byte leads in those blocks: no lookup */
        bic     x8, x8, x2
L(uni_done):
#if K3_O200K
        lsl     x6, x2, #3
        sub     x6, x6, x2
        ldr     x10, SLOT(S_CJK)
        orr     x10, x10, x6
        str     x10, SLOT(S_CJK)                /* the CJK leads' atoms: letters in both groups (dsv3: Han) */
        ADRP_ADD x0, SYM(k3o2_slot)
        str     xzr, SLOT(S_NNA)
        str     xzr, SLOT(S_WNA)
        cbz     x2, L(pages)                    /* no CJK leads: the page pass (out of line) */
L(pg_ret):
        cbz     x7, L(o2f3s)
        LOOP_ALIGN
L(o2f2):                                        /* each atom's bytes go to its class's slot */
        TZ      x1, x7
        neg     x9, x7
        and     x9, x9, x7
        BLSR    x7, x7, x10
        add     x1, x1, x22
        ldrh    w10, [x20, x1]                  /* b0 | b1 << 8 */
        ubfiz   w6, w10, #6, #5
        ubfx    w10, w10, #8, #6
        orr     w10, w10, w6                    /* cp */
        add     x6, x9, x9, lsl #1
        CLASS_SLOT
        ldr     x11, [sp, x1]
        orr     x11, x11, x6
        str     x11, [sp, x1]
        cbnz    x7, L(o2f2)
L(o2f3s):
        cbz     x8, L(o2f_done)
        LOOP_ALIGN
L(o2f3):
        TZ      x1, x8
        neg     x9, x8
        and     x9, x9, x8
        BLSR    x8, x8, x10
        add     x11, x20, x1
        add     x11, x11, x22
        ldrb    w6, [x11]
        ldrh    w10, [x11, #1]                  /* b1 | b2 << 8 */
        ubfiz   w6, w6, #12, #4
        ubfx    w11, w10, #0, #6
        orr     w6, w6, w11, lsl #6
        ubfx    w10, w10, #8, #6
        orr     w10, w10, w6                    /* cp */
        cmp     w10, #0x800
        b.lo    L(na_give_up)                   /* E0 80-9F: overlong */
        lsr     w11, w10, #11
        cmp     w11, #0x1B
        b.eq    L(na_give_up)                   /* ED A0-BF: surrogate */
        lsl     x6, x9, #3
        sub     x6, x6, x9
        CLASS_SLOT
        ldr     x11, [sp, x1]
        orr     x11, x11, x6
        str     x11, [sp, x1]
        cbnz    x8, L(o2f3)
L(o2f_done):
        ldr     x1, SLOT(S_E)                   /* no cut */
        b       L(na_fin)
#else
        mov     x0, #0                          /* leads of 2-byte letter atoms */
        str     xzr, SLOT(S_NNA)
        str     xzr, SLOT(S_WNA)
        cbz     x2, L(pages)                    /* no CJK leads: the page pass (out of line) */
L(pg_ret):
        cbz     x7, L(f3_start)
        LOOP_ALIGN
L(f2):  TZ      x1, x7
        neg     x9, x7
        and     x9, x9, x7
        BLSR    x7, x7, x10
        add     x1, x1, x22
        ldrh    w10, [x20, x1]                  /* b0 | b1 << 8 */
        ubfiz   w6, w10, #6, #5
        ubfx    w10, w10, #8, #6
        orr     w10, w10, w6                    /* cp */
        CLASS_CP
        b.hi    L(f2_other)
        csel    x6, x9, xzr, eq
        orr     x0, x0, x6
L(f2_next):
        cbnz    x7, L(f2)
L(f3_start):
        cbz     x8, L(f_done)
        LOOP_ALIGN
L(f3):  TZ      x1, x8
        neg     x9, x8
        and     x9, x9, x8
        BLSR    x8, x8, x10
        add     x11, x20, x1
        add     x11, x11, x22
        ldrb    w6, [x11]
        ldrh    w10, [x11, #1]                  /* b1 | b2 << 8 */
        ubfiz   w6, w6, #12, #4
        ubfx    w11, w10, #0, #6
        orr     w6, w6, w11, lsl #6
        ubfx    w10, w10, #8, #6
        orr     w10, w10, w6                    /* cp */
        cmp     w10, #0x800
        b.lo    L(na_give_up)
        lsr     w11, w10, #11
        cmp     w11, #0x1B
        b.eq    L(na_give_up)
        CLASS_CP
        b.hi    L(f3_other)
        csel    x6, x9, xzr, eq
        orr     x2, x2, x6
L(f3_next):
        cbnz    x8, L(f3)
L(f_done):
        add     x6, x0, x0, lsl #1              /* 2-byte letter atoms: lead and lead + 1 */
        orr     x4, x4, x6
        lsl     x6, x2, #3
        sub     x6, x6, x2                      /* 3-byte: lead .. lead + 2 */
        orr     x4, x4, x6
        ldr     x1, SLOT(S_E)                   /* no cut */
        b       L(na_fin)
L(f2_other):                                    /* w10 = base, x9 = the lead's bit */
        add     x6, x9, x9, lsl #1
        F_OTHER L(f2_next)
L(f3_other):
        lsl     x6, x9, #3
        sub     x6, x6, x9
        F_OTHER L(f3_next)
#endif
L(na_give_up):                                  /* a class or form the fast path leaves to the ordered pass */
        ldr     x0, SLOT(S_P1)
        b       L(na_slow)
L(na_sparse):
        SPILL
L(na_slow):
        mov     x7, #0                          /* cont */
        str     xzr, SLOT(S_NNA)
        str     xzr, SLOT(S_WNA)
#if K3_O200K
        ADRP_ADD x11, SYM(k3o2_slot)
        str     x11, SLOT(S_P1)                 /* the slot table */
#endif
L(na_loop):
        TZ      x1, x0                          /* j */
        add     x13, x22, x1
        bl      L(atom)                         /* w14 its class, x15 its length */
        mov     x10, #1
        lsl     x10, x10, x15
        sub     x10, x10, #1
        lsl     x10, x10, x1                    /* the atom's bytes */
#if K3_O200K
        ldr     x11, SLOT(S_P1)
        ldrb    w14, [x11, x14]
        cmp     w14, #0xFF
        b.eq    L(na_cut)
        ldr     x11, [sp, x14]
        orr     x11, x11, x10
        str     x11, [sp, x14]                  /* its slot */
#else
        tbnz    w14, #6, L(na_cut)              /* FOLD_S */
        and     w11, w14, #TOKS_C_BASE_MASK
        cmp     w11, #TOKS_C_L
        b.hi    L(na_other)
        csel    x11, x10, xzr, eq
        orr     x4, x4, x11                     /* L joins L, P stays P */
#endif
L(na_next):
        bic     x0, x0, x10
        mov     x11, #1
        lsl     x11, x11, x1
        bic     x10, x10, x11
        orr     x7, x7, x10
        cbnz    x0, L(na_loop)
        ldr     x1, SLOT(S_E)                   /* no cut */
L(na_cut):
        str     x7, SLOT(S_CONT)
L(na_fin):
        ldr     x2, SLOT(S_E)
        mov     x5, v4.d[0]
        mov     x6, v4.d[1]
        ldr     x10, SLOT(S_NNA)
        orr     x5, x5, x10
        ldr     x10, SLOT(S_WNA)
        orr     x6, x6, x10
        mov     x7, v5.d[0]
        mov     x8, v5.d[1]
        mov     x9, v6.d[0]
        b       L(blk_cut)
#if !K3_O200K
L(na_other):                                    /* WS joins W, N joins N */
        cmp     w11, #TOKS_C_WS
        b.eq    1f
        cmp     w11, #TOKS_C_N
        b.ne    L(na_cut)
        ldr     x11, SLOT(S_NNA)
        orr     x11, x11, x10
        str     x11, SLOT(S_NNA)
        b       L(na_next)
1:      ldr     x11, SLOT(S_WNA)
        orr     x11, x11, x10
        str     x11, SLOT(S_WNA)
        b       L(na_next)
#endif

/* a segment shorter than 64 bytes (>= 4: shorter remainders are the twin's): v0-v3 = text[0, len) zero-padded,
 * from loads inside it: full 16-byte chunks, the last partial one shifted out of the 16 bytes ending at len
 * by tbl (k3n2_shuf), then de-interleaved into ld4 lanes. The masks are shifted down by pos. */
L(short):
        mov     x3, x22
        movi    v4.16b, #0
        movi    v5.16b, #0
        movi    v6.16b, #0
        movi    v7.16b, #0
        cmp     x21, #16
        b.lo    3f
        lsr     x11, x21, #4                    /* full chunks */
        ldr     q4, [x20]
        cmp     x11, #2
        b.lo    1f
        ldr     q5, [x20, #16]
        cmp     x11, #3
        b.lo    1f
        ldr     q6, [x20, #32]
1:      ands    x12, x21, #15                   /* the partial chunk's bytes */
        b.eq    4f
        add     x13, x20, x21
        ldur    q28, [x13, #-16]
        ADRP_ADD x13, SYM(k3n2_shuf)
        add     x13, x13, #32
        sub     x13, x13, x12
        ldr     q29, [x13]
        tbl     v28.16b, {v28.16b}, v29.16b     /* the chunk's bytes at its start, zeros after */
        cmp     x11, #2
        b.lo    5f
        b.eq    6f
        mov     v7.16b, v28.16b
        b       4f
5:      mov     v5.16b, v28.16b
        b       4f
6:      mov     v6.16b, v28.16b
        b       4f
3:      add     x13, x20, x21                   /* 8 <= len < 16: text[0, 8) and text[len - 8, len) as d lanes */
        cmp     x21, #8
        b.lo    7f
        ldr     d4, [x20]
        ldur    d28, [x13, #-8]
        mov     v4.d[1], v28.d[0]
        movi    v31.16b, #8                     /* h: the second lane's first byte */
        b       8f
7:      ldr     s4, [x20]                       /* 4 <= len < 8: text[0, 4) and text[len - 4, len) as s lanes */
        ldur    s28, [x13, #-4]
        mov     v4.s[1], v28.s[0]
        movi    v31.16b, #4
8:      ADRP_ADD x13, SYM(k3n2_shuf)
        ldr     q29, [x13, #16]                 /* k = 0..15 */
        cmhs    v26.16b, v29.16b, v31.16b       /* k >= h */
        dup     v30.16b, w21
        cmhs    v27.16b, v29.16b, v30.16b       /* k >= len */
        sub     v30.16b, v31.16b, v30.16b
        add     v30.16b, v30.16b, v31.16b       /* 2h - len */
        and     v30.16b, v30.16b, v26.16b
        add     v29.16b, v29.16b, v30.16b       /* k, or k + 2h - len for k >= h */
        orr     v29.16b, v29.16b, v27.16b       /* 0xFF past len */
        tbl     v4.16b, {v4.16b}, v29.16b
4:      uzp1    v25.16b, v4.16b, v5.16b
        uzp2    v26.16b, v4.16b, v5.16b
        uzp1    v27.16b, v6.16b, v7.16b
        uzp2    v28.16b, v6.16b, v7.16b
        uzp1    v0.16b, v25.16b, v27.16b
        uzp2    v2.16b, v25.16b, v27.16b
        uzp1    v1.16b, v26.16b, v28.16b
        uzp2    v3.16b, v26.16b, v28.16b
        b       L(classify)
L(stuck):                                       /* no start in (0, u]: a piece longer than its block, the twin's */
        add     x0, x22, #1
        b       L(slow)
L(exit):
        mov     x0, #0
L(slow):                                        /* x0: 0, or the end of a stretch the c twin scans from pos */
        str     x0, [x19, #K3_RSV]
        ldr     x1, SLOT(S_ENDS0)
        sub     x0, x23, x1
        lsr     x0, x0, #2                      /* n */
        str     x0, [x19, #K3_N]
        cbz     x0, 1f
        ldur    w2, [x23, #-4]
        str     x2, [x19, #K3_POS]
1:      EPILOGUE 5, 0, LOCALS_N

/* ATOM: the class and length of the atom at text[x13] (x13 < len, a non-ascii byte): w14, x15 (toks_utf8_len's
 * well-formedness, never reading at or past len; an ill-formed first byte is a one-byte P atom; o200k: a Han atom
 * leaves both case groups). Clobbers x12 x16 x17. */
L(atom):
        ldrb    w16, [x20, x13]
        sub     x17, x21, x13                   /* bytes left */
        cmp     w16, #0xC2
        b.lo    L(atom_bad)
        cmp     w16, #0xE0
        b.hs    2f
        cmp     x17, #2
        b.lo    L(atom_bad)
        add     x12, x20, x13
        ldrb    w15, [x12, #1]
        eor     w15, w15, #0x80
        cmp     w15, #0x3F
        b.hi    L(atom_bad)
        ubfiz   w16, w16, #6, #5
        orr     w16, w16, w15
        mov     x15, #2
        b       L(atom_cls)
2:      cmp     w16, #0xF0
        b.hs    3f
        cmp     x17, #3
        b.lo    L(atom_bad)
        add     x12, x20, x13
        ldrb    w14, [x12, #1]
        ldrb    w17, [x12, #2]
        eor     w14, w14, #0x80
        eor     w17, w17, #0x80
        orr     w12, w14, w17
        cmp     w12, #0x3F
        b.hi    L(atom_bad)
        ubfiz   w16, w16, #12, #4
        orr     w16, w16, w14, lsl #6
        orr     w16, w16, w17
        cmp     w16, #0x800
        b.lo    L(atom_bad)                     /* overlong */
        lsr     w17, w16, #11
        cmp     w17, #0x1B
        b.eq    L(atom_bad)                     /* surrogate */
        mov     x15, #3
        b       L(atom_cls)
3:      cmp     w16, #0xF4
        b.hi    L(atom_bad)
        cmp     x17, #4
        b.lo    L(atom_bad)
        add     x12, x20, x13
        ldrb    w14, [x12, #1]
        ldrb    w15, [x12, #2]
        ldrb    w17, [x12, #3]
        eor     w14, w14, #0x80
        eor     w15, w15, #0x80
        eor     w17, w17, #0x80
        orr     w12, w14, w15
        orr     w12, w12, w17
        cmp     w12, #0x3F
        b.hi    L(atom_bad)
        ubfiz   w16, w16, #18, #3
        orr     w16, w16, w14, lsl #12
        orr     w16, w16, w15, lsl #6
        orr     w16, w16, w17
        sub     w17, w16, #0x10, lsl #12
        cmp     w17, #0x100, lsl #12
        b.hs    L(atom_bad)                     /* overlong or past U+10FFFF */
        mov     x15, #4
L(atom_cls):                                    /* w16 = the code point */
        lsr     w14, w16, #8
        ldrh    w14, [x26, x14, lsl #1]
        and     w16, w16, #0xFF
        add     w14, w16, w14, lsl #8
        ldrb    w14, [x27, x14]
#if K3_O200K
        tbz     w14, #7, 4f
        and     w14, w14, #~(TOKS_C_UPPER | TOKS_C_LOWER)  /* o200k.md §6: Han atoms leave both case groups */
4:
#endif
        ret
L(atom_bad):
        mov     w14, #TOKS_C_P
        mov     x15, #1
        ret
L(pages):                                       /* the page pass, out of line */
        PAGES
ENDFUNC K3_FN
