/*
 * src/asm/arm64/k7_spm_neon.S: K7, the sentencepiece-style bpe scan, tier neon (docs/kernels.md §8: contract,
 * algorithm, measurements; the c twin is spm_c.c's toks_k7_spm_c).
 *
 *   uint64_t toks_k7_spm_neon(const toks_spm *s, toks_k7_args *a)     returns the window's word starts
 *
 * Registers: x0 s, x19 a, x5 p, x11 the word starts; per char x2 one, x3 stage1, x4 stage2, x6 pairs, x7 their mask,
 * x8 TOKS_FIB64, w9 the split id (or ~0), x10 the chars left, w12 the last char's entry, x13 j, x20 the small-alphabet
 * bitmap, w23 the no-id value, x24 p - 1, temps x1 x14-x17 x21 x22; d20 / d21 the window's end and its leads.
 * Per char: both entries PLAIN (spm.h: an id, not the split image) -> the pair set alone decides (it holds the small
 * alphabet's spanned pairs too, and a left char that is not PAIRED has no key), else the general rule (§5.5).
 * Frame: PROLOGUE 3.
 */
#include "asm.h"
#include "layout.h"

/* x\xd = the movemask of v4..v7 (ld4 lanes: byte 4i + k of the block in lane i of v(4 + k), each 0 or 0xff) */
.macro MM xd
        sri     v5.16b, v4.16b, #1
        sri     v7.16b, v6.16b, #1
        sri     v7.16b, v5.16b, #2
        sri     v7.16b, v7.16b, #4
        shrn    v7.8b, v7.8h, #4
        fmov    \xd, d7
.endm

/* v4..v7 = op(v0..v3, v\c) */
.macro CMP4 op, c
        \op     v4.16b, v0.16b, v\c\().16b
        \op     v5.16b, v1.16b, v\c\().16b
        \op     v6.16b, v2.16b, v\c\().16b
        \op     v7.16b, v3.16b, v\c\().16b
.endm

/* position j + k: bit j + k of x\acc = cut_ab8 (x13) byte of the pair p[j + k - 1] p[j + k], the 16 bits at byte
 * o + k of x14 (8 text bytes) */
.macro PAIR j, k, o, acc
        ubfx    x16, x14, #(8 * (\o + \k)), #16
        ldrb    w16, [x13, x16]
        bfi     x\acc, x16, #(\j + \k), #1
.endm

/* positions j .. j + 3 from one 8-byte load (p[j - 1 ..], the last group p[56 .. 63]) into four accumulators
 * (x11 x12 x15 x21: no chain through one register) */
.macro PAIR4 j
.if \j == 60
        ldr     x14, [x5, #56]
        PAIR    \j, 0, 3, 11
        PAIR    \j, 1, 3, 12
        PAIR    \j, 2, 3, 15
        PAIR    \j, 3, 3, 21
.else
        ldur    x14, [x5, #(\j - 1)]
        PAIR    \j, 0, 0, 11
        PAIR    \j, 1, 0, 12
        PAIR    \j, 2, 0, 15
        PAIR    \j, 3, 0, 21
.endif
.endm

FUNC toks_k7_spm_neon
        PROLOGUE 3, 0, 0
        mov     x19, x1
        ldr     x9, [x1, #K7_FLAGS]
        ldp     x10, x11, [x1, #K7_LEN]                 /* len, pos */
        tbz     x9, #0, 9f
        cbz     x11, 9f
        sub     x12, x10, x11
        cmp     x12, #64
        b.lo    9f
        ldr     x5, [x1, #K7_TEXT]
        add     x5, x5, x11
        ld4     {v0.16b, v1.16b, v2.16b, v3.16b}, [x5]
        cmlt    v4.16b, v0.16b, #0
        cmlt    v5.16b, v1.16b, #0
        cmlt    v6.16b, v2.16b, #0
        cmlt    v7.16b, v3.16b, #0
        MM      x9                                      /* non-ascii */
        mov     x21, #64                                /* the window's end */
        mov     x22, #-1                                /* its positions */
        mov     x10, #0                                 /* continuation bytes */
        mov     x14, #0                                 /* leads */
        cbz     x9, 1f
        movi    v16.16b, #0xc0
        movi    v17.16b, #0x80
        CMP4    and, 16
        cmeq    v4.16b, v4.16b, v17.16b
        cmeq    v5.16b, v5.16b, v17.16b
        cmeq    v6.16b, v6.16b, v17.16b
        cmeq    v7.16b, v7.16b, v17.16b
        MM      x10
        movi    v16.16b, #0xe0
        CMP4    cmhs, 16
        MM      x12                                     /* 3-4 byte leads */
        movi    v16.16b, #0xf0
        CMP4    cmhs, 16
        MM      x13                                     /* 4 byte leads (f5..ff too: rejected by the decode) */
        bic     x14, x9, x10
        lsl     x15, x14, #1
        orr     x15, x15, x12, lsl #2
        orr     x15, x15, x13, lsl #3
        cmp     x15, x10
        b.ne    9f                                      /* not the leads' continuations: the twin */
        and     x15, x14, #0x8000000000000000           /* chars running past the block */
        and     x16, x12, #0x4000000000000000
        orr     x15, x15, x16
        and     x16, x13, #0x2000000000000000
        orr     x15, x15, x16
        rbit    x15, x15
        clz     x21, x15
        sub     x16, x21, #1
        mov     x15, #2
        lsl     x15, x15, x16
        sub     x22, x15, #1
1:      ldurb   w15, [x5, #-1]
        lsr     w15, w15, #7
        eor     w15, w15, #1                            /* the byte before p is ascii */
        mvn     x16, x9
        orr     x15, x15, x16, lsl #1
        and     x15, x15, x16
        and     x17, x15, x22                           /* ascii after ascii */
        bic     x10, x22, x10
        bic     x10, x10, x17                           /* every other char */
        and     x14, x14, x22
        fmov    d20, x21
        fmov    d21, x14
        ldr     x13, [x0, #SPM_CUT_AB8]
        mov     x11, #0
        fmov    d16, x17
        cnt     v16.8b, v16.8b
        addv    b16, v16.8b
        fmov    w16, s16
        cmp     w16, #24
        b.lo    3f
        mov     x12, #0
        mov     x15, #0
        mov     x21, #0
        .irp j, 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60
        PAIR4   \j
        .endr
        orr     x11, x11, x12
        orr     x15, x15, x21
        orr     x11, x11, x15
        and     x11, x11, x17
        b       4f
3:      cbz     x17, 4f
        rbit    x14, x17
        clz     x14, x14
        add     x16, x5, x14
        ldurh   w16, [x16, #-1]
        ldrb    w1, [x13, w16, uxtw]
        lsl     x1, x1, x14
        orr     x11, x11, x1
        sub     x14, x17, #1
        and     x17, x17, x14
        b       3b
4:      ldr     x2, [x19, #K7_ONE]
        add     x16, x0, #SPM_STAGE1
        ldp     x3, x4, [x16]                           /* stage1, stage2 */
        ldp     x6, x7, [x16, #(SPM_PAIRS - SPM_STAGE1)]  /* pairs (then the mirror slot), their mask */
        mov     x8, #0x7c15
        movk    x8, #0x7f4a, lsl #16
        movk    x8, #0x79b9, lsl #32
        movk    x8, #0x9e37, lsl #48                    /* TOKS_FIB64 */
        ldr     w9, [x0, #SPM_ID_REPL]
        ldr     w16, [x0, #SPM_MS_SPLIT]
        cmp     w16, #0
        csinv   w9, w9, wzr, ne                         /* the split id, or ~0: no split */
        add     x20, x0, #SPM_CUT
        mov     w23, #0x1fffff                          /* no id */
        sub     x24, x5, #1                             /* p - 1 */
        ldr     x12, [x19, #K7_PREV]
5:      cbz     x10, 8f
        rbit    x13, x10
        clz     x13, x13                                /* j */
        sub     x14, x10, #1
        and     x10, x10, x14
        ldrb    w15, [x24, x13]
        ldr     w17, [x0, w15, uxtw #2]                 /* ascii[] (a non-ascii byte reads inside *s: dropped) */
        cmp     w15, #0x80
        csel    w15, w17, w12, lo                       /* x: the previous char's entry */
        cmp     x13, #60
        b.hi    65f
        ldr     w17, [x5, x13]                          /* the char's bytes b0 b1 b2 b3 */
60:     tbz     w17, #7, 61f
        tbz     w17, #5, 62f
        tbnz    w17, #4, 64f
        ubfx    w1, w17, #10, #4                        /* 3 bytes: cp >> 8 and cp & 0xff from the bytes */
        bfi     w1, w17, #4, #4
        sub     w21, w1, #0xd8
        cmp     w21, #8
        ccmp    w1, #8, #0, hs
        b.lo    9f                                      /* overlong (e0 80..9f) or a surrogate (ed a0..bf) */
        ubfx    w21, w17, #16, #6
        lsr     w22, w17, #8
        bfi     w21, w22, #6, #2
        mov     w22, #(3 << 24)
63:     ldrh    w1, [x3, w1, uxtw #1]
        orr     w21, w21, w1, lsl #8
        ldr     w16, [x4, w21, uxtw #2]                 /* y: the char's entry */
        bfxil   w22, w16, #0, #21
        str     w22, [x2, x13, lsl #2]                  /* one[j] = k << 24 | id */
7:      and     w21, w15, w16
        mov     w12, w16
        tbz     w21, #31, 70f                           /* not both PLAIN: the general rule */
71:     ubfiz   x22, x15, #21, #21
        bfxil   x22, x16, #0, #21                       /* the pair set's key id_x << 21 | id_y */
        mul     x21, x22, x8
        lsr     x21, x21, #32
        and     x21, x21, x7                            /* its home slot i */
        add     x16, x6, x21, lsl #3
        ldp     x17, x14, [x16]                         /* slots i, i + 1 (the mirror slot: no wrap) */
        cmp     x17, x22
        ccmp    x14, x22, #4, ne                        /* eq: spanned (a key sits at most once, before any empty) */
        cset    w1, ne                                  /* not spanned: a certified cut */
        ccmn    x17, #1, #4, ne
        ccmn    x14, #1, #4, ne                         /* eq: decided (spanned, or an empty slot ends the probe) */
        b.ne    74f
76:     lsl     x1, x1, x13
        orr     x11, x11, x1
        b       5b
61:     and     w17, w17, #0xff                         /* an ascii char */
        ldr     w16, [x0, w17, uxtw #2]
        b       7b
62:     tst     w17, #0x1e                              /* 2 bytes */
        b.eq    9f                                      /* c0 c1: overlong */
        ubfx    w1, w17, #2, #3
        ubfx    w21, w17, #8, #6
        bfi     w21, w17, #6, #2
        mov     w22, #(2 << 24)
        b       63b
64:     and     w1, w17, #0xff                          /* 4 bytes: f0..f4 */
        cmp     w1, #0xf4
        b.hi    9f
        and     w1, w17, #7
        ubfx    w21, w17, #8, #6
        lsl     w1, w1, #18
        orr     w1, w1, w21, lsl #12
        ubfx    w21, w17, #16, #6
        orr     w1, w1, w21, lsl #6
        ubfx    w21, w17, #24, #6
        orr     w1, w1, w21
        sub     w21, w1, #0x10, lsl #12
        mov     w22, #0x100000
        cmp     w21, w22
        b.hs    9f                                      /* overlong or past U+10FFFF */
        and     w21, w1, #0xff
        lsr     w1, w1, #8
        mov     w22, #(4 << 24)
        b       63b
65:     ldr     w17, [x5, #60]                          /* a char at p + 61 or 62: its bytes end the block */
        sub     x1, x13, #60
        lsl     x1, x1, #3
        lsr     w17, w17, w1
        b       60b
70:     and     w14, w15, #0x1fffff                     /* the general rule (§5.5): id_x */
        and     w17, w16, #0x1fffff                     /* id_y */
        mov     w1, #1
        cmp     w17, w9
        b.eq    76b                                     /* the Metaspace split: a cut */
        mov     w1, #0
        cmp     w14, w23
        ccmp    w17, w23, #4, ne
        b.eq    76b                                     /* an id-less char: no certified cut */
        ubfx    w21, w15, #21, #8
        ubfx    w22, w16, #21, #8
        mov     w14, #0xff
        cmp     w21, w14
        ccmp    w22, w14, #4, ne
        b.eq    71b                                     /* not both small: the pair set decides */
        add     w21, w21, w21, lsl #7                   /* both small: bit xa * 129 + xb of the bitmap */
        add     w21, w21, w22
        lsr     w22, w21, #5
        ldr     w22, [x20, w22, uxtw #2]
        lsr     w1, w22, w21
        and     w1, w1, #1
        b       76b
74:     add     x21, x21, #2                            /* neither slot decides: probe on from slot i + 2 */
73:     and     x21, x21, x7
        ldr     x14, [x6, x21, lsl #3]
        add     x21, x21, #1
        cmp     x14, x22
        b.eq    72f
        cmn     x14, #1
        b.ne    73b
        mov     w1, #1                                  /* an empty slot: not spanned, a cut */
        b       76b
72:     mov     w1, #0                                  /* spanned */
        b       76b
8:      fmov    x14, d20
        add     x15, x5, x14
        ldurb   w16, [x15, #-1]
        ldr     w17, [x0, w16, uxtw #2]
        cmp     w16, #0x80
        csel    w12, w17, w12, lo                       /* the last char's entry */
        str     x12, [x19, #K7_PREV]
        ldr     x15, [x19, #K7_POS]
        add     x15, x15, x14
        str     x15, [x19, #K7_POS]
        fmov    x15, d21
        str     x15, [x19, #K7_MB]
        mov     x0, x11
        EPILOGUE 3, 0, 0
9:      mov     x0, #0                                  /* not this window: pos unchanged */
        EPILOGUE 3, 0, 0
ENDFUNC toks_k7_spm_neon
