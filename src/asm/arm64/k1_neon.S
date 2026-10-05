/*
 * src/asm/arm64/k1_neon.S: K1, added-token find, tier neon (docs/kernels.md §4; the c twin is src/core/k1_c.c).
 *
 *   uint64_t toks_k1_added_find_neon(const toks_tables *t, toks_k1_args *a)     returns n (matches in a->m)
 *
 * Outputs, return value and every read equal the c twin's: up to cap matches, each the leftmost position >= the
 * previous match's end (the first: >= pos) whose first byte passes the phase's shufti test (lo[b & 15] &
 * hi[b >> 4] != 0) and where some token of the phase occurs entirely inside [0, len) -- single[b0] first, then
 * index2[b0 | b1 << 8]'s candidates, longest first, the first exact match wins (`el <= best_len` ends the list, a
 * token running past len is skipped). After a match the block's candidates below its end are dropped and the
 * scan goes on in the same block (or from the end, when it lies past the block).
 *
 * Scan: 64-byte blocks from text + pos: per 16 bytes and, ushr, two tbl and an and give a nonzero byte per
 * candidate; one umaxp per block answers "any candidate" (the common case: none, next block). A block with
 * one gets its 64-bit mask (cmtst, bit weights, four addp) and its candidates in ascending order. The 1..63
 * bytes after the last full block are the 64-byte block that ends at len with the scanned positions masked
 * off. A text of 16..63 bytes is four 16-byte loads at 0, min(16, len - 16), min(32, len - 16) and len - 16
 * merged into one mask; one under 16 bytes builds its mask byte by byte. Every load lies inside
 * [text, text + len): no padding, no copy.
 * Verify (per candidate, kernels.md §4): with >= 4 bytes left, the h4 bucket of the first four bytes (tokens of
 * >= 4 bytes, longest first: the first exact match wins); else, or with none, index2's bucket from its end (the
 * 2- and 3-byte tokens, shortest first, the last match wins), then single. A compare reads exactly [q, q + el) and
 * the token's [off, off + el): 16-byte chunks from the end down (the last one at 0), or two overlapping 8 / 4 /
 * 2-byte loads.
 * Lineage: shufti (Hyperscan's nibble-table byte classification, as in simdjson), the movemask by bit
 * weights and pairwise adds (simdjson's arm64 kernels), overlapping head / tail loads instead of a padded
 * copy (the K5 key loads; memcmp implementations), daachorse's leftmost-longest semantics (hf).
 *
 * Registers: x0 t (then a temp), x1 a (n lives in a->n), x2 text, x4 last = text + len - 64 (end = x4 + 64), x5 the
 * valid positions of a block (~0; the remainder block: its unscanned top bits), x6 index2 + phase, x7 single +
 * phase, x9 block base, x10 candidate mask; verify: x11 q, x14 end - q, x13 best_len, x15 candidate cursor, x16
 * candidates left, x17 e, x3 el, x8 the token's bytes, x12 x0 x25 temps; x19 the h4 multiplier, x21 h4 + 8 phase,
 * x22 add_cand, x23 add_entries, x24 add_bytes. v16 / v17 lo / hi tables, v18 0x0f, v19 bit weights.
 * Frame: PROLOGUE 4.
 */
#include "asm.h"
#include "layout.h"

/* local labels: L on mach-o (an .L label there is a symbol, which breaks the frame's cfi), .L elsewhere */
#if defined(TOKS_MACHO)
#define LBL(x) L##x
#else
#define LBL(x) .L##x
#endif

/* v<r> = nonzero where byte of v<r> passes shufti (lo v16, hi v17, 0x0f v18); clobbers v<tmp> */
.macro SHUFTI r, tmp
        and     v\tmp\().16b, v\r\().16b, v18.16b
        ushr    v\r\().16b, v\r\().16b, #4
        tbl     v\tmp\().16b, {v16.16b}, v\tmp\().16b
        tbl     v\r\().16b, {v17.16b}, v\r\().16b
        and     v\r\().16b, v\r\().16b, v\tmp\().16b
.endm

/* x10 = the 64-bit mask of v0..v3 (bit k: byte k of v0 v1 v2 v3 nonzero) */
.macro MASK64
        cmtst   v0.16b, v0.16b, v0.16b
        cmtst   v1.16b, v1.16b, v1.16b
        cmtst   v2.16b, v2.16b, v2.16b
        cmtst   v3.16b, v3.16b, v3.16b
        and     v0.16b, v0.16b, v19.16b
        and     v1.16b, v1.16b, v19.16b
        and     v2.16b, v2.16b, v19.16b
        and     v3.16b, v3.16b, v19.16b
        addp    v0.16b, v0.16b, v1.16b
        addp    v2.16b, v2.16b, v3.16b
        addp    v0.16b, v0.16b, v2.16b
        addp    v0.16b, v0.16b, v0.16b
        fmov    x10, d0
.endm

/* RXCMP: the x12 bytes at x0 (a radix label) equal those at x25 (the text), else b LBL(k1_rdone); reads only those
 * bytes (chunks from the end down, overlapping 8 / 4 / 2-byte pairs, one byte). Clobbers x3, x26, v0, v2 */
.macro RXCMP
        cbz     x12, 39f
        cmp     x12, #16
        b.lo    35f
        sub     x26, x12, #16
34:     ldr     q0, [x0, x26]
        ldr     q2, [x25, x26]
        eor     v0.16b, v0.16b, v2.16b
        umaxp   v0.16b, v0.16b, v0.16b
        fmov    x3, d0
        cbnz    x3, LBL(k1_rdone)
        cbz     x26, 39f
        subs    x26, x26, #16
        csel    x26, x26, xzr, gt
        b       34b
35:     cmp     x12, #8
        b.lo    36f
        ldr     x3, [x0]
        ldr     x26, [x25]
        cmp     x3, x26
        b.ne    LBL(k1_rdone)
        add     x3, x0, x12
        ldur    x3, [x3, #-8]
        add     x26, x25, x12
        ldur    x26, [x26, #-8]
        b       38f
36:     cmp     x12, #4
        b.lo    37f
        ldr     w3, [x0]
        ldr     w26, [x25]
        cmp     w3, w26
        b.ne    LBL(k1_rdone)
        add     x3, x0, x12
        ldur    w3, [x3, #-4]
        add     x26, x25, x12
        ldur    w26, [x26, #-4]
        b       38f
37:     cmp     x12, #2
        b.lo    33f
        ldrh    w3, [x0]
        ldrh    w26, [x25]
        cmp     w3, w26
        b.ne    LBL(k1_rdone)
        add     x3, x0, x12
        ldurh   w3, [x3, #-2]
        add     x26, x25, x12
        ldurh   w26, [x26, #-2]
        b       38f
33:     ldrb    w3, [x0]
        ldrb    w26, [x25]
38:     cmp     x3, x26
        b.ne    LBL(k1_rdone)
39:
.endm

FUNC toks_k1_added_find_neon
        PROLOGUE 4, 0, 0
        ldp     x2, x3, [x1, #K1_TEXT]          /* text, len */
        ldp     x9, x5, [x1, #K1_POS]           /* pos, phase */
        stp     xzr, x3, [x1, #K1_N]            /* n = 0, next = len */
        cmp     x5, #1
        b.hi    LBL(k1_none)
        ldr     x6, [x0, #TT_ADD_PHASES]
        lsr     x6, x6, x5
        tbz     x6, #0, LBL(k1_none)            /* no tokens in this phase: the tables may be NULL */
        cmp     x9, x3
        b.hs    LBL(k1_none)
        ldr     x8, [x0, #TT_ADD_SHUFTI]
        add     x8, x8, x5, lsl #5              /* + 32 phase */
        ldp     q16, q17, [x8]
        movi    v18.16b, #0x0f
        movz    x12, #0x0201
        movk    x12, #0x0804, lsl #16
        movk    x12, #0x2010, lsl #32
        movk    x12, #0x8040, lsl #48
        dup     v19.2d, x12
        ldr     x6, [x0, #TT_ADD_INDEX]
        add     x21, x6, #0x100, lsl #12        /* h4 = index2 + 2 * 65536 * 8 */
        add     x21, x21, x5, lsl #3            /* slot << 1 | phase */
        add     x6, x6, x5, lsl #19             /* + 65536 * 8 phase */
        ldr     x7, [x0, #TT_ADD_SINGLE]
        add     x7, x7, x5, lsl #10             /* + 256 * 4 phase */
        ldr     x22, [x0, #TT_ADD_CAND]
        ldp     x23, x24, [x0, #TT_ADD_ENTRIES] /* entries, bytes */
        movz    w19, #0x79B1
        movk    w19, #0x9E37, lsl #16           /* toks_k1_h4's multiplier */
        add     x4, x2, x3
        sub     x4, x4, #64                     /* last */
        add     x9, x2, x9                      /* text + pos */
        mov     x5, #-1
        cmp     x3, #64
        b.lo    LBL(k1_short)
        cmp     x9, x4
        b.hi    LBL(k1_tail)

#if !defined(TOKS_COFF)
        .p2align 4                              /* (coff: a mid-function align defeats the seh length) */
#endif
LBL(k1_loop):
        ld1     {v0.16b, v1.16b, v2.16b, v3.16b}, [x9]
        SHUFTI  0, 4
        SHUFTI  1, 5
        SHUFTI  2, 6
        SHUFTI  3, 7
        orr     v4.16b, v0.16b, v1.16b
        orr     v5.16b, v2.16b, v3.16b
        orr     v4.16b, v4.16b, v5.16b
        umaxp   v4.16b, v4.16b, v4.16b
        fmov    x12, d4
        cbnz    x12, LBL(k1_block)
LBL(k1_next):
        add     x9, x9, #64
        cmp     x9, x4
        b.ls    LBL(k1_loop)
LBL(k1_tail):                                   /* 0..63 bytes left after x9 */
        add     x12, x4, #64                    /* end */
        subs    x12, x12, x9                    /* r */
        b.ls    LBL(k1_none)
        mov     x14, #64
        sub     x14, x14, x12
        lsl     x5, x5, x14                     /* the block ending at len: only its top r bytes are new */
        mov     x9, x4
        b       LBL(k1_loop)

LBL(k1_block):
        MASK64
        and     x10, x10, x5
LBL(k1_cand):                                   /* x10: candidates at x9 + bit */
        cbz     x10, LBL(k1_next)
LBL(k1_cloop):
        rbit    x11, x10
        clz     x11, x11
        sub     x12, x10, #1
        and     x10, x10, x12
        add     x11, x9, x11                    /* q */
        add     x14, x4, #64
        sub     x14, x14, x11                   /* end - q */
        cmp     x14, #4
        b.lo    LBL(k1_s23)                     /* under four bytes left: no token of four or more fits */
        ldr     w12, [x11]
        mul     w12, w12, w19
        lsr     w12, w12, #(32 - TOKS_K1_H4_BITS)
        add     x12, x21, x12, lsl #4
        ldr     x15, [x12]                      /* the h4 bucket: start | count << 32 */
        tbnz    x15, #63, LBL(k1_radix)         /* a long bucket: its radix tree */
        lsr     x16, x15, #32
        cbz     x16, LBL(k1_s23)
        add     x15, x22, w15, uxtw #2          /* candidate cursor */
LBL(k1_vloop):
        ldr     w17, [x15], #4                  /* e */
        add     x8, x23, x17, lsl #4
        ldrh    w3, [x8, #TAE_LEN]              /* el */
        cmp     x14, x3
        b.lo    LBL(k1_vnext)                   /* runs past len */
        ldr     w8, [x8, #TAE_OFF]
        add     x8, x24, x8                     /* the token's bytes */
        cmp     x3, #16
        b.lo    LBL(k1_lt16)
        sub     x12, x3, #16                    /* 16-byte chunks from the end down, the last at 0 */
LBL(k1_c16):
        ldr     q0, [x8, x12]
        ldr     q1, [x11, x12]
        eor     v0.16b, v0.16b, v1.16b
        umaxp   v0.16b, v0.16b, v0.16b
        fmov    x0, d0
        cbnz    x0, LBL(k1_vnext)
        cbz     x12, LBL(k1_hit)
        subs    x12, x12, #16
        csel    x12, x12, xzr, gt
        b       LBL(k1_c16)
LBL(k1_lt16):
        cmp     x3, #8
        b.lo    LBL(k1_lt8)
        ldr     x12, [x8]
        ldr     x0, [x11]
        cmp     x12, x0
        b.ne    LBL(k1_vnext)
        add     x12, x8, x3
        ldur    x12, [x12, #-8]
        add     x0, x11, x3
        ldur    x0, [x0, #-8]
        b       LBL(k1_cmp)
LBL(k1_lt8):                                    /* 4..7 bytes: two overlapping words */
        ldr     w12, [x8]
        ldr     w0, [x11]
        cmp     w12, w0
        b.ne    LBL(k1_vnext)
        add     x12, x8, x3
        ldur    w12, [x12, #-4]
        add     x0, x11, x3
        ldur    w0, [x0, #-4]
LBL(k1_cmp):
        cmp     x12, x0
        b.eq    LBL(k1_hit)
LBL(k1_vnext):
        subs    x16, x16, #1
        b.ne    LBL(k1_vloop)
LBL(k1_s23):                                    /* 2 or 3 bytes: index2's bucket from its end, shortest first */
        mov     x13, #0                         /* best_len */
        cmp     x14, #2
        b.lo    LBL(k1_s1)                      /* q + 1 == end: no second byte */
        ldrh    w12, [x11]                      /* b0 | b1 << 8 */
        ldr     x15, [x6, x12, lsl #3]          /* start | count << 32 */
        lsr     x16, x15, #32
        cbz     x16, LBL(k1_s1)
        add     x15, x22, w15, uxtw #2
        add     x15, x15, x16, lsl #2           /* the bucket's end */
LBL(k1_sloop):
        ldr     w12, [x15, #-4]!                /* e */
        add     x8, x23, x12, lsl #4
        ldrh    w3, [x8, #TAE_LEN]
        cmp     x3, #4
        b.hs    LBL(k1_s1)                      /* the rest are longer: h4's */
        cmp     x14, x3
        b.lo    LBL(k1_snext)                   /* runs past len */
        ldr     w8, [x8, #TAE_OFF]
        add     x8, x24, x8
        ldrh    w0, [x8]
        ldrh    w25, [x11]
        cmp     w0, w25
        b.ne    LBL(k1_snext)
        add     x0, x8, x3
        ldurh   w0, [x0, #-2]
        add     x25, x11, x3
        ldurh   w25, [x25, #-2]
        cmp     w0, w25
        b.ne    LBL(k1_snext)
        mov     x13, x3                         /* a match: the longest so far */
        mov     x17, x12
LBL(k1_snext):
        subs    x16, x16, #1
        b.ne    LBL(k1_sloop)
LBL(k1_s1):
        cbnz    x13, LBL(k1_shit)
        ldrb    w12, [x11]
        ldr     w17, [x7, x12, lsl #2]
        cbz     w17, LBL(k1_fp)                 /* no token here: a shufti false positive */
        sub     w17, w17, #1                    /* the one-byte token */
        mov     x3, #1
        b       LBL(k1_hit)
LBL(k1_shit):
        mov     x3, x13
        b       LBL(k1_hit)
LBL(k1_fp):
        cbnz    x10, LBL(k1_cloop)
        b       LBL(k1_next)
LBL(k1_hit):                                    /* q, el = x3, entry = x17 */
        sub     x12, x11, x2                    /* start */
        add     x13, x12, x3                    /* end */
        ldp     x14, x15, [x1, #K1_M]           /* m, cap */
        ldr     x16, [x1, #K1_N]
        add     x14, x14, x16, lsl #4
        stp     w12, w13, [x14]                 /* m[n] = { start, end, entry, 0 } */
        stp     w17, wzr, [x14, #8]
        add     x16, x16, #1
        str     x16, [x1, #K1_N]
        cmp     x16, x15
        b.hs    LBL(k1_full)
        add     x11, x11, x3                    /* text + end: the next search starts there */
        sub     x12, x11, x9                    /* its bit in the block (>= 1) */
        cmp     x12, #64
        b.hs    LBL(k1_resume)
        mov     x14, #-1
        lsl     x14, x14, x12
        and     x10, x10, x14                   /* drop the candidates inside the match */
        b       LBL(k1_cand)
LBL(k1_resume):                                 /* the match ends past this block (never past a remainder block's end) */
        mov     x9, x11
        cmp     x9, x4
        b.ls    LBL(k1_loop)
        b       LBL(k1_tail)
LBL(k1_full):
        str     x13, [x1, #K1_NEXT]
        mov     x0, x16
        EPILOGUE 4, 0, 0

LBL(k1_none):
        ldr     x0, [x1, #K1_N]
        EPILOGUE 4, 0, 0

LBL(k1_short):                                  /* len < 64 */
        cmp     x3, #16
        b.lo    LBL(k1_tiny)
        sub     x12, x3, #16                    /* the last chunk at len - 16 */
        mov     x13, #16
        cmp     x12, #16
        csel    x13, x13, x12, hs               /* min(16, len - 16) */
        mov     x14, #32
        cmp     x12, #32
        csel    x14, x14, x12, hs               /* min(32, len - 16) */
        ldr     q0, [x2]
        ldr     q1, [x2, x13]
        ldr     q2, [x2, x14]
        ldr     q3, [x2, x12]
        SHUFTI  0, 4
        SHUFTI  1, 5
        SHUFTI  2, 6
        SHUFTI  3, 7
        MASK64                                  /* 16 bits per chunk */
        ubfx    x15, x10, #16, #16
        lsl     x15, x15, x13
        ubfx    x16, x10, #32, #16
        lsl     x16, x16, x14
        lsr     x17, x10, #48
        lsl     x17, x17, x12
        and     x10, x10, #0xffff
        orr     x10, x10, x15
        orr     x16, x16, x17
        orr     x10, x10, x16
        b       LBL(k1_from)
LBL(k1_tiny):                                   /* len < 16: the mask byte by byte */
        sub     x12, x9, x2                     /* i = pos */
        mov     x10, #0
        mov     x15, #1
LBL(k1_tloop):
        ldrb    w13, [x2, x12]
        and     w14, w13, #15
        ldrb    w14, [x8, x14]
        lsr     w13, w13, #4
        add     x13, x8, x13
        ldrb    w13, [x13, #16]
        tst     w13, w14
        lsl     x13, x15, x12
        orr     x13, x10, x13
        csel    x10, x10, x13, eq
        add     x12, x12, #1
        cmp     x12, x3
        b.lo    LBL(k1_tloop)
LBL(k1_from):                                   /* x10: candidates at text + bit; drop those before pos */
        sub     x12, x9, x2
        lsl     x12, x5, x12
        and     x10, x10, x12
        mov     x9, x2
        b       LBL(k1_cand)
LBL(k1_radix):                                  /* kernels.md §4: down the tree from the root at add_cand + w15 */
        mov     x13, #0                         /* best_len */
        mov     x16, #0                         /* d: the bytes matched */
        mov     w15, w15
LBL(k1_rnode):
        add     x8, x22, x15                    /* the node */
        ldrh    w12, [x8, #RX_LLEN]
        sub     x0, x14, x16
        cmp     x12, x0
        b.hi    LBL(k1_rdone)                   /* its label runs past len */
        ldr     w0, [x8, #RX_LAB]
        add     x0, x22, x0                     /* the label, then the children's first bytes */
        add     x25, x11, x16                   /* text + q + d */
        RXCMP
        add     x16, x16, x12
        add     x0, x0, x12
        ldr     w3, [x8, #RX_ENT]
        cbz     w3, 31f
        sub     w17, w3, #1                     /* a token ends here: the longest so far */
        mov     x13, x16
31:     ldrh    w3, [x8, #RX_NCH]
        cmp     x16, x14
        ccmp    x3, #0, #4, lo
        b.eq    LBL(k1_rdone)                   /* no child, or the text ends here */
        ldrb    w26, [x11, x16]
        dup     v1.16b, w26                     /* the next byte, against the children's first bytes */
        mov     x12, #0
32:     ldr     q0, [x0, x12]
        cmeq    v0.16b, v0.16b, v1.16b
        shrn    v0.8b, v0.8h, #4
        fmov    x25, d0                         /* 4 bits a byte */
        sub     x26, x3, x12
        cmp     x26, #16
        b.hs    30f
        lsl     x26, x26, #2
        mov     x20, #-1
        lsl     x20, x20, x26
        bic     x25, x25, x20                   /* only the children's bytes */
30:     cbnz    x25, 29f
        add     x12, x12, #16
        cmp     x12, x3
        b.lo    32b
        b       LBL(k1_rdone)
29:     rbit    x25, x25
        clz     x25, x25
        add     x12, x12, x25, lsr #2           /* the child */
        ldr     w15, [x8, #RX_CH]
        add     x15, x15, x12, lsl #4
        b       LBL(k1_rnode)
LBL(k1_rdone):
        cbnz    x13, LBL(k1_shit)
        b       LBL(k1_s23)
ENDFUNC toks_k1_added_find_neon
