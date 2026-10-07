/* k5_neon.S: K5, encode pieces, tier neon (docs/kernels.md §6; the c twin is src/core/k5_c.c).
 *
 *   uint64_t toks_k5_encode_neon(const toks_tables *t, toks_k5_args *a)     returns n_out
 *
 * Per piece [s, e): one byte -> byte2id (a static hit); 2..15 bytes -> the 16-byte key (bytes, zero pad,
 * byte 15 = len) and its crc32cx hash, then the dynamic cache's bucket (two ways) and the static words table's
 * bucket h (three ways, layout.h TOKS_W3_*) and, only when it misses, bucket rotr32(h, 16) when h's spill bit is
 * set, else bucket h again (in L1, it misses again: a csel, no branch, no second line); a cache hit stores its 4
 * masked ids at once, a static hit its value's two ids, and each advances by the count. Otherwise (and for every
 * piece over 15 bytes) a
 * call to toks_k5_long_neon (K6, or a's lcache in front of it) writes the ids in place. A K6 answer of <= 4 ids
 * for a piece of <= 15 bytes fills the cache (way 0 -> way 1, the new entry -> way 0). A warm scratch (a's
 * counters sum to >= TOKS_K5_WARM pieces since init) probes the cache first and fills it with static answers
 * too; a fresh one probes the static table first and keeps static answers out (kernels.md §6). Cache entries
 * carry a's cache_tag in their val (layout.h): the way whose key matches (way 0 first) answers only when its tag
 * is the call's, and a fill writes the tag. This call's counts are added to a's three counters. Outputs,
 * counters and cache contents equal the c twin's.
 *
 * The key is one 16-byte ldp and a mask for len from a table (general registers: the crc32cx steps need no
 * vector-to-general moves) when the piece starts >= 16 bytes before the segment's end; otherwise it is
 * assembled from loads inside the piece (text[len..] is never read). Each bucket's ways are one compare
 * chain selected with csel and one branch; the cache's starts from NOWAY, a frame value whose tag word is never
 * the call's, so the chosen way's tag and count come from one ldp. Measured on spark (A725) and kept out: loading
 * the words buckets for every piece to drop the cache-hit branch (7-14% slower: hits then wait on words lines),
 * and prefetching the next 8 pieces' buckets (+-2%).
 *
 * Nothing is copied: the key comes from an in-place load of the piece, ids go straight to out (K6 writes its
 * ids in place too); the only spill is the 16-byte key in the frame across a K6 call, for the fill after it.
 * Lineage: e's tok v1 and gigatoken (the 16-byte key with its length byte from one masked load, crc32c, the
 * two-way 64-byte cache line with way 0 -> way 1 fills, the unconditional 4-id store advanced by the count,
 * tok v1's two-choice WORDS table, gigatoken's csel probe of one line).
 *
 * Frame: x19 t, x20 a, x21 ends cursor, x22 ends end, x23 text, x24 out cursor, x25 s, x26 cache hits,
 * x27 the tag word (toks_tag_word(cache_tag)), x28 cache (static hits = n - cache hits - misses); locals: K6's
 * args, the key, the bucket, the misses, warm << 63, NOWAY.
 * Reloaded after every K6 call: x9 cache_mask | warm << 63, x10 words, x11 words_mask, x12 byte2id, x13 text + len
 * - 16, x14 TOKS_HSEED, x15 the key mask table (+256: {0,1,2,3} for a fill's pack), v30 TOKS_ID_MASK.
 */
#include "asm.h"
#include "layout.h"

/* local labels: L on mach-o (an .L label there is a symbol, which breaks the frame's cfi), .L elsewhere */
#if defined(TOKS_MACHO)
#define LBL(x) L##x
#else
#define LBL(x) .L##x
#endif

        .arch_extension crc               /* tier neon = armv8.0-a + neon + crc32 (docs/kernels.md §1) */

#define K6ARGS  LOCALS
#define KEYSAVE (LOCALS + 64)
#define BKTSAVE (LOCALS + 80)
#define MISSAVE (LOCALS + 88)
#define WARM    (LOCALS + 96)
#define NOWAY   (LOCALS + 104)  /* 16 bytes: the probe's value when no cache way holds the key */

        RODATA
        .p2align 6
DATA toks_k5_neon_consts
        /* the key's byte mask for len 0..15: bytes 0-7, bytes 8-15 (byte 15 is set from len) */
        .quad   0, 0
        .quad   0xFF, 0
        .quad   0xFFFF, 0
        .quad   0xFFFFFF, 0
        .quad   0xFFFFFFFF, 0
        .quad   0xFFFFFFFFFF, 0
        .quad   0xFFFFFFFFFFFF, 0
        .quad   0xFFFFFFFFFFFFFF, 0
        .quad   0xFFFFFFFFFFFFFFFF, 0
        .quad   0xFFFFFFFFFFFFFFFF, 0xFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFFFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFF
        .quad   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFF
        .long   0, 1, 2, 3                      /* +256: lanes for the miss path's pack */
        TEXT

/* LOAD_LE d, p, n, t1, t2: x<d> = the n bytes (1 <= n <= 8, in x<n>) at x<p>, little-endian, zero-extended;
 * reads only [x<p>, x<p> + n) */
.macro LOAD_LE d, p, n, t1, t2
        cmp     x\n, #4
        b.lo    LBL(le_s\@)
        ldr     w\d, [x\p]
        add     x\t1, x\p, x\n
        ldur    w\t2, [x\t1, #-4]
        sub     w\t1, w\n, #4
        b       LBL(le_m\@)
LBL(le_s\@):
        cmp     x\n, #2
        b.lo    LBL(le_1\@)
        ldrh    w\d, [x\p]
        add     x\t1, x\p, x\n
        ldurh   w\t2, [x\t1, #-2]
        sub     w\t1, w\n, #2
LBL(le_m\@):
        lsl     w\t1, w\t1, #3
        lsl     x\t2, x\t2, x\t1
        orr     x\d, x\d, x\t2
        b       LBL(le_d\@)
LBL(le_1\@):
        ldrb    w\d, [x\p]
LBL(le_d\@):
.endm

.macro K5_RELOAD
        ldr     x9, [x20, #K5_CACHE_MASK]
        ldr     x16, [sp, #WARM]
        orr     x9, x9, x16                     /* bit 63: a warm scratch (h is 32 bits: the bucket never sees it) */
        ldp     x10, x11, [x19, #TT_WORDS]
        ldr     x12, [x19, #TT_BYTE2ID]
        ldr     x13, [x20, #K5_LEN]
        add     x13, x13, x23
        sub     x13, x13, #16                   /* the last piece start whose 16-byte load stays in the segment */
        movz    w14, #0x79B9
        movk    w14, #0x9E37, lsl #16
        ADRP_ADD x15, SYM(toks_k5_neon_consts)
        movi    v30.4s, #0x1f, msl #16
.endm

/* WAY b, off: the key (x4, x5) == the 16 bytes at x<b> + off -> x3 = the address of its value */
.macro WAY b, off
        ldp     x17, x8, [x\b, #\off]
        cmp     x17, x4
        ccmp    x8, x5, #0, eq
        add     x17, x\b, #(\off + 32)
        csel    x3, x17, x3, eq
.endm

/* WAY3 b, w: the key == way w's 16 bytes at x<b> + 16 w (a words bucket) -> x3 = the address 3 bytes before its
 * 5-byte value (TOKS_W3_VAL + 5 w), so one 8-byte load inside the bucket and lsr #24 give id0 | id1 << 20 */
.macro WAY3 b, w
        ldp     x17, x8, [x\b, #(16 * \w)]
        cmp     x17, x4
        ccmp    x8, x5, #0, eq
        add     x17, x\b, #(TOKS_W3_VAL - 3 + 5 * \w)
        csel    x3, x17, x3, eq
.endm

FUNC toks_k5_encode_neon
        PROLOGUE 5, 0, 128
        mov     x19, x0
        mov     x20, x1
        ldr     x23, [x20, #K5_TEXT]
        ldp     x21, x22, [x20, #K5_ENDS]
        add     x22, x21, x22, lsl #2
        ldp     x25, x24, [x20, #K5_START]
        mov     x26, #0
        str     xzr, [sp, #MISSAVE]
        ldr     x16, [x20, #K5_CACHE_TAG]       /* x27 = toks_tag_word(cache_tag) (layout.h) */
        ubfx    x17, x16, #0, #11
        ubfx    x16, x16, #11, #11
        lsl     x17, x17, #21
        orr     x27, x17, x16, lsl #53
        eor     x16, x27, #0xFFE00000FFE00000   /* NOWAY's tag word: never the call's (TOKS_TAG_MASK64) */
        str     x16, [sp, #(NOWAY + 8)]
        ldp     x16, x17, [x20, #K5_HITS_STATIC]
        ldr     x0, [x20, #K5_MISSES]
        add     x16, x16, x17
        add     x16, x16, x0
        cmp     x16, #TOKS_K5_WARM
        cset    x16, hs                         /* the pieces since init: a warm scratch (layout.h) */
        lsl     x16, x16, #63
        str     x16, [sp, #WARM]
        ldr     x28, [x20, #K5_CACHE]
        K5_RELOAD

LBL(k5_next):
        cmp     x21, x22
        b.hs    LBL(k5_end)
LBL(k5_piece):
        ldr     w0, [x21], #4                   /* e */
        sub     x1, x0, x25                     /* len */
        add     x2, x23, x25                    /* piece */
        mov     x25, x0
        cmp     x1, #1
        b.eq    LBL(k5_one)
        cmp     x1, #TOKS_KEY_MAXLEN
        b.hi    LBL(k5_long)
        cmp     x2, x13
        b.hi    LBL(k5_tail)
        ldp     x4, x5, [x2]
        add     x6, x15, x1, lsl #4
        ldp     x6, x7, [x6]
        and     x4, x4, x6
        and     x5, x5, x7
        orr     x5, x5, x1, lsl #56             /* byte 15 = len */
LBL(k5_key):
        crc32cx w6, w14, x4
        crc32cx w6, w6, x5
        and     x7, x6, x9
        add     x7, x28, x7, lsl #6             /* the cache bucket (when x28 != 0) */
        tbz     x9, #63, LBL(k5_static)         /* a fresh scratch: the static table first */
LBL(k5_cache):
        cbz     x28, LBL(k5_cmiss)
        add     x3, sp, #NOWAY
        WAY     7, 16
        WAY     7, 0                            /* the way by key, way 0 first (else NOWAY); then its tag decides */
        ldp     x17, x16, [x3]
        and     x16, x16, #0xFFE00000FFE00000   /* TOKS_TAG_MASK64 */
        cmp     x16, x27
        b.ne    LBL(k5_cmiss)
        ldr     q1, [x3]                        /* a cache hit: 4 masked ids, advance by the count */
        and     v1.16b, v1.16b, v30.16b
        str     q1, [x24]
        ubfx    x17, x17, #TOKS_VAL_COUNT_SHIFT, #3
        add     x24, x24, x17, lsl #2
        add     x26, x26, #1
        cmp     x21, x22
        b.lo    LBL(k5_piece)
        b       LBL(k5_end)
LBL(k5_cmiss):
        tbz     x9, #63, LBL(k5_miss)           /* fresh: the static table missed before */
LBL(k5_static):
        mov     x3, #0
        cbz     x10, LBL(k5_smiss)
        and     x0, x6, x11
        add     x0, x10, x0, lsl #6             /* words bucket h */
        WAY3    0, 2
        WAY3    0, 1
        WAY3    0, 0
        cbnz    x3, LBL(k5_shit)
        ldrb    w17, [x0, #TOKS_W3_META]
        tbz     w17, #0, LBL(k5_smiss)          /* EXPERIMENT W3b': TOKS_W3_SPILL (bit 0) clear: no key of h spilled, a miss */
        ror     w16, w6, #16
        and     x16, x16, x11
        add     x16, x10, x16, lsl #6           /* words bucket rotr32(h, 16) */
        WAY3    16, 2
        WAY3    16, 1
        WAY3    16, 0
        cbnz    x3, LBL(k5_shit)
LBL(k5_smiss):
        tbz     x9, #63, LBL(k5_cache)          /* fresh: now the cache */
LBL(k5_miss):
        cmp     x28, #0
        csel    x7, xzr, x7, eq                 /* the bucket to fill, or 0 */
        b       LBL(k5_k6)
LBL(k5_long):
        mov     x7, #0
LBL(k5_k6):
        stp     x2, x1, [sp, #(K6ARGS + K6_PIECE)]
        ldp     x16, x17, [x20, #K5_WORK]
        stp     x24, x16, [sp, #(K6ARGS + K6_OUT)]
        str     x17, [sp, #(K6ARGS + K6_WORK_BYTES)]
        stp     x4, x5, [sp, #KEYSAVE]
        str     x7, [sp, #BKTSAVE]
        mov     x0, x19
        add     x1, sp, #K6ARGS
        ldr     x2, [x20, #K5_LCACHE]           /* K6, or the long cache in front of it (k5_long.c) */
        bl      SYM(toks_k5_long_neon)
        ldr     x16, [sp, #MISSAVE]
        add     x16, x16, #1
        str     x16, [sp, #MISSAVE]
        K5_RELOAD
        ldr     x7, [sp, #BKTSAVE]
        cbz     x7, 1f
        cmp     x0, #4
        b.hi    1f
        ldr     q1, [x24]                       /* pack: ids >= n zeroed, masked, count in ids[0] */
        ldr     q29, [x15, #256]
        dup     v2.4s, w0
        cmhi    v2.4s, v2.4s, v29.4s
        and     v1.16b, v1.16b, v2.16b
        and     v1.16b, v1.16b, v30.16b
        fmov    w16, s1
        orr     w16, w16, w0, lsl #TOKS_VAL_COUNT_SHIFT
        mov     v1.s[0], w16
        mov     x16, v1.d[1]                    /* the tag in ids[2] / ids[3] (layout.h) */
        orr     x16, x16, x27
        mov     v1.d[1], x16
        ldr     q0, [sp, #KEYSAVE]
        ldr     q2, [x7]
        ldr     q3, [x7, #32]
        str     q2, [x7, #16]
        str     q3, [x7, #48]
        str     q0, [x7]
        str     q1, [x7, #32]
1:      add     x24, x24, x0, lsl #2
        b       LBL(k5_next)

LBL(k5_shit):
        ldr     x17, [x3]
        lsr     x17, x17, #24                   /* id0 | id1 << 20 (id1 = TOKS_W3_ONE: one id) */
        ubfx    x16, x17, #0, #20               /* id0 */
        lsr     x17, x17, #20                   /* id1 */
        add     w8, w17, #1
        tst     w8, #(TOKS_W3_ONE + 1)          /* id1 == TOKS_W3_ONE */
        csel    w17, wzr, w17, ne               /* one id: the second stored as 0 */
        cset    w8, eq
        add     w8, w8, #1                      /* the count */
        stp     w16, w17, [x24]
        add     x24, x24, x8, lsl #2
        cbz     x28, LBL(k5_next)
        tbz     x9, #63, LBL(k5_next)           /* a fresh scratch keeps static answers out */
        orr     w16, w16, w8, lsl #TOKS_VAL_COUNT_SHIFT
        orr     x16, x16, x17, lsl #32          /* the static answer as a cache value, tagged (kernels.md §6) */
        fmov    d1, x16
        mov     v1.d[1], x27
        ldr     q2, [x7]
        ldr     q3, [x7, #32]
        str     q2, [x7, #16]
        str     q3, [x7, #48]
        stp     x4, x5, [x7]
        str     q1, [x7, #32]
        b       LBL(k5_next)

LBL(k5_one):
        ldrb    w17, [x2]
        ldr     w17, [x12, x17, lsl #2]
        str     w17, [x24], #4
        b       LBL(k5_next)

        /* the piece starts < 16 bytes before the segment's end: the key from loads inside the piece */
LBL(k5_tail):
        cmp     x1, #8
        b.lo    1f
        ldr     x4, [x2]
        add     x16, x2, x1
        ldur    x5, [x16, #-8]
        lsr     x5, x5, #8
        mov     w16, #120
        sub     w16, w16, w1, lsl #3
        lsr     x5, x5, x16                     /* bytes 8 .. len - 1 */
        b       2f
1:      LOAD_LE 4, 2, 1, 16, 17
        mov     x5, #0
2:      orr     x5, x5, x1, lsl #56             /* byte 15 = len */
        b       LBL(k5_key)

LBL(k5_end):
        ldr     x27, [sp, #MISSAVE]
        ldr     x0, [x20, #K5_OUT]
        sub     x0, x24, x0
        lsr     x0, x0, #2
        ldr     x1, [x20, #K5_N]
        sub     x1, x1, x26
        sub     x1, x1, x27
        str     x0, [x20, #K5_N_OUT]
        ldp     x2, x3, [x20, #K5_HITS_STATIC]  /* the counters accumulate (kernels.md §6) */
        ldr     x4, [x20, #K5_MISSES]
        add     x1, x1, x2
        add     x26, x26, x3
        add     x27, x27, x4
        stp     x1, x26, [x20, #K5_HITS_STATIC]
        str     x27, [x20, #K5_MISSES]
        EPILOGUE 5, 0, 128
ENDFUNC toks_k5_encode_neon
