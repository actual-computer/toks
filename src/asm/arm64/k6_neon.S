/* k6_neon.S: K6, bpe of one piece, tier neon (docs/kernels.md §5; the c twin is src/core/k6_c.c).
 *
 *   uint64_t toks_k6_bpe_neon(const toks_tables *t, toks_k6_args *a)     returns n_out
 *
 * Same answer, n_out and merges as the c twin on every input: hf BPE::tokenize.
 *  - TOKS_TF_IGNORE_MERGES and the piece is in vhash (crc32cx; kernels.md 5: under TOKS_TF_PROBE_LONG only a
 *    piece over 15 bytes, under TOKS_TF_PROBE_ASCII only one that ends in an ascii byte): [that id].
 *  - a run, p[1, len) one ascii byte c, where neither (p[0], c) nor (c, c) merges (bytepair[]): the byte ids, as
 *    round one finds no pair (out of line behind p[1] == p[len - 1] < 0x80; kernels.md 5 "Runs").
 *  - otherwise symbols = byte2id[b]; merge the adjacent pair of lowest prio, leftmost on ties, until no pair
 *    merges (prio = the merged id under TOKS_TF_IDS_AS_RANK, else the rank and the id is rank2id[prio]).
 *
 * Every symbol is a record at its first byte's position i (16 bytes); a pair at a position only changes when
 * that position or its right neighbour merges, and its prio is then probed once (round one reads
 * bytepair[]): no pair is probed twice, unlike the c twin, which re-probes every popped heap entry to drop
 * stale ones.
 *   short (len <= SHORT_MAX)  records {ID, NID = ID[NXT], PID = ID[PRV], NXT u16 (>= len: none), PRV u16
 *                             (0xFFFF: none)}; KEY[i] = prio << 10 | i (NONE: no pair) in a separate array
 *                             padded to 64-byte blocks, so the next merge is the unsigned min of KEY: hf's
 *                             (rank, pos) order. Per merge the KEYs of r (merged away), pos and l go to NONE,
 *                             the least of the rest is taken in neon while the two new pairs are probed (each
 *                             behind the pair filter, PFILT: most new pairs have no merge), and the next merge
 *                             is the least of the three: the chain is decode, two record loads, one probe. work: KEY 4 * L16 bytes (L16 = len rounded up to 16), then
 *                             the records 16 len; <= 256 + 32 len.
 *   long                      records {ID, NXT, PRV, KEY = the prio of (i, NXT[i])}; a binary min-heap of
 *                             prio << 32 | i (the c twin's), a popped entry applies only when it equals its
 *                             position's KEY; the heap holds <= 2 (len - 1) entries. work: 32 len.
 * The merge table probe (layout.h: home bucket (key * TOKS_FIB64) >> merge_shift) reads one 64-byte bucket
 * with four ldp: xor with key << 22 turns the hit slot into its prio (< 2^22) and every other slot into
 * >= 2^22, so an unsigned min tree finds it; a bucket whose slot 7 is empty is not full (the builder fills
 * slots in order and never deletes), which ends the probe. Scalar on purpose: the merge loop is a chain of
 * dependent probes, and this is ~15 cycles shorter than a neon reduction. A found prio prefetches
 * rank2id[prio], the next load on that chain when the pair merges.
 * Measured on spark (A725) and kept out (+-6%): prefetching the buckets that round-one merges, or a found
 * pair's own merge, will probe; a scalar rest-scan for <= 8 bytes; deferring the ignore_merges vhash probe
 * behind round one.
 *
 * The piece is read in place (round one's bytes, the vhash compare); the work area holds only merge state.
 * Lineage: hf word.rs (the (rank, pos) order; the heap with stale entries for long pieces), tiktoken and
 * gigatoken's short cores (a rank per position, min scan, refresh the two neighbours), e's tok v1 (the packed
 * prio << k | pos key, the linked array, bytepair round one, 8-slot 64-byte buckets homed by Fibonacci
 * hashing); new here: the rest-scan overlapped with the probes, the scalar xor/min-tree probe, NID/PID.
 *
 * A leaf: x0-x17, v0-v7, v29 (a probe count), v30 (t), v31 (a); no stack.
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

#define SHORT_MAX 128

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

/* merge table registers: x5 slots, x7 shift, x8 TOKS_FIB64, x9 rank2id (0 when ids are ranks) */
.macro MERGE_PARAMS
        ldr     x5, [x0, #TT_MERGE_SLOTS]
        ldr     x7, [x0, #TT_MERGE_SHIFT]
        movz    x8, #0x7C15
        movk    x8, #0x7F4A, lsl #16
        movk    x8, #0x79B9, lsl #32
        movk    x8, #0x9E37, lsl #48
        ldr     w9, [x0, #TT_FLAGS]
        ldr     x11, [x0, #TT_RANK2ID]
        tst     w9, #TOKS_TF_IDS_AS_RANK
        csel    x9, xzr, x11, ne
.endm

/* BUCKET b, k, K, acc, t0, t1: the bucket at x<b>, x<K> = key << 22: w<acc> = the key's prio, or NONE;
 * flags eq when that ends the probe (found, or slot 7 empty). x<k> is a temp. */
.macro BUCKET b, k, K, acc, t0, t1
        ldp     x\t0, x\t1, [x\b]
        eor     x\t0, x\t0, x\K
        eor     x\t1, x\t1, x\K
        cmp     x\t0, x\t1
        csel    x\acc, x\t0, x\t1, lo
        ldp     x\t0, x\t1, [x\b, #16]
        eor     x\t0, x\t0, x\K
        eor     x\t1, x\t1, x\K
        cmp     x\t0, x\t1
        csel    x\k, x\t0, x\t1, lo
        cmp     x\acc, x\k
        csel    x\acc, x\acc, x\k, lo
        ldp     x\t0, x\t1, [x\b, #32]
        eor     x\t0, x\t0, x\K
        eor     x\t1, x\t1, x\K
        cmp     x\t0, x\t1
        csel    x\k, x\t0, x\t1, lo
        ldp     x\t0, x\t1, [x\b, #48]
        eor     x\t0, x\t0, x\K
        cmp     x\k, x\t0
        csel    x\k, x\k, x\t0, lo
        eor     x\t0, x\t1, x\K
        cmp     x\k, x\t0
        csel    x\k, x\k, x\t0, lo
        cmp     x\acc, x\k
        csel    x\acc, x\acc, x\k, lo
        cmp     x\acc, #0x400, lsl #12
        csinv   w\acc, w\acc, wzr, lo
        ccmn    x\t1, #1, #4, hs
.endm

/* PROBE k, b, K, acc, t0, t1: w<acc> = the prio of the pair key x<k> (left << 21 | right), or NONE (bpe.h's
 * bpe_merge_find: the home bucket, then the next ones, at most merge_maxprobe). x<k> is clobbered. */
.macro PROBE k, b, K, acc, t0, t1
        mul     x\b, x\k, x8
        lsl     x\K, x\k, #22
        lsr     x\b, x\b, x7
        add     x\b, x5, x\b, lsl #6
        BUCKET  \b, \k, \K, \acc, \t0, \t1
        b.eq    LBL(pr_done\@)
        fmov    x\k, d30
        ldr     x\k, [x\k, #TT_MERGE_MAXPROBE]
LBL(pr_next\@):
        subs    x\k, x\k, #1
        b.eq    LBL(pr_done\@)                    /* maxprobe buckets: absent (w<acc> = NONE) */
        fmov    d29, x\k                          /* the count, across BUCKET */
        fmov    x\t0, d30
        ldr     x\t0, [x\t0, #TT_MERGE_MASK]
        sub     x\b, x\b, x5
        lsr     x\b, x\b, #6
        add     x\b, x\b, #1
        and     x\b, x\b, x\t0
        add     x\b, x5, x\b, lsl #6
        BUCKET  \b, \k, \K, \acc, \t0, \t1
        fmov    x\k, d29
        b.ne    LBL(pr_next\@)
LBL(pr_done\@):
.endm

/* RPREF r, t: prefetch rank2id[w<r>] when w<r> is a prio (rank2id path only) */
.macro RPREF r, t
        cbz     x9, LBL(rp_done\@)
        cmn     w\r, #1
        csel    w\t, wzr, w\r, eq
        add     x\t, x9, x\t, lsl #2
        prfm    pldl1keep, [x\t]
LBL(rp_done\@):
.endm

/* PFILT k, t0, t1, t2, absent: branch to absent when the pair filter (d27; bpe.h bpe_pf_bits) lacks a bit of the
 * key x<k>: the pair has no merge. x<k> is kept. */
.macro PFILT k, t0, t1, t2, absent
        mul     x\t0, x\k, x8
        lsr     x\t1, x\t0, x7
        lsr     x\t1, x\t1, #1                   /* the word */
        fmov    x\t2, d27
        ldr     x\t2, [x\t2, x\t1, lsl #3]
        lsr     x\t1, x\t0, #20                  /* the bits: hash bits 20-25 and 26-31 */
        lsr     x\t0, x\t0, #26
        lsr     x\t1, x\t2, x\t1
        lsr     x\t2, x\t2, x\t0
        and     x\t0, x\t1, x\t2
        tbz     x\t0, #0, \absent
.endm

/* SCAN r: w<r> = the least KEY (short path: x4 KEY, x10 blocks of 16; v0-v7, x14, x17) */
.macro SCAN r
        cmp     x2, #8
        b.hi    LBL(sc_v\@)
        ldp     q0, q1, [x4]                    /* <= 8 bytes: KEY[0, 8) */
        umin    v0.4s, v0.4s, v1.4s
        b       LBL(sc_r\@)
LBL(sc_v\@):
        mov     x17, x4
        ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x17], #64
        subs    x14, x10, #1
        b.eq    LBL(sc_t\@)
LBL(sc_l\@):
        ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x17], #64
        umin    v0.4s, v0.4s, v4.4s
        umin    v1.4s, v1.4s, v5.4s
        umin    v2.4s, v2.4s, v6.4s
        umin    v3.4s, v3.4s, v7.4s
        subs    x14, x14, #1
        b.ne    LBL(sc_l\@)
LBL(sc_t\@):
        umin    v0.4s, v0.4s, v1.4s
        umin    v2.4s, v2.4s, v3.4s
        umin    v0.4s, v0.4s, v2.4s
LBL(sc_r\@):
        uminv   s0, v0.4s
        fmov    w\r, s0
.endm

/* HPUSH e, n, i, p, v: push x<e> on the heap at x4 holding x<n> entries (the c twin's heap_push) */
.macro HPUSH e, n, i, p, v
        mov     x\i, x\n
        add     x\n, x\n, #1
LBL(hp_up\@):
        cbz     x\i, LBL(hp_put\@)
        sub     x\p, x\i, #1
        lsr     x\p, x\p, #1
        ldr     x\v, [x4, x\p, lsl #3]
        cmp     x\v, x\e
        b.ls    LBL(hp_put\@)
        str     x\v, [x4, x\i, lsl #3]
        mov     x\i, x\p
        b       LBL(hp_up\@)
LBL(hp_put\@):
        str     x\e, [x4, x\i, lsl #3]
.endm


FUNC toks_k6_bpe_neon
        LEAF
        ldp     x10, x2, [x1, #K6_PIECE]        /* x10 piece, x2 len */
        cbz     x2, LBL(k6_empty)
        fmov    d30, x0
        fmov    d31, x1
        ldr     w11, [x0, #TT_FLAGS]
        tbz     w11, #0, LBL(k6_bpe)               /* TOKS_TF_IGNORE_MERGES */
        tbz     w11, #7, LBL(k6_vh)                /* TOKS_TF_PROBE_LONG: a piece over TOKS_KEY_MAXLEN bytes only, */
        cmp     x2, #TOKS_KEY_MAXLEN
        b.ls    LBL(k6_bpe)
        tbz     w11, #8, LBL(k6_vh)                /* TOKS_TF_PROBE_ASCII: and ending in ascii (vhash's tokens are */
        add     x12, x10, x2                       /* all-ascii then) */
        ldurb   w12, [x12, #-1]
        tbnz    w12, #7, LBL(k6_bpe)
LBL(k6_vh):
        ldr     x12, [x0, #TT_VHASH]
        cbz     x12, LBL(k6_bpe)
        ldr     x14, [x0, #TT_VHASH_MASK]       /* after the slots, u32 [256]: the longest token by its first byte */
        add     x15, x12, x14, lsl #3
        ldrb    w16, [x10]
        add     x15, x15, #8
        ldr     w16, [x15, x16, lsl #2]
        cmp     x2, x16
        b.hi    LBL(k6_bpe)                     /* longer than any token that starts so (zh runs > 18 B on llama 3) */

        /* ---- vhash: h = TOKS_VSEED ^ len, crc32c over the 8-byte words, the last one zero-padded */
        movz    w13, #0xCA6B
        movk    w13, #0x85EB, lsl #16
        eor     w13, w13, w2
        mov     x14, x10
        and     x15, x2, #~7
        add     x15, x10, x15
        b       2f
1:      ldr     x16, [x14], #8
        crc32cx w13, w13, x16
2:      cmp     x14, x15
        b.lo    1b
        ands    x16, x2, #7
        b.eq    4f
        cmp     x2, #8
        b.lo    3f
        add     x17, x10, x2
        ldur    x17, [x17, #-8]                 /* the last 8 bytes; keep the top r */
        neg     w16, w16, lsl #3
        lsr     x17, x17, x16
        b       35f
3:      LOAD_LE 17, 10, 2, 14, 16
35:     crc32cx w13, w13, x17
4:      ldr     x14, [x0, #TT_VHASH_MASK]       /* four slots from the home slot, without a branch: the first that is */
        and     x15, x13, x14                   /* empty (a miss) or holds h (a candidate) decides. Their line may */
        add     x16, x15, #1                    /* come from DRAM, and a branch per slot (empty? h?) mispredicted about */
        and     x16, x16, x14                   /* half the time, each miss waiting for the line (llama 3 zh 16..30 B */
        add     x17, x15, #2                    /* cold: ~19 of 111 ns); one branch on the outcome predicts "not a */
        and     x17, x17, x14                   /* token", and bpe runs on while the line comes */
        add     x3, x15, #3
        and     x3, x3, x14
        ldr     x4, [x12, x15, lsl #3]
        ldr     x5, [x12, x16, lsl #3]
        ldr     x6, [x12, x17, lsl #3]
        ldr     x7, [x12, x3, lsl #3]
        add     x9, x15, #4                     /* x9: where the probe goes on (the fifth slot, or the candidate) */
        and     x9, x9, x14
        mov     w8, #0                          /* w8: the deciding slot is empty */
        mov     w11, #0                         /* w11: one of the four decides; the first that does wins (the */
        cmp     w7, w13                         /* fourth slot first, the first slot last) */
        ccmn    x7, #1, #4, ne
        csel    x9, x3, x9, eq
        csinc   w11, w11, wzr, ne
        csel    w8, wzr, w8, eq
        cmn     x7, #1
        csinc   w8, w8, wzr, ne
        cmp     w6, w13
        ccmn    x6, #1, #4, ne
        csel    x9, x17, x9, eq
        csinc   w11, w11, wzr, ne
        csel    w8, wzr, w8, eq
        cmn     x6, #1
        csinc   w8, w8, wzr, ne
        cmp     w5, w13
        ccmn    x5, #1, #4, ne
        csel    x9, x16, x9, eq
        csinc   w11, w11, wzr, ne
        csel    w8, wzr, w8, eq
        cmn     x5, #1
        csinc   w8, w8, wzr, ne
        cmp     w4, w13
        ccmn    x4, #1, #4, ne
        csel    x9, x15, x9, eq
        csinc   w11, w11, wzr, ne
        csel    w8, wzr, w8, eq
        cmn     x4, #1
        csinc   w8, w8, wzr, ne
        add     x16, x14, #1                    /* bound: the slots */
        mov     x15, x9
        cbz     w11, 5f                         /* none of the four decides: probe on (rare) */
        cbnz    w8, LBL(k6_bpe)                 /* an empty slot decides: not a vocabulary token (the predicted way) */
5:      ldr     x17, [x12, x15, lsl #3]
        cmn     x17, #1
        b.eq    LBL(k6_bpe)                        /* empty: not a vocabulary token */
        cmp     w17, w13
        b.eq    7f
6:      add     x15, x15, #1
        and     x15, x15, x14
        subs    x16, x16, #1
        b.ne    5b
        b       LBL(k6_bpe)
7:      lsr     x3, x17, #32                    /* id: equal length and bytes? */
        ldr     x4, [x0, #TT_TOK_OFF]
        add     x4, x4, x3, lsl #2
        ldp     w5, w6, [x4]
        sub     w6, w6, w5
        cmp     x6, x2
        b.ne    6b
        ldr     x4, [x0, #TT_TOK_BYTES]
        add     x4, x4, x5
        cmp     x2, #8
        b.lo    9f
        mov     x5, #0
8:      ldr     x6, [x4, x5]
        ldr     x7, [x10, x5]
        cmp     x6, x7
        b.ne    6b
        add     x5, x5, #8
        sub     x6, x2, x5
        cmp     x6, #8
        b.hs    8b
        cbz     x6, 10f
        sub     x5, x2, #8                      /* the last 8 bytes, overlapping */
        ldr     x6, [x4, x5]
        ldr     x7, [x10, x5]
        cmp     x6, x7
        b.ne    6b
        b       10f
9:      LOAD_LE 6, 4, 2, 7, 8
        LOAD_LE 7, 10, 2, 8, 9
        cmp     x6, x7
        b.ne    6b
10:     ldr     x4, [x1, #K6_OUT]
        str     w3, [x4]
        mov     x0, #1
        stp     x0, xzr, [x1, #K6_N_OUT]        /* n_out 1, merges 0 */
        ret

LBL(k6_empty):
        stp     xzr, xzr, [x1, #K6_N_OUT]
        mov     x0, #0
        ret

        /* ---- a run? (from LBL(k6_bpe) below: len >= 2, x14 = p + len, w15 = c = p[1] = p[len - 1] < 0x80):
         * (c, c) has no merge, p[1, len) is all c, (p[0], c) has no merge: the ids out, n_out = len, merges 0;
         * else bpe. It sits here, ahead of the bpe path, padded with dead nops so that with the test at
         * LBL(k6_bpe) the code after it moves by a multiple of 64 bytes and the bpe path keeps its alignment
         * (moved by 32 bytes, the same code cost llama 3 code 1.2% on m2ultra2; kernels.md 5 "Runs"). */
LBL(k6_run):
        ldr     x13, [x0, #TT_BYTEPAIR]
        orr     w9, w15, w15, lsl #8
        ldr     w9, [x13, x9, lsl #2]
        cmn     w9, #1
        b.ne    LBL(k6_merge)                   /* (c, c) merges: bpe (a model with run tokens) */
        mov     x16, #0x0101010101010101
        mul     x16, x16, x15                   /* c in every byte */
        add     x6, x10, #1
        sub     x17, x2, #1                     /* p[1, len): n bytes */
        cmp     x17, #8
        b.lo    LBL(r_short)
LBL(r_w8):
        ldr     x7, [x6], #8
        cmp     x7, x16
        b.ne    LBL(k6_merge)
        sub     x17, x17, #8
        cmp     x17, #8
        b.hs    LBL(r_w8)
        cbz     x17, LBL(r_p0)
        ldur    x7, [x14, #-8]                  /* the last 8 bytes (none is p[0]: len >= 9) */
        cmp     x7, x16
        b.ne    LBL(k6_merge)
        b       LBL(r_p0)
LBL(r_short):
        LOAD_LE 7, 6, 17, 8, 9                  /* n = 1..7 */
        lsl     w8, w17, #3
        mov     x9, #-1
        lsl     x9, x9, x8
        bic     x9, x16, x9
        cmp     x7, x9
        b.ne    LBL(k6_merge)
LBL(r_p0):
        ldrb    w7, [x10]                       /* p[0] */
        orr     w8, w15, w7, lsl #8
        ldr     w8, [x13, x8, lsl #2]
        cmn     w8, #1
        b.ne    LBL(k6_merge)                   /* (p[0], c) merges: bpe */
        ldr     x12, [x0, #TT_BYTE2ID]
        ldr     w8, [x12, x7, lsl #2]
        ldr     w9, [x12, x15, lsl #2]
        ldr     x11, [x1, #K6_OUT]
        str     w8, [x11]
        dup     v0.4s, w9
        add     x14, x11, #4
        sub     x17, x2, #1
        cmp     x17, #4
        b.lo    LBL(r_st1)
LBL(r_st4):
        str     q0, [x14], #16
        sub     x17, x17, #4
        cmp     x17, #4
        b.hs    LBL(r_st4)
        cbz     x17, LBL(r_done)
        add     x14, x11, x2, lsl #2
        stur    q0, [x14, #-16]                 /* the last 4 (len >= 5: not out[0]) */
        b       LBL(r_done)
LBL(r_st1):
        str     w9, [x14], #4
        subs    x17, x17, #1
        b.ne    LBL(r_st1)
LBL(r_done):
        stp     x2, xzr, [x1, #K6_N_OUT]        /* n_out len, merges 0 */
        mov     x0, x2
        ret
LBL(k6_run_end):
        .rept   ((LBL(k6_run) - LBL(k6_run_end) - 32) & 63) >> 2
        nop
        .endr

        /* ---- bpe. A piece whose bytes after the first are one ascii byte c is its byte ids when neither (p[0], c)
         * nor (c, c) merges: round one finds no pair (kernels.md 5 "Runs"). p[1] == p[len - 1] < 0x80: LBL(k6_run),
         * above. */
LBL(k6_bpe):
        cmp     x2, #2
        b.lo    LBL(k6_merge)
        add     x14, x10, x2
        ldrb    w16, [x10, #1]
        ldurb   w15, [x14, #-1]                 /* c */
        eor     w16, w16, w15
        orr     w16, w16, w15, lsr #7           /* 0: p[1] == c, and c < 0x80 */
        cbz     w16, LBL(k6_run)
LBL(k6_merge):
        fmov    d28, x2                         /* len, for merges = len - n_out */
        ldr     x11, [x1, #K6_WORK]
        ldr     x12, [x0, #TT_BYTE2ID]
        ldr     x13, [x0, #TT_BYTEPAIR]
        cmp     x2, #SHORT_MAX
        b.hi    LBL(k6_long)

        /* short: x4 KEY, x3 records {ID, NID, PID, NXT u16, PRV u16}, x10 KEY blocks of 64 bytes */
        add     x14, x2, #15
        lsr     x14, x14, #4
        mov     x4, x11
        add     x3, x11, x14, lsl #6
        movi    v0.2d, #0xffffffffffffffff
        movi    v1.2d, #0xffffffffffffffff
        movi    v2.2d, #0xffffffffffffffff
        movi    v3.2d, #0xffffffffffffffff
        mov     x15, x4
        mov     x16, x14
1:      st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x15], #64
        subs    x16, x16, #1
        b.ne    1b
        ldr     x1, [x0, #TT_PREMERGE]
        mov     w8, #0x100
        mov     w9, #0xC0
        cmp     x1, #0
        csel    w1, w8, w9, eq                  /* a byte >= w1 may start a premerged char (0x100: no table) */
        ldr     x8, [x0, #TT_APM]
        sub     x9, x2, #4
        cmp     x9, #(TOKS_KEY_MAXLEN - 4)
        ccmp    x8, #0, #4, ls
        b.eq    5f                              /* not 4..15 bytes, or no ascii table */
        ldrb    w9, [x10, #1]
        orr     w9, w9, #0x20
        sub     w9, w9, #0x61
        cmp     w9, #26
        csel    w1, wzr, w1, lo                 /* a letter second (a word, maybe after one mark): every byte may */
5:      ldrb    w15, [x10]                      /* b_i */
        cmp     w15, w1
        b.hs    LBL(s_pm)
        ldr     w5, [x12, x15, lsl #2]          /* ID of i */
        mov     w6, #0                          /* ID of i - 1 (none at 0) */
        mov     x16, #0                         /* i */
        mov     x17, x3                         /* &rec[i] */
2:      add     w7, w16, #1
        sub     w14, w16, #1
        orr     w14, w7, w14, lsl #16           /* NXT = i + 1, PRV = i - 1 (0xFFFF at 0) */
        cmp     x7, x2
        b.hs    3f
        ldrb    w8, [x10, x7]                   /* b_i+1 */
        cmp     w8, w1
        b.hs    LBL(s_pm)
        ldr     w9, [x12, x8, lsl #2]
        orr     w11, w8, w15, lsl #8
        ldr     w11, [x13, x11, lsl #2]         /* round one: bytepair[b_i << 8 | b_i+1] */
        stp     w5, w9, [x17]                   /* ID, NID */
        stp     w6, w14, [x17, #8]              /* PID, NXT | PRV */
        orr     w14, w16, w11, lsl #10
        cmn     w11, #1
        csinv   w14, w14, wzr, ne
        str     w14, [x4, x16, lsl #2]          /* KEY[i] = prio << 10 | i, or NONE */
        add     x17, x17, #16
        mov     w6, w5
        mov     w5, w9
        mov     w15, w8
        mov     x16, x7
        b       2b
3:      stp     w5, wzr, [x17]
        stp     w6, w14, [x17, #8]
        add     x10, x2, #15
        lsr     x10, x10, #4
        MERGE_PARAMS
        SCAN    11
        ldr     x1, [x0, #TT_PAIRF]
        fmov    d27, x1                         /* the pair filter (bpe.h bpe_pf_bits), for the loop's probes */

        /* one merge per turn; w11 = the least KEY. The KEYs of r (merged away), pos and l are set to NONE,
         * the least of the rest is taken in neon while the two new pairs are probed, and the next least KEY
         * is the least of the three. */
LBL(s_loop):
        cmn     w11, #1
        b.eq    LBL(s_out)                      /* no pair merges */
        and     w12, w11, #0x3FF                /* pos */
        lsr     w11, w11, #10                   /* prio */
        cbz     x9, 1f
        ldr     w11, [x9, x11, lsl #2]          /* m = the merged id */
1:      add     x13, x3, x12, lsl #4
        ldrh    w14, [x13, #12]                 /* r = NXT[pos] */
        ldrh    w15, [x13, #14]                 /* l = PRV[pos], 0xFFFF: none */
        ldr     w16, [x13, #8]                  /* ID[l] */
        add     x17, x3, x14, lsl #4
        ldr     w0, [x17, #4]                   /* ID[rn] = NID[r] */
        ldrh    w1, [x17, #12]                  /* rn = NXT[r] */
        mov     w6, #-1
        str     w6, [x4, x14, lsl #2]           /* KEY[r] = NONE */
        str     w6, [x4, x12, lsl #2]           /* KEY[pos] = NONE until probed */
        stp     w11, w0, [x13]                  /* ID[pos] = m, NID[pos] = ID[rn] */
        strh    w1, [x13, #12]                  /* NXT[pos] = rn */
        tbnz    w15, #15, 2f
        str     w6, [x4, x15, lsl #2]           /* KEY[l] = NONE until probed */
        add     x17, x3, x15, lsl #4
        str     w11, [x17, #4]                  /* NID[l] = m */
        orr     x16, x11, x16, lsl #21          /* (ID[l], m) */
2:      cmp     x1, x2
        b.hs    3f
        add     x17, x3, x1, lsl #4
        str     w11, [x17, #8]                  /* PID[rn] = m */
        strh    w12, [x17, #14]                 /* PRV[rn] = pos */
        orr     x0, x0, x11, lsl #21            /* (m, ID[rn]) */
3:      SCAN    13                              /* w13 = the least of the rest */
        mov     w14, #-1
        cmp     x1, x2
        b.hs    4f
        PFILT   0, 1, 6, 11, 4f                 /* w14 stays NONE */
        PROBE   0, 1, 6, 11, 14, 17
        orr     w14, w12, w11, lsl #10
        cmn     w11, #1
        csinv   w14, w14, wzr, ne
        str     w14, [x4, x12, lsl #2]          /* KEY[pos] */
        RPREF   11, 17
4:      mov     w12, #-1
        tbnz    w15, #15, 5f
        PFILT   16, 0, 1, 17, 5f
        PROBE   16, 0, 1, 6, 11, 17
        orr     w12, w15, w6, lsl #10
        cmn     w6, #1
        csinv   w12, w12, wzr, ne
        str     w12, [x4, x15, lsl #2]          /* KEY[l] */
        RPREF   6, 17
5:      cmp     w13, w14
        csel    w11, w13, w14, lo
        cmp     w11, w12
        csel    w11, w11, w12, lo
        b       LBL(s_loop)

LBL(s_out):
        fmov    x1, d31
        ldr     x11, [x1, #K6_OUT]
        mov     x12, #0
        mov     x13, #0
1:      add     x14, x3, x12, lsl #4
        ldr     w15, [x14]                      /* ID */
        ldrh    w12, [x14, #12]                 /* NXT */
        str     w15, [x11, x13, lsl #2]
        add     x13, x13, #1
        cmp     x12, x2
        b.lo    1b
        b       LBL(k6_done)

        /* ---- the alive chain from 0 to out; merges = len - n */
LBL(k6_out):
        fmov    x1, d31
        ldr     x11, [x1, #K6_OUT]
        mov     x12, #0
        mov     x13, #0
1:      add     x14, x3, x12, lsl #4
        ldp     w15, w12, [x14]                 /* ID, NXT */
        str     w15, [x11, x13, lsl #2]
        add     x13, x13, #1
        cmp     x12, x2
        b.lo    1b
LBL(k6_done):
        fmov    x14, d28
        sub     x14, x14, x13
        stp     x13, x14, [x1, #K6_N_OUT]
        mov     x0, x13
        ret

        /* ---- short with premerged chars (kernels.md §5.1): the byte init above met a byte >= 0xC0 and the
         * premerge table exists. KEY is NONE-filled for len; the byte init wrote KEY[0, i) only for bytes
         * before that lead, all below n - 1, so every one is rewritten. 1: the symbols into SYM (after the
         * records, 16 len): a char whose entry is set and whose neighbour bytes are not risky as its token,
         * every other byte as byte2id. 2: the records from SYM. 3: KEY from a probe of every pair (round one
         * has no byte pair table for tokens). Then the merge loop runs on n = x2 symbols. */
LBL(s_pm):
        ldr     x9, [x0, #TT_PREMERGE]
        ADRP_ADD x12, SYM(toks_k6_neon_pmbits)  /* risk bit of the byte before [256], after [256] */
        ldr     x13, [x0, #TT_BYTE2ID]
        ldr     x0, [x0, #TT_APM]               /* the ascii table (layout.h TOKS_APM_*) for a word routed here */
        cmp     w1, #0                          /* from byte 0 (no KEY written), else 0: none */
        csel    x0, x0, xzr, eq
        add     x17, x3, x2, lsl #4             /* SYM */
        mov     x15, #0                         /* i */
        mov     x16, #0                         /* n */
1:      ldrb    w5, [x10, x15]
        cmp     w5, #0xC0
        b.lo    40f
        cbz     x9, 7f
        sub     x6, x2, x15                     /* bytes left */
        add     x7, x10, x15
        cmp     w5, #0xE0
        b.hs    2f
        cmp     x6, #2                          /* C0..DF + a continuation byte */
        b.lo    7f
        ldrb    w14, [x7, #1]
        eor     w14, w14, #0x80                 /* b1 & 63 when b1 is 80..BF, else >= 0x40 */
        cmp     w14, #0x40
        b.hs    7f
        sub     w11, w5, #0xC0                  /* s1 */
        mov     x1, #2
        b       3f
2:      cmp     w5, #0xF0                       /* E0..EF + two continuation bytes */
        b.hs    7f
        cmp     x6, #3
        b.lo    7f
        ldrb    w8, [x7, #1]
        eor     w8, w8, #0x80
        cmp     w8, #0x40
        b.hs    7f
        ldrb    w14, [x7, #2]
        eor     w14, w14, #0x80
        cmp     w14, #0x40
        b.hs    7f
        and     w11, w5, #15
        orr     w11, w8, w11, lsl #6
        add     w11, w11, #32                   /* s1 */
        mov     x1, #3
3:      ldr     w11, [x9, x11, lsl #2]          /* the block's offset */
        add     x11, x9, x11
        ldr     x11, [x11, x14, lsl #3]         /* the entry (0: none) */
        cbz     x11, 7f
        mov     w6, #TOKS_PM_NEVER
        cbz     x15, 4f
        ldurb   w6, [x7, #-1]
        ldrb    w6, [x12, x6]                   /* the byte before's risk bit */
4:      add     x8, x15, x1
        mov     w14, #TOKS_PM_NEVER
        cmp     x8, x2
        b.hs    5f
        ldrb    w14, [x10, x8]
        add     x14, x14, #256
        ldrb    w14, [x12, x14]                 /* the byte after's */
5:      lsr     x6, x11, x6
        lsr     x14, x11, x14
        orr     x6, x6, x14
        tbnz    x6, #0, 7f                      /* risky: bytes */
        and     w11, w11, #TOKS_ID_MASK
        str     w11, [x17, x16, lsl #2]
        add     x6, x17, x2, lsl #2
        mov     w14, #0x300
        strh    w14, [x6, x16, lsl #1]          /* BYT[n] = a char's token */
        add     x16, x16, #1
        add     x15, x15, x1
        b       8f
40:     cmp     w5, #0x80                       /* ascii: a certified bigram, else the byte */
        b.hs    7f
        cbz     x0, 7f
        add     x6, x15, #2
        cmp     x6, x2
        b.hi    7f
        add     x7, x10, x15
        ldrb    w14, [x7, #1]
        tbnz    w14, #7, 7f
        orr     w14, w14, w5, lsl #7
        add     x8, x0, x14, lsl #4
        add     x8, x8, #TOKS_APM_PAIRS
        ldr     w14, [x8]                       /* id (NONE: none) */
        cmn     w14, #1
        b.eq    7f
        cbz     x15, 41f
        ldurb   w1, [x7, #-1]
        ldrb    w1, [x0, x1]
        ldr     w11, [x8, #8]
        lsr     w11, w11, w1
        tbnz    w11, #0, 7f                     /* the byte before is risky */
41:     cmp     x6, x2
        b.hs    42f
        ldrb    w1, [x7, #2]
        ldrb    w1, [x0, x1]
        ldr     w11, [x8, #4]
        lsr     w11, w11, w1
        tbnz    w11, #0, 7f                     /* the byte after is */
42:     str     w14, [x17, x16, lsl #2]
        add     x6, x17, x2, lsl #2
        mov     w14, #0x100
        strh    w14, [x6, x16, lsl #1]          /* BYT[n] = a token */
        add     x16, x16, #1
        add     x15, x15, #2
        b       8f
7:      add     x6, x17, x2, lsl #2
        strh    w5, [x6, x16, lsl #1]           /* BYT[n] = the byte */
        ldr     w5, [x13, x5, lsl #2]           /* a byte */
        str     w5, [x17, x16, lsl #2]
        add     x16, x16, #1
        add     x15, x15, #1
8:      cmp     x15, x2
        b.lo    1b
LBL(s_recs):                                    /* toks_k6_merge_neon enters here */
        mov     x15, #0                         /* the records */
        mov     w6, #0
        mov     x7, x3
        sub     x8, x16, #1
9:      ldr     w5, [x17, x15, lsl #2]          /* ID */
        add     w14, w15, #1
        mov     w11, #0
        cmp     x15, x8
        b.hs    10f
        ldr     w11, [x17, x14, lsl #2]         /* NID */
10:     sub     w1, w15, #1
        orr     w14, w14, w1, lsl #16           /* NXT = k + 1, PRV = k - 1 (0xFFFF at 0) */
        stp     w5, w11, [x7]
        stp     w6, w14, [x7, #8]
        mov     w6, w5
        add     x7, x7, #16
        add     x15, x15, #1
        cmp     x15, x16
        b.lo    9b
        fmov    x0, d30
        ldr     x10, [x0, #TT_BYTEPAIR]
        ldr     x12, [x0, #TT_PAIRF]
        fmov    d27, x12                        /* the pair filter (bpe.h bpe_pf_bits) */
        MERGE_PARAMS
        sub     x13, x16, #1                    /* pairs */
        mov     x15, #0
11:     cmp     x15, x13
        b.hs    12f
        add     x1, x17, x2, lsl #2
        add     x1, x1, x15, lsl #1
        ldrh    w0, [x1]
        ldrh    w1, [x1, #2]
        orr     w12, w0, w1
        tbnz    w12, #8, 13f
        orr     w0, w1, w0, lsl #8
        ldr     w11, [x10, x0, lsl #2]          /* two bytes: bytepair */
        b       14f
13:     add     x1, x17, x15, lsl #2
        ldp     w0, w1, [x1]
        orr     x0, x1, x0, lsl #21             /* (SYM[k], SYM[k + 1]) */
        tbz     w12, #9, 15f                    /* a char's token in it: the filter first (most such pairs */
        mov     w11, #-1                        /* have no merge) */
        PFILT   0, 1, 14, 12, 14f
15:     PROBE   0, 1, 6, 11, 12, 14
14:     orr     w12, w15, w11, lsl #10
        cmn     w11, #1
        csinv   w12, w12, wzr, ne
        str     w12, [x4, x15, lsl #2]          /* KEY[k] */
        add     x15, x15, #1
        b       11b
12:     mov     x2, x16                         /* the merge loop counts symbols */
        add     x10, x2, #15
        lsr     x10, x10, #4
        SCAN    11
        b       LBL(s_loop)

        /* ---- long: x4 heap, x3 records {ID, NXT, PRV, KEY}, x10 heap entries */
LBL(k6_long):
        mov     x4, x11
        add     x3, x11, x2, lsl #4
        mov     x11, #0
        ldrb    w15, [x10]
        mov     x16, #0
        mov     x17, x3
1:      ldr     w5, [x12, x15, lsl #2]
        add     w6, w16, #1
        sub     w7, w16, #1
        stp     w5, w6, [x17]
        str     w7, [x17, #8]
        cmp     x6, x2
        b.hs    3f
        ldrb    w7, [x10, x6]
        orr     w8, w7, w15, lsl #8
        ldr     w8, [x13, x8, lsl #2]
        str     w8, [x17, #12]                  /* KEY = prio or NONE */
        cmn     w8, #1
        b.eq    2f
        orr     x9, x16, x8, lsl #32
        HPUSH   9, 11, 5, 8, 14
2:      add     x17, x17, #16
        mov     w15, w7
        mov     x16, x6
        b       1b
3:      mov     w8, #-1
        str     w8, [x17, #12]
        mov     x10, x11
        MERGE_PARAMS

LBL(l_loop):
        cbz     x10, LBL(k6_out)
        ldr     x11, [x4]                       /* pop (the c twin's heap_pop) */
        sub     x10, x10, #1
        ldr     x12, [x4, x10, lsl #3]
        mov     x13, #0
1:      lsl     x14, x13, #1
        add     x14, x14, #1
        cmp     x14, x10
        b.hs    3f
        ldr     x15, [x4, x14, lsl #3]
        add     x16, x14, #1
        cmp     x16, x10
        b.hs    2f
        ldr     x17, [x4, x16, lsl #3]
        cmp     x17, x15
        csel    x14, x16, x14, lo
        csel    x15, x17, x15, lo
2:      cmp     x15, x12
        b.hs    3f
        str     x15, [x4, x13, lsl #3]
        mov     x13, x14
        b       1b
3:      str     x12, [x4, x13, lsl #3]
        mov     w12, w11                        /* pos */
        lsr     x11, x11, #32                   /* prio */
        add     x13, x3, x12, lsl #4
        ldr     w14, [x13, #12]
        cmp     w14, w11
        b.ne    LBL(l_loop)                        /* stale: the pair at pos changed */
        cbz     x9, 4f
        ldr     w11, [x9, x11, lsl #2]
4:      ldp     w14, w15, [x13, #4]             /* r, l */
        add     x16, x3, x14, lsl #4
        mov     w17, #-1
        str     w17, [x16, #12]                 /* KEY[r] = NONE */
        ldr     w14, [x16, #4]                  /* rn */
        stp     w11, w14, [x13]
        cmp     x14, x2
        b.hs    5f
        add     x16, x3, x14, lsl #4
        str     w12, [x16, #8]
        ldr     w16, [x16]
        orr     x16, x16, x11, lsl #21
        PROBE   16, 17, 0, 1, 6, 14
        str     w1, [x13, #12]
        cmn     w1, #1
        b.eq    6f
        RPREF   1, 14
        orr     x16, x12, x1, lsl #32
        HPUSH   16, 10, 17, 0, 6
        b       6f
5:      mov     w17, #-1
        str     w17, [x13, #12]
6:      tbnz    w15, #31, LBL(l_loop)
        add     x16, x3, x15, lsl #4
        ldr     w17, [x16]
        orr     x17, x11, x17, lsl #21
        PROBE   17, 0, 1, 6, 12, 13
        str     w6, [x16, #12]
        cmn     w6, #1
        b.eq    LBL(l_loop)
        RPREF   6, 12
        orr     x17, x15, x6, lsl #32
        HPUSH   17, 10, 0, 1, 12
        b       LBL(l_loop)
ENDFUNC toks_k6_bpe_neon

/*   uint64_t toks_k6_merge_neon(const toks_tables *t, toks_k6_args *a)     returns n_out
 * K6's merge loop on given initial symbols (spm_c.c's model; the c twin is k6_c.c's toks_k6_merge): a->piece holds
 * n = a->len u32 symbols, 2 <= n <= SHORT_MAX, 4-aligned, with room for 2 n bytes after them (the BYT array, every
 * symbol a char's token: each pair probed, the filter first); a->out, a->work, n_out and merges as K6's. */
FUNC toks_k6_merge_neon
        LEAF
        ldp     x17, x2, [x1, #K6_PIECE]        /* x17 SYM, x2 n */
        fmov    d30, x0
        fmov    d31, x1
        fmov    d28, x2
        ldr     x11, [x1, #K6_WORK]
        add     x14, x2, #15
        lsr     x14, x14, #4
        mov     x4, x11                         /* KEY (NONE-filled), then the records */
        add     x3, x11, x14, lsl #6
        movi    v0.2d, #0xffffffffffffffff
        movi    v1.2d, #0xffffffffffffffff
        movi    v2.2d, #0xffffffffffffffff
        movi    v3.2d, #0xffffffffffffffff
1:      st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x11], #64
        subs    x14, x14, #1
        b.ne    1b
        add     x6, x17, x2, lsl #2
        mov     w5, #0x300
        mov     x15, #0
2:      strh    w5, [x6, x15, lsl #1]           /* BYT[k] = a char's token */
        add     x15, x15, #1
        cmp     x15, x2
        b.lo    2b
        mov     x16, x2
        b       LBL(s_recs)
ENDFUNC toks_k6_merge_neon

        RODATA
DATA toks_k6_neon_pmbits                        /* layout.h: the premerge risk bit of a neighbour byte */
        .rept   32                              /* before: 00..1F */
        .byte   57
        .endr
        .byte   56                              /* 20 */
        .rept   95                              /* 21..7F */
        .byte   57
        .endr
        .rept   4                               /* 80..BF: 40 + (b & 15) */
        .byte   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
        .endr
        .rept   64                              /* C0..FF */
        .byte   63
        .endr
        .rept   128                             /* after: 00..7F */
        .byte   58
        .endr
        .rept   64                              /* 80..BF */
        .byte   63
        .endr
        .rept   32                              /* C0..DF */
        .byte   59
        .endr
        .byte   24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39   /* E0..EF */
        .rept   16                              /* F0..FF */
        .byte   60
        .endr
        TEXT
