/*
 * src/asm/x86_64/k6_avx2.S: K6, bpe of one piece, avx2 tier (docs/kernels.md §5, layout.h K6_*).
 *
 *   uint64_t toks_k6_bpe_avx2(const toks_tables *t, toks_k6_args *a)
 *
 * Byte-identical to the c twin src/core/k6_c.c (out[0, n_out), n_out, merges) on every input:
 *   - TOKS_TF_IGNORE_MERGES and the piece is in vhash (kernels.md §5: under TOKS_TF_PROBE_LONG only a
 *     piece over 15 bytes, under TOKS_TF_PROBE_ASCII only one that ends in an ascii byte): [that id];
 *   - otherwise merge the adjacent pair of lowest prio, leftmost on ties, until no pair merges.
 * Every merge removes one symbol, so merges = len - n_out on every path.
 *
 * short (2 <= len <= SMAX): a linked array; rk[i] = prio << 8 | i for the pair starting at symbol i
 *   (>= RK_NONE when there is none: prio NONE, a dead symbol, the last one), mid[i] = that pair's
 *   merged id (prio itself, or rank2id[prio]), filled when rk[i] is, off the merge chain. A round
 *   merges the pair at min(rk). Its merge changes rk at three symbols only (pos, its dead right
 *   neighbour, its left neighbour pv), so those three are set to none first and the next round's
 *   min scan over the array runs beside the two probes of the new pairs (pv, pos) and (pos, nn);
 *   their results are folded into that min with two cmovs. Scan: up to 16 bytes two cmov chains over
 *   dword loads (they forward from the dword stores); above, vpminud over 32-byte rows (a ymm load
 *   over the fresh dword stores waits for them to retire, hidden behind the probes).
 * long (len > SMAX): a min-heap of prio << 32 | pos over per-position records {id, next, prev, cur},
 *   cur = the prio of the pair now starting there (NONE when dead, last or unmergeable): a popped
 *   entry is current iff cur[pos] == prio (the pair at a position only grows and a prio names one
 *   pair), so a stale entry costs no probe (the c twin re-probes).
 * MPROBE, the merge-table probe (layout.h): home bucket (key * FIB64) >> merge_shift, two ymm
 *   compares of the eight slots' key fields against key << 22; a bucket whose slot 7 is empty ends
 *   the probe (the builder fills slots front to back and never deletes), else the next bucket,
 *   wrapping, at most merge_maxprobe buckets.
 * Work area (64-aligned, stays inside TOKS_BPE_WORK_BYTES(len) = 256 + 32 len):
 *   short  rk u32[P] at 0, {id, mid} u32 pairs [P] at 4P, links u16[P] at 12P (lo byte next, hi
 *          byte prev, 0xFF none), P = len rounded up to 16: 14 P bytes
 *   long   heap u64[2 len] at 0, then the records, 16 bytes each: 32 len bytes (live heap entries
 *          <= 2 (len - 1): len - 1 at first, each merge pops one and pushes at most two)
 * Registers: r15 t, r12 len, r10 merge_slots, r9 merge_shift, ymm5 PRIO_MASK x4;
 *   short: rbp rk, r13 {id, mid}, rbx links, r14 the merged id, r11 the next round's min;
 *   long: rbp heap, r13 records, r14 heap size, r11 merges.
 * Frame: 6 gpr + 32 bytes; ymm0-5 only (nothing to save on win64).
 * Variants measured and rejected on tr9970x (Zen 5): docs/kernels.md §5 "the asm tiers".
 */
#include "asm.h"
#include "layout.h"

/* private labels: L-prefixed on mach-o (where .L names are symbols), .L elsewhere */
#if defined(TOKS_MACHO)
#  define L(x) L##x
#else
#  define L(x) .L##x
#endif

#define SMAX       128          /* short path up to this many bytes (positions fit a byte; P <= 128) */
#define PRIO_MASK  0x3FFFFF     /* (1 << TOKS_PRIO_BITS) - 1 */
#define F_IGNORE   1            /* TOKS_TF_IGNORE_MERGES */
#define F_ASRANK   2            /* TOKS_TF_IDS_AS_RANK */
#define F_LONG     0x80         /* TOKS_TF_PROBE_LONG */
#define F_ASCII    0x100        /* TOKS_TF_PROBE_ASCII */
#define VSEED      0x85EBCA6B   /* TOKS_VSEED */
#define RK_NONE    0x40000000   /* rk values >= this: no merge at that symbol */

#define S_A        0            /* the args pointer */
#define S_R2I      8            /* rank2id, or 0 when ids are ranks */
#define S_L        16           /* long path: the left neighbour across a probe */
#define LOCALS     32

/* MPROBE: rax = key (left << 21 | right) -> eax = its prio, or 0xFFFFFFFF when absent.
 * Reads r10 merge_slots, r9 merge_shift, ymm5 PRIO_MASK x4, r15 (mask and maxprobe, full buckets only).
 * Clobbers rax, rdx, rdi, ymm1-ymm3. */
.macro MPROBE
        mov     rdx, rax
        shl     rdx, 22
        vmovq   xmm1, rdx
        vpbroadcastq ymm1, xmm1                 /* key << 22: the slot's key field in place */
        imul    rax, qword ptr [rip + L(fib64)]
        shrx    rdx, rax, r9                    /* the home bucket */
        shl     rdx, 6
        add     rdx, r10
90:
        vpandn  ymm2, ymm5, ymmword ptr [rdx]
        vpandn  ymm3, ymm5, ymmword ptr [rdx + 32]
        vpcmpeqq ymm2, ymm2, ymm1
        vpcmpeqq ymm3, ymm3, ymm1
        vmovmskpd edi, ymm2
        vmovmskpd eax, ymm3
        shl     eax, 4
        or      edi, eax                        /* bit j: slot j holds the key (at most one does) */
        jz      91f
        tzcnt   edi, edi
        mov     eax, dword ptr [rdx + rdi*8]
        and     eax, PRIO_MASK
        jmp     93f
91:
        cmp     qword ptr [rdx + 56], -1        /* an empty slot ends the probe */
        je      92f
        mov     rax, rdx                        /* a full bucket: the next one, within maxprobe */
        sub     rax, r10
        shr     rax, 6
        inc     rax
        and     rax, qword ptr [r15 + TT_MERGE_MASK]
        vmovq   rdi, xmm1
        shr     rdi, 22
        imul    rdi, qword ptr [rip + L(fib64)]
        shrx    rdi, rdi, r9                    /* home again */
        neg     rdi
        add     rdi, rax
        and     rdi, qword ptr [r15 + TT_MERGE_MASK]     /* buckets probed so far */
        cmp     rdi, qword ptr [r15 + TT_MERGE_MAXPROBE]
        jae     92f
        mov     rdx, rax
        shl     rdx, 6
        add     rdx, r10
        jmp     90b
92:
        mov     eax, -1
93:
.endm

/* MID: eax = a prio or NONE -> eax = its merged id (don't care for NONE). Clobbers rdi. */
.macro MID
        mov     rdi, qword ptr [rsp + S_R2I]
        test    rdi, rdi
        jz      94f
        cmp     eax, -1
        jne     95f
        xor     eax, eax                        /* NONE: any index inside rank2id */
95:
        mov     eax, dword ptr [rdi + rax*4]
94:
.endm

/* SCAN: r11d = min(rk[0, P)). Clobbers rax, rdx, rdi, ymm0, ymm1. */
.macro SCAN
        cmp     r12, 16
        ja      72f
        mov     edx, dword ptr [rbp]            /* <= 15 pairs: two cmov chains */
        mov     edi, dword ptr [rbp + 4]
        mov     eax, 2
        jmp     71f
70:
        mov     r11d, dword ptr [rbp + rax*4]
        cmp     r11d, edx
        cmovb   edx, r11d
        mov     r11d, dword ptr [rbp + rax*4 + 4]
        cmp     r11d, edi
        cmovb   edi, r11d
        add     eax, 2
71:
        lea     r11d, [rax + 1]
        cmp     r11, r12
        jb      70b
        cmp     edi, edx
        cmovb   edx, edi
        mov     r11d, edx
        jmp     74f
72:
        vmovdqu ymm0, ymmword ptr [rbp]         /* longer: vpminud over 32-byte rows */
        vpminud ymm0, ymm0, ymmword ptr [rbp + 32]
        lea     rax, [r12 + 15]
        and     rax, -16
        shl     rax, 2                          /* 4 P bytes */
        mov     edx, 64
        jmp     76f
73:
        vpminud ymm0, ymm0, ymmword ptr [rbp + rdx]
        vpminud ymm0, ymm0, ymmword ptr [rbp + rdx + 32]
        add     rdx, 64
76:
        cmp     rdx, rax
        jb      73b
        vextracti128 xmm1, ymm0, 1
        vpminud xmm0, xmm0, xmm1
        vpshufd xmm1, xmm0, 0x4E
        vpminud xmm0, xmm0, xmm1
        vpshufd xmm1, xmm0, 0xB1
        vpminud xmm0, xmm0, xmm1
        vmovd   r11d, xmm0
74:
.endm

/* HPUSH: push rax onto the min-heap at rbp (r14 = size). Clobbers rdx, rdi, r8. */
.macro HPUSH
        mov     rdx, r14
        inc     r14
96:
        test    rdx, rdx
        jz      97f
        lea     rdi, [rdx - 1]
        shr     rdi, 1
        mov     r8, qword ptr [rbp + rdi*8]
        cmp     r8, rax
        jbe     97f
        mov     qword ptr [rbp + rdx*8], r8
        mov     rdx, rdi
        jmp     96b
97:
        mov     qword ptr [rbp + rdx*8], rax
.endm

/* HSIFT: place r8 at index rdx of the min-heap at rbp (r14 = size), sifting down.
 * Clobbers rax, rcx, rsi, rdi, rdx. */
.macro HSIFT
98:
        lea     rdi, [rdx*2 + 1]
        cmp     rdi, r14
        jae     99f
        mov     rax, qword ptr [rbp + rdi*8]
        lea     rcx, [rdi + 1]
        cmp     rcx, r14
        jae     89f
        mov     rsi, qword ptr [rbp + rcx*8]
        cmp     rsi, rax
        cmovb   rax, rsi
        cmovb   rdi, rcx
89:
        cmp     rax, r8
        jae     99f
        mov     qword ptr [rbp + rdx*8], rax
        mov     rdx, rdi
        jmp     98b
99:
        mov     qword ptr [rbp + rdx*8], r8
.endm

/* MERGE_SETUP: S_R2I, r10 merge_slots, r9 merge_shift, ymm5 PRIO_MASK x4 (r15 t). SHORT_INIT: the short path's
 * work for r12 symbols: rbp rk (none), r13 {id, mid}, rdi links {i + 1, i - 1}; S_L = r12 (r14 a). */
.macro MERGE_SETUP
        xor     eax, eax
        test    dword ptr [r15 + TT_FLAGS], F_ASRANK
        jnz     1f
        mov     rax, qword ptr [r15 + TT_RANK2ID]
1:
        mov     qword ptr [rsp + S_R2I], rax
        mov     r10, qword ptr [r15 + TT_MERGE_SLOTS]
        mov     r9, qword ptr [r15 + TT_MERGE_SHIFT]
        vpbroadcastq ymm5, qword ptr [rip + L(prio_mask)]
.endm
.macro SHORT_INIT
        mov     qword ptr [rsp + S_L], r12      /* len, for merges = len - n_out */
        mov     rbp, qword ptr [r14 + K6_WORK]  /* rk */
        lea     rax, [r12 + 15]
        and     rax, -16                        /* P */
        lea     r13, [rbp + rax*4]              /* {id, mid} */
        lea     rdx, [rax + rax*2]
        lea     rdi, [rbp + rdx*4]              /* links */
        vpcmpeqd ymm0, ymm0, ymm0
        lea     rsi, [rip + L(links)]
        xor     ecx, ecx
2:
        vmovdqu ymmword ptr [rbp + rcx*4], ymm0         /* rk[i, i + 16) = none */
        vmovdqu ymmword ptr [rbp + rcx*4 + 32], ymm0
        vmovdqu ymm1, ymmword ptr [rsi + rcx*2]         /* links[i, i + 16) = {i + 1, i - 1} */
        vmovdqu ymmword ptr [rdi + rcx*2], ymm1
        add     rcx, 16
        cmp     rcx, rax
        jb      2b
.endm

FUNC toks_k6_bpe_avx2
        PROLOGUE 6, 0, LOCALS
        mov     r15, ARG0
        mov     r14, ARG1
        mov     rbx, qword ptr [r14 + K6_PIECE]
        mov     r12, qword ptr [r14 + K6_LEN]
        mov     qword ptr [rsp + S_A], r14
        test    r12, r12
        jz      L(zero)
        test    dword ptr [r15 + TT_FLAGS], F_IGNORE
        jnz     L(vhash)
L(bpe):
        cmp     r12, 1
        je      L(one)
        MERGE_SETUP
        cmp     r12, SMAX
        ja      L(heap)

/* ---- short: linked array; the next round's scan beside this round's probes ------------------------ */
        SHORT_INIT
        mov     byte ptr [rdi + r12*2 - 2], 0xFF        /* the last symbol has no next */
        mov     r11d, 0x100                             /* a byte >= r11d may start a premerged char */
        mov     eax, 0xC0
        cmp     qword ptr [r15 + TT_PREMERGE], 0
        cmovne  r11d, eax                               /* (0x100: no table) */
        lea     rax, [r12 - 4]
        cmp     rax, TOKS_KEY_MAXLEN - 4
        ja      59f
        cmp     qword ptr [r15 + TT_APM], 0
        je      59f                                     /* not 4..15 bytes, or no ascii table */
        movzx   edx, byte ptr [rbx + 1]
        or      edx, 0x20
        sub     edx, 0x61
        cmp     edx, 26
        jae     59f
        xor     r11d, r11d                              /* a letter second (a word, maybe after one mark): every byte may */
59:
        movzx   eax, byte ptr [rbx]
        cmp     eax, r11d
        jae     L(s_pm)
        mov     rsi, qword ptr [r15 + TT_BYTE2ID]
        mov     rdx, qword ptr [r15 + TT_BYTEPAIR]
        xor     ecx, ecx
3:
        movzx   eax, byte ptr [rbx + rcx]
        movzx   r8d, byte ptr [rbx + rcx + 1]
        cmp     r8d, r11d
        jae     L(s_pm)
        mov     r14d, dword ptr [rsi + rax*4]
        mov     dword ptr [r13 + rcx*8], r14d           /* id[i] = byte2id[p[i]] */
        shl     eax, 8
        or      eax, r8d
        mov     eax, dword ptr [rdx + rax*4]            /* round one: bytepair[p[i] << 8 | p[i + 1]] */
        mov     r14d, eax
        shl     r14d, 8
        or      r14d, ecx
        mov     dword ptr [rbp + rcx*4], r14d           /* rk[i] (NONE << 8 | i >= RK_NONE) */
        mov     r8, rdi
        MID
        mov     rdi, r8
        mov     dword ptr [r13 + rcx*8 + 4], eax        /* mid[i] */
        inc     rcx
        lea     rax, [rcx + 1]
        cmp     rax, r12
        jb      3b
        movzx   eax, byte ptr [rbx + rcx]
        mov     eax, dword ptr [rsi + rax*4]
        mov     dword ptr [r13 + rcx*8], eax            /* id[len - 1] */
        mov     rbx, rdi                                /* links */
        SCAN
        mov     eax, r11d

        .p2align 4
L(merge):                                               /* eax = min(rk) */
        cmp     eax, RK_NONE
        jae     L(short_done)
        movzx   ecx, al                                 /* pos */
        mov     r14d, dword ptr [r13 + rcx*8 + 4]       /* id = mid[pos] */
        movzx   edx, byte ptr [rbx + rcx*2]             /* q = next[pos], merged away */
        movzx   esi, byte ptr [rbx + rcx*2 + 1]         /* pv = prev[pos] */
        movzx   r8d, byte ptr [rbx + rdx*2]             /* nn = next[q] */
        mov     dword ptr [r13 + rcx*8], r14d
        mov     dword ptr [rbp + rdx*4], -1             /* q: dead */
        mov     dword ptr [rbp + rcx*4], -1             /* pos, pv: none until probed below */
        mov     edi, esi
        cmp     esi, 0xFF
        cmove   edi, edx                                /* (no pv: q again) */
        mov     dword ptr [rbp + rdi*4], -1
        mov     byte ptr [rbx + rcx*2], r8b             /* next[pos] = nn */
        cmp     r8d, 0xFF
        je      4f
        mov     byte ptr [rbx + r8*2 + 1], cl           /* prev[nn] = pos */
4:
        SCAN                                            /* the rest's min, beside the probes */
        cmp     r8d, 0xFF
        je      5f
        mov     eax, dword ptr [r13 + r8*8]
        mov     rdx, r14
        shl     rdx, 21
        or      rax, rdx                                /* (id, id[nn]) */
        MPROBE
        mov     edx, eax
        shl     edx, 8
        or      edx, ecx
        mov     dword ptr [rbp + rcx*4], edx            /* rk[pos] */
        cmp     edx, r11d
        cmovb   r11d, edx
        MID
        mov     dword ptr [r13 + rcx*8 + 4], eax        /* mid[pos] */
5:
        cmp     esi, 0xFF
        je      6f
        mov     eax, dword ptr [r13 + rsi*8]
        shl     rax, 21
        or      rax, r14                                /* (id[pv], id) */
        MPROBE
        mov     edx, eax
        shl     edx, 8
        or      edx, esi
        mov     dword ptr [rbp + rsi*4], edx            /* rk[pv] */
        cmp     edx, r11d
        cmovb   r11d, edx
        MID
        mov     dword ptr [r13 + rsi*8 + 4], eax        /* mid[pv] */
6:
        mov     eax, r11d
        jmp     L(merge)

L(short_done):
        mov     r14, qword ptr [rsp + S_A]
        mov     rdi, qword ptr [r14 + K6_OUT]
        xor     ecx, ecx                                /* symbol 0 is always alive */
        xor     eax, eax
9:
        mov     edx, dword ptr [r13 + rcx*8]
        mov     dword ptr [rdi + rax*4], edx
        inc     rax
        movzx   ecx, byte ptr [rbx + rcx*2]
        cmp     ecx, 0xFF
        jne     9b
        mov     rdx, qword ptr [rsp + S_L]
        sub     rdx, rax                                /* merges */
        jmp     L(ret)

/* ---- short with premerged chars (kernels.md §5.1): the byte init met a byte >= 0xC0 and the premerge
 * table exists. rk is NONE-filled and the links laid out for len; the byte init wrote rk / mid only for
 * bytes before that lead, all below n - 1, so every one is rewritten. 1: the symbols into SYM (at 16 P): a
 * char whose entry is set and whose neighbour bytes are not risky as its token, every other byte as
 * byte2id. 2: id, rk and mid from a probe of every pair (round one has no byte pair table for tokens).
 * Then the merge loop runs on n = r12 symbols. */
L(s_pm):
        xor     edx, edx
        test    r11d, r11d
        cmovz   rdx, qword ptr [r15 + TT_APM]           /* the ascii table for a word routed here (no rk */
        vmovq   xmm2, rdx                               /* written yet), else 0: none */
        mov     r14, qword ptr [r15 + TT_PREMERGE]
        lea     rsi, [r12 + 15]
        and     rsi, -16
        shl     rsi, 4
        add     rsi, rbp                                /* SYM */
        xor     ecx, ecx                                /* i */
        xor     r8d, r8d                                /* n */
40:
        movzx   eax, byte ptr [rbx + rcx]
        cmp     eax, 0xC0
        jb      L(apm)
        mov     r14, qword ptr [r15 + TT_PREMERGE]
        test    r14, r14
        jz      47f
        mov     rdx, r12
        sub     rdx, rcx                                /* bytes left */
        cmp     eax, 0xE0
        jae     41f
        cmp     rdx, 2                                  /* C0..DF + a continuation byte */
        jb      47f
        movzx   edi, byte ptr [rbx + rcx + 1]
        xor     edi, 0x80                               /* b1 & 63 when b1 is 80..BF, else >= 0x40 */
        cmp     edi, 0x40
        jae     47f
        lea     edx, [rax - 0xC0]                       /* s1 */
        mov     r11d, 2
        jmp     42f
41:
        cmp     eax, 0xF0                               /* E0..EF + two continuation bytes */
        jae     47f
        cmp     rdx, 3
        jb      47f
        movzx   edx, byte ptr [rbx + rcx + 1]
        xor     edx, 0x80
        cmp     edx, 0x40
        jae     47f
        movzx   edi, byte ptr [rbx + rcx + 2]
        xor     edi, 0x80
        cmp     edi, 0x40
        jae     47f
        and     eax, 15
        shl     eax, 6
        lea     edx, [rdx + rax + 32]                   /* s1 */
        mov     r11d, 3
42:
        mov     edx, dword ptr [r14 + rdx*4]            /* the block's offset */
        add     rdx, r14
        mov     rdx, qword ptr [rdx + rdi*8]            /* the entry (0: none) */
        test    rdx, rdx
        jz      47f
        lea     rdi, [rip + L(pmbits)]
        mov     eax, TOKS_PM_NEVER
        test    rcx, rcx
        jz      43f
        movzx   eax, byte ptr [rbx + rcx - 1]
        movzx   eax, byte ptr [rdi + rax]               /* the byte before's risk bit */
43:
        shrx    rax, rdx, rax
        lea     r9, [rcx + r11]
        cmp     r9, r12
        jae     44f
        movzx   r9d, byte ptr [rbx + r9]
        movzx   r9d, byte ptr [rdi + r9 + 256]          /* the byte after's */
        shrx    r9, rdx, r9
        or      rax, r9
44:
        test    al, 1
        jnz     47f                                     /* risky: bytes */
        and     edx, TOKS_ID_MASK
        mov     dword ptr [rsi + r8*4], edx
        lea     rdx, [rsi + r12*4]
        mov     word ptr [rdx + r8*2], 0x100            /* BYT[n] = a token */
        inc     r8
        add     rcx, r11
        jmp     48f
L(apm):                                                 /* ascii: a certified bigram, else the byte */
        cmp     eax, 0x80
        jae     47f
        vmovq   rdx, xmm2
        test    rdx, rdx
        jz      47f
        lea     r9, [rcx + 2]
        cmp     r9, r12
        ja      47f
        movzx   edi, byte ptr [rbx + rcx + 1]
        test    edi, 0x80
        jnz     47f
        shl     eax, 7
        or      eax, edi
        shl     eax, 4
        lea     r11, [rdx + rax + TOKS_APM_PAIRS]       /* the pair's entry */
        cmp     dword ptr [r11], -1                     /* NONE: none */
        je      47f
        test    rcx, rcx
        jz      52f
        movzx   eax, byte ptr [rbx + rcx - 1]
        movzx   eax, byte ptr [rdx + rax]
        mov     edi, dword ptr [r11 + 8]
        bt      edi, eax
        jc      47f                                     /* the byte before is risky */
52:
        cmp     r9, r12
        jae     53f
        movzx   eax, byte ptr [rbx + rcx + 2]
        movzx   eax, byte ptr [rdx + rax]
        mov     edi, dword ptr [r11 + 4]
        bt      edi, eax
        jc      47f                                     /* the byte after is */
53:
        mov     eax, dword ptr [r11]
        mov     dword ptr [rsi + r8*4], eax
        lea     rdx, [rsi + r12*4]
        mov     word ptr [rdx + r8*2], 0x100            /* BYT[n] = a token */
        inc     r8
        add     rcx, 2
        jmp     48f
47:
        movzx   eax, byte ptr [rbx + rcx]               /* a byte */
        lea     rdx, [rsi + r12*4]
        mov     word ptr [rdx + r8*2], ax               /* BYT[n] = the byte */
        mov     rdx, qword ptr [r15 + TT_BYTE2ID]
        mov     eax, dword ptr [rdx + rax*4]
        mov     dword ptr [rsi + r8*4], eax
        inc     r8
        inc     rcx
48:
        cmp     rcx, r12
        jb      40b
L(s_recs):                                              /* toks_k6_merge_avx2 enters here */
        mov     r9, qword ptr [r15 + TT_MERGE_SHIFT]
        lea     r11, [rsi + r12*4]                      /* BYT */
        mov     r14, qword ptr [r15 + TT_BYTEPAIR]
        mov     r12, r8                                 /* the merge loop counts symbols */
        xor     ecx, ecx
50:
        mov     eax, dword ptr [rsi + rcx*4]
        mov     dword ptr [r13 + rcx*8], eax            /* id[k] */
        lea     rdx, [rcx + 1]
        cmp     rdx, r12
        jae     51f
        movzx   edx, word ptr [r11 + rcx*2]
        movzx   edi, word ptr [r11 + rcx*2 + 2]
        mov     r8d, edx
        or      r8d, edi
        test    r8d, 0x100
        jnz     57f
        shl     edx, 8
        or      edx, edi
        mov     eax, dword ptr [r14 + rdx*4]            /* two bytes: bytepair */
        jmp     58f
57:
        mov     edx, dword ptr [rsi + rcx*4 + 4]
        shl     rax, 21
        or      rax, rdx                                /* (id[k], id[k + 1]) */
        MPROBE
58:
        mov     edx, eax
        shl     edx, 8
        or      edx, ecx
        mov     dword ptr [rbp + rcx*4], edx            /* rk[k] (NONE << 8 | k >= RK_NONE) */
        MID
        mov     dword ptr [r13 + rcx*8 + 4], eax        /* mid[k] */
        inc     rcx
        jmp     50b
51:
        mov     rax, qword ptr [rsp + S_L]
        add     rax, 15
        and     rax, -16
        lea     rax, [rax + rax*2]
        lea     rbx, [rbp + rax*4]                      /* links */
        mov     byte ptr [rbx + r12*2 - 2], 0xFF        /* the last symbol has no next */
        SCAN
        mov     eax, r11d
        jmp     L(merge)

/* ---- long: heap + per-position records {id, next, prev, cur} ------------------------------------- */
L(heap):
        xor     r11d, r11d                              /* merges */
        mov     rbp, qword ptr [r14 + K6_WORK]          /* heap */
        mov     rax, r12
        shl     rax, 4
        lea     r13, [rbp + rax]                        /* records */
        mov     rsi, qword ptr [r15 + TT_BYTE2ID]
        mov     rdi, qword ptr [r15 + TT_BYTEPAIR]
        xor     r14d, r14d                              /* heap size */
        xor     ecx, ecx
10:
        movzx   eax, byte ptr [rbx + rcx]
        mov     eax, dword ptr [rsi + rax*4]
        mov     rdx, rcx
        shl     rdx, 4
        mov     dword ptr [r13 + rdx], eax              /* id */
        lea     eax, [rcx + 1]
        mov     dword ptr [r13 + rdx + 4], eax          /* next */
        lea     eax, [rcx - 1]
        mov     dword ptr [r13 + rdx + 8], eax          /* prev (-1 at 0) */
        lea     rax, [rcx + 1]
        cmp     rax, r12
        jae     12f
        movzx   eax, byte ptr [rbx + rcx]
        movzx   r8d, byte ptr [rbx + rcx + 1]
        shl     eax, 8
        or      eax, r8d
        mov     eax, dword ptr [rdi + rax*4]
        mov     dword ptr [r13 + rdx + 12], eax         /* cur */
        cmp     eax, -1
        je      11f
        shl     rax, 32
        or      rax, rcx
        mov     qword ptr [rbp + r14*8], rax            /* heapified below */
        inc     r14
11:
        inc     rcx
        jmp     10b
12:
        mov     dword ptr [r13 + rdx + 4], -1
        mov     dword ptr [r13 + rdx + 12], -1
        mov     rbx, r14                                /* heapify: sift down i = n / 2 - 1 .. 0 */
        shr     rbx, 1
13:
        test    rbx, rbx
        jz      L(hloop)
        dec     rbx
        mov     rdx, rbx
        mov     r8, qword ptr [rbp + rbx*8]
        HSIFT
        jmp     13b

        .p2align 4
L(hloop):
        test    r14, r14
        jz      L(heap_done)
        mov     rbx, qword ptr [rbp]                    /* the minimum */
        dec     r14
        mov     r8, qword ptr [rbp + r14*8]
        xor     edx, edx
        HSIFT
        mov     ecx, ebx                                /* pos */
        shr     rbx, 32                                 /* prio */
        mov     rdx, rcx
        shl     rdx, 4
        cmp     dword ptr [r13 + rdx + 12], ebx
        jne     L(hloop)                                /* stale */
        mov     rax, qword ptr [rsp + S_R2I]
        test    rax, rax
        jz      14f
        mov     ebx, dword ptr [rax + rbx*4]            /* rank2id[prio] */
14:
        mov     esi, dword ptr [r13 + rdx + 4]          /* r = next[pos], merged away */
        shl     rsi, 4
        mov     eax, dword ptr [r13 + rsi + 4]          /* rn = next[r] */
        mov     dword ptr [r13 + rsi + 12], -1          /* cur[r] = none */
        mov     dword ptr [r13 + rdx], ebx              /* id[pos] = the merged id */
        mov     dword ptr [r13 + rdx + 4], eax          /* next[pos] = rn */
        mov     esi, eax
        inc     r11
        cmp     esi, -1
        je      15f
        mov     rax, rsi
        shl     rax, 4
        mov     dword ptr [r13 + rax + 8], ecx          /* prev[rn] = pos */
15:
        mov     eax, dword ptr [r13 + rdx + 8]          /* l = prev[pos] */
        cmp     eax, -1
        je      16f
        mov     qword ptr [rsp + S_L], rax
        shl     rax, 4
        mov     eax, dword ptr [r13 + rax]
        shl     rax, 21
        or      rax, rbx                                /* (id[l], id) */
        MPROBE
        mov     rdx, qword ptr [rsp + S_L]
        mov     rdi, rdx
        shl     rdi, 4
        mov     dword ptr [r13 + rdi + 12], eax         /* cur[l] */
        cmp     eax, -1
        je      16f
        shl     rax, 32
        or      rax, rdx
        HPUSH
16:
        cmp     esi, -1
        je      17f
        mov     rax, rsi
        shl     rax, 4
        mov     eax, dword ptr [r13 + rax]
        mov     rdx, rbx
        shl     rdx, 21
        or      rax, rdx                                /* (id, id[rn]) */
        MPROBE
        mov     rdx, rcx
        shl     rdx, 4
        mov     dword ptr [r13 + rdx + 12], eax         /* cur[pos] */
        cmp     eax, -1
        je      L(hloop)
        shl     rax, 32
        or      rax, rcx
        HPUSH
        jmp     L(hloop)
17:
        mov     rdx, rcx
        shl     rdx, 4
        mov     dword ptr [r13 + rdx + 12], -1          /* pos is the last symbol now */
        jmp     L(hloop)

L(heap_done):
        mov     r14, qword ptr [rsp + S_A]
        mov     rdi, qword ptr [r14 + K6_OUT]
        xor     ecx, ecx
        xor     eax, eax
18:
        mov     rdx, rcx
        shl     rdx, 4
        mov     esi, dword ptr [r13 + rdx]
        mov     dword ptr [rdi + rax*4], esi
        inc     rax
        mov     ecx, dword ptr [r13 + rdx + 4]
        cmp     ecx, -1
        jne     18b
        mov     rdx, r11
        jmp     L(ret)

/* ---- one byte, empty piece, return ---------------------------------------------------------------- */
L(one):
        movzx   eax, byte ptr [rbx]
        mov     rcx, qword ptr [r15 + TT_BYTE2ID]
        mov     eax, dword ptr [rcx + rax*4]
        mov     rcx, qword ptr [r14 + K6_OUT]
        mov     dword ptr [rcx], eax
        mov     eax, 1
        xor     edx, edx
        jmp     L(ret)
L(zero):
        xor     eax, eax
        xor     edx, edx
L(ret):                                                 /* rax n_out, rdx merges, r14 a */
        mov     qword ptr [r14 + K6_N_OUT], rax
        mov     qword ptr [r14 + K6_MERGES], rdx
        vzeroupper
        EPILOGUE 6, 0, LOCALS

/* ---- ignore_merges: the whole piece in the vocab hash ---------------------------------------------- */
L(vhash):
        test    dword ptr [r15 + TT_FLAGS], F_LONG      /* TOKS_TF_PROBE_LONG: a piece over TOKS_KEY_MAXLEN */
        jz      L(vhash_any)                            /* bytes only */
        cmp     r12, TOKS_KEY_MAXLEN
        jbe     L(bpe)
        test    dword ptr [r15 + TT_FLAGS], F_ASCII     /* TOKS_TF_PROBE_ASCII: and ending in ascii */
        jz      L(vhash_any)
        cmp     byte ptr [rbx + r12 - 1], 0
        js      L(bpe)
L(vhash_any):
        mov     r8, qword ptr [r15 + TT_VHASH]
        test    r8, r8
        jz      L(bpe)
        mov     r9, qword ptr [r15 + TT_VHASH_MASK]     /* after the slots, u32 [256]: the longest token by its */
        lea     rdx, [r8 + r9*8 + 8]                    /* first byte */
        movzx   ecx, byte ptr [rbx]
        mov     ecx, dword ptr [rdx + rcx*4]
        cmp     r12, rcx
        ja      L(bpe)                                  /* longer than any token that starts so */
        mov     eax, VSEED
        xor     eax, r12d                               /* h = VSEED ^ (uint32_t)len */
        mov     rsi, rbx
        mov     rcx, r12
        cmp     rcx, 8
        jb      21f
20:
        crc32   rax, qword ptr [rsi]
        add     rsi, 8
        sub     rcx, 8
        cmp     rcx, 8
        jae     20b
21:
        test    rcx, rcx
        jz      24f
        cmp     ecx, 4                                  /* the last 1..7 bytes, zero-padded, read */
        jb      22f                                     /* with overlapping loads inside the piece */
        mov     edx, dword ptr [rsi]
        mov     edi, dword ptr [rsi + rcx - 4]
        lea     ecx, [rcx*8 - 32]
        shlx    rdi, rdi, rcx
        or      rdx, rdi
        jmp     23f
22:
        cmp     ecx, 2
        jb      25f
        movzx   edx, word ptr [rsi]
        movzx   edi, word ptr [rsi + rcx - 2]
        lea     ecx, [rcx*8 - 16]
        shlx    rdi, rdi, rcx
        or      rdx, rdi
        jmp     23f
25:
        movzx   edx, byte ptr [rsi]
23:
        crc32   rax, rdx
24:
        mov     r9, qword ptr [r15 + TT_VHASH_MASK]     /* four slots from the home slot, without a branch: the */
        mov     rdx, rax                                /* first that is empty (a miss) or holds h (a candidate) */
        and     rdx, r9                                 /* decides. Their line may come from DRAM, and a branch per */
        lea     rsi, [rdx + 1]                          /* slot (empty? h?) mispredicted about half the time, each */
        and     rsi, r9                                 /* miss waiting for the line; one branch on the outcome */
        lea     rdi, [rdx + 2]                          /* predicts "not a token", and bpe runs on while the line */
        and     rdi, r9                                 /* comes */
        lea     r10, [rdx + 3]
        and     r10, r9
        lea     r11, [rdx + 4]                          /* r11: where the probe goes on (the fifth slot, or the */
        and     r11, r9                                 /* candidate) */
        mov     r13d, 2                                 /* r13d: 1 an empty slot decides, 0 a candidate, 2 none */
        mov     rbp, qword ptr [r8 + r10*8]
        cmp     ebp, eax
        cmove   r11, r10
        cmove   r13d, dword ptr [rip + L(vh0)]
        cmp     rbp, -1
        cmove   r11, r10
        cmove   r13d, dword ptr [rip + L(vh1)]
        mov     rbp, qword ptr [r8 + rdi*8]
        cmp     ebp, eax
        cmove   r11, rdi
        cmove   r13d, dword ptr [rip + L(vh0)]
        cmp     rbp, -1
        cmove   r11, rdi
        cmove   r13d, dword ptr [rip + L(vh1)]
        mov     rbp, qword ptr [r8 + rsi*8]
        cmp     ebp, eax
        cmove   r11, rsi
        cmove   r13d, dword ptr [rip + L(vh0)]
        cmp     rbp, -1
        cmove   r11, rsi
        cmove   r13d, dword ptr [rip + L(vh1)]
        mov     rbp, qword ptr [r8 + rdx*8]
        cmp     ebp, eax
        cmove   r11, rdx
        cmove   r13d, dword ptr [rip + L(vh0)]
        cmp     rbp, -1
        cmove   r11, rdx
        cmove   r13d, dword ptr [rip + L(vh1)]
        mov     rdx, r11
        xor     r10d, r10d                              /* slots probed: <= mask + 1, as the c twin */
        cmp     r13d, 1
        je      L(bpe)                                  /* an empty slot decides: not a vocabulary token */
26:
        mov     rsi, qword ptr [r8 + rdx*8]
        cmp     rsi, -1
        je      L(bpe)
        cmp     esi, eax
        jne     29f
        shr     rsi, 32                                 /* id */
        mov     rdi, qword ptr [r15 + TT_TOK_OFF]
        mov     r11d, dword ptr [rdi + rsi*4]           /* o0 */
        mov     edi, dword ptr [rdi + rsi*4 + 4]        /* o1 */
        sub     rdi, r11
        cmp     rdi, r12
        jne     29f
        add     r11, qword ptr [r15 + TT_TOK_BYTES]     /* the token's bytes: compare len of them */
        xor     ecx, ecx
        cmp     r12, 8
        jb      28f
27:
        mov     rdi, qword ptr [r11 + rcx]
        cmp     rdi, qword ptr [rbx + rcx]
        jne     29f
        add     rcx, 8
        lea     rdi, [rcx + 8]
        cmp     rdi, r12
        jbe     27b
        mov     rdi, qword ptr [r11 + r12 - 8]          /* the last 8 (overlapping) */
        cmp     rdi, qword ptr [rbx + r12 - 8]
        jne     29f
        jmp     L(vhit)
28:
        movzx   edi, byte ptr [r11 + rcx]
        cmp     dil, byte ptr [rbx + rcx]
        jne     29f
        inc     rcx
        cmp     rcx, r12
        jb      28b
L(vhit):
        mov     rax, qword ptr [r8 + rdx*8]
        shr     rax, 32
        mov     rcx, qword ptr [r14 + K6_OUT]
        mov     dword ptr [rcx], eax
        mov     eax, 1
        xor     edx, edx
        jmp     L(ret)
29:
        inc     rdx
        and     rdx, r9
        inc     r10
        cmp     r10, r9
        jbe     26b
        jmp     L(bpe)
ENDFUNC toks_k6_bpe_avx2

/*   uint64_t toks_k6_merge_avx2(const toks_tables *t, toks_k6_args *a)
 * K6's merge loop on given initial symbols (spm_c.c's model; the c twin is k6_c.c's toks_k6_merge): a->piece holds
 * n = a->len u32 symbols, 2 <= n <= SMAX, 4-aligned, with room for 2 n bytes after them (BYT: every symbol a token,
 * each pair probed); a->out, a->work, n_out and merges as K6's. Same frame as toks_k6_bpe_avx2, whose short path it
 * joins. */
FUNC toks_k6_merge_avx2
        PROLOGUE 6, 0, LOCALS
        mov     r15, ARG0
        mov     r14, ARG1
        mov     rbx, qword ptr [r14 + K6_PIECE]
        mov     r12, qword ptr [r14 + K6_LEN]
        mov     qword ptr [rsp + S_A], r14
        MERGE_SETUP
        SHORT_INIT
        mov     rsi, rbx                                /* SYM */
        lea     rdx, [rbx + r12*4]
        xor     ecx, ecx
3:
        mov     word ptr [rdx + rcx*2], 0x100           /* BYT[k] = a token */
        inc     rcx
        cmp     rcx, r12
        jb      3b
        mov     r8, r12                                 /* the symbols */
        jmp     L(s_recs)
ENDFUNC toks_k6_merge_avx2

        RODATA
        .p2align 2
L(vh0): .long   0
L(vh1): .long   1
        .p2align 5
L(links):                                               /* u16 {next = i + 1, prev = i - 1}, i < 128 */
        .set    L(li), 0
        .rept   128
        .byte   (L(li) + 1) & 0xFF, (L(li) - 1) & 0xFF
        .set    L(li), L(li) + 1
        .endr
L(pmbits):                                              /* layout.h: the premerge risk bit of a neighbour byte */
        .rept   32                                      /* before: 00..1F */
        .byte   57
        .endr
        .byte   56                                      /* 20 */
        .rept   95                                      /* 21..7F */
        .byte   57
        .endr
        .rept   4                                       /* 80..BF: 40 + (b & 15) */
        .byte   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
        .endr
        .rept   64                                      /* C0..FF */
        .byte   63
        .endr
        .rept   128                                     /* after: 00..7F */
        .byte   58
        .endr
        .rept   64                                      /* 80..BF */
        .byte   63
        .endr
        .rept   32                                      /* C0..DF */
        .byte   59
        .endr
        .byte   24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39   /* E0..EF */
        .rept   16                                      /* F0..FF */
        .byte   60
        .endr
        .p2align 3
L(fib64):
        .quad   0x9E3779B97F4A7C15
L(prio_mask):
        .quad   PRIO_MASK
        TEXT
