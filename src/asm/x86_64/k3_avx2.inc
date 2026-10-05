/*
 * src/asm/x86_64/k3_avx2.inc: the avx2 K3 both templates assemble from (docs/kernels.md §3.1; o200k:
 * docs/templates/o200k.md §4): k3_cl100k_avx2.S sets K3_O200K 0, k3_o200k_avx2.S sets it 1, k3_dsv3_avx2.S
 * sets it and K3_DSV3 (o200k's algebra on dsv3's tables, docs/templates/dsv3.md §7). One loop of
 * 64-byte blocks at a piece start: the ascii class masks, the non-ascii atom pass (its 2/3-byte fast path and
 * CJK lead shortcut), register-padded short segments, the piece-start algebra (digit and whitespace terms
 * shared, each template's letters, punctuation tail and cutting classes under #if K3_O200K), the decided
 * range and the emission. A stretch the blocks leave alone (under 4 bytes, a piece longer than its block, the
 * pieces through a cut atom) returns with a->rsv = its end: kernels.h's toks_k3_parts runs the c twin there.
 *
 * Registers: r15 text, r14 len, r13 pos, r12 the next ends slot, rbx the end of ends (cap).
 * Frame: 6 gpr + LOCALS bytes; only ymm0-5 are used, so win64 saves no xmm.
 */
#include "asm.h"
#include "layout.h"
#ifndef K3_DSV3
#define K3_DSV3 0
#endif

/* private labels: L-prefixed on mach-o (where .L names are symbols that break the cfi), .L elsewhere */
#if defined(TOKS_MACHO)
#  define L(x) L##x
#else
#  define L(x) .L##x
#endif

#define S_A      0      /* toks_k3_args * */
#define S_ENDS0  8      /* ends */
#define S_UNTIL  16     /* the pieces before until are the twin's (through a cut atom); 0: none */
#define S_PARAM  24     /* t->tmpl_params; the dword at S_PARAM + 4 is t->flags */
#define S_E      32     /* block: bytes in play */
#define S_CAND   40     /* block: ' at piece starts */
#define S_P1     48     /* block: the non-ascii bytes (L(na_give_up)); o200k: P, the slot table */
#define S_STG1   56     /* t->cls_stage1 */
#define S_STG2   64     /* t->cls_stage2 */
#define S_CONT   72     /* block: the bytes after the first of each non-ascii atom */
#define S_NWN    80     /* block: a non-W byte follows */
#define S_NNA    88     /* block: non-ascii N atoms */
#define S_WNA    96     /* block: non-ascii WS atoms */
#if K3_O200K
#define S_TAILS  104    /* block: the '/' the last mark round's tails took */
#define S_AU     40     /* block: upper-only atoms (A-Z, Lu, Lt); cl100k's S_CAND */
#define S_AW     112    /* block: lower-only atoms (a-z, Ll) */
#define S_AB     120    /* block: base-L atoms in both groups (Lm, Lo) */
#define S_AM     128    /* block: marks (base P, both groups); from the tail step on, C (every case atom) */
#define S_AH     136    /* block: Han letters (TOKS_TP_HAN); dsv3: its CJK letters */
#if K3_DSV3
#define S_DM     144    /* dsv3 block: the ascii bytes 20-7E */
#define S_DX     152    /* dsv3 block: the ascii X bytes 00-08 0E-1F 7F */
#define S_DA     160    /* dsv3 block: the ascii letters */
#define LOCALS   176
#define K3_TF_CJK TOKS_TF_CJK_D /* U+4E00..U+9EFF atoms: Han letters */
#define S_CJK    S_AH
#else
#define LOCALS   144
#define K3_TF_CJK TOKS_TF_CJK_B /* CJK / hangul atoms: letters in both case groups */
#define S_CJK    S_AB
#endif
#else
#define S_LF     104    /* cl100k block (NL_CUT): the LF bytes, then the cuts C (kernels.md §3 A9); 0 without */
#define LOCALS   112
#define K3_TF_CJK TOKS_TF_CJK_L
#endif

/* NA_ATOM o, cls, k, t1, t2: the atom at text[o] whose first byte (t1, >= 0x80) is not ascii: its class
 * byte and length (toks_utf8_len's well-formedness, never reading at or past len; an ill-formed first
 * byte is a one-byte P atom). rdi = cls_stage1, rbp = cls_stage2. Clobbers t1, t2. */
.macro NA_ATOM o, clsd, kd, t1d, t1q, t2d, t2q
        cmp     \t1d, 0xC2
        jb      91f
        cmp     \t1d, 0xE0
        jae     92f
        lea     \t2q, [\o + 2]
        cmp     \t2q, r14
        ja      91f
        movzx   \t2d, byte ptr [r15 + \o + 1]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        shl     \t1d, 6
        and     \t1d, 0x7C0
        or      \t1d, \t2d
        mov     \kd, 2
        jmp     94f
92:
        cmp     \t1d, 0xF0
        jae     93f
        lea     \t2q, [\o + 3]
        cmp     \t2q, r14
        ja      91f
        and     \t1d, 0x0F
        shl     \t1d, 12
        movzx   \t2d, byte ptr [r15 + \o + 1]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        shl     \t2d, 6
        or      \t1d, \t2d
        movzx   \t2d, byte ptr [r15 + \o + 2]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        or      \t1d, \t2d
        cmp     \t1d, 0x800
        jb      91f                      /* overlong */
        lea     \t2d, [\t1q - 0xD800]
        cmp     \t2d, 0x800
        jb      91f                      /* surrogate */
        mov     \kd, 3
        jmp     94f
93:
        cmp     \t1d, 0xF4
        ja      91f
        lea     \t2q, [\o + 4]
        cmp     \t2q, r14
        ja      91f
        and     \t1d, 0x07
        shl     \t1d, 18
        movzx   \t2d, byte ptr [r15 + \o + 1]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        shl     \t2d, 12
        or      \t1d, \t2d
        movzx   \t2d, byte ptr [r15 + \o + 2]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        shl     \t2d, 6
        or      \t1d, \t2d
        movzx   \t2d, byte ptr [r15 + \o + 3]
        xor     \t2d, 0x80
        cmp     \t2d, 0x3F
        ja      91f
        or      \t1d, \t2d
        lea     \t2d, [\t1q - 0x10000]
        cmp     \t2d, 0x100000
        jae     91f                      /* overlong or past U+10FFFF */
        mov     \kd, 4
94:
        mov     \t2d, \t1d
        shr     \t2d, 8
        movzx   \t2d, word ptr [rdi + \t2q*2]
        shl     \t2d, 8
        and     \t1d, 0xFF
        or      \t2d, \t1d
        movzx   \clsd, byte ptr [rbp + \t2q]
#if K3_O200K
        test    \clsd, TOKS_C_HAN
        jz      95f
        and     \clsd, ~(TOKS_C_UPPER | TOKS_C_LOWER)  /* o200k.md §6: Han atoms leave both case groups */
#endif
        jmp     95f
91:
        xor     \clsd, \clsd                    /* TOKS_C_P */
        mov     \kd, 1
95:
.endm

/* CLASSIFY: the 64 bytes in ymm0:ymm1 into rax = non-ascii, rsi = L (o200k: '/', with A-Z and a-z in
 * S_AU and S_AW), rdi = N, r8 = W (\s), r9 = NL, r10 = ' ', r11 = '. Letters, digits and \t..\r map onto
 * the top of the signed byte range so one vpcmpgtb against a memory constant tests each range. Clobbers
 * rcx, ymm2-5. */
.macro CLASSIFY
        vpmovmskb eax, ymm0
        vpmovmskb ecx, ymm1
        shl     rcx, 32
        or      rax, rcx
#if K3_O200K
        MASK64  rsi, esi, k3a2_c25, k3a2_c65    /* A-Z */
        mov     qword ptr [rsp + S_AU], rsi
        MASK64  rsi, esi, k3a2_c05, k3a2_c65    /* a-z */
        mov     qword ptr [rsp + S_AW], rsi
        vpcmpeqb ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c2f)]
        vpcmpeqb ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c2f)]
        vpmovmskb esi, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      rsi, rcx                        /* '/' */
#else
        vpor    ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c20)]
        vpor    ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c20)]
        vpaddb  ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c05)]
        vpaddb  ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c05)]
        vpcmpgtb ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c65)]
        vpcmpgtb ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c65)]
        vpmovmskb esi, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      rsi, rcx
#endif
        vpaddb  ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c46)]
        vpaddb  ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c46)]
        vpcmpgtb ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c75)]
        vpcmpgtb ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c75)]
        vpmovmskb edi, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      rdi, rcx
        vpcmpeqb ymm4, ymm0, ymmword ptr [rip + SYM(k3a2_c20)]
        vpcmpeqb ymm5, ymm1, ymmword ptr [rip + SYM(k3a2_c20)]
        vpaddb  ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c72)]
        vpaddb  ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c72)]
        vpcmpgtb ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c7a)]
        vpcmpgtb ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c7a)]
        vpor    ymm2, ymm2, ymm4
        vpor    ymm3, ymm3, ymm5
        vpmovmskb r8d, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      r8, rcx
        vpmovmskb r10d, ymm4
        vpmovmskb ecx, ymm5
        shl     rcx, 32
        or      r10, rcx
#if K3_DSV3
        MASK64  r11, r11d, k3a2_c01, k3a2_c20   /* 20-7E */
        mov     qword ptr [rsp + S_DM], r11
        or      r11, rax
        or      r11, r8
        not     r11
        mov     qword ptr [rsp + S_DX], r11
#endif
        vpcmpeqb ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c0a)]
        vpcmpeqb ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c0a)]
#if !K3_O200K
        test    byte ptr [rsp + S_PARAM + 1], TOKS_TP_NL_CUT >> 8
        jz      96f
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        vpmovmskb r9d, ymm2
        or      rcx, r9
        mov     qword ptr [rsp + S_LF], rcx     /* LF (A9) */
96:
#endif
        vpcmpeqb ymm4, ymm0, ymmword ptr [rip + SYM(k3a2_c0d)]
        vpcmpeqb ymm5, ymm1, ymmword ptr [rip + SYM(k3a2_c0d)]
        vpor    ymm2, ymm2, ymm4
        vpor    ymm3, ymm3, ymm5
        vpmovmskb r9d, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      r9, rcx
        vpcmpeqb ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c27)]
        vpcmpeqb ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c27)]
        vpmovmskb r11d, ymm2
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        or      r11, rcx
.endm

/* C32 name, v: 32 copies of byte v (a ymm operand) */
.macro C32 name, v
DATA \name
        .fill   32, 1, \v
.endm

/* BITIDX b: the indices of b's set bits ascending, padded to 8 bytes with 0 */
.macro BITIDX b
        .set    k3a2_n, 0
        .irp    i, 0, 1, 2, 3, 4, 5, 6, 7
        .if     ((\b) >> \i) & 1
        .byte   \i
        .set    k3a2_n, k3a2_n + 1
        .endif
        .endr
        .rept   8 - k3a2_n
        .byte   0
        .endr
.endm

/* SPILL: the masks the non-ascii pass does not touch (N W NL ' ' ') and the end of ends wait in xmm2-4 */
.macro SPILL
        vmovq   xmm2, rdi
        vpinsrq xmm2, xmm2, r8, 1
        vmovq   xmm3, r9
        vpinsrq xmm3, xmm3, r10, 1
        vmovq   xmm4, r11
        vpinsrq xmm4, xmm4, rbx, 1
.endm

/* MASK64 dst, dstd, cadd, cgt: the 64 bytes in ymm0:ymm1 with b + cadd > cgt (signed) as a mask. Uses ymm5, rcx */
.macro MASK64 dst, dstd, cadd, cgt
        vpaddb  ymm5, ymm0, ymmword ptr [rip + SYM(\cadd)]
        vpcmpgtb ymm5, ymm5, ymmword ptr [rip + SYM(\cgt)]
        vpmovmskb \dstd, ymm5
        vpaddb  ymm5, ymm1, ymmword ptr [rip + SYM(\cadd)]
        vpcmpgtb ymm5, ymm5, ymmword ptr [rip + SYM(\cgt)]
        vpmovmskb ecx, ymm5
        shl     rcx, 32
        or      \dst, rcx
.endm

/* CLASS_CP: ebx = the class of code point ebx (rdi = cls_stage1, rbp = cls_stage2); FOLD_S goes to
 * L(na_give_up); ebx = its base, flags from cmp ebx, TOKS_C_L (ja: neither L nor P). Clobbers r8. */
.macro CLASS_CP
        mov     r8d, ebx
        shr     r8d, 8
        movzx   r8d, word ptr [rdi + r8*2]
        shl     r8d, 8
        movzx   ebx, bl
        or      ebx, r8d
        movzx   ebx, byte ptr [rbp + rbx]
        test    ebx, TOKS_C_FOLD_S
        jnz     L(na_give_up)
        and     ebx, TOKS_C_BASE_MASK
        cmp     ebx, TOKS_C_L
.endm

/* CLASS_SLOT (o200k): ecx = the frame slot of code point ebx's class (rax = the slot table, rdi =
 * cls_stage1, rbp = cls_stage2); a class that cuts the block goes to L(na_give_up). Clobbers ebx. */
.macro CLASS_SLOT
        mov     ecx, ebx
        shr     ecx, 8
        movzx   ecx, word ptr [rdi + rcx*2]
        shl     ecx, 8
        movzx   ebx, bl
        or      ebx, ecx
        movzx   ebx, byte ptr [rbp + rbx]
        movzx   ecx, byte ptr [rax + rbx]
        cmp     ecx, 0xFF
        je      L(na_give_up)
.endm

/* K3O2_SLOT c: the slot byte of class byte c for the atom pass (o200k.md §2): upper-only, lower-only and
 * both-group letters, marks (P in both groups), Han letters, N, WS, and P (a scratch slot); 0xFF cuts (FOLD_S,
 * NL, a case flag on N or WS, one flag on P, none on L) */
.macro K3O2_SLOT c
  .if ((\c) & 0x40) != 0
        .byte   0xFF
  .elseif ((\c) & 0x80) != 0
    .if ((\c) & 7) == 1
        .byte   S_AH
    .else
        .byte   0xFF
    .endif
  .elseif ((\c) & 7) == 1
    .if ((\c) & 0x18) == 0x08
        .byte   S_AU
    .elseif ((\c) & 0x18) == 0x10
        .byte   S_AW
    .elseif ((\c) & 0x18) == 0x18
        .byte   S_AB
    .else
#if K3_DSV3
        .byte   S_AB                            /* dsv3: letters have no case group */
#else
        .byte   0xFF
#endif
    .endif
  .elseif ((\c) & 7) == 0
    .if ((\c) & 0x18) == 0
        .byte   S_NWN
    .elseif ((\c) & 0x18) == 0x18
        .byte   S_AM
    .else
        .byte   0xFF
    .endif
  .elseif ((\c) & 0x1F) == 2
        .byte   S_NNA
  .elseif ((\c) & 0x1F) == 3
        .byte   S_WNA
  .else
        .byte   0xFF
  .endif
.endm

/* F_OTHER next: a WS atom (bytes r8) joins W, an N atom joins N; any other class goes to L(na_give_up) */
.macro F_OTHER next
        cmp     ebx, TOKS_C_WS
        je      77f
        cmp     ebx, TOKS_C_N
        jne     L(na_give_up)
        or      qword ptr [rsp + S_NNA], r8
        jmp     \next
77:     or      qword ptr [rsp + S_WNA], r8
        jmp     \next
.endm

/* EMIT: an end r13 + j for each set bit j of rax, ascending, at r12 (advanced). With room for the count
 * + 8, each mask byte stores its 8 table indices + base as one ymm and advances by its popcount (no
 * data-dependent branch; the slack past the count lies inside ends[0, cap)); otherwise one end per
 * bit while room lasts, and L(exit) when ends are left over. Clobbers rax, rcx, rdx, r10, ymm0-2. */
.macro EMIT
        popcnt  rcx, rax
        lea     rdx, [r12 + rcx*4 + 32]
        cmp     rdx, rbx
        ja      87f
        lea     r10, [rip + SYM(k3a2_bitidx)]
        vmovd   xmm1, r13d
        vpbroadcastd ymm1, xmm1
        vmovdqa ymm2, ymmword ptr [rip + SYM(k3a2_d8)]
        .rept   4
        movzx   ecx, al
        movzx   edx, ah
        vpmovzxbd ymm0, qword ptr [r10 + rcx*8]
        vpaddd  ymm0, ymm0, ymm1
        vmovdqu ymmword ptr [r12], ymm0
        popcnt  ecx, ecx
        lea     r12, [r12 + rcx*4]
        vpaddd  ymm1, ymm1, ymm2
        vpmovzxbd ymm0, qword ptr [r10 + rdx*8]
        vpaddd  ymm0, ymm0, ymm1
        vmovdqu ymmword ptr [r12], ymm0
        popcnt  edx, edx
        lea     r12, [r12 + rdx*4]
        vpaddd  ymm1, ymm1, ymm2
        shr     rax, 16
        .endr
        jmp     89f
87:     test    rax, rax
        jz      89f
88:     tzcnt   rdx, rax
        add     edx, r13d
        mov     dword ptr [r12], edx
        add     r12, 4
        blsr    rax, rax
        jz      89f
        cmp     r12, rbx
        jb      88b
        jmp     L(exit)
89:
.endm

FUNC K3_FN
        PROLOGUE 6, 0, LOCALS
        mov     r8, ARG0                        /* t */
        mov     r9, ARG1                        /* a */
        mov     qword ptr [rsp + S_A], r9
        mov     r15, qword ptr [r9 + K3_TEXT]
        mov     r14, qword ptr [r9 + K3_LEN]
        mov     r13, qword ptr [r9 + K3_POS]
        mov     r12, qword ptr [r9 + K3_ENDS]
        mov     qword ptr [rsp + S_ENDS0], r12
        mov     rbx, qword ptr [r9 + K3_CAP]
        lea     rbx, [r12 + rbx*4]
#if K3_DSV3
        mov     eax, dword ptr [r8 + TT_FLAGS]
        shl     rax, 32
        or      rax, TOKS_TP_NO_SLASH           /* dsv3.md §4: o200k's variant without contractions */
#else
        mov     rax, qword ptr [r8 + TT_TMPL_PARAMS]    /* tmpl_params, flags (adjacent dwords) */
#endif
        mov     qword ptr [rsp + S_PARAM], rax
        mov     rax, qword ptr [r8 + TT_CLS_STAGE1]
        mov     qword ptr [rsp + S_STG1], rax
        mov     rax, qword ptr [r8 + TT_CLS_STAGE2]
        mov     qword ptr [rsp + S_STG2], rax
        mov     qword ptr [rsp + S_UNTIL], 0
#if !K3_O200K
        mov     qword ptr [rsp + S_LF], 0       /* C stays 0 without NL_CUT */
#endif
        mov     rax, r14
        lea     rcx, [r13 + 4]
        cmp     rcx, r14
        ja      L(slow)                         /* under 4 bytes: the twin's (kernels.h checks it first) */
        .p2align 6                              /* the hot labels are aligned (kernels.md §3.1) */
L(main):
        cmp     r13, r14
        jae     L(exit)
        cmp     r12, rbx
        jae     L(exit)
        mov     rax, qword ptr [rsp + S_UNTIL]
        cmp     r13, rax
        jb      L(slow)                         /* the pieces through a cut atom */

/* ---- a block at pos -------------------------------------------------------------------------------- */
L(block):
        mov     rax, r14
        sub     rax, r13                        /* rem */
        lea     r10, [r15 + r13]                /* window */
        xor     ebp, ebp                        /* masks >> ebp */
        mov     edx, 64                         /* e */
        cmp     rax, 64
        jae     1f
        mov     rdx, rax
        cmp     r14, 64
        jb      L(short)
        lea     r10, [r15 + r14 - 64]           /* the segment's last 64 bytes, masks shifted down */
        mov     ebp, 64
        sub     ebp, eax
1:      vmovdqu ymm0, ymmword ptr [r10]
        vmovdqu ymm1, ymmword ptr [r10 + 32]
L(classify):
        CLASSIFY
        test    ebp, ebp
        jz      2f
        shrx    rax, rax, rbp
        shrx    rsi, rsi, rbp
        shrx    rdi, rdi, rbp
        shrx    r8, r8, rbp
        shrx    r9, r9, rbp
        shrx    r10, r10, rbp
        shrx    r11, r11, rbp
#if K3_O200K
        shrx    rcx, qword ptr [rsp + S_AU], rbp
        mov     qword ptr [rsp + S_AU], rcx
        shrx    rcx, qword ptr [rsp + S_AW], rbp
        mov     qword ptr [rsp + S_AW], rcx
#endif
#if K3_DSV3
        shrx    rcx, qword ptr [rsp + S_DM], rbp
        mov     qword ptr [rsp + S_DM], rcx
        shrx    rcx, qword ptr [rsp + S_DX], rbp
        mov     qword ptr [rsp + S_DX], rcx
#elif !K3_O200K
        shrx    rcx, qword ptr [rsp + S_LF], rbp
        mov     qword ptr [rsp + S_LF], rcx
#endif
2:      mov     qword ptr [rsp + S_CONT], 0
#if K3_O200K
        xor     ecx, ecx
        mov     qword ptr [rsp + S_AB], rcx
        mov     qword ptr [rsp + S_AM], rcx
        mov     qword ptr [rsp + S_AH], rcx
#endif
        mov     rcx, rdx                        /* the cut: none */
        test    rax, rax
        jnz     L(blk_na)
        .p2align 5
L(blk_cut):                                     /* rcx: the first atom the blocks leave to the twin */
#if K3_DSV3
        tzcnt   rax, qword ptr [rsp + S_DX]
        cmp     rax, rcx
        cmovb   rcx, rax                        /* an ascii X atom (D7's gaps are the twin's) */
#endif
        mov     rax, qword ptr [rsp + S_UNTIL]
        lea     rbp, [r13 + rcx + 1]
        cmp     rcx, rdx
        cmovb   rdx, rcx                        /* e = the cut */
        cmovb   rax, rbp                        /* and the twin's through the piece holding it */
        mov     qword ptr [rsp + S_UNTIL], rax
        test    rdx, rdx
        jz      L(stuck)                        /* the block starts with that atom */
        mov     qword ptr [rsp + S_E], rdx
        mov     rax, -1
        bzhi    rax, rax, rdx                   /* V: bits >= e of the other masks only reach B bits >= e,
                                                   which the bzhi before q (or V at the end) drops */
        and     r9, rax
        andn    rcx, r8, rax
#if K3_DSV3
        andn    rcx, rdi, rcx
        mov     rbp, qword ptr [rsp + S_AH]
        andn    rcx, rbp, rcx                   /* D5: an N or CJK atom ends the region, no give-back before it */
#elif !K3_O200K
        test    byte ptr [rsp + S_PARAM + 1], TOKS_TP_DIGIT_CUT >> 8
        jz      1f
        andn    rcx, rdi, rcx                   /* A8: an N atom ends the regex's input, no give-back before it */
1:
#endif
        shr     rcx, 1
        mov     qword ptr [rsp + S_NWN], rcx    /* a non-W byte follows */
#if K3_O200K
        /* o200k.md §3 over the block. Letters are the case groups: U upper-only, W lower-only, B' both (Lm Lo,
         * and the marks a case piece takes); P every base-P byte (marks, ' and / included); H Han letters.
         * ' and N wait in S_NNA and S_WNA. */
        mov     qword ptr [rsp + S_NNA], r11
        mov     qword ptr [rsp + S_WNA], rdi
        mov     rbp, qword ptr [rsp + S_AU]
        and     rbp, rax
        mov     qword ptr [rsp + S_AU], rbp     /* U */
        mov     rcx, qword ptr [rsp + S_AW]
        and     rcx, rax
        mov     qword ptr [rsp + S_AW], rcx     /* W */
        or      rbp, rcx
        or      rbp, qword ptr [rsp + S_AB]
        and     rbp, rax                        /* letters (base L, Han aside) */
#if K3_DSV3
        mov     rcx, qword ptr [rsp + S_AU]
        or      rcx, qword ptr [rsp + S_AW]
        mov     qword ptr [rsp + S_DA], rcx     /* the ascii letters (D1) */
        xor     ecx, ecx
        mov     qword ptr [rsp + S_AU], rcx
        mov     qword ptr [rsp + S_AW], rcx     /* no case groups: a letter run is whole */
#endif
        mov     rcx, qword ptr [rsp + S_AH]
        and     rcx, rax
        mov     qword ptr [rsp + S_AH], rcx     /* H */
        or      rcx, rbp
        or      rcx, rdi
        or      rcx, r8
        andn    rdx, rcx, rax                   /* P */
        test    byte ptr [rsp + S_PARAM + 1], TOKS_TP_NO_SLASH >> 8
        jz      1f
        xor     esi, esi
1:      or      rsi, r9                         /* X: the tail's bytes, CR LF / (NO_SLASH: CR LF) */
        mov     rcx, qword ptr [rsp + S_AM]
        and     rcx, rax                        /* marks */
        jnz     L(o2_marks)
        lea     rcx, [rdx + rdx]
        and     rcx, r9                         /* no marks: a P run that a CR or LF follows is O3's */
        .p2align 5
L(o2_tail):                                     /* rcx: the tails' first bytes; rdx P, rbp C (every case atom) */
        lea     rax, [rsi + rcx]
        andn    rax, rax, rsi
        or      rax, rcx                        /* the tails: the X run from each */
        andn    rdx, rax, rdx                   /* P* = P without them ('/' only) */
        andn    r8, rax, r8                     /* W* = W without them (CR LF) */
        mov     qword ptr [rsp + S_AM], rbp     /* C */
        mov     rcx, rdx
        or      rcx, r10
        add     rcx, rcx
        andn    rsi, rcx, rdx                   /* P* run starts not after ' ' */
        /* a case run's start goes to the atom before it when that is WS, or a P* run of one atom that does
         * not follow ' ' (E: the byte after each P* run with a second atom or after ' ') */
        mov     rdi, qword ptr [rsp + S_CONT]
        lea     rax, [rdx + rdx]
        andn    rax, rdi, rax
        lea     rcx, [r10 + r10]
        or      rax, rcx
        and     rax, rdx
        add     rax, rdx
        andn    rax, rdx, rax                   /* E */
        andn    rcx, r9, r8                     /* S */
#if K3_DSV3
        mov     r11, qword ptr [rsp + S_DM]
        and     r11, rdx
        add     r11, r11
        and     r11, qword ptr [rsp + S_DA]     /* D1: an ascii P atom before an ascii letter */
        add     rcx, rcx                        /* D2: WS before a letter (X atoms cut) */
        or      rcx, r11
        andn    rcx, rax, rcx                   /* pre */
        andn    r11, rax, r11                   /* the D1 runs' first letters */
        mov     rax, qword ptr [rsp + S_DA]
        add     r11, rax
        andn    r11, rax, r11                   /* the byte after each D1 run's ascii letters */
        andn    rax, rax, rbp
        and     r11, rax
        or      rsi, r11                        /* a non-ascii letter there starts a piece (D1 takes A-Z a-z) */
#else
        or      rcx, rdx
        add     rcx, rcx
        andn    rcx, rax, rcx                   /* pre */
#endif
        lea     rax, [rbp + rbp]
        or      rcx, rax
        andn    rax, rcx, rbp
        or      rsi, rax                        /* case run starts no prefix takes */
        or      rsi, 1                          /* B */
        /* the two-state walk (o200k.md §4): an upper-only atom after a lower-only one, B' atoms between */
        mov     rdi, qword ptr [rsp + S_AU]
        mov     rax, qword ptr [rsp + S_AW]
        mov     rcx, rdi
        or      rcx, rax
        andn    rcx, rcx, rbp                   /* B' */
        add     rax, rax
        mov     rdx, rax
        and     rax, rcx
        add     rax, rcx
        andn    rax, rcx, rax                   /* the byte after each B' run that follows a W */
        andn    rdx, rcx, rdx
        or      rax, rdx                        /* G */
        and     rax, rdi
        or      rsi, rax                        /* U & G */
        /* an upper-only run after B' and not in G ends its piece's UP phase: L1 backs up to the last LO
         * atom when the case run ends with it */
        lea     rcx, [rcx + rcx]
        and     rcx, rdi
        andn    rcx, rax, rcx
        jnz     L(o2_bt)
L(o2_btd):
        mov     rax, qword ptr [rsp + S_AH]
        lea     rcx, [rax + rax]
        andn    rax, rcx, rax
        or      rsi, rax                        /* Han run starts (O0) */
        mov     rdi, qword ptr [rsp + S_WNA]    /* N */
        andn    rbp, r9, r8                     /* S */
        jmp     L(digits)
L(o2_bt):                                       /* rcx: the candidates x; rdi U, rsi B */
        mov     rdx, qword ptr [rsp + S_AM]     /* C */
        mov     r11, qword ptr [rsp + S_E]
1:      tzcnt   rbp, rcx
        blsr    rcx, rcx
        shrx    rax, rdi, rbp
        not     rax
        tzcnt   rax, rax
        add     rax, rbp                        /* the end of x's upper-only run */
        cmp     rax, r11
        jb      2f
        lea     rax, [r13 + r11]
        cmp     rax, r14
        jne     3f                              /* it reaches e: the next block decides */
        jmp     4f
2:      bt      rdx, rax
        jc      3f                              /* the case run goes on */
4:      bts     rsi, rbp
3:      test    rcx, rcx
        jnz     1b
        jmp     L(o2_btd)
L(o2_marks):                                    /* rcx: the marks; rdx P, rsi X, rbp letters. A mark is a case
                                                 * atom unless O3 took its P run first (o200k.md §3) */
        mov     qword ptr [rsp + S_AM], rcx
        mov     qword ptr [rsp + S_P1], rdx     /* P */
        mov     qword ptr [rsp + S_TAILS], 0    /* the '/' last round's tails took */
L(o2_mloop):
        mov     rax, rbp
        or      rax, rcx
        mov     rdi, qword ptr [rsp + S_CONT]
        or      rax, rdi
        shr     rax, 1                          /* the atom goes on, or a case atom follows */
        andn    rax, rax, rdx
        andn    rax, rcx, rax                   /* the last bytes of non-mark P atoms that no case atom follows */
        .rept   3
        mov     r11, rax
        and     r11, rdi
        shr     r11, 1
        or      rax, r11                        /* whole atoms */
        .endr
        lea     r11, [r10 + r10]
        and     r11, rdx
        andn    r11, rcx, r11
        or      rax, r11                        /* and a non-mark P atom after ' ': O3 starts */
        lea     r11, [rdx + rax]
        andn    r11, r11, rdx
        or      rax, r11                        /* Q: each P run from its first O3 start on */
        andn    rcx, rax, rcx                   /* the marks before Q: case atoms */
        lea     r11, [rax + rax]
        and     r11, r9                         /* a CR or LF right after Q starts a tail */
        lea     rax, [rsi + r11]
        andn    rax, rax, rsi
        or      rax, r11
        and     rax, qword ptr [rsp + S_P1]     /* the '/' the tails take */
        cmp     rax, qword ptr [rsp + S_TAILS]
        je      1f                              /* as last round: a tail depends only on what lies left of it,
                                                   so the rounds settle from the left */
        mov     qword ptr [rsp + S_TAILS], rax
        mov     rdx, qword ptr [rsp + S_P1]
        andn    rdx, rax, rdx                   /* P without them: Q again */
        mov     rcx, qword ptr [rsp + S_AM]
        jmp     L(o2_mloop)
1:      or      rbp, rcx                        /* C */
        andn    rdx, rcx, rdx
        mov     rcx, r11
        jmp     L(o2_tail)
#else
        test    byte ptr [rsp + S_PARAM + 1], TOKS_TP_NL_CUT >> 8
        jz      1f
        /* A9: the cuts C, where a (\r?\n)+ run starts: an LF not after a CR / LF, a CR before an LF not after an LF */
        mov     rcx, qword ptr [rsp + S_LF]     /* LF */
        lea     rdx, [r9 + r9]
        andn    rdx, rdx, rcx                   /* LF & ~(NL << 1) */
        lea     rbp, [rcx + rcx]
        or      rbp, rcx
        andn    rbp, rbp, r9                    /* CR & ~(LF << 1) */
        shr     rcx, 1
        and     rbp, rcx                        /* ... & (LF >> 1) */
        or      rdx, rbp
        and     rdx, rax
        mov     qword ptr [rsp + S_LF], rdx     /* C (in play) */
1:      mov     rdx, rsi
        or      rdx, rdi
        or      rdx, r8
        andn    rax, rdx, rax                   /* P */
        mov     rdx, rax
        or      rdx, r10
        add     rdx, rdx
        andn    rdx, rdx, rax                   /* Pst: P run starts not after ' ' */
        and     r11, rdx
        mov     qword ptr [rsp + S_CAND], r11   /* ' at piece starts */
        andn    rbp, r9, r8                     /* S */
        /* letters: L run starts, minus those an admitted prefix atom before them takes */
        test    byte ptr [rsp + S_PARAM], TOKS_TP_LPREFIX_ANY
        jz      5f
        /* a P atom prefixes when its P run is that one atom and not after ' ': E = the byte after
         * each P run that holds a second atom or starts after ' ' */
        lea     r11, [rax + rax]
        mov     rcx, qword ptr [rsp + S_CONT]
        andn    rcx, rcx, r11                   /* (P << 1) & ~cont */
        lea     r11, [r10 + r10]
        or      rcx, r11
        and     rcx, rax
        add     rcx, rax
        andn    rcx, rax, rcx                   /* E */
        mov     r11, rbp
        or      r11, rax
        add     r11, r11
        andn    r11, rcx, r11                   /* pre = ((S | P) << 1) & ~E */
        jmp     6f
5:      lea     r11, [r10 + r10]                /* pre = SP << 1 */
6:      lea     r11, [r11 + rsi*2]              /* | L << 1 (disjoint classes) */
        andn    rsi, r11, rsi
        or      rsi, rdx
        or      rsi, 1                          /* B */
        or      rsi, qword ptr [rsp + S_LF]     /* A9: every cut starts a piece */
        test    byte ptr [rsp + S_PARAM], TOKS_TP_PUNCT_NL
        jz      L(digits)
        mov     rcx, qword ptr [rsp + S_LF]
        andn    rcx, rcx, r9                    /* the CR/LF bytes but the cuts (A9: a tail stops before one) */
        add     rax, rax
        and     rax, rcx                        /* CR/LF right after P */
        add     rax, rcx
        andn    rax, rax, rcx                   /* the CR/LF runs they start (absorbed by A4) */
        andn    r8, rax, r8                     /* W* */
#endif
L(digits):                                      /* digits */
        lea     rax, [rdi + rdi]
        andn    rax, rax, rdi                   /* N run starts */
        mov     edx, dword ptr [rsp + S_PARAM]
        and     edx, TOKS_TP_DIGITS_MASK
        jz      L(d13)
        cmp     edx, TOKS_TP_DIGITS_1
        je      L(d1)
        cmp     edx, TOKS_TP_DIGITS_SP_RUN
        je      L(dsp)
        or      rsi, rax
        jmp     L(ddone)
L(d1):  mov     rdx, qword ptr [rsp + S_CONT]
        andn    rdx, rdx, rdi                   /* each N atom's first byte */
        or      rsi, rdx
        jmp     L(ddone)
L(dsp): lea     rdx, [r10 + r10]
        andn    rax, rdx, rax
        or      rsi, rax
        jmp     L(ddone)
L(d13): or      rsi, rax
        mov     rdx, qword ptr [rsp + S_CONT]
        test    rdx, rdi
        jnz     L(d13a)                         /* an N atom of 2+ bytes: count atoms */
        lea     rdx, [rdi + rdi]
        and     rdx, rdi
        lea     r11, [rdx + rdx]
        and     rdx, r11                        /* N with two N before it */
7:      shl     rax, 3
        and     rax, rdx
        jz      L(ddone)
        or      rsi, rax
        jmp     7b
L(d13a):                                        /* each run's 1st, 4th, 7th ... N atom (rax: run starts) */
        andn    rdx, rdx, rdi                   /* N atom starts */
4:      tzcnt   r11, rdx
        blsr    rdx, rdx
        bt      rax, r11
        jc      5f
        dec     ecx
        jnz     6f
5:      bts     rsi, r11
        mov     ecx, 3
6:      test    rdx, rdx
        jnz     4b
L(ddone):
        mov     rcx, qword ptr [rsp + S_NWN]
        /* whitespace */
        lea     rax, [r8 + r8]
        andn    rax, rax, r8
        or      rsi, rax                        /* W* run starts */
#if !K3_O200K
        test    byte ptr [rsp + S_PARAM], TOKS_TP_WS_NL
        jz      L(w_plain)
#endif
        lea     rax, [r9 + r9]
        and     rax, rbp                        /* C: S right after NL */
        lea     rdx, [rbp + rax]
        andn    rdx, rbp, rdx                   /* the byte after each such S run */
        and     rdx, r9
#if !K3_O200K
        mov     r11, qword ptr [rsp + S_LF]
        andn    rdx, r11, rdx                   /* ... a NL but no cut (A9: then that S run is its stretch's tail) */
#endif
        jnz     L(w_fix)
L(w_c):  or      rsi, rax
        and     rcx, rbp                        /* A6 give-back: the last atom of the run */
        jmp     L(w_gb)
#if !K3_O200K
L(w_plain):
        and     rcx, r8
#endif
L(w_gb):                                        /* (its last byte, moved back to the atom's first) */
        mov     rdx, qword ptr [rsp + S_CONT]
        test    rdx, rdx
        jz      9f
        .rept   3
        mov     rax, rcx
        and     rax, rdx
        andn    rcx, rdx, rcx
        shr     rax, 1
        or      rcx, rax
        .endr
9:      or      rsi, rcx
        jmp     L(w_done)
L(w_fix):                                        /* drop the C bit of each S run that ends at a NL */
        mov     r10, rbp
        not     r10
8:      tzcnt   rdi, rdx
        blsr    rdx, rdx
        bzhi    r11, r10, rdi
        lzcnt   r11, r11
        neg     r11
        add     r11, 64
        btr     rax, r11
        test    rdx, rdx
        jnz     8b
        jmp     L(w_c)
L(w_done):
#if K3_O200K
        /* the suffix K (CONTR_CI): a ' right after a case atom ends that letter piece, and K takes ' and one
         * or two letters with it (o200k.md §3): their starts go, and a case atom right after it starts a
         * piece. A suffix the block cannot decide stops the decided range at its ' (S_NNA: the limit, 64 none) */
        mov     rax, qword ptr [rsp + S_NNA]    /* ' */
        mov     rcx, qword ptr [rsp + S_AM]     /* C */
        add     rcx, rcx
        and     rax, rcx
        mov     qword ptr [rsp + S_NNA], 64
        jz      L(c_done)
        test    byte ptr [rsp + S_PARAM], TOKS_TP_CONTR_MASK
        jz      L(c_done)
        mov     rdx, qword ptr [rsp + S_E]
        lea     rbp, [rip + SYM(k3a2_contr)]
        mov     qword ptr [rsp + S_WNA], 0      /* the last suffix's end */
L(k_loop):
        tzcnt   rcx, rax                        /* a */
        blsr    rax, rax
        cmp     rcx, qword ptr [rsp + S_WNA]
        je      L(k_next)                       /* after a suffix: that ' starts a piece (K comes once) */
        lea     r10, [r13 + rcx + 1]            /* c' */
        cmp     r10, r14
        jae     L(k_next)
        movzx   r11d, byte ptr [r15 + r10]
        cmp     r11d, 0x80
        jae     L(k_na)
        or      r11d, 0x20
        movzx   r11d, byte ptr [rbp + r11]      /* 0x80: s t m d; e l: the second letter of 're 've 'll */
        test    r11d, r11d
        jz      L(k_next)
        lea     r9, [rcx + 2]                   /* z */
        test    r11d, 0x80
        jnz     L(k_hit)
        lea     r10, [r13 + r9]
        cmp     r10, r14
        jae     L(k_next)
        movzx   edi, byte ptr [r15 + r10]
        or      edi, 0x20
        cmp     edi, r11d
        jne     L(k_next)
        inc     r9
L(k_hit):
        mov     qword ptr [rsp + S_WNA], r9
        mov     r10, r9
        sub     r10, rcx
        mov     edi, 1
        shlx    rdi, rdi, r10
        dec     rdi
        shlx    rdi, rdi, rcx
        andn    rsi, rdi, rsi                   /* no start at ' or the suffix's letters */
        cmp     r9, rdx
        jae     L(k_far)
        mov     r10, qword ptr [rsp + S_AM]
        bt      r10, r9
        jnc     L(k_next)                       /* z in no case group: a start already */
        mov     r10, qword ptr [rsp + S_AU]
        or      r10, qword ptr [rsp + S_AW]
        bt      r10, r9
        jnc     L(k_defer)                      /* both groups: the walk from z is the next block's */
        bts     rsi, r9                         /* upper- or lower-only: the walk restarts at z */
        jmp     L(k_next)
L(k_far):                                       /* z at or past e */
        lea     r10, [r13 + r9]
        cmp     r10, r14
        jae     L(k_next)
        cmp     byte ptr [r15 + r10], 0x80
        jb      L(k_next)                       /* ascii: whatever starts there lies past the decided range */
        jmp     L(k_defer)
L(k_na):                                        /* c' not ascii: a suffix only with FOLD_S, which cuts blocks */
        lea     r10, [rcx + 1]
        cmp     r10, rdx
        jb      L(k_next)
L(k_defer):
        mov     qword ptr [rsp + S_NNA], rcx    /* ascending: the first is the limit */
        jmp     L(c_done)
L(k_next):
        test    rax, rax
        jnz     L(k_loop)
#else
        /* A1 contractions at the ' piece starts */
        mov     ebp, dword ptr [rsp + S_PARAM]
        and     ebp, TOKS_TP_CONTR_MASK
        jz      L(c_done)
        mov     rax, qword ptr [rsp + S_CAND]
        test    rax, rax
        jz      L(c_done)
        xor     r9d, r9d
        cmp     ebp, TOKS_TP_CONTR_CI
        jne     1f
        mov     r9d, 0x20                       /* (?i) on ascii */
1:      mov     rdx, qword ptr [rsp + S_E]
        lea     rbp, [rip + SYM(k3a2_contr)]
L(c_loop):
        tzcnt   rcx, rax                        /* i */
        blsr    rax, rax
        lea     rdi, [rcx + 1]
        cmp     rdi, rdx
        jae     L(c_next)                       /* ' at e - 1: deferred with its run, or final */
        lea     r10, [r13 + rcx + 2]            /* c2's offset */
        movzx   r11d, byte ptr [r15 + r10 - 1]
        or      r11d, r9d
        movzx   r11d, byte ptr [rbp + r11]      /* t: 0x80 one-letter, or the second letter wanted, or 0 */
        xor     ecx, ecx                        /* c2: 0 past len */
        cmp     r10, r14
        jae     2f
        movzx   ecx, byte ptr [r15 + r10]
        or      ecx, r9d
2:      xor     r10d, r10d
        cmp     r11d, ecx
        sete    r10b
        xor     ecx, ecx
        test    r11d, 0x7F
        setnz   cl
        and     r10d, ecx                       /* m3: 're 've 'll */
        shr     r11d, 7                         /* m1: 's 't 'm 'd */
        lea     ecx, [r11 + r10]
        lea     r10d, [r11 + r10*2]
        add     r10d, r10d
        shlx    rcx, rcx, rdi                   /* c1 is inside the piece */
        shlx    r10, r10, rdi                   /* the start after it (past bit 63: dropped) */
        andn    rsi, rcx, rsi
        or      rsi, r10
L(c_next):
        test    rax, rax
        jnz     L(c_loop)
#endif
L(c_done):
        mov     rdx, qword ptr [rsp + S_E]
#if K3_O200K
        cmp     qword ptr [rsp + S_NNA], 64
        jb      L(c_part)                       /* a deferred suffix: the segment's end waits too */
#endif
        lea     rax, [r13 + rdx]
        cmp     rax, r14
        je      L(final)
L(c_part):
        /* u + 1 = e - max(r - 1, 0), r = the length of the W* run ending at byte e - 1 */
        mov     ecx, 64
        sub     ecx, edx
        shlx    rax, r8, rcx
        not     rax
        lzcnt   rax, rax
        sub     eax, 1
        adc     eax, 0
        mov     ecx, edx
        sub     ecx, eax
#if K3_O200K
        mov     rax, qword ptr [rsp + S_NNA]
        cmp     rcx, rax
        cmova   rcx, rax                        /* not past a deferred suffix's ' */
#endif
        bzhi    rax, rsi, rcx                   /* B in [0, u] */
        lzcnt   rcx, rax
        xor     ecx, 63                         /* q */
        jz      L(stuck)
        lea     r9, [r13 + rcx]                 /* the next block */
        and     rax, -2
        EMIT
        mov     r13, r9
        jmp     L(main)
L(final):                                       /* the block ends the segment: B in (0, e), then len */
        mov     rax, -1
        bzhi    rax, rax, rdx
        and     rax, rsi
        and     rax, -2
        EMIT
        cmp     r12, rbx
        jae     L(exit)
        mov     dword ptr [r12], r14d
        add     r12, 4
        mov     r13, r14
        jmp     L(exit)
/* non-ascii bytes in the block (rax): each atom's class goes to every byte of it; L atoms join L, P atoms
 * (and ill-formed bytes) stay P; cont marks the bytes after an atom's first. The first other atom (N,
 * WS, FOLD_S, a malformed class) cuts the block. The masks the pass does not touch wait in xmm2-5. */
        .p2align 5
L(blk_na):
        mov     qword ptr [rsp + S_E], rdx
        popcnt  rcx, rax
        cmp     ecx, 8
        jb      L(na_sparse)
        /* a dense block under TOKS_TF_CJK_L: for every byte, the cp >> 8 it would give as the lead of a
         * 3-byte atom, tested against the CJK ideograph / Yi (4E-A3) and hangul (AC-D6) blocks (dsv3: 4E-9E);
         * S_NWN holds the mask until the leads are known */
        test    byte ptr [rsp + S_PARAM + 4], K3_TF_CJK
        jz      2f
        vperm2i128 ymm5, ymm0, ymm1, 0x21
        vpalignr ymm4, ymm5, ymm0, 1            /* bytes 1-32 */
        vperm2i128 ymm5, ymm1, ymm1, 0x81
        vpalignr ymm5, ymm5, ymm1, 1            /* bytes 33-63, 0 */
        vpsrlw  ymm4, ymm4, 2
        vpsrlw  ymm5, ymm5, 2
        vpand   ymm4, ymm4, ymmword ptr [rip + SYM(k3a2_c0f)]
        vpand   ymm5, ymm5, ymmword ptr [rip + SYM(k3a2_c0f)]
        vpand   ymm2, ymm0, ymmword ptr [rip + SYM(k3a2_c0f)]
        vpand   ymm3, ymm1, ymmword ptr [rip + SYM(k3a2_c0f)]
        vpsllw  ymm2, ymm2, 4
        vpsllw  ymm3, ymm3, 4
        vpor    ymm2, ymm2, ymm4
        vpor    ymm3, ymm3, ymm5
#if K3_DSV3
        vpaddb  ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_ce1)]
        vpaddb  ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_ce1)]
        vpcmpgtb ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c2e)]
        vpcmpgtb ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c2e)]
#else
        vpaddb  ymm4, ymm2, ymmword ptr [rip + SYM(k3a2_cdc)]
        vpaddb  ymm5, ymm3, ymmword ptr [rip + SYM(k3a2_cdc)]
        vpcmpgtb ymm4, ymm4, ymmword ptr [rip + SYM(k3a2_c29)]
        vpcmpgtb ymm5, ymm5, ymmword ptr [rip + SYM(k3a2_c29)]
        vpaddb  ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_ca9)]
        vpaddb  ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_ca9)]
        vpcmpgtb ymm2, ymm2, ymmword ptr [rip + SYM(k3a2_c54)]
        vpcmpgtb ymm3, ymm3, ymmword ptr [rip + SYM(k3a2_c54)]
        vpor    ymm2, ymm2, ymm4
        vpor    ymm3, ymm3, ymm5
#endif
        vpmovmskb ecx, ymm3
        shl     rcx, 32
        mov     qword ptr [rsp + S_NWN], rcx
        vpmovmskb ecx, ymm2
        or      rcx, qword ptr [rsp + S_NWN]
        shrx    rcx, rcx, rbp
        mov     qword ptr [rsp + S_NWN], rcx
2:      SPILL
        /* fast path: every non-ascii byte is a 2- or 3-byte lead or a continuation byte some lead calls
         * for; then only E0/ED leads need a range check. A lead whose atom would cross e shrinks e to it
         * (the next block starts at or before it) */
        MASK64  r8, r8d, k3a2_cc0, k3a2_c3f     /* 80-BF */
        MASK64  r9, r9d, k3a2_ca0, k3a2_c61     /* C2-DF */
        MASK64  r10, r10d, k3a2_c90, k3a2_c6f   /* E0-EF */
        shrx    r8, r8, rbp
        shrx    r9, r9, rbp
        shrx    r10, r10, rbp
        mov     rdx, qword ptr [rsp + S_E]
        lea     ecx, [rdx - 1]
        mov     ebx, 1
        shlx    rbx, rbx, rcx                   /* bit e - 1 */
        mov     r11, rbx
        shr     r11, 1
        or      r11, rbx                        /* bits e - 2, e - 1 */
        and     rbx, r9
        and     r11, r10
        or      rbx, r11                        /* leads whose atoms would cross e */
        jz      1f
        tzcnt   rdx, rbx
        mov     qword ptr [rsp + S_E], rdx      /* e = the first of them */
        bzhi    rax, rax, rdx
        bzhi    r8, r8, rdx
        bzhi    r9, r9, rdx
        bzhi    r10, r10, rdx
1:      lea     rcx, [r9 + r10]
        lea     r11, [rcx + rcx]
        lea     rbx, [r10*4]
        or      r11, rbx
        cmp     r11, r8
        jne     L(na_slow)                      /* the continuation bytes are not exactly the leads' */
        or      rcx, r8
        cmp     rcx, rax
        jne     L(na_slow)                      /* other non-ascii bytes (4-byte leads, C0 C1 F5-FF) */
        mov     qword ptr [rsp + S_P1], rax     /* kept for L(na_give_up) */
        mov     qword ptr [rsp + S_CONT], r8
        xor     edx, edx                        /* leads of 3-byte L atoms */
        test    byte ptr [rsp + S_PARAM + 4], K3_TF_CJK
        jz      L(uni_done)
        mov     rdx, qword ptr [rsp + S_NWN]
        and     rdx, r10                        /* 3-byte leads in those blocks: L, no lookup */
        andn    r10, rdx, r10
L(uni_done):
        mov     rdi, qword ptr [rsp + S_STG1]
        mov     rbp, qword ptr [rsp + S_STG2]
#if K3_O200K
        lea     r8, [rdx*8]
        sub     r8, rdx
        or      qword ptr [rsp + S_CJK], r8     /* the CJK leads' atoms: letters in both groups (dsv3: Han) */
        lea     rax, [rip + SYM(k3o2_slot)]
        xor     edx, edx
        mov     qword ptr [rsp + S_NNA], rdx
        mov     qword ptr [rsp + S_WNA], rdx
        test    r9, r9
        jz      L(o2f3s)
L(o2f2):                                        /* each atom's bytes go to its class's slot */
        tzcnt   rcx, r9
        blsi    r11, r9
        blsr    r9, r9
        add     rcx, r13
        movzx   ebx, word ptr [r15 + rcx]       /* b0 | b1 << 8 */
        mov     r8d, ebx
        and     r8d, 0x1F
        shl     r8d, 6
        shr     ebx, 8
        and     ebx, 0x3F
        or      ebx, r8d                        /* cp */
        lea     r8, [r11 + r11*2]
        CLASS_SLOT
        or      qword ptr [rsp + rcx], r8
        test    r9, r9
        jnz     L(o2f2)
L(o2f3s):
        test    r10, r10
        jz      L(o2f_done)
L(o2f3):
        tzcnt   rcx, r10
        blsi    r11, r10
        blsr    r10, r10
        add     rcx, r13
        movzx   r8d, byte ptr [r15 + rcx]
        movzx   ebx, word ptr [r15 + rcx + 1]   /* b1 | b2 << 8 */
        and     r8d, 0x0F
        shl     r8d, 12
        mov     ecx, ebx
        and     ecx, 0x3F
        shl     ecx, 6
        or      r8d, ecx
        shr     ebx, 8
        and     ebx, 0x3F
        or      ebx, r8d                        /* cp */
        cmp     ebx, 0x800
        jb      L(na_give_up)                   /* E0 80-9F: overlong */
        lea     ecx, [rbx - 0xD800]
        cmp     ecx, 0x800
        jb      L(na_give_up)                   /* ED A0-BF: surrogate */
        lea     r8, [r11*8]
        sub     r8, r11
        CLASS_SLOT
        or      qword ptr [rsp + rcx], r8
        test    r10, r10
        jnz     L(o2f3)
L(o2f_done):
        mov     rcx, qword ptr [rsp + S_E]      /* no cut */
        jmp     L(na_fin)
#else
        xor     eax, eax                        /* leads of 2-byte L atoms */
        mov     qword ptr [rsp + S_NNA], rax
        mov     qword ptr [rsp + S_WNA], rax
        test    r9, r9
        jz      L(f3_start)
L(f2):  tzcnt   rcx, r9
        blsi    r11, r9
        blsr    r9, r9
        add     rcx, r13
        movzx   ebx, word ptr [r15 + rcx]       /* b0 | b1 << 8 */
        mov     r8d, ebx
        and     r8d, 0x1F
        shl     r8d, 6
        shr     ebx, 8
        and     ebx, 0x3F
        or      ebx, r8d                        /* cp */
        CLASS_CP
        ja      L(f2_other)
        mov     r8d, 0
        cmove   r8, r11
        or      rax, r8
L(f2_next):
        test    r9, r9
        jnz     L(f2)
L(f3_start):
        test    r10, r10
        jz      L(f_done)
L(f3):  tzcnt   rcx, r10
        blsi    r11, r10
        blsr    r10, r10
        add     rcx, r13
        movzx   r8d, byte ptr [r15 + rcx]
        movzx   ebx, word ptr [r15 + rcx + 1]   /* b1 | b2 << 8 */
        and     r8d, 0x0F
        shl     r8d, 12
        mov     ecx, ebx
        and     ecx, 0x3F
        shl     ecx, 6
        or      r8d, ecx
        shr     ebx, 8
        and     ebx, 0x3F
        or      ebx, r8d                        /* cp */
        cmp     ebx, 0x800
        jb      L(na_give_up)                   /* E0 80-9F: overlong */
        lea     ecx, [rbx - 0xD800]
        cmp     ecx, 0x800
        jb      L(na_give_up)                   /* ED A0-BF: surrogate */
        CLASS_CP
        ja      L(f3_other)
        mov     r8d, 0
        cmove   r8, r11
        or      rdx, r8
L(f3_next):
        test    r10, r10
        jnz     L(f3)
L(f_done):
        lea     r8, [rax + rax*2]               /* 2-byte L atoms: lead and lead + 1 */
        or      rsi, r8
        lea     r8, [rdx*8]
        sub     r8, rdx                         /* 3-byte: lead .. lead + 2 */
        or      rsi, r8
        mov     rcx, qword ptr [rsp + S_E]      /* no cut */
        jmp     L(na_fin)
L(f2_other):                                    /* ebx = base, r11 = the lead's bit */
        lea     r8, [r11 + r11*2]
        F_OTHER L(f2_next)
L(f3_other):
        lea     r8, [r11*8]
        sub     r8, r11
        F_OTHER L(f3_next)
#endif
L(na_give_up):                                  /* a class or form the fast path leaves to the ordered pass */
        mov     rax, qword ptr [rsp + S_P1]
        jmp     L(na_slow)
L(na_sparse):
        SPILL
L(na_slow):
        mov     rdi, qword ptr [rsp + S_STG1]
        mov     rbp, qword ptr [rsp + S_STG2]
        xor     r9d, r9d                        /* cont */
        mov     qword ptr [rsp + S_NNA], r9
        mov     qword ptr [rsp + S_WNA], r9
#if K3_O200K
        lea     r11, [rip + SYM(k3o2_slot)]
        mov     qword ptr [rsp + S_P1], r11     /* the slot table */
#endif
L(na_loop):
        tzcnt   rcx, rax                        /* j */
        lea     r8, [r13 + rcx]
        movzx   r11d, byte ptr [r15 + r8]
        NA_ATOM r8, edx, r10d, r11d, r11, ebx, rbx
#if K3_O200K
        mov     rbx, -1
        bzhi    rbx, rbx, r10
        shlx    rbx, rbx, rcx                   /* the atom's bytes */
        mov     r11, qword ptr [rsp + S_P1]
        movzx   edx, byte ptr [r11 + rdx]
        cmp     edx, 0xFF
        je      L(na_cut)
        or      qword ptr [rsp + rdx], rbx      /* its slot */
#else
        test    edx, TOKS_C_FOLD_S
        jnz     L(na_cut)
        mov     r11d, edx
        and     r11d, TOKS_C_BASE_MASK
        mov     rbx, -1
        bzhi    rbx, rbx, r10
        shlx    rbx, rbx, rcx                   /* the atom's bytes */
        cmp     r11d, TOKS_C_L
        ja      L(na_other)
        mov     edx, 0
        cmove   rdx, rbx
        or      rsi, rdx                        /* L joins L, P stays P */
#endif
L(na_next):
        andn    rax, rbx, rax
        btr     rbx, rcx
        or      r9, rbx
        test    rax, rax
        jnz     L(na_loop)
        mov     rcx, qword ptr [rsp + S_E]      /* no cut */
L(na_cut):
        mov     qword ptr [rsp + S_CONT], r9
L(na_fin):
        mov     rdx, qword ptr [rsp + S_E]
        vmovq   rdi, xmm2
        vpextrq r8, xmm2, 1
        or      rdi, qword ptr [rsp + S_NNA]
        or      r8, qword ptr [rsp + S_WNA]
        vmovq   r9, xmm3
        vpextrq r10, xmm3, 1
        vmovq   r11, xmm4
        vpextrq rbx, xmm4, 1
        jmp     L(blk_cut)
#if !K3_O200K
L(na_other):                                    /* WS joins W, N joins N */
        cmp     r11d, TOKS_C_WS
        je      1f
        cmp     r11d, TOKS_C_N
        jne     L(na_cut)
        or      qword ptr [rsp + S_NNA], rbx
        jmp     L(na_next)
1:      or      qword ptr [rsp + S_WNA], rbx
        jmp     L(na_next)
#endif

/* a segment shorter than 64 bytes (>= 4: shorter remainders are the twin's): ymm0:ymm1 = text[0, len)
 * zero-padded, from overlapping loads inside it and byte shuffles (k3a2_shuf: 16 x 0x80, 0..15, 32 x 0x80);
 * no store and reload. The masks are shifted down by pos. */
L(short):
        mov     rbp, r13
        lea     rcx, [rip + SYM(k3a2_shuf)]
        cmp     r14, 32
        jb      2f
        mov     r11d, 64
        sub     r11, r14                        /* s = 64 - len: ymm1 = text[len - 32, len) >> s bytes */
        vmovdqu ymm0, ymmword ptr [r15]
        vmovdqu ymm1, ymmword ptr [r15 + r14 - 32]
        vperm2i128 ymm3, ymm1, ymm1, 0x81       /* its high lane, low */
        vbroadcasti128 ymm4, xmmword ptr [rcx + r11 + 16]
        vbroadcasti128 ymm5, xmmword ptr [rcx + r11]
        vpshufb ymm1, ymm1, ymm4
        vpshufb ymm3, ymm3, ymm5
        vpor    ymm1, ymm1, ymm3
        jmp     L(classify)
2:      vpxor   xmm1, xmm1, xmm1
        cmp     r14, 16
        jb      3f
        mov     r11d, 32
        sub     r11, r14                        /* s = 32 - len: the high lane = text[len - 16, len) >> s */
        vmovdqu xmm0, xmmword ptr [r15]
        vmovdqu xmm3, xmmword ptr [r15 + r14 - 16]
        vpshufb xmm3, xmm3, xmmword ptr [rcx + r11 + 16]
        vinserti128 ymm0, ymm0, xmm3, 1
        jmp     L(classify)
3:      mov     r11d, 24
        sub     r11, r14                        /* 16 - (len - 8): text[len - 8, len) << (len - 8) bytes */
        cmp     r14, 8
        jb      4f
        vmovq   xmm0, qword ptr [r15]
        vmovq   xmm3, qword ptr [r15 + r14 - 8]
        vpshufb xmm3, xmm3, xmmword ptr [rcx + r11]
        vpor    xmm0, xmm0, xmm3
        jmp     L(classify)
4:      vmovd   xmm0, dword ptr [r15]           /* 4 <= len < 8: text[len - 4, len) << (len - 4) bytes */
        vmovd   xmm3, dword ptr [r15 + r14 - 4]
        vpshufb xmm3, xmm3, xmmword ptr [rcx + r11 - 4]
        vpor    xmm0, xmm0, xmm3
        jmp     L(classify)

L(stuck):                                       /* no start in (0, u]: a piece longer than its block, the twin's */
        lea     rax, [r13 + 1]
        jmp     L(slow)
L(exit):
        xor     eax, eax
L(slow):                                        /* rax: 0, or the end of a stretch the c twin scans from pos */
        mov     rcx, qword ptr [rsp + S_A]
        mov     qword ptr [rcx + K3_RSV], rax
        mov     rax, r12
        sub     rax, qword ptr [rsp + S_ENDS0]
        shr     rax, 2                          /* n */
        mov     qword ptr [rcx + K3_N], rax
        test    rax, rax
        jz      1f
        mov     edx, dword ptr [r12 - 4]
        mov     qword ptr [rcx + K3_POS], rdx
1:      vzeroupper
        EPILOGUE 6, 0, LOCALS
ENDFUNC K3_FN

        RODATA
        .p2align 5
        C32     k3a2_c20, 0x20
        C32     k3a2_c05, 0x05
        C32     k3a2_c65, 0x65
        C32     k3a2_c46, 0x46
        C32     k3a2_c75, 0x75
        C32     k3a2_c72, 0x72
        C32     k3a2_c7a, 0x7A
        C32     k3a2_c0a, 0x0A
        C32     k3a2_c0d, 0x0D
        C32     k3a2_c27, 0x27
        C32     k3a2_cc0, 0xC0
        C32     k3a2_c3f, 0x3F
        C32     k3a2_ca0, 0xA0
        C32     k3a2_c61, 0x61
        C32     k3a2_c90, 0x90
        C32     k3a2_c6f, 0x6F
        C32     k3a2_c0f, 0x0F
        C32     k3a2_cdc, 0xDC
        C32     k3a2_c29, 0x29
        C32     k3a2_ca9, 0xA9
        C32     k3a2_c54, 0x54
#if K3_DSV3
        C32     k3a2_c01, 0x01
        C32     k3a2_ce1, 0xE1
        C32     k3a2_c2e, 0x2E
#endif
#if K3_O200K
        C32     k3a2_c25, 0x25
        C32     k3a2_c2f, 0x2F
#endif
DATA k3a2_d8
        .fill   8, 4, 8
DATA k3a2_bitidx                                /* [b]: the indices of b's set bits, ascending, 0-padded */
        .set    k3a2_b, 0
        .rept   256
        BITIDX  k3a2_b
        .set    k3a2_b, k3a2_b + 1
        .endr
DATA k3a2_contr                                 /* [c1]: 0x80 for s t m d; e for r v; l for l */
        .fill   0x64, 1, 0
        .byte   0x80                            /* d */
        .fill   0x6C - 0x65, 1, 0
        .byte   0x6C, 0x80                      /* l m */
        .fill   0x72 - 0x6E, 1, 0
        .byte   0x65, 0x80, 0x80, 0, 0x65       /* r s t u v */
        .fill   0x100 - 0x77, 1, 0
#if K3_O200K
DATA k3o2_slot                                  /* [class]: its slot in the atom pass */
        .set    k3o2_c, 0
        .rept   256
        K3O2_SLOT k3o2_c
        .set    k3o2_c, k3o2_c + 1
        .endr
#endif
DATA k3a2_shuf                                  /* vpshufb windows for L(short) */
        .fill   16, 1, 0x80
        .byte   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        .fill   32, 1, 0x80
        TEXT
