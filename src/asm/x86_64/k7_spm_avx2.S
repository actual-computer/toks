/*
 * src/asm/x86_64/k7_spm_avx2.S: K7, the sentencepiece-style bpe scan, tier avx2 (docs/kernels.md §8: contract, the
 * algorithm shared with k7_spm_neon.S, measurements; the c twin is spm_c.c's toks_k7_spm_c).
 *
 *   uint64_t toks_k7_spm_avx2(const toks_spm *s, toks_k7_args *a)     returns the window's word starts
 *
 * Registers: r15 s, r13 p, rbx the word starts; per char r12 the chars left, r11 j, rbp the last char's entry, r10
 * stage1, r9 stage2, r8 pairs, rsi one, rdi the previous char's entry, r14 k7_tab, xmm3 all ones, xmm5 the pair key,
 * xmm0 / xmm1 its compares (vector registers 0-5 only: xmm6-15 are callee-saved on win64 and PROLOGUE saves none
 * here; ymm0 / ymm1 / ymm2 / ymm4 are dead once the window is classified). Per char: both
 * entries PLAIN (spm.h: an id, not the split image) -> the pair set alone decides, else the general rule. Locals:
 * [rsp] a, [rsp + 8] TOKS_FIB64, [rsp + 16] the pair set's mask, [rsp + 24] the split id (or ~0), [rsp + 32] the
 * window's end, [rsp + 40] its leads. Frame: PROLOGUE 6 (win64: + rdi rsi), 48 bytes of locals.
 */
#include "asm.h"
#include "layout.h"

/* pext masks of a bswapped 4-byte load for 2, 3, 4 byte chars, each length's smallest scalar, the class bytes */
        RODATA
        .p2align 4
DATA k7_tab
        .long   0x1F3F0000, 0x0F3F3F00, 0x073F3F3F
        .long   0x80, 0x800, 0x10000
        .byte   0xc0, 0xe0, 0xf0
        TEXT

/* position j (straight-line, j from 63 down): rbx = rbx * 2 + the cut_ab8 (r11) byte of p[j - 1] p[j] */
.macro PAIR j
        movzx   eax, word ptr [r13 + \j - 1]
        movzx   eax, byte ptr [r11 + rax]
        lea     rbx, [rax + rbx*2]
.endm

/* rax = the 64-bit movemask of ymm\a (bytes 0..31) and ymm\b (32..63); clobbers rcx */
.macro MM a, b
        vpmovmskb eax, ymm\a
        vpmovmskb ecx, ymm\b
        shl     rcx, 32
        or      rax, rcx
.endm

FUNC toks_k7_spm_avx2
        PROLOGUE 6, 0, 48
        mov     r15, ARG0
        mov     r14, ARG1
        mov     qword ptr [rsp], r14
        mov     rcx, qword ptr [r14 + K7_LEN]
        mov     rdx, qword ptr [r14 + K7_POS]
        test    byte ptr [r14 + K7_FLAGS], 1
        jz      9f
        test    rdx, rdx
        jz      9f
        sub     rcx, rdx
        cmp     rcx, 64
        jb      9f
        mov     r13, qword ptr [r14 + K7_TEXT]
        add     r13, rdx
        vmovdqu ymm0, ymmword ptr [r13]
        vmovdqu ymm1, ymmword ptr [r13 + 32]
        MM      0, 1
        mov     r8, rax                                 /* non-ascii */
        mov     r12d, 64                                /* the window's end */
        mov     rdx, -1                                 /* its positions */
        xor     r9d, r9d                                /* continuation bytes */
        xor     r10d, r10d                              /* leads */
        test    rax, rax
        jz      1f
        vpbroadcastb ymm2, byte ptr [rip + SYM(k7_tab) + 24]
        vpcmpgtb ymm3, ymm2, ymm0
        vpcmpgtb ymm4, ymm2, ymm1
        MM      3, 4
        mov     r9, rax
        vpbroadcastb ymm2, byte ptr [rip + SYM(k7_tab) + 25]
        vpmaxub ymm3, ymm0, ymm2
        vpcmpeqb ymm3, ymm3, ymm0
        vpmaxub ymm4, ymm1, ymm2
        vpcmpeqb ymm4, ymm4, ymm1
        MM      3, 4
        mov     rsi, rax                                /* 3-4 byte leads */
        vpbroadcastb ymm2, byte ptr [rip + SYM(k7_tab) + 26]
        vpmaxub ymm3, ymm0, ymm2
        vpcmpeqb ymm3, ymm3, ymm0
        vpmaxub ymm4, ymm1, ymm2
        vpcmpeqb ymm4, ymm4, ymm1
        MM      3, 4
        mov     rdi, rax                                /* 4 byte leads (f5..ff too: rejected by the decode) */
        andn    r10, r9, r8
        lea     rcx, [r10 + r10]
        lea     rax, [rsi*4]
        or      rcx, rax
        lea     rax, [rdi*8]
        or      rcx, rax
        cmp     rcx, r9
        jne     9f                                      /* not the leads' continuations: the twin */
        mov     rcx, r10                                /* chars running past the block */
        shr     rcx, 63
        shl     rcx, 63
        shr     rsi, 62
        and     esi, 1
        shl     rsi, 62
        or      rcx, rsi
        shr     rdi, 61
        and     edi, 1
        shl     rdi, 61
        or      rcx, rdi
        tzcnt   r12, rcx
        bzhi    rdx, rdx, r12
1:      mov     qword ptr [rsp + 32], r12
        and     r10, rdx
        mov     qword ptr [rsp + 40], r10
        movzx   ecx, byte ptr [r13 - 1]
        shr     ecx, 7
        xor     ecx, 1                                  /* the byte before p is ascii */
        not     r8
        lea     rdi, [r8 + r8]
        or      rdi, rcx
        and     rdi, r8
        and     rdi, rdx                                /* ascii after ascii */
        andn    r12, r9, rdx
        andn    r12, rdi, r12                           /* every other char */
        xor     ebx, ebx
        mov     r11, qword ptr [r15 + SPM_CUT_AB8]
        popcnt  rax, rdi
        cmp     eax, 24
        jb      3f
        .irp j, 63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32
        PAIR    \j
        .endr
        .irp j, 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
        PAIR    \j
        .endr
        and     rbx, rdi
        jmp     4f
3:      test    rdi, rdi
        jz      4f
        tzcnt   rcx, rdi
        blsr    rdi, rdi
        movzx   eax, word ptr [r13 + rcx - 1]
        movzx   edx, byte ptr [r11 + rax]
        shlx    rdx, rdx, rcx
        or      rbx, rdx
        jmp     3b
4:      mov     rsi, qword ptr [r14 + K7_ONE]
        mov     r10, qword ptr [r15 + SPM_STAGE1]
        mov     r9, qword ptr [r15 + SPM_STAGE1 + 8]
        mov     r8, qword ptr [r15 + SPM_PAIRS]
        mov     rax, qword ptr [r15 + SPM_PAIRS + 8]
        mov     qword ptr [rsp + 16], rax
        mov     rax, 0x9E3779B97F4A7C15                 /* TOKS_FIB64 */
        mov     qword ptr [rsp + 8], rax
        mov     eax, dword ptr [r15 + SPM_ID_REPL]
        mov     edx, -1
        cmp     dword ptr [r15 + SPM_MS_SPLIT], 0
        cmove   eax, edx
        mov     dword ptr [rsp + 24], eax               /* the split id, or ~0: no split */
        vpcmpeqd xmm3, xmm3, xmm3
        mov     rbp, qword ptr [r14 + K7_PREV]
        lea     r14, [rip + SYM(k7_tab)]
5:      test    r12, r12
        jz      8f
        tzcnt   r11, r12                                /* j */
        blsr    r12, r12
        movzx   ecx, byte ptr [r13 + r11 - 1]
        mov     edi, dword ptr [r15 + rcx*4]            /* ascii[] (a non-ascii byte reads inside *s: dropped) */
        cmp     ecx, 0x80
        cmovae  edi, ebp                                /* x: the previous char's entry */
        cmp     r11, 60
        ja      65f
        mov     edx, dword ptr [r13 + r11]              /* the char's bytes */
60:     test    dl, dl
        jns     61f
        mov     ecx, edx
        not     ecx
        shl     ecx, 24
        lzcnt   ecx, ecx                                /* its length: the lead's leading ones */
        cmp     ecx, 4
        ja      9f
        bswap   edx
        pext    edx, edx, dword ptr [r14 + rcx*4 - 8]   /* the scalar (r14: k7_tab) */
        cmp     edx, dword ptr [r14 + rcx*4 + 4]
        jb      9f                                      /* overlong */
        cmp     edx, 0x10ffff
        ja      9f                                      /* past U+10FFFF */
        mov     eax, edx
        shr     eax, 11
        cmp     eax, 0x1b
        je      9f                                      /* a surrogate */
        mov     eax, edx
        shr     eax, 8
        movzx   eax, word ptr [r10 + rax*2]
        shl     eax, 8
        movzx   edx, dl
        or      eax, edx
        mov     edx, dword ptr [r9 + rax*4]             /* y: the char's entry */
        mov     eax, edx
        and     eax, 0x1fffff
        shl     ecx, 24
        or      eax, ecx
        mov     dword ptr [rsi + r11*4], eax            /* one[j] = k << 24 | id */
7:      mov     ebp, edx
        mov     eax, edi
        and     eax, edx
        jns     70f                                     /* not both PLAIN: the general rule */
71:     mov     eax, edi
        and     eax, 0x1fffff
        shl     rax, 21
        mov     ecx, ebp
        and     ecx, 0x1fffff
        or      rax, rcx                                /* the pair set's key id_x << 21 | id_y */
        mov     rcx, rax
        imul    rcx, qword ptr [rsp + 8]
        shr     rcx, 32
        and     rcx, qword ptr [rsp + 16]               /* its home slot i */
        vmovq   xmm5, rax
        vpbroadcastq xmm5, xmm5
        vpcmpeqq xmm0, xmm5, xmmword ptr [r8 + rcx*8]   /* slots i, i + 1 (the mirror slot: no wrap) */
        vpcmpeqq xmm1, xmm3, xmmword ptr [r8 + rcx*8]
        vpor    xmm1, xmm1, xmm0
        vmovmskpd edx, xmm0                             /* spanned: the key in either slot (it sits before any empty) */
        vmovmskpd edi, xmm1                             /* decided: the key or an empty slot */
        test    edi, edi
        jz      74f
        xor     eax, eax
        test    edx, edx
        sete    al                                      /* not spanned: a certified cut */
76:     shlx    rax, rax, r11
        or      rbx, rax
        jmp     5b
61:     movzx   edx, dl                                 /* an ascii char */
        mov     edx, dword ptr [r15 + rdx*4]
        jmp     7b
65:     mov     edx, dword ptr [r13 + 60]               /* a char at p + 61 or 62: its bytes end the block */
        lea     ecx, [r11*8 - 480]
        shrx    edx, edx, ecx
        jmp     60b
70:     mov     eax, edi                                /* the general rule (§5.5) */
        and     eax, 0x1fffff                           /* id_x */
        mov     ecx, ebp
        and     ecx, 0x1fffff                           /* id_y */
        cmp     ecx, dword ptr [rsp + 24]
        je      79f                                     /* the Metaspace split: a cut */
        cmp     eax, 0x1fffff
        je      78f                                     /* an id-less char: no certified cut */
        cmp     ecx, 0x1fffff
        je      78f
        mov     ecx, edi
        shr     ecx, 21
        cmp     cl, 0xff
        je      71b                                     /* not both small: the pair set decides */
        mov     edx, ebp
        shr     edx, 21
        cmp     dl, 0xff
        je      71b
        movzx   ecx, cl                                 /* both small: bit xa * 129 + xb of the bitmap */
        movzx   edx, dl
        imul    ecx, ecx, 129
        add     ecx, edx
        mov     edx, ecx
        shr     edx, 5
        mov     edx, dword ptr [r15 + rdx*4 + SPM_CUT]
        xor     eax, eax
        bt      edx, ecx
        setc    al
        jmp     76b
78:     xor     eax, eax
        jmp     76b
79:     mov     eax, 1
        jmp     76b
74:     add     rcx, 2                                  /* neither slot decides: probe on from slot i + 2 */
73:     and     rcx, qword ptr [rsp + 16]
        mov     rdx, qword ptr [r8 + rcx*8]
        inc     rcx
        cmp     rdx, rax
        je      72f
        cmp     rdx, -1
        jne     73b
        mov     eax, 1                                  /* an empty slot: not spanned, a cut */
        jmp     76b
72:     xor     eax, eax                                /* spanned */
        jmp     76b
8:      mov     r14, qword ptr [rsp]
        mov     rcx, qword ptr [rsp + 32]
        movzx   edx, byte ptr [r13 + rcx - 1]
        mov     eax, dword ptr [r15 + rdx*4]
        cmp     edx, 0x80
        cmovae  eax, ebp                                /* the last char's entry */
        mov     qword ptr [r14 + K7_PREV], rax
        add     qword ptr [r14 + K7_POS], rcx
        mov     rax, qword ptr [rsp + 40]
        mov     qword ptr [r14 + K7_MB], rax
        mov     rax, rbx
        vzeroupper
        EPILOGUE 6, 0, 48
9:      xor     eax, eax                                /* not this window: pos unchanged */
        vzeroupper
        EPILOGUE 6, 0, 48
ENDFUNC toks_k7_spm_avx2
