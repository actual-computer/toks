/*
 * src/asm/x86_64/k1_avx2.S: K1, added-token find, tier avx2 (docs/kernels.md §4, layout.h K1_*; the c twin is
 * src/core/k1_c.c).
 *
 *   uint64_t toks_k1_added_find_avx2(const toks_tables *t, toks_k1_args *a)     returns n (matches in a->m)
 *
 * Outputs, return value and every read equal the c twin's: up to cap matches, each the leftmost position >= the
 * previous match's end (the first: >= pos) whose first byte passes the phase's shufti test (lo[b & 15] &
 * hi[b >> 4] != 0) and where some token of the phase occurs entirely inside [0, len) -- single[b0] first, then
 * index2[b0 | b1 << 8]'s candidates, longest first, the first exact match wins (`el <= best_len` ends the list, a
 * token running past len is skipped). After a match the block's candidates below its end are dropped and the
 * scan goes on in the same block (or from the end, when it lies past the block).
 *
 * Scan: 64-byte blocks from text + pos: per 32 bytes vpsrlw, two vpand, two vpshufb and a vpand give a
 * nonzero byte per candidate; one vptest per block answers "any candidate" (the common case: none, next
 * block). A block with one gets its 64-bit mask (vpcmpeqb against zero, two vpmovmskb, not) and its
 * candidates in ascending order (tzcnt, blsr). The 1..63 bytes after the last full block are the 64-byte block
 * that ends at len with the scanned positions masked off. A text of 16..63 bytes is four 16-byte loads at 0,
 * min(16, len - 16), min(32, len - 16) and len - 16 merged into one mask; one under 16 bytes builds its mask
 * byte by byte. Every load lies inside [text, text + len): no padding, no copy.
 * Verify (per candidate, kernels.md §4): with >= 4 bytes left, the h4 bucket of the first four bytes (tokens of
 * >= 4 bytes, longest first: the first exact match wins); else, or with none, index2's bucket from its end (the
 * 2- and 3-byte tokens, shortest first, the last match wins), then single. A compare reads exactly [q, q + el) and
 * the token's [off, off + el): 16-byte chunks from the end down (the last one at 0), or two overlapping 8 / 4 /
 * 2-byte loads.
 * Lineage: shufti (Hyperscan's nibble-table byte classification, as in simdjson), movemask candidate walks
 * (tzcnt / blsr), overlapping head / tail loads instead of a padded copy (the K5 key loads; memcmp
 * implementations), daachorse's leftmost-longest semantics (hf).
 *
 * Registers: r14 t, r13 a (n lives in a->n), r12 last = text + len - 64 (end = r12 + 64), rbp block base, rbx
 * candidate mask; verify: r11 q, r15 end - q, r10 best_len, r9 candidate cursor, r8 candidates left, rdi e, rsi el,
 * rdx the token's bytes, rcx rax temps. ymm0 / ymm1 lo / hi tables, ymm2 0x0f, ymm3-5 blocks. Frame: PROLOGUE 6
 * (win64: + rdi rsi).
 */
#include "asm.h"
#include "layout.h"

/* private labels: L-prefixed on mach-o (where .L names are symbols), .L elsewhere */
#if defined(TOKS_MACHO)
#  define L(x) L##x
#else
#  define L(x) .L##x
#endif

/* ymm<r> = nonzero where its byte passes shufti (lo ymm0, hi ymm1, 0x0f ymm2); clobbers ymm5 */
.macro SHUFTI r
        vpsrlw  ymm5, ymm\r, 4
        vpand   ymm\r, ymm\r, ymm2
        vpand   ymm5, ymm5, ymm2
        vpshufb ymm\r, ymm0, ymm\r
        vpshufb ymm5, ymm1, ymm5
        vpand   ymm\r, ymm\r, ymm5
.endm

/* rbx = the 64-bit mask of ymm3 (bits 0-31) and ymm4 (bits 32-63): bit k where byte k is nonzero; rax */
.macro MASK64
        vpxor   xmm5, xmm5, xmm5
        vpcmpeqb ymm3, ymm3, ymm5
        vpcmpeqb ymm4, ymm4, ymm5
        vpmovmskb ebx, ymm3
        vpmovmskb eax, ymm4
        shl     rax, 32
        or      rbx, rax
        not     rbx
.endm

/* RXCMP: the rsi bytes at rcx (a radix label) equal those at rax (the text), else jmp L(rdone); reads only those
 * bytes (16-byte chunks from the end down, overlapping 8 / 4 / 2-byte pairs, one byte). Clobbers rsi, r9, xmm3 */
.macro RXCMP
        test    rsi, rsi
        jz      49f
        cmp     rsi, 16
        jb      45f
        sub     rsi, 16
44:     vmovdqu xmm3, xmmword ptr [rcx + rsi]
        vpxor   xmm3, xmm3, xmmword ptr [rax + rsi]
        vptest  xmm3, xmm3
        jnz     L(rdone)
        test    rsi, rsi
        jz      49f
        sub     rsi, 16
        jae     44b
        xor     esi, esi
        jmp     44b
45:     cmp     rsi, 8
        jb      46f
        mov     r9, qword ptr [rcx]
        cmp     r9, qword ptr [rax]
        jne     L(rdone)
        mov     r9, qword ptr [rcx + rsi - 8]
        cmp     r9, qword ptr [rax + rsi - 8]
        jne     L(rdone)
        jmp     49f
46:     cmp     rsi, 4
        jb      47f
        mov     r9d, dword ptr [rcx]
        cmp     r9d, dword ptr [rax]
        jne     L(rdone)
        mov     r9d, dword ptr [rcx + rsi - 4]
        cmp     r9d, dword ptr [rax + rsi - 4]
        jne     L(rdone)
        jmp     49f
47:     cmp     rsi, 2
        jb      48f
        movzx   r9d, word ptr [rcx]
        cmp     r9w, word ptr [rax]
        jne     L(rdone)
        movzx   r9d, word ptr [rcx + rsi - 2]
        cmp     r9w, word ptr [rax + rsi - 2]
        jne     L(rdone)
        jmp     49f
48:     movzx   r9d, byte ptr [rcx]
        cmp     r9b, byte ptr [rax]
        jne     L(rdone)
49:
.endm

FUNC toks_k1_added_find_avx2
        PROLOGUE 6, 0, 0
        mov     r14, ARG0
        mov     r13, ARG1
        mov     rax, qword ptr [r13 + K1_LEN]
        mov     qword ptr [r13 + K1_N], 0
        mov     qword ptr [r13 + K1_NEXT], rax          /* n = 0, next = len */
        mov     rcx, qword ptr [r13 + K1_PHASE]
        cmp     rcx, 1
        ja      L(none)
        mov     rdx, qword ptr [r14 + TT_ADD_PHASES]
        bt      rdx, rcx
        jnc     L(none)                                 /* no tokens in this phase: the tables may be NULL */
        mov     rbp, qword ptr [r13 + K1_POS]
        cmp     rbp, rax
        jae     L(none)
        shl     ecx, 5
        mov     rdx, qword ptr [r14 + TT_ADD_SHUFTI]
        add     rdx, rcx                                /* + 32 phase */
        vbroadcasti128 ymm0, xmmword ptr [rdx]
        vbroadcasti128 ymm1, xmmword ptr [rdx + 16]
        mov     ecx, 0x0f0f0f0f
        vmovd   xmm2, ecx
        vpbroadcastd ymm2, xmm2
        mov     rsi, qword ptr [r13 + K1_TEXT]
        lea     r12, [rsi + rax - 64]                   /* last */
        add     rbp, rsi                                /* text + pos */
        cmp     rax, 64
        jb      L(short)
        cmp     rbp, r12
        ja      L(tail)

        .p2align 4
L(loop):
        vmovdqu ymm3, ymmword ptr [rbp]
        vmovdqu ymm4, ymmword ptr [rbp + 32]
        SHUFTI  3
        SHUFTI  4
        vpor    ymm5, ymm3, ymm4
        vptest  ymm5, ymm5
        jnz     L(block)
L(next):
        add     rbp, 64
        cmp     rbp, r12
        jbe     L(loop)
L(tail):                                                /* 0..63 bytes left after rbp */
        lea     rcx, [r12 + 64]
        sub     rcx, rbp                                /* r */
        jbe     L(none)
        mov     rbp, r12                                /* the block ending at len: its top r bytes are new */
        vmovdqu ymm3, ymmword ptr [rbp]
        vmovdqu ymm4, ymmword ptr [rbp + 32]
        SHUFTI  3
        SHUFTI  4
        MASK64
        neg     rcx
        add     rcx, 64
        mov     rax, -1
        shlx    rax, rax, rcx
        and     rbx, rax
        jmp     L(cand)

L(block):
        MASK64
L(cand):                                                /* rbx: candidates at rbp + bit */
        test    rbx, rbx
        jz      L(next)
L(cloop):
        tzcnt   r11, rbx
        blsr    rbx, rbx
        add     r11, rbp                                /* q */
        lea     r15, [r12 + 64]
        sub     r15, r11                                /* end - q */
        cmp     r15, 4
        jb      L(s23)                                  /* under four bytes left: no token of four or more fits */
        mov     eax, dword ptr [r11]
        imul    eax, eax, -1640531535                   /* toks_k1_h4: * 0x9E3779B1 */
        shr     eax, 32 - TOKS_K1_H4_BITS
        shl     rax, 4                                  /* slot << 1 | phase */
        add     rax, qword ptr [r14 + TT_ADD_INDEX]
        mov     rdx, qword ptr [r13 + K1_PHASE]
        mov     rcx, qword ptr [rax + rdx*8 + 0x100000] /* the h4 bucket (index2 + 2 * 65536 * 8): start | count << 32 */
        test    rcx, rcx
        js      L(radix)                                /* bit 63: a long bucket, its radix tree */
        mov     r8, rcx
        shr     r8, 32
        jz      L(s23)
        mov     ecx, ecx
        mov     r9, qword ptr [r14 + TT_ADD_CAND]
        lea     r9, [r9 + rcx*4]
L(vloop):
        mov     edi, dword ptr [r9]                     /* e */
        add     r9, 4
        mov     rax, rdi
        shl     rax, 4
        add     rax, qword ptr [r14 + TT_ADD_ENTRIES]
        movzx   esi, word ptr [rax + TAE_LEN]           /* el */
        cmp     r15, rsi
        jb      L(vnext)                                /* runs past len */
        mov     edx, dword ptr [rax + TAE_OFF]
        add     rdx, qword ptr [r14 + TT_ADD_BYTES]
        cmp     rsi, 16
        jb      L(lt16)
        lea     rcx, [rsi - 16]                         /* 16-byte chunks from the end down, the last at 0 */
L(c16):
        vmovdqu xmm3, xmmword ptr [rdx + rcx]
        vpcmpeqb xmm3, xmm3, xmmword ptr [r11 + rcx]
        vpmovmskb eax, xmm3
        cmp     eax, 0xffff
        jne     L(vnext)
        test    rcx, rcx
        jz      L(hit)
        xor     eax, eax
        sub     rcx, 16
        cmovs   rcx, rax
        jmp     L(c16)
L(lt16):
        cmp     rsi, 8
        jb      L(lt8)
        mov     rax, qword ptr [rdx]
        cmp     rax, qword ptr [r11]
        jne     L(vnext)
        mov     rax, qword ptr [rdx + rsi - 8]
        cmp     rax, qword ptr [r11 + rsi - 8]
        je      L(hit)
        jmp     L(vnext)
L(lt8):                                                 /* 4..7 bytes: two overlapping dwords */
        mov     eax, dword ptr [rdx]
        cmp     eax, dword ptr [r11]
        jne     L(vnext)
        mov     eax, dword ptr [rdx + rsi - 4]
        cmp     eax, dword ptr [r11 + rsi - 4]
        je      L(hit)
L(vnext):
        dec     r8
        jnz     L(vloop)
L(s23):                                                 /* 2 or 3 bytes: index2's bucket from its end, shortest first */
        xor     r10d, r10d                              /* best_len */
        cmp     r15, 2
        jb      L(s1)                                   /* q + 1 == end: no second byte */
        movzx   eax, word ptr [r11]                     /* b0 | b1 << 8 */
        mov     rdx, qword ptr [r13 + K1_PHASE]
        shl     rdx, 19                                 /* 65536 * 8 phase */
        add     rdx, qword ptr [r14 + TT_ADD_INDEX]
        mov     rcx, qword ptr [rdx + rax*8]            /* start | count << 32 */
        mov     r8, rcx
        shr     r8, 32
        jz      L(s1)
        mov     ecx, ecx
        add     rcx, r8                                 /* the bucket's end */
        mov     r9, qword ptr [r14 + TT_ADD_CAND]
        lea     r9, [r9 + rcx*4]
L(sloop):
        sub     r9, 4
        mov     eax, dword ptr [r9]                     /* e */
        mov     rdx, rax
        shl     rdx, 4
        add     rdx, qword ptr [r14 + TT_ADD_ENTRIES]
        movzx   esi, word ptr [rdx + TAE_LEN]
        cmp     rsi, 4
        jae     L(s1)                                   /* the rest are longer: h4's */
        cmp     r15, rsi
        jb      L(snext)                                /* runs past len */
        mov     edx, dword ptr [rdx + TAE_OFF]
        add     rdx, qword ptr [r14 + TT_ADD_BYTES]
        movzx   ecx, word ptr [rdx]
        cmp     cx, word ptr [r11]
        jne     L(snext)
        movzx   ecx, word ptr [rdx + rsi - 2]
        cmp     cx, word ptr [r11 + rsi - 2]
        jne     L(snext)
        mov     r10, rsi                                /* a match: the longest so far */
        mov     edi, eax
L(snext):
        dec     r8
        jnz     L(sloop)
L(s1):
        test    r10, r10
        jnz     L(shit)
        movzx   eax, byte ptr [r11]
        mov     rdx, qword ptr [r13 + K1_PHASE]
        shl     rdx, 10                                 /* 256 * 4 phase */
        add     rdx, qword ptr [r14 + TT_ADD_SINGLE]
        mov     edi, dword ptr [rdx + rax*4]
        test    edi, edi
        jz      L(fp)                                   /* no token here: a shufti false positive */
        dec     edi                                     /* the one-byte token */
        mov     esi, 1
        jmp     L(hit)
L(shit):
        mov     rsi, r10
        jmp     L(hit)
L(fp):
        test    rbx, rbx
        jnz     L(cloop)
        jmp     L(next)
L(hit):                                                 /* q r11, el rsi, entry rdi */
        mov     rax, r11
        sub     rax, qword ptr [r13 + K1_TEXT]          /* start */
        mov     rcx, qword ptr [r13 + K1_N]
        mov     rdx, rcx
        shl     rdx, 4
        add     rdx, qword ptr [r13 + K1_M]             /* m[n] = { start, end, entry, 0 } */
        mov     dword ptr [rdx], eax
        add     eax, esi                                /* end */
        mov     dword ptr [rdx + 4], eax
        mov     dword ptr [rdx + 8], edi
        mov     dword ptr [rdx + 12], 0
        inc     rcx
        mov     qword ptr [r13 + K1_N], rcx
        cmp     rcx, qword ptr [r13 + K1_CAP]
        jae     L(full)
        add     r11, rsi                                /* text + end: the next search starts there */
        mov     rcx, r11
        sub     rcx, rbp                                /* its bit in the block (>= 1) */
        cmp     rcx, 64
        jae     L(resume)
        mov     rax, -1
        shlx    rax, rax, rcx
        and     rbx, rax                                /* drop the candidates inside the match */
        jmp     L(cand)
L(resume):                                              /* the match ends past this block (never past len) */
        mov     rbp, r11
        cmp     rbp, r12
        jbe     L(loop)
        jmp     L(tail)
L(full):
        mov     qword ptr [r13 + K1_NEXT], rax
        mov     rax, rcx
        vzeroupper
        EPILOGUE 6, 0, 0

L(none):
        mov     rax, qword ptr [r13 + K1_N]
        vzeroupper
        EPILOGUE 6, 0, 0

L(short):                                               /* len < 64: rax len, rsi text, rdx the tables */
        cmp     rax, 16
        jb      L(tiny)
        lea     r8, [rax - 16]                          /* the last chunk at len - 16 */
        mov     r9d, 16
        cmp     r8, 16
        cmovb   r9, r8                                  /* min(16, len - 16) */
        mov     r10d, 32
        cmp     r8, 32
        cmovb   r10, r8                                 /* min(32, len - 16) */
        vmovdqu xmm3, xmmword ptr [rsi]
        vinserti128 ymm3, ymm3, xmmword ptr [rsi + r9], 1
        vmovdqu xmm4, xmmword ptr [rsi + r10]
        vinserti128 ymm4, ymm4, xmmword ptr [rsi + r8], 1
        SHUFTI  3
        SHUFTI  4
        MASK64                                          /* 16 bits per chunk */
        mov     rax, rbx
        shr     rax, 16
        movzx   eax, ax
        shlx    rax, rax, r9
        mov     rcx, rbx
        shr     rcx, 32
        movzx   ecx, cx
        shlx    rcx, rcx, r10
        or      rax, rcx
        mov     rcx, rbx
        shr     rcx, 48
        shlx    rcx, rcx, r8
        or      rax, rcx
        movzx   ebx, bx
        or      rbx, rax
        jmp     L(from)
L(tiny):                                                /* len < 16: the mask byte by byte */
        xor     ebx, ebx
        mov     rcx, rbp
        sub     rcx, rsi                                /* i = pos */
L(tloop):
        movzx   r8d, byte ptr [rsi + rcx]
        mov     r9d, r8d
        and     r9d, 15
        shr     r8d, 4
        movzx   r9d, byte ptr [rdx + r9]
        test    r9b, byte ptr [rdx + r8 + 16]
        jz      L(tskip)
        bts     rbx, rcx
L(tskip):
        inc     rcx
        cmp     rcx, rax
        jb      L(tloop)
L(from):                                                /* rbx: candidates at text + bit; drop those before pos */
        mov     rcx, rbp
        sub     rcx, rsi
        mov     rax, -1
        shlx    rax, rax, rcx
        and     rbx, rax
        mov     rbp, rsi
        jmp     L(cand)
L(radix):                                               /* kernels.md §4: down the tree from the root at add_cand + ecx */
        xor     r10d, r10d                              /* best_len */
        xor     edx, edx                                /* d: the bytes matched */
        mov     r8d, ecx
L(rnode):                                               /* r8: the node's offset */
        add     r8, qword ptr [r14 + TT_ADD_CAND]
        movzx   esi, word ptr [r8 + RX_LLEN]
        mov     rax, r15
        sub     rax, rdx
        cmp     rsi, rax
        ja      L(rdone)                                /* its label runs past len */
        mov     ecx, dword ptr [r8 + RX_LAB]
        add     rcx, qword ptr [r14 + TT_ADD_CAND]      /* the label, then the children's first bytes */
        lea     rax, [r11 + rdx]                        /* text + q + d */
        RXCMP
        movzx   esi, word ptr [r8 + RX_LLEN]
        add     rdx, rsi
        add     rcx, rsi
        mov     eax, dword ptr [r8 + RX_ENT]
        test    eax, eax
        jz      51f
        lea     edi, [rax - 1]                          /* a token ends here: the longest so far */
        mov     r10, rdx
51:     movzx   esi, word ptr [r8 + RX_NCH]
        test    esi, esi
        jz      L(rdone)                                /* no child */
        cmp     rdx, r15
        jae     L(rdone)                                /* the text ends here */
        vpbroadcastb xmm4, byte ptr [r11 + rdx]         /* the next byte, against the children's first bytes */
        xor     r9d, r9d                                /* the chunk's first child */
52:     vpcmpeqb xmm3, xmm4, xmmword ptr [rcx]
        vpmovmskb eax, xmm3
        cmp     rsi, 16
        jae     53f
        bzhi    eax, eax, esi                           /* only the children's bytes */
53:     test    eax, eax
        jnz     54f
        add     rcx, 16
        add     r9, 16
        sub     rsi, 16
        ja      52b
        jmp     L(rdone)
54:     tzcnt   eax, eax
        add     eax, r9d                                /* the child */
        shl     eax, 4
        mov     r8d, dword ptr [r8 + RX_CH]
        add     r8, rax
        jmp     L(rnode)
L(rdone):
        test    r10, r10
        jnz     L(shit)
        jmp     L(s23)
ENDFUNC toks_k1_added_find_avx2
