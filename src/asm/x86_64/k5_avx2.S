/*
 * src/asm/x86_64/k5_avx2.S: K5, encode pieces, avx2 tier (docs/kernels.md §6, layout.h K5_*).
 *
 *   uint64_t toks_k5_encode_avx2(const toks_tables *t, toks_k5_args *a)
 *
 * Byte-identical to the c twin src/core/k5_c.c: out[0, n_out), n_out, the three counters and the
 * dynamic cache's contents. Per piece [s, e) of length len:
 *   len 1      byte2id[b], a static hit;
 *   len 2..15  the key (bytes zero-padded, byte 15 = len) is built twice from one window: in gprs
 *              (lo, hi) for the crc32c hash and in xmm0 for the compares, masked by a per-len table
 *              row. The window is a 16-byte load at p, or, when the segment ends within 16 bytes of
 *              p, overlapping loads inside [p, p + len) (text[len..] is never read). Then the cache
 *              line h & cache_mask (both ways in one vpcmpeqb; a hit emits the entry's 4 ids and
 *              advances by its count) and the words line h and, only when it misses, the line
 *              rotr32(h, 16) (97-99% of the static hits are in line h), each line's way picked by
 *              cmov. A warm scratch (a's counters sum to >= TOKS_K5_WARM pieces since init) probes
 *              the cache first and a static hit fills the cache; a fresh one probes the words lines
 *              first and fills nothing from them (kernels.md §6); else
 *   (longer)   toks_k5_long_avx2 (K6, or a's lcache in front of it) at the out cursor, a miss; a piece
 *              of <= 15 bytes with <= 4 ids fills the cache (the c twin's bpe_val_pack).
 * Fill: way 0 -> way 1, the new entry -> way 0 (two 32-byte stores).
 * Cache entries carry a's cache_tag in their val (layout.h): the way whose key matches (way 0 first)
 * answers only when its tag word equals the call's, and a fill writes it. Each piece counts once, so this
 * call's hits_cache = n - hits_static - misses; the three counters accumulate into a.
 * Registers: r15 t, r14 text, r13 the ends cursor, r12 the out cursor, rbx s, rbp cache (0: none);
 *   r8 byte2id, r9 cache_mask, r11 the key table, ymm5 ID_MASK, and text + len - 16 in the frame's first
 *   slot (the K6 call's win64 home space): reloaded after the K6 call. Frame: 6 gpr + LOCALS; ymm0-5 only.
 */
#include "asm.h"
#include "layout.h"

/* private labels: L-prefixed on mach-o (where .L names are symbols), .L elsewhere */
#if defined(TOKS_MACHO)
#  define L(x) L##x
#else
#  define L(x) .L##x
#endif

#define HSEED      0x9E3779B9   /* TOKS_HSEED */
#define ID_MASK    0x001FFFFF   /* TOKS_ID_MASK */

#define S_HOME     0            /* 32 bytes: home space for the K6 call (win64) */
#define S_K6       32           /* toks_k6_args, 64 bytes */
#define S_A        96           /* the args pointer */
#define S_WARM     104          /* 1: a warm scratch (TOKS_K5_WARM pieces since init), else 0 */
#define S_ENDE     112          /* ends + n */
#define S_HS       120          /* hits_static */
#define S_MS       128          /* misses */
#define S_BKT      136          /* the cache line, across the K6 call */
#define S_KEY      144          /* the key (16 bytes), across the K6 call */
#define S_TWV      160          /* 16 bytes: 0, then toks_tag_word(cache_tag) (the val's high half) */
#define LOCALS     176

FUNC toks_k5_encode_avx2
        PROLOGUE 6, 0, LOCALS
        mov     r15, ARG0
        mov     rax, ARG1
        mov     qword ptr [rsp + S_A], rax
        mov     r14, qword ptr [rax + K5_TEXT]
        mov     r13, qword ptr [rax + K5_ENDS]
        mov     rcx, qword ptr [rax + K5_N]
        lea     rcx, [r13 + rcx*4]
        mov     qword ptr [rsp + S_ENDE], rcx
        mov     rbx, qword ptr [rax + K5_START]
        mov     r12, qword ptr [rax + K5_OUT]
        mov     rcx, qword ptr [rax + K5_HITS_STATIC]   /* the pieces since init: a warm scratch? (layout.h) */
        add     rcx, qword ptr [rax + K5_HITS_CACHE]
        add     rcx, qword ptr [rax + K5_MISSES]
        xor     edx, edx
        cmp     rcx, TOKS_K5_WARM
        setae   dl
        mov     qword ptr [rsp + S_WARM], rdx
        mov     rbp, qword ptr [rax + K5_CACHE]
        mov     rcx, qword ptr [rax + K5_WORK]
        mov     qword ptr [rsp + S_K6 + K6_WORK], rcx
        mov     rcx, qword ptr [rax + K5_WORK_BYTES]
        mov     qword ptr [rsp + S_K6 + K6_WORK_BYTES], rcx
        xor     ecx, ecx
        mov     qword ptr [rsp + S_HS], rcx
        mov     qword ptr [rsp + S_MS], rcx
        mov     qword ptr [rsp + S_TWV], rcx
        mov     rcx, qword ptr [rax + K5_CACHE_TAG]     /* toks_tag_word(cache_tag) (layout.h) */
        mov     edx, ecx
        and     edx, 0x7FF
        shl     rdx, 21
        shr     rcx, 11
        and     ecx, 0x7FF
        shl     rcx, 53
        or      rcx, rdx
        mov     qword ptr [rsp + S_TWV + 8], rcx
        cmp     r13, qword ptr [rsp + S_ENDE]
        jae     L(done)
L(reload):
        mov     rax, qword ptr [rsp + S_A]
        mov     r8, qword ptr [r15 + TT_BYTE2ID]
        mov     r9, qword ptr [rax + K5_CACHE_MASK]
        mov     r10, qword ptr [rax + K5_LEN]
        lea     r10, [r14 + r10 - 16]
        mov     qword ptr [rsp + S_HOME], r10           /* text + len - 16 */
        lea     r11, [rip + L(keytab)]
        vpbroadcastd ymm5, dword ptr [rip + L(idmask)]

        .p2align 4
L(loop):
        mov     ecx, dword ptr [r13]                    /* e */
        sub     rcx, rbx                                /* len */
        lea     rsi, [r14 + rbx]                        /* p */
        cmp     rcx, 1
        je      L(one)
        cmp     rcx, 15
        ja      L(k6)
        cmp     rsi, qword ptr [rsp + S_HOME]
        ja      L(tail)
        mov     rax, qword ptr [rsi]
        mov     rdx, qword ptr [rsi + 8]
        vmovdqu xmm0, xmmword ptr [rsi]
        lea     rdi, [rcx*4]                            /* row len of the key table: r11 + 32 len */
        and     rax, qword ptr [r11 + rdi*8]
        and     rdx, qword ptr [r11 + rdi*8 + 8]
        or      rdx, qword ptr [r11 + rdi*8 + 24]
        vpand   xmm0, xmm0, xmmword ptr [r11 + rdi*8]
        vpor    xmm0, xmm0, xmmword ptr [r11 + rdi*8 + 16]
L(key):                                                 /* rax lo, rdx hi, xmm0 the key */
        mov     edi, HSEED
        crc32   rdi, rax
        crc32   rdi, rdx                                /* h */
        vinserti128 ymm1, ymm0, xmm0, 1                 /* the key in both lanes */
        mov     eax, edi
        and     rax, r9
        shl     rax, 6
        add     rax, rbp                                /* the cache line (when rbp != 0) */
        cmp     byte ptr [rsp + S_WARM], 0
        je      L(words)                                /* a fresh scratch: the words lines first */
L(cache):
        test    rbp, rbp
        jz      L(cmiss)
        vpcmpeqb ymm2, ymm1, ymmword ptr [rax]
        vpmovmskb edx, ymm2
        not     edx
        movzx   esi, dx                                 /* bytes where way 0 differs */
        shr     edx, 16                                 /* way 1 */
        cmp     esi, 1                                  /* CF: way 0 holds the key */
        sbb     rcx, rcx
        and     rcx, -16
        lea     rcx, [rax + rcx + 48]                   /* that way's value */
        cmp     edx, esi
        cmovb   esi, edx                                /* zero: a way holds the key */
        mov     rdx, qword ptr [rcx + 8]                /* and that way's tag is the call's */
        xor     rdx, qword ptr [rsp + S_TWV + 8]
        and     rdx, qword ptr [rip + L(tagmask)]
        or      rsi, rdx
        jnz     L(cmiss)
        vpand   xmm2, xmm5, xmmword ptr [rcx]
        vmovdqu xmmword ptr [r12], xmm2
        mov     ecx, dword ptr [rcx]
        shr     ecx, 29
        lea     r12, [r12 + rcx*4]
L(next):
        mov     ebx, dword ptr [r13]                    /* s = e */
        add     r13, 4
        cmp     r13, qword ptr [rsp + S_ENDE]
        jb      L(loop)
L(done):
        mov     rcx, qword ptr [rsp + S_A]
        mov     rax, r12
        sub     rax, qword ptr [rcx + K5_OUT]
        shr     rax, 2
        mov     qword ptr [rcx + K5_N_OUT], rax
        mov     rdx, qword ptr [rsp + S_HS]
        mov     rsi, qword ptr [rsp + S_MS]
        add     qword ptr [rcx + K5_HITS_STATIC], rdx   /* the counters accumulate */
        add     qword ptr [rcx + K5_MISSES], rsi
        mov     rdi, qword ptr [rcx + K5_N]
        sub     rdi, rdx
        sub     rdi, rsi
        add     qword ptr [rcx + K5_HITS_CACHE], rdi
        vzeroupper
        EPILOGUE 6, 0, LOCALS

L(one):
        movzx   eax, byte ptr [rsi]
        mov     eax, dword ptr [r8 + rax*4]
        mov     dword ptr [r12], eax
        add     r12, 4
        inc     qword ptr [rsp + S_HS]
        jmp     L(next)

L(tail):                                                /* read only [p, p + len), len 0..15 */
        xor     eax, eax
        xor     edx, edx
        cmp     ecx, 8
        jae     31f
        cmp     ecx, 4
        jb      32f
        mov     eax, dword ptr [rsi]                    /* 4..7: two overlapping dwords */
        mov     edi, dword ptr [rsi + rcx - 4]
        lea     edx, [rcx*8 - 32]
        shlx    rdi, rdi, rdx
        or      rax, rdi
        xor     edx, edx
        jmp     34f
32:
        cmp     ecx, 2
        jb      33f
        movzx   eax, word ptr [rsi]                     /* 2..3: two overlapping words */
        movzx   edi, word ptr [rsi + rcx - 2]
        lea     edx, [rcx*8 - 16]
        shlx    rdi, rdi, rdx
        or      rax, rdi
        xor     edx, edx
        jmp     34f
33:
        test    ecx, ecx
        jz      34f
        movzx   eax, byte ptr [rsi]
        jmp     34f
31:
        mov     rax, qword ptr [rsi]                    /* 8..15: bytes [0, 8), then [8, len) out */
        mov     rdx, qword ptr [rsi + rcx - 8]          /* of the 8 bytes ending at p + len */
        shr     rdx, 8
        mov     edi, 15
        sub     edi, ecx
        shl     edi, 3
        shrx    rdx, rdx, rdi
34:
        mov     rdi, rcx
        shl     rdi, 56
        or      rdx, rdi                                /* byte 15 = len */
        vmovq   xmm0, rax
        vpinsrq xmm0, xmm0, rdx, 1
        jmp     L(key)

L(cmiss):
        cmp     byte ptr [rsp + S_WARM], 0
        je      L(k6)                                   /* fresh: the words lines missed before */
L(words):                                               /* rdi h, ymm1 the key twice, xmm0 the key, rax the cache line */
        mov     rdx, qword ptr [r15 + TT_WORDS]
        test    rdx, rdx
        jz      L(smiss)
        mov     ecx, edi
        and     rcx, qword ptr [r15 + TT_WORDS_MASK]
        shl     rcx, 6
        add     rcx, rdx                                /* words line h */
        vpcmpeqb ymm2, ymm1, ymmword ptr [rcx]
        vpmovmskb r10d, ymm2
        not     r10d
        movzx   esi, r10w                               /* bytes where way 0 differs */
        shr     r10d, 16                                /* way 1 */
        cmp     esi, 1                                  /* CF: way 0 holds the key */
        sbb     rdx, rdx
        and     rdx, -16
        lea     rdx, [rcx + rdx + 48]                   /* that way's value */
        cmp     r10d, esi
        cmovb   esi, r10d
        test    esi, esi
        jz      L(shit)
        rorx    edi, edi, 16                            /* words line rotr32(h, 16) */
        and     rdi, qword ptr [r15 + TT_WORDS_MASK]
        shl     rdi, 6
        add     rdi, qword ptr [r15 + TT_WORDS]
        vpcmpeqb ymm2, ymm1, ymmword ptr [rdi]
        vpmovmskb r10d, ymm2
        not     r10d
        movzx   esi, r10w
        shr     r10d, 16
        cmp     esi, 1
        sbb     rdx, rdx
        and     rdx, -16
        lea     rdx, [rdi + rdx + 48]
        cmp     r10d, esi
        cmovb   esi, r10d
        test    esi, esi
        jz      L(shit)
L(smiss):
        cmp     byte ptr [rsp + S_WARM], 0
        je      L(cache)                                /* fresh: now the cache */
        jmp     L(k6)
L(shit):                                                /* rdx: the static entry's value */
        vmovdqu xmm3, xmmword ptr [rdx]
        vpand   xmm2, xmm5, xmm3                        /* its ids, straight to out */
        vmovdqu xmmword ptr [r12], xmm2
        mov     edx, dword ptr [rdx]
        shr     edx, 29
        lea     r12, [r12 + rdx*4]
        inc     qword ptr [rsp + S_HS]
        test    rbp, rbp
        jz      L(next)
        cmp     byte ptr [rsp + S_WARM], 0
        je      L(next)                                 /* a fresh scratch keeps static answers out */
        vpor    xmm3, xmm3, xmmword ptr [rsp + S_TWV]   /* tagged, it fills the cache: way 0 -> way 1 */
        vinserti128 ymm4, ymm0, xmmword ptr [rax], 1
        vinserti128 ymm3, ymm3, xmmword ptr [rax + 32], 1
        vmovdqu ymmword ptr [rax], ymm4
        vmovdqu ymmword ptr [rax + 32], ymm3
        jmp     L(next)

L(k6):                                                  /* rax the cache line, xmm0 the key (len <= 15) */
        mov     qword ptr [rsp + S_BKT], rax
        vmovdqu xmmword ptr [rsp + S_KEY], xmm0
        mov     ecx, dword ptr [r13]
        sub     rcx, rbx                                /* len */
        lea     rsi, [r14 + rbx]
        mov     qword ptr [rsp + S_K6 + K6_PIECE], rsi
        mov     qword ptr [rsp + S_K6 + K6_LEN], rcx
        mov     qword ptr [rsp + S_K6 + K6_OUT], r12
        mov     ARG0, r15
        lea     ARG1, [rsp + S_K6]
        mov     rax, qword ptr [rsp + S_A]
        mov     ARG2, qword ptr [rax + K5_LCACHE]       /* K6, or the long cache in front of it (k5_long.c) */
        call    SYM(toks_k5_long_avx2)
        inc     qword ptr [rsp + S_MS]
        test    rbp, rbp
        jz      51f
        cmp     rax, 4
        ja      51f
        mov     ecx, dword ptr [r13]
        sub     rcx, rbx
        cmp     rcx, 15
        ja      51f
        mov     rdx, qword ptr [rsp + S_BKT]
        lea     rcx, [rip + L(packtab)]
        mov     rdi, rax
        shl     rdi, 5
        vmovdqu xmm2, xmmword ptr [r12]                 /* K6's ids (n <= 4; the rest masked off) */
        vpand   xmm2, xmm2, xmmword ptr [rcx + rdi]
        vpor    xmm2, xmm2, xmmword ptr [rcx + rdi + 16]
        vpor    xmm2, xmm2, xmmword ptr [rsp + S_TWV]   /* the tag in ids[2] / ids[3] */
        vmovdqu xmm3, xmmword ptr [rsp + S_KEY]
        vinserti128 ymm3, ymm3, xmmword ptr [rdx], 1
        vinserti128 ymm2, ymm2, xmmword ptr [rdx + 32], 1
        vmovdqu ymmword ptr [rdx], ymm3
        vmovdqu ymmword ptr [rdx + 32], ymm2
51:
        lea     r12, [r12 + rax*4]
        mov     ebx, dword ptr [r13]
        add     r13, 4
        cmp     r13, qword ptr [rsp + S_ENDE]
        jb      L(reload)
        jmp     L(done)
ENDFUNC toks_k5_encode_avx2

/* key table row len (32 bytes): lo mask, hi mask, 0, len << 56 */
.macro KEYROW n
  .if \n >= 8
        .quad   -1
        .quad   (1 << (8 * (\n - 8))) - 1
  .else
        .quad   (1 << (8 * \n)) - 1
        .quad   0
  .endif
        .quad   0
        .quad   \n << 56
.endm

/* pack table row n (32 bytes): ID_MASK in the lanes below n, then n << 29 in lane 0 */
.macro PACKLANE n, k
  .if \k < \n
        .long   ID_MASK
  .else
        .long   0
  .endif
.endm
.macro PACKROW n
        PACKLANE \n, 0
        PACKLANE \n, 1
        PACKLANE \n, 2
        PACKLANE \n, 3
        .long   \n << 29, 0, 0, 0
.endm

        RODATA
        .p2align 5
L(keytab):
        KEYROW 0
        KEYROW 1
        KEYROW 2
        KEYROW 3
        KEYROW 4
        KEYROW 5
        KEYROW 6
        KEYROW 7
        KEYROW 8
        KEYROW 9
        KEYROW 10
        KEYROW 11
        KEYROW 12
        KEYROW 13
        KEYROW 14
        KEYROW 15
L(packtab):
        PACKROW 0
        PACKROW 1
        PACKROW 2
        PACKROW 3
        PACKROW 4
L(idmask):
        .long   ID_MASK
        .p2align 3
L(tagmask):
        .quad   0xFFE00000FFE00000                      /* TOKS_TAG_MASK64 */
        TEXT
