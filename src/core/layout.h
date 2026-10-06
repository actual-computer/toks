/* layout.h: the internal contract between the c core (docs/notes/c-core.md §layout.h.1) */
#ifndef TOKS_LAYOUT_H
#define TOKS_LAYOUT_H

/* ---- constants shared by c and asm ----------------------------------------------------------------- */

#define TOKS_TABLES_MAGIC      0x544B4F54u  /* 'TOKT' little-endian */
#define TOKS_TABLES_VERSION    2u

#define TOKS_CHUNK_PIECES      256          /* pieces per scan/encode round in the driver */
#define TOKS_ID_MASK           0x001FFFFFu  /* ids are < 2^21 - 1 */

/* ---- class bytes (cls_ascii / cls_stage2), built by src/core/classes.c ---------------------------------- */
#define TOKS_C_BASE_MASK       0x07u
#define TOKS_C_P               0u    /* not L, not N, not \s (punctuation, symbols, invalid utf-8 bytes, ...); in dsv3
                                        tables only \p{P} | \p{S} and invalid bytes (the rest is TOKS_C_X there) */
#define TOKS_C_L               1u    /* \p{L}; also \p{M} when the template folds marks into letters (qwen 3.5, dsv3) */
#define TOKS_C_N               2u    /* \p{N} */
#define TOKS_C_WS              3u    /* \s except CR and LF */
#define TOKS_C_NL              4u    /* CR or LF */
#define TOKS_C_X               5u    /* dsv3 tables only (classes.h TOKS_CLASSES_DSV3): none of \p{L} \p{M} \p{N} \p{P} \p{S}
                                        \s -- unassigned, private use, format, the controls outside \s; every other
                                        table keeps these code points P */
#define TOKS_C_UPPER           0x08u /* Lu Lt Lm Lo M: may be in o200k's upper-case group */
#define TOKS_C_LOWER           0x10u /* Ll Lm Lo M:   may be in o200k's lower-case group */
#define TOKS_C_MARK            0x20u /* \p{M}, whatever the base class */
#define TOKS_C_FOLD_S          0x40u /* non-ascii char that (?i) folds to 's' (U+017F) */
#define TOKS_C_HAN             0x80u /* \p{Han} as tiktoken resolves it (Script=Han, not Script_Extensions: U+3001 is
                                        not Han). The last free class bit, reserved to TOKS_TP_HAN: set only in tables
                                        built for it (classes.c TOKS_CLASSES_HAN), 0 in every other table; load refuses
                                        tables where the bit and the parameter disagree (compile.h toks_tmpl_invalid) */
#define TOKS_C_CJK             0x80u /* dsv3 tables only (TOKS_TMPL_DSV3): U+3040..U+30FF and U+4E00..U+9FA5, deepseek
                                        v3's second split. A dsv3 class byte is base | TOKS_C_CJK and nothing else
                                        (docs/templates/dsv3.md §2): bit 7 means Han in o200k tables, CJK in dsv3 ones */

/* ---- scanner templates and their parameters (toks_tables.tmpl / tmpl_params) ---------------------------- */
#define TOKS_TMPL_NONE         0u    /* no pre-tokenizer split: the segment is one piece */
#define TOKS_TMPL_CL100K       1u    /* the gpt-2 / cl100k / llama-3 / qwen-2 / qwen-3.5 family (docs/kernels.md) */
#define TOKS_TMPL_O200K        2u    /* the o200k family (docs/templates/o200k.md): tmpl_params 0x02 o200k (gpt-oss,
                                        gpt-4o: CONTR_CI | DIGITS_1_3), 0x08 nemo (Mistral-Nemo, Nemotron: CONTR_NONE |
                                        DIGITS_1), 0x182 kimi (CONTR_CI | DIGITS_1_3 | HAN | NO_SLASH) */
#define TOKS_TMPL_DSV3         3u    /* the deepseek v3 / r1 / v3.1 / v3.2 / v4 / v4.1 three-split Sequence
                                        (docs/templates/dsv3.md); tmpl_params 0 */

/* TOKS_TMPL_CL100K parameters. TOKS_TMPL_O200K reads CONTR_MASK (NONE, or any other value: the (?i) suffix) and
 * DIGITS_MASK (DIGITS_1, or any other value: up to 3) with these values, plus TOKS_TP_HAN and TOKS_TP_NO_SLASH. */
#define TOKS_TP_CONTR_MASK     0x03u /* contractions alternative: */
#define TOKS_TP_CONTR_NONE     0x00u /*   absent */
#define TOKS_TP_CONTR_CS       0x01u /*   's|'t|'re|'ve|'m|'ll|'d   (case-sensitive, gpt-2) */
#define TOKS_TP_CONTR_CI       0x02u /*   (?i:'s|'t|'re|'ve|'m|'ll|'d)  (ascii case-insensitive + U+017F as s) */
#define TOKS_TP_LPREFIX_ANY    0x04u /* letter-run prefix: set = [^\r\n\p{L}\p{N}]?  clear = ' '? (U+0020 only) */
#define TOKS_TP_DIGITS_MASK    0x18u /* digits alternative: */
#define TOKS_TP_DIGITS_1_3     0x00u /*   \p{N}{1,3} */
#define TOKS_TP_DIGITS_1       0x08u /*   \p{N} */
#define TOKS_TP_DIGITS_SP_RUN  0x10u /*   ' '?\p{N}+   (gpt-2) */
#define TOKS_TP_DIGITS_RUN     0x18u /*   \p{N}+ */
#define TOKS_TP_PUNCT_NL       0x20u /* punctuation alternative ends with [\r\n]* (else ' '?[^\s\p{L}\p{N}]+ only) */
#define TOKS_TP_WS_NL          0x40u /* the \s*[\r\n]+ alternative is present (before \s+(?!\S)) */
#define TOKS_TP_DIGIT_CUT      0x200u /* hf Digits(individual_digits) runs first: every N atom is a piece and ends the
                                         regex's input (kernels.md §3 A8) */
#define TOKS_TP_NL_CUT         0x400u /* a newline run's start ends the regex's input (kernels.md §3 A9: Laguna) */
#define TOKS_TP_GB_SP          0x800u /* A6 gives back only a U+0020 (kernels.md §3 A10: bloom) */

/* TOKS_TMPL_O200K only (the kimi variant, docs/templates/o200k.md §6) */
#define TOKS_TP_HAN            0x80u /* rule O0 ([\p{Han}]+ first); TOKS_C_HAN atoms leave both case groups */
#define TOKS_TP_NO_SLASH       0x100u /* the punctuation tail is [\r\n]*, not [\r\n/]* */

/* ---- toks_tables.flags ---------------------------------------------------------------------------------- */
#define TOKS_TF_IGNORE_MERGES  0x01u /* bpe: a whole piece found in vhash encodes as that token (K6's probe is on) */
#define TOKS_TF_IDS_AS_RANK    0x02u /* merged ids rise strictly with merge rank: merge priorities ARE the
                                        merged ids and rank2id is unused (gpt-2, qwen, deepseek v3; NOT llama 3 or o200k,
                                        whose merges list several splits per merged id) */
#define TOKS_TF_CJK_L          0x04u /* every code point in U+4E00..U+A3FF and U+AC00..U+D6FF (CJK ideographs, Yi,
                                        hangul) has base L and no FOLD_S: K3 tiers may class those atoms without a
                                        lookup (set by the compiler, docs/kernels.md §2) */
#define TOKS_TF_CJK_B          0x08u /* every code point there has the class byte L | UPPER | LOWER exactly (letters in
                                        both of o200k's case groups, no mark, fold or Han bit): the o200k K3 tiers'
                                        shortcut (set by the compiler, docs/templates/o200k.md §2) */
#define TOKS_TF_CJK_D          0x10u /* dsv3 tables: every code point in U+4E00..U+9EFF has the class byte L | CJK
                                        exactly: the dsv3 K3 tiers' Han shortcut (set by the compiler, dsv3.md §4) */
#define TOKS_TF_TWIN           0x20u /* K3 runs the c twin on every tier (the compiler: kernels.md §2, §3.1) */
#define TOKS_TF_ASM_MERGE      0x40u /* spm on an asm tier: the model's merge loop is K6's asm (load.c, spm_c.c) */
#define TOKS_TF_PROBE_LONG     0x80u /* bpe: K6's whole-piece probe only for a piece over TOKS_KEY_MAXLEN bytes: K5's
                                     * words hold every shorter token the probe would be needed for (kernels.md §5) */
#define TOKS_TF_PROBE_ASCII    0x100u /* with TOKS_TF_PROBE_LONG: only for one that ends in an ascii byte (every vhash
                                     * token that ends in another byte bpes back to itself) */

/* rationale: docs/notes/c-core.md §layout.h.2 */
#define TOKS_KEY_MAXLEN        15
#define TOKS_HSEED             0x9E3779B9u
#define TOKS_VAL_COUNT_SHIFT   29
#define TOKS_BUCKET            64
#define TOKS_TAG_BITS          22
#define TOKS_TAG_MAX           0x3FFFFFu                 /* the largest tag (the epoch counter wraps after it) */
#define TOKS_TAG_MASK64        0xFFE00000FFE00000ull     /* a tag's bits in the val's high 8 bytes */
#define TOKS_K5_WARM           4096                      /* pieces since init after which static answers fill the cache */

/* rationale: docs/notes/c-core.md §layout.h.3 */
#define TOKS_FIB64             0x9E3779B97F4A7C15ull
#define TOKS_PRIO_BITS         22
#define TOKS_PRIO_NONE         0xFFFFFFFFu

/* rationale: docs/notes/c-core.md §layout.h.4 */
#define TOKS_PM_STAGE1         1056         /* 32 two-byte leads + 16 x 64 (three-byte lead, b1) */
#define TOKS_PM_HDR            4352         /* stage1's bytes rounded up to 64: the first block's offset */
#define TOKS_PM_BLOCK          512          /* 64 entries of 8 bytes */
#define TOKS_PM_NEVER          21
#define TOKS_PM_ALWAYS         63

/* the ascii premerge table (kernels.md 5.1): fold[256] (a neighbour byte's risk class, 0..31), then [16384] pair
 * entries of 16 B indexed b0 << 7 | b1: u32 id (TOKS_PRIO_NONE: none), u32 risk after, u32 risk before, 0 */
#define TOKS_APM_PAIRS         256
#define TOKS_APM_BYTES         (256 + 16384 * 16)

/* rationale: docs/notes/c-core.md §layout.h.5 */
#define TOKS_VSEED             0x85EBCA6Bu

/* ---- added-token entries (toks_tables.add_entries) ------------------------------------------------------ */
#define TOKS_AF_SPECIAL        0x01u
#define TOKS_AF_LSTRIP         0x02u
#define TOKS_AF_RSTRIP         0x04u
#define TOKS_AF_SINGLE_WORD    0x08u
#define TOKS_AF_NORMALIZED     0x10u
#define TOKS_AF_PFX            0x20u   /* matches only after a ▁ of the normalized gap (config.h pfx) */

/* ---- toks_tables: what kernels read (offsets are the contract) ------------------------------------- */

#define TT_MAGIC               0x000  /* u32 */
#define TT_VERSION             0x004  /* u32 */
#define TT_ALGO                0x008  /* u32 TOKS_ALGO_* */
#define TT_TMPL                0x00C  /* u32 TOKS_TMPL_* */
#define TT_TMPL_PARAMS         0x010  /* u32 TOKS_TP_* */
#define TT_FLAGS               0x014  /* u32 TOKS_TF_* */
#define TT_N_IDS               0x018  /* u32 every id < n_ids */
#define TT_N_MERGES            0x01C  /* u32 */
/* scanner classes */
#define TT_CLS_ASCII           0x020  /* const uint8_t *  [128] */
#define TT_CLS_STAGE1          0x028  /* const uint16_t * [0x1100]  block index of cp >> 8 */
#define TT_CLS_STAGE2          0x030  /* const uint8_t *  [n_blocks * 256] */
#define TT_CLS_NBLOCKS         0x038  /* u64 */
/* bpe */
#define TT_BYTE2ID             0x040  /* const uint32_t * [256] initial symbol of each byte */
#define TT_BYTEPAIR            0x048  /* const uint32_t * [65536] prio of a byte pair, or TOKS_PRIO_NONE */
#define TT_MERGE_SLOTS         0x050  /* const uint64_t * [(merge_mask + 1) * 8], 64-aligned */
#define TT_MERGE_MASK          0x058  /* u64 bucket mask (buckets are a power of two) */
#define TT_MERGE_SHIFT         0x060  /* u64 64 - log2(buckets) */
#define TT_MERGE_MAXPROBE      0x068  /* u64 >= 1 */
#define TT_RANK2ID             0x070  /* const uint32_t * [n_merges]; NULL when TOKS_TF_IDS_AS_RANK */
#define TT_PREMERGE            0x078  /* const uint8_t * the premerge table, or NULL (none certified) */
/* static shortcut table */
#define TT_WORDS               0x080  /* const uint8_t * TOKS_BUCKET-byte buckets, 64-aligned (may be empty) */
#define TT_WORDS_MASK          0x088  /* u64 bucket mask */
/* vocab hash */
#define TT_VHASH               0x090  /* const uint64_t *: mask + 1 slots; bpe tables: then u32 [256], the longest
                                               model token's length by its first byte (K6 skips the probe above it) */
#define TT_VHASH_MASK          0x098  /* u64 slot mask */
/* token bytes */
#define TT_TOK_OFF             0x0A0  /* const uint32_t * [n_ids + 1] */
#define TT_TOK_BYTES           0x0A8  /* const uint8_t * */
/* added tokens, per phase (0 = matched on the raw text, 1 = matched on the normalized text) */
#define TOKS_K1_H4_BITS        10     /* K1's h4 index, add_index + 2 * 65536: [1024][2] (slot << 1 | phase, kernels.md §4) */
#define TOKS_K1_LIST           8      /* an h4 bucket of more tokens is a radix tree (its entry's bit 63; kernels.md §4) */
#define RX_CH                  0x0    /* a radix node (16 bytes at add_cand + its offset): u32 first child's offset */
#define RX_LAB                 0x4    /* u32 offset of its label, then its children's first bytes */
#define RX_LLEN                0x8    /* u16 label length */
#define RX_NCH                 0xA    /* u16 children */
#define RX_ENT                 0xC    /* u32 entry + 1 of the token ending here, or 0 */
#define TT_ADD_SHUFTI          0x0B0  /* const uint8_t *  [2][32]: lo-nibble[16], hi-nibble[16] first-byte buckets */
#define TT_ADD_INDEX           0x0B8  /* const uint64_t * [2][65536]: first two bytes -> start | count << 32; then h4 */
#define TT_ADD_SINGLE          0x0C0  /* const uint32_t * [2][256]: one-byte token entry + 1, or 0 */
#define TT_ADD_CAND            0x0C8  /* const uint32_t * candidate entry indices, per bucket longest first (index2's, then h4's) */
#define TT_ADD_ENTRIES         0x0D0  /* const toks_added_entry * */
#define TT_ADD_BYTES           0x0D8  /* const uint8_t * token contents */
#define TT_ADD_N               0x0E0  /* u64 entries */
#define TT_ADD_PHASES          0x0E8  /* u64 bit p set = phase p has tokens */
#define TT_APM                 0x0F0  /* const uint8_t * the ascii premerge table, or NULL */
#define TT_PAIRF               0x0F8  /* const uint64_t * the merge keys' filter (bpe.h bpe_pf_bits), or NULL */
#define TT_RSV                 0x100  /* u64 [8] reserved for the next tables: zero; a table takes a slot by name */
#define TT_SIZE                0x140

/* toks_added_entry (16 bytes) */
#define TAE_OFF                0x0    /* u32 offset into add_bytes */
#define TAE_LEN                0x4    /* u16 */
#define TAE_FLAGS              0x6    /* u8 TOKS_AF_* */
#define TAE_PHASE              0x7    /* u8 */
#define TAE_ID                 0x8    /* u32 */
#define TAE_RSV                0xC
#define TAE_SIZE               0x10

/* ---- kernel argument structs (offsets are the contract) -------------------------------------------- */

/* K1 added-token find: up to cap leftmost-longest matches of the phase's tokens, the first at or after pos, each
 * next one at or after the previous one's end (kernels.md §4). uint64_t toks_k1_added_find_<tier>(t, a) returns n. */
#define K1_TEXT                0x00   /* const uint8_t * */
#define K1_LEN                 0x08   /* u64 */
#define K1_POS                 0x10   /* u64 in */
#define K1_PHASE               0x18   /* u64 in: 0 or 1 */
#define K1_M                   0x20   /* toks_k1_match * out: m[0, n) */
#define K1_CAP                 0x28   /* u64 in: >= 1 */
#define K1_N                   0x30   /* u64 out */
#define K1_NEXT                0x38   /* u64 out: where a next call resumes (the last match's end at n == cap, else len) */
#define K1_SIZE                0x40

/* K3 scan: pieces of text[pos, len) under the template; writes up to cap piece end offsets.
 * uint64_t toks_k3_scan_<tmpl>_<tier>(t, a) returns n (pieces written). */
#define K3_TEXT                0x00   /* const uint8_t * segment base */
#define K3_LEN                 0x08   /* u64 segment length (the true end for lookahead) */
#define K3_POS                 0x10   /* u64 in: a piece start; out: end of the last piece written */
#define K3_ENDS                0x18   /* uint32_t * out */
#define K3_CAP                 0x20   /* u64 in: >= 1 */
#define K3_N                   0x28   /* u64 out */
#define K3_FLAGS               0x30   /* u64 in: 0 (reserved) */
#define K3_RSV                 0x38
#define K3_SIZE                0x40

/* K5 encode pieces: shortcut probe, bpe on a miss, cache fill, emit.
 * uint64_t toks_k5_encode_<tier>(t, a) returns n_out (ids written at out). */
#define K5_TEXT                0x00   /* const uint8_t * segment base; pieces index into it */
#define K5_LEN                 0x08   /* u64 segment length: text[len..] is never read */
#define K5_ENDS                0x10   /* const uint32_t * piece ends */
#define K5_N                   0x18   /* u64 pieces */
#define K5_START               0x20   /* u64 start offset of the first piece */
#define K5_OUT                 0x28   /* uint32_t * */
#define K5_ROOM                0x30   /* u64 ids writable at out: >= bytes covered by the pieces + 4 */
#define K5_N_OUT               0x38   /* u64 out */
#define K5_CACHE               0x40   /* uint8_t * dynamic cache buckets (64-aligned) or NULL */
#define K5_CACHE_MASK          0x48   /* u64 bucket mask */
#define K5_WORK                0x50   /* uint8_t * bpe work area (64-aligned) */
#define K5_WORK_BYTES          0x58   /* u64 >= TOKS_BPE_WORK_BYTES(longest piece) */
#define K5_HITS_STATIC         0x60   /* u64 in/out: counters, K5 adds this call's counts (kernels.md §6) */
#define K5_HITS_CACHE          0x68   /* u64 in/out */
#define K5_MISSES              0x70   /* u64 in/out */
#define K5_CACHE_TAG           0x78   /* u64 in: the cache's tag (low TOKS_TAG_BITS bits used): entries hit only
                                         with it, fills write it (layout.h's val, kernels.md §6) */
#define K5_LCACHE              0x80   /* toks_lcache * or NULL: every K6 call goes through toks_k5_long (bpe.h) */
#define K5_SIZE                0x88

/* K6 bpe of one piece (hf BPE::tokenize semantics, incl. ignore_merges).
 * uint64_t toks_k6_bpe_<tier>(t, a) returns n_out. */
#define K6_PIECE               0x00   /* const uint8_t * */
#define K6_LEN                 0x08   /* u64 >= 1 */
#define K6_OUT                 0x10   /* uint32_t * room >= len */
#define K6_WORK                0x18   /* uint8_t * (64-aligned) */
#define K6_WORK_BYTES          0x20   /* u64 >= TOKS_BPE_WORK_BYTES(len) */
#define K6_N_OUT               0x28   /* u64 out */
#define K6_MERGES              0x30   /* u64 out: merges applied (counter) */
#define K6_RSV                 0x38
#define K6_SIZE                0x40

#define K7_TEXT                0x00   /* K7, the spm scan (kernels.md §8): const uint8_t * the unit */
#define K7_LEN                 0x08   /* u64 */
#define K7_POS                 0x10   /* u64 in: the window's first char; out: the next one's (unchanged: the twin's) */
#define K7_PREV                0x18   /* u64 in / out: the entry (spm.h) of the char before pos */
#define K7_ONE                 0x20   /* uint32_t * out: id | length << 24 of each 2-4 byte char at pos + j */
#define K7_MB                  0x28   /* u64 out: nonzero when the window holds one */
#define K7_FLAGS               0x30   /* u64 in: 1 = certified cuts */
#define SPM_STAGE1             0x200  /* the toks_spm fields K7 reads (spm.h checks them; ascii[] at 0) */
#define SPM_PAIRS              0x210
#define SPM_CUT                0x230
#define SPM_ID_REPL            0xA64
#define SPM_MS_SPLIT           0xA68
#define SPM_CUT_AB             0xC48
#define SPM_CUT_AB8            0x1C48 /* cut_ab8: the ascii pair bytes (spm.h) */

/* bpe work-area bound for a piece of n bytes (bytes); kernels must stay within it */
#define TOKS_BPE_WORK_BYTES(n) (256u + 32u * (n))

/* ---- c only ---------------------------------------------------------------------------------------- */
#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

typedef struct toks_added_entry {
    uint32_t off;
    uint16_t len;
    uint8_t  flags;
    uint8_t  phase;
    uint32_t id;
    uint32_t rsv;
} toks_added_entry;

typedef struct toks_tables {
    uint32_t magic, version, algo, tmpl, tmpl_params, flags, n_ids, n_merges;
    const uint8_t  *cls_ascii;
    const uint16_t *cls_stage1;
    const uint8_t  *cls_stage2;
    uint64_t        cls_nblocks;
    const uint32_t *byte2id;
    const uint32_t *bytepair;
    const uint64_t *merge_slots;
    uint64_t        merge_mask, merge_shift, merge_maxprobe;
    const uint32_t *rank2id;
    const uint8_t  *premerge;
    const uint8_t  *words;
    uint64_t        words_mask;
    const uint64_t *vhash;
    uint64_t        vhash_mask;
    const uint32_t *tok_off;
    const uint8_t  *tok_bytes;
    const uint8_t  *add_shufti;
    const uint64_t *add_index;
    const uint32_t *add_single;
    const uint32_t *add_cand;
    const toks_added_entry *add_entries;
    const uint8_t  *add_bytes;
    uint64_t        add_n, add_phases;
    const uint8_t  *apm;
    const uint64_t *pairf;
    uint64_t        rsv[8];
} toks_tables;

/* ---- table extents (core.h toks_tab; docs/kernels.md §1, docs/testing.md): every table a context reads, how far past
 * its end a reader may read (pad, its builder reserves it) and the alignment a reader may assume (align). The guard
 * build maps each table alone, a no-access page flush against its end + pad and again against its start: a read past
 * what is declared here faults, from c or from asm. A kernel that needs more says so here first. */
typedef struct toks_ext { uint32_t pad, align; } toks_ext;
#define TOKS_X(pad, align)     ((toks_ext){ (pad), (align) })
/* toks_tables: compile.c */
#define TOKS_X_TOK_OFF         TOKS_X(0u, 4u)    /* [n_ids + 1] */
#define TOKS_X_TOK_BYTES       TOKS_X(0u, 1u)    /* every id's bytes */
#define TOKS_X_CLS_ASCII       TOKS_X(0u, 1u)    /* [128] */
#define TOKS_X_CLS_STAGE1      TOKS_X(0u, 2u)    /* [0x1100] */
#define TOKS_X_CLS_STAGE2      TOKS_X(0u, 1u)    /* [n_blocks * 256] */
#define TOKS_X_SPECIAL         TOKS_X(0u, 4u)    /* n_ids bits in u64 words (ctx->special_ids) */
#define TOKS_X_ADD_ENTRIES     TOKS_X(0u, 4u)    /* [add_n] */
#define TOKS_X_ADD_BYTES       TOKS_X(0u, 1u)
#define TOKS_X_ADD_SHUFTI      TOKS_X(0u, 1u)    /* [2][32] */
#define TOKS_X_ADD_INDEX       TOKS_X(0u, 8u)    /* [2][65536], then h4 */
#define TOKS_X_ADD_SINGLE      TOKS_X(0u, 4u)    /* [2][256] */
#define TOKS_X_ADD_CAND        TOKS_X(0u, 4u)
/* the bpe tables: bpe_build.c (byte-level), spm_build.c (sentencepiece-style) */
#define TOKS_X_BYTE2ID         TOKS_X(0u, 4u)    /* [256] */
#define TOKS_X_MERGE_SLOTS     TOKS_X(0u, 64u)   /* 64-byte buckets */
#define TOKS_X_PAIRF           TOKS_X(0u, 8u)
#define TOKS_X_RANK2ID         TOKS_X(0u, 4u)
#define TOKS_X_BYTEPAIR        TOKS_X(0u, 4u)    /* [65536] */
#define TOKS_X_VHASH           TOKS_X(0u, 8u)    /* mask + 1 slots, then (byte-level) u32 [256] */
#define TOKS_X_WORDS           TOKS_X(0u, 64u)   /* TOKS_BUCKET-byte buckets (also unigram's, wordpiece's wtab) */
#define TOKS_X_PREMERGE        TOKS_X(0u, 8u)
#define TOKS_X_APM             TOKS_X(0u, 4u)    /* TOKS_APM_BYTES */
#define TOKS_X_SPM             TOKS_X(0u, 8u)    /* struct toks_spm (spm.h), K7 reads it by SPM_* */
#define TOKS_X_SPM_STAGE1      TOKS_X(0u, 2u)    /* [0x1100] */
#define TOKS_X_SPM_STAGE2      TOKS_X(0u, 4u)
#define TOKS_X_SPM_PAIRS       TOKS_X(0u, 8u)    /* mask + 2 slots: the last mirrors slot 0 */
#define TOKS_X_SPM_HOLES       TOKS_X(0u, 4u)    /* n_ids bits */
#define TOKS_X_SPM_AB8         TOKS_X(0u, 1u)    /* [65536] */
/* the vocabulary's lookups (vocab.c) and decode's table (stream.c) */
#define TOKS_X_VOC_SLOTS       TOKS_X(0u, 4u)
#define TOKS_X_VOC_ADD         TOKS_X(0u, 4u)
#define TOKS_X_VOC_DEC         TOKS_X(0u, 4u)    /* voc_n_dec 16-byte records (added_tokens_decoder) */
#define TOKS_X_VOC_BITS        TOKS_X(0u, 4u)    /* voc_added, voc_special: n_ids bits each */
#define TOKS_X_VOC_POOL        TOKS_X(0u, 1u)
#define TOKS_X_DEC_SLOT        TOKS_X(0u, 1u)    /* 16 bytes per id */
#define TOKS_X_DEC_LEN         TOKS_X(0u, 1u)    /* a byte per id */
/* wordpiece (wp.c) */
#define TOKS_X_WP              TOKS_X(0u, 8u)    /* struct toks_wp_tables */
#define TOKS_X_WP_KEYS         TOKS_X(0u, 1u)
#define TOKS_X_WP_ENTRIES      TOKS_X(0u, 4u)    /* word, cont */
#define TOKS_X_WP_CELLS        TOKS_X(0u, 4u)    /* da_len + 256 cells: the last 256 a step can reach */
#define TOKS_X_WP_TERM         TOKS_X(0u, 4u)
/* unigram (unigram.c) and its precompiled charsmap (precompiled.c) */
#define TOKS_X_UNI             TOKS_X(0u, 8u)    /* struct toks_uni */
#define TOKS_X_UNI_CELLS       TOKS_X(0u, 4u)    /* da_len + 256 cells: the last 256 a step can reach */
#define TOKS_X_UNI_SCORE       TOKS_X(0u, 8u)
#define TOKS_X_UNI_TERM        TOKS_X(0u, 4u)
#define TOKS_X_PC_STAGE1       TOKS_X(0u, 2u)    /* [0x1100] */
#define TOKS_X_PC_STAGE2       TOKS_X(0u, 4u)
#define TOKS_X_PC_KEYS         TOKS_X(0u, 8u)
#define TOKS_X_PC_VALS         TOKS_X(0u, 4u)
#define TOKS_X_PC_POOL         TOKS_X(0u, 1u)

typedef struct toks_k1_match { uint32_t start, end, entry, rsv; } toks_k1_match;   /* entry: index into add_entries */

typedef struct toks_k1_args {
    const uint8_t *text; uint64_t len, pos, phase; toks_k1_match *m; uint64_t cap, n, next;
} toks_k1_args;

typedef struct toks_k3_args {
    const uint8_t *text; uint64_t len, pos; uint32_t *ends; uint64_t cap, n, flags, rsv;
} toks_k3_args;

typedef struct toks_k5_args {
    const uint8_t *text; uint64_t len; const uint32_t *ends; uint64_t n, start; uint32_t *out;
    uint64_t room, n_out; uint8_t *cache; uint64_t cache_mask; uint8_t *work; uint64_t work_bytes;
    uint64_t hits_static, hits_cache, misses, cache_tag; void *lcache;
} toks_k5_args;

typedef struct toks_k6_args {
    const uint8_t *piece; uint64_t len; uint32_t *out; uint8_t *work; uint64_t work_bytes, n_out, merges, rsv;
} toks_k6_args;

typedef struct toks_k7_args {
    const uint8_t *text; uint64_t len, pos, prev; uint32_t *one; uint64_t mb, flags, rsv;
} toks_k7_args;

#define TOKS_LAYOUT_CHECK(T, f, off) _Static_assert(offsetof(T, f) == (off), #T "." #f)
TOKS_LAYOUT_CHECK(toks_tables, magic, TT_MAGIC);            TOKS_LAYOUT_CHECK(toks_tables, version, TT_VERSION);
TOKS_LAYOUT_CHECK(toks_tables, algo, TT_ALGO);              TOKS_LAYOUT_CHECK(toks_tables, tmpl, TT_TMPL);
TOKS_LAYOUT_CHECK(toks_tables, tmpl_params, TT_TMPL_PARAMS); TOKS_LAYOUT_CHECK(toks_tables, flags, TT_FLAGS);
TOKS_LAYOUT_CHECK(toks_tables, n_ids, TT_N_IDS);            TOKS_LAYOUT_CHECK(toks_tables, n_merges, TT_N_MERGES);
TOKS_LAYOUT_CHECK(toks_tables, cls_ascii, TT_CLS_ASCII);    TOKS_LAYOUT_CHECK(toks_tables, cls_stage1, TT_CLS_STAGE1);
TOKS_LAYOUT_CHECK(toks_tables, cls_stage2, TT_CLS_STAGE2);  TOKS_LAYOUT_CHECK(toks_tables, cls_nblocks, TT_CLS_NBLOCKS);
TOKS_LAYOUT_CHECK(toks_tables, byte2id, TT_BYTE2ID);        TOKS_LAYOUT_CHECK(toks_tables, bytepair, TT_BYTEPAIR);
TOKS_LAYOUT_CHECK(toks_tables, merge_slots, TT_MERGE_SLOTS); TOKS_LAYOUT_CHECK(toks_tables, merge_mask, TT_MERGE_MASK);
TOKS_LAYOUT_CHECK(toks_tables, merge_shift, TT_MERGE_SHIFT); TOKS_LAYOUT_CHECK(toks_tables, merge_maxprobe, TT_MERGE_MAXPROBE);
TOKS_LAYOUT_CHECK(toks_tables, rank2id, TT_RANK2ID);        TOKS_LAYOUT_CHECK(toks_tables, words, TT_WORDS);
TOKS_LAYOUT_CHECK(toks_tables, premerge, TT_PREMERGE);
TOKS_LAYOUT_CHECK(toks_tables, words_mask, TT_WORDS_MASK);  TOKS_LAYOUT_CHECK(toks_tables, vhash, TT_VHASH);
TOKS_LAYOUT_CHECK(toks_tables, vhash_mask, TT_VHASH_MASK);  TOKS_LAYOUT_CHECK(toks_tables, tok_off, TT_TOK_OFF);
TOKS_LAYOUT_CHECK(toks_tables, tok_bytes, TT_TOK_BYTES);    TOKS_LAYOUT_CHECK(toks_tables, add_shufti, TT_ADD_SHUFTI);
TOKS_LAYOUT_CHECK(toks_tables, add_index, TT_ADD_INDEX);    TOKS_LAYOUT_CHECK(toks_tables, add_single, TT_ADD_SINGLE);
TOKS_LAYOUT_CHECK(toks_tables, add_cand, TT_ADD_CAND);      TOKS_LAYOUT_CHECK(toks_tables, add_entries, TT_ADD_ENTRIES);
TOKS_LAYOUT_CHECK(toks_tables, add_bytes, TT_ADD_BYTES);    TOKS_LAYOUT_CHECK(toks_tables, add_n, TT_ADD_N);
TOKS_LAYOUT_CHECK(toks_tables, add_phases, TT_ADD_PHASES); TOKS_LAYOUT_CHECK(toks_tables, apm, TT_APM);
TOKS_LAYOUT_CHECK(toks_tables, pairf, TT_PAIRF);           TOKS_LAYOUT_CHECK(toks_tables, rsv, TT_RSV);
_Static_assert(sizeof(toks_tables) == TT_SIZE, "toks_tables size");
TOKS_LAYOUT_CHECK(toks_added_entry, off, TAE_OFF);   TOKS_LAYOUT_CHECK(toks_added_entry, len, TAE_LEN);
TOKS_LAYOUT_CHECK(toks_added_entry, flags, TAE_FLAGS); TOKS_LAYOUT_CHECK(toks_added_entry, phase, TAE_PHASE);
TOKS_LAYOUT_CHECK(toks_added_entry, id, TAE_ID);
_Static_assert(sizeof(toks_added_entry) == TAE_SIZE, "toks_added_entry size");
TOKS_LAYOUT_CHECK(toks_k1_args, text, K1_TEXT);  TOKS_LAYOUT_CHECK(toks_k1_args, len, K1_LEN);
TOKS_LAYOUT_CHECK(toks_k1_args, pos, K1_POS);    TOKS_LAYOUT_CHECK(toks_k1_args, phase, K1_PHASE);
TOKS_LAYOUT_CHECK(toks_k1_args, m, K1_M);        TOKS_LAYOUT_CHECK(toks_k1_args, cap, K1_CAP);
TOKS_LAYOUT_CHECK(toks_k1_args, n, K1_N);        TOKS_LAYOUT_CHECK(toks_k1_args, next, K1_NEXT);
_Static_assert(sizeof(toks_k1_args) == K1_SIZE, "k1 args size");
TOKS_LAYOUT_CHECK(toks_k3_args, text, K3_TEXT);  TOKS_LAYOUT_CHECK(toks_k3_args, len, K3_LEN);
TOKS_LAYOUT_CHECK(toks_k3_args, pos, K3_POS);    TOKS_LAYOUT_CHECK(toks_k3_args, ends, K3_ENDS);
TOKS_LAYOUT_CHECK(toks_k3_args, cap, K3_CAP);    TOKS_LAYOUT_CHECK(toks_k3_args, n, K3_N);
TOKS_LAYOUT_CHECK(toks_k3_args, flags, K3_FLAGS);
_Static_assert(sizeof(toks_k3_args) == K3_SIZE, "k3 args size");
TOKS_LAYOUT_CHECK(toks_k5_args, text, K5_TEXT);  TOKS_LAYOUT_CHECK(toks_k5_args, len, K5_LEN);
TOKS_LAYOUT_CHECK(toks_k5_args, ends, K5_ENDS);  TOKS_LAYOUT_CHECK(toks_k5_args, n, K5_N);
TOKS_LAYOUT_CHECK(toks_k5_args, start, K5_START); TOKS_LAYOUT_CHECK(toks_k5_args, out, K5_OUT);
TOKS_LAYOUT_CHECK(toks_k5_args, room, K5_ROOM);  TOKS_LAYOUT_CHECK(toks_k5_args, n_out, K5_N_OUT);
TOKS_LAYOUT_CHECK(toks_k5_args, cache, K5_CACHE); TOKS_LAYOUT_CHECK(toks_k5_args, cache_mask, K5_CACHE_MASK);
TOKS_LAYOUT_CHECK(toks_k5_args, work, K5_WORK);  TOKS_LAYOUT_CHECK(toks_k5_args, work_bytes, K5_WORK_BYTES);
TOKS_LAYOUT_CHECK(toks_k5_args, hits_static, K5_HITS_STATIC); TOKS_LAYOUT_CHECK(toks_k5_args, hits_cache, K5_HITS_CACHE);
TOKS_LAYOUT_CHECK(toks_k5_args, misses, K5_MISSES); TOKS_LAYOUT_CHECK(toks_k5_args, cache_tag, K5_CACHE_TAG);
TOKS_LAYOUT_CHECK(toks_k5_args, lcache, K5_LCACHE);
_Static_assert(sizeof(toks_k5_args) == K5_SIZE, "k5 args size");
TOKS_LAYOUT_CHECK(toks_k6_args, piece, K6_PIECE); TOKS_LAYOUT_CHECK(toks_k6_args, len, K6_LEN);
TOKS_LAYOUT_CHECK(toks_k6_args, out, K6_OUT);    TOKS_LAYOUT_CHECK(toks_k6_args, work, K6_WORK);
TOKS_LAYOUT_CHECK(toks_k6_args, work_bytes, K6_WORK_BYTES); TOKS_LAYOUT_CHECK(toks_k6_args, n_out, K6_N_OUT);
TOKS_LAYOUT_CHECK(toks_k6_args, merges, K6_MERGES);
_Static_assert(sizeof(toks_k6_args) == K6_SIZE, "k6 args size");
TOKS_LAYOUT_CHECK(toks_k7_args, pos, K7_POS);    TOKS_LAYOUT_CHECK(toks_k7_args, prev, K7_PREV);
TOKS_LAYOUT_CHECK(toks_k7_args, one, K7_ONE);    TOKS_LAYOUT_CHECK(toks_k7_args, flags, K7_FLAGS);
_Static_assert(sizeof(toks_k7_args) == 0x40, "k7 args size");

/* a dynamic-cache tag's bits as they sit in the val's high 8 bytes (the tag's low TOKS_TAG_BITS bits): an entry
 * of tag g has (val[8..16) as a little-endian u64) & TOKS_TAG_MASK64 == toks_tag_word(g) */
static inline uint64_t toks_tag_word(uint64_t tag)
{
    return ((tag & 0x7FFu) << 21) | (((tag >> 11) & 0x7FFu) << 53);
}

/* the portable definition of the crc32c step every hash above uses (bitwise; the core uses a table-driven
 * equivalent; the asm uses the instruction). */
static inline uint32_t toks_crc32c_u64_ref(uint32_t crc, uint64_t v)
{
    for (uint32_t i = 0; i < 64u; i++) {                    /* bound: 64 */
        uint32_t b = (crc ^ (uint32_t)v) & 1u;
        crc = (crc >> 1) ^ (0x82F63B78u & (0u - b));
        v >>= 1;
    }
    return crc;
}

#endif /* !__ASSEMBLER__ */
#endif /* TOKS_LAYOUT_H */
