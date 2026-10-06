/* core.h: internal shared decls for the c core (docs/notes/c-core.md §core.h.1) */
#ifndef TOKS_CORE_H
#define TOKS_CORE_H

#include <stddef.h>
#include <stdint.h>

#include "toks.h"
#include "layout.h"

/* rationale: docs/notes/c-core.md §core.h.2 */
void *(memcpy)(void *dst, const void *src, size_t n);
void *(memset)(void *dst, int c, size_t n);
int   (memcmp)(const void *a, const void *b, size_t n);

/* rationale: docs/notes/c-core.md §core.h.3 */
static inline uint32_t toks_ld32(const void *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static inline void toks_st32(void *p, uint32_t v) { memcpy(p, &v, 4); }
static inline void toks_st64(void *p, uint64_t v) { memcpy(p, &v, 8); }   /* toks_split_points' offs: no alignment */
/* n bytes to a caller array: through void pointers, so neither the compiler nor UBSan assumes the element
 * type's alignment (a memcpy whose argument still has type uint32_t * is assumed 4-aligned) */
static inline void toks_cpy(void *dst, const void *src, uint64_t n) { memcpy(dst, src, (size_t)n); }

/* ---- platform (src/platform/, the only files allowed the os) ------------------------------------------ */

/* load-time allocation; returns NULL on failure or n == 0. free takes the size back because the
 * windows heap wants the handle-free form and callers know their sizes (no stowage of lengths). */
void *toks_plat_alloc(uint64_t n);
void  toks_plat_free(void *p, uint64_t n);

/* rationale: docs/notes/c-core.md §core.h.4 */
uint8_t *toks_plat_arena(uint64_t n);
void     toks_plat_arena_free(uint8_t *p, uint64_t n);
void     toks_plat_hint_huge(void *p, uint64_t n);   /* huge pages for [p, p + n)'s 2 MiB-aligned interior, if any */

/* rationale: docs/notes/c-core.md §core.h.5 */
int64_t toks_plat_getenv(const char *name, char *buf, uint64_t cap);

/* reads a whole file into memory (toks_plat_alloc'd, *len set; freed with toks_plat_free(p, len)).
 * TOKS_E_OPEN / TOKS_E_LIMIT / TOKS_E_NOMEM on failure; *is_dir = 1 when path is a directory. */
int64_t toks_plat_read_file(const char *path, uint8_t **out, uint64_t *len, int *is_dir);
/* the path of the tokenizer file inside a model directory, written NUL-terminated into
 * buf[0, cap): 0 when <dir>/tokenizer.json is a regular file, else TOKS_E_OPEN. */
int64_t toks_plat_dir_lookup(const char *dir, char *buf, uint64_t cap);

/* ---- diagnostics -------------------------------------------------------------------------------------- */

typedef struct toks_err {
    int64_t     code;
    const char *what;   /* static string, or NULL */
} toks_err;

#define TOKS_FAIL(c, w)      { toks_err e_ = { (c), (w) }; return e_; }
#define TOKS_FAILF(c)        { toks_err e_ = { (c), NULL }; return e_; }

/* ---- arena (all load-time memory; sized before use, never grown) -------------------------------------- */

typedef struct toks_arena {
    uint8_t *base;
    uint64_t len;
    uint64_t pos;
} toks_arena;

/* returns NULL when the arena is exhausted; align must be a power of two. [p, p + n) lies in [0, len) or the call
 * returns NULL: p > len (the aligned start past the end) is refused before len - p is taken */
static inline void *toks_ar_alloc(toks_arena *a, uint64_t n, uint64_t align)
{
    uint64_t p = (a->pos + (align - 1u)) & ~(align - 1u);
    if (p < a->pos || p > a->len || n > a->len - p) { return NULL; }   /* overflow / past the end / exhausted */
    a->pos = p + n;
    return a->base + p;
}

/* ---- tables: every one a context reads is taken here, its extent declared (layout.h TOKS_X_*) -------------------- */

/* A table is n bytes its builder places in a block (toks_plat_arena) at offset o, or takes from an arena; x says how far
 * past its end a reader may read (pad: the builder reserves it) and the alignment a reader may assume. The guard build
 * (TOKS_GUARD 1 or 2: make test-guard, docs/testing.md; the hooks in tests/common/guard.c) maps each table on its own
 * pages instead, a no-access page flush against its end + pad (1) or its start (2); at the seal it checks the block's
 * shipped placement (every table inside the block, no two overlapping) and closes the block, so a read outside a table,
 * or through a pointer that bypassed these, faults on its first byte. Shipped, each hook is the expression it stands for. */
#if defined(TOKS_GUARD)
#  define TOKS_GUARD_HOOK(guard, shipped) guard
void       *toks_guard_tab(const void *owner, const void *at, uint64_t n, uint64_t align);
const void *toks_guard_fit(const void *p, uint64_t n, uint64_t align);
uint8_t    *toks_guard_owner(const void *p);
void        toks_guard_seal(void *block, uint64_t n);
void        toks_guard_block(const void *block, uint64_t n);
void        toks_guard_release(const void *block, uint64_t n);
#else
#  define TOKS_GUARD_HOOK(guard, shipped) shipped
#endif
static inline void *toks_tab(uint8_t *block, uint64_t o, uint64_t n, toks_ext x)
{
    return TOKS_GUARD_HOOK(toks_guard_tab(block, block + o, n + x.pad, x.align), ((void)n, (void)x, block + o));
}
static inline void *toks_tab_ar(toks_arena *a, uint64_t n, uint64_t align, toks_ext x)
{
    void *p = toks_ar_alloc(a, n + x.pad, align);         /* the guard build too: the same accounting and refusals */
    return TOKS_GUARD_HOOK(p != NULL ? toks_guard_tab(a->base, p, n + x.pad, x.align) : NULL, p);
}
/* a table sized by a bound whose contents turned out to be its first n bytes: the guard build moves them to a table of
 * exactly n, so its end is the contents' end */
static inline const void *toks_tab_fit(const void *p, uint64_t n, toks_ext x)
{
    return TOKS_GUARD_HOOK(toks_guard_fit(p, n + x.pad, x.align), ((void)n, (void)x, p));
}
/* the block a table at its start was carved from (to free it), the builder has taken every table of block, and
 * toks_plat_arena's block mapped or about to be unmapped (mem.c) */
static inline uint8_t *toks_tab_owner(const void *p) { return TOKS_GUARD_HOOK(toks_guard_owner(p), (uint8_t *)(uintptr_t)p); }
static inline void toks_tab_seal(void *b, uint64_t n) { TOKS_GUARD_HOOK(toks_guard_seal(b, n), ((void)b, (void)n)); }
static inline void toks_tab_mapped(const void *b, uint64_t n) { TOKS_GUARD_HOOK(toks_guard_block(b, n), ((void)b, (void)n)); }
static inline void toks_tab_unmapped(const void *b, uint64_t n) { TOKS_GUARD_HOOK(toks_guard_release(b, n), ((void)b, (void)n)); }
/* a table block freed: its tables' maps too, whatever toks_plat_arena_free is (a test may stand in for it) */
static inline void toks_tab_free(uint8_t *b, uint64_t n) { toks_tab_unmapped(b, n); toks_plat_arena_free(b, n); }

/* ---- crc32c (table-driven; equals toks_crc32c_u64_ref, unit-tested) ----------------------------------- */

extern const uint32_t TOKS_CRC32C_TAB[256];

static inline uint32_t toks_crc32c_u32(uint32_t crc, uint32_t v)   /* 4 LE bytes */
{
    crc = TOKS_CRC32C_TAB[(crc ^ v) & 0xFFu] ^ (crc >> 8);
    crc = TOKS_CRC32C_TAB[(crc ^ (v >> 8)) & 0xFFu] ^ (crc >> 8);
    crc = TOKS_CRC32C_TAB[(crc ^ (v >> 16)) & 0xFFu] ^ (crc >> 8);
    crc = TOKS_CRC32C_TAB[(crc ^ (v >> 24)) & 0xFFu] ^ (crc >> 8);
    return crc;
}

static inline uint32_t toks_crc32c_u64(uint32_t crc, uint64_t v)    /* 8 LE bytes */
{
    crc = toks_crc32c_u32(crc, (uint32_t)v);
    crc = toks_crc32c_u32(crc, (uint32_t)(v >> 32));
    return crc;
}

/* key hashes for the shortcut tables (layout.h): lo = bytes 0-7 LE, hi = bytes 8-15 LE. */
static inline uint32_t toks_key_hash(uint32_t seed, const uint8_t *key /* 16 bytes */)
{
    uint64_t lo, hi;
    memcpy(&lo, key, 8);
    memcpy(&hi, key + 8, 8);
    return toks_crc32c_u64(toks_crc32c_u64(seed, lo), hi);
}

/* ---- utf-8 (shared by json validation and the kernels' atom reader) ----------------------------------- */

/* length of the well-formed utf-8 sequence at p with avail bytes remaining (1..4), or 0 when
 * ill-formed (then the caller consumes one byte: an invalid atom). */
static inline uint32_t toks_utf8_len(const uint8_t *p, uint64_t avail)
{
    uint8_t b = p[0];
    if (b < 0x80u) { return 1u; }
    if (b < 0xC2u) { return 0u; }                        /* continuation, C0, C1 */
    if (b < 0xE0u) {                                    /* C2-DF + 1 cont */
        if (avail < 2u || (p[1] & 0xC0u) != 0x80u) { return 0u; }
        return 2u;
    }
    if (b < 0xF0u) {
        if (avail < 3u) { return 0u; }
        if (b == 0xE0u && p[1] < 0xA0u) { return 0u; }  /* E0 A0-BF */
        if (b == 0xEDu && p[1] >= 0xA0u) { return 0u; } /* ED 80-9F (no surrogates) */
        if ((p[1] & 0xC0u) != 0x80u || (p[2] & 0xC0u) != 0x80u) { return 0u; }
        return 3u;
    }
    if (b == 0xF0u) {
        if (avail < 4u || p[1] < 0x90u || (p[1] & 0xC0u) != 0x80u) { return 0u; }
    } else if (b <= 0xF3u) {
        if (avail < 4u || (p[1] & 0xC0u) != 0x80u) { return 0u; }
    } else if (b == 0xF4u) {
        if (avail < 4u || p[1] > 0x8Fu || (p[1] & 0xC0u) != 0x80u) { return 0u; }
    } else {
        return 0u;
    }
    if ((p[2] & 0xC0u) != 0x80u || (p[3] & 0xC0u) != 0x80u) { return 0u; }
    return 4u;
}

/* one more byte of a utf-8 run (unicode §3.9 table 3-7): 0 when no extension of the bytes so far is
 * well-formed; need = the continuation bytes the last char still needs, [lo, hi] the next byte's range */
typedef struct toks_u8 { uint32_t need; uint8_t lo, hi; } toks_u8;
static inline int toks_u8_feed(toks_u8 *v, uint8_t c)
{
    if (v->need == 0u) {
        if (c < 0x80u) { return 1; }
        if (c < 0xC2u || c > 0xF4u) { return 0; }
        v->need = (c < 0xE0u) ? 1u : (c < 0xF0u) ? 2u : 3u;
        v->lo = (c == 0xE0u) ? 0xA0u : (c == 0xF0u) ? 0x90u : 0x80u;
        v->hi = (c == 0xEDu) ? 0x9Fu : (c == 0xF4u) ? 0x8Fu : 0xBFu;
        return 1;
    }
    if (c < v->lo || c > v->hi) { return 0; }
    v->lo = 0x80u;
    v->hi = 0xBFu;
    v->need--;
    return 1;
}

/* hf ByteFallback's byte token: "<0x" h h ">" with u8::from_str_radix(hh, 16), which takes a '+': its byte, or -1 */
static inline int toks_hexd(uint8_t c)
{
    return (c >= '0' && c <= '9') ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
}
static inline int toks_byte_token(const uint8_t *p, uint64_t k)
{
    if (k != 6u || p[0] != '<' || p[1] != '0' || p[2] != 'x' || p[5] != '>') { return -1; }
    if (p[3] == '+') { return toks_hexd(p[4]); }
    int hi = toks_hexd(p[3]), lo = toks_hexd(p[4]);
    return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
}

/* canonical base64 (standard alphabet, '=' padding, unused bits zero; precompiled.c): a char's 6-bit value or
 * -1; the decoded length of p[0, n) or -1; the decode into out[0, cap): its length, or -1 */
int     toks_b64v(uint8_t c);
int64_t toks_b64_len(const uint8_t *p, uint64_t n);
int64_t toks_b64_decode(const uint8_t *p, uint64_t n, uint8_t *out, uint64_t cap);

/* bit id of a bitmap of 32-bit words (special ids, holes) */
static inline uint32_t toks_bit(const uint32_t *bits, uint32_t id) { return (bits[id >> 5] >> (id & 31u)) & 1u; }

/* hf encode()'s steps after the post-processor (truncation, padding: wordpiece.md §7) and the WordPiece decoder,
 * as config.c reads them; the context holds a copy */
typedef struct toks_opts {
    uint32_t trunc_on;             /* truncation (Right, LongestFirst / OnlyFirst): the first L text ids are kept */
    uint32_t trunc_max;            /* max_length (L = it minus the template's ids when post-processing) */
    uint32_t trunc_stride;         /* only the overflow uses it: never in the ids */
    uint32_t pad_on;               /* padding (Fixed, or a multiple), after the template */
    uint32_t pad_fixed;            /* 1: Fixed(pad_len), 0: BatchLongest (its own length for one sequence) */
    uint32_t pad_len;
    uint32_t pad_multiple;         /* pad_to_multiple_of, 0 = none */
    uint32_t pad_id;
    uint32_t pad_left;             /* direction Left */
    uint32_t wp_win;               /* 1: no phase-1 form holds an ascii space / tab / cr / lf (wp_api.c windows) */
    uint32_t dec_wordpiece;        /* 1: hf's WordPiece decoder (prefix, cleanup); 0: tokens joined by " " */
    uint32_t dec_cleanup;
    uint32_t dec_prefix_len;       /* <= 16 */
    uint8_t  dec_prefix[16];
    uint32_t pad_file;             /* 1: the file pads (hf's padding is not None): pad_on, or BatchLongest alone, which
                                      pads no single text (toks_info's pad_on) */
    uint32_t pad_type_id;          /* the pad ids' type id (hf Encoding.type_ids; toks_info) */
} toks_opts;

/* encode's output cursor (wordpiece, unigram): counts ids up to lim (truncation), stores what fits in out[0, cap) */
typedef struct toks_emit { uint32_t *out; uint64_t cap, n, lim; } toks_emit;
static inline void toks_put(toks_emit *e, uint32_t v)
{
    if (e->n >= e->lim) { return; }
    if (e->n < e->cap) { toks_st32(e->out + e->n, v); }
    e->n++;
}

/* hf encode()'s padding of one encoding, after the template (wordpiece.md §7): to pad_len (Fixed) or the encoding's
 * own length, rounded up to pad_multiple; Left shifts what out holds. Padding never truncates. */
static inline void toks_pad(const toks_opts *o, toks_emit *e)
{
    uint64_t target = o->pad_fixed ? o->pad_len : e->n;
    uint64_t m = o->pad_multiple;
    if (m != 0u && target % m != 0u) { target += m - target % m; }
    if (e->n >= target) { return; }
    uint64_t p = target - e->n;
    if (o->pad_left == 0u) {
        for (uint64_t i = 0; i < p; i++) { toks_put(e, o->pad_id); }   /* bound: p <= 2^29 + 2^20 */
        return;
    }
    uint64_t w0 = (e->n < e->cap) ? e->n : e->cap;
    if (e->cap > p) {
        uint64_t keep = (w0 < e->cap - p) ? w0 : e->cap - p;
        for (uint64_t i = keep; i > 0u; i--) { toks_st32(e->out + p + i - 1u, toks_ld32(e->out + i - 1u)); }   /* bound: keep */
    }
    uint64_t f = (p < e->cap) ? p : e->cap;
    for (uint64_t i = 0; i < f; i++) { toks_st32(e->out + i, o->pad_id); }   /* bound: f */
    e->n += p;
}

/* decode's output cursor: counts every byte, stores what fits in out[0, cap) */
typedef struct toks_dsink { uint8_t *out; uint64_t cap, n; } toks_dsink;
static inline void toks_dput(toks_dsink *d, const void *p, uint64_t k)
{
    if (d->n < d->cap) { memcpy(d->out + d->n, p, (size_t)(d->cap - d->n < k ? d->cap - d->n : k)); }
    d->n += k;
}

/* whole-buffer strict utf-8 validation: 1 ok, 0 ill-formed. bound: one pass over n bytes. */
static inline int toks_utf8_valid(const uint8_t *p, uint64_t n)
{
    uint64_t i = 0;
    while (i < n) {                                     /* bound: n (i advances >= 1) */
        uint32_t k = toks_utf8_len(p + i, n - i);
        if (k == 0u) { return 0; }
        i += k;
    }
    return 1;
}

/* ---- context (driver-side; kernels only ever see toks_tables) ----------------------------------------- */

/* rationale: docs/notes/c-core.md §core.h.6 */
struct toks_spm_op;
/* the decoder chain stream.c runs for sentencepiece-style bpe and unigram: spm.h's ops (one per-token Replace or
 * Metaspace, ByteFallback, Fuse, Strip), pointed at the model's at load */
typedef struct toks_dchain {
    const struct toks_spm_op *dec;
    uint32_t       n_dec;
    uint32_t       has_decoder;    /* 0: hf joins the strings with " " */
    uint32_t       bf_first;       /* 1: ByteFallback runs before the per-token step, whose input its output is */
    uint32_t       on;             /* 1: this context decodes through the chain */
    const uint32_t *holes;         /* ids with no string at all (spm), or NULL */
} toks_dchain;

struct toks_ctx {
    toks_tables t;                 /* the compiled tables (layout.h) */
    uint32_t     tier;             /* TOKS_TIER_*: the tier this context dispatches to */
    uint32_t     dec_byte_level;   /* 1: the decoder is hf's ByteLevel (m1a refuses any other) */
    uint32_t     pp_ids[64];       /* the post-processor's single template: prefix ids, then suffix ids */
    uint32_t     n_pp_prefix;      /* ids before $A */
    uint32_t     n_pp_suffix;      /* ids after $A (n_pp_prefix + n_pp_suffix <= 64) */
    uint32_t     n_nonspecial;     /* added-token entries without TOKS_AF_SPECIAL (0: NONSPECIAL == NONE) */
    uint32_t     dec_max;          /* the longest toks_decode of one id (load.c, toks_dec_max): the weight of
                                      toks_stream_bound (stream.c) */
    uint64_t     identity;         /* the scratch binding (toks_ctx_identity) */
    uint64_t     cpu_features;     /* toks_cpu_features() at load: the word the tier was picked from, which
                                      toks_get_info reports (no syscall after load, SPEC §4.1) */
    uint32_t    *special_ids;      /* n_ids bits (in mem_tables), 1 = special: decode's TOKS_SKIP_SPECIAL */
    uint8_t     *dec_slot;         /* decode's fast table (toks_dec_build, one toks_plat_arena block): a 16-byte
                                      slot per id, then dec_len */
    uint8_t     *dec_len;          /* a length byte per id: 0..16 its bytes in its slot, 0xFE long, 0xFF slow */
    /* the vocabulary's lookups (vocab.c, one toks_plat_arena block, mem_voc): an open-addressed index of every id's
     * string and every added token's content (voc_slots, voc_mask + 1 u32 slots), the contents (voc_add: offset,
     * length, id, any listing special, per content, voc_n_add of them; their bytes in voc_pool), the added and
     * special bits per id */
    const uint32_t *voc_slots;
    uint64_t     voc_mask;
    const uint32_t *voc_add;
    const uint8_t  *voc_pool;
    const uint32_t *voc_added;
    const uint32_t *voc_special;
    uint32_t     voc_n_add;
    uint32_t     voc_bf;           /* TOKS_ID_BYTE's test: 1 a <0xHH> string (a ByteFallback chain), 2 a one-byte
                                      string (byte-level), 0 none */
    uint8_t     *mem_voc;
    uint64_t     mem_voc_len;
    uint8_t     *mem_tables;       /* compile's block (toks_plat_arena), or NULL */
    uint64_t     mem_tables_len;
    uint8_t     *mem_bpe;          /* the bpe tables' block (toks_plat_arena), or NULL */
    uint64_t     mem_bpe_len;
    char         name[64];
    uint8_t      source_sha256[32];
    uint32_t     nfc;              /* config.h's: the normalizer's form (norm.h); byte-level bpe normalizes every text
                                      gap into the scratch's norm region before phase-1 matching and K3 (kernels.md
                                      §7 step 2) */
    /* sentencepiece-style bpe (spm.h, the spm lane): the folded tables in their own block (toks_plat_arena),
     * built by load.c after compile; NULL for byte-level bpe */
    const struct toks_spm *spm;
    uint8_t     *mem_spm;
    uint64_t     mem_spm_len;
    toks_dchain  dc;               /* the decoder chain (spm, unigram), else dc.on == 0 */
    uint32_t     cut_chunk;        /* a tiktoken wrapper's cuts (config.h; docs/models/kimi.md §2.3), 0: none */
    uint32_t     cut_run;
    uint32_t     has_drop;         /* 1: a byte-level vocab lacks some byte chars (config.c): hf drops those bytes
                                      inside the model, so run_text encodes a piece holding one with them removed;
                                      the scratch then has the norm region whatever nfc says (api.c) */
    uint8_t      drop[32];         /* the dropped bytes: bit b & 7 of drop[b >> 3] */
    uint32_t     drop_unk, drop_fuse;   /* config.h: an unk for a dropped byte (id + 1), else 0 */

    /* ---- the wordpiece lane (docs/algorithms/wordpiece.md §7, §8, §12; wp_api.c): appended ------------- */
    const struct toks_wp_tables *wp;   /* the WordPiece model + normalizer tables (in mem_wp), else NULL */
    uint8_t     *mem_wp;           /* their block (toks_plat_arena), freed with the context */
    uint64_t     mem_wp_len;
    uint64_t     scr_extra;        /* work-region bytes beyond toks_scr_work: pieces + the scan's copy buffer */
    uint64_t     wp_mat_cap;       /* the scan's copy buffer (>= toks_wp_mat_min) */
    toks_opts    o;                /* encode options and the WordPiece decoder (config.c, copied whole) */

    /* ---- unigram (docs/algorithms/unigram.md; uni_api.c): appended. The model's tables (unigram.c, one
     * toks_plat_arena block), else NULL; truncation is trunc_max above (0: none), which hf encode() applies. */
    const struct toks_uni *uni;
    uint8_t     *mem_uni;
    uint64_t     mem_uni_len;

    /* ---- the generic pre-tokenizer (gen.c; docs/algorithms/generic.md): its program (toks_plat_alloc'd), else
     * NULL; its per-call memory is the scratch's extra region (scr_extra) */
    const struct toks_gen *gen;
    uint8_t     *mem_gen;
    uint64_t     mem_gen_len;

    /* ---- toks_encode_bound (compile.c toks_bound_terms, set by load.c): r = bound_num / bound_den ids per input
     * byte, bound_g ids; appended */
    uint32_t     bound_num, bound_den, bound_g, bound_rsv;

    /* ---- toks_template and toks_added (api.c, vocab.c): appended. The template's type ids in pp_ids' order and the
     * text's ($A's); hf's added_tokens_decoder, one record per added id in id order (voc_dec, voc_n_dec of them, in
     * mem_voc): the offset and length in voc_pool of the content listed last for the id, the id, its TOKS_ID_* flags
     * (ADDED, SPECIAL, LSTRIP, RSTRIP, SINGLE_WORD, NORMALIZED as that listing says) */
    uint32_t     pp_type[64];
    uint32_t     pp_seq_type;
    uint32_t     voc_n_dec;
    const uint32_t *voc_dec;
};

/* rationale: docs/notes/c-core.md §core.h.7 */
static inline uint64_t toks_ctx_identity(const uint8_t sha256[32])
{
    uint64_t h = (uint64_t)TOKS_TABLES_VERSION << 56;
    for (uint32_t i = 0; i < 8u; i++) { h ^= (uint64_t)sha256[i] << (8u * i); }   /* bound: 8 */
    return h | 1u;                                     /* never 0: a zeroed buffer is never bound */
}

/* ---- scratch (api.c): one caller buffer, its regions 64-aligned (docs/notes/c-core.md §core.h.8) -------- */

#define TOKS_SCRATCH_MAGIC   0x31524353534B4F54ull                        /* "TOKSSCR1" */
#define TOKS_CACHE_BUCKETS   32768u          /* the dynamic cache: 2 MiB (kernels.md §7 says why) */
#define TOKS_CACHE_MASK      (TOKS_CACHE_BUCKETS - 1u)
#define TOKS_CACHE_BYTES     ((uint64_t)TOKS_CACHE_BUCKETS * TOKS_BUCKET)
#ifndef TOKS_TEST_DEGEN                 /* the degenerate-hash build (tests/c/test_*_hash.c): every piece cache one */
#  define TOKS_TEST_DEGEN(x) (x)        /* bucket, the long cache's and the memo's hashes constant (SPEC §6, T2) */
#endif
#define TOKS_SCR_HDR         128u            /* the binding line + the counters line */
#define TOKS_SCR_FIXED       (TOKS_SCR_HDR + 4u * (uint64_t)TOKS_CHUNK_PIECES)   /* header + ends */
#define TOKS_NFC_X           3u     /* NFC writes at most 3 bytes per input byte (proof: norm.h; norm.c checks it
                                       against the generated tables' TOKS_NFC_EXPANSION) */
#define TOKS_NFKC_X          11u    /* NFKC / NFKD: U+FDFA's 3 bytes become 33 (TOKS_NFKC_EXPANSION) */

typedef struct toks_scratch {
    uint64_t magic;           /* line 0: the binding (the only line read before it is proven) */
    uint64_t identity;        /* the bound ctx's identity */
    uint64_t base;            /* the scr pointer init was given */
    uint64_t bytes;           /* the buffer size init was given */
    uint64_t max_len;         /* the longest text the regions hold */
    uint64_t off_cache;       /* region offsets from scr (the addresses are 64-aligned) */
    uint64_t off_work;
    uint64_t off_bounce;
    uint64_t hits_static;     /* line 1: K5's counters, summed over every encode call since init (zeroed by */
    uint64_t hits_cache;      /* init; the driver passes them to K5, which adds its counts: kernels.md §6) */
    uint64_t misses;
    uint64_t epoch;           /* the dynamic cache's tag, 1..TOKS_TAG_MAX (kernels.md §7: init moves to the next) */
    uint64_t cache_mib;       /* the flags' TOKS_SCRATCH_CACHE_MIB n (0: the default 2 MiB, no long cache) */
    uint64_t off_long;        /* the long-piece cache's buckets (the arena follows), or 0 */
    uint64_t long_pos;        /* the long cache's arena fill and generation (bpe.h toks_lcache; K5 updates them) */
    uint64_t long_gen;
} toks_scratch;
_Static_assert(sizeof(toks_scratch) == TOKS_SCR_HDR, "toks_scratch size");

static inline uint64_t toks_align64(uint64_t v) { return (v + 63u) & ~(uint64_t)63u; }
static inline uint64_t toks_mix64(uint64_t x, uint64_t w)       /* a multiply-xorshift step: the memo's and the */
{                                                               /* long cache's hashes (api.c, k5_long.c) */
    x = (x ^ w) * 0xFF51AFD7ED558CCDull;
    return x ^ (x >> 32);
}

/* the header of a scratch (at scr rounded up to 64), bound or not: tools and tests read K5's running
 * counters there (hits_static, hits_cache, misses) */
static inline toks_scratch *toks_scr_header(void *scr)
{
    uint64_t base = (uint64_t)(uintptr_t)scr;
    return (toks_scratch *)(void *)((uint8_t *)scr + (toks_align64(base) - base));
}
/* every pointer into a bound scratch's regions is formed in its shipped form p (base + an offset of core.h's layout) and
 * passed through toks_scr_p, which is p; the guard build (docs/testing.md) has toks_scratch_init map every region on its
 * own pages (toks_scr_carved: tests/common/guard.c), and these translate. toks_scr_zero zeroes [p, p + n), which the
 * guard build may find in more than one region; toks_scr_at and toks_scr_ends are the tests' forms */
#if defined(TOKS_GUARD)
struct toks_ctx;
void     toks_guard_scr(const struct toks_ctx *ctx, toks_scratch *h);
uint8_t *toks_guard_scr_at(const toks_scratch *h, uint64_t off);
void     toks_guard_scr_zero(const toks_scratch *h, uint64_t off, uint64_t n);
#endif
static inline uint8_t *toks_scr_p(const toks_scratch *h, uint8_t *p)
{
    return TOKS_GUARD_HOOK(toks_guard_scr_at(h, (uint64_t)(p - (uint8_t *)(uintptr_t)h->base)), ((void)h, p));
}
static inline void toks_scr_zero(const toks_scratch *h, uint8_t *p, uint64_t n)
{
    TOKS_GUARD_HOOK(toks_guard_scr_zero(h, (uint64_t)(p - (uint8_t *)(uintptr_t)h->base), n), ((void)h, memset(p, 0, (size_t)n)));
}
static inline void toks_scr_carved(const struct toks_ctx *ctx, toks_scratch *h)
{
    TOKS_GUARD_HOOK(toks_guard_scr(ctx, h), ((void)ctx, (void)h));
}
static inline uint8_t *toks_scr_at(const toks_scratch *h, uint64_t off) { return toks_scr_p(h, (uint8_t *)(uintptr_t)h->base + off); }
static inline uint32_t *toks_scr_ends(toks_scratch *h) { return (uint32_t *)(void *)toks_scr_p(h, (uint8_t *)h + TOKS_SCR_HDR); }
static inline uint64_t toks_scr_work(uint64_t max_len) { return toks_align64(TOKS_BPE_WORK_BYTES(max_len)); }
static inline uint64_t toks_scr_bounce(uint64_t max_len) { return toks_align64(4u * (max_len + 4u)); }
static inline uint64_t toks_scr_tmax(uint64_t max_len, uint32_t x)    /* x: the norm region's factor, 0 none */
{
    return x != 0u ? (uint64_t)x * max_len : max_len;
}
static inline uint64_t toks_scr_norm(uint64_t max_len, uint32_t x)
{
    return x != 0u ? toks_align64(toks_scr_tmax(max_len, x)) : 0u;
}
/* TOKS_SCRATCH_CACHE_MIB(n): 0 = TOKS_CACHE_BYTES of short cache only, else n = 4..128 (a power of two): n / 2 MiB of
 * short cache, n / 8 MiB of long-cache buckets, 3n / 8 MiB of long-cache arena; any other n: TOKS_E_ARG */
static inline uint64_t toks_scr_cache_mib(uint32_t flags) { return ((uint64_t)flags >> 12) & 0xFFu; }
static inline int toks_scr_cache_ok(uint64_t n) { return n == 0u || (n >= 4u && n <= 128u && (n & (n - 1u)) == 0u); }
static inline uint64_t toks_scr_short(uint64_t n) { return n == 0u ? TOKS_CACHE_BYTES : n << 19; }
static inline uint64_t toks_scr_long_buckets(uint64_t n) { return n << 17; }
static inline uint64_t toks_scr_long_arena(uint64_t n) { return 3u * (n << 17); }
static inline uint64_t toks_scr_caches(uint64_t n) { return toks_scr_short(n) + (n << 19); }
/* TOKS_SCRATCH_MEMO_MIB(m): the segment memo, m MiB between the piece caches and work (api.c), else TOKS_MEMO_MIB (SPEC
 * §6's default: toks.h); its base, or NULL */
#define TOKS_MEMO_MIB 4u
static inline uint64_t toks_scr_memo_bytes(uint32_t flags)
{
    return ((flags & TOKS_SCRATCH_MEMO_MIB(0xFFFu)) != 0u ? (uint64_t)flags & 0xFFFu : TOKS_MEMO_MIB) << 20;
}
static inline uint8_t *toks_scr_memo(toks_scratch *h, uint64_t *bytes)
{
    *bytes = h->off_work - h->off_cache - toks_scr_caches(h->cache_mib);
    return *bytes != 0u ? toks_scr_p(h, (uint8_t *)(uintptr_t)h->base + h->off_cache + toks_scr_caches(h->cache_mib)) : NULL;
}
/* the memo's first line, its header (api.c memo_*, kernels.md §7): every reader names a field, so moving one is a
 * compile error there, never a silently wrong number */
typedef struct toks_memo_head {
    uint64_t pos;            /* the ring's write position: record bytes written since init */
    uint64_t hits;           /* segments answered since init */
    uint64_t drought;        /* record bytes written since the last hit (admission) */
    uint64_t probes;         /* lookups */
    uint64_t differ;         /* slots that matched a lookup whose bytes then differed (the hash's windows) */
    uint64_t vpos;           /* where a ring that never stopped would write: every record, refused record and hit */
    uint64_t lap;            /* where this lap started (a multiple of the ring) */
    uint64_t run;            /* refused records since the last hit, a second sight a lapping ring held weighing more */
} toks_memo_head;
/* a full lap keeps what it holds until its refused records since the last hit reach TOKS_MEMO_DRY, each counting 1,
 * or TOKS_MEMO_DRY / TOKS_MEMO_GHOSTS when it is a second sight that a lapping ring would still hold (kernels.md §7) */
#define TOKS_MEMO_DRY 1024u
#define TOKS_MEMO_GHOSTS 2u
_Static_assert(sizeof(toks_memo_head) == 64, "toks_memo_head size");
/* the memo's write position and hits since init; 0 and 0 without a memo */
static inline void toks_scr_memo_ctr(toks_scratch *h, uint64_t *pos, uint64_t *hits)
{
    uint64_t mb;
    const toks_memo_head *m = (const toks_memo_head *)(const void *)toks_scr_memo(h, &mb);
    *pos = m != NULL ? m->pos : 0u;
    *hits = m != NULL ? m->hits : 0u;
}
static inline uint64_t toks_scratch_size(uint64_t max_len, uint32_t x, uint64_t n)    /* max_len <= TOKS_MAX_TEXT */
{
    uint64_t t = toks_scr_tmax(max_len, x);
    return 63u + TOKS_SCR_FIXED + toks_scr_caches(n) + toks_scr_work(t) + toks_scr_bounce(t) +
           toks_scr_norm(max_len, x);
}
/* rationale: docs/notes/c-core.md §core.h.9 */
#define TOKS_WP_MAT_CAP      24704u
#define TOKS_WP_WIN          8192u   /* raw bytes per window of the normalized path (wp_api.c) */
#define TOKS_WP_SCR_EXTRA    (16u * (uint64_t)TOKS_CHUNK_PIECES + TOKS_WP_MAT_CAP + 512u)

/* U+2581 (▁, sentencepiece's meta symbol) at p[i, i + 3) of p[0, n) */
static inline int toks_meta_at(const uint8_t *p, uint64_t n, uint64_t i)
{
    return n - i >= 3u && i < n && p[i] == 0xE2u && p[i + 1u] == 0x96u && p[i + 2u] == 0x81u;
}

/* rationale: docs/notes/c-core.md §core.h.10 */
#define TOKS_SEG_TOKEN   1u
#define TOKS_SEG_GAP     2u

typedef struct toks_seg_out {
    uint32_t kind;      /* TOKS_SEG_* */
    uint32_t id;        /* TOKS_SEG_TOKEN: the added token's id */
    uint64_t start;     /* byte range of the unit, within the cursor's text */
    uint64_t end;
} toks_seg_out;

#define TOKS_SEG_BATCH   16u         /* K1 matches per call (kernels.md §4) */

/* K1's h4 index (kernels.md §4): the slot of the first four bytes */
static inline uint32_t toks_k1_h4(const uint8_t *s)
{
    uint32_t k;
    memcpy(&k, s, 4u);
    return (uint32_t)(k * 0x9E3779B1u) >> (32u - TOKS_K1_H4_BITS);
}

typedef struct toks_seg_iter {
    const toks_tables *t;
    const uint8_t     *text;
    uint64_t           len;
    uint64_t           pos;          /* where the next K1 call searches from (the raw resume point after m[mn - 1]) */
    uint64_t           prev_end;     /* hf's start_offset: end of the last emitted unit */
    uint64_t           pend_start;   /* the pending token (its gap was the last unit) */
    uint64_t           pend_end;
    uint64_t           rs_from, rs_to;  /* the last rstrip walk: [rs_from, rs_to) is whole \s atoms, rs_to its end */
    uint32_t           pend_id;
    uint32_t           mode;         /* TOKS_ADDED_* */
    uint32_t           phase;        /* 0 or 1 */
    uint32_t           tier;         /* TOKS_TIER_* (dispatch at the kernel call site) */
    uint32_t           fin;          /* 1 once the final gap has been emitted */
    uint32_t           pend;         /* 1 when a token waits behind its gap */
    uint32_t           mi, mn;       /* m[mi, mn): K1's raw matches not yet taken */
    toks_k1_match      m[TOKS_SEG_BATCH];
} toks_seg_iter;

void toks_seg_begin(toks_seg_iter *it, const toks_tables *t, uint32_t mode, uint32_t phase,
                    uint32_t tier, const uint8_t *text, uint64_t len);
int  toks_seg_next(toks_seg_iter *it, toks_seg_out *u);   /* 1 = *u is a unit, 0 = done */

/* class of a non-ascii code point (k3_c.c reads it; classes.c builds the tables):
 * stage1[cp >> 8] selects the 256-entry block, stage2 the class inside it. */
static inline uint8_t toks_cls_cp(const toks_tables *t, uint32_t cp)
{
    uint32_t b1 = (uint32_t)t->cls_stage1[cp >> 8];
    return t->cls_stage2[b1 * 256u + (cp & 0xFFu)];
}

/* encodes cp (a scalar value) at o (room >= 4); returns the length */
static inline uint32_t toks_utf8_put(uint8_t *o, uint32_t cp)
{
    if (cp < 0x80u) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800u) {
        o[0] = (uint8_t)(0xC0u | (cp >> 6)); o[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        o[0] = (uint8_t)(0xE0u | (cp >> 12)); o[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        o[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    o[0] = (uint8_t)(0xF0u | (cp >> 18)); o[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
    o[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)); o[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 4;
}

/* utf-8 -> code point for a sequence known well-formed (len 2..4). */
static inline uint32_t toks_cp_decode(const uint8_t *p, uint32_t k)
{
    if (k == 2u) { return (((uint32_t)p[0] & 0x1Fu) << 6) | ((uint32_t)p[1] & 0x3Fu); }
    if (k == 3u) { return (((uint32_t)p[0] & 0x0Fu) << 12) | (((uint32_t)p[1] & 0x3Fu) << 6) | ((uint32_t)p[2] & 0x3Fu); }
    return (((uint32_t)p[0] & 0x07u) << 18) | (((uint32_t)p[1] & 0x3Fu) << 12) | (((uint32_t)p[2] & 0x3Fu) << 6) | ((uint32_t)p[3] & 0x3Fu);
}

/* the regex-crate \s set (the parity lane's verified 25 code points, hf's lstrip/rstrip runs):
 * 09-0D, 20, 85, A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000. */
int toks_is_regex_ws(uint32_t cp);
/* the regex-crate \w of hf's single_word (src/gen/rx_word.c, probed through hf: 144667 code points). */
int toks_is_regex_word(uint32_t cp);

/* rationale: docs/notes/c-core.md §core.h.11 */
typedef struct toks_lossy {
    uint8_t *out;
    uint64_t cap;
    uint64_t n;            /* bytes produced so far; out[0, min(cap, n)) holds them */
    uint32_t np;           /* the held prefix's length, 0..3 */
    uint8_t  pend[4];      /* the held prefix */
} toks_lossy;

/* appends the bytes of the ids hf's decode keeps (ids without a string add nothing; specials are
 * skipped under TOKS_SKIP_SPECIAL). TOKS_E_ID at the first id >= n_ids (d advanced up to it), else 0. */
int64_t  toks_lossy_ids(const struct toks_ctx *ctx, toks_lossy *d, const uint32_t *ids, uint64_t n,
                        uint32_t flags);
/* the end of the input: a held prefix becomes one U+FFFD. */
void     toks_lossy_end(toks_lossy *d);
/* rationale: docs/notes/c-core.md §core.h.12 */
int64_t  toks_dec_build(struct toks_ctx *c);
void     toks_dec_free(struct toks_ctx *c);           /* NULL-safe on dec_slot */

/* rationale: docs/notes/c-core.md §core.h.13 */
struct toks_config;
#include "json.h"
#include "config.h"

/* load pipeline pieces implemented elsewhere */
int64_t toks_compile(const struct toks_config *cfg, struct toks_ctx *ctx, toks_arena *parse_ar);
int64_t toks_vocab_build(struct toks_ctx *c, const struct toks_config *cfg);   /* vocab.c: the lookups' index */

/* ---- the generic pre-tokenizer (gen.c, docs/algorithms/generic.md): a chain's steps as config.c reads them -------- */
enum { TOKS_GS_SPLIT, TOKS_GS_DIGITS, TOKS_GS_PUNCT, TOKS_GS_BYTELEVEL };
enum { TOKS_GB_ISOLATED, TOKS_GB_REMOVED, TOKS_GB_PREV, TOKS_GB_NEXT, TOKS_GB_CONTIGUOUS };   /* hf's behaviors */
typedef struct toks_gen_spec {
    uint32_t kind, beh, inv;      /* TOKS_GS_*, TOKS_GB_*, invert */
    uint32_t lit;                 /* Split: a String pattern; ByteLevel: use_regex */
    const uint8_t *s;             /* Split: the pattern */
    uint32_t n;
} toks_gen_spec;
struct toks_gen;
int64_t  toks_gen_compile(const toks_gen_spec *sp, uint32_t n, toks_arena *ar, const struct toks_gen **out,
                          uint64_t *bytes, toks_err *err);
uint64_t toks_gen_scr(const struct toks_gen *g);
void     toks_gen_run(const struct toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len, uint64_t base,
                      toks_emit *e, int ids);
/* one round of consecutive pieces seg[pos, ends[0]), [ends[0], ends[1]), .. (n >= 1): their ends (base + end) or
 * their ids (api.c: K5, or the dropped-byte path) */
void     toks_round(const struct toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t len, uint64_t pos,
                    const uint32_t *ends, uint64_t n, uint64_t base, toks_emit *e, int ids);
void    toks_sha256(const uint8_t *data, uint64_t len, uint8_t out[32]);

/* bpe tables (the bpe lane, src/core/bpe.h; the driver's wire-up, present once it lands). */
uint64_t toks_bpe_tables_bytes(const struct toks_config *cfg);
int64_t  toks_bpe_build(toks_tables *t, toks_arena *ar, const struct toks_config *cfg);

/* the Unigram family's driver (src/core/uni_api.c): encode for a context whose t.algo is TOKS_ALGO_UNIGRAM,
 * called by api.c's run() after its argument and scratch checks; the scratch header is h. */
int64_t toks_uni_run(const struct toks_ctx *ctx, const toks_scratch *h, const uint8_t *text, uint64_t len,
                     uint32_t flags, uint32_t *out, uint64_t cap, int ids);
int64_t toks_uni_load(struct toks_ctx *c, const struct toks_config *cfg, const char **why);
int64_t toks_uni_dec(const struct toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *out,
                     uint64_t cap);

/* the byte-level alphabet (gpt-2 bytes_to_unicode): the byte a char stands for, or -1 (alloc.c). */
int32_t  toks_char_byte(uint32_t cp);

/* WordPiece contexts (wp_api.c): api.c hands them encode / pieces (ids = 0) once the arguments and the
 * scratch are checked, and decode once every id is below n_ids. */
int64_t  toks_wp_run(const struct toks_ctx *ctx, toks_scratch *h, const uint8_t *text, uint64_t len,
                     uint32_t flags, uint32_t *out, uint64_t cap, int ids);
int64_t  toks_wp_decode(const struct toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags,
                        uint8_t *out, uint64_t cap);
int64_t  toks_wp_decode_k(const struct toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags,
                          uint8_t *out, uint64_t cap, uint64_t *kept);      /* the stream's step (stream.c) */

#endif /* TOKS_CORE_H */
