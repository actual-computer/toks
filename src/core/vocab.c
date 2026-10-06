/* vocab.c: the vocabulary lookups of include/toks.h, toks_token_to_id and toks_id_flags (docs/notes/c-core.md
 * §vocab.c.1) */
#include "core.h"
#include "spm.h"

/* the index's key bytes: an added token's content (key < n_add: the block's content pool), else an id's string
 * (key - n_add: the id; toks_token's bytes) */
static const uint8_t *key_bytes(const toks_ctx *c, uint32_t key, uint64_t *n)
{
    if (key < c->voc_n_add) {
        const uint32_t *r = c->voc_add + 4u * (uint64_t)key;
        *n = r[1];
        return c->voc_pool + r[0];
    }
    uint32_t id = key - c->voc_n_add;
    *n = (uint64_t)c->t.tok_off[id + 1u] - c->t.tok_off[id];
    return c->t.tok_bytes + c->t.tok_off[id];
}

static uint32_t key_id(const toks_ctx *c, uint32_t key)
{
    return key < c->voc_n_add ? c->voc_add[4u * (uint64_t)key + 2u] : key - c->voc_n_add;
}

/* a multiply-xorshift hash of s[0, n), read 8 bytes at a time (the tail through a zeroed word: nothing past s[n) is
 * read); its low bits pick the slot, its top 9 bits are the slot's tag */
static uint64_t vhash(const uint8_t *s, uint64_t n)
{
    uint64_t h = 0x9E3779B97F4A7C15ull ^ n, w;
    uint64_t i = 0u;
    for (; i + 8u <= n; i += 8u) {                           /* bound: n / 8 */
        memcpy(&w, s + i, 8);
        h = toks_mix64(h, w);
    }
    if (i < n) {
        w = 0u;
        memcpy(&w, s + i, (size_t)(n - i));
        h = toks_mix64(h, w);
    }
    return toks_mix64(h, 0xD6E8FEB86659FD93ull);
}

#define VOC_KEY(v) (((v) & 0x7FFFFFu) - 1u)                  /* a slot's key; 0 = empty */
#define VOC_TAG(h) ((uint32_t)((h) >> 55))

/* the slot of s[0, n): the first in its probe sequence that is empty or holds a key with these bytes */
static uint32_t *voc_find(const toks_ctx *c, uint32_t *slots, const uint8_t *s, uint64_t n, uint64_t h)
{
    uint64_t i = h & c->voc_mask;
    uint32_t tag = VOC_TAG(h);
    for (uint64_t k = 0; k <= c->voc_mask; k++) {             /* bound: the slots (load <= 0.8 ends it sooner) */
        uint32_t v = slots[i];
        if (v == 0u) { return &slots[i]; }
        if ((v >> 23) == tag) {
            uint64_t kn;
            const uint8_t *kb = key_bytes(c, VOC_KEY(v), &kn);
            if (kn == n && memcmp(kb, s, (size_t)n) == 0) { return &slots[i]; }
        }
        i = (i + 1u) & c->voc_mask;
    }
    return NULL;                                             /* unreachable: a slot is always empty */
}

/* 1 when id's vocabulary string, as the file writes it, is s[0, n) itself (hf's token_to_id of that text) */
static int written(const struct toks_config *cfg, uint32_t id, const uint8_t *s, uint64_t n)
{
    return id < cfg->n_vocab && cfg->vocab_len[id] == n && memcmp(cfg->vocab[id], s, (size_t)n) == 0;
}

/* rationale: docs/notes/c-core.md §vocab.c.2 */
int64_t toks_vocab_build(toks_ctx *c, const struct toks_config *cfg)
{
    uint32_t n_ids = c->t.n_ids, n_add = cfg->n_added;
    uint64_t pool = 0u, keys = (uint64_t)n_add + n_ids;
    for (uint32_t i = 0u; i < n_add; i++) { pool += cfg->added[i].len; }   /* bound: n_added */
    uint64_t size = 16u;
    while (size < keys + keys / 4u) { size <<= 1; }          /* bound: 23 doublings (keys < 2^22): load <= 0.8 */
    uint64_t words = ((uint64_t)n_ids + 31u) / 32u;
    uint64_t o_add = 4u * size, o_flags = o_add + 16u * (uint64_t)n_add, o_pool = o_flags + 8u * words;
    uint64_t total = (o_pool + pool + 63u) & ~(uint64_t)63u;
    uint8_t *blk = toks_plat_arena(total);
    if (blk == NULL) { return TOKS_E_NOMEM; }
    memset(blk, 0, (size_t)total);
    c->mem_voc = blk;
    c->mem_voc_len = total;
    uint32_t *slots = (uint32_t *)toks_tab(blk, 0u, 4u * size, TOKS_X_VOC_SLOTS);
    c->voc_slots = slots;
    c->voc_mask = size - 1u;
    c->voc_add = (uint32_t *)toks_tab(blk, o_add, 16u * (uint64_t)n_add, TOKS_X_VOC_ADD);
    c->voc_added = (uint32_t *)toks_tab(blk, o_flags, 4u * words, TOKS_X_VOC_BITS);
    c->voc_special = (uint32_t *)toks_tab(blk, o_flags + 4u * words, 4u * words, TOKS_X_VOC_BITS);
    c->voc_pool = (uint8_t *)toks_tab(blk, o_pool, pool, TOKS_X_VOC_POOL);
    toks_tab_seal(blk, total);

    /* the added contents first, in file order: a content met again keeps its first id, as hf's added map does
     * (a later token with that content is given the id the content already has); a record also notes whether any
     * listing of its content is special (hf's special_tokens_set) */
    uint32_t na = 0u, at = 0u;
    uint32_t *add = (uint32_t *)(uintptr_t)c->voc_add;
    uint8_t *pl = (uint8_t *)(uintptr_t)c->voc_pool;
    c->voc_n_add = n_add;                                    /* key_bytes reads the records by this bound */
    for (uint32_t i = 0u; i < n_add; i++) {                  /* bound: n_added */
        const toks_cfg_added *a = &cfg->added[i];
        if (a->id >= n_ids) { continue; }                    /* unreachable: compile numbers every added id */
        memcpy(pl + at, a->content, a->len);
        add[4u * na] = at, add[4u * na + 1u] = a->len, add[4u * na + 2u] = a->id, add[4u * na + 3u] = 0u;
        uint64_t h = vhash(pl + at, a->len);
        uint32_t *sl = voc_find(c, slots, pl + at, a->len, h);
        if (sl == NULL) { return TOKS_E_NOMEM; }             /* unreachable */
        if (*sl == 0u) {
            *sl = (na + 1u) | VOC_TAG(h) << 23;
            at += a->len;
            na++;
        }
        add[4u * (VOC_KEY(*sl)) + 3u] |= a->special != 0u ? 1u : 0u;
    }
    c->voc_pool = (const uint8_t *)toks_tab_fit(c->voc_pool, at, TOKS_X_VOC_POOL);   /* pool was a bound */
    /* the id's flags: added; special when any listing of the content that holds the id (the last listed under it,
     * hf's added_tokens_decoder) is special */
    for (uint32_t i = 0u; i < n_add; i++) {                  /* bound: n_added */
        const toks_cfg_added *a = &cfg->added[i];
        if (a->id >= n_ids) { continue; }
        uint32_t *sl = voc_find(c, slots, a->content, a->len, vhash(a->content, a->len));
        uint32_t *ad = (uint32_t *)(uintptr_t)c->voc_added, *sp = (uint32_t *)(uintptr_t)c->voc_special;
        uint32_t m = 1u << (a->id & 31u);
        ad[a->id >> 5] |= m;
        if (sl != NULL && *sl != 0u && add[4u * VOC_KEY(*sl) + 3u] != 0u) { sp[a->id >> 5] |= m; } else { sp[a->id >> 5] &= ~m; }
    }
    c->voc_n_add = na;                                       /* keys >= na are ids from here on */
    /* then every id's string (toks_token's bytes): a content of the same bytes wins (hf reads the added map
     * first). Two ids whose strings are the same bytes: the one hf's token_to_id gives that text, i.e. the one
     * whose written vocabulary string is the bytes themselves (a ByteLevel file can hold a raw "\u200d" next to
     * the alphabet form "âĢį" of the same bytes), else the later */
    for (uint32_t id = 0u; id < n_ids; id++) {               /* bound: n_ids */
        uint64_t o = c->t.tok_off[id], n = (uint64_t)c->t.tok_off[id + 1u] - o;
        if (n == 0u) { continue; }                           /* no string (toks_token NULL) */
        const uint8_t *s = c->t.tok_bytes + o;
        uint64_t h = vhash(s, n);
        uint32_t *sl = voc_find(c, slots, s, n, h);
        if (sl == NULL) { return TOKS_E_NOMEM; }             /* unreachable */
        if (*sl != 0u && VOC_KEY(*sl) < na) { continue; }    /* an added content's */
        if (*sl != 0u && written(cfg, VOC_KEY(*sl) - na, s, n) && !written(cfg, id, s, n)) { continue; }
        *sl = (na + id + 1u) | VOC_TAG(h) << 23;
    }
    /* TOKS_ID_BYTE: an id that stands for exactly one raw byte: a <0xHH> string a chain with ByteFallback reads
     * as one byte (spm, unigram: voc_bf 1), or a byte-level id whose string is one byte (voc_bf 2) */
    c->voc_bf = (!c->dc.on && c->wp == NULL && c->dec_byte_level != 0u) ? 2u : 0u;
    for (uint32_t i = 0u; c->dc.on && i < c->dc.n_dec && i < TOKS_SPM_MAX_OPS; i++) {   /* bound: 8 ops */
        if (c->dc.dec[i].kind == TOKS_SPM_D_BYTE_FALLBACK) { c->voc_bf = 1u; }
    }
    return 0;
}

int64_t toks_token_to_id(const toks_ctx *ctx, const void *s, uint64_t len)
{
    if (ctx == NULL || (s == NULL && len != 0u)) { return TOKS_E_ARG; }
    if (len == 0u || len > TOKS_MAX_TOKEN_BYTES || ctx->voc_slots == NULL) { return TOKS_E_ID; }   /* no such string */
    const uint8_t *b = (const uint8_t *)s;
    uint32_t *sl = voc_find(ctx, (uint32_t *)(uintptr_t)ctx->voc_slots, b, len, vhash(b, len));
    if (sl == NULL || *sl == 0u) { return TOKS_E_ID; }
    return (int64_t)key_id(ctx, VOC_KEY(*sl));
}

int64_t toks_id_flags(const toks_ctx *ctx, uint32_t id)
{
    if (ctx == NULL) { return TOKS_E_ARG; }
    if (id >= ctx->t.n_ids) { return TOKS_E_ID; }
    uint32_t f = 0u;
    if (ctx->voc_added != NULL && toks_bit(ctx->voc_added, id) != 0u) {
        f |= TOKS_ID_ADDED | (toks_bit(ctx->voc_special, id) != 0u ? TOKS_ID_SPECIAL : 0u);
    }
    uint32_t o = ctx->t.tok_off[id];
    uint64_t n = (uint64_t)ctx->t.tok_off[id + 1u] - o;
    if ((ctx->voc_bf == 1u && toks_byte_token(ctx->t.tok_bytes + o, n) >= 0) || (ctx->voc_bf == 2u && n == 1u)) {
        f |= TOKS_ID_BYTE;
    }
    return (int64_t)f;
}
