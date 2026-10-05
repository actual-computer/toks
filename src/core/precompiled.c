/* precompiled.c: hf's Precompiled normalizer (docs/notes/c-core.md §precompiled.c.1) */
#include "unigram.h"

/* ---- the grapheme automaton (unicode-segmentation 1.13.3 check_pair + its look-backs, forward form) ---- */

int toks_gc_step(toks_gc_state *s, uint8_t cls)
{
    uint32_t a = (uint32_t)cls & TOKS_GC_CAT_MASK;
    int brk = 1;
    if (s->prev != 0xFFu) {
        uint32_t b = (uint32_t)s->prev & TOKS_GC_CAT_MASK;
        if (b == TOKS_GC_CR && a == TOKS_GC_LF) {                                  /* GB3 */
            brk = 0;
        } else if (b == TOKS_GC_CONTROL || b == TOKS_GC_CR || b == TOKS_GC_LF) {   /* GB4 */
            brk = 1;
        } else if (a == TOKS_GC_CONTROL || a == TOKS_GC_CR || a == TOKS_GC_LF) {   /* GB5 */
            brk = 1;
        } else if (b == TOKS_GC_L && (a == TOKS_GC_L || a == TOKS_GC_V || a == TOKS_GC_LV || a == TOKS_GC_LVT)) {
            brk = 0;                                                               /* GB6 */
        } else if ((b == TOKS_GC_LV || b == TOKS_GC_V) && (a == TOKS_GC_V || a == TOKS_GC_T)) {
            brk = 0;                                                               /* GB7 */
        } else if ((b == TOKS_GC_LVT || b == TOKS_GC_T) && a == TOKS_GC_T) {
            brk = 0;                                                               /* GB8 */
        } else if (a == TOKS_GC_EXTEND || a == TOKS_GC_ZWJ) {
            brk = 0;                                                               /* GB9 */
        } else if (a == TOKS_GC_SPACINGMARK) {
            brk = 0;                                                               /* GB9a */
        } else if (b == TOKS_GC_PREPEND) {
            brk = 0;                                                               /* GB9b */
        } else if (a == TOKS_GC_INCB_CONSONANT) {
            brk = !(s->incb_cons && s->incb_link);                                 /* GB9c */
        } else if (b == TOKS_GC_ZWJ && a == TOKS_GC_EXTPICT) {
            brk = !s->ep_zwj;                                                      /* GB11 */
        } else if (b == TOKS_GC_RI && a == TOKS_GC_RI) {
            brk = !s->ri_odd;                                                      /* GB12, GB13 */
        } else {
            brk = 1;                                                               /* GB999 */
        }
    }
    /* consume the atom */
    s->ri_odd = (uint8_t)((a == TOKS_GC_RI) ? !s->ri_odd : 0);
    uint8_t ep = s->ep_run;
    s->ep_run = (uint8_t)(a == TOKS_GC_EXTPICT || (a == TOKS_GC_EXTEND && ep));
    s->ep_zwj = (uint8_t)(a == TOKS_GC_ZWJ && ep);
    if (a == TOKS_GC_INCB_CONSONANT) {
        s->incb_cons = 1u;
        s->incb_link = 0u;
    } else if ((cls & TOKS_GC_INCB_LINKER) != 0u) {
        if (s->incb_cons) { s->incb_link = 1u; }
    } else if ((cls & TOKS_GC_INCB_EXTEND) == 0u) {
        s->incb_cons = 0u;
        s->incb_link = 0u;
    }
    s->prev = cls;
    return brk;
}

/* ---- base64 (the charsmap travels as a json string) ------------------------------------------------------ */

int toks_b64v(uint8_t c)
{
    if (c >= 'A' && c <= 'Z') { return c - 'A'; }
    if (c >= 'a' && c <= 'z') { return c - 'a' + 26; }
    if (c >= '0' && c <= '9') { return c - '0' + 52; }
    if (c == '+') { return 62; }
    if (c == '/') { return 63; }
    return -1;
}

int64_t toks_b64_len(const uint8_t *p, uint64_t n)
{
    if (n == 0u || (n & 3u) != 0u) { return -1; }
    uint64_t pad = (p[n - 1u] == '=') ? ((p[n - 2u] == '=') ? 2u : 1u) : 0u;
    for (uint64_t i = 0; i < n - pad; i++) {                /* bound: n */
        if (toks_b64v(p[i]) < 0) { return -1; }
    }
    if (pad == 2u && (toks_b64v(p[n - 3u]) & 0x0F) != 0) { return -1; }
    if (pad == 1u && (toks_b64v(p[n - 2u]) & 0x03) != 0) { return -1; }
    return (int64_t)(n / 4u * 3u - pad);
}

int64_t toks_b64_decode(const uint8_t *p, uint64_t n, uint8_t *out, uint64_t cap)
{
    int64_t m = toks_b64_len(p, n);
    if (m < 0 || (uint64_t)m > cap) { return -1; }
    uint64_t o = 0u;
    for (uint64_t i = 0; i < n; i += 4u) {                  /* bound: n / 4 groups */
        uint32_t v = 0u, k = 0u;
        for (uint32_t j = 0; j < 4u; j++) {                 /* bound: 4 */
            if (p[i + j] == '=') { break; }
            v |= (uint32_t)toks_b64v(p[i + j]) << (18u - 6u * j);
            k++;
        }
        out[o++] = (uint8_t)(v >> 16);
        if (k > 2u) { out[o++] = (uint8_t)(v >> 8); }
        if (k > 3u) { out[o++] = (uint8_t)v; }
    }
    return m;
}

/* ---- the double array ------------------------------------------------------------------------------------ */

static uint32_t du_offset(uint32_t u) { return (u >> 10) << ((u & (1u << 9)) >> 6); }
static uint32_t du_label(uint32_t u) { return u & 0x800000FFu; }
static uint32_t du_value(uint32_t u) { return u & 0x7FFFFFFFu; }
static int du_leaf(uint32_t u) { return ((u >> 8) & 1u) != 0u; }

typedef struct pc_blob {
    const uint8_t *units;        /* little-endian u32s */
    uint32_t       n_units;
    const uint8_t *str;          /* the strings */
    uint32_t       str_len;
} pc_blob;

static uint32_t unit_at(const pc_blob *b, uint32_t i)
{
    const uint8_t *p = b->units + 4u * (uint64_t)i;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int pc_parse(pc_blob *b, const uint8_t *blob, uint64_t len, const char **why)
{
    if (len < 4u) { *why = "precompiled charsmap: shorter than its header"; return -1; }
    uint32_t t = (uint32_t)blob[0] | ((uint32_t)blob[1] << 8) | ((uint32_t)blob[2] << 16) | ((uint32_t)blob[3] << 24);
    uint64_t nu = (uint64_t)t / 4u;
    if (4u + 4u * nu > len) { *why = "precompiled charsmap: trie past the blob"; return -1; }
    if (nu == 0u) { *why = "precompiled charsmap: empty trie"; return -1; }
    if (len - (4u + 4u * nu) > 0x3FFFFu) { *why = "precompiled charsmap: strings over 256 KiB"; return -1; }
    b->units = blob + 4;
    b->n_units = (uint32_t)nu;
    b->str = blob + 4u + 4u * nu;
    b->str_len = (uint32_t)(len - (4u + 4u * nu));
    if (!toks_utf8_valid(b->str, b->str_len)) { *why = "precompiled charsmap: strings not utf-8"; return -1; }
    return 0;
}

/* value v of a key: [v, the next NUL or the end) of the strings; -1 when hf would panic slicing it */
static int64_t pc_value_len(const pc_blob *b, uint32_t v)
{
    if (v > b->str_len) { return -1; }
    if (v < b->str_len && (b->str[v] & 0xC0u) == 0x80u) { return -1; }   /* off a char boundary */
    uint32_t e = v;
    while (e < b->str_len && b->str[e] != 0u) { e++; }       /* bound: str_len */
    return (int64_t)(e - v);
}

/* one walk over every reachable node: counts (when pc == NULL) or fills the tables. Returns 0 or -1. */
#define PC_MAX_DEPTH 64u
#define PC_MAX_PREFIXES (1u << 21)

typedef struct pc_frame {
    uint32_t node;         /* the node whose children are next */
    uint32_t next_label;   /* 1..256: the next label to try */
    uint32_t depth;        /* key bytes so far */
    uint32_t dead;         /* a shorter key is a prefix of every key below */
} pc_frame;

typedef struct pc_count {
    uint32_t n_keys, n_single, n_multi, n_blocks, max_expand;
    uint8_t  block_used[0x1100];
} pc_count;

static uint64_t mk_pack(const uint8_t *k, uint32_t n)
{
    uint64_t v = (uint64_t)n << 56;
    for (uint32_t i = 0u; i < n; i++) { v |= (uint64_t)k[i] << (8u * i); }  /* bound: 5 */
    return v;
}

static uint32_t mk_hash(uint64_t v) { return toks_crc32c_u64(0x9E3779B9u, v); }

static void mk_insert(toks_pc *pc, uint64_t key, uint32_t ent)
{
    uint32_t h = mk_hash(key) & pc->mk_mask;
    while (pc->mk_key[h] != 0u) { h = (h + 1u) & pc->mk_mask; }  /* bound: the table is <= half full */
    pc->mk_key[h] = key;
    pc->mk_val[h] = ent;
}

uint32_t toks_pc_multi(const toks_pc *pc, const uint8_t *key, uint32_t n)
{
    if (pc->mk_mask == 0u || n < 2u || n > 5u) { return 0u; }
    uint64_t v = mk_pack(key, n);
    uint32_t h = mk_hash(v) & pc->mk_mask;
    for (uint32_t i = 0u; i <= pc->mk_mask; i++) {          /* bound: table size */
        uint64_t k = pc->mk_key[h];
        if (k == 0u) { return 0u; }
        if (k == v) { return pc->mk_val[h]; }
        h = (h + 1u) & pc->mk_mask;
    }
    return 0u;
}

static uint32_t pc_entry(uint32_t off, uint32_t len) { return ((uint32_t)TOKS_PC_VALUE << 30) | (off << 12) | len; }

/* decodes the single utf-8 char that is all of k[0, n); returns its code point or -1 */
static int64_t one_char(const uint8_t *k, uint32_t n)
{
    uint32_t l = toks_utf8_len(k, n);
    if (l == 0u || l != n) { return -1; }
    return (l == 1u) ? (int64_t)k[0] : (int64_t)toks_cp_decode(k, l);
}

static int pc_walk(const pc_blob *b, pc_count *cnt, toks_pc *pc, const char **why)
{
    pc_frame st[PC_MAX_DEPTH + 1u];
    uint8_t key[PC_MAX_DEPTH + 1u];
    uint32_t root = du_offset(unit_at(b, 0u));             /* hf: node_pos = 0 ^ offset(array[0]) */
    uint32_t sp = 0u;
    st[0].node = root;
    st[0].next_label = 1u;
    st[0].depth = 0u;
    st[0].dead = 0u;
    uint64_t pushes = 0u;
    /* rationale: docs/notes/c-core.md §precompiled.c.2 */
    while (1) {
        pc_frame *f = &st[sp];
        if (f->next_label > 255u) {
            if (sp == 0u) { break; }
            sp--;
            continue;
        }
        uint32_t c = f->next_label++;
        uint32_t slot = f->node ^ c;
        if (slot >= b->n_units) {
            /* hf indexes array[node ^ c] for every input byte c at a reachable node: out of range panics */
            *why = "precompiled charsmap: a transition outside the array (hf panics on such input)";
            return -1;
        }
        uint32_t u = unit_at(b, slot);
        if (du_label(u) != c) { continue; }
        uint32_t nxt = slot ^ du_offset(u);
        uint32_t d = f->depth;
        if (d >= PC_MAX_DEPTH) { *why = "precompiled charsmap: key longer than 64 bytes"; return -1; }
        key[d] = (uint8_t)c;
        uint32_t dead = f->dead;
        if (du_leaf(u)) {
            if (nxt >= b->n_units) { *why = "precompiled charsmap: a leaf outside the array (hf panics)"; return -1; }
            uint32_t v = du_value(unit_at(b, nxt));
            int64_t vl = pc_value_len(b, v);
            if (vl < 0) { *why = "precompiled charsmap: a value off a char boundary (hf panics)"; return -1; }
            if (!toks_utf8_valid(key, d + 1u)) { *why = "precompiled charsmap: a key that is not utf-8"; return -1; }
            cnt->n_keys++;
            if (!dead) {
                int64_t cp = one_char(key, d + 1u);
                uint32_t ent = pc_entry(v, (uint32_t)vl);
                if ((uint32_t)vl > 0xFFFu) { *why = "precompiled charsmap: a value over 4 KiB"; return -1; }
                uint32_t ex = ((uint32_t)vl + d) / (d + 1u);  /* ceil(value bytes / key bytes) */
                if (cp >= 0) {
                    cnt->n_single++;
                    if (ex > cnt->max_expand) { cnt->max_expand = ex; }
                    uint32_t blk = (uint32_t)cp >> 8;
                    if (pc == NULL) {
                        if (!cnt->block_used[blk]) { cnt->block_used[blk] = 1u; cnt->n_blocks++; }
                    } else {
                        pc->stage2[(uint32_t)pc->stage1[blk] * 256u + ((uint32_t)cp & 0xFFu)] = ent;
                    }
                } else if (d + 1u <= 5u) {
                    cnt->n_multi++;
                    if (ex > cnt->max_expand) { cnt->max_expand = ex; }
                    if (pc != NULL) { mk_insert(pc, mk_pack(key, d + 1u), ent); }
                }
            }
            dead = 1u;                                       /* every longer key below is shadowed */
        }
        if (++pushes > PC_MAX_PREFIXES) { *why = "precompiled charsmap: more than 2^21 key prefixes"; return -1; }
        sp++;
        st[sp].node = nxt;
        st[sp].next_label = 1u;
        st[sp].depth = d + 1u;
        st[sp].dead = dead;
    }
    return 0;
}

static uint32_t pow2_at_least(uint32_t v)
{
    uint32_t p = 16u;
    while (p < v) { p <<= 1; }                              /* bound: 28 doublings */
    return p;
}

uint64_t toks_pc_bytes(const uint8_t *blob, uint64_t len, const char **why)
{
    pc_blob b;
    if (pc_parse(&b, blob, len, why) != 0) { return 0u; }
    pc_count cnt;
    memset(&cnt, 0, sizeof cnt);
    if (pc_walk(&b, &cnt, NULL, why) != 0) { return 0u; }
    uint64_t blocks = (uint64_t)cnt.n_blocks + 1u;
    uint32_t mk = pow2_at_least(2u * cnt.n_multi + 2u);
    return 64u + 2u * 0x1100u + 64u + blocks * 256u * 4u + 64u + (uint64_t)mk * 12u + 64u + (uint64_t)b.str_len + 64u;
}

static void *ar_take(toks_arena *ar, uint64_t n) { return toks_ar_alloc(ar, n, 64u); }

int64_t toks_pc_build(toks_pc *pc, const uint8_t *blob, uint64_t len, toks_arena *ar, const char **why)
{
    memset(pc, 0, sizeof *pc);
    pc_blob b;
    if (pc_parse(&b, blob, len, why) != 0) { return TOKS_E_UNSUPPORTED; }
    pc_count cnt;
    memset(&cnt, 0, sizeof cnt);
    if (pc_walk(&b, &cnt, NULL, why) != 0) { return TOKS_E_UNSUPPORTED; }

    uint32_t nblk = cnt.n_blocks + 1u;
    pc->stage1 = (uint16_t *)ar_take(ar, 2u * 0x1100u);
    pc->stage2 = (uint32_t *)ar_take(ar, (uint64_t)nblk * 256u * 4u);
    uint32_t mk = pow2_at_least(2u * cnt.n_multi + 2u);
    pc->mk_key = (uint64_t *)ar_take(ar, (uint64_t)mk * 8u);
    pc->mk_val = (uint32_t *)ar_take(ar, (uint64_t)mk * 4u);
    pc->pool = (uint8_t *)ar_take(ar, (uint64_t)b.str_len + 1u);
    if (pc->stage1 == NULL || pc->stage2 == NULL || pc->mk_key == NULL || pc->mk_val == NULL || pc->pool == NULL) {
        *why = "precompiled charsmap: arena";
        return TOKS_E_NOMEM;
    }
    memset(pc->stage1, 0, 2u * 0x1100u);
    memset(pc->stage2, 0, (size_t)nblk * 256u * 4u);
    memset(pc->mk_key, 0, (size_t)mk * 8u);
    memset(pc->mk_val, 0, (size_t)mk * 4u);
    uint32_t next = 1u;                                      /* block 0 stays all-identity */
    for (uint32_t blk = 0u; blk < 0x1100u; blk++) {          /* bound: 0x1100 */
        if (cnt.block_used[blk]) { pc->stage1[blk] = (uint16_t)next++; }
    }
    pc->n_blocks = nblk;
    pc->mk_mask = mk - 1u;
    memcpy(pc->pool, b.str, b.str_len);
    pc->pool_len = b.str_len;
    pc_count cnt2;
    memset(&cnt2, 0, sizeof cnt2);
    if (pc_walk(&b, &cnt2, pc, why) != 0) { return TOKS_E_UNSUPPORTED; }
    pc->n_keys = cnt.n_keys;
    pc->n_live_single = cnt.n_single;
    pc->n_live_multi = cnt.n_multi;
    pc->max_expand = cnt.max_expand > 1u ? cnt.max_expand : 1u;
    return 0;
}
