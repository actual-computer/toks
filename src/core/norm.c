/* norm.c: the normalizer (docs/notes/c-core.md §norm.c.1) */
#include "norm.h"

#include "../gen/bert_tables.h"
#include "../gen/norm_nfc.h"

_Static_assert(TOKS_NFC_X == TOKS_NFC_EXPANSION && TOKS_NFKC_X == TOKS_NFKC_EXPANSION, "the scratch's factors are the "
               "generated worst-case expansions");
_Static_assert(TOKS_NORM_MAX_OUT == TOKS_BERT_MAX_DECOMP * TOKS_BERT_MAX_LOWER, "a char's chars, each lowercased");

/* ---- K2: the first byte >= hot (0xCC: NFC, 0xC2: NFKC) ------------------------------------------------- */

/* bit 7 of byte i set iff byte i >= hot (>= 0xC0): its top two bits set and its low six >= hot's. The add (a =
 * 0x40 - hot's low six in every byte) cannot carry across bytes: (b & 0x3F) + a <= 0x7F. */
static inline uint64_t hot_bytes(uint64_t w, uint64_t a)
{
    uint64_t lo6 = (w & 0x3F3F3F3F3F3F3F3Full) + a;
    return w & (w << 1) & (lo6 << 1) & 0x8080808080808080ull;
}

static inline uint64_t load64(const uint8_t *p)   /* little-endian on every target (layout.h) */
{
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

uint64_t toks_k2(const uint8_t *p, uint64_t i, uint64_t n, uint8_t hot)
{
    uint64_t a = 0x0101010101010101ull * (0x40u - (hot & 0x3Fu));
    if (n - i >= 8u) {                                       /* short cold gaps (a space between words) */
        uint64_t m = hot_bytes(load64(p + i), a);
        if (m != 0) { return i + ((uint64_t)__builtin_ctzll(m) >> 3); }
        i += 8u;
    }
    while (n - i >= 64u) {                                   /* bound: (len - pos) / 64 blocks */
        uint64_t m = 0;
        for (uint32_t k = 0; k < 8u; k++) {                  /* bound: 8 words per block */
            m |= hot_bytes(load64(p + i + 8u * k), a);
        }
        if (m != 0) { break; }
        i += 64u;
    }
    while (n - i >= 8u) {                                    /* bound: 8 words (the block that failed), or the tail */
        uint64_t m = hot_bytes(load64(p + i), a);
        if (m != 0) { return i + ((uint64_t)__builtin_ctzll(m) >> 3); }
        i += 8u;
    }
    while (i < n && p[i] < hot) { i++; }                     /* bound: 7 bytes */
    return i;
}

/* rationale: docs/notes/c-core.md §norm.c.2 */
#define BAD     0x00400000u
#define CLS(x)  (((x) >> TOKS_NFC_CLS_SHIFT) & TOKS_NFC_CLS_MASK)
#define DECOMP  (TOKS_NS_CANON | TOKS_NS_COMPAT | TOKS_NS_STRIP_MN)   /* the steps that decompose */
#define MAX_E   (TOKS_NFC_MAX_KDECOMP + 2u)                  /* an atom's chars: chinese padding + 18 */
#define NB_OF(f) (((f) & TOKS_NS_COMPAT) != 0u ? TOKS_NFC_NBK : TOKS_NFC_NB)

#define S_BASE  0xAC00u
#define L_BASE  0x1100u
#define V_BASE  0x1161u
#define T_BASE  0x11A7u
#define L_COUNT 19u
#define V_COUNT 21u
#define T_COUNT 28u
#define S_COUNT (L_COUNT * V_COUNT * T_COUNT)

/* rationale: docs/notes/c-core.md §norm.c.3 */
static uint32_t decomp(uint32_t f, uint32_t cp, uint32_t *e)
{
    uint32_t s = cp - S_BASE;
    if ((f & DECOMP) == 0u) {
        e[0] = cp;
        return 1;
    }
    uint32_t w = toks_nfc_info(cp), dl = (w >> TOKS_NFC_DLEN_SHIFT) & TOKS_NFC_DLEN_MASK, off = w & TOKS_NFC_DOFF_MASK;
    if ((f & TOKS_NS_COMPAT) != 0u && (w & TOKS_NFC_KX) != 0u) {   /* the compatibility decomposition instead */
        dl = toks_nfc_kinfo(cp) >> 16;
        off = toks_nfc_kinfo(cp) & 0xFFFFu;
    }
    if ((f & TOKS_NS_COMPOSE) == 0u && s < S_COUNT) {
        e[0] = L_BASE + s / (V_COUNT * T_COUNT);
        e[1] = V_BASE + s % (V_COUNT * T_COUNT) / T_COUNT;
        e[2] = T_BASE + s % T_COUNT;
        return s % T_COUNT == 0u ? 2u : 3u;
    }
    if (dl == 0) {
        e[0] = cp | (w & (TOKS_NFC_SECOND | (TOKS_NFC_CLS_MASK << TOKS_NFC_CLS_SHIFT)));
        return 1;
    }
    for (uint32_t j = 0; j < dl; j++) { e[j] = toks_nfc_pool[off + j]; }   /* bound: dl <= 18 */
    return dl;
}

/* rationale: docs/notes/c-core.md §norm.c.4 */
static uint32_t atom_chars(uint32_t f, const uint8_t *s, uint64_t i, uint64_t len, uint32_t *e, uint64_t *nb)
{
    uint32_t k = toks_utf8_len(s + i, len - i);
    *nb = k != 0u ? k : 1u;
    if (k == 0u) {
        e[0] = BAD | s[i];
        return 1;
    }
    uint32_t cp = k == 1u ? s[i] : toks_cp_decode(s + i, k);
    uint8_t bc = (f & (TOKS_NS_CLEAN | TOKS_NS_CJK)) != 0u ? toks_bert_cls(cp) : 0u;
    if ((f & TOKS_NS_CLEAN) != 0u && (bc & (TOKS_BC_REMOVE | TOKS_BC_WS)) != 0u) {
        e[0] = 0x20u;
        return (bc & TOKS_BC_REMOVE) != 0u ? 0u : 1u;
    }
    if ((f & TOKS_NS_CJK) == 0u || (bc & TOKS_BC_CJK) == 0u) { return decomp(f, cp, e); }
    e[0] = 0x20u;
    uint32_t n = decomp(f, cp, e + 1);
    e[n + 1u] = 0x20u;
    return n + 2u;
}

/* the primary composite of (a, b) when f composes, or 0. a is a starter or a composite, b any char. */
static uint32_t compose(uint32_t f, uint32_t a, uint32_t b)
{
    if ((f & TOKS_NS_COMPOSE) == 0u || ((a | b) & BAD) != 0 || (b & TOKS_NFC_SECOND) == 0) { return 0; }
    uint32_t ca = a & TOKS_NFC_CP_MASK, cb = b & TOKS_NFC_CP_MASK;
    if (ca - L_BASE < L_COUNT && cb - V_BASE < V_COUNT) {
        return S_BASE + ((ca - L_BASE) * V_COUNT + (cb - V_BASE)) * T_COUNT;
    }
    if (ca - S_BASE < S_COUNT && cb - (T_BASE + 1u) < T_COUNT - 1u && (ca - S_BASE) % T_COUNT == 0) {
        return ca + (cb - T_BASE);
    }
    uint64_t key = ((uint64_t)ca << 21) | cb;
    uint64_t h = (key * TOKS_NFC_FIB64) >> (64 - TOKS_NFC_COMP_LOG2);
    for (uint32_t p = 0; p < TOKS_NFC_COMP_MAXPROBE; p++) {  /* bound: the longest probe of a present pair */
        uint64_t slot = toks_nfc_comp[(h + p) & ((1u << TOKS_NFC_COMP_LOG2) - 1u)];
        if (slot == 0) { return 0; }
        if ((slot >> 21) == key) { return (uint32_t)(slot & 0x1FFFFFu); }
    }
    return 0;
}

/* the chars x (not BAD) becomes after the reordering: none (strip_accents drops a Mn), its lowercase (1 or 2
 * chars) or itself, into o; returns their count */
static uint32_t map_char(uint32_t f, uint32_t x, uint32_t *o)
{
    uint32_t c = x & TOKS_NFC_CP_MASK;
    uint8_t bc = (f & (TOKS_NS_STRIP_MN | TOKS_NS_LOWER)) != 0u ? toks_bert_cls(c) : 0u;
    if ((f & TOKS_NS_STRIP_MN) != 0u && (bc & TOKS_BC_MN) != 0u) { return 0; }
    if ((f & TOKS_NS_STRIP_M) != 0u && (toks_nfc_info(c) & TOKS_NFC_MARK) != 0u) { return 0; }
    if ((f & TOKS_NS_LOWER) == 0u || (bc & TOKS_BC_LOWER) == 0u) {
        o[0] = c;
        return 1;
    }
    const toks_bert_map *m = &toks_bert_maps[toks_bert_map_index(c)];
    for (uint32_t k = 0; k < m->low_len; k++) { o[k] = toks_bert_pool[(uint32_t)m->low_off + k]; }   /* bound: 2 */
    return m->low_len;
}

/* writes what the char x becomes at o, or only counts it (o NULL); returns its bytes */
static uint64_t put(uint32_t f, uint8_t *o, uint32_t x)
{
    uint8_t tmp[4];
    uint32_t m[TOKS_BERT_MAX_LOWER];
    if ((x & BAD) != 0) {
        if (o != NULL) { o[0] = (uint8_t)x; }
        return 1;
    }
    uint32_t n = map_char(f, x, m);
    uint64_t w = 0;
    for (uint32_t k = 0; k < n; k++) { w += toks_utf8_put(o != NULL ? o + w : tmp, m[k]); }   /* bound: 2 */
    return w;
}

/* ---- a cursor over the chars of text[a, b) under the steps f -------------------------------------------- */
typedef struct cur {
    uint64_t i;                         /* the current atom */
    uint64_t nb;                        /* its bytes */
    uint32_t n;                         /* its chars; 0: the cursor is at b */
    uint32_t j;                         /* the current one, e[j] */
    uint32_t f;
    uint32_t e[MAX_E];
} cur;

static void cur_at(cur *c, const uint8_t *s, uint64_t len, uint64_t i, uint64_t b)
{
    c->j = 0;
    c->n = 0;
    c->nb = 0;
    while (i < b) {                                          /* bound: b - i (clean_text's atoms give no chars) */
        c->n = atom_chars(c->f, s, i, len, c->e, &c->nb);
        if (c->n != 0u) { break; }
        i += c->nb;
    }
    c->i = i;
}

static inline void cur_next(cur *c, const uint8_t *s, uint64_t len, uint64_t b)
{
    if (++c->j < c->n) { return; }
    cur_at(c, s, len, c->i + c->nb, b);
}

/* the k-class mark number nth (0-based) of the non-starter run starting at r0, or 0 */
static uint32_t nth_of_class(const cur *r0, const uint8_t *s, uint64_t len, uint64_t b, uint32_t k, uint32_t nth)
{
    cur c = *r0;
    while (c.n != 0 && CLS(c.e[c.j]) != 0) {                 /* bound: the run's chars */
        if (CLS(c.e[c.j]) == k) {
            if (nth == 0) { return c.e[c.j]; }
            nth--;
        }
        cur_next(&c, s, len, b);
    }
    return 0;
}

/* rationale: docs/notes/c-core.md §norm.c.5 */
static uint64_t nrun(uint32_t f, const uint8_t *s, uint64_t len, uint64_t a, uint64_t b, uint8_t *out)
{
    uint64_t acc[64];                   /* per class: output bytes of its unabsorbed marks, then their offset */
    uint32_t first[64];                 /* per class: its first mark */
    uint8_t  cons[64];                  /* per class: marks absorbed by C (a prefix of the class) */
    uint64_t o = 0;
    uint32_t C = 0;                     /* the last starter (the composee), valid when have */
    int have = 0;
    cur c;
    c.f = f;
    cur_at(&c, s, len, a, b);
    while (c.n != 0) {                                       /* bound: the chars of [a, b), <= MAX_E per atom */
        uint32_t x = c.e[c.j];
        if (CLS(x) == 0) {
            uint32_t r = have ? compose(f, C, x) : 0u;
            if (r != 0) {
                C = r;
            } else {
                if (have) { o += put(f, out + o, C); }
                C = x;
                have = 1;
            }
            cur_next(&c, s, len, b);
            continue;
        }
        /* a run of non-starters: its extent and class census */
        cur r0 = c;
        uint64_t mask = 0, m = 0, ab = 0;
        while (c.n != 0 && CLS(c.e[c.j]) != 0) {             /* bound: the chars of [a, b) */
            uint32_t y = c.e[c.j], k = CLS(y);
            if (((mask >> k) & 1u) == 0) { mask |= 1ull << k; acc[k] = 0; first[k] = y; cons[k] = 0; }
            acc[k] += put(f, NULL, y);
            m++;
            cur_next(&c, s, len, b);
        }
        if (m == 1) {
            uint32_t r = have ? compose(f, C, x) : 0u;
            if (r != 0) { C = r; continue; }
            if (have) { o += put(f, out + o, C); have = 0; }
            o += put(f, out + o, x);
            continue;
        }
        if (have) {
            for (uint64_t mm = mask; mm != 0; mm &= mm - 1u) {   /* bound: TOKS_NFC_N_CLS classes */
                uint32_t k = (uint32_t)__builtin_ctzll(mm);
                uint32_t y = first[k];
                for (uint32_t t = 0; t < TOKS_NFC_MAX_DECOMP && y != 0; t++) {   /* bound: <= 3 absorptions */
                    uint32_t r = compose(f, C, y);
                    if (r == 0) { break; }
                    C = r;
                    cons[k]++;
                    ab++;
                    acc[k] -= put(f, NULL, y);
                    y = nth_of_class(&r0, s, len, b, k, cons[k]);
                }
            }
        }
        if (ab == m) { continue; }                           /* every mark absorbed: C stays open */
        if (have) { o += put(f, out + o, C); have = 0; }     /* blocked from here on: C is final */
        for (uint64_t mm = mask; mm != 0; mm &= mm - 1u) {   /* bound: TOKS_NFC_N_CLS classes */
            uint32_t k = (uint32_t)__builtin_ctzll(mm);
            uint64_t nk = acc[k];
            acc[k] = o;
            o += nk;
        }
        c = r0;
        for (uint64_t i = 0; i < m; i++) {                   /* bound: m, the run's chars */
            uint32_t y = c.e[c.j], k = CLS(y);
            if (cons[k] != 0) { cons[k]--; } else { acc[k] += put(f, out + acc[k], y); }
            cur_next(&c, s, len, b);
        }
    }
    if (have) { o += put(f, out + o, C); }
    return o;
}

int64_t toks_norm(uint32_t f, const uint8_t *text, uint64_t len, uint8_t *out, uint64_t cap)
{
    if (cap < TOKS_NORM_BOUND(f, len)) { return TOKS_E_CAP; }
    if ((f & ~TOKS_NS_COMPAT) != TOKS_NS_NFC) { return (int64_t)nrun(f, text, len, 0u, len, out); }
    uint64_t pos = 0, o = 0, e;
    while (pos < len) {                                      /* bound: one changed run per pass */
        uint64_t s = toks_nfc_scan(f, text, len, pos, &e);   /* text[pos, s) is in the form already */
        memcpy(out + o, text + pos, s - pos);
        o += s - pos;
        if (s == len) { break; }
        o += nrun(f, text, len, s, e, out + o);
        pos = e;
    }
    return (int64_t)o;
}

uint32_t toks_norm_char(uint32_t f, uint32_t cp, uint32_t o[TOKS_NORM_MAX_OUT])
{
    uint32_t e[MAX_E], m[TOKS_BERT_MAX_LOWER], n = 0, nd = decomp(f, cp, e);
    for (uint32_t i = 0; i < nd; i++) {                      /* bound: 4 */
        uint32_t k = map_char(f, e[i], m);
        if (k == 0u && CLS(e[i]) == 0u) { o[n++] = TOKS_NORM_GHOST; }   /* a dropped starter still ends a run */
        for (uint32_t j = 0; j < k; j++) { o[n++] = (e[i] & (TOKS_NFC_CLS_MASK << TOKS_NFC_CLS_SHIFT)) | m[j]; }
    }
    return n;
}

/* ---- the read-only scan ---------------------------------------------------------------------------------- */

/* rationale: docs/notes/c-core.md §norm.c.6 */
static int run_is_nfc(uint32_t f, const uint8_t *s, uint64_t len, uint64_t a, uint64_t b)
{
    uint32_t last = 0, dec = TOKS_NFC_SECOND | (TOKS_NFC_DLEN_MASK << TOKS_NFC_DLEN_SHIFT) | ((f & TOKS_NS_COMPAT) != 0u ?
                    TOKS_NFC_KX : 0u);
    uint64_t i = a;
    while (i < b) {                                          /* bound: b - a bytes */
        uint32_t k = toks_utf8_len(s + i, len - i);
        if (k == 0) {                                        /* an invalid byte: only as the first atom */
            if (i != a) { return 0; }
            i++;
            continue;
        }
        uint32_t w = toks_nfc_info(k == 1 ? s[i] : toks_cp_decode(s + i, k));
        uint32_t c = (w >> TOKS_NFC_CLS_SHIFT) & TOKS_NFC_CLS_MASK;
        if (i == a && (w & NB_OF(f)) == 0) { i += k; continue; }       /* the run's boundary atom */
        if ((w & dec) != 0 || c < last) {
            return 0;
        }
        last = c;
        i += k;
    }
    return 1;
}

/* 1 when the form f (NFC, NFKC) changes the run text[a, b) (a a boundary atom, b the next boundary or len) */
static int run_changes(uint32_t f, const uint8_t *s, uint64_t len, uint64_t a, uint64_t b)
{
    if (run_is_nfc(f, s, len, a, b)) { return 0; }
    if (b - a > TOKS_NFC_SCAN_RUN) { return 1; }             /* conservative (norm.h) */
    uint8_t buf[TOKS_NFKC_X * TOKS_NFC_SCAN_RUN];            /* >= TOKS_NORM_BOUND(f, b - a) */
    uint64_t n = nrun(f, s, len, a, b, buf);
    return n != b - a || memcmp(buf, s + a, (size_t)n) != 0;
}

/* the start of the last atom of text[p, h), p an atom start, every byte < 0xCC: such atoms are one byte, or a
 * lead C2..CB with one continuation (a lead byte is never inside an earlier atom) */
static inline uint64_t last_atom(const uint8_t *s, uint64_t p, uint64_t h)
{
    if (h - p >= 2u && s[h - 2] >= 0xC2u && s[h - 1] >= 0x80u && s[h - 1] < 0xC0u) { return h - 2u; }
    return h - 1u;
}

/* toks_nfc_scan's fast stretch: from the atom start p, in a run with nothing to check, every atom the form leaves
 * alone in a row: hot atoms whose bit in toks_nfc_fast (norm_nfc.h: NB / NBK clear, the per-atom loop's own test,
 * looked up per code point) is set, and cold gaps up to a hot byte within the next 8 bytes (a longer cold stretch
 * is K2's). *rs follows the last atom's start, as the per-atom loop would set it. Returns where it stopped: a cold
 * stretch of >= 8 bytes, an invalid or 4-byte atom, an atom without its bit, or len. */
static inline uint64_t clean_run(uint32_t f, const uint8_t *text, uint64_t len, uint64_t p, uint8_t hot, uint64_t *rs)
{
    uint64_t a = 0x0101010101010101ull * (0x40u - (hot & 0x3Fu)), last = *rs;
    const uint64_t *fast = toks_nfc_fast + ((f & TOKS_NS_COMPAT) != 0u ? 1024u : 0u);
    while (p < len) {                                        /* bound: p rises by >= 1 per iteration */
        uint32_t b0 = text[p];
        if (b0 < hot) {
            if (len - p < 8u) { break; }
            uint64_t m = hot_bytes(load64(text + p), a);
            if (m == 0u) { break; }
            uint64_t h = p + ((uint64_t)__builtin_ctzll(m) >> 3);
            last = last_atom(text, p, h);
            p = h;
            continue;
        }
        if (b0 >= 0xF0u || len - p < 3u) { break; }
        uint32_t b1 = text[p + 1u], b2 = text[p + 2u], cp, k;
        if (b0 < 0xE0u) {
            if ((b1 & 0xC0u) != 0x80u) { break; }
            cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            k = 2u;
        } else {
            if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u) { break; }
            cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            if (cp < 0x800u) { break; }                      /* overlong (E0 80..9F): invalid, the general path */
            k = 3u;
        }
        if (((fast[cp >> 6] >> (cp & 63u)) & 1u) == 0u) { break; }   /* surrogates have no bit */
        last = p;
        p += k;
    }
    *rs = last;
    return p;
}

/* toks_nfc_scan: K2 skips the cold bytes, every hot atom is classified; a dirty run is tested (run_changes) */
uint64_t toks_nfc_scan(uint32_t f, const uint8_t *text, uint64_t len, uint64_t pos, uint64_t *run_end)
{
    uint64_t rs = pos, dirty = 0;
    uint8_t hot = (f & TOKS_NS_COMPAT) != 0u ? TOKS_NFKC_HOT_BYTE : TOKS_NFC_HOT_BYTE;
    while (pos < len) {                                      /* bound: pos rises by >= 1 per iteration */
        uint64_t h = text[pos] >= hot ? pos : toks_k2(text, pos, len, hot);
        if (h > pos) {                                       /* [pos, h): boundaries the form leaves unchanged */
            if (dirty != 0 && run_changes(f, text, len, rs, pos)) { *run_end = pos; return rs; }
            dirty = 0;
            rs = last_atom(text, pos, h);
            pos = h;
            if (pos == len) { break; }
        }
        if (dirty == 0) {
            uint64_t q = clean_run(f, text, len, pos, hot, &rs);
            if (q != pos) {
                pos = q;
                continue;
            }
        }
        uint32_t k = toks_utf8_len(text + pos, len - pos);
        uint32_t w = k != 0 ? toks_nfc_info(toks_cp_decode(text + pos, k)) : 0u;
        if ((w & NB_OF(f)) == 0) {                           /* a boundary (or an invalid byte) */
            if (dirty != 0 && run_changes(f, text, len, rs, pos)) { *run_end = pos; return rs; }
            dirty = 0;
            rs = pos;
        } else {
            dirty = 1;
        }
        pos += k != 0 ? k : 1u;
    }
    if (dirty != 0 && run_changes(f, text, len, rs, len)) { *run_end = len; return rs; }
    *run_end = len;
    return len;
}

int toks_nfc_boundary(uint32_t f, const uint8_t *text, uint64_t len, uint64_t i)
{
    uint32_t k = toks_utf8_len(text + i, len - i);
    /* rationale: docs/notes/c-core.md §norm.c.7 */
    if (k <= 1u) { return 1; }
    return (toks_nfc_info(toks_cp_decode(text + i, k)) & NB_OF(f)) == 0;
}

/* rationale: docs/notes/c-core.md §norm.c.8 */

/* the base class of the atom at g[i] (i < n, an atom start) and its length */
static uint32_t atom_base(const toks_tables *t, const uint8_t *g, uint64_t n, uint64_t i, uint32_t *k)
{
    if (g[i] < 0x80u) {
        *k = 1u;
        return t->cls_ascii[g[i]] & TOKS_C_BASE_MASK;
    }
    uint32_t m = toks_utf8_len(g + i, n - i);
    *k = m != 0u ? m : 1u;
    return m != 0u ? (uint32_t)(toks_cls_cp(t, toks_cp_decode(g + i, m)) & TOKS_C_BASE_MASK) : TOKS_C_P;
}

/* (a) for the atom y at p < n: base(y) + 1 when p qualifies, else 0; at_change: p starts a changed run. A byte
 * >= 0xC0 always starts an atom; a continuation byte is passed over (it starts one only when invalid). */
static uint32_t y_base(const toks_tables *t, uint32_t f, const uint8_t *g, uint64_t n, uint64_t p, int at_change)
{
    uint8_t c = g[p];
    if (c < 0x80u) {
        uint32_t b = t->cls_ascii[c] & TOKS_C_BASE_MASK;
        return (at_change == 0 || b == TOKS_C_WS || b == TOKS_C_NL) ? b + 1u : 0u;
    }
    if (at_change != 0 || c < 0xC0u || toks_nfc_boundary(f, g, n, p) == 0) { return 0u; }
    uint32_t k;
    return atom_base(t, g, n, p, &k) + 1u;
}

/* (b) */
static int restart_pair(uint32_t bx, uint32_t by)
{
    if (bx == TOKS_C_L) { return by != TOKS_C_L; }
    if (bx == TOKS_C_N) { return by != TOKS_C_N; }
    if (bx == TOKS_C_P) { return by == TOKS_C_WS || by == TOKS_C_N; }
    return 0;                                                /* WS, NL */
}

/* rationale: docs/notes/c-core.md §norm.c.9 */
static uint64_t atom_before(const uint8_t *g, uint64_t n, uint64_t lo, uint64_t p)
{
    if (g[p - 1u] < 0x80u) { return p - 1u; }
    for (uint64_t k = 2u; k <= 4u && k <= p - lo; k++) {     /* bound: 3 */
        if ((uint64_t)toks_utf8_len(g + p - k, n - (p - k)) == k) { return p - k; }
    }
    return p - 1u;
}

/* the last restart point in (lo, hi], hi the first changed run after lo; lo when there is none */
static uint64_t restart_before(const toks_tables *t, uint32_t f, const uint8_t *g, uint64_t n, uint64_t lo, uint64_t hi)
{
    if (t->tmpl != TOKS_TMPL_CL100K) { return lo; }
    for (uint64_t p = hi; p > lo; p--) {                     /* bound: hi - lo */
        uint32_t by = y_base(t, f, g, n, p, p == hi), k;
        if (by != 0u && restart_pair(atom_base(t, g, n, atom_before(g, n, lo, p), &k), by - 1u)) { return p; }
    }
    return lo;
}

/* the first restart point in (lo, hi] with x at or after lo, lo the end of a changed run and hi the start of
 * the next one (n: none); n when the walk reaches the end; 0 when there is none up to hi < n */
static uint64_t restart_after(const toks_tables *t, uint32_t f, const uint8_t *g, uint64_t n, uint64_t lo, uint64_t hi)
{
    if (t->tmpl != TOKS_TMPL_CL100K) { return hi == n ? n : 0u; }
    uint64_t q = lo;
    while (q < hi) {                                         /* bound: hi - lo bytes, >= 1 per atom */
        uint32_t k, bx = atom_base(t, g, n, q, &k);
        uint64_t p = q + k;
        if (p >= n) { return n; }
        uint32_t by = y_base(t, f, g, n, p, p == hi);
        if (by != 0u && restart_pair(bx, by - 1u)) { return p; }
        q = p;
    }
    return hi == n ? n : 0u;
}

void toks_nfc_plan_begin(toks_nfc_plan *p, uint32_t f, const uint8_t *g, uint64_t n)
{
    p->r = 0u;
    p->d1 = n;
    p->f = f;
    p->d0 = toks_nfc_scan(f, g, n, 0u, &p->d1);
}

int toks_nfc_plan_next(toks_nfc_plan *p, const toks_tables *t, const uint8_t *g, uint64_t n, uint64_t *s,
                       uint64_t *e)
{
    if (p->d0 >= n) { return 0; }
    uint64_t d1 = p->d1, e0, e1, end;
    *s = restart_before(t, p->f, g, n, p->r, p->d0);
    for (;;) {                                               /* bound: each pass takes one more changed run */
        e0 = toks_nfc_scan(p->f, g, n, d1, &e1);             /* the next changed run (e0 = n: none) */
        end = restart_after(t, p->f, g, n, d1, e0);
        if (end != 0u) { break; }
        d1 = e1;
    }
    *e = end;
    p->r = end;
    p->d0 = e0;
    p->d1 = e1;
    return 1;
}
