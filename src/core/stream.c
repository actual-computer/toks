/* stream.c: stream decode (SPEC §3.4, §4.7) for hf's ByteLevel decoder (docs/notes/c-core.md §stream.c.1) */
#include "core.h"
#include "spm.h"

/* the opaque words of toks_stream as the stream keeps them (punned through memcpy only). */
#define SST_RUN 44u                  /* spm: the longest byte-fallback run st itself holds */
typedef struct sst {
    uint64_t tag;            /* SST_MAGIC ^ ctx->identity: set up by toks_stream_init for that tokenizer */
    uint32_t flags;          /* 0 or TOKS_SKIP_SPECIAL */
    uint32_t strip_left;     /* spm: Strip's leading chars still to drop */
    uint8_t  np;             /* held bytes: ByteLevel 0..3 (a utf-8 prefix), spm 0..SST_RUN (a byte run); 0 with
                                SST_HOLD */
    uint8_t  mode;           /* spm: SST_FIRST | SST_VALID | SST_INVALID | SST_HOLD */
    uint8_t  rsv[2];         /* 0 */
    uint8_t  buf[SST_RUN];   /* the held bytes: buf[0, np); with SST_HOLD, the hold's record (sst_x) */
} sst;

#define SST_FIRST   1u       /* spm: a string has been kept since init / flush (index > 0) */
#define SST_VALID   2u       /* spm: a byte run is open and still valid utf-8 so far (held) */
#define SST_INVALID 4u       /* spm: a byte run is open and invalid (each of its bytes is one U+FFFD) */
#define SST_HOLD    8u       /* spm: the held run lives in the caller's memory (toks_stream_hold), not in buf */

_Static_assert(sizeof(sst) == sizeof(toks_stream), "the stream state is toks_stream");

/* a caller's hold (toks_stream_hold), kept in buf with SST_HOLD: the run is p[0, n) of p[0, cap) */
typedef struct sst_x {
    uint8_t *p;
    uint64_t cap, n;
    uint64_t chk;            /* sst_chk: p and cap as toks_stream_hold set them (a stray write's check, not a secret) */
    uint8_t  need, lo, hi;   /* the open run's utf-8 state (u8_state of p[0, n)); 0 0 0 with no valid run open */
} sst_x;
_Static_assert(sizeof(sst_x) <= SST_RUN, "the hold's record fits buf");

static inline sst_x sst_x_get(const sst *s) { sst_x x; memcpy(&x, s->buf, sizeof x); return x; }
static inline void sst_x_put(sst *s, const sst_x *x) { memcpy(s->buf, x, sizeof *x); }
static inline uint64_t sst_chk(uint64_t tag, const sst_x *x)
{
    return toks_mix64(toks_mix64(tag, (uint64_t)(uintptr_t)x->p), x->cap);
}
/* the held run: its bytes, their count, the most it may hold */
static inline const uint8_t *sst_bytes(const sst *s) { return (s->mode & SST_HOLD) != 0u ? sst_x_get(s).p : s->buf; }
static inline uint64_t sst_held(const sst *s) { return (s->mode & SST_HOLD) != 0u ? sst_x_get(s).n : s->np; }
static inline uint64_t sst_cap(const sst *s) { return (s->mode & SST_HOLD) != 0u ? sst_x_get(s).cap : SST_RUN; }
static inline void sst_set_held(sst *s, uint64_t n)
{
    if ((s->mode & SST_HOLD) == 0u) { s->np = (uint8_t)n; return; }
    sst_x x = sst_x_get(s);
    x.n = n;
    sst_x_put(s, &x);
}

#define SST_MAGIC 0x314D5254534B4F54ull                                  /* "TOKSTRM1" */

/* dec_len: 0..16 = well-formed alone and in its zero-padded slot; DL_LONG = well-formed alone, longer,
 * and every 16-byte load covering it inside tok_bytes; DL_SLOW = the byte-wise path. */
#define DL_LONG 0xFEu
#define DL_SLOW 0xFFu

/* decode's block: n_ids 16-byte slots, then n_ids length bytes */
static uint64_t dec_block_bytes(uint32_t n_ids) { return 17u * (uint64_t)n_ids; }

static const uint8_t FFFD[3] = { 0xEFu, 0xBFu, 0xBDu };

/* ---- the writer ------------------------------------------------------------------------------------------ */

/* d[0, k) = s[0, k) with fixed-size moves only (no libc call): overlapping words for the tail. */
static void copy(uint8_t *d, const uint8_t *s, uint64_t k)
{
    if (k >= 8u) {
        uint64_t v, w;
        for (uint64_t i = 0u; i + 8u < k; i += 8u) {     /* bound: k / 8 */
            memcpy(&v, s + i, 8);
            memcpy(d + i, &v, 8);
        }
        memcpy(&w, s + k - 8u, 8);
        memcpy(d + k - 8u, &w, 8);
    } else if (k >= 4u) {
        uint32_t v, w;
        memcpy(&v, s, 4);
        memcpy(&w, s + k - 4u, 4);
        memcpy(d, &v, 4);
        memcpy(d + k - 4u, &w, 4);
    } else if (k >= 2u) {
        uint16_t v, w;
        memcpy(&v, s, 2);
        memcpy(&w, s + k - 2u, 2);
        memcpy(d, &v, 2);
        memcpy(d + k - 2u, &w, 2);
    } else if (k == 1u) {
        d[0] = s[0];
    }
}

/* a caller's id: its array needs no alignment (toks.h), so it is read through memcpy (one load) */
static inline uint32_t ld32(const void *p) { return toks_ld32(p); }   /* a void * parameter: no alignment assumed */

/* 16 bytes from s to d: one vector load and store where the target has them. */
static void copy16(uint8_t *d, const uint8_t *s)
{
    uint8_t v[16];
    memcpy(v, s, 16);
    memcpy(d, v, 16);
}

/* d[0, k) = s[0, k) as whole 16-byte moves: d has room for them and s may be read that far (DL_LONG). */
static void move16s(uint8_t *d, const uint8_t *s, uint64_t k)
{
    for (uint64_t j = 0u; j < k; j += 16u) { copy16(d + j, s + j); }     /* bound: k / 16 + 1 */
}

/* the whole 16-byte moves k bytes take */
static uint64_t round16(uint64_t k) { return (k + 15u) & ~(uint64_t)15u; }

/* k bytes of output: counted always, stored while they fit in out[0, cap). */
static void put(toks_lossy *d, const uint8_t *p, uint64_t k)
{
    uint64_t n = d->n;
    d->n = n + k;
    if (n < d->cap) { copy(d->out + n, p, (d->cap - n < k) ? d->cap - n : k); }
}

/* unicode §3.9 table 3-7 for a lead byte c (0xC2..0xF4): its sequence length and the range of the
 * byte after it (every later byte is 0x80..0xBF). */
static uint32_t seq_need(uint8_t c) { return (c < 0xE0u) ? 2u : (c < 0xF0u) ? 3u : 4u; }
static uint8_t  seq_lo(uint8_t c)   { return (c == 0xE0u) ? 0xA0u : (c == 0xF0u) ? 0x90u : 0x80u; }
static uint8_t  seq_hi(uint8_t c)   { return (c == 0xEDu) ? 0x9Fu : (c == 0xF4u) ? 0x8Fu : 0xBFu; }

/* the bytes p[0, k) of one token: the general path (toks_lossy_ids takes dec_len's ids first). Inline,
 * so the caller's cursor (a local) stays in registers. */
static inline __attribute__((always_inline)) void lossy_bytes(toks_lossy *d, const uint8_t *p, uint64_t k)
{
    uint64_t j = 0u;
    while (j < k) {             /* bound: 2k (a pass consumes a byte, or drops the held prefix and the
                                   next pass consumes one) */
        if (d->np == 0u) {
            uint64_t s = j;
            while (j < k) {                              /* bound: k - j (the well-formed run from s) */
                if (p[j] < 0x80u) { j++; continue; }
                uint32_t w = toks_utf8_len(p + j, k - j);
                if (w == 0u) { break; }
                j += w;
            }
            if (j > s) { put(d, p + s, j - s); }
            if (j == k) { break; }
            uint8_t c = p[j];
            j++;
            if (c < 0xC2u || c > 0xF4u) { put(d, FFFD, 3u); continue; }    /* never a lead */
            d->pend[0] = c;            /* a lead whose sequence breaks or runs past the token: hold */
            d->np = 1u;
            continue;
        }
        uint8_t c = p[j], c0 = d->pend[0];
        uint8_t lo = (d->np == 1u) ? seq_lo(c0) : 0x80u;
        uint8_t hi = (d->np == 1u) ? seq_hi(c0) : 0xBFu;
        if (c < lo || c > hi) {                          /* the held prefix is a maximal subpart */
            put(d, FFFD, 3u);
            d->np = 0u;
            continue;                                    /* c starts afresh */
        }
        d->pend[d->np] = c;
        d->np++;
        j++;
        if (d->np == seq_need(c0)) { put(d, d->pend, d->np); d->np = 0u; }
    }
}

int64_t toks_lossy_ids(const toks_ctx *ctx, toks_lossy *d, const uint32_t *ids, uint64_t n, uint32_t flags)
{
    const uint32_t *off = ctx->t.tok_off;
    const uint8_t *tb = ctx->t.tok_bytes;
    const uint8_t *slot = ctx->dec_slot, *dlen = ctx->dec_len;
    const uint32_t *spec = ((flags & TOKS_SKIP_SPECIAL) != 0u) ? ctx->special_ids : NULL;
    uint32_t nid = ctx->t.n_ids;
    /* a local cursor: out is bytes, so a store through it could alias *d, but not a local whose address
     * stays here (lossy_bytes is inlined) */
    toks_lossy c = *d;
    int64_t r = 0;
    uint64_t i = 0u;
    while (i < n) {                                      /* bound: n (a pass consumes >= 1 id) */
        if (c.np == 0u && dlen != NULL && c.cap >= 16u) {
            /* the run that is most of a decode: nothing held, room for a whole 16-byte move, a short id
             * that is its own output: one move from its slot, the exact length forward */
            uint64_t lim = c.cap - 16u;
            while (i < n && c.n <= lim) {                /* bound: n - i */
                uint32_t id = ld32(ids + i);
                if (id >= nid || dlen[id] > 16u) { break; }
                i++;
                if (spec != NULL && toks_bit(spec, id) != 0u) { continue; }
                copy16(c.out + c.n, slot + 16u * (uint64_t)id);
                c.n += dlen[id];
            }
            if (i == n) { break; }
        }
        uint32_t id = ld32(ids + i);
        i++;
        if (id >= nid) { r = TOKS_E_ID; break; }
        if (spec != NULL && toks_bit(spec, id) != 0u) { continue; }
        uint32_t l = (dlen != NULL) ? dlen[id] : DL_SLOW;
        uint64_t room = (c.n < c.cap) ? c.cap - c.n : 0u;
        if (c.np == 0u && l <= 16u) {
            /* most tokens: well-formed alone, so the bytes are the output and nothing is held. One 16-byte
             * move from the slot while out has room for it, else an exact copy of what fits. */
            const uint8_t *q = slot + 16u * (uint64_t)id;
            if (room >= 16u) { copy16(c.out + c.n, q); }
            else if (room != 0u) { copy(c.out + c.n, q, (room < l) ? room : l); }
            c.n += l;
            continue;
        }
        const uint8_t *p = tb + off[id];
        uint64_t k = (uint64_t)(off[id + 1u] - off[id]);
        if (c.np == 0u && l == DL_LONG) {                /* the same from tok_bytes, whole 16-byte moves */
            if (room >= round16(k)) { move16s(c.out + c.n, p, k); }
            else if (room != 0u) { copy(c.out + c.n, p, (room < k) ? room : k); }
            c.n += k;
            continue;
        }
        lossy_bytes(&c, p, k);
    }
    *d = c;
    return r;
}

void toks_lossy_end(toks_lossy *d)
{
    if (d->np != 0u) { put(d, FFFD, 3u); d->np = 0u; }
}

/* rationale: docs/notes/c-core.md §stream.c.2 */
static const toks_spm_op *spm_ops(const toks_dchain *s, int *bf, const toks_spm_op **strip)
{
    const toks_spm_op *per_token = NULL;
    *bf = 0;
    *strip = NULL;
    for (uint32_t i = 0u; i < s->n_dec && i < TOKS_SPM_MAX_OPS; i++) {     /* bound: TOKS_SPM_MAX_OPS */
        const toks_spm_op *op = &s->dec[i];
        if (op->kind == TOKS_SPM_D_REPLACE || op->kind == TOKS_SPM_D_METASPACE) { per_token = op; }
        else if (op->kind == TOKS_SPM_D_BYTE_FALLBACK) { *bf = 1; }
        else if (op->kind == TOKS_SPM_D_STRIP) { *strip = op; }
    }
    return per_token;
}

int64_t toks_dec_build(toks_ctx *c)
{
    const toks_tables *t = &c->t;
    c->dec_max = 0u;
    c->dec_slot = NULL;
    c->dec_len = NULL;
    if (t->n_ids == 0u) { return 0; }
    if (c->wp != NULL) {                            /* WordPiece: a token's bytes + the joining " " at most */
        uint64_t best = 0u;
        for (uint32_t id = 0u; id < t->n_ids; id++) {   /* bound: n_ids */
            uint64_t k = (uint64_t)t->tok_off[id + 1u] - t->tok_off[id];
            if (k > best) { best = k; }
        }
        c->dec_max = (uint32_t)(best + 1u);
        return 0;
    }
    if (c->dc.on) {
        /* rationale: docs/notes/c-core.md §stream.c.3 */
        int bf;
        const toks_spm_op *strip;
        const toks_spm_op *op = spm_ops(&c->dc, &bf, &strip);
        uint64_t mult = (op != NULL && op->kind == TOKS_SPM_D_REPLACE && op->b.n > 1u) ? op->b.n : 1u;
        uint64_t best = 0u;
        for (uint32_t id = 0u; id < t->n_ids; id++) {   /* bound: n_ids */
            uint64_t o = t->tok_off[id], k = (uint64_t)t->tok_off[id + 1u] - o;
            uint64_t w = (bf && toks_byte_token(t->tok_bytes + o, k) >= 0) ? 3u
                       : k * mult + (c->dc.has_decoder ? 0u : 1u);
            if (w > best) { best = w; }
        }
        c->dec_max = (uint32_t)best;  /* <= 16 x TOKS_MAX_TOKEN_BYTES + 1 */
        return 0;
    }
    if (c->dec_byte_level == 0u) { return 0; }      /* no decoder the stream knows: it says unsupported */
    uint64_t bytes = dec_block_bytes(t->n_ids);
    uint8_t *blk = toks_plat_arena(bytes);
    if (blk == NULL) { return TOKS_E_NOMEM; }
    memset(blk, 0, (size_t)bytes);
    uint8_t *len = blk + 16u * (uint64_t)t->n_ids;
    uint64_t end = t->tok_off[t->n_ids], best = 0u;
    for (uint32_t id = 0u; id < t->n_ids; id++) {       /* bound: n_ids (one pass over tok_bytes) */
        uint64_t o = t->tok_off[id], k = (uint64_t)t->tok_off[id + 1u] - o;
        const uint8_t *p = t->tok_bytes + o;
        toks_lossy d;
        memset(&d, 0, sizeof d);
        lossy_bytes(&d, p, k);
        toks_lossy_end(&d);
        if (d.n > best) { best = d.n; }
        len[id] = (uint8_t)DL_SLOW;
        if (toks_utf8_valid(p, k)) {
            if (k <= 16u) { copy(blk + 16u * (uint64_t)id, p, k); len[id] = (uint8_t)k; }
            else if (o + round16(k) <= end) { len[id] = (uint8_t)DL_LONG; }
        }
    }
    c->dec_max = (uint32_t)best;      /* <= 3 * TOKS_MAX_TOKEN_BYTES: one U+FFFD per byte at most */
    c->dec_slot = blk;
    c->dec_len = len;
    return 0;
}

void toks_dec_free(toks_ctx *c)
{
    if (c->dec_slot != NULL) { toks_plat_arena_free(c->dec_slot, dec_block_bytes(c->t.n_ids)); }
    c->dec_slot = NULL;
    c->dec_len = NULL;
}

/* ---- the stream ------------------------------------------------------------------------------------------- */

/* rationale: docs/notes/c-core.md §stream.c.4 */
static int u8_state(const uint8_t *b, uint32_t n, uint32_t *need, uint8_t *lo, uint8_t *hi)
{
    toks_u8 v = { 0u, 0x80u, 0xBFu };
    for (uint32_t i = 0u; i < n; i++) {                  /* bound: n <= SST_RUN */
        if (!toks_u8_feed(&v, b[i])) { return 0; }
    }
    *need = v.need;
    *lo = v.lo;
    *hi = v.hi;
    return 1;
}

/* st's state when toks_stream_init set it up for ctx and it is intact, else 0. A corrupted state never
 * reaches a writer: they rely on np and buf being what the decoder can hold. */
static int sst_get(const toks_ctx *ctx, const toks_stream *st, sst *s)
{
    memcpy(s, st, sizeof *s);
    if (s->tag != (SST_MAGIC ^ ctx->identity) || (s->flags & ~TOKS_SKIP_SPECIAL) != 0u || s->rsv[0] != 0u ||
        s->rsv[1] != 0u) {
        return 0;
    }
    uint32_t need;
    uint8_t lo, hi;
    if (ctx->wp != NULL) { return (s->mode & ~SST_FIRST) == 0u && s->np == 0u && s->strip_left == 0u; }   /* WordPiece */
    if (!ctx->dc.on) {                              /* ByteLevel: a proper prefix of one sequence */
        if (s->mode != 0u || s->strip_left != 0u || s->np > 3u) { return 0; }
        return s->np == 0u || ((s->buf[0] >= 0xC2u && s->buf[0] <= 0xF4u) && u8_state(s->buf, s->np, &need, &lo, &hi) &&
                               need != 0u);
    }
    int bf;
    const toks_spm_op *strip;
    (void)spm_ops(&ctx->dc, &bf, &strip);
    uint64_t n = sst_held(s);
    if ((s->mode & ~(SST_FIRST | SST_VALID | SST_INVALID | SST_HOLD)) != 0u || s->np > SST_RUN ||
        (s->mode & (SST_VALID | SST_INVALID)) == (SST_VALID | SST_INVALID) ||
        ((s->mode & SST_VALID) == 0u && n != 0u) || ((s->mode & SST_VALID) != 0u && n == 0u) ||
        s->strip_left > (strip != NULL ? strip->start : 0u)) {
        return 0;
    }
    if ((s->mode & SST_HOLD) == 0u) { return (s->mode & SST_VALID) == 0u || u8_state(s->buf, s->np, &need, &lo, &hi); }
    /* a caller's hold: its record, O(1) (the held bytes are the caller's memory, not read again here) */
    sst_x x = sst_x_get(s);
    for (uint32_t i = (uint32_t)sizeof x; i < SST_RUN; i++) { if (s->buf[i] != 0u) { return 0; } }   /* bound: 44 */
    if (s->np != 0u || x.p == NULL || x.cap == 0u || x.n > x.cap || x.chk != sst_chk(s->tag, &x)) { return 0; }
    if ((s->mode & SST_VALID) == 0u) { return x.need == 0u && x.lo == 0u && x.hi == 0u; }
    return x.need <= 3u && 0x80u <= x.lo && x.lo <= x.hi && x.hi <= 0xBFu;
}

/* the open run's utf-8 state: kept in the record with a caller's hold, else read again from buf (<= SST_RUN bytes) */
static void sst_u8(const sst *s, uint32_t *need, uint8_t *lo, uint8_t *hi)
{
    *need = 0u;
    *lo = 0x80u;
    *hi = 0xBFu;
    if ((s->mode & SST_VALID) == 0u) { return; }
    if ((s->mode & SST_HOLD) != 0u) {
        sst_x x = sst_x_get(s);
        *need = x.need, *lo = x.lo, *hi = x.hi;
        return;
    }
    (void)u8_state(s->buf, s->np, need, lo, hi);
}

/* the state toks_stream_init left, with a caller's hold (toks_stream_hold) kept */
static void sst_reset(const toks_ctx *ctx, sst *s)
{
    uint64_t tag = s->tag;
    uint32_t flags = s->flags;
    int hold = (s->mode & SST_HOLD) != 0u;
    sst_x x = sst_x_get(s);
    memset(s, 0, sizeof *s);
    s->tag = tag;
    s->flags = flags;
    if (hold) {
        s->mode = (uint8_t)SST_HOLD;
        x.n = 0u, x.need = 0u, x.lo = 0u, x.hi = 0u;
        sst_x_put(s, &x);
    }
    if (ctx->dc.on) {
        int bf;
        const toks_spm_op *strip;
        (void)spm_ops(&ctx->dc, &bf, &strip);
        s->strip_left = (strip != NULL) ? strip->start : 0u;
    }
}

void toks_stream_init(const toks_ctx *ctx, toks_stream *st, uint32_t flags)
{
    if (st == NULL) { return; }
    memset(st, 0, sizeof *st);           /* not initialized: push and flush return TOKS_E_ARG */
    if (ctx == NULL || (flags & ~TOKS_SKIP_SPECIAL) != 0u) { return; }
    sst s;
    memset(&s, 0, sizeof s);
    s.tag = SST_MAGIC ^ ctx->identity;
    s.flags = flags;
    sst_reset(ctx, &s);
    memcpy(st, &s, sizeof s);
}

/* rationale: docs/notes/c-core.md §stream.c.6 */
int64_t toks_stream_hold(const toks_ctx *ctx, toks_stream *st, void *hold, uint64_t cap)
{
    if (ctx == NULL || st == NULL || (hold == NULL && cap != 0u)) { return TOKS_E_ARG; }
    sst s;
    if (!sst_get(ctx, st, &s)) { return TOKS_E_ARG; }
    if (!ctx->dc.on && ctx->wp == NULL && ctx->dec_byte_level == 0u) { return TOKS_E_UNSUPPORTED; }
    if (!ctx->dc.on) { return 0; }                       /* ByteLevel, WordPiece: at most 3 bytes, in st itself */
    uint64_t n = sst_held(&s);
    if (n > (cap != 0u ? cap : SST_RUN)) { return TOKS_E_LIMIT; }  /* st unchanged */
    uint32_t need;
    uint8_t lo, hi;
    sst_u8(&s, &need, &lo, &hi);
    const uint8_t *from = sst_bytes(&s);                 /* st's own bytes, or the current hold (alive during the call) */
    if (cap == 0u) {                                     /* back into st's own bytes */
        uint8_t tmp[SST_RUN];
        copy(tmp, from, n);
        memset(s.buf, 0, sizeof s.buf);
        copy(s.buf, tmp, n);
        s.np = (uint8_t)n;
        s.mode = (uint8_t)(s.mode & ~SST_HOLD);
    } else {
        uint8_t *to = (uint8_t *)hold;
        uintptr_t dt = (uintptr_t)to, df = (uintptr_t)from;
        if (dt < df) {                                   /* the two may overlap: copy away from the overlap */
            for (uint64_t i = 0u; i < n; i++) { to[i] = from[i]; }            /* bound: n */
        } else if (dt > df) {
            for (uint64_t i = n; i > 0u; i--) { to[i - 1u] = from[i - 1u]; }  /* bound: n */
        }
        sst_x x;
        memset(&x, 0, sizeof x);
        x.p = to, x.cap = cap, x.n = n;
        x.chk = sst_chk(s.tag, &x);
        if ((s.mode & SST_VALID) != 0u) { x.need = (uint8_t)need, x.lo = lo, x.hi = hi; }
        memset(s.buf, 0, sizeof s.buf);
        sst_x_put(&s, &x);
        s.np = 0u;
        s.mode = (uint8_t)(s.mode | SST_HOLD);
    }
    memcpy(st, &s, sizeof s);
    return (int64_t)n;
}

uint64_t toks_stream_bound(const toks_ctx *ctx, uint64_t n_ids)
{
    if (ctx == NULL) { return 0u; }
    uint64_t d = ctx->dec_max, h = ctx->dc.on ? 3u * SST_RUN : 3u;   /* held bytes' worst */
    if (d != 0u && n_ids > (UINT64_MAX - h) / d) { return UINT64_MAX; }
    return n_ids * d + h;
}

/* ---- the chain (spm, unigram), incrementally (spm's batch is spm_c.c's toks_spm_decode, unigram's below) ---- */

typedef struct sw {                      /* the chain's writer: output cursor plus the stream's state */
    uint8_t *out;
    uint64_t cap, n;
    sst      s;
    const toks_spm_op *bfop;             /* bf_first: the per-token step ByteFallback's chars go through */
    uint64_t t_from, t_k;                /* spm_push: the open run's part in this push, ids[t_from, n) holding t_k run
                                            bytes, held by sw_commit once the call cannot fail */
} sw;

static void sw_raw(sw *w, const uint8_t *p, uint64_t k)
{
    uint64_t n = w->n;
    w->n = n + k;
    if (n < w->cap) { copy(w->out + n, p, (w->cap - n < k) ? w->cap - n : k); }
}

/* p[0, k) holds whole chars: Strip drops leading content chars first (spm_c.c put) */
static void sw_put(sw *w, const toks_spm_op *strip, const uint8_t *p, uint64_t k)
{
    while (w->s.strip_left != 0u && k != 0u) {          /* bound: Strip start */
        if (k >= strip->a.n && memcmp(p, strip->a.b, strip->a.n) == 0) {
            p += strip->a.n;
            k -= strip->a.n;
            w->s.strip_left--;
        } else {
            w->s.strip_left = 0u;
        }
    }
    sw_raw(w, p, k);
}

/* a U+FFFD per byte of a run that is not valid utf-8 as a whole */
static void sw_fffd(sw *w, const toks_spm_op *strip, uint64_t k)
{
    for (uint64_t i = 0u; i < k; i++) { sw_put(w, strip, FFFD, 3u); }    /* bound: k (bytes of one run) */
}

/* the bytes of a valid run, whole chars at a time (Strip compares whole chars) */
typedef struct chr { uint8_t b[4]; uint32_t have, need; } chr;

/* a string through the per-token step (spm_c.c put_token): Replace a -> b, or Metaspace r -> " ", every
 * r dropped from the first kept string unless the scheme is never */
static void sw_token(sw *w, const toks_spm_op *op, const toks_spm_op *strip, const uint8_t *p, uint64_t k, int first)
{
    if (op == NULL) { sw_put(w, strip, p, k); return; }
    static const uint8_t SP = ' ';
    const uint8_t *to = op->b.b;
    uint64_t to_n = op->b.n;
    if (op->kind == TOKS_SPM_D_METASPACE) {
        to = &SP;
        to_n = (first && op->scheme != TOKS_SPM_PS_NEVER) ? 0u : 1u;
    }
    uint64_t i = 0u, from = 0u;
    while (i + op->a.n <= k) {                           /* bound: k (i advances >= 1) */
        if (p[i] == op->a.b[0] && memcmp(p + i, op->a.b, op->a.n) == 0) {
            sw_put(w, strip, p + from, i - from);
            sw_put(w, strip, to, to_n);
            i += op->a.n;
            from = i;
        } else {
            i++;
        }
    }
    sw_put(w, strip, p + from, k - from);
}

static void sw_byte(sw *w, const toks_spm_op *strip, chr *c, uint8_t b)
{
    if (c->have == 0u) { c->need = (b < 0x80u) ? 1u : seq_need(b); }
    c->b[c->have++] = b;
    if (c->have != c->need) { return; }
    if (w->bfop != NULL) { sw_token(w, w->bfop, strip, c->b, c->have, 0); } else { sw_put(w, strip, c->b, c->have); }
    c->have = 0u;
}

/* ids[from, to)'s run bytes (kept byte tokens) through sw_byte, or into buf when buf != NULL */
static void run_bytes(const toks_ctx *ctx, const uint32_t *ids, uint64_t from, uint64_t to, int skip, sw *w,
                      const toks_spm_op *strip, chr *c, uint8_t *buf)
{
    for (uint64_t j = from; j < to; j++) {               /* bound: to - from */
        uint32_t id = ld32(ids + j);
        if (skip && toks_bit(ctx->special_ids, id) != 0u) { continue; }
        if (ctx->dc.holes != NULL && toks_bit(ctx->dc.holes, id) != 0u) { continue; }
        uint32_t o = ctx->t.tok_off[id];
        int b = toks_byte_token(ctx->t.tok_bytes + o, (uint64_t)ctx->t.tok_off[id + 1u] - o);
        if (b < 0) { continue; }                         /* unreachable: the run holds byte tokens only */
        if (buf != NULL) { *buf++ = (uint8_t)b; } else { sw_byte(w, strip, c, (uint8_t)b); }
    }
}

/* the open valid run (held bytes + ids[from, to)) that just ended: its chars when whole utf-8, else a U+FFFD each */
static void run_end(const toks_ctx *ctx, sw *w, const uint32_t *ids, uint64_t from, uint64_t to, int skip,
                    const toks_spm_op *strip, uint32_t need, uint64_t k_run)
{
    chr ch = { { 0u, 0u, 0u, 0u }, 0u, 0u };
    uint64_t np = sst_held(&w->s);
    if (need != 0u) { sw_fffd(w, strip, np + k_run); return; }
    const uint8_t *hb = sst_bytes(&w->s);
    for (uint64_t j = 0u; j < np; j++) { sw_byte(w, strip, &ch, hb[j]); }   /* bound: the held bytes */
    run_bytes(ctx, ids, from, to, skip, w, strip, &ch, NULL);
}

/* rationale: docs/notes/c-core.md §stream.c.5 */
static int64_t spm_push(const toks_ctx *ctx, sw *w, const uint32_t *ids, uint64_t n, int fin)
{
    const toks_dchain *sp = &ctx->dc;
    int bf;
    const toks_spm_op *strip;
    const toks_spm_op *op = spm_ops(sp, &bf, &strip);
    w->bfop = sp->bf_first ? op : NULL;
    int skip = (w->s.flags & TOKS_SKIP_SPECIAL) != 0u && ctx->special_ids != NULL;
    uint32_t need;                                       /* the open valid run's utf-8 state */
    uint8_t lo, hi;
    sst_u8(&w->s, &need, &lo, &hi);
    w->t_from = 0u, w->t_k = 0u;
    uint64_t from = 0u, k_run = 0u;                      /* this push's part of the run: from ids[from], k_run bytes */
    for (uint64_t i = 0u; i < n; i++) {                  /* bound: n */
        uint32_t id = ld32(ids + i);
        if (id >= ctx->t.n_ids) { return TOKS_E_ID; }
        if (skip && toks_bit(ctx->special_ids, id) != 0u) { continue; }
        if (sp->holes != NULL && ((sp->holes[id >> 5] >> (id & 31u)) & 1u) != 0u) { continue; }
        uint32_t o = ctx->t.tok_off[id];
        const uint8_t *p = ctx->t.tok_bytes + o;
        uint64_t k = (uint64_t)ctx->t.tok_off[id + 1u] - o;
        int first = (w->s.mode & SST_FIRST) == 0u;
        w->s.mode |= SST_FIRST;
        if (!sp->has_decoder) {                          /* hf joins the strings with " " */
            if (!first) { sw_raw(w, (const uint8_t *)" ", 1u); }
            sw_raw(w, p, k);
            continue;
        }
        int b = bf ? toks_byte_token(p, k) : -1;
        if (b >= 0) {                                    /* a byte of the open run */
            if ((w->s.mode & (SST_VALID | SST_INVALID)) == 0u) {
                w->s.mode |= SST_VALID;
                need = 0u;
                lo = 0x80u;
                hi = 0xBFu;
            }
            if ((w->s.mode & SST_INVALID) != 0u) { sw_put(w, strip, FFFD, 3u); continue; }
            if (k_run == 0u) { from = i; }
            uint8_t c = (uint8_t)b;
            int fits = (need == 0u) ? (c < 0x80u || (c >= 0xC2u && c <= 0xF4u)) : (c >= lo && c <= hi);
            if (fits) {
                if (need == 0u && c >= 0x80u) { need = seq_need(c) - 1u; lo = seq_lo(c); hi = seq_hi(c); }
                else if (need != 0u) { need--; lo = 0x80u; hi = 0xBFu; }
                k_run++;
                continue;
            }
            sw_fffd(w, strip, sst_held(&w->s) + k_run + 1u);   /* not valid as a whole: a U+FFFD per byte */
            sst_set_held(&w->s, 0u);
            k_run = 0u;
            w->s.mode = (uint8_t)((w->s.mode & ~SST_VALID) | SST_INVALID);
            continue;
        }
        if ((w->s.mode & SST_VALID) != 0u) { run_end(ctx, w, ids, from, i, skip, strip, need, k_run); }   /* a string ends it */
        w->s.mode &= (uint8_t)~(SST_VALID | SST_INVALID);
        sst_set_held(&w->s, 0u);
        k_run = 0u;
        sw_token(w, op, strip, p, k, first);
    }
    if (fin && (w->s.mode & SST_VALID) != 0u) {          /* the batch: the open run ends with the ids */
        run_end(ctx, w, ids, from, n, skip, strip, need, k_run);
        sst_set_held(&w->s, 0u);
    } else if ((w->s.mode & SST_VALID) != 0u && k_run != 0u) {  /* the run stays open: held for the next push */
        if (sst_held(&w->s) + k_run > sst_cap(&w->s)) { return TOKS_E_LIMIT; }   /* its output waits on its end */
        w->t_from = from, w->t_k = k_run;
    }
    if ((w->s.mode & SST_HOLD) != 0u) {                  /* the run's utf-8 state, kept in the hold's record */
        int open = (w->s.mode & SST_VALID) != 0u;
        sst_x x = sst_x_get(&w->s);
        x.need = open ? (uint8_t)need : 0u, x.lo = open ? lo : 0u, x.hi = open ? hi : 0u;
        sst_x_put(&w->s, &x);
    }
    return 0;
}

/* the push cannot fail any more: the open run's bytes from it join the held ones (in a caller's hold only now, so
 * a failed push never wrote there: TOKS_E_CAP and TOKS_E_LIMIT leave the hold as they leave st) */
static void sw_commit(const toks_ctx *ctx, sw *w, const uint32_t *ids, uint64_t n)
{
    if (w->t_k == 0u) { return; }
    int skip = (w->s.flags & TOKS_SKIP_SPECIAL) != 0u && ctx->special_ids != NULL;
    uint64_t np = sst_held(&w->s);
    uint8_t *hb = (w->s.mode & SST_HOLD) != 0u ? sst_x_get(&w->s).p : w->s.buf;
    run_bytes(ctx, ids, w->t_from, n, skip, w, NULL, NULL, hb + np);
    sst_set_held(&w->s, np + w->t_k);
}

int64_t toks_stream_push(const toks_ctx *ctx, toks_stream *st, const uint32_t *ids, uint64_t n,
                         uint8_t *out, uint64_t cap)
{
    if (ctx == NULL || st == NULL || (ids == NULL && n != 0u) || (out == NULL && cap != 0u)) { return TOKS_E_ARG; }
    sst s;
    if (!sst_get(ctx, st, &s)) { return TOKS_E_ARG; }
    if (!ctx->dc.on && ctx->wp == NULL && ctx->dec_byte_level == 0u) { return TOKS_E_UNSUPPORTED; }
    if (n > TOKS_MAX_TEXT) { return TOKS_E_LIMIT; }     /* so every count below stays < 2^47 */
    if (ctx->wp != NULL) {                               /* WordPiece: per token, nothing held (above) */
        for (uint64_t i = 0u; i < n; i++) {              /* bound: n (nothing is written on TOKS_E_ID) */
            if (ld32(ids + i) >= ctx->t.n_ids) { return TOKS_E_ID; }
        }
        uint64_t k = (s.mode & SST_FIRST) != 0u ? 1u : 0u;
        int64_t r = toks_wp_decode_k(ctx, ids, n, s.flags, out, cap, &k);
        if (r < 0) { return r; }
        if ((uint64_t)r > cap) { return TOKS_E_CAP; }    /* out[0, cap): the exact prefix; st as it was */
        if (k != 0u) { s.mode = (uint8_t)SST_FIRST; }
        memcpy(st, &s, sizeof s);
        return r;
    }
    if (ctx->dc.on) {
        sw w = { out, cap, 0u, s, NULL, 0u, 0u };
        int64_t r = spm_push(ctx, &w, ids, n, 0);
        if (r != 0) { return r; }
        if (w.n > cap) { return TOKS_E_CAP; }
        sw_commit(ctx, &w, ids, n);
        memcpy(st, &w.s, sizeof w.s);
        return (int64_t)w.n;
    }
    if (n == 1u && s.np == 0u && ctx->dec_len != NULL && cap >= 16u) {
        /* the serving path, one generated token: nothing held and a short id that is its own output (the
         * writer's fast path without its loop): one 16-byte move, and st stays as it is */
        uint32_t id = ld32(ids);
        if (id < ctx->t.n_ids && ctx->dec_len[id] <= 16u &&
            ((s.flags & TOKS_SKIP_SPECIAL) == 0u || ctx->special_ids == NULL ||
             toks_bit(ctx->special_ids, id) == 0u)) {
            copy16(out, ctx->dec_slot + 16u * (uint64_t)id);
            return (int64_t)ctx->dec_len[id];
        }
    }
    toks_lossy d;
    d.out = out;
    d.cap = cap;
    d.n = 0u;
    d.np = s.np;
    memcpy(d.pend, s.buf, 4);
    int64_t r = toks_lossy_ids(ctx, &d, ids, n, s.flags);
    if (r != 0) { return r; }
    if (d.n > cap) { return TOKS_E_CAP; }
    s.np = (uint8_t)d.np;
    memcpy(s.buf, d.pend, 4);
    memcpy(st, &s, sizeof s);
    return (int64_t)d.n;
}

int64_t toks_stream_flush(const toks_ctx *ctx, toks_stream *st, uint8_t *out, uint64_t cap)
{
    if (ctx == NULL || st == NULL || (out == NULL && cap != 0u)) { return TOKS_E_ARG; }
    sst s;
    if (!sst_get(ctx, st, &s)) { return TOKS_E_ARG; }
    if (!ctx->dc.on && ctx->wp == NULL && ctx->dec_byte_level == 0u) { return TOKS_E_UNSUPPORTED; }
    sw w = { out, cap, 0u, s, NULL, 0u, 0u };            /* the held bytes' output into out[0, cap) */
    if (ctx->dc.on) {
        int bf;
        const toks_spm_op *strip;
        const toks_spm_op *op = spm_ops(&ctx->dc, &bf, &strip);
        w.bfop = ctx->dc.bf_first ? op : NULL;
        uint32_t need;
        uint8_t lo, hi;
        sst_u8(&s, &need, &lo, &hi);                     /* need 0: the held run is whole chars */
        if ((s.mode & SST_VALID) != 0u) { run_end(ctx, &w, NULL, 0u, 0u, 0, strip, need, 0u); }
    } else if (s.np != 0u) {
        sw_raw(&w, FFFD, 3u);                            /* the held prefix's U+FFFD */
    }
    if (w.n > cap) { return TOKS_E_CAP; }                /* out[0, cap): the exact prefix */
    sst_reset(ctx, &s);
    memcpy(st, &s, sizeof s);                            /* == the state toks_stream_init left */
    return (int64_t)w.n;
}

/* the batch decode of unigram's chain (api.c toks_decode): the stream's step from the initial state, every run
 * decided at the end */
int64_t toks_uni_dec(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *out, uint64_t cap)
{
    sw w = { out, cap, 0u, { 0 }, NULL, 0u, 0u };
    w.s.flags = flags & TOKS_SKIP_SPECIAL;
    sst_reset(ctx, &w.s);
    int64_t r = spm_push(ctx, &w, ids, n, 1);
    return r != 0 ? r : (int64_t)w.n;
}
