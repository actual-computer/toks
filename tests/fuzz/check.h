/*
 * tests/fuzz/check.h: the oracles of docs/fuzz.md §2 on one input, for a tokenizer under test (fuzz.h fz_tok):
 * fz_check_text (encode / pieces), fz_check_decode, fz_check_stream, fz_check_par. sel: bits the harness derived
 * from its input (alignments, capacities, partitions), so every choice is reproducible from the saved input.
 */
#ifndef TOKS_FUZZ_CHECK_H
#define TOKS_FUZZ_CHECK_H

#include "fuzz.h"

/* decode: the documented errors with nothing written, tiers agree, the capacity rule, valid utf-8, and for FZ_RT
 * files exactly from_utf8_lossy(concat of toks_token bytes) without TOKS_SKIP_SPECIAL */
FZ_FN void fz_check_decode(fz_tok *t, const uint32_t *ids, uint64_t n, uint32_t flags, uint64_t sel)
{
    uint8_t *o, *o2, guard[8], *blk;
    int bad_id = 0;
    fz_mis = (uint32_t)(sel >> 4) & 3u;
    const uint32_t *mids = FZ_U32P(fz_mis_ids(&blk, ids, n));
    for (uint64_t i = 0; i < n; i++) { bad_id |= ids[i] >= t->info.n_ids; }
    int64_t m = fz_decode(t->a, ids, n, flags, &o);
    if (bad_id) {
        FZ_CHECK(m == TOKS_E_ID, "decode of an id >= n_ids (%u) returned %" PRId64, t->info.n_ids, m);
        memset(guard, 0x5A, sizeof guard);
        FZ_CHECK(toks_decode(t->a, mids, n, flags, guard, sizeof guard) == TOKS_E_ID, "TOKS_E_ID with a buffer");
        for (uint64_t i = 0; i < sizeof guard; i++) { FZ_CHECK(guard[i] == 0x5Au, "TOKS_E_ID wrote out[%" PRIu64 "]", i); }
        free(blk);
        return;
    }
    FZ_CHECK(m >= 0, "decode returned %" PRId64 " (n %" PRIu64 ", flags %u)", m, n, flags);
    FZ_CHECK(fz_utf8_valid(o, (uint64_t)m), "decode output is not valid utf-8");
    int64_t m2 = fz_decode(t->s, ids, n, flags, &o2);
    FZ_CHECK(m2 == m && (m == 0 || memcmp(o, o2, (size_t)m) == 0), "decode: the scalar and auto contexts differ");
    free(o2);
    FZ_CHECK(toks_decode(t->a, mids, n, flags | 2u, NULL, 0u) == TOKS_E_ARG, "decode accepted unknown flags");
    uint64_t cap = (uint64_t)m == 0u ? 0u : (sel >> 8) % ((uint64_t)m + 1u);   /* the capacity rule at one cap */
    uint8_t *c = (uint8_t *)fz_alloc(cap + 8u);
    memset(c, 0x5A, (size_t)cap + 8u);
    FZ_CHECK(toks_decode(t->a, mids, n, flags, c, cap) == m, "decode at cap %" PRIu64 " changed the total", cap);
    FZ_CHECK(cap == 0u || memcmp(c, o, (size_t)cap) == 0, "decode at cap %" PRIu64 ": not the exact prefix", cap);
    for (uint64_t i = cap; i < cap + 8u; i++) { FZ_CHECK(c[i] == 0x5Au, "decode at cap %" PRIu64 " wrote out[%" PRIu64 "]", cap, i); }
    free(c);
    if ((t->oracles & FZ_RT) && flags == 0u) {
        fz_bytes raw = { NULL, 0, 0 };
        for (uint64_t i = 0; i < n; i++) {
            uint64_t k = 0;
            const uint8_t *p = toks_token(t->a, ids[i], &k);
            if (p != NULL) { fz_bytes_put(&raw, p, k); }
        }
        uint8_t *ref = (uint8_t *)fz_alloc(3u * raw.n + 1u);
        uint64_t rl = fz_lossy(raw.p, raw.n, ref);
        FZ_CHECK(rl == (uint64_t)m && (rl == 0u || memcmp(ref, o, (size_t)rl) == 0),
                 "decode != from_utf8_lossy(token bytes) (%" PRIu64 " vs %" PRId64 " bytes)", rl, m);
        free(ref);
        fz_bytes_free(&raw);
    }
    free(o);
    free(blk);
}

/* stream: the ids pushed in random parts (sizes from sel's rng) + flush == batch decode; a push or flush with
 * one byte less than it needs returns TOKS_E_CAP with out[0, cap) the exact prefix and st unchanged */
FZ_FN void fz_check_stream(fz_tok *t, const uint32_t *ids, uint64_t n, uint32_t flags, uint64_t sel)
{
    uint8_t *ref, *blk;
    for (uint64_t i = 0; i < n; i++) { if (ids[i] >= t->info.n_ids) { return; } }
    fz_mis = (uint32_t)(sel >> 2) & 3u;
    const uint32_t *mids = FZ_U32P(fz_mis_ids(&blk, ids, n));
    int64_t m = fz_decode(t->a, ids, n, flags, &ref);
    FZ_CHECK(m >= 0, "decode %" PRId64, m);
    fz_bytes got = { NULL, 0, 0 };
    toks_stream st;
    toks_stream_init(t->a, &st, flags);
    uint64_t s = sel | 1u, i = 0;
    int limited = 0;
    while (i <= n) {
        int fin = i == n;
        uint64_t k = fin ? 0u : 1u + fz_below(&s, fz_below(&s, 4u) == 0u ? 64u : 4u);
        if (k > n - i) { k = n - i; }
        uint64_t bound = toks_stream_bound(t->a, k);
        uint8_t *o = (uint8_t *)fz_alloc(bound + 8u);
        toks_stream before = st;
        int64_t r = fin ? toks_stream_flush(t->a, &st, o, bound) : toks_stream_push(t->a, &st, mids + i, k, o, bound);
        if (r == TOKS_E_LIMIT && !fin) {                /* an spm byte run beyond toks_stream's room: documented */
            FZ_CHECK(memcmp(&before, &st, sizeof st) == 0, "TOKS_E_LIMIT changed the stream state");
            limited = 1;
            free(o);
            break;
        }
        FZ_CHECK(r >= 0 && (uint64_t)r <= bound, "stream %s of %" PRIu64 " ids returned %" PRId64 " (bound %" PRIu64 ")",
                 fin ? "flush" : "push", k, r, bound);
        if (r > 0 && fz_below(&s, 4u) == 0u) {          /* atomicity: replay the call one byte short */
            toks_stream after = st, again = before;
            uint8_t *c = (uint8_t *)fz_alloc((uint64_t)r + 8u);
            memset(c, 0x5A, (size_t)r + 8u);
            int64_t r2 = fin ? toks_stream_flush(t->a, &again, c, (uint64_t)r - 1u)
                             : toks_stream_push(t->a, &again, mids + i, k, c, (uint64_t)r - 1u);
            FZ_CHECK(r2 == TOKS_E_CAP, "a stream call one byte short returned %" PRId64, r2);
            FZ_CHECK(memcmp(&again, &before, sizeof st) == 0, "TOKS_E_CAP changed the stream state");
            FZ_CHECK(memcmp(c, o, (size_t)r - 1u) == 0, "TOKS_E_CAP: out[0, cap) is not the exact prefix");
            for (uint64_t j = (uint64_t)r - 1u; j < (uint64_t)r + 8u; j++) { FZ_CHECK(c[j] == 0x5Au, "TOKS_E_CAP wrote out[%" PRIu64 "]", j); }
            free(c);
            st = after;
        }
        fz_bytes_put(&got, o, (uint64_t)r);
        free(o);
        if (fin) { break; }
        i += k;
    }
    if (!limited) {
        FZ_CHECK(got.n == (uint64_t)m && (m == 0 || memcmp(got.p, ref, (size_t)m) == 0),
                 "stream (%" PRIu64 " bytes) != batch decode (%" PRId64 " bytes), %" PRIu64 " ids, flags %u", got.n, m, n, flags);
    }
    fz_bytes_free(&got);
    free(ref);
    free(blk);
}

/* encode (pieces = 0) or pieces (pieces = 1) of one text: tiers, scratch states, rebinding, the capacity rule,
 * the pieces' shape, and for encode the decode / stream / roundtrip oracles on the ids */
FZ_FN void fz_check_text(fz_tok *t, int pieces, const uint8_t *x, uint64_t len, uint32_t flags, uint64_t sel)
{
    fz_scr s0, s1;
    uint32_t *ref, *o, *o2;
    fz_mis = (uint32_t)(sel >> 36) & 3u;
    fz_scr_new(&s0, t->s, len, 0u, (uint32_t)sel);
    fz_scr_new(&s1, t->a, len, 0u, (uint32_t)(sel >> 6));
    uint64_t n = fz_run(pieces, t->s, x, len, flags, s0.p, &ref), m;
    m = fz_run(pieces, t->a, x, len, flags, s1.p, &o);
    FZ_SAME_IDS(ref, n, o, m, "auto tier vs scalar");
    free(o);
    m = fz_run(pieces, t->a, x, len, flags, s1.p, &o);
    FZ_SAME_IDS(ref, n, o, m, "auto, scratch warm with this input");
    free(o);
    fz_warm_fit(t, len);
    m = fz_run(pieces, t->a, x, len, flags, t->warm, &o);
    FZ_SAME_IDS(ref, n, o, m, "auto, scratch warm across inputs");
    free(o);
    m = fz_run(pieces, t->a, x, len, flags, t->warm32, &o);
    FZ_SAME_IDS(ref, n, o, m, "auto, 32 MiB scratch warm across inputs");
    free(o);
    if ((sel >> 12) & 1u) {                             /* rebind: s0 to ANOTHER tokenizer and back */
        fz_tok *o = fz_other(t, (uint32_t)(sel >> 40));
        if (o != NULL && toks_scratch_init(o->a, s0.p, s0.len, 0u) == 0) {
            int64_t r = toks_encode(t->s, x, len, flags, NULL, 0u, s0.p);
            FZ_CHECK(r == TOKS_E_SCRATCH, "a scratch bound to %s returned %" PRId64, o->name, r);
            FZ_CHECK(toks_scratch_init(t->s, s0.p, s0.len, 0u) == 0, "rebind back");
            m = fz_run(pieces, t->s, x, len, flags, s0.p, &o2);
            FZ_SAME_IDS(ref, n, o2, m, "scalar after rebinding back from another tokenizer");
            free(o2);
        }
        m = fz_run(pieces, t->a, x, len, flags, s0.p, &o2);  /* a scratch of the same source serves both tiers */
        FZ_SAME_IDS(ref, n, o2, m, "auto on the scalar context's scratch");
        free(o2);
    }
    uint64_t cap = n == 0u ? 0u : (sel >> 13) % n;     /* the capacity rule below and above the count */
    fz_check_cap(pieces, t->a, x, len, flags, s1.p, ref, n, cap);
    fz_check_cap(pieces, t->s, x, len, flags, s0.p, ref, n, n + ((sel >> 30) & 3u));
    FZ_CHECK((pieces ? toks_pieces(t->a, x, len, flags | 16u, NULL, 0u, s1.p)
                     : toks_encode(t->a, x, len, flags | 16u, NULL, 0u, s1.p)) == TOKS_E_ARG, "unknown flags accepted");
    if (pieces) {
        for (uint64_t i = 0; i < n; i++) {             /* ends may go back: an rstrip token's span overlaps the matches */
            FZ_CHECK(ref[i] <= 3u * len + 3u, "piece end %u past 3 x len %" PRIu64, ref[i], len);   /* after it, as in hf */
        }
        if ((t->oracles & FZ_RT) && len != 0u) {
            FZ_CHECK(n != 0u && ref[n - 1] == len, "pieces do not end at len %" PRIu64 " (last %u)", len, n ? ref[n - 1] : 0u);
        }
    } else {
        for (uint64_t i = 0; i < n; i++) { FZ_CHECK(ref[i] < t->info.n_ids, "id %u >= n_ids %u", ref[i], t->info.n_ids); }
        fz_check_decode(t, ref, n, (uint32_t)(sel >> 32) & 1u, sel >> 33);
        fz_check_stream(t, ref, n, (uint32_t)(sel >> 34) & 1u, sel >> 35);
        if ((t->oracles & FZ_RT) && (flags & ~TOKS_CONTINUATION) == (TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS)) {
            uint8_t *d, *want = (uint8_t *)fz_alloc(3u * len + 1u);
            int64_t dl = fz_decode(t->a, ref, n, 0u, &d);
            uint64_t wl = fz_lossy(x, len, want);
            FZ_CHECK(dl == (int64_t)wl && (wl == 0u || memcmp(d, want, (size_t)wl) == 0),
                     "decode(encode(x)) != from_utf8_lossy(x) (%" PRId64 " vs %" PRIu64 " bytes)", dl, wl);
            free(d);
            free(want);
        }
    }
    free(ref);
    fz_scr_free(&s0);
    fz_scr_free(&s1);
}

/* a pool of 3 over t's auto context: eager = env TOKS_PAR_EAGER=1 at create (every call of >= 16 KiB goes wide:
 * the parallel paths), else the cost model's choice (calls of < 16 KiB never leave the caller) */
FZ_FN toks_par *fz_par_pool(fz_tok *t, int eager, uint64_t sel)
{
    toks_par **pp = eager ? &t->par : &t->par1;
    if (*pp == NULL) {
        if (eager) { setenv("TOKS_PAR_EAGER", "1", 1); }
        int64_t r = toks_par_create(pp, t->a, 3u, (sel & 1u) ? TOKS_SCRATCH_CACHE_MIB(32) : 0u);
        if (eager) { unsetenv("TOKS_PAR_EAGER"); }
        FZ_CHECK(r == 0 && *pp != NULL, "toks_par_create returned %" PRId64, r);
    }
    return *pp;
}

/* the text tiled with a separator to 16 .. 144 KiB, in an exact-size heap block: toks_par splits only calls of
 * >= 16 KiB, into units of >= 8 KiB, so a fuzz input alone (<= 8 KiB) never reaches its parallel paths */
FZ_FN uint8_t *fz_tile(const uint8_t *x, uint64_t len, uint64_t sel, uint64_t *out_len)
{
    static const char *const SEP[] = { "", "\n", " ", "\n\n", ". ", "\t", "\r\n" };
    const char *sep = SEP[sel % FZ_N(SEP)];
    uint64_t want = (16u << 10) + (sel >> 3) % (128u << 10);
    fz_bytes b = { NULL, 0, 0 };
    while (b.n < want) {
        if (len == 0u && sep[0] == 0) { fz_bytes_put(&b, "a ", 2u); }
        fz_bytes_put(&b, x, len);
        fz_bytes_put(&b, sep, strlen(sep));
    }
    uint8_t *blk = (uint8_t *)fz_alloc(b.n);
    memcpy(blk, b.p, (size_t)b.n);
    *out_len = b.n;
    fz_bytes_free(&b);
    return blk;
}

/* toks_par_encode_batch: x cut into 1-12 documents (now and then a NULL text, a text past TOKS_MAX_TEXT, a
 * count-only item), each into its own misaligned out at a random cap: every item's n and ids == toks_encode's
 * on it, nothing written at or past its cap; unknown flags are TOKS_E_ARG with no item touched */
FZ_FN void fz_check_par_batch(fz_tok *t, toks_par *par, const uint8_t *x, uint64_t len, uint32_t flags, uint64_t sel,
                              void *scr)
{
    uint64_t s = sel ^ 0x5DEECE66Dull, m = 1u + fz_below(&s, 12u), cut[13];
    toks_par_item it[12];
    uint8_t *blk[12];
    int64_t want[12];
    uint32_t *ref[12];
    cut[0] = 0u;
    for (uint64_t i = 1; i < m; i++) { cut[i] = fz_below(&s, len + 1u); }
    for (uint64_t i = 1; i < m; i++) {                  /* sort cut[1, m) */
        for (uint64_t j = i; j > 1u && cut[j - 1] > cut[j]; j--) { uint64_t c = cut[j]; cut[j] = cut[j - 1]; cut[j - 1] = c; }
    }
    cut[m] = len;
    fz_mis = (uint32_t)(sel >> 7) & 3u;
    for (uint64_t i = 0; i < m; i++) {
        const uint8_t *d = x + cut[i];
        uint64_t dl = cut[i + 1] - cut[i], kind = fz_below(&s, 16u);
        ref[i] = NULL;
        if (kind == 0u) { d = NULL; dl = 1u + fz_below(&s, 64u); }               /* a NULL text */
        else if (kind == 1u) { dl = TOKS_MAX_TEXT + 1u + fz_below(&s, 4u); }    /* past the limit: never read */
        if (kind <= 1u) {
            want[i] = toks_encode(t->a, d, dl, flags, NULL, 0u, scr);
            FZ_CHECK(want[i] < 0, "toks_encode accepted a %s text (%" PRId64 ")", d == NULL ? "NULL" : "too long", want[i]);
        } else {
            want[i] = (int64_t)fz_run(0, t->a, d, dl, flags, scr, &ref[i]);
        }
        uint64_t n = want[i] > 0 ? (uint64_t)want[i] : 0u, cap;
        switch (kind <= 2u ? 0u : fz_below(&s, 4u)) {
        case 0: cap = 0u; break;
        case 1: cap = n / 2u; break;
        case 2: cap = n; break;
        default: cap = n + 3u; break;
        }
        uint8_t *mb = fz_mis_ids(&blk[i], NULL, cap + 8u);
        memset(mb, 0xA5, (size_t)(cap + 8u) * 4u);
        it[i].text = d;
        it[i].len = dl;
        it[i].out = (kind == 2u && fz_below(&s, 2u) == 0u) ? NULL : FZ_U32P(mb);   /* count-only: NULL, cap 0 */
        it[i].cap = it[i].out == NULL ? 0u : cap;
        it[i].n = 0x7777;
    }
    FZ_CHECK(toks_par_encode_batch(par, it, m, flags | 16u) == TOKS_E_ARG, "toks_par_encode_batch: unknown flags");
    for (uint64_t i = 0; i < m; i++) { FZ_CHECK(it[i].n == 0x7777, "a refused batch touched item %" PRIu64, i); }
    FZ_CHECK(toks_par_encode_batch(par, it, m, flags) == 0, "toks_par_encode_batch refused %" PRIu64 " items", m);
    for (uint64_t i = 0; i < m; i++) {
        FZ_CHECK(it[i].n == want[i], "batch item %" PRIu64 " of %" PRIu64 " (%" PRIu64 " bytes): n %" PRId64 " vs %" PRId64,
                 i, m, it[i].len, it[i].n, want[i]);
        uint8_t *mb = blk[i] + fz_mis;
        uint64_t cap = it[i].cap, ni = want[i] > 0 ? (uint64_t)want[i] : 0u, k = ni < cap ? ni : cap;
        if (it[i].out != NULL) {
            FZ_CHECK(k == 0u || memcmp(mb, ref[i], (size_t)k * 4u) == 0, "batch item %" PRIu64 ": not the exact prefix", i);
            for (uint64_t j = cap * 4u; j < (cap + 8u) * 4u; j++) {
                FZ_CHECK(mb[j] == 0xA5u, "batch item %" PRIu64 " (cap %" PRIu64 ") wrote byte %" PRIu64, i, cap, j);
            }
        }
        free(ref[i]);
        free(blk[i]);
    }
}

/* toks_par: toks_par_encode == toks_encode at any cap (half the inputs tiled past 16 KiB so the pool splits them,
 * 7 in 8 on the eager pool), the batch call, and every certified cut exact (the parts concatenate to the whole's
 * ids without post-processing) */
FZ_FN void fz_check_par(fz_tok *t, const uint8_t *x, uint64_t len, uint32_t flags, uint64_t sel)
{
    uint64_t ylen = len;
    uint8_t *big = ((sel >> 50) & 1u) ? fz_tile(x, len, sel >> 51, &ylen) : NULL;
    const uint8_t *y = big != NULL ? big : x;
    toks_par *par = fz_par_pool(t, ((sel >> 2) & 7u) != 0u, sel);
    fz_scr s;
    uint32_t *ref, *o;
    fz_scr_new(&s, t->a, ylen, 0u, (uint32_t)sel);
    uint64_t n = fz_run(0, t->a, y, ylen, flags, s.p, &ref);
    uint64_t cap = ((sel >> 5) & 3u) == 0u ? (n == 0u ? 0u : (sel >> 9) % n) : n + ((sel >> 5) & 3u) - 1u;
    fz_mis = (uint32_t)(sel >> 4) & 3u;
    uint8_t *blk, *mb = fz_mis_ids(&blk, NULL, cap + 8u);
    memset(mb, 0xA5, (size_t)(cap + 8u) * 4u);
    int64_t r = toks_par_encode(par, y, ylen, flags, FZ_U32P(mb), cap);
    FZ_CHECK(r == (int64_t)n, "toks_par_encode returned %" PRId64 " instead of %" PRIu64 " (len %" PRIu64 ", cap %" PRIu64 ")",
             r, n, ylen, cap);
    uint64_t k = cap < n ? cap : n;
    if (k != 0u && memcmp(mb, ref, (size_t)k * 4u) != 0) {
        o = (uint32_t *)fz_alloc(k * 4u);
        memcpy(o, mb, (size_t)k * 4u);
        FZ_SAME_IDS(ref, k, o, k, "toks_par_encode vs toks_encode");
        free(o);
    }
    for (uint64_t i = cap * 4u; i < (cap + 8u) * 4u; i++) {
        FZ_CHECK(mb[i] == 0xA5u, "toks_par_encode (cap %" PRIu64 ", n %" PRIu64 ") wrote byte %" PRIu64 " of out", cap, n, i);
    }
    free(blk);
    free(ref);
    fz_check_par_batch(t, par, y, ylen, flags, sel >> 3, s.p);
    uint32_t want = 2u + (uint32_t)((sel >> 8) % 15u), base = flags | TOKS_NO_POSTPROCESS;
    uint64_t offs[16];
    int64_t c = toks_split_points(t->a, y, ylen, base, want, offs, 16u, NULL);
    FZ_CHECK(c >= 0 && c <= (int64_t)want - 1, "toks_split_points returned %" PRId64 " for %u parts", c, want);
    if (c > 0) {
        uint64_t nw = fz_run(0, t->a, y, ylen, base, s.p, &ref), at = 0, got = 0;
        uint32_t *cat = (uint32_t *)fz_alloc((nw + 1u) * 4u);
        for (int64_t i = 0; i <= c; i++) {
            uint64_t e = i < c ? offs[i] : ylen;
            FZ_CHECK(e > at || (i == c && e == at), "cuts not strictly increasing inside (0, len): %" PRIu64 " after %" PRIu64, e, at);
            uint64_t kk = fz_run(0, t->a, y + at, e - at, base | (i > 0 ? TOKS_CONTINUATION : 0u), s.p, &o);
            FZ_CHECK(got + kk <= nw, "the parts give more ids than the whole (cut %" PRId64 " at %" PRIu64 ")", i, e);
            memcpy(cat + got, o, (size_t)kk * 4u);
            got += kk;
            free(o);
            at = e;
        }
        FZ_SAME_IDS(ref, nw, cat, got, "parts at toks_split_points' cuts vs the whole");
        free(cat);
        free(ref);
    }
    fz_scr_free(&s);
    free(big);
}

#endif /* TOKS_FUZZ_CHECK_H */
