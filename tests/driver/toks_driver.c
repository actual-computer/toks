// toks_driver.c -- the toks parity driver: a tiny program speaking a length-prefixed binary
// protocol on stdin/stdout so the Python oracle can drive toks on any target (local binary,
// `arch -x86_64` binary, or `ssh host cmd`) without caring where it runs.
//
// It uses only include/toks.h and is not part of `make test`. Build it against the static library
// (`make lib` writes build/<os>-<isa>/libtoks.a, e.g. build/macos-arm64), as tests/parity/dsv3_parity.sh does:
//     clang -std=c17 -O2 -Iinclude -o build/toks_driver tests/driver/toks_driver.c build/<os>-<isa>/libtoks.a
//
// Protocol (all integers little-endian, u32 unless noted):
//   Request:  u32 magic 'T','K','R','1' | u32 op | u32 flags | u32 path_len | path bytes
//             then per op:
//               LOAD    : nothing more
//               ENCODE  : u32 len | text bytes          -> i64 n, u32 ids[min(n, cap)]
//               PIECES  : u32 len | text bytes          -> i64 n, u32 ends[min(n, cap)]
//               DECODE  : u32 n | u32 ids[n]            -> i64 n, u8 bytes[min(n, cap)]
//               STREAM  : u32 k | k x (u32 n | u32 ids[n])  -> per push, then the flush: u32 len | bytes
//               VOCAB   : u32 k | k x (u32 len | bytes)  -> per id: i32 toks_id_flags, i32 toks_token_to_id of
//                         toks_token's bytes (-1000: no string); then per string: i32 toks_token_to_id
//               UNLOAD  : nothing more                  -> i64 0
//   Response: i64 status (0 ok, negative TOKS_E_*), i64 nbytes, then nbytes of payload (the ids or ends
//             as little-endian u32, or the decoded bytes; nbytes = 0 for LOAD, UNLOAD and errors).
//   STREAM runs one toks_stream (flags & 1 = its init flags) through the k pushes and a flush, and checks
//   SPEC §4.7 on every call: the bytes fit toks_stream_bound; one byte less is TOKS_E_CAP with st
//   unchanged and out[0, cap) the exact prefix; the exact capacity gives the same bytes. A violation is
//   status -990 (atomicity), -989 (bound) or -991 (exact capacity differs). Flag 2: a push that
//   returns TOKS_E_LIMIT is answered as toks.h says, the run moved into a caller's hold of at least
//   the current size plus the push's n (doubled at least) and the same ids pushed again, so no case
//   is refused (-992: the move failed); the trial calls get a private copy of the hold.
//   The driver exits 0 on clean EOF, 2 on protocol or load errors (after writing a status).
//
// The Python runner batches many requests per process (protocol is request/response per write,
// with responses flushed after each request) and treats a nonzero exit as a hard failure.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toks.h"

// ---------------------------------------------------------------- portability shims

static int rd_full(uint8_t *p, size_t n) {            // read exactly n bytes from stdin
    size_t got = 0;
    while (got < n) {
        size_t r = fread(p + got, 1, n - got, stdin);
        if (r == 0) return 0;
        got += r;
    }
    return 1;
}

static int rd_u32(uint32_t *v) { return rd_full((uint8_t *)v, 4); }

static void reply(int64_t status, const void *p, uint64_t n) {   // status, byte count, payload
    int64_t nb = (int64_t)n;
    fwrite(&status, 8, 1, stdout);
    fwrite(&nb, 8, 1, stdout);
    if (n) fwrite(p, 1, (size_t)n, stdout);
    fflush(stdout);
}
static void wr_i64(int64_t v) { reply(v, NULL, 0); }

static uint8_t *slurp_stdin(uint32_t len) {
    uint8_t *b = malloc(len ? len : 1);
    if (!b || !rd_full(b, len)) { free(b); return NULL; }
    return b;
}

static toks_ctx *ctx = NULL;
static void *scr = NULL;
static uint64_t scr_len = 0;          // the longest text the bound scratch holds

static int ensure_scratch(uint64_t len) {
    if (scr && len <= scr_len) return 0;
    uint64_t want = len < 65536 ? 65536 : len;     // grow geometrically: few rebinds
    if (want < 2 * scr_len) want = 2 * scr_len;
    uint64_t need = toks_scratch_bytes(ctx, want, 0);
    free(scr);
    scr = malloc(need);
    if (!scr) return -1;
    if (toks_scratch_init(ctx, scr, need, 0) != 0) return -1;
    scr_len = want;
    return 0;
}

static void do_ids(uint32_t flags, int pieces) {   // ENCODE / PIECES
    uint32_t len;
    if (!rd_u32(&len)) { wr_i64(-999); return; }
    uint8_t *text = slurp_stdin(len);
    if (!text) { wr_i64(-999); return; }
    if (ensure_scratch(len) != 0) { wr_i64(-998); free(text); return; }
    uint64_t cap = (uint64_t)len + 16;                  // a retry covers a count above cap
    uint32_t *ids = malloc(4 * cap);
    int64_t n = pieces ? toks_pieces(ctx, text, len, flags, ids, cap, scr) : toks_encode(ctx, text, len, flags, ids, cap, scr);
    if (n > (int64_t)cap) {
        free(ids);
        cap = (uint64_t)n;
        ids = malloc(4 * cap);
        n = pieces ? toks_pieces(ctx, text, len, flags, ids, cap, scr) : toks_encode(ctx, text, len, flags, ids, cap, scr);
    }
    if (n < 0) reply(n, NULL, 0); else reply(0, ids, 4 * (uint64_t)n);
    free(ids);
    free(text);
}

static void do_decode(uint32_t flags) {
    uint32_t n;
    if (!rd_u32(&n)) { wr_i64(-999); return; }
    uint32_t *ids = malloc(4 * (n ? n : 1));
    if (!rd_full((uint8_t *)ids, 4 * (size_t)n)) { wr_i64(-999); free(ids); return; }
    uint64_t cap = 64ull * (n ? n : 1) + 256;           // a retry covers a longer output
    uint8_t *out = malloc(cap);
    int64_t k = toks_decode(ctx, ids, n, flags, out, cap);
    if (k > (int64_t)cap) {
        free(out);
        cap = (uint64_t)k;
        out = malloc(cap);
        k = toks_decode(ctx, ids, n, flags, out, cap);
    }
    if (k < 0) reply(k, NULL, 0); else reply(0, out, (uint64_t)k);
    free(out);
    free(ids);
}

// the vocabulary's lookups: every id's flags and its string's id, then each of the k strings' ids
static void do_vocab(void) {
    uint32_t k;
    if (!rd_u32(&k)) { wr_i64(-999); return; }
    toks_info info;
    info.size = sizeof info;
    if (toks_get_info(ctx, &info) != 0) { wr_i64(-998); return; }
    uint64_t n = 2u * (uint64_t)info.n_ids + k;
    int32_t *res = malloc(4 * (size_t)(n ? n : 1));
    if (!res) { wr_i64(-998); return; }
    for (uint32_t id = 0; id < info.n_ids; id++) {
        uint64_t len = 0;
        const uint8_t *s = toks_token(ctx, id, &len);
        res[2u * id] = (int32_t)toks_id_flags(ctx, id);
        res[2u * id + 1u] = s != NULL ? (int32_t)toks_token_to_id(ctx, s, len) : -1000;
    }
    for (uint32_t i = 0; i < k; i++) {
        uint32_t len;
        if (!rd_u32(&len)) { wr_i64(-999); free(res); return; }
        uint8_t *b = slurp_stdin(len);
        if (!b) { wr_i64(-999); free(res); return; }
        res[2u * (uint64_t)info.n_ids + i] = (int32_t)toks_token_to_id(ctx, b, len);
        free(b);
    }
    reply(0, res, 4u * n);
    free(res);
}

// one stream through k pushes and a flush (see the protocol above)
static void do_stream(uint32_t flags) {
    uint32_t k;
    if (!rd_u32(&k)) { wr_i64(-999); return; }
    uint8_t *res = NULL;                                // the reply: per call, u32 len | bytes
    uint64_t rlen = 0, rcap = 0;
    int64_t status = 0;
    int grow = (flags & 2u) != 0u;                      // flag 2: grow a caller's hold on TOKS_E_LIMIT
    uint8_t *hold = NULL, *tmp = NULL;                  // st's hold (hcap bytes, 0: st's own) and the trials' copy
    uint64_t hcap = 0;
    toks_stream st, st2, st3;
    toks_stream_init(ctx, &st, flags & ~2u);
    for (uint32_t c = 0; c <= k; c++) {                 // k pushes, then the flush (c == k)
        uint32_t n = 0;
        uint32_t *ids = NULL;
        if (c < k) {
            if (!rd_u32(&n)) { wr_i64(-999); free(res); return; }
            ids = malloc(4 * (size_t)(n ? n : 1));
            if (!rd_full((uint8_t *)ids, 4 * (size_t)n)) { wr_i64(-999); free(ids); free(res); return; }
        }
        if (status == 0) {
            uint64_t bound;
            uint8_t *a = NULL, *b = NULL;
            int64_t r;
            int grown = 0;
            for (;;) {                                  // a refusal grows the hold, then the same call again (once)
                bound = toks_stream_bound(ctx, c < k ? n : 0u) + 3 * hcap;   // bound(0) covers the flush
                a = malloc(bound + 1), b = malloc(bound + 1);
                memcpy(&st2, &st, sizeof st);
                if (hcap) toks_stream_hold(ctx, &st2, tmp, hcap);           // the trial writes its own copy
                r = c < k ? toks_stream_push(ctx, &st2, ids, n, a, bound) : toks_stream_flush(ctx, &st2, a, bound);
                if (r != TOKS_E_LIMIT || !grow || c == k || grown) break;
                grown = 1;
                uint64_t cur = hcap ? hcap : 44, ncap = 2 * cur > cur + n ? 2 * cur : cur + n;
                uint8_t *nh = malloc(ncap), *nt = realloc(tmp, ncap);
                tmp = nt;
                free(a);
                free(b);
                if (!nh || !nt || toks_stream_hold(ctx, &st, nh, ncap) < 0) { free(nh); a = b = NULL; r = -992; break; }
                free(hold);                             // the old hold, once the move returned
                hold = nh, hcap = ncap;
            }
            if (r >= 0 && (uint64_t)r > bound) r = -989;
            if (r > 0) {                                // one byte less: E_CAP, st unchanged, the exact prefix
                memcpy(&st2, &st, sizeof st);
                if (hcap) toks_stream_hold(ctx, &st2, tmp, hcap);
                memcpy(&st3, &st2, sizeof st2);
                memset(b, 0x5A, (size_t)r);
                int64_t e = c < k ? toks_stream_push(ctx, &st2, ids, n, b, (uint64_t)r - 1)
                                  : toks_stream_flush(ctx, &st2, b, (uint64_t)r - 1);
                if (e != TOKS_E_CAP || memcmp(&st2, &st3, sizeof st2) != 0 || memcmp(a, b, (size_t)r - 1) != 0) r = -990;
            }
            if (r >= 0) {                               // the real call, at exactly its capacity
                int64_t x = c < k ? toks_stream_push(ctx, &st, ids, n, b, (uint64_t)r)
                                  : toks_stream_flush(ctx, &st, b, (uint64_t)r);
                if (x != r || memcmp(a, b, (size_t)r) != 0) r = -991;
            }
            if (r < 0) {
                status = r;
            } else {
                if (rlen + 4 + (uint64_t)r > rcap) {
                    rcap = 2 * (rlen + 4 + (uint64_t)r) + 256;
                    res = realloc(res, rcap);
                }
                uint32_t len = (uint32_t)r;
                memcpy(res + rlen, &len, 4);
                memcpy(res + rlen + 4, b, (size_t)r);
                rlen += 4 + (uint64_t)r;
            }
            free(a);
            free(b);
        }
        free(ids);
    }
    if (status != 0) reply(status, NULL, 0); else reply(0, res, rlen);
    free(res);
    free(hold);
    free(tmp);
}

int main(void) {
    for (;;) {
        uint32_t magic, op, flags, plen;
        if (!rd_u32(&magic)) break;                     // clean EOF
        if (magic != 0x31524B54u) { wr_i64(-997); return 2; }   // "TKR1"
        if (!rd_u32(&op) || !rd_u32(&flags) || !rd_u32(&plen)) { wr_i64(-999); return 2; }
        char *path = malloc(plen + 1);
        if (!rd_full((uint8_t *)path, plen)) { wr_i64(-999); return 2; }
        path[plen] = 0;

        if (op == 1) {                                   // LOAD
            if (ctx) { toks_unload(ctx); ctx = NULL; }
            free(scr); scr = NULL; scr_len = 0;
            int64_t r = toks_load(&ctx, path, NULL);
            wr_i64(r);
            free(path);
            if (r < 0) return 2;
            continue;
        }
        if (op == 5) {                                   // UNLOAD
            if (ctx) { toks_unload(ctx); ctx = NULL; }
            free(scr); scr = NULL; scr_len = 0;
            wr_i64(0);
            free(path);
            continue;
        }
        if (!ctx) { wr_i64(-996); free(path); return 2; }
        switch (op) {
        case 2: do_ids(flags, 0); break;                // ENCODE
        case 3: do_ids(flags, 1); break;                // PIECES
        case 4: do_decode(flags); break;                // DECODE
        case 6: do_stream(flags); break;                // STREAM
        case 7: do_vocab(); break;                      // VOCAB
        default: wr_i64(-995); free(path); return 2;
        }
        free(path);
    }
    if (ctx) toks_unload(ctx);
    free(scr);
    return 0;
}
