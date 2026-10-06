/* test_check.c: the segment memo's keyed check (src/core/check.c; SPEC §6, kernels.md §7 "the segment memo").
 * - known answers: under a fixed key, the check of sixteen texts (0 .. 12,345 bytes: a group's edges, a block's
 *   edges, several blocks) equals an independent reference's (a python transcription of the definition in check.c's
 *   header: bit-serial carry-less products, python integers mod 2^127 - 1), through the portable carry-less multiply
 *   and, where the cpu has one, PMULL / PCLMULQDQ;
 * - the cpu's path equals the portable one on random texts of every length 0 .. 4,500 and on 64 longer ones;
 * - one flipped bit anywhere in a 4,196-byte text changes the check (sampled: every byte, a rotating bit);
 * - the two record kinds (api.c memo_bytes): where each record sits, what it keeps (the bytes in the ring's first
 *   half, the check past it), how far it moves the ring's position, and that both kinds answer a second pass;
 * - a context without a key (the os gave no randomness at load: memo_keyed 0, the key zeroed) keeps the bytes in every
 *   record (its lap is full at half the ring) and never compares a record that keeps the check;
 * - admission takes the segments master took (a record with its bytes over half the ring is never made).
 * The loaded context's key is replaced by the fixed one (the test owns the context). */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "toks.h"
#include "core.h"
#include "cpu.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { failures++; if (failures < 20) { printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); } } } while (0)

static uint64_t sm_x;
static uint64_t sm_next(void)
{
    uint64_t z = (sm_x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void fill(uint8_t *p, uint64_t n, uint64_t seed)    /* the reference's text(n, seed): little-endian words */
{
    sm_x = seed;
    for (uint64_t i = 0; i < n; i += 8u) {
        uint64_t w = sm_next();
        for (uint64_t j = 0; j < 8u && i + j < n; j++) { p[i + j] = (uint8_t)(w >> (8u * j)); }
    }
}

/* api.c memo_need's two record sizes: the head, the bytes padded to 8 (or the 16-byte check), the ids, 64-aligned */
static uint64_t mem_need_b(uint64_t n, uint64_t k) { return (32u + ((n + 7u) & ~7ull) + 4u * k + 63u) & ~63ull; }
static uint64_t mem_need_c(uint64_t k) { return (48u + 4u * k + 63u) & ~63ull; }

static const struct { uint64_t n, lo, hi; } KAT[] = {     /* tests/c/check_ref.py, key = splitmix64 from 0x746F6B73 */
    {     0, 0x0000000000000000ull, 0x0000000000000000ull },
    {     1, 0x3929D468E090B73Bull, 0x138C32256F8C4BBCull },
    {     7, 0xC0CBB3358DD5B479ull, 0x0E64AD92EC8EBAF7ull },
    {     8, 0x50478A72AD74F675ull, 0x09E758533EB2C120ull },
    {    31, 0xEFE8D12C4921A811ull, 0x4844774BA76AB816ull },
    {    32, 0xE1AC1E984B31C25Eull, 0x235BEEF77210388Dull },
    {    33, 0x50B55F1D3A9EB018ull, 0x5324301612AF9FB0ull },
    {   255, 0x4159FA0EC1954D78ull, 0x4358F7F53CD29F84ull },
    {   256, 0x5AA146E49AF30843ull, 0x2D03576A85CD6E42ull },
    {  1000, 0x06A9EA876DD6A17Eull, 0x0759A9D6057231BBull },
    {  4095, 0x3653221E1E9FF736ull, 0x5F8F8E8E187D3AFEull },
    {  4096, 0xCECFA39EB6BD9E9Aull, 0x0043C72BA737F261ull },
    {  4097, 0x93FC8D1EA3C2C696ull, 0x185973D755958164ull },
    {  8191, 0xC195F811E9AD7865ull, 0x7E7A377A4B820CE0ull },
    {  8192, 0xF5660441336C783Full, 0x601620D839E19617ull },
    { 12345, 0xD85F10FC3C3B2BFFull, 0x7CA5C59CCCC74A4Dull },
};

int main(void)
{
    toks_ctx *ctx = NULL;
    if (toks_load(&ctx, "tests/data/compile/llama3style.json", NULL) != 0) { printf("test_check: load failed\n"); return 1; }
#if defined(TOKS_ARCH_ARM64)
    int hw = TOKS_CPU_HAS(ctx->cpu_features, TOKS_ARM64_PMULL);
#else
    int hw = TOKS_CPU_HAS(ctx->cpu_features, TOKS_X86_PCLMUL);
#endif
    sm_x = 0x746F6B73u;
    for (uint32_t i = 0; i < TOKS_MEMO_KEY_W; i++) { ctx->memo_key[i] = sm_next(); }
    toks_memo_rpow(ctx);

    static uint8_t t[1u << 16], u[1u << 16];
    uint64_t c0[2], c1[2], cases = 0;
    for (size_t i = 0; i < sizeof KAT / sizeof KAT[0]; i++) {
        fill(t, KAT[i].n, 0x5EEDu + KAT[i].n);
        toks_memo_check_with(ctx, 0, t, KAT[i].n, c0);
        CHECK(c0[0] == KAT[i].lo && c0[1] == KAT[i].hi, "kat n %" PRIu64 ": portable %016" PRIx64 "%016" PRIx64, KAT[i].n,
              c0[1], c0[0]);
        if (hw) {
            toks_memo_check_with(ctx, 1, t, KAT[i].n, c1);
            CHECK(c1[0] == KAT[i].lo && c1[1] == KAT[i].hi, "kat n %" PRIu64 ": cpu %016" PRIx64 "%016" PRIx64, KAT[i].n,
                  c1[1], c1[0]);
        }
        cases++;
    }
    if (hw) {
        fill(t, sizeof t, 1u);
        for (uint64_t n = 0; n <= 4500u; n++) {          /* every length, at a moving offset */
            toks_memo_check_with(ctx, 0, t + n % 61u, n, c0), toks_memo_check_with(ctx, 1, t + n % 61u, n, c1);
            CHECK(c0[0] == c1[0] && c0[1] == c1[1], "length %" PRIu64 ": the cpu's path differs from the portable one", n);
            cases++;
        }
        for (uint64_t r = 0; r < 64u; r++) {
            uint64_t n = 4500u + sm_next() % (sizeof t - 4600u);
            toks_memo_check_with(ctx, 0, t + r, n, c0), toks_memo_check_with(ctx, 1, t + r, n, c1);
            CHECK(c0[0] == c1[0] && c0[1] == c1[1], "length %" PRIu64 ": the cpu's path differs from the portable one", n);
            cases++;
        }
    }
    fill(t, 4196u, 7u);
    toks_memo_check(ctx, t, 4196u, c0);
    for (uint64_t i = 0; i < 4196u; i++) {
        memcpy(u, t, 4196u);
        u[i] ^= (uint8_t)(1u << (i % 8u));
        toks_memo_check(ctx, u, 4196u, c1);
        CHECK(c0[0] != c1[0] || c0[1] != c1[1], "byte %" PRIu64 " flipped: same check", i);
        cases++;
    }
    /* the two record kinds (api.c memo_bytes), one segment per call (letters and spaces: no added token) on a 1 MiB
     * memo until its first lap is full: each record sits at the ring position before its call and holds its length and
     * ids, then the bytes while the record with them ends in the first half of the ring, else the bytes' check; the
     * position moves by the record's size of its kind; both kinds appear, and the lap holds more segments than records
     * that all keep the bytes would; a second pass answers every one, with the ids of a memo-off scratch; a text of a
     * check record's length whose locating windows agree with its segment's gets its own ids (counted: differ). Then a
     * context without a key (the os gave none at load: memo_keyed 0, the key zeroed): on the same scratch it answers
     * the records that keep the bytes and never compares one that keeps the check; on a fresh one every record keeps
     * the bytes, so its lap is full at half the ring; two texts of one length and one set of windows each get their
     * own ids */
    {
        static uint8_t seg[600][2048];
        static uint64_t sl[600], rpos[600];
        static uint8_t rbytes[600], live[600];
        static uint32_t ids[8192], ref[8192];
        uint64_t sb = toks_scratch_bytes(ctx, 1u << 16, TOKS_SCRATCH_MEMO_MIB(1));
        uint64_t rb = toks_scratch_bytes(ctx, 1u << 16, TOKS_SCRATCH_MEMO_MIB(0)), mbytes = 0u;
        void *ms = malloc((size_t)sb), *rs = malloc((size_t)rb);
        uint8_t *mm = NULL;
        if (ms == NULL || rs == NULL || toks_scratch_init(ctx, ms, sb, TOKS_SCRATCH_MEMO_MIB(1)) != 0 ||
            toks_scratch_init(ctx, rs, rb, TOKS_SCRATCH_MEMO_MIB(0)) != 0 ||
            (mm = toks_scr_memo(toks_scr_header(ms), &mbytes)) == NULL) {
            printf("test_check: kinds scratches\n");
            return 1;
        }
        const toks_memo_head *mh = (const toks_memo_head *)(const void *)mm;
        uint64_t ring = mbytes - 64u - (mbytes >> 5), nseg = 0u, kinds[2] = { 0u, 0u }, all_b = 0u, n_b = 0u, first_b = 0u;
        for (uint32_t i = 0; i < 600u; i++) {            /* bound: 600 segments, the lap is full well before */
            uint64_t n = 300u + (i * 37u) % 1500u, p0 = mh->pos, c[2];
            fill(seg[i], n, 100u + i);
            for (uint64_t j = 0; j < n; j++) {           /* no 'h': no added token ("hello") can match */
                seg[i][j] = (uint8_t)(j % 9u == 8u ? ' ' : "abcdefgijklmnopqrstuvwxyz"[seg[i][j] % 25u]);
            }
            sl[i] = n;
            int64_t k = toks_encode(ctx, seg[i], n, TOKS_NO_POSTPROCESS, ids, 8192u, ms);
            int64_t kr = toks_encode(ctx, seg[i], n, TOKS_NO_POSTPROCESS, ref, 8192u, rs);
            CHECK(k > 0 && k == kr && memcmp(ids, ref, 4u * (uint64_t)k) == 0, "kinds %u: the memo-off ids", i);
            if (all_b + mem_need_b(n, (uint64_t)k) <= ring) { all_b += mem_need_b(n, (uint64_t)k), n_b++; }
            if (mh->pos == p0) { break; }                /* the lap is full: refused */
            uint64_t o = p0 % ring, at;
            const uint8_t *rec = mm + 64u + (mbytes >> 5) + o;        /* api.c memo_at: the head, then the body */
            uint32_t rn, rlen;
            memcpy(&rn, rec + 20, 4), memcpy(&rlen, rec + 24, 4);
            int bytes = o + mem_need_b(n, (uint64_t)k) <= ring / 2u;
            CHECK(rlen == n && rn == (uint64_t)k, "kinds %u: the record's length %u and ids %u", i, rlen, rn);
            if (bytes) {
                CHECK(memcmp(rec + 32, seg[i], (size_t)n) == 0, "kinds %u: a record in the first half keeps the bytes", i);
                at = 32u + ((n + 7u) & ~7ull);
            } else {
                toks_memo_check(ctx, seg[i], n, c);
                CHECK(memcmp(rec + 32, c, 16) == 0, "kinds %u: a record past the first half keeps the check", i);
                at = 48u;
            }
            CHECK(memcmp(rec + at, ids, 4u * (uint64_t)k) == 0, "kinds %u: the record's ids", i);
            CHECK(mh->pos == p0 + (bytes ? mem_need_b(n, (uint64_t)k) : mem_need_c((uint64_t)k)),
                  "kinds %u: the position moved %" PRIu64 " bytes", i, mh->pos - p0);
            rpos[nseg] = p0, rbytes[nseg] = (uint8_t)bytes;
            kinds[bytes]++, nseg++;
            first_b += bytes && kinds[0] == 0u;   /* the records before the first that keeps the check */
        }
        CHECK(kinds[0] > 0u && kinds[1] > 0u && nseg > n_b, "kinds: %" PRIu64 " records keep the bytes, %" PRIu64
              " the check (%" PRIu64 " that all keep the bytes would fit)", kinds[1], kinds[0], n_b);
        uint64_t alive[2] = { 0u, 0u }, j = 0u;          /* records a slot still names (two ways: a third segment */
        for (uint64_t i = 0; i < nseg; i++) {            /* of a set takes the older record's slot) */
            for (uint64_t w = 0; w < (mbytes >> 5) / 32u; w++) {   /* bound: the slots */
                uint64_t spos;
                uint32_t sn, slen;
                memcpy(&spos, mm + 64u + 32u * w + 8u, 8), memcpy(&sn, mm + 64u + 32u * w + 20u, 4);
                memcpy(&slen, mm + 64u + 32u * w + 24u, 4);
                if (spos == rpos[i] && slen == sl[i] && sn != UINT32_MAX) {
                    live[i] = 1u, alive[rbytes[i]]++;
                    if (!rbytes[i] && sl[i] >= 400u) { j = i; }   /* a live check record of >= 400 bytes */
                    break;
                }
            }
        }
        uint64_t h0 = mh->hits, d0;
        for (uint64_t i = 0; i < nseg; i++) {            /* bound: the records; a missed one could take a live */
            if (!live[i]) { continue; }                  /* one's slot, so only those a slot names */
            int64_t k = toks_encode(ctx, seg[i], sl[i], TOKS_NO_POSTPROCESS, ids, 8192u, ms);
            int64_t kr = toks_encode(ctx, seg[i], sl[i], TOKS_NO_POSTPROCESS, ref, 8192u, rs);
            CHECK(k > 0 && k == kr && memcmp(ids, ref, 4u * (uint64_t)k) == 0, "kinds, again %" PRIu64 ": ids", i);
        }
        CHECK(mh->hits - h0 == alive[0] + alive[1] && alive[0] > 0u && alive[1] > 0u && alive[0] + alive[1] > nseg - nseg / 10u,
              "kinds: the second pass answered %" PRIu64 " of %" PRIu64 " live records (%" PRIu64 " keep the check)",
              mh->hits - h0, alive[0] + alive[1], alive[0]);
        memcpy(u, seg[j], (size_t)sl[j]);                /* windows: [0, 64), from n / 2 - 32 >= 168 on: 100..139 */
        for (uint64_t i = 100u; i < 140u; i++) { u[i] = u[i] == 'a' ? 'b' : 'a'; }
        h0 = mh->hits, d0 = mh->differ;
        int64_t ku = toks_encode(ctx, u, sl[j], TOKS_NO_POSTPROCESS, ids, 8192u, ms);
        int64_t kru = toks_encode(ctx, u, sl[j], TOKS_NO_POSTPROCESS, ref, 8192u, rs);
        CHECK(j > 0u && sl[j] >= 400u && ku > 0 && ku == kru && memcmp(ids, ref, 4u * (uint64_t)ku) == 0 && mh->hits == h0 &&
              mh->differ == d0 + 1u, "kinds: a text with a check record's windows: its own ids (differ %" PRIu64 ")",
              mh->differ - d0);

        memset(ctx->memo_key, 0, sizeof ctx->memo_key), memset(ctx->memo_rpow, 0, sizeof ctx->memo_rpow);
        ctx->memo_keyed = 0u;                            /* load.c, when keygen fails */
        h0 = mh->hits;
        for (uint64_t i = 0; i < nseg; i++) {            /* bound: the records */
            if (!live[i]) { continue; }
            int64_t k = toks_encode(ctx, seg[i], sl[i], TOKS_NO_POSTPROCESS, ids, 8192u, ms);
            int64_t kr = toks_encode(ctx, seg[i], sl[i], TOKS_NO_POSTPROCESS, ref, 8192u, rs);
            CHECK(k > 0 && k == kr && memcmp(ids, ref, 4u * (uint64_t)k) == 0, "no key, the keyed records %" PRIu64, i);
        }
        CHECK(mh->hits - h0 == alive[1], "no key: %" PRIu64 " answers from %" PRIu64 " live records that keep the bytes",
              mh->hits - h0, alive[1]);
        CHECK(toks_scratch_init(ctx, ms, sb, TOKS_SCRATCH_MEMO_MIB(1)) == 0, "no key: init");
        uint64_t nk = 0u;
        for (uint64_t i = 0; i < nseg; i++) {            /* bound: the records */
            uint64_t p0 = mh->pos;
            int64_t k = toks_encode(ctx, seg[i], sl[i], TOKS_NO_POSTPROCESS, ids, 8192u, ms);
            if (mh->pos == p0) { break; }
            const uint8_t *rec = mm + 64u + (mbytes >> 5) + p0 % ring;
            CHECK(k > 0 && memcmp(rec + 32, seg[i], (size_t)sl[i]) == 0 && mh->pos == p0 + mem_need_b(sl[i], (uint64_t)k) &&
                  mh->pos <= ring / 2u, "no key %" PRIu64 ": the record keeps the bytes, in the first half", i);
            nk++;
        }
        CHECK(nk == first_b, "no key: the lap holds %" PRIu64 " records (the keyed one's before its first check: %" PRIu64 ")",
              nk, first_b);
        for (int x = 0; x < 3; x++) {                    /* seg[j], u (one length, one set of windows), seg[j] */
            const uint8_t *tx = x == 1 ? u : seg[j];
            int64_t k = toks_encode(ctx, tx, sl[j], TOKS_NO_POSTPROCESS, ids, 8192u, ms);
            int64_t kr = toks_encode(ctx, tx, sl[j], TOKS_NO_POSTPROCESS, ref, 8192u, rs);
            CHECK(k > 0 && k == kr && memcmp(ids, ref, 4u * (uint64_t)k) == 0, "no key, text %d: its own ids", x);
        }
        CHECK(toks_memo_keygen(ctx) == 0, "a key again");
        ctx->memo_keyed = 1u;
        free(ms), free(rs);
        cases += 8u;
    }
    /* admission takes the segments master took: one whose record with its bytes is over half the ring (4 MiB here) is
     * never recorded; a smaller one is recorded on its first sight and replayed on the next two */
    for (int big = 0; big < 2; big++) {
        uint64_t len = big ? 440000u : 150000u, bb = toks_scratch_bytes(ctx, len, 0u), p1 = 0u, h1 = 0u, h3 = 0u;
        uint8_t *bt = (uint8_t *)malloc((size_t)len);
        uint32_t *bo = (uint32_t *)malloc(4u * (size_t)len + 64u);
        void *bs = malloc((size_t)bb);
        if (bt != NULL && bo != NULL && bs != NULL && toks_scratch_init(ctx, bs, bb, 0u) == 0) {
            fill(bt, len, 11u);
            for (uint64_t i = 0; i < len; i++) { bt[i] = (uint8_t)(i % 7u == 6u ? ' ' : "abcdefgijklmnopqrstuvwxyz"[bt[i] % 25u]); }
            int64_t n1 = toks_encode(ctx, bt, len, TOKS_NO_POSTPROCESS, bo, len + 16u, bs);
            toks_scr_memo_ctr(toks_scr_header(bs), &p1, &h1);
            int64_t n2 = toks_encode(ctx, bt, len, TOKS_NO_POSTPROCESS, bo, len + 16u, bs);
            int64_t n3 = toks_encode(ctx, bt, len, TOKS_NO_POSTPROCESS, bo, len + 16u, bs);
            toks_scr_memo_ctr(toks_scr_header(bs), &p1, &h3);
            uint64_t need = mem_need_b(len, (uint64_t)(n1 > 0 ? n1 : 0));
            uint64_t ring = (4u << 20) - 64u - (4u << 15);                           /* api.c memo_ring, 4 MiB */
            CHECK(n1 > 0 && n1 == n2 && n2 == n3 && (need > ring / 2u) == big,
                  "%s segment: its record with the bytes %" PRIu64 " B, half the ring %" PRIu64 " B",
                  big ? "a big" : "a smaller", need, ring / 2u);
            CHECK(big ? (p1 == 0u && h3 == 0u) : (p1 > 0u && h3 == 2u), "%s segment: ring %" PRIu64 ", %" PRIu64 " hits",
                  big ? "a big" : "a smaller", p1, h3);
        } else {
            CHECK(0, "big scratch");
        }
        free(bt), free(bo), free(bs);
        cases++;
    }
    printf("test_check: %" PRIu64 " cases, the cpu's carry-less multiply %s, %d failures\n", cases,
           hw ? "compared with the portable one" : "absent (portable only)", failures);
    toks_unload(ctx);
    return failures != 0;
}
