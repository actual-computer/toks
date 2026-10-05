/*
 * test_spm.c: sentencepiece-style bpe end to end (config.c route, compile.c, spm_build.c, spm_c.c, the driver;
 * docs/algorithms/spm_bpe.md):
 *   - hand fixtures tests/data/spm/<name>.json against <name>.expect (gen.py wrote both from hf tokenizers
 *     0.23.2 and checked them against tests/model/spm_model.py): S gaps through toks_encode (mode NONE, no
 *     post-processing; at_start 0 = TOKS_CONTINUATION) and through toks_spm_encode on whole pieces without
 *     the cache (§5.5: the same ids), P the model alone, D toks_decode, K toks_decode with TOKS_SKIP_SPECIAL (D and K
 *     through the stream decoder too, an id a push and one push), A the file as is through toks_encode (modes ALL and
 *     NONSPECIAL);
 *   - refuse.txt: every file there is refused by toks_load_mem_copy with TOKS_E_UNSUPPORTED, the reason named;
 *   - certified words + cache vs whole pieces on random strings over each fixture's alphabet;
 *   - gemma 4, the pinned file ($TOKS_TOKENIZER_CACHE, default ~/.cache/toks/tokenizers/gemma4), when present:
 *     loaded through toks_load, eight hf goldens through encode (modes NONE and ALL) and decode;
 *   - llama 2 and llama 1, the pinned files, when present: §8.1's normalized specials kept by skip_special.
 * Run from the repository root (make test does).
 */
#include "core.h"
#include "bpe.h"
#include "spm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lines.inc"

static int fails;
static uint64_t checks;
#define CHECK(c, ...)                                                                                      \
    do {                                                                                                   \
        checks++;                                                                                          \
        if (!(c)) {                                                                                        \
            if (++fails <= 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
        }                                                                                                  \
    } while (0)

static uint8_t *read_all(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n + 1u);
    if (b == NULL || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return b;
}

static toks_ctx *load_bytes(const uint8_t *data, uint64_t len, toks_diag *diag)
{
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = diag;
    toks_ctx *ctx = NULL;
    if (toks_load_mem_copy(&ctx, data, len, &o) != 0) { return NULL; }
    return ctx;
}

#define MAXT (1u << 16)
static uint8_t *SCR;
static uint64_t SCR_N;
static uint8_t WORK[TOKS_SPM_WORK_BYTES(MAXT + 1u) + 64u];

static void scratch_for(const toks_ctx *ctx, uint32_t flags)
{
    free(SCR);
    SCR_N = toks_scratch_bytes(ctx, MAXT, flags);
    SCR = (uint8_t *)malloc((size_t)SCR_N);
    if (SCR == NULL || toks_scratch_init(ctx, SCR, SCR_N, flags) != 0) { printf("scratch\n"); exit(2); }
}

static uint64_t unhex(const char *h, uint8_t *out)
{
    if (h[0] == '-') { return 0; }
    uint64_t n = 0;
    for (; h[0] && h[1] && h[0] != ' ' && h[0] != '\n'; h += 2) {
        unsigned v;
        sscanf(h, "%2x", &v);
        out[n++] = (uint8_t)v;
    }
    return n;
}

static uint64_t parse_ids(const char *s, uint32_t *ids)
{
    uint64_t n = 0;
    char *e;
    for (;;) {
        while (*s == ' ') { s++; }
        if (*s < '0' || *s > '9') { break; }
        ids[n++] = (uint32_t)strtoul(s, &e, 10);
        s = e;
    }
    return n;
}

/* a gap through the public encode (mode NONE, no pp) */
static int64_t enc(const toks_ctx *ctx, const uint8_t *t, uint64_t n, int at_start, uint32_t *out, uint64_t cap)
{
    uint32_t fl = TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS | (at_start ? 0u : TOKS_CONTINUATION);
    return toks_encode(ctx, t, n, fl, out, cap, SCR);
}

/* the same gap on whole pieces, no cache (the §5.5 control) */
static uint64_t enc_whole(const toks_ctx *ctx, const uint8_t *t, uint64_t n, int at_start, uint32_t *out, uint64_t cap)
{
    return toks_spm_encode(&ctx->t, ctx->spm, t, n, at_start, out, cap, 0, NULL, 0u, 0u, WORK,
                           TOKS_SPM_NOCUTS | TOKS_SPM_NOCACHE);
}

/* the stream decoder on ids[0, n): pushes of step ids (step >= 1 when n > 0), then the flush; the bytes written in
 * all, or the first error (SPEC §3.4: stream == batch) */
static int64_t stream_dec(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint64_t step,
                          uint8_t *out, uint64_t cap)
{
    toks_stream st;
    toks_stream_init(ctx, &st, flags);
    uint64_t w = 0;
    for (uint64_t i = 0; i < n; i += step) {
        int64_t r = toks_stream_push(ctx, &st, ids + i, n - i < step ? n - i : step, out + w, cap - w);
        if (r < 0) { return r; }
        w += (uint64_t)r;
    }
    int64_t r = toks_stream_flush(ctx, &st, out + w, cap - w);
    return r < 0 ? r : (int64_t)(w + (uint64_t)r);
}

static void run_expect(const char *name, const toks_ctx *ctx)
{
    char path[512];
    snprintf(path, sizeof path, "tests/data/spm/%s.expect", name);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "%s missing", path);
    if (f == NULL) { return; }
    static char line[1 << 16];
    static uint8_t text[1 << 14];
    static uint32_t want[1 << 14], got[1 << 14];
    static uint8_t dec[1 << 15];
    td_at at = { path, 0 };
    while (fgets(line, sizeof line, f) != NULL) {
        at.line++;
        if (!td_line(&at, line, sizeof line)) { fails++; break; }
        if (line[0] != 'S' && line[0] != 'P' && line[0] != 'Q' && line[0] != 'D' && line[0] != 'K' && line[0] != 'A') {
            fails += !td_bad(&at, line, "a line kind S, P, Q, D, K or A expected");
            break;
        }
        if ((line[0] == 'D' || line[0] == 'K') && strchr(line, '|') == NULL) {
            fails += !td_bad(&at, line, "'D <ids> | <hex>' expected");
            break;
        }
        if (line[0] == 'S') {
            int at_start = line[2] - '0';
            char *h = line + 4;
            uint64_t n = unhex(h, text);
            char *sp = strchr(h, ' ');
            uint64_t nw = sp ? parse_ids(sp, want) : 0;
            int64_t k = enc(ctx, text, n, at_start, got, 1u << 14);
            CHECK(k == (int64_t)nw && memcmp(got, want, nw * 4u) == 0, "%s S %s: got %lld ids, want %llu", name, h,
                  (long long)k, (unsigned long long)nw);
            uint64_t k2 = enc_whole(ctx, text, n, at_start, got, 1u << 14);
            CHECK(k2 == nw && memcmp(got, want, nw * 4u) == 0, "%s S whole %s: got %llu", name, h, (unsigned long long)k2);
        } else if (line[0] == 'A') {                        /* the file as is: 0 mode ALL, 1 NONSPECIAL without pp */
            uint32_t fl = line[2] == '0' ? TOKS_ADDED_ALL : (TOKS_ADDED_NONSPECIAL | TOKS_NO_POSTPROCESS);
            char *h = line + 4;
            uint64_t n = unhex(h, text);
            char *sp = strchr(h, ' ');
            uint64_t nw = sp ? parse_ids(sp, want) : 0;
            int64_t k = toks_encode(ctx, text, n, fl, got, 1u << 14, SCR);
            CHECK(k == (int64_t)nw && memcmp(got, want, nw * 4u) == 0, "%s A %c %s: got %lld ids, want %llu", name, line[2],
                  h, (long long)k, (unsigned long long)nw);
        } else if (line[0] == 'P') {
            char *h = line + 2;
            uint64_t n = unhex(h, text);
            char *sp = strchr(h, ' ');
            uint64_t nw = sp ? parse_ids(sp, want) : 0;
            uint64_t k = toks_spm_model(&ctx->t, ctx->spm, text, n, got, WORK);
            CHECK(k == nw && memcmp(got, want, nw * 4u) == 0, "%s P %s: got %llu", name, h, (unsigned long long)k);
        } else if (line[0] == 'Q') {
            char *h = line + 2;
            uint64_t n = unhex(h, text);
            char *sp = strchr(h, ' ');
            uint64_t nw = sp ? parse_ids(sp, want) : 0;
            int64_t k = toks_pieces(ctx, text, n, TOKS_ADDED_NONE, got, 1u << 14, SCR);
            CHECK(k == (int64_t)nw && memcmp(got, want, nw * 4u) == 0, "%s Q %s: got %lld pieces", name, h, (long long)k);
        } else {                                            /* D: decode; K: decode with TOKS_SKIP_SPECIAL */
            uint32_t fl = line[0] == 'K' ? TOKS_SKIP_SPECIAL : 0u;
            char *bar = strchr(line, '|');
            uint64_t ni = parse_ids(line + 1, want);
            uint64_t nb = unhex(bar + 2, text);
            int64_t k = toks_decode(ctx, want, ni, fl, dec, sizeof dec);
            CHECK(k == (int64_t)nb && memcmp(dec, text, nb) == 0, "%s %c %s: got %lld bytes, want %llu", name, line[0],
                  line + 2, (long long)k, (unsigned long long)nb);
            if (nb > 3u) {                                  /* a short buffer: the count, and a prefix */
                int64_t k3 = toks_decode(ctx, want, ni, fl, dec, 3u);
                CHECK(k3 == (int64_t)nb && memcmp(dec, text, 3u) == 0, "%s %c cap 3", name, line[0]);
            }
            for (uint64_t step = 1u; step <= 2u; step++) {  /* the stream decoder: one id a push, then one push */
                k = stream_dec(ctx, want, ni, fl, step == 1u ? 1u : ni, dec, sizeof dec);
                CHECK(k == (int64_t)nb && memcmp(dec, text, nb) == 0, "%s %c stream (%s) %s: got %lld bytes", name,
                      line[0], step == 1u ? "an id a push" : "one push", line + 2, (long long)k);
            }
        }
    }
    fclose(f);
}

/* random strings over a small alphabet: certified words + cache == whole pieces; the last 2,000 are long and
 * mostly ASCII (the scan's 8-byte steps, its 64-byte windows and the switches between steps and chars) */
static void cut_fuzz(const char *name, const toks_ctx *ctx, uint64_t seed)
{
    static const char *const AL[] = { "a", "b", "c", "h", "e", "l", "o", "w", "r", "d", "\xe2\x96\x81", " ", "<", ">",
                                      "/", "\xc3\xa9", "\n", "\xf0\x9f\x98\x80", "x", "z" };
    static const uint8_t ASC[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 18, 19 };
    static uint8_t buf[2048];
    static uint32_t ids1[2048], ids2[2048];
    uint64_t s = seed;
    for (int it = 0; it < 6000; it++) {
        uint64_t n = 0;
        int lng = it >= 4000;
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        int len = (int)((s >> 33) % (lng ? 300u : 40u));
        for (int j = 0; j < len; j++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            uint64_t r = s >> 33;
            const char *a = (lng && (r & 15u) != 0u) ? AL[ASC[(r >> 4) % sizeof ASC]] : AL[r % (sizeof AL / sizeof AL[0])];
            uint64_t al = strlen(a);
            memcpy(buf + n, a, al);
            n += al;
        }
        int at_start = (int)((s >> 20) & 1u);
        uint64_t k2 = enc_whole(ctx, buf, n, at_start, ids2, 2048);
        for (int rep = 0; rep < 2; rep++) {                 /* the second pass hits what the first filled */
            int64_t k1 = enc(ctx, buf, n, at_start, ids1, 2048);
            CHECK(k1 >= 0 && (uint64_t)k1 == k2 && memcmp(ids1, ids2, k2 * 4u) == 0, "%s cut fuzz #%d/%d: words %lld whole %llu",
                  name, it, rep, (long long)k1, (unsigned long long)k2);
        }
    }
}

static const char *const FIXTURES[] = {
    "gemma4like", "llamalike", "mistrallike", "meta_always_split", "meta_never_split", "unk_fused", "unk_unfused",
    "no_unk_no_bf", "merge_order", "pm_order", "ignore_merges", "holes_nodec", "space_in_vocab", "replace_chain",
    "norm_specials", "holes_added",
};

static void fixtures(void)
{
    for (size_t i = 0; i < sizeof FIXTURES / sizeof FIXTURES[0]; i++) {
        char path[512];
        snprintf(path, sizeof path, "tests/data/spm/%s.json", FIXTURES[i]);
        uint64_t len;
        uint8_t *data = read_all(path, &len);
        CHECK(data != NULL, "%s missing", path);
        if (data == NULL) { continue; }
        toks_diag dg;
        memset(&dg, 0, sizeof dg);
        toks_ctx *ctx = load_bytes(data, len, &dg);
        free(data);
        CHECK(ctx != NULL, "%s: load %lld (%s)", FIXTURES[i], (long long)dg.code, dg.what);
        if (ctx == NULL) { continue; }
        CHECK(ctx->spm != NULL && ctx->t.algo == TOKS_ALGO_BPE_SPM, "%s: not routed to spm", FIXTURES[i]);
        scratch_for(ctx, 0u);
        run_expect(FIXTURES[i], ctx);
        cut_fuzz(FIXTURES[i], ctx, 1234567u + i);
        toks_unload(ctx);
    }
}

static void refusals(void)
{
    FILE *f = fopen("tests/data/spm/refuse.txt", "rb");
    CHECK(f != NULL, "refuse.txt missing");
    if (f == NULL) { return; }
    char line[1024];
    td_at at = { "tests/data/spm/refuse.txt", 0 };
    while (fgets(line, sizeof line, f) != NULL) {
        at.line++;
        if (!td_line(&at, line, sizeof line)) { fails++; break; }
        char *tab = strchr(line, '\t');
        if (tab == NULL) { fails += !td_bad(&at, line, "'<file>\\t<reason>[\\t<code>]' expected"); break; }
        *tab = 0;
        char *why = tab + 1, *col = strchr(why, '\t');   /* an optional third column: the code (else UNSUPPORTED) */
        int64_t code = (col != NULL) ? strtoll(col + 1, NULL, 10) : TOKS_E_UNSUPPORTED;
        if (col != NULL) { *col = 0; }
        why[strcspn(why, "\n")] = 0;
        char path[512];
        snprintf(path, sizeof path, "tests/data/spm/%s", line);
        uint64_t len;
        uint8_t *data = read_all(path, &len);
        CHECK(data != NULL, "%s missing", path);
        if (data == NULL) { continue; }
        toks_diag dg;
        memset(&dg, 0, sizeof dg);
        toks_ctx *ctx = load_bytes(data, len, &dg);
        CHECK(ctx == NULL && dg.code == code && strcmp(dg.what, why) == 0, "%s: got %lld (%s), want '%s'",
              line, (long long)dg.code, dg.what, why);
        toks_unload(ctx);
        free(data);
    }
    fclose(f);
}

/* gemma 4 (google/gemma-4-26B-A4B-it, sha256 cc8d3a0c...): hf 0.23.2 ids with added_tokens [] and no pp */
static const struct { const char *hex; uint32_t ids[16]; uint32_t n; } G4[] = {
    { "48656c6c6f20776f726c64", { 9259, 1902 }, 2 },
    { "202074776f202073706163657320", { 138, 13498, 138, 35220, 236743 }, 5 },
    { "68c3a96c6c6f20f09f988020e4b8ade69687", { 236754, 7869, 844, 163543, 17346, 237364 }, 6 },
    { "3c307834313e203c756e6b3e", { 236820, 236771, 236781, 236812, 236770, 236813, 655, 4984, 236813 }, 9 },
    { "613e20203c2f623e", { 236746, 236813, 138, 954, 236763, 236813 }, 6 },
    { "090964656620662878293a0a0972657475726e20780a", { 255969, 2063, 517, 236769, 236781, 1473, 107, 255968, 2060, 1123, 107 }, 11 },
    { "e29681e2968178", { 138, 236781 }, 2 },
    { "f3a08181f48fbfbd", { 481, 398, 367, 367, 482, 381, 429, 427 }, 8 },
};

static void gemma4(void)
{
    const char *dir = getenv("TOKS_TOKENIZER_CACHE");
    char path[1024];
    if (dir != NULL) { snprintf(path, sizeof path, "%s/gemma4", dir); }
    else if (getenv("HOME") != NULL) { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/gemma4", getenv("HOME")); }
    else { return; }
    FILE *probe = fopen(path, "rb");
    if (probe == NULL) { printf("gemma4: %s absent, skipped\n", path); return; }
    fclose(probe);
    toks_diag dg;
    memset(&dg, 0, sizeof dg);
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &dg;
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, path, &o);
    CHECK(r == 0, "gemma4: toks_load %lld (%s)", (long long)r, dg.what);
    if (r != 0) { return; }
    int64_t k;
    CHECK(ctx->t.n_ids == 262144u && ctx->t.n_merges == 514906u, "gemma4 sizes %u %u", ctx->t.n_ids, ctx->t.n_merges);
    scratch_for(ctx, 0u);
    static uint8_t text[256], dec[1024];
    static uint32_t got[256];
    for (size_t i = 0; i < sizeof G4 / sizeof G4[0]; i++) {
        uint64_t n = unhex(G4[i].hex, text);
        k = enc(ctx, text, n, 1, got, 256);
        CHECK(k == (int64_t)G4[i].n && memcmp(got, G4[i].ids, G4[i].n * 4u) == 0, "gemma4 %s: got %lld", G4[i].hex, (long long)k);
        k = toks_encode(ctx, text, n, TOKS_ADDED_ALL, got, 256, SCR);   /* [$A] adds none */
        if (i == 3u) {                                      /* "<unk>" is a special added token (hf: [... 236743, 3]) */
            static const uint32_t W[8] = { 236820, 236771, 236781, 236812, 236770, 236813, 236743, 3 };
            CHECK(k == 8 && memcmp(got, W, sizeof W) == 0, "gemma4 ALL %s: got %lld", G4[i].hex, (long long)k);
        } else {
            CHECK(k == (int64_t)G4[i].n && memcmp(got, G4[i].ids, G4[i].n * 4u) == 0, "gemma4 ALL %s: got %lld", G4[i].hex, (long long)k);
        }
        int64_t d = toks_decode(ctx, G4[i].ids, G4[i].n, 0, dec, sizeof dec);
        /* the decoder maps U+2581 back to a space: every golden but #6 (literal U+2581 input) round-trips */
        if (i != 6) { CHECK(d == (int64_t)n && memcmp(dec, text, n) == 0, "gemma4 decode %s", G4[i].hex); }
    }
    /* the word cache's wide entries (words of 16..30 bytes, words of 5..7 ids) and longer words: each text twice
     * (fill, then hit) == whole pieces without the cache, at full room and at every short room */
    static const char *const WIDE[] = {
        "            return x\n            return y\n                return z\n",
        "CheckConsistency CheckConsistency PyInterpreterState",
        "end.\r\n\r\nNext.\r\n\r\n\r\n\r\n\r\nLast.\r\n\r\n",
        " ------------------------------------------------------ x",
        "\xe4\xb8\xad\xe6\x96\x87\xe5\xad\x97\xe7\xac\xa6\xe4\xb8\xb2\xe9\x95\xbf\xe4\xb8\x80\xe7\x82\xb9",
    };
    static uint32_t whole[256];
    /* twice: the default 2 MiB cache, then an 8 MiB scratch (one 8 MiB word cache: the whole caches region) */
    for (size_t i = 0; i < 2u * (sizeof WIDE / sizeof WIDE[0]); i++) {
        if (i == sizeof WIDE / sizeof WIDE[0]) {
            scratch_for(ctx, TOKS_SCRATCH_CACHE_MIB(8));
        }
        const char *wi = WIDE[i % (sizeof WIDE / sizeof WIDE[0])];
        uint64_t n = strlen(wi);
        memcpy(text, wi, n);
        uint64_t kw = enc_whole(ctx, text, n, 1, whole, 256);
        for (int rep = 0; rep < 2; rep++) {
            k = enc(ctx, text, n, 1, got, 256);
            CHECK(k == (int64_t)kw && memcmp(got, whole, kw * 4u) == 0, "gemma4 wide #%zu/%d: %lld vs %llu", i, rep,
                  (long long)k, (unsigned long long)kw);
        }
        for (uint64_t cap = 0; cap < kw; cap++) {           /* short room: the exact prefix, the full count */
            memset(got, 0xEE, sizeof got);
            k = enc(ctx, text, n, 1, got, cap);
            CHECK(k == (int64_t)kw && memcmp(got, whole, cap * 4u) == 0 && got[cap] == 0xEEEEEEEEu,
                  "gemma4 wide #%zu cap %llu: %lld", i, (unsigned long long)cap, (long long)k);
        }
    }
    /* a cache of one pair (mask 1) and of two (mask 3): the words fight for four ways (back-bucket hits move to the
     * front, wide entries take back buckets), every text three times == whole pieces without the cache */
    static uint8_t tiny[4u * TOKS_BUCKET];
    static const char *const MIX[] = {
        "the cat and the hat, and the cat sat on the mat; the end. The Cat and The Hat",
        "            return x\n            return y\n                return z\n",
        "end.\r\n\r\nNext.\r\n\r\n CheckConsistency PyInterpreterState the cat",
        "one two three four one two three four five one two three four five six one two three four five six seven",
    };
    for (uint64_t cm = 1u; cm <= 3u; cm += 2u) {
        memset(tiny, 0, sizeof tiny);
        for (int rep = 0; rep < 3; rep++) {
            for (size_t i = 0; i < sizeof MIX / sizeof MIX[0]; i++) {
                uint64_t n = strlen(MIX[i]);
                memcpy(text, MIX[i], n);
                uint64_t kw = enc_whole(ctx, text, n, 1, whole, 256);
                uint64_t g = toks_spm_encode(&ctx->t, ctx->spm, text, n, 1, got, 256, 0, tiny, cm, 7u, WORK, 0u);
                CHECK(g == kw && memcmp(got, whole, kw * 4u) == 0, "gemma4 pair cache mask %llu #%zu/%d: %llu vs %llu",
                      (unsigned long long)cm, i, rep, (unsigned long long)g, (unsigned long long)kw);
            }
        }
    }
    /* pieces: a gap is one piece (§1.1); an added token is its own unit */
    static const uint8_t two[] = "a b<bos>c";
    k = toks_pieces(ctx, two, 9, TOKS_ADDED_ALL, got, 256, SCR);
    CHECK(k == 3 && got[0] == 3u && got[1] == 8u && got[2] == 9u, "gemma4 pieces: %lld [%u %u %u]", (long long)k, got[0], got[1], got[2]);
    /* an added token in the text: <bos> (2) is special; ALL matches it, NONSPECIAL leaves it text */
    static const uint8_t bos_hello[] = "<bos>Hello";
    k = toks_encode(ctx, bos_hello, 10, TOKS_ADDED_ALL, got, 256, SCR);
    CHECK(k == 2 && got[0] == 2u && got[1] == 9259u, "gemma4 <bos>Hello: %lld [%u %u]", (long long)k, got[0], got[1]);
    k = toks_decode(ctx, got, 2, TOKS_SKIP_SPECIAL, dec, sizeof dec);
    CHECK(k == 5 && memcmp(dec, "Hello", 5) == 0, "gemma4 skip special decode: %lld", (long long)k);
    /* the control of §8.1's rule: <bos> is normalized=false, its string is its content: dropped, by the stream too */
    k = stream_dec(ctx, got, 2, TOKS_SKIP_SPECIAL, 1u, dec, sizeof dec);
    CHECK(k == 5 && memcmp(dec, "Hello", 5) == 0, "gemma4 skip special stream: %lld", (long long)k);
    k = stream_dec(ctx, got, 2, 0u, 1u, dec, sizeof dec);
    CHECK(k == 10 && memcmp(dec, "<bos>Hello", 10) == 0, "gemma4 stream: %lld", (long long)k);
    printf("gemma4: %llu spanned pairs (%llu slots), %llu merges dropped as unreachable; tables %llu + %llu bytes\n",
           (unsigned long long)ctx->spm->n_pairs, (unsigned long long)ctx->spm->pairs_mask + 1u,
           (unsigned long long)ctx->spm->n_dropped, (unsigned long long)ctx->mem_tables_len,
           (unsigned long long)ctx->mem_spm_len);
    toks_unload(ctx);
}

/* llama 2 and llama 1 (tests/spm/pins.json): <unk> <s> </s> are special and normalized=true under Prepend ▁ + Replace,
 * so each one's string is "▁" + content, which is no special's content: hf 0.23.2 keeps them under skip_special_tokens
 * (docs/algorithms/spm_bpe.md §8.1), batch and stream. A file that is absent or not the pin prints a SKIP line. */
static void llama(void)
{
    static const struct { const char *name, *sha; } PIN[] = {
        { "llama2", "f7b50bcf6d6672eade5e43514d48e9c1e4e63a56aef7b14acdaca94ce93436f7" },
        { "llama1", "f9ffc4aede0845ab65324ce5dccb823dca2427f9a0710981e5bc2398d73d8162" },
    };
    static const struct { uint32_t ids[3]; uint64_t n; const char *want; } C[] = {   /* hf 0.23.2, skip off and on */
        { { 1, 263 }, 2, "<s> a" }, { { 1, 22172 }, 2, "<s> hello" }, { { 0, 1, 2 }, 3, "<unk> <s> </s>" },
    };
    const char *dir = getenv("TOKS_TOKENIZER_CACHE"), *home = getenv("HOME");
    for (size_t f = 0; f < sizeof PIN / sizeof PIN[0]; f++) {
        char path[1024], hex[65];
        if (dir != NULL) { snprintf(path, sizeof path, "%s/%s", dir, PIN[f].name); }
        else { snprintf(path, sizeof path, "%s/.cache/toks/tokenizers/%s", home != NULL ? home : ".", PIN[f].name); }
        toks_ctx *ctx = NULL;
        if (toks_load(&ctx, path, NULL) != 0) { printf("%s: SKIP (not here)\n", PIN[f].name); continue; }
        for (int b = 0; b < 32; b++) { snprintf(hex + 2 * b, 3, "%02x", ctx->source_sha256[b]); }
        if (strcmp(hex, PIN[f].sha) != 0) {
            printf("%s: SKIP (sha256 %.16s, not the pin)\n", PIN[f].name, hex);
            toks_unload(ctx);
            continue;
        }
        static uint8_t dec[256];
        int f0 = fails;
        for (size_t c = 0; c < sizeof C / sizeof C[0]; c++) {
            int64_t nw = (int64_t)strlen(C[c].want);
            for (uint32_t fl = 0; fl <= TOKS_SKIP_SPECIAL; fl++) {
                int64_t k = toks_decode(ctx, C[c].ids, C[c].n, fl, dec, sizeof dec);
                CHECK(k == nw && memcmp(dec, C[c].want, (size_t)nw) == 0, "%s decode #%zu flags %u: %lld bytes '%.*s', want '%s'",
                      PIN[f].name, c, fl, (long long)k, (int)(k > 0 && k < 256 ? k : 0), (const char *)dec, C[c].want);
                for (uint64_t step = 1u; step <= C[c].n; step++) {   /* the stream: every split into pushes of step ids */
                    k = stream_dec(ctx, C[c].ids, C[c].n, fl, step, dec, sizeof dec);
                    CHECK(k == nw && memcmp(dec, C[c].want, (size_t)nw) == 0, "%s stream #%zu flags %u step %llu: %lld bytes",
                          PIN[f].name, c, fl, (unsigned long long)step, (long long)k);
                }
            }
        }
        printf("%s: [1, 263] [1, 22172] [0, 1, 2] by decode and stream at flags 0 and 1: %s\n", PIN[f].name,
               fails == f0 ? "<unk> <s> </s> kept, as hf" : "FAILED");
        toks_unload(ctx);
    }
}

/* a scratch moved between a byte-level context and an spm one (toks_scratch_init on the same buffer, same flags):
 * an spm context's word cache spans the whole caches region (api.c run_text), where a byte-level one keeps K5's
 * long-piece buckets and an arena of raw text, so the move must be a first init (toks_scratch_init: off_long says
 * whose the long region is); a move within a family stays O(1). Forged here: a word-cache entry with wrong ids, tagged
 * with the epoch an O(1) move would give, in the long region of a byte-level scratch; after the move the word must
 * get its own ids. */
static void moves(void)
{
    uint64_t lb = 0, ls = 0;
    uint8_t *jb = read_all("tests/data/compile/gpt2style.json", &lb), *js = read_all("tests/data/spm/gemma4like.json", &ls);
    toks_diag dg;
    memset(&dg, 0, sizeof dg);
    toks_ctx *bl = jb != NULL ? load_bytes(jb, lb, &dg) : NULL, *sp = js != NULL ? load_bytes(js, ls, &dg) : NULL;
    free(jb);
    free(js);
    CHECK(bl != NULL && sp != NULL && bl->spm == NULL && sp->spm != NULL, "moves: the two fixtures");
    if (bl == NULL || sp == NULL) { toks_unload(bl); toks_unload(sp); return; }
    uint32_t fl = TOKS_SCRATCH_CACHE_MIB(8);
    uint64_t nb = toks_scratch_bytes(bl, MAXT, fl), ns = toks_scratch_bytes(sp, MAXT, fl), n = nb > ns ? nb : ns;
    uint8_t *scr = (uint8_t *)malloc((size_t)n);
    CHECK(scr != NULL, "moves: scratch");
    if (scr == NULL) { toks_unload(bl); toks_unload(sp); return; }
    toks_scratch *h = toks_scr_header(scr);
    CHECK(toks_scratch_init(sp, scr, n, fl) == 0 && h->epoch == 1u && toks_scratch_init(sp, scr, n, fl) == 0 &&
          h->epoch == 2u && h->off_long == 0u, "moves: spm -> spm is O(1) (epoch %llu)", (unsigned long long)h->epoch);
    CHECK(toks_scratch_init(bl, scr, n, fl) == 0 && h->epoch == 1u && h->off_long != 0u &&
          toks_scratch_init(bl, scr, n, fl) == 0 && h->epoch == 2u, "moves: spm -> byte-level is a first init");
    static uint32_t got[64], want[64];
    static const char L[] = "etaoinshrdlucmfwyp";
    uint32_t in_buckets = 0, in_arena = 0;                  /* forged words in K5's long buckets, in its arena */
    for (uint32_t i = 0; i < 18u * 18u * 18u && (in_buckets < 4u || in_arena < 4u); i++) {   /* bound: 5,832 */
        const uint8_t w[3] = { (uint8_t)L[i % 18u], (uint8_t)L[(i / 18u) % 18u], (uint8_t)L[i / 324u] };
        uint64_t len = 3u;
        uint64_t kw = toks_spm_encode(&sp->t, sp->spm, w, len, 0, want, 64, 0, NULL, 0u, 0u, WORK,
                                      TOKS_SPM_NOCUTS | TOKS_SPM_NOCACHE);
        bpe_key k = bpe_key_at(w, len, 0u, len);            /* the word's short key, no prefix (a continuation) */
        uint64_t b = (toks_spm_whash(k.lo, k.hi) & (toks_scr_caches(8u) / TOKS_BUCKET - 1u)) * TOKS_BUCKET;
        uint64_t sh = toks_scr_short(8u), lb = sh + toks_scr_long_buckets(8u);
        if (b < sh || kw == 0u || kw > 3u || (b < lb ? in_buckets : in_arena) >= 4u) { continue; }
        /* w must be one word of the scan (else its words have other keys): a fill of k at b on a zeroed region */
        CHECK(toks_scratch_init(sp, scr, n, fl) == 0, "moves: spm init");
        memset(scr + h->off_cache, 0, (size_t)toks_scr_caches(8u));
        (void)toks_encode(sp, w, len, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION, got, 64, scr);
        if (memcmp(scr + h->off_cache + b, &k.lo, 8) != 0 || memcmp(scr + h->off_cache + b + 8, &k.hi, 8) != 0) { continue; }
        CHECK(toks_scratch_init(bl, scr, n, fl) == 0 && h->off_long != 0u && h->epoch == 1u, "moves: spm -> byte-level");
        uint32_t val[4], bad[4] = { want[0] ^ 1u, want[0] ^ 1u, want[0] ^ 1u, want[0] ^ 1u };
        bpe_val_pack_tag(val, bad, (uint32_t)kw + 1u, toks_tag_word(h->epoch + 1u));   /* wrong ids: the epoch of an */
        bpe_cache_fill(scr + h->off_cache + b, k, val);                                  /* O(1) move, and of a first */
        bpe_val_pack_tag(val, bad, (uint32_t)kw + 1u, toks_tag_word(1u));               /* init (two ways)           */
        bpe_cache_fill(scr + h->off_cache + b, k, val);
        CHECK(toks_scratch_init(sp, scr, n, fl) == 0 && h->epoch == 1u, "moves: byte-level -> spm is a first init");
        int64_t g = toks_encode(sp, w, len, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS | TOKS_CONTINUATION, got, 64, scr);
        CHECK(g == (int64_t)kw && memcmp(got, want, kw * 4u) == 0, "moves: '%.3s' (%s) after a byte-level scratch: %lld ids, "
              "want %llu", (const char *)w, b < lb ? "long buckets" : "arena", (long long)g, (unsigned long long)kw);
        if (b < lb) { in_buckets++; } else { in_arena++; }
    }
    CHECK(in_buckets == 4u && in_arena == 4u, "moves: %u / %u words forged in the long buckets / the arena", in_buckets,
          in_arena);
    free(scr);
    toks_unload(bl);
    toks_unload(sp);
}

int main(void)
{
    fixtures();
    refusals();
    gemma4();
    llama();
    moves();
    free(SCR);
    printf("%llu checks, %d failures\n", (unsigned long long)checks, fails);
    return fails != 0;
}
