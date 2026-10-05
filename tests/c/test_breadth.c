/*
 * test_breadth.c: the byte-level breadth semantics through include/toks.h (docs/breadth.md), against hf tokenizers
 * 0.23.2's own outputs:
 *   - every fixture of tests/data/breadth (the spellings Split Removed+invert / Isolated+invert, dropout 0,
 *     BatchLongest padding; vocab strings outside the byte-level alphabet; added tokens with lstrip, rstrip,
 *     single_word and normalized in both phases; RobertaProcessing alone and in a Sequence; truncation and
 *     padding, Fixed / a multiple, Right / Left) loads, and
 *     encode in every mode with and without the post-processor, decode of the encodings and decode of every
 *     id alone equal hf's (cases.inc, written by tests/data/breadth/gen.py);
 *   - the \w table (src/gen/rx_word.c) is sorted, disjoint and agrees with fixed probes;
 *   - toks's byte-input rule next to added tokens (SPEC §3.3): an ill-formed byte is neither \s nor \w,
 *     so lstrip / rstrip stop at it and single_word treats it as a boundary.
 * Run from the repository root (make test does).
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif
#include "toks.h"
#include "core.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../data/breadth/cases.inc"

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

typedef struct fx {
    toks_ctx *ctx;
    void *scr;
} fx;

static int fx_open(fx *f, const char *name)
{
    char path[512];
    snprintf(path, sizeof path, "tests/data/breadth/%s.json", name);
    toks_diag d;
    memset(&d, 0, sizeof d);
    toks_load_opts o = { sizeof(toks_load_opts), TOKS_TIER_AUTO, 0u, 0u, &d };
    int64_t r = toks_load(&f->ctx, path, &o);
    CHECK(r == 0, "%s: load %" PRId64 " (%s)", name, r, d.what);
    if (r != 0) { return 0; }
    toks_info inf;
    memset(&inf, 0, sizeof inf);
    inf.size = (uint32_t)sizeof inf;
    int generic = strncmp(name, "gen_", 4) == 0;           /* the gen_* fixtures run on the generic engine, every */
    CHECK(toks_get_info(f->ctx, &inf) == 0 && ((inf.paths & TOKS_PATH_SCAN) == 0u) == generic,   /* other on a template */
          "%s: paths 0x%x", name, inf.paths);
    uint64_t sb = toks_scratch_bytes(f->ctx, 1u << 16, 0u);
    f->scr = malloc((size_t)sb);
    CHECK(f->scr != NULL && toks_scratch_init(f->ctx, f->scr, sb, 0u) == 0, "%s: scratch", name);
    return 1;
}

static void fx_close(fx *f)
{
    toks_unload(f->ctx);
    free(f->scr);
}

static const uint32_t *find_ids(const char *tok, uint32_t text, uint32_t flags, uint32_t *n)
{
    for (int k = 0; k < BR_N; k++) {
        if (strcmp(BR[k].tok, tok) == 0 && BR[k].text == text && BR[k].flags == flags) {
            *n = BR[k].n;
            return BR[k].v;
        }
    }
    *n = 0;
    return NULL;
}

static const struct br_dec1 *dec1_of(const char *name, uint32_t *n)
{
#define DEC1(x) if (strcmp(name, #x) == 0) { *n = BR_N_IDS_##x; return BR_DEC1_##x; }
    DEC1(added_opts) DEC1(spell_removed) DEC1(spell_isoinv) DEC1(nonalpha) DEC1(roberta) DEC1(roberta_seq)
    DEC1(pp_shapes) DEC1(pp_notype) DEC1(dup_added) DEC1(drop_gpt2) DEC1(drop_nfc) DEC1(trunc_pad) DEC1(trunc_left)
    DEC1(digits_gpt2)
#undef DEC1
    *n = 0;
    return NULL;
}

static void run_fixture(const char *name)
{
    fx f;
    if (!fx_open(&f, name)) { return; }
    uint32_t *out = (uint32_t *)malloc(4u * 4096u);
    uint64_t n_cases = 0;
    for (int k = 0; k < BR_N; k++) {
        if (strcmp(BR[k].tok, name) != 0) { continue; }
        const char *t = BR_TEXT[BR[k].text];
        uint64_t len = BR_LEN[BR[k].text], n = BR[k].n;
        int64_t r = toks_encode(f.ctx, t, len, BR[k].flags, out, 4096u, f.scr);
        int ok = r == (int64_t)n && memcmp(out, BR[k].v, (size_t)n * 4u) == 0;
        CHECK(ok, "%s encode text %u flags %u: got %" PRId64 " ids, want %u", name, BR[k].text, BR[k].flags, r, BR[k].n);
        if (!ok && r > 0 && failures < 40) {
            fprintf(stderr, "  text \"");
            for (uint64_t i = 0; i < len; i++) { fprintf(stderr, (t[i] >= 0x20 && t[i] < 0x7F) ? "%c" : "\\x%02x", (unsigned char)t[i]); }
            fprintf(stderr, "\"\n  got ");
            for (int64_t i = 0; i < r && i < 40; i++) { fprintf(stderr, "%u ", out[i]); }
            fprintf(stderr, "\n  want ");
            for (uint64_t i = 0; i < n && i < 40; i++) { fprintf(stderr, "%u ", BR[k].v[i]); }
            fprintf(stderr, "\n");
        }
        n_cases++;
        if ((BR[k].flags & 4u) != 0u) { continue; }
        /* pieces: the oracle's ends in the normalized stream (an rstrip overlap re-covers bytes there) */
        r = toks_pieces(f.ctx, t, len, BR[k].flags & 3u, out, 4096u, f.scr);
        CHECK(r == (int64_t)BR[k].np && memcmp(out, BR[k].p, (size_t)BR[k].np * 4u) == 0,
              "%s pieces text %u flags %u: got %" PRId64 " ends, want %u", name, BR[k].text, BR[k].flags, r, BR[k].np);
    }
    uint8_t *dec = (uint8_t *)malloc(65536u);
    for (int k = 0; k < BR_N_DEC; k++) {
        if (strcmp(BR_DEC[k].tok, name) != 0) { continue; }
        uint32_t n;
        const uint32_t *ids = find_ids(name, BR_DEC[k].text, 0u, &n);
        int64_t r = toks_decode(f.ctx, ids, n, BR_DEC[k].skip ? TOKS_SKIP_SPECIAL : 0u, dec, 65536u);
        CHECK(r == (int64_t)BR_DEC[k].n && memcmp(dec, BR_DEC[k].s, BR_DEC[k].n) == 0,
              "%s decode text %u skip %u: %" PRId64 " bytes, want %u", name, BR_DEC[k].text, BR_DEC[k].skip, r, BR_DEC[k].n);
    }
    uint32_t n_ids = 0;
    const struct br_dec1 *d1 = dec1_of(name, &n_ids);
    for (uint32_t id = 0; id < n_ids; id++) {
        int64_t r = toks_decode(f.ctx, &id, 1u, 0u, dec, 65536u);
        CHECK(r == (int64_t)d1[id].n && memcmp(dec, d1[id].s, d1[id].n) == 0, "%s decode id %u alone: %" PRId64, name, id, r);
    }
    printf("fixture %-14s ok: %" PRIu64 " encode cases, %u ids decoded alone\n", name, n_cases, n_ids);
    free(dec);
    free(out);
    fx_close(&f);
}

/* ---- the \w table ------------------------------------------------------------------------------------- */

static void test_word_table(void)
{
    extern const uint32_t toks_rx_word[][2];
    uint32_t prev_hi = 0, n = 0;
    for (uint32_t i = 0; i < 796u; i++) {
        uint32_t lo = toks_rx_word[i][0], hi = toks_rx_word[i][1];
        CHECK(lo <= hi && hi <= 0x10FFFFu && (i == 0u || lo > prev_hi + 1u), "rx_word range %u", i);
        CHECK(toks_is_regex_word(lo) && toks_is_regex_word(hi), "rx_word ends %u", i);
        CHECK(lo == 0u || !toks_is_regex_word(lo - 1u), "rx_word before %u", i);
        CHECK(!toks_is_regex_word(hi + 1u) || hi + 1u > 0x10FFFFu, "rx_word after %u", i);
        n += hi - lo + 1u;
        prev_hi = hi;
    }
    CHECK(n == 144667u, "rx_word: %u code points", n);
    static const uint32_t word[] = { '0', '9', 'A', 'z', '_', 0xAA, 0xB5, 0xE9, 0x0301, 0x0663, 0x200C, 0x200D,
                                     0x203F, 0x4E00, 0x2160 };
    static const uint32_t not_word[] = { ' ', '-', '.', '@', 0x7F, 0xA0, 0xD7, 0x2028, 0x3000, 0x20AC, 0x1F600,
                                         0xFFFD };
    for (uint32_t i = 0; i < sizeof word / sizeof word[0]; i++) { CHECK(toks_is_regex_word(word[i]), "\\w U+%04X", word[i]); }
    for (uint32_t i = 0; i < sizeof not_word / sizeof not_word[0]; i++) {
        CHECK(!toks_is_regex_word(not_word[i]), "not \\w U+%04X", not_word[i]);
    }
}

/* ---- ill-formed bytes beside added tokens (SPEC §3.3; hf cannot take these inputs) ---------------- */

static int64_t pieces(fx *f, const char *t, uint64_t len, uint32_t *ends)
{
    return toks_pieces(f->ctx, t, len, 0u, ends, 64u, f->scr);
}

static void test_ill_formed(void)
{
    fx f;
    if (!fx_open(&f, "added_opts")) { return; }
    uint32_t e[64];
    /* "\x85<mask>": 0x85 alone is not U+0085 (NEL, \s): lstrip stops, the token starts at 1 */
    int64_t r = pieces(&f, "\x85<mask>", 7u, e);
    CHECK(r == 2 && e[0] == 1u && e[1] == 7u, "lstrip over an ill-formed 0x85: %" PRId64, r);
    /* "\xC2\x85<mask>": the well-formed NEL is \s: one piece, the token swallows it */
    r = pieces(&f, "\xC2\x85<mask>", 8u, e);
    CHECK(r == 1 && e[0] == 8u, "lstrip over U+0085: %" PRId64, r);
    /* "<|end|>\xA0x": rstrip stops at the ill-formed 0xA0 (U+00A0 would be \s) */
    r = pieces(&f, "<|end|>\xA0x", 9u, e);
    CHECK(r >= 2 && e[0] == 7u, "rstrip over an ill-formed 0xA0: %" PRId64 " first end %u", r, e[0]);
    r = pieces(&f, "<|end|>\xC2\xA0x", 10u, e);
    CHECK(r >= 2 && e[0] == 9u, "rstrip over U+00A0: %" PRId64 " first end %u", r, e[0]);
    /* "[W]" single_word: an ill-formed neighbour is not \w (the match stays), a well-formed letter is */
    uint32_t ids[16], w_id = 0;
    CHECK(toks_encode(f.ctx, "[W]", 3u, 4u, &w_id, 1u, f.scr) == 1, "[W] alone");
    int64_t a = toks_encode(f.ctx, "\xE9[W]", 4u, 4u, ids, 16u, f.scr);
    CHECK(a == 2 && ids[1] == w_id, "single_word after an ill-formed 0xE9 keeps the match: %" PRId64, a);
    a = toks_encode(f.ctx, "[W]\xE9", 4u, 4u, ids, 16u, f.scr);
    CHECK(a == 2 && ids[0] == w_id, "single_word before an ill-formed 0xE9 keeps the match: %" PRId64, a);
    a = toks_encode(f.ctx, "\xC3\xA9[W]", 5u, 4u, ids, 16u, f.scr);
    int kept = 0;
    for (int64_t i = 0; i < a; i++) { kept |= ids[i] == w_id; }
    CHECK(a > 0 && !kept, "single_word after U+00E9 (\\w) drops the match");
    fx_close(&f);
}

/* ---- split planning beside strip / single_word tokens (SPEC §5.2) ---------------------------------- */

/* toks_split_points' cuts stay exact with every breadth feature loaded: for each fixture, mode and n_want, the
 * parts encoded alone (no post-processing, every part after the first with TOKS_CONTINUATION) concatenate to
 * the whole input's ids. Inputs: each text alone and runs of consecutive texts joined (longer inputs give the
 * planner room: D = len / (4 n_want)). */
static void test_split_exact(void)
{
    static const uint32_t WANT[] = { 2u, 3u, 5u, 8u, 16u };
    uint8_t *buf = (uint8_t *)malloc(8192u);
    uint32_t *whole = (uint32_t *)malloc(4u * 8192u), *parts = (uint32_t *)malloc(4u * 8192u);
    uint64_t offs[32];
    long n_inputs = 0, n_cuts = 0;
    for (int fi = 0; fi < BR_N_FIXTURES; fi++) {
        fx f;
        if (!fx_open(&f, BR_FIXTURES[fi])) { continue; }
        for (int ti = 0; ti < BR_N_TEXTS; ti++) {
            for (int join = 1; join <= 12; join += 11) {       /* the text alone, and twelve texts from it on */
                uint64_t len = 0;
                for (int j = 0; j < join && len + BR_LEN[(ti + j) % BR_N_TEXTS] < 8192u; j++) {
                    memcpy(buf + len, BR_TEXT[(ti + j) % BR_N_TEXTS], BR_LEN[(ti + j) % BR_N_TEXTS]);
                    len += BR_LEN[(ti + j) % BR_N_TEXTS];
                }
                for (uint32_t mode = 0u; mode < 3u; mode++) {
                    uint32_t fl = mode | TOKS_NO_POSTPROCESS;
                    int64_t nw = toks_encode(f.ctx, buf, len, fl, whole, 8192u, f.scr);
                    for (uint32_t w = 0; w < sizeof WANT / sizeof WANT[0]; w++) {
                        int64_t nc = toks_split_points(f.ctx, buf, len, fl, WANT[w], offs, 32u, NULL);
                        CHECK(nc >= 0, "%s split_points text %d mode %u: %" PRId64, BR_FIXTURES[fi], ti, mode, nc);
                        if (nc <= 0) { continue; }
                        uint64_t at = 0, np = 0;
                        for (int64_t c = 0; c <= nc; c++) {
                            uint64_t end = (c < nc) ? offs[c] : len;
                            int64_t r = toks_encode(f.ctx, buf + at, end - at, fl | (at != 0u ? TOKS_CONTINUATION : 0u),
                                                    parts + np, 8192u - np, f.scr);
                            CHECK(r >= 0, "%s part encode: %" PRId64, BR_FIXTURES[fi], r);
                            if (r < 0) { break; }
                            np += (uint64_t)r;
                            at = end;
                        }
                        CHECK((int64_t)np == nw && memcmp(parts, whole, (size_t)np * 4u) == 0,
                              "%s split text %d join %d mode %u n_want %u: %" PRId64 " cuts, parts give %" PRIu64
                              " ids, whole %" PRId64, BR_FIXTURES[fi], ti, join, mode, WANT[w], nc, np, nw);
                        n_cuts += nc;
                    }
                    n_inputs++;
                }
            }
        }
        fx_close(&f);
    }
    printf("split: %ld inputs x modes, %ld certified cuts, parts == whole\n", n_inputs, n_cuts);
    free(buf);
    free(whole);
    free(parts);
}

/* ---- dropped bytes on toks's byte input (SPEC §3.3; docs/breadth.md §4) ------------------------------- */

/* drop_gpt2's vocab lacks F1's char: an ill-formed lone F1 (hf cannot take one) is a P atom, its own gpt-2 piece,
 * and the model drops it like any byte the vocab lacks: no id, its neighbours' pieces unchanged. */
static void test_drop_bytes(void)
{
    fx f;
    if (!fx_open(&f, "drop_gpt2")) { return; }
    uint32_t a[16], b[16], ab[32];
    int64_t na = toks_encode(f.ctx, "abc", 3u, 0u, a, 16u, f.scr);
    int64_t nb = toks_encode(f.ctx, "xyz", 3u, 0u, b, 16u, f.scr);
    CHECK(toks_encode(f.ctx, "\xF1", 1u, 0u, ab, 32u, f.scr) == 0, "a lone dropped ill-formed byte emits nothing");
    CHECK(toks_encode(f.ctx, "\xF1\xF1\xF1", 3u, 0u, ab, 32u, f.scr) == 0, "a run of dropped bytes emits nothing");
    int64_t n = toks_encode(f.ctx, "abc\xF1xyz", 7u, 0u, ab, 32u, f.scr);
    CHECK(na > 0 && nb > 0 && n == na + nb && memcmp(ab, a, (size_t)na * 4u) == 0 &&
          memcmp(ab + na, b, (size_t)nb * 4u) == 0, "abc F1 xyz = abc, xyz: %" PRId64, n);
    uint32_t e[8];
    CHECK(toks_pieces(f.ctx, "abc\xF1xyz", 7u, 0u, e, 8u, f.scr) == 3 && e[0] == 3u && e[1] == 4u && e[2] == 7u,
          "the pre-tokenizer still sees the dropped byte");
    fx_close(&f);
}

int main(void)
{
    for (int i = 0; i < BR_N_FIXTURES; i++) { run_fixture(BR_FIXTURES[i]); }
    test_word_table();
    test_ill_formed();
    test_split_exact();
    test_drop_bytes();
    printf("test_breadth: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
