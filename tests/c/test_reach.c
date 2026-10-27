/*
 * test_reach.c: SPEC §3.6, structural control-token isolation (src/core/reach.c, docs/algorithms/reach.md).
 *  - red: a byte-level bpe file where an added token IS reachable from plain text is not certified
 *    (toks_info.control_isolation 0), and a text whose TOKS_ADDED_NONE encode gives that id proves the reach real:
 *    the id a merge makes ("a" + "b"), a one-byte token, a whole piece under ignore_merges (gpt-2's regex takes "<|>"
 *    whole; ByteLevel without a regex takes any text whole), a gap between matches under ignore_merges (a letters-only
 *    split leaves "<>" as a gap), the unk a dropped byte gives (an added "<unk>", a vocab without byte 0xFF).
 *  - green: certified files where no added id can come out: an added id outside the model's vocabulary; Nemotron's
 *    shape, ids in the vocabulary under ignore_merges, behind the o200k nemo regex, which can neither match "<s>" or
 *    "[INST]" whole nor leave them as a gap. Each certified file is also searched (every added token spelled in 15
 *    contexts and 2000 random byte strings): no added id in a NONE encode.
 *  - the real files (TOKS_TOKENIZER_CACHE, else ~/.cache/toks/tokenizers; a missing one is SKIP): the pinned results
 *    (the 0.3.3 census, docs/algorithms/reach.md §4), and the same search on each certified one.
 */
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* ---- synthetic byte-level bpe files ------------------------------------------------------------------------- */

static char J[1u << 16];
static size_t jn;

static void put(const char *s) { size_t n = strlen(s); if (jn + n < sizeof J) { memcpy(J + jn, s, n); jn += n; J[jn] = 0; } }

static void put_utf8(uint32_t c)
{
    char b[5] = { 0 };
    if (c < 0x80u) { b[0] = (char)c; }
    else if (c < 0x800u) { b[0] = (char)(0xC0u | (c >> 6)); b[1] = (char)(0x80u | (c & 63u)); }
    else { b[0] = (char)(0xE0u | (c >> 12)); b[1] = (char)(0x80u | ((c >> 6) & 63u)); b[2] = (char)(0x80u | (c & 63u)); }
    if (c == '"' || c == '\\') { put("\\"); }
    put(b);
}

static uint32_t byte_char(uint32_t b) { return b <= 0x20u ? 256u + b : (b >= 0x7Fu && b <= 0xA0u) ? b + 162u : b == 0xADu ? 323u : b; }

/* vocab: the 256 byte chars (ids 0..255, but byte 255's when skip), then extra[i] at 256 + i (255 + i when skip); merges; added tokens as
 * `"id": content` pairs; pre: the pre_tokenizer json */
static const char *mk(int skip, const char *const *extra, int n_extra, const char *merges, const char *added,
                      int ignore_merges, const char *unk, const char *pre)
{
    jn = 0;
    J[0] = 0;
    put("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[");
    put(added);
    put("],\"normalizer\":null,\"pre_tokenizer\":");
    put(pre);
    put(",\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,"
        "\"use_regex\":true},\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":");
    put(unk);
    put(",\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,");
    put(ignore_merges ? "\"ignore_merges\":true," : "\"ignore_merges\":false,");
    put("\"vocab\":{");
    char num[32];
    for (int b = 0, first = 1; b < 256; b++) {
        if (skip && b == 255) { continue; }
        put(first ? "\"" : ",\"");
        first = 0;
        put_utf8(byte_char((uint32_t)b));
        snprintf(num, sizeof num, "\":%d", b);
        put(num);
    }
    for (int i = 0; i < n_extra; i++) {
        put(",\"");
        put(extra[i]);
        snprintf(num, sizeof num, "\":%d", 256 - (skip != 0) + i);
        put(num);
    }
    put("},\"merges\":[");
    put(merges);
    put("]}}");
    return J;
}

#define ADD(id, s) "{\"id\":" #id ",\"content\":\"" s "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false," \
                   "\"normalized\":false,\"special\":true}"
#define BL_RX  "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":true}"
#define BL_RAW "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}"
#define SPLIT(rx) "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" rx "\"}," \
                  "\"behavior\":\"Isolated\",\"invert\":false}," BL_RAW "]}"
#define NEMO "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+" \
    "|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*" \
    "|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"

static toks_ctx *load_json(const char *what, const char *json)
{
    toks_ctx *c = NULL;
    toks_diag d;
    toks_load_opts o;
    memset(&o, 0, sizeof o);
    o.size = (uint32_t)sizeof o;
    o.diag = &d;
    int64_t r = toks_load_mem_copy(&c, json, strlen(json), &o);
    CHECK(r == 0 && c != NULL, "%s: load %" PRId64 " (%s)", what, r, d.what);
    return r == 0 ? c : NULL;
}

static uint32_t iso(const toks_ctx *c)
{
    toks_info i;
    memset(&i, 0, sizeof i);
    i.size = (uint32_t)sizeof i;
    return toks_get_info(c, &i) == 0 ? i.control_isolation : 99u;
}

/* 1 when text's NONE encode holds an added id */
static int none_hits(const toks_ctx *c, void *scr, const void *text, uint64_t len, uint32_t *out, uint64_t cap)
{
    int64_t n = toks_encode(c, text, len, TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS, out, cap, scr);
    for (int64_t k = 0; k < n && (uint64_t)k < cap; k++) {
        if (toks_id_flags(c, out[k]) > 0 && (toks_id_flags(c, out[k]) & (int64_t)TOKS_ID_ADDED) != 0) { return 1; }
    }
    return 0;
}

static void red(const char *what, const char *json, const char *text, uint64_t len)
{
    toks_ctx *c = load_json(what, json);
    if (c == NULL) { return; }
    uint64_t sb = toks_scratch_bytes(c, 4096u, 0u);
    void *scr = malloc((size_t)sb);
    uint32_t out[512];
    CHECK(iso(c) == 0u, "%s: certified, and an added token is reachable", what);
    CHECK(scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0 && none_hits(c, scr, text, len, out, 512u),
          "%s: the reach is not real: no added id in the NONE encode of its text", what);
    free(scr);
    toks_unload(c);
}

/* every added token spelled in CTX's contexts and 2000 random byte strings (seeded): no added id under NONE */
static const char *const CTX[] = { "%s", " %s", "x%s", "%sx", "a %s b", "\n%s\n", "%s%s", "'%s'", "\"%s\"", "(%s)",
                                   "1%s2", "  %s", "\t%s", "<%s>", "%s." };

static void search(const char *what, const toks_ctx *c)
{
    uint64_t sb = toks_scratch_bytes(c, 4096u, 0u);
    void *scr = malloc((size_t)sb);
    uint32_t *out = (uint32_t *)malloc(4u * 8192u);
    char t[2048];
    int hit = 0;
    if (scr == NULL || out == NULL || toks_scratch_init(c, scr, sb, 0u) != 0) { CHECK(0, "%s: scratch", what); }
    for (uint32_t i = 0; scr != NULL && out != NULL && hit == 0; i++) {
        const void *s;
        uint64_t len, id;
        uint32_t id32;
        if (toks_added(c, i, &s, &len, &id32) < 0) { break; }
        id = id32;
        for (size_t k = 0; k < sizeof CTX / sizeof CTX[0] && len < 400u; k++) {
            char a[512];
            memcpy(a, s, (size_t)len);
            a[len] = 0;
            int m = snprintf(t, sizeof t, CTX[k], a, a);
            if (m > 0 && none_hits(c, scr, t, (uint64_t)m, out, 8192u)) {
                hit = 1;
                CHECK(0, "%s: certified, but context %zu of added token %" PRIu64 " gives an added id", what, k, id);
                break;
            }
        }
    }
    uint64_t x = 0x9E3779B97F4A7C15u;
    for (int r = 0; scr != NULL && out != NULL && hit == 0 && r < 2000; r++) {
        uint64_t len = 1u + (r % 64);
        for (uint64_t k = 0; k < len; k++) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            t[k] = (char)(r % 3 == 0 ? (x & 0xFFu) : "<>|[]_/sSINTunk0 \n"[x % 18u]);
        }
        if (none_hits(c, scr, t, len, out, 8192u)) { hit = 1; CHECK(0, "%s: certified, but random string %d gives an added id", what, r); }
    }
    free(out);
    free(scr);
}

static void green(const char *what, const char *json)
{
    toks_ctx *c = load_json(what, json);
    if (c == NULL) { return; }
    CHECK(iso(c) == 1u, "%s: not certified", what);
    search(what, c);
    toks_unload(c);
}

static const char *path_of(const char *name)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    static char buf[1024];
    if (root != NULL && root[0] != 0) { snprintf(buf, sizeof buf, "%s/%s", root, name); }
    else { snprintf(buf, sizeof buf, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
}

/* the census pins (docs/algorithms/reach.md §4): byte-level bpe certified unless named; sentencepiece-style bpe,
 * unigram and wordpiece are not analyzed (0) */
static const struct { const char *name; uint32_t want; } PIN[] = {
    { "gpt2", 1u }, { "llama3", 1u }, { "glm53", 1u }, { "qwen38", 1u }, { "dsv3", 1u }, { "dsv4", 1u },
    { "dsv41flash", 1u }, { "o200k", 1u }, { "mistral-nemo", 1u }, { "nemotron3-4b", 1u }, { "nemo3nano30b", 1u },
    { "llama4", 1u }, { "minimaxm2", 1u }, { "modernbert", 1u },
    { "gx-spark-x25", 0u },          /* bytes 0xC0 0xC1 0xF5..0xFF lack their char: their unk "<unk>" is added */
    { "gemma3", 0u }, { "uni_t5base", 0u }, { "wp-bert-base-uncased", 0u },
};

static void real(void)
{
    for (size_t i = 0; i < sizeof PIN / sizeof PIN[0]; i++) {
        toks_ctx *c = NULL;
        const char *p = path_of(PIN[i].name);
        FILE *f = fopen(p, "rb");
        if (f == NULL) {
            char d[1100];
            snprintf(d, sizeof d, "%s/tokenizer.json", p);
            f = fopen(d, "rb");
        }
        if (f == NULL) { printf("SKIP %s: %s not found\n", PIN[i].name, p); continue; }
        fclose(f);
        if (toks_load(&c, p, NULL) != 0) { CHECK(0, "%s: load", PIN[i].name); continue; }
        CHECK(iso(c) == PIN[i].want, "%s: control_isolation %u, want %u", PIN[i].name, iso(c), PIN[i].want);
        if (iso(c) == 1u) { search(PIN[i].name, c); }
        toks_unload(c);
    }
    toks_ctx *c = NULL;                                 /* the spark reach is real: byte 0xFF is its unk */
    if (toks_load(&c, path_of("gx-spark-x25"), NULL) == 0) {
        uint64_t sb = toks_scratch_bytes(c, 64u, 0u);
        void *scr = malloc((size_t)sb);
        uint32_t out[64];
        CHECK(scr != NULL && toks_scratch_init(c, scr, sb, 0u) == 0 && none_hits(c, scr, "a\xFF", 2u, out, 64u),
              "gx-spark-x25: no unk for byte 0xFF");
        free(scr);
        toks_unload(c);
    }
}

int main(void)
{
    static const char *const AB[] = { "ab" }, *const PIPE[] = { "<|>" }, *const S[] = { "<s>" }, *const LT[] = { "<>" },
                             *const UNK[] = { "<unk>" }, *const NM[] = { "<s>", "[INST]", "<SPECIAL_12>" };
    red("merge", mk(0, AB, 1, "\"a b\"", ADD(256, "ab"), 0, "null", BL_RX), "ab", 2u);
    red("one byte", mk(0, NULL, 0, "", ADD(120, "x"), 0, "null", BL_RX), "x", 1u);
    red("ignore_merges, gpt-2's match", mk(0, PIPE, 1, "", ADD(256, "<|>"), 1, "null", BL_RX), "a<|>", 4u);
    red("ignore_merges, whole text", mk(0, S, 1, "", ADD(256, "<s>"), 1, "null", BL_RAW), "<s>", 3u);
    red("ignore_merges, a gap", mk(0, LT, 1, "", ADD(256, "<>"), 1, "null", SPLIT("\\\\p{L}+")), "ab<>cd", 6u);
    red("dropped byte's unk", mk(1, UNK, 1, "", ADD(255, "<unk>"), 0, "\"<unk>\"", BL_RX), "a\xFF", 2u);
    green("added id outside the vocabulary", mk(0, NULL, 0, "", ADD(256, "<|end|>"), 1, "null", BL_RX));
    green("nemo shape: in the vocabulary, ignore_merges",
          mk(0, NM, 3, "", ADD(256, "<s>") "," ADD(257, "[INST]") "," ADD(258, "<SPECIAL_12>"), 1, "null", SPLIT(NEMO)));
    green("in the vocabulary, merges only", mk(0, S, 1, "", ADD(256, "<s>"), 0, "null", BL_RAW));
    real();
    printf("test_reach: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
