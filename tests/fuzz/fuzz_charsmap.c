/* tests/fuzz/fuzz_charsmap.c: libFuzzer harness for the precompiled charsmap reader (precompiled.c) and the unigram
 * normalizer that runs the map, at the map's real size. The load harnesses cap an input at 64 KiB and the census's
 * charsmap is 237,539 bytes (316,720 of base64), so no tokenizer file of theirs carries a real one. An input here is
 * a shape byte and a charsmap's bytes (as sentencepiece writes them, before base64), up to 1 MiB: the harness writes
 * the bytes, base64-encoded, into one of four Unigram tokenizer.json shapes of the census with a small vocabulary,
 * and loads that as the load harnesses do (load.h: both tiers give one verdict, a refusal is a documented code, and
 * TOKS_E_NOMEM is a finding); an accepted file gets the load battery, its slices taken from the charsmap's bytes.
 * Shapes (the shape byte mod 4): bge-m3's (Precompiled, Replace ' {2,}', Metaspace), t5's (Precompiled alone,
 * WhitespaceSplit + Metaspace), albert's (Replace, NFKD, StripAccents, Lowercase, Precompiled) and bge-m3's with byte
 * fallback. Seeds: tests/fuzz/seeds.py charsmap (the cached files' maps, whole and cut). docs/fuzz.md. */
#include "load.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);

#define CM_META "\xe2\x96\x81"                    /* U+2581 */
#define CM_ADDED "\"added_tokens\":[{\"id\":0,\"content\":\"<unk>\",\"single_word\":false,\"lstrip\":false," \
                 "\"rstrip\":false,\"normalized\":false,\"special\":true}],"
#define CM_PC "{\"type\":\"Precompiled\",\"precompiled_charsmap\":\""
#define CM_MS "{\"type\":\"Metaspace\",\"replacement\":\"" CM_META "\",\"add_prefix_space\":true}"
#define CM_WS "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"WhitespaceSplit\"}," CM_MS "]}"

/* the file before the base64 and after it, up to the vocabulary's first entry */
static const char *const CM_PRE[4] = {
    "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null," CM_ADDED "\"normalizer\":{\"type\":\"Sequence\","
    "\"normalizers\":[" CM_PC,
    "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null," CM_ADDED "\"normalizer\":" CM_PC,
    "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null," CM_ADDED "\"normalizer\":{\"type\":\"Sequence\","
    "\"normalizers\":[{\"type\":\"Replace\",\"pattern\":{\"String\":\"``\"},\"content\":\"\\\"\"},{\"type\":\"Replace\","
    "\"pattern\":{\"String\":\"''\"},\"content\":\"\\\"\"},{\"type\":\"NFKD\"},{\"type\":\"StripAccents\"},"
    "{\"type\":\"Lowercase\"}," CM_PC,
    "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null," CM_ADDED "\"normalizer\":{\"type\":\"Sequence\","
    "\"normalizers\":[" CM_PC,
};
static const char *const CM_POST[4] = {
    "\"},{\"type\":\"Replace\",\"pattern\":{\"Regex\":\" {2,}\"},\"content\":\" \"}]},\"pre_tokenizer\":" CM_MS ","
    "\"post_processor\":null,\"decoder\":" CM_MS ",\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[",
    "\"},\"pre_tokenizer\":" CM_WS ",\"post_processor\":null,\"decoder\":" CM_MS ",\"model\":{\"type\":\"Unigram\","
    "\"unk_id\":0,\"vocab\":[",
    "\"}]},\"pre_tokenizer\":" CM_WS ",\"post_processor\":null,\"decoder\":" CM_MS ",\"model\":{\"type\":\"Unigram\","
    "\"unk_id\":0,\"vocab\":[",
    "\"},{\"type\":\"Replace\",\"pattern\":{\"Regex\":\" {2,}\"},\"content\":\" \"}]},\"pre_tokenizer\":" CM_MS ","
    "\"post_processor\":null,\"decoder\":" CM_MS ",\"model\":{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[",
};

/* the vocabulary: <unk>, the metaspace, every printable ascii char alone and after the metaspace, a few words and
 * chars the map writes (kana, cjk, accents), and with byte fallback the 256 <0xHH> pieces */
static void cm_vocab(fz_bytes *b, int bytes)
{
    static const char *const WORDS[] = { CM_META "the", CM_META "of", CM_META "and", CM_META "in", "ing",
        "ed", CM_META "de", "\xe3\x81\xae", "\xe3\x81\xaf", CM_META "\xe4\xb8\xad", "\xe6\x96\x87", "\xc3\xa9",
        "\xc3\xbc", "\xc3\x9f", "\xe3\x82\xad\xe3\x83\xad" };
    char e[64];
    fz_bytes_put(b, "[\"<unk>\",0.0],[\"" CM_META "\",-2.0]", strlen("[\"<unk>\",0.0],[\"" CM_META "\",-2.0]"));
    for (int c = 0x21; c < 0x7f; c++) {
        const char *q = (c == '"' || c == '\\') ? "\\" : "";
        int k = snprintf(e, sizeof e, ",[\"%s%c\",-%d.%d],[\"" CM_META "%s%c\",-5.0]", q, c, 6 + c % 3, c % 10, q, c);
        fz_bytes_put(b, e, (uint64_t)k);
    }
    for (size_t i = 0; i < FZ_N(WORDS); i++) {
        int k = snprintf(e, sizeof e, ",[\"%s\",-%d.5]", WORDS[i], 3 + (int)(i % 4));
        fz_bytes_put(b, e, (uint64_t)k);
    }
    for (int c = 0; bytes && c < 256; c++) {
        int k = snprintf(e, sizeof e, ",[\"<0x%02X>\",0.0]", c);
        fz_bytes_put(b, e, (uint64_t)k);
    }
    fz_bytes_put(b, bytes ? "],\"byte_fallback\":true}}" : "],\"byte_fallback\":false}}",
                 bytes ? strlen("],\"byte_fallback\":true}}") : strlen("],\"byte_fallback\":false}}"));
}

static void cm_base64(fz_bytes *b, const uint8_t *d, size_t n)
{
    static const char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char q[4];
    for (size_t i = 0; i < n; i += 3) {                 /* bound: n / 3 */
        uint32_t v = (uint32_t)d[i] << 16 | (i + 1 < n ? (uint32_t)d[i + 1] << 8 : 0u) | (i + 2 < n ? d[i + 2] : 0u);
        q[0] = A[v >> 18];
        q[1] = A[(v >> 12) & 63u];
        q[2] = i + 1 < n ? A[(v >> 6) & 63u] : '=';
        q[3] = i + 2 < n ? A[v & 63u] : '=';
        fz_bytes_put(b, q, 4u);
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n)
{
    if (n == 0u) { return 0; }
    uint32_t shape = d[0] & 3u;
    fz_bytes j = { NULL, 0, 0 };
    fz_bytes_put(&j, CM_PRE[shape], strlen(CM_PRE[shape]));
    cm_base64(&j, d + 1, n - 1u);
    fz_bytes_put(&j, CM_POST[shape], strlen(CM_POST[shape]));
    cm_vocab(&j, shape == 3u);
    fz_tok t;
    fz_what = "charsmap file";
    int64_t r = fz_tok_open(&t, "charsmap file", NULL, j.p, j.n);
    if (r != 0) {
        FZ_CHECK(r != TOKS_E_NOMEM, "TOKS_E_NOMEM on a %zu-byte charsmap (shape %u)", n - 1u, shape);
    } else {
        fz_battery(&t, d + 1, n - 1u, fz_hash(d, n));
        fz_tok_close(&t);
    }
    fz_bytes_free(&j);
    return 0;
}
