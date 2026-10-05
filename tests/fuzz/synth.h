/*
 * tests/fuzz/synth.h: fz_synth, a byte-level bpe tokenizer.json drawn at random over the axes the reader
 * distinguishes (ids permuted or not, merges as strings or pairs, duplicate merges, several merges to one token,
 * unreachable tokens, ignore_merges, added tokens special / normalized / overlapping / equal to vocabulary strings
 * / single chars, the pre-tokenizer forms and the compiled patterns, post-processors with right, wrong or out of
 * range template ids, both decoder forms), now and then with a field toks refuses. fuzz_load_json.c starts from it
 * whenever its corpus is empty or an input does not parse (jsonmut.h). Ported from the first fuzz harnesses
 * (commit 82cdeda).
 */
#ifndef TOKS_FUZZ_SYNTH_H
#define TOKS_FUZZ_SYNTH_H

#include "fuzz.h"

size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);
#define fz_mutate_bytes LLVMFuzzerMutate

/* ======================================================================================================
 * the synthesizer
 * ====================================================================================================== */

static uint32_t FZ_BU[256];          /* gpt-2 bytes_to_unicode, built here from its definition */
static int16_t  FZ_UB[0x144];        /* its inverse: the byte a char stands for, or -1 */

FZ_FN void fz_bu_init(void)
{
    if (FZ_BU['!'] == '!') { return; }
    uint32_t n = 0;
    for (uint32_t c = 0; c < 0x144u; c++) { FZ_UB[c] = -1; }
    for (uint32_t b = 0; b < 256u; b++) {
        if ((b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || (b >= 0xAEu && b <= 0xFFu)) { FZ_BU[b] = b; }
        else { FZ_BU[b] = 0x100u + n++; }
        FZ_UB[FZ_BU[b]] = (int16_t)b;
    }
}

/* the json string body (no quotes) of the alphabet image of raw bytes */
FZ_FN void fz_put_alpha(fz_bytes *o, const uint8_t *raw, uint32_t n)
{
    uint8_t u[4];
    for (uint32_t i = 0; i < n; i++) {
        uint32_t cp = FZ_BU[raw[i]];
        if (cp == '"' || cp == '\\') { fz_bytes_put(o, "\\", 1u); }
        fz_bytes_put(o, u, fz_utf8_put(cp, u));
    }
}

/* the json string body of arbitrary utf-8 text */
FZ_FN void fz_put_jstr(fz_bytes *o, const uint8_t *s, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint8_t c = s[i];
        if (c == '"' || c == '\\') { fz_bytes_put(o, "\\", 1u); fz_bytes_put(o, &c, 1u); }
        else if (c < 0x20u) { char e[8]; snprintf(e, sizeof e, "\\u%04x", c); fz_bytes_put(o, e, 6u); }
        else { fz_bytes_put(o, &c, 1u); }
    }
}

FZ_FN void fz_puts(fz_bytes *o, const char *s) { fz_bytes_put(o, s, strlen(s)); }

FZ_FN void fz_putu(fz_bytes *o, uint64_t v)
{
    char b[24];
    int k = snprintf(b, sizeof b, "%" PRIu64, v);
    fz_bytes_put(o, b, (uint64_t)k);
}

/* the four exact patterns config.c compiles (kernels.md §3), as json string bodies */
static const char *const FZ_PAT_JSON[4] = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\\\p{L}+| ?\\\\p{N}+| ?[^\\\\s\\\\p{L}\\\\p{N}]+|\\\\s+(?!\\\\S)|\\\\s+",
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+",
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+",
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{L}\\\\p{M}]+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+",
};

#define FZ_SYN_MAXTOK 1024u
#define FZ_SYN_MAXLEN 48u

typedef struct fz_syn {
    uint8_t  raw[FZ_SYN_MAXTOK][FZ_SYN_MAXLEN];
    uint32_t len[FZ_SYN_MAXTOK];
    uint32_t id[FZ_SYN_MAXTOK];       /* the json id of token i (a permutation of 0..n-1) */
    uint32_t n;
    uint32_t ml[1024], mr[1024], nm;  /* merges as token indices */
} fz_syn;

FZ_FN int64_t fz_syn_find(const fz_syn *y, const uint8_t *p, uint32_t l)
{
    for (uint32_t i = 0; i < y->n; i++) {
        if (y->len[i] == l && memcmp(y->raw[i], p, l) == 0) { return (int64_t)i; }
    }
    return -1;
}

/* the alphabet image of token i, as utf-8 (for the added-token id rule: a content equal to a vocabulary
 * string takes that token's id) */
FZ_FN uint32_t fz_syn_image(const fz_syn *y, uint32_t i, uint8_t *out)
{
    uint32_t k = 0;
    for (uint32_t j = 0; j < y->len[i]; j++) { k += fz_utf8_put(FZ_BU[y->raw[i][j]], out + k); }
    return k;
}

FZ_FN void fz_synth(fz_bytes *o, uint64_t *s)
{
    fz_bu_init();
    fz_syn *y = (fz_syn *)fz_alloc(sizeof *y);
    memset(y, 0, sizeof *y);
    for (uint32_t b = 0; b < 256u; b++) { y->raw[b][0] = (uint8_t)b; y->len[b] = 1u; }
    y->n = 256u;
    uint64_t K = fz_below(s, 4u) == 0u ? fz_below(s, 8u) : fz_below(s, 400u);
    for (uint64_t k = 0; k < K && y->n < FZ_SYN_MAXTOK && y->nm < 1024u; k++) {
        if (y->nm != 0u && fz_below(s, 30u) == 0u) {               /* a duplicate merge: the last one counts */
            uint32_t j = (uint32_t)fz_below(s, y->nm);
            y->ml[y->nm] = y->ml[j]; y->mr[y->nm] = y->mr[j]; y->nm++;
            continue;
        }
        uint32_t a = fz_below(s, 2u) == 0u && y->n > 266u ? y->n - 1u - (uint32_t)fz_below(s, 10u) : (uint32_t)fz_below(s, y->n);
        uint32_t c = fz_below(s, 3u) == 0u ? (uint32_t)fz_below(s, y->n) : (uint32_t)fz_below(s, 96u) + 32u;
        if (y->len[a] + y->len[c] > FZ_SYN_MAXLEN) { continue; }
        uint8_t prod[2u * FZ_SYN_MAXLEN];
        memcpy(prod, y->raw[a], y->len[a]);
        memcpy(prod + y->len[a], y->raw[c], y->len[c]);
        int64_t have = fz_syn_find(y, prod, y->len[a] + y->len[c]);
        if (have >= 0 && fz_below(s, 3u) != 0u) { continue; }     /* else a second split of one token */
        if (have < 0) {
            memcpy(y->raw[y->n], prod, y->len[a] + y->len[c]);
            y->len[y->n] = y->len[a] + y->len[c];
            y->n++;
        }
        y->ml[y->nm] = a; y->mr[y->nm] = c; y->nm++;
    }
    if (fz_below(s, 4u) == 0u) {                                   /* tokens no merge reaches */
        uint64_t e = 1u + fz_below(s, 5u);
        for (uint64_t k = 0; k < e && y->n < FZ_SYN_MAXTOK; k++) {
            uint32_t l = 2u + (uint32_t)fz_below(s, 5u);
            for (uint32_t j = 0; j < l; j++) { y->raw[y->n][j] = (uint8_t)fz_rng(s); }
            if (fz_syn_find(y, y->raw[y->n], l) >= 0) { continue; }
            y->len[y->n] = l;
            y->n++;
        }
    }
    for (uint32_t i = 0; i < y->n; i++) { y->id[i] = i; }
    uint64_t perm = fz_below(s, 4u);
    if (perm == 1u) {                                              /* every id shuffled */
        for (uint32_t i = y->n - 1u; i > 0u; i--) { uint32_t j = (uint32_t)fz_below(s, i + 1u), t = y->id[i]; y->id[i] = y->id[j]; y->id[j] = t; }
    } else if (perm == 2u) {                                       /* merged tokens first */
        for (uint32_t i = 0; i < y->n; i++) { y->id[i] = i < 256u ? y->n - 256u + i : i - 256u; }
    }

    /* added tokens: content, hf id (vocabulary id of an equal string, else counted up from n_vocab) */
    uint64_t na = fz_below(s, 3u) == 0u ? 0u : fz_below(s, 10u);
    uint8_t ac[16][64];
    uint32_t al[16], aid[16], aspecial[16];
    uint32_t next_id = y->n, n_ids = y->n;
    static const char *const ONE[] = { "\n", " ", "x", "\xc3\xa9", "\xe4\xb8\xad", "<", "|", "'", "\xc4\xa0", "a\xcc\x81" };
    for (uint64_t i = 0; i < na; i++) {
        uint8_t *c = ac[i];
        uint32_t l = 0;
        switch (fz_below(s, 6u)) {
        case 0: case 1: {
            memcpy(c, "<|", 2u); l = 2u;
            uint32_t k = 1u + (uint32_t)fz_below(s, 12u);
            for (uint32_t j = 0; j < k; j++) { c[l++] = (uint8_t)("abcdefghijklmnopqrstuvwxyz_0123456789"[fz_below(s, 37u)]); }
            memcpy(c + l, "|>", 2u); l += 2u;
            break;
        }
        case 2: {                                                  /* a vocabulary string */
            uint32_t t = (uint32_t)fz_below(s, y->n);
            uint8_t img[4u * FZ_SYN_MAXLEN];
            uint32_t k = fz_syn_image(y, t, img);
            if (k > 60u) { k = fz_syn_image(y, (uint32_t)fz_below(s, 256u), img); }
            memcpy(c, img, k); l = k;
            break;
        }
        case 3:                                                    /* a prefix / extension of another */
            if (i != 0u) {
                uint32_t j = (uint32_t)fz_below(s, i);
                l = al[j] > 1u && fz_below(s, 2u) == 0u ? al[j] - 1u : al[j];
                while (l > 0u && l < al[j] && (ac[j][l] & 0xC0u) == 0x80u) { l--; }   /* a char boundary */
                memcpy(c, ac[j], l);
                if (l == al[j] && l < 60u) { c[l++] = (uint8_t)'x'; }
                break;
            }
            /* fall through */
        case 4: {
            const char *t = ONE[fz_below(s, sizeof ONE / sizeof ONE[0])];
            l = (uint32_t)strlen(t);
            memcpy(c, t, l);
            break;
        }
        default: {
            static const uint32_t CPS[] = { 'a', 'Z', '1', ' ', '\n', 0xE9u, 0x4E2Du, 0x1F600u, 0x301u, '<', '>', 0x3000u };
            uint32_t k = 1u + (uint32_t)fz_below(s, 5u);
            for (uint32_t j = 0; j < k; j++) { l += fz_utf8_put(CPS[fz_below(s, sizeof CPS / sizeof CPS[0])], c + l); }
            break;
        }
        }
        al[i] = l;
        int64_t v = -1;
        for (uint32_t t = 0; t < y->n && v < 0; t++) {
            uint8_t img[4u * FZ_SYN_MAXLEN];
            uint32_t k = fz_syn_image(y, t, img);
            if (k == l && memcmp(img, c, l) == 0) { v = (int64_t)y->id[t]; }
        }
        int dup = 0;
        for (uint64_t j = 0; j < i; j++) { if (al[j] == l && memcmp(ac[j], c, l) == 0) { dup = 1; } }
        aid[i] = v >= 0 ? (uint32_t)v : (dup ? 0u : next_id++);
        if (aid[i] + 1u > n_ids) { n_ids = aid[i] + 1u; }
        aspecial[i] = fz_below(s, 2u) == 0u;
    }

    int as_pairs = fz_below(s, 2u) == 0u;
    fz_puts(o, "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[");
    for (uint64_t i = 0; i < na; i++) {
        if (i != 0u) { fz_puts(o, ","); }
        fz_puts(o, "{\"id\":");
        fz_putu(o, fz_below(s, 8u) == 0u ? fz_below(s, 100000u) : aid[i]);
        fz_puts(o, ",\"content\":\"");
        if (fz_below(s, 40u) != 0u) { fz_put_jstr(o, ac[i], al[i]); }
        fz_puts(o, "\",\"single_word\":");
        fz_puts(o, fz_below(s, 40u) == 0u ? "true" : "false");
        fz_puts(o, ",\"lstrip\":");
        fz_puts(o, fz_below(s, 40u) == 0u ? "true" : "false");
        fz_puts(o, ",\"rstrip\":");
        fz_puts(o, fz_below(s, 40u) == 0u ? "true" : "false");
        fz_puts(o, ",\"normalized\":");
        fz_puts(o, fz_below(s, 3u) == 0u ? "true" : "false");
        fz_puts(o, ",\"special\":");
        fz_puts(o, aspecial[i] ? "true" : "false");
        fz_puts(o, "}");
    }
    fz_puts(o, "],\"normalizer\":");
    fz_puts(o, fz_below(s, 30u) == 0u ? "{\"type\":\"NFC\"}" : "null");

    fz_puts(o, ",\"pre_tokenizer\":");
    uint64_t pt = fz_below(s, 8u);
    const char *bl = fz_below(s, 30u) == 0u ? "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true"
                                            : "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true";
    if (pt == 0u) { fz_puts(o, bl); fz_puts(o, ",\"use_regex\":true}"); }
    else if (pt == 1u) { fz_puts(o, bl); fz_puts(o, ",\"use_regex\":false}"); }
    else if (pt == 2u) { fz_puts(o, bl); fz_puts(o, "}"); }
    else if (pt == 7u) { fz_puts(o, "{\"type\":\"Sequence\",\"pretokenizers\":["); fz_puts(o, bl); fz_puts(o, ",\"use_regex\":true}]}"); }
    else {
        fz_puts(o, "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"");
        fz_puts(o, FZ_PAT_JSON[pt - 3u]);
        fz_puts(o, "\"},\"behavior\":\"Isolated\",\"invert\":false},");
        fz_puts(o, bl);
        fz_puts(o, ",\"use_regex\":false}]}");
    }

    fz_puts(o, ",\"post_processor\":");
    uint64_t pp = fz_below(s, 4u);
    if (pp == 0u) { fz_puts(o, "null"); }
    else if (pp == 1u) { fz_puts(o, "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false,\"use_regex\":true}"); }
    else {
        if (pp == 3u) { fz_puts(o, "{\"type\":\"Sequence\",\"processors\":[{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":false,\"use_regex\":true},"); }
        int bos = fz_below(s, 3u) != 0u, eos = fz_below(s, 2u) == 0u;
        uint32_t ids[2][3], nid[2];
        for (int w = 0; w < 2; w++) {
            nid[w] = 1u + (fz_below(s, 6u) == 0u ? 1u : 0u);
            for (uint32_t k = 0; k < nid[w]; k++) {
                uint64_t c = fz_below(s, 8u);
                ids[w][k] = (c < 5u && na != 0u) ? aid[fz_below(s, na)]
                          : c < 7u ? (uint32_t)fz_below(s, n_ids) : n_ids + (uint32_t)fz_below(s, 3u);
            }
        }
        fz_puts(o, "{\"type\":\"TemplateProcessing\",\"single\":[");
        if (bos) { fz_puts(o, "{\"SpecialToken\":{\"id\":\"<bos>\",\"type_id\":0}},"); }
        fz_puts(o, fz_below(s, 40u) == 0u ? "{\"Sequence\":{\"id\":\"B\",\"type_id\":0}}" : "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}");
        if (eos) { fz_puts(o, ",{\"SpecialToken\":{\"id\":\"<eos>\",\"type_id\":0}}"); }
        fz_puts(o, "],\"pair\":[{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"Sequence\":{\"id\":\"B\",\"type_id\":1}}],\"special_tokens\":{");
        for (int w = 0; w < 2; w++) {
            fz_puts(o, w == 0 ? "\"<bos>\":{\"id\":\"<bos>\",\"ids\":[" : ",\"<eos>\":{\"id\":\"<eos>\",\"ids\":[");
            for (uint32_t k = 0; k < nid[w]; k++) { if (k != 0u) { fz_puts(o, ","); } fz_putu(o, ids[w][k]); }
            fz_puts(o, "],\"tokens\":[");
            for (uint32_t k = 0; k < nid[w]; k++) { fz_puts(o, k != 0u ? ",\"t\"" : "\"t\""); }
            fz_puts(o, "]}");
        }
        fz_puts(o, "}}");
        if (pp == 3u) { fz_puts(o, "]}"); }
    }

    fz_puts(o, ",\"decoder\":");
    fz_puts(o, fz_below(s, 2u) == 0u ? "{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}"
                                     : "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true}]}");

    fz_puts(o, ",\"model\":{");
    if (fz_below(s, 4u) != 0u) { fz_puts(o, "\"type\":\"BPE\","); }
    fz_puts(o, "\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":");
    fz_puts(o, fz_below(s, 2u) == 0u ? "null" : "\"\"");
    fz_puts(o, ",\"end_of_word_suffix\":");
    fz_puts(o, fz_below(s, 2u) == 0u ? "null" : "\"\"");
    fz_puts(o, ",\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":");
    fz_puts(o, fz_below(s, 2u) == 0u ? "true" : "false");
    fz_puts(o, ",\"vocab\":{");
    int by_id = fz_below(s, 2u) == 0u;
    for (uint32_t k = 0; k < y->n; k++) {
        uint32_t i = k;
        if (by_id) { for (i = 0; i < y->n && y->id[i] != k; i++) { } }
        if (k != 0u) { fz_puts(o, ","); }
        fz_puts(o, "\"");
        fz_put_alpha(o, y->raw[i], y->len[i]);
        fz_puts(o, "\":");
        fz_putu(o, y->id[i]);
    }
    fz_puts(o, "},\"merges\":[");
    for (uint32_t m = 0; m < y->nm; m++) {
        if (m != 0u) { fz_puts(o, ","); }
        fz_puts(o, as_pairs ? "[\"" : "\"");
        fz_put_alpha(o, y->raw[y->ml[m]], y->len[y->ml[m]]);
        fz_puts(o, as_pairs ? "\",\"" : " ");
        fz_put_alpha(o, y->raw[y->mr[m]], y->len[y->mr[m]]);
        fz_puts(o, as_pairs ? "\"]" : "\"");
    }
    fz_puts(o, "]}}");
    free(y);
}

#endif /* TOKS_FUZZ_SYNTH_H */
