/*
 * tests/fuzz/gen.h: the text harnesses' input format and the class-aware generator (SPEC T6), their custom
 * mutator. Ported from the first fuzz harnesses (commit 82cdeda), with the tokenizer's literals taken through toks.h.
 *
 * Input: an 8-byte header, then the text.
 *   d[0]     the pinned tokenizer (fuzz.h fz_pin: byte mod 17, all four algorithms and the kimi wrapper)
 *   d[1]     bits 0-1 the added-token mode (3 = ALL), bit 2 TOKS_NO_POSTPROCESS, bit 3 TOKS_CONTINUATION
 *   d[2..7]  sel: alignments, capacities, rebinding, partitions (check.h); its top bits come from the text's hash
 * The text is copied so that it ends flush with its heap block (ASan sees any read past len).
 * The generator writes atoms of every class the scanners distinguish (L, N, P, WS, NL, marks), contractions with
 * their case and U+017F variants, the 25 \s chars, invalid utf-8 of every SPEC §3.3 kind, the tokenizer's own
 * markup tokens (whole, cut short, overlapping, doubled, with a char inside), runs whose lengths straddle 16 / 32 /
 * 64-byte blocks and the driver's 256-piece rounds, single pieces of up to 6,000 letters, NFC changes (growth up
 * to 3x), sentencepiece texts (U+2581, space runs, "<0x41>"-like strings, chars only byte fallback covers);
 * a quarter of the mutations are libFuzzer's byte-level ones.
 */
#ifndef TOKS_FUZZ_GEN_H
#define TOKS_FUZZ_GEN_H

#include "check.h"

size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize);

#define FZ_HDR 8u

typedef struct fz_text {
    fz_tok   *t;
    uint32_t  flags;
    uint64_t  sel, len;
    uint8_t  *block;       /* the text's heap block: the text ends flush with it */
    const uint8_t *x;
} fz_text;

/* the header + text of one input; 0 when it is too short or no pinned file exists */
FZ_FN int fz_text_open(fz_text *tx, const uint8_t *d, size_t n)
{
    if (n < FZ_HDR) { return 0; }
    tx->t = fz_pin(d[0]);
    if (tx->t == NULL) { return 0; }
    tx->flags = (d[1] & 3u) == 3u ? TOKS_ADDED_ALL : (d[1] & 3u);
    if (d[1] & 4u) { tx->flags |= TOKS_NO_POSTPROCESS; }
    if (d[1] & 8u) { tx->flags |= TOKS_CONTINUATION; }
    tx->len = n - FZ_HDR;
    if (tx->len > FZ_MAX_TEXT) { tx->len = FZ_MAX_TEXT; }
    tx->sel = 0;
    for (int i = 2; i < 8; i++) { tx->sel |= (uint64_t)d[i] << (8 * (i - 2)); }
    tx->sel |= fz_hash(d + FZ_HDR, tx->len) << 48;
    tx->block = (uint8_t *)fz_alloc(tx->len);
    memcpy(tx->block, d + FZ_HDR, (size_t)tx->len);
    tx->x = tx->block;
    return 1;
}

FZ_FN void fz_text_close(fz_text *tx) { free(tx->block); tx->block = NULL; }

static const uint32_t FZ_CP_L[] = {
    'a', 'b', 'e', 's', 't', 'x', 'Z', 'K', 'S', 'L', 0xE9u, 0xDFu, 0xF1u, 0x17Fu, 0x212Au, 0x3C9u, 0x416u,
    0x5D0u, 0x627u, 0x905u, 0x4E2Du, 0x65E5u, 0xD55Cu, 0x306Eu, 0x30ABu, 0x1C5u, 0x2B0u, 0xAAu, 0x10400u,
    0x1D400u, 0x20000u, 0x130u, 0x131u, 0x1E9Eu, 0xFB01u,
};
static const uint32_t FZ_CP_M[] = { 0x301u, 0x316u, 0x308u, 0x903u, 0x20DDu, 0x3099u, 0xFE0Fu, 0xE0100u, 0x1AB0u };
static const uint32_t FZ_CP_N[] = {
    '0', '1', '5', '9', 0x663u, 0x6F4u, 0x966u, 0xB2u, 0xB9u, 0xBDu, 0x216Bu, 0xFF10u, 0x1D7D8u, 0x2460u, 0x3007u,
    0x11DE0u, 0x16FF4u,
};
static const uint32_t FZ_CP_P[] = {
    '.', ',', '!', '?', '-', '_', '\'', '"', '(', ')', '<', '>', '|', '/', '\\', '@', '#', '$', '%', '^', '&',
    '*', '+', '=', '~', '`', ';', ':', '[', ']', '{', '}', 0x2019u, 0x201Cu, 0x2014u, 0x2026u, 0x20ACu, 0xA9u,
    0xADu, 0x1F600u, 0x1F469u, 0x200Du, 0x200Bu, 0xFEFFu, 0x7Fu, 0x0u, 0x1u, 0x1Bu, 0x1Cu, 0x1Fu, 0xE000u,
    0xFFFDu, 0xFFFFu, 0x10FFFFu, 0x1F3FBu,
};
static const uint32_t FZ_CP_WS[] = {
    ' ', '\t', 0x0Bu, 0x0Cu, 0x85u, 0xA0u, 0x1680u, 0x2000u, 0x2001u, 0x2002u, 0x2003u, 0x2004u, 0x2005u,
    0x2006u, 0x2007u, 0x2008u, 0x2009u, 0x200Au, 0x2028u, 0x2029u, 0x202Fu, 0x205Fu, 0x3000u, ' ', ' ', ' ',
};
static const uint32_t FZ_CP_NL[] = { '\n', '\r' };

static const char *const FZ_STR[] = {
    "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'LL", "'D", "'\xc5\xbf", "'x", "''",
    "'", "\r\n", "\n\r", " \n", "\n ", "  \n  ", "Hello, world!", "don't", "DON'T", "y'all'd've", "1234567890",
    " 123", "3.14159", "<|endoftext|>", "<|begin_of_text|>", "<|eot_id|>", "<|im_start|>", "<|im_end|>", "[MASK]",
    "[CLS]", "<s>", "</s>", "<|", "|>", "<<|", "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb", "e\xcc\x81",
    "\xef\xac\x81", "x\xe2\x80\x8by", " '", "'\n", "a\xcc\x81\xcc\x96", "\xe4\xb8\xad\xe6\x96\x87", "  ",
    "\t\t", "\xe3\x80\x80", "\xc2\xa0", "def f(x):\n    return x  # ok\n", "http://x.y/z?a=1&b=2", "HTMLParser",
    "ABC.", "AB\xe4\xb8\xad\x43", "##ing", "un##", "\xe2\x96\x81the",
};
static const char *const FZ_BAD[] = {
    "\x80", "\xbf", "\xc0\x80", "\xc1\xbf", "\xc2", "\xc2\x41", "\xe0\x80\x80", "\xe0\xa0", "\xe0\x9f\xbf",
    "\xed\xa0\x80", "\xed\xbf\xbf", "\xf0\x80\x80\x80", "\xf0\x8f\xbf\xbf", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80",
    "\xf8\x88\x80\x80\x80", "\xfe", "\xff", "\xe4\xb8", "\xf0\x9f\x98", "\xc3", "\xe2\x82", "\xf0\x9f",
};
static const char *const FZ_NFC[] = {
    "e\xcc\x81", "A\xcc\x8a", "\xe2\x84\xab", "\xe2\x84\xa6", "\xcd\x84", "\xe0\xa5\x98", "\xf0\x9d\x85\xa0",
    "\xf0\x9d\x85\x9e", "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "\xea\xb0\x80\xe1\x86\xa8", "a\xcc\x81\xcc\x96",
    "\xe0\xbd\xb3", "\xe0\xbd\xb5", "\xe0\xbe\x81", "\xef\xac\xac", "\xe2\xab\x9c", "\xe1\xba\x9b\xcc\xa3",
    "\xcc\x81", "\xff\xcc\x81", "\xc3\x85\xcc\x81", "\xcd\x80", "\xcd\xb4", "\xcd\xbe", "\xce\x87",
};
static const char *const FZ_SPM[] = {
    "\xe2\x96\x81", " ", "  ", " \xe2\x96\x81 ", "<0x41>", "<0xff>", "<0x+F>", "<unk>", "<bos>", "<eos>",
    "\xf3\xb0\x80\x80", "\xcd\xb8", "\xf0\x9f\xab\xa8", "a\xe2\x96\x81\x62", "\t\n", "\xe2\x96\x81\xe2\x96\x81",
};
static const uint32_t FZ_RUNLEN[] = {
    1, 1, 1, 2, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 1000,
};
static const uint32_t *const FZ_CLS[6] = { FZ_CP_L, FZ_CP_M, FZ_CP_N, FZ_CP_P, FZ_CP_WS, FZ_CP_NL };
static const uint64_t FZ_CLS_N[6] = { FZ_N(FZ_CP_L), FZ_N(FZ_CP_M), FZ_N(FZ_CP_N), FZ_N(FZ_CP_P), FZ_N(FZ_CP_WS),
                                      FZ_N(FZ_CP_NL) };

FZ_FN void fz_put_cp(fz_bytes *b, uint32_t cp)
{
    uint8_t u[4];
    fz_bytes_put(b, u, fz_utf8_put(cp, u));
}

FZ_FN void fz_put_rep(fz_bytes *b, const char *t, uint64_t k)
{
    for (uint64_t i = 0; i < k; i++) { fz_bytes_put(b, t, strlen(t)); }
}

/* appends one generated fragment to b (t: the selected tokenizer, for its markup literals; may be NULL) */
FZ_FN void fz_gen_fragment(fz_bytes *b, uint64_t *s, const fz_tok *t)
{
    switch (fz_below(s, 13u)) {
    case 11:                                          /* NFC: one change, or a run of it (growth up to 3x) */
        fz_put_rep(b, FZ_NFC[fz_below(s, FZ_N(FZ_NFC))], fz_below(s, 3u) == 0u ? FZ_RUNLEN[fz_below(s, FZ_N(FZ_RUNLEN))] : 1u);
        break;
    case 12:                                          /* sentencepiece: U+2581, spaces, byte-token strings */
        fz_put_rep(b, FZ_SPM[fz_below(s, FZ_N(FZ_SPM))], fz_below(s, 4u) == 0u ? FZ_RUNLEN[fz_below(s, 16u)] : 1u);
        break;
    case 0: case 1: case 2: {                         /* a run of one class: one atom repeated, or mixed */
        uint64_t c = fz_below(s, 6u), k = FZ_RUNLEN[fz_below(s, FZ_N(FZ_RUNLEN))];
        int same = fz_below(s, 2u) == 0u;
        uint32_t cp = FZ_CLS[c][fz_below(s, FZ_CLS_N[c])];
        for (uint64_t i = 0; i < k; i++) {
            if (!same) { cp = FZ_CLS[c][fz_below(s, FZ_CLS_N[c])]; }
            fz_put_cp(b, cp);
        }
        break;
    }
    case 3: {                                         /* a word: optional prefix atom, letters, marks */
        if (fz_below(s, 2u) == 0u) {
            uint64_t c = 3u + fz_below(s, 2u);
            fz_put_cp(b, FZ_CLS[c][fz_below(s, FZ_CLS_N[c])]);
        }
        uint64_t k = 1u + fz_below(s, 12u);
        for (uint64_t i = 0; i < k; i++) {
            uint64_t c = fz_below(s, 8u) == 0u ? 1u : 0u;
            fz_put_cp(b, FZ_CLS[c][fz_below(s, FZ_CLS_N[c])]);
        }
        break;
    }
    case 4:                                           /* a fixed string: contractions, specials, code, ... */
        fz_put_rep(b, FZ_STR[fz_below(s, FZ_N(FZ_STR))], 1u);
        break;
    case 5:                                           /* invalid utf-8 (SPEC §3.3 kinds) */
        fz_put_rep(b, FZ_BAD[fz_below(s, FZ_N(FZ_BAD))], fz_below(s, 4u) == 0u ? FZ_RUNLEN[fz_below(s, 12u)] : 1u);
        break;
    case 6: case 7: {                                 /* the tokenizer's own markup tokens */
        const char *c = fz_tok_lit(t, fz_rng(s));
        if (c == NULL) { fz_bytes_put(b, "<|", 2u); break; }
        uint64_t l = strlen(c);
        switch (fz_below(s, 6u)) {
        case 0: fz_bytes_put(b, c, l); fz_bytes_put(b, c, l); break;                 /* doubled */
        case 1: fz_bytes_put(b, c, l - 1u); break;                                   /* cut short */
        case 2: {                                                                    /* overlapping */
            const char *c2 = fz_tok_lit(t, fz_rng(s));
            fz_bytes_put(b, c, l - fz_below(s, l));
            fz_bytes_put(b, c2, strlen(c2));
            break;
        }
        case 3: {                                                                    /* a char inside it */
            uint64_t k = fz_below(s, l);
            fz_bytes_put(b, c, k);
            fz_put_cp(b, FZ_CLS[fz_below(s, 6u)][0]);
            fz_bytes_put(b, c + k, l - k);
            break;
        }
        case 4: fz_bytes_put(b, " ", 1u); fz_bytes_put(b, c, l); fz_bytes_put(b, "\n\n", 2u); break;
        default: fz_bytes_put(b, c, l); break;
        }
        break;
    }
    case 8: {                                         /* many short pieces: crosses the 256-piece rounds */
        static const char *const SHORT[] = { "a.", "a b", "1,", " x", "\n", "'s", "\xe4\xb8\xad.", "!?", "a\xc3" };
        fz_put_rep(b, SHORT[fz_below(s, FZ_N(SHORT))], 100u + fz_below(s, 400u));
        break;
    }
    case 9: {                                         /* one long piece: K6's heap path, the bounce */
        uint64_t k = 16u + fz_below(s, fz_below(s, 8u) == 0u ? 6000u : 300u);
        uint32_t c0 = FZ_CP_L[fz_below(s, 10u)], c1 = FZ_CP_L[fz_below(s, FZ_N(FZ_CP_L))];
        for (uint64_t i = 0; i < k; i++) { fz_put_cp(b, fz_below(s, 3u) == 0u ? c1 : c0); }
        break;
    }
    default: {                                        /* any scalar (and, rarely, a surrogate's bytes) */
        uint32_t cp = (uint32_t)fz_below(s, 0x110000u);
        if (cp >= 0xD800u && cp <= 0xDFFFu && fz_below(s, 4u) != 0u) { cp = 0xFFFDu; }
        if (cp >= 0xD800u && cp <= 0xDFFFu) {
            uint8_t u[3] = { (uint8_t)(0xE0u | (cp >> 12)), (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu)), (uint8_t)(0x80u | (cp & 0x3Fu)) };
            fz_bytes_put(b, u, 3u);
        } else {
            fz_put_cp(b, cp);
        }
        break;
    }
    }
}

/* the custom mutator of the text harnesses: libFuzzer's byte mutations a quarter of the time, a header byte now
 * and then, else 1-3 class-aware edits of the text (insert, replace, truncate, splice, regrow) */
FZ_FN size_t fz_text_mutate(uint8_t *d, size_t n, size_t max, unsigned int seed)
{
    uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ull + n;
    if (max <= FZ_HDR + 1u) { return LLVMFuzzerMutate(d, n, max); }
    if (n < FZ_HDR) {
        for (size_t i = n; i < FZ_HDR; i++) { d[i] = (uint8_t)fz_rng(&s); }
        n = FZ_HDR;
    }
    uint64_t r = fz_below(&s, 16u);
    if (r < 4u) { return LLVMFuzzerMutate(d, n, max); }
    if (r == 4u) { d[fz_below(&s, FZ_HDR)] = (uint8_t)fz_rng(&s); return n; }
    const fz_tok *t = fz_pin(d[0]);
    uint64_t ops = 1u + fz_below(&s, 3u);
    for (uint64_t k = 0; k < ops; k++) {
        uint64_t tl = n - FZ_HDR, room = max - FZ_HDR, op = fz_below(&s, 10u);
        uint8_t *x = d + FZ_HDR;
        if (op == 0u && tl != 0u) {                   /* truncate (often mid-sequence) */
            n = FZ_HDR + (size_t)fz_below(&s, tl);
            continue;
        }
        if (op == 1u && tl >= 2u) {                   /* splice a range of the text elsewhere */
            uint64_t a = fz_below(&s, tl), l = 1u + fz_below(&s, tl - a), at = fz_below(&s, tl + 1u);
            if (l > room - tl) { l = room - tl; }
            if (l == 0u) { continue; }
            uint8_t *tmp = (uint8_t *)fz_alloc(l);
            memcpy(tmp, x + a, (size_t)l);
            memmove(x + at + l, x + at, (size_t)(tl - at));
            memcpy(x + at, tmp, (size_t)l);
            free(tmp);
            n += (size_t)l;
            continue;
        }
        fz_bytes frag = { NULL, 0, 0 };
        fz_gen_fragment(&frag, &s, t);
        uint64_t fl = frag.n;
        if (op == 2u) { tl = 0u; n = FZ_HDR; }        /* regrow: a fresh text of a few fragments */
        uint64_t at = fz_below(&s, tl + 1u), cut = 0u;
        if (op == 3u || op == 4u) { cut = fz_below(&s, tl - at + 1u); }   /* replace a range */
        if (fl > room - (tl - cut)) { fl = room - (tl - cut); }
        memmove(x + at + fl, x + at + cut, (size_t)(tl - at - cut));
        memcpy(x + at, frag.p, (size_t)fl);
        n = FZ_HDR + (size_t)(tl - cut + fl);
        fz_bytes_free(&frag);
    }
    return n;
}

#endif /* TOKS_FUZZ_GEN_H */
