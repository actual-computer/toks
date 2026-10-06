/* config.c: tokenizer.json -> toks_config, the one reader of every algorithm (docs/notes/c-core.md §config.c.1) */
#include "core.h"
#include "classes.h"
#include "norm.h"
#include "spm.h"
#include "wp.h"
#include "unigram.h"

/* ---- the known pattern table (exact strings; kernels.md §3) -------------------------------------------- */

const toks_pattern TOKS_PATTERNS[] = {
    /* gpt-2 (ByteLevel use_regex=true) */
    { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN, TOKS_TMPL_CL100K, 0u },
    /* cl100k / llama 3 */
    { "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1_3 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, TOKS_TMPL_CL100K, 0u },
    /* qwen 2 / 2.5 / 3 (\p{N} single) */
    { "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, TOKS_TMPL_CL100K, 0u },
    /* qwen 3.5 / 3.6 / 3.8 (marks folded into letters), the hub's exact string: the letter prefix class leaves
     * \p{M} in (an M either prefixes the L|M run or starts it: same span, kernels.md §3) */
    { "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, TOKS_TMPL_CL100K,
      TOKS_CLASSES_MARKS_ARE_LETTERS },
    /* o200k: gpt-oss, gpt-4o (docs/templates/o200k.md) */
    { "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
      "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
      "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1_3, TOKS_TMPL_O200K, 0u },
    /* o200k, nemo variant: no contraction suffix, one digit (Mistral-Nemo, Nemotron; docs/templates/o200k.md §1) */
    { "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
      "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
      "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_NONE | TOKS_TP_DIGITS_1, TOKS_TMPL_O200K, 0u },
    /* LLaDA / Ling (P22): qwen 2 spelled with possessives, \\s*[\\r\\n] and '(?i:[sdmt]|..) (kernels.md §3: equal) */
    { "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, TOKS_TMPL_CL100K, 0u },
    /* zeta (P25): qwen 2 without the punctuation tail; the hub string holds literal CR LF bytes */
    { "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
      TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_WS_NL, TOKS_TMPL_CL100K, 0u },
    /* bloom (P21): ' '? and one class, \\s and 14 literals out (TOKS_CLASSES_BLOOM): A4 over P runs, A10 */
    { " ?[^(\\s|[.,!?\xE2\x80\xA6\xE3\x80\x82\xEF\xBC\x8C\xE3\x80\x81\xE0\xA5\xA4\xDB\x94\xD8\x8C])]+",
      TOKS_TP_GB_SP, TOKS_TMPL_CL100K, TOKS_CLASSES_BLOOM },
};
const uint32_t TOKS_PATTERNS_N = (uint32_t)(sizeof(TOKS_PATTERNS) / sizeof(TOKS_PATTERNS[0]));

/* digits-gpt2 (SmolLM, PowerMoE, starcoder): pre_tokenizer [Digits(individual_digits), ByteLevel(use_regex)]: every
 * hf Digits numeric char is a piece and ends the gpt-2 regex's input (kernels.md §3 A8) */
const toks_pattern TOKS_PATTERN_DIGITS = {
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_1 | TOKS_TP_DIGIT_CUT, TOKS_TMPL_CL100K, TOKS_CLASSES_DIGITS };

/* MiniCPM5 (P16): [Split \\p{N}{1,3}, Split (this), ByteLevel]: every N run of up to 3 is a piece and ends the
 * regex's input, so this \\p{N}+ sees those runs whole: the cl100k template with A8 (kernels.md §3) */
const toks_pattern TOKS_PATTERN_P16 = {
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1_3 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL | TOKS_TP_DIGIT_CUT,
    TOKS_TMPL_CL100K, 0u };

/* Laguna (P20): [Split (?:\\r?\\n)+(?!\\r?\\n) MergedWithNext, Split (this), ByteLevel]: qwen 2 with A9 */
const toks_pattern TOKS_PATTERN_P20 = {
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL | TOKS_TP_NL_CUT,
    TOKS_TMPL_CL100K, 0u };

/* kimi: tokenization_kimi.py's pat_str (its eight lines joined by '|'; tiktoken.c checks the file holds them) */
const toks_pattern TOKS_PATTERN_KIMI = {
    "[\\p{Han}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    TOKS_TP_CONTR_CI | TOKS_TP_DIGITS_1_3 | TOKS_TP_HAN | TOKS_TP_NO_SLASH, TOKS_TMPL_O200K, TOKS_CLASSES_HAN };

/* deepseek v3's chain (docs/templates/dsv3.md §1), the hub's exact strings: split 2 holds the literal characters
 * U+4E00 '-' U+9FA5 U+3040 '-' U+309F U+30A0 '-' U+30FF (utf-8 below), split 3 literal CR LF bytes. */
const char *const TOKS_DSV3_SPLITS[3] = {
    "\\p{N}{1,3}",
    "[\xE4\xB8\x80-\xE9\xBE\xA5\xE3\x81\x80-\xE3\x82\x9F\xE3\x82\xA0-\xE3\x83\xBF]+",
    ("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
     " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"),
};
const toks_pattern TOKS_PATTERN_DSV3 = {
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
    " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
    0u, TOKS_TMPL_DSV3, TOKS_CLASSES_DSV3 };

#define META_UTF8 "\xE2\x96\x81"   /* U+2581 */

/* a json number whose value is exactly zero in any spelling (0, -0, 0.0, 0e7): every mantissa digit is 0 */
static int jv_num_zero(const jv *v)
{
    if (v == NULL || v->type != JV_NUM || v->s == NULL) { return 0; }
    for (uint32_t i = 0; i < v->s_len; i++) {               /* bound: the lexeme */
        uint8_t c = v->s[i];
        if (c == 'e' || c == 'E') { break; }
        if (c >= '1' && c <= '9') { return 0; }
    }
    return 1;
}

/* a bool field: absent / null -> dflt; else it must be a bool (-1) */
static int jflag(const jv *o, const char *key, int dflt, int *out)
{
    const jv *v = toks_jv_get(o, key);
    if (toks_jnull(v)) { *out = dflt; return 0; }
    if (v->type != JV_BOOL) { return -1; }
    *out = v->num != 0;
    return 0;
}

/* v (a component) or one of its Sequence members (list key) has the given type */
static int has_type(const jv *v, const char *list_key, const char *type)
{
    if (toks_jtype(v, type)) { return 1; }
    const jv *list = toks_jtype(v, "Sequence") ? toks_jv_get(v, list_key) : NULL;
    for (const jv *e = (list != NULL && list->type == JV_ARR) ? list->child : NULL; e != NULL; e = e->next) {
        if (toks_jtype(e, type)) { return 1; }                  /* bound: elements */
    }
    return 0;
}

/* a Replace step's pattern (String or Regex, at most one non-NULL) and content; 0 when n is no Replace with an
 * object pattern */
static int replace_of(const jv *n, const jv **str, const jv **rx, const jv **content)
{
    const jv *pat = toks_jtype(n, "Replace") ? toks_jv_get(n, "pattern") : NULL;
    if (pat == NULL || pat->type != JV_OBJ) { return 0; }
    *str = toks_jv_get(pat, "String");
    *rx = toks_jv_get(pat, "Regex");
    *content = toks_jv_get(n, "content");
    return 1;
}

/* ---- string index (vocab lookups, added-token dedup): open addressing, slot = entry + 1 ---------------- */

static uint32_t str_hash(uint32_t h, const uint8_t *p, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++) { h = TOKS_CRC32C_TAB[(h ^ p[i]) & 0xFFu] ^ (h >> 8); }   /* bound: n */
    return h;
}

int64_t toks_sidx_init(toks_sidx *x, toks_arena *ar, const uint8_t *const *s, const uint32_t *len, uint64_t n)
{
    uint64_t slots = 2u;
    while (slots < 2u * n) { slots <<= 1; }                 /* bound: 24 doublings (n < 2^22) */
    x->s = s;
    x->len = len;
    x->mask = slots - 1u;
    x->slot = (uint32_t *)toks_ar_alloc(ar, slots * 4u, 8u);
    if (x->slot == NULL) { return TOKS_E_NOMEM; }
    memset(x->slot, 0, (size_t)(slots * 4u));
    return 0;
}

/* the entry whose string is a || b (b may be empty), or -1. bound: the table is at most half full. */
int64_t toks_sidx_find(const toks_sidx *x, const uint8_t *a, uint32_t al, const uint8_t *b, uint32_t bl)
{
    uint64_t i = (uint64_t)str_hash(str_hash(TOKS_HSEED, a, al), b, bl) & x->mask;
    for (uint64_t k = 0; k <= x->mask; k++) {               /* bound: mask + 1 slots */
        uint32_t e = x->slot[i];
        if (e == 0u) { return -1; }
        const uint8_t *t = x->s[e - 1u];
        if (x->len[e - 1u] == (uint64_t)al + bl && (al == 0u || memcmp(t, a, al) == 0) &&
            (bl == 0u || memcmp(t + al, b, bl) == 0)) {
            return (int64_t)(e - 1u);
        }
        i = (i + 1u) & x->mask;
    }
    return -1;
}

/* inserts entry e; returns 0, or 1 when an equal string is already present. */
int toks_sidx_add(toks_sidx *x, uint32_t e)
{
    if (toks_sidx_find(x, x->s[e], x->len[e], NULL, 0u) >= 0) { return 1; }
    uint64_t i = (uint64_t)str_hash(TOKS_HSEED, x->s[e], x->len[e]) & x->mask;
    while (x->slot[i] != 0u) { i = (i + 1u) & x->mask; }    /* bound: at most half full */
    x->slot[i] = e + 1u;
    return 0;
}

/* the id of the spm model-vocab string s[0, n), or TOKS_SPM_NONE */
static uint32_t spm_vocab_find(const toks_spm_config *c, const uint8_t *s, uint32_t n)
{
    toks_sidx x = { c->vocab, c->vocab_len, (uint32_t *)(uintptr_t)c->vslot, c->vmask };
    int64_t id = toks_sidx_find(&x, s, n, NULL, 0u);
    return id < 0 ? TOKS_SPM_NONE : (uint32_t)id;
}

/* ---- the byte-level alphabet ----------------------------------------------------------------------------- */

int64_t toks_alpha_bytes(const uint8_t *s, uint32_t n, uint8_t *out)
{
    uint32_t i = 0;
    int64_t m = 0;
    while (i < n) {                                         /* bound: n (i advances >= 1) */
        uint32_t k = toks_utf8_len(s + i, n - i);
        if (k == 0u) { return -1; }
        int32_t b = toks_char_byte((k == 1u) ? (uint32_t)s[i] : toks_cp_decode(s + i, k));
        if (b < 0) { return -1; }
        if (out != NULL) { out[m] = (uint8_t)b; }
        m++;
        i += k;
    }
    return m;
}

uint32_t toks_token_bytes(const uint8_t *s, uint32_t n, uint8_t *out)
{
    int64_t m = toks_alpha_bytes(s, n, out);
    if (m >= 0) { return (uint32_t)m; }
    if (out != NULL && n != 0u) { memcpy(out, s, (size_t)n); }
    return n;
}

/* ---- model: vocab, merges, BPE fields --------------------------------------------------------------------- */

/* model.vocab (token -> id) by id into vx's arrays, indexed. hf allows id gaps (vocab.len() < max id + 1):
 * sentencepiece-style bpe keeps them (holes), every other model needs ids dense 0..n-1. Strings are distinct. */
static int64_t read_vocab(const jv *model, toks_arena *ar, int holes, toks_sidx *vx, uint32_t *n_ids, uint32_t *n_strings,
                          toks_err *err)
{
    const jv *vocab = toks_jv_get(model, "vocab");
    if (vocab == NULL || vocab->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "model.vocab missing"); }
    uint64_t n = 0, max_id = 0;
    for (const jv *m = vocab->child; m != NULL; m = m->next) {   /* bound: members */
        const jv *v = m->child;
        if (v == NULL || v->type != JV_NUM || v->num_float != 0 || v->num < 0) {
            return toks_fail(err, TOKS_E_FORMAT, "model.vocab entry not an id");
        }
        if (v->num >= (int64_t)TOKS_MAX_IDS) { return toks_fail(err, TOKS_E_LIMIT, "model.vocab id >= TOKS_MAX_IDS"); }
        if (m->s_len > TOKS_MAX_TOKEN_BYTES) { return toks_fail(err, TOKS_E_LIMIT, "model.vocab token > TOKS_MAX_TOKEN_BYTES"); }
        if ((uint64_t)v->num > max_id) { max_id = (uint64_t)v->num; }
        n++;
    }
    if (n == 0u) { return toks_fail(err, TOKS_E_FORMAT, "model.vocab empty"); }
    if (!holes && n != max_id + 1u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model.vocab ids not dense"); }
    uint64_t ni = max_id + 1u;
    const uint8_t **vs = (const uint8_t **)toks_ar_alloc(ar, ni * sizeof(uint8_t *), 8u);
    uint32_t *vl = (uint32_t *)toks_ar_alloc(ar, ni * 4u + 4u, 8u);
    if (vs == NULL || vl == NULL || toks_sidx_init(vx, ar, vs, vl, n) != 0) {   /* the index holds the n strings */
        /* the parse arena is 32 B a source byte (config.h): a dense vocab always fits, ids far past the entries not */
        return toks_fail(err, ni > n ? TOKS_E_LIMIT : TOKS_E_NOMEM, ni > n ? "model.vocab: id holes past what the file's size gives the parse arena (32 B a source byte)" : "model.vocab arrays");
    }
    memset(vs, 0, (size_t)(ni * sizeof(uint8_t *)));
    memset(vl, 0, (size_t)(ni * 4u));
    for (const jv *m = vocab->child; m != NULL; m = m->next) {   /* bound: members */
        uint64_t id = (uint64_t)m->child->num;
        if (vs[id] != NULL) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, holes ? "model.vocab: two strings share an id" : "model.vocab ids not dense");
        }
        vs[id] = (m->s != NULL) ? m->s : (const uint8_t *)"";
        vl[id] = m->s_len;
        if (toks_sidx_add(vx, (uint32_t)id) != 0) {
            return toks_fail(err, holes ? TOKS_E_UNSUPPORTED : TOKS_E_FORMAT, "model.vocab repeats a token");
        }
    }
    *n_ids = (uint32_t)ni;
    *n_strings = (uint32_t)n;
    return 0;
}

/* rationale: docs/notes/c-core.md §config.c.2 */
static int64_t read_merges(const jv *model, toks_arena *ar, const toks_sidx *vx, const uint32_t *lro[3], uint32_t *n_out,
                           toks_err *err)
{
    const jv *merges = toks_jv_get(model, "merges");
    if (merges == NULL || merges->type != JV_ARR) { return toks_fail(err, TOKS_E_FORMAT, "model.merges missing"); }
    uint64_t n = 0;
    for (const jv *e = merges->child; e != NULL; e = e->next) { n++; }   /* bound: elements */
    if (n >= (1u << TOKS_PRIO_BITS)) { return toks_fail(err, TOKS_E_LIMIT, "model.merges > 2^22"); }
    uint32_t *ids[3];
    for (uint32_t j = 0; j < 3u; j++) {                         /* bound: 3 */
        ids[j] = (uint32_t *)toks_ar_alloc(ar, n * 4u + 4u, 8u);
        if (ids[j] == NULL) { return toks_fail(err, TOKS_E_NOMEM, "model.merges arrays"); }
        lro[j] = ids[j];
    }
    static const uint8_t VER[8] = { '#', 'v', 'e', 'r', 's', 'i', 'o', 'n' };
    uint32_t k = 0;
    for (const jv *e = merges->child; e != NULL; e = e->next) {        /* bound: n */
        const uint8_t *ls, *rs;
        uint32_t ll, rl;
        if (e->type == JV_STR) {
            if (e->s_len >= 8u && memcmp(e->s, VER, 8) == 0) { continue; }
            uint32_t sp = 0, spaces = 0;
            for (uint32_t i = 0; i < e->s_len; i++) {                  /* bound: s_len */
                if (e->s[i] == ' ') { spaces++; sp = i; }
            }
            if (spaces != 1u) { return toks_fail(err, TOKS_E_FORMAT, "model.merges entry is not 'left right'"); }
            ls = e->s; ll = sp;
            rs = e->s + sp + 1u; rl = e->s_len - sp - 1u;
        } else if (e->type == JV_ARR && e->child != NULL && e->child->next != NULL && e->child->next->next == NULL &&
                   e->child->type == JV_STR && e->child->next->type == JV_STR) {
            ls = e->child->s; ll = e->child->s_len;
            rs = e->child->next->s; rl = e->child->next->s_len;
        } else {
            return toks_fail(err, TOKS_E_FORMAT, "model.merges entry is not a pair");
        }
        int64_t l = toks_sidx_find(vx, ls, ll, NULL, 0u), r = toks_sidx_find(vx, rs, rl, NULL, 0u), o = toks_sidx_find(vx, ls, ll, rs, rl);
        if (l < 0 || r < 0 || o < 0) { return toks_fail(err, TOKS_E_FORMAT, "model.merges token out of vocabulary"); }
        ids[0][k] = (uint32_t)l;
        ids[1][k] = (uint32_t)r;
        ids[2][k] = (uint32_t)o;
        k++;
    }
    *n_out = k;
    return 0;
}

/* rationale: docs/notes/c-core.md §config.c.3 */
static int64_t read_bpe(const jv *model, uint32_t *flags, toks_err *err)
{
    const jv *type = toks_jv_get(model, "type");
    if (type != NULL && !toks_jstr(type, "BPE")) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model type (not BPE, WordPiece or Unigram)"); }
    const jv *dr = toks_jv_get(model, "dropout");
    if (!toks_jnull(dr) && dr->type != JV_NUM) { return toks_fail(err, TOKS_E_FORMAT, "model dropout"); }
    if (!toks_jnull(dr) && !jv_num_zero(dr)) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model dropout (not 0.0)"); }
    const jv *csp = toks_jv_get(model, "continuing_subword_prefix"), *ews = toks_jv_get(model, "end_of_word_suffix");
    if (!toks_jnull(csp) && !toks_jstr(csp, "")) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model continuing_subword_prefix"); }
    if (!toks_jnull(ews) && !toks_jstr(ews, "")) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model end_of_word_suffix"); }
    int bf, fu, im;
    if (jflag(model, "byte_fallback", 0, &bf) != 0 || jflag(model, "fuse_unk", 0, &fu) != 0 ||
        jflag(model, "ignore_merges", 0, &im) != 0) {
        return toks_fail(err, TOKS_E_FORMAT, "model flags");
    }
    *flags = (bf ? TOKS_SPM_BYTE_FALLBACK : 0u) | (fu ? TOKS_SPM_FUSE_UNK : 0u) | (im ? TOKS_SPM_IGNORE_MERGES : 0u);
    return 0;
}

/* ---- components shared by several algorithms ---------------------------------------------------------------- */

/* Metaspace's prepend_scheme (serde default always; add_prefix_space false needs never spelled out) */
static int64_t meta_scheme(const jv *o, uint32_t *scheme, toks_err *err)
{
    const jv *ps = toks_jv_get(o, "prepend_scheme");
    uint32_t s = TOKS_SPM_PS_ALWAYS;
    if (!toks_jnull(ps)) {
        if (toks_jstr(ps, "always")) { s = TOKS_SPM_PS_ALWAYS; }
        else if (toks_jstr(ps, "first")) { s = TOKS_SPM_PS_FIRST; }
        else if (toks_jstr(ps, "never")) { s = TOKS_SPM_PS_NEVER; }
        else { return toks_fail(err, TOKS_E_FORMAT, "Metaspace prepend_scheme"); }
    }
    int aps;
    if (jflag(o, "add_prefix_space", 1, &aps) != 0) { return toks_fail(err, TOKS_E_FORMAT, "Metaspace add_prefix_space"); }
    if (!aps && s != TOKS_SPM_PS_NEVER) {
        return toks_fail(err, TOKS_E_FORMAT, "Metaspace add_prefix_space does not match prepend_scheme");
    }
    *scheme = s;
    return 0;
}

/* ---- added tokens ---------------------------------------------------------------------------------------- */

/* rationale: docs/notes/c-core.md §config.c.4 */
static int64_t read_added(const jv *root, toks_arena *ar, toks_config *cfg, const toks_sidx *vx, toks_err *err)
{
    const jv *list = toks_jv_get(root, "added_tokens");
    cfg->added = NULL;
    cfg->n_added = 0;
    cfg->n_ids = cfg->n_vocab;
    if (toks_jnull(list)) { return 0; }
    if (list->type != JV_ARR) { return toks_fail(err, TOKS_E_FORMAT, "added_tokens not an array"); }
    uint64_t n = 0;
    for (const jv *e = list->child; e != NULL; e = e->next) { n++; }   /* bound: elements */
    if (n > 0xFFFFu) { return toks_fail(err, TOKS_E_LIMIT, "added_tokens > 65535"); }
    if (n == 0u) { return 0; }
    toks_cfg_added *ad = (toks_cfg_added *)toks_ar_alloc(ar, n * sizeof(toks_cfg_added), 8u);
    const uint8_t **cs = (const uint8_t **)toks_ar_alloc(ar, n * sizeof(uint8_t *), 8u);
    uint32_t *cl = (uint32_t *)toks_ar_alloc(ar, n * 4u, 8u);
    toks_sidx ax;
    if (ad == NULL || cs == NULL || cl == NULL || toks_sidx_init(&ax, ar, cs, cl, n) != 0) {
        return toks_fail(err, TOKS_E_NOMEM, "added_tokens arrays");
    }
    uint32_t k = 0, j = 0;                                             /* j: the listing's index in the file's list */
    uint64_t next_id = cfg->n_strings;                                 /* hf: vocab.len() */
    for (const jv *e = list->child; e != NULL; e = e->next, j++) {     /* bound: n */
        if (e->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "added_tokens entry"); }
        const jv *id = toks_jv_get(e, "id"), *content = toks_jv_get(e, "content"), *special = toks_jv_get(e, "special");
        const jv *norm = toks_jv_get(e, "normalized"), *sw = toks_jv_get(e, "single_word");
        const jv *ls = toks_jv_get(e, "lstrip"), *rs = toks_jv_get(e, "rstrip");
        if (!toks_juint(id, 0xFFFFFFFFu) || content == NULL || content->type != JV_STR || !toks_jbool(special) ||
            !toks_jbool(norm) || !toks_jbool(sw) || !toks_jbool(ls) || !toks_jbool(rs)) {
            return toks_fail(err, TOKS_E_FORMAT, "added_tokens entry fields");
        }
        if (content->s_len == 0u) { continue; }                        /* hf ignores it */
        if (content->s_len > TOKS_MAX_ADDED_BYTES) { return toks_fail(err, TOKS_E_LIMIT, "added token > TOKS_MAX_ADDED_BYTES"); }
        cs[k] = content->s;
        cl[k] = content->s_len;
        int64_t prev = toks_sidx_find(&ax, content->s, content->s_len, NULL, 0u);
        toks_cfg_added *a = (prev >= 0) ? &ad[prev] : &ad[k];
        if (prev < 0) {
            (void)toks_sidx_add(&ax, k);                                     /* new: cannot collide */
            int64_t vid = (cfg->spm != NULL) ? (int64_t)spm_vocab_find(cfg->spm, content->s, content->s_len)
                                             : toks_sidx_find(vx, content->s, content->s_len, NULL, 0u);
            if (cfg->spm != NULL && vid == (int64_t)TOKS_SPM_NONE) { vid = -1; }
            uint64_t tid = (vid >= 0) ? (uint64_t)vid : next_id++;
            if (tid >= TOKS_MAX_IDS) { return toks_fail(err, TOKS_E_LIMIT, "added token id >= TOKS_MAX_IDS"); }
            memset(a, 0, sizeof(*a));
            a->content = content->s;
            a->len = content->s_len;
            a->id = (uint32_t)tid;
            if (tid + 1u > cfg->n_ids) { cfg->n_ids = (uint32_t)(tid + 1u); }
            k++;
        }
        a->special = (uint8_t)(a->special | (special->num != 0));      /* specialness sticks (special_tokens_set) */
        a->normalized = (uint8_t)(norm->num != 0);                     /* the last entry's flags */
        a->lstrip = (uint8_t)(ls->num != 0);
        a->rstrip = (uint8_t)(rs->num != 0);
        a->single_word = (uint8_t)(sw->num != 0);
        a->attr = (uint8_t)((special->num != 0 ? TOKS_ID_SPECIAL : 0u) | (ls->num != 0 ? TOKS_ID_LSTRIP : 0u) |
                            (rs->num != 0 ? TOKS_ID_RSTRIP : 0u) | (sw->num != 0 ? TOKS_ID_SINGLE_WORD : 0u) |
                            (norm->num != 0 ? TOKS_ID_NORMALIZED : 0u));   /* the last entry's, special included */
        a->last = j;
    }
    uint32_t rstrip_in[2] = { 0u, 0u };
    for (uint32_t i = 0; i < k; i++) {                                  /* bound: k */
        if (ad[i].rstrip != 0u) { rstrip_in[ad[i].normalized] = 1u; }
    }
    for (uint32_t i = 0; i < k; i++) {                                  /* bound: k */
        if (ad[i].lstrip == 0u || ad[i].rstrip != 0u || rstrip_in[ad[i].normalized] == 0u) { continue; }
        uint32_t p = 0u, all_ws = 1u;
        while (p < ad[i].len && all_ws != 0u) {                         /* bound: len (p advances >= 1) */
            uint32_t q = toks_utf8_len(ad[i].content + p, ad[i].len - p);   /* >= 1: json strings are utf-8 */
            all_ws = (uint32_t)toks_is_regex_ws((q == 1u) ? (uint32_t)ad[i].content[p] : toks_cp_decode(ad[i].content + p, q));
            p += q;
        }
        if (all_ws != 0u) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "added token: all-whitespace lstrip token beside rstrip tokens (hf panics)");
        }
    }
    cfg->added = ad;
    cfg->n_added = k;
    return 0;
}

/* ---- post-processor -------------------------------------------------------------------------------------- */

/* TemplateProcessing's single template, flattened into flat[0, *nf) (each SpecialToken piece expanded to
 * its special_tokens entry's ids, hf apply_template); exactly one $A. */
static int64_t read_template(const jv *tp, const toks_config *cfg, toks_pp_piece flat[64], uint32_t *nf_out,
                             toks_err *err)
{
    uint32_t nf = 0, seen_a = 0;
    const jv *single = toks_jv_get(tp, "single"), *specials = toks_jv_get(tp, "special_tokens");
    if (single == NULL || single->type != JV_ARR || specials == NULL || specials->type != JV_OBJ) {
        return toks_fail(err, TOKS_E_FORMAT, "post_processor TemplateProcessing fields");
    }
    for (const jv *pc = single->child; pc != NULL; pc = pc->next) {    /* bound: pieces */
        const jv *sq = (pc->type == JV_OBJ) ? toks_jv_get(pc, "Sequence") : NULL;
        const jv *st = (pc->type == JV_OBJ) ? toks_jv_get(pc, "SpecialToken") : NULL;
        const jv *pid = toks_jv_get((sq != NULL) ? sq : st, "id"), *ty = toks_jv_get((sq != NULL) ? sq : st, "type_id");
        if ((sq == NULL) == (st == NULL) || pid == NULL || pid->type != JV_STR) {
            return toks_fail(err, TOKS_E_FORMAT, "post_processor template piece");
        }
        uint32_t type = toks_juint(ty, 0xFFFFFFFFu) ? (uint32_t)ty->num : 0u;    /* hf Encoding.type_ids (toks_template) */
        if (sq != NULL) {
            if (!toks_jstr(pid, "A")) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor single template uses $B"); }
            if (seen_a != 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor template: $A twice"); }
            if (nf >= 64u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor template > 64 pieces"); }
            seen_a = 1u;
            flat[nf].kind = TOKS_PPS_SEQ;
            flat[nf].id = 0;
            flat[nf].type = type;
            nf++;
            continue;
        }
        const jv *sp = NULL;     /* the special_tokens entry keyed by the piece's id (last member wins) */
        for (const jv *m = specials->child; m != NULL; m = m->next) {  /* bound: members */
            if (m->s_len == pid->s_len && (m->s_len == 0u || memcmp(m->s, pid->s, m->s_len) == 0)) { sp = m->child; }
        }
        const jv *ids = (sp != NULL && sp->type == JV_OBJ) ? toks_jv_get(sp, "ids") : NULL;
        if (ids == NULL || ids->type != JV_ARR) { return toks_fail(err, TOKS_E_FORMAT, "post_processor SpecialToken not in special_tokens"); }
        for (const jv *v = ids->child; v != NULL; v = v->next) {       /* bound: ids */
            if (v->type != JV_NUM || v->num_float != 0 || v->num < 0) { return toks_fail(err, TOKS_E_FORMAT, "post_processor special ids"); }
            if (v->num >= (int64_t)cfg->n_ids) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor id beyond the vocabulary"); }
            if (nf >= 64u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor template > 64 pieces"); }
            flat[nf].kind = TOKS_PPS_TOK;
            flat[nf].id = (uint32_t)v->num;
            flat[nf].type = type;
            nf++;
        }
    }
    if (seen_a == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor template without $A"); }
    *nf_out = nf;
    return 0;
}

/* one (String, u32) pair of RobertaProcessing / BertProcessing (serde: a two-element array). */
static int pp_pair(const jv *v, uint32_t *id)
{
    if (v == NULL || v->type != JV_ARR || v->child == NULL || v->child->type != JV_STR) { return 0; }
    const jv *n = v->child->next;
    if (!toks_juint(n, 0xFFFFFFFFu) || n->next != NULL) { return 0; }
    *id = (uint32_t)n->num;
    return 1;
}

/* rationale: docs/notes/c-core.md §config.c.5 */
enum { PP_NONE = 0, PP_CLS_SEP, PP_BYTELEVEL, PP_TEMPLATE, PP_SEQUENCE };

static uint32_t pp_kind(const jv *e, uint32_t *cls, uint32_t *sep)
{
    if (e->type != JV_OBJ) { return PP_NONE; }
    if (pp_pair(toks_jv_get(e, "sep"), sep) && pp_pair(toks_jv_get(e, "cls"), cls)) { return PP_CLS_SEP; }
    const jv *ur = toks_jv_get(e, "use_regex");
    if (toks_jtype(e, "ByteLevel") && toks_jbool(toks_jv_get(e, "add_prefix_space")) &&
        toks_jbool(toks_jv_get(e, "trim_offsets")) && (ur == NULL || ur->type == JV_BOOL)) {
        return PP_BYTELEVEL;
    }
    const jv *single = toks_jv_get(e, "single"), *pair = toks_jv_get(e, "pair"), *sp = toks_jv_get(e, "special_tokens");
    if (single != NULL && single->type == JV_ARR && pair != NULL && pair->type == JV_ARR && sp != NULL && sp->type == JV_OBJ) {
        return PP_TEMPLATE;
    }
    const jv *list = toks_jv_get(e, "processors");
    if (toks_jtype(e, "Sequence") && list != NULL && list->type == JV_ARR) { return PP_SEQUENCE; }
    return PP_NONE;
}

/* the single-sequence ids of the post-processor, flattened into pp_single (prefix ids, $A, suffix ids). */
static int64_t read_post_processor(const jv *root, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    const jv *pp = toks_jv_get(root, "post_processor");
    cfg->pp_single = NULL;
    cfg->n_pp_single = 0;
    if (toks_jnull(pp)) { return 0; }
    if (pp->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "post_processor"); }
    uint32_t cls = 0, sep = 0, tp_kind = PP_NONE, tp_cls = 0, tp_sep = 0;
    int seq = pp_kind(pp, &cls, &sep) == PP_SEQUENCE;
    const jv *tp = NULL;
    for (const jv *e = seq ? toks_jv_get(pp, "processors")->child : pp; e != NULL; e = seq ? e->next : NULL) {
        uint32_t k = pp_kind(e, &cls, &sep);                            /* bound: elements */
        if (k == PP_BYTELEVEL) { continue; }
        if (k == PP_NONE) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor type (ByteLevel, TemplateProcessing, RobertaProcessing, BertProcessing)"); }
        if (k == PP_SEQUENCE) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor: Sequence inside a Sequence"); }
        if (tp != NULL) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor: two that add ids"); }
        tp = e;
        tp_kind = k;
        tp_cls = cls;
        tp_sep = sep;
    }
    if (tp == NULL) { return 0; }
    toks_pp_piece flat[64];
    uint32_t nf = 3u;
    if (tp_kind == PP_CLS_SEP) {
        if (tp_cls >= cfg->n_ids || tp_sep >= cfg->n_ids) { return toks_fail(err, TOKS_E_UNSUPPORTED, "post_processor id beyond the vocabulary"); }
        memset(flat, 0, 3u * sizeof flat[0]);                           /* type ids 0 (hf's single sequence) */
        flat[0].kind = TOKS_PPS_TOK;
        flat[0].id = tp_cls;
        flat[1].kind = TOKS_PPS_SEQ;
        flat[1].id = 0;
        flat[2].kind = TOKS_PPS_TOK;
        flat[2].id = tp_sep;
    } else {
        int64_t r = read_template(tp, cfg, flat, &nf, err);
        if (r != 0) { return r; }
    }
    toks_pp_piece *out = (toks_pp_piece *)toks_ar_alloc(ar, (uint64_t)nf * sizeof(toks_pp_piece), 8u);
    if (out == NULL) { return toks_fail(err, TOKS_E_NOMEM, "post_processor pieces"); }
    memcpy(out, flat, (size_t)nf * sizeof(toks_pp_piece));
    cfg->pp_single = out;
    cfg->n_pp_single = nf;
    return 0;
}

/* ---- truncation and padding (hf encode() applies them after the post-processor; wordpiece.md §7) ---------- */

static int64_t read_trunc_pad(const jv *root, toks_config *cfg, toks_err *err)
{
    const jv *tr = toks_jv_get(root, "truncation");
    if (!toks_jnull(tr)) {
        if (tr->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "truncation"); }
        const jv *dir = toks_jv_get(tr, "direction"), *st = toks_jv_get(tr, "strategy");
        const jv *ml = toks_jv_get(tr, "max_length"), *sd = toks_jv_get(tr, "stride");
        if (!toks_juint(ml, UINT64_MAX >> 1) || !toks_juint(sd, UINT64_MAX >> 1) || st == NULL || st->type != JV_STR ||
            (dir != NULL && dir->type != JV_STR)) {
            return toks_fail(err, TOKS_E_FORMAT, "truncation fields");
        }
        if (dir != NULL && !toks_jstr(dir, "Right")) {
            return toks_fail(err, toks_jstr(dir, "Left") ? TOKS_E_UNSUPPORTED : TOKS_E_FORMAT,
                             toks_jstr(dir, "Left") ? "truncation direction Left" : "truncation direction");
        }
        if (toks_jstr(st, "OnlySecond")) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "truncation OnlySecond (hf fails every long single-sequence encode)");
        }
        if (!toks_jstr(st, "LongestFirst") && !toks_jstr(st, "OnlyFirst")) { return toks_fail(err, TOKS_E_FORMAT, "truncation strategy"); }
        cfg->o.trunc_on = 1;
        cfg->o.trunc_max = (uint64_t)ml->num > TOKS_MAX_TEXT ? (uint32_t)TOKS_MAX_TEXT : (uint32_t)ml->num;
        cfg->o.trunc_stride = (uint64_t)sd->num > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)sd->num;
    }
    const jv *pd = toks_jv_get(root, "padding");
    if (toks_jnull(pd)) { return 0; }
    if (pd->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "padding"); }
    const jv *st = toks_jv_get(pd, "strategy"), *dir = toks_jv_get(pd, "direction");
    const jv *mul = toks_jv_get(pd, "pad_to_multiple_of"), *ptok = toks_jv_get(pd, "pad_token");
    const jv *pid = toks_jv_get(pd, "pad_id"), *fx = (st != NULL && st->type == JV_OBJ) ? toks_jv_get(st, "Fixed") : NULL;
    if (!toks_juint(pid, 0xFFFFFFFFu) || !toks_juint(toks_jv_get(pd, "pad_type_id"), 0xFFFFFFFFu) || ptok == NULL ||
        ptok->type != JV_STR || dir == NULL || dir->type != JV_STR || st == NULL ||
        (!toks_jnull(mul) && !toks_juint(mul, UINT64_MAX >> 1))) {
        return toks_fail(err, TOKS_E_FORMAT, "padding fields");
    }
    if (toks_juint(fx, UINT64_MAX >> 1)) {
        if ((uint64_t)fx->num > TOKS_MAX_TEXT) { return toks_fail(err, TOKS_E_UNSUPPORTED, "padding Fixed above 2^29"); }
        cfg->o.pad_fixed = 1;
        cfg->o.pad_len = (uint32_t)fx->num;
    } else if (!toks_jstr(st, "BatchLongest")) {
        return toks_fail(err, TOKS_E_FORMAT, "padding strategy");
    }
    if (!toks_jstr(dir, "Left") && !toks_jstr(dir, "Right")) { return toks_fail(err, TOKS_E_FORMAT, "padding direction"); }
    uint64_t m = toks_jnull(mul) ? 0u : (uint64_t)mul->num;
    if (m > (1u << 20)) { return toks_fail(err, TOKS_E_UNSUPPORTED, "padding pad_to_multiple_of above 2^20"); }
    cfg->o.pad_left = (uint32_t)toks_jstr(dir, "Left");
    cfg->o.pad_multiple = (uint32_t)m;
    cfg->o.pad_id = (uint32_t)pid->num;
    cfg->o.pad_type_id = (uint32_t)toks_jv_get(pd, "pad_type_id")->num;   /* checked above */
    cfg->o.pad_file = 1u;
    cfg->o.pad_on = (uint8_t)(cfg->o.pad_fixed != 0u || m > 0u);    /* BatchLongest alone: one sequence, nothing */
    return 0;
}

/* once the post-processor is read: what hf would wrap or panic on (wordpiece.md §7.3), a pad id beyond the ids */
static int64_t trunc_check(const toks_config *cfg, toks_err *err)
{
    if (cfg->o.pad_on && cfg->o.pad_id >= cfg->n_ids) { return toks_fail(err, TOKS_E_UNSUPPORTED, "padding pad_id beyond the vocabulary"); }
    if (!cfg->o.trunc_on) { return 0; }
    uint32_t n_added = cfg->n_pp_single > 0u ? cfg->n_pp_single - 1u : 0u;    /* the template's ids */
    if (cfg->o.trunc_max < n_added) {                                 /* hf's usize max_length - n_added wraps */
        return toks_fail(err, TOKS_E_UNSUPPORTED, "truncation max_length below the post-processor's ids");
    }
    uint32_t l_pp = cfg->o.trunc_max - n_added, l_raw = cfg->o.trunc_max;   /* Encoding::truncate panics on stride >= */
    if ((l_pp > 0u && cfg->o.trunc_stride >= l_pp) || (l_raw > 0u && cfg->o.trunc_stride >= l_raw)) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "truncation stride >= max_length (hf panics)");
    }
    return 0;
}

/* ---- components: one reader for the normalizer, the pre-tokenizer and the decoder ---------------------------- */

/* every step type hf 0.23.2 reads, by component; an algorithm accepts a set of them (a mask of indices) and reads
 * the fields of the ones it runs. A refused step's diag is its entry. */
enum {
    N_NFC, N_NFD, N_NFKC, N_NFKD, N_LOWER, N_STRIPACC, N_STRIP, N_REPLACE, N_PREPEND, N_PRECOMPILED, N_BERT, N_NMT,
    N_BYTELEVEL, P_BYTELEVEL, P_SPLIT, P_METASPACE, P_WSSPLIT, P_BERT, P_WS, P_DIGITS, P_PUNCT, P_SCRIPTS, P_CHARDELIM,
    P_FIXED, D_BYTELEVEL, D_METASPACE, D_REPLACE, D_BYTEFALLBACK, D_FUSE, D_STRIP, D_WORDPIECE, D_BPE, D_CTC, STEP_N
};
static const char *const STEP[STEP_N] = {
    "normalizer NFC", "normalizer NFD", "normalizer NFKC", "normalizer NFKD", "normalizer Lowercase",
    "normalizer StripAccents", "normalizer Strip", "normalizer Replace", "normalizer Prepend", "normalizer Precompiled",
    "normalizer BertNormalizer", "normalizer Nmt", "normalizer ByteLevel", "pre_tokenizer ByteLevel",
    "pre_tokenizer Split", "pre_tokenizer Metaspace", "pre_tokenizer WhitespaceSplit", "pre_tokenizer BertPreTokenizer",
    "pre_tokenizer Whitespace", "pre_tokenizer Digits", "pre_tokenizer Punctuation", "pre_tokenizer UnicodeScripts",
    "pre_tokenizer CharDelimiterSplit", "pre_tokenizer FixedLength", "decoder ByteLevel", "decoder Metaspace",
    "decoder Replace", "decoder ByteFallback", "decoder Fuse", "decoder Strip", "decoder WordPiece", "decoder BPEDecoder",
    "decoder CTC",
};
#define BIT(k) (1ull << (k))
enum { C_NORM = 0, C_PRE = 1, C_DEC = 2 };

typedef struct steps {
    const jv *o[16];
    uint8_t   k[16];
    uint32_t  n;
    uint32_t  present;      /* the component is there (an empty Sequence is not "none": hf's decode then joins with "") */
} steps;

/* root's component c flattened into s: Sequences of any depth (to 4) in order, as hf applies them; a step whose type
 * is outside accept is refused by name */
static int64_t read_steps(const jv *root, uint32_t c, uint64_t accept, steps *s, toks_err *err)
{
    static const char *const KEY[3] = { "normalizer", "pre_tokenizer", "decoder" };
    static const char *const LIST[3] = { "normalizers", "pretokenizers", "decoders" };
    static const char *const BAD[3] = { "normalizer Sequence list", "pre_tokenizer Sequence list", "decoder Sequence list" };
    static const char *const UNKNOWN[3] = { "normalizer (unknown type)", "pre_tokenizer (unknown type)", "decoder (unknown type)" };
    static const uint8_t FIRST[4] = { N_NFC, P_BYTELEVEL, D_BYTELEVEL, STEP_N }, SKIP[3] = { 11, 14, 8 };
    const jv *stk[4], *e = toks_jv_get(root, KEY[c]);
    uint32_t d = 0, top = 1;
    s->n = 0;
    s->present = !toks_jnull(e);
    while (s->present && (e != NULL || d > 0u)) {               /* bound: the component's nodes */
        if (e == NULL) { e = stk[--d]; continue; }
        const jv *next = top ? NULL : e->next;
        top = 0;
        if (e->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, KEY[c]); }
        if (toks_jtype(e, "Sequence")) {
            const jv *l = toks_jv_get(e, LIST[c]);
            if (l == NULL || l->type != JV_ARR) { return toks_fail(err, TOKS_E_FORMAT, BAD[c]); }
            if (d == 4u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "component Sequences nested deeper than 4"); }
            stk[d++] = next;
            e = l->child;
            continue;
        }
        uint32_t k = FIRST[c];
        while (k < FIRST[c + 1u] && !toks_jtype(e, STEP[k] + SKIP[c])) { k++; }   /* bound: STEP_N */
        if (k == FIRST[c + 1u]) { return toks_fail(err, TOKS_E_UNSUPPORTED, UNKNOWN[c]); }
        if ((accept & BIT(k)) == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, STEP[k]); }
        if (s->n == 16u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "component: more than 16 steps"); }
        s->o[s->n] = e;
        s->k[s->n++] = (uint8_t)k;
        e = next;
    }
    return 0;
}

/* ---- byte-level bpe ---------------------------------------------------------------------------------------- */

/* NFC / NFKC run (cfg->nfc: the form, norm.h; any mix of them is NFKC when one is, else NFC), no step normalizes
 * nothing (deepseek v3) */
static int64_t bl_normalizer(const jv *root, toks_config *cfg, toks_err *err)
{
    steps s;
    int64_t r = read_steps(root, C_NORM, BIT(N_NFC) | BIT(N_NFKC), &s, err);
    for (uint32_t k = 0; k < s.n; k++) { cfg->nfc |= (uint8_t)(s.k[k] == N_NFKC ? TOKS_NS_NFKC : TOKS_NS_NFC); }   /* bound: 16 */
    return r;
}

/* ByteLevel: add_prefix_space is a required bool (hf) and must be false; use_regex defaults to true. */
static int64_t bl_bytelevel(const jv *bl, int *use_regex, toks_err *err)
{
    const jv *aps = toks_jv_get(bl, "add_prefix_space"), *ur = toks_jv_get(bl, "use_regex");
    if (!toks_jbool(aps) || (ur != NULL && ur->type != JV_BOOL)) { return toks_fail(err, TOKS_E_FORMAT, "pre_tokenizer ByteLevel fields"); }
    if (aps->num != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer ByteLevel add_prefix_space=true"); }
    *use_regex = (ur == NULL || ur->num != 0);
    return 0;
}

/* hf's SplitDelimiterBehavior (TOKS_GB_*), 5 for none of them */
static uint32_t beh_of(const jv *b)
{
    static const char *const BEH[5] = { "Isolated", "Removed", "MergedWithPrevious", "MergedWithNext", "Contiguous" };
    uint32_t k = 0;
    while (k < 5u && !toks_jstr(b, BEH[k])) { k++; }        /* bound: 5 */
    return k;
}

/* a Split step: hf's behavior, invert, its Regex (*re) or String pattern */
static int64_t bl_split(const jv *s, toks_gen_spec *g, const jv **re, toks_err *err)
{
    const jv *inv = toks_jv_get(s, "invert"), *pat = toks_jv_get(s, "pattern");
    const jv *str = (pat != NULL && pat->type == JV_OBJ) ? toks_jv_get(pat, "String") : NULL, *p;
    *re = (pat != NULL && pat->type == JV_OBJ) ? toks_jv_get(pat, "Regex") : NULL;
    p = (*re != NULL) ? *re : str;
    g->beh = beh_of(toks_jv_get(s, "behavior"));
    if (g->beh == 5u || !toks_jbool(inv) || p == NULL || p->type != JV_STR) {
        return toks_fail(err, TOKS_E_FORMAT, "pre_tokenizer Split fields");
    }
    g->inv = (uint32_t)(inv->num != 0);
    g->lit = (uint32_t)(*re == NULL);
    g->s = p->s;
    g->n = p->s_len;
    return 0;
}

/* rationale: docs/notes/c-core.md §config.c.6 */
static int64_t bl_pretok(const jv *root, toks_config *cfg, toks_arena *ar, toks_err *err)
{
    steps s;
    toks_gen_spec g[16];
    const jv *re[16];
    int v = 0, bl = 0;
    int64_t r = read_steps(root, C_PRE, BIT(P_BYTELEVEL) | BIT(P_SPLIT) | BIT(P_DIGITS) | BIT(P_PUNCT), &s, err);
    uint32_t n = s.n, k = 0;
    for (uint32_t i = 0; r == 0 && i < n; i++) {                /* bound: 16 steps */
        memset(&g[i], 0, sizeof g[i]);
        re[i] = NULL;
        if (s.k[i] == P_SPLIT) {
            g[i].kind = TOKS_GS_SPLIT;
            r = bl_split(s.o[i], &g[i], &re[i], err);
        } else if (s.k[i] == P_BYTELEVEL) {
            g[i].kind = TOKS_GS_BYTELEVEL;
            r = bl_bytelevel(s.o[i], &v, err);
            g[i].lit = (uint32_t)v;
            bl = 1;
        } else if (s.k[i] == P_DIGITS) {
            g[i].kind = TOKS_GS_DIGITS;
            r = jflag(s.o[i], "individual_digits", 0, &v) != 0 ? toks_fail(err, TOKS_E_FORMAT, "pre_tokenizer Digits") : 0;
            g[i].beh = v ? TOKS_GB_ISOLATED : TOKS_GB_CONTIGUOUS;
        } else {                                                /* Punctuation: its behavior, Isolated by default */
            const jv *b = toks_jv_get(s.o[i], "behavior");
            g[i].kind = TOKS_GS_PUNCT;
            g[i].beh = toks_jnull(b) ? TOKS_GB_ISOLATED : beh_of(b);
            if (g[i].beh == 5u) { r = toks_fail(err, TOKS_E_FORMAT, "pre_tokenizer Punctuation behavior"); }
        }
    }
    if (r == 0 && !bl) {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, n == 0u ? "pre_tokenizer absent (byte-level bpe needs ByteLevel)" : "pre_tokenizer without ByteLevel (byte-level bpe)");
    }
    if (r != 0) { return r; }
    /* the compiled templates first (kernels.md §3): gpt-2, digits-gpt2, one known Split (Isolated, or Removed +
     * invert: docs/breadth.md §1.1), MiniCPM5's two, dsv3's three; else the generic engine */
    const toks_gen_spec *z = &g[n - 1u];
    if (n == 1u && z->kind == TOKS_GS_BYTELEVEL) { cfg->pattern = z->lit ? &TOKS_PATTERNS[0] : NULL; return 0; }
    if (n == 3u && re[0] != NULL && re[1] != NULL && g[0].beh == TOKS_GB_NEXT && !g[0].inv && g[1].beh == TOKS_GB_ISOLATED &&
        z->kind == TOKS_GS_BYTELEVEL && !z->lit && toks_jstr(re[0], "(?:\\r?\\n)+(?!\\r?\\n)") && toks_jstr(re[1], TOKS_PATTERN_P20.regex)) {
        cfg->pattern = &TOKS_PATTERN_P20;                       /* Laguna (P20) */
        return 0;
    }
    if (n == 2u && g[0].kind == TOKS_GS_DIGITS && g[0].beh == TOKS_GB_ISOLATED && z->kind == TOKS_GS_BYTELEVEL && z->lit) {
        cfg->pattern = &TOKS_PATTERN_DIGITS;
        return 0;
    }
    while (k + 1u < n && re[k] != NULL && (g[k].beh == TOKS_GB_ISOLATED || (n == 2u && g[k].beh == TOKS_GB_REMOVED && g[k].inv)) &&
           (n != 4u || toks_jstr(re[k], TOKS_DSV3_SPLITS[k]))) {
        k++;                                                    /* bound: n */
    }
    if (k + 1u == n && n >= 2u && n <= 4u && z->kind == TOKS_GS_BYTELEVEL && !z->lit) {
        if (n == 4u) { cfg->pattern = &TOKS_PATTERN_DSV3; return 0; }
        if (n == 3u && toks_jstr(re[0], TOKS_DSV3_SPLITS[0]) && toks_jstr(re[1], TOKS_PATTERN_P16.regex)) {
            cfg->pattern = &TOKS_PATTERN_P16;
            return 0;
        }
        for (uint32_t i = 0; n == 2u && i < TOKS_PATTERNS_N; i++) {   /* bound: TOKS_PATTERNS_N */
            if (toks_jstr(re[0], TOKS_PATTERNS[i].regex)) { cfg->pattern = &TOKS_PATTERNS[i]; return 0; }
        }
    }
    return toks_gen_compile(g, n, ar, &cfg->gen, &cfg->gen_bytes, err);   /* the generic engine (gen.c) */
}

/* ByteLevel once. An absent decoder makes hf join the alphabet strings with spaces, which the raw token bytes
 * cannot give back. */
static int64_t bl_decoder(const jv *root, toks_config *cfg, toks_err *err)
{
    steps s;
    int64_t r = read_steps(root, C_DEC, BIT(D_BYTELEVEL), &s, err);
    if (r == 0 && s.n != 1u) {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, s.n == 0u ? "decoder absent (byte-level BPE needs ByteLevel)" : "decoder Sequence (byte-level BPE needs [ByteLevel])");
    }
    cfg->dec_byte_level = (uint8_t)(r == 0);
    return r;
}

/* rationale: docs/notes/c-core.md §config.c.7 */
static int64_t bl_alphabet(const jv *model, toks_config *cfg, const toks_sidx *vx, toks_err *err)
{
    uint8_t seen[256];
    uint32_t n_seen = 0, n_raw = 0;
    memset(seen, 0, sizeof(seen));
    for (uint32_t id = 0; id < cfg->n_vocab; id++) {            /* bound: n_vocab */
        uint8_t b = 0;
        int64_t m = toks_alpha_bytes(cfg->vocab[id], cfg->vocab_len[id], NULL);
        if (m < 0) { n_raw++; continue; }
        if (m == 1 && toks_alpha_bytes(cfg->vocab[id], cfg->vocab_len[id], &b) == 1 && seen[b] == 0u) { seen[b] = 1u; n_seen++; }
    }
    cfg->n_vocab_raw = n_raw;
    if (n_seen == 256u) { return 0; }
    const jv *unk = toks_jv_get(model, "unk_token");
    if (toks_jtrue(toks_jv_get(model, "byte_fallback"))) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "model.vocab lacks byte-level alphabet chars (with byte_fallback)");
    }
    if (!toks_jnull(unk)) {                                      /* hf puts unk_token's id in the word for the char */
        int64_t id = (unk->type == JV_STR) ? toks_sidx_find(vx, unk->s, unk->s_len, NULL, 0u) : -1;
        if (id < 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "model.unk_token not in model.vocab (a byte-level vocab lacking alphabet chars)"); }
        cfg->drop_unk = (uint32_t)id + 1u;
        cfg->drop_fuse = (uint32_t)toks_jtrue(toks_jv_get(model, "fuse_unk"));
    }
    if (cfg->ignore_merges != 0u) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "model.vocab lacks byte-level alphabet chars (under ignore_merges)");
    }
    for (uint32_t b = 0; b < 256u; b++) {                       /* bound: 256 */
        if (seen[b] == 0u) { cfg->drop[b >> 3] = (uint8_t)(cfg->drop[b >> 3] | (1u << (b & 7u))); }
    }
    cfg->has_drop = 1u;
    return 0;
}

static int64_t parse_bytelevel(const jv *root, const jv *model, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    uint32_t fl = 0;
    int64_t r = read_bpe(model, &fl, err);
    if (r == 0) { r = bl_normalizer(root, cfg, err); }
    if (r == 0) { r = bl_pretok(root, cfg, ar, err); }
    if (r == 0) { r = bl_decoder(root, cfg, err); }
    if (r != 0) { return r; }
    cfg->ignore_merges = (uint8_t)((fl & TOKS_SPM_IGNORE_MERGES) != 0u);
    toks_sidx vx;
    const uint32_t *lro[3];
    r = read_vocab(model, ar, 0, &vx, &cfg->n_vocab, &cfg->n_strings, err);
    if (r == 0) { cfg->vocab = vx.s; cfg->vocab_len = vx.len; r = bl_alphabet(model, cfg, &vx, err); }
    if (r == 0) { r = read_merges(model, ar, &vx, lro, &cfg->n_merges, err); }
    for (uint32_t i = 0; r == 0 && cfg->drop_unk != 0u && i < cfg->n_merges; i++) {   /* bound: n_merges */
        if (lro[0][i] + 1u == cfg->drop_unk || lro[1][i] + 1u == cfg->drop_unk) {     /* api.c run_drop's premise */
            r = toks_fail(err, TOKS_E_UNSUPPORTED, "a merge of model.unk_token (a byte-level vocab lacking alphabet chars)");
        }
    }
    if (r == 0) { r = read_added(root, ar, cfg, &vx, err); }
    if (r != 0) { return r; }
    cfg->m_left_id = lro[0];
    cfg->m_right_id = lro[1];
    cfg->m_out_id = lro[2];
    /* hf matches a normalized:true token by its content's normalized form (added_vocabulary.rs normalized_cache);
     * phase 1 indexes the content as written: equal under NFC / NFKC only for a stable content */
    for (uint32_t i = 0; i < cfg->n_added && cfg->nfc != 0u; i++) {    /* bound: n_added */
        uint64_t run_end = 0;
        if (cfg->added[i].normalized != 0u &&
            toks_nfc_scan(cfg->nfc, cfg->added[i].content, cfg->added[i].len, 0u, &run_end) != cfg->added[i].len) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "added token normalized=true whose content the normalizer changes");
        }
    }
    return read_post_processor(root, ar, cfg, err);
}

/* ---- sentencepiece-style bpe (docs/algorithms/spm_bpe.md §11 is the accepted set) -------------------------- */

static int set_str(toks_spm_str *d, const jv *v)
{
    if (v == NULL || v->type != JV_STR || v->s_len > TOKS_SPM_MAX_STR) { return -1; }
    memset(d, 0, sizeof(*d));
    if (v->s_len != 0u) { memcpy(d->b, v->s, v->s_len); }
    d->n = v->s_len;
    return 0;
}

static int one_char(const toks_spm_str *s) { return s->n >= 1u && toks_utf8_len(s->b, s->n) == s->n; }

/* one normalizer step (Prepend / Replace of a String) */
static int64_t spm_norm_step(const jv *n, uint32_t k, toks_spm_text *tx, toks_err *err)
{
    const jv *ps = NULL, *rx = NULL, *con = NULL;
    if (tx->n_norm >= TOKS_SPM_MAX_OPS) { return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer: more than 8 steps"); }
    toks_spm_op *op = &tx->norm[tx->n_norm++];
    memset(op, 0, sizeof(*op));
    if (k == N_PREPEND) {
        op->kind = TOKS_SPM_N_PREPEND;
        return set_str(&op->a, toks_jv_get(n, "prepend")) != 0 ? toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Prepend string") : 0;
    }
    (void)replace_of(n, &ps, &rx, &con);
    if (rx != NULL) { return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace Regex"); }
    if (set_str(&op->a, ps) != 0 || op->a.n == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace pattern"); }
    if (set_str(&op->b, con) != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace content"); }
    op->kind = TOKS_SPM_N_REPLACE;
    return 0;
}

/* Replace ' ' -> ▁: 1, Prepend ▁: 2, else 0 */
static uint32_t meta_op(const toks_spm_op *o)
{
    if (o->kind == TOKS_SPM_N_PREPEND) { return o->a.n == 3u && toks_meta_at(o->a.b, 3u, 0u) ? 2u : 0u; }
    return o->a.n == 1u && o->a.b[0] == 0x20u && o->b.n == 3u && toks_meta_at(o->b.b, 3u, 0u);
}

/* a Split pre-tokenizer that never cuts: invert false, a one-char pattern the normalizer's last step replaces
 * everywhere by a content without that char, and no Metaspace (which writes new chars) */
static int split_is_noop(const jv *sp, const toks_spm_text *tx)
{
    const jv *pat = toks_jv_get(sp, "pattern"), *inv = toks_jv_get(sp, "invert"), *beh = toks_jv_get(sp, "behavior");
    toks_spm_str p;
    if (set_str(&p, (pat != NULL && pat->type == JV_OBJ) ? toks_jv_get(pat, "String") : NULL) != 0 || !one_char(&p) ||
        beh == NULL || beh->type != JV_STR || tx->n_norm == 0u || (!toks_jnull(inv) && !(toks_jbool(inv) && inv->num == 0))) {
        return 0;
    }
    const toks_spm_op *last = &tx->norm[tx->n_norm - 1u];
    if (last->kind != TOKS_SPM_N_REPLACE || last->a.n != p.n || memcmp(last->a.b, p.b, p.n) != 0) { return 0; }
    for (uint32_t i = 0; i + p.n <= last->b.n; i++) {          /* bound: content bytes */
        if (memcmp(last->b.b + i, p.b, p.n) == 0) { return 0; }
    }
    return 1;
}

/* Metaspace's fields, pre-tokenizer or decoder: replacement (one char), split (default true), prepend_scheme */
static int64_t read_meta(const jv *o, toks_spm_str *repl, uint32_t *scheme, uint32_t *split, toks_err *err)
{
    int sp;
    if (set_str(repl, toks_jv_get(o, "replacement")) != 0 || !one_char(repl)) { return toks_fail(err, TOKS_E_FORMAT, "Metaspace replacement"); }
    if (jflag(o, "split", 1, &sp) != 0) { return toks_fail(err, TOKS_E_FORMAT, "Metaspace split"); }
    *split = (uint32_t)sp;
    return meta_scheme(o, scheme, err);
}

static int64_t spm_pretok_step(const jv *pt, uint32_t k, toks_spm_text *tx, toks_err *err)
{
    if (k == P_SPLIT) {
        return (tx->metaspace || !split_is_noop(pt, tx)) ? toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer Split (one that can cut)") : 0;
    }
    if (tx->metaspace) { return toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer: two Metaspace"); }
    int64_t r = read_meta(pt, &tx->ms_repl, &tx->ms_scheme, &tx->ms_split, err);
    tx->metaspace = 1u;
    if (r == 0 && tx->ms_scheme == TOKS_SPM_PS_FIRST && tx->n_norm != 0u) {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, "Metaspace prepend_scheme first after a normalizer");
    }
    return r;
}

/* a char of hf ByteFallback's token test ("<0x" h h ">", u8::from_str_radix accepts a '+'): a per-token step that
 * writes or removes one could change which strings are byte tokens */
static int clear_of_byte_tokens(const toks_spm_str *x)
{
    for (uint32_t i = 0; i < x->n; i++) {                   /* bound: TOKS_SPM_MAX_STR */
        uint8_t c = x->b[i];
        if (c == '<' || c == '>' || c == 'x' || c == '+' || (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') ||
            (c >= 'a' && c <= 'f')) {
            return 0;
        }
    }
    return 1;
}

static int64_t spm_dec_step(const jv *d, uint32_t k, toks_spm_config *sc, toks_err *err)
{
    const jv *ps = NULL, *rx = NULL, *con = NULL;
    uint32_t split;
    if (sc->n_dec >= TOKS_SPM_MAX_OPS) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder: more than 8 steps"); }
    toks_spm_op *op = &sc->dec[sc->n_dec++];
    memset(op, 0, sizeof(*op));
    op->kind = (k == D_FUSE) ? TOKS_SPM_D_FUSE : TOKS_SPM_D_BYTE_FALLBACK;
    if (k == D_REPLACE) {
        (void)replace_of(d, &ps, &rx, &con);
        if (set_str(&op->a, ps) != 0 || op->a.n == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Replace pattern"); }
        if (set_str(&op->b, con) != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Replace content"); }
        if (!clear_of_byte_tokens(&op->a) || !clear_of_byte_tokens(&op->b)) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Replace touching byte-token chars");
        }
        op->kind = TOKS_SPM_D_REPLACE;
    } else if (k == D_STRIP) {
        const jv *st = toks_jv_get(d, "start"), *sp = toks_jv_get(d, "stop");
        if (set_str(&op->a, toks_jv_get(d, "content")) != 0 || !one_char(&op->a) || !toks_juint(st, INT64_MAX) ||
            !toks_juint(sp, INT64_MAX)) {
            return toks_fail(err, TOKS_E_FORMAT, "decoder Strip fields");
        }
        if (sp->num != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Strip stop > 0"); }
        if (st->num > 0xFFFF) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Strip start > 65535"); }
        op->start = (uint32_t)st->num;
        op->kind = TOKS_SPM_D_STRIP;
    } else if (k == D_METASPACE) {
        int64_t r = read_meta(d, &op->a, &op->scheme, &split, err);
        if (r != 0) { return r; }
        if (!clear_of_byte_tokens(&op->a)) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder Metaspace replacement is a byte-token char"); }
        op->kind = TOKS_SPM_D_METASPACE;
    }
    return 0;
}

/* the decoder runs streaming (spm_c.c): at most one per-token step (Replace, Metaspace), then ByteFallback, Fuse,
 * Strip, each at most once and in this order; Strip only after Fuse */
static int64_t spm_dec_order(const toks_spm_config *sc, toks_err *err)
{
    static const char *const WHY = "decoder order (sentencepiece-style bpe: one Replace or Metaspace, ByteFallback, Fuse, Strip)";
    uint32_t stage = 0, per_token = 0;
    for (uint32_t i = 0; i < sc->n_dec; i++) {              /* bound: TOKS_SPM_MAX_OPS */
        uint32_t k = sc->dec[i].kind;
        uint32_t want = (k == TOKS_SPM_D_BYTE_FALLBACK) ? 1u : (k == TOKS_SPM_D_FUSE) ? 2u : (k == TOKS_SPM_D_STRIP) ? 3u : 0u;
        if (want == 0u ? (stage != 0u || per_token != 0u) : (want == 3u ? stage != 2u : stage >= want)) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, WHY);
        }
        if (want == 0u) { per_token = 1; } else { stage = want; }
    }
    return 0;
}

/* a BPE model with no ByteLevel component anywhere is sentencepiece-style bpe */
static int is_spm(const jv *root, const jv *model)
{
    const jv *type = toks_jv_get(model, "type");
    if (type != NULL && !toks_jstr(type, "BPE")) { return 0; }
    return !has_type(toks_jv_get(root, "normalizer"), "normalizers", "ByteLevel") &&
           !has_type(toks_jv_get(root, "pre_tokenizer"), "pretokenizers", "ByteLevel") &&
           !has_type(toks_jv_get(root, "decoder"), "decoders", "ByteLevel") &&
           !has_type(toks_jv_get(root, "post_processor"), "processors", "ByteLevel");
}

/* rationale: docs/notes/c-core.md §config.c.8 */
static int64_t parse_spm(const jv *root, const jv *model, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    toks_spm_config *sc = (toks_spm_config *)toks_ar_alloc(ar, sizeof(toks_spm_config), 8u);
    if (sc == NULL) { return toks_fail(err, TOKS_E_NOMEM, "spm config"); }
    memset(sc, 0, sizeof(*sc));
    steps s;
    int64_t r = read_steps(root, C_NORM, BIT(N_PREPEND) | BIT(N_REPLACE), &s, err);
    for (uint32_t i = 0; r == 0 && i < s.n; i++) { r = spm_norm_step(s.o[i], s.k[i], &sc->text, err); }   /* bound: 16 */
    if (r == 0) { r = read_steps(root, C_PRE, BIT(P_METASPACE) | BIT(P_SPLIT), &s, err); }
    for (uint32_t i = 0; r == 0 && i < s.n; i++) { r = spm_pretok_step(s.o[i], s.k[i], &sc->text, err); }  /* bound: 16 */
    if (r == 0) {
        r = read_steps(root, C_DEC, BIT(D_REPLACE) | BIT(D_BYTEFALLBACK) | BIT(D_FUSE) | BIT(D_STRIP) | BIT(D_METASPACE), &s, err);
    }
    sc->has_decoder = s.present;
    for (uint32_t i = 0; r == 0 && i < s.n; i++) { r = spm_dec_step(s.o[i], s.k[i], sc, err); }            /* bound: 16 */
    if (r == 0) { r = spm_dec_order(sc, err); }
    if (r == 0) { r = read_bpe(model, &sc->sflags, err); }
    toks_sidx vx;
    const uint32_t *lro[3];
    if (r == 0) { r = read_vocab(model, ar, 1, &vx, &sc->n_ids, &sc->n_strings, err); }
    if (r == 0) { r = read_merges(model, ar, &vx, lro, &sc->n_merges, err); }
    if (r != 0) { return r; }
    sc->vocab = vx.s;
    sc->vocab_len = vx.len;
    sc->vslot = vx.slot;
    sc->vmask = vx.mask;
    sc->m_left = lro[0];
    sc->m_right = lro[1];
    sc->m_out = lro[2];
    sc->unk_id = TOKS_SPM_NONE;
    const jv *unk = toks_jv_get(model, "unk_token");
    if (!toks_jnull(unk)) {
        if (unk->type != JV_STR) { return toks_fail(err, TOKS_E_FORMAT, "model unk_token"); }
        sc->unk_id = spm_vocab_find(sc, unk->s, unk->s_len);
        if (sc->unk_id == TOKS_SPM_NONE) { sc->sflags |= TOKS_SPM_UNK_ERROR; }
    }
    cfg->algo = TOKS_ALGO_BPE_SPM;
    cfg->spm = sc;
    cfg->vocab = sc->vocab;
    cfg->vocab_len = sc->vocab_len;
    cfg->n_vocab = sc->n_ids;
    cfg->n_strings = sc->n_strings;
    cfg->n_merges = sc->n_merges;
    cfg->ignore_merges = (uint8_t)((sc->sflags & TOKS_SPM_IGNORE_MERGES) != 0u);
    r = read_added(root, ar, cfg, NULL, err);
    if (r != 0) { return r; }
    /* rationale: docs/notes/c-core.md §config.c.9 */
    uint32_t nn = sc->text.n_norm, sh = nn == 0u ? 0u : nn == 1u ? meta_op(&sc->text.norm[0]) :
                  nn == 2u ? meta_op(&sc->text.norm[0]) * 4u + meta_op(&sc->text.norm[1]) : 15u;
    for (uint32_t i = 0; i < cfg->n_added && nn != 0u; i++) {  /* bound: n_added */
        toks_cfg_added *a = &cfg->added[i];
        int bad = (sh != 1u && sh != 6u && sh != 9u) || sc->text.metaspace != 0u || a->lstrip != 0u || a->rstrip != 0u ||
                  a->single_word != 0u;
        for (uint32_t j = 0; j < a->len; j++) { bad |= a->content[j] == 0x20u || toks_meta_at(a->content, a->len, j); }   /* bound: len */
        if (a->normalized != 0u && bad) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "added token normalized=true under a normalizer (phase 1 on the folded text: m2)");
        }
        a->pfx = (uint8_t)(a->normalized != 0u && nn == 2u);
    }
    return read_post_processor(root, ar, cfg, err);
}

/* ---- wordpiece (docs/algorithms/wordpiece.md §3, §5, §8, §11) ---------------------------------------------- */

/* a model without "type": hf's untagged enum (models/mod.rs: BPE, WordPiece, WordLevel, Unigram, the first that
 * deserializes): no merges, the four WordPiece fields and a map vocab make a WordPiece (bert-base-uncased) */
static int legacy_wordpiece(const jv *model)
{
    const jv *v = toks_jv_get(model, "vocab");
    return toks_jv_get(model, "type") == NULL && toks_jv_get(model, "merges") == NULL &&
           toks_jv_get(model, "unk_token") != NULL && toks_jv_get(model, "continuing_subword_prefix") != NULL &&
           toks_jv_get(model, "max_input_chars_per_word") != NULL && v != NULL && v->type == JV_OBJ;
}

/* every added token's form (what phase 1 matches): the content through the BertNormalizer when normalized (hf's
 * normalized_cache), else the content; wp_win (wordpiece.md §12.6); a form twice in phase 1 is hf's load error */
static int64_t wp_forms(toks_config *cfg, toks_arena *ar, toks_err *err)
{
    cfg->o.wp_win = 1;
    for (uint32_t i = 0; i < cfg->n_added; i++) {           /* bound: n_added */
        toks_cfg_added *a = &cfg->added[i];
        a->form = a->content;
        a->form_len = a->len;
        if (a->normalized && cfg->wp_flags != 0u) {
            uint64_t room = 4u * (uint64_t)a->len + 16u;    /* > the 3x bound (wordpiece.md §12.5) */
            uint8_t *f = (uint8_t *)toks_ar_alloc(ar, room, 8u);
            if (f == NULL) { return toks_fail(err, TOKS_E_NOMEM, "added token forms"); }
            int64_t n = toks_norm(cfg->wp_flags, a->content, a->len, f, room);
            if (n < 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "added token form above its bound"); }   /* unreachable */
            if (n == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "added token whose normalized form is empty"); }
            if (n > TOKS_MAX_ADDED_BYTES) { return toks_fail(err, TOKS_E_UNSUPPORTED, "added token normalized form above 255 bytes"); }
            a->form = f;
            a->form_len = (uint32_t)n;
        }
        for (uint32_t k = 0; a->normalized && k < a->form_len; k++) {  /* bound: form_len <= 255 */
            uint8_t b = a->form[k];
            if (b == 0x20u || b == 0x09u || b == 0x0Au || b == 0x0Du) { cfg->o.wp_win = 0; }
        }
        for (uint32_t j = 0; a->normalized && j < i; j++) {          /* bound: n_added (<= 65535, at load) */
            const toks_cfg_added *b = &cfg->added[j];
            if (b->normalized && b->form_len == a->form_len && memcmp(b->form, a->form, a->form_len) == 0) {
                return toks_fail(err, TOKS_E_FORMAT, "added tokens: a normalized form twice (hf refuses the file)");
            }
        }
    }
    return 0;
}

static int64_t parse_wordpiece(const jv *root, const jv *model, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    const jv *unk = toks_jv_get(model, "unk_token"), *pre = toks_jv_get(model, "continuing_subword_prefix");
    const jv *maxc = toks_jv_get(model, "max_input_chars_per_word");
    if (unk == NULL || unk->type != JV_STR || pre == NULL || pre->type != JV_STR || !toks_juint(maxc, UINT64_MAX >> 1)) {
        return toks_fail(err, TOKS_E_FORMAT, "WordPiece model fields");
    }
    if ((uint64_t)maxc->num > TOKS_WP_MAX_CHARS) { return toks_fail(err, TOKS_E_UNSUPPORTED, "WordPiece max_input_chars_per_word above 1024"); }
    cfg->algo = TOKS_ALGO_WORDPIECE;
    cfg->wp_max_chars = (uint32_t)maxc->num;
    cfg->wp_unk = unk->s;
    cfg->wp_unk_len = unk->s_len;
    cfg->wp_prefix = pre->s;
    cfg->wp_prefix_len = pre->s_len;
    steps s;                                                    /* BertNormalizer or none */
    int64_t r = read_steps(root, C_NORM, BIT(N_BERT), &s, err);
    if (r == 0 && s.n > 1u) { r = toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer BertNormalizer twice"); }
    if (r == 0 && s.n == 1u) {                                  /* strip_accents null follows lowercase */
        const jv *ct = toks_jv_get(s.o[0], "clean_text"), *hc = toks_jv_get(s.o[0], "handle_chinese_chars");
        const jv *sa = toks_jv_get(s.o[0], "strip_accents"), *lc = toks_jv_get(s.o[0], "lowercase");
        if (!toks_jbool(ct) || !toks_jbool(hc) || !toks_jbool(lc) || (sa != NULL && sa->type != JV_BOOL && sa->type != JV_NULL)) {
            return toks_fail(err, TOKS_E_FORMAT, "BertNormalizer fields");
        }
        cfg->wp_flags = (ct->num ? TOKS_WPF_CLEAN : 0u) | (hc->num ? TOKS_WPF_CHINESE : 0u) | (lc->num ? TOKS_WPF_LOWER : 0u) |
                        ((toks_jnull(sa) ? lc->num != 0 : sa->num != 0) ? TOKS_WPF_STRIP : 0u);
    }
    if (r == 0) { r = read_steps(root, C_PRE, BIT(P_BERT), &s, err); }
    if (r == 0 && s.n != 1u) { r = toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer (wordpiece: BertPreTokenizer)"); }
    if (r == 0) { r = read_steps(root, C_DEC, BIT(D_WORDPIECE), &s, err); }   /* none: hf joins with " " (§8) */
    if (r == 0 && (s.n > 1u || (s.n == 0u && s.present))) { r = toks_fail(err, TOKS_E_UNSUPPORTED, "decoder (wordpiece: WordPiece or none)"); }
    if (r == 0 && s.n == 1u) {
        const jv *dp = toks_jv_get(s.o[0], "prefix"), *cu = toks_jv_get(s.o[0], "cleanup");
        if (dp == NULL || dp->type != JV_STR || !toks_jbool(cu)) { return toks_fail(err, TOKS_E_FORMAT, "decoder WordPiece fields"); }
        if (dp->s_len > TOKS_WP_DEC_PREFIX) { return toks_fail(err, TOKS_E_UNSUPPORTED, "decoder WordPiece prefix above 16 bytes"); }
        cfg->o.dec_wordpiece = 1;
        cfg->o.dec_cleanup = (uint8_t)(cu->num != 0);
        memcpy(cfg->o.dec_prefix, dp->s, dp->s_len);
        cfg->o.dec_prefix_len = dp->s_len;
    }
    if (r == 0) { r = read_trunc_pad(root, cfg, err); }
    toks_sidx vx;
    if (r == 0) { r = read_vocab(model, ar, 0, &vx, &cfg->n_vocab, &cfg->n_strings, err); }
    if (r != 0) { return r; }
    cfg->vocab = vx.s;
    cfg->vocab_len = vx.len;
    int64_t u = toks_sidx_find(&vx, cfg->wp_unk, cfg->wp_unk_len, NULL, 0u);
    if (u < 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "WordPiece unk_token not in vocab"); }
    cfg->wp_unk_id = (uint32_t)u;
    r = read_added(root, ar, cfg, &vx, err);
    if (r == 0) { r = wp_forms(cfg, ar, err); }
    if (r == 0) { r = read_post_processor(root, ar, cfg, err); }
    return (r != 0) ? r : trunc_check(cfg, err);
}

/* ---- unigram (docs/algorithms/unigram.md §1-§2, §4, §5, §7, §8) ---------------------------------------------- */

/* hf's untagged model enum: a typeless model whose vocab is an array is a Unigram (the others need an object) */
static int uni_model(const jv *model)
{
    const jv *t = toks_jv_get(model, "type"), *v = toks_jv_get(model, "vocab");
    return (t != NULL) ? toks_jstr(t, "Unigram") : (v != NULL && v->type == JV_ARR);
}

enum { ST_REP = 1, ST_NF = 2, ST_SA = 3, ST_LOW = 4, ST_STRIP = 5, ST_PC = 6, ST_COLLAPSE = 7, ST_PREFIX = 8, ST_REPLACE = 9 };

/* one step of [Replace(String)]* [NF*] [StripAccents] [Lowercase] (*form: the engine's steps, norm.h) [Strip]
 * [Precompiled] [Replace ' {2,}'] [Replace '(?<!\n)^' -> ▁] [Replace ' ' -> ▁], in that order */
static int64_t uni_norm_one(const jv *n, uint32_t k, toks_uni_src *src, toks_arena *ar, uint32_t *stage, toks_err *err)
{
    uint8_t *form = &src->cfg.form;
    static const uint8_t NF[4] = { TOKS_NS_NFC, TOKS_NS_CANON, TOKS_NS_NFKC, TOKS_NS_NFKD };   /* N_NFC .. N_NFKD */
    uint32_t st = ST_REPLACE;
    const jv *str = NULL, *rx = NULL, *con = NULL;
    if (k <= N_NFKD || k == N_STRIPACC || k == N_LOWER) {
        *form = (uint8_t)(*form | (k <= N_NFKD ? NF[k] : k == N_LOWER ? TOKS_NS_LOWER : TOKS_NS_STRIP_M));
        st = k <= N_NFKD ? ST_NF : k == N_LOWER ? ST_LOW : ST_SA;
    } else if (k == N_STRIP) {
        int l, r;
        if (jflag(n, "strip_left", 1, &l) != 0 || jflag(n, "strip_right", 1, &r) != 0) {   /* serde defaults: true */
            return toks_fail(err, TOKS_E_FORMAT, "normalizer Strip fields");
        }
        src->cfg.strip_left = (uint8_t)l;
        src->cfg.strip_right = (uint8_t)r;
        st = ST_STRIP;
    } else if (k == N_PRECOMPILED) {
        const jv *b = toks_jv_get(n, "precompiled_charsmap");
        if (b == NULL || b->type != JV_STR) { return toks_fail(err, TOKS_E_FORMAT, "normalizer Precompiled charsmap"); }
        uint64_t cap = (uint64_t)b->s_len / 4u * 3u + 3u;
        uint8_t *blob = (uint8_t *)toks_ar_alloc(ar, cap + 1u, 8u);
        if (blob == NULL) { return toks_fail(err, TOKS_E_NOMEM, "normalizer Precompiled charsmap"); }
        int64_t m = toks_b64_decode(b->s, b->s_len, blob, cap);
        if (m <= 0) { return toks_fail(err, TOKS_E_FORMAT, m < 0 ? "normalizer Precompiled charsmap base64" : "normalizer Precompiled charsmap empty"); }
        src->charsmap = blob;
        src->charsmap_len = (uint64_t)m;
        src->cfg.has_charsmap = 1u;
        st = ST_PC;
    } else if (!replace_of(n, &str, &rx, &con) || con == NULL || con->type != JV_STR) {
        return toks_fail(err, TOKS_E_FORMAT, "normalizer Replace fields");
    } else if (toks_jstr(rx, " {2,}") && toks_jstr(con, " ")) {
        src->cfg.collapse = 1u;
        st = ST_COLLAPSE;
    } else if (toks_jstr(rx, "(?<!\\n)^") && toks_jstr(con, META_UTF8)) {
        src->cfg.meta_prefix = 1u;
        st = ST_PREFIX;
    } else if ((toks_jstr(rx, " ") || toks_jstr(str, " ")) && toks_jstr(con, META_UTF8)) {
        src->cfg.meta_replace = 1u;
    } else if (str != NULL && str->type == JV_STR && str->s_len - 1u < 4u && con->s_len <= str->s_len && src->cfg.rep_n < 4u) {
        uint8_t i = src->cfg.rep_n++;                               /* albert's ``, '' -> ": no longer than the pattern */
        memcpy(src->cfg.rep_p[i], str->s, str->s_len);
        memcpy(src->cfg.rep_c[i], con->s, con->s_len);
        src->cfg.rep_pl[i] = (uint8_t)str->s_len;
        src->cfg.rep_cl[i] = (uint8_t)con->s_len;
        st = ST_REP;
    } else {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace (Unigram: ' {2,}' -> ' ', '(?<!\\n)^' -> '▁', ' ' -> '▁', String -> no longer)");
    }
    if (st < *stage || (st == *stage && st != ST_REP)) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer order (Unigram: Replace(String), NF*, StripAccents, Lowercase, Strip, Precompiled, Replace)");
    }
    *stage = st;
    return 0;
}

/* Metaspace with the replacement U+2581: *always = the scheme is always (never = 0; first is refused) */
static int64_t uni_meta(const jv *m, uint8_t *always, uint8_t *split, toks_err *err)
{
    toks_spm_str rep;
    uint32_t scheme, sp;
    int64_t r = read_meta(m, &rep, &scheme, &sp, err);
    if (r != 0) { return r; }
    if (rep.n != 3u || memcmp(rep.b, META_UTF8, 3) != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "Metaspace replacement (toks: U+2581)"); }
    if (scheme == TOKS_SPM_PS_FIRST) { return toks_fail(err, TOKS_E_UNSUPPORTED, "Metaspace prepend_scheme first (Unigram)"); }
    *always = (uint8_t)(scheme == TOKS_SPM_PS_ALWAYS);
    *split = (uint8_t)sp;
    return 0;
}

/* a Replace decoder of pat (String or Regex) by content */
static int uni_replace(const jv *d, const char *pat, const char *content)
{
    const jv *str, *rx, *con;
    return replace_of(d, &str, &rx, &con) && (toks_jstr(rx, pat) || toks_jstr(str, pat)) && toks_jstr(con, content);
}

/* the model ([piece, score] pairs, unk_id, byte_fallback), normalizer, pre-tokenizer, decoder into a toks_uni_src;
 * truncation (max_length > 0) as for every algorithm; padding only as BatchLongest (one sequence: nothing) */
static int64_t parse_unigram(const jv *root, const jv *model, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    toks_uni_src *src = (toks_uni_src *)toks_ar_alloc(ar, sizeof(toks_uni_src), 8u);
    const jv *vocab = toks_jv_get(model, "vocab");
    if (src == NULL) { return toks_fail(err, TOKS_E_NOMEM, "unigram config"); }
    memset(src, 0, sizeof *src);
    if (vocab == NULL || vocab->type != JV_ARR) { return toks_fail(err, TOKS_E_FORMAT, "model.vocab missing"); }
    uint64_t n = 0u;
    for (const jv *e = vocab->child; e != NULL; e = e->next) { n++; }   /* bound: elements */
    if (n == 0u) { return toks_fail(err, TOKS_E_FORMAT, "model.vocab empty"); }
    if (n >= (uint64_t)TOKS_MAX_IDS) { return toks_fail(err, TOKS_E_LIMIT, "model.vocab >= TOKS_MAX_IDS"); }
    const uint8_t **ps = (const uint8_t **)toks_ar_alloc(ar, n * sizeof(uint8_t *), 8u);
    uint32_t *pl = (uint32_t *)toks_ar_alloc(ar, n * 4u, 8u);
    const uint8_t **ss = (const uint8_t **)toks_ar_alloc(ar, n * sizeof(uint8_t *), 8u);
    uint32_t *sl = (uint32_t *)toks_ar_alloc(ar, n * 4u, 8u);
    toks_sidx ux;
    if (ps == NULL || pl == NULL || ss == NULL || sl == NULL || toks_sidx_init(&ux, ar, ps, pl, n) != 0) {
        return toks_fail(err, TOKS_E_NOMEM, "model.vocab arrays");
    }
    uint32_t i = 0u;
    for (const jv *e = vocab->child; e != NULL; e = e->next, i++) {    /* bound: n */
        const jv *pc = (e->type == JV_ARR) ? e->child : NULL, *sc = (pc != NULL) ? pc->next : NULL;
        if (pc == NULL || sc == NULL || sc->next != NULL || pc->type != JV_STR || sc->type != JV_NUM) {
            return toks_fail(err, TOKS_E_FORMAT, "model.vocab entry is not [piece, score]");
        }
        if (pc->s_len > TOKS_MAX_TOKEN_BYTES) { return toks_fail(err, TOKS_E_LIMIT, "model.vocab piece > TOKS_MAX_TOKEN_BYTES"); }
        ps[i] = (pc->s != NULL) ? pc->s : (const uint8_t *)"";
        pl[i] = pc->s_len;
        ss[i] = sc->s;
        sl[i] = sc->s_len;
    }
    const jv *unk = toks_jv_get(model, "unk_id"), *bf = toks_jv_get(model, "byte_fallback");
    if (!toks_jnull(unk) && !toks_juint(unk, INT64_MAX)) { return toks_fail(err, TOKS_E_FORMAT, "model.unk_id"); }
    if (bf != NULL && bf->type != JV_BOOL) { return toks_fail(err, TOKS_E_FORMAT, "model.byte_fallback"); }
    src->n = (uint32_t)n;
    src->piece = ps;
    src->piece_len = pl;
    src->score_txt = ss;
    src->score_len = sl;
    src->unk_id = toks_jnull(unk) ? -1 : unk->num;
    src->byte_fallback = (uint32_t)toks_jtrue(bf);
    steps s;                                                    /* normalizer: uni_norm_one's order */
    uint32_t stage = 0u;
    int64_t r = read_steps(root, C_NORM, BIT(N_STRIP) | BIT(N_PRECOMPILED) | BIT(N_REPLACE) | BIT(N_NFC) | BIT(N_NFD) |
                           BIT(N_NFKC) | BIT(N_NFKD) | BIT(N_STRIPACC) | BIT(N_LOWER), &s, err);
    for (uint32_t k = 0; r == 0 && k < s.n; k++) { r = uni_norm_one(s.o[k], s.k[k], src, ar, &stage, err); }  /* bound: 16 */
    cfg->nfc = src->cfg.form;                                   /* the scratch's factor (api.c scr_x) */
    if (r == 0 && (cfg->nfc & TOKS_NS_COMPOSE) != 0u && (cfg->nfc & (TOKS_NS_STRIP_M | TOKS_NS_LOWER)) != 0u) {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer NFC / NFKC then StripAccents or Lowercase (Unigram)");
    }
    if (r == 0) { r = read_steps(root, C_PRE, BIT(P_WSSPLIT) | BIT(P_METASPACE), &s, err); }   /* [WhitespaceSplit] [Metaspace] */
    if (r == 0 && (s.n > 2u || (s.n == 2u && (s.k[0] != P_WSSPLIT || s.k[1] != P_METASPACE)))) {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, "pre_tokenizer Sequence (Unigram: [WhitespaceSplit, Metaspace])");
    }
    if (r == 0 && s.n != 0u) {
        src->cfg.ws_split = (uint8_t)(s.k[0] == P_WSSPLIT);
        src->cfg.metaspace = (uint8_t)(s.k[s.n - 1u] == P_METASPACE);
        if (src->cfg.metaspace) { r = uni_meta(s.o[s.n - 1u], &src->cfg.meta_prepend, &src->cfg.meta_split, err); }
    }
    if (r == 0 && src->cfg.meta_prefix != 0u && src->cfg.meta_replace == 0u && src->cfg.metaspace == 0u) {
        /* the prefix is fed to the model as the virtual space symbol, which stands for U+2581 only under the remap
         * that ' ' -> U+2581 or Metaspace turns on (unigram.c); alone it would encode as a plain space: refused */
        r = toks_fail(err, TOKS_E_UNSUPPORTED, "normalizer Replace '(?<!\\n)^' -> '\u2581' without ' ' -> '\u2581' or Metaspace (Unigram)");
    }
    /* the decoder chains of the census (unigram.md §8): none; Metaspace; Replace > ByteFallback > Fuse; ByteFallback >
     * Replace > Fuse > Replace('(?<!\n)^ ' -> '') */
    if (r == 0) { r = read_steps(root, C_DEC, BIT(D_METASPACE) | BIT(D_REPLACE) | BIT(D_BYTEFALLBACK) | BIT(D_FUSE), &s, err); }
    uint8_t dsplit = 1u;
    if (r != 0 || !s.present) {
        src->cfg.dec = TOKS_UNI_DEC_NONE;
    } else if (s.n == 1u && s.k[0] == D_METASPACE) {
        src->cfg.dec = TOKS_UNI_DEC_META;
        r = uni_meta(s.o[0], &src->cfg.dec_prepend, &dsplit, err);
    } else if (s.n == 3u && uni_replace(s.o[0], META_UTF8, " ") && s.k[1] == D_BYTEFALLBACK && s.k[2] == D_FUSE) {
        src->cfg.dec = TOKS_UNI_DEC_RBF;
    } else if (s.n == 4u && s.k[0] == D_BYTEFALLBACK && uni_replace(s.o[1], META_UTF8, " ") && s.k[2] == D_FUSE &&
               uni_replace(s.o[3], "(?<!\\n)^ ", "")) {
        src->cfg.dec = TOKS_UNI_DEC_BFRF;
    } else {
        r = toks_fail(err, TOKS_E_UNSUPPORTED, "decoder (Unigram: Metaspace; Replace > ByteFallback > Fuse; ByteFallback > Replace > Fuse > Replace)");
    }
    if (r == 0) { r = read_trunc_pad(root, cfg, err); }
    if (r != 0) { return r; }
    if (cfg->o.trunc_on && cfg->o.trunc_max == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "truncation max_length"); }
    if (src->cfg.meta_replace && src->cfg.metaspace) { return toks_fail(err, TOKS_E_UNSUPPORTED, "Replace ' ' -> '▁' before Metaspace"); }
    for (uint32_t id = 0; id < (uint32_t)n; id++) {             /* bound: n */
        if (toks_sidx_add(&ux, id) != 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "Unigram vocab repeats a piece"); }
    }
    cfg->vocab = ps;
    cfg->vocab_len = pl;
    cfg->n_vocab = (uint32_t)n;
    cfg->algo = TOKS_ALGO_UNIGRAM;
    cfg->n_strings = (uint32_t)n;                               /* hf: vocab.len() */
    cfg->uni = src;
    r = read_added(root, ar, cfg, &ux, err);
    if (r == 0) { r = read_post_processor(root, ar, cfg, err); }
    return (r != 0) ? r : trunc_check(cfg, err);
}

/* ---- the entry point ------------------------------------------------------------------------------------- */

int64_t toks_config_parse(const uint8_t *data, uint64_t len, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    memset(cfg, 0, sizeof(*cfg));
    err->code = 0;
    err->what = NULL;
    jv *root = NULL;
    int64_t r = toks_json_parse(data, len, ar, &root);
    if (r != 0) { return toks_fail(err, r, "json syntax"); }
    if (root->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "json root is not an object"); }
    const jv *model = toks_jv_get(root, "model");
    if (model == NULL || model->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "model missing"); }
    if (toks_jtype(model, "WordPiece") || legacy_wordpiece(model)) { return parse_wordpiece(root, model, ar, cfg, err); }
    if (uni_model(model)) { return parse_unigram(root, model, ar, cfg, err); }
    r = read_trunc_pad(root, cfg, err);                     /* bpe: truncation and padding as every algorithm */
    if (r != 0) { return r; }
    cfg->algo = TOKS_ALGO_BPE_BYTELEVEL;
    r = is_spm(root, model) ? parse_spm(root, model, ar, cfg, err) : parse_bytelevel(root, model, ar, cfg, err);
    return (r != 0) ? r : trunc_check(cfg, err);
}
