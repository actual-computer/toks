/* tiktoken.c: tiktoken-format models -> toks_config (docs/notes/c-core.md §tiktoken.c.1) */
#include "tiktoken.h"
#include "norm.h"

/* ---- the byte-level alphabet (gpt-2 bytes_to_unicode): byte -> code point --------------------------- */

static void alphabet(uint16_t b2u[256])
{
    uint16_t n = 0u;
    for (uint32_t b = 0; b < 256u; b++) {                   /* bound: 256 */
        int self = (b >= 0x21u && b <= 0x7Eu) || (b >= 0xA1u && b <= 0xACu) || b >= 0xAEu;
        b2u[b] = self ? (uint16_t)b : (uint16_t)(0x100u + n++);
    }
}

/* ---- the ranks file ---------------------------------------------------------------------------------- */

/* one line [s, e) (no LF; a trailing CR already cut off): its fields, or a reason. */
typedef struct line {
    uint64_t tok, tok_n;      /* the base64 field */
    uint32_t raw_n;           /* its decoded length */
    uint32_t rank;
} line;

static const char *parse_line(const uint8_t *d, uint64_t s, uint64_t e, line *l)
{
    uint64_t sp = s;
    while (sp < e && d[sp] != ' ') { sp++; }                /* bound: e - s */
    if (sp == e) { return "tiktoken.model: a line without the space between token and rank"; }
    for (uint64_t i = s; i < e; i++) {                      /* bound: e - s */
        if (d[i] == '\r' || d[i] == '\t' || (d[i] == ' ' && i != sp)) {
            return "tiktoken.model: a line that is not '<base64> <rank>' (one space, LF line ends)";
        }
    }
    int64_t rn = toks_b64_len(d + s, sp - s);
    if (rn < 0) { return "tiktoken.model: a token that is not canonical base64"; }
    if (rn > (int64_t)TOKS_MAX_TOKEN_BYTES) { return "tiktoken.model: a token above TOKS_MAX_TOKEN_BYTES"; }
    uint64_t ds = sp + 1u, dn = e - ds;
    if (dn == 0u || dn > 7u || (d[ds] == '0' && dn > 1u)) {
        return "tiktoken.model: a rank that is not a canonical decimal below TOKS_MAX_IDS";
    }
    uint64_t r = 0u;
    for (uint64_t i = ds; i < e; i++) {                     /* bound: 7 */
        if (d[i] < '0' || d[i] > '9') { return "tiktoken.model: a rank that is not a canonical decimal below TOKS_MAX_IDS"; }
        r = r * 10u + (uint64_t)(d[i] - '0');
    }
    if (r >= TOKS_MAX_IDS) { return "tiktoken.model: a rank that is not a canonical decimal below TOKS_MAX_IDS"; }
    l->tok = s;
    l->tok_n = sp - s;
    l->raw_n = (uint32_t)rn;
    l->rank = (uint32_t)r;
    return NULL;
}

/* the line ending at the first LF at or after s (or at len): [s, *e) without a final CR; *next after it. */
static void next_line(const uint8_t *d, uint64_t len, uint64_t s, uint64_t *e, uint64_t *next)
{
    uint64_t i = s;
    while (i < len && d[i] != '\n') { i++; }                /* bound: len - s */
    *next = (i < len) ? i + 1u : len;
    *e = (i > s && d[i - 1u] == '\r') ? i - 1u : i;
}

uint64_t toks_tiktoken_arena_bound(uint64_t ranks_len, uint64_t config_len, uint64_t wrapper_len)
{
    /* rationale: docs/notes/c-core.md §tiktoken.c.2 */
    (void)wrapper_len;
    return 16u * ranks_len + 24u * (uint64_t)TOKS_MAX_IDS + toks_config_arena_bound(config_len) +
           TOKS_TIKTOKEN_RESERVED * (3u * sizeof(toks_cfg_added) + 64u) + 65536u;
}

int64_t toks_tiktoken_ranks(const uint8_t *d, uint64_t len, toks_arena *ar, toks_config *cfg, toks_err *err)
{
    memset(cfg, 0, sizeof *cfg);
    err->code = 0;
    err->what = NULL;
    if (len == 0u) { return toks_fail(err, TOKS_E_FORMAT, "tiktoken.model: empty"); }
    if (len > TOKS_MAX_SOURCE_BYTES) { return toks_fail(err, TOKS_E_LIMIT, "tiktoken.model: above 256 MiB"); }

    /* pass 1: syntax, counts */
    uint64_t n_tok = 0u, raw_total = 0u, max_rank = 0u;
    for (uint64_t s = 0u, e = 0u, nx = 0u; s < len; s = nx) {   /* bound: len (nx > s) */
        next_line(d, len, s, &e, &nx);
        if (e == s) { continue; }
        line l;
        const char *why = parse_line(d, s, e, &l);
        if (why != NULL) { return toks_fail(err, TOKS_E_FORMAT, why); }
        n_tok++;
        raw_total += l.raw_n;
        if (l.rank > max_rank) { max_rank = l.rank; }
    }
    if (n_tok == 0u) { return toks_fail(err, TOKS_E_FORMAT, "tiktoken.model: no tokens"); }
    uint32_t nv = (uint32_t)max_rank + 1u;

    uint8_t  *raw = (uint8_t *)toks_ar_alloc(ar, raw_total + 1u, 8u);
    const uint8_t **rp = (const uint8_t **)toks_ar_alloc(ar, 8u * (uint64_t)nv, 8u);   /* [rank] its bytes */
    uint32_t *rl  = (uint32_t *)toks_ar_alloc(ar, 4u * (uint64_t)nv, 8u);
    toks_sidx x;
    if (raw == NULL || rp == NULL || rl == NULL || toks_sidx_init(&x, ar, rp, rl, n_tok) != 0) {
        return toks_fail(err, TOKS_E_NOMEM, "tiktoken.model arrays");
    }
    memset(rl, 0, 4u * (size_t)nv);
    memset(rp, 0, 8u * (size_t)nv);

    /* pass 2: decode, place by rank, index (a rank or a token met twice is refused) */
    uint64_t ro = 0u;
    for (uint64_t s = 0u, e = 0u, nx = 0u; s < len; s = nx) {   /* bound: len (nx > s) */
        next_line(d, len, s, &e, &nx);
        if (e == s) { continue; }
        line l;
        (void)parse_line(d, s, e, &l);                      /* pass 1 accepted every line */
        if (rl[l.rank] != 0u) { return toks_fail(err, TOKS_E_FORMAT, "tiktoken.model: a rank appears twice"); }
        (void)toks_b64_decode(d + l.tok, l.tok_n, raw + ro, l.raw_n);
        rp[l.rank] = raw + ro;
        rl[l.rank] = l.raw_n;
        if (toks_sidx_add(&x, l.rank) != 0) { return toks_fail(err, TOKS_E_FORMAT, "tiktoken.model: a token appears twice"); }
        ro += l.raw_n;
    }
    for (uint32_t b = 0; b < 256u; b++) {                   /* bound: 256 */
        uint8_t one = (uint8_t)b;
        if (toks_sidx_find(&x, &one, 1u, NULL, 0u) < 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "tiktoken.model lacks a one-byte token"); }
    }

    /* the vocabulary in the byte-level alphabet (holes: empty strings) */
    uint16_t b2u[256];
    alphabet(b2u);
    uint64_t alen = 0u;
    for (uint64_t i = 0; i < raw_total; i++) { alen += (b2u[raw[i]] < 0x80u) ? 1u : 2u; }   /* bound: raw_total */
    uint8_t *astr = (uint8_t *)toks_ar_alloc(ar, alen + 1u, 8u);
    const uint8_t **vs = (const uint8_t **)toks_ar_alloc(ar, 8u * (uint64_t)nv, 8u);
    uint32_t *vl = (uint32_t *)toks_ar_alloc(ar, 4u * (uint64_t)nv, 8u);
    if (astr == NULL || vs == NULL || vl == NULL) { return toks_fail(err, TOKS_E_NOMEM, "tiktoken.model vocabulary"); }
    uint64_t ao = 0u;
    for (uint32_t id = 0; id < nv; id++) {                  /* bound: nv */
        vs[id] = astr + ao;
        for (uint32_t k = 0; k < rl[id]; k++) {             /* bound: the token's length */
            uint32_t cp = b2u[rp[id][k]];
            if (cp < 0x80u) {
                astr[ao++] = (uint8_t)cp;
            } else {
                astr[ao++] = (uint8_t)(0xC0u | (cp >> 6));
                astr[ao++] = (uint8_t)(0x80u | (cp & 0x3Fu));
            }
        }
        vl[id] = (uint32_t)(astr + ao - vs[id]);
    }

    /* the merges: every split of every token into two tokens, by merged id, then split position.
     * pass a counts, pass b fills. */
    uint64_t nm = 0u;
    for (int pass = 0; pass < 2; pass++) {                  /* bound: 2 */
        uint32_t *ml = NULL, *mr = NULL, *mo = NULL;
        if (pass == 1) {
            if (nm >= (1u << TOKS_PRIO_BITS)) { return toks_fail(err, TOKS_E_LIMIT, "tiktoken.model: merges > 2^22"); }
            ml = (uint32_t *)toks_ar_alloc(ar, 4u * nm + 4u, 8u);
            mr = (uint32_t *)toks_ar_alloc(ar, 4u * nm + 4u, 8u);
            mo = (uint32_t *)toks_ar_alloc(ar, 4u * nm + 4u, 8u);
            if (ml == NULL || mr == NULL || mo == NULL) { return toks_fail(err, TOKS_E_NOMEM, "tiktoken.model merges"); }
        }
        uint64_t m = 0u;
        for (uint32_t id = 0; id < nv; id++) {              /* bound: nv */
            const uint8_t *t = rp[id];
            for (uint32_t k = 1u; k < rl[id]; k++) {        /* bound: the token's length */
                int64_t a = toks_sidx_find(&x, t, k, NULL, 0u);
                if (a < 0) { continue; }
                int64_t b = toks_sidx_find(&x, t + k, rl[id] - k, NULL, 0u);
                if (b < 0) { continue; }
                if (pass == 1) {
                    ml[m] = (uint32_t)a;
                    mr[m] = (uint32_t)b;
                    mo[m] = id;
                }
                m++;
            }
        }
        nm = m;
        if (pass == 1) {
            cfg->m_left_id = ml;
            cfg->m_right_id = mr;
            cfg->m_out_id = mo;
        }
    }
    cfg->vocab = vs;
    cfg->vocab_len = vl;
    cfg->n_vocab = nv;
    cfg->n_merges = (uint32_t)nm;
    cfg->n_ids = nv;
    cfg->ignore_merges = 1u;
    cfg->ids_as_rank = 1u;
    cfg->dec_byte_level = 1u;
    cfg->pattern = NULL;
    return 0;
}

/* ---- the kimi wrapper -------------------------------------------------------------------------------- */

/* tokenization_kimi.py must hold each line (blanks around it ignored); [1 .. 8] are the pattern's
 * alternatives, which must also be consecutive lines in this order. Each entry names what is missing. */
static const char *const WRAPPER_LINES[] = {
    "num_reserved_special_tokens = 256",
    "r\"\"\"[\\p{Han}]+\"\"\",",
    "r\"\"\"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?\"\"\",",
    "r\"\"\"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?\"\"\",",
    "r\"\"\"\\p{N}{1,3}\"\"\",",
    "r\"\"\" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\"\"\",",
    "r\"\"\"\\s*[\\r\\n]+\"\"\",",
    "r\"\"\"\\s+(?!\\S)\"\"\",",
    "r\"\"\"\\s+\"\"\",",
    "TIKTOKEN_MAX_ENCODE_CHARS = 400_000",
    "MAX_NO_WHITESPACES_CHARS = 25_000",
    "current_slice_is_space = s[0].isspace() if len(s) > 0 else False",
    "if current_slice_len > max_consecutive_slice_len:",
    "allowed_special=\"all\",",
    "disallowed_special=(),",
};
static const char *const WRAPPER_WHY[] = {
    "tokenization_kimi.py lacks: num_reserved_special_tokens = 256",
    "tokenization_kimi.py lacks the pattern line [\\p{Han}]+",
    "tokenization_kimi.py lacks the pattern's first letter alternative",
    "tokenization_kimi.py lacks the pattern's second letter alternative",
    "tokenization_kimi.py lacks the pattern line \\p{N}{1,3}",
    "tokenization_kimi.py lacks the pattern's punctuation alternative ( ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*)",
    "tokenization_kimi.py lacks the pattern line \\s*[\\r\\n]+",
    "tokenization_kimi.py lacks the pattern line \\s+(?!\\S)",
    "tokenization_kimi.py lacks the pattern line \\s+",
    "tokenization_kimi.py lacks: TIKTOKEN_MAX_ENCODE_CHARS = 400_000",
    "tokenization_kimi.py lacks: MAX_NO_WHITESPACES_CHARS = 25_000",
    "tokenization_kimi.py lacks the run splitter's first line (s[0].isspace())",
    "tokenization_kimi.py lacks the run splitter's cut (current_slice_len > max_consecutive_slice_len)",
    "tokenization_kimi.py lacks: allowed_special=\"all\",",
    "tokenization_kimi.py lacks: disallowed_special=(),",
};
#define N_WRAPPER_LINES ((uint32_t)(sizeof(WRAPPER_LINES) / sizeof(WRAPPER_LINES[0])))

static uint64_t cstr_len(const char *s)
{
    uint64_t n = 0u;
    while (s[n] != 0) { n++; }                              /* bound: NUL of a static string */
    return n;
}

/* 1 when p[0, n), without its leading / trailing blanks, equals s. */
static int line_is(const uint8_t *p, uint64_t n, const char *s)
{
    while (n > 0u && (p[0] == ' ' || p[0] == '\t')) { p++; n--; }               /* bound: n */
    while (n > 0u && (p[n - 1u] == ' ' || p[n - 1u] == '\t' || p[n - 1u] == '\r')) { n--; }   /* bound: n */
    uint64_t k = cstr_len(s);
    return n == k && memcmp(p, s, (size_t)k) == 0;
}

static int line_blank(const uint8_t *p, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) {                      /* bound: n */
        if (p[i] != ' ' && p[i] != '\t' && p[i] != '\r') { return 0; }
    }
    return 1;
}

/* the lines of w[0, n) equal to L[i] (blanks around them ignored): bit i; *run: L[r0 .. r1] one after another
 * (blank lines aside) somewhere */
static uint32_t scan_lines(const uint8_t *w, uint64_t n, const char *const *L, uint32_t nl, uint32_t r0, uint32_t r1,
                           uint32_t *run_ok)
{
    uint32_t seen = 0u, run = 0u;
    *run_ok = 0u;
    for (uint64_t s = 0u, e = 0u, nx = 0u; s < n; s = nx) {     /* bound: n (nx > s) */
        next_line(w, n, s, &e, &nx);
        if (line_blank(w + s, e - s)) { continue; }
        uint32_t hit = nl;
        for (uint32_t i = 0; i < nl; i++) {                 /* bound: nl <= 32 */
            if (line_is(w + s, e - s, L[i])) { hit = i; break; }
        }
        if (hit < nl) { seen |= 1u << hit; }
        run = (hit >= r0 && hit <= r1 && hit == r0 + run) ? run + 1u : (hit == r0 ? 1u : 0u);
        if (run == r1 - r0 + 1u) { *run_ok = 1u; }
    }
    return seen;
}

static const char *check_wrapper(const uint8_t *w, uint64_t n)
{
    if (n == 0u || n > (1u << 20) || !toks_utf8_valid(w, n)) { return "tokenization_kimi.py: empty, above 1 MiB or not utf-8"; }
    uint32_t pattern_ok, seen = scan_lines(w, n, WRAPPER_LINES, N_WRAPPER_LINES, 1u, 8u, &pattern_ok);   /* the pattern */
    for (uint32_t i = 0; i < N_WRAPPER_LINES; i++) {        /* bound: 15 */
        if ((seen & (1u << i)) == 0u) { return WRAPPER_WHY[i]; }
    }
    if (pattern_ok == 0u) { return "tokenization_kimi.py: the pattern's eight alternatives are not consecutive, in order"; }
    return NULL;
}

/* a special-token attribute: a string, or an AddedToken object ({"content": ...}) whose lstrip / rstrip /
 * single_word / normalized are absent or false; NULL when absent or otherwise. */
static const jv *attr_str(const jv *v)
{
    if (v == NULL || v->type == JV_NULL) { return NULL; }
    if (v->type == JV_OBJ) {
        static const char *const FLAGS[] = { "lstrip", "rstrip", "single_word", "normalized" };
        for (uint32_t k = 0; k < 4u; k++) {                 /* bound: 4 */
            const jv *f = toks_jv_get(v, FLAGS[k]);
            if (f != NULL && !(f->type == JV_BOOL && f->num == 0)) { return NULL; }
        }
        v = toks_jv_get(v, "content");
    }
    return (v != NULL && v->type == JV_STR) ? v : NULL;
}

static int32_t special_of(const toks_cfg_added *sp, const jv *v)
{
    if (v == NULL || v->type != JV_STR) { return -1; }
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {     /* bound: 256 */
        if (sp[i].len == v->s_len && memcmp(sp[i].content, v->s, v->s_len) == 0) { return (int32_t)i; }
    }
    return -1;
}

/* rationale: docs/notes/c-core.md §tiktoken.c.3 */
static int names_disjoint(const toks_cfg_added *sp)
{
    for (uint32_t a = 0; a < TOKS_TIKTOKEN_RESERVED; a++) {         /* bound: 256 */
        for (uint32_t p = 0; p < sp[a].len; p++) {                  /* bound: the name's length (<= 255) */
            for (uint32_t b = 0; b < TOKS_TIKTOKEN_RESERVED; b++) { /* bound: 256 */
                if (b == a || sp[b].content[0] != sp[a].content[p]) { continue; }
                uint32_t k = (sp[a].len - p < sp[b].len) ? sp[a].len - p : sp[b].len;
                /* equal over k: b inside a, a suffix of a = a prefix of b, or (p = 0) one prefixes the other */
                if (memcmp(sp[a].content + p, sp[b].content, k) == 0) { return 0; }
            }
        }
    }
    return 1;
}

static const char *const CONFIG_KEYS[] = {
    "added_tokens_decoder", "additional_special_tokens", "bos_token", "eos_token", "unk_token", "pad_token",
    "clean_up_tokenization_spaces", "extra_special_tokens", "model_max_length", "tokenizer_class", "auto_map",
    "chat_template", "backend", "is_local", "local_files_only", "tool_parser_type", "processor_class", "padding_side",
    "model_specific_special_tokens",       /* transformers 5's save_pretrained (docs/models/kimi.md §3.5) */
};

int64_t toks_tiktoken_kimi(const uint8_t *config, uint64_t config_len, const uint8_t *wrapper, uint64_t wrapper_len,
                           toks_arena *ar, toks_config *cfg, toks_tiktoken_info *info, toks_err *err)
{
    memset(info, 0, sizeof *info);
    const char *why = check_wrapper(wrapper, wrapper_len);
    if (why != NULL) { return toks_fail(err, TOKS_E_UNSUPPORTED, why); }

    /* the ranks must be 0 .. n-1: the wrapper numbers its specials from len(ranks) */
    uint32_t n_ranks = 0u;
    for (uint32_t id = 0; id < cfg->n_vocab; id++) { n_ranks += cfg->vocab_len[id] != 0u ? 1u : 0u; }   /* bound: n_vocab */
    if (n_ranks != cfg->n_vocab) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tiktoken.model ranks are not 0..n-1 (the kimi wrapper numbers its specials from the token count)");
    }
    uint32_t first = n_ranks;
    if ((uint64_t)first + TOKS_TIKTOKEN_RESERVED > TOKS_MAX_IDS) { return toks_fail(err, TOKS_E_LIMIT, "kimi specials: ids >= TOKS_MAX_IDS"); }

    jv *root = NULL;
    if (toks_json_parse(config, config_len, ar, &root) != 0 || root->type != JV_OBJ) {
        return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: not a json object");
    }
    for (const jv *m = root->child; m != NULL; m = m->next) {   /* bound: members */
        uint32_t known = 0u;
        for (uint32_t k = 0; k < (uint32_t)(sizeof CONFIG_KEYS / sizeof CONFIG_KEYS[0]); k++) {   /* bound: 19 */
            uint64_t kl = cstr_len(CONFIG_KEYS[k]);
            if (m->s_len == kl && memcmp(m->s, CONFIG_KEYS[k], (size_t)kl) == 0) { known = 1u; }
        }
        if (known == 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: a key the kimi reader does not know"); }
    }
    if (!toks_jstr(toks_jv_get(root, "tokenizer_class"), "TikTokenTokenizer")) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: tokenizer_class is not TikTokenTokenizer");
    }
    const jv *am = toks_jv_get(root, "auto_map");
    const jv *at = (am != NULL && am->type == JV_OBJ) ? toks_jv_get(am, "AutoTokenizer") : NULL;
    if (at == NULL || at->type != JV_ARR || !toks_jstr(at->child, "tokenization_kimi.TikTokenTokenizer")) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: auto_map AutoTokenizer is not tokenization_kimi.TikTokenTokenizer");
    }
    const jv *ex = toks_jv_get(root, "extra_special_tokens"), *ms = toks_jv_get(root, "model_specific_special_tokens"),
             *be = toks_jv_get(root, "backend");
    if ((ex != NULL && ex->type == JV_OBJ && ex->child != NULL) || (ms != NULL && ms->type != JV_NULL &&
        !(ms->type == JV_OBJ && ms->child == NULL)) || (be != NULL && !toks_jstr(be, "custom"))) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: named extra special tokens or a backend other than custom");
    }

    /* the 256 specials: added_tokens_decoder's names, else <|reserved_token_{id}|> */
    toks_cfg_added *sp = (toks_cfg_added *)toks_ar_alloc(ar, TOKS_TIKTOKEN_RESERVED * sizeof(toks_cfg_added), 8u);
    uint8_t *names = (uint8_t *)toks_ar_alloc(ar, TOKS_TIKTOKEN_RESERVED * 32u, 8u);
    uint8_t in_dec[TOKS_TIKTOKEN_RESERVED];
    if (sp == NULL || names == NULL) { return toks_fail(err, TOKS_E_NOMEM, "kimi specials"); }
    memset(sp, 0, TOKS_TIKTOKEN_RESERVED * sizeof(toks_cfg_added));
    memset(in_dec, 0, sizeof in_dec);
    const jv *dec = toks_jv_get(root, "added_tokens_decoder");
    if (dec != NULL && dec->type != JV_OBJ) { return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: added_tokens_decoder"); }
    for (const jv *m = (dec != NULL) ? dec->child : NULL; m != NULL; m = m->next) {   /* bound: members */
        uint64_t id = 0u;
        if (m->s_len == 0u || m->s_len > 7u || (m->s[0] == '0' && m->s_len > 1u)) {
            return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: an added_tokens_decoder key is not an id");
        }
        for (uint32_t k = 0; k < m->s_len; k++) {           /* bound: 7 */
            if (m->s[k] < '0' || m->s[k] > '9') { return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: an added_tokens_decoder key is not an id"); }
            id = id * 10u + (uint64_t)(m->s[k] - '0');
        }
        if (id < first || id >= (uint64_t)first + TOKS_TIKTOKEN_RESERVED) {
            return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: an added token outside the wrapper's 256 special ids");
        }
        uint32_t i = (uint32_t)(id - first);
        if (in_dec[i] != 0u) { return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: added_tokens_decoder repeats an id"); }
        const jv *v = m->child;
        const jv *c = (v != NULL && v->type == JV_OBJ) ? toks_jv_get(v, "content") : NULL;
        const jv *spc = (v != NULL && v->type == JV_OBJ) ? toks_jv_get(v, "special") : NULL;
        if (c == NULL || c->type != JV_STR || c->s_len == 0u || spc == NULL || spc->type != JV_BOOL) {
            return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: an added token without content / special");
        }
        /* the trie path (docs/models/kimi.md 3.3) and the model assume plain tokens: each flag present
         * and false (transformers' AddedToken default for normalized is true) */
        static const char *const FLAGS[] = { "lstrip", "rstrip", "single_word", "normalized" };
        for (uint32_t k = 0; k < 4u; k++) {                 /* bound: 4 */
            const jv *f = toks_jv_get(v, FLAGS[k]);
            if (f == NULL || f->type != JV_BOOL || f->num != 0) {
                return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: an added token whose lstrip / rstrip / single_word / normalized is not false");
            }
        }
        if (c->s_len > TOKS_MAX_ADDED_BYTES) { return toks_fail(err, TOKS_E_LIMIT, "tokenizer_config.json: an added token above TOKS_MAX_ADDED_BYTES"); }
        in_dec[i] = 1u;
        sp[i].content = c->s;
        sp[i].len = c->s_len;
    }
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {     /* bound: 256 */
        sp[i].id = first + i;
        if (in_dec[i] != 0u) {
            for (uint32_t k = 0; k < sp[i].len; k++) {      /* bound: 255: ascii printable (decode == the name's bytes) */
                if (sp[i].content[k] < 0x21u || sp[i].content[k] > 0x7Eu) {
                    return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: a special name outside printable ascii");
                }
            }
            continue;
        }
        static const char pre[] = "<|reserved_token_";
        uint8_t *o = names + 32u * i;
        uint32_t n = 0u;
        for (; pre[n] != 0; n++) { o[n] = (uint8_t)pre[n]; }   /* bound: 17 */
        uint8_t dg[8];
        uint32_t nd = 0u;
        for (uint32_t v = first + i; nd == 0u || v != 0u; v /= 10u) { dg[nd++] = (uint8_t)('0' + v % 10u); }   /* bound: 7 digits */
        while (nd > 0u) { o[n++] = dg[--nd]; }              /* bound: 7 */
        o[n++] = '|';
        o[n++] = '>';
        sp[i].content = o;
        sp[i].len = n;
    }
    if (!names_disjoint(sp)) { return toks_fail(err, TOKS_E_UNSUPPORTED, "kimi specials: two names overlap or repeat"); }

    /* rationale: docs/notes/c-core.md §tiktoken.c.4 */
    uint8_t named[TOKS_TIKTOKEN_RESERVED];
    memset(named, 0, sizeof named);
    static const char *const ATTRS[] = { "bos_token", "eos_token", "unk_token", "pad_token" };
    for (uint32_t k = 0; k < 4u; k++) {                     /* bound: 4 */
        int32_t i = special_of(sp, attr_str(toks_jv_get(root, ATTRS[k])));
        if (i < 0) { return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: bos / eos / unk / pad_token must each name a special token"); }
        named[i] = 1u;
    }
    for (uint32_t k = 0; k < 2u; k++) {                     /* bound: additional_special_tokens, extra (a list) */
        const jv *add = k == 0u ? toks_jv_get(root, "additional_special_tokens") : ex;
        if (add != NULL && add->type != JV_NULL && add->type != JV_ARR && k == 0u) { return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: additional_special_tokens"); }
        for (const jv *v = (add != NULL && add->type == JV_ARR) ? add->child : NULL; v != NULL; v = v->next) {   /* bound: elements */
            int32_t i = special_of(sp, attr_str(v));
            if (i < 0) { return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: an additional special token that is not one of the 256 specials"); }
            named[i] = 1u;
        }
    }
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {     /* bound: 256 */
        sp[i].special = named[i];
        sp[i].normalized = 0u;
        if (in_dec[i] != 0u || named[i] != 0u) {
            info->trie[i >> 3] = (uint8_t)(info->trie[i >> 3] | (1u << (i & 7u)));
            info->n_trie++;
        }
        info->n_named += named[i];
    }
    cfg->added = sp;
    cfg->n_added = TOKS_TIKTOKEN_RESERVED;
    cfg->n_ids = first + TOKS_TIKTOKEN_RESERVED;
    info->n_ranks = n_ranks;
    info->first_special = first;
    info->chunk_chars = 400000u;
    info->run_chars = 25000u;
    return 0;
}

/* ---- the Qwen-1 wrapper (docs/models/qwen1.md): tokenize = tiktoken's encode of NFC(text), every special allowed ---- */

static const char *const QWEN_LINES[] = {
    "class QWenTokenizer(PreTrainedTokenizer):",
    "ENDOFTEXT,", "IMSTART,", "IMEND,",                     /* [1 .. 3]: one after another, the specials' order */
    "PAT_STR = r\"\"\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+\"\"\"",
    "ENDOFTEXT = \"<|endoftext|>\"", "IMSTART = \"<|im_start|>\"", "IMEND = \"<|im_end|>\"",
    "EXTRAS = tuple((f\"<|extra_{i}|>\" for i in range(205)))", "SPECIAL_START_ID = 151643", "+ EXTRAS",
    "start=SPECIAL_START_ID,", "text = unicodedata.normalize(\"NFC\", text)", "allowed_special: Union[Set, str] = \"all\",",
    "disallowed_special: Union[Collection, str] = (),", "text, allowed_special=allowed_special, disallowed_special=disallowed_special",
    "token_ids = [i for i in token_ids if i < self.eod_id]",
};
#define N_QWEN_LINES ((uint32_t)(sizeof(QWEN_LINES) / sizeof(QWEN_LINES[0])))
#define QWEN_FIRST 151643u                                  /* SPECIAL_START_ID = len(ranks) */
#define QWEN_SPECIALS 208u                                  /* ENDOFTEXT, IMSTART, IMEND, 205 EXTRAS */

static int64_t qwen(const uint8_t *config, uint64_t config_len, const uint8_t *wrapper, uint64_t wrapper_len,
                    toks_arena *ar, toks_config *cfg, toks_err *err)
{
    uint32_t run, seen = scan_lines(wrapper, wrapper_len, QWEN_LINES, N_QWEN_LINES, 1u, 3u, &run);
    if (seen + 1u != 1u << N_QWEN_LINES || run == 0u) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenization_qwen.py: not the Qwen-1 wrapper toks reads (a line of docs/models/qwen1.md differs)");
    }
    uint32_t n_ranks = 0u;
    for (uint32_t id = 0; id < cfg->n_vocab; id++) { n_ranks += cfg->vocab_len[id] != 0u ? 1u : 0u; }   /* bound: n_vocab */
    if (n_ranks != QWEN_FIRST || cfg->n_vocab != QWEN_FIRST) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "qwen.tiktoken: not ranks 0 .. 151642 (SPECIAL_START_ID = 151643)");
    }
    jv *root = NULL;
    if (toks_json_parse(config, config_len, ar, &root) != 0 || root->type != JV_OBJ) {
        return toks_fail(err, TOKS_E_FORMAT, "tokenizer_config.json: not a json object");
    }
    uint32_t keys = 0u;
    for (const jv *m = root->child; m != NULL; m = m->next) { keys++; }   /* bound: members */
    const jv *am = toks_jv_get(root, "auto_map"), *at = (am != NULL && am->type == JV_OBJ) ? toks_jv_get(am, "AutoTokenizer") : NULL;
    if (keys != 2u + (toks_jv_get(root, "model_max_length") != NULL) || !toks_jstr(toks_jv_get(root, "tokenizer_class"), "QWenTokenizer") ||
        at == NULL || at->type != JV_ARR || !toks_jstr(at->child, "tokenization_qwen.QWenTokenizer")) {
        return toks_fail(err, TOKS_E_UNSUPPORTED, "tokenizer_config.json: not QWenTokenizer's (tokenizer_class, auto_map, model_max_length only)");
    }
    toks_cfg_added *sp = (toks_cfg_added *)toks_ar_alloc(ar, QWEN_SPECIALS * sizeof(toks_cfg_added), 8u);
    uint8_t *names = (uint8_t *)toks_ar_alloc(ar, QWEN_SPECIALS * 16u, 8u);
    if (sp == NULL || names == NULL) { return toks_fail(err, TOKS_E_NOMEM, "qwen specials"); }
    memset(sp, 0, QWEN_SPECIALS * sizeof(toks_cfg_added));
    static const char *const FIXED[3] = { "<|endoftext|>", "<|im_start|>", "<|im_end|>" };
    for (uint32_t i = 0; i < QWEN_SPECIALS; i++) {          /* bound: 208; the name of each, then its flags */
        uint8_t *o = names + 16u * i;
        uint32_t n = 0u, v = i - 3u;
        const char *s = (i < 3u) ? FIXED[i] : "<|extra_";
        for (; s[n] != 0; n++) { o[n] = (uint8_t)s[n]; }    /* bound: 13 */
        if (i >= 3u) {                                      /* <|extra_{i - 3}|> */
            if (v >= 100u) { o[n++] = (uint8_t)('0' + v / 100u); }
            if (v >= 10u) { o[n++] = (uint8_t)('0' + v / 10u % 10u); }
            o[n++] = (uint8_t)('0' + v % 10u);
            o[n++] = '|';
            o[n++] = '>';
        }
        sp[i] = (toks_cfg_added){ .content = o, .len = n, .id = QWEN_FIRST + i, .special = 1u, .normalized = 1u };
    }
    cfg->added = sp;                                        /* matched in NFC(text) (phase 1), all special */
    cfg->n_added = QWEN_SPECIALS;
    cfg->n_ids = QWEN_FIRST + QWEN_SPECIALS;
    cfg->nfc = TOKS_NS_NFC;
    cfg->pattern = &TOKS_PATTERNS[2];                       /* PAT_STR = the qwen 2 template's string */
    return 0;
}

int64_t toks_tiktoken_parse(const uint8_t *ranks, uint64_t ranks_len, const uint8_t *config, uint64_t config_len,
                            const uint8_t *wrapper, uint64_t wrapper_len, toks_arena *ar, toks_config *cfg,
                            toks_tiktoken_info *info, toks_err *err)
{
    int64_t r = toks_tiktoken_ranks(ranks, ranks_len, ar, cfg, err);
    if (r != 0) { return r; }
    uint32_t run;
    memset(info, 0, sizeof *info);
    if ((scan_lines(wrapper, wrapper_len, QWEN_LINES, 1u, 1u, 0u, &run) & 1u) != 0u) {   /* class QWenTokenizer */
        return qwen(config, config_len, wrapper, wrapper_len, ar, cfg, err);
    }
    r = toks_tiktoken_kimi(config, config_len, wrapper, wrapper_len, ar, cfg, info, err);
    if (r != 0) { return r; }
    /* rationale: docs/notes/c-core.md §tiktoken.c.5 */
    toks_cfg_added *ph = (toks_cfg_added *)toks_ar_alloc(ar, (uint64_t)(info->n_trie + TOKS_TIKTOKEN_RESERVED) *
                                                         sizeof(toks_cfg_added), 8u);
    if (ph == NULL) { return toks_fail(err, TOKS_E_NOMEM, "tiktoken: parse arena"); }
    uint32_t k = 0u;
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {     /* bound: 256 */
        if (((info->trie[i >> 3] >> (i & 7u)) & 1u) != 0u) {
            ph[k] = cfg->added[i];
            ph[k].normalized = 0u;
            k++;
        }
    }
    for (uint32_t i = 0; i < TOKS_TIKTOKEN_RESERVED; i++) {     /* bound: 256 */
        ph[k] = cfg->added[i];
        ph[k].normalized = 1u;
        k++;
    }
    cfg->added = ph;
    cfg->n_added = k;
    cfg->pattern = &TOKS_PATTERN_KIMI;
    cfg->cut_chunk = info->chunk_chars;
    cfg->cut_run = info->run_chars;
    return 0;
}

int toks_tiktoken_sniff(const uint8_t *data, uint64_t len)
{
    return len > 0u && toks_b64v(data[0]) >= 0;
}
