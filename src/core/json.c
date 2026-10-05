/* json.c: strict, bounded rfc 8259 reader (docs/notes/c-core.md §json.c.1) */
#include "core.h"

#define TOKS_JSON_DEPTH 64u

/* frame phases */
enum {
    JF_VALUE      = 0,   /* a value is expected: the root, an array element, a member's value */
    JF_KEY        = 1,   /* an object member key (the object is non-empty, so '"' only) */
    JF_MEMBER_END = 2,   /* after a member's value: ',' or '}' */
    JF_ELEM_END   = 3,   /* after an array element: ',' or ']' */
    JF_DONE       = 4    /* the root value is complete (root frame only) */
};

/* rationale: docs/notes/c-core.md §json.c.2 */
typedef struct jframe {
    jv     *obj;
    jv     *mem;
    jv     *last;
    uint8_t phase;
} jframe;

static jv *jv_new(toks_arena *ar, uint8_t type)
{
    jv *v = (jv *)toks_ar_alloc(ar, sizeof(jv), 8u);
    if (v == NULL) { return NULL; }
    memset(v, 0, sizeof(jv));
    v->type = type;
    return v;
}

static const uint8_t *skip_ws(const uint8_t *p, const uint8_t *end)
{
    while (p < end && (*p == 0x20u || *p == 0x09u || *p == 0x0Au || *p == 0x0Du)) { p++; } /* bound: end - p */
    return p;
}

static int hex4(const uint8_t *p, uint32_t *out)
{
    uint32_t v = 0;
    for (uint32_t i = 0; i < 4u; i++) {                 /* bound: 4 */
        uint8_t c = p[i];
        uint32_t d;
        if (c >= '0' && c <= '9') { d = (uint32_t)(c - '0'); }
        else if (c >= 'a' && c <= 'f') { d = (uint32_t)(c - 'a') + 10u; }
        else if (c >= 'A' && c <= 'F') { d = (uint32_t)(c - 'A') + 10u; }
        else { return 0; }
        v = (v << 4) | d;
    }
    *out = v;
    return 1;
}

/* parse_string result codes */
enum { JS_OK = 0, JS_FMT = 1, JS_NOMEM = 2 };

/* rationale: docs/notes/c-core.md §json.c.3 */
static int parse_string(const uint8_t *p, const uint8_t *end, toks_arena *ar,
                        const uint8_t **s, uint32_t *s_len, const uint8_t **next)
{
    if (p >= end || *p != '"') { return JS_FMT; }
    p++;
    /* pass 1: scan to the closing quote */
    const uint8_t *q = p;
    while (q < end && *q != '"') {                       /* bound: end - q */
        if (*q < 0x20u) { return JS_FMT; }
        if (*q == '\\') {
            q++;
            if (q >= end) { return JS_FMT; }
            if (*q == 'u') {
                if (end - q <= 4) { return JS_FMT; }      /* 4 digits must remain (q + 4u could pass end + 1) */
                q += 4u;
            } else if (*q == '"' || *q == '\\' || *q == '/' || *q == 'b' || *q == 'f' ||
                       *q == 'n' || *q == 'r' || *q == 't') {
                /* one simple escape */
            } else {
                return JS_FMT;
            }
        }
        q++;
    }
    if (q >= end) { return JS_FMT; }                      /* unterminated */
    uint64_t raw = (uint64_t)(q - p);
    uint8_t *dst = (uint8_t *)toks_ar_alloc(ar, raw, 8u);
    if (dst == NULL) { return JS_NOMEM; }
    /* pass 2: decode */
    uint64_t n = 0;
    const uint8_t *r = p;
    while (r < q) {                                      /* bound: raw (r advances >= 1) */
        uint8_t c = *r;
        if (c == '\\') {
            r++;
            uint8_t e = *r;
            if (e == 'u') {
                uint32_t cp;
                if (q - r < 5 || !hex4(r + 1u, &cp)) { return JS_FMT; }
                r += 5u;
                if (cp >= 0xD800u && cp <= 0xDBFFu) {
                    /* must be followed by \uDC00-\uDFFF */
                    if (q - r < 6 || r[0] != '\\' || r[1] != 'u') { return JS_FMT; }
                    uint32_t lo;
                    if (!hex4(r + 2u, &lo)) { return JS_FMT; }
                    if (lo < 0xDC00u || lo > 0xDFFFu) { return JS_FMT; }
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    r += 6u;
                } else if (cp >= 0xDC00u && cp <= 0xDFFFu) {
                    return JS_FMT;    /* lone low surrogate */
                }
                n += toks_utf8_put(dst + n, cp);                 /* <= the 6 or 12 escape bytes it replaces */
            } else if (e == 'n') { dst[n++] = '\n'; r++; }
            else if (e == 't') { dst[n++] = '\t'; r++; }
            else if (e == 'r') { dst[n++] = '\r'; r++; }
            else if (e == 'b') { dst[n++] = 0x08; r++; }
            else if (e == 'f') { dst[n++] = 0x0C; r++; }
            else { dst[n++] = e; r++; }   /* \" \\ \/ */
        } else {
            dst[n++] = c;
            r++;
        }
    }
    if (!toks_utf8_valid(dst, n)) { return JS_FMT; }
    *s = dst;
    *s_len = (uint32_t)n;
    *next = q + 1u;
    return JS_OK;
}

/* parses a number at p; returns new p or NULL. */
static const uint8_t *parse_num(const uint8_t *p, const uint8_t *end, jv *v)
{
    const uint8_t *q = p;
    int neg = 0;
    if (q < end && *q == '-') { neg = 1; q++; }
    if (q >= end) { return NULL; }
    if (*q == '0') {
        q++;
    } else if (*q >= '1' && *q <= '9') {
        while (q < end && *q >= '0' && *q <= '9') { q++; }   /* bound: digits */
    } else {
        return NULL;
    }
    uint64_t mag = 0;
    int overflow = 0;
    for (const uint8_t *d = (neg ? p + 1 : p); d < q; d++) {  /* bound: digits */
        uint64_t dg = (uint64_t)(*d - '0');
        if (mag > (0x7FFFFFFFFFFFFFFFull - dg) / 10u) { overflow = 1; }   /* past 2^63 - 1 */
        else { mag = mag * 10u + dg; }
    }
    int is_float = 0;
    if (q < end && *q == '.') {
        is_float = 1;
        q++;
        if (q >= end || *q < '0' || *q > '9') { return NULL; }
        while (q < end && *q >= '0' && *q <= '9') { q++; }
    }
    if (q < end && (*q == 'e' || *q == 'E')) {
        is_float = 1;
        q++;
        if (q < end && (*q == '+' || *q == '-')) { q++; }
        if (q >= end || *q < '0' || *q > '9') { return NULL; }
        while (q < end && *q >= '0' && *q <= '9') { q++; }
    }
    if (overflow) { is_float = 1; mag = 0; }
    v->num_float = (uint8_t)(is_float != 0);
    v->num = neg ? -(int64_t)mag : (int64_t)mag;
    v->s = p;                                               /* the lexeme: config.c reads 0.0 exactly */
    v->s_len = (uint32_t)(q - p);                           /* < len <= TOKS_MAX_SOURCE_BYTES < 2^32 */
    return q;
}

/* links v into f: an object member's value goes to f->mem->child; otherwise v joins the child
 * chain of f's container (or becomes the document root when f is the root frame). */
static void jf_link(jframe *f, jv **root, jv *v)
{
    if (f->mem != NULL) {
        f->mem->child = v;
        f->mem = NULL;
        return;
    }
    if (f->obj == NULL) {
        *root = v;
        return;
    }
    if (f->last == NULL) { f->obj->child = v; } else { f->last->next = v; }
    f->last = v;
}

/* the phase of f once its newest value is complete (f just took a value that is not still
 * open: a scalar, or an empty container that never pushed a frame). */
static void jf_after_value(jframe *f)
{
    if (f->obj == NULL) { f->phase = JF_DONE; }
    else if (f->obj->type == JV_ARR) { f->phase = JF_ELEM_END; }
    else { f->phase = JF_MEMBER_END; }
}

int64_t toks_json_parse(const uint8_t *data, uint64_t len, toks_arena *ar, jv **out)
{
    *out = NULL;
    const uint8_t *p = data, *end = data + len;
    jframe st[TOKS_JSON_DEPTH];
    jv *root = NULL;

    /* st[0]: the document root frame -- exactly one value, never closed */
    st[0].obj = NULL; st[0].mem = NULL; st[0].last = NULL; st[0].phase = JF_VALUE;
    uint32_t depth = 1;      /* st[0..depth) live */

    p = skip_ws(p, end);
    if (p >= end) { return TOKS_E_FORMAT; }

    for (;;) {                                          /* bound: input bytes (p advances) */
        jframe *f = &st[depth - 1u];

        if (f->phase == JF_VALUE) {
            /* ---- a value is expected at p ---------------------------------------------- */
            jv *v = NULL;
            if (*p == '{' || *p == '[') {
                uint8_t type = (*p == '{') ? JV_OBJ : JV_ARR;
                v = jv_new(ar, type);
                if (v == NULL) { return TOKS_E_NOMEM; }
                if (depth >= TOKS_JSON_DEPTH) { return TOKS_E_FORMAT; }
                jf_link(f, &root, v);
                const uint8_t *q = skip_ws(p + 1u, end);
                if (q < end && *q == ((*p == '{') ? '}' : ']')) {
                    /* empty container: complete here, no frame pushed */
                    p = q + 1u;
                    jf_after_value(f);
                    continue;
                }
                st[depth].obj = v; st[depth].mem = NULL; st[depth].last = NULL;
                st[depth].phase = (type == JV_OBJ) ? JF_KEY : JF_VALUE;
                depth++;
                p = q;
                if (p >= end) { return TOKS_E_FORMAT; }
                continue;
            }
            if (*p == '"') {
                v = jv_new(ar, JV_STR);
                if (v == NULL) { return TOKS_E_NOMEM; }
                const uint8_t *next = NULL;
                int r = parse_string(p, end, ar, &v->s, &v->s_len, &next);
                if (r == JS_FMT) { return TOKS_E_FORMAT; }
                if (r == JS_NOMEM) { return TOKS_E_NOMEM; }
                p = next;
            } else if (end - p >= 4 && p[0] == 't' && p[1] == 'r' && p[2] == 'u' && p[3] == 'e') {
                v = jv_new(ar, JV_BOOL);
                if (v == NULL) { return TOKS_E_NOMEM; }
                v->num = 1;
                p += 4u;
            } else if (end - p >= 5 && p[0] == 'f' && p[1] == 'a' && p[2] == 'l' && p[3] == 's' &&
                       p[4] == 'e') {
                v = jv_new(ar, JV_BOOL);
                if (v == NULL) { return TOKS_E_NOMEM; }
                v->num = 0;
                p += 5u;
            } else if (end - p >= 4 && p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l') {
                v = jv_new(ar, JV_NULL);
                if (v == NULL) { return TOKS_E_NOMEM; }
                p += 4u;
            } else {
                v = jv_new(ar, JV_NUM);
                if (v == NULL) { return TOKS_E_NOMEM; }
                p = parse_num(p, end, v);
                if (p == NULL) { return TOKS_E_FORMAT; }
            }
            jf_link(f, &root, v);
            jf_after_value(f);

        } else if (f->phase == JF_KEY) {
            /* ---- an object member key: '"' only (empty objects push no frame) ---------- */
            if (p >= end || *p != '"') { return TOKS_E_FORMAT; }
            jv *mem = jv_new(ar, JV_MEM);
            if (mem == NULL) { return TOKS_E_NOMEM; }
            const uint8_t *next = NULL;
            int r = parse_string(p, end, ar, &mem->s, &mem->s_len, &next);
            if (r == JS_FMT) { return TOKS_E_FORMAT; }
            if (r == JS_NOMEM) { return TOKS_E_NOMEM; }
            if (f->last == NULL) { f->obj->child = mem; } else { f->last->next = mem; }
            f->last = mem;
            f->mem = mem;
            p = skip_ws(next, end);
            if (p >= end || *p != ':') { return TOKS_E_FORMAT; }
            p = skip_ws(p + 1u, end);
            if (p >= end) { return TOKS_E_FORMAT; }
            f->phase = JF_VALUE;

        } else if (f->phase == JF_MEMBER_END || f->phase == JF_ELEM_END) {
            /* ---- ',' continues, the matching bracket closes --------------------------- */
            uint8_t close = (f->obj->type == JV_OBJ) ? (uint8_t)'}' : (uint8_t)']';
            p = skip_ws(p, end);   /* ws is legal before ',' and before the close bracket */
            if (p >= end) { return TOKS_E_FORMAT; }
            if (*p == ',') {
                f->phase = (f->phase == JF_MEMBER_END) ? JF_KEY : JF_VALUE;
                p = skip_ws(p + 1u, end);
                if (p >= end) { return TOKS_E_FORMAT; }
            } else if (*p == close) {
                depth--;
                p = skip_ws(p + 1u, end);
                jf_after_value(&st[depth - 1u]);   /* the closed container was a value of it */
            } else {
                return TOKS_E_FORMAT;
            }

        } else {
            /* ---- JF_DONE: the root value is complete ------------------------------------ */
            p = skip_ws(p, end);
            if (p != end) { return TOKS_E_FORMAT; }
            break;
        }
    }

    if (root == NULL) { return TOKS_E_FORMAT; }   /* unreachable for non-empty input */
    *out = root;
    return 0;
}

jv *toks_jv_get(const jv *obj, const char *key)
{
    if (obj == NULL || obj->type != JV_OBJ) { return NULL; }
    uint64_t kl = 0;
    while (key[kl] != 0) { kl++; }                        /* bound: NUL */
    jv *found = NULL;
    for (jv *m = obj->child; m != NULL; m = m->next) {    /* bound: members */
        if (m->type == JV_MEM && (uint64_t)m->s_len == kl &&
            memcmp(m->s, key, (size_t)kl) == 0) {
            found = m->child;    /* last wins (map semantics) */
        }
    }
    return found;
}
