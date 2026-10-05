/* json.h: strict, bounded rfc 8259 reader (no recursion, depth-limited, arena-allocated), and the field
 * helpers every reader shares (config.c, tiktoken.c). */
#ifndef TOKS_JSON_H
#define TOKS_JSON_H

#include <stdint.h>
#include "core.h"

/* json is loaded from a buffer the caller owns for the duration of parsing; all output lives
 * in the arena given to toks_json_parse. */

enum {
    JV_NULL = 0,
    JV_BOOL = 1,
    JV_NUM  = 2,   /* integer in num; num_float set when frac/exp were present; s / s_len: the lexeme */
    JV_STR  = 3,   /* s / s_len: decoded utf-8 bytes */
    JV_ARR  = 4,   /* child = first element; elements chained by next */
    JV_OBJ  = 5,   /* child = first member; member: s = key, child = value, next = next member */
    JV_MEM  = 6    /* internal: an object member */
};

typedef struct jv {
    uint8_t       type;
    uint8_t       num_float;
    uint16_t      rsv;
    uint32_t      s_len;
    const uint8_t *s;
    struct jv     *child;
    struct jv     *next;
    int64_t       num;
} jv;

/* returns 0 (and *out = the root, never NULL for a valid document) or TOKS_E_FORMAT. */
int64_t toks_json_parse(const uint8_t *data, uint64_t len, toks_arena *ar, jv **out);

/* helpers: obj_get returns the LAST member with the key (serde/json map semantics). */
jv *toks_jv_get(const jv *obj, const char *key);

/* v is the string s (a NUL-terminated literal) */
static inline int toks_jstr(const jv *v, const char *s)
{
    uint64_t n = 0;
    while (s[n] != 0) { n++; }                              /* bound: NUL of a static string */
    return v != NULL && v->type == JV_STR && (uint64_t)v->s_len == n && (n == 0u || memcmp(v->s, s, (size_t)n) == 0);
}
static inline int toks_jnull(const jv *v) { return v == NULL || v->type == JV_NULL; }      /* absent or null */
static inline int toks_jbool(const jv *v) { return v != NULL && v->type == JV_BOOL; }
static inline int toks_jtrue(const jv *v) { return toks_jbool(v) && v->num != 0; }
static inline int toks_jtype(const jv *o, const char *t)
{
    return o != NULL && o->type == JV_OBJ && toks_jstr(toks_jv_get(o, "type"), t);
}
/* a json integer in [0, max] (a float lexeme is not one: serde's unsigned ints) */
static inline int toks_juint(const jv *v, uint64_t max)
{
    return v != NULL && v->type == JV_NUM && v->num_float == 0 && v->num >= 0 && (uint64_t)v->num <= max;
}
/* a reader's refusal: err names the reason (a static string) */
static inline int64_t toks_fail(toks_err *err, int64_t code, const char *what)
{
    err->code = code;
    err->what = what;
    return code;
}

#endif /* TOKS_JSON_H */
