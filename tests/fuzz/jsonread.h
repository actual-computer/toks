/*
 * tests/fuzz/jsonread.h: the harnesses' own json reader (no part of toks): a tolerant recursive reader into a
 * node tree whose strings and numbers keep their raw text (escapes and all), used by the json-structure
 * mutator (jsonmut.h) to edit files. Ported from the first fuzz harnesses (commit 82cdeda).
 */
#ifndef TOKS_FUZZ_JSONREAD_H
#define TOKS_FUZZ_JSONREAD_H

#include "fuzz.h"

typedef struct jm_node {
    uint8_t  type;          /* 'n' 't' 'f' '0' (number) 's' (string) '[' '{' */
    uint8_t  has_key;
    uint32_t v, vl;         /* raw text in the pool: a number, or a string's body */
    uint32_t k, kl;         /* an object member's raw key */
    int32_t  child, next;
} jm_node;

typedef struct jm_doc {
    jm_node *n;
    uint32_t nn, cap;
    fz_bytes pool;
    int32_t  root;
} jm_doc;

typedef struct jm_ref { int32_t node, parent, prev; } jm_ref;

#define JM_MAX_NODES 400000u

FZ_FN void jm_free(jm_doc *d) { free(d->n); fz_bytes_free(&d->pool); memset(d, 0, sizeof *d); }

FZ_FN int32_t jm_new(jm_doc *d, uint8_t type)
{
    if (d->nn >= JM_MAX_NODES) { return -1; }
    if (d->nn == d->cap) {
        d->cap = d->cap != 0u ? 2u * d->cap : 1024u;
        d->n = (jm_node *)realloc(d->n, (size_t)d->cap * sizeof(jm_node));
        FZ_CHECK(d->n != NULL, "realloc");
    }
    jm_node *x = &d->n[d->nn];
    memset(x, 0, sizeof *x);
    x->type = type;
    x->child = x->next = -1;
    return (int32_t)d->nn++;
}

FZ_FN uint32_t jm_raw(jm_doc *d, const void *p, uint64_t l)
{
    uint32_t at = (uint32_t)d->pool.n;
    fz_bytes_put(&d->pool, p, l);
    return at;
}

/* ---- reader (tolerant: raw spans, depth <= 512) -------------------------------------------------------- */

FZ_FN void jm_ws(const uint8_t *s, uint64_t n, uint64_t *i)
{
    while (*i < n && (s[*i] == ' ' || s[*i] == '\t' || s[*i] == '\n' || s[*i] == '\r')) { (*i)++; }
}

/* a string body at s[*i] (after the quote): its raw span, *i after the closing quote */
FZ_FN int jm_str(const uint8_t *s, uint64_t n, uint64_t *i, uint32_t *v, uint32_t *vl)
{
    uint64_t a = *i;
    while (*i < n && s[*i] != '"') { *i += s[*i] == '\\' ? 2u : 1u; }
    if (*i >= n) { return 0; }
    *v = (uint32_t)a;
    *vl = (uint32_t)(*i - a);
    (*i)++;
    return 1;
}

FZ_FN int32_t jm_value(jm_doc *d, const uint8_t *s, uint64_t n, uint64_t *i, int depth)
{
    jm_ws(s, n, i);
    if (*i >= n || depth > 512) { return -1; }
    uint8_t c = s[*i];
    int32_t x;
    if (c == '{' || c == '[') {
        x = jm_new(d, c);
        if (x < 0) { return -1; }
        (*i)++;
        int32_t last = -1;
        jm_ws(s, n, i);
        if (*i < n && s[*i] == (c == '{' ? '}' : ']')) { (*i)++; return x; }
        for (;;) {
            uint32_t k = 0, kl = 0;
            if (c == '{') {
                jm_ws(s, n, i);
                if (*i >= n || s[*i] != '"') { return -1; }
                (*i)++;
                if (!jm_str(s, n, i, &k, &kl)) { return -1; }
                jm_ws(s, n, i);
                if (*i >= n || s[*i] != ':') { return -1; }
                (*i)++;
            }
            int32_t ch = jm_value(d, s, n, i, depth + 1);
            if (ch < 0) { return -1; }
            if (c == '{') { d->n[ch].has_key = 1u; d->n[ch].k = k; d->n[ch].kl = kl; }
            if (last < 0) { d->n[x].child = ch; } else { d->n[last].next = ch; }
            last = ch;
            jm_ws(s, n, i);
            if (*i >= n) { return -1; }
            if (s[*i] == ',') { (*i)++; continue; }
            if (s[*i] == (c == '{' ? '}' : ']')) { (*i)++; return x; }
            return -1;
        }
    }
    if (c == '"') {
        x = jm_new(d, 's');
        if (x < 0) { return -1; }
        (*i)++;
        uint32_t v = 0, vl = 0;
        if (!jm_str(s, n, i, &v, &vl)) { return -1; }
        d->n[x].v = v;
        d->n[x].vl = vl;
        return x;
    }
    if (n - *i >= 4u && memcmp(s + *i, "true", 4) == 0) { *i += 4u; return jm_new(d, 't'); }
    if (n - *i >= 5u && memcmp(s + *i, "false", 5) == 0) { *i += 5u; return jm_new(d, 'f'); }
    if (n - *i >= 4u && memcmp(s + *i, "null", 4) == 0) { *i += 4u; return jm_new(d, 'n'); }
    uint64_t a = *i;
    while (*i < n && (strchr("+-0123456789.eE", s[*i]) != NULL) && s[*i] != 0) { (*i)++; }
    if (*i == a) { return -1; }
    x = jm_new(d, '0');
    if (x < 0) { return -1; }
    d->n[x].v = (uint32_t)a;
    d->n[x].vl = (uint32_t)(*i - a);
    return x;
}

FZ_FN int jm_parse(jm_doc *d, const uint8_t *s, uint64_t n)
{
    memset(d, 0, sizeof *d);
    if (n >= 0x7FFFFFFFu) { return 0; }
    fz_bytes_put(&d->pool, s, n);
    uint64_t i = 0;
    d->root = jm_value(d, s, n, &i, 0);
    jm_ws(s, n, &i);
    return d->root >= 0 && i == n;
}

/* the member of object x with key (raw compare), last wins; -1 */
FZ_FN int32_t jm_get(const jm_doc *d, int32_t x, const char *key)
{
    if (x < 0 || d->n[x].type != '{') { return -1; }
    uint64_t kl = strlen(key);
    int32_t f = -1;
    for (int32_t c = d->n[x].child; c >= 0; c = d->n[c].next) {
        if (d->n[c].kl == kl && memcmp(d->pool.p + d->n[c].k, key, kl) == 0) { f = c; }
    }
    return f;
}

FZ_FN uint32_t jm_children(const jm_doc *d, int32_t x, int32_t *out, uint32_t cap)
{
    uint32_t k = 0;
    for (int32_t c = d->n[x].child; c >= 0 && k < cap; c = d->n[c].next) { out[k++] = c; }
    return k;
}

#endif /* TOKS_FUZZ_JSONREAD_H */
