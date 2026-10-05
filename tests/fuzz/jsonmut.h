/*
 * tests/fuzz/jsonmut.h: the structure-preserving mutator of fuzz_load_json.c (SPEC T6, T9: "every reader
 * and the compiler on ... structurally valid hostile tokenizer files").
 *
 * The input is parsed into a tree (a tolerant reader of its own: strings and numbers keep their raw text,
 * escapes and all), 1-4 edits are applied, and the tree is written back as compact json, so most outputs
 * stay valid json and reach config.c and the compiler instead of dying in the json reader. Edits:
 *   generic   any node -> an interesting scalar (ids at 2^21 - 1, 2^32, 2^63 ...), numbers nudged,
 *             strings swapped for schema words / patterns / other strings / edited, subtrees copied over
 *             other nodes, elements and members deleted / duplicated (duplicate keys) / swapped, keys
 *             renamed to schema keys, members added, a node wrapped in up to 80 containers (the reader's
 *             depth limit is 64);
 *   tokenizer model.vocab tokens added (products of two tokens) / re-id'd / dropped, model.merges
 *             appended (either format) / duplicated (last one wins) / swapped / dropped, added_tokens
 *             entries added (contents from the vocabulary, other contents, specials) and flags flipped;
 *   whole     a fresh fz_synth tokenizer; libFuzzer's byte mutations on the written text (1 in 16).
 * Unparseable inputs get a fresh synthetic tokenizer or byte mutations.
 */
#ifndef TOKS_FUZZ_JSONMUT_H
#define TOKS_FUZZ_JSONMUT_H

#include "jsonread.h"
#include "synth.h"

/* ---- writer --------------------------------------------------------------------------------------------- */

FZ_FN void jm_write(const jm_doc *d, int32_t x, fz_bytes *o, int in_obj)
{
    const jm_node *e = &d->n[x];
    if (in_obj) {
        fz_bytes_put(o, "\"", 1u);
        fz_bytes_put(o, d->pool.p + e->k, e->kl);
        fz_bytes_put(o, "\":", 2u);
    }
    switch (e->type) {
    case 'n': fz_bytes_put(o, "null", 4u); break;
    case 't': fz_bytes_put(o, "true", 4u); break;
    case 'f': fz_bytes_put(o, "false", 5u); break;
    case '0': fz_bytes_put(o, d->pool.p + e->v, e->vl); break;
    case 's': fz_bytes_put(o, "\"", 1u); fz_bytes_put(o, d->pool.p + e->v, e->vl); fz_bytes_put(o, "\"", 1u); break;
    default: {
        fz_bytes_put(o, e->type == '{' ? "{" : "[", 1u);
        for (int32_t c = e->child; c >= 0; c = d->n[c].next) {
            jm_write(d, c, o, e->type == '{');
            if (d->n[c].next >= 0) { fz_bytes_put(o, ",", 1u); }
        }
        fz_bytes_put(o, e->type == '{' ? "}" : "]", 1u);
        break;
    }
    }
}

/* ---- tree edits ----------------------------------------------------------------------------------------- */

FZ_FN void jm_collect_rec(const jm_doc *d, int32_t x, int32_t parent, int32_t prev, jm_ref *out, uint32_t *k,
                          uint32_t cap)
{
    if (*k >= cap) { return; }
    out[*k].node = x; out[*k].parent = parent; out[*k].prev = prev;
    (*k)++;
    int32_t p = -1;
    for (int32_t c = d->n[x].child; c >= 0; c = d->n[c].next) { jm_collect_rec(d, c, x, p, out, k, cap); p = c; }
}

/* a deep copy of src's subtree x (src may be d) into d; -1 past JM_MAX_NODES */
FZ_FN int32_t jm_copy(jm_doc *d, const jm_doc *src, int32_t x, int depth)
{
    if (depth > 600) { return -1; }
    const jm_node e = src->n[x];
    int32_t y = jm_new(d, e.type);
    if (y < 0) { return -1; }
    d->n[y].has_key = e.has_key;
    if (src == d) { d->n[y].v = e.v; d->n[y].vl = e.vl; d->n[y].k = e.k; d->n[y].kl = e.kl; }
    else {
        d->n[y].v = jm_raw(d, src->pool.p + e.v, e.vl); d->n[y].vl = e.vl;
        d->n[y].k = jm_raw(d, src->pool.p + e.k, e.kl); d->n[y].kl = e.kl;
    }
    int32_t last = -1;
    for (int32_t c = e.child; c >= 0; c = src->n[c].next) {
        int32_t cc = jm_copy(d, src, c, depth + 1);
        if (cc < 0) { return -1; }
        if (last < 0) { d->n[y].child = cc; } else { d->n[last].next = cc; }
        last = cc;
    }
    return y;
}

FZ_FN void jm_append(jm_doc *d, int32_t parent, int32_t x)
{
    int32_t c = d->n[parent].child;
    if (c < 0) { d->n[parent].child = x; return; }
    while (d->n[c].next >= 0) { c = d->n[c].next; }
    d->n[c].next = x;
}

FZ_FN void jm_set_scalar(jm_doc *d, int32_t x, uint8_t type, const char *raw)
{
    d->n[x].type = type;
    d->n[x].child = -1;
    if (raw != NULL) { d->n[x].v = jm_raw(d, raw, strlen(raw)); d->n[x].vl = (uint32_t)strlen(raw); }
}

static const char *const JM_NUMS[] = {
    "0", "-1", "1", "2", "255", "256", "257", "65535", "65536", "2097150", "2097151", "2097152", "4194303",
    "4194304", "4294967295", "4294967296", "9223372036854775807", "9223372036854775808", "-9223372036854775808",
    "18446744073709551616", "1e3", "1.5", "-0", "0.0", "1E-7", "123456789012345678901234567890", "00", "01",
};
static const char *const JM_STRS[] = {
    "BPE", "WordPiece", "Unigram", "WordLevel", "ByteLevel", "Split", "Sequence", "Metaspace", "Whitespace",
    "Digits", "Punctuation", "BertPreTokenizer", "TemplateProcessing", "RobertaProcessing", "BertProcessing",
    "NFC", "NFKC", "Lowercase", "Strip", "Replace", "Prepend", "Precompiled", "Isolated", "Removed",
    "MergedWithPrevious", "Contiguous", "A", "B", "<|endoftext|>", "<s>", "</s>", "\xc4\xa0", "\xc4\xa0the",
    "\\u0120", "\\ud83d\\ude00", "\\u0000", "\\\\", "\\\"", "a b", "", "1.0", "\\ud800", "\\udc00x", "\\n",
    "\xc4\xa0 \xc4\xa0", " ", "  ", "a  b", "\xff", "x\\u0020y",
};
static const char *const JM_KEYS[] = {
    "type", "vocab", "merges", "model", "added_tokens", "pre_tokenizer", "pretokenizers", "normalizer",
    "post_processor", "processors", "decoder", "decoders", "truncation", "padding", "id", "content", "special",
    "normalized", "lstrip", "rstrip", "single_word", "ignore_merges", "dropout", "unk_token",
    "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "add_prefix_space",
    "use_regex", "trim_offsets", "pattern", "Regex", "String", "behavior", "invert", "single", "pair",
    "special_tokens", "ids", "tokens", "SpecialToken", "Sequence", "type_id", "version",
};
#define JM_N(a) ((uint64_t)(sizeof(a) / sizeof((a)[0])))

/* one edit; 1 when something changed */
FZ_FN int jm_edit(jm_doc *d, uint64_t *s, const jm_doc *other)
{
    uint32_t cap = d->nn + 1u;
    jm_ref *refs = (jm_ref *)fz_alloc((uint64_t)cap * sizeof(jm_ref));
    uint32_t nr = 0;
    jm_collect_rec(d, d->root, -1, -1, refs, &nr, cap);
    int32_t model = jm_get(d, d->root, "model");
    int32_t bulk_v = jm_get(d, model, "vocab"), bulk_m = jm_get(d, model, "merges");
    jm_ref r = refs[fz_below(s, nr)];
    int prefer = fz_below(s, 4u) != 0u;                   /* mostly, an edit of the structure */
    for (uint32_t tries = 0; prefer && tries < 64u && (r.parent == bulk_v || r.parent == bulk_m ||
         (r.parent >= 0 && d->n[r.parent].type == '[' && d->n[r.parent].has_key == 0u && r.parent != d->root)); tries++) {
        r = refs[fz_below(s, nr)];
    }
    int32_t x = r.node;
    int changed = 1;
    switch (fz_below(s, 17u)) {
    case 0:                                               /* an interesting scalar */
        if (fz_below(s, 3u) == 0u) { jm_set_scalar(d, x, (uint8_t)"ntf"[fz_below(s, 3u)], NULL); }
        else { jm_set_scalar(d, x, '0', JM_NUMS[fz_below(s, JM_N(JM_NUMS))]); }
        break;
    case 1: {                                             /* a number nudged */
        uint32_t tries = 0;
        while (d->n[x].type != '0' && tries++ < 32u) { x = refs[fz_below(s, nr)].node; }
        if (d->n[x].type != '0') { changed = 0; break; }
        char buf[64];
        long long v = 0;
        uint32_t vl = d->n[x].vl < 40u ? d->n[x].vl : 40u;
        memcpy(buf, d->pool.p + d->n[x].v, vl);
        buf[vl] = 0;
        v = strtoll(buf, NULL, 10);
        static const long long DEL[] = { 1, -1, 2, -2, 255, 256, 1 << 21, -(1 << 21) };
        uint64_t m = fz_below(s, 4u);
        long long w = m == 0u ? v + DEL[fz_below(s, 8u)] : m == 1u ? v * 2 : m == 2u ? -v : v / 2;
        snprintf(buf, sizeof buf, "%lld", w);
        jm_set_scalar(d, x, '0', buf);
        break;
    }
    case 2: {                                             /* a string replaced or edited */
        uint32_t tries = 0;
        while (d->n[x].type != 's' && tries++ < 32u) { x = refs[fz_below(s, nr)].node; }
        uint64_t m = fz_below(s, 4u);
        if (m == 0u || d->n[x].type != 's') { jm_set_scalar(d, x, 's', JM_STRS[fz_below(s, JM_N(JM_STRS))]); }
        else if (m == 1u) { jm_set_scalar(d, x, 's', FZ_PAT_JSON[fz_below(s, 4u)]); }
        else if (m == 2u) {
            int32_t y = refs[fz_below(s, nr)].node;
            if (d->n[y].type == 's') { d->n[x].v = d->n[y].v; d->n[x].vl = d->n[y].vl; }
            else if (d->n[y].has_key) { d->n[x].v = d->n[y].k; d->n[x].vl = d->n[y].kl; }
        } else {                                          /* one raw-text edit: insert, delete, double */
            fz_bytes t = { NULL, 0, 0 };
            fz_bytes_put(&t, d->pool.p + d->n[x].v, d->n[x].vl);
            uint64_t at = fz_below(s, t.n + 1u), e = fz_below(s, 3u);
            fz_bytes u = { NULL, 0, 0 };
            fz_bytes_put(&u, t.p, at);
            if (e == 0u) { static const char *const INS[] = { "x", "\xc4\xa0", " ", "\\n", "\\u00e9", "\xe4\xb8\xad", "\\\\" }; const char *c = INS[fz_below(s, 7u)]; fz_bytes_put(&u, c, strlen(c)); }
            if (e == 2u) { fz_bytes_put(&u, t.p, t.n); }
            uint64_t skip = (e == 1u && at < t.n && t.p[at] != '\\' && t.p[at] < 0x80u &&
                             (at == 0u || t.p[at - 1u] != '\\')) ? 1u : 0u;
            fz_bytes_put(&u, t.p + at + skip, t.n - at - skip);
            d->n[x].v = jm_raw(d, u.p, u.n);
            d->n[x].vl = (uint32_t)u.n;
            fz_bytes_free(&t);
            fz_bytes_free(&u);
        }
        break;
    }
    case 3: {                                             /* a subtree copied over a node */
        int32_t y = other != NULL && fz_below(s, 2u) == 0u ? jm_copy(d, other, (int32_t)fz_below(s, other->nn), 0)
                                                           : jm_copy(d, d, refs[fz_below(s, nr)].node, 0);
        if (y < 0 || d->nn > 200000u) { changed = 0; break; }
        d->n[x].type = d->n[y].type; d->n[x].v = d->n[y].v; d->n[x].vl = d->n[y].vl; d->n[x].child = d->n[y].child;
        break;
    }
    case 4:                                               /* delete an element / member */
        if (r.parent < 0) { changed = 0; break; }
        if (r.prev < 0) { d->n[r.parent].child = d->n[x].next; } else { d->n[r.prev].next = d->n[x].next; }
        break;
    case 5: {                                             /* duplicate one (a repeated key / merge / entry) */
        if (r.parent < 0) { changed = 0; break; }
        int32_t y = jm_copy(d, d, x, 0);
        if (y < 0) { changed = 0; break; }
        if (fz_below(s, 2u) == 0u) { d->n[y].next = d->n[x].next; d->n[x].next = y; } else { jm_append(d, r.parent, y); }
        break;
    }
    case 6: {                                             /* swap with the next sibling */
        int32_t nx = d->n[x].next;
        if (r.parent < 0 || nx < 0) { changed = 0; break; }
        d->n[x].next = d->n[nx].next;
        d->n[nx].next = x;
        if (r.prev < 0) { d->n[r.parent].child = nx; } else { d->n[r.prev].next = nx; }
        break;
    }
    case 7: {                                             /* wrapped in up to 80 containers */
        uint64_t k = 1u + fz_below(s, fz_below(s, 2u) == 0u ? 3u : 80u);
        for (uint64_t i = 0; i < k; i++) {
            int32_t y = jm_new(d, d->n[x].type);
            if (y < 0) { break; }
            d->n[y].v = d->n[x].v; d->n[y].vl = d->n[x].vl; d->n[y].child = d->n[x].child;
            int obj = fz_below(s, 2u) == 0u;
            if (obj) { d->n[y].has_key = 1u; d->n[y].k = jm_raw(d, "x", 1u); d->n[y].kl = 1u; }
            d->n[x].type = obj ? '{' : '[';
            d->n[x].child = y;
        }
        break;
    }
    case 8:                                               /* rename a key */
        if (!d->n[x].has_key) { changed = 0; break; }
        { const char *k = JM_KEYS[fz_below(s, JM_N(JM_KEYS))]; d->n[x].k = jm_raw(d, k, strlen(k)); d->n[x].kl = (uint32_t)strlen(k); }
        break;
    case 9: {                                             /* add a member to an object */
        uint32_t tries = 0;
        while (d->n[x].type != '{' && tries++ < 64u) { x = refs[fz_below(s, nr)].node; }
        if (d->n[x].type != '{') { changed = 0; break; }
        int32_t y = fz_below(s, 2u) == 0u ? jm_copy(d, d, refs[fz_below(s, nr)].node, 0) : jm_new(d, 't');
        if (y < 0) { changed = 0; break; }
        if (d->n[y].type == 't' && fz_below(s, 2u) == 0u) { jm_set_scalar(d, y, 's', JM_STRS[fz_below(s, JM_N(JM_STRS))]); }
        const char *k = JM_KEYS[fz_below(s, JM_N(JM_KEYS))];
        d->n[y].has_key = 1u; d->n[y].k = jm_raw(d, k, strlen(k)); d->n[y].kl = (uint32_t)strlen(k); d->n[y].next = -1;
        jm_append(d, x, y);
        break;
    }
    case 10: case 11: {                                   /* model.vocab */
        int32_t voc = jm_get(d, model, "vocab");
        if (voc < 0 || d->n[voc].type != '{' || d->n[voc].child < 0) { changed = 0; break; }
        int32_t kids[4096];
        uint32_t nk = jm_children(d, voc, kids, 4096u), total = 0;
        for (int32_t c = d->n[voc].child; c >= 0; c = d->n[c].next) { total++; }
        uint64_t m = fz_below(s, 4u);
        if (m <= 1u) {                                    /* a new token: the product of two */
            int32_t a = kids[fz_below(s, nk)], b = kids[fz_below(s, nk)];
            int32_t y = jm_new(d, '0');
            if (y < 0) { changed = 0; break; }
            fz_bytes t = { NULL, 0, 0 };
            fz_bytes_put(&t, d->pool.p + d->n[a].k, d->n[a].kl);
            fz_bytes_put(&t, d->pool.p + d->n[b].k, d->n[b].kl);
            d->n[y].has_key = 1u; d->n[y].k = jm_raw(d, t.p, t.n); d->n[y].kl = (uint32_t)t.n;
            fz_bytes_free(&t);
            char buf[24];
            snprintf(buf, sizeof buf, "%u", fz_below(s, 4u) != 0u ? total : total + (uint32_t)fz_below(s, 3u));
            d->n[y].v = jm_raw(d, buf, strlen(buf)); d->n[y].vl = (uint32_t)strlen(buf);
            jm_append(d, voc, y);
            int32_t mer = jm_get(d, model, "merges");
            if (m == 0u && mer >= 0 && d->n[mer].type == '[' && d->n[mer].child >= 0) {   /* and its merge */
                int pairs = d->n[d->n[mer].child].type == '[';
                int32_t z = jm_new(d, pairs ? '[' : 's');
                if (z < 0) { break; }
                if (pairs) {
                    int32_t za = jm_new(d, 's'), zb = jm_new(d, 's');
                    if (za < 0 || zb < 0) { break; }
                    d->n[za].v = d->n[a].k; d->n[za].vl = d->n[a].kl;
                    d->n[zb].v = d->n[b].k; d->n[zb].vl = d->n[b].kl;
                    d->n[z].child = za; d->n[za].next = zb;
                } else {
                    fz_bytes t2 = { NULL, 0, 0 };
                    fz_bytes_put(&t2, d->pool.p + d->n[a].k, d->n[a].kl);
                    fz_bytes_put(&t2, " ", 1u);
                    fz_bytes_put(&t2, d->pool.p + d->n[b].k, d->n[b].kl);
                    d->n[z].v = jm_raw(d, t2.p, t2.n); d->n[z].vl = (uint32_t)t2.n;
                    fz_bytes_free(&t2);
                }
                jm_append(d, mer, z);
            }
        } else if (m == 2u) {                             /* re-id one */
            int32_t a = kids[fz_below(s, nk)];
            char buf[24];
            snprintf(buf, sizeof buf, "%u", (uint32_t)fz_below(s, total + 2u));
            jm_set_scalar(d, a, '0', buf);
        } else {                                          /* drop the last one (ids stay dense) or any */
            int32_t a = fz_below(s, 2u) == 0u ? kids[nk - 1u] : kids[fz_below(s, nk)];
            int32_t p = -1;
            for (int32_t c = d->n[voc].child; c >= 0 && c != a; c = d->n[c].next) { p = c; }
            if (p < 0) { d->n[voc].child = d->n[a].next; } else { d->n[p].next = d->n[a].next; }
        }
        break;
    }
    case 12: case 13: {                                   /* model.merges */
        int32_t mer = jm_get(d, model, "merges"), voc = jm_get(d, model, "vocab");
        if (mer < 0 || d->n[mer].type != '[' || voc < 0 || d->n[voc].type != '{' || d->n[voc].child < 0) { changed = 0; break; }
        int32_t kids[4096], ms[4096];
        uint32_t nk = jm_children(d, voc, kids, 4096u), nm = jm_children(d, mer, ms, 4096u);
        uint64_t m = fz_below(s, 4u);
        if (m <= 1u || nm == 0u) {                        /* a new merge of two vocabulary tokens */
            int32_t a = kids[fz_below(s, nk)], b = kids[fz_below(s, nk)];
            int pairs = nm != 0u ? d->n[ms[0]].type == '[' : fz_below(s, 2u) == 0u;
            int32_t y;
            if (pairs) {
                y = jm_new(d, '[');
                int32_t ya = jm_new(d, 's'), yb = jm_new(d, 's');
                if (y < 0 || ya < 0 || yb < 0) { changed = 0; break; }
                d->n[ya].v = d->n[a].k; d->n[ya].vl = d->n[a].kl;
                d->n[yb].v = d->n[b].k; d->n[yb].vl = d->n[b].kl;
                d->n[y].child = ya; d->n[ya].next = yb;
            } else {
                y = jm_new(d, 's');
                if (y < 0) { changed = 0; break; }
                fz_bytes t = { NULL, 0, 0 };
                fz_bytes_put(&t, d->pool.p + d->n[a].k, d->n[a].kl);
                fz_bytes_put(&t, " ", 1u);
                fz_bytes_put(&t, d->pool.p + d->n[b].k, d->n[b].kl);
                d->n[y].v = jm_raw(d, t.p, t.n); d->n[y].vl = (uint32_t)t.n;
                fz_bytes_free(&t);
            }
            if (fz_below(s, 2u) == 0u || d->n[mer].child < 0) { jm_append(d, mer, y); }
            else { d->n[y].next = d->n[mer].child; d->n[mer].child = y; }
        } else if (m == 2u) {                             /* an earlier merge repeated at the end */
            int32_t y = jm_copy(d, d, ms[fz_below(s, nm)], 0);
            if (y < 0) { changed = 0; break; }
            jm_append(d, mer, y);
        } else {                                          /* two merges swap ranks */
            int32_t a = ms[fz_below(s, nm)], b = ms[fz_below(s, nm)];
            jm_node ta = d->n[a];
            d->n[a].type = d->n[b].type; d->n[a].v = d->n[b].v; d->n[a].vl = d->n[b].vl; d->n[a].child = d->n[b].child;
            d->n[b].type = ta.type; d->n[b].v = ta.v; d->n[b].vl = ta.vl; d->n[b].child = ta.child;
        }
        break;
    }
    case 14: case 15: {                                   /* added_tokens */
        int32_t at = jm_get(d, d->root, "added_tokens"), voc = jm_get(d, model, "vocab");
        if (at < 0 || d->n[at].type != '[') { changed = 0; break; }
        int32_t es[1024];
        uint32_t ne = jm_children(d, at, es, 1024u);
        uint64_t m = fz_below(s, 3u);
        if (m == 0u && ne != 0u) {                        /* flip a flag */
            static const char *const FL[] = { "special", "normalized", "lstrip", "rstrip", "single_word" };
            int32_t f = jm_get(d, es[fz_below(s, ne)], FL[fz_below(s, 5u)]);
            if (f < 0) { changed = 0; break; }
            jm_set_scalar(d, f, d->n[f].type == 't' ? 'f' : 't', NULL);
            break;
        }
        int32_t y;
        if (ne != 0u) { y = jm_copy(d, d, es[fz_below(s, ne)], 0); }
        else {
            static const char E[] = "{\"id\":0,\"content\":\"<|x|>\",\"single_word\":false,\"lstrip\":false,"
                                    "\"rstrip\":false,\"normalized\":false,\"special\":true}";
            jm_doc t;
            if (!jm_parse(&t, (const uint8_t *)E, sizeof E - 1u)) { jm_free(&t); changed = 0; break; }
            y = jm_copy(d, &t, t.root, 0);
            jm_free(&t);
        }
        if (y < 0) { changed = 0; break; }
        int32_t c = jm_get(d, y, "content");
        if (c >= 0) {                                     /* a vocabulary string, another content, a cut */
            uint64_t k = fz_below(s, 3u);
            if (k == 0u && voc >= 0 && d->n[voc].child >= 0) {
                int32_t kids[4096];
                uint32_t nk = jm_children(d, voc, kids, 4096u);
                int32_t a = kids[fz_below(s, nk)];
                d->n[c].type = 's'; d->n[c].v = d->n[a].k; d->n[c].vl = d->n[a].kl;
            } else if (k == 1u && d->n[c].type == 's' && d->n[c].vl > 1u) {   /* cut at a char boundary */
                uint32_t nl = d->n[c].vl - 1u - (uint32_t)fz_below(s, d->n[c].vl - 1u);
                const uint8_t *cp = d->pool.p + d->n[c].v;
                while (nl > 0u && ((cp[nl] & 0xC0u) == 0x80u || cp[nl - 1u] == '\\')) { nl--; }
                if (nl > 0u) { d->n[c].vl = nl; }
            } else {
                jm_set_scalar(d, c, 's', JM_STRS[fz_below(s, JM_N(JM_STRS))]);
            }
        }
        d->n[y].next = -1;
        jm_append(d, at, y);
        break;
    }
    default: {                                            /* type confusion: an object's values as an array */
        if (d->n[x].type == '{') { d->n[x].type = '['; }
        else if (d->n[x].type == '[') {
            d->n[x].type = '{';
            for (int32_t c = d->n[x].child; c >= 0; c = d->n[c].next) {
                if (!d->n[c].has_key) { d->n[c].has_key = 1u; d->n[c].k = jm_raw(d, "type", 4u); d->n[c].kl = 4u; }
            }
        } else if (d->n[x].type == 's' && fz_below(s, 8u) == 0u) { d->n[x].type = '0'; }
        else { changed = 0; }
        break;
    }
    }
    free(refs);
    return changed;
}

/* parse, edit, write back; the libFuzzer custom mutator of fuzz_load_json.c */
FZ_FN size_t jm_mutate(uint8_t *data, size_t size, size_t max, unsigned int seed, const uint8_t *data2, size_t size2)
{
    uint64_t s = (uint64_t)seed * 0x2545F4914F6CDD1Dull + size;
    jm_doc d, d2;
    memset(&d2, 0, sizeof d2);
    int ok = jm_parse(&d, data, size);
    int ok2 = data2 != NULL && jm_parse(&d2, data2, size2);
    uint64_t r = fz_below(&s, 64u);
    fz_bytes out = { NULL, 0, 0 };
    if (!ok || r == 0u) {                                 /* unparseable (or now and then): start afresh */
        if (fz_below(&s, 2u) == 0u) { jm_free(&d); jm_free(&d2); return fz_mutate_bytes(data, size, max); }
        fz_synth(&out, &s);
    } else {
        uint64_t edits = 1u + fz_below(&s, fz_below(&s, 4u) == 0u ? 8u : 3u);
        for (uint64_t i = 0; i < edits; i++) { (void)jm_edit(&d, &s, ok2 ? &d2 : NULL); }
        jm_write(&d, d.root, &out, 0);
    }
    jm_free(&d);
    jm_free(&d2);
    if (out.n > max || out.n == 0u) { fz_bytes_free(&out); return fz_mutate_bytes(data, size, max); }
    memcpy(data, out.p, out.n);
    size_t n = (size_t)out.n;
    fz_bytes_free(&out);
    if (fz_below(&s, 16u) == 0u) { n = fz_mutate_bytes(data, n, max); }  /* byte noise on top */
    return n;
}

#endif /* TOKS_FUZZ_JSONMUT_H */
