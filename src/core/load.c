/* load.c: toks_load / toks_load_mem_copy / toks_unload and tier (docs/notes/c-core.md §load.c.1) */
#include "kernels.h"
#include "compile.h"
#include "cpu.h"
#include "spm.h"
#include "tiktoken.h"
#include "wp.h"

static void ctx_free(toks_ctx *c)
{
    if (c == NULL) { return; }
    toks_dec_free(c);
    toks_plat_arena_free(c->mem_tables, c->mem_tables_len);
    toks_plat_arena_free(c->mem_bpe, c->mem_bpe_len);
    toks_plat_arena_free(c->mem_spm, c->mem_spm_len);
    toks_plat_arena_free(c->mem_wp, c->mem_wp_len);
    toks_plat_arena_free(c->mem_uni, c->mem_uni_len);
    toks_plat_arena_free(c->mem_voc, c->mem_voc_len);
    if (c->mem_gen != NULL) { toks_plat_free(c->mem_gen, c->mem_gen_len); }
    toks_plat_free(c, sizeof *c);
}

void toks_unload(toks_ctx *ctx) { ctx_free(ctx); }

static int64_t fail(const toks_load_opts *o, int64_t code, const char *what)
{
    if (o != NULL && o->diag != NULL) {
        o->diag->code = code;
        uint64_t i = 0u;
        if (what != NULL) {
            while (what[i] != 0 && i < sizeof o->diag->what - 1u) { o->diag->what[i] = what[i]; i++; }  /* bound: 247 */
        }
        o->diag->what[i] = 0;
    }
    return code;
}

static int str_eq(const char *a, const char *b)
{
    uint64_t i = 0u;
    while (a[i] != 0 && a[i] == b[i] && i < 16u) { i++; }      /* bound: 16 (b is a short literal) */
    return a[i] == b[i];
}

/* rationale: docs/notes/c-core.md §load.c.2 */
static int64_t pick_tier(uint32_t want, uint64_t f, const char **why)
{
    if (want == TOKS_TIER_AUTO) {
        char e[16];
        int64_t el = toks_plat_getenv("TOKS_TIER", e, sizeof e);
        if (el == -2 || (el > 0 && !str_eq(e, "auto"))) {
            if (el > 0 && str_eq(e, "scalar")) { want = TOKS_TIER_SCALAR; }
            else if (el > 0 && str_eq(e, "neon")) { want = TOKS_TIER_NEON; }
            else if (el > 0 && str_eq(e, "avx2")) { want = TOKS_TIER_AVX2; }
            else if (el > 0 && str_eq(e, "avx512")) { want = TOKS_TIER_AVX512; }
            else { *why = "TOKS_TIER: not one of auto, scalar, neon, avx2, avx512"; return TOKS_E_TIER; }
        }
    }
    int neon = TOKS_HAVE_NEON && TOKS_CPU_HAS(f, TOKS_FEAT_NEON_TIER);
    int avx2 = TOKS_HAVE_AVX2 && TOKS_CPU_HAS(f, TOKS_FEAT_AVX2_TIER);
    int avx512 = TOKS_HAVE_AVX512 && TOKS_CPU_HAS(f, TOKS_FEAT_AVX512_TIER);
    switch (want) {
    case TOKS_TIER_AUTO:
        return avx512 ? TOKS_TIER_AVX512 : avx2 ? TOKS_TIER_AVX2 : neon ? TOKS_TIER_NEON : TOKS_TIER_SCALAR;
    case TOKS_TIER_SCALAR:
        return TOKS_TIER_SCALAR;
    case TOKS_TIER_NEON:
        if (neon) { return TOKS_TIER_NEON; }
        *why = "tier neon: not on this cpu or not in this build";
        return TOKS_E_TIER;
    case TOKS_TIER_AVX2:
        if (avx2) { return TOKS_TIER_AVX2; }
        *why = "tier avx2: not on this cpu or not in this build";
        return TOKS_E_TIER;
    case TOKS_TIER_AVX512:
        if (avx512) { return TOKS_TIER_AVX512; }
        *why = "tier avx512: not on this cpu or not in this build";
        return TOKS_E_TIER;
    default:
        *why = "tier: unknown value";
        return TOKS_E_TIER;
    }
}

/* a zeroed context named name and a parse arena of plen bytes (both toks_plat_alloc'd; NULL when either failed) */
static toks_ctx *ctx_new(const char *name, uint64_t plen, uint8_t **parse)
{
    *parse = (uint8_t *)toks_plat_alloc(plen);
    toks_ctx *c = (toks_ctx *)toks_plat_alloc(sizeof *c);
    if (c != NULL) { memset(c, 0, sizeof *c); }
    for (uint32_t i = 0u; c != NULL && name[i] != 0 && i < 63u; i++) { c->name[i] = name[i]; }   /* bound: 63 */
    return c;
}

/* the context from a parsed configuration: compile, bpe tables, identity, tier (*why names a failure). */
static int64_t build_cfg(toks_ctx *c, const toks_config *cfg, toks_arena *par, const toks_load_opts *o,
                         const char **why)
{
    *why = "compile";
    int64_t r = toks_compile(cfg, c, par);
    if (r != 0) { return r; }
    const char *bad = toks_tmpl_invalid(&c->t);             /* SPEC §8.3: split parameters */
    if (bad != NULL) { *why = bad; return TOKS_E_UNSUPPORTED; }

    if (cfg->algo == TOKS_ALGO_WORDPIECE) {                 /* the wordpiece tables: wp.c */
        *why = "wordpiece tables";
        toks_err werr = { 0, NULL };
        r = toks_wp_ctx_build(cfg, par, &c->mem_wp, &c->mem_wp_len, &c->wp, &werr);
        if (r != 0) { if (werr.what != NULL) { *why = werr.what; } return r; }
        c->wp_mat_cap = TOKS_WP_MAT_CAP;                    /* >= toks_wp_mat_min (core.h) */
        c->scr_extra = TOKS_WP_SCR_EXTRA;
    } else if (cfg->algo == TOKS_ALGO_UNIGRAM) {            /* the Unigram model's tables (uni_api.c) */
        r = toks_uni_load(c, cfg, why);
        if (r != 0) { return r; }
    } else if (cfg->algo == TOKS_ALGO_BPE_SPM) {            /* the spm lane's tables (spm_build.c) */
        toks_err e2 = { 0, NULL };
        r = toks_spm_build(&c->t, cfg->spm, &c->mem_spm, &c->mem_spm_len, &c->spm, &e2);
        if (r != 0) { *why = (e2.what != NULL) ? e2.what : "spm tables"; return r; }
        c->dc = (toks_dchain){ c->spm->dec, c->spm->n_dec, c->spm->has_decoder, 0u, 1u, c->spm->holes };
    } else {
        *why = "bpe tables";
        c->mem_bpe_len = toks_bpe_tables_bytes(cfg);
        if (c->mem_bpe_len != 0u) {
            c->mem_bpe = toks_plat_arena(c->mem_bpe_len);
            if (c->mem_bpe == NULL) { c->mem_bpe_len = 0u; return TOKS_E_NOMEM; }
        }
        toks_arena bar = { c->mem_bpe, c->mem_bpe_len, 0u };
        r = toks_bpe_build(&c->t, &bar, cfg);
        if (r != 0) { return r; }
        toks_tab_seal(c->mem_bpe, c->mem_bpe_len);
        if (cfg->gen != NULL) {                             /* the generic pre-tokenizer's program (gen.c) */
            c->mem_gen = (uint8_t *)toks_plat_alloc(cfg->gen_bytes);
            if (c->mem_gen == NULL) { return TOKS_E_NOMEM; }
            c->mem_gen_len = cfg->gen_bytes;
            memcpy(c->mem_gen, cfg->gen, (size_t)cfg->gen_bytes);
            c->gen = (const struct toks_gen *)(const void *)c->mem_gen;
            c->scr_extra = toks_gen_scr(c->gen);
        }
    }

    *why = "decode tables";
    r = toks_dec_build(c);                              /* dec_max, dec_slot, dec_len (stream.c; after spm) */
    if (r != 0) { return r; }
    *why = "vocabulary index";
    r = toks_vocab_build(c, cfg);                       /* toks_token_to_id, toks_id_flags (vocab.c; after dc) */
    if (r != 0) { return r; }
    toks_bound_terms bt;                                /* toks_encode_bound's r and g (compile.c) */
    toks_bound_terms_of(c, &bt);
    c->bound_num = bt.num, c->bound_den = bt.den, c->bound_g = bt.g;

    for (uint64_t i = 0u; i < c->t.add_n; i++) {           /* bound: add_n */
        if ((c->t.add_entries[i].flags & TOKS_AF_SPECIAL) == 0u) { c->n_nonspecial++; }
    }
    c->identity = toks_ctx_identity(c->source_sha256);
    c->cpu_features = toks_cpu_features();
    r = pick_tier(o != NULL ? o->tier : TOKS_TIER_AUTO, c->cpu_features, why);
    if (r < 0) { return r; }
    c->tier = (uint32_t)r;
    if (c->spm != NULL && c->tier != TOKS_TIER_SCALAR) { c->t.flags |= TOKS_TF_ASM_MERGE; }
    return 0;
}

/* the context from the source bytes (*why names a failure). */
static int64_t build(toks_ctx *c, const uint8_t *src, uint64_t len, toks_arena *par, const toks_load_opts *o,
                     const char **why)
{
    toks_config cfg;
    toks_err err = { 0, NULL };
    int64_t r = toks_config_parse(src, len, par, &cfg, &err);
    if (r != 0) { *why = (err.what != NULL) ? err.what : "tokenizer config"; return r; }
    return build_cfg(c, &cfg, par, o, why);
}

/* src (len bytes, toks_plat_alloc'd) is the load's from the first line on: freed on every path. */
static int64_t load_src(toks_ctx **out, uint8_t *src, uint64_t len, const char *name,
                        const toks_load_opts *o)
{
    const char *why = "context";
    int64_t r = TOKS_E_NOMEM;
    uint64_t plen = toks_config_arena_bound(len);
    uint8_t *parse;
    toks_ctx *c = ctx_new(name, plen, &parse);
    if (c != NULL && parse != NULL) {
        toks_sha256(src, len, c->source_sha256);
        toks_arena par = { parse, plen, 0u };
        r = build(c, src, len, &par, o, &why);
    } else if (c == NULL) {
        why = "context";
    } else {
        why = "parse arena";
    }
    toks_plat_free(parse, plen);
    toks_plat_free(src, len);
    if (r != 0) { ctx_free(c); return fail(o, r, why); }
    *out = c;
    return fail(o, 0, NULL);
}

/* dir[0, dl) joined with leaf into buf (a separator added unless dir is empty or ends with one). */
static int join(const char *dir, uint64_t dl, const char *leaf, char *buf, uint64_t cap)
{
    uint64_t ll = 0u;
    while (leaf[ll] != 0) { ll++; }                         /* bound: NUL of a short literal */
    uint64_t sep = (dl > 0u && dir[dl - 1u] != '/' && dir[dl - 1u] != '\\') ? 1u : 0u;
    if (dl + sep + ll + 1u > cap) { return -1; }
    memcpy(buf, dir, (size_t)dl);
    if (sep != 0u) { buf[dl] = '/'; }
    memcpy(buf + dl + sep, leaf, (size_t)ll + 1u);
    return 0;
}

/* a tiktoken model: ranks (rlen bytes, toks_plat_alloc'd, freed on every path) read from a file whose
 * directory part is dir[0, dl); the companions come from the same directory. */
static int64_t load_tiktoken(toks_ctx **out, uint8_t *ranks, uint64_t rlen, const char *dir, uint64_t dl,
                             const char *name, int q, const toks_load_opts *o)
{
    char p[1024];
    uint8_t *conf = NULL, *wrap = NULL;
    uint64_t cl = 0u, wl = 0u;
    int is_dir = 0;
    int64_t r = TOKS_E_OPEN;
    const char *why = q ? "qwen.tiktoken needs tokenizer_config.json and tokenization_qwen.py beside it"
                        : "tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it";
    if (join(dir, dl, "tokenizer_config.json", p, sizeof p) == 0 && toks_plat_read_file(p, &conf, &cl, &is_dir) == 0 &&
        join(dir, dl, q ? "tokenization_qwen.py" : "tokenization_kimi.py", p, sizeof p) == 0 &&
        toks_plat_read_file(p, &wrap, &wl, &is_dir) == 0) {
        uint64_t plen = toks_tiktoken_arena_bound(rlen, cl, wl);
        uint8_t *parse;
        toks_ctx *c = ctx_new(name, plen, &parse);
        r = TOKS_E_NOMEM;
        why = "context";
        if (c != NULL && parse != NULL) {
            uint8_t h[96];                                  /* the identity covers all three files */
            toks_sha256(ranks, rlen, h);
            toks_sha256(conf, cl, h + 32);
            toks_sha256(wrap, wl, h + 64);
            toks_sha256(h, sizeof h, c->source_sha256);
            toks_arena par = { parse, plen, 0u };
            toks_config cfg;
            toks_tiktoken_info info;
            toks_err err = { 0, NULL };
            r = toks_tiktoken_parse(ranks, rlen, conf, cl, wrap, wl, &par, &cfg, &info, &err);
            why = (err.what != NULL) ? err.what : "tiktoken model";
            if (r == 0) { r = build_cfg(c, &cfg, &par, o, &why); }
        }
        toks_plat_free(parse, plen);
        if (r != 0) { ctx_free(c); c = NULL; }
        if (c != NULL) { *out = c; }
    }
    toks_plat_free(ranks, rlen);
    if (conf != NULL) { toks_plat_free(conf, cl); }
    if (wrap != NULL) { toks_plat_free(wrap, wl); }
    return fail(o, r, r != 0 ? why : NULL);
}

/* argument checks shared by the entry points. */
static int64_t check_opts(toks_ctx **out, const toks_load_opts *o)
{
    if (out == NULL) { return TOKS_E_ARG; }
    *out = NULL;
    if (o == NULL) { return 0; }
    if (o->size != (uint32_t)sizeof(toks_load_opts)) { return TOKS_E_ARG; }   /* another size: o->diag may lie past
                                                                               the caller's struct, so no diag */
    if (o->rsv != 0u || o->flags != 0u) {
        return fail(o, TOKS_E_ARG, "toks_load_opts: rsv or flags (no load flags are defined)");
    }
    return 0;
}

int64_t toks_load(toks_ctx **out, const char *path, const toks_load_opts *o)
{
    int64_t r = check_opts(out, o);
    if (r != 0) { return r; }
    if (path == NULL) { return fail(o, TOKS_E_ARG, "path"); }

    char buf[1024];
    const char *file = path;
    uint8_t *src = NULL;
    uint64_t len = 0u;
    int is_dir = 0;
    r = toks_plat_read_file(file, &src, &len, &is_dir);
    if (r != 0 && is_dir) {                             /* a model directory */
        r = toks_plat_dir_lookup(path, buf, sizeof buf);
        if (r != 0) {                                   /* no tokenizer.json: a tiktoken model? */
            uint64_t pl = 0u;
            while (path[pl] != 0 && pl < sizeof buf) { pl++; }   /* bound: 1024 */
            if ((join(path, pl, "tiktoken.model", buf, sizeof buf) != 0 || toks_plat_read_file(buf, &src, &len, &is_dir) != 0) &&
                (join(path, pl, "qwen.tiktoken", buf, sizeof buf) != 0 || toks_plat_read_file(buf, &src, &len, &is_dir) != 0)) {
                return fail(o, TOKS_E_OPEN, "model directory without tokenizer.json, tiktoken.model or qwen.tiktoken");
            }
            r = 0;
        }
        file = buf;
        if (src == NULL) { r = toks_plat_read_file(file, &src, &len, &is_dir); }
    }
    if (r != 0) { return fail(o, r, file); }

    /* the name: the file's basename, or its directory's when that is tokenizer.json */
    uint64_t n = 0u, b0 = 0u, b1 = 0u;                  /* [b0, n): the last component */
    while (file[n] != 0 && n < sizeof buf) {           /* bound: 1024 (a longer path keeps the prefix) */
        if (file[n] == '/' || file[n] == '\\') { b1 = b0; b0 = n + 1u; }
        n++;
    }
    char name[64];
    uint64_t s = b0, e = n;
    int q = str_eq(file + b0, "qwen.tiktoken");         /* a Qwen-1 model (tiktoken.c qwen) */
    if ((q || str_eq(file + b0, "tokenizer.json") || str_eq(file + b0, "tiktoken.model")) && b0 > b1) {
        s = b1;
        e = b0 - 1u;
    }
    uint64_t k = 0u;
    while (s + k < e && k < 63u) { name[k] = file[s + k]; k++; }   /* bound: 63 */
    name[k] = 0;
    if (toks_tiktoken_sniff(src, len)) { return load_tiktoken(out, src, len, file, b0, name, q, o); }
    return load_src(out, src, len, name, o);
}

int64_t toks_load_mem_copy(toks_ctx **out, const void *data, uint64_t len, const toks_load_opts *o)
{
    int64_t r = check_opts(out, o);
    if (r != 0) { return r; }
    if (data == NULL && len != 0u) { return fail(o, TOKS_E_ARG, "data"); }
    if (len > TOKS_MAX_SOURCE_BYTES) { return fail(o, TOKS_E_LIMIT, "source above 256 MiB"); }
    if (len == 0u) { return fail(o, TOKS_E_FORMAT, "empty source"); }
    if (toks_tiktoken_sniff((const uint8_t *)data, len)) {
        return fail(o, TOKS_E_UNSUPPORTED, "a tiktoken model is three files: toks_load its directory");
    }
    uint8_t *src = (uint8_t *)toks_plat_alloc(len);
    if (src == NULL) { return fail(o, TOKS_E_NOMEM, "source copy"); }
    memcpy(src, data, (size_t)len);
    return load_src(out, src, len, "", o);
}

