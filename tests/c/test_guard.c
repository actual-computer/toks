/*
 * test_guard.c: the guard geometry's own check (docs/testing.md, "The guard geometry"). Under make test-guard every
 * table a context reads (core.h toks_tab) and every region of a scratch (toks_scr_at) is a mapping of its own,
 * placed by tests/common/guard.c; here each one is probed, for the fixtures of every family (byte-level, the generic
 * engine, sentencepiece-style bpe, unigram; gpt2, gemma4, t5 and a wordpiece model from the tokenizer cache when there):
 *  - nothing is missed: every table pointer a context holds is the first byte of a guard table, so none still points
 *    into its builder's block (which the guard build seals besides);
 *  - run 1: a table's last byte (its declared pad included) reads and the byte after it faults; a region's too;
 *  - run 2: its first byte reads and the byte before it faults;
 *  - the context encodes through the scratch (the regions probed are the ones the library uses), and an unload
 *    unmaps its tables;
 *  - the kernels at the tables' ends, on the tier the build binds (TOKS_TIER as make test): K1 on the added token whose
 *    bytes end add_bytes, whole and a byte short; K5 / K6 on the token whose bytes end tok_bytes, and K5 on a token
 *    whose words bucket is the table's last (byte-level);
 *  - the seal's placement check: an empty table may share its offset with the next in either order, and a table past
 *    its block's end, across it, or over another stops the load.
 * Each probe that must fault runs in a child process. In the shipped build the test checks that toks_tab and
 * toks_scr_at are the shipped placement (block + o, base + off) and says the probes are make test-guard's.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
#include "bpe.h"
#include "spm.h"
#include "unigram.h"
#include "wp.h"
#include "guard.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(TOKS_GUARD)
#  include <signal.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

static const char *path_of(const char *name, char *buf, size_t cap)
{
    const char *root = getenv("TOKS_TOKENIZER_CACHE");
    if (strchr(name, '/') != NULL) { snprintf(buf, cap, "%s", name); }
    else if (root != NULL && root[0] != 0) { snprintf(buf, cap, "%s/%s", root, name); }
    else { snprintf(buf, cap, "%s/.cache/toks/tokenizers/%s", getenv("HOME") ? getenv("HOME") : ".", name); }
    return buf;
}

#if defined(TOKS_GUARD)
/* 1 when a read of *p faults (SIGSEGV or SIGBUS) in a child process; the child catches its fault and exits 77, so no
 * core is written and no crash reporter runs per probe */
static void caught(int sig) { (void)sig; _exit(77); }
static int faults(const uint8_t *p)
{
    fflush(NULL);
    pid_t c = fork();
    if (c == 0) {
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = caught;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
        volatile uint8_t v = *(const volatile uint8_t *)p;
        (void)v;
        _exit(0);
    }
    int st = 0;
    if (c < 0 || waitpid(c, &st, 0) != c) { return 0; }
    return WIFEXITED(st) && WEXITSTATUS(st) == 77;
}

/* the geometry's promise for one table or region of n bytes at p */
static void probe(const char *what, const char *name, const uint8_t *p, uint64_t n)
{
#if TOKS_GUARD == 1
    if (n != 0u) { volatile uint8_t v = p[n - 1u]; (void)v; }
    CHECK(faults(p + n), "%s: %s of %" PRIu64 " bytes: the byte after it does not fault (run 1)", name, what, n);
#else
    if (n != 0u) { volatile uint8_t v = p[0]; (void)v; }
    CHECK(faults(p - 1), "%s: %s of %" PRIu64 " bytes: the byte before it does not fault (run 2)", name, what, n);
#endif
}

/* every table pointer the context holds: the first byte of a guard table */
static uint32_t tables_of(const toks_ctx *c, const char *name)
{
    const toks_tables *t = &c->t;
    const void *p[64];
    const char *w[64];
    uint32_t k = 0;
#define T(x) do { if ((x) != NULL && k < 64u) { p[k] = (x); w[k] = #x; k++; } } while (0)
    T(t->cls_ascii); T(t->cls_stage1); T(t->cls_stage2); T(t->byte2id); T(t->bytepair); T(t->merge_slots);
    T(t->rank2id); T(t->premerge); T(t->words); T(t->vhash); T(t->tok_off); T(t->tok_bytes); T(t->add_shufti);
    T(t->add_index); T(t->add_single); T(t->add_cand); T(t->add_entries); T(t->add_bytes); T(t->apm); T(t->pairf);
    T(c->special_ids); T(c->dec_slot); T(c->dec_len); T(c->voc_slots); T(c->voc_add); T(c->voc_pool);
    T(c->voc_added); T(c->voc_special);
    if (c->spm != NULL) {
        T(c->spm); T(c->spm->stage1); T(c->spm->stage2); T(c->spm->pairs); T(c->spm->holes); T(c->spm->cut_ab8);
    }
    if (c->wp != NULL) {
        T(c->wp); T(c->wp->word); T(c->wp->cont); T(c->wp->keys); T(c->wp->wtab); T(c->wp->wcell); T(c->wp->ccell);
        T(c->wp->wterm); T(c->wp->cterm);
    }
    if (c->uni != NULL) {
        T(c->uni); T(c->uni->cell); T(c->uni->score); T(c->uni->term); T(c->uni->words); T(c->uni->pc.stage1);
        T(c->uni->pc.stage2); T(c->uni->pc.mk_key); T(c->uni->pc.mk_val); T(c->uni->pc.pool);
    }
#undef T
    for (uint32_t i = 0; i < k; i++) {
        const uint8_t *s = NULL;
        uint64_t n = 0;
        CHECK(guard_tab_of(p[i], &s, &n) && s == (const uint8_t *)p[i], "%s: %s is not a guard table's first byte",
              name, w[i]);
    }
    return k;
}
#endif

/* encodes text[0, n) in mode, its ids into out: the kernels read what they read, under the geometry */
static int64_t enc(const toks_ctx *c, uint8_t *scr, const uint8_t *text, uint64_t n, uint32_t mode, uint32_t *out)
{
    return n == 0u ? 0 : toks_encode(c, text, n, mode | TOKS_NO_POSTPROCESS, out, 1024u, scr);
}

/* the kernels at the tables' ends */
static uint32_t edges(const toks_ctx *c, uint8_t *scr, const char *name, uint32_t *out)
{
    const toks_tables *t = &c->t;
    uint32_t done = 0;
    uint8_t buf[300];
    if (t->add_n != 0u) {                                   /* K1: the entry whose bytes end add_bytes */
        uint64_t best = 0u, e = 0u;
        for (uint64_t i = 0; i < t->add_n; i++) {
            uint64_t end = (uint64_t)t->add_entries[i].off + t->add_entries[i].len;
            if (end >= best) { best = end; e = i; }
        }
        uint32_t l = t->add_entries[e].len < 256u ? t->add_entries[e].len : 256u;
        if (l == 0u) { return done; }
        memcpy(buf + 1, t->add_bytes + t->add_entries[e].off, l);
        buf[0] = ' ';
        CHECK(enc(c, scr, buf + 1, l, TOKS_ADDED_ALL, out) >= 0 && enc(c, scr, buf, l + 1u, TOKS_ADDED_ALL, out) >= 0 &&
              enc(c, scr, buf + 1, l - 1u, TOKS_ADDED_ALL, out) >= 0, "%s: K1 at add_bytes' end", name);
        done++;
    }
    uint32_t last = t->n_ids;                               /* K5 / K6: the token whose bytes end tok_bytes */
    while (last > 0u && t->tok_off[last] == t->tok_off[t->n_ids] && t->tok_off[last - 1u] == t->tok_off[last]) { last--; }
    if (last > 0u) {
        uint32_t o = t->tok_off[last - 1u], l = t->tok_off[last] - o;
        l = l < 256u ? l : 256u;
        memcpy(buf, t->tok_bytes + o, l);
        CHECK(enc(c, scr, buf, l, TOKS_ADDED_NONE, out) >= 0, "%s: K5 / K6 on the last token's bytes", name);
        done++;
    }
    if (t->words != NULL && c->spm == NULL && c->wp == NULL && c->uni == NULL) {   /* K5: a key in the last bucket */
        for (uint32_t id = 0; id < t->n_ids; id++) {
            uint32_t o = t->tok_off[id], l = t->tok_off[id + 1u] - o;
            if (l < 2u || l > (uint32_t)TOKS_KEY_MAXLEN) { continue; }
            uint32_t h = bpe_key_hash(bpe_key_at(t->tok_bytes + o, l, 0u, l));
            if ((h & t->words_mask) != t->words_mask && (((h >> 16) | (h << 16)) & t->words_mask) != t->words_mask) { continue; }
            memcpy(buf, t->tok_bytes + o, l);
            CHECK(enc(c, scr, buf, l, TOKS_ADDED_NONE, out) >= 0, "%s: K5 on the last words bucket", name);
            done++;
            break;
        }
    }
    return done;
}

#if defined(TOKS_GUARD)
/* the seal's placement check on a block of its own: tables at offsets o (n bytes each), in the order given; 1 when the
 * seal stops the process (abort, in a child that catches it), 0 when it passes */
static void aborted(int sig) { (void)sig; _exit(78); }
static int seal_stops(const uint64_t *o, const uint64_t *n, int k)
{
    fflush(NULL);
    pid_t c = fork();
    if (c == 0) {
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = aborted;                            /* no core, no crash reporter */
        sigaction(SIGABRT, &sa, NULL);
        if (freopen("/dev/null", "w", stderr) == NULL) { _exit(3); }   /* the seal's message is expected here */
        uint8_t *b = toks_plat_arena(4096u);
        if (b == NULL) { _exit(3); }
        for (int i = 0; i < k; i++) { (void)toks_tab(b, o[i], n[i], TOKS_X_TOK_BYTES); }
        toks_tab_seal(b, 4096u);
        toks_tab_free(b, 4096u);
        _exit(0);
    }
    int st = 0;
    if (c < 0 || waitpid(c, &st, 0) != c) { return -1; }
    return !WIFEXITED(st) ? -1 : WEXITSTATUS(st) == 78 ? 1 : WEXITSTATUS(st) == 0 ? 0 : -1;
}

/* an empty table may share its offset with the next, in either order (the order the registry holds them in is the
 * order threads loaded and freed); a table past the block's end, across it, or over another stops the load */
static void seals(void)
{
    static const uint64_t o1[] = { 2048u, 2048u }, n1[] = { 40u, 0u }, n1r[] = { 0u, 40u };
    static const uint64_t o2[] = { 0u, 4096u }, n2[] = { 4096u, 0u };
    static const uint64_t o3[] = { 0u, 8192u }, n3[] = { 64u, 16u };
    static const uint64_t o4[] = { 0u, 4090u }, n4[] = { 64u, 16u };
    static const uint64_t o5[] = { 0u, 63u }, n5[] = { 64u, 16u };
    CHECK(seal_stops(o1, n1, 2) == 0 && seal_stops(o1, n1r, 2) == 0, "seal: an empty table sharing an offset stops the load");
    CHECK(seal_stops(o2, n2, 2) == 0, "seal: an empty table at the block's end stops the load");
    CHECK(seal_stops(o3, n3, 2) == 1, "seal: a table past the block's end passes");
    CHECK(seal_stops(o4, n4, 2) == 1, "seal: a table across the block's end passes");
    CHECK(seal_stops(o5, n5, 2) == 1, "seal: two overlapping tables pass");
}
#endif

static void one(const char *file)
{
    char buf[1024];
    const char *path = path_of(file, buf, sizeof buf);
#if defined(TOKS_GUARD)
    size_t t0 = guard_tabs();
#endif
    toks_ctx *c = NULL;
    if (toks_load(&c, path, NULL) != 0) {
        if (strchr(file, '/') != NULL) { CHECK(0, "%s: does not load", file); }
        else { printf("%s: SKIP (not in the tokenizer cache)\n", file); }
        return;
    }
    static const char text[] = "The guard pages sit flush against every table: 12 tables, 9 regions, 0 faults. "
                               "\xE4\xB8\xAD\xE6\x96\x87 caf\xC3\xA9 \xF0\x9F\x99\x82 <s> x\xE2\x96\x81y";
    uint32_t flags = TOKS_SCRATCH_CACHE_MIB(8);            /* the long cache's two regions as well */
    uint64_t sb = toks_scratch_bytes(c, 4096u, flags);
    uint8_t *scr = (uint8_t *)malloc((size_t)sb);
    uint32_t out[1024];
    CHECK(scr != NULL && toks_scratch_init(c, scr, sb, flags) == 0, "%s: scratch", file);
    if (scr == NULL) { toks_unload(c); return; }
    int64_t n = toks_encode(c, text, sizeof text - 1u, 0u, out, 1024u, scr);
    CHECK(n > 0, "%s: encode %" PRId64, file, n);
    uint32_t ne = edges(c, scr, file, out);
    toks_scratch *h = toks_scr_header(scr);
#if defined(TOKS_GUARD)
    uint32_t k = tables_of(c, file);
    size_t t1 = guard_tabs();
    for (size_t i = t0; i < t1; i++) {
        const uint8_t *p = NULL;
        uint64_t m = 0;
        if (guard_tab(i, &p, &m)) { probe("a table", file, p, m); }
    }
    const uint8_t *rp[16];
    uint64_t rn[16];
    uint32_t nr = guard_regions(h, rp, rn, 16u);
    CHECK(nr >= 5u, "%s: %u scratch regions", file, nr);   /* ends, caches, (memo), work, bounce, (...) */
    for (uint32_t i = 0; i < nr; i++) { probe("a scratch region", file, rp[i], rn[i]); }
    free(scr);
    toks_unload(c);
    CHECK(guard_tabs() == t0, "%s: %zu table maps outlive the unload", file, guard_tabs() - t0);
    printf("%s: %u table pointers, all guard tables; %zu tables and %u scratch regions probed (run %d); %" PRId64 " ids; "
           "%u kernel edges\n", file, k, t1 - t0, nr, TOKS_GUARD, n, ne);
#else
    CHECK(toks_scr_at(h, h->off_work) == scr + h->off_work && (uint8_t *)toks_scr_ends(h) == (uint8_t *)h + TOKS_SCR_HDR,
          "%s: toks_scr_at is not the shipped placement", file);
    printf("%s: %" PRId64 " ids, %u kernel edges (the shipped placement; the probes are make test-guard's)\n", file, n, ne);
    free(scr);
    toks_unload(c);
#endif
}

int main(void)
{
    static const char *const FILES[] = {
        "tests/data/compile/gpt2style.json", "tests/data/compile/dsv3style.json", "tests/data/compile/nosplit.json",
        "tests/data/spm/llamalike.json", "tests/data/unigram/bound_bf_meta.json", "gpt2", "gemma4", "uni_t5base",
        "wp-minilm-l6",
    };
    for (size_t i = 0; i < sizeof FILES / sizeof FILES[0]; i++) { one(FILES[i]); }
    uint8_t blk[256];                                       /* the two calls themselves */
    toks_arena ar = { blk, sizeof blk, 0u };
    void *a = toks_tab(blk, 64u, 16u, TOKS_X_TOK_BYTES), *b = toks_tab_ar(&ar, 16u, 64u, TOKS_X_TOK_OFF);
#if defined(TOKS_GUARD)
    CHECK(a != (void *)(blk + 64) && b != (void *)blk && ar.pos == 16u, "toks_tab in the guard build: not its own pages");
    seals();
#else
    CHECK(a == (void *)(blk + 64) && b == (void *)blk && ar.pos == 16u, "toks_tab is not the shipped placement");
#endif
    printf("test_guard: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
