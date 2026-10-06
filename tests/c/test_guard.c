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
 *    unmaps its tables.
 * Each probe that must fault runs in a child process. In the shipped build the test checks that toks_tab and
 * toks_scr_at are the shipped placement (block + o, base + off) and says the probes are make test-guard's.
 */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DARWIN_C_SOURCE 1
#endif
#include "core.h"
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
/* 1 when a read of *p kills a child process (SIGSEGV or SIGBUS) */
static int faults(const uint8_t *p)
{
    fflush(NULL);
    pid_t c = fork();
    if (c == 0) {
        volatile uint8_t v = *(const volatile uint8_t *)p;
        (void)v;
        _exit(0);
    }
    int st = 0;
    if (c < 0 || waitpid(c, &st, 0) != c) { return 0; }
    return WIFSIGNALED(st) && (WTERMSIG(st) == SIGSEGV || WTERMSIG(st) == SIGBUS);
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
    printf("%s: %u table pointers, all guard tables; %zu tables and %u scratch regions probed (run %d); %" PRId64 " ids\n",
           file, k, t1 - t0, nr, TOKS_GUARD, n);
#else
    CHECK(toks_scr_at(h, h->off_work) == scr + h->off_work && (uint8_t *)toks_scr_ends(h) == (uint8_t *)h + TOKS_SCR_HDR,
          "%s: toks_scr_at is not the shipped placement", file);
    printf("%s: %" PRId64 " ids (the shipped placement; the probes are make test-guard's)\n", file, n);
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
#else
    CHECK(a == (void *)(blk + 64) && b == (void *)blk && ar.pos == 16u, "toks_tab is not the shipped placement");
#endif
    printf("test_guard: %ld checks, %d failures\n", checks, failures);
    return failures != 0;
}
