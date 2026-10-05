/* tests/fuzz/arprobe.c: fz_ar_probe (arprobe.h), linked into every harness of the fuzz build. Per toks_ar_alloc
 * site: the calls, the refusals (the call returned NULL) and the least slack seen, with that call's bound and end:
 *   arprobe <file>:<line> calls <n> refused <r> min_slack <bytes> bound <len> end <aligned start + n>
 * one line per site at exit, sorted by file and line; a refusal also prints at once, as
 *   arprobe REFUSED <file>:<line> bound <len> pos <pos> n <n> align <align>
 * (the harness then sees the builder's TOKS_E_NOMEM and reports the input as a finding, load.h). */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fz_ar_probe(const char *file, uint32_t line, uint64_t len, uint64_t pos, uint64_t n, uint64_t align, int got);

#define FZ_AR_SITES 256u                        /* toks_ar_alloc sites in src/core: 51 at this writing */

typedef struct {
    const char *file;
    uint32_t line;
    uint64_t calls, refused, min_slack, len, end;
} fz_ar_site;

static fz_ar_site fz_ar[FZ_AR_SITES];
static uint32_t fz_ar_n;
static int fz_ar_reg;

static int fz_ar_cmp(const void *x, const void *y)
{
    const fz_ar_site *a = (const fz_ar_site *)x, *b = (const fz_ar_site *)y;
    int c = strcmp(a->file, b->file);
    return c != 0 ? c : (a->line > b->line) - (a->line < b->line);
}

static void fz_ar_dump(void)
{
    qsort(fz_ar, fz_ar_n, sizeof fz_ar[0], fz_ar_cmp);
    for (uint32_t i = 0; i < fz_ar_n; i++) {
        const fz_ar_site *s = &fz_ar[i];
        fprintf(stderr, "arprobe %s:%" PRIu32 " calls %" PRIu64 " refused %" PRIu64 " min_slack %" PRIu64 " bound %" PRIu64
                " end %" PRIu64 "\n", s->file, s->line, s->calls, s->refused, s->min_slack, s->len, s->end);
    }
}

void fz_ar_probe(const char *file, uint32_t line, uint64_t len, uint64_t pos, uint64_t n, uint64_t align, int got)
{
    if (!fz_ar_reg) { fz_ar_reg = 1; atexit(fz_ar_dump); }
    fz_ar_site *s = NULL;
    for (uint32_t i = 0; i < fz_ar_n && s == NULL; i++) {
        if (fz_ar[i].line == line && strcmp(fz_ar[i].file, file) == 0) { s = &fz_ar[i]; }
    }
    if (s == NULL) {
        if (fz_ar_n == FZ_AR_SITES) { fprintf(stderr, "arprobe: more than %u sites\n", FZ_AR_SITES); abort(); }
        s = &fz_ar[fz_ar_n++];
        s->file = file;
        s->line = line;
        s->min_slack = UINT64_MAX;
    }
    s->calls++;
    if (!got) {
        s->refused++;
        fprintf(stderr, "arprobe REFUSED %s:%" PRIu32 " bound %" PRIu64 " pos %" PRIu64 " n %" PRIu64 " align %" PRIu64 "\n",
                file, line, len, pos, n, align);
        return;
    }
    uint64_t end = ((pos + (align - 1u)) & ~(align - 1u)) + n;
    if (len - end < s->min_slack) {
        s->min_slack = len - end;
        s->len = len;
        s->end = end;
    }
}
