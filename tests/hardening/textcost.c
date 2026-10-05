/* tests/hardening/textcost.c: what one text costs toks_encode, against an en text of the same size (SPEC §7.3: for
 * inputs <= 1 MiB the worst adversarial class stays <= 10x the en-prose median), to judge a fuzz slow unit: a stall
 * of the library, or the harness's own battery under the sanitizers (docs/hardening.md). Cold calls: the scratch is
 * initialized before each one, outside the timer; best of -r reps.
 *
 *   textcost <tokenizer file> <text file> [-o off] [-l len] [-t tile] [-e en.txt] [-f flags] [-r reps]
 *
 * -o / -l take a slice of the text file (a fuzz input's payload starts after its 8-byte header: -o 8); -t repeats
 * the slice to that many bytes (a growth curve: run it at 4x steps). The en text is tiled to the same length
 * (default: a built-in paragraph). Prints the call's count, its ns per byte, en's and the ratio. Builds against libtoks.a: clang -O2 -Iinclude tests/hardening/textcost.c build/<os>-<isa>/libtoks.a -lpthread */
#define _POSIX_C_SOURCE 200809L
#include "toks.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint8_t *slurp(const char *path, uint64_t *len)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)n + 1u);
    if (p == NULL || fread(p, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(p); return NULL; }
    fclose(f);
    *len = (uint64_t)n;
    return p;
}

static const char EN[] = "The committee met on Tuesday to review the proposal, and after a long discussion of the "
    "budget, the schedule and the risks, it agreed to fund the first phase. Members asked for a written report "
    "by the end of the month, with costs broken down by quarter. ";

/* best of reps cold calls: ns per byte, *count the call's result */
static double cost(const toks_ctx *ctx, const uint8_t *t, uint64_t n, uint32_t flags, int reps, int64_t *count)
{
    uint64_t sb = toks_scratch_bytes(ctx, n, 0);
    void *scr = malloc((size_t)sb);
    uint64_t cap = 4u * n + 64u, best = UINT64_MAX;
    uint32_t *out = malloc((size_t)cap * 4u);
    for (int r = 0; r < reps; r++) {
        if (toks_scratch_init(ctx, scr, sb, 0) != 0) { fprintf(stderr, "scratch init\n"); exit(2); }
        uint64_t a = now_ns();
        *count = toks_encode(ctx, t, n, flags, out, cap, scr);
        uint64_t b = now_ns() - a;
        if (b < best) { best = b; }
    }
    free(out);
    free(scr);
    return n != 0 ? (double)best / (double)n : 0.0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: textcost <tokenizer> <text> [-o off] [-l len] [-e en] [-f flags] [-r reps]\n"); return 2; }
    uint64_t off = 0, len = UINT64_MAX, tile = 0;
    uint32_t flags = 0;
    int reps = 5;
    const char *enp = NULL;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-o") == 0) { off = strtoull(argv[i + 1], NULL, 10); }
        else if (strcmp(argv[i], "-l") == 0) { len = strtoull(argv[i + 1], NULL, 10); }
        else if (strcmp(argv[i], "-t") == 0) { tile = strtoull(argv[i + 1], NULL, 10); }
        else if (strcmp(argv[i], "-e") == 0) { enp = argv[i + 1]; }
        else if (strcmp(argv[i], "-f") == 0) { flags = (uint32_t)strtoul(argv[i + 1], NULL, 0); }
        else if (strcmp(argv[i], "-r") == 0) { reps = atoi(argv[i + 1]); }
    }
    toks_ctx *ctx = NULL;
    int64_t r = toks_load(&ctx, argv[1], NULL);
    if (r != 0) { fprintf(stderr, "%s: toks_load %" PRId64 "\n", argv[1], r); return 2; }
    uint64_t fl = 0;
    uint8_t *f = slurp(argv[2], &fl);
    if (f == NULL || off > fl) { fprintf(stderr, "%s: unreadable or shorter than -o\n", argv[2]); return 2; }
    uint64_t n = fl - off < len ? fl - off : len;
    uint8_t *text = f + off;
    if (tile != 0u && n != 0u) {                        /* the slice repeated to tile bytes */
        uint8_t *t2 = malloc((size_t)tile);
        for (uint64_t i = 0; i < tile; i++) { t2[i] = text[i % n]; }
        text = t2;
        n = tile;
    }
    uint64_t el = 0;
    uint8_t *e0 = enp != NULL ? slurp(enp, &el) : (uint8_t *)EN;
    if (enp == NULL) { el = sizeof EN - 1u; }
    uint8_t *en = malloc((size_t)n + 1u);
    for (uint64_t i = 0; i < n; i++) { en[i] = e0[i % el]; }
    int64_t ct = 0, ce = 0;
    double t = cost(ctx, text, n, flags, reps, &ct), e = cost(ctx, en, n, flags, reps, &ce);
    printf("%s: %" PRIu64 " bytes, %" PRId64 " ids, %.2f ns/B; en at the same size %" PRId64 " ids, %.2f ns/B; ratio %.2f\n",
           argv[2], n, ct, t, ce, e, e > 0.0 ? t / e : 0.0);
    toks_unload(ctx);
    return 0;
}
