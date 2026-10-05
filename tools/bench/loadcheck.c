/* tools/bench/loadcheck.c: load files through toks_load (or toks_load_mem_copy with -m) and print the verdict,
 * the diag, the wall time and the process's peak rss after each load (T9: refused with a diag or loaded, in
 * bounded time and memory). One line per file:
 *
 *   loadcheck [-m] [-t auto|scalar] file...
 *   <file> <code> <ms> <peak rss MiB> <n_ids> <algorithm> <diag>
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE                       /* ru_maxrss */
#define _DARWIN_C_SOURCE
#include "toks.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static double peak_mib(void)
{
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
#if defined(__APPLE__)
    return (double)u.ru_maxrss / 1048576.0;
#else
    return (double)u.ru_maxrss / 1024.0;
#endif
}

int main(int argc, char **argv)
{
    int mem = 0, opt;
    uint32_t tier = TOKS_TIER_AUTO;
    while ((opt = getopt(argc, argv, "mt:")) != -1) {
        if (opt == 'm') { mem = 1; }
        else if (opt == 't') { tier = strcmp(optarg, "scalar") == 0 ? TOKS_TIER_SCALAR : TOKS_TIER_AUTO; }
        else { fprintf(stderr, "usage: loadcheck [-m] [-t auto|scalar] file...\n"); return 2; }
    }
    for (int a = optind; a < argc; a++) {
        toks_load_opts o;
        toks_diag dg;
        memset(&o, 0, sizeof o);
        memset(&dg, 0, sizeof dg);
        o.size = sizeof o;
        o.tier = tier;
        o.diag = &dg;
        toks_ctx *ctx = NULL;
        uint8_t *src = NULL;
        uint64_t n = 0;
        if (mem) {
            FILE *f = fopen(argv[a], "rb");
            if (f == NULL) { printf("%s open-failed\n", argv[a]); continue; }
            fseek(f, 0, SEEK_END);
            long m = ftell(f);
            fseek(f, 0, SEEK_SET);
            src = (uint8_t *)malloc(m > 0 ? (size_t)m : 1u);
            n = fread(src, 1, m > 0 ? (size_t)m : 0u, f);
            fclose(f);
        }
        double t0 = now_ms();
        int64_t r = mem ? toks_load_mem_copy(&ctx, src, n, &o) : toks_load(&ctx, argv[a], &o);
        double t1 = now_ms();
        toks_info info;
        memset(&info, 0, sizeof info);
        info.size = sizeof info;
        if (r == 0) { toks_get_info(ctx, &info); }
        printf("%s %lld %.2f %.1f %u %u %s\n", argv[a], (long long)r, t1 - t0, peak_mib(), info.n_ids, info.algorithm,
               r != 0 ? dg.what : "-");
        fflush(stdout);
        toks_unload(ctx);
        free(src);
    }
    return 0;
}
