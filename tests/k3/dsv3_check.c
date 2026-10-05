/*
 * dsv3_check.c: differential checker of the dsv3 K3 c twin against hf tokenizers 0.23.2 at scale.
 * NOT part of `make test` (it reads the case stream of tests/k3/dsv3_gen.py); tests/k3/dsv3_run.sh builds and
 * runs it:
 *
 *   uv run --python 3.12 --with tokenizers==0.23.2 python tests/k3/dsv3_gen.py stream ... | ./dsv3_check -
 *   ./dsv3_check --classes classes.bin   (the class byte of every code point, for dsv3_gen.py classes)
 *
 * Records: u8 oracle (0 hf, 1 the python model on ill-formed bytes), u32 len, text bytes, u32 n_pieces,
 * n_pieces x (u32 start, u32 end). Per record: one call with a big cap must give exactly the oracle's ends; a
 * cap-1 resume tiling and a resume tiling with a pseudo-random cap per call must give the same ends.
 * Exit 0 iff all matched.
 */
/* kernels.h -> core.h redeclares memcpy/memset/memcmp (freestanding core): it must come before <string.h>. */
#include "../../src/core/classes.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"

/* the tier under test: the c twin by default, an asm tier with -DK3_SCAN=toks_k3_scan_dsv3_<tier> */
#ifndef K3_SCAN
#define K3_SCAN toks_k3_scan_dsv3_c
#endif
uint64_t K3_SCAN(const toks_tables *t, toks_k3_args *a);
/* the tier's whole kernel: an asm part runs with the c twin's stretches (kernels.h toks_k3_tier) */
static uint64_t k3_scan(const toks_tables *t, toks_k3_args *a)
{
    return toks_k3_tier(K3_SCAN, toks_k3_scan_dsv3_c, t, a);
}

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXREC (1u << 23)                       /* bytes of text and pieces per record (dsv3_gen.py) */

static uint8_t cls_buf[128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CT;
static toks_tables TT;

/* the tables compile.c builds for the dsv3 template: class_flags TOKS_CLASSES_DSV3 */
static void build(void)
{
    if (toks_classes_build(TOKS_CLASSES_DSV3, cls_buf, sizeof cls_buf, &CT) <= 0) {
        fprintf(stderr, "classes build failed\n");
        exit(2);
    }
    memset(&TT, 0, sizeof TT);
    TT.magic = TOKS_TABLES_MAGIC;
    TT.version = TOKS_TABLES_VERSION;
    TT.tmpl = TOKS_TMPL_DSV3;
    TT.cls_ascii = CT.ascii;
    TT.cls_stage1 = CT.stage1;
    TT.cls_stage2 = CT.stage2;
    TT.cls_nblocks = CT.n_blocks;
#ifdef K3_CLS_FLAGS
    TT.flags = (uint32_t)toks_compile_cls_flags(&TT);  /* an asm tier (dsv3_run.sh K3_SCAN): its Han shortcut */
#endif
}

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n ? 0 : -1; }

static uint8_t text[MAXREC];
static uint32_t want[2 * MAXREC], got[MAXREC], acc[MAXREC];

/* tile [0, len) with calls of the given caps; 0 = ok */
static int tile(uint64_t len, uint32_t n_ends, uint64_t seed, int fixed1)
{
    uint64_t pos = 0, an = 0;
    while (pos < len) {                                 /* bound: pieces (each call advances >= 1) */
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        a.text = text; a.len = len; a.pos = pos; a.ends = acc + an;
        a.cap = fixed1 ? 1u : 1u + ((seed >> 33) % 7u);
        if (an + a.cap > MAXREC) { a.cap = MAXREC - an; }
        uint64_t c = k3_scan(&TT, &a);
        if (c == 0u || c > a.cap || a.n != c || a.pos <= pos || a.pos != acc[an + c - 1u]) { return 1; }
        an += c;
        pos = a.pos;
    }
    return (an != n_ends || (n_ends != 0u && memcmp(acc, want, (size_t)n_ends * 4u) != 0)) ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s cases.bin|- | --classes out.bin\n", argv[0]); return 2; }
    build();
    if (strcmp(argv[1], "--classes") == 0) {
        if (argc < 3) { return 2; }
        FILE *o = fopen(argv[2], "wb");
        if (!o) { perror(argv[2]); return 2; }
        for (uint32_t cp = 0; cp < 0x110000u; cp++) {   /* bound: 0x110000 code points */
            uint8_t c = (cp >= 0xD800u && cp <= 0xDFFFu) ? 0u : toks_class_of(&CT, cp);
            fputc(c, o);
        }
        fclose(o);
        printf("dsv3_check: wrote the class byte of 0x110000 code points (TOKS_CLASSES_DSV3)\n");
        return 0;
    }
    FILE *f = strcmp(argv[1], "-") == 0 ? stdin : fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    uint64_t cases = 0, bad = 0, bytes = 0, pieces = 0, model_cases = 0;
    uint8_t oracle;
    while (rd(f, &oracle, 1) == 0) {
        uint32_t len, n_pairs;
        if (oracle > 1 || rd(f, &len, 4) != 0 || len > MAXREC || rd(f, text, len) != 0 ||
            rd(f, &n_pairs, 4) != 0 || n_pairs > MAXREC || rd(f, want, (size_t)n_pairs * 8u) != 0) {
            fprintf(stderr, "corrupt case stream at case %" PRIu64 "\n", cases);
            return 2;
        }
        cases++;
        model_cases += oracle;
        bytes += len;
        pieces += n_pairs;
        /* the oracle's pieces must tile [0, len): piece i is [want[2i], want[2i+1]) */
        int tiles = 1;
        for (uint32_t i = 0; i < n_pairs; i++) {        /* bound: n_pairs */
            if (want[2u * i] != (i == 0u ? 0u : want[2u * i - 1u]) || want[2u * i] >= want[2u * i + 1u] ||
                want[2u * i + 1u] > len) { tiles = 0; }
        }
        for (uint32_t i = 0; i < n_pairs; i++) { want[i] = want[2u * i + 1u]; }   /* bound: n_pairs */
        if ((n_pairs != 0u && want[n_pairs - 1u] != len) || (n_pairs == 0u && len != 0u)) { tiles = 0; }
        if (!tiles) {
            fprintf(stderr, "BAD ORACLE case %" PRIu64 ": the pieces do not tile [0, len)\n", cases);
            bad++;
            continue;
        }
        toks_k3_args a;
        memset(&a, 0, sizeof a);
        a.text = text; a.len = len; a.pos = 0; a.ends = got; a.cap = MAXREC;
        uint64_t n = k3_scan(&TT, &a);
        if (n != n_pairs || (n_pairs != 0u && memcmp(got, want, (size_t)n_pairs * 4u) != 0)) {
            if (bad < 10) {
                fprintf(stderr, "MISMATCH case %" PRIu64 " oracle %u len %u: n %" PRIu64 " want %u; text", cases,
                        oracle, len, n, n_pairs);
                for (uint32_t i = 0; i < len && i < 64u; i++) { fprintf(stderr, " %02X", text[i]); }
                fprintf(stderr, "\n");
            }
            bad++;
            continue;
        }
        if (tile(len, n_pairs, 1u, 1) != 0 || tile(len, n_pairs, cases, 0) != 0) {
            if (bad < 10) { fprintf(stderr, "RESUME MISMATCH case %" PRIu64 "\n", cases); }
            bad++;
        }
    }
    printf("dsv3_check: %" PRIu64 " cases (%" PRIu64 " ill-formed, model oracle; %" PRIu64 " bytes, %" PRIu64
           " pieces), %" PRIu64 " mismatches\n", cases, model_cases, bytes, pieces, bad);
    return bad ? 1 : 0;
}
