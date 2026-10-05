/*
 * test_tmpl.c: the load-time template invariants (compile.h toks_tmpl_invalid; SPEC §8.3 split parameters).
 *
 * Every (tmpl, tmpl_params, class tables) combination over tmpl 0..3 and params 0..0x3FF plus high bits, on
 * the four class-table builds (class_flags 0, MARKS_ARE_LETTERS, HAN, both), is accepted exactly when its
 * template's document defines it:
 *   TOKS_TMPL_NONE    params 0
 *   TOKS_TMPL_CL100K  bits within 0x7F, contractions not 0x03; tables without the Han bit (marks folded or not)
 *   TOKS_TMPL_O200K   bits within CONTR_CI | DIGITS_1 | HAN | NO_SLASH (contractions none or (?i), digits 1..3
 *                     or 1); marks keep base P; the Han bit in the tables iff TOKS_TP_HAN
 * plus broken tables (missing, a stage-1 index past n_blocks, the Han bit on an ascii byte), and every entry
 * of config.c's pattern table built the way compile.c builds it (its class_flags) passes.
 */
#include "core.h"
#include "classes.h"
#include "compile.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static long checks;

#define CHECK(cond, ...) do { checks++; if (!(cond)) { failures++; if (failures < 40) { \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

#define CLS_BYTES (128u + 2u * 0x1100u + 0x1100u * 256u)

static uint8_t BUF[4][CLS_BYTES];
static toks_class_tables CT[4];                         /* [class_flags]: 0, MARKS, HAN, HAN | MARKS */

static void use(toks_tables *t, uint32_t tmpl, uint32_t params, const toks_class_tables *c)
{
    memset(t, 0, sizeof *t);
    t->magic = TOKS_TABLES_MAGIC;
    t->version = TOKS_TABLES_VERSION;
    t->tmpl = tmpl;
    t->tmpl_params = params;
    if (c != NULL) {
        t->cls_ascii = c->ascii;
        t->cls_stage1 = c->stage1;
        t->cls_stage2 = c->stage2;
        t->cls_nblocks = c->n_blocks;
    }
}

/* the rule, restated from the documents (not from compile.c) */
static int want_ok(uint32_t tmpl, uint32_t p, uint32_t cf)
{
    int han_tables = (cf & TOKS_CLASSES_HAN) != 0u, marks = (cf & TOKS_CLASSES_MARKS_ARE_LETTERS) != 0u;
    switch (tmpl) {
    case TOKS_TMPL_NONE:
        return p == 0u;
    case TOKS_TMPL_CL100K:                  /* DIGIT_CUT (0x200) only with DIGITS_1 or 1_3 (kernels.md §3 A8); A9 A10 */
        return (p & ~0xE00u) <= 0x7Fu && (p & 3u) != 3u && ((p & 0x200u) == 0u || (p & 0x18u) <= 0x08u) && !han_tables;
    case TOKS_TMPL_O200K: {
        int contr_ok = (p & 3u) == 0u || (p & 3u) == 2u;
        int digits_ok = (p & 0x18u) == 0u || (p & 0x18u) == 0x08u;
        int other_ok = (p & ~(uint32_t)0x19Bu) == 0u;     /* no LPREFIX_ANY, PUNCT_NL, WS_NL, nothing past 0x100 */
        int han = (p & 0x80u) != 0u;
        return contr_ok && digits_ok && other_ok && !marks && han == han_tables;
    }
    default:
        return 0;
    }
}

int main(void)
{
    static const uint32_t FLAGS[4] = { 0u, TOKS_CLASSES_MARKS_ARE_LETTERS, TOKS_CLASSES_HAN,
                                       TOKS_CLASSES_HAN | TOKS_CLASSES_MARKS_ARE_LETTERS };
    for (int i = 0; i < 4; i++) {
        if (toks_classes_build(FLAGS[i], BUF[i], CLS_BYTES, &CT[i]) <= 0) { fprintf(stderr, "classes\n"); return 2; }
    }
    toks_tables t;
    long accepted = 0;

    /* every combination; high bits on top */
    static const uint32_t HIGH[] = { 0u, 0x200u, 0x400u, 0x800u, 0x1000u, 0x10000u, 0x80000000u };
    for (uint32_t tmpl = 0; tmpl < 5u; tmpl++) {                                /* bound: 5 (4 = unknown) */
        uint32_t tm = tmpl == 4u ? 0xFFFFFFFFu : tmpl;
        for (uint32_t h = 0; h < sizeof HIGH / sizeof HIGH[0]; h++) {             /* bound: 7 */
            for (uint32_t p = 0; p < 0x200u; p++) {                               /* bound: 0x200 */
                for (int c = 0; c < 4; c++) {                                     /* bound: 4 builds */
                    uint32_t params = p | HIGH[h];
                    use(&t, tm, params, &CT[c]);
                    const char *why = toks_tmpl_invalid(&t);
                    int ok = want_ok(tm, params, FLAGS[c]);
                    accepted += ok;
                    CHECK((why == NULL) == ok, "tmpl %#x params %#x class_flags %#x: %s (want %s)", tm, params,
                          FLAGS[c], why != NULL ? why : "accepted", ok ? "accepted" : "refused");
                }
            }
        }
    }
    /* the defined sets, counted: cl100k 96 params + 48 with DIGIT_CUT + 96 with NL_CUT + 96 with GB_SP x the 2 builds
     * without the Han bit; o200k 16 params (8 with HAN on the Han build, 8 without on the plain one); none: params 0
     * on any of the 4 builds */
    CHECK(accepted == (96 + 48 + 96 + 96) * 2 + 16 + 4, "accepted %ld combinations", accepted);

    /* broken tables */
    use(&t, TOKS_TMPL_CL100K, 0x66u, NULL);
    CHECK(toks_tmpl_invalid(&t) != NULL, "cl100k without class tables");
    use(&t, TOKS_TMPL_NONE, 0u, NULL);
    CHECK(toks_tmpl_invalid(&t) == NULL, "none without class tables");
    {
        static uint16_t s1[0x1100];
        static uint8_t ascii[128];
        memcpy(s1, CT[0].stage1, sizeof s1);
        s1[0x4Eu] = (uint16_t)CT[0].n_blocks;                                     /* one past the last block */
        use(&t, TOKS_TMPL_O200K, 0x02u, &CT[0]);
        t.cls_stage1 = s1;
        CHECK(toks_tmpl_invalid(&t) != NULL, "a stage-1 index past n_blocks");
        memcpy(ascii, CT[2].ascii, sizeof ascii);
        ascii['a'] = (uint8_t)(ascii['a'] | TOKS_C_HAN);
        use(&t, TOKS_TMPL_O200K, 0x182u, &CT[2]);
        CHECK(toks_tmpl_invalid(&t) == NULL, "kimi on the Han build");
        t.cls_ascii = ascii;
        CHECK(toks_tmpl_invalid(&t) != NULL, "the Han bit on an ascii byte");
        use(&t, TOKS_TMPL_O200K, 0x182u, &CT[2]);
        t.cls_nblocks = 0x1101u;
        CHECK(toks_tmpl_invalid(&t) != NULL, "n_blocks past 0x1100");
    }

    /* config.c's pattern table, built as compile.c builds it: every entry passes */
    for (uint32_t i = 0; i < TOKS_PATTERNS_N; i++) {                              /* bound: the table */
        const toks_pattern *pt = &TOKS_PATTERNS[i];
        int c = (int)(((pt->class_flags & TOKS_CLASSES_HAN) != 0u ? 2u : 0u) |
                      ((pt->class_flags & TOKS_CLASSES_MARKS_ARE_LETTERS) != 0u ? 1u : 0u));
        CHECK((pt->class_flags & ~(uint32_t)TOKS_CLASSES_KNOWN) == 0u, "pattern [%u] class_flags", i);
        use(&t, pt->tmpl, pt->params, &CT[c]);
        CHECK(toks_tmpl_invalid(&t) == NULL, "pattern [%u] (tmpl %u params %#x): %s", i, pt->tmpl, pt->params,
              toks_tmpl_invalid(&t));
    }

    printf("test_tmpl: %ld checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
