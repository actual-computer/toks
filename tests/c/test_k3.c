/*
 * test_k3.c: hand cases for the K3 c twin (kernels.md §2-3): every rule, every §3 note, every
 * variant, caps 1..n with resume equivalence, guard-page geometry (no read at/past len), invalid
 * utf-8 at every position, empty and one-byte inputs.
 *
 * Expectations come from a plain reference scanner that restates the doc's rules (atoms via
 * toks_utf8_len's contract, classes via the built class tables, A1-A7 with the variant's
 * TOKS_TP_* bits) -- written from the prose, not from the kernel. The outer oracle is hf 0.23.2 at
 * scale (tests/k3/, run on lab hosts); this file is the fast gate under `make test`.
 */
/* kernels.h -> core.h redeclares memcpy/memset/memcmp (freestanding core): it must come before
 * <string.h>, whose secure _chk macros would collide with the declarations. */
#include "../../src/core/classes.h"
#include "../../src/core/compile.h"
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"
#include "../common/guard.h"

/* the tier under test: the c twin by default; for an asm tier, build this file by hand with
 * -DK3_SCAN=toks_k3_scan_cl100k_<tier> plus its .S file */
#ifndef K3_SCAN
#define K3_SCAN toks_k3_scan_cl100k_c
#endif
uint64_t K3_SCAN(const toks_tables *t, toks_k3_args *a);
/* the tier's whole kernel: an asm part runs with the c twin's stretches (kernels.h toks_k3_tier) */
static uint64_t k3_scan(const toks_tables *t, toks_k3_args *a)
{
    return toks_k3_tier(K3_SCAN, toks_k3_scan_cl100k_c, t, a);
}

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- variants */

typedef struct { const char *name; uint32_t params; uint32_t cls_flags; } variant;

static const variant VARIANTS[] = {
    { "gpt2",   TOKS_TP_CONTR_CS | TOKS_TP_DIGITS_SP_RUN, 0u },
    { "cl100k", TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_PUNCT_NL | TOKS_TP_WS_NL, 0u },
    { "qwen2",  TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL
                | TOKS_TP_WS_NL, 0u },
    { "qwen35", TOKS_TP_CONTR_CI | TOKS_TP_LPREFIX_ANY | TOKS_TP_DIGITS_1 | TOKS_TP_PUNCT_NL
                | TOKS_TP_WS_NL, TOKS_CLASSES_MARKS_ARE_LETTERS },
};
#define NVARIANTS (sizeof VARIANTS / sizeof VARIANTS[0])

/* class tables: one build per distinct cls_flags (plain, marks-folded) */
static uint8_t cls_buf[2][128 + 2 * 0x1100 + 0x1100 * 256];
static toks_class_tables CTAB[2];
static toks_tables TT[NVARIANTS];
static int ready;

static void build_tables(void)
{
    if (ready) { return; }
    for (size_t v = 0; v < NVARIANTS; v++) {
        int idx = (VARIANTS[v].cls_flags & TOKS_CLASSES_MARKS_ARE_LETTERS) != 0;
        if (CTAB[idx].stage1 == NULL) {
            int used = toks_classes_build(VARIANTS[v].cls_flags, cls_buf[idx], sizeof cls_buf[idx],
                                          &CTAB[idx]);
            if (used <= 0) { fprintf(stderr, "classes build failed\n"); exit(2); }
        }
        toks_tables *t = &TT[v];
        memset(t, 0, sizeof *t);
        t->magic = TOKS_TABLES_MAGIC;
        t->version = TOKS_TABLES_VERSION;
        t->tmpl = TOKS_TMPL_CL100K;
        t->tmpl_params = VARIANTS[v].params;
        t->cls_ascii = CTAB[idx].ascii;
        t->cls_stage1 = CTAB[idx].stage1;
        t->cls_stage2 = CTAB[idx].stage2;
        t->cls_nblocks = CTAB[idx].n_blocks;
        int64_t cf = toks_compile_cls_flags(t);           /* TOKS_TF_CJK_L, as the compiler sets it */
        if (cf < 0) { fprintf(stderr, "class tables break kernels.md §2\n"); exit(2); }
        t->flags = (uint32_t)cf;
    }
    ready = 1;
}

/* ---------------------------------------------------------------- the reference scanner */

typedef struct { uint8_t cls; uint32_t cp; uint32_t k; } atom;

/* the atom at p (avail bytes left); an ill-formed first byte is a one-byte P atom (§2) */
static atom ref_atom(const toks_class_tables *ct, const uint8_t *p, uint64_t avail)
{
    atom a;
    uint32_t k = toks_utf8_len(p, avail);
    if (k == 0u) { a.cls = TOKS_CLS_P; a.cp = p[0]; a.k = 1u; return a; }
    a.k = k;
    if (k == 1u) { a.cp = p[0]; a.cls = ct->ascii[p[0]]; return a; }
    a.cp = toks_cp_decode(p, k);   /* core.h: exact for the well-formed lengths */
    a.cls = toks_class_of(ct, a.cp);
    return a;
}

#define BASE(c) ((uint8_t)((c) & 0x07u))

/* A1-A7 as written (§3); returns pieces, fills ends. Absent atoms have cls 0xFF. */
static uint64_t ref_scan(const variant *v, const toks_class_tables *ct,
                         const uint8_t *text, uint64_t len, uint64_t pos, uint64_t cap,
                         uint32_t *ends)
{
    uint32_t p = v->params;
    uint32_t contr = p & TOKS_TP_CONTR_MASK;
    int lprefix_any = (p & TOKS_TP_LPREFIX_ANY) != 0;
    uint32_t digits = p & TOKS_TP_DIGITS_MASK;
    int punct_nl = (p & TOKS_TP_PUNCT_NL) != 0;
    int ws_nl = (p & TOKS_TP_WS_NL) != 0;
    uint64_t n = 0;
    while (pos < len && n < cap) {                       /* bound: min(pieces, cap) */
        atom c = ref_atom(ct, text + pos, len - pos);
        uint64_t p1 = pos + c.k;
        atom c1 = { 0xFFu, 0, 0 };
        if (p1 < len) { c1 = ref_atom(ct, text + p1, len - p1); }
        uint64_t end = pos;

        /* A1 contractions */
        if (contr != TOKS_TP_CONTR_NONE && c.cp == 0x27u && p1 < len) {
            if (contr == TOKS_TP_CONTR_CS) {
                if (c1.cp == 0x73u || c1.cp == 0x74u || c1.cp == 0x6Du || c1.cp == 0x64u) {
                    end = p1 + c1.k;
                } else if (p1 + c1.k < len) {
                    atom c2 = ref_atom(ct, text + p1 + c1.k, len - p1 - c1.k);
                    if ((c1.cp == 0x72u && c2.cp == 0x65u) || (c1.cp == 0x76u && c2.cp == 0x65u) ||
                        (c1.cp == 0x6Cu && c2.cp == 0x6Cu)) {
                        end = p1 + c1.k + c2.k;
                    }
                }
            } else {
                if (c1.cp == 0x73u || c1.cp == 0x53u || c1.cp == 0x74u || c1.cp == 0x54u ||
                    c1.cp == 0x6Du || c1.cp == 0x4Du || c1.cp == 0x64u || c1.cp == 0x44u ||
                    (c1.cls & TOKS_CLS_FOLD_S) != 0u) {
                    end = p1 + c1.k;
                } else if (p1 + c1.k < len) {
                    atom c2 = ref_atom(ct, text + p1 + c1.k, len - p1 - c1.k);
                    if (((c1.cp == 0x72u || c1.cp == 0x52u) && (c2.cp == 0x65u || c2.cp == 0x45u)) ||
                        ((c1.cp == 0x76u || c1.cp == 0x56u) && (c2.cp == 0x65u || c2.cp == 0x45u)) ||
                        ((c1.cp == 0x6Cu || c1.cp == 0x4Cu) && (c2.cp == 0x6Cu || c2.cp == 0x4Cu))) {
                        end = p1 + c1.k + c2.k;
                    }
                }
            }
        }

        /* A2 letters */
        if (end == pos) {
            if (BASE(c.cls) == TOKS_CLS_L) {
                end = p1;
                while (end < len) {                       /* bound: len - end (maximal L run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_L) { break; }
                    end += x.k;
                }
            } else if (BASE(c1.cls) == TOKS_CLS_L &&
                       (lprefix_any ? (BASE(c.cls) == TOKS_CLS_P || BASE(c.cls) == TOKS_CLS_WS)
                                    : c.cp == 0x20u)) {
                end = p1 + c1.k;
                while (end < len) {                       /* bound: len - end (maximal L run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_L) { break; }
                    end += x.k;
                }
            }
        }

        /* A3 digits */
        if (end == pos && BASE(c.cls) == TOKS_CLS_N) {
            if (digits == TOKS_TP_DIGITS_1_3) {
                end = p1;
                uint32_t cnt = 1u;
                while (cnt < 3u && end < len) {           /* bound: 3 atoms (\p{N}{1,3}) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_N) { break; }
                    end += x.k;
                    cnt++;
                }
            } else if (digits == TOKS_TP_DIGITS_1) {
                end = p1;
            } else {
                end = p1;
                while (end < len) {                       /* bound: len - end (maximal N run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_N) { break; }
                    end += x.k;
                }
            }
        }
        if (end == pos && digits == TOKS_TP_DIGITS_SP_RUN && c.cp == 0x20u &&
            BASE(c1.cls) == TOKS_CLS_N) {
            end = p1 + c1.k;
            while (end < len) {                           /* bound: len - end (maximal N run) */
                atom x = ref_atom(ct, text + end, len - end);
                if (BASE(x.cls) != TOKS_CLS_N) { break; }
                end += x.k;
            }
        }

        /* A4 punctuation */
        if (end == pos) {
            if (BASE(c.cls) == TOKS_CLS_P) {
                end = p1;
                while (end < len) {                       /* bound: len - end (maximal P run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_P) { break; }
                    end += x.k;
                }
            } else if (c.cp == 0x20u && BASE(c1.cls) == TOKS_CLS_P) {
                end = p1 + c1.k;
                while (end < len) {                       /* bound: len - end (maximal P run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_P) { break; }
                    end += x.k;
                }
            }
            if (end != pos && punct_nl) {
                while (end < len) {                       /* bound: len - end (maximal NL run) */
                    atom x = ref_atom(ct, text + end, len - end);
                    if (BASE(x.cls) != TOKS_CLS_NL) { break; }
                    end += x.k;
                }
            }
        }

        /* A5/A6/A7 whitespace */
        if (end == pos) {
            uint64_t e = pos, last_nl = 0, last_start = pos;
            while (e < len) {                             /* bound: len - e (maximal WS|NL run) */
                atom x = ref_atom(ct, text + e, len - e);
                if (BASE(x.cls) != TOKS_CLS_WS && BASE(x.cls) != TOKS_CLS_NL) { break; }
                if (BASE(x.cls) == TOKS_CLS_NL) { last_nl = e + x.k; }
                last_start = e;
                e += x.k;
            }
            if (ws_nl && last_nl != 0u) { end = last_nl; }        /* A5 */
            else if (e == len) { end = e; }                        /* A6, run ends segment */
            else if (last_start > pos) { end = last_start; }       /* A6, give back one atom */
            else { end = e; }                                     /* A7, one atom */
        }

        ends[n] = (uint32_t)end;
        n++;
        pos = end;
    }
    return n;
}

/* ---------------------------------------------------------------- harness */

#define MAXCASE 4096
static int failures;
static uint64_t checks;

static void run_case(const variant *v, const toks_tables *t, const toks_class_tables *ct,
                     const char *label, const uint8_t *text, uint64_t len)
{
    static uint32_t want[MAXCASE], got[MAXCASE], acc[MAXCASE];
    uint64_t rn = ref_scan(v, ct, text, len, 0, MAXCASE, want);
    toks_k3_args a;
    memset(&a, 0, sizeof a);
    a.text = text; a.len = len; a.pos = 0; a.ends = got; a.cap = MAXCASE;
    uint64_t kn = k3_scan(t, &a);
    checks++;
    if (kn != rn || memcmp(want, got, (size_t)rn * sizeof(uint32_t)) != 0) {
        if (failures < 20) {
            fprintf(stderr, "FAIL %s %s: n want %" PRIu64 " got %" PRIu64 "\n  ref:",
                    v->name, label, rn, kn);
            for (uint64_t i = 0; i < rn; i++) fprintf(stderr, " %" PRIu32, want[i]);
            fprintf(stderr, "\n  got:");
            for (uint64_t i = 0; i < kn; i++) fprintf(stderr, " %" PRIu32, got[i]);
            fprintf(stderr, "\n");
        }
        failures++;
        return;
    }
    /* contract: strictly increasing, each in (prev, len]; pos = last end */
    uint64_t prev = 0;
    for (uint64_t i = 0; i < rn; i++) {
        if (got[i] <= prev || got[i] > len) {
            fprintf(stderr, "FAIL %s %s: end %" PRIu32 " #%zu not in (%" PRIu64 ", %" PRIu64 "]\n",
                    v->name, label, got[i], (size_t)i, prev, len);
            failures++;
            return;
        }
        prev = got[i];
    }
    if (rn != 0u && a.pos != got[rn - 1u]) {
        fprintf(stderr, "FAIL %s %s: pos %" PRIu64 " != last end\n", v->name, label, a.pos);
        failures++;
        return;
    }

    /* resume equivalence: every cap 1..rn+1 must tile to the same pieces */
    for (uint64_t cap = 1; cap <= rn + 1u; cap++) {
        uint64_t pos = 0, an = 0;
        while (pos < len) {
            memset(&a, 0, sizeof a);
            a.text = text; a.len = len; a.pos = pos; a.ends = acc + an; a.cap = cap;
            uint64_t c = k3_scan(t, &a);
            if (c == 0u || c > cap || a.pos <= pos) {
                fprintf(stderr, "FAIL %s %s cap %" PRIu64 ": c=%" PRIu64 " pos=%" PRIu64 ">%" PRIu64
                        "\n", v->name, label, cap, c, a.pos, pos);
                failures++;
                return;
            }
            an += c;
            pos = a.pos;
        }
        if (an != rn || memcmp(acc, want, (size_t)rn * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "FAIL %s %s: resume cap %" PRIu64 " gives %" PRIu64 " pieces (want %"
                    PRIu64 ")\n", v->name, label, cap, an, rn);
            failures++;
            return;
        }
        checks++;
    }
}

static void case_all(const char *label, const uint8_t *text, uint64_t len)
{
    for (size_t v = 0; v < NVARIANTS; v++) {
        int idx = (VARIANTS[v].cls_flags & TOKS_CLASSES_MARKS_ARE_LETTERS) != 0;
        run_case(&VARIANTS[v], &TT[v], &CTAB[idx], label, text, len);
    }
}

static void s(const char *label, const char *utf8)   /* a C string (no embedded NUL) */
{
    case_all(label, (const uint8_t *)utf8, strlen(utf8));
}

/* ---------------------------------------------------------------- cases */

int main(void)
{
    build_tables();

    /* empty and one-byte inputs */
    s("empty", "");
    s("'a'", "a");
    s("space", " ");
    s("tab", "\t");
    s("CR", "\r");
    s("LF", "\n");
    s("digit", "7");
    s("punct", "!");
    { const uint8_t nul[] = { 0 }; case_all("NUL", nul, 1); }

    /* A1 contractions: CS vs CI, and the (?i) fold notes */
    s("'s", "'s"); s("'t", "'t"); s("'re", "'re"); s("'ve", "'ve");
    s("'m", "'m"); s("'ll", "'ll"); s("'d", "'d");
    s("'S", "'S"); s("'T", "'T"); s("'RE", "'RE"); s("'Ve", "'Ve");
    s("'LL", "'LL"); s("'D", "'D");
    s("'rE", "'rE"); s("'Ll", "'Ll"); s("'lL", "'lL"); s("'vE", "'vE");
    s("'x", "'x"); s("''", "''"); s("''s", "''s"); s("' s", "' s");
    s("don't", "don't"); s("'s's's", "'s's's");
    /* U+017F long s: CI matches 'ſ as 's; CS does not; nothing else folds in */
    s("'U+017F", "'\xC5\xBF");
    s("'U+017F end", "x'\xC5\xBF");
    s("a'U+017Fb", "a'\xC5\xBF" "b");
    s("'U+017F then x", "'\xC5\xBF" "x");
    /* U+212A KELVIN is not a contraction letter under (?i) */
    s("'U+212A", "'\xE2\x84\xAA");
    /* a 2-byte atom whose bytes could look like e in a byte-wise check */
    s("' + U+00E9", "'\xC3\xA9");
    s("' at end", "x'");
    s("'r at end", "x'r");

    /* A2 letters and both prefix rules */
    s("letters", "hello");
    s("letters run", "helloWorldXYZ");
    s("letter digit", "abc123");
    s("space prefix", " abc");
    s("tab prefix", "\tabc");
    s("NBSP prefix", "\xC2\xA0" "abc");      /* WS: ANY admits it, gpt-2 does not */
    s("CR prefix", "\rabc");
    s("LF prefix", "\nabc");
    s("punct prefix", "!abc");
    s("digit prefix", "1abc");
    s("accented letters", "\xC3\xA9" "abc");
    s("cjk run", "\xE4\xB8\xAD\xE6\x96\x87");
    s("emoji then letter", "\xF0\x9F\x98\x80" "x");
    s("zwj chain", "\xF0\x9F\x91\x8D\xE2\x80\x8D\xF0\x9F\x8F\xBB");
    s("mark prefix", "a\xCC\x81" "b");       /* U+0301: L only for qwen35 */
    s("mark run", "e\xCC\x81\xCC\x81");
    s("mark alone", "\xCC\x81");
    s("mark then punct", "\xCC\x81!");
    s("mark then digit", "\xCC\x81" "1");
    s("mark starts run", "\xCC\x81" "ab");

    /* A3 digits in each mode */
    s("1 digit", "5");
    s("2 digits", "55");
    s("3 digits", "555");
    s("4 digits", "5555");
    s("5 digits", "55555");
    s("9 digits", "123456789");
    s("digits then letter", "123abc");
    s("arabic-indic", "\xD9\xB0\xD9\xB1\xD9\xB2");
    s("circled digit", "\xE2\x91\xA0");
    s("fullwidth digit", "\xEF\xBC\x90");
    s("superscript two", "\xC2\xB2");
    s("digit ws digit", "1 2");
    s("space digits", " 12");
    s("tab digits", "\t12");
    s("NBSP digits", "\xC2\xA0" "12");       /* gpt-2: WS is not the ' ' prefix */

    /* A4 punctuation, PUNCT_NL */
    s("punct run", "!!!");
    s("punct mixed", "!?..");
    s("punct then LF", "!\n");
    s("punct then CRLF", "?\r\n");
    s("punct then NL run", "!?\r\n\r\n\n");
    s("punct NL punct", "?\n!");
    s("space punct", " !");
    s("tab punct", "\t!");
    s("NBSP punct", "\xC2\xA0!");
    s("punct digits", "!123");
    s("emoji run", "\xF0\x9F\x98\x80\xF0\x9F\x98\x80");

    /* A5 newline runs (WS_NL) */
    s("LF", "\n");
    s("CRLF", "\r\n");
    s("LFCR", "\n\r");
    s("NL run", "\n\n\n");
    s("WS then NL", " \n");
    s("NL then WS", "\n ");
    s("tab NL tab", "\t\n\t");
    s("WS NL WS NL", " \n \n");
    s("NL run then text", "\n\nx");
    s("WS NL WS then text", "  \n x");
    s("NBSP NL", "\xC2\xA0\n");
    s("U+3000 NL", "\xE3\x80\x80\n");
    s("WS NL WS at end", " \n ");

    /* A6 whitespace not before a non-space */
    s("two spaces", "  ");
    s("three spaces", "   ");
    s("two tabs", "\t\t");
    s("space tab", " \t");
    s("ws then letter", "  x");
    s("ws then digit", "  1");
    s("ws then punct", "  !");
    s("ws at end", "x  ");
    s("ws only", "   \t ");
    s("NBSP run", "\xC2\xA0\xC2\xA0");
    s("NBSP then letter", "\xC2\xA0" "x");
    s("U+3000 run", "\xE3\x80\x80\xE3\x80\x80");
    s("U+2003 then letter", "\xE2\x80\x83" "x");

    /* all 25 \s code points: alone, runs of 3, runs then a letter */
    {
        static const uint32_t ws25[] = {
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680,
            0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
            0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000
        };
        for (size_t i = 0; i < sizeof ws25 / sizeof ws25[0]; i++) {   /* bound: 25 */
            uint8_t buf[48];
            uint64_t n = 0;
            uint32_t cp = ws25[i];
            if (cp < 0x80u) { buf[n++] = (uint8_t)cp; }
            else if (cp < 0x800u) { buf[n++] = 0xC0u | (uint8_t)(cp >> 6); buf[n++] = 0x80u | (uint8_t)(cp & 0x3Fu); }
            else if (cp < 0x10000u) { buf[n++] = 0xE0u | (uint8_t)(cp >> 12); buf[n++] = 0x80u | (uint8_t)((cp >> 6) & 0x3Fu); buf[n++] = 0x80u | (uint8_t)(cp & 0x3Fu); }
            else { buf[n++] = 0xF0u | (uint8_t)(cp >> 18); buf[n++] = 0x80u | (uint8_t)((cp >> 12) & 0x3Fu); buf[n++] = 0x80u | (uint8_t)((cp >> 6) & 0x3Fu); buf[n++] = 0x80u | (uint8_t)(cp & 0x3Fu); }
            char label[64];
            snprintf(label, sizeof label, "ws U+%04X alone", cp);
            case_all(label, buf, n);
            for (int r = 0; r < 3; r++) { memcpy(buf + r * n, buf, (size_t)n); }
            snprintf(label, sizeof label, "ws U+%04X x3", cp);
            case_all(label, buf, 3 * n);
            buf[3 * n] = 'x';
            snprintf(label, sizeof label, "ws U+%04X x3 + x", cp);
            case_all(label, buf, 3 * n + 1);
        }
    }

    /* lookahead stops at the segment end (text[len] is the end of input) */
    s("ws at end", "x ");
    s("ws ws at end", "x  ");
    s("ws NL at end", "x \n");
    s("ws NL ws at end", "x \n ");
    s("crlf at end", "x\r\n");
    s("digits at end 1", "x1");
    s("digits at end 2", "x12");
    s("digits at end 4", "x1234");
    s("mark at end", "x\xCC\x81");
    s("punct NL at end", "x!\n");

    /* invalid utf-8: every ill-formed kind, every position, truncation */
    {
        static const uint8_t salad[] = {
            'a', 0xC3, 0xA9, 'b', 0xE0, 0x80, 0x80, 'c',     /* overlong E0 80 80 */
            0xF0, 0x9F, 0x98, 'd',                            /* truncated 4-byte */
            0x80, 'e',                                         /* lone continuation */
            0xC2, 'f',                                         /* C2 then non-continuation */
            0xED, 0xA0, 0x80, 'g',                            /* encoded surrogate */
            0xF4, 0x90, 0x80, 0x80, 'h',                       /* above U+10FFFF */
            0xC0, 0xAF, 'i',                                  /* overlong 2-byte / C0 */
            0xE0, 0x9F, 0xBF, 'j',                            /* E0 out of range */
            0xF0, 0x80, 0x80, 0x80, 'k'                        /* overlong 4-byte */
        };
        case_all("utf8 salad", salad, sizeof salad);
        for (int x = 0x80; x < 0x100; x++) {                   /* bound: 128 bad bytes */
            uint8_t buf[3] = { 'a', (uint8_t)x, 'b' };
            char label[64];
            snprintf(label, sizeof label, "bad %02X between letters", x);
            case_all(label, buf, 3);
            const uint8_t one[1] = { (uint8_t)x };
            snprintf(label, sizeof label, "bad %02X alone", x);
            case_all(label, one, 1);
        }
        for (uint64_t cut = 0; cut <= sizeof salad; cut++) {    /* bound: salad length */
            char label[64];
            snprintf(label, sizeof label, "salad cut %" PRIu64, cut);
            case_all(label, salad, cut);
        }
        {   /* invalid bytes beside marks and CR/LF, and a lone C3 at the very end */
            const uint8_t buf[] = { 0xCC, 0x81, 0xFF, '\r', 0xC3 };
            case_all("mark bad CR C3-end", buf, sizeof buf);
        }
    }

    /* CR/LF runs of every shape */
    for (int r = 1; r <= 6; r++) {                             /* bound: 6 run lengths */
        uint8_t buf[32];
        char label[64];
        for (int i = 0; i < r; i++) { buf[i] = '\r'; }
        snprintf(label, sizeof label, "CR x%d", r); case_all(label, buf, (uint64_t)r);
        for (int i = 0; i < r; i++) { buf[i] = '\n'; }
        snprintf(label, sizeof label, "LF x%d", r); case_all(label, buf, (uint64_t)r);
        for (int i = 0; i < r; i++) { buf[2 * i] = '\r'; buf[2 * i + 1] = '\n'; }
        snprintf(label, sizeof label, "CRLF x%d", r); case_all(label, buf, (uint64_t)(2 * r));
        for (int i = 0; i < r; i++) { buf[2 * i] = '\n'; buf[2 * i + 1] = '\r'; }
        snprintf(label, sizeof label, "LFCR x%d", r); case_all(label, buf, (uint64_t)(2 * r));
        for (int i = 0; i < r; i++) { buf[2 * i] = ' '; buf[2 * i + 1] = '\n'; }
        snprintf(label, sizeof label, "SPNL x%d", r); case_all(label, buf, (uint64_t)(2 * r));
        for (int i = 0; i < r; i++) { buf[2 * i] = '\n'; buf[2 * i + 1] = ' '; }
        snprintf(label, sizeof label, "NLSP x%d", r); case_all(label, buf, (uint64_t)(2 * r));
        for (int i = 0; i < r; i++) { buf[2 * i] = '\t'; buf[2 * i + 1] = '\n'; }
        snprintf(label, sizeof label, "TABNL x%d", r); case_all(label, buf, (uint64_t)(2 * r));
        buf[2 * r] = 'x';
        snprintf(label, sizeof label, "CRLF x%d + x", r); case_all(label, buf, (uint64_t)(2 * r + 1));
    }

    /* combinations the §3 notes call out */
    s("soup", "Hello,  world!\n Don't stop 123\t456'iseum \xE2\x9C\x93 done.");
    s("qwen35 marks in run", "a\xCC\x81\xCC\x82" "b");
    s("qwen35 mark in punct alt", "!\xCC\x81!");
    s("apos digit", "'1");
    s("apos punct run", "'!");
    s("apos letters", "'abc");
    s("space apos", " '");
    s("digits apos letters", "12'ab");

    /* guard-page geometry: deterministic pseudo-random text, lengths 0..255, both guards */
    {
        uint64_t seed = 0x544F4B53ull;
        static const uint8_t pool[] =
            "ab'5 !\t\r\n\xC3\xA9\xE4\xB8\xAD\xCC\x81\xC2\xA0\xF0\x9F\x98\x80\x80\xFF";
        uint8_t buf[256];
        for (uint64_t len = 0; len <= 255; len++) {            /* bound: 256 lengths */
            for (uint64_t i = 0; i < len; i++) {               /* bound: len */
                buf[i] = pool[guard_rng(&seed) % (sizeof pool - 1u)];
            }
            for (int where = 0; where < 2; where++) {           /* bound: 2 guard kinds */
                guard_buf g = { 0 };
                uint8_t *p = guard_alloc(&g, (size_t)len, where, 0);
                if (!p) { fprintf(stderr, "guard_alloc failed\n"); return 2; }
                memcpy(p, buf, (size_t)len);
                case_all("guard", p, len);
                guard_free(&g);
            }
        }
    }

    printf("test_k3: %" PRIu64 " checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
