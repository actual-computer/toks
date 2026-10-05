/*
 * test_segment.c: the driver's added-token policy (segment.c, kernels.md §4 + §7 step 2) over
 * hand-built added-token tables in K1's index format (layout.h TT_ADD_*): shufti first-byte
 * buckets, the two-byte index (longest first), the one-byte single table, entry flags.
 *
 * Every rule the policy owes hf 0.23.2's find_matches has a case:
 *   - gap / token order, the final gap, empty text, no tokens
 *   - leftmost-longest: shared prefixes, one-byte tokens against longer ones, an overlong
 *     candidate skipped inside a bucket, a bucket at the very end of the text
 *   - NONSPECIAL drops specials and resumes at the RAW m_end (no rescan inside a dropped
 *     match -- tests/model/REVIEW.md's <mask>/sk> repro, the case that pins it)
 *   - lstrip / rstrip over the regex-crate \s set (multi-byte members, and U+001C-001F,
 *     which are NOT members), the lstrip clamp to the previous split's end, the rstrip
 *     overlap (the next match inside the stripped run: no gap for it)
 *   - phases: a phase-1 cursor matches only phase-1 tokens; a phase with no tokens
 * The cursor's done state is stable (next after done keeps returning 0).
 */
/* kernels.h -> core.h redeclares memcpy/memset/memcmp (freestanding core): it must come before
 * <string.h>, whose secure _chk macros would collide with the declarations. */
#include "../../src/core/kernels.h"
#include "../../src/core/layout.h"
#include "../../src/core/compile.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* ---------------------------------------------------------------- hand-built tables */

#define MAXTOK 64

typedef struct { const char *s; uint32_t id; uint8_t flags; uint8_t phase; } add_tok;

static toks_added_entry ENTRIES[MAXTOK];
static uint8_t          POOL[2048];
static uint64_t         pool_len;
static uint8_t          SHUFTI[2][32];
static uint64_t         INDEX2[2u * 65536u + (2u << TOKS_K1_H4_BITS)];   /* index2, then h4 */
static uint32_t         SINGLE[2][256];
static uint32_t         CAND[2 * MAXTOK];
static toks_tables      T;

/* the entries, then K1's index by compile.c's own builder (toks_added_index, kernels.md §4) */
static void build(const add_tok *toks, size_t n)
{
    memset(&T, 0, sizeof T);
    memset(ENTRIES, 0, sizeof ENTRIES);
    memset(SHUFTI, 0, sizeof SHUFTI);
    memset(INDEX2, 0, sizeof INDEX2);
    memset(SINGLE, 0, sizeof SINGLE);
    pool_len = 0;

    T.magic = TOKS_TABLES_MAGIC;
    T.version = TOKS_TABLES_VERSION;
    T.add_entries = ENTRIES;
    T.add_bytes = POOL;
    T.add_n = (uint64_t)n;
    T.add_shufti = &SHUFTI[0][0];
    T.add_index = INDEX2;
    T.add_single = &SINGLE[0][0];
    T.add_cand = CAND;
    T.add_phases = 0;

    for (size_t i = 0; i < n; i++) {                        /* bound: n tokens */
        size_t len = strlen(toks[i].s);
        ENTRIES[i].off = (uint32_t)pool_len;
        ENTRIES[i].len = (uint16_t)len;
        ENTRIES[i].flags = toks[i].flags;
        ENTRIES[i].phase = toks[i].phase;
        ENTRIES[i].id = toks[i].id;
        memcpy(POOL + pool_len, toks[i].s, len);
        pool_len += len;

    }
    toks_added_index(&T, &SHUFTI[0][0], INDEX2, &SINGLE[0][0], CAND);   /* the compiler's own builder */
}

/* ---------------------------------------------------------------- the runner */

typedef struct { uint32_t kind, id; uint64_t start, end; } unit;

#define U_GAP(s, e)      { TOKS_SEG_GAP,   0u,   (s), (e) }
#define U_TOK(i, s, e)   { TOKS_SEG_TOKEN, (i),  (s), (e) }

static int failures;
static int cases;

static int run_units(const char *text, uint32_t mode, uint32_t phase, unit *out, int max)
{
    toks_seg_iter it;
    toks_seg_out u;
    int n = 0;
    toks_seg_begin(&it, &T, mode, phase, TOKS_TIER_SCALAR, (const uint8_t *)text, strlen(text));
    while (toks_seg_next(&it, &u)) {                        /* bound: <= 2 * len + 1 */
        if (n < max) { out[n].kind = u.kind; out[n].id = u.id; out[n].start = u.start; out[n].end = u.end; }
        n++;
    }
    toks_seg_out u2;
    if (toks_seg_next(&it, &u2)) {                          /* done is stable */
        fprintf(stderr, "FAIL next() returned a unit after done\n");
        failures++;
    }
    return n;
}

static void check(const char *what, const unit *got, int ng, const unit *want, int nw)
{
    cases++;
    int ok = (ng == nw);
    for (int i = 0; ok && i < nw; i++) {                    /* bound: nw */
        ok = got[i].kind == want[i].kind && got[i].id == want[i].id
          && got[i].start == want[i].start && got[i].end == want[i].end;
    }
    if (!ok) {
        failures++;
        fprintf(stderr, "FAIL %s: %d units, want %d\n", what, ng, nw);
        for (int i = 0; i < ng && i < 16; i++) {            /* bound: min(ng, 16) reported */
            fprintf(stderr, "  got  %s id=%u [%llu,%llu)\n",
                    got[i].kind == TOKS_SEG_TOKEN ? "tok" : "gap", got[i].id,
                    (unsigned long long)got[i].start, (unsigned long long)got[i].end);
        }
        for (int i = 0; i < nw && i < 16; i++) {            /* bound: min(nw, 16) reported */
            fprintf(stderr, "  want %s id=%u [%llu,%llu)\n",
                    want[i].kind == TOKS_SEG_TOKEN ? "tok" : "gap", want[i].id,
                    (unsigned long long)want[i].start, (unsigned long long)want[i].end);
        }
    }
}

#define CHECK(what, text_, mode_, phase_, ...) \
    do { \
        unit w_[] = { __VA_ARGS__ }; \
        unit g_[256]; \
        int ng_ = run_units((text_), (mode_), (phase_), g_, 256); \
        check((what), g_, ng_, w_, (int)(sizeof w_ / sizeof w_[0])); \
    } while (0)

/* ---------------------------------------------------------------- the policy */

static void test_gap_token_order(void)
{
    build((const add_tok[]){ { "foo", 10, 0, 0 }, { "bar", 11, 0, 0 } }, 2);
    CHECK("gap/token order", "a foo b bar c", TOKS_ADDED_ALL, 0,
          U_GAP(0, 2), U_TOK(10, 2, 5), U_GAP(5, 8), U_TOK(11, 8, 11), U_GAP(11, 13));
    build((const add_tok[]){ { "abc", 7, 0, 0 } }, 1);
    CHECK("whole text is the token", "abc", TOKS_ADDED_ALL, 0, U_TOK(7, 0, 3));

    build((const add_tok[]){ { "q", 9, 0, 0 } }, 1);
    CHECK("no match: one gap", "abc", TOKS_ADDED_ALL, 0, U_GAP(0, 3));

    build((const add_tok[]){ { "z", 9, 0, 0 } }, 1);
    CHECK("one-byte token at the last byte", "az", TOKS_ADDED_ALL, 0, U_GAP(0, 1), U_TOK(9, 1, 2));
}

static void test_leftmost_longest(void)
{
    build((const add_tok[]){ { "ab", 1, 0, 0 }, { "abc", 2, 0, 0 } }, 2);
    CHECK("longest wins at a shared prefix", "abc ab xabc", TOKS_ADDED_ALL, 0,
          U_TOK(2, 0, 3), U_GAP(3, 4), U_TOK(1, 4, 6), U_GAP(6, 8), U_TOK(2, 8, 11));

    build((const add_tok[]){ { "x", 1, 0, 0 }, { " xx", 2, 0, 0 } }, 2);
    CHECK("leftmost wins over longer later", "x xx", TOKS_ADDED_ALL, 0,
          U_TOK(1, 0, 1), U_TOK(2, 1, 4));

    build((const add_tok[]){ { "a", 1, 0, 0 }, { "ab", 2, 0, 0 }, { "abc", 3, 0, 0 } }, 3);
    CHECK("single vs index, bucket falls through", "ab abc a", TOKS_ADDED_ALL, 0,
          U_TOK(2, 0, 2), U_GAP(2, 3), U_TOK(3, 3, 6), U_GAP(6, 7), U_TOK(1, 7, 8));

    build((const add_tok[]){ { "abc", 3, 0, 0 }, { "ab", 2, 0, 0 } }, 2);
    CHECK("overlong candidate skipped in its bucket", "zab", TOKS_ADDED_ALL, 0,
          U_GAP(0, 1), U_TOK(2, 1, 3));

    build((const add_tok[]){ { "az", 1, 0, 0 } }, 1);
    CHECK("bucket match at the very end", "zaz", TOKS_ADDED_ALL, 0, U_GAP(0, 1), U_TOK(1, 1, 3));
}

static void test_special_nonspecial(void)
{
    /* THE pinned case: a special dropped in NONSPECIAL mode is not rescanned inside, so a
     * non-special token whose bytes occur within the dropped match is NOT found there. */
    build((const add_tok[]){ { "ab", 1, TOKS_AF_SPECIAL, 0 }, { "b", 2, 0, 0 } }, 2);
    CHECK("dropped special: no rescan inside", "ab", TOKS_ADDED_NONSPECIAL, 0, U_GAP(0, 2));
    CHECK("dropped special kept in ALL mode", "ab", TOKS_ADDED_ALL, 0, U_TOK(1, 0, 2));
    CHECK("dropped special: search resumes after it", "abxb", TOKS_ADDED_NONSPECIAL, 0,
          U_GAP(0, 3), U_TOK(2, 3, 4));

    build((const add_tok[]){ { "<p>", 1, TOKS_AF_SPECIAL, 0 }, { "</p>", 2, TOKS_AF_SPECIAL, 0 },
                             { "x", 3, 0, 0 } }, 3);
    CHECK("specials dropped around plain text", "<p>x</p>", TOKS_ADDED_NONSPECIAL, 0,
          U_GAP(0, 3), U_TOK(3, 3, 4), U_GAP(4, 8));
}

static void test_lstrip(void)
{
    build((const add_tok[]){ { "X", 5, TOKS_AF_LSTRIP, 0 } }, 1);
    CHECK("lstrip eats the leading ws run", "  X", TOKS_ADDED_ALL, 0, U_TOK(5, 0, 3));
    CHECK("lstrip: gap ends at the run start", "a  X", TOKS_ADDED_ALL, 0,
          U_GAP(0, 1), U_TOK(5, 1, 4));

    /* the clamp: the previous token ended inside the ws run, so lstrip may not cross it */
    build((const add_tok[]){ { "a ", 1, 0, 0 }, { "X", 5, TOKS_AF_LSTRIP, 0 } }, 2);
    CHECK("lstrip clamped to the previous split's end", "a  X", TOKS_ADDED_ALL, 0,
          U_TOK(1, 0, 2), U_TOK(5, 2, 4));
}

static void test_rstrip(void)
{
    build((const add_tok[]){ { "<x>", 1, TOKS_AF_RSTRIP, 0 } }, 1);
    CHECK("rstrip eats the trailing ws run", "<x>  y", TOKS_ADDED_ALL, 0, U_TOK(1, 0, 5), U_GAP(5, 6));

    /* the overlap: the next match starts inside the stripped run, and no gap is emitted
     * for it (hf keeps this exactly). The search resumed at the RAW m_end, or " y" at [1,3)
     * would be missed. */
    build((const add_tok[]){ { "x", 1, TOKS_AF_RSTRIP, 0 }, { " y", 2, 0, 0 } }, 2);
    CHECK("rstrip overlap: no gap, raw resume", "x y", TOKS_ADDED_ALL, 0,
          U_TOK(1, 0, 2), U_TOK(2, 1, 3));

    build((const add_tok[]){ { "X", 1, TOKS_AF_LSTRIP | TOKS_AF_RSTRIP, 0 } }, 1);
    CHECK("both strips", "a X b", TOKS_ADDED_ALL, 0,
          U_GAP(0, 1), U_TOK(1, 1, 4), U_GAP(4, 5));
}

static void test_ws_set(void)
{
    cases++;   /* the \s membership checks below count as one case */
    /* the 25 members, the near misses that are NOT members */
    static const uint32_t WS[25] = {
        0x0009u, 0x000Au, 0x000Bu, 0x000Cu, 0x000Du, 0x0020u, 0x0085u, 0x00A0u, 0x1680u,
        0x2000u, 0x2001u, 0x2002u, 0x2003u, 0x2004u, 0x2005u, 0x2006u, 0x2007u, 0x2008u,
        0x2009u, 0x200Au, 0x2028u, 0x2029u, 0x202Fu, 0x205Fu, 0x3000u
    };
    static const uint32_t NOT_WS[] = {
        0x0000u, 0x0008u, 0x000Eu, 0x001Cu, 0x001Du, 0x001Eu, 0x001Fu, 0x0021u, 0x007Fu,
        0x00A1u, 0x180Eu, 0x200Bu, 0x2060u, 0xFEFFu, 0x10FFFFu
    };
    for (int i = 0; i < 25; i++) {                          /* bound: 25 members */
        if (!toks_is_regex_ws(WS[i])) { fprintf(stderr, "FAIL ws: U+%04X should be \\s\n", WS[i]); failures++; }
    }
    for (size_t i = 0; i < sizeof NOT_WS / sizeof NOT_WS[0]; i++) {  /* bound: the list */
        if (toks_is_regex_ws(NOT_WS[i])) { fprintf(stderr, "FAIL ws: U+%04X should not be \\s\n", NOT_WS[i]); failures++; }
    }

    /* multi-byte ws members strip as one char; U+001C does not strip at all */
    build((const add_tok[]){ { "X", 1, TOKS_AF_LSTRIP, 0 } }, 1);
    CHECK("lstrip over U+00A0", "a\xc2\xa0" "X", TOKS_ADDED_ALL, 0, U_GAP(0, 1), U_TOK(1, 1, 4));
    CHECK("lstrip over U+2028", "\xe2\x80\xa8" "X", TOKS_ADDED_ALL, 0, U_TOK(1, 0, 4));
    CHECK("no lstrip over U+001C", "\x1c" "X", TOKS_ADDED_ALL, 0, U_GAP(0, 1), U_TOK(1, 1, 2));

    build((const add_tok[]){ { "x", 1, TOKS_AF_RSTRIP, 0 } }, 1);
    CHECK("rstrip over U+00A0 at the end", "x\xc2\xa0", TOKS_ADDED_ALL, 0, U_TOK(1, 0, 3));
    CHECK("no rstrip over U+001C at the end", "x\x1c", TOKS_ADDED_ALL, 0, U_TOK(1, 0, 1), U_GAP(1, 2));
}

static void test_phases(void)
{
    build((const add_tok[]){ { "a", 1, 0, 0 }, { "b", 2, 0, 1 } }, 2);
    CHECK("phase 1 matches only phase-1 tokens", "ab", TOKS_ADDED_ALL, 1,
          U_GAP(0, 1), U_TOK(2, 1, 2));
    CHECK("phase 0 matches only phase-0 tokens", "ab", TOKS_ADDED_ALL, 0,
          U_TOK(1, 0, 1), U_GAP(1, 2));

    build((const add_tok[]){ { "a", 1, 0, 0 } }, 1);
    CHECK("a phase with no tokens: one gap", "aa", TOKS_ADDED_ALL, 1, U_GAP(0, 2));
}

static void test_empty(void)
{
    build((const add_tok[]){ { "a", 1, 0, 0 } }, 1);
    unit g[256];
    int ng = run_units("", TOKS_ADDED_ALL, 0, g, 256);
    check("empty text: no units", g, ng, g, 0);   /* want array unused at nw == 0 */
}

int main(void)
{
    test_gap_token_order();
    test_leftmost_longest();
    test_special_nonspecial();
    test_lstrip();
    test_rstrip();
    test_ws_set();
    test_phases();
    test_empty();
    if (failures != 0) { fprintf(stderr, "test_segment: %d failures over %d cases\n", failures, cases); return 1; }
    printf("test_segment: %d cases, 0 failures\n", cases);
    return 0;
}
