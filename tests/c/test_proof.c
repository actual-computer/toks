/* tests/c/test_proof.c: regression tests for the defects a proof pass over the json reader found (J1, J2 below).
 * One block per finding, named by its id; each pins the fixed behaviour and runs the fixed code path
 * with the input's end flush against a guard page, so a reintroduced over-read faults here too. */
#include "../../src/core/core.h"
#include "guard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

#define CHECK(c, ...) do { if (!(c)) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

/* parses s[0, n) placed flush against a guard page (GUARD_END) or right after one (GUARD_START);
 * returns the status, *root the document */
static int64_t parse_guarded(const char *s, size_t n, int where, jv **root, toks_arena *ar)
{
    guard_buf g;
    uint8_t *p = guard_alloc(&g, n == 0 ? 1 : n, where, 0);
    if (p == NULL) { printf("FAIL guard_alloc\n"); fails++; return -100; }
    if (where == GUARD_END && n == 0) { p += 1; }
    memcpy(p, s, n);
    int64_t r = toks_json_parse(p, n, ar, root);
    guard_free(&g);
    return r;
}

static int64_t parse_both(const char *s, jv **root, toks_arena *ar)
{
    size_t n = strlen(s);
    jv *r0 = NULL;
    ar->pos = 0;
    int64_t a = parse_guarded(s, n, GUARD_START, &r0, ar);
    ar->pos = 0;
    int64_t b = parse_guarded(s, n, GUARD_END, root, ar);
    CHECK(a == b, "%s: start %lld end %lld", s, (long long)a, (long long)b);
    return b;
}

/* J1: a json integer whose magnitude is past 2^63 - 1 is not an integer (json.h: num_float), and no signed
 * conversion or negation overflows reading it (eva: signed downcast at parse_num). */
static void j1_num_magnitude(toks_arena *ar)
{
    static const struct { const char *s; int is_float; int64_t num; } cases[] = {
        { "9223372036854775807", 0, INT64_MAX },
        { "-9223372036854775807", 0, -INT64_MAX },
        { "9223372036854775808", 1, 0 },
        { "-9223372036854775808", 1, 0 },
        { "9999999999999999999", 1, 0 },
        { "10000000000000000000", 1, 0 },
        { "11529215046068469750", 1, 0 },
        { "18446744073709551615", 1, 0 },
        { "18446744073709551616", 1, 0 },
        { "123456789012345678901234567890", 1, 0 },
        { "1152921504606846975", 0, 1152921504606846975ll },
        { "0", 0, 0 },
        { "-0", 0, 0 },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        jv *root = NULL;
        int64_t r = parse_both(cases[i].s, &root, ar);
        CHECK(r == 0 && root != NULL && root->type == JV_NUM, "%s: r=%lld", cases[i].s, (long long)r);
        if (r != 0 || root == NULL) { continue; }
        CHECK(root->num_float == cases[i].is_float, "%s: num_float %d want %d", cases[i].s, root->num_float,
              cases[i].is_float);
        if (!cases[i].is_float) {
            CHECK(root->num == cases[i].num, "%s: num %lld", cases[i].s, (long long)root->num);
        }
    }
}

/* J2: literal and escape checks near the end of the input form no pointer past one-past-the-end (eva:
 * invalid pointer creation at the true/false/null and \u checks); each truncated form is a format error,
 * and the input's last byte sits on a guard page boundary. */
static void j2_tail_checks(toks_arena *ar)
{
    static const char *bad[] = {
        "t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul", "[tru", "[fals", "{\"a\":nul",
        "\"\\u", "\"\\u1", "\"\\u12", "\"\\u123", "\"\\u1234", "\"\\uD800\"", "\"\\uD800\\u\"", "\"\\uD800\\uDC0\"",
        "\"\\uD83D\"", "[\"\\uD83D\\\"]", "\"\\uDC00\"",
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        jv *root = NULL;
        int64_t r = parse_both(bad[i], &root, ar);
        CHECK(r == TOKS_E_FORMAT, "%s: r=%lld want TOKS_E_FORMAT", bad[i], (long long)r);
    }
    static const char *good[] = { "true", "false", "null", "[true]", "\"\\u0041\"", "\"\\uD83D\\uDE00\"" };
    for (size_t i = 0; i < sizeof good / sizeof good[0]; i++) {
        jv *root = NULL;
        int64_t r = parse_both(good[i], &root, ar);
        CHECK(r == 0, "%s: r=%lld want 0", good[i], (long long)r);
    }
}

int main(void)
{
    toks_arena ar;
    ar.len = 1u << 16;
    ar.base = malloc(ar.len);
    ar.pos = 0;
    j1_num_magnitude(&ar);
    j2_tail_checks(&ar);
    free(ar.base);
    if (fails) { printf("test_proof: %d FAILED\n", fails); return 1; }
    printf("test_proof: ok\n");
    return 0;
}
