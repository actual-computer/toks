/* core.h first: it hand-declares memcpy/memset/memcmp for the freestanding core, which must precede
 * <string.h>'s fortify macros, and it pulls json.h + config.h in the order its prototypes need. */
#include "../../src/core/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* hostile-input cases beyond test_json_smoke: depth edges, root-close interactions, whitespace
 * storms at every container boundary, unicode escapes, number shapes, junk after the root. */
static int fails;

static void t(const char *src, int want_ok)
{
    toks_arena ar;
    size_t cap = strlen(src) * 8 + 4096;
    ar.base = malloc(cap);
    ar.len = cap; ar.pos = 0;
    jv *root = NULL;
    int64_t r = toks_json_parse((const uint8_t *)src, strlen(src), &ar, &root);
    int ok = (r == 0);
    if (ok != want_ok) {
        printf("FAIL %s -> r=%lld (want %s)\n", src, (long long)r, want_ok ? "ok" : "err");
        fails++;
    } else if (want_ok && root == NULL) {
        printf("FAIL %s -> ok but NULL root\n", src); fails++;
    }
    free(ar.base);
}

static void t_bin(const char *label, const char *s, size_t n, int want_ok)
{
    toks_arena ar;
    ar.base = malloc(n * 8 + 4096);
    ar.len = n * 8 + 4096; ar.pos = 0;
    jv *root = NULL;
    int64_t r = toks_json_parse((const uint8_t *)s, n, &ar, &root);
    int ok = (r == 0);
    if (ok != want_ok) {
        printf("FAIL %s (len %zu) -> r=%lld (want %s)\n", label, n, (long long)r,
               want_ok ? "ok" : "err");
        fails++;
    }
    free(ar.base);
}

int main(void)
{
    /* ---- whitespace storms at container boundaries ---------------------------------------- */
    t(" \t\r\n{ \t\r\n\"a\" \t\r\n: \t\r\n[ \t\r\n1 \t\r\n, \t\r\n{ \t\r\n\"b\" \t\r\n: \t\r\n\"c\" \t\r\n} \t\r\n] \t\r\n} \t\r\n", 1);
    t("[ 1 , [ 2 , [ 3 ] ] ]", 1);
    t("{ \"a\" : { \"b\" : { } } }", 1);
    t("[ [ ] , [ ] ]", 1);
    t("[ { } , { } ]", 1);
    t("{ \"a\" : [ ] , \"b\" : { } }", 1);
    t(" [ 1 ] [ 2 ]", 0);                /* two roots */
    t("[ 1 ] , [ 2 ]", 0);
    t("[ 1 ] ]", 0);
    t("{ \"a\" : 1 } }", 0);
    t("[ 1 ] x", 0);
    t("  {  }  ", 1);
    t("\t[ 1 , 2 , 3 ]\t", 1);

    /* ---- close-crossing: container closes while the parent expects its value ---------------- */
    t("[ { \"a\" : [ } ]", 0);           /* '}' closes an array */
    t("{ \"a\" : [ } }", 0);
    t("[ [ } ]", 0);
    t("[ 1 , }", 0);
    t("{ \"a\" : } }", 0);
    t("[ 1 ] }", 0);                     /* trailing '}' after a closed root */
    t("[ } ]", 0);
    t("{ } ]", 0);
    t("[ 1 ] { 2 }", 0);
    t("[ { } ] }", 0);

    /* ---- empty and near-empty containers everywhere ---------------------------------------- */
    t("[[],[],[]]", 1);
    t("[{},{},{}]", 1);
    t("{\"a\":[{}],\"b\":[[]]}", 1);
    t("[ {}, [] ]", 1);
    t("[ [], {} ]", 1);
    t("{\"a\":{},\"b\":[],\"c\":1}", 1);
    t("{\"a\":1,\"b\":{},\"c\":[]}", 1);
    t("{[]}", 0);
    t("{\"a\"}", 0);
    t("{\"a\":}", 0);

    /* ---- depth boundary: 63 nests ok, 64 rejected ------------------------------------------- */
    {
        char buf[400]; int n = 0;
        for (int i = 0; i < 63; i++) buf[n++] = '[';   /* root frame + 63 = 64 frames */
        buf[n++] = '1';
        for (int i = 0; i < 63; i++) buf[n++] = ']';
        buf[n] = 0;
        t(buf, 1);
        n = 0;
        for (int i = 0; i < 64; i++) buf[n++] = '[';
        buf[n++] = '1';
        for (int i = 0; i < 64; i++) buf[n++] = ']';
        buf[n] = 0;
        t(buf, 0);
        n = 0;
        for (int i = 0; i < 63; i++) {   /* {"k":{"k": ... 1 ...}} — 63 nested objects */
            buf[n++] = '{'; buf[n++] = '"'; buf[n++] = 'k'; buf[n++] = '"'; buf[n++] = ':';
        }
        buf[n++] = '1';
        for (int i = 0; i < 63; i++) buf[n++] = '}';
        buf[n] = 0;
        t(buf, 1);
        n = 0;
        for (int i = 0; i < 64; i++) {
            buf[n++] = '{'; buf[n++] = '"'; buf[n++] = 'k'; buf[n++] = '"'; buf[n++] = ':';
        }
        buf[n++] = '1';
        for (int i = 0; i < 64; i++) buf[n++] = '}';
        buf[n] = 0;
        t(buf, 0);
        n = 0;
        for (int i = 0; i < 64; i++) buf[n++] = '{';   /* bare object in object position: invalid */
        buf[n++] = '"';
        buf[n++] = 'k';
        buf[n++] = '"';
        buf[n++] = ':';
        buf[n++] = '1';
        for (int i = 0; i < 64; i++) buf[n++] = '}';
        buf[n] = 0;
        t(buf, 0);
    }

    /* ---- strings: escapes, surrogates, control bytes, raw utf-8 ------------------------------ */
    t("[\"\\u0041\\u00e9\\ud83d\\ude00\"]", 1);   /* A, é, 😀 */
    t("[\"\\uD83D\\uDE00\"]", 1);                 /* uppercase hex, pair */
    t("[\"\\ud83d\"]", 0);                        /* lone high surrogate */
    t("[\"\\ude00\\ud83d\"]", 0);                 /* reversed pair */
    t("[\"\\uDC00\"]", 0);                        /* lone low surrogate */
    t("[\"\\ud83dx\"]", 0);                       /* high surrogate + junk */
    t("[\"\\u00\"]", 0);                          /* short \u */
    t("[\"\\uZZZZ\"]", 0);
    t("[\"\\u12\"]", 0);
    t("[\"\\\"]", 0);                            /* trailing backslash */
    t("[\"\\x41\"]", 0);
    t("[\"\\u0000\"]", 1);                        /* escape of a control byte is legal */
    t("[\"\\\\n\"]", 1);
    t("[\"\\\\\\\"\"]", 1);
    t("[\"\\u0041\"]", 1);
    t("[\"\\/\"]", 1);
    t("[\"\xC3\xA9\"]", 1);                       /* raw 2-byte utf-8 */
    t("[\"\xE2\x82\xAC\"]", 1);                   /* raw 3-byte */
    t("[\"\xF0\x9F\x98\x80\"]", 1);               /* raw 4-byte */
    t("[\"\xE2\x82\"]", 0);                       /* truncated 3-byte */
    t("[\"\xC0\x80\"]", 0);                       /* overlong */
    t("[\"\xED\xA0\x80\"]", 0);                   /* utf-8 encoded surrogate */
    t("[\"\xF4\x90\x80\x80\"]", 0);               /* > U+10FFFF */
    t("[\"\xFF\"]", 0);
    /* control chars raw are rejected, incl. in keys */
    t("[\"a\tb\"]", 0);
    t("{\"k\tk\":1}", 0);
    t("[\"a\nb\"]", 0);

    /* ---- numbers ------------------------------------------------------------------------------ */
    t("[0]", 1);
    t("[-0]", 1);
    t("[0.0]", 1);
    t("[1e10]", 1);
    t("[1E+10]", 1);
    t("[-1.5e-3]", 1);
    t("[1,2,3,4,5,6,7,8,9,0]", 1);
    t("[18446744073709551616]", 1);       /* overflow: kept as float flag, not an error */
    t("[-9223372036854775808]", 1);
    t("[1.7976931348623157e309]", 1);
    t("[00]", 0);
    t("[01]", 0);
    t("[+1]", 0);
    t("[-]", 0);
    t("[.5]", 0);
    t("[5.]", 0);
    t("[1e]", 0);
    t("[1e+]", 0);
    t("[1.e5]", 0);
    t("[--1]", 0);
    t("[0x10]", 0);
    t("[NaN]", 0);
    t("[Infinity]", 0);
    t("[true,false,null]", 1);
    t("[truex]", 0);
    t("[nullx]", 0);
    t("[falsex]", 0);
    t("[TRUE]", 0);
    t("[Null]", 0);

    /* ---- members: duplicates, key types, separators ------------------------------------------- */
    t("{\"a\":1,\"a\":2}", 1);            /* last wins (map semantics) */
    t("{\"\":1}", 1);                     /* empty key */
    t("{\"a\":1,\"\":2}", 1);
    t("{1:2}", 0);
    t("{true:2}", 0);
    t("{null:2}", 0);
    t("{[1]:2}", 0);
    t("{\"a\" 1}", 0);
    t("{\"a\":1 \"b\":2}", 0);
    t("{\"a\"::1}", 0);
    t("{\"a\":,}", 0);
    t("{\"a\":1,,\"b\":2}", 0);
    t("[,1]", 0);
    t("[1,,2]", 0);
    t("[ , ]", 0);
    t("{ , }", 0);

    /* ---- junk after a complete root ----------------------------------------------------------- */
    t("1x", 0);
    t("\"a\"x", 0);
    t("nullx", 0);
    t("true x", 0);                       /* ws after root is fine; token chars are not */

    /* ---- raw binary shapes (no NUL terminator, leading/trailing junk) ------------------------- */
    {
        static const char a[] = "[1,2,3]";           /* no NUL: exact length */
        t_bin("no-nul array", a, sizeof(a) - 1, 1);
        static const char b[] = "[1,2,3]xxx";        /* junk beyond the buffer */
        t_bin("junk past len", b, 7, 1);
        static const char c[] = "[1,2,3";            /* truncated */
        t_bin("truncated array", c, sizeof(c) - 1, 0);
        static const char d[] = "\x00\x00\x00";      /* NUL bytes: format error */
        t_bin("nul body", d, 3, 0);
        static const char e[] = "[\"a\"]\x00";       /* NUL after root: not ws, junk */
        t_bin("nul after root", e, sizeof(e) - 1, 0);
        static const char f[] = " \t ";              /* only ws */
        t_bin("ws only", f, 3, 0);
        static const char g[] = "[ 1 ] ";           /* trailing ws, no NUL */
        t_bin("trailing ws no nul", g, sizeof(g) - 1, 1);
        static const char h[] = "[ \" a b \" ]";      /* spaces inside a string */
        t_bin("ws inside string", h, sizeof(h) - 1, 1);
    }

    if (fails == 0) { printf("json hostile: all ok\n"); return 0; }
    printf("%d failures\n", fails);
    return 1;
}
