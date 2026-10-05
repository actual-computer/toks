/* core.h first: it hand-declares memcpy/memset/memcmp for the freestanding core, which must precede
 * <string.h>'s fortify macros, and it pulls json.h + config.h in the order its prototypes need. */
#include "../../src/core/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
int main(void)
{
    t("{}", 1); t("[]", 1); t("{\"a\":1}", 1); t("[1,2,3]", 1);
    t("  {  \"a\" : [ 1 , { \"b\" : \"c\" } ] }  ", 1);
    t("{\"a\":{\"b\":{\"c\":[{},[],null,true,false,-0,1e3]}}}", 1);
    t("\"hello\"", 1); t("42", 1); t("-17", 1); t("true", 1); t("null", 1);
    t("[\"\\u0041\\u00e9\\ud83d\\ude00\"]", 1);
    t("[\"\\\\\\\"\\/\\b\\f\\n\\r\\t\"]", 1);
    t("{\"a\":1,\"a\":2}", 1);
    t("[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]", 1);
    t("{\"k\":[1,2],\"j\":3}", 1);
    t("{\"k\":{\"deep\":[1,{\"x\":2}]}}", 1);
    t("[{}]", 1); t("{\"a\":[]}", 1);
    t("", 0); t("   ", 0); t("{", 0); t("}", 0); t("[", 0); t("]", 0);
    t("{}}", 0); t("{} {}", 0); t("1 2", 0); t("[1 2]", 0); t("{1:2}", 0);
    t("{\"a\" 1}", 0); t("{\"a\":}", 0); t("{\"a\":1,}", 0); t("[1,]", 0);
    t("[1,2", 0); t("{\"a\":1", 0); t("{\"a\"}", 0);
    t("{'a':1}", 0); t("'x'", 0); t("tru", 0); t("nulll", 0);
    t("01", 0); t("+1", 0); t(".5", 0); t("1.", 0); t("-", 0);
    t("[\"a\\q\"]", 0);
    t("[\"a", 0);
    t("[\"\\u12\"]", 0);
    t("[\"\\ud800\"]", 0);
    t("[\"\\udc00x\"]", 0);
    t("[\x01]", 0);
    t("[\xff]", 0);
    t("{\"a\":1}]", 0);
    t("{[]:1}" , 0);
    {
        char buf[200]; int n = 0;
        for (int i = 0; i < 63; i++) buf[n++] = '[';
        for (int i = 0; i < 63; i++) buf[n++] = ']';
        buf[n] = 0;
        t(buf, 1);
        n = 0;
        for (int i = 0; i < 64; i++) buf[n++] = '[';
        for (int i = 0; i < 64; i++) buf[n++] = ']';
        buf[n] = 0;
        t(buf, 0);
    }
    if (fails == 0) { printf("json smoke: all ok\n"); return 0; }
    printf("%d failures\n", fails);
    return 1;
}
