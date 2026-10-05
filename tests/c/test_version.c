/* test_version.c: the release macros agree with each other and with the library (docs/release.md). */
#include <stdio.h>
#include <string.h>

#include "toks.h"

#define STR_(x) #x
#define STR(x) STR_(x)

int main(void)
{
    int bad = 0;
    const char *want = STR(TOKS_VERSION_MAJOR) "." STR(TOKS_VERSION_MINOR) "." STR(TOKS_VERSION_PATCH);
    if (strcmp(TOKS_VERSION, want) != 0) { printf("TOKS_VERSION %s != %s\n", TOKS_VERSION, want); bad++; }
    if (strcmp(toks_version(), TOKS_VERSION) != 0) { printf("toks_version() %s\n", toks_version()); bad++; }
    printf("test_version: toks %s (abi %d.%d), %d failures\n", toks_version(), TOKS_ABI_MAJOR, TOKS_ABI_MINOR, bad);
    return bad != 0;
}
