/* tools/oracle/dump_classes.c: writes toks' class byte of every code point U+0000..U+10FFFF (0x110000 bytes,
 * src/core/classes.c with class_flags 0, the onig data of src/gen/ucd_flags.c) to stdout, for
 * tools/oracle/classes_diff.py.
 *
 *   clang -std=c17 -O2 -Isrc/core tools/oracle/dump_classes.c src/core/classes.c src/gen/ucd_flags.c \
 *       -o build/kimi/dump_classes && build/kimi/dump_classes > build/kimi/toks_classes.bin
 */
#include <stdio.h>
#include <stdlib.h>

#include "classes.h"

int main(void)
{
    uint64_t need = toks_classes_bytes(0u);
    uint8_t *buf = (uint8_t *)malloc((size_t)need);
    toks_class_tables t;
    if (buf == NULL || toks_classes_build(0u, buf, need, &t) <= 0) {
        fprintf(stderr, "dump_classes: build failed\n");
        return 1;
    }
    for (uint32_t cp = 0; cp < 0x110000u; cp++) {          /* bound: 0x110000 */
        fputc(toks_class_of(&t, cp), stdout);
    }
    free(buf);
    return 0;
}
