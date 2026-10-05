/* tests/abi/cf_teeth.c: the teeth of tests/abi/cf_audit.py. Compiled with the library's own flags by the audit
 * itself, never linked; each function breaks one rule of SPEC §9 / §4.1 exactly once, cft_switch holds the one jump
 * table, and the audit must report exactly these and nothing else, or it fails:
 *   cft_call    R1  a call through a function pointer
 *   cft_leaf    R2  its address stored in data (cft_table)
 *   cft_libc    R3  a libc call that is not memcpy / memset / memcmp
 *   cft_alloc   R3  an allocation
 *   cft_rec     R4  recursion
 *   cft_vla     R5  a variable-length array
 *   cft_entry   R3 + R6  the teeth's after-load entry point, calling getenv
 *   cft_switch  a jump table (counted, not a §9 item) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t (*cft_fn)(uint64_t);
uint64_t cft_leaf(uint64_t x);
uint64_t cft_call(uint64_t i, uint64_t x);
uint64_t cft_switch(uint64_t k, uint64_t x);
uint64_t cft_libc(const char *s);
void *cft_alloc(uint64_t n);
uint64_t cft_rec(uint64_t n);
uint64_t cft_vla(uint64_t n);
const char *cft_entry(void);

uint64_t cft_leaf(uint64_t x) { return x * 3u + 1u; }

cft_fn cft_table[2] = { cft_leaf, cft_leaf };
volatile uint64_t cft_sink[16];

uint64_t cft_call(uint64_t i, uint64_t x) { return cft_table[i & 1u](x) + 1u; }

/* volatile stores to different places: no select chain or lookup table can stand in for the branches, and 20 dense
 * cases clear every target's minimum for a jump table (arm64 clang wants 13) */
uint64_t cft_switch(uint64_t k, uint64_t x)
{
    switch (k) {
    case 0: cft_sink[3] = x * 7u + 1u; break;
    case 1: cft_sink[9] = x ^ 0x55u; break;
    case 2: cft_sink[1] = x >> 3; break;
    case 3: cft_sink[12] = x * x; break;
    case 4: cft_sink[5] = x + 99u; break;
    case 5: cft_sink[0] = ~x; break;
    case 6: cft_sink[14] = x << 5; break;
    case 7: cft_sink[7] = x / 3u; break;
    case 8: cft_sink[2] = x % 11u; break;
    case 9: cft_sink[4] = x - 17u; break;
    case 10: cft_sink[6] = x * 13u; break;
    case 11: cft_sink[8] = x | 0x700u; break;
    case 12: cft_sink[10] = x & 0x3Fu; break;
    case 13: cft_sink[11] = x >> 9; break;
    case 14: cft_sink[13] = x * 31u + 5u; break;
    case 15: cft_sink[15] = x ^ (x >> 7); break;
    case 16: cft_sink[3] = x / 5u; break;
    case 17: cft_sink[9] = x % 7u; break;
    case 18: cft_sink[1] = x + (x << 2); break;
    case 19: cft_sink[12] = x - (x >> 1); break;
    default: break;
    }
    return x;
}

uint64_t cft_libc(const char *s) { return (uint64_t)strlen(s); }

void *cft_alloc(uint64_t n) { return malloc((size_t)n); }

uint64_t cft_rec(uint64_t n) { return n < 2u ? n : cft_rec(n - 1u) + cft_rec(n - 2u); }

uint64_t cft_vla(uint64_t n)
{
    volatile uint8_t buf[(n & 0xFFFu) + 1u];
    for (uint64_t i = 0; i <= (n & 0xFFFu); i++) { buf[i] = (uint8_t)i; }
    return buf[n & 0xFFFu];
}

const char *cft_entry(void) { return getenv("TOKS_CFT"); }
