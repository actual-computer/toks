/* test_state_hash.c: T2's forced collisions (SPEC §6, T2): api.c and k5_long.c compiled into this test with every
 * piece cache one bucket (K5's short cache, the long cache, spm's word cache: TOKS_TEST_DEGEN masks them to 0) and
 * the long cache's and the memo's hashes constant, so every probe meets every other key and only the exact compares
 * keep the ids right; the library's api.o and k5_long.o are then not linked. The cells: test_state.inc. */
#define TOKS_TEST_DEGEN(x) ((x) & 0u)
#include "../../src/core/api.c"
#include "../../src/core/k5_long.c"
#define ST_NAME "test_state_hash"
#define ST_DEGEN 1
#include "test_state.inc"
