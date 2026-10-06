/* test_memo_hash.c: SPEC §6's degenerate-hash build: api.c compiled into this test with every segment hashing
 * alike (one set, every key and length colliding), so only the keyed check (check.c) keeps the ids right; the
 * library's api.o is then not linked. The cases: test_memo.inc. */
#define TOKS_TEST_DEGEN(x) ((x) & 0u)               /* core.h; every piece cache one bucket too */
#define MT_DEGEN 1
#include "../../src/core/api.c"
#define MT_NAME "test_memo_hash"
#include "test_memo.inc"
