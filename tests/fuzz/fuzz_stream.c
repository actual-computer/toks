/* tests/fuzz/fuzz_stream.c: libFuzzer harness for toks_stream_* on arbitrary id sequences over the pinned set
 * (ids.h has the input format; check.h fz_check_stream the oracles: random push partitions + flush == batch decode,
 * pushes within toks_stream_bound, TOKS_E_CAP one byte short leaving the state and the prefix exact, TOKS_E_LIMIT
 * only as documented). docs/fuzz.md. */
#include "ids.h"

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n);

int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n)
{
    fz_ids q;
    if (!fz_ids_open(&q, d, n)) { return 0; }
    fz_check_stream(q.t, q.v, q.n, q.flags, q.sel);
    free(q.v);
    return 0;
}
