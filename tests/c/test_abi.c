/*
 * test_abi.c: toks.h's abi written out (T11: V1, G8b, ST1, T5b). toks.h: "An abi change bumps TOKS_ABI_MAJOR"
 * (0.x's additive changes bumped TOKS_ABI_MINOR). Every public struct's size and field offsets, every constant's
 * value and every entry point's prototype are pinned here at abi 0.3, so a change to any of them fails this test
 * until the pin moves with the version: the change is then made, and reviewed, as an abi change. The prototypes
 * are pinned at compile time (each function assigned to a pointer of the written type: a changed parameter or
 * return type does not compile under -Werror), the rest at run time, each mismatch printed with its pinned value.
 * The error codes pinned here include the four the wheel tests do not (SCRATCH -6, CAP -8, ARG -10, NOMEM -11).
 * T5b: cpu.h's tier masks against toks.h's list of what each tier needs (neon + crc32; avx2, bmi1, bmi2, lzcnt,
 * popcnt + sse4.2 crc32; avx-512 f, bw, vl, vbmi + bmi2, on top of the avx2 tier's), so a feature dropped from a
 * mask fails here and not only on a machine without it.
 */
#include "toks.h"
#include "cpu.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures;
static long checks;

static const char *why = "an abi change: move the pin with TOKS_ABI_*";
#define PIN(expr, want) do { checks++; if ((long long)(expr) != (long long)(want)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s is %lld, pinned %lld (%s)\n", __FILE__, __LINE__, #expr, (long long)(expr), \
            (long long)(want), why); } } while (0)
#define PIN_FIELD(T, f, off, size) do { PIN(offsetof(T, f), off); PIN(sizeof(((T *)0)->f), size); } while (0)

/* every entry point at its written type */
static const struct {
    int64_t (*load)(toks_ctx **, const char *, const toks_load_opts *);
    int64_t (*load_mem_copy)(toks_ctx **, const void *, uint64_t, const toks_load_opts *);
    void (*unload)(toks_ctx *);
    uint64_t (*scratch_bytes)(const toks_ctx *, uint64_t, uint32_t);
    int64_t (*scratch_init)(const toks_ctx *, void *, uint64_t, uint32_t);
    int64_t (*encode)(const toks_ctx *, const void *, uint64_t, uint32_t, uint32_t *, uint64_t, void *);
    int64_t (*pieces)(const toks_ctx *, const void *, uint64_t, uint32_t, uint32_t *, uint64_t, void *);
    uint64_t (*encode_bound)(const toks_ctx *, uint64_t);
    int64_t (*split_points)(const toks_ctx *, const void *, uint64_t, uint32_t, uint32_t, uint64_t *, uint64_t, void *);
    int64_t (*decode)(const toks_ctx *, const uint32_t *, uint64_t, uint32_t, uint8_t *, uint64_t);
    const uint8_t *(*token)(const toks_ctx *, uint32_t, uint64_t *);
    int64_t (*token_to_id)(const toks_ctx *, const void *, uint64_t);
    int64_t (*id_flags)(const toks_ctx *, uint32_t);
    void (*stream_init)(const toks_ctx *, toks_stream *, uint32_t);
    uint64_t (*stream_bound)(const toks_ctx *, uint64_t);
    int64_t (*stream_push)(const toks_ctx *, toks_stream *, const uint32_t *, uint64_t, uint8_t *, uint64_t);
    int64_t (*stream_flush)(const toks_ctx *, toks_stream *, uint8_t *, uint64_t);
    int64_t (*stream_hold)(const toks_ctx *, toks_stream *, void *, uint64_t);
    int64_t (*get_info)(const toks_ctx *, toks_info *);
    const char *(*version)(void);
    int64_t (*par_create)(toks_par **, const toks_ctx *, uint32_t, uint32_t);
    void (*par_destroy)(toks_par *);
    int64_t (*par_encode_batch)(toks_par *, toks_par_item *, uint64_t, uint32_t);
    int64_t (*par_encode)(toks_par *, const void *, uint64_t, uint32_t, uint32_t *, uint64_t);
    int64_t (*par_get_info)(const toks_par *, toks_par_info *);
} API = {
    toks_load, toks_load_mem_copy, toks_unload, toks_scratch_bytes, toks_scratch_init, toks_encode, toks_pieces,
    toks_encode_bound, toks_split_points, toks_decode, toks_token, toks_token_to_id, toks_id_flags, toks_stream_init,
    toks_stream_bound, toks_stream_push, toks_stream_flush, toks_stream_hold, toks_get_info, toks_version,
    toks_par_create, toks_par_destroy, toks_par_encode_batch, toks_par_encode, toks_par_get_info,
};

int main(void)
{
    /* the version these pins describe */
    PIN(TOKS_ABI_MAJOR, 0);
    PIN(TOKS_ABI_MINOR, 3);

    /* errors (fixed for the life of the major version); -4 is not assigned */
    PIN(TOKS_E_OPEN, -1);
    PIN(TOKS_E_FORMAT, -2);
    PIN(TOKS_E_UNSUPPORTED, -3);
    PIN(TOKS_E_TIER, -5);
    PIN(TOKS_E_SCRATCH, -6);
    PIN(TOKS_E_ID, -7);
    PIN(TOKS_E_CAP, -8);
    PIN(TOKS_E_LIMIT, -9);
    PIN(TOKS_E_ARG, -10);
    PIN(TOKS_E_NOMEM, -11);

    /* limits */
    PIN(TOKS_MAX_TEXT, 536870912);
    PIN(TOKS_MAX_TOKEN_BYTES, 65535);
    PIN(TOKS_MAX_ADDED_BYTES, 255);
    PIN(TOKS_MAX_IDS, 2097151);
    PIN(TOKS_MAX_SOURCE_BYTES, 268435456);

    /* tiers */
    PIN(TOKS_TIER_AUTO, 0);
    PIN(TOKS_TIER_SCALAR, 1);
    PIN(TOKS_TIER_NEON, 2);
    PIN(TOKS_TIER_AVX2, 3);
    PIN(TOKS_TIER_AVX512, 4);

    /* scratch flags: the memo's MiB in bits 0-11 with bit 20 set, the caches' MiB in bits 12-19 */
    PIN(TOKS_SCRATCH_MEMO_SET, 0x100000);
    PIN(TOKS_SCRATCH_MEMO_MIB(0), 0x100000);
    PIN(TOKS_SCRATCH_MEMO_MIB(4095), 0x100FFF);
    PIN(TOKS_SCRATCH_MEMO_MIB(4096), 0x100000);
    PIN(TOKS_SCRATCH_CACHE_MIB(4), 0x4000);
    PIN(TOKS_SCRATCH_CACHE_MIB(128), 0x80000);
    PIN(TOKS_SCRATCH_CACHE_MIB(256), 0);

    /* encode, decode, id and info flags */
    PIN(TOKS_ADDED_ALL, 0);
    PIN(TOKS_ADDED_NONSPECIAL, 1);
    PIN(TOKS_ADDED_NONE, 2);
    PIN(TOKS_ADDED_MASK, 3);
    PIN(TOKS_NO_POSTPROCESS, 4);
    PIN(TOKS_CONTINUATION, 8);
    PIN(TOKS_SKIP_SPECIAL, 1);
    PIN(TOKS_ID_ADDED, 1);
    PIN(TOKS_ID_SPECIAL, 2);
    PIN(TOKS_ID_BYTE, 4);
    PIN(TOKS_ALGO_BPE_BYTELEVEL, 1);
    PIN(TOKS_ALGO_BPE_SPM, 2);
    PIN(TOKS_ALGO_UNIGRAM, 3);
    PIN(TOKS_ALGO_WORDPIECE, 4);
    PIN(TOKS_PATH_SCAN, 1);
    PIN(TOKS_PATH_NORMALIZE, 2);
    PIN(TOKS_PATH_MEMO, 4);
    PIN(TOKS_PAR_HAS_INFO, 1);

    /* structs: size, alignment, and each field's offset and size */
    PIN(sizeof(toks_diag), 256);
    PIN(_Alignof(toks_diag), 8);
    PIN_FIELD(toks_diag, code, 0, 8);
    PIN_FIELD(toks_diag, what, 8, 248);

    PIN(sizeof(toks_load_opts), 24);
    PIN(_Alignof(toks_load_opts), 8);
    PIN_FIELD(toks_load_opts, size, 0, 4);
    PIN_FIELD(toks_load_opts, tier, 4, 4);
    PIN_FIELD(toks_load_opts, flags, 8, 4);
    PIN_FIELD(toks_load_opts, rsv, 12, 4);
    PIN_FIELD(toks_load_opts, diag, 16, 8);

    PIN(sizeof(toks_info), 184);
    PIN(_Alignof(toks_info), 8);
    PIN_FIELD(toks_info, size, 0, 4);
    PIN_FIELD(toks_info, abi_major, 4, 4);
    PIN_FIELD(toks_info, abi_minor, 8, 4);
    PIN_FIELD(toks_info, algorithm, 12, 4);
    PIN_FIELD(toks_info, tier, 16, 4);
    PIN_FIELD(toks_info, n_ids, 20, 4);
    PIN_FIELD(toks_info, n_added, 24, 4);
    PIN_FIELD(toks_info, paths, 28, 4);
    PIN_FIELD(toks_info, cpu_features, 32, 8);
    PIN_FIELD(toks_info, max_text, 40, 8);
    PIN_FIELD(toks_info, control_isolation, 48, 4);
    PIN_FIELD(toks_info, rsv, 52, 4);
    PIN_FIELD(toks_info, source_sha256, 56, 32);
    PIN_FIELD(toks_info, image_sha256, 88, 32);
    PIN_FIELD(toks_info, name, 120, 64);

    PIN(sizeof(toks_stream), 64);                       /* the state is these 64 bytes (stream.c asserts it) */
    PIN(_Alignof(toks_stream), 8);
    PIN_FIELD(toks_stream, opaque, 0, 64);

    PIN(sizeof(toks_par_item), 40);
    PIN(_Alignof(toks_par_item), 8);
    PIN_FIELD(toks_par_item, text, 0, 8);
    PIN_FIELD(toks_par_item, len, 8, 8);
    PIN_FIELD(toks_par_item, out, 16, 8);
    PIN_FIELD(toks_par_item, cap, 24, 8);
    PIN_FIELD(toks_par_item, n, 32, 8);

    PIN(sizeof(toks_par_info), 48);
    PIN(_Alignof(toks_par_info), 8);
    PIN_FIELD(toks_par_info, size, 0, 4);
    PIN_FIELD(toks_par_info, threads, 4, 4);
    PIN_FIELD(toks_par_info, fast, 8, 4);
    PIN_FIELD(toks_par_info, last, 12, 4);
    PIN_FIELD(toks_par_info, ns_per_mib, 16, 8);
    PIN_FIELD(toks_par_info, wake_ns, 24, 8);
    PIN_FIELD(toks_par_info, join_ns, 32, 8);
    PIN_FIELD(toks_par_info, min_bytes, 40, 8);

    /* the entry points link (their types are checked where API is initialized) */
    int linked = API.load != NULL && API.load_mem_copy != NULL && API.unload != NULL && API.scratch_bytes != NULL &&
                 API.scratch_init != NULL && API.encode != NULL && API.pieces != NULL && API.encode_bound != NULL &&
                 API.split_points != NULL && API.decode != NULL && API.token != NULL && API.token_to_id != NULL &&
                 API.id_flags != NULL && API.stream_init != NULL && API.stream_bound != NULL && API.stream_push != NULL &&
                 API.stream_flush != NULL && API.stream_hold != NULL && API.get_info != NULL && API.version != NULL &&
                 API.par_create != NULL && API.par_destroy != NULL && API.par_encode_batch != NULL &&
                 API.par_encode != NULL && API.par_get_info != NULL;
    PIN(linked, 1);
    PIN(strcmp(API.version(), TOKS_VERSION), 0);

    /* T5b: what each tier needs (toks.h's tier list), as cpu.h's masks say */
    why = "toks.h's tier list: what a tier needs changed";
    PIN(TOKS_FEAT_NEON_TIER, TOKS_ARM64_NEON | TOKS_ARM64_CRC32);
    PIN(TOKS_FEAT_AVX2_TIER, TOKS_X86_AVX2 | TOKS_X86_BMI1 | TOKS_X86_BMI2 | TOKS_X86_LZCNT | TOKS_X86_POPCNT | TOKS_X86_SSE42);
    PIN(TOKS_FEAT_AVX512_TIER, TOKS_FEAT_AVX2_TIER | TOKS_X86_AVX512F | TOKS_X86_AVX512BW | TOKS_X86_AVX512VL |
                               TOKS_X86_AVX512VBMI);
    PIN(TOKS_FEAT_AVX512_TIER & TOKS_X86_BMI2, TOKS_X86_BMI2);

    printf("test_abi: %ld checks, %d failures (abi %d.%d)\n", checks, failures, TOKS_ABI_MAJOR, TOKS_ABI_MINOR);
    return failures != 0;
}
