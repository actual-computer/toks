/* SPDX-License-Identifier: BUSL-1.1 */
/*
 * toks.h: the public abi. This header is the source of truth (SPEC §4.1); the asm include and any
 * language mirror are generated from it. An abi change bumps TOKS_ABI_MAJOR.
 *
 * Rules that hold for every function (SPEC §4.1):
 *  - pointers are valid for their stated lengths and lifetimes; NULL is allowed exactly where a length or cap
 *    of 0 is passed; no alignment is required of any buffer; out must not overlap text, the context or the
 *    scratch.
 *  - for arbitrary contents of valid buffers, nothing outside SPEC §7.2's regions is read or written.
 *  - units: bytes, uint32_t ids, counts in ids.
 *  - reentrant: a context is read-only after load and may be shared by any number of threads; a scratch and a
 *    toks_stream belong to one thread at a time; no global is written; after load there is no allocation,
 *    syscall, lock, recursion or callback.
 *  - errors are stable negative codes; a failure leaves the context unchanged and the scratch valid.
 */
#ifndef TOKS_H
#define TOKS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TOKS_ABI_MAJOR 0
#define TOKS_ABI_MINOR 4

/* the release, semver (docs/release.md): tag v<TOKS_VERSION>, and the python wheel's version. The abi's own
 * compatibility is TOKS_ABI_*: a release that changes the abi bumps both. */
#define TOKS_VERSION_MAJOR 0
#define TOKS_VERSION_MINOR 3
#define TOKS_VERSION_PATCH 1
#define TOKS_VERSION       "0.3.1"

#if defined(_WIN32)
#  if defined(TOKS_BUILD_SHARED)
#    define TOKS_API __declspec(dllexport)
#  else
#    define TOKS_API
#  endif
#else
#  define TOKS_API __attribute__((visibility("default")))
#endif

/* ---- errors (fixed for the life of the major version, SPEC §4.9) ------------------------------------ */

#define TOKS_E_OPEN             (-1)   /* the file could not be opened or read */
#define TOKS_E_FORMAT           (-2)   /* not a valid tokenizer file / image (syntax, structure, checksum) */
#define TOKS_E_UNSUPPORTED      (-3)   /* a valid tokenizer using a feature toks does not support (named in diag) */
/* -4 is not assigned */
#define TOKS_E_TIER             (-5)   /* a forced tier this machine cannot run */
#define TOKS_E_SCRATCH          (-6)   /* scratch NULL, too small, not initialized, or bound to another context */
#define TOKS_E_ID               (-7)   /* an id beyond the table (decode) */
#define TOKS_E_CAP              (-8)   /* output capacity below what an atomic call needs (stream decode,
                                          toks_template) */
#define TOKS_E_LIMIT            (-9)   /* an input beyond SPEC §8.3's limits (text > 2^29 bytes, ...), or a
                                          byte-fallback run longer than a stream's hold in toks_stream_push
                                          (44 bytes, or toks_stream_hold's: grow it and push again) */
#define TOKS_E_ARG              (-10)  /* a precondition the call can check: NULL with a nonzero length, unknown
                                          flag bits, ... (addition to SPEC §4.9, see docs/design.md) */
#define TOKS_E_NOMEM            (-11)  /* load-time allocation failed (addition to SPEC §4.9) */

/* ---- limits (SPEC §8.3) ---------------------------------------------------------------------------- */

#define TOKS_MAX_TEXT          (1ull << 29)        /* bytes per encode call */
#define TOKS_MAX_TOKEN_BYTES   65535u              /* bytes of one vocabulary token (unigram pieces: 127, -3 above) */
#define TOKS_MAX_ADDED_BYTES   255u                /* bytes of one added token (plus strip runs) */
#define TOKS_MAX_IDS           ((1u << 21) - 1u)   /* every id is < TOKS_MAX_IDS */
#define TOKS_MAX_SOURCE_BYTES  (256ull << 20)      /* tokenizer file / image size */

/* ---- types -------------------------------------------------------------------------------------------- */

typedef struct toks_ctx toks_ctx;                  /* opaque; created by toks_load*, freed by toks_unload */

typedef struct toks_diag {                         /* optional load diagnostics */
    int64_t code;                                  /* the TOKS_E_* returned */
    char    what[248];                             /* NUL-terminated: the blocking feature or the reason */
} toks_diag;

/* tiers (SPEC §11). TOKS_TIER_AUTO picks the fastest the machine supports. */
#define TOKS_TIER_AUTO     0u
#define TOKS_TIER_SCALAR   1u   /* portable c (the c twin of every kernel) */
#define TOKS_TIER_NEON     2u   /* arm64: armv8.0-a + neon (+ crc32) */
#define TOKS_TIER_AVX2     3u   /* x86-64: avx2, bmi1, bmi2, lzcnt, popcnt (+ sse4.2 crc32) */
#define TOKS_TIER_AVX512   4u   /* x86-64: avx-512 f, bw, vl, vbmi + bmi2 */

typedef struct toks_load_opts {
    uint32_t   size;    /* sizeof(toks_load_opts) */
    uint32_t   tier;    /* TOKS_TIER_*; AUTO also honours the TOKS_TIER environment variable */
    uint32_t   flags;   /* 0: no load flags are defined (a nonzero value is TOKS_E_ARG) */
    uint32_t   rsv;     /* 0 */
    toks_diag *diag;    /* optional */
} toks_load_opts;

/* ---- loading (SPEC §4.2) --------------------------------------------------------------------------- */

/* path: a tokenizer.json, a tiktoken model's tiktoken.model or qwen.tiktoken (its other files beside it), or a
 * model directory holding one of them. Compiles, validates and certifies; bounded work. opts may be NULL. On
 * success *out is a new context and 0 is returned. Mistral tekken.json is not read. */
TOKS_API int64_t toks_load(toks_ctx **out, const char *path, const toks_load_opts *opts);

/* the bytes of a tokenizer file, copied into memory the context owns (a tiktoken model: toks_load its directory). */
TOKS_API int64_t toks_load_mem_copy(toks_ctx **out, const void *data, uint64_t len, const toks_load_opts *opts);

TOKS_API void toks_unload(toks_ctx *ctx);          /* NULL-safe */

/* ---- scratch (SPEC §4.3) --------------------------------------------------------------------------- */

/* scratch flags: the segment memo (SPEC §6), m MiB, m < 4096. Flags without it give the default, 4 MiB: a text or
 * conversation sent again is answered from it at GB/s (docs/usage.md "Scratch"). TOKS_SCRATCH_MEMO_MIB(0) is a budget
 * of zero, no memo: for batch jobs over text that never comes back, whose first ~2 MB per scratch a memo slows by
 * 1-14% (writing its records). Wordpiece and unigram have no memo at any m. abi, since 0.2.0: flags 0 had no memo
 * there and bit 20 (TOKS_SCRATCH_MEMO_SET) is new; a 0.2.0 binary's TOKS_SCRATCH_MEMO_MIB(m > 0) means what it did. */
#define TOKS_SCRATCH_MEMO_MIB(m)   (TOKS_SCRATCH_MEMO_SET | ((uint32_t)(m) & 0xFFFu))
#define TOKS_SCRATCH_MEMO_SET      (1u << 20)   /* the flags set the memo's size (alone: a budget of zero) */
/* scratch flags: the BPE piece caches in MiB: 0 = the default (2 MiB, short pieces only), else a power of two
 * 4..128 (anything else: TOKS_E_ARG): n / 2 MiB for short pieces and n / 2 MiB for long ones (SPEC §7.2). A
 * memory budget only: outputs are identical at every size; a scratch reused across calls (warm) gets faster. */
#define TOKS_SCRATCH_CACHE_MIB(n)  (((uint32_t)(n) & 0xFFu) << 12)

/* pure: the scratch size for texts of up to max_len bytes (the published formula, SPEC §7.2). */
TOKS_API uint64_t toks_scratch_bytes(const toks_ctx *ctx, uint64_t max_len, uint32_t flags);

/* binds scr (any alignment, `bytes` long) to ctx and clears its caches. A scratch bound to another context
 * returns TOKS_E_SCRATCH from encode calls until it is initialized again; so does any buffer of at least
 * 127 bytes that was never initialized (a call reads only its header before the binding is proven). */
TOKS_API int64_t toks_scratch_init(const toks_ctx *ctx, void *scr, uint64_t bytes, uint32_t flags);

/* ---- encode (SPEC §4.4) ---------------------------------------------------------------------------- */

/* encode flags. The mode is bits 0-1 (SPEC §3.2):
 *   ALL         special and non-special added tokens recognized (hf default)
 *   NONSPECIAL  only non-special added tokens recognized (hf encode_special_tokens=True)
 *   NONE        no added token recognized (hf pipeline with added-token extraction removed)
 * flags 0 returns exactly what hf encode(text) returns by default (post-processor applied, then the file's
 * truncation and padding: toks_info's trunc_* / pad_*). Truncation and padding are whole-document steps: a
 * TOKS_CONTINUATION call applies neither. */
#define TOKS_ADDED_ALL          0u
#define TOKS_ADDED_NONSPECIAL   1u
#define TOKS_ADDED_NONE         2u
#define TOKS_ADDED_MASK         3u
#define TOKS_NO_POSTPROCESS     4u   /* skip bos / eos / template tokens (hf add_special_tokens=False) */
#define TOKS_CONTINUATION       8u   /* the text continues a document: no start-of-input behaviour (§5.3) */
#define TOKS_NO_TRUNCATE       16u   /* the file's truncation is not applied (hf no_truncation()) */
#define TOKS_NO_PAD            32u   /* the file's padding is not applied (hf no_padding()). With both the count is
                                        the whole text's, unpadded: the ids an embedding server masks and pads itself */

/* returns n >= 0, the total number of ids; out[0 .. min(cap, n)) holds the exact prefix. out may be NULL
 * only with cap 0. Entries of out at index >= min(cap, n) and < cap may be overwritten. */
TOKS_API int64_t toks_encode(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                             uint32_t *out, uint64_t cap, void *scr);

/* the pieces the model sees, as end offsets into the caller's bytes; where a materializing normalizer (NFC)
 * changed the text, into its normalized form (SPEC §3.5). An added-token match is one piece; same capacity
 * rule as toks_encode. A view for inspection: a piece end is not a cut point (to cut text, toks_split_points;
 * docs/usage.md "Cutting text"). */
TOKS_API int64_t toks_pieces(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                             uint32_t *ends, uint64_t cap, void *scr);

/* the post-processor's single-sequence template: the ids it puts before the text's ids (*n_prefix of them) and then
 * after them, written to ids[0, count), and their type ids to type_ids (NULL: not written); returns the count, hf's
 * num_special_tokens_to_add(False) (the text's own type id is toks_info's seq_type_id). ids NULL with cap 0 sizes
 * it (the count, *n_prefix set); a cap below the count is TOKS_E_CAP and writes nothing. TOKS_E_ARG for ctx NULL or
 * ids NULL with cap > 0. n_prefix may be NULL. */
TOKS_API int64_t toks_template(const toks_ctx *ctx, uint32_t *ids, uint32_t *type_ids, uint64_t cap,
                               uint32_t *n_prefix);

/* ---- capacity ----------------------------------------------------------------------------------------- */

/* the most ids toks_encode can return for a text of len bytes, under any flags: ceil(r len) + g, with r (ids per input
 * byte) and g computed per context at load (a context that pads: at least its padded length), so an out of this many
 * ids is never short. g is the template's ids plus the ▁ a model prepends (1; 3 for unigram byte fallback with no ▁
 * piece: its 3 bytes). r by normalizer: NFC byte-level 3 (U+1D160's 4 bytes become three 4-byte chars: attained where
 * the vocab splits them into bytes); NFKC 11 (U+FDFA's 3 bytes become 33; the worst measured 3.67); a unigram
 * nmt_nfkc charsmap 6 (U+FDFA's 3 bytes become 18 chars; the worst measured 2.04), 11 under byte fallback (its 33
 * bytes); NFKD then the charsmap 66 (albert, xlnet: conservative, the worst measured 2.4); times 3 for unigram byte
 * fallback with no ▁ piece (the ▁ for a space is 3 ids); at least (1 + p) / l where an added token of l bytes can be
 * followed by a prepended ▁ of p ids (one byte: 2, "#a" 3 ids); everything else 1, and tight: len + the template's ids
 * (+ 1 for a prepended ▁), which a text of len >= 1 bytes that are one id each reaches (the empty text gets no ▁). Any
 * len has a bound, but toks_encode returns TOKS_E_LIMIT above TOKS_MAX_TEXT. Saturates at UINT64_MAX; NULL gives 0. */
TOKS_API uint64_t toks_encode_bound(const toks_ctx *ctx, uint64_t len);

/* ---- split planning (SPEC §4.5, §5) ---------------------------------------------------------------- */

/* up to min(cap, n_want - 1) cuts, strictly increasing offsets in (0, len), written to offs; returns their
 * count c. Each is the certified cut nearest its target len * i / n_want (lower on a tie) within
 * D = min(4096, len / (4 n_want)) bytes; a target without one is skipped. At a certified cut, encoding the
 * parts (no post-processing, every part after the first with TOKS_CONTINUATION) and concatenating gives the
 * whole input's ids under flags without post-processing (SPEC §5.2); flags matter because the added-token
 * mode changes the valid cuts, and a file's truncation or single-text padding (Fixed, a multiple) leaves none
 * unless the flags turn it off (TOKS_NO_TRUNCATE, TOKS_NO_PAD). n_want <= 1 returns 0; a tokenizer family without
 * certified rules returns 0 (its inputs encode serially). Reads at most n_want x (2D + W) bytes, W per tokenizer
 * (docs/split.md); scr is reserved and may be NULL. */
TOKS_API int64_t toks_split_points(const toks_ctx *ctx, const void *text, uint64_t len, uint32_t flags,
                                   uint32_t n_want, uint64_t *offs, uint64_t cap, void *scr);

/* ---- decode (SPEC §3.4, §4.6) ---------------------------------------------------------------------- */

#define TOKS_SKIP_SPECIAL   1u   /* hf skip_special_tokens=True */
#define TOKS_DECODE_RAW     2u   /* the bytes the ids spell, never repaired: where decode writes U+FFFD for bytes that
                                    are not utf-8 (a byte-level id's, a byte-fallback run's), those bytes stand instead
                                    (tiktoken's decode_bytes); every other step is decode's, so a result that is valid
                                    utf-8 is decode's. Not a stream flag. */

/* returns the total number of bytes; out[0 .. min(cap, n)) holds the exact prefix. An id beyond the
 * table returns TOKS_E_ID. */
TOKS_API int64_t toks_decode(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags,
                             uint8_t *out, uint64_t cap);

/* zero-copy raw bytes of one id, or NULL if the id has no string. */
TOKS_API const uint8_t *toks_token(const toks_ctx *ctx, uint32_t id, uint64_t *len);

/* ---- vocabulary ------------------------------------------------------------------------------------ */

/* the id whose string is the bytes s[0, len): toks' key is the bytes an id decodes to (toks_token's), NOT hf's
 * vocabulary spelling. hf's token_to_id("Ġhello") is 23748 in gpt2; here that is " hello", and "Ġhello" is
 * TOKS_E_ID. Beware the footgun: an hf spelling fed here as bytes can silently name ANOTHER valid id. gpt2's "¢"
 * (C2 A2) is hf's spelling of the byte token A2 (id 95), but as bytes it is the token hf spells "Â¢" (id 44359);
 * 56 to 148 such strings per byte-level file. A sentencepiece piece is as written ("▁hello", "<0x41>"), and an
 * added token's content finds it ("<s>", "<|endoftext|>"; hf reads its added map first), as does its normalized
 * form where hf decodes one (llama2's "▁<s>"). Where ids share a string, the bytes name the one hf's token_to_id
 * gives that text: an added token's content first (a byte-level vocab "ĠĠ" and an added "  " decode alike: "  "
 * is the added token), then the id the file writes as those very bytes, then the later one. The empty string is
 * never found (hf finds wp-specter2's empty token, id 30106: TOKS_E_ID here). TOKS_E_ID when no id has that
 * string, TOKS_E_ARG for ctx NULL or s NULL with len > 0. The index is built at load: 5.5 to 9 bytes per id. */
TOKS_API int64_t toks_token_to_id(const toks_ctx *ctx, const void *s, uint64_t len);

/* an id's flags: TOKS_ID_ADDED, TOKS_ID_SPECIAL, TOKS_ID_BYTE (0: a plain vocabulary id, or an id with no string),
 * TOKS_E_ID beyond the table, TOKS_E_ARG for ctx NULL */
#define TOKS_ID_ADDED     1u   /* an added token holds the id (hf's added_tokens_decoder has it) */
#define TOKS_ID_SPECIAL   2u   /* special if any listing of the content that holds the id is special (hf's
                                  special_tokens_set rule; hf's added_tokens_decoder keeps the last listing instead;
                                  no cached file lists one content with two flags): the token's own flag, not
                                  decode's skip rule (llama2's id 1 is special, yet TOKS_SKIP_SPECIAL keeps its
                                  "▁<s>") */
#define TOKS_ID_BYTE      4u   /* an id that stands for exactly one raw byte: a byte-fallback <0xHH> (or hf's <0x+F>)
                                  that a ByteFallback chain decodes as one byte, or a byte-level one-byte token */
TOKS_API int64_t toks_id_flags(const toks_ctx *ctx, uint32_t id);

/* the added tokens in id order, hf's get_added_tokens_decoder(): for i = 0, 1, ... the i-th id an added token holds
 * (*id), the bytes of the content listed last for that id as the file writes them (*content, *len: they live as long
 * as ctx), and its flags as the return value: TOKS_ID_ADDED, TOKS_ID_SPECIAL as that listing says (hf's
 * AddedToken.special; differs from toks_id_flags' only for a content listed both special and not), TOKS_ID_BYTE as
 * toks_id_flags, and that listing's options, the four below (segment.c matches by them; toks_id_flags never returns
 * them). TOKS_E_ARG past the last one (i == their count) or for ctx NULL; content, len and id may be NULL. A tiktoken
 * file's specials: TOKS_ID_SPECIAL where its config names them, none of the four. */
#define TOKS_ID_LSTRIP       8u   /* lstrip: a match takes the whitespace before it */
#define TOKS_ID_RSTRIP      16u   /* rstrip: and the whitespace after it */
#define TOKS_ID_SINGLE_WORD 32u   /* single_word: never matched inside a word */
#define TOKS_ID_NORMALIZED  64u   /* normalized: matched on the normalized text */
TOKS_API int64_t toks_added(const toks_ctx *ctx, uint32_t i, const void **content, uint64_t *len, uint32_t *id);

/* ---- stream decode (SPEC §3.4, §4.7) --------------------------------------------------------------- */

typedef struct toks_stream { uint64_t opaque[8]; } toks_stream;

TOKS_API void     toks_stream_init(const toks_ctx *ctx, toks_stream *st, uint32_t flags);
TOKS_API uint64_t toks_stream_bound(const toks_ctx *ctx, uint64_t n_ids);   /* worst-case bytes of one push */
/* push and flush are atomic: with cap below what the call needs they return TOKS_E_CAP and leave st as is. */
TOKS_API int64_t  toks_stream_push(const toks_ctx *ctx, toks_stream *st, const uint32_t *ids, uint64_t n,
                                   uint8_t *out, uint64_t cap);
TOKS_API int64_t  toks_stream_flush(const toks_ctx *ctx, toks_stream *st, uint8_t *out, uint64_t cap);

/* the stream's hold. hf decides a byte-fallback run (<0xHH> ids in a row, sentencepiece-style and unigram
 * chains) as a whole: its chars when its bytes are valid utf-8, else one U+FFFD per byte. So a stream holds
 * a run that is valid so far until a string or the flush ends it, in st itself up to 44 bytes; a push that
 * would hold more returns TOKS_E_LIMIT. toks_stream_hold gives st caller memory instead: the held bytes are
 * copied into hold[0, cap) (any alignment) and the run's limit is then cap. A run of c chars needs at most
 * 4c bytes (1 KiB holds any run of 256 chars). Call it after toks_stream_init (a flush keeps the hold), or
 * when a push returns TOKS_E_LIMIT for a byte-fallback run longer than the hold: st and its hold are
 * unchanged (out[0, cap) may hold a prefix of the push's output), and a hold of at least the current one's
 * size (44 for st's own) plus that push's n bytes takes it, so the recovery is to grow the hold and push
 * the same ids again. Lifetime: hold[0, cap) stays alive and untouched by the caller
 * until toks_stream_init is called on st again, st is abandoned, or another toks_stream_hold moves the
 * run out (the current hold must be alive during that call, which copies from it: no realloc); a copy
 * of st shares its hold. hold NULL / cap 0 moves the run back into st's own 44 bytes, which is how a
 * caller frees its buffer. Returns the bytes held (>= 0); TOKS_E_LIMIT when they exceed cap (44 for
 * hold NULL / cap 0; st unchanged); TOKS_E_ARG, TOKS_E_UNSUPPORTED as push. A push's out[0, cap) and the
 * hold must not overlap. ByteLevel and WordPiece streams hold at most 3
 * bytes, in st: the call returns 0 and changes nothing. With a hold of cap bytes a push writes at most
 * toks_stream_bound(ctx, n) + 3 cap bytes. */
TOKS_API int64_t  toks_stream_hold(const toks_ctx *ctx, toks_stream *st, void *hold, uint64_t cap);

/* ---- info (SPEC §4.8) ------------------------------------------------------------------------------ */

#define TOKS_ALGO_BPE_BYTELEVEL   1u   /* byte-level bpe (gpt-2 alphabet) */
#define TOKS_ALGO_BPE_SPM         2u   /* sentencepiece-style bpe (metaspace, byte fallback) */
#define TOKS_ALGO_UNIGRAM         3u
#define TOKS_ALGO_WORDPIECE       4u

typedef struct toks_info {
    uint32_t size;                  /* in: sizeof(toks_info), or abi 0.3's 184 (offsetof(toks_info, trunc_on)): then
                                       only the fields before trunc_on are written */
    uint32_t abi_major, abi_minor;
    uint32_t algorithm;             /* TOKS_ALGO_* */
    uint32_t tier;                  /* TOKS_TIER_* in use */
    uint32_t n_ids;                 /* every id is < n_ids */
    uint32_t n_added;
    uint32_t paths;                 /* bitset of TOKS_PATH_*: stages running on compiled fast paths */
    uint64_t cpu_features;          /* the platform layer's feature bits */
    uint64_t max_text;              /* TOKS_MAX_TEXT */
    uint32_t control_isolation;     /* 1 when SPEC §3.6 is certified for this tokenizer */
    uint32_t rsv;
    uint8_t  source_sha256[32];
    uint8_t  image_sha256[32];
    char     name[64];              /* NUL-terminated */
    /* abi 0.4: the file's truncation and padding as hf's Tokenizer.truncation / .padding report them (encode applies
     * them unless TOKS_NO_TRUNCATE / TOKS_NO_PAD), and the template's shape (toks_template) */
    uint32_t trunc_on;              /* 1: the file truncates (direction Right; LongestFirst and OnlyFirst alike) */
    uint32_t trunc_max;             /* max_length, at most TOKS_MAX_TEXT: the template's ids count in it */
    uint32_t trunc_stride;          /* stride: the overflow's, which encode never returns */
    uint32_t pad_on;                /* 1: the file pads. A single text is padded under Fixed or a pad_multiple only
                                       (BatchLongest pads a batch to its longest member: the caller's step) */
    uint32_t pad_fixed;             /* 1: Fixed, to pad_len; 0: BatchLongest */
    uint32_t pad_id, pad_type_id;
    uint32_t pad_len;
    uint32_t pad_multiple;          /* pad_to_multiple_of: the target rounds up to a multiple of it (0: none) */
    uint32_t pad_left;              /* 1: direction Left */
    uint32_t n_template_prefix;     /* the template's ids before the text's and after them (flags 0) */
    uint32_t n_template_suffix;
    uint32_t seq_type_id;           /* the type id of the text's ids in the template (hf Encoding.type_ids) */
    uint32_t rsv2;
} toks_info;

#define TOKS_PATH_SCAN       1u     /* pre-tokenizer on a compiled template (else the generic engine) */
#define TOKS_PATH_NORMALIZE  2u     /* normalizer on the compiled path */

TOKS_API int64_t toks_get_info(const toks_ctx *ctx, toks_info *out);

/* TOKS_VERSION of the library the program runs with (a program compiled against another toks.h can tell). */
TOKS_API const char *toks_version(void);

/* ---- toks_par: batches and one big input across a few threads (SPEC §0.3, §2.4, §4.10) -------------
 * A companion layer outside the core: a small persistent pool of workers (the calling thread is one of them)
 * sharing one read-only context, each with its own scratch, which the pool grows to the largest unit it
 * meets and keeps. It adds no semantics: every result equals the serial toks_encode call's (T2). A call uses
 * only as many participants as its work justifies, by a cost model the pool measures on its own host and
 * tokenizer (its encode time per byte, its workers' wake and join delays); a call one core finishes faster
 * (and every call under 16 KiB) runs on the caller alone and wakes no thread. Work is cut into units (whole
 * documents grouped, a big document split at toks_split_points' cuts) that the participants claim in order
 * from one queue. Idle workers spin about as long as a wake costs, then sleep; a call wakes exactly the
 * workers it uses; no thread is created per call. One call at a time per pool (concurrent calls wait for
 * each other). docs/usage.md (threads) has the policy and its numbers. */

typedef struct toks_par toks_par;                  /* opaque */

/* a pool of at most n_threads participants for ctx, the caller included (0: the fast cores the process may run
 * on; never more than the cpus it may run on); ctx must outlive it. Workers prefer the fast cores (linux: the
 * highest cpu_capacity class; apple: the caller's QoS class). scratch_flags: every participant's
 * toks_scratch_init flags (TOKS_SCRATCH_*). 0, TOKS_E_ARG or TOKS_E_NOMEM. */
TOKS_API int64_t toks_par_create(toks_par **out, const toks_ctx *ctx, uint32_t n_threads, uint32_t scratch_flags);
TOKS_API void    toks_par_destroy(toks_par *par);  /* NULL-safe; joins the threads */

typedef struct toks_par_item {
    const void *text;      /* in: the input's bytes, read in place */
    uint64_t    len;
    uint32_t   *out;       /* in: this input's ids go straight here (no two items' out ranges may overlap) */
    uint64_t    cap;
    int64_t     n;         /* out: toks_encode(ctx, text, len, flags, out, cap, scr)'s return value */
} toks_par_item;

/* every item encoded as toks_encode would, the items spread over the pool. 0 (each item's result in its n),
 * or TOKS_E_ARG for a bad pool / flags / array (then no item is touched). */
TOKS_API int64_t toks_par_encode_batch(toks_par *par, toks_par_item *items, uint64_t n_items, uint32_t flags);

/* one input across the pool: split at toks_split_points' cuts, the parts encoded in parallel and assembled
 * in out. Returns exactly what toks_encode returns, out[0, min(cap, n)) the same ids; entries at index
 * >= min(cap, n) and < cap may be overwritten. TOKS_E_NOMEM when a worker's scratch cannot grow. */
TOKS_API int64_t toks_par_encode(toks_par *par, const void *text, uint64_t len, uint32_t flags,
                                 uint32_t *out, uint64_t cap);

/* what the pool measured and decides by (docs/usage.md, threads) */
#define TOKS_PAR_HAS_INFO 1
typedef struct toks_par_info {
    uint32_t size;         /* in: sizeof(toks_par_info) */
    uint32_t threads;      /* participants at most, the caller included */
    uint32_t fast;         /* the fast cpus the process may run on (all of them where every core is alike) */
    uint32_t last;         /* participants of the last call */
    uint64_t ns_per_mib;   /* the measured encode cost on this host and tokenizer */
    uint64_t wake_ns;      /* the measured cost of one more participant that has to be woken from sleep */
    uint64_t join_ns;      /* the same for a worker still spinning after its last call */
    uint64_t min_bytes;    /* the smallest call that would take a second participant now (0: never) */
} toks_par_info;

TOKS_API int64_t toks_par_get_info(const toks_par *par, toks_par_info *out);   /* 0 or TOKS_E_ARG */

#ifdef __cplusplus
}
#endif

#endif /* TOKS_H */
