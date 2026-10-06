# c core notes: the long rationale behind src/core and src/platform

Moved verbatim from the sources when T10 became a check (`make size`, docs/design.md d18: code lines count; a long
rationale still lives in docs/ with a section pointer in the code). Each section is named after its source file and
number; the code keeps a one-line pointer `/* rationale: docs/notes/c-core.md §<file>.<n> */` where the comment
stood, and the line after the heading names the code it explains.

## src/core/alloc.c

### §alloc.c.1

Before `#include "core.h"`:

```text
alloc.c: the os-free part of the load path: the crc32c table, the byte-level alphabet and
sha256. Lives in the core (src/core), so its only libc is memcpy/memset/memcmp (SPEC §9);
the allocation wrappers moved to src/platform/mem.c (the os belongs there).
```

### §alloc.c.2

Before `int32_t toks_char_byte(uint32_t cp)`:

```text
toks_char_byte: the byte a char of gpt-2's bytes_to_unicode alphabet stands for, or -1. bytes
printable in ascii (! .. ~) and latin-1 (0xA1 .. 0xFF but U+00AD) map to themselves; the other 68
bytes, in byte order, to U+0100 .. U+0143 (so 0x00 is U+0100, 0x20 is U+0120 'Ġ', 0xAD is U+0143).
compile.c's token bytes (and so decode) go through it; tested against the python construction.
```

## src/core/api.c

### §api.c.1

Before `#include "kernels.h"`:

```text
api.c: the per-call driver of include/toks.h (kernels.md §7): scratch, encode, pieces, decode,
token, info. The c here runs once per call, per segment and per TOKS_CHUNK_PIECES round; the
per-byte and per-piece work is the kernels' (kernels.h dispatches them). Loading is load.c, split
planning split.c, stream decode stream.c.
```

### §api.c.2

Before `static void run_units(const toks_ctx *ctx, toks_scratch *h, const uint8_t *seg, uint64_t n, uint64_t nb,`:

```text
phase-1 tokens (none in mode NONE) split seg[0, n), normalized text at offset nb of the normalized stream;
every text unit goes through K3/K5. A unit's offsets are its positions in seg (SPEC §3.5): after an rstrip
token the next unit may start inside the run the token swallowed (hf resumes matching at the raw end; the
splits overlap, docs/breadth.md §2), and its offsets still index those bytes.
```

### §api.c.3

Before `static void run_cuts(const toks_ctx *ctx, toks_scratch *h, const uint8_t *t, uint64_t len, uint64_t base,`:

```text
t[0, len) at text offset base, cut where the wrapper cuts (_encode_text_piece): chunks of cut_chunk code
points; inside a chunk, before a code point that would make the run of equal py_isspace() longer than
cut_run (it starts a run of 1). A code point is an atom (kernels.md §2): an ill-formed byte counts as one,
not a space (python never sees one). Every cut piece is an offset range of the caller's buffer. at_start:
t begins at input offset 0 of a call without TOKS_CONTINUATION (run_text's).
The walk only counts until a cut, and a cut needs more than cut_run code points (a run) or more than cut_chunk
(a chunk), while a code point takes at least one byte: a t of at most min(cut_run, cut_chunk) = 25,000 bytes
holds no cut, so pos starts at len, the loop is skipped and t is the one piece the walk would have ended with
(every 4 KiB call, every prompt under 25 KB). Before, kimi paid ~1.8-2.4 ns per code point for the walk.
On longer t the walk is cheap where text is: at an ascii atom it takes the next 8 bytes at once when they are all
ascii and hold no chunk edge (run8: a SWAR py_isspace, then the bytes up to the first class change extend the run
and, after a change, the last byte's class and its trailing stretch are the new run; a run that could pass cut_run
inside the word, or cut_run < 8, falls back to one atom at a time, so every cut lands where the per-atom walk puts
it). A multi-byte atom is decoded only when its lead is C2, E1, E2 or E3: no other lead begins one of the 29
spaces (U+0085, U+00A0; U+1680; U+2000..200A, 2028, 2029, 202F, 205F; U+3000), so a CJK atom is classed by two
compares. tests/c/test_cuts.c checks the walk against the per-atom one of d11f9d1 (small limits, guard pages).
Measured and not taken (m2ultra2, whole corpus, abba): the word check ahead of every atom (cjk -1.5..-2.0%: a multi-byte
atom paid it) and, inside the ascii branch, a filter that tries the word only after an ascii byte (no gain).
```

### §api.c.4

Before `#define MEMO_MIN 256u                                   /* the shortest segment recorded */`:

```text
the segment memo (SPEC §6, kernels.md §7 "the segment memo"): a 64-byte header (core.h toks_memo_head), mb / 2048
sets of two slots (one line) and a ring of records { head, bytes, ids } written in laps; a call publishes its records
only if it returns n <= cap (§4.4); after a ring of records without a hit, only second sights are recorded; a
full lap keeps what it holds until its run of refused records reaches TOKS_MEMO_DRY (a second sight a lapping ring
would still hold weighs TOKS_MEMO_DRY / TOKS_MEMO_GHOSTS; a hit ends the run)
the default and the abi (decided 2026-10-05): flags 0 gives a 4 MiB memo (core.h toks_scr_memo_bytes, TOKS_MEMO_MIB).
In 0.2.0 flags 0 had none and TOKS_SCRATCH_MEMO_MIB(m) was m alone, so m = 0 was flags 0. Now the macro also sets
bit 20 (TOKS_SCRATCH_MEMO_SET): flags that set a size get m MiB, m = 0 none; flags with neither bit 20 nor bits 0-11
get the default. A 0.2.0 binary's TOKS_SCRATCH_MEMO_MIB(m > 0) (bits 0-11 alone) still means m MiB; its m = 0 was
compiled to 0 and gets the default, as every flags 0 does; a newer binary's bit 20 makes a 0.2.0 library return
TOKS_E_ARG (an unknown bit), never another size. A set bit rather than an off bit: SPEC §4.3 / §6's "0 = off" stays
true of the macro, no combination contradicts itself (an off bit with m > 0 would need a rule), and the masks built
from the macros (api.c, par.c) take the bit with no edit. Wordpiece and unigram have no memo path: their scratch has
no memo at any m (api.c scr_memo).
```

### §api.c.5

Before `uint64_t nb = 0u, re = 0u;`:

```text
(re, nb): the furthest unit end so far in the input and its offset in the normalized stream (SPEC
§3.5: the caller's offset wherever NFC left the text alone). Units tile the input except after an
rstrip token (segment.c, docs/breadth.md §2): hf resumes matching at the token's raw end, so the next
unit may start inside the whitespace the token swallowed. Those are raw token bytes (phase-0 tokens
are not normalized, hf), so a unit starting at u.start < re starts at nb - (re - u.start).
```

### §api.c.6

Before `int64_t toks_decode(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags,`:

```text
decode (SPEC §3.4): hf's ByteLevel decoder. Each kept id contributes its bytes (tok_bytes:
what the decoder's char -> byte map gives for its string); the concatenation goes through rust's
String::from_utf8_lossy: every maximal subpart of an ill-formed sequence (unicode §3.9) becomes
one U+FFFD, also across token boundaries and at the end. The writer is stream.c's (toks_lossy),
which the stream decoder shares, so a stream equals this over every partition by construction.
```

### §api.c.7

Before `uint64_t toks_encode_bound(const toks_ctx *ctx, uint64_t len)`:

```text
toks_encode_bound (include/toks.h "capacity"): ceil(bound_num len / bound_den) + bound_g, the context's
terms (core.h's appended bound_num, bound_den, bound_g, bound_rsv, set once by load.c after the
vocabulary index from compile.c's toks_bound_terms_of, whose comment is the proof per family). r is
kept unreduced (albert: 198 / 3) and the ceil is num q + ceil(num (len mod den) / den) with q = len /
den, so nothing wraps below the saturation test, which answers UINT64_MAX once q passes (UINT64_MAX -
g - num) / num: up to num + g early, still a bound. A context that pads: hf's target on top, pad_len
(Fixed) or the bound itself (BatchLongest, one sequence) rounded up to pad_multiple, and the larger of
that and the bound (padding never truncates). NULL gives 0. tests/c/test_bound.c checks the count
against it on every fixture and cached file under all 12 flags, and the arithmetic on constructed terms.
```

## src/core/bpe_build.c

### §bpe_build.c.1

Before `#include "bpe.h"`:

```text
bpe_build.c: the byte-level bpe tables of layout.h built from a toks_config: byte2id, the merge table and rank2id,
bytepair, vhash, premerge and words (the model's tokens, then the piece dictionary of src/gen/dict.c in the free
ways). What each holds, its sizing and why: docs/kernels.md §5 "Tables" and §6 "Static table".
```

### §bpe_build.c.2

Before `uint32_t as_rank = 1u, prev = 0u, have = 0u;`:

```text
do the surviving pairs' merged ids rise strictly in rank order? (vacuously, with none). A
config whose priorities ARE the merged ids (cfg->ids_as_rank: tiktoken, where every split of a
token shares its rank and the leftmost pair wins the tie) takes the ids whatever the order.
```

## src/core/classes.c

### §classes.c.1

Before `#include "classes.h"`:

```text
toks: per-template class-table builder (SPEC 1.4) -- implementation.

Reads the generated oniguruma property table (src/gen/ucd_flags.c, produced
by tools/gen/unicode.py) and packs the per-template class bytes (layout in
classes.h) into a two-level table with deduplicated 256-code-point blocks,
block 0 first (stage1[0] == 0), so Latin-1 lookups are a direct
stage2[c] load and the ASCII byte table is that block's first 128 bytes.

Buffer layout (buf must be 2-byte aligned; stage1 is read back as u16):
  [0, 128)              cls_ascii[128]
  [128, 128 + 0x2200)   cls_stage1[0x1100]  (u16 block index)
  [128 + 0x2200, used)  cls_stage2 blocks (u8 per code point)
used = 128 + 0x2200 + n_blocks * 256  (returned).

Determinism: blocks are deduplicated in first-seen order over the fixed
generated table, so the output is bit-for-bit deterministic for a given
class_flags.

Strict C subset (SPEC 9): C17, no allocation (caller's buffer), no
recursion, every loop below is bounded (stated per loop; the dedup scan is
at most 0x1100 * 0x1100 / 2 * 256 byte compares on the fixed input), no
libc.
```

### §classes.c.2

Before `#define TOKS_DSV3_CJK_N 2u`:

```text
deepseek v3's second split, [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff] (literal ranges, not a script
property; U+309F + 1 = U+30A0, so two ranges). Probed equal through hf's onig over every scalar value
(tests/model/dsv3_model.py; docs/templates/dsv3.md §2). Only TOKS_CLASSES_DSV3 reads it.
```

### §classes.c.3

Before `static uint8_t toks_class_dsv3(uint16_t f, uint32_t cp)`:

```text
the dsv3 byte of one code point (classes.h; docs/templates/dsv3.md §2): base L for \p{L} | \p{M}, N, NL for
CR / LF, WS for the rest of \s, P for \p{P} | \p{S}, X for everything else (flags 0: unassigned, private
use, format, the controls outside \s); TOKS_CLS_CJK on the two literal ranges; no other bit.
```

### §classes.c.4

Before `while (hr < TOKS_HAN_N && cp > toks_han_ranges[hr][1]) {`:

```text
the Han bit: \p{Han} as tiktoken resolves it (the kimi variant, docs/templates/o200k.md §6),
src/gen/han_ranges.c from tools/gen/han.py (ascending, disjoint ranges). Code points rise, so
the cursor only moves forward: bound TOKS_HAN_N steps over the whole build.
```

## src/core/classes.h

### §classes.h.1

Before `#ifndef TOKS_CLASSES_H`:

```text
toks: per-template class-table builder (SPEC 1.4).

Scanner templates classify code points by data: per tokenizer, the compiler
builds this table from src/gen/ucd_flags.* (oniguruma's properties, as hf
tokenizers 0.23.2 runs them) and the template's class_flags.

Class byte layout (the maintainer's contract; the canonical copy lives in
src/core/layout.h when it lands -- values here are identical, guarded so the
two definitions never collide):
  bits 0-2 base class: 0 P (anything that is not L, N or \s), 1 L (letter;
              \p{M} too when the template folds marks into letters),
              2 N (\p{N}), 3 WS (\s except CR and LF), 4 NL (CR or LF)
  bit 3 UPPER  (Lu, Lt, Lm, Lo, M: may be in the upper-case group)
  bit 4 LOWER  (Ll, Lm, Lo, M: may be in the lower-case group)
  bit 5 MARK  (\p{M}, whatever its base class)
  bit 6 FOLD_S (non-ASCII char that (?i) folds to 's': U+017F)
  bit 7 HAN    (\p{Han} as tiktoken resolves it; only under TOKS_CLASSES_HAN, else 0)
Tables built with TOKS_CLASSES_DSV3 use another byte (docs/templates/dsv3.md
§2): base 0 P (\p{P} | \p{S} only), 1 L (\p{L} | \p{M}), 2 N, 3 WS, 4 NL,
5 X (none of those), and bit 7 CJK (U+3040..U+30FF, U+4E00..U+9FA5);
no other bit.

Strict C subset (SPEC 9): no allocation, no recursion, every loop
bounded, libc only via memcmp/memcpy.
```

### §classes.h.2

Before `#define TOKS_CLASSES_MARKS_ARE_LETTERS 0x00000001u`:

```text
class_flags: bit 0 = \p{M} chars get base class L instead of P (templates
that fold marks into letters, e.g. the Qwen Split pattern).
bit 1 = the code points of \p{Han} as tiktoken's engine resolves it get
bit 7 (TOKS_CLS_HAN): the kimi variant of o200k (TOKS_TP_HAN). That is the
regex crate's Script=Han (regex-syntax 0.8.11, Unicode 16.0.0: 22 ranges,
99,030 code points), not Script_Extensions (U+3001 is not Han).
bit 3 = the dsv3 byte above (deepseek v3's three-split Sequence; marks are
letters there whatever bit 0 says). Not with bit 1.
```

### §classes.h.3

Before `uint64_t toks_classes_bytes(uint32_t class_flags);`:

```text
Published sizing bound for toks_classes_build's buffer: 128 (ascii) +
2 * 0x1100 (stage 1) + 0x1100 * 256 (stage 2, before dedup).  A buffer of
this size always suffices; build typically uses far less (deduplicated
blocks).  Returns 0 for a class_flags with unknown bits (build refuses).
```

### §classes.h.4

Before `int toks_classes_build(uint32_t class_flags, uint8_t *buf, uint64_t buf_len,`:

```text
Build the class tables into buf (owned by the caller; out points into it).
Returns the number of bytes used (> 0), or a negative TOKS_CLASSES_E_ code.
The result is bit-for-bit deterministic for a given class_flags: blocks are
deduplicated in first-seen order over the fixed ucd_flags table.
```

## src/core/compile.c

### §compile.c.1

Before `#include "core.h"`:

```text
compile.c: toks_config -> toks_ctx (compile.h is the contract). config.c has already refused, by name,
everything m1a cannot run exactly; what is left here is arithmetic: size one block for every table,
then fill it. The bpe tables are toks_bpe_build's (called by toks_load after this).
```

### §compile.c.2

Before `static uint8_t ascii_class(uint32_t b)`:

```text
the class byte every template gives ascii byte b (docs/kernels.md §2: the ascii members of \p{L}, \p{N} and
\s are fixed by Unicode, A-Z are upper-only and a-z lower-only in o200k's case groups, and marks, folds and
Han never touch ascii)
```

### §compile.c.3

Before `static uint32_t added_bytes(const toks_config *cfg, const toks_cfg_added *a, uint8_t *out)`:

```text
the bytes hf's decoder gives an added token alone: byte-level maps its content through the alphabet when
every char is in it, else keeps the content's own utf-8 (decoders/byte_level.rs decode_chain;
toks_token_bytes); sentencepiece-style bpe and unigram keep the content as it is: their decoder chains run on
the strings (spm_c.c, unigram.c). out may be NULL (count only); room >= a->len.
```

### §compile.c.4

Before `static uint32_t vocab_bytes(const toks_config *cfg, uint32_t id, uint8_t *out)`:

```text
the token bytes of model-vocab id (< n_vocab): byte-level maps its alphabet string back to bytes, and a
string outside the alphabet (a decode-only id, docs/breadth.md §1.4) keeps its own utf-8
(toks_token_bytes); sentencepiece-style bpe and wordpiece keep the raw string (a hole has none). out may
be NULL.
```

## src/core/compile.h

### §compile.h.1

Before `int64_t toks_compile(const struct toks_config *cfg, struct toks_ctx *ctx, toks_arena *parse_ar);`:

```text
toks_compile fills, from cfg (hf tokenizers 0.23.2 is the definition of every value):

  ctx->t   magic, version, algo = TOKS_ALGO_BPE_BYTELEVEL, flags (TOKS_TF_IGNORE_MERGES; TOKS_TF_CJK_L and
           TOKS_TF_CJK_B from toks_compile_cls_flags below),
           tmpl / tmpl_params (the pattern's template, cl100k or o200k, and TOKS_TP_* bits; TOKS_TMPL_NONE and no
           class tables when the pre-tokenizer does not split),
           the class tables (toks_classes_build with the pattern's class_flags: marks folded into letters for
             the qwen 3.5 pattern, the Han bit for an o200k pattern with TOKS_TP_HAN),
           n_ids = max(n_vocab, the largest added id + 1),
           tok_off[n_ids + 1] / tok_bytes: the bytes hf's ByteLevel decoder gives each id alone
             (toks_token_bytes): a model vocab string or an added-only content mapped back through the
             byte-level alphabet when every char is in it, else its own utf-8 (hf's fallback; such a
             vocab string is a decode-only id the bpe tables leave out); an id no token covers an empty
             range; an id with added tokens their string, never the vocab's (spm_bpe.md §8.1: the last
             listed, or the last with a stored form),
           the added-token index (layout.h, docs/kernels.md §4): add_entries grouped by phase (phase 0
             = normalized:false, matched on the raw text, then phase 1 = normalized:true), file order
             inside a phase; add_bytes the contents; add_shufti[p] lo/hi nibble tables over the first
             bytes (bucket bit = 1 << (hi nibble & 7)); add_index[p][b0 | b1 << 8] = start | count << 32
             over add_cand, the bucket's entries longest first (ties in entry order); add_single[p][b] =
             entry + 1 of the one-byte token b; add_n, add_phases; each entry's flags carry special,
             normalized, lstrip, rstrip and single_word (segment.c applies them). No surviving added
             token: every add_* pointer NULL, add_n = add_phases = 0.
  ctx->pp_ids          the single template's ids: the prefix (before $A), then the suffix;
                       n_pp_prefix / n_pp_suffix their counts.
  ctx->special_ids     n_ids bits (little-endian bit order, whole 64-bit words): set = an added id whose
                       string equals the content of an added token with special:true (hf decode's
                       skip_special_tokens, by string: spm_bpe.md §8.1); the string is tok_bytes for
                       spm, unigram and wordpiece, the content for byte-level (config.c accepts no
                       normalized token its NFC / NFKC changes).
  ctx->dec_byte_level  1 (config.c refuses every other decoder).
  ctx->cut_chunk / cut_run  cfg's: a tiktoken wrapper's cuts (0 for tokenizer.json; the driver walks them).
  ctx->mem_tables / mem_tables_len: one block (toks_plat_arena: 2 MiB-aligned, huge-page advised
                       before the first touch) holding all of the above, sized before it is filled;
                       owned by ctx on success (toks_plat_arena_free), freed here on failure.

The bpe fields (byte2id, bytepair, merge_*, rank2id, vhash, words, n_merges) stay 0: the driver's
toks_load calls toks_bpe_build after this. parse_ar is not used (cfg's memory stays valid).

Returns 0, TOKS_E_UNSUPPORTED (class tables that break docs/kernels.md §2's ascii precondition -- classes.c
never builds those), TOKS_E_LIMIT (token bytes beyond 32-bit offsets) or TOKS_E_NOMEM (config.c has refused
everything else).
```

### §compile.h.2

Before `int64_t toks_compile_cls_flags(const toks_tables *t);`:

```text
the class-table facts K3 tiers rely on (docs/kernels.md §2), from t's cls_ascii / cls_stage1 / cls_stage2:
-1 when cls_ascii breaks the precondition (any ascii class byte other than the fixed one); else TOKS_TF_CJK_L
when every code point in U+4E00..U+A3FF and U+AC00..U+D6FF has base L and no TOKS_C_FOLD_S, plus TOKS_TF_CJK_B
when each has the class byte L | UPPER | LOWER exactly; dsv3 tables (t->tmpl): their own ascii bytes (letters
L, the controls outside \s X), and TOKS_TF_CJK_D when U+4E00..U+9EFF are L | CJK. toks_compile ORs it into
t->flags; test harnesses that build toks_tables by hand call it the same way.
```

### §compile.h.3

Before `const char *toks_tmpl_invalid(const toks_tables *t);`:

```text
the template invariants the scan kernels rely on (SPEC §8.3, split parameters), checked at load on every
table set: toks_load after toks_compile, and the .toks image loader on every image when it lands (images are
untrusted, SPEC §8). NULL when they hold, else the broken one (a static string). Every parameter
combination either has the meaning its template's document gives it or is refused here:
  TOKS_TMPL_NONE    tmpl_params 0 (no class tables needed)
  TOKS_TMPL_CL100K  bits within 0x7F, TOKS_TP_CONTR_MASK not 0x03 (kernels.md §3)
  TOKS_TMPL_O200K   bits within CONTR_CI | DIGITS_1 | HAN | NO_SLASH: contractions none or (?i), digits
                    1..3 or 1 (docs/templates/o200k.md §1, §6); marks keep base P (U+0301 probed)
  any other tmpl    refused
and, with a template: class tables present, every stage-1 block index below cls_nblocks, class bit 7
(TOKS_C_HAN) on no ascii byte, set somewhere (U+4E00 probed) iff tmpl_params has TOKS_TP_HAN.
Bounded: 0x1100 + 128 + cls_nblocks * 256 reads.
```

## src/core/config.c

### §config.c.1

Before `#include "core.h"`:

```text
config.c: tokenizer.json -> toks_config, the one reader of every algorithm (byte-level bpe, sentencepiece-style
bpe, wordpiece, unigram; docs/design.md d18). hf tokenizers 0.23.2 is the definition: a file hf refuses is
TOKS_E_FORMAT, a feature toks cannot run exactly TOKS_E_UNSUPPORTED with the feature named in err->what (a
static string). Shared parts first (string index, vocab, merges, components, added tokens, post-processor,
truncation / padding), then each algorithm's checks, then the entry point. All output is the parse arena's.
```

### §config.c.2

Before `static int64_t read_merges(const jv *model, toks_arena *ar, const sidx *vx, const uint32_t *lro[3], uint32_t *`:

```text
merges: "left right" strings (exactly one space, hf's legacy format; "#version" lines dropped: bpe/model.rs
convert_merges_to_hashmap) or [left, right] pairs; left, right and left||right must all be vocab tokens
(hf: MergeTokenOutOfVocabulary). lro[0..2]: the left, right and merged ids by rank.
```

### §config.c.3

Before `static int64_t read_bpe(const jv *model, uint32_t *flags, toks_err *err)`:

```text
hf BPE fields (models/bpe/serialization.rs) into TOKS_SPM_* flags. dropout: hf tokenizes without it when None
or == 0.0 (bpe/model.rs tokenize takes the cached path; merge_all never skips a merge), so an exact zero is
null's spelling; any other value is refused.
```

### §config.c.4

Before `static int64_t read_added(const jv *root, toks_arena *ar, toks_config *cfg, const sidx *vx, toks_err *err)`:

```text
hf AddedVocabulary::add_tokens over the file's list, in order (docs/breadth.md §2): an empty content is ignored;
the id is the model vocab id of the content (vx, or the spm index), else the next id counting up from
vocab.len() (the file's id field is required but ignored). Every flag is a required bool. A repeated content
keeps its first id and takes the last entry's flags, specialness sticking. One refusal: in a phase that has an
rstrip token, an lstrip-only token whose content is all \s can sit inside a run the rstrip token swallowed, and
hf 0.23.2 then panics ("AddedVocabulary bad split").
```

### §config.c.5

Before `enum { PP_NONE = 0, PP_CLS_SEP, PP_BYTELEVEL, PP_TEMPLATE, PP_SEQUENCE };`:

```text
hf reads a post-processor through an untagged enum (processors/mod.rs: Roberta, Bert, ByteLevel, Template,
Sequence, in that order; only ByteLevel's and Sequence's deserializers look at "type"), so the kind is the
object's shape, in hf's order (docs/breadth.md §3): sep + cls pairs (Roberta / Bert: cls $A sep), ByteLevel (no
ids), Template, Sequence (top level only).
```

### §config.c.6

Before `static int64_t bl_pretok(const jv *root, toks_config *cfg, toks_err *err)`:

```text
accepted (docs/breadth.md §1.1; the byte-level mapping must be the LAST step): ByteLevel(use_regex=true) -> the
gpt-2 pattern; ByteLevel(use_regex=false) -> no split; Sequence[ByteLevel]; Sequence[Split(known Regex),
ByteLevel(use_regex=false)]; Sequence[Split x3 = TOKS_DSV3_SPLITS, ByteLevel(use_regex=false)] -> TOKS_TMPL_DSV3.
```

### §config.c.7

Before `static int64_t bl_alphabet(const jv *model, toks_config *cfg, toks_err *err)`:

```text
the vocab's byte-level side: strings outside the alphabet are decode-only ids (hf's BPE never forms them,
docs/breadth.md §1.4); a byte whose char is missing is dropped inside the model as hf drops it (merge_word),
exact only with unk_token null, byte_fallback off and ignore_merges off (docs/breadth.md §4).
```

### §config.c.8

Before `static int64_t parse_spm(const jv *root, const jv *model, toks_arena *ar, toks_config *cfg, toks_err *err)`:

```text
the text model, the decoder and the model (spm.h); added tokens and the post-processor as for every algorithm.
Normalized added tokens under a normalizer: hf matches them on the normalized text, which toks never writes
(docs/algorithms/spm_bpe.md §6.5); §config.c.9 says which it matches on the raw text instead.
```

### §config.c.9

Before `uint32_t nn = sc->text.n_norm, sh = ...` in parse_spm:

```text
hf matches a normalized:true added token on each gap's normalized form, by the content's own normalized form
(added_vocabulary.rs: the normalized trie, leftmost-longest). Under the llama 2023 files' [Prepend ▁, Replace
' ' -> ▁] (either order) that form is "▁" + content, the gap's is "▁" + gap with every ' ' as ▁: for a content
holding no ' ' and no ▁, a match in raw terms is the content at the gap start (the Prepend's ▁) or right after a
raw ' ' or ▁, which the token absorbs (segment.c §5, TOKS_AF_PFX). Under Replace ' ' -> ▁ alone the forms are the
raw ones (no flag). Anything else (another normalizer, a content with ' ' or ▁, lstrip / rstrip / single_word, a
Metaspace pre-tokenizer) keeps the refusal.
```
## src/core/config.h

### §config.h.1

Before `#ifndef TOKS_CONFIG_H`:

```text
config.h: the typed tokenizer.json configuration (m1a: byte-level bpe, cl100k template).
hf tokenizers 0.23.2 is the definition of every field; config.c refuses (TOKS_E_UNSUPPORTED,
the feature named) whatever m1a cannot run exactly.
```

### §config.h.2

Before `extern const char *const  TOKS_DSV3_SPLITS[3];`:

```text
deepseek v3 / r1 / v3.1 / v3.2 / v4 / v4.1 (docs/templates/dsv3.md §1): a Sequence of these three Splits
(each Isolated, either invert; exact strings, in order) then ByteLevel(add_prefix_space false, use_regex
false) runs on TOKS_TMPL_DSV3 (TOKS_PATTERN_DSV3: its last Split's string, its class tables).
```

### §config.h.3

Before `typedef struct toks_cfg_added {`:

```text
one surviving added token (hf AddedVocabulary::add_tokens): empty contents are dropped, the id is
hf's (the model vocab id of the content, else the next id from n_vocab up, in file order); the
file's own id field is ignored exactly as hf ignores it. Contents are distinct.
```

### §config.h.4

Before `const uint8_t *const *vocab;    /* [n_vocab] pointers into parse memory */`:

```text
model vocab: token strings by id, dense 0 .. n_vocab-1, distinct, except that an id without a token
(a tiktoken rank gap) is an empty string. A string of byte-level alphabet chars stands for the bytes
they map to; any other string (n_vocab_raw of them) is decode-only: hf's BPE never forms it, its
bytes are its own utf-8 (docs/breadth.md §1.4).
```

### §config.h.5

Before `uint32_t cut_chunk;`:

```text
a tiktoken wrapper's cuts (docs/models/kimi.md §2.3; the tiktoken reader sets them, 0 elsewhere): the
text is cut into chunks of cut_chunk code points, and inside a chunk before a code point that would make
a run of python-str.isspace()-equal code points longer than cut_run. Then the added tokens' phases mean
the wrapper's two matchers: phase 0 = transformers' trie (mode ALL only, on the whole text; each gap is
cut), phase 1 = tiktoken's specials (modes ALL and NONSPECIAL, inside each cut piece); NONE cuts and
matches nothing.
```

### §config.h.6

Before `uint32_t                      algo;`:

```text
the algorithm (include/toks.h TOKS_ALGO_*). TOKS_ALGO_BPE_SPM: spm holds the model, the text model
and the decoder (spm.h); vocab / vocab_len / n_vocab are its id range (holes: NULL, length 0),
n_strings is hf's vocab.len() (where added-only ids start). Byte-level: n_strings = n_vocab.
```

### §config.h.7

Before `int64_t toks_config_parse(const uint8_t *data, uint64_t len, toks_arena *ar, toks_config *cfg,`:

```text
parses a tokenizer.json (data[0,len)) into cfg using ar for every string and array. returns 0 or
TOKS_E_FORMAT (hf would refuse the file too), TOKS_E_UNSUPPORTED (a feature m1a does not run),
TOKS_E_LIMIT (SPEC §8.3) or TOKS_E_NOMEM; err->what names the reason (a static string).
```

### §config.h.8

Before `static inline uint64_t toks_config_arena_bound(uint64_t len) { return 32u * len + 65536u + toks_gen_max_bytes(); }`:

```text
a parse arena of this many bytes always suffices for toks_config_parse over len bytes: json.c spends at
most 20 bytes per source byte (a 40-byte node per value, values >= 2 bytes apart, strings decoded in
place of their quotes), config.c at most 5 more (vocab: pointer, length and two index slots per entry of
>= 6 source bytes; merges: 3 ids per >= 6 bytes), plus small fixed parts (the 64 KiB). Measured on the pinned
files: gpt2 6.4x, llama3 3.3x. The generic engine's program is not proportional to the source (a 7-byte
quantifier can give 500 instructions): its largest size (gen.c's limits, 129,304 B) is added whole. Its
compiler's work (~130 KiB, also fixed) is a block of its own, freed before toks_gen_compile returns: in the
arena it ran small files out of memory (fuzz finding 3, tests/fuzz/regress/expect.txt).
```

### §config.h.9

Before `int64_t toks_alpha_bytes(const uint8_t *s, uint32_t n, uint8_t *out);`:

```text
the raw bytes of a byte-level alphabet string s[0, n) (each char -> its byte, toks_char_byte), written
to out when out != NULL (room >= n). returns their count, or -1 when a char is not in the alphabet
(or s is not utf-8). config.c counts the vocab strings that fail (decode-only ids); compile.c fills
tok_bytes with toks_token_bytes.
```

### §config.h.10

Before `uint32_t toks_token_bytes(const uint8_t *s, uint32_t n, uint8_t *out);`:

```text
the bytes hf's ByteLevel decoder gives one token string s[0, n) (decoders/byte_level.rs decode_chain):
its alphabet image when every char is an alphabet char, else its own utf-8. out may be NULL (count
only); room >= n.
```

## src/core/core.h

### §core.h.1

Before `#ifndef TOKS_CORE_H`:

```text
core.h: internal shared decls for the c core (not public). Includes the public abi and the
kernel layout contract. Everything here is strict-c (SPEC §9): no libc in the core beyond
memcpy/memset/memcmp (declared by hand for the freestanding compile checks; the os lives in
src/platform and carries the toks_plat_ prefix).
```

### §core.h.2

Before `void *(memcpy)(void *dst, const void *src, size_t n);`:

```text
libc surface of the core: exactly these three (SPEC §9). Declared by hand so the core
compiles freestanding for the cross-target checks; clang's builtins still apply. The
parentheses keep a fortify macro (<string.h> included first) from expanding the name.
```

### §core.h.3

Before `static inline uint32_t toks_ld32(const void *p)`:

```text
caller arrays (out, ids, ends): SPEC §4.1 requires no alignment of any buffer, so the core reads and
writes them only through these. A 4-byte memcpy is one plain load or store on arm64 and x86-64, and no
typed access goes through a pointer the caller may have misaligned (C17 6.3.2.3p7; UBSan's alignment
check). The library's own arrays (tables, scratch regions, frames) stay typed.
```

### §core.h.4

Before `uint8_t *toks_plat_arena(uint64_t n);`:

```text
the big table arena: one private mapping, 2 MiB-aligned, madvise(MADV_HUGEPAGE)'d BEFORE its
first touch (kernels.md §7; the advice must precede the zeroing memset on zen). freed with
toks_plat_arena_free(p, n).
```

### §core.h.5

Before `int64_t toks_plat_getenv(const char *name, char *buf, uint64_t cap);`:

```text
the environment variable name, load-time only (the TOKS_TIER override), copied NUL-terminated into
buf[0, cap): its length; -1 when unset (or empty), -2 when longer than cap - 1. Windows reads the
process environment block (GetEnvironmentVariableA), not a c runtime's copy of it: a toks.dll with its
own c runtime then still sees a variable its host set after loading it.
```

### §core.h.6

Before `struct toks_ctx {`:

```text
compile.c fills t (every table but bpe's), pp_ids, n_pp_*, dec_byte_level, special_ids and
mem_tables; toks_bpe_build adds bpe's tables (mem_bpe); load.c the rest. toks_unload frees the two
blocks and the struct. later fields are appended at the end, commented.
```

### §core.h.7

Before `static inline uint64_t toks_ctx_identity(const uint8_t sha256[32])`:

```text
the identity a scratch binds to (kernels.md §7): a function of the compiled tables' content, so
a cache filled under one tokenizer is never read under another, even when a later context
reuses the freed one's addresses. the tables are a pure function of the source file and the
table format, hence: its sha-256 and TOKS_TABLES_VERSION.
```

### §core.h.8

Before `#define TOKS_SCRATCH_MAGIC   0x31524353534B4F54ull                        /* "TOKSSCR1" */`:

```text
scratch (api.c): one caller buffer of any alignment, its regions 64-aligned from scr rounded up to 64 in
this order -- header (toks_scratch: line 0 the binding, the only line read before it is proven; line 1 K5's
counters and the caches' epoch / generation), ends, the piece caches, the segment memo, work, bounce, NFC's
norm -- each listed with its size in kernels.md §7, whose formula toks_scratch_size publishes (toks_scratch_bytes).
```

### §core.h.9

Before `#define TOKS_WP_MAT_CAP      24704u`:

```text
WordPiece contexts add TOKS_WP_SCR_EXTRA bytes to the work region (ctx->scr_extra; 0 for every other algorithm):
pieces[TOKS_CHUNK_PIECES] (16 B each), the scan's copy buffer (TOKS_WP_MAT_CAP >= toks_wp_mat_min at
max_input_chars_per_word <= 1024, plus 16 KiB so accented text rarely ends a round early) and alignment. The
normalized gap (3 bytes per text byte) and the normalizer's sort area (4) fit the 32 bytes per byte of
toks_scr_work (wp_api.c).
```

### §core.h.10

Before `#define TOKS_SEG_TOKEN   1u`:

```text
---- the added-token cursor (segment.c; kernels.md §4 + §7 step 2) -----------------------------------
begin, then next until it returns 0. Units come out in order (gaps of text bytes and matched
token ids); the caller consumes them one at a time, so nothing is sized by the text.
```

### §core.h.11

Before `typedef struct toks_lossy {`:

```text
---- decode: the lossy utf-8 writer (stream.c), shared by toks_decode and the stream --------------------
hf's ByteLevel decoder ends in rust's String::from_utf8_lossy (one U+FFFD per maximal subpart,
unicode §3.9). The writer runs it incrementally, straight into out: valid utf-8 is copied, and a
well-formed proper prefix at the end of the input so far (pend[0, np), np <= 3) is held until
the bytes that complete or break it arrive.
```

### §core.h.12

Before `int64_t  toks_dec_build(struct toks_ctx *c);`:

```text
load (load.c, after compile): ctx->dec_max = max over ids of the byte length of toks_decode of that id
alone; ctx->dec_slot / dec_len = decode's fast path (stream.c): for an id whose bytes are well-formed
utf-8 on their own, its length when it is <= 16 bytes, its bytes then zero-padded in its 16-byte slot;
0xFE when longer and every 16-byte load covering it stays inside tok_bytes; else 0xFF. 17 bytes per
id (GLM 5.3: 2.6 MB). One pass; TOKS_E_NOMEM.
```

### §core.h.13

Before `struct toks_config;`:

```text
config + compile (their own headers). core.h must not depend on their include order:
json.h and config.h both include core.h first, so the guards make the includes below a
no-op there and the struct is only forward-declared at this point -- which is why every
prototype in this header spells it `struct toks_config`.
```

## src/core/json.c

### §json.c.1

Before `#include "core.h"`:

```text
json.c: strict, bounded rfc 8259 reader.
 - no recursion: an explicit stack of TOKS_JSON_DEPTH frames (document nesting < 64; deeper
   is hostile and rejected with TOKS_E_FORMAT).
 - exactly one root value: the root frame (st[0], obj == NULL) accepts one value and never
   closes; after it only whitespace may remain ("{} {}" / "1 2" are format errors).
 - every value is linked into its parent frame at the moment it is created, and a container
   that opens gets its own frame (phase machine below). A member's value is linked to the
   member node by that same rule, so closing brackets never guess their target: this is the
   parent-frame / one-root discipline the earlier draft's `have_root` and `pf` locals were
   meant to enforce, now structural instead of checked after the fact.
 - one pass over [data, data + len): every iteration consumes at least one byte; every inner
   loop is bounded and says so.
 - all nodes and decoded strings live in the caller's arena (sized before the parse starts).
 - numbers: integers kept in num; a fraction/exponent (or magnitude past 2^63 - 1) sets
   num_float and the config layer rejects the value wherever an integer is required.
 - strings: full escape handling incl. surrogate pairs; control chars < 0x20 rejected;
   valid utf-8 required (beyond rfc 8259, needed for our byte strings).
```

### §json.c.2

Before `typedef struct jframe {`:

```text
one open container. st[0] is the document root: obj == NULL, mem == NULL; it takes exactly
one value and never closes. Every other frame owns its container in obj; last is the newest
child (array element or object member), and mem the member whose value is expected next.
```

### §json.c.3

Before `static int parse_string(const uint8_t *p, const uint8_t *end, toks_arena *ar,`:

```text
parses the string at p (at the opening quote). JS_OK: decoded bytes (arena-allocated) in
*s / *s_len, position after the closing quote in *next. pass 1 finds the closing quote and
validates escape structure; pass 2 decodes, checking the \u digits and the utf-8 of the
result. decoded length <= raw length always (every escape is longer than its output), so the
raw length sizes the buffer.
```

## src/core/k1_c.c

### §k1_c.c.1

Before `#include "kernels.h"`:

```text
k1_c.c: the c twin of K1 (added-token find), kernels.md §4.

Up to cap leftmost-longest matches of the phase's added tokens, the first in text[pos, len),
each next one from the previous one's end: the smallest start with some token of the phase
occurring there (entirely inside [0, len)), the longest such token. Index: shufti nibble
tables over the first byte (false positives allowed), h4[toks_k1_h4(first four bytes) << 1 |
phase] (after index2) = the tokens of >= 4 bytes (longest first) whose first four bytes hash there,
add_index[b0 | b1<<8] = the tokens of >= 2 bytes (longest first) sharing those first two
bytes (K1 walks only the 2- and 3-byte tail, from the end), add_single[b] = a one-byte token
(entry+1). Verification: exact compare (kernels.md §4).
```

## src/core/k3_c.c

### §k3_c.c.1

Before `#include "kernels.h"`:

```text
k3_c.c: the c twin of K3 (scan), template cl100k -- kernels.md §2-3: atoms and classes, the rules A1-A7 in
the template's alternation order parameterised by t->tmpl_params, the output contract (ends, n, pos, resume).

One forward pass, O(len): each piece consumes at least one atom, run scans (A2/A3/A4/A5) walk forward only,
and a whitespace run yields at most an A5 piece plus one A6 piece (A6 consumes all but the last atom of what
remains). Text is read only inside [text, text + len) (toks_utf8_len is given the exact remainder). No
allocation, no state beyond a's fields.
```

### §k3_c.c.2

Before `uint32_t cp1 = 0, l1 = 0;`:

```text
the second atom, decoded only when a rule consults it: A1 (c is '), the letter-run
prefix (A2), the gpt-2 digit and punctuation space prefixes (A3/A4). When the prefix
rule holds, this decode always happened: its condition implies one of the three below.
```

## src/core/k3_dsv3_c.c

### §k3_dsv3_c.c.1

Before `#include "kernels.h"`:

```text
k3_dsv3_c.c: the c twin of K3 (scan), template dsv3 -- docs/templates/dsv3.md §2-3 (the rules D0-D7, tried in
order; the class byte of a TOKS_CLASSES_DSV3 table is a kind: base P, L, N, WS, NL or X | TOKS_C_CJK) on
kernels.md §2's atoms and §3's output contract (ends, n, pos, resume). The three Splits are made in one forward
pass without materializing the first two: their pieces (the N chunks and the regions) are implicit in the
class bytes, because every run stops at an atom of another kind. tmpl_params is ignored (one variant).

O(len): a piece consumes >= 1 atom, every scan walks forward, D5 and D7 give back at most one atom. Text is read
only inside [text, text + len): toks_utf8_len is given the exact remainder and every byte test is guarded by its
offset < len. No allocation, no state beyond a's fields.
```

### §k3_dsv3_c.c.2

Before `static uint64_t toks_d3_run(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t from, uint8_t want,`:

```text
end of the maximal run of atoms of kind `want` from `from` (an atom start; from itself when the atom there is
another kind). *last = the start of the run's last atom (from when the run is empty). The kind of the atom
that ended the run goes to *stop (0xFF: the segment end). Bound: len - from, >= 1 byte per atom.
```

## src/core/k3_o200k_c.c

### §k3_o200k_c.c.1

Before `#include "kernels.h"`:

```text
k3_o200k_c.c: the c twin of K3 (scan), template o200k -- docs/templates/o200k.md §2-3 (the rules O0-O6; the
case groups TOKS_C_UPPER / TOKS_C_LOWER of tables built with class_flags 0; tmpl_params CONTR, DIGITS, HAN,
NO_SLASH) on kernels.md §2's atoms and §3's output contract (ends, n, pos, resume).

One forward pass, O(len): a piece consumes >= 1 atom; an atom is visited by at most three case-run scans
(L1(i+1), L1(i), and the next piece's L1/L2, which share one scan); a whitespace run yields at most an O4
piece plus one O5 piece. Text is read only inside [text, text + len): toks_utf8_len is given the exact
remainder and every byte test is guarded by its offset < len. No allocation, no state beyond a's fields.
```

### §k3_o200k_c.c.2

Before `static uint64_t toks_o2_case(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t s, uint8_t cu,`:

```text
The case alternatives from s (o200k.md §3), cu / k the class and length of the atom at s. Returns the end of
L1(s) when it applies, else 0 and *l2 = the end of L2(s) (0 when that does not apply either). One scan of
the UP run serves both: u = u(s), lastlo = the end of the last LO atom inside [s, u) (0 when none).
```
## src/core/k6_c.c

### §k6_c.c.1

Before `#include "bpe.h"`:

```text
k6_c.c: the c twin of K6, bpe of one piece (kernels.md §5; its paths, stale test and work area: §5 "the c
twin"), and toks_k6_merge, the same merge loop on given initial symbols (the sentencepiece path, spm_c.c).
```

## src/core/kernels.h

### §kernels.h.1

Before `#ifndef TOKS_KERNELS_H`:

```text
kernels.h: the kernel prototypes of every tier and the driver's call sites (kernels.md §7):
each is a switch on the version a tier runs, compiled to direct calls, no function pointers
(SPEC §9).

Kernel sources and flags: the build defines TOKS_HAVE_<NAME>=1 for every kernel source
src/asm/<isa>/<name>.S it assembles, the name upper-cased (Makefile: k3_cl100k_neon.S ->
TOKS_HAVE_K3_CL100K_NEON, k5_avx2.S -> TOKS_HAVE_K5_AVX2); a kernel K of tier T is k<n>[_<variant>]_<t>.S.
Tiers fall back per kernel (TOKS_RUN, one rule for every kernel): a context's tier runs each kernel's
version for that tier when it is built, else the best version built for a lower tier the cpu also
runs (avx512 -> avx2 on x86-64), else the c twin. A tier is offered at load only when one of its own
kernels is built (TOKS_HAVE_<TIER>, load.c), so toks_info reports the highest tier whose code runs.
K6 runs inside K5 (each K5 tier calls its own K6), so it has no call site here.
K3 is one kernel per template: toks_k3 (K3_CL100K), toks_k3_o200k (K3_O200K) and toks_k3_dsv3 (K3_DSV3;
a k3_<template>_<t>.S defines TOKS_HAVE_K3_<TEMPLATE>_<T>; every template has neon and avx2, which the avx512
tier falls back to); api.c's run_text picks by t->tmpl. An asm K3 runs through toks_k3_tier.
```

### §kernels.h.2

Before `#define TOKS_RUN(K, tier)                                                                             \`:

```text
the version of kernel K (its flags TOKS_HAVE_<K>_<TIER>) that a context of the given tier runs: the
tier's own when built, else the highest lower tier's that is built and that the cpu runs too (the
avx512 tier's features include avx2's), else TOKS_TIER_SCALAR (the c twin).
```

### §kernels.h.3

Before `static inline uint8_t toks_k3_atom(const toks_tables *t, const uint8_t *p, uint64_t avail, uint32_t *cp,`:

```text
the class, code point and byte length of the atom at p (avail >= 1 bytes left): an ill-formed sequence starts
over at each byte, a one-byte P atom (kernels.md §2). strip: the class bits a Han atom drops (o200k's
TOKS_TP_HAN: Han atoms are in neither case group), else 0.
```

### §kernels.h.4

Before `static inline uint64_t toks_k3_ws(const toks_tables *t, const uint8_t *text, uint64_t len, uint64_t pos,`:

```text
A5-A7 (o200k's O4-O6) at pos, R = [pos, e) the maximal WS|NL run: the end of R's last NL atom when it has one
(A5, with ws_nl), e when R ends the segment (A6) or, with ncut, an N atom follows it (A8: it ends the regex's
input), the start of R's last atom when R has two or more (A6's give-back), else e (A7: R is one atom).
```

### §kernels.h.5

Before `#if defined(__aarch64__) || defined(_M_ARM64)`:

```text
K3 on every tier (kernels.md §3.1): toks_k3_short at each call site before any asm entry (one atom = one piece; a
remainder under MIN_N bytes is the twin's call unless it has MIN_A+ bytes and under 4 non-ascii ones in its first
8: the measured crossovers), then an asm tier's toks_k3_parts (the part, the twin over its stretches)
```

## src/core/layout.h

### §layout.h.1

Before `#ifndef TOKS_LAYOUT_H`:

```text
layout.h: the internal contract between the c core (compiler, loader, driver, scalar tier) and the asm
kernels (src/asm/arm64, src/asm/x86_64). Included from c and from .S (clang preprocesses .S).

Everything a kernel reads lives in toks_tables (built by the loader, read-only after load) or in the
argument struct of its call (filled by the driver; carries in/out state). Kernels never parse the scratch
header and never see toks_ctx. Offsets below are part of the contract: the c structs are asserted against
them, the asm uses the #defines. Every field is 8-byte sized or packed into an 8-byte slot so the layout is
identical on all five targets (all are 64-bit little-endian; long is never used).

Kernel naming: toks_k<N>_<name>[_<template>]_<tier>, tier in {c, neon, avx2, avx512}. The c twin (tier c)
of every kernel is the scalar tier and the oracle for the asm tiers (SPEC §2.1, T5). Calling
convention: every kernel is  uint64_t k(const toks_tables *t, <kind>_args *a)  (two pointer arguments on
every abi), returns the value documented below and writes its outputs into *a.

docs/kernels.md is the prose half of this contract (semantics, geometry, carry rules).
```

### §layout.h.2

Before `#define TOKS_KEY_MAXLEN        15`:

```text
---- shortcut entries (static table and the dynamic piece cache) ----------------------------------------
key    16 bytes: piece bytes [0, len) zero-padded to 15 bytes, byte 15 = len (1..15). An all-zero key is an
       empty entry (a real key has len >= 1).
val    16 bytes: uint32 ids[4]; ids[0] bits 29..31 = count (1..4); ids beyond count are don't-care. Every
       id is < 2^21, so emit = store (ids & TOKS_ID_MASK) as 16 bytes, advance by count. A dynamic-cache
       entry also carries its tag (TOKS_TAG_BITS = 22 bits) in the val's spare bits: tag bits 0..10 in ids[2]
       bits 21..31, tag bits 11..21 in ids[3] bits 21..31, i.e. the val's high 8 bytes (little-endian u64)
       masked by TOKS_TAG_MASK64 equal toks_tag_word(tag). Static entries hold 0 there; emit masks it off.
bucket 64 bytes, 64-aligned = one cache line on every target: keys[2] at +0, vals[2] at +32. Both tables
       use this bucket, so a hit never touches a second line.
hash   h = crc32c_u64(crc32c_u64(TOKS_HSEED, lo), hi) over the key as two little-endian u64 halves lo (bytes
       0-7) and hi (bytes 8-15). crc32c_u64 = the x86 sse4.2 `crc32 r32, r/m64` / arm64 `crc32cx` step
       (reflected 0x82F63B78, no inversion); toks_crc32c_u64_ref below is its portable definition.
static table: a key is in bucket h & words_mask or rotr32(h, 16) & words_mask. dynamic cache: bucket h &
       cache_mask; the way chosen by key alone (way 0 first), then its tag decides (docs/kernels.md §6).
```

### §layout.h.3

Before `#define TOKS_FIB64             0x9E3779B97F4A7C15ull`:

```text
---- merge table ----------------------------------------------------------------------------------------
key  = left << 21 | right   (both ids < 2^21)
slot = key << 22 | prio     empty slot = all ones (key all ones is never a real pair)
prio = the merged id when TOKS_TF_IDS_AS_RANK (ids rise strictly with rank, so comparing ids compares
       ranks and the merged id needs no lookup); otherwise the rank (< 2^22) and the merged id is
       rank2id[prio]. The pair to merge next is the one with the lowest prio, ties to the leftmost: hf's
       (rank, pos) order.
buckets of 8 slots (64 bytes, 64-aligned); home bucket = (key * TOKS_FIB64) >> merge_shift; probe the home
bucket, then the following ones (wrapping), at most merge_maxprobe buckets; a bucket with an empty slot
ends the probe (absent). Invariant (the builder's; every probe relies on it): a bucket fills front to back
and never loses a slot, so its occupied slots precede its empty ones and a probe may stop at the first
empty slot; a key sits d < merge_maxprobe buckets past its home, with every bucket in between full.
bytepair[b0 << 8 | b1] = prio of the pair (byte2id[b0], byte2id[b1]), or TOKS_PRIO_NONE.
```

### §layout.h.4

Before `#define TOKS_PM_STAGE1         1056         /* 32 two-byte leads + 16 x 64 (three-byte lead, b1) */`:

```text
---- premerge table: K6's certified initial symbols for multi-byte chars (docs/kernels.md §5.1) ----------
A 2- or 3-byte utf-8 char pattern c with an entry starts K6 as ONE symbol, its token, instead of its bytes,
unless its entry's risk bit for the byte before it in the piece or for the byte after it is set. The
builder certifies every entry against the merge table (kernels.md §5.1): the ids never change.
index  2-byte char (lead C0..DF, then one continuation byte b1):       s1 = lead - 0xC0,     s2 = b1 & 63
       3-byte char (lead E0..EF, then continuation bytes b1, b2):      s1 = 32 + ((lead & 15) << 6 | (b1 & 63)),
                                                                       s2 = b2 & 63
       stage1 = (const uint32_t *)premerge, TOKS_PM_STAGE1 entries: the byte offset of s1's block from
       premerge (blocks start at TOKS_PM_HDR); entry = *(const uint64_t *)(premerge + stage1[s1] + 8 * s2).
       entry 0: no premerge (an empty block is all zeros). The index is one-to-one on these byte patterns
       (valid utf-8 or not), so an entry belongs to exactly one byte string.
entry  bits 0..20 the token id; bit 63 (TOKS_PM_ALWAYS) set; the char stays bytes when bit pm_lbit(the byte
       before it) or bit pm_rbit(the byte after it) is set. No byte before / after it in the piece: bit
       TOKS_PM_NEVER (always clear).
         byte before:  20 -> 56,  other 00..7F -> 57,  80..BF -> 40 + (b & 15),  C0..FF -> 63
         byte after:   00..7F -> 58,  80..BF -> 63,  C0..DF -> 59,  E0..EF -> 24 + (b & 15),  F0..FF -> 60
```

### §layout.h.5

Before `#define TOKS_VSEED             0x85EBCA6Bu`:

```text
---- vocab hash (exact whole-piece lookup: ignore_merges, long pieces) ----------------------------------
h = TOKS_VSEED ^ len; for each 8-byte little-endian word w of the bytes (last one zero-padded):
    h = crc32c_u64(h, w)
slots: uint64 (id << 32 | h), empty = all ones; linear probe from h & vhash_mask until empty; a hit needs
equal h and equal bytes (tok_bytes[tok_off[id] .. tok_off[id + 1])).
```

## src/core/load.c

### §load.c.1

Before `#include "kernels.h"`:

```text
load.c: toks_load / toks_load_mem_copy / toks_unload and tier selection
(include/toks.h, SPEC §4.2; kernels.md §7). The pipeline: the source bytes (a tokenizer.json
path, a model directory holding one, or the bytes themselves) -> toks_config_parse (into a parse
arena) -> toks_compile (every kernel table but bpe's, in one toks_plat_arena block) ->
toks_bpe_build (bpe's, in a second) -> the identity and the tier. The source bytes and the parse
arena are freed before load returns: compile and the bpe builder copy what they keep. A context
owns exactly its struct, the two table blocks and decode's block (toks_dec_build).
A tiktoken model (a ranks file: tiktoken.model, recognized by content) takes the tiktoken reader
instead of toks_config_parse, with its two companions read from the same directory
(tokenizer_config.json, tokenization_kimi.py; docs/models/kimi.md section 3); a model directory
without tokenizer.json is tried for tiktoken.model.
```

### §load.c.2

Before `static int64_t pick_tier(uint32_t want, uint64_t f, const char **why)`:

```text
the tier (toks.h): opts->tier, or for AUTO the TOKS_TIER environment variable (auto, scalar,
neon, avx2, avx512), else the best this machine and build offer. A forced tier the cpu lacks,
or whose kernels this build does not have, is TOKS_E_TIER: it never runs what it names. f is
toks_cpu_features(), read once at load and kept in the context for toks_get_info.
```

## src/core/norm.c

### §norm.c.1

Before `#include "norm.h"`:

```text
norm.c: the normalizer. One engine whose steps are data (norm.h's TOKS_NS_* bits): hf tokenizers 0.23.2's NFC
(unicode-normalization-alignments 0.1.12, unicode 9.0) and its BertNormalizer (clean_text, handle_chinese_chars,
strip_accents = NFD then Mn removed, lowercase) are step sets of it; the K2 c twin; the read-only scan the driver's
zero-copy NFC path runs, and the stretch planner. Semantics, proofs and the api: norm.h.

Tables: src/gen/norm_nfc.h (tools/gen/norm.py: decompositions, classes, composition pairs) and
src/gen/bert_tables.h (tools/gen/bert_tables.py: the BERT classes, Mn of unicode 8.0, to_lowercase); each is
checked against hf on every scalar by its generator.

Strict c (SPEC §9): no allocation, no recursion, no function pointers, libc = memcpy / memcmp; every loop
states its bound. Runs are normalized without storing their decomposition: a run is re-read from the input
(atoms -> chars through the tables) once per pass.
```

### §norm.c.2

Before `#define BAD     0x00400000u`:

```text
---- decomposed chars ------------------------------------------------------------------------------------
a char of a run's decomposition is one u32: bits 0..20 the code point, bit 21 SECOND, bits 24..29 cls (the
pool's format, norm_nfc.h), or an invalid byte: BAD | the byte (cls 0, never composes, maps or is dropped).
```

### §norm.c.3

Before `static uint32_t decomp(uint32_t f, uint32_t cp, uint32_t *e)`:

```text
the chars of cp's full canonical decomposition (unicode 9.0, the crate's) into e (the pool's format); returns
their count. Without a decomposing step (canonical, strip_accents) cp alone with class 0: then nothing reorders
(hf's Lowercase and the BertNormalizer without strip_accents map char by char). Hangul syllables decompose by
arithmetic (L V [T], all starters) unless f composes: NFC keeps them whole, since they are boundaries and L+V /
LV+T recompose by arithmetic, which gives what decomposing and recomposing them (as the crate does) gives.
```

### §norm.c.4

Before `static uint32_t atom_chars(uint32_t f, const uint8_t *s, uint64_t i, uint64_t len, uint32_t *e, uint64_t *nb)`:

```text
the chars of the atom at s[i] (i < len, an atom start) under the steps f into e; returns their count, *nb =
the atom's bytes. BertNormalizer's first two steps come first, as hf runs them before NFD: clean_text removes
U+0000, U+FFFD and Cc / Cf / Co but \t \n \r (0 chars: a non-starter run goes on across the removed atom, as
it does in hf's string) and maps White_Space to ' '; handle_chinese_chars writes ' ' before and after each CJK
char (whose own decomposition follows the steps). An invalid byte is a barrier (BAD | the byte: SPEC §3.3).
```

### §norm.c.5

Before `static uint64_t nrun(uint32_t f, const uint8_t *s, uint64_t len, uint64_t a, uint64_t b, uint8_t *out)`:

```text
the form f of the atoms of text[a, b) (a starts at a boundary atom or at the text's start, b is an atom start)
into out; returns the bytes written (<= 3 (b - a), norm.h). The crate's algorithm, run by run of non-starters:
within a run, sorted stably by class, the composee C absorbs a prefix of each class's marks (classes
ascending: the first mark of a class composes unless an uncomposed mark of the same class precedes it, which
only an earlier mark of that class can be); the rest is written after C, class by class. Steps after the
reordering (strip_accents' Mn removal, lowercase) apply to each char as it is written (put), so a dropped
starter still ends a run and a dropped mark still blocks, as in hf where they run on the reordered string;
a run's byte census counts what put writes. Without composition C is just the last starter.
```

### §norm.c.6

Before `static int run_is_nfc(const uint8_t *s, uint64_t len, uint64_t a, uint64_t b)`:

```text
1 when NFC leaves the run text[a, b) as it is, decided without normalizing it: after its first atom (when
that is a boundary) every atom has no decomposition, composes with nothing (not SECOND) and the classes do not
decrease (such an atom is a mark: an NB atom of class 0 without a decomposition is a SECOND). Why that
suffices for the crate's algorithm: the boundary atom's full decomposition s0 m1..mk recomposes to itself
alone; the stable sort merges the marks y after the m's (ties keep the m's first), so every buffered y before
some mi has a class below mi's and never blocks it -- the composee absorbs m1..mk exactly as alone -- and no y
composes (not SECOND); the y's come out in their own (already sorted) order. Exercised by marks2 (every
ordered pair of marks behind 7 bases) and the corpus.
```

### §norm.c.7

Before `if (k <= 1u) { return 1; }`:

```text
an invalid byte, or ascii: every ascii char has NB clear (test_norm checks the table). toks_cp_decode takes
2..4-byte sequences only: an ascii atom used to read text[i + 1 .. i + 3] (past len) and index stage1 out
of bounds
```

### §norm.c.8

Before `/* the base class of the atom at g[i] (i < n, an atom start) and its length */`:

```text
---- the stretch planner: what the driver materializes (maintainer doctrine, speed) --------------------------------

A gap NFC leaves unchanged (toks_nfc_scan finds no changed run: all already-NFC text) is scanned in place and
no byte of it is copied. Otherwise only stretches [s, t) around the changed runs are normalized into scratch
and scanned there; everything between them is scanned in place. s and t are restart points of the cl100k
template: positions p (0 < p < n) where
  (a) the atom y at p is an NFC boundary that normalization leaves in place: any boundary atom inside an
      unchanged stretch, or an ascii WS / NL byte starting a changed run (no composition starts with one).
      So NFC(g) = NFC(g[0, p)) NFC(g[p, n)), and y and the atom x before p are the same atoms in the
      normalized text;
  (b) (base(x), base(y)) is a restart pair: x in L and y not in L; x in N and y not in N; x in P and y in WS
      or N.
Then K3 (docs/kernels.md §3, rules A1-A7) over g[a, p) and over g[p, b) alone gives exactly the pieces K3
over g[a, b) gives:
  - the piece holding x ends at p: an L run (A2; also a contraction A1 or a prefixed run ending in x) takes no
    non-L atom, an N piece (A3) no non-N atom, a P run (A4) stops before WS / N and extends over NL only; and
    x prefixes nothing from p (A2's prefix is a P / WS atom before an L atom, A3's and A4's a space, A1's a ');
  - no rule deciding an earlier piece reads p: A1, A2 and A4 read at most two atoms past a piece start, so at
    most into x's piece and y, which they test for "a letter / an e or l / NL / L / P", false for every y the
    pair allows after x just as for no atom at all (the end of the segment); A5-A7's whitespace runs stop at
    x;
  - no rule looks behind its piece start, so the scan from p does not depend on g[a, p).
x is never WS / NL: A5 and A6 read a whitespace run to its end, however far. Other templates have no restart
points (NONE: one piece per segment), and neither does a stretch without one: such a gap is one stretch.
```

### §norm.c.9

Before `static uint64_t atom_before(const uint8_t *g, uint64_t n, uint64_t lo, uint64_t p)`:

```text
the start of the atom ending at the atom start p (lo < p, lo an atom start): the well-formed sequence ending
at p, else the invalid byte g[p - 1] (a lead byte is never inside an earlier atom, and no sequence runs past
the atom start p)
```

## src/core/norm.h

### §norm.h.1

Before `#ifndef TOKS_NORM_H`:

```text
norm.h: S1 normalization (SPEC §2.6): one engine whose steps are data. A form is a set of TOKS_NS_* steps,
applied to every char in this order: BertNormalizer's clean_text and chinese padding, decomposition (canonical)
with the non-starters of each run sorted stably by class, composition, strip_accents' Mn removal, lowercase.
hf's normalizers.NFC is CANON | COMPOSE; its BertNormalizer is its four flags (strip_accents brings the
canonical decomposition, as hf's does: NFD first). Every step is exactly as hf tokenizers 0.23.2 runs it.

NFC. hf's normalizers.NFC is the crate unicode-normalization-alignments 0.1.12 applied to the whole
normalized string: Unicode 9.0 data (src/gen/norm_nfc.h; tools/gen/norm.py parses the crate's own tables and
verifies them against hf), the standard algorithm: decompose every char (canonical, full), stably sort every
run of non-starters by canonical combining class, compose left to right (a non-starter composes with the
last starter unless a char between them was left uncomposed with a class >= its own; a starter composes only
with the starter right before it). Invalid utf-8 (SPEC §3.3, docs/kernels.md §2: every byte of an
ill-formed sequence is its own atom) is a barrier: a starter that never composes, decomposes or maps, its
raw byte copied.

Runs. A boundary atom (norm_nfc.h: NB clear, or an invalid byte) is a starter that nothing before it composes
with or reorders past, and that NFC leaves unchanged alone: NFC(x b y) = NFC(x) NFC(b y). So a text is a
sequence of runs [boundary atom, next boundary atom); a run holding no NB atom is copied, the others are
normalized whole (toks_norm copies what toks_nfc_scan finds unchanged).

K2 answers "no normalization work inside this block": a block with no byte >= 0xCC holds only code points
below U+0300 (and invalid bytes), all of them boundaries NFC leaves unchanged (asserted for every one by the
generator). It never means "this block may be flushed": its last atom may still compose with what follows.

Output bound: TOKS_NORM_BOUND(NFC, n) = 3n bytes for n input bytes, and 3n is reached. Proof for NFC: the output is
the concatenation of verbatim spans and invalid bytes (1 byte per byte) and normalized runs. A normalized run
writes its decomposed chars that stay uncomposed plus composites, each replacing a composee and the chars it
absorbed. Composing never lengthens the utf-8 (len(c) <= len(a) + len(b) for every pair, asserted by the
generator; hangul L+V and LV+T take 3 + 3 bytes to 3; syllables are never decomposed here), so a run writes at
most the utf-8 length of its chars' full decompositions, and a char's full decomposition is at most 3x its
own utf-8 (asserted for every scalar; reached by U+0390 -> U+03B9 U+0308 U+0301, 2 -> 6 bytes, and by
U+1D160 -> three 4-byte chars, 4 -> 12, which NFC keeps decomposed: n/4 copies of U+1D160 write exactly 3n).
The BertNormalizer's forms stay within 3n too (tools/gen/bert_tables.py checks every scalar under every flag
set; the worst is a hangul syllable under strip_accents, 3 -> 9 bytes; tests/c/test_wp.c bounds).

NFKC / NFKD (COMPAT): the compatibility decomposition replaces the canonical one where the char has one
(norm_nfc.h KX, 3,678 chars, <= 18 chars each); NFKC's runs use NBK (the NFKC boundary set; K2's hot byte is
0xC2: every code point below U+00A0 is an NFKC boundary it leaves unchanged). Output bound TOKS_NORM_X = 11n,
reached by U+FDFA (3 -> 33 bytes). StripAccents (STRIP_M: General_Category Mark of the crate's 9.0 tables,
removed after composition) and Lowercase only shorten or map one char at a time, so for the forms that do not
compose with them the output is the concatenation of each input char's own; tests/c/test_norm.c checks every
scalar alone under every form a reader accepts against TOKS_NORM_X (composing forms with StripAccents or
Lowercase are refused by the Unigram reader, the only one that reads them).

Work and memory: linear in the input. Each atom is classified once; each normalized run is read at most
four times (its extent and class census, the composition of the first marks of each class -- a composee
absorbs at most 3 marks, since each absorption lengthens its full decomposition, which has <= 4 chars -- and
the scatter that writes the marks class by class straight into the output). No intermediate decomposition is
stored: memory is < 1 KiB of stack, whatever the run length (1 M combining marks included).
```

### §norm.h.2

Before `uint64_t toks_k2(const uint8_t *text, uint64_t pos, uint64_t len);`:

```text
K2, the nfc quick check (its c twin; no asm tier yet): the first position in [pos, len) whose byte is >= 0xCC
(TOKS_NFC_HOT_BYTE), or len. Per 64-byte block the predicate is "no byte >= 0xCC"; the first block where it
fails is then searched for the byte. Reads only text[pos, len). Precondition: pos <= len (len - pos is
computed unsigned).
```

### §norm.h.3

Before `int64_t toks_norm(uint32_t steps, const uint8_t *text, uint64_t len, uint8_t *out, uint64_t cap);`:

```text
toks_norm: the form `steps` of text[0, len) into out; returns the bytes written, or TOKS_E_CAP (nothing
written) when cap < TOKS_NORM_BOUND(steps, len). out must not overlap text. NFC copies the runs toks_nfc_scan finds
unchanged and normalizes only the others; every other form runs the engine over the whole text.
```

### §norm.h.4

Before `#define TOKS_NORM_MAX_OUT 8u`:

```text
toks_norm_char: the chars ONE char becomes under the steps after BertNormalizer's first two (decomposition,
strip_accents, lowercase): writes at most TOKS_NORM_MAX_OUT items to o (a 4-char decomposition, each char
lowercased to <= 2) and returns their count. An item is cls << 24 | the char, cls the dense rank of its
combining class when a decomposing step applies (it orders a non-starter run: wp_scan.c sorts by it), else 0.
An item TOKS_NORM_GHOST (cls 0) stands for a starter that strip_accents dropped: it writes nothing but ends the
non-starter run (wordpiece.md §3.2 N3). The scan calls it on the rare chars N3 / N4 rewrite.
```

### §norm.h.5

Before `uint64_t toks_nfc_scan(const uint8_t *text, uint64_t len, uint64_t pos, uint64_t *run_end);`:

```text
toks_nfc_scan: the first run of text[pos, len) that NFC changes, reading only. Returns its start (a
boundary atom) and stores its end (the next boundary atom, or len) in *run_end; returns len, and stores len,
when NFC leaves text[pos, len) unchanged. A run that the quick check cannot clear is normalized into a stack
buffer and compared when it is at most TOKS_NFC_SCAN_RUN bytes; a longer one is reported as changed (the
caller then normalizes a run NFC may leave alone, which is exact either way).
precondition: pos <= len, and pos is 0 or the start of a boundary atom (NFC(text[pos, len)) is then exactly
the part of NFC(text[0, len)) that starts there).
```

### §norm.h.6

Before `int toks_nfc_boundary(const uint8_t *text, uint64_t len, uint64_t i);`:

```text
1 when the atom at text[i] (i < len, an atom start) is a boundary: an invalid byte, or a code point with NB
clear (every ascii char). A segment split before a boundary normalizes as its two parts. Reads text[i, len)
only.
```

### §norm.h.7

Before `typedef struct toks_nfc_plan {`:

```text
---- the stretch planner (norm.c has the restart-point proof) ------------------------------------------
which spans of a gap g[0, n) the driver normalizes into scratch; everything else it scans in place. begin
finds the first changed run (d0 == n: the gap is already NFC, nothing is copied). Each next returns 1 with
the next stretch [*s, *e): ordered, disjoint, holding every changed run, cut only at K3 restart points (or
0 and n); 0 when no stretch is left. t: the tokenizer's tables (template and classes).
```

## src/core/precompiled.c

### §precompiled.c.1

Before `#include "unigram.h"`:

```text
precompiled.c: hf's Precompiled normalizer (spm_precompiled 0.1.4 inside tokenizers 0.23.2) compiled into
tables, and the extended-grapheme automaton it walks with (docs/algorithms/unigram.md §3).

Load time: the sentencepiece double array is enumerated once (an explicit stack over the trie, every unit
read bounds-checked) and only what hf can ever use is kept: one-char keys in a two-stage table by code
point, multi-char keys of <= 5 bytes without a shorter key prefix in a small open-addressing table, both
pointing into the charsmap's own string blob. A charsmap on which hf would panic for some input (a
transition or leaf unit outside the array, a value offset off a char boundary) or that hf refuses (strings
not utf-8) is refused here with the reason named; so is a key that is not utf-8 (no census charsmap has one).
Encode time: unigram.c walks graphemes with toks_gc_step and asks toks_pc_char / toks_pc_multi.
```

### §precompiled.c.2

Before `while (1) {`:

```text
bound: sentencepiece's darts-clone arrays are DAWGs (equal suffix subtrees shared), so the walk counts
key prefixes, not units: at most PC_MAX_PREFIXES pushes (nmt_nfkc: 262,073), each costing 256 probes
and one pop
```

## src/core/segment.c

### §segment.c.1

Before `#include "kernels.h"`:

```text
segment.c: the driver's added-token policy (kernels.md §4, §7 step 2), the c that runs
once per match around K1. Exposed as a cursor (toks_seg_begin / toks_seg_next) so the
encode and pieces walks stream units (gap, token, gap, ...) with O(1) state, no recursion
and no output array to size (a text can be one one-byte token per byte).

hf's added_vocabulary.rs find_matches, exactly (the parity suites verified this policy over
400K differential cases against hf 0.23.2; tests/model/toks_model.py added_split is the
same algorithm):
  1. mode NONSPECIAL (hf encode_special_tokens = true) and the token is special: drop the
     match; its bytes stay plain text; the search resumes at the raw m_end (hf does not
     rescan inside a dropped match -- pinned by tests/model/REVIEW.md's <mask>/sk> repro).
  2. single_word: drop unless the match is neither preceded nor followed by a regex-crate
     \w char (ends_with_word(text[..start]) / starts_with_word(text[stop..]) on the raw match,
     before any strip; the text is the cursor's: the whole input in phase 0, the gap in
     phase 1). \w is src/gen/rx_word.c, probed through hf (tests/data/breadth/probe_hf.py).
     An ill-formed byte (toks's byte input, SPEC §3.3) is neither \w nor \s.
  3. lstrip: the start moves left over the regex-crate \s run before it, never before the
     previous split's end (hf: start = max(space_leftmost_at_end, start_offset)).
  4. rstrip: the end moves right over the \s run after it (the next match may then start
     inside the stripped run: no gap is emitted for the overlap; hf keeps this exactly).
  5. the gap [prev_end, start) is emitted when non-empty, then the token; prev_end = end.
     The search resumes at the RAW m_end even after rstrip.
Mode NONE never starts a cursor. Phase 1 (the caller) runs the same policy on each
normalized gap's bytes with the phase-1 token set (api.c run_gap: the gap NFC'd first when the
tokenizer's normalizer is NFC, its bytes as they are when it has none, as hf does).
```

### §segment.c.2

Before `static const uint32_t WS_SET[25] = {`:

```text
the regex-crate \s set: the parity suites' 25 verified code points (hf's strip runs are
LEFTMOST_SPACE_AT_END / RIGHTMOST_SPACE_AT_START over exactly these; rust regex \s =
\p{White_Space}). sorted; binary search. U+001C..001F are NOT members.
```

### §segment.c.3

Before `static uint32_t atom_before(const uint8_t *text, uint64_t end, uint32_t *len)`:

```text
the atom that ends at end (end >= 1): the well-formed char text[k, end) when there is one (k is the
last non-continuation byte within 3 bytes, which the forward atomization always lands on), else
the ill-formed byte text[end - 1].
```

### §segment.c.4

Before `if (start >= end) { continue; }`:

```text
an lstrip clamped to prev_end inside a run an earlier rstrip swallowed can reach end: hf
pushes the empty split (start == stop) and PreTokenizedString::split filters it out, so the
token vanishes and nothing moves (start == prev_end here, so there is no gap either).
start > stop panics in hf; config.c refuses the token sets that can get there.
```

### §segment.c.5

Before `if ((e->flags & TOKS_AF_PFX) != 0u)`:

```text
a phase-1 token under Prepend ▁ + Replace ' ' -> ▁ (config.c §9) matches in hf only after a ▁ of the normalized
gap: the Prepend's at offset 0, or a raw ' ' / ▁ right before it, which becomes the token's first char (its raw
span starts there). Elsewhere the match is dropped and not rescanned: a content holds no ' ' or ▁, so no other
match can start inside it. The gap's units after such a token get no Prepend (api.c run_units: TOKS_SPM_NOPFX);
a token that absorbed the gap's first char leaves the Prepend alone in front of it: hf encodes a lone "▁" there,
whose original offsets are that char's (hf aligns the Prepend to the gap's first char), e.g. " <s>" -> "▁" (0, 1)
then "▁<s>" (0, 4).
```

## src/core/split.c

### §split.c.1

Before `#include "kernels.h"`:

```text
split.c: exact split planning (SPEC §4.5, §5). docs/split.md states every family's rules, their proof and the
enumeration evidence; this note is the cl100k summary.

toks_split_points returns cuts c in (0, len) such that encoding the parts alone (no post-processing, every
part after the first with TOKS_CONTINUATION) and concatenating the ids gives exactly the whole input's ids
without post-processing (§5.2). A cut is certified by local predicates only (§5.4), from both sides
(§5.3): the whole input's pieces have a boundary at c, the prefix scanned to its own end gives exactly the
whole input's pieces before c, and the suffix scanned alone gives the ones after it. The model encodes
pieces independently, so equal pieces give equal ids. Conjuncts, all over a window of atoms at c:

  added tokens  (modes ALL and NONSPECIAL, when the context has any): no occurrence of any added token of
                either phase overlaps the atoms the rule reads. Then the leftmost-longest matches (dropped
                specials included) are the same on the whole input and on both parts, and those atoms lie
                in one text gap. Strip and single_word tokens (docs/breadth.md §2) keep this sound:
                lstrip / rstrip extend a split only over a whitespace run next to the token, and no rule
                cuts inside a whitespace run or after a one-atom run the token itself touches; single_word
                reads one char beside the match, which lies on the match's side of c (test_breadth checks
                parts == whole over every breadth fixture).
  normalizer    NFC (ctx->nfc): every atom the rule reads and the atom after them is an NFC boundary atom
                (norm.h), so NFC(x) = NFC(x[..c]) NFC(x[c..]) and NFC leaves those atoms in place: the rule
                sees the same atoms in the normalized text. Phase-1 tokens match the normalized text, so an
                NFC tokenizer with them gets no cut (no pinned file has them). Sentencepiece-style bpe folds
                its normalizer into tables (spm.h): its own test below covers it.
  scan          template TOKS_TMPL_CL100K: one rule per base class of a, the atom that ends at c, over b, the
                atom that starts at c, and z, the atom before a (kernels.md §2-3; P includes invalid atoms):
                  R1  a L                 b not L
                  R2  a N                 b not N; any b under DIGITS_1
                  R3  a P                 b N or WS; b NL without PUNCT_NL; b L when a cannot start a
                                          letter piece: z is P, or neither LPREFIX_ANY nor a contraction
                                          apostrophe
                  R4  a NL                b not WS / NL, under WS_NL
                  R5  a WS (NL w/o WS_NL) b not WS / NL, z not WS / NL (a is a one-atom run), and a is
                                          not a prefix of b (U+0020 before L / P, and before N under
                                          DIGITS_SP_RUN; any WS before L under LPREFIX_ANY)
  spm           sentencepiece-style bpe (ctx->spm): toks_spm_cut, a Metaspace piece start or a certified word
                cut (spm_bpe.md §5.5, whose load-time preconditions the spm loader enforces), where the
                continuation part gets no prefix symbol the whole lacks (never under a Prepend normalizer).
  o200k, dsv3,  docs/split.md §3-7 (rules O1-O5, D1-D5, the wordpiece and unigram ascii rules), each a family of
  wordpiece,    toks_cuts_of / toks_cut_ok (split.h), shared with the incremental api.
  unigram
  other         templates without rules (TOKS_TMPL_NONE: a gap is one piece), truncation / padding, kimi's cuts,
                the cl100k cut-first parameters and the one-piece unigram chains get no cut (docs/split.md §1).

Each offset is the certified cut nearest its target len * i / n_want, lower on a tie, within
D = min(TOKS_SPLIT_D, len / (4 n_want)) (so parts stay within 1.5 / n_want of len when no target is
skipped); a target without one is skipped; equal neighbours collapse (nearest-point maps are monotone,
so the offsets come out strictly increasing). Work: at most n_want x (2D + 1) predicate checks, each
reading a window of W = 2 x (longest added token) + 24 bytes around its position; no allocation, no
scratch (scr is reserved).
```

## src/core/split.h

### §split.h.1

Before `#ifndef TOKS_SPLIT_H`:

```text
split.h: what the core gives toks_par beyond the public abi (SPEC §2.4: toks_par adds no semantics, so
it asks the core for the post-processor instead of knowing its representation). Internal, not installed.
```

### §split.h.2

Before `void toks_pp_ids(const toks_ctx *ctx, uint32_t flags, const uint32_t **pre, uint32_t *n_pre,`:

```text
the post-processor around one whole document under flags (SPEC §5.2: applied once, after assembly):
n_pre ids at pre go before the content, n_suf ids at suf after it (both 0 with TOKS_NO_POSTPROCESS).
The pointers stay valid while ctx is loaded.
```

## src/core/spm.h

### §spm.h.1

Before `#ifndef TOKS_SPM_H`:

```text
spm.h: sentencepiece-style bpe (SPEC §1.2 algorithm 2; docs/algorithms/spm_bpe.md is the contract).

Load:  config.c reads a BPE tokenizer.json without ByteLevel into a toks_spm_config, compile.c fills
       tok_off / tok_bytes with the raw strings, then toks_spm_build compiles the rest (doc §6: the text
       model is folded into the tables, the text is never rewritten).
Run:   toks_spm_encode per text unit (a phase-0 gap or a phase-1 split), toks_spm_pieces for toks_pieces,
       toks_spm_decode for toks_decode (streaming, no scratch).
```

### §spm.h.2

Before `#define TOKS_SPM_E_ID          0x001FFFFFu`:

```text
---- the compiled tables (doc §6) ---------------------------------------------------------------------

The char table maps every code point c of the INPUT to the entry of its image phi(c) under the text
model's char substitutions (normalizer Replace steps and Metaspace's U+0020 -> replacement): the space
byte and a literal U+2581 share one entry. A prefix (Prepend, Metaspace) is one virtual symbol, never
written. Entry bits:
  0..20   the id of the image as a one-char vocab string, or TOKS_SPM_E_NOID
  21..28  the small index of the image: its code point when < 128, 128 for U+2581, 255 otherwise (the
          2 KiB cut bitmap answers a pair of small indexes with one bit)
  30      TOKS_SPM_E_PAIRED: the image is followed by another char inside some reachable vocab string
```

### §spm.h.3

Before `int64_t toks_spm_build(toks_tables *t, const toks_spm_config *cfg, uint8_t **mem, uint64_t *mem_len,`:

```text
compiles cfg into t (merge table, rank2id, byte2id, vhash, flags, n_merges; t->n_ids, tok_off and
tok_bytes are compile.c's and already set) and one new block (*mem, toks_plat_arena) that holds *out.
0, TOKS_E_UNSUPPORTED (err->what: a text model the tables cannot absorb), TOKS_E_LIMIT or TOKS_E_NOMEM.
```

### §spm.h.4

Before `uint64_t toks_spm_encode(const toks_tables *t, const toks_spm *s, const uint8_t *text, uint64_t len, int at_st`:

```text
encodes one text unit text[0, len) (a phase-0 gap, or a phase-1 split); at_start: it begins at input
offset 0 and the call is no continuation (Metaspace first). ids go to out[k] for n <= k < cap; returns
the new count (n + this unit's ids). cache: the scratch's short cache, cache_mask + 1 buckets (K5's format and
tags, kernels.md §6; tag: the scratch's epoch) or NULL; work: TOKS_SPM_WORK_BYTES(len + 1) + 8 bytes. The driver
passes the whole short cache (TOKS_SCRATCH_CACHE_MIB(n)'s n / 2 MiB, else 2 MiB) from the first call: unlike K5
(a fresh scratch gets 2 MiB until it is warm), spm's misses are model calls, so a fresh scratch loses nothing to the
larger region, and a switch would lose the entries filled before it (measured: gb10e, gemma 4 at 32 MiB: pass and
warm 2-10% above a K5-style switch, cold equal).
```

### §spm.h.5

Before `int toks_spm_cut(const toks_spm *s, const uint8_t *text, uint64_t len, uint64_t a, uint64_t c);`:

```text
the split planner's cut test inside one text unit (docs/split.md §3): 1 when text[.., c) and text[c, ..)
(the second a continuation: no Metaspace-first prefix) encode to the unit's ids; a is the start of the char
ending at c.
```

## src/core/spm_build.c

### §spm_build.c.1

Before `#include "spm.h"`:

```text
spm_build.c: toks_spm_config -> the merge tables in toks_tables + one block holding toks_spm (spm.h). What is built
(the fold, byte2id, the char table, the merge table, vhash, the pairs and cut tables, holes) and why:
docs/algorithms/spm_bpe.md §6 (the text model is folded into the tables: encode never writes text).
```

## src/core/spm_c.c

### §spm_c.c.1

Before `#include "spm.h"`:

```text
spm_c.c: sentencepiece-style bpe over the folded tables (spm.h; docs/algorithms/spm_bpe.md): toks_spm_encode
(one text unit read in place: the scan §5.5, the word cache §6.6, the model §5.2 + K6's merge loop §5.3),
toks_spm_pieces, toks_spm_model (tests), toks_spm_decode (§8, streaming into the caller's buffer).
```

### §spm_c.c.2

Before `#define SPM_WIDE_LEN   30u`:

```text
the word cache (spm_bpe.md §6.6): the scratch's dynamic cache region (kernels.md §6-7 buckets and tags), this
path's alone, hashed with one multiply. Short entries take one way; wide entries (16..30 bytes or 5..7 ids)
the whole bucket, keys marked 0x40 in bytes 15 and 31, which no short or K5 key has there.
```

### §spm_c.c.3

Before `uint32_t o = (text[ws] >= 0xC0u && ws >= base) ? one[ws - base] : 0u;`:

```text
a word of one 2-4 byte vocab char (CJK text: most words) is its id (§5.2 a, nothing to merge): the
char was a per-char step of this window, as its lead byte is not ASCII. Like the cache, a shortcut:
TOKS_SPM_NOCACHE (the tests' control) sends every word to the model.
```

## src/core/stream.c

### §stream.c.1

Before `#include "core.h"`:

```text
stream.c: stream decode (SPEC §3.4, §4.7) for hf's ByteLevel decoder, and the lossy utf-8
writer that toks_decode (api.c) shares with it.

hf decodes ids to the concatenation of their bytes (tok_bytes; ids without a string add nothing,
specials are dropped under skip_special_tokens) followed by rust's String::from_utf8_lossy: one
U+FFFD per maximal subpart of an ill-formed sequence (unicode §3.9). The writer below is that
conversion run incrementally: valid utf-8 goes straight from tok_bytes into the caller's buffer,
and a well-formed proper prefix at the end of the input so far (a lead byte plus at most two
continuation bytes, the only thing a later byte can still change) is held until the bytes that
complete or break it arrive. Nothing else is staged: no scratch, no copy of a push.

What a push emits (the rule). B = the bytes of
every id pushed since toks_stream_init or the last flush that toks_decode keeps (same flags);
held(B) = the length of B's longest suffix that starts with a lead byte and is a proper prefix of
a well-formed sequence (0..3; second-byte ranges of unicode table 3-7). After every successful
push, everything the stream has emitted since init / flush, concatenated, equals
lossy(B[0, |B| - held(B))): a push returns exactly the bytes by which that grew, i.e. every byte
of the batch decode that no later id can change, and nothing else. A flush returns the rest of
lossy(B) (one U+FFFD when held(B) > 0, else nothing) and returns st to its initial state. Hence,
for every partition of ids into pushes, push_1 ++ ... ++ push_k ++ flush == toks_decode(ids).

The bound: toks_stream_bound(ctx, n) = n D + 3 (saturating), D = the longest toks_decode of a
single id (ctx->dec_max). Proof: for a token t entering with held prefix h and leaving h',
out(t) + R(h') <= R(h) + lossy_len(t), where R(h) = 3 when h is non-empty, else 0 (h broken: one
U+FFFD, then t parses as alone; h completed by t's first c continuation bytes: np + c <= 3 + c
bytes, where t alone spends 3c on them; t's own held tail would be one U+FFFD alone). Summed over
a push it telescopes: out(push) <= 3 + sum lossy_len(t_i) <= 3 + n D; a flush writes <= 3 =
bound(ctx, 0). Every push and flush is atomic: on any error st is unchanged; after TOKS_E_CAP
out[0, cap) holds the first cap bytes the call would have returned, after TOKS_E_ID it is
unspecified. A successful call may also write out[r, cap) (whole 16-byte moves).

sentencepiece-style decoders (spm.h, spm_c.c's toks_spm_decode is the batch): the chain is at most one
per-token step (Replace a -> b, or Metaspace whose first kept string loses every replacement char), then
ByteFallback (a maximal run of <0xHH> tokens is ONE string: its bytes when they are valid utf-8 as a
whole, else one U+FFFD per byte), Fuse, Strip (up to `start` leading chars at the start of the output);
no decoder: the strings joined with " ". The rule is the same: a push emits every byte no later id can
change. That is everything but an open byte run that is still valid so far (its output, its bytes or a
U+FFFD per byte, is decided by its end): the stream holds such a run, up to SST_RUN bytes in st or the
size of a caller's hold (toks_stream_hold; a push that would hold more returns TOKS_E_LIMIT with st and the
hold unchanged, out[0, cap) possibly holding a prefix of its output), and emits it when a string ends it or
at flush; a run once invalid is final, one U+FFFD per
byte as it comes. st holds the run (or the caller's hold's record), Strip's count and whether a string
was kept yet (Metaspace's first string, the joining " "). With a hold of H bytes the bound grows by 3H:
each held byte is at most 3 output bytes when its run ends (one U+FFFD, or its share of a valid run's
chars: the only per-token step a run's chars go through, unigram BFRF's Replace ▁ -> " ", shrinks).

WordPiece (wp_api.c): hf's decoder sees one token at a time (the prefix rule, " " before every kept token but
the first, cleanup on that token's string alone), so no later id changes an emitted byte: a push emits its ids'
decode and holds nothing, and the state is whether a token was kept since init / flush (SST_FIRST); the
flush emits nothing. dec_max = the longest token + 1 (the joining " "; prefix removal and cleanup only shrink).

Unigram (unigram.c toks_uni_stream): Metaspace / no decoder work per token like WordPiece; the byte-fallback
chains (RBF, BFRF) hold an open byte run like the spm chains above (up to SST_RUN bytes, TOKS_E_LIMIT beyond),
and BFRF's leading-space strip is the state's SST_FIRST bit.

Speed (maintainer doctrine: copy memory as little as needed): the bytes go once from the tables straight into
out. toks_dec_build (load) gives every id a length byte (dec_len) and short ids a 16-byte slot
(dec_slot): an id whose bytes are well-formed utf-8 on its own entering with nothing held is its own
output, so a short one is one 16-byte load from its slot and one store while out has room for it (an
exact copy at the end of out), a long one whole 16-byte moves from tok_bytes. The rest (partial
tokens, invalid bytes, a held prefix) takes the byte-wise path. No libc call per token or byte.
```

### §stream.c.6

Before `int64_t toks_stream_hold(const toks_ctx *ctx, toks_stream *st, void *hold, uint64_t cap)`:

```text
the stream's hold (toks.h): the open run's bytes move into hold[0, cap) (cap 0: back into st's own
SST_RUN bytes), and st keeps a record of it in buf: the pointer, cap, the held count, the run's utf-8
state (so no push re-reads the held bytes: a run of any length costs O(1) per push to resume) and a
check of pointer and cap against a stray write. A push writes the hold only after it can no longer
fail (sw_commit after the TOKS_E_CAP test), so TOKS_E_CAP and TOKS_E_LIMIT leave the hold as they
leave st (out[0, cap) may hold a prefix of the output: the decided bytes stream into out as the push
runs, and the run's length is known at its end; checking it first would scan the push twice), and the
recovery from TOKS_E_LIMIT is a larger hold and the same push again: the push needs at most
held + n <= cap + n bytes. The move copies away from any overlap (no memmove: the core calls
no libc but memcpy / memset / memcmp). Decided for 0.3.0: a setter rather than a second init, because
an init cannot grow a live run; toks_stream_init and its 44 bytes are unchanged.
```

### §stream.c.2

Before `static const toks_spm_op *spm_ops(const toks_spm *s, int *bf, const toks_spm_op **strip)`:

```text
the per-token step of an spm decoder chain, or NULL; *bf = 1 with ByteFallback; *strip = the Strip
step or NULL (spm_config.c accepts at most one of each, in the order per-token, ByteFallback, Fuse,
Strip).
```

### §stream.c.3

Before `int bf;`:

```text
the weight of toks_stream_bound for an spm chain: a byte token is at most one U+FFFD; any other
string at most its bytes times the per-token step's growth (Replace a -> b: b.n per byte of a
match, Metaspace and Strip only shrink), plus the joining " " without a decoder
```

### §stream.c.4

Before `static int u8_state(const uint8_t *b, uint32_t n, uint32_t *need, uint8_t *lo, uint8_t *hi)`:

```text
the utf-8 state after b[0, n): 1 when the bytes are a well-formed prefix (complete chars, then possibly
one incomplete), with *need = the continuation bytes the last char still needs and [*lo, *hi] the next
byte's range; 0 when no extension is well-formed.
```

### §stream.c.5

Before `static int64_t spm_push(const toks_ctx *ctx, sw *w, const uint32_t *ids, uint64_t n)`:

```text
one push through an spm chain. An open run's bytes from earlier pushes are in st (buf[0, np)) or the
caller's hold; this push's part of it is re-read from ids when the run ends (spm_c.c does the same in
batch), so only a run still open AND valid at the end of the push is held, and only that one can be too
long (TOKS_E_LIMIT). Its bytes join the held ones in sw_commit, once the push cannot fail.
```

## src/core/tiktoken.c

### §tiktoken.c.1

Before `#include "tiktoken.h"`:

```text
tiktoken.c: tiktoken-format models -> toks_config (tiktoken.h; docs/models/kimi.md section 3 is the
definition this file reads).

The ranks file: lines "<base64 token> <rank>" separated by LF (a CR LF ending is accepted, as tiktoken's
splitlines does; empty lines are skipped, as tiktoken does). toks takes exactly one space, canonical
base64, a canonical decimal rank < TOKS_MAX_IDS, a token of 1 .. TOKS_MAX_TOKEN_BYTES bytes, every token
and every rank once, and all 256 one-byte tokens (tiktoken needs each byte as a token: byte_pair_encode
indexes ranks[byte]). Everything else is refused with its reason.

Kimi's wrapper is identified by content: tokenization_kimi.py must hold every line of WRAPPER_LINES
(compared without leading / trailing blanks), the eight pattern lines consecutively and in order.

Strict c (SPEC section 9): no recursion, no allocation but the caller's arena, every loop bounded.
```

### §tiktoken.c.2

Before `(void)wrapper_len;`:

```text
ranks: raw bytes <= 3/4 len, alphabet strings <= 2 raw, merges <= raw (12 B each), per-id arrays
(TOKS_MAX_IDS ids at most: pointer + length + offset + raw length), the index (2 slots per line of
>= 7 bytes); config: json.c's bound (config.h); names: 256 x 32 B; the specials twice over (the
reader's list, then parse's two phases: <= 2 x 256 entries); alignment slack.
```

### §tiktoken.c.3

Before `static int names_disjoint(const toks_cfg_added *sp)`:

```text
every occurrence of every name is disjoint from every other's: no name is inside another and no proper
suffix of one is a proper prefix of another (then leftmost matching is order-independent: tiktoken's
alternation in hash order, transformers' trie and K1's leftmost-longest find the same occurrences).
```

### §tiktoken.c.4

Before `uint8_t named[TOKS_TIKTOKEN_RESERVED];`:

```text
the special-token attributes: bos, eos, unk, pad must name specials (the wrapper indexes them), and
with additional_special_tokens (and extra_special_tokens as a list, transformers 5) they are transformers'
named specials: its all_special_ids, whatever added_tokens_decoder's special flag says (docs/models/kimi.md
§3.5). The trie splits on every added_tokens_decoder entry and every attribute.
```

### §tiktoken.c.5

Before `toks_cfg_added *ph = (toks_cfg_added *)toks_ar_alloc(ar, (uint64_t)(info->n_trie + TOKS_TIKTOKEN_RESERVED) *`:

```text
the pre-tokenizer: the o200k template's kimi variant (docs/models/kimi.md §5) and the wrapper's cuts
(§2.3). The added tokens become the two matchers of config.h's cut_run: phase 0 = the trie's tokens
(transformers' split, mode ALL), phase 1 = all 256 names (tiktoken's specials inside each cut piece),
each in id order; decode's skip set stays the named specials (.special).
```

## src/core/tiktoken.h

### §tiktoken.h.1

Before `#ifndef TOKS_TIKTOKEN_H`:

```text
tiktoken.h: tiktoken-format models -> toks_config (docs/models/kimi.md section 3).

A tiktoken model ships a ranks file ("<base64 token> <rank>" per line: tiktoken.model) and the code that names
its pattern and special tokens. The reader is three calls:

  toks_tiktoken_ranks   the ranks file alone (what every tiktoken encoding shares): the vocabulary (id = rank,
                        a rank with no token is a hole: an empty string), and tiktoken's bpe written as
                        merges: every split of every token into two tokens, each with the merged token's id;
                        ids_as_rank = 1 (a merge's priority is its merged id: equal for every split of one
                        token, ties to the leftmost pair, which is tiktoken's byte_pair_merge) and
                        ignore_merges = 1 (tiktoken looks the whole piece up first).
  toks_tiktoken_kimi    + tokenizer_config.json + tokenization_kimi.py: the TikTokenTokenizer wrapper of Kimi
                        K2 / K2.5 / K3, identified by its content, never by a name: the 256 special ids from
                        len(ranks), their names, which of them transformers' added-token trie splits on, which
                        ones decode(skip_special_tokens=True) drops, and the wrapper's chunking constants.
  toks_tiktoken_parse   both, then the pre-tokenizer: the o200k template's kimi variant (TOKS_PATTERN_KIMI)
                        and the wrapper's cuts (cfg->cut_chunk / cut_run, which the driver walks), with the
                        specials as the two matchers config.h's cut_run describes (phase 0: the trie's
                        tokens; phase 1: all 256 names).

Strict where the reference is loose (docs/models/kimi.md 3.1): tiktoken's load_tiktoken_bpe accepts any
whitespace between the fields, non-canonical base64 (it drops foreign characters), python int() syntax for
the rank, and a repeated token (the last rank wins); toks refuses each, by name. All memory comes from the
caller's arena (toks_tiktoken_arena_bound).
```

### §tiktoken.h.2

Before `int64_t toks_tiktoken_ranks(const uint8_t *data, uint64_t len, toks_arena *ar, toks_config *cfg, toks_err *err`:

```text
the ranks file -> cfg (vocab, merges, n_vocab = n_ids = max rank + 1, ignore_merges, ids_as_rank,
dec_byte_level; no pattern, no added tokens). 0, or TOKS_E_FORMAT / TOKS_E_UNSUPPORTED / TOKS_E_LIMIT /
TOKS_E_NOMEM with err->what naming the reason.
```

## src/core/uni_api.c

### §uni_api.c.1

Before `#include "kernels.h"`:

```text
uni_api.c: the Unigram family behind include/toks.h (docs/algorithms/unigram.md §7, §8, §10): load (the model's
tables, the checks on phase-1 added tokens and truncation), encode (added tokens -> segments -> the c twin,
truncation, the post-processor) and decode. api.c's run() and toks_decode() call in here for contexts whose
algorithm is TOKS_ALGO_UNIGRAM, after their own argument and scratch checks.
```

### §uni_api.c.2

Before `for (uint32_t i = 0u; i < cfg->n_added; i++) {           /* bound: n_added */`:

```text
phase-1 (normalized) added tokens match on the normalized gap: toks normalizes the gap into the scratch
and needs the content to be its own normalized form and no U+2581 to survive the chain (the gap is then
read in place by the pieces, unigram.md §10)
```

### §uni_api.c.3

Before `int64_t toks_uni_run(const struct toks_ctx *ctx, const toks_scratch *h, const uint8_t *text, uint64_t len,`:

```text
encode (ids = 1) and pieces (ids = 0) for the Unigram family (unigram.md §7, §10). The work areas sit in the
scratch's work + bounce regions (K5's bounce is unused here): 3 x toks_uni_area(len) <= 33 len + 198 bytes
(work_x <= 11, refused above at load), and the regions hold 256 + 32 tmax + 4 tmax + 16 >= 36 len + 272.
```

## src/core/unigram.c

### §unigram.c.1

Before `#include "unigram.h"`:

```text
unigram.c: the Unigram model's c twin (docs/algorithms/unigram.md §2, §5, §6, §8, §10).

  - scores: the json lexeme parsed exactly as serde_json 1.0.151 parses it in hf's build (§2)
  - the trie: a double array over the pieces with U+2581 rewritten to the byte 0x20 (the Metaspace symbol is
    a table property: a space in the caller's text IS the symbol, a prepended one is a virtual first byte)
  - the Viterbi of §6.2 over one piece read in place: a ring of max_piece + 1 path scores and ONE back-pointer
    byte per position (the length of the best edge into it); ids are recovered on the way out by an exact
    trie walk of each path span, so the hot state is 1 byte per input byte
  - the segment encoder: normalizer chain + pre-tokenizers + model in one pass over the segment, pieces as
    spans of the caller's buffer; a piece is copied to the work area only when the chain changed one of its
    chars (a charsmap value, a deleted char, a literal U+2581)
  - the decoder chains of the census (§8)
```

### §unigram.c.2

Before `extern const double toks_pow10[309];               /* src/gen/pow10.c (tools/gen/pow10.py) */`:

```text
======================================================================================================
scores (serde_json 1.0.151, float_roundtrip off: de.rs parse_integer / parse_decimal / parse_exponent /
f64_from_parts)
======================================================================================================
```

### §unigram.c.3

Before `#define UNI_MAX_PIECE 127u            /* back-pointer bytes keep bit 7 for the path mark */`:

```text
======================================================================================================
build
======================================================================================================
```

### §unigram.c.4

Before `/* the byte at position i of the piece (0x20 for the virtual first symbol) */`:

```text
======================================================================================================
the Viterbi on one piece (§6)
======================================================================================================
```

### §unigram.c.5

Before `#define NOSRC UINT64_MAX`:

```text
======================================================================================================
the segment encoder: normalizer chain -> pre-tokenizers -> pieces -> model (§3-§5, §10)
======================================================================================================
```

## src/core/unigram.h

### §unigram.h.1

Before `#ifndef TOKS_UNIGRAM_H`:

```text
unigram.h: the Unigram family's tables and c twins (docs/algorithms/unigram.md is the contract).

  toks_pc_*   the Precompiled charsmap, compiled from sentencepiece's double array into what hf can ever use
              (unigram.md §3.2: one-char keys + live multi-char keys of <= 5 bytes), walked grapheme by
              grapheme (§3.3) with an identity fast path: chars that map to themselves are never copied.
  toks_gc_*   the Unicode 17 extended-grapheme automaton (unicode-segmentation 1.13.3, src/gen/grapheme17.c).
  toks_uni_*  the model: a double-array trie over the pieces with U+2581 remapped to the byte 0x20 (§10: the
              Metaspace / Replace ' ' -> U+2581 symbol is a table property, never a rewritten byte), f64 scores
              parsed as serde_json parses them (§2), the Viterbi of §6, and the segment encoder that turns a
              text segment between added tokens into ids, reading the caller's buffer in place.

Everything here is load-time construction plus per-call work with caller-provided memory; no allocation,
recursion or function pointer at encode time (SPEC §9).
```

### §unigram.h.2

Before `uint64_t toks_uni_area(const toks_uni *u, uint64_t len);`:

```text
one work area for a text of len bytes: pre_x work_x len + 3 bytes, 64-aligned (a piece copy, the back-pointers, a
normalized gap, the steps before the charsmap); toks_uni_run lays out 3 of them in the scratch's work + bounce
regions, sized for pre_x len (api.c scr_x): 3 (11 pre_x len + 66) <= 36 pre_x len + 272 (work_x <= 11).
```

### §unigram.h.3

Before `int64_t toks_uni_normalize(const toks_uni *u, const uint8_t *text, uint64_t len, int gap_start, uint8_t *dst,`:

```text
the normalizer chain alone on text[0, len) into dst[0, cap) (hf's normalized string, U+2581 as 3 bytes):
its length, or -1 when it does not fit. Phase-1 added tokens match on it. gap_start 0 (TOKS_CONTINUATION
at input offset 0): no Strip on the left.
```

### §unigram.h.4

Before `typedef struct toks_uni_call {`:

```text
the output of a segment: ids (toks_encode), or the end offsets of the pre-tokenizer's pieces in the
normalized byte stream (toks_pieces, SPEC §3.5: nbase + the piece's end in the segment's normalized
form). out[n] is written while n < cap; n counts on. work: 2 areas (3 with a phase-1 gap, uni_api.c).
```

### §unigram.h.5

Before `int64_t toks_uni_encode_segment(const toks_uni *u, const uint8_t *text, uint64_t len, int pre_normalized,`:

```text
one text segment between added tokens: normalizer chain (skipped when pre_normalized: text is already
toks_uni_normalize's output), pre-tokenizers, model. gap_start 0: the segment continues a gap
(TOKS_CONTINUATION at input offset 0: no Strip on the left, no '(?<!\n)^' prefix). Returns the segment's
normalized length, or TOKS_E_SCRATCH when c->work_bytes is short (unreachable from toks_uni_run). Invalid
utf-8: unigram.md §9.
```

## src/core/vocab.c

### §vocab.c.1

Before `#include "core.h"`:

```text
vocab.c: the vocabulary lookups of include/toks.h. The key is toks' bytes, not hf's vocabulary spelling (the
tiktoken choice, the one a c engine holds, and self-consistent with toks_token): an added token's content first
(hf reads its added map before the model), then the id whose string is the bytes toks_token returns
(docs/algorithms/spm_bpe.md §8.1: the string decode prints for the id, which is the model vocabulary's string
except in two places: a ByteLevel id's string is its raw bytes, not the alphabet form hf's vocabulary writes,
and an added token hf normalizes is its normalized form, llama2's "▁<s>"). An hf spelling fed as bytes can
name another valid id (gpt2 "¢": hf's byte token 95, toks' 44359 spelled "Â¢"), which toks.h says first. So
toks_token_to_id(toks_token(id)) is id unless another id wins that string as hf would pick it: an added
token with that content, or another id of the same bytes (below). The empty string is never a key. The key is bytes, so strings that decode alike are one key: a ByteLevel vocab string and an
added content ("ĠĠ" and "  ": the added token's, hf's token_to_id("  ")), or two vocab strings, a raw
"\u200d" and the alphabet form "âĢį" (the one the file writes as the bytes themselves: hf's
token_to_id of that text). hf tells them apart by their written forms, which toks keeps only at load
(python/toks/_vocab.py reads them from the file). toks_id_flags: ADDED when some added token was given the id (hf's added_tokens_decoder); SPECIAL when any
listing of the content that holds the id (the last listed under it, map_r's and_modify) is special (hf's
special_tokens_set; no cached file lists one content with two flags), not decode's skip bit, which is §8.1's
string-equality rule (llama2's id 1 is special and decode keeps it); BYTE for an id that stands for one raw
byte: decode's ByteFallback test (§8.3) on the id's string for chains that have the step, or a byte-level
id whose string is one byte. 0: a plain vocabulary id, or an id with no string.
```

### §vocab.c.2

Before `int64_t toks_vocab_build(toks_ctx *c, const struct toks_config *cfg)`:

```text
the index, built once at load in one toks_plat_arena block (mem_voc): a power-of-two table of u32 slots at
load <= 0.8, linear probing; a slot is key + 1 (23 bits) with the top 9 bits of the string's hash as a tag,
so most probes that miss compare no bytes. A key below voc_n_add names an added content (offset, length, id
in voc_add; the bytes in voc_pool, copied from the file's list: the tables keep only the phase-1 forms), any
other key an id (its string is tok_bytes). Contents go in first, in file order (a content seen again keeps
its first id: hf gives a later token with that content the id it already has), then every id with a string:
a slot holding a content of the same bytes stays (hf's precedence), one holding an id stays when the file
writes that id as the bytes themselves and not this one, else this one takes it (the later wins; a unigram
vocab never repeats a piece, config.c refuses one). A content's record notes whether any listing of it is
special; a second pass over the listings sets each id's bits from the content that holds it. Then the added and special bits: one bit each per id, the last listed token deciding special.
Cost: 4 bytes per slot (the table is 1.25 to 2.5 slots per key), 16 bytes per added content plus its bytes,
2 bits per id: 5.5 to 9 bytes per id on the pinned files, built in one pass over tok_bytes (a hash per
string, one probe sequence each). The hash reads whole words while 8 bytes remain and the tail through a
zeroed word, so it reads nothing past the string.
```

## src/core/wp.c

### §wp.c.1

Before `#include "wp.h"`:

```text
wp.c: the WordPiece model of toks (docs/algorithms/wordpiece.md §5, §12.3-12.4): the vocab tables built at load,
and the c twin of K9 (pieces -> ids).

Every lookup hashes and compares the piece where it lies (the caller's text or the scan's mat buffer) with
8-byte loads; for an uncased tokenizer each loaded word is ascii-folded in the register (wp_fold), so a
lowercased copy of the piece never exists. A whole piece in the vocab is answered by one probe (certified: W2
tries the whole piece first); the greedy longest match runs on misses only, its candidates capped at the
longest key and their hashes sharing the 8-byte words of the common prefix.
```

## src/core/wp.h

### §wp.h.1

Before `#ifndef TOKS_WP_H`:

```text
wp.h: the WordPiece / BERT text model of toks (docs/algorithms/wordpiece.md; §12 is this file's design).

  toks_wp_scan_c     BertNormalizer + BertPreTokenizer in one pass over the caller's text. Pieces are offsets
                     into that text; a piece is copied (into the caller's mat buffer) only when the normalizer
                     really rewrites something inside it (copy on write, §12.2).
  toks_wp_encode_c   the WordPiece model (K9) over those pieces: the whole-piece vocab probe (a certified
                     shortcut: a vocab string encodes as itself, wordpiece.md §5), the greedy longest match on a
                     miss. Keys are hashed and compared from in-place 8-byte loads, ascii A-Z folded in the
                     register for uncased tokenizers: the lowercased piece never exists in memory.
  (the BertNormalizer alone is norm.h's toks_norm with these flags: they are its steps)
  toks_wp_build      the tables, at load, from the vocab strings and the model / normalizer parameters.

Kernel shape (layout.h's rules): uint64_t k(const toks_wp_tables *t, <k>_args *a), pure functions of (t, a);
reads only [text, text + len) and t; writes only the output ranges in a. The offsets below are the contract
the asm tiers will read (proposed for layout.h).
```

### §wp.h.2

Before `typedef struct toks_wp_entry {`:

```text
---- vocab hash entries (16 bytes): open addressing, power-of-two size, load <= 1/2, linear probe ---------
h    = toks_wp_hash(key) (below); an empty entry has len 0 (vocab keys are never empty: hf has no "" token
       reachable by a piece, and the build skips "" keys).
key  = keys[off .. off + len)
```

### §wp.h.3

Before `int64_t toks_wp_build(toks_wp_tables *t, toks_arena *ar, const toks_wp_vocab *v, const toks_wp_params *p,`:

```text
fills *t from v and p in ar. Returns 0, TOKS_E_UNSUPPORTED (unk_token not in the vocab: err names it),
TOKS_E_LIMIT (a key over TOKS_WP_MAX_KEY) or TOKS_E_NOMEM. A duplicated string keeps its LAST id (json map
semantics, as hf).
```

### §wp.h.4

Before `static inline uint64_t wp_fold(uint64_t w)`:

```text
---- hashing (shared by build, the c twin and the tests) ---------------------------------------------------
h(key) = crc32c over the key's 8-byte little-endian words (the last one zero-padded), from TOKS_WP_HSEED,
then one crc32c step over the length. The words of a candidate's prefix are shared by every longer candidate
from the same start (wordpiece.md §12.4). Lookups apply wp_fold to every word when the tokenizer lowercases.
```

## src/core/wp_api.c

### §wp_api.c.1

Before `#include "core.h"`:

```text
wp_api.c: include/toks.h's encode, pieces and decode for a WordPiece context (docs/algorithms/wordpiece.md §2,
§7, §8, §12). api.c checks the arguments and the scratch, then hands WordPiece calls here.

encode = hf encode(text): the added-token phases (segment.c), the text model on each gap, then truncation
(Right: the first L ids, L = max_length - the template's ids when post-processing is on; the walk stops as soon
as L ids exist, hf's early exit), the template, padding. Two text paths, the same ids (tests/wordpiece/check.c
proves both against hf):
  in place    no phase-1 token can match: one scan does the normalizer and the pre-tokenizer over the caller's
              bytes and copies only the pieces it rewrites (§12.2)
  normalized  phase-1 (normalized=true) tokens exist, or the call is toks_pieces (offsets are in the normalized
              stream): the gap is normalized into scratch, phase 1 matches there, the scan then runs the
              pre-tokenizer alone over the normalized bytes
The scratch's work region holds: pieces[TOKS_CHUNK_PIECES], the scan's copy buffer, a normalized gap (3 bytes
per text byte: §12.5) and the normalizer's sort area (4 bytes per text byte); the bounce takes the ids of a
round that does not fit the caller's room.
```

### §wp_api.c.2

Before `static void gap(const wpw *w, const uint8_t *g, uint64_t n, wemit *e, int norm, int match, uint64_t *nbase)`:

```text
one phase-0 gap g[0, n). The normalized path runs in windows of about TOKS_WP_WIN raw bytes when the context
allows it (ctx->wp_win, wordpiece.md §12.6): each window ends before an ascii space, tab, cr or lf, which every
BertNormalizer maps to one whitespace byte and no phase-1 form contains, so no match, piece or strip_accents
run crosses the cut and the windows' results concatenate to the whole gap's. A truncated encode then normalizes
only the text it needs (hf normalizes all of it).
```

### §wp_api.c.3

Before `int64_t toks_wp_decode_k(const toks_ctx *ctx, const uint32_t *ids, uint64_t n, uint32_t flags, uint8_t *out,`:

```text
hf's WordPiece decoder works per token (decoders/wordpiece.rs decode_chain: the prefix rule and cleanup see one
token's string, plus whether it is the first kept one), so a stream holds nothing: *k = the tokens kept before
ids (in) and after them (out) is its whole state (stream.c).
```

## src/core/wp_scan.c

### §wp_scan.c.1

Before `#include "wp.h"`:

```text
wp_scan.c: BertNormalizer + BertPreTokenizer in one pass over the caller's text, the c twin of the scan kernel
(docs/algorithms/wordpiece.md §12.1-12.2).

Pieces are offsets into the text. A piece is copied into the mat buffer (copy on write) only when the
normalizer rewrites something inside it: a char clean_text removes from the middle of a word, a char whose
strip_accents / lowercase output differs from itself, a kept non-starter (strip_accents' run sort). ascii A-Z
under lowercase is NOT a rewrite: the model folds it in the register. Every chunk boundary (a.pos) is a char
boundary where no piece is open, or the start of a piece that is re-scanned whole by the next call.
```

### §wp_scan.c.2

Before `if ((f & TOKS_WPF_STRIP) && (cls & TOKS_BC_MN) && !(cls & TOKS_BC_DECOMP)) { return SC_HOLE_MN; }`:

```text
a mark strip_accents drops whole. A mark that DECOMPOSES (U+0F73, U+0F75, U+0F81, U+0344, ...) goes the
SC_CHANGE way instead: U+0F75 has class 0 but decomposes into two non-starters, so it does NOT end the run
around it (hf sorts U+302E U+0F75 U+1D165 into U+1D165 U+302E)
```

## src/platform/cpu.c

### §cpu.c.1

Before `#include "cpu.h"`:

```text
toks_cpu_features (SPEC §2.3): x86 cpuid + xgetbv (os-enabled state) everywhere; arm64 getauxval
(linux), sysctlbyname (mac), IsProcessorFeaturePresent (windows). Called at load; nothing is cached, so
no global is ever written (SPEC §4.1).
```

## src/platform/cpu.h

### §cpu.h.1

Before `#ifndef TOKS_CPU_H`:

```text
toks cpu feature detection (SPEC §2.3, §11): one bitfield, one accessor, read at load time.
Tier rule (SPEC §11; kernels.md §1):
  x86 avx512 = avx-512 f, bw, vl, vbmi + bmi2 (+ the avx2 set)
  x86 avx2   = avx2, bmi1, bmi2, lzcnt, popcnt, sse4.2 (crc32)
  x86 scalar = x86-64-v2 (the c twins)
  arm64 neon = armv8.0-a + neon + crc32
```

## src/platform/file.c

### §file.c.1

Before `#if !defined(_WIN32)`:

```text
file.c: the os surface of loading (posix + win32): whole-file reads, the model-directory
lookup, getenv. No logic beyond open / seek / read; the core never calls the os itself.
```

### §file.c.2

Before `static int path_is_dir(const char *path)`:

```text
No stat / fstat: glibc 2.33 made them real symbols (stat@GLIBC_2.33), and a module that calls them needs glibc
>= 2.33 (the python wheel could not be manylinux2014). open(O_DIRECTORY) and lseek answer the same questions with
symbols every glibc has. O_NONBLOCK: a fifo opens at once and then fails to seek instead of blocking the load.
```

## src/platform/mem.c

### §mem.c.1

Before `#if !defined(_WIN32)`:

```text
mem.c: platform load-time memory, the os part of the arena layer (SPEC §9's libc rule stops at src/platform).
 - toks_plat_alloc / toks_plat_free: the heap, for small load-time allocations (callers know the sizes).
 - toks_plat_arena / toks_plat_arena_free: the table arena: zeroed, 2 MiB-aligned, huge-page advised before its
   first touch where the os has the advice, and that first touch a write. Why and how per os: docs/kernels.md §7,
   the arena.
 - toks_plat_hint_huge: the same advice for a caller's memory (the piece caches of a large scratch, kernels.md §7).
```
