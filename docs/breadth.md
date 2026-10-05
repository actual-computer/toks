byte-level breadth: spellings, added-token options, RobertaProcessing, missing bytes, digits
==========================================================================================

The byte-level breadth semantics, each written from hf tokenizers 0.23.2's source (the local clone,
commit 88a4498) and proven against hf itself; what toks does with it; what is refused; and what remains.
Every receipt below names its command and host; the scripts live in tests/data/breadth/.

  item                                         toks today                  proof
  1.1 Split Removed + invert / Isolated + invert accepted (table patterns)  enumeration + 0 diffs
  1.2 dropout 0.0 / 0 / -0.0                    accepted                   source + fixtures
  1.3 padding BatchLongest                      accepted                   source + fixtures
  1.5 truncation, padding Fixed / multiple      accepted (also spm)        fixtures + real files vs hf
  1.4 vocab strings outside the byte alphabet   decode-only ids            fixtures + real files
  2   added tokens lstrip / rstrip / single_word / normalized   driver policy   36.5M synthetic cases
  3   RobertaProcessing (and the Bert shape)    accepted                   fixtures + real files
  4   vocabularies missing byte chars           dropped in the driver      fixtures + real files vs hf
  5   SmolLM Digits + ByteLevel(use_regex)      accepted (kernels.md A8)   model == hf, 1M texts


1. equivalent spellings
-----------------------

1.1 Split: Removed + invert, Isolated + invert

hf (pre_tokenizers/split.rs, tokenizer/pattern.rs, tokenizer/normalizer.rs split): the regex's find_iter
gives the matches; every stretch between two matches (and before the first, after the last) is a gap.
Each split carries a flag: true for a match, false for a gap; Invert flips the flags and nothing else.
The behavior then reads the flags: Isolated keeps every split (it maps every flag to "keep"), Removed drops
the splits flagged true. So:
  - Isolated + invert == Isolated, for any pattern;
  - Removed + invert keeps the matches and drops the gaps: it equals Isolated exactly when there is no gap.
Empty splits are dropped by PreTokenizedString::split either way.

No gap, for every pattern in config.c's table: at any position some alternative matches the atom there
without looking at anything else -- a letter starts `\p{L}+` (or `[\p{L}\p{M}]+`), a digit `\p{N}...`,
whitespace `\s+`, every other atom `[^\s\p{L}\p{N}]+` (the complement class; `[^\s\p{L}\p{M}\p{N}]` with
marks folded, the marks being letters there) -- and toks's ill-formed bytes are class P (SPEC §3.3). So
find_iter's leftmost match always starts where the previous one ended, and the matches tile the text; no
alternative matches empty. The o200k template's two strings (o200k and its nemo variant, config.c [4] and
[5]) split the letters by case, and the two letter alternatives together still take every letter alone:
`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` matches one Ll / Lm / Lo / M,
`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*` one Lu / Lt, and L = Lu + Ll + Lt + Lm + Lo; digits
take `\p{N}{1,3}` / `\p{N}`, whitespace `\s+`, the rest ` ?[^\s\p{L}\p{N}]+`. MiniMax (Text-01, M1, M2.x, M3)
spells its o200k split this way: NFC + Split(o200k string, Removed, invert true) + ByteLevel.

Proof by enumeration (probe_hf.py invert, gb10e, hf 0.23.2): for each of config.c's 4 patterns, Split
Removed+invert and Split Isolated give identical pieces on all 1,112,064 scalars in 4 contexts ("c", "acb",
" c1", "\nc'") and on 200,000 random strings over letters, marks, digits, every \s kind, punctuation, CJK and
emoji: 0 differing. Rerun with the table's 6 patterns (the o200k and nemo strings added; gb10d, 2026-10-04,
load 2.2 -> 1.2, 1 process, ~2 min): 6 x (4 x 1,112,064 scalars + 200,000 random strings), 0 differing. The o200k
strings (P11 / P17) were proven by the o200k template work (3.3M cases). Real files: granite-4.1, Olmo-3 Think /
Instruct (OLMo 2) and phi-4 spell their split this way and match hf on 1.2M cases each (§6).

toks: config.c accepts behavior Isolated with either invert, or Removed with invert true, on the table's
patterns; every other behavior (Removed without invert, Contiguous, MergedWith*) stays refused by name.

1.2 dropout 0.0

hf (models/bpe/model.rs): the builder refuses a dropout outside [0, 1]; tokenize takes the cached path when
`dropout.is_none() || dropout == Some(0.0)`, and merge_all skips a merge only when `random::<f32>() <
dropout`, never for 0.0 (-0.0 == 0.0 too). So an exact zero is null's spelling. toks: json.c now keeps every
number's lexeme; config.c accepts a dropout whose mantissa digits are all zero (0, 0.0, -0.0, 0e5, an
integer 0) and refuses any other number by name ("model dropout (not 0.0)"; a non-number is TOKS_E_FORMAT,
as hf refuses it). Census: poolside Laguna (P20) and JetBrains Mellum2 (P13) carry dropout 0.0; both still
need their templates.

1.3 padding BatchLongest

hf (tokenizer/mod.rs post_process, utils/padding.rs pad_encodings, tokenizer/encoding.rs pad): padding runs
after the post-processor; BatchLongest pads every encoding of the call to the longest one -- for the one
sequence encode() returns, its own length -- and pad_to_multiple_of > 0 rounds that length up; Encoding::pad
returns at once when the encoding is already long enough. So BatchLongest without a multiple (null or 0)
changes nothing; Fixed(n) pads every encoding. toks: accepted with every field serde requires (strategy,
direction Left / Right, u32 pad_id and pad_type_id, string pad_token; a malformed object is TOKS_E_FORMAT, as
hf refuses it); Fixed and pad_to_multiple_of > 0 pad as hf does (§1.5).

1.4 vocab strings outside the byte-level alphabet

hf: merge_word starts every word from the piece's chars, which ByteLevel has mapped into the alphabet; a
merge concatenates two symbols' strings; ignore_merges looks up the piece's alphabet image. So no encoding
reaches a vocab string with a char outside the alphabet, except through an added token whose content is
that string (it gets the vocab id) or a post-processor id. Decoding it: the ByteLevel decoder maps a token
string through the alphabet only when every char is in it, else takes the string's own utf-8.

Census: deepseek v3 / r1 / v3.1 / v3.2 / v4 ("<｜begin▁of▁sentence｜>" ...), granite-embedding and
ModernBERT (raw space runs "   " -- also added tokens), EXAONE, A.X-K2 (space runs), tiny-Cohere (1,403 raw
emoji / symbols), Laguna ("〈|UNK|〉" ...), zeta.

toks: such a string is a decode-only id: config.c counts them (cfg->n_vocab_raw) and requires the 256
alphabet chars among the other strings; compile.c gives every vocab id hf's decoder bytes (toks_token_bytes:
the alphabet image, else the raw utf-8); bpe_build.c leaves decode-only ids out of byte2id, vhash and the
words table. That last part is load-bearing: a raw string's bytes can equal an alphabet token's (raw "   " is
20 20 20, the bytes of "ĠĠĠ"; raw " " is 20, the bytes of "Ġ"), and letting it into vhash would answer an
ignore_merges piece with an id hf never gives. tests/data/breadth/nonalpha.json builds exactly that trap
(ignore_merges on, raw "   " and raw " " in the vocab, "ĠĠĠ" not); mutants 11 and 12 of mutants.sh (raw ids
let into byte2id / vhash) are killed by it.


1.5 truncation and padding (byte-level and sentencepiece-style bpe)

hf encode() (tokenizer/mod.rs post_process) cuts the model's ids to max_length minus the post-processor's added
ids (add_special_tokens; 0 without), Right, then runs the post-processor, then pads: to Fixed(n) or the
encoding's own length, rounded up to pad_to_multiple_of, Right or Left. toks does the same for one document
(api.c run(); a TOKS_CONTINUATION call is a document's later part: neither applies, as for wordpiece):
config.c reads truncation and padding as for every algorithm (wordpiece.md §7: Left truncation, OnlySecond, a
stride hf would panic on, a max_length below the template's ids, a pad id beyond the ids are refused), run()
clamps the count before the suffix ids and pads after them (core.h toks_pad). The text is still encoded whole;
an early stop at the limit is an optimization left to the bpe driver. toks_split_points certifies no cuts for a
truncating or padding tokenizer (the parts would not concatenate to the truncated whole). Proof: the breadth
fixtures trunc_pad (RobertaProcessing, max_length 9, Fixed 12) and trunc_left (no template, max_length 5,
stride 2, a multiple of 4 on the Left) against hf in every mode, and the census's truncation group pinned in
tests/data/targets/ledger.txt (all-distilroberta-v1, gte-reranker-modernbert, jina v5 omni small,
Llama-3.2-3B FP8, Giga-Embeddings, OTel-LLM-E4B) through the parity harness.


2. added tokens: lstrip, rstrip, single_word, normalized
------------------------------------------------------

hf (tokenizer/added_vocabulary.rs). Two tries, built at load from the final added vocabulary: phase 0 holds
normalized:false tokens and runs on the raw input; phase 1 holds normalized:true tokens (their contents
normalized by the normalizer, when there is one) and runs on each gap phase 0 left, after normalizing it.
find_matches(sentence) walks daachorse's LeftmostLongest matches and, per raw match [start, stop):
  1. encode_special_tokens and the token is special: skip it (its bytes stay text; no rescan inside it);
  2. single_word: skip it unless `start == 0 || !ends_with_word(sentence[..start])` and
     `stop == len || !starts_with_word(sentence[stop..])` -- the regex crate's \w on the char before and
     the char after the RAW match;
  3. lstrip: start = max(start of the \s* run that ends at start, start_offset) (start_offset = the end of
     the previous split pushed);
  4. rstrip: stop += length of the \s* run that starts at stop;
  5. push the gap (start_offset, start) when start_offset < start, then (token, (start, stop));
     start_offset = stop.
  The iterator resumes at the RAW match end, so after an rstrip the next match can start inside the run the
  previous token swallowed: no gap is pushed and the two splits overlap (those bytes are covered twice).
  toks_pieces reports each split's end where it lies in the caller's bytes (SPEC §3.5; the normalized
  form where NFC changed the text), so an overlap shows as an end that does not grow ("\t\t" with "\t"
  rstrip: ends 2, 2).
  The final gap (start_offset, len) is pushed when non-empty. `sentence` is the phase's text: the whole
  input in phase 0, the gap in phase 1, so a phase-1 strip or word check never sees past its gap.
Two consequences no source comment states, both probed:
  - start == stop happens when an lstrip token lies entirely inside a run an earlier rstrip took (start is
    clamped to start_offset == stop): the split is empty and PreTokenizedString::split drops it, so the
    token vanishes and nothing else changes;
  - start > stop happens when such a token has no rstrip of its own: hf panics ("AddedVocabulary bad
    split", added_vocabulary.rs:505), e.g. tokens "<a>" rstrip + "  " lstrip on "<a>    x".

The sets, probed through hf over every scalar value (probe_hf.py word, gb10e): \w (single_word, both
sides of a match) = 144,667 code points in 796 ranges (the regex crate's Alphabetic, M, Nd, Pc and
Join_Control); \s (lstrip and rstrip) = exactly the 25 code points segment.c already had.

toks: config.c keeps the three flags (compile.c sets TOKS_AF_LSTRIP / RSTRIP / SINGLE_WORD on the entries)
and refuses, by name, the token sets that can make hf panic: in one phase, an rstrip token beside an
lstrip-only token whose content is all \s. segment.c (the driver's policy, kernels.md §4) applies single_word
with src/gen/rx_word.c's ranges (generated by probe_hf.py word --emit), drops a token whose stripped span is
empty, and reads an ill-formed byte (toks's byte input, SPEC §3.3: hf cannot take one) as neither \s nor
\w -- before this change a lone 0x85 or 0xA0 byte counted as U+0085 / U+00A0 whitespace and lstrip swallowed
it. normalized:true needs nothing new without a normalizer (phase 1 runs on the gap's bytes); with one, those
contents must be normalized at load (hf's normalized_cache) before they are indexed.

Proof: test_breadth.c (make test) checks 8 fixtures x 153 texts x 4 flags against hf (added_opts holds
every option in both phases, special and not, an all-\s lstrip+rstrip token, a phase-1 single_word with a
non-ascii neighbour); diff.py synth (gb10e, 8 shards, seed 1): 22,787 random tokenizers (2,213 more refused
exactly as the panic rule says), 36,459,200 encode cases in modes ALL / NONSPECIAL / NONE / ALL without the
post-processor and 18,229,600 decodes, 0 differing; phi-4 (lstrip + rstrip on 100 tokens), clap and
Florence-2 (<mask> lstrip) on real text: §6. mutants.sh: 18 one-line mutants of this code, all killed.


3. RobertaProcessing
--------------------

hf (processors/roberta.rs): for one sequence with add_special_tokens, ids = [cls.1] + ids + [sep.1], the
numbers as written in the pairs (never looked up in the vocab); type ids all 0; nothing is added without
add_special_tokens. trim_offsets and add_prefix_space reach process_offsets only: they move offsets, never
ids. BertProcessing gives the same single-sequence ids.

How hf picks the post-processor (probed): PostProcessorWrapper is an untagged enum tried in the order
Roberta, Bert, ByteLevel, Template, Sequence, and only ByteLevel's and Sequence's deserializers look at
"type". A Roberta-shaped object loads as RobertaProcessing under any "type" (or none); a "RobertaProcessing"
without its two bools loads as BertProcessing; an object with sep / cls pairs is Roberta or Bert even when
typed "TemplateProcessing" or "Sequence"; a Template-shaped object needs "pair" and loads under any type
(even "ByteLevel" when ByteLevel's own fields are missing); ByteLevel needs add_prefix_space and
trim_offsets.

toks: config.c classifies each processor by its shape in hf's order (pp_kind) and flattens the one that
adds ids into pp_single: cls, $A, sep for the Roberta / Bert shapes, the template's pieces otherwise; ids
beyond the vocabulary, two id-adding processors and a Sequence inside a Sequence are refused by name.
Proof: fixtures roberta (BatchLongest + <mask> lstrip, like clap), roberta_seq (Sequence[ByteLevel(aps),
Roberta(aps)], cls not its string's id), pp_shapes (a "RobertaProcessing" without its bools = Bert),
pp_notype (no "type"); real files clap-htsat-unfused and Florence-2: §6.


4. byte-level vocabularies missing some byte chars
---------------------------------------------------

The census files (facts.py over census/coverage.json's byte-level tokenizers): 21 byte-level vocabularies
lack some of the 256 byte chars. For most the missing bytes are exactly C0 C1 F5..FF, which never occur in
valid utf-8, so hf never meets them (pythia, granite-embedding / ModernBERT, OLMoE, mamba, deepseek-coder,
Spark-X); some lack bytes valid text carries:
  SmolLM / SmolLM2 / SmolVLM2   04 06 13 14 16 1D, and F1 F2 (leads of U+40000..BFFFF)
  PowerMoE, tiny_starcoder      F1
  bloom                         0B
  falcon                        03 04 05 0B 11 13 14 15 16 17 19 1C 1D 1E, F2
  tencent Hy3, HunyuanOCR       0D (carriage return)
  LLaMmlein                     00 01 04 05 06 0B 0C 0D 0E 0F 10 12 1C 1D 1E 1F (unk + byte_fallback + fuse_unk)

hf (models/bpe/model.rs merge_word), per char of the piece: a vocab char is a symbol (flushing a pending
unk first); else with byte_fallback the char's utf-8 bytes as "<0xXX>" tokens when all exist (added at once,
a pending unk not flushed first); else with unk_token a pending unk (fused with the previous one under
fuse_unk); else nothing: the char vanishes and its neighbours become adjacent, so they can merge. The
pre-tokenizer saw the byte (splitting is unchanged); ignore_merges would look the whole piece up first
(no census file combines it with missing bytes).

So what toks must emit, unk_token null and byte_fallback off (every census case but Spark-X, whose missing
bytes are unreachable, and LLaMmlein): for each piece, the bpe of the piece with those bytes removed; a
piece made only of them emits nothing. For toks's byte input (SPEC §3.3: "byte-level models see the
byte") the same rule covers an ill-formed byte whose char is missing.

Proof (semantics.py drop, gb10b, load ~9 of 20 cores): the rule equals hf (no added tokens, no
post-processor; texts mixing every reachable missing byte with letters, digits, spaces, punctuation, CJK
and emoji) with merges done by hf's own model on SmolLM2-135M-Instruct (300,000 texts), falcon-7b (300,000)
and bloomz-560m (200,000), and again with merges done by an independent quadratic transcription of
merge_all on SmolLM2 (300,000), falcon-7b (300,000), PowerMoE-3b (200,000) and Hy3 (200,000, the CR case):
0 differing everywhere. The full merge_word model (unk pending and fused, byte_fallback's <0xXX> lookup of
the alphabet char's utf-8, merges over the whole symbol list) equals hf on LLaMmlein (200,000 texts, 16
reachable missing bytes): 0 differing.

Implemented in the driver (maintainer, 2026-10-04: no kernel change, a load-time flag): config.c accepts a
byte-level vocab missing some byte chars when unk_token is null, byte_fallback off and ignore_merges off (else
refused by name: an unk / byte_fallback emits for the char, and ignore_merges would look the uncompacted piece up
first), and marks the dropped bytes (cfg->drop, ctx->drop / has_drop); bpe_build leaves their byte2id unset. For
such a tokenizer only, api.c's run_text checks each K3 round for a dropped byte (a plain loop; real text rarely
holds one); a round without one goes to K5 / K6 as before; in a round with one, clean pieces go to K5 in runs and
each dirty piece is compacted into the scratch's norm region (in place when NFC already wrote the segment there;
the scratch keeps that region for has_drop tokenizers) and encoded as one K5 piece; an all-dropped piece emits
nothing and never reaches K5. K5 / K6 and their asm are untouched. toks_pieces is unchanged (the pre-tokenizer saw
the bytes). For toks's byte input (SPEC §3.3) the same rule drops an ill-formed byte whose char is missing.

Proof: test_breadth fixtures drop_gpt2 (VT, EOT, A9, F1 missing; gpt-2 pattern) and drop_nfc (CR and A9 missing;
NFC + cl100k split) against hf 0.23.2 over 180 texts x 4 flags each, the byte-input checks (test_drop_bytes) and
the split planner over them; the real files pythia-160m / -14m, OLMoE-1B-7B-0125-Instruct, granite-embedding-small-
english-r2 and ModernBERT-base (pinned in fetch_tokenizers.py; ModernBERT in test_e2e) now load and match hf on the
light check. The census files with reachable missing bytes (SmolLM, falcon, bloom, Hy3, PowerMoE, tiny_starcoder)
load too, with their pre-tokenizers (census/coverage.json; falcon on the generic engine).


5. SmolLM's digits-gpt2 chain (P13)
------------------------------------

The chain (census appendix A, P13; SmolLM / SmolLM2 / SmolVLM2, PowerMoE, tiny_starcoder, Mellum2,
EXAONE, tiny-Cohere):
  Sequence[Digits(individual_digits=true), ByteLevel(add_prefix_space=false, use_regex=true)]
hf: Digits splits with rust's char::is_numeric, Isolated (pre_tokenizers/digits.rs): every numeric char is a
piece of its own; ByteLevel then runs the gpt-2 regex on each remaining piece separately, so its
`\s+(?!\S)` lookahead stops at the piece end. is_numeric (probed through hf's Digits over every scalar) is
oniguruma's \p{N} (1,911 code points) plus 13 more from a newer unicode: U+11DE0..11DE9 and U+16FF4..16FF6
(1,924); oniguruma has those 13 unassigned (class P).

Model (semantics.py digits, gb10b): split at the is_numeric chars, each its own piece; hf's own gpt-2 Split
on every run between them == hf's chain on 1,000,000 random texts for SmolLM2-135M-Instruct and 1,000,000
for Mellum2 (digits, the 13 extra chars, spaces before digits, letters, punctuation, CJK): 0 differing. The
negative control -- the same model splitting at oniguruma's \p{N} -- differs on 18,205 of 300,000 texts.

Onto the cl100k template: it did not fit the earlier parameters. The gpt-2 set (CONTR_CS, LPREFIX ' ', no
PUNCT_NL, no WS_NL) stays, and it needs:
  - DIGITS_1 (each N atom one piece, no ' ' prefix: Digits put the space in another piece);
  - a new boundary bit (say TOKS_TP_N_BREAK): an N atom ends the whitespace run for A6 as the segment end
    does, so "a  1" gives "a", "  ", "1" (DIGITS_1 alone would give "a", " ", " ", "1");
  - class tables with the 13 extra code points as N for this template (a classes.c option, like the marks
    fold): with them as P, "!\U00011DE3?" would be one punctuation piece instead of three.
These are layout.h / kernels.md §3 / k3_c.c changes (and the asm tiers'), shipped as kernels.md §3 A8
(TOKS_TP_DIGIT_CUT, TOKS_CLASSES_DIGITS). SmolLM additionally needs §4 (its missing bytes are reachable); Mellum2
needs only this.


6. receipts (hf tokenizers 0.23.2 everywhere)
---------------------------------------------

On 2026-10-04:
  make -j8 test: green on a developer laptop, gb10b (linux arm64, taskset -c 0-4,10-14) and tr9970x
    (linux x86-64, taskset -c 0-7), the last two with every pinned real file fetched into build/tokcache
    (TOKS_TOKENIZER_CACHE); test_breadth 16,646 checks.
  tests/data/breadth/receipts.sh on gb10e (taskset -c 0-4,10-14, off the X925 timing cores):
    census probe: master's tools/census/probe.c, linked with the library under test, over the census cache's 185
      tokenizer.json files (an NFC-refused file probed again without its normalizer, as the census does), compared
      with census/coverage.json by probe_flips.py: 7 tokenizers now load (granite-4.1-3b, clap-htsat-unfused,
      tiny-random-OPT, Olmo-3-7B-Think, phi-4, Olmo-3-7B-Instruct, Florence-2-base: 5,813,968 (b) downloads, 0.52% of
      (b)), 2 needed NFC (Qwen2.5-VL / Qwen2-VL 7B AWQ; it has landed since), 11 reach their next refusal, 165
      unchanged; the probe's stages and toks_load_mem_copy agree on every file.
    e2e parity (diff_real.sh 300000): each of those 9 files through the toks driver against hf, 1,200,000
      encode cases (modes ALL / NONSPECIAL / NONE / ALL without the post-processor) + 600,000 decodes per
      file, 0 differing; texts: lines of Gutenberg 1342 / 1661 / 2000 / 2600 / 7849 / 23863 / 24264 and
      cpython typing.py / listobject.c (fetch_text.sh) and random mixes around each file's added tokens.
  Earlier runs, same scripts: probe_hf.py invert / word / numeric (gb10e): §1.1, §2, §5; semantics.py drop / digits
    (gb10b): §4, §5; diff.sh synth seed 1 (gb10e): 36,459,200 encode cases, 0 differing; mutants.sh: 18 one-line
    mutants, all killed (§2).


7. what remains
---------------

  - Today's misses, by what blocks each: docs/coverage.md §8; each distinct tokenizer's status: §6 there ("needs
    beyond today").
  - kernels.md §4 (K1's per-match policy), proposed additions: "word character" is the regex crate's \w
    (src/gen/rx_word.c, probed through hf), the strip runs' whitespace the 25 code points of segment.c, an
    ill-formed byte neither; an lstrip clamped to the previous split's end can leave an empty token split,
    which hf drops (the token vanishes); start > end is an hf panic, and config.c refuses the token sets that
    can reach it.
