unigram: the semantic contract for the Unigram family (Precompiled + Metaspace + Viterbi)
=========================================================================================

hf tokenizers 0.23.2 (88a4498) is the definition; this file is its reading for every census tokenizer whose model
is Unigram (sentencepiece unigram models converted to tokenizer.json: XLM-R and its multilingual embedders, T5 /
FLAN-T5, ALBERT, XLNet, DeBERTa-v3, llm-jp, ruri). tests/model/unigram_model.py is the executable transcription
and tests/unigram/run_diff.py measures it against hf (§11 has the counts). Style and vocabulary follow
docs/kernels.md: an atom is a well-formed utf-8 sequence or one ill-formed byte (kernels.md §2).

  §1  scope: the census rows, the pins                    §7  added tokens, post-processing, truncation
  §2  reading the file (vocab, scores, ids)                §8  decode
  §3  Precompiled: the charsmap walked by grapheme         §9  invalid utf-8 (the byte-input spec)
  §4  the other normalizers of the chains                  §10 the c twin: tables, not text
  §5  pre-tokenizers: WhitespaceSplit, Metaspace           §11 evidence
  §6  the model: lattice, Viterbi, unk, byte fallback      §12 what toks refuses, named


1. scope: the census rows
--------------------------

census/coverage.json (b) holds 15 distinct Unigram tokenizers behind 24 models: 140.2M of the slice's 1,125.9M
monthly downloads (12.45%). Ranked by downloads; "pin" is the file tests/unigram/pins.json fixes by hub revision
and sha-256 (tests/unigram/fetch.py downloads it; every pinned file's sha-256 equals the census's file_sha256);
"(spm only)" rows ship only a sentencepiece .model that transformers converts (input:slow-sentencepiece, census
package 10). Charsmap 0942789e is sentencepiece's nmt_nfkc (237,539 bytes; every pinned charsmap is this blob).

  #  tokenizer     pin           dl/30d  share  model (first of n)                      vocab / unk / byte_fb
  1  649fe51ccdb9  uni_pmminilm  62.51M  44.6%  paraphrase-multilingual-MiniLM-L12-v2 (3)  250002 / 3 / -
  2  7cd74b54072d  uni_bgem3     34.78M  24.8%  BAAI/bge-m3                                250002 / 3 / -
  3  5ff6fd2bd616  uni_me5large  24.16M  17.2%  intfloat/multilingual-e5-large (8)         250002 / 3 / -
  4  9d9e4878827a  uni_me5small  11.50M   8.2%  intfloat/multilingual-e5-small             250002 / 3 / -
  5  5afe8f4134af  uni_bgererank  2.41M   1.7%  BAAI/bge-reranker-large                    250002 / 3 / -
  6  be17c35a0d1f  uni_arctic2    0.97M   0.7%  Snowflake/snowflake-arctic-embed-l-v2.0    250002 / 3 / -
  7  d5630c68501c  uni_mxbaixs    0.68M   0.5%  mixedbread-ai/mxbai-rerank-xsmall-v1       128000 / 3 / -
  8  daa7a670668c  uni_albert     0.64M   0.5%  sentence-transformers/paraphrase-albert... 30000 / 1 / -
  9  f43dda922229  uni_ruri3      0.57M   0.4%  cl-nagoya/ruri-v3-310m                     102400 / 0 / yes
 10  9a47e902e45f  (spm only)     0.52M   0.4%  rinna/japanese-gpt-neox-small              44301 / 2 / -
 11  4684b25350b4  (spm only)     0.42M   0.3%  Voicelab/vlt5-base-keywords                50000 / 2 / -
 12  c65f31009867  uni_llmjp4     0.32M   0.2%  llm-jp/llm-jp-4-33b-thinking(-gguf)        196608 / 0 / yes
 13  9b62b74f480e  (spm only)     0.31M   0.2%  XLabs-AI/xflux_text_encoders (t5 v1.1)     32100 / 2 / -
 14  3c5e2462bf14  uni_xlnet      0.22M   0.2%  xlnet/xlnet-base-cased                     32000 / 0 / -
 15  622e3314ea0a  uni_llmjp3     0.20M   0.1%  llm-jp/llm-jp-3-150m                       99574 / 0 / yes
     outside (b)   uni_t5base     -       -     google-t5/t5-base (same hf-normalized model, normalizer,
                                                pre-tokenizer, decoder as row 13; added tokens differ)
     outside (b)   uni_flant5     -       -     google/flan-t5-base

The chains (hf's re-serialization; "M(always, split)" = Metaspace replacement U+2581, prepend_scheme, split):

  rows  normalizer                                            pre-tokenizer                decoder
  1 3   Precompiled                                           WhitespaceSplit > M(always,  Metaspace
  11 13 (row 10: its own charsmap 51d36847)                   split)
  t5
  2 4   Precompiled > Replace(regex ' {2,}' -> ' ')           M(always, split)             Metaspace
  5 6
  flan
  7     Strip(both) > Precompiled > Replace(' {2,}' -> ' ')   M(always, split)             Metaspace
  8     Replace('``' -> '"') > Replace("''" -> '"') > NFKD >  WhitespaceSplit > M(always,  Metaspace
        StripAccents > Lowercase > Precompiled                split)
  14    as 8 without Lowercase                                as 8                         Metaspace
  9     -                                                     M(never, no split)           Replace('▁'->' ') >
                                                                                           ByteFallback > Fuse
  12 15 Replace(regex '(?<!\n)^' -> '▁') >                    -                            ByteFallback >
        Replace(regex ' ' -> '▁')                                                          Replace('▁'->' ') >
                                                                                           Fuse > Replace(regex
                                                                                           '(?<!\n)^ ' -> '')

Post-processors are TemplateProcessing everywhere (<s> $A </s>, [CLS] $A [SEP], $A </s>, $A <sep> <cls>, $A).
Added-token options: lstrip on the mask token (rows 1-3, 5, 6, 8; normalized=true on row 5's <mask> and row 7's
[UNK]), lstrip + rstrip on the 100 <extra_id_N> of row 13. truncation in the file: rows 1 (128), 6 (512), 7 (500),
8 (100); padding BatchLongest (a no-op for one sequence) on 1, 6, 7, 8. No Unigram tokenizer in the census uses
single_word, prepend_scheme first, dropout-like sampling (alpha / nbest are not serialized), or a post-processor
other than TemplateProcessing.

Coverage by feature: rows 1-7 (97.7% of the family's downloads) are one stack -- the nmt_nfkc charsmap, at most a
Strip before it and the ' {2,}' collapse after it, Metaspace(always, split) with or without WhitespaceSplit, and
the Unigram model without byte fallback. Rows 9, 12, 15 add byte fallback and the literal-▁ chains. Rows 8 and 14
add Replace(String), NFKD, StripAccents and Lowercase (§2: the steps before the charsmap).


2. reading the file
--------------------

model: {"type": "Unigram" (or absent: hf tries the model types in order and the Unigram shape matches), "unk_id":
int or null, "vocab": [[piece, score], ...], "byte_fallback": bool (default false)}. hf (Unigram::from):
  - unk_id must be < len(vocab) when present (else the file is refused: "Unable to load vocab").
  - token_to_ids: piece -> index, inserted in file order, so a duplicated piece maps to its LAST index; the
    Viterbi reads the score through that index (the last duplicate's score). The trie holds the piece once.
  - min_score = the minimum of every score in the vocab (special pieces included); unk_score = min_score - 10.0.
  - fuse_unk is always true and the optimized Viterbi is always used (neither is serialized).

Scores are f64 as serde_json 1.0.151 parses them WITHOUT its float_roundtrip feature (hf's build): the decimal
digits accumulate into a u64 significand while significand * 10 + digit fits (later digits are dropped; dropped
integer digits still count in the exponent); then f = (f64)significand rounded to nearest-even, then ONE multiply
(exponent >= 0) or divide (exponent < 0) by 10^|exponent| from a table of correctly rounded powers 1e0..1e308
(below 1e-308 the value is divided by 1e308 first, repeatedly); a minus sign negates; an integer literal (no '.',
no exponent) is converted exactly as an integer. This is NOT correctly rounded: 17-significant-digit scores land
one ulp away from strtod's value -- 65,856 of bge-m3's 250,002 scores, 26,900 of ruri's, 48,518 of llm-jp-4's --
and the compiler must reproduce hf's value bit for bit (tests: the model's parse equals the f64 hf re-serializes
for all 2.4M scores of the pins). The ulp shows in ids: Viterbi sums are not associative, so permutations of the
same pieces, which tie in exact arithmetic, are ordered by their last bits. ruri on "\x08^^^..." takes ^^ + ^
with hf's scores and ^ + ^^ with correctly rounded ones ((B + s(^)) + s(^^) against (B + s(^^)) + s(^), B the
prefix sum); mutants.py's float_cr mutant (strtod scores) is caught this way.

Added tokens: hf's AddedVocabulary::add_tokens ignores the json "id": a token's id is the model vocab id of its
content when the content is a piece, else the next id counting up from len(vocab), in file order; empty contents
are dropped. The pins are consistent (json id == hf id).


3. Precompiled: the charsmap walked grapheme by grapheme
---------------------------------------------------------

3.1 the blob (spm_precompiled 0.1.4). base64 of: u32 le T (trie bytes), then T / 4 little-endian u32 units (the
double array of sentencepiece's Darts), then the normalized strings, each NUL-terminated (bytes past 4 + 4 *
floor(T / 4) are the strings; hf refuses the file when they are not utf-8). Unit u: has_leaf = bit 8, value =
u & 0x7FFFFFFF, label = u & 0x800000FF, offset = (u >> 10) << ((u & 0x200) >> 6).

3.2 transform(key): walk from node = offset(array[0]); for each key byte c in order: c == 0 stops the walk; node
^= c; u = array[node]; label(u) != c stops the walk; node ^= offset(u); if has_leaf(u), record value(array[node]).
The result is the value of the FIRST leaf met -- the SHORTEST key that is a prefix of the input -- or none. The
replacement string runs from that offset in the strings to the next NUL (or the end). Out-of-range units or an
offset off a char boundary make hf panic; toks refuses such a charsmap at load (§12).
Notes that bite: sentencepiece itself takes the LONGEST match; hf takes the shortest. In nmt_nfkc 217,839 of the
224,711 keys have a shorter key as a prefix and can never be chosen by hf; what hf can ever use is 4,837
one-char keys plus 675 multi-char keys of <= 5 bytes without a key prefix (compositions like A + U+0300 -> À,
'=' + U+0338 -> ≠): 5,512 live entries. Key lengths are 1-10 bytes, replacements 0-33 bytes (U+FDFA ->
"صلى الله عليه وسلم", the 11x expansion bound of the family). 32 one-byte keys: \t \n \f \r -> ' ', the other C0
controls and DEL -> ''. NUL is never a key (the walk stops), so NUL maps to itself.

3.3 the walk. The segment (a gap between added tokens, already through the normalizers before this one) is cut
into extended grapheme clusters (UAX #29 as unicode-segmentation 1.13.3 implements it: Unicode 17.0.0, rules
GB3-GB13 + GB999 including GB9c's Indic conjunct rule; tables in tests/data/unigram/grapheme17.txt, generated
from the crate pinned by hf's Cargo.lock). For each grapheme G, in order:
  - if G is shorter than 6 bytes and transform(G) finds a key: emit its replacement for the WHOLE of G;
  - otherwise, for each char c of G: emit transform(c)'s replacement if it finds a key, else c itself.
Consequences, all observed in hf 0.23.2 (normalizer output; in the pinned chains WhitespaceSplit or the ' {2,}'
collapse then makes "a b" and "a  b" the same pieces, so the CR LF rule never reaches an id there):
  - "a\r\nb" -> "a b": CR LF is one grapheme, its shortest key prefix is "\r" -> " ", and the LF goes with it.
  - "Ａ\u0301x" -> "Ax": fullwidth A + combining acute is one 5-byte grapheme; "Ａ" -> "A" replaces it, the
    accent is gone. With a 6-byte grapheme (two marks after a 2-byte letter, ...) every char maps alone.
  - U+2581 -> " ", U+3000 / NBSP / U+2000-200F / U+2028-2029 / U+202F / U+205F / U+1680 / U+FEFF / U+FFFD ->
    " " (30 one-char keys map to exactly " ", 30 to "", 52 to strings that contain a space, like U+00A8 -> " ̈").
  - Hangul jamo L V (T) are never composed (two jamo are already 6 bytes): "\u1100\u1161\u11a8" stays.
Which grapheme rules can change the output: only graphemes of 2+ chars and < 6 bytes take the whole-grapheme
path, so GB3 (CR LF), GB9 / GB9a (a char followed by Extend / ZWJ / SpacingMark within 5 bytes), GB9b (a 2- or
3-byte Prepend before a short char) and GB4 / GB5 (controls never join) decide outputs directly; GB9c and GB11
decide them indirectly (a conjunct or an emoji zwj sequence swallows a following key char + mark, e.g. क्क़́ or
❤‍™́, into one long grapheme where they map per char); GB12-GB13 never change a result for any charsmap (an
RI is 4 bytes and no 1-byte char joins it) and GB6-GB8 none for nmt_nfkc (two jamo are 6 bytes, no jamo is a
key). The c twin still implements every rule (§10.2): cheaper than proving the exceptions per charsmap.
Offsets (alignments) are rebuilt by hf with odd rules here (a dropped LF is charged to the CR); they never change
ids, and toks reports normalized offsets only (SPEC §3.5).


4. the other normalizers of the chains
---------------------------------------

Sequence applies its normalizers in order, each to the whole segment.
  - Replace String s -> t (onig on the escaped literal): leftmost, non-overlapping occurrences, left to right.
    '``' -> '"' and "''" -> '"' (rows 8, 14).
  - Replace Regex: the census spells three patterns, each with fixed semantics (any other pattern is refused,
    §12): ' {2,}' -> ' ' collapses every run of 2+ U+0020 into one U+0020 (only U+0020: tabs etc. were already
    turned into spaces by the charsmap before it); ' ' -> '▁' replaces every U+0020; '(?<!\n)^' -> '▁' inserts '▁'
    at the start of the segment only (onig's ruby ^ also matches after every \n, which the look-behind removes;
    an empty segment never reaches a normalizer).
  - Strip(left, right): removes the leading / trailing run of White_Space chars (the 25 code points of rust's
    char::is_whitespace: U+0009-000D, 0020, 0085, 00A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000).
  - before Strip and the charsmap, in hf's order (rows 8, 14): Replace with a String pattern of 1..4 bytes whose
    content is no longer (albert's '``' -> '"', "''" -> '"'; every non-overlapping match, left to right), then
    one of NFC / NFD / NFKC / NFKD, StripAccents (General_Category Mark of unicode-normalization-alignments 0.1.12,
    Unicode 9.0.0) and Lowercase (rust's char::to_lowercase), as the normalizer engine's steps (src/core/norm.h).
    They run per segment into the scratch (uni_api.c uni_pre: the text is materialized only for these files), the
    walk reads the result in place; the work areas grow by the steps' factor (TOKS_NORM_X: 11 with a
    compatibility decomposition, else 3). NFC / NFKC followed by StripAccents or Lowercase is refused.
A segment that normalizes to "" produces nothing (hf drops empty splits).


5. pre-tokenizers
------------------

Sequence applies each pre-tokenizer to every split the previous one produced; empty splits are dropped.
  - WhitespaceSplit: split on White_Space chars (§4's 25), which are removed.
  - Metaspace(replacement r, prepend_scheme, split), per split:
      1. every U+0020 becomes r (only U+0020; no other whitespace);
      2. prepend r when the split does not now start with r and the scheme is "always", or the scheme is
         "first" and the split's original offset is 0 (the first split of an input that starts with text);
         "never" prepends nothing;
      3. split = true: cut before every r (MergedWithNext: each r starts a piece, "▁a▁▁b" -> "▁a" "▁" "▁b");
         split = false: the split is one piece.
    Deserialization: prepend_scheme defaults to "always"; add_prefix_space=false forces "never" (and is refused
    with any other explicit scheme); split defaults to true.
Pieces of the census chains, on the normalized segment s:
  - WhitespaceSplit > M(always, split): r + w for each maximal run w of non-White_Space chars (a literal r inside
    w would cut it, but the charsmap maps U+2581 to a space first).
  - M(always, split) after the ' {2,}' collapse: r + w for each U+0020-separated field w (w may be empty: "a "
    gives "▁a", "▁"); a leading U+0020 is the first piece's r (no prepend), so " a" gives one piece "▁a".
  - M(never, no split) (ruri): the whole segment with U+0020 -> r, no prefix.
  - none (llm-jp): the whole segment; its normalizer already prefixed r and replaced every U+0020.


6. the model: lattice, Viterbi, unk, byte fallback
---------------------------------------------------

hf Unigram::encode_optimized on one piece p (bytes, valid utf-8), then Unigram::tokenize:

6.1 lattice. Nodes are the char boundaries of p. From each char start s, every vocab piece that is a byte prefix
of p[s..] (common prefix search in a byte trie of all pieces, shortest first) is an edge s -> s + len with its
score. If no piece of exactly one char starts at s, an unk edge s -> s + one char with unk_score is added -- only
then (a char that starts a longer piece but has no own piece still gets the unk edge). With unk_id null, choosing
an unk edge is hf's MissingUnkId error.

6.2 Viterbi, exactly. best[0] = 0.0; for s over char starts in increasing order, for each edge from s in the
order above (pieces by increasing length, then the unk edge): cand = score + best[s] (f64); the end node takes
(cand, s, id) when it has no candidate yet or cand > its best (strictly greater). So among equal scores the
candidate seen first wins: the one from the smallest start, i.e. the path whose last piece is the longest. Ties
are common, not theoretical: repeated pieces permute with equal sums ("ab=====" -> ▁ab = == == in bge-m3, not
▁ab == = ==; the >= mutant of mutants.py changes ids in 168-1,126 of 3,000 random cases per pin). Sums run left
to right along the path (best[s] already holds the prefix sum) in binary64 without contraction; the order is
part of the contract (§2: permutations differ in their last bits).
Backtrack from the last node.

6.3 unk and fuse_unk. Consecutive unk edges on the best path become one string (fuse_unk). Then each path string
is looked up in token_to_ids again: a piece maps to its id; a fused unk string that happens to be a piece maps to
that piece's id (only reachable with positive scores; synthetic tests have it); otherwise the string is unknown.
When the unk penalty matters: an unk edge only competes with pieces that contain a char without its own piece.
The pinned XLM-R and DeBERTa vocabularies have no such char (every char of every multi-char piece is also a
piece), so their unk edges are forced; llm-jp-3 has 258 such chars (Korean), ruri 3, t5 one ('<'), and with
real (<= 0) scores the penalty of 10 never flips a decision there either; synthetic tokenizers (tests/unigram/
synth.py) are where the exact unk_score shows.

6.4 byte fallback (byte_fallback = true). An unknown string becomes, byte by byte of its utf-8 (all bytes of a
fused unk run), the pieces "<0xXX>" (upper-case hex) -- if EVERY byte has its piece; if any is missing, the whole
string becomes one unk_id. Without byte fallback an unknown string is one unk_id.

6.5 hf caches results per piece (pieces < 256 bytes); the cache never changes ids.


7. added tokens, post-processing, truncation
---------------------------------------------

Extraction is docs/kernels.md §4's (two phases; daachorse leftmost-longest; resume at the raw match end; a match
dropped for NONSPECIAL is not rescanned). lstrip moves the token's start left over the White_Space run before it,
never before the previous split's end; rstrip moves its end right over the run after it. Phase 1 (normalized=true
tokens) matches the token's content as normalized by the whole normalizer (hf's normalized_cache; "<mask>" and
"[UNK]" normalize to themselves here) on each normalized gap. Each gap is normalized on its own, so a grapheme
never spans an added token and "(?<!\n)^" prefixes every gap.
Post-processing (TemplateProcessing, single): the special tokens' ids from post_processor.special_tokens[name].ids
around the sequence. truncation (file:truncation): with add_special_tokens the model ids are cut to max_length
minus the template's added count (direction Right keeps the first), then the template is applied, for one
document only (a TOKS_CONTINUATION call is a document's later part: nothing is cut, as for bpe and wordpiece); padding
BatchLongest never changes one sequence (hf's encode_batch pads a batch to its longest member; the differential
compares per-text encode() for those files).


8. decode
----------

ids -> strings: an added token's content (its normalized form when it has one), else the vocab piece; ids past
both are skipped. skip_special_tokens (hf's default true) drops strings equal to a special token's content. Then
the decoder chain, joined with "":
  - Metaspace: in every token, each r becomes ' ', except in token 0 when the scheme is not never, where every r
    is dropped (all of them, not only a leading one).
  - Replace String / Regex (r -> ' '; '(?<!\n)^ ' -> '' removes a leading space of each token): per token.
  - ByteFallback: a run of tokens that are exactly 6 bytes "<0x" hh ">" (rust's u8::from_str_radix(hh, 16); the
    vocabularies' byte pieces are upper-case <0xXX>) becomes the utf-8 decoding of those bytes, or one U+FFFD per
    byte when the run is not valid utf-8 as a whole.
  - Fuse: all tokens into one.
  - no decoder: tokens joined with ' '.

Decode and stream decode (SPEC §3.4) run on stream.c's chain decoder, the one sentencepiece-style bpe streams
with: unigram.c writes the chain as spm.h ops at load (Metaspace; Replace('▁' -> ' ') > ByteFallback > Fuse;
ByteFallback > Replace > Fuse > Strip(' ', 1), the last being hf's Replace('(?<!\n)^ ' -> '') on the fused
string; no decoder: tokens joined with ' '), and the context's toks_dchain says ByteFallback comes first in BFRF
(its chars then go through the Replace). Metaspace and no decoder work per token, so a push emits its ids' decode
and holds nothing. The byte-fallback chains hold an open run of byte tokens that is still valid utf-8 so far (the
run's output is decided by its end: its chars, or one U+FFFD per byte), up to the stream's hold at a push's end
(44 bytes, or the caller's buffer from toks_stream_hold; more is TOKS_E_LIMIT); a run once invalid is final, one U+FFFD per byte as it comes; Strip's count is the state's other
part. The batch decode is the same step from the initial state with every run decided at the end of the ids.


9. invalid utf-8 (SPEC §3.3)
--------------------------------

hf cannot be called with ill-formed input; toks defines it, consistently with the other families: every byte of
an ill-formed sequence is an invalid atom. For this family: an invalid atom is its own grapheme (a break before
and after, like a control), never part of a charsmap key (emitted unchanged), not White_Space (WhitespaceSplit
keeps it inside a word), never a U+0020 (Metaspace and Replace leave it), and never a byte of a vocab piece, so in
the lattice it has exactly one edge: an unk edge of one byte. fuse_unk joins it with neighbouring unk chars; with
byte fallback the fused run's bytes become <0xXX> pieces (all present) or one unk_id.


10. the c twin: transform the tables, not the text (maintainer doctrine, speed)
----------------------------------------------------------------------

The input is read in place. Of the steps above, these are absorbed into tables at load and never touch the text:

  step                                       absorbed as
  ----------------------------------------   ---------------------------------------------------------------------
  Metaspace r (' ' -> r) and its prepend     the vocab trie is built with every r (E2 96 81) of every piece
                                             rewritten to the byte 0x20: a U+0020 in the text IS r, read in place;
                                             the prepend is a virtual first symbol (the walk starts at the root's
                                             0x20 child), never a written byte. Valid because after the chain no
                                             0x20 can remain in a piece (Metaspace / Replace ' ' -> r replace them
                                             all), so pieces holding 0x20 are unreachable and dropped at load.
  Replace ' ' -> r, Replace '(?<!\n)^' -> r  the same virtual symbol: llm-jp's segment is one piece = r + text.
  WhitespaceSplit, Metaspace split           the piece scanner's classes (White_Space, U+0020): pieces are offset
                                             pairs into the caller's buffer.
  Replace ' {2,}' -> ' '                     a scanner rule: a run of normalized spaces is one boundary.
  Strip                                      the segment's span shrinks; nothing is copied.
  Precompiled, identity chars                one class lookup per char (a 256-entry ascii table, a two-stage table
                                             above); a run of chars that map to themselves and whose graphemes are
                                             single chars or carry no key is read in place.
  Precompiled, chars -> ' ' / -> ''          scanner classes too: a key whose value is exactly ' ' is a separator,
                                             one whose value is '' is skipped (both 30 entries in nmt_nfkc).
  TemplateProcessing, truncation             ids written around / cut from the caller's output.

and these materialize, only the piece they touch, into the scratch (<= 11x of the piece, the family's expansion
bound from §3.2):

  - a char or grapheme whose charsmap value is anything else (fullwidth, ligatures, compatibility forms,
    compositions: 5,512 live entries in nmt_nfkc), and a key char followed by joiners inside 5 bytes;
  - a literal U+2581 in the text (rewritten to 0x20 for the remapped trie);
  - phase-1 added tokens whose first byte the normalized piece could contain (rows 5 and 7): the gap is
    normalized into the scratch before matching;
  - the steps before the charsmap (rows 8, 14), on every segment: materialized whole (uni_pre; a byte map for
    ascii Lowercase could be absorbed into the walk's input translation instead).

10.1 tables built at load (src/core/precompiled.c, src/core/unigram.c):
  - charsmap: the double array is enumerated once (an explicit stack, bounded by its units), every out-of-range
    unit or off-boundary value refused; the result is a per-code-point entry (identity / separator / delete /
    value offset into a packed string pool) for the 4,837 one-char keys and a small table of the live multi-char
    keys keyed by their first char (675 entries); the 217,839 dead keys are never stored.
  - grapheme classes: the Unicode 17 GraphemeCat + InCB tables (src/gen/, from tests/data/unigram/grapheme17.txt)
    as a two-stage table, one byte per code point class.
  - vocab: a double-array byte trie over the remapped pieces (r -> 0x20), each terminal holding (id, score as
    the serde_json f64); token_to_ids as a hash of the same remapped bytes (for fused-unk lookups); the <0xXX>
    ids for byte fallback; unk_id and unk_score.
  - shortcuts (SPEC §2.7): certified whole-piece results (key: the remapped piece bytes, value: its ids) for
    the most frequent pieces, each computed by the scalar Viterbi at load and re-certified on image load; a
    per-scratch dynamic cache like K5's for the rest.

10.2 the scan (K7) and the lattice (K8), per segment:
  1. the scanner walks the segment: ascii identity runs by table (and simd: classify 64 bytes, find the next
     byte that is a separator, a control, CR, or >= 0x80); a non-ascii char takes its class from the two-stage
     table; the grapheme automaton (GB3-GB999, state = previous class, RI parity, emoji and InCB states) runs only
     from the char before a joiner / after a prepend / at CR, i.e. only where §3.3 can differ from per-char.
  2. it emits pieces as (virtual-r flag, start, end) in the caller's buffer, or (flag, start, end) in the scratch
     for a materialized piece.
  3. each piece: shortcut probe (static, then dynamic); on a miss the Viterbi over the piece's bytes with the
     virtual r first: best[] and back-pointers in the scratch (two arrays of len + 2), the trie walk per start
     reading the caller's bytes, candidates in the order of §6.2, strict >; backtrack; fuse unk; byte fallback.
  4. ids go straight into out (the bounce buffer only when cap forces it, kernels.md §7).
Work: scan O(n); Viterbi sum over pieces of len x (longest trie path from a start) <= len x 64 (SPEC §7.1's
c * m_i * L_max with L_max = the longest piece in bytes).

10.3 the piece cache and the walk's fast paths (unigram.c; spm-speed, 2026-10-04). Measured first (master b02ac89,
tools/bench/emb_stages.sh, bge-m3, gb10e X925): 43 MB/s en with NO piece cache (cold = pass = warm), half the time in
the walk (a dozen function calls per char), half in the Viterbi (83 ns a piece; the best[] ring indexed % (max_piece
+ 1), a divide per probe).
  - exactness: a piece's ids are hf's model.tokenize of the piece alone, the lattice starting from 0.0 (§6), so they
    are a function of (the virtual U+2581 or not, the remapped bytes) and nothing else. The key is exactly that:
    the bytes (<= 15, read in place or from the copy) and len in byte 15, bit 7 there the virtual symbol; a remapped
    string starting with 0x20 is keyed as (virtual, the rest), which is what the walk produces too (piece_add makes
    a piece's leading V the virtual symbol), so a vocab piece's static key and the text's key coincide. Under no
    remap (no Metaspace / Replace to U+2581) nothing is stripped. (Splitting a piece at a char boundary no piece
    spans is NOT exact here: the right part's sums would start from best[x], not 0.0, and f64 sums are not
    associative, §2.)
  - the dynamic cache: the scratch's piece cache (kernels.md §6-7: 64-byte buckets of two ways, epoch-tagged; the
    region every scratch has and no unigram context used), hashed with spm.h's one-multiply toks_spm_whash; a
    Viterbi answer of 1..4 ids fills it (K5's policy).
  - the static table (toks_uni.words, bpe.h's words-table format and probe): every vocab piece whose canonical key
    fits, valued by the Viterbi itself at load (exact by construction), inserted in descending score order so the
    most probable pieces get the slots (two buckets of two ways, pow2(n / 2 + 1) buckets): bge-m3 places 197,722 of
    its 250,002 pieces in 8 MiB (131,072 buckets), t5-base 26,591 of 32,100 in 1 MiB; a static hit fills the dynamic
    cache (K5's warm rule). Load (Mac, master fc5808a -> this, 4 runs): bge-m3 274 -> 282 ms, multilingual-e5-small
    272 -> 280 ms, t5-base 98 -> 103 ms.
  - probes overlap: an in-place piece of <= 15 bytes is queued (UNI_Q = 16) with its cache and static-table lines
    prefetched; the queue is answered in order when full, before a piece that cannot wait (copied, long), and at the
    segment's end.
  - the walk: ascii chars followed by an ascii char are graphemes of their own (GB4 / GB5 / GB999; a CR waits for
    its LF, GB3), so their charsmap entry decides them alone (ascii_run); for the census chain (toks_uni.fast: a
    charsmap, [collapse], [WhitespaceSplit], Metaspace(always, split), ids only) ascii_fast writes out the stages of
    a simple run (it only extends the open piece) and of a space on the segment state, the same steps norm_char ..
    piece_add take; a run of other simple chars (charsmap identity, grapheme class Any without InCB bits, not
    U+0020 / U+2581 / White_Space) is one bulk append but its last char, which can still join a mark.
  - the trie: 8-byte cells (base with the terminal bit, check; one load a step), placed depth first (a path's
    cells sit close), scores and ids in node-indexed arrays, padded by 256 cells so a step needs no bounds check;
    the Viterbi keeps each end's id (pieces < 512 bytes) so the backtrack walks nothing again; best[] is a ring of
    128 (& 127). The builder is da.h (shared with wordpiece.md §12.7).
  Speed (master fc5808a -> uni-speed, tools/bench spm_stages.c's ship variant A B B A, best of 3, MB/s cold / pass /
  warm), bge-m3, gb10e X925 cpu 9, 4 KiB chunks: en 43 -> 84 / 125 / 159, code 50 -> 111 / 128 / 152, ml 45 -> 73 /
  87 / 91, zh 81 -> 108 / 109 / 110; whole files: en 43 -> 124, code 50 -> 128, ml 45 -> 86, zh 81 -> 109 (cold).
  tr9970x Zen 5 cpu 16: 1.35-4.15x the same way. zh moves least: its pieces are long (58 bytes a piece), no key, so the
  Viterbi (8-byte cells, no re-walk) is all that changed for them.
  Profile after (master cdf19d1, emb_stages.sh, gb10e X925 cpu 9, 4 KiB, cold): zh 107 MB/s = walk 35%, Viterbi 65%
  (390 ns a 58-byte piece); ml 73 MB/s = walk 24%, model 76% (the Viterbi's own timer 56%); en 85 MB/s = walk 21%,
  model 79% (52%). Measured and not taken (bge-m3, master cdf19d1 -> the change, gb10e cpu 9, 4 KiB, ids= equal):
  - 8 starts' trie walks in lockstep (their cell loads overlap), then the lattice over their edges in the original
    order: zh 0.82x, ml 0.75x, en 0.79-0.87x, code 0.70x. The walk is not bound by misses: the cells it touches stay
    in L1 / L2, and the lanes' bookkeeping and unpredictable exits cost more than any overlap.
  - a table from a 2-3 byte char to the trie node after its bytes (U+0080..U+FFFF, 256 KiB, exact by construction:
    walked at load, "walk the bytes" where a piece ends inside them): zh 1.04x, ml / en / code 1.00x: the first
    steps of a start are not the cost either. Not worth 256 KiB a model.
  What is left in the Viterbi is per start: the char's length, the exit branch of each walk and the lattice's
  compare (both data-dependent), about 80 cycles a char on zh. The lever for ml / en cold is fewer Viterbi runs:
  more pieces answered at load (a trained word list valued by the Viterbi, as for spm in spm_bpe.md 6.6).


11. evidence
-------------

Through toks.h (2026-10-04, with the c twin of §10): the twelve loading pins vs hf 0.23.2 on the
light check (every 20th line of tests/parity/gen_cases.py --quick: encode in 6 modes, pieces, decode, stream;
gb10d), ~100.5k cases per file, 0 diffs; byte-fallback stream cases (3,000 per file of rare-char texts and
random byte-token runs cut into random pushes) on ruri3, llmjp3, llmjp4, bgem3, t5base, 0 diffs (20-30 E_LIMIT
cases per byte-fallback file, as the reference). make test runs uni_pmminilm and uni_llmjp4 in test_e2e.


12. what toks refuses, named
-----------------------------

  - NFC / NFKC then StripAccents or Lowercase ("normalizer NFC / NFKC then StripAccents or Lowercase (Unigram)": none
    in the census); two NF steps, or a step out of hf's order above ("normalizer order"); normalized added tokens
    under these steps
  - Replace with a regex other than ' {2,}', ' ', '(?<!\n)^' (decoder: '▁', '(?<!\n)^ ')
  - a charsmap whose double array reaches an out-of-range unit or a value off a char boundary (hf panics on some
    input), or whose strings are not utf-8 (hf refuses)
  - Metaspace prepend_scheme first after another pre-tokenizer (spm_bpe.md §4.1's offsets rule; no census Unigram
    tokenizer has it). Added tokens with lstrip / rstrip / single_word go through master's segment.c, as for
    every family.
  - a Unigram model with unk_id >= len(vocab) (hf refuses)
