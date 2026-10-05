Kimi K3: the reference semantics toks matches
=============================================

Kimi K3 (and K2, K2.5: §3.4) ships no tokenizer.json. Its tokenizer is the model's own python, run by
transformers: `AutoTokenizer.from_pretrained(dir, trust_remote_code=True)` loads `tokenization_kimi.py`, whose
`TikTokenTokenizer` (a transformers `PreTrainedTokenizer`) wraps a `tiktoken.Encoding` built from `tiktoken.model`.
SPEC §3.1: for tiktoken inputs the reference is tiktoken, here as the wrapper drives it. Everything below was
read in the pinned sources and then executed (§9); tests/model/kimi_model.py is this file as code.


1. pins
-------

  model        moonshotai/Kimi-K3 @ f831ab66814297da540d832a5235f8e904f29d06 (public, not gated)
                 tiktoken.model         sha256 b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103
                 tokenizer_config.json  sha256 5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e
                 tokenization_kimi.py   sha256 f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944
                 encoding_k3.py         sha256 49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676
                                        (imported by the wrapper: the chat renderer; needed to load it)
  python       3.12 (uv); str.isspace() is the same 29 code points in 3.10 .. 3.14 (§2.3)
  tiktoken     0.14.0 (2026-08-17). No Cargo.lock ships (neither the sdist nor the repo tag): the crates are read
               from the panic paths inside the wheels (macosx_11_0_arm64, manylinux_2_28_aarch64 and _x86_64, cp312,
               identical): fancy-regex 0.19.0, regex-automata 0.4.18, regex-syntax 0.8.11 (unicode 16.0.0),
               aho-corasick 1.1.5, memchr 2.8.3, bstr 1.13.1, pyo3 0.29.2. tools/oracle/Cargo.lock pins the same.
  transformers 5.18.0 (latest, 2026-09-30). vllm 0.30.0 requires >= 5.10.4; sglang 0.5.21 pins 5.12.1. The tokenize
               path (§2.2) is transformers' `PythonBackend` (tokenization_python.py).

Run everything with tools/oracle/py.sh (uv run --python 3.12 --with tiktoken==0.14.0 --with transformers==5.18.0).
tests/parity/oracle_tiktoken.py pins the four file hashes and the two versions, and fetches the files.


2. the call paths
-----------------

2.1 the stack. `TikTokenTokenizer.__init__` reads `tiktoken.model` with tiktoken's `load_tiktoken_bpe` (§3.1),
numbers 256 special ids from `len(ranks)` (§3.2) and builds `tiktoken.Encoding(pat_str, ranks, specials)` with the
pattern of §5. Every tokenizing path ends in the wrapper's `_encode_text_piece(text, allow_special_tokens)`:

    for i in range(0, len(text), 400_000):                        # TIKTOKEN_MAX_ENCODE_CHARS
        for sub in _split_whitespaces_or_nonwhitespaces(text[i:i + 400_000], 25_000):
            ids += model.encode(sub, allowed_special="all")       # allow_special_tokens=True
            ids += model.encode(sub, disallowed_special=())       # allow_special_tokens=False: every name is text

(tiktoken's `encode`: §7 finds the specials of `sub`, the pattern splits the text between them, §6 encodes each
piece.) The chunking is §2.3.

2.2 three ways in, as people call them.

  direct    `tok.encode(text)` -- no kwargs: the wrapper's own `encode` runs `_encode_text_piece(text, True)` over
            the whole text. This is what sglang 0.5.21 does for a non-fast tokenizer
            (TokenizerManager._tokenize_texts: `self.tokenizer.encode(t)`). `tok.encode(text,
            split_special_tokens=True)` and `tok(text, split_special_tokens=True)` give the same ids (transformers
            then skips the trie and calls `_tokenize(text)`, which is `encode(text)`, and tiktoken still
            recognizes every special).
  serving   `tok.encode(text, add_special_tokens=...)`, `tok(text, ...)`: any kwargs send the wrapper's `encode`
            to `PreTrainedTokenizer.encode`, and `__call__` never meets the wrapper's `encode` at all. transformers
            then: (1) splits the text on its added tokens with its `Trie` (§3.3); (2) runs `_tokenize(segment)`
            on each gap -- the wrapper's `[decoder[i] for i in encode(segment)]`, i.e. direct on the segment;
            (3) maps each string back with `_added_tokens_encoder`, then the wrapper's `encoder` (string -> id).
            The round trip (3) is the identity: the 163,840 decoder strings (`bytes_to_unicode` of each id's
            bytes, special names included) are distinct (checked at load by the model and the C reader's
            consequences: no special name is also a ranked token). vllm 0.30.0's completions endpoint calls
            `tokenizer(prompt, truncation=..., max_length=..., add_special_tokens=True)`
            (vllm/renderers/base.py `_tokenize_prompt`, renderers/params.py `get_encode_kwargs`).
  none      `tok.encode(text, allow_special_tokens=False)` (a named parameter, not a kwarg): direct with
            `disallowed_special=()`, every special name encoded as text.

  Nothing adds a BOS or EOS: `special_tokens_pattern` is unset, so `add_special_tokens=True` and `False` give the
  same ids (executed: oracle_tiktoken.py --self-check). Chat: vllm 0.30.0's KimiK3Renderer calls
  `apply_chat_template(..., tokenize=True)`, which encodes encoding_k3's segments one by one with
  `_encode_text_piece(segment.text, segment.allow_special)`: structural markers direct, user and tool text none.
  That is per-segment direct / none, which the front composes from toks' ALL / NONE calls.

2.3 where they differ: the cuts. The cuts count code points (python str indexes: not bytes, not utf-16 units)
and class them with python's `str.isspace()`, which holds U+001C..U+001F -- chars the pattern's \s does not
hold (below). `_split_whitespaces_or_nonwhitespaces(s, 25_000)` walks s with `run` = the length of the current
run of chars that agree on `str.isspace()`:
a char that flips the class starts a run of 1; a char that continues it makes run + 1, and when that exceeds
25,000 a cut falls before the char, which starts a new run of 1. So inside one long run of one class, cuts fall
after 25,000, 50,000, ... chars of it; a substring between cuts may hold many short runs. The 400,000-char
chunks come first and restart the count. Every cut is a hard boundary: tiktoken sees each substring alone (its
lookahead `(?!\S)` stops at the cut, §5) and a special name across a cut is two pieces of text, not the
special. A name is non-space, so it is part of a non-space run: in direct and none, "a" * 24998 + "[BOS]" + "b" * 10
is cut after 25,000 chars, between "[B" and "OS]", and [BOS] is not recognized; with "a" * 24990 or "a" * 25000 in
front it is. In serving the 16 trie names are out of the text before the counting, so they are always recognized,
but a cut can still split one of the 240 reserved names (<|reserved_token_163600|> after "a" * 24998: plain text
in both paths). All executed against the wrapper (oracle_tiktoken.py).

  str.isspace() (python 3.10 .. 3.14, unicode 13 .. 16, the same 29): U+0009..000D, U+001C..001F, U+0020, U+0085,
  U+00A0, U+1680, U+2000..200A, U+2028, U+2029, U+202F, U+205F, U+3000. The regex's \s (§4) is these minus
  U+001C..001F (25): U+001C..001F count as whitespace for the cuts and as punctuation for the pattern.

  direct and none cut the whole text; serving cuts each trie gap separately (the trie tokens are out of the text
  before the counting starts). The ids agree exactly when no cut falls in either: the text has <= 400,000 chars
  and no run of one isspace class longer than 25,000 (in direct; serving's segments are sub-runs of those). Where
  a cut falls they differ, e.g. "a" * 20000 + "[BOS]" + "b" * 10000: direct cuts the non-space run of 30,005 after
  25,000 chars (inside the b's), serving encodes "a" * 20000 and "b" * 10000 whole (oracle_tiktoken.py
  --self-check executes both).

  toks follows SPEC §3.2's mode table with these paths: ALL = serving (toks' default: transformers' own call,
  vllm's), NONSPECIAL = direct (transformers' split_special_tokens=True, sglang's encode), NONE = none. The cuts
  are offsets into the caller's buffer, never copies (decided 2026-10-04): segment boundaries the driver walks.

2.4 decode. No kwargs: tiktoken's `decode` -- every id's bytes (a special: its name's utf-8), then
`bytes.decode("utf-8", "replace")`: one U+FFFD per maximal ill-formed subpart (unicode §3.9), toks_decode's rule.
With kwargs (`skip_special_tokens=True`, ...): `PythonBackend._decode`: `convert_ids_to_tokens` drops
`all_special_ids` when asked -- the 13 named specials of §3.3, not the other 243 -- then the wrapper's
`convert_tokens_to_string` maps chars back through `byte_decoder` and decodes with "replace"; clean-up is off.


3. the files
------------

3.1 tiktoken.model. tiktoken's `load_tiktoken_bpe`: `contents.splitlines()` (LF, CR LF, CR), empty lines skipped,
`token, rank = line.split()` (any whitespace), `base64.b64decode(token)` (non-validating: foreign chars are
dropped), `int(rank)` (python syntax: signs, leading zeros, underscores), `ret[token] = rank` (a repeated token:
the last rank wins). `tiktoken.Encoding` then panics (pyo3 PanicException, a BaseException) on a repeated rank
(CoreBPE asserts encoder and decoder sizes agree) and `byte_pair_encode` panics on a byte with no token. Bite:
`load_tiktoken_bpe` caches every file, local paths too, under sha1(path) in $TMPDIR/data-gym-cache and serves that
copy on the next load of the same path: the oracle sets TIKTOKEN_CACHE_DIR="".

toks reads it strictly (src/core/tiktoken.c): "<canonical base64> <canonical decimal>" per line, exactly one space,
LF or CR LF ends, empty lines skipped; a token of 1 .. 65,535 bytes; a rank < 2^21 - 1; every token and every rank
once; all 256 one-byte tokens. Anything else is refused with its reason. Ranks may have gaps (holes: ids without
a token); the kimi wrapper, numbering its specials from len(ranks), needs them dense (§3.2).

The pinned file: 163,584 lines in rank order, ranks 0 .. 163,583 dense, canonical base64, LF ends, no CR; the 256
bytes at ranks 0 .. 255 in gpt-2's byte order ('!' first); tokens up to 256 bytes; 1,172 tokens are not utf-8 on
their own. 296,461 splits of a token into two tokens exist (every multi-byte token has one); in 40,748 of them a
half has a higher rank than the token.

3.2 the specials. `special_tokens = {added_tokens_decoder[i].content if present else f"<|reserved_token_{i}|>": i
for i in range(n, n + 256)}` with n = len(ranks) = 163,584: ids 163,584 .. 163,839. K3's config names 16 of them
(163,584 .. 163,593, 163,602 .. 163,605, 163,649, 163,838, 163,839); the other 240 are `<|reserved_token_i|>`.
tiktoken recognizes all 256 under `allowed_special="all"` (§7). No name occurs inside another and no name's
proper suffix is another's proper prefix (checked at load): every occurrence of every name is disjoint, so any
leftmost search -- tiktoken's alternation in hash order, transformers' trie, toks' K1 -- finds the same ones.

3.3 transformers' added tokens. `PreTrainedTokenizer.__init__` takes `added_tokens_decoder` (16 entries, all
lstrip / rstrip / single_word / normalized false) and adds any special-token attribute missing from it (bos
`[BOS]`, eos `[EOS]`, unk `[UNK]`, pad `[PAD]`, 9 additional_special_tokens: all already there). The trie splits
on those 16; 13 are special (`all_special_ids`: 163584-163586, 163590, 163591, 163593, 163602-163605, 163649,
163838, 163839); `<|open|>`, `<|close|>`, `<|sep|>` (163587-163589) are added but not special. The trie
(tokenization_python.py `Trie.split`) is a leftmost-longest scan with lookahead; for names whose occurrences are
disjoint (§3.2) it cuts at exactly every occurrence.

3.4 one wrapper, three models. toks identifies the wrapper by content, never by name: tokenization_kimi.py must
hold these lines (compared without surrounding blanks), the eight pattern lines consecutive and in order:

    num_reserved_special_tokens = 256
    r"""[\p{Han}]+""",   ...the eight pattern lines of §5...   r"""\s+""",
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000
    MAX_NO_WHITESPACES_CHARS = 25_000
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    if current_slice_len > max_consecutive_slice_len:
    allowed_special="all",
    disallowed_special=(),

and tokenizer_config.json must name `tokenization_kimi.TikTokenTokenizer` in auto_map. Kimi K2
(moonshotai/Kimi-K2-Instruct @ fd1984e2, wrapper b52cbff6) and K2.5 (moonshotai/Kimi-K2.5 @ 4d01dfe0, wrapper
56aae6f3) ship the same tiktoken.model (same sha256) and wrappers that pass (K2 indents the pattern differently;
both add `pre_tokenizer_process`, which returns [text]); their encode is K3's. Their configs differ: K2 has 17
added tokens (12 named specials), K2.5 has 23 (16), including the non-special `<think>`, `</think>` and the
tool-call markers -- read from the config like K3's (the C reader's counts equal transformers' on both).


3.5 transformers 5's configs. Re-saved repos (mlx-community/Kimi-K2.5, the K3 quantizations) carry the keys
transformers 5's save_pretrained writes: `backend` (must be "custom": the wrapper class), `is_local`,
`local_files_only`, `tool_parser_type`, `processor_class`, `padding_side` (none of them reaches encode or decode),
`model_specific_special_tokens` (must be empty) and `extra_special_tokens` as a list, which names specials as
`additional_special_tokens` does (mlx-community/Kimi-K2.5: 16 all_special_ids, the 4 attributes and the 12 names
of both lists). The special flag of an added_tokens_decoder entry never decides: nvidia/Kimi-K2.5-NVFP4 marks
`[PAD]` special while its pad_token is `[EOS]`, and transformers 5.18.0 leaves `[PAD]` out of all_special_ids (15
ids; skip_special_tokens keeps it), so toks' special set is the attributes' names. Probed with transformers 5.18.0
on both repos' files at main (tokenizer_config.json sha256 5c5a4469.. and d6cb65d9.., the census's).

4. classes
----------

Probed through tiktoken's own engine (tests/model/kimi_model.py `run_probes`: an Encoding whose 256 tokens are
the bytes returns the matched bytes of a pattern; every scalar value, surrogates excluded) for each set the pattern
writes, as it writes it:

  set                                                  count   as a class byte (layout.h; HAN: below)
  [\p{Han}]                                           99,030   HAN
  [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]         42,584   UPPER and not HAN
  [\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]               42,953   LOWER and not HAN
  [^\r\n\p{L}\p{N}]                                  969,123   base P or WS
  [^\s\p{L}\p{N}]                                    969,100   base P
  \p{N} 1,911;  \p{L} 141,028 (= Lu|Ll|Lt|Lm|Lo);  \p{M} 2,501;  \s 25 (§2.3)
  (?i:s) = {s, S, U+017F}; t m d r e v l fold to their two ascii forms only; the contraction group consumes
  exactly s S t T m M d D U+017F, and r R v V before e E, l L before l L.

Against toks' class tables (src/core/classes.c, the onig data of hf tokenizers 0.23.2, unicode 16.0): base class,
UPPER, LOWER, MARK and FOLD_S agree on every scalar (tools/oracle/classes_diff.py: 0 differences), and onig's
[\p{Han}] is the same 99,030 code points. Both engines read \p{Han} as Script=Han (not Script_Extensions). Han by
category (unicode 16.0): Lo 98,682 (the unified ideographs and extensions A .. I, the compatibility blocks), Lm 3
(U+3005 々, U+303B 〻, U+16FE3), Nl 13 (U+3007 〇, U+3021..3029, U+3038..303A), So 329 (the radicals
U+2E80..2E99, U+2E9B..2EF3, U+2F00..2FD5), Mc 2 (U+16FF0..16FF1), Po 1 (U+16FE2); 22 ranges.

Implemented (docs/templates/o200k.md §6 describes the same design): class bit 0x80 = TOKS_C_HAN (layout.h),
Script=Han, generated from tiktoken's own \p{Han} by tools/gen/han.py (src/gen/han_ranges.c) and set by classes.c
only under the class_flags bit TOKS_CLASSES_HAN, so every other template keeps bit 7 clear; the o200k template reads
it under TOKS_TP_HAN (0x80: rule K1, and UP / LO exclude HAN atoms) and TOKS_TP_NO_SLASH (0x100: the [\r\n]* tail);
kimi = CONTR_CI | DIGITS_1_3 | HAN | NO_SLASH = tmpl_params 0x182. The probes above back o200k.md §6's engine notes:
\p{Han} is Script=Han (U+3001 、, Common with Han in its Script_Extensions, is not in it), and every other class kimi
needs is the onig data toks already builds. Deltas between the engines: none today; a future tiktoken or tokenizers
bump re-runs tools/oracle/classes_diff.py, and any delta becomes a per-engine list of (code point, class byte)
overrides in the generated data, applied by classes.c for tiktoken models.


5. the pattern
--------------

    [\p{Han}]+
   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
   |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
   |\p{N}{1,3}
   | ?[^\s\p{L}\p{N}]+[\r\n]*
   |\s*[\r\n]+
   |\s+(?!\S)
   |\s+

o200k's pattern (docs/templates/o200k.md) with three changes: a Han run first, Han out of
the two case classes, and the punctuation tail [\r\n]* without '/'. tiktoken runs it with fancy-regex
(leftmost-first; fancy-regex hands its non-backtracking parts to regex-automata, same semantics). Written out
over atoms (kernels.md §2) and the classes of §4, with UP = UPPER and not HAN, LO = LOWER and not HAN, u(s), K(e),
L1(s), L2(s) exactly as o200k.md §3 defines them over UP / LO, at piece start i with atom c:

  K1 Han.          c is HAN -> [i, end of the maximal run of HAN atoms).
  K2 letters.      base(c) in {P, WS}: the first of L1(i+1), L1(i), L2(i+1) that applies (L1(i) applies exactly
                   when c is a mark); base(c) = L: L1(i), else L2(i).            (o200k O1)
  K3 digits.       base(c) = N -> [i, end of up to 3 consecutive N atoms).      (O2, DIGITS_1_3)
  K4 punctuation.  j = i if base(c) = P, j = i + 1 if c is U+0020 and base(c1) = P; -> [i, end of the maximal P
                   run from j), extended over the maximal run of CR / LF atoms.  (O3 with the tail {CR, LF})
  K5-K7 whitespace exactly kernels.md A5-A7 (o200k O4-O6).

A HAN atom is never a piece's prefix or case atom (K1 takes it first and UP / LO exclude it), but inside K3's run
a Han N atom (〇) is N, and inside K4's run a Han P atom (a radical) is P. Pieces from tools/oracle/notes.py (the
piece oracle; | separates pieces):

  " 中文"          ->  " " | "中文"           a space before Han is its own piece (K2 needs a case atom)
  "\u3000中"       ->  "\u3000" | "中"
  " ⺀x"           ->  " ⺀" | "x"            a radical is P after the K4 space ...
  "中⺀", "!⺀"     ->  one piece each         ... Han in a Han run, P in a P run
  "1〇〇〇"         ->  "1〇〇" | "〇"          Han digits join a digit run, start a Han run
  "x々y", "々x"    ->  "x" | "々" | "y",  "々" | "x"     (o200k: "x々y" whole: 々 is Lm)
  "A中B"           ->  "A" | "中" | "B"       (o200k: "A中" | "B")
  "中A's"          ->  "中" | "A's"
  "한국ABC", "ABC한국DEF"  ->  "한국" | "ABC",  "ABC한국" | "DEF"   (hangul is Lo, not Han: o200k's walk)
  "あい中う", "カタカナー中"  ->  "あい" | "中" | "う",  "カタカナー" | "中"
  "中文字。日本語です"  ->  "中文字" | "。" | "日本語" | "です"
  "０１２３"        ->  "０１２" | "３"
  "!\n/x"         ->  "!\n" | "/x"          (o200k: "!\n/" | "x")
  "\x1c\x1c a"    ->  "\x1c\x1c" | " a"      U+001C is P here, whitespace for the cuts (§2.3)
  "\u0301A!", "!\u0301", "!!\u0301", "'llama", "1's", "don't", "it'ſ", "camelCase", "HTMLParser"
                  ->  as o200k.md: "\u0301" | "A" | "!",  whole,  whole,  whole,  "1" | "'s",  whole,  whole,
                      "camel" | "Case",  whole

Lookahead stops at the end of the text tiktoken was given (§2.3, §7).

Where tiktoken raises. fancy-regex stops a match attempt at 1,000,000 backtracks or 1,000,000 stack entries;
tiktoken's encode then raises ValueError ("Regex error while tokenizing: ... Max stack size exceeded for
backtracking"). The bare engine does so on a whitespace run of about a million chars (tools/oracle/limits.py:
1,200,000 spaces fail; 400,000 pass; runs of letters, marks, Han, digits, punctuation or newlines of 1,200,000
pass). The wrapper's cuts keep every same-class run at <= 25,000 chars (a \s run is an isspace run), so none of
its paths raised on any input we found: limits.py's 1,200,000-space texts encode through direct, and §9's 6.0 M
cases had 0 reference errors. toks has no such limit (SPEC §7.3: no work-budget termination): for valid utf-8
it returns the ids of §5-§7, which is what tiktoken returns whenever it does not raise -- and through the wrapper,
on every input we know of, it does not.


6. bpe
------

tiktoken (src/lib.rs `byte_pair_encode` after `encoder.get(piece)`): a piece that is a token is that token
(toks: ignore_merges); a one-byte piece is its byte's token; otherwise parts start as single bytes and the
adjacent pair whose concatenation is a token of the lowest rank merges, the leftmost on ties, until no adjacent
concatenation is a token. Pieces under 100 bytes run the linear scan (`_byte_pair_merge`), longer ones a heap of
(rank, start) with stale entries skipped (`_byte_pair_merge_large`): the same order, so the same result (the
fuzz's long pieces, §9, and test_tiktoken's 677 tiktoken pieces include both). The rank of a pair is looked up
by bytes, so the merge table toks needs is every split (left, right) of every token whose halves are both
tokens, with priority = the merged token's id (ranks are ids): equal for all splits of one token, ties to the
leftmost pair -- K6 under TOKS_TF_IDS_AS_RANK. config.h's `ids_as_rank` makes the bpe builder take that reading for
tiktoken input although merged ids repeat (296,461 merges, 163,328 merged ids). Ordered by (merged id, split
position), raw ranks + rank2id give the same ids on 1,033,000 random (vocabulary, piece) pairs tried; the flag
keeps the exact rule and drops the rank -> id load. In Kimi's file every one of the 163,328 multi-byte tokens is
its own bpe (tools/oracle/selfmerge.py), so the whole-piece lookup never changes an id there; it stays (it is
tiktoken's rule, and it saves the merging).


7. specials inside a chunk
--------------------------

tiktoken's `encode(text, allowed_special)`: from `start`, find the leftmost special name (the alternation of all
256 names, escaped; a name not allowed is skipped by searching again from its start + 1), run the pattern on
`text[start:end]` only -- the regex sees end of input there, so `\s+(?!\S)` before a special takes the whole run
-- encode its pieces, emit the special, continue after it. `allowed_special="all"` allows all 256;
`disallowed_special=()` with the default allowed set allows none and checks none (none mode).


8. what toks needs, and where it stands
---------------------------------------

  reader     src/core/tiktoken.c (§3): toks_config with the 256 specials, ids_as_rank, ignore_merges, and
             toks_tiktoken_info: chunk 400,000 chars, runs 25,000, the trie bitmap (16), the named specials (13);
             toks_tiktoken_parse returns the kimi pattern (config.c TOKS_PATTERN_KIMI), the cuts (cfg->cut_chunk,
             cut_run) and the specials as two matchers: phase 0 = the 16 trie tokens, phase 1 = all 256 names
  load       toks_load(dir) / toks_load(dir/tiktoken.model) load (src/core/load.c -> the reader -> build_cfg)
  template   the o200k template's kimi variant (docs/templates/o200k.md §6): tmpl_params 0x182 = CONTR_CI |
             DIGITS_1_3 | TOKS_TP_HAN | TOKS_TP_NO_SLASH, class tables with TOKS_C_HAN (classes.c TOKS_CLASSES_HAN
             from src/gen/han_ranges.c, generated from tiktoken's own \p{Han} by tools/gen/han.py)
  driver     src/core/api.c: the cuts of §2.3 as offsets into the caller's buffer (ALL: phase 0 over the text,
             then each gap cut; NONSPECIAL and NONE: the whole text cut), phase 1 (every name allowed) inside each
             cut piece in ALL and NONSPECIAL, nothing in NONE; str.isspace's 29 code points for the run classes,
             atoms for code points (an ill-formed byte: one non-space); a segment of <= 25,000 bytes holds <=
             25,000 code points, so no cut, and skips the walk (run_cuts); a longer one walks 8 ascii atoms per
             step where no run limit or chunk edge can fall inside, and decodes a multi-byte atom only when its lead
             (C2, E1, E2, E3) can begin a space; decode's TOKS_SKIP_SPECIAL = the 13 named
             specials (compile's special bitmap)
  gate       tests/c/test_kimi.c in make test (29 texts at the cuts and at the walk's 25,000 / 25,001-byte edge, 3
             modes + decode, against the reference); tests/parity/run_kimi.py at scale (§9)


9. proof
--------

tests/model/kimi_model.py transcribes §2-§7 (classes from §4's probes, rules K1-K7, bpe, specials, cuts, trie,
modes; its own reading of the three files, never the oracle's objects). tests/model/run_kimi_fuzz.py diffs it
against tests/parity/oracle_tiktoken.py (the real wrapper) in serving, direct and none, and its pieces against
tools/oracle's tiktoken-pieces (fancy-regex 0.19.0 / regex-automata 0.4.18 / regex-syntax 0.8.11, tiktoken's own
regex stack; pattern-agnostic: o200k_base, cl100k_base, r50k_base run through the same binary). Sources: short
class-aware strings heavy on CJK (Han of every category beside kana, hangul, latin case pairs, Lt, Lm, marks,
digits of 11 scripts, CJK punctuation, the 25 \s and 4 isspace-only chars, apostrophes, special names and
fragments); exh = every string of 0 .. 4 atoms over 25 behaviours; vocab = 1 .. 30 Kimi tokens; long = runs of
24,999 .. 75,003 same-class chars, names straddling the 25,000 and 400,000 cuts, texts over 400,000 chars; real =
every line, paragraph and random 1 .. 16,384-char chunk of FLORES-200 dev + devtest (204 languages), whole
wikipedia articles as wikitext (zh, ja, ko, yue, classical chinese, wuu and 7 more) and gutenberg CJK classics.

  run   host        seed       exh       short      vocab    long   real      cases      mismatches  ref. errors
  run1  m2ultra2    20261004   406,901   1,000,000  200,000    300   494,767  2,101,968  0           0
  run2  m2ultra2    2          -         3,000,000  300,000  2,000   629,013  3,931,013  0           0
  run3  tr9970x     3          406,901   1,000,000  100,000    500   629,013  2,136,414  0           0
  total                                                                       8,169,395  0           0
  (each case: ids in 3 modes + pieces. m2ultra2: macos arm64, 24 cores, 20 jobs, load 1.8 / 4.3 before. tr9970x: linux
  x86-64, 8 jobs, load 47.9 from other users at the start; there the probes, the class claims and the self-check
  were re-run on the manylinux x86_64 wheel: same counts.)

Teeth: tests/model/mutants_kimi.py breaks one rule at a time (no Han run, '/' in the tail, Han in the case
classes, cuts on \s instead of isspace, run 25,001, chunk 400,001, serving without the trie, all 256 names in
the trie, bpe ties to the right, L2(i+1) before L1(i), 4 digits, specials off in direct, K5 losing the last NL):
a small fuzz (20,000 short, exh 3, 40 long, 2,000 vocab) kills all 13 (m2ultra2). Two candidates are equivalent and
left out: L1(i) before L1(i+1) (they end at the same atom whenever both apply) and dropping the whole-piece
lookup (§6). tests/c/mutants_tiktoken.py: 20 one-line mutants of src/core/tiktoken.c, all killed by
test_tiktoken.

tests/c/test_tiktoken.c checks the C reader against tiktoken's own reader (tests/data/kimi/gen.py: token bytes,
merges and names by sha-256, the trie and named ids) and K6 on 677 tiktoken pieces (`_encode_single_piece`),
plus hand, hostile and wrapper / config cases (`make test`).
