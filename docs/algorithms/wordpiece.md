wordpiece: the semantic contract (algorithm 4)
==============================================

SPEC §1.2's fourth algorithm: hf tokenizers' WordPiece model behind BertNormalizer and BertPreTokenizer, the
stack of every BERT-family encoder on the hub, plus what hf's default encode() does after the model: the truncation
and padding a tokenizer.json carries. hf tokenizers 0.23.2 (88a4498) is the definition; this file writes it out in
docs/kernels.md's style. tests/model/wordpiece_model.py is its executable reading and
tests/model/run_wordpiece_fuzz.py diffs that reading against hf (evidence in §10). When this file, the model and
hf disagree, hf is right and the other two are fixed.

  §1  the family: variants in use, ranked by downloads; the pinned files
  §2  segments: added tokens around the text model
  §3  normalizers: BertNormalizer (and NFD, Lowercase, StripAccents, Sequence)
  §4  pre-tokenizers: BertPreTokenizer (and WhitespaceSplit, Whitespace, Sequence)
  §5  the model: WordPiece
  §6  post-processors: TemplateProcessing, BertProcessing, RobertaProcessing
  §7  what encode() returns: truncation, post-processing, padding, and toks's flags
  §8  decode: the WordPiece decoder
  §9  notes that bite
  §10 evidence
  §11 what toks accepts and refuses
  §12 the c side: scan in place, fold in tables and registers, copy only what changes

Notation: a string is a sequence of unicode scalars (chars); "the vocab" is tokenizer.json's model.vocab (string ->
id), never the added tokens; cp is a code point; ranges are inclusive. "Unicode 8.0 Mn" means "the set the named
source lists", not a property of the current standard (§3.1).


1. the family
--------------

1.1 variants in use (census/coverage.json, (b) = the top 500 text-generation + top 100 embedding / reranker models)

22 distinct WordPiece tokenizers carry 45.38% of (b)'s downloads (59 models; package 1 of docs/coverage.md, with
truncation and padding: +42.28%). Every one of them is the same stack with different data:

  component        variant in use                                              tokenizers  share of (b)
  normalizer       BertNormalizer(clean_text, handle_chinese_chars,                    17       44.63%
                     strip_accents null, lowercase true)        "uncased"
                   BertNormalizer(... lowercase false)          "cased"                 5        0.75%
                   (strip_accents null follows lowercase: uncased strips accents, cased does not)
  pre-tokenizer    BertPreTokenizer                                                    22       45.38%
  model            WordPiece(unk "[UNK]", prefix "##", max_input_chars_per_word 100)   22       45.38%
                   vocab 30522 (10 tok, 42.06%), 30527 (3, 2.09%), 21128 (2, 0.66%), 119547, 501153 (2),
                   31090, 30528, 29514, 32000
  post-processor   TemplateProcessing [CLS] $A [SEP], ids 101 / 102                   15       42.92%
                   RobertaProcessing <s> $A </s>, ids 0 / 2                             2        1.88%
                   TemplateProcessing [CLS] $A [SEP], ids 2 / 3                         2        0.22%
                   TemplateProcessing <s> $A </s>, ids 0 / 2                            1        0.22%
                   TemplateProcessing [CLS] $A [SEP], ids 102 / 103 and 0 / 2           2        0.15%
  truncation       none                                                                11       20.30%
                   max_length 128, Right, LongestFirst, stride 0                        4       23.91%
                   max_length 512 / 250 / 256 / 300, same otherwise                     7        1.16%
  padding          none                                                                11       20.30%
                   Fixed(128), Right, pad_id 0 "[PAD]"   (all-MiniLM-L6-v2 alone)       1       22.05%
                   Fixed(128) pad_id 1 "<pad>"; Fixed(512); Fixed(250) x2               4        2.12%
                   BatchLongest (pad_id 0 or 1): a no-op for one sequence (§7.4)        6        0.91%
                   pad_to_multiple_of: null everywhere; direction: Right everywhere
  decoder          WordPiece(prefix "##", cleanup true)                                22       45.38%
  added tokens     5 specials ([PAD] [UNK] [CLS] [SEP] [MASK] or <s> <pad> </s> [UNK] <mask>), normalized false,
                   no options                                                          18       43.24%
                   + <mask> lstrip (mpnet family)                                       3        2.10%
                   + <unk> normalized true (all-mpnet-base-v2, multi-qa-mpnet)          2        1.88%
                   none at all (setu4993/LaBSE)                                         1        0.05%

Not in the census with WordPiece (each is specified below and modelled, toks refuses the ones §11 lists):
Whitespace / WhitespaceSplit / Sequence pre-tokenizers, Sequence[NFD, Lowercase, StripAccents] normalizers,
another prefix or max_input_chars_per_word, BertProcessing, truncation Left / OnlyFirst / OnlySecond / stride > 0,
padding Left / pad_to_multiple_of, single_word / rstrip added tokens. The same BERT text model (§3, §4) also fronts one
BPE tokenizer (openai-gpt, 0.02%: BPE with end_of_word_suffix); truncation and padding (§7) are not WordPiece
features at all: 32 tokenizers of every algorithm set truncation (31.69% of (b)) and 7 set Fixed padding (24.64%).

1.2 the pinned files (tests/wordpiece/pins.json: repo, revision, sha256; tests/wordpiece/fetch.py downloads and
    verifies them into $TOKS_TOKENIZER_CACHE/wp-<name>; anonymous hub only)

The ten most-downloaded distinct WordPiece tokenizers of (b) that ship a tokenizer.json, plus three extras:

  name               repo @ revision (12)                                       census id     (b)    norm   pp        trunc/pad
  minilm-l6          sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4       138bd678dcd0  22.05%  unc  [CLS]101  128 / F128
  bert-uncased       cross-encoder/ms-marco-MiniLM-L6-v2 @ 233902d25c44          3efa0e94a985  18.78%  unc  [CLS]101  - / -
  mpnet              sentence-transformers/all-mpnet-base-v2 @ e8c3b32edf54      621dd603b963   1.75%  unc  Roberta   128 / F128 id 1
  arctic-l           Snowflake/snowflake-arctic-embed-l @ d8fb21ca8d90           86398199810a   0.55%  unc  [CLS]101  512 / BL
  bge-small-zh       BAAI/bge-small-zh-v1.5 @ 7999e1d33597                       fcceb9bd867f   0.41%  cased [CLS]101 - / -
  text2vec-zh        shibing624/text2vec-base-chinese @ 183bb99aa7af (onnx/)     4ce95143ff04   0.25%  unc  [CLS]101  - / -
  paraphrase-mpnet   sentence-transformers/paraphrase-mpnet-base-v2 @ 6cc9279c672d 9f72c96c7c37 0.22%  unc  <s> 0     - / -
  distiluse-ml       sentence-transformers/distiluse-base-multilingual-cased-v1 @ 826fee3d516e
                                                                                 7e2ee938df13   0.19%  cased [CLS]101 - / -
  gte-large          Alibaba-NLP/gte-large-en-v1.5 @ 104333d6af6f                5e5000be8cd4   0.17%  unc  [CLS]101  512 / F512
  multiqa-mpnet      sentence-transformers/multi-qa-mpnet-base-dot-v1 @ 17997f24dca0 5b756c4ddcd7 0.13%  unc  Roberta   250 / F250 id 1
  labse              sentence-transformers/LaBSE @ 836121a0533e                  f1ea497baf73   0.06%  cased [CLS]101 - / -   (vocab 501153)
  ko-sroberta        jhgan/ko-sroberta-multitask @ 8fca7c9c98c2                  bd96ac8021e2   0.05%  cased [CLS]0   128 / BL id 1
  bert-base-uncased  google-bert/bert-base-uncased @ 86b5e0934494                3efa0e94a985  (SPEC §1.1 (c) "bert"; other bytes,
                                                                                               same hf-normalized config as bert-uncased)

  added 2026-10-04: the census's other wordpiece tokenizer.json files, so the pins hold 20 of its 22
  wordpiece ids (every one that ships a tokenizer.json):
  qdrant-minilm-l6   Qdrant/all-MiniLM-L6-v2-onnx @ d13954661f83                 e216dd9d1bf6   0.11%  unc  [CLS]101  256 / BL
  specter2           allenai/specter2_base @ 3447645e1def                        9c2440a506c3   0.10%  unc  [CLS]101  - / -   (vocab 31090)
  pubmedbert         NeuML/pubmedbert-base-embeddings @ b79526d6ef36             4338400377e7   0.09%  unc  [CLS]101  512 / BL
  jina-v2-small      jinaai/jina-embeddings-v2-small-en @ 44e7d1d6caec           64b96dc078ad   0.07%  unc  [CLS]101  - / -   (vocab 30528)
  multiqa-minilm     sentence-transformers/multi-qa-MiniLM-L6-cos-v1 @ b20736733232 11e120f7b013 0.07%  unc  [CLS]101  250 / F250
  paraphrase-minilm-l3 sentence-transformers/paraphrase-MiniLM-L3-v2 @ 4ca70771034a e0b329d86937 0.06%  unc  [CLS]101  128 / BL
  labse-setu         setu4993/LaBSE @ 5afa72968dfe                               e09ab19b4e7b   0.05%  cased [CLS]101 - / -   (no added tokens)
  msmarco-bert       sentence-transformers/msmarco-bert-base-dot-v5 @ dbf04e3911e5 8084d71a96af 0.04%  unc  [CLS]101  300 / BL
  google-bert/bert-base-uncased's model has no "type": hf reads it through its untagged model enum (BPE needs
  merges, WordPiece its four fields), and so does config.c (legacy_wordpiece).

  (cambridgeltl/SapBERT-from-PubMedBERT-fulltext, 0.13%, and YituTech/conv-bert-base, 0.06%, ship only vocab.txt:
  input:slow-wordpiece, package 10.)
  arctic-l's bytes equal datasocietyco/bge-base-en-v1.5-course-recommender-v5 @ 2b069eed51ce, the census's most
  downloaded holder of that config.


2. segments: added tokens around the text model
------------------------------------------------

Exactly docs/kernels.md §4 and the driver's policy (hf added_vocabulary.rs), restated where WordPiece files bite:

  2.1 ids. hf AddedVocabulary::add_tokens runs once at load over the file's list in order: an empty content is
      dropped; a content already present with identical options is skipped; otherwise the id is the content's
      existing added id, else its model vocab id, else the next id counting up from the NUMBER of vocab strings;
      the JSON "id" is ignored. special_tokens_set only grows (a later non-special duplicate does not remove it).
  2.2 phase 0. Tokens with normalized=false are found in the raw text: leftmost-longest, non-overlapping
      (daachorse LeftmostLongest), resuming at the raw match end; then mode (NONSPECIAL drops specials), single_word
      (rust regex \w on the neighbours), lstrip (start moves left over White_Space, never before the previous split's
      end), rstrip (end moves right over White_Space). The gaps between accepted matches are text.
  2.3 phase 1. Each phase-0 gap is normalized BY ITSELF (§3); tokens with normalized=true are found in the normalized
      gap by their normalized form (the content run through the same normalizer; the content itself when that is
      equal). all-mpnet-base-v2's <unk> is normalized=true under an uncased normalizer: "<UNK>", "<Unk>" and "<unk>"
      in the text all encode as <unk> (3).
  2.4 empty splits vanish: hf's PreTokenizedString::split drops every empty split, token splits included (a match
      whose lstrip / rstrip clamp leaves start == stop yields no id); start > stop panics in hf ("AddedVocabulary bad
      split"). Two tokens of one phase with the same form make hf refuse the file (daachorse: duplicate pattern).
  2.5 mode NONE is the file with added_tokens: [] (SPEC §3.2): no matching at all; post-processor ids stay.

Each text split then goes through §4 then §5; token splits are one id each.


3. normalizers
---------------

3.1 the four sources. hf's BertNormalizer reads per-code-point facts from three places, of three unicode versions;
    none of them is python's unicodedata. tools/gen/bert_tables.py downloads the exact sources (sha256-pinned), and
    checks every derived table against hf itself on every one of the 1,112,064 scalar values (0 mismatches):

  unicode_categories 0.1.1 (crate; its tables equal Unicode 8.0.0's UnicodeData.txt exactly)
       Cc (65), Cf (150), Co (137,468: three ranges + the table), Mn (1,567), P = Pc Pd Pe Pf Pi Po Ps (717)
  unicode-normalization-alignments 0.1.12 (crate; UNICODE_VERSION 9.0.0)
       canonical decompositions (2,060 + hangul arithmetic), canonical combining classes (814), and
       is_combining_mark = General_Category M (2,097; only the standalone StripAccents normalizer reads it)
  rust std of the wheel's compiler (Unicode 17.0.0: UCD 17.0.0 reproduces every probe; 16.0 misses 28 chars)
       char::to_lowercase = the simple lowercase mapping + SpecialCasing's unconditional entries (only U+0130 ->
       "i" U+0307 is longer than one char); char::is_whitespace = White_Space (25 cps: U+0009-000D, 0020, 0085,
       00A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000)
  hard-coded in bert.rs
       is_chinese_char: 4E00-9FFF 3400-4DBF 20000-2A6DF 2A700-2B73F 2B740-2B81F 2B920-2CEAF F900-FAFF 2F800-2FA1F
       (not 2B820-2B91F, not 2CEB0+, not 30000+: CJK extensions E (part), F, G, H, I are NOT "chinese");
       char::is_ascii_punctuation: the 32 ascii chars !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ (symbols like $ + < = > ^ ` |
       ~ included)

    The tables live in tests/data/wordpiece/unicode.json (the model's data) and src/gen/bert_tables.{c,h} (the c
    twin's two-level tables); the generator prints both files' sha256.

3.2 BertNormalizer {clean_text, handle_chinese_chars, strip_accents, lowercase}. On a string s, in this order, each
    step over the whole string (hf normalizers/bert.rs):

  N1 clean_text: drop every char c with c = U+0000, c = U+FFFD, or (c is Cc, Cf or Co per §3.1 and c is not \t \n
     \r); then replace every remaining char c with ' ' when c is \t \n \r or White_Space. Removal comes first: \v
     (U+000B), \f (U+000C) and U+0085 are White_Space AND Cc, so they vanish ("a\vb" -> "ab", one word), while
     U+00A0, U+3000, U+2028 become ' '. Cn (unassigned) chars are kept (the crate's is_other omits Cn although
     bert.rs's comment says otherwise); U+200B, U+200D (ZWJ), U+FEFF, U+00AD, U+180E (Cf in 8.0) vanish.
  N2 handle_chinese_chars: replace every chinese char c (§3.1) with ' ' c ' '.
  N3 strip_accents (a missing / null value means: the value of lowercase): NFD with the 9.0 tables (full canonical
     decomposition of every char, hangul arithmetically, then a stable sort by combining class of every maximal run
     of chars with class > 0), then drop every Mn (8.0) char.
  N4 lowercase: replace every char by char::to_lowercase (17.0), per char, no context (final sigma is never
     applied: "ΟΔΟΣ" -> "οδοσ"); U+0130 -> "i" U+0307.

  Facts the rest of this file leans on (each asserted by the generator or the fuzz):
  - every step is a function of one char except N3's sort, and N3's sort only reorders chars that N3 keeps when two
    of them meet in one run: the 83 "kept non-starters" (class > 0 in 9.0 but not Mn in 8.0: 14 Arabic marks
    U+08D4-08E1, 38 Glagolitic combining letters U+1E000-1E02A, 7 Adlam marks U+1E944-1E94A, viramas and nuktas
    of Balinese, Sundanese, Batak, Rejang, Javanese, Sharada, Khojki, Grantha, Newa, Takri, Bhaiksuki, the Hangul
    tone marks U+302E/F, 8 musical symbols U+1D165-1D172, U+1DFB). A ccc-0 Mn char ends a run even though N3 then
    drops it; a char N1 removed is gone before N3 runs, so a run continues across it.
  - no White_Space, punctuation or chinese char has a combining class > 0, so no run crosses a piece boundary (§4).
  - one input char grows to at most 3x its utf-8 bytes (hangul syllable with a final: 3 -> 9 bytes as three jamo).
  - ascii: N2, N3 never change an ascii char; N1 drops 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F and maps \t \n \r ' '
    to ' '; N4 maps A-Z to a-z.

  Consequences people trip over (all from hf, all in the fuzz):
  - uncased models see hangul as jamo: "한국어" -> ᄒ ##ᅡ ##ᆫ ##ᄀ ##ᅮ ##ᆨ ##ᄋ ##ᅥ; "İstanbul" -> "istanbul" (N3
    runs before N4, so U+0130 is decomposed and its dot stripped); "Å" (U+212B) -> "a"; "ẞ" -> "ß"; "ｆ" stays
    fullwidth (NFD is canonical, not compatibility); U+FE0F (Mn) and U+0F73's two marks vanish; U+1FEF (Sk) becomes
    "`" and ≠ ≮ ≯ (U+2260, U+226E, U+226F: Sm) become "=" "<" ">" (their U+0338 is stripped): ascii punctuation, so
    §4 isolates them ("a≠b" -> a = b; cased: one piece); U+F900-FAFF become their unified ideographs.
  - cased models keep accents and case; N1 and N2 still apply.

3.3 the other normalizers (not with WordPiece in the census; modelled and fuzzed as variants):
    NFD: N3's decomposition and sort without the drop. Lowercase: N4. StripAccents: drop every General_Category M
    char of 9.0 (is_combining_mark: Mn, Mc AND Me, a different set from N3's Mn of 8.0; no NFD inside).
    Sequence: each in turn. Any other normalizer: §11.


4. pre-tokenizers
------------------

Each runs on one normalized text split; empty pieces are dropped (hf PreTokenizedString::split).

  BertPreTokenizer   split on every White_Space char (removed), then isolate every char c with
                     is_ascii_punctuation(c) or c in P (8.0): each is a one-char piece. Everything else joins the
                     surrounding run. Classes are of the NORMALIZED chars ("`" made from U+1FEF is isolated).
                     Not punctuation: Sc/Sm/Sk/So outside ascii (€ ∑ ° ™ emoji), P chars added after 8.0 (U+2E43,
                     U+1E95E, ...). Punctuation: § ¶ « » “ ” ‘ ’ ‐ – — ¿ ¡ 、 。 ・ ‿ and the ascii 32.
  WhitespaceSplit    split on White_Space (removed).
  Whitespace         the matches of \w+|[^\w\s]+ (rust regex, unicode \w: 144,667 scalars, probed): maximal
                     runs of word chars and maximal runs of chars that are neither word nor White_Space; White_Space
                     runs vanish.
  Sequence           each in turn on every piece of the previous one.


5. the model: WordPiece {vocab, unk_token, continuing_subword_prefix, max_input_chars_per_word}
-----------------------------------------------------------------------------------------------

hf models/wordpiece/mod.rs, on one piece w (a non-empty string):

  W1 if w has more than max_input_chars_per_word CHARS (scalars of the normalized piece, not bytes): [unk].
  W2 start = 0. While start < len(w):
       for end = len(w), len(w)-1 char, ..., start+1 char:
         key = w[start:end] when start = 0, else prefix + w[start:end];
         if key is in the vocab: emit vocab[key]; start = end; next start.
       no end matched: the WHOLE piece is [unk] (sub-tokens found so far are discarded).
  W3 [unk] is vocab[unk_token]; when unk_token is not in the vocab, hf fails the encode (MissingUnkToken) exactly
     when W1 or W2 needs it (and only for pieces tokenized at all, §7.2).

  Properties: greedy longest-prefix match, never a dynamic program. A piece that is itself in the vocab (and short
  enough for W1) is exactly [vocab[piece]]: the first key tried is the whole piece. A vocab string starting with the
  prefix can match at start 0 too ("##" itself is a vocab string; BertPreTokenizer never produces it because "#" is
  punctuation, Whitespace / WhitespaceSplit can). The prefix may be "" (then continuation keys are plain
  substrings). Duplicate vocab keys: the json's last value; ids may repeat or have holes (decode then takes an
  arbitrary string of a repeated id: hf builds vocab_r from a hash map).


6. post-processors (single sequence; add_special_tokens = post-processing on)
-----------------------------------------------------------------------------

  TemplateProcessing   the single template's pieces in order: $A is the sequence, each SpecialToken expands to
                       special_tokens[name].ids (several ids possible); n_added = the number of those ids.
  BertProcessing       [cls] $A [sep]; n_added = 2.
  RobertaProcessing    [cls] $A [sep]; n_added = 2 (trim_offsets / add_prefix_space change offsets only).
  none                 $A; n_added = 0.
  Post-processing off: $A alone, whatever the processor.


7. what encode() returns: truncation, post-processing, padding
---------------------------------------------------------------

hf encode(text, add_special_tokens) = encode_single_sequence (§2-§5, with the early exit 7.2) then post_process
(truncate 7.3, post-process §6, pad 7.4). toks returns exactly Encoding.ids of that result.

7.1 parameters. truncation {max_length, strategy LongestFirst | OnlyFirst | OnlySecond, stride, direction Left |
    Right}; padding {strategy BatchLongest | Fixed(n), direction Left | Right, pad_to_multiple_of, pad_id, pad_type_id,
    pad_token}. Every field is required in tokenizer.json (serde, no defaults) except truncation.direction (default
    Right).

7.2 the early exit (do_tokenize -> tokenize_with_limit). With truncation set and not (strategy OnlySecond on this
    single sequence), the model runs over the splits in order (Right: first to last; Left: last to first) keeping a
    running count of ids: an added-token split counts 1 and never stops the walk; after each text piece is tokenized,
    the walk stops when count >= max_length (the RAW max_length, n_added not subtracted). Splits not reached are
    dropped and never tokenized (a missing unk there cannot fail the encode). Whenever the walk stops, at least
    max_length ids were kept, so for max_length >= n_added this changes nothing beyond 7.3; it shows when 7.3's
    subtraction wraps.

7.3 truncate. L = max_length - n_added when post-processing is on and n_added > 0, else max_length; the
    subtraction is usize arithmetic in a release build: it WRAPS (max_length 1, n_added 2 -> L = 2^64 - 1).
      L = 0                  -> no ids (all overflow); stride is not checked.
      count <= L             -> unchanged.
      strategy OnlySecond    -> hf fails the encode ("Second sequence not provided").
      stride >= L            -> hf panics ("stride must be strictly less than max_len"; PanicException).
      otherwise              -> Right keeps the first L ids, Left the last L. The rest is Encoding.overflowing
                                (windows of L ids advancing by L - stride), which encode().ids never contains:
                                toks never computes it.
    toks runs Right (every file of the census) and refuses Left, OnlySecond, a stride hf would panic on and a
    max_length below the template's ids (§11). With Right, the first L ids of the whole encoding are the answer
    whatever 7.2 tokenized, so toks stops the walk as soon as L ids exist: a 1 MB document under all-MiniLM-L6-v2
    tokenizes ~126 words, not the megabyte.
    LongestFirst and OnlyFirst are the same for one sequence. Examples (all-MiniLM-L6-v2's file, [CLS] / [SEP]
    around, text "hello world, unbelievably tokenization works", 11 ids):
      max_length 0 (wraps)    -> [CLS] hello [SEP]   (7.2 stops after "hello"; 7.3 keeps everything)
      max_length 1 (wraps)    -> [CLS] hello [SEP]
      max_length 2            -> [CLS] [SEP]          (L = 0)
      max_length 5            -> [CLS] hello world , [SEP]
      max_length 6, stride 4  -> panics (L = 4)
      post-processing off, max_length 0 -> []         (L = 0)

7.4 pad. target = n (Fixed) or the encoding's own length (BatchLongest, one sequence); then, when
    pad_to_multiple_of = m > 0 and target % m > 0, target rounds up to the next multiple of m. If the length is below
    target: (target - length) copies of pad_id are appended (Right) or prepended (Left). Never truncates. Padding
    applies with post-processing on AND off. BatchLongest without pad_to_multiple_of is a no-op for one sequence
    (hf encode_batch pads a batch to its longest: a batch api concern, not encode()).

7.5 what toks's flags mean here (flags 0 = hf encode(text), SPEC §0.3):
      mode ALL / NONSPECIAL / NONE   only §2's matching changes; truncation and padding are the same.
      TOKS_NO_POSTPROCESS            add_special_tokens=False: n_added does not reduce max_length (7.3), no
                                     template ids, padding still applies.
      TOKS_NO_TRUNCATE               hf no_truncation(): 7.3 is skipped (7.2's walk runs to the end of the text).
      TOKS_NO_PAD                    hf no_padding(): 7.4 is skipped.
      TOKS_CONTINUATION              a later part of one document (SPEC §5.2): truncation and padding are
                                     whole-document steps, applied once by whoever assembles the parts; a
                                     continuation call applies neither (the template follows TOKS_NO_POSTPROCESS, as
                                     for every algorithm). The BERT stack has no start-of-input behaviour otherwise.
    The ids toks returns with flags 0 include the pad ids: all-MiniLM-L6-v2 always returns 128 ids. Pad ids are
    ordinary ids (all-MiniLM-L6-v2 pads with 0 = "[PAD]", which a "[PAD]" literal in the text also produces), so a
    caller that needs hf's attention_mask cannot recover it from the ids alone: it encodes with TOKS_NO_PAD (and
    TOKS_NO_TRUNCATE for the whole text), takes the count as the mask's ones, and pads itself from toks_info's
    pad_id, pad_len / pad_multiple and pad_left. flags 0 stays hf's default. A file that truncates or pads has no
    certified cut (toks_split_points; docs/usage.md "Cutting text") unless the call opts out of both, so toks_par splits
    its big inputs only then.


8. decode: the WordPiece decoder {prefix, cleanup}
---------------------------------------------------

hf decode(ids, skip_special_tokens):
  D1 each id -> its string: an added token's normalized form when one was cached (2.3), else its content; else the
     vocab string; ids with neither are skipped (toks: SPEC §3.4 returns TOKS_E_ID for an id beyond the table).
  D2 skip_special_tokens: drop strings in special_tokens_set (by string: a normalized=true special whose form differs
     from its content is NOT dropped).
  D3 the decoder, per string i (no decoder: join with " "): for i > 0, the prefix is removed when the string starts
     with it, else " " is prepended; then with cleanup, on that string alone, in this order: " ." -> ".", " ?" ->
     "?", " !" -> "!", " ," -> ",", " ' " -> "'", " n't" -> "n't", " 'm" -> "'m", " do not" -> " don't", " 's" ->
     "'s", " 've" -> "'ve", " 're" -> "'re". Then everything is concatenated.


9. notes that bite (collected)
-------------------------------

  - three unicode versions in one normalizer (§3.1): 8.0 categories, 9.0 NFD, 17.0 case; and CJK ranges that stop
    at extension F. Probe, never assume (python's unicodedata disagrees with each of them).
  - U+0085, \v, \f vanish under clean_text instead of splitting (Cc beats White_Space); without clean_text they
    split. U+200B / ZWJ / soft hyphen join the words around them ("a\u200bb" -> one piece "ab").
  - strip_accents null = lowercase: uncased strips, cased keeps. Its filter is Mn of 8.0 (newer marks survive, Mc
    and Me survive); StripAccents the normalizer filters all of M of 9.0.
  - N3 before N4: "İ" -> "i" uncased, but "i̇" with strip_accents false + lowercase true.
  - punctuation is is_ascii_punctuation OR P (8.0): ascii symbols ($ + < = > ^ ` | ~) are punctuation, non-ascii
    symbols are not; classes are taken AFTER normalization.
  - max_input_chars_per_word counts chars of the normalized piece (101 "é" uncased = 101 "e" -> [unk]; 51 "e" +
    U+0301 = 51 chars after N3 -> tokenized).
  - one unmatched position makes the whole piece [unk] (W2), not just the rest.
  - truncation: L subtracts n_added only with post-processing on, wraps below n_added, panics on stride >= L, and
    the early exit tokenizes only the words it needs. Padding still applies with post-processing off.
  - BatchLongest is a no-op for one sequence; Fixed(n) never truncates.


10. evidence
------------

  tables     tools/gen/bert_tables.py (macOS, tokenizers 0.23.2): every step of §3.2 alone, the uncased, cased and
             strip+lower BertNormalizer, NFD, Lowercase and StripAccents on all 1,112,064 scalars, 0 mismatches;
             48,223 reorder sequences (every ordered pair of the 83 kept non-starters, alone, around a removed char,
             around stripped non-starters and ccc-0 marks), 0 mismatches; BertPreTokenizer and WhitespaceSplit classes
             of every scalar, 0 mismatches.
  model      tests/model/run_wordpiece_fuzz.py: its full runs' receipts (host, load, counts) are not in this tree.


11. what toks accepts and refuses (config.c; every census WordPiece file loads)
--------------------------------------------------------------------------------------------

  accepts    BertNormalizer (all flag combinations) or none; BertPreTokenizer; WordPiece with any prefix and
             max_input_chars_per_word <= 1024; TemplateProcessing / BertProcessing / RobertaProcessing or none; the
             WordPiece decoder (prefix <= 16 bytes, cleanup on or off) or none; truncation Right (LongestFirst,
             OnlyFirst, any stride hf does not panic on); padding Fixed / BatchLongest, Right or Left, any
             pad_to_multiple_of <= 2^20; added tokens with lstrip / rstrip / normalized.
  refuses    (TOKS_E_UNSUPPORTED, the feature named)
             - unk_token not in the vocab ("WordPiece unk_token not in vocab": hf would fail the encode of some
               inputs; toks has no per-input failure for a model fact)
             - truncation stride >= the smallest L it can meet (max_length - n_added and max_length) when that L > 0
               (hf panics on long inputs); max_length below the template's ids (hf's subtraction wraps)
             - truncation OnlySecond (hf fails every long single-sequence encode); truncation direction Left (no
               census file; its answer is the LAST L ids, which needs the whole text)
             - a normalized added token whose form is empty or longer than 255 bytes
             - max_input_chars_per_word above 1024 (the scan's copy buffer is sized by it)
             - normalizers other than BertNormalizer, pre-tokenizers other than BertPreTokenizer, other decoders,
               single_word added tokens: the python model covers them (§3.3, §4); the c path does not (none is in
               the census with WordPiece)
  hf refuses (TOKS_E_FORMAT): missing model / normalizer / decoder / truncation / padding fields, two normalized
             added tokens with the same form (daachorse), unknown strategies or directions.


12. the c side (src/core/wp*.c, norm.c): scan in place, fold in tables and registers
-----------------------------------------------------------------------------------------

the maintainer doctrine's speed rule applied to this stack: the caller's text is read in place, the ascii part of the
normalizer is a table lookup on the byte the scanner already holds, and text is materialized only for the pieces
§3 really rewrites.

12.1 one scan does N1-N4 and BertPreTokenizer together. Every char is classified once:
       byte < 0x80: a 128-entry per-tokenizer class table (split / punct / removed / fold / word), built at load
       from the four flags; the asm tiers classify 16-64 bytes at a time with the same table (nibble lookups).
       byte >= 0x80: decode, then toks_bert_cls(cp) (src/gen/bert_tables.h: REMOVE WS CJK PUNCT DECOMP LOWER NS MN).
     The scanner keeps one open piece: its start offset in the text, its length, its char count, and a flag. Split
     chars close it; punctuation (and chinese chars under N2) close it and emit a one-char piece; word chars extend
     it; ascii A-Z under N4 extend it and set FOLD. All of these are offsets into the caller's buffer: no byte is
     copied.
12.2 copy on write. A piece is materialized (into scratch, at most 3x its bytes) only when something inside it is
     not "the raw bytes, ascii-folded": a char N1 removes in the middle of a word, a char whose N3 / N4 output
     differs from itself (DECOMP, MN, LOWER under the config), a kept non-starter under N3 (the run sort). From that
     char on, the piece's normalized chars are written to scratch (its in-place prefix copied once); the chars the
     rewrite produces are classified again, so a "`" made from U+1FEF still splits. Pieces never interact (§3.2: no
     run crosses a boundary), so materialization is per piece. English text materializes nothing; accented Latin
     materializes the accented words only.
12.3 the model reads pieces where they are. The whole-piece lookup and every greedy candidate hash the piece's bytes
     with 8-byte loads, fold A-Z in the register (SWAR: m = (x + 0x3F..) & ~(x + 0x25..) & ~w & 0x80..; w | m >> 2,
     with x = w & 0x7F..), and compare against the vocab bytes the same way: the lowercased word never exists in
     memory. Continuation candidates look up the "cont" table keyed by the vocab strings without their prefix, so
     "##" + substring is never built either.
12.4 certified shortcut. The whole-piece lookup IS a certified table: by W2, a vocab string that W1 admits encodes as
     itself, so the vocab hash answers most pieces in one probe; the greedy longest match (W2) runs only on misses,
     with candidates capped at the longest key (18 bytes for the BERT vocab) and prefix hashes shared across
     candidates (the hash chains over 8-byte words and mixes the length last).
12.5 the bounds the driver rests on (tests/c/test_wp.c proves both on every scalar and flag set): one char normalizes
     to at most 3 bytes per byte of it (Hangul under strip_accents is the worst: 3 bytes -> 3 jamo of 3 bytes) and to
     at most as many chars as it has bytes. Hence a normalized gap of n bytes needs <= 3n bytes, and the ids of any
     stretch of text are at most its raw byte count (an id takes >= 1 char of a piece, a piece's chars come from its
     raw bytes): a round of pieces covering raw bytes [pos, end) writes at most end - pos ids, and a whole call at
     most len + the template + padding.
12.6 the driver (wp_api.c: toks_encode / toks_pieces / toks_decode of a WordPiece context).
       encode   phase 0 (segment.c, K1) over the raw text; each gap takes one of two paths, the same ids:
                  in place     no phase-1 token can match (the file has none, or the mode drops them): the scan of
                               12.1 over the caller's bytes, rounds of <= 256 pieces, K9 writes ids straight into the
                               caller's buffer when 12.5's bound fits its room, else into the scratch bounce
                  normalized   phase-1 tokens exist (all-mpnet-base-v2's <unk>): the gap is normalized into scratch
                               (toks_norm), phase 1 matches there, and the scan runs the pre-tokenizer alone
                               over the normalized bytes (scan flags 0)
                then 7.3 with the early exit (the walk stops once L ids exist; a round's scan is capped at the ids
                still wanted), the template, 7.4 (Left padding shifts what was written).
       pieces   always the normalized path: piece ends are offsets in the normalized stream (include/toks.h), each
                token unit counting its raw bytes, each gap its normalized length.
       decode   §8: D1's strings are the compiled token strings (vocab strings; an added id's form), D2's skip set
                is compiled by string (compile.c), D3's cleanup runs as eleven chained streaming str::replace passes
                (each holds <= 7 bytes: leftmost-first, non-overlapping, the next pass sees only final bytes), with a
                fast path for strings without an inner space (one pattern at offset 0 at most).
       stream   hf's decoder is per token (D1-D3 see one token's string and whether it is the first kept one), so
                no later id changes an emitted byte: toks_stream_push emits its ids' decode (toks_wp_decode_k, the
                kept count as the stream's state) and holds nothing; the flush emits nothing (stream.c).
12.7 the answer tables and the tries (wp.c, wp_scan.c; spm-speed, 2026-10-04). Measured first (master b02ac89,
     tools/bench/emb_stages.sh, bert-uncased = all-MiniLM-L6-v2's model without its 128-id truncation, gb10e X925):
     the model took 72-87% (en 38 ns a piece at 94% whole-word hits: crc32c + fmix + an entry line + a keys line;
     ml's 35% misses ran the greedy probes, ~250 ns each), the scan 13-27%.
       whole words   t->wtab: every vocab string of 1..15 bytes in bpe.h's words-table format (K5's key: the bytes,
                     len in byte 15; under lowercase the key is folded in the register, so a vocab string holding A-Z
                     can never match and is left out), valued with the word table's id (the json's last value). A
                     piece of <= 15 bytes, not OVER / INVALID, that W1 admits, is a vocab string iff its key is
                     there, and then by 12.4 its ids are that one id: one line, no keys line. A full pair of buckets
                     leaves a string out (the greedy path still answers it).
       seen misses   the scratch's piece cache (kernels.md §6-7, epoch-tagged, the region every scratch has): a
                     greedy answer of 1..4 ids is filled under the same key (K5's policy); the next piece with that
                     key reads it. The answer is a function of the folded bytes and the flags 12.4 already checked.
       batch         the model hashes 64 pieces' keys and prefetches their word-table lines before probing them
                     (the misses overlap: memory-level parallelism).
       the tries     W2's longest match walks two double-array tries (da.h, the builder unigram.md §10.3 shares):
                     the vocab strings (start 0) and the prefixed strings without the prefix (start > 0), 8-byte
                     cells (base with a terminal bit, check), padded 256 cells, over the folded bytes: one pass per
                     start finds its longest key (instead of a hash probe per candidate end); a start with no key
                     makes the whole piece [UNK] (W2).
       the scan      an ascii run of word / fold chars is one step (add_raw's per-char work at once while the piece
                     stays in place); the ascii class tables of both scan flag sets are built at load.
     Speed (master fc5808a -> wp-speed, tools/bench spm_stages.c's ship variant A B B A, best of 3, 4 KiB chunks, MB/s
     cold / pass / warm), gb10e X925 cpu 9: bert-uncased en 94 -> 250 / 264 / 286, code 79 -> 230 / 245 / 252, ml
     44 -> 153 / 176 / 183, zh 68 -> 154 / 185 / 187; all-MiniLM-L6-v2 (hf stops at 128 ids) en 614 -> 1617 / 1721 /
     1924, ml 262 -> 887 / 981 / 1100. tr9970x Zen 5 cpu 16: 2.1-4.0x the same way.
