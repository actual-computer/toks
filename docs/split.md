# Exact cuts: toks_split_points' rules for every algorithm

SPEC §5 is the contract: at a certified cut c of x, encoding x[0, c) and x[c, len) (the second part with
TOKS_CONTINUATION, neither post-processed) and concatenating the ids gives x's ids without post-processing (§5.2).
src/core/split.c implements the rules below; toks_split_points (docs/usage.md, "Cutting text") and toks_par build on
the same predicates (split.h `toks_cut_ok`).

## 1. What a rule must show

A cut is certified by a predicate over a window of atoms around c (§5.4), never by encoding. For a pipeline
"added tokens -> normalizer -> pre-tokenizer -> model", c is exact when:

1. **added tokens**: no occurrence of any added token of either phase overlaps the atoms the rule reads
   (`tokens_clear`, every mode: a special dropped under NONSPECIAL still moves hf's cursor). Strip / single_word
   tokens stay sound (the rules never cut inside a whitespace run a strip could consume; split.c's §split.c.1 note).
   Tokens matched on normalized text (phase 1) are only checked on raw text, so a family whose normalizer can change
   text near c gets no cut when the mode recognizes them (NFC, wordpiece, unigram).
2. **normalizer**: N(x) = N(x[0, c)) N(x[c, len)) and N leaves the atoms the rule reads in place (NFC: every atom of
   the window is an NFC boundary atom; wordpiece / unigram: ascii neighbours, below).
3. **pre-tokenizer, both sides**: the pieces of x have a boundary at c; the pieces of x[0, c) alone (its end is the
   end of input: lookaheads see nothing) equal x's pieces before c; the pieces of x[c, len) alone equal x's after c.
   Every template classifies forward from a piece start (no lookbehind), so the third follows from the first; the
   rules below argue the first two.
4. **model**: pieces are encoded independently (bpe, unigram, wordpiece), so equal pieces give equal ids.
5. **document state**: start-of-input behaviour (a Metaspace prefix "first", Strip(left), an inserted '^' prefix)
   happens only in the first part; a later part runs with TOKS_CONTINUATION, which turns it off. A family where the
   continuation part would still differ gets no cut.

`toks_cuts_of` refuses (no cut, inputs encode serially, exactly) tokenizers with truncation or padding (whole-document
steps), a tiktoken wrapper's 400,000-code-point / 25,000-whitespace cuts (kimi: they count from the call's start),
the generic engine, the cl100k template's cut-first parameters (TOKS_TP_DIGIT_CUT / NL_CUT / GB_SP: no rules
derived yet), o200k with TOKS_TP_HAN (kimi again), sentencepiece-style bpe under a Prepend normalizer, and the
unigram chains of §7 that keep a segment in one piece.

A predicate reads at most `win` = (longest recognized added token) + 16 bytes on either side of c, and reads the end
of input only when its window reaches it: a cut with c + win <= len stays certified for every extension of x. That
is what makes the cut usable for appends (docs/usage.md, "Cutting text").

Atoms, classes and the templates' piece rules: docs/kernels.md §2-3 (cl100k), docs/templates/o200k.md,
docs/templates/dsv3.md. Below, a = the atom ending at c, b = the atom starting at c, z = the atom before a; WS =
whitespace but CR / LF, NL = CR or LF, "case" = o200k's UP or LO flag.

## 2. cl100k template (gpt-2, llama 3, qwen 2 / 3 / 3.5, glm, deepseek's distills, ...)

kernels.md's R1-R5, unchanged since they were written (docs/notes/c-core.md §split.c.1 has the table).

## 3. o200k template (gpt-oss, llama 4, nemotron, minimax, mistral-nemo)

- **O1** a has base L, b is no case atom and not U+0027. a's piece is a letter piece (no other rule takes an L
  atom); the pieces of a maximal case run tile it and the last one ends where the run ends (+ the contraction
  suffix K, which needs b = '). b ends the run (not UP / LO), so a boundary is at c. Alone, x[0, c)'s scans read
  the atom at c only to test "LO?" / "' ?": absent there, not LO / not ' in x: the same decisions.
- **O2** a has base N, and b is not N (or the nemo variant's \p{N}: every digit is its own piece). Digit groups
  tile an N run from its start (no other rule takes N); the last group ends at the run's end. {1,3} reads at most
  the next atom, which is not N in either case.
- **O3** a has base P with no case flag (a mark is excluded: inside a P run it is P, before letters it starts a
  letter piece), b has base N or WS. a is not a letter prefix (b is no case atom), so it ends O3's P run; the tail
  [\r\n/]* stops at b (N / WS are not CR, LF, '/').
- **O4** a is NL, b is neither WS, NL nor the tail's '/' (CR under kimi's NO_SLASH). The WS / NL run ending at a is
  taken through its last NL, by \s*[\r\n]+ or by an O3 tail, both ending at c; alone, the same.
- **O5** a is WS, z is not WS / NL (a one-atom run), b is neither WS, NL nor a case atom (WS is a letter prefix),
  nor P when a is U+0020 (O3's optional space). Then a is a piece by \s+ in x (one atom before \S) and by
  \s+(?!\S) alone: the same piece.

Evidence (tests/c/test_split.c, every cut of every string): o200kstyle / nemostyle = llama3style.json with the
pattern rewritten (o200k.md §1's strings), o200k+nfc (minimax's normalizer), over 27 class representatives
(upper-only, lower-only, Lm, Lo, Lt, a mark, U+017F, digits, ' ! / < and invalid bytes, WS, NL); real files o200k,
nemotron3-4b, llama4, minimaxm2.

## 4. deepseek v3 (three Splits: \p{N}{1,3}, the CJK runs, then a cl100k-like pattern on each piece)

Each Split runs on every piece the previous one left, and its lookaheads see that piece's end as the end of input,
so a cut must be a boundary of all three steps. X atoms (none of L M N P S \s: format, private use, controls) are
gaps no alternative matches: no cut beside one.

- **D1** a is L (letters and marks), b is not L. The letter run ends at c: [\p{L}\p{M}]+ (or the ascii
  "[punct][A-Za-z]+" alternative) ends there; c is no digit or CJK match interior.
- **D2** a is N, b is not N: the first Split's groups tile a digit run from its start.
- **D3** a is P (\p{P} or \p{S}), b is N or WS: the P run ends, the tail [\r\n]* stops at b; with b = N the first
  Split cuts there anyway.
- **D4** a is NL, b is neither WS nor NL: \s*[\r\n]+ or a P piece's tail ends at the last NL.
- **D5** a is WS, z is neither WS, NL nor X, b is N or P (not L: WS prefixes letters; not P after U+0020).

Evidence: tests/data/compile/dsv3style.json over 21 representatives (ascii and other letters, a mark, Han,
katakana, digits, ascii and other P / S, WS, NL, U+200B, a C0 control, invalid bytes); real dsv3, dsv4.

## 5. sentencepiece-style bpe (gemma, mistral, llama 2 without its Prepend)

`toks_spm_cut`: a Metaspace piece start, or a certified word cut (spm_bpe.md §5.5: both chars one-char vocab
strings, no reachable string spans them, one string per id), never under a Prepend normalizer, and never where the
continuation part would get a prefix the whole lacks. It is exactly K7's word-boundary test
(`toks_spm_cut_between`, the same function the scan calls), so the split points are the scan's proven word
boundaries. Evidence: the spm fixtures + gemma4, gemma4-base, mistral-v0.3.

## 6. wordpiece (BertNormalizer + BertPreTokenizer: every pinned file)

c with an ascii whitespace byte before it (class SPLIT after the normalizer's ascii table: space, tab, CR, LF) and
an ascii word, folded or punctuation byte at it. BertPreTokenizer removes whitespace and isolates punctuation, so no
word crosses c; the normalizer is per char (clean_text, chinese padding, lowercase, strip_accents: an ascii starter
at c stops canonical reordering), so both sides normalize to their part of N(x); max_input_chars_per_word counts per
word. Truncation / padding (most sentence-transformers exports) and recognized phase-1 tokens give no cut.
Evidence: wp-bert-base-uncased, wp-labse (wp-minilm-l6 truncates: 0 cuts, checked).

## 7. unigram (the census chains with a cutting pre-tokenizer)

c at a U+0020 whose neighbours are both "simple" ascii chars (unigram.h `simple[]`: their own grapheme, mapped to
themselves by the charsmap, only extending the open piece), on a chain with WhitespaceSplit or Metaspace(split),
no inserted '(?<!\n)^' prefix, no Replace(' ' -> '▁'), a charsmap that maps U+0020 to itself and no Replace(String)
holding a space:
- the charsmap rewrites whole graphemes; c is a grapheme boundary (ascii | space | ascii), so N(x) splits at c;
  ' {2,}' collapse cannot reach a lone space; NF* / StripAccents / Lowercase keep ascii boundaries;
- Strip(left) applies to the first part only (toks's continuation turns it off), Strip(right) never sees the
  space (the left part ends with a simple char);
- WhitespaceSplit removes the space; Metaspace(split) turns it into the '▁' that starts the next piece, and needs
  no prepend there (the right part starts with it).
Refused: ruri (Metaspace without split: one piece per segment), llm-jp (no pre-tokenizer), recognized phase-1
tokens. Evidence: uni_bgem3, uni_me5small, uni_flant5 (Metaspace after the collapse), uni_t5base (WhitespaceSplit +
Metaspace); uni_ruri3 0 cuts, checked.

## 8. Evidence runs

`test_split [L [random [seed]]]`: make test runs L = 3 (fixtures; real files L - 1) with 60 random strings; a
host run passes L = 4 and more random strings per seed. tests/fuzz/fuzz_par.c checks every cut on random texts over
the fuzz pin set (gpt2 llama3 qwen38 glm53 o200k dsv3 gemma4 mistral-v0.3 wp-bert-uncased uni_t5base uni_bgem3
minimaxm2 nemotron3-4b dg-smollm2 pythia tinyllama). Receipts per PR.
