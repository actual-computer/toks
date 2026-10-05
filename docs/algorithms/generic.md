The generic pre-tokenizer (SPEC §1.4)
==========================================

src/core/gen.c runs every byte-level bpe pre-tokenizer chain no compiled template takes, with hf tokenizers
0.23.2's semantics (oniguruma for the regexes). The templates (kernels.md §3, docs/templates/) stay the fast paths:
config.c tries them first and compiles a chain for this engine only when none matches. toks_info reports such a
tokenizer without TOKS_PATH_SCAN.

  §1  scope: the census rows                    §5  the steps: hf's split behaviors
  §2  the program                               §6  integration (driver, scratch, NFC)
  §3  the construct set and the named misses    §7  bounds and cost
  §4  quantifiers                               §8  proof: the differential against hf


1. scope: the census rows
--------------------------

docs/coverage.md §3's patterns no template compiles, all over byte-level bpe:

  P16 MiniCPM5           Split \p{N}{1,3} > Split (cl100k with \p{N}+) > ByteLevel
  P19 DeepSeek-Coder-V2  Split [\r\n] > Split \s?[letters]+ > Split \s?[punct]+ > Split \s+$ > Split [CJK]+ >
                         Digits(individual) > ByteLevel
  P20 Laguna             Split (?:\r?\n)+(?!\r?\n) MergedWithNext > Split (qwen 2) > ByteLevel
  P21 bloom              Split  ?[^(\s|[.,!?…。，、।۔،])]+ > ByteLevel
  P22 LLaDA / Ling       Split with possessive ?+ ++ > ByteLevel
  P23 Spark-X2.5         Split \p{N}{1,3} > Split [CJK]+ > Split (dsv3-like) > Digits > ByteLevel
  P25 zeta               Split (cl100k-like, \p{N}{1}) > ByteLevel
  P26 falcon             Punctuation(Contiguous) > ByteLevel(use_regex) > Digits(contiguous) > Split [0-9]{3}
  P28 A.X-K2             Split [CJK]+ > Split (qwen 2) > ByteLevel
  P29 deepseek-coder     Split [\r\n] > Split \s?\p{L}+ > Split \s?\p{P}+ > Split [CJK]+ > Digits > ByteLevel
  P30 tiny-Cohere2       Split \d{1,3}(?=(?:\d{3})*\b) > Split (o200k) > ByteLevel

P16, P20, P21, P22 and P25 compile onto the cl100k template since (kernels.md §3: P16 with A8 / DIGITS_1_3, P20
with A9, P21 with A10 on bloom's tables, P22 and P25 as qwen 2 spellings), so they leave this engine; the engine still
runs them when a file spells them otherwise.

The steps: Split (a Regex or a String pattern; every behavior; invert), Digits (individual or contiguous),
Punctuation (any behavior), ByteLevel (add_prefix_space false; use_regex: gpt-2's regex as an Isolated split, and
from there on the later steps see the byte-level alphabet's chars, one byte each: falcon's Digits cuts "ü" = C3 BC
because BC is "¼"). A chain needs a ByteLevel somewhere (byte-level bpe). Whitespace, WhitespaceSplit, Metaspace,
Bert, UnicodeScripts, CharDelimiterSplit, FixedLength stay refused by name (no byte-level census chain has them).


2. the program
---------------

One blob (indices only, so load.c copies it): up to 16 steps, then the code of every regex step, the classes and
their sorted code-point range pairs. An instruction is G_CHAR (one atom in a class), G_SPLIT x y (x first), G_JMP,
G_ASSERT (^ $ \A \z \Z \b \B), G_LOOK (a lookahead: its body follows inline and ends in G_MATCH; the main program
jumps over it) or G_MATCH; targets are relative, so the compiler inserts in front of a sub-expression's code
(alternation, quantifiers) without relocating anything inside it. A class is a 128-bit ascii bitmap (computed after
negation) plus, for the rest, a mask of src/gen/ucd_flags.h properties (onig's own data, probed through hf) and the
ranges, then the negation.

Atoms are kernels.md §2's: a well-formed utf-8 sequence is its code point, every byte of an ill-formed one an atom
of its own that is \p{P} (SPEC §3.3's class P for pattern pre-tokenizers) and equals no literal or range. In the
byte domain (after a ByteLevel) an atom is one byte whose code point is its alphabet char.


3. the construct set and the named misses
------------------------------------------

Supported, with oniguruma's semantics (ONIG_SYNTAX_ONIGURUMA, ONIG_OPTION_NONE, utf-8: what hf's SysRegex::new
builds), each probed through hf:

  - literals (utf-8), escapes \t \n \r \f \v \a \e and any escaped non-alphanumeric char
  - classes [...]: ranges, negation, escapes, nested classes (a union: bloom's [^(\s|[.,!?…])] excludes neither '['
    nor ']'), \s \d \w \p{L Lu Ll Lt Lm Lo M N P S Nd} and their negations \S \D \W \P{..} outside a class
  - . (every char but LF), alternation, (?:..), (..) (a capture splits like (?:..)), (?i:..) on ascii letters
    (their other case; s also folds U+017F, k U+212A: the only non-ascii folds onto ascii letters, probed)
  - quantifiers ? * + {n} {n,} {n,m} {,m}, lazy ?? *? +? {..}?, possessive ?+ *+ ++ on one class (§4)
  - (?=..) and (?!..) (not nested), ^ $ (line anchors: $ also matches before every LF), \A \z \Z, \b \B (onig's
    \w: L M Nd Pc ..., src/gen/ucd_flags.h TOKS_UCD_WORD)

Anything else is a named miss at load ("pre_tokenizer Split regex: <what> (generic engine)"): lookbehind, atomic
groups, backreferences, named groups, inline flags other than (?i:..), (?i) on non-ascii letters or on a class
escape, class intersection, POSIX brackets, \x \u \h escapes, a quantifier on a quantifier or on an assertion, a
pattern that can match the empty string (hf's empty matches: no census pattern has one), more than 512
instructions in one step. The sets that are not oniguruma's come from hf's own code, probed too: Digits' rust
char::is_numeric (TOKS_UCD_RNUM: \p{N} plus 13 unicode-17 code points) and Punctuation's is_punc (TOKS_UCD_PUNC:
ascii punctuation plus an old category table, not \p{P}).


4. quantifiers
---------------

x? x* x+ are one G_SPLIT (+ a G_JMP for x*) around x's code; {n,m} copies x's code n times, then m - n optional
copies whose skip jumps past all of them (skipping one skips the rest, the order of nested optionals); {n,} is n
copies and x*. Lazy swaps the split's targets. A possessive quantifier on one class C is rewritten with a one-class
negative lookahead: C?+ = (?:C|(?!C)), C*+ = C*(?!C), C++ = C+(?!C) -- exactly onig's (C never gives back a char it
can take); on anything longer it is a miss.


5. the steps: hf's split behaviors
-----------------------------------

hf splits every piece of the previous step on its own (a piece is the regex's whole input: its lookaheads and $ end
at the piece's end), so the engine recurses: piece(i, [s, e)) runs step i over [s, e) and hands each piece it makes
to step i + 1; past the last step a piece is final. A step's elements are its matches and the gaps between them
(hf's find_matches; for Digits / Punctuation each char of the set is a match of its own), each flagged a match
(xor invert), then the behavior, streamed with one held piece:

  Isolated    every element is a piece             Removed     the non-matches are
  MergedWithPrevious  a match joins the piece before it unless the element before it was a match
  MergedWithNext      a match joins the element after it unless that is a match too (hf folds from the end)
  Contiguous  neighbours with the same flag join (only matches can be neighbours)

Empty pieces vanish (hf drops them). Final pieces leave in rounds of TOKS_CHUNK_PIECES consecutive pieces
(api.c toks_round: their ends, or K5 / the dropped-byte path); a removed span between two pieces closes a round.
toks_pieces reports a piece as hf's offsets do: ByteLevel aligns each alphabet char with the whole char it came from,
so a cut inside a char ends at that char's end: falcon's Digits after ByteLevel cuts "½" (C2 BD: the alphabet chars
"Â" and "½", a number) after C2, and hf reports the two pieces with ends 2, 2. The ids use the byte cut.

The regex: a Pike VM (Thompson's simulation with priority-ordered threads, RE2's / the rust regex crate's PikeVM): a
thread per program position, kept in priority order, a new start thread at each position (lowest priority) until a
match is found; a G_MATCH cuts every thread after it; the search ends when no thread is left. That is exactly the
leftmost-first match onig's backtracking finds (the threads are its alternatives in its order; a thread is dropped
only when a higher-priority one holds the same position, which would reach every match it could). Assertions are
checked at the position; a lookahead runs its body as an anchored search at the position (any thread reaching its
G_MATCH), with its own lists and marks.


6. integration (driver, scratch, NFC)
--------------------------------------

config.c reads the chain into toks_gen_spec steps (it stays the only json reader) and gen.c compiles them into the
parse arena; load.c copies the blob (ctx->gen). api.c run_text hands a text unit to toks_gen_run, which calls
toks_round per round. The VM's memory (two mark arrays, the thread lists, two closure stacks: 48 bytes per
instruction of the longest step) is the scratch's extra region (ctx->scr_extra, between K5's work region and the
bounce; K5's work_bytes stops before it, so a round's K5 call never touches the lists of the step that called it).
Under NFC the driver normalizes the whole gap first (run_gap's phase-1 path): its restart points are the templates'
rules, not this engine's. toks_split_points certifies no cuts for a generic tokenizer (tmpl NONE).


7. bounds and cost
-------------------

A search costs O(m) per position it reads (m = the step's instructions; marks dedupe a thread per instruction), and
a match's search reads at most to where its higher-priority threads die: for the census patterns that is the
match's own run (a whitespace or digit run), so a chain is O(m n) per step. A lookahead adds an anchored search at
each position it is reached: O(L m) with L its body's reach -- one char for (?!\S), two for (?!\r?\n), and a digit
run for P30's (?=(?:\d{3})*\b), which makes that one step O(m r²) over a run of r digits (the compiler's bound for
an unbounded lookahead body; reported, not hidden). Stack: no recursion beyond the chain's 16 steps.

Measured (a quick look with tools/bench/e2e.c, not a §12 cell: gb10d X925 cpu 16, 4 KiB chunks, pass state, best
of 3, ids sha-checked per chunk stream): the generic engine runs 11-20 MB/s on en / code and 14-25 MB/s on cjk (bloom's
one-step chain 45-55 MB/s, falcon 12-14, tiny-Cohere2 11-14), against 312 / 488 / 160 MB/s for llama3 on its compiled
template and 115-180 for digits-gpt2 on the K3 c twin: a chain on this engine is 10-30x slower than a template. A
Pike VM pays a thread closure per position and step. The census's heavy chains went to the templates as data
(P16, P20, P21, P22, P25: §1); what is left, by (b) downloads, and why each stays here:
  P19 DeepSeek-Coder-V2 (0.22%: Coder-V2-Lite, coder-7b-v1.5, DeepSeek-V2-Lite, natural-sql) and P29 deepseek-coder
      (0.02%): no template's parameter set. Six cascaded Isolated splits, each over the previous one's pieces:
      [\r\n], \s?[cased letters]+, \s?[!-/:-~ and fullwidth / quote / CJK punctuation]+ (ascii letters included,
      so it re-cuts the letter pieces: " héllo" -> " h", "é", "llo"), \s+$, [U+0800..U+9FA5, U+AC00..U+D7FF]+,
      Digits. A template for the family would be a class-run cascade: one class bit per step class and a step list
      as data (P19, P21 and P29 share the shape: X?[C]+, [C]+, [C], \s+$).
  P23 Spark-X2.5 (0.10%): dsv3's first two splits, then a regex that is not dsv3's third (no [\r\n]* tail on the
      punctuation run, [\r\n] alone instead of \s*[\r\n]+), then Digits: a dsv3 variant needs new rules, not data.
  P26 falcon (0.03%): ByteLevel's own regex in the middle of the chain (the later steps see the byte alphabet).
  P28 A.X-K2 (0.02%): a CJK split before qwen 2 (dsv3's regions on the cl100k template: a candidate for a cut
      parameter like A8 / A9).
  P30 tiny-Cohere2 (0.02%): a lookahead digit-grouping split before o200k.
Then, for whatever stays: a first-char prefilter per step (a class scan where matches are rare), then a lazy DFA
(RE2's) for the cl100k-like steps.


8. proof: the differential against hf
--------------------------------------

tests/generic/diff.py: every pinned file of §1 (tools/corpora/fetch_tokenizers.py GENERIC; P23 Spark-X2.5's chain,
whose own file is refused for its vocab, through --chain-from on DeepSeek-V2-Lite's file), random texts over an
alphabet of every pattern's edge characters (each contraction and fold, letters in and out of P19's class, every
\p{N} kind and the 13 rust-only numerics, CJK in and out of the literal ranges, kana, hangul, bloom's set, every
whitespace kind and newline run), compared by piece ends (hf's offsets as bytes) and by ids; plus the light hf
check (tests/parity) per pinned file. make test carries three breadth fixtures (tests/data/breadth gen_falcon,
gen_behaviors, gen_groups: every behavior, invert, a String pattern, the lookaheads, folds, possessives, digit
groups, nested classes, a cut inside a char) against hf; each of these mutants fails them: MergedWithNext dropped,
MergedWithPrevious after a match, $ as end of string only, \b inverted, x? lazy, Contiguous as Isolated, cuts
inside a char reported as such.
