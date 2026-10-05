sentencepiece-style bpe: the semantic contract (algorithm 2)
=============================================================

SPEC §1.2's second algorithm: hf tokenizers' BPE model run WITHOUT a byte-level alphabet, as gemma 1-4,
llama 2, mistral v0.1-v0.3 and their kin ship it in tokenizer.json. hf tokenizers 0.23.2 (88a4498) is the
definition; this file writes it out in docs/kernels.md's style. tests/model/spm_model.py is its executable
reading and tests/model/run_spm_fuzz.py diffs that reading against hf (§10 has the counts). When this file, the
model and hf disagree, hf is right and the other two are fixed.

Gemma 4 is one of the critical targets: §1.1 is its whole contract in one place, §5.5 the certified cuts
that make its one-word-per-document pieces cheap, §6 how toks runs the text model without writing text.

  §1 the family: gemma 4 first, then every pinned file and variant
  §2 segments: added tokens around the text model
  §3 normalizers: Prepend, Replace
  §4 pre-tokenizers: Metaspace, Split
  §5 the model: symbols, byte fallback, unk, merges, certified cuts
  §6 tables, not text: the text model folded into the char table (toks' design, proven equal to §3-§5)
  §7 post-processing
  §8 decode
  §9 notes that bite
  §10 evidence
  §11 what toks accepts, and the c side

Notation: "▁" is U+2581 (the metaspace char of every pinned file); a string is a sequence of unicode scalars
(chars); "the model vocab" is tokenizer.json's model.vocab (string -> id), never the added tokens.


1. the family
--------------

1.1 gemma 4 (critical target)

  file    google/gemma-4-26B-A4B-it @ 4d7ae4984b7d, tokenizer.json, 32,169,626 bytes, sha256
          cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f (pin "gemma4"). Every instruct
          size ships this exact file: E2B-it @ 3e22461f65e8, E4B-it @ ee0ef6023621, 12B-it @ 707f0a3b8a3c,
          31B-it @ 842da3794eaa (and unsloth/gemma-4-E2B-it @ d36bdd3855c8). Every base size (E2B
          @ d29ff6b45f08, E4B @ 411aa17b749a, 12B @ 023679ed352d, 26B-A4B @ 24548b62aa02, 31B @ 5bbc2fb1c1b2)
          ships one other file, sha256 12bac982b793c44b... (pin "gemma4-base", 444 bytes longer), identical
          except the post-processor: [<bos> $A], <bos> = 2. All public (not gated); read 2026-10-03.
  model   BPE, vocab 262,144 (ids 0..262143, dense, one string each), 514,906 merges (more merges than
          merged ids: an id can be reachable by several splits), unk "<unk>" = 3, fuse_unk true,
          byte_fallback true with all 256 <0xHH>, ignore_merges false, dropout null.
  text    normalizer Replace(" " -> "▁") and nothing else (no Prepend: no prefix is ever added);
          pre-tokenizer Split(String " ", MergedWithPrevious, invert false), which never cuts (below).
  decode  Sequence[Replace("▁" -> " "), ByteFallback, Fuse] (no Strip: a leading space survives).
  post    TemplateProcessing [$A]: adds nothing (base files: [<bos> $A]).
  added   24, all special, all normalized=false (<pad> <eos> <bos> <unk> <mask> <|tool> <tool|> ...): phase 0
          only, ids = their vocab ids.

  one bpe word per segment. gemma 4's model sees each phase-0 gap g, normalized, as ONE piece:
    (1) no added token has normalized=true, so phase 1 (§2.2) matches nothing: the one split of g is N(g);
    (2) N = Replace(" " -> "▁"): the leftmost non-overlapping occurrences of a one-char literal are all of
        its occurrences, "▁" holds no U+0020 and no step follows, so N(g) holds no U+0020; N(g) is
        non-empty when g is (one char written per char replaced);
    (3) the Split's matches in N(g) are the occurrences of " ": there are none. The cover of N(g) is one
        non-match part, which every behaviour keeps whole when invert is false (MergedWithPrevious only
        moves matches): one piece, N(g).
  So the piece is the whole text between two added tokens; in mode NONE, or in a text without added tokens,
  the whole document. (The same holds for gemma 1 with no pre-tokenizer at all, and for gemma 2 / 3 / 3n.)

  speed. With a whole document as one bpe word, the merge loop would be the whole cost: one loop over every
  char of the document (every byte under byte fallback), O(m log m) with a heap, and no cache can help
  (documents do not repeat). §5.5 certifies cuts that no merge can ever cross; in gemma 4 they fall before
  every "▁" (except after "▁" or ">") and between any two adjacent chars no vocab string joins. On 374,324
  chars of real text (this repo's docs and code plus 25 sentences in 20 scripts) that gives 115,107 words:
  mean 3.25 chars, median 2, longest 105. toks runs the model per word, each word answered by the scratch
  cache keyed by its RAW bytes (§6.6: a space is one byte there, not the three of "▁"), with the same ids
  (proven in §5.5 and §6, checked in §10).

1.2 every pinned file

Pinned files (tests/spm/pins.json: repo, revision, sha256; tests/spm/fetch.py downloads and verifies them;
anonymous hub only, gated originals through public mirrors):

  name          repo @ revision (12)                          sha256 (16)       vocab   merges  added
  gemma4        google/gemma-4-26B-A4B-it @ 4d7ae4984b7d       cc8d3a0ce36466cc  262144  514906    24
  gemma4-base   google/gemma-4-E2B @ d29ff6b45f08              12bac982b793c44b  262144  514906    24
  llama2        NousResearch/Llama-2-7b-hf @ 8efe6c9b9365      f7b50bcf6d6672ea  32000   61249     3
  tinyllama     TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1ffed  bcd04f0eadf90287  32000   61249     3
  codellama     codellama/CodeLlama-7b-hf @ 6c284d1468fe       dcb4337ba92f948b  32016   61260     3
  mistral-v0.1  mistralai/Mistral-7B-v0.1 @ 27d67f1b5f57       11c08db21487c885  32000   58980     3
  mistral-v0.3  mistralai/Mistral-7B-v0.3 @ caa1feb0e54d       e553af6fff7d7ad7  32768   58980   771
  gemma1        unsloth/gemma-2b @ 7ac9d201a57c                7da53ca29fb16f6b  256000  580604  217
  gemma2        unsloth/gemma-2-9b @ 9145841c83ad              5f7eee611703c5ce  256000  580604  249
  gemma3        unsloth/gemma-3-4b-it @ bf46152c47f5           4667f2089529e8e7  262144  514906  6415
  gemma3n       unsloth/gemma-3n-E2B-it @ c4291af6285b         b6c35ee648c07754  262400  514906  6670
  phi3          microsoft/Phi-3-mini-tr9970x-instruct @ f39ac1d28e92  072ab882d6c7192a  32000   61249    14
  yi            01-ai/Yi-6B @ 80080be87ec5                     a13ccc285aea27f5  63992   110029  232

  meta-llama/Llama-2-7b-hf and google/gemma-{7b,2-9b,3-4b-it} are gated (401 anonymously); mistralai and
  google/gemma-4 are not. Byte-identical copies and semantic twins are listed in pins.json ("same_as").
  hf-internal-testing/llama-tokenizer @ d02ad6cb9dd2 (sha256 8eea70c4866c4f13) is NOT llama 2's file: the
  last 119 merges (the ▁-run merges) are in another order, so runs of spaces encode differently.

Every pinned model: type BPE, dropout null, unk_token "<unk>", fuse_unk true, byte_fallback true,
ignore_merges null or false, continuing_subword_prefix / end_of_word_suffix null; truncation / padding null.

  text model (normalizer + pre-tokenizer)                                              files
  T1  Sequence[Prepend "▁", Replace " " -> "▁"]; no pre-tokenizer                   llama2 tinyllama codellama phi3
  T2  Replace " " -> "▁" (yi: inside a Sequence); no pre-tokenizer                   gemma1 yi
  T3  Replace " " -> "▁"; Split(String " ", MergedWithPrevious, invert false)       gemma2 gemma3 gemma3n gemma4
      (never cuts, §1.1: no U+0020 survives the normalizer, and the pre-tokenizer only sees normalized text)
  T4  no normalizer; Metaspace(replacement "▁", prepend_scheme first, split false)   mistral-v0.1 mistral-v0.3

  decoder
  D1  Sequence[Replace "▁" -> " ", ByteFallback, Fuse, Strip(" ", start 1, stop 0)] llama2 tinyllama codellama
                                                                                     mistral-* phi3 yi
  D2  Sequence[Replace "▁" -> " ", ByteFallback, Fuse]                               gemma1 2 3 3n 4

  post-processor: TemplateProcessing [<s> $A] (id 1: llama2, tinyllama, codellama, mistral-*), [<bos> $A] (id 2:
  gemma1-3n, gemma4-base), [$A] (gemma4, phi3); none (yi).

  added tokens (all have normalized, special, lstrip, rstrip, single_word written out):
    llama2        <unk> <s> </s>: special, normalized=TRUE (the original file; matched as "▁<s>", §2.2)
    tinyllama, codellama, mistral-v0.1: the same three, normalized=false
    mistral-v0.3  771 special ([INST], [/INST], [TOOL_CALLS], [control_8] ...)
    gemma1/2      217 / 249: 6 special (<pad> <eos> <bos> <unk> <start_of_turn> <end_of_turn>), the rest
                  non-special: <mask>, <unusedN>, html tags, runs of \n, runs of ▁ (matched on the RAW text,
                  so only a literal ▁ in the input reaches them)
    gemma3/3n     6415 / 6670 (9 / 264 special), runs of \t too; gemma3's <image_soft_token> is beyond the vocab
                  (id 262144, hf's next-id rule)
    gemma4(-base) 24 special
    phi3          14: 11 beyond the vocab, rstrip=true on all 11 (<|end|>, <|assistant|>, </s> non-special ...)
    yi            232 special, one content listed twice (<fim_suffix>)
  vocab quirks: gemma1/2 have 255 byte tokens (<0x09> missing; "\t" is a vocab char, so no text needs it);
  yi's ids have holes (63992 strings, max id 63999); every file: one string per id, no merge involving a
  byte token, merged ids NOT rising with rank (so TOKS_TF_IDS_AS_RANK never applies).


2. segments: added tokens around the text model
------------------------------------------------

Input: text (valid utf-8; invalid bytes in §9), the mode (SPEC §3.2), post-processing on or off.

  2.1 phase 0. The added tokens with normalized=false are found in the raw text with hf's per-match policy,
      exactly docs/kernels.md §4 (leftmost-longest, mode, single_word, lstrip, rstrip, resume at the raw
      match end). What lies between the accepted matches are the phase-0 gaps.
  2.2 phase 1. Each phase-0 gap g is normalized BY ITSELF (§3), giving N(g); an empty N(g) yields nothing.
      The added tokens with normalized=true are found in N(g), with the same policy, by their NORMALIZED
      FORM: the content run through the same normalizer chain as a string of its own. With T1 that is
      "▁" + content: llama2's "</s>" is matched as "▁</s>" only, i.e. at the start of a gap or after a
      space ("a </s>" -> ▁a, ▁</s>; "a</s>" -> the chars < / s > as text). With no normalizer the form is
      the content. A form that normalizes to "" is refused (§11). toks matches them on the raw gap
      (docs/notes/c-core.md config.c §9, segment.c §5): a content holding no ' ' or ▁, at the gap start or after
      a raw ' ' / ▁ that the token absorbs; the gap's units after such a token get no prefix, and a token that
      absorbed the gap's first char leaves a lone "▁" before it (" <s>" -> ▁ (0, 1), ▁<s> (0, 4)).
  2.3 the empty-span rule (kernels.md §4 does not say it; hf does it): lstrip moves a match's start left
      over whitespace but never before the previous accepted split's end; when the previous token's rstrip
      run covers this match, the clamp can put start at or past stop:
        start == stop  the token vanishes: no id, no text; the gap before it is still emitted, and the
                       next split starts at stop
        start >  stop  hf panics ("AddedVocabulary bad split"): the encode fails
      (needs a token with lstrip inside the whitespace run a previous rstrip token consumed; no pinned
      file has lstrip.)
  2.4 pieces. Each remaining split s (a slice of N(g) between phase-1 matches) goes through the
      pre-tokenizer (§4) together with its original offset; every resulting piece goes through the model
      (§5); the ids follow in text order, an accepted added token contributes its id.
  2.5 modes. ALL: both phases. NONSPECIAL: matches of special tokens are dropped in both phases (their
      bytes stay text, never rescanned; kernels.md §4 step 1). NONE: no added token is matched; the whole
      text is one phase-0 gap, normalized and pre-tokenized as such.
  2.6 an empty text yields no ids (post-processing still applies).


3. normalizers: applied in order to each phase-0 gap as a whole
-----------------------------------------------------------------

  Prepend(p)        if the string is non-empty: p + string. It does not look at what is already there
                    (a gap starting with "▁" becomes "▁▁...").
  Replace(String pat, content)
                    every leftmost non-overlapping occurrence of the literal pat (scanning left to right)
                    is replaced by content. Regex patterns and an empty pat are refused (§11).
  Sequence[...]     each in order.

  T1 on "a b" -> "▁a▁b"; on " a" -> "▁▁a"; T2 on " a" -> "▁a". Only U+0020 is replaced: tab, U+00A0,
  U+3000, \n stay what they are.

  original offsets (consulted only by Metaspace prepend_scheme=first, §4.1): every char of N(g) carries the
  input offset it aligns to. An unchanged char keeps its own; the chars Prepend adds carry the first char's;
  the chars Replace writes carry the offset of the LAST char of the occurrence they replace.


4. pre-tokenizers: applied to each split of §2.4
-------------------------------------------------

  4.1 Metaspace(r, prepend_scheme, split), on a split s whose first char has original offset o:
        1. every U+0020 of s becomes r.
        2. if s is non-empty and does not start with r (after step 1), r is prepended when
             always   -
             first    o == 0: s begins at the very start of the input (not after an added token, not in a
                      later gap; a text starting with a space already starts with r and gets nothing)
             never    -
        3. split=true: cut before every r except at offset 0 (each r starts a piece: "a▁▁b" -> a, ▁, ▁b;
           "▁▁" -> ▁, ▁); split=false: s is one piece.
      json: replacement is one char; prepend_scheme defaults to always; split defaults to true; the legacy
      add_prefix_space=false is accepted only with prepend_scheme never (hf refuses the file otherwise),
      add_prefix_space=true changes nothing.
  4.2 Split(String pat, behavior, invert) on s: the matches are the leftmost non-overlapping occurrences of
      the literal pat; s is covered by matches and the non-matches between them (invert swaps the two
      kinds); then, as hf's NormalizedString::split:
        Removed             the non-matches
        Isolated            every part
        MergedWithPrevious  a match joins the part before it (a leading match stands alone)
        MergedWithNext      a match joins the part after it (scanning from the end; a trailing match alone)
        Contiguous          consecutive parts of the same kind join
      empty pieces are dropped. Regex patterns and an empty pat are refused (§11).
  4.3 Sequence[...]: each pre-tokenizer in order, on every split the previous one produced; each split keeps
      the original offset of its own first char.
  4.4 no pre-tokenizer: s is one piece. With T1 / T2 / T3 the whole normalized gap is ONE bpe piece: a 1 MiB
      paragraph without added tokens is one piece of 1 MiB (the model's cost term is per gap, not per word).


5. the model: hf BPE on one piece P (a non-empty string)
---------------------------------------------------------

  5.1 ignore_merges. If set and P is a model-vocab string: [its id], nothing else runs.
  5.2 symbols. Walk the chars c of P in order with pending = none:
        a. c, as a one-char string, is in the model vocab: emit pending if any (pending = none), then
           emit id(c).
        b. else, byte_fallback is set and EVERY byte b of c's utf-8 has the model-vocab token "<0xHH>"
           (HH = two UPPER-case hex digits, Rust's {:#04X}): emit those ids, one per byte, in order.
           A pending unk is NOT emitted first: it lands after them (and keeps fusing, step c).
        c. else, unk_token is set: if pending and fuse_unk: nothing (c fuses into the pending unk); else
           emit pending if any, then pending = id(unk_token). An unk_token that is not a model-vocab string
           makes the encode fail (only when this step runs).
        d. else: c contributes nothing (dropped silently).
      At the end, emit pending if any.
  5.3 merges. The merge list in file order ("left right" strings or [left, right] pairs; for strings, those
      starting with "#version" are dropped first, then ranks are the indices); merge (a, b) of rank k joins
      adjacent symbols id(a), id(b) into id(a + b); a pair listed twice keeps its LAST rank; left, right
      and left + right must all be model-vocab strings (else hf refuses the file).
      Repeatedly take the adjacent pair whose merge has the lowest rank, ties to the leftmost, and replace
      it by the merged id; stop when no adjacent pair has a merge. Byte-fallback and unk symbols are ids
      like any other here (they merge if the list joins them; no pinned file does).
      hf runs a min-heap on (rank, position) and skips entries whose pair has changed (it compares merged
      ids, exact because every string has its own id); the merge order is the one stated here.
      toks runs K6's loop on the symbols (kernels.md §5), its probes behind the merge table's pair filter
      (kernels.md §5 "Tables"; gemma 4 cold en / code +4%, ml +8%); on an asm tier (TOKS_TF_ASM_MERGE, set by
      load.c) a word of < 128 bytes runs the asm merge loop (toks_k6_merge_neon / _avx2: K6's short path entered
      at its records with every symbol a token, each round-one pair probed): the same loop, so the same ids.
      Merges were ~41% of gemma 4's cold en time after the static word table (6.6; the model's answers replayed
      free: 116 -> 197 MB/s, symbols() included 210). Measured, 4 KiB cold / pass / warm vs the c loop: gb10d X925
      gemma 4 en 1.04 / 1.05 / 1.03x, code 1.05 / 1.07 / 1.07x, ml 1.07x; mistral-v0.3 / phi3 en 1.09x, code 1.17-
      1.20x; tr9970x Zen 5 gemma 4 en 1.17 / 1.15 / 1.07x, code 1.27 / 1.24 / 1.13x, ml 1.26x, mistral-v0.3 en 1.25x,
      code 1.47x; zh 1.01-1.05x everywhere (most zh words are one-char words, no model call). Rejected (measured): a hot table of the 8,192
      lowest-rank merges (128 KiB) asked first, a pair it lacks being "unknown, rank >= 8,192" and decided by
      the 8 MiB merge table only when it is the least; exact (every pair of rank < 8,192 is in it), and the
      merge-table probes of a gemma 4 en word fall from 15.1 to 3.3, yet cold ran at 0.64-0.82x on gb10e
      (0.70x on M2): the merge table sits in L2 / L3 on these cores, so the second lookup only adds work.
  5.4 output: the symbols' ids in order.

  Examples (llama2): "▁hello" -> [22172]; "😀" -> <0xF0> <0x9F> <0x98> <0x80> = [243, 162, 155, 131];
  mistral has "😀" in its vocab: [30575].

  5.5 certified cuts: where bpe factorizes (a property of the vocab, proven here; hf stays the definition)

  preconditions (the loader refuses a file without them, by name, so every implementation that copies the rule must
  copy the refusals too):
    (i)   one string per id: no two model-vocab strings share an id [model.vocab: two strings share an id].
          Without it (b) below fails: hf loads vocab {"a": 0, "q": 0, "b": 1, "qb": 2} with the merge "q b"
          and encodes "ab" as [2] (the symbol of "a" has id 0, the merge is keyed by ids), while no vocab
          string spans "ab", so the rule would cut it to [0, 1].
    (ii)  dropout null [model dropout]: dropout skips merges at random; there is no fixed answer to factor.
    (iii) ignore_merges off (with it toks makes no cut at all: (e) below).
    (iv)  no continuing_subword_prefix / end_of_word_suffix [model continuing_subword_prefix]: merged strings
          would not be plain concatenations.

  theorem. Let P be a piece, (i)-(iv) hold, and p (0 < p < |P|) a char boundary whose chars x = P[p-1] and
  y = P[p] are both one-char model-vocab strings, and no model-vocab string contains x immediately followed
  by y. Then model(P) = model(P[:p]) ++ model(P[p:]).

  proof.
    (a) the symbols factorize. x is a vocab char, so §5.2 step a leaves nothing pending after it: the walk
        over P[:p] emits exactly what the walk over P emits up to p, ending with id(x); y is a vocab char,
        so the walk over P[p:] starts with id(y) from the same empty state the walk over P is in at p. The
        initial sequence is L ++ R with L ending in id(x) and R starting with id(y).
    (b) strings concatenate. hf's merged id is the vocab id of left string ++ right string (iv), and an id
        names one string (i): by induction every symbol's string is the concatenation of the strings of
        the initial symbols it covers.
    (c) nothing crosses. A merge joining a symbol that ends at id(x) with one that starts at id(y) would, by
        (b), name a vocab string containing x y: there is none. At every moment the pair at the junction is
        such a pair, so it never has a merge, and no symbol ever covers both sides.
    (d) the loop interleaves. Each step takes the lowest-rank adjacent pair, ties to the leftmost. No pair
        straddles the junction (c), and a merge on one side changes no pair of the other side. When the
        step's pair is on the left it is the minimum of the left's pairs (the minimum of a union, found in
        one part, is that part's minimum; positions keep their order), i.e. exactly the step model(P[:p])
        takes next; likewise on the right. The loop ends when neither side has a merge, i.e. when both
        separate loops have ended: the result is model(P[:p]) ++ model(P[p:]).
    (e) ignore_merges must be off: hf first looks the WHOLE piece up (§5.1), which words cannot reproduce.
  Applying the theorem at every such boundary encodes a piece word by word with identical ids. The test is
  local (two chars) and conservative: (c) needs only the strings a merge can produce, so any superset of
  them will do; toks uses the reachable strings of §6.5 (all of them but those holding a char the text
  model maps away). It needs nothing about the merge list, so it holds for every hf BPE under (i)-(iv),
  byte-level included (there the chars are the alphabet's).

  the compiled rule (spm_build.c; toks_spm_cut_between in spm.h), on the IMAGES of the input chars (§6):
    - the pair set: every adjacent pair x y inside a reachable vocab string with x and y both one-char
      vocab strings, keyed id(x) << 21 | id(y), open addressing at load <= 1/4 (K7 reads two slots at once and
      they mostly decide). gemma 4: 33,945 pairs in 262,144 slots = 2 MiB.
    - the cut bitmap: the answer for every pair of images in ASCII + "▁" (129 x 129 bits = 2,081 bytes),
      read with one bit test; the reviewer measured that it answers 98-100% of the checks on English and
      code. Other pairs: a char whose image begins no pair (PAIRED clear in its entry) cuts without a probe,
      else one probe of the pair set.
    - the byte-pair table (cut_ab, 4 KiB): for two ASCII INPUT bytes x y, bit x | y << 8 is the scan's whole
      decision (a Metaspace split before y, or the rule above on their entries), computed at load from
      those same functions. The scan (toks_spm_encode) reads 8 ASCII bytes after an ASCII byte in one step,
      8 bit tests, and every other char by the per-char rule; it collects a 64-byte window's word starts as
      a bit mask before it encodes any of its words, so no branch waits on where a word ends.
  The one gemma 4 vocab string with a char other than "▁" right before "▁" is ">▁</" (id 107068, also in
  gemma 3 / 3n), so ">" is the only char after which "▁" certifies nothing; llama 2, mistral, yi, phi-3,
  gemma 1 / 2 have no such string at all. Checked in §10: the python model word by word, the c side word by
  word (toks_encode) and on whole pieces, all against hf.
  Measured and not taken (2026-10-04, experiment not merged): a filter in front of the pair
  set. 64 KB, one 64-bit word per key ((key * FIB64) >> 51) and two bits in it (id(y) mod 64, id(x) mod 64: the
  shift counts K7 already holds), built at load from the set itself, so a spanned pair always passes (a test
  walked all 33,945 of gemma 4's; 1.8% of random keys pass). On the bench text it answers "not spanned" for 80%
  of cjk's pair-set queries (972,816 per 3.1 MB, 18% of them spanned) and 20% of ml's (79% spanned). K7 sent a
  clear bit's probe to two empty slots in L1 (no branch: a branch on the filter goes the rare way on ~20% of the
  queries in both, m2ultra2 cjk pass 263 -> 234 MB/s). gemma 4, 4 KiB, cold / pass / warm MB/s, e2e_commits.sh
  abba vs d11f9d1, ids equal: m2ultra2 M2 Ultra cjk 197 -> 197 / 258 -> 259 / 287 -> 288, ml pass 124 -> 122, warm
  135 -> 132; gb10d X925 cpu 7 cjk 190 -> 205 / 243 -> 265 / 292 -> 312, ml and en -1%. Below +10% on cjk
  pass, and M2's ml loses 2%: M2's L2 holds the 2 MiB set and out-of-order execution hides its latency (taking
  80% of the set's loads away moved the cjk scan 0% there and -5% on the X925, whose L2 is a private 2 MB). Probe
  builds that answer every plain pair without the set overstate its share (m2ultra2 cjk scan 7.02 ms -> 3.25
  answered "spanned", 4.47 answered "cut"; tr9970x avx2 "spanned" 7.04 -> 2.97): they also change which words are
  one char, i.e. the flush loop's per-word branch (a one-char word's id vs word(): ~850k words per cjk pass,
  unpredictable in CJK), which is where the rest of cjk's scan time is suspected to go; gb10d and tr9970x allow no
  PMU access (perf_event_paranoid 4) to count it.

  5.6 the split planner's cut (toks_spm_cut, split.c): cutting one text unit into text[.., c) and text[c, ..),
  the second encoded as a continuation, keeps its ids when the second unit gets no prefix symbol the whole has
  none of (FIRST: a continuation never does; ALWAYS: only when its first image is the replacement; GAP: always
  does, so never) and c is a Metaspace piece start or a certified word cut (§5.5).

  5.7 premerge: certified ascii pairs (spm_build.c spm_apm, spm_c.c symbols(); t->apm in layout.h's TOKS_APM_*
  format, as byte-level bpe's, kernels.md 5.1). Two adjacent ascii input bytes x y of a word whose images' ids a, b
  have a merge (a, b) -> t of rank R start as the one symbol t when neither the class of the input byte before
  them is in the entry's before mask nor the class of the byte after them in its after mask (none: never risky).
  Pairs and chars are taken left to right without overlap; the first pair of a word with the virtual prefix is
  never taken. The masks, by one pass over the merge list in rank order: before = the end classes of x over every
  merge (x, a) or (x, t) of rank <= R; after = the start classes of y over every (b, y) or (t, y) of rank <= R. The
  end (start) classes of an id: those of every input byte that can end (start) the id's string as a symbol: of
  the last (first) char cp, cp's own bytes unless the fold maps cp away, and every input char the fold maps to cp
  (" " for "▁"); of a byte-fallback id <0xHH>, the byte HH; of unk, all.
  proof. kernels.md 5.1's claim and its proof (1-3) carry over word for word with "the byte before / after B" read
  as the input byte before / after the pair: they use only that every symbol is a vocab string, a merge joins
  two adjacent symbols into their concatenation, one string per id, the pair of least rank merges first (ties
  to the leftmost; a pair's rank is its last listing, §5.3), and that any symbol x that ends right before the pair
  has the input byte there as an end byte. That last fact holds here because the end classes are complete: a
  symbol's last input char is its last leaf's (a char's, a byte-fallback byte's or an unk's), and no merge joins a
  byte-fallback or unk id (else the table is not built), so the leaf is a char of its string's last char's
  preimage; and a symbol does end right before the pair because no char vanishes (the table needs an unk token,
  §5.2 c). Earlier listings of a re-listed pair and merges the table drops (6.5) only add risk.
  Measured, 4 KiB cold / pass / warm vs §5.3's asm loop alone: gb10d X925 gemma 4 en 1.15 / 1.13 / 1.07x, code
  1.05x, ml 1.16x; mistral-v0.3 / phi3 en 1.19-1.20x, code 1.09x, ml 1.18x; tr9970x Zen 5 gemma 4 en 1.14x, ml 1.10x,
  mistral-v0.3 en 1.15x, ml 1.14x; zh 0.99-1.01x. Load unchanged (tr9970x gemma 4 319 -> 311 ms: the static table's
  model runs get faster); tables: 256 KiB. tests/data/spm pm_order: "c ab" ranked before "a b", so "dcab" keeps
  [dc, ab] only with the (x, t) term (without it [d, cab]).


6. tables, not text: the text model folded into the char table
---------------------------------------------------------------

the maintainer doctrine (2026-10-04): "copy memory as little as needed"; transform the tables, not the text. toks
never writes a normalized string. It reads the caller's bytes once and computes hf's pieces and initial
symbols (§3-§5.2) from tables compiled at load. This section states that design and proves it gives hf's
ids on every input, including inputs that hold U+2581 themselves.

  6.1 the fold. Read the text model's steps in order and keep (phi, pi):
        phi  a map char -> char, the identity but on its sources; pi  the prefix char, or none.
        Replace(a -> b), a and b one char each:   phi := (a -> b) o phi; pi := b if pi == a
        Prepend(p), p one char:                   pi := p (a second Prepend is refused)
        Prepend(""):                              nothing
        Metaspace(r, scheme, split), after them:  phi := (U+0020 -> r) o phi; pi := r if pi == U+0020
      Refused, by name, because they are not char -> char or would need more than one prefix symbol:
      Replace with a pattern or content of other than one char; Prepend of more than one char; a Metaspace
      prefix when pi is set to another char than r; Metaspace split after a Prepend (a piece of the prefix
      alone has no width in the input); ignore_merges unless phi is the identity and pi is none (§5.1 looks
      the whole piece up, which needs its image); an image phi(c) != c or a prefix that is not a one-char
      vocab string (then its symbols would be the IMAGE's fallback bytes, more symbols than input bytes).
      No pinned or census file needs any of them.

  6.2 lemma (normalization is char-wise). For a gap g = c1 .. cn (n >= 1) and a text model the fold accepts,
      N(g) = [pi] phi(c1) .. phi(cn); N("") = "".
      proof. Induction over the steps. Replace with a one-char pattern: the leftmost non-overlapping
      occurrences of a one-char literal are all of its occurrences, and a one-char content keeps the length,
      so the step maps each char by (a -> b) and keeps a non-empty string non-empty; on the prefix char, if
      any, it acts the same way (that is how pi was mapped). Prepend(p) prepends p to a non-empty string
      (every gap stays non-empty under the char maps, so it always fires on non-empty gaps and never on "").
      Metaspace's step 1 is the char map U+0020 -> r on the split, prefix included. Composing gives the
      claim. Original offsets: phi keeps one char per char, so char i of the image is input char i; the
      prefix has none (zero width).

  6.3 the pieces and the virtual prefix. Let u be a text unit (a phase-0 gap, or a split of one between
      phase-1 matches), at_start = u begins at input offset 0 and the call is not a TOKS_CONTINUATION.
      u's string as hf's pre-tokenizer sees it is [pi] phi(u) (6.2); toks gives u ONE virtual leading
      symbol, the prefix char's id, when
        GAP     pi is set (normalizer Prepend): every non-empty gap;
        ALWAYS  Metaspace always: phi(u[0]) != r;
        FIRST   Metaspace first: phi(u[0]) != r and at_start;
      which is §4.1 step 2 rewritten through 6.2 (u starts with r after step 1 iff phi(u[0]) == r). With
      split = true a piece starts before every input char whose image is r, except at the unit's first
      position (the prefix's when there is one): §4.1 step 3 on [prefix] phi(u). A Split that never cuts
      (§1.1) is nothing. So hf's pieces are toks' pieces with phi applied and the prefix symbol in front.

  6.4 theorem (the fold). Under 6.1, for every input, toks' ids equal hf's.
      proof. By 6.3 the pieces agree. In one piece, hf's §5.2 walk reads the image chars phi(c); toks reads
      the input chars c through the char table, whose entry for c is built from phi(c): the id of phi(c)
      when it is a one-char vocab string, else none. For c with phi(c) != c the image is a vocab char
      (6.1), so both walks emit that one id (step a), nothing pending. For c with phi(c) == c the walks read
      the same char: the same id, or the same byte-fallback ids (c's own bytes), or the same unk decision.
      The prefix symbol is the prefix char's id, step a in both. So the walks visit the same sequence of
      (vocab id | fallback ids | unk) per char, with the same pending state: the initial symbols are equal,
      the merge loop (§5.3) reads nothing but them, and the outputs are equal. A literal U+2581 in the input
      is phi-fixed (no step maps it) and its entry is id("▁"), the entry U+0020 gets from phi: in both hf
      and toks the two are indistinguishable from the normalizer on (§9). ∎

  6.5 unreachable strings (the maintainer: "drop vocab entries that contain a literal ' '"). A char c with phi(c) != c
      that no char maps to and that is not the prefix never occurs in a piece (6.2). The reachable ids are
      the one-char vocab strings of every other char, every byte-fallback id, unk, and, to a fixpoint, the
      output of every merge whose two operands are reachable; a merge with an unreachable operand can
      never fire, so the table drops it (the ranks of the others are unchanged), and the pair set (§5.5)
      keeps only the pairs of reachable strings. Gemma 4: the space is unreachable, 0 merges dropped (no
      merge operand holds a literal space), 33,945 pairs. Decode keeps every string (tok_bytes is
      untouched).

  6.6 the word cache. A word is answered by the scratch's dynamic cache (kernels.md §6's format, two ways)
      when its INPUT bytes are <= 15: key = those bytes, length and a bit for the virtual prefix. Equal input
      bytes and prefix bit give equal images, hence equal ids (6.4), so the key is exact. A space is one
      byte in the key where "▁" would be three: the reviewer measured that 41.7% of code bytes lie in words
      longer than 15 bytes when keyed on the normalized text and 14.5% when keyed on the raw text.
      A word that is one 2-4 byte vocab char without the prefix (79% of gemma 4's zh words) skips the cache:
      its ids are [the char's image id] (§5.2 a, one symbol, nothing to merge; 5.1 gives that id too), which
      the scan read from the char table already. TOKS_SPM_NOCACHE (the tests' control) turns that shortcut
      off with the cache.
      Wide entries: a word of 16..30 bytes, or of up to 15 bytes with 5..7 ids, takes a whole 64-byte bucket
      (spm_c.c): key = 32 bytes (the word's bytes 0..14, a marker byte 0x40 | 0x20 with the prefix, its bytes
      15..29, then 0x40 | len; zero-padded), val = up to 7 ids, count and tag in the first 16 bytes as a
      short val. The marker bytes are never a short key's byte 15, so no short probe matches a wide key. A
      short word's wide entry sits in its own short bucket (one line answers both probes); a longer word's in
      the bucket of its key's hash. Measured on gemma 4: 15,243 code words per 3 MB were over 15 bytes (11%
      of the bytes, e.g. 12 spaces + "return": 2 ids) and en's paragraph ends ".\r\n\r\n" have 5 ids; each
      was a model call per occurrence before.
      Size: the scratch's whole caches region (TOKS_SCRATCH_CACHE_MIB(n)'s n MiB, else 2 MiB) from the first
      call, not K5's 2 MiB until warm (kernels.md §7): a miss here is a model call (~280 ns for a gemma 4 en
      word), so a fresh scratch loses nothing to the larger region, and a switch at K5's warm point drops every
      entry filled before it (gb10e, 32 MiB: pass and warm 2-10% below no switch). An spm context runs no K5, so
      the half of the region K5 keeps for pieces over 15 bytes (its long buckets and arena) is word cache too:
      toks_scratch_init marks whose the long region is (off_long 0: spm buckets), and a move between an spm
      context and a byte-level one on the same buffer is a first init (an arena of raw text read as buckets could
      forge entries; tests/c/test_spm.c moves() forges them), while moves within a family stay O(1).
      Model calls of a second pass over the same text (gemma 4, 4 KiB chunks, tools/bench/spm_stages.sh prof,
      a word cache of 2 / 4 / 8 / 16 MiB): en 7,158 / 4,000 / 1,589 / 849 of 416,617 words; code 1,192 / 955 /
      771 / 663 of 149,361; zh 6,267 / 4,663 / 3,601 / 3,300 of 225,072; ml 116,906 / 88,914 / 55,053 / 31,346
      of 1,235,886. Spanning the region (2026-10-04; e2e_commits.sh abba, master d11f9d1 -> this, ids equal):
      CACHE_MIB(8), 4 KiB, warm en x1.18 / x1.19, code x1.15 / x1.12, ml x1.24 / x1.21, zh-ja-ko x1.10 / x1.09
      (m2ultra2 M2 Ultra / tr9970x Zen 5 cpu 21); pass x1.00-1.05; cold x0.99-1.00, as on an identical path at flags 0.
      CACHE_MIB(16): warm x1.02-1.16, cold x0.99-1.00; the whole-corpus pass of cjk / ml x0.986 / x0.993 on
      m2ultra2 (first sights spread over 16 MiB instead of 8). The arm cold tax of a larger scratch (usage.md:
      CACHE_MIB(8) cold -4..-8%) is K5's: gemma 4's cold does not fall with the budget (m2ultra2 en 4 KiB 152 / 155 /
      154 MB/s at 0 / 8 / 16 MiB).
      Measured and not taken: a long-word cache (words over 15 bytes through K5's long-piece cache, lc_get /
      lc_put). The words the word cache can never hold (over 30 bytes, or up to 30 with more than 7 ids) make
      273 (en), 550 (code), 3,054 (zh), 5,851 (ml) of the model calls above at every size: 3.8 / 46 / 49 / 5% of
      them at 2 MiB, 32 / 83 / 93 / 19% at 16 MiB. ml's gap is capacity, not long words, and the long-piece
      cache would take half the region from the short words. The only shape worth a later card: a small arena
      for exactly those words (their entries are ~0.03-0.7 MB per corpus) carved from the region, for zh and code
      at large budgets (~13-14% of their warm pass at 16 MiB, at ~260-400 ns per model call).
      Rejected (measured on gb10e and M2): a bucket pair as four ways (the front bucket probed inline, the
      back one out of line, a back hit moved to the front): en's second pass at 2 MiB +19% (model calls
      16.3k -> 6.9k), but the second line per miss costs cold 3-5% on every corpus and code / ml / cjk do
      not move; inserting at way 1 with promotion on hits: no fewer misses than today's policy.
      Rejected too (gb10d X925, 32 MiB, ids equal): a span cache in the long-cache region (an spm context
      runs no K5): runs of words between certified cuts at a space after a non-space, over 15 bytes, keyed by
      bytes and the prefix bit, probed at the run's start. zh's warm pass gained 2.64x only because a 4 KiB
      chunk without spaces is one run (a replay of the same chunk), while its pass lost 14% and en / code warm
      lost 27-38% (a call at every word start). Limited to runs whose first char takes 3-4 bytes and split at
      newlines too, it still cost en / code warm 14-22% (the test at every word start in the scan's flush
      loop), and only 22% of zh's bytes fell in probed runs (zh warm 0.97x).
      The static word table (t->words, which an spm context uses for nothing else; kernels.md §6's format,
      probe and sizing: two buckets of two ways, pow2(n / 2 + 1) buckets, filled in id order up to load 0.85).
      One entry per vocab string whose input form has 1..15 bytes: the string with each image char that a
      one-byte input char maps to written as that byte ("▁" -> " " under Replace(" ", "▁")), valued by running
      the model on that input form at load. The entry is exact whatever the form: equal input bytes and prefix
      bit give equal ids (6.4), so the key only chooses which words are precomputed (the prefix bit is 0 in
      every entry; a first word with the prefix just misses). word() probes it after a cache miss, before the
      model, and a hit fills the cache with the epoch's tag (K5's warm rule), so a word seen again is one line.
      Why: 80.1% of gemma 4's cold en model calls (104 per KB, ~6 symbols each) return one id, a vocab
      string's (code 74%, ml 50%; mistral-v0.3 en 68%). Measured, 4 KiB chunks, cold / pass / warm vs master
      82cdeda, which also lacks §5.3's pair filter (1.03-1.09x of these on its own): gemma 4 en 2.29 / 1.43 / 1.24x on gb10d X925 (cold
      62 -> 143 MB/s), 2.20 / 1.39 / 1.23x on tr9970x Zen 5 (55 -> 122); code 1.55-1.62x cold, ml 1.33x, zh 1.11-
      1.15x; mistral-v0.3 / phi3 en 1.73-1.83x, code 1.37-1.44x. Load: gemma 4 274 -> 324 ms (gb10d), 243 ->
      310 ms (tr9970x); mistral-v0.3 / phi3 +5 ms. Memory: gemma 4 8 MiB (207,066 of 247,089 forms placed), mistral-v0.3 /
      phi3 1 MiB (26,717 / 26,624). Measured and not taken: a table for every form (16 MiB for gemma 4): cold en
      +4%, code +2%, ml 0.
      Where gemma 4's cold 4 KiB time goes (2026-10-04; tools/bench/spm_stages.sh, tr9970x Zen 5 cpu 21; cold
      en / code / ml / cjk 170 / 164 / 114 / 203 MB/s): model calls 40 / 51 / 48 / 18% (the words that are no single
      vocab string: en 46,906 calls a 2 MB pass at 110 ns; K6's merge loop), the miss path beyond the model (the
      static probe, the fill, the keys) 38 / 28 / 24 / 32%, the scan 10 / 8 / 14 / 45%, cache hits 8-14%. Measured
      and not taken there, both as ablations (ids equal): no word-cache fill (every within-call repeat then misses):
      cold x0.99 / x0.69 / x0.92 / x0.96 on tr9970x, the non-model time unchanged on en: a fill costs about what the
      static re-probe of a repeat costs. A static table holding exactly the corpus's words (an oracle, 0.5-2 MiB):
      cold x0.94 / x0.99 / x1.01 / x0.99 on tr9970x, x0.99 / x0.98 / x1.07 / x1.03 on gb10d X925: the window batch already
      hides the 8 MiB table in cold. Its buildable form, a hot table of the lowest ids, would hold only 26% / 42% of
      en's distinct static words at 16k / 32k ids (ml 17 / 28%, cjk 12 / 19%): gemma 4's id order is frequency order
      only for the common words.
      The window batch (spm_c.c window_batch; spm-speed, 2026-10-04). A static-table answer is a line from an
      8 MiB table: when a window's words find many, their lines are worth asking for together. A window after
      one whose words got >= 3 static-table answers (word_slow returns that fact in bit 63 of its count, so the
      counter stays in a register) goes out of line: its words' keys and both lines (the cache's, the table's)
      first, then each word in order through the same probe / table / model; any other window runs the plain
      loop. Same answers by construction (the probes only move earlier); ids= equal per state in every A / B run.
      Measured, gemma 4, master fc5808a -> this, cold / pass / warm, gb10e X925 cpu 9, 4 KiB: en 1.07 / 1.00 /
      1.00x, code 1.00 / 0.99 / 0.99x, ml 1.03 / 1.00 / 1.00x, zh 1.00x. Measured and not taken (same harness): every
      window batched (keys + both lines, then in order): en 1.12 / 1.10 / 1.07x, ml 1.08 / 1.07x, but code warm
      0.93x, zh pass / warm 0.92 / 0.90x; without the table line: en cold 1.06x; ascii windows only, keys built
      twice: code warm 0.89x; one at a time until the window's first miss, then deferred: 0.92-0.97x everywhere,
      warm too (the loop's code alone); batching after any miss: code pass 0.97x; after any static answer with
      the count through a pointer (it lives in memory): code warm 0.96x; after >= 1 with the count in a register:
      en cold 1.09x but code pass 0.97x, zh cold 0.98x.

  6.7 phase 1 under a normalizer. Added tokens with normalized=true are matched by hf on the normalized
      gap (§2.2). toks does not write it, so they are refused while a normalizer is present [added token
      normalized=true under a normalizer]: llama 1 and the 2023 llama 2 file (§1.2). Matching their forms
      on the images, with the prefix as a zero-width first char, is the fold of phase 1 (m2). Without a
      normalizer (Metaspace only) the form is the content and the driver matches it on the raw bytes.

  6.8 the builder (spm_build.c). Two phases: A, a temporary block (the raw char table, byte2id, the reachable ids
      as a fixpoint over the merges, the spanned pairs counted in a table sized by the vocab's bytes); B, the
      block the context keeps, sized exactly:
        the fold          phi (Replace steps, Metaspace's U+0020 -> replacement), the prefix, its mode
        byte2id           the <0xHH> id of every byte (TOKS_SPM_NONE when missing)
        char table        the entry of every input code point: its image's id, small index, PAIRED
        merge table       layout.h's format (bpe_build.c's toks_merge_slots), prio = the rank, a pair listed
                          twice keeps its last rank; merges with an unreachable operand are dropped (§6.5);
                          rank2id for every rank
        vhash             layout.h's vocab hash, only with ignore_merges (identity text model only)
        pairs + cut       the spanned pairs of reachable strings keyed by ids (§5.5) at load <= 0.5, the 129 x
                          129 cut bitmap, the scan's 128 x 128 byte-pair table (spm.h cut_ab)
        holes             ids without any string (decode skips them)


7. post-processing
-------------------

  TemplateProcessing, single sequence: the ids of every SpecialToken before $A (each expands to its
  special_tokens entry's ids), the encoded text, the ids of every SpecialToken after $A. Only with
  post-processing on (hf add_special_tokens=true); off: nothing. No post-processor: nothing.


8. decode: ids -> text
-----------------------

  8.1 token strings. Per id in order: an added token's string is its NORMALIZED FORM when it has
      normalized=true and the tokenizer has a normalizer (llama2: "▁<s>"), else its content; any other id
      its model-vocab string; an id with neither (an id hole, an id beyond the table) is skipped.
      Precisely (hf's normalized_cache, keyed by id and only ever written while the list is read): a
      normalized token whose form differs from its content stores the form under its id; the last token
      listed under an id holds the id; the id's string is the stored form if any, else that last token's
      content. Two tokens can share an id when the vocab has id holes (an added-only token gets
      vocab.len(), which can be a vocab id): then a stale form outlives its token.
      skip_special_tokens drops a string that EQUALS the content of a special added token. llama2's
      "▁<s>" never equals "<s>": hf 0.23.2 decodes [1, 22172] to "<s> hello" with skip_special_tokens on
      and off (kept exactly). toks builds each id's string by this rule and sets decode's skip bit from it at
      compile, for every family (compile.c): an added id's bit is whether its string equals some special's content;
      K1's index holds the token that holds each id, as hf's tries do (a shadowed token is never matched).
  8.2 the chain: each decoder maps the list of strings to a new list, in order; the result is the
      concatenation of the last list. No decoder: the strings joined with " ".
  8.3 Replace(String p, c): in every string, p -> c (leftmost non-overlapping).
      ByteFallback: a string is a byte token when it is exactly 6 bytes, starts "<0x", ends ">" and its
      bytes 3..4 parse as Rust's u8::from_str_radix(.., 16): two hex digits of either case, or "+" and one
      hex digit ("<0x+F>" is byte 0x0F). Each maximal run of consecutive byte tokens becomes ONE string:
      the run's bytes when they are valid utf-8 as a whole, else one U+FFFD per byte of the run (not per
      maximal subpart: <0xE2> <0x41> gives two U+FFFD, the "A" is lost). Other strings pass unchanged.
  8.4 Fuse: one string, the concatenation. Strip(c, start, stop): per string, drop up to `start` leading
      chars equal to c and up to `stop` trailing ones; hf panics when stop exceeds the string's length or
      the two cuts cross (no pinned file has stop > 0).
  8.5 Metaspace(r, prepend_scheme): per string i, every r becomes " ", except in string 0 (the first that
      survived 7.1) when prepend_scheme != never: there EVERY r is removed, not just a leading one.
  D1 on [▁hello, ▁world] -> " hello world" -> strip one leading " " -> "hello world"; D2 keeps the leading
  space: gemma decodes [▁hello] to " hello".
  8.6 toks runs the chain in one pass into the caller's buffer (no scratch, spm_c.c toks_spm_decode): the
      accepted chains are at most one per-token step (Replace or Metaspace, whose strings hold no byte-token
      char, so they never change which strings are byte tokens), then ByteFallback, Fuse, Strip, in this
      order (Strip only after Fuse). A byte run is resolved at its end from the ids themselves (its bytes,
      or one U+FFFD per byte); Strip drops leading content chars of the output as they come. Bytes beyond
      cap are counted, not written.


9. notes that bite
-------------------

  - Prepend runs per phase-0 GAP: with T1, "a<s>b" in tinyllama (normalized=false specials) gives ▁a, <s>,
    ▁b: every gap after an added token gets its own "▁". Metaspace first (T4) gives the prefix to the first
    split only: mistral "<s>b" -> <s>, b (no ▁).
  - llama2's normalized=true specials (§2.2): "<s>" is only recognized after a space or at a gap start, and
    the space before it is consumed by the match ("a <s> b" -> ▁a, ▁<s>, ▁b). The re-saved llama 2 files
    (tinyllama, unsloth) flip this to normalized=false; the two are different tokenizers.
  - decode of llama2's specials does not skip them (§8.1), and their string carries the "▁": " <s>" in
    the middle of a decode, "<s>" at the front after Strip.
  - Prepend never checks for an existing "▁"; Metaspace always / first do. A text beginning with " " gives
    "▁▁..." under T1 and "▁..." under T4.
  - only U+0020 becomes "▁". Every other whitespace char is a plain char (tab is <0x09> in llama, a vocab
    char in gemma; U+3000 and U+00A0 have vocab tokens or byte fallback).
  - a literal "▁" in the input is indistinguishable from a replaced space from the normalizer on: llama2
    "▁x" -> ▁, ▁x (Prepend's ▁ plus the text's); gemma's "▁▁" added tokens only see literal ▁ (phase 0
    runs on the raw text, before Replace).
  - byte fallback is all-or-nothing per char; a pending unk is not flushed by byte-fallback ids (§5.2 b);
    unk ids come out after them. Reachable only when some byte token is missing AND a char needs it
    (gemma1/2 miss <0x09> but have "\t", so no pinned file reaches it; the synthetic tokenizers do).
  - no pre-tokenizer (or gemma's Split that never cuts) means one bpe piece per gap (§1.1, §4.4): the merge
    loop must handle pieces as long as the text (O(m log m) per piece with a heap, as hf); §5.5's cuts give
    the same ids word by word.
  - gemma 4 instruct and base files differ only in the template: base adds <bos> (2), instruct adds nothing.
  - gemma 3 / 3n / 4 have the vocab string ">▁</" (gemma 4 id 107068): a blanket "cut before every ▁" is
    WRONG for them: hf encodes "<p> </p>" as <, p, >▁</, p, > (the merge crosses the space). The pair test
    of §5.5 is what certifies.
  - merge ranks are NOT monotone in the merged id for any pinned file: the rank must be stored (rank2id).
  - the empty-span rule of §2.3 is new relative to kernels.md §4 (not folded in there yet).
  - the cut rule's preconditions (§5.5 (i)-(iv)) are load-time refusals: code that copies the rule copies them.
  - toks never builds N(g): the space byte and a literal "▁" share one char-table entry (§6.4), and the
    pieces, prefix and words are computed on the caller's bytes. A literal space in the vocab is
    unreachable after Replace / Metaspace; its merges are dropped from the table, not from decode (§6.5).
  - single_word's word chars are the rust regex crate's \w as compiled into hf 0.23.2 (probed through hf
    over every scalar: 144,667 chars, U+200C / U+200D included); a fresh unicode.org property file is not
    the same set (17,591 code points differ).
  - decoding is not a byte-exact inverse: Strip eats a real leading space ("  a" round-trips to " a"), and
    invalid byte runs become U+FFFD (SPEC §3.4: decode(encode(x)) == x is never a criterion).
  - invalid utf-8 (SPEC §3.3, no hf oracle): an invalid byte is its own atom; it is never U+0020, never
    "▁", and no vocab or pattern string contains it, so normalizers and Metaspace pass it through and the
    model takes step 5.2 b for it (its <0xHH> token) or c / d when that token is missing. tokref decides.


10. evidence
-------------

  Every number below is a run's output; the logs stay on the host named. hf is
  tokenizers 0.23.2 (uv run --with tokenizers==0.23.2).

  10.1 the c side end to end (tests/spm/run.sh: tests/spm/gen.py | tests/spm/check.c; gb10c, linux arm64, 20
       cores, 18 pipelines, code bae8373, load 1.27 before the run). Per text: the gap through toks_encode
       (mode NONE, no pp; TOKS_CONTINUATION for a gap after an added token) AND through toks_spm_encode on
       whole pieces without the cache, the file as is through toks_encode in mode ALL with post-processing and
       in mode NONSPECIAL without, toks_decode of hf's ids with skip_special on and off, the model alone on
       unsubstituted pieces.

         tokenizer           gaps (x2)  full-file   pieces    decodes  mismatches
         gemma4              1,080,000  2,160,000  537,119  1,079,223  0
         gemma4-base           360,000    720,000  179,112    360,048  0
         yi-dolphin            180,000    360,000   89,260    180,124  0
         gemma3                180,000    360,000   89,460    180,228  0
         embeddinggemma        180,000    360,000   89,460    180,228  0
         tinyllama             180,000    360,000   89,472    179,640  0
         mistral-v0.1          180,000    360,000   89,472    179,640  0
         gemma2                180,000    360,000   89,675    179,639  0
         gemma1                180,000    360,000   89,618    179,974  0
         codellama             180,000    360,000   89,362    179,845  0
         mistral-v0.3          180,000    360,000   89,723    180,016  0
         mistral-small-2409    180,000    360,000   89,745    180,199  0
         gemma3n               180,000    360,000   89,352    180,243  0
       (equal counts for files of equal vocab size: the generator's draws depend on the seed and the vocab
       size only.) Refused by name at load: llama1 and llama2 (2023 file) [added token normalized=true under a
       normalizer], phi3 / phi3.5 [added token lstrip/rstrip (m2)], yi [added_tokens repeat a content].
       The first run of the same harness at b4aa272 (same encode, one-pass table build): gemma4 1,080,000 gaps,
       0 mismatches.

  10.2 make -j8 test (code d553b4d, master 2c4283e merged): macos-arm64 (this Mac), linux-arm64 (gb10e,
       -j20, load 11.43; its cache lacks the glm / gpt-oss / qwen target files: test_targets skips them),
       linux-x86_64 (tr9970x, -j8, load 4.71): every suite 0 failures. test_spm: 53,939 checks (13 hf-generated
       fixtures through toks_load / toks_encode / toks_pieces / toks_decode, both encode paths, pieces against
       hf's pre-tokenizer offsets, 14 refusals by name, the gemma 4 goldens in modes NONE and ALL, its pieces,
       <bos> and skip-special decode); test_targets: gemma-4 compiles (vocab 262144, merges 514906, added 24).

  10.3 the python model against hf (tests/model/run_spm_fuzz.py, 20 processes per host on the GB10 boxes):
         real files: 23,300,000 texts over the 18 pins (1,000,000 for each of gemma4, gemma4-base, yi-dolphin,
         gemma3, embeddinggemma, tinyllama, mistral-v0.1, gemma2, phi3.5, phi3, gemma1, llama1, codellama,
         mistral-v0.3, llama2, mistral-small-2409, yi, gemma3n; 300,000 more on eleven of them with the cut
         checks): 139,800,000 encodes (modes ALL / NONSPECIAL / NONE x post-processing on / off), 60,581,656
         decodes, 11,652,250 pieces through the model alone, 16,952,387 word-by-word encodes (§5.5): 0
         mismatches.
         synthetic: 100,000 small tokenizers built to reach what real files do not (missing byte tokens, unk
         absent / outside the vocab, fuse_unk off, ignore_merges, duplicate and competing merges, id holes,
         every normalizer / pre-tokenizer / decoder shape, added tokens with every flag) x 100 texts:
         60,000,000 encodes, 23,948,630 decodes, 5,002,818 pieces, 15,002,818 word-by-word: 0 mismatches
         (gb10e). The run before found two hf rules the model then took in (§8.1's id-keyed normalized
         cache; single_word's \w is hf's own, §9).
  10.4 mutants (tests/model/spm_mutants.py --synthetic 300 --synth-n 40 --n 300; gb10c, one process, load
       3.59 before): each breaks one rule of this file (§5.2 unk / fuse / fallback, §5.3 ties / duplicates, §5.1
       ignore_merges, §4.1 first / split, §3 Prepend, §8.3 byte runs, §8.1 skip_special, §2.2 / §2.3 phase 1,
       §8.4 Strip, §5.5 two wrong cut rules, §8.5 Metaspace decode); the harness must report a mismatch for
       every one: 17 / 17 killed, the unmutated baseline 0 mismatches.


11. what toks accepts, and the c side
--------------------------------------

  config.c routes a tokenizer.json whose model is BPE (or untyped) and which has no ByteLevel normalizer,
  pre-tokenizer, decoder or post-processor here (src/core/config.c reads the model, the text model and
  the decoder; config.c the added tokens and the post-processor); spm_build.c refuses what the fold (§6.1)
  cannot absorb. Anything else is TOKS_E_UNSUPPORTED with the name in brackets, or TOKS_E_FORMAT where hf
  refuses the file too:
    model        type BPE or untyped; dropout null [model dropout]; continuing_subword_prefix /
                 end_of_word_suffix null or "" [model continuing_subword_prefix]; any fuse_unk, byte_fallback;
                 ignore_merges only with the identity text model [ignore_merges with a Replace / Prepend /
                 Metaspace text model]; unk_token null, a vocab string, or missing from the vocab only when
                 byte fallback covers all 256 bytes [model unk_token missing from the vocab (...)]; vocab ids
                 < 2^21 - 1 with holes allowed, one string per id [model.vocab: two strings share an id], no
                 string twice [model.vocab repeats a string]; merges as strings or pairs, < 2^22
    normalizer   null, Prepend (one char), Replace (String, one char -> one char), Sequence (two levels), at
                 most 8 steps (§6.1's refusals) [normalizer Replace Regex] [normalizer (sentencepiece-style
                 bpe: Prepend, Replace, Sequence)]
    pre-token.   null; one Metaspace (one-char replacement; prepend_scheme first only without a normalizer
                 [Metaspace prepend_scheme first after a normalizer]); Split only where it can never cut
                 (gemma 2 / 3 / 3n / 4, §1.1) [pre_tokenizer Split (one that can cut)]; Sequence of those
    decoder      null; at most one Replace (String, no byte-token char) or Metaspace, then ByteFallback, Fuse,
                 Strip (one char, stop 0, after Fuse) in this order [decoder order (...)] [decoder Strip
                 stop > 0]
    added        config.c's readers (m1a: single_word and lstrip / rstrip refused, a repeated content
                 refused); normalized=true refused under a normalizer (§6.7)
    truncation / padding: null (config.c)
  the python model (tests/model/spm_model.py) accepts more: any literal Split, Metaspace first after a
  normalizer (with §3's original offsets), normalized added tokens. No census file needs what the c side
  refuses but phase 1 under a normalizer (llama 1, the 2023 llama 2 file) and lstrip / rstrip (phi-3).

  the c side (all in toks_load / toks_encode / toks_pieces / toks_decode):
    load      config.c -> compile.c (tok_off / tok_bytes: the raw strings; the added-token index; the
              template) -> spm_build.c (one block: byte2id, the folded char table, the merge table of
              reachable merges and rank2id, the pair set, the cut bitmap, the holes). Gemma 4: 4,449,216 +
              11,798,528 bytes.
    encode    the driver's segments (kernels.md §4, §7) -> toks_spm_encode per text unit: one scan of the
              caller's bytes finds pieces and certified words, each word from the cache or the model
              (§5.2-§5.3, a heap of (rank, position) over the word's symbols, work <= 32 bytes per byte).
    decode    toks_spm_decode (§8.6).
  Not yet: asm (the rent rule), phase 1 under a normalizer, the static words table for spm.
