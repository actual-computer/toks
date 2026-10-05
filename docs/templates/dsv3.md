dsv3: the K3 scan template TOKS_TMPL_DSV3
===========================================

This file extends docs/kernels.md the way its §3 does for cl100k (and docs/templates/o200k.md for o200k): §2
(atoms and classes) and the K3 contract (§1, the output paragraph of §3: ends, cap, pos, resume, lookahead to the
segment end only) apply unchanged; this file says which pieces the template makes. src/core/k3_dsv3_c.c is its
executable reading; tests/model/dsv3_model.py is its python transcription, diffed against hf tokenizers 0.23.2
(§5).

  toks_k3_scan_dsv3_<tier>(const toks_tables *t, toks_k3_args *a)      tmpl = TOKS_TMPL_DSV3, tmpl_params 0


1. the pre-tokenizer
--------------------

deepseek v3's pre-tokenizer is not one Split but a Sequence of three, then the byte-level mapping (one line per
alternative of the third pattern; the real strings have no line breaks):

  Sequence[
    Split(Regex "\p{N}{1,3}",                                         Isolated, invert false),
    Split(Regex "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",         Isolated, invert false),
    Split(Regex "[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|
                 [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
                  ?[\p{P}\p{S}]+[\r\n]*|
                 \s*[\r\n]+|
                 \s+(?!\S)|
                 \s+",                                                Isolated, invert false),
    ByteLevel(add_prefix_space false, trim_offsets true, use_regex false) ]

In the file the second pattern holds the literal characters U+4E00 '-' U+9FA5 U+3040 '-' U+309F U+30A0 '-'
U+30FF (json \u escapes, decoded by the json reader: the regex sees the characters, not escapes), and the
third pattern's \r \n are literal CR and LF bytes inside the brackets. The normalizer is an empty Sequence (no
normalization); the model is byte-level BPE (128,000 ids, 127,741 merges).

Sources, read from the files:
  - deepseek-ai/DeepSeek-V3 tokenizer.json, sha256 621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41
    (~/.cache/toks/tokenizers/dsv3, the pin): the chain above exactly (tests/model/dsv3_model.py checks the file's
    three strings, behaviors and the ByteLevel flags against its constants).
  - the census (census/coverage.json, pattern P09): 10 distinct tokenizers share hf's normalized pre_tokenizer
    part 6613066caf77 and normalizer part 29490adc115d (the empty Sequence), so the chain above is the one
    template for all of them:
      51ae50a68b7b  DeepSeek-V3, DeepSeek-V3-0324 (the pin)     232addf8641e  DeepSeek-R1, DeepSeek-R1-0528
      96a097bd7900  DeepSeek-V3.1                                1cc2ffdb5ec2  DeepSeek-V3.2
      cf4cfab77830  DeepSeek-V4-Flash, -0731, V4-Pro, -DSpark    f4f5806a817d  DeepSeek-V4.1-Flash,
                                                                               V4-Flash-Vision-Exp
      2bc0b185cc58  DeepSeek-OCR, OCR-2, baidu/Unlimited-OCR     d9b353ec2d62  stepfun-ai/Step-3.5-Flash
      fd9da9f4fa48  tencent/Hy3                                  fa1c70765b18  tencent/HunyuanOCR
    The deepseek / stepfun files share one model (part 069b30ff069e) and differ in added tokens and post-
    processor (OCR and Step: TemplateProcessing with a BOS; the others: ByteLevel); tencent's model is another
    (120,000 ids, 242 of the 256 byte chars). None of that touches this template.
  - not this template: deepseek-coder 6.7b (P29) and Coder-V2 / coder-v1.5 (P19) chain other Splits plus
    Digits(individual); the R1 distills ship qwen 2.5 / llama 3 tokenizers (cl100k template).

hf's semantics (tokenizers 0.23.2 = commit 88a4498): Sequence::pre_tokenize runs each step over every piece the
previous steps left (sequence.rs, PreTokenizedString::split); Split with Isolated keeps every match and every
non-empty gap between matches as pieces (normalizer.rs split + pattern.rs find_matches for SysRegex, onig
find_iter); each step's regex runs on one piece alone, so a lookahead sees that piece's end as the end of input
and no match crosses a piece boundary. ByteLevel with use_regex false maps bytes and splits nothing.


2. classes
----------

The class tables are built with class_flags TOKS_CLASSES_DSV3 (§6). The class byte of every code point is then a
kind, and nothing else is set:

  base (bits 0-2)  L   \p{L} or \p{M}               (marks are letters: see D2)
                   N   \p{N}
                   WS  \s except CR, LF
                   NL  CR, LF
                   P   \p{P} or \p{S}               (and every invalid atom: kernels.md §2, SPEC §3.3)
                   X   none of these (TOKS_C_X = 5) [^\p{L}\p{M}\p{N}\p{P}\p{S}\s]: unassigned, private use,
                                                    format (U+00AD, U+200B..U+200F, U+FEFF, ...), and the
                                                    controls outside \s (U+0000..0008, 000E..001F, 007F..0084,
                                                    0086..009F)
  bit 7 CJK        U+3040..U+30FF and U+4E00..U+9FA5 (TOKS_C_CJK): the second split's class, two ranges
                   because U+309F + 1 = U+30A0

The pattern's sets are exactly class-byte tests (each probed through hf's onig over every scalar value and
asserted against the built tables, §5):

  [\p{L}\p{M}]                         base L
  [\p{P}\p{S}]                         base P
  [^\r\n\p{L}\p{P}\p{S}]               base in {N, WS, X}, or a mark (base L)
  [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]   an ascii atom of base P (the 32 ascii P|S characters, no other)
  [A-Za-z]                             an ascii atom of base L (no other ascii atom is L)
  \p{N}                                base N;    \s = base in {WS, NL};    [\r\n] = base NL
  [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]    bit CJK

L, M, N, P, S and \s are pairwise disjoint (asserted). No CJK code point is N or \s (asserted), so a CJK atom is
L|CJK (Lo, the Lm iteration marks U+309D, U+309E, U+30FC..U+30FE, and the combining kana marks U+3099,
U+309A), P|CJK (U+309B, U+309C Sk; U+30A0 Pd; U+30FB Po) or X|CJK (the unassigned U+3040, U+3097, U+3098).
An absent atom (at or past len) is in no set.

Why a new base value and bit 7: dsv3 needs nine kinds (P, L, N, WS, NL, X, and L, P, X inside the CJK ranges),
none of the case or fold bits, and no table of another template changes: code points of kind X stay base P in
every table built without TOKS_CLASSES_DSV3, and bit 7 is the per-template set bit (TOKS_C_HAN in kimi tables,
TOKS_C_CJK in dsv3 tables; no table has both).


3. rules
--------

Regions (splits 1 and 2). The segment's N atoms are cut into chunks: a maximal run of N atoms gives pieces of 3
atoms from its start, the last one 1..3 atoms. Every other atom belongs to a region: a maximal run of non-N atoms
whose CJK bits are equal. Split 3 runs inside each region alone. At a piece start i with c the atom at i, let
r = c's CJK bit; an atom is in-region when it is present, not N, and its CJK bit is r, so "an in-region atom of
base B" is an atom of kind B | r. (WS, NL and N atoms never carry the bit.)

Every piece starts where the previous one ended (every atom is taken by some rule below), so the pieces tile the
segment. At piece start i, with c the atom at i and c1 the next one, the piece is given by the first rule that
applies:

  D0 digits.        base(c) = N -> [i, end of up to 3 consecutive N atoms from i).
  D1 ascii word.    c is an ascii atom of base P and c1 an ascii atom of base L -> [i, end of the maximal run of
                    ascii L atoms from c1).
  D2 letters.       base(c) = L -> [i, end of the maximal run of in-region L atoms from i).
                    otherwise, if base(c) is WS or X and c1 is an in-region L atom -> [i, end of the maximal run
                    of in-region L atoms from c1).
  D3 punctuation.   j = i if base(c) = P; j = the start of c1 if c is U+0020 and c1 is an in-region P atom;
                    otherwise D3 does not apply. -> [i, end of the maximal run of in-region P atoms from j), then
                    extended over the maximal run of in-region NL atoms that follows.
  D4 newline runs.  base(c) in {WS, NL}: let R = [i, k) be the maximal run of atoms with base WS or NL. If R holds
                    an NL atom -> [i, end of the last NL atom in R).
  D5 whitespace not before a non-space. R as in D4: the atom at k is absent or not in-region (k = len, or the
                    atom at k is N or CJK) -> [i, k);  R has >= 2 atoms -> [i, start of the last atom of R).
  D6 whitespace.    -> [i, k)  (R is one atom here).
  D7 gap.           base(c) = X (D2 did not apply): let G = [i, g) be the maximal run of in-region X atoms from i.
                    The atom at g is an in-region L atom -> [i, start of the last atom of G)  (G has >= 2 atoms
                    here, or D2 would have applied);  otherwise -> [i, g).

These rules are the chain's semantics written out:
  - regions: split 1 (\p{N}{1,3}, greedy, find_iter restarting after each match) leaves the N chunks above and
    the maximal non-N runs. Split 2 finds no match in an N chunk (no N is CJK) and split 3 none either (each of
    its alternatives needs an L, M, P, S or \s atom; the prefix class admits N, but only before [\p{L}\p{M}]+),
    so a chunk is a final piece (D0). Split 2 cuts a non-N run into its maximal CJK runs and the runs between
    them: the regions. Split 3 sees one region at a time: no piece crosses a region end, and (?!\S) succeeds
    there (D5).
  - D1-D7 are split 3's alternatives in order (leftmost-first, onig): D1 is alternative 1, whose bracket is the
    32 ascii P|S characters and whose [A-Za-z]+ takes ascii letters only (a non-ascii letter or a mark after it
    starts the next piece). D2 is alternative 2: the optional prefix is greedy, so a prefix-class atom before an
    L|M atom is taken, then the maximal L|M run. A mark is in the prefix class and in [\p{L}\p{M}] at once: as
    the prefix it takes the run from c1, without it the run starts at c -- the same span either way, so marks
    are base L and the prefix test is "WS or X" (the class minus marks; N never starts a piece inside a
    region). D3 is alternative 3: ' ?' is U+0020 only, the P|S run is greedy, the tail is the greedy CR/LF run.
    D4-D6 are alternatives 4-6 = kernels.md's A5-A7 with the segment end read as the region end.
  - D7: an atom no alternative can start at is an X atom not followed by an in-region L atom (alternative 1 needs
    an ascii P atom, 3 a P atom or a space, 4-6 a \s atom, 2 an L|M atom at c or at c1 after a prefix-class c).
    hf keeps the span between two matches as one piece (Isolated). The next match starts at the first atom that
    is not X (an L, P, WS or NL atom always starts one), at the region end, or at an X atom whose next atom is an
    in-region L (D2 with that X as its prefix): the maximal X run, less its last atom in that last case.
  - inside a CJK region only L, P and X atoms occur, so only D2, D3 and D7 apply there, and D3 never takes a
    space or a CR/LF tail (neither is CJK).

Notes that bite (each a case in tests/c/test_k3_dsv3.c with hf's pieces; pieces shown as |, ␣ = U+0020):
  - the regions: "a␣␣b" -> a | ␣ | ␣b, but "a␣␣1" -> a | ␣␣ | 1 and "a␣␣中" -> a | ␣␣ | 中 (the whitespace run ends
    its region, D5 keeps it whole); "␣カ" -> ␣ | カ and "\u3000カ" -> \u3000 | カ (no prefix across a region end);
    "\x00中" -> \x00 | 中; "!中" -> ! | 中, "!・" -> ! | ・, "・\n" -> ・ | \n (the tail cannot leave the CJK
    region) while "!\n" is one piece.
  - the ranges are literal, not a script property: 々 U+3005 and 〆 U+3006 (below U+3040), U+9FA6.. and U+4DFF
    are outside: "\u9fa5\u9fa6" -> 2 pieces, "〇〆々" -> 〇 | 〆々 (〇 U+3007 is \p{N}, a digit chunk).
  - split 3 runs again inside a CJK run: "カ・カ" -> カ | ・ | カ; "ー・ー" -> 3 pieces; "゛カ" -> ゛ | カ (U+309B is
    Sk); "゠゠カ" -> ゠゠ | カ; "\u3040カ" one piece (an unassigned code point prefixes letters), "カ\u3040" ->
    カ | \u3040, "\u3040\u3040" one piece (a gap), "\u3097\u3098\u3099" -> \u3097 | \u3098\u3099 (U+3099 is a
    mark: a letter); "\u3041\u3099\u3099" one piece.
  - D1 takes ascii letters after one ascii P|S atom that starts a piece: "!hello", ".com", "'ll" whole, but
    "!!hello" -> !! | hello, "␣!a" -> ␣! | a, "␣.com" -> ␣. | com, "''a" -> '' | a, "a.b.c" -> a | .b | .c,
    "don't" -> don | 't, "'sé" -> 's | é, "!abé" -> !ab | é, "!a\u0301" -> !a | \u0301.
  - the letter prefix is a WS or X atom only, never P, CR or LF: "«a" -> « | a, "!é" -> ! | é, "＄a" -> ＄ | a
    (non-ascii punctuation never starts a D1 piece either); "\tword", "\u00a0word", "\u2028a", "\u0085a",
    "\x00a", "\x7fa", "\ue000a" (private use), "\U000e0001a" (a tag, Cf) are one piece each; "a\x1fb" ->
    a | \x1fb; "\nword" -> \n | word.
  - a mark is a letter: "\u0301x", "e\u0301\u0301" whole, "\u0301!" -> \u0301 | !, "❤\ufe0fa" -> ❤ | \ufe0fa
    (U+FE0F is Mn), but in a CJK region only CJK marks count: "\u4e00\u0301" -> 2 pieces.
  - gaps: "\x00\x00a" -> \x00 | \x00a, "\x00\x00␣a" -> \x00\x00 | ␣a, "\u00ad\u00ad\u0301x" -> \u00ad |
    \u00ad\u0301x, "␣\x00" -> ␣ | \x00, "␣␣\x00" -> ␣ | ␣ | \x00, and an emoji zwj sequence splits at the joiner:
    "👨\u200d👩" -> 👨 | \u200d | 👩 (U+200D is Cf).
  - digits: "1234" -> 123 | 4, "12a34" -> 12 | a | 34, "1½Ⅷx" -> 1½Ⅷ | x (No and Nl are \p{N}), "x'0" -> x | ' | 0,
    "١٢٣٤" -> ١٢٣ | ٤.
  - punctuation is P|S: "😀😀" and "␣😀" are one piece each, "＄￥" one piece, "!!\n\n" one piece.
  - whitespace (cl100k's A5-A7 inside a region): "x␣\n\n␣" -> x | ␣\n\n | ␣, "\n␣␣1" -> \n | ␣␣ | 1,
    "\x0b\x0bx" -> \x0b | \x0bx, "\t\t\n" whole.

Invalid utf-8 (hf cannot see it; kernels.md §2, SPEC §3.3): each byte of an ill-formed sequence is a one-byte
atom of base P without the CJK bit. So it joins P runs and takes a CR/LF tail, never prefixes a letter (unlike
cl100k, where P atoms do): "\xFFabc" -> \xFF | abc, "!\xFF\n" whole, "␣\xFF" whole, "a\xFFb" -> a | \xFF | b,
"\xE3\x82" + "カ" -> \xE3\x82 | カ (a cut CJK sequence is a run of two P atoms outside the カ's region).

Output, cap, pos and resume are kernels.md §3's: ends strictly increasing in (pos, len], a.pos = the last end,
n <= cap. Every rule classifies from its piece start forward (lookahead only); a region end is a property of the
atoms on both sides of it, and a piece start inside a run of N atoms is a chunk boundary, so resuming at any end
written gives the same pieces. Work is O(len): a piece consumes >= 1 atom, every scan walks forward, and D5 / D7
give back at most one atom.


4. the simd tiers
-----------------

The asm parts are o200k's body (k3_avx2.inc / k3_neon.inc under K3_O200K) assembled with K3_DSV3
(k3_dsv3_<tier>.S; kernels.md §3.1 for the part / c twin split): dsv3 as data on o200k's algebra. The match:
  - letters: every L atom (marks are L here) is a case atom in both groups (the slot table sends class L to
    S_AB, and the ascii A-Z a-z masks join it), so a case run is a maximal L run. A CJK letter (L | CJK) takes
    o200k's Han slot: a Han run is a piece of its own, which is D2 inside a CJK region (no prefix can join it: WS
    and N atoms never carry the bit);
  - D2's prefix is WS (S << 1). D1's is an ascii P atom that starts a piece (o200k's one-atom P* run not after
    U+0020) right before an ascii letter, and its piece takes only the ascii letters: a non-ascii letter right
    after that run starts a piece (a carry over the ascii-letter mask finds the run's end);
  - D3 is O3 with NO_SLASH (the CR LF tail); D0 is \p{N}{1,3} (N atoms of any length, counted by first bytes);
  - D4-D6 are O4-O6, except D5's region end: no give-back when the run is followed by an N or CJK atom;
  - everything else cuts the block, and the c twin takes the piece that holds it: an X atom (the ascii controls
    00-08 0E-1F 7F included: the ascii classes are hard-coded, so the part tests for them), a P or X atom with the
    CJK bit, a non-ascii NL;
  - Han shortcut: when the compiler has checked that U+4E00..U+9EFF are all L | CJK (TOKS_TF_CJK_D, set by
    toks_compile_cls_flags with dsv3's own ascii precondition: letters L, the controls outside \s X), a dense
    block's 3-byte leads in that range are Han atoms without a lookup (o200k's CJK lead shortcut on another
    range and slot); without it the zh corpus ran at the twin's speed or below;
  - tmpl_params and TOKS_TF_CJK_L / _B are ignored (the part sets its own: NO_SLASH, no contraction, \p{N}{1,3}).
gigatoken's scalar scanner (deepseek_v3.rs: advance_main / advance_cjk with a cjk_region flag) walks the same
structure; it has no mask path.


5. proof
--------

tests/model/dsv3_model.py transcribes §2-§3 literally: classes from hf's own onig (every scalar value through
Split(Regex(set), 'removed') for \p{L} \p{M} \p{N} \p{P} \p{S} \s, [\p{L}\p{M}], [\p{P}\p{S}], the prefix
class, alternative 1's bracket, [A-Za-z], [\r\n], split 2's CJK bracket and the complement X; §2's claims
asserted on every scalar), the rules over kernels.md atoms (so ill-formed bytes too), the doc's side claims
asserted on every piece. tests/model/run_dsv3_fuzz.py diffs it against the pinned file's own pre-tokenizer
(Tokenizer.from_file(dsv3).pre_tokenizer: the three Splits and ByteLevel; character offsets converted to bytes,
each piece string checked against the byte-level image of its slice, the pieces must tile the text), and on every
7th case also against the three Splits built from the model's constants without ByteLevel. tests/k3/dsv3_run.sh
diffs the c twin against the same oracle (tests/k3/dsv3_gen.py | tests/k3/dsv3_check.c), the class tables of
src/core/classes.c against the probes on every scalar, and ill-formed byte strings against the python model.

Receipts, 2026-10-04 UTC, hf tokenizers 0.23.2, python 3.12 under uv, seed 20261004:

  source                                                                 cases      model   c twin
  short   gen_case: class-aware strings of 0..~120 atoms             3,000,000          0        0
  long    gen_long: 60..1000 chars (words, cjk, letters, punct,        300,000          0        0
          ws, digits, base64, code)
  exh     every string of 0..4 atoms over REP (20 behaviours)          168,421          0        0
  small   every string of 0..6 atoms over REP_SMALL (12)             3,257,437          0        0
  vocab   1..30 DeepSeek-V3 vocabulary tokens                          300,000          0        0
  real    FLORES-200 dev + devtest (204 languages), python 3.12's    1,138,612          0        0
          stdlib, clang 21's headers (311.5 MB): every line,
          every paragraph, random chunks of 1..16384 chars
  bad     ill-formed byte strings (the python model's pieces)          300,000          -        0

  model   m2ultra1 (macOS arm64, 24 cores, load 3.77 at start, 12 jobs): 8,164,470 cases, 802,081,449 bytes,
          144,658,437 pieces, 0 mismatches; the model's three Splits == the file's pre-tokenizer on 1,164,138.
  c twin  m2ultra1 (same session, load 1.96 at start, 12 shards): 8,464,470 cases, 0 mismatches, each also re-tiled
          with cap 1 and with pseudo-random caps 1..7 (resume); classes.c's TOKS_CLASSES_DSV3 tables == the
          onig probes on all 1,112,064 scalars. (m2ultra1 became a timing host after these runs; later runs go to
          the GB10 boxes and tr9970x.)

Probe facts: \p{L} 141,028, \p{M} 2,501, \p{N} 1,911, \p{P} 855, \p{S} 8,514, \s 25, [\p{L}\p{M}] 143,529,
[\p{P}\p{S}] 9,369, [^\r\n\p{L}\p{P}\p{S}] 961,665, alternative 1's bracket 32, [A-Za-z] 52, the CJK bracket
21,094, X 957,230. Kinds over the scalars: P 9,365, L 122,442, N 1,911, WS 23, NL 2, X 957,227, P|CJK 4,
L|CJK 21,087, X|CJK 3; the CJK atoms that are not L: U+3040 X, U+3097 X, U+3098 X, U+309B P, U+309C P,
U+30A0 P, U+30FB P. The dsv3 class table dedups to 151 blocks (38,656 B stage 2 + 8,704 B stage 1).

tests/c/test_k3_dsv3.c (make test, 1,444,505 checks): 219 strings with hf's pieces (every rule and note
above) and 30 ill-formed ones with the python model's, the class table against §2 on every code point (with the
kind counts above), every string of 0..3 atoms over 29 representatives (two ill-formed) and of 4 atoms over 13,
ill-formed bytes at every position, long runs of every kind, caps 1..n with resume, guard pages (lengths 0..255,
end flush, start at every alignment 0..63), tmpl_params ignored, callee-saved registers (toks_abicheck_call).
tests/k3/dsv3_mutants.sh: 18 one-line mutants of k3_dsv3_c.c and the dsv3 class build (4 digits per chunk,
ascii letters past a non-ascii byte, a CJK run crossing the region, a tail out of a CJK region, any WS as D3's
space, a region end without the CJK or the N test, no D7 give-back, a WS / X prefix across a region, an
ill-formed byte as X, the D4 end, no D5 give-back, D1 from a non-ascii P, D7's give-back across a region, the
range end U+9FA6, P without S, marks not letters) each fail it.


6. contract changes (what dsv3 added)
-------------------------------------

  layout.h       TOKS_TMPL_DSV3 3u; TOKS_C_X 5u (base value, dsv3 tables only); TOKS_C_CJK 0x80u (dsv3 tables
                 only; o200k's kimi tables use bit 7 for TOKS_C_HAN -- never both in one table).
  classes.{c,h}  TOKS_CLASSES_DSV3 0x8 (0x2 is TOKS_CLASSES_HAN), in TOKS_CLASSES_KNOWN: the byte is §2's kind, no
                 other bit; the two CJK ranges are data in classes.c (as kimi's Han ranges). Every other build is
                 unchanged (checked on every code point).
  config.{c,h}   TOKS_DSV3_SPLITS[3] and TOKS_PATTERN_DSV3 (tmpl TOKS_TMPL_DSV3, class_flags TOKS_CLASSES_DSV3).
  compile.{c,h}  toks_tmpl_invalid knows TOKS_TMPL_DSV3 (params 0; bit 7 is CJK there, U+4E00 must carry it and
                 ascii controls are X); toks_compile_cls_flags (the cl100k / o200k K3 tiers' ascii precondition)
                 is not applied to dsv3 tables (their ascii controls are X; only the c twin reads them).
  unicode.py     two more onig properties in src/gen/ucd_flags.*: TOKS_UCD_P 0x0200 (\p{P}), TOKS_UCD_S 0x0400
                 (\p{S}); == python 3.14's unicodedata 16.0 on every code point; 155 blocks (was 141).
  kernels.h      K3_DSV3 like K3_O200K: TOKS_HAVE_K3_DSV3_<TIER> (a k3_dsv3_<tier>.S would set it), the call site
                 toks_k3_dsv3 = TOKS_RUN(K3_DSV3, tier) (the c twin on every tier until an asm tier lands); api.c's
                 run_text picks it for TOKS_TMPL_DSV3. toks_k3 (cl100k) never reads t->tmpl (test_dispatch).


7. load path and end to end
---------------------------

config.c maps the file's pre-tokenizer onto the template: a four-step Sequence whose Splits are TOKS_DSV3_SPLITS
exactly (Isolated, invert either way: Isolated keeps every split whatever its flag) and whose last step is
ByteLevel(use_regex false) gives TOKS_PATTERN_DSV3; any other four-step chain is refused by name (Removed is not
Isolated here: these Splits leave gaps). The empty normalizer Sequence reads as no normalizer (a non-empty one is
still refused).
DeepSeek-V3's model vocab holds three strings outside the byte-level alphabet (ids 0..2, <｜begin▁of▁sentence｜>,
<｜end▁of▁sentence｜>, <｜▁pad▁｜>, added specials as well; census feature compile:non-alphabet-vocab): they are
kept as decode-only ids (docs/breadth.md §1.4's mechanism: cfg->n_vocab_raw), no merge touches
them and no bpe symbol can equal them, so only an added-token match emits them; their decode is their own utf-8
(hf's ByteLevel decoder falls back so), and they stay out of byte2id, vhash and the words table. compile.c builds
the TOKS_CLASSES_DSV3 tables; toks_k3_dsv3 runs the tier's part with the dsv3 c twin (§4).

The six DeepSeek files of SPEC 1.1 (d) (tests/data/targets/ledger.txt; pins in tools/corpora/fetch_tokenizers.py)
all carry this chain and V3's model (normalizer, pre_tokenizer, decoder, post_processor and model json equal V3's;
checked 2026-10-04): dsv3 (V3, V3-0324), dsr1 (R1, R1-0528), dsv31 (V3.1, V3.1-Terminus, V3.2-Exp), dsv32 (V3.2),
dsv4 (V4-Flash, V4-Pro, V4-Flash-0731), dsv41flash (V4.1-Flash). They differ in added tokens only: 818 for V3 ..
V3.2 (14 / 16 / 18 / 20 normalized non-specials), 1,283 for V4 / V4.1 (46 / 47 normalized, all non-special, such
as "<dsml:" and "</dsml:", matched in phase 1), none with lstrip / rstrip / single_word.

Receipts:
  - make -j8 test, green on a developer laptop (macOS arm64), gb10d (linux arm64, load 3.97) and tr9970x (linux
    x86-64, taskset 0-7, load 3.85), each with the pinned files: test_compile loads the real DeepSeek-V3 file
    (n_ids 128,815, 818 added in both phases, 804 special, id 0 decodes to its utf-8) plus two hf-checked
    fixtures (nonalpha, dsv3style), four refusals of near-miss chains and of a non-empty normalizer Sequence;
    test_e2e encodes 30 texts (five of them
    deepseek chat / tool / region texts) in modes ALL, NONSPECIAL, NONE, with and without post-processing, and
    decodes them, against hf for gpt2, llama3, glm53 and dsv3: 0 failures.
  - parity: the 0.2.0 release's light sample for dsv4 (DeepSeek V4: this chain and V3's model, its own added tokens;
    tools/release/rc_host.sh, tests/parity/gen_cases.py --quick against hf 0.23.2) at 5f71528: 221,783 id cases and
    97,584 piece cases, 0 diffs on gb10a (neon, scalar), tr9970x (avx2, scalar) and m2ultra1 (neon); the receipts are
    in docs/release/0.2.md.
