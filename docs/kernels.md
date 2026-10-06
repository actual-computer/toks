kernels: the semantic contract every tier implements
=====================================================

src/core/layout.h is the binary half of this contract (tables, argument structs, offsets, hash and entry
formats). This file is the prose half: what each kernel computes, what it may touch, and how the driver uses
it. The c twin (tier c, src/core/k*_c.c) is the executable reading of this file; when this file and the c twin
disagree, that is a bug in one of them, found by the parity suites against hf.

  K1  added-token find            toks_k1_added_find_<tier>
  K3  scan (pre-tokenizer)        toks_k3_scan_<tmpl>_<tier>        tmpl in {cl100k (§3), o200k (docs/templates/o200k.md),
                                                                             dsv3 (docs/templates/dsv3.md)}
  K5  encode pieces               toks_k5_encode_<tier>
  K6  bpe of one piece            toks_k6_bpe_<tier>
  tiers: c (portable, every target), neon (arm64), avx2 and avx512 (x86-64)

every kernel:  uint64_t k(const toks_tables *t, <k>_args *a)  -- two pointer arguments on every abi.


1. rules for every kernel (SPEC §7.2, §10, §14.3)
-----------------------------------------------------

  reads    t's tables; a's input fields; text only inside [text, text + len) -- never a byte beyond, not even
           inside the same page or cache line. simd over full blocks inside the range; the kernel owns its
           tail: a padded copy of the last partial block in its own frame (<= 256 B), masked loads (avx-512
           fault suppression), or scalar code. callers never pad (SPEC §10.3's guarantee; design.md d8 on
           the mechanism). tested with the buffer's end flush against a guard page and again with its start
           flush against one, every length 0..255 x alignment 0..63 (SPEC §14.3).
  tables   a table only inside [p, p + n + pad), assuming no more than its alignment: the extent layout.h declares
           next to toks_tables (TOKS_X_<TABLE>: pad bytes past the end, the alignment); a kernel that needs more
           declares it there first. tested by make test-guard (docs/testing.md): every table and every scratch region
           on its own pages, a no-access page flush against its end (plus pad) and again against its start.
  writes   only the output ranges named in the argument struct (out[0, room), ends[0, cap), work[0, work_bytes),
           the cache buckets) and a's output fields.
  stack    a true leaf uses none; a kernel needing callee-saved registers or a spill area uses the PROLOGUE /
           EPILOGUE macros of src/asm/<isa>/asm.h: frame <= 256 bytes, unwind info from one description.
  state    nothing is carried between calls except through a's fields; a kernel is a pure function of
           (t, inputs, cache contents), and cache contents never change outputs.
  tiers    a tier uses only the instructions its feature set guarantees (layout.h, SPEC §11):
           neon = armv8.0-a + neon + crc32; avx2 = avx2 + bmi1 + bmi2 + lzcnt + popcnt + sse4.2 crc32;
           avx512 = avx-512 f/bw/vl/vbmi + bmi2 (+ the avx2 set). the driver never calls a tier the machine lacks.
  agree    every tier produces byte-identical outputs and output fields to the c twin on every input (T2, T5),
           including invalid utf-8.


2. text model: atoms and classes
---------------------------------

A segment text[0, len) is read as a sequence of atoms:
  - at byte i, if text[i..] begins with a well-formed utf-8 sequence (unicode §3.9 table 3-7: no overlongs, no
    surrogates, nothing above U+10FFFF) of k bytes, the atom is that code point, k bytes long;
  - otherwise the atom is the single byte text[i] (an invalid atom), and decoding resumes at i + 1.
    (SPEC §3.3: every byte of an ill-formed sequence is its own atom.)

class(atom):
  - ascii byte b:          cls_ascii[b]
  - code point cp >= 0x80: cls_stage2[cls_stage1[cp >> 8] * 256 + (cp & 0xFF)]
  - invalid atom:          TOKS_C_P (no flag bits)
base(atom) = class & TOKS_C_BASE_MASK, one of P, L, N, WS, NL (layout.h). The class tables are built per
tokenizer by src/core/classes.c from the generated oniguruma data (docs/unicode.md) with the pattern's
class_flags: a template that folds \p{M} into letters gets marks with base L; bit 7 (TOKS_C_HAN) is set only for
o200k's kimi variant (TOKS_TP_HAN, from src/gen/han_ranges.c) and is 0 in every other table.

What tiers may assume about the class tables (the compiler checks them: toks_compile_cls_flags, compile.c):
  - every template: the ascii class bytes are fixed. cls_ascii[b] is L | UPPER for A-Z, L | LOWER for a-z, N for
    0-9, WS for \t \v \f and U+0020, NL for \n \r, and P for every other ascii byte, with no other bit (the
    ascii members of \p{L}, \p{N}, \s and o200k's case groups are fixed by Unicode, and marks, folds and Han
    never touch ascii, so classes.c builds exactly this for every template but bloom's, TOKS_CLASSES_BLOOM). Tiers
    hard-code these classes: the compiler marks a table that breaks them with TOKS_TF_TWIN, and run_text then
    runs that tokenizer's K3 on the c twin on every tier.
  - TOKS_TF_CJK_L in toks_tables.flags: every code point in U+4E00..U+A3FF and U+AC00..U+D6FF (CJK ideographs,
    Yi, hangul) has base L and no FOLD_S. TOKS_TF_CJK_B: each has the class byte L | UPPER | LOWER exactly (a
    letter in both of o200k's case groups; kimi's Han bit clears it). The compiler sets each when it holds
    (both do for every table classes.c builds without Han); a tier may then class those atoms without a lookup,
    and must look them up without it.


3. K3 scan, template cl100k
----------------------------

The template covers one family of Split patterns (behavior Isolated, leftmost-first alternation, the oniguruma
semantics hf runs). Its parameters (toks_tables.tmpl_params, layout.h TOKS_TP_*) select the variant:

  gpt-2 (ByteLevel use_regex=true)  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
                                    = CONTR_CS, LPREFIX ' '?, DIGITS_SP_RUN, no PUNCT_NL, no WS_NL
  cl100k, llama 3                   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
                                    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
                                    = CONTR_CI, LPREFIX_ANY, DIGITS_1_3, PUNCT_NL, WS_NL
  qwen 2 / 2.5 / 3                  as cl100k with \p{N}           = DIGITS_1
  qwen 3.5 / 3.6 / 3.8              as qwen 2 with [\p{L}\p{M}]+ for the letter run and [^\s\p{L}\p{M}\p{N}] for
                                    punctuation: class tables with marks folded into L (an M prefix is
                                    equivalent: an M char either starts the L|M run or prefixes it, same span)
  LLaDA / Ling (P22)                qwen 2's set spelled '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}|
                                     ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+: the same pieces (below)
  zeta (P25)                        qwen 2 without PUNCT_NL (its classes hold literal CR LF, \p{N}{1} = \p{N})
  Laguna (P20)                      [Split (?:\r?\n)+(?!\r?\n) MergedWithNext, Split qwen 2]: qwen 2 + NL_CUT (A9)
  bloom (P21)                        ?[^(\s|[.,!?…。，、।۔،])]+ : GB_SP (A10) on TOKS_CLASSES_BLOOM tables, where every
                                    char but \s and those 14 literals is P (so A4 makes ' '? + a P run) and the 14
                                    are WS: no contraction, no L or N atom, A5 off

Template detection compares the file's regex string byte for byte, so the table in config.c holds the hub's exact
strings, checked against the pinned files (tests/c/test_targets.c loads them). The two that are easy to get wrong:

  qwen 2 / 2.5 / 3     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  qwen 3.5 / 3.6 / 3.8 (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
                       (the letter prefix class does NOT exclude \p{M}; only the punctuation class does)

Every piece starts where the previous one ended (every atom is matched by some alternative, so pieces tile
the segment). At piece start i, with c the atom at i and c1, c2 the next atoms (absent past len), the piece is
given by the first alternative that applies:

  A1 contractions (CONTR != NONE). c is U+0027 (') and
       CS: c1 in {s t m d} -> [i, end c1);  c1 c2 in {re ve ll} -> [i, end c2)
       CI: c1 in {s S t T m M d D} or c1 has FOLD_S -> [i, end c1);
           c1 c2 in {r R}{e E}, {v V}{e E}, {l L}{l L} -> [i, end c2)
  A2 letters.  base(c) = L -> [i, end of the maximal L run from i).
       otherwise, if c1 exists, base(c1) = L, and the prefix rule holds for c:
         LPREFIX_ANY: base(c) in {P, WS}       (i.e. [^\r\n\p{L}\p{N}])
         else:        c is U+0020
       -> [i, end of the maximal L run from c1).
  A3 digits.
       DIGITS_1_3:     base(c) = N -> [i, end of up to 3 consecutive N atoms)
       DIGITS_1:       base(c) = N -> [i, end c)
       DIGITS_RUN:     base(c) = N -> [i, end of the maximal N run)
       DIGITS_SP_RUN:  base(c) = N -> [i, end of the maximal N run);
                       c is U+0020 and base(c1) = N -> [i, end of the maximal N run from c1)
  A4 punctuation. j = i if base(c) = P; j = position of c1 if c is U+0020 and base(c1) = P; otherwise A4
       does not apply. -> [i, end of the maximal P run from j), then, with PUNCT_NL, extended over the maximal
       run of NL atoms that follows.
  A5 newline runs (WS_NL). base(c) in {WS, NL}: let R = [i, k) be the maximal run of atoms with base in
       {WS, NL}. If R contains an NL atom -> [i, end of the last NL atom in R).
  A6 whitespace not before a non-space. base(c) in {WS, NL}, R = [i, k) as above:
       k = len -> [i, k);  R has >= 2 atoms -> [i, start of the last atom of R).
  A7 whitespace. base(c) in {WS, NL} -> [i, k) (R is one atom here).
  A8 digit cut (DIGIT_CUT, with DIGITS_1 or DIGITS_1_3). hf's Digits(individual_digits=true) runs before the
       regex (the digits-gpt2 template: SmolLM, PowerMoE, starcoder = [Digits, ByteLevel(use_regex)]): every N
       atom is its own piece and the regex runs on each stretch between them with the stretch's end as its end of
       input. With DIGITS_1_3 the cut is MiniCPM5's Split \p{N}{1,3} (P16: [Split \p{N}{1,3}, Split cl100k with
       \p{N}+, ByteLevel]): each run of up to 3 N atoms is a piece (the N13 groups) and N is onig's \p{N}. Only
       A6 reads past a stretch's end (its lookahead and k = len); every other rule's run stops at an N atom. So:
       in A6 a run whose next atom is N is taken whole, as at k = len (A5 still ends a run holding an NL at its
       last NL: the WS atoms after it are the next piece, whole). For digits-gpt2 N is hf Digits' class (rust
       char::is_numeric): \p{N} plus 13 newer code points (U+11DE0..11DE9, U+16FF4..16FF6; src/gen/ucd_flags.h
       TOKS_UCD_RNUM), the class tables are built with TOKS_CLASSES_DIGITS; the regex never sees such a char, so
       the base class serves both. The asm parts take A8 as data: with DIGIT_CUT a W run that an N atom
       follows has no A6 give-back (dsv3's D5 term, docs/templates/dsv3.md §4), and DIGITS_1 cuts at each N atom.
  A9 newline-run cut (NL_CUT; Laguna). hf's Split (?:\r?\n)+(?!\r?\n) MergedWithNext runs first: its matches are
       the maximal runs of \r?\n units and each joins the text after it, so the regex runs on the stretches that
       start at a run's start, each with its own end of input. A run starts at q when a unit starts there and the
       byte before is no LF: x[q-1] != LF, and x[q] is LF with x[q-1] != CR, or x[q] x[q+1] = CR LF (kernels.h
       toks_k3_nlcut). Only A4's tail and the whitespace run can reach such a q (it is an NL atom): A4's [\r\n]*
       stops before it (so it holds lone CRs only), and A5-A7's run R ends there as at len (no give-back). The
       asm parts take A9 as data: a cut mask C from the LF and NL masks (LF & ~(NL << 1) | CR & (LF >> 1) &
       ~(LF << 1)) starts pieces, A4's absorbed CR/LF runs are taken over NL & ~C, and an S run that ends at a
       cut keeps its A5-tail start (C's bits are NL bytes: no give-back before one).
  A10 space give-back (GB_SP; bloom). A6 gives back the last atom only when it is U+0020: R's other last atoms
       stay in R ([i, k)). With bloom's tables that is the pattern's gap rule: a run of the 14 literals and \s is
       one piece, less a final ' ' that a P atom follows (that ' ' and the P run are the next match).

These rules are the regex's leftmost-first semantics written out (A5's backtracking ends at the last CR/LF of
the run; A6's lookahead gives back exactly one atom; the run is maximal because the next atom is \S or the
end). LLaDA's spelling gives the same pieces: a possessive ?+ or ++ on a class only refuses to give back atoms
that the rest of its alternative never needs (the letter prefix is no letter; [\r\n]* follows the P run), \s*[\r\n]
backtracks to R's last CR/LF exactly as \s*[\r\n]+ does, and '(?i:[sdmt]|ll|ve|re) reads the same contractions
(no two of its branches share a first letter; (?i) folds U+017F onto s in both). A9 / A10 were checked by
enumeration against hf's own pre-tokenizers of the pinned files (tests/generic/diff.py --enum: every string of up
to 6 chars over each chain's class representatives) besides the breadth fixtures tmpl_p20..p25. Notes that bite:
  - only U+0020 is the optional space of ' ?'; every other \s char is WS but not ' '.
  - [^\r\n\p{L}\p{N}] includes every WS char (tab, U+3000, ...) and every P char including invalid atoms.
  - (?i) folds U+017F to s, so CI matches "'ſ"; docs/unicode.md records the probe that shows no other fold
    reaches the contraction letters.
  - lookahead is to the end of the segment (text[len] is the end of input for the regex), never further.

Output: piece end offsets ends[0..n), strictly increasing, each in (pos, len]; a.pos = ends[n - 1]. The call
stops when the segment is exhausted or n = cap. pos must be a piece start (0, or an end this kernel wrote).
ends[n, cap) holds tier-dependent garbage after a call (a tier may store slack there; the avx2 tier does), so
ends[0, cap) must be memory whose tail the caller does not need (the driver passes its scratch).

3.1 K3 on the asm tiers
-----------------------

An asm K3 is a PART: src/asm/x86_64/k3_avx2.inc (avx2) and src/asm/arm64/k3_neon.inc (neon, the same blocks,
algebra, labels and frame slots transcribed), each assembled once per template (k3_<tmpl>_<tier>.S sets K3_O200K,
and K3_DSV3 for dsv3; the o200k parts are docs/templates/o200k.md §4, dsv3's docs/templates/dsv3.md §4). The
driver runs it through kernels.h with the template's c twin, and the pair is the tier's kernel, byte-identical to
the twin:

  part     scans 64-byte blocks from a piece start pos and writes the ends it can decide. Where the blocks stop
           it returns at a piece start (its a->pos, or the pos it was given) with a->rsv = the end of a stretch
           it leaves to the twin, 0 when it stopped at len or cap. Its stretches: a remainder under 4 bytes
           (rsv = len), a piece longer than its block (rsv = pos + 1), and the pieces from the block's last start
           through the piece that holds an atom of a class the blocks do not take (a non-ascii NL, FOLD_S,
           o200k's flag combinations no real table has; rsv = that atom's start + 1). N atoms of any length join
           N: under \p{N}{1,3} a block holding one counts each run's atoms by their first bytes.
  short    toks_k3_short, at every K3 call site and for every tier, the scalar one included, before any asm
           entry: a remainder of one atom is one piece (no call); one under MIN_N bytes is the twin's whole call
           unless it has MIN_A bytes or more and fewer than 4 non-ascii bytes in its first 8 (two loads inside
           it). A short segment so runs the same code on every tier, and a part only gets the short remainders
           it scans faster than the twin. MIN_A / MIN_N are the crossovers measured with tests/k3/bench_k3.c segs
           (the part's block setup against the twin's per-atom cost): x86 4 / 16 (dsv3 7 / 16), arm64 7 / 24
           (dsv3 16 / 32). A dense non-ascii remainder (CJK) costs the block path ~10 ns (Zen 5) / ~14 ns (X925)
           more than an ascii one, hence the density test.
  tier     toks_k3_parts after the short rule (always inline at the call site): the part, then the twin one
           piece a call (cap 1) while pos < rsv, then the short rule on the rest, then the part again. The twin
           restarts at a piece start, so the pieces are the twin's (§3's resume); a's fields other than n and
           pos stay as they were. tests/k3 run a part through toks_k3_tier (= short + parts); calling a part
           directly is for tests (abi, faults) only.

The block (bit j of every mask = byte pos + j; X << 1 = "the byte before is X"; bit 0 is pos itself): two ymm
loads (neon: ld4 lanes through a 128-entry tbl / tbx code table and the sri / shrn movemask) give 64-bit masks of
the ascii classes (L, N, W = S | NL, NL, SP = ' ', AP = ') and of the non-ascii bytes. Each non-ascii atom's
class goes to all of its bytes (strict utf-8: an ill-formed byte is a one-byte P atom); cont marks the bytes after
an atom's first; the first atom of a class the blocks do not take cuts the block before it; e = the bytes in
play. Blocks dense in non-ascii bytes take a faster atom pass when a simd shape check shows only 2/3-byte atoms,
and 3-byte atoms in the CJK ideograph / Yi / hangul blocks need no lookup under TOKS_TF_CJK_L (o200k: _B). B, the
piece starts, is one term per class (cl100k):
  L   L & ~(L << 1) & ~pre          an L run start not taken as a prefix: pre = ((S | P) << 1) & ~E under
                                    LPREFIX_ANY (E: after a P run of 2+ atoms or after ' '), (SP << 1) otherwise
  N   N & ~cont | Nst | Nst & ~(SP << 1) | Nst + every 3rd N of a run (DIGITS_1 / RUN / SP_RUN / 1_3)
  P   Pst = P & ~((P | SP) << 1)    a P run start not after ' '
  W   W* run starts                 W* = W minus the CR/LF runs a P run absorbs (PUNCT_NL)
      S & (NL << 1) whose S run ends at a non-NL      the tail after A5's last CR/LF (WS_NL)
      the first byte of the last atom of a run that a non-W byte follows (A6's give-back)
then A1: each ' in Pst whose next bytes spell a contraction moves the start after it. The bits up to u are decided
(u = e - 1, or the start of the W* run holding byte e - 1, whose ends lie past the block); q = the last start
<= u; the ends B in (0, q] are written and the next block starts at q (no start in (0, u]: the piece is longer
than the block, the twin's). A block that ends the segment writes B in (0, e) and len. The masks come from loads
inside [text, text + len) only: a window at pos, the last 64 bytes of the segment (masks shifted down), or, when
len < 64, the segment zero-padded in registers by byte shuffles of overlapping loads (no store and reload).
Emission is branch-free: each mask byte stores its 8 table indices + base as one vector and advances by its
popcount (the slack lies inside ends[0, cap)). The ascii classes are hard-coded (§2's precondition).

The page pass (neon, PAGES in k3_neon.inc): in a dense block without CJK leads whose first lead is a 2-byte or
E0 / E1 lead, with 10+ leads left for the per-atom loops, the atoms of one 64-code-point page (fixed by a 2-byte
lead, or a 3-byte lead and its first continuation: the page is a contiguous 64-byte run of cls_stage2) take their
classes with one 4-register tbl over the block's last bytes, and their slot masks from two 16-entry nibble tables
(k3n2_plut: class lo / hi -> one bit per slot, the template's slot function as two product sets; 0 cuts). Up to 4
pages of 4+ atoms a block; the rest take the loops. It pays where a script's letters share few pages (Cyrillic,
Arabic, Devanagari, Greek, Hebrew); kana, symbols and hangul spread over many and stay in the loops.

Measured lessons (tests/k3/fuzz_k3.c and tests/k3/bench_k3.c take any tier by symbol, -DK3_O200K for o200k):
  - the parts once carried their own scalar pieces and a run mode (a long piece's run followed through windows
    classified as blocks): ~700 lines per isa. Handing those stretches to the twin cost 0-4% on whole files
    (pieces >= 64 B are 0-0.6% of real text's bytes, 6.6% of wiki ja's), and the twin's speed on long-piece-only
    input (scanner run 2). Two cells stay under what the old kernels did, by choice
    (scanner run 3; tr9970x cpu 18 / gb10c X925 cpu 7, abba x21):
      * the hf case set's "piece >= 64" cell (99.5% of its bytes in such pieces): tier 0.97x (x86) / 0.96x
        (neon) the twin, the old kernels' run mode 2.0x / 1.6x. Both sides make those pieces with the twin; the
        tier pays one part entry per long piece to learn that it is long (not a routing cost: no rule before the
        entry can know). A run mode in asm would win it back for ~300 lines per isa.
      * dense non-ascii (CJK) remainders under MIN_N: the twin's on every tier, 0.82-0.88x (cl100k) /
        0.73-0.78x (o200k) of the old kernels' scalar pieces on x86, 0.87-0.93x / 0.85-0.88x on neon (zh
        6-16 B). The block path costs more than the twin there, so only scalar pieces in asm (the old ~700
        lines per isa minus the run mode) would recover it; x86's T10 room goes to the page pass first.
  - x86 code alignment moves whole-file cells by 1-5% (Zen 5): the block loop's head and the labels the
    non-ascii and template paths join at are aligned (.p2align); re-measure after moving code in the part.
  - the page pass (scanner run 2; gb10c X925, vs the part before it): wiki hi / ru / ar +8-16% (cl100k),
    +8-22% (o200k), +10-24% (dsv3), qwen35 hi +15%; ja / ko / zh / ascii within 1%. Gating matters: a pass that
    also took kana / punctuation pages (or skipped small pages to find big ones) lost 2-14% on ja / ko; the neon
    per-atom loops' heads are aligned (.p2align 5: an unaligned head cost cl100k ja / ko 1.5%). x86 has no page
    pass, by decision (scanner run 3; one kernel for every abi, no SysV-only path): a vpshufb page lookup needs
    ymm6-9 beyond the parts' ymm0-5 (their low halves are callee-saved on win64), or SPILL's six masks (xmm2-4)
    moved to the frame, and either needs the o200k / dsv3 frames to shrink first (win64: LOCALS + 16 nxmm <= 184):
    o200k 144 -> 112, dsv3 176 -> 112 (136 with SPILL in memory). The cheap cuts (ends0 from a, the two stage
    pointers from t, until packed beside the params) free 24 B; the rest means moving the o200k mark rounds' P and
    tails and dsv3's D1 / X masks into vector registers across the shared paths. For ~200 lines plus that
    surgery the gain is small: x86's per-atom loop takes a 2-byte atom in ~11 ops + 2 loads (tr9970x: wiki ru / hi
    1.16 / 1.32 GB/s cl100k, 0.98 / 1.10 o200k), neon's page pass gave hi / ru / ar +8-24% K3 alone, and K3 is
    ~6-20% of cold encode there: ~1-4% end to end. Revisit when a change frees those slots.
  - neon replaced the first neon K3 (per-call code table, a lead loop per non-ascii atom, a 430-line scalar
    fallback) with the parts its review proposed: ld4 + tbl / tbx planes, sri / shrn movemask, byte-LUT emission.
  - avx512 runs the avx2 part (kernels.h falls back per kernel). An avx512 K3 (not merged: vpcompressd emission,
    masked tail loads, pext atom masks; 1,267 lines) lost to the avx2 kernel on tr9970x (Zen 5; same binary and core,
    30 paired rounds, 95% bootstrap intervals, avx512 / avx2): enwik8 0.901, real64 0.845 (gpt2 / qwen2 / qwen35
    0.872 / 0.879 / 0.878), en / code / py_stdlib 0.905 / 0.909 / 0.906, War and Peace 0.932, zh.txt 0.571,
    classical Chinese 0.575 / 0.573, ml 0.651, eu 0.828, wiki ja / ko / zh 0.813 / 0.484 / 0.727, wiki ru / ar / en
    0.979 / 0.883 / 0.959, gpt2 tokenizer.json 0.832; it won only wiki hi (1.119) and segments under 64 B
    (1.29-1.32, a gap the avx2 kernel's short-segment paths have since closed). Rejected under SPEC §10.5 and
    T10. Its decode-ahead grid (3 zmm blocks + vpermi2b windows) had lost 20-22% on ascii for +10% on CJK.
  - the restart at the last piece start (block k + 1's loads wait for block k's whole algebra) is not what the
    avx2 part costs (measured 2026-10-04 on the 9970X, cpu 20, bench_k3 31 rounds; receipts docs/bench/raw/k3-tr9970x-d11f9d1-*):
    the part runs gpt2 en at 4.35 GB/s, ~74 cycles a 64-byte block for ~250 instructions on the ascii path. A
    probe that took each next block's start from a list recorded on an earlier pass of the same segment (same
    instructions and ends; it trapped if the list ever differed from the algebra's answer) ran 0.999 / 1.000 (gpt2
    en), 0.989 / 1.000 (code), 0.95-0.96 (cl100k: its own state loads) of the part. A fixed-stride, pipelined
    redesign therefore buys nothing by itself; the block is bound by its own work. gigatoken's whole pre-tokenizer
    on Zen 5 (avx-512 classification + boundary algebra + flatten; its profiling/zen5_st_profile.md cold shares x
    690 MB/s, gpt2 on OWT) comes to ~3.4 GB/s, under the part.


4. K1 added-token find
-----------------------

Tokens of phase p are the entries with phase == p (0: matched on the raw text, hf's non-normalized trie;
1: matched on normalized text, hf's normalized trie). A match is the leftmost-longest match of the phase's
token set in text[from, len): the smallest start >= from at which some token's bytes occur (entirely inside
[0, len)), and among the tokens occurring there the longest; end = start + its length; entry its index. K1
writes a BATCH of such matches to a->m: the first from pos, each next one from the previous one's end, until
cap matches (then next = the last end) or none is left (then next = len); it returns their number n (a->n).
This is daachorse's LeftmostLongest find_iter resumed at pos, which hf uses (tokenizers 0.23.2
added_vocabulary.rs: refresh_added_tokens, find_matches). The raw sequence never depends on the driver's
policy below (a dropped match, an rstrip: hf resumes at the raw end either way), so the driver takes the
batch one match at a time and calls K1 again from next when it runs out (segment.c, TOKS_SEG_BATCH = 16):
one call per 16 matches instead of one per match, which is what token-dense chat pays for (scanner run 4:
K1 was 12% of qwen38 chat pass time at ~20 ns a call, mostly the call's fixed cost). Inside a call the scan
goes on in the block of the match: its candidates below the match's end are dropped (a match ending past the
block resumes at its end). Offsets are u32 (len <= TOKS_MAX_TEXT, 2^29); cap >= 1.

Index (per phase; compile.c toks_added_index builds it): the shufti tables classify the first byte (candidate
when lo[b & 15] & hi[b >> 4] != 0; false positives allowed, false negatives not); h4[slot << 1 | phase] (stored
right after index2: add_index + 2 * 65536) lists
the tokens of >= 4 bytes whose first four bytes hash to slot (core.h toks_k1_h4: k * 0x9E3779B1 >> (32 -
TOKS_K1_H4_BITS), k their little-endian u32; 1024 slots per phase, 16 KiB), longest first; add_index[b0 | b1 << 8] lists
every token of >= 2 bytes whose first two bytes are b0 b1, longest first (split.c reads whole buckets; K1 reads
only their tails, the 2- and 3-byte tokens, from the end); add_single[b] is a one-byte token (entry + 1). At a
candidate position q with >= 4 bytes left, the h4 bucket's first exact match is the longest token at q (every
token of >= 4 bytes that occurs at q shares q's first four bytes, so it is in that bucket); without one, the
longest 2- or 3-byte token (index2's bucket from its end, the last match), else the one-byte token. Verification
is an exact byte compare. Why h4 (measured, scanner run 4): with index2 alone a token was found after every
longer token sharing its first two bytes -- qwen38's <|im_end|> after 22 longer "<|" tokens, llama3's <|eot_id|>
after ~250 <|reserved_special_token_N|> -- so K1 cost ~20 ns (qwen38) and ~184 ns (llama3) per match on token-dense
chat, about 70% of llama3's chat time.

Long buckets (scanner run 6): tokens sharing four bytes keep an h4 bucket long, and walking it costs every member
at every candidate: llama 3's "<|re" holds 248 tokens, nemotron's "<SPE" 970-990, llama 4's "<|vi" 1,048, gemma 3 /
embeddinggemma / sarvam "<unu" 6,217-6,242 (gemma 3n 6,494), and DeepSeek V3..V4.1's "<｜" 812-862 -- the last one
holds every DeepSeek chat token, so its chat paid ~800 compares a special token (K1 alone 104 MB/s on tr9970x vs 4.5-5.5
GB/s for llama 3 / nemotron chat), and a text of shared prefixes ("<unused" over and over) paid thousands a
candidate. So an h4 bucket of more than TOKS_K1_LIST (8) tokens is a RADIX TREE instead of a list: its h4 entry has
bit 63 set and the root's byte offset from add_cand in its low half. A node is 16 bytes at add_cand + its offset
(layout.h RX_*: the first child's offset, the label's offset, the label's length, the number of children, entry + 1
of the token ending there or 0); a node's label is followed by its children's first bytes, and its children are
consecutive and sorted by first byte. compile.c builds the tree from the bucket's tokens sorted by bytes (a prefix
first): a node's label is its range's common bytes from the parent's depth (first vs last of the sorted range),
the token whose bytes end there is the range's first, and the rest split by their next byte; every node ends a
token or has two or more children, so n tokens make at most 2n - 1 nodes, and the labels and first bytes take at
most the tokens' bytes plus one byte a node (toks_added_cand_words sizes the region). The walk at q (avail = len -
q bytes): the node's label must lie inside the avail bytes and match (one compare: 16-byte chunks from the end
down, or overlapping 8 / 4 / 2-byte pairs, or one byte); a token ending there is the longest so far; the next text
byte picks the child among the first bytes (16 at a time); stop when none, when the text ends or at a leaf. The
last token found is the longest at q (the walk visits every prefix of text[q..] that the bucket's tokens share);
none falls through to index2 and single as before. A walk costs the nodes on its path, at most the length of the
longest token (a child's label is never empty), whatever the bucket's size.

The driver, not the kernel, applies hf's per-match policy (find_matches), in this order, per raw match:
  1. mode NONSPECIAL and the token is special: drop the match; its bytes stay plain text; the search resumes
     at the raw end (hf does not rescan inside a dropped match).
  2. single_word: drop unless the match is not preceded and not followed by a word character.
  3. lstrip: start moves left over the whitespace run before it, never before the previous split's end.
  4. rstrip: end moves right over the whitespace run after it.
  5. emit the gap [prev_end, start) as a text segment if non-empty, then the token; prev_end = end.
     The search resumes at the raw end (not the rstripped end), so the next match may start inside the
     stripped run: then no gap is emitted (hf behaviour, kept exactly).
Mode NONE does not call K1. Phase 1 runs on each normalized gap of phase 0.


5. K6 bpe of one piece
-----------------------

Tables (bpe_build.c, bpe.h): toks_bpe_build runs in toks_load after the compile step, with t->n_ids / tok_off /
tok_bytes filled. tok_bytes holds each token's RAW bytes (its byte-level alphabet string decoded through
toks_char_byte: the bytes it stands for in the text); vhash, the words table and every kernel probe are keyed by
them. Merge ranks are RAW ranks (the index in cfg's merge list, whose merges are id triples in that order);
t->n_merges is the raw count, the length of rank2id. Built (layout.h is the format):
  byte2id      the model token of each raw byte.
  merge table  toks_merge_slots (spm_build.c's too): one slot per distinct (left, right) pair holding its LAST raw
               rank (hf's MergeMap: the later insert overwrites). The prio is the pair's merged id when the
               surviving pairs' merged ids rise strictly in raw rank order (TOKS_TF_IDS_AS_RANK: ids compare as
               ranks, rank2id stays NULL; a tiktoken config, cfg->ids_as_rank, takes the ids whatever the order),
               else the raw rank with rank2id[rank] = merged id. Buckets: the first power of two >= n / 5 + 1 (load
               <= 0.625), at least 2 (merge_shift <= 63); merge_maxprobe = the longest landing distance + 1, so a
               present key is always reached and an absent one stops at an empty slot or there.
               Merged ids as priorities where they do not rise strictly, counted and not taken: hf's tiktoken
               conversions (llama 3, o200k, glm 5.3, llama 4) list every split of a token (280k-446k merges for
               128k-200k tokens), so every merge reads rank2id, 0.21-0.24 bytes of rank2id lines per input byte a
               pass (tools/bench/stages.c's counting build, llama 3 en 4 KiB: 6.8k distinct lines). Taking the
               merged id as the prio is hf's order on every input when (A) merged ids never decrease in rank order,
               (B) every merge's output id is above both parts' and (C) no token has two different merges (x, y),
               (y, z) sharing the middle token: then one level's pairs that overlap are one merge, and those that do
               not commute. Counted on the four (each tokenizer.json): (A) holds, but (B) fails for 49.6k-72.2k
               merges (the conversion lists splits whose part outranks the token) and (C) for 5.4k-5.9k tokens,
               e.g. (ĠĠĠ, Ġ) and (Ġ, ĠĠĠ) -> ĠĠĠĠ: on symbols [ĠĠĠ, Ġ, ĠĠĠ] hf takes the lower-ranked merge, merged
               ids the leftmost, and the two results differ. Whether such a configuration is reachable from real
               bytes is a per-token reachability question, not a table property: no certificate, rank2id stays.
  bytepair     the prio of each byte pair, K6's round one.
  pair filter  one u64 per two merge-table buckets; each merge key sets two bits in the word of its home bucket:
               bits 20-25 and 26-31 of key * FIB64, below any bucket index's (bpe.h bpe_pf_bits). A key without
               both bits set has no merge (no false negatives). Filled by toks_merge_slots with the slots, so the
               sentencepiece merge table has one too; 256 KiB for llama 3, 512 KiB for gemma 4, an L2-resident read
               where the bucket is not.
  vhash        the tokens a whole piece encodes to: under ignore_merges the model tokens with a raw form; without it
               the all-ascii tokens over TOKS_KEY_MAXLEN bytes whose own bytes bpe back to them (each certified at
               build by K6's c twin with no vhash and no premerge yet; qwen 3.8 201 of 248,044 tokens do not, gpt2
               one: id 50256, <|endoftext|>; dsv3 / dsv4 / qwen 3 / minimax m2 none; over 256 bytes: not certified).
               TOKS_TF_IGNORE_MERGES: the probe is on; TOKS_TF_PROBE_LONG: only for a piece over 15 bytes;
               TOKS_TF_PROBE_ASCII: only for one ending in an ascii byte. Without ignore_merges both are set when a
               token certified. Why so narrow (GB10 X925, e2e 4 KiB, qwen 3.8): every vocab token probed, code warm
               549 -> 715 MB/s but en -9%, ml -8%, cjk -15% (a missed probe hashes the piece and reads a random line
               of a 4 MiB vhash on every K6 call; K5's words already answer the short vocab pieces, and a long
               non-ascii piece is a letter run: zh 10 hits in 49,638 probes); narrowed, code 548 -> 720, every other
               cell within +-1%. Under ignore_merges the same probe was on for every K6 call, and 95% of the short
               ones missed (llama 3 en, K5's c twin: 3,731 hits in 80,289 probes of 2..15-byte pieces; a hit is a
               token K5's words left out). PROBE_LONG is set when K5's words hold every token of 2..15 bytes whose
               own bytes do not bpe back to it (llama 3 532 of 126,153, glm 5.3 792 of 152,150, o200k / llama 4
               none): the words fill seats those whose two buckets it finds full in the place of an entry that
               moves to its other bucket, or of one that bpes back to itself (bpe_build.c words_seat; which entry
               goes does not matter: the highest id of those that can, instead of the first found, measured flat,
               llama 3 / glm 5.3 within +-0.4% in every cell, docs/bench/raw/commits-caa6427-seat-gb10e-*.log; the
               fill runs the merges alone only on the tokens it
               left out, and the vhash fill on the long ones that end in a non-ascii byte: load llama 3 176 -> 180
               ms, o200k 305 -> 314, llama 4 311 -> 322, medians of 60). K6 then never probes a piece of 2..15 bytes
               that K5 hands it, and K6 alone answers such a piece with the merges (test_bpe's ref_k6: K6's contract
               under the flag). PROBE_ASCII is set when every
               token over 15 bytes that ends in a non-ascii byte bpes back to itself (o200k, llama 4; not llama 3 or
               glm 5.3, whose long Arabic / CJK tokens include some that do not). e2e_commits, e2e.md's states, GB10
               X925 one core, 4 KiB, cold / pass (after other text) MB/s, ids equal everywhere: llama 3 en 254 -> 268,
               262 -> 288; code 414 -> 420, 311 -> 330; ml 109 -> 114, 128 -> 135; cjk 146 -> 152, 144 -> 148; o200k
               en 239 -> 260, 222 -> 255; code 449 -> 453, 282 -> 320; cjk 134 -> 165, 130 -> 156; glm 5.3 en 252 ->
               277, 242 -> 272; llama 4 cjk 130 -> 163, 126 -> 155; qwen 3.8 and gpt2 (no change) within +-0.6%. The
               polluted pass gains most: the probe's vhash line is the one that comes from DRAM (llama 3 en 17.4k
               short K6 calls a pass: 0.54 bytes of vhash lines per input byte no longer read). The region keeps the size of every
               model token although few are in it: sized to the certified tokens (qwen 3.8 18 -> 15 MiB of tables,
               gpt2 4.8 -> 3.8) it moves the arena's other tables, and that alone cost gpt2 en cold 1.0% (GB10 X925:
               the same smaller tables with the probe off -1.0%, the probe on top 0.0%). Two layout fixes for the
               smaller tables, measured and not taken: the arena on whole huge pages (gpt2 +1.0%, llama 3 pass -1.5%,
               code -2.3%, qwen 3.8 cjk -1.4%), words allocated first (gpt2 -1.4..-2.5%, the rest -0.3..-1.2%).
  premerge     5.1: a model token whose raw bytes B are one 2/3-byte char pattern and merge-only BPE(B) = [it] (run
               on the tables just built), with the risk bits: R = the highest prio BPE(B) applies; an edge symbol is
               the symbol at B's left (right) end while B is not yet whole, its bound the highest prio pending inside
               B while it is the edge; the byte before B collects the last byte of x over every merge (x, P), P a
               left edge, prio <= P's bound, and over every (x, t), prio <= R; the byte after B the first byte of y
               over (S, y), S a right edge, prio <= S's bound, and over (t, y), prio <= R. NULL when no char
               qualifies.
  apm          5.1's certificate for two ascii bytes: a model token t of two ascii bytes with merge-only BPE(B) = [t],
               its risk as two 32-bit masks over a neighbour byte's class (layout.h TOKS_APM_*: letters case-folded,
               space, digits, other ascii, continuation bytes, the two lead ranges). NULL when no pair qualifies.
  words        the certified shortcuts (6. "Static table").
A vocab string outside the alphabet (cfg->n_vocab_raw of them; their tok_bytes are their own utf-8) is
decode-only: hf's BPE never forms it (merge_word starts from alphabet chars, a merge concatenates two symbols,
ignore_merges looks up the alphabet image of the piece), so byte2id, vhash, premerge and words leave it out -- its
bytes may equal an alphabet token's (granite-embedding's "   " beside "ĠĠĠ"), and taking it there would change ids
(docs/breadth.md §1.4). Memory: toks_bpe_tables_bytes is the sum of the allocations at their worst-case sizes, from
the sizing functions the builder uses, plus 64 bytes of alignment each: an arena of exactly that many bytes never
runs out, whatever its base alignment.

hf BPE::tokenize on the piece's bytes (byte-level alphabet implicit):
  - TOKS_TF_IGNORE_MERGES and the piece is in vhash (added-only tokens are not; without ignore_merges only the
    tokens certified to encode to themselves are, so the answer is the one the merges below would give): [that id].
    Under TOKS_TF_PROBE_LONG only a piece over 15 bytes is looked up, under TOKS_TF_PROBE_ASCII only one ending in
    an ascii byte: every piece K5 hands K6 still gets hf's answer (K5's words answer the 2..15-byte tokens that
    need the rule), and K6 alone answers a token whose own merges do not rebuild it with those merges.
  - otherwise symbols = byte2id[b] for each byte; repeatedly take the adjacent pair whose merge has the
    lowest prio (layout.h: the merged id under TOKS_TF_IDS_AS_RANK, else the rank), ties to the leftmost
    pair, and replace it by the merged id (prio itself, or rank2id[prio]); stop when no adjacent pair has a
    merge. (hf: a priority queue on (rank, pos) with stale entries skipped -- equivalent to this.) Round one
    may use bytepair[] instead of the merge table. Duplicate pairs in merges: the compiler keeps the one hf
    keeps (the last occurrence), so the table has one prio per pair; TOKS_TF_IDS_AS_RANK is set only when the
    merged ids of the deduplicated merges, in rank order, rise strictly.
Output ids in order; n_out returned; merges = len - n_out (0 for an ignore_merges hit). Work area:
TOKS_BPE_WORK_BYTES(len); the algorithm (linked array with a linear min scan for short pieces, a heap for long
ones, premerged chars, 5.1) is the tier's choice; the result is not.

The c twin (k6_c.c; spm_c.c's §5.3 calls the same loop through toks_k6_merge on its own initial symbols).
Symbols are numbered 0..n-1 left to right (the number orders pairs as the byte position does); round one's
prio of a pair of two bytes is bytepair[], of any other pair a merge-table probe. Both paths keep, per symbol,
the prio of the pair starting there (NONE when it has no merge, is dead or is the last symbol) and update only
the two pairs a merge changes: (left neighbour, merged) and (merged, right neighbour).
  - short (n <= 32): no heap; key[i] = prio << 5 | i in a fixed local array, the next merge is the unsigned min
    of key[] (lowest prio, then leftmost: hf's (rank, pos) order). tiktoken's and gigatoken's short cores;
    redteam-bpe measured 1.43-1.65x the heap on <= 16-byte misses.
  - long: hf's own procedure (word.rs merge_all): a min-heap of (prio, pos) built in one Floyd pass from round
    one; a popped entry applies only while it is current, cur[pos] == prio. The pair at a position only grows
    (merges extend symbols rightwards), so a changed pair is a longer string with another prio (one prio per
    pair, a merged id names one string): the stale test needs no merge-table probe, as in both asm tiers
    (redteam-arch measured +16% on llama3 Chinese against re-probing). Each merge pushes at most two entries:
    O(n log n), a 1 MiB piece included.
  Work area for pieces over 32 bytes (TOKS_BPE_WORK_BYTES(len) = 256 + 32 len, 64-aligned): heap u64 [2 len) at
  work (live entries <= 2 (n - 1): n - 1 from round one, then each merge pops one and pushes at most two),
  then u32 [len) each: ids, prev, nxt (alive neighbours, END at the ends), cur (the prio of the pair starting
  at symbol i, NONE when there is none). toks_k6_merge takes the same layout for cap >= n symbols, the
  symbols already at work + 16 cap.
The asm tiers also export the merge loop alone, toks_k6_merge_<tier> (spm's model, spm_bpe.md 5.3): given u32
symbols (2..128) enter the short path's records and round one as if every symbol were a premerged char's token.
The asm tiers: k6_avx2.S variants measured and rejected on tr9970x (Zen 5; tests/bpe/bench_x86.c, K6 over real-text
pieces, time against the kept code): prefetching the buckets each pair's merge would probe next +1..14% (the lines
hit already); branch-free MPROBE (tzcnt CF + cmov) and MID (cmov clamp) +3..9% in a same-run A/B (the hit branch
predicts well); neighbour-id records {id, mid, lid, rid} (one load per new key) +2..9% (more stores than it saves);
probes first, then a one-chain scan +33% (the scan becomes the chain); avx-512 MPROBE as one zmm compare into a
mask register -1..2% in a same-run A/B, below any rent, so the avx512 tier calls the avx2 kernel.
k6_neon.S, measured and not taken (GB10 X925 one core, e2e_commits, e2e.md's states, ids equal): prefetching
rank2id[prio] for every pair round one finds, so the first merge's merged id is not an L2 / L3 load on the chain
(the loop's own pairs are prefetched already, RPREF): llama 3 / o200k / glm 5.3 en pass +0.7..1.2%, cold +-0.8%,
ml / cjk -1.6..2.0%, the ids-as-rank controls (qwen 3.8, gpt2: one more cbz) -0.5..1.0%.
The ignore_merges probe (both asm tiers): no probe at all for a piece longer than every model token that starts with
its first byte (the vhash allocation of the bpe tables ends with u32 [256], each byte's longest token; llama 3: 15-18
bytes for the CJK leads E4..E9, so a zh run of 7+ chars skips the hash and the probe; the c twin does the same).
Otherwise the home vhash slot and the next three are read without a branch and the first that is empty or holds h
decides; ONE branch on that outcome ("an empty slot first: not a vocabulary token", the predicted way) is all bpe
waits on, and a candidate (or none of the four deciding) falls into the probe loop. The vhash line is a random one of
2 MiB (llama 3) and often comes from DRAM; the loop's per-slot branches (empty? h?) mispredicted about half the time
and each such miss waited for the line (llama 3 zh 16..30-byte calls, K6 alone, cold: 19 of 111 ns were the probe, 2
of 59 hot; with both changes the zh calls of 16..30 / 31..60 bytes take 102 / 167 ns, were 112 / 185). e2e.c 4 KiB, A
master 7220c92 vs B, ABBA x3, ids sha-equal, cold / pass / warm: gb10d X925 llama 3 en 1.04 / 1.04 / 1.01x, code
1.00 / 1.01 / 1.00x, zh 1.08 / 1.07 / 1.07x; o200k en 1.02 / 1.02 / 1.01x, code 1.01 / 1.01 / 1.00x, zh 1.05 / 1.05 /
1.04x; tr9970x Zen 5 llama 3 en 1.04 / 1.02 / 1.00x, code 1.01 / 1.01 / 1.00x, zh 1.07 / 1.07 / 1.07x; o200k en 1.08 /
1.02 / 1.01x, code 1.00x, zh 1.05 / 1.06 / 1.04x; qwen 3.8 (no ignore_merges, same path) 0.98-1.00x (tr9970x at load 3-4).
Measured and kept out: the probe after bpe, its line prefetched first (a one-id bpe answer is then the vocabulary
token itself): K6 alone zh 0.94x, en 1.0x, code 1.43x (whitespace runs and other vocabulary-token pieces ran bpe
first). A branch-free short path for 3..16-byte pieces (unmerged experiment 011f2c9; neon: one symbol loop with round
one in it, sentinel symbols at both ends, a one-block KEY scan, ascii pairs and two-byte pairs decided by selects,
NID / PID): e2e cold 0.97-0.985x on llama 3 / qwen 3.8 / o200k en and code (gb10d). The
mispredict cost it went after is real (the asm predictor untrained vs trained on the same hot data: 99 vs 55 ns a short
miss) but lives in the merge decisions themselves (does a new pair merge, is this an ascii pair, the loop exits), and
taking those without a branch puts loads in series: the pair filter's word must arrive before the bucket's address
exists (that variant: hot 55 -> 92 ns, cold 112 -> 135 ns), and the premerge advance becomes a chain i -> bytes -> apm
entry -> i. A predicted branch lets the bucket load and the next symbol run ahead; a select cannot. (Where pieces are
interleaved, as in a batch, those chains hide behind other pieces' work instead.)
Runs (k6_neon.S, LBL(k6_run); a fast path, so the c twin, the oracle, has none: its merge loop finds no pair and gives
the same ids). A piece whose bytes after the first are one ascii byte c, p[1, len) all c, is its byte ids, byte2id[p[0]]
then byte2id[c] x (len - 1), when neither (c, c) nor (p[0], c) has a merge (bytepair[]): round one finds no pair, so no
merge can follow. That argument alone makes it exact for every model: premerge and the ascii pairs start a symbol only
where a pair of the piece merges, and the path comes after the ignore_merges probe. The bpe entry pays one test, p[1] ==
p[len - 1] < 0x80, and a branch predicted not taken; out of line the (c, c) prio comes first (a model whose runs merge
goes back after one load), then the word compares, the (p[0], c) prio and the stores. Measured on the way (e2e_commits,
ids equal): inline, the path cost every other cell 0.6-1.8% (gb10e: gpt2 cjk -1.5%); without the ascii test, cjk pieces
whose second and last bytes are one continuation byte went out and back (gpt2 cjk -0.9%); the (c, c) prio read
branch-free in the test, so that runs which merge never leave, cost cjk / ml 1.0-1.3% (gb10e); and moving the bpe path
by 32 bytes, with 8 dead nops and nothing else, cost llama 3 code 1.2% on m2ultra2, so the out-of-line block sits ahead
of the bpe path, padded so that the path moves by a multiple of 64 bytes. gpt2 cuts '\n' + indentation as one piece and
has no merge in it ('\n' + 7 spaces is 8 ids), so code is where it pays: e2e_commits (gb10e X925 cpu 7, master
0d5fbad -> this, abba x5, ids equal in every cell) gpt2 code whole cold 421 -> 458 MB/s, pass 379 -> 408, warm 566 ->
630; 4 KiB cold 304 -> 326, pass 370 -> 398, warm 563 -> 625; the other 46 cells of gpt2 / llama 3 / o200k / glm 5.3 /
qwen 3.8 / kimi k3 x en / code / ml / cjk x 4 KiB / whole within -0.9..+0.9% in every state. m2ultra2 (unpinned) gpt2
code whole cold 361 -> 387.
A run-answer table, measured and not taken: answers built at load for one byte repeated k times and '\n' + one byte k
times (k <= 32), consulted at K6's entry. K6 timed by piece shape (tools/bench/stages.c, cold, gb10e X925): of gpt2 code
whole's 11,787 K6 calls 7,149 are '\n' + one byte repeated (118 us, 16.5 ns a call: a piece with no merge is K6's cheap
case), 368 one byte repeated (10 us), 491 other whitespace ('\n\n' + spaces, 13 us), 199 other two-run pieces (14 us).
One byte repeated or '\n' + it are 5.1-6.6% of gpt2 code's time (4 KiB / whole), 0.2-0.3% of llama 3 / o200k / glm 5.3 /
kimi k3 / qwen 3.8 code (their runs are tokens or cache hits) and 0.00-0.04% of every en / ml / cjk cell. Ceiling +7% on
one cell; the no-merge runs above take nearly all of it with no table, and a table adds only the runs that merge (under
2.5% of the cell) for a FORMAT slot and a build at load. What these pieces still cost is K5's: a hash, two probes and the
call, as their > 4 ids do not fit the dynamic cache.

5.1 premerge: certified initial symbols

A tier may start a 2- or 3-byte char c of the piece (B = its bytes, at [s, e): a lead C0..DF and one
continuation byte, or a lead E0..EF and two) as ONE symbol, its token t, instead of its bytes, when the premerge
table (layout.h) has an entry for B and neither the entry's risk bit for the byte before c (piece[s - 1]) nor
the one for the byte after c (piece[e]) is set; a side with no byte (the piece's start or end) is never risky.
Chars are taken left to right without overlap. The ids are unchanged; only the work shrinks (measured with
the same rule in python on the zh bench text: 88.6% (llama3) / 87.6% (glm53) / 97.7% (qwen38) of its
multibyte chars start premerged, saving 1.77 / 1.75 / 1.95 merges each).

The builder (bpe_build.c) gives B an entry when B is a model token t with merge-only BPE(B) = [t]
(ignore_merges is a whole-piece rule, applied before K6 merges anything). Let R be the highest prio BPE(B)
applies. While BPE(B) runs, the symbol at B's left end walks a chain b1 = P0, P1, ..., t; bound(P) = the
highest prio BPE(B) applies while P is the left end, the merge that ends P's turn included (it is pending all
along). Likewise the right-end chain S and bound(S). The risk bits:
  byte before:  the last byte of x, over every merge (x, P) with P a left end other than t and prio <= bound(P),
                and every merge (x, t) with prio <= R;
  byte after:   the first byte of y, over every merge (S, y) with S a right end other than t and prio <=
                bound(S), and every merge (t, y) with prio <= R.
(Bytes fold into layout.h's 22 risk bits; folding only adds risk.)

Claim: when neither neighbour byte is risky, hf's merges from the bytes (run A) and from the premerged symbols
(run B) end in the same ids. Facts used: every symbol is a vocabulary token, a merge's result is the
concatenation of its operands, one string per id (config.c), and two different pairs never share a prio (raw
ranks: one per pair; ids-as-rank: the merged id names one string, and two different pairs at one place make
different strings), so "<=" above is only conservative.
  1. In run A, until B is whole, B's bytes merge only with each other, in BPE(B)'s order. B's internal pairs
     change only through internal merges, and a chosen internal merge is the least of B's pairs, so B's part
     of the state walks BPE(B)'s states; one of them is pending at every step (BPE(B) has not ended), with prio
     <= bound(current left end), <= bound(current right end) and <= R. A merge across B's left edge pairs the
     symbol x before B (x ends at s: its last byte is piece[s - 1]) with the current left end P; not risky
     means its prio > bound(P) >= the pending internal prio, so it is never the least while P is the left end,
     and P stops being the left end only when B's own merges extend it. The same holds at the right edge. So
     B becomes t, and every merge before that is inside B or outside B, each outside one with prio <= the
     internal one then pending, <= R.
  2. Run B applies those outside merges first, in the same order: by induction both runs have the same outside
     symbols, the outside merge run A picks is the least outside pair, and run B's only other candidates, (x,
     t) and (t, y) with x ending in piece[s - 1] and y starting with piece[e], have prio > R (not risky). When
     run A completes t, both runs are in the same state, and from there on they are one run.
  3. Several chars: apply 1-2 to one char at a time. Nothing in them looks at what the other symbols are,
     only at the bytes next to B, so they hold with other chars already premerged.
Tests: test_bpe's hostile random models (alphabet with e-acute and U+4E2D, duplicate / reordered / multi-split
merges, rank2id, ignore_merges) against hf's rule restated from the merge list; mutants that drop the
before-byte bits, the after-byte bits, the (x, t) / (t, y) part, a tier's risk test or a bound each fail it.
tests/bpe/run.sh-style differentials against hf 0.23.2 on real tokenizers + zh / en / ml text, every tier.

Ascii pairs (the apm table). Nothing in 1-3 uses B's length or that B is one char: the claim holds for any B with
merge-only BPE(B) = [t] whose risk is computed the same way. So a tier may also start two ascii bytes b0 b1 at
[s, s + 2) as their token when the apm entry of b0 b1 is set and neither the class of piece[s - 1] is in its
before-mask nor the class of piece[s + 2] in its after-mask (folding into 32 classes only adds risk). Chars and
pairs are taken left to right without overlap; a pair of two byte symbols still takes its round-one prio from
bytepair. The asm tiers do it for a piece of 4..15 bytes whose second byte is an ascii letter (a word, maybe after
one mark): K5's short misses on english, where an 8-byte miss runs ~6 merges, each a dependent merge-table probe
(llama 3 en 5.98 -> 3.14 merges a piece); the c twin for every piece of 4..15 bytes. Measured and kept out (GB10
X925, 4 KiB cold): every 4..15-byte piece through the asm path (code 0.98-0.99x: punctuation runs gain nothing and
pay the per-byte test), a stricter word test (' '? letter letter: code 0.99x, its 12 instructions a call), longer
pieces (code 0.92-0.96x), certified ascii trigrams (2.20 merges a piece, but a second probe in the per-symbol chain:
code -2..-4%), prefetching the would-be probes of round one's pairs and of every new pair, alone (en +1%, code -2%),
a 2-token split of the miss certified over the edge chains (sound, 100% of 2-id answers, but ~30 random lines: 0.72x),
and rust-gems' backtracking longest-match encoder for the whole short miss (unmerged experiment 965f2cf; c in front
of K5's K6 call): the longest usable token prefix (words table + a per-token birth / split record), shorter ones on an
invalid pair, backtracking over dead positions; (a, b) is valid when merge-only bpe(a b) = [a, b], decided by
walking a's right edge and b's left edge back from (a, b) through the births (the prio of the merge that made each
symbol in its own run): the pair across before the later birth must not have a lower prio (ties: a's own merge
wins, the pair across beats b's); usable = merge-only bpe(bytes) = [it] with a non-decreasing run (then the
two-token run is the merge of two non-decreasing sequences). Exact (agrees with K6 on every short call of the en
bench text, llama 3 / qwen 3.8; test_bpe's hostile models), but greedy longest-match is right only 48-49% of the
time (a short miss is a word whose longest vocab prefix is usually wrong), so a call costs ~10 words probes + ~11
walk steps (a record load and a merge-table probe each) + 1.1 backtracks: cold en 0.45-0.48x, code 0.62-0.70x.
A cheaper cut finder for the 2-token misses (72% of short misses), costed on paper and not built: any exact 2-token
answer needs three dependent rounds of table reads that do not fit the 2 MiB L2 (the cut: which prefix and suffix are
tokens; their edge chains; the cross pairs under the bound), against ~28 ns for a 4x on a ~114 ns miss. The best
form (an L2-resident bloom filter over the vocabulary tested on every prefix at once, words probes only for the 1-3
surviving cuts, then both edge records and the cross probes in parallel) is ~50-60 ns hot, 1.9-2.3x on the class, and
~230 ns after other text, where each round is a DRAM line, against K6's 165-200 ns; the 28% that are not 2 tokens pay
the attempt and then K6. Net cold 4 KiB en +5..10% at the very best and ~0 in the pass, for ~150 lines a tier and
1.5-2.5 MiB of new tables. What reaches the 4x for such a piece is not computing it: the piece dictionary (§6) answers
the frequent ones at a static hit's cost.
Certified ascii 3- and 4-byte windows (commit ec02551: the claim above for 3..4-byte B, pm_sim and
pm_cand to three edge symbols a side, a direct-mapped 512 KiB window table probed 4 bytes first, then 3, then the
pair; 0 mismatches against hf 0.23.2 on 4.77 M pieces of llama 3 / qwen 3.8 / gpt2 / o200k, 26 / 26 mutants killed):
llama 3 en short misses 2.90 -> 1.66 merges a piece, yet GB10 X925 K6 miss en 1.006x, code 1.07x, zh 1.00x, K5
cold en 0.95x and warm 0.94x. The merges it removes are the cheap ones. Cutting stages out of the en miss (116 ns):
entry, premerge, records and output 42 ns, round one 13, the merge loop 61 for 2.9 merges; with the windows the loop
drops 16 ns for 1.24 merges (13 ns a merge; the merges left cost ~25) and the premerge grows 16 ns: probing both
windows at the walk's 3.5 positions a piece costs 8 ns with nothing taken, the rest is the take deciding after a
slot load and mispredicting. 74% of the takes are past the piece's first byte (a first-window-only form saves 0.31
merges). A branch-free form (every position's certificates at once, then a walk) probes 1.6x the positions and adds
the walk: more than the 11.5 ns the 13 ns merges leave for +5%, not built. Of the K5 loss the probes are 1-3%; it
is not the table's placement (master with a 512 KiB block before words, the same layout: 1.00x; huge pages: §7);
builds that differ only in an unprobed table measure 0.98 vs 0.96x (binary layout, not pinned down).
A cheaper per-char test in the premerge walk (one load and a mask compare for lead + continuations instead of the
compare ladder): its whole ceiling, the ladder deleted and every lead taken as E0..EF unchecked (a timing-only
variant), is K6 miss zh +2.2% (llama 3) / +2.4% (qwen 3.8), K5 cold zh +1.9% (GB10 X925): the ladder's branches
predict on real text and hide behind the char's two dependent table loads. Not built.

The pair filter. On chinese, 76-82% of K6's probes (llama 3 / qwen 3, pieces over 15 bytes) are pairs of tokens
with no merge, and the filter answers 94-95% of those without the bucket; on english / code only 19-21% of the
probes are such pairs. The c twin consults it on every probe (bpe_mt_find; dropping it for pieces without a lead
byte measured en 0.96x / code 0.98x on the scalar tier, so it stays), neon on every probe of the short merge loop
and in round one of the premerge path for a pair holding a premerged char's token (BYT 0x300), the bucket probe
following only when both bits are set. The merge loop's two probes sit on the dependent chain (the next merge waits
for them) while round one's overlap, so the loop is where it pays: GB10 X925 4 KiB cold vs round one alone zh
+10..16% (whole +11..20%), en +4..6%, code +1..2%, ja / ko / ml +3..6%. avx2 does not read it: in round one it
measured 0.89-0.95x on zh (tr9970x Zen 5): its bucket probe is one 8-slot avx2 compare, about the filter's own
instruction count, and round one's bucket misses overlap anyway; in the merge loop it measured zh 0.94-0.98x, en
0.96-0.98x (llama 3 / gpt-oss / GLM 5.3 / qwen 3.8, tr9970x cpu 14): Zen 5's bucket probe is short enough that the filter's
extra instructions on the chain cost more than the misses it saves.

Where K6's time goes (GB10 X925 cpu 7, master c008952's k6_neon.S; timing-only variants that each stop after one
stage, K6 alone over one class of pieces, best of 5 rounds a run, the median of 3-4 cycles in rotating order;
raw/k6-cut-gb10e-c008952.log). Each variant's total varies <= 1.7% over its cycles on en / code / zh (llama 3 code
4.5%) and up to 10% on ml short pieces; a stage is the difference of two variants' medians, so a row's stages sum to
its total (within 0.1 ns of rounding) and a stage under ~15 ns carries +-1-2 ns. ns a piece: llama 3 en short misses
(2..15 B the words table does not answer, 8.0 B mean) 101.4 = entry and premerge walk 27.0 + records 5.2 + round one
13.1 + merge loop 56.1; o200k en 105.8 = 27.2 + 4.9 + 13.1 + 60.6; llama 3 code 79.7 = 26.2 + 3.6 + 9.5 + 40.4; zh
over 15 B (26 B mean) llama 3 130.2 = 43.2 + 12.0 + 35.8 + 39.3, qwen 3.8 111.2 = 26.5 + 8.1 + 40.2 + 36.5; ml
(gen.py over the ml corpus) llama 3 short 86.8 = 41.8 through round one + 45.0, over 15 B 225.5 = 91.9 + 133.6. Round
one's char-token pairs split again (the bucket probe dropped, then the filter's branch too, its bits still computed):
the branch reads 16.0 / 23.5 ns a zh piece over 15 B (llama 3 / qwen 3.8), the probes 4.0 / 7.4; ml over 15 B 12.9 /
17.6 and 12.9 / 15.5. Round one in two passes, measured and not taken (exact: pass 1 computes each such pair's filter
bits without a branch and appends its position to a survivor list, the store always made and the end advanced by the
bit; pass 2 probes the survivors; test_bpe_neon in rb's tree and bench_neon on llama 3 / qwen 3.8 zh and ml exact,
both outputs in the log): K6 alone zh over 15 B 130.6 -> 131.5 ns (llama 3), 112.0 -> 107.9 (qwen 3.8), ml over 15 B
1.00 / 1.01x, every short-piece cell slower (zh 0.96 / 0.93x, ml 0.98x; the per-cycle ratios agree within 1.2%). The
cut's 16-24 ns was not a mispredict to recover: behind a predicted branch each bucket load starts right after its
pair's filter word, while in two passes the probes wait for the list as data and the second loop costs the short
pieces more than the branch did (the branch-free short path's lesson in 5 again).

5.2 rejected: certified sub-word cuts for long CJK pieces

Measured and rejected (2026-10-04). The idea: zh pieces over 15 bytes hold ~67% of the bytes and are ~99% first
occurrences, so cut them where merge-only bpe M provably factorizes and answer the sub-words from the cache. The
rule: cut between chars c1 | c2 when both are premerged in context (5.1) and no vocab string holds bytes(c1)
bytes(c2) (a load-time pair filter). Proof (spm_bpe.md 5.5's argument at char level): by 5.1 M(P) is the run from
P's bytes with c1, c2 premerged; a symbol covering both sides would come from a merge whose result, a vocab string,
holds bytes(c1) bytes(c2); so the pair at the cut never merges, every step is its own side's least pair, and the
run is M(left) ++ M(right) (only "prio is a function of the pair, ties to the leftmost" is used, so tiktoken's shared
ranks are covered); certification survives into the parts, which drop neighbours. A piece with a cut is no vocab
token, so ignore_merges cannot answer it. Checked (unmerged experiment 5cffccb): every
piece over 15 bytes of the bench cjk + ml corpora on llama 3, qwen 3.8, GLM 5.3, gpt-oss, Kimi K3 (920,419 pieces,
3,579,344 cuts) against hf 0.23.2's merge-only BPE (tiktoken's for Kimi): 0 mismatches; without the contextual risk
test 9,087 pieces break (the context-free "single-token chars" rule is wrong). 86-97% of the cjk long-piece bytes
land in sub-words <= 15 B. It does not pay (the walk in K5's long path, tr9970x Zen 5 and m2ultra2 M2, 4 KiB
chunks, B/A cold / pass / warm): llama3 zh 0.92 / 0.99 / 1.01, qwen38 zh 0.79 / 0.86 / 0.85, glm53 zh 0.88 / 0.94 /
0.94, ja up to 1.33 warm only, ml 0.98-1.00. After 5.1 a whole long zh piece costs K6 only 125-191 ns; the walk's
premerge lookup + filter probe per char cost 56-73 ns a piece before any cache probe, 78% of the sub-words are one
char (free either way), and the multi-char ones hit 53-71% in one pass with 15-30% of them over 4 ids.

5.3 not taken on avx2: the low-id pair grid

Measured and not taken on the avx2 tier (2026-10-05, tr9970x Zen 5 cpu 20; unmerged experiment 0e488b1 over the c twin
of unmerged experiment 871852b; receipts docs/bench/raw/k6-grid-tr9970x-*.log). The grid: 4 MiB of u32 prios, cell
(l<<10)|r for every pair of ids below 2^10, filled from the merge table's final slots (the table's own answers, never
a second truth). avx2: MPROBE tests the key first (one `test rax, imm32` / jnz: an id at or past 2^10 goes to the
table as before), then loads the grid pointer and answers with one load; no new register, and all five call sites take
it. Exact: make test both tiers; test_bpe with models whose ids reach past 2^10, so pairs sit below, across and above
the gate (a build without the gate fails it with 4,842 failures; with every test id below 2^10, as the test had it,
nothing could see the gate); tests/bpe/check.c against hf 0.23.2 on llama 3, llama 4, gpt2, o200k, qwen 3.8 and GLM
5.3. In isolation it pays: bench_x86, the same tables, the shipped k6_avx2.S in the same run: K6 first (each distinct
piece once) en 1.08-1.16x, code 1.06-1.12x; K6 all 1.11-1.15x. End to end against master ac14d02 (e2e_commits, 5 abba
rounds; cold back to back, pass after other text): en cold +5.9..8.3% at 4 KiB, +3.8..6.0% whole; en pass +0.7..4.8% /
+0.3..5.6%; llama 3 / gpt2 code +1.0..4.1% in every state. It does not meet the bar (no cell below -0.5%): gpt2 cjk
0.982-0.989 in every state and both chunkings, o200k cjk 0.967-0.995. Four variants were measured (the gate after the
pointer load; the gate first; the long path's loop head on a 64 B line; the grid allocated after every other table, so
their arena offsets are master's): o200k cjk lost in all four, gpt2 cjk in the last three (+0.4..0.9% in the first,
whose K6 is 16 B longer than the second's). What holds the binary fixed sees nothing on cjk: the same code bytes with
the grid pointer read from a zero slot against the grid read, 0.994-1.005 at 4 KiB; bench_x86 on cjk pieces, K6 all
0.99-1.01x. Only gpt2 cjk's K6 first is down (0.984-0.993x): its byte-level pieces lose on their first sight with the
grid code in K6. The bytes, from the counting build (c twins, one fresh-scratch pass, distinct 64 B lines of K6's
tables): the grid halves the merge buckets' lines and adds about as many of its own, llama 3 en 24,771 -> 25,861
lines, gpt2 en 10,305 -> 14,087, o200k en 30,953 -> 30,491, code -10..+11%, cjk about 2k grid lines.
A method lesson from the same work, for every A/B on tr9970x: 128 B of never-called text linked before the asm (the
kernels start 128 B later, at the same offset mod 64; nothing else changes) moved e2e cells by up to 2% at 4 KiB:
o200k code pass 0.979, llama 3 code pass / warm / lang-x 0.987 / 0.993 / 0.989, every abba round one-signed but one
(1.001); en and cjk within 0.8%. A difference of that size on a code cell is not a change's until a layout control
says so. The o200k code rows above (pass 0.993 at 4 KiB, 0.979 whole) are inside that band.

6. K5 encode pieces
--------------------

For each piece [s_i, e_i) (s_0 = start, s_i = e_{i-1}, e_i = ends[i]): ids = K6(piece), appended to out.
Shortcuts may answer instead of K6, and must give exactly K6's answer (hf's: under ignore_merges with
TOKS_TF_PROBE_LONG, for a token of 2..15 bytes whose own merges do not rebuild it, that is the static table's
answer, not K6 alone's; the CONTRACT below):
  - a one-byte piece is [byte2id[b]] unless ignore_merges says otherwise (it never does for a single byte
    that is a vocabulary token: byte2id[b] is that token);
  - static table (words, two lines): certified entries (key -> up to 4 ids) for pieces of 2..15 bytes. Static
    table: model-vocabulary tokens of 2..15 raw bytes whose K6 answer has 1..4 ids, each valued by running K6's
    c twin on the tables just built (never "the token whose string this is": SPEC §2.7); two-choice (bucket
    h & words_mask, else rotr32(h, 16) & words_mask), two ways, filled first fit in id order (low ids are the
    frequent ones); a key whose two buckets are full is left out (a shortcut: K6 computes whatever the tables
    lack), except, under ignore_merges, a token whose own merges do not rebuild it: bpe_build.c makes room for those
    (§5 vhash) and only then sets TOKS_TF_PROBE_LONG, so for them the table is a requirement.
    Every way still free then takes the piece dictionary (src/gen/dict.c, tools/gen/dict.py): pieces of 2..15 bytes
    that common text cuts and that no reference model holds as one token (" Bingley" = " Bing" + "ley"), in the
    order of their frequency in public text held out from the bench (enwik8; llvm 21.1.8's headers and four python
    packages: THIRD_PARTY_NOTICES.md), each valued the same way (the list picks which pieces get an entry, never what
    one says; a piece of more than 4 ids, one with a byte the model drops and one whose two buckets are full are
    passed over). A list piece sits in a line K5 reads for its first probe anyway, so its hit costs a static hit
    where it was a K5 miss and a K6 run, and the table keeps its size. Seated (131,072 pieces, 1.15 MB of list):
    llama 3 107,383 -> 129,684 entries, gpt2 45,938 -> 65,499, o200k 180,044 -> 236,788, qwen 3.8 208,726 ->
    247,409, glm 5.3 146,992 -> 222,650 (the list runs out first). The build pays 0.3-0.9 us a seated entry (the
    bpe build on a GB10 A725: llama 3 49 -> 67 ms, gpt2 20 -> 30, o200k 92 -> 126, qwen 3.8 111 -> 124, glm 5.3 53
    -> 101); toks_load as a whole, the median of the 64 loads a side in the GB10 X925 runs below: llama 3 178.5 ->
    187 ms, gpt2 33 -> 39, o200k 309 -> 333, qwen 3.8 268 -> 280, glm 5.3 218 -> 244, kimi k3 182 -> 206.
    e2e_commits, GB10 X925 cpu 7, e2e.md's states, ids equal, cold / pass / lang-x MB/s at 4 KiB (abba x5): llama 3
    en 265 / 255 / 276 -> 276 / 259 / 280, code 410 / 293 / 309 -> 430 / 298 / 314; gpt2 en 294 / 363 / 388 -> 303
    / 369 / 393, code 320 / 377 / 410 -> 365 / 392 / 424; o200k en 256 / 227 / 243 -> 280 / 235 / 253, code 441 /
    286 / 300 -> 488 / 302 / 316; qwen 3.8 en 276 / 257 / 275 -> 291 / 263 / 281, code 379 / 302 / 316 -> 421 / 310
    / 325; glm 5.3 en 272 / 239 / 257 -> 309 / 252 / 270, code 412 / 274 / 288 -> 464 / 288 / 303; kimi k3 en 274 /
    259 / 279 -> 305 / 271 / 291, code 458 / 320 / 336 -> 510 / 334 / 348; dsv3 (a later run, 1cb69dc -> 10dee09)
    en 255 / 292 / 320 -> 262 / 294 / 323, code 311 / 319 / 335 -> 330 / 326 / 342; whole files (abba x3): llama 3
    en 340 / 274 / 277 -> 348 / 280 / 281, o200k en 307 / 243 / 244 -> 324 / 253 / 254, glm 5.3 en 330 / 255 / 258
    -> 358 / 270 / 274, gpt2 code 434 / 391 / 418 -> 453 / 407 / 438, dsv3 en 345 / 307 / 310 -> 350 / 312 / 315,
    code 405 / 332 / 341 -> 416 / 341 / 348; every ml / cjk cell -0.6..+3% (dsv3 ml whole cold 166 -> 165). The
    memo's replay (warm) never reads the table (K5 does not run on a replay), and its ~40 us passes, timed at 1 us,
    move a few % between runs with nothing changed: the three replay cells that read lower at 5 or 3 rounds (qwen
    3.8 code 4 KiB 14,133 -> 13,970, whole 13,660 -> 13,511, glm 5.3 code 4 KiB 8,781 -> 8,720) read 13,660 ->
    13,660, 13,221 -> 13,363 and 8,537 -> 8,598 at 9 rounds, and the same code with the list off (toks_dict_n = 0)
    reads qwen 3.8 code 4 KiB 13,363 -> 13,660 against the same master. Bytes moved fall with it (the counting
    build, one fresh-scratch pass at 4 KiB, B per input byte): K6 o200k en 11.52 -> 10.20 (its table lines 0.97 ->
    0.92), glm 5.3 en 11.59 -> 9.75, llama 3 en 11.87 -> 11.30, o200k code 11.55 -> 10.50; the words table's
    distinct lines o200k en 39,666 -> 38,839 (a list hit replaces a two-line miss). Receipts, with their tables in
    docs/bench/commits.md: docs/bench/raw/commits-4913495-9be533c-gb10e-{4096,whole,whole-replay9}.log and
    commits-1cb69dc-10dee09-gb10e-{dsv3-4096,dsv3-whole,replay9-4096,replay9-whole}.log, the list off in
    commits-{4913495,1cb69dc}-dictoff-gb10e-*.log; an M2 Ultra's (unpinned: shape) in commits-3647c5b-*-m2ultra1-*.log,
    and its gate against gigatoken (kimi k3 4 KiB cold en 1.48x -> 1.65x, code 1.72x -> 1.89x) in
    gate-m2ultra1-neon-{245cc5c,42e16c9}.log.
    One-byte tokens stay out: K5 answers a one-byte piece from byte2id before any
    probe, and their 256 keys differ in one byte, so their crc32c hashes span 8 bits and clog the buckets;
    the keys left out (llama 3 18,904 of 126,153, gpt2 3,933 of 49,871, o200k 14,206 of 194,250) are not worth a
    third way at the same bucket count (counted, not built: of K5's K6 calls the single-id answers of 2..15 B,
    which a complete table would give, are llama 3 en 2,071 of 37,336 at 4 KiB cold (gpt2 en 2,481 of 38,785,
    o200k en 1,825 of 35,913, qwen 3.8 en 246) and 584 of 17,357 in a whole-text pass; x their in-context K6 call
    + K5's miss side: +2.8..3.0% of the 4 KiB cold pass on GB10 X925 (bench_neon pass +0.3..3.1%), before the
    third key compare on every hit, for a format change in three kernels; bigger tables: docs/bench/stages.md);
    nor is a denser layout worth its format: 3 ways x 1 choice at the same bucket count (keys 16 B + one u32 id,
    one line per probe, more keys seated than today) touches fewer table lines a pass (llama 3 en 33,998 ->
    23,807, code 10,429 -> 7,893, o200k en 39,662 -> 26,486, gpt2 en 24,606 -> 18,906: a replay of one
    fresh-scratch pass) and predicted +9..12% after other text at ~75 ns a line; measured (K5's c twin on both
    sides, neon K6, unmerged experiment 2ffe7c1, e2e.md's states, GB10 X925 cpu 7): pass (after other
    text) llama 3 en +1.5..2.3%, code +1.9..2.4%, o200k / gpt2 / qwen 3.8 +0.6..3.8%, cold (back to back)
    +1.4..4.3%: the table's lines are about a third of what the polluted pass loses, under the +7% bar;
    nor is a fuller table at the same shape: first fit of the model's tokens saturates near 0.82 load, and moving
    entries to their other bucket (cuckoo, depth 3, only entries with an id >= 1/3 of the key being seated, so the
    frequent low ids keep their first bucket) seats gpt2 45,938 -> 49,113 keys, llama 3 107,383 -> 115,606, o200k
    180,044 -> 191,961, qwen 3.8 208,726 -> 225,174; measured (e2e_commits, e2e.md's states, GB10 X925 one core,
    ids equal): 4 KiB cold gpt2 en +1.7%, o200k en +1.2%, llama 3 en -0.4%, qwen 3.8 en -1.4%, glm 5.3 code -1.2%;
    stages (llama 3 en 4 KiB cold): K6 calls 37,685 -> 36,214 (-0.19 ms) but K5's static hits +0.45 ns each
    (+0.18 ms): a hit moved to its second bucket costs ~19 ns hot (the mispredicted first-bucket miss, then the
    line), not the 2-5 ns a line model gives, and the replay of K5's probes over-counted the left-out keys that
    occur (2,106 predicted, 1,471 measured). Seating the list's pieces that way too (a piece whose buckets are full
    takes the place of any entry that can move, bucket h's first way first) seats o200k 236,788 -> 250,073 entries,
    glm 5.3 222,650 -> 239,084, qwen 3.8 247,409 -> 256,064, and loses every 4 KiB cold en / code cell 2.7-12.4%
    (ml / cjk cold -1.6..0%, the pass, lang-x and whole files -1.7..+0.9%; gpt2's 21 such moves alone cost K5 +0.81
    ms a pass: a bucket's first way holds its most frequent token, which then pays the second line on every hit;
    docs/bench/raw/commits-9be533c-moves-gb10e-*.log). Not taken; nor is a table doubled for the list (when that
    adds <= 4 MiB: llama 3 4 -> 8 MiB, gpt2 2 -> 4): the list's own hits pay there too (against the same doubled table
    without the list, 4 KiB cold / pass: llama 3 en 276 / 240 -> 322 / 256 MB/s, code 427 / 279 -> 491 / 295, gpt2
    code 335 / 363 -> 456 / 396), but the doubling alone gains cold en / code +4..5.5% (every vocabulary key
    seated) and loses every corpus after other text (master -> doubled, pass / lang-x: llama 3 en 254 / 276 -> 240
    / 256, ml 123 / 129 -> 118 / 123, cjk 135 / 143 -> 129 / 134, gpt2 ml / cjk -3..-6%; llama 3 cjk cold 148 ->
    140): a pass after other text waits on the table's lines from L3 / DRAM, and the doubled table spreads the
    same keys over more of them (docs/bench/raw/commits-4913495-doubled-dict-gb10e-*.log, the list in the doubled
    table, and commits-4913495-doubled-gb10e-*.log, the doubling alone). Not taken; nor is a second table K6 probes
    for what the words table leaves out (its passed-over list pieces and left-out tokens answer 14-36% of the 4 KiB
    cold K6 calls on en: llama 3 10,672 of the 32,162 that would probe, gpt2 11,130 of 33,035; on ml after other
    text 4-13% of those). Its costs, measured before any feature code (unmerged scratch on 15e3e02, K6 neon, never a
    hit, ids equal; GB10 X925 cpu 7, e2e_commits, 24 cells, medians of the unrounded runs): every 2..15-byte ascii
    piece that is no tail run hashes its key, tests 2 bits of one 64-byte block of a 256 KiB bloom and, on a pass,
    both buckets of a 4 MiB 2-way table. The bloom all clear costs cold -0.8..-2.9% and after other text
    -0.7..-5.1% in every cell, cjk included (the test itself, beyond the probe-off control below and pooled over
    the six models' en: 5.4 ns a gated piece cold, 10.7 ns after other text); the bloom all set (every gated piece
    probes) cold -1.1..-12.6%, after other text -3.1..-14.3% (both buckets 19.6 / 42.1 ns, §5 vhash's one line 12 /
    41 ns). Of the bloom's cost, the same code and arena with the probe off (a test of the bloom's pointer) already
    take cjk -1.3..-1.9% (gpt2 -0.4%) and en after other text -1.7..-3.0% (gpt2 +0.2%): the 70 instructions K6 grew
    by, or the tables' +4.25 MiB after the words table, not separated. With the hits counted at those costs, en /
    code cold gain (+3.0..+10.7% / +0.6..+5.8%) and every ml / cjk cell loses more than 0.5% (cold and after other
    text, -0.7..-2.6%), as do qwen 3.8 / glm 5.3 / kimi k3 en after other text (-0.9..-1.2%)
    (docs/bench/raw/commits-15e3e02-bloom{clear,set,off}-gb10e-4096.log). Not taken;
  - dynamic cache (one line): the answers K5 has looked up (static) or computed (K6, at most 4 ids) for pieces
    of 2..15 bytes.
CONTRACT (K5 and K6, every tier): K5 looks every piece of 2..15 bytes up in the static table before it calls K6
(in both orders below, warm and fresh, and on every tier: a prefetch, a bypass or a new order must keep it), and
K6 under TOKS_TF_PROBE_LONG answers a piece of 2..15 bytes with the merges alone (§5). A K5 that skipped the lookup
for any piece class breaks llama 3's and glm 5.3's ids (532 / 792 such tokens), and the tests say so loudly: in
make test, test_breadth for the asm tiers (k5_neon.S with the lookup skipped for every piece, or for 3-byte pieces:
"nonalpha encode text 39: got 4 ids, want 3") and test_bpe's K5 checks for the c twin, which they run unless built
for a tier with -DK5_ENCODE (expect() on the trap / unreachable / test_seat tokens, k5_stream against the
reference: the lookup skipped for 3-byte pieces, 448 failures; for all, 1,023); tests/bpe/check.c reports K5
MISMATCH and FAIL on any tier it is built for (llama 3; o200k, with no such token, stays exact). check.c's 'by K5's words' count is
K6 alone, never K5: a K6 answer may differ from hf only on a model token of 2..15 bytes whose own merges do not
rebuild it and whose static entry is hf's answer, and every such piece is counted and printed.
Key: a piece of 2..15 bytes is keyed in place (layout.h's key, never copied): one 16-byte load masked to the
piece when s_i + 16 <= len, otherwise loads inside the piece (text[len..] is never read, §1). One hash per
piece selects the cache bucket and both static buckets.
Order and fill: a K6 answer of at most 4 ids for a piece of 2..15 bytes is written to the piece's cache bucket
(new entry in way 0, old way 0 to way 1, way 1's entry dropped); a cache hit fills nothing. The rest depends on
the scratch's age, the pieces it has seen since toks_scratch_init (the sum of K5's in counters, §7): a WARM
scratch (>= TOKS_K5_WARM = 4096 pieces) probes the cache first, then the static table (bucket h, then bucket
rotr32(h, 16)), then K6, and a static answer fills the cache too, so a piece seen before is one cache line
whichever table answered it first; a FRESH scratch probes the static table first, then the cache, then K6, and
static answers stay out of the cache (a call that starts from an empty cache meets few pieces twice, and a fill
or a probe of an empty line is then pure cost). Measured, K5 alone on Zen 5 (tr9970x) and M2 (m2ultra1), asm, 2 MiB,
against the old rule (cache first, K6 answers only): from an empty cache over a whole text +7-23%; the same text again
+40-60%; warmed on other text of the same languages +5-38%; per 4 KiB call from an empty cache (a scratch bound
per call) +6-10% (filling static answers there too measured -14..-21%). The old rule's c-twin sweep had shown the
opposite for whole texts because the c twin's table-driven crc dominates its time (measured with neon's K6 behind
it, gb10c X925: the c twin is 2.1-2.4x behind k5_neon.S on every byte-level cell, llama3 code 4096 cold 201 vs 424
MB/s; the scalar tier ships for exactness, not speed, and keeps no isa builtins). Every tier follows the same
rule, so outputs, counters and cache bytes do not depend on the tier. Prefetching is the tier's choice; the
result is not.
The avx2 tier on Zen 5 (tr9970x cpu 20; tests/bpe/bench_x86.c K5 alone, A B B A x 9-11 passes, and tools/bench/
e2e_commits.sh x 3 rounds, ids equal; receipts docs/bench/raw/k5-tr9970x-d11f9d1-*.log, commits-tr9970x-d11f9d1-staging-*):
at 2 MiB the probe line's latency is already hidden by the out-of-order window, so prefetching buys almost nothing.
Measured and not taken (measured 2026-10-04 on the 9970X): round staging (a pass A
over the round's pieces, branch-free: key, crc32c, one prefetcht0 of the first line the probe order reads; pass B
the probe loop as it is): pass A costs ~0.9 ns a piece and its prefetches give ~0.25 back (gpt2 en, 483,761 pieces:
+0.44 ms without the prefetch, +0.33 ms with it; real loads, prefetcht1 or prefetchnta instead: the same); e2e 4096
cold 0.94-0.97x, pass 0.94-0.97x, warm 0.89-0.90x (llama3 / gpt2 x en / code), and at TOKS_SCRATCH_CACHE_MIB(16)
warm 0.86-0.88x, lang-x 0.95-1.00x (the bigger cache's cold lines are hidden too). A variant storing pass A's key
and hash only saves pass B's rebuild, never pass A's cost: not built. Putting the miss chain's next line on its way
(words line h before a warm scratch's cache probe, line rotr32(h, 16) on a cache miss, the cache line when a fresh
scratch's line h misses): warm 1.04-1.06x slower, the rest within 1%. Huge pages for every malloc'd region (glibc
hugetlb=1): warm 0.98-1.00x. The pattern of one-byte pieces matters (gpt2 en 23.4% of its pieces, llama3 / o200k en
10.6%, code 14-26%): the same pieces with the one-byte ones at a fixed period run K5 warm 15.5% faster on gpt2 en
(2.33 -> 1.97 ms), 1.4% on llama3 en. No branch-free shape recovers it. Taking the next piece without a branch when it
is one byte (cursors advanced by 0 or 1 at the loop's exit) put a load on the loop-carried chain and lost 4-36%.
Sending one-byte pieces through the warm probe path (built: every piece of <= 15 bytes probes its line, a one-byte
piece's miss masked and its id and count selected by cmov, its static-hit count taken from the ends in the epilogue;
exact: selfcheck, test_bpe, hf) costs ~3 cycles on every piece (K5 wo hot 0.72-0.76x) and returns little of it:
wo warm 0.88-0.95x, wo cold 0.94-0.97x (gpt2 / llama3 en, gpt2 / o200k code). The run lengths rule out a SIMD run
path (one-byte pieces are isolated: runs average 1.01-1.44 pieces, 51-99% stand alone). What the fixed period buys is
then most likely the predictor's (a regular global history for every branch; tr9970x has no perf counters to show it), not
a lever the kernel has; the branch stays.
Two pieces a step on a warm scratch, measured and not taken (measured 2026-10-04 on the 9970X, cpu 20, bench_x86 with tier avx2
the pair build and tier x the shipped k5_avx2.S in the same run; receipts docs/bench/raw/k5-tr9970x-d11f9d1-pair-*.log).
Built: when the next two pieces are both 2..15 bytes and both windows end inside the segment, both keys, hashes and
cache probes go out before either result is used, and a double hit writes both ids; a one-byte or long piece, or a
miss in either, sends the first piece through the one-piece loop, so there is no new exit (exact: selfcheck, test_bpe,
hf). What it saves is shared loop control, and that shows only where the predictor has learned the stream (K5 wo hot,
a 4096-piece window: llama3 en +2.9%, o200k code +3.1%, gpt2 en -1.2%, gpt2 code -6.2%). In text order the pair test
is a branch of its own and replaces nothing: it fails on 28-60% of its tests (gpt2 en 50%, llama3 en 28%, gpt2 code
60%, o200k code 35%; the best fixed rule over its own last 8 outcomes still misses 27-37%:
docs/bench/raw/k5-tr9970x-pair-stats.log), and every failure then takes the one-byte branch as well. K5 wo warm 0.89x (gpt2
en 2.30 -> 2.58 ms), 0.93x (llama3 en 2.21 -> 2.39 ms), 0.89x / 0.92x (gpt2 / o200k code); wo cold 0.89-0.92x.
Predicted before the build at +0..3%, counting the pair test as a replacement for the one-byte branch, which it is
not; the kill bar was +3% warm en. The warm hit path stops here: the probe has no lever left on Zen 5.
A second stream on the same core shows how much of it one stream leaves idle (measured 2026-10-06 on the 9970X,
master 474f0fc, tests/par/run_par.sh with TOKS_PAR_EAGER=1, llama 3 on enwik8, flags 0; receipts
docs/bench/raw/par-tr9970x-474f0fc-smt-*.log and docs/bench/par.md's SMT sections): toks_par with two participants on
cpu 8 and its sibling 40 against one on cpu 8 runs 1.40-1.48x in the pass state (one input of 16 or 32 MiB 335 ->
468-475 MB/s, 16 MiB of 4 KiB documents 336 -> 494-497; two runs), gigatoken 1.33-1.47x on the same cpus. So one
stream leaves about a third of a Zen 5 core's issue capacity to stalls, mispredicts or load latency (a sibling cannot
tell the two apart), where an execution-bound loop typically gains 0-10% from a sibling. The two-piece loop above was
the software attempt at those slots and lost to its own extra branch; a sibling takes them with no branch added. On a
CCD, 16 threads on 8 cores against 8 (A B B A x 3, medians of six runs a side): pass +15.5% for the
documents (2217 -> 2560 MB/s), +23% for the 16 MiB input (1876 -> 2308), +31% for 32 MiB (2019 -> 2654), every round
one-signed; first 0.83-1.02x (twice the scratches to fault in: kernel work a sibling does not hide). Why a CCD gains
less than one core is not measured; the likely reason is the L3, sixteen scratches and llama 3's tables in the CCD's
32 MiB where there were eight. gigatoken at 16 threads against 8 loses on the 16 MiB input and the documents (0.875x,
0.894x).
Long pieces (opt-in: TOKS_SCRATCH_CACHE_MIB, §7): every K6 call goes through toks_k5_long_<tier>(t, k6 args,
a.lcache) (k5_long.c, one c file for every tier). lcache NULL (the default, or a fresh scratch: the driver
passes it only to a warm one) is K6 itself. Else a piece of 5 bytes or more (shorter ones have at most 4 ids,
the short cache's) is looked up in the long-piece cache (bpe.h toks_lcache): 64-byte buckets of 4 slots { hash,
len, arena offset, generation } in front of an append-only arena of entries { n, the piece's bytes, its ids },
an entry of <= 64 bytes never straddling a line. A hit needs the slot's generation (the call's), hash and length
equal and every byte of the piece equal to the stored copy (the piece is read in place, never copied for the
lookup); its ids go straight to K5's out cursor (16 bytes at a time: n <= len and the room is len + 4). A miss
runs K6 and, when the short cache cannot hold the answer (more than 15 bytes or more than 4 ids), appends it and
puts its slot first in the bucket; a full arena starts the next generation (every slot stale at once). Whatever
the cache memory holds, a slot answers only inside the arena's filled part and with n <= len, and the
descriptor's pointers and sizes come from the scratch header (the driver builds it per call), so no content of
the scratch can make K5 read or write outside its regions (SPEC §7.2). K5's misses counter counts these
calls whether the long cache or K6 answered.
Measured and not taken (the wipe stays; not merged): a full arena that keeps answering instead
of starting the next generation. The zh replay win needs a freeze before any replay exists; that freeze is the
switch cost. Rules simulated exactly on the real pieces (scratch tool: lc_get / lc_put over toks_pieces + K6;
llama3 / gpt2 / qwen38 x zh / ml / code / en, 4..32 MiB, pass / warm / third pass / lang long-cache hits), the first
also timed (tr9970x cpu 28, CACHE_MIB 8 / 16 / 32, abba vs master, 5 rounds):
- the dry rule (a full arena appends nothing until 1024 lookups in a row miss): warm llama3 zh 4096 188.2 -> 266.5
  MB/s at CACHE_MIB(8), 180.5 -> 391.3 at 16 (qwen38 zh x1.37 / x1.98, gpt2 zh x1.29 / x1.70), but lang on code
  476.1 -> 427.3 (qwen38, 8), 459.2 -> 394.0 (gpt2, 16), and gpt2 ml lang x0.967 / pass x0.983 at 8: 1024 misses
  in a row arrive late for a small repetitive corpus on an arena full of other text (gpt2 code lang: 8393 long-cache
  hits with the wipe, 5906 dry), and a trickle of shared pieces keeps resetting the run;
- a freeze for good, or for an arena (or a quarter) of refused appends without a hit: lang collapses to ~0 hits;
  an arena written as a ring: as bad as the wipe on a replay; dry runs of 256 (the thrash returns at 4-8 MiB) or
  4096 (gpt2 ml lang -17% at 4 MiB);
- ghosts (a full arena leaves a ghost per refused piece at its would-be position; G of them in a row that a
  wiping arena would still hold start a generation): G = 8 brings code / ml / en lang back to the wipe's (gpt2 code
  8393 / 8393 / 8377 at 8 / 16 / 32) but breaks zh replays (qwen38 zh 16 MiB third pass 72563 -> 4012 hits, gpt2 zh
  8 / 16 lose their second pass: 21525 vs 51032); G = 16..64 break the same cells and keep code lang short at 16;
- windowed net evidence (ghost hits minus hits per 1024 lookups, threshold 16..128): the same zh replays break, and
  code lang at 16 MiB stays short (qwen38 868 vs 1336).
Also measured: on zh, CACHE_MIB(8 / 16) is slower than the 2 MiB default (no memo) in pass as well (llama3 zh 4096 188 -> 166 / 158
MB/s): its appends (6.9 MB of entries per 2.9 MB of text) cost more than its within-pass hits save. A replay is the
segment memo's job (§7).
Coverage, and two designs measured on paper and not built (master, gb10c, the 2 MiB cache without a memo, counters per pass
of the e2e 4096 chunks): K5 at 2 MiB answers 99.0% of en's pieces and 98.8% of code's on their second sight (warm;
gpt2 code 94.2%), so a hit bit for the cache's replacement could recover at most ~1% of en's pieces (~4% on ml, ~2.6%
on zh). That 1% is not small in time: each is a K6 run, 150-180 ns against a hit's few, so it is ~20% of en's warm
pass (an 8 MiB short cache answering it: +19..25% en warm). zh's and ml's misses are structural: 31.6% of zh's pieces
are over 15 bytes and 8% have more than 4 ids (ml: 5.7% and 3.1%). Wide entries for them in one direct-mapped array
(64 or 128-byte slots, overwrite on fill, no arena, no admission; half the budget), simulated on the real pieces:
llama3 zh's second sight answers 2.1k / 9.3k / 26.6k of its 90.1k long pieces at 2 / 8 / 16 MiB with 128-byte slots
(coverage 57.7% -> 58.6 / 61.8 / 69.4%; 64-byte slots refuse ~41% of them), below the arena from 8 MiB on ml, ~1.6-3k
on a first sight at any size: zh's distinct long pieces need ~7 MB of slots, and a fill costs about what a hit saves
(predicted neutral to -8% warm at 16 MiB, a loss in pass).
Two-choice placement for the dynamic cache, built in the c twin and not taken.
tools/bench/k5_cache.c models K5's cache exactly on e2e's piece stream: its counters equal the library's in 96 of 96
(cell, state) triples (llama3 / gpt2 / qwen38 x en code ml zh x 4096 + whole), and those of K5's c twin built with its
two-choice rules in 96 of 96 (2shft and 2k6sf x 12 cells x 4 states). On en's second sight 93% of the avoidable misses
are conflicts (llama3 en 4096: 4,469 misses; every answer two-choice 614; unbounded 337). The rule built (2k6sf): a
static answer fills b1 as today; a K6 answer goes to b1 while b1's way 1 is not live, else to b2 = rotr32(h, 16) &
mask while b2's is not, else to b1; a lookup reads b1, the static table, then b2 only while b1 has spilled (bit 31 of
way 1's val word 1, which no id uses, carried by every shift fill while way 1 is live: exact). Measured (gb10c cpu 8,
the c twin on neon with neon K6 and a hardware crc, experiment only; e2e.md's states, 5 rounds, ids equal), x vs
today's c twin, llama3 / qwen38 / gpt2: en warm +12.0 / +8.0 / +6.7% (with no other text first +10.8 / +7.3 / +7.0%),
code warm about 0 (its warm misses are pieces over 15 bytes or 4 ids), ml +7.5 / +6.0 / +1.0%, zh +2.5 / +4.3 / -0.8%;
lang -0.8..-6.1% (gpt2 code x0.939, gpt2 en x0.960), pass -0.5..-2.1%. An avoided K6 run was worth 89-153 ns, and a
line touched after other text cost 3-15 ns (a lookup's read of b2, a fill's check of it). Simulated beside it: every
answer two-choice recovers 93% of en but reads 12-59% more lines in lang (measured x0.869..0.992); spilling only while
the fills since init are under 32,768 (2k6b32) keeps the en recovery and cuts lang's extra lines to +0.2..1.6% (from
+5..44%), and leaves en warm as 2k6sf has it; a length gate, a fingerprint of the last spill, blind spills and an
uncarried bit were each worse. Not taken: the en warm bar (+10% in the c twin) failed on qwen38 and gpt2 under every
rule, and the gain lives in exact replay, the segment memo's state (§7: 4.4-5.8 GB/s on these replays), for a FORMAT
change in three tiers and a fills counter. Revisit only if the memo stays off by default and the toks_par gate's
warm-lang row shows exact replay without it as a serving state to win: then 2k6b32 with its budget on K5's misses
counter (no header line), predicted en warm +11..17% (llama3 / qwen38) and +9% (gpt2) in the asm, lang within -0.5%,
pass -0.5..-1.3%.
Prefetching K5's next buckets, measured three ways and simulated a fourth, not taken (gb10c cpu 8, K5's c twin on neon
with neon K6, ids equal, 3 rounds, tools/bench/e2e.c's states plus a pass and a cold call after other text on a second
scratch). A pass after other text waits on cache and static-table lines from L3 / DRAM; prefetching piece i + D's
cache bucket and static bucket h hides K5's share of that wait. v1 and v2 keyed and hashed piece i + D ahead and piece
i again. v1 was void: with the twin's table crc the twin ran 2.1-2.4x behind k5_neon.S, the hash dominating. v2, with
a hardware crc: D = 16 gave lang code +4.6 / +7.6% (llama3 / o200k) and the pass after other text +14..17% on code and
+13..14% on en, but cold and warm code x0.868..0.902, the second hash. v3 hashed once, a ring of the next 16 pieces'
key and hash: lang code +7.8 / +12.1%, the pass after other text +18..22% on code, +18..19% on en, +5..8% on zh, but
cold and warm code x0.911..0.927, the ring and the two prefetches on short pieces that hit. v4, v3 opened only while
K5's misses among the last 64 pieces reach T, simulated (tools/bench/k5_cache.c's model with the gate): K5's counters
do not see the cpu caches, and the cpu-cache-hot cold state (a fresh scratch per call) misses the most, so the gate
opens more there than in the pass after other text at every T (code at T = 2: 63-67% of pieces against 42-46%; en
91-92% against 63-64%; zh 80-98% everywhere); warm stays closed (<= 5% at T >= 4). From v3's measured effects: T = 2
leaves cold code at x0.94..0.95 with lang code +3..5%, T = 4 x0.97..0.98 with +1..2%. Closed: the bar was no loss over
2% in any cpu-cache-hot cell with lang code +5% on llama3 and o200k, and no signal K5 has separates a cpu-cache-hot
first sight from a polluted one (a timer per round would make speed depend on timing noise). Revisit only if the speed
table's first-sight states are measured after other text: then v3 in asm (prfm pldl2keep, the ring in registers) is
the build, and its hot cost is measured there.
Runs of whitespace, measured with an oracle and not taken (gb10c cpu 8, K5's c twin on neon with neon K6, ids equal, 3
rounds). gpt2 has no merge between two spaces or between '\n' and a space, so its indentation pieces ('\n' + 7 / 11 /
15 spaces) encode to one id per byte: more than 4 ids, never cached. On gpt2 code they are 8,083 of the 11,787 K6
calls of a whole-file pass, and they miss in every state (tools/bench/k5_cache.c's model with run classes); every
other measured model has multi-space tokens, and its only run misses are 1-id space runs of 16-39 bytes that the long
probe answers in 6-7 ns. The oracle wrote a certified run (lead byte L then x repeated, L = x or '\n'; certified once
per (L, x) by K6 returning the plain bytes on an instance of 3 or more bytes, so no merge applies at any length) as
byte2id[L] then byte2id[x] repeated. At K5's miss path (K6 skipped; a shortcut inside K6 saves about the same), gpt2
code ran x1.044..1.074 in every state (about +7% cold and +8.6% warm on the asm K5), gpt2 en x0.985..0.998 (its CRLF
runs fail the shape) and the other models within noise. At K5's entry, before the key and the probes: gpt2 code
x0.851..0.947, the controls x0.669..0.865. A check on every piece costs more than any run saves, the lesson of the
prefetch ring above. Parked: static run-length entries (a words-table val 'id0 then id1 x m', so that a run is a
first-probe hit that also skips K5's miss half; predicted +11..14% cold on gpt2 code whole, gpt2 only) wait on the
long-piece cache's default, which reaches the same pieces for every model, and nobody builds them before the branch
they add to every hit of every model (n > 4: expand) is measured within 0.3% on llama3 and o200k code warm.
The dynamic cache belongs to the scratch and is valid for exactly the context the scratch is bound to: its
entries depend only on piece bytes and the model, so it stays valid across calls, modes and flags. Every entry
carries a tag (layout.h: 22 bits in its val's spare bits); K5 gets the scratch's epoch as cache_tag, and the way
whose key matches (way 0 first) answers only when its tag is cache_tag; a fill writes cache_tag. So an entry of
an earlier epoch, of whatever context, is a miss, and toks_scratch_init (the only way to bind a scratch) empties
the cache by moving to the next epoch (§7). Its size comes in the arguments: cache_mask + 1 buckets (the driver
passes the short cache's mask: TOKS_CACHE_MASK, or with TOKS_SCRATCH_CACHE_MIB(n) and a warm scratch
n / 2 MiB's, §7).
Writes: out[0, room); the driver guarantees room >= (bytes covered by the pieces) + 4, so a tier may store 4
ids at once and advance by the count.
Counters (in-out): K5 adds this call's counts to hits_static (one-byte pieces and static-table answers),
hits_cache (cache answers) and misses (K6 runs, pieces over 15 bytes included); every piece counts in exactly
one. The driver's K5 argument block lives in the scratch header, so they sum over every encode call since
toks_scratch_init (§7).


7. the driver (c, src/core) around the kernels
-----------------------------------------------

scratch: one caller buffer; its regions are 64-aligned from scr rounded up to 64, in this order:
  header  128 B: line 0 the binding (magic, the context's 64-bit identity, scr, size, max_len, region
          offsets), line 1 K5's counters (hits_static, hits_cache, misses: §6) summed since init
  ends    4 TOKS_CHUNK_PIECES B: the piece ends of one K3 round
  cache   TOKS_CACHE_BYTES = TOKS_CACHE_BUCKETS (32,768) 64-byte buckets = 2 MiB: the dynamic cache (§6; the
          sentencepiece path, spm.h, keeps its word cache there in the same entry format)
  work    align64(TOKS_BPE_WORK_BYTES(tmax)): K6's work area
  bounce  align64(4 (tmax + 4)): K5's output when cap cannot hold a round
  norm    NFC tokenizers only: align64(TOKS_NFC_X max_len), the normalized gap (norm.h)
  long    TOKS_SCRATCH_CACHE_MIB(n) only, after the short cache: the long-piece cache (n / 8 MiB of buckets, then
          3n / 8 MiB of arena; §6 "Long pieces")
  memo    right after the caches (before work): the segment memo, 4 MiB by default, m MiB with
          TOKS_SCRATCH_MEMO_MIB(m), none with m = 0 and for wordpiece and unigram (below)
  tmax = max_len, or TOKS_NFC_X (3) max_len for NFC tokenizers: the longest text K3..K6 see (one piece can grow
  that much: a run of U+0958 is one letter piece twice as long, U+1D160 x n with no pre-tokenizer one piece three
  times as long; norm.h proves the bound). init lays the regions out for max_len, the longest text the buffer
  holds (a binary search of the formula), and every call also checks len <= max_len (else TOKS_E_SCRATCH).
toks_scratch_bytes(ctx, max_len, flags) = 63 + 128 + 4 TOKS_CHUNK_PIECES + caches + memo
  + align64(256 + 32 tmax) + align64(4 (tmax + 4)) [+ align64(3 max_len)] bytes, the published formula
  (SPEC §7.2), caches = TOKS_CACHE_BYTES (2 MiB) by default and n MiB with TOKS_SCRATCH_CACHE_MIB(n) (n a
  power of two 4..128: n / 2 MiB of short cache, then the long-piece cache's n / 8 MiB of buckets and 3n / 8
  MiB of arena), memo = 4 MiB by default and m MiB with TOKS_SCRATCH_MEMO_MIB(m) (0 for wordpiece and unigram):
  with the default flags 6 MiB + 1,487 B + 36 B per byte of max_len (NFC: 111 B per byte), plus at most 189 B of
  rounding (tok v1: 8.13 MiB + ~111 B per byte); 2 MiB + 1,487 B + 36 B per byte with TOKS_SCRATCH_MEMO_MIB(0).
  The 63 bytes are the alignment slack: a buffer of exactly that size holds every region at any alignment.
the piece caches' size (TOKS_SCRATCH_CACHE_MIB): a memory budget, not a mode (identical outputs at every size).
A fresh scratch (< TOKS_K5_WARM pieces since init) probes and fills only the short cache's first 2 MiB and no
long cache, so a scratch bound per call runs as with the default; a warm one gets all of both. The sentencepiece
path runs no K5 and takes the whole caches region (short and long) as its word cache from the first call (its
misses are model calls: spm_bpe.md §6.6); init marks whose the long region is (off_long 0: spm buckets) and a move
between an spm context and a byte-level one is a first init. Measured end to end (tools/bench/e2e.c, 4 KiB chunks,
llama3 / gpt2 / GLM 5.3 x en / code / ml on M2 and Zen 5): the same text again gains +26-43% (en, code) and +83-154%
(ml) at 32 MiB over the default,
ahead of gigatoken's warm state on 16 of 18 cells (gpt2 code and ml on a loaded Zen 5 trail by 3-6%); warmed on
other text -9..+13%; a pass from a fresh init -14..+9% (English loses most: its fills spread over 16 MiB instead
of 2); a call on a fresh scratch -3..0%. 128 MiB gains nothing more on these corpora (~2-6 MB each) and costs TLB
misses without huge pages. Hosts that keep a scratch per thread (servers, toks_par) take 32 MiB. A first init of
such a scratch advises huge pages for its caches' 2 MiB-aligned interior before it zeroes them
(toks_plat_hint_huge: madvise MADV_HUGEPAGE on linux, nothing elsewhere), so a malloc'd scratch gets what toks_par's
toks_plat_arena block gets: tr9970x Zen 5, 32 MiB, 4 KiB chunks, llama3 / gpt2 x en / code / ml / zh: warm +2-10%, a
pass +0-5%, cold unchanged (the same as E2E_HUGE=1's arena).
The default scratch (flags 0: one 2 MiB cache, not 2 MiB-aligned inside the caller's buffer) gets no hint and stays
on 4 KiB pages. Measured and not taken (measured 2026-10-04 on the 9970X, cpu 20, 5 abba rounds, thp_kb read from the run):
the whole scratch huge-paged (E2E_HUGE=1 arena, written before init) runs gpt2 en 4096 warm 708.8 -> 722.9 MB/s
(+2.0%), pass +1.3%, cold flat; llama3 en 4096 warm +2.2%, pass +0.9%; gpt2 en whole pass +1.5%, warm +1.9%; gpt2
code whole cold -1.8%, warm +0.8%. The cache's 512 pages miss the 96-entry L1 DTLB but hit the 4096-entry L2 one
(no walk), and the probe's latency is hidden (§6), so it stays under the 3% that would pay for 2 MiB of alignment
slack in the formula.
The tables' arena (toks_plat_arena) is 2 MiB-aligned and advised too, but its tail past the last whole 2 MiB stays on
4 KiB pages, and words, built last, ends there (llama 3: 0.07 MiB of its 4, gpt2 0.6 of 2, qwen 3.8 1.0 of 8, o200k
1.0 of 8). Rounding the arena up to whole 2 MiB measured 0.99-1.01x on K5 cold and warm (GB10 X925 cpu 7,
bench_neon, llama 3 / qwen 3.8 / gpt2 / o200k en, qwen 3.8 code), and words placed 512 KiB later 1.00x: not taken.
End to end (e2e_commits, e2e.md's states, GB10 X925 cpu 7, on a tree whose smaller vhash leaves gpt2 a
3.8 MiB arena with 1.6 of its 2 MiB words on 4 KiB pages): whole huge pages gpt2 +1.0%, but llama 3 pass / lang-x
en -1.5%, code -2.3%, qwen 3.8 cjk -1.4%; words allocated first (all of it on huge pages for every tokenizer) gpt2
-1.4..-2.5%, the rest -0.3..-1.2%. Which 4 KiB page holds what moves single cells by 1-2% either way; neither taken.
the cache's size (tools/bench/k5_sweep.c: K5 alone, gpt2 / llama3 / GLM 5.3 x English / code / multilingual,
64 KiB .. 16 MiB, on the three cores of SPEC §11's hosts, re-measured with the warm fill): a pass from an empty cache
stops gaining at 1-2 MiB on M2, Zen 5 and X925 alike; the same text again gains +20-30% more at 8 MiB than at 2 MiB
(the warm fill holds every distinct piece), warm-lang 1-4%.
2 MiB holds the distinct pieces of ~1 MB of English with room; it is the X925's private L2 and an M2 P-core's
share of its cluster's 16 MiB L2, and it spills Zen 5's 1 MiB L2 into the 32 MiB L3 of its CCD without a
measured loss. A first init zeroes it in ~10 us, a re-init only moves the epoch; 64 threads hold 128 MiB of
caches. A scratch bound per call starts every call with an empty cache: bind once per thread (toks_par does).
toks_scratch_init writes the header and zeroes the counters, and empties the dynamic cache in O(1) when the buffer
already holds a scratch toks laid out with the same flags (its header's magic, address, size, cache offset and
cache size name this buffer, so
the cache region holds only entries K5 wrote, tagged with epochs up to the header's): the header moves to the next
epoch, 1..2^22 - 1, and K5 takes no entry of an earlier one (§6); the long-piece cache likewise moves to its next
generation (its arena restarts empty). Any other memory, and a scratch at its last epoch or generation, gets the
caches zeroed (the long cache's buckets only: its arena is read only through slots), epoch 1 and generation 1
(~10 us for 2 MiB). So binding a scratch per call costs nothing
measurable after its first init; the caller must not write a scratch's memory between toks calls (the header
then still names the buffer while the cache region holds the caller's bytes). Every call checks the header
against ctx (TOKS_E_SCRATCH on any mismatch), and every cache entry against the epoch, so a cache filled under
one tokenizer is never read under another. The header comes first: a call reads only its line 0 (the 64 bytes at the first 64-aligned address)
before the binding is proven, so a buffer init never bound is refused without a read past its end (any
buffer of at least 127 bytes).
the segment memo (api.c memo_*, run_seg; SPEC §6), on by default (decided 2026-10-05: flags 0 gives 4 MiB; the
  receipts behind that call and the first-sight price it accepts are in docs/usage.md "Scratch"; measurements
  here from before it call the memo-off scratch flags 0), m MiB with TOKS_SCRATCH_MEMO_MIB(m), none with m = 0 (the
  macro sets bit 20, TOKS_SCRATCH_MEMO_SET, so a set size of 0 differs from flags without one, and a 0.2.0 binary's
  m > 0, compiled without the bit, keeps its meaning); wordpiece and unigram, which have no memo path, get no
  region. A 64-byte header (core.h toks_memo_head:
  write position, hits, drought, probes, differ, vpos, lap, run), m x 512 sets of two 32-byte slots (one 64-byte line)
  { hash, ring position, key, ids, length, epoch } and a ring of 64-aligned records { a 32-byte head (a slot's
  fields), the bytes padded to 8, the ids }. A segment is a phase-0 unit (a run between recognized added tokens that
  are not normalized; the whole text in NONE, for a tokenizer without added tokens, or a tiktoken wrapper's specials
  run) of >= 256 bytes in an encode call; pieces never use the memo, nor do wordpiece and unigram. Its key: the call's
  flags, the document state (the unit starts the input of a call without TOKS_CONTINUATION), the path (run_cuts or
  run_gap), the length, and a 64-bit hash of the length and three 64-byte windows of the bytes (the start, the middle,
  the end: eight multiply chains, O(1) per segment) keyed by the context's identity, fixed at toks_load, so a text's
  segments meet the same slots in every run (never the scratch's address: ASLR moved it, and with it which segments
  collided, so 3 of 50 identical runs lost 2 of 29 replays); the hash's high half scaled to the set count picks the
  set, whose line a probe reads whole. Two ways, because a direct-mapped index loses replays at the birthday rate:
  simulated on the e2e segment streams and the chat server before it was built (16 seeds), 29 segments seen twice lose
  a replay in 10.3% of runs with one way (measured: 3 of 50) and 0.15% with two; llama3 en 4096's second sight answers
  466 -> 490 of 497 segments, zh's 385 -> 402 of 683 (the lap binds there), and the chat server at 64 conversations
  60.8 -> 70.0% of its bytes. A mark takes its own slot, else an empty or dead one (another epoch's, a lapped record),
  else the older mark, never a live record; a record takes its own slot, else an empty or dead one, else the older
  mark, else the older live record. The hash only locates: segments that share the windows and the length cost a byte
  compare (counted: differ), never ids. A hit needs a slot of the set with the epoch (init moves it: rebinding to any
  context empties the memo in O(1)), hash, key and length, a record the write position has not lapped (pos - slot.pos
  <= ring) and every byte equal to the ring's copy; its ids are then copied out (what cap holds) and counted. A miss
  encodes the unit as without the memo and, when all its ids are in out (n <= cap at its end), records it or marks its
  slot (hash, key, length, epoch, ids = 2^32 - 1: never an answer; position = vpos). Admission: after a whole ring of
  record bytes written without a hit, a first sight only marks and a segment is recorded on its second sight, until
  the next hit opens admission again. Writing records is the memo's cost on text that never comes back, and it is paid
  twice: 2 bytes stored per byte encoded in the pass that writes them (gb10c, e2e.md's states, the first ring after
  init: pass x0.877..0.969 at 4096, x0.928..0.990 whole), and their write-backs in the next timed pass, even after an
  init and MBs of other text (lang after a warm pass that recorded 4 MB: x0.963..0.985 of the same memo recording
  nothing; after 1 MB: x0.99..1.00). Neither payment moves with the store type: stnp of whole lines and dc zva before
  plain stores were measured and not taken (recording pass x0.985..0.990 and next pass x0.986..1.012 against plain
  stores; gb10c cpu 8, an A/B on the admission change). So a stream without replays stops paying after one ring; a
  conversation is answered from its second prompt. Closing admission at start (record only second sights) was measured
  and not taken: pass flat (x0.962..1.006) but nothing answered on the second sight, which is T8's warm cell (qwen38
  en 4096 vs tok v1: x1.07 -> x0.20). Skipping the memo in the first call after init (so a call on a fresh scratch,
  and e2e's cold state, run exactly as without a memo) was measured and not taken: cold x0.989..1.008 against the
  recording memo's x0.951..0.989, but pass unchanged (x0.859..0.938 at 4096: the calls after the first record the ring
  anyway) and a second call can no longer replay the first (llama3 code whole, e2e's warm = the second call: 11,176
  -> 711 MB/s; gb10c cpu 8, master d11f9d1, 3 palindromic rounds). Probing and marking cost nothing measurable: a
  memo that marks every first sight
  and records nothing runs lang at x0.987..1.005 of no memo (the 2-3% once blamed on the marks was the previous pass's
  record write-backs: one binary, the marks kept and the records dropped, removes it; moving the slot array to its own
  2 MiB-aligned region or writing a one-byte mark only looked cheaper because each recorded less). The ring is written
  in laps from its start, and a full lap keeps what it holds: a record that does not fit is refused (its slot marked)
  and adds 1 to the lap's run; a refused second sight that a lapping ring would still hold (vpos, the position a ring
  that never stopped would be at, at most a ring past its mark's) adds TOKS_MEMO_DRY / TOKS_MEMO_GHOSTS = 512; a hit
  ends the run, and at TOKS_MEMO_DRY = 1024 the next lap starts (a head of length 0 skips the old lap's tail; core.h).
  A mark never replaces another segment's live record. Text whose records pass the ring therefore keeps the lap it
  filled and is answered alike on every replay, where a lapping ring alternates (simulated on the e2e chunks, a model
  of the ring, 4 MiB: qwen38 zh 4096, 6.13 MB of records, sights 3 / 4 answer 447 / 447 of 683 segments, a lapping
  ring 462 / 0); a chat server's ring keeps lapping (synthetic, a system prompt and the conversation so far per
  request: 16 / 64 / 256 conversations at once, 88.7 / 60.5 / 24.1% of segment bytes answered, a lapping ring 88.8 /
  61.0 / 24.1, 1-23% fewer record bytes); and a switch to other text starts a lap after two second sights (zh x3 then
  en x3: en's third sight 467 hits) or, for text past the ring, after 1024 refused records (en x3 then zh x3: zh's
  third 327). Measured and not taken for the ring: lapping as before (the alternation above); the dry rule alone (a
  lap starts after 1024 refused records only: a chat server's system prompt hits every request, so the ring never laps
  again, 7.7% answered at 16 conversations); two half-ring generations (85 / 41 / 16% on the chat server, 0 on the
  fourth sight); ghosts alone (no backstop: a switch to text past the ring stays on the old lap, zh's third sight 1
  hit). A call that returns n <= cap publishes its records at its end (SPEC §4.4, design.md d3): it walks them from
  its start position and writes each one's slot, the length last; a call whose records lapped each other publishes
  none (a slot then never points at a head the call overwrote). A new lap overwrites the oldest records and a newer
  record takes an older one's slot: speed only. A tokenizer whose added tokens are normalized (phase 1) has them
  inside its units, so it memoizes whole units: exact, fewer hits. Tests: tests/c/test_memo.inc, built twice:
  test_memo on the library and test_memo_hash with api.c compiled in and one hash for every segment (SPEC §6's
  degenerate-hash build), 11 tokenizers; every encode equals a memo-off scratch's: conversations, eviction, a full lap
  replayed (with refused first sights between), second sights and refused records starting laps, a call lapping the
  ring, changed bytes at one address, identical bytes at another, small caps, truncation, modes, flags, document
  state, rebinding, migration across scratches; a call returning n > cap publishes nothing. Mutants (removed byte
  compare, epoch, key, length, lap check, the per-unit cap check, the call's cap check; the full lap never kept, a
  second sight weighing 1, no backstop, a hit not ending the run, no lapping-ring check on a mark, a mark over a live
  record) all fail a test except the publish walk's lap check, whose failure needs a hash collision crafted against
  the hash (kept: it is what makes a slot never name a head that was overwritten).
  Measured and not taken (decision 18, #17, shelved 2026-10-06): a record that keeps a 16-byte keyed check of its
  bytes instead of the bytes (CLNH in two Toeplitz passes over 4 KiB blocks, then a polynomial modulo 2^127 - 1 over
  the sums and the length, keyed by a secret the context draws from the os at load: 2^-121 for two different 4 KiB
  segments chosen without the key), so that a record takes 1-2 bytes per byte of text instead of 2-3 and the ring
  holds about twice the text. The floor is the core's: on the X925 the check of a 4 KiB segment in L1 costs 88.2 ns
  and memcmp of the bytes 39.0..39.1 (gb10c cpu 8, three processes of tools/bench/check_bench.c, which carries that
  check and its known answers: docs/bench/raw/memo-check-gb10c-check.log), so every replay whose record sits in L1 /
  L2 pays twice the compare; through records in L2 / L3 (2 MB of text) the X925's check with the id copy wins,
  157.6..158.1 ns a segment against 178.4..181.8. Master 361883a -> #17's e1d4296 (gb10c cpu 8, e2e_commits.sh, 3
  abba rounds, ids equal; memo-check-gb10c-e1d4296-4096.log, -whole.log): warm code at 4 KiB x0.891 (llama 3) and
  x0.863 (qwen 3.8), against warm en x1.13..2.78, ml x1.19..3.25, cjk x1.17..24.6 and warmo x1.17..22.7 where the
  ring was the limit; tok v1's conversation replays x0.874 (qwen 3.8), x1.003 (glm 5.3), x0.885 (nemotron 3 omni) and
  code 4096 warm x0.881..0.917 (tokv1.sh and tokv1_ab.py, abba: tokv1-memo-check-gb10c-*.log). Counted without the
  bytes, a whole-text en segment's 2 MB record fit half the ring and its cpu-cache-hot first sight paid for it (whole
  cold en x0.861 / x0.898, llama 3 / o200k: memo-check-gb10c-2ac8756-whole.log); admission counting the bytes brought
  it back to x0.996, and deferring a record over an eighth of the ring to its second sight moved the write into the
  warm pass instead (whole warm code x0.049..0.058: memo-check-gb10c-defer8-whole.log).
  On Zen 5 (tr9970x cpu 26, untimed, check_bench on #17's library, 4096-byte segments) the check is 247 ns for 4 KiB
  with PCLMULQDQ (16.5 GB/s) and 79.5 ns with VPCLMULQDQ on zmm, four CLNH groups an instruction (51.5 GB/s; branch
  toks/x86-memo-zmm), against memcmp's 34 ns (120 GB/s); through records in L2 / L3 the check with the id copy takes
  288-293 ns a segment with PCLMULQDQ and 124 with zmm against the compare's 111-121: a keyed 128-bit check at 51 GB/s
  cannot beat memcmp at 120 GB/s on L1-resident records.
Dropped bytes (api.c run_drop): a K3 round holding a byte the vocab lacks (ctx->has_drop; config.c). hf drops
such a byte inside the model, per word (merge_word), so a piece holding one is encoded with those bytes removed,
and a piece of them alone emits nothing. Runs of clean pieces go through K5 as usual; a dirty piece is compacted
into the scratch's norm region -- the one copy, only for these pieces -- or in place when the segment already lies
there (NFC wrote it; compaction only rewrites the piece's own bytes, forward). The compacted piece is one K5 piece:
K5's answer for its bytes is their bpe (ignore_merges is refused with dropped bytes), so caching it is right; an
empty one never reaches K5.
NFC gaps (api.c run_gap): text is materialized only for the spans NFC changes (maintainer doctrine, speed). An already-NFC
gap is scanned in place, no byte copied; otherwise the plan's stretches (norm.h; cut at K3 restart points, proof in
norm.c) are normalized into the norm region and scanned there, everything between them in place. Phase-1 tokens
(normalized: true) may match across any point, so with them a changed gap is one stretch. The norm region holds
TOKS_NORM_BOUND(f, max_len) bytes (f the form: 3 or 11 per byte), the work area and bounce its tmax.
the arena (src/platform/mem.c): memory toks allocates itself -- the tables at load, toks_par's scratches -- comes
zeroed from toks_plat_arena at a 2 MiB-aligned address, so 2 MiB of tables can sit in one huge page.
  huge pages (linux): madvise(MADV_HUGEPAGE) BEFORE the first touch (gigatoken measured ~15% cold / ~7% warm lost
  when the advice came after the zeroing memset: zen drops software prefetches that miss the dtlb), and the first
  touch of each 2 MiB frame a WRITE. On a THP-eligible mapping (enabled=always, or madvise with this advice) a read
  fault maps the huge zero page when use_zero_page is 1 (else it allocates a huge page itself), and what the first
  write then does depends on the kernel (mm/huge_memory.c, do_huge_pmd_wp_page): up to 5.7, and again from 6.13
  (do_huge_zero_wp_pmd), it allocates a huge page; from 5.8 through 6.12 it splits the frame into 4 KiB pages
  (goto fallback), which only khugepaged can collapse back later (max_ptes_none permitting, 511 by default; its scan
  runs every 10 s by default). Measured with test_load's arena check (a 4 MiB arena read first, then written, three
  runs a side; both hosts at enabled=madvise, defrag=madvise, use_zero_page=1, max_ptes_none=511): tr9970x, linux
  6.8, 2048 of 4096 kB huge against master's library and 4096 with this write; gb10e, linux 6.17, 4096 either way
  (docs/bench/raw/par-tr9970x-arena-teeth.log, par-gb10e-arena-teeth.log). toks_plat_arena writes its first byte
  right after the advice, so toks_par's scratches, which toks_scratch_init reads before it writes (the binding
  check), keep their first frame huge whichever way the kernel takes that write: +2048 kB a scratch on tr9970x, its
  first pass on a fresh pool +6..19% (median of 10; docs/bench/raw/par-tr9970x-46a410a-arena-first-write-c4096.log,
  tools/bench/par_warm_ab.sh).
  The write also puts that frame on the allocating thread's numa node (every receipt host has one node). a caller's
  own madvised scratch must be written (zeroed) once before its first toks_scratch_init the same way, or on a
  splitting kernel the frame holding the header stays small.
  posix: one private anonymous mmap of n rounded up to whole pages plus 2 MiB of slack, then the slack's head
  and tail unmapped. munmap takes whole pages: trimmed at n instead, the tail started inside a page, munmap
  failed, and up to 2 MiB per arena stayed mapped (+3.7 GiB over 2000 loads on the mac; tests/c/test_load.c).
  windows: VirtualAlloc2 with a MEM_ADDRESS_REQUIREMENTS alignment of 2 MiB (windows 10 1803+; looked up in
  kernelbase.dll at run time, so toks still loads where it is missing); without it, a reservation of n + 2 MiB
  of which only the aligned n bytes are committed (the slack is address space, never memory). Plain VirtualAlloc
  aligns to 64 KiB only. toks_plat_arena_free releases the whole reservation from its base (VirtualQuery's
  AllocationBase), which covers both. Large pages need SeLockMemoryPrivilege on windows: not used.

toks_encode(ctx, text, len, flags, out, cap, scr):
  checks, in this order: ctx, unknown flag bits, NULL only with 0 (TOKS_E_ARG); len <= 2^29 (TOKS_E_LIMIT);
  scratch bound to ctx and sized for len (TOKS_E_SCRATCH).
  1. prefix ids of the post-processor (unless TOKS_NO_POSTPROCESS).
  2. segments: mode NONE -> the whole text; else phase-0 matches (K1 + policy) split the text into gaps and
     tokens; each gap is normalized (when the tokenizer has a normalizer) and split by phase-1 matches.
  3. each text segment: pos = 0; while pos < seg_len: K3 (cap TOKS_CHUNK_PIECES) then K5 over those pieces.
     K5 writes straight into out when cap leaves room (bytes of the chunk + 4); otherwise into a bounce buffer
     in scratch, from which the part that still fits is copied; counting continues past cap.
  4. each token: its id. 5. suffix ids.
  returns the total; out[0, min(cap, n)) exact.
Dispatch is a switch on ctx's tier at each call site, compiled to direct calls (no function pointers); K3 has
one call site per template (kernels.h toks_k3 = K3_CL100K, toks_k3_o200k = K3_O200K, each through TOKS_RUN), and
run_text picks by t->tmpl (TOKS_TMPL_NONE runs no K3: the segment is one piece).
load: toks_load refuses (TOKS_E_UNSUPPORTED, named in diag) tables whose template invariants fail
(compile.h toks_tmpl_invalid: parameters each template defines, class tables present and in range, the Han
bit iff TOKS_TP_HAN); the .toks image loader applies the same check to every image (SPEC §8.3).
The segment memo (SPEC §6) and split planning (§5) sit in this layer later.


8. K7 the spm scan (sentencepiece-style bpe: gemma 1-4, mistral, phi-3, yi, llama-2 style files)
--------------------------------------------------------------------------------------------------

Contract (layout.h K7_*, spm.h): uint64_t toks_k7_spm_<tier>(s, a) returns the word starts of ONE window of the
unit a->text[0, len) from a->pos (a char start) as a mask, bit j = a word starts at pos + j (spm_bpe.md §5.5: a
certified cut, or a Metaspace split, between the char before and the char at pos + j); a->pos becomes the next
window's first char, a->prev the entry of the window's last char (the one before the next window), a->mb nonzero
when the window holds a 2-4 byte char, and one[j] = id | length << 24 for each of them (the flush's one-char word
shortcut). The unit's first char gets a bit like any other (toks_spm_encode drops it without a prefix symbol).
Windows are the kernel's own business: the twin's are <= 57 bytes plus a char, the asm parts' 64-byte blocks
minus a char that runs past the block; toks_spm_encode flushes words window by window, so only the union of the
word starts must agree (test_k7.c checks it position by position, and one[] at every 2-4 byte char). An asm part
leaves a window to the twin by returning with pos unchanged: pos 0, fewer than 64 bytes left, no certified cuts
(a->flags bit 0: no pair set or TOKS_SPM_NOCUTS), or invalid utf-8 in the block, so the parts see valid utf-8
only and need no state for it.

The c twin (spm_c.c toks_k7_spm_c, always inline in toks_spm_encode): 8 ascii bytes per step through cut_ab
after an ascii byte, every other char by the per-char rule (entry_at + toks_spm_cut_between), as first written.

The asm parts (k7_spm_neon.S, k7_spm_avx2.S), one 64-byte block:
  1. classify: non-ascii, continuation, 3-4 and 4 byte lead masks (neon: ld4 + compares + sri / shrn movemask;
     avx2: vpmovmskb of signed compares and vpmaxub). The leads' expected continuations (L << 1 | L34 << 2 |
     L4 << 3) must equal the continuation mask; a lead in the last 1-3 bytes whose char runs past the block ends
     the window there.
  2. an ascii char after an ascii byte: bit p[j - 1] | p[j] << 8 of cut_ab (both & 0x7f, so a non-ascii neighbour
     reads inside the table): straight-line over the 64 positions when >= 24 are such pairs (neon: four
     accumulators so no bfi chain runs through one register; avx2: bt + adc from position 63 down), else one per
     set bit.
  3. every other char, one per set bit: its entry (ascii[], or the stage tables at cp >> 8 / cp & 0xff computed
     from the bytes by length -- neon: bitfield moves; avx2: bswap + pext), rejecting overlongs (C0 C1, E0 80-9F,
     F0 80-8F), surrogates and > U+10FFFF (the twin then takes the window); the previous char's entry (ascii[] of
     the byte before, else the last entry); then the pair rule:
     - both entries PLAIN (spm.h TOKS_SPM_E_PLAIN, bit 31, set in the char-table fold: an id, and not the Metaspace
       split image): the pair set alone decides, a cut iff id_x << 21 | id_y is not in it. That equals
       toks_spm_cut_between: the small-alphabet bitmap is built from the same set (a bit is a cut iff the pair is
       not in it), and a left char that is not PAIRED is the left of no key, so its probe misses. The probe reads
       slots i, i + 1 at once (spm_build.c stores slot 0 again after the table: no wrap; neon one ldp, avx2 one
       16-byte load): spanned iff either slot holds the key (a key sits at its home or later, before any empty
       slot, so an empty slot i leaves no key at i + 1); decided iff either slot holds the key or is empty, else a
       loop from slot i + 2 (rare: the set is built at load <= 1/4, gemma 4 0.13, 2 MiB);
     - else the general rule: the Metaspace split image is a cut; an id-less char is none; two small-alphabet chars
       read the bitmap; else the pair set as above.

Measured (scanner run 5, gemma 4, master aa46e18; tools/bench/spm_stages.sh before, tests/k7/bench_k7.c for the
kernel alone, an e2e A B B A for the rest). Before K7, the scan was 48-79% of gemma 4's zh time, 10-23%
of ml, 4-21% of en / code; the twin's per-char step cost ~9 ns (X925) / ~12 ns (Zen 5), mostly mispredicted
data-dependent branches on PAIRED and the pair probe. K7 alone vs the twin (4 KiB units): neon gb10c X925 en 1.39x,
code 1.24x, de 1.47x, zh 1.24-1.29x, ja 1.41x, ko 1.48x, ru / ar / th / hi 1.21x; avx2 tr9970x Zen 5 en 2.03x, code 1.97x,
de 1.95x, zh 1.57x, ja 1.89x, ko 1.80x, ar 1.39x, hi 1.34x, ru 1.30x.
Left on the table (measured): the pair probe is half of the asm's zh time on the X925 (gut-zh 1.05 of 2.16 ns/B);
a pair set at load 0.13 (2 MiB) instead of 0.26 makes both the twin and the part ~10-13% faster on zh (fewer
second-slot probes) while one at 0.52 makes both 1.4-1.6x slower; a per-length decode is ~1/3 of the rest.

Scanner run 6 (the PLAIN rule, the mirror slot, the pair set at load <= 1/4): the per-char step of a 3-byte char
went from 72 to 50 instructions on neon (avx2 ~88 to ~72): no PAIRED / small / split / id tests and no wrap check
on the common path, one load of both slots, the decided test as a ccmn chain. K7 alone (gemma 4, 4 KiB units,
ABBA, master fc5808a -> k7-cjk): neon gb10c X925 zh 484 -> 762 MB/s (1.58x), ja 610 -> 919, ko 560 -> 811, ru 505
-> 688, hi 605 -> 865, en +3%, code 0; avx2 tr9970x Zen 5 zh 453 -> 675 (1.49x), ja 524 -> 722, ko 473 -> 655, ru 454 ->
584, hi 549 -> 745, en +2%, code -1%. Ids unchanged (e2e sha equal on zh ja ko hi ru en code, both isas).
