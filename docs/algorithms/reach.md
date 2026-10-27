Structural control-token isolation (SPEC §3.6)
===============================================

src/core/reach.c decides at load, per tokenizer file, whether any added token's id can come out of
`toks_encode(..., TOKS_ADDED_NONE)` for any input bytes. toks_info.control_isolation is 1 when it proves that none
can, else 0. A consumer that places control tokens by position needs this. It encodes client text with no added-token
recognition and puts the template's tokens in by id, as DeepSeek's API encoder does. Without the
certificate, plain text could still spell a control id through the model itself.

  §1  the claim                §3  the piece test (gen.c toks_gen_piece)
  §2  the proof, per added id  §4  the census


1. the claim
-------------

R is the ids of every added token in the file, special or not: harmony-format tokens and most families' `<think>` /
`<tool_call>` are often special=false, and they are what a prompt template places. For all byte strings x, no id of R is
in the NONE encode of x. The encode runs the normalizer, the pre-tokenizer chain and the model, and nothing else
emits an id.

Analyzed: byte-level bpe, including the tiktoken readers. Sentencepiece-style bpe, unigram and wordpiece report 0, not
analyzed: their models emit an unk or byte-fallback piece, and their vocabularies usually hold the added tokens.
A 0 means "not proven", never "proven reachable".


2. the proof, per added id
---------------------------

Byte-level bpe emits only (a) the model's vocabulary ids, each an initial symbol (one byte's alphabet char) or a
merge's product, (b) under `ignore_merges` (or tiktoken's ranks, `ids_as_rank`) the id whose string is a whole piece,
and (c) the unk_token for a byte whose char the vocabulary lacks (config.c `drop_unk`). An id a of R is certified
unreachable when every one of these is excluded:

  - a is the dropped bytes' unk: reachable (input bytes are any bytes; gx-spark-x25 lacks 0xC0, 0xC1, 0xF5..0xFF).
  - a is not a vocabulary id (a >= n_vocab, or a tiktoken rank gap): the model never emits it.
  - a's string is not over the byte-level alphabet (decode-only, docs/breadth.md §1.4): bpe never forms it.
  - its string is one byte: an initial symbol, reachable. A merge makes it: reachable. Whether that merge can
    actually fire is not analyzed, which is sound and conservative.
  - otherwise, only (b) can emit it, so with neither ignore_merges nor ids_as_rank it is unreachable. With either,
    the question is whether its bytes b can be a final piece of the pre-tokenizer chain. A vocabulary with dropped
    bytes answers yes (run_text removes them from inside a piece). Otherwise §3 decides.

Normalization only narrows the texts that reach the pre-tokenizer, so a proof over all texts covers it.


3. the piece test
------------------

config.c records the chain's last splitting step (config.h `reach_*`). Every final piece is an element of that step,
run over some text. Steps after it are ByteLevel without a regex, which splits nothing. If no step splits, every text
is one piece and the answer is yes. If the last split is anything other than an Isolated, uninverted Split
(Regex or String) or ByteLevel's gpt-2 regex, there is no proof and the answer is yes. Otherwise reach.c compiles that
one step with gen.c's compiler, the template patterns included: their regex is the exact string. A ByteLevel before
the step puts it in the byte domain. gen.c's toks_gen_piece then answers for b:

  - a match: a piece equal to b needs a path of the step's program through exactly b's atoms (the VM's own atoms, an
    ill-formed byte as 0x110000 + byte). Assertions and lookaheads are relaxed: they may pass, since the text around
    the piece is anyone's. If no G_MATCH is reachable at b's end, b is no match in any context.
  - a gap: a gap holds no position where a match starts (the Pike VM adds a start thread at every position until it
    finds one, generic.md §5). Take a strict path from the start (no assertion, no lookahead) through one char c to
    G_MATCH. A match then starts at any position whose atom is c, whatever surrounds it. If b holds such an atom that
    needs no context (an ascii byte, or any byte in the byte domain), b is no gap.

Neither case means b is never a piece, so a is unreachable. Example: Nemotron's `<SPECIAL_12>` and `[INST]` under the
o200k nemo regex. `[^\r\n\p{L}\p{N}]?[letters]+` stops at `_` and `]`, and ` ?[^\s\p{L}\p{N}]+[\r\n/]*` takes no
letter, so no match spans the whole token. The second alternative matches `<` and `[` alone, strictly, so no gap holds
them either.


4. the census
--------------

The tokenizer cache, 2026-10-27 (tests/c/test_reach.c pins the critical targets and gx-spark-x25):

  certified (1)   every byte-level bpe file but one: gpt-2, llama 3 / 3.1 / 3.2 / 4, GLM-5.3, Qwen 3 / 3.8,
                  DeepSeek v3 / r1 / v3.1 / v3.2 / v4 / v4.1, o200k / gpt-oss, Mistral-Nemo, Nemotron 3 (4b, nano,
                  omni), MiniMax text-01 / m1 / m2 / m3, ModernBERT, bloom, falcon, the generic-engine rows.
  not certified   gx-spark-x25: its added `<unk>` is the unk of the bytes its vocabulary lacks, so `toks_encode("\xFF")`
                  gives it (test_reach.c). Not analyzed: spm (llama 1 / 2, Mistral v0.1 to v0.3, gemma 1 to 4, phi-3,
                  Yi, codellama), unigram, wordpiece.

The red proofs (tests/c/test_reach.c) are synthetic files where an added id is reachable: a merge's product, a
one-byte token, a whole gpt-2 match under ignore_merges, any text under a ByteLevel without a regex, a gap of a
letters-only split, and a dropped byte's unk. Each must be uncertified, and a concrete text's NONE encode must give
the id. Mutants of each branch above (the drop unk, the byte and merge test, the match test, the gap test, the piece
test as a whole) each fail at least one of them. Every certified file is also searched: each added token spelled in 15
contexts, plus 2000 random byte strings, with no added id in a NONE encode.
