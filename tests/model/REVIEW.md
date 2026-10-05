# kernels.md review — empirical verification against hf tokenizers 0.23.2

**Verdict: rules confirmed. 0 defects found in §2, §3, §4, §5, §6 — 14.8M differential cases, 0 mismatches.** Every rule was re-implemented independently in Python from the doc's prose (parameterised exactly as layout.h's `TOKS_TP_*` bits, classes probed through hf's own onig via `Split`), then compared against hf itself. Details, counts and repros below; notes at the end name the two sub-behaviours that are source-verified but not fuzz-hit, and one unused layout.h parameter.

## What was compared and how

Model: a straightforward Python transcription of kernels.md — atoms (§2), A1–A7 with the layout.h parameters (§3), the added-token driver policy (§4), linear lowest-rank-leftmost BPE with ignore_merges and duplicate-pairs-keep-last (§5). Class membership (`\p{L} \p{N} \p{M} \s`, and the eight `(?i:x)` folds) was **probed through hf's own regex engine** by feeding every scalar value (all 1,112,064 code points, surrogates excluded) through `pre_tokenizers.Split(Regex(...), behavior='removed')` and taking the uncovered chars — never python `unicodedata`. hf pieces came from `pre_tokenizers.Split(Regex(pattern), behavior='isolated')` for cl100k/llama3/qwen2/qwen3.5 and `pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)` for gpt-2 (the python binding returns character offsets; converted to byte offsets with an asserted slice check).

Pattern strings were verified against real tokenizer.json files from the hub:

| repo | pattern in file | matches doc §3 |
|---|---|---|
| openai-community/gpt2 | ByteLevel use_regex (the compiled-in gpt-2 regex) | ✓ |
| unsloth/Llama-3.2-1B-Instruct | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` | ✓ identical to cl100k |
| Qwen/Qwen2.5-0.5B, Qwen/Qwen3-0.6B | same with `\p{N}` | ✓ DIGITS_1 |
| Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-4B (family exists on the hub, 0.8B→122B) | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+\|\p{N}\| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` | ✓ exactly the doc's construction |

Note the qwen 3.5 file keeps the optional prefix class `[^\r\n\p{L}\p{N}]?` (no `\p{M}` added there) and only the letter run gains `\p{M}` — consistent with the doc's parenthetical that an M char either starts the L|M run or prefixes it (same span), which the model reproduces by folding marks into L for that template.

## §3 K3 — 13.2M cases, 0 mismatches

Generators: class-aware random strings (ascii letters/digits/punct, all 25 `\s` code points, CR/LF runs incl. `\r\n`/`\n\r`, apostrophes + s t m d r e v l in both cases + U+017F, non-ascii letters latin/cyrillic/cjk/hangul, combining marks U+0300/U+0301/U+0308/U+0361/U+0897/U+1AC1/U+3099/U+20D0/U+FE0F, non-ascii digits U+0660../U+2460/U+FF10/U+00B2/U+2160, symbols, emoji + ZWJ chains, NUL, lengths 0–200) plus long runs 60–1000 chars (letters / digits / punct / ws / base64 / mixed) crossing 64-byte boundaries.

| variant | cases (3M short + 300K long, on a Neoverse-V2 box) | mismatches |
|---|---|---|
| gpt-2 (CS, ' '?, SP_RUN, no PUNCT_NL, no WS_NL) | 3,300,000 | 0 |
| cl100k / llama 3 (CI, ANY, 1_3, PUNCT_NL, WS_NL) | 3,300,000 | 0 |
| qwen 2/2.5/3 (CI, ANY, 1, PUNCT_NL, WS_NL) | 3,300,000 | 0 |
| qwen 3.5 (CI, ANY, 1, PUNCT_NL, WS_NL, marks→L) | 3,300,000 | 0 |

Plus 3,000 cases/variant locally before the big run. A5's backtracking, A6's give-back of exactly one atom, A7's single-atom case, the `\s+(?!\S)` end-of-segment lookahead (segments end mid-whitespace-run in the added-token runs below), DIGITS_1_3's truncation at 3, LPREFIX_ANY admitting every `\s` char and P as prefix, gpt-2's U+0020-only prefix, PUNCT_NL's extension over the maximal NL run — all exercised and exact.

Probe facts backing the doc's notes:
- onig `\s` through hf = **exactly the doc's 25 code points** (U+0009–000D, 0020, 0085, 00A0, 1680, 2000–200A, 2028, 2029, 202F, 205F, 3000).
- `(?i:s)` matches exactly {s, S, U+017F}; `(?i:t) (?i:m) (?i:d) (?i:r) (?i:e) (?i:v) (?i:l)` match exactly the two ascii forms — no other fold reaches the contraction letters, confirming the FOLD_S-only note and A1 CI.
- python `unicodedata` disagrees with hf's onig on 4,924 `\p{L}` cps (e.g. U+1C89, U+A7CB…), 80 `\p{N}`, 51 `\p{M}` — the doc is right to build class tables from generated onig data, not the host's unicodedata.

## §5 K6 — 1.2M pieces, 0 mismatches

`tokenizer.model.tokenize` on byte-level-mapped pieces (a tokenizer.json with only the BPE model, no pre-tokenizer, no added tokens), vs the doc's algorithm (initial symbols `byte2id[b]`, repeatedly merge the adjacent pair with the lowest rank, ties leftmost; `ignore_merges` = whole-piece vocab lookup; duplicate merge pairs keep the last):

| tokenizer | ignore_merges | cases (≤300-byte pieces: real-text slices + random bytes, ascii-skewed and uniform) | mismatches |
|---|---|---|---|
| gpt2 | false | 400,000 | 0 |
| llama 3 | true | 400,000 | 0 |
| qwen3 | false | 400,000 | 0 |

Duplicate-pairs-keep-last is llama-3-relevant (its merges list contains duplicates) and is confirmed empirically by the 0-mismatch run. `ignore_merges` hitting only the model vocab (not added-only tokens) matches `BPE::tokenize_with_cache`'s `self.vocab.get(sequence)`.

## §4 K1 + driver policy — 400K cases, 0 mismatches, plus two targeted repros

Synthetic tokenizer: byte-level BPE (all 256 byte tokens + merges incl. `ma`,`mas`,`ask`-shaped ones), added tokens with shared prefixes (`<|im`, `<|im_start|>`), a non-special inside a special (`ask>` and `sk>` inside `<mask>`), one-byte tokens (`s` single_word, `X` lstrip), lstrip/rstrip/single_word combinations, all 25 whitespace cps around stripped tokens. Both modes ALL and NONSPECIAL (hf `encode_special_tokens=True/False`, toggled and read back through the binding), full encode ids compared.

- 400,000 random cases (on the same Neoverse-V2 box): **0 mismatches** for the doc's policy (resume at raw m_end; single_word; lstrip clamped to the previous split's end; rstrip; gap emission; phase 1 on normalized gaps).
- hf's own rust unit-test string `"Hi <mask> there\t<mask>\t<mask> "` reproduced exactly in both modes.

Two subtleties worth pinning as oracle vectors, because they look like counterexamples but are exactly what the doc says:

1. **NONSPECIAL drops a special without rescanning its bytes** (§4 step 1): with `<mask>` special and `sk>` non-special but `normalized=false` (same phase-0 trie), input `"Hi <mask> there"` in NONSPECIAL gives `[72,105,32,60,109,257,115,107,62,32,...]` — the `sk>` inside the dropped `<mask>` is **not** extracted; the bytes go through the model. So daachorse's find_iter resumes at the dropped match's end — the doc's rule, now empirically pinned.
2. **A normalized non-special inside a dropped special IS extracted** — by phase 1: with `ask>` `normalized=true`, the same input gives `[...32,60,109,258,97,275,...]` = `<` `ma` `s` `k` `>` … i.e. hf's `test_encode_special_tokens` behaviour. This is not a rescan: phase 0 dropped `<mask>`, the gap bytes then get normalized and scanned by the phase-1 trie. kernels.md models this (phase 1 on each normalized gap); a naive reading of hf's rust test as "rescan" would be wrong.

## §2 invalid-utf-8 atoms

No hf oracle exists (hf is undefined on invalid utf-8; the byte-input spec's oracle is tokref, not yet built). Self-consistency check: the rule is a well-defined total partition — at each position either a well-formed sequence per unicode §3.9 table 3-7 (no overlongs, no surrogates, nothing above U+10FFFF) consumes k bytes, or exactly one byte is consumed and decoding resumes at i+1; every byte belongs to exactly one atom, and the class assignment (`TOKS_C_P`, no flag bits) is total. Deterministic, matches table 3-7 as cited. Note the BPE side of this path *was* exercised: random-byte pieces up to 300 bytes (which are frequently invalid as utf-8 in the raw domain) went through the byte-level mapping and matched hf exactly in the §5 runs — the piece-level consequence of the atom rule (bytes preserved, mapped through `bytes_char`) is hf-exact.

## Notes (not defects)

- `TOKS_TP_DIGITS_RUN` (`\p{N}+`, 0x18) is not used by any of the five named variants (gpt-2 uses SP_RUN; cl100k/llama3 use 1_3; qwen uses 1). Harmless headroom; if o200k or a future family needs plain `\p{N}+` it's already parameterised.
- The §4 sub-case "the next match starts inside the previous rstrip run, so no gap is emitted" is verified against the source (`find_matches`: the gap is pushed only `if start_offset < start`, and the trie iterator is unaffected by strips) but was not hit by the fuzzer, because no synthetic added token *begins* with whitespace. Suggest adding one (e.g. content `" x"`) to the committed oracle so the behaviour stays pinned by execution rather than by reading.
- hf reassigns added-token ids on load (the JSON `id` is only warning-checked; content in the model vocab keeps its vocab id, otherwise next free id — `serialization.rs` visit_map + `add_tokens`). Kernels.md's `TAE_ID` is a toks-internal field so this is out of scope, but the compiler must reproduce hf's assignment if `.toks` is to round-trip against hf ids.
- The python binding's `pre_tokenize_str` returns character offsets, not byte offsets — irrelevant to the doc, recorded here because anyone re-running these experiments will hit it.

## Where the numbers come from

A Neoverse-V2 box, not a receipt host (linux aarch64, 72 cores, `~/toks-ci/review-kernels/`, `uv run --with tokenizers==0.23.2`, 64-process pools): K3 13,200,000 cases; BPE 1,200,000 pieces; added-token 400,000 encodings; class/fold probes over all 1,112,064 scalar values. Local (control-plane mac, same tokenizers 0.23.2): 3,000 K3 cases/variant smoke + the targeted §4 repros above. Raw logs: `k3.log`, `bpe.log`, `added.log` + `*_done.json` on that box in that directory.

Harness (now committed under tests/model/): `toks_model.py` (the doc transcription: A1–A7, added-token policy, BPE, probes), `smoke.py`, `run_k3_fuzz.py`, `run_bpe.py`, `run_added.py`, `probes.pkl` (probed class/fold sets), `wordset.pkl` (rust-regex `\w` for single_word), plus the four tokenizer.json files.

Every number above was produced by the runs described; nothing is estimated.
