# Gigatoken pre-tokenizer scanners: a brief for toks' asm authors

Source scanned: `https://github.com/marcelroed/gigatoken` @ `fac0114` (rust, MIT): `src/pretokenize/fast/`, `src/pretokenize/mod.rs`, `src/pretokenize/options.rs`, `pretokenizer_optimization_log.md`.
Purpose: exact reference for what gigatoken's SIMD pre-tokenizer scanners compute per family, so toks can re-derive the same algebra in hand-written NEON / AVX-512 asm and beat it.
Citations are `gigatoken/<file>:<line>` relative to that repo root. Numbers are quoted, never invented; the two honest gaps are flagged in §2 and §4.

## Contents

1. [Map: family → regex → scanner fns](#1-map)
2. [The SIMD machinery (mask.rs)](#2-simd-machinery)
3. [cl100k-family boundary algebra](#3-cl100k-family-boundary-algebra)
4. [o200k and deepseek v3](#4-o200k-and-deepseek-v3)
5. [Measured numbers; optimization log wins and negatives](#5-measured-numbers)
6. [Parallel splitting (pretokenize/mod.rs)](#6-parallel-splitting)
7. [Takeaways for toks](#7-takeaways-for-toks)

## 1. Map

Architecture: every family is a *thin wrapper* around shared engines.

- **`cl100k_family`** (`fast/cl100k_family.rs`): one boundary algebra shared by cl100k, olmo3, qwen2, qwen3.5. Knob: `digits3: bool` (`\p{N}{1,3}` vs `\p{N}`) plus a `class_of` closure for non-ASCII (`pub(crate) fn batch_masks` cl100k_family.rs:118; avx512/avx2 dispatch cl100k_family.rs:203-236).
- **`o200k_family`** (`fast/o200k_family.rs`): 4 const-generic knobs `<CONTRACTIONS, DIGITS3, SLASH, HAN>` — o200k = `<true,true,true,false>` (o200k.rs:16-29), nemotron = `<false,false,true,false>` (nemotron.rs:17-30), kimi = `<true,true,false,true>` (kimi.rs:19-32). `pub(crate) fn advance_pos` o200k_family.rs:324, `pub(crate) fn batch_masks` o200k_family.rs:1292.
- **r50k**: its own standalone engine (r50k.rs:66 `batch_masks`, 169 `ascii_batch_algebra`, 278 `extended_masks`, 512 `advance_pos` scalar).
- **deepseek_v3**: scalar-only (see §4).

Each wrapper defines `struct FastXxxPretokenizer` (`{bytes, state: MaskState}`), `MaskScheme::advance` (scalar reference + no-SIMD path), `batch_masks` (aarch64) / `batch_masks_x86::<AVX512>` (x86_64, monomorphized both ways, runtime-detected), and `Iterator::next`. `PretokenizerType::pretokenize` dispatches once per iterator (options.rs:37-67); canonical names/aliases at options.rs:72-101; scheme identification from tokenizer.json Split regexes at options.rs:106-142. The shared chunked pull is `fill_spans_keyed_mask` (fast/mod.rs:53-71): with SIMD it is the two-phase walker `MaskState::fill_spans_two_phase`, else a fused scalar `next_span` loop.

| scheme | regex (as quoted in-repo) | scalar | SIMD |
|---|---|---|---|
| gpt2 / r50k | `'(?:[sdmt]\|ll\|ve\|re)\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+` (mod.rs:701-702; r50k.rs:1-2) | `r50k::advance_pos` (r50k.rs:512) | `r50k::batch_masks` → `ascii_batch_algebra`/`extended_masks` (r50k.rs:66/169/278) |
| gpt4 / cl100k | `'(?i:[sdmt]\|ll\|ve\|re)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s+$\|\s*[\r\n]\|\s+(?!\S)\|\s+` (cl100k.rs:266) | `cl100k::advance_pos` (cl100k.rs:143) | `cl100k_family::batch_masks` → `family_algebra` (cl100k_family.rs:118, 379) |
| qwen2 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` (qwen2.rs:263-264) | `qwen2::advance_pos` (qwen2.rs:204) | cl100k_family, `digits3=false` (qwen2.rs:16-18) |
| qwen35 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+\|\p{N}\| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` (qwen3_5.rs:325-326) | `qwen3_5::advance_pos` (qwen3_5.rs:204) | cl100k_family, `digits3=false` (qwen3_5.rs:20-22) |
| olmo3 | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` (olmo3.rs:261-262) | `olmo3::advance_pos` (olmo3.rs:139) | cl100k_family, `digits3=true` (olmo3.rs:14-16) |
| o200k | `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:...)?\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n/]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` (o200k.rs:80) | `o200k_family::advance_pos::<true,true,true,false>` (o200k_family.rs:324) | `o200k_family::batch_masks` → `o200k_algebra` (o200k_family.rs:1292, 1014) |
| nemotron | o200k shape without contractions and with `\p{N}` (nemotron.rs:80) | `advance_pos::<false,false,true,false>` (nemotron.rs:17) | `batch_masks::<false,false,true,false>` (nemotron.rs:23-30) |
| kimi | o200k + leading `[\p{Han}]+`, Han excluded from letter brackets, no `/` tail (kimi.rs:3-8) | `advance_pos::<true,true,false,true>` (kimi.rs:19) | `batch_masks::<true,true,false,true>` (kimi.rs:25-32) |
| deepseek_v3 | Sequence of 3 Splits: `\p{N}{1,3}`; `[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+`; `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+\|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+\| ?[\p{P}\p{S}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` (options.rs:25-29) | `advance_main`/`advance_cjk`/`advance_pos` + `scan_lm_from`/`scan_ps_from`/`scan_gap_from` (deepseek_v3.rs:251/354/373, 125/148/225) | none |

Llama3 note: gigatoken has **no llama3 scheme** (NAMES list, options.rs:72-82). Its olmo3 regex (olmo3.rs:261-262) is exactly llama3's published pretokenizer regex — cl100k minus `\s+$`, with `\s*[\r\n]+` instead of cl100k's `\s*[\r\n]`, contraction moved to front as `(?i:'s|...)`. For toks, "llama3/olmo3" is one family: qwen2's contraction shape + cl100k's `\p{N}{1,3}`, `digits3=true`, no `\s+$`.

## 2. SIMD machinery

Structure (mask.rs:1-34, the module doc): 64-byte batches; SIMD classifies every byte into per-class u64 bitmasks; scalar (GPR) bitmask algebra derives "a token starts here" bits; the walker pops one bit per token — no per-token dispatch branches, "~2x the serial scalar scanners". Layers: (1) platform SIMD primitives — `movemask64`/ascii masks on NEON, `ascii_masks_avx512`/`ascii_masks_avx2` on x86-64, "the only per-platform code"; (2) shared bit-domain helpers (`classify_uni_chars`, `char_through`, `nn_at_full`, `digit_run_splits3`); (3) per-scheme `batch_masks` algebra; (4) `MaskState` walker — segments, bad-zone gaps, scalar tail, one-batch-ahead precompute, scalar overruns stay on the 64-byte grid; (5) `fill_spans_two_phase` — chunked harvest into a flat boundary buffer, branch-free emission.

ISAs / detection:
- aarch64 NEON: compile-time (mask.rs:38-41). `movemask64(v0..v3: uint8x16_t)` folds 4×16B vectors into one u64 (mask.rs:113-117); the cl100k classifier produces 6+ such u64s per batch (l/d/s/wt/n/apostrophe/hi) — cl100k_family.rs:158-172.
- x86-64: `avx512_scanner_available()` = avx512f **and** avx512bw **and** avx512vl **and** bmi1, bmi2, lzcnt, popcnt (mask.rs:46-58). AVX-512 classifier = "one 64-byte load and one k-register compare per class" (cl100k_family.rs:203); AVX2 = "two 32-byte loads" (cl100k_family.rs:204). A higher fill tier additionally requires VBMI2 (native 512-bit `vpcompressb`: Zen 4/5, Ice Lake+; Skylake-X lacks it and stays plain AVX-512) — mask.rs:60-63, tier constants at mask.rs:669-673.
- No portable_simd / std::simd: raw `std::arch` intrinsics only.

Block → masks: a batch at `scan` (guarded `scan + 70 <= len`, so bytes scan..scan+70 are readable) produces `AsciiMasks { l, d, s, wt, n, hi, ap, ... }` — one u64 per class (field usage cl100k_family.rs:393-397, 554) — plus scheme extras (o200k: `OAsciiExtra` case/tail masks, o200k_family.rs:1449/1468).

Non-ASCII classification: `classify_uni_chars` (mask.rs:457+) walks the `hi` (non-ASCII lead) mask with `tzcnt` + `m &= m - 1` (blsr) over lead bytes, hand-decodes each codepoint (`decode_cp_inbounds`), and **broadcasts the char's class to every byte of the char** — "every byte of a classified char carries the char's class, so byte-adjacency == char-adjacency" (cl100k_family.rs:391-392). Output `UniClasses { l, ws, o, cont, resid, lead2, lead3, lead4, w2, w3 }`: class masks, continuation bytes, 2/3-byte-ws lead positions, residual unclassified. Chars straddling the batch edge resolve via `char_through` (mask.rs:373) plus a prev-char walk-back in the carries computation (r50k.rs:15-16). Schemes plug in their own `class_of` (packed unicode table, `unicode::ClassTable::get()` resolved once per batch — qwen2.rs:38-42).

Carries between blocks: `Carries { pl, ps, pwt, po, pws, pd, c2_os, b2b_in }` (cl100k_family.rs:78 `ascii_carries`, destructured :387). Each is the bit-0 seed for one class's shifted prev-byte test: `p_l = (lb << 1) | pl` etc. (cl100k_family.rs:410-413). `c2_os` = "prev char entirely before the batch"; `b2b_in` = "prev char straddles in; its own prev is P2" — both for the letters optional-prefix rule (cl100k_family.rs:408-409). `pd` seeds digit-run continuation (a run continuing from before the batch breaks the `{1,3}` phase — cl100k_family.rs:487-501).

Byte-64 lookahead: `nb64 = bytes[scan + 64]` (in bounds via the scan+70 guard). ASCII case is branchless: `nn64 = !is_ascii_ws(nb64)`; a non-ASCII byte 64 branches to `nn_at_full` (mask.rs:343-346 — table-backed decode of the full char). `nn64m = nn64 as u64 .wrapping_neg()` (all-ones when next char is non-ws) feeds the `(?!\S)` split bits for runs touching bit 63 (cl100k_family.rs:438-455). Rationale quoted: "the dominant deferral cause before this existed (~16% of OWT batches ended in a single space)" (cl100k_family.rs:440-441).

Bad zones and scalar fallback: `batch_masks` returns `(usable, bad)`; `bad` marks zones that `MaskState` re-derives through the scalar `advance`, "never emitting a token across an unresolved zone" (mask.rs:9-15). Sources of bad: unclassified non-ASCII (`resid` smeared ±1 byte, cl100k_family.rs:436); a ws run touching bit 63 whose resolution depends on bytes past 64 (`bad |= MAX << (h+1)`; entire-batch-ws short-circuits to `(0, u64::MAX)` — cl100k_family.rs:466-478); punct-absorbed newline runs at the edge with a ws byte 64 (cl100k_family.rs:457-464); digit runs straddling in via `pd` or following a bad zone (cl100k_family.rs:487-501, incl. the quoted latent Arabic-Indic-digits bug note); contractions at `i >= 61` (cl100k_family.rs:558-561); apostrophe before any non-ASCII (U+017F, §3). The walker splits each batch into a usable prefix + bad gap (`rem`, `scalar_until` — mask.rs:734-742), runs scalar through the gap, and stays on the 64-byte grid so the one-batch-ahead precompute survives overruns (mask.rs:27-29). Measured bad-zone rate on OWT: "~0.4% of batches" (r50k.rs:16-18).

Harvesting piece starts: `next_span` pops one bit per token from `usable` (mask.rs:754); the encode path uses `fill_spans_two_phase` (mask.rs:1047) — phase A flattens boundary indices into a flat buffer, phase B emits spans branch-free. x86 tiers: `fill_spans_two_phase_avx512_vbmi2_crc` (mask.rs:1111, the only divergence being `vpcompressb` in `flatten_bits_avx512`), `..._avx512_crc` (1132), `..._avx2_crc` (1151), generic `..._crc` (1173), all selected via `fill_spans_two_phase_impl<const X86_CRC, const X86_TIER>` (mask.rs:1189). Non-VBMI2 / NEON harvest is the tzcnt-and-clear loop (r50k.rs:7-11).

**Unverified (out of time):** exact instruction sequences inside `ascii_masks_avx512`/`ascii_masks_avx2` (mask.rs:228-340) and NEON `movemask64` (mask.rs:113-227), and `MaskState`'s field layout (mask.rs:680-753). Cited by signature/doc only.

## 3. cl100k-family boundary algebra

All from `family_algebra` (cl100k_family.rs:379-585); masks are u64, bit i = byte `scan + i`; `x << 1` = "the test one byte earlier".

Effective classes (cl100k_family.rs:391-397):
```
lb     = am.l | uni.l                    // letters (ASCII + classified)
sb     = am.s                            // the ` ?` prefix: ASCII 0x20 ONLY
wtb    = am.wt | uni.ws                  // tab-class ws
ob     = !(am.l|am.d|am.s|am.wt|am.n|am.hi) | uni.o   // "other" incl. non-ASCII unclassified
ws_all = sb | wtb | am.n
```

Letters `[^\r\n\p{L}\p{N}]?\p{L}+` (cl100k_family.rs:399-415): a letter char starts a token unless preceded by a letter (in-run), by a space (the ` ?` prefix continuation), by tab-ws — or unless it is itself absorbed as the optional single-char prefix `[^\r\n\p{L}\p{N}]?` of a following letter. B = "the char two back is punct-or-space", evaluated at each char's lead by shifting the prev-byte test `c_test = ((ob|sb) << 1) | po | ps` by the PREV char's byte-length (1/2/3/4 via `len1`, `uni.lead2/3/4`), plus `c2_os`/`b2b_in` for chars before/straddling the batch (cl100k_family.rs:402-409):
```
absorb    = p_o & !b2back          // this punct char is the optional prefix
b_letters = lb & !contm & !p_l & !p_s & !p_wt & !absorb
```

Digits `\p{N}{1,3}` (cl100k_family.rs:417-423): `b_digits = digit_run_splits3(am.d)` when `digits3 && am.d & (am.d >> 1) != 0` (a 2+ run exists), else `am.d`. `digit_run_splits3` (mask.rs:555+): starts are `d & !(d << 1)`; "a start at p re-arms at p+3 while the run continues" — a hop loop phasing runs into 1-3 char groups. Note it counts CHARS: non-ASCII digits are pushed out-of-mask into bad zones (the `pd` note, cl100k_family.rs:487-501).

Punct ` ?[^\s\p{L}\p{N}]+` (cl100k_family.rs:425-426):
```
b_punct = ob & !contm & !p_o & !p_s
```

CR/LF absorption smear `[\r\n]*` after punct (cl100k_family.rs:429-434): newlines directly after a punct run are absorbed into the punct token. Upward smear seeded at each such newline, bounded by the newline class:
```
abs_seed = am.n & ((ob << 1) | po)
abs_n    = smear_up(abs_seed, am.n)     // skipped entirely when abs_seed == 0
ws_eff   = ws_all & !abs_n
```

Whitespace runs (cl100k_family.rs:503-548). Base rule — a ws char starts a token at run start, or splits before the last char of a run when followed by a non-ws char (`\s+(?!\S)`):
```
ws_leads1 = (am.s | am.wt | am.n) & ws_eff
ws_leads  = (ws_leads1 | uni.w2 | uni.w3) & !abs_n
p_ws      = (ws_eff << 1) | pws
split_ok  = (ws_leads1 & (nonws >> 1)) | (uni.w2 & (nonws >> 2)) | (uni.w3 & (nonws >> 3))
          | (edge_last & nn64m)         // run touching bit 63: byte-64 lookahead
b_ws      = ws_leads & (!p_ws | split_ok)
```
(`edge_last` covers a 2-byte ws led at 62 / 3-byte ws led at 61 ending exactly at the batch edge — cl100k_family.rs:511.)

Runs containing a newline are then overridden (cl100k_family.rs:518-548): one token through the run's **last** newline (`\s*[\r\n]+`-style), then the tail after it gets r50k-style rules — a tail start bit at `q+1` (q = last NL in the run) and the tail's own last-char split via `tail_leads`. Implemented as a per-run loop over `runs_n` with tzcnt; a fully branchless downward-smear formulation "measured 0.95x" and was rejected (cl100k_family.rs:519-523).

Contractions `'(?i:[sdmt]|ll|ve|re)` (cl100k_family.rs:552-582): a post-pass over `cand = am.ap & boundary & !bad`. At each apostrophe bit i: `i >= 61` defers (bad); a non-ASCII byte at i+1 defers — because `(?i:'s)` also matches `'ſ` (U+017F LATIN SMALL LETTER LONG S, case-folds to `s`), which the table classifies as a letter, so an apostrophe before ANY non-ASCII defers to scalar (cl100k_family.rs:553, 563-569). ASCII case: `k = 2` for `s|d|m|t` (case-folded `|0x20`), `k = 3` for `ll|ve|re` (second byte at i+2); the fixup **clears the token-start bit at i+1** (consumes the letter-run start) and **sets one at i+k** (next token starts after the contraction):
```
boundary &= !(1 << (i + 1));  boundary |= 1 << (i + k);
```

Return: `(boundary & !bad, bad)` (cl100k_family.rs:584).

Family variants within this algebra:
- **r50k/gpt-2** (own engine): case-SENSITIVE contraction `'(?:[sdmt]|ll|ve|re)` (no U+017F issue), and the ` ?` prefix belongs to letters/digits/punct alike; boundary algebra `ascii_batch_algebra` (r50k.rs:169) / `extended_masks` (r50k.rs:278 — folds the unicode classes into the same algebra); ws special cases in `advance_ws` (r50k.rs:589).
- **cl100k**: `digits3=true`; contractions case-insensitive; `\s*[\r\n]` (single trailing newline); **`\s+$`** — trailing ws at end-of-input is ONE token, which is why a ws run touching bit 63 defers unless byte 64 resolves it (cl100k_family.rs:466-478).
- **llama3/olmo3**: `digits3=true`; `\s*[\r\n]+` — a ws run containing a newline ALWAYS splits right after its last newline, even at EOS; no `\s+$` (qwen2.rs:10-12 doc, applies to the whole `\s*[\r\n]+` family shape).
- **qwen2**: `digits3=false` — every digit is a token start; no `digit_run_splits3`.
- **qwen3.5**: as qwen2 but `\p{M}` joins the letter class (`[\p{L}\p{M}]+`) AND leaves the punct negation (`[^\s\p{L}\p{M}\p{N}]`), so marks classify into `lb`, not `ob` (qwen3_5.rs:325-326; class closure lives in each wrapper's `class_of`).

## 4. o200k and deepseek v3

o200k family (o200k_family.rs), knobs `<CONTRACTIONS, DIGITS3, SLASH, HAN>`:

- Case-split letter groups: the two big alternatives split letters into Upper `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and Lower `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`. Scalar: `ascii_letter_state` (o200k_family.rs:100) + `scan_case_run` (140) walk a `CaseState` (Upper→Lower keeps one token `Upper* Lower+`; Upper-only runs are `Upper+ Lower*`); `letter_run_first` (111). Mask domain: `o200k_algebra` (1014) consumes per-case masks; `smear_up` (461) is the shared upward smear (same helper as cl100k_family.rs:42).
- Contractions as **suffix**: `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` trails the letter run (not a separate leading alternative). Scalar `try_suffix` (o200k_family.rs:195); the mask path handles it at run ends (CONTRACTIONS knob).
- Punct tail `[\r\n/]*`: `is_tail_byte::<SLASH>` (o200k_family.rs:63), `scan_punct_from` (214), `scan_tail` (244); a tail crossing the batch edge is resolved by `prev_tail_absorbed` (559) and its carries `tail_carries` (689) / `ascii_batch_carries` (1273).
- Han: `scan_han_run` (o200k_family.rs:254), `family_class_of::<HAN>` (71) — kimi's leading `[\p{Han}]+` alternative and Han exclusion from the Upper/Lower brackets.
- Digits: `digit_token_end::<DIGITS3>` (o200k_family.rs:313).
- SIMD entry: `o200k_extended_masks` (o200k_family.rs:829) — the "carries walk" analogue of cl100k's `family_extended_masks`; `ascii_extra_avx512`/`avx2` (1449/1468) produce the case/tail masks with the same one-load-per-class k-register compares.

deepseek v3 (deepseek_v3.rs; scalar-only): a **Sequence of 3 Splits** (options.rs:25-29; `from_split_regexes` matches the exact 3-pattern list, options.rs:106-112). Semantics: split 1 (`\p{N}{1,3}`) and split 2 (CJK/Hiragana/Katakana runs) carve the text first; split 3 (the GPT-ish pattern with `[\p{P}\p{S}]` punct classes and `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+` punct-prefixed words) applies within the remaining gaps. The scanner tracks a `cjk_region: bool` — `advance_main` (deepseek_v3.rs:251) vs `advance_cjk` (354), with `lm_end_at`/`scan_lm_from` (105/125, letter-or-mark runs), `scan_ps_from` (148, punct/symbol), `scan_gap_from` (225), `ws_token_end` (179), and `advance_pos` (373) dispatching. There is no mask path: the Sequence semantics (re-splitting the output of earlier splits) did not fold into the batch algebra.

**Unverified (out of time):** the interiors of `o200k_algebra` (o200k_family.rs:1014-1273) and the deepseek scalar bodies — signatures, doc comments and the cl100k structural analogue above are verified; the boolean formulas inside are not.

## 5. Measured numbers

All numbers below are quoted from the repo; they are r50k/GPT-2 unless noted, on **Apple Silicon ARM** (opt log platform note, log:10; 100 MB OWT, `cargo bench`, `lto="fat"`).

Optimization log (pretokenizer_optimization_log.md), the scalar-era history:

| step | throughput | note |
|---|---|---|
| fancy-regex baseline | ~47 MiB/s | log:16 |
| hand-rolled state machine | ~380 MiB/s | log:17 |
| winnow + NEON | ~462 MiB/s | log:18 |
| 1: LUT dispatch + SWAR 8-byte runs | 830 MiB/s | +1.80x, log:47 |
| 2: `get_unchecked` | 840 MiB/s | log:59 |
| 3: arithmetic space dispatch | ~840 | no change, log:70 |
| **4: hot/cold split of the unicode path** | **580 MiB/s** | **REVERTED — the `#[inline(never)]` barrier cost more than the icache win (−31%, log:74-81)** |
| 5: advance/count separation | 848 MiB/s | log:94 |
| **6: two-pass classification buffer** | **306-354 MiB/s** | **not used — memory traffic dominates, 2.5x slower (log:98-117)** |
| 7: PGO | 842 MiB/s | no effect on the branchless SWAR loop; hurt the state machine by 13% (log:121-131) |
| 8: dual-cursor ILP | 1,049 MiB/s | +25%, crossed 1 GiB/s (log:135-162) |

Totals: 2.27x over winnow+NEON, 22.3x over regex (log:180-181). Lessons quoted at log:183-188 (SWAR the biggest single win; framework overhead > SIMD width; multi-pass loses to single-pass; PGO can't fix data-dependent branches; dual cursors fill pipeline bubbles).

The mask-scanner era (r50k.rs module doc, not in the log file): scalar scanner ~**983 MB/s**; step 15 (vector-register algebra) **2,132 MB/s**; current form (scalar-register algebra, step 17) **2,460-2,600 MB/s** on 1 GB OWT, `pretokenize_profile`, min-of-N interleaved (r50k.rs:18-20). The scalar scanner's floor is "~8 cy/token branch-miss" (r50k.rs:10-11). `advance_pos` as a free function (cursor in a register, no `self.pos` stores per step) is "worth ~30% throughput" (r50k.rs:26-29). Routing span pulls through `Iterator::next` cost "~23% of warm encode time" (fast/mod.rs:51-52).

No per-CPU (Zen 5 / Neoverse) breakdown exists in the scanned files; the only platforms evidenced are the Apple Silicon bench and the x86 tier structure.

## 6. Parallel splitting

`safe_split_ranges(bytes, target, added_tokens)` (mod.rs:603-660) cuts the input into ranges that are exact token-boundary-safe:

- **Cut rule** (mod.rs:641-645): a boundary is `bytes[probe] == b' '` with `bytes[probe-1].is_ascii_alphanumeric()` and `bytes[probe+1].is_ascii_alphabetic()`. Why exact: the space sits after an alnum byte, so it cannot be inside a token that started earlier (no alternative spans alnum→space), and as a token start the space is the ` ?` prefix of the following letter token — the boundary lands between two tokens. The probe walks byte-by-byte from `start + target` (chunks are therefore ≥ `target.max(1)` bytes, mod.rs:635-639); the remainder becomes one final range (mod.rs:653).
- **Added tokens** (mod.rs:608-631): only tokens containing a space can span such a cut; `memmem::Finder`s search a `max_blocker`-radius window around each candidate cut and reject any cut crossed by an occurrence (`cuts_added_token`, mod.rs:624-631).
- **rstrip tokens** (mod.rs:596-602, 614-620, 632-633): an `rstrip` added token absorbs trailing whitespace, so a cut exactly at its end is also rejected (`ends_rstrip_token`: `bytes[..p].ends_with(t)`, for tokens whose last byte is ASCII alnum). `lstrip` needs no counterpart (mod.rs:598-602).
- Verified by tests: `safe_split_ranges_pretoken_equivalent` — per-range pretokenize + concat == whole-input pretokenize (mod.rs:720); added-token avoidance (mod.rs:779); rstrip ends (mod.rs:814).
- `pretokenize_par_bytes` (mod.rs:668-691): rayon over `par_document_chunks(separator, n_threads)`, per-document `pretoken_count`, `par_merge_counts` into an FxHashMap.

## 7. Takeaways for toks

Ranked by expected value for hand-written asm:

1. **Copy the two-phase structure wholesale**: 64B batch → per-class u64 bitmasks (one compare + movemask per class) → GPR shifted-mask algebra for starts → harvest (`vpcompressb` / tzcnt loop) into a flat boundary buffer → branch-free span emission. This is the ~2x-over-scalar win; the boundary algebra (§3) is fully specified above.
2. **Do the algebra in GPRs, not vector registers.** Gigatoken measured the vector-register formulation equal and retired it for the simpler scalar form (r50k.rs:7-10; step 15 → 17 = 2,132 → 2,460-2,600 MB/s). In asm: one 64-byte load, N class compares, N mask extractions, then everything is BMI2 (`tzcnt`, `blsr`, `andn`, shifts). Don't spend vector ports after the masks exist.
3. **Byte-64 lookahead + the `scan + 70 <= len` guard** kills the dominant ws-run deferral (~16% of OWT batches, cl100k_family.rs:440-446). Branchless for the ASCII byte (common case), rare branch for non-ASCII. Cheap, high-value.
4. **Class broadcast per char** ("byte-adjacency == char-adjacency"): tzcnt/blsr walk over lead bytes + packed codepoint-class table (cl100k_family.rs:391-392, mask.rs:457+). Keep unclassified chars in bad zones (resid ±1 smear) and keep the scalar executor on the 64-byte grid so batch pipelining survives overruns.
5. **Families as data over one algebra**: cl100k/llama3/qwen2/qwen3.5 differ by `digits3` + which classes feed `lb`/`ob` + the ws-tail rule (`\s+$` vs split-after-last-newline); the o200k family by 4 const knobs. This matches toks' "families are data, not code" doctrine — implement the cl100k algebra once, parameterize.
6. **ILP discipline from the log**: free-function cursor in a register (+30%); never route span pulls through `Iterator::next` (−23% encode time); dual-cursor interleaving for the scalar path (+25%). In asm: keep cursor + masks entirely in registers; software-pipeline two batches (classify batch B+1 while harvesting A) — gigatoken's "one-batch-ahead precompute" (mask.rs:28) is the seed of this.
7. **Negatives to not repeat**: hot/cold (outline) split of the unicode path (−31%, log step 4); two-pass classify-then-count (2.5x slower, step 6); PGO on branchless loops (step 7); branchless-smear NL-run override (0.95x, cl100k_family.rs:519-523). And gigatoken's own latent bug class: `\p{N}{1,3}` counts CHARS, not bytes — Arabic-Indic digits (cl100k_family.rs:492-495); toks must count classified chars.
8. **Beat these numbers**: r50k mask scanner 2,460-2,600 MB/s on Apple Silicon (1 GB OWT); scalar 983. On NEON/Apple M, win by fusing classify/extract/harvest (Rust's autovec classifier + tier dispatch leaves gaps) and keeping the whole walker in registers. On Zen 5 AVX-512, VBMI2 `vpcompressb` harvesting is already gigatoken's top tier — the wins there are pipelining two batches and cutting the per-class k-register extract overhead.

Gaps left for a follow-up pass: exact instruction sequences of `ascii_masks_avx512/avx2` and NEON `movemask64` (mask.rs:113-340), `MaskState` walker internals (mask.rs:680-753), `o200k_algebra` formulas (o200k_family.rs:1014+), and the deepseek scalar bodies.
