# toks' unicode tables

The property tables the scanner templates read are **oniguruma's**, exactly as
hf `tokenizers` 0.23.2 runs them in `pre_tokenizers.Split(Regex(...))`.  They
are *not* python's `unicodedata`, and the two disagree on this machine
(python 3.12 = unicode 15.0).  Everything here is probed, never assumed.

## Files

| path | what |
|---|---|
| `tools/gen/unicode.py` | the probe + generator (PEP 723, `tokenizers==0.23.2`) |
| `src/gen/ucd_flags.c/.h` | per-code-point property flags, two-level, generated and checked in |
| `src/core/classes.c/.h` | the per-template class-table builder (`toks_classes_build`) |
| `tests/unicode/` | fresh-probe conformance test (C driver + probe script) |

## The probing method

`pre_tokenizers.Split(Regex(P), behavior="removed")` over batched strings of
code points (32 Ki chars per call): a code point the regex matches is removed
from the output, so "not kept" == "in the property".  `pre_tokenize_str`
offsets are **char** offsets (verified with mixed 1/3/4-byte chars).  One
full-property sweep over all 1,112,064 scalar values takes ~0.2 s; a complete
generation (~20 sweeps) runs in a few seconds on a developer laptop.

## What was verified (2026-10-03, tokenizers 0.23.2)

- **`\p{L}` == `\p{Lu} | \p{Ll} | \p{Lt} | \p{Lm} | \p{Lo}`** on every code
  point (probed both ways: `\p{L}` directly and each sub-property).  Counts:
  Lu 1858, Ll 2258, Lt 31, Lm 404, Lo 136477, **L 141028**; **M 2501**; **N 1911**;
  **P 855**, **S 8514** (added for the dsv3 template, docs/templates/dsv3.md; L, M, N, P, S, `\s`
  pairwise disjoint, and P / S equal python 3.14's unicodedata 16.0 on every code point).
  (These match tok v1's CASE-image counts: U = Lu+Lt = 1889, LL = 2258,
  O = Lm+Lo = 136881.)
- **`\s` is exactly 25 code points**: U+0009–000D, 0020, 0085, 00A0, 1680,
  2000–200A, 2028, 2029, 202F, 205F, 3000.  U+180E, U+200B and U+FEFF are
  *not* `\s` (probed; python's `str.isspace` wrongly adds 1C–1F).
- **The oniguruma data is unicode 16.0**: L/M/N/`\s` agree with python 3.14's
  `unicodedata` (16.0) on **every** code point (0 disagreements), and disagree
  with python 3.12's (15.0) on exactly the post-15.0 additions: 4924 letter
  diffs (e.g. U+1C89, U+A7CB, U+105C0+), 51 mark diffs (U+0897, U+10D69..,
  U+113B8..), 80 N diffs (U+10D40.., U+116D0..).  tok v1's note "16.0" is
  confirmed exactly.
- **The contraction fold is only U+017F (ſ → s)**: probed per letter
  (`(?i:s)`, `(?i:t)`, `(?i:m)`, `(?i:d)`, `(?i:r)`, `(?i:e)`, `(?i:v)`,
  `(?i:l)`) over every scalar — the only non-ASCII code point any of them
  matches is U+017F (matched by `(?i:s)`).  The full contraction group
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)` was then probed over every scalar in three
  contexts (`'X`, `'Xe`, `'Xl`, NUL-separated so no match spans items): the
  participant set is exactly `D L M R S T V d l m r s t v ſ` — ASCII pairs
  plus `ſ`.  **No multi-char fold reaches the contraction set** (ß→ss, ﬅ→st,
  ﬆ→st, ẞ, and all the ﬀ ﬁ ﬂ ﬃ ﬄ ligatures probed: none match).
- Surrogates U+D800–DFFF keep flags 0 (they are not scalar values; they can
  never be decoded from valid UTF-8).
- Post-unicode-9 spot checks: U+0897 (Todhri mark, 14.0) is `\p{M}`;
  U+9FEA (CJK ext H, 13.0) and U+30000 (ext G, 13.0) are letters;
  U+105C0 (Todhri letter, 16.0) is a letter.

## The generated table (`src/gen/ucd_flags.*`)

Two-level over U+0000..U+10FFFF: `toks_ucd_stage1[0x1100]` (u16 block index
per 256 code points) and `toks_ucd_stage2[n_blocks * 256]` (u16 flags per code
point), blocks deduplicated in first-seen order, block 0 = U+0000..U+00FF.
`toks_ucd_flags(cp)` returns 0 for cp > U+10FFFF.  Flag bits: `TOKS_UCD_LU
0x0001, LL 0x0002, LT 0x0004, LM 0x0008, LO 0x0010, M 0x0020, N 0x0040,
WS 0x0080, FOLD_S 0x0100, P 0x0200, S 0x0400`.

- n_blocks: **155**, stage1 8704 B, stage2 79360 B (141 / 72192 B before the P and S bits)
- table data sha256 (n_blocks u32 LE ‖ stage1 ‖ stage2):
  `99476d4b2feedbbae022701b2f7439247838ff0c12959a2d845f9d89794d1201`
- files: `src/gen/ucd_flags.h` sha256
  `0fa84a2e66e0e49c5481a27deca17e01123219251e640f60c04af09f80348261`,
  `src/gen/ucd_flags.c` sha256
  `3728645eb7d97e9fef1e8257714bdd11405bf46eec0f395a1a00f45a04648d24`

## The class-table builder (`src/core/classes.*`)

`toks_classes_build(class_flags, buf, buf_len, out)` packs the per-template
class bytes (the layout contract: `src/core/layout.h`):

- bits 0–2 base: 0 P, 1 L (letters; `\p{M}` too under
  `TOKS_CLASSES_MARKS_ARE_LETTERS`), 2 N, 3 WS (`\s` except CR/LF), 4 NL
  (CR/LF)
- bit 3 UPPER (Lu, Lt, Lm, Lo, M), bit 4 LOWER (Ll, Lm, Lo, M), bit 5 MARK
  (`\p{M}`, whatever its base), bit 6 FOLD_S (U+017F), bit 7 reserved 0
- `TOKS_CLASSES_DSV3` (class_flags 0x8) builds another byte, the dsv3 kind
  (docs/templates/dsv3.md §2): base P (`\p{P}`|`\p{S}`), L (`\p{L}`|`\p{M}`),
  N, WS, NL, X = 5 (none of those) and bit 7 CJK (U+3040..30FF, U+4E00..9FA5),
  nothing else: 151 blocks.

Output is a two-level u8 table (ascii[128] + stage1 u16[0x1100] + deduplicated
stage2 blocks) in the caller's buffer; `toks_classes_bytes` publishes the
worst-case bound (1,152,440 B; actual use 140 blocks = 44,704 B).  Determinism
is by construction: first-seen block order over the fixed generated table —
verified by building twice and comparing.  `toks_class_of` fast-paths ASCII
through the 128-byte table and returns P for cp > U+10FFFF.

## Regenerate / test

```
uv run tools/gen/unicode.py                        # writes src/gen/ucd_flags.*
clang -std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra \
  -Wconversion -Wsign-conversion -Werror -I src \
  tests/unicode/main.c src/core/classes.c src/gen/ucd_flags.c -o build/unicode_test
uv run tests/unicode/probe.py > build/gt.txt       # fresh hf probe
./build/unicode_test < build/gt.txt
```

The test checks every scalar value under both class_flags settings against
the fresh probe, the error/sizing contracts, determinism, the ASCII fast
path, and the targeted set (the 25 `\s` code points, CR/LF vs U+0085/U+2028/
U+2029, U+180E/200B/FEFF not `\s`, ſ alone carrying FOLD_S, U+0897/U+9FEA/
U+30000/U+105C0).  `make test` does not build it.

A `tokenizers` upgrade (or any oniguruma data change) invalidates both
`src/gen/ucd_flags.*` and the counts above: regenerate, re-run the test, and
record the new facts here.
