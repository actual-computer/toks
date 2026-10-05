# Parity oracle: how the modes map to hf 0.23.2

## Verified facts (tokenizers 0.23.2, probed 2026-10-03)

1. **Mode mapping** (§3.2):
   - ALL = `t.encode(text)` / `t.encode(text, add_special_tokens=True)` — default.
   - NONSPECIAL = `t.encode_special_tokens = True; t.encode(text, add_special_tokens=False)`.
     (post-processor is a separate flag; encode_special_tokens only affects added-token matching.)
   - NONE = tokenizer.json with `added_tokens: []` and everything else byte-identical
     ("the reference pipeline with added-token extraction removed and every model table unchanged").
     Note: the post-processor is retained (llama's Template still emits <|begin_of_text|> from its own
     special_tokens map — that is HF's own behavior for pp; toks' TOKS_NO_POSTPROCESS handles that
     separately). Verified: removing added_tokens keeps model vocab/merges identical
     (get_vocab_size(with_added=False) unchanged), and pieces/encode then flow through the plain pipeline.
   - gpt2's <|endoftext|> is `normalized: True` and NONSPECIAL hides it too (encode_special_tokens=True
     hides *all* added tokens from matching, not just special ones). Verified: gpt2 ALL [64,275,269] vs
     NONSPECIAL same (no literal in text), and with a literal ALL maps it to 50256 while NONSPECIAL
     spells it out.

2. **Pieces / offsets**: HF's `encode().offsets` are ORIGINAL-string char offsets (normalization-aware,
   per added-token segment). SPEC §3.5 wants piece *end offsets in the normalized byte stream*: the oracle
   derives them by normalizing each added-token segment, calling `pre_tokenizer.pre_tokenize_str` on the
   normalized segment, and converting the (start, end) char offsets to byte offsets in the concatenated
   normalized stream. Added tokens appear as one piece each spanning their literal bytes.
   Verified with Qwen (NFC shrink case e+U+0301) and added-token-adjacent segments.

3. **Decode**: use the ALL tokenizer. Ids with no string are skipped by HF; skip_special_tokens=True
   drops ids whose added-token entry is special=true (deepseek's ids 0..2 inside the model vocab are
   dropped when special — verified). Out-of-range ids raise (toks: TOKS_E_ID).

4. **Stream**: stream == batch == `bytes.decode('utf-8', 'replace')` over the concatenated token bytes
   (minus skipped specials), i.e. Rust `String::from_utf8_lossy`. Per-push eagerness: after each push the
   output equals the full lossy decode of all bytes so far minus a held valid-prefix tail (held(): ported
   to src/core/stream.c and to python/toks_oracle/oracle.py's stream reference).

5. **tiktoken** (cl100k): ALL = `encode(text, allowed_special="all")`, NONSPECIAL/NONE =
   `encode_ordinary(text)` (tiktoken has no non-special added tokens; ordinary mode never matches
   specials). Decode = decode_bytes (no skip_special concept; toks SKIP_SPECIAL for tiktoken inputs
   must be defined by the .toks-side: specials = the pattern file's specials — record, don't guess).
   Post-processing for tiktoken inputs: none (no template); TOKS_NO_POSTPROCESS is a no-op difference.

6. **NFC facts**: tokenizers 0.23.2 pins Unicode 9.0 NFC semantics via unicode-normalization-alignments;
   Python's unicodedata may differ (probe showed e.g. U+0898+ ccc=0 in HF). For the oracle we always
   use the tokenizer's own normalizer object, never unicodedata.
