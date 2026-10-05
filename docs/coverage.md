# toks coverage: the templates and components "every model" needs, in order

Census of 2026-10-04 by `tools/census/coverage.py` (data: `census/coverage.json`; method: section 10). Population, SPEC 1.1: **(b)** the top 500 text-generation and the top 100 embedding / reranker models on the hf hub by downloads (the hub's 30-day count), **(c)** the named families, and **(d)** an extension slice, the top 100 image-text-to-text / any-to-any models (multimodal llms; their text side is an llm tokenizer), reported in its own column and never mixed into (b). Shares are of (b)'s **1.13B downloads** (600 models, 169 distinct tokenizers by normalized-configuration hash); a model whose tokenizer could not be read counts as a miss. *Covered today* means toks's own load path (`toks_load` at `d0b927a418`, section 9) accepts the model's tokenizer as the model ships it; toks refuses at load what it cannot reproduce exactly, and the pinned files of tests/data/targets/ledger.txt carry the hf parity checks. The orders below start from there: a feature some loading tokenizer uses counts as done.

## 0. Critical targets

The five critical targets: step 0 of every build order below (sections 1 and 2), whatever their rank. *toks today* is toks's own load path run on the model's files as it ships them (section 9). Exact patterns are json string literals of the regex.

| target | repo / tokenizer | component signature | exact pattern | toks today | blocks it | family + mirrors |
|---|---|---|---|---|---|---|
| **GLM 5.3** | `zai-org/GLM-5.3`<br>`1ca61c4fd0d9` (P07, cl100k: cl100k / llama 3) | bpe-bytelevel (ignore_merges); vocab 154820, 321649 merges; normalizer none; pre-tokenizer Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0); decoder ByteLevel; post-processor ByteLevel; 36 added tokens (18 special) | `"(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+"` | loads | nothing | 42 of 43 other repos load it exactly; 1 ship another tokenizer (below) |
| **Kimi K3** | `moonshotai/Kimi-K3`<br>`664949c0d15d` (P14, kimi: kimi k2 / k2.5 / k3 / linear) | tiktoken ranks (163584): byte-level BPE by rank; no normalizer; the pattern of the next column; 256 reserved special ids; decoder: bytes | `"[\\p{Han}]+\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+"` | loads | nothing | 21 of 33 other repos load it exactly; 10 ship another tokenizer (below) |
| **gpt-oss** | `openai/gpt-oss-20b`<br>`5d3cc21cbb6c` (P11, o200k: o200k (gpt-oss, llama 4)) | bpe-bytelevel (ignore_merges); vocab 199998, 446189 merges; normalizer none; pre-tokenizer Split[Isolated](re:"[^\\r\\n\\p{L}\\p{N}]?[\\p...") &gt; ByteLevel(regex=0,prefix_space=0); decoder ByteLevel; post-processor ByteLevel; 21 added tokens (21 special) | `"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)?\|\\p{N}{1,3}\| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+"` | loads | nothing | 47 of 54 other repos load it exactly; 4 ship another tokenizer (below) |
| **Qwen 3.8** | `Qwen/Qwen3.8-27B`<br>`d500bb03c594` (P06, cl100k: qwen 3.5 / 3.6 / 3.8 (marks)) | bpe-bytelevel; vocab 248044, 247587 merges; normalizer NFC; pre-tokenizer Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0); decoder ByteLevel; post-processor ByteLevel; 33 added tokens (21 special) | `"(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+\|\\p{N}\| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*\|\\s*[\\r\\n]+\|\\s+(?!\\S)\|\\s+"` | loads | nothing | 33 of 54 other repos load it exactly; 21 ship another tokenizer (below) |
| **Gemma 4** | `google/gemma-4-26B-A4B-it`<br>`22d7d15144bd` (P08, none: Split(' ') after Replace(' ' -&gt; U+2581): no split) | bpe-byte-fallback (byte_fallback, fuse_unk, unk &lt;unk&gt;); vocab 262144, 514906 merges; normalizer Replace(" "-&gt;"▁"); pre-tokenizer Split[MergedWithPrevious](str:" "); decoder Replace &gt; ByteFallback &gt; Fuse; post-processor TemplateProcessing[$A:0]; 24 added tokens (24 special) | Split[MergedWithPrevious](str:" ") | loads | nothing | 50 of 54 other repos load it exactly; 4 ship another tokenizer (below) |

**GLM 5.3** (`zai-org/GLM-5.3`)

- 1 repo (55k downloads; `RedHatAI/GLM-5.3-Flash-NVFP4`): tokenizer `2cd8ea58943f` (P07: cl100k / llama 3); against the target it differs: truncation.

43 other repos (the family's other sizes, the most downloaded repos whose id carries the family name, and the quantizer / mirror orgs unsloth, nvidia, lmstudio-community, mlx-community, bartowski, RedHatAI); 42 load the target's tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).

| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |
|---|---:|---|---|---|---|
| `zai-org/GLM-5.3` | 1.4M | tokenizer.json | `1ca61c4fd0d9` | target |  |
| `zai-org/GLM-5.3-Flash` | 5.4M | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `unsloth/GLM-5.3-Flash-GGUF` | 1.0M | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) | gpt2 / glm4 / 154880 |
| `unsloth/GLM-5.3-GGUF` | 561k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3) | gpt2 / glm4 / 154880 |
| `orcarouter/GLM-5.3-Flash-Uncensored-FP8` | 224k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `antirez/glm-5.3-flash-gguf` | 191k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash (name)) | gpt2 / glm4 / 154880 |
| `RadixArk/GLM-5.3-Flash-NVFP4` | 86k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `dealignai/GLM-5.3-CYBERSECURITY-FP8` | 86k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RadixArk/GLM-5.3-NVFP4` | 83k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `Inferact/GLM-5.3-NVFP4` | 70k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `nvidia/GLM-5.3-Flash-NVFP4` | 69k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `zai-org/GLM-5.3-Flash-BF16` | 62k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3-Flash-NVFP4` | 55k | tokenizer.json | `2cd8ea58943f` | differs: truncation |  |
| `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` | 49k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `incoai/GLM-5.3-Flash-DFlash2` | 43k | no tokenizer files | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) |  |
| `AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF` | 34k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via dealignai/GLM-5.3-Flash-UNCENSORED-FP8) | gpt2 / glm4 / 154880 |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 34k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `zai-org/GLM-5.3-BF16` | 33k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `DogContext/GLM-5.3-Flash-Uncensored-Q2-ds4` | 30k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via orcarouter/GLM-5.3-Flash-Uncensored-FP8) | gpt2 / glm4 / 154880 |
| `orcarouter/GLM-5.3-Flash-Uncensored-GGUF` | 27k | no tokenizer files | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) |  |
| `pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF` | 26k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) | gpt2 / glm4 / 154880 |
| `nvidia/GLM-5.3-NVFP4` | 24k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `BoldingBuilds/orcarouter_GLM-5.3-Flash-Uncensored-GGUF` | 23k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via orcarouter/GLM-5.3-Flash-Uncensored-FP8) | gpt2 / glm4 / 154880 |
| `aj9o9/GLM-5.3-Flash-GGUF` | 22k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) | gpt2 / glm4 / 154880 |
| `Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2` | 21k | tokenizer.json | `1ca61c4fd0d9` | identical file | gpt2 / glm4 / 154880 |
| `amd/GLM-5.3-Quark-MXFP4-AttnFP8` | 19k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `antirez/glm-5.3-gguf` | 19k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3 (gguf-header)) | gpt2 / glm4 / 154880 |
| `Justvugg/GLM-5.3-colibri-int4-g64` | 18k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers` | 18k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) | gpt2 / glm4 / 154880 |
| `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` | 16k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3-MXFP4` | 4k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `bartowski/GLM-5.3-Flash-BF16-GGUF` | 4k | gguf / weights only | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash-BF16) | - |
| `RedHatAI/GLM-5.3-NVFP4` | 3k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `mlx-community/GLM-5.3-4bit` | 1k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `unsloth/GLM-5.3-Flash-FP8` | 1k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `mlx-community/GLM-5.3-Flash-4bit` | 1k | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `unsloth/GLM-5.3-Flash` | 914 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `mlx-community/GLM-5.3-mixed-4_5bit` | 865 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3-speculator.dspark` | 724 | no tokenizer files | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3) |  |
| `unsloth/GLM-5.3` | 555 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3-Flash` | 363 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` | 317 | no tokenizer files | `1ca61c4fd0d9` | identical file (repo has no tokenizer; via zai-org/GLM-5.3-Flash) |  |
| `RedHatAI/GLM-5.3-Flash-BF16` | 269 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |
| `RedHatAI/GLM-5.3` | 269 | tokenizer.json | `1ca61c4fd0d9` | identical file |  |

**Kimi K3** (`moonshotai/Kimi-K3`)

- The tokenizer is `tokenization_kimi.py` (remote code) over `tiktoken.model`; there is no tokenizer.json, so hf tokenizers cannot load it and the model's own code is the reference (SPEC 0.3).
- `encode()` adds no BOS and recognizes every special token (`allowed_special="all"`): 256 reserved ids after the ranks, named from tokenizer_config.json's `added_tokens_decoder`, else `<|reserved_token_i|>`.
- It cuts the text into 400,000-character chunks, then cuts each chunk wherever a run of whitespace or of non-whitespace (python `str.isspace`) passes 25,000 characters, and runs tiktoken on each piece: pieces never span those cuts, so toks must reproduce them to stay exact on long inputs.
- The pattern is tiktoken (fancy-regex) syntax: `\p{Han}` and class intersection `&&[^\p{Han}]`, which neither the cl100k nor the o200k template expresses.
- tiktoken file `b6c497a7469b33ce`, tokenization_kimi.py `git:5df9d462b43295245084fbab46fd956020d4b84e`, 16 named special tokens in tokenizer_config.json (bos `[BOS]`, eos `[EOS]`).
- 2 repos (21k downloads; `mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-i1-GGUF`, `mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF`): tokenizer `d500bb03c594` (P06: qwen 3.5 / 3.6 / 3.8 (marks)); against the target it differs: another tokenizer kind (tokenizer.json).
- 6 repos (7k downloads; `inference-optimization/Kimi-K3-0.40B-MXFP4`, `catalystsec/Kimi-K3-Audit-451E-MLX-MXFP4`, `RedHatAI/Kimi-K3-NVFP4`, `kernelpool/Kimi-K3-2bit-UVMAX`, ...): tokenizer `664949c0d15d` (P14: kimi k2 / k2.5 / k3 / linear); against the target it differs: special-token names.
- 2 repos (4k downloads; `mradermacher/Kimi-K3-0.40B-i1-GGUF`, `Uniboshi/Kimi-K3-Abliterated-V1`): tokenizer `664949c0d15d` (P14: kimi k2 / k2.5 / k3 / linear); against the target it differs: tokenization_kimi.py, special-token names.

33 other repos (the family's other sizes, the most downloaded repos whose id carries the family name, and the quantizer / mirror orgs unsloth, nvidia, lmstudio-community, mlx-community, bartowski, RedHatAI); 21 load the target's tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).

| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |
|---|---:|---|---|---|---|
| `moonshotai/Kimi-K3` | 1.2M | tiktoken | `664949c0d15d` | target |  |
| `RadixArk/Kimi-K3-DSpark` | 6.4M | no tokenizer files | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3 (name)) |  |
| `unsloth/Kimi-K3-GGUF` | 372k | gguf / weights only | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) | gpt2 / kimi-k2 / 163840 |
| `nvidia/Kimi-K3-NVFP4` | 20k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-i1-GGUF` | 12k | gguf / weights only | `d500bb03c594` | differs: another tokenizer kind (tokenizer.json) (repo has no tokenizer; via khazarai/Qwen3.5-9B-Kimi-k3-Distilled) | gpt2 / qwen35 / 248320 |
| `Inferact/Kimi-K3-DSpark` | 11k | no tokenizer files | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) |  |
| `lightseekorg/kimi-k3-dflash2` | 10k | - | `-` | no tokenizer found |  |
| `mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF` | 9k | gguf / weights only | `d500bb03c594` | differs: another tokenizer kind (tokenizer.json) (repo has no tokenizer; via khazarai/Qwen3.5-9B-Kimi-k3-Distilled) | gpt2 / qwen35 / 248320 |
| `meshllm/Kimi-K3-UD-Q4_K_XL-layers` | 8k | gguf / weights only | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) | gpt2 / kimi-k2 / 163840 |
| `pipenetwork/Kimi-K3-REAP80-MLX-mxfp4-q8` | 6k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8` | 4k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `AMAImedia/Kimi-K3-MXFP-GGUF` | 4k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `inference-optimization/Kimi-K3-0.40B-MXFP4` | 3k | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `SentieAI/Sentie1.0-3B-Claude-Fable-5-GPT5.2-Sol-Kimi-K3-GLM-5.2-GGUF` | 3k | gguf / weights only | `-` | no tokenizer found (repo has no tokenizer; via Nanbeige/Nanbeige-4.1-3B) | llama / default / 166144 |
| `Ryanchen911/Kimi-K3-Uncensored-GGUF` | 3k | gguf / weights only | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) | gpt2 / kimi-k2 / 163840 |
| `RedHatAI/Kimi-K3-speculator.dspark` | 3k | no tokenizer files | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) |  |
| `hellohazime/Kimi-K3-REAP-512GB-GGUF` | 2k | gguf / weights only | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) | gpt2 / kimi-k2 / 163840 |
| `mradermacher/Kimi-K3-0.40B-i1-GGUF` | 2k | gguf / weights only | `664949c0d15d` | differs: tokenization_kimi.py, special-token names (repo has no tokenizer; via Sadatsami/Kimi-K3-0.40B) | gpt2 / kimi-k2 / 163840 |
| `Uniboshi/Kimi-K3-Abliterated-V1` | 2k | tiktoken | `664949c0d15d` | differs: tokenization_kimi.py, special-token names |  |
| `modal-labs/Kimi-K3-DFlash` | 2k | no tokenizer files | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) |  |
| `0xTank/Kimi-K3-IQ1S-REAP568-64K-4XSPARKS` | 2k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) | gpt2 / kimi-k2 / 163840 |
| `mlx-community/Kimi-K3-mlx-reap160-2bit` | 2k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `AtomicChat/Kimi-K3-GGUF` | 1k | gguf / weights only | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) | gpt2 / kimi-k2 / 163840 |
| `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw` | 1k | no tokenizer files | `664949c0d15d` | identical (ranks, code, special names) (repo has no tokenizer; via moonshotai/Kimi-K3) |  |
| `pipenetwork/Kimi-K3-REAP73-zh-code-MLX-mxfp4-q8` | 1k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `catalystsec/Kimi-K3-Audit-451E-MLX-MXFP4` | 1k | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `RedHatAI/Kimi-K3-NVFP4` | 1k | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `kernelpool/Kimi-K3-2bit-UVMAX` | 1k | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `riverclouds/Kimi-K3-8L-dummy` | 1k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `pipenetwork/Kimi-K3-REAPgraded-MLX-mxfp4-q8` | 1k | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `RedHatAI/Kimi-K3-FP8-BLOCK` | 360 | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |
| `unsloth/Kimi-K3` | 314 | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `RedHatAI/Kimi-K3-INT4` | 47 | tiktoken | `664949c0d15d` | differs: special-token names |  |
| `RedHatAI/Kimi-K3` | 34 | tiktoken | `664949c0d15d` | identical (ranks, code, special names) |  |

**gpt-oss** (`openai/gpt-oss-20b`)

- o200k pattern (case-split letter runs with an optional contraction suffix, `\p{N}{1,3}`, punctuation tail `[\r\n/]*`), ignore_merges, 21 special added tokens (harmony format).
- 1 repo (581k downloads; `OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF`): tokenizer `1b7a8d450b12` (P11: o200k (gpt-oss, llama 4)); against the target it differs: added_tokens.
- 1 repo (40k downloads; `mradermacher/OpenAI-gpt-oss-20B-Claude-4.5-Opus-Heretic-Uncensored-i1-GGUF`): tokenizer `fc75aa668195` (P11: o200k (gpt-oss, llama 4)); against the target it differs: padding.
- 1 repo (18k downloads; `OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview`): tokenizer `d280788bf830` (P11: o200k (gpt-oss, llama 4)); against the target it differs: added_tokens.
- 1 repo (840 downloads; `bartowski/kldzj_gpt-oss-120b-heretic-v2-GGUF`): tokenizer `a242eeb0ec00` (P11: o200k (gpt-oss, llama 4)); against the target it differs: padding.

54 other repos (the family's other sizes, the most downloaded repos whose id carries the family name, and the quantizer / mirror orgs unsloth, nvidia, lmstudio-community, mlx-community, bartowski, RedHatAI); 47 load the target's tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).

| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |
|---|---:|---|---|---|---|
| `openai/gpt-oss-20b` | 6.6M | tokenizer.json | `5d3cc21cbb6c` | target |  |
| `openai/gpt-oss-120b` | 4.5M | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF` | 581k | tokenizer.json | `1b7a8d450b12` | differs: added_tokens |  |
| `unsloth/gpt-oss-20b-GGUF` | 488k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-20b) | gpt2 / gpt-4o / 201088 |
| `FastFlowLM/GPT-OSS-20B-NPU2` | 301k | tokenizer.json | `5d3cc21cbb6c` | same tokenizer (file bytes differ) |  |
| `mlx-community/gpt-oss-20b-MXFP4-Q8` | 257k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `ggml-org/gpt-oss-120b-GGUF` | 171k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) | gpt2 / gpt-4o / 201088 |
| `ggml-org/gpt-oss-20b-GGUF` | 130k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-20b) | gpt2 / gpt-4o / 201088 |
| `lmsys/gpt-oss-20b-bf16` | 118k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `MuXodious/gpt-oss-20b-RichardErkhov-heresy` | 114k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `lmstudio-community/gpt-oss-20b-GGUF` | 100k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-20b) | gpt2 / gpt-4o / 201088 |
| `MaziyarPanahi/gpt-oss-20b-Derestricted-GGUF` | 82k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via ArliAI/gpt-oss-20b-Derestricted) | gpt2 / gpt-4o / 201088 |
| `unsloth/gpt-oss-20b-BF16` | 81k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `unsloth/gpt-oss-120b-GGUF` | 73k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) | gpt2 / gpt-4o / 201088 |
| `unsloth/gpt-oss-20b-unsloth-bnb-4bit` | 67k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `lmsys/gpt-oss-120b-bf16` | 67k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `tiny-random/gpt-oss-bf16` | 58k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `nvidia/gpt-oss-120b-Eagle3-v3` | 52k | no tokenizer files | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) |  |
| `lmsys/EAGLE3-gpt-oss-120b-bf16` | 49k | - | `-` | no tokenizer found |  |
| `DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf` | 40k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated) | gpt2 / gpt-4o / 201088 |
| `mradermacher/OpenAI-gpt-oss-20B-Claude-4.5-Opus-Heretic-Uncensored-i1-GGUF` | 40k | gguf / weights only | `fc75aa668195` | differs: padding (repo has no tokenizer; via DavidAU/OpenAI-gpt-oss-20B-Claude-4.5-Opus-Heretic-Uncensored) | gpt2 / gpt-4o / 201088 |
| `openai/gpt-oss-safeguard-20b` | 36k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `lmstudio-community/gpt-oss-120b-MLX-8bit` | 36k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `zhuyksir/EAGLE3-gpt-oss-20b-bf16` | 31k | - | `-` | no tokenizer found |  |
| `DavidAU/OpenAi-GPT-oss-20b-HERETIC-uncensored-NEO-Imatrix-gguf` | 31k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via p-e-w/gpt-oss-20b-heretic) | gpt2 / gpt-4o / 201088 |
| `lmstudio-community/gpt-oss-120b-GGUF` | 21k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) | gpt2 / gpt-4o / 201088 |
| `nvidia/gpt-oss-120b-Eagle3-short-context` | 21k | no tokenizer files | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) |  |
| `OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview` | 18k | tokenizer.json | `d280788bf830` | differs: added_tokens |  |
| `unsloth/gpt-oss-20b` | 18k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/openai_gpt-oss-20b-GGUF` | 15k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-20b) | gpt2 / gpt-4o / 201088 |
| `mlx-community/gpt-oss-20b-OptiQ-4bit` | 12k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/huihui-ai_Huihui-gpt-oss-20b-BF16-abliterated-GGUF` | 11k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated) | gpt2 / gpt-4o / 201088 |
| `lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4` | 10k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/openai_gpt-oss-120b-GGUF` | 9k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) | - |
| `mlx-community/gpt-oss-120b-MXFP4-Q8` | 6k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/p-e-w_gpt-oss-20b-heretic-GGUF` | 5k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via p-e-w/gpt-oss-20b-heretic) | gpt2 / gpt-4o / 201088 |
| `unsloth/gpt-oss-120b-unsloth-bnb-4bit` | 4k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `lmstudio-community/gpt-oss-20b-MLX-8bit` | 4k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `unsloth/gpt-oss-120b-BF16` | 4k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `lmstudio-community/gpt-oss-safeguard-120b-MLX-MXFP4` | 4k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `unsloth/gpt-oss-safeguard-20b-GGUF` | 3k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-safeguard-20b) | gpt2 / gpt-4o / 201088 |
| `RedHatAI/gpt-oss-20b` | 3k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `mlx-community/gpt-oss-20b-MXFP4-Q4` | 3k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `RedHatAI/gpt-oss-20b-speculator.eagle3` | 2k | - | `-` | no tokenizer found |  |
| `unsloth/gpt-oss-120b` | 2k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `nvidia/gpt-oss-120b-Eagle3-long-context` | 2k | no tokenizer files | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) |  |
| `mlx-community/gpt-oss-120b-MXFP4-Q4` | 2k | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/huizimao_gpt-oss-120b-uncensored-bf16-GGUF` | 1k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via huizimao/gpt-oss-120b-uncensored-bf16) | - |
| `lmstudio-community/gpt-oss-safeguard-20b-GGUF` | 1k | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-safeguard-20b) | gpt2 / gpt-4o / 201088 |
| `mlx-community/gpt-oss-120b-4bit` | 989 | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `unsloth/gpt-oss-safeguard-20b` | 904 | tokenizer.json | `5d3cc21cbb6c` | identical file |  |
| `bartowski/kldzj_gpt-oss-120b-heretic-v2-GGUF` | 840 | gguf / weights only | `a242eeb0ec00` | differs: padding (repo has no tokenizer; via kldzj/gpt-oss-120b-heretic-v2) | - |
| `nvidia/gpt-oss-120b-Eagle3-throughput` | 813 | no tokenizer files | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-120b) |  |
| `bartowski/openai_gpt-oss-20b-GGUF-MXFP4-Experimental` | 753 | gguf / weights only | `5d3cc21cbb6c` | identical file (repo has no tokenizer; via openai/gpt-oss-20b) | gpt2 / gpt-4o / 201088 |
| `mlx-community/gpt-oss-120b-mxfp4-bf16` | 716 | tokenizer.json | `5d3cc21cbb6c` | identical file |  |

**Qwen 3.8** (`Qwen/Qwen3.8-27B`)

- The pattern is the hub's qwen 3.5 string (kernels.md section 3), which `src/core/config.c` compiles (TOKS_PATTERNS).
- 19 repos (20.5M downloads; `lmstudio-community/Qwen3.8-27B-MLX-4bit`, `lmstudio-community/Qwen3.8-27B-MLX-8bit`, `lmstudio-community/Qwen3.8-27B-MLX-6bit`, `lmstudio-community/Qwen3.8-27B-MLX-5bit`, ...): tokenizer `902570914a58` (P02: qwen 2 / 2.5 / 3); against the target it differs: pre_tokenizer, decoder.
- 2 repos (1.3M downloads; `empero-ai/Qwen3.8-9B-Distill-GGUF`, `empero-ai/Qwen3.8-4B-Distill-GGUF`): tokenizer `70f961ee7ce2` (P06: qwen 3.5 / 3.6 / 3.8 (marks)); against the target it differs: added_tokens.

54 other repos (the family's other sizes, the most downloaded repos whose id carries the family name, and the quantizer / mirror orgs unsloth, nvidia, lmstudio-community, mlx-community, bartowski, RedHatAI); 33 load the target's tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).

| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |
|---|---:|---|---|---|---|
| `Qwen/Qwen3.8-27B` | 6.9M | tokenizer.json | `d500bb03c594` | target |  |
| `unsloth/Qwen3.8-27B-GGUF` | 6.3M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | - |
| `Qwen/Qwen3.8-27B-FP8` | 5.0M | tokenizer.json | `d500bb03c594` | identical file |  |
| `lmstudio-community/Qwen3.8-27B-MLX-4bit` | 4.2M | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF` | 4.1M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `lmstudio-community/Qwen3.8-27B-MLX-8bit` | 4.1M | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `lmstudio-community/Qwen3.8-27B-MLX-6bit` | 4.0M | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `lmstudio-community/Qwen3.8-27B-MLX-5bit` | 4.0M | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `unsloth/Qwen3.8-27B-NVFP4` | 2.4M | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` | 2.2M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF` | 2.1M | gguf / weights only | `d500bb03c594` | same tokenizer (file bytes differ) (repo has no tokenizer; via DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU) | gpt2 / qwen35 / 248320 |
| `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` | 2.1M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | token list past the first 4 MiB |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF` | 2.0M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` | 1.7M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | - |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | 1.5M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-Flash-Next) | gpt2 / qwen35 / 248320 |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | 1.4M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-Flash-Next) | gpt2 / qwen35 / 248320 |
| `Qwen/Qwen3.8-Flash-Next` | 1.4M | tokenizer.json | `d500bb03c594` | identical file |  |
| `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` | 1.4M | tokenizer.json | `d500bb03c594` | identical file | gpt2 / qwen35 / 248320 |
| `peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF` | 1.3M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `lmstudio-community/Qwen3.8-27B-GGUF` | 1.3M | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `RadixArk/Qwen3.8-27B-NVFP4` | 1.3M | tokenizer.json | `d500bb03c594` | identical file |  |
| `Inferact/Qwen3.8-27B-NVFP4` | 1.2M | tokenizer.json | `d500bb03c594` | identical file |  |
| `ggml-org/Qwen3.8-27B-GGUF` | 950k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `byteshape/Qwen3.8-27B-GGUF` | 765k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | 762k | tokenizer.json | `d500bb03c594` | identical file | gpt2 / qwen35 / 248320 |
| `esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF` | 760k | gguf / weights only | `902570914a58` | differs: pre_tokenizer, decoder (repo has no tokenizer; via unsloth/Qwen3.8-27B-NVFP4) | gpt2 / qwen35 / 248320 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 754k | tokenizer.json | `d500bb03c594` | identical file |  |
| `empero-ai/Qwen3.8-9B-Distill-GGUF` | 673k | gguf / weights only | `70f961ee7ce2` | differs: added_tokens (repo has no tokenizer; via empero-ai/Qwen3.8-9B) | gpt2 / qwen35 / 248320 |
| `empero-ai/Qwen3.8-4B-Distill-GGUF` | 651k | gguf / weights only | `70f961ee7ce2` | differs: added_tokens (repo has no tokenizer; via empero-ai/Qwen3.8-4B) | gpt2 / qwen35 / 248320 |
| `AtomicChat/Qwen3.8-Flash-Next-GGUF` | 651k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-Flash-Next) | - |
| `RedHatAI/Qwen3.8-27B-INT4` | 567k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `nvidia/Qwen3.8-27B-NVFP4` | 557k | tokenizer.json | `d500bb03c594` | identical file |  |
| `bartowski/Qwen3.8-27B-GGUF` | 474k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-27B) | gpt2 / qwen35 / 248320 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 295k | tokenizer.json | `d500bb03c594` | identical file |  |
| `unsloth/Qwen3.8-27B` | 226k | tokenizer.json | `d500bb03c594` | identical file |  |
| `mlx-community/Qwen3.8-27B-4bit` | 177k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `bartowski/orcarouter_Qwen3.8-27B-Uncensored-GGUF` | 153k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via orcarouter/Qwen3.8-27B-Uncensored) | gpt2 / qwen35 / 248320 |
| `mlx-community/Qwen3.8-27B-8bit` | 78k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | 73k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `bartowski/Qwen3.8-Flash-Next-GGUF` | 59k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-Flash-Next) | - |
| `RedHatAI/Qwen3.8-27B-NVFP4` | 38k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `bartowski/ukisai_Swift-Qwen3.8-27b-GGUF` | 36k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via ukisai/Swift-Qwen3.8-27b) | gpt2 / qwen35 / 248320 |
| `unsloth/Qwen3.8-27B-FP8` | 24k | tokenizer.json | `d500bb03c594` | identical file |  |
| `mlx-community/Qwen3.8-27B-MTP-4bit` | 22k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `lmstudio-community/Qwen3.8-Flash-Next-GGUF` | 19k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-Flash-Next) | gpt2 / qwen35 / 248320 |
| `mlx-community/Qwen3.8-27B-Uncensored-OptiQ-4bit` | 15k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `mlx-community/Qwen3.8-27B-bf16` | 11k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `bartowski/ukisai_Swift-1.5-Qwen3.8-27b-GGUF` | 10k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via ukisai/Swift-1.5-Qwen3.8-27b) | gpt2 / qwen35 / 248320 |
| `mlx-community/Qwen3.8-Flash-Next-4bit` | 9k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `unsloth/Qwen3.8-2.4T-A95B-GGUF` | 8k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via Qwen/Qwen3.8-2.4T-A95B) | gpt2 / qwen35 / 248320 |
| `mlx-community/Qwen3.8-27B-OptiQ-4bit` | 6k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `mlx-community/Qwen3.8-27B-MTP-8bit` | 5k | tokenizer.json | `902570914a58` | differs: pre_tokenizer, decoder |  |
| `bartowski/ukisai_Swift-Qwen3.8-Flash-Next-GGUF` | 4k | gguf / weights only | `902570914a58` | differs: pre_tokenizer, decoder (repo has no tokenizer; via ukisai/Swift-Qwen3.8-Flash-Next) | - |
| `bartowski/JetBrains-Qwen3.8-3.6-27B-blend-GGUF` | 4k | gguf / weights only | `d500bb03c594` | identical file (repo has no tokenizer; via JetBrains/Qwen3.8-3.6-27B-blend) | gpt2 / qwen35 / 248320 |
| `bartowski/bytkim_Qwen3.8-27B-pi-GGUF` | 4k | gguf / weights only | `902570914a58` | differs: pre_tokenizer, decoder (repo has no tokenizer; via bytkim/Qwen3.8-27B-pi) | gpt2 / qwen35 / 248320 |

**Gemma 4** (`google/gemma-4-26B-A4B-it`)

- Sentencepiece-style BPE: byte_fallback with `<0x00>`..`<0xFF>`, unk `<unk>` with fuse_unk. The normalizer turns every ' ' into U+2581 before the pre-tokenizer's `Split(' ', MergedWithPrevious)`, so the split never fires and each segment between added tokens is one BPE piece. The post-processor adds no BOS (single template `$A`).
- 2 repos (1.2M downloads; `google/gemma-4-31B`, `google/gemma-4-E4B`): tokenizer `4795adc15fae` (P08: Split(' ') after Replace(' ' -&gt; U+2581): no split); against the target it differs: post_processor.
- 1 repo (207k downloads; `RedHatAI/gemma-4-26B-A4B-it-NVFP4`): tokenizer `a2a30274603c` (P08: Split(' ') after Replace(' ' -&gt; U+2581): no split); against the target it differs: truncation.
- 1 repo (152k downloads; `unsloth/gemma-4-26B-A4B-it-NVFP4`): tokenizer `09ffd0f7c4b0` (P08: Split(' ') after Replace(' ' -&gt; U+2581): no split); against the target it differs: truncation.

54 other repos (the family's other sizes, the most downloaded repos whose id carries the family name, and the quantizer / mirror orgs unsloth, nvidia, lmstudio-community, mlx-community, bartowski, RedHatAI); 50 load the target's tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).

| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |
|---|---:|---|---|---|---|
| `google/gemma-4-26B-A4B-it` | 13.0M | tokenizer.json | `22d7d15144bd` | target |  |
| `google/gemma-4-31B-it` | 9.9M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `google/gemma-4-E4B-it` | 4.4M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `google/gemma-4-E2B-it` | 3.0M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `google/gemma-4-12B-it` | 1.9M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `nvidia/Gemma-4-31B-IT-NVFP4` | 1.6M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `unsloth/gemma-4-12b-it-GGUF` | 1.5M | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-12B-it) | gemma4 / - / 262144 |
| `litert-community/gemma-4-E2B-it-litert-lm` | 1.4M | no tokenizer files | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E2B-it) |  |
| `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` | 1.4M | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-e4b-it) | gemma4 / - / 262144 |
| `nvidia/Gemma-4-26B-A4B-NVFP4` | 1.2M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E4B-it-MLX-4bit` | 1.1M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E4B-it-MLX-8bit` | 1.1M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `ggml-org/gemma-4-E4B-it-GGUF` | 1.1M | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E4B-it) | gemma4 / - / 262144 |
| `lmstudio-community/gemma-4-E4B-it-MLX-6bit` | 1.1M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E4B-it-MLX-5bit` | 1.1M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` | 1.1M | tokenizer.json | `22d7d15144bd` | identical file |  |
| `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | 898k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `google/gemma-4-12B-it-qat-q4_0-gguf` | 802k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-12B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 798k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `google/gemma-4-E4B-it-qat-q4_0-gguf` | 741k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E4B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` | 698k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-12B-it) | gemma4 / - / 262144 |
| `google/gemma-4-31B` | 638k | tokenizer.json | `4795adc15fae` | differs: post_processor |  |
| `unsloth/gemma-4-E4B-it-GGUF` | 615k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E4B-it) | gemma4 / - / 262144 |
| `unsloth/gemma-4-12B-it-qat-GGUF` | 610k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-12B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `google/gemma-4-E4B` | 561k | tokenizer.json | `4795adc15fae` | differs: post_processor |  |
| `unsloth/gemma-4-26B-A4B-it-qat-GGUF` | 530k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `unsloth/gemma-4-26B-A4B-it-GGUF` | 514k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-26B-A4B-it) | gemma4 / - / 262144 |
| `unsloth/gemma-4-E4B-it-qat-GGUF` | 476k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E4B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `google/gemma-4-E2B-it-qat-q4_0-gguf` | 458k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E2B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `lmstudio-community/gemma-4-E4B-it-GGUF` | 449k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E4B-it) | gemma4 / - / 262144 |
| `unsloth/gemma-4-E2B-it-qat-GGUF` | 414k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E2B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `unsloth/gemma-4-31B-it-qat-GGUF` | 398k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-31B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `RedHatAI/gemma-4-31B-it-FP8-block` | 393k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `unsloth/gemma-4-E2B-it-GGUF` | 386k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E2B-it) | gemma4 / - / 262144 |
| `bartowski/google_gemma-4-E2B-it-GGUF` | 358k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-E2B-it) | gemma4 / - / 262144 |
| `lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit` | 356k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `unsloth/gemma-4-31B-it-GGUF` | 322k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-31B-it) | gemma4 / - / 262144 |
| `lmstudio-community/gemma-4-12B-it-MLX-4bit` | 307k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-12B-it-MLX-8bit` | 294k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-12B-it-MLX-6bit` | 290k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-12B-it-MLX-5bit` | 289k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-12B-it-QAT-GGUF` | 259k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-12B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |
| `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` | 235k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `RedHatAI/gemma-4-31B-it-FP8-dynamic` | 226k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | 207k | tokenizer.json | `a2a30274603c` | differs: truncation |  |
| `lmstudio-community/gemma-4-E2B-it-MLX-4bit` | 195k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E2B-it-MLX-8bit` | 187k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E2B-it-MLX-5bit` | 186k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-E2B-it-MLX-6bit` | 186k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit` | 157k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `unsloth/gemma-4-26B-A4B-it-NVFP4` | 152k | tokenizer.json | `09ffd0f7c4b0` | differs: truncation |  |
| `lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit` | 151k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-26B-A4B-it-MLX-6bit` | 150k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-26B-A4B-it-MLX-5bit` | 149k | tokenizer.json | `22d7d15144bd` | identical file |  |
| `lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF` | 127k | gguf / weights only | `22d7d15144bd` | identical file (repo has no tokenizer; via google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) | gemma4 / - / 262144 |

## 1. Headline

**Today toks loads 98.40% of (b) by downloads** (text-generation 96.99%, embedding / reranker 99.29%; (d) image-text 99.03%; (c) named families 99.99%) at `d0b927a418` (section 9). SPEC 1.5 keeps tokenizer inputs beyond SPEC 0.3's list out (slow files alone: a sentencepiece .model, vocab.txt, vocab.json + merges.txt; remote code without a tiktoken file); they take 1.42% of (b) (2.54% of text-gen, 0.71% of embed) and count as misses here. With them out of the denominator (the maintainer's call, not made yet) toks loads **99.82% of (b)** (text-gen 99.52%, embed 100.00%). Section 8 lists every miss with what blocks it.

**On compiled fast paths (SPEC 1.4's gate: >= 99% of census downloads): 98.00% of (b)** (99.40% without SPEC 1.5's inputs; text-generation 95.93%, embedding / reranker 99.29%). The rest of what loads, 0.41% of (b), runs its pre-tokenizer on the generic engine (docs/algorithms/generic.md: exact, onig's semantics for the census's constructs, no compiled template): `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, `deepseek-ai/deepseek-coder-7b-instruct-v1.5`, `XHToken/Spark-X2.5-4B-GGUF`, `tiiuae/falcon-7b`, `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF`, `abenzerps/Spark-X2.5-4B-GGUF`, `deepseek-ai/DeepSeek-V2-Lite`, `skt/A.X-K2-NVFP4`, `XHToken/Spark-X2.5-1.7B-GGUF`, `deepseek-ai/deepseek-coder-6.7b-instruct`, `chatdb/natural-sql-7b`, `trl-internal-testing/tiny-Cohere2ForCausalLM` and 2 more.

Every critical target loads today (section 0), so no order needs a step 0. The work packages still missing, added in this order from today, reach **99.73% of (b) with package 0**. Order: the package that unlocks the most downloads per unit of work first (a template, algorithm, normalizer, encode option or input format is one unit; a decoder, a post-processor mapping or a compiler spelling a quarter; a refusal of the load path in components other tokenizers already load (`code:`) one); a model counts only when every component its tokenizer uses is covered. (b) counts every download, so its order is led by the embedding models; the text-generation order follows this table.

| # | package | adds | unlocks (b) | cumulative (b) | text-gen | embed | tokenizers (count) | (d) vlm | example models |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| today | toks `d0b927a418` (loads) | | | 98.40% | 96.99% | 99.29% | 87.50% | 99.03% | |
| 0 | **slow-file inputs (vocab.json + merges.txt, vocab.txt, sentencepiece .model)** | `input:slow-bpe`, `input:slow-sentencepiece`, `input:slow-wordpiece` | +1.32% | 99.73% | 99.40% | 99.93% | 93.75% | 99.03% | `facebook/opt-125m`, `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `sshleifer/tiny-gpt2` |
| 1 | load path: tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | `code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` | +0.02% | 99.75% | 99.45% | 99.93% | 94.32% | 99.03% | `yujiepan/kimi-linear-tiny-random` |
| 2 | load path: pre_tokenizer Split (one that can cut) | `code:pre_tokenizer Split (one that can cut)` | +0.02% | 99.77% | 99.50% | 99.93% | 94.89% | 99.03% | `sarvamai/sarvam-30b` |
| 3 | byte-level BPE with a partial byte alphabet | `bpe:byte-fallback+bytelevel` | +0.02% | 99.78% | 99.54% | 99.93% | 95.45% | 99.03% | `LSX-UniWue/LLaMmlein_1B_prerelease` |
| 4 | other algorithms (plain BPE with &lt;/w&gt;, WordLevel) | `bpe:end_of_word_suffix`, `dec:BPEDecoder`, `model:bpe-plain` | +0.02% | 99.80% | 99.60% | 99.93% | 96.02% | 99.03% | `openai-community/openai-gpt` |

**Text-generation order.** The same greedy over the same packages, weighted by the text-generation half of (b) alone (433.6M downloads, 500 models); the (b) order's first package, slow-file inputs (vocab.json + merges.txt, vocab.txt, sentencepiece .model), unlocks +1.32% of (b) but +2.41% of text-gen. Today 96.99%; the packages reach **99.40% of text-gen with package 0**; 6 unreadable text-gen models cap it at 99.60%. Packages no text-generation model needs are left out; `census/coverage.json` `packages_textgen` has the rows.

| # | package | adds | unlocks text-gen | cumulative text-gen | (d) vlm | (b) | example text-gen models |
|---:|---|---|---:|---:|---:|---:|---|
| today | toks `d0b927a418` (loads) | | | 96.99% | 99.03% | 98.40% | |
| 0 | **slow-file inputs (vocab.json + merges.txt, vocab.txt, sentencepiece .model)** | `input:slow-bpe`, `input:slow-sentencepiece` | +2.41% | 99.40% | 99.03% | 99.33% | `facebook/opt-125m`, `sshleifer/tiny-gpt2`, `rinna/japanese-gpt-neox-small` |
| 1 | load path: tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | `code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` | +0.05% | 99.45% | 99.03% | 99.35% | `yujiepan/kimi-linear-tiny-random` |
| 2 | load path: pre_tokenizer Split (one that can cut) | `code:pre_tokenizer Split (one that can cut)` | +0.05% | 99.50% | 99.03% | 99.37% | `sarvamai/sarvam-30b` |
| 3 | byte-level BPE with a partial byte alphabet | `bpe:byte-fallback+bytelevel` | +0.04% | 99.54% | 99.03% | 99.39% | `LSX-UniWue/LLaMmlein_1B_prerelease` |
| 4 | other algorithms (plain BPE with &lt;/w&gt;, WordLevel) | `bpe:end_of_word_suffix`, `dec:BPEDecoder`, `model:bpe-plain` | +0.05% | 99.60% | 99.03% | 99.41% | `openai-community/openai-gpt` |

**Each package alone.** What one package unlocks from today's coverage with nothing else added (no step 0): the attribution the orders cannot show. A package whose tokenizers also need another package's features unlocks little alone, which is why the orders add them together; the one-off pattern templates are in `census/coverage.json` `packages_alone`.

| package | features | (b) | text-gen | embed | (d) vlm |
|---|---|---:|---:|---:|---:|
| slow-file inputs (vocab.json + merges.txt, vocab.txt, sentencepiece .model) | `input:slow-bpe`, `input:slow-sentencepiece`, `input:slow-wordpiece` | +1.32% | +2.41% | +0.64% | +0.00% |
| other algorithms (plain BPE with &lt;/w&gt;, WordLevel) | `bpe:end_of_word_suffix`, `dec:BPEDecoder`, `model:bpe-plain`, `model:wordlevel` | +0.02% | +0.05% | +0.00% | +0.00% |
| load path: tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | `code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` | +0.02% | +0.05% | +0.00% | +0.00% |
| load path: pre_tokenizer Split (one that can cut) | `code:pre_tokenizer Split (one that can cut)` | +0.02% | +0.05% | +0.00% | +0.00% |
| byte-level BPE with a partial byte alphabet | `bpe:byte-fallback+bytelevel` | +0.02% | +0.04% | +0.00% | +0.00% |
| load path: model.vocab ids not dense | `code:model.vocab ids not dense` | +0.00% | +0.00% | +0.00% | +0.00% |
| load path: tiktoken model: tokenizer_config.json: an added token outside the wrapper's 256 special ids | `code:tiktoken model: tokenizer_config.json: an added token outside the wrapper's 256 special ids` | +0.00% | +0.00% | +0.00% | +0.00% |

Ceiling: 7 unreadable models cap (b) at 99.80%; without the slow-file package (outside SPEC 0.3's input list today) the ceiling is 98.42%, so **SPEC 1.1's >= 99% of (b) by downloads needs slow-file inputs in scope** (or those models out of the census). By count of distinct tokenizers (SPEC 1.1 asks for both) the tail is long: every one-off pattern is a tokenizer, so 99% by count needs the generic engine for certification while the fast-path gate (SPEC 1.4) is by downloads.

Feature names: `tmpl:*` a K3 scanner template (`tmpl:Pnn` a template for one pattern of section 3), `model:*` an algorithm, `bpe:*` a model flag, `norm:*` / `pretok:*` / `dec:*` / `pp:*` SPEC 1.3 components, `added:*` added-token options, `file:*` tokenizer.json fields hf applies inside `encode()`, `compile:*` spellings the compiler must accept as equal to a covered form (no new kernel), `input:*` a tokenizer source other than tokenizer.json. Section 5 defines each one and ranks it on its own.

Robustness: the same order computed without CI / testing repos (trl-internal-testing, hf-internal-testing, peft-internal-testing, ...) starts with `input:slow-bpe`, `input:slow-wordpiece`, `input:slow-sentencepiece`, `code:pre_tokenizer Split (one that can cut)`, `bpe:byte-fallback+bytelevel`; (c)'s named families are 99.99% covered by downloads after the packages that reach 99% (section 7 lists each family's needs).

## 2. The same order, feature by feature

The greedy behind section 1 at the finest grain: each step adds the missing features of one tokenizer group (features that only ever occur together form one step), most downloads per unit of work first. Step 0 crosses 99%.

| # | adds | unlocks (b) | cumulative (b) | text-gen | example models |
|---:|---|---:|---:|---:|---|
| 0 | `input:slow-bpe` | +0.77% | 99.17% | 98.99% | `facebook/opt-125m`, `sshleifer/tiny-gpt2` |
| 1 | `input:slow-wordpiece` | +0.39% | 99.57% | 98.99% | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `nvidia/dragon-multiturn-query-encoder` |
| 2 | `input:slow-sentencepiece` | +0.16% | 99.73% | 99.40% | `rinna/japanese-gpt-neox-small`, `Voicelab/vlt5-base-keywords` |
| 3 | `code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` | +0.02% | 99.75% | 99.45% | `yujiepan/kimi-linear-tiny-random` |
| 4 | `code:pre_tokenizer Split (one that can cut)` | +0.02% | 99.77% | 99.50% | `sarvamai/sarvam-30b` |
| 5 | `bpe:byte-fallback+bytelevel` | +0.02% | 99.78% | 99.54% | `LSX-UniWue/LLaMmlein_1B_prerelease` |
| 6 | `bpe:end_of_word_suffix`, `dec:BPEDecoder`, `model:bpe-plain` | +0.02% | 99.80% | 99.60% | `openai-community/openai-gpt` |

## 3. Pre-tokenizer patterns, ranked

One row per distinct pre-tokenizer chain; the exact strings are in appendix A by id. status: `covered` = a kernels.md section 3 string in the spelling toks compiles; `compile` = covered semantics in another spelling the compiler must accept; `params` = the cl100k template's existing parameters express it; `new` = needs a new template (or the generic engine, which SPEC 1.4 does not count as a fast path).

| id | template | variant | status | tokenizers | models (b) | share (b) | examples |
|---|---|---|---|---:|---:|---:|---|
| P01 | bert | BertPreTokenizer | new | 24 | 60 | 45.40% | `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5` |
| P02 | cl100k | qwen 2 / 2.5 / 3 | covered | 26 | 178 | 19.54% | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-8B` |
| P03 | metaspace | WhitespaceSplit &gt; Metaspace(prepend=always, split=true) | new | 7 | 16 | 7.88% | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `intfloat/multilingual-e5-large` |
| P04 | cl100k | gpt-2 (ByteLevel use_regex) | covered | 18 | 33 | 4.52% | `openai-community/gpt2`, `facebook/opt-125m`, `ibm-granite/granite-embedding-small-english-r2` |
| P05 | metaspace | Metaspace(prepend=always, split=true) | new | 5 | 5 | 4.47% | `BAAI/bge-m3`, `intfloat/multilingual-e5-small`, `BAAI/bge-reranker-large` |
| P06 | cl100k | qwen 3.5 / 3.6 / 3.8 (marks) | covered | 4 | 52 | 3.89% | `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.8-27B` |
| P07 | cl100k | cl100k / llama 3 | covered | 16 | 67 | 3.66% | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `zai-org/GLM-5.3-Flash` |
| P08 | none | Split(' ') after Replace(' ' -&gt; U+2581): no split | compile | 10 | 25 | 1.78% | `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`, `google/gemma-4-E4B-it` |
| P09 | deepseek-v3 | deepseek v3 / r1 / v3.1 / v3.2 / v4 | new | 10 | 16 | 1.54% | `deepseek-ai/DeepSeek-V4-Flash-0731`, `deepseek-ai/DeepSeek-V3.2`, `deepseek-ai/DeepSeek-OCR` |
| P10 | none | no pre-tokenizer (one piece per segment) | covered | 20 | 30 | 1.38% | `dphn/dolphin-2.9.1-yi-1.5-34b`, `llava-hf/llava-1.5-7b-hf`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| P11 | o200k | o200k (gpt-oss, llama 4) | new | 8 | 7 | 1.14% | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `unsloth/Inkling-Small-GGUF` |
| P12 | o200k | o200k no contractions, digits 1, punct tail [\r\n/]* | new | 7 | 20 | 1.10% | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` |
| P13 | digits-gpt2 | Digits(individual) &gt; ByteLevel(gpt-2 regex) | new | 7 | 13 | 0.70% | `HuggingFaceTB/SmolLM2-135M-Instruct`, `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| P14 | kimi | kimi k2 / k2.5 / k3 / linear | new | 3 | 6 | 0.67% | `RadixArk/Kimi-K3-DSpark`, `moonshotai/Kimi-K3`, `unsloth/Kimi-K3-GGUF` |
| P15 | cl100k | cl100k / llama 3 | compile | 5 | 10 | 0.31% | `ibm-granite/granite-4.1-3b`, `allenai/Olmo-3-7B-Instruct`, `microsoft/phi-4` |
| P16 | other | Split[Isolated](re:"\\p{N}{1,3}") &gt; Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 7 | 0.30% | `openbmb/MiniCPM5-2B`, `openbmb/MiniCPM5-2B-DSpark`, `openbmb/MiniCPM5-1B` |
| P17 | o200k | o200k (gpt-oss, llama 4) | new | 5 | 7 | 0.29% | `MiniMaxAI/MiniMax-M2.7`, `ai-sage/Giga-Embeddings-instruct`, `MiniMaxAI/MiniMax-M2.5` |
| P18 | none | Metaspace(prepend=first, split=false): no split | compile | 3 | 7 | 0.28% | `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-7B-Instruct-v0.2`, `mistralai/Mistral-7B-v0.1` |
| P19 | other | Split[Isolated](re:"[\r\n]") &gt; Split[Isolated](re:"\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ...") &gt; Split[Isolated](re:"\\s?[!-/:-~！-／：-～‘-‟　-。]+") &gt; Split[Isolated](re:"\\s+$") &gt; Split[Isolated](re:"[一-龥ࠀ-一가-퟿]+") &gt; Digits(individual=1) &gt; ByteLevel(regex=0,prefix_space=0) | new | 4 | 6 | 0.22% | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, `deepseek-ai/deepseek-coder-7b-instruct-v1.5`, `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` |
| P20 | other | Split[MergedWithNext](re:"(?:\\r?\\n)+(?!\\r?\\n)") &gt; Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 2 | 8 | 0.15% | `poolside/Laguna-S-2.1-NVFP4`, `poolside/Laguna-XS.2`, `poolside/Laguna-XS-2.1-GGUF` |
| P21 | other | Split[Isolated](re:" ?[^(\\s\|[.,!?…。，、।۔،])]+") &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 2 | 0.14% | `bigscience/bloomz-560m`, `bigscience/bloom-560m` |
| P22 | other | Split[Isolated](re:"'(?i:[sdmt]\|ll\|ve\|re)...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 3 | 4 | 0.10% | `GSAI-ML/LLaDA-8B-Instruct`, `AtomicChat/Ling-3.0-flash-GGUF`, `bloomer010/Ling-3.0-tiny-GGUF` |
| P23 | other | Split[Isolated](re:"\\p{N}{1,3}") &gt; Split[Isolated](re:"[一-龥぀-ゟ゠-ヿ]+") &gt; Split[Isolated](re:"[!\"#$%&'()*+,\\-./:;&lt;=...") &gt; Digits(individual=1) &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 4 | 0.10% | `XHToken/Spark-X2.5-4B-GGUF`, `abenzerps/Spark-X2.5-4B-GGUF`, `XHToken/Spark-X2.5-1.7B-GGUF` |
| P24 | none | Metaspace(prepend=never, split=false): no split | compile | 3 | 4 | 0.10% | `cl-nagoya/ruri-v3-310m`, `MaziyarPanahi/Yi-Coder-9B-Chat-GGUF`, `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` |
| P25 | other | Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 1 | 0.04% | `LeaderboardModel1/zeta-2.1-autoround-W4A16` |
| P26 | other | Punctuation(Contiguous) &gt; ByteLevel(regex=1,prefix_space=0) &gt; Digits(individual=0) &gt; Split[Isolated](re:"[0-9][0-9][0-9]") | new | 1 | 1 | 0.03% | `tiiuae/falcon-7b` |
| P27 | none | Split(' ') after Replace(' ' -&gt; U+2581): no split | compile | 1 | 1 | 0.02% | `sarvamai/sarvam-30b` |
| P28 | other | Split[Isolated](re:"[一-龥぀-ゟ゠-ヿ]+") &gt; Split[Isolated](re:"(?i:'s\|'t\|'re\|'ve\|'m\|...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 1 | 0.02% | `skt/A.X-K2-NVFP4` |
| P29 | other | Split[Isolated](re:"[\r\n]") &gt; Split[Isolated](re:"\\s?\\p{L}+") &gt; Split[Isolated](re:"\\s?\\p{P}+") &gt; Split[Isolated](re:"[一-龥ࠀ-一가-퟿]+") &gt; Digits(individual=1) &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 1 | 0.02% | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| P30 | other | Split[Isolated](re:"\\d{1,3}(?=(?:\\d{3})*\\b)") &gt; Split[Isolated](re:"[^\\r\\n\\p{L}\\p{N}]?[\\p...") &gt; ByteLevel(regex=0,prefix_space=0) | new | 1 | 1 | 0.02% | `trl-internal-testing/tiny-Cohere2ForCausalLM` |
| P31 | other | Split[Isolated](re:".") | new | 1 | 0 | 0.00% | `datalab-to/surya-ocr-2`, `datalab-to/surya-ocr-2-gguf` |

## 4. Templates, summed

| template | patterns | statuses | tokenizers | models (b) | share (b) |
|---|---|---|---:|---:|---:|
| bert | P01 | new | 24 | 60 | 45.40% |
| cl100k | P02, P04, P06, P07, P15 | compile, covered | 69 | 340 | 31.92% |
| metaspace | P03, P05 | new | 12 | 21 | 12.36% |
| none | P08, P10, P18, P24, P27 | compile, covered | 37 | 67 | 3.56% |
| o200k | P11, P12, P17 | new | 20 | 34 | 2.52% |
| deepseek-v3 | P09 | new | 10 | 16 | 1.54% |
| other | P16, P19, P20, P21, P22, P23, P25, P26, ... | new | 18 | 36 | 1.14% |
| digits-gpt2 | P13 | new | 7 | 13 | 0.70% |
| kimi | P14 | new | 3 | 6 | 0.67% |

## 5. Components, ranked

Every feature some tokenizer needs beyond today's coverage, with the (b) share of the tokenizers using it (a tokenizer needs all of its features, so shares overlap; sections 1-2 give the order).

| feature | category | what | tokenizers | models (b) | share (b) | examples |
|---|---|---|---:|---:|---:|---|
| `input:slow-bpe` | input | only vocab.json + merges.txt (transformers builds the byte-level BPE) | 3 | 5 | 0.77% | `facebook/opt-125m`, `sshleifer/tiny-gpt2`, `microsoft/DialoGPT-medium` |
| `input:slow-wordpiece` | input | only vocab.txt (transformers builds the WordPiece tokenizer) | 3 | 4 | 0.39% | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, `nvidia/dragon-multiturn-query-encoder`, `nvidia/dragon-multiturn-context-encoder` |
| `input:slow-sentencepiece` | input | only a sentencepiece .model (transformers converts it) | 5 | 6 | 0.16% | `rinna/japanese-gpt-neox-small`, `Voicelab/vlt5-base-keywords`, `XLabs-AI/xflux_text_encoders` |
| `bpe:end_of_word_suffix` | model | BPE end_of_word_suffix (&lt;/w&gt;) | 1 | 1 | 0.02% | `openai-community/openai-gpt` |
| `dec:BPEDecoder` | decoder | BPEDecoder (&lt;/w&gt; suffix) | 1 | 1 | 0.02% | `openai-community/openai-gpt` |
| `model:bpe-plain` | model | BPE over chars without byte-level or byte fallback (unk, end_of_word_suffix) | 1 | 1 | 0.02% | `openai-community/openai-gpt` |
| `code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it` | load path | toks's load path refuses: tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | 1 | 1 | 0.02% | `yujiepan/kimi-linear-tiny-random`, `Uniboshi/Kimi-K3-Abliterated-V1` |
| `code:pre_tokenizer Split (one that can cut)` | load path | toks's load path refuses: pre_tokenizer Split (one that can cut) | 1 | 1 | 0.02% | `sarvamai/sarvam-30b` |
| `bpe:byte-fallback+bytelevel` | model | byte_fallback set on a byte-level BPE | 1 | 1 | 0.02% | `LSX-UniWue/LLaMmlein_1B_prerelease` |
| `model:wordlevel` | model | WordLevel (whole-piece lookup) | 1 | 0 | 0.00% | `datalab-to/surya-ocr-2`, `datalab-to/surya-ocr-2-gguf` |
| `tmpl:P31` | template | a template for pattern P31 (Split[Isolated](re:".")) | 1 | 0 | 0.00% | `datalab-to/surya-ocr-2`, `datalab-to/surya-ocr-2-gguf` |
| `code:model.vocab ids not dense` | load path | toks's load path refuses: model.vocab ids not dense | 2 | 0 | 0.00% | `Xenova/gpt-4`, `Xenova/gpt-4o` |
| `code:tiktoken model: tokenizer_config.json: an added token outside the wrapper's 256 special ids` | load path | toks's load path refuses: tiktoken model: tokenizer_config.json: an added token outside the wrapper's 256 special ids | 1 | 0 | 0.00% | `inference-optimization/Kimi-K3-0.40B-MXFP4`, `mradermacher/Kimi-K3-0.40B-i1-GGUF`, `catalystsec/Kimi-K3-Audit-451E-MLX-MXFP4` |

Components in use that today's coverage already handles (counts over all distinct tokenizers): byte-level BPE 122; pp TemplateProcessing 108; pp ByteLevel 76; norm NFC 42; added tokens with normalized=true 35; ignore_merges 29.

## 6. Distinct tokenizers

200 distinct tokenizers across (b), (c), (d) and the critical-target repos (169 in (b); id = first 12 hex of the sha-256 of hf 0.23.2's normalized configuration). The 60 with the most (b) downloads; `census/coverage.json` `tokenizers` has every one with its full signature (model flags, normalizer / pre-tokenizer / decoder chains, post-processor, added-token option counts, truncation, padding).

| tokenizer | algorithm | pattern | models | share (b) | needs beyond today | most downloaded model |
|---|---|---|---:|---:|---|---|
| `138bd678dcd0` | wordpiece | P01 | 5 | 22.05% | nothing (covered) | `sentence-transformers/all-MiniLM-L6-v2` |
| `3efa0e94a985` | wordpiece | P01 | 25 | 18.58% | nothing (covered) | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `a9c253b323d3` | bpe-bytelevel | P02 | 75 | 9.27% | nothing (covered) | `Qwen/Qwen3-0.6B` |
| `dd8c154de3b6` | bpe-bytelevel | P02 | 56 | 6.36% | nothing (covered) | `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` |
| `649fe51ccdb9` | unigram | P03 | 3 | 5.55% | nothing (covered) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `7cd74b54072d` | unigram | P05 | 1 | 3.09% | nothing (covered) | `BAAI/bge-m3` |
| `70f961ee7ce2` | bpe-bytelevel | P06 | 50 | 2.47% | nothing (covered) | `Qwen/Qwen3.5-9B` |
| `1e18cbe58ee9` | bpe-bytelevel | P07 | 31 | 2.25% | nothing (covered) | `meta-llama/Llama-3.2-1B-Instruct` |
| `5ff6fd2bd616` | unigram | P03 | 8 | 2.15% | nothing (covered) | `intfloat/multilingual-e5-large` |
| `8cab9f128c5f` | bpe-bytelevel | P04 | 7 | 1.78% | nothing (covered) | `openai-community/gpt2` |
| `621dd603b963` | wordpiece | P01 | 1 | 1.75% | nothing (covered) | `sentence-transformers/all-mpnet-base-v2` |
| `d500bb03c594` | bpe-bytelevel | P06 | 55 | 1.39% | nothing (covered) | `Qwen/Qwen3.8-27B` |
| `5d3cc21cbb6c` | bpe-bytelevel | P11 | 49 | 1.14% | nothing (covered) | `openai/gpt-oss-20b` |
| `8d502ad89c54` | bpe-bytelevel | P02 | 3 | 1.08% | nothing (covered) | `Qwen/Qwen3-Embedding-0.6B` |
| `9d9e4878827a` | unigram | P05 | 1 | 1.02% | nothing (covered) | `intfloat/multilingual-e5-small` |
| `902570914a58` | bpe-bytelevel | P02 | 37 | 1.00% | nothing (covered) | `lmstudio-community/Qwen3.8-27B-MLX-4bit` |
| `6b4def62afcc` | bpe-bytelevel | P12 | 14 | 0.96% | nothing (covered) | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| `cf4cfab77830` | bpe-bytelevel | P09 | 6 | 0.76% | nothing (covered) | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| `1ca61c4fd0d9` | bpe-bytelevel | P07 | 54 | 0.75% | nothing (covered) | `zai-org/GLM-5.3-Flash` |
| `22d7d15144bd` | bpe-byte-fallback | P08 | 59 | 0.73% | nothing (covered) | `google/gemma-4-26B-A4B-it` |
| `664949c0d15d` | bpe-bytelevel(tiktoken ranks) | P14 | 26 | 0.65% | nothing (covered) | `RadixArk/Kimi-K3-DSpark` |
| `1b99cea2673c` | bpe-bytelevel | P04 | 2 | 0.63% | input:slow-bpe | `facebook/opt-125m` |
| `633e543f7611` | bpe-bytelevel | P04 | 8 | 0.60% | nothing (covered) | `EleutherAI/pythia-160m` |
| `86398199810a` | wordpiece | P01 | 3 | 0.55% | nothing (covered) | `datasocietyco/bge-base-en-v1.5-course-recommender-v5` |
| `c99276a2b627` | bpe-bytelevel | P02 | 3 | 0.48% | nothing (covered) | `Qwen/Qwen3-Embedding-8B` |
| `d800841edf64` | bpe-bytelevel | P04 | 2 | 0.47% | nothing (covered) | `ibm-granite/granite-embedding-small-english-r2` |
| `2741f1152569` | bpe-byte-fallback | P10 | 1 | 0.42% | nothing (covered) | `dphn/dolphin-2.9.1-yi-1.5-34b` |
| `4131558d7634` | bpe-bytelevel | P13 | 6 | 0.42% | nothing (covered) | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `fcceb9bd867f` | wordpiece | P01 | 1 | 0.41% | nothing (covered) | `BAAI/bge-small-zh-v1.5` |
| `69817adb19bb` | bpe-byte-fallback | P08 | 7 | 0.38% | nothing (covered) | `google/gemma-3-1b-it` |
| `863c57ac1739` | bpe-byte-fallback | P08 | 2 | 0.37% | nothing (covered) | `google/embeddinggemma-300m` |
| `1cc2ffdb5ec2` | bpe-bytelevel | P09 | 1 | 0.32% | nothing (covered) | `deepseek-ai/DeepSeek-V3.2` |
| `f9fb7adedfd1` | bpe-bytelevel | P02 | 2 | 0.31% | nothing (covered) | `CMSManhattan/JiRackUltra_1b` |
| `3f8acdc5bca4` | bpe-bytelevel | P16 | 7 | 0.30% | nothing (covered) | `openbmb/MiniCPM5-2B` |
| `842086ded6bb` | bpe-bytelevel | P04 | 2 | 0.29% | nothing (covered) | `sentence-transformers/all-distilroberta-v1` |
| `af77ec8edc3b` | bpe-bytelevel | P02 | 9 | 0.28% | nothing (covered) | `Qwen/Qwen2-1.5B-Instruct` |
| `51ae50a68b7b` | bpe-bytelevel | P09 | 3 | 0.26% | nothing (covered) | `deepseek-ai/DeepSeek-V3-0324` |
| `4ce95143ff04` | wordpiece | P01 | 4 | 0.25% | nothing (covered) | `shibing624/text2vec-base-chinese` |
| `869257236fc2` | bpe-byte-fallback | P10 | 4 | 0.25% | nothing (covered) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `bc9c684947ab` | bpe-byte-fallback | P18 | 5 | 0.23% | nothing (covered) | `mistralai/Mistral-7B-Instruct-v0.2` |
| `9f72c96c7c37` | wordpiece | P01 | 2 | 0.22% | nothing (covered) | `sentence-transformers/paraphrase-mpnet-base-v2` |
| `5afe8f4134af` | unigram | P05 | 1 | 0.21% | nothing (covered) | `BAAI/bge-reranker-large` |
| `3efa0e94a985~wordpiece-vocab-only` | wordpiece | P01 | 2 | 0.21% | input:slow-wordpiece | `nvidia/dragon-multiturn-query-encoder` |
| `19a6af204069` | bpe-byte-fallback | P08 | 7 | 0.20% | nothing (covered) | `google/gemma-2-9b-it` |
| `7e2ee938df13` | wordpiece | P01 | 2 | 0.19% | nothing (covered) | `sentence-transformers/distiluse-base-multilingual-cased-v1` |
| `19954f5f0817` | bpe-bytelevel | P02 | 4 | 0.18% | nothing (covered) | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `5e5000be8cd4` | wordpiece | P01 | 2 | 0.17% | nothing (covered) | `Alibaba-NLP/gte-large-en-v1.5` |
| `f36cfa048a19` | bpe-bytelevel | P04 | 1 | 0.17% | nothing (covered) | `Alibaba-NLP/gte-reranker-modernbert-base` |
| `7d27765e3990` | bpe-bytelevel | P17 | 3 | 0.16% | nothing (covered) | `MiniMaxAI/MiniMax-M2.7` |
| `1b91164eb009` | bpe-bytelevel | P07 | 4 | 0.16% | nothing (covered) | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `bc29a7ce654d` | bpe-bytelevel | P07 | 3 | 0.15% | nothing (covered) | `LiquidAI/LFM2.5-2.6B-GGUF` |
| `17a63a0eaf0d` | bpe-bytelevel | P21 | 2 | 0.14% | nothing (covered) | `bigscience/bloomz-560m` |
| `cdd7bb8c22a8` | bpe-bytelevel | P15 | 5 | 0.13% | nothing (covered) | `ibm-granite/granite-4.1-3b` |
| `3623b6ca2665` | wordpiece | P01 | 1 | 0.13% | input:slow-wordpiece | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` |
| `e40dccfa827c` | bpe-byte-fallback | P10 | 3 | 0.13% | nothing (covered) | `trl-internal-testing/tiny-random-LlamaForCausalLM` |
| `5b756c4ddcd7` | wordpiece | P01 | 1 | 0.13% | nothing (covered) | `sentence-transformers/multi-qa-mpnet-base-dot-v1` |
| `af230e3333bb` | bpe-bytelevel | P13 | 2 | 0.12% | nothing (covered) | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ` |
| `0043b879a2ab` | bpe-bytelevel | P04 | 1 | 0.12% | nothing (covered) | `laion/clap-htsat-unfused` |
| `14e20c653674` | bpe-bytelevel(tiktoken ranks) | P02 | 1 | 0.12% | nothing (covered) | `Qwen/Qwen-72B` |
| `1c5c8f3c3ed9` | bpe-byte-fallback | P10 | 6 | 0.12% | nothing (covered) | `microsoft/Phi-3.5-mini-instruct` |

## 7. Named families (SPEC 1.1(c))

| family | repo | tokenizer | pattern | template / variant | needs beyond today |
|---|---|---|---|---|---|
| gpt-2 / r50k | `openai-community/gpt2` | `8cab9f128c5f` | P04 | cl100k / gpt-2 (ByteLevel use_regex) | nothing (covered) |
| cl100k | `Xenova/gpt-4` | `823f52bd0734` | P15 | cl100k / cl100k / llama 3 | code:model.vocab ids not dense |
| o200k | `Xenova/gpt-4o` | `b3be9cbb4bf2` | P17 | o200k / o200k (gpt-oss, llama 4) | code:model.vocab ids not dense |
| gpt-oss (o200k harmony) | `openai/gpt-oss-20b` | `5d3cc21cbb6c` | P11 | o200k / o200k (gpt-oss, llama 4) | nothing (covered) |
| llama 2 | `meta-llama/Llama-2-7b-hf` (gated; byte-identical copy `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) | `869257236fc2` | P10 | none / no pre-tokenizer (one piece per segment) | nothing (covered) |
| llama 3 | `meta-llama/Meta-Llama-3-8B` (gated; byte-identical copy `IlyaGusev/saiga_llama3_8b`) | `1b91164eb009` | P07 | cl100k / cl100k / llama 3 | nothing (covered) |
| llama 3.3 | `meta-llama/Llama-3.3-70B-Instruct` (gated; byte-identical copy `RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic`) | `1e18cbe58ee9` | P07 | cl100k / cl100k / llama 3 | nothing (covered) |
| llama 4 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` (gated; byte-identical copy `RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16`) | `e24a02090c29` | P11 | o200k / o200k (gpt-oss, llama 4) | nothing (covered) |
| qwen 2 | `Qwen/Qwen2-7B-Instruct` | `af77ec8edc3b` | P02 | cl100k / qwen 2 / 2.5 / 3 | nothing (covered) |
| qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | `dd8c154de3b6` | P02 | cl100k / qwen 2 / 2.5 / 3 | nothing (covered) |
| qwen 3 | `Qwen/Qwen3-8B` | `a9c253b323d3` | P02 | cl100k / qwen 2 / 2.5 / 3 | nothing (covered) |
| qwen 3.5 | `Qwen/Qwen3.5-9B` | `70f961ee7ce2` | P06 | cl100k / qwen 3.5 / 3.6 / 3.8 (marks) | nothing (covered) |
| qwen 3.6 | `Qwen/Qwen3.6-35B-A3B` | `70f961ee7ce2` | P06 | cl100k / qwen 3.5 / 3.6 / 3.8 (marks) | nothing (covered) |
| qwen 3.8 | `Qwen/Qwen3.8-27B` | `d500bb03c594` | P06 | cl100k / qwen 3.5 / 3.6 / 3.8 (marks) | nothing (covered) |
| deepseek v3 | `deepseek-ai/DeepSeek-V3` | `51ae50a68b7b` | P09 | deepseek-v3 / deepseek v3 / r1 / v3.1 / v3.2 / v4 | nothing (covered) |
| deepseek r1 | `deepseek-ai/DeepSeek-R1` | `232addf8641e` | P09 | deepseek-v3 / deepseek v3 / r1 / v3.1 / v3.2 / v4 | nothing (covered) |
| deepseek v4 | `deepseek-ai/DeepSeek-V4-Flash` | `cf4cfab77830` | P09 | deepseek-v3 / deepseek v3 / r1 / v3.1 / v3.2 / v4 | nothing (covered) |
| kimi k2 | `moonshotai/Kimi-K2-Instruct` | `664949c0d15d` | P14 | kimi / kimi k2 / k2.5 / k3 / linear | nothing (covered) |
| glm 4 | `zai-org/glm-4-9b-chat-hf` | `9dd9ddae10b0` | P07 | cl100k / cl100k / llama 3 | nothing (covered) |
| glm 4.5-4.7 | `zai-org/GLM-4.7-Flash` | `1ca61c4fd0d9` | P07 | cl100k / cl100k / llama 3 | nothing (covered) |
| glm 5 | `zai-org/GLM-5` | `1ca61c4fd0d9` | P07 | cl100k / cl100k / llama 3 | nothing (covered) |
| nemotron 3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `6b4def62afcc` | P12 | o200k / o200k no contractions, digits 1, punct tail [\r\n/]* | nothing (covered) |
| olmo 2 | `allenai/OLMo-2-1124-7B-Instruct` | `4761df9ebf53` | P15 | cl100k / cl100k / llama 3 | nothing (covered) |
| olmo 3 | `allenai/Olmo-3-7B-Instruct` | `056200c1500d` | P15 | cl100k / cl100k / llama 3 | nothing (covered) |
| phi-3 | `microsoft/Phi-3-mini-4k-instruct` | `1c5c8f3c3ed9` | P10 | none / no pre-tokenizer (one piece per segment) | nothing (covered) |
| phi-4 | `microsoft/phi-4` | `b11c48240446` | P15 | cl100k / cl100k / llama 3 | nothing (covered) |
| mistral v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `efc14fcf5577` | P18 | none / Metaspace(prepend=first, split=false): no split | nothing (covered) |
| mistral tekken | `mistralai/Mistral-Nemo-Instruct-2407` | `8b94cfa677dd` | P12 | o200k / o200k no contractions, digits 1, punct tail [\r\n/]* | nothing (covered) |
| gemma 1 | `google/gemma-7b`: gated, no byte-identical copy; stand-in `unsloth/gemma-2b` (not verified identical) | `c7743828d146` | - | none / no pre-tokenizer (one piece per segment) | bpe:partial-byte-fallback, dec:ByteFallback, dec:Fuse, dec:Replace, model:bpe-byte-fallback, norm:Replace |
| gemma 2 | `google/gemma-2-9b` (gated; byte-identical copy `unsloth/gemma-2-9b-it`) | `19a6af204069` | P08 | none / Split(' ') after Replace(' ' -&gt; U+2581): no split | nothing (covered) |
| gemma 3 | `google/gemma-3-4b-it` (gated; byte-identical copy `QCRI/Fanar-2-27B-Instruct`) | `69817adb19bb` | P08 | none / Split(' ') after Replace(' ' -&gt; U+2581): no split | nothing (covered) |
| gemma 4 | `google/gemma-4-26B-A4B-it` | `22d7d15144bd` | P08 | none / Split(' ') after Replace(' ' -&gt; U+2581): no split | nothing (covered) |
| modernbert | `answerdotai/ModernBERT-base` | `d800841edf64` | P04 | cl100k / gpt-2 (ByteLevel use_regex) | nothing (covered) |
| bert | `google-bert/bert-base-uncased` | `3efa0e94a985` | P01 | bert / BertPreTokenizer | nothing (covered) |

The openai encodings have no hf repo of their own: r50k_base is represented by `openai-community/gpt2`'s tokenizer.json; cl100k_base is represented by `Xenova/gpt-4`'s tokenizer.json; o200k_base is represented by `Xenova/gpt-4o`'s tokenizer.json. toks also takes the `.tiktoken` file with its pattern (SPEC 0.3). Against tiktoken 0.14.0's own pattern strings (`tiktoken_ext/openai_public.py`, `coverage.py tiktoken`): o200k_base, o200k_harmony equal the census string exactly; r50k_base, cl100k_base are spelled differently from their hub tokenizer.json, so a `.tiktoken` input with tiktoken's string needs its own exact-string entry (or a proof that both split alike).

| encoding | census string (hub tokenizer.json) | tiktoken's string (json) |
|---|---|---|
| r50k_base | P04 (`openai-community/gpt2`) | `"'(?:[sdmt]\|ll\|ve\|re)\| ?\\p{L}++\| ?\\p{N}++\| ?[^\\s\\p{L}\\p{N}]++\|\\s++$\|\\s+(?!\\S)\|\\s"` |
| cl100k_base | P15 (`Xenova/gpt-4`) | `"'(?i:[sdmt]\|ll\|ve\|re)\|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}++\|\\p{N}{1,3}+\| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+\|\\s++$\|\\s*[\\r\\n]\|\\s+(?!\\S)\|\\s"` |
| o200k_base | P17 (`Xenova/gpt-4o`) | identical |
| o200k_harmony | P11 (`openai/gpt-oss-20b`) | identical |

## 8. Misses

Every model in (b) that toks does not load today: 26 models, 1.60% of (b), grouped by what blocks them (the load path's own refusal: `toks_load`'s diag on the file, or on the model directory for a tiktoken model; else why no tokenizer could be read), most downloads first. 19 of them (1.42% of (b)) ship an input SPEC 1.5 keeps out. `needs` = the census features of their tokenizers that no loading tokenizer uses yet (section 5), or `code:` the load path's refusal where every feature is already in use elsewhere. Shares of text-gen / embed are of those slices.

| blocked by | (b) | text-gen | embed | needs | models (downloads) |
|---|---:|---:|---:|---|---|
| bpe-vocab-merges-only (SPEC 1.5: input out of scope; its converted tokenizer.json loads) | 0.77% | 2.00% | 0.00% | input:slow-bpe | `facebook/opt-125m` (6.9M), `sshleifer/tiny-gpt2` (1.3M), `microsoft/DialoGPT-medium` (217k), `erwanf/gpt2-mini` (146k), `facebook/opt-350m` (145k) |
| wordpiece-vocab-only (SPEC 1.5: input out of scope; its converted tokenizer.json loads) | 0.34% | 0.00% | 0.55% | input:slow-wordpiece | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` (1.5M), `nvidia/dragon-multiturn-query-encoder` (1.2M), `nvidia/dragon-multiturn-context-encoder` (1.2M) |
| sentencepiece-only (SPEC 1.5: input out of scope; its converted tokenizer.json loads) | 0.16% | 0.41% | 0.00% | input:slow-sentencepiece | `rinna/japanese-gpt-neox-small` (516k), `Voicelab/vlt5-base-keywords` (421k), `XLabs-AI/xflux_text_encoders` (310k), `MaziyarPanahi/Yi-Coder-9B-Chat-GGUF` (215k), `sbintuitions/sarashina2.2-0.5b-instruct-v0.1` (159k), `MaziyarPanahi/Yi-Coder-1.5B-Chat-GGUF` (159k) |
| no readable tokenizer: no tokenizer files and no base model / name match | 0.08% | 0.21% | 0.00% | - | `apple/OpenELM-1_1B-Instruct` (906k) |
| wordpiece-vocab-only (SPEC 1.5: input out of scope; its converted tokenizer.json is refused: model.vocab ids not dense) | 0.06% | 0.00% | 0.09% | input:slow-wordpiece | `YituTech/conv-bert-base` (636k) |
| wordpiece-vocab-only (SPEC 1.5: input out of scope; no converted tokenizer.json) | 0.04% | 0.00% | 0.07% | - | `bkai-foundation-models/vietnamese-bi-encoder` (485k) |
| remote code only (SPEC 1.5: input out of scope) | 0.03% | 0.09% | 0.00% | - | `LongSafari/hyenadna-medium-450k-seqlen-hf` (222k), `legraphista/glm-4-9b-chat-IMat-GGUF` (160k) |
| normalizer BertNormalizer | 0.02% | 0.05% | 0.00% | bpe:end_of_word_suffix, dec:BPEDecoder, model:bpe-plain | `openai-community/openai-gpt` (226k) |
| tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | 0.02% | 0.05% | 0.00% | code:tiktoken model: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | `yujiepan/kimi-linear-tiny-random` (223k) |
| pre_tokenizer Split (one that can cut) | 0.02% | 0.05% | 0.00% | code:pre_tokenizer Split (one that can cut) | `sarvamai/sarvam-30b` (217k) |
| model.vocab lacks byte-level alphabet chars (with byte_fallback) | 0.02% | 0.04% | 0.00% | bpe:byte-fallback+bytelevel | `LSX-UniWue/LLaMmlein_1B_prerelease` (182k) |
| sentencepiece-only (SPEC 1.5: input out of scope; no converted tokenizer.json) | 0.02% | 0.04% | 0.00% | - | `allenai/wildguard` (171k) |
| no readable tokenizer: base-missing-401 | 0.01% | 0.03% | 0.00% | - | `MaziyarPanahi/Llama-3-8B-Instruct-32k-v0.1-GGUF` (148k) |
| no readable tokenizer: gated, no byte-identical ungated copy found | 0.01% | 0.03% | 0.00% | - | `google/gemma-2b` (144k) |

## 9. Findings for the rest of toks

- **Whole-segment BPE.** gemma 2 / 3 / 4 (P08, P27) normalize ' ' to U+2581 before `Split(' ', MergedWithPrevious)`, so the split never fires; Metaspace(split=false) files (P18, P24: `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-7B-Instruct-v0.2`, `cl-nagoya/ruri-v3-310m`) replace and prepend without splitting; llama 2 / phi-3 / mistral v0.1 style files (P10) have no pre-tokenizer at all. In all of them (37 tokenizers, 3.56% of (b)) each segment between added tokens is a single BPE piece as long as the input: K6 needs its heap path for long pieces and K5's piece caches never hit.
- `ignore_merges` is set on 29 tokenizers (5.80% of (b): llama 3.x, gpt-oss, glm 4-5, nemotron, mistral nemo, ...); TOKS_TF_IGNORE_MERGES covers it.
- **tiktoken spells r50k_base, cl100k_base differently from the hub.** tiktoken 0.14.0's pattern strings for them are not the strings of their hub tokenizer.json (P04 `openai-community/gpt2`, P15 `Xenova/gpt-4`: tiktoken's strings in section 7, the hub's in appendix A), while o200k_base, o200k_harmony match exactly. config.c compares pattern strings exactly, so a `.tiktoken` reader (SPEC 0.3) that hands it tiktoken's string needs that spelling in the table too, with a proof (or an enumeration) that it splits like the hub string.
- kimi k2 / k2.5 / k3 / linear (and the RadixArk k3 draft) load one tiktoken file (`tiktoken.model`, 163584 ranks) with one pattern from `tokenization_kimi.py`: `[\p{Han}]+` first, then o200k-style letter runs written with class intersection (`[\p{Lu}...\p{M}&&[^\p{Han}]]`): the o200k template's kimi parameters; the special tokens live in tokenizer_config.json, not in the tiktoken file, and the repos' wrappers and configs differ (section 8 lists the ones the kimi reader refuses).
- **Qwen 3.8: family repos with another tokenizer.** 19 repos (20.5M downloads, e.g. `lmstudio-community/Qwen3.8-27B-MLX-4bit`, `lmstudio-community/Qwen3.8-27B-MLX-8bit`, `lmstudio-community/Qwen3.8-27B-MLX-6bit`) ship tokenizer `902570914a58` (P02: qwen 2 / 2.5 / 3); against `Qwen/Qwen3.8-27B` it differs: pre_tokenizer, decoder, so they split or merge some text differently from the target. Parity suites must pin the exact file they test.
- **Qwen 3.8: family repos with another tokenizer.** 2 repos (1.3M downloads, e.g. `empero-ai/Qwen3.8-9B-Distill-GGUF`, `empero-ai/Qwen3.8-4B-Distill-GGUF`) ship tokenizer `70f961ee7ce2` (P06: qwen 3.5 / 3.6 / 3.8 (marks)); against `Qwen/Qwen3.8-27B` it differs: added_tokens, so their added tokens differ from the target's. Parity suites must pin the exact file they test.
- **Gemma 4: family repos with another tokenizer.** 2 repos (1.2M downloads, e.g. `google/gemma-4-31B`, `google/gemma-4-E4B`) ship tokenizer `4795adc15fae` (P08: Split(' ') after Replace(' ' -> U+2581): no split); against `google/gemma-4-26B-A4B-it` it differs: post_processor, so `encode()` adds different special tokens around the text. Parity suites must pin the exact file they test.
- 124 models in (b) (4.41%) carry no tokenizer and resolve through their card's base_model (gguf quantizations, adapters); 10 more (1.97%) name no base and resolve by the same model name without the packaging suffix (gguf repos also by the gguf header; their `n_tokens` is recorded in coverage.json). Without them the denominators shrink and the order does not change.
- 17 models in (b) (1.38%) ship only slow files (facebook/opt, DialoGPT, rinna, t5 variants, vocab.txt-only BERTs). The census analyzed the fast tokenizer transformers builds from them. SPEC 1.5 keeps these inputs out; section 8 says whether each conversion would load.


**The code today.** `tools/census/probe.c` runs `toks_load` (src/core/load.c: the config parse or the tiktoken reader, compile, the split check, the algorithm's tables, the decode tables, the tier) at commit `d0b927a418` (library sources as of `d0b927a`) on gb10e at 2026-10-04T12:47:54Z over every distinct tokenizer.json (226 files, the conversions of slow inputs included) and every distinct tiktoken model directory (12: ranks file, tokenization_*.py and tokenizer_config.json under the repo's own names); each file also loads through `toks_load_mem_copy`, and its config parse alone names the stage (no disagreement). Slow-input conversions are in section 8.

| load path result | tokenizers | share (b) |
|---|---:|---:|
| loads | 179 | 97.63% |
| tiktoken model directory: loads | 2 | 0.77% |
| config refuses: normalizer BertNormalizer | 1 | 0.02% |
| tiktoken model directory: load refuses: tiktoken.model needs tokenizer_config.json and tokenization_kimi.py beside it | 1 | 0.02% |
| config refuses: pre_tokenizer Split (one that can cut) | 1 | 0.02% |
| config refuses: model.vocab lacks byte-level alphabet chars (with byte_fallback) | 1 | 0.02% |
| config refuses: model type (m1a: BPE) | 1 | 0.00% |
| config refuses: model.vocab ids not dense | 2 | 0.00% |
| tiktoken model directory: load refuses: tokenizer_config.json: an added token outside the wrapper's 256 special ids | 1 | 0.00% |

## 10. Method

- **Population.** Anonymous hub API, `GET /api/models?pipeline_tag=<tag>&sort=downloads&direction=-1`. (b): text-generation top 500; embedding / reranker: the union of sentence-similarity, feature-extraction, text-ranking and `filter=reranker` listings in download order, the first 100 that carry a text tokenizer (the non-text repos skipped on the way are in coverage.json `models` without a slice). (d): image-text-to-text and any-to-any, top 100 together. (c): one canonical repo per family (section 7). `downloads` is the hub's 30-day count when listed.
- **Which tokenizer a model loads.** tokenizer.json at the repo root (else one directory down); else tekken.json, a tiktoken file (pattern parsed from tokenization_*.py), remote code only, or slow files. A repo with none of them (gguf quantizations, adapters, speculative-decoding drafts) follows its card's base_model chain, then the gguf header's `general.base_model.*.repo_url`, then the same model name without its packaging suffix (-GGUF, -DSpark, -FP8, ...); coverage.json `via` records the route. Slow-file repos get the fast tokenizer transformers 5.18 builds from them (AutoTokenizer, no remote code).
- **Gated repos.** The hub's blob listing gives the file's content key (git blob sha-1 or lfs sha-256) without access; the census uses an ungated repo whose file has that key (byte-identical, verified by hashing the download). Without one the model is listed as gated; for a named family a same-name ungated repo is shown as a labelled stand-in, never counted.
- **Dedupe.** hf tokenizers 0.23.2 `Tokenizer.from_str(file).to_str()`, json with sorted keys, `version` dropped, sha-256. Files hf reads identically hash identically (old `"a b"` vs new `["a", "b"]` merges, gpt-2's untyped model, Metaspace's `add_prefix_space` vs `prepend_scheme`). Every file in the census loads in hf 0.23.2 (coverage.json `hf_loads`).
- **Signature.** From the normalized form: model type and flags (byte_fallback, ignore_merges, unk_token, fuse_unk, dropout, continuing_subword_prefix, end_of_word_suffix, byte-alphabet / byte-token completeness), normalizer chain (Precompiled charsmaps by hash), pre-tokenizer chain with the exact regex strings, decoder chain, post-processor (template shapes and ids), added-token option counts, truncation, padding.
- **Covered today.** toks's own load path accepts the model's files (section 9). A census feature (template, algorithm, normalizer and other components, model flag, added-token or encode option, input format; rules in `tools/census/report.py`) counts as done when a loading tokenizer uses it; a refused tokenizer needs its features nobody loads yet, or, when every one is in use elsewhere, the load path's refusal (`code:`). The orders of sections 1-2 start from there.
- **Critical targets.** For each of section 0's five: the exact repo, then hub searches for the family stem (the 30 most downloaded repos whose id carries it, plus every such repo of unsloth, nvidia, lmstudio-community, mlx-community, bartowski and RedHatAI). Each repo goes through the same resolution; gguf repos also get their header read (llama.cpp's `tokenizer.ggml.model` / `.pre` / token count). A repo matches the target when its file has the same bytes or hf's normalized configuration hashes the same; otherwise the differing top-level parts are named (model, added_tokens, normalizer, pre_tokenizer, post_processor, decoder, truncation, padding). tiktoken repos compare the ranks file, tokenization_*.py and tokenizer_config.json's special names.
- **The code today.** `tools/census/probe.c` links the library and calls `toks_load` on every tokenizer.json the census read (slow-input conversions included) and on one model directory per distinct tiktoken model (the repo's ranks file, tokenization_*.py and tokenizer_config.json under their own names; coverage.py fetches each distinct tokenizer_config.json once and keeps it only when its sha-256 equals the one recorded at census time). Each file also loads through `toks_load_mem_copy` and its config parse alone names the stage; a disagreement is stage 8.
- **Order.** Greedy budgeted coverage: step 0 is everything the critical targets' own tokenizers need, whatever it unlocks; then each step adds the missing features of one tokenizer group, the group whose features unlock the most downloads per unit of work (template, algorithm, normalizer, pre-tokenizer component, model flag, added-token option, encode option, input format: 1; decoder, post-processor mapping, compiler spelling: 1/4). Section 1 runs the same greedy over work packages (fixed groups of features, `PACKAGES` in report.py: a feature belongs to the package whose tokenizers spell with it, e.g. gemma's no-op Split and mistral's Metaspace(split=false) to the sentencepiece-style BPE stack), once weighted by (b) downloads and once by text-generation downloads alone, and shows what each package unlocks alone.
- **Caveats.** Downloads include CI traffic (testing repos are flagged; section 1 shows the order without them). Name-matched gguf / draft repos are inferences (gguf `n_tokens` recorded for checking). The `compile` equivalences are argued from the regexes, not yet proven by enumeration (SPEC 5.3 style); the generic engine counts as coverage (it is exact) but not as a fast path (SPEC 1.4; section 1 gives both). The census sees each file as hf does; transformers-only behaviour (tokenizer_config.json's add_bos_token, chat templates) is outside it.
- **An earlier, unreviewed census, corrected here.** It used `Xenova/gpt-4o` (o200k) for cl100k, reported kimi k2 and glm 4 as missing (both exist: tiktoken + remote code, and `zai-org/glm-4-9b-chat-hf`), called gpt-2's untyped model `unknown`, labelled the qwen 2 pattern `gpt2-family`, deduped by file bytes before hf's normalization, counted gguf-only repos (8 of the top text-generation models) as misses, and transcribed o200k_base's pattern without the '/' of `[\r\n/]*` (its snapshot.json `tiktoken_fallbacks`, found by the o200k template work; never merged). This census copies no pattern by hand: tokenizer.json strings are hf's own re-serialization, tiktoken strings come from tiktoken itself (section 7).

Re-run (network phases on a lab host, maintainer doctrine): `tools/remote.sh <host> 'for p in collect critical resolve fetch; do uv run tools/census/coverage.py $p; done && uv run --with transformers --with sentencepiece --with protobuf tools/census/coverage.py convert && uv run tools/census/coverage.py analyze && uv run --with tiktoken tools/census/coverage.py tiktoken && CENSUS_COMMIT=<sha> uv run tools/census/coverage.py probe && uv run tools/census/coverage.py state'` (state under `$TOKS_CENSUS_CACHE`, default `~/.cache/toks/census`; CENSUS_COMMIT names the commit, remote.sh syncs without .git), copy `coverage/state.json` back, then `uv run tools/census/coverage.py report --state state.json` writes census/coverage.json and docs/coverage.md. A code-only refresh re-runs just `tiktoken`, `probe` and `state` over the same files.

## Appendix A: exact pre-tokenizer chains

Each chain as hf 0.23.2 re-serializes it (json; `trim_offsets` dropped: it changes offsets, never ids). Regex strings are json string literals: the regex the engine sees is the decoded string.

**P01** (bert, new; 24 tokenizers, 45.40% of (b); e.g. `sentence-transformers/all-MiniLM-L6-v2`)

```json
{"type": "BertPreTokenizer"}
```

**P02** (cl100k, covered; 26 tokenizers, 19.54% of (b); e.g. `Qwen/Qwen3-0.6B`)

```json
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P03** (metaspace, new; 7 tokenizers, 7.88% of (b); e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)

```json
{"type": "WhitespaceSplit"}
{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}
```

**P04** (cl100k, covered; 18 tokenizers, 4.52% of (b); e.g. `openai-community/gpt2`)

```json
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
```

**P05** (metaspace, new; 5 tokenizers, 4.47% of (b); e.g. `BAAI/bge-m3`)

```json
{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}
```

**P06** (cl100k, covered; 4 tokenizers, 3.89% of (b); e.g. `Qwen/Qwen3.5-9B`)

```json
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P07** (cl100k, covered; 16 tokenizers, 3.66% of (b); e.g. `meta-llama/Llama-3.2-1B-Instruct`)

```json
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P08** (none, compile; 10 tokenizers, 1.78% of (b); e.g. `google/gemma-4-26B-A4B-it`)

```json
{"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false}
```

**P09** (deepseek-v3, new; 10 tokenizers, 1.54% of (b); e.g. `deepseek-ai/DeepSeek-V4-Flash-0731`)

```json
{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P10** (none, covered; 20 tokenizers, 1.38% of (b); e.g. `dphn/dolphin-2.9.1-yi-1.5-34b`)

```json
```

**P11** (o200k, new; 8 tokenizers, 1.14% of (b); e.g. `openai/gpt-oss-20b`)

```json
{"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P12** (o200k, new; 7 tokenizers, 1.10% of (b); e.g. `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`)

```json
{"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P13** (digits-gpt2, new; 7 tokenizers, 0.70% of (b); e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`)

```json
{"type": "Digits", "individual_digits": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
```

**P14** (kimi, new; 3 tokenizers, 0.67% of (b); e.g. `RadixArk/Kimi-K3-DSpark`)

```json
{"type": "Split", "pattern": {"Regex": "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P15** (cl100k, compile; 5 tokenizers, 0.31% of (b); e.g. `ibm-granite/granite-4.1-3b`)

```json
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Removed", "invert": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P16** (other, new; 1 tokenizers, 0.30% of (b); e.g. `openbmb/MiniCPM5-2B`)

```json
{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P17** (o200k, new; 5 tokenizers, 0.29% of (b); e.g. `MiniMaxAI/MiniMax-M2.7`)

```json
{"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Removed", "invert": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P18** (none, compile; 3 tokenizers, 0.28% of (b); e.g. `mistralai/Mistral-7B-Instruct-v0.3`)

```json
{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}
```

**P19** (other, new; 4 tokenizers, 0.22% of (b); e.g. `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`)

```json
{"type": "Split", "pattern": {"Regex": "[\r\n]"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "\\s?[!-/:-~！-／：-～‘-‟　-。]+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "\\s+$"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[一-龥ࠀ-一가-퟿]+"}, "behavior": "Isolated", "invert": false}
{"type": "Digits", "individual_digits": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P20** (other, new; 2 tokenizers, 0.15% of (b); e.g. `poolside/Laguna-S-2.1-NVFP4`)

```json
{"type": "Split", "pattern": {"Regex": "(?:\\r?\\n)+(?!\\r?\\n)"}, "behavior": "MergedWithNext", "invert": false}
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P21** (other, new; 1 tokenizers, 0.14% of (b); e.g. `bigscience/bloomz-560m`)

```json
{"type": "Split", "pattern": {"Regex": " ?[^(\\s|[.,!?…。，、।۔،])]+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P22** (other, new; 3 tokenizers, 0.10% of (b); e.g. `GSAI-ML/LLaDA-8B-Instruct`)

```json
{"type": "Split", "pattern": {"Regex": "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P23** (other, new; 1 tokenizers, 0.10% of (b); e.g. `XHToken/Spark-X2.5-4B-GGUF`)

```json
{"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "Digits", "individual_digits": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P24** (none, compile; 3 tokenizers, 0.10% of (b); e.g. `cl-nagoya/ruri-v3-310m`)

```json
{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": false}
```

**P25** (other, new; 1 tokenizers, 0.04% of (b); e.g. `LeaderboardModel1/zeta-2.1-autoround-W4A16`)

```json
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P26** (other, new; 1 tokenizers, 0.03% of (b); e.g. `tiiuae/falcon-7b`)

```json
{"type": "Punctuation", "behavior": "Contiguous"}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
{"type": "Digits", "individual_digits": false}
{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Isolated", "invert": false}
```

**P27** (none, compile; 1 tokenizers, 0.02% of (b); e.g. `sarvamai/sarvam-30b`)

```json
{"type": "Split", "pattern": {"Regex": " "}, "behavior": "MergedWithPrevious", "invert": false}
```

**P28** (other, new; 1 tokenizers, 0.02% of (b); e.g. `skt/A.X-K2-NVFP4`)

```json
{"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P29** (other, new; 1 tokenizers, 0.02% of (b); e.g. `deepseek-ai/deepseek-coder-6.7b-instruct`)

```json
{"type": "Split", "pattern": {"Regex": "[\r\n]"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "\\s?\\p{L}+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "\\s?\\p{P}+"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[一-龥ࠀ-一가-퟿]+"}, "behavior": "Isolated", "invert": false}
{"type": "Digits", "individual_digits": true}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P30** (other, new; 1 tokenizers, 0.02% of (b); e.g. `trl-internal-testing/tiny-Cohere2ForCausalLM`)

```json
{"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated", "invert": false}
{"type": "Split", "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}
{"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
```

**P31** (other, new; 1 tokenizers, 0.00% of (b); e.g. `datalab-to/surya-ocr-2`)

```json
{"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false}
```
