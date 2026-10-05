#!/usr/bin/env python3
# tools/corpora/fetch_tokenizers.py — pin-verified tokenizer.json download (SPEC §0.6 references).
#
# Pins (urls + sha256): the TOKENIZERS table below, every file pinned by hub revision and sha256.
# Files land in ~/.cache/toks/tokenizers/<name> where python/toks_oracle and the
# parity runner expect them (TOKS_TOKENIZER_CACHE overrides the root).
#
# Usage (uv, PEP 723):
#   uv run tools/corpora/fetch_tokenizers.py            # gpt2 + llama3 (the default)
#   uv run tools/corpora/fetch_tokenizers.py --all      # every pinned tokenizer
#   uv run tools/corpora/fetch_tokenizers.py --root /path
#
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.31"]
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import requests

TOKENIZERS = {
    "gpt2": {
        "url": "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
        "sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    },
    "llama3": {
        "url": "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/5a8abab4a5d6f164389b1079fb721cfab8d7126c/tokenizer.json",
        "sha256": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    },
    "qwen3": {
        "url": "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json",
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    # openai/gpt-oss-20b: pinned to a revision (main moved nothing since 2025-08-04, but a moving ref can turn a
    # critical target's pin into a mismatch on any fresh cache)
    "o200k": {
        "url": "https://huggingface.co/openai/gpt-oss-20b/resolve/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json",
        "sha256": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    },
    # mistralai/Mistral-Nemo-Instruct-2407 = Mistral-Nemo-Base-2407 @ a4477a2f (9,264,445 bytes): tests/c/test_e2e.c's
    # "mistral-nemo" (tekken); anonymous
    "mistral-nemo": {
        "url": "https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407/resolve/04d8a90549d23fc6bd7f642064003592df51e9b3/tokenizer.json",
        "sha256": "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db",
    },
    # deepseek-ai/DeepSeek-V3 = DeepSeek-V3-0324 (git blob 51083c60)
    "dsv3": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json",
        "sha256": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
    },
    # byte-level vocabularies missing some byte chars (hf drops those bytes in the model: docs/breadth.md §4); their
    # missing bytes are C0 C1 F5..FF, which valid utf-8 never holds (toks's byte input drops them)
    # EleutherAI/pythia-160m = pythia-70m/410m/1.4b/6.9b(-deduped), gpt-neox-20b (census 633e543f): NFC, 243 of 256 byte chars
    "pythia": {
        "url": "https://huggingface.co/EleutherAI/pythia-160m/resolve/50f5173d932e8e61f858120bcb800b97af589f46/tokenizer.json",
        "sha256": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
    },
    # EleutherAI/pythia-14m (census b67d44d0): pythia's vocab with an empty TemplateProcessing
    "pythia14m": {
        "url": "https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/tokenizer.json",
        "sha256": "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2",
    },
    # allenai/OLMoE-1B-7B-0125-Instruct (census c568c500)
    "olmoe": {
        "url": "https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/resolve/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e/tokenizer.json",
        "sha256": "d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea",
    },
    # ibm-granite/granite-embedding-small-english-r2 (census d800841e, = ModernBERT-base's tokenization): 23 decode-only vocab strings
    "granite-emb": {
        "url": "https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/resolve/2ab6fa8ea2d674564defd37171ae19079b864b33/tokenizer.json",
        "sha256": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    },
    # answerdotai/ModernBERT-base (the current main file; hashes like granite-emb's in the census)
    "modernbert": {
        "url": "https://huggingface.co/answerdotai/ModernBERT-base/resolve/8949b909ec900327062f0ebf497f51aef5e6f0c8/tokenizer.json",
        "sha256": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    },
    # deepseek r1 / v3.1 / v3.2 / v4 / v4.1: the dsv3 template too (the same normalizer, pre-tokenizer, decoder and
    # post-processor as V3; they differ in added tokens)
    # deepseek-ai/DeepSeek-R1 = DeepSeek-R1-0528 (git blob d81c3a6e)
    "dsr1": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-R1/resolve/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/tokenizer.json",
        "sha256": "ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d",
    },
    # deepseek-ai/DeepSeek-V3.1 = DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp (git blob bfa20fbe)
    "dsv31": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3.1/resolve/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/tokenizer.json",
        "sha256": "32b34a41212e92f62e859cbbea121ae705a1fabbf157d9acf22d134ecd8dcf70",
    },
    # deepseek-ai/DeepSeek-V3.2 (git blob 9b4d3197)
    "dsv32": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/tokenizer.json",
        "sha256": "cd050be35cae877f8f0aa847f45aa87e23835a56ca32b29b28545597852784e5",
    },
    # deepseek-ai/DeepSeek-V4-Flash = DeepSeek-V4-Pro, DeepSeek-V4-Flash-0731 (git blob 628e3364)
    "dsv4": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json",
        "sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    },
    # deepseek-ai/DeepSeek-V4.1-Flash (git blob 6a15814d)
    "dsv41flash": {
        "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/tokenizer.json",
        "sha256": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    },
    # the critical targets (gpt-oss is "o200k" above), pinned to a hub revision
    "glm53": {
        "url": "https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/tokenizer.json",
        "sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    },
    "qwen38": {
        "url": "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json",
        "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    },
    "gemma4": {
        "url": "https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/4d7ae4984b7db7de8f8457170b3f1a419ee76d52/tokenizer.json",
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    # critical targets added 2026-10-04: nemotron, llama, minimax. One entry per distinct
    # file; the comment names the models that ship the same bytes (hub x-linked-etag / git blob id, checked
    # 2026-10-04). Gated meta-llama files come from ungated copies with identical bytes.
    # NousResearch/Meta-Llama-3.1-8B-Instruct the meta-llama 3.1 / 3.2 Instruct file (git blob 5cc5f00a: Llama-3.1-8B/70B/405B-Instruct, Llama-3.2-1B/3B-Instruct;
    # 3.3-70B-Instruct ships llama3's bytes, git blob 1c1d8d5c)
    "llama31meta": {
        "url": "https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct/resolve/d10aef7999a2b5ba950ab3974312feeedbfe0b77/tokenizer.json",
        "sha256": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    },
    # unsloth/Llama-4-Scout-17B-16E-Instruct = meta-llama/Llama-4-Scout-17B-16E-Instruct and -Maverick-17B-128E-Instruct (git blob b1fde397)
    "llama4": {
        "url": "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/resolve/afd8e498c87bda51c7ea8ec68ea2f7c066e6340b/tokenizer.json",
        "sha256": "172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4",
    },
    # nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 = Nemotron-3-Super-120B-A12B, -Ultra-550B-A55B, -3.5-Lightning-30B-A3B, -Nano-30B-A3B-Base
    "nemotron3-4b": {
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/tokenizer.json",
        "sha256": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    },
    # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 = its FP8 and NVFP4 repos
    "nemo3nano30b": {
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/tokenizer.json",
        "sha256": "c6021eb6847e682f89aa52d5eb6e8c7d902a23acfc8137e25211cf84828f1592",
    },
    # nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 = its FP8 repo
    "nemotron3-omni": {
        "url": "https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/resolve/e5e9932441de940c9a62185c870ea5bcd4cd24e2/tokenizer.json",
        "sha256": "e5e7dc84d72e8f248321611c3d6dce23407b135f55f8caf5b26119798d12f85f",
    },
    # nvidia/NVIDIA-Nemotron-Nano-9B-v2 = NVIDIA-Nemotron-Nano-12B-v2, Nano-9B-v2-FP8
    "nemonano2": {
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/resolve/6533e8de2c68e4536bf7c411d7a3ce5734111476/tokenizer.json",
        "sha256": "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    },
    # nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 = its FP8 repo
    "nemonano2vl": {
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/ca9543b126e8bf3176916d3d305ccc415f89fd4d/tokenizer.json",
        "sha256": "db8e35444fca3a2b98e2c8e927a8f1d8b1ba9d4b349e13ce5aafdb11b6404205",
    },
    # MiniMaxAI/MiniMax-Text-01-hf = MiniMaxAI/MiniMax-Text-01
    "minimaxt01": {
        "url": "https://huggingface.co/MiniMaxAI/MiniMax-Text-01-hf/resolve/f7ce01366e8585a8948f19aedc8e20628c6965e5/tokenizer.json",
        "sha256": "ece04384257543dd1c1312991b6042efdc5be09103729a62cc84d718bcc3b1a6",
    },
    # MiniMaxAI/MiniMax-M1-80k = MiniMax-M1-80k-hf, M1-40k, M1-40k-hf
    "minimaxm1": {
        "url": "https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/resolve/8d1494b1a260e22040d5b9b2eb332eb44500b34d/tokenizer.json",
        "sha256": "369f547b736fad84af7c5bd8523ab1414b7116b5d167d84a10cf37c45dc79348",
    },
    # MiniMaxAI/MiniMax-M2 = MiniMax-M2.1, M2.5, M2.7
    "minimaxm2": {
        "url": "https://huggingface.co/MiniMaxAI/MiniMax-M2/resolve/757303d492a50514c312788b5247a4f696a4c6a3/tokenizer.json",
        "sha256": "757622126525aeeb131756849d93298070ff3f0319c455ec8c5bb0f6b1cebbe8",
    },
    "minimaxm3": {
        "url": "https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/f0e1c1e04d40177e4673a22097036854f536e9c0/tokenizer.json",
        "sha256": "bb1f1626cf01448f1e3b6036d0a061ffc66c91d9046aada14ea23a5441b5ad6e",
    },
    "kimik3.tiktoken": {
        "url": "https://huggingface.co/moonshotai/Kimi-K3/resolve/f831ab66814297da540d832a5235f8e904f29d06/tiktoken.model",
        "sha256": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    },
    "kimik3_tokenizer_config.json": {
        "url": "https://huggingface.co/moonshotai/Kimi-K3/resolve/f831ab66814297da540d832a5235f8e904f29d06/tokenizer_config.json",
        "sha256": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    },
    "kimik3_tokenization_kimi.py": {
        "url": "https://huggingface.co/moonshotai/Kimi-K3/resolve/f831ab66814297da540d832a5235f8e904f29d06/tokenization_kimi.py",
        "sha256": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    },
    # Qwen-1 (Qwen-72B): qwen.tiktoken + tokenization_qwen.py + tokenizer_config.json (docs/models/qwen1.md)
    "qwen1-72b.tiktoken": {
        "url": "https://huggingface.co/Qwen/Qwen-72B/resolve/b8e18ac61df64d35308695769ff46b976b6a00f4/qwen.tiktoken",
        "sha256": "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186",
    },
    "qwen1-72b_tokenizer_config.json": {
        "url": "https://huggingface.co/Qwen/Qwen-72B/resolve/b8e18ac61df64d35308695769ff46b976b6a00f4/tokenizer_config.json",
        "sha256": "b045185f7e1e53cb05b658aa48940c1e5a002488bd55369cbf171b28306844de",
    },
    "qwen1-72b_tokenization_qwen.py": {
        "url": "https://huggingface.co/Qwen/Qwen-72B/resolve/b8e18ac61df64d35308695769ff46b976b6a00f4/tokenization_qwen.py",
        "sha256": "c9bbb7710bacfaa323d48b8414643d8d7893d174f85f65e83fbc363de271b69a",
    },
}

# WordPiece: the census's wordpiece tokenizer.json files, pinned in tests/wordpiece/pins.json (one list, read here),
# stored as wp-<name> (tests/wordpiece/fetch.py writes the same files).
_WP_PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "wordpiece", "pins.json")
with open(_WP_PINS, encoding="utf-8") as _f:
    WORDPIECE = tuple("wp-" + k for k, v in json.load(_f).items() if not k.startswith("_"))
with open(_WP_PINS, encoding="utf-8") as _f:
    for _k, _v in json.load(_f).items():
        if not _k.startswith("_"):
            TOKENIZERS["wp-" + _k] = {
                "url": f"https://huggingface.co/{_v['repo']}/resolve/{_v['revision']}/{_v['path']}",
                "sha256": _v["sha256"],
            }

# Unigram: the census's unigram tokenizer.json files (+ t5-base, flan-t5-base), pinned in tests/unigram/pins.json
# (read here), stored under their pin names (uni_<name>; tests/unigram/fetch.py writes the same files).
_UNI_PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "unigram", "pins.json")
with open(_UNI_PINS, encoding="utf-8") as _f:
    _uni = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}
UNIGRAM = tuple(_uni)
for _k, _v in _uni.items():
    TOKENIZERS[_k] = {"url": f"https://huggingface.co/{_v['repo']}/resolve/{_v['revision']}/{_v['file']}",
                      "sha256": _v["sha256"]}

# the census's truncation group: byte-level / sentencepiece-style bpe files whose tokenizer.json sets truncation and
# padding (docs/coverage.md section 8), stored under these names
TRUNC = {
    "bpe-distilroberta": ("https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/842eaed40bee4d61673a81c92d5689a8fed7a09f/tokenizer.json",
        "7a6751507c44ab383cc8ba8ab97cd857d35025ff1b29162b1690cc5e0c6030e9"),
    "bpe-gte-rerank-mbert": ("https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/resolve/f7481e6055501a30fb19d090657df9ec1f79ab2c/tokenizer.json",
        "2aea6ff4701d063e7e029b6be695a1659f2caaa2ae4fb0e8b18285818271becd"),
    "bpe-jina5-omni": ("https://huggingface.co/jinaai/jina-embeddings-v5-omni-small/resolve/5c54692f22e186fc12ea38a9193e3ff9e1cda2a3/tokenizer.json",
        "7bbf567d9e053a72f0bd8ea475e672d10d537578f6adb521ab42da2badfa8bef"),
    "bpe-llama32-fp8": ("https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-FP8/resolve/377571d314b30f1d58448499e4100e2deafe7d7d/tokenizer.json",
        "4745787bf5429f4558dbadb95086d68ccc290ca1fac62bdb3d05c233fab5bc40"),
    "bpe-giga-emb": ("https://huggingface.co/ai-sage/Giga-Embeddings-instruct/resolve/2cf0fdc97194aaedf10ac0e6bf798834acd31042/tokenizer.json",
        "0ec0a1cffcc9192f5ee3d7b273673a062918055238bda3d23cfb6d2512e947ff"),
    "spm-otel-e4b": ("https://huggingface.co/farbodtavakkoli/OTel-LLM-E4B-IT/resolve/12ffb1ef5812f53ea2c7732b4cc3703c2d2171a9/tokenizer.json",
        "c8c1be090333d4535bdcdb49d883b4aed36c290598b56244ad5e357b37e08883"),
}
for _k, (_u, _s) in TRUNC.items():
    TOKENIZERS[_k] = {"url": _u, "sha256": _s}

# the census's digits-gpt2 group (pre_tokenizer [Digits(individual_digits), ByteLevel(use_regex)]: kernels.md §3 A8),
# one file per distinct tokenizer (docs/coverage.md section 8; EXAONE also needs NFKC)
DIGITS = {
    "dg-smollm2": ("https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json",
        "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
    "dg-powermoe": ("https://huggingface.co/ibm-research/PowerMoE-3b/resolve/13fcb5a98001438bed01cf1ac4b423751dc4c2ea/tokenizer.json",
        "d3b4df07a0ce3940b15c77e2ea17ab5627ca0c4bf982d2eb37966aea0a81c918"),
    "dg-smollm-17b-w4": ("https://huggingface.co/nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16/resolve/e66aefd5fd7c1e4995223b86816643ced975af98/tokenizer.json",
        "3d23ca43a58430ff97f8da6fb2d3cef2c49d30633e28ec826c4fc59c60f5f790"),
    "dg-mellum2": ("https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct/resolve/273d9c1c6882ac8e1562e5b7e043857d35a787b6/tokenizer.json",
        "58548a346eb073e5132bf7d8ad17dc6971bca36ade378ca4d2bfbc49bf60da2a"),
    "dg-tiny-cohere": ("https://huggingface.co/trl-internal-testing/tiny-CohereForCausalLM/resolve/90bc56bd12c642f6bf5e710ebd7c626dd7c0201c/tokenizer.json",
        "345ccf04a5257f473e331715ecc69365c5ac8fc2490923fe7155560af809ec1a"),
    "dg-smolvlm2": ("https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/7b375e1b73b11138ff12fe22c8f2822d8fe03467/tokenizer.json",
        "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a"),
    "dg-exaone35": ("https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/resolve/553ea250b9a5317231459279d5847d6cf955b9aa/tokenizer.json",
        "7c7507fef57dd4daa4c37103fa656a793748e4cbceb03f808ef292d0324aab10"),
}
for _k, (_u, _s) in DIGITS.items():
    TOKENIZERS[_k] = {"url": _u, "sha256": _s}

# the census's chains no template compiles (the generic engine, docs/algorithms/generic.md §1): one file per distinct
# tokenizer (docs/coverage.md section 8)
GENERIC = {
    "gx-minicpm5": ("https://huggingface.co/openbmb/MiniCPM5-2B/resolve/f97400052a43d642bbc6e9975e2397e3ae6a6b52/tokenizer.json",
        "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"),
    "gx-dscoder2-lite": ("https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/resolve/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11/tokenizer.json",
        "091b9dadb9845f0e8386c38bdb87e98db8adc7b0aacf36cb9257c00d0a668714"),
    "gx-dscoder-7b15": ("https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5/resolve/2a050a4c59d687a85324d32e147517992117ed30/tokenizer.json",
        "a6abdd41a79c1de6b68bbb819ad9c8d63b97080b0b90949d1c37ddbf64ee118c"),
    "gx-dsv2-lite": ("https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/resolve/604d5664dddd88a0433dbae533b7fe9472482de0/tokenizer.json",
        "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669"),
    "gx-natural-sql": ("https://huggingface.co/chatdb/natural-sql-7b/resolve/ac0e30435282bb6e3ab6459358e9521a557c5fdf/tokenizer.json",
        "e3a7a6c6facd433987b529a82406ecdffb04853f96bd909b0194d070c9880def"),
    "gx-laguna-xs2": ("https://huggingface.co/poolside/Laguna-XS.2/resolve/a397bde04501591d0ac3d74a5d9d0a4ec298ffb8/tokenizer.json",
        "807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279"),
    "gx-laguna-s21": ("https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/resolve/826aacdf6d8b2699d4e367def6f17c83b06044c2/tokenizer.json",
        "809240f7a182cde859a4fc4ebc902e619a173d507e99304c1092aa04e7a6658e"),
    "gx-bloom": ("https://huggingface.co/bigscience/bloomz-560m/resolve/a2845d7e13dd12efae154a9f1c63fcc2e0cc4b05/tokenizer.json",
        "3fa39cd4b1500feb205bcce3b9703a4373414cafe4970e0657b413f7ddd2a9d3"),
    "gx-llada8b": ("https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/resolve/08b83a6feb34df1a6011b80c3c00c7563e963b07/tokenizer.json",
        "071eb20fe7bd601550b1b7838ff696d6c93b88f51257f753303d3df23163c381"),
    "gx-ling3": ("https://huggingface.co/inclusionAI/Ling-3.0-flash/resolve/ef06d91fe382109ae82647da88ff99b0f11745b0/tokenizer.json",
        "40fb9d7d7795b8bd305aeff39ce9963f3f450915b9553f2938e009be9a1fed60"),
    "gx-llada2-mini": ("https://huggingface.co/inclusionAI/LLaDA2.0-mini/resolve/dad945cac317da394b390f82c7b40691d8a881ed/tokenizer.json",
        "0a1db22fe3dfb28f9b2be6d87f472af371b50235d7f5366ecac14b52ff8c002b"),
    "gx-spark-x25": ("https://huggingface.co/XHToken/Spark-X2.5-4B/resolve/0bcb35678590218655dff3765b9e61c83b35e9c4/tokenizer.json",
        "710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c"),
    "gx-zeta21": ("https://huggingface.co/LeaderboardModel1/zeta-2.1-autoround-W4A16/resolve/d9a6ffca5b588e0c291d160ea0c59a7941557005/tokenizer.json",
        "db6520146c388c495a98bbea62ff6d00c0a8935bed33622e33bb33ec71aaafed"),
    "gx-falcon7b": ("https://huggingface.co/tiiuae/falcon-7b/resolve/ec89142b67d748a1865ea4451372db8313ada0d8/tokenizer.json",
        "d6c5cdac1421ea998722aa88ea1cb630b2a9ec63bf8a85331bb3d9d321507186"),
    "gx-ax-k2": ("https://huggingface.co/skt/A.X-K2-NVFP4/resolve/9e2e804e80f8d1b3afba5d7938173cec1ed46b49/tokenizer.json",
        "4d565a2103b2e69fabc9f1c2dda96c6ba26555c6dc05464e00fafc6467b08b2f"),
    "gx-dscoder-67b": ("https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/resolve/e5d64addd26a6a1db0f9b863abf6ee3141936807/tokenizer.json",
        "ef48ebdc8546c2d8092349d321f1d162de804a1c8900df2b615c7dc8b02ce141"),
    "gx-tiny-cohere2": ("https://huggingface.co/trl-internal-testing/tiny-Cohere2ForCausalLM/resolve/11632b48cc25820056f7cc5730b1a2f0a89ce47e/tokenizer.json",
        "2227ea9c52e8afb3f98bfed2679008b275f2664de69dfde174b374389eb0225d"),
}
for _k, (_u, _s) in GENERIC.items():
    TOKENIZERS[_k] = {"url": _u, "sha256": _s}

M1A = ("gpt2", "llama3")
TARGETS = ("glm53", "o200k", "qwen38", "gemma4", "kimik3.tiktoken", "kimik3_tokenizer_config.json",
           "kimik3_tokenization_kimi.py", "qwen1-72b.tiktoken", "qwen1-72b_tokenizer_config.json",
           "qwen1-72b_tokenization_qwen.py", "llama3", "llama31meta", "llama4", "nemotron3-4b", "nemo3nano30b",
           "nemotron3-omni", "nemonano2", "nemonano2vl", "minimaxt01", "minimaxm1", "minimaxm2", "minimaxm3",
           "dsv3", "dsr1", "dsv31", "dsv32", "dsv4", "dsv41flash") + WORDPIECE + UNIGRAM + tuple(TRUNC) + tuple(DIGITS) + tuple(GENERIC)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: str, sha: str) -> None:
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(20, 600), allow_redirects=True) as r:
                r.raise_for_status()
                tmp = dest + ".part"
                h = hashlib.sha256()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        h.update(chunk)
            got = h.hexdigest()
            if got != sha:
                raise RuntimeError(f"sha256 mismatch {dest}: got {got} want {sha}")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/toks/tokenizers"))
    ap.add_argument("--all", action="store_true", help="fetch every pinned tokenizer, not just the default gpt2 + llama3")
    ap.add_argument("--targets", action="store_true", help="fetch the critical targets (tests/data/targets/ledger.txt)")
    ap.add_argument("names", nargs="*", help="specific names (default: gpt2 llama3)")
    args = ap.parse_args()
    names = args.names or (list(TOKENIZERS) if args.all else list(TARGETS) if args.targets else list(M1A))
    os.makedirs(args.root, exist_ok=True)
    for name in names:
        pin = TOKENIZERS[name]
        dest = os.path.join(args.root, name)
        if os.path.isfile(dest):
            got = sha256_file(dest)
            if got == pin["sha256"]:
                print(f"{name}: ok ({got[:16]}…)")
                continue
            print(f"{name}: sha mismatch on disk, refetching")
            os.remove(dest)
        print(f"{name}: fetching {pin['url']}")
        fetch(pin["url"], dest, pin["sha256"])
        print(f"{name}: ok ({pin['sha256'][:16]}…)")


if __name__ == "__main__":
    main()
