#!/usr/bin/env python3
"""tests/c/test_e2e.py: writes test_e2e.inc next to it, the expected ids test_e2e.c checks toks against.

    uv run --with tokenizers==0.23.2 python tests/c/test_e2e.py

hf tokenizers 0.23.2 is the definition (python/toks_oracle: mode ALL = hf default, NONSPECIAL =
encode_special_tokens=True, NONE = the file with added_tokens removed; flag 4 = add_special_tokens=False).
The tokenizer files are the pinned ones (tools/corpora/fetch_tokenizers.py) under
$TOKS_TOKENIZER_CACHE or ~/.cache/toks/tokenizers.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from toks_oracle import oracle as O  # noqa: E402

ROOT = os.environ.get("TOKS_TOKENIZER_CACHE", os.path.expanduser("~/.cache/toks/tokenizers"))
PINS = {
    "gpt2": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "llama3": "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    # zai-org/GLM-5.3 tokenizer.json @ aca966e4e02791568aa6a4ced368624b3d897f42
    "glm53": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    # deepseek-ai/DeepSeek-V3 tokenizer.json: the dsv3 template (docs/templates/dsv3.md)
    "dsv3": "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
    # Qwen/Qwen3.8-27B tokenizer.json @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0: normalizer NFC
    "qwen38": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    # openai/gpt-oss-20b tokenizer.json (the o200k template; tests/data/targets/ledger.txt gpt-oss)
    "o200k": "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
    # mistralai/Mistral-Nemo-Instruct-2407 tokenizer.json (the o200k template's nemo variant)
    "mistral-nemo": "e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db",
    # one file per family of the 2026-10-04 critical targets (tests/data/targets/ledger.txt; the others in the family share
    # its pre-tokenizer, normalizer and decoder): nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (the nemo variant),
    # unsloth/Llama-4-Scout-17B-16E-Instruct (o200k), MiniMaxAI/MiniMax-M2 (NFC + o200k spelled Split Removed+invert)
    "nemotron3-4b": "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    "llama4": "172c9eb4beafc72601690da3ccfcede5c2e6806a8d5ec1fca33e22acea8023a4",
    "minimaxm2": "757622126525aeeb131756849d93298070ff3f0319c455ec8c5bb0f6b1cebbe8",
    # deepseek-ai/DeepSeek-V4-Flash (= V4-Pro): V3's chain and model with 1,283 added tokens (46 normalized: <dsml: ... in phase 1)
    "dsv4": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    # answerdotai/ModernBERT-base @ 8949b909ec90: NFC + gpt-2 ByteLevel, a vocab missing C0 C1 F5..FF (dropped),
    # 23 decode-only vocab strings, an lstrip [MASK], a [CLS] $A [SEP] template (docs/breadth.md)
    "modernbert": "9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1",
    # unigram (docs/algorithms/unigram.md): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Precompiled
    # nmt_nfkc + WhitespaceSplit + Metaspace, the family's top row) and llm-jp/llm-jp-4-33b-thinking (byte fallback,
    # the literal-U+2581 chains)
    "uni_pmminilm": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",
    "uni_llmjp4": "a35f390c6489427e4dcac7d957f34d9dacd7332972c990f4b4eb6b724c8873f4",
}

WORDS = ("the quick brown fox jumps over the lazy dog while 42 cats watch; "
         "it's 3:14pm and we'll see 1,234,567 stars. ").split(" ")
LONG = " ".join(WORDS[i * 7 % len(WORDS)] for i in range(700))

TEXTS = [
    "",
    "Hello world",
    "Hello, world! How's it going? I'm fine; we'll see.",
    "  leading spaces and trailing   ",
    "numbers 1234567 and 3.14159, 1e10 and 007",
    "line1\nline2\r\n\r\nline3\n\n\n",
    "tabs\tand\u00a0nbsp and \u3000 ideographic space \u2028 sep",
    "h\u00e9llo w\u00f6rld \u2014 na\u00efve caf\u00e9",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3001\u4e2d\u6587\u6587\u672c\u3002\ud55c\uad6d\uc5b4",
    "emoji \U0001F600\U0001F44D\U0001F3FD family \U0001F468\u200d\U0001F469\u200d\U0001F467",
    "don't DON'T 'S 'll I'VE \u017f",
    "<|endoftext|>",
    "a<|endoftext|>b <|endoftext|><|endoftext|>",
    "<|begin_of_text|>Hi<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nyo",
    "<|reserved_special_token_5|> <|endoftext|",
    "def f(x):\n    return x**2  # square\n\tpass\n",
    "The year 2024 had 365 days.\tTAB\x0bVT\x0cFF \x85NEL",
    "[gMASK]<sop><|system|>\nYou are helpful.<|user|>\nhi /nothink<|assistant|>\n<think></think>\nHello!",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><tool_response>{}</tool_response>",
    "<|code_prefix|>def f():<|code_suffix|>  return 1<|code_middle|><|observation|>",
    "<|begin_of_image|><|image|><|end_of_image|> <|begin_of_box|>x<|end_of_box|><|video|><|begin_of_video|",
    "/nothink /nothin /nothinks <thinking></think > [MASK][sMASK]<eop><|endoftext|>",
    "a" * 1000,
    "z" + " " * 300 + "!",
    LONG,
    # deepseek v3: its chat and tool tokens (specials in phase 0, the normalized non-specials and <|EOT|> in phase 1),
    # then every region / rule of docs/templates/dsv3.md
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>You are a helpful assistant.<\uff5cUser\uff5c>\u3053\u3093\u306b\u3061"
    "\u306f\u3001\u4e16\u754c\uff01123456 apples<\uff5cAssistant\uff5c>OK<\uff5cend\u2581of\u2581sentence\uff5c>",
    "<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>function<\uff5ctool\u2581sep\uff5c>"
    "get_weather\n```json\n{\"city\": \"\u6771\u4eac\"}\n```<\uff5ctool\u2581call\u2581end\uff5c>"
    "<\uff5ctool\u2581calls\u2581end\uff5c><|EOT|><\uff5cplace\u2581holder\u2581no\u25817\uff5c><\uff5c\u2581pad\u2581\uff5c>",
    "<\uff5cUser\uff5c <\uff5cUser\uff5c><\uff5cAssistant <\uff5cbegin\u2581of\u2581sentence\uff5c <|EOT| <\uff5cfim\u2581hole\uff5c>x",
    "\u30ab\u30fb\u30ab \u30fc\u30fb\u30fc \u309b\u30ab \u30a0\u30a0\u30ab \u3040\u30ab \u30ab\u3040 \u3097\u3098\u3099 "
    "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8 \u6f22\u5b57test x\u5b57y \u3007\u3006\u3005 \u9fa5\u9fa6 a  \u4e2d a  1",
    "!hello !!hello .com  .com don't 's\u00e9 !ab\u00e9 a.b.c \u00aba \uff04a \tword \x00\x00a \u00ad\u00ad\u0301x "
    "\U0001F468\u200d\U0001F469 x \n\n  \n  1 1234567 12a34 1\u00bd\u2167x \u2764\ufe0fa \U0001F600\U0001F600!\r\n",
    # NFC (qwen 2 .. 3.8 normalize; the others must leave these bytes alone): decomposed accents, singletons,
    # mark reordering, hangul jamo, an exclusion, vietnamese NFD, marks right after added tokens (hf normalizes
    # each gap alone), a qwen chat turn
    "Cafe\u0301 cre\u0300me bru\u0302le\u0301e \u212b\u2126 a\u0301\u0323 e\u0323\u0302 \u00e9\u0316 e\u0316\u0301",
    "\u1100\u1161\u11a8 \u1100\u1161 \uac00\u11a8 \u0915\u093c \u0958 Tie\u0302\u0301ng Vie\u0323\u0302t",
    "<|im_start|>user\nNe\u0301e<|im_end|>\u0301\n<|im_start|>assistant\n<think>\n\n</think>\n\nOK\u0301<|im_end|>",
    "<|endoftext|>\u0338<|im_end|>\u0301x \u226e<\u0338",
    # NFC lengthens these (exclusions and singletons decompose): one letter piece twice as long, a symbol run
    # three times as long, marks that split; with a scratch sized for the raw length (test_e2e.c's tight pass)
    "\u0958" * 700,
    "\U0001d160" * 300 + "\U0001d15f" * 10,
    " \u0344" * 200 + "\u0f73\u0f75" * 100 + "\u2adc" * 50,
    # the o200k template (docs/templates/o200k.md §3 notes): case runs, the contraction suffix, marks, '/' tails
    "HTMLParser camelCase ABCdEF don't DON'T I'M it'sx they'r 'llama 1's x'\u017f",
    "\u0301A! \u4e2dAB! A\u4e2dB \u02b0A a\u01c5 A\u01c5b !\u0301 !!\u0301 \u0301\u0301 \t!\tword\nword x\u3000y",
    "a/b !\n/ !\n//\n/x !\n/! path/to/file.txt http://x.y/z?q=1 C:\\dir\\f\r\n/ 1234567 \u0663\u0664\u0665 \u2460\u2461",
    "<|start|>user<|message|>What is 2+2?<|end|><|start|>assistant<|channel|>final<|message|>4<|return|><|endofprompt|>",
    "<s>[INST] hi [/INST] ok</s>[AVAILABLE_TOOLS][TOOL_CALLS]<SPECIAL_20><SPECIAL_999> <unk>[PREFIX]",
    # llama 4, nemotron and minimax chat turns (their specials beside text the template and NFC change)
    "<|begin_of_text|><|header_start|>user<|header_end|>\n\nCafe\u0301 ok?<|eot|><|header_start|>assistant<|header_end|>\n\n",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\u212b x</think>\n\nI'M DONE's<|im_end|><SPECIAL_5>",
    "]~!b[]~b]system\nYou\u0301 are MiniMax.[e~[\n]~b]user\n\u1100\u1161\u11a8!<think>a</think>"
    "<minimax:tool_call>{\"x\":1}</minimax:tool_call>]!p~[<fim_prefix>def<fim_suffix>",
    # deepseek v4: its system / dsml tokens (phase-1 prefixes of each other) and the v3 chain around them
    "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cbegin\u2581sys\uff5c>Be brief.<\uff5cend\u2581sys\uff5c><\uff5cUser\uff5c>"
    "\u4f60\u597d<\uff5cAssistant\uff5c><think>x</think><dsml:a>1</dsml:a><dsml:</dsml:<\uff5cDSML\uff5c>"
    "\uff5cDSML\uff5c <\uff5csearch\uff5c>q<\uff5canswer\uff5c>",
]
FLAGS = [0, 1, 2, 4]


def c_bytes(b: bytes) -> str:
    """a C string literal of b: printable ascii as is, the rest as 3-digit octal escapes."""
    out, n = [], 0
    for x in b:
        if 0x20 <= x < 0x7F and x not in (0x22, 0x5C, 0x3F):
            out.append(chr(x))
        else:
            out.append(f"\\{x:03o}")
        n += 1
        if n % 64 == 0:
            out.append('"\n        "')
    return '"' + "".join(out) + '"'


def c_str(s: str) -> str:
    return c_bytes(s.encode("utf-8"))


def ids_lit(ids) -> str:
    rows = []
    for i in range(0, len(ids), 16):
        rows.append(", ".join(str(x) for x in ids[i:i + 16]))
    return "{ " + ",\n          ".join(rows) + " }" if ids else "{ 0 }"


def main():
    out = ["/* generated by tests/c/test_e2e.py from hf tokenizers 0.23.2; do not edit */"]
    texts_lit = ["static const char *const E2E_TEXT[] = {"]
    for t in TEXTS:
        texts_lit.append(f"    {c_str(t)},")
    texts_lit.append("};")
    out += texts_lit
    out.append(f"#define E2E_N_TEXTS {len(TEXTS)}")
    out.append("static const uint32_t E2E_LEN[] = { " + ", ".join(str(len(t.encode())) for t in TEXTS) + " };")
    cases, decs, pcs = [], [], []
    for name, pin in PINS.items():
        path = os.path.join(ROOT, name)
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != pin:
            sys.exit(f"{path}: sha256 {got} is not the pin {pin}")
        tok = O.load(path)
        for ti, t in enumerate(TEXTS):
            for fl in FLAGS:
                mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[fl & 3]
                ids = tok.encode(t, mode=mode, add_special_tokens=not (fl & 4))
                nlen, ends = tok.pieces(t, mode=mode)       # ends in the normalized stream (they tile [0, nlen))
                cases.append((name, ti, fl, ids, nlen))
                if not fl & 4:
                    pcs.append((name, ti, fl, ends))
            ids = tok.encode(t, mode="ALL", add_special_tokens=True)
            for skip in (0, 1):
                decs.append((name, ti, skip, tok.decode(ids, skip_special_tokens=bool(skip)).encode("utf-8")))
    for k, (name, ti, fl, ids, nlen) in enumerate(cases):
        out.append(f"static const uint32_t E2E_V{k}[] = {ids_lit(ids)};")
    out.append("static const struct { const char *tok; uint32_t text, flags, n, nlen; const uint32_t *v; } E2E[] = {")
    for k, (name, ti, fl, ids, nlen) in enumerate(cases):
        out.append(f'    {{ "{name}", {ti}u, {fl}u, {len(ids)}u, {nlen}u, E2E_V{k} }},')
    out.append("};")
    out.append(f"#define E2E_N {len(cases)}")
    out.append("/* pieces (mode = flags 0..2): hf's piece ends in bytes of the normalized stream (python/toks_oracle")
    out.append(" * pieces: added-token matches, then the pre-tokenizer's split of each normalized gap) */")
    for k, (name, ti, fl, ends) in enumerate(pcs):
        out.append(f"static const uint32_t E2E_P{k}[] = {ids_lit(ends)};")
    out.append("static const struct { const char *tok; uint32_t text, flags, n; const uint32_t *v; } E2E_PCS[] = {")
    for k, (name, ti, fl, ends) in enumerate(pcs):
        out.append(f'    {{ "{name}", {ti}u, {fl}u, {len(ends)}u, E2E_P{k} }},')
    out.append("};")
    out.append(f"#define E2E_N_PCS {len(pcs)}")
    out.append("/* decode(encode(text, ALL)) with TOKS_SKIP_SPECIAL off / on */")
    out.append("static const struct { const char *tok; uint32_t text, skip, n; const char *s; } E2E_DEC[] = {")
    for name, ti, skip, d in decs:
        out.append(f'    {{ "{name}", {ti}u, {skip}u, {len(d)}u,\n        {c_bytes(d)} }},')
    out.append("};")
    out.append(f"#define E2E_N_DEC {len(decs)}")
    dst = os.path.join(HERE, "test_e2e.inc")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {dst}: {len(cases)} encode cases, {len(pcs)} pieces cases, {len(decs)} decode cases over "
          f"{len(TEXTS)} texts")


if __name__ == "__main__":
    main()
