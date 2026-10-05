#!/usr/bin/env python3
"""tests/parity/qwen1_check.py: Qwen-1 (qwen.tiktoken + tokenization_qwen.py + tokenizer_config.json) against its
own wrapper's path (docs/models/qwen1.md).

    uv run --with tiktoken==0.14.0 python tests/parity/qwen1_check.py --driver build/toks_driver --dir <model dir>
        [--n 20000] [--seed 1]

The reference is tokenization_qwen.py's QWenTokenizer.tokenize (what encode() runs: it overrides tokenize, so
transformers' added-token trie never runs): tiktoken.Encoding("Qwen", PAT_STR, the ranks, SPECIAL_TOKENS)
.encode(unicodedata.normalize("NFC", text), allowed_special="all", disallowed_special=()). The constants come from the
wrapper file itself (its lines from PAT_STR to SPECIAL_TOKENS_SET, exec'd), the ranks from the model's qwen.tiktoken.
Checked per text: toks ALL (flags 0) == that; toks NONSPECIAL and NONE (1, 2) == encode_ordinary(NFC(text)) (every
special is special); decode(ids) == the tokens' bytes joined; decode with TOKS_SKIP_SPECIAL == the wrapper's
`i < self.eod_id` filter. Random texts over an alphabet of the specials, broken specials, NFC-changing sequences
(composed by NFC, or not), the pattern's edge chars.
"""
import argparse
import base64
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import Driver  # noqa: E402

os.environ["TIKTOKEN_CACHE_DIR"] = ""
import tiktoken  # noqa: E402

POOL = (
    ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|extra_0|>", "<|extra_9|>", "<|extra_10|>", "<|extra_99|>",
     "<|extra_100|>", "<|extra_204|>", "<|extra_205|>", "<|extra_", "<|im_", "<|", "|>", "<", ">", "|"] +
    ["e\u0301", "A\u030a", "\u1100\u1161", "\u1100\u1161\u11a8", "\u00e9", "\u212b", "\u0041\u0308\u0301", "\u0301",
     "\uac00", "\u0915\u093c", "\u0958", "\u2126", "\u1e9b\u0323", "\u0344", "\U0001d15e"] +
    list("abcXYZ0123456789!?.,'\"-_/") + [" ", " ", "  ", "\t", "\n", "\n\n", "\r\n", "\u00a0", "\u3000"] +
    ["'s", "'LL", "hello", " world", "\u4e2d\u6587", "\u3053\u3093", "\U0001f600", "caf\u00e9", "1234567"]
)


def reference(model_dir):
    src = open(os.path.join(model_dir, "tokenization_qwen.py"), encoding="utf-8").read()
    m = re.search(r"^PAT_STR = .*?^SPECIAL_TOKENS_SET = [^\n]*$", src, re.S | re.M)
    ns = {}
    exec(m.group(0), ns)                                  # the wrapper's own constants
    ranks = {}
    for line in open(os.path.join(model_dir, "qwen.tiktoken"), "rb").read().splitlines():
        if line:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    specials = {t: i for i, t in ns["SPECIAL_TOKENS"]}
    enc = tiktoken.Encoding("Qwen", pat_str=ns["PAT_STR"], mergeable_ranks=ranks, special_tokens=specials)
    return enc, enc.eot_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    enc, eod = reference(a.dir)
    drv = Driver([a.driver])
    drv.load(a.dir)
    rng = random.Random(a.seed)
    bad = cases = 0
    for _ in range(a.n):
        t = "".join(rng.choice(POOL) for _ in range(rng.randint(1, 20)))
        tb = t.encode("utf-8")
        nfc = unicodedata.normalize("NFC", t)
        want = {0: enc.encode(nfc, allowed_special="all", disallowed_special=())}
        want[1] = want[2] = enc.encode_ordinary(nfc)
        for fl, w in want.items():
            st, got = drv.encode(tb, fl)
            cases += 1
            if st != 0 or got != w:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH encode flags {fl} {t!r}\n  ref  {w[:24]}\n  toks {(got or [])[:24]}")
        ids = want[0]
        for skip in (0, 1):
            kept = [i for i in ids if i < eod] if skip else ids
            w = enc.decode_bytes(kept)
            st, got = drv.decode(ids, skip)
            cases += 1
            if st != 0 or got != w:
                bad += 1
                if bad <= 5:
                    print(f"MISMATCH decode skip {skip} {t!r}\n  ref  {w!r}\n  toks {got!r}")
    drv.close()
    print(f"qwen1 {os.path.basename(os.path.normpath(a.dir))}: {cases} cases ({a.n} texts x encode ALL / NONSPECIAL /"
          f" NONE + decode / skip), {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
