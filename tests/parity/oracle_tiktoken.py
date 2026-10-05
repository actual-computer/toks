#!/usr/bin/env python3
"""tests/parity/oracle_tiktoken.py: the ids oracle for tiktoken-format models (Kimi K3 first).

The reference is the model's own tokenizer as people run it (docs/models/kimi.md):
transformers AutoTokenizer(trust_remote_code=True) -> tokenization_kimi.TikTokenTokenizer -> tiktoken.

    uv run --python 3.12 --with tiktoken==0.14.0 --with transformers==5.18.0 \
        python tests/parity/oracle_tiktoken.py --self-check
    ... python tests/parity/oracle_tiktoken.py --mode serving < cases.jsonl > ids.jsonl

Modes (one per call path; docs/models/kimi.md section 2 says where they differ):
  serving   tok.encode(text, add_special_tokens=False): PreTrainedTokenizer's path (added-token trie split,
            then the wrapper per segment). == tok(text)["input_ids"] (vLLM 0.30 completions). toks ALL.
  direct    tok.encode(text): no kwargs, the wrapper's own tiktoken path over the whole text (sglang 0.5.21).
            == tok.encode(text, split_special_tokens=True). toks NONSPECIAL.
  none      tok.encode(text, allow_special_tokens=False): every special name is plain text (the chat path's
            user / tool segments). toks NONE.
  tiktoken  the bare engine: tiktoken.Encoding(pat_str, ranks, specials).encode(text, allowed_special="all"),
            no 400,000 / 25,000 chunking (for piece-level work on short inputs).
  ordinary  the bare engine's encode_ordinary(text).
Case file: one JSON object per line {"text": "..."}; output: one JSON list of ids per line ({"error": msg}
when the reference raises).

Pins (checked at load): the four model files' sha256 (MODEL_FILES), tiktoken 0.14.0, transformers 5.18.0.
TIKTOKEN_CACHE_DIR is forced to "" before tiktoken loads: load_tiktoken_bpe caches every file, local paths
included, under sha1(path) in the temp dir and serves that copy on the next load of the same path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ["TIKTOKEN_CACHE_DIR"] = ""          # never serve a stale copy of a file (see above)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PINS = {"tiktoken": "0.14.0", "transformers": "5.18.0"}
KIMI_DIR = os.path.expanduser(os.environ.get("TOKS_KIMI_DIR", "~/.cache/toks/kimik3"))
KIMI_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"   # moonshotai/Kimi-K3
MODEL_FILES = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "encoding_k3.py": "49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676",
}
MODES = ("serving", "direct", "none", "tiktoken", "ordinary")


def fetch_kimi(dest=KIMI_DIR):
    """downloads the pinned files from the hub into dest (curl; the repo is public)."""
    import subprocess
    os.makedirs(dest, exist_ok=True)
    for name in MODEL_FILES:
        p = os.path.join(dest, name)
        if not os.path.exists(p):
            url = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{KIMI_REVISION}/{name}"
            subprocess.run(["curl", "-sfL", "-m", "300", "-o", p, url], check=True)


def check_files(d=KIMI_DIR):
    for name, want in MODEL_FILES.items():
        with open(os.path.join(d, name), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            raise SystemExit(f"oracle_tiktoken: {name} sha256 {got} != pinned {want}")


class Kimi:
    """the reference, loaded once per process."""

    def __init__(self, d=KIMI_DIR, check_pins=True):
        import tiktoken
        import transformers
        if check_pins:
            for mod, v in (("tiktoken", tiktoken.__version__), ("transformers", transformers.__version__)):
                if v != PINS[mod]:
                    raise SystemExit(f"oracle_tiktoken: {mod} {v} != pinned {PINS[mod]}")
        check_files(d)
        import logging
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        # the wrapper logs a warning on every call with kwargs ("Calling super().encode with ...")
        logging.getLogger("transformers_modules").setLevel(logging.ERROR)
        m = self.tok.model                       # the wrapper's tiktoken.Encoding
        self.enc = tiktoken.Encoding(name="kimi-bare", pat_str=m._pat_str,
                                     mergeable_ranks=m._mergeable_ranks, special_tokens=m._special_tokens)
        self.pat_str = m._pat_str
        self.special_tokens = dict(m._special_tokens)          # all 256 names -> ids
        self.added = dict(self.tok._added_tokens_encoder)      # the trie's tokens -> ids

    def encode(self, text, mode):
        if mode == "serving":
            return self.tok.encode(text, add_special_tokens=False)
        if mode == "direct":
            return self.tok.encode(text)
        if mode == "none":
            return self.tok.encode(text, allow_special_tokens=False)
        if mode == "tiktoken":
            return self.enc.encode(text, allowed_special="all")
        if mode == "ordinary":
            return self.enc.encode_ordinary(text)
        raise ValueError(mode)

    def safe_encode(self, text, mode):
        """ids, or {"error": message} when the reference raises (BaseException: pyo3 panics are not
        Exceptions)."""
        try:
            return self.encode(text, mode)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            return {"error": f"{type(e).__name__}: {e}"}

    def decode(self, ids, skip_special=False):
        if skip_special:
            return self.tok.decode(ids, skip_special_tokens=True)
        return self.tok.decode(ids)


def self_check(k):
    """fixed facts the doc states, each one executed (exit 1 on the first that fails)."""
    ok = True

    def want(name, got, exp):
        nonlocal ok
        good = got == exp
        ok &= good
        print(("ok   " if good else "FAIL ") + name + ("" if good else f": got {got!r} want {exp!r}"))

    t = k.tok
    want("n_words", t.n_words, 163840)
    want("special names", len(k.special_tokens), 256)
    want("trie tokens", sorted(k.added.values()),
         [163584, 163585, 163586, 163587, 163588, 163589, 163590, 163591, 163593, 163602, 163603, 163604,
          163605, 163649, 163838, 163839])
    s = "Hello world! \u4f60\u597d\u4e16\u754c [BOS] <|open|>x<|reserved_token_163600|> done"
    a = k.encode(s, "serving")
    want("serving == __call__", t(s)["input_ids"], a)
    want("serving == add_special_tokens=True (no bos/eos)", t.encode(s, add_special_tokens=True), a)
    want("direct == serving (short text)", k.encode(s, "direct"), a)
    want("split_special_tokens=True == direct", t.encode(s, split_special_tokens=True), k.encode(s, "direct"))
    want("specials recognized", [i for i in a if i >= 163584], [163584, 163587, 163600])
    want("none: no special id", [i for i in k.encode(s, "none") if i >= 163584], [])
    want("tiktoken == direct (short text)", k.encode(s, "tiktoken"), k.encode(s, "direct"))
    # chunking: a non-space run of 25,001 chars is cut after 25,000 (direct); a trie token resets the count
    # in serving only.
    x = "a" * 20000 + "[BOS]" + "b" * 10000
    d, v = k.encode(x, "direct"), k.encode(x, "serving")
    want("25k cut: direct != serving across a trie token", d != v, True)
    want("25k cut: serving == per-segment", v,
         k.encode("a" * 20000, "tiktoken") + [163584] + k.encode("b" * 10000, "tiktoken"))
    want("25k cut: direct == two chunks", d,
         k.encode(x[:25000], "tiktoken") + k.encode(x[25000:], "tiktoken"))
    want("decode round trip (valid utf-8)", k.decode(a), s)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="serving")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download the pinned files into $TOKS_KIMI_DIR")
    args = ap.parse_args()
    if args.fetch:
        fetch_kimi()
    k = Kimi()
    if args.self_check:
        sys.exit(0 if self_check(k) else 1)
    out = sys.stdout
    for line in sys.stdin:
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        out.write(json.dumps(k.safe_encode(text, args.mode)) + "\n")


if __name__ == "__main__":
    main()
