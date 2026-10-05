#!/usr/bin/env python3
"""tools/oracle/pieces.py: the piece oracle's client and the pinned pattern strings.

    from pieces import pattern, split
    split(pattern("kimi_k3"), ["Hello world"])  ->  [[(0, 5), (5, 11)]]

split() runs build/oracle/release/tiktoken-pieces (tools/oracle/build.sh builds it): tiktoken's own regex
crate stack (fancy-regex 0.19.0 / regex-automata 0.4.18 / regex-syntax 0.8.11, the tiktoken 0.14.0 wheels'
versions) applied to each text exactly as tiktoken's CoreBPE applies it between two special tokens. Each
result is the list of (start, end) byte offsets of the matches, or a string when tiktoken would fail (the text
is not utf-8, or a regex runtime error).

Patterns, read from their pinned sources, never retyped:
  kimi_k3      TikTokenTokenizer.pat_str of tokenization_kimi.py (moonshotai/Kimi-K3 @ f831ab66, sha256
               f28ea66e...), taken with ast from the file (no import: it needs transformers)
  o200k_base, cl100k_base, r50k_base, p50k_base
               tiktoken_ext.openai_public of the installed tiktoken (0.14.0), with its file loader stubbed
               out so nothing is downloaded
"""
from __future__ import annotations

import ast
import os
import struct
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BIN = os.environ.get("TOKS_PIECES_BIN", os.path.join(ROOT, "build", "oracle", "release", "tiktoken-pieces"))
KIMI_WRAPPER = os.path.expanduser(os.environ.get("TOKS_KIMI_DIR", "~/.cache/toks/kimik3")) + \
    "/tokenization_kimi.py"


def kimi_pat_str(path=KIMI_WRAPPER):
    """TikTokenTokenizer.pat_str = "|".join([...raw strings...]) evaluated from the file's syntax tree."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TikTokenTokenizer":
            for st in node.body:
                if isinstance(st, ast.Assign) and [getattr(t, "id", None) for t in st.targets] == ["pat_str"]:
                    call = st.value
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "join" and isinstance(call.func.value, ast.Constant)):
                        return call.func.value.value.join(ast.literal_eval(call.args[0]))
    raise ValueError(f"{path}: no TikTokenTokenizer.pat_str = '|'.join([...])")


def openai_pat_str(name):
    import tiktoken_ext.openai_public as op
    saved = op.load_tiktoken_bpe
    op.load_tiktoken_bpe = lambda *a, **k: {}
    try:
        return getattr(op, name)()["pat_str"]
    finally:
        op.load_tiktoken_bpe = saved


def pattern(name):
    if name == "kimi_k3":
        return kimi_pat_str()
    if name in ("o200k_base", "o200k_harmony", "cl100k_base", "r50k_base", "p50k_base", "gpt2"):
        return openai_pat_str(name)
    raise ValueError(f"unknown pattern {name!r}")


def split(pat, texts, binary=BIN):
    """texts (str or bytes) -> per text: [(start, end), ...] byte offsets, or an error string."""
    parts = []
    for t in texts:
        b = t.encode("utf-8") if isinstance(t, str) else bytes(t)
        parts.append(struct.pack("<I", len(b)))
        parts.append(b)
    out = subprocess.run([binary, pat], input=b"".join(parts), stdout=subprocess.PIPE, check=True).stdout
    res, o = [], 0
    for _ in texts:
        (k,) = struct.unpack_from("<I", out, o)
        o += 4
        if k == 0xFFFFFFFF:
            (m,) = struct.unpack_from("<I", out, o)
            res.append(out[o + 4:o + 4 + m].decode("utf-8", "replace"))
            o += 4 + m
            continue
        v = struct.unpack_from(f"<{2 * k}I", out, o)
        o += 8 * k
        res.append([(v[2 * i], v[2 * i + 1]) for i in range(k)])
    if o != len(out):
        raise RuntimeError(f"tiktoken-pieces: {len(out) - o} trailing bytes")
    return res


if __name__ == "__main__":
    import json
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "kimi_k3"
    pat = pattern(name)
    print(json.dumps(pat))
    for line in sys.stdin:
        s = json.loads(line)["text"]
        print(json.dumps(split(pat, [s])[0]))
