#!/usr/bin/env python3
"""tools/oracle/notes.py: the examples docs/models/kimi.md quotes, re-derived from the references (the piece
oracle = tiktoken's own regex crates; the ids oracle = the real wrapper). Prints markdown-ready lines.

    tools/oracle/py.sh tools/oracle/notes.py
"""
import os
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "oracle"))
sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))
import pieces  # noqa: E402

NOTES = [
    " \u4e2d\u6587", "\u3000\u4e2d", " \u2e80x", "\u4e2d\u2e80", "!\u2e80", "1\u3007\u3007\u3007", "\u3007\u3007",
    "a\u4e2d", "A\u4e2dB", "\u4e2dA's", "\ud55c\uad6dABC", "ABC\ud55c\uad6dDEF", "HTMLParser", "camelCase",
    "don't", "DON'T", "it'\u017f", "'llama", "1's", "\u0301A!", "!\u0301", "!!\u0301", "a/b", "!\n/x",
    "x  ", "  a", "\t\u4e2d", "\u4e2d\u6587\u5b57\u3002\u65e5\u672c\u8a9e\u3067\u3059", "\u3042\u3044\u4e2d\u3046",
    "\u30ab\u30bf\u30ab\u30ca\u30fc\u4e2d", "\x1c\x1c a", "x\u3005y", "\u3005x", "\uff10\uff11\uff12\uff13",
    "\u2f00\u2f01 \u2f02", "\U00016ff0a",
]


def esc(s):
    return s.encode("unicode_escape").decode("ascii").replace("<|", "<\\u007c")


def main():
    pat = pieces.pattern("kimi_k3")
    res = pieces.split(pat, NOTES)
    for t, p in zip(NOTES, res):
        b = t.encode("utf-8")
        print(f"  {esc(t)!s:40s} -> " + " | ".join(esc(b[x:y].decode('utf-8')) for x, y in p))
    sp = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and chr(c).isspace()]
    print(f"python {sys.version.split()[0]} (unicodedata {unicodedata.unidata_version}) str.isspace: {len(sp)} code points:",
          " ".join(f"{c:04X}" for c in sp))


if __name__ == "__main__":
    main()
