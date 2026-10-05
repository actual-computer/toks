#!/usr/bin/env python3
"""tools/oracle/limits.py: where tiktoken's regex fails at run time (docs/models/kimi.md section 5).

The bare engine (no wrapper cuts) on long single-class runs, then the wrapper (direct) on the same texts.
    tools/oracle/py.sh tools/oracle/limits.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))
from oracle_tiktoken import Kimi  # noqa: E402

CASES = [
    ("spaces then a letter", lambda n: " " * n + "a"),
    ("spaces", lambda n: " " * n),
    ("newlines", lambda n: "\n" * n),
    ("letters", lambda n: "a" * n),
    ("upper then no lower", lambda n: "A" * n + "!"),
    ("marks", lambda n: "\u0301" * n),
    ("han", lambda n: "\u4e2d" * n),
    ("punctuation", lambda n: "!" * n),
    ("digits", lambda n: "1" * n),
]


def main():
    k = Kimi()
    for n in (25_000, 400_000, 1_200_000):
        for name, f in CASES:
            t = f(n)
            out = []
            for mode in ("tiktoken", "direct"):
                r = k.safe_encode(t, mode)
                out.append(f"{mode}: " + (r["error"][:90] if isinstance(r, dict) else f"{len(r)} ids"))
            print(f"{n:9d} {name:22s} " + "; ".join(out), flush=True)


if __name__ == "__main__":
    main()
