#!/usr/bin/env python3
"""tools/oracle/selfmerge.py: the Kimi tokens that bpe does NOT rebuild from their own bytes. For those, and
only those, tiktoken's whole-piece lookup (encoder.get(piece) before byte_pair_encode; toks: ignore_merges)
changes the ids of a piece equal to the token. The bpe here is tiktoken's rule (lowest rank of the
concatenation, leftmost) without the lookup.

    tools/oracle/py.sh tools/oracle/selfmerge.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "model"))
import kimi_model  # noqa: E402


def merge_only(piece, r):
    b = list(range(len(piece) + 1))
    while len(b) > 2:
        best, at = None, -1
        for k in range(len(b) - 2):
            x = r.get(piece[b[k]:b[k + 2]])
            if x is not None and (best is None or x < best):
                best, at = x, k
        if best is None:
            break
        del b[at + 1]
    return [r[piece[b[k]:b[k + 1]]] for k in range(len(b) - 1)]


def main():
    r = kimi_model.read_ranks(os.path.join(kimi_model.KIMI_DIR, "tiktoken.model"))
    bad = [(rank, tok) for tok, rank in r.items() if len(tok) > 1 and merge_only(tok, r) != [rank]]
    print(f"{len(bad)} of {len(r) - 256} multi-byte tokens are not rebuilt by bpe from their own bytes")
    for rank, tok in sorted(bad)[:40]:
        print(f"  {rank:6d} {tok!r} -> {merge_only(tok, r)}")


if __name__ == "__main__":
    main()
