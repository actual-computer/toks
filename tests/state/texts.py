#!/usr/bin/env python3
"""tests/state/texts.py: the parity quick set's texts for SPEC T2's state matrix (tests/state/run.sh).

    python3 tests/state/texts.py --out build/state/texts.bin [--cp-sample 4000]

The texts of tests/parity/gen_cases.py --quick that do not depend on the tokenizer (python/toks_oracle/gen.py:
golden, a codepoint sweep sample in its 12 contexts, random, ws, scan, cjk, emoji, marks, long) plus the invalid
utf-8 byte strings; test_state adds the tokenizer's own added-token texts (g_special's shapes) itself. Records of
u32 little-endian length + bytes; seeded, byte-identical on every host. No hf needed (gen.py is pure python).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from toks_oracle import gen  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cp-sample", type=int, default=4000, help="sampled scalars of the codepoint sweep (x 12 contexts)")
    a = ap.parse_args()
    texts = {name: g(True) for name, g in gen.ALL_TEXT_GENERATORS.items() if name != "codepoints"}
    texts["codepoints"] = gen.g_codepoints(True, sample=a.cp_sample)
    blobs = [t.encode("utf-8", "surrogatepass") for ts in texts.values() for t in ts] + gen.g_invalid_utf8(True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    h = hashlib.sha256()
    with open(a.out, "wb") as f:
        for b in blobs:
            r = struct.pack("<I", len(b)) + b
            h.update(r)
            f.write(r)
    print(f"{a.out}: {len(blobs)} texts, {sum(map(len, blobs))} bytes, sha256 {h.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
