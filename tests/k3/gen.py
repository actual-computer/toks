#!/usr/bin/env python3
"""K3 differential case writer: generates cases with tests/model/toks_model.py's generators and
writes a binary stream of (variant, text, hf-expected piece ends) records for the C checker
(tests/k3/check.c; tests/k3/run.sh builds and runs it, not part of `make test`).

hf is the oracle: pre_tokenizers.Split(Regex(pattern), behavior='isolated') for cl100k / qwen,
pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True) for gpt-2. The python binding
returns CHARACTER offsets; converted to byte offsets with an asserted slice check (toks_model._c2b).

Record format (all little-endian):
  u8 variant      0 gpt2, 1 cl100k, 2 qwen2, 3 qwen35
  u32 n_bytes     text length (0 .. 2^24)
  u8  text[n_bytes]
  u32 n_ends     piece count (<= 4096; long inputs are chunked by the generator)
  u32 ends[n_ends]
Written to stdout in batches; run: gen.py N_SHORT N_LONG SEED > cases.bin
"""
import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import toks_model as M  # noqa: E402

VARIANTS = ["gpt2", "cl100k", "qwen2", "qwen35"]


def emit(f, vi, s):
    raw = s.encode("utf-8")
    if len(raw) >= (1 << 24):
        raw = raw[: (1 << 24) - 1]
        s = raw.decode("utf-8", "ignore")
    want = M.hf_pieces(VARIANTS[vi], s)
    assert len(want) <= 4096, "generator bug: over-long piece list"
    f.write(struct.pack("<BI", vi, len(raw)))
    f.write(raw)
    f.write(struct.pack("<I", len(want)))
    for a, b in want:
        assert 0 <= a < b <= len(raw), (a, b, len(raw))
        f.write(struct.pack("<II", a, b))


def main():
    n_short = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    n_long = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = sys.stdout.buffer
    rng = random.Random(seed)
    total = 0
    # equal shares of short (gen_case) and long (gen_long) across the four variants
    for i in range(n_short):
        v = rng.randrange(4)
        s = M.gen_case(random.Random(rng.randrange(1 << 62)))
        emit(out, v, s)
        total += 1
    for i in range(n_long):
        v = rng.randrange(4)
        s = M.gen_long(random.Random(rng.randrange(1 << 62)))
        # gen_long can give up to 1000 chars (~4000 bytes): cap the record at 2^24 bytes (kept above)
        emit(out, v, s)
        total += 1
    sys.stderr.write(f"gen.py: wrote {total} cases\n")


if __name__ == "__main__":
    main()
