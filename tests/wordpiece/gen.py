#!/usr/bin/env python3
"""tests/wordpiece/gen.py: the wordpiece c differential's hf side (tests/wordpiece/run.sh).

Writes to stdout, for one pinned tokenizer.json, a binary stream of cases the c side (tests/wordpiece/check.c)
replays through toks_bert_normalize, toks_wp_scan_c and toks_wp_encode_c:

  "WPC1"
  per case: u32 n, text[n]                  the input (valid utf-8: hf takes str)
            u32 n, norm[n]                  hf's normalizer on the whole text
            u32 k, k x (u32 n, word[n])     hf's pre-tokenizer on norm (the pieces the model sees)
            u32 k, k x u32                  hf encode(text, add_special_tokens=False).ids of the file with
                                            added_tokens [], truncation and padding removed (the model alone)
  u32 0xFFFFFFFF

    uv run --with tokenizers==0.23.2 tests/wordpiece/gen.py <tokenizer.json> <n_generated> <seed> [text files...]
"""
import json
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))
sys.path.insert(0, HERE)

from tokenizers import Tokenizer  # noqa: E402
import run_wordpiece_fuzz as R  # noqa: E402  (its text generator)


def main():
    path, n_gen, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tj = json.load(open(path, encoding="utf-8"))
    tj["added_tokens"] = []
    tj["post_processor"] = None
    tj["truncation"] = None
    tj["padding"] = None
    tk = Tokenizer.from_str(json.dumps(tj))
    norm = tk.normalizer
    pre = tk.pre_tokenizer
    words = sorted(tj["model"]["vocab"])
    rng = random.Random(seed)
    texts = (R.gen_text(rng, words) for _ in range(n_gen))
    real = []
    for p in sys.argv[4:]:
        with open(p, encoding="utf-8") as f:
            real.extend(line.rstrip("\n") for line in f if line.strip())
    out = sys.stdout.buffer
    out.write(b"WPC1")

    def put(b):
        out.write(struct.pack("<I", len(b)))
        out.write(b)

    def case(t):
        try:
            ids = tk.encode(t, add_special_tokens=False).ids
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            return
        nt = norm.normalize_str(t) if norm is not None else t
        ws = [w for w, _ in pre.pre_tokenize_str(nt)] if pre is not None else [nt]
        put(t.encode("utf-8"))
        put(nt.encode("utf-8"))
        out.write(struct.pack("<I", len(ws)))
        for w in ws:
            put(w.encode("utf-8"))
        out.write(struct.pack("<I", len(ids)))
        out.write(struct.pack("<%dI" % len(ids), *ids))

    for t in texts:
        case(t)
    for i, t in enumerate(real):
        case(t)
        if i % 4 == 3:                                       # and joined documents
            case(" ".join(real[i - 3:i + 1]))
    out.write(struct.pack("<I", 0xFFFFFFFF))


if __name__ == "__main__":
    main()
