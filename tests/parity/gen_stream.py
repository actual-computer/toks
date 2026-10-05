#!/usr/bin/env python3
"""tests/parity/gen_stream.py — adversarial stream cases for tests/parity/run.py.

    uv run --with tokenizers==0.23.2 python tests/parity/gen_stream.py \\
        --tokenizer ~/.cache/toks/tokenizers/gpt2 --out build/parity/gpt2-stream-adv.jsonl [--n 20000]

gen_cases.py's stream cases cut encoded text, which is well-formed utf-8 almost everywhere. These cut
random id sequences drawn mostly from where a stream decoder can go wrong: ByteLevel: tokens whose bytes
are not well-formed utf-8 on their own (partial characters, by hf's own token strings through the
ByteLevel alphabet) and single high-byte tokens (stray leads and continuations); sentencepiece-style:
the <0xHH> byte tokens (ByteFallback runs, ascii and high); both: added tokens (special and not, for
SKIP_SPECIAL), plus uniform ids; pushes of 0..6 ids (empty pushes included); every sequence with flags 0
and 1. After them (the lines before are unchanged): sentencepiece-style, every long-run sequence again
with flags 2 and 3 (a caller's hold, toks_stream_hold, grown on TOKS_E_LIMIT: no case refused), and the
U+13000 runs (12, 300, and 12 then a stray 0x80, its byte tokens <0xF0> <0x93> <0x80> <0x80>) one id per
push and in random pushes, flags 0 to 3. --with-text first writes the stream cases gen_cases.py writes
(encoded text, cut). Seeded: the same tokenizer and arguments give byte-identical output; the sha-256 is
printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toks_oracle import oracle as O  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000, help="id sequences (each written with flags 0 and 1)")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--with-text", action="store_true",
                    help="first the stream cases gen_cases.py writes for this tokenizer (encoded text, cut)")
    args = ap.parse_args()

    tok = O.load(args.tokenizer)
    t = tok.t
    nv = t.get_vocab_size(True)
    import tokenizers

    u2b = O._bytelevel_map()
    bytelevel = isinstance(t.decoder, tokenizers.decoders.ByteLevel)
    partial, high, added = [], [], sorted(int(i) for i in t.get_added_tokens_decoder())
    for i in range(nv):
        s = t.id_to_token(i)
        if s is None:
            continue
        if not bytelevel:            # sentencepiece-style: the <0xHH> byte tokens (ByteFallback runs)
            if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
                try:
                    b = int(s[3:5], 16)
                except ValueError:
                    continue
                (high if b >= 0x80 else partial).append(i)
            continue
        b = bytes(u2b[c] for c in s) if all(c in u2b for c in s) else s.encode("utf-8")
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            partial.append(i)
        if len(b) == 1 and b[0] >= 0x80:
            high.append(i)
    rng = random.Random(args.seed)

    def pick():
        x = rng.random()
        if x < 0.35 and partial:
            return rng.choice(partial)
        if x < 0.55 and high:
            return rng.choice(high)
        if x < 0.65 and added:
            return rng.choice(added)
        return rng.randrange(nv)

    h = hashlib.sha256()
    n = 0
    long_runs = []                   # the long-run sequences' pushes, written again with a caller's hold
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        if args.with_text:
            import gen_cases  # noqa: E402  (tests/parity, beside this file)

            for c in gen_cases.build_cases(tok, False, cp_sample=50000):
                if c["op"] == "stream":
                    line = json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n"
                    h.update(line.encode("utf-8"))
                    f.write(line)
                    n += 1
        for _ in range(args.n):
            ids = [pick() for _ in range(rng.randint(0, 48))]
            long_run = False
            if not bytelevel and partial and rng.random() < 0.03:
                # a long run of ascii byte tokens: valid so far, so the stream holds it (over 44 bytes it
                # must refuse with TOKS_E_LIMIT in st's own bytes)
                ids = [pick()] + [rng.choice(partial) for _ in range(rng.randint(30, 60))] + [pick()]
                long_run = True
            pushes, i = [], 0
            while i < len(ids) or not pushes:
                k = rng.randint(0, 6)
                pushes.append(ids[i:i + k])
                i += k
            if long_run:
                long_runs.append(pushes)
            for fl in (0, 1):
                line = json.dumps({"op": "stream", "gen": "stream_adv", "pushes": pushes, "flags": fl},
                                  sort_keys=True) + "\n"
                h.update(line.encode("utf-8"))
                f.write(line)
                n += 1
        hold = []                    # (pushes, flags) with a caller's hold
        for pushes in long_runs:
            hold += [(pushes, 2), (pushes, 3)]
        byte_id = {}
        if not bytelevel:            # sentencepiece-style: the byte of each <0xHH> token (partial / high hold
            for i in partial + high:     # alphabet tokens on a ByteLevel file: nothing to parse there)
                try:
                    byte_id.setdefault(int(t.id_to_token(i)[3:5], 16), i)
                except ValueError:
                    continue
        if not bytelevel and all(b in byte_id for b in (0xF0, 0x93, 0x80)):
            hrng = random.Random(args.seed + 1)
            u = [byte_id[0xF0], byte_id[0x93], byte_id[0x80], byte_id[0x80]]
            for ids in (u * 12, u * 300, u * 12 + [byte_id[0x80]]):
                parts, i = [], 0
                while i < len(ids):
                    k = hrng.randint(1, 9)
                    parts.append(ids[i:i + k])
                    i += k
                for pushes in ([[x] for x in ids], parts):
                    hold += [(pushes, fl) for fl in (0, 1, 2, 3)]
        for pushes, fl in hold:
            line = json.dumps({"op": "stream", "gen": "stream_hold", "pushes": pushes, "flags": fl},
                              sort_keys=True) + "\n"
            h.update(line.encode("utf-8"))
            f.write(line)
            n += 1
    print(json.dumps({"tokenizer": os.path.basename(args.tokenizer), "n_cases": n, "partial_ids": len(partial),
                      "high_byte_ids": len(high), "added_ids": len(added), "sha256": h.hexdigest()}))


if __name__ == "__main__":
    main()
