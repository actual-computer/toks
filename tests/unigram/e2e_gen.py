#!/usr/bin/env python3
"""tests/unigram/e2e_gen.py: the hf side of the Unigram end-to-end differential (toks.h against hf 0.23.2).

Streams cases for one pinned tokenizer to stdout for tests/unigram/e2e_check.c:
  encode: every text in modes ALL / NONSPECIAL / NONE x add_special_tokens on / off (python/toks_oracle's views:
          NONSPECIAL = encode_special_tokens, NONE = the file with added_tokens removed), hf encode() exactly
          (the file's truncation included);
  continuation: every text in mode NONE without the template and with TOKS_CONTINUATION, against hf on the file
          whose gap-start rules are gone (Strip keeps strip_right only, Replace '(?<!\n)^' -> '▁' removed:
          unigram.md §7) and whose added tokens are gone;
  pieces: every text in mode NONE: the ends of hf's pre-tokenizer splits of hf's normalized string, in bytes
          (SPEC §3.5);
  decode: the ALL ids of every 4th text with skip_special_tokens on and off, and random id sequences (pieces,
          byte pieces, specials, ids past the vocab are skipped by the C side's TOKS_E_ID rule: never sent).
Texts come from tests/unigram/gen.py (random | real | exhaustive). Wire (little endian):
  b"TKUE2E01", then records: u32 op (0 encode, 1 decode, 2 pieces), u32 flags (toks.h: 1 NONSPECIAL, 2 NONE,
  4 NO_POSTPROCESS, 8 CONTINUATION; decode: 1 SKIP_SPECIAL), u32 text_len, text, u32 n, n x u32 ids (pieces:
  ends), [decode: u32 len, bytes].

  uv run --with tokenizers==0.23.2 python tests/unigram/e2e_gen.py --tok uni_bgem3 --kind random --n 1000 \\
      | build/unigram-e2e/e2e_check ~/.cache/toks/tokenizers/uni_bgem3
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import struct
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

import gen  # noqa: E402
import unigram_model as um  # noqa: E402
from fetch import path_of  # noqa: E402
from toks_oracle import oracle as O  # noqa: E402

MODES = ((0, "ALL"), (1, "NONSPECIAL"), (2, "NONE"))


def continuation_twin(path: str):
    """hf on the file without its gap-start rules and without added tokens (TOKS_CONTINUATION, mode NONE)."""
    from tokenizers import Tokenizer

    d = json.load(open(path, encoding="utf-8"))
    d = copy.deepcopy(d)
    d["added_tokens"] = []

    def fix(x):
        if x is None:
            return None
        if x.get("type") == "Strip":
            x = dict(x)
            x["strip_left"] = False
            return x
        if x.get("type") == "Replace" and x["pattern"].get("Regex") == "(?<!\n)^":
            return None
        return x

    n = d.get("normalizer")
    if n is not None and n.get("type") == "Sequence":
        n["normalizers"] = [y for y in (fix(x) for x in n["normalizers"]) if y is not None]
    else:
        d["normalizer"] = fix(n)
    return Tokenizer.from_str(json.dumps(d))


def hf_pieces(t, text: str) -> list[int]:
    """the byte ends of hf's pre-tokenizer splits of hf's normalized text (mode NONE: one gap)"""
    norm = t.normalizer.normalize_str(text) if t.normalizer is not None else text
    if t.pre_tokenizer is None:
        splits = [(norm, (0, len(norm)))] if norm else []
    else:
        splits = t.pre_tokenizer.pre_tokenize_str(norm)
    return [len(norm[:e].encode("utf-8")) for _, (_, e) in splits]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--kind", default="random", choices=["random", "real", "exhaustive"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--text", nargs="*", default=[])
    a = ap.parse_args()
    k, nsh = (int(x) for x in a.shard.split("/"))
    path = path_of(a.tok)
    tok = O.load(path)
    twin = continuation_twin(path)
    model = um.Tokenizer(path)
    specials = sorted({t.content for t in model.added})
    ctx = gen.make_ctx(model)
    lines = []
    if a.kind == "real":
        for p in a.text:
            files = [os.path.join(p, f) for f in sorted(os.listdir(p))] if os.path.isdir(p) else [p]
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    lines.extend(x.rstrip("\n") for x in fh if x.strip())
    n = gen.EXHAUSTIVE_COUNT if a.kind == "exhaustive" and a.n <= 0 else a.n
    out = sys.stdout.buffer
    out.write(b"TKUE2E01")
    vocab_n = tok.t.get_vocab_size(with_added_tokens=True)
    rng = random.Random(a.seed * 7919 + k)
    for i in range(a.start + k, a.start + n, nsh):
        if a.kind == "random":
            t = gen.gen_random(a.seed, i, specials, ctx)
        elif a.kind == "real":
            t = gen.gen_real(lines, a.seed, i, specials, ctx)
        else:
            t = gen.gen_exhaustive(i)
        tb = t.encode("utf-8")
        all_ids = None
        for fl, mode in MODES:
            for pp in (1, 0):
                ids = tok.encode(t, mode=mode, add_special_tokens=bool(pp))
                if fl == 0 and pp == 1:
                    all_ids = ids
                out.write(struct.pack("<III", 0, fl | (0 if pp else 4), len(tb)) + tb)
                out.write(struct.pack(f"<I{len(ids)}I", len(ids), *ids))
        ids = twin.encode(t, add_special_tokens=False).ids     # TOKS_CONTINUATION | NONE | NO_POSTPROCESS
        out.write(struct.pack("<III", 0, 2 | 4 | 8, len(tb)) + tb)
        out.write(struct.pack(f"<I{len(ids)}I", len(ids), *ids))
        ends = hf_pieces(tok.t, t)                           # toks_pieces, mode NONE
        out.write(struct.pack("<III", 2, 2, len(tb)) + tb)
        out.write(struct.pack(f"<I{len(ends)}I", len(ends), *ends))
        if i % 4 == 0:
            for skip in (1, 0):
                d = tok.decode(all_ids, skip_special_tokens=bool(skip)).encode("utf-8")
                out.write(struct.pack("<III", 1, skip, 0))
                out.write(struct.pack(f"<I{len(all_ids)}I", len(all_ids), *all_ids))
                out.write(struct.pack("<I", len(d)) + d)
        if i % 16 == 0:                                     # random id sequences through decode
            m = rng.choice([1, 2, 3, 5, 8, 20])
            ids = [rng.randrange(vocab_n) for _ in range(m)]
            if rng.random() < 0.5:                          # byte pieces in runs (byte fallback models)
                ids += [model.model.tok2id.get(f"<0x{b:02X}>", 0) for b in rng.choice([b"\xe3\x81\x82", b"\xff", b"a\xc3", b"\xe2\x96\x81"])]
            for skip in (1, 0):
                d = tok.decode(ids, skip_special_tokens=bool(skip)).encode("utf-8")
                out.write(struct.pack("<III", 1, skip, 0))
                out.write(struct.pack(f"<I{len(ids)}I", len(ids), *ids))
                out.write(struct.pack("<I", len(d)) + d)
    out.flush()


if __name__ == "__main__":
    main()
