#!/usr/bin/env python3
"""tests/parity/gen_cases.py — generate the parity case set for one tokenizer.

    uv run --with tokenizers==0.23.2 python tests/parity/gen_cases.py \
        --tokenizer ~/.cache/toks/tokenizers/gpt2 --out ~/.cache/toks/cases/gpt2.jsonl \
        [--quick]

Cases are the runner's jsonl (see run.py's header): every generator from
python/toks_oracle/gen.py (golden, codepoints, ws, scan, cjk, emoji, marks,
long, random, special over the tokenizer's own added literals) x every flag
combination, plus decode id sequences from the tokenizer's real vocab
(including out-of-range ids hf skips) and stream cases.

quick=True (default) is sized to finish in minutes on a lab host; quick=False
is the full set. The manifest (sha256 + counts) is printed on stdout and
written next to the file as <out>.manifest.json. Everything is seeded: the
same tree + arguments give byte-identical output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from toks_oracle import gen  # noqa: E402
from toks_oracle import oracle as O  # noqa: E402

TEXT_FLAG_COMBOS = [0, 1, 2, 4, 5, 6]      # ALL/NONSPECIAL/NONE x postproc on/off (NONE x pp irrelevant but harmless)
DECODE_FLAG_COMBOS = [0, 1]                 # plain / skip_special


def build_cases(tok: "O.Tok", quick: bool, cp_sample: int = 0) -> list:
    added = [a.content for a in tok.spec.added]
    lit = added[0] if added else "<|endoftext|>"
    gen_args = {
        "golden": (),
        "codepoints": (quick,),
        "random": (quick,),
        "ws": (quick,),
        "scan": (quick,),
        "cjk": (quick,),
        "emoji": (quick,),
        "marks": (quick,),
        "long": (quick,),
    }
    texts: dict = {name: getattr(gen, f"g_{name}")(*a) for name, a in gen_args.items()}
    # sizing: the codepoint sweep is sampled (full sweep = 1.1M scalars x
    # contexts x flags is a fuzz campaign, not a case set; quick generators stay
    # unsampled). The 13th context: chars inside this tokenizer's own first
    # added literal.
    if cp_sample:
        texts["codepoints"] = gen.g_codepoints(quick, sample=cp_sample)
    texts["codepoints"] = texts["codepoints"] + gen.g_codepoints(
        quick, sample=min(cp_sample, 4000) if cp_sample else None, extra_literal=lit)
    if added:
        texts["special"] = gen.g_special(quick, added)

    cases = []
    for name, ts in texts.items():
        for t in ts:
            for fl in TEXT_FLAG_COMBOS:
                cases.append({"op": "encode", "gen": name, "text": t, "flags": fl})
            # pieces: mode axis only (post-processing does not change pieces)
            cases.append({"op": "pieces", "gen": name, "text": t, "flags": fl & 3})  # noqa: B023

    # decode: real vocab + out-of-range ids hf skips
    nv = tok.t.get_vocab_size(True)
    aid = sorted(tok.added_ids())
    sid = sorted(tok.special_ids())
    for ids in gen.g_decode_ids(quick, nv, aid, sid):
        for fl in DECODE_FLAG_COMBOS:
            cases.append({"op": "decode", "gen": "decode_ids", "ids": ids, "flags": fl})

    # stream: encoded-id partitions of a sample of the generated texts
    rng = random.Random(41)
    sample = [t for name, ts in texts.items() for t in ts]
    step = max(1, len(sample) // (300 if quick else 4000))
    n_stream = 0
    for t in sample[::step]:
        ids = tok.encode(t, mode="ALL", add_special_tokens=False)
        if len(ids) > 64:
            ids = ids[:64]
        for _ in range(1 if quick else 3):
            pushes = _partition(rng, ids)
            for fl in DECODE_FLAG_COMBOS:
                cases.append({"op": "stream", "gen": "stream", "pushes": pushes, "flags": fl})
                n_stream += 1
    return cases


def _partition(rng: random.Random, ids: list) -> list:
    n = len(ids)
    if n == 0:
        return [[]]
    k = rng.randint(1, min(5, n))
    cuts = sorted(rng.sample(range(1, n), k - 1)) if k > 1 else []
    b = [0] + cuts + [n]
    return [ids[i:j] for i, j in zip(b, b[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="minutes-scale set (default: the full set)")
    ap.add_argument("--codepoint-sample", type=int, default=50000,
                    help="sampled scalars for the codepoint sweep (0 = all; default 50000)")
    args = ap.parse_args()
    quick = args.quick

    tok = O.load(args.tokenizer)
    cases = build_cases(tok, quick, cp_sample=args.codepoint_sample)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h = hashlib.sha256()
    counts: dict = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for c in cases:
            line = json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n"
            h.update(line.encode("utf-8"))
            f.write(line)
            counts[c["op"]] = counts.get(c["op"], 0) + 1

    manifest = {
        "tokenizer": os.path.basename(args.tokenizer),
        "quick": quick,
        "n_cases": len(cases),
        "counts": counts,
        "sha256": h.hexdigest(),
        "generators": sorted({c.get("gen", "?") for c in cases}),
        "toks_py_gen_git": os.environ.get("TOKS_GEN_GIT", ""),
    }
    mpath = args.out + ".manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "toks_py_gen_git"}, indent=2))
    print(f"wrote {args.out} and {mpath}")


if __name__ == "__main__":
    main()
