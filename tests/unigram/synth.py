#!/usr/bin/env python3
"""tests/unigram/synth.py: synthetic Unigram tokenizers against hf tokenizers 0.23.2 (SPEC §2.7, §14.4).

The pinned vocabularies cannot show some rules: real sentencepiece scores are log-probabilities <= 0 and every
char inside a multi-char XLM-R piece also has its own piece, so the unk penalty, the "single-char piece at this
start" test, duplicate pieces, positive scores, MissingUnkId and byte_fallback with missing <0xXX> pieces never
decide anything there. Here every tokenizer is random: a small alphabet, pieces of 1-4 chars with scores drawn
from a few values (ties on purpose) in [-25, +12], duplicates, unk_id present / absent / pointing at a normal
piece, byte_fallback with some byte pieces missing, and the pre-tokenizer shapes of the census (none, Metaspace
always / never x split, WhitespaceSplit + Metaspace). Each tokenizer encodes random strings over its alphabet
in the model and in hf; an hf exception must be an exception in the model too.

  uv run --with tokenizers==0.23.2 tests/unigram/synth.py --tokenizers 2000 --seed 1 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import unigram_model as um  # noqa: E402

ALPHA = ["a", "b", "c", "\u00e9", "\u65e5", "\u2581", "\U0001F642", "x"]
SCORES = [-25.0, -20.0, -12.5, -10.0, -9.5, -7.0, -3.0, -1.0, -0.5, 0.0, 0.5, 3.0, 9.5, 12.0, -10.000000000000002,
          -1e-300, 1.5e-5]


def make_tok(seed: int):
    rng = random.Random(seed)
    alpha = rng.sample(ALPHA, rng.randint(2, len(ALPHA)))
    pieces = []
    n = rng.randint(1, 40)
    for _ in range(n):
        p = "".join(rng.choice(alpha) for _ in range(rng.choice([1, 1, 2, 2, 3, 4])))
        pieces.append(p)
    if rng.random() < 0.3 and pieces:
        pieces.append(rng.choice(pieces))  # a duplicate: hf keeps the last id and its score
    byte_fallback = rng.random() < 0.3
    if byte_fallback:
        for b in rng.sample(range(256), rng.choice([0, 8, 64, 200, 256])):
            pieces.append(f"<0x{b:02X}>")
    rng.shuffle(pieces)
    vocab = [[p, rng.choice(SCORES)] for p in pieces]
    unk_mode = rng.random()
    if unk_mode < 0.7:
        k = rng.randrange(len(vocab) + 1)
        vocab.insert(k, ["<unk>", rng.choice(SCORES)])
        unk_id = k
    elif unk_mode < 0.85 and vocab:
        unk_id = rng.randrange(len(vocab))  # unk_id on an ordinary piece
    else:
        unk_id = None

    def meta(scheme, split):
        return {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": scheme, "split": split}
    shape = rng.choice(["none", "meta_always_split", "meta_always_nosplit", "meta_never_split", "meta_never_nosplit",
                        "ws_meta"])
    pre = {"none": None, "meta_always_split": meta("always", True), "meta_always_nosplit": meta("always", False),
           "meta_never_split": meta("never", True), "meta_never_nosplit": meta("never", False),
           "ws_meta": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, meta("always", True)]}}[shape]
    model = {"type": "Unigram", "unk_id": unk_id, "vocab": vocab, "byte_fallback": byte_fallback}
    tj = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
          "pre_tokenizer": pre, "post_processor": None,
          "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always", "split": True},
          "model": model}
    texts = []
    for _ in range(64):
        k = rng.choice([0, 1, 2, 3, 5, 8, 13])
        texts.append("".join(rng.choice(alpha + [" ", " ", "q"]) for _ in range(k)))
    return tj, texts


def run_one(seed: int):
    from tokenizers import Tokenizer
    tj, texts = make_tok(seed)
    raw = json.dumps(tj, ensure_ascii=False)
    try:
        hf = Tokenizer.from_str(raw)
        hf_load = None
    except BaseException as e:  # noqa: BLE001
        hf, hf_load = None, f"{type(e).__name__}"
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(raw)
    try:
        try:
            m = um.Tokenizer(path)
            m_load = None
        except (um.Unsupported, um.HfError) as e:
            m, m_load = None, type(e).__name__
    finally:
        os.remove(path)
    if (hf is None) != (m is None):
        return 1, 0, [{"seed": seed, "what": "load", "hf": hf_load, "model": m_load}]
    if hf is None:
        return 0, 0, []
    bad = []
    n = 0
    for t in texts:
        n += 1
        try:
            h = hf.encode(t).ids
        except BaseException:  # noqa: BLE001
            h = "error"
        try:
            mm = m.encode(t)
        except um.HfError:
            mm = "error"
        if h != mm:
            bad.append({"seed": seed, "text": t.encode("unicode_escape").decode(), "hf": h, "model": mm})
    return 0, n, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    seeds = [a.seed * 1_000_000 + i for i in range(a.tokenizers)]
    cases = 0
    nbad = 0
    load_bad = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for lb, n, bad in pool.imap_unordered(run_one, seeds, chunksize=8):
            load_bad += lb
            cases += n
            for b in bad:
                if nbad < 8:
                    print("MISMATCH", json.dumps(b, ensure_ascii=True)[:1500], flush=True)
                nbad += 1
    print("SUMMARY", json.dumps({"kind": "synthetic", "tokenizers": a.tokenizers, "seed": a.seed, "cases": cases,
                                 "mismatch": nbad, "load_mismatch": load_bad, "seconds": round(time.time() - t0, 1)}))
    sys.exit(1 if nbad or load_bad else 0)


if __name__ == "__main__":
    main()
