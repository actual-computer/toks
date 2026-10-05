#!/usr/bin/env python3
"""tests/unigram/run_diff.py: the unigram python model (tests/model/unigram_model.py) against hf tokenizers 0.23.2.

  uv run --with tokenizers==0.23.2 tests/unigram/run_diff.py --tok uni_bgem3 --kind random --n 200000 --workers 8
  uv run --with tokenizers==0.23.2 tests/unigram/run_diff.py --tok all --kind exhaustive
  uv run --with tokenizers==0.23.2 tests/unigram/run_diff.py --tok uni_pmminilm --kind real --text DIR_OR_FILES...
  uv run --with tokenizers==0.23.2 tests/unigram/run_diff.py --tok uni_pmminilm --kind scalars

Per case: encode in hf's default call (mode ALL: added tokens recognized, post-processor on, the file's own
truncation) and with add_special_tokens=False (mode ALL-nopp), both against the model; mode NONSPECIAL
(encode_special_tokens=True) whenever the text holds an added token's string (and every 16th case anyway);
decode(ids) (skip_special_tokens=True) of the default ids; decode with skip_special_tokens=False every 8th case.
Prints one summary line per tokenizer x kind and writes mismatches to --out (jsonl, texts escaped).
Exit status 1 on any mismatch. Heavy: run on a lab host (maintainer doctrine), never on the control-plane mac.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, HERE)

import gen  # noqa: E402
import unigram_model as um  # noqa: E402
from fetch import path_of, pins  # noqa: E402

CHUNK = 500

_W = {}


def esc(s: str) -> str:
    return s.encode("unicode_escape").decode("ascii")


def _prepare(tok: str, kind: str, seed: int, text_lines):
    """in the parent, before the fork: the model and the generator context are shared copy-on-write"""
    _W["m"] = um.Tokenizer(path_of(tok))
    _W["tok"], _W["kind"], _W["seed"] = tok, kind, seed
    _W["lines"] = text_lines
    _W["specials"] = sorted({t.content for t in _W["m"].added})
    _W["ctx"] = gen.make_ctx(_W["m"])


def _init(tok: str):
    from tokenizers import Tokenizer
    path = path_of(tok)
    _W["hf"] = Tokenizer.from_file(path)
    _W["hf_ns"] = Tokenizer.from_file(path)
    _W["hf_ns"].encode_special_tokens = True


def _case(idx: int):
    kind = _W["kind"]
    if kind == "random":
        return gen.gen_random(_W["seed"], idx, _W["specials"], _W["ctx"])
    if kind == "real":
        return gen.gen_real(_W["lines"], _W["seed"], idx, _W["specials"], _W["ctx"])
    if kind == "exhaustive":
        return gen.gen_exhaustive(idx)
    if kind == "scalars":
        return gen.gen_scalar(idx)
    raise SystemExit(f"unknown kind {kind}")


def _run_chunk(lo_hi):
    lo, hi = lo_hi
    hf, hf_ns, m = _W["hf"], _W["hf_ns"], _W["m"]
    texts = []
    idxs = []
    for i in range(lo, hi):
        t = _case(i)
        if t is not None:
            texts.append(t)
            idxs.append(i)
    stats = {"cases": 0, "ALL": 0, "ALL-nopp": 0, "NONSPECIAL": 0, "decode": 0, "decode-keep": 0, "mismatch": 0,
             "hf_error": 0}
    bad = []
    enc_a, enc_b = None, None
    if hf.padding is None:  # encode_batch pads a batch to its longest member; encode() of one text never does
        try:
            enc_a = [e.ids for e in hf.encode_batch(texts)]
            enc_b = [e.ids for e in hf.encode_batch(texts, add_special_tokens=False)]
        except BaseException:  # noqa: BLE001  (pyo3 panics are BaseException)
            enc_a, enc_b = None, None
    for k, (i, t) in enumerate(zip(idxs, texts)):
        stats["cases"] += 1
        if enc_a is None:
            try:
                ha = hf.encode(t).ids
                hb = hf.encode(t, add_special_tokens=False).ids
            except BaseException as e:  # noqa: BLE001
                ha = hb = f"hf error {type(e).__name__}: {e}"
                stats["hf_error"] += 1
        else:
            ha, hb = enc_a[k], enc_b[k]
        try:
            ma = m.encode(t)
            mb = m.encode(t, add_special_tokens=False)
        except (um.HfError, um.Unsupported) as e:
            ma = mb = f"model error {type(e).__name__}: {e}"
        for mode, h, mm in (("ALL", ha, ma), ("ALL-nopp", hb, mb)):
            stats[mode] += 1
            if h != mm:
                stats["mismatch"] += 1
                bad.append({"idx": i, "mode": mode, "text": esc(t), "hf": h, "model": mm})
        if any(s in t for s in _W["specials"]) or i % 16 == 0:
            stats["NONSPECIAL"] += 1
            try:
                h = hf_ns.encode(t).ids
            except BaseException as e:  # noqa: BLE001
                h = f"hf error {type(e).__name__}: {e}"
            try:
                mm = m.encode(t, drop_specials=True)
            except (um.HfError, um.Unsupported) as e:
                mm = f"model error {type(e).__name__}: {e}"
            if h != mm:
                stats["mismatch"] += 1
                bad.append({"idx": i, "mode": "NONSPECIAL", "text": esc(t), "hf": h, "model": mm})
        if isinstance(ha, list):
            stats["decode"] += 1
            hd = hf.decode(ha)
            md = m.decode(ha)
            if hd != md:
                stats["mismatch"] += 1
                bad.append({"idx": i, "mode": "decode", "text": esc(t), "ids": ha, "hf": esc(hd), "model": esc(md)})
            if i % 8 == 0:
                stats["decode-keep"] += 1
                hd = hf.decode(ha, skip_special_tokens=False)
                md = m.decode(ha, skip_special_tokens=False)
                if hd != md:
                    stats["mismatch"] += 1
                    bad.append({"idx": i, "mode": "decode-keep", "text": esc(t), "ids": ha, "hf": esc(hd),
                                "model": esc(md)})
    return stats, bad


def _load_lines(paths):
    lines = []
    for p in paths:
        files = [os.path.join(p, f) for f in sorted(os.listdir(p))] if os.path.isdir(p) else [p]
        for f in files:
            if os.path.isdir(f):
                continue
            with open(f, encoding="utf-8", errors="strict") as fh:
                for line in fh:
                    line = line.rstrip("\n")
                    if line:
                        lines.append(line)
    return lines


def loadavg() -> str:
    try:
        return " ".join(f"{x:.1f}" for x in os.getloadavg())
    except OSError:
        return "?"


def run(tok: str, kind: str, n: int, seed: int, workers: int, start: int, lines, out_path: str) -> int:
    if kind == "exhaustive":
        n = gen.EXHAUSTIVE_COUNT if n <= 0 else n
    elif kind == "scalars":
        n = gen.scalar_count() if n <= 0 else n
    chunks = [(lo, min(lo + CHUNK, start + n)) for lo in range(start, start + n, CHUNK)]
    t0 = time.time()
    la0 = loadavg()
    tot = {}
    nbad = 0
    _prepare(tok, kind, seed, lines)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(tok,)) as pool, \
            open(out_path, "a", encoding="utf-8") as out:
        for stats, bad in pool.imap_unordered(_run_chunk, chunks):
            for k, v in stats.items():
                tot[k] = tot.get(k, 0) + v
            for b in bad:
                b["tok"], b["kind"], b["seed"] = tok, kind, seed
                out.write(json.dumps(b, ensure_ascii=True) + "\n")
                if nbad < 5:
                    print("MISMATCH", json.dumps(b, ensure_ascii=True)[:2000], flush=True)
                nbad += 1
    dt = time.time() - t0
    summary = {"tok": tok, "kind": kind, "seed": seed, "start": start, "n": n, **tot, "seconds": round(dt, 1),
               "host": socket.gethostname(), "workers": workers, "load_start": la0, "load_end": loadavg()}
    print("SUMMARY", json.dumps(summary), flush=True)
    return nbad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, help="pin name from tests/unigram/pins.json, or all")
    ap.add_argument("--kind", default="random", choices=["random", "real", "exhaustive", "scalars"])
    ap.add_argument("--n", type=int, default=10000, help="cases (exhaustive / scalars: 0 = all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--text", nargs="*", default=[], help="files or dirs of utf-8 lines (kind real)")
    ap.add_argument("--out", default="unigram_mismatches.jsonl")
    a = ap.parse_args()
    toks = [t for t in pins() if t not in ("uni_albert", "uni_xlnet")] if a.tok == "all" else a.tok.split(",")
    lines = _load_lines(a.text) if a.kind == "real" else []
    if a.kind == "real" and not lines:
        raise SystemExit("kind real needs --text")
    bad = 0
    for t in toks:
        bad += run(t, a.kind, a.n, a.seed, a.workers, a.start, lines, a.out)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
