#!/usr/bin/env python3
"""tests/parity/run_kimi.py: Kimi K3 through toks.h (tests/driver/toks_driver.c) against the model's own
tokenizer (tests/parity/oracle_tiktoken.py: transformers' TikTokenTokenizer + tiktoken, pinned), at scale.

    uv run --python 3.12 --with tiktoken==0.14.0 --with transformers==5.18.0 python tests/parity/run_kimi.py \\
        --toks ./toks_driver --short 200000 --long 40 --exh 4 --vocab 20000 --texts texts.jsonl \\
        --seed 1 --shard 0/8 --out shard-0.json

Per text: ids in toks ALL / NONSPECIAL / NONE against the reference's serving / direct / none
(docs/models/kimi.md §2.2-2.3), and toks_decode of the serving ids (plain, TOKS_SKIP_SPECIAL) against the
reference's decode (skip_special_tokens false / true). Sources (tests/model/run_kimi_fuzz.py's generators):
  short   class-aware strings heavy on CJK, every special kind and fragment, the 4 isspace-only chars
  long    the wrapper's cut edges: runs of 24,999..75,003 of every class, names straddling the 25,000 cut and the
          400,000 chunk edge, texts over 400,000 code points
  exh     every string of 0..EXH atoms over run_kimi_fuzz.REP (25 behaviours)
  vocab   1..30 random Kimi tokens joined (their utf-8 ones)
  texts   a jsonl file of {"text": ...} (e.g. a gen_cases.py case set's texts), line i to shard i mod N
Generated sources split by shard (each shard its own seeded stream: seed * 1000003 + shard); exh and texts
by index. Exit 0 iff no mismatch. A reference error (the reference raised) is counted, never compared.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tests", "model"))

import oracle_tiktoken  # noqa: E402
import run_kimi_fuzz as G  # noqa: E402
from run import Driver  # noqa: E402

MODES = (("serving", 0), ("direct", 1), ("none", 2))


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toks", required=True, help="toks driver argv (quoted)")
    ap.add_argument("--kimi-dir", default=oracle_tiktoken.KIMI_DIR)
    ap.add_argument("--short", type=int, default=0)
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--exh", type=int, default=-1, help="every string of 0..EXH atoms (-1: none)")
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--texts", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--any-files", action="store_true", help="another kimi repo's files (no K3 sha256 pins)")
    a = ap.parse_args()
    if a.any_files:
        oracle_tiktoken.check_files = lambda d: None
    si, sn = (int(x) for x in a.shard.split("/"))

    ref = oracle_tiktoken.Kimi(a.kimi_dir)
    drv = Driver(a.toks.split())
    drv.load(a.kimi_dir)
    toks_vocab = list(ref.tok.model._mergeable_ranks.keys())

    def sources():
        rng = random.Random(a.seed * 1000003 + si)
        for _ in range(a.short // sn + (1 if si < a.short % sn else 0)):
            yield "short", G.gen_short(rng)
        for _ in range(a.long // sn + (1 if si < a.long % sn else 0)):
            yield "long", G.gen_long(rng)
        if a.exh >= 0:
            i = 0
            for n in range(a.exh + 1):
                for combo in itertools.product(G.REP, repeat=n):
                    if i % sn == si:
                        yield "exh", "".join(combo)
                    i += 1
        for _ in range(a.vocab // sn + (1 if si < a.vocab % sn else 0)):
            yield "vocab", G.gen_vocab(rng, toks_vocab)
        if a.texts:
            with open(a.texts, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i % sn == si:
                        yield "texts", json.loads(line)["text"]

    t0 = time.time()
    rep = {"shard": a.shard, "seed": a.seed, "cases": {}, "ids": 0, "bytes": 0, "mismatches": 0, "ref_errors": 0,
           "skipped": 0, "compared": 0, "examples": []}
    for src, text in sources():
        c = rep["cases"].setdefault(src, {"texts": 0, "mismatch": 0})
        try:
            tb = text.encode("utf-8")
        except UnicodeEncodeError:                      # a lone surrogate: no utf-8, no toks input
            rep["skipped"] += 1
            continue
        c["texts"] += 1
        rep["bytes"] += len(tb)
        serving = None
        for name, fl in MODES:
            want = ref.safe_encode(text, name)
            if isinstance(want, dict):
                rep["ref_errors"] += 1
                continue
            st, got = drv.encode(tb, fl)
            if name == "serving":
                serving = want
            rep["ids"] += len(want)
            rep["compared"] += 1
            if st != 0 or got != want:
                rep["mismatches"] += 1
                c["mismatch"] += 1
                if len(rep["examples"]) < 20:
                    rep["examples"].append({"src": src, "mode": name, "status": st, "len": len(text),
                                            "n_want": len(want), "n_got": len(got or []),
                                            "first_diff": first_diff(got or [], want), "text": text[:400]})
        if serving is not None:
            for skip in (0, 1):
                want = ref.decode(serving, bool(skip)).encode("utf-8")
                st, got = drv.decode(serving, skip)
                rep["compared"] += 1
                if st != 0 or got != want:
                    rep["mismatches"] += 1
                    c["mismatch"] += 1
                    if len(rep["examples"]) < 20:
                        rep["examples"].append({"src": src, "mode": f"decode skip {skip}", "status": st,
                                                "len": len(text), "text": text[:400]})
    drv.close()
    rep["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}))
    # the count line, "PASS|FAIL <n> compared parity kimi shard <i> (<detail>)": encodes (per mode) and decodes
    # compared; reference errors and skipped (lone-surrogate) texts are not compared and are named in the detail;
    # nothing compared fails, and so does a run whose texts held no ids at all (--short 0 still yields the empty
    # text: 5 comparisons of nothing)
    why = [w for w, bad in ((f"{rep['mismatches']} mismatches", rep["mismatches"]),
                            ("nothing compared", rep["compared"] == 0),
                            ("no ids compared (only empty texts)", rep["compared"] > 0 and rep["ids"] == 0)) if bad]
    detail = (f"{sum(c['texts'] for c in rep['cases'].values())} texts, {rep['ids']} ids, {rep['ref_errors']} reference "
              f"errors, {rep['skipped']} skipped")
    print(f"{'FAIL' if why else 'PASS'} {rep['compared']} compared parity kimi shard {a.shard} "
          f"({'; '.join(why) if why else detail})", flush=True)
    sys.exit(1 if why else 0)


if __name__ == "__main__":
    main()
