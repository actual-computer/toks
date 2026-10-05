"""python/bench/bench.py: per-call overhead and throughput of the toks wheel vs hf tokenizers and tiktoken,
single thread, through each library's Python API (what a Python caller pays).

    uv run --no-project --with build/wheels/<toks wheel> --with tokenizers==0.23.2 --with tiktoken \\
        python python/bench/bench.py --corpus <english.txt> [--tok gpt2 --tok llama3] [--seconds 0.3]

Cells: '' (pure call overhead), 64 B (1024 distinct slices of the corpus, cycled) and 64 KiB (16 distinct
slices), each library's default encode returning a list of ints: toks Tokenizer.encode(t), hf
Tokenizer.encode(t).ids, tiktoken Encoding.encode(t) and its fastest, encode_ordinary(t) (the ratio
uses the faster of the two). tiktoken gets the same vocabulary (ranks = the
tokenizer.json ids, pattern = its pre-tokenizer regex, the added tokens as specials), built here from the
pinned file: nothing is downloaded. toks's ids are checked equal to hf's on every input before timing;
tiktoken's (encode_ordinary vs hf without the post-processor) are reported. Timing: rounds in A B C C B A order,
best and median of --rounds rounds of >= --seconds each; load and the busiest processes are recorded
before and after.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import tiktoken
import tokenizers

import toks

TOKENIZERS = os.path.expanduser(os.environ.get("TOKS_TOKENIZER_CACHE", "~/.cache/toks/tokenizers"))


def bytelevel_decoder():
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


GPT2_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tiktoken_for(path):
    obj = json.load(open(path, encoding="utf-8"))
    dec = bytelevel_decoder()
    ranks = {bytes(dec[c] for c in s): i for s, i in obj["model"]["vocab"].items()}
    pat = GPT2_PAT
    pre = obj.get("pre_tokenizer") or {}
    for p in [pre] + list(pre.get("pretokenizers", [])):
        if p.get("type") == "Split":
            pat = p["pattern"]["Regex"]
    specials = {a["content"]: a["id"] for a in obj.get("added_tokens", []) if a["content"].encode() not in ranks}
    return tiktoken.Encoding(name=os.path.basename(path), pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)


def load():
    try:
        la = os.getloadavg()
    except OSError:
        la = (-1, -1, -1)
    top = ""
    try:
        ps = subprocess.run(["ps", "-Ao", "pcpu,comm", "-r"] if sys.platform == "darwin"
                            else ["ps", "-eo", "pcpu,comm", "--sort=-pcpu"], capture_output=True, text=True).stdout
        top = "; ".join(" ".join(x.split()) for x in ps.splitlines()[1:5])
    except OSError:
        pass
    return {"loadavg": [round(x, 2) for x in la], "top": top}


def time_call(fn, inputs, seconds):
    """ns per call: cycles through inputs until >= seconds elapsed"""
    n = len(inputs)
    reps = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            for x in inputs:
                fn(x)
        dt = time.perf_counter_ns() - t0
        if dt >= seconds * 1e9:
            return dt / (reps * n)
        reps = max(reps * 2, int(reps * seconds * 1.2e9 / max(dt, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tok", action="append")
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out")
    a = ap.parse_args()
    raw = open(a.corpus, "rb").read()
    text = raw.decode("utf-8", "ignore")
    enc = text.encode()

    def slices(size, count):
        out, step = [], max(1, (len(enc) - size) // count)
        for k in range(count):
            b = enc[k * step: k * step + size]
            out.append(b.decode("utf-8", "ignore"))
        return out

    cells = {"empty": [""], "64B": slices(64, 1024), "64KiB": slices(65536, 16)}
    res = {"host": platform.node(), "machine": platform.machine(), "python": sys.version.split()[0],
           "versions": {"toks": toks.__version__, "tokenizers": tokenizers.__version__,
                        "tiktoken": getattr(tiktoken, "__version__", "?")},
           "corpus": os.path.basename(a.corpus), "corpus_bytes": len(raw), "load_before": load(), "cells": {}}
    for name in a.tok or ["gpt2", "llama3"]:
        path = os.path.join(TOKENIZERS, name)
        t, h, k = toks.Tokenizer.from_file(path), tokenizers.Tokenizer.from_file(path), tiktoken_for(path)
        res.setdefault("tiers", {})[name] = t.info()["tier"]
        libs = {"toks": t.encode, "hf": lambda x, h=h: h.encode(x).ids, "tiktoken": k.encode,
                "tiktoken_ordinary": k.encode_ordinary}
        for cell, inputs in cells.items():
            want = [libs["hf"](x) for x in inputs]
            assert [libs["toks"](x) for x in inputs] == want, f"toks != hf on {name} {cell}"
            plain = [h.encode(x, add_special_tokens=False).ids for x in inputs]      # tiktoken has no bos
            tk_equal = sum(k.encode_ordinary(x) == w for x, w in zip(inputs, plain))
            nbytes = sum(len(x.encode()) for x in inputs) / len(inputs)
            nids = sum(len(w) for w in want) / len(inputs)
            ns = {lib: [] for lib in libs}
            order = list(libs)
            for r in range(a.rounds):
                for lib in (order if r % 2 == 0 else order[::-1]):
                    ns[lib].append(time_call(libs[lib], inputs, a.seconds))
            row = {"bytes_per_call": round(nbytes, 1), "ids_per_call": round(nids, 1),
                   "tiktoken_equal_inputs": f"{tk_equal}/{len(inputs)}"}
            for lib, v in ns.items():
                best, med = min(v), statistics.median(v)
                row[lib] = {"ns_best": round(best, 1), "ns_median": round(med, 1),
                            "MBps_best": round(nbytes / best * 1e3, 2) if nbytes else None}
            row["toks_vs_hf"] = round(row["hf"]["ns_best"] / row["toks"]["ns_best"], 2)
            tk = min(row["tiktoken"]["ns_best"], row["tiktoken_ordinary"]["ns_best"])
            row["toks_vs_tiktoken"] = round(tk / row["toks"]["ns_best"], 2)
            res["cells"][f"{name}/{cell}"] = row
            print(f"{name:7s} {cell:6s} {nbytes:8.0f} B {nids:7.1f} ids | toks {row['toks']['ns_best']:>11.1f} ns"
                  f" | hf {row['hf']['ns_best']:>11.1f} ns | tiktoken {row['tiktoken']['ns_best']:>11.1f}"
                  f" / {row['tiktoken_ordinary']['ns_best']:.1f} ns"
                  f" | x{row['toks_vs_hf']} vs hf, x{row['toks_vs_tiktoken']} vs tiktoken"
                  f" (tiktoken == hf on {row['tiktoken_equal_inputs']})", flush=True)
    res["load_after"] = load()
    print(json.dumps({k: res[k] for k in ("host", "python", "versions", "tiers", "load_before", "load_after")}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
