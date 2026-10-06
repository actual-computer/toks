#!/usr/bin/env python3
"""tools/bench/e2e_commits.py: the before / after table of tools/bench/e2e_commits.sh's log.

    python3 tools/bench/e2e_commits.py build/commits.log [docs/bench/e2e.md]

Per cell (tokenizer x corpus): MB/s = corpus bytes / seconds of the state (e2e.c: best of reps inside each run), the
median over each side's runs (abba: two runs a side per round). A cell whose runs disagree on the ids sha is VOID.
With docs/bench/e2e.md, hf 0.23.2's and tiktoken's MB/s of the same tokenizer / corpus / chunk on a host of the same
kind are shown for scale (their own receipts, another run).
"""
import re
import statistics
import sys

STATES = ("cold", "pass", "warm", "lang")       # lang (lang-x): runs with e2e.c's OTHER text only


def parse(path):
    cells, order, meta = {}, [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith(("HOST", "COMMITS", "ENV", "UPTIME", "SKIP")):
            meta.append(line.rstrip())
            continue
        if not line.startswith("RUN "):
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        key = (kv["tk"], kv["corp"])
        if key not in cells:
            cells[key] = {"A": [], "B": [], "sha": set(), "load": []}
            order.append(key)
        c = cells[key]
        n = int(kv["bytes"])
        c[kv["side"]].append({s: n / float(kv[s + "_s"]) / 1e6 for s in STATES if float(kv.get(s + "_s", 0)) > 0})
        c["sha"].add(kv["sha"])
        c["load"].append(float(kv["load"]))
    return cells, order, meta


def refs(path):
    """hf / tiktoken MB/s per (tokenizer, corpus) from docs/bench/e2e.md's 4096-byte rows on GB10 hosts (gb10* labels)."""
    out = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return out
    for ln in lines:
        f = [x.strip() for x in ln.split("|")]
        if len(f) < 14 or not f[1].startswith("gb10") or f[4] != "4096" or f[5] != "neon":
            continue
        try:
            hf = float(f[11])
        except ValueError:
            continue
        try:
            tt = float(f[12])
        except ValueError:
            tt = None
        out[(f[2], f[3])] = (hf, tt)
    return out


def main():
    cells, order, meta = parse(sys.argv[1])
    ref = refs(sys.argv[2]) if len(sys.argv) > 2 else {}
    for m in meta:
        print(m)
    w = max(len(f"{tk} {corp}") for tk, corp in order) if order else 10
    print(f"\n{'cell':<{w}}  {'cold MB/s':>17}  {'pass MB/s':>17}  {'warm MB/s':>17}  {'lang-x MB/s':>17}  {'hf / tiktoken':>15}  load")
    for key in order:
        c = cells[key]
        tk, corp = key
        if len(c["sha"]) != 1:
            print(f"{tk + ' ' + corp:<{w}}  VOID: ids differ between runs {sorted(c['sha'])}")
            continue
        cols = []
        for s in STATES:
            if not all(s in r for r in c["A"] + c["B"]):
                cols.append("-")
                continue
            a = statistics.median(r[s] for r in c["A"])
            b = statistics.median(r[s] for r in c["B"])
            cols.append(f"{a:7.0f} -> {b:7.0f}")
        r = ref.get(key)
        rs = (f"{r[0]:6.1f} / {r[1]:6.1f}" if r[1] is not None else f"{r[0]:6.1f} /    n/a") if r else ""
        print(f"{tk + ' ' + corp:<{w}}  {cols[0]:>17}  {cols[1]:>17}  {cols[2]:>17}  {cols[3]:>17}  {rs:>15}  "
              f"{min(c['load']):.1f}-{max(c['load']):.1f}")


if __name__ == "__main__":
    main()
