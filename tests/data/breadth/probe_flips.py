#!/usr/bin/env python3
"""tests/data/breadth/probe_flips.py PROBE_TSV [--coverage census/coverage.json]

Compares a probe run (tests/data/breadth/probe_census.sh: tools/census/probe.c with this branch's library)
with the probe column of census/coverage.json (master's library) for every census tokenizer.json, and prints
the ones whose outcome changed -- "loads", "loads once NFC lands (m1b)" or the refusal that now comes
first -- with their (b) downloads. Runs anywhere (plain python, no hf).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def outcome(raw, nfc):
    if raw[0] == 0:
        return "loads"
    if nfc is not None and nfc[0] == 0:
        return "loads once NFC lands (m1b)"
    r = nfc if nfc is not None else raw
    return f"refused: {r[2]}"


def old_outcome(p):
    if not p:
        return "?"
    if p.get("stage") == 0:
        return "loads"
    a = p.get("after_nfc") or {}
    if a.get("stage") == 0:
        return "loads once NFC lands (m1b)"
    return "refused: " + (a.get("what") or p.get("what") or "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--coverage", default=os.path.join(HERE, "..", "..", "..", "census", "coverage.json"))
    a = ap.parse_args()
    rows = {}
    for line in open(a.tsv, encoding="utf-8"):
        path, stage, code, what = line.rstrip("\n").split("\t", 3)
        nfc = path.startswith("nfc-off:")
        sha = os.path.basename(path.replace("nfc-off:", "")).split("_", 1)[1]
        rows.setdefault(sha, {})["nfc" if nfc else "raw"] = (int(stage), int(code), what)
    cov = json.load(open(a.coverage, encoding="utf-8"))
    total_b = cov["totals"]["b_downloads"]
    flips, same, missing, disagree = [], 0, 0, 0
    for tid, t in cov["tokenizers"].items():
        if t.get("kind") != "tokenizer.json":
            continue
        r = rows.get(t["file_sha256"])
        if r is None:
            missing += 1
            continue
        disagree += r["raw"][0] == 8
        new, old = outcome(r["raw"], r.get("nfc")), old_outcome(t.get("probe"))
        if new == old:
            same += 1
        else:
            flips.append((t.get("dl_b", 0), tid, t["models"][0], old, new))
    flips.sort(reverse=True)
    print(f"census tokenizer.json files: {same} unchanged, {len(flips)} changed, {missing} not in the probe run, "
          f"{disagree} where the stages and toks_load_mem_copy disagree")
    for want in ("loads", "loads once NFC lands (m1b)"):
        sel = [f for f in flips if f[4] == want]
        dl = sum(f[0] for f in sel)
        print(f"\n-> {want}: {len(sel)} tokenizers, {dl:,} (b) downloads ({100.0 * dl / total_b:.2f}% of (b))")
        for f in sel:
            print(f"   {f[1]}  {f[2]:58s} {f[0]:>11,}  was: {f[3]}")
    rest = [f for f in flips if f[4] not in ("loads", "loads once NFC lands (m1b)")]
    print(f"\n-> the next refusal: {len(rest)} tokenizers")
    for f in rest:
        print(f"   {f[1]}  {f[2]:58s} {f[0]:>11,}  {f[3]}  ->  {f[4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
