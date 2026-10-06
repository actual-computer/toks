#!/usr/bin/env python3
"""tools/bench/tokv1_ab.py: the before / after table of tools/bench/tokv1.sh runs of two commits (abba: each run's log
named ...-A.log or ...-B.log by the side it measured). Per cell (tokenizer, corpus, chunk) and state: toks' MB/s with
the memo (side toksm), the median over a side's runs of each run's median, A -> B with the ratio; and toks' x against
tok v1 (x_med), the same medians, A -> B.

    python3 tools/bench/tokv1_ab.py docs/bench/raw/tokv1-memo-check-gb10c-*.log
"""
import re
import statistics
import sys

cells = {}
for path in sys.argv[1:]:
    side = path.rsplit("-", 1)[1].split(".")[0]
    if side not in ("A", "B"):
        sys.exit(f"{path}: the name must end in -A.log or -B.log")
    cur = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"== (\S+) (\S+) chunk (\S+)", line)
        if m:
            cur = (m.group(1), m.group(2), m.group(3))
            continue
        if line.startswith("TV1 ") and cur:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            if kv.get("side") != "toksm":
                continue
            cells.setdefault(cur + (kv["state"],), {"A": [], "B": []})[side].append(
                (float(kv["mbs_med"]), float(kv["x_med"])))
print("| cell | state | toksm MB/s A -> B | x vs tok v1 A -> B |")
print("|---|---|---|---|")
for key, c in cells.items():
    if not c["A"] or not c["B"]:
        continue
    ma, mb = (statistics.median(x[0] for x in c[s]) for s in ("A", "B"))
    xa, xb = (statistics.median(x[1] for x in c[s]) for s in ("A", "B"))
    print(f"| {key[0]} {key[1]} {key[2]} | {key[3]} | {ma:,.0f} -> {mb:,.0f} (x{mb / ma:.3f}) | {xa:.2f} -> {xb:.2f} |")
