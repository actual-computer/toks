#!/usr/bin/env python3
"""tools/bench/spm_stages_ab.py <log>...: the table of a tools/bench/spm_stages_ab.sh log: per (corpus, chunk) and
state the best MB/s of each side and B / A; ids: "=" when every run of both sides printed the same ids= per state,
"DIFF" otherwise (the ids of a cell must not depend on the build)."""
import re
import sys

for path in sys.argv[1:]:
    cells, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"== (\S+) (\S+) chunk (\d+) LOAD before (\S+)", line)
        if m:
            cur = (m.group(2), m.group(3))
            cells[cur] = {"load": m.group(4), "ids": {}}
        m = re.match(r"LOAD after (\S+)", line)
        if m and cur:
            cells[cur]["load"] += "-" + m.group(1)
        if line.startswith("SP ") and cur:
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            key = (kv["variant"], kv["state"])
            cells[cur][key] = max(cells[cur].get(key, 0.0), float(kv["mbs"]))
            cells[cur]["ids"].setdefault(kv["state"], set()).add(kv.get("ids", "?"))
    print("| corpus | chunk | cold A -> B | pass A -> B | warm A -> B | ids | load |")
    print("|---|---|---|---|---|---|---|")
    for (corp, ch), c in cells.items():
        row = []
        for st in ("cold", "pass", "warm"):
            a, b = c.get(("A", st)), c.get(("B", st))
            row.append("%.1f -> %.1f (%.2fx)" % (a, b, b / a) if a and b else "-")
        same = all(len(v) == 1 and "?" not in v for v in c["ids"].values()) and len(c["ids"]) == 3
        print("| %s | %s | %s | %s | %s |" % (corp, ch if ch != "0" else "whole", " | ".join(row),
                                              "=" if same else "DIFF", c["load"]))
