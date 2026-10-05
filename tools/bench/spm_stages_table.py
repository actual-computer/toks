#!/usr/bin/env python3
"""tools/bench/spm_stages_table.py <log>...: the stage table of tools/bench/spm_stages.sh logs (shares of the shipping
time per cell: driver, scan, cache, model; ablation differences, not disjoint samples)."""
import re
import sys


def main():
    for path in sys.argv[1:]:
        cells, host, cur = {}, "", None
        for line in open(path, errors="replace"):
            if line.startswith("HOST"):
                host = line.split()[1]
            m = re.match(r"== (\S+) (\S+) chunk (\d+) LOAD before (\S+)", line)
            if m:
                cur = (m.group(2), int(m.group(3)))
                cells[cur] = {"load0": m.group(4)}
            m = re.match(r"LOAD after (\S+)", line)
            if m and cur:
                cells[cur]["load1"] = m.group(1)
            m = re.match(r"BUSY (.*)", line)
            if m and cur:
                cells[cur]["busy"] = m.group(1).strip()
            if line.startswith("SP "):
                kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
                cells[cur][(kv["variant"], kv["state"])] = kv
        print("## %s (%s)" % (host, path))
        print("| corpus | chunk | state | MB/s | driver | scan | cache | model | words/KB | model calls/KB | one-char | per-char steps/KB | load |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (corp, ch), c in cells.items():
            for st in ("cold", "pass", "warm"):
                try:
                    ship, prof, scan, drv = (c[(v, st)] for v in ("ship", "prof", "scan", "drv"))
                except KeyError:
                    continue
                T = float(ship["ns"])
                kb = float(ship["bytes"]) / 1024.0
                d = float(drv["ns"]) / T
                sc = (float(scan["ns"]) - float(drv["ns"])) / T
                mo = float(prof["model_ns"]) / float(prof["ns"])
                ca = 1.0 - d - sc - mo
                words = float(prof["words"]) + float(prof["one_char"])
                print("| %s | %s | %s | %.1f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f | %.1f | %.0f%% | %.0f | %s-%s %s |" % (
                    corp, ch if ch else "whole", st, float(ship["mbs"]), 100 * d, 100 * sc, 100 * ca, 100 * mo,
                    words / kb, float(prof["model_calls"]) / kb, 100 * float(prof["one_char"]) / max(words, 1),
                    float(prof["char_steps"]) / kb, c.get("load0", "?"), c.get("load1", "?"), c.get("busy", "")))


main()
