#!/usr/bin/env python3
"""tools/bench/emb_stages_table.py <log>...: the stage table of tools/bench/emb_stages.sh logs (shares of the shipping
time per cell; ablation differences, not disjoint samples; prof's timer share of the model as a cross-check).
  uni: driver = drv, walk (normalizer + pre-tokenizers) = walk - drv, model (Viterbi) = ship - walk
  wp:  driver = drv, scan (BertNormalizer + BertPreTokenizer) = scan - drv, model = ship - scan"""
import re
import sys


def main():
    for path in sys.argv[1:]:
        cells, host, algo, tok, cur = {}, "", "uni", "", None
        for line in open(path, errors="replace"):
            if line.startswith("HOST"):
                host = line.split()[1]
                m = re.search(r"ALGO (\S+) TOK (\S+)", line)
                if m:
                    algo, tok = m.group(1), m.group(2)
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
            if line.startswith("SP ") and cur:
                kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
                cells[cur][(kv["variant"], kv["state"])] = kv
        mid = "walk" if algo == "uni" else "scan"
        first = "walk" if algo == "uni" else "scan"
        print("## %s %s (%s)" % (host, tok, path))
        print("| corpus | chunk | state | MB/s | driver | %s | model | model (timer) | pieces/KB | bytes/piece | ns/piece (model) | load |" % first)
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (corp, ch), c in cells.items():
            for st in ("cold", "pass", "warm"):
                try:
                    ship, prof, mv, drv = (c[(v, st)] for v in ("ship", "prof", mid, "drv"))
                except KeyError:
                    continue
                T = float(ship["ns"])
                kb = float(ship["bytes"]) / 1024.0
                d = float(drv["ns"]) / T
                w = (float(mv["ns"]) - float(drv["ns"])) / T
                mo = 1.0 - d - w
                pieces = float(prof["words"])
                tm = float(prof["model_ns"]) / float(prof["ns"])
                npc = float(prof["model_ns"]) / max(pieces, 1)
                bpp = float(prof["word_bytes"]) / max(pieces, 1) if algo == "uni" else float(ship["bytes"]) / max(pieces, 1)
                print("| %s | %s | %s | %.1f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.0f | %.1f | %.0f | %s-%s %s |" % (
                    corp, ch if ch else "whole", st, float(ship["mbs"]), 100 * d, 100 * w, 100 * mo, 100 * tm,
                    pieces / kb, bpp, npc, c.get("load0", "?"), c.get("load1", "?"), c.get("busy", "")))


main()
