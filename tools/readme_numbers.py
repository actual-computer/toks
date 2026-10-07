#!/usr/bin/env python3
"""tools/readme_numbers.py: every number README.md quotes, recomputed from the generated speed tables, with its source.

    python3 tools/readme_numbers.py [docs/bench/e2e.md [docs/bench/par.md [docs/release/<major.minor>.md]]]

Reads docs/bench/e2e.md (tools/bench/e2e_table.py writes it from a release candidate's raw logs: the two headline
tables, Gates, Incumbent, the receipts) and docs/bench/par.md (tests/par/par_table.py), and prints one line per README
number: its key, its value and the cell or gate line it comes from. Then it prints the README's mechanical parts: the
tally table and every cell toks does not win. The charts come from the same files (tools/readme_charts.py). The README
quotes no number this does not print. A PR that updates the README lists every changed number before -> after, from
this script run on the old tables and on the new ones.

The rules, so that a number cannot drift between runs:
  - asm tiers: the labels without -scalar. A cell is (tokenizer, corpus, chunk); chunk 4096 or whole.
  - "A-Bx faster than hf": cold toks / hf over every asm-tier cell, both chunks; A and B are rounded down.
  - "A-Bx faster than tiktoken": the same rule over the asm-tier cells where tiktoken counts (its ids equal hf's
    without the post-processor).
  - "won of total cold cells against gigatoken": the Gates' cold vs cold counts summed over the asm tiers.
  - one ratio (charts, captions): one decimal below 10, else a whole number (8.0x, 85x, 154x).
  - a lost cell: one the Gates line lists after "NOT in:" (toks MB/s vs the other tool's, the medians the tables
    show; no interval) or the Incumbent line lists after "SLOWER in:".
  - the status note's count: the release report's UNMET table (tools/release/rc_report.py), less the tag row (the tag
    exists by the time the README is read); the report defaults to include/toks.h's TOKS_VERSION's."""
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E2E = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/bench/e2e.md"
PAR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/bench/par.md"
_V = re.search(r'#define TOKS_VERSION\s+"(\d+\.\d+)', (ROOT / "include/toks.h").read_text())
REPORT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / f"docs/release/{_V.group(1) if _V else '?'}.md"
CHUNK = {"Headline: 4 KiB chunks": "4096", "Headline: whole corpus in one call": "whole"}


def num(cell):
    m = re.match(r"\s*([0-9][0-9,]*\.?[0-9]*)", cell or "")
    return float(m.group(1).replace(",", "")) if m else None


def fx(v):
    return f"{v:.1f}x" if v < 10 else f"{v:.0f}x"


def sections(text):
    return {s.split("\n", 1)[0].strip(): s for s in re.split(r"\n## ", text)[1:]}


def headline(body, chunk, out):
    label, cols = None, None
    for line in body.splitlines():
        m = re.match(r"\*\*([\w.-]+)\*\*", line)
        if m:
            label, cols = m.group(1), None
            continue
        if not line.startswith("|") or label is None or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "tokenizer":
            cols = cells
            continue
        if cols is None:
            continue
        row = {k: num(v) for k, v in zip(cols, cells)}
        out.setdefault(label, {})[(cells[0], cells[1], chunk)] = row


def gates(body):
    """{label: {"cells": n, "exact": n, "vs": {what: (won, of, [(tok, corpus, chunk, toks, other)])}}}, asm twins"""
    out, label, twins = {}, None, []
    for line in body.splitlines():
        m = re.match(r"\*\*([\w.-]+)\*\*: (\d+) cells, (\d+) exact", line)
        if m:
            label = m.group(1)
            out[label] = {"cells": int(m.group(2)), "exact": int(m.group(3)), "vs": {}}
            continue
        m = re.match(r"- faster than (\S+(?: \S+)?) \(.*?\): (\d+) / (\d+) cells(?:; NOT in: (.*))?$", line)
        if m and label:
            lost = [(a, b, c, float(x) if x else None, float(y) if y else None) for a, b, c, x, y in   # tiktoken's: names only
                    re.findall(r"([\w.-]+) (\w+) (4096|whole)(?: \(([0-9.]+) vs ([0-9.]+)\))?", m.group(4) or "")]
            out[label]["vs"][m.group(1)] = (int(m.group(2)), int(m.group(3)), lost)
            continue
        m = re.match(r"\*\*asm vs c twin, ([\w.-]+) / ([\w.-]+)\*\* .*?: (\d+) / (\d+) cells .*median cold ([0-9.]+)x", line)
        if m:
            twins.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), float(m.group(5))))
    return out, twins


def incumbent(body):
    rows = []
    for line in body.splitlines():
        m = re.match(r"\*\*Incumbent gate\*\* \((.*?)\): toks faster in (\d+) / (\d+) \(median x > 1\), (\d+) with "
                     r"the interval's lower bound > 1(?:; SLOWER in: (.*?))?\.?$", line)
        if m:
            slow = re.findall(r"(tokv1-[\w.-]+ [^,(]+?) \(([0-9.]+)x\)", m.group(5) or "")
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), slow))
    return rows


DESC = {"gb10": "NVIDIA GB10, Cortex-X925", "x86": "AMD Threadripper 9970X, Zen 5", "mac": "Apple M2 Ultra"}
HEAD = {"tiktoken": "faster than tiktoken, cold", "gigatoken cold": "faster than gigatoken, cold vs cold",
        "gigatoken pass": "faster than gigatoken, pass vs pass", "gigatoken warm": "faster than gigatoken, warm vs warm",
        "gigatoken pass-same": "faster than gigatoken, pass vs pass (after its own cold pass)"}


def role(lb):
    """a label's machine role, so the keys pair across releases whose host keys differ (gb10a -> gb10c)"""
    return "gb10" if lb.startswith("gb10") else "x86" if lb.endswith("-avx2") else "mac" if lb.startswith("m2") else lb


def report_unmet(text):
    """[(item, status)] from the report's UNMET table (or the old '## Unmet' list), the tag row left out"""
    m = re.search(r"\n## UNMET: [^\n]*\n(.*?)(?=\n## )", text, re.S)
    if m:
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in m.group(1).splitlines() if ln.startswith("|")]
        rows = [(r[0], r[1]) for r in rows if len(r) >= 2 and r[0] != "item" and not set(r[0]) <= set("-: ")]
    else:
        m = re.search(r"\n## Unmet\n(.*?)(?=\n## )", text, re.S)
        rows = [(ln[2:].strip(), "") for ln in (m.group(1) if m else "").splitlines() if ln.startswith("- ") and ln[2:] != "none"]
    return [r for r in rows if not r[0].startswith("tag v")]


def par_after(text):
    out = {}
    for sec in re.split(r"\n## ", text)[1:]:
        m = re.match(r"(\w+), after \([^ ]+ ([0-9a-f]+)\)", sec)
        if not m:
            continue
        rows = {}
        for line in sec.splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 11 and cells[0] == "16 MiB" and cells[1] != "serial":
                rows[int(cells[1].split()[0])] = (num(cells[2]), num(cells[6]))    # toks pass MB/s, x serial
        if rows:
            out[m.group(1)] = (m.group(2), rows)
    return out


def main():
    text = E2E.read_text()
    sec = sections(text)
    tab = {}
    for title, chunk in CHUNK.items():
        headline(sec[title], chunk, tab)
    gate, twins = gates(sec["Gates"])
    for lb, g in gate.items():                       # a loss the Gates name without MB/s: the headline table's
        if "tiktoken" in g["vs"]:
            won, of, lost = g["vs"]["tiktoken"]
            g["vs"]["tiktoken"] = (won, of, [(t, c, ch, x or tab[lb][(t, c, ch)]["toks cold"], y or tab[lb][(t, c, ch)]["tiktoken"])
                                             for t, c, ch, x, y in lost])
    inc = incumbent(next((v for k, v in sec.items() if k.startswith("Incumbent")), ""))
    commits = {m.group(1): m.group(2)[:7] for m in
               re.finditer(r"### ([\w.-]+): `[^`]+`\n\n(?:- [^\n]*\n)*?- git: `([0-9a-f]+)`", text)}
    asm = [lb for lb in tab if not lb.endswith("-scalar")]
    # a section that parses to nothing is a format change, never a number to leave out quietly
    gone = [f"Gates {lb}" for lb in tab if not gate.get(lb, {}).get("vs")] + ([] if twins else ["Gates asm vs c twin"]) + \
           (["Incumbent gate lines"] if any(k.startswith("Incumbent") for k in sec) and not inc else [])
    for g in gone:
        print(f"!! {g}: nothing parsed (the generator's format changed?): fix tools/readme_numbers.py before using this sheet")
    gb10 = next(lb for lb in asm if lb.startswith("gb10"))
    x86 = next(lb for lb in asm if lb.endswith("-avx2"))
    say = lambda key, val, src: print(f"{key:<28} {val}  <-  {src}")  # noqa: E731

    print(f"# {E2E.relative_to(ROOT) if E2E.is_relative_to(ROOT) else E2E}: asm tiers {', '.join(asm)}; "
          f"commits {', '.join(sorted(set(commits.get(lb, '?') for lb in tab)))}")
    cold = [(r["toks cold"] / r["hf"], lb, k) for lb in asm for k, r in tab[lb].items() if r.get("hf") and r.get("toks cold")]
    lo, hi = min(cold), max(cold)
    say("hf.range", f"{math.floor(lo[0])}-{math.floor(hi[0])}x",
        f"cold toks/hf, {len(cold)} asm cells: min {lo[0]:.2f} {lo[1]} {' '.join(lo[2])}, max {hi[0]:.2f} {hi[1]} {' '.join(hi[2])}")
    tkr = [(r["toks cold"] / r["tiktoken"], lb, k) for lb in asm for k, r in tab[lb].items() if r.get("tiktoken") and r.get("toks cold")]
    lo, hi = min(tkr), max(tkr)
    say("tiktoken.range", f"{math.floor(lo[0])}-{math.floor(hi[0])}x",
        f"cold toks/tiktoken, {len(tkr)} asm cells where tiktoken counts: min {lo[0]:.2f} {lo[1]} {' '.join(lo[2])}, "
        f"max {hi[0]:.2f} {hi[1]} {' '.join(hi[2])}")
    for lb in asm:
        g = gate[lb]
        say(f"cells.{role(lb)}", f"{g['cells']} cells, {g['exact']} exact", f"Gates **{lb}**, commit {commits.get(lb, '?')}")
    r = tab[gb10][("llama3", "en", "4096")]
    say("caption.gb10.llama3en", f"toks {r['toks cold']}, gigatoken {r['gigatoken cold']}, tiktoken {r['tiktoken']}, "
        f"hf {r['hf']:.2f}", f"{gb10} llama3 en 4096 cold")
    toks4 = sorted({k[0] for k in tab[gb10] if k[2] == "4096"})
    say("chart.tokenizers", len(toks4), f"{gb10} 4 KiB rows")
    na = [f"{k[0]} {k[1]}" for k in sorted(tab[gb10]) if k[2] == "4096" and tab[gb10][k].get("tiktoken") is None]
    say("chart.tiktoken_na", len(na), ", ".join(na))
    for lb, models in ((gb10, ("glm53", "kimik3")), (x86, ("glm53", "kimik3"))):
        for tk in models:
            for c in ("en", "code"):
                r = tab[lb][(tk, c, "4096")]
                say(f"models.{role(lb)}.{tk}.{c}", f"toks {r['toks cold']}, giga {r['gigatoken cold']}, "
                    f"tiktoken {r['tiktoken']}, hf {r['hf']:.2f}", f"{lb} {tk} {c} 4096 cold")
    mm = [(tab[gb10][(tk, c, "4096")]["toks cold"] / tab[gb10][(tk, c, "4096")]["hf"], tk, c)
          for tk in ("glm53", "kimik3") for c in ("en", "code", "ml", "cjk") if (tk, c, "4096") in tab[gb10]]
    say("models.hf.range", f"{fx(min(mm)[0])} to {fx(max(mm)[0])}", f"{gb10}: min {min(mm)[1:]} max {max(mm)[1:]}")
    v = [(tab[lb][k]["toks cold"] / tab[lb][k]["hf"], lb, k[0], k[1]) for lb in (gb10, x86) for k in tab[lb]
         if k[2] == "4096" and tab[lb][k].get("hf") and tab[lb][k].get("toks cold")]
    say("vs_hf.cold", f"{fx(min(v)[0])} to {fx(max(v)[0])}", f"min {min(v)[1:]} max {max(v)[1:]}")
    # warm (the same text a second time) is not a ratio to hf: a replay the default scratch's 4 MiB memo holds is
    # answered from it. The README gives toks's warm MB/s on the two sides of the measured split instead.
    for key, corp in (("warm.en_code", ("en", "code")), ("warm.ml_cjk", ("ml", "cjk"))):
        w = [(tab[lb][k]["toks warm"], lb, k[0], k[1]) for lb in asm for k in tab[lb]
             if k[2] == "4096" and k[1] in corp and tab[lb][k].get("toks warm")]
        say(key, f"{min(w)[0]:,.0f}-{max(w)[0]:,.0f} MB/s, {len(w)} cells",
            f"toks warm, 4 KiB, {' + '.join(corp)}, {', '.join(asm)}: min {min(w)[1:]} max {max(w)[1:]}")

    print("\n# tally (README Gates table)")
    measured = [w for w in gate[asm[0]]["vs"] if all(w in gate[lb]["vs"] for lb in asm)]   # the table's order
    states = [w for w in measured if w in HEAD]      # the README's tally: tiktoken, cold, pass, warm
    others = sorted({w for lb in asm for w in gate[lb]["vs"]} - set(states))
    if others:                                       # e.g. coldo / lang-x / warmo, or a state some machines lack
        print("# measured but not in the tally (every lost cell of theirs is listed below): " + ", ".join(
            f"{w} (" + ", ".join(f"{lb} {gate[lb]['vs'][w][0]} / {gate[lb]['vs'][w][1]}" for lb in asm if w in gate[lb]["vs"]) + ")"
            for w in others))
    print("| machine · tier | " + " | ".join(states) + " |")
    for lb in asm:
        print(f"| {lb} | " + " | ".join(f"{gate[lb]['vs'][w][0]} / {gate[lb]['vs'][w][1]}" for w in states) + " |")
    if "gigatoken warm" in states:
        warm = [gate[lb]["vs"]["gigatoken warm"] for lb in asm]
        say("warm.wins", f"{min(w[0] for w in warm)}-{max(w[0] for w in warm)} of {warm[0][1]}", "Gates warm vs warm")
    if "gigatoken cold" in states:
        gc = [gate[lb]["vs"]["gigatoken cold"] for lb in asm]
        say("gigatoken.cold.total", f"{sum(g[0] for g in gc)} of {sum(g[1] for g in gc)}",
            "Gates cold vs cold, summed: " + " + ".join(f"{lb} {g[0]} / {g[1]}" for lb, g in zip(asm, gc)))
    for a, b, won, of, med in twins:
        say(f"asm_vs_c.{role(a)}", f"{won} / {of}, median cold {med:.2f}x", f"Gates asm vs c twin {a} / {b}")
    for desc, won, of, lb_n, slow in inc:
        say("tokv1", f"{won} / {of} ({lb_n} lower bound > 1)", desc[:90])

    print("\n# README block: the tally")
    print("| machine · tier | " + " | ".join(HEAD.get(w, f"faster than {w}") for w in states) + " |")
    print("|---|" + "---:|" * len(states))
    for lb in asm:
        key, tier = lb.rsplit("-", 1)
        print(f"| {key} · {tier} ({DESC.get(role(lb), lb)}) | " +
              " | ".join(f"{gate[lb]['vs'][w][0]} / {gate[lb]['vs'][w][1]}" for w in states) + " |")
    print("\n# README block: where toks is not ahead yet (every lost cell, toks / the other tool, worst first)")
    print("| machine | state | cells toks does not win |")
    print("|---|---|---|")
    for lb in asm:
        for w in gate[lb]["vs"]:
            won, of, lost = gate[lb]["vs"][w]
            if lost:
                cells = sorted(lost, key=lambda c: c[3] / c[4])
                print(f"| {lb.rsplit('-', 1)[0]} | {HEAD.get(w, 'vs ' + w).replace('faster than ', 'vs ')} ({of - won} of {of}) | " +
                      ", ".join(f"{t} {c} {'4 KiB' if ch == '4096' else ch} {x / y:.2f}x" for t, c, ch, x, y in cells) + " |")
    for desc, won, of, lb_n, slow in inc:
        for host in sorted({s_.split()[0] for s_, _ in slow}):
            cells = [(" ".join(s_.split()[1:]).replace(" 4096 ", " 4 KiB "), x) for s_, x in slow if s_.split()[0] == host]
            print(f"| {host.replace('tokv1-', '')} | vs tok v1 ({desc.split(',')[0]}; {of - won} of {of} on both machines) | "
                  + ", ".join(f"{c} {x}x" for c, x in cells) + " |")

    print("\n# every cell toks does not win (Gates NOT in: / Incumbent SLOWER in:), worst first")
    for lb in asm:
        for w in gate[lb]["vs"]:
            lost = gate[lb]["vs"][w][2]
            if lost:
                cells = sorted(lost, key=lambda c: c[3] / c[4])
                print(f"- {lb} vs {w}: {len(lost)}: " +
                      ", ".join(f"{t} {c} {ch} {x / y:.2f}x ({x:g} vs {y:g})" for t, c, ch, x, y in cells))
    for desc, won, of, lb_n, slow in inc:
        if slow:
            print(f"- tok v1 ({desc[:60]}): " + ", ".join(f"{s} {x}x" for s, x in slow))

    print("\n# status note (README: the release report's unmet items, the tag left out)")
    if REPORT.exists():
        un = report_unmet(REPORT.read_text())
        say("status.unmet", len(un), f"{REPORT.name}: " + "; ".join(f"{i} [{st}]" if st else i for i, st in un))
    else:
        say("status.unmet", "?", f"{REPORT.name} not generated yet (tools/release/rc.sh writes it)")

    if PAR.exists():
        print("\n# par (README par-scaling caption)")
        for host, (commit, rows) in sorted(par_after(PAR.read_text()).items()):
            k = max(rows)
            say(f"par.{host}", f"{rows[k][0]:,.0f} MB/s at {k} threads, {rows[k][1]:.2f}x serial", f"par.md {host} after {commit}")
        for lb in (x86, gb10):
            r = tab[lb][("llama3", "en", "4096")]
            say(f"par.flat.{role(lb)}", f"tiktoken {r['tiktoken']}, hf {r['hf']:.2f}", f"{lb} llama3 en 4096, {commits.get(lb, '?')}")


if __name__ == "__main__":
    main()
