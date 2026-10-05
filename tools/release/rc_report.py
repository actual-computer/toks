# /// script
# requires-python = ">=3.10"
# ///
"""tools/release/rc_report.py: tools/release/rc.sh's collected host outputs -> docs/release/<major.minor>.md.

    uv run tools/release/rc_report.py --sha <sha> --ref <ref> --stage build/release/rc/<sha12> \
        --receipts docs/release/rc/<sha12> --hosts "gb10a tr9970x m2ultra1" --win-hosts aimax395 --version 0.2.0 \
        > docs/release/0.2.md

Hosts are chipset keys (docs/machines.md): rc.sh maps each to its ssh alias and names the stage and receipt
directories after the key.

Reads, per host, <stage>/<host>/{host.txt, steps.txt, test-*.log, san.log, parity/*/report.json, wheels.log,
bench-*.log, rent-*.log} (tools/release/rc_host.sh writes them), and per windows host {host.txt, steps.txt,
win-build.log, win-test.log} (rc.sh's run_win), and tokv1.log (the incumbent cells). Every item of the goal gets met /
UNMET / not run with its
evidence; nothing is inferred from a step that did not run. What a release candidate cannot measure (a proof, fuzz
hours, an integration package, a decision) is declared with its evidence in docs/release/<major.minor>-declared.md, a
table of item | status | evidence rows that the report copies into the checklist. Every item whose status starts with
UNMET or "not run" is listed again in the UNMET table right under the title, with the count. The rent rule (asm >= the c twin) is decided per cell by
the paired abba medians of a rent step where one re-measured the cell, else by the two separate bench runs."""
import argparse
import datetime
import importlib.util
import json
import os
import re
import sys

# the critical targets (SPEC 1.1 (d)): the five + one pinned file per newer critical family (o200k = gpt-oss's file)
GOAL_TARGETS = ("glm53", "qwen38", "gemma4", "o200k", "kimik3", "nemotron3-4b", "llama4", "minimaxm2", "dsv4")
NAMES = {"o200k": "gpt-oss (o200k)", "kimik3": "Kimi K3", "glm53": "GLM 5.3", "qwen38": "Qwen 3.8", "gemma4": "Gemma 4",
         "gpt2": "gpt2", "llama3": "llama 3", "nemotron3-4b": "Nemotron 3 (4B file)", "llama4": "Llama 4",
         "minimaxm2": "MiniMax M2", "dsv4": "DeepSeek V4"}


def load_table(name="e2e_table"):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", name + ".py")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gate(stage, hosts, tiers):
    """the gigatoken gate's tally per host (rc_host.sh's gate step, tools/bench/gate_table.py), or None"""
    G = load_table("gate_table")
    logs = {f"{h}-{tiers.get(h) or '?'}": G.parse(f"{stage}/{h}/gate.log") for h in hosts
            if os.path.exists(f"{stage}/{h}/gate.log")}
    if not logs:
        return None
    c = G.compute(logs)
    out = []
    for host in logs:
        widest, nover, nn = c["band"].get(host, (0.0, 0, 0))
        st = []
        for s in G.STATES:
            rows = [v for k, v in c["res"].items() if k[0] == host and k[1] == "default" and k[5] == s]
            if rows:
                cert = [v for v in rows if v["cert"]]
                n = {x: sum(1 for v in cert if v["v"] == x) for x in ("WIN", "TIE", "LOSS")}
                shape = f" + {len(rows) - len(cert)} shape" if len(rows) > len(cert) else ""
                st.append(f"{G.SNAME[s]}{' (unmatched)' if s == 'warm' else ''} {n['WIN']} W / {n['TIE']} T / {n['LOSS']} L{shape}")
        miss = sum(1 for d in G.DECLARED if d[0] == host and d not in c["status"])
        out.append(f"{host}: null rows {nover} of {nn} over 2%, those cells shape (widest {widest:.2%}); " + ", ".join(st)
                   + (f"; {miss} declared cells MISSING" if miss else ""))
    return "; ".join(out)


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return None


def steps(stage, h):
    out = {}
    for line in (read(f"{stage}/{h}/steps.txt") or "").splitlines():
        m = re.match(r"STEP (\S+) status=(\d+) secs=(\d+) load=(\S+)", line)
        if m:
            out[m.group(1)] = {"status": int(m.group(2)), "secs": int(m.group(3)), "load": m.group(4).replace("->", "→")}
    return out


def tier_of(stage, h):
    """the tier toks_load binds on this host (from the bench log, else test_tier's output)"""
    for f in sorted(os.listdir(f"{stage}/{h}")) if os.path.isdir(f"{stage}/{h}") else []:
        if f.startswith("bench-auto"):
            m = re.search(r"^E2E tool=toks tier=(\w+)", read(f"{stage}/{h}/{f}") or "", re.M)
            if m:
                return m.group(1)
    t = read(f"{stage}/{h}/test-auto.log") or ""
    m = re.search(r"auto = (\w+)", t)
    return m.group(1) if m else "auto"


def suites(log):
    if log is None:
        return None
    n = len(re.findall(r"^== \S+/tests/\S+", log, re.M))
    fails = re.findall(r"^FAILED (\S+)", log, re.M)
    lines = [l for l in log.splitlines() if re.match(r"^test_\w+: .*failures", l)]
    checks = sum(int(x.replace(",", "")) for l in lines for x in re.findall(r"([\d,]+) checks", l))
    return {"suites": n, "failed": fails, "checks": checks}


def win_tests(log):
    """tools/win/test.cmd's log: programs run, counted checks, FAIL / SKIP lines, the final FAIL=n"""
    if log is None:
        return None
    lines = log.splitlines()
    checks = sum(int(x) for l in lines if re.match(r"^\w+: .*failures", l) for x in re.findall(r"(\d+) checks", l))
    return {"programs": sum(1 for l in lines if l.startswith("== ")), "checks": checks,
            "failed": [l[8:].split()[0] for l in lines if l.startswith("-- FAIL ")],
            "skips": [l.strip() for l in lines if "SKIP" in l], "final": next((l for l in lines if l.startswith("test.cmd: FAIL=")), None)}


def rent_pairs(stage, h):
    """(tk, corp, chunk) -> (asm tier, pass B/A, cold B/A) from a rent step's tools/bench/e2e_ab.sh lines (B = scalar)"""
    out = {}
    d = f"{stage}/{h}"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith("rent-") and f.endswith(".log"):
            for line in (read(f"{d}/{f}") or "").splitlines():
                if line.startswith("AB ") and "pass_B_over_A=" in line:
                    k = dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))
                    out[(k["tk"], k["corp"], int(k["chunk"]))] = (k["A"], float(k["pass_B_over_A"]), float(k["cold_B_over_A"]))
    return out


def fmt_status(st):
    if st is None:
        return "not run"
    return "pass" if st["status"] == 0 else f"FAIL (exit {st['status']})"


def incumbent(stage, hosts):
    """(cells x states, slower ones, void cells, hosts) of toksm vs tok v1 in the hosts' tokv1.log, or None"""
    T = load_table()
    n, slow, void, run = 0, [], [], []
    for h in hosts:
        p = f"{stage}/{h}/tokv1.log"
        if not os.path.exists(p):
            continue
        run.append(h)
        _, cells = T.parse_tokv1(p)
        for c in cells:
            t = c.get("tokv1")
            if not t or t.get("exact") != "yes":
                void.append(f"{h} {c['tk']} {c['corp']} {c['chunk']}")
                continue
            for (st, side), d in c["tv1"].items():
                if side != "toksm":
                    continue
                n += 1
                if float(d["x_med"]) <= 1.0:
                    slow.append(f"{h} {c['tk']} {c['corp']} {c['chunk']} {st} ({float(d['x_med']):.2f}x)")
    return (n, slow, void, run) if run else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--hosts", required=True)
    ap.add_argument("--win-hosts", default="")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--speed-note", default="")   # printed under the speed table's heading: e.g. the commit it was measured at
    a = ap.parse_args()
    V = a.version
    MM = V.rsplit(".", 1)[0]
    hosts = a.hosts.split()
    win = a.win_hosts.split()
    S = a.sha[:12]
    st = {h: steps(a.stage, h) for h in hosts + win}
    tiers = {h: tier_of(a.stage, h) for h in hosts}
    out = []
    w = out.append
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w(f"# toks {MM}: release candidate report\n")
    w(f"Generated by `tools/release/rc.sh {a.ref}` (`tools/release/rc_report.py`) on {now}: commit `{a.sha}`, hosts "
      + ", ".join([f"{h} ({tiers[h]})" for h in hosts] + [f"{h} (windows x86-64, tools/win)" for h in win])
      + ". Never edited by hand: rerun rc.sh on the release commit.")
    raw = f"`docs/bench/raw/*-{S}.log`" if not a.speed_note else "named in the speed table's section"
    w(f"Receipts: `{a.receipts}/<host>/` (steps with wall time and load, host fingerprint, log summaries, parity"
      f" reports); the speed table's raw logs: {raw}; everything the hosts wrote:"
      f" `{a.stage}/` on the machine that ran rc.sh.")
    w("host names are chipset keys (see docs/machines.md); home directories were replaced by $HOME in the receipts.\n")
    top = len(out)                                   # the UNMET table goes here, once every item is known

    # ---- host steps ---------------------------------------------------------------------------------
    w("## Steps per host\n")
    w("| host | step | status | wall | load 1m before → after | summary |")
    w("|---|---|---|---:|---|---|")
    for h in hosts + win:
        if not st[h]:
            w(f"| {h} | (all) | not run / no steps.txt | - | - | see `{a.stage}/{h}.out` |")
            continue
        for name, s in st[h].items():
            summ = ""
            log = read(f"{a.stage}/{h}/{name}.log")
            if name.startswith("test-") or name == "san":
                x = suites(log)
                if x:
                    summ = f"{x['suites']} test programs, {x['checks']:,} counted checks" + (
                        f"; FAILED: {', '.join(os.path.basename(f) for f in x['failed'])}" if x["failed"] else "")
                if name == "san" and log and "ThreadSanitizer" in log:
                    summ += "; ThreadSanitizer warnings present"
            elif name == "wheels" and log:
                whl = re.findall(r"^== build/wheels/(\S+)", log, re.M)
                passed = re.findall(r"(\d+) passed", log)
                summ = f"{len(whl)} wheels ({', '.join(sorted(set(re.sub(r'toks-[^-]+-', '', x).split('-')[0] for x in whl)))}); pytest passed: {' + '.join(passed)}"
                tags = sorted(set(x.rsplit("-", 1)[-1].replace(".whl", "") for x in whl))
                summ += f"; platform tags: {', '.join(tags)}"
            elif name.startswith("parity-") and log:
                parts = []
                for l in log.splitlines():
                    m = re.match(r"tier (\S+)(?: (ids|pieces))?: .*\"n_cases\":(\d+).*\"n_diffs\":(\d+)", l)
                    k = re.match(r"tier (\S+): (\{.*?\}) sources", l)
                    if m:
                        parts.append(f"{m.group(1)}{' ' + m.group(2) if m.group(2) else ''}: {int(m.group(3)):,} cases, {m.group(4)} diffs")
                    elif k:
                        d = json.loads(k.group(2))
                        parts.append(f"{k.group(1)}: {d.get('texts', 0):,} texts, {d.get('mismatches')} mismatches")
                summ = "; ".join(parts)
            elif name == "win-build" and log:
                k = re.search(r"^kernels:(.*)$", log, re.M)
                f = re.search(r"^build\.cmd: FAIL=(\d+)", log, re.M)
                summ = f"build.cmd FAIL={f.group(1) if f else '?'}; kernels:{k.group(1) if k else ' ?'}"
            elif name == "win-test" and log:
                x = win_tests(log)
                summ = (f"{x['programs']} test programs, {x['checks']:,} counted checks; {x['final'] or 'no final line'}"
                        + (f"; FAILED: {', '.join(x['failed'])}" if x["failed"] else "")
                        + (f"; SKIP lines: {'; '.join(x['skips'][:6])}" if x["skips"] else ""))
            elif name.startswith("rent-") and log:
                ab = re.findall(r"^AB tk=(\S+) corp=(\S+) chunk=(\d+) .*pass_B_over_A=(\S+) cold_B_over_A=(\S+)", log, re.M)
                summ = ("; ".join(f"{t} {c} {'whole' if ch == '0' else ch}: asm/c pass {p}x, cold {q}x" for t, c, ch, p, q in ab)
                        or "no cell slower than the c twin in the separate runs: nothing to re-measure")
            elif name == "tokv1" and log:
                n = len(re.findall(r"^TOKV1 ", log, re.M))
                ex = len(re.findall(r"^TOKV1 .* exact=yes", log, re.M))
                summ = f"{n} cells vs tok v1, {ex} with ids equal on every call (docs/bench/e2e.md \"Incumbent\")"
            elif name.startswith("bench-") and log:
                n = len(re.findall(r"^== ", log, re.M))
                ex = len(re.findall(r"^EXACT yes", log, re.M))
                summ = f"{n} cells, {ex} exact by the reference's ids" if ex else f"{n} cells (no comparators: exactness by ids sha against the exact cells)"
            w(f"| {h} | {name} | {fmt_status(s)} | {s['secs']} s | {s['load']} | {summ.replace('|', '/')} |")
    w("")

    # ---- parity ---------------------------------------------------------------------------------------
    w("## Parity samples (tests/parity, light: every 7th encode / pieces case of the quick set, every decode and stream case, gen_stream.py's adversarial streams)\n")
    w("| target | host | tier | cases | encode | pieces | decode | stream | diffs | receipt |")
    w("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    parity = {}                                          # (target, host, tier) -> ids (encode / decode / stream) ok
    pieces_ok = {}                                       # (target, host, tier) -> toks_pieces offsets ok
    for h in hosts:
        pd = f"{a.stage}/{h}/parity"
        if not os.path.isdir(pd):
            continue
        for d in sorted(os.listdir(pd)):
            rp = f"{pd}/{d}/report.json"
            if not os.path.isfile(rp):
                continue
            t, tier = d.rsplit("-", 1)
            part = "ids"
            if t.rsplit("-", 1)[-1] in ("ids", "pieces"):
                t, part = t.rsplit("-", 1)
            tier = tiers[h] if tier == "auto" else tier
            r = json.load(open(rp))
            if "totals" in r:                            # run_kimi.py shards
                tot = r["totals"]
                ok = tot.get("mismatches", 1) == 0 and not r.get("missing_shards")
                parity[(t, h, tier)] = ok
                w(f"| {NAMES.get(t, t)} | {h} | {tier} | {tot.get('texts', 0):,} texts x 3 modes + decode | - | - | - | - |"
                  f" {tot.get('mismatches')} mismatches, {tot.get('ref_errors')} reference errors | `{a.receipts}/{h}/parity-{d}.json` |")
            else:
                c = r.get("counts", {})
                n = lambda op: sum(v for k, v in c.items() if k.startswith(op))
                ok = r.get("n_diffs", 1) == 0 and not r.get("missing_shards") and r.get("stream_ref_errors", 0) == 0
                if part == "pieces":
                    pieces_ok[(t, h, tier)] = ok
                else:
                    parity[(t, h, tier)] = ok
                w(f"| {NAMES.get(t, t)} | {h} | {tier} | {r.get('n_cases', 0):,} | {n('encode'):,} | {n('pieces'):,} | {n('decode'):,} |"
                  f" {n('stream'):,} | {r.get('n_diffs')} | `{a.receipts}/{h}/parity-{d}.json` |")
    w("")

    # ---- the speed table ------------------------------------------------------------------------------
    T = load_table()
    runs = {}
    for h in hosts:
        for f in sorted(os.listdir(f"{a.stage}/{h}")) if os.path.isdir(f"{a.stage}/{h}") else []:
            if f.startswith("bench-") and f.endswith(".log"):
                head, cells = T.parse(f"{a.stage}/{h}/{f}")
                tier = next((c["toks"]["tier"] for c in cells if "toks" in c), None)
                if tier:
                    runs[(h, tier)] = (head, cells)
    exact_sha = {(c["tk"], c["corp"], c["chunk"]): c["toks"]["sha"] for (_, _), (_, cs) in runs.items() for c in cs
                 if "toks" in c and c.get("exact") == "yes"}
    comp = {}
    for (h, _), (_, cs) in runs.items():
        for c in cs:
            hf_s, tk_s, _, g_s, _ = T.comparators(c)
            d = comp.setdefault((h, c["tk"], c["corp"], c["chunk"]), {})
            if tk_s:
                d["tk"] = min(tk_s, d.get("tk", 1e30))
            if g_s:
                d["g"] = g_s if "g" not in d else {k: min(v, d["g"].get(k, v)) for k, v in g_s.items()}
    bench = {}
    for (h, tier), (head, cs) in runs.items():
        cells = [T.Cell(f"{h}-{tier}", h, c) for c in cs if "toks" in c]
        for x in cells:
            if x.exact != "yes":
                x.exact = "yes" if exact_sha.get((x.tk, x.corp, x.chunk)) == x.sha else "unchecked"
        tk = [(x, comp.get((h, x.tk, x.corp, x.chunk), {}).get("tk")) for x in cells if x.exact == "yes"]
        tk = [(x, s) for x, s in tk if s]
        gg = [(x, comp.get((h, x.tk, x.corp, x.chunk), {}).get("g")) for x in cells if x.exact == "yes"]
        gg = [(x, g) for x, g in gg if g]
        bench[(h, tier)] = {
            "cells": cells, "exact": sum(x.exact == "yes" for x in cells),
            "tk_win": [x for x, s in tk if x.s["cold"] < s], "tk_lose": [x for x, s in tk if x.s["cold"] >= s],
            "g_cold_lose": [x for x, g in gg if x.s["cold"] >= g["cold"]], "g_warm_lose": [x for x, g in gg if x.s["warm"] >= g["warm"]],
            "n_g": len(gg), "n_tk": len(tk)}
    w("## Speed table (docs/bench/e2e.md)\n")
    if a.speed_note:
        w(a.speed_note + "\n")
    if not bench:
        w("not run\n")
    else:
        w("| host | tier | cells | exact | faster than tiktoken (cold) | faster than gigatoken cold | faster than gigatoken warm |")
        w("|---|---|---:|---:|---|---|---|")
        for (h, tier), b in bench.items():
            lab = lambda xs: ", ".join(f"{x.tk} {x.corp} {T.chs(x.chunk)}" for x in xs[:12]) + (" ..." if len(xs) > 12 else "")
            w(f"| {h} | {tier} | {len(b['cells'])} | {b['exact']} | {b['n_tk'] - len(b['tk_lose'])} / {b['n_tk']}"
              + (f" (not: {lab(b['tk_lose'])})" if b["tk_lose"] else "")
              + f" | {b['n_g'] - len(b['g_cold_lose'])} / {b['n_g']} | {b['n_g'] - len(b['g_warm_lose'])} / {b['n_g']} |")
        w("")
    # asm vs the c twin per host
    rent, repaired = {}, {}
    for (h, tier), b in bench.items():
        if tier == "scalar" or (h, "scalar") not in bench:
            continue
        twin = {(x.tk, x.corp, x.chunk): x for x in bench[(h, "scalar")]["cells"]}
        pairs = rent_pairs(a.stage, h)
        slow, fixed = [], []
        n = 0
        for x in b["cells"]:
            y = twin.get((x.tk, x.corp, x.chunk))
            if y is None:
                continue
            n += 1
            if x.s["cold"] > y.s["cold"] or x.s["pass"] > y.s["pass"]:
                sep = f"separate runs cold {y.s['cold'] / x.s['cold']:.2f}x, pass {y.s['pass'] / x.s['pass']:.2f}x"
                pr = pairs.get((x.tk, x.corp, x.chunk))
                if pr is None:
                    slow.append(f"{x.tk} {x.corp} {T.chs(x.chunk)} ({sep}; not re-measured)")
                elif pr[1] >= 1.0 and pr[2] >= 1.0:
                    fixed.append(f"{x.tk} {x.corp} {T.chs(x.chunk)} ({sep}; paired abba {pr[0]} / scalar: pass {pr[1]:.3f}x, cold {pr[2]:.3f}x)")
                else:
                    slow.append(f"{x.tk} {x.corp} {T.chs(x.chunk)} ({sep}; paired abba {pr[0]} / scalar: pass {pr[1]:.3f}x, cold {pr[2]:.3f}x)")
        rent[(h, tier)] = (n, slow)
        repaired[(h, tier)] = fixed
    if rent:
        w("**asm vs the c twin** (same host, same cells, toks MB/s asm / scalar; a cell slower in the two separate runs is"
          " decided by its paired abba re-measure, tools/bench/e2e_ab.sh in the host's rent step):\n")
        for (h, tier), (n, slow) in rent.items():
            w(f"- {h} {tier}: {n - len(slow)} / {n} cells at least as fast in cold and pass" + (f"; slower: {', '.join(slow)}" if slow else "")
              + (f"; at least as fast once paired: {', '.join(repaired[(h, tier)])}" if repaired[(h, tier)] else ""))
        w("")

    # ---- the checklist ---------------------------------------------------------------------------------
    w(f"## The {MM} goal, item by item\n")
    w("| item | status | evidence |")
    w("|---|---|---|")
    items = []
    # 1. critical targets exact on every tier
    want_tiers = {}                                  # the tiers rc_host.sh was told to run (host.txt), auto resolved
    for h in hosts:
        m = re.search(r"^tiers (.*?) jobs ", read(f"{a.stage}/{h}/host.txt") or "", re.M)
        want_tiers[h] = [tiers[h] if t == "auto" else t for t in (m.group(1).split() if m else ["auto"])]
    rows, missing, bad = [], [], []
    for t in GOAL_TARGETS:
        for h in hosts:
            for tier in want_tiers[h]:
                k = (t, h, tier)
                if k not in parity:
                    missing.append(f"{t} {h} {tier}")
                elif not parity[k]:
                    bad.append(f"{t} {h} {tier}")
    n_ok = sum(1 for k, v in parity.items() if v and k[0] in GOAL_TARGETS)
    status = "met" if not missing and not bad and n_ok else "UNMET"
    items.append(("critical targets exact on every tier (glm 5.3, qwen 3.8, gemma 4, gpt-oss, kimi k3; nemotron 3, llama 4, minimax m2, deepseek v4)", status,
                  f"{n_ok} target x host x tier samples clean" + (f"; diffs: {', '.join(bad)}" if bad else "")
                  + (f"; not run: {', '.join(missing)}" if missing else "")))
    pb = sorted(f"{t} {h} {tier}" for (t, h, tier), v in pieces_ok.items() if not v)
    items.append(("toks_pieces offsets equal the reference's (every target, tier)",
                  "met" if pieces_ok and not pb else ("UNMET" if pieces_ok else "not run"),
                  f"{sum(pieces_ok.values())} / {len(pieces_ok)} target x host x tier samples clean" + (f"; diffs: {', '.join(pb)}" if pb else "")))
    # 2. c api
    api_ok = all(st[h].get("test-auto", {}).get("status") == 0 for h in hosts if st[h])
    ran = [h for h in hosts if "test-auto" in st[h]]
    items.append(("c api complete: encode / pieces / decode / stream / par (make test: test_api, test_e2e, test_stream, test_par, ...)",
                  "met" if api_ok and len(ran) == len(hosts) else ("UNMET" if ran else "not run"),
                  "make test " + ", ".join(f"{h} {fmt_status(st[h].get('test-auto'))}" for h in hosts)
                  + "; forced tiers: " + ", ".join(f"{h} {k[5:]} {fmt_status(v)}" for h in hosts for k, v in st[h].items()
                                                  if k.startswith("test-") and k != "test-auto")))
    # 3. python wheels
    wl = {h: read(f"{a.stage}/{h}/wheels.log") for h in hosts}
    linux_plain = [h for h, l in wl.items() if l and re.search(r"^== build/wheels/\S+-linux_(x86_64|aarch64)\.whl", l, re.M)]
    wok = all(st[h].get("wheels", {}).get("status") == 0 for h in hosts if "wheels" in st[h]) and any("wheels" in st[h] for h in hosts)
    items.append(("python wheels (cp310-cp314, tests incl. hf parity on 3.13)",
                  ("met" if wok else "UNMET") + (" (linux wheels are plain linux_*, not manylinux: pip on other distros refuses them)" if linux_plain else ""),
                  ", ".join(f"{h} {fmt_status(st[h].get('wheels'))}" for h in hosts)))
    # 4. asm no cell slower than the c twin
    isas = {"neon": "arm64", "avx2": "x86-64", "avx512": "x86-64"}
    covered = sorted(set(isas.get(t, t) for (h, t) in rent))
    slow_all = [f"{h} {t}: {s}" for (h, t), (n, sl) in rent.items() for s in sl]
    st4 = "met" if rent and not slow_all and set(covered) >= {"arm64", "x86-64"} else ("UNMET" if rent else "not run")
    items.append(("asm on both isas, no cell slower than the c twin", st4,
                  ("isas compared: " + ", ".join(covered) if covered else "no asm/scalar pair") +
                  (f"; slower cells: {'; '.join(slow_all[:10])}" if slow_all else "")))
    # 5. speed table + tiktoken
    lose = [f"{h} {t}: {x.tk} {x.corp} {T.chs(x.chunk)}" for (h, t), b in bench.items() for x in b["tk_lose"]]
    voids = [f"{h} {t}: {x.tk} {x.corp} {T.chs(x.chunk)}" for (h, t), b in bench.items() for x in b["cells"] if x.exact != "yes"]
    items.append(("published speed table, faster than tiktoken in every cell (docs/bench/e2e.md)",
                  "met" if bench and not lose and not voids else ("UNMET" if bench else "not run"),
                  f"{sum(len(b['cells']) for b in bench.values())} cells on {len(bench)} host-tier runs"
                  + (f"; slower than tiktoken: {', '.join(lose[:10])}" if lose else "")
                  + (f"; VOID / unchecked: {', '.join(voids[:10])}" if voids else "")))
    gt = gate(a.stage, hosts, tiers)
    if gt:                                             # the gate (docs/bench/gigatoken-gate.md): paired, intervals
        items.append((f"gigatoken (the bar to beat; not a {MM} gate): the gate, docs/bench/gigatoken-gate.md", "informational", gt))
    else:                                              # no gate step: the speed table's best-of tallies
        gl = sum(len(b["g_cold_lose"]) for b in bench.values())
        gw = sum(len(b["g_warm_lose"]) for b in bench.values())
        gn = sum(b["n_g"] for b in bench.values())
        items.append((f"gigatoken (the bar to beat; not a {MM} gate)", "informational",
                      f"best of reps, no intervals (no gate step ran): toks faster in {gn - gl} / {gn} cells cold vs cold, "
                      f"{gn - gw} / {gn} warm vs warm"))
    # 6. sanitizers
    san = [(h, st[h]["san"]) for h in hosts if "san" in st[h]]
    items.append(("sanitizers clean (every test program under ASan + UBSan, test_par under TSan)",
                  "met" if san and all(s["status"] == 0 for _, s in san) else ("UNMET" if san else "not run"),
                  ", ".join(f"{h} {fmt_status(s)}" for h, s in san) or "no host ran SAN=1"))
    # 7. windows (docs/release.md step 1)
    wrun = [(h, st[h].get("win-build"), st[h].get("win-test"), win_tests(read(f"{a.stage}/{h}/win-test.log"))) for h in win]
    wok = bool(wrun) and all(b and t and b["status"] == 0 and t["status"] == 0 and x and not x["failed"] for h, b, t, x in wrun)
    items.append(("windows x86-64: tools/win/build.cmd + tools/win/test.cmd green (docs/release.md step 1)",
                  "met" if wok else ("UNMET" if wrun else "not run"),
                  "; ".join(f"{h}: build {fmt_status(b)}, test {fmt_status(t)}" + (f", {x['programs']} programs, {x['checks']:,} counted checks"
                            + (f", FAILED: {', '.join(x['failed'])}" if x["failed"] else "") if x else "") for h, b, t, x in wrun)
                  or "no windows host"))
    # 8. traffic (maintainer doctrine: bytes moved per input byte next to MB/s)
    items.append((f"bytes moved per input byte next to MB/s (the maintainer doctrine's speed rule; not a {MM} gate)", "informational: not measured",
                  "rc.sh does not run tools/bench/e2e.sh COUNT=1: that pass counts the loads / stores of the library's C only (the"
                  " shipping asm kernels are not instrumented) and docs/bench/e2e.md has no column for it"))
    # 9. the incumbent: tok v1, with its integration flags (toksm = TOKS_SCRATCH_MEMO_MIB(4))
    inc = incumbent(a.stage, hosts)
    if inc is None:
        items.append(("incumbent: faster than tok v1 in every tools/bench/tokv1.sh cell, with the incumbent's flags", "not run",
                      "no host ran the tokv1 step (rc.sh needs E_SRC with the e commit; linux hosts only)"))
    else:
        n, slow, void, hosts_run = inc
        items.append(("incumbent: faster than tok v1 in every tools/bench/tokv1.sh cell, with the incumbent's flags"
                      " (toksm = TOKS_SCRATCH_MEMO_MIB(4): new prompts and replays)",
                      "met" if n and not slow and not void else "UNMET",
                      f"{n - len(slow)} / {n} cell x state medians faster on {', '.join(hosts_run)} (docs/bench/e2e.md \"Incumbent\")"
                      + (f"; slower: {', '.join(slow[:10])}" if slow else "") + (f"; VOID: {', '.join(void)}" if void else "")))
    # 10. the gigatoken gate (abba pairs, intervals): met only when this stage ran the gate step (a gate.log per host,
    # this commit's: the tally above is its evidence). A gate doc in the tree that measured another library never makes
    # the row met; the evidence names the library it did measure, read from the doc's own Tally line.
    gd = "docs/bench/gigatoken-gate.md"
    doc = read(gd) or ""
    lib = re.search(r"library master ([0-9a-f]{7,40})", doc)
    memo = re.search(r"UNMATCHED: toks \d+ MiB piece cache \+ (\d+) MiB memo", doc)
    items.append(("the gigatoken gate on the new default: abba pairs, cluster-bootstrap intervals, WIN / LOSS / TIE per cell",
                  "met" if gt else "UNMET",
                  "the gate step's logs at this commit, the gigatoken item's tally (tools/bench/gate_table.py)" if gt else
                  f"{gd} is the picture at {lib.group(1)}{' (before the memo default)' if memo and memo.group(1) == '0' else ''}; "
                  "this release's picture is not measured" if lib else
                  f"no {gd} at this commit and no gate step ran; docs/bench/e2e.md's Gates compare medians without intervals"))
    # 11. what an rc cannot measure: declared with its evidence (docs/release/<major.minor>-declared.md). A row
    # still in progress is refused: the report would ship a state nobody measured.
    dp = f"docs/release/{MM}-declared.md"
    for line in (read(dp) or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) == 3 and cells[0] != "item" and not set(cells[0]) <= set("-: "):
            if "in progress" in (cells[1] + " " + cells[2]).lower():
                sys.exit(f"rc_report: {dp}: '{cells[0]}' is still in progress; refresh the row before the report")
            items.append((cells[0], cells[1], cells[2] + f" ({dp})"))
    # 12. tag
    items.append((f"tag v{V}", "UNMET", "the maintainer tags the release commit after this report is all met (docs/release.md)"))
    for it, s, ev in items:
        w(f"| {it} | {s} | {ev.replace('|', '/')} |")
    w("")
    unmet = [(it, s, ev) for it, s, ev in items if s.startswith(("UNMET", "not run")) or "not manylinux" in s]
    block = [f"## UNMET: {len(unmet) or 'none'}\n",
             "Every item of the goal that is not met, measured here or declared with its evidence: the full checklist"
             f" is [below](#the-{MM.replace('.', '')}-goal-item-by-item).\n",
             "| item | status | evidence |", "|---|---|---|"]
    block += [f"| {it} | {s} | {ev.replace('|', '/')} |" for it, s, ev in unmet] + [""]
    out[top:top] = block
    w("## Hosts\n")
    for h in hosts + win:
        w(f"**{h}**\n")
        w("```")
        w((read(f"{a.stage}/{h}/host.txt") or "no host.txt").strip())
        w("```\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
