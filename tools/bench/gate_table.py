#!/usr/bin/env python3
"""tools/bench/gate_table.py: docs/bench/gigatoken-gate.md from tools/bench/gate.sh logs (generated, never edited).

    python3 tools/bench/gate_table.py gb10a-neon=<log> tr9970x-avx2=<log> m2ultra1-neon=<log> > docs/bench/gigatoken-gate.md
    ... [--before gb10a-neon=<log> --before tr9970x-avx2=<log> ...]: adds the before -> after section (a train's receipt)

The statistic (decided 2026-10-05): per host, cell (tokenizer x corpus x chunk), cache config and state, the paired ratios t_gigatoken / t_toks
(> 1 = toks faster) of every block that is not VOID, grouped by run-pair ((A, B) and (B, A) of each A B B A block: a
run is the independent unit, its reps move together); their median and a 95% cluster-bootstrap interval (2000
resamples of the run-pairs, each concatenating its reps; tokv1.c's xorshift and seed, lo = m[49], hi = m[1950]: with
one rep per run-pair it is tokv1.c's boot() line for line). Every cell's own null (A N N A: toks against the same toks,
the same statistic) decides it: WIN iff the gate's interval lies wholly above the null's interval widened to at least
[0.99, 1.01], LOSS iff wholly below, else TIE; a cell whose null |median - 1| exceeds 2% is UNCERTIFIED SHAPE (as is
every cell of an unpinned log). The host-wide widest null is printed as context, never as a band.
Exit status 1 when a declared cell is MISSING, a cell config is wholly void, or a pinned host has no null rows.
"""
import math
import re
import sys
from collections import defaultdict

HOSTS = ("gb10a-neon", "tr9970x-avx2", "m2ultra1-neon")      # chipset keys (docs/machines.md)
TOKENIZERS = ("gpt2", "llama3", "glm53", "qwen38", "o200k", "gemma4", "nemotron3-4b", "llama4", "minimaxm2", "dsv4",
              "kimik3")
CORPORA = ("en", "code", "ml", "cjk")
CHUNKS = ("4096", "0")
DECLARED = [(h, t, c, k) for h in HOSTS for t in TOKENIZERS for c in CORPORA for k in CHUNKS]
NA = {("kimik3", c, "0"): "ids differ: gigatoken runs kimi without the wrapper's cuts" for c in CORPORA}
PAR_HOSTS = ("gb10a-neon", "tr9970x-avx2")   # linux, pinned (tests/par/gate_par.sh); m2ultra1's rows are unpinned (shape)
PAR_ROWS = [("doc", s, k) for s in ("1", "4", "16", "32") for k in ("1", "2", "4", "8")] + \
           [("batch", "16", k) for k in ("1", "2", "4", "8")]
STATES = ("coldo", "cold", "pass", "lang", "warm", "warmo")   # coldo first: the headline (decided 2026-10-05)
BYTE_LEVEL = [t for t in TOKENIZERS if t != "gemma4"]   # T8's bar is byte-level bpe; gemma4 is sentencepiece bpe
T8_BAR = 600.0          # the T8 en bar: encode en-prose >= 600 MB/s per core (set on gb10), read in the pass state
SNAME = {"cold": "cold", "coldo": "coldo", "pass": "pass", "lang": "lang-x", "warm": "warm", "warmo": "warmo"}
CONFIGS = {   # config: (its states in the gate, its name); the caches each run measured are read from its lines
    "default": (STATES, "each tool's default caches, UNMATCHED"),
    "m6": (("warm", "warmo"), "matched bytes, gpt2 and gemma4 only: toks' default caches vs GIGA_CACHE_MIB=6"),
}
M6_NOTE = (   # why m6 covers two tokenizers (decided 2026-10-06); the probe: docs/bench/raw/gate-gb10a-neon-89adbba-m6probe.log
    "The matched config m6 (toks' default caches against GIGA_CACHE_MIB=6, the same cache bytes) is run for gpt2 and\n"
    "gemma4 alone: gigatoken floors a budget below its vocabulary seed, and only those two seeds fit 6 MiB (65,602 and\n"
    "7,085 entries; the other nine start from 142,738 to 256,944), so for them the row would not be matched, and at the\n"
    "floor gigatoken's untimed passes over the OTHER text thrash: 6.4-41 s a rep against 1.3-2.2 s at its default budget\n"
    "(gb10a, en 4096, one rep: `docs/bench/raw/gate-gb10a-neon-89adbba-m6probe.log`), ~11 h a host over the 88 cells.\n"
    "Their replay rows stay default-vs-default, UNMATCHED: a race of cache sizes (512 MiB against 6 MiB), not of\n"
    "tokenizer speed. A 2 MiB match (toks' memo off) is not a configuration gigatoken runs: below its seed it floors.\n")
MAX_NULL = 0.02         # a cell whose own null |median - 1| exceeds this is UNCERTIFIED SHAPE
FLOOR = 0.01            # the null interval is widened to at least [1 - FLOOR, 1 + FLOOR] (the 1% tolerance)
LIB = {   # the tree a log's GIT label names -> the master library it measures (the gate trees add bench tooling only)
    "c2d88ef": "d11f9d1", "ef1f79a": "d11f9d1", "b5c7116": "d11f9d1", "5737a95": "d11f9d1",   # the before picture
    "f4b23a7": "ac14d02",                                                                     # the after picture
    "9eb1c6b": "89adbba",                          # the new default (the memo on): 0.3.0, the public root's library
}
FULL_N = 30
CONTROL = 0.03         # before -> after: a cell whose gigatoken (the control) moved this much or more is the host's drift


def boot(clusters):
    """(median, lo, hi) of the ratios in clusters (a list of run-pairs' rep ratios): the cluster bootstrap
    (decided 2026-10-05). With one ratio per cluster it is tools/bench/tokv1.c's boot(), line for line (the same xorshift, seed, draws)"""
    def med(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    k, x, m, mask = len(clusters), 0x9E3779B97F4A7C15, [], (1 << 64) - 1
    for _ in range(2000):
        v = []
        for _ in range(k):
            x ^= (x << 13) & mask
            x ^= x >> 7
            x ^= (x << 17) & mask
            v += clusters[x % k]
        m.append(med(v))
    m.sort()
    return med([r for c in clusters for r in c]), m[49], m[1950]


def ctx_of(null_rows):
    """(the widest |median - 1|, how many exceed MAX_NULL, how many): a host's null rows as context, not a band"""
    d = [abs(v["med"] - 1.0) for v in null_rows]
    return max(d, default=0.0), sum(1 for x in d if x > MAX_NULL), len(d)


def judge(lo, hi, nv, shape):
    """(verdict, certified, null lo, null hi): the gate interval [lo, hi] against the cell's null interval nv widened
    to at least [1 - FLOOR, 1 + FLOOR] (no null: the floor alone, and never certified)"""
    nlo = min(nv["lo"], 1.0 - FLOOR) if nv else 1.0 - FLOOR
    nhi = max(nv["hi"], 1.0 + FLOOR) if nv else 1.0 + FLOOR
    v = "WIN" if lo > nhi else "LOSS" if hi < nlo else "TIE"
    return v, not shape and nv is not None and abs(nv["med"] - 1.0) <= MAX_NULL, nlo, nhi


def reps(kv, st):
    v = kv.get(st + "_reps_s")
    return [float(x) for x in v.split(",")] if v else []


def parse(path):
    log = {"runs": defaultdict(dict), "blocks": {}, "void": set(), "exact": {}, "meta": [], "na": {},
           "pruns": defaultdict(dict), "pblocks": {}, "pvoid": set(), "pfail": []}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"RUN par=1 (.*?) side=(\w) (PAR|GIGA|FAILED) (.*)", line)
        if m:
            t = dict(re.findall(r"(\w+)=(\S+)", m.group(1)))
            key = (t["mode"], t["size"], t["k"], int(t["round"]), int(t["try"]), t["kind"])
            if m.group(3) == "FAILED":
                log["pfail"].append(line)
            else:
                log["pruns"][key][int(t["pos"])] = (m.group(2), dict(re.findall(r"(\w+)=(\S+)", m.group(4))))
            continue
        if line.startswith(("BLOCK par=1", "VOID par=1")):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            k = (kv["mode"], kv["size"], kv["k"], int(kv["round"]))
            if line.startswith("BLOCK"):
                log["pblocks"][k + (int(kv["try"]), kv["kind"])] = kv
            else:
                log["pvoid"].add(k + (kv["kind"],))
            continue
        m = re.match(r"RUN (.*?) side=(\w) (E2E|GIGA|CTR) (.*)", line)
        if m:
            tags = dict(re.findall(r"(\w+)=(\S+)", m.group(1)))
            if m.group(3) == "CTR":
                continue
            kv = dict(re.findall(r"(\w+)=(\S+)", m.group(4)))
            key = (tags["cfg"], tags["tk"], tags["corp"], tags["chunk"], int(tags["round"]), int(tags["try"]), tags["kind"])
            log["runs"][key][int(tags["pos"])] = (m.group(2), kv)
            if m.group(3) == "GIGA" and "na" in kv:
                log["na"][(tags["tk"], tags["corp"], tags["chunk"])] = "gigatoken refused: " + kv["na"]
        elif line.startswith("BLOCK "):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            log["blocks"][(kv["cfg"], kv["tk"], kv["corp"], kv["chunk"], int(kv["round"]), int(kv["try"]), kv["kind"])] = kv
        elif line.startswith("VOID "):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            log["void"].add((kv["cfg"], kv["tk"], kv["corp"], kv["chunk"], int(kv["round"]), kv["kind"]))
        elif line.startswith("EXACT "):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            m = re.search(r"toks_ids='\d+ ([0-9a-f]+)'", line)
            kv["toks_sha"] = m.group(1) if m else ""
            log["exact"][(kv["tk"], kv["corp"], kv["chunk"])] = kv
        elif line.startswith(("HOST", "GATE", "GIT", "UPTIME", "CC ", "GIGATOKEN", "CPU", "OS ", "TOKPAR")):
            log["meta"].append(line)
    gits = [m.split()[1] for m in log["meta"] if m.startswith("GIT ")]
    log["dgit"] = gits if log["runs"] else []       # the trees a host's doc rows and par rows measure, kept apart:
    log["pgit"] = gits if log["pruns"] else []      # one label can merge a doc log with a par log of another tree
    return log


def pairs(log, cfg, cell, st, kind):
    """the paired ratios of state st (other side / A) over the cell's rounds, one list per run-pair: the last try of
    each round, VOID rounds dropped; (clusters, toks MB/s reps, other side's MB/s reps, rounds used, rounds void)"""
    sides = "B" if kind == "gate" else "N"
    rat, a_mbs, b_mbs, used, void = [], [], [], 0, 0
    rounds = sorted({k[4] for k in log["runs"] if k[:4] == (cfg,) + cell})
    for r in rounds:
        if (cfg,) + cell + (r, kind) in log["void"]:
            void += 1
            continue
        tries = sorted(t for (c, tk, co, ch, rr, t, kd) in log["blocks"] if (c, tk, co, ch, rr, kd) == (cfg,) + cell + (r, kind))
        if not tries:
            continue
        run = log["runs"].get((cfg,) + cell + (r, tries[-1], kind))
        if not run or sorted(run) != [1, 2, 3, 4] or [run[p][0] for p in (1, 2, 3, 4)] != ["A", sides, sides, "A"]:
            continue
        used += 1
        for pa, pb in ((1, 2), (4, 3)):
            a, b = reps(run[pa][1], st), reps(run[pb][1], st)
            n = min(len(a), len(b))
            c = [b[i] / a[i] for i in range(n) if a[i] > 0]
            if c:
                rat.append(c)
            byt = float(run[pa][1].get("bytes", 0))
            a_mbs += [byt / s / 1e6 for s in a[:n] if s > 0]
            b_mbs += [byt / s / 1e6 for s in b[:n] if s > 0]
    return rat, a_mbs, b_mbs, used, void


def par_pairs(log, row, kind):
    """the par row's pass samples paired like pairs(): (clusters, toks MB/s, gigatoken MB/s, ids the same, every
    sample new text (pass_seen 0), rounds void)"""
    side = "B" if kind == "gate" else "N"
    cl, am, bm, same, fresh, void = [], [], [], True, True, 0
    for r in sorted({k[3] for k in log["pruns"] if k[:3] == row}):
        if row + (r, kind) in log["pvoid"]:
            void += 1
            continue
        tries = sorted(k[4] for k in log["pblocks"] if k[:4] == row + (r,) and k[5] == kind)
        run = log["pruns"].get(row + (r, tries[-1], kind)) if tries else None
        if not run or [run.get(p, ("?",))[0] for p in (1, 2, 3, 4)] != ["A", side, side, "A"]:
            continue
        for pa, pb in ((1, 2), (4, 3)):
            a, b = run[pa][1], run[pb][1]
            same &= a.get("pass_fnv_nopp") == b.get("pass_fnv_nopp")
            fresh &= float(a.get("pass_seen", 1)) == 0.0 and float(b.get("pass_seen", 1)) == 0.0
            ta = [float(x) for x in a.get("pass_reps_ms", "").split(",") if x]
            tb = [float(x) for x in b.get("pass_reps_ms", "").split(",") if x]
            n = min(len(ta), len(tb))
            if n:
                cl.append([tb[i] / ta[i] for i in range(n) if ta[i] > 0])
                byt = float(a.get("bytes", 0))
                am += [byt / t / 1e3 for t in ta[:n] if t > 0]
                bm += [byt / t / 1e3 for t in tb[:n] if t > 0]
    return cl, am, bm, same, fresh, void


def median(v):
    v = sorted(v)
    return (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])) if v else 0.0


def chs(ch):
    return "whole" if ch == "0" else ch


def compute(logs):
    """every row's statistic and verdict: a dict of res / null / status / band (the doc rows) and pres / pnull /
    pstat / pband (the par rows), keyed by host first"""
    res = {}          # (host, cfg, tk, corp, ch, st) -> dict
    null = {}         # (host, tk, corp, ch, st) -> dict
    status = {}       # (host, tk, corp, ch) -> "ok" | "VOID (...)" | "n/a (...)"
    band = {}         # host -> (resolution, band)
    pre = set()       # (host, tk, corp, ch): pinned cells measured before the L3 rule
    vfull = {}        # (host, cfg, tk, corp, ch, st) -> rounds void: logged, but every round void (no pairs)
    for host, log in logs.items():
        pinned = any(re.search(r"PINCPU '\d", m) for m in log["meta"])   # macOS logs: unpinned, no L3 rule
        cells = sorted({k[1:4] for k in log["runs"]} | set(log["exact"]))
        ok = []
        for cell in cells:
            ex = log["exact"].get(cell, {})
            if ex.get("toks") != "yes":
                status[(host,) + cell] = "VOID (toks ids != hf)" if ex else "VOID (no exactness line)"
                continue
            want = ex.get("toks_sha", "")[:16]                 # every timed toks run: the same ids
            bad = sorted({kv.get("sha") for k, run in log["runs"].items() if k[1:4] == cell
                          for side, kv in run.values() if side in "AN" and kv.get("sha", want)[:16] != want})
            if bad:
                status[(host,) + cell] = f"VOID (a timed toks run's ids differ from the checked ones: {', '.join(bad)})"
                continue
            why = NA.get(cell) or log["na"].get(cell) or (None if ex.get("gigatoken") == "yes" else
                                                           f"ids differ (gigatoken vs hf without the post-processor: {ex.get('gigatoken')})")
            status[(host,) + cell] = f"n/a ({why})" if why else "ok"
            ok.append((cell, why))
            if pinned and any("l3_busy" not in b for k, b in log["blocks"].items() if k[1:4] == cell):
                pre.add((host,) + cell)                # pinned, measured before the L3 rule: provisional until re-run
        for cell, _ in ok:            # the null first: its resolution is the host's band
            for st in STATES:
                cl = pairs(log, "default", cell, st, "null")[0]
                if cl:
                    med, lo, hi = boot(cl)
                    null[(host,) + cell + (st,)] = dict(n=sum(map(len, cl)), k=len(cl), med=med, lo=lo, hi=hi)
        band[host] = ctx_of([v for k, v in null.items() if k[0] == host])
        for cell, why in ok:
            if why:
                continue
            for cfg, (sts, _) in CONFIGS.items():
                for st in sts:
                    cl, am, bm, used, void = pairs(log, cfg, cell, st, "gate")
                    if not cl and void:                 # runs logged, every round void: the state is VOID, not absent
                        vfull[(host, cfg) + cell + (st,)] = void
                    if cl:                              # the matched configs (warm) take the cell's default warm null
                        med, lo, hi = boot(cl)
                        v, cert, nlo, nhi = judge(lo, hi, null.get((host,) + cell + (st,)), not pinned)
                        res[(host, cfg) + cell + (st,)] = dict(n=sum(map(len, cl)), k=len(cl), med=med, lo=lo, hi=hi,
                                                              v=v, cert=cert, nlo=nlo, nhi=nhi, a=median(am), b=median(bm),
                                                              void=void, used=used)
    budget = {}       # (host, cfg) -> the caches its runs measured: toks' piece cache / memo, gigatoken's budget
    for host, log in logs.items():
        tk, gb, spm = defaultdict(set), defaultdict(set), set()
        for k, run in log["runs"].items():
            for side, kv in run.values():
                if side in "AN" and "memo_mib" in kv:      # cache_mib 0 = the default piece cache, 2 MiB
                    tk[k[0]].add(f"{int(kv.get('cache_mib', 0)) or 2} MiB piece cache + {kv['memo_mib']} MiB memo")
                elif side == "B" and "cache_mib" in kv:
                    gb[k[0]].add(f"{kv['cache_mib']} MiB")
                    if kv.get("kind") == "spm":
                        spm.add(k[0])
        for cfg in tk:
            budget[(host, cfg)] = (f"toks {' / '.join(sorted(tk[cfg]))} vs gigatoken {' / '.join(sorted(gb[cfg])) or '?'}"
                                   f"{' (+ its unit memo on sentencepiece)' if cfg in spm else ''}")
    pres, pnull, pstat, pband = {}, {}, {}, {}   # par rows: (host, mode, size, k) -> ...
    for host, log in logs.items():
        rows = sorted({k[:3] for k in log["pruns"]}, key=lambda x: (x[0], float(x[1]), int(x[2])))
        for row in rows:
            cl = par_pairs(log, row, "null")[0]
            if cl:
                med, lo, hi = boot(cl)
                pnull[(host,) + row] = dict(n=sum(map(len, cl)), med=med, lo=lo, hi=hi)
        if rows:
            pband[host] = ctx_of([v for k, v in pnull.items() if k[0] == host])
        for row in rows:
            cl, am, bm, same, fresh, void = par_pairs(log, row, "gate")
            pstat[(host,) + row] = "ok" if same and fresh else "n/a (ids differ)" if not same else "n/a (HALF-WARM)"
            if cl and same and fresh:
                med, lo, hi = boot(cl)
                v, cert, nlo, nhi = judge(lo, hi, pnull.get((host,) + row), False)
                pres[(host,) + row] = dict(n=sum(map(len, cl)), med=med, lo=lo, hi=hi, v=v, cert=cert, nlo=nlo, nhi=nhi,
                                           a=median(am), b=median(bm), void=void)
    return dict(res=res, null=null, status=status, band=band, pres=pres, pnull=pnull, pstat=pstat, pband=pband, pre=pre,
                vfull=vfull, budget=budget)


def load(specs):
    """label=log[,log] specs -> {label: parsed log}"""
    logs = {}
    for a in specs:                                 # label=log[,log]: a host's doc, matched and par logs, in order;
        label, paths = a.split("=", 1)              # a later log's (config, cell) or par row replaces an earlier one's
        for path in paths.split(","):               # (a re-run of contended cells)
            one = parse(path)
            if label not in logs:
                logs[label] = one
                continue
            old = logs[label]
            cc = {(k[0],) + k[1:4] for k in one["runs"]}
            pr = {k[:3] for k in one["pruns"]}
            for key in ("runs", "blocks"):
                for k in [k for k in old[key] if (k[0],) + k[1:4] in cc]:
                    del old[key][k]
            old["void"] -= {k for k in old["void"] if (k[0],) + k[1:4] in cc}
            for key in ("pruns", "pblocks"):
                for k in [k for k in old[key] if k[:3] in pr]:
                    del old[key][k]
            old["pvoid"] -= {k for k in old["pvoid"] if k[:3] in pr}
            for key, v in one.items():
                if isinstance(v, dict):
                    logs[label][key].update(v)
                elif isinstance(v, set):
                    logs[label][key] |= v
                else:
                    logs[label][key] += v
    return logs


def gitlib(gits):
    """git `a`, `b`, library master x, y: the trees a log list names and the master libraries they measure"""
    gits = list(dict.fromkeys(gits))
    return f"git `{'`, `'.join(gits) or '?'}`, library master {', '.join(dict.fromkeys(LIB.get(x, '?') for x in gits)) or '?'}"


def pinned(log):
    return any(re.search(r"PINCPU '\d", m) for m in log["meta"])     # macOS logs: unpinned (shape), no L3 rule


def train(w, logs, c, blogs, b):
    """the before -> after section: per host and state (default config), toks' MB/s change per cell beside
    gigatoken's (the control: its library did not change, so its column is the host's drift), the certified verdict
    transitions, the cells whose verdict moved and the largest toks moves where the control held"""
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    w("\n## Before -> after (default config; MB/s are the medians of the paired reps, per cell)\n")
    w("gigatoken is the control: its library is the same in both pictures, so its change is the host's drift between them.")
    w("Verdict transitions count the cells certified in both pictures.\n")
    w("| host | state | cells | toks MB/s change: median [min, max] | gigatoken (control): median [min, max] | verdicts before -> after |")
    w("|---|---|---:|---|---|---|")
    moved, top, drift = [], [], 0
    for host in logs:
        if host not in blogs:
            continue
        for st in STATES:
            ks = [k for k in c["res"] if k[0] == host and k[1] == "default" and k[5] == st and k in b["res"]]
            if not ks:
                continue
            dt = [c["res"][k]["a"] / b["res"][k]["a"] - 1.0 for k in ks]
            dg = [c["res"][k]["b"] / b["res"][k]["b"] - 1.0 for k in ks]
            tr = {}
            for k in ks:
                va, vb = c["res"][k], b["res"][k]
                if va["cert"] and vb["cert"] and va["v"] != vb["v"]:
                    tr[f"{vb['v']} -> {va['v']}"] = tr.get(f"{vb['v']} -> {va['v']}", 0) + 1
                    moved.append((host, k, vb, va))
                if abs(va["b"] / vb["b"] - 1.0) < CONTROL:
                    top.append((abs(va["a"] / vb["a"] - 1.0), host, k, vb, va))
                else:
                    drift += 1
            w(f"| {host} | {SNAME[st]} | {len(ks)} | {med(dt):+.1%} [{min(dt):+.1%}, {max(dt):+.1%}] | "
              f"{med(dg):+.1%} [{min(dg):+.1%}, {max(dg):+.1%}] | "
              f"{', '.join(f'{x} {n}' for x, n in sorted(tr.items())) or 'none moved'} |")
    w("")
    for host in logs:
        if host in blogs:
            npre = sum(1 for k in b["pre"] if k[0] == host)
            w(f"- {host}: before {gitlib(blogs[host]['dgit'])}"
              f"{f' ({npre} cells measured before the L3 rule)' if npre else ''}, "
              f"{b['budget'].get((host, 'default'), '?')}; after {gitlib(logs[host]['dgit'])}, "
              f"{c['budget'].get((host, 'default'), '?')}")
    if moved:
        w("\nCells whose certified verdict moved (toks / gigatoken MB/s, ratio [95%], before -> after):\n")
        for host, k, vb, va in sorted(moved, key=lambda x: (x[0], x[1])):
            w(f"- {host} {k[2]} {k[3]} {chs(k[4])} {SNAME[k[5]]}: {vb['a']:.1f} / {vb['b']:.1f}, {vb['med']:.2f} "
              f"[{vb['lo']:.2f}, {vb['hi']:.2f}] {vb['v']} -> {va['a']:.1f} / {va['b']:.1f}, {va['med']:.2f} "
              f"[{va['lo']:.2f}, {va['hi']:.2f}] {va['v']}")
    top = [x for x in top if x[0] >= 0.01]          # moves under 1% are within a host's run-to-run noise
    w(f"\nThe largest toks moves where the control held (gigatoken within {CONTROL:.0%}; {drift} cell states where it moved "
      f"more are left out as the host's drift): MB/s before -> after, change; the ratio to gigatoken before -> after"
      + (":\n" if top else ": none of 1% or more\n"))
    for d, host, k, vb, va in sorted(top, key=lambda x: -x[0])[:12]:
        w(f"- {host} {k[2]} {k[3]} {chs(k[4])} {SNAME[k[5]]}: {vb['a']:.1f} -> {va['a']:.1f} "
          f"({va['a'] / vb['a'] - 1.0:+.1%}); {vb['med']:.2f} -> {va['med']:.2f}")


def main():
    specs, before, argv, i = [], [], sys.argv[1:], 0
    while i < len(argv):                            # --before label=log[,log]: the picture the train started from
        if argv[i] == "--before" and i + 1 < len(argv):
            before.append(argv[i + 1])
            i += 2
        else:
            specs.append(argv[i])
            i += 1
    logs = load(specs)
    blogs = load(before) if before else {}
    c = compute(logs)
    res, null, status, band, pre, vfull = c["res"], c["null"], c["status"], c["band"], c["pre"], c["vfull"]
    pres, pnull, pstat, pband = c["pres"], c["pnull"], c["pstat"], c["pband"]
    out = []
    w = out.append
    pin = [h for h in logs if pinned(logs[h])]
    rc = 0
    w("# The gigatoken gate\n")
    # the finish line: every certified cell a WIN in every state, the par rows included (the pinned hosts certify)
    parts, nshape, nlost = [], 0, 0
    groups = [(SNAME[st], [v for k, v in res.items() if k[0] in pin and k[1] == "default" and k[5] == st]) for st in STATES]
    for name, rows in groups + [("par", [v for k, v in pres.items() if k[0] in pin])]:
        cert = [v for v in rows if v["cert"]]
        lost = sum(1 for v in cert if v["v"] != "WIN")
        nshape += len(rows) - len(cert)
        nlost += lost
        parts.append(f"{name} {lost} of {len(cert)}")
    nmiss = sum(1 for d in DECLARED if d not in status)
    won = not (nlost or nmiss or nshape)
    w(f"**Finish line**: every certified cell a WIN in every state, the par rows included, on the pinned hosts "
      f"({', '.join(pin) or 'none logged'}): **{'reached' if won else 'not reached'}**. Certified cells not won: "
      f"{'; '.join(parts)}. {nshape} rows UNCERTIFIED SHAPE, {nmiss} declared cells MISSING."
      f"{' The unpinned hosts (' + ', '.join(h for h in logs if h not in pin) + ') are shape only.' if len(pin) < len(logs) else ''}\n")
    w("Generated by `tools/bench/gate_table.py` from `tools/bench/gate.sh` logs (never edit by hand): toks against gigatoken")
    w("on one thread, the same chunks, the same pinned cpu; ratio = t_gigatoken / t_toks (> 1 = toks faster); WIN / TIE /")
    w("LOSS by each cell's own toks-vs-toks null (the protocol is below the tables). coldo is the headline, cold beside it;")
    w("the T8 en bar reads the pass state. The default config's warm and warmo replay from each tool's DEFAULT caches,")
    w("unmatched; the Tally's warm lines name the caches each config measured.\n")
    w(M6_NOTE if any(k[1] == "m6" for k in res) else
      "The matched config m6 is not in this picture: its logs hold the default config only.\n")
    w("## Tally\n")
    for host in list(logs) + [h for h in HOSTS if h not in logs]:
        if host not in logs:
            w(f"**{host}**: no log: all {len(CORPORA) * len(CHUNKS) * len(TOKENIZERS)} declared cells MISSING\n")
            rc = 1
            continue
        log = logs[host]
        shape = host not in pin
        st_cells = [k for k in status if k[0] == host]
        nv = sum(1 for k in st_cells if status[k].startswith("VOID"))
        nna = sum(1 for k in st_cells if status[k].startswith("n/a"))
        missing = [d for d in DECLARED if d[0] == host and d not in status]
        nl = [v for k, v in null.items() if k[0] == host]
        widest, nover, _ = band.get(host, (0.0, 0, 0))
        npre = sum(1 for k in pre if k[0] == host)
        nvf = len({k[:5] for k in vfull if k[0] == host})
        rc |= 1 if missing or nvf or (not shape and not nl) else 0
        bl = list(log["blocks"].values())
        ld = sorted(float(b.get("load0", 0)) for b in bl)
        l3 = sorted(float(b["l3_busy"]) for b in bl if b.get("l3_busy", "na") != "na")
        w(f"**{host}** ({gitlib(log['dgit'])})"
          f"{' **UNCERTIFIED SHAPE** (unpinned: macOS cannot pin; its verdicts are shape, the pinned hosts certify)' if shape else ''}"
          f": null rows {nover} of {len(nl)} over 2% (those cells UNCERTIFIED SHAPE), the widest {widest:.2%} (context, not a band)"
          f"{' **NO NULL ROWS: nothing certifiable**' if not nl else ''}; {len(st_cells)} cells logged, "
          f"{nv} VOID, {nna} n/a, {len(missing)} MISSING"
          f"{f'; {nvf} cell configs with a state VOID by the void rule (every round void)' if nvf else ''}"
          f"{f'; {npre} measured before the L3 rule (provisional until re-run)' if npre else ''}. "
          f"Blocks {len(bl)}, {sum(1 for b in bl if b.get('try') == '2')} run again, {len(log['void'])} VOID; 1-minute load "
          f"median {ld[len(ld) // 2] if ld else 0:.1f} (max {ld[-1] if ld else 0:.1f})"
          f"{f'; the busiest other L3 cpu: p90 {l3[int(len(l3) * 0.9)]:.0f}%, max {l3[-1]:.0f}%' if l3 else ''}\n")
        prow = [v for k, v in pres.items() if k[0] == host]
        pmiss = [r for r in PAR_ROWS if host in PAR_HOSTS and (host,) + r not in pstat]
        pfail = log["pfail"]
        if prow or pmiss or pfail:
            pw, po, pk = pband.get(host, (0.0, 0, 0))
            cert = [v for v in prow if v["cert"]]
            n = {x: sum(1 for v in cert if v["v"] == x) for x in ("WIN", "TIE", "LOSS")}
            rc |= 1 if pmiss or pfail or not pk else 0
            w(f"- **par** ({gitlib(log['pgit'])}; toks_par pass vs gigatoken pass, k threads, every timed byte new to its "
              f"pool; each row's own null, {po} of {pk} over 2%, the widest {pw:.2%}): {n['WIN']} WIN / {n['TIE']} TIE / "
              f"{n['LOSS']} LOSS of {len(cert)} certified{f', {len(prow) - len(cert)} UNCERTIFIED SHAPE' if len(prow) > len(cert) else ''}"
              f"{f', {len(pmiss)} MISSING' if pmiss else ''}{f', {len(pfail)} FAILED runs' if pfail else ''}")
        for cfg, (sts, name) in CONFIGS.items():
            for st in sts:
                rows = [v for k, v in res.items() if k[0] == host and k[1] == cfg and k[5] == st]
                if not rows:
                    continue
                cert = [v for v in rows if v["cert"]]
                shp = [v for v in rows if not v["cert"]]
                n = {x: sum(1 for v in cert if v["v"] == x) for x in ("WIN", "TIE", "LOSS")}
                cs = {x: sum(1 for v in shp if v["v"] == x) for x in ("WIN", "TIE", "LOSS")}
                small = sum(1 for v in rows if v["n"] < FULL_N)
                shape_txt = f"; {len(shp)} UNCERTIFIED SHAPE ({cs['WIN']} / {cs['TIE']} / {cs['LOSS']})" if shp else ""
                bud = c["budget"].get((host, cfg), "?")
                head = "**coldo** (defaults; the headline)" if st == "coldo" and cfg == "default" else \
                    f"{SNAME[st]}{'' if cfg == 'default' else ' ' + cfg} " \
                    f"({name + ': ' + bud if st == 'warm' or cfg != 'default' else 'defaults'})"
                tally = f"shape {cs['WIN']} WIN / {cs['TIE']} TIE / {cs['LOSS']} LOSS of {len(shp)}" if shape else \
                    f"{n['WIN']} WIN / {n['TIE']} TIE / {n['LOSS']} LOSS of {len(cert)} certified{shape_txt}"
                w(f"- {head}: {tally}{f'; {small} with n < {FULL_N} (provisional)' if small else ''}")
        t8 = []
        for ch in CHUNKS:
            part = []
            for st in ("pass", "cold"):
                v = sorted((res[k]["a"], k[2]) for k in res if k[0] == host and k[1] == "default" and k[3] == "en"
                           and k[4] == ch and k[5] == st and k[2] in BYTE_LEVEL)
                if v:
                    part.append(f"{st} {sum(1 for a, _ in v if a >= T8_BAR)} of {len(v)} (toks {v[0][0]:.0f} {v[0][1]} .. "
                                f"{v[-1][0]:.0f} {v[-1][1]})")
            if part:
                t8.append(f"{chs(ch)}: {', '.join(part)}")
        if t8:
            w(f"- **T8 en bar** (en >= {T8_BAR:.0f} MB/s per core, the pass state, cold beside it; byte-level bpe cells at or "
              f"over it): {'; '.join(t8)}")
        w("")
    order = {s: i for i, s in enumerate(STATES)}
    w("## Not won (every TIE and LOSS, by state: toks MB/s vs gigatoken MB/s, the medians of the paired reps; ratio")
    w("[interval], n; the cell's null interval, widened to at least [0.99, 1.01]; shape = UNCERTIFIED SHAPE)\n")
    w("| state | host | config | tokenizer | corpus | chunk | toks MB/s | gigatoken MB/s | ratio [95%] | n | null [95%] | verdict |")
    w("|---|---|---|---|---|---|---:|---:|---|---:|---|---|")
    for k, v in sorted(res.items(), key=lambda x: (order[x[0][5]],) + x[0]):
        if v["v"] != "WIN":
            host, cfg, tk, corp, ch, st = k
            w(f"| {SNAME[st]} | {host} | {cfg} | {tk} | {corp} | {chs(ch)} | {v['a']:.1f} | {v['b']:.1f} | "
              f"{v['med']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] | {v['n']}{' (' + str(v['void']) + ' void)' if v['void'] else ''} | "
              f"[{v['nlo']:.3f}, {v['nhi']:.3f}] | {v['v']}{'' if v['cert'] else ' shape'} |")
    if pres or pstat:
        w("\n## Par rows (toks_par_encode / _batch pass vs gigatoken's parallel api, k threads on the same pinned cpus)\n")
        for host, log in logs.items():
            if log["pgit"]:
                w(f"- {host}: {gitlib(log['pgit'])}")
        w("")
        w("| host | mode | MiB | k | toks MB/s | gigatoken MB/s | ratio [95%] | n | null [95%] | verdict |")
        w("|---|---|---:|---:|---:|---:|---|---:|---|---|")
        for k in sorted(pstat, key=lambda x: (x[0], x[1], float(x[2]), int(x[3]))):
            v = pres.get(k)
            if v:
                w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v['a']:.0f} | {v['b']:.0f} | {v['med']:.2f} [{v['lo']:.2f}, "
                  f"{v['hi']:.2f}] | {v['n']}{' (' + str(v['void']) + ' void)' if v['void'] else ''} | "
                  f"[{v['nlo']:.3f}, {v['nhi']:.3f}] | {v['v']}{'' if v['cert'] else ' shape'} |")
            else:
                w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | | | | | | {pstat[k]} |")
        for host, log in logs.items():
            for line in log["pfail"]:
                w(f"- {host} FAILED: `{line[:200]}`")
    w("\n## Protocol\n")
    w("One thread, the tier toks_load binds, toks (`tools/bench/e2e.c`) against gigatoken (`tools/bench/gigatoken`, its Rust")
    w("API, pinned source) on exactly the same chunks, both on one pinned cpu (macOS: unpinned). Per cell, ROUNDS blocks of")
    w("A B B A (A = toks, B = gigatoken); ratio = t_gigatoken / t_toks (> 1 = toks faster), paired rep i with rep i within")
    w("each run-pair ((A, B) and (B, A) of a block). **Statistic** (decided 2026-10-05):")
    w("the median of the pairs and a 95% cluster-bootstrap interval that resamples run-pairs (a run is the independent unit:")
    w("its reps move together), 2000 resamples, tokv1.c's seed. **Each cell's own null decides it**: every round of a cell")
    w("also runs a null block A N N A (toks against the same toks), whose pairs give the cell's null median and interval")
    w("(6 run-pairs today, 30 at release). **WIN** iff the gate's interval lies wholly above the null's interval widened to")
    w("at least [0.99, 1.01] (the 1% tolerance: a lucky 6-pair null cannot certify a 0.5% win), **LOSS** iff")
    w("wholly below it, else **TIE** (not a win). The matched configs (warm) take the same cell's default-config warm null.")
    w("**Certification is per cell**: a cell whose null |median - 1| exceeds 2% is **UNCERTIFIED SHAPE** and prints so, as")
    w("does every cell of an unpinned host (macOS cannot pin); a host's widest null is printed as")
    w("context, never as a band. Why per cell, the worked example (tr9970x-avx2's before picture, 2026-10-04): 19 of its 352")
    w("null rows exceed 2% (cold 11, pass 5, lang-x 3, warm 0; nearly all cjk / ml, the worst minimaxm2 cjk 4096 cold")
    w("1.084). The noise is run-level, not clock-level: a whole process's 5 reps come out slow together (one N run's best")
    w("cold 20.95 ms against 18.5-19.4 for the other eleven) while cpu 12 held 5.03 GHz (p90 run-to-run 0.53%, correlation")
    w("0.21 with the cold difference): page placement, or memory traffic from builds on the other CCDs, which neither the")
    w("sibling, L3 nor load rule sees. The first rule's host-wide band (the largest null, rounded up) gave every tr9970x cell")
    w("+-8.5%; the per-cell null keeps cjk cold's noise in cjk cold. n = the pairs (30 at ROUNDS 3, REPS 5). **Cold pairs**")
    w("(the gate's rule): a chunked cell's gigatoken runs take ONE cold rep each (its cold forks a vocabulary-seeded")
    w("state before every call, 5-60 ms), so its cold n is 6 and the row is provisional (n < 30); the 30-pair cold matrix is a")
    w("release-time run. **Void rule** per block (one rule for every row, decided 2026-10-05): the")
    w("pinned cpu's smt sibling > 5% busy over the block, or ANY other cpu sharing its last-level cache > 50% busy (a build or")
    w("test on the same CCD / cluster shares the cache and memory bandwidth the timed run reads: on tr9970x a make test on CCD1's")
    w("8-11 never touched cpu 12's sibling; on gb10b a foreign process at 100% on cpu 5 shares cpu 7's L3 with cpus 0-9 and")
    w("marks every block there, so gb10b is no gate host), or the 1-minute load up by more than 2 across it: the block runs")
    w("again, and a second such block is VOID (its pairs dropped; the row's n says so). BLOCK lines carry the times and every")
    w("L3 cpu's busy share. macOS hosts cannot pin and expose no cache topology: their rows run unpinned, load rule only. A")
    w("re-run of a cell (contended, or measured before a rule existed) replaces it. States: **cold** (a fresh scratch / fork before every call),")
    w("**coldo** (each cold rep after an untimed pass over the OTHER text: a new request after others),")
    w("**pass** (after an untimed pass over OTHER text, scratch / state re-initialized), **lang-x** (the timed pass on the")
    w("scratch / state the OTHER text left; OTHER = every bench corpus file outside the measured corpus, cut at 4096 bytes:")
    w("cross-language, not a same-language warm state), **warm** (the replay right after the pass), **warmo** (init,")
    w("a pass over the text, one over the OTHER text, the timed pass over the text, no init between: the serving replay, a")
    w("system prompt seen again after other requests) in the declared")
    w("cache configs. Exactness per cell, before its blocks: toks' ids == hf's (else VOID); gigatoken's == hf's without the")
    w("post-processor (else n/a).\n")
    w("Cold reps are cpu-cache-hot (back to back on one text); coldo's each follow an untimed pass over the OTHER text,")
    w("the state a new request meets after others: **coldo is the headline, cold is published beside it**")
    w("(decided 2026-10-05). Both tools are measured as their apis are called, which makes their cold differ: gigatoken's cold forks")
    w("a vocabulary-seeded state, untimed, right before every timed call, and that fork writes its state into the cpu")
    w("caches, so its coldo barely moves; toks' per-call init touches the scratch header only, so toks' coldo carries the")
    w("real misses. Per host, coldo against cold (the median and range over its cells of coldo MB/s / cold MB/s - 1):")
    for host in logs:
        ks = [k for k in res if k[0] == host and k[1] == "default" and k[5] == "coldo" and k[:5] + ("cold",) in res]
        if ks:
            ta = sorted(res[k]["a"] / res[k[:5] + ("cold",)]["a"] - 1.0 for k in ks)
            tb = sorted(res[k]["b"] / res[k[:5] + ("cold",)]["b"] - 1.0 for k in ks)
            w(f"{host} toks {median(ta):+.1%} [{ta[0]:+.1%}, {ta[-1]:+.1%}], gigatoken {median(tb):+.1%} [{tb[0]:+.1%}, "
              f"{tb[-1]:+.1%}] ({len(ks)} cells);")
    w("The T8 en bar (encode en-prose >= 600 MB/s per core, set on gb10) reads the **pass** state, cold beside it")
    w("(decided 2026-10-05): the Tally's T8 line per host, byte-level bpe tokenizers only (gemma4 is sentencepiece bpe).")
    w("\n## Cells n/a, VOID, MISSING\n")
    for k in sorted(status):
        if status[k] != "ok":
            w(f"- {k[0]} {k[1]} {k[2]} {chs(k[3])}: {status[k]}")
    for host in logs:
        for d in DECLARED:
            if d[0] == host and d not in status:
                w(f"- {host} {d[1]} {d[2]} {chs(d[3])}: MISSING")
    for k, nvoid in sorted(vfull.items()):
        w(f"- {k[0]} {k[2]} {k[3]} {chs(k[4])} {k[1]} {SNAME[k[5]]}: VOID (void rule: all {nvoid} rounds void, no pairs)")
    w("\n## Null rows over 2% (toks vs toks; each one's cell is UNCERTIFIED SHAPE), then each host's widest medians\n")
    for k, v in sorted(null.items()):
        if abs(v["med"] - 1.0) > MAX_NULL:
            w(f"- {k[0]} {k[1]} {k[2]} {chs(k[3])} {SNAME[k[4]]}: {v['med']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] n {v['n']}")
    w("")
    for host in logs:
        nl = sorted(((abs(v["med"] - 1.0), k) for k, v in null.items() if k[0] == host), reverse=True)[:3]
        if nl:
            w(f"- {host}: the widest null medians: " + ", ".join(f"{k[1]} {k[2]} {chs(k[3])} {SNAME[k[4]]} "
                                                               f"{null[k]['med']:.3f}" for _, k in nl))
    w("\n## All rows\n")
    w("| host | config | tokenizer | corpus | chunk | " + " | ".join(SNAME[s] for s in STATES) + " |")
    w("|---|---|---|---|---|" + "---|" * len(STATES))
    for host in logs:
        for cfg in CONFIGS:
            for cell in sorted({k[2:5] for k in list(res) + list(vfull) if k[0] == host and k[1] == cfg}):
                cols = []
                for st in STATES:
                    v = res.get((host, cfg) + cell + (st,))
                    vf = vfull.get((host, cfg) + cell + (st,))
                    cols.append(f"{v['v']}{'' if v['cert'] else ' shape'} {v['med']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] n {v['n']}" if v else
                                f"VOID (void rule) n 0" if vf else "-")
                mark = " (pre-L3)" if (host,) + cell in pre else ""
                w(f"| {host} | {cfg} | {cell[0]} | {cell[1]} | {chs(cell[2])}{mark} | " + " | ".join(cols) + " |")
    if blogs:
        train(w, logs, c, blogs, compute(blogs))
    w("\n## Logs\n")
    w(f"Regenerate: `python3 tools/bench/gate_table.py {' '.join(sys.argv[1:])} > docs/bench/gigatoken-gate.md`\n")
    files = {}
    for name, sp in (("", specs), ("before: ", before)):
        for a in sp:
            label, paths = a.split("=", 1)
            files.setdefault(label, []).append(name + ", ".join(f"`{x}`" for x in paths.split(",")))
    for host, log in logs.items():
        w(f"### {host}\n")
        for f in files.get(host, []):
            w(f"- {f}")
        for m in log["meta"]:
            w(f"- `{m}`")
        w("")
    print("\n".join(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
