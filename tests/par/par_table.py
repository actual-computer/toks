#!/usr/bin/env python3
"""tests/par/par_table.py: the toks_par scaling tables (docs/usage.md, threads) from tests/par/run_par.sh logs.

    python3 tests/par/par_table.py [--doc] <label>=<log> ...      (stdlib only)
    python3 tests/par/par_table.py --summary <host>=<log> <host>=<log>
    --doc: docs/bench/par.md as a whole (its header, then every log's tables)
    --summary: docs/usage.md's two tables (scaling: pass MB/s per k beside gigatoken; where going wide starts:
    the pool's participants and the median speedup per size, new text every call) for two hosts side by side

Per log: one big input (doc) and batches of documents (batch): for each size and k, toks_par's MB/s and latency
in the pass state (new text on a warm pool; first and warm beside it), its speedup over serial toks_encode
(k = 0, one scratch) and its efficiency (speedup / k), gigatoken's parallel api on the same cpus (RAYON threads
= k) and toks / gigatoken; batches also tiktoken's and hf's encode_batch with k threads (tools/bench/par_ref.py) and
toks / each; ids: every row's fnv_nopp must equal the serial row's (else the row says MISMATCH).
Then the sweep: serial vs the pool's own choice vs every participant forced, per size."""
import re
import sys


def parse(path):
    """rows (PAR / GIGA key=values, each with the cpus, busy shares and busiest processes of its run_par.sh cell) and
    the 1-minute loads"""
    rows, load, cell, cpus = [], [], [], ""
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith(("PAR ", "GIGA ", "PYREF ")):
            kv = dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))
            if not line.startswith("PYREF "):              # PYREF lines name their tool: tiktoken or hf
                kv["tool"] = "toks" if line.startswith("PAR ") else "giga"
            kv["cpus"] = cpus
            rows.append(kv)
            cell.append(kv)
        elif line.startswith("LOAD before"):
            load.append(float(line.split()[2]))
            m = re.search(r"cpus=(\S+)", line)
            cpus, cell = (m.group(1) if m else ""), []
        elif line.startswith("LOAD after"):
            load.append(float(line.split()[2]))
            for r in cell:
                r["busy"] = {int(c): int(v) for c, v in re.findall(r"cpu(\d+)=(\d+)%", line)}
        elif line.startswith("TOP "):
            for r in cell:
                r["top"] = [e.split() for e in line[4:].strip().split(";") if len(e.split()) >= 3]   # pcpu psr comm
        elif line.startswith("HOST"):
            rows.append({"tool": "host", "line": line.strip()})
    return rows, load


def receipts(r):
    """'min-max%' busy of the cell's pinned cpus (+ their smt siblings'), FOREIGN when a sibling was >= 50% busy or a
    process not ours ran >= 50% on one of them (run_par.sh's TOP line: pcpu psr comm; the receipts carry no user)"""
    pinned = {int(c) for c in r.get("cpus", "").split(",") if c}
    busy = r.get("busy", {})
    if not pinned or not busy:
        return ""
    own = [v for c, v in busy.items() if c in pinned]
    sib = [v for c, v in busy.items() if c not in pinned]
    foreign = any(v >= 50 for v in sib) or any(
        float(p) >= 50 and int(c) in busy and comm not in ("bench_par", "gigatoken-par")
        for p, c, comm, *_ in r.get("top", []) if re.match(r"^[0-9.]+$", p) and c.isdigit())
    s = f"{min(own)}-{max(own)}%" if own else ""
    s += f", sib {max(sib)}%" if sib else ""
    return s + (" FOREIGN" if foreign else "")


def mb(n):
    n = int(n)
    return f"{n / 1048576:g} MiB" if n >= 1048576 else f"{n // 1024} KiB"


def table(label, rows, load):
    host = next((r["line"] for r in rows if r["tool"] == "host"), "")
    out = [f"## {label}", "", f"`{host}`; 1-minute load over the run {min(load, default='?')} .. {max(load, default='?')}", ""]
    for mode in ("doc", "batch"):
        cells = [r for r in rows if r.get("mode") == mode]
        if not cells:
            continue
        refs = [t for t in ("tiktoken", "hf") if any(r["tool"] == t for r in cells)]
        out += ["| input | k | toks pass MB/s | pass ms | first MB/s | warm MB/s | x serial | eff | gigatoken pass MB/s "
                "| giga warm MB/s | toks / giga (pass) |" + "".join(f" {t} pass MB/s | toks / {t} (pass) |" for t in refs) +
                " pass_seen | busy (toks; giga) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|---:|" * len(refs) + "---:|---|"]
        sizes = sorted({int(r["bytes"]) for r in cells})
        for b in sizes:
            skip = next((r for r in cells if int(r["bytes"]) == b and "skip" in r), None)
            if skip is not None:
                what = mb(b) + (f" of {int(skip['doc']) // 1024} KiB docs" if mode == "batch" else "")
                out.append(f"| {what} | all | n/a: the text holds {int(skip['text_bytes']) / 1e6:.0f} MB "
                           f"| | | | | | | | | | |")
                continue
            ser = next((r for r in cells if r["tool"] == "toks" and int(r["bytes"]) == b and r["k"] == "0"), None)
            if ser is None:
                continue
            what = mb(b) + (f" of {int(ser['doc']) // 1024} KiB docs" if mode == "batch" else "")
            for r in [x for x in cells if x["tool"] == "toks" and int(x["bytes"]) == b and "pass_mbps" in x]:
                k = int(r["k"])
                same = r.get("fnv_nopp") == ser.get("fnv_nopp") and r.get("pass_fnv_nopp") == ser.get("pass_fnv_nopp")
                g = next((x for x in cells if x["tool"] == "giga" and int(x["bytes"]) == b and x["threads"] == str(k)), None)
                gsame = g is None or (g.get("fnv_nopp") == ser.get("fnv_nopp") and
                                      g.get("pass_fnv_nopp") == ser.get("pass_fnv_nopp"))
                x = float(r["pass_mbps"]) / float(ser["pass_mbps"])
                ks = "serial" if k == 0 else f"{k} (used {r.get('used', '?')})"
                gp = f"{float(g['pass_mbps']):.0f}" if g else ""
                gw = f"{float(g['warm_mbps']):.0f}" if g else ""
                ratio = f"{float(r['pass_mbps']) / float(g['pass_mbps']):.2f}" if g else ""
                bad = "" if same and gsame else " MISMATCH"
                seen = float(r.get("pass_seen", 0))
                half = " HALF-WARM" if seen > 0 else ""
                rcpt = receipts(r) + (f"; {receipts(g)}" if g else "")
                rf = ""
                for t in refs:                                  # tiktoken / hf: the same docs, k threads
                    p = next((x for x in cells if x["tool"] == t and int(x["bytes"]) == b and x["threads"] == str(k)), None)
                    if p is None or k == 0:
                        rf += " | |"
                        continue
                    tb = "" if p.get("fnv_nopp") == ser.get("fnv_nopp") else " MISMATCH"
                    rf += f" {float(p['pass_mbps']):.0f}{tb} | {float(r['pass_mbps']) / float(p['pass_mbps']):.2f} |"
                out.append(f"| {what} | {ks} | {float(r['pass_mbps']):.0f}{bad}{half} | {float(r['pass_ms']):.2f} | "
                           f"{float(r['first_mbps']):.0f} | {float(r['warm_mbps']):.0f} | {x:.2f} | "
                           f"{'' if k == 0 else f'{x / k:.0%}'} | {gp} | {gw} | {ratio} |{rf} {seen:.2f} | {rcpt} |")
        out.append("")
    sweeps = [r for r in rows if r.get("mode") == "sweep" and "eager_us" in r]
    for fr, gap in sorted({(r.get("fresh", "0"), r.get("gap_us", "0")) for r in sweeps}, key=lambda x: (-int(x[0]), int(x[1]))):
        sw = [r for r in sweeps if r.get("gap_us", "0") == gap and r.get("fresh", "0") == fr]
        txt = "new text every call" if fr == "1" else "the same text every call"
        out += [f"sweep, {txt}, {'back to back' if gap == '0' else f'{int(gap) // 1000} ms idle before each call'} "
                f"(us best / median of the reps; x = serial / variant, best and median):", "",
                "| size | serial us | pool's choice: participants | us | x | all forced: us | x | 2 forced: us | x |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]

        def bm(r, b, m):
            return f"{float(r[b]):.0f} / {float(r[m]):.0f}"

        def xs(r, b, m):
            return f"{float(r['serial_us']) / float(r[b]):.2f} / {float(r['med_serial_us']) / float(r[m]):.2f}"
        for r in sw:
            out.append(f"| {mb(r['bytes'])} | {bm(r, 'serial_us', 'med_serial_us')} | {r.get('used', '?')} | "
                       f"{bm(r, 'auto_us', 'med_auto_us')} | {xs(r, 'auto_us', 'med_auto_us')} | "
                       f"{bm(r, 'eager_us', 'med_eager_us')} | {xs(r, 'eager_us', 'med_eager_us')} | "
                       f"{bm(r, 'eager2_us', 'med_eager2_us')} | {xs(r, 'eager2_us', 'med_eager2_us')} |")
        last = sw[-1]
        out += ["", f"model after the sweep: {int(last.get('ns_per_mib', 0)) / 1e6:.2f} ms/MiB, a sleeping worker joins in "
                f"{int(last.get('wake_ns', 0)) / 1e3:.1f} us, a spinning one in {int(last.get('join_ns', 0)) / 1e3:.2f} us, "
                f"min_bytes {last.get('min_bytes', '?')}", ""]
    return "\n".join(out)


HEADER = """# toks_par scaling receipts

Generated by `tests/par/par_table.py --doc` from the raw logs in `docs/bench/raw/par-*.log` (`tests/par/run_par.sh`
on each host; never edit the numbers by hand). docs/usage.md (threads) has the policy these numbers come from.
host names are chipset keys (see docs/machines.md); home directories were replaced by $HOME in the receipts.

- **setup**: llama 3's tokenizer.json (pinned, sha in each log), enwik8 (100 MB) as text, flags 0 (hf's default call).
  Host cores: gb10b = the GB10's X925s (5-9, 15-17, 19; cpu 18 runs another user's busy-polling scheduler; the
  before run took 5-9, 16-19 while that scheduler sat on cpu 15), tr9970x = CCD1 of the Zen 5 (8-15; we left the SMT
  siblings 40-47 idle, another user's qemu VM floats over every core); a cell of k runs on the first k of them
  (`taskset`). Load (1 min) before and after every cell, each pinned cpu's and sibling's busy share and the busiest
  processes are in the raw logs.
- **doc** = one input of S MiB (`toks_par_encode`), **batch** = 16 MiB cut into documents of about 4 KiB at line
  ends (`toks_par_encode_batch`, every document its own out). serial = `toks_encode` on one scratch (a loop for the
  batch). k = the pool's cap n (`toks_par_create(.., k, 0)`), the participants it used in parentheses (the pool
  decides per call). x serial = MB/s / serial MB/s of the same state; eff = x / k.
- **states**: first = a fresh pool's first call (scratches and threads' first touches included); **pass** = new
  text on a warm pool (the serving state). From bench_par's windowed pass on (logs with `windows=`): REPS samples, each a
  fresh pool that encodes one window of the text untimed, then another window timed (every timed byte new to its
  pool), the median sample printed, every sample in pass_reps_ms. Logs before it: one pool after one call on the
  file's last S bytes. **pass_seen** = the share of the timed text that the warm-up held: 0 for every row of a file
  that holds the size twice; enwik8 (95.4 MiB) does not hold 64 MiB twice, so those rows of the older logs read
  0.51 and are flagged **HALF-WARM**: no pass row, outside the gate (which takes pass_seen = 0 rows only). warm =
  the same text again (best of 5 reps). The pass column is the one to read.
- **busy** (toks; gigatoken): the busy share range of the run's pinned cpus over the run, `sib x%` the most busy SMT
  sibling, FOREIGN when a sibling or a process not ours took >= 50% of a pinned cpu (run_par.sh's LOAD / TOP lines).
- **gigatoken** (the bar; pinned source, read-only): its parallel api (`encode_docs_ragged` over rayon with
  `RAYON_NUM_THREADS` = k; `tools/bench/gigatoken/par.rs`) on the same cpus, text and states. Its ids equal toks's
  without the post-processor (fnv over the ids, checked per row: a row that differs says MISMATCH).
- **tiktoken** and **hf** (batch rows; `tools/bench/par_ref.py`, what people run from python): tiktoken 0.14.0's
  `encode_batch(docs, num_threads=k)` over the Encoding the e2e harness builds from the same tokenizer.json
  (`tools/bench/e2e_ref.py`), and hf tokenizers 0.23.2's `encode_batch(docs)` (hf's default call) on rayon's pool of
  `RAYON_NUM_THREADS` = k, on the same cpus, documents and states. Both parallelize over documents only, so they have
  batch rows, not doc rows. hf's ids are the oracle: a row is EXACT when toks' ids equal them (fnv without the
  post-processor, per row; else MISMATCH); tiktoken's are checked the same way. pass_seen = the share of the timed text
  the warm-up call held (> 0 once 2 x the size exceeds enwik8: the 64 MiB rows are a half replay).
- **sweep** = where going wide starts to pay: one input of 4 KiB .. 4 MiB, serial vs the pool's own choice (n = 8)
  vs every participant forced (`TOKS_PAR_EAGER=1`, n = 8) vs a pool of two forced. new text every call (a server;
  each variant its own windows of the file, so compare the medians) or the same text every call (one core's caches
  hold it all while each participant sees only its units: the worst case for going wide); back to back or after
  5 ms idle (sporadic calls find the workers asleep: on gb10b every variant then runs 3-10x slower at small sizes,
  the cores' wake from deep idle). 5 reps per cell (run_par.sh REPS=5); us best / median, x = serial / variant.
- **before** = master b02ac89's toks_par (no cost model: n participants on every call it split, wake-all,
  default one thread per cpu) with the same bench: its "pool's choice" is what master did with n = 8, its "all
  forced" the same (master has no TOKS_PAR_EAGER) and "2 forced" a pool of two.
- Not a certified cell: no abba pairs, no bootstrap intervals; the hosts are shared (load recorded).
"""


def find(rows, tool, mode, b, k):
    for r in rows:
        if r.get("tool") == tool and r.get("mode") == mode and int(r.get("bytes", 0)) == b and \
                r.get("k" if tool == "toks" else "threads") == str(k):
            return r
    return None


def summary(hosts):
    """docs/usage.md's tables for [(name, rows)]: every toks / gigatoken row checked equal to the serial ids"""
    mib = 1 << 20
    head = " | ".join(f"{h} MB/s (x serial, eff) | ms | gigatoken | toks / giga" for h, _ in hosts)
    out = [f"| input | k | {head} |", "|---|---:|" + "---:|" * (4 * len(hosts))]
    for mode, b, label in (("doc", mib, "one input, 1 MiB"), ("doc", 4 * mib, "one input, 4 MiB"),
                           ("doc", 16 * mib, "one input, 16 MiB"), ("doc", 64 * mib, "one input, 64 MiB"),
                           ("batch", 16 * mib, "16 MiB of 4 KiB docs")):
        cells = []
        for _, rows in hosts:
            s = find(rows, "toks", mode, b, 0)
            cells += [f"{float(s['pass_mbps']):.0f} (serial)", f"{float(s['pass_ms']):.2f}", "", ""]
        out.append(f"| {label} | serial | " + " | ".join(cells) + " |")
        for k in (1, 2, 4, 8):
            cells = []
            for _, rows in hosts:
                s, t, g = find(rows, "toks", mode, b, 0), find(rows, "toks", mode, b, k), find(rows, "giga", mode, b, k)
                if t.get("fnv_nopp") != s.get("fnv_nopp") or (g is not None and g.get("fnv_nopp") != s.get("fnv_nopp")):
                    raise SystemExit(f"ids differ: {label} k {k}")
                x = float(t["pass_mbps"]) / float(s["pass_mbps"])
                cells += [f"{float(t['pass_mbps']):.0f} ({x:.2f}x, {x / k:.0%})", f"{float(t['pass_ms']):.2f}",
                          f"{float(g['pass_mbps']):.0f}" if g else "",
                          f"{float(t['pass_mbps']) / float(g['pass_mbps']):.2f}" if g else ""]
            out.append(f"| {label} | {k} | " + " | ".join(cells) + " |")
    out.append("")
    head = " | ".join(f"{h}, back to back | {h}, 5 ms idle" for h, _ in hosts)
    out += [f"| size | {head} |", "|---|" + "---:|" * (2 * len(hosts))]
    cols = []
    for _, rows in hosts:
        for gap in ("0", "5000"):
            cols.append([r for r in rows if r.get("mode") == "sweep" and "eager_us" in r and r.get("fresh") == "1"
                         and r.get("gap_us") == gap])
    for i in range(min(len(c) for c in cols)):
        cells = [f"{c[i]['used']}: {float(c[i]['med_serial_us']) / float(c[i]['med_auto_us']):.2f}x" for c in cols]
        out.append(f"| {mb(cols[0][i]['bytes'])} | " + " | ".join(cells) + " |")
    mins = [f"{h}: " + " / ".join(str(int(c[-1]["min_bytes"]) // 1000) + " KB" for c in cols[2 * j:2 * j + 2])
            for j, (h, _) in enumerate(hosts)]
    out += ["", "min_bytes after the sweeps (back to back / idle): " + "; ".join(mins)]
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        print(summary([(a.partition("=")[0], parse(a.partition("=")[2])[0]) for a in args[1:]]))
        return
    if args and args[0] == "--doc":
        print(HEADER)
        args = args[1:]
    for arg in args:
        label, _, path = arg.partition("=")
        rows, load = parse(path or label)
        print(table(label, rows, load))


if __name__ == "__main__":
    main()
