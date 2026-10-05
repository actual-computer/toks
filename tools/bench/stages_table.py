# /// script
# requires-python = ">=3.10"
# ///
"""tools/bench/stages_table.py: the stage profile of e2e.sh logs (STAGES=1 [COUNT=1], E2E_STATE=cold|pass) as a
markdown table: per cell the uninstrumented MB/s of the same state, then where the instrumented run's time goes,
in ns per input byte and per piece: K1 + K3, K5's own time split by the fit over its piece classes (1-byte,
static 2..15 B, cache hits, the K5 side of short misses, long pieces, per round), K6 (short / long calls), the
driver; then the K6 calls by cause and the counted traffic (COUNT=1: the c code's loads + stores per input byte).

    uv run tools/bench/stages_table.py label=log [label=log ...]
"""
import re
import sys


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def cells(path):
    out, cur = [], None
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"== (\S+) (\S+) chunk (\d+)", line)
        if m:
            cur = {"tok": m[1], "corp": m[2], "chunk": int(m[3])}
            out.append(cur)
        elif cur is None:
            continue
        elif line.startswith("LOAD before"):
            cur["load0"] = line.split("before ")[1].split()[0]
        elif line.startswith("LOAD after"):
            cur["load1"] = line.split("after ")[1].split()[0]
        elif line.startswith("BUSY"):
            cur["busy"] = line[5:].strip()
        elif line.startswith("E2E tool=toks"):
            cur["e2e"] = kv(line)
        elif line.startswith("STAGES"):
            cur["st"] = kv(line)
        elif line.startswith("CLASSIFY"):
            cur["cl"] = kv(line)
        elif line.startswith("COUNT"):
            cur["cnt"] = kv(line)
    return [c for c in out if "st" in c]


def row(label, c):
    st, e = c["st"], c.get("e2e", {})
    n, state = float(st["bytes"]), st.get("state", "pass")
    pieces = float(st["k5_pieces"])
    sec = {"cold": "cold_s", "coldo": "coldo_s", "pass": "pass_s"}[state]   # the uninstrumented twin of the state
    mbs = n / float(e[sec]) / 1e6 if sec in e else float("nan")
    tot = float(st["total_s"])
    fit = [float(x) for x in st["k5fit_ns"].split(",")]
    cls = [float(x) for x in st["k5_classes"].split(",")]
    k5calls = float(st["k5_calls"])
    k6c, k6s = float(st["k6_calls"]), float(st["k6_s"])
    k6sc, k6ss = float(st["k6short_calls"]), float(st["k6short_s"])
    k6lc, k6ls = k6c - k6sc, k6s - k6ss
    k5own = float(st["k5own_s"])
    parts = [
        ("K1+K3", float(st["k1_s"]) + float(st["k3_s"])),
        ("K5 1-byte", fit[0] * cls[0] * 1e-9),
        ("K5 static", fit[1] * cls[1] * 1e-9),
        ("K5 cache hits", fit[2] * cls[2] * 1e-9),
        ("K5 miss side", (fit[3] * cls[3] + fit[4] * cls[4]) * 1e-9),
        ("K5 rounds+resid", k5own - sum(f * x for f, x in zip(fit[:5], cls)) * 1e-9),
        ("K6 short", k6ss),
        ("K6 long", k6ls),
        ("driver", float(st["driver_s"])),
    ]
    cells_ = [f"{label} {c['tok']} {c['corp']} {c['chunk'] or 'whole'} {state} (after {st.get('pass_after', 'same')})",
              f"{mbs:.0f}", f"{n / tot / 1e6:.0f}",
              f"{tot / n * 1e9:.2f}", f"{tot / pieces * 1e9:.1f}"]
    for _, s in parts:
        cells_.append(f"{s / n * 1e9:.2f} ({100 * s / tot:.0f}%)")
    cells_.append(f"{100 * cls[0] / pieces:.1f} / {100 * cls[1] / pieces:.1f} / {100 * cls[2] / pieces:.1f} / "
                  f"{100 * cls[3] / pieces:.1f} / {100 * cls[4] / pieces:.2f}")
    cells_.append(f"{fit[0]:.1f} / {fit[1]:.1f} / {fit[2]:.1f} / {fit[3]:.1f} / {fit[4]:.1f} / {fit[5]:.0f} "
                  f"(r2 {float(st['k5fit_r2']):.2f})")
    cells_.append(f"{(k6ss / k6sc * 1e9 if k6sc else 0):.0f} / {(k6ls / k6lc * 1e9 if k6lc else 0):.0f}")
    cl = c.get("cl", {})
    if cl:
        cells_.append(f"{cl['m_multi_first']} / {cl['m_multi_again']} / {cl['m_vocab_unplaced']} / "
                      f"{int(cl['m_vocab_gt4']) + int(cl['m_multi_gt4'])} / {cl['m_long']}")
    else:
        cells_.append("")
    cnt = c.get("cnt")
    if cnt:
        ld = st_ = 0
        for k, v in cnt.items():
            if re.match(r"(drv|k1|k3|k5|k6)_", k) and "/" in v:
                a, b = v.split("/")[:2]               # read / written / line changes / distinct lines
                ld += int(a)
                st_ += int(b)
        cells_.append(f"{ld / n:.1f} + {st_ / n:.1f} ({cnt.get('tier', '?')})")
    else:
        cells_.append("")
    cells_.append(f"{c.get('load0', '?')} -> {c.get('load1', '?')}; {c.get('busy', '')}")
    return "| " + " | ".join(cells_) + " |"


def main():
    hdr = ["cell", "MB/s (uninstr.)", "MB/s (instr.)", "ns/B", "ns/piece", "K1+K3", "K5 1-byte", "K5 static",
           "K5 cache hits", "K5 miss side", "K5 rounds+resid", "K6 short", "K6 long", "driver",
           "pieces % 1B / static / cache / short miss / long", "K5 fit ns: 1B / static / cache / miss / long / round",
           "K6 ns/call short / long", "K6 calls: first / again / unplaced vocab / > 4 ids / long",
           "traffic B/B load + store", "load; busy"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        for c in cells(path):
            print(row(label, c))


if __name__ == "__main__":
    main()
