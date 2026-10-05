#!/usr/bin/env python3
"""tools/ci/suites.py: what a `make test` log ran, counted per test binary, so a vacuous pass cannot hide in CI.

    python3 tools/ci/suites.py build/ci/native.log build/ci/scalar.log

Each log is one `make test` or one tools\\win\\test.cmd (one tier; the file's stem labels its column; the first
"== " header tells which: make's carry the binary's path, test.cmd's do not). Per binary (every tests/c/*.c, and
what else test.cmd runs: test_e2e_dll, the K3 tier exes): the count it printed (the largest number before checks /
cases / compared / targets / vectors), its SKIP lines, and the tier line test_tier prints. Exit 1 when a binary did
not run, when a binary's counts are all 0, when a SKIP names a critical target's file (tools/ci/fetch_tokenizers.py
CRITICAL, SPEC §1.1 (d)), when the build has an asm tier but the cpu binds none of them (test_tier's "tier auto =
scalar (...; built: avx2)": no asm kernel ran), when a make test log lacks the asmcheck / abi lint / cf_audit pass
lines or a size budget line, or has a budget over its limit, or when a test.cmd log did not end FAIL=0 or has a
"-- FAIL" line (the Windows job has no asmcheck or size step: test.yml's jobs run them, coff objects included). A
binary that prints no count is listed (no count), not failed. docs/ci.md.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_tokenizers  # noqa: E402  (CRITICAL and the ledger's critical files)

ROOT = fetch_tokenizers.ROOT
HEAD = re.compile(r"^== (\S*/tests/)?(\w+)(?: .*)?$")     # make test: == <dir>/tests/<binary>; test.cmd: == <binary> [arg]
WINEND = re.compile(r"^test\.cmd: FAIL=\d+$")             # tools\win\test.cmd's last line
WINFAIL = re.compile(r"^-- FAIL ")                         # test.cmd: a binary's nonzero exit (or a per-suite kill)
COUNT = re.compile(r"(\d+) (?:checks|cases|calls compared|targets|vectors)\b")
SKIP = re.compile(r"\bSKIP\b|\bskip\b|(?<!\d )\bskipped\b|\bnot present\b")
TIER = re.compile(r"^tier auto = (\S+) \(cpu features \S+; built:([a-z0-9 ]*)\)")   # test_tier's selection line
BUDGET = re.compile(r"^\s+\S.*\s\d+ /\s+\d+ (?:lines|bytes)\b")   # tools/size.sh: "  <what>  <n> / <budget> <unit> ok"
PRELUDE = (("asmcheck", re.compile(r"^asmcheck: \d+ objects .* assemble$")),
           ("abi lint", re.compile(r"^asm_regs_audit: .* 0 violations$")),
           ("cf_audit", re.compile(r"^cf_audit: .* 0 violations, \d+ allowed$")),   # SPEC §9 on the objects
           ("size", BUDGET))


def critical_names():
    table, _ = fetch_tokenizers.pin_table()
    crit = fetch_tokenizers.critical_files(table, fetch_tokenizers.ledger())
    names = {n for _, n in crit} | set(crit.values()) | {"kimik3"}
    return re.compile(r"(?<![A-Za-z0-9_-])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                      + r")(?![A-Za-z0-9_-])")


def parse(path):
    """{binary: {"count": int | None, "skips": [line], "tier": str}}, the prelude lines, the over-budget lines, and for
    a test.cmd log its own lines (the last "test.cmd: FAIL=" and every "-- FAIL"), None for a make test log."""
    out, cur, pre, oversize, win = {}, None, [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = HEAD.match(line)
            if m:
                if cur is None and m.group(1) is None:       # the first header decides the log's kind
                    win = {"end": "", "fails": []}
                cur = out.setdefault(m.group(2), {"count": None, "skips": [], "tier": ""})
                continue
            if win is not None and WINEND.match(line):
                win["end"] = line
                continue
            if win is not None and WINFAIL.match(line):
                win["fails"].append(line.strip())
                continue
            if cur is None:
                pre.append(line)
                if BUDGET.match(line) and not line.rstrip().endswith(" ok"):
                    oversize.append(line.strip())
                continue
            for n in COUNT.findall(line):
                cur["count"] = max(cur["count"] or 0, int(n))
            if SKIP.search(line):
                cur["skips"].append(line.strip())
            if line.startswith("tier auto = "):
                cur["tier"] = line
    return out, pre, oversize, win


def main():
    logs = sys.argv[1:]
    if not logs:
        sys.exit(__doc__)
    want = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "tests", "c", "*.c")))
    crit = critical_names()
    cols, bad = [], []
    for path in logs:
        label = os.path.splitext(os.path.basename(path))[0]
        runs, pre, oversize, win = parse(path) if os.path.exists(path) else ({}, [], [], None)
        cols.append((label, runs))
        if win is None:
            for name, rx in PRELUDE:
                if not any(rx.match(p) for p in pre):
                    bad.append(f"{label}: no {name} line (asmcheck / abi lint / cf_audit pass line, size budget lines)")
        else:
            if win["end"] != "test.cmd: FAIL=0":
                bad.append(f"{label}: test.cmd ended " + (repr(win["end"]) if win["end"] else "without its FAIL= line"))
            bad += [f"{label}: {x}" for x in win["fails"]]
        bad += [f"{label}: size over budget: {o}" for o in oversize]
        for b in want + sorted(set(runs) - set(want)):           # test.cmd also runs test_e2e_dll and the K3 tier exes
            r = runs.get(b)
            if r is None:
                bad.append(f"{label}: {b} did not run")
            elif r["count"] == 0:
                bad.append(f"{label}: {b} reported 0 checks")
            for s in (r["skips"] if r else []):
                if crit.search(s):
                    bad.append(f"{label}: {b} skipped a critical target: {s}")
        m = TIER.match(runs.get("test_tier", {}).get("tier", ""))
        if m and m.group(2).split() and m.group(1) not in m.group(2).split():   # an asm tier built, none bound
            bad.append(f"{label}: this cpu runs none of the asm tiers built ({m.group(0)}): no asm kernel was tested")

    names = want + sorted({b for _, runs in cols for b in runs} - set(want))
    out = ["suites: per binary, the count it printed (- = none printed), then its SKIPs",
           f"{'binary':20}" + "".join(f" {label:>14}" for label, _ in cols)]
    for b in names:
        cells = []
        for _, runs in cols:
            r = runs.get(b)
            cells.append("NOT RUN" if r is None else "-" if r["count"] is None else str(r["count"]))
        out.append(f"{b:20}" + "".join(f" {c:>14}" for c in cells))
    for label, runs in cols:
        ran = [b for b in names if b in runs]
        skips = [(b, s) for b in ran for s in runs[b]["skips"]]
        tier = next((runs[b]["tier"] for b in ran if runs[b]["tier"]), "no tier line")
        nocount = [b for b in ran if runs[b]["count"] is None]
        forced = "unset" if label == "native" else label
        out.append(f"suites: {label} (TOKS_TIER={forced}): {len(ran)} binaries run ({len(want)} of tests/c), "
                   f"{len(skips)} SKIP lines, {len(nocount)} without a count ({' '.join(nocount) or 'none'}); "
                   f"test_tier's {tier}")
        out += [f"  SKIP {label} {b}: {s}" for b, s in skips]
    out += [f"suites: FAIL {x}" for x in bad]
    if not bad:
        out.append(f"suites: ok ({len(cols)} logs, {len(want)} binaries each, no critical target skipped)")
    print("\n".join(out))
    if os.environ.get("GITHUB_STEP_SUMMARY"):                   # the run page shows the same table
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("```\n" + "\n".join(out) + "\n```\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
