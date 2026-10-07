#!/usr/bin/env python3
"""tests/proof/ledger.py: every Eva alarm of the proof package is in one of SPEC §14.1's two categories (T4).

    python3 tests/proof/ledger.py [build/proof/eva | <job>.<model>.csv ...]

Reads every <job>.<model>.csv Frama-C's -report-csv wrote (tests/proof/eva.sh: the reports of the jobs it ran; by
hand: a directory's, or the files named) and every open property in it: a status other than Valid, Considered valid
(an assumption: a libc or platform contract, docs/proof.md) or Dead (unreachable in the analysis). Each must match a
class of tests/proof/alarms.md, a line

    <id> | <jobs> | <file>:<lines> | <function or *> | <property kind> | <property regex> | proven: <how> / review: <argument>

(<jobs> a glob on the report's <job>.<model> name, e.g. json-* or *: a class whose argument holds only for some
callers names the jobs it was checked for; <lines> the source lines the argument was checked at, comma-separated
numbers or a-b ranges, so a new site of the same shape needs its own review; the regex is matched against the
property text with Frama-C's tmp_<n> names folded to tmp). It fails on an open property no class matches, on a class
whose jobs match a report it read and that matched nothing there (a stale class: the code moved or the alarm went),
and on an unreadable report. Properties in Frama-C's own share (its libc) are skipped only when closed. Prints per
report the open properties and per class the count.
"""
import csv
import fnmatch
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLOSED = {"Valid", "Considered valid", "Dead"}


def lines_of(spec, where):
    out = set()
    for part in spec.split(","):
        m = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", part.strip())
        if m is None:
            sys.exit(f"ledger: {where}: lines '{spec}': want numbers or a-b ranges, comma-separated")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.update(range(a, b + 1))
    return out


def load_classes(path):
    classes = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not re.match(r"^[A-Z]+[0-9]+ \|", line):
            continue
        f = [x.strip() for x in line.rstrip("\n").split(" | ")]
        if len(f) != 7 or not re.match(r"^(proven|review): ", f[6]) or ":" not in f[2]:
            sys.exit(f"ledger: {path}:{n}: want '<id> | <jobs> | <file>:<lines> | <function> | <kind> | <regex> | "
                     "proven: / review: ...'")
        file, spec = f[2].split(":", 1)
        classes.append({"id": f[0], "jobs": f[1], "file": file, "lines": lines_of(spec, f"{path}:{n}"), "spec": spec,
                        "func": f[3], "kind": f[4], "rx": re.compile(f[5]), "why": f[6], "n": 0, "seen": 0})
    return classes


def main():
    args = sys.argv[1:] or [os.path.join(ROOT, "build", "proof", "eva")]
    classes = load_classes(os.path.join(ROOT, "tests", "proof", "alarms.md"))
    one_dir = len(args) == 1 and os.path.isdir(args[0])
    reports = sorted(glob.glob(os.path.join(args[0], "*.csv"))) if one_dir else sorted(args)
    if not reports:
        sys.exit(f"ledger: no reports in {args[0]} (tests/proof/eva.sh writes them)")
    bad = 0
    for rep in reports:
        rows = list(csv.reader(open(rep, encoding="utf-8"), delimiter="\t"))
        if not rows or rows[0][:7] != ["directory", "file", "line", "function", "property kind", "status", "property"]:
            print(f"ledger: {rep}: not a Frama-C report"); bad += 1; continue
        job = os.path.basename(rep)[:-len(".csv")]
        for c in classes:                                   # the classes this report is checked against
            if fnmatch.fnmatch(job, c["jobs"]):
                c["seen"] += 1
        n_open = 0
        for r in rows[1:]:
            d, f, line, func, kind, status, prop = r[:7]
            if status in CLOSED:
                continue
            n_open += 1
            text = re.sub(r"\btmp_[0-9]+\b", "tmp", prop)
            ln = int(line) if line.isdigit() else -1
            hit = next((c for c in classes if fnmatch.fnmatch(job, c["jobs"]) and c["file"] == f and ln in c["lines"]
                        and c["func"] in ("*", func) and c["kind"] == kind and c["rx"].search(text)), None)
            if hit is None:
                print(f"UNCLASSIFIED {job} {f}:{line} {func} {kind} [{status}] {text}" +
                      (" (Frama-C's share)" if "FRAMAC_SHARE" in d else ""))
                bad += 1
            else:
                hit["n"] += 1
        print(f"ledger: {job}: {n_open} open properties")
    stale = 0
    for c in classes:
        print(f"  {c['id']:5} {c['n']:5}  {c['jobs']} {c['file']}:{c['spec']} {c['func']} {c['kind']}")
        if c["n"] == 0 and c["seen"] > 0:
            print(f"STALE class {c['id']}: its jobs ({c['jobs']}) match {c['seen']} of these reports and it matched nothing")
            stale += 1
    print(f"ledger: {len(reports)} reports, {bad} unclassified, {stale} stale")
    sys.exit(1 if bad or stale else 0)


if __name__ == "__main__":
    main()
