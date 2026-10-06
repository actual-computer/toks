#!/usr/bin/env python3
"""tests/proof/ledger.py: every Eva alarm of the proof package is in one of SPEC §14.1's two categories (T4).

    python3 tests/proof/ledger.py [build/proof/eva | <job>.<model>.csv ...]

Reads every <job>.<model>.csv Frama-C's -report-csv wrote (tests/proof/eva.sh: the reports of the jobs it ran; by
hand: a directory's, or the files named) and every open property in it: a status
other than Valid, Considered valid (an assumption: a libc or platform contract, docs/proof.md) or Dead (unreachable
in the analysis). Each must match a class of tests/proof/alarms.md, a line

    <id> | <jobs> | <file> | <function or *> | <property kind> | <property regex> | proven: <how> / review: <argument>

(<jobs> a glob on the report's <job>.<model> name, e.g. json-* or *: a class whose argument holds only for some
callers names the jobs it was checked for; the regex is matched against the property text with Frama-C's tmp_<n>
names folded to tmp). An open property no class matches fails the ledger; a class that matches nothing is printed as stale. Prints per job the open properties and per
class the count. Exit 1 on any unmatched property or an unreadable report.
"""
import csv
import fnmatch
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLOSED = {"Valid", "Considered valid", "Dead"}


def load_classes(path):
    classes = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not re.match(r"^[A-Z]+[0-9]+ \|", line):
            continue
        f = [x.strip() for x in line.rstrip("\n").split(" | ")]
        if len(f) != 7 or not re.match(r"^(proven|review): ", f[6]):
            sys.exit(f"ledger: {path}:{n}: want '<id> | <jobs> | <file> | <function> | <kind> | <regex> | "
                     "proven: / review: ...'")
        classes.append({"id": f[0], "jobs": f[1], "file": f[2], "func": f[3], "kind": f[4], "rx": re.compile(f[5]),
                        "why": f[6], "n": 0})
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
        n_open = 0
        for r in rows[1:]:
            d, f, line, func, kind, status, prop = r[:7]
            if status in CLOSED or "FRAMAC_SHARE" in d:
                continue
            n_open += 1
            text = re.sub(r"\btmp_[0-9]+\b", "tmp", prop)
            job = os.path.basename(rep)[:-len(".csv")]
            hit = next((c for c in classes if fnmatch.fnmatch(job, c["jobs"]) and c["file"] == f
                        and c["func"] in ("*", func) and c["kind"] == kind and c["rx"].search(text)), None)
            if hit is None:
                print(f"UNCLASSIFIED {os.path.basename(rep)} {f}:{line} {func} {kind} [{status}] {text}")
                bad += 1
            else:
                hit["n"] += 1
        print(f"ledger: {os.path.basename(rep)}: {n_open} open properties")
    for c in classes:
        print(f"  {c['id']:5} {c['n']:5}  {c['jobs']} {c['file']} {c['func']} {c['kind']}: {c['why'][:80]}")
        if c["n"] == 0:
            print(f"  note: class {c['id']} matched nothing in these reports")
    print(f"ledger: {len(reports)} reports, {bad} unclassified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
