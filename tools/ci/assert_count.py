#!/usr/bin/env python3
"""tools/ci/assert_count.py: the count rule of the nightly gate (.github/workflows/nightly.yml, docs/ci.md).

A differential that compared nothing has not passed (tests/norm/run.sh passed on zero records for a day before
PR#175). Every runner ends with its count line

    PASS <n> compared <suite> (<detail>)        n > 0 cases / ids / records compared, 0 mismatches
    FAIL <n> compared <suite> (<why>)

and this script fails the job (exit 1) on a FAIL line, on n = 0, on a runner that printed no line, on fewer lines
than the job runs (--expect N), and on a critical target that did not run (tools/ci/fetch_tokenizers.py's CRITICAL,
the list test.yml enforces; for rc_host.sh's parity step, the five critical targets + Llama 3, the files it runs):

    assert_count.py lines LOG... --expect N     the runners' own lines (tests/norm/run.sh, tests/bpe/run.sh)
    assert_count.py parity build/rc             tools/release/rc_host.sh's parity step: every target it ran
                                                (steps.txt) plus the critical ones, every tier it ran (host.txt):
                                                tests/parity/shards.sh's line per target / part / tier, and
                                                run_kimi.py's line per shard for kimik3
    assert_count.py wheels REPORT.json          python/tests' TOKS_PY_REPORT (pytest prints no count line): the
                                                parity interpreter's counts per tokenizer and ledger target

With GITHUB_STEP_SUMMARY set (Actions) the lines are appended to the job's summary page.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_tokenizers import CRITICAL  # noqa: E402  the critical targets, SPEC §1.1 (d): one list, test.yml's

CRITICAL_PARITY = ("glm53", "kimik3", "o200k", "qwen38", "gemma4", "llama3")        # rc_host.sh's target names
CRITICAL_LEDGER = tuple(t for t in CRITICAL if t != "kimi-k3")  # python/tests skips kimi (tiktoken's: parity job)
PY_NAMES = ("gpt2", "llama3", "glm53", "qwen38", "gemma4", "o200k")
LINE = re.compile(r"^(PASS|FAIL) (\d+) compared (.+)$", re.M)

out = []


def note(ok, n, suite, detail):
    if ok and n <= 0:
        detail += "; nothing compared, and a pass on nothing is a failure"
    out.append((bool(ok) and n > 0, n, suite, detail))


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def lines(logs):
    for p in logs:
        found = LINE.findall(text(p))
        if not found:
            note(False, 0, os.path.basename(p), "no count line")
        for verdict, n, rest in found:
            note(verdict == "PASS", int(n), rest, os.path.basename(p))


def parity(rc):
    steps = dict(re.findall(r"^STEP parity-(\S+) status=(\d+)", text(f"{rc}/steps.txt"), re.M))
    tiers = re.search(r"^tiers (.*?) jobs ", text(f"{rc}/host.txt"), re.M)
    tiers = tiers.group(1).split() if tiers else []
    if not tiers:
        note(False, 0, "parity", f"no tiers in {rc}/host.txt (rc_host.sh did not start)")
    for t in [*steps, *(c for c in CRITICAL_PARITY if c not in steps)]:
        if steps.get(t) != "0":
            note(False, 0, f"parity {t}", f"rc_host.sh step parity-{t}: " +
                 ("never ran (a critical target)" if t not in steps else f"exit {steps[t]}, {rc}/parity-{t}.log"))
        for tier in tiers:
            if t == "kimik3":                       # a run_kimi.py line per shard; rc_host.sh merges the shards
                d = f"{rc}/parity/kimik3-{tier}"
                r = load(f"{d}/report.json") or {}
                got = [g[-1] for g in (LINE.findall(text(p)) for p in sorted(glob.glob(f"{d}/shard-*.log"))) if g]
                shards, miss, s = r.get("shards", 0), r.get("missing_shards", ["no report.json"]), r.get("totals", {})
                note(shards > 0 and not miss and len(got) == shards and all(v == "PASS" for v, _, _ in got),
                     sum(int(n) for _, n, _ in got), f"parity kimik3 {tier}",
                     f"{len(got)} of {shards} shards' run_kimi.py lines, missing shards {miss}: {s.get('texts')} "
                     f"texts x 3 modes + 2 decodes vs transformers + tiktoken, {s.get('ids')} ids, "
                     f"{s.get('mismatches')} mismatches, {s.get('ref_errors')} reference errors")
                continue
            for part in ("ids", "pieces"):         # a shards.sh line per target / part / tier
                suite, f = f"parity {t} {part} {tier}", f"{rc}/parity/{t}-{part}-{tier}.txt"
                got = LINE.findall(text(f))
                if not got:
                    note(False, 0, suite, f"no count line in {f}")
                    continue
                verdict, n, rest = got[-1]
                note(verdict == "PASS", int(n), suite, rest)


def wheels(path):
    r = load(path)
    if r is None:
        note(False, 0, "wheels", f"no {path} (the parity interpreter's tests did not run)")
        return
    for name in PY_NAMES:
        for part in ("generated", "stream", "cases"):
            c = r.get(f"{name}/{part}")
            if c is None:
                note(False, 0, f"wheel {name} {part}", "absent: skipped or not run")
                continue
            n = sum(d["cases"] for d in c.values())
            ok = sum(d["equal"] + d.get("spec_e_id", 0) + d.get("spec_limit", 0) for d in c.values()) == n
            note(ok, n, f"wheel {name} {part}", "vs hf 0.23.2 through the Python API: " +
                 ", ".join(f"{op} {d['cases']}" for op, d in sorted(c.items())))
        v = r.get(f"{name}/vocab")
        note(v is not None, (v or {}).get("ids", 0), f"wheel {name} vocab",
             f"id_to_token over every id, {(v or {}).get('tokens', 0)} vocab entries both ways" if v else "absent")
    for key in sorted(k for k in r if k.startswith("target/")):
        c = r[key]
        note(c["diffs"] == 0, c["encodes"], f"wheel {key}", f"{c['texts']} texts x 3 modes x 2, {c['diffs']} diffs")
    for t in CRITICAL_LEDGER:
        if f"target/{t}" not in r:
            note(False, 0, f"wheel target/{t}", "absent: a critical target skipped")


def main():
    a = sys.argv[1:]
    expect = 0
    if "--expect" in a:
        i = a.index("--expect")
        expect = int(a[i + 1])
        del a[i:i + 2]
    if len(a) < 2 or a[0] not in ("lines", "parity", "wheels"):
        raise SystemExit(__doc__)
    kind, args = a[0], a[1:]
    {"lines": lambda: lines(args), "parity": lambda: parity(args[0]), "wheels": lambda: wheels(args[0])}[kind]()
    if len(out) < expect:
        note(False, 0, kind, f"{len(out)} count lines, the job runs {expect}")
    rows = [f"{'PASS' if ok else 'FAIL'} {n} compared {suite} ({detail})" for ok, n, suite, detail in out]
    bad = sum(1 for ok, *_ in out if not ok)
    tail = f"assert_count {kind}: {len(out)} suites, {sum(n for _, n, *_ in out)} compared, {bad} failed"
    print("\n".join(rows + [tail]))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### {tail}\n\n```\n" + "\n".join(rows) + "\n```\n")
    sys.exit(1 if bad or not out else 0)


if __name__ == "__main__":
    main()
