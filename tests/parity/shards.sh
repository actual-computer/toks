#!/bin/sh
# tests/parity/shards.sh <tokenizer> <cases.jsonl> <toks driver argv> <outdir> [N]
#
# Runs N shards of run.py (case lines i mod N) in parallel against hf tokenizers 0.23.2 and merges
# their reports into <outdir>/report.json (counts summed, diffs concatenated). Records `uptime` before
# and after (shared hosts: maintainer doctrine). Exit 0 iff no shard found a diff.
#
#   tests/parity/shards.sh ~/.cache/toks/tokenizers/gpt2 ~/.cache/toks/cases/gpt2.jsonl \
#       ./build/toks_driver out/gpt2 16
set -u
tok=$1 cases=$2 drv=$3 out=$4 n=${5:-16}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
uptime > "$out/uptime.start"
uv run -q --with tokenizers==0.23.2 python -c "import tokenizers" || exit 2     # warm the env once
i=0
while [ "$i" -lt "$n" ]; do
  uv run -q --with tokenizers==0.23.2 python "$here/run.py" --tokenizer "$tok" --cases "$cases" \
      --toks "$drv" --shard "$i/$n" --out "$out/shard-$i.json" > "$out/shard-$i.log" 2>&1 &
  i=$((i + 1))
done
wait
uptime > "$out/uptime.end"
uv run -q python - "$out" "$n" <<'EOF'
import json, sys
out, n = sys.argv[1], int(sys.argv[2])
tot = {"n_lines": 0, "n_cases": 0, "counts": {}, "unsupported": {}, "spec_e_id": 0, "stream_ref_errors": 0,
       "stream_e_limit": 0,
       "n_diffs": 0, "n_minimized": 0, "elapsed_s": 0.0, "diffs": [], "missing_shards": []}
for i in range(n):
    try:
        r = json.load(open(f"{out}/shard-{i}.json"))
    except (OSError, ValueError):
        tot["missing_shards"].append(i)
        continue
    tot["tokenizer"], tot["cases"] = r["tokenizer"], r["cases"]
    for k in ("n_lines", "n_cases", "spec_e_id", "stream_ref_errors", "stream_e_limit", "n_diffs", "n_minimized"):
        tot[k] += r.get(k, 0)
    tot["elapsed_s"] = max(tot["elapsed_s"], r["elapsed_s"])
    for k, v in r["counts"].items():
        tot["counts"][k] = tot["counts"].get(k, 0) + v
    for k, v in r["unsupported"].items():
        tot["unsupported"][k] = tot["unsupported"].get(k, 0) + v
    tot["diffs"] += r["diffs"]
tot["counts"] = dict(sorted(tot["counts"].items()))
tot["uptime_start"] = open(f"{out}/uptime.start").read().strip()
tot["uptime_end"] = open(f"{out}/uptime.end").read().strip()
tot["n_compared"] = tot["n_cases"] - sum(tot["unsupported"].values()) - tot["spec_e_id"]   # the cases compared
json.dump(tot, open(f"{out}/report.json", "w"), indent=2)
print(json.dumps({k: v for k, v in tot.items() if k != "diffs"}, indent=2))
why = [w for w, bad in ((f"{tot['n_diffs']} diffs", tot["n_diffs"]),
                        (f"{tot['stream_ref_errors']} stream reference errors", tot["stream_ref_errors"]),
                        (f"missing shards {tot['missing_shards']}", tot["missing_shards"]),
                        ("nothing compared", tot["n_compared"] <= 0)) if bad]
suite = f"parity {tot.get('tokenizer', '?').rsplit('/', 1)[-1]} {tot.get('cases', '?').rsplit('/', 1)[-1]} ({n} shards)"
detail = f"{tot['n_cases']} cases, {sum(tot['unsupported'].values())} unsupported, {tot['spec_e_id']} spec E_ID"
print(f"{'FAIL' if why else 'PASS'} {max(tot['n_compared'], 0)} compared {suite} ({'; '.join(why) if why else detail})")
sys.exit(1 if why else 0)
EOF
