#!/bin/sh
# tests/data/breadth/diff.sh <synth|real> [args...]: build the library and the toks driver, then run
# tests/data/breadth/diff.py in $SHARDS parallel shards (default 8: the shared-host limit, maintainer doctrine) and
# merge the shard reports. On a lab host from the repository root:
#   tools/remote.sh <host> 'sh tests/data/breadth/diff.sh synth --tokenizers 2000 --texts 300 --seed 1'
#   tools/remote.sh <host> 'sh tests/data/breadth/diff.sh real FILE --texts 400000 --seed 1 --corpus ...'
set -u
cmd=$1; shift
n=${SHARDS:-8}
isa=$(uname -m | sed 's/aarch64/arm64/')
os=$([ "$(uname)" = Darwin ] && echo macos || echo linux)
make -s -j8 lib >/dev/null || exit 2
mkdir -p build/breadth
clang -std=c17 -O2 -Iinclude tests/driver/toks_driver.c "build/$os-$isa/libtoks.a" -o build/breadth/toks_driver || exit 2
uv run -q --with tokenizers==0.23.2 python -c "import tokenizers" || exit 2
tag=$(date +%s)
uptime > "build/breadth/$tag.uptime"
i=0
while [ "$i" -lt "$n" ]; do
  uv run -q --with tokenizers==0.23.2 python tests/data/breadth/diff.py --driver build/breadth/toks_driver \
      --out "build/breadth/$tag.$i.json" "$cmd" "$@" --shard "$i/$n" > "build/breadth/$tag.$i.log" 2>&1 &
  i=$((i + 1))
done
wait
uptime >> "build/breadth/$tag.uptime"
uv run -q python - "build/breadth/$tag" "$n" <<'EOF'
import json, sys
base, n = sys.argv[1], int(sys.argv[2])
tot = {"cases": 0, "decodes": 0, "loaded": 0, "refused_as_hf_can_panic": 0, "hf_panics": 0, "n_diffs": 0, "seconds": 0.0}
diffs, missing = [], []
for i in range(n):
    try:
        r = json.load(open(f"{base}.{i}.json"))
    except (OSError, ValueError):
        missing.append(i)
        continue
    for k in ("cases", "decodes", "loaded", "refused_as_hf_can_panic", "hf_panics", "n_diffs"):
        tot[k] += r[k]
    tot["seconds"] = max(tot["seconds"], r["seconds"])
    diffs += r["diffs"]
tot["missing_shards"] = missing
tot["uptime"] = open(base + ".uptime").read().strip().split("\n")
print(json.dumps(tot, indent=1))
for d in diffs[:8]:
    print("DIFF", json.dumps(d, ensure_ascii=True)[:700])
sys.exit(1 if tot["n_diffs"] or missing else 0)
EOF
