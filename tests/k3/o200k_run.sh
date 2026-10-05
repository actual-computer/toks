#!/bin/sh
# tests/k3/o200k_run.sh: the o200k K3 c twin against hf tokenizers 0.23.2 (tests/k3/o200k_gen.py feeds
# tests/k3/o200k_check.c), plus the class tables against the onig probes.
#
# usage: tests/k3/o200k_run.sh [JOBS] [N_SHORT] [N_LONG] [EXH] [VOCAB] [REAL_DIR] [SEED]
#   N_SHORT / N_LONG / VOCAB are totals, split over JOBS shards (each shard: a generator piped into its
#   own checker); EXH = exhaustive strings of length 0..EXH over o200k_model.REP; REAL_DIR = real-text
#   files (tests/model/run_o200k_fuzz.py --fetch DIR makes one); SEED (default 1) as the model run's.
#   local smoke:  tests/k3/o200k_run.sh 1 2000 200 2
#   lab host:     tools/remote.sh <host> 'tests/k3/o200k_run.sh 18 3000000 300000 4 200000 build/realtext'
#   kimi (tiktoken's pieces): GENARGS="--variants kimi --kshort N --kexh L --kreal N" tests/k3/o200k_run.sh 16 0 0 -1 0 DIR
set -eu
cd "$(dirname "$0")/../.."
JOBS=${1:-1}
NS=${2:-2000}
NL=${3:-200}
EXH=${4:-2}
VOCAB=${5:-0}
REAL=${6:-}
SEED=${7:-1}
OUT=build/o200k
mkdir -p "$OUT"
rm -f "$OUT"/check.*.log "$OUT"/gen.*.log
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
  -o "$OUT/check" tests/k3/o200k_check.c src/core/classes.c src/core/k3_o200k_c.c src/gen/ucd_flags.c
GEN="uv run -q --python 3.12 --with tokenizers==0.23.2 --with tiktoken==0.14.0 python tests/k3/o200k_gen.py"

"$OUT/check" --classes "$OUT/classes.bin"
$GEN classes "$OUT/classes.bin"

REALARG=""
[ -n "$REAL" ] && REALARG="--real $REAL"
i=1
while [ "$i" -le "$JOBS" ]; do
  ( set +e
    { $GEN stream --shard "$i/$JOBS" --short "$NS" --long "$NL" --exh "$EXH" --vocab "$VOCAB" --seed "$SEED" $REALARG ${GENARGS:-} \
        2> "$OUT/gen.$i.log"; echo "gen exit $?" >> "$OUT/gen.$i.log"; } \
      | "$OUT/check" - > "$OUT/check.$i.log" 2>&1
    echo "exit $?" >> "$OUT/check.$i.log" ) &
  i=$((i + 1))
done
wait
cat "$OUT"/gen.*.log "$OUT"/check.*.log
awk '/^o200k_check:/ { s++; c += $2; b += $(NF-1) } END { printf "o200k_run: %d shards, %d cases, %d mismatches\n", s, c, b }' "$OUT"/check.*.log
! grep -q "^exit [^0]" "$OUT"/check.*.log && ! grep -q "^gen exit [^0]" "$OUT"/gen.*.log
