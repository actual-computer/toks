#!/bin/sh
# tests/k3/dsv3_run.sh: the dsv3 K3 c twin against hf tokenizers 0.23.2 (tests/k3/dsv3_gen.py feeds
# tests/k3/dsv3_check.c), plus the class tables against the onig probes.
#
# usage: tests/k3/dsv3_run.sh [JOBS] [N_SHORT] [N_LONG] [EXH] [EXH_SMALL] [VOCAB] [BAD] [REAL_DIR] [SEED]
#   N_SHORT / N_LONG / VOCAB / BAD are totals, split over JOBS shards (each shard: a generator piped into its own
#   checker); EXH / EXH_SMALL = exhaustive strings of length 0..EXH over dsv3_model.REP (0..EXH_SMALL over
#   REP_SMALL); BAD = ill-formed byte strings with the python model's pieces; REAL_DIR = real-text files
#   (tests/model/run_dsv3_fuzz.py --fetch DIR makes one); SEED (default 1).
#   The tokenizer: TOKS_DSV3_TOKENIZER (default ~/.cache/toks/tokenizers/dsv3).
#   local smoke:  tests/k3/dsv3_run.sh 1 2000 100 2
#   lab host:     tools/remote.sh <host> 'tests/k3/dsv3_run.sh 12 3000000 300000 4 6 300000 300000 build/realtext 20261004'
set -eu
cd "$(dirname "$0")/../.."
JOBS=${1:-1}
NS=${2:-2000}
NL=${3:-200}
EXH=${4:-2}
EXHS=${5:-0}
VOCAB=${6:-0}
BAD=${7:-0}
REAL=${8:-}
SEED=${9:-1}
OUT=build/dsv3
mkdir -p "$OUT"
rm -f "$OUT"/check.*.log "$OUT"/gen.*.log
# the c twin; K3_SCAN=toks_k3_scan_dsv3_<tier> K3_LIB=build/<dir>/libtoks.a checks that tier (its part + the twin)
SRCS="src/core/classes.c src/core/k3_dsv3_c.c src/gen/ucd_flags.c src/gen/han_ranges.c"
[ -n "${K3_SCAN:-}" ] && SRCS="-DK3_SCAN=$K3_SCAN -DK3_CLS_FLAGS $K3_LIB"
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
  -o "$OUT/check" tests/k3/dsv3_check.c $SRCS
GEN="uv run -q --python 3.12 --with tokenizers==0.23.2 python tests/k3/dsv3_gen.py"

"$OUT/check" --classes "$OUT/classes.bin"
$GEN classes "$OUT/classes.bin"

REALARG=""
[ -n "$REAL" ] && REALARG="--real $REAL"
i=1
while [ "$i" -le "$JOBS" ]; do
  ( set +e
    { $GEN stream --shard "$i/$JOBS" --short "$NS" --long "$NL" --exh "$EXH" --exh-small "$EXHS" --vocab "$VOCAB" \
        --bad "$BAD" --seed "$SEED" $REALARG 2> "$OUT/gen.$i.log"; echo "gen exit $?" >> "$OUT/gen.$i.log"; } \
      | "$OUT/check" - > "$OUT/check.$i.log" 2>&1
    echo "exit $?" >> "$OUT/check.$i.log" ) &
  i=$((i + 1))
done
wait
cat "$OUT"/gen.*.log "$OUT"/check.*.log
awk '/^dsv3_check:/ { s++; c += $2; m += substr($4, 2); b += $(NF-1) }
     END { printf "dsv3_run: %d shards, %d cases (%d ill-formed), %d mismatches\n", s, c, m, b }' "$OUT"/check.*.log
! grep -q "^exit [^0]" "$OUT"/check.*.log && ! grep -q "^gen exit [^0]" "$OUT"/gen.*.log
