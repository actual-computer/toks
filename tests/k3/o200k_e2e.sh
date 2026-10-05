#!/bin/sh
# tests/k3/o200k_e2e.sh: a tokenizer.json with the o200k pattern (gpt-oss) through toks' real load path
# (toks_config_parse + toks_compile + toks_bpe_build), then text -> ids (K3 o200k + K5, the driver's
# chunking) against hf tokenizers 0.23.2 (tests/k3/o200k_gen.py e2e feeds tests/k3/o200k_e2e.c).
#
# usage: tests/k3/o200k_e2e.sh TOKENIZER [JOBS] [N_SHORT] [N_LONG] [VOCAB] [REAL_DIR] [SEED]
#   local smoke:  tests/k3/o200k_e2e.sh ~/.cache/toks/tokenizers/o200k 1 300 30
#   lab host:     tools/remote.sh <host> 'tests/k3/o200k_e2e.sh ~/.cache/toks/tokenizers/o200k 16 1000000 100000 100000 build/realtext'
set -eu
cd "$(dirname "$0")/../.."
TOK=$1
JOBS=${2:-1}
NS=${3:-300}
NL=${4:-30}
VOCAB=${5:-0}
REAL=${6:-}
SEED=${7:-1}
OUT=build/o200k-e2e
mkdir -p "$OUT"
rm -f "$OUT"/e2e.*.log "$OUT"/gen.*.log
BUILD_DIR=$OUT make -s -j8 lib > /dev/null
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
  -o "$OUT/e2e" tests/k3/o200k_e2e.c "$OUT/libtoks.a"
GEN="uv run -q --python 3.12 --with tokenizers==0.23.2 python tests/k3/o200k_gen.py"
export TOKENIZERS_PARALLELISM=false
REALARG=""
[ -n "$REAL" ] && REALARG="--real $REAL"
i=1
while [ "$i" -le "$JOBS" ]; do
  ( set +e
    { $GEN e2e "$TOK" --shard "$i/$JOBS" --short "$NS" --long "$NL" --exh 2 --vocab "$VOCAB" --seed "$SEED" \
        $REALARG 2> "$OUT/gen.$i.log"; echo "gen exit $?" >> "$OUT/gen.$i.log"; } \
      | "$OUT/e2e" "$TOK" > "$OUT/e2e.$i.log" 2>&1
    echo "exit $?" >> "$OUT/e2e.$i.log" ) &
  i=$((i + 1))
done
wait
cat "$OUT"/gen.*.log "$OUT"/e2e.*.log
awk '/^o200k_e2e:/ { s++; c += $2; b += $(NF-1) } END { printf "o200k_e2e.sh: %d shards, %d texts, %d mismatches\n", s, c, b }' "$OUT"/e2e.*.log
! grep -q "^exit [^0]" "$OUT"/e2e.*.log && ! grep -q "^gen exit [^0]" "$OUT"/gen.*.log
