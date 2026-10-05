#!/bin/sh
# tests/spm/run.sh: the c twin of sentencepiece-style bpe against hf 0.23.2 on a pinned tokenizer
# (tests/spm/pins.json), WORKERS parallel gen.py | check pipelines with seeds SEED .. SEED + WORKERS - 1.
#
#   tests/spm/run.sh NAME N_PER_WORKER [SEED]           e.g. tests/spm/run.sh gemma4 2000      (local smoke)
#   tools/remote.sh <host> 'WORKERS=20 tests/spm/run.sh gemma4 50000'                       (a lab host)
set -eu
cd "$(dirname "$0")/../.."
name=$1
n=$2
seed=${3:-1}
workers=${WORKERS:-4}
BUILD_DIR=build/spm-check make -s -j8 lib >/dev/null
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
  -o build/spm-check/check tests/spm/check.c build/spm-check/libtoks.a
fetched=$(uv run --python 3.12 python tests/spm/fetch.py "$name")
path=$(echo "$fetched" | awk '{print $NF}')
sha=$(echo "$fetched" | awk '{print $3}')
mkdir -p build/spm-check/out && rm -f build/spm-check/out/"$name".*.log
i=0
while [ "$i" -lt "$workers" ]; do
  s=$((seed + i))
  ( set +e                                           # the pipeline's status is recorded, not fatal
    uv run --python 3.12 --with tokenizers==0.23.2 python tests/spm/gen.py "$name" "$n" "$s" 2>/dev/null \
      | build/spm-check/check "$path" > "build/spm-check/out/$name.$s.log" 2>&1
    echo "exit $?" >> "build/spm-check/out/$name.$s.log" ) &
  i=$((i + 1))
done
wait
cat build/spm-check/out/"$name".*.log | grep -E "RESULT|MISMATCH|exit [^0]|LOAD|bad record" || true
fail=$(cat build/spm-check/out/"$name".*.log | grep -c "exit [^0]" || true)
gaps=$(cat build/spm-check/out/"$name".*.log | sed -n "s/^RESULT [^:]*: gaps \([0-9]*\) .*/\1/p" | awk '{s+=$1} END {print s+0}')
echo "TOTAL $name sha256 $sha: $workers workers, $gaps gaps, failing pipelines $fail"
[ "$fail" = 0 ]
