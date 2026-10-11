#!/bin/bash
# tests/proof/eva.sh [job ...]: the Frama-C Eva half of the proof package (SPEC §14.1, T4; docs/proof.md). On a lab
# host with tests/proof/install.sh's tools, from the repository root (`make proof` runs it):
#
#   tools/remote.sh <gb10> 'nice -n 10 taskset -c 10-13 bash tests/proof/eva.sh'            the gate's jobs (ALL), both models
#   tools/remote.sh <gb10> 'nice -n 10 taskset -c 10-13 bash tests/proof/eva.sh config-end' one job
#
# A job is one entry of tests/proof/entries.c over the src/ files it reaches, run under both data models SPEC §14.1
# names (lp64: Frama-C's gcc_x86_64 machdep; llp64: msvc_x86_64), at most $JOBS at once, into
# build/proof/eva/<job>.<model>.{log,csv} (the csv is Frama-C's report: every property and its status). Then
# tests/proof/ledger.py over those reports: every open property must be in tests/proof/alarms.md. A job fails when
# Frama-C errs, writes no report, runs past 2 h, or reports a function that never terminates (NON TERMINATING
# FUNCTION in Eva's final states: the properties after a bottom state are Dead, which the ledger would close) or a
# degeneration. Each log starts with the command; each job's line quotes Eva's coverage. Exit 1 when a job fails or
# the ledger does.
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
. "${TOKS_PROOF_TOOLS:-$HOME/toks-ci/opam}/env.sh"
out=${EVA_OUT:-build/proof/eva}
mkdir -p "$out"
JOBS=${JOBS:-4}
MODELS=${MODELS:-"lp64 llp64"}

# what toks_config_parse reaches (llvm-nm -u of config.o, closed over its callees)
CONFIG_FILES="src/core/json.c src/core/config.c src/core/alloc.c src/core/precompiled.c src/core/segment.c \
src/core/norm.c src/core/gen.c src/core/api.c src/gen/ucd_flags.c src/gen/norm_nfc.c src/gen/bert_tables.c"

# job -> main|defines|precision|files (src/ paths; entries.c and plat.c are always added)
job_def() {
  case $1 in
    json-end)     echo "eva_json|-DEVA_PLACE=0|5|src/core/json.c src/core/gen.c" ;;
    json-start)   echo "eva_json|-DEVA_PLACE=1|5|src/core/json.c src/core/gen.c" ;;
    config-end)   echo "eva_config|-DEVA_PLACE=0|3|$CONFIG_FILES" ;;
    config-start) echo "eva_config|-DEVA_PLACE=1|3|$CONFIG_FILES" ;;
    *) echo "eva.sh: unknown job $1" >&2; exit 2 ;;
  esac
}
# the gate (make proof): every job whose report the ledger closes. config-end / config-start run by name and are not
# yet in it (docs/proof.md 3: Eva stops at gen.c's recursion, cf_audit R4's debt)
ALL="json-end json-start"

machdep() { case $1 in lp64) echo gcc_x86_64 ;; llp64) echo msvc_x86_64 ;; esac; }

run_one() {   # job model
  def=$(job_def "$1"); main=${def%%|*}; rest=${def#*|}; defs=${rest%%|*}; rest=${rest#*|}
  prec=${rest%%|*}; files=${rest#*|}
  base="$out/$1.$2"
  rm -f "$base.csv" "$base.log"
  start=$(date +%s)
  # shellcheck disable=SC2086
  set -- timeout 7200 frama-c -machdep "$(machdep "$2")" \
    -cpp-extra-args="-Iinclude -Isrc/core -Isrc/platform -Itests/proof $defs -include prelude.h" \
    -warn-invalid-pointer -warn-signed-downcast \
    $files tests/proof/entries.c tests/proof/plat.c \
    -main "$main" -eva -eva-precision "$prec" ${EVA_EXTRA:-} \
    -then -report-csv "$base.csv"
  { printf 'eva.sh:'; printf ' %q' "$@"; printf '\neva.sh: EVA_EXTRA=%q\n' "${EVA_EXTRA:-}"; } > "$base.log"
  "$@" >> "$base.log" 2>&1 || echo "FAILED (frama-c exit $?: 124 is the 2 h timeout)" >> "$base.log"
  if grep -q 'NON TERMINATING FUNCTION' "$base.log"; then echo "FAILED (a function that never terminates)" >> "$base.log"; fi
  if grep -qi 'degeneration' "$base.log"; then echo "FAILED (a degeneration)" >> "$base.log"; fi
  cov=$(grep -o '[0-9]* functions analyzed (out of [0-9]*): [0-9]*% coverage\|[0-9]* statements reached (out of [0-9]*): [0-9]*% coverage' "$base.log" | tr '\n' ';' | sed 's/;$//')
  if grep -q '^FAILED' "$base.log" || [ ! -s "$base.csv" ]; then
    echo "${base##*/}: $(( $(date +%s) - start )) s, FAILED: $(grep -m1 -A2 'User Error\|^FAILED' "$base.log" | tr '\n' ' ' | cut -c1-200)"
    return
  fi
  v=$(awk -F'\t' 'NR > 1 && $6 == "Valid"' "$base.csv" | wc -l)
  n=$(awk -F'\t' 'NR > 1 && $6 != "Valid" && $6 != "Considered valid" && $6 != "Dead"' "$base.csv" | wc -l)
  echo "${base##*/}: $(( $(date +%s) - start )) s, $v valid, $n open properties; $cov"
}

cat "${TOKS_PROOF_TOOLS:-$HOME/toks-ci/opam}/versions.txt"     # the receipt's versions (tests/proof/install.sh)
echo "frama-c's preprocessor: $(gcc --version | head -1)"
jobs=${*:-$ALL}
for j in $jobs; do job_def "$j" > /dev/null; done
for j in $jobs; do
  for m in $MODELS; do
    while [ "$(jobs -p | wc -l)" -ge "$JOBS" ]; do sleep 1; done
    run_one "$j" "$m" &
  done
done
wait
fail=0
reps=""
for j in $jobs; do
  for m in $MODELS; do                     # a job that did not finish is a failed gate, not a missing report
    if [ ! -s "$out/$j.$m.csv" ] || grep -q '^FAILED' "$out/$j.$m.log"; then echo "eva.sh: $j.$m did not finish"; fail=1
    else reps="$reps $out/$j.$m.csv"; fi
  done
done
# every open property of the reports just written in tests/proof/alarms.md
# shellcheck disable=SC2086
if [ -n "$reps" ]; then python3 tests/proof/ledger.py $reps || fail=1; fi
exit "$fail"
