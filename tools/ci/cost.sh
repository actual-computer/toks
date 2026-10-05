#!/bin/sh
# tools/ci/cost.sh <job> <units a minute>: what this CI job has cost so far in x64 2-vcpu minutes (docs/ci.md,
# Minutes): the seconds since the workflow's start step stored T0 (the runner's own setup, a few seconds, is not in
# it), rounded up to whole minutes as billed, times the runner's rate. Printed and added to the job summary, so a
# drift in what CI costs shows on every run.
set -eu
s=$(($(date +%s) - ${T0:?the start step sets T0}))
line=$(awk -v s="$s" -v r="$2" -v j="$1" 'BEGIN { m = int((s + 59) / 60);
    printf "cost: %s: %d s, %d billed min x %s = %g x64 2-vcpu minutes\n", j, s, m, r, m * r }')
echo "$line"
[ -z "${GITHUB_STEP_SUMMARY:-}" ] || echo "$line" >> "$GITHUB_STEP_SUMMARY"
