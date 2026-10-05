#!/usr/bin/env bash
# tools/ci/test.sh [tier ...]: the CI exactness gate on this host (docs/ci.md). One `make -j all asmcheck` builds
# what every tier shares, then `make -j test` once per tier, one tier after the other (default: native, i.e.
# TOKS_TIER unset, the tier the cpu and the build pick; then scalar, the c twins): make runs a tier's test binaries
# side by side on every cpu, and the tiers do not overlap, so the other tier's load never reaches test_stall's
# timing checks. Each tier's output is build/ci/<tier>.log, printed whole once it is done; then tools/ci/suites.py
# over the logs (per-binary counts and SKIPs; a critical target skipped fails). The tokenizer files must be in
# place first (tools/ci/fetch_tokenizers.py). The same command reproduces a CI job on a benchmark machine
# (docs/machines.md), e.g. a gb10's A725 cores:
#   tools/remote.sh <host> 'taskset -c 0-4,10-14 tools/ci/test.sh'
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
tiers=("$@")
[ ${#tiers[@]} -gt 0 ] || tiers=(native scalar)
j=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)    # nproc: the cpus a taskset leaves on a shared machine
gh=${GITHUB_ACTIONS:-}
mkdir -p build/ci
cpu=$( (sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 -E '^(model name|CPU part)' /proc/cpuinfo) | sed 's/.*: *//')
echo "test.sh: $(uname -srm), $j cpus ($cpu), $(${CC:-clang} --version | head -1), $(python3 --version 2>&1)"
[ -z "$gh" ] || echo "::group::make -j$j all asmcheck (the build every tier uses)"
make -j"$j" all asmcheck 2>&1 | tee build/ci/build.log
rc=$?
[ -z "$gh" ] || echo "::endgroup::"
if [ "$rc" != 0 ]; then
    echo "test.sh: make all asmcheck FAILED (exit $rc: a compile error, a .S that does not assemble for one of" \
         "mach-o / elf / coff, or the abi lint); no tier ran. build/ci/build.log"
    exit 1
fi
status=0
logs=()
for t in "${tiers[@]}"; do
    if [ "$t" = native ]; then
        (unset TOKS_TIER; exec make -j"$j" test) > "build/ci/$t.log" 2>&1
    else
        (export TOKS_TIER="$t"; exec make -j"$j" test) > "build/ci/$t.log" 2>&1
    fi
    rc=$?
    logs+=("build/ci/$t.log")
    [ -z "$gh" ] || echo "::group::make -j$j test ($t)"
    cat "build/ci/$t.log"
    [ -z "$gh" ] || echo "::endgroup::"
    [ "$rc" = 0 ] || status=1
    echo "test.sh: make test ($t, TOKS_TIER=$([ "$t" = native ] && echo unset || echo "$t")):" \
         "$([ "$rc" = 0 ] && echo passed || echo "FAILED, exit $rc"), done at ${SECONDS} s"
done
python3 tools/ci/suites.py "${logs[@]}" || status=1
exit $status
