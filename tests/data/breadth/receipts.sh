#!/bin/sh
# tests/data/breadth/receipts.sh <head>: every receipt of docs/breadth.md §6 in one run on a lab host (at most
# 8 processes at a time), written to build/breadth/receipts.txt:
#   make test; the census probe (probe_census.sh + probe_flips.py against census/coverage.json); e2e parity
#   on the census files that now load (diff_real.sh); the synthetic added-token differential (diff.sh synth);
#   the mutants (mutants.sh). Needs the census cache (~/.cache/toks/census/coverage/files) and uv. On a gb10 host,
#   pinned off the X925 cores (5-9, 15-19: the timing cores):
#   tools/remote.sh <host> 'nohup taskset -c 0-4,10-14 sh tests/data/breadth/receipts.sh <sha> > /dev/null 2>&1 &'
set -u
mkdir -p build/breadth
{
    echo "head $1"; uptime; taskset -p $$ 2>/dev/null || true
    sh tests/data/breadth/fetch_text.sh
    make -j8 test > build/make_test.log 2>&1; echo "make test exit $?"
    grep -E "checks|failures|SKIP" build/make_test.log
    echo "=== census probe"; sh tests/data/breadth/probe_census.sh
    uv run -q python tests/data/breadth/probe_flips.py build/breadth/probe.tsv --coverage census/coverage.json
    echo "=== e2e parity on the files that now load"; sh tests/data/breadth/diff_real.sh 300000
    echo "=== synthetic added-token differential"
    sh tests/data/breadth/diff.sh synth --tokenizers 25000 --texts 400 --seed 3 | tail -12
    echo "=== mutants"; sh tests/data/breadth/mutants.sh
    uptime; echo "receipts done"
} > build/breadth/receipts.txt 2>&1
