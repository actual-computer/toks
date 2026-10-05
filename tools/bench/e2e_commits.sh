#!/bin/sh
# tools/bench/e2e_commits.sh: before / after throughput of two commits of toks on one pinned core, the same harness
# (this tree's tools/bench/e2e.c, built against each commit's own include/ and libtoks.a), abba per cell so a shared
# host's slow drifts cancel. Prints raw E2E lines tagged with the side; tools/bench/e2e_commits.py makes the table
# (MB/s before -> after per state). Exactness: both sides' ids sha must agree per cell (the table checks). The OTHER
# text of e2e.c's pass and lang-x states is every corpus file outside the measured corpus (as tools/bench/e2e.sh).
# Both sides must carry core.h's toks_scr_memo and the long cache's counters (be5f887, ac78876; master d11f9d1 has
# both): this tree's e2e.c reads them for its CTR line, so an older side does not compile.
#
#   tools/bench/e2e_commits.sh <dirA (before)> <dirB (after)> [pin]
#       e.g. tools/bench/e2e_commits.sh ~/toks-ci/bench-before ~/toks-ci/bench-commits "taskset -c 9"
#
# env: TOKS_LIST, CORPORA (en code cjk), CHUNK (4096; 0 = whole corpus), REPS (3, best of, inside e2e.c), ROUNDS (1),
#      TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers), TOKS_KIMI_DIR (~/.cache/toks/kimik3), TOKS_BENCH_TEXT (build/text)
set -e
A=$1
B=$2
PIN="$3"
HERE=$(pwd)
. tools/bench/common.sh
T=${TOKS_BENCH_TEXT:-$HERE/build/text}   # absolute: each side runs in its own dir
TOKS_LIST=${TOKS_LIST:-"llama3 o200k qwen38 glm53 kimik3 dsv3 gemma4 uni_bgem3 wp-bert-uncased wp-minilm-l6"}
CORPORA=${CORPORA:-"en code cjk"}
CHUNK=${CHUNK:-4096}
REPS=${REPS:-3}
ROUNDS=${ROUNDS:-1}
CC=${CC:-clang}
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }
build() {
    (
        cd "$1"
        make -j8 lib >/dev/null
        bd=$(mkvar BUILD_DIR)
        case $(mkvar ISA) in x86_64) native=-march=native ;; *) native=-mcpu=native ;; esac
        # shellcheck disable=SC2046
        $CC -std=c17 -O3 $native -Wall -Wextra $(mkvar CPPFLAGS) -o build/e2e-commits "$HERE/tools/bench/e2e.c" \
            "$bd/libtoks.a" -lpthread
    )
}
build "$A"
build "$B"
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm) PIN '$PIN' CHUNK $CHUNK REPS $REPS ROUNDS $ROUNDS A=$(rel "$A") B=$(rel "$B")"
echo "UPTIME $(uptime)"
for tk in $TOKS_LIST; do
    P=$(tpath "$tk")
    [ -e "$P" ] || { echo "SKIP tk=$tk (no $(rel "$P"))"; continue; }
    for corp in $CORPORA; do
        F=$(files "$corp")
        W=$(others "$corp")
        r=0
        while [ $r -lt "$ROUNDS" ]; do
            for side in A B B A; do
                d=$A; [ "$side" = B ] && d=$B
                # shellcheck disable=SC2086
                line=$(cd "$d" && E2E_WARM_ON="$W" $PIN ./build/e2e-commits "$P" "$CHUNK" "$REPS" $F | grep '^E2E')
                echo "RUN tk=$tk corp=$corp side=$side load=$(load1) $line"
            done
            r=$((r + 1))
        done
    done
done
echo "UPTIME $(uptime)"
