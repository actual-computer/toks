#!/bin/sh
# tools/bench/e2e_commits.sh: before / after throughput of two commits of toks on one pinned core, the same harness
# (this tree's tools/bench/e2e.c, built against each commit's own include/ and libtoks.a), abba per cell so a shared
# host's slow drifts cancel. Prints raw E2E lines tagged with the side; tools/bench/e2e_commits.py makes the table
# (MB/s before -> after per state). Exactness: both sides' ids sha must agree per cell (the table checks). The OTHER
# text of e2e.c's pass and lang-x states is every corpus file outside the measured corpus (as tools/bench/e2e.sh).
# Both sides must carry core.h's toks_scr_memo and the long cache's counters (be5f887, ac78876; master d11f9d1 has
# both): this tree's e2e.c reads them for its CTR line, so an older side does not compile. The COMMITS line names each
# side's commit (the .toks-rev tools/remote.sh writes into a synced tree, else git in the side's directory, else
# "unknown"); each cell's first A and first B run also print their CTR line (the caches' and the memo's counters: hits),
# tagged like RUN lines.
#
#   tools/bench/e2e_commits.sh <dirA (before)> <dirB (after)> [pin]
#       e.g. tools/bench/e2e_commits.sh ~/toks-ci/bench-before ~/toks-ci/bench-commits "taskset -c 9"
#
# env: TOKS_LIST, CORPORA (en code cjk), CHUNK (4096; 0 = whole corpus), REPS (3, best of, inside e2e.c), ROUNDS (1),
#      TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers), TOKS_KIMI_DIR (~/.cache/toks/kimik3), TOKS_BENCH_TEXT (build/text),
#      ENV_A / ENV_B (assignments set for that side's runs only, e.g. ENV_B=E2E_HUGE=1 with one tree as both sides: a
#      variant of e2e.c's own knobs against the same library; printed in the ENV line)
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
rev() { cat "$1/.toks-rev" 2>/dev/null || git -C "$1" describe --always --dirty --abbrev=12 2>/dev/null || echo unknown; }
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
echo "COMMITS A=$(rev "$A") B=$(rev "$B")"
echo "ENV A='${ENV_A:-}' B='${ENV_B:-}'"
echo "UPTIME $(uptime)"
warm=1                                              # one untimed run a side first: right after the builds, a
for tk in $TOKS_LIST; do                            # binary's first run read slow, up to 25% in some states
    P=$(tpath "$tk")
    [ -e "$P" ] || { echo "SKIP tk=$tk (no $(rel "$P"))"; continue; }
    for corp in $CORPORA; do
        F=$(files "$corp")
        W=$(others "$corp")
        # shellcheck disable=SC2086
        [ $warm -eq 0 ] || for s in A B; do
            d=$A e=${ENV_A:-}; [ $s = B ] && d=$B e=${ENV_B:-}
            (cd "$d" && env $e E2E_WARM_ON="$W" $PIN ./build/e2e-commits "$P" "$CHUNK" 1 $F >/dev/null)
        done
        warm=0
        r=0
        while [ $r -lt "$ROUNDS" ]; do
            n=0
            for side in A B B A; do
                n=$((n + 1))
                d=$A e=${ENV_A:-}; [ "$side" = B ] && d=$B e=${ENV_B:-}
                # shellcheck disable=SC2086
                out=$(cd "$d" && env $e E2E_WARM_ON="$W" $PIN ./build/e2e-commits "$P" "$CHUNK" "$REPS" $F)
                echo "RUN tk=$tk corp=$corp side=$side load=$(load1) $(echo "$out" | grep '^E2E')"
                [ $r -gt 0 ] || [ $n -gt 2 ] || echo "CTR tk=$tk corp=$corp side=$side $(echo "$out" | grep '^CTR' | sed 's/^CTR //')"
            done
            r=$((r + 1))
        done
    done
done
echo "UPTIME $(uptime)"
