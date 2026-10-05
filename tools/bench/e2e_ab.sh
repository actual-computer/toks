#!/bin/sh
# tools/bench/e2e_ab.sh: a paired comparison of two tiers of one build, abba per cell (A B B A, ROUNDS times),
# so slow drifts in a shared host's load cancel. Uses tools/bench/e2e.c (best of REPS per state inside each run; its
# pass after the other corpora, as tools/bench/e2e.sh).
#
#   tools/bench/e2e_ab.sh <tierA> <tierB> [pin]       e.g. BUILD_DIR=build/x86-k35 tools/bench/e2e_ab.sh avx2 avx512 "taskset -c 20"
#
# env: BUILD_DIR, TOKS_LIST, CORPORA, CHUNKS, REPS (5), ROUNDS (2), TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers),
#      TOKS_KIMI_DIR, TOKS_BENCH_TEXT (build/text): the cells are e2e.sh's (same corpora, same files)
# prints per cell the pass and cold seconds of every run and B / A of the medians (> 1: B slower). Exactness is
# e2e.sh's job; this script only times. tools/release/rc_host.sh's rent step re-measures with it every cell where a
# separate asm run looked slower than the scalar run.
set -e
A=$1
B=$2
PIN="$3"
. tools/bench/common.sh
TOKS_LIST=${TOKS_LIST:-"gpt2 llama3 glm53"}
CORPORA=${CORPORA:-"en code ml"}
CHUNKS=${CHUNKS:-"0 4096 256"}
REPS=${REPS:-5}
ROUNDS=${ROUNDS:-2}
CC=${CC:-clang}
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }
make -j8 lib >/dev/null
BD=$(mkvar BUILD_DIR)
case $(mkvar ISA) in x86_64) NATIVE=-march=native ;; *) NATIVE=-mcpu=native ;; esac
$CC -std=c17 -O3 $NATIVE -Wall -Wextra -Werror $(mkvar CPPFLAGS) -o build/e2e-ab tools/bench/e2e.c "$BD/libtoks.a" -lpthread
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm) AB $A $B PIN '$PIN' REPS $REPS ROUNDS $ROUNDS GIT ${GIT_SHA:-unknown}"
echo "UPTIME $(uptime)"
for tk in $TOKS_LIST; do
    for corp in $CORPORA; do
        F=$(files $corp)
        W=$(others $corp)
        for ch in $CHUNKS; do
            pa=""; pb=""; ca=""; cb=""; l0=$(load1)
            r=0
            while [ $r -lt "$ROUNDS" ]; do
                for t in $A $B $B $A; do
                    # shellcheck disable=SC2086
                    line=$(E2E_WARM_ON="$W" TOKS_TIER=$t $PIN ./build/e2e-ab "$(tpath "$tk")" "$ch" "$REPS" $F | grep '^E2E')
                    p=$(echo "$line" | sed 's/.* pass_s=\([0-9.]*\).*/\1/')
                    c=$(echo "$line" | sed 's/.* cold_s=\([0-9.]*\).*/\1/')
                    if [ "$t" = "$A" ]; then pa="$pa $p"; ca="$ca $c"; else pb="$pb $p"; cb="$cb $c"; fi
                done
                r=$((r + 1))
            done
            med() { echo "$@" | tr ' ' '\n' | sed '/^$/d' | sort -g | awk '{v[NR]=$1} END {print (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2}'; }
            echo "AB tk=$tk corp=$corp chunk=$ch A=$A B=$B load=$l0->$(load1) passA=$(echo $pa | tr ' ' ,) passB=$(echo $pb | tr ' ' ,)" \
                 "coldA=$(echo $ca | tr ' ' ,) coldB=$(echo $cb | tr ' ' ,)" \
                 "pass_B_over_A=$(echo "$(med $pb) $(med $pa)" | awk '{printf "%.4f", $1 / $2}')" \
                 "cold_B_over_A=$(echo "$(med $cb) $(med $ca)" | awk '{printf "%.4f", $1 / $2}')"
        done
    done
done
echo "UPTIME $(uptime)"
