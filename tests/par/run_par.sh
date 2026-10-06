#!/bin/sh
# tests/par/run_par.sh: the toks_par receipts on a lab host (bench_par.c): every cell pinned to the first k cpus of
# CPUS (fast cores first), the 1-minute load, every pinned cpu's busy share (/proc/stat) and the five busiest
# processes (TOP: share, cpu, comm; no user column) recorded around it. The HOST line names this machine by its
# chipset key (TOKS_HOST_KEY, docs/machines.md; default uname -m); paths under $HOME print as $HOME/...
#
#   sh tests/par/run_par.sh <libtoks.a> <tokenizer> <text> <out.log>      (from the repo root on the host)
#
# env: CPUS (comma list, fast cores first; default: every cpu of this process), KS ("1 2 4 8"), DOC_MIB
# ("1 4 16 64"), BATCH_MIB ("16"), DOC_BYTES (4096), REPS (7), SWEEP_K (the largest k), SWEEP_GAPS ("0 5000": us
# of idle before each timed sweep call: back to back, then sporadic), SWEEP_TEXTS ("fresh same": new text every
# call, then the same text every call), MODES ("doc batch sweep"),
# BENCH (an existing bench binary: skip the build), GIGA (tools/bench/gigatoken.sh's gigatoken-par: after each
# toks row of k >= 1, gigatoken's parallel api on the same cpus with RAYON_NUM_THREADS = k), REFS ("tiktoken hf":
# after each batch row of k >= 1, tools/bench/par_ref.py's encode_batch of each on the same cpus with k threads, via
# uv), SIB_OFF (x86: the SMT sibling's offset, e.g. 32 on tr9970x: their busy shares are recorded too).
set -eu
LIB=$1; TOK=$2; TXT=$3; LOG=$4
KS=${KS:-"1 2 4 8"}
DOC_MIB=${DOC_MIB:-"1 4 16 64"}
BATCH_MIB=${BATCH_MIB:-16}
DOC_BYTES=${DOC_BYTES:-4096}
REPS=${REPS:-7}
REFS=${REFS-"tiktoken hf"}
MODES=${MODES:-"doc batch sweep"}
CPUS=${CPUS:-$(awk '/^processor/ {printf "%s%s", s, $3; s=","}' /proc/cpuinfo)}
mkdir -p build
B=${BENCH:-build/bench_par}
if [ -z "${BENCH:-}" ]; then
    INFO=""
    grep -q TOKS_PAR_HAS_INFO include/toks.h && INFO=-DTOKS_PAR_HAS_INFO
    clang -std=c17 -O2 $INFO -Iinclude tests/par/bench_par.c "$LIB" -pthread -o "$B"
fi
first() { echo "$CPUS" | tr ',' '\n' | head -n "$1" | paste -sd, -; }
rel() { case $1 in "$HOME"/*) echo "\$HOME/${1#"$HOME"/}" ;; *) echo "$1" ;; esac; }   # no home dir in receipts
busy() {   # busy share per cpu of $1 (and its SMT sibling cpu + SIB_OFF) since the snapshot in $2 (a /proc/stat copy)
    l=$1
    if [ -n "${SIB_OFF:-}" ]; then for c in $(echo "$1" | tr ',' ' '); do l="$l,$((c + SIB_OFF))"; done; fi
    for c in $(echo "$l" | tr ',' ' '); do
        a=$(grep "^cpu$c " "$2"); b=$(grep "^cpu$c " /proc/stat)
        echo "$a" "$b" | awk -v c="$c" '{u1=$2+$3+$4+$7+$8; t1=u1+$5+$6; u2=$13+$14+$15+$18+$19; t2=u2+$16+$17;
            printf "cpu%s=%d%% ", c, (t2>t1) ? 100*(u2-u1)/(t2-t1) : 0}'
    done
}
run1() {   # cpus, the binary, then its arguments
    c=$1; shift
    rm -f build/stat0
    cat /proc/stat > build/stat0
    echo "LOAD before $(cut -d' ' -f1-3 /proc/loadavg) cpus=$c" >> "$LOG"
    taskset -c "$c" "$@" | tee -a "$LOG"
    echo "LOAD after $(cut -d' ' -f1-3 /proc/loadavg) busy: $(busy "$c" build/stat0)" >> "$LOG"
    ps -eo pcpu,psr,comm --sort=-pcpu | head -6 | tail -5 | tr -s ' ' | paste -sd';' - | sed 's/^/TOP /' >> "$LOG"
}
cell() {   # cpus, k, then the bench arguments (k is the last of them before reps for toks; the comparators take it as threads)
    c=$1; k=$2; shift 2
    run1 "$c" "$B" "$TOK" "$TXT" "$@"
    [ "$k" -gt 0 ] || return 0
    export RAYON_NUM_THREADS=$k
    if [ -n "${GIGA:-}" ]; then
        case $1 in
        doc) run1 "$c" "$GIGA" "$TOK" "$TXT" doc "$2" "$REPS" ;;
        batch) run1 "$c" "$GIGA" "$TOK" "$TXT" batch "$2" "$3" "$REPS" ;;
        esac
    fi
    if [ "$1" = batch ]; then   # encode_batch parallelizes over documents only: batch rows, not one big doc
        for t in $REFS; do run1 "$c" uv run -q tools/bench/par_ref.py "$t" "$TOK" "$TXT" "$2" "$3" "$k" "$REPS"; done
    fi
    unset RAYON_NUM_THREADS
}
{
    echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -m) $(date -u +%FT%TZ) git=${GIT_SHA:-$(cat .toks-rev 2>/dev/null || echo ?)} cpus=$CPUS reps=$REPS"
    echo "TOK $(rel "$TOK") $(sha256sum "$TOK" | cut -c1-16) TEXT $(rel "$TXT") $(sha256sum "$TXT" | cut -c1-16) $(wc -c < "$TXT") bytes"
} >> "$LOG"
for m in $MODES; do
    case $m in
    doc)
        for s in $DOC_MIB; do
            cell "$(first 1)" 0 doc "$s" 0 "$REPS"
            for k in $KS; do cell "$(first "$k")" "$k" doc "$s" "$k" "$REPS"; done
        done ;;
    batch)
        for s in $BATCH_MIB; do
            cell "$(first 1)" 0 batch "$DOC_BYTES" "$s" 0 "$REPS"
            for k in $KS; do cell "$(first "$k")" "$k" batch "$DOC_BYTES" "$s" "$k" "$REPS"; done
        done ;;
    sweep)
        k=${SWEEP_K:-$(echo $KS | tr ' ' '\n' | tail -n 1)}
        for tx in ${SWEEP_TEXTS:-fresh same}; do
            for gap in ${SWEEP_GAPS:-0 5000}; do run1 "$(first "$k")" "$B" "$TOK" "$TXT" sweep "$k" "$REPS" "$gap" "$tx"; done
        done ;;
    esac
done
echo "DONE $(date -u +%FT%TZ)" >> "$LOG"
