#!/bin/sh
# tests/fuzz/run.sh: run the libFuzzer harnesses of tests/fuzz/ (docs/fuzz.md). From the repository root on a
# lab host (tools/remote.sh puts you there), never on a laptop:
#
#   tests/fuzz/run.sh build                       make -f tests/fuzz/Makefile -j8 (asan + ubsan)
#   tests/fuzz/run.sh seeds                       tests/fuzz/seeds.py: texts, ids and shrunk tokenizer files
#   tests/fuzz/run.sh start <secs> [harness...]   one detached libFuzzer process per harness, <secs> each
#   tests/fuzz/run.sh status                      per harness: state, execs, exec/s, cov, ft, corpus, findings
#   tests/fuzz/run.sh stop [harness...]           stops the processes this script started (pid files only); a stopped
#                                                 run keeps its cpu-s in the ledger
#   tests/fuzz/run.sh merge <harness>             minimizes the harness's corpus into corpus-min/<harness>
#   tests/fuzz/run.sh regress                     replays tests/fuzz/regress/<harness>/* (the findings' repros)
#   tests/fuzz/run.sh probe [harness...]          the arena probe (arprobe.h): load / load_json replay corpus, seeds
#                                                 and repros; per toks_ar_alloc site calls, refusals, least slack
#   tests/fuzz/run.sh cover                       line / branch coverage of src/ by the corpora (llvm-cov)
#   tests/fuzz/run.sh hours [state...]            the cpu-hour ledger: every run's cpu-s per harness and build sha,
#                                                 runs in flight included (docs/fuzz.md §4: T6 counts cpu-h per isa
#                                                 per entry point)
#
# Harnesses: encode pieces decode stream par load load_json (default: all seven, i.e. seven processes; keep the
# host's shared limit in mind: <= 8 per lab host). FUZZ_BIN=<dir> runs binaries from another build directory (a rebuilt
# harness can join a running campaign without overwriting the binaries in use). FUZZ_SHA=<sha> names the source
# the binaries were built from in each run's .meta (tools/remote.sh syncs without .git); TOKS_FUZZ_PINS narrows the
# text harnesses to some tokenizers (fuzz.h: targeted campaigns), and goes to the .meta too. State lives in build/fuzz/
# (FUZZ_STATE=<dir> elsewhere, e.g. for an msan run beside an asan one; tools/remote.sh keeps build/):
#   corpus/<h>/      grows across runs (libFuzzer writes new units here)
#   seeds/<h>/       read-only seed inputs (tests/fuzz/seeds.py writes them)
#   runs/<h>.<stamp>.log, pids/<h>.pid
#   findings/<h>/    crash-*, leak-*, timeout-*, oom-*, slow-unit-* (each is a reproducer: ./fuzz_<h> <file>)
set -eu

ALL="encode pieces decode stream par load load_json"
case "$(uname -s)" in Darwin) os=macos ;; *) os=linux ;; esac
case "$(uname -m)" in aarch64|arm64) isa=arm64 ;; *) isa=x86_64 ;; esac
# host rule (2026-10-04): fuzzing stays off the cores others time on. GB10 sparks: the Cortex-A725 cores 0-4 and
# 10-14 (the X925 cores 5-9 / 15-19 stay free); tr9970x: CCD0 only, cpus 0-7 and their siblings 32-39 (the other
# CCDs carry timed runs). FUZZ_CPUS overrides ("" = no pin).
if [ "$isa" = arm64 ]; then cpus_default=0-4,10-14; else cpus_default=0-7,32-39; fi
CPUS=${FUZZ_CPUS-$cpus_default}
PIN=""
if [ -n "$CPUS" ] && command -v taskset >/dev/null 2>&1; then PIN="taskset -c $CPUS"; fi
BIN=${FUZZ_BIN:-build/fuzz-$os-$isa}
ST=${FUZZ_STATE:-build/fuzz}

maxlen() {
    case "$1" in
        load|load_json) echo 65536 ;;
        decode|stream) echo 16388 ;;
        *) echo 8200 ;;
    esac
}

extra() {
    case "$1" in
        # the load harnesses start fast (no pinned files): fork mode keeps them going past a finding, each one saved
        load) echo "-use_value_profile=1 -dict=tests/fuzz/tokenizer.dict -fork=1 -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1" ;;
        load_json) echo "-dict=tests/fuzz/tokenizer.dict -fork=1 -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1" ;;
        *) echo "" ;;
    esac
}

# the cpu-s so far of a run in flight (linux): every process of the run's session but its /usr/bin/time (the session
# leader, whose pid the pid file holds), user + sys of each and of the children it has reaped (/proc/<pid>/stat
# fields 14-17): the fuzzer and a fork-mode run's job in flight; each cpu-second counted once
livecpu() {
    hz=$(getconf CLK_TCK)
    tot=0
    for c in $(ps -o pid= -s "$1" 2>/dev/null); do
        [ "$c" = "$1" ] && continue
        v=$(awk '{ sub(/^.*\) /, ""); printf "%d", $12 + $13 + $14 + $15 }' "/proc/$c/stat" 2>/dev/null) || v=0
        tot=$((tot + ${v:-0}))
    done
    echo $((tot / hz))
}

# one ledger line for `hours`: <run path without extension> <cpu-s> <1 if in flight>
runline() {
    h=$(basename "$1" | sed 's/\.[0-9]*-[0-9]*$//')
    sha=$(sed -n 's/^sha=\([^ ]*\).*/\1/p' "$1.meta" 2>/dev/null)
    pins=$(sed -n 's/.* pins=\([^ ]*\).*/\1/p' "$1.meta" 2>/dev/null)
    ex=$(grep -a -E '^#[0-9]+' "$1.log" 2>/dev/null | tail -1 | sed -n 's/^#\([0-9]*\).*/\1/p')
    echo "$h ${sha:-unknown} ${pins:-default} $2 ${ex:-0} $3"
}

cmd=${1:-status}
[ $# -gt 0 ] && shift

case "$cmd" in
build)
    exec $PIN make -f tests/fuzz/Makefile -j8
    ;;
seeds)
    $PIN uv run -q tests/fuzz/seeds.py texts --out "$ST/seeds" ${TOKS_BENCH_TEXT:-build/text}
    $PIN uv run -q tests/fuzz/seeds.py shrink --out "$ST/seeds" "${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}" tests/data/compile
    $PIN uv run -q tests/fuzz/seeds.py sweep --out "$ST/seeds" "${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}" tests/data/compile
    $PIN uv run -q tests/fuzz/seeds.py tiktoken --out "$ST/seeds" "${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}"
    ;;
start)
    secs=${1:?seconds}; shift
    hs=${*:-$ALL}
    stamp=$(date +%Y%m%d-%H%M%S)
    for h in $hs; do
        mkdir -p "$ST/corpus/$h" "$ST/seeds/$h" "$ST/runs" "$ST/pids" "$ST/findings/$h"
        if [ -f "$ST/pids/$h.pid" ] && kill -0 "$(cat "$ST/pids/$h.pid")" 2>/dev/null; then
            echo "$h: already running (pid $(cat "$ST/pids/$h.pid"))"; continue
        fi
        log="$ST/runs/$h.$stamp.log"
        # T6 / the 1.0 gate count asan + ubsan + lsan hours: leak detection is forced on (last wins in ASAN_OPTIONS)
        # and recorded in the run's .meta and as its log's first line
        asan="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=1"
        printf 'sha=%s isa=%s cpus=%s pins=%s secs=%s bin=%s asan=%s\n' "${FUZZ_SHA:-unknown}" "$isa" "${CPUS:-all}" \
            "${TOKS_FUZZ_PINS:-default}" "$secs" "$BIN" "$asan" > "$ST/runs/$h.$stamp.meta"
        printf 'ASAN_OPTIONS=%s\n' "$asan" > "$log"
        timer=""                                   # cpu time of the run (user + sys) for the cpu-hour count
        [ -x /usr/bin/time ] && timer="/usr/bin/time -v -o $ST/runs/$h.$stamp.time"
        # shellcheck disable=SC2046,SC2086
        ASAN_OPTIONS="$asan" setsid nohup $PIN $timer "$BIN/fuzz_$h" -max_total_time="$secs" -max_len="$(maxlen "$h")" \
            -timeout=120 -rss_limit_mb=6144 -malloc_limit_mb=4096 -print_final_stats=1 -report_slow_units=30 \
            -artifact_prefix="$ST/findings/$h/" $(extra "$h") "$ST/corpus/$h" "$ST/seeds/$h" \
            >> "$log" 2>&1 < /dev/null &
        echo $! > "$ST/pids/$h.pid"
        echo "$h: pid $! for ${secs}s, log $log"
    done
    ;;
stop)
    # the pid file holds the run's /usr/bin/time, the leader of the setsid group: SIGTERM goes to every other process of
    # the group (libFuzzer exits on it, fork-mode jobs included), so time outlives its child and writes the run's .time
    # file and a stopped run keeps its cpu-s in the ledger (`hours`), less a fork-mode run's job in flight (<= 300 s:
    # the ledger under-counts, never over-counts). Whatever is left after 30 s gets SIGKILL.
    for f in "$ST"/pids/*.pid; do
        [ -f "$f" ] || continue
        h=$(basename "$f" .pid)
        if [ $# -gt 0 ] && ! echo " $* " | grep -q " $h "; then continue; fi
        p=$(cat "$f")
        if kill -0 "$p" 2>/dev/null; then
            if [ "$(ps -o comm= -p "$p" 2>/dev/null)" = time ]; then
                for q in $(pgrep -g "$p"); do [ "$q" = "$p" ] || kill -TERM "$q" 2>/dev/null || true; done
            else
                kill -TERM -- "-$p" 2>/dev/null || true
            fi
            n=0
            while kill -0 "$p" 2>/dev/null && [ $n -lt 30 ]; do sleep 1; n=$((n + 1)); done
            if kill -0 "$p" 2>/dev/null; then
                kill -KILL -- "-$p" 2>/dev/null || true
                echo "stopped $h ($p): SIGKILL after 30 s, its cpu-s are not in the ledger"
            else
                echo "stopped $h ($p)"
            fi
        fi
        rm -f "$f"
    done
    ;;
status)
    printf '%-10s %-8s %12s %8s %6s %7s %7s %9s %8s %s\n' harness state execs exec/s cov ft corpus elapsed cpu-s findings
    for h in $ALL; do
        log=$(ls -t "$ST"/runs/"$h".*.log 2>/dev/null | head -1 || true)
        [ -n "$log" ] || continue
        state=done
        if [ -f "$ST/pids/$h.pid" ] && kill -0 "$(cat "$ST/pids/$h.pid")" 2>/dev/null; then state=running; fi
        last=$(grep -E '^#[0-9]+' "$log" | tail -1 || true)
        last=$(printf '%s' "$last" | cut -d' ' -f1-12 | tr -cd '[:print:]')   # MS:/DE: may carry raw bytes
        execs=$(echo "$last" | sed -n 's/^#\([0-9]*\).*/\1/p')
        eps=$(echo "$last" | sed -n 's/.*exec\/s: \([0-9]*\).*/\1/p')
        cov=$(echo "$last" | sed -n 's/.*cov: \([0-9]*\).*/\1/p')
        ft=$(echo "$last" | sed -n 's/.*ft: \([0-9]*\).*/\1/p')
        el=$(sed -n 's/^Done [0-9]* runs in \([0-9]*\) second.*/\1/p' "$log" | tail -1)
        [ -n "$el" ] || el="$(( $(date +%s) - $(stat -c %W "$log" 2>/dev/null || stat -f %B "$log") ))*"
        tf="${log%.log}.time"
        cpu=-
        if [ -s "$tf" ]; then                     # time creates its file at the start and fills it at the end
            cpu=$(awk -F': ' '/User time|System time/ { s += $2 } END { printf "%d", s }' "$tf")
        elif [ "$state" = running ]; then
            cpu="$(livecpu "$(cat "$ST/pids/$h.pid")")*"
        fi
        nf=$(( $(ls "$ST/findings/$h" 2>/dev/null | wc -l) + $(grep -a -h "runtime error" "$ST"/runs/"$h".*.log 2>/dev/null | sort -u | wc -l) ))
        nc=$(ls "$ST/corpus/$h" 2>/dev/null | wc -l | tr -d ' ')
        printf '%-10s %-8s %12s %8s %6s %7s %7s %9s %8s %s\n' "$h" "$state" "${execs:--}" "${eps:--}" "${cov:--}" \
            "${ft:--}" "$nc" "$el" "$cpu" "$nf"
    done
    ;;
cover)
    # source-based coverage of src/ by the corpora: every harness's corpus (and seeds) replayed once
    $PIN make -f tests/fuzz/Makefile SAN=coverage -j8 >/dev/null
    CB=build/fuzz-cov-$os-$isa
    rm -rf "$ST/cov" && mkdir -p "$ST/cov"
    objs=""
    for h in $ALL; do
        dirs="$ST/corpus/$h"
        [ -d "$ST/corpus-min/$h" ] && dirs="$ST/corpus-min/$h"
        LLVM_PROFILE_FILE="$ST/cov/$h.%p.profraw" $PIN "$CB/fuzz_$h" -runs=0 -max_len="$(maxlen "$h")" -rss_limit_mb=6144 \
            $dirs "$ST/seeds/$h" > "$ST/cov/$h.log" 2>&1 || echo "$h: replay exit $?"
        if [ -z "$objs" ]; then objs="$CB/fuzz_$h"; else objs="$objs -object $CB/fuzz_$h"; fi
    done
    llvm-profdata merge -sparse "$ST"/cov/*.profraw -o "$ST/cov/all.profdata"
    # shellcheck disable=SC2086
    llvm-cov report -instr-profile="$ST/cov/all.profdata" $objs src/core src/platform | tee "$ST/cov/report.txt"
    ;;
hours)
    # every finished run's cpu-s (user + sys, its /usr/bin/time file) and every run in flight's so far (livecpu), per
    # harness, per build sha and in total, over the state dirs given (default: $ST; e.g. two processes per harness
    # in build/fuzz-a and build/fuzz-b)
    printf '%-10s %-10s %-24s %10s %14s\n' harness sha pins cpu-h execs
    for st in ${*:-$ST}; do
        for tf in "$st"/runs/*.time; do
            [ -s "$tf" ] || continue              # empty until the run ends: counted below while in flight
            runline "${tf%.time}" "$(awk -F': ' '/User time|System time/ { s += $2 } END { printf "%d", s }' "$tf")" 0
        done
        for f in "$st"/pids/*.pid; do
            [ -f "$f" ] || continue
            p=$(cat "$f")
            kill -0 "$p" 2>/dev/null || continue
            b=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | sed -n 's/.* -o \([^ ]*\)\.time .*/\1/p')
            [ -n "$b" ] && [ ! -s "$b.time" ] || continue
            runline "$b" "$(livecpu "$p")" 1
        done
    done | awk -v isa="$isa" '
        { cpu[$1] += $4; ex[$1] += $5; fl[$1] += $6; key = $1 " " $2 " " $3; c2[key] += $4; e2[key] += $5 }
        END {
            for (k in c2) { split(k, a, " "); printf "%-10s %-10s %-24s %10.2f %14d\n", a[1], substr(a[2], 1, 10), a[3], c2[k] / 3600, e2[k] }
            for (h in cpu) printf "%-10s %-10s %-24s %10.2f %14d  (total, %s, %d in flight)\n", h, "all", "-", cpu[h] / 3600, ex[h], isa, fl[h]
        }' | sort
    ;;
regress)
    # every repro in tests/fuzz/regress/<harness>/ through its harness (each must pass once its fix is in)
    fail=0
    mkdir -p "$ST"
    for d in tests/fuzz/regress/*/; do
        h=$(basename "$d")
        for f in "$d"*; do
            [ -f "$f" ] || continue
            if $PIN "$BIN/fuzz_$h" -rss_limit_mb=6144 "$f" > "$ST/regress.log" 2>&1 && ! grep -q "runtime error" "$ST/regress.log"; then echo "pass $h $f"
            else echo "FAIL $h $f: $(grep -a -m1 'INVARIANT\|ERROR:\|runtime error' "$ST/regress.log" | cut -c1-200)"; fail=1; fi
        done
    done
    exit $fail
    ;;
probe)
    # the arena probe (arprobe.h): each load harness replays its corpus, its seeds and the regress repros in one
    # process (-runs=0); per toks_ar_alloc site the calls, the refusals and the least slack (bound - end) seen. A
    # refusal the builder turns into TOKS_E_NOMEM stops the replay (load.h's check) and fails here; the others are
    # the loaders' designed limits (docs/fuzz.md §5.7 names each site) and are listed with their site.
    fail=0
    mkdir -p "$ST/probe"
    for h in ${*:-load load_json}; do
        dirs=""
        for d in "$ST/corpus/$h" "$ST/seeds/$h" "tests/fuzz/regress/$h"; do [ -d "$d" ] && dirs="$dirs $d"; done
        # shellcheck disable=SC2086
        $PIN "$BIN/fuzz_$h" -runs=0 -max_len="$(maxlen "$h")" -timeout=120 -rss_limit_mb=6144 $dirs \
            > "$ST/probe/$h.log" 2>&1 || { echo "FAIL $h: the replay stopped: $(grep -a -m1 'INVARIANT\|ERROR:\|runtime error' "$ST/probe/$h.log" | cut -c1-200)"; fail=1; }
        grep -a '^arprobe ' "$ST/probe/$h.log" | grep -v '^arprobe REFUSED' | sed "s|^arprobe|probe $h|"
        units=$(grep -a -m1 -o 'INITED.*' "$ST/probe/$h.log" || true)
        nsite=$(grep -a '^arprobe ' "$ST/probe/$h.log" | grep -v -c REFUSED || true)
        refs=$(grep -a '^arprobe REFUSED' "$ST/probe/$h.log" | awk '{print $3}' | sort | uniq -c | awk '{printf " %s x%s", $2, $1}')
        echo "probe $h: $nsite sites, refusals:${refs:- none}; $units"
        nre=$(grep -a -c 'runtime error' "$ST/probe/$h.log" || true)
        [ "$nre" = 0 ] || { echo "FAIL $h: $nre UBSan 'runtime error' lines: $(grep -a -m1 'runtime error' "$ST/probe/$h.log" | cut -c1-200)"; fail=1; }
    done
    exit $fail
    ;;
merge)
    h=${1:?harness}
    mkdir -p "$ST/corpus-min/$h"
    exec $PIN "$BIN/fuzz_$h" -merge=1 -max_len="$(maxlen "$h")" -timeout=120 -rss_limit_mb=6144 \
        "$ST/corpus-min/$h" "$ST/corpus/$h" "$ST/seeds/$h"
    ;;
*)
    echo "usage: tests/fuzz/run.sh build | seeds | start <secs> [harness...] | status | stop | merge <harness> | regress | probe [harness...] | cover | hours" >&2
    exit 2
    ;;
esac
