#!/bin/sh
# tests/par/gate_par.sh: the par section of the gigatoken gate (tools/bench/gate.sh's rules, tools/bench/gate_table.py
# reads both logs): per (mode, size, k), ROUNDS blocks of A B B A with A = bench_par's toks_par (a pool of k) and B =
# gigatoken-par (gigatoken's parallel api, RAYON_NUM_THREADS = k), both pinned to the same first k cpus of CPUS, plus a
# null block A N N A (toks_par against itself) per round. Each run times REPS pass samples (bench_par.c: sample i a
# fresh pool warmed on window i + 1 of the text, then timed on window i: every timed byte new to its pool); only
# sizes the text holds twice are run (pass_seen = 0: enwik8's 95.4 MiB holds 32 MiB twice, not 64). The void rule per
# block: an SMT sibling of a pinned cpu > 5% busy (SIB_OFF), another cpu of the pinned cpus' L3 domains > 50% busy (a
# build on the same CCD / cluster shares the cache and memory bandwidth), or the 1-minute load up by more than 2
# across the block: once more, then VOID.
#
#   sh tests/par/gate_par.sh <libtoks.a> <tokenizer> <text> > build/gate-par-<host>.log     (from the repo root)
#
# env: CPUS (comma list, fast cores first)  SIB_OFF (x86: the smt sibling offset, 32 on tr9970x)  KS (1 2 4 8)  DOC_MIB
#      (1 4 16 32)  BATCH_MIB (16)  DOC_BYTES (4096)  ROUNDS (3)  REPS (5)  GIGA (gigatoken-par, tools/bench/gigatoken.sh)
#      GIT_SHA. linux only (taskset, /proc/stat).
set -eu
LIB=$1; TOK=$2; TXT=$3
KS=${KS:-"1 2 4 8"}
DOC_MIB=${DOC_MIB:-"1 4 16 32"}
BATCH_MIB=${BATCH_MIB:-16}
DOC_BYTES=${DOC_BYTES:-4096}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-5}
GIGA=${GIGA:-build/giga-target/release/gigatoken-par}
CPUS=${CPUS:-$(awk '/^processor/ {printf "%s%s", s, $3; s=","}' /proc/cpuinfo)}
[ -x "$GIGA" ] || { echo "gate_par.sh: $GIGA missing (tools/bench/gigatoken.sh)" >&2; exit 1; }
mkdir -p build
INFO=""
grep -q TOKS_PAR_HAS_INFO include/toks.h && INFO=-DTOKS_PAR_HAS_INFO
clang -std=c17 -O2 $INFO -Iinclude tests/par/bench_par.c "$LIB" -pthread -o build/bench_par
first() { echo "$CPUS" | tr ',' '\n' | head -n "$1" | paste -sd, -; }
busy() {   # "cpuN=x%" of every cpu in $1 since the /proc/stat copy $2
    for c in $(echo "$1" | tr ',' ' '); do
        a=$(echo "$2" | grep "^cpu$c "); b=$(grep "^cpu$c " /proc/stat)
        echo "$a" "$b" | awk -v c="$c" '{u1=$2+$3+$4+$7+$8; t1=u1+$5+$6; u2=$13+$14+$15+$18+$19; t2=u2+$16+$17;
            printf "cpu%s=%d%% ", c, (t2>t1) ? 100*(u2-u1)/(t2-t1) : 0}'
    done
}
sibs() { if [ -n "${SIB_OFF:-}" ]; then for x in $(echo "$1" | tr ',' ' '); do printf "%s," $((x + SIB_OFF)); done | sed 's/,$//'; fi; }
run() {   # side $1: A / N = bench_par, B = gigatoken-par; its PAR / GIGA line tagged (a failed run says so: bench_par
          # exits on any id that differs from serial toks_encode, gigatoken-par on a pass id count that moves)
    tag="RUN par=1 mode=$mode size=$s k=$k round=$r try=$try kind=$kind pos=$pos side=$1"
    if [ "$mode" = doc ]; then args="doc $s"; else args="batch $DOC_BYTES $s"; fi
    # shellcheck disable=SC2086
    case $1 in
        A|N) out=$(taskset -c "$c" build/bench_par "$TOK" "$TXT" $args "$k" "$REPS" 2>&1) && rc=0 || rc=$? ;;
        B) out=$(RAYON_NUM_THREADS=$k taskset -c "$c" "$GIGA" "$TOK" "$TXT" $args "$REPS" 2>&1) && rc=0 || rc=$? ;;
    esac
    echo "$out" | grep '^PAR\|^GIGA' | sed "s/^/$tag /"
    if [ "$rc" != 0 ]; then echo "$tag FAILED exit=$rc $(echo "$out" | tail -n 1)"; fi
}
l3of() {   # the cpus sharing a last-level cache with any of the cpus $1, comma-separated, without $1 and its siblings
    ex=" $(echo "$1,$(sibs "$1")" | tr ',' ' ') "
    for x in $(echo "$1" | tr ',' ' '); do cat /sys/devices/system/cpu/cpu"$x"/cache/index3/shared_cpu_list 2>/dev/null; done |
        tr ',' '\n' | awk -F- '{if (NF == 2) for (i = $1; i <= $2; i++) print i; else if ($1 != "") print $1}' | sort -un |
        awk -v ex="$ex" 'index(ex, " " $1 " ") == 0' | paste -sd, -
}
block() {   # the sides $1 in order, a block of kind $2; sets bad=1 when the void rule fires
    kind=$2
    t0=$(date +%s)
    l0=$(cut -d' ' -f1 /proc/loadavg)
    s0=$(cat /proc/stat)
    pos=0
    for side in $1; do pos=$((pos + 1)); run "$side"; done
    l1=$(cut -d' ' -f1 /proc/loadavg)
    sb=$(sib=$(sibs "$c"); [ -n "$sib" ] && busy "$sib" "$s0" || true)
    nb=$(n3=$(l3of "$c"); [ -n "$n3" ] && busy "$n3" "$s0" || true)
    pb=$(busy "$c" "$s0")
    sm=$(echo "$sb" | tr ' ' '\n' | sed -n 's/.*=\([0-9]*\)%/\1/p' | sort -n | tail -n 1)
    nm=$(echo "$nb" | tr ' ' '\n' | sed -n 's/.*=\([0-9]*\)%/\1/p' | sort -n | tail -n 1)
    bad=$(awk -v a="$l0" -v b="$l1" -v s="${sm:-0}" -v n="${nm:-0}" 'BEGIN {print (b - a > 2 || s > 5 || n > 50) ? 1 : 0}')
    echo "BLOCK par=1 mode=$mode size=$s k=$k round=$r try=$try kind=$2 t0=$t0 t1=$(date +%s) load0=$l0 load1=$l1 sib_busy=${sm:-na} l3_busy=${nm:-na} bad=$bad cpus=$c pinned: $pb siblings: $sb l3: $nb"
}
guarded() {
    try=1
    block "$1" "$2"
    if [ "$bad" = 1 ]; then
        try=2
        block "$1" "$2"
        if [ "$bad" = 1 ]; then echo "VOID par=1 mode=$mode size=$s k=$k round=$r kind=$2"; fi
    fi
    return 0
}
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"   # the chipset key (docs/machines.md), never a hostname
echo "GATEPAR CPUS $CPUS SIB_OFF ${SIB_OFF:-none} KS '$KS' DOC_MIB '$DOC_MIB' BATCH_MIB '$BATCH_MIB' DOC_BYTES $DOC_BYTES ROUNDS $ROUNDS REPS $REPS"
echo "GIT ${GIT_SHA:-$(cat .toks-rev 2>/dev/null || echo unknown)}"   # remote.sh syncs without .git, with .toks-rev
echo "TOKPAR $TOK $(sha256sum "$TOK" | cut -c1-16) TEXT $TXT $(sha256sum "$TXT" | cut -c1-16) $(wc -c < "$TXT") bytes"
sha256sum build/bench_par "$GIGA" | sed 's/^/BIN /'
echo "UPTIME $(uptime)"
for mode in doc batch; do
    if [ "$mode" = doc ]; then sizes=$DOC_MIB; else sizes=$BATCH_MIB; fi
    for s in $sizes; do
        for k in $KS; do
            c=$(first "$k")
            r=1
            while [ "$r" -le "$ROUNDS" ]; do
                guarded "A B B A" gate
                guarded "A N N A" null
                r=$((r + 1))
            done
        done
    done
done
echo "UPTIME $(uptime)"
