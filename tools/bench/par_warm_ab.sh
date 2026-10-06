#!/bin/sh
# tools/bench/par_warm_ab.sh: A B B A rounds of two tools/bench/par_warm.c builds (two libraries) on one pinned cpu, the
# receipt for a change to toks_par's default scratch. Run from the repository root, under the host's timing lock.
# Prints the host key, kernel and THP state (the arena's huge pages depend on them: docs/kernels.md §7), both commits,
# then one RUN line per run with the load and the pinned cpu's SMT sibling's busy share over the run (sib=). Every run
# of a cell must give the cell's first digest (A's ids = B's ids): one that does not is printed VOID and the script
# exits 1.
#
#   clang -std=c17 -O3 -march=native -Iinclude -o build/par_warm_A tools/bench/par_warm.c <A's tree>/build/<os>-<isa>/libtoks.a -lpthread
#   (the same for B), then
#   sh tools/bench/par_warm_ab.sh <par_warm A> <commit A> <par_warm B> <commit B> "<tokenizers>" "<corpora>" <chunk> \
#       <rounds> <cpu>          (each run: taskset -c <cpu> par_warm <tokenizer> <chunk> 3 5 <the corpus's files>)
#
# env: TOKS_HOST_KEY (docs/machines.md), TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers), TOKS_BENCH_TEXT
# (~/.cache/toks/bench-text; its files are checked against tools/bench/corpus.sha256 first). Corpora: en, code.
set -eu
[ $# -eq 9 ] || { echo "usage: par_warm_ab.sh <bin A> <commit A> <bin B> <commit B> <toks> <corpora> <chunk> <rounds> <cpu>" >&2; exit 2; }
BA=$1; CA=$2; BB=$3; CB=$4; TL=$5; CS=$6; CH=$7; R=$8; CPU=$9
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
T=${TOKS_BENCH_TEXT:-$HOME/.cache/toks/bench-text}
SUMS=$(pwd)/tools/bench/corpus.sha256
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
    esac
}
for c in $CS; do
    case $c in en | code) ;; *) echo "par_warm_ab.sh: corpus $c? (en, code)" >&2; exit 2 ;; esac
    for f in $(files "$c"); do
        b=$(basename "$f")
        w=$(awk -v n="$b" '$2 == n { print $1 }' "$SUMS")
        [ "$(sha256sum "$f" | cut -c1-64)" = "$w" ] || { echo "par_warm_ab.sh: $b does not match corpus.sha256" >&2; exit 1; }
    done
done
SIB=$(sed 's/[,-]/\n/g' "/sys/devices/system/cpu/cpu$CPU/topology/thread_siblings_list" | grep -vx "$CPU" | head -n 1 || true)
busy() {   # cpu $1's busy share since the /proc/stat copy in $2 ("none" without a sibling)
    [ -n "$1" ] || { echo none; return 0; }
    a=$(grep "^cpu$1 " "$2"); b=$(grep "^cpu$1 " /proc/stat)
    echo "$a" "$b" | awk '{u1=$2+$3+$4+$7+$8; t1=u1+$5+$6; u2=$13+$14+$15+$18+$19; t2=u2+$16+$17;
        printf "%d%%", (t2>t1) ? 100*(u2-u1)/(t2-t1) : 0}'
}
H=/sys/kernel/mm/transparent_hugepage
act() { sed 's/.*\[\(.*\)\].*/\1/' "$1"; }
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm) PIN cpu $CPU (sibling ${SIB:-none}) CHUNK $CH ROUNDS $R"
echo "THP enabled=$(act $H/enabled) defrag=$(act $H/defrag) use_zero_page=$(cat $H/use_zero_page) max_ptes_none=$(cat $H/khugepaged/max_ptes_none)"
echo "A $BA = commit $CA; B $BB = commit $CB"
echo "LOAD start $(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%FT%TZ)"
mkdir -p build
st=build/par_warm_ab.stat
void=0
for tk in $TL; do for c in $CS; do
    d0=
    r=0
    while [ "$r" -lt "$R" ]; do
        for side in A B B A; do
            b=$BA; [ $side = B ] && b=$BB
            cat /proc/stat > "$st"
            # shellcheck disable=SC2046
            l=$(taskset -c "$CPU" "$b" "$K/$tk" "$CH" 3 5 $(files "$c"))
            echo "RUN tk=$tk corp=$c chunk=$CH side=$side load=$(cut -d' ' -f1 /proc/loadavg) sib=$(busy "$SIB" "$st") $l"
            d=$(echo "$l" | sed -n 's/.*digest=\([0-9a-f]*\).*/\1/p')
            [ -n "$d0" ] || d0=$d
            [ -n "$d" ] && [ "$d" = "$d0" ] || { echo "VOID tk=$tk corp=$c side=$side: digest '$d', the cell's first run '$d0'"; void=1; }
        done
        r=$((r + 1))
    done
done; done
rm -f "$st"
echo "LOAD end $(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%FT%TZ)"
[ "$void" -eq 0 ] || { echo "par_warm_ab.sh: VOID runs above: A and B gave different ids" >&2; exit 1; }
