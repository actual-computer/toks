#!/bin/sh
# tools/bench/spm_stages_ab.sh <dirA> <dirB> <tok> <pin> "<corpora>" ["<chunks>"] [reps]: an A / B timing of two source
# trees (two worktrees, e.g. master and a branch), spm_stages.c's ship variant (cold / pass / warm) linked against each
# tree's library, runs in the order A B B A per (corpus, chunk) on the pinned core; each SP line carries ids= (the
# FNV-1a of every call's ids), so the same run proves ids equality. Table: tools/bench/spm_stages_ab.py <log>.
#   e.g. tools/bench/spm_stages_ab.sh ~/toks-ci/master ~/toks-ci/my-branch gemma4 "taskset -c 9" "en code ml zh" \
#        "4096 0" 3 > build/ab.log; python3 tools/bench/spm_stages_ab.py build/ab.log
# env: BUILD_CPUS (the cores the two library builds use, default 0-4,10-12), CC, TOKS_BENCH_TEXT (build/text: the
#      corpus files of tools/bench/e2e.sh), TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers)
set -e
A=$1; B=$2; TOK=$3; PIN=$4; CORPORA=$5; CHUNKS=${6:-4096}; REPS=${7:-3}
R=$(cd "$(dirname "$0")/../.." && pwd)
T=${TOKS_BENCH_TEXT:-$R/build/text}
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
CC=${CC:-clang}
W=$R/build/ab.$$
mkdir -p "$W"
for side in A B; do
    d=$A; [ $side = B ] && d=$B
    (cd "$d" && rm -rf build/ab && mkdir -p build/ab && cp -R Makefile include src build/ab/ && cd build/ab &&
     taskset -c "${BUILD_CPUS:-0-4,10-12}" make -j8 lib > /dev/null)
    $CC -std=c17 -O2 -I"$d/include" -o "$W/$side" "$R/tools/bench/spm_stages.c" "$d"/build/ab/build/*/libtoks.a -lpthread
done
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        ml) echo "$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt $T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
    esac
}
loadavg() { cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo "?"; }
for corp in $CORPORA; do
    for ch in $CHUNKS; do
        echo "== $TOK $corp chunk $ch LOAD before $(loadavg)"
        for side in A B B A; do
            # shellcheck disable=SC2046
            SP_VARIANT=$side $PIN "$W/$side" "$K/$TOK" "$ch" "$REPS" $(files "$corp")
        done
        echo "LOAD after $(loadavg)"
    done
done
rm -rf "$W"
