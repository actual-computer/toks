#!/bin/sh
# tools/bench/emb_stages.sh [pin]: the stage profile of an embedder tokenizer (unigram: bge-m3 by default; wordpiece:
# all-MiniLM-L6-v2), one thread, by ablation: library variants of the algorithm's sources (tools/bench/emb_stages.py),
# each linked into tools/bench/spm_stages.c (its counters: sp_cnt), run back to back per cell. Shares of the shipping time:
#   uni  driver = drv / ship, walk (normalizer + pre-tokenizers) = (walk - drv) / ship, model (Viterbi) = (ship - walk) / ship
#   wp   driver = drv / ship, scan = (scan - drv) / ship, model = (ship - scan) / ship
# prof's timers (enc_ns / model_ns in its lines: uni = segment / Viterbi, wp = scan / model) cross-check the ablations.
#   e.g. PINCPU=9 EM_ALGO=uni tools/bench/emb_stages.sh "taskset -c 9"
# env: EM_ALGO (uni | wp)  EM_TOK (uni_bgem3 | wp-minilm-l6)  CORPORA (en code zh ml)  CHUNKS (4096 0)  REPS (5)
#      VARIANTS (ship prof walk drv | ship prof scan drv)  SP_CACHE_MIB  TOKS_BENCH_TEXT (build/text)
set -e
PIN="$1"
T=${TOKS_BENCH_TEXT:-build/text}
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
ALGO=${EM_ALGO:-uni}
case $ALGO in
    uni) TOK=${EM_TOK:-uni_bgem3}; VARIANTS=${VARIANTS:-"ship prof walk drv"} ;;
    wp) TOK=${EM_TOK:-wp-minilm-l6}; VARIANTS=${VARIANTS:-"ship prof scan drv"} ;;
    *) echo "EM_ALGO: uni or wp" >&2; exit 2 ;;
esac
CORPORA=${CORPORA:-"en code zh ml"}
CHUNKS=${CHUNKS:-"4096 0"}
REPS=${REPS:-5}
CC=${CC:-clang}
PY=$(command -v python3 || echo "uv run --no-project -q python")
for v in $VARIANTS; do
    d=build/emv/$ALGO-$v
    rm -rf "$d"
    mkdir -p "$d"
    cp -R Makefile include src "$d"/
    $PY tools/bench/emb_stages.py "$ALGO" "$v" "$d"/src/core
    (cd "$d" && make -j8 lib >/dev/null)
    $CC -std=c17 -O2 -Iinclude -o build/emv/emb_stages_"$ALGO"_"$v" tools/bench/spm_stages.c "$d"/build/*/libtoks.a -lpthread
done
loadavg() { if [ -r /proc/loadavg ]; then cut -d' ' -f1-3 /proc/loadavg; else sysctl -n vm.loadavg | tr -d '{}'; fi; }
cpustat() {
    [ -n "$PINCPU" ] && [ -r /proc/stat ] || return 0
    sib=$(cat /sys/devices/system/cpu/cpu"$PINCPU"/topology/thread_siblings_list 2>/dev/null || echo "$PINCPU")
    for c in $(echo "$sib" | tr ',' ' '); do
        awk -v c="cpu$c" '$1 == c {b = $2 + $3 + $4 + $7 + $8 + $9; t = b + $5 + $6; print c, b, t}' /proc/stat
    done
}
busy() {
    [ -n "$1" ] || return 0
    echo "BUSY $(printf '%s\n%s\n' "$1" "$2" | awk '{if ($1 in b) {db = $2 - b[$1]; dt = $3 - t[$1]; v = 0; if (dt > 0) v = 100 * db / dt; printf "%s %.0f%% ", $1, v} else {b[$1] = $2; t[$1] = $3}}')"
}
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        ml) echo "$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt $T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt" ;;
        cjk) echo "$T/wiki-zh.txt $T/wiki-ja.txt $T/wiki-ko.txt $T/gut-zh-24264.txt" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
    esac
}
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm) PIN '$PIN' PINCPU '$PINCPU' ALGO $ALGO TOK $TOK REPS $REPS GIT ${GIT_SHA:-unknown}"
for corp in $CORPORA; do
    for ch in $CHUNKS; do
        echo "== $TOK $corp chunk $ch LOAD before $(loadavg)"
        s0=$(cpustat)
        for v in $VARIANTS; do
            # shellcheck disable=SC2046
            SP_VARIANT=$v $PIN ./build/emv/emb_stages_"$ALGO"_"$v" "$K/$TOK" "$ch" "$REPS" $(files "$corp")
        done
        busy "$s0" "$(cpustat)"
        echo "LOAD after $(loadavg)"
    done
done
