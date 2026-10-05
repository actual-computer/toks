#!/bin/sh
# tools/bench/spm_stages.sh [pin]: the stage profile of a sentencepiece-style bpe tokenizer (gemma 4 by default),
# one thread, by ablation: four library variants of src/core/spm_c.c (tools/bench/spm_stages.py: ship, prof, scan,
# drv), each linked into tools/bench/spm_stages.c, run back to back per cell. Shares of the shipping time:
#   driver = drv / ship, scan = (scan - drv) / ship, model = prof's timed model calls / prof's wall,
#   cache (probe, emit, fill, wide entries, flush loops) = the rest.
#   e.g. PINCPU=7 tools/bench/spm_stages.sh "taskset -c 7"
# env: SP_TOK (gemma4)  CORPORA (en code zh ml)  CHUNKS (4096 0)  REPS (5)  VARIANTS (ship prof scan drv)
#      SP_CACHE_MIB (the scratch's TOKS_SCRATCH_CACHE_MIB, default 0 = 2 MiB)  TOKS_BENCH_TEXT (build/text)
set -e
PIN="$1"
T=${TOKS_BENCH_TEXT:-build/text}
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
TOK=${SP_TOK:-gemma4}
CORPORA=${CORPORA:-"en code zh ml"}
CHUNKS=${CHUNKS:-"4096 0"}
REPS=${REPS:-5}
VARIANTS=${VARIANTS:-"ship prof scan drv"}
CC=${CC:-clang}
PY=$(command -v python3 || echo "uv run --no-project -q python")
for v in $VARIANTS; do
    d=build/spv/$v
    rm -rf "$d"
    mkdir -p "$d"
    cp -R Makefile include src "$d"/
    $PY tools/bench/spm_stages.py "$v" "$d"/src/core/spm_c.c
    (cd "$d" && make -j8 lib >/dev/null)
    $CC -std=c17 -O2 -Iinclude -o build/spv/spm_stages_"$v" tools/bench/spm_stages.c "$d"/build/*/libtoks.a -lpthread
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
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm) PIN '$PIN' PINCPU '$PINCPU' TOK $TOK REPS $REPS SP_CACHE_MIB ${SP_CACHE_MIB:-0} GIT ${GIT_SHA:-unknown}"
for corp in $CORPORA; do
    for ch in $CHUNKS; do
        echo "== $TOK $corp chunk $ch LOAD before $(loadavg)"
        s0=$(cpustat)
        for v in $VARIANTS; do
            # shellcheck disable=SC2046
            SP_VARIANT=$v $PIN ./build/spv/spm_stages_"$v" "$K/$TOK" "$ch" "$REPS" $(files "$corp")
        done
        busy "$s0" "$(cpustat)"
        echo "LOAD after $(loadavg)"
    done
done
