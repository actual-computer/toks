#!/bin/sh
# tools/bench/k5_sweep.sh: tools/bench/k5_sweep.c over gpt2 / llama3 / glm53 x en / code / ml, with receipts (host,
# cpu, load before and after every cell, the pinned cpu's other running threads on linux, binary and corpus sha-256).
#
#   tools/bench/k5_sweep.sh "<pin>" [reps] [log2 buckets lo] [log2 buckets hi]
#   e.g. PINCPU=38 tools/bench/k5_sweep.sh "taskset -c 38" 5 10 18        (macOS: no pin, "")
#
# needs build/text/ (the corpora of tools/bench/e2e.sh: en = gut-en-1342 + gut-en-2701, code = code-cpython.py +
# code-toks.c, ml = wiki-*.txt + gut-de-2229 + gut-es-2000 + gut-fr-17489 + gut-zh-24264) and build/tok/<name>.
# env: TOKS_LIST (gpt2 llama3 glm53)  CORPORA (en code ml)  HUGE=1 (linux: the cache madvise'd to huge pages)
set -e
PIN="$1"
REPS=${2:-5}
LO=${3:-10}
HI=${4:-18}
T=build/text
K=build/tok
TOKS_LIST=${TOKS_LIST:-"gpt2 llama3 glm53"}
CORPORA=${CORPORA:-"en code ml"}
CC=${CC:-clang}
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }
make -j8 lib >/dev/null
BD=$(mkvar BUILD_DIR)
# the library's own code-generation flags (SPEC §9's CSTRICT minus the warnings), so every variant compiles alike
$CC -std=c17 -O3 -fno-strict-aliasing -fwrapv -fPIC -Wall -Wextra -Werror $(mkvar CPPFLAGS) -o build/k5_sweep \
    tools/bench/k5_sweep.c "$BD/libtoks.a"
loadavg() { if [ -r /proc/loadavg ]; then cut -d' ' -f1-3 /proc/loadavg; else sysctl -n vm.loadavg | tr -d '{}'; fi; }
pinned() {
    [ -n "$PINCPU" ] && [ -r /proc/loadavg ] || return 0
    sib=$(cat /sys/devices/system/cpu/cpu"$PINCPU"/topology/thread_siblings_list 2>/dev/null || echo "$PINCPU")
    echo "PINNED cpu $PINCPU siblings $sib: $(ps -eLo psr,stat,pcpu,comm --no-headers | awk -v s="$sib" \
        'BEGIN{n=split(s,a,/[,-]/); for(i=1;i<=n;i++) c[a[i]]=1} ($1 in c) && $2 ~ /R/ {printf "%s/%s/%s ", $1, $4, $3}')"
}
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"
if [ "$(uname)" = Darwin ]; then
    echo "CPU $(sysctl -n machdep.cpu.brand_string) l2 $(sysctl -n hw.perflevel0.l2cachesize) page $(sysctl -n hw.pagesize)"
else
    echo "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)$(lscpu | grep -m1 'Model name' | cut -d: -f2-)"
    [ -n "$PINCPU" ] && for i in 0 2 3; do echo "CACHE cpu$PINCPU $(cat /sys/devices/system/cpu/cpu"$PINCPU"/cache/index$i/level) $(cat /sys/devices/system/cpu/cpu"$PINCPU"/cache/index$i/size)"; done
    echo "THP $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null) HUGE=${HUGE:-0}"
fi
echo "PIN '$PIN' PINCPU '$PINCPU' REPS $REPS buckets 2^$LO..2^$HI GIT ${GIT_SHA:-unknown}"
echo "CC $($CC --version | head -1)"
$SHA build/k5_sweep "$BD/libtoks.a" | sed 's/^/BIN /'
for tk in $TOKS_LIST; do echo "TOKENIZER $tk $($SHA "$K/$tk" | cut -c1-64)"; done
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        ml) echo "$(ls $T/wiki-*.txt | tr '\n' ' ')$T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt $T/gut-zh-24264.txt" ;;
    esac
}
for corp in $CORPORA; do
    # shellcheck disable=SC2046
    echo "CORPUS $corp $(cat $(files $corp) | $SHA | cut -c1-64) $(cat $(files $corp) | wc -c | tr -d ' ') bytes"
done
echo "UPTIME $(uptime)"
for tk in $TOKS_LIST; do
    for corp in $CORPORA; do
        echo "== $tk $corp"
        echo "LOAD before $(loadavg)"
        pinned
        # shellcheck disable=SC2086
        $PIN ./build/k5_sweep "$K/$tk" "$tk/$corp" "$REPS" "$LO" "$HI" $(files $corp)
        pinned
        echo "LOAD after $(loadavg)"
    done
done
echo "UPTIME $(uptime)"
