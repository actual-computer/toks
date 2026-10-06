#!/bin/sh
# tools/bench/gate.sh: the gigatoken gate's runs on this host. One thread, the
# tier toks_load binds. Per cell (tokenizer x corpus x chunk) and cache config, ROUNDS blocks of A B B A: A =
# build/e2e (toks), B = gigatoken-bench, both on the same pinned cpu, REPS reps per run, every rep's seconds per state
# (e2e.c's and main.rs's <state>_reps_s). tools/bench/gate_table.py pairs (A, B) and (B, A) of each block rep i with
# rep i: 2 x REPS paired ratios a block, ROUNDS x 2 x REPS a cell (30 at the defaults), tokv1.c's boot over them, WIN
# iff the interval's low end > 1.00, LOSS iff its high end < 1.00, else TIE. States: cold, pass, lang-x (the OTHER text:
# every bench corpus file outside the measured corpus, tokv1.sh's rule, not a same-language warm state),
# warm. In the default config each round also runs a null block, A N N A: toks against itself
# (N = the same binary), which gate_table.py requires to read TIE on >= 95% of cells (the null calibration).
#
# The cold-pair rule (approved 2026-10-04): gigatoken's cold forks a vocabulary-seeded state
# before every call (5-60 ms, outside the timer; a 30-pair cold matrix of the chunked cells costs ~5.5 h on gb10a),
# so a chunked cell's B runs take B_COLD_REPS (1) cold reps: 2 cold pairs a block, 6 a cell, printed as n and flagged
# provisional (n < 30) by gate_table.py. Whole-corpus cells (one call) and every other state take the full REPS. The
# 30-pair cold matrix of the chunked cells is a release-time run (B_COLD_REPS=5).
#
# The void rule per block (as the gate enforces it, decided 2026-10-05): a block whose pinned
# cpu's smt sibling was > 5% busy over the block, or another cpu of its L3 domain > 50% (a build or test on the same
# CCD / cluster shares the cache and memory bandwidth the timed run reads), or whose 1-minute load rose by more than 2
# across it, runs once more; a second such block is VOID (gate_table.py drops its pairs and prints the cell's n).
# BLOCK lines carry the start / end times and every L3 cpu's busy share. macOS: unpinned, load only.
#
# Exactness, once per cell before its blocks (outside every timer): toks' ids sha-256 == hf's (tools/bench/e2e_ref.py),
# else the cell is VOID; gigatoken's ids == hf's without the post-processor, else its comparison is n/a.
#
# Cache configs (CONFIGS), the declared rows of the replay states (decided 2026-10-05): default (each tool's default
# caches: every state, the replays UNMATCHED: toks' 2 MiB piece cache + 4 MiB segment memo vs gigatoken's 512 MiB per
# state + its spm unit memo), m6 (toks' default vs GIGA_CACHE_MIB=6: gigatoken given toks' default cache bytes), m2
# (both 2 MiB, toks' memo off: E2E_MEMO_MIB=0 vs GIGA_CACHE_MIB=2, the piece-cache race alone; gigatoken floors a
# budget at its vocabulary seed plus headroom). The matched configs time warm and warmo only; their B runs skip cold.
#
#   PINCPU=7 tools/bench/gate.sh "taskset -c 7" > build/gate-gb10a.log 2>&1       (macOS: no pin)
#
# env: TOKS_LIST (the declared 11)  CORPORA (en code ml cjk)  CHUNKS (4096 0)  ROUNDS (3)  REPS (5)  B_COLD_REPS (1)
#      CONFIGS (default m6 m2)  GIT_SHA (remote.sh syncs without .git)  and common.sh's (corpus, tokenizers,
#      PINCPU). The gigatoken binary: tools/bench/gigatoken.sh.
set -e
PIN="$1"
. tools/bench/common.sh
TOKS_LIST=${TOKS_LIST:-"gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4 kimik3"}
CORPORA=${CORPORA:-"en code ml cjk"}
CHUNKS=${CHUNKS:-"4096 0"}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-5}
B_COLD_REPS=${B_COLD_REPS:-1}
CONFIGS=${CONFIGS:-"default m6 m2"}
GIGA_BIN=${GIGA_BIN:-build/giga-target/release/gigatoken-bench}
CC=${CC:-clang}
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-only-managed}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$PWD/build/uv-python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/build/uv-cache}
[ -x "$GIGA_BIN" ] || { echo "gate.sh: $GIGA_BIN missing (tools/bench/gigatoken.sh builds it)" >&2; exit 1; }
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }

make -j8 lib >/dev/null
BD=$(mkvar BUILD_DIR)
case $(mkvar ISA) in x86_64) NATIVE=-march=native ;; *) NATIVE=-mcpu=native ;; esac
$CC -std=c17 -O3 $NATIVE -Wall -Wextra -Werror $(mkvar CPPFLAGS) -o build/e2e tools/bench/e2e.c "$BD/libtoks.a" -lpthread

host_lines
echo "GATE PIN '$PIN' PINCPU '${PINCPU:-}' ROUNDS $ROUNDS REPS $REPS B_COLD_REPS $B_COLD_REPS CONFIGS '$CONFIGS'"
echo "GIT ${GIT_SHA:-unknown}"
echo "CC $($CC --version | head -1)"
echo "KERNELS $(cat "$BD/have.txt")"
$SHA build/e2e "$BD/libtoks.a" "$GIGA_BIN" | sed 's/^/BIN /'
echo "GIGATOKEN $(cat build/third-party/gigatoken/.toks-rev 2>/dev/null || echo unknown) $(cat build/giga-target/rustc.txt 2>/dev/null)"
for tk in $TOKS_LIST; do
    p=$(tpath "$tk")
    if [ -e "$p" ]; then echo "TOKENIZER $tk $(tsha "$p")"; else echo "TOKENIZER $tk SKIP missing $p"; fi
done
corpus_lines $CORPORA
echo "UPTIME $(uptime)"

cfg_env() {   # the env of side $2 (A or B) in config $1
    case "$1/$2" in
        m6/B) echo "GIGA_CACHE_MIB=6" ;;
        m2/A) echo "E2E_MEMO_MIB=0" ;;
        m2/B) echo "GIGA_CACHE_MIB=2" ;;
    esac
}
run() {   # one run of side $1 (A, B, or N: toks again, the null), its result lines tagged
    tag="RUN cfg=$cfg tk=$tk corp=$corp chunk=$ch round=$r try=$try kind=$kind pos=$pos side=$1"
    # shellcheck disable=SC2046,SC2086
    case $1 in
        A|N) env $(cfg_env "$cfg" A) E2E_WARM_ON="$W" $PIN ./build/e2e "$P" "$ch" "$REPS" $F | grep '^E2E\|^CTR' ;;
        B) env $(cfg_env "$cfg" B) GIGA_COLD_REPS="$bcold" GIGA_WARM_ON="$W" $PIN "$GIGA_BIN" "$P" "$ch" "$REPS" \
               /dev/null $F 2>/dev/null | grep '^GIGA' || echo "GIGA tool=gigatoken na=exit" ;;
    esac | sed "s/^/$tag /"
}
l3cpus() {   # the cpus sharing the pinned cpu's last-level cache (linux), one per line
    [ -n "${PINCPU:-}" ] || return 0
    cat /sys/devices/system/cpu/cpu"$PINCPU"/cache/index3/shared_cpu_list 2>/dev/null | tr ',' '\n' |
        awk -F- '{if (NF == 2) for (i = $1; i <= $2; i++) print i; else if ($1 != "") print $1}'
}
l3stat() {   # "cpuN busy total" jiffies of every cpu of the L3 domain
    for c in $(l3cpus); do
        awk -v c="cpu$c" '$1 == c {b = $2 + $3 + $4 + $7 + $8 + $9; t = b + $5 + $6; print c, b, t}' /proc/stat
    done
}
block() {   # the sides $1 in order (a block of kind $2), then its receipts; sets bad=1 when the void rule fires
    kind=$2
    t0=$(date +%s)
    l0=$(load1)
    s0=$(cpustat)
    c0=$(l3stat)
    pos=0
    for side in $1; do pos=$((pos + 1)); run "$side"; done
    l1=$(load1)
    b=$(busy "$s0" "$(cpustat)")
    c3=$(busy "$c0" "$(l3stat)")
    sib=$(echo "$b" | awk -v p="cpu${PINCPU:-x}" '{m = 0; for (i = 2; i < NF; i += 2) if ($i != p) {v = $(i + 1) + 0; if (v > m) m = v} print m}')
    sibs=" cpu${PINCPU:-x} $(echo "$b" | awk '{for (i = 2; i < NF; i += 2) printf "%s ", $i}')"
    l3=$(echo "$c3" | awk -v s="$sibs" '{m = 0; for (i = 2; i < NF; i += 2) if (index(s, " " $i " ") == 0) {v = $(i + 1) + 0; if (v > m) m = v} print m}')
    bad=$(awk -v a="$l0" -v c="$l1" -v s="${sib:-0}" -v l="${l3:-0}" 'BEGIN {print (c - a > 2 || s > 5 || l > 50) ? 1 : 0}')
    echo "BLOCK cfg=$cfg tk=$tk corp=$corp chunk=$ch round=$r try=$try kind=$2 t0=$t0 t1=$(date +%s) load0=$l0 load1=$l1 sib_busy=${sib:-na} l3_busy=${l3:-na} bad=$bad $b L3 $c3"
}
guarded() {   # a block under the void rule: once more if it fired, then VOID if it fires again
    try=1
    block "$1" "$2"
    if [ "$bad" = 1 ]; then
        try=2
        block "$1" "$2"
        [ "$bad" = 1 ] && echo "VOID cfg=$cfg tk=$tk corp=$corp chunk=$ch round=$r kind=$2"
    fi
    return 0
}

for tk in $TOKS_LIST; do
    P=$(tpath "$tk")
    [ -e "$P" ] || continue
    for corp in $CORPORA; do
        F=$(files "$corp")
        W=$(others "$corp")
        for ch in $CHUNKS; do
            echo "== $tk $corp chunk $ch"
            rm -f build/ids.u32 build/giga.u32
            # shellcheck disable=SC2086
            a=$(E2E_IDS_OUT=build/ids.u32 $PIN ./build/e2e "$P" "$ch" 1 $F | sed -n 's/^E2E .* ids=\([0-9]*\) sha=\([0-9a-f]*\).*/\1 \2/p')
            # shellcheck disable=SC2086
            GIGA_COLD_REPS=0 $PIN "$GIGA_BIN" "$P" "$ch" 1 build/giga.u32 $F >/dev/null 2>&1 || true
            G=""
            [ -s build/giga.u32 ] && G="--giga build/giga.u32"
            # shellcheck disable=SC2086
            ref=$($PIN uv run -q tools/bench/e2e_ref.py "$P" "$ch" 1 $F --ids build/ids.u32 $G)
            h=$(echo "$ref" | sed -n 's/^REF tool=hf .* ids=\([0-9]*\) sha=\([0-9a-f]*\).*/\1 \2/p')
            g=$(echo "$ref" | sed -n 's/^REF tool=gigatoken .* same_as_hf_nopp=\([a-z]*\).*/\1/p')
            t=no
            if [ -n "$a" ] && [ "$a" = "$h" ]; then t=yes; fi
            echo "EXACT tk=$tk corp=$corp chunk=$ch toks=$t gigatoken=${g:-na} toks_ids='$a' hf_ids='$h'"
            [ "$t" = yes ] || continue
            for cfg in $CONFIGS; do
                if [ "$cfg" != default ]; then bcold=0; elif [ "$ch" = 0 ]; then bcold=$REPS; else bcold=$B_COLD_REPS; fi
                r=1
                while [ "$r" -le "$ROUNDS" ]; do
                    guarded "A B B A" gate
                    if [ "$cfg" = default ]; then guarded "A N N A" null; fi
                    r=$((r + 1))
                done
            done
        done
    done
done
echo "UPTIME $(uptime)"
