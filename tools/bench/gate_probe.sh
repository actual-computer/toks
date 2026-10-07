#!/bin/sh
# tools/bench/gate_probe.sh: what gate.sh's m6 config costs per tokenizer, the receipt behind its M6_LIST. Per
# tokenizer, gigatoken-bench once at GIGA_CACHE_MIB=6 and once at its default budget (chunk 4096, corpus en, one rep,
# every state but cold, the OTHER text as the gate's warm-up): the wall time of the whole run (its untimed passes
# included: evidence of cost, not a speed cell), the entries its fresh state starts from (its vocabulary seed) and the
# entries after warmo. tools/bench/gate_table.py --probe <log> writes the gate doc's m6 paragraph from it. From the
# gate tree's root, on linux (date +%N), after tools/bench/gigatoken.sh:
#
#   PINCPU=7 tools/bench/gate_probe.sh "taskset -c 7" > build/gate-probe.log 2>&1
#
# env: TOKS_LIST (the declared 11), GIGA_BIN, GIT_SHA, and common.sh's (corpus, tokenizers, PINCPU).
set -e
PIN="$1"
. tools/bench/common.sh
TOKS_LIST=${TOKS_LIST:-"gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4 kimik3"}
GIGA_BIN=${GIGA_BIN:-build/giga-target/release/gigatoken-bench}
[ -x "$GIGA_BIN" ] || { echo "gate_probe.sh: $GIGA_BIN missing (tools/bench/gigatoken.sh builds it)" >&2; exit 1; }
F=$(files en)
W=$(others en)
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"
echo "GIT ${GIT_SHA:-$(cat .toks-rev 2>/dev/null || echo unknown)}"
$SHA "$GIGA_BIN" | sed 's/^/BIN /'
# shellcheck disable=SC2086
echo "PROBE cpu ${PINCPU:-none}, reps 1, chunk 4096, corpus en ($(cat $F | wc -c | tr -d ' ') bytes)," \
    "GIGA_COLD_REPS 0, OTHER text $(cat $W | wc -c | tr -d ' ') bytes"
echo "UPTIME $(uptime)"
KEYS='cache_entries=[0-9]*\|cache_mib=[0-9]*\|pass_s=[0-9.]*\|warmo_s=[0-9.]*\|entries_warmo=[0-9]*'
for tk in $TOKS_LIST; do
    P=$(tpath "$tk")
    for m in 6 default; do
        e=""; [ "$m" = default ] || e="GIGA_CACHE_MIB=$m"
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        l=$(env $e GIGA_COLD_REPS=0 GIGA_WARM_ON="$W" $PIN "$GIGA_BIN" "$P" 4096 1 /dev/null $F | grep '^GIGA')
        t1=$(date +%s.%N)
        echo "PROBE tk=$tk budget_mib=$m wall_s=$(echo "$t1 - $t0" | bc | sed 's/^\./0./' | cut -c1-6)" \
            "$(echo "$l" | grep -o "$KEYS" | tr '\n' ' ')"
    done
done
echo "UPTIME $(uptime)"
