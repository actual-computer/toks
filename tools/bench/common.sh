#!/bin/sh
# tools/bench/common.sh: what the bench scripts share, sourced (. tools/bench/common.sh) from the source root by
# e2e.sh, gate.sh, e2e_ab.sh and e2e_commits.sh: the corpora (one list), the OTHER text of the pass and lang-x states,
# the tokenizer paths, and the void rule's receipts (load, the pinned cpu's other threads, busy shares).
# env: TOKS_BENCH_TEXT (build/text: the corpus files, tools/bench/corpus.sha256; the default is verified and seeded
#      from ~/.cache/toks/bench-text by tools/release/rc_host.sh's corpus step)  TOKS_TOKENIZER_DIR
#      (~/.cache/toks/tokenizers)  TOKS_KIMI_DIR (~/.cache/toks/kimik3)  PINCPU (the pinned cpu, linux receipts)
T=${TOKS_BENCH_TEXT:-build/text}
[ -n "${TOKS_BENCH_TEXT:-}" ] || sh tools/release/rc_host.sh corpus >&2 || exit 1
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
tpath() { case $1 in kimik3) echo "${TOKS_KIMI_DIR:-$HOME/.cache/toks/kimik3}" ;; *) echo "$K/$1" ;; esac; }
tsha() { if [ -d "$1" ]; then cat "$1/tiktoken.model" "$1/tokenizer_config.json" | $SHA | cut -c1-64; else $SHA "$1" | cut -c1-64; fi; }
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        en1) echo "$T/gut-en-2701.txt" ;;   # one book of en's two (1.28 MB): records past half a 4 MiB memo's ring
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        ml) echo "$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt $T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt" ;;
        cjk) echo "$T/wiki-zh.txt $T/wiki-ja.txt $T/wiki-ko.txt $T/gut-zh-24264.txt" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
    esac
}
others() {   # the OTHER text of the pass and lang-x states: every corpus file outside corpus $1 (tokv1.sh's rule)
    o=""
    for c in en code ml cjk; do for f in $(files $c); do
        case " $(files "$1") $o " in *" $f "*) ;; *) o="$o $f" ;; esac
    done; done
    echo $o
}
corpus_lines() {   # CORPUS receipts for corpora $*; a missing file stops the run (it once printed 0-byte void cells)
    for corp in "$@"; do
        for f in $(files "$corp"); do
            [ -s "$f" ] || { echo "$f missing (tools/release/rc_host.sh corpus seeds build/text)" >&2; exit 1; }
        done
        # shellcheck disable=SC2046
        echo "CORPUS $corp $(cat $(files $corp) | $SHA | cut -c1-64) $(cat $(files $corp) | wc -c | tr -d ' ') bytes: $(files $corp)"
    done
}
loadavg() { if [ -r /proc/loadavg ]; then cut -d' ' -f1-3 /proc/loadavg; else sysctl -n vm.loadavg | tr -d '{}'; fi; }
load1() { loadavg | awk '{print $1}'; }   # the 1-minute load alone
siblings() { cat /sys/devices/system/cpu/cpu"$PINCPU"/topology/thread_siblings_list 2>/dev/null || echo "$PINCPU"; }
pinned() {   # other threads on the pinned cpu (linux), for the void rule
    [ -n "${PINCPU:-}" ] && [ -r /proc/loadavg ] || return 0
    sib=$(siblings)
    echo "PINNED cpu $PINCPU siblings $sib: $(ps -eLo psr,stat,pcpu,comm --no-headers | awk -v s="$sib" \
        'BEGIN{n=split(s,a,/[,-]/); for(i=1;i<=n;i++) c[a[i]]=1} ($1 in c) && $2 ~ /R/ {printf "%s/%s/%s ", $1, $4, $3}')"
}
cpustat() {  # "busy total" jiffies of the pinned cpu and its smt siblings (linux), for the busy-share receipt
    [ -n "${PINCPU:-}" ] && [ -r /proc/stat ] || return 0
    for c in $(siblings | tr ',' ' '); do
        awk -v c="cpu$c" '$1 == c {b = $2 + $3 + $4 + $7 + $8 + $9; t = b + $5 + $6; print c, b, t}' /proc/stat
    done
}
busy() {  # busy share of every pinned cpu / sibling between two cpustat snapshots
    [ -n "$1" ] || return 0
    echo "BUSY $(printf '%s\n%s\n' "$1" "$2" | awk '{if ($1 in b) {db = $2 - b[$1]; dt = $3 - t[$1]; v = 0; if (dt > 0) v = 100 * db / dt; printf "%s %.0f%% ", $1, v} else {b[$1] = $2; t[$1] = $3}}')"
}
rel() { case $1 in "$HOME"/*) echo "\$HOME/${1#"$HOME"/}" ;; *) echo "$1" ;; esac; }   # no home dir in receipts
host_lines() {   # the host fingerprint: the chipset key (TOKS_HOST_KEY, docs/machines.md; default uname -m), never a hostname
    echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"
    if [ "$(uname)" = Darwin ]; then
        echo "CPU $(sysctl -n machdep.cpu.brand_string) p-cores $(sysctl -n hw.perflevel0.physicalcpu) e-cores $(sysctl -n hw.perflevel1.physicalcpu)"
        echo "OS $(sw_vers -productVersion) $(sw_vers -buildVersion)"
    else
        echo "CPU $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) microcode $(grep -m1 microcode /proc/cpuinfo | cut -d: -f2-)"
        echo "OS $(uname -v)"
        [ -n "${PINCPU:-}" ] && echo "GOVERNOR cpu$PINCPU $(cat /sys/devices/system/cpu/cpu"$PINCPU"/cpufreq/scaling_governor 2>/dev/null) boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
        echo "THP $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    fi
}
