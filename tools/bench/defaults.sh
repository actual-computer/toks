#!/bin/sh
# tools/bench/defaults.sh <pin>: master's e2e (build/e2e-A, ab2.sh builds it) over the scratch's piece-cache size and the
# segment memo, 4 KiB calls: cold / pass / warm / lang MB/s (lang = an untimed pass over 20 MB of OTHER text of the same
# language first, build/corp/lang-<corpus>: e2e.c E2E_WARM_ON), to pick the fastest defaults (docs/usage.md).
PIN="$1"
T=build/text
K=$HOME/.cache/toks/tokenizers
TOKS_LIST=${TOKS_LIST:-"llama3"}
CORPORA=${CORPORA:-"en code zh"}
CACHES=${CACHES:-"0 4 8 16 32"}
MEMOS=${MEMOS:-"0 4"}
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
    esac
}
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} PIN '$PIN' UPTIME $(uptime)"
for tk in $TOKS_LIST; do
    for c in $CORPORA; do
        F=$(files $c)
        bytes=$(cat $F | wc -c)
        for mm in $MEMOS; do
            for cm in $CACHES; do
                # shellcheck disable=SC2086
                line=$(E2E_CACHE_MIB=$cm E2E_MEMO_MIB=$mm E2E_WARM_ON="build/corp/lang-$c" $PIN ./build/e2e-A "$K/$tk" 4096 5 $F | grep '^E2E')
                v=""
                for st in cold pass warm lang; do
                    sec=$(echo "$line" | sed "s/.* ${st}_s=\([0-9.]*\).*/\1/")
                    v="$v $st $(echo "$bytes $sec" | awk '{printf "%.0f", $1 / $2 / 1e6}')"
                done
                echo "$tk $c memo_mib=$mm cache_mib=$cm:$v MB/s load $(cut -d' ' -f1 /proc/loadavg)"
            done
        done
    done
done
