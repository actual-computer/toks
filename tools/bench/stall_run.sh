#!/bin/sh
# tools/bench/stall_run.sh <out> <tok...>: the stall audit over tokenizers on the measurement cpu ($MEAS), run from
# the tree this script sits in (tools/remote.sh syncs it to <host>:~/toks-ci/<branch>/)
out=$1; shift
cd "$(dirname "$0")/../.." || exit 1
T=$HOME/.cache/toks/tokenizers
{
    echo "# host ${TOKS_HOST_KEY:-$(uname -m)} cpu $MEAS start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
    for t in "$@"; do
        taskset -c "$MEAS" build/stall -s ${SIZES:-4096,65536,1048576} -r ${REPS:-3} -T ${TMO:-120} -e build/text/gut-en-1342.txt \
            ${CLASSES:+-c $CLASSES} ${OP:+-o $OP} ${TIER:+-t $TIER} "$T/$t" | grep -v '^#\|^tokenizer'
    done
    echo "# end $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$out" 2>&1
