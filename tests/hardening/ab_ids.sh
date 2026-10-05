#!/bin/sh
# tests/hardening/ab_ids.sh <old tree> <n> <tok[:literal,literal...]>...: ids / pieces digests of n random texts
# per tokenizer from the library in <old tree> (e.g. a `git archive` of master under build/) and from this tree;
# prints the first differing lines per tokenizer and a summary. Run from the repository root on a host
# (docs/hardening.md: every semantics-preserving fix carries this receipt). Literals use idsdiff's escapes.
set -eu
old=$1; n=$2; shift 2
isa=$(uname -m | sed 's/aarch64/arm64/;s/arm64/arm64/')
case "$isa" in arm64) bd=linux-arm64 ;; *) bd=linux-x86_64 ;; esac
[ "$(uname -s)" = Darwin ] && bd=macos-$isa
pin=${MEAS:+taskset -c $MEAS}
(cd "$old" && $pin make -s -j2 lib >/dev/null) && $pin make -s -j2 lib >/dev/null
mkdir -p build/abids
clang -O2 -std=c17 -Iinclude tests/hardening/idsdiff.c "$old/build/$bd/libtoks.a" -lm -lpthread -o build/abids/old
clang -O2 -std=c17 -Iinclude tests/hardening/idsdiff.c "build/$bd/libtoks.a" -lm -lpthread -o build/abids/new
T=${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}
bad=0
for spec in "$@"; do
    tok=${spec%%:*}
    lits=""
    [ "$spec" != "$tok" ] && lits=$(echo "${spec#*:}" | tr ',' ' ')
    f=$tok
    [ -e "$f" ] || f=$T/$tok
    # shellcheck disable=SC2086
    $pin build/abids/old "$f" "$n" 7 $lits > build/abids/a.txt || true
    # shellcheck disable=SC2086
    $pin build/abids/new "$f" "$n" 7 $lits > build/abids/b.txt || true
    if [ "$(wc -l < build/abids/a.txt)" -le 1 ]; then echo "$tok: $(cat build/abids/a.txt) (skipped)"; continue; fi
    d=$(diff build/abids/a.txt build/abids/b.txt | grep -c '^>' || true)
    echo "$tok: $(wc -l < build/abids/a.txt) texts, $d differ"
    [ "$d" = 0 ] || { bad=1; diff build/abids/a.txt build/abids/b.txt | head -6; }
done
exit $bad
