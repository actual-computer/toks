#!/bin/sh
# tests/data/breadth/probe_census.sh: tools/census/probe.c (toks's own load path, cross-checked against
# toks_load_mem_copy) over every tokenizer.json of the census cache with this branch's library, the way the
# census probes (a file refused for NFC is probed again with only its normalizer removed). Writes
# build/breadth/probe.tsv; tests/data/breadth/probe_flips.py compares it with census/coverage.json.
#   tools/remote.sh <host> 'sh tests/data/breadth/probe_census.sh'
set -eu
files=${CENSUS_FILES:-$HOME/.cache/toks/census/coverage/files}
isa=$(uname -m | sed 's/aarch64/arm64/')
os=$([ "$(uname)" = Darwin ] && echo macos || echo linux)
make -s -j8 lib >/dev/null
mkdir -p build/breadth/nfc-off
clang -std=c17 -O2 -Iinclude -Isrc/core -Isrc/platform tools/census/probe.c "build/$os-$isa/libtoks.a" \
    -o build/breadth/census-probe
: > build/breadth/probe.tsv
for f in "$files"/sha256_*; do
    head -c 1 "$f" | grep -q '{' || continue          # json files only (the cache also holds tiktoken files)
    build/breadth/census-probe "$f" >> build/breadth/probe.tsv
done
# NFC-refused files: again with only the normalizer removed
grep -F 'normalizer NFC (m1b)' build/breadth/probe.tsv | cut -f1 | while read -r f; do
    b=$(basename "$f")
    uv run -q python -c "import json,sys; o=json.load(open(sys.argv[1])); o['normalizer']=None; json.dump(o, open(sys.argv[2],'w'), ensure_ascii=False)" "$f" "build/breadth/nfc-off/$b"
    build/breadth/census-probe "build/breadth/nfc-off/$b" | sed 's/^/nfc-off:/' >> build/breadth/probe.tsv
done
wc -l build/breadth/probe.tsv
