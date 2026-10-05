#!/bin/sh
# tests/parity/dsv3_parity.sh [SHARDS] [TOKENIZER]: deepseek v3 end to end through toks.h against hf tokenizers
# 0.23.2 -- the full parity set (tests/parity/gen_cases.py) for the pinned DeepSeek-V3 tokenizer.json,
# run by tests/parity/shards.sh through tests/driver/toks_driver.c on this host's library, in a run directory of
# its own (tools/remote.sh's rsync --delete never pulls the harness out from under it). Writes
# build/dsv3-parity/{cases.jsonl, report.json}; exit 0 iff no shard found a diff.
#   tools/remote.sh <host> 'tests/parity/dsv3_parity.sh 8 build/dsv3.json'
set -eu
cd "$(dirname "$0")/../.."
SHARDS=${1:-8}
TOK=$(cd "$(dirname "${2:-$HOME/.cache/toks/tokenizers/dsv3}")" && pwd)/$(basename "${2:-dsv3}")
[ -f "$TOK" ] || { echo "dsv3_parity: $TOK missing"; exit 2; }
OS=$(uname -s | tr A-Z a-z | sed s/darwin/macos/)
ISA=$(uname -m | sed -e s/aarch64/arm64/ -e s/amd64/x86_64/)
RUN=build/dsv3-parity
rm -rf "$RUN" && mkdir -p "$RUN"
make -j8 lib > "$RUN/make.log" 2>&1
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude \
  -o "$RUN/toks_driver" tests/driver/toks_driver.c "build/$OS-$ISA/libtoks.a"
cp -R tests python "$RUN/"
cd "$RUN"
uv run -q --with tokenizers==0.23.2 python tests/parity/gen_cases.py --tokenizer "$TOK" --out cases.jsonl > gen.log 2>&1
tail -3 gen.log
tests/parity/shards.sh "$TOK" cases.jsonl ./toks_driver out "$SHARDS"
