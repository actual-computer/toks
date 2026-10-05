#!/bin/sh
# tests/bpe/run.sh: K6 / K5 against hf 0.23.2's own BPE model on a real tokenizer (gen.py | check.c).
#
#   tests/bpe/run.sh TOKENIZER_JSON N [SEED [TEXT ...]]      N generated pieces + the vocabulary + TEXT pieces
#
# Small runs only on the control-plane mac; big ones on a lab host, e.g.
#   tools/remote.sh <host> 'WORKERS=16 tests/bpe/run.sh build/tokenizers/llama3 5000000 1 build/text/*'
#
# Passes only when gen.py exits 0, check exits 0, and check compared > 0 pieces: a generator that dies mid-stream
# feeds check a short (or empty) stream, which alone would read as a pass. The last line is the count line:
# "PASS <pieces> compared bpe <tokenizer> (...)" or "FAIL <pieces> compared bpe <tokenizer> (<why>)".
set -eu
cd "$(dirname "$0")/../.."
tok=$1
shift
B=build/bpe-check
BUILD_DIR=$B make -s -j8 lib >/dev/null
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform \
  -o $B/check tests/bpe/check.c $B/libtoks.a
t=$B/$(basename "$tok")
echo 0 > "$t.gen.exit"
echo 0 > "$t.check.exit"
{ uv run --with tokenizers==0.23.2 python tests/bpe/gen.py "$tok" "$@" || echo $? > "$t.gen.exit"; } |
    { $B/check "$tok" || echo $? > "$t.check.exit"; } | tee "$t.check.out"
g=$(cat "$t.gen.exit") c=$(cat "$t.check.exit")
n=$(sed -n 's/^K6: \([0-9]*\) pieces.*/\1/p' "$t.check.out")
why=""
[ "$g" = 0 ] || why="gen.py exit $g"
[ "$c" = 0 ] || why="${why:+$why; }check exit $c"
[ "${n:-0}" -gt 0 ] || why="${why:+$why; }no pieces compared"
if [ -z "$why" ]; then
    echo "PASS $n compared bpe $(basename "$tok") (K6 pieces; $(sed -n 's/^K5: \([0-9]*\) batches.*/K5 \1 batches/p' "$t.check.out"))"
else
    echo "FAIL ${n:-0} compared bpe $(basename "$tok") ($why)"
    exit 1
fi
