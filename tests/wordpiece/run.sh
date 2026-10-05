#!/bin/sh
# tests/wordpiece/run.sh: the c twins (toks_bert_normalize, toks_wp_scan_c, toks_wp_encode_c) against hf 0.23.2 on
# a real WordPiece tokenizer.json (gen.py | check.c). SANITIZE=1 builds both with asan + ubsan.
#
#   tests/wordpiece/run.sh TOKENIZER_JSON N_GENERATED [SEED [TEXT ...]]
#
# Small runs only on the control-plane mac; big ones on a lab host, e.g.
#   tools/remote.sh <host> 'tests/wordpiece/run.sh ~/.cache/toks/tokenizers/wp-minilm-l6 200000 1 build/text/flores.txt'
set -eu
cd "$(dirname "$0")/../.."
tok=$1
shift
if [ "${SANITIZE:-0}" = 1 ]; then
  SAN="-fsanitize=address,undefined -fno-sanitize-recover=all -g"
  BD=build/wp-check-san
else
  SAN=""
  BD=build/wp-check
fi
BUILD_DIR=$BD make -s -j8 $BD/libtoks.a CSTRICT="-std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Werror $SAN" >/dev/null
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror $SAN -Iinclude -Isrc/core -Isrc/platform \
  -o $BD/check tests/wordpiece/check.c $BD/libtoks.a
uv run --with tokenizers==0.23.2 python tests/wordpiece/gen.py "$tok" "$@" | $BD/check "$tok"
