#!/bin/sh
# tests/k3/run.sh: generate hf-oracle cases and check the K3 c twin against them (local, small run).
# Big runs (millions of cases) go through tools/remote.sh on a lab host.
#
# usage: tests/k3/run.sh [N_SHORT] [N_LONG] [SEED]
set -eu
cd "$(dirname "$0")/../.."
N_SHORT=${1:-20000}
N_LONG=${2:-2000}
SEED=${3:-1}
OUT=build/k3cases.bin
mkdir -p build
uv run --with tokenizers==0.23.2 python tests/k3/gen.py "$N_SHORT" "$N_LONG" "$SEED" > "$OUT"
clang -std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror \
  -Iinclude -Isrc/core -Isrc/platform -o build/k3check tests/k3/check.c \
  src/core/*.c src/platform/*.c src/gen/*.c
./build/k3check "$OUT"
