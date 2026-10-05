#!/bin/sh
# tests/k3/dsv3_mutants.sh [REPO]: one-line mutants of the dsv3 c twin and its class build, each in a scratch copy
# (build/dsv3-mut/mN); tests/c/test_k3_dsv3.c must fail on every one ("killed"). Seconds on any host.
set -u
REPO=$(cd "${1:-.}" && pwd)
W=${MUT_DIR:-$REPO/build/dsv3-mut}
LIB=$(ls "$REPO"/build/*/libtoks.a | head -1)   # the rest of the library (make lib first): the mutated files win
rm -rf "$W"; mkdir -p "$W"
i=0
mut() {   # file, from, to
  i=$((i + 1))
  d="$W/m$i"; mkdir -p "$d"
  cp -R "$REPO/src" "$REPO/include" "$REPO/tests" "$d/"
  python3 - "$d/$1" "$2" "$3" <<'EOF'
import sys
p, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if s.count(a) < 1:
    print("NOMATCH", a); sys.exit(3)
s = s.replace(a, b, 1)
open(p, "w").write(s)
EOF
  [ $? -eq 0 ] || { echo "m$i: pattern not found"; return; }
  ( cd "$d" && clang -std=c17 -O1 -fno-strict-aliasing -fwrapv -w -Iinclude -Isrc/core -Isrc/platform -Itests/common \
      -o t tests/c/test_k3_dsv3.c src/core/k3_dsv3_c.c src/core/classes.c src/gen/ucd_flags.c src/gen/han_ranges.c tests/common/guard.c \
      tests/common/abicheck_$(uname -m | sed -e s/aarch64/arm64/ -e s/amd64/x86_64/).S "$LIB" -lpthread 2> build.log ) ||
      { echo "m$i: BUILD FAILED ($1: $2)"; tail -3 "$d/build.log"; return; }
  out=$("$d/t" 2>&1 | tail -1)
  if "$d/t" > /dev/null 2>&1; then echo "m$i: SURVIVED  ($1: $2 -> $3)"; else echo "m$i: killed    ($out)"; fi
}
mut src/core/kernels.h 'for (uint32_t c = 1u; c < 3u && e < len; c++)' 'for (uint32_t c = 1u; c < 4u && e < len; c++)'
mut src/core/k3_dsv3_c.c 'while (end < len && text[end] < 0x80u && asc[text[end]] == TOKS_C_L)' 'while (end < len && asc[text[end] & 0x7Fu] == TOKS_C_L)'
mut src/core/k3_dsv3_c.c 'end = toks_d3_run(t, text, len, p1, (uint8_t)(TOKS_C_L | r), &last, &stop);' 'end = toks_d3_run(t, text, len, p1, TOKS_C_L, &last, &stop);'
mut src/core/k3_dsv3_c.c 'end = r != 0u ? end : toks_k3_run(' 'end = 0u != 0u ? end : toks_k3_run('
mut src/core/k3_dsv3_c.c 'if (text[pos] == 0x20u && c1 == TOKS_C_P) {' 'if (c1 == TOKS_C_P) {'
mut src/core/k3_dsv3_c.c 'TOKS_TP_DIGIT_CUT | TOKS_C_CJK);' 'TOKS_TP_DIGIT_CUT);'
mut src/core/k3_dsv3_c.c 'TOKS_TP_DIGIT_CUT | TOKS_C_CJK);' 'TOKS_C_CJK);'
mut src/core/k3_dsv3_c.c 'end = last;                                 /* D7' 'end = end;                                 /* D7'
mut src/core/k3_dsv3_c.c 'if (b == TOKS_C_WS && c1 == TOKS_C_L) {' 'if (b == TOKS_C_WS && (c1 & TOKS_C_BASE_MASK) == TOKS_C_L) {'
mut src/core/k3_dsv3_c.c 'if (c1 == (uint8_t)(TOKS_C_L | r)) {' 'if ((c1 & TOKS_C_BASE_MASK) == TOKS_C_L) {'
mut src/core/kernels.h 'return n == 0u ? (uint8_t)TOKS_C_P :' 'return n == 0u ? (uint8_t)TOKS_C_X :'
mut src/core/kernels.h 'if (b == TOKS_C_NL) { last_nl = e + k; }' 'if (b == TOKS_C_NL) { last_nl = e; }'
mut src/core/kernels.h 'return e == len || last_start == pos ||' 'return 1 ||'
mut src/core/k3_dsv3_c.c 'if (text[pos] < 0x80u && p1 < len && text[p1] < 0x80u && asc[text[p1]] == TOKS_C_L) {' 'if (p1 < len && text[p1] < 0x80u && asc[text[p1]] == TOKS_C_L) {'
mut src/core/k3_dsv3_c.c 'if (stop == (uint8_t)(TOKS_C_L | r) && last > pos) {' 'if ((stop & TOKS_C_BASE_MASK) == TOKS_C_L && last > pos) {'
mut src/core/classes.c '{ 0x4E00u, 0x9FA5u }' '{ 0x4E00u, 0x9FA6u }'
mut src/core/classes.c 'if ((f & (TOKS_UCD_P | TOKS_UCD_S)) != 0) {' 'if ((f & TOKS_UCD_P) != 0) {'
mut src/core/classes.c 'if ((f & (TOKS_UCD_LETTERS | TOKS_UCD_M)) != 0) {
        b = TOKS_CLS_L;
    } else if ((f & TOKS_UCD_N) != 0) {
        b = TOKS_CLS_N;
    } else if (cp == 0x0Au' 'if ((f & TOKS_UCD_LETTERS) != 0) {
        b = TOKS_CLS_L;
    } else if ((f & TOKS_UCD_N) != 0) {
        b = TOKS_CLS_N;
    } else if (cp == 0x0Au'
