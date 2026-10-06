#!/bin/sh
# tests/hardening/stall_mutant.sh: test_stall's teeth (docs/hardening.md §2). A copy of src/, include/, the Makefile,
# tests/common/ and tests/c/test_stall.c under build/stall-mutant/ with a quadratic walk put in front of toks_encode:
# a text that starts with '<' first spins len * len / D times (D = the argument, default 128; the review's mutant),
# a term that grows 16x per 4x the bytes. The mutant's test_stall runs from the repository root, on the same fixtures
# and pinned tokenizers, and must fail the class lt ('<' x n) on every tokenizer it loads (added_cut fails too where
# its text starts with '<'). It prints the FAIL lines, the count by kind and the wall time, and exits 0 when every
# loaded tokenizer's lt failed. From the repository root, on any host with a C toolchain (minutes: a quadratic's
# 2 MiB calls take seconds each):
#   sh tests/hardening/stall_mutant.sh [D]
set -eu
D=${1:-128}
case "$D" in '' | *[!0-9]*) echo "stall_mutant: D is a positive integer, not '$D'" >&2; exit 2 ;; esac
M=build/stall-mutant
rm -rf "$M" && mkdir -p "$M/tests/c"
cp -R src include Makefile "$M/" && cp -R tests/common "$M/tests/" && cp tests/c/test_stall.c "$M/tests/c/"
call='    return run(ctx, text, len, flags, out, cap, scr, 1);'
grep -qxF "$call" "$M/src/core/api.c" || { echo "stall_mutant: toks_encode's body is not in src/core/api.c" >&2; exit 1; }
sed -i.orig "s|^    return run(ctx, text, len, flags, out, cap, scr, 1);\$|    if (len > 0u \&\& *(const unsigned char *)text == '<') { volatile uint64_t q = 0; for (uint64_t i = 0; i < len * len / ${D}u; i++) { q += i; } } return run(ctx, text, len, flags, out, cap, scr, 1);|" "$M/src/core/api.c"
grep -qF "len * len / ${D}u" "$M/src/core/api.c" || { echo "stall_mutant: the walk is not in the copy" >&2; exit 1; }
(cd "$M" && make -s -j8 BUILD_DIR=out out/tests/test_stall > build.log 2>&1) ||
    { tail -5 "$M/build.log" >&2; echo "stall_mutant: the mutant's test_stall did not build" >&2; exit 1; }
bin=$M/out/tests/test_stall
echo "stall_mutant: '<'-led texts spin len * len / $D first ($bin); test_stall running $(date -u +%H:%M:%SZ)"
t0=$(date +%s)
"$bin" > "$M/stall.txt" 2>&1 || true
t1=$(date +%s)
grep -E '^  (suspect|FAIL) ' "$M/stall.txt" || true
loaded=$(grep -E '^  [^ ]+ +en +[0-9.]+ ns/B' "$M/stall.txt" | awk '{ print $1 }')
nl=0; nf=0; miss=""
for t in $loaded; do
    nl=$((nl + 1))
    if grep -qE "^  FAIL $t lt: " "$M/stall.txt"; then nf=$((nf + 1)); else miss="$miss $t"; fi
done
echo "stall_mutant D=$D: $(grep -c '^  FAIL .*superlinear' "$M/stall.txt" || true) failures by growth," \
     "$(grep -c '^  FAIL .* ns/B = ' "$M/stall.txt" || true) by the bound; lt failed on $nf of $nl tokenizers loaded" \
     "${miss:+(passed on:$miss)}; $(tail -1 "$M/stall.txt"); $((t1 - t0)) s"
[ "$nf" = "$nl" ] && [ "$nl" -gt 0 ]
