#!/bin/sh
# T10 size budgets (SPEC §15), one row per budget, exit 1 when any is exceeded.
#
#   tools/size.sh [build_dir]     (make size runs it on the host build; make test runs make size)
#
# Lines are CODE lines of the hand-written sources: blank lines and comment-only lines (starting with //, /*, *
# or ;) do not count (docs/design.md d18). With a build directory it also sums the objects' machine code
# (__text / .text*) and the generated tables' data (src/gen objects: __const / __data / __cstring / .rodata* /
# .data*), the binary half of T10.
set -u
B=${1:-}
fail=0

row() {   # label value budget unit (lines are reported, not gated)
    if [ "$2" -le "$3" ]; then st=ok; else st=OVER; [ "$4" = lines ] || fail=1; fi
    printf '  %-46s %9d / %9d %-5s %s\n' "$1" "$2" "$3" "$4" "$st"
}
lines() { cat "$@" 2>/dev/null | grep -v -E '^[[:space:]]*$' | grep -v -E '^[[:space:]]*(//|/\*|\*|;)' | wc -l | tr -d ' '; }

echo "size (SPEC T10; code lines, docs/design.md d18):"
row "c core (src/core, src/platform)" "$(lines src/core/* src/platform/*)" 14000 lines
row "toks_par (src/par)" "$(lines src/par/*)" 1500 lines
for isa in arm64 x86_64; do
    row "asm $isa (src/asm/$isa)" "$(lines src/asm/$isa/*)" 5000 lines
done

if [ -n "$B" ] && [ -d "$B/obj" ]; then
    SZ=$(command -v llvm-size 2>/dev/null || xcrun --find llvm-size 2>/dev/null || true)
    [ -n "$SZ" ] || SZ="$(dirname "$(command -v "${CC:-clang}")")/llvm-size"
    code=$(find "$B/obj" -name '*.o' -exec "$SZ" -A {} + | awk '$1 ~ /^(__text|\.text)/ { s += $2 } END { print s + 0 }')
    tabs=$(find "$B/obj/src/gen" -name '*.o' -exec "$SZ" -A {} + \
        | awk '$1 ~ /^(__const|__data|__cstring|\.rodata|\.data)/ { s += $2 } END { print s + 0 }')
    row "code bytes ($B)" "$code" 1048576 bytes
    row "generated tables ($B, src/gen)" "$tabs" 2097152 bytes
fi
exit $fail
