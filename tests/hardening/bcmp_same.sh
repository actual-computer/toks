#!/bin/sh
# tests/hardening/bcmp_same.sh: the library's objects built twice with the Makefile's own flags, without and with
# -fno-builtin-bcmp, disassembled; the first build's bcmp calls renamed memcmp; the two must be the same text, i.e.
# the flag moves no instruction, only the name of the call target (glibc's bcmp is an alias of memcmp).
# From the repository root on a lab host: sh tests/hardening/bcmp_same.sh [make -j args]
set -eu
strict='-std=c17 -O3 -fno-strict-aliasing -fwrapv -Wall -Wextra -Wconversion -Wsign-conversion -Werror -fno-builtin-strlen'
make -s BUILD_DIR=build/bcmp-a CSTRICT="$strict" "$@" lib >/dev/null
make -s BUILD_DIR=build/bcmp-b CSTRICT="$strict -fno-builtin-bcmp" "$@" lib >/dev/null
n=0; nb=0; diff=0
for a in $(cd build/bcmp-a/obj && find . -name '*.o' | sort); do
    b=build/bcmp-b/obj/$a
    llvm-objdump -d -r --no-show-raw-insn --no-leading-addr "build/bcmp-a/obj/$a" | tail -n +3 | sed 's/\bbcmp\b/memcmp/g' > build/bcmp-a.dis
    llvm-objdump -d -r --no-show-raw-insn --no-leading-addr "$b" | tail -n +3 > build/bcmp-b.dis
    n=$((n + 1))
    c=$(llvm-nm -u "build/bcmp-a/obj/$a" | grep -c ' bcmp$' || true)
    [ "$c" = 0 ] || nb=$((nb + 1))
    if ! cmp -s build/bcmp-a.dis build/bcmp-b.dis; then echo "differs: $a"; diff=$((diff + 1)); fi
    if llvm-nm -u "$b" | grep -q ' bcmp$'; then echo "bcmp left in $b"; diff=$((diff + 1)); fi
done
echo "bcmp_same: $n objects, $nb call bcmp without the flag, $diff differ beyond the name (with the flag: none calls bcmp)"
[ "$diff" = 0 ]
