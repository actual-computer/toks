#!/bin/sh
# tests/common/guard_mutant.sh: the guard geometry's teeth (docs/testing.md 1.2). Three mutants, each on a copy of the
# tree under build/guard-mutant/, each built shipped and in runs 1 and 2 (make test-guard's two builds), each run through
# one test program:
#   extent   toks_compile_cls_flags reads cls_ascii[0..128]: one byte past its 128-byte table (compile.c). The shipped
#            build cannot see it; run 1 faults on that byte at load; run 2 (the start guarded) passes.
#   bound    decode's block one byte short of its two tables (stream.c dec_block_bytes): dec_len's last byte lies past
#            the block, in the page's slack. The shipped build cannot see it; the seal's placement check stops both runs
#            at load ("runs past its block").
#   overlap  decode's dec_len placed one byte early, over the last slot's 16th byte (stream.c). The shipped test_stream
#            sees it as 3 wrong decodes; the seal stops both runs at load, naming the two intervals.
# Exit 0 when every outcome is the one above. From the repository root, on any posix host:
#   sh tests/common/guard_mutant.sh [make -j args]
set -u
M=build/guard-mutant
rm -rf "$M" && mkdir -p "$M"
fail=0
mutant() {   # mutant <name> <file> <perl substitution> <test> <shipped want> <run 1 want> <run 2 want>; want: pass, fault, abort
    name=$1 file=$2 sub=$3 test=$4
    T=$M/$name
    mkdir -p "$T"
    tar cf - --exclude ./build --exclude ./.git --exclude ./.worktrees . | (cd "$T" && tar xf -)
    perl -pi -e "$sub" "$T/$file"
    if cmp -s "$file" "$T/$file"; then echo "guard_mutant $name: the edit did not apply to $file"; fail=1; return; fi
    i=0
    for g in "" 1 2; do
        bd=build/m$g
        (cd "$T" && make -s ${MAKEFLAGS_MUTANT:-} ${g:+GUARD=$g} BUILD_DIR=$bd "$bd/tests/$test" > "../$name-build$g.log" 2>&1) ||
            { echo "guard_mutant $name: build ${g:-shipped} failed ($M/$name-build$g.log)"; fail=1; return; }
        (cd "$T" && "$bd/tests/$test" > "../$name-run$g.log" 2>&1)
        rc=$?
        if [ $rc = 0 ]; then got=pass; elif [ $rc -gt 128 ] && grep -q '^guard: ' "$M/$name-run$g.log"; then got=abort
        elif [ $rc -gt 128 ]; then got=fault; else got="fail($rc)"; fi
        i=$((i + 1))
        eval want=\${$((4 + i))}
        echo "guard_mutant $name $([ -n "$g" ] && echo "run $g" || echo shipped): $got (want $want) $(grep -m1 '^guard: ' "$M/$name-run$g.log")"
        [ "$got" = "$want" ] || fail=1
    done
}
MAKEFLAGS_MUTANT="$*"
mutant extent src/core/compile.c 's/for \(uint32_t b = 0; b < 128u; b\+\+\) \{(\s+)\/\* bound: 128 ascii bytes/for (uint32_t b = 0; b <= 128u; b++) {$1\/* MUTANT 129 ascii bytes/' \
    test_e2e pass fault pass
mutant bound src/core/stream.c 's/return 17u \* \(uint64_t\)n_ids; \}/return 17u * (uint64_t)n_ids - 1u; }/' \
    test_stream pass abort abort
mutant overlap src/core/stream.c 's/toks_tab\(blk, 16u \* \(uint64_t\)t->n_ids, t->n_ids, TOKS_X_DEC_LEN\)/toks_tab(blk, 16u * (uint64_t)t->n_ids - 1u, t->n_ids, TOKS_X_DEC_LEN)/' \
    test_stream "fail(1)" abort abort
echo "guard_mutant: $([ $fail = 0 ] && echo pass || echo FAIL)"
exit $fail
