#!/bin/sh
# tests/common/guard_mutant.sh: the guard geometry's teeth (docs/testing.md 1.2). Seven mutants, each on a copy of the
# tree under build/guard-mutant/, each built shipped and in runs 1 and 2 (make test-guard's two builds), each run through
# one test program. A run marked - may go either way: which run must catch a mutant is the point, the other run's
# outcome depends on where the byte falls in its page.
#   extent   toks_compile_cls_flags reads cls_ascii[0..128]: one byte past its 128-byte table (compile.c). The shipped
#            build cannot see it; run 1 faults on that byte at load; run 2 (the start guarded) passes.
#   start    toks_compile_cls_flags reads cls_ascii[-1]: one byte before the table. Shipped and run 1 pass; run 2
#            faults at load.
#   arena    bpe_build reads byte2id[256], one u32 past an arena table (toks_tab_ar). Shipped passes; run 1 faults at
#            load; run 2 passes.
#   scratch  k5_run writes the byte after the work region, the bounce's first, before K5 runs (api.c; byte-level
#            contexts without the generic engine's lists). Shipped passes, K5 writes the bounce over it; run 1 faults
#            on the first encode.
#   bound    decode's block one byte short of its two tables (stream.c dec_block_bytes): dec_len's last byte lies past
#            the block, in the page's slack. The shipped build cannot see it; the seal's placement check stops both runs
#            at load ("runs past its block").
#   take     bpe_build takes one more arena table that no context pointer holds (bpe_build.c). Shipped test_guard
#            passes; in both runs test_guard fails ("a table pointer tables_of misses") on the byte-level contexts, and
#            among its fixtures (tests/data, always there) on exactly the three byte-level ones: the tokenizer cache's
#            contexts may be absent on a host, so they are not counted.
#   overlap  decode's dec_len placed one byte early, over the last slot's 16th byte (stream.c). The shipped test_stream
#            sees it as 3 wrong decodes; the seal stops both runs at load, naming the two intervals.
# A fault counts only with guard.c's "guard: fault at" line (the address on a guard page, in a sealed block or in the
# poison area) and an abort only with the seal's message; any other death is crash(<rc>), which no mutant wants.
# Exit 0 when every outcome is the one above. From the repository root, on any posix host:
#   sh tests/common/guard_mutant.sh [make -j args]
set -u
M=build/guard-mutant
rm -rf "$M" && mkdir -p "$M"
fail=0
mutant() {   # mutant <name> <file> <perl substitution> <test> <shipped want> <run 1 want> <run 2 want> [<pattern>:<count>]
             # want: pass, fault (a guard fault: gfault's line), abort (the seal's message), fail(<rc>), -
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
        log=$M/$name-run$g.log
        if [ $rc = 0 ]; then got=pass
        elif [ $rc -gt 128 ] && grep -q '^guard: fault at ' "$log"; then got=fault       # on a guard page (guard.c gfault)
        elif [ $rc -gt 128 ] && grep -qE '^guard: (tables overlap in their block|a table of [0-9]+ bytes at offset)' "$log"
        then got=abort                                                                     # the seal refused a placement
        elif [ $rc -gt 128 ]; then got="crash($rc)"                                        # a death the guard did not name
        else got="fail($rc)"; fi
        i=$((i + 1))
        eval want=\${$((4 + i))}
        echo "guard_mutant $name $([ -n "$g" ] && echo "run $g" || echo shipped): $got (want $want) $(grep -m1 '^guard: ' "$log")"
        [ "$want" = - ] || [ "$got" = "$want" ] || fail=1
        if [ -n "$g" ] && [ -n "${8:-}" ]; then              # <pattern>:<count>: lines each guard run must print
            k=$(grep -c "${8%:*}" "$log")
            echo "guard_mutant $name run $g: $k lines '${8%:*}' (want ${8##*:})"
            [ "$k" = "${8##*:}" ] || fail=1
        fi
    done
}
MAKEFLAGS_MUTANT="$*"
mutant extent src/core/compile.c 's/for \(uint32_t b = 0; b < 128u; b\+\+\) \{(\s+)\/\* bound: 128 ascii bytes/for (uint32_t b = 0; b <= 128u; b++) {$1\/* MUTANT 129 ascii bytes/' \
    test_e2e pass fault pass
mutant start src/core/compile.c 's/^(    int d = t->tmpl == TOKS_TMPL_DSV3;.*)$/$1\n    (void)*(volatile const uint8_t *)(t->cls_ascii - 1);   \/* MUTANT *\//' \
    test_e2e pass pass fault
mutant arena src/core/bpe_build.c 's/^(    memset\(byte2id, 0xFF, 256u \* 4u\);)$/$1\n    (void)*(volatile const uint32_t *)(byte2id + 256);   \/* MUTANT *\//' \
    test_e2e pass fault pass
mutant scratch src/core/api.c 's/^(    uint64_t m = toks_k5\(&ctx->t, &k, ctx->tier\);)$/    if (ctx->scr_extra == 0u) { ((volatile uint8_t *)k.work)[k.work_bytes] = 0u; }   \/* MUTANT *\/\n$1/' \
    test_e2e pass fault -
mutant bound src/core/stream.c 's/return 17u \* \(uint64_t\)n_ids; \}/return 17u * (uint64_t)n_ids - 1u; }/' \
    test_stream pass abort abort
mutant take src/core/bpe_build.c 's/^(    uint32_t \*byte2id = \(uint32_t \*\)toks_tab_ar\(ar, 256u \* 4u, 64u, TOKS_X_BYTE2ID\);)$/$1\n    (void)toks_tab_ar(ar, 1u, 1u, TOKS_X_BYTE2ID);   \/* MUTANT: a table nobody lists *\//' \
    test_guard pass "fail(1)" "fail(1)" 'tests/data/.*\.json: .*a table pointer tables_of misses:3'
mutant overlap src/core/stream.c 's/toks_tab\(blk, 16u \* \(uint64_t\)t->n_ids, t->n_ids, TOKS_X_DEC_LEN\)/toks_tab(blk, 16u * (uint64_t)t->n_ids - 1u, t->n_ids, TOKS_X_DEC_LEN)/' \
    test_stream "fail(1)" abort abort
echo "guard_mutant: $([ $fail = 0 ] && echo pass || echo FAIL)"
exit $fail
