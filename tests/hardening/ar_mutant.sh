#!/bin/sh
# tests/hardening/ar_mutant.sh: the arena guard's mutant (docs/hardening.md §6). A copy of src/, include/ and
# tests/fuzz/ under build/mutant/ with toks_ar_alloc's `p > a->len` clause taken out, its fuzz harnesses built
# (ASan + UBSan, the arena probe in), and the load harnesses' corpora, seeds and regress repros replayed through
# them (tests/fuzz/run.sh probe, its logs in build/mutant/state/probe/), then every harness's regress repros
# (run.sh regress). The guard is defense: the mutant must pass everything the guarded build passes. From the
# repository root on a lab host, after a campaign or `run.sh seeds` filled $FUZZ_STATE (default build/fuzz):
#   sh tests/hardening/ar_mutant.sh [make -j args]
set -eu
case "$(uname -m)" in aarch64|arm64) isa=arm64 ;; *) isa=x86_64 ;; esac
ST=${FUZZ_STATE:-build/fuzz}
M=build/mutant
rm -rf "$M" && mkdir -p "$M/tests" "$M/state"
cp -R src include "$M/" && cp -R tests/fuzz "$M/tests/"
for d in corpus seeds; do [ -d "$ST/$d" ] && ln -s "$(pwd)/$ST/$d" "$M/state/$d"; done
guard='if (p < a->pos || p > a->len || n > a->len - p)'
grep -qF "$guard" "$M/src/core/core.h" || { echo "ar_mutant: the guard line is not in src/core/core.h" >&2; exit 1; }
sed -i.orig 's/if (p < a->pos || p > a->len || n > a->len - p)/if (p < a->pos || n > a->len - p)/' "$M/src/core/core.h"
if grep -qF "$guard" "$M/src/core/core.h"; then echo "ar_mutant: the guard survived the edit" >&2; exit 1; fi
(cd "$M" && make -s -f tests/fuzz/Makefile "$@" "build/fuzz-linux-$isa/fuzz_load" "build/fuzz-linux-$isa/fuzz_load_json" \
    "build/fuzz-linux-$isa/fuzz_encode" "build/fuzz-linux-$isa/fuzz_par" >/dev/null)
echo "ar_mutant: built without the guard in $M/build/fuzz-linux-$isa"
rc=0
FUZZ_BIN="$M/build/fuzz-linux-$isa" FUZZ_STATE="$M/state" sh tests/fuzz/run.sh probe > "$M/probe.txt" || rc=1
grep -v '^probe [a-z_]* src/' "$M/probe.txt"
FUZZ_BIN="$M/build/fuzz-linux-$isa" FUZZ_STATE="$M/state" sh tests/fuzz/run.sh regress > "$M/regress.txt" || rc=1
echo "ar_mutant regress: $(grep -c '^pass' "$M/regress.txt") pass, $(grep -c '^FAIL' "$M/regress.txt" || true) FAIL"
echo "ar_mutant: $([ $rc = 0 ] && echo pass || echo FAIL)"
exit $rc
