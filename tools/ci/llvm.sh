#!/bin/sh
# tools/ci/llvm.sh: the pinned llvm on a linux CI runner (SPEC §11; docs/ci.md): the official 21.1.8 release tarball
# the benchmark machines use (docs/machines.md), verified by its sha256, into ~/.cache/toks-llvm/21.1.8 (their path;
# tools/remote.sh puts its bin first on PATH). Only what `make test` runs is unpacked: clang, llvm-objdump and llvm-nm (the abi
# lint and the control-flow audit, tests/abi), llvm-ar / llvm-ranlib and clang's own headers, ~350 MB of the tarball's ~2 GB. Nothing to do when the CI
# cache restored it. The workflow caches the directory keyed on this file's hash: a change here is a new cache.
set -eu
V=21.1.8
case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  A=LLVM-$V-Linux-X64;   SHA=b3b7f2801d15d50736acea3c73982994d025b01c2f035b91ae3b49d1b575732b ;;
    Linux-aarch64) A=LLVM-$V-Linux-ARM64; SHA=65ce0b329514e5643407db2d02a5bd34bf33d159055dafa82825c8385bd01993 ;;
    *) echo "llvm.sh: no pinned llvm tarball for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
D=$HOME/.cache/toks-llvm/$V
if [ ! -x "$D/bin/clang" ]; then
    T=${RUNNER_TEMP:-/tmp}/$A.tar.xz
    t0=$(date +%s)
    curl -fsSL --retry 3 -o "$T" "https://github.com/llvm/llvm-project/releases/download/llvmorg-$V/$A.tar.xz"
    echo "$SHA  $T" | sha256sum -c -
    t1=$(date +%s)
    rm -rf "$D" && mkdir -p "$D"
    xz -T0 -dc "$T" | tar -x -C "$D" --strip-components=1 --wildcards \
        '*/bin/clang' '*/bin/clang-21' '*/bin/llvm-objdump' '*/bin/llvm-nm' '*/bin/llvm-ar' '*/bin/llvm-ranlib' '*/lib/clang/21/include/*'
    rm -f "$T"
    echo "llvm.sh: $A: download + sha256 $((t1 - t0)) s, unpack $(($(date +%s) - t1)) s, $(du -sh "$D" | cut -f1) kept"
fi
"$D/bin/clang" --version | head -1 | grep "clang version $V" || { echo "llvm.sh: $D/bin/clang is not $V" >&2; exit 1; }
