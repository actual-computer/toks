#!/bin/sh
# python/build.sh [3.10 3.11 ...]: build the toks wheel for each CPython (uv-managed, default 3.10 .. 3.14) on
# this host into build/wheels/ (linux: retagged manylinux_2_X by auditwheel repair, X = the glibc the module needs),
# then test every wheel (python/tests: api, threads; everything on PARITY_PY: parity, the drop-in surface against hf
# and tiktoken, Kimi K3's oracle).
# Run from the repository root, e.g. tools/remote.sh <host> 'sh python/build.sh'.
#
#   PARITY_PY=3.13        the interpreter that also runs test_parity.py ("" skips parity)
#   TOKS_PY_N, TOKS_PY_CASES, TOKS_CASES, TOKS_PY_REPORT, TOKS_TOKENIZER_CACHE   passed to the tests
#   TASKSET="taskset -c 0-7"                                                    prefix for build and test commands
#
# uv's interpreters and cache live under build/ (UV_PYTHON_INSTALL_DIR, UV_CACHE_DIR), so a lab host's shared
# locations are never written; set them to override.
set -eu
vers=${*:-3.10 3.11 3.12 3.13 3.14}
parity=${PARITY_PY-3.13}
run=${TASKSET:-}
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PWD/build/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/build/uv-cache}"
export UV_PYTHON_BIN_DIR="${UV_PYTHON_BIN_DIR:-$PWD/build/uv-bin}"   # uv python install's shims stay under build/ too
out=build/wheels
mkdir -p "$out"
if command -v sha256sum >/dev/null 2>&1; then sum=sha256sum; else sum="shasum -a 256"; fi
uv python install $vers >/dev/null
for v in $vers; do
    tag=cp$(echo "$v" | tr -d .)
    rm -f "$out"/toks-*-"$tag"-"$tag"-*.whl
    $run uv build -q --wheel python --python "$v" --out-dir "$out"
    if [ "$(uname -s)" = Linux ]; then
        # pip installs only manylinux / musllinux tags: auditwheel reads the module's symbol versions and libraries
        # and retags it with the lowest manylinux policy they allow (nothing is grafted: the module links libc only;
        # the glibc floor is the build host's pthread_* / stat symbols, docs/release.md)
        raw=$(ls "$out"/toks-*-"$tag"-"$tag"-linux_*.whl)
        aw="uvx -q --from auditwheel==6.1.0 --with patchelf==0.17.2.1 auditwheel"
        plat=$($run $aw show "$raw" | tr -d '\n' | sed -n 's/.*platform tag: *"\([a-z0-9_]*\)".*/\1/p')
        [ -n "$plat" ] || { echo "build.sh: auditwheel show found no manylinux tag for $raw" >&2; exit 1; }
        $run $aw repair --plat "$plat" -w "$out" "$raw"
        rm -f "$raw"
    fi
done
for v in $vers; do
    tag=cp$(echo "$v" | tr -d .)
    whl=$(ls "$out"/toks-*-"$tag"-"$tag"-*.whl)
    tests="python/tests/test_api.py python/tests/test_threads.py"
    with=""
    [ "$v" != "$parity" ] || { tests="python/tests"; with="--with tiktoken==0.14.0 --with transformers==5.18.0"; }
    echo "== $whl ($(du -k "$whl" | cut -f1) KiB)"
    # every build of a version has the same wheel file name, and uv keeps the --with environment of a requirement set:
    # gb10a's uv 0.9.24 ran rc run 3's tests in rc run 2's environment, i.e. on the old module (test_cache_mib caught
    # it; --refresh-package / --reinstall-package did not help). A copy under a path named by the wheel's sha256 is a
    # new requirement whenever the wheel changes.
    t=build/wheel-test/$($sum "$whl" | cut -c1-16)
    mkdir -p "$t" && cp "$whl" "$t/"
    $run uv run -q --no-project --isolated --python "$v" --with "$t/$(basename "$whl")" --with pytest \
        --with tokenizers==0.23.2 $with pytest -q -p no:cacheprovider $tests
done
