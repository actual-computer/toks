#!/bin/sh
# tools/bench/gigatoken.sh: build the gigatoken comparator of tools/bench/e2e.sh on this host. Everything lives
# under build/ (lab-host hygiene); nothing here is a dependency of the library or its tests.
#
#   build/third-party/gigatoken      the pinned source, unpacked from build/third-party/gigatoken-<rev>.tar.gz
#                                    (on the mac: git -C <gigatoken checkout> archive -o <that file> <rev>; the
#                                    checkout itself stays read-only)
#   build/rustup, build/cargo        a user-local rust nightly (gigatoken needs portable_simd); RUSTUP_SEED=<dir>
#                                    copies <dir>/rustup + <dir>/cargo (an existing user-local install) instead of
#                                    downloading rustup-init
#   build/giga-target/release/gigatoken-bench, build/giga-target/rustc.txt
#
#   sh tools/bench/gigatoken.sh                  (GIGA_REV: the pinned commit; TOOLCHAIN: the rustup toolchain)
set -eu
GIGA_REV=${GIGA_REV:-fac0114b37120ec8a76362e9ee8e1c742aaafaef}
TOOLCHAIN=${TOOLCHAIN:-nightly}
ROOT=$PWD
export RUSTUP_HOME=$ROOT/build/rustup CARGO_HOME=$ROOT/build/cargo
export PATH=$CARGO_HOME/bin:$PATH
export CARGO_TARGET_DIR=$ROOT/build/giga-target
export RUSTUP_TOOLCHAIN=$TOOLCHAIN
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-only-managed}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ROOT/build/uv-python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/build/uv-cache}
mkdir -p build/third-party
src=build/third-party/gigatoken
if [ "$(cat "$src/.toks-rev" 2>/dev/null)" != "$GIGA_REV" ]; then
    tgz=build/third-party/gigatoken-$GIGA_REV.tar.gz
    [ -s "$tgz" ] || { echo "gigatoken.sh: $tgz missing (copy the pinned source here first)" >&2; exit 1; }
    rm -rf "$src"
    mkdir -p "$src"
    tar -xzf "$tgz" -C "$src"
    echo "$GIGA_REV" > "$src/.toks-rev"
fi
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    if [ -n "${RUSTUP_SEED:-}" ] && [ -d "$RUSTUP_SEED/rustup" ] && [ -d "$RUSTUP_SEED/cargo" ]; then
        cp -a "$RUSTUP_SEED/rustup" "$RUSTUP_HOME"
        cp -a "$RUSTUP_SEED/cargo" "$CARGO_HOME"
    else
        curl -sSf https://sh.rustup.rs -o build/rustup-init.sh
        sh build/rustup-init.sh -y -q --no-modify-path --profile minimal --default-toolchain "$TOOLCHAIN"
    fi
fi
rustup toolchain list | grep -q "^$TOOLCHAIN" || rustup toolchain install -q --profile minimal "$TOOLCHAIN"
# gigatoken's pyo3 (abi3) wants an interpreter at build time: a uv-managed one under build/
[ -x build/giga-py/bin/python ] || uv venv -q --python 3.13 build/giga-py
export PYO3_PYTHON=$ROOT/build/giga-py/bin/python
# pyo3 may link the bench binary against libpython (a uv python under build/): run-path it so the binary starts
PYLIB=$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}${PYLIB:+ -C link-arg=-Wl,-rpath,$PYLIB}"
export GIGA_REV
M=tools/bench/gigatoken/Cargo.toml
# gigatoken's manifest sets per-profile rustflags (nightly cargo -Z profile-rustflags)
cargo -Z profile-rustflags build -q --release --locked --offline --manifest-path $M 2>/dev/null ||
    cargo -Z profile-rustflags build -q --release --locked --manifest-path $M
rustc --version > build/giga-target/rustc.txt
echo "gigatoken.sh: $CARGO_TARGET_DIR/release/gigatoken-bench (gigatoken $GIGA_REV, $(cat build/giga-target/rustc.txt), RUSTFLAGS $RUSTFLAGS)"
