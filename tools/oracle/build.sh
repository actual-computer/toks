#!/bin/sh
# tools/oracle/build.sh: builds the piece oracle (tiktoken-pieces) into build/oracle/release/.
#
# The rust toolchain lives under build/rust/ of this checkout (RUSTUP_HOME / CARGO_HOME there, never the host's
# own; build/ survives tools/remote.sh's rsync), installed on first use with rustup's minimal profile. The crate
# versions are fixed by tools/oracle/Cargo.lock (--locked).
#
#   tools/remote.sh <host> 'sh tools/oracle/build.sh'
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
RUSTUP_HOME="$root/build/rust/rustup"
CARGO_HOME="$root/build/rust/cargo"
export RUSTUP_HOME CARGO_HOME
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    mkdir -p "$root/build/rust"
    curl -sSf https://sh.rustup.rs -o "$root/build/rust/rustup-init.sh"
    sh "$root/build/rust/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain stable >/dev/null
fi
"$CARGO_HOME/bin/rustc" --version
"$CARGO_HOME/bin/cargo" build -q --release --locked --manifest-path "$root/tools/oracle/Cargo.toml" \
    --target-dir "$root/build/oracle"
ls -l "$root/build/oracle/release/tiktoken-pieces"
