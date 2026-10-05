#!/bin/sh
# tools/remote.sh <host> <command...>
#
# Syncs this worktree (without .git, build/, .venv/ and nested worktrees)
# to <host>:~/toks-ci/<branch>/ and runs <command> there
# with the pinned llvm (~/.cache/toks-llvm/21.1.8/bin) and uv (~/.local/bin) first on PATH. <host> is your ssh
# alias for one of the machines in docs/machines.md (named there by chipset key). Each branch gets its own
# directory, so two worktrees sharing a machine never collide; build/ stays on the host between runs.
#
#   tools/remote.sh <host> make -j20 test
#   tools/remote.sh <host> 'make -j32 test && ./build/linux-x86_64/tests/test_k0'
#   tools/remote.sh <host> 'TOKS_HOST_KEY=gb10a PINCPU=7 sh tools/bench/e2e.sh "taskset -c 7"'   (receipts name the key)
set -eu
[ $# -ge 2 ] || { echo "usage: tools/remote.sh <host> <command...>" >&2; exit 2; }
host=$1; shift
root=$(git rev-parse --show-toplevel)
branch=$(git -C "$root" rev-parse --abbrev-ref HEAD)
[ "$branch" != HEAD ] || branch=$(basename "$root")   # a detached worktree (reviews): one directory per worktree
dir="toks-ci/$branch"
ssh -o BatchMode=yes "$host" "mkdir -p ~/$dir"
rsync -az --delete --exclude /.git --exclude /build/ --exclude /.venv/ --exclude /.worktrees/ --exclude /.claude/ "$root"/ "$host:$dir/"
ssh -o BatchMode=yes "$host" "cd ~/$dir && export PATH=\$HOME/.cache/toks-llvm/21.1.8/bin:\$HOME/.local/bin:\$PATH && { [ \"\$(uname)\" != Darwin ] || export SDKROOT=\"\$(xcrun --show-sdk-path)\"; } && $*"
