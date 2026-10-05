#!/bin/sh
# tools/ci/changes.sh: what a pull request changes, as outputs for the jobs that run on it (docs/ci.md, The required
# checks). In a pull_request job after a checkout of depth 2, HEAD is the merge commit GitHub made and HEAD^1 the base
# it merges into, so the diff is what merging the PR changes (a rename counts both of its paths):
#   code=false     only docs changed (**.md, docs/**, LICENSE*: what the push triggers ignore); test.yml's jobs skip
#   windows=true   something the Windows job tests changed (src/platform, src/asm/x86_64, tools/win, tools/ci, the
#                  Makefile, tests/common, windows.yml); windows.yml's job runs
# Prints the outputs as key=value lines (for $GITHUB_OUTPUT) and the changed paths on stderr. A failure leaves the
# outputs unset, and the jobs run.
set -eu
files=$(git diff --name-only --no-renames HEAD^1 HEAD)
printf '%s\n' "$files" | sed 's/^/  changed: /' >&2
any() { if printf '%s\n' "$files" | grep -Eq "$@"; then echo true; else echo false; fi; }
echo "code=$(any -v '\.md$|^docs/|^LICENSE[^/]*$')"
echo "windows=$(any '^(src/platform/|src/asm/x86_64/|tools/win/|tools/ci/|tests/common/)|^(Makefile|\.github/workflows/windows\.yml)$')"
