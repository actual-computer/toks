#!/bin/sh
# tools/release/rc.sh [ref]: a toks release candidate, run from the control-plane mac. It measures <ref>
# (default HEAD) on every release host at once and writes what it checked, where, with which counts. Machines are
# named by chipset key (docs/machines.md); RC_HOSTS maps each key to the ssh alias that reaches it, and the key
# alone names the stage and receipt directories:
#
#   gb10a     arm64 linux (NVIDIA GB10, Cortex-X925), tiers neon + scalar: make test (+ every test program under
#             TOKS_TIER=scalar), ASan + UBSan make test, TSan test_par, parity samples, wheels, the speed table
#             (pinned to an X925 core)
#   tr9970x   x86-64 linux (AMD Threadripper 9970X, Zen 5), tiers avx2 + scalar: the same without sanitizers
#             (speed table pinned to cpu 12, its smt sibling 44 idle)
#   m2ultra1  arm64 macOS (Apple M2 Ultra), tier neon: make test, parity samples (2 shards: a timing host), wheels,
#             the speed table
#   aimax395  x86-64 windows (AMD Ryzen AI MAX+ 395, Windows 11; cmd.exe, no make): tools\win\build.cmd +
#             tools\win\test.cmd (docs/release.md step 1)
#
# Each host runs tools/release/rc_host.sh (the steps and their exact commands are in its header) in
# ~/toks-ci/release-rc/ (build/ persists there between runs: uv pythons, the rust toolchain, the corpus). The tree
# is `git archive <ref>` with this checkout's tools/bench and tools/release over it, so a dry run of an older ref
# measures that ref with today's tools (a no-op at the release commit).
#
# Outputs (on the mac):
#   build/release/rc/<sha12>/<key>/       everything the host wrote (build/rc/ without the case files)
#   docs/bench/raw/<key>-<tier>-<sha12>.log  the speed table's raw logs; docs/bench/e2e.md regenerated from them
#   docs/release/rc/<sha12>/<key>/        the receipts that get committed (steps, host, summaries, parity reports)
#   docs/release/<major.minor>.md         the report (tools/release/rc_report.py; the version is the staged
#                                         include/toks.h's TOKS_VERSION): checks x hosts with counts and loads,
#                                         every unmet item listed as unmet
#   docs/bench/raw/tokv1-<key>-<sha12>.log  the incumbent cells (tools/bench/tokv1.sh: linux hosts, only when E_SRC
#                                         is given and holds E_REV; an optional bench input like gigatoken)
#
#   RC_HOSTS="<host>=gb10a <host>=tr9970x <host>=m2ultra1" RC_WIN_HOSTS="<host>=aimax395" tools/release/rc.sh
#                                          the release commit (HEAD); each <host> is your ssh alias for that machine
#   RC_HOSTS=... tools/release/rc.sh origin/master       a dry run of master
#   RC_HOSTS="<host>=tr9970x" RC_STEPS="test parity" tools/release/rc.sh     a subset (the report says what was not run)
#   RC_HOSTS=... RC_REPORT_ONLY=1 tools/release/rc.sh <ref>   the table, receipts and report again from
#                                          build/release/rc/<sha12>/ (after a subset rerun of the same ref: the stage
#                                          keeps every host; only the keys are read then)
#
# env: RC_HOSTS ("<ssh-alias>=<key> ...": required, there is no default host list)  RC_WIN_HOSTS (same form, the
#      windows machines; default none)  RC_STEPS (rc_host.sh's)  RC_DIR (toks-ci/release-rc)  RC_TEXT_SEED
#      (<ssh-alias>:<dir> holding the bench corpus, tools/bench/corpus.sha256; default none: a host without a
#      verified corpus reports it)  GIGA_SRC (a gigatoken checkout, read only)  GIGA_REV  E_SRC (a checkout holding
#      tok v1, read only: git archive of E_REV, b7af21b; no default: unset, the incumbent cells are not run)
#      RC_GB10_CPU (7) / RC_TR9970X_CPU (12): the timing cores
#      RC_TR9970X_TASKSET (0-7,32-39: CCD0, the builds' ccd) / RC_GB10_TASKSET (0-4,10-12): the heavy steps' cpus
#      RC_TR9970X_NASM (nasm for tok v1 on the 9970X)  RC_REPORT_ONLY
set -eu
cd "$(git rev-parse --show-toplevel)"
REF=${1:-HEAD}
SHA=$(git rev-parse "$REF^{commit}")
S=$(echo "$SHA" | cut -c1-12)
ST=build/release/rc/$S
HOSTS=${RC_HOSTS:?"rc.sh: RC_HOSTS=\"<ssh-alias>=<key> ...\" names the machines (keys: docs/machines.md); there is no default"}
WIN_HOSTS=${RC_WIN_HOSTS:-}
DIR=${RC_DIR:-toks-ci/release-rc}
WDIR=$(echo "$DIR" | tr / '\\')
TEXT_SEED=${RC_TEXT_SEED:-}
GIGA_SRC=${GIGA_SRC:-$HOME/repos/gigatoken}
GIGA_REV=${GIGA_REV:-fac0114b37120ec8a76362e9ee8e1c742aaafaef}
E_SRC=${E_SRC:-}
E_REV=${E_REV:-b7af21b}
GB10_CPU=${RC_GB10_CPU:-7}
TR_CPU=${RC_TR9970X_CPU:-12}
TR_NASM=${RC_TR9970X_NASM:-\$HOME/toks-ci/tools/nasm/bin/nasm}
TR_SET=${RC_TR9970X_TASKSET:-0-7,32-39}
GB10_SET=${RC_GB10_TASKSET:-0-4,10-12}
PATHS='export PATH=$HOME/.cache/toks-llvm/21.1.8/bin:$HOME/.local/bin:$PATH; [ "$(uname)" != Darwin ] || export SDKROOT="$(xcrun --show-sdk-path)"'

key_of() { echo "${1#*=}"; }      # "<ssh-alias>=<key>" -> key (a bare word is both)
alias_of() { echo "${1%%=*}"; }
KEYS=$(for p in $HOSTS; do key_of "$p"; done | paste -s -d ' ' -)
WIN_KEYS=$(for p in $WIN_HOSTS; do key_of "$p"; done | paste -s -d ' ' -)

tv1_env() { [ -s "$ST/e-$E_REV.tar" ] && echo " TOKV1_E_DIR=build/e"; }   # tok v1 staged: the linux hosts run tokv1
role() {   # role <key>: the per-machine environment of rc_host.sh
    case $1 in
        tr9970x) echo "TIERS=\"auto scalar\" JOBS=8 CARGO_BUILD_JOBS=8 TASKSET=\"taskset -c $TR_SET\" SAN=0 PARITY_SHARDS=8 BENCH_CPU=$TR_CPU BENCH_PIN=\"taskset -c $TR_CPU\"$(tv1_env) TOKV1_NASM=$TR_NASM" ;;
        gb10*) echo "TIERS=\"auto scalar\" JOBS=8 CARGO_BUILD_JOBS=8 TASKSET=\"taskset -c $GB10_SET\" SAN=1 PARITY_SHARDS=8 BENCH_CPU=$GB10_CPU BENCH_PIN=\"taskset -c $GB10_CPU\"$(tv1_env)" ;;
        m2ultra*) echo 'TIERS="auto" JOBS=2 CARGO_BUILD_JOBS=2 TASKSET="" SAN=0 PARITY_SHARDS=2 BENCH_CPU="" BENCH_PIN=""' ;;
        *) echo 'TIERS="auto"' ;;
    esac
}

run_host() {   # run_host <ssh-alias>=<key>: sync, stage the inputs, run rc_host.sh, collect under the key
    h=$(alias_of "$1"); k=$(key_of "$1")
    ssh -o BatchMode=yes "$h" "mkdir -p ~/$DIR/build/third-party ~/$DIR/build/text"
    rsync -az --delete --exclude /build/ "$ST/src/" "$h:$DIR/"
    scp -q "$ST/gigatoken-$GIGA_REV.tar.gz" "$h:$DIR/build/third-party/"
    if [ -s "$ST/e-$E_REV.tar" ] && [ "$(ssh -o BatchMode=yes "$h" uname)" = Linux ]; then   # tok v1 (read-only e archive)
        ssh -o BatchMode=yes "$h" "rm -rf ~/$DIR/build/e && mkdir -p ~/$DIR/build/e"
        scp -q "$ST/e-$E_REV.tar" "$h:$DIR/build/e/e.tar"
        ssh -o BatchMode=yes "$h" "cd ~/$DIR/build/e && tar -xf e.tar && rm e.tar && echo $E_REV > .e-rev"
    fi
    if ! ssh -o BatchMode=yes "$h" "cd ~/$DIR && sh tools/release/rc_host.sh corpus"; then
        [ -z "$TEXT_SEED" ] || scp -3 -q "$TEXT_SEED/*" "$h:$DIR/build/text/"
        ssh -o BatchMode=yes "$h" "cd ~/$DIR && sh tools/release/rc_host.sh corpus" || echo "rc: $k: corpus does not verify (RC_TEXT_SEED seeds one)"
    fi
    # shellcheck disable=SC2029
    ssh -o BatchMode=yes "$h" "cd ~/$DIR && $PATHS; env TOKS_HOST_KEY=$k $(role "$k") GIT_SHA=$SHA ${RC_STEPS:+RC_STEPS=\"$RC_STEPS\"} sh tools/release/rc_host.sh" \
        > "$ST/$k.out" 2>&1 || true
    mkdir -p "$ST/$k"
    rsync -az --delete --exclude /cases/ "$h:$DIR/build/rc/" "$ST/$k/"
    echo "rc: $k done: $(/usr/bin/grep -c 'status=0' "$ST/$k/steps.txt" 2>/dev/null || echo 0) of $(wc -l < "$ST/$k/steps.txt" 2>/dev/null | tr -d ' ' || echo 0) steps passed"
}

run_win() {   # run_win <ssh-alias>=<key>: windows x86-64 over cmd.exe: the tree as a tarball, build.cmd, test.cmd, collect
    h=$(alias_of "$1"); k=$(key_of "$1")
    o=$ST/$k
    rm -rf "$o"
    mkdir -p "$o"
    (cd "$ST/src" && COPYFILE_DISABLE=1 tar -czf "../$k-src.tgz" .)
    ssh -o BatchMode=yes "$h" "if exist %USERPROFILE%\\$WDIR rmdir /s /q %USERPROFILE%\\$WDIR"
    ssh -o BatchMode=yes "$h" "mkdir %USERPROFILE%\\$WDIR" && scp -q "$ST/$k-src.tgz" "$h:$DIR/src.tgz" &&
        ssh -o BatchMode=yes "$h" "cd /d %USERPROFILE%\\$WDIR && tar -xzf src.tgz && del src.tgz" || echo "rc: $k: sync failed"
    { echo "host $k"; ssh -o BatchMode=yes "$h" "ver"
      ssh -o BatchMode=yes "$h" 'powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"'
      echo "git $SHA"; ssh -o BatchMode=yes "$h" '%USERPROFILE%\.cache\toks-llvm\21.1.8\bin\clang.exe --version' | head -1
      echo "tiers auto jobs 1 taskset '' bench '' cpu"; } 2>&1 | tr -d '\r' | sed '/^$/d' > "$o/host.txt"
    : > "$o/steps.txt"
    for s in build test; do
        t0=$(date +%s)
        rc=0
        ssh -o BatchMode=yes "$h" "cd /d %USERPROFILE%\\$WDIR && tools\\win\\$s.cmd" > "$o/win-$s.raw" 2>&1 || rc=$?
        tr -d '\r' < "$o/win-$s.raw" | sed 's/C:\\Users\\[^\\]*/%USERPROFILE%/g' > "$o/win-$s.log" && rm -f "$o/win-$s.raw"
        echo "STEP win-$s status=$rc secs=$(( $(date +%s) - t0 )) load=n/a" >> "$o/steps.txt"
    done
    echo "rc: $k done: $(/usr/bin/grep -c 'status=0' "$o/steps.txt") of 2 steps passed"
}

if [ "${RC_REPORT_ONLY:-0}" != 1 ]; then
    rm -rf "$ST/src"                  # the stage keeps the other hosts' results: a subset rerun, then RC_REPORT_ONLY=1
    mkdir -p "$ST/src"
    git archive "$SHA" | tar -x -C "$ST/src"
    for d in tools/bench tools/release; do
        rm -rf "${ST:?}/src/$d"
        mkdir -p "$ST/src/$d"
        cp -R "$d/." "$ST/src/$d/"
    done
    git -C "$GIGA_SRC" archive -o "$PWD/$ST/gigatoken-$GIGA_REV.tar.gz" "$GIGA_REV"
    rm -f "$ST/e-$E_REV.tar"
    [ -n "$E_SRC" ] && git -C "$E_SRC" archive -o "$PWD/$ST/e-$E_REV.tar" "$E_REV" host/tok.asm host/inc host/a64/tok.S \
        host/a64/inc tests/host/tok.h tools/tokpack 2>/dev/null ||
        echo "rc: no tok v1 checkout (E_SRC '$E_SRC', E_REV $E_REV): the incumbent cells are not run"
    echo "rc: $REF = $SHA on $KEYS $WIN_KEYS (stage $ST)"
    for p in $HOSTS; do run_host "$p" & done
    for p in $WIN_HOSTS; do run_win "$p" & done
    wait
fi

# ---- the speed table ---------------------------------------------------------------------------------
mkdir -p docs/bench/raw
args=""
for h in $KEYS; do
    for f in "$ST/$h"/bench-*.log; do
        [ -s "$f" ] || continue
        tier=$(sed -n 's/^E2E tool=toks tier=\([a-z0-9]*\).*/\1/p' "$f" | head -1)
        [ -n "$tier" ] || continue
        cp "$f" "docs/bench/raw/$h-$tier-$S.log"
        args="$args $h-$tier=docs/bench/raw/$h-$tier-$S.log"
    done
done
tv1=""
for h in $KEYS; do   # the incumbent section: this rc's tools/bench/tokv1.sh logs, else the latest ones committed
    if [ -s "$ST/$h/tokv1.log" ] && /usr/bin/grep -q '^TOKV1 ' "$ST/$h/tokv1.log"; then
        cp "$ST/$h/tokv1.log" "docs/bench/raw/tokv1-$h-$S.log"
        tv1="$tv1 tokv1-$h=docs/bench/raw/tokv1-$h-$S.log"
    fi
done
if [ -z "$tv1" ]; then
    for f in docs/bench/raw/tokv1-*.log; do [ -s "$f" ] && tv1="$tv1 $(basename "$f" .log)=$f"; done
fi
args="$args$tv1"
if [ -n "$args" ]; then
    # shellcheck disable=SC2086
    uv run -q tools/bench/e2e_table.py $args > docs/bench/e2e.md
fi
gargs=""
for h in $KEYS; do   # the gigatoken gate (rc_host.sh's gate step): its logs and the generated file (labels: <key>-<tier>)
    f="$ST/$h/gate.log"
    [ -s "$f" ] && /usr/bin/grep -q '^RUN ' "$f" || continue
    tier=$(sed -n 's/^RUN .* E2E tool=toks tier=\([a-z0-9]*\).*/\1/p' "$f" | head -1)
    cp "$f" "docs/bench/raw/gate-$h-$tier-$S.log"
    gargs="$gargs $h-$tier=docs/bench/raw/gate-$h-$tier-$S.log"
done
if [ -n "$gargs" ]; then   # exit 1 = MISSING cells or a wholly void cell config: the file says which; the report reads it
    # shellcheck disable=SC2086
    uv run -q tools/bench/gate_table.py $gargs > docs/bench/gigatoken-gate.md || echo "rc: the gate is not complete (docs/bench/gigatoken-gate.md says why)"
fi

# ---- receipts + report -------------------------------------------------------------------------------
R=docs/release/rc/$S
rm -rf "$R"
for h in $KEYS $WIN_KEYS; do
    mkdir -p "$R/$h"
    for f in steps.txt host.txt; do [ -f "$ST/$h/$f" ] && cp "$ST/$h/$f" "$R/$h/"; done
    for f in "$ST/$h"/*.log; do   # summaries: suite results, failures, pytest totals, the last lines
        [ -f "$f" ] || continue
        case $(basename "$f") in bench-*|gate.log|gigatoken-build.log) continue ;; esac
        { /usr/bin/grep -E '^(== |test_|asmcheck|FAILED|FAIL |-- FAIL|SKIP|ERROR|AB |HOST |kernels:|build\.cmd|test\.cmd|[0-9]+ (passed|failed)|.* (passed|failed|error)( |,|$)|tier |cases |SUMMARY|WARNING: ThreadSanitizer|==[0-9]+==ERROR)' "$f" | cut -c1-400 | head -400
          echo "-- last lines"; tail -n 5 "$f" | cut -c1-400; } > "$R/$h/$(basename "$f" .log).txt"
    done
    for d in "$ST/$h"/parity/*/; do
        [ -f "$d/report.json" ] || continue
        uv run -q --no-project python -c '
import json, sys
r = json.load(open(sys.argv[1]))
if isinstance(r.get("diffs"), list):
    r["n_diffs_listed"] = len(r["diffs"]); r["diffs"] = r["diffs"][:5]
json.dump(r, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)' "$d/report.json" "$R/$h/parity-$(basename "$d").json"
    done
done
V=$(sed -n 's/^#define TOKS_VERSION *"\(.*\)"/\1/p' "$ST/src/include/toks.h")
MM=${V%.*}
uv run -q tools/release/rc_report.py --sha "$SHA" --ref "$REF" --stage "$ST" --receipts "$R" --hosts "$KEYS" \
    --win-hosts "$WIN_KEYS" --version "$V" > "docs/release/$MM.md"
echo "rc: wrote docs/release/$MM.md, docs/bench/e2e.md$( [ -n "$gargs" ] && echo ', docs/bench/gigatoken-gate.md'), $R"
