#!/bin/sh
# tools/bench/e2e.sh: the speed table's cells on this host, one thread: toks (the tier toks_load binds;
# TOKS_TIER=scalar|neon|avx2|avx512 forces one) + hf tokenizers 0.23.2 + tiktoken 0.14.0 + gigatoken (pinned,
# tools/bench/gigatoken.sh) on exactly the same chunks. Best of reps (no abba pairs, no intervals: a quick look at
# the gate's cells, not a certified cell), with the void rule's receipts: host fingerprint, load and the
# pinned cpus' other threads before and after every cell, binary / corpus / tokenizer sha-256, every raw line, and
# exactness on the measured chunks, outside every timer (toks ids sha-256 == the reference's, else the cell is
# void; tiktoken and gigatoken ids == hf's without the post-processor, else their column is n/a). linux: BUSY
# lines give the busy share of the pinned cpu and its smt siblings over each cell (the sibling should be ~0%).
# States (e2e.c's): cold (the reps first, back to back), pass (after an untimed pass over OTHER text, the scratch
# re-initialized), warm (the replay right after it), and lang-x = warmed on the OTHER text, scratch kept. The OTHER
# text is every bench corpus file outside the measured corpus (E2E_WARM_ON / GIGA_WARM_ON; tools/bench/tokv1.sh's
# rule: cross-language, not a same-language warm state).
#
#   tools/bench/e2e.sh [pin]        e.g. PINCPU=12 tools/bench/e2e.sh "taskset -c 12"   (macOS: no pin)
#
# env: TOKS_LIST (gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 kimik3)  CORPORA (en code ml cjk)
#      CHUNKS (4096 0; 0 = the whole corpus in one call)  REPS (5)  REF_REPS (3, the python comparators')
#      REF=0 skips hf / tiktoken (and the exactness check: tools/bench/e2e_table.py then takes exactness from
#      another log of the same host with the same ids sha)   GIGA=0 skips gigatoken (default: on when
#      build/giga-target/release/gigatoken-bench exists)
#      STAGES=1 adds the e2e-stages breakdown, COUNT=1 the e2e-count traffic pass (linux; tools/bench/stages.c;
#      the cl100k and o200k kernels' call sites; E2E_STATE=cold: both in the cold state)
#      PINCPU: the pinned cpu, for the receipts   GIT_SHA: the commit (tools/remote.sh syncs without .git)
#      TOKS_HOST_KEY: this machine's chipset key (docs/machines.md), the HOST line (default: uname -m); paths under
#      $HOME print as $HOME/...
#      TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers: the pinned files, tools/corpora/fetch_tokenizers.py)
#      TOKS_KIMI_DIR (~/.cache/toks/kimik3: a directory toks_load takes)   TOKS_BENCH_TEXT (build/text: the
#      corpus files, tools/bench/corpus.sha256)
# A tokenizer missing on this host prints SKIP and its cells are absent from the log (the table says so).
set -e
PIN="$1"
. tools/bench/common.sh
TOKS_LIST=${TOKS_LIST:-"gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 kimik3"}
CORPORA=${CORPORA:-"en code ml cjk"}
CHUNKS=${CHUNKS:-"4096 0"}
REPS=${REPS:-5}
REF=${REF:-1}
REF_REPS=${REF_REPS:-3}
STAGES=${STAGES:-0}
COUNT=${COUNT:-0}
GIGA_BIN=${GIGA_BIN:-build/giga-target/release/gigatoken-bench}
if [ -z "${GIGA+x}" ]; then if [ -x "$GIGA_BIN" ]; then GIGA=1; else GIGA=0; fi; fi
CC=${CC:-clang}
# python only through uv, its pythons and caches under build/ (lab-host hygiene)
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-only-managed}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$PWD/build/uv-python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/build/uv-cache}
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }

# ---- build: the shipping library, the bench binary, and (STAGES=1) the instrumented twin ---------------
make -j8 lib >/dev/null
BD=$(mkvar BUILD_DIR)
ISA=$(mkvar ISA)
CPPF=$(mkvar CPPFLAGS)
case $ISA in x86_64) NATIVE=-march=native ;; *) NATIVE=-mcpu=native ;; esac
$CC -std=c17 -O3 $NATIVE -Wall -Wextra -Werror $CPPF -o build/e2e tools/bench/e2e.c "$BD/libtoks.a" -lpthread
STAGE_RC=""
for k in k1_added_find:k1 k3_scan_cl100k:k3 k3_scan_o200k:k3o k5_encode:k5; do
    for tier in c neon avx2 avx512; do STAGE_RC="$STAGE_RC -Dtoks_${k%%:*}_$tier=tb_${k##*:}_$tier"; done
done
instrumented() {   # $1 = output dir, $2 = extra flags for every library TU, $3 = the binary, $4 = its -D flags
    rm -rf "$1"
    mkdir -p "$1"
    for f in src/core/*.c src/platform/*.c src/gen/*.c src/par/*.c; do
        [ -e "$f" ] || continue
        case $f in
            src/core/api.c|src/core/segment.c) X=$STAGE_RC ;;
            src/core/k5_long.c) X="-Dtoks_k6_bpe_c=tb_k6_c -Dtoks_k6_bpe_neon=tb_k6_neon -Dtoks_k6_bpe_avx2=tb_k6_avx2" ;;
            *) X="" ;;
        esac
        # shellcheck disable=SC2086
        $CC $2 $X -c "$f" -o "$1/$(echo "$f" | tr / _).o"
    done
    for f in src/asm/"$ISA"/*.S; do
        [ -e "$f" ] || continue
        $CC $(mkvar TFLAGS) $(mkvar PIC) $CPPF -Isrc/asm/"$ISA" -c "$f" -o "$1/$(echo "$f" | tr / _).o"
    done
    # shellcheck disable=SC2086
    $CC -std=c17 -O3 $NATIVE -Wall -Wextra -Werror $4 $CPPF $(mkvar HAVE) -o "$3" tools/bench/e2e.c tools/bench/stages.c "$1"/*.o -lpthread
}
if [ "$STAGES" = 1 ]; then
    instrumented "build/stages-$(basename "$BD")" "$(mkvar TFLAGS) $(mkvar CSTRICT) $(mkvar PIC) $(mkvar HARDEN) -fvisibility=hidden $CPPF $(mkvar HAVE)" \
        build/e2e-stages "-DE2E_STAGES"
fi
if [ "$COUNT" = 1 ]; then   # the counting build: the c of the library instrumented, the asm as shipped
    instrumented "build/count-$(basename "$BD")" "$(mkvar TFLAGS) $(mkvar CSTRICT) $(mkvar PIC) -fvisibility=hidden $CPPF $(mkvar HAVE) -fsanitize-coverage=func,trace-loads,trace-stores -Dmemcpy=tb_cmemcpy -Dmemset=tb_cmemset -Dmemcmp=tb_cmemcmp" \
        build/e2e-count "-DE2E_STAGES -DTB_COUNT"
fi

# ---- receipts ------------------------------------------------------------------------------------------
host_lines
echo "PIN '${PIN}' PINCPU '${PINCPU}' TOKS_TIER '${TOKS_TIER}' REPS $REPS REF $REF REF_REPS $REF_REPS GIGA $GIGA"
echo "GIT ${GIT_SHA:-unknown}"
echo "CC $($CC --version | head -1)"
echo "KERNELS $(cat "$BD/have.txt")"
$SHA build/e2e "$BD/libtoks.a" $( [ "$STAGES" = 1 ] && echo build/e2e-stages ) $( [ "$COUNT" = 1 ] && echo build/e2e-count ) \
    $( [ "$GIGA" = 1 ] && echo "$GIGA_BIN" ) | sed 's/^/BIN /'
[ "$GIGA" = 1 ] && echo "GIGATOKEN $(cat build/third-party/gigatoken/.toks-rev 2>/dev/null || echo unknown) $(cat build/giga-target/rustc.txt 2>/dev/null)"
for tk in $TOKS_LIST; do
    p=$(tpath "$tk")
    if [ -e "$p" ]; then echo "TOKENIZER $tk $(tsha "$p")"; else echo "TOKENIZER $tk SKIP missing $(rel "$p")"; fi
done
corpus_lines $CORPORA
echo "UPTIME $(uptime)"

# ---- cells ---------------------------------------------------------------------------------------------
nx=0 ni=0 voids=""                # exact cells, ids compared in them, void cells (REF=1)
for tk in $TOKS_LIST; do
    P=$(tpath "$tk")
    [ -e "$P" ] || continue
    for corp in $CORPORA; do
        F=$(files $corp)
        W=$(others $corp)
        for ch in $CHUNKS; do
            echo "== $tk $corp chunk $ch"
            echo "LOAD before $(loadavg)"
            pinned
            s0=$(cpustat)
            rm -f build/ids.u32 build/giga.u32
            # shellcheck disable=SC2086
            E2E_IDS_OUT=build/ids.u32 E2E_WARM_ON="$W" $PIN ./build/e2e "$P" "$ch" "$REPS" $F | tee build/cell.out
            G=""
            if [ "$GIGA" = 1 ]; then
                # shellcheck disable=SC2086
                GIGA_WARM_ON="$W" $PIN "$GIGA_BIN" "$P" "$ch" "$REPS" build/giga.u32 $F || echo "GIGA tool=gigatoken na=exit_$?"
                [ -s build/giga.u32 ] && G="--giga build/giga.u32"
            fi
            if [ "$REF" = 1 ]; then
                # shellcheck disable=SC2086
                $PIN uv run -q tools/bench/e2e_ref.py "$P" "$ch" "$REF_REPS" $F --ids build/ids.u32 $G | tee build/ref.out
                a=$(sed -n 's/^E2E .* ids=\([0-9]*\) sha=\([0-9a-f]*\).*/\1 \2/p' build/cell.out)
                b=$(sed -n 's/^REF tool=hf .* ids=\([0-9]*\) sha=\([0-9a-f]*\).*/\1 \2/p' build/ref.out)
                if [ -n "$a" ] && [ "$a" = "$b" ] && [ "${a%% *}" -gt 0 ]; then
                    echo "EXACT yes toks == hf ($a)"
                    nx=$((nx + 1)) ni=$((ni + ${a%% *}))
                else
                    echo "EXACT NO toks ($a) hf ($b): cell void"
                    voids="$voids $tk/$corp/$ch"
                fi
            fi
            if [ "$STAGES" = 1 ]; then                 # the same OTHER text as e2e's pass (pass_after=other)
                # shellcheck disable=SC2086
                E2E_WARM_ON="$W" $PIN ./build/e2e-stages "$P" "$ch" "$REPS" $F
            fi
            if [ "$COUNT" = 1 ]; then
                # shellcheck disable=SC2086
                $PIN ./build/e2e-count "$P" "$ch" 1 $F
            fi
            pinned
            busy "$s0" "$(cpustat)"
            echo "LOAD after $(loadavg)"
        done
    done
done
echo "UPTIME $(uptime)"
# the count line (REF=1): every cell's ids compared with hf's, none void, more than nothing compared (REF=0 compares
# nothing and claims nothing)
if [ "$REF" = 1 ]; then
    why=""
    [ -z "$voids" ] || why="void:$voids"
    [ "$ni" -gt 0 ] || why="${why:+$why; }nothing compared"
    if [ -z "$why" ]; then echo "PASS $ni compared e2e exact ($nx cells, toks == hf)"; else echo "FAIL $ni compared e2e exact ($why)"; exit 1; fi
fi
