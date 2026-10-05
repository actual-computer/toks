#!/bin/sh
# tools/bench/tokv1.sh: the incumbent row on this host: toks vs tok v1 (the incumbent tokenizer: x86-64 asm with
# avx-512, arm64 asm with sve2 + neon) on the same chunks in one binary (tools/bench/tokv1.c),
# one pinned core, the sides alternating per rep, ids compared per call outside every timer (a difference voids
# the cell). E_DIR is a checkout of the incumbent's source (not public; a git archive of its host/, tests/host/ and
# tools/tokpack/ suffices) given by path: an optional bench input like gigatoken, never a build or test dependency;
# it is only read.
#
#   E_DIR=<e> tools/bench/tokv1.sh [pin]        e.g. E_DIR=build/e PINCPU=10 tools/bench/tokv1.sh "taskset -c 10"
#
# env: E_DIR (required; E_REV names its commit when it is not a git checkout, else E_DIR/.e-rev is read)
#      TOKV1_LIST (qwen38: the tokenizers tok v1 packs: qwen38 = version 3, glm53 = version 4)
#      CORPORA (en code zh ml cjk; zh = wiki-zh + the Gutenberg Chinese novel)  CHUNKS (4096 0)  REPLAY (1: one
#      conversation-replay cell per tokenizer, tools/bench/tokv1.c)  REPS (21)
#      NASM (nasm; x86-64 only)  PINCPU, GIT_SHA, TOKS_TOKENIZER_DIR, TOKS_BENCH_TEXT as tools/bench/e2e.sh
# Each tokenizer's tok.bin is packed once by e's tools/tokpack/tokpack.py (hf tokenizers 0.23.2 through uv) into
# build/tokv1/<name>-<json sha>-<e rev>.tok.bin. lang = warmed on the other corpora's files (never a measured file).
set -e
PIN="$1"
[ -n "$E_DIR" ] && [ -d "$E_DIR/host" ] || { echo "tokv1.sh: E_DIR must name an e checkout (host/, tests/host/, tools/tokpack/)" >&2; exit 2; }
T=${TOKS_BENCH_TEXT:-build/text}
K=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
TOKV1_LIST=${TOKV1_LIST:-qwen38}
CORPORA=${CORPORA:-"en code zh ml cjk"}
CHUNKS=${CHUNKS:-"4096 0"}
REPLAY=${REPLAY:-1}
REPS=${REPS:-21}
CC=${CC:-clang}
NASM=${NASM:-nasm}
export UV_PYTHON_PREFERENCE=${UV_PYTHON_PREFERENCE:-only-managed}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$PWD/build/uv-python}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/build/uv-cache}
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
if [ -n "$E_REV" ]; then :; elif git -C "$E_DIR" rev-parse HEAD >/dev/null 2>&1; then E_REV=$(git -C "$E_DIR" rev-parse --short=7 HEAD);
elif [ -r "$E_DIR/.e-rev" ]; then E_REV=$(cat "$E_DIR/.e-rev"); else E_REV=unknown; fi
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }

# ---- build: libtoks.a, tok v1's object from E_DIR, the bench binary --------------------------------------
make -j8 lib >/dev/null
BD=$(mkvar BUILD_DIR)
ISA=$(mkvar ISA)
B=build/tokv1
mkdir -p $B
case $ISA in
    x86_64)
        NATIVE=-march=native
        $NASM -f elf64 -w+all -w-unknown-warning -w-reloc-rel-dword -w-reloc-abs-dword -w-reloc-abs-qword \
            -I"$E_DIR/host/inc/" "$E_DIR/host/tok.asm" -o $B/tok.o
        ASV="$($NASM -v)" ;;
    *)
        NATIVE=-mcpu=native
        gcc -c -x assembler-with-cpp -Wa,--fatal-warnings -I"$E_DIR/host/a64/inc" "$E_DIR/host/a64/tok.S" -o $B/tok.o
        ASV="$(gcc --version | head -1)" ;;
esac
$CC -std=c17 -O2 $NATIVE -no-pie -Wall -Wextra -Werror $(mkvar CPPFLAGS) -I"$E_DIR/tests/host" -o $B/tokv1 \
    tools/bench/tokv1.c $B/tok.o "$BD/libtoks.a" -lpthread
pack() {   # $1 = tokenizer name -> the tok.bin path (packed once per json sha and e rev)
    j="$K/$1"
    out="$B/$1-$($SHA "$j" | cut -c1-12)-$E_REV.tok.bin"
    if [ ! -s "$out" ]; then
        uv run -q --python 3.13 --with tokenizers==0.23.2 --with numpy python "$E_DIR/tools/tokpack/tokpack.py" "$j" "$out" \
            > "$out.log" 2>&1 || { echo "tokv1.sh: tokpack failed for $1 (see $out.log)" >&2; rm -f "$out"; return 1; }
    fi
    echo "$out"
}

# ---- receipts (as tools/bench/e2e.sh) --------------------------------------------------------------------
loadavg() { cut -d' ' -f1-3 /proc/loadavg; }
pinned() {
    [ -n "$PINCPU" ] || return 0
    sib=$(cat /sys/devices/system/cpu/cpu"$PINCPU"/topology/thread_siblings_list 2>/dev/null || echo "$PINCPU")
    echo "PINNED cpu $PINCPU siblings $sib: $(ps -eLo psr,stat,pcpu,comm --no-headers | awk -v s="$sib" \
        'BEGIN{n=split(s,a,/[,-]/); for(i=1;i<=n;i++) c[a[i]]=1} ($1 in c) && $2 ~ /R/ {printf "%s/%s/%s ", $1, $4, $3}')"
}
cpustat() {
    [ -n "$PINCPU" ] || return 0
    sib=$(cat /sys/devices/system/cpu/cpu"$PINCPU"/topology/thread_siblings_list 2>/dev/null || echo "$PINCPU")
    for c in $(echo "$sib" | tr ',' ' '); do
        awk -v c="cpu$c" '$1 == c {b = $2 + $3 + $4 + $7 + $8 + $9; t = b + $5 + $6; print c, b, t}' /proc/stat
    done
}
busy() {
    [ -n "$1" ] || return 0
    echo "BUSY $(printf '%s\n%s\n' "$1" "$2" | awk '{if ($1 in b) {db = $2 - b[$1]; dt = $3 - t[$1]; v = 0; if (dt > 0) v = 100 * db / dt; printf "%s %.0f%% ", $1, v} else {b[$1] = $2; t[$1] = $3}}')"
}
echo "HOST ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"
echo "CPU $(/usr/bin/grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) microcode $(/usr/bin/grep -m1 microcode /proc/cpuinfo | cut -d: -f2-)"
echo "OS $(uname -v)"
[ -n "$PINCPU" ] && echo "GOVERNOR cpu$PINCPU $(cat /sys/devices/system/cpu/cpu"$PINCPU"/cpufreq/scaling_governor 2>/dev/null) boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
echo "THP $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "PIN '${PIN}' PINCPU '${PINCPU}' TOKS_TIER '${TOKS_TIER}' REPS $REPS"
echo "GIT ${GIT_SHA:-unknown}"
echo "EREV $E_REV $E_DIR"
echo "CC $($CC --version | head -1)"
echo "AS $ASV"
echo "KERNELS $(cat "$BD/have.txt")"
$SHA $B/tokv1 $B/tok.o "$BD/libtoks.a" | sed 's/^/BIN /'
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
        ml) echo "$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt $T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt" ;;
        cjk) echo "$T/wiki-zh.txt $T/wiki-ja.txt $T/wiki-ko.txt $T/gut-zh-24264.txt" ;;
    esac
}
others() {   # every corpus file not in corpus $1's list (the lang state's warm-up text)
    o=""
    for c in en code ml cjk; do for f in $(files $c); do
        case " $(files "$1") $o " in *" $f "*) ;; *) o="$o $f" ;; esac
    done; done
    echo $o
}
for corp in $CORPORA; do
    # shellcheck disable=SC2046
    echo "CORPUS $corp $(cat $(files $corp) | $SHA | cut -c1-64) $(cat $(files $corp) | wc -c | tr -d ' ') bytes: $(files $corp)"
done
for tk in $TOKV1_LIST; do
    bin=$(pack "$tk") || continue
    echo "TOKENIZER $tk $($SHA "$K/$tk" | cut -c1-64) tokbin $($SHA "$bin" | cut -c1-64) $(/usr/bin/grep -o 'version [0-9]*' "$bin.log" 2>/dev/null | head -1)"
done
echo "UPTIME $(uptime)"

# ---- cells ---------------------------------------------------------------------------------------------
cell() {   # $1 tk, $2 corpus label, $3 chunk|replay, $4 files, $5 warm-up files, $6 tok.bin
    echo "== $1 $2 chunk $3"
    echo "LOAD before $(loadavg)"
    pinned
    s0=$(cpustat)
    # shellcheck disable=SC2086
    TOKV1_WARM_ON="$5" $PIN ./$B/tokv1 "$K/$1" "$6" "$3" "$REPS" $4 > "$B/cell.out" 2>&1 || echo "TOKV1 FAILED exit $?" >> "$B/cell.out"
    cat "$B/cell.out"
    grep "^TOKV1" "$B/cell.out" >> "$B/tokv1.lines" || echo "TOKV1 FAILED no TOKV1 line" >> "$B/tokv1.lines"
    pinned
    busy "$s0" "$(cpustat)"
    echo "LOAD after $(loadavg)"
}
: > "$B/tokv1.lines"            # every cell's TOKV1 line, for the count line at the end
for tk in $TOKV1_LIST; do
    bin=$(pack "$tk") || continue
    for corp in $CORPORA; do
        for ch in $CHUNKS; do cell "$tk" "$corp" "$ch" "$(files $corp)" "$(others $corp)" "$bin"; done
    done
    [ "$REPLAY" = 1 ] && cell "$tk" chat replay "$(files en)" "$(others en)" "$bin"
done
echo "UPTIME $(uptime)"
# the count line: every cell exact (ids compared per call, fresh and warm, against tok v1's), none failed, more than
# nothing compared
n=$(sed -n 's/^TOKV1 .* ids=\([0-9]*\) .*exact=yes.*/\1/p' "$B/tokv1.lines" | awk '{ s += $1 } END { print s + 0 }')
nc=$(grep -c "^TOKV1 .*exact=yes" "$B/tokv1.lines" || true)
bad=$(grep -c "^TOKV1 .*exact=NO\|^TOKV1 FAILED" "$B/tokv1.lines" || true)
why=""
[ "$bad" = 0 ] || why="$bad cells not exact or failed"
[ "$n" -gt 0 ] || why="${why:+$why; }nothing compared"
if [ -z "$why" ]; then
    echo "PASS $n compared tokv1 exact ($nc cells, toks == tok v1, fresh and warm)"
else
    echo "FAIL $n compared tokv1 exact ($why)"
    exit 1
fi
