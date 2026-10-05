#!/bin/sh
# tests/state/run.sh: SPEC T2 (state independence) on this host: every ledger target that compiles x the scalar
# and the AUTO tier x test_state's cells (fresh, warm, warm32, memo, rebind, migrate, errors, wrap, mode, par, and
# tiermix), plus the degenerate-hash build (test_state_hash: every piece cache one bucket, constant long / memo
# hashes), over the parity quick set's texts (tests/state/texts.py) + each target's added-token texts. Every call is
# compared with the scalar tier's cold result; a mismatch prints its text in hex (logs) and fails the run.
#
#   tests/state/run.sh [out-dir]          (default build/state; run from the repository root)
#   env: JOBS (parallel targets, default 4), CPUS (taskset list, e.g. 0-4,10-14), THREADS (toks_par participants
#        per target, default 2), STRIDE (every k-th text, default 1), HSTRIDE (the hash build's, default 4),
#        CP_SAMPLE (codepoint sweep scalars, default 4000), TARGETS (ledger names, default all that compile),
#        TOKS_TOKENIZER_CACHE (default ~/.cache/toks/tokenizers)
# Output: <out>/summary.txt (a line per target and cell), <out>/logs/<target>{,.hash}.log; exit 1 on a mismatch.
set -eu
OUT=${1:-build/state}
JOBS=${JOBS:-4}
THREADS=${THREADS:-2}
STRIDE=${STRIDE:-1}
HSTRIDE=${HSTRIDE:-4}
CP_SAMPLE=${CP_SAMPLE:-4000}
CACHE=${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}
GREP=/usr/bin/grep
rm -rf "$OUT/logs" "$OUT/models"
mkdir -p "$OUT/logs" "$OUT/models"
OUT=$(cd "$OUT" && pwd)

case $(uname -s) in Darwin) os=macos ;; Linux) os=linux ;; *) os=windows ;; esac
case $(uname -m) in arm64|aarch64) isa=arm64 ;; *) isa=x86_64 ;; esac
BD=${BUILD_DIR:-build/$os-$isa}
make -j"$JOBS" "$BD/tests/test_state" "$BD/tests/test_state_hash" > "$OUT/build.log" 2>&1 || { tail -20 "$OUT/build.log"; exit 2; }
python3 tests/state/texts.py --out "$OUT/texts.bin" --cp-sample "$CP_SAMPLE"

# the targets: name file family, from the ledger (compiles only); tiktoken files get a model directory of links.
# all.txt: every compiling target present on this host (the other tokenizers are chosen from it); targets.txt: the
# ones TARGETS selects (default all)
: > "$OUT/targets.txt"
: > "$OUT/all.txt"
$GREP -v '^#' tests/data/targets/ledger.txt | while read -r name file sha kind rest; do
    [ "$kind" = compiles ] || continue
    sel=1
    if [ -n "${TARGETS:-}" ]; then case " $TARGETS " in *" $name "*) ;; *) sel=0 ;; esac; fi
    case $file in
    *.tiktoken)
        stem=${file%.tiktoken}; d="$OUT/models/$stem"; mkdir -p "$d"
        case $stem in
        qwen1*) ln -sf "$CACHE/$file" "$d/qwen.tiktoken"; ln -sf "$CACHE/${stem}_tokenization_qwen.py" "$d/tokenization_qwen.py" ;;
        *) ln -sf "$CACHE/$file" "$d/tiktoken.model"; ln -sf "$CACHE/${stem}_tokenization_kimi.py" "$d/tokenization_kimi.py" ;;
        esac
        ln -sf "$CACHE/${stem}_tokenizer_config.json" "$d/tokenizer_config.json"
        path=$d ;;
    *) path=$CACHE/$file ;;
    esac
    [ -e "$path" ] || { [ "$sel" = 0 ] || echo "SKIP $name: $path missing" >> "$OUT/skipped.txt"; continue; }
    case $file in wp-*) fam=wp ;; uni_*) fam=uni ;; gemma4*|spm-*) fam=spm ;; *) fam=bl ;; esac
    echo "$name $path $fam" >> "$OUT/all.txt"
    [ "$sel" = 0 ] || echo "$name $path $fam" >> "$OUT/targets.txt"
done

# each target's other tokenizer (its scratch is rebound to it): the next target of all.txt of the same family (cache
# entries of the same format, another vocabulary), circular; a family of one takes the next target of any family.
# From all.txt, not targets.txt: a TARGETS of one tokenizer would otherwise be its own other, and the errors cell's
# "another context" call would pass a scratch bound to that very tokenizer (TOKS_E_SCRATCH expected, a success seen)
awk 'NR == FNR { n[NR] = $1; p[NR] = $2; f[NR] = $3; N = NR; next }
     { for (i = 1; i <= N && n[i] != $1; i++) ;
       o = "";
       for (k = 1; k < N && o == ""; k++) { j = (i + k - 1) % N + 1; if (f[j] == f[i]) o = p[j] }
       if (o == "") o = p[i % N + 1];
       print $1, $2, o }' "$OUT/all.txt" "$OUT/targets.txt" > "$OUT/jobs.txt"

cat > "$OUT/one.sh" <<EOF
#!/bin/sh
# one target: the library as built, then the degenerate-hash build
name=\$1 path=\$2 other=\$3
$BD/tests/test_state --target "\$path" --other "\$other" --texts "$OUT/texts.bin" --stride $STRIDE --threads $THREADS \
    > "$OUT/logs/\$name.log" 2>&1; a=\$?
$BD/tests/test_state_hash --target "\$path" --other "\$other" --texts "$OUT/texts.bin" --stride $HSTRIDE --threads $THREADS \
    > "$OUT/logs/\$name.hash.log" 2>&1; b=\$?
echo "\$name \$a \$b"
EOF
chmod +x "$OUT/one.sh"
PIN=""
if [ -n "${CPUS:-}" ] && command -v taskset > /dev/null 2>&1; then PIN="taskset -c $CPUS"; fi
start=$(date +%s)
load0=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime | sed 's/.*average[s]*: //')
$PIN xargs -r -P "$JOBS" -L 1 "$OUT/one.sh" < "$OUT/jobs.txt" > "$OUT/status.txt"   # -r: no targets, no run
load1=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime | sed 's/.*average[s]*: //')

# the table: per target and build, every cell's calls compared and mismatches (logs' "cell" lines)
{
    echo "# T2 state matrix: $(hostname) $(date -u +%Y-%m-%dT%H:%MZ), $(wc -l < "$OUT/jobs.txt") targets, wall $(( $(date +%s) - start )) s"
    aff=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')
    echo "# load before $load0, after $load1; JOBS=$JOBS THREADS=$THREADS STRIDE=$STRIDE HSTRIDE=$HSTRIDE CP_SAMPLE=$CP_SAMPLE cpus ${CPUS:-${aff:-all}}"
    echo "# $(python3 --version 2>&1) (tests/state/texts.py: $(wc -c < "$OUT/texts.bin") bytes), $(git rev-parse --short HEAD 2>/dev/null || echo "${GIT_SHA:-?}")"
    echo "# columns: target build cell tier calls mismatches [memo hits / K5 cache hits] cpu"
    for f in "$OUT"/logs/*.log; do
        b=plain; case $f in *.hash.log) b=hash ;; esac
        $GREP '^cell ' "$f" | awk -v b="$b" '{ n = split($2, a, "/"); t = a[n]; print t, b, $3, $4, $5, $6, $7, $8 }'
    done | sort
} > "$OUT/summary.txt"
awk '!/^#/ { c[$2 " " $3 " " $4] += $5; m[$2 " " $3 " " $4] += $6; C += $5; M += $6 }
     END { for (k in c) printf "%-28s %12d calls %6d mismatches\n", k, c[k], m[k] | "sort"; close("sort");
           printf "TOTAL %d calls compared, %d mismatches\n", C, M }' "$OUT/summary.txt" > "$OUT/cells.txt"
cat "$OUT/cells.txt"
bad=$(awk '$2 != 0 || $3 != 0' "$OUT/status.txt")
[ -z "$bad" ] || { echo "FAILED (target plain-exit hash-exit):"; echo "$bad"; }
# the count line: every target's plain and hash logs must have compared calls (a build or target that ran nothing,
# or a matrix of no targets, would otherwise read as a pass)
nt=$(wc -l < "$OUT/jobs.txt" | tr -d ' ')
zero=""
for name in $(awk '{ print $1 }' "$OUT/jobs.txt"); do
    for f in "$OUT/logs/$name.log" "$OUT/logs/$name.hash.log"; do
        c=$($GREP '^cell ' "$f" 2>/dev/null | awk '{ s += $5 } END { print s + 0 }')
        [ "$c" -gt 0 ] || zero="$zero $(basename "$f" .log)"
    done
done
C=$(awk '/^TOTAL/ { print $2 }' "$OUT/cells.txt")
M=$(awk '/^TOTAL/ { print $5 }' "$OUT/cells.txt")
why=""
[ -z "$bad" ] || why="a target exited nonzero"
[ "${M:-0}" = 0 ] || why="${why:+$why; }$M mismatches"
[ "$nt" -gt 0 ] || why="${why:+$why; }no targets"
[ -z "$zero" ] || why="${why:+$why; }0 calls compared in$zero"
[ "${C:-0}" -gt 0 ] || why="${why:+$why; }nothing compared"
if [ -z "$why" ]; then
    echo "PASS $C compared state ($nt targets, plain + hash builds, every cell vs the scalar tier's cold result)"
else
    echo "FAIL ${C:-0} compared state ($why)"
    exit 1
fi
