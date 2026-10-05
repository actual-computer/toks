#!/bin/sh
# tools/release/rc_host.sh: one host's part of a toks release candidate (tools/release/rc.sh runs it on every host
# over ssh, from the host's copy of the tree). Every step logs to build/rc/<step>.log and appends one line to
# build/rc/steps.txt:   STEP <step> status=<exit> secs=<wall> load=<1-min before>-><after>
# A failing step is recorded and the next one runs (the report lists it as unmet); the exit status is the
# number of failed steps. A rerun of the same commit (GIT_SHA) replaces only the steps it runs (RC_STEPS /
# TARGETS subsets); another commit starts build/rc/ afresh.
#
#   test-<tier>        make -j test (tier auto), then every test program again under TOKS_TIER=<tier> per extra tier
#   san                (SAN=1) every test program under ASan + UBSan (make all, then each binary, tier auto), then
#                      test_par under TSan. Not make test: its object audits (asm_regs_audit, cf_audit) read the
#                      shipped objects, which test-auto audited; on instrumented objects they do not apply
#   parity-<t>-<tier>  a light parity sample per target, each tier: tests/parity/gen_cases.py --quick (seeded:
#                      byte-identical on every host, its sha-256 in the manifest), every 7th encode / pieces case
#                      (7 is prime to the flag cycles), every decode and stream case, plus gen_stream.py's
#                      adversarial stream cases; tests/parity/shards.sh against hf 0.23.2. kimik3: run_kimi.py
#                      against its own reference (transformers + tiktoken, pinned), every mode + decode.
#   wheels             python/build.sh over PYTHONS (tests: api + threads, hf parity on 3.13)
#   bench-<tier>       tools/bench/gigatoken.sh + tools/bench/e2e.sh pinned to BENCH_CPU, last and alone (the
#                      extra tiers without comparators: REF=0 GIGA=0)
#   rent-<tier>        (TIERS with scalar) every cell where bench-<tier> was slower than bench-scalar (cold or pass:
#                      tools/release/rc_rent.py), again as RENT_ROUNDS (5) abba rounds of tools/bench/e2e_ab.sh <tier>
#                      scalar on one build and core: two separate runs minutes apart are not a paired comparison
#   gate               (listed in RC_STEPS, not a default: ~2.5 h a host) the gigatoken gate (tools/bench/gate.sh,
#                      the default cache config, BENCH_TOKS on the bench core): ABBA blocks, the null blocks, the
#                      per-block void rule; rc.sh turns the hosts' logs into docs/bench/gigatoken-gate.md
#                      (tools/bench/gate_table.py), and the report's gate row is met only with this step's logs
#   tokv1              (TOKV1_E_DIR = an e checkout: linux only) T8's incumbent cells: tools/bench/tokv1.sh pinned
#                      like the bench (TOKV1_LIST, default qwen38 glm53 nemotron3-omni; sides tok v1 / toks / toks32 /
#                      toksm = TOKS_SCRATCH_MEMO_MIB(4), e's integration flags); TOKV1_NASM names nasm on x86-64
#
# env: TIERS ("auto scalar"; auto = the tier toks_load binds)  JOBS (8)  TASKSET (prefix for the heavy steps)
#      SAN (0)  TARGETS (gpt2 llama3 + the critical targets: glm53 qwen38 o200k gemma4 kimik3, one file per new
#      critical family: nemotron3-4b llama4 minimaxm2 dsv4)  PARITY_SHARDS (8)  PYTHONS (3.10 .. 3.14)
#      BENCH_PIN / BENCH_CPU (the timing core)  BENCH_TOKS (e2e.sh's TOKS_LIST)  RC_STEPS ("test san parity wheels
#      bench")  GIT_SHA  RUSTUP_SEED (an existing user-local rust install to copy, tools/bench/gigatoken.sh)
#      TOKS_HOST_KEY (this machine's chipset key, docs/machines.md: the host line of host.txt; rc.sh sets it)
# Receipts carry no home directory: paths under $HOME are written as $HOME/... (rel_home).
set -u
if [ "${1:-}" = corpus ]; then   # build/text == tools/bench/corpus.sha256; else copy a verified copy on this host
    if command -v sha256sum >/dev/null 2>&1; then CHK="sha256sum"; else CHK="shasum -a 256"; fi
    M=$PWD/tools/bench/corpus.sha256
    ok() { (cd "$1" 2>/dev/null && $CHK -c --status "$M") 2>/dev/null; }
    mkdir -p build/text
    ok build/text && { echo "corpus: ok"; exit 0; }
    for d in "$HOME/.cache/toks/bench-text" "$HOME"/toks-ci/*/build/text; do   # the staged copy first
        if ok "$d"; then
            cut -c67- "$M" | while read -r f; do cp "$d/$f" build/text/; done
            ok build/text && { echo "corpus: copied from $d"; exit 0; }
        fi
    done
    echo "corpus: missing"
    exit 1
fi
RUSTUP_SEED=${RUSTUP_SEED:-$(for d in "$HOME"/toks-ci/bench-e2e/build "$HOME"/toks-ci/research-ceiling/build; do
    [ -x "$d/cargo/bin/cargo" ] && echo "$d" && break; done)}
export RUSTUP_SEED
TIERS=${TIERS:-"auto scalar"}
JOBS=${JOBS:-8}
TASKSET=${TASKSET:-}
SAN=${SAN:-0}
TARGETS=${TARGETS:-"gpt2 llama3 glm53 qwen38 o200k gemma4 kimik3 nemotron3-4b llama4 minimaxm2 dsv4"}
PARITY_SHARDS=${PARITY_SHARDS:-8}
PYTHONS=${PYTHONS:-"3.10 3.11 3.12 3.13 3.14"}
RC_STEPS=${RC_STEPS:-"test san parity wheels bench"}   # the gate step is listed, not a default: ~2.5 h a host
export GIT_SHA=${GIT_SHA:-unknown}
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_INSTALL_DIR=$PWD/build/uv-python
export UV_CACHE_DIR=$PWD/build/uv-cache
TOKDIR=${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}
KIMIDIR=${TOKS_KIMI_DIR:-$HOME/.cache/toks/kimik3}
R=build/rc
mkdir -p $R
if ! grep -qx "git $GIT_SHA" $R/host.txt 2>/dev/null; then
    find $R -mindepth 1 -maxdepth 1 -exec rm -rf {} +
fi
touch $R/steps.txt
fails=0
loadavg() { if [ -r /proc/loadavg ]; then cut -d' ' -f1 /proc/loadavg; else sysctl -n vm.loadavg | awk '{print $2}'; fi; }
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }
step() {   # step <name> <command...>: run, log, record
    name=$1; shift
    l0=$(loadavg); t0=$(date +%s)
    "$@" > "$R/$name.log" 2>&1
    rc=$?
    grep -v "^STEP $name " $R/steps.txt > $R/steps.tmp; mv $R/steps.tmp $R/steps.txt
    echo "STEP $name status=$rc secs=$(( $(date +%s) - t0 )) load=$l0->$(loadavg)" >> $R/steps.txt
    [ $rc = 0 ] || fails=$((fails + 1))
    tail -n 3 "$R/$name.log" | sed "s/^/  [$name] /"
    return 0
}
tierenv() { [ "$1" = auto ] && echo "" || echo "TOKS_TIER=$1"; }
rel_home() {   # rel_home <file...>: this machine's home directory -> the literal $HOME inside receipt files
    for f in "$@"; do [ -f "$f" ] && perl -pi -e 's#\Q'"$HOME"'\E/#\$HOME/#g' "$f"; done
    return 0
}

{
    echo "host ${TOKS_HOST_KEY:-$(uname -m)} $(uname -srm)"
    if [ "$(uname)" = Darwin ]; then sysctl -n machdep.cpu.brand_string; else grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-; fi
    echo "uptime $(uptime)"
    echo "git $GIT_SHA"
    echo "cc $(clang --version | head -1)"
    echo "uv $(uv --version)"
    echo "tiers $TIERS jobs $JOBS taskset '$TASKSET' bench '${BENCH_PIN:-}' cpu ${BENCH_CPU:-}"
} > $R/host.txt

BD=$(mkvar BUILD_DIR)

# ---- make test, each tier ----------------------------------------------------------------------------
run_tests() {   # every test program in build dir $2 (default: the one make test built) under TOKS_TIER=$1; returns the failures
    nf=0
    for b in "${2:-$BD}"/tests/*; do
        [ -x "$b" ] || continue
        echo "== $b"
        env TOKS_TIER="$1" $TASKSET "$b" || { echo "FAILED $b (exit $?)"; nf=$((nf + 1)); }
    done
    return $nf
}
case " $RC_STEPS " in *" test "*)
    step test-auto $TASKSET make -j"$JOBS" test
    for t in $TIERS; do [ "$t" = auto ] || step "test-$t" run_tests "$t"; done ;;
esac

# ---- sanitizers --------------------------------------------------------------------------------------
san() {
    $TASKSET make -j"$JOBS" BUILD_DIR=build/rc-asan \
        CC="clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer" all || return 1
    run_tests auto build/rc-asan       # sets nf: the programs that failed; TSan runs either way
    $TASKSET make -j"$JOBS" BUILD_DIR=build/rc-tsan CC="clang -fsanitize=thread" build/rc-tsan/tests/test_par || return 1
    echo "== build/rc-tsan/tests/test_par"
    $TASKSET build/rc-tsan/tests/test_par || nf=$((nf + 1))
    return $nf
}
case " $RC_STEPS " in *" san "*) [ "$SAN" = 1 ] && step san san ;; esac

# ---- parity samples ----------------------------------------------------------------------------------
sample() {   # sample <in> <out>: every 7th encode / pieces line, every decode and stream line
    uv run -q --no-project python - "$1" "$2" <<'EOF'
import json, sys
k = {"encode": 7, "pieces": 7}
c = {}
with open(sys.argv[1]) as f, open(sys.argv[2], "w") as o:
    for line in f:
        op = json.loads(line)["op"]
        c[op] = c.get(op, 0) + 1
        if c[op] % k.get(op, 1) == 0:
            o.write(line)
print({op: (n + k.get(op, 1) - 1) // k.get(op, 1) for op, n in c.items()})
EOF
}
parity_target() {   # parity_target <t>: cases (once) then every tier
    t=$1
    tok=$TOKDIR/$t
    mkdir -p $R/cases
    if [ ! -s $R/cases/$t.jsonl ]; then
        $TASKSET uv run -q --with tokenizers==0.23.2 python tests/parity/gen_cases.py --tokenizer "$tok" \
            --out $R/cases/$t-quick.jsonl --quick > /dev/null || return 1
        sample $R/cases/$t-quick.jsonl $R/cases/$t.jsonl || return 1
        $TASKSET uv run -q --with tokenizers==0.23.2 python tests/parity/gen_stream.py --tokenizer "$tok" \
            --out $R/cases/$t-stream.jsonl --n 10000 || return 1
        cat $R/cases/$t-stream.jsonl >> $R/cases/$t.jsonl
        rm -f $R/cases/$t-quick.jsonl
    fi
    grep -v '"op": *"pieces"' $R/cases/$t.jsonl > $R/cases/$t-ids.jsonl
    grep '"op": *"pieces"' $R/cases/$t.jsonl > $R/cases/$t-pieces.jsonl
    echo "cases $(wc -l < $R/cases/$t.jsonl) (ids $(wc -l < $R/cases/$t-ids.jsonl), pieces $(wc -l < $R/cases/$t-pieces.jsonl)) quick-set $(sed -n 's/.*"sha256": "\([0-9a-f]*\)".*/\1/p' $R/cases/$t-quick.jsonl.manifest.json)"
    rc=0
    for tier in $TIERS; do
        for part in ids pieces; do   # ids = encode / decode / stream; pieces = toks_pieces' offsets, reported apart
            o=$R/parity/$t-$part-$tier
            rm -rf $o
            env $(tierenv "$tier") $TASKSET sh tests/parity/shards.sh "$tok" $R/cases/$t-$part.jsonl $R/toks_driver \
                $o "$PARITY_SHARDS" > $o.txt 2>&1 || rc=1
            rel_home $o/report.json $o.txt
            echo "tier $tier $part: $(tr -d '\n ' < $o.txt | grep -o '"n_cases":[0-9]*,"counts":{[^}]*}' | head -1) diffs $(tr -d '\n ' < $o.txt | grep -o '"n_diffs":[0-9]*' | head -1)"
        done
    done
    return $rc
}
parity_kimi() {
    rc=0
    for tier in $TIERS; do
        o=$R/parity/kimik3-$tier
        rm -rf $o
        mkdir -p $o
        i=0
        while [ $i -lt "$PARITY_SHARDS" ]; do
            env $(tierenv "$tier") TOKS_KIMI_DIR="$KIMIDIR" $TASKSET uv run -q --python 3.12 --with tiktoken==0.14.0 \
                --with transformers==5.18.0 python tests/parity/run_kimi.py --toks $R/toks_driver --kimi-dir "$KIMIDIR" --short 120000 \
                --long 8 --exh 3 --vocab 30000 --seed 1 --shard $i/"$PARITY_SHARDS" --out $o/shard-$i.json \
                > $o/shard-$i.log 2>&1 &
            i=$((i + 1))
        done
        wait
        uv run -q --no-project python - $o "$PARITY_SHARDS" "$tier" <<'EOF' || rc=1
import json, sys
o, n, tier = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tot, src, miss = {"texts": 0, "ids": 0, "bytes": 0, "mismatches": 0, "ref_errors": 0}, {}, []
for i in range(n):
    try:
        r = json.load(open(f"{o}/shard-{i}.json"))
    except (OSError, ValueError):
        miss.append(i)
        continue
    for k in ("ids", "bytes", "mismatches", "ref_errors"):
        tot[k] += r.get(k, 0)
    for s, c in r.get("cases", {}).items():
        d = src.setdefault(s, {"texts": 0, "mismatch": 0})
        d["texts"] += c.get("texts", 0)
        d["mismatch"] += c.get("mismatch", 0)
        tot["texts"] += c.get("texts", 0)
json.dump({"shards": n, "missing_shards": miss, "totals": tot, "sources": src}, open(f"{o}/report.json", "w"), indent=2)
print(f"tier {tier}: {json.dumps(tot)} sources {json.dumps(src)} missing {miss}")
sys.exit(1 if miss or tot["mismatches"] else 0)
EOF
    done
    return $rc
}
case " $RC_STEPS " in *" parity "*)
    rm -rf $R/parity
    mkdir -p $R/parity
    make -j"$JOBS" lib > /dev/null
    clang -std=c17 -O2 -Iinclude tests/driver/toks_driver.c "$BD/libtoks.a" -lpthread -o $R/toks_driver
    for t in $TARGETS; do
        if [ "$t" = kimik3 ]; then step parity-kimik3 parity_kimi; else step "parity-$t" parity_target "$t"; fi
    done ;;
esac

# ---- python wheels -----------------------------------------------------------------------------------
case " $RC_STEPS " in *" wheels "*) step wheels env TASKSET="$TASKSET" PARITY_PY=3.13 sh python/build.sh $PYTHONS ;; esac

# ---- the speed table (last, alone) -------------------------------------------------------------------
bench() {   # bench <tier>
    tier=$1
    if [ "$tier" = auto ]; then
        sh tools/bench/gigatoken.sh > $R/gigatoken-build.log 2>&1 || echo "gigatoken: build failed ($R/gigatoken-build.log), its columns stay n/a" >&2
        env PINCPU="${BENCH_CPU:-}" TOKS_LIST="${BENCH_TOKS:-gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4 kimik3}" \
            sh tools/bench/e2e.sh "${BENCH_PIN:-}"
    else
        env TOKS_TIER="$tier" REF=0 GIGA=0 PINCPU="${BENCH_CPU:-}" \
            TOKS_LIST="${BENCH_TOKS:-gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4 kimik3}" \
            sh tools/bench/e2e.sh "${BENCH_PIN:-}"
    fi
}
rent() {   # rent <tier>: the paired re-measure of the cells rc_rent.py lists (none: nothing to do)
    cells=$(uv run -q --no-project tools/release/rc_rent.py "$R/bench-$1.log" "$R/bench-scalar.log") || return 1
    [ -n "$cells" ] || { echo "no cell slower than the c twin: nothing to re-measure"; return 0; }
    echo "$cells" | while read -r tier tk corp ch; do
        env TOKS_LIST="$tk" CORPORA="$corp" CHUNKS="$ch" ROUNDS="${RENT_ROUNDS:-5}" GIT_SHA="$GIT_SHA" \
            sh tools/bench/e2e_ab.sh "$tier" scalar "${BENCH_PIN:-}" || return 1
    done
}
gate() {   # the gigatoken gate on the bench core, the default config (the matched rows are a separate run)
    env PINCPU="${BENCH_CPU:-}" CONFIGS=default \
        TOKS_LIST="${BENCH_TOKS:-gpt2 llama3 glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4 kimik3}" \
        sh tools/bench/gate.sh "${BENCH_PIN:-}"
}
tokv1() {   # the incumbent cells (tools/bench/tokv1.sh): toks vs tok v1 in one binary, the bench core
    env E_DIR="$TOKV1_E_DIR" NASM="${TOKV1_NASM:-nasm}" PINCPU="${BENCH_CPU:-}" REPS="${TOKV1_REPS:-21}" \
        TOKV1_LIST="${TOKV1_LIST:-qwen38 glm53 nemotron3-omni}" sh tools/bench/tokv1.sh "${BENCH_PIN:-}"
}
case " $RC_STEPS " in *" bench "*)
    for t in $TIERS; do step "bench-$t" bench "$t"; done
    case " $TIERS " in *" scalar "*) for t in $TIERS; do [ "$t" = scalar ] || step "rent-$t" rent "$t"; done ;; esac
    if [ -n "${TOKV1_E_DIR:-}" ] && [ -d "${TOKV1_E_DIR:-}/host" ]; then step tokv1 tokv1; fi ;;
esac
case " $RC_STEPS " in *" gate "*) step gate gate ;; esac   # after bench: its gigatoken.sh built the comparator
echo "rc_host: $fails failed step(s); build/rc/steps.txt:"
cat $R/steps.txt
exit $fails
