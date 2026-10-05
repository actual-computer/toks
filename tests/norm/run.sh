#!/bin/sh
# tests/norm/run.sh: the NFC differential against hf tokenizers 0.23.2 (every family of tests/norm/gen.py,
# piped into tests/norm/nfc_check) and the single-core bench, on a lab host. Never on a laptop.
#
#   tools/remote.sh <host> 'TOKS_SHA=<commit> sh tests/norm/run.sh check 4'   # differential, 4 jobs at a time
#   tools/remote.sh <host> 'TOKS_SHA=<commit> sh tests/norm/run.sh bench 7'   # bench pinned to cpu 7 (taskset)
#   (REPS=n overrides the 31 timed passes; with build/norm/qwen38.tokenizer.json present the bench adds the
#   driver plan's bytes and toks_encode MB/s with NFC on / off)
#
# The oracle is normalizers.NFC(), and for a second pass Qwen3.8-27B's own tokenizer.json normalizer (pinned
# revision + sha256 in gen.py): the target model itself. Logs land in build/norm/; the corpus
# (tests/norm/corpus.py) in build/norm-corpus/ (a re-run fetches only what is missing).
#
# What passes: every job's nfc_check exits 0 with 0 mismatches, every gen.py exits 0, and no job reads 0 records.
# The last two are there because from 34bb3b5 (the NFKC tables: tools/gen/norm.py's parse_tables went from three
# results to five) to e6e087a this script passed on nothing in 17 of its 22 jobs: gen.py's crate_sets still unpacked
# three, so marks2, blocked, pairs, corpus and adversarial (and their qwen38 passes) died in the generator, nfc_check
# read an empty stream and exited 0, and only nfc_check's exit was looked at. Only alone, hangul and normtest (and
# the qwen38 passes of the last two) checked anything in that window. Re-run at e6e087a with gen.py fixed: 22 jobs,
# 313,357 records, 3.47 GB, 0 mismatches.
set -eu
cd "$(dirname "$0")/../.."
what=${1:-check}
arg=${2:-4}
B=build/norm
mkdir -p "$B"
make -s BUILD_DIR="$B/lib" lib
CFLAGS="-std=c17 -O2 -fno-strict-aliasing -fwrapv -Wall -Wextra -Werror -Iinclude -Isrc/core -Isrc/platform"
clang $CFLAGS -o "$B/nfc_check" tests/norm/nfc_check.c "$B/lib/libtoks.a"
clang $CFLAGS -o "$B/nfc_bench" tests/norm/nfc_bench.c "$B/lib/libtoks.a"
GEN="uv run tests/norm/gen.py"
fetch() {   # the corpus (only what is missing) and the qwen38 oracle file
    uv run tests/norm/corpus.py build/norm-corpus 3 > "$B/corpus.log"
    $GEN fetch --oracle qwen38 2> "$B/oracle.txt"
}

echo "host $(hostname), $(uname -sm), $(uptime)"
sha() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -c1-16; else shasum -a 256 "$1" | cut -c1-16; fi; }
echo "commit ${TOKS_SHA:-unknown}, libtoks.a sha256 $(sha "$B/lib/libtoks.a")"

running=0
job() {   # job TAG FAMILY [gen.py options]; the log ends with nfc_check's exit and gen.py's (a dead generator feeds
          # nfc_check nothing: 0 records, exit 0, so both exits and the record count decide)
    tag=$1; shift
    ( { $GEN "$@" 2> "$B/$tag.gen"; echo "gen exit $?" > "$B/$tag.genexit"; } | "$B/nfc_check" "$tag" > "$B/$tag.log" 2>&1
      echo "exit $?" >> "$B/$tag.log"; cat "$B/$tag.genexit" >> "$B/$tag.log" ) &
    running=$((running + 1))
    if [ "$running" -ge "$arg" ]; then wait; running=0; fi
}

if [ "$what" = check ]; then
    fetch
    echo "corpus $(wc -l < build/norm-corpus/MANIFEST) languages, $(cat build/norm-corpus/*.jsonl | wc -c) bytes"
    rm -f "$B"/*.log "$B"/*.gen "$B"/*.genexit
    for fam in alone marks2 blocked hangul normtest corpus adversarial; do job "$fam" "$fam"; done
    for k in 0 1 2 3 4 5 6 7; do job "pairs_$k" pairs --shard "$k/8"; done
    for fam in marks2 blocked hangul normtest adversarial; do job "qwen38_$fam" "$fam" --oracle qwen38; done
    job qwen38_corpus corpus --oracle qwen38 --forms raw,nfd,shuffled
    job qwen38_pairs_0 pairs --shard 0/8 --oracle qwen38
    wait
    cat "$B"/oracle.txt "$B"/*.gen | grep -v '^$' | sort | uniq -c || true
    grep -h "nfc_check\|MISMATCH\|exit" "$B"/*.log || true
    echo "uptime after: $(uptime)"
    # the count line: records compared over every job; every job must have printed its nfc_check line
    nj=$(ls "$B"/*.log | wc -l | tr -d ' ')
    nl=$(cat "$B"/*.log | grep -c "^nfc_check [^:]*: [0-9]* records" || true)
    n=$(cat "$B"/*.log | sed -n 's/^nfc_check [^:]*: \([0-9]*\) records.*/\1/p' | awk '{ s += $1 } END { print s + 0 }')
    why=""
    ! grep -q "^exit [1-9]" "$B"/*.log || why="a checker exited nonzero"
    ! grep -q "^gen exit [1-9]" "$B"/*.log || why="${why:+$why; }a generator exited nonzero"
    ! grep -q ": 0 records" "$B"/*.log || why="${why:+$why; }a job read 0 records"
    [ "$nl" = "$nj" ] || why="${why:+$why; }$((nj - nl)) of $nj jobs printed no count"
    [ "$n" -gt 0 ] || why="${why:+$why; }nothing compared"
    if [ -z "$why" ]; then echo "PASS $n compared norm nfc ($nj jobs)"; else echo "FAIL $n compared norm nfc ($why)"; exit 1; fi
elif [ "$what" = bench ]; then
    for f in en ml cjk nfd; do [ -s "$B/bench-$f.bin" ] || { fetch; break; }; done
    [ -s "$B/bench-en.bin" ] || $GEN gutenberg > "$B/bench-en.bin" 2> "$B/bench-en.txt"
    [ -s "$B/bench-ml.bin" ] || $GEN corpus --forms raw > "$B/bench-ml.bin" 2> "$B/bench-ml.txt"
    [ -s "$B/bench-cjk.bin" ] || $GEN corpus --forms raw --langs zh,ja,ko > "$B/bench-cjk.bin" 2> "$B/bench-cjk.txt"
    [ -s "$B/bench-nfd.bin" ] || $GEN corpus --forms nfd > "$B/bench-nfd.bin" 2> "$B/bench-nfd.txt"
    PIN=""
    command -v taskset > /dev/null && PIN="taskset -c $arg"
    TOK=""                      # with the Qwen 3.8 file: the driver's plan and toks_encode with NFC on / off
    [ -s "$B/qwen38.tokenizer.json" ] && TOK="$B/qwen38.tokenizer.json"
    for f in en ml cjk nfd; do
        echo "load before: $(uptime | sed 's/.*load/load/'); bench-$f.bin sha256 $(sha "$B/bench-$f.bin")"
        $PIN "$B/nfc_bench" "$B/bench-$f.bin" "${REPS:-31}" $TOK
    done
    echo "load after: $(uptime | sed 's/.*load/load/')"
fi
