#!/bin/sh
# tools/bench/spm_stages_ids.sh <dirA> <dirB> "<toks>" "<corpora>" "<chunks>" [cpus] [jobs]: ids equality of two source
# trees over many tokenizers: spm_stages.c linked against each tree's library, one rep per (tok, corpus, chunk, side)
# run in parallel on the given cores, then the ids= of A and B compared per state (cold / pass / warm). Prints one line
# per tokenizer and a total; exit 1 on any difference (a file both sides refuse at load is listed, not a difference
# in ids).
#   e.g. tools/bench/spm_stages_ids.sh ~/toks-ci/master ~/toks-ci/my-branch "gemma4 mistral-v0.3" \
#        "en code ml zh" "4096 0" 0-4 5
# env: CC, TOKS_BENCH_TEXT (build/text), TOKS_TOKENIZER_DIR (~/.cache/toks/tokenizers)
set -e
A=$1; B=$2; TOKS=$3; CORPORA=$4; CHUNKS=$5; CPUS=${6:-0-4}; J=${7:-5}
R=$(cd "$(dirname "$0")/../.." && pwd)
CC=${CC:-clang}
W=$R/build/idc.$$
mkdir -p "$W/out"
for side in A B; do
    d=$A; [ $side = B ] && d=$B
    (cd "$d" && rm -rf build/idc && mkdir -p build/idc && cp -R Makefile include src build/idc/ && cd build/idc &&
     taskset -c "$CPUS" make -j"$J" lib > /dev/null)
    $CC -std=c17 -O2 -I"$d/include" -o "$W/$side" "$R/tools/bench/spm_stages.c" "$d"/build/idc/build/*/libtoks.a -lpthread
done
cat > "$W/one.sh" <<'EOF'
#!/bin/sh
W=$1; T=$2; K=$3; tok=$4; corp=$5; ch=$6; side=$7
case $corp in
    en) F="$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
    code) F="$T/code-cpython.py $T/code-toks.c" ;;
    ml) F="$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt $T/gut-de-2229.txt $T/gut-es-2000.txt $T/gut-fr-17489.txt" ;;
    zh) F="$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
esac
# shellcheck disable=SC2086
SP_VARIANT=$side "$W/$side" "$K/$tok" "$ch" 1 $F 2>&1 | sed "s/^/$tok $corp $ch $side /" > "$W/out/$tok.$corp.$ch.$side"
EOF
for tok in $TOKS; do for corp in $CORPORA; do for ch in $CHUNKS; do for side in A B; do
    echo "$tok $corp $ch $side"
done; done; done; done > "$W/jobs"
xargs -P "$J" -L 1 taskset -c "$CPUS" sh "$W/one.sh" "$W" "${TOKS_BENCH_TEXT:-$R/build/text}" \
    "${TOKS_TOKENIZER_DIR:-$HOME/.cache/toks/tokenizers}" < "$W/jobs"
cat "$W"/out/* > "$W/all.log"
set +e
python3 - "$W/all.log" <<'EOF'
import collections, re, sys
ids, refused, bad = collections.defaultdict(dict), collections.defaultdict(set), []
for line in open(sys.argv[1], errors="replace"):
    f = line.split()
    if len(f) < 6:
        continue
    if f[4] != "SP":
        if "toks_load" in line:
            refused[f[0]].add(f[3])
        elif "rror" in line or "toks_" in line:
            bad.append(line.strip())
        continue
    m = re.search(r"state=(\w+).* ids=([0-9a-f]+)", line)
    ids[(f[0], f[1], f[2], m.group(1))][f[3]] = m.group(2)
per = collections.defaultdict(lambda: [0, 0])
for k, v in sorted(ids.items()):
    ok = v.get("A") is not None and v.get("A") == v.get("B")
    per[k[0]][0 if ok else 1] += 1
    if not ok:
        bad.append("DIFF %s %s" % (" ".join(k), v))
for t, (o, d) in sorted(per.items()):
    print("%-26s equal %3d  diff %d" % (t, o, d))
for t, sides in sorted(refused.items()):
    print("%-26s refused at load by %s" % (t, " and ".join(sorted(sides))))
    if sides != {"A", "B"}:
        bad.append("%s refused by one side only" % t)
print("TOTAL cells %d, differences %d" % (sum(o + d for o, d in per.values()), len(bad)))
for b in bad[:20]:
    print(b)
sys.exit(1 if bad else 0)
EOF
r=$?
rm -rf "$W"
exit $r
