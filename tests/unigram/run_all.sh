#!/bin/sh
# tests/unigram/run_all.sh: the unigram python model against hf tokenizers 0.23.2 at full size (lab host only).
#
#   tests/unigram/run_all.sh <out_dir> <flores_dir> [workers] [seed] [plan]
#
# plan: "main" (default) = random 1M + exhaustive + real 1M on every pin but albert / xlnet;
#       "scalars:<pin>[,<pin>...]" = every unicode scalar in gen.CONTEXTS (14.5M cases per pin).
# Writes <out_dir>/summary.jsonl (one line per tokenizer x kind) and <out_dir>/mismatches.jsonl.
set -eu
out=$1; flores=$2; workers=${3:-8}; seed=${4:-1}; plan=${5:-main}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
run() {
  uv run -q --with tokenizers==0.23.2 python "$here/run_diff.py" --workers "$workers" --seed "$seed" \
    --out "$out/mismatches.jsonl" "$@" 2>&1 | tee -a "$out/log.txt" | grep '^SUMMARY' | sed 's/^SUMMARY //' >> "$out/summary.jsonl" || true
}
pins="uni_pmminilm uni_bgem3 uni_me5large uni_me5small uni_bgererank uni_arctic2 uni_mxbaixs uni_ruri3 uni_llmjp4 uni_llmjp3 uni_t5base uni_flant5"
case "$plan" in
  main)
    for p in $pins; do run --tok "$p" --kind random --n 1000000; done
    for p in $pins; do run --tok "$p" --kind exhaustive --n 0; done
    for p in $pins; do run --tok "$p" --kind real --n 1000000 --text "$flores/dev" "$flores/devtest"; done
    ;;
  scalars:*)
    for p in $(echo "${plan#scalars:}" | tr ',' ' '); do run --tok "$p" --kind scalars --n 0; done
    ;;
  *) echo "unknown plan $plan" >&2; exit 2 ;;
esac
echo "done: $(wc -l < "$out/summary.jsonl") summaries, $(cat "$out/mismatches.jsonl" 2>/dev/null | wc -l) mismatches"
