#!/bin/sh
# tests/wordpiece/parity.sh: toks.h end to end against hf 0.23.2 on the pinned WordPiece tokenizers, through the
# parity harness (tests/parity: gen_cases.py's full case set per tokenizer, run.py over tests/driver/toks_driver
# in N shards). A lab host only (maintainer doctrine); the pins come from tests/wordpiece/fetch.py.
#
#   tests/wordpiece/parity.sh OUTDIR [N_SHARDS [PIN ...]]       (default: 8 shards, every pin)
#
# Writes OUTDIR/<pin>/report.json (shards.sh) and prints one line per pin; exit 0 iff no pin has a diff.
set -u
cd "$(dirname "$0")/../.."
out=$1
n=${2:-8}
shift 2 2>/dev/null || shift $#
pins=${*:-"minilm-l6 bert-uncased mpnet arctic-l bge-small-zh text2vec-zh paraphrase-mpnet distiluse-ml gte-large multiqa-mpnet labse ko-sroberta bert-base-uncased"}
root=${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}
mkdir -p "$out"
make -s -j8 >/dev/null || exit 2
lib=$(ls build/*/libtoks.a | head -1)
clang -std=c17 -O2 -Iinclude -o build/toks_driver tests/driver/toks_driver.c "$lib" || exit 2
rc=0
for p in $pins; do
  tok=$root/wp-$p
  [ -f "$tok" ] || { echo "$p: MISSING $tok"; rc=1; continue; }
  uv run -q --with tokenizers==0.23.2 python tests/parity/gen_cases.py --tokenizer "$tok" \
      --out "$out/$p.jsonl" > "$out/$p.gen.log" 2>&1 || { echo "$p: gen_cases failed"; rc=1; continue; }
  sh tests/parity/shards.sh "$tok" "$out/$p.jsonl" ./build/toks_driver "$out/$p" "$n" > "$out/$p.log" 2>&1
  r=$?
  uv run -q python -c "
import json,sys
r=json.load(open('$out/$p/report.json'))
print('$p:', r['n_cases'], 'cases,', r['n_diffs'], 'diffs, unsupported', r['unsupported'], ', spec E_ID', r['spec_e_id'],
      ', missing shards', r['missing_shards'], ', %.0fs' % r['elapsed_s'])
print('   load', r['uptime_start'].split('load average:')[-1].strip(), '->', r['uptime_end'].split('load average:')[-1].strip())"
  [ $r -eq 0 ] || rc=1
done
exit $rc
