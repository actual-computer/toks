#!/bin/sh
# tests/wordpiece/diff_all.sh: tests/wordpiece/run.sh (the c twins against hf 0.23.2) over every pinned WordPiece
# tokenizer, P pins at a time (each pin is two processes: gen.py | check). A lab host only; prefix with taskset on
# shared hosts (maintainer doctrine).
#
#   tests/wordpiece/diff_all.sh N_GENERATED SEED P [TEXT]      e.g. 600000 1 4 build/text/flores.txt
#
# Prints the summary line of each pin and `uptime` before and after; exit 0 iff no pin has a mismatch.
set -u
cd "$(dirname "$0")/../.."
n=$1 seed=$2 par=$3 text=${4:-}
root=${TOKS_TOKENIZER_CACHE:-$HOME/.cache/toks/tokenizers}
pins="minilm-l6 bert-uncased mpnet arctic-l bge-small-zh text2vec-zh paraphrase-mpnet distiluse-ml gte-large multiqa-mpnet labse ko-sroberta bert-base-uncased"
mkdir -p build/diff
uptime
sh tests/wordpiece/run.sh "$root/wp-minilm-l6" 10 1 >/dev/null 2>&1    # build once, before the parallel runs
for p in $pins; do echo "$p"; done | xargs -P "$par" -I{} sh -c \
  "sh tests/wordpiece/run.sh '$root/wp-{}' $n $seed $text > build/diff/{}.log 2>&1; echo \"{} exit \$?\" >> build/diff/{}.log"
rc=0
for p in $pins; do
  echo "== $p: $(grep '^cases' build/diff/$p.log | tail -1) [$(tail -1 build/diff/$p.log)]"
  grep -q "^$p exit 0" build/diff/$p.log || rc=1
done
uptime
exit $rc
