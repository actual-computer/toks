#!/bin/sh
# tests/data/breadth/diff_real.sh [texts]: the real census files the breadth work makes loadable (docs/breadth.md
# §6), each through diff.sh real with the corpora in build/text (Gutenberg books in several scripts + cpython
# sources; tests/data/breadth/fetch_text.sh fetches them). Files come from the census cache on the host.
# The qwen 2 / 2.5 VL AWQ files are probed with their NFC normalizer removed on both sides (NFC: tests/norm).
set -u
n=${1:-300000}
c=$HOME/.cache/toks/census/coverage/files
corp=$(ls build/text/*)
run() {   # run <name> <sha256> [extra]
    echo "== $1"
    sh tests/data/breadth/diff.sh real "$c/sha256_$2" --texts "$n" --seed 11 --corpus $corp $3 2>&1 | tail -16
}
run granite-4.1-3b e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253 ""
run clap-htsat-unfused 77ef92283d67f0d97e1454909a964afcbfa2019f0fb9f18f8e88d5c25c3ba729 ""
run tiny-random-OPT 45b1ea0c2d1bff740df60b340e8c7e553290e7006c2de3de15178efd75d7a914 ""
run Olmo-3-7B-Think 73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca ""
run phi-4 9f38d05d9d25756bb2f181ab5a0cebcd59e638df10336fc7ed1010f7296d0298 ""
run Olmo-3-7B-Instruct 7738a25c46a6043a0e2e605138851baa450269ce9d800af6ca9bf6192e45510c ""
run Florence-2-base 847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e ""
run Qwen2.5-VL-7B-AWQ-nfc-off 5eee858c5123a4279c3e1f7b81247343f356ac767940b2692a928ad929543214 --strip-normalizer
run Qwen2-VL-7B-AWQ-nfc-off bfb053cacf2735b3a5bebbc40e4a853183efec30da0bc8f6304aef14ec0083bc --strip-normalizer
