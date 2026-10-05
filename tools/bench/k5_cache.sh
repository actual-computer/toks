#!/bin/sh
# tools/bench/k5_cache.sh: build tools/bench/k5_cache.c against this tree's library and run it over tokenizers x
# corpora x chunks, the other text being every other bench corpus (tools/bench/tokv1.sh's rule; e2e.sh's lang-x).
# Counts, not timings: any host, no timing lock. env: TOKS_LIST CORPORA CHUNKS PIN (a cpu for taskset, linux)
set -e
T=build/text
K=$HOME/.cache/toks/tokenizers
TOKS_LIST=${TOKS_LIST:-"llama3 gpt2 qwen38"}
CORPORA=${CORPORA:-"en code ml zh"}
CHUNKS=${CHUNKS:-"4096 0"}
mkvar() { printf 'print-%%:\n\t@echo $($*)\n' | make -s -f Makefile -f - "print-$1"; }
RUN=""
if [ -n "${PIN:-}" ] && command -v taskset >/dev/null 2>&1; then RUN="taskset -c $PIN"; fi
mkdir -p $T
for f in $HOME/.cache/toks/bench-text/*; do [ -e "$T/$(basename "$f")" ] || cp "$f" "$T/"; done
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
(cd "$T" && $SHA -c --quiet "$OLDPWD/tools/bench/corpus.sha256") || { echo "CORPUS sha mismatch"; exit 1; }
make -j4 lib >/dev/null
BD=$(mkvar BUILD_DIR)
cc -std=c17 -O2 -Wall -Wextra $(mkvar CPPFLAGS) -o build/k5_cache tools/bench/k5_cache.c "$BD/libtoks.a" -lpthread
files() {
    case $1 in
        en) echo "$T/gut-en-1342.txt $T/gut-en-2701.txt" ;;
        code) echo "$T/code-cpython.py $T/code-toks.c" ;;
        zh) echo "$T/wiki-zh.txt $T/gut-zh-24264.txt" ;;
        ml) echo "$T/wiki-ar.txt $T/wiki-de.txt $T/wiki-el.txt $T/wiki-fr.txt $T/wiki-he.txt $T/wiki-hi.txt $T/wiki-ka.txt $T/wiki-ru.txt $T/wiki-ta.txt $T/wiki-th.txt $T/wiki-uk.txt $T/wiki-vi.txt" ;;
        cjk) echo "$T/wiki-zh.txt $T/wiki-ja.txt $T/wiki-ko.txt $T/gut-zh-24264.txt" ;;
    esac
}
others() {
    o=""
    for c in en code ml cjk; do for f in $(files $c); do
        case " $(files "$1") $o " in *" $f "*) ;; *) o="$o $f" ;; esac
    done; done
    echo $o
}
for tk in $TOKS_LIST; do
    for c in $CORPORA; do
        for ch in $CHUNKS; do
            echo "== $tk $c $ch"
            # shellcheck disable=SC2046
            $RUN ./build/k5_cache "$K/$tk" "$ch" $(files "$c") -- $(others "$c") || echo "K5CACHE FAILED $tk $c $ch"
        done
    done
done
