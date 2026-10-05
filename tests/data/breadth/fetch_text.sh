#!/bin/sh
# tests/data/breadth/fetch_text.sh: the corpora tests/data/breadth/diff_real.sh reads, into build/text (fetched
# once; build/ is kept by tools/remote.sh). Gutenberg books in several scripts (english, spanish, french,
# chinese, german, russian) and two cpython sources. The sha256 of each file behind docs/breadth.md §6's receipts:
#   pg1342 3f6bb9d6..   pg1661 922e2a12..   pg2000 534f41d5..   pg23863 9d5b723b..   pg24264 ff152699..
#   pg2600 2d5bb2ad..   pg7849 31351cb9..   code.py (typing.py) b73770ed..   code2.c (listobject.c) fe0a926a..
set -eu
mkdir -p build/text
get() {   # get <name> <url>
    [ -s "build/text/$1" ] || curl -fsSL "$2" -o "build/text/$1"
}
for n in 1342 1661 2000 2600 7849 23863 24264; do
    get "pg$n.txt" "https://www.gutenberg.org/cache/epub/$n/pg$n.txt"
done
get code.py https://raw.githubusercontent.com/python/cpython/main/Lib/typing.py
get code2.c https://raw.githubusercontent.com/python/cpython/main/Objects/listobject.c
sha256sum build/text/* 2>/dev/null || shasum -a 256 build/text/*
