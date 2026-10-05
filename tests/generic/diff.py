#!/usr/bin/env python3
"""tests/generic/diff.py: the generic pre-tokenizer (src/core/gen.c) against hf tokenizers 0.23.2 (oniguruma).

    uv run --with tokenizers==0.23.2 python tests/generic/diff.py --driver build/toks_driver [--n 20000] [--seed 1]
        [--chain-from <tokenizer.json>] <tokenizer.json> ...

For each tokenizer file: random texts over an alphabet of the edge characters of every census pattern the generic
engine runs (docs/algorithms/generic.md §3: P16 P19 P20 P21 P22 P23 P25 P26 P28 P29 P30), each compared by its
pre-tokenizer piece ends (toks_pieces, TOKS_ADDED_NONE) and its ids (toks_encode, TOKS_ADDED_NONE | no
post-processing) with hf's. hf's ends: each piece's offsets (chars of the text, NFC'd first under NFC: the generic
engine sees the normalized gap) as bytes; a cut inside a char (falcon's Digits after ByteLevel) ends at the char's end
there, as ByteLevel's alignments give it. --chain-from runs each file with another file's pre_tokenizer (P23
Spark-X2.5's chain, whose own file does not load: its vocab lacks byte-level alphabet chars). Prints one line per
file and the first few mismatches. --enum K runs every string of 0..K chars over --alphabet instead (the template
proofs of kernels.md §3 A9 / A10: the census chains compiled onto the cl100k template).
"""
import argparse
import codecs
import itertools
import json
import os
import random
import sys
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "parity"))
from run import Driver  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

POOL = (
    # ascii: letters (contraction letters, both cases), digits, every punctuation / symbol, the whitespace kinds
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") +
    list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") +
    [" ", " ", " ", "  ", "   ", "\t", "\n", "\n", "\r", "\r\n", "\n\n", "\r\n\r\n", "\x0b", "\x0c", " \n", "\n "] +
    # contractions and (?i) folds
    ["'s", "'S", "'\u017f", "'ll", "'LL", "'lL", "'re", "'RE", "'ve", "'d", "'m", "'t", "'T", "'k", "'\u212a"] +
    # letters: latin-1 / extended (in and out of P19's class), greek, cyrillic, armenian, georgian, cherokee,
    # glagolitic, fullwidth, title / modifier letters, marks, other scripts
    ["\u00e9", "\u00df", "\u00b5", "\u00c0", "\u00d7", "\u00f7", "\u01bb", "\u0294", "\u02b0", "\u01c5", "\u2126",
     "\u03b1", "\u0436", "\u0531", "\u10a0", "\u13a0", "\u2c00", "\uff21", "\uff41", "\u0301", "\u0903", "\u05d0",
     "\u0628", "\u0915", "\u0e01", "\u1100", "\U00010400", "\U0001e900"] +
    # numbers beyond ascii: arabic-indic, superscripts, roman, fractions, math digits, rust-only numerics
    ["\u0663", "\u00b2", "\u00b3", "\u216b", "\u00bd", "\U0001d7ce", "\U00011de0", "\U00016ff4", "\u2460"] +
    # cjk in and out of the patterns' literal ranges, kana, hangul, U+0800 (P19's range start), cjk punctuation
    ["\u4e00", "\u9fa5", "\u9fa6", "\u4e2d", "\u3041", "\u309f", "\u30a0", "\u30ff", "\u30a2", "\uac00", "\ud7a3",
     "\ud7fb", "\u0800", "\u3002", "\u3001", "\uff0c", "\u300c", "\u3000", "\u00a0", "\u2028", "\u0085"] +
    # bloom's set and other punctuation / symbols: ellipsis, danda, arabic full stop / comma, euro, emoji
    ["\u2026", "\u0964", "\u06d4", "\u060c", "\u20ac", "\u00a9", "\U0001f600", "\u2122", "\u2018", "\u201c"] +
    # runs
    ["1234", "12345678", "999", "hello", "world", "Hello", "WORLD", "   \n", "\n\n\n", "  1", "\t\t", "--", "...",
     "\u4e2d\u6587", "\u3053\u3093\u306b\u3061\u306f", "\uac00\ub098", "caf\u00e9"]
)

def hf_ends(tok, text, nfc):
    """the piece ends hf's pre-tokenizer gives text (no added tokens), as byte offsets"""
    s = unicodedata.normalize("NFC", text) if nfc else text
    return [len(s[:end].encode("utf-8")) for _piece, (_start, end) in tok.pre_tokenizer.pre_tokenize_str(s)]


def texts(rng, n):
    out = []
    for _ in range(n):
        k = rng.randint(1, 24)
        out.append("".join(rng.choice(POOL) for _ in range(k)))
    return out


def every(k, alphabet, shard):
    """every string of 0..k chars over alphabet (the template proofs' enumeration), shard "i/n" of them"""
    i, n = (int(x) for x in shard.split("/"))
    for m in range(k + 1):
        for j, s in enumerate(itertools.product(alphabet, repeat=m)):
            if j % n == i:
                yield "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--enum", type=int, help="every string of 0..K chars of --alphabet instead of random texts")
    ap.add_argument("--alphabet", default="a1. \\t\\r\\n", help="python escapes (codecs unicode_escape)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--chain-from")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.chain_from:                                      # the files with that chain, written beside the scratch
        chain = json.load(open(a.chain_from, encoding="utf-8"))["pre_tokenizer"]
        tmp = tempfile.mkdtemp()
        for i, f in enumerate(a.files):
            j = json.load(open(f, encoding="utf-8"))
            j["pre_tokenizer"] = chain
            a.files[i] = os.path.join(tmp, os.path.basename(f) + "+" + os.path.basename(a.chain_from))
            json.dump(j, open(a.files[i], "w", encoding="utf-8"), ensure_ascii=False)
    bad_total = 0
    for path in a.files:
        j = json.load(open(path, encoding="utf-8"))
        nfc = json.dumps(j.get("normalizer") or {}).find('"NFC"') >= 0
        tok = Tokenizer.from_file(path)
        drv = Driver([a.driver])
        drv.load(path)
        rng = random.Random(a.seed)
        n_bad, n_cases, shown = 0, 0, 0
        alphabet = codecs.decode(a.alphabet, "unicode_escape")
        for t in (every(a.enum, alphabet, a.shard) if a.enum is not None else texts(rng, a.n)):
            tb = t.encode("utf-8")
            want_p = hf_ends(tok, t, nfc)
            want_i = tok_none(path).encode(t, add_special_tokens=False).ids   # hf without added-token extraction
            st, got_p = drv.pieces(tb, 2)                 # TOKS_ADDED_NONE
            st2, got_i = drv.encode(tb, 2 | 4)            # NONE, no post-processing
            n_cases += 2
            bad = (st != 0 or got_p != want_p) + (st2 != 0 or got_i != want_i)
            if bad:
                n_bad += bad
                if shown < 5:
                    shown += 1
                    print(f"  MISMATCH {os.path.basename(path)} {t!r}\n    hf pieces {want_p}\n    toks      {got_p}"
                          f"\n    hf ids {want_i[:20]}\n    toks   {(got_i or [])[:20]}")
        drv.close()
        bad_total += n_bad
        what = f"every string of <= {a.enum} of {alphabet!r}, shard {a.shard}" if a.enum is not None else f"{a.n} texts"
        print(f"{os.path.basename(path)}: {n_cases} cases ({what} x pieces + ids), {n_bad} mismatches", flush=True)
    return 1 if bad_total else 0


_NONE = {}


def tok_none(path):
    """the file with its added tokens removed (hf's NONE view: no extraction), loaded once"""
    if path not in _NONE:
        j = json.load(open(path, encoding="utf-8"))
        j["added_tokens"] = []
        j["post_processor"] = None
        _NONE[path] = Tokenizer.from_str(json.dumps(j))
    return _NONE[path]


if __name__ == "__main__":
    sys.exit(main())
