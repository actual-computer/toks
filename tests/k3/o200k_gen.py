#!/usr/bin/env python3
"""o200k differential cases for the c twin (tests/k3/o200k_check.c), hf tokenizers 0.23.2 as the oracle.

  uv run --with tokenizers==0.23.2 python tests/k3/o200k_gen.py stream [options] > cases.bin
      --short N --long N --exh L --vocab N --real DIR --seed S --variants o200k,nemo
      every case once per variant: generators of tests/model/o200k_model.py, pieces from
      Split(Regex(pattern), 'isolated') (the binding's character offsets converted to bytes, asserted)
  uv run --with tokenizers==0.23.2 --with tiktoken==0.14.0 python tests/k3/o200k_gen.py golden > tests/k3/o200k_golden.h
      the hand cases of tests/c/test_k3_o200k.c with hf's piece ends, as c data
  uv run --with tokenizers==0.23.2 python tests/k3/o200k_gen.py e2e TOKENIZER [options] > records.bin
      texts from the same sources with hf's ids for the whole text (Tokenizer.encode, no special tokens,
      encode_special_tokens on), for tests/k3/o200k_e2e.c (TOKENIZER: a file with the o200k pattern)
  uv run --with tokenizers==0.23.2 python tests/k3/o200k_gen.py classes DUMP
      DUMP = `o200k_check --classes DUMP` (the class byte of every code point, built by src/core/classes.c
      with class_flags 0): compared with the onig probes (o200k_model.build_classes) on every scalar

Record format (little-endian): u8 variant (0 o200k, 1 nemo, 2 kimi), u32 n_bytes, text, u32 n_pieces,
n_pieces x (u32 start, u32 end).
"""
import argparse
import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import o200k_model as M  # noqa: E402

MAXREC = 1 << 23          # o200k_check.c's text and piece buffers

# the hand cases of tests/c/test_k3_o200k.c: one or more per rule and per note of docs/templates/o200k.md
GOLDEN = [
    # empty and one atom of every kind
    "", "a", "A", "\u01c5", "\u02b0", "\u4e2d", "\u0301", "5", " ", "\t", "\r", "\n", "!", "/", "'", "\x00",
    "\u3000", "\U0001F600", "\u017f",
    # O1 case runs: L1 (upper* lower+, the backtrack), L2 (upper+)
    "hello", "Hello", "HELLO", "HTMLParser", "ABCdEF", "camelCase", "a\u01c5", "A\u01c5b", "\u02b0A",
    "\u4e2dAB!", "A\u4e2dB", "A\u4e2d", "\u4e2dA", "\u4e2d\u6587!", "A\u4e2dCd", "Ab\u4e2dCd", "A\u0301!",
    "\u0301A!", "\u0301a", "\u0301\u0301", "a\u0301", "\u00e9t\u00e9", "\u00c9T\u00c9", "\u0416\u0436",
    "\U0001D400\U0001D41A", "\u3042\u30fc\u30a2", "AbCdEf", "ABCD\u02b0", "\u02b0\u02b0A", "Aa\u02b0B",
    # O1 contraction suffix K
    "don't", "DON'T", "I'M", "it's", "it'sx", "it'\u017f", "they're", "THEY'RE", "we've", "you'll",
    "she'd", "they'r", "'llama", "'s", "1's", "x's", "X'S", "\u017f's", "\u0301's", " \u0301's",
    "\u4e2dA's", "A\u4e2d's", "it'S'S", "it''s", "o'clock", "rock'n'roll", "it\u2019s", "x'\u212a",
    "A'll", "ab'LL'", "x'Re", "x'vE", "x'm", "x'D", "x'T",
    # O1 prefix: any P or WS atom, never CR / LF / L / N
    " word", "\tword", "\u3000word", "\u00a0word", "!word", "/word", "a/b", "\nword", "\r\nword",
    "1word", "  word", " \u0301", "!\u0301", "!!\u0301", " \u0301x", "\t\u0301A!", "'S", "\U0001F600a",
    # O2 digits
    "1", "12", "123", "1234", "1234567", "abc123", "123abc", "\u0660\u0661\u0662\u0663", "\u00b2\u00b3",
    "1 2", " 123", "\t12", "1'2",
    # O3 punctuation and the [\r\n/]* tail
    "!!!", "!\n", "!\r\n", "!\n/", "!\n/!", "!\n//\n/x", " /\n", "a/", "/", "//", "///\n\n", " !",
    " !!\n", "!/\n/", ".\r\r\n\n/x", "https://a.b/c\n/d", "*/\n/*", " //\n", "!\n /",
    # O4-O6 whitespace
    "x  ", "  a", "\t!", "  ", "   ", " \n", "\n ", " \n \n", "\n\nx", "  \n x", "\u00a0\n", "x \n ",
    "x\r\n", "\u3000\u3000x", "\n\r\n", " \t\n\t ", "x\u2028y",
    # soups
    "Hello, world! It's 2024.\nDON'T PANIC/42\r\n",
    "\u4e2d\u6587Text\u0301 MixedCASE'S x/y/z\n/ \u0660\u0661\u0662\u0663 ",
]


# kimi's notes (docs/templates/o200k.md §6) plus Han atoms of every base class; also run for o200k and nemo
KIMI_GOLDEN = [
    "a\u4e2d", "\u4e2d\u6587a", " \u4e2d\u6587", "A\u3005", "1\u3007", "\u3007" "1", "!\u2e80", "\u2e80!",
    "!\n/", "!\n/x", "\u4e2d\u0301", "x\U00016ff0y", "a\u3001b", "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "\ud55c\uad6d\uc5b4", "\u4e2dA's", "A\u4e2d's",
    "\u3007\u3007\u3007\u3007", "12\u3007" "3", "\u4e2d\n\u6587", "\u4e2d \u6587", " \u3007", "!\u4e2d",
    "'\u4e2d", "\u4e2d's", "\u3005\u3005a", "\u2f00a", "\U00016fe2!", "a\U00016ff0", "\U00020000\U00020000x",
    "\u4e2d\u3000\u6587", "\u30fc\u4e2d", "\u4e2d\u30fc", "\u302a\u4e2d", "\u4e2d\u302a",
]


def c_str(b):
    out = []
    for x in b:
        out.append("\\x%02X" % x)
    return '"' + "".join(out) + '"'


def golden():
    M.init_kimi()
    cases, ends = [], []
    for v in ("o200k", "nemo", "kimi"):
        strings = [s for s in GOLDEN + KIMI_GOLDEN
                   if M.VARIANTS[v]["engine"] == "hf" or len(s.encode("utf-8")) <= M.KIMI_MAXB]
        for s, want in zip(strings, M.oracle_batch(v, strings)):
            raw = s.encode("utf-8")
            assert want == M.k3_scan(s, v), (v, s)        # the model agrees (the doc is proven elsewhere)
            cases.append((M.VARIANTS[v]["vi"], raw, len(ends), len(want)))
            ends += [b for _a, b in want]
    print("/* GENERATED by tests/k3/o200k_gen.py golden -- piece ends in bytes from hf tokenizers 0.23.2's")
    print(" * Split(Regex(pattern), 'isolated') (variants 0 o200k, 1 nemo) and tiktoken 0.14.0's own split (2 kimi,")
    print(" * strings of <= 64 bytes): the hand cases of tests/c/test_k3_o200k.c. DO NOT EDIT. */")
    print("static const uint32_t GOLDEN_ENDS[] = {")
    for i in range(0, len(ends), 16):
        print("    " + ", ".join(str(e) for e in ends[i:i + 16]) + ",")
    print("};")
    print("static const golden_case GOLDEN[] = {")
    for vi, raw, first, n in cases:
        print("    { %d, %s, %d, %d, %d }," % (vi, c_str(raw), len(raw), first, n))
    print("};")


def emit(out, vi, variant, s):
    raw = s.encode("utf-8")
    if len(raw) > MAXREC:
        raise SystemExit("o200k_gen.py: a %d-byte segment exceeds o200k_check's %d" % (len(raw), MAXREC))
    want = M.hf_pieces(variant, s)
    out.write(struct.pack("<BI", vi, len(raw)))
    out.write(raw)
    out.write(struct.pack("<I", len(want)))
    out.write(b"".join(struct.pack("<II", a, b) for a, b in want))
    return 1


def _tasks(args):
    """the case sources of args, the same list in every shard; shard k of K takes every K-th task."""
    tasks = []
    for st in range(0, args.short, 2000):
        tasks.append(("short", st, min(st + 2000, args.short), args.seed + st))
    for st in range(0, args.long, 200):
        tasks.append(("long", st, min(st + 200, args.long), args.seed + 10**9 + st))
    for length in range(0, args.exh + 1):
        n = len(M.REP) ** length
        for st in range(0, n, 20000):
            tasks.append(("exh", st, min(st + 20000, n), length))
    for st in range(0, args.vocab, 2000):
        tasks.append(("vocab", st, min(st + 2000, args.vocab), args.seed + 2 * 10**9 + st))
    if args.real:
        for dp, _dn, fns in sorted(os.walk(args.real)):
            for fn in sorted(fns):
                tasks.append(("real", os.path.join(dp, fn), 0, args.seed))
    k, kk = (int(x) for x in args.shard.split("/"))
    return [t for ti, t in enumerate(tasks) if ti % kk == k - 1]


def _texts(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
    import run_o200k_fuzz as R
    for kind, a, b, seed in _tasks(args):
        yield from R.cases((None, kind, a, b, seed))


def _kimi_tasks(args):
    """kimi's sources (strings of <= KIMI_MAXB bytes, the bound of its tiktoken oracle), sharded."""
    tasks = []
    for st in range(0, args.kshort, 2000):
        tasks.append(("kshort", st, min(st + 2000, args.kshort), args.seed + st))
    for length in range(0, args.kexh + 1):
        n = len(M.REP_KIMI) ** length
        for st in range(0, n, 20000):
            tasks.append(("kexh", st, min(st + 20000, n), length))
    if args.real and args.kreal:
        for dp, _dn, fns in sorted(os.walk(args.real)):
            for fn in sorted(fns):
                tasks.append(("kreal", os.path.join(dp, fn), args.kreal, args.seed))
    k, kk = (int(x) for x in args.shard.split("/"))
    return [t for ti, t in enumerate(tasks) if ti % kk == k - 1]


def stream(args):
    out = sys.stdout.buffer
    total = 0
    variants = args.variants.split(",")
    hfv = [v for v in variants if M.VARIANTS[v]["engine"] == "hf"]
    if hfv:
        M.init()
        for s in _texts(args):
            for v in hfv:
                total += emit(out, M.VARIANTS[v]["vi"], v, s)
    if "kimi" in variants:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
        import run_o200k_fuzz as R
        batch = []

        def flush():
            for s, want in zip(batch, M.tiktoken_pieces(batch)):
                raw = s.encode("utf-8")
                out.write(struct.pack("<BI", 2, len(raw)) + raw + struct.pack("<I", len(want)))
                out.write(b"".join(struct.pack("<II", a, b) for a, b in want))
            batch.clear()

        for kind, a, b, seed in _kimi_tasks(args):
            for s in R.cases((None, kind, a, b, seed)):
                batch.append(s)
                total += 1
                if len(batch) == 500:
                    flush()
        flush()
    sys.stderr.write("o200k_gen.py: shard %s wrote %d records\n" % (args.shard, total))


def e2e(args):
    """records for tests/k3/o200k_e2e.c: u32 len, text, u32 n_ids, hf's ids for the whole text."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.dump)
    tok.encode_special_tokens = True          # added tokens stay text: pre-tokenizer + bpe only
    out = sys.stdout.buffer
    total = 0
    batch = []

    def flush():
        for s, enc in zip(batch, tok.encode_batch(batch, add_special_tokens=False)):
            raw = s.encode("utf-8")
            if len(raw) > MAXREC:
                raise SystemExit("o200k_gen.py: a %d-byte text exceeds o200k_e2e's %d" % (len(raw), MAXREC))
            out.write(struct.pack("<I", len(raw)) + raw + struct.pack("<I", len(enc.ids)))
            out.write(struct.pack("<%dI" % len(enc.ids), *enc.ids))
        batch.clear()

    for s in _texts(args):
        batch.append(s)
        total += 1
        if len(batch) == 1000:
            flush()
    flush()
    sys.stderr.write("o200k_gen.py: e2e shard %s wrote %d records\n" % (args.shard, total))


def classes(path):
    M.init()
    with open(path, "rb") as f:
        dump = f.read()
    if len(dump) != 0x110000:
        raise SystemExit("classes: dump has %d bytes, want 0x110000" % len(dump))
    bad = [cp for cp in M.scalars() if dump[cp] != M.CLS[cp]]
    for cp in bad[:20]:
        print("U+%04X: c tables 0x%02X, onig probes 0x%02X" % (cp, dump[cp], M.CLS[cp]))
    print("classes: %d scalar values compared (base class, UPPER, LOWER, MARK, FOLD_S), %d differ"
          % (len(M.scalars()), len(bad)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stream", "golden", "classes", "e2e"])
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--short", type=int, default=20000)
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--exh", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--real", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", default="o200k,nemo")
    ap.add_argument("--shard", default="1/1", help="k/K: emit every K-th task starting at the k-th")
    ap.add_argument("--kshort", type=int, default=0, help="kimi: gen_kimi strings")
    ap.add_argument("--kexh", type=int, default=-1, help="kimi: every string of 0..L atoms over REP_KIMI")
    ap.add_argument("--kreal", type=int, default=0, help="kimi: slices per real-text file")
    args = ap.parse_args()
    if args.mode == "golden":
        golden()
        return 0
    if args.mode == "classes":
        return classes(args.dump)
    random.seed(args.seed)
    if args.mode == "e2e":
        e2e(args)
    else:
        stream(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
