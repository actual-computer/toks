#!/usr/bin/env python3
"""dsv3 differential cases for the c twin (tests/k3/dsv3_check.c), hf tokenizers 0.23.2 as the oracle.

  uv run --python 3.12 --with tokenizers==0.23.2 python tests/k3/dsv3_gen.py stream [options] > cases.bin
      --short N --long N --exh L --exh-small L --vocab N --real DIR --bad N --seed S --shard k/K
      cases from the generators of tests/model/dsv3_model.py with the pinned file's own pieces (hf_pieces);
      --bad N adds N ill-formed byte strings with the python model's pieces (hf cannot take them)
  uv run --python 3.12 --with tokenizers==0.23.2 python tests/k3/dsv3_gen.py golden > tests/k3/dsv3_golden.h
      the hand cases of tests/c/test_k3_dsv3.c with hf's piece ends (and, for ill-formed utf-8, the model's), as c data
  uv run --python 3.12 --with tokenizers==0.23.2 python tests/k3/dsv3_gen.py classes DUMP
      DUMP = `dsv3_check --classes DUMP` (the class byte of every code point, src/core/classes.c with
      TOKS_CLASSES_DSV3): compared with the onig probes (dsv3_model.build_classes) on every scalar

Record format (little-endian): u8 oracle (0: hf, 1: the python model on ill-formed bytes), u32 n_bytes, text,
u32 n_pieces, n_pieces x (u32 start, u32 end).
"""
import argparse
import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import dsv3_model as M  # noqa: E402

MAXREC = 1 << 23          # dsv3_check.c's text and piece buffers

# the hand cases of tests/c/test_k3_dsv3.c: one or more per rule and per note of docs/templates/dsv3.md
GOLDEN = [
    # empty and one atom of every kind
    "", "a", "Z", "\u00e9", "\u0301", "\u4e2d", "\u30ab", "\u3099", "\u30fb", "\u309b", "\u3040", "1", "\u0663",
    "\u00bd", " ", "\t", "\u3000", "\u00a0", "\r", "\n", "!", "'", "\u00ab", "\U0001F600", "\x00", "\u00ad",
    "\ue000", "\U0001F468",
    # D0 digits (split 1)
    "1234", "1234567", "12a34", "1\u00bd\u2167x", "x'0", "\u0661\u0662\u0663\u0664", "a1", "1a", " 1", "  1",
    "1 ", "\u4e2d1", "1\u4e2d", "\u3007\u3006\u3005", "!1", "1!", "1\n", "\n1", "\x001", "1\x00",
    # regions (split 2 inside split 1's gaps)
    "a  b", "a  1", "a  \u4e2d", " \u30ab", "\u3000\u30ab", "\x00\u4e2d", "!\u4e2d", "!\u30fb", "\u30fb\n",
    "!\n", "\u9fa5\u9fa6", "\u4dff\u4e00", "\u3005\u4e00", "\u4e00\u3005", "\u4e2d\u6587",
    "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8", " \u65e5\u672c\u8a9e", " \u65e5\u672c\u8a9e abc", "\u6f22\u5b57test",
    "x\u5b57y", "\u4e00\u0301", "\u0301\u4e00", "a\u3000b", "\u3000\u3000a", "\u4e2d \u6587", "\u4e2d\n\u6587",
    "\u4e2d  ", "  \u4e2d",
    # split 3 inside a CJK run
    "\u30ab\u30fb\u30ab", "\u30fc\u30fb\u30fc", "\u309b\u30ab", "\u30a0\u30a0\u30ab", "\u3040\u30ab",
    "\u30ab\u3040", "\u3040\u3040", "\u3097\u3098\u3099", "\u3041\u3099\u3099", "\u309f\u30ff", "\u30ab\u30fc",
    "\u3042\u309b\u304c", "\u3093\u30fb\u30ab\u30ca", "\u3040 ", " \u3040", "\u3040\n", "\u3040a", "\u30fb\u30fb",
    "\u309b\u309c", "\u3040\u3040\u30ab", "\u3099\u30ab", "\u30ab\u3099", "\u3099", "\u4e00\u3040", "\u3040\u4e00",
    # D1 ascii word
    "!hello", ".com", "'ll", "'LL", "!!hello", " !a", " .com", "''a", "a.b.c", "don't", "DON'T", "'s\u00e9",
    "'s\u0301", "!ab\u00e9", "!a\u0301", "!a\u4e2d", "!a1", "!a b", "x.y_z", "@user #tag",
    "https://example.com/path?q=1", "$var", "_init", "`cmd`", "~/x", "a-b", "\\n", "'", "!'a", "!\u00e9a",
    # the letter prefix: WS or X only
    "\u00aba", "!\u00e9", "\uff04a", "\tword", "\u00a0word", "\u2028a", "\u0085a", "\x00a", "\x7fa", "\ue000a",
    "\U000e0001a", "a\x1fb", "\nword", "\r\nword", " word", "  word", "\x0b\x0bx", "\x0c a", "\u3000word",
    "\u200bx",
    # marks are letters
    "\u0301x", "e\u0301\u0301!", "\u0301!", "\u2764\ufe0fa", "\u2764\ufe0f", " \u0301", "!\u0301", "\u0301\u0301",
    "a\u0301", "\u093e\u094d", "\u00ad\u0301", "\t\u0301",
    # D7 gaps
    "\x00\x00a", "\x00\x00 a", "\u00ad\u00ad\u0301x", " \x00", "  \x00", "\U0001F468\u200d\U0001F469", "\x00\x00",
    "\x00 ", "\x00\n", "\x00!", "\x00\x00\x00a", "\ue000\ue000x", "x\u200by", "\x00\u0301", "\x00\x00\u3040\u30ab",
    # D3 punctuation (P = \p{P} | \p{S})
    "\U0001F600\U0001F600", " \U0001F600", "\uff04\uffe5", "!!\n\n", "!\r\n", "!\n\r\n", " !!", "...", "\u00ab\u00bb",
    "!\u00ab", "\u00ab \u00bb", "!\n/", "\U0001F3FB", " !\n\n", "!\n \n", " \u00ab",
    # D4-D6 whitespace
    "x \n\n ", "\n  1", "\t\t\n", " ", "  ", "   ", " \n", "\n ", " \n \n", "\n\nx", "  \n x", "x  ", "  a",
    "x \n ", "x\r\n", "\u2028\u2029", "a \u3000 b", "\r\r", "\n\r", "  !", "  \u00ab",
    # soups
    "Hello, world! It's 2024.\nDON'T PANIC/42\r\n",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3001\u30ab\u30bf\u30ab\u30ca\u30fb\u3072\u3089\u304c\u306a\u3002"
    "123\u5186\uff01\n",
    "def f(x):\n    return x**2  # \u6ce8\u91ca\n",
    "\u4e2d\u6587Text\u0301 MixedCASE'S x/y/z\n/ \u0660\u0661\u0662\u0663 ",
]

# ill-formed utf-8 (hf cannot take it): the python model's pieces (docs/templates/dsv3.md §3, invalid atoms)
GOLDEN_BAD = [
    b"\xffabc", b"!\xff\n", b" \xff", b"a\xffb", b"\xe3\x82" + "\u30ab".encode(), b"\xff", b"\x80", b"\xc3",
    b"\xe4\xb8", b"\xe4\xb8\xad\xe4\xb8", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xc0\xaf", b"\xe0\x80\x80",
    b"\xf0\x9f\x98", b"a\xcc", b"\xff\xcc\x81", b"!\xffa", b"\xff!a", b" \xff\n", b"\x00\xff", b"\xff\x00a",
    b"1\xff2", "\u3000".encode() + b"\xff", "\u3040".encode() + b"\xff", b"\xff" + "\u4e2d".encode(),
    "\u30fb".encode() + b"\xff", b"\xe3\x83", b"\xe3\x83\xbb\xe3\x83", b"\x00\x00\xff",
]


def c_str(b):
    return '"' + "".join("\\x%02X" % x for x in b) + '"'


def golden():
    M.init()
    cases, ends = [], []
    for s in GOLDEN:
        want = M.hf_pieces(s)
        got = M.k3_scan(s)
        if want != got:
            raise SystemExit("dsv3_gen.py golden: the model disagrees with hf on %s: hf %r model %r"
                             % (s.encode("unicode_escape").decode(), want, got))
        raw = s.encode("utf-8")
        cases.append((1, raw, len(ends), len(want)))
        ends += [b for _a, b in want]
    for raw in GOLDEN_BAD:
        try:
            raw.decode("utf-8")
            raise SystemExit("dsv3_gen.py golden: %r is well-formed" % raw)
        except UnicodeDecodeError:
            pass
        want = M.k3_scan_bytes(raw)
        cases.append((0, raw, len(ends), len(want)))
        ends += [b for _a, b in want]
    print("/* GENERATED by tests/k3/dsv3_gen.py golden -- piece ends in bytes: hf tokenizers 0.23.2's own pre-tokenizer of")
    print(" * deepseek-ai/DeepSeek-V3's tokenizer.json (hf = 1), or for ill-formed utf-8, which hf cannot take, the python")
    print(" * model of docs/templates/dsv3.md (hf = 0). The hand cases of tests/c/test_k3_dsv3.c. DO NOT EDIT. */")
    print("static const uint32_t GOLDEN_ENDS[] = {")
    for i in range(0, len(ends), 16):
        print("    " + ", ".join(str(e) for e in ends[i:i + 16]) + ",")
    print("};")
    print("static const golden_case GOLDEN[] = {")
    for hf, raw, first, n in cases:
        print("    { %d, %s, %d, %d, %d }," % (hf, c_str(raw), len(raw), first, n))
    print("};")


def _tasks(args):
    """the case sources of args, the same list in every shard; shard k of K takes every K-th task."""
    tasks = []
    for st in range(0, args.short, 2000):
        tasks.append(("short", st, min(st + 2000, args.short), args.seed + st))
    for st in range(0, args.long, 200):
        tasks.append(("long", st, min(st + 200, args.long), args.seed + 10**9 + st))
    for kind, alpha, lmax in (("exh", M.REP, args.exh), ("exh_small", M.REP_SMALL, args.exh_small)):
        for length in range(0, lmax + 1):
            n = len(alpha) ** length
            for st in range(0, n, 20000):
                tasks.append((kind, st, min(st + 20000, n), length))
    for st in range(0, args.vocab, 2000):
        tasks.append(("vocab", st, min(st + 2000, args.vocab), args.seed + 2 * 10**9 + st))
    if args.real:
        for dp, _dn, fns in sorted(os.walk(args.real)):
            for fn in sorted(fns):
                tasks.append(("real", os.path.join(dp, fn), 0, args.seed))
    for st in range(0, args.bad, 2000):
        tasks.append(("bad", st, min(st + 2000, args.bad), args.seed + 3 * 10**9 + st))
    k, kk = (int(x) for x in args.shard.split("/"))
    return [t for ti, t in enumerate(tasks) if ti % kk == k - 1]


BAD_BYTES = [b"\xff", b"\xfe", b"\x80", b"\xbf", b"\xc0", b"\xc1", b"\xc3", b"\xe0\x80", b"\xe3\x81", b"\xe4\xb8",
             b"\xed\xa0\x80", b"\xf0\x9f", b"\xf4\x90\x80\x80", b"\xf5", b"\xe0\x9f\xbf", b"\xf0\x80\x80\x80"]


def bad_case(rng):
    """a byte string mixing generated text with ill-formed sequences at random positions."""
    s = M.gen_case(rng).encode("utf-8")
    out = bytearray()
    i = 0
    while i <= len(s):
        if rng.random() < 0.15:
            out += rng.choice(BAD_BYTES)
        if i < len(s):
            out.append(s[i])
        i += 1
    return bytes(out)


def stream(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
    import run_dsv3_fuzz as R
    M.init()
    out = sys.stdout.buffer
    total = 0
    for task in _tasks(args):
        if task[0] == "bad":
            rng = random.Random(task[3])
            for _ in range(task[2] - task[1]):
                raw = bad_case(rng)
                want = M.k3_scan_bytes(raw)
                out.write(struct.pack("<BI", 1, len(raw)) + raw + struct.pack("<I", len(want)))
                out.write(b"".join(struct.pack("<II", a, b) for a, b in want))
                total += 1
            continue
        for s in R.cases(task):
            raw = s.encode("utf-8")
            if len(raw) > MAXREC:
                raise SystemExit("dsv3_gen.py: a %d-byte segment exceeds dsv3_check's %d" % (len(raw), MAXREC))
            want = M.hf_pieces(s)
            out.write(struct.pack("<BI", 0, len(raw)) + raw + struct.pack("<I", len(want)))
            out.write(b"".join(struct.pack("<II", a, b) for a, b in want))
            total += 1
    sys.stderr.write("dsv3_gen.py: shard %s wrote %d records\n" % (args.shard, total))


def classes(path):
    M.init()
    with open(path, "rb") as f:
        dump = f.read()
    if len(dump) != 0x110000:
        raise SystemExit("classes: dump has %d bytes, want 0x110000" % len(dump))
    bad = [cp for cp in M.scalars() if dump[cp] != M.CLS[cp]]
    for cp in bad[:20]:
        print("U+%04X: c tables 0x%02X, onig probes 0x%02X" % (cp, dump[cp], M.CLS[cp]))
    print("classes: %d scalar values compared (kind: base | CJK, nothing else), %d differ" % (len(M.scalars()), len(bad)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stream", "golden", "classes"])
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--short", type=int, default=20000)
    ap.add_argument("--long", type=int, default=2000)
    ap.add_argument("--exh", type=int, default=3)
    ap.add_argument("--exh-small", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--real", default=None)
    ap.add_argument("--bad", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", default="1/1", help="k/K: emit every K-th task starting at the k-th")
    args = ap.parse_args()
    if args.mode == "golden":
        golden()
        return 0
    if args.mode == "classes":
        return classes(args.dump)
    stream(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
