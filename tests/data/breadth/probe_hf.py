#!/usr/bin/env python3
"""tests/data/breadth/probe_hf.py: the character sets and equivalences the breadth code relies on, probed
through hf tokenizers 0.23.2 itself (never through python's unicodedata or a regex re-implementation).

    uv run --with tokenizers==0.23.2 python tests/data/breadth/probe_hf.py <what> [--out FILE]

  word     the regex-crate \\w that hf's single_word uses (added_vocabulary.rs ends_with_word /
           starts_with_word), every scalar, both sides; and the \\s of lstrip / rstrip (LEFTMOST_SPACE_AT_END,
           RIGHTMOST_SPACE_AT_START), every scalar, both sides. --out writes the \\w ranges as a c table.
  numeric  rust char::is_numeric as hf's Digits pre-tokenizer applies it, every scalar, against oniguruma's
           \\p{N} as hf's Split sees it (the P13 digits-gpt2 analysis).
  invert   for every pattern string in src/core/config.c's table: Split(Removed, invert=true) against
           Split(Isolated) over every scalar in four contexts, then random strings (--n, --seed).

Everything is printed; any disagreement with an expectation exits 1.
"""
import argparse
import json
import os
import random
import re
import sys

from tokenizers import Regex, Tokenizer, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

SCALARS = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}


def ranges(cps):
    out = []
    for cp in sorted(cps):
        if out and out[-1][1] + 1 == cp:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def bytelevel_tok(added):
    """a byte-level bpe tokenizer with the full alphabet, no merges, no split, and the given added tokens."""
    j = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                                            "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {B2U[b]: b for b in range(256)}, "merges": []},
    }
    return Tokenizer.from_str(json.dumps(j))


def added(content, **kw):
    t = {"id": 256, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
    t.update(kw)
    return t


BATCH = 4096
T = "\u20ac"      # the probe token: not a vocab string, so its id is 256


def token_spans(tok, s, tid):
    """char spans (start, end) of every occurrence of added token tid in hf's encoding of s (the token is
    U+20AC, not a vocab string, so hf gives it the new id 256 and a dropped match encodes as 3 byte ids)."""
    enc = tok.encode(s, add_special_tokens=False)
    # offsets are char offsets in python's binding
    return [o for i, o in zip(enc.ids, enc.offsets) if i == tid]


def probe_word():
    bad = 0
    # \w, left side: "<c>\u20ac\0" -- the match is kept iff c is not a word char (\0 is not \w: checked)
    tok = bytelevel_tok([added(T, single_word=True)])
    word_l, word_r = set(), set()
    cps = [cp for cp in SCALARS if cp != ord(T)]
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        s = "".join(chr(cp) + T + "\0" for cp in part)
        kept = {a for a, b in token_spans(tok, s, 256)}
        for k, cp in enumerate(part):
            if 3 * k + 1 not in kept:
                word_l.add(cp)
        s = "".join("\0" + T + chr(cp) for cp in part)
        kept = {a for a, b in token_spans(tok, s, 256)}
        for k, cp in enumerate(part):
            if 3 * k + 1 not in kept:
                word_r.add(cp)
    if 0 in word_l or 0 in word_r:
        print("FAIL: U+0000 probed as a word char (the separator must not be one)")
        bad += 1
    if word_l != word_r:
        print(f"FAIL: \\w differs by side: {len(word_l ^ word_r)} code points")
        bad += 1
    wr = ranges(word_l)
    print(f"\\w (single_word): {len(word_l)} code points in {len(wr)} ranges; both sides agree: {word_l == word_r}")
    # \s, lstrip: "x<c>T" -> the token starts at the c when c is \s
    tok = bytelevel_tok([added(T, lstrip=True)])
    ws_l, ws_r = set(), set()
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        s = "".join("x" + chr(cp) + T for cp in part)
        starts = {a for a, b in token_spans(tok, s, 256)}
        for k, cp in enumerate(part):
            if 3 * k + 1 in starts:
                ws_l.add(cp)
    tok = bytelevel_tok([added(T, rstrip=True)])
    for i in range(0, len(cps), BATCH):
        part = cps[i:i + BATCH]
        s = "".join(T + chr(cp) + "x" for cp in part)
        ends = {b for a, b in token_spans(tok, s, 256)}
        for k, cp in enumerate(part):
            if 3 * k + 2 in ends:
                ws_r.add(cp)
    expect = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B)) + \
             [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    print(f"\\s (lstrip): {len(ws_l)} code points; (rstrip): {len(ws_r)}; "
          f"== the 25 of segment.c: {ws_l == set(expect) and ws_r == set(expect)}")
    if ws_l != set(expect) or ws_r != set(expect):
        print("  lstrip:", [hex(c) for c in sorted(ws_l)])
        print("  rstrip:", [hex(c) for c in sorted(ws_r)])
        bad += 1
    return bad, wr


def onig_class(pat):
    sp = pre_tokenizers.Split(Regex(pat), "removed")
    hit = set()
    for i in range(0, len(SCALARS), 0x8000):
        part = SCALARS[i:i + 0x8000]
        s = "".join(map(chr, part))
        keep = bytearray(len(s))
        for _, (a, b) in sp.pre_tokenize_str(s):
            keep[a:b] = b"\1" * (b - a)
        for k, cp in enumerate(part):
            if not keep[k]:
                hit.add(cp)
    return hit


def probe_numeric():
    dg = pre_tokenizers.Digits(individual_digits=True)
    num = set()
    for i in range(0, len(SCALARS), 0x4000):
        part = SCALARS[i:i + 0x4000]
        s = "".join(chr(cp) + "x" for cp in part)       # 'x' is not numeric: a numeric char is a 1-char piece
        for _, (a, b) in dg.pre_tokenize_str(s):
            if b - a == 1 and a % 2 == 0:
                num.add(part[a // 2])
    n_onig = onig_class(r"\p{N}")
    print(f"rust char::is_numeric (Digits): {len(num)} code points; oniguruma \\p{{N}}: {len(n_onig)}; "
          f"equal: {num == n_onig}")
    if num != n_onig:
        only_r = sorted(num - n_onig)
        only_o = sorted(n_onig - num)
        print(f"  only is_numeric: {len(only_r)} {[hex(c) for c in only_r[:40]]}")
        print(f"  only \\p{{N}}:     {len(only_o)} {[hex(c) for c in only_o[:40]]}")
    return 0, num, n_onig


def config_patterns():
    src = open(os.path.join(ROOT, "src/core/config.c"), encoding="utf-8").read()
    body = src[src.index("TOKS_PATTERNS[] = {"):]
    body = body[:body.index("};")]
    out = []
    for m in re.finditer(r'\{\s*((?:"(?:[^"\\]|\\.)*"\s*)+),', body):
        lit = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
        out.append(lit.encode("latin-1").decode("unicode_escape"))
    return out


ALPHA = ("abcXYZ éÉßſİı ÆÇ Ωω Жж αβ ٣ 汉字かカ한 ‿_' \"!?.,;:/\\-+*=<>()[]{}@#$%^&|~` 0123456789 ²½Ⅻ٤ \t\n\r"
         "\u000b\u000c\u0085\u00a0\u1680\u2000\u2028\u2029\u202f\u205f\u3000 \u0301\u0308\u200d\u200c "
         "😀👍🏽\U00010400\U0001d400")


def probe_invert(n, seed):
    bad = 0
    pats = config_patterns()
    print(f"{len(pats)} patterns in src/core/config.c")
    for pat in pats:
        iso = pre_tokenizers.Split(Regex(pat), "isolated", invert=False)
        rem = pre_tokenizers.Split(Regex(pat), "removed", invert=True)
        diff = 0
        for ctx in ("{}", "a{}b", " {}1", "\n{}'"):
            for i in range(0, len(SCALARS), 0x2000):
                part = SCALARS[i:i + 0x2000]
                s = "".join(ctx.format(chr(cp)) for cp in part)
                a = iso.pre_tokenize_str(s)
                b = rem.pre_tokenize_str(s)
                if a != b:
                    diff += 1
        rng = random.Random(seed)
        for _ in range(n):
            s = "".join(rng.choice(ALPHA) for _ in range(rng.randint(0, 40)))
            if iso.pre_tokenize_str(s) != rem.pre_tokenize_str(s):
                diff += 1
                if diff < 5:
                    print("  diff:", s.encode("unicode_escape"))
        print(f"  {'ok  ' if diff == 0 else 'FAIL'} {pat.encode('unicode_escape').decode()[:70]}... "
              f"4 contexts x {len(SCALARS)} scalars + {n} random strings: {diff} differing")
        bad += diff != 0
    return bad


def emit_word_c(wr, ws):
    """src/gen/rx_word.{c,h}: the regex-crate \\w of hf's single_word as ranges (binary searched by segment.c)."""
    import hashlib
    data = b"".join(lo.to_bytes(4, "little") + hi.to_bytes(4, "little") for lo, hi in wr)
    sha = hashlib.sha256(data).hexdigest()
    head = (
        "/* toks: the \\w of hf tokenizers 0.23.2's added-token single_word check (added_vocabulary.rs\n"
        " * ends_with_word / starts_with_word: the rust regex crate's unicode \\w), as probed through hf itself.\n"
        " *\n"
        " * GENERATED FILE -- DO NOT EDIT.  Regenerate on a lab host with:\n"
        " *     uv run --with tokenizers==0.23.2 python tests/data/breadth/probe_hf.py word --emit\n"
        " *\n"
        f" * {sum(hi - lo + 1 for lo, hi in wr)} code points in {len(wr)} ranges, probed on both sides of a match over every\n"
        " * scalar value. lstrip / rstrip's \\s probed the same way: exactly the 25 code points of segment.c.\n"
        f" * data (lo u32 LE || hi u32 LE per range) sha256: {sha}\n"
        " */\n")
    root = ROOT
    with open(os.path.join(root, "src/gen/rx_word.h"), "w") as f:
        f.write(head)
        f.write("#ifndef TOKS_RX_WORD_H\n#define TOKS_RX_WORD_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define TOKS_RX_WORD_N {len(wr)}u\n")
        f.write(f"#define TOKS_RX_WORD_SHA256 \"{sha}\"\n\n")
        f.write("/* sorted, disjoint, non-adjacent [lo, hi] ranges */\n")
        f.write("extern const uint32_t toks_rx_word[TOKS_RX_WORD_N][2];\n\n#endif /* TOKS_RX_WORD_H */\n")
    with open(os.path.join(root, "src/gen/rx_word.c"), "w") as f:
        f.write(head)
        f.write("#include \"rx_word.h\"\n\nconst uint32_t toks_rx_word[TOKS_RX_WORD_N][2] = {\n")
        for i in range(0, len(wr), 4):
            f.write("    " + " ".join(f"{{0x{lo:05X}u, 0x{hi:05X}u}}," for lo, hi in wr[i:i + 4]) + "\n")
        f.write("};\n")
    print(f"wrote src/gen/rx_word.{{c,h}}: {len(wr)} ranges, sha256 {sha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("word", "numeric", "invert"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit", action="store_true", help="word: write src/gen/rx_word.{c,h}")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import tokenizers
    print(f"tokenizers {tokenizers.__version__}")
    if a.what == "word":
        bad, wr = probe_word()
        if a.out:
            with open(a.out, "w") as f:
                json.dump(wr, f)
        if a.emit and not bad:
            emit_word_c(wr, None)
    elif a.what == "numeric":
        bad, num, onig = probe_numeric()
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"is_numeric": ranges(num), "onig_N": ranges(onig)}, f)
    else:
        bad = probe_invert(a.n, a.seed)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
