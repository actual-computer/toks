#!/usr/bin/env python3
"""tests/data/breadth/semantics.py: docs/breadth.md §4 and §5 written as executable models and checked
against hf tokenizers 0.23.2 on the real census files (a lab host; the files are the census cache's).

    uv run --with tokenizers==0.23.2 python tests/data/breadth/semantics.py drop FILE --n N --seed S
    uv run --with tokenizers==0.23.2 python tests/data/breadth/semantics.py digits FILE --n N --seed S

drop (§4, byte-level vocabularies missing byte chars): hf's ids (no added tokens, no post-processor)
  == for each piece of hf's own pre-tokenizer: the model run on the piece with every char the vocab lacks
  removed (unk_token null, byte_fallback off: hf merge_word skips the char, its neighbours become adjacent
  and may merge). With unk_token set the model is checked too: the piece splits at each run of missing
  chars, the run becomes one unk (fuse_unk) or one per char, and no merge crosses an unk.
  Texts: random mixes of the file's missing-but-reachable chars (valid utf-8 only: hf takes str) with
  letters, digits, spaces, punctuation and real lines.
digits (§5, the SmolLM chain Sequence[Digits(individual_digits), ByteLevel(use_regex)]): hf's pieces ==
  every char rust's char::is_numeric accepts is a piece of its own, and the gpt-2 regex (hf's own Split, so
  oniguruma) runs on each maximal run between them. The numeric set is probed here through hf's Digits.
"""
import argparse
import json
import os
import random
import sys

from tokenizers import Regex, Tokenizer, pre_tokenizers

GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}
U2B = {u: b for b, u in B2U.items()}

BASE = list("abcdefghijklmnopqrstuvwxyzABCXYZéßñ汉字かカ한 ") + ["  ", "\t", "\n", "\r\n", "1", "23", "456", "'s",
       "don't", "hello", " world", ".", ",", "!", "?", "-", "(", ")", "\"", "😀", "\u00a0", "\u3000", "٣", "²"]


def no_added(path):
    obj = json.load(open(path, encoding="utf-8"))
    obj["added_tokens"] = []
    obj["post_processor"] = None
    obj["normalizer"] = None            # the census files here carry none, or NFC (tested on its own: tests/norm)
    obj["truncation"] = None
    obj["padding"] = None
    return obj


def merges_of(m, vocab):
    """hf's merge map: (left id, right id) -> (rank, merged id); a repeated pair keeps its LAST rank."""
    out = {}
    for r, mg in enumerate(m["merges"]):
        x, y = mg.split(" ") if isinstance(mg, str) else mg
        out[(vocab[x], vocab[y])] = (r, vocab[x + y])
    return out


def bpe_symbols(syms, merges):
    """hf word.rs merge_all without dropout == repeatedly merge the adjacent pair of lowest rank, leftmost
    on ties (as the parity suites verified): a quadratic transcription, fine for short pieces."""
    syms = list(syms)
    while True:
        best = None
        for i in range(len(syms) - 1):
            mg = merges.get((syms[i], syms[i + 1]))
            if mg is not None and (best is None or mg[0] < best[0]):
                best = (mg[0], i, mg[1])
        if best is None:
            return syms
        _, i, new = best
        syms[i:i + 2] = [new]


def merge_word_model(piece, vocab, merges, unk_id, fuse, bf):
    """models/bpe/model.rs merge_word, char by char: a vocab char is a symbol (flushing a pending unk
    first); a missing char takes byte_fallback's <0xXX> tokens when all exist (added at once, a pending
    unk NOT flushed first: hf's order), else becomes unk (pending, fused with the previous one when
    fuse_unk), else (no unk_token) vanishes; then merge_all over the whole symbol list."""
    syms, unk = [], None
    for c in piece:
        if c in vocab:
            if unk is not None:
                syms.append(unk)
                unk = None
            syms.append(vocab[c])
            continue
        if bf:
            bt = [vocab.get(f"<0x{x:02X}>") for x in c.encode("utf-8")]
            if all(t is not None for t in bt):
                syms += bt
                continue
        if unk_id is not None:
            if unk is not None and not fuse:
                syms.append(unk)
            unk = unk_id
    if unk is not None:
        syms.append(unk)
    return bpe_symbols(syms, merges)


def run_drop(a):
    obj = no_added(a.file)
    tok = Tokenizer.from_str(json.dumps(obj))
    m = obj["model"]
    vocab = m["vocab"]
    missing = [b for b in range(256) if B2U[b] not in vocab]
    unk = m.get("unk_token")
    unk_id = vocab.get(unk) if unk is not None else None
    fuse = bool(m.get("fuse_unk"))
    bf = bool(m.get("byte_fallback"))
    merges = merges_of(m, vocab)
    # reachable from valid utf-8: an ascii byte alone, or a lead / continuation inside some code point
    lead_cp = {0xF1: 0x40000, 0xF2: 0x80000, 0xF3: 0xC0000, 0xF4: 0x100000}      # 4-byte leads
    probes = []
    for b in missing:
        if b < 0x80:
            probes.append(chr(b))
        elif 0x80 <= b <= 0xBF:
            probes.append(chr(0x80 | (b & 0x3F)))                         # U+0080..00BF: c2 <b>
        elif b in lead_cp:
            probes.append(chr(lead_cp[b] + 0x41))
        elif 0xC2 <= b <= 0xDF:
            probes.append(chr(((b & 0x1F) << 6) | 0x01))
        elif 0xE1 <= b <= 0xEF and b != 0xED:
            probes.append(chr(((b & 0x0F) << 12) | 0x0101))
    probes = [p for p in probes if p]
    print(f"missing bytes: {' '.join(f'{b:02x}' for b in missing)}; reachable probes {len(probes)}; "
          f"unk {unk!r} fuse_unk {fuse} byte_fallback {bf} ignore_merges {m.get('ignore_merges')}")
    if not probes:
        print("no missing byte is reachable from valid utf-8: hf never drops anything for this file")
        return 0
    if m.get("ignore_merges"):
        raise SystemExit("ignore_merges: the model would look the whole piece up first (not written)")
    pt = tok.pre_tokenizer
    rng = random.Random(a.seed)
    pool = BASE + probes * 4
    bad = 0
    for i in range(a.n):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 18)))
        want = tok.encode(text, add_special_tokens=False).ids
        got = []
        pieces = pt.pre_tokenize_str(text) if pt is not None else [(alpha_of(text), None)]
        for piece, _ in pieces:
            if unk_id is None and not bf:                     # the plain rule: drop, then bpe
                kept = "".join(c for c in piece if c in vocab)
                got += bpe_symbols([vocab[c] for c in kept], merges) if kept else []
            else:
                got += merge_word_model(piece, vocab, merges, unk_id, fuse, bf)
        if got != want:
            bad += 1
            if bad <= 5:
                print("DIFF", text.encode("unicode_escape"), "hf", want, "model", got)
    print(f"drop model vs hf: {a.n} texts, {bad} differing")
    return 1 if bad else 0


def alpha_of(text):
    return "".join(B2U[b] for b in text.encode("utf-8"))


def rust_numeric():
    dg = pre_tokenizers.Digits(individual_digits=True)
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    num = set()
    for i in range(0, len(cps), 0x4000):
        part = cps[i:i + 0x4000]
        s = "".join(chr(cp) + "x" for cp in part)
        for _, (x, y) in dg.pre_tokenize_str(s):
            if y - x == 1 and x % 2 == 0:
                num.add(part[x // 2])
    return num


def run_digits(a):
    obj = no_added(a.file)
    chain = obj["pre_tokenizer"]
    print("pre_tokenizer:", json.dumps(chain))
    tok = Tokenizer.from_str(json.dumps(obj))
    num = rust_numeric()
    if a.onig_n:                                   # the negative control: oniguruma's \p{N} instead
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from probe_hf import onig_class
        num = onig_class(r"\p{N}")
    extra = sorted(c for c in num if c >= 0x11000)
    gpt2 = pre_tokenizers.Split(Regex(GPT2), "isolated")
    rng = random.Random(a.seed)
    pool = BASE + ["0", "7", "٣", "²", "½", "Ⅻ", chr(0x11DE3), chr(0x16FF5), " 1", "  2", "a1b", "x 9 y", "\t3"]
    bad = 0
    for i in range(a.n):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 18)))
        want = [o for _, o in tok.pre_tokenizer.pre_tokenize_str(text)]
        got, run_start = [], 0
        for k, ch in enumerate(text + "\0"):
            if k == len(text) or ord(ch) in num:
                if k > run_start:
                    got += [(run_start + x, run_start + y) for _, (x, y) in gpt2.pre_tokenize_str(text[run_start:k])]
                if k < len(text):
                    got.append((k, k + 1))
                run_start = k + 1
        if got != want:
            bad += 1
            if bad <= 5:
                print("DIFF", text.encode("unicode_escape"), "hf", want, "model", got)
    print(f"digits model vs hf: {a.n} texts, {bad} differing (split set {len(num)} code points"
          f"{', oniguruma' if a.onig_n else ', rust is_numeric'})")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("drop", "digits"))
    ap.add_argument("file")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--onig-n", action="store_true", help="digits: split at oniguruma's \\p{N} (must differ)")
    a = ap.parse_args()
    import tokenizers
    print(f"tokenizers {tokenizers.__version__}; {a.file}")
    return run_drop(a) if a.what == "drop" else run_digits(a)


if __name__ == "__main__":
    sys.exit(main())
