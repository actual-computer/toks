#!/usr/bin/env python3
"""tests/bpe/gen.py: hf-oracle cases for the bpe differential; tests/bpe/check.c reads them on stdin.

    uv run --with tokenizers==0.23.2 python tests/bpe/gen.py TOKENIZER_JSON N [SEED [TEXT ...]] > cases.bin

Each case is one piece: its raw bytes and the ids of Tokenizer.from_file(TOKENIZER_JSON).model.tokenize(
the piece's byte-level-alphabet form) -- hf 0.23.2's own BPE::tokenize (ignore_merges and its cache
included; no normalizer, pre-tokenizer or added tokens). The file must match a sha256 pin in
tools/corpora/fetch_tokenizers.py.

The stream starts with the model derived here, independently of toks's config / compile code (check.c
compares it with what toks_config_parse + toks_compile make of the same file): "TOKSBPE1", u32 n_vocab,
n_ids, n_merges, ignore_merges; per id u32 len + its raw bytes (a model token's byte-level string
decoded, empty when it is not an alphabet image; an added-only id's content); per merge u32 left,
right, merged id in rank order, as hf builds its MergeMap: vocab[a], vocab[b], vocab[a + b].

Pieces (deterministic for the arguments):
  vocab  every model-vocabulary token with a raw form, every added token's content
  text   the distinct pieces of each TEXT file under the tokenizer's own pre-tokenizer
  gen    N generated: vocabulary tokens joined / cut / mutated, random bytes, random utf-8 over a
         dozen scripts, ascii-heavy strings, digit / space / punctuation runs, pre-tokenizer-shaped
         words, and 1 in 400 long (1-64 KiB, every 40th of those up to 1 MiB)
Record: u32 len, len bytes, u32 n, n x u32 ids, little-endian; the stream ends with u32 0xFFFFFFFF.
Counts and sha256s go to stderr.
"""
import ast
import hashlib
import json
import multiprocessing as mp
import os
import random
import struct
import sys

from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
CHUNK = 4000


def byte_alphabet():
    """gpt-2's bytes_to_unicode: byte -> char"""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2C = byte_alphabet()
C2B = {c: b for b, c in B2C.items()}


def pins():
    src = open(os.path.join(ROOT, "tools", "corpora", "fetch_tokenizers.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TOKENIZERS":
            return {v["sha256"]: k for k, v in ast.literal_eval(node.value).items()}
    raise SystemExit("no TOKENIZERS pins in tools/corpora/fetch_tokenizers.py")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def raw(s):
    """alphabet string -> raw bytes, or None when s is not an alphabet image"""
    try:
        return bytes(C2B[c] for c in s)
    except KeyError:
        return None


TK = None
VOCAB = None


def init(path):
    global TK, VOCAB
    TK = Tokenizer.from_file(path)
    VOCAB = sorted(r for r in (raw(s) for s in TK.get_vocab(with_added_tokens=False)) if r)


def model_header(path):
    tj = json.load(open(path, encoding="utf-8"))
    m = tj["model"]
    vocab = m["vocab"]
    n_vocab = max(vocab.values()) + 1
    assert len(vocab) == n_vocab, "model.vocab ids are not dense"
    toks = [b""] * n_vocab
    for s, i in vocab.items():
        toks[i] = raw(s) or b""
    added = {a["id"]: a["content"].encode("utf-8") for a in tj.get("added_tokens", [])}
    for i in sorted(added):
        if i >= n_vocab:
            toks += [b""] * (i + 1 - len(toks))
            toks[i] = added[i]
    pairs = [x.split(" ") if isinstance(x, str) else x for x in m["merges"]]
    trip = [(vocab[a], vocab[b], vocab[a + b]) for a, b in pairs]
    h = b"TOKSBPE1" + struct.pack("<4I", n_vocab, len(toks), len(trip), int(bool(m.get("ignore_merges"))))
    h += b"".join(struct.pack("<I", len(t)) + t for t in toks)
    return h + b"".join(struct.pack("<3I", *x) for x in trip)


def record(piece):
    ids = [t.id for t in TK.model.tokenize("".join(B2C[b] for b in piece))]
    return struct.pack("<I", len(piece)) + piece + struct.pack(f"<I{len(ids)}I", len(ids), *ids)


def records(pieces):
    return b"".join(record(p) for p in pieces)


SCRIPTS = [(0x41, 0x7A), (0xC0, 0x24F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x5FF), (0x600, 0x6FF),
           (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x3040, 0x30FF), (0x4E00, 0x9FFF),
           (0xAC00, 0xD7A3), (0x1F300, 0x1FAFF), (0x300, 0x36F), (0x2000, 0x206F), (0xFF00, 0xFFEF)]
PUNCT = b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
B64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def utf8_text(rng, n):
    out = []
    lo, hi = rng.choice(SCRIPTS)
    for _ in range(n):
        if rng.random() < 0.2:
            lo, hi = rng.choice(SCRIPTS)
        cp = rng.randint(lo, hi)
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0x20
        out.append(" " if rng.random() < 0.12 else chr(cp))
    return "".join(out).encode("utf-8")


def gen_piece(rng):
    r = rng.random()
    if r < 0.0025:                                   # long: base64, digits, a repeated unit
        n = rng.randint(1 << 10, 1 << 16) if rng.random() < 0.975 else rng.randint(1 << 16, 1 << 20)
        k = rng.randrange(3)
        if k == 0:
            return bytes(rng.choice(B64) for _ in range(n))
        if k == 1:
            return bytes(rng.choice(b"0123456789") for _ in range(n))
        unit = rng.choice(VOCAB)
        return (unit * (n // len(unit) + 1))[:n]
    if r < 0.25:                                     # 2-4 vocabulary tokens joined
        return b"".join(rng.choice(VOCAB) for _ in range(rng.randint(2, 4)))
    if r < 0.40:                                     # a vocabulary token cut or mutated
        t = bytearray(rng.choice(VOCAB))
        i, op = rng.randrange(len(t) + 1), rng.randrange(4)
        if op == 0:
            t.insert(i, rng.randrange(256))
        elif op == 1 and len(t) > 1:
            del t[min(i, len(t) - 1)]
        elif op == 2:
            t[min(i, len(t) - 1)] = rng.randrange(256)
        else:
            t = t[: max(1, i)] if rng.random() < 0.5 else t[min(i, len(t) - 1):]
        return bytes(t)
    if r < 0.55:
        return utf8_text(rng, rng.randint(1, 40))
    if r < 0.65:
        return bytes(rng.randrange(256) for _ in range(rng.randint(1, 64)))
    if r < 0.80:
        return bytes(rng.randint(0x20, 0x7E) if rng.random() < 0.9 else rng.randrange(256)
                     for _ in range(rng.randint(1, 80)))
    if r < 0.90:                                     # runs
        alpha = rng.choice([b"0123456789", b" ", b"\n", b" \t\n\r", PUNCT, bytes([rng.randrange(256)])])
        return bytes(rng.choice(alpha) for _ in range(rng.randint(1, 200)))
    word = rng.choice(VOCAB) if rng.random() < 0.5 else utf8_text(rng, rng.randint(1, 12))
    return rng.choice([b" ", b"", b"'", b"\n", b"  "]) + word.strip() if word.strip() else word


def gen_chunk(args):
    seed, n = args
    rng = random.Random(seed)
    return records([gen_piece(rng) or b"a" for _ in range(n)])


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, n_gen = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    texts = sys.argv[4:]
    sha = sha256(path)
    name = pins().get(sha)
    if name is None:
        raise SystemExit(f"{path}: sha256 {sha} is not pinned in tools/corpora/fetch_tokenizers.py")
    init(path)
    vocab = sorted(VOCAB)
    added = [a.content.encode("utf-8") for a in TK.get_added_tokens_decoder().values() if a.content]
    text_pieces = set()
    for tp in texts:
        s = open(tp, "rb").read().decode("utf-8", "replace")
        n0 = len(text_pieces)
        for p, _ in TK.pre_tokenizer.pre_tokenize_str(s):
            r = raw(p)
            if r:
                text_pieces.add(r)
        print(f"text {tp}: sha256 {sha256(tp)}, {len(text_pieces) - n0} new distinct pieces", file=sys.stderr)
    fixed = vocab + added + sorted(text_pieces)
    tasks = [("fixed", fixed[i:i + CHUNK]) for i in range(0, len(fixed), CHUNK)]
    tasks += [("gen", (seed * 1000003 + i, min(CHUNK, n_gen - i))) for i in range(0, n_gen, CHUNK)]
    print(f"tokenizer {name} sha256 {sha}: {len(vocab)} vocab + {len(added)} added + {len(text_pieces)} text "
          f"+ {n_gen} generated pieces, seed {seed}", file=sys.stderr)
    out = sys.stdout.buffer
    out.write(model_header(path))
    with mp.Pool(int(os.environ.get("WORKERS", os.cpu_count() or 4)), initializer=init, initargs=(path,)) as pool:
        work = [pool.apply_async(records if k == "fixed" else gen_chunk, (a,)) for k, a in tasks]
        for w in work:
            out.write(w.get())
    out.write(struct.pack("<I", 0xFFFFFFFF))
    out.flush()


if __name__ == "__main__":
    main()
