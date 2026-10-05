# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""tests/fuzz/seeds.py: seed corpora for the libFuzzer harnesses (docs/fuzz.md).

  uv run tests/fuzz/seeds.py texts --out build/fuzz/seeds build/text
      slices of the bench corpora (tools/bench/corpus.sha256: en, code, ml, cjk; cut at utf-8 boundaries, 16 B -
      4 KiB) and hand-written NFC / sentencepiece / markup texts, in the text harnesses' format (gen.h: an 8-byte
      header picking the tokenizer, flags and sel) for encode, pieces and par, every pinned tokenizer; random id
      sequences in the id harnesses' format (ids.h) for decode and stream.
  uv run tests/fuzz/seeds.py shrink --out build/fuzz/seeds FILE_OR_DIR...
      shrinks tokenizer files (the pinned files, tests/data/compile/*.json, the census files) to small seeds for
      load and load_json: every top-level section kept as it is, the model vocabulary cut to its first ids plus the
      byte-level alphabet and renumbered densely, the merges whose three tokens survive, the first added tokens,
      template ids remapped by content (all four algorithms: a map vocabulary with or without merges, a list one).
  uv run tests/fuzz/seeds.py sweep --out build/fuzz/seeds FILE_OR_DIR...
      the arena sweep (docs/fuzz.md §5.7) for load and load_json: per kind of tokenizer file (model, normalizer,
      pre_tokenizer and decoder types; the first file of each), shrunk as above, 64 consecutive
      vocabulary counts, and an unused top-level key holding an array (of 0, [], "" or {}) at 64 consecutive lengths
      up to the harnesses' 65536-byte limit, then an 8 x 8 grid of both.

Stdlib only. Deterministic (seeded). Ported from the first fuzz harnesses (commit 82cdeda).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys

N_PIN = 16                                   # fuzz.h FZ_NPIN: the pinned tokenizers, selected by header byte 0

EXTRA = [
    "e\u0301 caf\u00e9 \u212b \u2126 \u0344", "\u0958" * 40, "\U0001D160" * 30 + " x", "\u1100\u1161\u11a8 \uac00\u11a8",
    "a\u0301\u0316 o\u0316\u0301", "\u0f73\u0f75\u0f81 \ufb2c \u2adc", "\u1e9b\u0323 \u00c5\u0301 \u0301x",
    "  hello \u2581 world  ", "<0x41><0xff> <unk> <bos>", "\U000f0000\u0378 \U0001faa8", "\u2581\u2581a\u2581b",
    " leading", "trailing ", "\t\n\r\n x", "Hello, world! It's 2026.", "<|im_start|>user\nhi<|im_end|>\n",
    "[CLS] the ##ing [SEP]", "HTMLParser ABC. AB\u4e2dC", "1234567890 3.14159 \u0663\u0664", "don't DON'T y'all'd've",
]


def bytes_to_unicode() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + \
        list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


ALPHABET = set(bytes_to_unicode().values())


def put(outdir: str, data: bytes) -> None:
    os.makedirs(outdir, exist_ok=True)
    name = hashlib.sha1(data).hexdigest()
    with open(os.path.join(outdir, name), "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------------------------- texts

def utf8_cut(b: bytes, a: int, e: int) -> bytes:
    while a < len(b) and (b[a] & 0xC0) == 0x80:
        a += 1
    while e < len(b) and (b[e] & 0xC0) == 0x80:
        e += 1
    return b[a:e]


def texts(args) -> None:
    rng = random.Random(1234)
    srcs = []
    for p in args.paths:
        p = os.path.expanduser(p)
        files = [os.path.join(p, f) for f in sorted(os.listdir(p))] if os.path.isdir(p) else [p]
        for f in files:
            try:
                srcs.append(open(f, "rb").read())
            except OSError:
                pass
    pieces = [t.encode("utf-8") for t in EXTRA]
    for b in srcs:
        for _ in range(12):
            if len(b) < 32:
                break
            ln = rng.choice((16, 64, 256, 1024, 4096))
            a = rng.randrange(0, max(1, len(b) - ln))
            pieces.append(utf8_cut(b, a, a + ln))
    n = 0
    for t in range(N_PIN):
        for x in rng.sample(pieces, min(len(pieces), 40)):
            hdr = bytes([t, rng.randrange(16)]) + bytes(rng.randrange(256) for _ in range(6))
            for h in ("encode", "pieces", "par"):
                put(os.path.join(args.out, h), hdr + x)
                n += 1
        for _ in range(12):
            k = rng.choice((1, 2, 5, 17, 64, 300))
            ids = b"".join(struct.pack("<I", rng.getrandbits(32) | (1 << 31 if rng.random() < 0.95 else 0)) for _ in range(k))
            hdr = bytes([t, rng.randrange(2), rng.randrange(256), rng.randrange(256)])
            for h in ("decode", "stream"):
                put(os.path.join(args.out, h), hdr + ids)
                n += 1
    print(f"seeds: {n} text and id seeds from {len(srcs)} files", file=sys.stderr)


# ---------------------------------------------------------------------------------------------- shrink

def split_merge(e):
    if isinstance(e, str):
        p = e.split(" ")
        return (p[0], p[1]) if len(p) == 2 else None
    if isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) for x in e):
        return (e[0], e[1])
    return None


def shrink(j: dict, keep_ids: int, keep_merges: int, keep_added: int) -> dict:
    m = j.get("model")
    added = j.get("added_tokens") if isinstance(j.get("added_tokens"), list) else None
    if isinstance(m, dict) and isinstance(m.get("vocab"), dict):
        vocab = m["vocab"]
        byid = sorted(((v, k) for k, v in vocab.items() if isinstance(v, int)), key=lambda x: x[0])
        keep = [k for v, k in byid if v < keep_ids or k in ALPHABET]
        merges = m.get("merges") if isinstance(m.get("merges"), list) else []
        kset = set(keep)
        out = []
        extra = []
        for e in merges:
            p = split_merge(e)
            if p is None:
                continue
            a, b = p
            if a in kset and b in kset and a + b in kset:
                out.append(e)
            elif a in kset and b in kset and len(extra) < keep_merges // 3 and (a + b) in vocab:
                extra.append(a + b)                        # grow a few chains past the id cut
                kset.add(a + b)
                keep.append(a + b)
                out.append(e)
            if len(out) >= keep_merges:
                break
        order = {k: i for i, (v, k) in enumerate(byid)}
        keep.sort(key=lambda k: order.get(k, 1 << 30))
        m["vocab"] = {k: i for i, k in enumerate(keep)}
        m["merges"] = out
    elif isinstance(m, dict) and isinstance(m.get("vocab"), list):   # unigram
        m["vocab"] = m["vocab"][:keep_ids]
    if added is not None:
        j["added_tokens"] = added[:keep_added]
        # template ids follow the contents (hf: a vocabulary string's id, else counted up from the vocab)
        if isinstance(m, dict) and isinstance(m.get("vocab"), dict):
            nv = len(m["vocab"])
            ids = {}
            nxt = nv
            for a in j["added_tokens"]:
                c = a.get("content") if isinstance(a, dict) else None
                if not isinstance(c, str) or c == "" or c in ids:
                    continue
                if c in m["vocab"]:
                    ids[c] = m["vocab"][c]
                else:
                    ids[c] = nxt
                    nxt += 1
            for a in j["added_tokens"]:
                if isinstance(a, dict) and a.get("content") in ids:
                    a["id"] = ids[a["content"]]
            pp = j.get("post_processor")
            procs = pp.get("processors") if isinstance(pp, dict) and pp.get("type") == "Sequence" else [pp]
            for p in procs or []:
                st = p.get("special_tokens") if isinstance(p, dict) else None
                if not isinstance(st, dict):
                    continue
                for name, ent in st.items():
                    if isinstance(ent, dict) and isinstance(ent.get("tokens"), list):
                        ent["ids"] = [ids.get(t, nv - 1) for t in ent["tokens"]]
    return j


def shrink_cmd(args) -> None:
    files = []
    for p in args.paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p))]
        else:
            files.append(p)
    n_ok = n_skip = 0
    for f in files:
        try:
            raw = open(f, "rb").read()
        except OSError:
            continue
        docs = []
        if f.endswith(".txt"):                              # refuse.txt: "<code> <why>" then a json line
            for line in raw.splitlines():
                if line.startswith(b"{"):
                    docs.append(line)
        else:
            docs.append(raw)
        for d in docs:
            try:
                j = json.loads(d)
            except (ValueError, UnicodeDecodeError):
                n_skip += 1
                continue
            if not isinstance(j, dict):
                n_skip += 1
                continue
            for keep_ids, keep_merges in ((300, 200), (600, 500)):
                s = json.dumps(shrink(json.loads(d), keep_ids, keep_merges, 24), ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")
                if len(s) <= 65536:
                    put(os.path.join(args.out, "load"), s)
                    put(os.path.join(args.out, "load_json"), s)
            n_ok += 1
    print(f"seeds: shrunk {n_ok} tokenizer files, skipped {n_skip} non-json", file=sys.stderr)


# ---------------------------------------------------------------------------------------------- sweep

PAD_ELEMS = ("0", "[]", "\"\"", "{}")          # the unused key's array elements
SWEEP = 64                                     # consecutive steps: every residue of a 64-byte alignment


def cut_vocab(j: dict, drop: int) -> dict | None:
    """the shrunk file with its `drop` highest vocabulary ids removed (renumbered densely, the merges whose strings
    are gone dropped with them, the unk token kept); None when the vocabulary is not a map or a list"""
    m = j.get("model")
    if not isinstance(m, dict):
        return None
    v = m.get("vocab")
    if isinstance(v, list):
        if drop >= len(v):
            return None
        m["vocab"] = v[:len(v) - drop]
        return j
    if not isinstance(v, dict):
        return None
    byid = sorted(((i, k) for k, i in v.items() if isinstance(i, int)), key=lambda x: x[0])
    unk = m.get("unk_token")
    gone = set()
    for i, k in reversed(byid):
        if len(gone) >= drop:
            break
        if k != unk:
            gone.add(k)
    keep = [k for i, k in byid if k not in gone]
    m["vocab"] = {k: n for n, k in enumerate(keep)}
    if isinstance(m.get("merges"), list):
        kset = set(keep)
        out = []
        for e in m["merges"]:
            p = split_merge(e)
            if p is not None and p[0] in kset and p[1] in kset and p[0] + p[1] in kset:
                out.append(e)
        m["merges"] = out
    return j


def with_pad(j: dict, elem: str, k: int) -> bytes:
    s = json.dumps(j, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    pad = ("," + elem) * (k - 1) + elem if k > 0 else ""
    return s[:-1] + b',"x_unused":[' + pad.encode("utf-8") + b"]}"


def sweep_cmd(args) -> None:
    """per tokenizer file (shrunk as `shrink` does): SWEEP vocabulary counts (the highest ids cut one at a time), and
    for each element of PAD_ELEMS an unused top-level key holding an array of it, SWEEP consecutive lengths ending at
    the harness's 65536-byte input limit; then an 8 x 8 grid of both. Load and load_json seeds."""
    files = []
    for p in args.paths:
        p = os.path.expanduser(p)
        files += [os.path.join(p, f) for f in sorted(os.listdir(p))] if os.path.isdir(p) else [p]
    n = 0
    seen = set()
    for f in files:
        try:
            base = shrink(json.loads(open(f, "rb").read()), 300, 200, 24)
        except (OSError, ValueError, UnicodeDecodeError, AttributeError, TypeError):
            continue
        if not isinstance(base, dict) or not base:
            continue
        kind = tuple(json.dumps(x.get("type") if isinstance(x, dict) else None) for x in
                     (base.get("model"), base.get("normalizer"), base.get("pre_tokenizer"), base.get("decoder")))
        if kind in seen:                               # one file per (model, normalizer, pre_tokenizer, decoder) kind
            continue
        seen.add(kind)
        outs = []
        for d in range(SWEEP):
            c = cut_vocab(json.loads(json.dumps(base)), d)
            if c is not None:
                outs.append(with_pad(c, "0", 0))
        b0 = len(json.dumps(base, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        for elem in PAD_ELEMS:
            kmax = (65536 - b0 - 16) // (len(elem) + 1)
            for k in range(max(0, kmax - SWEEP + 1), kmax + 1):
                outs.append(with_pad(base, elem, k))
        for d in range(0, SWEEP, SWEEP // 8):
            c = cut_vocab(json.loads(json.dumps(base)), d)
            if c is None:
                continue
            kmax = (65536 - b0 - 16) // 2
            for k in range(max(0, kmax - SWEEP + 1), kmax + 1, SWEEP // 8):
                outs.append(with_pad(c, "0", k))
        for s in outs:
            if len(s) <= 65536:
                put(os.path.join(args.out, "load"), s)
                put(os.path.join(args.out, "load_json"), s)
                n += 1
    print(f"seeds: {n} sweep seeds (vocabulary counts, unused-key arrays) from {len(seen)} kinds of {len(files)} files",
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("texts")
    a.add_argument("--out", default="build/fuzz/seeds")
    a.add_argument("paths", nargs="+")
    a.set_defaults(fn=texts)
    b = sub.add_parser("shrink")
    b.add_argument("--out", default="build/fuzz/seeds")
    b.add_argument("paths", nargs="+")
    b.set_defaults(fn=shrink_cmd)
    c = sub.add_parser("sweep")
    c.add_argument("--out", default="build/fuzz/seeds")
    c.add_argument("paths", nargs="+")
    c.set_defaults(fn=sweep_cmd)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
