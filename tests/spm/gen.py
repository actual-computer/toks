#!/usr/bin/env python3
"""tests/spm/gen.py: hf tokenizers 0.23.2 cases for tests/spm/check.c (the c side's differential, end to end).

    uv run --with tokenizers==0.23.2 tests/spm/gen.py <pinned name> <n> <seed> > cases.bin

Records (little-endian), every expectation straight from hf:
    u8 1, u8 at_start, u32 n, text[n], u32 k, u32 ids[k]   a gap: hf with added_tokens [] and no post-processor
                                                            (toks: mode NONE, no pp); at_start 0 = the gap follows
                                                            an added token (toks: TOKS_CONTINUATION)
    u8 2, u32 n, piece[n], u32 k, u32 ids[k]               hf Tokenizer.model.tokenize(piece), pieces of chars
                                                            the text model does not substitute (doc §6)
    u8 3, u8 skip, u32 k, u32 ids[k], u32 n, bytes[n]      hf decode(ids, skip_special_tokens=skip), the file as is
    u8 4, u8 at_start, u32 n, text[n]                      hf fails on this gap (toks must fail too)
    u8 5, u8 how, u32 n, text[n], u32 k, u32 ids[k]        the file as is: how 0 = encode(text) (mode ALL, post-
                                                            processing on), 1 = encode_special_tokens + no pp
                                                            (mode NONSPECIAL)
Texts, pieces and ids come from tests/model/run_spm_fuzz.py's generators.
"""
import json
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests", "model"))
sys.path.insert(0, HERE)
import run_spm_fuzz as R  # noqa: E402
import fetch as F  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PUA = "\ue000"


def substituted(tj):
    """the chars the text model substitutes: toks' char table reads each as its image (doc §6)"""
    out = set()
    n = tj.get("normalizer")
    steps = [n] if n and n.get("type") != "Sequence" else (n or {}).get("normalizers", [])
    for st in steps:
        if st.get("type") == "Replace" and "String" in st.get("pattern", {}):
            out.add(st["pattern"]["String"])
    pre = tj.get("pre_tokenizer")
    pres = [pre] if pre and pre.get("type") != "Sequence" else (pre or {}).get("pretokenizers", [])
    if any(x.get("type") == "Metaspace" for x in pres):
        out.add(" ")
    return out


def hf_ok(f):
    try:
        return f()
    except BaseException as e:  # noqa: BLE001  (pyo3 panics are BaseException)
        if isinstance(e, KeyboardInterrupt):
            raise
        return None


def main():
    name, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    path = F.fetch(name, F.pins()[name])
    R.init(name, path)
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    full = Tokenizer.from_str(raw)
    tj = json.loads(raw)
    subst = substituted(tj)
    tj["added_tokens"] = []
    tj["post_processor"] = None
    none = Tokenizer.from_str(json.dumps(tj))
    tj["added_tokens"] = [{"id": 0, "content": PUA, "single_word": False, "lstrip": False, "rstrip": False,
                           "normalized": False, "special": True}]
    shifted = Tokenizer.from_str(json.dumps(tj))
    n_ids = max(tj["model"]["vocab"].values()) + 1
    rng = random.Random(seed)
    w = sys.stdout.buffer.write

    def ids_rec(kind, flag, b, ids):
        return struct.pack("<BBI", kind, flag, len(b)) + b + struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids)

    def dec_rec(skip, ids, s):
        return (struct.pack("<BBI", 3, int(skip), len(ids)) + struct.pack(f"<{len(ids)}I", *ids) +
                struct.pack("<I", len(s)) + s)

    for _ in range(n):
        _, text = R.gen_text(rng)
        text = text.replace(PUA, "")
        b = text.encode("utf-8")
        at_start = 0 if rng.random() < 0.2 else 1
        if at_start:
            ids = hf_ok(lambda: none.encode(text, add_special_tokens=False).ids)
        else:
            ids = hf_ok(lambda: shifted.encode(PUA + text, add_special_tokens=False).ids[1:])
        if ids is None:
            w(struct.pack("<BBI", 4, at_start, len(b)) + b)
            ids = []
        else:
            w(ids_rec(1, at_start, b, ids))
        for how in (0, 1):
            full.encode_special_tokens = how == 1
            fids = hf_ok(lambda: full.encode(text, add_special_tokens=how == 0).ids)
            if fids is None:
                continue
            w(ids_rec(5, how, b, fids))
            if how == 0 and rng.random() < 0.5:
                skip = rng.random() < 0.5
                s = hf_ok(lambda: full.decode(fids, skip_special_tokens=skip))
                if s is not None:
                    w(dec_rec(skip, fids, s.encode("utf-8")))
        full.encode_special_tokens = False
        if rng.random() < 0.5:
            piece = R.gen_piece(rng)
            if not any(c in subst for c in piece):
                pids = hf_ok(lambda: [t.id for t in none.model.tokenize(piece)])
                if pids is not None:
                    pb = piece.encode("utf-8")
                    w(struct.pack("<BI", 2, len(pb)) + pb + struct.pack("<I", len(pids)) +
                      struct.pack(f"<{len(pids)}I", *pids))
        if rng.random() < 0.5:
            dids = [i for i in (ids if rng.random() < 0.5 else R.gen_ids(rng)) if i < n_ids]
            skip = rng.random() < 0.3
            s = hf_ok(lambda: full.decode(dids, skip_special_tokens=skip))
            if s is not None:
                w(dec_rec(skip, dids, s.encode("utf-8")))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
