#!/usr/bin/env python3
"""tests/parity/stream_hold.py — the stream hold's receipt (toks.h toks_stream_hold) against hf.

    uv run --with tokenizers==0.23.2 python tests/parity/stream_hold.py \\
        --toks ./build/toks_driver ~/.cache/toks/tokenizers/gemma4 ~/.cache/toks/tokenizers/mistral-v0.3 ...

Per tokenizer with byte-fallback tokens <0xF0> <0x93> <0x80>: the runs of U+13000 (F0 93 80 80) that a
stream holds whole (12 of them, 48 bytes, over st's own 44; 300 of them, 1200 bytes; hf's own encode of
12 x U+13000; 12 then a stray 0x80, which makes the run invalid as a whole), each pushed one id at a time
through toks_driver's STREAM: flag 0 (st's own 44 bytes) and flag 2 (a caller's hold grown on
TOKS_E_LIMIT). Printed per case: hf.decode == the stream's concatenation (must hold always, flag 2),
hf's DecodeStream == the stream (must hold for the valid runs; the stray byte shows where DecodeStream
contradicts hf.decode), and flag 0's status (TOKS_E_LIMIT, -9, past 44 held bytes). Exit 0 iff the
equalities that must hold do.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toks_oracle import oracle as O  # noqa: E402
from run import Driver  # noqa: E402


def decode_stream(t, ids):
    """hf's DecodeStream, one id per step: the concatenation of what it emitted"""
    from tokenizers.decoders import DecodeStream

    ds = DecodeStream(skip_special_tokens=False)
    out = []
    for i in ids:
        s = ds.step(t, i)
        if s is not None:
            out.append(s)
    return "".join(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toks", required=True, help="the driver binary (tests/driver/toks_driver.c)")
    ap.add_argument("tokenizers", nargs="+")
    args = ap.parse_args()
    bad = 0
    for path in args.tokenizers:
        tok = O.load(path)
        t = tok.t
        byte_id = {}
        for i in range(t.get_vocab_size(True)):
            s = t.id_to_token(i)
            if s is not None and len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
                try:
                    byte_id.setdefault(int(s[3:5], 16), i)
                except ValueError:
                    pass
        name = os.path.basename(path.rstrip("/"))
        if not all(b in byte_id for b in (0xF0, 0x93, 0x80)):
            print(f"{name}: SKIP (no byte tokens for F0 93 80)")
            continue
        u = [byte_id[0xF0], byte_id[0x93], byte_id[0x80], byte_id[0x80]]
        enc = t.encode("\U00013000" * 12, add_special_tokens=False).ids
        cases = [("12 x U+13000", u * 12, True), ("300 x U+13000", u * 300, True),
                 ("hf encode(12 x U+13000)", enc, True), ("12 x U+13000 + <0x80>", u * 12 + [byte_id[0x80]], False)]
        drv = Driver([args.toks])
        drv.load(path)
        for label, ids, valid in cases:
            want = t.decode(ids, skip_special_tokens=False).encode("utf-8")
            st2, outs = drv.stream([[i] for i in ids], 2)
            got = b"".join(outs) if st2 == 0 else None
            held = sum(1 for o in outs[:-1] if o == b"") if st2 == 0 else -1
            st0, _ = drv.stream([[i] for i in ids], 0)
            ds = decode_stream(t, ids)
            eq_dec = got == want
            eq_ds = got == ds
            ok = eq_dec and (eq_ds or not valid)
            bad += not ok
            print(f"{name:14s} {label:26s} ids {len(ids):5d}  hf.decode {len(want):5d} B  stream(hold) == hf.decode: "
                  f"{eq_dec}  == DecodeStream ({len(ds)} B): {eq_ds}  pushes that emitted nothing {held}  "
                  f"st's own 44: status {st0}{'' if ok else '  FAIL'}")
        drv.close()
    print("stream_hold: " + ("ok" if bad == 0 else f"{bad} FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
