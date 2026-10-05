#!/usr/bin/env python3
"""tests/parity/vocab_sweep.py — toks_token_to_id / toks_id_flags (toks.h) against hf on every tokenizer given.

    uv run --with tokenizers==0.23.2 python tests/parity/vocab_sweep.py --toks ./build/toks_driver \\
        ~/.cache/toks/tokenizers/*

Per file toks loads (a refusal is listed, not compared), through toks_driver's VOCAB op:
  ids      for every id with hf's string s = id_to_token(id): toks_token_to_id(toks_token(id)) == hf's
           token_to_id(s); or, where another id decodes to the same bytes (a ByteLevel vocab "ĠĠ" and an
           added "  "; a raw "\u200d" and the alphabet form "âĢį"), the id hf gives that text, hf
           token_to_id(decode([id])): toks' key is the bytes (counted as "shared bytes", not a diff)
  contents for every added token in the file's list: toks_token_to_id(content) == hf token_to_id(content)
  flags    ADDED == the id is in hf's added_tokens_decoder, SPECIAL == some listing of its content in the file's
           added_tokens is special (hf's special_tokens_set), BYTE == the id stands for one raw byte: hf's string is a
           <0xHH> byte token and the decoder has ByteFallback, or the decoder is ByteLevel and the id decodes to one
           byte (its string through the alphabet when every char is in it, else its utf-8)
Prints one line per file and the totals; exit 0 iff every compared value agrees.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import Driver, DriverError  # noqa: E402


def byte_token(s):
    """hf ByteFallback's rule: '<0x' h h '>' with u8::from_str_radix(.., 16) (two hex digits, or '+' and one)"""
    if s is None or len(s) != 6 or not s.startswith("<0x") or not s.endswith(">"):
        return False
    h = s[3:5]
    if h[0] == "+":
        h = h[1:]
    return bool(h) and all(c in "0123456789abcdefABCDEF" for c in h)


def _alphabet():
    """hf ByteLevel's char -> byte map (gpt-2's bytes_to_unicode, inverted)"""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


U2B = _alphabet()


def bl_bytes(s):
    """the bytes hf's ByteLevel decoder gives one token string: through the alphabet when every char is in it"""
    return bytes(U2B[c] for c in s) if all(c in U2B for c in s) else s.encode("utf-8")


def tok_file(p):
    """the tokenizer.json hf reads for a path toks loads (a file, or a model directory holding one)"""
    if os.path.isdir(p):
        q = os.path.join(p, "tokenizer.json")
        return q if os.path.isfile(q) else None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toks", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tot = {"files": 0, "refused": 0, "ids": 0, "contents": 0, "flags": 0, "diffs": 0}
    for path in sorted(args.paths):
        f = tok_file(path)
        if f is None:
            continue
        name = os.path.basename(path.rstrip("/"))
        try:
            t = Tokenizer.from_file(f)
        except Exception as e:  # noqa: BLE001
            print(f"{name:28s} hf refuses: {str(e)[:60]}")
            continue
        drv = Driver([args.toks])
        try:
            drv.load(path)
        except DriverError as e:
            print(f"{name:28s} toks refuses ({e})")
            tot["refused"] += 1
            drv.close()
            continue
        js = json.load(open(f, encoding="utf-8"))
        contents = [a["content"] for a in js.get("added_tokens", []) if a.get("content")]
        payload = len(contents).to_bytes(4, "little") + b"".join(
            len(c.encode()).to_bytes(4, "little") + c.encode() for c in contents)
        st, out = drv._req(7, 0, b"", payload)
        drv.close()
        if st != 0:
            print(f"{name:28s} VOCAB status {st}")
            tot["diffs"] += 1
            continue
        v = [int.from_bytes(out[4 * i:4 * i + 4], "little", signed=True) for i in range(len(out) // 4)]
        n_ids = (len(v) - len(contents)) // 2
        dec = t.get_added_tokens_decoder()
        dj = json.dumps(js.get("decoder"))
        bf = '"ByteFallback"' in dj
        bl = '"ByteLevel"' in dj
        specials = {a["content"] for a in js.get("added_tokens", []) if a.get("special")}
        diffs, ex = 0, []
        n_id = n_fl = shared = 0
        for i in range(n_ids):
            fl, rt = v[2 * i], v[2 * i + 1]
            s = t.id_to_token(i)
            one = s is not None and ((bf and byte_token(s)) or (bl and len(bl_bytes(s)) == 1))
            want = (1 if i in dec else 0) | (2 if i in dec and dec[i].content in specials else 0) | (4 if one else 0)
            n_fl += 1
            if fl != want:
                diffs += 1
                ex.append(f"flags({i}) {fl} want {want}")
            if s is None or rt == -1000:
                if (s is None) != (rt == -1000):       # one side has a string, the other none: reported, not a diff
                    ex.append(f"id {i}: hf string {s!r}, toks {'none' if rt == -1000 else 'one'}")
                continue
            n_id += 1
            hr = t.token_to_id(s)
            if hr is not None and rt != hr:
                text = t.decode([i], skip_special_tokens=False)
                if 0 <= rt < n_ids and t.token_to_id(text) == rt:
                    shared += 1                         # the bytes name the id hf gives that text
                    continue
                diffs += 1
                ex.append(f"id {i} {s!r}: toks {rt} hf {hr}")
        for j, c in enumerate(contents):
            got, hr = v[2 * n_ids + j], t.token_to_id(c)
            if got != (hr if hr is not None else -7):
                diffs += 1
                ex.append(f"content {c!r}: toks {got} hf {hr}")
        tot["files"] += 1
        tot["ids"] += n_id
        tot["contents"] += len(contents)
        tot["flags"] += n_fl
        tot["diffs"] += diffs
        tot["shared"] = tot.get("shared", 0) + shared
        print(f"{name:28s} ids {n_id:7d} contents {len(contents):5d} flags {n_fl:7d} shared bytes {shared:3d}"
              f" diffs {diffs}" + (f"  e.g. {'; '.join(ex[:3])}" if ex else ""))
    print(f"vocab_sweep: {tot['files']} files ({tot['refused']} refused by toks), {tot['ids']} id strings "
          f"({tot.get('shared', 0)} whose bytes name the id hf gives that text), {tot['contents']} contents, "
          f"{tot['flags']} flags compared, {tot['diffs']} diffs")
    return 1 if tot["diffs"] else 0


if __name__ == "__main__":
    sys.exit(main())
