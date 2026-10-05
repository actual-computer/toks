#!/usr/bin/env python3
"""tests/data/unigram/gen.py: the unigram fixtures of tests/c/test_bound.c, with expectations taken from hf
tokenizers 0.23.2 itself.

    uv run --with tokenizers==0.23.2 tests/data/unigram/gen.py     # rewrites tests/data/unigram/bound_*.json + .expect

Two small unigram models under Metaspace (prepend_scheme always, split) with byte fallback, alike but for the piece ▁:
bound_bf_nometa has none, so a ▁ (a space, or the prefix) is its 3 utf-8 bytes as 3 byte ids: toks_encode_bound's
f = 3 (compile.c toks_bound_terms_of: r 3, g 3); bound_bf_meta has it (r 1, g 1). <name>.expect holds one case per
line, tests/data/spm/gen.py's A lines: A <how> <text hex> <ids...>, how 0 = hf encode(text) (toks flags 0), 1 =
encode_special_tokens, no pp (TOKS_ADDED_NONSPECIAL | TOKS_NO_POSTPROCESS); hex = utf-8 bytes, '-' for empty.
"""
import json
import os

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
R = "\u2581"
META = {"type": "Metaspace", "replacement": R, "prepend_scheme": "always", "split": True}
UNK = {"id": 0, "content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
       "special": True}


def tok(meta_piece):
    pieces = [["<unk>", 0.0]] + [["<0x%02X>" % b, 0.0] for b in range(256)]
    pieces += [[c, -2.0] for c in "abcdehlorw"] + [["he", -3.0], ["ll", -3.0], ["hello", -4.0], ["or", -3.0]]
    if meta_piece:
        pieces += [[R, -2.5], [R + "a", -3.5], [R + "hello", -4.5]]
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [UNK], "normalizer": None,
            "pre_tokenizer": META, "post_processor": None, "decoder": META,
            "model": {"type": "Unigram", "unk_id": 0, "vocab": pieces, "byte_fallback": True}}


TEXTS = ["", "a", "b", " ", "  ", "   a", "a b", "a  b ", "hello", "hello world", " hello  world ", "\u00e9",
         "\U0001F600", "x", "xyz", "a\nb", "<unk>", "a<unk>b", R, R + "a", R + R, "bbbb", "  " * 8, " a" * 8]


def esc(b):
    return b.hex() if b else "-"


def main():
    for name, meta_piece in (("bound_bf_nometa", False), ("bound_bf_meta", True)):
        raw = json.dumps(tok(meta_piece), ensure_ascii=False)
        with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as f:
            f.write(raw)
        hf = Tokenizer.from_str(raw)
        lines = []
        for t in TEXTS:
            for how in (0, 1):
                hf.encode_special_tokens = how == 1
                ids = hf.encode(t, add_special_tokens=how == 0).ids
                lines.append(f"A {how} {esc(t.encode())} {' '.join(map(str, ids))}".rstrip())
        with open(os.path.join(HERE, name + ".expect"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} cases")


if __name__ == "__main__":
    main()
