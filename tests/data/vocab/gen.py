#!/usr/bin/env python3
"""tests/data/vocab/gen.py: the vocabulary-lookup fixtures test_vocab.c loads (run from the repository root).

    python3 tests/data/vocab/gen.py

written_tie.json: tests/data/breadth/dup_added.json's byte-level model (295 vocabulary entries), its added tokens
dropped, plus two entries whose strings decode to the same bytes E2 80 8D: the raw U+200D (id 295: a char outside
the ByteLevel alphabet, so hf's decoder takes its utf-8) and that alphabet's spelling of the same three bytes,
"\\u00e2\\u0122\\u012f" (id 296). hf 0.23.2: token_to_id("\\u200d") = 295, token_to_id("\\u00e2\\u0122\\u012f") = 296.
toks_token_to_id(b"\\xe2\\x80\\x8d") must be 295, the id the file writes as those very bytes, not the later 296
(dg-tiny-cohere holds this pair: 264 and 35927).

byte_plus.json: tests/data/spm/llamalike.json (byte fallback, a ByteFallback decoder) with the vocabulary's "<0x0F>"
(id 16) spelled "<0x+F>". hf's ByteFallback decoder reads a token "<0x" h h ">" with u8::from_str_radix, which takes a
'+': hf 0.23.2 decodes id 16 as the one byte 0F. So does toks: TOKS_ID_BYTE, decode 0F (toks.h's "(or hf's
<0x+F>)"); its string stays the piece as written, as every sentencepiece piece's.

content_first.json: tests/data/spm/norm_specials.json plus two special added tokens listed after it: the content
"\\u2581<q>" (id 299, normalized false) and "<q>" (id 300, normalized true, so its string is "\\u2581<q>" as </s>'s is
"\\u2581</s>"). Both ids decode to the same bytes and neither is a vocabulary string, so only "an added token's content
first" (toks.h) decides: toks_token_to_id(b"\\xe2\\x96\\x81<q>") must be 299, which hf 0.23.2's token_to_id gives that
text, not the later 300 (every other file holds such a content as the later id, where "the later one" agrees).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def dump(d, name):
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def written_tie():
    with open(os.path.join(HERE, "..", "breadth", "dup_added.json"), encoding="utf-8") as f:
        d = json.load(f)
    d["added_tokens"] = []
    v = d["model"]["vocab"]
    n = len(v)
    v["\u200d"] = n
    v["\u00e2\u0122\u012f"] = n + 1
    dump(d, "written_tie.json")


def content_first():
    with open(os.path.join(HERE, "..", "spm", "norm_specials.json"), encoding="utf-8") as f:
        d = json.load(f)
    n = max(a["id"] for a in d["added_tokens"]) + 1          # 299: after <pad> (297) and "\u2581</s>" (298)
    for i, (content, normalized) in enumerate((("\u2581<q>", False), ("<q>", True))):
        d["added_tokens"].append({"id": n + i, "content": content, "single_word": False, "lstrip": False,
                                  "rstrip": False, "normalized": normalized, "special": True})
    dump(d, "content_first.json")


def byte_plus():
    with open(os.path.join(HERE, "..", "spm", "llamalike.json"), encoding="utf-8") as f:
        d = json.load(f)
    v = d["model"]["vocab"]
    assert v.pop("<0x0F>") == 16
    v["<0x+F>"] = 16
    dump(d, "byte_plus.json")


def main():
    written_tie()
    content_first()
    byte_plus()


if __name__ == "__main__":
    main()
