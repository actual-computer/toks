#!/usr/bin/env python3
"""tests/data/hfshape/gen.py: the fixtures of test_load's test_hf_shape, each one edit of tests/data/compile/gpt2style.json.

hf tokenizers 0.23.2 refuses a refuse_*.json file before any model reads it (its Tokenizer visitor: a top-level key
outside its nine, a version other than the string "1.0"; its derived structs: a declared field given twice, an enum
object of other than one key) and toks refuses it the same way (config.c hf_refuses, TOKS_E_FORMAT). hf loads an
accept_*.json file, reading a repeated key last-wins as toks does. hf loads panic_template_missing.json (its template
names a special token its special_tokens map lacks) and panics on every encode that adds special tokens; toks refuses
it at load (TOKS_E_FORMAT, the diag says so). Run from the source root; --check loads each
file with hf (uv run --with tokenizers==0.23.2) and fails on a verdict other than the file name's.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, os.pardir, "compile", "gpt2style.json")

AT = ('{"id":261,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,'
      '"special":true}')
SEQ_A = '{"Sequence":{"id":"A","type_id":0}}'
TP = ('{"type":"TemplateProcessing","single":[SEQ_A,{"SpecialToken":{"id":"<|endoftext|>","type_id":0}}],'
      '"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{SP}}')
SP = '"<|endoftext|>":{"id":"<|endoftext|>","ids":[261],"tokens":["<|endoftext|>"]}'
TRUNC = '{"direction":"Right","max_length":8,"strategy":"LongestFirst","stride":0}'
PAD = ('{"strategy":{"Fixed":8},"direction":"Right","pad_to_multiple_of":null,"pad_id":261,"pad_type_id":0,'
       '"pad_token":"<|endoftext|>"}')


def doc(top: str = "", version: str | None = '"1.0"', added: str = AT, trunc: str = "null", pad: str = "null",
        post: str | None = None, norm: str = "null") -> str:
    """gpt2style.json with its fields replaced; top is spliced in after the opening brace"""
    j = json.load(open(BASE, encoding="utf-8"))
    parts = [] if version is None else [f'"version":{version}']
    parts += [f'"truncation":{trunc}', f'"padding":{pad}', f'"added_tokens":[{added}]', f'"normalizer":{norm}',
              '"pre_tokenizer":' + json.dumps(j["pre_tokenizer"], separators=(",", ":")),
              '"post_processor":' + (post if post is not None else json.dumps(j["post_processor"], separators=(",", ":"))),
              '"decoder":' + json.dumps(j["decoder"], separators=(",", ":")),
              '"model":' + json.dumps(j["model"], separators=(",", ":"), ensure_ascii=False)]
    return "{" + top + ",".join(parts) + "}\n"


FILES = {
    "refuse_top_key.json": doc(top='"x_unused":[],'),
    "refuse_version_1_1.json": doc(version='"1.1"'),
    "refuse_version_number.json": doc(version="1.0"),
    "refuse_added_twice.json": doc(added=AT.replace('"special":true', '"special":true,"special":true')),
    "refuse_truncation_twice.json": doc(trunc=TRUNC.replace('"stride":0', '"stride":0,"max_length":8')),
    "refuse_padding_twice.json": doc(pad=PAD.replace('"direction":"Right"', '"direction":"Right","direction":"Left"')),
    "refuse_padding_strategy.json": doc(pad=PAD.replace('{"Fixed":8}', '{"Fixed":8,"Fixed":8}')),
    "refuse_post_twice.json": doc(post='{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,'
                                       '"trim_offsets":false}'),
    "refuse_template_piece.json": doc(post=TP.replace("SEQ_A", '{"Sequence":{"id":"A","type_id":0},"Sequence":'
                                                      '{"id":"A","type_id":0}}').replace("SP", SP)),
    "refuse_special_twice.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace(
        "SP", SP.replace('"ids":[261]', '"ids":[261],"ids":[261]'))),
    "accept_top_twice.json": doc(top='"normalizer":{"type":"Lowercase"},'),
    "accept_special_key_twice.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace("SP", SP + "," + SP)),
    "accept_added_unknown.json": doc(added=AT.replace('"special":true', '"special":true,"zzz":1,"zzz":2')),
    "accept_no_version.json": doc(version=None),
    "accept_template.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace("SP", SP)),
    "panic_template_missing.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace(
        "SP", SP.replace('"<|endoftext|>":{"id"', '"<|other|>":{"id"'))),
}


def main() -> None:
    for name, text in FILES.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(text)
    if "--check" in sys.argv:
        import tokenizers
        bad = 0
        for name, text in FILES.items():
            try:
                t = tokenizers.Tokenizer.from_str(text)
                verdict = "accept"
                try:
                    t.encode("a", add_special_tokens=True)
                except BaseException as e:             # loads, then fails the encode that adds special tokens
                    verdict = "panic: " + str(e).splitlines()[0][:80]
            except BaseException as e:                 # a refusal (an exception or a panic)
                verdict = "refuse: " + str(e).splitlines()[0][:80]
            ok = verdict.startswith(name.split("_")[0])
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {name:34s} hf {verdict}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
