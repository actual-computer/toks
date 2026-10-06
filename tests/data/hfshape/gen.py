#!/usr/bin/env python3
"""tests/data/hfshape/gen.py: the fixtures of test_load's test_hf_shape, each one edit of tests/data/compile/gpt2style.json.

hf tokenizers 0.23.2 refuses a refuse_*.json file before any model reads it (its Tokenizer visitor: a top-level key
outside its nine, a version other than the string "1.0"; its derived structs: a declared field given twice, an enum
object of other than one key; its post-processor, an untagged enum: an object every variant refuses) and toks refuses
it the same way (config.c hf_refuses, TOKS_E_FORMAT). hf loads an accept_*.json file, reading a repeated key
last-wins as toks does; where a post-processor's duplicate makes a variant refuse, hf takes the object as the next
variant in its order (PP below: the variant hf picks), and so does toks (config.c pp_kind). hf loads
panic_template_missing.json (its template names a special token its special_tokens map lacks) and panics on every
encode that adds special tokens; toks refuses it at load (TOKS_E_FORMAT, the diag says so). Run from the source root;
--check loads each file with hf (uv run --with tokenizers==0.23.2) and fails on a verdict other than the file
name's, or a post-processor other than PP's.
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
EOT = '["<|endoftext|>",261]'
ROB = '{"type":"RobertaProcessing","sep":EOT,"cls":EOT,"trim_offsets":true,"add_prefix_space":false}'.replace("EOT", EOT)
BERT = '{"type":"BertProcessing","sep":EOT,"cls":EOT}'.replace("EOT", EOT)
BL = '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}'
BL_TRIM_TWICE = '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"trim_offsets":false}'
TPL = TP.replace("SEQ_A", SEQ_A).replace("SP", SP)


def seq_tpl(edit) -> str:
    """a Sequence[ByteLevel] that also carries a Template's fields, edited: hf takes it as the Sequence when the edit
    makes the Template refuse"""
    return '{"type":"Sequence","processors":[' + BL + '],' + edit(TPL[len('{"type":"TemplateProcessing",'):])


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
    "refuse_post_twice.json": doc(post=BL_TRIM_TWICE),
    "refuse_bytelevel_type_twice.json": doc(post=BL.replace('{"type":"ByteLevel",', '{"type":"ByteLevel","type":"ByteLevel",')),
    "refuse_roberta_sep_twice.json": doc(post=ROB.replace('"cls":', '"sep":' + EOT + ',"cls":')),
    "refuse_bert_cls_twice.json": doc(post=BERT[:-1] + ',"cls":' + EOT + '}'),
    "refuse_template_twice.json": doc(post=TPL.replace('"pair":', '"single":[' + SEQ_A + '],"pair":', 1)),
    "refuse_template_piece_field.json": doc(post=TP.replace("SEQ_A", '{"Sequence":{"id":"A","type_id":0,"type_id":0}}')
                                            .replace("SP", SP)),
    "refuse_sequence_twice.json": doc(post='{"type":"Sequence","processors":[' + BL + '],"processors":[' + BL + ']}'),
    "refuse_sequence_type_twice.json": doc(post='{"type":"Sequence","type":"Sequence","processors":[' + BL + ']}'),
    "refuse_sequence_element.json": doc(post='{"type":"Sequence","processors":[' + BL_TRIM_TWICE + ']}'),
    "refuse_template_piece.json": doc(post=TP.replace("SEQ_A", '{"Sequence":{"id":"A","type_id":0},"Sequence":'
                                                      '{"id":"A","type_id":0}}').replace("SP", SP)),
    "refuse_special_twice.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace(
        "SP", SP.replace('"ids":[261]', '"ids":[261],"ids":[261]'))),
    "accept_top_twice.json": doc(top='"normalizer":{"type":"Lowercase"},'),
    "accept_special_key_twice.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace("SP", SP + "," + SP)),
    "accept_added_unknown.json": doc(added=AT.replace('"special":true', '"special":true,"zzz":1,"zzz":2')),
    "accept_no_version.json": doc(version=None),
    "accept_template.json": doc(post=TPL),
    "accept_roberta_trim_twice.json": doc(post=ROB.replace('"add_prefix_space"', '"trim_offsets":true,"add_prefix_space"')),
    "accept_template_sep_twice.json": doc(post=TPL[:-1] + ',"sep":' + EOT + ',"sep":' + EOT + ',"cls":' + EOT + '}'),
    "accept_bytelevel_sep_twice.json": doc(post=BL[:-1] + ',"sep":' + EOT + ',"sep":' + EOT + ',"cls":' + EOT + '}'),
    "accept_bytelevel_template.json": doc(post=BL_TRIM_TWICE[:-1] + "," + TPL[len('{"type":"TemplateProcessing",'):]),
    "accept_sequence_template_twice.json": doc(post='{"type":"Sequence","processors":[' + BL + '],"single":[' + SEQ_A
                                               + '],' + TPL[len('{"type":"TemplateProcessing",'):]),
    "panic_template_missing.json": doc(post=TP.replace("SEQ_A", SEQ_A).replace(
        "SP", SP.replace('"<|endoftext|>":{"id"', '"<|other|>":{"id"'))),
    # field types: a Template field of a type serde does not read refuses the Template, and hf takes the next variant
    "accept_sequence_typeid_neg.json": doc(post='{"type":"Sequence","processors":[' + BL + '],' + TPL[len(
        '{"type":"TemplateProcessing",'):].replace('{"SpecialToken":{"id":"<|endoftext|>","type_id":0}}',
                                                   '{"SpecialToken":{"id":"<|endoftext|>","type_id":-1}}', 1)),
    "accept_sequence_tokens_int.json": doc(post='{"type":"Sequence","processors":[' + BL + '],' + TPL[len(
        '{"type":"TemplateProcessing",'):].replace('"tokens":["<|endoftext|>"]', '"tokens":[5]')),
    "accept_sequence_piece_id_c.json": doc(post=seq_tpl(lambda b: b.replace('{"Sequence":{"id":"A","type_id":0}},{"Sp',
                                                                       '{"Sequence":{"id":"C","type_id":0}},{"Sp', 1))),
    "accept_sequence_piece_foo.json": doc(post=seq_tpl(lambda b: b.replace('{"Sequence":{"id":"A","type_id":0}},{"Sp',
                                                                      '{"Foo":{"id":"A","type_id":0}},{"Sp', 1))),
    "accept_sequence_special_id_int.json": doc(post=seq_tpl(lambda b: b.replace(
        '{"SpecialToken":{"id":"<|endoftext|>","type_id":0}}', '{"SpecialToken":{"id":5,"type_id":0}}', 1))),
    "accept_sequence_entry_id_int.json": doc(post=seq_tpl(lambda b: b.replace('{"id":"<|endoftext|>","ids"', '{"id":5,"ids"'))),
    "accept_sequence_entry_ids_str.json": doc(post=seq_tpl(lambda b: b.replace('"ids":[261]', '"ids":"261"'))),
    "accept_sequence_entry_ids_neg.json": doc(post=seq_tpl(lambda b: b.replace('"ids":[261]', '"ids":[-1]'))),
    "accept_sequence_single_str.json": doc(post=seq_tpl(lambda b: b[:b.index('"single":')] + '"single":"x",' +
                                                        b[b.index('"pair":'):])),
    "accept_sequence_special_arr.json": doc(post=seq_tpl(lambda b: b[:b.index('"special_tokens":')] +
                                                         '"special_tokens":[]}')),
    # serde reads a struct from an array, by position: hf's Template takes these (toks does not read the form: it
    # refuses the file as unsupported, as it cannot tell whether serde reads the array)
    "accept_template_piece_positional.json": doc(post=seq_tpl(lambda b: b.replace(SEQ_A, '{"Sequence":["A",0]}', 1))),
    "accept_template_entry_positional.json": doc(post=seq_tpl(lambda b: b.replace(
        SP, '"<|endoftext|>":["<|endoftext|>",[261],["<|endoftext|>"]]'))),
    # an entry of the wrong length, which single never names: serde refuses the Template, hf takes the Sequence
    "accept_sequence_entry_positional_short.json": doc(post=seq_tpl(lambda b: b.replace(SP, SP + ',"x":["x",[5]]'))),
    # a sure refusal wins over an array: pair's type_id -1 fails serde's Template whatever single's array holds
    "accept_sequence_positional_refused.json": doc(post=seq_tpl(lambda b: b.replace(SEQ_A, '{"Sequence":["A",0]}', 1)
                                                                .replace('{"Sequence":{"id":"B","type_id":1}}',
                                                                         '{"Sequence":{"id":"B","type_id":-1}}'))),
    # a plain Template with a piece by position: every other variant refuses it, so the file is not FORMAT
    "accept_template_positional_plain.json": doc(post=TPL.replace(SEQ_A, '{"Sequence":["A",0]}', 1)),
    # the whole post-processor as an array: Bert's two fields by position
    "accept_bert_positional.json": doc(post='[' + EOT + ',' + EOT + ']'),
    # serde's map spelling of a unit variant: {"ByteLevel": null} is the type ByteLevel, {"A": null} the sequence A
    "accept_bytelevel_map_type.json": doc(post=BL.replace('"type":"ByteLevel"', '"type":{"ByteLevel":null}')),
    "accept_sequence_map_type.json": doc(post='{"type":{"Sequence":null},"processors":[' + BL + ']}'),
    "accept_template_piece_map_id.json": doc(post=TP.replace("SEQ_A", '{"Sequence":{"id":{"A":null},"type_id":0}}')
                                             .replace("SP", SP)),
    # ByteLevel and Sequence read their own type: missing or misnamed, they refuse
    "refuse_bytelevel_no_type.json": doc(post=BL.replace('"type":"ByteLevel",', '')),
    "refuse_bytelevel_bad_type.json": doc(post=BL.replace('"type":"ByteLevel"', '"type":"Bytelevel"')),
    "refuse_sequence_no_type.json": doc(post='{"processors":[' + BL + ']}'),
    "refuse_sequence_bad_type.json": doc(post='{"type":"sequence","processors":[' + BL + ']}'),
    # field types the other variants read: Bert's sep a (String, u32) pair, ByteLevel's flags booleans
    "refuse_bert_sep_string.json": doc(post=BERT.replace('"sep":' + EOT, '"sep":"<|endoftext|>"')),
    "refuse_bytelevel_flag_int.json": doc(post=BL.replace('"trim_offsets":false', '"trim_offsets":0')),
}

# the post-processor hf builds for an accept_*.json whose duplicate makes a variant refuse (test_load checks the ids)
PP = {"accept_roberta_trim_twice.json": "BertProcessing", "accept_template_sep_twice.json": "TemplateProcessing",
      "accept_bytelevel_sep_twice.json": "ByteLevel", "accept_bytelevel_template.json": "TemplateProcessing",
      "accept_sequence_template_twice.json": "Sequence", "accept_sequence_typeid_neg.json": "Sequence",
      "accept_sequence_tokens_int.json": "Sequence", "accept_bytelevel_map_type.json": "ByteLevel",
      "accept_sequence_map_type.json": "Sequence", "accept_template_piece_map_id.json": "TemplateProcessing",
      "accept_sequence_piece_id_c.json": "Sequence", "accept_sequence_special_id_int.json": "Sequence",
      "accept_sequence_piece_foo.json": "Sequence",
      "accept_sequence_entry_id_int.json": "Sequence", "accept_sequence_entry_ids_str.json": "Sequence",
      "accept_sequence_entry_ids_neg.json": "Sequence", "accept_sequence_single_str.json": "Sequence",
      "accept_sequence_special_arr.json": "Sequence", "accept_template_piece_positional.json": "TemplateProcessing",
      "accept_template_entry_positional.json": "TemplateProcessing", "accept_sequence_entry_positional_short.json": "Sequence",
      "accept_bert_positional.json": "BertProcessing", "accept_sequence_positional_refused.json": "Sequence",
      "accept_template_positional_plain.json": "TemplateProcessing"}


def main() -> None:
    for name, text in FILES.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(text)
    if "--check" in sys.argv:
        import tokenizers
        bad = 0
        for name, text in FILES.items():
            pp = ""
            try:
                t = tokenizers.Tokenizer.from_str(text)
                verdict = "accept"
                pp = type(t.post_processor).__name__
                try:
                    ids = t.encode("a", add_special_tokens=True).ids
                    verdict += f" ({pp}: 'a' -> {ids})"
                except BaseException as e:             # loads, then fails the encode that adds special tokens
                    verdict = "panic: " + str(e).splitlines()[0][:80]
            except BaseException as e:                 # a refusal (an exception or a panic)
                verdict = "refuse: " + str(e).splitlines()[0][:80]
            ok = verdict.startswith(name.split("_")[0]) and (name not in PP or pp == PP[name])
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {name:34s} hf {verdict}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
