#!/usr/bin/env python3
"""tests/data/breadth/gen.py: the breadth fixtures and their hf tokenizers 0.23.2 expectations.

    uv run --with tokenizers==0.23.2 python tests/data/breadth/gen.py

writes next to itself:
  <name>.json   tokenizer.json files toks must load (each spells one breadth feature, docs/breadth.md)
  cases.inc     the texts, and for every fixture x text x flags (0 ALL, 1 NONSPECIAL, 2 NONE, 4 ALL without
                the post-processor) the ids hf returns and (flags 0-2) the pieces' ends python/toks_oracle gives
                (hf's two-phase extraction, hf's own pre-tokenizer on the gaps), plus decode(ids, skip) for flags
                0; read by tests/c/test_breadth.c
Every expectation comes from hf (python/toks_oracle: NONE = the file with added_tokens removed); the json
literals' ids are never trusted (hf re-ids added tokens). The texts are the hand cases below plus RANDOM
seeded strings over an alphabet of the features' edge chars.
"""
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))

from toks_oracle import oracle as O  # noqa: E402

_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}


def alpha(s):
    return "".join(B2U[b] for b in s.encode("utf-8"))


ALPHABET = [B2U[b] for b in range(256)]
GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# merges over byte-level strings (applied in this order: rank = index); the vocab is the alphabet, then
# every merge product in rank order
MERGE_WORDS = ["he", "ll", "hell", "hello", "Ġh", "Ġhello", "or", "ld", "wor", "world", "Ġworld", "ĠĠ",
               "ĠĠĠĠ", "in", "ing", "Ġt", "th", "Ġth", "the", "Ġthe", "er", "Ġw", "Ġwor", "Ġa", "an", "Ġan",
               "Ġand", "ma", "mas", "mask", "wo", "word", "Ġword", "Ċ", "ĊĊ", "Ġ1", "12", "123", "Ã©", "ĉĉ"]


def bpe_model(extra_vocab=(), ignore_merges=False, dropout=None, drop=()):
    """drop: bytes whose alphabet chars the vocab lacks (hf drops them inside the model: docs/breadth.md §4);
    merge words holding one are left out (no merge can make them)."""
    gone = {B2U[b] for b in drop}
    vocab = [c for c in ALPHABET if c not in gone]
    merges = []
    have = set(vocab)
    for w in MERGE_WORDS:
        if w in have or any(c in gone for c in w):
            continue
        # split w into two known parts: the longest known prefix whose rest is known
        for k in range(len(w) - 1, 0, -1):
            if w[:k] in have and w[k:] in have:
                merges.append([w[:k], w[k:]])
                break
        else:
            raise SystemExit(f"cannot build merge for {w!r}")
        vocab.append(w)
        have.add(w)
    for x in extra_vocab:
        if x not in have:
            vocab.append(x)
            have.add(x)
    return {"type": "BPE", "dropout": dropout, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": ignore_merges,
            "vocab": {s: i for i, s in enumerate(vocab)}, "merges": merges}


def added(content, special=True, normalized=False, **kw):
    t = {"id": 0, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": normalized, "special": special}
    t.update(kw)
    return t


def bytelevel(aps=False, trim=True, ur=False):
    return {"type": "ByteLevel", "add_prefix_space": aps, "trim_offsets": trim, "use_regex": ur}


def split(regex, behavior="Isolated", invert=False):
    return {"type": "Split", "pattern": {"Regex": regex}, "behavior": behavior, "invert": invert}


def tokenizer(model, added_tokens, pre, post=None, padding=None):
    n = len(model["vocab"])
    for i, t in enumerate(added_tokens):                     # json ids: hf ignores them; keep them plausible
        t["id"] = model["vocab"].get(t["content"], n + i)
    return {"version": "1.0", "truncation": None, "padding": padding, "added_tokens": added_tokens,
            "normalizer": None, "pre_tokenizer": pre, "post_processor": post,
            "decoder": bytelevel(aps=True, trim=True, ur=True), "model": model}


def fx_added_opts():
    """lstrip / rstrip / single_word / normalized in both phases, special and not (docs/breadth.md §2)."""
    toks = [
        added("<mask>", lstrip=True),                                  # roberta's <mask>
        added("<|end|>", rstrip=True),                                 # phi-3.5 style
        added("<|both|>", lstrip=True, rstrip=True),                   # phi-4 style
        added("[W]", special=False, single_word=True),
        added("word", special=False, single_word=True),                # in the model vocab: its id
        added("dé", special=False, normalized=True, single_word=True), # phase 1, non-ascii
        added(" hi ", special=False, normalized=True),                 # spaces inside, phase 1
        added("  ", special=False, lstrip=True, rstrip=True),          # all \s with both strips
        added("\t", rstrip=True),
        added("@@", single_word=True, lstrip=True, rstrip=True),
        added("<x>", special=False, normalized=True, rstrip=True),     # phase-1 rstrip
        added("<y>", special=False, normalized=True, lstrip=True),     # phase-1 lstrip
        added("_", special=False, single_word=True),                   # \w itself
    ]
    return tokenizer(bpe_model(), toks, {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel()]},
                     post={"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|end|>", "type_id": 0}},
                                                                   {"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "special_tokens": {"<|end|>": {"id": "<|end|>", "ids": [None], "tokens": ["<|end|>"]}}})


def fx_spell_removed():
    """Split(Removed, invert=true) == Isolated; dropout 0.0 == null; BatchLongest == no padding."""
    toks = [added("<|endoftext|>"), added("<|x|>", special=False)]
    return tokenizer(bpe_model(dropout=0.0), toks,
                     {"type": "Sequence", "pretokenizers": [split(CL100K, "Removed", True), bytelevel()]},
                     padding={"strategy": "BatchLongest", "direction": "Left", "pad_to_multiple_of": None,
                              "pad_id": 0, "pad_type_id": 0, "pad_token": "<|endoftext|>"})


def fx_spell_isoinv():
    """Split(Isolated, invert=true) == Isolated; dropout 0 (an integer) and pad_to_multiple_of 0."""
    toks = [added("<|endoftext|>")]
    j = tokenizer(bpe_model(dropout=0), toks,
                  {"type": "Sequence", "pretokenizers": [split(CL100K, "Isolated", True), bytelevel()]},
                  padding={"strategy": "BatchLongest", "direction": "Right", "pad_to_multiple_of": 0,
                           "pad_id": 5, "pad_type_id": 1, "pad_token": "x"})
    return j


RAW_SP = "\u0020"


def fx_nonalpha():
    """vocab strings outside the byte-level alphabet: decode-only ids (docs/breadth.md §1.4). ignore_merges
    is on, "ĠĠĠ" is not a vocab string but raw "   " is, and a raw " " sits after "Ġ": a toks that let the
    raw strings into byte2id / vhash would answer them for model pieces."""
    extra = [RAW_SP * 3, RAW_SP, "€", "<｜begin▁of▁sentence｜>", "Ĳ€", "a b"]
    toks = [added("<｜begin▁of▁sentence｜>"), added(RAW_SP * 3, special=False, normalized=True),
            added("€x", special=False)]
    return tokenizer(bpe_model(extra_vocab=extra, ignore_merges=True), toks,
                     {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel()]},
                     post={"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "<｜begin▁of▁sentence｜>", "type_id": 0}},
                                      {"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "special_tokens": {"<｜begin▁of▁sentence｜>": {"id": "<｜begin▁of▁sentence｜>", "ids": [None],
                                                                     "tokens": ["<｜begin▁of▁sentence｜>"]}}})


def fx_roberta():
    """RobertaProcessing (cls / sep, roberta-base's specials, <mask> lstrip) over ByteLevel(use_regex)."""
    extra = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    m = bpe_model(extra_vocab=extra)
    toks = [added("<s>", normalized=True), added("<pad>", normalized=True), added("</s>", normalized=True),
            added("<unk>", normalized=True), added("<mask>", lstrip=True, normalized=True)]
    j = tokenizer(m, toks, bytelevel(ur=True),
                  post={"type": "RobertaProcessing", "sep": ["</s>", m["vocab"]["</s>"]],
                        "cls": ["<s>", m["vocab"]["<s>"]], "trim_offsets": True, "add_prefix_space": False},
                  padding={"strategy": "BatchLongest", "direction": "Right", "pad_to_multiple_of": None,
                           "pad_id": 1, "pad_type_id": 0, "pad_token": "<pad>"})
    return j


def fx_roberta_seq():
    """RobertaProcessing inside a Sequence after ByteLevel, add_prefix_space true, cls / sep ids that are
    not their strings' vocab ids (hf takes the numbers as written)."""
    m = bpe_model(extra_vocab=["<s>", "</s>"])
    toks = [added("<s>"), added("</s>")]
    return tokenizer(m, toks, bytelevel(ur=True),
                     post={"type": "Sequence", "processors": [
                         bytelevel(aps=True, trim=False, ur=True),
                         {"type": "RobertaProcessing", "sep": ["</s>", 7], "cls": ["<cls>", len(m["vocab"]) - 3],
                          "trim_offsets": False, "add_prefix_space": True}]})


def resolve_template_ids(j):
    """TemplateProcessing ids written as None: the id hf gives the special token's content (loaded without
    the post-processor first; hf assigns added ids at load, the json literal is ignored)."""
    pp = j.get("post_processor")
    if not pp or pp.get("type") != "TemplateProcessing":
        return j
    from tokenizers import Tokenizer
    bare = copy.deepcopy(j)
    bare["post_processor"] = None
    t = Tokenizer.from_str(json.dumps(bare))
    for k, v in pp["special_tokens"].items():
        if v["ids"] == [None]:
            v["ids"] = [t.token_to_id(k)]
    return j


def fx_pp_shapes():
    """hf picks a post-processor by its fields, not its "type" (an untagged enum: Roberta, Bert, ByteLevel,
    Template, Sequence): a "RobertaProcessing" without its two flags loads as BertProcessing (the same
    cls + ids + sep), inside a Sequence whose ByteLevel step adds nothing."""
    m = bpe_model(extra_vocab=["<s>", "</s>"])
    toks = [added("<s>"), added("</s>")]
    return tokenizer(m, toks, bytelevel(ur=True),
                     post={"type": "Sequence", "processors": [
                         {"type": "RobertaProcessing", "sep": ["</s>", m["vocab"]["</s>"]], "cls": ["<s>", 5]},
                         bytelevel(aps=False, trim=True, ur=False)]})


def fx_pp_notype():
    """a Roberta-shaped post-processor with no "type" at all: hf loads it as RobertaProcessing."""
    m = bpe_model(extra_vocab=["<s>", "</s>"])
    toks = [added("<s>"), added("</s>")]
    return tokenizer(m, toks, bytelevel(ur=True),
                     post={"sep": ["</s>", m["vocab"]["</s>"]], "cls": ["<s>", m["vocab"]["<s>"]],
                           "trim_offsets": False, "add_prefix_space": False})


def fx_dup_added():
    """a content listed twice (yi lists <fim_suffix> twice): hf add_tokens keeps the first id, takes the last
    entry's flags (here: rstrip gained, lstrip lost, a move to phase 1) and keeps specialness once given."""
    toks = [
        added("<a>"), added("<a>"),                                       # identical: nothing changes
        added("<b>"), added("<b>", special=False, rstrip=True),           # rstrip gained, still special
        added("<c>"), added("c>", special=False),                         # <c> moves to phase 1, below c>
        added("<c>", special=False, normalized=True),
        added("<d>", special=False, lstrip=True), added("<d>", special=False),   # lstrip lost
        added("  ", special=False, lstrip=True, rstrip=True),
    ]
    return tokenizer(bpe_model(), toks, {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel()]})


def fx_drop_gpt2():
    """a byte-level vocab missing some byte chars (docs/breadth.md §4: SmolLM2 / falcon / bloom / PowerMoE shapes):
    VT (0B), EOT (04), the continuation byte A9 (the second byte of e-acute, U+00A9), the lead F1 (U+40000..); the
    gpt-2 pattern (ByteLevel use_regex)."""
    toks = [added("<|endoftext|>")]
    return tokenizer(bpe_model(drop=(0x0B, 0x04, 0xA9, 0xF1)), toks, bytelevel(ur=True))


def fx_drop_unk(fuse):
    """drop_gpt2's vocab with an unk_token (Spark-X2.5's shape: it lacks the 13 bytes no utf-8 text has): hf puts the
    unk in the word for a char the vocab lacks (one per char, or one per run under fuse_unk) and no merge crosses it."""
    m = bpe_model(extra_vocab=["<unk>"], drop=(0x0B, 0x04, 0xA9, 0xF1))
    m["unk_token"], m["fuse_unk"] = "<unk>", fuse
    return tokenizer(m, [added("<|endoftext|>")], {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel()]})


def fx_drop_nfc():
    """NFC (it composes e + U+0301 into C3 A9 first) + the cl100k split, missing CR (Hy3's 0D) and A9."""
    toks = [added("<|endoftext|>"), added("<|x|>", special=False)]
    j = tokenizer(bpe_model(drop=(0x0D, 0xA9)), toks, {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel()]})
    j["normalizer"] = {"type": "NFC"}
    return j


def fx_trunc_pad():
    """truncation (Right, max_length 9) and Fixed padding (12, Right) over RobertaProcessing: the text's ids cut to
    max_length minus the template's two, then <pad> up to 12 (hf encode()'s order; docs/breadth.md §5)."""
    j = fx_roberta()
    j["truncation"] = {"direction": "Right", "max_length": 9, "strategy": "LongestFirst", "stride": 0}
    j["padding"] = {"strategy": {"Fixed": 12}, "direction": "Right", "pad_to_multiple_of": None,
                    "pad_id": 1, "pad_type_id": 0, "pad_token": "<pad>"}
    return j


def fx_trunc_left():
    """truncation without a post-processor (max_length 5, OnlyFirst, stride 2) and BatchLongest padding to a
    multiple of 4 on the Left: one encoding padded up from its own length."""
    m = bpe_model(extra_vocab=["<pad>"])
    j = tokenizer(m, [added("<pad>")], bytelevel(ur=True),
                  padding={"strategy": "BatchLongest", "direction": "Left", "pad_to_multiple_of": 4,
                           "pad_id": m["vocab"]["<pad>"], "pad_type_id": 0, "pad_token": "<pad>"})
    j["truncation"] = {"direction": "Right", "max_length": 5, "strategy": "OnlyFirst", "stride": 2}
    return j


def fx_digits_gpt2():
    """the digits-gpt2 template (SmolLM2 / PowerMoE / starcoder): Digits(individual_digits) then ByteLevel's gpt-2
    regex on each stretch between the digits (kernels.md §3 A8), with hf Digits' is_numeric (13 code points beyond
    \\p{N}: U+11DE0..11DE9, U+16FF4..16FF6)."""
    toks = [added("<|endoftext|>"), added("<|x|>", special=False)]
    return tokenizer(bpe_model(), toks, {"type": "Sequence", "pretokenizers": [
        {"type": "Digits", "individual_digits": True}, bytelevel(ur=True)]})


def fx_digits_p16():
    """MiniCPM5's chain (P16): Split \\p{N}{1,3} first, then cl100k with \\p{N}+ on each stretch: the cl100k template
    with A8 on N13 groups (kernels.md §3); A5 still ends a whitespace run at its last NL ("I\\n\\n 1")."""
    p16 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}+")
    return tokenizer(bpe_model(), [added("<|endoftext|>")], {"type": "Sequence", "pretokenizers": [
        split(r"\p{N}{1,3}"), split(p16), bytelevel()]})


def fx_gen_falcon():
    """the generic engine (docs/algorithms/generic.md), falcon's chain: Punctuation(Contiguous), ByteLevel's gpt-2
    regex, then Digits(contiguous) and a Split in the byte-level alphabet ("\u00bd" = C2 BD: BD's char is a number, so
    the cut falls inside the char; pieces end at the char's end, as hf's offsets do)."""
    return tokenizer(bpe_model(), [added("<|endoftext|>")], {"type": "Sequence", "pretokenizers": [
        {"type": "Punctuation", "behavior": "Contiguous"}, bytelevel(ur=True), {"type": "Digits", "individual_digits": False},
        split("[0-9][0-9][0-9]")]})


def fx_gen_behaviors():
    """the generic engine: Laguna's newline runs MergedWithNext (a lookahead), P19's \\s+$, a String pattern
    MergedWithPrevious, Removed, MergedWithPrevious with invert, Contiguous, then LLaDA's possessive quantifiers with
    (?i:) and \\p{..}, \\p{N}{1,3}."""
    lit = {"type": "Split", "pattern": {"String": "--"}, "behavior": "MergedWithPrevious", "invert": False}
    return tokenizer(bpe_model(), [added("<|endoftext|>")], {"type": "Sequence", "pretokenizers": [
        split(r"(?:\r?\n)+(?!\r?\n)", "MergedWithNext"), split(r"\s+$"), lit, split(r"\t", "Removed"),
        split("q", "MergedWithPrevious", invert=True), split("zz?", "Contiguous"),
        split(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"
              r"\s+(?!\S)|\s+"), bytelevel()]})


def fx_gen_groups():
    """the generic engine: tiny-Cohere2's digit groups (a lookahead with a repeat and \\b), bloom's nested class
    (it excludes neither '[' nor ']'), DeepSeek-Coder-V2's CJK / letter ranges, Digits(individual)."""
    return tokenizer(bpe_model(), [added("<|endoftext|>")], {"type": "Sequence", "pretokenizers": [
        split(r"\d{1,3}(?=(?:\d{3})*\b)"), split("[\u4e00-\u9fa5\u0800-\u4e00\uac00-\ud7ff]+"),
        split(" ?[^(\\s|[.,!?\u2026\u3002\uff0c\u3001\u0964\u06d4\u060c])]+"), {"type": "Digits", "individual_digits": True},
        bytelevel()]})


QWEN2 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}")
BLOOM = " ?[^(\\s|[.,!?\u2026\u3002\uff0c\u3001\u0964\u06d4\u060c])]+"


def fx_tmpl_chains():
    """four census chains as DATA on the cl100k template (kernels.md §3): Laguna's newline runs MergedWithNext before
    qwen 2 (A9), bloom's one class (A10, TOKS_CLASSES_BLOOM), LLaDA / Ling's and zeta's spellings of qwen 2."""
    llada = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    zeta = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
    seq = lambda *s: {"type": "Sequence", "pretokenizers": list(s) + [bytelevel()]}
    toks = [added("<|endoftext|>")]
    return {"tmpl_p20": tokenizer(bpe_model(), toks, seq(split(r"(?:\r?\n)+(?!\r?\n)", "MergedWithNext"), split(QWEN2))),
            "tmpl_p21": tokenizer(bpe_model(), toks, seq(split(BLOOM))),
            "tmpl_p22": tokenizer(bpe_model(), toks, seq(split(llada))),
            "tmpl_p25": tokenizer(bpe_model(), toks, seq(split(zeta)))}


FIXTURES = {"added_opts": fx_added_opts, "spell_removed": fx_spell_removed, "spell_isoinv": fx_spell_isoinv,
            "nonalpha": fx_nonalpha, "roberta": fx_roberta, "roberta_seq": fx_roberta_seq,
            "pp_shapes": fx_pp_shapes, "pp_notype": fx_pp_notype, "dup_added": fx_dup_added,
            "drop_gpt2": fx_drop_gpt2, "drop_nfc": fx_drop_nfc, "drop_unk": lambda: fx_drop_unk(False),
            "drop_unk_fuse": lambda: fx_drop_unk(True), "trunc_pad": fx_trunc_pad,
            "trunc_left": fx_trunc_left, "digits_gpt2": fx_digits_gpt2, "digits_p16": fx_digits_p16, "gen_falcon": fx_gen_falcon,
            "gen_behaviors": fx_gen_behaviors, "gen_groups": fx_gen_groups}
FIXTURES.update({k: (lambda k=k: fx_tmpl_chains()[k]) for k in ("tmpl_p20", "tmpl_p21", "tmpl_p22", "tmpl_p25")})

HAND = [
    "", "hello world", "hello  world", "hello   world   ", "a\tb", "\t\t", "  \t  ",
    "<mask>", " <mask>", "a   <mask> b", "x<mask>", "\u3000\u00a0<mask>", "<mask><mask>", "  <mask>  <mask>",
    "<|end|>", "<|end|>   tail", "<|end|>\n\nx", "a <|end|>  <|end|> b", "<|end|>    x", "<|end|>  x",
    "<|both|>", "  <|both|>  ", "x  <|both|>  y", "<|both|><|both|>", " <|both|> <|both|> ",
    "[W]", "a[W]", "[W]a", " [W] ", "_[W]_", "-[W]-", "é[W]", "[W]\u200d", "\u0301[W]", "٣[W]", "[W]\u00a0",
    "word", "words", "sword", " word ", "word-word", "wordword", "a word, a sword, a word_",
    "dé", "dédé", "adé", " dé ", "déa", "Dé dé", "<mask>dé", "dé<mask>",
    " hi ", "say hi to me", "  hi  ", " hi  hi ", "<|end|> hi there",
    "@@", "a@@b", " @@ ", "x @@ y", "@@@@", "@@ @@",
    "<x>", "<x>   y", "a <y>", "   <y>", "<x>  <y>", "<x><y>", "<y>  <x>",
    "_", "a_b", " _ ", "__", "<|endoftext|>", "a<|endoftext|>b", "<|x|>", "x <|x|> y",
    "\u20ac", "\u20acx", "a   b", "a   ", "   ", " ", "a b", "<｜begin▁of▁sentence｜>hi",
    "<s>hello</s>", "<pad><unk><mask>", "the <mask> and <mask>", "hello\n\nworld 123 1234",
    "caf\u00e9 \u00e9t\u00e9", "don't stop", "\U0001F600 emoji", "\u2028line\u2029para",
    "<a><a>", " <a> ", "<b>  x", "x<b>   <b> y", "<c>", "a<c>b", "<c>>", "  <d>", "x  <d>  y", "<b>  <d>",
    "<x>  hi ", "a<x>  hi there", "<|end|>  hi ",
    # bytes the drop fixtures lack (VT, EOT, CR, A9 inside e-acute and the copyright sign, the F1 lead)
    "a\x0bb", "\x04", "\x04\x04 x\x04", "hello\x0bworld \x0b\x0b", "caf\u00e9 \u00e9\u00e9 \u00a9 x\u00a9y",
    "cafe\u0301 e\u0301\u0301", "\U00040000x \U00040001\U00040002 y", "x\r\ny\r", "line\r\n\r\nend\r",
    "\r", "\u00e9", " \x0b", "\x0b ", "<|endoftext|>\x04<|x|>\r\u00e9",
    # digits as hard cuts (digits_gpt2): whitespace runs before a digit, the 13 newer numerics, other N
    "a 1", "a  1", "a   1b", "x\t\t2", "1 2  3   4", "\U00011de0\U00011de1 a", " \U00016ff4", "x  \U00011de5y",
    "12345678", "call(1, 2)", "v1.2.3", "price: $1,234.56", "a\n\n1", "\u0663\u0663 \u0663", "\u00b2\u00b3 x",
    "\u3000\u3000 9", "1'2", "'1s", "don't 1't", "\u216b x  \u00bd", "a \n 7 \r\n8", "  1  ", "1\u0301",
    "hello world 2024 \U0001d7ce", " \U00011de9 \U00011de9  ",
    # the generic engine's chains (gen_*): newline runs, lookaheads, folds, possessives, digit groups, CJK ranges
    "x\r\n\r\ny", "a\n\n\nb\n", "1234567 89", "12,345,678.9x", "IT'S 'Ll 'Re", "a'\u017f \u212a", "\u00bd \u00be1",
    "\u4e00\u9fa5\u9fa6\uac00\u0800", "a.b,c!d?e\u2026f\u3002 [x] (y)", "  x  \n", "tail \n  ", "a--b--", "aqqbq q",
    "zzzz zz z", "a\tb\t\tc", "x\u00a0\u3000y\u2028", "--\t--", "\u0663\u0664\u0665 123", "a----b", "------ x--", "I\n\n 1234", "a \n 1", "x\r\n 5", "\n  12 \n\n7",
    # the cl100k template as data (tmpl_p2x): A9's newline-run starts (CR LF, lone CR, runs with spaces between, a P
    # run's tail), A10's ' '-only give-back over bloom's class and its 14 literals, qwen 2 spelled two more ways
    "a.\nb", "a.\r\nb", "a.\r\rb", "a.\r\r\nb", "a  \nb", "x \n \n y", "\n\n  \n", "\r\n\r\n x", "a\r b\r\n",
    "a.\n\n\nb.\n", "!!\r\n\r\n", "  \r\n  x", "x\n\r\ny", "\n \r x", "end.\n", "\t\n\t", "a\r\n\n\r\n b",
    "a  b", "x..y", "x. .y", " (x)", "a|b", "1.5 x", "hello, world!", "a\u2026 b", "\u0964\u06d4\u060c x",
    "\u3001\u3002\uff0c\u4e2d\u6587", "  ,  x", "a ,b", "? ?x", "a\u00a0 b", "x\u3000 y", "x\t y", "(a) [b] {c}",
    "it's 'S 'll", "a\u0301 b\u0301", "x  \n  y", "IT'S 'LL", "f(x) = y;\n",
]

EDGE = (["a", "b", "z", "Z", "_", "-", ".", ",", "'", "1", "٣", "é", "ß", "汉", "\u0301", "\u200d", "\u200c",
         "\x0b", "\x04", "\r", "\u00a9", "\U00040000",
         " ", " ", " ", "  ", "\t", "\n", "\r\n", "\u00a0", "\u3000", "\u2028", "\u0085", "\u001c", "\u20ac",
         "hello", "world", "word", "the", "and"] +
        ["<mask>", "<|end|>", "<|both|>", "[W]", "dé", " hi ", "@@", "<x>", "<y>", "<|endoftext|>", "<|x|>",
         "<s>", "</s>", "<pad>", "<unk>", "   ", "€x", "<｜begin▁of▁sentence｜>", "<a>", "<b>", "<c>", "c>", "<d>"])


def texts(n_random, seed):
    rng = random.Random(seed)
    out = list(HAND)
    for _ in range(n_random):
        out.append("".join(rng.choice(EDGE) for _ in range(rng.randint(1, 14))))
    return out


FLAGS = [0, 1, 2, 4]


def c_bytes(b):
    out, n = [], 0
    for x in b:
        if 0x20 <= x < 0x7F and x not in (0x22, 0x5C, 0x3F):
            out.append(chr(x))
        else:
            out.append(f"\\{x:03o}")
        n += 1
        if n % 64 == 0 and n < len(b):
            out.append('"\n        "')
    lit = '"' + "".join(out) + '"'
    return f"({lit})" if len(b) > 64 else lit     # parenthesized: clang's -Wstring-concatenation in arrays


def main():
    ts = texts(int(os.environ.get("BREADTH_RANDOM", "60")), 7)
    lines = ["/* generated by tests/data/breadth/gen.py from hf tokenizers 0.23.2; do not edit */",
             "struct br_dec1 { uint32_t n; const char *s; };"]
    lines.append("static const char *const BR_TEXT[] = {")
    lines += [f"    {c_bytes(t.encode())}," for t in ts]
    lines.append("};")
    lines.append(f"#define BR_N_TEXTS {len(ts)}")
    lines.append("static const uint32_t BR_LEN[] = { " + ", ".join(str(len(t.encode())) for t in ts) + " };")
    cases, decs = [], []
    names = []
    for name, fn in FIXTURES.items():
        j = resolve_template_ids(fn())
        with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=1)
            f.write("\n")
        tok = O.load_synthetic(copy.deepcopy(j), name)
        names.append(name)
        for ti, t in enumerate(ts):
            for fl in FLAGS:
                mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[fl & 3]
                pe = [] if fl & 4 else tok.pieces(t, mode=mode)[1]
                cases.append((name, ti, fl, tok.encode(t, mode=mode, add_special_tokens=not (fl & 4)), pe))
            ids = tok.encode(t, mode="ALL", add_special_tokens=True)
            for skip in (0, 1):
                decs.append((name, ti, skip, tok.decode(ids, skip_special_tokens=bool(skip)).encode("utf-8")))
        # every id decoded alone (the decode-only ids included)
        n_ids = max(tok.t.get_vocab(with_added_tokens=True).values()) + 1
        lines.append(f"static const struct br_dec1 BR_DEC1_{name}[{n_ids}] = {{")
        for i in range(n_ids):
            d = tok.decode([i], skip_special_tokens=False).encode("utf-8")
            lines.append(f"    {{ {len(d)}u, {c_bytes(d)} }},")
        lines.append("};")
        lines.append(f"#define BR_N_IDS_{name} {n_ids}u")
        print(f"{name}: n_ids {n_ids}")
    for k, (name, ti, fl, ids, pe) in enumerate(cases):
        lines.append(f"static const uint32_t BR_V{k}[] = {{ {', '.join(map(str, ids)) if ids else '0'} }};")
        lines.append(f"static const uint32_t BR_P{k}[] = {{ {', '.join(map(str, pe)) if pe else '0'} }};")
    lines.append("static const struct { const char *tok; uint32_t text, flags, n; const uint32_t *v; uint32_t np; "
                 "const uint32_t *p; } BR[] = {")
    for k, (name, ti, fl, ids, pe) in enumerate(cases):
        lines.append(f'    {{ "{name}", {ti}u, {fl}u, {len(ids)}u, BR_V{k}, {len(pe)}u, BR_P{k} }},')
    lines.append("};")
    lines.append(f"#define BR_N {len(cases)}")
    lines.append("static const struct { const char *tok; uint32_t text, skip, n; const char *s; } BR_DEC[] = {")
    for name, ti, skip, d in decs:
        lines.append(f'    {{ "{name}", {ti}u, {skip}u, {len(d)}u, {c_bytes(d)} }},')
    lines.append("};")
    lines.append(f"#define BR_N_DEC {len(decs)}")
    lines.append("static const char *const BR_FIXTURES[] = { " + ", ".join(f'"{n}"' for n in names) + " };")
    lines.append(f"#define BR_N_FIXTURES {len(names)}")
    with open(os.path.join(HERE, "cases.inc"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote cases.inc: {len(cases)} encode cases, {len(decs)} decode cases, {len(ts)} texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
