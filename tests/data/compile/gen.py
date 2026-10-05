#!/usr/bin/env python3
"""tests/data/compile/gen.py: the compile step's hand fixtures, checked against hf tokenizers 0.23.2.

    uv run --with tokenizers==0.23.2 python tests/data/compile/gen.py

writes next to itself (tests/c/test_compile.c reads them):
  <name>.json    tokenizer.json files toks must accept
  <name>.expect  what toks_compile must build for them; every id comes from hf (get_added_tokens_decoder,
                 encode), never from the json literals (hf ignores an added token's "id" field)
  refuse.txt     line pairs: "<code> <diag substring>", then a one-line tokenizer.json toks must refuse
and fails loudly when hf disagrees with an expectation written here.
"""
import copy
import json
import os
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the byte-level alphabet (gpt-2 bytes_to_unicode) -----------------------------------------------------
_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}
U2B = {u: b for b, u in B2U.items()}


def alpha(bs):
    return "".join(B2U[b] for b in bs)


def decode_bytes(s):
    """hf ByteLevel decoder on one token string: the alphabet image when every char maps, else raw utf-8."""
    if all(c in U2B for c in s):
        return bytes(U2B[c] for c in s)
    return s.encode()


GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN35 = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
TP = {GPT2: 0x11, CL100K: 0x66, QWEN35: 0x6E}   # layout.h TOKS_TP_* bits (kernels.md §3)
# deepseek v3's three Splits (docs/templates/dsv3.md §1): TOKS_TMPL_DSV3 (3), tmpl_params 0
DSV3_SPLITS = ["\\p{N}{1,3}", "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
               "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
               " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"]

ALPHABET = [alpha([b]) for b in range(256)]       # ids 0..255 = byte b's char


def added(content, id_, special=True, normalized=False, **kw):
    t = {"id": id_, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": normalized, "special": special}
    t.update(kw)
    return t


def bytelevel(aps=False, trim=True, ur=None):
    d = {"type": "ByteLevel", "add_prefix_space": aps, "trim_offsets": trim}
    if ur is not None:
        d["use_regex"] = ur
    return d


def split(regex):
    return {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": False}


def template(single, specials):
    return {"type": "TemplateProcessing", "single": single, "pair": single,
            "special_tokens": {k: {"id": k, "ids": v, "tokens": [k] * len(v)} for k, v in specials.items()}}


def st(tok):
    return {"SpecialToken": {"id": tok, "type_id": 0}}


SEQ_A = {"Sequence": {"id": "A", "type_id": 0}}


# ---- fixture A: gpt-2 shaped (untyped model, "a b" merges, ByteLevel split, a normalized special in vocab)
def fixture_gpt2style():
    vocab = ALPHABET + ["he", "ll", "hell", "hello", "Ġh", "<|endoftext|>"]
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [added("<|endoftext|>", 261, special=True, normalized=True)],
        "normalizer": None,
        "pre_tokenizer": bytelevel(),
        "post_processor": bytelevel(aps=True, trim=False),
        "decoder": bytelevel(aps=True),
        "model": {"dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "vocab": {s: i for i, s in enumerate(vocab)},
                  "merges": ["h e", "l l", "he ll", "hell o", "Ġ h"]},
    }


# ---- fixture B: llama-3 shaped (Split + ByteLevel, pair merges, ignore_merges, template, many added tokens)
def fixture_llama3style():
    vocab = ALPHABET + ["he", "ll", "hell", "hello"]           # n_vocab 260
    toks = [
        added("<|begin|>", 260), added("<|end|>", 261),
        added("<|a|>", 262, special=False), added("<|ab|>", 263), added("<|abc|>", 264),
        added("<|x", 265, special=False), added("<|xy|>", 266),
        added("@", 267, special=False),                          # one byte: add_single
        added("", 999),                                          # empty: hf ignores it
        added("hello", 268),                                     # in the model vocab: hf gives it 259
        added("«ñ»", 268, special=False, normalized=True),       # alphabet chars: decodes to ab f1 bb
        added(" hi there ", 269, special=False, normalized=True),  # ' ' is not an alphabet char: raw
        added("ĠĠ", 270, special=False),                         # matched as c4 a0 c4 a0, decodes to "  "
        added("é", 271, normalized=True),                        # two content bytes, decodes to e9
        added("\t", 272, special=False, normalized=True),        # one byte in phase 1
    ]
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": toks,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [split(CL100K), bytelevel(ur=False)]},
        "post_processor": {"type": "Sequence", "processors": [
            bytelevel(aps=True, trim=False, ur=True),
            template([st("<|begin|>"), SEQ_A, st("<|end|>")], {"<|begin|>": [260], "<|end|>": [261, 7]})]},
        "decoder": bytelevel(aps=True, ur=True),
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": {s: i for i, s in enumerate(vocab)},
                  "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"]]},
    }


# ---- fixture C: the qwen 3.5 marks pattern, decoder Sequence[ByteLevel], no added tokens, no post-processor
def fixture_qwen35style():
    vocab = ALPHABET + ["he"]
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [split(QWEN35), bytelevel(trim=False, ur=False)]},
        "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [bytelevel(trim=False, ur=False)]},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {s: i for i, s in enumerate(vocab)}, "merges": [["h", "e"]]},
    }


# ---- fixture E: model vocab strings outside the byte-level alphabet (deepseek v3's <｜begin▁of▁sentence｜> shape:
# one is an added special too, one is not added): no bpe reaches them, their decode is their own utf-8
def fixture_nonalpha():
    j = fixture_llama3style()
    j["model"]["vocab"].update({"<\uff5cx\u2581y\uff5c>": 260, "\u20ac": 261})
    j["added_tokens"].append(added("<\uff5cx\u2581y\uff5c>", 260))
    return j


# ---- fixture F: deepseek v3 shaped: the three-Split chain, an empty normalizer Sequence, no ignore_merges
def fixture_dsv3style():
    j = fixture_nonalpha()
    j["normalizer"] = {"type": "Sequence", "normalizers": []}
    j["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [split(s) for s in DSV3_SPLITS] + [bytelevel(ur=False)]}
    j["model"]["ignore_merges"] = False
    return j


# ---- fixture D: ByteLevel(use_regex=false) alone: no split, no class tables; no added_tokens key at all
def fixture_nosplit():
    return {
        "normalizer": None,
        "pre_tokenizer": bytelevel(trim=False, ur=False),
        "decoder": bytelevel(),
        "model": {"type": "BPE", "vocab": {s: i for i, s in enumerate(ALPHABET + ["ll"])}, "merges": ["l l"]},
    }


ACCEPT = {"gpt2style": fixture_gpt2style, "llama3style": fixture_llama3style,
          "qwen35style": fixture_qwen35style, "nosplit": fixture_nosplit, "nonalpha": fixture_nonalpha,
          "dsv3style": fixture_dsv3style}


def tmpl_of(j):
    """-> (TOKS_TMPL_*, tmpl_params, marks_folded) toks must compile j's pre-tokenizer to."""
    pt = j["pre_tokenizer"]
    if pt["type"] == "ByteLevel":
        return (1, TP[GPT2], 0) if pt.get("use_regex", True) else (0, 0, 0)
    steps = pt["pretokenizers"]
    if [s.get("pattern", {}).get("Regex") for s in steps[:3]] == DSV3_SPLITS:
        return 3, 0, 1
    pat = steps[0]["pattern"]["Regex"]
    return 1, TP[pat], 1 if pat == QWEN35 else 0


def expect(name, j):
    tok = Tokenizer.from_str(json.dumps(j))
    n_vocab = len(j["model"]["vocab"])
    dec = tok.get_added_tokens_decoder()                       # hf's final added vocabulary: id -> token
    by_content = {t.content: i for i, t in dec.items()}
    ids = list(tok.get_vocab(with_added_tokens=True).values())
    n_ids = max(ids) + 1
    tmpl, params, marks = tmpl_of(j)
    lines = [f"n_vocab {n_vocab}", f"n_ids {n_ids}",
             f"tmpl {tmpl} {params:#x}",
             f"marks {marks}",
             f"ignore_merges {1 if j['model'].get('ignore_merges') else 0}"]
    prefix, suffix = [], []
    pp = j.get("post_processor")
    for p in (pp["processors"] if pp and pp["type"] == "Sequence" else [pp] if pp else []):
        if p["type"] == "TemplateProcessing":
            side = prefix
            for piece in p["single"]:
                if "Sequence" in piece:
                    side = suffix
                else:
                    side.extend(p["special_tokens"][piece["SpecialToken"]["id"]]["ids"])
    got = tok.encode("", add_special_tokens=True).ids
    assert got == prefix + suffix, (name, got, prefix, suffix)
    lines.append("prefix " + " ".join(map(str, prefix)))
    lines.append("suffix " + " ".join(map(str, suffix)))
    lines.append("special " + " ".join(str(i) for i in sorted(i for i, t in dec.items() if t.special)))
    survivors = [t for t in j.get("added_tokens", []) if t["content"]]
    assert len(survivors) == len(dec), (name, "an added token hf did not keep")
    for phase in (0, 1):
        for t in survivors:
            if int(t["normalized"]) != phase:
                continue
            i = by_content[t["content"]]
            if t["id"] != i:
                print(f"  {name}: {t['content']!r} json id {t['id']} -> hf id {i}")
            raw = decode_bytes(t["content"]) if i >= n_vocab else decode_bytes(alpha_of(j, i))
            assert tok.decode([i], skip_special_tokens=False) == raw.decode("utf-8", "replace"), (name, i)
            assert tok.encode(t["content"], add_special_tokens=False).ids == [i], (name, t["content"])
            lines.append(f"added {i} {phase} {int(t['special'])} {t['content'].encode().hex()} {raw.hex()}")
    for i in range(n_vocab):                                   # every vocab id decodes to its alphabet bytes
        assert tok.decode([i], skip_special_tokens=False) == decode_bytes(alpha_of(j, i)).decode("utf-8", "replace"), (name, i)
    return "\n".join(lines) + "\n"


def alpha_of(j, i):
    return {v: k for k, v in j["model"]["vocab"].items()}[i]


# ---- refusals: (code, diag substring, mutation of a fixture) ---------------------------------------------
def mut(base, fn):
    j = copy.deepcopy(base())
    fn(j)
    return j


def set_(path, value):
    def f(j):
        d = j
        for k in path[:-1]:
            d = d[k]
        if value is DEL:
            del d[path[-1]]
        else:
            d[path[-1]] = value
    return f


DEL = object()
A, B = fixture_gpt2style, fixture_llama3style
RX0 = ["pre_tokenizer", "pretokenizers", 0, "pattern", "Regex"]


def renumber(j, drop):
    v = [s for s, _ in sorted(j["model"]["vocab"].items(), key=lambda kv: kv[1]) if s != drop]
    j["model"]["vocab"] = {s: i for i, s in enumerate(v)}


PAD = {"strategy": "BatchLongest", "direction": "Right", "pad_to_multiple_of": None, "pad_id": 0, "pad_type_id": 0,
       "pad_token": "!"}

REFUSE = [
    (-3, "normalizer NFD", mut(B, set_(["normalizer"], {"type": "NFD"}))),
    (-3, "normalizer Lowercase", mut(B, set_(["normalizer"], {"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Sequence", "normalizers": [{"type": "Lowercase"}]}]}))),
    (-3, "add_prefix_space=true", mut(A, set_(["pre_tokenizer", "add_prefix_space"], True))),
    (-3, "add_prefix_space=true", mut(B, set_(["pre_tokenizer", "pretokenizers", 1, "add_prefix_space"], True))),
    (-3, "pre_tokenizer absent", mut(A, set_(["pre_tokenizer"], None))),
    # a chain without a template now runs on the generic engine (docs/algorithms/generic.md): an unknown Split regex,
    # a String pattern, any behavior, ByteLevel before or after Splits; only a chain without ByteLevel stays refused
    (-3, "without ByteLevel", mut(B, set_(["pre_tokenizer"], split(CL100K)))),
    # the generic engine's named misses (docs/algorithms/generic.md §3)
    (-3, "ByteLevel twice", mut(A, set_(["pre_tokenizer"], {"type": "Sequence", "pretokenizers": [bytelevel(), bytelevel()]}))),
    (-3, "Split regex: a group other than", mut(B, set_(RX0, "(?<=a)b"))),
    (-3, "Split regex: an escape or property outside", mut(B, set_(RX0, "(a)\\1"))),
    (-3, "Split regex: an escape or property outside", mut(B, set_(RX0, "\\p{Greek}+"))),
    (-3, "Split regex: one that matches the empty string", mut(B, set_(RX0, "a*|b"))),
    (-3, "Split regex: a possessive quantifier beyond one class", mut(B, set_(RX0, "(?:ab)++"))),
    (-3, "Split regex: a lookahead inside a lookahead", mut(B, set_(RX0, "a(?=b(?!c))"))),
    (-3, "Split regex: class intersection", mut(B, set_(RX0, "[a-z&&[^aeiou]]+"))),
    (-3, "Split regex: (?i) beyond ascii letters", mut(B, set_(RX0, "(?i:\u00e9)"))),
    (-3, "Split regex: a quantifier on an assertion or on a quantifier", mut(B, set_(RX0, "a{2}{3}"))),
    (-3, "truncation direction Left", mut(B, set_(["truncation"], {"direction": "Left", "max_length": 8, "strategy": "LongestFirst", "stride": 0}))),
    (-3, "decoder absent", mut(A, set_(["decoder"], None))),
    (-3, "decoder Strip", mut(B, set_(["decoder"], {"type": "Sequence", "decoders": [bytelevel(), {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}))),
    (-3, "decoder Fuse", mut(B, set_(["decoder"], {"type": "Fuse"}))),
    (-3, "without ByteLevel", mut(fixture_dsv3style, set_(["pre_tokenizer", "pretokenizers", 3], split(CL100K)))),
    (-3, "lacks byte-level alphabet chars (under ignore_merges)", mut(B, lambda j: renumber(j, alpha([0x41])))),
    (-3, "lacks byte-level alphabet chars (with byte_fallback)", mut(B, lambda j: (renumber(j, alpha([0x41])),
                                                                             j["model"].update(ignore_merges=False, byte_fallback=True)))),
    (-3, "unk_token not in model.vocab", mut(B, lambda j: (renumber(j, alpha([0x41])),
                                                           j["model"].update(ignore_merges=False, unk_token="<nope>")))),
    (-3, "a merge of model.unk_token", mut(B, lambda j: (renumber(j, alpha([0x41])),
                                                         j["model"].update(ignore_merges=False, unk_token="h")))),
    (-2, "out of vocabulary", mut(B, lambda j: j["model"]["merges"].append(["h", "q!"]))),
    (-2, "left right", mut(A, lambda j: j["model"]["merges"].append("h e l"))),
    (-3, "$B", mut(B, lambda j: j["post_processor"]["processors"][1]["single"].append({"Sequence": {"id": "B", "type_id": 1}}))),
    (-3, "without $A", mut(B, lambda j: j["post_processor"]["processors"][1]["single"].remove(SEQ_A))),
    (-2, "not in special_tokens", mut(B, lambda j: j["post_processor"]["processors"][1]["single"].append(st("<|zz|>")))),
    (-3, "beyond the vocabulary", mut(B, set_(["post_processor", "processors", 1, "special_tokens", "<|end|>", "ids"], [5000]))),
    (-3, "two that add ids", mut(B, lambda j: j["post_processor"]["processors"].append(j["post_processor"]["processors"][1]))),
    (-2, "WordPiece model fields", mut(B, set_(["model", "type"], "WordPiece"))),   # hf refuses it too (a BPE model's fields)
    (-3, "continuing_subword_prefix", mut(B, set_(["model", "continuing_subword_prefix"], "##"))),
    (-3, "end_of_word_suffix", mut(B, set_(["model", "end_of_word_suffix"], "</w>"))),
    (-3, "dropout", mut(B, set_(["model", "dropout"], 0.1))),
    (-3, "not dense", mut(B, lambda j: j["model"]["vocab"].update({"hello": 300}))),
    (-9, "TOKS_MAX_ADDED_BYTES", mut(B, lambda j: j["added_tokens"].append(added("<" + "x" * 300 + ">", 400)))),
    (-2, "added_tokens entry fields", mut(B, set_(["added_tokens", 0, "special"], DEL))),
    (-2, "ByteLevel fields", mut(A, set_(["pre_tokenizer", "add_prefix_space"], DEL))),
    # the breadth edges (docs/breadth.md): what stays refused next to the spellings it accepts
    (-3, "padding pad_id beyond the vocabulary", mut(B, set_(["padding"], dict(PAD, strategy={"Fixed": 16}, pad_id=99999)))),
    (-3, "pad_to_multiple_of above 2^20", mut(B, set_(["padding"], dict(PAD, pad_to_multiple_of=2097152)))),
    (-2, "padding fields", mut(B, set_(["padding"], {k: v for k, v in PAD.items() if k != "pad_token"}))),
    (-2, "model dropout", mut(B, set_(["model", "dropout"], "0.0"))),
    (-3, "post_processor type", mut(B, set_(["post_processor"], {"type": "RobertaProcessing", "sep": ["</s>", 2]}))),
    (-3, "Sequence inside a Sequence", mut(B, lambda j: j["post_processor"]["processors"].append({"type": "Sequence", "processors": []}))),
    (-3, "all-whitespace lstrip", mut(B, lambda j: j["added_tokens"].extend([added("<|r|>", 300, rstrip=True), added("  ", 301, special=False, lstrip=True)]))),
    # NFC + a normalized:true token whose content NFC changes (hf matches its NFC form)
    (-3, "whose content the normalizer changes", mut(B, lambda j: (j.update(normalizer={"type": "NFC"}),
                                                         j["added_tokens"].append(added("e\u0301x", 300, special=False, normalized=True))))),
]


def main():
    for name, fn in ACCEPT.items():
        j = fn()
        with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=1)
            f.write("\n")
        with open(os.path.join(HERE, name + ".expect"), "w") as f:
            f.write(expect(name, j))
        print(f"accept {name}: ok (hf agrees)")
    with open(os.path.join(HERE, "refuse.txt"), "w", encoding="utf-8") as f:
        for code, what, j in REFUSE:
            s = json.dumps(j, ensure_ascii=False, separators=(",", ":"))
            try:
                Tokenizer.from_str(s)
                hf = "hf loads it"
            except BaseException as e:  # noqa: BLE001 - hf's own refusal (pyo3 panics are BaseException)
                hf = "hf refuses: " + str(e).splitlines()[0][:60]
            f.write(f"{code} {what}\n{s}\n")
            print(f"refuse {code} {what!r}: {hf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
