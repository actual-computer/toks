#!/usr/bin/env python3
"""tests/model/run_wordpiece_fuzz.py: wordpiece_model.py (docs/algorithms/wordpiece.md) vs hf tokenizers 0.23.2.

For every pinned WordPiece tokenizer (tests/wordpiece/pins.json, fetched by tests/wordpiece/fetch.py), generated and
real texts go through both sides and every result is compared:

  encode   the file as shipped: mode ALL / NONSPECIAL / NONE (SPEC §3.2: hf default / encode_special_tokens=True /
           the file with added_tokens: []) x post-processing on / off (add_special_tokens)      6 checks per text
  variant  the same file with one random variant from a pool built per worker: truncation (max_length 0..600,
           Left / Right, LongestFirst / OnlyFirst / OnlySecond, stride 0..4), padding (Fixed / BatchLongest,
           pad_to_multiple_of, Left / Right, pad_id), BertNormalizer flags (all 24), the other normalizers (NFD,
           Lowercase, StripAccents, Sequence), pre-tokenizers (Bert, WhitespaceSplit, Whitespace, Sequence),
           max_input_chars_per_word, continuing_subword_prefix, a missing unk, the post-processor (Template, Bert,
           Roberta, none), added-token options (lstrip, rstrip, single_word, normalized)        4 checks per text
  decode   the ids of the ALL + pp encode and a random id sequence, skip_special_tokens on / off  4 checks per text
  model    hf Tokenizer.model.tokenize(word) vs WordPiece.tokenize on the text's words           1 check per word

An input on which hf raises must make the model raise the same kind (Exception / PanicException), and the reverse;
both count as agreement.

    uv run --with tokenizers==0.23.2 tests/model/run_wordpiece_fuzz.py --n 300 --workers 2 minilm-l6     # smoke
    tools/remote.sh <host> 'uv run --with tokenizers==0.23.2 tests/model/run_wordpiece_fuzz.py --n 200000 \
        --workers 8 --text build/text/flores.txt'

Output: one RESULT line per tokenizer with its counts, every mismatch with its repro (escaped), and --out (json).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tests", "wordpiece"))

import wordpiece_model as WM  # noqa: E402
import fetch as F  # noqa: E402

MODES = ("ALL", "NONSPECIAL", "NONE")
W = {}


def esc(s):
    return s.encode("unicode_escape").decode("ascii")


# ------------------------------------------------------------------------------------------------- text pools
def _r(a, b):
    return [chr(c) for c in range(a, b + 1) if not (0xD800 <= c <= 0xDFFF)]


POOLS = {
    "ascii_word": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    "ascii_punct": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
    "ws": [" ", " ", " ", "\t", "\n", "\r", "\x0b", "\x0c", "\x85", "\xa0", "\u1680", "\u2000", "\u2002", "\u2007",
           "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\u180e", "\u200b", "\ufeff", "\x1c", "\x1f"],
    "control": ["\x00", "\x01", "\x07", "\x08", "\x0e", "\x1b", "\x7f", "\x80", "\x9f", "\xad", "\u0600", "\u061c",
                "\u200c", "\u200d", "\u200e", "\u2060", "\u2066", "\ufff9", "\ufffd", "\U000110bd", "\U0001d173",
                "\U000e0001", "\U000e0041", "\ue000", "\uf8ff", "\U000f0000", "\U0010fffd", "\U0010ffff", "\uffff",
                "\ufdd0", "\u0378", "\U0001fffe"],
    "latin": _r(0xC0, 0x24F) + ["\u0130", "\u0131", "\u017f", "\u1e9e", "\ufb01", "\ufb06", "\u212a", "\u212b",
                                "\u2126", "\u1e9b", "\u01c5", "\u01c4", "\u1f88", "\u0149"],
    "combining": _r(0x300, 0x36F) + ["\u0345", "\u0483", "\u0488", "\u0489", "\u05b0", "\u05c1", "\u064b", "\u0670",
                                     "\u093c", "\u094d", "\u0e31", "\u0f71", "\u0f72", "\u0f74", "\u0f80", "\u1ab0",
                                     "\u1dc0", "\u20d0", "\u20dd", "\u302a", "\ufe20", "\ufe0f", "\ufe0e",
                                     "\U0001d167", "\U000e0100"],
    "keepns": [chr(c) for c in [0x8D4, 0x8D9, 0x8E1, 0x1B44, 0x1BAA, 0x1BF2, 0x1DFB, 0x302E, 0x302F, 0xA953, 0xA9C0,
                                0x111C0, 0x11235, 0x1134D, 0x11442, 0x11446, 0x116B6, 0x11C3F, 0x1D165, 0x1D166,
                                0x1D16D, 0x1D16E, 0x1D172, 0x1E000, 0x1E00F, 0x1E02A, 0x1E944, 0x1E94A]],
    "spacing_marks": ["\u0903", "\u093e", "\u09be", "\u0bbe", "\u0f3e", "\u0f7f", "\u1b35", "\U0001d16e"],
    "greek_cyr": _r(0x370, 0x3FF) + _r(0x400, 0x4FF) + _r(0x1F00, 0x1FFF) + ["\u03a3", "\u03c2"],
    "cased_rare": _r(0x10A0, 0x10FF) + _r(0x1C90, 0x1CBF) + _r(0x13A0, 0x13F5) + _r(0xAB70, 0xABBF) +
    _r(0x10400, 0x1044F) + _r(0x104B0, 0x104FB) + _r(0x10570, 0x105BC) + _r(0x1E900, 0x1E95F) + _r(0x2C00, 0x2C5F) +
    _r(0xA640, 0xA69F) + _r(0xA720, 0xA7FF) + _r(0x16E40, 0x16E9A) + _r(0x16EA0, 0x16EDF) + _r(0x10C80, 0x10CFF) +
    _r(0x24B6, 0x24E9) + _r(0x2160, 0x2188) + _r(0xFF21, 0xFF5A),
    "cjk": _r(0x4E00, 0x4E3F) + _r(0x9F80, 0x9FFF) + _r(0x3400, 0x3420) + _r(0x4DB0, 0x4DBF) + _r(0xF900, 0xFAFF) +
    _r(0x20000, 0x20010) + _r(0x2A6D0, 0x2A6E0) + _r(0x2A700, 0x2A710) + _r(0x2B730, 0x2B745) + _r(0x2B810, 0x2B825) +
    _r(0x2B915, 0x2B925) + _r(0x2CEA0, 0x2CEB5) + _r(0x2F800, 0x2F810) + _r(0x2FA10, 0x2FA20) + _r(0x30000, 0x30008) +
    _r(0x31350, 0x31358) + _r(0x2F00, 0x2F10) + _r(0x2E80, 0x2E90) + _r(0x3000, 0x303F) + _r(0xFF01, 0xFF20),
    "kana_hangul": _r(0x3041, 0x30FF) + _r(0xFF65, 0xFF9F) + _r(0xAC00, 0xAC40) + _r(0xD780, 0xD7A3) +
    _r(0x1100, 0x1112) + _r(0x1161, 0x1175) + _r(0x11A8, 0x11C2) + _r(0x3131, 0x318E),
    "scripts": _r(0x5D0, 0x5EA) + _r(0x620, 0x64A) + _r(0x660, 0x669) + _r(0x905, 0x939) + _r(0x966, 0x96F) +
    _r(0xE01, 0xE2E) + _r(0xE47, 0xE4E) + _r(0x1000, 0x1021) + _r(0x1780, 0x17A2) + _r(0xF40, 0xF6C) +
    _r(0x0F73, 0x0F81) + _r(0xFB1D, 0xFB4F) + _r(0x1200, 0x1248) + _r(0x13000, 0x13010),
    "symbols": ["\u20ac", "\xa3", "\xa5", "\u20bf", "\u2211", "\u221e", "\xb0", "\xa7", "\xb6", "\xab", "\xbb",
                "\u201c", "\u201d", "\u2018", "\u2019", "\u2010", "\u2013", "\u2014", "\u203f", "\u2039", "\u203a",
                "\xbf", "\xa1", "\u30fb", "\u3001", "\u3002", "\u2e42", "\u2e43", "\u2e44", "\u2e49", "\u2e4e",
                "\u2e52", "\u2e5d", "\U0001e95e", "\U0001e95f", "\u104a", "\u0964", "\u060c", "\u061f", "\u05be",
                "\u00b7", "\u0387", "\u037e", "\u1fef", "\u1fed", "\u1fee", "\u2024", "\u2026", "\u2047", "\u00b4",
                "\u02dc", "\u2122", "\u00a9", "\u00ae", "\u2030", "\u2032", "\u2116", "\u2030", "\u00bd", "\u00b2",
                "\u2460", "\u2153", "\u0f3a", "\u0f3b", "\U00011c41", "\U00016af5", "\U0001da87", "\U00011a9e"],
    "emoji": ["\U0001f600", "\U0001f44d", "\U0001f3fd", "\U0001f468", "\u200d", "\U0001f469", "\U0001f467",
              "\U0001f1fa", "\U0001f1f8", "\u2764", "\ufe0f", "\u20e3", "1", "\U0001f9d1", "\U0001fac3", "\u2b50",
              "\U000e0067", "\U000e007f", "\U0001f3f4"],
}
SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]", "<s>", "</s>", "<mask>", "<pad>", "<unk>",
                    "[mask]", "[cls]", "<MASK>", "[MASK", "MASK]", "[ MASK ]", "<mask >", "[UNK]]"]
SEPS = ["", " ", " ", " ", "  ", "\n", "\t", ",", ".", " - ", "\u3000", "\xa0", "'", "\u200b", "#", "##"]


def gen_text(rng, words):
    """a random text: tokens from the pools and the vocabulary, joined by random separators."""
    n = rng.choice((1, 2, 3, 5, 8, 13, 21, 40))
    parts = []
    for _ in range(n):
        k = rng.random()
        if k < 0.22:                                   # vocabulary words and joins of vocab pieces
            w = rng.choice(words)
            for _ in range(rng.choice((0, 0, 1, 2, 3))):
                x = rng.choice(words)
                w += x[2:] if x.startswith("##") and rng.random() < 0.8 else x
            if rng.random() < 0.3:
                w = w.upper() if rng.random() < 0.5 else w.capitalize()
            parts.append(w)
        elif k < 0.30:
            parts.append(rng.choice(SPECIAL_LITERALS))
        elif k < 0.33:                                  # long words around max_input_chars_per_word
            c = rng.choice(POOLS[rng.choice(("ascii_word", "latin", "cjk", "greek_cyr"))])
            ln = rng.choice((99, 100, 101, 150, 200, rng.randint(1, 300)))
            if rng.random() < 0.5:
                parts.append(c * ln)
            else:
                parts.append("".join(rng.choice(POOLS["latin"] + POOLS["combining"]) for _ in range(ln)))
        else:
            pool = POOLS[rng.choice(list(POOLS))]
            ln = rng.choice((1, 1, 2, 3, 4, 6, 9, 15))
            mix = rng.random() < 0.3
            s = []
            for _ in range(ln):
                p = POOLS[rng.choice(list(POOLS))] if mix else pool
                s.append(rng.choice(p))
                if rng.random() < 0.08:
                    s.append(rng.choice(POOLS["combining"] + POOLS["keepns"]))
            parts.append("".join(s))
    out = []
    for p in parts:
        out.append(p)
        out.append(rng.choice(SEPS))
    t = "".join(out)
    if rng.random() < 0.1:
        t = rng.choice(POOLS["ws"]) * rng.randint(1, 4) + t
    return t


# ---------------------------------------------------------------------------------------------- the variants
def variants(tj, rng, n):
    """n random variants of the tokenizer.json dict tj (each a full config both sides load)."""
    out = []
    vocab = tj["model"]["vocab"]
    added = tj.get("added_tokens") or []
    for _ in range(n):
        d = json.loads(json.dumps(tj))
        kinds = rng.sample(["trunc", "pad", "norm", "pre", "model", "pp", "added"], rng.choice((1, 1, 2, 3)))
        desc = []
        if "trunc" in kinds:
            ml = rng.choice((0, 1, 2, 3, 4, 5, 8, 16, 32, 64, 128, rng.randint(0, 600)))
            d["truncation"] = {"direction": rng.choice(("Left", "Right")), "max_length": ml,
                               "strategy": rng.choice(("LongestFirst", "LongestFirst", "OnlyFirst", "OnlySecond")),
                               "stride": rng.choice((0, 0, 0, 1, 2, 4))}
            desc.append("trunc=%s" % json.dumps(d["truncation"]))
        if "pad" in kinds:
            st = rng.choice(("BatchLongest", {"Fixed": rng.choice((0, 1, 5, 16, 64, 128, 512))}))
            d["padding"] = {"strategy": st, "direction": rng.choice(("Left", "Right")),
                            "pad_to_multiple_of": rng.choice((None, None, 0, 1, 3, 8, 64)),
                            "pad_id": rng.choice((0, 1, 7, 99999)), "pad_type_id": 0, "pad_token": "[PAD]"}
            desc.append("pad=%s" % json.dumps(d["padding"]))
        if "norm" in kinds:
            r = rng.random()
            if r < 0.7:
                d["normalizer"] = {"type": "BertNormalizer", "clean_text": rng.random() < 0.5,
                                   "handle_chinese_chars": rng.random() < 0.5,
                                   "strip_accents": rng.choice((None, True, False)), "lowercase": rng.random() < 0.5}
            elif r < 0.8:
                d["normalizer"] = None
            else:
                steps = [rng.choice(({"type": "NFD"}, {"type": "Lowercase"}, {"type": "StripAccents"},
                                     {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
                                      "strip_accents": None, "lowercase": False}))
                         for _ in range(rng.randint(1, 3))]
                d["normalizer"] = {"type": "Sequence", "normalizers": steps}
            desc.append("norm=%s" % json.dumps(d["normalizer"]))
        if "pre" in kinds:
            d["pre_tokenizer"] = rng.choice((
                {"type": "WhitespaceSplit"}, {"type": "Whitespace"}, None,
                {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "BertPreTokenizer"}]},
                {"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "BertPreTokenizer"}]}))
            desc.append("pre=%s" % json.dumps(d["pre_tokenizer"]))
        if "model" in kinds:
            m = d["model"]
            r = rng.random()
            if r < 0.4:
                m["max_input_chars_per_word"] = rng.choice((0, 1, 2, 3, 5, 8, 20))
            elif r < 0.6:
                m["continuing_subword_prefix"] = rng.choice(("", "#", "@@", "##"))
            elif r < 0.8:
                m["unk_token"] = "[NOT-IN-VOCAB]"
            else:
                drop = set(rng.sample(sorted(vocab), min(len(vocab) // 3, 2000)))
                m["vocab"] = {k: v for k, v in vocab.items() if k not in drop or k == m["unk_token"]}
            desc.append("model=%s" % json.dumps({k: v for k, v in m.items() if k != "vocab"}) +
                        (" vocab-%d" % (len(vocab) - len(m["vocab"])) if len(m["vocab"]) != len(vocab) else ""))
        if "pp" in kinds:
            r = rng.random()
            ids = sorted(vocab.values())
            a, b = rng.choice(ids), rng.choice(ids)
            if r < 0.25:
                d["post_processor"] = None
            elif r < 0.5:
                d["post_processor"] = {"type": "BertProcessing", "sep": ["[SEP]", b], "cls": ["[CLS]", a]}
            elif r < 0.75:
                d["post_processor"] = {"type": "RobertaProcessing", "sep": ["</s>", b], "cls": ["<s>", a],
                                       "trim_offsets": True, "add_prefix_space": False}
            else:
                pieces = [{"SpecialToken": {"id": "X", "type_id": 0}}] * rng.randint(0, 2) + \
                    [{"Sequence": {"id": "A", "type_id": 0}}] + [{"SpecialToken": {"id": "Y", "type_id": 0}}] * rng.randint(0, 2)
                d["post_processor"] = {"type": "TemplateProcessing", "single": pieces,
                                       "pair": pieces + [{"Sequence": {"id": "B", "type_id": 1}}],
                                       "special_tokens": {"X": {"id": "X", "ids": [a, b][:rng.randint(1, 2)], "tokens": ["X", "X"][:1]},
                                                          "Y": {"id": "Y", "ids": [b], "tokens": ["Y"]}}}
                d["post_processor"]["special_tokens"]["X"]["tokens"] = ["X"] * len(d["post_processor"]["special_tokens"]["X"]["ids"])
            desc.append("pp=%s" % json.dumps(d["post_processor"]))
        if "added" in kinds and added:
            for t in d["added_tokens"]:
                if rng.random() < 0.5:
                    t["lstrip"] = rng.random() < 0.4
                    t["rstrip"] = rng.random() < 0.4
                    t["single_word"] = rng.random() < 0.3
                    t["normalized"] = rng.random() < 0.4
                    t["special"] = rng.random() < 0.8
            if rng.random() < 0.5:
                d["added_tokens"].append({"id": 0, "content": rng.choice(("hello", "##ing", "ab", "\u4e2d", "a b",
                                                                          "Hello", "\u00e9t\u00e9", " ")),
                                          "single_word": rng.random() < 0.3, "lstrip": rng.random() < 0.3,
                                          "rstrip": rng.random() < 0.3, "normalized": rng.random() < 0.5,
                                          "special": rng.random() < 0.5})
            desc.append("added=%s" % json.dumps([{k: v for k, v in t.items() if k != "id"} for t in d["added_tokens"]],
                                                 ensure_ascii=True))
        out.append((d, "; ".join(desc)))
    return out


# ------------------------------------------------------------------------------------------------- comparison
def run_both(fm, fh):
    """(kind, value): kind 'ok', 'err' (Exception), 'panic' (pyo3 PanicException), 'unsup' (model refuses)."""
    try:
        m = ("ok", fm())
    except WM.Unsupported as e:
        m = ("unsup", str(e))
    except WM.HfPanic as e:
        m = ("panic", str(e))
    except WM.HfError as e:
        m = ("err", str(e))
    try:
        h = ("ok", fh())
    except Exception as e:  # noqa: BLE001
        h = ("err", str(e))
    except BaseException as e:  # noqa: BLE001  (pyo3 panics)
        if isinstance(e, KeyboardInterrupt):
            raise
        h = ("panic", str(e))
    return m, h


def agree(m, h):
    if m[0] == "unsup":
        return True        # counted separately: toks refuses the configuration
    if m[0] != h[0]:
        return False
    return m[0] != "ok" or m[1] == h[1]


def build_pair(raw_dict):
    from tokenizers import Tokenizer
    try:
        model = WM.Tokenizer(raw_dict)
    except WM.Unsupported as e:
        model = ("unsup", str(e))
    except WM.HfError as e:
        model = ("err", str(e))
    try:
        hf = Tokenizer.from_str(json.dumps(raw_dict))
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        hf = ("err", str(e))
    return model, hf


def init(name, path, nvar, seed):
    with open(path, encoding="utf-8") as f:
        tj = json.load(f)
    none = json.loads(json.dumps(tj))
    none["added_tokens"] = []
    W["name"] = name
    W["tj"] = tj
    W["main"] = build_pair(tj)
    W["none"] = build_pair(none)
    for k in ("main", "none"):
        if not isinstance(W[k][0], WM.Tokenizer) or isinstance(W[k][1], tuple):
            raise SystemExit("%s: %s does not load on both sides: %r" % (name, k, W[k]))
    from tokenizers import Tokenizer
    W["hf_nonspecial"] = Tokenizer.from_str(json.dumps(tj))
    W["hf_nonspecial"].encode_special_tokens = True
    rng = random.Random(seed ^ 0x5EED)
    W["variants"] = []
    nvar = max(4, min(nvar, nvar * 40000 // max(1, len(tj["model"]["vocab"]))))   # memory: big vocabs get fewer
    for d, desc in variants(tj, rng, nvar):
        m, h = build_pair(d)
        W["variants"].append((m, h, desc))
    vocab = tj["model"]["vocab"]
    W["words"] = sorted(vocab)
    W["ids"] = sorted(set(vocab.values()))
    W["max_id"] = max(W["ids"] + [t["id"] for t in tj.get("added_tokens") or []])


def hf_res(f):
    try:
        return ("ok", f())
    except Exception as e:  # noqa: BLE001
        return ("err", str(e))
    except BaseException as e:  # noqa: BLE001  (pyo3 panics)
        if isinstance(e, KeyboardInterrupt):
            raise
        return ("panic", str(e))


def work(args):
    seed, texts = args
    rng = random.Random(seed)
    cnt = {"encode": 0, "variant": 0, "decode": 0, "model": 0, "refused": 0, "both_raise": 0, "texts": 0,
           "variant_load_refused": 0}
    bad = []
    mt, ht = W["main"]
    mn, hn = W["none"]
    hs = W["hf_nonspecial"]
    real = [t for t in texts if t is not None]
    for idx, t in enumerate(texts):
        if t is None:
            t = gen_text(rng, W["words"])
        elif rng.random() < 0.25:                      # real lines joined into a longer document
            k = rng.randint(2, 12)
            t = rng.choice((" ", "\n", "  ", "\n\n")).join([t] + [rng.choice(real) for _ in range(k)])
        cnt["texts"] += 1
        enc_ids = None
        for mode in MODES:
            if mode == "NONE":
                mres = mn.encode_pp_both(t, "ALL")
                hfun = hn
            elif mode == "NONSPECIAL":
                mres = mt.encode_pp_both(t, "NONSPECIAL")
                hfun = hs
            else:
                mres = mt.encode_pp_both(t, "ALL")
                hfun = ht
            for pp in (True, False):
                m = mres[pp]
                h = hf_res(lambda: hfun.encode(t, add_special_tokens=pp).ids)
                cnt["encode"] += 1
                if m[0] != "ok" and m[0] == h[0]:
                    cnt["both_raise"] += 1
                if not agree(m, h):
                    bad.append(("encode", mode, pp, "", t, m, h))
                if mode == "ALL" and pp and m[0] == "ok":
                    enc_ids = m[1]
        # a random variant of the same file
        mv, hv, desc = W["variants"][rng.randrange(len(W["variants"]))]
        if isinstance(mv, tuple) or isinstance(hv, tuple):
            cnt["variant_load_refused"] += 1
            if isinstance(mv, tuple) and mv[0] == "unsup":
                pass
            elif (isinstance(mv, tuple)) != (isinstance(hv, tuple)):
                bad.append(("variant-load", "", "", desc, "", mv if isinstance(mv, tuple) else "loads",
                            hv if isinstance(hv, tuple) else "loads"))
        else:
            for mode, pp in (("ALL", True), ("ALL", False), ("NONSPECIAL", True), ("ALL", rng.random() < 0.5)):
                if mode == "NONSPECIAL":
                    hv.encode_special_tokens = True
                m, h = run_both(lambda: mv.encode(t, mode, pp), lambda: hv.encode(t, add_special_tokens=pp).ids)
                if mode == "NONSPECIAL":
                    hv.encode_special_tokens = False
                cnt["variant"] += 1
                if m[0] == "unsup":
                    cnt["refused"] += 1
                elif m[0] != "ok" and m[0] == h[0]:
                    cnt["both_raise"] += 1
                if not agree(m, h):
                    bad.append(("variant", mode, pp, desc, t, m, h))
            if rng.random() < 0.3:
                ids = [rng.randrange(W["max_id"] + 3) for _ in range(rng.randint(0, 12))]
                for skip in (False, True):
                    m, h = run_both(lambda: mv.decode(ids, skip), lambda: hv.decode(ids, skip_special_tokens=skip))
                    cnt["variant"] += 1
                    if not agree(m, h):
                        bad.append(("variant-decode", skip, "", desc, repr(ids), m, h))
        # decode
        seqs = [enc_ids or [], [rng.choice(W["ids"]) if rng.random() < 0.9 else rng.randrange(W["max_id"] + 50)
                                for _ in range(rng.randint(0, 16))]]
        for ids in seqs:
            for skip in (False, True):
                m, h = run_both(lambda: mt.decode(ids, skip), lambda: ht.decode(ids, skip_special_tokens=skip))
                cnt["decode"] += 1
                if not agree(m, h):
                    bad.append(("decode", skip, "", "", repr(ids), m, h))
        # the model alone, on the words the pipeline produced
        try:
            ws = mt.words(t)
        except Exception:  # noqa: BLE001
            ws = []
        for w in ws[:24]:
            m, h = run_both(lambda: mt.model.tokenize(w), lambda: [x.id for x in ht.model.tokenize(w)])
            cnt["model"] += 1
            if not agree(m, h):
                bad.append(("model", "", "", "", w, m, h))
        if len(bad) > 50:
            break
    return cnt, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="generated texts per tokenizer")
    ap.add_argument("--text", action="append", default=[], help="real text file(s): one case per non-empty line")
    ap.add_argument("--max-real", type=int, default=0, help="cap on real lines per tokenizer (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--variants", type=int, default=48, help="variant configs per worker")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    pins = F.pins()
    names = args.names or list(pins)
    real = []
    for p in args.text:
        with open(p, encoding="utf-8", errors="strict") as f:
            real.extend(line.rstrip("\n") for line in f if line.strip())
    if args.max_real:
        real = real[:args.max_real]
    import tokenizers
    report = {"tokenizers_version": tokenizers.__version__, "seed": args.seed, "n_generated": args.n,
              "n_real": len(real), "results": {}}
    total_bad = 0
    for name in names:
        path = F.fetch(name, pins[name])
        t0 = time.time()
        jobs = []
        k = 0
        for i in range(0, args.n, args.chunk):
            jobs.append((args.seed * 1000003 + k, [None] * min(args.chunk, args.n - i)))
            k += 1
        for i in range(0, len(real), args.chunk):
            jobs.append((args.seed * 1000003 + k, real[i:i + args.chunk]))
            k += 1
        tot = {}
        bad = []
        with mp.get_context("fork").Pool(args.workers, initializer=init,
                                        initargs=(name, path, args.variants, args.seed)) as pool:
            for cnt, b in pool.imap_unordered(work, jobs):
                for kk, v in cnt.items():
                    tot[kk] = tot.get(kk, 0) + v
                bad.extend(b)
                if len(bad) > 200:
                    pool.terminate()
                    break
        checks = tot.get("encode", 0) + tot.get("variant", 0) + tot.get("decode", 0) + tot.get("model", 0)
        dt = time.time() - t0
        print("RESULT %s: %d checks (texts %d: encode %d, variant %d, decode %d, model %d; both raise %d, "
              "model refuses %d, variant configs refused %d) mismatches %d  %.0fs" % (
                  name, checks, tot.get("texts", 0), tot.get("encode", 0), tot.get("variant", 0), tot.get("decode", 0),
                  tot.get("model", 0), tot.get("both_raise", 0), tot.get("refused", 0),
                  tot.get("variant_load_refused", 0), len(bad), dt), flush=True)
        for b in bad[:20]:
            kind, mode, pp, desc, text, m, h = b
            print("  MISMATCH %s mode=%s pp=%s %s\n    text=%s\n    model=%s\n    hf   =%s" % (
                kind, mode, pp, desc[:600], esc(str(text))[:600], esc(str(m))[:600], esc(str(h))[:600]), flush=True)
        report["results"][name] = dict(tot, checks=checks, mismatches=len(bad), seconds=round(dt, 1),
                                       sha256=pins[name]["sha256"])
        total_bad += len(bad)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    print("TOTAL mismatches %d" % total_bad)
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
