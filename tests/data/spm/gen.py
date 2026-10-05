#!/usr/bin/env python3
"""tests/data/spm/gen.py: the hand fixtures of tests/c/test_spm.c, with expectations taken from hf
tokenizers 0.23.2 itself (and checked against tests/model/spm_model.py on the way).

    uv run --with tokenizers==0.23.2 tests/data/spm/gen.py        # rewrites tests/data/spm/*.json + *.expect

Every fixture is a small tokenizer.json built to pin one rule of docs/algorithms/spm_bpe.md. <name>.expect has
one case per line (hex = utf-8 bytes as hex, '-' for empty):
    S <at_start 0|1> <text hex> <ids...>      a gap through the text model + model (hf: added_tokens [], no pp)
    P <piece hex> <ids...>                    the model alone (hf Tokenizer.model.tokenize)
    D <ids...> | <bytes hex>                  decode with model-vocab ids (hf decode, skip_special_tokens off)
    K <ids...> | <bytes hex>                  decode, skip_special_tokens on (fixtures with added tokens: their ids
                                              too, both K and D)
    A <how> <text hex> <ids...>               fixtures with added tokens, the file as is: how 0 = hf encode(text)
                                              (toks mode ALL), 1 = encode_special_tokens, no pp (mode NONSPECIAL)
    Q <text hex> <ends...>                    the pre-tokenizer's pieces as byte end offsets (fixtures without a
                                              normalizer: hf pre_tokenize_str on the text itself)
    E <text hex>                              hf fails on this gap (the unk token missing from the vocab)
refuse.txt lists fixtures toks must refuse, with the name of the feature.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "tests", "model"))
import spm_model as S  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

R = "\u2581"
BYTES = ["<0x%02X>" % b for b in range(256)]


def tok(vocab_list, merges, *, norm=None, pre=None, dec=None, unk="<unk>", fuse=True, bf=True, im=False,
        vocab_ids=None):
    vocab = {}
    for s in vocab_list:
        if s not in vocab:
            vocab[s] = len(vocab)
    for a, b in merges:
        for s in (a, b, a + b):
            if s not in vocab:
                vocab[s] = len(vocab)
    if vocab_ids:
        vocab = {s: vocab_ids(i) for s, i in vocab.items()}
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": norm,
            "pre_tokenizer": pre, "post_processor": None, "decoder": dec,
            "model": {"type": "BPE", "dropout": None, "unk_token": unk, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": fuse, "byte_fallback": bf, "ignore_merges": im,
                      "vocab": vocab, "merges": [[a, b] for a, b in merges]}}


REPL = {"type": "Replace", "pattern": {"String": " "}, "content": R}
PREP = {"type": "Prepend", "prepend": R}
SPLIT_NOOP = {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False}
D_GEMMA = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": R}, "content": " "},
                                            {"type": "ByteFallback"}, {"type": "Fuse"}]}
D_LLAMA = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": R}, "content": " "},
                                            {"type": "ByteFallback"}, {"type": "Fuse"},
                                            {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}


def meta(scheme, split):
    return {"type": "Metaspace", "replacement": R, "prepend_scheme": scheme, "split": split}


def added(content, tid, special, normalized):
    return {"id": tid, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": normalized, "special": special}


CHARS = list("abcdehlorstw") + [R, "<", ">", "/", "é", "\n"]
WORDS_MERGES = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), (R, "w"), ("o", "r"), (R + "w", "or"),
                (R + "wor", "l"), (R + "worl", "d"), (R, "h"), (R + "h", "ello"), ("e", "l"), ("l", "o"),
                (R, R), (">", R), (">" + R, "<"), ("a", "b"), ("b", "c"), ("ab", "c")]
WORDS_MERGES = [(a, b) for a, b in WORDS_MERGES]
SPELL = ["he", "ll", "hell", "hello", R + "w", "or", R + "wor", R + "worl", R + "world", R + "h", R + "hello",
         "el", "lo", R + R, ">" + R, ">" + R + "<", "ab", "bc", "abc"]


def fixtures():
    F = {}
    base = ["<unk>"] + BYTES + CHARS
    # gemma 4 shape: Replace only, the no-op Split, decoder without Strip, all byte tokens
    F["gemma4like"] = tok(base, WORDS_MERGES + [("he", "llo")],
                          norm=REPL, pre=SPLIT_NOOP, dec=D_GEMMA)
    # llama 2 legacy: Prepend + Replace, no pre-tokenizer, Strip decoder
    F["llamalike"] = tok(base, WORDS_MERGES, norm={"type": "Sequence", "normalizers": [PREP, REPL]}, dec=D_LLAMA)
    # mistral: Metaspace first, no split
    F["mistrallike"] = tok(base, WORDS_MERGES, pre=meta("first", False), dec=D_LLAMA)
    F["meta_always_split"] = tok(base, WORDS_MERGES, pre=meta("always", True),
                                 dec={"type": "Metaspace", "replacement": R, "prepend_scheme": "always", "split": True})
    F["meta_never_split"] = tok(base, WORDS_MERGES, pre=meta("never", True),
                                dec={"type": "Metaspace", "replacement": R, "prepend_scheme": "never", "split": True})
    # byte fallback with holes: no <0xC3> (é needs C3 A9), no <0x0A>: unk, fused / not fused
    holes = ["<unk>"] + [b for i, b in enumerate(BYTES) if i not in (0xC3, 0x0A, 0xF0)] + list("abcxyz") + [R]
    F["unk_fused"] = tok(holes, [("a", "b"), ("<unk>", "<unk>")], norm=REPL, dec=D_GEMMA)
    F["unk_unfused"] = tok(holes, [("a", "b")], norm=REPL, dec=D_GEMMA, fuse=False)
    F["no_unk_no_bf"] = tok(list("abcxyz") + [R], [("a", "b")], norm=REPL, dec=D_GEMMA, unk=None, bf=False)
    # merge order: competing merges, ties to the leftmost, a duplicate pair keeps its last rank
    F["merge_order"] = tok(["<unk>"] + BYTES + list("abc") + [R], [("b", "c"), ("a", "b"), ("ab", "c"), ("a", "a"),
                                                             ("c", "c"), ("b", "c")], norm=REPL, dec=D_GEMMA)
    # doc §5.7: "c ab" ranks before "a b": the pair premerge must see the c before "ab" as a risk ((x, t), prio <= R)
    F["pm_order"] = tok(["<unk>"] + BYTES + list("abcd") + [R], [("c", "ab"), ("d", "c"), ("a", "b")], norm=REPL,
                        dec=D_GEMMA)
    # ignore_merges: the identity text model only (doc §6.4: the whole-piece lookup reads the raw bytes)
    F["ignore_merges"] = tok(["<unk>"] + BYTES + list("abc") + ["acb"], [("a", "b"), ("ab", "c")], dec=D_GEMMA, im=True)
    # id holes, no decoder (hf joins with " ")
    F["holes_nodec"] = tok(["<unk>"] + BYTES + list("abc") + [R], [("a", "b")], norm=REPL, dec=None,
                           vocab_ids=lambda i: 2 * i + 1)
    # doc §6.4: a literal space in the vocab is unreachable after Replace: its merges are dropped, same ids
    F["space_in_vocab"] = tok(base + [" ", " a", "a "], WORDS_MERGES + [(" ", "a"), ("a", " "), (R, "a")],
                              norm=REPL, pre=SPLIT_NOOP, dec=D_GEMMA)
    # doc §6.1: substitutions compose; x -> " " makes the space reachable again (its merges stay)
    F["replace_chain"] = tok(base + [" ", " a"], WORDS_MERGES + [(" ", "a")],
                             norm={"type": "Sequence", "normalizers": [REPL, {"type": "Replace", "pattern": {"String": "x"},
                                                                              "content": " "}]}, dec=D_GEMMA)
    # doc §8.1: llama 2's specials, normalized=true under Prepend + Replace: each one's string is "▁" + content, no
    # special's content, so skip_special keeps it ([<s>, ▁hello] -> "<s> hello" both ways); <pad> (normalized=false)
    # is dropped, and so is </s>: its string "▁</s>" is the content of the special "▁</s>" (by string, not by id)
    sp = tok(["<unk>", "<s>", "</s>"] + BYTES + CHARS, WORDS_MERGES, norm={"type": "Sequence", "normalizers": [PREP, REPL]},
             dec=D_LLAMA)
    nv = len(sp["model"]["vocab"])
    sp["added_tokens"] = [added("<unk>", 0, True, True), added("<s>", 1, True, True), added("</s>", 2, True, True),
                          added("<pad>", nv, True, False), added(R + "</s>", nv + 1, True, False)]
    F["norm_specials"] = sp
    # doc §8.1, ids two tokens share: with id holes (ids 0, 2, 4, ...) hf numbers a new token from vocab.len() (39, a
    # hole), so every second new token lands on a vocab id. The id prints its token, never the vocab string: the last
    # listed holds it ("hell" after <D>, <F> after "hello"), except that a stored form (normalized=true under Prepend:
    # ▁ + content) outlives a later token without one (<D>'s ▁<D> over "hell"), and the longer form listed first
    # (▁<JJJJJJJJJJ>, then "or"'s ▁or) must not spill into the ids after it (▁wor, ▁worl)
    ho = tok(["<unk>"] + CHARS, WORDS_MERGES, norm={"type": "Sequence", "normalizers": [PREP, REPL]}, dec=D_GEMMA,
             bf=False, vocab_ids=lambda i: 2 * i)
    hv, nxt = ho["model"]["vocab"], len(ho["model"]["vocab"])
    for c, special, normalized in [("<A>", 1, 1), ("<B>", 1, 0), ("<C>", 0, 0), ("<D>", 1, 1), ("hell", 0, 0),
                                   ("<E>", 1, 0), ("hello", 1, 0), ("<F>", 0, 0), ("<G>", 0, 1), ("<H>", 0, 0),
                                   (R + "w", 1, 0), ("<I>", 0, 0), ("<JJJJJJJJJJ>", 0, 1), ("or", 1, 1)]:
        tid, nxt = (hv[c], nxt) if c in hv else (nxt, nxt + 1)
        ho["added_tokens"].append(added(c, tid, bool(special), bool(normalized)))
    F["holes_added"] = ho
    # the bound's token terms (tests/c/test_bound.c, compile.c toks_bound_terms_of): a one-byte added token whose match
    # gives the unit after it the prefix ▁ makes "#a" 3 ids for 2 bytes ("#", "▁", "a": no ▁a merge), past r = 1.
    # Phase 0 (normalized=false) under Prepend, every gap prefixed: r = (1 + p0) / l0 = 2; phase 1 (normalized=true)
    # under Metaspace always, every unit prefixed: r = (1 + p1) / l1 = 2
    b0 = tok(base, WORDS_MERGES, norm={"type": "Sequence", "normalizers": [PREP, REPL]}, dec=D_LLAMA)
    b0["added_tokens"] = [added("#", len(b0["model"]["vocab"]), False, False)]
    F["bound_phase0"] = b0
    b1 = tok(base, WORDS_MERGES, pre=meta("always", True),
             dec={"type": "Metaspace", "replacement": R, "prepend_scheme": "always", "split": True})
    b1["added_tokens"] = [added("#", len(b1["model"]["vocab"]), False, True)]
    F["bound_phase1"] = b1
    return F


REFUSE = {
    "refuse_regex_replace": ("normalizer Replace Regex", lambda t: t.update(
        normalizer={"type": "Replace", "pattern": {"Regex": " +"}, "content": R})),
    "refuse_cutting_split": ("pre_tokenizer Split (one that can cut)", lambda t: t.update(
        normalizer=None, pre_tokenizer=SPLIT_NOOP)),
    "refuse_bytelevel": ("normalizer Replace", lambda t: t.update(
        pre_tokenizer={"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True})),
    "refuse_strip_stop": ("decoder Strip stop > 0", lambda t: t.update(
        decoder={"type": "Strip", "content": " ", "start": 0, "stop": 1})),
    "refuse_first_after_norm": ("Metaspace prepend_scheme first after a normalizer", lambda t: t.update(
        normalizer=REPL, pre_tokenizer=meta("first", False))),
    "refuse_nfc": ("normalizer NFC", lambda t: t.update(
        normalizer={"type": "NFC"})),
    "refuse_unk_missing": ("model unk_token missing from the vocab (hf fails on a char without a token)",
                           lambda t: t["model"].update(unk_token="<nope>", byte_fallback=False)),
    "refuse_im_fold": ("ignore_merges with a Replace / Prepend / Metaspace text model",
                       lambda t: t["model"].update(ignore_merges=True)),
    "refuse_replace_multi": ("normalizer Replace other than one char -> one char (not absorbed by the tables)",
                             lambda t: t.update(normalizer={"type": "Replace", "pattern": {"String": "  "},
                                                            "content": R}, pre_tokenizer=None)),
    "refuse_prepend_two": ("normalizer Prepend of more than one char (the tables hold a one-symbol prefix)",
                           lambda t: t.update(normalizer={"type": "Prepend", "prepend": R + R}, pre_tokenizer=None)),
    "refuse_split_after_prepend": ("Metaspace split after a Prepend prefix", lambda t: t.update(
        normalizer=PREP, pre_tokenizer=meta("never", True))),
    "refuse_dec_order": ("decoder order (sentencepiece-style bpe: one Replace or Metaspace, ByteFallback, Fuse, Strip)",
                         lambda t: t.update(decoder={"type": "Sequence", "decoders": [
                             {"type": "ByteFallback"}, {"type": "Replace", "pattern": {"String": R}, "content": " "},
                             {"type": "Fuse"}]})),
    "refuse_norm_added": ("added token normalized=true under a normalizer (phase 1 on the folded text: m2)",
                          lambda t: t.update(added_tokens=[{"id": 0, "content": "<x y>", "single_word": False,
                                                            "lstrip": False, "rstrip": False, "normalized": True,
                                                            "special": False}])),   # a space: refused (config.c)
    "refuse_image_not_vocab": ("text model substitutes a char that is not a vocab char", lambda t: t.update(
        normalizer={"type": "Replace", "pattern": {"String": " "}, "content": "\u2582"})),
    # fuzz finding 1: an id far past the entries needs id-indexed arrays the parse arena (32 B a source byte) lacks
    "refuse_id_holes": ("model.vocab: id holes past what the file's size gives the parse arena (32 B a source byte)",
                        lambda t: t["model"]["vocab"].update(zz=2_000_000), -9),
}

TEXTS = ["", "a", " ", "  ", "hello", "hello world", " hello world", "hello  world ", "  hello   world  ",
         R + "x", "a" + R + R + "b", "abc", "aabcc", "abcabc", "bcab", "é", "é" + "x", "xéy", "😀", "a😀b",
         "\n", "a\nb", "ab\n\nab", "x\ny\n", "<" + R + ">", "> <", ">" + R + "</", "a>" + " " + "<b", "hello>  <world",
         "acb", "acbacb", "zz", "ééé", "😀😀", "a é 😀 \n b", "world hello", "w o r l d", " " * 10 + "abc", "dcab",
         "dcabab", "cab ab"]


def substituted(tj):
    """the chars the text model substitutes (Replace patterns, Metaspace's space): toks' char table reads each as
    its image (doc §6), so the model-alone cases (P lines) leave them out"""
    out = set()
    n = tj.get("normalizer")
    steps = [n] if n and n.get("type") != "Sequence" else (n or {}).get("normalizers", [])
    for st in steps:
        if st.get("type") == "Replace":
            out.add(st["pattern"]["String"])
    pre = tj.get("pre_tokenizer")
    pres = [pre] if pre and pre.get("type") != "Sequence" else (pre or {}).get("pretokenizers", [])
    if any(x.get("type") == "Metaspace" for x in pres):
        out.add(" ")
    return out


def esc(b):
    return b.hex() if b else "-"


def main():
    F = fixtures()
    for name, tj in F.items():
        raw = json.dumps(tj, ensure_ascii=False)
        with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as f:
            f.write(raw)
        hf = Tokenizer.from_str(raw)
        hf0 = Tokenizer.from_str(json.dumps(dict(tj, added_tokens=[]), ensure_ascii=False)) if tj["added_tokens"] else hf
        model = S.SpmTokenizer(tj)
        lines = []
        for t in TEXTS:
            for at_start in (1, 0) if tj["pre_tokenizer"] and tj["pre_tokenizer"].get("prepend_scheme") == "first" else (1,):
                text = t if at_start else t                    # at_start 0: the gap is not at offset 0
                if at_start == 0:
                    if not t:
                        continue
                    full = "<s>" + t                            # an added token in front moves the gap off 0
                    tj2 = dict(tj, added_tokens=[{"id": 0, "content": "<s>", "single_word": False, "lstrip": False,
                                                  "rstrip": False, "normalized": False, "special": True}])
                    h = Tokenizer.from_str(json.dumps(tj2, ensure_ascii=False))
                    try:
                        ids = h.encode(full, add_special_tokens=False).ids[1:]
                    except BaseException:  # noqa: BLE001
                        lines.append(f"E {esc(t.encode())}")
                        continue
                else:
                    try:
                        ids = hf0.encode(text, add_special_tokens=False).ids
                    except BaseException:  # noqa: BLE001
                        lines.append(f"E {esc(t.encode())}")
                        continue
                    assert model.encode(text, "NONE", False) == ids, (name, text)
                lines.append(f"S {at_start} {esc(t.encode())} {' '.join(map(str, ids))}".rstrip())
        subst = substituted(tj)
        for t in TEXTS:
            p = t.replace(" ", R)
            if not p or any(c in subst for c in p):         # the folded table reads a substituted char as its image
                continue
            try:
                ids = [x.id for x in hf.model.tokenize(p)]
            except BaseException:  # noqa: BLE001
                continue
            assert model.model.tokenize(p) == ids, (name, p)
            lines.append(f"P {esc(p.encode())} {' '.join(map(str, ids))}".rstrip())
        if not tj["normalizer"]:                          # pieces: hf's pre-tokenizer offsets (no normalizer)
            for t in TEXTS:
                if not t:
                    continue
                if tj["pre_tokenizer"]:
                    ends = [e for _, (b, e) in hf.pre_tokenizer.pre_tokenize_str(t)]
                else:
                    ends = [len(t)]
                ends = [len(t[:e].encode()) for e in ends]      # char offsets -> byte offsets
                lines.append(f"Q {esc(t.encode())} {' '.join(map(str, ends))}")
        v = tj["model"]["vocab"]
        seqs = [[v[k] for k in ["a", "b"] if k in v], [v[k] for k in ["<0xE2>", "<0x82>", "<0xAC>"] if k in v],
                [v[k] for k in ["<0xE2>", "<0x41>", R] if k in v], [v[k] for k in [R, "a", R, R, "b"] if k in v],
                [v[k] for k in ["<0xFF>", "a", "<0xC3>", "<0xA9>"] if k in v], [], list(v.values())[:40]]
        for ids in seqs:
            out = hf.decode(ids, skip_special_tokens=False)
            assert model.decode(ids, False) == out, (name, ids)
            lines.append(f"D {' '.join(map(str, ids))} | {esc(out.encode())}")
        if tj["added_tokens"]:                              # doc §8.1: the added ids, alone, before a word, inside a
            ai = sorted({hf.token_to_id(d["content"]) for d in tj["added_tokens"]})   # byte run, all together
            w = v[R + "hello"]
            run = [v[k] for k in ["<0xE2>", "<0x82>", "<0xAC>"] if k in v]
            dec = [[i] for i in ai] + [[i, w] for i in ai] + [[run[0], i] + run[1:] for i in ai if len(run) == 3]
            dec += [ai, ai + seqs[-1]]
            if max(v.values()) >= len(v):                   # id holes: every other id near the added ones, alone
                dec += [[i] for i in range(len(v) - 1, max(ai) + 6) if i not in ai]
            for ids in dec:
                for skip in (False, True):
                    out = hf.decode(ids, skip_special_tokens=skip)
                    assert model.decode(ids, skip) == out, (name, ids, skip)
                    lines.append(f"{'K' if skip else 'D'} {' '.join(map(str, ids))} | {esc(out.encode())}")
            # encode of the file as is: each content alone, between words, twice; hf matches one token an id (§2)
            cs = [d["content"] for d in tj["added_tokens"]]
            texts = [t for c in cs for t in (c, f"a {c} b", c + c)] + [" ".join(cs), "<H> <F> <B>", "hello hell"]
            if name.startswith("bound_"):                   # the runs the bound's token term pays for
                texts += ["#a" * 8, "#a#b #", "a#" * 5 + " ", "##a##", "# a #", "#" + R + "#", "#\n#é#"]
            for t in texts:
                for how in (0, 1):
                    hf.encode_special_tokens = how == 1
                    ids = hf.encode(t, add_special_tokens=how == 0).ids
                    assert model.encode(t, "NONSPECIAL" if how else "ALL", how == 0) == ids, (name, t, how)
                    lines.append(f"A {how} {esc(t.encode())} {' '.join(map(str, ids))}".rstrip())
            hf.encode_special_tokens = False
        with open(os.path.join(HERE, name + ".expect"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} cases")
    base = F["gemma4like"]
    with open(os.path.join(HERE, "refuse.txt"), "w") as f:
        for name, (why, mut, *code) in REFUSE.items():     # code: TOKS_E_UNSUPPORTED unless given
            tj = json.loads(json.dumps(base))
            mut(tj)
            raw = json.dumps(tj, ensure_ascii=False)
            with open(os.path.join(HERE, name + ".json"), "w", encoding="utf-8") as g:
                g.write(raw)
            f.write(f"{name}.json\t{why}" + (f"\t{code[0]}" if code else "") + "\n")
    print(f"refuse: {len(REFUSE)} files")


if __name__ == "__main__":
    main()
