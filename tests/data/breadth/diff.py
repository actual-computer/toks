#!/usr/bin/env python3
"""tests/data/breadth/diff.py: the breadth differential against hf tokenizers 0.23.2, at scale, on a
lab host (maintainer doctrine: never on a laptop).

    uv run --with tokenizers==0.23.2 python tests/data/breadth/diff.py --driver DRV synth --tokenizers N \
        --texts K --seed S [--shard i/n]
    uv run --with tokenizers==0.23.2 python tests/data/breadth/diff.py --driver DRV real FILE --texts K \
        --seed S [--corpus TXT ...] [--shard i/n] [--strip-normalizer]

synth: N random tokenizers built from gen.py's model, each with a random set of added tokens (contents
from a pool of words, whitespace runs, punctuation and non-ascii, every option drawn independently:
special, normalized, lstrip, rstrip, single_word), a random spelling of the split (Isolated, Isolated +
invert, Removed + invert; cl100k or gpt-2 strings; ByteLevel use_regex), dropout null / 0, padding none /
BatchLongest, and a random post-processor (none, Template, Roberta, the Bert shape, Sequence[ByteLevel,
Roberta]); K random texts each, made of the tokens' contents, their pieces and edge chars. A token set hf
can panic on must be refused by toks (config.c); any other refusal is a failure.
real: K texts for one tokenizer file (random edge texts around its added tokens + lines of the corpora).
Every text runs in modes ALL, NONSPECIAL, NONE (+ ALL without the post-processor); ids must be equal.
DRV is the toks driver (tests/driver/toks_driver.c built against the library). Prints counts, writes the
first diffs, exits 1 on any.
"""
import argparse
import copy
import json
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "parity"))

import gen as G  # noqa: E402
from run import Driver  # noqa: E402
from toks_oracle import oracle as O  # noqa: E402

POOL = ["<mask>", "<s>", "</s>", "<|end|>", "<|x|>", "[W]", "word", "hello", "he", "dé", "é", "汉字", "a", "_",
        "@@", " hi ", "  ", "   ", "\t", "\n", " ", "x y", "<x>", "<y>", "ll", "lo w", "€", "Ĳ", "ĠĠ", "1", "12",
        "'s", "the", "and", " the", "\u3000", "\u00a0x", "\u200d", "\u0301"]
EDGE = ["a", "b", "z", "Q", "_", "-", ".", ",", "'", "1", "2", "٣", "é", "ß", "汉", "\u0301", "\u200d", "\u200c",
        " ", " ", " ", "  ", "\t", "\n", "\r\n", "\u00a0", "\u3000", "\u2028", "\u0085", "\u001c", "\u20ac",
        "hello", "world", "word", "the", "and", "😀"]
WS = set("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
         "\u2028\u2029\u202f\u205f\u3000")


def panics_possible(toks):
    """config.c's refusal: an rstrip token and an lstrip-only all-\\s token in the same phase."""
    for ph in (False, True):
        if not any(t["rstrip"] for t in toks if t["normalized"] == ph):
            continue
        if any(t["lstrip"] and not t["rstrip"] and all(c in WS for c in t["content"])
               for t in toks if t["normalized"] == ph):
            return True
    return False


def synth_tokenizer(rng):
    contents = rng.sample(POOL, rng.randint(0, 9))
    toks = []
    for c in contents:
        toks.append(G.added(c, special=rng.random() < 0.5, normalized=rng.random() < 0.4,
                            lstrip=rng.random() < 0.35, rstrip=rng.random() < 0.35,
                            single_word=rng.random() < 0.3))
    pat = rng.choice([G.CL100K, G.GPT2])
    sp = rng.choice(["iso", "isoinv", "reminv", "bytelevel"])
    if sp == "bytelevel":
        pre = G.bytelevel(ur=True)
    else:
        beh, inv = {"iso": ("Isolated", False), "isoinv": ("Isolated", True), "reminv": ("Removed", True)}[sp]
        pre = {"type": "Sequence", "pretokenizers": [G.split(pat, beh, inv), G.bytelevel()]}
    m = G.bpe_model(extra_vocab=["<s>", "</s>", "   ", " ", "€"], ignore_merges=rng.random() < 0.5,
                    dropout=rng.choice([None, 0.0]))
    nv = len(m["vocab"])
    ppk = rng.choice(["none", "template", "roberta", "bert", "seq"])
    pp = None
    if ppk == "template":
        pp = {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}},
                                                       {"Sequence": {"id": "A", "type_id": 0}},
                                                       {"SpecialToken": {"id": "</s>", "type_id": 0}}],
              "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
              "special_tokens": {"<s>": {"id": "<s>", "ids": [m["vocab"]["<s>"]], "tokens": ["<s>"]},
                                 "</s>": {"id": "</s>", "ids": [m["vocab"]["</s>"], 7], "tokens": ["</s>", "x"]}}}
    elif ppk == "roberta":
        pp = {"type": "RobertaProcessing", "sep": ["</s>", m["vocab"]["</s>"]], "cls": ["<s>", rng.randrange(nv)],
              "trim_offsets": rng.random() < 0.5, "add_prefix_space": rng.random() < 0.5}
    elif ppk == "bert":
        pp = {"type": rng.choice(["BertProcessing", "RobertaProcessing"]), "sep": ["</s>", rng.randrange(nv)],
              "cls": ["<s>", m["vocab"]["<s>"]]}
    elif ppk == "seq":
        pp = {"type": "Sequence", "processors": [G.bytelevel(aps=True, trim=False, ur=True),
                                                 {"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0],
                                                  "trim_offsets": True, "add_prefix_space": True}]}
    pad = rng.choice([None, {"strategy": "BatchLongest", "direction": rng.choice(["Left", "Right"]),
                             "pad_to_multiple_of": rng.choice([None, 0]), "pad_id": 0, "pad_type_id": 0,
                             "pad_token": "<s>"}])
    return G.tokenizer(m, toks, pre, post=pp, padding=pad), contents


def synth_text(rng, contents):
    parts = contents + [c[:k] for c in contents for k in range(1, len(c))] if contents else []
    pool = EDGE + parts * 3
    return "".join(rng.choice(pool) for _ in range(rng.randint(0, 16)))


MODES = [(0, "ALL", True), (1, "NONSPECIAL", True), (2, "NONE", True), (4, "ALL", False)]


class Stats:
    def __init__(self):
        self.cases = 0
        self.decodes = 0
        self.diffs = []
        self.refused_ok = 0
        self.hf_panics = 0
        self.loaded = 0


def compare(st, drv, tok, text, label):
    b = text.encode("utf-8")
    for fl, mode, pp in MODES:
        try:
            want = tok.encode(text, mode=mode, add_special_tokens=pp)
        except BaseException as e:  # noqa: BLE001 - pyo3 panics are BaseException
            st.hf_panics += 1
            st.diffs.append({"where": label, "text": text, "flags": fl, "hf": "panic: " + str(e)[:80]})
            continue
        status, got = drv.encode(b, fl)
        st.cases += 1
        if status != 0 or got != want:
            st.diffs.append({"where": label, "text": text, "flags": fl, "hf": want, "toks": got, "status": status})
        if fl == 0:
            for skip in (0, 1):
                hd = tok.decode(want, skip_special_tokens=bool(skip)).encode("utf-8")
                ds, td = drv.decode(want, skip)
                st.decodes += 1
                if ds != 0 or td != hd:
                    st.diffs.append({"where": label, "text": text, "decode_skip": skip, "ids": want,
                                     "hf": hd.decode("utf-8", "replace"), "toks": td.decode("utf-8", "replace")})


def run_synth(a, drv, st):
    lo, n = a.shard
    tmp = tempfile.mkdtemp(prefix="breadth-", dir=a.tmp)
    for i in range(a.tokenizers):
        if i % n != lo:
            continue
        rng = random.Random(a.seed * 1000003 + i)
        j, contents = synth_tokenizer(rng)
        try:
            j = G.resolve_template_ids(j)
            tok = O.load_synthetic(copy.deepcopy(j), f"synth{i}")
        except BaseException as e:  # noqa: BLE001
            print(f"synth {i}: hf refuses ({str(e)[:60]}), skipped")
            continue
        path = os.path.join(tmp, f"s{i}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False)
        try:
            drv.load(path)
        except Exception as e:  # noqa: BLE001
            drv.close()                                  # the driver exits after a refused load: restart it
            if panics_possible(j["added_tokens"]):
                st.refused_ok += 1
                continue
            st.diffs.append({"where": f"synth {i}", "load": str(e), "json": path})
            continue
        if panics_possible(j["added_tokens"]):
            st.diffs.append({"where": f"synth {i}", "load": "toks loaded a token set hf can panic on", "json": path})
            continue
        st.loaded += 1
        for _ in range(a.texts):
            compare(st, drv, tok, synth_text(rng, contents), f"synth {i} ({path})")
        if len(st.diffs) > 50:
            break


def run_real(a, drv, st):
    lo, n = a.shard
    path = a.file
    obj = json.load(open(path, encoding="utf-8"))
    if a.strip_normalizer:
        obj["normalizer"] = None
        path = os.path.join(tempfile.mkdtemp(prefix="breadth-", dir=a.tmp), "real.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    tok = O.load(path)
    drv.load(path)
    st.loaded += 1
    contents = [t["content"] for t in obj.get("added_tokens") or []]
    lines = []
    for c in a.corpus or []:
        with open(c, encoding="utf-8", errors="replace") as f:
            lines += [ln for ln in f.read().split("\n") if ln]
    rng = random.Random(a.seed * 7919 + lo)
    for k in range(a.texts):
        if k % n != lo:
            continue
        r = rng.random()
        if lines and r < 0.4:
            text = rng.choice(lines)
        elif lines and r < 0.6:
            text = rng.choice(lines) + rng.choice(EDGE + contents) + rng.choice(lines)
        else:
            text = synth_text(rng, rng.sample(contents, min(len(contents), 6)))
        compare(st, drv, tok, text, f"real {os.path.basename(a.file)}")
        if len(st.diffs) > 50:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--tmp", default=os.path.join(HERE, "..", "..", "..", "build", "breadth"))
    ap.add_argument("--out", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--tokenizers", type=int, default=100)
    s.add_argument("--texts", type=int, default=200)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--shard", default="0/1")
    r = sub.add_parser("real")
    r.add_argument("file")
    r.add_argument("--texts", type=int, default=10000)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--corpus", nargs="*")
    r.add_argument("--strip-normalizer", action="store_true")
    a = ap.parse_args()
    a.shard = tuple(int(x) for x in a.shard.split("/"))
    os.makedirs(a.tmp, exist_ok=True)
    drv = Driver([a.driver])
    st = Stats()
    t0 = time.time()
    try:
        if a.cmd == "synth":
            run_synth(a, drv, st)
        else:
            run_real(a, drv, st)
    finally:
        drv.close()
    res = {"cmd": a.cmd, "shard": f"{a.shard[0]}/{a.shard[1]}", "cases": st.cases, "decodes": st.decodes,
           "loaded": st.loaded,
           "refused_as_hf_can_panic": st.refused_ok, "hf_panics": st.hf_panics, "n_diffs": len(st.diffs),
           "seconds": round(time.time() - t0, 1), "diffs": st.diffs[:20]}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "diffs"}))
    for d in st.diffs[:5]:
        print("DIFF", json.dumps(d, ensure_ascii=True)[:600])
    return 1 if st.diffs else 0


if __name__ == "__main__":
    sys.exit(main())
