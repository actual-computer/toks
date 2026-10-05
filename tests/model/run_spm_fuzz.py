#!/usr/bin/env python3
"""tests/model/run_spm_fuzz.py: spm_model.py (docs/algorithms/spm_bpe.md) vs hf tokenizers 0.23.2.

For every pinned sentencepiece-style tokenizer (tests/spm/pins.json, fetched by tests/spm/fetch.py), generated
and real texts go through both sides and every result is compared:

  encode   mode ALL / NONSPECIAL / NONE (SPEC §3.2: hf default / encode_special_tokens=True / the file with
           added_tokens: []) x post-processing on / off (add_special_tokens)        6 checks per text
  decode   the ids of every ALL+pp encode, skip_special_tokens on / off            2 checks per text
           plus random id sequences (byte tokens, specials, holes, ids beyond the vocab)
  pieces   hf Tokenizer.model.tokenize(piece) vs spm_model's Model.tokenize (doc §5 alone)

An input on which hf raises must make the model raise too (and the reverse); both count as agreement.

    uv run --with tokenizers==0.23.2 tests/model/run_spm_fuzz.py --n 2000 --workers 4 llama2     # smoke
    tools/remote.sh <host> 'uv run --with tokenizers==0.23.2 tests/model/run_spm_fuzz.py --n 1000000 --workers 20'

Output: one RESULT line per tokenizer, mismatches with their repro (escaped), and --out (json) with counts.
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
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests", "spm"))

import spm_model as S  # noqa: E402
import fetch as F  # noqa: E402

CHUNK = 500
MODES = ("ALL", "NONSPECIAL", "NONE")

# a few sentences per script, for the real-text generator (with the repo's own text files)
MULTI = [
    "The quick brown fox jumps over the lazy dog. Isn't it? I'd say we've seen 12,345 cases since 2019.",
    "Le cœur a ses raisons que la raison ne connaît point. Où est la bibliothèque ? À bientôt !",
    "Falsches Üben von Xylophonmusik quält jeden größeren Zwerg. Straße, Maß, Fußgänger.",
    "El veloz murciélago hindú comía feliz cardillo y kiwi. ¿Qué tal? ¡Olé!",
    "Съешь же ещё этих мягких французских булок да выпей чаю. Привет, мир!",
    "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία. Καλημέρα κόσμε.",
    "صِف خَلقَ خَودِ كَمِثلِ الشَمسِ إِذ بَزَغَت يَحظى الضَجيعُ بِها نَجلاءَ مِعطارِ",
    "דג סקרן שט בים מאוכזב ולפתע מצא חברה. שלום עולם",
    "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश करने वाले विष्णुवतार भगवान श्रीराम",
    "আমি বাংলায় গান গাই। আমার সোনার বাংলা।",
    "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน",
    "我能吞下玻璃而不伤身体。天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。",
    "いろはにほへと ちりぬるを わかよたれそ つねならむ。私はガラスを食べられます。",
    "다람쥐 헌 쳇바퀴에 타고파. 안녕하세요, 세계!",
    "Tiếng Việt có dấu: Chào thế giới, tôi có thể ăn thủy tinh mà không hại gì.",
    "Pijamalı hasta yağız şoföre çabucak güvendi. İstanbul, ılık, Iğdır.",
    "Pchnąć w tę łódź jeża lub ośm skrzyń fig. Zażółć gęślą jaźń.",
    "ქართული ენა ძალიან ლამაზია. Հայերեն լեզու. ኢትዮጵያ አማርኛ ቋንቋ።",
    "emoji 👍🏽🔥 and a ZWJ family 👨‍👩‍👧‍👦 and flags 🇺🇸🇯🇵 and keycaps 1️⃣ ©️",
    "math: ∀x∈ℝ, ∃y: x² + y² ≥ 2xy; ∫₀^∞ e^{-x} dx = 1; ½ ⅓ ¼ Ⅻ ①②③",
    "def tokenize(x):\n    return [t for t in x.split(' ') if t]  # comment\n\tindented\ttabs\n",
    "<html><body class='x'>1 &amp; 2 &lt; 3</body></html>\r\nwindows\rline\u00a0nbsp\u3000ideographic",
    "int main(void) { printf(\"%d\\n\", 42); return 0; }  /* c */ // ok",
    "  leading spaces, trailing spaces   ,  double  spaces  and\u2581the\u2581metaspace\u2581char itself",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕒𝕤𝕥𝕣𝕒𝕝 𝓅𝓁𝒶𝓃𝑒𝓈 𠀀𠀁𪚥 \U000E0041 \U0010FFFD \uE000\uF8FF private use",
]

W = {}


def esc(s):
    return s.encode("unicode_escape").decode("ascii")


def _corpus():
    parts = list(MULTI)
    for d in ("docs", "tests/model", "src/core", "include"):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.endswith((".md", ".py", ".c", ".h")):
                try:
                    with open(os.path.join(p, fn), encoding="utf-8") as f:
                        parts.append(f.read())
                except (OSError, UnicodeDecodeError):
                    pass
    return "\n".join(parts)


def init(name, path, raw=None):
    from tokenizers import Tokenizer
    if raw is None:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8")
    tj = json.loads(raw)
    hf = Tokenizer.from_str(raw)
    none = dict(tj)
    none["added_tokens"] = []
    hf_none = Tokenizer.from_str(json.dumps(none))
    model = S.SpmTokenizer(tj)
    v = model.model.vocab
    byte_ids = [i for i in model.model.byte_id if i is not None]
    added_ids = sorted(model.added)
    special_ids = sorted(a.id for a in model.added.values() if a.content in model.specials)
    words = [s for s in v if not (len(s) == 6 and s.startswith("<0x")) and s not in model.added] or ["a"]
    max_id = max(max(v.values()), max(added_ids) if added_ids else 0)
    holes = sorted(set(range(max_id + 1)) - set(v.values()) - set(added_ids))
    W.update(name=name, hf=hf, hf_none=hf_none, model=model, words=words, byte_ids=byte_ids or [0],
             added_ids=added_ids, special_ids=special_ids, holes=holes, max_id=max_id,
             added=[a.content for a in model.added.values()], matches=[a.match for a in model.added.values()],
             corpus=W.get("corpus") or _corpus(), pools=W.get("pools") or _pools(),
             chars=[s for s in v if len(s) == 1], synth=None)


def _pools():
    try:
        from toks_oracle import gen as G
        return G.interesting_pools(), G.CONTEXTS
    except Exception:  # noqa: BLE001
        return [[chr(c) for c in range(32, 127)]], ["x{}y", " {}", "{} "]


# ---------------------------------------------------------------------------------------------- generators
SP = [" ", "  ", "   ", "\u2581", "\u2581\u2581", " \u2581", "\u2581 ", "\t", "\n", "\r\n", "\u00a0", "\u3000"]


def rand_cp(rng):
    r = rng.random()
    if r < 0.25:
        return chr(rng.randint(0x20, 0x7E))
    if r < 0.40:
        return chr(rng.randint(0x80, 0x7FF))
    if r < 0.60:
        return chr(rng.choice([rng.randint(0x800, 0xD7FF), rng.randint(0xE000, 0xFFFD)]))
    if r < 0.75:
        return chr(rng.randint(0x10000, 0x10FFFF))
    if r < 0.85:
        return chr(rng.choice([0x1F600, 0x1F44D, 0x2764, 0x200D, 0xFE0F, 0x1F1E6, 0x1F3FD, 0x20AC, 0xA9]))
    return chr(rng.randint(0x4E00, 0x9FFF))


def g_words(rng):
    w = W["words"]
    out = []
    for _ in range(rng.randint(1, 30)):
        s = rng.choice(w)
        if rng.random() < 0.85:
            s = s.replace("\u2581", " ")
        out.append(s)
        if rng.random() < 0.15:
            out.append(rng.choice(SP))
    return "".join(out)


def g_spaces(rng):
    atoms = SP + ["a", "ab", "x y", "Hello", "!", "1", "\u2581x", "é", "中", "😀", ""]
    return "".join(rng.choice(atoms) for _ in range(rng.randint(0, 12)))


def g_fallback(rng):
    out = []
    for _ in range(rng.randint(1, 20)):
        r = rng.random()
        out.append(rand_cp(rng) if r < 0.6 else rng.choice(SP) if r < 0.8 else rng.choice(["a", "the", "x", "."]))
    return "".join(out)


def g_added(rng):
    lit = W["added"] + W["matches"]
    if not lit:
        return g_spaces(rng)
    out = []
    for _ in range(rng.randint(1, 8)):
        r = rng.random()
        if r < 0.45:
            t = rng.choice(lit)
            if rng.random() < 0.15 and len(t) > 1:
                t = t[:rng.randint(1, len(t) - 1)]          # a prefix of a literal
            out.append(t)
        elif r < 0.75:
            out.append(rng.choice(SP))
        else:
            out.append(rng.choice(["a", "hello", "x", "<", ">", "</", "s>", "\u2581", "中", "😀"]))
    return "".join(out)


def g_literals(rng):
    lits = ["<0x%02X>" % rng.randint(0, 255), "<0x%02x>" % rng.randint(0, 255), "<0x+F>", "<unk>", "<s>", "</s>",
            "<pad>", "<bos>", "<eos>", "<mask>", "[INST]", "<0x", "0x41>", "<|endoftext|>"]
    return "".join(rng.choice(lits + SP + ["a", "b"]) for _ in range(rng.randint(1, 8)))


def g_pools(rng):
    pools, _ = W["pools"]
    pool = rng.choice(pools)
    return "".join(rng.choice(pool) for _ in range(rng.randint(1, 40)))


def g_context(rng):
    _, ctx = W["pools"]
    cp = rng.randint(0, 0x10FFFF - 0x800)
    if cp >= 0xD800:
        cp += 0x800
    ch = chr(cp)
    return rng.choice(ctx).format(ch, ch)


def g_real(rng, lo=1, hi=600):
    c = W["corpus"]
    a = rng.randrange(0, len(c))
    return c[a:a + rng.randint(lo, hi)]


def g_long(rng):
    if rng.random() < 0.5:
        return g_real(rng, 1000, 6000)
    return "".join(g_words(rng) for _ in range(rng.randint(10, 60)))


# ---------------------------------------------------------------------------------------------- synthetic
# small tokenizers built to reach what the real files never do: missing byte tokens, unk absent / present /
# outside the vocab, fuse_unk off, ignore_merges on, competing and duplicate merges, id holes, every
# normalizer / pre-tokenizer / decoder shape the doc accepts, added tokens with every flag.
SYN_CHARS = ["a", "b", "c", "d", "s", "x", "\u2581", " ", "_", "\u00e9", "\u4e2d", "\n", "\U0001F600", "<", ">", "/"]
SYN_TEXT = SYN_CHARS + ["  ", "\u2581\u2581", "ab", "\t", "\u00a0", "\u0301", "\u3000", "z", "\U00010348"]


def synth(rng):
    vocab = {}

    def add(s):
        if s not in vocab:
            vocab[s] = len(vocab)
    unk_kind = rng.choice(["in", "in", "in", "none", "missing"])
    if unk_kind == "in":
        add("<unk>")
    add("<s>")
    add("</s>")
    bkind = rng.choice(["all", "all", "some", "none"])
    for b in range(256):
        if bkind == "all" or (bkind == "some" and rng.random() < 0.6):
            add("<0x%02X>" % b)
    for c in SYN_CHARS:
        if rng.random() < 0.75:
            add(c)
    toks = [s for s in vocab if not s.startswith("<0x")] or ["a"]
    merges = []
    for _ in range(rng.randint(0, 60)):
        a, b = rng.choice(toks), rng.choice(toks)
        if a not in vocab or b not in vocab or len(a) + len(b) > 8:
            continue
        add(a + b)
        toks.append(a + b)
        merges.append([a, b])
    for _ in range(rng.randint(0, 3)):
        if merges:
            merges.append(list(rng.choice(merges)))       # duplicate pairs: the last rank wins
    if rng.random() < 0.5:
        rng.shuffle(merges)
    if rng.random() < 0.15:
        vocab = {s: 2 * i + 1 for s, i in vocab.items()}    # id holes
    space_in = any(" " in x for m in merges for x in m)
    model = {"type": "BPE", "dropout": None,
             "unk_token": None if unk_kind == "none" else ("<unk>" if unk_kind == "in" else "<nope>"),
             "continuing_subword_prefix": None, "end_of_word_suffix": None,
             "fuse_unk": rng.random() < 0.6, "byte_fallback": rng.random() < 0.7,
             "ignore_merges": rng.random() < 0.3, "vocab": vocab,
             "merges": merges if (space_in or rng.random() < 0.5) else [" ".join(m) for m in merges]}
    P = lambda s: {"type": "Prepend", "prepend": s}  # noqa: E731
    R = lambda p, c: {"type": "Replace", "pattern": {"String": p}, "content": c}  # noqa: E731
    Q = lambda xs: {"type": "Sequence", "normalizers": xs}  # noqa: E731
    norm = rng.choice([None, None, P("\u2581"), R(" ", "\u2581"), Q([P("\u2581"), R(" ", "\u2581")]),
                       Q([R(" ", "\u2581"), P("\u2581")]), Q([R(" ", "\u2581")]), R("ab", "X"), P("\u2581\u2581"),
                       R(" ", ""), Q([R(" ", "\u2581"), R("\u2581\u2581", "\u2581")]), R("a", "aa"), P("_")])
    rep = rng.choice(["\u2581", "\u2581", "_"])

    def meta():
        d = {"type": "Metaspace", "replacement": rep}
        k = rng.random()
        if k < 0.75:
            d["prepend_scheme"] = rng.choice(["always", "first", "never"])
            d["split"] = rng.random() < 0.5
        elif k < 0.9:
            d["add_prefix_space"] = True
        else:
            d["add_prefix_space"] = False
            d["prepend_scheme"] = "never"
        return d

    def split():
        return {"type": "Split", "pattern": {"String": rng.choice([" ", "\u2581", "a", "ab", "_"])},
                "behavior": rng.choice(["Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous"]),
                "invert": rng.random() < 0.25}
    pt = rng.choice([None, None, "m", "m", "s", "ms", "sm"])
    if pt == "m":
        pre = meta()
    elif pt == "s":
        pre = split()
    elif pt in ("ms", "sm"):
        xs = [meta(), split()] if pt == "ms" else [split(), meta()]
        pre = {"type": "Sequence", "pretokenizers": xs}
    else:
        pre = None
    decs = [{"type": "Replace", "pattern": {"String": rep}, "content": " "}, {"type": "ByteFallback"},
            {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": rng.randint(0, 2), "stop": rng.randint(0, 1)},
            {"type": "Metaspace", "replacement": rep, "prepend_scheme": rng.choice(["always", "first", "never"]),
             "split": True}]
    k = rng.random()
    if k < 0.15:
        dec = None
    elif k < 0.3:
        dec = decs[4]
    else:
        dec = {"type": "Sequence", "decoders": [d for d in decs if rng.random() < 0.6]}
        if rng.random() < 0.5:
            rng.shuffle(dec["decoders"])
    added = []
    for c in rng.sample(["<s>", "</s>", "<unk>", "ab", "\u2581a", " x", "<0x41>", "a", "\u4e2d", "<mask>", "\n",
                         "s>", "<s", "xx"], rng.randint(0, 5)):
        sp = rng.random() < 0.5
        added.append({"id": 0, "content": c, "single_word": rng.random() < 0.1, "lstrip": rng.random() < 0.2,
                      "rstrip": rng.random() < 0.2, "normalized": rng.random() < 0.5, "special": sp})
    if added and rng.random() < 0.1:
        added.append(dict(added[0], special=not added[0]["special"]))   # a repeated content
    pp = None
    if rng.random() < 0.5:
        ids_s = [vocab.get("<s>", 0)]
        single = [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}]
        if rng.random() < 0.5:
            single.append({"SpecialToken": {"id": "</s>", "type_id": 0}})
        pp = {"type": "TemplateProcessing", "single": single, "pair": single,
              "special_tokens": {"<s>": {"id": "<s>", "ids": ids_s, "tokens": ["<s>"]},
                                 "</s>": {"id": "</s>", "ids": [vocab.get("</s>", 1)], "tokens": ["</s>"]}}}
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": norm,
            "pre_tokenizer": pre, "post_processor": pp, "decoder": dec, "model": model}


def g_synth(rng):
    lit = W["added"] + W["matches"]
    out = []
    for _ in range(rng.randint(0, 14)):
        if lit and rng.random() < 0.15:
            out.append(rng.choice(lit))
        else:
            out.append(rng.choice(SYN_TEXT))
    return "".join(out)


def work_synth(task):
    k, seed, n = task
    rng = random.Random(seed)
    tj = synth(rng)
    raw = json.dumps(tj)
    try:
        init(f"synth{k}", None, raw)
    except S.Unsupported as e:
        return {"refused_model": 1}, {}, [{"kind": "model-refused", "k": k, "why": str(e)}]
    except BaseException as e:  # noqa: BLE001  (hf refuses the file: so must toks; the model may accept)
        if isinstance(e, KeyboardInterrupt):
            raise
        return {"refused_hf": 1}, {}, []
    W["synth"] = True
    c, gens, ex = work((seed, n))
    c["tokenizers"] = 1
    for e in ex:
        e["tokenizer"] = raw
    return c, gens, ex


GENS = [(0.18, g_words), (0.10, g_spaces), (0.12, g_fallback), (0.14, g_added), (0.05, g_literals),
        (0.11, g_pools), (0.08, g_context), (0.20, g_real), (0.02, g_long)]


def gen_text(rng):
    if W.get("synth"):
        if rng.random() < 0.8:
            return "g_synth", g_synth(rng)
    r = rng.random()
    for p, g in GENS:
        if r < p:
            return g.__name__, g(rng)
        r -= p
    return "g_real", g_real(rng)


def gen_piece(rng):
    if W.get("synth") and rng.random() < 0.8:
        return g_synth(rng) or "a"
    r = rng.random()
    if r < 0.4:
        return g_words(rng).replace(" ", "\u2581")[:200]
    if r < 0.7:
        return "".join(rand_cp(rng) if rng.random() < 0.5 else rng.choice(["\u2581", "a", "e", "s"])
                       for _ in range(rng.randint(1, 30)))
    return g_real(rng, 1, 200).replace(" ", "\u2581")


def gen_ids(rng):
    pools = [W["byte_ids"], W["special_ids"] or [0], W["added_ids"] or [0], W["holes"] or [0],
             [W["max_id"] + 1, W["max_id"] + 1000], list(range(0, 300))]
    out = []
    for _ in range(rng.randint(0, 24)):
        r = rng.random()
        if r < 0.45:
            out.append(rng.randint(0, W["max_id"]))
        else:
            out.append(rng.choice(rng.choice(pools)))
    return out


# ---------------------------------------------------------------------------------------------- checks
def run_hf(f):
    try:
        return ("ok", f())
    except BaseException as e:  # noqa: BLE001  (pyo3 panics are BaseException)
        if isinstance(e, KeyboardInterrupt):
            raise
        return ("err", type(e).__name__)


def run_model(f):
    try:
        return ("ok", f())
    except S.HfError as e:
        return ("err", str(e))


def same(h, m):
    return h[0] == m[0] if (h[0] == "err" or m[0] == "err") else h == m


def work(task):
    seed, n = task
    rng = random.Random(seed)
    hf, hf_none, model = W["hf"], W["hf_none"], W["model"]
    c = {"texts": 0, "encode": 0, "decode": 0, "pieces": 0, "cuts": 0, "bad_encode": 0, "bad_decode": 0,
         "bad_pieces": 0, "bad_cuts": 0}
    gens = {}
    ex = []
    for _ in range(n):
        gname, text = gen_text(rng)
        gens[gname] = gens.get(gname, 0) + 1
        c["texts"] += 1
        ids_all_pp = None
        for mode in MODES:
            for pp in (False, True):
                if mode == "NONE":
                    h = run_hf(lambda: hf_none.encode(text, add_special_tokens=pp).ids)
                else:
                    hf.encode_special_tokens = (mode == "NONSPECIAL")
                    h = run_hf(lambda: hf.encode(text, add_special_tokens=pp).ids)
                m = run_model(lambda: model.encode(text, mode, pp))
                c["encode"] += 1
                if not same(h, m):
                    c["bad_encode"] += 1
                    if len(ex) < 5:
                        ex.append({"kind": "encode", "mode": mode, "pp": pp, "text": esc(text), "hf": h, "model": m})
                if mode == "ALL" and pp and h[0] == "ok":
                    ids_all_pp = h[1]
                if mode == "ALL" and not pp:                # doc §5.5: word by word at the certified cuts
                    mc = run_model(lambda: model.encode(text, "ALL", False, cuts=True))
                    c["cuts"] += 1
                    if not same(h, mc):
                        c["bad_cuts"] += 1
                        if len(ex) < 5:
                            ex.append({"kind": "cut-encode", "text": esc(text), "hf": h, "model": mc})
        hf.encode_special_tokens = False
        id_lists = [ids_all_pp] if ids_all_pp is not None else []
        if rng.random() < 0.3:
            id_lists.append(gen_ids(rng))
        for ids in id_lists:
            for skip in (False, True):
                h = run_hf(lambda: hf.decode(ids, skip_special_tokens=skip))
                m = run_model(lambda: model.decode(ids, skip))
                c["decode"] += 1
                if not same(h, m):
                    c["bad_decode"] += 1
                    if len(ex) < 10:
                        ex.append({"kind": "decode", "skip": skip, "ids": ids, "hf": esc(str(h)),
                                   "model": esc(str(m))})
        if rng.random() < 0.5:
            piece = gen_piece(rng)
            h = run_hf(lambda: [t.id for t in hf.model.tokenize(piece)])
            m = run_model(lambda: model.model.tokenize(piece))
            c["pieces"] += 1
            if not same(h, m):
                c["bad_pieces"] += 1
                if len(ex) < 5:
                    ex.append({"kind": "piece", "piece": esc(piece), "hf": h, "model": m})
            mc = run_model(lambda: model.model.tokenize_words(piece))
            c["cuts"] += 1
            if not same(h, mc):
                c["bad_cuts"] += 1
                if len(ex) < 5:
                    ex.append({"kind": "cut-piece", "piece": esc(piece), "hf": h, "model": mc})
    return c, gens, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--n", type=int, default=2000, help="texts per tokenizer")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--synthetic", type=int, default=0, help="also fuzz K synthetic tokenizers")
    ap.add_argument("--synth-n", type=int, default=200, help="texts per synthetic tokenizer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=8, help="mismatch examples printed per run")
    ap.add_argument("--only-kind", default=None, help="print only mismatches of this kind (encode, decode, ...)")
    args = ap.parse_args()
    import tokenizers
    S._word_set()                                           # probed once here, inherited by the workers
    pins = F.pins()
    names = (args.names or list(pins)) if args.n > 0 else []
    summary = {"tokenizers_version": tokenizers.__version__, "n": args.n, "seed": args.seed, "results": {}}
    total_bad = 0
    ctx = mp.get_context("fork")
    if args.synthetic:
        t0 = time.time()
        tasks = [(k, args.seed * 7 + k * 1000003, args.synth_n) for k in range(args.synthetic)]
        agg, gens, shown = {}, {}, 0
        with ctx.Pool(args.workers) as pool:
            for cc, gg, ex in pool.imap_unordered(work_synth, tasks, chunksize=4):
                for k, v in cc.items():
                    agg[k] = agg.get(k, 0) + v
                for k, v in gg.items():
                    gens[k] = gens.get(k, 0) + v
                for e in ex:
                    if shown < args.show and (args.only_kind is None or e["kind"] == args.only_kind):
                        print(f"[synthetic] {e['kind'].upper()} {json.dumps(e, ensure_ascii=True)}", flush=True)
                        shown += 1
        bad = sum(agg.get(k, 0) for k in ("bad_encode", "bad_decode", "bad_pieces", "bad_cuts")) + agg.get("refused_model", 0)
        total_bad += bad
        print(f"RESULT synthetic: tokenizers {agg.get('tokenizers', 0)} (hf refused {agg.get('refused_hf', 0)}, "
              f"model refused {agg.get('refused_model', 0)}) texts {agg.get('texts', 0)} encode {agg.get('encode', 0)} "
              f"(bad {agg.get('bad_encode', 0)}) decode {agg.get('decode', 0)} (bad {agg.get('bad_decode', 0)}) "
              f"pieces {agg.get('pieces', 0)} (bad {agg.get('bad_pieces', 0)}) cuts {agg.get('cuts', 0)} "
              f"(bad {agg.get('bad_cuts', 0)}) in {time.time() - t0:.0f}s", flush=True)
        summary["results"]["synthetic"] = {"counts": agg, "generators": gens, "seconds": round(time.time() - t0, 1)}
    for name in names:
        path = F.fetch(name, pins[name])
        t0 = time.time()
        tasks = []
        for i, st in enumerate(range(0, args.n, CHUNK)):
            seed = args.seed * 1000003 + i * 7919 + sum(map(ord, name)) * 104729
            tasks.append((seed, min(CHUNK, args.n - st)))
        agg = {}
        gens = {}
        shown = 0
        with ctx.Pool(args.workers, initializer=init, initargs=(name, path)) as pool:
            for cc, gg, ex in pool.imap_unordered(work, tasks, chunksize=1):
                for k, v in cc.items():
                    agg[k] = agg.get(k, 0) + v
                for k, v in gg.items():
                    gens[k] = gens.get(k, 0) + v
                for e in ex:
                    if shown < args.show and (args.only_kind is None or e["kind"] == args.only_kind):
                        print(f"[{name}] MISMATCH {json.dumps(e, ensure_ascii=True)}", flush=True)
                        shown += 1
        bad = agg["bad_encode"] + agg["bad_decode"] + agg["bad_pieces"] + agg["bad_cuts"]
        total_bad += bad
        dt = time.time() - t0
        print(f"RESULT {name} sha256 {pins[name]['sha256'][:16]}: texts {agg['texts']} encode {agg['encode']} "
              f"(bad {agg['bad_encode']}) decode {agg['decode']} (bad {agg['bad_decode']}) pieces {agg['pieces']} "
              f"(bad {agg['bad_pieces']}) cuts {agg['cuts']} (bad {agg['bad_cuts']}) in {dt:.0f}s", flush=True)
        summary["results"][name] = {"sha256": pins[name]["sha256"], "counts": agg, "generators": gens,
                                    "seconds": round(dt, 1)}
    print(f"TOTAL mismatches {total_bad}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
