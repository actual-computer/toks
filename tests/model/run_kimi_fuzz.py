#!/usr/bin/env python3
"""tests/model/run_kimi_fuzz.py: tests/model/kimi_model.py against the reference (tests/parity/
oracle_tiktoken.py: the real TikTokenTokenizer + tiktoken) at the ids level, in the three modes, and the
model's scanner against tiktoken's own regex engine (tools/oracle/tiktoken-pieces) at the piece level.

    tools/oracle/py.sh tests/model/run_kimi_fuzz.py --jobs 20 --short 1000000 --long 300 --exh 4 \
        --vocab 100000 --real build/realtext --seed 1 --out build/kimi/fuzz.json
    tools/oracle/py.sh tests/model/run_kimi_fuzz.py --fetch build/realtext      # the real-text corpus

Sources (each case runs serving, direct and none; pieces = the whole case text as one regex segment):
  short   class-aware strings of 0..~200 atoms, heavy on CJK (Han of every kind beside kana, hangul, latin,
          digits, marks, CJK punctuation, the 25 \\s chars and the 4 str.isspace-only ones, apostrophes,
          special names and their fragments)
  long    the chunking edges: same-isspace runs of 24,999..75,003 chars of every class, special names
          straddling the 25,000 cuts and the 400,000 chunk edge, texts over 400,000 chars
  exh     every string of 0..EXH atoms over REP (one atom per behaviour)
  vocab   1..30 random Kimi tokens (their bytes, where utf-8) joined
  real    every line / paragraph / random chunk (1..16,384 chars) of the files under --real
Exit 0 iff 0 mismatches. Counts per source and mode go to --out (json) and stdout.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))
sys.path.insert(0, os.path.join(ROOT, "tools", "oracle"))
if os.environ.get("TOKS_KIMI_MODEL_DIR"):          # tests/model/mutants_kimi.py: a mutated kimi_model.py
    sys.path.insert(0, os.environ["TOKS_KIMI_MODEL_DIR"])

MODES = ("serving", "direct", "none")

# ---------------------------------------------------------------- atoms (one list per behaviour)
C = chr
HAN = ["\u4e2d", "\u6587", "\u5b57", "\u56fd", "\u4eba", "\u65e5", "\u672c", "\u8a9e", "\u5b78", "\u3005", "\u303b",
       "\u3007", "\u3021", "\u3029", "\u3038", "\u2e80", "\u2f00", "\u2fd5", C(0x20000), C(0x2a6df), C(0x2ebf0),
       "\uf900", "\ufa6d", C(0x16ff0), C(0x16fe2), C(0x16fe3), C(0x30000), C(0x323af), "\u3400", "\u9fff"]
KANA = ["\u3042", "\u3044", "\u30a2", "\u30a4", "\uff71", "\u30fc", "\u309d", "\u3099", "\u309b", "\u30fb", "\u3063",
        "\u30f3"]
HANGUL = ["\ud55c", "\uad6d", "\uc5b4", "\u1100", "\u3131", "\u1161"]
UPPER = ["A", "B", "M", "S", "T", "Z", "\u00c9", "\u03a9", "\u0414", "\u01c5", "\u1f88", "\uff21"]
LOWER = ["a", "b", "e", "l", "s", "t", "z", "\u00e9", "\u03c9", "\u0434", "\u00df", "\u017f", "\uff41"]
BOTHL = ["\u02b0", "\u00aa", "\u02c6", "\u05d0", "\u0628", "\u0e01", "\u1100", "\ua7f9"]   # Lm / Lo, not Han
MARKS = ["\u0301", "\u0308", "\u302a", "\u302d", "\u20d0", "\u0e31", "\u093e", "\ufe0f", "\u0897"]
DIGITS = ["0", "1", "5", "9", "\uff10", "\u0663", "\u00b2", "\u216b", "\u2460", "\u3007", "\u0e51"]
PUNCT = list("!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~") + [
    "\u3002", "\u3001", "\u300c", "\u300d", "\u300a", "\u300b", "\uff01", "\uff1f", "\u30fb", "\u2026", "\u2014",
    "\u00a9", "\u20ac", "\U0001f600", "\U0001f44d", "\u200d", "\ufeff", "\u00ad", "\u200b", "\u180e"]
WSP = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x85", "\xa0", "\u1680", "\u2000", "\u2005", "\u200a", "\u2028",
       "\u2029", "\u202f", "\u205f", "\u3000", "\x1c", "\x1d", "\x1e", "\x1f"]
APOS = ["'", "'s", "'S", "'\u017f", "'t", "'T", "'re", "'RE", "'rE", "'ve", "'m", "'ll", "'lL", "'d", "'D", "'r",
        "'l", "'x"]
SPECIAL = ["[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>", "[start_header_id]",
           "[end_header_id]", "[EOT]", "<|media_begin|>", "<|media_content|>", "<|media_end|>", "<|media_pad|>",
           "<osagent_mode>", "[UNK]", "[PAD]", "<|reserved_token_163592|>", "<|reserved_token_163600|>",
           "<|reserved_token_163837|>", "<|reserved_token_163650|>"]
FRAGS = ["[BOS", "BOS]", "<|open|", "<|", "|>", "<|reserved_token_16", "<|reserved_token_163583|>",
         "<|reserved_token_163840|>", "<|im_end|>", "[eos]", "<osagent_mode"]
REP = ["\u4e2d", "\u3007", "\u2e80", "\u3042", "\u30fc", "\ud55c", "A", "a", "\u01c5", "\u02b0", "\u0301", "1",
       "\uff10", "!", "\u3002", "'", "s", " ", "\t", "\n", "\r", "\u3000", "\x1c", "[BOS]",
       "<|reserved_token_163600|>"]

POOLS = {"han": HAN, "kana": KANA, "hangul": HANGUL, "upper": UPPER, "lower": LOWER, "bothl": BOTHL,
         "marks": MARKS, "digits": DIGITS, "punct": PUNCT, "ws": WSP, "apos": APOS, "special": SPECIAL,
         "frags": FRAGS}
PROFILES = {   # pool weights
    "cjk": {"han": 30, "kana": 10, "hangul": 6, "punct": 10, "ws": 8, "digits": 6, "lower": 6, "upper": 4,
            "marks": 3, "special": 2, "apos": 2, "bothl": 2},
    "mixed": {k: 5 for k in POOLS},
    "case": {"upper": 20, "lower": 20, "bothl": 10, "marks": 8, "apos": 12, "han": 6, "ws": 5, "punct": 5,
             "digits": 3},
    "ws": {"ws": 40, "lower": 10, "han": 10, "punct": 10, "digits": 5, "special": 3},
    "digits": {"digits": 40, "han": 10, "ws": 10, "punct": 10, "lower": 5},
    "punct": {"punct": 40, "ws": 15, "han": 10, "marks": 8, "lower": 5, "apos": 5},
    "special": {"special": 15, "frags": 10, "han": 15, "lower": 10, "ws": 10, "punct": 10},
}


def gen_short(rng):
    prof = rng.choice(list(PROFILES) + ["cjk", "cjk", "tiny"])
    if prof == "tiny":
        return "".join(rng.choice(REP + PUNCT[:5] + WSP[:6]) for _ in range(rng.randint(0, 4)))
    w = PROFILES[prof]
    names, weights = list(w), [w[k] for k in w]
    out = []
    for _ in range(rng.randint(1, 120)):
        pool = POOLS[rng.choices(names, weights)[0]]
        a = rng.choice(pool)
        out.append(a * (rng.randint(1, 4) if rng.random() < 0.25 else 1))
    return "".join(out)


def gen_long(rng):
    """the wrapper's cut edges (docs/models/kimi.md 2.3)."""
    kind = rng.choice(["run", "run", "run2", "straddle", "chunk", "mixedlong"])
    if kind in ("run", "run2"):
        L = rng.choice([24999, 25000, 25001, 25002, 49999, 50000, 50001, 75003])
        if kind == "run":
            body = rng.choice(["a", "\u4e2d", " ", "\t", "\u3000", "\x1c", "\n", "!", "1", "\u0301", "A"]) * L
        else:
            pool = rng.choice([["a", "\u4e2d", "1", "!", "B", "\u3042"], [" ", "\t", "\n", "\u3000", "\x1c", "\r"]])
            body = "".join(rng.choice(pool) for _ in range(L))
        return gen_short(rng)[:50] + body + gen_short(rng)[:50]
    if kind == "straddle":
        name = rng.choice(SPECIAL)
        k = rng.randint(1, len(name) - 1)
        lead = rng.choice(["a", "\u4e2d", "x"]) * (25000 - k)
        return lead + name + rng.choice(["b", "\u6587", " "]) * rng.randint(1, 30000)
    if kind == "chunk":
        name = rng.choice(SPECIAL + ["  ", "\u4e2d\u6587", "ab"])
        k = rng.randint(0, len(name))
        parts = []
        n = 0
        while n < 400000 - k:
            p = gen_short(rng) or "x"
            parts.append(p)
            n += len(p)
        s = "".join(parts)[:400000 - k]
        return s + name + gen_short(rng)
    n = rng.randint(400001, 520000)
    parts, m = [], 0
    while m < n:
        p = gen_short(rng) or "y"
        parts.append(p)
        m += len(p)
    return "".join(parts)[:n]


def gen_vocab(rng, toks):
    out = []
    for _ in range(rng.randint(1, 30)):
        b = toks[rng.randrange(len(toks))]
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return "".join(out)


def real_cases(dirpath, rng, per_file_chunks=200):
    for root, _dirs, files in os.walk(dirpath):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            try:
                with open(p, encoding="utf-8") as f:
                    txt = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            for line in txt.split("\n"):
                if line:
                    yield line
            for para in txt.split("\n\n"):
                if para:
                    yield para
            for _ in range(per_file_chunks):
                if not txt:
                    break
                a = rng.randrange(len(txt))
                yield txt[a:a + rng.randint(1, 16384)]


# ---------------------------------------------------------------- worker
_W = {}


def _init():
    import kimi_model
    import oracle_tiktoken
    _W["model"] = kimi_model.Model()
    _W["ref"] = oracle_tiktoken.Kimi()
    import pieces
    _W["pat"] = pieces.pattern("kimi_k3")
    _W["split"] = pieces.split


def _check(job):
    src, texts = job
    m, ref = _W["model"], _W["ref"]
    res = {"src": src, "cases": len(texts), "mism": {k: 0 for k in MODES}, "ref_err": 0, "pieces": 0,
           "piece_mism": 0, "examples": []}
    for t in texts:
        for mode in MODES:
            r = ref.safe_encode(t, mode)
            if isinstance(r, dict):
                res["ref_err"] += 1
                if len(res["examples"]) < 5:
                    res["examples"].append({"mode": mode, "ref_error": r["error"], "text": t[:300]})
                continue
            try:
                g = m.encode(t, mode)
            except Exception as e:  # noqa: BLE001
                g = f"model raised {type(e).__name__}: {e}"
            if g != r:
                res["mism"][mode] += 1
                if len(res["examples"]) < 5:
                    res["examples"].append({"mode": mode, "text": t if len(t) < 2000 else t[:2000] + "...",
                                            "len": len(t), "model": str(g)[:500], "ref": str(r)[:500]})
    got = _W["split"](_W["pat"], texts)
    for t, pc in zip(texts, got):
        res["pieces"] += 1
        try:
            mine = m.pieces(t)
        except Exception as e:  # noqa: BLE001
            mine = f"model raised {type(e).__name__}: {e}"
        if mine != pc:
            res["piece_mism"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append({"pieces": True, "text": t[:2000], "model": mine[:60],
                                        "ref": pc if isinstance(pc, str) else pc[:60]})
    return res


def jobs(args):
    rng = random.Random(args.seed)
    B = args.batch
    if args.exh:
        n = 0
        for L in range(args.exh + 1):
            batch = []
            for combo in itertools.product(REP, repeat=L):
                batch.append("".join(combo))
                if len(batch) == B:
                    yield ("exh", batch)
                    batch = []
                n += 1
            if batch:
                yield ("exh", batch)
    for k in range(0, args.short, B):
        r = random.Random(f"{args.seed}/short/{k}")
        yield ("short", [gen_short(r) for _ in range(min(B, args.short - k))])
    if args.vocab:
        import kimi_model
        toks = sorted(kimi_model.read_ranks(os.path.join(kimi_model.KIMI_DIR, "tiktoken.model")),
                      key=lambda b: b)
        for k in range(0, args.vocab, B):
            r = random.Random(f"{args.seed}/vocab/{k}")
            yield ("vocab", [gen_vocab(r, toks) for _ in range(min(B, args.vocab - k))])
    for k in range(args.long):
        r = random.Random(f"{args.seed}/long/{k}")
        yield ("long", [gen_long(r)])
    for real in (args.real.split(",") if args.real else []):
        batch = []
        for t in real_cases(real, rng):
            batch.append(t)
            if len(batch) == max(1, B // 4):
                yield ("real", batch)
                batch = []
        if batch:
            yield ("real", batch)


# ---------------------------------------------------------------- real-text corpus
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
# whole articles as wikitext (action=raw): CJK prose with markup, tables, references, templates
WIKI_PAGES = {
    "zh": ["中华人民共和国", "北京市", "汉字", "中国历史", "长城", "孔子", "四大发明", "上海市"],
    "ja": ["日本", "東京都", "日本語", "漢字", "平仮名", "京都市", "源氏物語", "明治維新"],
    "ko": ["대한민국", "서울특별시", "한국어", "한글", "조선", "세종", "부산광역시"],
    "zh-yue": ["香港", "粵語"], "zh-classical": ["論語", "史記"], "wuu": ["上海"],
    "en": ["Python_(programming_language)", "Unicode", "China", "Japanese_writing_system"],
    "ru": ["Россия"], "ar": ["مصر"], "hi": ["भारत"], "th": ["ประเทศไทย"], "vi": ["Việt_Nam"],
}
# project gutenberg utf-8 texts: chinese classics (red chamber, three kingdoms, journey to the west, water
# margin, analects), japanese (rashomon), korean
GUTENBERG = [24264, 23950, 23962, 23863, 23839, 1982, 23931]


def fetch(dirpath):
    import io
    import tarfile
    import urllib.parse
    import urllib.request
    os.makedirs(dirpath, exist_ok=True)
    hdr = {"User-Agent": "toks-kimi-fuzz/0.1 (tokenizer conformance tests)"}

    def get(url, timeout=120):
        return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout).read()

    if not os.path.isdir(os.path.join(dirpath, "flores")):
        data = get(FLORES_URL, 600)
        n = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for mem in tf.getmembers():
                if mem.isfile() and (mem.name.endswith(".dev") or mem.name.endswith(".devtest")):
                    out = os.path.join(dirpath, "flores", os.path.basename(mem.name) + ".txt")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(tf.extractfile(mem).read())
                    n += 1
        print(f"flores-200: {n} files", file=sys.stderr)
    for lang, pages in WIKI_PAGES.items():
        for title in pages:
            out = os.path.join(dirpath, "wiki", f"{lang}-{title[:40]}.txt")
            if os.path.exists(out):
                continue
            url = f"https://{lang}.wikipedia.org/w/index.php?action=raw&title=" + urllib.parse.quote(title)
            try:
                body = get(url).decode("utf-8")
            except Exception as e:  # noqa: BLE001
                print(f"wiki {lang} {title}: {e}", file=sys.stderr)
                time.sleep(3.0)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(body)
            print(f"wiki {lang} {title}: {len(body)} chars", file=sys.stderr, flush=True)
            time.sleep(1.5)                 # wikimedia rate limits anonymous clients
    for gid in GUTENBERG:
        out = os.path.join(dirpath, "gutenberg", f"{gid}.txt")
        if os.path.exists(out):
            continue
        try:
            body = get(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", 300).decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"gutenberg {gid}: {e}", file=sys.stderr)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(body)
        print(f"gutenberg {gid}: {len(body)} chars", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--short", type=int, default=2000)
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--exh", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--real", default=None, help="comma-separated directories of utf-8 text")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fetch", default=None)
    args = ap.parse_args()
    if args.fetch:
        fetch(args.fetch)
        return
    import multiprocessing as mp
    t0 = time.time()
    tot = {}
    examples = []
    done = 0
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init) as pool:
        for res in pool.imap_unordered(_check, jobs(args), chunksize=1):
            s = tot.setdefault(res["src"], {"cases": 0, "pieces": 0, "piece_mism": 0, "ref_err": 0,
                                           "mism": {k: 0 for k in MODES}})
            s["cases"] += res["cases"]
            s["pieces"] += res["pieces"]
            s["piece_mism"] += res["piece_mism"]
            s["ref_err"] += res["ref_err"]
            for k in MODES:
                s["mism"][k] += res["mism"][k]
            for ex in res["examples"]:
                if len(examples) < 40:
                    examples.append(dict(ex, src=res["src"]))
            done += 1
            if done % 200 == 0 or res["src"] == "long":
                print(f"[{time.time() - t0:7.0f} s] {done} jobs, " + ", ".join(
                    f"{k} {v['cases']}" for k, v in tot.items()), file=sys.stderr, flush=True)
    bad = sum(sum(s["mism"].values()) + s["piece_mism"] for s in tot.values())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "totals": tot, "examples": examples, "seconds": time.time() - t0,
                       "mismatches": bad}, f, indent=1)
    print(f"{'source':8s} {'cases':>10s} " + " ".join(f"{k:>9s}" for k in MODES) + f" {'pieces':>10s} {'p.mism':>7s}"
          f" {'ref.err':>7s}")
    for src, s in tot.items():
        print(f"{src:8s} {s['cases']:10d} " + " ".join(f"{s['mism'][k]:9d}" for k in MODES) +
              f" {s['pieces']:10d} {s['piece_mism']:7d} {s['ref_err']:7d}")
    print(f"total cases {sum(s['cases'] for s in tot.values())}, mismatches {bad}, {time.time() - t0:.0f} s")
    for ex in examples[:10]:
        print(json.dumps(ex, ensure_ascii=True)[:1500].replace("<|", "<\\u007c"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
