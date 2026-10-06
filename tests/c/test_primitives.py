#!/usr/bin/env python3
"""tests/c/test_primitives.py: writes test_primitives.inc next to it, what test_primitives.c checks the parity
primitives against (include/toks.h: TOKS_NO_TRUNCATE / TOKS_NO_PAD, toks_template, toks_added, TOKS_DECODE_RAW,
toks_info's truncation / padding / template fields).

    uv run --python 3.12 --with tokenizers==0.23.2 --with tiktoken==0.14.0 --with transformers==5.18.0 \\
        python tests/c/test_primitives.py [--show NAME]

The references, per primitive (python/toks_oracle/primitives.py has the hf ones):
  template    hf num_special_tokens_to_add(False); post_process() of the empty encoding (no padding, no truncation):
              the ids and type_ids; the text's type id and the prefix count from post_process() of "hello"
  added       hf get_added_tokens_decoder(): id, content, special, lstrip, rstrip, single_word, normalized
  info        hf Tokenizer.truncation / .padding
  no_trunc    hf encode(text, add_special_tokens) after no_truncation() / no_padding() (each, both, neither)
  raw         tiktoken Encoding.decode_bytes (kimik3); ByteLevel files: hf's ByteLevel decoder before its lossy
              step (each token's chars through the byte table, else its utf-8); ByteFallback chains: hf decode with
              each U+FFFD its ByteFallback wrote for a run that is not utf-8 replaced by the byte it stood for
Files: every tokenizer.json under tests/data and under $TOKS_TOKENIZER_CACHE (default ~/.cache/toks/tokenizers),
each pinned by its sha-256 (the C test skips a file that is absent or not these bytes), and the Kimi K3 directory
($TOKS_KIMI_DIR, default ~/.cache/toks/kimik3) through tests/parity/oracle_tiktoken.py. Ids and expected bytes are
kept as FNV-1a 64 digests (test_primitives.c computes the same).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))

from toks_oracle import primitives as P  # noqa: E402

CACHE = os.environ.get("TOKS_TOKENIZER_CACHE", os.path.expanduser("~/.cache/toks/tokenizers"))
KIMI = os.environ.get("TOKS_KIMI_DIR", os.path.expanduser("~/.cache/toks/kimik3"))

# the texts (test_primitives.c builds T3 the same way: T2 100 times, past 8,196 ids); T4 is per file (its specials)
T2 = ("The quick brown fox jumps over the lazy dog, twice: 1234567 and 3.14159!  Grüße aus Köln, ça va?\n"
      "def f(x):\n\treturn x ** 2  # square\n"
      "日本語のテキスト、中文文本。한국어 텍스트. Emoji 😀👍🏽 and 👨\u200d👩\u200d👧 families.\n"
      "    indented   runs   of   spaces\t\ttabs\r\nand CRLF, don't DON'T 'S 'll I'VE.\n"
      "Ελληνικά, русский текст, עברית, العربية, हिन्दी.\n")
T3_REPEAT = 100
TEXTS = ["", "Hello, world! It's 2026.", T2, T2 * T3_REPEAT]
FLAG_SETS = (0, 16, 32, 48, 4, 20, 36, 52)      # TOKS_NO_POSTPROCESS 4, TOKS_NO_TRUNCATE 16, TOKS_NO_PAD 32


def fnv(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def ids_bytes(ids) -> bytes:
    return b"".join(int(i).to_bytes(4, "little") for i in ids)


def cstr(b: bytes) -> str:
    out, col = ['"'], 0
    for c in b:
        if c in (0x22, 0x5C):
            out.append("\\" + chr(c))
        elif 0x20 <= c < 0x7F and c != 0x3F:
            out.append(chr(c))
        else:
            out.append("\\%03o" % c)
        col += 1
        if col >= 64 and c < 0x80:
            out.append('"\n        "')
            col = 0
    out.append('"')
    return "".join(out)


def write_fixtures():
    """tests/data/primitives/types_left_pad.json: what no cached file has, so CI sees it too: a single template whose
    specials and text carry type ids 2 / 1 / 3, Fixed padding on the Left with pad_type_id 5 and a multiple of 4,
    truncation to 8 (WordPiece, a 90-entry vocabulary)."""
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    vocab += ["##" + chr(c) for c in range(ord("a"), ord("z") + 1)] + [str(d) for d in range(10)]
    vocab += ["##" + str(d) for d in range(10)] + [".", ",", "!", "?", "'", "hello", "world", "the", "fox", "dog"]

    def added(i, c, **kw):
        a = {"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False,
             "special": True}
        a.update(kw)
        return a
    j = {
        "version": "1.0",
        "truncation": {"direction": "Right", "max_length": 8, "strategy": "LongestFirst", "stride": 0},
        "padding": {"strategy": {"Fixed": 13}, "direction": "Left", "pad_to_multiple_of": 4, "pad_id": 0,
                    "pad_type_id": 5, "pad_token": "[PAD]"},
        "added_tokens": [added(0, "[PAD]"), added(1, "[UNK]"), added(2, "[CLS]"), added(3, "[SEP]"),
                         added(4, "[MASK]", lstrip=True)],
        "normalizer": {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True, "strip_accents": None,
                       "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 2}}, {"Sequence": {"id": "A", "type_id": 1}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 3}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 2}}, {"Sequence": {"id": "A", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 3}}, {"Sequence": {"id": "B", "type_id": 4}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 4}}],
            "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
                               "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}}},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 100, "vocab": {t: i for i, t in enumerate(vocab)}},
    }
    d = os.path.join(ROOT, "tests", "data", "primitives")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "types_left_pad.json"), "w", encoding="utf-8") as fh:
        json.dump(j, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    # tests/data/primitives/uni_fixed_pad.json: Unigram with Fixed padding (to 20, a multiple of 8: 24) on the Left and
    # truncation to 16, around a <s> $A </s> template: tests/data/unigram/bound_bf_meta.json (Metaspace, byte fallback:
    # a stack with certified cuts) with <s>, </s>, <pad> appended as pieces and specials
    with open(os.path.join(ROOT, "tests", "data", "unigram", "bound_bf_meta.json"), encoding="utf-8") as fh:
        u = json.load(fh)
    n = len(u["model"]["vocab"])
    u["model"]["vocab"] += [["<s>", 0.0], ["</s>", 0.0], ["<pad>", 0.0]]
    u["added_tokens"] += [added(n, "<s>"), added(n + 1, "</s>"), added(n + 2, "<pad>")]
    u["truncation"] = {"direction": "Right", "max_length": 16, "strategy": "LongestFirst", "stride": 0}
    u["padding"] = {"strategy": {"Fixed": 20}, "direction": "Left", "pad_to_multiple_of": 8, "pad_id": n + 2,
                    "pad_type_id": 0, "pad_token": "<pad>"}
    u["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                 {"SpecialToken": {"id": "</s>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                 {"SpecialToken": {"id": "</s>", "type_id": 1}}],
        "special_tokens": {"<s>": {"id": "<s>", "ids": [n], "tokens": ["<s>"]},
                           "</s>": {"id": "</s>", "ids": [n + 1], "tokens": ["</s>"]}}}
    with open(os.path.join(d, "uni_fixed_pad.json"), "w", encoding="utf-8") as fh:
        json.dump(u, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def tokenizer_files():
    """(name, path, fixture) of every tokenizer.json: the fixtures under tests/data, then the cache's."""
    write_fixtures()
    out = []
    for d, _, fs in sorted(os.walk(os.path.join(ROOT, "tests", "data"))):
        for f in sorted(fs):
            if f.endswith(".json"):
                p = os.path.join(d, f)
                out.append((os.path.relpath(p, ROOT), p, 1))
    if os.path.isdir(CACHE):
        for f in sorted(os.listdir(CACHE)):
            p = os.path.join(CACHE, f)
            if os.path.isfile(p):
                out.append((f, p, 0))
    keep = []
    for name, p, fx in out:
        try:
            with open(p, "rb") as fh:
                src = fh.read()
            j = json.loads(src)
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        if isinstance(j, dict) and "model" in j:
            keep.append((name, p, fx, src, j))
    return keep


def raw_cases_hf(name, tok, j, rnd):
    """ByteLevel or ByteFallback files: (flags, ids, expected bytes), else []."""
    kind = P.decoder_kind(j)
    if kind is None:
        return []
    n_ids = tok.get_vocab_size(with_added_tokens=True)
    strings = {}

    def s(i):
        if i not in strings:
            strings[i] = tok.id_to_token(i)
        return strings[i]

    pool = [i for i in range(n_ids) if s(i) is not None and "\ufffd" not in s(i)]
    if not pool:
        return []
    cases = []
    if kind == "bytelevel":
        high = [i for i in pool if len(P.bytelevel_bytes(s(i))) == 1 and P.bytelevel_bytes(s(i))[0] >= 0x80]
        for _ in range(8):
            ids = [rnd.choice(high if (high and rnd.random() < 0.4) else pool) for _ in range(rnd.randint(1, 12))]
            cases.append((2, ids, b"".join(P.bytelevel_bytes(s(i)) for i in ids)))
        return cases
    byte_id = {}
    for i in pool:
        b = P.byte_fallback(s(i))
        if b is not None and b not in byte_id:
            byte_id[b] = i
    if not byte_id:
        return []
    plain = [i for i in pool if P.byte_fallback(s(i)) is None]
    runs = [b"\xe2\x96", b"\xe2\x96\x81", b"\xc3\xa9", b"\xf0\x9f\x98\x80", b"\xf0\x9f\x98", b"\x80", b"\xff", b"\xc0\xaf",
            b" \xff", b"\n\xc3", b"A\xe2\x82\xac", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xc3", b"\xbf\xbf"]
    tries = 0
    singles = sorted(byte_id.values())
    while len(cases) < 16 and tries < 400:
        tries += 1
        ids = []
        for _ in range(rnd.randint(1, 5)):
            u = rnd.random()
            if u < 0.45:
                r = rnd.choice(runs)
                if all(b in byte_id for b in r):
                    ids += [byte_id[b] for b in r]
            elif u < 0.65:                                  # whatever bytes the vocab has, 1 to 4 in a row
                ids += [rnd.choice(singles) for _ in range(rnd.randint(1, 4))]
            elif plain:
                ids.append(rnd.choice(plain))
        if not ids:
            continue
        got = P.hf_decode_raw(tok, ids, [s(i) for i in ids])
        if got is not None:
            cases.append((2, ids, got))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", default="", help="print one file's records (hf's side of a mismatch) and exit")
    ap.add_argument("--out", default=os.path.join(HERE, "test_primitives.inc"))
    args = ap.parse_args()
    import tokenizers
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit(f"tokenizers {tokenizers.__version__}, want 0.23.2")
    rnd = random.Random(20261006)
    files = []
    for name, path, fixture, src, j in tokenizer_files():
        rec = P.records(src, j, TEXTS, FLAG_SETS)
        if rec is None:
            print(f"  {name}: hf refuses it (skipped)", file=sys.stderr)
            continue
        rec["name"], rec["fixture"], rec["sha"] = name, fixture, hashlib.sha256(src).hexdigest()
        rec["raw"] = raw_cases_hf(name, rec["tok"], j, rnd)
        if args.show == name:
            P.show(rec)
            return
        files.append(rec)
    kimi = P.kimi_records(KIMI, TEXTS, rnd) if os.path.isdir(KIMI) else None
    if kimi is not None:
        kimi["name"], kimi["fixture"] = "kimik3", 2
        if args.show == "kimik3":
            P.show(kimi)
            return
        files.append(kimi)
    write(args.out, files)


def write(path, files):
    o = []
    o.append("/* generated by tests/c/test_primitives.py from hf tokenizers 0.23.2, tiktoken 0.14.0 and transformers 5.18.0"
             " (kimik3); do not edit */")
    o.append("#define PR_N_TEXTS %d                         /* + T4, the file's own: PR_FILE.text4 */" % len(TEXTS))
    o.append("static const char PR_T1[] = %s;" % cstr(TEXTS[1].encode()))
    o.append("static const char PR_T2[] = %s;" % cstr(TEXTS[2].encode()))
    o.append("#define PR_T3_REPEAT %du" % T3_REPEAT)
    o.append("static const uint32_t PR_FLAGS[%d] = { %s };" % (len(FLAG_SETS), ", ".join("%du" % f for f in FLAG_SETS)))
    o.append("")
    for k, f in enumerate(files):
        tids = f["tmpl_ids"]
        o.append("static const uint32_t PR_TI%d[] = { %s };" % (k, ", ".join("%du" % x for x in tids) or "0u"))
        o.append("static const uint32_t PR_TT%d[] = { %s };" % (k, ", ".join("%du" % x for x in f["tmpl_types"]) or "0u"))
        enc = f["enc"]
        o.append("static const pr_enc PR_E%d[] = {%s};" % (k, " ".join(
            "{ %du, %du, %du, 0x%016xull }," % (t, fl, n, d) for (t, fl, n, d) in enc) or " { 0u, 0u, 0u, 0ull } "))
        raws = f["raw"]
        for r, (fl, ids, exp) in enumerate(raws):
            o.append("static const uint32_t PR_R%d_%d[] = { %s };" % (k, r, ", ".join("%du" % x for x in ids)))
        o.append("static const pr_raw PR_RAW%d[] = {%s};" % (k, " ".join(
            "{ PR_R%d_%d, %du, %du, %du, 0x%016xull }," % (k, r, len(ids), fl, len(exp), fnv(exp))
            for r, (fl, ids, exp) in enumerate(raws)) or " { NULL, 0u, 0u, 0u, 0ull } "))
    o.append("")
    o.append("static const pr_file PR_FILE[] = {")
    for k, f in enumerate(files):
        info = f["info"]
        o.append("    { %s, %d, \"%s\"," % (cstr(f["name"].encode()), f["fixture"], f.get("sha", "")))
        o.append("      %du, %du, %du, PR_TI%d, PR_TT%d," % (len(f["tmpl_ids"]), f["n_prefix"], f["seq_type"], k, k))
        o.append("      { %s }," % ", ".join("%du" % info[x] for x in P.INFO_FIELDS))
        o.append("      %du, 0x%016xull, %s, %du, 0x%016xull," % (f["n_added"], f["added_digest"], cstr(f["text4"].encode()),
                                                             f["empty"][0], f["empty"][1]))
        o.append("      PR_E%d, %du, PR_RAW%d, %du }," % (k, len(f["enc"]), k, len(f["raw"])))
    o.append("};")
    o.append("#define PR_N_FILES %du" % len(files))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    n_enc = sum(len(f["enc"]) for f in files)
    n_raw = sum(len(f["raw"]) for f in files)
    n_add = sum(f["n_added"] for f in files)
    print(f"test_primitives.inc: {len(files)} files, {n_add} added tokens, {n_enc} encode cases, {n_raw} raw decode "
          f"cases", file=sys.stderr)


if __name__ == "__main__":
    main()
