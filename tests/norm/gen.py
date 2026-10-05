# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""NFC differential cases with hf tokenizers 0.23.2 as the oracle, for tests/norm/nfc_check.c.

Writes records to stdout: u32 n_in, utf-8 in, u32 n_out, utf-8 hf NFC(in) (little-endian). Items inside a
record are joined by "\\n" (a starter that composes with nothing), so a record is many cases at once. The case
sets come from the crate tables tools/gen/norm.py parses and verifies; the expected bytes always come from hf.

families (python tests/norm/gen.py FAMILY [--corpus DIR] [--seed N]):
  alone       every scalar value alone
  pairs       every scalar a followed by every composition second b (110 of them) -- 122 M cases
  marks2      base + m1 + m2 for every ordered pair of the 814 non-starters, 7 bases
  blocked     every composition pair (a, b) with a mark of every class between them, doubled marks, b twice
  hangul      every L V T triple (incl. T_BASE and non-composing jamo), every syllable + every T, syllables +
              V, L + syllables, jamo soup
  normtest    NormalizationTest.txt of unicode 9.0.0 (sha256 pinned), every column as an input
  corpus      the documents under --corpus (tests/norm/corpus.py) as they are, in NFD, NFKD, NFD with every
              mark run shuffled, and zalgo-ized (marks of every class after letters)
  adversarial long combining runs (10 .. 1 M marks: one class, mixed, mixed with seconds) behind 16 kinds of
              base, random strings over the NB set, zalgo storms
"""

import argparse
import hashlib
import json
import random
import struct
import sys
import unicodedata
import urllib.request
from pathlib import Path

import tokenizers
from tokenizers import normalizers

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gen"))
import norm as G  # noqa: E402  (tools/gen/norm.py: the crate tables)

NFC = normalizers.NFC()
OUT = sys.stdout.buffer
SCALARS = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
NORMTEST_URL = "https://www.unicode.org/Public/9.0.0/ucd/NormalizationTest.txt"
NORMTEST_SHA256 = "2d48d848656b3cf889df59980ab13551988950c4ca8c5190a17c691a17f8b9bb"

# --oracle: whose normalizer gives the expected bytes. "nfc" is normalizers.NFC(); a model name is that model's
# own tokenizer.json (pinned revision + sha256) through Tokenizer.from_file(...).normalizer.
ORACLES = {
    "qwen38": ("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json",
               "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
}


def model_normalizer(name):
    url, pin = ORACLES[name]
    path = Path("build/norm/%s.tokenizer.json" % name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != pin:
        raise SystemExit("gen.py: %s sha256 %s != pinned %s" % (path, hashlib.sha256(blob).hexdigest(), pin))
    if json.loads(blob)["normalizer"] != {"type": "NFC"}:
        raise SystemExit("gen.py: %s: normalizer is not NFC alone" % name)
    sys.stderr.write("oracle: %s tokenizer.json sha256 %s, its own normalizer (NFC)\n" % (name, pin))
    return tokenizers.Tokenizer.from_file(str(path)).normalizer


def emit(s):
    raw = s.encode("utf-8")
    exp = NFC.normalize_str(s).encode("utf-8")
    OUT.write(struct.pack("<I", len(raw)) + raw + struct.pack("<I", len(exp)) + exp)


def emit_items(items, per=50000):
    for i in range(0, len(items), per):
        emit("\n".join(items[i:i + per]))


def crate_sets():
    """non-starters (by class), composition pairs, seconds and firsts of the crate hf runs."""
    ccc, _decomp, comp, _kdecomp, _marks = G.parse_tables(G.fetch_crate())
    marks = sorted(ccc)
    seconds = sorted({b for (_a, b) in comp} | set(range(0x1161, 0x1176)) | set(range(0x11A8, 0x11C3)))
    firsts = sorted({a for (a, _b) in comp})
    rep = {}
    for m in marks:
        rep.setdefault(ccc[m], m)
    return marks, seconds, firsts, sorted(comp), sorted(rep.values())


def fam_fetch(args):
    """nothing: main() has fetched and checked the --oracle tokenizer.json (run once before parallel jobs)."""


def fam_alone(args):
    emit_items([chr(c) for c in SCALARS])


def fam_pairs(args):
    _m, seconds, _f, _p, _r = crate_sets()
    k, n = (int(x) for x in args.shard.split("/"))
    for b in seconds[k::n]:
        B = chr(b)
        emit("".join(chr(a) + B + "\n" for a in SCALARS))


def fam_marks2(args):
    marks, _s, _f, _p, _r = crate_sets()
    for base in "aA\u03b1\u1f00\u0915\u05d0\u3046":
        emit_items([base + chr(m1) + chr(m2) for m1 in marks for m2 in marks])


def fam_blocked(args):
    _m, _s, _f, pairs, reps = crate_sets()
    items = []
    for a, b in pairs:
        A, B = chr(a), chr(b)
        for m in reps:
            M = chr(m)
            items += [A + M + B, A + M + M + B, A + B + M + B, A + B + B + M]
    emit_items(items)


def fam_hangul(args):
    L = [chr(0x1100 + i) for i in range(19)] + ["\u1113", "\u115f"]
    V = [chr(0x1161 + i) for i in range(21)] + ["\u1160", "\u1176"]
    T = [chr(0x11A7 + i) for i in range(28)] + ["\u11c3"]
    items = [l + v + t for l in L for v in V for t in T]
    syl = [chr(0xAC00 + i) for i in range(11172)]
    items += [s + t for s in syl for t in T]
    items += [s + v for s in syl[::7] for v in V]
    items += [l + s for l in L for s in syl[::13]]
    rnd = random.Random(args.seed)
    soup = L + V + T + syl[::97] + ["\u0301", "\u302e", "a"]
    items += ["".join(rnd.choice(soup) for _ in range(rnd.randint(2, 12))) for _ in range(200000)]
    emit_items(items)


def fam_normtest(args):
    with urllib.request.urlopen(NORMTEST_URL, timeout=120) as r:
        blob = r.read()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != NORMTEST_SHA256:
        raise SystemExit("gen.py: NormalizationTest.txt sha256 %s != pinned %s" % (sha, NORMTEST_SHA256))
    items, agree, total = [], 0, 0
    for line in blob.decode("utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("@"):
            continue
        cols = ["".join(chr(int(h, 16)) for h in c.split()) for c in line.split(";")[:5]]
        items.extend(cols)
        total += 1
        agree += all(NFC.normalize_str(c) == cols[1] for c in cols[:3])
    sys.stderr.write("normtest: hf NFC(c1) == NFC(c2) == NFC(c3) == c2 on %d of %d lines\n" % (agree, total))
    emit_items(items)


def shuffle_marks(s, rnd):
    out, run = [], []
    for ch in s:
        if unicodedata.combining(ch):
            run.append(ch)
            continue
        rnd.shuffle(run)
        out.extend(run)
        run = []
        out.append(ch)
    rnd.shuffle(run)
    out.extend(run)
    return "".join(out)


def zalgo(s, rnd, marks):
    out = []
    for ch in s:
        out.append(ch)
        if ch.isalpha() and rnd.random() < 0.3:
            out.extend(chr(rnd.choice(marks)) for _ in range(rnd.randint(1, 6)))
    return "".join(out)


def fam_corpus(args):
    marks, seconds, _f, _p, _r = crate_sets()
    rnd = random.Random(args.seed)
    forms = args.forms.split(",")
    n = 0
    for path in sorted(Path(args.corpus).glob("*.jsonl")):
        if args.langs and path.stem not in args.langs.split(","):
            continue
        for line in path.open(encoding="utf-8"):
            doc = json.loads(line)
            if "raw" in forms:
                emit(doc)
            if "nfd" in forms:
                emit(unicodedata.normalize("NFD", doc))
            if "nfkd" in forms:
                emit(unicodedata.normalize("NFKD", doc))
            if "shuffled" in forms:
                emit(shuffle_marks(unicodedata.normalize("NFD", doc), rnd))
            if "zalgo" in forms:
                emit(zalgo(doc[:20000], rnd, marks + seconds))
            n += 1
    sys.stderr.write("corpus: %d documents x forms %s\n" % (n, args.forms))


GUTENBERG = (1342, 2701, 84, 1661, 98, 2600, 11, 74)  # en-prose for the bench: public-domain books


def fam_gutenberg(args):
    for book in GUTENBERG:
        url = "https://www.gutenberg.org/cache/epub/%d/pg%d.txt" % (book, book)
        with urllib.request.urlopen(url, timeout=120) as r:
            blob = r.read()
        sys.stderr.write("gutenberg %d: %d bytes sha256 %s\n" % (book, len(blob), hashlib.sha256(blob).hexdigest()))
        emit(blob.decode("utf-8"))


def fam_adversarial(args):
    marks, seconds, firsts, _p, _r = crate_sets()
    rnd = random.Random(args.seed)
    bases = ["", "a", "A", "\u03b1", "\u1f00", "\u0915", "\u05d0", "\uac00", "\u1100", "\U00011099",
             "\u0b47", "\u304b", "\u00c5", "\u212b", "\u0958", "\U0001d160"]
    nb = [chr(c) for c in marks + seconds + firsts]
    for n in (10, 100, 1000, 10000, 100000, 1000000):
        for base in bases:
            emit(base + chr(rnd.choice(marks)) * n + "b")
            emit(base + "".join(chr(rnd.choice(marks)) for _ in range(n)) + "b")
            emit(base + "".join(chr(rnd.choice(marks + seconds)) for _ in range(n)))
    for _ in range(2000):
        emit("".join(rnd.choice(nb) for _ in range(rnd.randint(1, 400))))
    for _ in range(200):
        emit(zalgo("".join(chr(rnd.choice(firsts)) for _ in range(2000)), rnd, marks + seconds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--corpus", default="build/norm-corpus")
    ap.add_argument("--forms", default="raw,nfd,nfkd,shuffled,zalgo")
    ap.add_argument("--langs", default="", help="corpus languages (default: all)")
    ap.add_argument("--shard", default="0/1", help="pairs: seconds k::n")
    ap.add_argument("--oracle", default="nfc", help="nfc (normalizers.NFC) or a pinned model: %s" % ", ".join(ORACLES))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("gen.py: tokenizers %s != 0.23.2" % tokenizers.__version__)
    if args.oracle != "nfc":
        global NFC
        NFC = model_normalizer(args.oracle)
    globals()["fam_" + args.family](args)


if __name__ == "__main__":
    main()
