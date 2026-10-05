"""Deterministic case generators for the toks parity harness.

Ported (ideas, not code) from the engine's tok v1 parity suite
(read-only reference; toks never imports e — maintainer doctrine). All generators are seeded and
reproducible; `quick=True` (default) keeps the whole run under ~1 min on the control-plane
Mac; `quick=False` is the full size for the CI runners.

The oracle side needs NO binary: generators return plain str / bytes / id lists, so the
same cases later feed both the hf oracle (this tranche) and the toks runner (later tranche).
"""
from __future__ import annotations

import random
import unicodedata
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .oracle import AddedToken  # noqa: F401

SCALARS = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

# e's tok v1 13 contexts (x{}y, {}{}, ' {}', '{} ', '  {}a', 1{}, '{}, '{}x, \n{}\n,
# {}  x, a{}+U+0301, {}+U+0316+U+0301+b, and an added-token literal with the char inside)
CONTEXTS = [
    "x{}y",
    "{}{}",
    " {}",
    "{} ",
    "  {}a",
    "1{}",
    "'{}",
    "'{}x",
    "\n{}\n",
    "{}  x",
    "a{}\u0301",
    "{}\u0316\u0301b",
]


def added_literal_context(literal: str, ch: str) -> str:
    """The 13th context: an added-token literal with the char inside (literal split around ch)."""
    k = len(literal) // 2
    return literal[:k] + ch + literal[k:]


# ---------------------------------------------------------------- pools


def _categories() -> Dict[str, List[str]]:
    cat: Dict[str, List[str]] = {}
    for cp in SCALARS:
        cat.setdefault(unicodedata.category(chr(cp)), []).append(chr(cp))
    return cat


def interesting_pools() -> List[List[str]]:
    cat = _categories()
    tc = [chr(c) for c in list(range(32, 127)) + [9, 10, 13, 0xA0, 0x301, 0x3000, 0x4E2D, 0x1F600,
                                                  0x0661, 0x2028, 0x17F, 0xFB01, 0x200B, 0x85, 0xB2, 0x2160]]
    ws = [chr(c) for c in [9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B))
          + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]]
    marks = cat["Mn"] + cat["Mc"] + cat["Me"]
    letters = cat["Lu"] + cat["Ll"] + cat["Lo"][:5000] + cat["Lm"]
    nums = cat["Nd"] + cat["No"] + cat["Nl"]
    punct = [c for k in ("Po", "Ps", "Pe", "Pd", "Pc", "Pi", "Pf", "Sm", "Sc", "Sk", "So", "Cc", "Cf")
             for c in cat[k]][:20000]
    allp = [chr(c) for c in SCALARS]
    ascii_ = [chr(c) for c in range(32, 127)] + ["\n", "\t", "'"]
    mixed = ascii_ * 20 + ws * 3 + marks[:300] + letters[:500] + nums[:100] + punct[:300]
    fold = list("'sStTmMdDlLrReEvV") + ["\u017f", "\u212a", "\u00df", "\u1e9e", "\ufb05", "\ufb06", "x", " "]
    cjk = [chr(c) for c in range(0x4E00, 0x9FFF)]
    emoji = [chr(c) for c in list(range(0x1F300, 0x1F650)) + list(range(0x1F900, 0x1FA00))
            + list(range(0x2600, 0x27C0))]
    return [tc, ws + ascii_, marks + letters[:200], mixed, allp, ascii_ + ["'s", "'LL", "'re"] * 5,
            ws + nums + punct[:200] + ["\r\n"], fold, cjk, emoji]


def rand_texts(rng: random.Random, n: int, pools: Sequence[List[str]], lo: int = 1, hi: int = 40) -> List[str]:
    out = []
    for _ in range(n):
        pool = rng.choice(pools)
        out.append("".join(rng.choice(pool) for _ in range(rng.randint(lo, hi))))
    return out


# ---------------------------------------------------------------- generators
# each generator is a function (quick: bool, **kw) -> list of cases; cases are
#   str                for encode / pieces (valid utf-8 domain)
#   bytes              invalid-utf-8 byte strings (differential domain: no hf oracle)
#   tuple('ids', [..]) id sequences for decode / stream


def g_golden() -> List[str]:
    """Golden strings: short, hand-picked, cover every interesting class at least once."""
    base = [
        "", "a", " hello", "hello ", "Hello, world!", "\n", "\r\n", "\t\t", "  lead", "trail  ",
        "e\u0301", "\u0301", "a\u0301\u0316\u0301b", "\uFDFA", "\uFB01", "\u017f", "S\u017f",
        "\u212a", "\u00df", "İ", "ı", "中", "日本語", "한국어", "\U0001F600", "\U0001F469\u200D\U0001F4BB",
        "\U0001F1FA\U0001F1F8", "1️⃣", "ﬀ", "̀", "a" * 300, "=" * 130, "0" * 64,
        "x" * 63 + "y", " \n", "\n ", "  \n  x  ", "don't", "DON'T", "y'all'd've",
        "1234567890", "٣١٤", "½", "Ⅻ", "1st 2nd", "3.14159", "e\u00A0n",
        "a\r\nb", "a\x0bb", "a\x0cb", "​", "a​b", "­", "\U000E0001",
    ]
    return base


def g_codepoints(quick: bool, sample: Optional[int] = None, extra_literal: Optional[str] = None) -> List[str]:
    """Every unicode scalar in the 13 contexts (a sampling knob for quick runs)."""
    ctxs = CONTEXTS[:3] if quick else CONTEXTS
    cps = SCALARS
    if sample:
        rng = random.Random(12345)
        cps = sorted(rng.sample(SCALARS, min(sample, len(SCALARS))))
    texts: List[str] = []
    for f in ctxs:
        for cp in cps:
            ch = chr(cp)
            texts.append(f.format(ch, ch))
    if extra_literal:
        # the 13th context: the char inside an added-token literal
        for cp in cps[:: 7 if quick else 1]:
            texts.append(added_literal_context(extra_literal, chr(cp)))
    return texts


def g_random(quick: bool, n: Optional[int] = None) -> List[str]:
    rng = random.Random(7)
    pools = interesting_pools()
    n = n or (8000 if quick else 200000)
    texts = rand_texts(rng, n, pools)
    texts += rand_texts(rng, (n // 20) if quick else n // 20, pools, 40, 2000)
    return texts


def g_ws(quick: bool) -> List[str]:
    rng = random.Random(11)
    ws = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x85", "\xa0", " "] + \
        [chr(c) for c in range(0x2000, 0x200B)] + [" ", " ", " ", " ", "　"]
    tails = ["", "a", "1", "!", "中", "'s", "\u0301", "\U0001F600", " x", "\n"]
    texts = []
    lens = list(range(1, 80)) + [100, 127, 128, 129, 200, 255, 256, 300, 1000] + \
        ([5000] if not quick else [])
    for n in lens:
        for c in (" ", "\t", "\n", "　", "\xa0", " \n", "\r\n", "\n "):
            for t in tails:
                texts.append(c * n + t)
                texts.append("x" + c * n + t)
    for _ in range(2000 if quick else 20000):
        texts.append("".join(rng.choice(ws + ["a", "1", ".", "\r\n"]) for _ in range(rng.randint(1, 60))))
    return texts


def g_scan(quick: bool) -> List[str]:
    """Class runs of length 0..129 across 64-byte boundaries, sprinkled with non-ascii."""
    rng = random.Random(31)
    atoms = (list("abxstmdrevlSTMDREVLIq") * 3 + list("0123456789") + [" "] * 12 +
             ["  ", "\t", "\n", "\n", "\r", "\r\n", "\x0b", "\x0c", " \n", "\n "] * 2 +
             list(".,'\"-()!#;:_/") * 2 +
             ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'Re", "'x", "''", " '", "'\n"] +
             ["é", "’", "中", "\xa0", "ſ", "'ſ", "\u0301", "\u3000", "\x85"])
    runs = [" ", "\n", "\t", " \n", "\r\n", "a", "Z", ".", "'", "1", "-"]
    n1 = 4000 if quick else 300000

    def gen(n):
        out = []
        for _ in range(n):
            if rng.random() < 0.08:
                out.append(rng.choice(runs) * rng.randint(2, 140))
            else:
                out.append(rng.choice(atoms))
        return "".join(out)

    texts = [gen(rng.randint(1, 60)) for _ in range(n1)]
    texts += [gen(rng.randint(200, 800 if quick else 3000)) for _ in range(100 if quick else 3000)]
    texts += ["x" * k + t for k in range(0, 130) for t in ("'s", "'re", " \n x", "\n  \n y", ".\n\n z", "  .", " 1")]
    texts += [r * k + c + t for k in range(0, 66) for r in ("x", " ", ".", "\n", "1", "'", " \n")
              for c in ("é", "’", "中", "\xa0", "ſ", "\u0301", "\u3000", "\x85", "\u0661") for t in ("", " s")]
    return texts


def g_cjk(quick: bool) -> List[str]:
    rng = random.Random(13)
    cjk = [chr(c) for c in range(0x4E00, 0x9FFF)]
    kana = [chr(c) for c in range(0x3041, 0x30FF)]
    hangul = [chr(c) for c in range(0xAC00, 0xD7A4)]
    punct = list("，。！？、：；「」『』（）《》…—") + [" ", "\n", "1", "a"]
    texts = []
    for n in [1, 2, 3, 5, 10, 20, 30, 50, 64, 65, 100, 300] + ([] if quick else [1000, 5000, 20000]):
        for pool in (cjk, kana, hangul, cjk[:300], cjk + kana):
            texts.append("".join(rng.choice(pool) for _ in range(n)))
    for _ in range(4000 if quick else 20000):
        pool = rng.choice([cjk[:2000] + punct, kana + punct, hangul[:3000] + punct, cjk[:500] + kana + punct * 3])
        texts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 200))))
    return texts


def g_emoji(quick: bool) -> List[str]:
    rng = random.Random(17)
    base = [chr(c) for c in list(range(0x1F300, 0x1F650)) + list(range(0x1F900, 0x1FA00))
            + list(range(0x2600, 0x27C0))]
    mods = [chr(c) for c in range(0x1F3FB, 0x1F400)]
    ri = [chr(c) for c in range(0x1F1E6, 0x1F200)]
    parts = base + mods + ri + ["\u200d", "\ufe0f", "\ufe0e", "\u20e3", "#", "1", " ", "\n", "a"]
    fixed = ["\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466", "\U0001F44D\U0001F3FD",
             "\U0001F1FA\U0001F1F8\U0001F1EF\U0001F1F5", "1️⃣", "❤️", "\U0001F3F3️‍\U0001F308",
             "\U0001F469\U0001F3FF‍\U0001F680", "\U0001F9D1‍\U0001F91D‍\U0001F9D1", "\U0001F600" * 100]
    texts = fixed + ["".join(rng.choice(parts) for _ in range(rng.randint(1, 30)))
                     for _ in range(4000 if quick else 30000)]
    return texts


def g_marks(quick: bool) -> List[str]:
    """Combining-mark storms incl. U+0897 and U+1AC1 (NFC-relevant, in tok v1's pool)."""
    rng = random.Random(19)
    marks = [chr(c) for c in SCALARS if unicodedata.combining(chr(c))] + ["\u0897", "\u1ac1"]
    common = [chr(c) for c in list(range(0x300, 0x370)) + list(range(0x1DC0, 0x1E00))
              + list(range(0x20D0, 0x20F1))] + ["\u0897", "\u1ac1"]
    bases = list("aeiouAEIOUnNcCsSzZ") + [chr(c) for c in (0x3B1, 0x3B5, 0x397, 0x436, 0x915, 0xB95,
                                                          0x1100, 0xAC00, 0xC0, 0x1E00, 0x5D0, 0x627)]
    jamo = [chr(c) for c in list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176))
            + list(range(0x11A8, 0x11C3))]
    texts = []
    for _ in range(6000 if quick else 40000):
        texts.append("".join(rng.choice(bases) + "".join(rng.choice(common if rng.random() < 0.7 else marks)
                                                         for _ in range(rng.randint(0, 6)))
                             for _ in range(rng.randint(1, 8))))
    for n in [31, 32, 33, 64, 200] + ([] if quick else [3000]):
        texts.append("a" + "".join(rng.choice(marks) for _ in range(n)))
        texts.append("e" + "\u0301\u0316" * n + "x")
    for _ in range(2000 if quick else 20000):
        texts.append("".join(rng.choice(jamo + [chr(c) for c in (0xAC00, 0xAC01, 0xB098, 0x20)])
                             for _ in range(rng.randint(1, 20))))
    for _ in range(2000 if quick else 10000):
        texts.append("".join(unicodedata.normalize(rng.choice(["NFC", "NFD", "NFKC"]),
                                                   rng.choice(bases) + rng.choice(common) + rng.choice(common))
                             for _ in range(rng.randint(1, 5))))
    return texts


def g_long(quick: bool) -> List[str]:
    rng = random.Random(23)
    texts = []
    ns = [63, 64, 65, 100, 1000] + ([] if quick else [10000, 100000])
    for n in ns:
        texts += ["a" * n, "=" * n, "ab" * (n // 2), "-" * n + "\n", "0" * n, "é" * n]
        texts += ["\U0001d160" * (n // 4), "\u0344" * (n // 2), "e\u0301" * (n // 3)]
        texts.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n)))
        texts.append("".join(rng.choice("abcdefghijklmnopqrstuvwxyz"
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/") for _ in range(n)))
    return texts


def g_special(quick: bool, added: Sequence[str]) -> List[str]:
    """Added-token literals in all modes: overlaps, nested, shared prefixes, adjacent ws."""
    rng = random.Random(29)
    added = list(added)
    frags = added + ["<", "|", ">", "<|", "|>", "<|im_", "im_start", "</", "<think", "think>",
                     "<|im_start", "<|endoftext", "hello", " ", "\n", "a", "<<", "é", "\u0301",
                     "中文", "<tool_call", "[", "]", "[g", "MASK]", "/", "/no", "think", "<sop",
                     "<arg_", "_key>", "123", "4567"]
    texts = [a for a in added] + [a + a for a in added] + ["<" + a for a in added] + [a[:-1] for a in added if a]
    # shared prefixes & nesting: for each pair of literals where one is a prefix of the other
    for a in added:
        for b in added:
            if a != b and (a.startswith(b) or b.startswith(a)):
                texts += [b + "x" + a, a + b, " " + a + b + " "]
            if a != b and b in a:
                k = a.find(b)
                texts.append(a[:k] + "y" + a[k:])
    n = 4000 if quick else 40000
    texts += ["".join(rng.choice(frags) for _ in range(rng.randint(1, 12))) for _ in range(n)]
    return texts


def g_decode_ids(quick: bool, n_vocab: int, added_ids: Sequence[int], special_ids: Sequence[int]) -> List[List[int]]:
    """Id sequences: broken utf-8 (byte-ish ids), special ids, out-of-range ids, mixed."""
    rng = random.Random(31)
    added = list(added_ids)
    seqs = []
    for _ in range(3000 if quick else 30000):
        k = rng.randint(0, 20)
        s = []
        for _ in range(k):
            r = rng.random()
            if r < 0.05 and added:
                s.append(rng.choice(added))
            elif r < 0.08:
                s.append(rng.randint(n_vocab, n_vocab + 323))  # no string: hf skips
            elif r < 0.5:
                s.append(rng.randint(0, 255))  # byte-ish: broken utf-8 galore
            else:
                s.append(rng.randint(0, n_vocab - 1))
        seqs.append(s)
    seqs += [[], [0], [n_vocab - 1], [n_vocab], [n_vocab + 5], list(range(0, 256))]
    return seqs


def g_invalid_utf8(quick: bool) -> List[bytes]:
    """Invalid utf-8 byte strings — no hf oracle (text encode refuses them); generated
    for the differential (byte-input spec) side and decode(encode(x)) checks."""
    rng = random.Random(37)
    blobs = []
    for _ in range(3000 if quick else 50000):
        n = rng.randint(1, 40)
        blobs.append(bytes(rng.choice([rng.randint(0, 255), rng.randint(0x80, 0xBF),
                                       rng.choice(b"\xc0\xc1\xe0\xed\xf0\xf4\xf5\xff"),
                                       ord("a"), 0xCC, 0x81]) for _ in range(n)))
    blobs += [b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe0\x80\xaf", b"a\xcc", b"\xcc\x81\xcc",
              b"\xff\xcc\x81", bytes([0xC0, 0xAF, 0x0E]), b"\x80", b"\xbf", b"\xc0", b"\xc1",
              b"\xf5\x80\x80\x80", b"\x00", b"a\x00b", b"\xc3"]
    return blobs


def g_stream_partitions(rng: random.Random, data: bytes, max_parts: int = 5) -> List[bytes]:
    """Random partition of a byte string into 1..max_parts pushes (returned as a flat
    list of consecutive byte strings = one stream's pushes)."""
    n = len(data)
    if n == 0:
        return [b""]
    k = rng.randint(1, min(max_parts, n))
    cuts = sorted(rng.sample(range(1, n), k - 1)) if k > 1 else []
    bounds = [0] + cuts + [n]
    return [data[i:j] for i, j in zip(bounds, bounds[1:])]


ALL_TEXT_GENERATORS = {
    "golden": lambda quick: g_golden(),
    "codepoints": lambda quick: g_codepoints(quick),
    "random": lambda quick: g_random(quick),
    "ws": lambda quick: g_ws(quick),
    "scan": lambda quick: g_scan(quick),
    "cjk": lambda quick: g_cjk(quick),
    "emoji": lambda quick: g_emoji(quick),
    "marks": lambda quick: g_marks(quick),
    "long": lambda quick: g_long(quick),
}


def generate_all(quick: bool = True, added: Optional[Sequence[str]] = None,
                 n_vocab: int = 0, added_ids: Sequence[int] = (), special_ids: Sequence[int] = ()) -> dict:
    """Everything in one dict: {'texts': {name: [str]}, 'bytes': [bytes], 'ids': [[int]]}."""
    out: dict = {"texts": {name: g(quick) for name, g in ALL_TEXT_GENERATORS.items()}}
    if added:
        out["texts"]["special"] = g_special(quick, added)
    out["bytes"] = g_invalid_utf8(quick)
    if n_vocab:
        out["ids"] = g_decode_ids(quick, n_vocab, added_ids, special_ids)
    return out
