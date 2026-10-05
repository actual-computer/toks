"""tests/unigram/gen.py: deterministic case generators for the unigram differential (run_diff.py).

Every generator is a pure function of (seed, index): workers regenerate their own slice, nothing is shipped.
Kinds:
  random      mixtures of text units weighted toward what the unigram stack finds hard: CJK (common, rare,
              compatibility ideographs, radicals), kana with combining voiced marks and halfwidth forms, hangul
              syllables / jamo / compatibility jamo, indic conjuncts (InCB), prepend marks, emoji zwj / skin
              tone / flag runs, latin with precomposed and stacked combining marks, fullwidth ascii, every
              White_Space char and the near-misses (ZWSP, BOM, U+180E), C0 / C1 controls and NUL, ligatures and
              other NFKC compatibility chars (math alphanumerics: 4-byte charsmap keys), U+2581, byte-token
              lookalikes, the tokenizer's own added-token strings with spaces around them (lstrip / rstrip),
              unk material (private use, unassigned, noncharacters, rare scripts, tags, variation selectors)
  real        lines of real text (FLORES-200 or any utf-8 files given), mutated with the units above
  exhaustive  every string of length 1..3 over ALPHABET (index order)
  scalars     every unicode scalar value in CONTEXTS (index = scalar * len(CONTEXTS) + context)
"""

from __future__ import annotations

import random

WS = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x85", "\xa0", "\u1680", *[chr(c) for c in range(0x2000, 0x200B)],
      "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"]
NEAR_WS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u180e", "\u2060", "\u00ad", "\u034f"]
COMBINING = [chr(c) for c in range(0x300, 0x370)] + ["\u0483", "\u05b0", "\u064b", "\u093c", "\u0e31", "\u1ab0",
                                                    "\u1dc0", "\u20d0", "\u20e3", "\u302a", "\u3099", "\u309a",
                                                    "\ufe20", "\U0001d165"]
LATIN_PRE = [chr(c) for c in range(0xC0, 0x250) if c not in (0xD7, 0xF7)]
FULLWIDTH = [chr(c) for c in range(0xFF01, 0xFF5F)]
HALFKANA = [chr(c) for c in range(0xFF61, 0xFFA0)]
HIRA = [chr(c) for c in range(0x3041, 0x3097)]
KATA = [chr(c) for c in range(0x30A1, 0x30FB)]
CJK_COMMON = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深区即求品士转量空甚众技轻程告江语英基派满式李息写呢识极令黄德收脸钱党倒未持取设始版双历越史商千片容研像找友孩站广改议形委早房音火际则首单据导影失拿网香似斯专石若兵弟谁校读志飞观争究包组造落视济"
CJK_RARE = ["\u3400", "\u4db5", "\u9fa5", "\u9fff", "\U00020000", "\U0002a6d6", "\U0002b740", "\U00030000",
            "\U00031350"]
CJK_COMPAT = [chr(c) for c in range(0xF900, 0xFA6E)] + [chr(c) for c in range(0x2F800, 0x2F820)]
RADICALS = [chr(c) for c in range(0x2E80, 0x2E9A)] + [chr(c) for c in range(0x2F00, 0x2FD6)]
HANGUL = [chr(c) for c in range(0xAC00, 0xD7A4, 97)]
JAMO_L = [chr(c) for c in range(0x1100, 0x1113)]
JAMO_V = [chr(c) for c in range(0x1161, 0x1176)]
JAMO_T = [chr(c) for c in range(0x11A8, 0x11C3)]
COMPAT_JAMO = [chr(c) for c in range(0x3131, 0x318F)]
DEVA_C = [chr(c) for c in range(0x0915, 0x093A)]
INDIC = ["\u094d", "\u093f", "\u0940", "\u093c", "\u0902", "\u0903", "\u09cd", "\u0995", "\u09b7", "\u0acd",
         "\u0a95", "\u0b4d", "\u0b15", "\u0c4d", "\u0c15", "\u0d4d", "\u0d15", "\u0d7a", "\u1039", "\u1000",
         "\u17d2", "\u1780", "\u0e33", "\u0e01", "\u0e48", "\u0eb3"]
PREPEND = ["\u0600", "\u0601", "\u0602", "\u0603", "\u0604", "\u0605", "\u06dd", "\u070f", "\u0890", "\u08e2",
           "\u0d4e", "\U000110bd", "\U000111c2"]
ARABIC = [chr(c) for c in range(0x0621, 0x064B)] + [chr(c) for c in range(0xFB50, 0xFB60)] + \
         [chr(c) for c in range(0xFE70, 0xFE80)] + ["\ufdfa", "\ufdfb", "\ufdf2"]
EMOJI = ["\U0001F600", "\U0001F469", "\U0001F468", "\U0001F467", "\u2764", "\u2764\ufe0f", "\U0001F44D",
         "\U0001F3FB", "\U0001F3FF", "\u200d", "\U0001F1FA", "\U0001F1F8", "\U0001F1EB", "\U0001F1F7", "\u00a9",
         "#\ufe0f\u20e3", "\U0001F3F4\U000E0067\U000E0062\U000E007F", "\u261d", "\U0001F9D1\u200d\U0001F4BB"]
COMPAT = ["\ufb00", "\ufb01", "\ufb02", "\ufb03", "\ufb04", "\ufb05", "\ufb06", "\u2460", "\u2473", "\u3392",
          "\u337f", "\u2163", "\u216b", "\u00b2", "\u00bd", "\u2153", "\u00aa", "\u2122", "\u2026", "\u2025",
          "\u00b5", "\u212b", "\u2126", "\U0001d400", "\U0001d41a", "\U0001d7ce", "\U0001d7ff", "\u2474", "\u2488",
          "\u24b6", "\u3200", "\u3250", "\u32ff", "\u33ff", "\u2100", "\u2103", "\u2109", "\u017f", "\u1e9b",
          "\u0149", "\u01c4", "\u01c5", "\u2154", "\u215f", "\u2189", "\u3000", "\u2002"]
CONTROL = ["\x00", "\x01", "\x02", "\x07", "\x08", "\x0e", "\x1b", "\x1f", "\x7f", "\x80", "\x8f", "\x9f"]
UNK = ["\ue000", "\uf8ff", "\U000f0000", "\U0010fffd", "\u0378", "\u0560", "\ufffe", "\uffff", "\ufdd0",
       "\U0001fffe", "\U00010000", "\U00012000", "\U00013000", "\U0001e900", "\U000e0001", "\U000e0020",
       "\ufe00", "\ufe0f", "\U000e0100", "\u1d00", "\u2c00", "\ua000", "\U00016800", "\U0001b000",
       "\U0001f000", "\U0001fa70", "\u0d00", "\u10fc", "\u19b0", "\u2e3b"]
META = ["\u2581", "\u2581\u2581", "\u2581 ", " \u2581", "<0x41>", "<0x0A>", "<0xE3><0x81>", "<0xZZ>", "<unk>",
        "<0x+1>"]
ASCII_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
WORDS = ["the", "of", "and", "Hello", "world", "token", "Tokyo", "naïve", "café", "über", "Straße", "école",
         "ﬁnance", "ＡＢＣ", "ｶﾀｶﾅ", "東京", "日本語", "한국어", "русский", "ελληνικά", "हिन्दी", "العربية", "עברית",
         "ไทย", "tiếng", "Việt", "x", "I", "a", "```", "''", "``", "--", "...", "http://example.com/a?b=c",
         "user@example.org", "2026-10-03", "3.14159", "1,000,000", "$5", "100%", "C++", "#include", "<html>",
         "</div>", "{\"k\": [1, 2]}", "e\u0301", "A\u030a", "\u212b", "ﾞ", "か\u3099", "\u1100\u1161\u11a8"]


def _rand_word(rng: random.Random) -> str:
    n = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20])
    alpha = rng.choice(["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                        "0123456789", "aeiou", "etaoinshrdlu"])
    return "".join(rng.choice(alpha) for _ in range(n))


def _ws_run(rng: random.Random) -> str:
    n = rng.choice([1, 1, 1, 2, 2, 3, 4, 7])
    pool = rng.choice([[" "], [" ", " ", "\n"], ["\r\n"], WS, WS + NEAR_WS, ["\t", " "], ["\u3000"], ["\xa0"]])
    return "".join(rng.choice(pool) for _ in range(n))


def _cjk(rng: random.Random) -> str:
    n = rng.choice([1, 2, 3, 5, 8, 16])
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.75:
            out.append(rng.choice(CJK_COMMON))
        elif r < 0.85:
            out.append(rng.choice(CJK_RARE))
        elif r < 0.93:
            out.append(rng.choice(CJK_COMPAT))
        else:
            out.append(rng.choice(RADICALS))
        if rng.random() < 0.05:
            out.append(rng.choice("。、，！？「」（）・：；"))
    return "".join(out)


def _kana(rng: random.Random) -> str:
    out = []
    for _ in range(rng.choice([1, 2, 4, 7])):
        r = rng.random()
        if r < 0.4:
            out.append(rng.choice(HIRA))
        elif r < 0.7:
            out.append(rng.choice(KATA))
        else:
            out.append(rng.choice(HALFKANA))
        if rng.random() < 0.2:
            out.append(rng.choice(["\u3099", "\u309a", "\uff9e", "\uff9f", "\u309b", "\u309c"]))
    return "".join(out)


def _hangul(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return "".join(rng.choice(HANGUL) for _ in range(rng.choice([1, 2, 3, 5])))
    if r < 0.8:
        s = rng.choice(JAMO_L) + rng.choice(JAMO_V)
        if rng.random() < 0.5:
            s += rng.choice(JAMO_T)
        if rng.random() < 0.2:
            s = rng.choice(JAMO_L) + s
        return s
    return "".join(rng.choice(COMPAT_JAMO) for _ in range(rng.choice([1, 2, 3])))


def _indic(rng: random.Random) -> str:
    out = [rng.choice(DEVA_C)]
    for _ in range(rng.choice([0, 1, 2, 3, 4])):
        r = rng.random()
        if r < 0.4:
            out.append("\u094d")
            out.append(rng.choice(DEVA_C))
        elif r < 0.7:
            out.append(rng.choice(INDIC))
        else:
            out.append(rng.choice(DEVA_C))
    return "".join(out)


def _latin(rng: random.Random) -> str:
    out = []
    for _ in range(rng.choice([1, 2, 4, 6])):
        r = rng.random()
        if r < 0.4:
            out.append(rng.choice(LATIN_PRE))
        elif r < 0.8:
            out.append(rng.choice("aeiouyAEIOUcnsz"))
            for _ in range(rng.choice([1, 1, 1, 2, 3, 5])):
                out.append(rng.choice(COMBINING))
        else:
            out.append(rng.choice(FULLWIDTH))
            if rng.random() < 0.3:
                out.append(rng.choice(COMBINING))
    return "".join(out)


def _emoji(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return "".join(rng.choice(EMOJI) for _ in range(rng.choice([1, 2, 3])))
    if r < 0.6:
        return "\u200d".join(rng.choice(["\U0001F469", "\U0001F468", "\U0001F467", "\u2764\ufe0f", "\U0001F48B"])
                             for _ in range(rng.choice([2, 3, 4])))
    if r < 0.8:
        return "".join(rng.choice(["\U0001F1FA", "\U0001F1F8", "\U0001F1EF", "\U0001F1F5"])
                       for _ in range(rng.choice([1, 2, 3, 4, 5])))
    return rng.choice(["\U0001F44D", "\u261d", "\U0001F9D1"]) + rng.choice(["\U0001F3FB", "\U0001F3FD", ""]) + \
        rng.choice(["", "\ufe0f", "\u200d\U0001F4BB"])


def _special(rng: random.Random, specials) -> str:
    if not specials:
        return "<s>"
    s = rng.choice(specials)
    r = rng.random()
    if r < 0.3:
        return " " * rng.choice([1, 2, 3]) + s
    if r < 0.5:
        return s + " " * rng.choice([1, 2])
    if r < 0.6:
        return rng.choice(WS) + s + rng.choice(WS)
    return s


def _any_scalar(rng: random.Random) -> str:
    while True:  # bounded in expectation: surrogates are 2048 / 1.1M of the range
        c = rng.randrange(0x110000)
        if not 0xD800 <= c < 0xE000:
            return chr(c)


# grapheme interactions the Precompiled walk can see (docs/algorithms/unigram.md §3.3): a key char followed
# by joiners inside 5 bytes, the joiners after controls (GB4), prepend marks (GB9b), spacing marks (GB9a),
# conjuncts (GB9c) and emoji zwj (GB11) that swallow a key char into a long grapheme, hangul L V T runs
KEYISH = ["\uff21", "\uff41", "\ufb01", "\u2460", "\u2122", "\u00a8", "\u00b2", "\u00bd", "\u00b5", "\u017f",
          "\u2139", "\u203c", "\u2049", "\u3297", "\u3299", "\u24c2", "\U0001F202", "\U0001F21A", "\u0958",
          "\u095b", "\u09dc", "\u0a59", "\u0b5c", "\u2002", "\u3000", "\u00a0", "\x01", "\r", "\t", "\u2581",
          "\uff9e", "\u309b", "\u3131", "\u1d2c", "\u02b0", "\u00aa"]
JOINERS = ["\u0301", "\u0308", "\u030a", "\u0327", "\u20e3", "\ufe0f", "\u200d", "\u3099", "\u093c", "\u0903",
           "\u0e33", "\u094d", "\U0001d165", "\U000e0020", "\u200c", "\u0345", "\u1ab0"]


def _grapheme_combo(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return rng.choice(KEYISH) + "".join(rng.choice(JOINERS) for _ in range(rng.choice([1, 1, 2, 3])))
    if r < 0.45:  # GB9c: a conjunct swallowing a key consonant
        return rng.choice(DEVA_C) + "\u094d" + rng.choice(["\u0958", "\u095b", "\u095e", rng.choice(DEVA_C)]) + \
            rng.choice(["\u0301", "\u093c", "", "\u0300\u0301"])
    if r < 0.6:  # GB11: emoji zwj swallowing a key ExtPict
        return rng.choice(["\u2764", "\U0001F469", "\u00a9"]) + rng.choice(["", "\u0301", "\U0001F3FB"]) + \
            "\u200d" + rng.choice(["\u2122", "\u2139", "\u203c", "\u3297", "\U0001F202", "\u2764"]) + \
            rng.choice(["\u0301", "\ufe0f", ""])
    if r < 0.7:  # GB9b prepend
        return rng.choice(PREPEND) + rng.choice(KEYISH + ["a", "\u0301", "\u0628"]) + rng.choice(["", "\u0301"])
    if r < 0.8:  # hangul runs, with joiners after
        return rng.choice(JAMO_L) + rng.choice(JAMO_V) + rng.choice(["", rng.choice(JAMO_T)]) + \
            rng.choice(["", "\u0301", "\u302e"])
    if r < 0.9:  # controls then joiners (GB4 / GB5), CR LF runs
        return rng.choice(["\r", "\n", "\x01", "\x7f", "\x85", "\u200b", "\r\n"]) + rng.choice(JOINERS) + \
            rng.choice(["", "\r", "\n"])
    return rng.choice(RI_RUN) * rng.choice([1, 2, 3]) + rng.choice(["", "\u0301", "\u200d"])


RI_RUN = ["\U0001F1FA", "\U0001F1F8\U0001F1FA", "\U0001F1EF\U0001F1F5\U0001F1EF"]


def make_ctx(model) -> dict:
    """vocabulary- and charsmap-derived material for one tokenizer (unigram_model.Tokenizer)"""
    pieces = sorted(model.model.pieces)
    meta = "\u2581"
    single = {p for p in pieces if len(p) == 1}
    inner = sorted({p for p in pieces if len(p) > 1 and any(c not in single for c in p)})
    rng = random.Random(12345)
    sample = [p for p in pieces if p and len(p) <= 24]
    if len(sample) > 40000:
        sample = rng.sample(sample, 40000)
    keys = []
    cm = getattr(model, "charsmap", None)
    if cm is not None:
        keys = cm.live_keys()
    return {"pieces": [p.replace(meta, " ") for p in sample], "inner": [p.replace(meta, " ") for p in inner],
            "keys": keys}


def _ctx_unit(rng: random.Random, ctx) -> str:
    r = rng.random()
    if r < 0.4 and ctx["pieces"]:
        s = rng.choice(ctx["pieces"])
        if rng.random() < 0.3 and len(s) > 1:  # a cut piece: the lattice's neighbours
            a = rng.randrange(len(s))
            b = rng.randrange(a, len(s)) + 1
            s = s[a:b]
        return s
    if r < 0.6 and ctx["inner"]:
        s = rng.choice(ctx["inner"])
        if rng.random() < 0.5 and len(s) > 1:
            a = rng.randrange(len(s))
            s = s[a:] if rng.random() < 0.5 else s[:a + 1]
        return s
    if r < 0.8 and ctx["keys"]:
        k = rng.choice(ctx["keys"])
        return k + (rng.choice(JOINERS) if rng.random() < 0.3 else "")
    return _grapheme_combo(rng)


def unit(rng: random.Random, specials, ctx=None) -> str:
    r = rng.random()
    if ctx is not None and r < 0.22:
        return _ctx_unit(rng, ctx)
    r = rng.random()
    if r < 0.17:
        return _rand_word(rng)
    if r < 0.28:
        return _ws_run(rng)
    if r < 0.38:
        return _cjk(rng)
    if r < 0.44:
        return _kana(rng)
    if r < 0.49:
        return _hangul(rng)
    if r < 0.54:
        return _indic(rng)
    if r < 0.62:
        return _latin(rng)
    if r < 0.66:
        return _emoji(rng)
    if r < 0.70:
        return rng.choice(COMPAT)
    if r < 0.73:
        return rng.choice(CONTROL)
    if r < 0.77:
        return rng.choice(UNK)
    if r < 0.80:
        return rng.choice(META)
    if r < 0.83:
        return _special(rng, specials)
    if r < 0.86:
        return rng.choice(ASCII_PUNCT) * rng.choice([1, 1, 2, 3])
    if r < 0.89:
        return rng.choice(PREPEND) + rng.choice(["a", "1", " ", "\u0628", ""])
    if r < 0.92:
        return rng.choice(ARABIC) + rng.choice(["", rng.choice(ARABIC)])
    if r < 0.97:
        return rng.choice(WORDS)
    return _any_scalar(rng)


def gen_random(seed: int, idx: int, specials, ctx=None) -> str:
    rng = random.Random(seed * 1_000_003 + idx)
    n = rng.choice([1, 1, 2, 3, 4, 6, 9, 14, 24, 40, 80])
    parts = []
    for _ in range(n):
        parts.append(unit(rng, specials, ctx))
        if rng.random() < 0.45:
            parts.append(" " if rng.random() < 0.8 else _ws_run(rng))
    s = "".join(parts)
    if rng.random() < 0.1:
        s = " " * rng.choice([1, 2]) + s
    if rng.random() < 0.1:
        s = s + rng.choice([" ", "  ", "\n", "\r\n"])
    return s


def gen_real(lines, seed: int, idx: int, specials, ctx=None) -> str:
    rng = random.Random(seed * 1_000_033 + idx)
    s = lines[idx % len(lines)]
    r = rng.random()
    if r < 0.45:
        return s
    if r < 0.6 and len(lines) > 1:
        return s + rng.choice(["\n", " ", "\r\n", "  "]) + lines[rng.randrange(len(lines))]
    chars = list(s)
    for _ in range(rng.choice([1, 2, 3, 6])):
        k = rng.randrange(len(chars) + 1)
        chars.insert(k, unit(rng, specials, ctx))
    return "".join(chars)


ALPHABET = ["a", "A", "1", ".", "<", ">", " ", "\t", "\r", "\n", "\x00", "\x01", "\u3000", "\xa0", "\u200b",
            "\ufeff", "\u0301", "\u0308", "\u030a", "\u3099", "\uff9e", "\uff21", "\uff76", "\u304b", "\u30ab",
            "\u1100", "\u1161", "\u11a8", "\uac00", "\u0915", "\u094d", "\u0937", "\u093f", "\u0600", "\u200d",
            "\U0001F469", "\U0001F1FA", "\u2581", "\ufb01", "\u2460", "\u00e9", "\ue000", "\u0378", "\U00020000",
            "\U0001d400", "\ufdfa", "\u0e33", "\u2764"]


def gen_exhaustive(idx: int) -> str:
    """index order: all length-1 strings, then length 2, then length 3"""
    n = len(ALPHABET)
    k = 1
    while idx >= n ** k:  # bounded: k <= 3 for idx < n + n^2 + n^3
        idx -= n ** k
        k += 1
    out = []
    for _ in range(k):
        out.append(ALPHABET[idx % n])
        idx //= n
    return "".join(reversed(out))


EXHAUSTIVE_COUNT = len(ALPHABET) + len(ALPHABET) ** 2 + len(ALPHABET) ** 3

CONTEXTS = ["{}", "a{}", "{}a", " {}", "{} ", "x{}y", "{}\u0301", "{}{}", "\u0915\u094d{}", "{}\u200d\U0001F469",
            "\uff21{}", "<s> {}", "\u2581{}\u2581"]


def scalar_count() -> int:
    return 0x110000 * len(CONTEXTS)


def gen_scalar(idx: int):
    cp, k = divmod(idx, len(CONTEXTS))
    if 0xD800 <= cp < 0xE000:
        return None
    return CONTEXTS[k].replace("{}", chr(cp))
