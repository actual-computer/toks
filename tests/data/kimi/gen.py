#!/usr/bin/env python3
"""tests/data/kimi/gen.py: the reference values tests/c/test_tiktoken.c checks the tiktoken reader against,
taken from tiktoken 0.14.0 and the Kimi K3 wrapper themselves (never from toks):

    tools/oracle/py.sh tests/data/kimi/gen.py          # writes expect.txt and bpe_cases.txt here

expect.txt     counts and sha-256s of what the reader must produce from the pinned files:
                 ranks          tokens in tiktoken.model (tiktoken's load_tiktoken_bpe)
                 tokens_sha256  sha-256 over ids 0..n-1 of (u32 le length, token bytes)
                 merges         every split of every token into two tokens (both halves tokens), ordered by
                                merged id then split position; merges_sha256 over (u32 le left, right, merged)
                 specials_sha256  over the wrapper's 256 special ids in order of (u32 le length, name)
                 trie           the ids transformers' added-token trie splits on (tok._added_tokens_encoder)
                 named          tok.all_special_ids (what decode(skip_special_tokens=True) drops)
                 n_ids          tok.n_words
bpe_cases.txt  one piece per line: its bytes in hex, then tiktoken's ids for it (encode_single_piece: the
               whole piece if it is a token, else byte_pair_encode; pieces of >= 100 bytes take tiktoken's heap
               path): real-text pieces in a dozen scripts, random bytes (most not utf-8), long runs.
"""
import hashlib
import os
import random
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))

TEXT = [
    "The quick brown fox jumps over the lazy dog. It's 2026; we'll see what they've done!",
    "中华人民共和国是位于东亚的社会主义国家，首都北京。汉字是世界上最古老的文字之一。",
    "日本語は主に日本で使われている言語である。ひらがな、カタカナ、漢字を組み合わせて書く。",
    "대한민국은 동아시아의 한반도 남부에 위치한 민주공화국이다. 수도는 서울특별시이다.",
    "Россия — государство в Восточной Европе и Северной Азии. Столица — Москва.",
    "مصر دولة عربية تقع في الركن الشمالي الشرقي من قارة أفريقيا، عاصمتها القاهرة.",
    "भारत दक्षिण एशिया में स्थित एक विशाल देश है। इसकी राजधानी नई दिल्ली है।",
    "ประเทศไทยเป็นประเทศในเอเชียตะวันออกเฉียงใต้ เมืองหลวงคือกรุงเทพมหานคร",
    "def encode(self, text: str) -> list[int]:\n    return [self.vocab[t] for t in text.split()]  # 42\n",
    "Việt Nam, tên chính thức là Cộng hòa Xã hội Chủ nghĩa Việt Nam, là một quốc gia ở Đông Nam Á.",
    "Ελλάδα είναι χώρα της νοτιοανατολικής Ευρώπης. 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 😀👍🏽 ︎ ❤️ 1234567890 ０１２",
    "    \t\n\n  \r\n\u3000\u3000  ...!!! ??? <<< >>> ''' \"\"\" ``` ### $$$ %%% ^^^ &&& *** ((( )))",
]


def main():
    from oracle_tiktoken import Kimi
    from tiktoken.load import load_tiktoken_bpe
    k = Kimi()
    ranks = load_tiktoken_bpe(os.path.join(os.path.expanduser("~/.cache/toks/kimik3"), "tiktoken.model"))
    by_id = sorted(ranks.items(), key=lambda kv: kv[1])
    assert [r for _, r in by_id] == list(range(len(by_id)))
    h = hashlib.sha256()
    for b, _ in by_id:
        h.update(struct.pack("<I", len(b)) + b)
    tokens_sha = h.hexdigest()
    hm = hashlib.sha256()
    nm = 0
    for b, r in by_id:
        for s in range(1, len(b)):
            if b[:s] in ranks and b[s:] in ranks:
                hm.update(struct.pack("<III", ranks[b[:s]], ranks[b[s:]], r))
                nm += 1
    hs = hashlib.sha256()
    sp = sorted(k.special_tokens.items(), key=lambda kv: kv[1])
    assert [i for _, i in sp] == list(range(len(by_id), len(by_id) + 256))
    for name, _ in sp:
        nb = name.encode("utf-8")
        hs.update(struct.pack("<I", len(nb)) + nb)
    trie = sorted(k.added.values())
    named = sorted(k.tok.all_special_ids)
    with open(os.path.join(HERE, "expect.txt"), "w") as f:
        f.write("# tests/data/kimi/gen.py (tiktoken 0.14.0, transformers 5.18.0; moonshotai/Kimi-K3 @ f831ab66)\n")
        f.write(f"ranks {len(by_id)}\n")
        f.write(f"tokens_sha256 {tokens_sha}\n")
        f.write(f"merges {nm}\n")
        f.write(f"merges_sha256 {hm.hexdigest()}\n")
        f.write(f"specials_sha256 {hs.hexdigest()}\n")
        f.write("trie " + " ".join(map(str, trie)) + "\n")
        f.write("named " + " ".join(map(str, named)) + "\n")
        f.write(f"n_ids {k.tok.n_words}\n")

    rng = random.Random(20261004)
    pieces = []
    for t in TEXT:                                   # word-like pieces (' ?\S+' and whitespace runs)
        for w in re.findall(r" ?\S+|\s+", t):
            pieces.append(w.encode("utf-8"))
    for _ in range(300):                             # substrings of real text (any byte range)
        t = rng.choice(TEXT).encode("utf-8")
        a = rng.randrange(len(t))
        pieces.append(t[a:a + rng.randint(1, 48)])
    for _ in range(150):                             # random bytes, mostly not utf-8
        pieces.append(bytes(rng.randrange(256) for _ in range(rng.randint(1, 40))))
    for _ in range(60):                              # >= 100 bytes: tiktoken's heap path
        t = rng.choice(TEXT).encode("utf-8")
        a = rng.randrange(len(t))
        p = (t[a:] + t)[:rng.randint(100, 300)]
        pieces.append(p)
    for unit, n in ((b"a", 257), (" ".encode(), 130), ("中".encode(), 120), (b"1", 101), (b"ab", 75),
                    ("\u3000".encode(), 40), (b"=", 200), (b"\n", 150), ("é".encode(), 64)):
        pieces.append(unit * n)
    for _ in range(40):                              # base64-like and hex-like long words
        alpha = rng.choice([b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", b"0123456789abcdef"])
        pieces.append(bytes(rng.choice(alpha) for _ in range(rng.randint(20, 160))))
    seen, out = set(), []
    for p in pieces:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    with open(os.path.join(HERE, "bpe_cases.txt"), "w") as f:
        f.write("# tests/data/kimi/gen.py: <piece hex> <tiktoken 0.14.0 encode_single_piece ids>\n")
        for p in out:
            ids = k.enc._encode_single_piece(p)
            f.write(p.hex() + " " + " ".join(map(str, ids)) + "\n")
    print(f"expect.txt: ranks {len(by_id)} merges {nm}; bpe_cases.txt: {len(out)} pieces")


if __name__ == "__main__":
    main()
