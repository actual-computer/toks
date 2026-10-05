#!/usr/bin/env python3
"""tests/wordpiece/golden.py: BertNormalizer golden vectors for tests/c/test_wp.c, straight from hf 0.23.2.

    uv run --with tokenizers==0.23.2 tests/wordpiece/golden.py > tests/data/wordpiece/norm_golden.inc

Each line: { toks flags (TOKS_WPF_*: 1 clean, 2 chinese, 4 strip, 8 lower), S(input), S(hf output) }.
"""
from tokenizers import normalizers
BN = normalizers.BertNormalizer
cfg = {0xF: BN(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True),
       0x3: BN(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=False),
       0xB: BN(clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=True),
       0x0: BN(clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=False),
       0x4: BN(clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False)}
ex = ["\u0130stanbul", "\ud55c\uad6d\uc5b4", "a\u00a0b", "a\u0085b", "a\u200bb", "a\x0bb", "a\x0cb", "\U0001f44d\u200d\U0001f525",
      "e\u0301\ufe0f", "\u1fef", "\U00030000x", "\uff21\uff22", "\u039f\u0394\u039f\u03a3", "caf\u00e9 CAF\u00c9",
      "\uf900\u4e00", "\u2000x\u2001", "\u212b", "\u1e9e\u00df", "a\U0001d165\u0301\U0001d16eb", "x\ufffdy", "\x00a",
      "\u0f73", "a\U0001e000\U0001e94a", "a\U0001e000\u034f\U0001e94a", "a\U0001e000\u200b\U0001e94a",
      "a\U0001e000\u0301\U0001e94a", "x\u2260b", "\u037e\u0387", "Hello\tWorld\r\n", "\u3000\u2028", "\U000e0041z\ue000"]
def c(s):
    b = s.encode("utf-8")
    return '"' + "".join(chr(x) if 0x20 <= x < 0x7f and chr(x) not in '"\\?' else "\\x%02x\"\"" % x for x in b) + '"'
for f, n in cfg.items():
    for s in ex:
        print('    { 0x%Xu, S(%s), S(%s) },' % (f, c(s), c(n.normalize_str(s))))
