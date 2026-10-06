"""tests/fuzz/hostile/texts.py: the hostile texts and call-shape battery (docs/fuzz.md "hostile"; SPEC
T6 / T9 on the text side). gen.py --texts writes this as texts.json; hostile_driver.c reads it.

Schema (json, one object):
  texts   [[name, description, hex-bytes], ...]            hostile texts, every boundary reasoned
  calls   [[kind, name, args...], ...]                     call-shape cases per entry point, by kind:
            size     [text-index or null, len]              encode/pieces at a length: 0, 1, cap-1, cap,
                                                             cap+1, TOKS_MAX_TEXT-1/exact/+1 (the driver
                                                             builds the text or a guard-mapped buffer)
            stream   [text-index, n_ids]                    decode of encode's ids one id at a time
                                                             (byte-fallback stress) + random partitions
            par      [text-index, len, n_pool]               toks_par_encode with pools of 1 and 64
            split    [text-index, n_want]                    toks_split_points with n_want 0 / 1 / huge
            misalign [text-index, +1..+7]                     every caller buffer 1..7 bytes off

The driver derives the rest (cap 0/short/exact, out NULL, overlap, decode past n_ids, toks_info sizes,
toks_template / toks_added probing) from these: the lists here are the inputs, the battery is in the C.
"""
from __future__ import annotations

import json

MAX_TEXT = 1 << 29          # TOKS_MAX_TEXT

TEXTS = []      # (name, description, bytes)


def t(name, desc, data):
    TEXTS.append((name, desc, data if isinstance(data, bytes) else data.encode("utf-8")))


def gen_texts():
    # ---- size boundaries: 0, 1, 15, 16, 17, 31, 32, 63, 64, 65 (block edges), 255, 256, 257, 4095, 4096
    for n in (0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024,
          1025, 4095, 4096, 4097, 65535, 65536, 65537):
        t("size-%d" % n, "a text of exactly %d bytes (the size boundary)" % n, b"a" * n if n else b"")
    # TOKS_MAX_TEXT - 1 / exact / +1: the driver guards these (a 512 MiB read would be slow); spelled
    # as lengths in the calls, with a one-byte text, so the length check fires before any read.
    # ---- invalid utf-8 at every position and kind
    t("bad-isolated-80", "an isolated continuation byte", b"\x80")
    t("bad-isolated-bf", "an isolated late continuation byte", b"\xbf")
    t("bad-c1", "an overlong C1 lead with a valid tail", b"\xc1\xbf")
    t("bad-c0", "the C0 overlong lead", b"\xc0\x80")
    t("bad-e0-80", "an overlong 3-byte form", b"\xe0\x80\x80")
    t("bad-e0-9f", "an overlong 3-byte form (9f tail)", b"\xe0\x9f\xbf")
    t("bad-f0-80", "an overlong 4-byte form", b"\xf0\x80\x80\x80")
    t("bad-f0-8f", "an overlong 4-byte form (8f tail)", b"\xf0\x8f\xbf\xbf")
    t("bad-f4-90", "out of range (F4 90)", b"\xf4\x90\x80\x80")
    t("bad-f5", "an F5 lead", b"\xf5\x80\x80\x80")
    t("bad-fd", "an FD lead (5-byte era)", b"\xfd\x80\x80\x80\x80")
    t("bad-ed-a0", "an encoded surrogate D800", b"\xed\xa0\x80")
    t("bad-ed-bf", "an encoded surrogate DFFF", b"\xed\xbf\xbf")
    t("bad-trunc-e0", "a 3-byte lead truncated at the end", b"abc\xe0")
    t("bad-trunc-e0-1", "a 3-byte lead missing its last byte", b"abc\xe0\xa0")
    t("bad-trunc-f0", "a 4-byte lead truncated at the end", b"abc\xf0\x9f")
    t("bad-trunc-f0-2", "a 4-byte lead missing two bytes", b"abc\xf0")
    t("bad-tail-abort", "a 3-byte sequence whose tail is broken", b"abc\xe0\xa0\x41")
    t("bad-in-e4-b8", "a truncated CJK char (e4 b8) then a letter", b"\xe4\xb8" + b"x")
    t("bad-then-fffd", "an invalid byte then U+FFFD", b"\x80\xef\xbf\xbd")
    t("fffd-then-bad", "U+FFFD then an invalid byte", b"\xef\xbf\xbd\x80")
    t("bad-nul", "an embedded NUL", b"a\x00b")
    t("bad-nul-only", "a single NUL", b"\x00")
    # a truncated 4-byte sequence straddling split boundaries: the driver cuts at every byte
    t("straddle-f0", "a 4-byte char cut in the middle (split points hit it)", b"ab\xf0\x9f\x92\xa9cd")
    t("straddle-e0", "a 3-byte char cut in the middle", b"ab\xe2\x82\xaccd")
    t("straddle-marks", "combining marks across cut points", b"a\xcc\x81\xcc\x86b\xcc\x88\xcc\xa9")
    # 1 MiB of one byte for every byte value (the driver samples: one case per byte, 1 MiB each, is a
    # driver-side loop over the same text with the byte swapped; the text here is the 256-byte table)
    t("all-bytes", "every byte value once", bytes(range(256)))
    t("all-bytes-rev", "every byte value once, reversed", bytes(reversed(range(256))))
    t("all-bad", "every ill-formed lead byte with its best tail", b"".join(
        bytes([b]) + b"\xbf" * 3 for b in range(0x80, 0x100) if b not in (0xC2, 0xC3)))
    # added-token spam: 10^5 occurrences of a markup token (the driver tiles this)
    t("spam-seed", "a markup token and text to tile 10^5 times", b"<|im_start|>user\nhi<|im_end|>\n")
    # texts whose bound saturates: the driver derives them per tokenizer (r * len + g)
    t("saturate-seed", "the worst-expansion seed (U+FDFA, U+1D160, marks)", "\ufdfa\U0001D160\u0301\u0306")
    # NFC / NFKC changes
    t("nfc-change", "text NFC changes (e + combining acute)", b"e\xcc\x81\xcc\x86 o\xcc\x88")
    t("nfkc-bomb", "U+FDFA x16 (3 bytes -> 18 chars each under NFKC)", "\ufdfa" * 16)
    t("musical", "U+1D160 x8 (4 bytes -> 3 four-byte chars under NFC)", "\U0001D160" * 8)
    t("hangul-jamo", "precomposed Hangul vs jamo", "\u1100\u1161\u11a8 \uac00\u11a8")
    t("reorder-marks", "combining marks that reorder (cc81 cc86 vs cc86 cc81)", b"a\xcc\x81\xcc\x86 b\xcc\x86\xcc\x81")
    # the 25 \s chars
    WS = [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B)) + \
        [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    t("all-ws", "every regex \\s char once", "".join(chr(c) for c in WS).encode())
    t("all-ws-pairs", "every \\s char followed by every other (the \\s+(?!\\S) lookahead)",
      "".join(chr(a) + "x" for a in WS).encode())
    t("ws-run", "a 1000-byte space run", b" " * 1000)
    t("ws-nl-mix", "CR LF NL mixes", b"\r\n\n\r\r\n\n\n\r a \r\n\r \n")
    t("nl-before-letter", "the SPEC §5.3 lookahead case: \\n\\n then A, cut at 2", b"\n\nA")
    t("tab-tab-1", "the SPEC §5.3 case: \\t\\t1, cut at 2", b"\t\t1")
    # contractions and case
    t("contractions", "every contraction in both cases", b"'s'S't'T're'RE've'Ve'm'M'll'LL'd'D")
    t("long-s", "U+017F (the (?i) s fold)", b"\xc5\xbf's \xc5\xbf\x74")
    t("dotless-i", "U+0131 and U+0049 under (?i)", b"\xc4\xb1I \xc4\xb1i I\xc4\xb1")
    # markup-like texts that collide with added tokens
    t("markup-cut", "an added token cut short", b"<|im_start")
    t("markup-overlap", "an added token with a char inside", b"<<|im_start|>>")
    t("markup-double", "an added token doubled", b"<|im_start|><|im_start|>")
    t("markup-space", "an added token with spaces around", b" <|im_start|> ")
    t("markup-strip", "text an lstrip/rstrip token eats into", b"   <|tok|>   rest")
    t("single-word", "single_word token boundaries", b"xmidy mid xmidx mid")
    # byte-fallback and spm texts
    t("spm-spaces", "spaces (each becomes U+2581 under the fold)", b" " * 33)
    t("spm-mixed", "text around U+2581-remapped spaces", "▁a b▁c ▁▁d")
    t("byte-hex-text", "byte-fallback spellings in the text", b"<0x41><0xFF><0x00> <0xE2><0x96><0x81>")
    t("cjk", "CJK and Hangul", b"\xe4\xb8\xad\xe6\x96\x87 \xed\x95\x9c\xea\xb5\xad\xec\x96\xb4")
    t("zwj-chain", "a zwj emoji chain (grapheme stress)", "\U0001F469\u200d\U0001F4BB\u200d" * 8)
    t("zwj-tail", "a zwj at the very end", "ab\U0001F469\u200d")
    t("emoji-flags", "regional indicator pairs", "\U0001F1FA\U0001F1F8\U0001F1FA\U0001F1F8")
    t("marks-only", "only combining marks", b"\xcc\x81\xcc\x86\xcc\x88" * 8)
    t("zero-width", "zero-width joiners and spaces", b"a\xe2\x80\x8db\xef\xbb\xbf" * 8)
    t("digits-run", "a long digit run", b"1234567890" * 30)
    t("mixed-classes", "alternating classes", b"a1 !b2\t?c3\n#d4 ,e5")
    t("punct-run", "a punctuation run", b".,!?;:-_()" * 20)
    t("bidi", "bidi controls and arabic", b"\xd8\xa7\xd9\x84\xd8\xb9\xd8\xb1\xd8\xa8\xd9\x8a\xd8\xa9 \xe2\x80\x8e\xe2\x80\x8f")
    t("high-plane", "supplementary plane chars", b"\xf0\x90\x80\x80\xf0\x9f\x98\x80\xf4\x8f\xbf\xbf")
    t("max-cp", "U+10FFFF and U+FFFE", b"\xf4\x8f\xbf\xbf\xef\xbf\xbe")
    t("grapheme-incb", "GB9c linker sequences", "\u0BA8\u0BBF\u0BCD\u0BA8\u0BCD\u0BB0\u0BCD")
    t("old-hang", "precomposed Hangul NFD triples", "\u1102\u1167\u11ba" * 16)


def gen_calls():
    CALLS = []
    # size cases on the fixed texts: the driver maps text + len (a text shorter than len is a guard-mapped
    # buffer, len past TOKS_MAX_TEXT checks the limit before reading)
    idx = {name: i for i, (name, _, _) in enumerate(TEXTS)}
    for n in (0, 1, 2):
        CALLS.append(["size", "size-%d" % n, n])
    CALLS.append(["size", "size-1", MAX_TEXT - 1])      # TOKS_MAX_TEXT - 1: guard-mapped, not read
    CALLS.append(["size", "size-1", MAX_TEXT])          # exact: guard-mapped, no touch
    CALLS.append(["size", "size-1", MAX_TEXT + 1])     # TOKS_E_LIMIT before any read
    CALLS.append(["size", "size-1", 2**64 - 1])       # TOKS_E_LIMIT
    # stream: the byte-fallback one-at-a-time stress and random partitions
    CALLS.append(["stream", "spam-seed", 100000])                  # 10^5 ids one at a time
    CALLS.append(["stream", "spm-spaces", 0])                      # ids of encode(spaces), one at a time
    CALLS.append(["stream", "nfkc-bomb", 0])
    CALLS.append(["stream", "byte-hex-text", 0])
    CALLS.append(["stream", "all-bytes", 0])
    # par: 1-byte and 64 MiB inputs, pools of 1 and 64
    CALLS.append(["par", "size-1", 1, 1])
    CALLS.append(["par", "size-1", 1, 64])
    CALLS.append(["par", "spam-seed", 64 << 20, 1])                # 64 MiB tiled
    CALLS.append(["par", "spam-seed", 64 << 20, 64])
    CALLS.append(["par", "ws-run", 16 << 20, 4])                  # no cuts: the serial fallback
    CALLS.append(["par", "nl-before-letter", 32 << 20, 8])
    # split_points: n_want 0 / 1 / huge / cap edges (the driver adds cap 0..n)
    CALLS.append(["split", "nl-before-letter", 0])
    CALLS.append(["split", "nl-before-letter", 1])
    CALLS.append(["split", "spam-seed", 2])
    CALLS.append(["split", "spam-seed", 15])
    CALLS.append(["split", "spam-seed", 16])
    CALLS.append(["split", "spam-seed", 2**31])                    # huge n_want
    CALLS.append(["split", "straddle-f0", 3])
    CALLS.append(["split", "straddle-marks", 4])
    CALLS.append(["split", "cjk", 2])
    # misalignment: every fixed text at +1..+7 (the driver loops the byte offsets over these)
    for i in (idx["markup-overlap"], idx["spm-mixed"], idx["contractions"]):
        for off in range(1, 8):
            CALLS.append(["misalign", TEXTS[i][0], off])
    return CALLS


def write(path: str) -> None:
    gen_texts()
    calls = gen_calls()
    obj = {"texts": [{"name": n, "desc": d, "hex": b.hex()} for n, d, b in TEXTS], "calls": calls}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print("hostile: %d texts, %d call cases -> %s" % (len(TEXTS), len(calls), path))
