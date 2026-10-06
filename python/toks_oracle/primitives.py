"""The parity primitives' references (include/toks.h: toks_template, toks_added, toks_info's truncation / padding /
template fields, TOKS_NO_TRUNCATE / TOKS_NO_PAD, TOKS_DECODE_RAW), as hf tokenizers 0.23.2 and tiktoken 0.14.0
define them. tests/c/test_primitives.py writes them into the C test's data; the python binding's tests can call the
same functions.

  template   num_special_tokens_to_add(False); post_process() of the empty encoding with padding and truncation off:
             its ids and type_ids; the prefix count and the text's type id from post_process() of a short text (the
             template's specials carry special_tokens_mask 1, the text's ids 0)
  added      get_added_tokens_decoder(), one AddedToken per id (hf keeps the content listed last for the id, with
             that listing's options): flags TOKS_ID_ADDED 1 | SPECIAL 2 | LSTRIP 8 | RSTRIP 16 | SINGLE_WORD 32 |
             NORMALIZED 64
  info       Tokenizer.truncation / .padding (max_length capped at 2^29, as toks reads it)
  no_*       encode() after no_truncation() / no_padding()
  raw        ByteLevel: each token's chars through the byte table (else the token's utf-8), before the decoder's
             from_utf8_lossy; ByteFallback: decode() with each U+FFFD the ByteFallback step wrote for a run that is not
             utf-8 replaced by the byte it stood for; tiktoken: Encoding.decode_bytes
"""
from __future__ import annotations

import os
import sys

INFO_FIELDS = ("trunc_on", "trunc_max", "trunc_stride", "pad_on", "pad_fixed", "pad_id", "pad_type_id", "pad_len",
               "pad_multiple", "pad_left")
ID_ADDED, ID_SPECIAL, ID_LSTRIP, ID_RSTRIP, ID_SINGLE_WORD, ID_NORMALIZED = 1, 2, 8, 16, 32, 64
MAX_TEXT = 1 << 29


def fnv(data: bytes) -> int:
    """FNV-1a 64 (test_primitives.c's pr_fnv)."""
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def ids_bytes(ids) -> bytes:
    return b"".join(int(i).to_bytes(4, "little") for i in ids)


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


CHAR_BYTES = _bytes_to_unicode()


def bytelevel_bytes(s: str) -> bytes:
    """hf's ByteLevel decoder on one token: its chars' bytes, or its utf-8 when a char is outside the table."""
    if all(c in CHAR_BYTES for c in s):
        return bytes(CHAR_BYTES[c] for c in s)
    return s.encode("utf-8")


def byte_fallback(s):
    """hf ByteFallback's byte token "<0xHH>" (u8::from_str_radix, which takes a leading '+'): its byte, else None."""
    if s is None or len(s.encode("utf-8")) != 6 or not s.startswith("<0x") or not s.endswith(">"):
        return None
    h = s[3:5]
    if h[0] == "+":
        h = h[1:]
    if not h or any(c not in "0123456789abcdefABCDEF" for c in h):
        return None
    return int(h, 16)


def decoder_kind(j):
    """'bytelevel' (the decoder is ByteLevel), 'bytefallback' (a chain with ByteFallback), else None."""
    d = j.get("decoder")
    if not isinstance(d, dict):
        return None
    if d.get("type") == "ByteLevel":
        return "bytelevel"

    def has_bf(x):
        if not isinstance(x, dict):
            return False
        return x.get("type") == "ByteFallback" or (x.get("type") == "Sequence" and any(has_bf(y) for y in x.get("decoders", [])))
    return "bytefallback" if has_bf(d) else None


def hf_decode_raw(tok, ids, strs):
    """hf decode(ids) with each U+FFFD that ByteFallback wrote for a run that is not utf-8 replaced, in order, by the
    byte it stood for; None when the output's U+FFFDs are not exactly those (the caller draws another sequence)."""
    out = tok.decode(ids, skip_special_tokens=False)
    bad, run = [], []
    for s in list(strs) + [None]:
        b = byte_fallback(s) if s is not None else None
        if b is not None:
            run.append(b)
            continue
        if run:
            try:
                bytes(run).decode("utf-8")
            except UnicodeDecodeError:
                bad += run
            run = []
    if out.count("\ufffd") != len(bad):
        return None
    enc, res, k, i = out.encode("utf-8"), bytearray(), 0, 0
    while i < len(enc):
        if enc[i:i + 3] == b"\xef\xbf\xbd":
            res.append(bad[k])
            k += 1
            i += 3
        else:
            res.append(enc[i])
            i += 1
    return bytes(res)


def added_records(entries):
    """(count, FNV-1a of id, flags, length and content per entry in id order): entries are (id, content, flags)."""
    blob = b""
    for i, c, fl in entries:
        cb = c.encode("utf-8")
        blob += int(i).to_bytes(4, "little") + int(fl).to_bytes(4, "little") + len(cb).to_bytes(4, "little") + cb
    return len(entries), fnv(blob)


def added_flags(at) -> int:
    return (ID_ADDED | (ID_SPECIAL if at.special else 0) | (ID_LSTRIP if at.lstrip else 0) |
            (ID_RSTRIP if at.rstrip else 0) | (ID_SINGLE_WORD if at.single_word else 0) |
            (ID_NORMALIZED if at.normalized else 0))


def _fresh(src: str, trunc: bool = True, pad: bool = True):
    from tokenizers import Tokenizer
    t = Tokenizer.from_str(src)
    if not trunc:
        t.no_truncation()
    if not pad:
        t.no_padding()
    return t


def records(src_bytes: bytes, j, texts, flag_sets):
    """every reference of one tokenizer.json, or None when hf refuses the file."""
    src = src_bytes.decode("utf-8")
    try:
        tok = _fresh(src)
        bare = _fresh(src, trunc=False, pad=False)
    except BaseException as e:  # noqa: BLE001 (pyo3 panics are BaseException)
        if isinstance(e, KeyboardInterrupt):
            raise
        return None
    rec = {"tok": tok}
    pe = bare.post_process(bare.encode("", add_special_tokens=False))
    if len(pe.ids) != tok.num_special_tokens_to_add(False):
        raise SystemExit("the template's count is not num_special_tokens_to_add(False)")
    rec["tmpl_ids"], rec["tmpl_types"] = list(pe.ids), list(pe.type_ids)
    rec["n_prefix"], rec["seq_type"] = len(pe.ids), 0
    for probe in ("hello", "a", "x y z", "1", "\u4e2d"):
        e1 = bare.encode(probe, add_special_tokens=False)
        if len(e1.ids) == 0:
            continue
        p1 = bare.post_process(e1)
        n = 0
        while n < len(p1.special_tokens_mask) and p1.special_tokens_mask[n] == 1:
            n += 1
        if list(p1.ids[:n]) + list(p1.ids[n + len(e1.ids):]) != rec["tmpl_ids"]:
            raise SystemExit("the template around a text is not the empty text's")
        rec["n_prefix"], rec["seq_type"] = n, p1.type_ids[n]
        break
    empty = bare.encode("").ids                          # the card: encode("") after the template equals it
    rec["empty"] = (len(empty), fnv(ids_bytes(empty)))
    tr, pd = tok.truncation, tok.padding
    rec["info"] = {
        "trunc_on": int(tr is not None),
        "trunc_max": min(tr["max_length"], MAX_TEXT) if tr else 0,
        "trunc_stride": min(tr["stride"], 0xFFFFFFFF) if tr else 0,
        "pad_on": int(pd is not None),
        "pad_fixed": int(pd is not None and pd["length"] is not None),
        "pad_id": pd["pad_id"] if pd else 0,
        "pad_type_id": pd["pad_type_id"] if pd else 0,
        "pad_len": (pd["length"] or 0) if pd else 0,
        "pad_multiple": (pd["pad_to_multiple_of"] or 0) if pd else 0,
        "pad_left": int(pd is not None and pd["direction"] == "left"),
    }
    dec = sorted(tok.get_added_tokens_decoder().items())
    entries = [(i, at.content, added_flags(at)) for i, at in dec]
    rec["added"] = entries
    rec["n_added"], rec["added_digest"] = added_records(entries)
    specials = [c for _, c, fl in entries if fl & ID_SPECIAL][:2] + [c for _, c, fl in entries if not fl & ID_SPECIAL][:1]
    rec["text4"] = ("a " + " x ".join(specials) + " b") if specials else "no added tokens here"
    rec["enc"] = []
    if tr is not None or pd is not None:
        variants = {f: _fresh(src, trunc=not f & 16, pad=not f & 32) for f in (0, 16, 32, 48)}
        for t, text in enumerate(list(texts) + [rec["text4"]]):
            for f in flag_sets:
                try:
                    ids = variants[f & 48].encode(text, add_special_tokens=not f & 4).ids
                except BaseException as e:  # noqa: BLE001
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    continue
                rec["enc"].append((t, f, len(ids), fnv(ids_bytes(ids))))
    return rec


def kimi_records(d, texts, rnd):
    """Kimi K3 through its own stack (tests/parity/oracle_tiktoken.py: transformers' TikTokenTokenizer + tiktoken):
    no template, truncation or padding; its 256 specials (tiktoken's names) with TOKS_ID_SPECIAL where transformers
    lists the id special; decode_bytes for raw."""
    import oracle_tiktoken as OT
    k = OT.Kimi(d)
    special = set(k.tok.all_special_ids)
    entries = sorted((i, name, ID_ADDED | (ID_SPECIAL if i in special else 0)) for name, i in k.special_tokens.items())
    rec = {"tok": None, "tmpl_ids": [], "tmpl_types": [], "n_prefix": 0, "seq_type": 0, "sha": "",
           "info": {x: 0 for x in INFO_FIELDS}, "added": entries, "enc": [], "empty": (0, fnv(b""))}
    rec["n_added"], rec["added_digest"] = added_records(entries)
    rec["text4"] = "[BOS] hi <|open|>x[EOS]"
    enc = k.enc
    n = enc.n_vocab
    high = [i for i in range(n - 256) if len(enc.decode_single_token_bytes(i)) == 1 and enc.decode_single_token_bytes(i)[0] >= 0x80]
    raw = []
    for _ in range(32):
        ids = [rnd.choice(high) if rnd.random() < 0.4 else rnd.randrange(n) for _ in range(rnd.randint(1, 12))]
        raw.append((2, ids, enc.decode_bytes(ids)))
    for text in texts[:3]:
        ids = enc.encode(text, allowed_special="all")
        raw.append((2, ids, enc.decode_bytes(ids)))
    rec["raw"] = raw
    return rec


def show(rec):
    """one file's references, readable (a mismatch's other side)."""
    print("template ids", rec["tmpl_ids"], "types", rec["tmpl_types"], "n_prefix", rec["n_prefix"], "seq type",
          rec["seq_type"])
    print("info", rec["info"])
    print("empty encode (count, digest)", rec["empty"])
    print("added:", rec["n_added"], "digest %016x" % rec["added_digest"])
    for i, c, fl in rec["added"]:
        print("  %8d  flags %3d  %r" % (i, fl, c))
    for t, f, n, h in rec["enc"]:
        print("  encode text %d flags %2d: %d ids, %016x" % (t, f, n, h))
    for fl, ids, exp in rec.get("raw", []):
        print("  raw", ids, exp)
    sys.stdout.flush()


__all__ = ["INFO_FIELDS", "fnv", "ids_bytes", "bytelevel_bytes", "byte_fallback", "decoder_kind", "hf_decode_raw",
           "records", "kimi_records", "show", "added_records", "added_flags"]

if os.environ.get("TOKS_PRIMITIVES_SELFTEST"):          # the fnv constant against a known vector
    assert fnv(b"a") == 0xAF63DC4C8601EC8C
