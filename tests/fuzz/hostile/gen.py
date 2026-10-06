# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""tests/fuzz/hostile/gen.py: hand-reasoned hostile tokenizer files, by reader / table-builder category
(docs/fuzz.md "hostile"; SPEC T9).

Every case is an attack reasoned off the reader or builder it names (src/core/config.c, json.c, tiktoken.c,
precompiled.c, unigram.c, bpe_build.c, wp.c, compile.c): the limit it stresses is named in the case's
description, not discovered by mutation. Deterministic (seeded), stdlib only.

  uv run tests/fuzz/hostile/gen.py files --out DIR [--kind K] [--big]
      writes <category>/<name>.json (or .tiktokenmodel / .bin for the non-json readers), one file a case,
      plus cases.txt: `path <TAB> sha256 <TAB> description <TAB> expects` where expects is one of
      accept / refuse:<reason>. `--kind` narrows to one category; `--big` includes the multi-megabyte
      cases (off by default so a laptop run stays seconds).
  uv run tests/fuzz/hostile/gen.py texts --out FILE
      writes texts.json: the hostile texts and call-shape battery (see texts.py for the schema), the
      same file the driver reads.

Categories (each maps to the code that must survive it):
  json          the reader: depth 63/64/65, dup keys, giant arrays, string/number edges, truncation at
                every structural boundary
  vocab         read_vocab: id holes, 2^21-2/2^21-1/2^21 ids, max token bytes 65535/127, two strings one
                id, dense-check, 256-byte-token-but-one-more
  merges        read_merges: unknown tokens, cyclic, self-referential, rank-order contradictions, both
                formats mixed, #version lines, > 2^22 merges
  added         read_added: prefixes of each other, empty/huge/invalid-utf8 contents, ids colliding with
                vocab ids, 65535/65536 entries, strip flags, single_word
  normalizer    bl_normalizer / uni / spm readers: NFKC bombs (U+FDFA), Replace with huge content,
                prepend chains, Precompiled charsmap with corrupt base64 / bad trie offsets / huge
                expansions, out-of-order unigram normalizer steps
  pretok        bl_pretok: a known template's exact string differing by one byte, lookbehind-looking
                generics, 16-step sequences, nested Sequences to depth 5, invert / behaviors
  template      read_template: 64 pieces, unknown special tokens, ids beyond the vocabulary, type ids,
                RobertaProcessing pairs
  truncpad      read_trunc_pad / trunc_check: max_length 0, stride == max, pad_to_multiple_of 2^20+1,
                Fixed 2^29+1, Left, pad_id beyond vocab
  unigram       parse_unigram / unigram.c: scores NaN-spelled / 1e309 / -1e-309 / 1e308, byte fallback
                with missing byte pieces (the uni_resolve class), unk_id out of range, pieces 127/128
                bytes, empty pieces, duplicate pieces
  wordpiece     parse_wordpiece: max_input_chars_per_word 0 / 1024 / 1025 / 2^31, empty and 16-byte
                continuing prefix, unk_token missing from vocab, BertNormalizer field shapes
  tiktoken      tiktoken.c ranks: bad base64, out-of-order and duplicate ranks, missing one-byte tokens,
                1-byte and 1 MiB tokens, CRLF / bare-LF / space-broken lines, a rank at 2^21-1, the
                kimi / qwen wrappers' exact-line checks with one byte off
  spm           parse_spm: normalizer chains at the 8-op limit, Metaspace schemes, decoder orders,
                hole vocabularies, byte_fallback shapes, unk_token not in vocab
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys

# ---- gpt-2 bytes_to_unicode, the byte-level alphabet (config.c toks_alpha_bytes) --------------------------

_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}
ALPHABET = [B2U[b] for b in range(256)]

GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"""
          r"""\s*[\r\n]+|\s+(?!\S)|\s+""")
META = "\u2581"          # U+2581
FDFA = "\ufdfa"          # ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM: 18 chars under NFKC
MUSICAL = "\U0001D160"   # MUSICAL SYMBOL G CLEF: 4 utf-8 bytes, composes under NFC


def alpha(bs) -> str:
    return "".join(B2U[b] for b in bs if isinstance(b, int)) if not isinstance(bs, str) else bs


# ---- a small byte-level bpe model ------------------------------------------------------------------------

MERGES = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("Ġh", "e"), ("Ġhello", None)]


def bpe_vocab(extra=()):
    vocab = ALPHABET[:]
    have = set(vocab)
    for a, b in MERGES:
        if b is None:
            w = a
            parts = _split_known(w, have)
        else:
            w = a + b
            parts = (a, b)
        if w in have or parts is None:
            continue
        vocab.append(w)
        have.add(w)
    for x in extra:
        if x not in have:
            vocab.append(x)
            have.add(x)
    return vocab


def _split_known(w, have):
    for k in range(len(w) - 1, 0, -1):
        if w[:k] in have and w[k:] in have:
            return (w[:k], w[k:])
    return None


def bpe_merges(vocab):
    have = set(vocab)
    out = []
    for w in vocab[256:]:
        parts = _split_known(w, have - {w} | set(vocab[:256]))
        if parts is not None:
            out.append(list(parts))
    return out


def bl_model(vocab=None, merges="pairs", ignore_merges=False, **over):
    vocab = vocab if vocab is not None else bpe_vocab()
    m = bpe_merges(vocab)
    if merges == "strings":
        m = [" ".join(p) for p in m]
    elif merges == "none":
        m = []
    d = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
         "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
         "ignore_merges": ignore_merges, "vocab": {s: i for i, s in enumerate(vocab)}, "merges": m}
    d.update(over)
    return d


def bytelevel(aps=False, trim=True, ur=True):
    return {"type": "ByteLevel", "add_prefix_space": aps, "trim_offsets": trim, "use_regex": ur}


def split_re(regex, behavior="Isolated", invert=False):
    return {"type": "Split", "pattern": {"Regex": regex}, "behavior": behavior, "invert": invert}


_PRE_DEFAULT = object()


def tok(model, added=None, normalizer=None, pre=_PRE_DEFAULT, post=None, decoder=None, trunc=None, pad=None):
    """assembles a tokenizer.json. pre defaults to ByteLevel for a plain bpe model (callers doing
    byte-level work never spell it); passing pre=None asks for NO pre-tokenizer."""
    if pre is _PRE_DEFAULT:
        t = model.get("type") if isinstance(model, dict) else None
        if t in ("BPE", None) and model.get("unk_token") is None and "unk_id" not in model and \
                "max_input_chars_per_word" not in model:
            pre = bytelevel()           # a plain byte-level bpe (gpt2-style: no unk)
        else:
            pre = None                  # spm / unigram / wordpiece spell their own
    if decoder is None and isinstance(model, dict) and (
            model.get("type") == "Unigram" or "unk_id" in model or "max_input_chars_per_word" in model):
        decoder = None           # Unigram / WordPiece spell their own (or none); ByteLevel is wrong there
    else:
        decoder = decoder or (bytelevel() if not (isinstance(model, dict) and (model.get("type") == "Unigram"
                      or "unk_id" in model or "max_input_chars_per_word" in model)) else None)
    return {"version": "1.0", "truncation": trunc, "padding": pad,
            "added_tokens": added or [], "normalizer": normalizer, "pre_tokenizer": pre,
            "post_processor": post, "decoder": decoder, "model": model}


def added(content, id=0, special=True, normalized=False, **kw):     # noqa: A002 (the json field is "id")
    d = {"id": id, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": normalized, "special": special}
    d.update(kw)
    return d


def spm_model(holes=False, vocab=None, unk="<unk>", **over):
    """a sentencepiece-style bpe: no ByteLevel anywhere, the gemma/mistral component shapes"""
    v = ["<unk>", "<s>", "</s>", META] + ["%s%s" % (META, w) for w in
                                           ("a", "b", "c", "ab", "abc", "d", "e", "de")]
    v += [chr(0x4E2D), chr(0x6587)]
    v += ["<0x41>", "<0x42>", "<0xE4>", "<0xB8>", "<0xAD>", "<0xFF>"]
    v += ["t", "h", "th"]
    if holes:
        ids = {}
        nid = 0
        for s in v:
            if nid == 4:
                nid += 1          # a hole at id 4
            ids[s] = nid
            nid += 1
        vocab = ids
        over["merges"] = []        # the merge tokens' ids moved; no merge may reference them
    elif vocab is not None:
        over.setdefault("merges", [])   # a custom vocab's pieces are not the merge baseline's
    else:
        vocab = {s: i for i, s in enumerate(v)}
    d = {"type": "BPE", "dropout": None, "unk_token": unk, "continuing_subword_prefix": None,
         "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
         "ignore_merges": False, "vocab": vocab,
         "merges": [["t", "h"]]}
    d.update(over)
    return d


def spm_normalizer(prepend=META, replace=True):
    steps = []
    if replace:
        steps.append({"type": "Replace", "pattern": {"String": " "}, "content": META})
    if prepend is not None:
        steps.append({"type": "Prepend", "prepend": prepend})
    return {"type": "Sequence", "normalizers": steps} if steps else None


def unigram_model(pieces=None, unk_id=0, byte_fallback=False, **over):
    pieces = pieces if pieces is not None else [
        ("<unk>", 0.0), ("<s>", -1.0), ("</s>", -1.0), (META + "a", -2.0), (META + "b", -3.0),
        ("a", -4.0), ("b", -5.0), (META, -6.0), ("<0x41>", -7.0), ("<0xFF>", -8.0),
        (chr(0x4E2D), -9.0), ("th", -10.0)]
    d = {"type": "Unigram", "unk_id": unk_id, "byte_fallback": byte_fallback,
         "vocab": [list(p) for p in pieces]}
    d.update(over)
    return d


def wp_model(vocab=None, unk="[UNK]", prefix="##", maxc=100, **over):
    v = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "##a", "##b", "##c", "th", "##ing"]
    if vocab is not None:
        v = vocab
    d = {"type": "WordPiece", "unk_token": unk, "continuing_subword_prefix": prefix,
         "max_input_chars_per_word": maxc, "vocab": {s: i for i, s in enumerate(v)}}
    d.update(over)
    return d


def bert_norm(**over):
    d = {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
         "strip_accents": None, "lowercase": True}
    d.update(over)
    return d


def bert_pre():
    return {"type": "BertPreTokenizer"}


def wp_dec(prefix="##", cleanup=True):
    return {"type": "WordPiece", "prefix": prefix, "cleanup": cleanup}


# ---- the case registry ------------------------------------------------------------------------------------

CASES = []          # (category, name, bytes, description, expects)


def case(category, name, data, description, expects="accept"):
    """data: bytes / str / dict (a tokenizer.json) / {path: bytes} with 'dir' truthy in kw? -- a
    directory case passes a dict of leaf-name -> bytes and expects is unchanged; the writer makes a
    directory and the driver is given the directory."""
    if isinstance(data, dict) and not any(isinstance(v, bytes) for v in data.values()):
        data = jbytes(data)
    elif isinstance(data, str):
        data = data.encode("utf-8")
    CASES.append((category, name, data, description, expects))


def jbytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# =========================================================================================================
# json: the reader (json.c)
# =========================================================================================================

def gen_json():
    base = tok(bl_model(), pre=bytelevel())
    b = jbytes(base)

    # depth: TOKS_JSON_DEPTH 64 (st[0] is the root frame; a container at depth 64 is refused).
    # The nested value is a scalar so only the array nesting counts; the real model stays flat.
    for depth in (1, 2, 60, 61, 62, 63, 64, 80):
        arr = b"1"
        for _ in range(depth):
            arr = b"[" + arr + b"]"
        doc = b'{"version":"1.0","x":' + arr + b',"model":' + jbytes(bl_model()) + b"}"
        # containers on the stack at the deepest point: the root object (1) + depth arrays
        case("json", "depth%d" % depth, doc,
             "the root object + %d nested arrays = %d containers (the reader's limit is 64)" % (
                 depth, depth + 1),
             "accept" if depth + 1 <= 63 else "refuse")
    # the model itself nested in objects under another key: the reader's depth is stack frames, so the
    # model at the bottom of 60 objects (root + 60 + 1 = 62 frames) is inside the limit; "model" is a
    # second, flat member the reader reads by name
    obj = jbytes(bl_model())
    for _ in range(55):
        obj = b'{"m":' + obj + b"}"
    case("json", "model-nested-55", b'{"x":' + obj + b',"model":' + jbytes(bl_model()) + b"}",
         "an object tree 55 deep holding a copy of the model; the model the reader uses is flat", "accept")

    # duplicate keys: the LAST wins (map semantics); a vocab written twice with different sizes
    v1 = {s: i for i, s in enumerate(ALPHABET)}
    v2 = {s: i for i, s in enumerate(ALPHABET + ["he", "ll"])}
    doc = tok(bl_model(vocab=ALPHABET))
    d = json.loads(b)
    d["model"] = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": False, "vocab": v1, "merges": []}
    raw = jbytes(d)
    dup = jbytes(dict(type="BPE", vocab=v2, merges=[["h", "e"]]))
    case("json", "dup-model-key", raw[:-1] + b"," + dup[1:],
         "model twice: the last member must win (toks_jv_get keeps the last)", "accept")

    # giant arrays under unknown keys: 100k empty members (the member chain walk), and a 100k merges list
    d = json.loads(b)
    d["junk"] = [0] * 100000
    case("json", "junk-array-100k", jbytes(d), "an unused key holding 100k numbers (member walk)", "accept")
    d = json.loads(b)
    d["model"]["merges"] = [["h", "e"]] * 100000
    case("json", "merges-100k", jbytes(d), "100k duplicate merges (read_merges n bound, last-wins)", "accept")

    # truncation at every structural boundary
    for cut in (0, 1, 2, 5, 11, len(b) // 2, len(b) - 3, len(b) - 1):
        case("json", "cut-%d" % cut, b[:cut], "truncated at byte %d of %d" % (cut, len(b)), "refuse")
    # number shapes: id as float, negative, 2^63, 2^64-ish, exponent forms
    for name, val in (("float", 3.0), ("neg", -1), ("big", 2**63), ("bigger", 2**64 + 7),
                      ("exp", 3e2), ("nan", float("nan")), ("inf", float("inf"))):
        if val != val or val in (float("inf"), float("-inf")):
            text = "NaN" if val != val else "Infinity"
            raw = b'{"model":{"type":"BPE","vocab":{"a":' + text.encode() + b'},"merges":[]}}'
        else:
            raw = jbytes({"model": {"type": "BPE", "vocab": {"a": val}, "merges": []}})
        case("json", "id-%s" % name, raw, "a vocab id spelled %s (parse_num: float/overflow bits)" % name, "refuse")
    # string edges: lone surrogate escapes, overlong escapes, a 100 KB token string, NUL escapes
    case("json", "lone-surrogate", b'{"model":{"type":"BPE","vocab":{"\\ud800":"a"},"merges":[]}}',
         "a lone \\ud800 surrogate escape in a token string", "refuse")
    case("json", "surrogate-pair", b'{"model":{"type":"BPE","vocab":{"\\ud83d\\ude00":0,"a":1},"merges":[]}}',
         "a valid surrogate pair escape as a token", "accept")
    case("json", "nul-escape", b'{"model":{"type":"BPE","vocab":{"\\u0000":0,"a":1},"merges":[]}}',
         "a NUL escape inside a token string", "accept")
    big_tok = "x" * 100000
    case("json", "token-100k", jbytes({"model": {"type": "BPE", "vocab": {"a": 0, big_tok: 1}, "merges": []}}),
         "a 100 KB token string (TOKS_MAX_TOKEN_BYTES 65535)", "refuse")
    # whitespace storms and junk after the root
    case("json", "ws-storm", b" \t\r\n" + b + b" \t\r\n", "whitespace on both sides of the root", "accept")
    case("json", "junk-after", b + b" junk", "junk after the root value", "refuse")


# =========================================================================================================
# vocab: read_vocab (config.c)
# =========================================================================================================

def gen_vocab():
    # id holes: spm keeps them, byte-level needs dense
    hv = {}
    nid = 0
    for k in spm_model()["vocab"]:
        hv[k] = nid
        nid += 2 if nid == 2 else 1
    hv["z"] = 1000
    case("vocab", "holes-spm", tok(spm_model(vocab=hv),
         pre={"type": "Metaspace", "replacement": META, "prepend_scheme": "first", "split": False},
         decoder={"type": "Sequence", "decoders": [
             {"type": "Replace", "pattern": {"String": META}, "content": " "},
             {"type": "ByteFallback"}, {"type": "Fuse"}]}),
         "spm vocab with holes at 3,4,6,7,8 (kept, decode-only ids)", "accept")
    case("vocab", "holes-bytelevel", tok(bl_model(vocab=None), pre=bytelevel()),
         "dense check baseline (byte-level)", "accept")
    d = tok(bl_model(vocab=ALPHABET))
    d["model"]["vocab"] = {s: i for i, s in enumerate(ALPHABET)}
    d["model"]["vocab"]["zz"] = 300          # ids 0..255 then 300: 257 entries, max 300
    case("vocab", "not-dense", jbytes(d), "a byte-level vocab with ids 0..255 and 300 (not dense)", "refuse")

    # two strings share one id
    d = tok(bl_model())
    v = d["model"]["vocab"]
    v["hh"] = v["he"]
    case("vocab", "two-strings-one-id", jbytes(d), "hh and he both id 256", "refuse")

    # the same token twice (sidx_add: 'repeats a token')
    d = tok(bl_model())
    raw = jbytes(d)
    raw = raw.replace(b'"vocab":{', b'"vocab":{"he":300,', 1)
    case("vocab", "token-twice", raw, "he written twice with two ids", "refuse")

    # ids at the 2^21 boundary: 2097150, 2097151 (TOKS_MAX_IDS-1 ok), 2097152 (over)
    for name, id_ in (("2p21-2", 2**21 - 2), ("2p21-1", 2**21 - 1), ("2p21", 2**21)):
        d = tok(bl_model(vocab=["a", "b"]))
        d["model"]["vocab"] = {"a": 0, "b": id_}
        case("vocab", "id-%s" % name, jbytes(d), "a byte-level vocab id %d (dense + TOKS_MAX_IDS checks)" % id_,
             "refuse" if id_ >= 2**21 - 1 else "refuse")   # byte-level needs dense ids; spm keeps holes
    # max token bytes: 65535 ok for bpe, 65536 over; unigram 127/128
    d = tok(bl_model(vocab=["a", "y" * 65535]))
    d["model"]["vocab"] = {"a": 0, "y" * 65535: 1}
    case("vocab", "token-65535", jbytes(d), "a 65535-byte token (TOKS_MAX_TOKEN_BYTES)", "accept")
    d["model"]["vocab"] = {"a": 0, "y" * 65536: 1}
    d["model"]["merges"] = []
    case("vocab", "token-65536", jbytes(d), "a 65536-byte token", "refuse")
    case("vocab", "uni-piece-127", tok(unigram_model(pieces=[("<unk>", 0.0), ("y" * 127, -1.0)])),
         "a 127-byte unigram piece", "accept")
    case("vocab", "uni-piece-128", tok(unigram_model(pieces=[("<unk>", 0.0), ("y" * 128, -1.0)])),
         "a 128-byte unigram piece", "refuse")
    # empty vocab, empty string token
    d = tok(bl_model())
    d["model"]["vocab"] = {}
    case("vocab", "empty", jbytes(d), "an empty vocab", "refuse")
    d["model"]["vocab"] = {"": 0, "a": 1}
    d["model"]["merges"] = []
    case("vocab", "empty-string", jbytes(d), "an empty token string (hf: wp-specter2's empty token)", "accept")
    # vocab as an array (a unigram shape) under BPE
    d = tok(bl_model())
    d["model"]["vocab"] = [["a", 0]]
    d["model"]["merges"] = []
    case("vocab", "array-under-bpe", jbytes(d), "vocab as [piece, score] pairs under type BPE", "refuse")
    # the empty-piece unigram: kept, never yielded by the trie
    case("vocab", "uni-empty-piece", tok(unigram_model(pieces=[("<unk>", 0.0), ("", -1.0), ("a", -2.0)])),
         "an empty unigram piece (dropped by the trie)", "accept")


# =========================================================================================================
# merges: read_merges + bpe_build
# =========================================================================================================

def gen_merges():
    d0 = tok(bl_model())

    def with_merges(m):
        d = json.loads(jbytes(d0))
        d["model"]["merges"] = m
        return jbytes(d)

    # unknown tokens (left, right, product)
    case("merges", "unknown-left", with_merges([["zz", "a"]]), "a merge of a token not in the vocab", "refuse")
    case("merges", "unknown-product", with_merges([["h", "e", "he"]]),
         "a three-element merge entry (pair shape check)", "refuse")
    # cyclic: (a,b)->ab and (ab,a)->aba with aba in vocab
    v = ALPHABET + ["he", "eh", "heh", "ehe"]
    d = tok(bl_model(vocab=v))
    d["model"]["merges"] = [["h", "e"], ["e", "h"], ["he", "h"], ["e", "he"]]   # he, eh, heh, ehe
    case("merges", "cyclic", jbytes(d), "merges that rebuild each other's inputs (he<->ehe cycles)", "accept")
    # self-referential: (he, he) -> he
    d = tok(bl_model())
    d["model"]["merges"] = [["h", "e"], ["he", "he"]]
    case("merges", "self", jbytes(d), "a merge whose product is also its left (he + he -> he)", "refuse")
    # rank order contradicting the vocab: product id lower than both parts
    v = ALPHABET + ["ll", "he"]
    d = tok(bl_model(vocab=v))
    d["model"]["merges"] = [["l", "l"], ["h", "e"]]      # he (257) built from h(104),e(101): ids_as_rank check
    case("merges", "rank-vs-vocab", jbytes(d), "merge rank order contradicting vocab id order", "accept")
    # both formats mixed in one list; #version lines as strings
    d = tok(bl_model())
    d["model"]["merges"] = [["h", "e"], "l l", "#version: 0.2 - Koichi Yasuoka, Tue Oct 18 23:16:35 2022",
                             ["he", "ll"]]
    case("merges", "mixed-formats", jbytes(d), "pair and string merges mixed, with a #version line", "accept")
    # a merge string with zero or two spaces
    d = tok(bl_model())
    d["model"]["merges"] = ["he"]
    case("merges", "no-space", jbytes(d), "a string merge with no space", "refuse")
    d["model"]["merges"] = ["h  e"]
    case("merges", "two-spaces", jbytes(d), "a string merge with two spaces", "refuse")
    # a product that is not in the vocab
    d = tok(bl_model(vocab=ALPHABET))
    d["model"]["merges"] = [["h", "e"]]      # he not in this vocab
    case("merges", "unknown-out", jbytes(d), "a merge whose product is not in the vocab", "refuse")
    # duplicate merges: the last wins
    v = ALPHABET + ["he"]
    d = tok(bl_model(vocab=v))
    d["model"]["merges"] = [["h", "e"], ["h", "e"]]
    case("merges", "duplicate", jbytes(d), "the same merge twice (hf: last wins)", "accept")
    # 2^22 merges: over TOKS_PRIO_BITS
    if BIG:
        v = ALPHABET + ["h" * i for i in range(2, 2**22 + 2)]
        m = [["h" * i, "h"] for i in range(1, 2**22 + 1)]
        d = tok(bl_model(vocab=v))
        d["model"]["merges"] = m
        case("merges", "2p22", jbytes(d), "2^22 + 1 merges (TOKS_PRIO_BITS limit)", "refuse")


# =========================================================================================================
# added tokens: read_added + segment.c
# =========================================================================================================

def gen_added():
    v = bpe_vocab()

    def with_added(a):
        return jbytes(tok(bl_model(vocab=v), added=a, pre=bytelevel()))

    # prefixes of each other, longest-first must win
    a = [added("<|a|>", 300), added("<|ab|>", 301), added("<|abc|>", 302), added("<|", 303),
         added("<", 304)]
    case("added", "prefix-family", with_added(a), "added tokens that are prefixes of each other", "accept")
    # contents equal to vocab strings (the id rule: content names the vocab token)
    a = [added("he", 0), added("hello world", 400), added("Ġ", 32)]
    case("added", "vocab-strings", with_added(a), "contents equal to vocabulary strings take their ids", "accept")
    # empty content (hf ignores) then a huge one (255 ok, 256 over)
    a = [added("", 500), added("x" * 255, 501)]
    case("added", "empty-and-255", with_added(a), "an empty content (ignored) and a 255-byte one", "accept")
    a = [added("x" * 256, 502)]
    case("added", "bytes-256", with_added(a), "a 256-byte content (TOKS_MAX_ADDED_BYTES 255)", "refuse")
    # invalid utf-8 can't be spelled in json, but escapes can build odd strings: control chars via \u0000
    a = [added("\x00\x01\x1f", 503)]
    case("added", "control-chars", with_added(a), "content with NUL and control bytes", "accept")
    # lstrip / rstrip / single_word: every combination on whitespace-ish content
    a = [added(" ", 504, lstrip=True), added("\n", 505, single_word=True),
         added("\t", 506, lstrip=True), added("\r", 507, single_word=True)]
    case("added", "strip-flags", with_added(a), "strip flags on whitespace contents", "accept")
    # all-whitespace lstrip token beside rstrip tokens (hf panics: the reader refuses)
    a = [added("<|r|>", 508, rstrip=True), added("  ", 509, lstrip=True)]
    case("added", "ws-lstrip-beside-rstrip", with_added(a), "an all-whitespace lstrip token beside rstrip tokens",
         "refuse")
    # ids colliding with vocab ids and with each other (same content twice: flags merge, id keeps first)
    a = [added("dup", 600), added("dup", 601, special=True)]
    case("added", "content-twice", with_added(a), "the same content twice: specialness sticks, first id keeps",
         "accept")
    # 65535 entries (ok) is too slow to build as json; 300 is enough to stress the sidx and cand arrays
    a = [added("<|t%d|>" % i, 1000 + i) for i in range(300)]
    case("added", "many-300", with_added(a), "300 added tokens (h4 index, radix trees)", "accept")
    # an added token whose id is past TOKS_MAX_IDS
    a = [added("<|big|>", 2**21)]
    case("added", "id-2p21", with_added(a), "an added id field at 2^21 (hf re-ids: the field is ignored)",
         "accept")
    # normalized=true under an NFC normalizer whose form changes the content (refused: phase-1 on folded text)
    a = [added("éx", 700, normalized=True)]
    case("added", "normalized-nfc-changes", jbytes(tok(bl_model(vocab=v), added=a, normalizer={"type": "NFC"},
                                                       pre=bytelevel())),
         "normalized=true whose content NFC changes (éx)", "refuse")
    a = [added("éx", 701, normalized=True)]        # precomposed: NFC-stable
    case("added", "normalized-nfc-stable", jbytes(tok(bl_model(vocab=v), added=a, normalizer={"type": "NFC"},
                                                       pre=bytelevel())),
         "normalized=true whose content NFC leaves alone", "accept")
    # single_word at word boundaries
    a = [added("mid", 702, single_word=True)]
    case("added", "single-word", with_added(a), "single_word=true (xmidx must not match)", "accept")


# =========================================================================================================
# normalizers: bl_normalizer / uni / spm readers + precompiled.c
# =========================================================================================================

def gen_normalizer():
    # NFKC expansion bombs: U+FDFA -> 18 chars; a file whose every byte is one
    d = tok(bl_model(vocab=bpe_vocab()), normalizer={"type": "NFKC"}, pre=bytelevel())
    case("normalizer", "nfkc", jbytes(d), "NFKC (U+FDFA expands 3 bytes -> 18 chars; the scratch's 11x)", "accept")
    d = tok(bl_model(vocab=bpe_vocab()), normalizer={"type": "NFKD"}, pre=bytelevel())
    case("normalizer", "nfkd-bytelevel", jbytes(d), "NFKD under byte-level bpe (refused: NFC/NFKC only)",
         "refuse")
    d = tok(unigram_model(), normalizer={"type": "NFKD"})
    case("normalizer", "nfkd-unigram", jbytes(d), "NFKD under unigram (accepted there)", "accept")
    # both in a sequence (an NFKC anywhere makes it NFKC)
    d = tok(bl_model(), normalizer={"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "NFKC"}]},
            pre=bytelevel())
    case("normalizer", "nfc-then-nfkc", jbytes(d), "NFC then NFKC in a sequence (the mix is NFKC)", "accept")
    # Replace with an enormous content (16-step chain of long replacements)
    steps = [{"type": "Replace", "pattern": {"String": "x"}, "content": "y" * 1000} for _ in range(8)]
    d = tok(bl_model(), normalizer={"type": "Sequence", "normalizers": steps}, pre=bytelevel())
    case("normalizer", "replace-huge", jbytes(d), "Replace with a 1000-byte content x8 (byte-level refuses)", "refuse")
    # unigram: the accepted Replace shapes and their order
    ok = [{"type": "Replace", "pattern": {"String": "``"}, "content": "\""},
          {"type": "NFKC"},
          {"type": "Replace", "pattern": {"String": " "}, "content": META}]
    d = tok(unigram_model(), normalizer={"type": "Sequence", "normalizers": ok})
    case("normalizer", "uni-order-ok", jbytes(d), "unigram Replace(String), then ' '->▁, then NFKC (order ok)", "accept")
    bad = [{"type": "NFKC"}, {"type": "Replace", "pattern": {"String": "``"}, "content": "\""}]
    d = tok(unigram_model(), normalizer={"type": "Sequence", "normalizers": bad})
    case("normalizer", "uni-order-bad", jbytes(d), "unigram Replace AFTER NFKC (order refused)", "refuse")
    bad = [{"type": "Replace", "pattern": {"Regex": "[abc]+"}, "content": "x"}]
    d = tok(unigram_model(), normalizer={"type": "Sequence", "normalizers": bad})
    case("normalizer", "uni-replace-regex", jbytes(d), "unigram Replace(Regex) of an unsupported shape", "refuse")
    # a Replace whose content is longer than its pattern (albert's are <=; longer refused)
    bad = [{"type": "Replace", "pattern": {"String": "`"}, "content": "xxxx"}]
    d = tok(unigram_model(), normalizer={"type": "Sequence", "normalizers": bad})
    case("normalizer", "uni-replace-grow", jbytes(d), "unigram Replace String -> longer content", "refuse")
    # the two named collapse/prefix regex replaces, in the wrong order (prefix before collapse)
    steps = [{"type": "Replace", "pattern": {"Regex": "(?<!\\n)^"}, "content": META},
             {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]
    d = tok(unigram_model(), normalizer={"type": "Sequence", "normalizers": steps})
    case("normalizer", "uni-prefix-before-collapse", jbytes(d),
         "Replace ^->▁ before ' {2,}'->' ' (wrong order)", "refuse")
    # spm: an 8-step chain at the limit and one over
    # 8 spm Replace steps whose contents are all vocab chars (the fold's requirement), then 9
    m = spm_model()
    # 8 distinct one-char vocab strings (the fold maps each to another vocab char), plus 'z'-'w' style
    # extras in the vocab for the mapping targets
    one = [k for k in m["vocab"] if len(k) == 1]           # ▁ 中 文 t h
    extra = "pqrsvwxy"
    v = dict(m["vocab"])
    nid = max(v.values()) + 1
    for c in extra:
        v[c] = nid
        nid += 1
    m["vocab"] = v
    keys = one + list(extra)                 # 13 one-char vocab strings: 6 pairs + 1 spare
    steps = [{"type": "Replace", "pattern": {"String": keys[2 * i]}, "content": keys[2 * i + 1]}
             for i in range(6)]
    d = tok(m, normalizer={"type": "Sequence", "normalizers": steps},
            pre={"type": "Metaspace", "replacement": META, "prepend_scheme": "always", "split": False},
            decoder={"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]})
    case("normalizer", "spm-6-steps", jbytes(d),
         "6 spm Replace steps over one-char vocab strings (the 8-op limit leaves room)", "accept")
    for k in range(3):
        steps.append({"type": "Replace", "pattern": {"String": keys[12 + k % 1]}, "content": keys[0]})
    d = json.loads(jbytes(d))
    d["normalizer"] = {"type": "Sequence", "normalizers": steps[:9]}
    case("normalizer", "spm-9-steps", jbytes(d), "9 spm normalizer steps (over the 8-op limit)", "refuse")

    # ---- precompiled charsmap (precompiled.c): the double-array trie -------------------------
    # a minimal valid charsmap: root node, 'a' -> identity is implicit; build "aa" -> "b"
    def charsmap(units, strings):
        blob = struct.pack("<I", len(units) * 4) + b"".join(struct.pack("<I", u) for u in units) + strings
        return base64.b64encode(blob).decode()

    # well-formed: root node 0 (unit[0] = 0), the child 'a' at slot 0^97 = 97 as a leaf whose
    # offset o makes the child node nxt = 97 ^ du_offset(o); the unit at nxt is the value slot: its
    # leaf bit set, du_value = the strings offset. The array must cover every slot the walk touches
    # from any reachable node: the root alone touches 1..255.
    units = [0] * 512
    u = 97 | (1 << 8) | (96 << 10)        # label 'a', leaf, offset 96
    units[97] = u
    nxt = 97 ^ du_off(96 << 10)           # 97 ^ 96 = 1
    units[nxt] = 0                         # the value slot: value 0 -> strings[0:] = "b"
    blob_units = units
    strings = b"b\x00"
    case("normalizer", "charsmap-valid", tok(unigram_model(), normalizer={"type": "Precompiled",
         "precompiled_charsmap": charsmap(blob_units, strings)}),
         "a precompiled charsmap mapping 'aa' -> 'b' (hand-built double array)", "accept")
    # corrupt base64
    case("normalizer", "charsmap-bad-b64", tok(unigram_model(), normalizer={"type": "Precompiled",
         "precompiled_charsmap": "!!!!"}),
         "a charsmap whose base64 is not base64", "refuse")
    # a trie pointing outside the array (hf panics; toks refuses)
    units_bad = [0] * 4
    units_bad[0] = 0
    units_bad[1] = 97 | (1 << 8) | (0xFFFF << 10)   # offset past the array
    case("normalizer", "charsmap-trie-oob", tok(unigram_model(), normalizer={"type": "Precompiled",
         "precompiled_charsmap": charsmap(units_bad, b"xy")}),
         "a charsmap whose trie offset leaves the array", "refuse")
    # a value past the strings
    units_v = [0] * 200
    units_v[97] = 97 | (1 << 8) | (99 << 10)
    units_v[97 ^ du_off(99 << 10)] = 0xFFFFFF       # value 16M, strings 2 bytes
    case("normalizer", "charsmap-value-oob", tok(unigram_model(), normalizer={"type": "Precompiled",
         "precompiled_charsmap": charsmap(units_v, b"b\x00")}),
         "a charsmap whose value index is past its strings", "refuse")
    # strings not utf-8
    case("normalizer", "charsmap-strings-utf8", tok(unigram_model(), normalizer={"type": "Precompiled",
         "precompiled_charsmap": charsmap(units_v, b"\xff\xfe\x00")}),
         "a charsmap whose strings are not utf-8", "refuse")
    # the real nmt_nfkc map (from a cached unigram file) truncated at every 64 bytes
    src = None
    for p in (os.path.expanduser("~/.cache/toks/tokenizers/uni_t5base/tokenizer.json"),
              os.path.expanduser("~/.cache/toks/tokenizers/uni_albert/tokenizer.json")):
        try:
            src = json.load(open(p, encoding="utf-8"))["normalizer"]["precompiled_charsmap"]
            break
        except Exception:
            continue
    if src is not None:
        raw = base64.b64decode(src)
        for cut in (0, 1, 64, 1024, len(raw) // 2, len(raw) - 1):
            if cut >= len(raw):
                continue
            blob = base64.b64encode(raw[:cut]).decode()
            case("normalizer", "charsmap-trunc-%d" % cut,
                 tok(unigram_model(), normalizer={"type": "Precompiled", "precompiled_charsmap": blob}),
                 "the real nmt charsmap truncated at %d of %d bytes" % (cut, len(raw)), "refuse")
        # byte-flipped offsets in the real map: one unit's offset field randomized (deterministic)
        raw2 = bytearray(raw)
        off = 4 + 4 * 3            # a unit in the middle
        u = struct.unpack_from("<I", raw2, off)[0]
        raw2[off:off + 4] = struct.pack("<I", u ^ (0x3FF << 10))
        case("normalizer", "charsmap-flip", tok(unigram_model(), normalizer={"type": "Precompiled",
             "precompiled_charsmap": base64.b64encode(bytes(raw2)).decode()}),
             "the real nmt charsmap with one unit's trie offset flipped", "refuse:or-refuses-clean")


def du_off(u):
    return (u >> 10) << ((u & (1 << 9)) >> 6)


# =========================================================================================================
# pre-tokenizers: bl_pretok + gen.c (the generic engine)
# =========================================================================================================

def gen_pretok():
    # a known template's exact string, one byte off
    for name, pat, exp in (("gpt2-one-off", GPT2.replace("'s|", "'S|", 1), "accept"),
                           ("cl100k-one-off", CL100K.replace("{1,3}", "{1,4}"), "accept"),
                           ("gpt2-lookbehind", r"(?<=a)\s+|\s+", "refuse"),
                           ("gpt2-nested", r"(?:(?:'s)|'t)", "accept")):
        d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [split_re(pat), bytelevel()]})
        case("pretok", name, jbytes(d),
             "a near-template pattern the exact-match compiler must miss: %s" % pat, exp)
    # a pattern that is exactly a known one but through a String Split
    d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"String": GPT2}, "behavior": "Isolated", "invert": False}, bytelevel()]})
    case("pretok", "template-as-string", jbytes(d), "the gpt-2 pattern as a String (not Regex)", "accept")
    # 16 steps (the reader's step limit) and 17
    steps = [split_re(r"[0-9]+")] * 15 + [bytelevel()]
    d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": steps})
    case("pretok", "16-steps", jbytes(d), "16 pre-tokenizer steps (the reader's limit)", "accept")
    steps = [split_re(r"[0-9]+")] * 16 + [bytelevel()]
    d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": steps})
    case("pretok", "17-steps", jbytes(d), "17 pre-tokenizer steps (over the limit)", "refuse")
    # Sequences nested to depth 4 and 5
    def nest(depth):
        node = bytelevel()
        for _ in range(depth):
            node = {"type": "Sequence", "pretokenizers": [node]}
        return node
    d = tok(bl_model(), pre=nest(3))
    case("pretok", "nested-3", jbytes(d), "ByteLevel inside 3 nested Sequences", "accept")
    d = tok(bl_model(), pre=nest(5))
    case("pretok", "nested-5", jbytes(d), "ByteLevel inside 5 nested Sequences (over depth 4)", "refuse")
    # every behavior and invert
    for beh in ("Isolated", "Removed", "MergedWithPrevious", "MergedWithNext", "Contiguous"):
        for inv in (False, True):
            d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [split_re(r"\s+", beh, inv), bytelevel()]})
            case("pretok", "beh-%s-%s" % (beh, inv), jbytes(d),
                 "Split \\s+ behavior %s invert %s (generic engine)" % (beh, inv), "accept")
    # patterns with every construct the generic engine may meet
    for name, pat in (("class-punct", r"\p{P}+"), ("class-neg", r"[^\p{L}\p{N}]+"),
                      ("lookahead", r"a(?=b)"), ("lookbehind-neg", r"(?<!a)b"),
                      ("backref", r"(a)\1"), ("atomic", r"(?>ab)"), ("possessive", r"[0-9]++"),
                      ("named", r"(?<x>a)"), ("flag", r"(?i)abc"), ("flag-off", r"(?i:a(?-i:b))"),
                      ("empty", r"(?:)"), ("anchors", r"^[a-z]+$"), ("bound-word", r"\ba\b"),
                      ("hex-esc", r"\x41+"), ("unicode-prop", r"\p{Script=Latin}+"),
                      ("alt-empty", r"a|"), ("nested-quant", r"(a{2,3}){2,3}"),
                      ("dotall", r"(?s).*"), ("multiline", r"(?m)^x")):
        d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [split_re(pat), bytelevel()]})
        case("pretok", "generic-%s" % name, jbytes(d), "generic engine pattern: %s" % pat, "accept:or-refuses")
    # an empty pattern, an unbalanced one, an evil one
    for name, pat in (("empty", ""), ("unbalanced", "(a"), ("star-of-star", "(a*)*b"),
                      ("catastrophic", "(a+)+$"), ("case-ins-class", "(?i)[A-Z]+")):
        d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [split_re(pat), bytelevel()]})
        case("pretok", "bad-%s" % name, jbytes(d), "pattern %r (compile or refuse, never hang)" % pat,
             "accept:or-refuses")
    # Digits / Punctuation shapes
    d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [
        {"type": "Digits", "individual_digits": True}, bytelevel()]})
    case("pretok", "digits-indiv", jbytes(d), "Digits(individual_digits=true) then ByteLevel", "accept")
    d = tok(bl_model(), pre={"type": "Sequence", "pretokenizers": [
        {"type": "Punctuation", "behavior": "Isolated"}, bytelevel()]})
    case("pretok", "punct-isolated", jbytes(d), "Punctuation(Isolated) then ByteLevel", "accept")


# =========================================================================================================
# post-processor templates: read_template / read_post_processor
# =========================================================================================================

def gen_template():
    v = bpe_vocab()

    def tmpl(single, specials):
        return {"type": "TemplateProcessing", "single": single,
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": specials}

    def one(single, specials):
        return jbytes(tok(bl_model(vocab=v), post=tmpl(single, specials)))

    A = {"Sequence": {"id": "A", "type_id": 0}}
    # 63 and 64 pieces around $A (the flat[64] limit is pieces after flattening: specials' ids count too)
    single = [A] + [{"SpecialToken": {"id": "s", "type_id": 0}}] * 62
    case("template", "pieces-63", one(single, {"s": {"id": "s", "ids": [1], "tokens": ["t"]}}),
         "63 template pieces (flat limit 64)", "accept")
    single = [A] + [{"SpecialToken": {"id": "s", "type_id": 0}}] * 63
    case("template", "pieces-64", one(single, {"s": {"id": "s", "ids": [1], "tokens": ["t"]}}),
         "64 template pieces (flat limit 64)", "accept")
    single = [A] + [{"SpecialToken": {"id": "s", "type_id": 0}}] * 64
    case("template", "pieces-65", one(single, {"s": {"id": "s", "ids": [1], "tokens": ["t"]}}),
         "65 template pieces (over flat[64])", "refuse")
    # a special token whose ids list expands to 70 ids ($A + one special of 70 ids)
    single = [A, {"SpecialToken": {"id": "s", "type_id": 0}}]
    case("template", "ids-70", one(single, {"s": {"id": "s", "ids": list(range(70)), "tokens": ["t"] * 70}}),
         "one special token with 70 ids (flat[64] over)", "refuse")
    # unknown special token (not in special_tokens)
    single = [A, {"SpecialToken": {"id": "nope", "type_id": 0}}]
    case("template", "unknown-special", one(single, {"s": {"id": "s", "ids": [1], "tokens": ["t"]}}),
         "a SpecialToken not in special_tokens", "refuse")
    # an id beyond the vocabulary
    single = [A, {"SpecialToken": {"id": "s", "type_id": 0}}]
    case("template", "id-beyond-vocab", one(single, {"s": {"id": "s", "ids": [99999], "tokens": ["t"]}}),
         "a template id beyond the vocabulary", "refuse")
    # $A twice / missing; $B in single
    case("template", "a-twice", one([A, A], {}), "$A twice in single", "refuse")
    case("template", "no-a", one([{"SpecialToken": {"id": "s", "type_id": 0}}], {}), "no $A in single", "refuse")
    case("template", "b-in-single", one([{"Sequence": {"id": "B", "type_id": 0}}], {}), "$B in single", "refuse")
    # RobertaProcessing: cls/sep pairs, one missing, ids beyond vocab
    case("template", "roberta", jbytes(tok(bl_model(vocab=v), post={"type": "RobertaProcessing",
         "sep": ["</s>", 2], "cls": ["<s>", 1], "trim_offsets": True, "add_prefix_space": False})),
         "RobertaProcessing with cls/sep", "accept")
    case("template", "roberta-no-cls", jbytes(tok(bl_model(vocab=v), post={"type": "RobertaProcessing",
         "sep": ["</s>", 2], "trim_offsets": True})), "RobertaProcessing without cls", "refuse")
    case("template", "roberta-id-oob", jbytes(tok(bl_model(vocab=v), post={"type": "RobertaProcessing",
         "sep": ["</s>", 2], "cls": ["<s>", 99999]})), "RobertaProcessing with an id beyond the vocab", "refuse")
    # BertProcessing
    case("template", "bert", jbytes(tok(bl_model(vocab=v), post={"type": "BertProcessing",
         "sep": ["[SEP]", 3], "cls": ["[CLS]", 2]})), "BertProcessing with cls/sep", "accept")
    # a Sequence of ByteLevel + Template (the accepted census shape)
    case("template", "seq-bl-tmpl", jbytes(tok(bl_model(vocab=v), post={"type": "Sequence", "processors": [
        {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        tmpl([A, {"SpecialToken": {"id": "s", "type_id": 0}}], {"s": {"id": "s", "ids": [1], "tokens": ["t"]}})]})),
         "Sequence[ByteLevel, TemplateProcessing]", "accept")


# =========================================================================================================
# truncation / padding: read_trunc_pad + trunc_check
# =========================================================================================================

def gen_truncpad():
    v = bpe_vocab()

    def tp(trunc=None, pad=None):
        return jbytes(tok(bl_model(vocab=v), trunc=trunc, pad=pad))

    def tr(**kw):
        d = {"max_length": 20, "stride": 0, "strategy": "LongestFirst", "direction": "Right"}
        d.update(kw)
        return d

    def pd(**kw):
        d = {"strategy": {"Fixed": 10}, "direction": "Right", "pad_to_multiple_of": None,
             "pad_id": 0, "pad_type_id": 0, "pad_token": "[PAD]"}
        d.update(kw)
        return d

    case("truncpad", "trunc-max-0", tp(trunc=tr(max_length=0)), "truncation max_length 0 (every encode: 0 ids)",
         "accept")
    case("truncpad", "trunc-max-1", tp(trunc=tr(max_length=1)), "truncation max_length 1", "accept")
    case("truncpad", "trunc-stride-eq", tp(trunc=tr(max_length=20, stride=20)), "stride == max_length (hf panics)",
         "refuse")
    case("truncpad", "trunc-stride-over", tp(trunc=tr(max_length=5, stride=6)), "stride > max_length (hf panics)",
         "refuse")
    case("truncpad", "trunc-stride-under", tp(trunc=tr(max_length=5, stride=4)), "stride = max - 1", "accept")
    case("truncpad", "trunc-left", tp(trunc=tr(direction="Left")), "truncation direction Left", "refuse")
    case("truncpad", "trunc-onlysecond", tp(trunc=tr(strategy="OnlySecond")), "truncation OnlySecond", "refuse")
    case("truncpad", "trunc-2p29", tp(trunc=tr(max_length=2**29)), "max_length 2^29 (clamped to TOKS_MAX_TEXT)",
         "accept")
    case("truncpad", "trunc-2p29-plus", tp(trunc=tr(max_length=2**29 + 1)), "max_length 2^29 + 1 (clamped)", "accept")
    case("truncpad", "pad-fixed-2p29", tp(pad=pd(strategy={"Fixed": 2**29})), "padding Fixed 2^29", "accept")
    case("truncpad", "pad-fixed-2p29-plus", tp(pad=pd(strategy={"Fixed": 2**29 + 1})), "padding Fixed 2^29 + 1",
         "refuse")
    case("truncpad", "pad-mul-2p20", tp(pad=pd(pad_to_multiple_of=2**20)), "pad_to_multiple_of 2^20", "accept")
    case("truncpad", "pad-mul-2p20-plus", tp(pad=pd(pad_to_multiple_of=2**20 + 1)), "pad_to_multiple_of 2^20 + 1",
         "refuse")
    case("truncpad", "pad-id-oob", tp(pad=pd(pad_id=99999)), "pad_id beyond the vocabulary", "refuse")
    case("truncpad", "pad-left", tp(pad=pd(direction="Left")), "padding direction Left", "accept")
    case("truncpad", "pad-batchlongest", tp(pad=pd(strategy="BatchLongest")), "padding BatchLongest (a no-op for one)",
         "accept")
    case("truncpad", "pad-no-strategy", tp(pad={"direction": "Right", "pad_id": 0, "pad_type_id": 0,
                                                 "pad_token": "x"}), "padding without a strategy", "refuse")


# =========================================================================================================
# unigram models: parse_unigram + unigram.c
# =========================================================================================================

def gen_unigram():
    # scores: the shapes serde refuses and accepts (parse_score mirrors serde_json 1.0.151)
    for name, score, exp in (
            ("nan", "NaN", "refuse"), ("inf", "Infinity", "refuse"),
            ("1e309", "1e309", "refuse"), ("-1e309", "-1e309", "refuse"),
            ("1e308", "1e308", "accept"), ("1e-309", "1e-309", "accept"),
            ("5e-324", "5e-324", "accept"), ("17976931348623157e292", "17976931348623157e292", "accept"),
            ("-0", "-0", "accept"), ("1e+", "1e+", "refuse"), (".5", ".5", "refuse"),
            ("1.", "1.", "refuse"), ("1_000", "1_000", "refuse"), ("0x10", "0x10", "refuse"),
            ("1e2147483648", "1e2147483648", "refuse"), ("1e-2147483648", "1e-2147483648", "accept")):
        # build directly: a json text with the raw lexeme as the score
        raw = ('{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,'
               '"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"Unigram",'
               '"unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0.0],["a",' + score + ']]}}')
        case("unigram", "score-%s" % name, raw, "an unigram score spelled %s (parse_score / serde)" % score, exp)
    # byte fallback with missing byte pieces: the uni_resolve class (spaces under the ▁ Replace)
    pieces = [("<unk>", 0.0), ("<0xE2>", -1.0), ("<0x96>", -2.0), ("<0x81>", -3.0), ("a", -4.0)]
    d = tok(unigram_model(pieces=pieces, byte_fallback=True),
            normalizer={"type": "Sequence", "normalizers": [
                {"type": "Replace", "pattern": {"String": " "}, "content": META}]},
            decoder={"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": META},
                                                       "content": " "},
                                                      {"type": "ByteFallback"}, {"type": "Fuse"}]})
    case("unigram", "bf-spaces-no-meta-piece", jbytes(d),
         "byte fallback + ' '->▁ Replace, no ▁ piece: one space = 3 ids (the uni_resolve class)", "accept")
    # byte fallback missing one byte piece entirely
    pieces = [("<unk>", 0.0)] + [("<0x%02X>" % b, -1.0) for b in range(255)]      # no <0xFF>
    d = tok(unigram_model(pieces=pieces, byte_fallback=True))
    case("unigram", "bf-missing-0xFF", jbytes(d), "byte fallback with 255 of the 256 byte pieces", "accept")
    # unk_id: missing, in range, out of range, -1
    d = json.loads(jbytes(tok(unigram_model())))
    d["model"]["unk_id"] = None
    case("unigram", "unk-null", jbytes(d), "unk_id null (hf: MissingUnkId on unknown chars)", "refuse")
    d = json.loads(jbytes(tok(unigram_model())))
    del d["model"]["unk_id"]
    case("unigram", "unk-absent", jbytes(d), "unk_id deleted", "refuse")
    d = tok(unigram_model(unk_id=5))
    case("unigram", "unk-5", jbytes(d), "unk_id 5 in a 12-piece vocab", "accept")
    d = tok(unigram_model(unk_id=12))
    case("unigram", "unk-out", jbytes(d), "unk_id == n (outside the vocab)", "refuse")
    d = tok(unigram_model(unk_id=-1))
    case("unigram", "unk-neg", jbytes(d), "unk_id -1", "refuse")
    # duplicate pieces
    d = tok(unigram_model(pieces=[("<unk>", 0.0), ("a", -1.0), ("a", -2.0)]))
    case("unigram", "dup-piece", jbytes(d), "the same piece twice", "refuse")
    # a piece that is exactly a byte token spelling but lowercase / with a plus (hf's radix accepts +)
    d = tok(unigram_model(pieces=[("<unk>", 0.0), ("<0x41>", -1.0), ("<0x+41>", -2.0), ("<0xg1>", -3.0)]))
    case("unigram", "byte-spellings", jbytes(d), "byte-token spellings with a + and a bad digit", "accept")
    # the metaspace families
    for scheme in ("always", "first", "never"):
        d = tok(unigram_model(),
                pre={"type": "Metaspace", "replacement": META, "prepend_scheme": scheme, "split": True})
        case("unigram", "meta-%s" % scheme, jbytes(d), "Metaspace prepend_scheme %s alone" % scheme,
             "accept" if scheme != "first" else "refuse")
    # decoder chains: the census shapes and wrong orders
    ok = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": META}, "content": " "},
                                           {"type": "ByteFallback"}, {"type": "Fuse"}]}
    case("unigram", "dec-rbf", jbytes(tok(unigram_model(), decoder=ok)), "Replace>ByteFallback>Fuse", "accept")
    bad = {"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"},
                                            {"type": "Replace", "pattern": {"String": META}, "content": " "}]}
    case("unigram", "dec-wrong-order", jbytes(tok(unigram_model(), decoder=bad)), "Fuse before ByteFallback",
         "refuse")
    # truncation under unigram
    d = tok(unigram_model(), trunc={"max_length": 4, "stride": 0, "strategy": "LongestFirst", "direction": "Right"})
    case("unigram", "trunc-4", jbytes(d), "unigram with truncation max_length 4", "accept")
    d = tok(unigram_model(), pad={"strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": None,
                                  "pad_id": 0, "pad_type_id": 0, "pad_token": "x"})
    case("unigram", "pad-refused", jbytes(d), "unigram with padding Fixed (refused: no pad under unigram)",
         "refuse")


# =========================================================================================================
# wordpiece: parse_wordpiece + wp.c
# =========================================================================================================

def gen_wordpiece():
    def wtok(model, normalizer=None, pre=None, dec=None, **kw):
        return jbytes(tok(model, normalizer=normalizer, pre=pre or bert_pre(), post=None,
                          decoder=dec or wp_dec(), **kw))

    case("wordpiece", "maxc-0", wtok(wp_model(maxc=0)), "max_input_chars_per_word 0 (every word is [UNK])",
         "accept")
    case("wordpiece", "maxc-1024", wtok(wp_model(maxc=1024)), "max_input_chars_per_word 1024 (the cap)",
         "accept")
    case("wordpiece", "maxc-1025", wtok(wp_model(maxc=1025)), "max_input_chars_per_word 1025 (over the cap)",
         "refuse")
    case("wordpiece", "maxc-2p31", wtok(wp_model(maxc=2**31)), "max_input_chars_per_word 2^31 (over the cap)",
         "refuse")
    case("wordpiece", "prefix-empty", wtok(wp_model(prefix="")), "continuing_subword_prefix empty", "accept")
    case("wordpiece", "prefix-16", wtok(wp_model(prefix="x" * 16)), "continuing_subword_prefix 16 bytes", "accept")
    case("wordpiece", "prefix-17", wtok(wp_model(prefix="x" * 17)), "continuing_subword_prefix 17 bytes (the "
         "reader takes any string; the ## pieces must exist in the vocab)", "accept")
    case("wordpiece", "prefix-100", wtok(wp_model(prefix="x" * 100)),
         "continuing_subword_prefix 100 bytes (a 100-byte prefix token: piece length stress)", "accept")
    # unk_token not in vocab; a 100-byte unk
    case("wordpiece", "unk-missing", wtok(wp_model(vocab=["[PAD]", "a", "##a"], unk="[UNK]")),
         "unk_token not in the vocab", "refuse")
    case("wordpiece", "unk-100", wtok(wp_model(vocab=["[PAD]", "[UNK]" + "x" * 94, "a", "##a"],
                                               unk="[UNK]" + "x" * 94)),
         "a 98-byte unk_token (it IS in the vocab)", "accept")
    # BertNormalizer field shapes
    case("wordpiece", "bert-null-strip", wtok(wp_model(), normalizer=bert_norm(strip_accents=None)),
         "BertNormalizer strip_accents null (follows lowercase)", "accept")
    case("wordpiece", "bert-strip-false", wtok(wp_model(), normalizer=bert_norm(strip_accents=False)),
         "BertNormalizer strip_accents false", "accept")
    case("wordpiece", "bert-all-off", wtok(wp_model(), normalizer=bert_norm(clean_text=False,
         handle_chinese_chars=False, strip_accents=False, lowercase=False)),
         "BertNormalizer every flag off", "accept")
    case("wordpiece", "bert-bad-field", wtok(wp_model(), normalizer={"type": "BertNormalizer", "lowercase": "yes"}),
         "BertNormalizer lowercase as a string", "refuse")
    # two BertNormalizers
    d = json.loads(wtok(wp_model()))
    d["normalizer"] = {"type": "Sequence", "normalizers": [bert_norm(), bert_norm()]}
    case("wordpiece", "bert-twice", jbytes(d), "BertNormalizer twice in a Sequence", "refuse")
    # decoder shapes
    case("wordpiece", "dec-prefix-16", wtok(wp_model(), dec=wp_dec(prefix="x" * 16)),
         "WordPiece decoder prefix 16 bytes", "accept")
    case("wordpiece", "dec-prefix-17", wtok(wp_model(), dec=wp_dec(prefix="x" * 17)),
         "WordPiece decoder prefix 17 bytes", "refuse")
    case("wordpiece", "dec-cleanup-false", wtok(wp_model(), dec=wp_dec(cleanup=False)),
         "WordPiece decoder cleanup false", "accept")
    # no decoder at all (hf joins with spaces)
    case("wordpiece", "dec-none", wtok(wp_model(), dec=None), "no decoder (hf joins with spaces)", "accept")
    # a vocab whose strings are the ## forms only, a one-char vocab
    case("wordpiece", "only-hash", wtok(wp_model(vocab=["[PAD]", "[UNK]", "##a", "##b"])),
         "a vocab with only ## pieces", "accept")
    # truncation / padding under wordpiece
    case("wordpiece", "trunc-1", wtok(wp_model(maxc=100), trunc={"max_length": 1, "stride": 0,
         "strategy": "LongestFirst", "direction": "Right"}), "truncation max_length 1 below the template's ids"
         " (no template: ok)", "accept")
    case("wordpiece", "pad-fixed", wtok(wp_model(), pad={"strategy": {"Fixed": 8}, "direction": "Right",
         "pad_to_multiple_of": None, "pad_id": 1, "pad_type_id": 0, "pad_token": "[PAD]"}),
         "padding Fixed 8 with pad_id 1", "accept")
    case("wordpiece", "pad-mul", wtok(wp_model(), pad={"strategy": "BatchLongest", "direction": "Right",
         "pad_to_multiple_of": 2, "pad_id": 1, "pad_type_id": 0, "pad_token": "[PAD]"}),
         "padding BatchLongest + pad_to_multiple_of 2", "accept")


# =========================================================================================================
# spm: parse_spm (sentencepiece-style bpe)
# =========================================================================================================

def gen_spm():
    """the sentencepiece-style families as the census spells them: gemma's Replace + Split fold,
    tinyllama's Prepend fold with no pre-tokenizer, mistral's Metaspace; each attacked at its seams."""
    meta_pre = {"type": "Metaspace", "replacement": META, "prepend_scheme": "first", "split": False}
    meta_dec = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": META}, "content": " "},
                                                {"type": "ByteFallback"}, {"type": "Fuse"}]}

    def spm_file(vocab=None, normalizer="gemma", pre="gemma", dec=None, **over):
        m = spm_model(vocab=vocab, **over)
        if normalizer == "gemma":
            n = {"type": "Replace", "pattern": {"String": " "}, "content": META}
        elif normalizer == "tinyllama":
            n = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": META},
                                                    {"type": "Replace", "pattern": {"String": " "},
                                                     "content": META}]}
        else:
            n = None
        if pre == "gemma":
            pt = {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False}
        elif pre == "none":
            pt = None
        elif isinstance(pre, dict):
            pt = pre
        else:
            pt = pre
        return tok(m, normalizer=n, pre=pt, decoder=dec if dec is not None else meta_dec)

    # the three census baselines: each must load
    case("spm", "gemma-base", spm_file(), "gemma's fold: Replace ' '->U+2581 + Split(' ', MergedWithPrevious)",
         "accept")
    case("spm", "tinyllama-base", spm_file(normalizer="tinyllama", pre="none"),
         "tinyllama's fold: Prepend U+2581 + Replace, no pre-tokenizer", "accept")
    case("spm", "mistral-base", spm_file(normalizer=None, pre=meta_pre),
         "mistral's Metaspace(prepend_scheme=first, split=false) alone", "accept")

    # vocab holes (kept under spm): the fold's chars stay, ids past them get holes
    base_v = spm_model()["vocab"]          # has <unk> <s> </s> ▁ ▁a ... t h th
    hv = {}
    nid = 0
    for k in base_v:
        hv[k] = nid
        nid += 2 if nid == 2 else 1        # holes at 4-5, 7, 9, ...
    hv["z"] = 1000
    case("spm", "holes", spm_file(vocab=hv), "spm vocab with holes between ids (decode-only ids)", "accept")
    hv2 = {k: v * 7 for k, v in hv.items()}
    case("spm", "holes-7000", spm_file(vocab=hv2), "spm vocab holes to id 7000 on a tiny source (arena 32 B/byte)",
         "accept")
    # unk_token not in vocab: loads, needing it fails
    case("spm", "unk-not-in-vocab", spm_file(unk="<nope>"),
         "unk_token not a model-vocab string and no full byte_fallback (hf fails on a char without a token)",
         "refuse")
    bf_v = {("<unk>" if i == 0 else s): i for i, s in enumerate(
        ["<unk>", META] + ["<0x%02X>" % b for b in range(256)])}
    case("spm", "unk-not-in-vocab-bf", spm_file(vocab=bf_v, unk="<nope>", byte_fallback=True),
         "unk_token not in the vocab but all 256 byte pieces there (loads; needing it fails)", "accept")
    # byte_fallback without the byte pieces; fuse_unk; ignore_merges (mistral's shape takes all three)
    case("spm", "bf-no-bytes", spm_file(normalizer=None, pre=meta_pre, byte_fallback=True),
         "byte_fallback without any <0xHH> pieces", "accept")
    case("spm", "fuse-unk", spm_file(normalizer=None, pre=meta_pre, fuse_unk=True), "fuse_unk", "accept")
    case("spm", "ignore-merges", spm_file(normalizer=None, pre=None, ignore_merges=True),
         "ignore_merges with no text model at all (no normalizer, no pre-tokenizer)", "accept")
    # Metaspace schemes (mistral's family): first is the census one; always/never also accepted
    for scheme in ("first", "always", "never"):
        case("spm", "meta-%s" % scheme,
             spm_file(normalizer=None, pre={"type": "Metaspace", "replacement": META,
                                            "prepend_scheme": scheme, "split": False}),
             "Metaspace prepend_scheme %s, split=false" % scheme, "accept")
    # split=true: refused under a Prepend prefix (the reader's rule), fine under Replace-only
    case("spm", "meta-split-true-replace", spm_file(normalizer="gemma",
             pre={"type": "Metaspace", "replacement": META, "prepend_scheme": "always", "split": True}),
         "Metaspace split=true over the Replace fold (no Prepend)", "accept")
    case("spm", "meta-split-true-prepend", spm_file(normalizer="tinyllama",
             pre={"type": "Metaspace", "replacement": META, "prepend_scheme": "always", "split": True}),
         "Metaspace split=true after a Prepend prefix (refused)", "refuse")
    sp_v = dict(spm_model()["vocab"])
    sp_v[" "] = max(sp_v.values()) + 1
    case("spm", "meta-repl-space", spm_file(vocab=sp_v, normalizer=None,
             pre={"type": "Metaspace", "replacement": " ", "prepend_scheme": "first", "split": False}),
         "Metaspace replacement ' ' (the remapped byte, in the vocab)", "accept")
    # decoder chains: the census chain and a wrong order; Strip start 65536; Replace on byte-token chars
    chain = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": META}, "content": " "},
                                              {"type": "ByteFallback"}, {"type": "Fuse"},
                                              {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}
    case("spm", "dec-chain", spm_file(normalizer=None, pre=meta_pre, dec=chain),
         "the full decoder chain Replace>ByteFallback>Fuse>Strip", "accept")
    bad = {"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}
    case("spm", "dec-wrong-order", spm_file(dec=bad), "Fuse before ByteFallback", "refuse")
    bad = {"type": "Sequence", "decoders": [{"type": "Strip", "content": " ", "start": 65536, "stop": 0}]}
    case("spm", "dec-strip-65536", spm_file(dec=bad), "decoder Strip start 65536", "refuse")
    bad = {"type": "Sequence", "decoders": [{"type": "Strip", "content": " ", "start": 1, "stop": 1}]}
    case("spm", "dec-strip-stop", spm_file(dec=bad), "decoder Strip stop 1", "refuse")
    bad = {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "<"}, "content": "("}]}
    case("spm", "dec-replace-bytetoken", spm_file(dec=bad), "decoder Replace touching byte-token chars",
         "refuse")
    # no decoder at all (hf joins with spaces)
    case("spm", "no-dec", spm_file(dec=None), "no decoder (hf joins with spaces)", "accept")
    # added tokens under spm: normalized=true rules
    a = [added(META + "hello", 100, normalized=True)]
    case("spm", "added-normalized-meta", spm_file(normalizer="tinyllama", pre="none", added=a),
         "normalized=true with U+2581 inside, under the tinyllama fold (the folded text matches it)",
         "accept")
    a = [added("hello", 100, normalized=True)]
    case("spm", "added-normalized-plain", spm_file(added=a), "normalized=true without space or U+2581", "accept")
    a = [added("hello", 100, normalized=False)]
    case("spm", "added-raw", spm_file(added=a), "normalized=false under the fold", "accept")
    # the tinyllama GAP prefix mode: added matches after the prepended U+2581
    a = [added("hello", 100, normalized=True)]
    case("spm", "pfx-gap", spm_file(normalizer="tinyllama", pre="none", added=a),
         "Prepend U+2581 + Replace: an added token matches after the prefix", "accept")


# =========================================================================================================
# tiktoken: tiktoken.c (ranks + wrappers)
# =========================================================================================================

def gen_tiktoken():
    def ranks(lines):
        return ("\n".join(lines) + "\n").encode()

    def b64(bs):
        return base64.b64encode(bs).decode()

    def line(bs, rank):
        return "%s %d" % (b64(bs), rank)

    # a minimal kimi wrapper: the pattern's 8 alternatives in order plus the constants the checker
    # wants (tiktoken.c WRAPPER_LINES); blanks elsewhere are free
    PAT = [
        r'"""[\p{Han}]+"""',            # checked as the line holding [\p{Han}]+
    ]
    CONTR = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    wrapper_lines = [
        "import tiktoken",
        "",
        "num_reserved_special_tokens = 256",
        "",
        "pat_str = [",
        '    r"""[\\p{Han}]+""",',
        '    r"""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*'
        '[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+' + CONTR + '""",',
        '    r"""[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+'
        '[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*' + CONTR + '""",',
        '    r"""\\p{N}{1,3}""",',
        '    r""" ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*""",',
        '    r"""\\s*[\\r\\n]+""",',
        '    r"""\\s+(?!\\S)""",',
        '    r"""\\s+""",',
        "]",
        "",
        "TIKTOKEN_MAX_ENCODE_CHARS = 400_000",
        "MAX_NO_WHITESPACES_CHARS = 25_000",
        "",
        "current_slice_is_space = s[0].isspace() if len(s) > 0 else False",
        "    if current_slice_len > max_consecutive_slice_len:",
        '    allowed_special="all",',
        "    disallowed_special=(),",
    ]
    wrapper = ("\n".join(wrapper_lines) + "\n").encode()

    def config(n_ranks, specials=None):
        dec = {}
        for i, (content, sid) in enumerate(specials or [("<|bos|>", 0), ("<|eos|>", 1)]):
            dec[str(n_ranks + sid)] = {"content": content, "special": True, "lstrip": False,
                                       "rstrip": False, "normalized": False, "single_word": False}
        return json.dumps({
            "tokenizer_class": "TikTokenTokenizer",
            "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer"]},
            "added_tokens_decoder": dec,
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "unk_token": "<|bos|>",
            "pad_token": "<|eos|>",
            "additional_special_tokens": ["<|bos|>", "<|eos|>"],
        }, ensure_ascii=False).encode()

    def tik_dir(ranks_bytes, conf=None):
        return {"tiktoken.model": ranks_bytes,
                "tokenizer_config.json": conf if conf is not None else config(258),
                "tokenization_kimi.py": wrapper}

    # a minimal valid model: the 256 one-byte tokens 0..255, then ab, abc (ranks 256, 257)
    ok = [line(bytes([b]), b) for b in range(256)] + [line(b"ab", 256), line(b"abc", 257)]
    case("tiktoken", "ranks-ok", tik_dir(ranks(ok), config(258)),
         "256 one-byte tokens + 2 merges (valid ranks + kimi wrapper)", "accept")
    # bad base64, non-canonical
    case("tiktoken", "bad-b64", tik_dir(ranks(["!! 0"] + ok[1:])), "a token that is not base64", "refuse")
    case("tiktoken", "noncanon-b64", tik_dir(ranks([b64(b"a") + "A 0"] + ok[1:])),
         "non-canonical base64 (extra bits)", "refuse")
    # duplicate rank / token; out-of-order (accepted)
    case("tiktoken", "dup-rank", tik_dir(ranks(ok + [line(b"cd", 257)])), "a rank that appears twice",
         "refuse")
    case("tiktoken", "dup-token", tik_dir(ranks(ok[:-1] + [line(b"b", 258)])), "a token that appears twice",
         "refuse")
    case("tiktoken", "out-of-order", tik_dir(ranks([line(b"a", 300)] + [l for l in ok[1:]])),
         "ranks out of order (ids are the ranks; the kimi wrapper wants 0..n-1 though)", "refuse")
    # rank at 2^21-1, 2^31
    case("tiktoken", "rank-2p21-1", tik_dir(ranks(ok + [line(b"z", 2**21 - 1)])), "a rank at 2^21-1",
         "refuse")
    case("tiktoken", "rank-huge", tik_dir(ranks(ok + ["AAAA " + str(2**31)])), "a rank at 2^31", "refuse")
    # a missing one-byte token
    missing = [line(bytes([b]), b) for b in range(1, 256)] + [line(b"a", 256), line(b"b", 257)]
    case("tiktoken", "missing-byte0", tik_dir(ranks(missing)), "no token for byte 0", "refuse")
    # a 65535-byte token (at the limit); 1 MiB under --big
    case("tiktoken", "token-65535", tik_dir(ranks(ok + [line(b"z" * 65535, 258)]), config(260)),
         "a 65535-byte token (at the limit)", "accept")
    # line shapes: CRLF, no final newline, empty lines, blank first, two spaces, tab, +rank, -rank,
    # no space, empty file, bare CR
    case("tiktoken", "crlf", tik_dir(("\r\n".join(ok) + "\r\n").encode()), "CRLF line endings", "accept")
    case("tiktoken", "no-final-newline", tik_dir(("\n".join(ok)).encode()), "no final newline", "accept")
    case("tiktoken", "empty-lines", tik_dir(ranks([""] + ok)),
         "an empty first line: the sniff misses (the base64 check) and the json reader refuses", "refuse")
    case("tiktoken", "blank-first", tik_dir(ranks(["", ""] + ok)), "blank lines first (the sniff misses)", "refuse")
    case("tiktoken", "two-spaces", tik_dir(ranks([b64(b"a") + "  5"] + ok[1:])), "two spaces in a line",
         "refuse")
    case("tiktoken", "tab-sep", tik_dir(ranks([b64(b"a") + "\t5"] + ok[1:])), "a tab between token and rank",
         "refuse")
    case("tiktoken", "leading-zero-rank", tik_dir(ranks([b64(b"a").ljust(len(b64(b"a")) + 1, "=")[:-1] + " 05"] + ok[1:])),
         "a rank spelled 05 (canonical check)", "refuse")
    case("tiktoken", "rank-plus", tik_dir(ranks([b64(b"a") + " +5"] + ok[1:])), "a rank spelled +5", "refuse")
    case("tiktoken", "rank-neg", tik_dir(ranks([b64(b"a") + " -5"] + ok[1:])), "a negative rank", "refuse")
    case("tiktoken", "no-space", tik_dir(ranks([b64(b"a") + "5"] + ok[1:])), "no space between token and rank",
         "refuse")
    case("tiktoken", "empty", tik_dir(b""), "an empty ranks file", "refuse")
    case("tiktoken", "cr-only", tik_dir(("\r".join(ok) + "\r").encode()), "bare-CR line endings", "refuse")
    # the wrapper, one byte off in each critical line
    lines = wrapper.split(b"\n")
    for i, ln in enumerate(lines):
        if b"[\\p{Han}]+" in ln or b"\\s+(?!\\S)" in ln or b"num_reserved_special_tokens" in ln:
            mut = (ln[:3] + bytes([ln[3] ^ 1]) + ln[4:]) if len(ln) > 4 else ln + b" "
            broken = tik_dir(ranks(ok), None)
            broken["tokenization_kimi.py"] = b"\n".join(lines[:i] + [mut] + lines[i + 1:])
            kind = "han" if b"[\\p{Han}]+" in ln else "look" if b"(?!\\S)" in ln else "num"
            case("tiktoken", "kimi-wrapper-%s-mut" % kind, broken,
                 "the kimi wrapper's %s line, one byte off" % kind, "refuse")
    # the config: an added id outside the specials' range; the wrong class; a special name that
    # is not printable ascii; overlapping names
    c = json.loads(config(258))
    c["added_tokens_decoder"]["5"] = {"content": "<|x|>", "special": True, "lstrip": False,
                                      "rstrip": False, "normalized": False, "single_word": False}
    case("tiktoken", "kimi-config-id-outside", tik_dir(ranks(ok), json.dumps(c).encode()),
         "an added id below the specials' range", "refuse")
    c = json.loads(config(258))
    c["tokenizer_class"] = "NotTikToken"
    case("tiktoken", "kimi-config-class", tik_dir(ranks(ok), json.dumps(c).encode()),
         "the wrong tokenizer_class", "refuse")
    c = json.loads(config(258))
    c["added_tokens_decoder"]["258"] = {"content": "<|\u00e9|>", "special": True, "lstrip": False,
                                        "rstrip": False, "normalized": False, "single_word": False}
    case("tiktoken", "kimi-config-nonascii", tik_dir(ranks(ok), json.dumps(c, ensure_ascii=False).encode()),
         "a special name outside printable ascii", "refuse")
    c = json.loads(config(258))
    c["added_tokens_decoder"]["259"] = {"content": "<|bos|>", "special": True, "lstrip": False,
                                        "rstrip": False, "normalized": False, "single_word": False}
    case("tiktoken", "kimi-config-overlap", tik_dir(ranks(ok), json.dumps(c).encode()),
         "two specials with the same name", "refuse")


# ---- the big flag -------------------------------------------------------------------------------------------

BIG = False


def main() -> int:
    global BIG
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["files", "texts"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--big", action="store_true")
    a = ap.parse_args()
    BIG = a.big

    if a.what == "texts":
        import texts
        texts.write(a.out)
        return 0

    gen_json()
    gen_vocab()
    gen_merges()
    gen_added()
    gen_normalizer()
    gen_pretok()
    gen_template()
    gen_truncpad()
    gen_unigram()
    gen_wordpiece()
    gen_spm()
    gen_tiktoken()

    out = a.out
    n = 0
    lines = ["# tests/fuzz/hostile: hand-reasoned tokenizer files (docs/fuzz.md). "
             "path <TAB> sha256 <TAB> description <TAB> expects (accept | refuse | accept:or-refuses)"]
    for cat, name, data, desc, exp in CASES:
        if a.kind and cat != a.kind:
            continue
        d = os.path.join(out, cat)
        os.makedirs(d, exist_ok=True)
        if isinstance(data, dict):          # a model directory: tiktoken + its companions
            p = os.path.join(d, name)
            os.makedirs(p, exist_ok=True)
            h = hashlib.sha256()
            for leaf in sorted(data):
                with open(os.path.join(p, leaf), "wb") as f:
                    f.write(data[leaf])
                h.update(leaf.encode() + b"\0" + data[leaf])
            data = None
            sha = h.hexdigest()
        else:
            p = os.path.join(d, name + (".json" if cat not in ("tiktoken",) else ".model"))
            with open(p, "wb") as f:
                f.write(data)
            sha = hashlib.sha256(data).hexdigest()
        lines.append("%s\t%s\t%s\t%s" % (p, sha, desc, exp))
        n += 1
    with open(os.path.join(out, "cases.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("hostile: %d files in %d categories -> %s" % (n, len({c for c, *_ in CASES}), out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
