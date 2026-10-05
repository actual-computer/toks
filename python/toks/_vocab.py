"""toks._vocab: hf tokenizers 0.23.2's token strings for a toks.Tokenizer (token_to_id, id_to_token,
get_vocab, get_vocab_size), read from its tokenizer.json on first use.

The core (include/toks.h) keeps the bytes every id decodes to, not the vocabulary's strings, and the
strings are not a function of those bytes (an added token "  " and the byte-level "ĠĠ" decode alike). So
this module reads the same source again, checks it is byte-identical to what was loaded (its sha256 is the
context's toks_info.source_sha256), and applies hf's rules:

- model vocabulary: BPE / WordPiece / WordLevel {token: id}; Unigram [[piece, score], ...] (id = index,
  a later duplicate piece wins token_to_id). get_vocab_size(False) = the model's entry count.
- added tokens (AddedVocabulary::add_tokens over the file's list in order): empty contents are dropped;
  an entry equal in every field to the one already mapped for its content is skipped; otherwise its id is
  the id its content already has (an earlier added token, else the model vocabulary), else the next free
  one counting up from the model's size. The json "id" is never read.
- token_to_id: the added map, then the model. id_to_token: the added token (its content, or its
  normalized content when normalized is true and the normalizer changes it: hf's normalized_cache), then
  the model. get_vocab(True) = model vocab updated with the added map; get_vocab_size(True) = model size +
  added contents not in the model vocabulary.

Normalizers evaluated for normalized_cache: NFC, NFD, NFKC, NFKD, Lowercase, Strip, StripAccents, Prepend,
Replace (string pattern), BertNormalizer and Sequences of them. Any other (Precompiled, Replace with a
regex, Nmt, ByteLevel) leaves those ids' strings unknown: id_to_token raises toks.Error for them instead
of guessing (census, 2026-10-04: 2 of 230 tokenizers, one id each).
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata

from . import _toks

TOKS_E_FORMAT = -2
TOKS_E_UNSUPPORTED = -3

# char::is_whitespace (Unicode White_Space): Strip and BertNormalizer use it, not str.isspace()
_WS = frozenset(map(chr, [*range(9, 14), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
                          0x202F, 0x205F, 0x3000]))


class _Unknown(Exception):
    pass


def _bert(n: dict, s: str) -> str:
    if n.get("clean_text", True):
        s = "".join(" " if c in "\t\n\r" or c in _WS else c for c in s
                    if not (c in ("\0", "\ufffd") or (c not in "\t\n\r" and unicodedata.category(c)[0] == "C")))
    if n.get("handle_chinese_chars", True):
        def cjk(c):
            o = ord(c)
            return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2B73F
                    or 0x2B740 <= o <= 0x2B81F or 0x2B920 <= o <= 0x2CEAF or 0xF900 <= o <= 0xFAFF
                    or 0x2F800 <= o <= 0x2FA1F)
        s = "".join(f" {c} " if cjk(c) else c for c in s)
    lower = n.get("lowercase", True)
    strip = n.get("strip_accents")
    if strip if strip is not None else lower:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.lower() if lower else s


def _normalize(n: dict, s: str) -> str:
    t = n.get("type")
    if t in ("NFC", "NFD", "NFKC", "NFKD"):
        return unicodedata.normalize(t, s)
    if t == "Lowercase":
        return s.lower()
    if t == "Strip":
        ws = "".join(_WS)
        if n.get("strip_left"):
            s = s.lstrip(ws)
        return s.rstrip(ws) if n.get("strip_right") else s
    if t == "StripAccents":
        return "".join(c for c in s if unicodedata.category(c)[0] != "M")
    if t == "Prepend":
        return n["prepend"] + s if s else s
    if t == "Replace" and "String" in n.get("pattern", {}):
        return s.replace(n["pattern"]["String"], n["content"])
    if t == "BertNormalizer":
        return _bert(n, s)
    if t == "Sequence":
        for m in n.get("normalizers", []):
            s = _normalize(m, s)
        return s
    raise _Unknown(t if t != "Replace" else "Replace (regex)")


def _read(kind: int, source) -> bytes:
    if kind == 1:
        return bytes(source)
    if kind == 2:
        return source.encode("utf-8")
    path = os.fspath(source)
    if os.path.isdir(path):                          # toks_load's rule: <dir>/tokenizer.json
        path = os.path.join(path, "tokenizer.json" if isinstance(path, str) else b"tokenizer.json")
    with open(path, "rb") as f:
        return f.read()


def load(kind: int, source, sha256: bytes):
    """-> (token_to_id, id_to_token, model_vocab, model_size, size_with_added) for the loaded source."""
    data = _read(kind, source)
    if hashlib.sha256(data).digest() != sha256:
        raise _toks._error(TOKS_E_FORMAT, f"{source!r} changed since it was loaded (source sha256 differs)")
    try:
        obj = json.loads(data)
        model = obj["model"]
        vocab = model["vocab"]
    except (ValueError, KeyError, TypeError):
        raise _toks._error(TOKS_E_UNSUPPORTED, "token strings need a tokenizer.json source "
                           "(the model's vocab and added_tokens)") from None
    if isinstance(vocab, list):                      # Unigram
        model_vocab = {}
        id2tok = {}
        for i, entry in enumerate(vocab):
            model_vocab[entry[0]] = i
            id2tok[i] = entry[0]
        n_model = len(vocab)
    else:                                            # BPE, WordPiece, WordLevel
        model_vocab = dict(vocab)
        id2tok = {i: t for t, i in vocab.items()}
        n_model = len(vocab)

    added: dict[str, int] = {}                       # added_tokens_map
    added_r: dict[int, tuple] = {}                   # added_tokens_map_r
    normed: dict[int, object] = {}                   # normalized_cache (or the reason it is unknown)
    normalizer = obj.get("normalizer")
    nxt = n_model
    for a in obj.get("added_tokens") or []:
        content = a["content"]
        if not content:
            continue
        key = (content, a["single_word"], a["lstrip"], a["rstrip"], a["normalized"], a["special"])
        if content in added and added_r.get(added[content]) == key:
            continue
        new = added.get(content, model_vocab.get(content))
        if new is None:
            new = nxt
            nxt += 1
        if a["normalized"] and normalizer:
            try:
                n = _normalize(normalizer, content)
                if n != content:
                    normed[new] = n
            except _Unknown as e:
                normed[new] = _toks._error(TOKS_E_UNSUPPORTED, f"id_to_token({new}): the token {content!r} is "
                                           f"normalized=true and toks does not evaluate the {e} normalizer")
        added[content] = new
        added_r[new] = key

    tok2id = dict(model_vocab)
    tok2id.update(added)
    for i, key in added_r.items():
        id2tok[i] = normed.get(i, key[0])
    n_total = n_model + sum(1 for t in added if t not in model_vocab)
    return tok2id, id2tok, model_vocab, n_model, n_total
