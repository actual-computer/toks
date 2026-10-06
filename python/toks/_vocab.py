"""toks._vocab: hf tokenizers 0.23.2's token strings for a toks.Tokenizer (token_to_id, id_to_token,
get_vocab, get_vocab_size, Encoding.tokens), read from its tokenizer.json on first use, hf's AddedToken, and the
tiktoken view's pieces that are not the library's (which tokens are special, tiktoken's error, its split).

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


def _templates(pp):
    """every single-sequence template hf could read from the post-processor: (prefix tokens, suffix tokens, prefix
    ids, suffix ids) around the text. hf takes the first variant of its untagged enum (Roberta, Bert, ByteLevel,
    Template, Sequence) that accepts the object, which need not be the one "type" names (a repeated field refuses a
    variant: hfshape/accept_bytelevel_template.json is a template); so every reading the fields allow is a candidate,
    and the ids toks_template wrote pick the one that holds."""
    if not isinstance(pp, dict):
        return [([], [], [], [])]
    out = [([], [], [], [])]                         # ByteLevel, or no template
    try:
        out.append(([pp["cls"][0]], [pp["sep"][0]], [pp["cls"][1]], [pp["sep"][1]]))    # Roberta, Bert
    except (KeyError, IndexError, TypeError):
        pass
    try:                                             # Template
        pre, suf, pi, si, seen = [], [], [], [], False
        for piece in pp["single"]:
            if "Sequence" in piece:
                seen = True
                continue
            st = pp["special_tokens"][piece["SpecialToken"]["id"]]
            (suf if seen else pre).extend(st["tokens"])
            (si if seen else pi).extend(st["ids"])
        out.append((pre, suf, pi, si))
    except (KeyError, IndexError, TypeError):
        pass
    if isinstance(pp.get("processors"), list):     # Sequence: each processor wraps what the ones before it made
        acc = [([], [], [], [])]
        for q in pp["processors"]:
            acc = [(r[0] + a[0], a[1] + r[1], r[2] + a[2], a[3] + r[3]) for a in acc for r in _templates(q)]
        out += acc
    return out


def load(kind: int, source, sha256: bytes):
    """-> (token_to_id, id_to_token, model_vocab, model_size, size_with_added, extra) for the loaded source; extra
    holds what Encoding.tokens reads: the added tokens by id (hf's added_tokens_map_r), the special contents, the
    model's own string of an id an added token also holds, the template's strings, the pad token, Unigram's unk."""
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
    special: set[str] = set()                        # special_tokens_set: a content listed special anywhere
    normalizer = obj.get("normalizer")
    nxt = n_model
    for a in obj.get("added_tokens") or []:
        content = a["content"]
        if not content:
            continue
        if a["special"]:
            special.add(content)
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
    model_of_added = {i: id2tok[i] for i in added_r if i in id2tok}
    for i, key in added_r.items():
        id2tok[i] = normed.get(i, key[0])
    n_total = n_model + sum(1 for t in added if t not in model_vocab)
    pad = obj.get("padding")
    differ = [i for i, s in model_of_added.items() if s != id2tok[i]]
    extra = {"added_r": added_r, "special": special, "model_of_added": model_of_added,
             "model_writes": _model_writes(model, differ, model_of_added) if differ else frozenset(),
             "templates": _templates(obj.get("post_processor")), "pad_token": pad.get("pad_token") if pad else None,
             "unk": model.get("unk_id") if isinstance(vocab, list) else None,
             "normalizer": bool(normalizer)}
    return tok2id, id2tok, model_vocab, n_model, n_total, extra


def _model_writes(model, ids, model_of):
    """the ids among `ids` (each held by an added token under another string than the model's) that the model can
    write itself, so that a match of the added token is not their only source. A BPE model writes its unk when byte
    fallback cannot cover a character, a one-character symbol, a byte-fallback token and a merge's result (any piece
    under ignore_merges); the other models are taken to write every id. A model without "type" is read by its shape,
    as hf does (merges: BPE)."""
    if model.get("type", "BPE" if "merges" in model else None) != "BPE" or model.get("ignore_merges"):
        return frozenset(ids)
    vocab = model["vocab"]
    fb = bool(model.get("byte_fallback"))
    unk = vocab.get(model["unk_token"]) if model.get("unk_token") is not None else None
    unk_writes = unk is not None and not (fb and all(f"<0x{b:02X}>" in vocab for b in range(256)))
    pre = model.get("continuing_subword_prefix") or ""
    suf = model.get("end_of_word_suffix") or ""
    out, want = set(), {}
    for i in ids:
        s = model_of[i]
        core = s[len(pre):] if pre and s.startswith(pre) else s
        core = core[:len(core) - len(suf)] if suf and core.endswith(suf) else core
        if (i == unk and unk_writes) or len(core) == 1 or (fb and len(s) == 6 and s.startswith("<0x") and s[5] == ">"):
            out.add(i)
        else:
            want[s] = i
    if want:
        for m in model.get("merges") or []:
            a, b = m if isinstance(m, list) else m.split(" ", 1)
            r = a + b[len(pre):]                    # hf's merge: a + b without the continuing-subword prefix
            if r in want:
                out.add(want[r])
    return frozenset(out)


# ---- hf's AddedToken and Encoding.tokens --------------------------------------------------------------------------

class AddedToken:
    """hf tokenizers' AddedToken as Tokenizer.get_added_tokens_decoder() returns it: the content and its options.
    Equal when every field is; str() is the content."""
    __slots__ = ("content", "single_word", "lstrip", "rstrip", "normalized", "special")

    def __init__(self, content="", single_word=False, lstrip=False, rstrip=False, normalized=None, special=False):
        self.content = content
        self.single_word = bool(single_word)
        self.lstrip = bool(lstrip)
        self.rstrip = bool(rstrip)
        self.normalized = (not special) if normalized is None else bool(normalized)   # hf's default
        self.special = bool(special)

    def _key(self):
        return (self.content, self.single_word, self.lstrip, self.rstrip, self.normalized, self.special)

    def __eq__(self, other):
        return self._key() == other._key() if isinstance(other, AddedToken) else NotImplemented

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        return self.content

    def __repr__(self):
        return (f'AddedToken("{self.content}", rstrip={self.rstrip}, lstrip={self.lstrip}, '
                f'single_word={self.single_word}, normalized={self.normalized}, special={self.special})')

    def __getstate__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, d):
        for k in self.__slots__:
            setattr(self, k, d[k])


def added_tokens_decoder(entries):
    """{id: AddedToken} from toks_added's (id, content, flags) in id order (hf's get_added_tokens_decoder)."""
    return {i: AddedToken(c, single_word=f & 32, lstrip=f & 8, rstrip=f & 16, normalized=bool(f & 64), special=f & 2)
            for i, c, f in entries}


_MODES = ("all", "nonspecial", "none")


def tokens(tok, enc):
    """Encoding.tokens, hf's: the template's strings as its post-processor names them, the pad token, and for the
    text's ids what hf's model and added vocabulary write: the model's string of a piece; an added token's matched
    text (its content, normalized when it is normalized=true, with the whitespace an lstrip / rstrip token took, read
    from toks_pieces: the token is one piece spanning exactly that text). Raises toks.Error where the ids and the
    pieces do not determine hf's string: a Unigram unknown piece (hf writes the normalized text it covers), an id the
    model and an added token both write under different strings in a text holding the token, lstrip / rstrip matches
    the pieces do not single out, a post-processor two of whose readings give the template's ids under other
    strings."""
    v = tok._vocab()
    id2tok, extra = v[1], v[5]
    ids = enc.ids
    n_pre, n_suf, n_pad, pad_left, flags = enc._layout
    body = ids[n_pad:] if pad_left else ids[:len(ids) - n_pad]
    pre, suf = [], []
    if n_pre or n_suf:
        want = (body[:n_pre], body[len(body) - n_suf:])
        hits = {(tuple(c[0]), tuple(c[1])) for c in extra["templates"] if (c[2], c[3]) == want}
        if len(hits) != 1:
            raise _toks._error(TOKS_E_UNSUPPORTED, "Encoding.tokens: no reading of the file's post_processor, or two "
                               "with other strings, gives the template's ids (its strings are unknown)")
        pre, suf = map(list, hits.pop())
    mid = _text_tokens(tok, id2tok, extra, body[n_pre:len(body) - n_suf], enc._text, flags)
    pads = [extra["pad_token"]] * n_pad
    return pads + pre + mid + suf if pad_left else pre + mid + suf + pads


def _text_tokens(tok, id2tok, extra, ids, text, flags):
    mode = flags & 3
    added_r, special, unk, model_of = extra["added_r"], extra["special"], extra["unk"], extra["model_of_added"]
    tb = text if isinstance(text, bytes) else text.encode("utf-8", "surrogatepass")
    out, strip = [], []
    for i in ids:
        key = added_r.get(i)
        if i == unk:
            raise _toks._error(TOKS_E_UNSUPPORTED, f"Encoding.tokens: id {i} is the Unigram model's unknown piece; hf "
                               "writes the normalized text it covers, which toks does not keep")
        if key is not None and (mode == 0 or (mode == 1 and key[0] not in special)):     # an added token matched
            if i in extra["model_writes"]:
                # the model writes this id too, under another string (its unk, or a piece an added token was
                # given): the model's piece unless the text holds the token
                if key[0].encode("utf-8") in tb:
                    raise _toks._error(TOKS_E_UNSUPPORTED, f"Encoding.tokens: id {i} is the model's {model_of[i]!r} "
                                       f"and the added token {key[0]!r}, which the text holds")
                out.append(model_of[i])
                continue
            if key[2] or key[3]:
                strip.append((len(out), key))
                out.append(None)
                continue
            s = id2tok.get(i)
        else:                                                                          # a piece of the model
            s = model_of.get(i) if key is not None else id2tok.get(i)
        if isinstance(s, BaseException):
            raise s
        if s is None:
            raise _toks._error(TOKS_E_UNSUPPORTED, f"Encoding.tokens: id {i} has no string in the file")
        out.append(s)
    if strip:
        _strip_tokens(tok, extra, out, strip, text, tb, flags)
    return out


def _strip_tokens(tok, extra, out, strip, text, tb, flags):
    """the text an lstrip / rstrip token matched, one piece of toks_pieces: the k-th piece that is the token's
    content within whitespace (a whitespace content: an all-whitespace piece holding it) is its k-th match. In a
    mode that recognizes the token every occurrence of the content is in a match, so the pieces that qualify are
    exactly its matches; any other count is refused, never shifted."""
    if extra["normalizer"] and any(key[4] for _, key in strip):
        raise _toks._error(TOKS_E_UNSUPPORTED, "Encoding.tokens: a normalized=true lstrip / rstrip token matches the "
                           "normalized text, which toks does not keep")
    ends = tok.pieces(text, added_tokens=_MODES[flags & 3], continuation=bool(flags & 8))
    if ends and ends[-1] != len(tb):
        raise _toks._error(TOKS_E_UNSUPPORTED, "Encoding.tokens: the pieces index the normalized text")
    ws = "".join(_WS)
    found = {key[0]: [] for _, key in strip}
    core = {c: c.strip(ws) for c in found}                   # "" for a whitespace content
    a = 0
    for b in ends:
        try:
            s = tb[a:b].decode("utf-8")
        except UnicodeDecodeError:
            s = ""
        a = b
        sc = s.strip(ws)
        held = [c for c in found if core[c] == sc and c in s]
        if len(held) > 1:
            raise _toks._error(TOKS_E_UNSUPPORTED, f"Encoding.tokens: the piece {s!r} could be the match of any of "
                               f"{held!r}")
        if held:
            found[held[0]].append(s)
    want = {}
    for _, key in strip:
        want[key[0]] = want.get(key[0], 0) + 1
    for c, n in want.items():
        if len(found[c]) != n:
            raise _toks._error(TOKS_E_UNSUPPORTED, f"Encoding.tokens: {len(found[c])} pieces hold {c!r} within "
                               f"whitespace for its {n} matches")
    taken = {c: 0 for c in found}
    for idx, key in strip:
        c = key[0]
        out[idx] = found[c][taken[c]]
        taken[c] += 1


# ---- the tiktoken view ---------------------------------------------------------------------------------------------

TOKS_ADDED_NONSPECIAL = 1
TOKS_ADDED_NONE = 2


def source_is_json(kind: int, source) -> bool:
    """toks_load's rule: a model directory is its tokenizer.json when it has one; a file is a tokenizer.json when it
    holds a json object (a tiktoken model is base64 lines); memory sources are tokenizer configurations"""
    if kind != 0:
        return True
    path = os.fspath(source)
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, "tokenizer.json" if isinstance(path, str) else b"tokenizer.json"))
    with open(path, "rb") as f:
        return f.read(4096).lstrip(b" \t\r\n")[:1] == b"{"


def tiktoken_view(tok, kind: int, source, entries):
    """(specials {id: content}, {content: id}, the mode that recognizes none of them). A tokenizer.json's special
    tokens are hf's special_tokens_set, every content the file lists special (read from the file: two contents can
    share an id, and toks_added keeps the one listed last), each with the id hf gives it; NONSPECIAL is then ordinary
    (encode_special_tokens leaves exactly those unrecognized). A tiktoken model's are every special it defines
    (toks_added's entries (id, content, toks_added flags, toks_id_flags): tiktoken's special_tokens, recognized only in
    ALL, so NONE is ordinary)"""
    if source_is_json(kind, source):
        v = tok._vocab()
        bystr = {c: v[0][c] for c in sorted(v[5]["special"]) if c in v[0]}
        return {i: c for c, i in bystr.items()}, bystr, TOKS_ADDED_NONSPECIAL
    byid = {i: c for i, c, f, idf in entries}
    return byid, {c: i for i, c in byid.items()}, TOKS_ADDED_NONE


def disallowed(token: str) -> ValueError:
    """tiktoken 0.14.0's raise_disallowed_special_token, word for word"""
    return ValueError(
        f"Encountered text corresponding to disallowed special token {token!r}.\n"
        "If you want this text to be encoded as a special token, "
        f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
        f"If you want this text to be encoded as normal text, disable the check for this token "
        f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
        "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
    )


def encode_subset(tok, text, allowed):
    """tiktoken's encode with some specials allowed and others present as text (its check off for them): the text
    between two allowed specials is encoded alone, ordinary (CoreBPE::encode_native); allowed is {content: id}"""
    import re
    if isinstance(text, str):
        ids, bar = allowed, "|"
    else:
        text = bytes(text)
        ids, bar = {c.encode("utf-8"): i for c, i in allowed.items()}, b"|"
    pat = re.compile(bar.join(re.escape(k) for k in sorted(ids, key=len, reverse=True)))   # leftmost, longest first
    out, pos = [], 0
    for m in pat.finditer(text):
        if m.start() > pos:
            out += tok.encode_ordinary(text[pos:m.start()])
        out.append(ids[m.group()])
        pos = m.end()
    if pos < len(text):
        out += tok.encode_ordinary(text[pos:])
    return out
