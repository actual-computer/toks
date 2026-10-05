"""The reference oracle: hf `tokenizers` 0.23.2 (the semantic reference, SPEC §0.6) + tiktoken.

Verified semantics (docs/parity-notes.md is the record; every point below was probed
against tokenizers 0.23.2 and is pinned by tests/parity/test_oracle.py):

Mode mapping (§3.2) — two orthogonal axes:
  * matching mode: ALL (default) / NONSPECIAL (`tokenizer.encode_special_tokens = True`)
    / NONE (tokenizer.json with `added_tokens: []`, every other table byte-identical).
  * post-processing: the `add_special_tokens` argument of encode() controls ONLY the
    post-processor (Template / ByteLevel add_prefix_space). It never changes matching.

NONSPECIAL (settled with a synthetic tokenizer, gpt2 alone cannot distinguish):
  `encode_special_tokens = True` hides ONLY tokens whose `special` flag is true. Non-
  special added tokens still match. A special match that is skipped is NOT rescanned
  for other (non-special) tokens overlapping its span — the trie's leftmost-longest
  iteration consumes the whole span ("day" special + "ay" non-special: text "day" in
  NONSPECIAL encodes as ordinary text, not as "ay"+...).

NONE:
  `added_tokens: []` keeps the model vocab/merges byte-identical (proven:
  get_vocab_size(False) and the vocab dict are unchanged; gpt2's <|endoftext|> sits
  inside the model vocab as a key and still the BPE pipeline never emits it — no
  special-casing needed; a WordLevel tokenizer would look it up by piece).

Pieces (§3.5): end offsets in the NORMALIZED byte stream, computed as
  1. added-token split of the original text with hf find_matches semantics
     (leftmost-longest; special skipped in NONSPECIAL and its span not rescanned;
     single_word / lstrip / rstrip honored; normalized:true tokens re-matched after
     normalization of the remaining segment),
  2. each ordinary segment: normalize -> pre_tokenizer.pre_tokenize_str (char
     offsets) -> char-to-byte offsets, all segments concatenated. An added-token
     match is ONE piece spanning its literal's bytes (hf does not normalize
     added-token content — the normalized:true form is only used for *matching*).

Decode: hf decode(ids, skip_special_tokens). Ids with no vocab entry are SKIPPED
(verified up to 10^7 and beyond: returns '' rather than raising; negative ids raise
OverflowError at the pybind boundary — toks maps that to TOKS_E_ID anyway per §3.4:
"id beyond the table"). skip_special_tokens drops ids whose added-token entry has
special=true (and only those).

Stream: Python's codecs incremental 'replace' decoder is NOT the reference (CPython
defers ED A0..BF for surrogatepass). The reference is: after each push, output =
lossy_decode(all bytes so far) minus the held valid-prefix tail (e+U+0301-style
completions must not be emitted early); the flush emits the rest. `held()` is ported
from e's nm_stream.py.

tiktoken: ALL = encode(allowed_special='all'); NONSPECIAL/NONE = encode_ordinary
(tiktoken has no non-special added tokens). Decode = decode_bytes; there is no
skip_special in tiktoken — the toks side defines it from the pattern file.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# download / cache (huggingface_hub -> ~/.cache/toks/tokenizers/)


def _cache_dir() -> str:
    return os.path.expanduser(os.environ.get("TOKS_TOKENIZER_CACHE", "~/.cache/toks/tokenizers"))


def download(repo: str) -> str:
    """Idempotently download <repo>/tokenizer.json; returns the local path."""
    path = os.path.join(_cache_dir(), repo.replace("/", "_"))
    if os.path.isfile(path):
        return path
    import huggingface_hub

    got = huggingface_hub.hf_hub_download(repo, "tokenizer.json")
    with open(got, "rb") as f:
        data = f.read()
    os.makedirs(_cache_dir(), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


NAMES = {
    "gpt2": "openai-community/gpt2",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
}


def load_cached(name: str) -> str:
    return download(NAMES.get(name, name))


# ---------------------------------------------------------------------------
# added-token matching — a port of hf added_vocabulary.rs find_matches


@dataclass
class AddedToken:
    id: int
    content: str
    special: bool
    normalized: bool = False
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False


# Rust `regex` \s (hf's added_vocabulary uses regex::Regex, not is_whitespace): exactly these 25.
# Python str.isspace() agrees on them but also matches U+001C..001F — must NOT be used.
RUST_S = frozenset(
    chr(c)
    for c in list(range(9, 14)) + [32, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B)) + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)
# hf's single_word test (ends_with_word / starts_with_word) is the rust regex crate's unicode \w:
# [\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}], 144667 code points as probed through hf 0.23.2
# (tests/data/breadth/probe_hf.py word). src/gen/rx_word.c holds the probed ranges; python's isalnum()
# misses 17591 of them, so the oracle reads that table (lazily) instead of approximating it.
_WORD_RANGES: Optional[List[Tuple[int, int]]] = None


def _word_ranges() -> List[Tuple[int, int]]:
    global _WORD_RANGES
    if _WORD_RANGES is None:
        import re

        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "gen", "rx_word.c")
        with open(src, encoding="utf-8") as f:
            text = f.read()
        _WORD_RANGES = [(int(a, 16), int(b, 16)) for a, b in re.findall(r"\{0x([0-9A-F]+)u, 0x([0-9A-F]+)u\}", text)]
        assert len(_WORD_RANGES) > 700, "src/gen/rx_word.c: the \\w table did not parse"
    return _WORD_RANGES


def _is_word(ch: str) -> bool:
    import bisect

    r = _word_ranges()
    cp = ord(ch)
    i = bisect.bisect_right(r, (cp, 0x10FFFF)) - 1
    return i >= 0 and r[i][0] <= cp <= r[i][1]


def _space_leftmost_at_end(s: str) -> int:
    i = len(s)
    while i > 0 and s[i - 1] in RUST_S:
        i -= 1
    return i


def _space_rightmost_at_start(s: str) -> int:
    i = 0
    while i < len(s) and s[i] in RUST_S:
        i += 1
    return i


class Trie:
    """Leftmost-longest trie over added-token patterns (hf uses DoubleArrayAhoCorasick
    with MatchKind::LeftmostLongest). At each position the longest pattern wins;
    ties (same content) impossible (id map is a dict on content)."""

    def __init__(self, tokens: Sequence[AddedToken]):
        self.root: dict = {}
        self.by_content = {}
        for t in tokens:
            node = self.root
            for ch in t.content:
                node = node.setdefault(ch, {})
            node["\0"] = t  # terminal
            self.by_content[t.content] = t

    def leftmost_longest(self, text: str, start: int) -> Optional[Tuple[AddedToken, int, int]]:
        """Longest match at exactly `start`, or None."""
        node = self.root
        best = None
        for i in range(start, len(text)):
            node = node.get(text[i])
            if node is None:
                break
            t = node.get("\0")
            if t is not None:
                best = (t, start, i + 1)
        return best


def find_matches(
    text: str, tokens: Sequence[AddedToken], hide_special: bool
) -> List[Tuple[Optional[int], int, int]]:
    """hf find_matches: (id|None, start, end) splits of `text` (BYTE offsets are
    used by hf; we use CHARACTER offsets here and convert later — equivalent since
    we operate on str). A hidden (special) or failed single_word match consumes its span without a
    rescan. The search resumes at the RAW match end, so after an rstrip the next match may start
    inside the run the token swallowed: no gap, and the two splits overlap (hf, kept exactly). An
    lstrip clamped to the previous split's end can leave an empty token split, which hf drops; start >
    end makes hf panic ("AddedVocabulary bad split"; toks refuses the token sets that reach it)."""
    splits: List[Tuple[Optional[int], int, int]] = []
    if not text:
        return splits
    trie = Trie(tokens)
    special_contents = {t.content for t in tokens if t.special}
    start_offset = 0
    i = 0
    n = len(text)
    while i < n:
        m = trie.leftmost_longest(text, i)
        if m is None:
            i += 1
            continue
        t, s, e = m
        raw_e = e
        if hide_special and t.content in special_contents:
            # skipped: span consumed, NOT rescanned (verified probe13: "day" special +
            # "ay" non-special -> NONSPECIAL('day') is ordinary text)
            i = raw_e
            continue
        if t.single_word:
            # hf: ends_with_word / starts_with_word are regex \w checks on the raw match (Unicode word
            # chars), NOT whitespace checks; a discarded single_word match consumes its span in the
            # leftmost-longest iteration exactly like a skipped special (probe: special "day"
            # single_word + "1ay" added: "day1ay" -> day discarded, "1ay" matches AFTER day's span).
            start_ok = s == 0 or not _is_word(text[s - 1])
            end_ok = e >= n or not _is_word(text[e])
            if not (start_ok and end_ok):
                i = raw_e
                continue
        if t.lstrip:
            s = max(_space_leftmost_at_end(text[:s]), start_offset)
        if t.rstrip:
            e += _space_rightmost_at_start(text[e:])
        if s > e:
            raise ValueError("hf panics: AddedVocabulary bad split (an lstrip token inside an rstrip run)")
        if s < e:                                   # an empty token split is filtered out by hf
            if start_offset < s:
                splits.append((None, start_offset, s))
            splits.append((t.id, s, e))
            start_offset = e
        i = raw_e
    if start_offset < n:
        splits.append((None, start_offset, n))
    return splits


# ---------------------------------------------------------------------------
# stream reference (ported from e's nm_stream.py, adjusted: see notes)


def held(bs: bytes) -> int:
    """Bytes a correct stream still holds: a valid proper prefix of a utf-8 sequence
    (lead byte + continuation bytes so far), 0 when there is none."""
    for t in (1, 2, 3):
        if len(bs) < t:
            break
        s = bs[-t:]
        c = s[0]
        if c < 0xC2 or c > 0xF4:
            continue
        need = 2 if c < 0xE0 else 3 if c < 0xF0 else 4
        if t >= need:
            return 0
        lo, hi = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F), 0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}.get(
            c, (0x80, 0xBF)
        )
        ok = all(0x80 <= x <= 0xBF for x in s[1:]) and (t < 2 or lo <= s[1] <= hi)
        return t if ok else 0
    return 0


def stream_steps(pushes: Sequence[bytes], final: bool = True) -> List[bytes]:
    """Per-push outputs of the reference stream decoder (+ the flush as the last element
    when final=True). After push k: emitted == lossy(all bytes so far) minus held tail."""
    allb = b""
    done = b""
    out: List[bytes] = []
    for p in pushes:
        allb += p
        e = allb[: len(allb) - held(allb)].decode("utf-8", "replace").encode("utf-8")
        out.append(e[len(done) :])
        done = e
    if final:
        f = allb.decode("utf-8", "replace").encode("utf-8")
        out.append(f[len(done) :])
    return out


def stream_reference(pushes: Sequence[bytes]) -> List[str]:
    """Reference stream decoder output per push (decoded strings)."""
    return [b.decode("utf-8", "replace") for b in stream_steps(pushes)]


# ---------------------------------------------------------------------------
# the oracle


class ModeError(ValueError):
    pass


MODES = ("ALL", "NONSPECIAL", "NONE")


@dataclass
class TokSpec:
    path: str
    obj: dict
    json_text: str
    added: List[AddedToken]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _spec_from_obj(obj: dict, path: str) -> TokSpec:
    added = [
        AddedToken(
            id=a["id"],
            content=a["content"],
            special=a.get("special", True),
            normalized=a.get("normalized", not a.get("special", True)),
            single_word=a.get("single_word", False),
            lstrip=a.get("lstrip", False),
            rstrip=a.get("rstrip", False),
        )
        for a in obj.get("added_tokens", [])
        if a.get("content")                         # hf ignores an empty content
    ]
    # hf add_tokens: a repeated content keeps its first place and id and takes the last entry's flags
    # (added_tokens_map_r[id] = token), while specialness sticks (special_tokens_set only grows)
    first: dict = {}
    for a in added:
        if a.content in first:
            p = first[a.content]
            first[a.content] = AddedToken(id=p.id, content=a.content, special=p.special or a.special,
                                          normalized=a.normalized, single_word=a.single_word,
                                          lstrip=a.lstrip, rstrip=a.rstrip)
        else:
            first[a.content] = a
    added = list(first.values())
    return TokSpec(path=path, obj=obj, json_text=json.dumps(obj), added=added)


def _read(path: str) -> dict:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


class Tok:
    def __init__(self, tokenizer, spec: TokSpec):
        import tokenizers  # noqa: F401  (assert availability early)

        self.t = tokenizer
        self.spec = spec
        self._nonspecial = None
        self._none = None
        # sanity: the added ids hf actually assigned (it re-ids on collision; verified
        # it preserves the json ids when they don't collide, and re-ids otherwise —
        # our spec keeps the json view; tests assert consistency for real tokenizers)
        self._refresh_ids()

    def _refresh_ids(self) -> None:
        v = self.t.get_vocab()
        self._vocab = v
        self._vocab_size = len(v)
        # map content -> actual id from the tokenizer's own vocab when present
        self._content_id = {}
        for a in self.spec.added:
            self._content_id[a.content] = v.get(a.content, a.id)

    # -- views ---------------------------------------------------------------
    def view(self, mode: str):
        if mode == "ALL":
            return self.t
        if mode == "NONSPECIAL":
            if self._nonspecial is None:
                # a fresh load of the file, not copy.deepcopy(self.t): a deepcopy goes through hf's own
                # serializer, which writes one entry per content and so drops the specialness a repeated
                # content's first entry gave it (special_tokens_set only grows at load; hf 0.23.2 then
                # matches "<b>" special-then-nonspecial in NONSPECIAL after a round trip, but not before)
                import tokenizers

                t = tokenizers.Tokenizer.from_str(self.spec.json_text)
                t.encode_special_tokens = True
                self._nonspecial = t
            return self._nonspecial
        if mode == "NONE":
            if self._none is None:
                import tokenizers

                obj = copy.deepcopy(self.spec.obj)
                obj["added_tokens"] = []
                t = tokenizers.Tokenizer.from_str(json.dumps(obj))
                t.encode_special_tokens = True  # no added tokens: irrelevant
                self._none = t
            return self._none
        raise ModeError(mode)

    # -- encode ---------------------------------------------------------------
    def encode(self, text: str, mode: str = "ALL", add_special_tokens: bool = False) -> List[int]:
        return self.view(mode).encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(
        self, texts: Sequence[str], mode: str = "ALL", add_special_tokens: bool = False
    ) -> List[List[int]]:
        encs = self.view(mode).encode_batch(list(texts), add_special_tokens=add_special_tokens)
        return [e.ids for e in encs]

    # -- pieces (§3.5) ----------------------------------------------------------
    def pieces(self, text: str, mode: str = "ALL") -> Tuple[int, List[int]]:
        """(normalized_byte_length, [piece end offsets in normalized bytes]).

        hf applies, per ordinary segment: normalize -> pre_tokenize_str; the returned
        (start, end) are CHAR offsets into the normalized segment (verified with
        é/emoji probes); we convert to byte offsets in the concatenated normalized
        stream. Added-token matches are one piece each, spanning their literal bytes
        (the ORIGINAL bytes — added-token content is never normalized; the
        normalized:true flag only affects *matching*, and the matched span is a span
        of the normalized text, so we take the normalized bytes for those).

        hf's two-pass extraction (raw pass then normalized pass on the leftover
        segments) is reproduced: raw-pattern tokens match first, then each ordinary
        segment is normalized and normalized-pattern tokens match within it.
        """
        return _pieces(self, text, mode)

    # -- decode ---------------------------------------------------------------
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self.t.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, seqs: Sequence[Sequence[int]], skip_special_tokens: bool = False
    ) -> List[str]:
        return self.t.decode_batch([list(s) for s in seqs], skip_special_tokens=skip_special_tokens)

    def special_ids(self) -> set:
        v = self.t.get_vocab()
        return {v.get(a.content, a.id) for a in self.spec.added if a.special}

    def added_ids(self) -> set:
        v = self.t.get_vocab()
        return {v.get(a.content, a.id) for a in self.spec.added}

    # -- stream ---------------------------------------------------------------
    def stream(self, pushes: Sequence[Sequence[int]], skip_special_tokens: bool = False) -> List[str]:
        """Reference stream decode of id-sequence pushes: per-push eager output with
        the held-tail rule, then the flush."""
        tb = self._token_bytes_map()
        special = self.special_ids()
        byte_pushes = [
            b"".join(b"" if (skip_special_tokens and i in special) else tb[i] for i in p if i in tb)
            for p in pushes
        ]
        return stream_reference(byte_pushes)

    def _token_bytes_map(self) -> dict:
        if not hasattr(self, "_tb"):
            import tokenizers

            obj = self.spec.obj
            tb: dict = {}
            dec = self.t.decoder
            model_vocab = obj["model"]["vocab"]
            if isinstance(dec, tokenizers.decoders.ByteLevel):
                u2b = _bytelevel_map()
                for s, i in model_vocab.items():
                    try:
                        tb[i] = bytes(u2b[ch] for ch in s)
                    except KeyError:
                        pass
            else:
                for s, i in model_vocab.items():
                    tb[i] = s.encode("utf-8")
            for a in self.spec.added:
                v = self.t.get_vocab()
                tb[v.get(a.content, a.id)] = a.content.encode("utf-8")
            self._tb = tb
        return self._tb


def _bytelevel_map() -> dict:
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


# ---------------------------------------------------------------------------
# pieces computation (§3.5) — mirrors hf's extract_and_normalize


def _pieces(tok: "Tok", text: str, mode: str) -> Tuple[int, List[int]]:
    """(normalized byte length, piece ends) per SPEC §3.5: an end indexes the caller's bytes wherever the
    normalizer left the text alone, else the normalized form. Splits overlap after an rstrip token (hf resumes
    matching at the token's raw end: find_matches); an end is the position of the split's end, so the bytes
    the two splits share are counted once."""
    t = tok.view(mode)
    spec = tok.spec
    hide_special = mode == "NONSPECIAL"
    no_added = mode == "NONE"

    raw_tokens = [] if no_added else [a for a in spec.added if not a.normalized]
    norm_tokens = [] if no_added else [a for a in spec.added if a.normalized]

    def blen(s: str) -> int:
        return len(s.encode("utf-8"))

    # the normalizers toks folds into its tables (sentencepiece-style Replace / Prepend): their text is never written,
    # so ends index the caller's bytes. Not an empty Sequence (deepseek: no normalizer, and phase-1 tokens match on
    # the gap as it is) nor NFC (materialized: the normalized stream).
    # Only bpe folds: a Unigram's chain (Precompiled, Replace) is materialized like NFC (unigram.md §10).
    norm_json = json.dumps(spec.obj.get("normalizer"))
    model = spec.obj.get("model") or {}
    bpe = model.get("type") == "BPE" or (model.get("type") is None and "merges" in model)
    folded = bpe and ('"Replace"' in norm_json or '"Prepend"' in norm_json)

    ends: List[int] = []

    def pre_tokenized(seg: str, base: int) -> None:
        pt = t.pre_tokenizer
        if pt is None:
            if seg:
                ends.append(base + blen(seg))
            return
        for _piece, (cs, ce) in pt.pre_tokenize_str(seg):
            if ce > cs:
                ends.append(base + blen(seg[:ce]))

    def ordinary(seg: str, base: int) -> int:
        """a phase-0 gap at normalized offset base: its normalized length"""
        if folded:
            # SPEC §3.5: a normalizer toks folds into its tables (sentencepiece-style Replace / Prepend) never
            # writes text, so the ends index the caller's bytes: hf's own original-referential offsets of the
            # pre-tokenizer's splits (a Prepend's symbol aligns to the gap's first char). Phase-1 tokens: hf's
            # added_vocabulary.rs splits the normalized gap at its normalized trie's matches (leftmost-longest;
            # toks accepts them only without a pre-tokenizer, config.c parse_spm).
            import re

            import tokenizers

            pts = tokenizers.PreTokenizedString(seg)
            pts.normalize(lambda ns: t.normalizer.normalize(ns))
            forms = sorted({_normalize_str(t, a.content) for a in norm_tokens if not (hide_special and a.special)},
                           key=len, reverse=True)
            if forms:
                rx = re.compile("|".join(re.escape(f) for f in forms))

                def split(_i, ns):
                    s, out, p = ns.normalized, [], 0
                    for m in rx.finditer(s):
                        out += [ns.slice((p, m.start()))] if m.start() > p else []
                        out.append(ns.slice(m.span()))
                        p = m.end()
                    return out + ([ns.slice((p, len(s)))] if p < len(s) else [])

                pts.split(split)
            if t.pre_tokenizer is not None:
                t.pre_tokenizer.pre_tokenize(pts)
            for _s, (b0, b1), _tok in pts.get_splits(offset_referential="original", offset_type="byte"):
                if b1 > b0:
                    ends.append(base + b1)
            return blen(seg)
        nseg = _normalize_str(t, seg)
        if norm_tokens:
            for tid, s, e in find_matches(nseg, norm_tokens, hide_special):     # positions inside nseg
                if tid is None:
                    pre_tokenized(nseg[s:e], base + blen(nseg[:s]))
                else:
                    ends.append(base + blen(nseg[:e]))
        else:
            pre_tokenized(nseg, base)
        return blen(nseg)

    re_c, nb = 0, 0              # the furthest split end so far (chars of text) and its normalized offset
    for tid, s, e in find_matches(text, raw_tokens, hide_special):
        s_nb = nb - (blen(text[s:re_c]) if s < re_c else 0)   # an overlap re-covers raw token bytes
        if tid is not None:
            e_nb = s_nb + blen(text[s:e])
            ends.append(e_nb)
        else:
            e_nb = s_nb + ordinary(text[s:e], s_nb)
        if e >= re_c:
            re_c, nb = e, e_nb
    return nb, ends


def _normalize_str(t, s: str) -> str:
    n = t.normalizer
    return s if n is None else n.normalize_str(s)


# ---------------------------------------------------------------------------
# loading


def load(path: str) -> Tok:
    import tokenizers

    obj = _read(path)
    spec = _spec_from_obj(obj, path)
    t = tokenizers.Tokenizer.from_str(spec.json_text)
    return Tok(t, spec)


def load_synthetic(obj: dict, name: str = "<synthetic>") -> Tok:
    import tokenizers

    spec = _spec_from_obj(obj, name)
    t = tokenizers.Tokenizer.from_str(spec.json_text)
    return Tok(t, spec)


# ---------------------------------------------------------------------------
# tiktoken oracle


class TikOracle:
    def __init__(self, encoding_name: str):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def encode(self, text: str, mode: str = "ALL") -> List[int]:
        if mode == "ALL":
            return self.enc.encode(text, allowed_special="all")
        if mode in ("NONSPECIAL", "NONE"):
            return self.enc.encode_ordinary(text)
        raise ModeError(mode)

    def encode_batch(self, texts: Sequence[str], mode: str = "ALL") -> List[List[int]]:
        return [self.encode(x, mode) for x in texts]

    def decode(self, ids: Sequence[int]) -> bytes:
        return self.enc.decode_bytes(list(ids))

    def special_tokens(self) -> dict:
        return dict(self.enc._special_tokens)
