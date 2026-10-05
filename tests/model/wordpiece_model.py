#!/usr/bin/env python3
"""tests/model/wordpiece_model.py: the executable reading of docs/algorithms/wordpiece.md.

hf tokenizers 0.23.2's encode() / decode() for the WordPiece family, written out from the contract, with no hf code:
added tokens (both phases, every option), BertNormalizer (and NFD / Lowercase / StripAccents / Sequence), the
BertPreTokenizer / WhitespaceSplit / Whitespace pre-tokenizers, the WordPiece model, the post-processors
(TemplateProcessing, BertProcessing, RobertaProcessing), truncation (with hf's early exit and its usize wrap),
padding and the WordPiece decoder. tests/model/run_wordpiece_fuzz.py diffs it against hf.

The per-code-point facts come from tests/data/wordpiece/unicode.json (tools/gen/bert_tables.py: the crates hf links
plus rust std's unicode 17.0, every table verified against hf on every scalar value). Everything above the
code-point level (orders, runs, splits, matching, truncation, padding) is this file's own reading of the doc.

A file whose features the doc does not cover raises Unsupported (toks refuses it at load, naming the feature).
Inputs on which hf fails raise HfError (hf's Exception) or HfPanic (a pyo3 PanicException in hf).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
U64 = 1 << 64


class Unsupported(Exception):
    pass


class HfError(Exception):
    pass


class HfPanic(Exception):
    pass


# ------------------------------------------------------------------------------------------- unicode data (§3)
class _U:
    def __init__(self):
        with open(os.path.join(ROOT, "tests", "data", "wordpiece", "unicode.json")) as f:
            d = json.load(f)

        def rset(rs):
            s = set()
            for a, b in rs:
                s.update(range(a, b + 1))
            return s

        self.remove = rset(d["remove"])          # clean_text removes
        self.ws = rset(d["whitespace"])           # White_Space (17.0): char::is_whitespace
        self.cjk = rset(d["cjk"])                 # is_chinese_char
        self.punct = rset(d["punct"])             # is_bert_punc
        self.mn = rset(d["mn"])                   # Mn (8.0): BertNormalizer's strip filter
        self.mark9 = rset(d["mark9"])             # General_Category=Mark (9.0): StripAccents' filter
        self.ccc = {c: v for c, v in d["ccc"]}    # canonical combining class (9.0), 0 when absent
        self.dec = {c: m for c, m in d["decomp"]}  # canonical decomposition (9.0), one level
        self.lower = {c: m for c, m in d["lower"]}  # char::to_lowercase (17.0) where it changes the char
        self.word = rset(d["word"])               # rust regex \w (probed)
        self._full = {}

    def decompose(self, c):
        r = self._full.get(c)
        if r is not None:
            return r
        if 0xAC00 <= c < 0xAC00 + 11172:          # hangul syllables: arithmetic
            k = c - 0xAC00
            r = [0x1100 + k // 588, 0x1161 + (k % 588) // 28]
            if k % 28:
                r.append(0x11A7 + k % 28)
        elif c in self.dec:
            r = []
            for x in self.dec[c]:
                r.extend(self.decompose(x))
        else:
            r = [c]
        self._full[c] = r
        return r


_UD = None


def U():
    global _UD
    if _UD is None:
        _UD = _U()
    return _UD


# --------------------------------------------------------------------------------------------- normalizers (§3)
def nfd(s):
    """NFD with the 9.0 tables: full decomposition, then a stable sort of every maximal non-starter run."""
    u = U()
    out = []
    for ch in s:
        out.extend(u.decompose(ord(ch)))
    res, i, n = [], 0, len(out)
    while i < n:
        if u.ccc.get(out[i], 0) == 0:
            res.append(out[i])
            i += 1
            continue
        j = i
        while j < n and u.ccc.get(out[j], 0) != 0:
            j += 1
        res.extend(sorted(out[i:j], key=lambda x: u.ccc.get(x, 0)))
        i = j
    return "".join(map(chr, res))


def lowercase(s):
    lo = U().lower
    return "".join("".join(map(chr, lo[ord(ch)])) if ord(ch) in lo else ch for ch in s)


class BertNormalizer:
    def __init__(self, d):
        self.clean_text = bool(d.get("clean_text", True))
        self.handle_chinese_chars = bool(d.get("handle_chinese_chars", True))
        sa = d.get("strip_accents", None)
        self.lowercase = bool(d.get("lowercase", True))
        self.strip = self.lowercase if sa is None else bool(sa)    # strip_accents None follows lowercase
        # ascii text: every step is a per-char table (NFD and Mn never touch ascii, no ascii char is CJK)
        t = {}
        if self.clean_text:
            for c in range(0x80):
                if c in U().remove:                 # removal first: \v \f are White_Space AND Cc (removed)
                    t[c] = None
                elif c in U().ws:
                    t[c] = " "
        if self.lowercase:
            for c in range(0x41, 0x5B):
                t[c] = chr(c + 32)
        self._ascii = str.maketrans(t)

    def __call__(self, s):
        if s.isascii():
            return s.translate(self._ascii)
        u = U()
        if self.clean_text:
            s = "".join(" " if ord(ch) in u.ws else ch for ch in s if ord(ch) not in u.remove)
        if self.handle_chinese_chars:
            s = "".join(" " + ch + " " if ord(ch) in u.cjk else ch for ch in s)
        if self.strip:
            s = "".join(ch for ch in nfd(s) if ord(ch) not in u.mn)
        if self.lowercase:
            s = lowercase(s)
        return s


def make_normalizer(d):
    if d is None:
        return None
    t = d.get("type")
    if t == "BertNormalizer":
        return BertNormalizer(d)
    if t == "NFD":
        return nfd
    if t == "Lowercase":
        return lowercase
    if t == "StripAccents":
        m9 = U().mark9
        return lambda s: "".join(ch for ch in s if ord(ch) not in m9)
    if t == "Sequence":
        steps = [make_normalizer(x) for x in d.get("normalizers", [])]
        steps = [x for x in steps if x is not None]

        def seq(s):
            for f in steps:
                s = f(s)
            return s
        return seq
    raise Unsupported("normalizer %s" % t)


# ------------------------------------------------------------------------------------------ pre-tokenizers (§4)
def bert_pre(words):
    u = U()
    out = []
    for w in words:
        cur = []
        for ch in w:
            c = ord(ch)
            if c in u.ws:
                if cur:
                    out.append("".join(cur))
                    cur = []
            elif c in u.punct:
                if cur:
                    out.append("".join(cur))
                    cur = []
                out.append(ch)
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
    return out


def ws_split(words):
    ws = U().ws
    out = []
    for w in words:
        cur = []
        for ch in w:
            if ord(ch) in ws:
                if cur:
                    out.append("".join(cur))
                    cur = []
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
    return out


def whitespace_pre(words):
    """Whitespace: the matches of \\w+|[^\\w\\s]+ are the pieces; \\s runs vanish."""
    u = U()
    out = []
    for w in words:
        cur, kind = [], None
        for ch in w:
            c = ord(ch)
            k = "w" if c in u.word else "s" if c in u.ws else "p"
            if k != kind and cur:
                if kind != "s":
                    out.append("".join(cur))
                cur = []
            kind = k
            cur.append(ch)
        if cur and kind != "s":
            out.append("".join(cur))
    return out


def make_pretok(d):
    if d is None:
        return lambda words: [w for w in words if w]
    t = d.get("type")
    if t == "BertPreTokenizer":
        return bert_pre
    if t == "WhitespaceSplit":
        return ws_split
    if t == "Whitespace":
        return whitespace_pre
    if t == "Sequence":
        steps = [make_pretok(x) for x in d.get("pretokenizers", [])]

        def seq(words):
            for f in steps:
                words = f(words)
            return words
        return seq
    raise Unsupported("pre_tokenizer %s" % t)


# ----------------------------------------------------------------------------------------------- the model (§5)
class WordPiece:
    def __init__(self, d):
        if d.get("type", "WordPiece") != "WordPiece":
            raise Unsupported("model %s" % d.get("type"))
        self.vocab = dict(d["vocab"])
        self.unk = d["unk_token"]
        self.prefix = d["continuing_subword_prefix"]
        self.max_chars = int(d["max_input_chars_per_word"])
        self.id2tok = {}
        for k, v in self.vocab.items():
            self.id2tok[v] = k                      # (hf's vocab_r: the last string of a duplicated id wins)
        self._memo = {}

    def unk_id(self):
        if self.unk not in self.vocab:
            raise HfError("WordPiece error: Missing [UNK] token from the vocabulary")
        return self.vocab[self.unk]

    def tokenize(self, w):
        r = self._memo.get(w)
        if r is None:
            r = self._tokenize(w)
            if len(self._memo) > 200000:
                self._memo.clear()
            self._memo[w] = r
        return list(r)

    def _tokenize(self, w):
        if len(w) > self.max_chars:                 # chars (unicode scalars), not bytes
            return [self.unk_id()]
        out = []
        start, n = 0, len(w)
        while start < n:
            end = n
            hit = None
            while start < end:
                sub = w[start:end]
                if start > 0:
                    sub = self.prefix + sub
                if sub in self.vocab:
                    hit = self.vocab[sub]
                    break
                end -= 1                            # one char shorter (hf: the last char's utf-8 length)
            if hit is None:
                return [self.unk_id()]
            out.append(hit)
            start = end
        return out


# --------------------------------------------------------------------------------------- added tokens (§2)
class Added:
    __slots__ = ("content", "id", "special", "single_word", "lstrip", "rstrip", "normalized", "form")


class AddedVocab:
    """hf AddedVocabulary::add_tokens over the file's list, in file order (one call, at load)."""

    def __init__(self, tj, model, normalizer):
        content_id = {}                       # added_tokens_map: content -> id
        by_id = {}                            # added_tokens_map_r: id -> token (a later token replaces)
        cache = {}                            # normalized_cache: id -> normalized form (only when it differs)
        special = set()                       # special_tokens_set: insert-only
        next_id = len(model.vocab)            # model.get_vocab_size(): the number of vocab strings
        for d in tj.get("added_tokens") or []:
            content = d["content"]
            if content == "":
                continue
            t = Added()
            t.content = content
            t.special = bool(d["special"])
            t.single_word = bool(d["single_word"])
            t.lstrip = bool(d["lstrip"])
            t.rstrip = bool(d["rstrip"])
            t.normalized = bool(d["normalized"])
            key = (content, t.single_word, t.lstrip, t.rstrip, t.normalized, t.special)
            if content in content_id:
                old = by_id.get(content_id[content])
                if old is not None and (old.content, old.single_word, old.lstrip, old.rstrip, old.normalized,
                                        old.special) == key:
                    continue
            if content in content_id:
                tid = content_id[content]
            elif content in model.vocab:
                tid = model.vocab[content]
            else:
                tid = next_id
                next_id += 1
            t.id = tid
            if t.normalized and normalizer is not None:
                f = normalizer(content)
                if f != content:
                    cache[tid] = f
            content_id[content] = tid
            if t.special:
                special.add(content)
            by_id[tid] = t
        for tid, t in by_id.items():
            t.form = cache.get(tid, t.content) if t.normalized else t.content
        self.by_id = by_id
        self.cache = cache
        self.tokens = list(by_id.values())
        self.special_contents = special
        for ph in (False, True):
            forms = [t.form for t in self.tokens if t.normalized == ph]
            if len(set(forms)) != len(forms):
                raise HfError("added tokens: a duplicated pattern in one phase (daachorse refuses it)")
        for t in self.tokens:
            if t.form == "":
                raise Unsupported("added token whose normalized form is empty")

    def phase(self, normalized):
        return [t for t in self.tokens if t.normalized == normalized]

    def split(self, s, toks, drop_specials):
        """hf find_matches: leftmost-longest non-overlapping raw matches (daachorse), then the per-match policy."""
        if not s:
            return [(None, 0, 0)]
        if not toks:
            return [(None, 0, len(s))]
        u = U()
        forms = sorted(((t.form, t) for t in toks), key=lambda x: -len(x[0]))
        firsts = {}
        for f, t in forms:
            firsts.setdefault(f[0], []).append((f, t))
        splits = []
        start_offset = 0
        i, n = 0, len(s)
        while i < n:
            m = None
            for f, t in firsts.get(s[i], ()):
                if s.startswith(f, i):
                    m = (i, i + len(f), t)
                    break
            if m is None:
                i += 1
                continue
            i = m[1]                                # the raw match end (never the stripped end)
            start, stop, t = m
            if drop_specials and t.content in self.special_contents:
                continue
            if t.single_word:
                start_space = start == 0 or ord(s[start - 1]) not in u.word
                stop_space = stop == n or ord(s[stop]) not in u.word
                if not (start_space and stop_space):
                    continue
            if t.lstrip:
                j = start
                while j > 0 and ord(s[j - 1]) in u.ws:
                    j -= 1
                start = max(j, start_offset)
            if t.rstrip:
                while stop < n and ord(s[stop]) in u.ws:
                    stop += 1
            if start_offset < start:
                splits.append((None, start_offset, start))
            splits.append((t.id, start, stop))
            start_offset = stop
        if start_offset != n:
            splits.append((None, start_offset, n))
        return splits


# ---------------------------------------------------------------------------------------- post-processors (§6)
class PostProcessor:
    def __init__(self, d):
        self.prefix, self.suffix = [], []
        self.n_added = 0
        if d is None:
            return
        t = d.get("type")
        if t == "TemplateProcessing":
            st = d.get("special_tokens", {})
            seen_seq = False
            for piece in d["single"]:
                if "Sequence" in piece:
                    if piece["Sequence"]["id"] != "A":
                        raise Unsupported("template single with $B")
                    seen_seq = True
                    continue
                name = piece["SpecialToken"]["id"]
                ids = list(st[name]["ids"]) if name in st else []
                (self.suffix if seen_seq else self.prefix).extend(ids)
            self.n_added = len(self.prefix) + len(self.suffix)
        elif t in ("BertProcessing", "RobertaProcessing"):
            self.prefix = [d["cls"][1]]
            self.suffix = [d["sep"][1]]
            self.n_added = 2
        else:
            raise Unsupported("post_processor %s" % t)

    def apply(self, ids, add_special_tokens):
        return self.prefix + ids + self.suffix if add_special_tokens else ids


# ------------------------------------------------------------------------------------------------ the tokenizer
class Tokenizer:
    def __init__(self, tj):
        if isinstance(tj, (str, bytes)):
            tj = json.loads(tj)
        self.tj = tj
        self.normalizer = make_normalizer(tj.get("normalizer"))
        self.pretok = make_pretok(tj.get("pre_tokenizer"))
        self.model = WordPiece(tj["model"])
        self.added = AddedVocab(tj, self.model, self.normalizer)
        self.pp = PostProcessor(tj.get("post_processor"))
        self.trunc = tj.get("truncation")
        self.pad = tj.get("padding")
        dec = tj.get("decoder")
        if dec is not None and dec.get("type") != "WordPiece":
            raise Unsupported("decoder %s" % dec.get("type"))
        self.decoder = dec

    # -------------------------------------------------------------------------------------------- §2 + §3 + §4
    def splits(self, text, mode="ALL"):
        """the splits the model sees, in order: ('tok', id) for added tokens, ('word', str) for pre-tokens."""
        if mode == "NONE":
            p0, p1 = [], []
        else:
            p0, p1 = self.added.phase(False), self.added.phase(True)
        drop = mode == "NONSPECIAL"
        out = []
        for tid, a, b in self.added.split(text, p0, drop):
            if tid is not None:
                if b > a:                                    # an empty token split is dropped (hf filters it)
                    out.append(("tok", tid))
                elif a > b:
                    raise HfPanic("AddedVocabulary bad split")
                continue
            gap = text[a:b]
            if gap == "":
                continue
            norm = self.normalizer(gap) if self.normalizer is not None else gap
            for tid1, a1, b1 in self.added.split(norm, p1, drop):
                if tid1 is not None:
                    if b1 > a1:
                        out.append(("tok", tid1))
                    elif a1 > b1:
                        raise HfPanic("AddedVocabulary bad split")
                    continue
                seg = norm[a1:b1]
                if seg == "":
                    continue
                for w in self.pretok([seg]):
                    if w:
                        out.append(("word", w))
        return out

    def words(self, text, mode="ALL"):
        return [x[1] for x in self.splits(text, mode) if x[0] == "word"]

    # --------------------------------------------------------------------------------------------- §5 + §7
    def _tokenize(self, sp):
        """PreTokenizedString::tokenize / tokenize_with_limit (do_tokenize's early exit)."""
        tr = self.trunc
        limit = None
        if tr is not None and tr.get("strategy", "LongestFirst") != "OnlySecond":
            limit = (int(tr["max_length"]), tr.get("direction", "Right"))
        if limit is None:
            ids = []
            for kind, v in sp:
                ids.extend([v] if kind == "tok" else self.model.tokenize(v))
            return ids
        max_tokens, direction = limit
        parts = []
        total = 0
        seq = sp if direction == "Right" else list(reversed(sp))
        for kind, v in seq:
            if kind == "tok":
                parts.append([v])
                total += 1
                continue                                     # no limit check after an added token
            t = self.model.tokenize(v)
            parts.append(t)
            total += len(t)
            if total >= max_tokens:
                break
        if direction != "Right":
            parts.reverse()
        return [i for p in parts for i in p]

    def _truncate(self, ids, add_special_tokens):
        tr = self.trunc
        if tr is None:
            return ids
        max_length = int(tr["max_length"])
        stride = int(tr.get("stride", 0))
        direction = tr.get("direction", "Right")
        strategy = tr.get("strategy", "LongestFirst")
        n_added = self.pp.n_added
        if add_special_tokens and n_added > 0:
            max_length = (max_length - n_added) % U64      # usize subtraction: wraps in the release build
        if max_length == 0:
            return []
        if len(ids) <= max_length:
            return ids
        if strategy == "OnlySecond":
            raise HfError("Truncation error: Second sequence not provided")
        if stride >= max_length:
            raise HfPanic("`stride` must be strictly less than `max_len`")
        return ids[:max_length] if direction == "Right" else ids[len(ids) - max_length:]

    def _pad(self, ids):
        p = self.pad
        if p is None:
            return ids
        st = p.get("strategy", "BatchLongest")
        target = len(ids) if st == "BatchLongest" else int(st["Fixed"])
        m = p.get("pad_to_multiple_of")
        if m and target % m:
            target += m - target % m
        if len(ids) >= target:
            return ids
        pad = [int(p.get("pad_id", 0))] * (target - len(ids))
        return ids + pad if p.get("direction", "Right") == "Right" else pad + ids

    def encode(self, text, mode="ALL", add_special_tokens=True):
        sp = self.splits(text, mode)
        ids = self._tokenize(sp)
        ids = self._truncate(ids, add_special_tokens)
        ids = self.pp.apply(ids, add_special_tokens)
        return self._pad(ids)

    def encode_pp_both(self, text, mode="ALL"):
        """{True: result, False: result} for add_special_tokens on / off, sharing the splits and the model run;
        a result is ('ok', ids), ('err', msg) or ('panic', msg)."""
        try:
            ids = self._tokenize(self.splits(text, mode))
        except HfPanic as e:
            return {True: ("panic", str(e)), False: ("panic", str(e))}
        except HfError as e:
            return {True: ("err", str(e)), False: ("err", str(e))}
        res = {}
        for pp in (True, False):
            try:
                res[pp] = ("ok", self._pad(self.pp.apply(self._truncate(list(ids), pp), pp)))
            except HfPanic as e:
                res[pp] = ("panic", str(e))
            except HfError as e:
                res[pp] = ("err", str(e))
        return res

    # ------------------------------------------------------------------------------------------------ §8
    def id_to_token(self, i):
        t = self.added.by_id.get(i)
        if t is not None:
            return self.added.cache.get(i, t.content)       # simple_id_to_token: the cached normalized form
        return self.model.id2tok.get(i)

    def decode(self, ids, skip_special_tokens=False):
        toks = []
        for i in ids:
            s = self.id_to_token(i)
            if s is None:
                continue                                    # hf skips ids without a string
            if skip_special_tokens and s in self.added.special_contents:
                continue
            toks.append(s)
        if self.decoder is None:
            return " ".join(toks)
        prefix = self.decoder.get("prefix", "##")
        cleanup = self.decoder.get("cleanup", True)
        out = []
        for k, t in enumerate(toks):
            if k != 0:
                if t.startswith(prefix):
                    t = t[len(prefix):]
                else:
                    t = " " + t
            if cleanup:
                t = (t.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
                     .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't")
                     .replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re"))
            out.append(t)
        return "".join(out)


def load(path):
    with open(path, encoding="utf-8") as f:
        return Tokenizer(json.load(f))
