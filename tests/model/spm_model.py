#!/usr/bin/env python3
"""spm_model.py: the executable reading of docs/algorithms/spm_bpe.md (sentencepiece-style bpe).

A straightforward python transcription of how hf tokenizers 0.23.2 runs a BPE model WITHOUT a byte-level
alphabet (llama 2, mistral v0.1-v0.3, gemma 1-4, phi-3, yi, codellama, tinyllama, ...):

  - the added-token split: two phases (normalized=false tokens on the raw text, normalized=true tokens on
    each normalized gap, matched by their normalized form), daachorse leftmost-longest, hf's per-match
    policy (mode, single_word, lstrip, rstrip, resume at the raw match end)            doc §2
  - the normalizers Prepend and Replace (literal pattern), Sequence                      doc §3
  - the pre-tokenizers Metaspace (replacement, prepend_scheme, split) and Split (literal pattern, every
    behaviour, invert), Sequence                                                         doc §4
  - the model: one symbol per char (vocab lookup), byte fallback to <0xNN>, unk with fuse_unk,
    ignore_merges, then lowest-rank-leftmost merging                                     doc §5
  - post-processing (TemplateProcessing, single sequence)                                doc §7
  - decode: token strings, skip_special, the decoder chain (Replace, ByteFallback, Fuse, Strip,
    Metaspace, Sequence)                                                                 doc §8

Nothing here imports hf. tests/model/run_spm_fuzz.py diffs this file against tokenizers 0.23.2; a
mismatch is a defect in the doc (or in this transcription of it). Anything outside the doc raises
Unsupported(feature): that is the set toks refuses at load.

Text is handled as python str (code points). Offsets are code-point indices; the only offset the
semantics consult (Metaspace prepend_scheme=first) is compared with 0, where code points and bytes agree.
"""
import json

REPL = "\u2581"
# rust regex \s (White_Space): the 25 code points (tests/model/REVIEW.md: probed equal to onig's \s too)
WHITE_SPACE = frozenset(map(chr, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
                                  0x2028, 0x2029, 0x202F, 0x205F, 0x3000]))


class Unsupported(Exception):
    """a tokenizer.json feature outside docs/algorithms/spm_bpe.md (toks refuses it by this name)"""


class HfError(Exception):
    """an input on which hf itself fails (an error or a panic), never a result"""


def _absent(v):
    return v is None


# ================================================================================ §5 the model
class Model:
    def __init__(self, m):
        t = m.get("type", "BPE")
        if t != "BPE":
            raise Unsupported(f"model type {t}")
        if m.get("dropout") not in (None, 0, 0.0):
            raise Unsupported("model dropout")
        for k in ("continuing_subword_prefix", "end_of_word_suffix"):
            if m.get(k) not in (None, ""):
                raise Unsupported(f"model {k}")
        self.vocab = dict(m["vocab"])                       # str -> id (ids may have holes)
        self.vocab_r = {i: s for s, i in self.vocab.items()}
        if len(self.vocab_r) != len(self.vocab):
            raise Unsupported("model.vocab: two strings share an id")
        self.unk_token = m.get("unk_token")
        self.fuse_unk = bool(m.get("fuse_unk", False))
        self.byte_fallback = bool(m.get("byte_fallback", False))
        self.ignore_merges = bool(m.get("ignore_merges", False))
        self.byte_id = [self.vocab.get("<0x%02X>" % b) for b in range(256)]
        self.merges = {}                                    # (left id, right id) -> (rank, merged id)
        merges = m.get("merges", [])
        if merges and isinstance(merges[0], str):           # legacy "left right" strings: hf drops
            merges = [x for x in merges if not x.startswith("#version")]   # "#version..." lines first
        for rank, mg in enumerate(merges):
            if isinstance(mg, str):
                parts = mg.split(" ")
                if len(parts) != 2:
                    raise Unsupported("model.merges entry is not 'left right'")
                a, b = parts
            else:
                a, b = mg
            try:
                self.merges[(self.vocab[a], self.vocab[b])] = (rank, self.vocab[a + b])   # duplicates: last wins
            except KeyError:
                raise Unsupported("model.merges token out of vocabulary")  # hf refuses the file too

    def symbols(self, piece):
        """doc §5.1: one symbol per char; byte fallback; unk (+ fuse_unk). Returns the initial ids."""
        out = []
        unk = None                                          # the pending unk id (one per run when fused)
        for ch in piece:
            tid = self.vocab.get(ch)
            if tid is not None:
                if unk is not None:
                    out.append(unk)
                    unk = None
                out.append(tid)
                continue
            if self.byte_fallback:
                bids = [self.byte_id[b] for b in ch.encode("utf-8")]
                if None not in bids:
                    out.extend(bids)                        # NOTE: a pending unk is NOT flushed first
                    continue
            if self.unk_token is not None:
                if unk is not None and self.fuse_unk:
                    continue                                # fused into the pending unk
                uid = self.vocab.get(self.unk_token)
                if uid is None:
                    raise HfError("UnkTokenOutOfVocabulary")
                if unk is not None:
                    out.append(unk)
                unk = uid
            # no unk token: the char vanishes
        if unk is not None:
            out.append(unk)
        return out

    def merge(self, syms):
        """doc §5.2: repeatedly merge the adjacent pair with the lowest rank, ties to the leftmost."""
        mm = self.merges
        INF = 1 << 62
        n = len(syms)
        if n < 2:
            return syms
        get = mm.get
        ranks = [get((syms[i], syms[i + 1]), (INF, 0))[0] for i in range(n - 1)]
        while ranks:                                        # bound: n - 1 merges at most
            r = min(ranks)
            if r == INF:
                break
            i = ranks.index(r)                              # leftmost pair of that rank
            syms[i:i + 2] = [mm[(syms[i], syms[i + 1])][1]]
            del ranks[i]
            if i > 0:
                ranks[i - 1] = get((syms[i - 1], syms[i]), (INF, 0))[0]
            if i < len(syms) - 1:
                ranks[i] = get((syms[i], syms[i + 1]), (INF, 0))[0]
        return syms

    def tokenize(self, piece):
        """hf Model::tokenize on one pre-token (doc §5)."""
        if piece == "":
            return []
        if self.ignore_merges:
            tid = self.vocab.get(piece)
            if tid is not None:
                return [tid]
        return self.merge(self.symbols(piece))

    # ---- doc §5.5 certified cuts ----------------------------------------------------------------------
    def pairs(self):
        """every adjacent (char, char) inside a model-vocab string"""
        if getattr(self, "_pairs", None) is None:
            ps = set()
            for s in self.vocab:
                for i in range(1, len(s)):
                    ps.add((s[i - 1], s[i]))
            self._pairs = ps
        return self._pairs

    def words(self, piece):
        """the piece cut at every boundary x|y where x and y are vocab chars and no vocab string holds xy"""
        if self.ignore_merges or not piece:
            return [piece] if piece else []
        ps, v = self.pairs(), self.vocab
        out, start = [], 0
        for i in range(1, len(piece)):
            x, y = piece[i - 1], piece[i]
            if x in v and y in v and (x, y) not in ps:
                out.append(piece[start:i])
                start = i
        out.append(piece[start:])
        return out

    def tokenize_words(self, piece):
        """doc §5.5: == tokenize(piece) (the theorem); the fuzz checks it against hf"""
        out = []
        for w in self.words(piece):
            out.extend(self.tokenize(w))
        return out


# ================================================================================ §3 normalizers
def _literal_matches(s, pat):
    """leftmost non-overlapping occurrences of a literal (onig on the escaped string): [(start, end)]"""
    out = []
    i = s.find(pat)
    while i >= 0:                                           # bound: len(s) / len(pat)
        out.append((i, i + len(pat)))
        i = s.find(pat, i + len(pat))
    return out


def parse_normalizer(n):
    """-> list of ('prepend', s) | ('replace', pattern, content)"""
    if n is None:
        return []
    t = n.get("type")
    if t == "Sequence":
        ops = []
        for x in n["normalizers"]:
            ops.extend(parse_normalizer(x))
        return ops
    if t == "Prepend":
        return [("prepend", n["prepend"])]
    if t == "Replace":
        p = n["pattern"]
        if "String" not in p:
            raise Unsupported("normalizer Replace with a Regex pattern")
        if p["String"] == "":
            raise Unsupported("normalizer Replace with an empty pattern")
        return [("replace", p["String"], n["content"])]
    raise Unsupported(f"normalizer {t}")


def normalize(ops, chars, orig):
    """doc §3. chars: list of code points (str of len 1); orig: their original offsets (alignment starts).
    Returns new (chars, orig)."""
    for op in ops:
        if op[0] == "prepend":
            if chars:                                       # Prepend skips an empty string
                a = orig[0]
                chars = list(op[1]) + chars
                orig = [a] * len(op[1]) + orig
        else:
            _, pat, content = op
            s = "".join(chars)
            ms = _literal_matches(s, pat)
            if not ms:
                continue
            nc, no, last = [], [], 0
            for (a, b) in ms:
                nc.extend(chars[last:a])
                no.extend(orig[last:a])
                nc.extend(content)
                no.extend([orig[b - 1]] * len(content))     # inserted chars align to the match's last char
                last = b
            nc.extend(chars[last:])
            no.extend(orig[last:])
            chars, orig = nc, no
    return chars, orig


# ================================================================================ §4 pre-tokenizers
BEHAVIORS = ("removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous")


def _behavior(b):
    k = {"Removed": "removed", "Isolated": "isolated", "MergedWithPrevious": "merged_with_previous",
         "MergedWithNext": "merged_with_next", "Contiguous": "contiguous"}.get(b, b)
    if k not in BEHAVIORS:
        raise Unsupported(f"pre_tokenizer Split behavior {b}")
    return k


def parse_pretok(p):
    """-> list of ('metaspace', repl, scheme, split) | ('split', pattern, behavior, invert)"""
    if p is None:
        return []
    t = p.get("type")
    if t == "Sequence":
        ops = []
        for x in p["pretokenizers"]:
            ops.extend(parse_pretok(x))
        return ops
    if t == "Metaspace":
        scheme = p.get("prepend_scheme", "always")
        if p.get("add_prefix_space") is False:
            if scheme != "never":
                raise HfError("add_prefix_space does not match declared prepend_scheme")
            scheme = "never"
        if scheme not in ("always", "first", "never"):
            raise Unsupported(f"Metaspace prepend_scheme {scheme}")
        r = p["replacement"]
        if len(r) != 1:
            raise Unsupported("Metaspace replacement is not one char")
        return [("metaspace", r, scheme, bool(p.get("split", True)))]
    if t == "Split":
        pat = p["pattern"]
        if "String" not in pat:
            raise Unsupported("pre_tokenizer Split with a Regex pattern")
        if pat["String"] == "":
            raise Unsupported("pre_tokenizer Split with an empty pattern")
        return [("split", pat["String"], _behavior(p["behavior"]), bool(p.get("invert", False)))]
    raise Unsupported(f"pre_tokenizer {t}")


def _split_spans(n, matches, behavior):
    """hf NormalizedString::split: matches = [((a, b), is_match)] covering [0, n) -> kept spans"""
    if behavior == "isolated":
        return [ab for ab, _ in matches]
    if behavior == "removed":
        return [ab for ab, m in matches if not m]
    acc = []
    prev = False
    if behavior == "contiguous":
        for (a, b), m in matches:
            if m == prev and acc:
                acc[-1] = (acc[-1][0], b)
            else:
                acc.append((a, b))
            prev = m
        return acc
    if behavior == "merged_with_previous":
        for (a, b), m in matches:
            if m and not prev and acc:
                acc[-1] = (acc[-1][0], b)
            else:
                acc.append((a, b))
            prev = m
        return acc
    for (a, b), m in reversed(matches):                     # merged_with_next
        if m and not prev and acc:
            acc[-1] = (a, acc[-1][1])
        else:
            acc.append((a, b))
        prev = m
    acc.reverse()
    return acc


def _cover(n, ms, invert=False):
    """matched spans -> the full cover [((a, b), is_match)] (hf Pattern::find_matches), optionally inverted"""
    out, last = [], 0
    for (a, b) in ms:
        if a > last:
            out.append(((last, a), False))
        out.append(((a, b), True))
        last = b
    if last < n:
        out.append(((last, n), False))
    if invert:
        out = [(ab, not m) for ab, m in out]
    return out


def pretokenize(ops, chars, orig):
    """doc §4: one normalized split (chars, orig) -> list of pieces (str)."""
    splits = [(chars, orig)]
    for op in ops:
        nxt = []
        for (cs, og) in splits:
            if op[0] == "metaspace":
                _, r, scheme, split = op
                cs = [r if c == " " else c for c in cs]
                if not cs or cs[0] != r:
                    if scheme == "always" or (scheme == "first" and og and og[0] == 0):
                        if cs:                              # NormalizedString::prepend needs a first char
                            cs = [r] + cs
                            og = [og[0]] + og
                if not split:
                    nxt.append((cs, og))
                    continue
                cuts = [i for i, c in enumerate(cs) if c == r]
                spans = _split_spans(len(cs), _cover(len(cs), [(i, i + 1) for i in cuts]), "merged_with_next")
            else:
                _, pat, behavior, invert = op
                s = "".join(cs)
                spans = _split_spans(len(cs), _cover(len(cs), _literal_matches(s, pat), invert), behavior)
            for (a, b) in spans:
                if b > a:                                   # empty splits are dropped
                    nxt.append((cs[a:b], og[a:b]))
        splits = nxt
    return ["".join(cs) for cs, _ in splits if cs]


# ================================================================================ §2 added tokens
class Added:
    __slots__ = ("content", "id", "special", "single_word", "lstrip", "rstrip", "normalized", "match")


def build_added(tj, model, norm_ops):
    """hf AddedVocabulary::add_tokens over the file's list, in order. Returns (id -> Added, specials, cache):
    the last token listed under an id holds it; specials are contents (specialness sticks); cache is hf's
    normalized_cache, keyed by ID and only ever written: a normalized token whose form differs from its
    content stores the form under its id, and a later token taking the same id does not clear it."""
    by_id, by_content, specials, cache = {}, {}, set(), {}
    next_id = len(model.vocab)                              # vocab.len(), not max id + 1
    for d in tj.get("added_tokens") or []:
        c = d["content"]
        if c == "":
            continue
        a = Added()
        a.content = c
        a.special = bool(d.get("special", False))
        a.single_word = bool(d.get("single_word", False))
        a.lstrip = bool(d.get("lstrip", False))
        a.rstrip = bool(d.get("rstrip", False))
        a.normalized = bool(d.get("normalized", not a.special))
        if c in by_content:
            tid = by_content[c]
        elif c in model.vocab:
            tid = model.vocab[c]
        else:
            tid = next_id
            next_id += 1
        a.id = tid
        if a.normalized and norm_ops:
            nc, _ = normalize(norm_ops, list(c), list(range(len(c))))
            if "".join(nc) != c:
                cache[tid] = "".join(nc)
        by_content[c] = tid
        if a.special:
            specials.add(c)
        by_id[tid] = a                                      # the last token listed under an id holds it
    for tid, a in by_id.items():                            # hf refresh_added_tokens: the trie patterns
        a.match = cache.get(tid, a.content) if a.normalized else a.content
        if a.match == "":
            raise Unsupported("added token whose normalized form is empty")
    return by_id, specials, cache


_WORD = None


def _word_set():
    """rust regex \\w as hf 0.23.2 runs it (single_word's ends_with_word / starts_with_word), probed through
    hf itself over every scalar value: c is a word char iff hf refuses the single_word token Q in c + "Q".
    (tests/model/toks_model.py's load_word_set misses Join_Control, U+200C / U+200D: PropList.txt, not
    DerivedCoreProperties.txt.) Cached in tests/model/wordset_hf.pkl."""
    global _WORD
    if _WORD is not None:
        return _WORD
    import os
    import pickle
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordset_hf.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            _WORD = pickle.load(f)
        return _WORD
    from tokenizers import Tokenizer
    vocab = {"<0x%02X>" % b: b for b in range(256)}            # "Q" itself is not in the vocab: unmatched
    tj = {"version": "1.0", "truncation": None, "padding": None, "normalizer": None, "pre_tokenizer": None,
          "post_processor": None, "decoder": None,
          "added_tokens": [{"id": 257, "content": "Q", "single_word": True, "lstrip": False, "rstrip": False,
                            "normalized": False, "special": False}],
          "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                    "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True, "ignore_merges": False,
                    "vocab": vocab, "merges": []}}
    tk = Tokenizer.from_str(json.dumps(tj))
    qid = tk.token_to_id("Q")
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    word = set()
    for st in range(0, len(cps), 4096):
        chunk = cps[st:st + 4096]
        for c, e in zip(chunk, tk.encode_batch([chr(c) + "Q" for c in chunk], add_special_tokens=False)):
            if qid not in e.ids:
                word.add(c)
    with open(path, "wb") as f:
        pickle.dump(word, f)
    _WORD = word
    return word


def added_split(text, toks, drop, specials):
    """kernels.md §4 policy on a str: [('gap', a, b) | ('tok', a, b, id)]. toks: the phase's tokens."""
    if not toks:
        return [("gap", 0, len(text))] if text else []
    by_first = {}
    for t in toks:
        by_first.setdefault(t.match[0], []).append(t)
    for k in by_first:
        by_first[k].sort(key=lambda t: -len(t.match))
    out, prev, n, s = [], 0, len(text), 0
    while s < n:                                            # bound: n (s advances >= 1)
        lst = by_first.get(text[s])
        m = None
        if lst:
            for t in lst:
                if text.startswith(t.match, s):
                    m = t
                    break
        if m is None:
            s += 1
            continue
        start, stop = s, s + len(m.match)
        s = stop                                            # resume at the raw match end, whatever happens
        if drop and m.content in specials:
            continue
        if m.single_word:
            w = _word_set()
            if (start > 0 and ord(text[start - 1]) in w) or (stop < n and ord(text[stop]) in w):
                continue
        if m.lstrip:
            j = start
            while j > 0 and text[j - 1] in WHITE_SPACE:
                j -= 1
            start = max(j, prev)
        if m.rstrip:
            while stop < n and text[stop] in WHITE_SPACE:
                stop += 1
        if prev < start:
            out.append(("gap", prev, start))
        if start > stop:                                    # lstrip clamped past the match: hf panics
            raise HfError("AddedVocabulary bad split (a token span with start > end)")
        if start < stop:                                    # an empty token span vanishes with its id
            out.append(("tok", start, stop, m.id))
        prev = stop
    if prev != n:
        out.append(("gap", prev, n))
    return out


# ================================================================================ §7 post-processor
def parse_post(pp, token_to_id):
    if pp is None:
        return [], []
    t = pp.get("type")
    if t == "Sequence":
        pre, post = [], []
        for x in pp["processors"]:
            a, b = parse_post(x, token_to_id)
            pre, post = pre + a, b + post
        return pre, post
    if t != "TemplateProcessing":
        raise Unsupported(f"post_processor {t}")
    pre, post, seen = [], [], False
    for piece in pp["single"]:
        if "Sequence" in piece:
            if piece["Sequence"]["id"] != "A" or seen:
                raise Unsupported("post_processor single template")
            seen = True
            continue
        ids = pp["special_tokens"][piece["SpecialToken"]["id"]]["ids"]
        (post if seen else pre).extend(ids)
    if not seen:
        raise Unsupported("post_processor template without $A")
    return pre, post


# ================================================================================ §8 decode
def parse_decoder(d):
    if d is None:
        return None
    t = d.get("type")
    if t == "Sequence":
        ops = []
        for x in d["decoders"]:
            sub = parse_decoder(x)
            ops.extend(sub)
        return ops
    if t == "Replace":
        p = d["pattern"]
        if "String" not in p or p["String"] == "":
            raise Unsupported("decoder Replace pattern")
        return [("replace", p["String"], d["content"])]
    if t == "ByteFallback":
        return [("byte_fallback",)]
    if t == "Fuse":
        return [("fuse",)]
    if t == "Strip":
        return [("strip", d["content"], int(d["start"]), int(d["stop"]))]
    if t == "Metaspace":
        scheme = d.get("prepend_scheme", "always")
        if d.get("add_prefix_space") is False:
            scheme = "never"
        return [("metaspace", d["replacement"], scheme)]
    raise Unsupported(f"decoder {t}")


def _byte_token(tok):
    """hf ByteFallback: 6 bytes, '<0x' .. '>', u8::from_str_radix(tok[3..5], 16)"""
    b = tok.encode("utf-8")
    if len(b) != 6 or not b.startswith(b"<0x") or not b.endswith(b">"):
        return None
    h = b[3:5]
    hexd = b"0123456789abcdefABCDEF"
    if h[0:1] == b"+":
        if h[1] in hexd:
            return int(h[1:2], 16)
        return None
    if h[0] in hexd and h[1] in hexd:
        return int(h, 16)
    return None


def decode_chain(ops, tokens):
    for op in ops:
        k = op[0]
        if k == "replace":
            tokens = [t.replace(op[1], op[2]) for t in tokens]
        elif k == "byte_fallback":
            out, run = [], []
            for t in tokens:
                v = _byte_token(t)
                if v is not None:
                    run.append(v)
                    continue
                if run:
                    try:
                        out.append(bytes(run).decode("utf-8"))
                    except UnicodeDecodeError:
                        out.extend(["\ufffd"] * len(run))   # the whole run, one U+FFFD per byte
                    run = []
                out.append(t)
            if run:
                try:
                    out.append(bytes(run).decode("utf-8"))
                except UnicodeDecodeError:
                    out.extend(["\ufffd"] * len(run))
            tokens = out
        elif k == "fuse":
            tokens = ["".join(tokens)]
        elif k == "strip":
            _, c, start, stop = op
            new = []
            for t in tokens:
                chars = list(t)
                a = 0
                for i in range(min(start, len(chars))):
                    if chars[i] == c:
                        a = i + 1
                    else:
                        break
                z = len(chars)
                for i in range(stop):
                    j = len(chars) - i - 1
                    if j < 0:
                        raise HfError("Strip decoder index out of bounds (hf panics)")
                    if chars[j] == c:
                        z = j
                    else:
                        break
                if a > z:
                    raise HfError("Strip decoder slice start > end (hf panics)")
                new.append("".join(chars[a:z]))
            tokens = new
        elif k == "metaspace":
            _, r, scheme = op
            tokens = ["".join(("" if (i == 0 and scheme != "never") else " ") if ch == r else ch for ch in t)
                      for i, t in enumerate(tokens)]
    return tokens


# ================================================================================ the pipeline
class SpmTokenizer:
    def __init__(self, tj):
        if isinstance(tj, (str, bytes)):
            tj = json.loads(tj)
        for k in ("truncation", "padding"):
            if tj.get(k) is not None:
                raise Unsupported(k)
        self.model = Model(tj["model"])
        self.norm = parse_normalizer(tj.get("normalizer"))
        self.pretok = parse_pretok(tj.get("pre_tokenizer"))
        self.decoder = parse_decoder(tj.get("decoder"))
        self.added, self.specials, self.norm_cache = build_added(tj, self.model, self.norm)
        # phase 0: normalized=false tokens on the raw text; phase 1: normalized=true tokens, by their
        # normalized form, on each normalized gap (with no normalizer the form is the content itself)
        self.phase0 = [a for a in self.added.values() if not a.normalized]
        self.phase1 = [a for a in self.added.values() if a.normalized]
        self.pre, self.post = parse_post(tj.get("post_processor"), None)

    # -- encode ---------------------------------------------------------------------------------------
    def encode(self, text, mode="ALL", add_special_tokens=True, cuts=False):
        """mode: ALL | NONSPECIAL | NONE (SPEC §3.2). cuts: encode every piece word by word at the
        certified cuts of doc §5.5 (the same ids, by the theorem)."""
        ids = []
        drop = mode == "NONSPECIAL"
        none = mode == "NONE"
        tokenize = self.model.tokenize_words if cuts else self.model.tokenize
        for seg in added_split(text, [] if none else self.phase0, drop, self.specials):
            if seg[0] == "tok":
                ids.append(seg[3])
                continue
            a, b = seg[1], seg[2]
            chars, orig = normalize(self.norm, list(text[a:b]), list(range(a, b)))
            if not chars:
                continue
            s = "".join(chars)
            for seg1 in added_split(s, [] if none else self.phase1, drop, self.specials):
                if seg1[0] == "tok":
                    ids.append(seg1[3])
                    continue
                x, y = seg1[1], seg1[2]
                for piece in pretokenize(self.pretok, chars[x:y], orig[x:y]):
                    ids.extend(tokenize(piece))
        if add_special_tokens:
            ids = self.pre + ids + self.post
        return ids

    # -- decode ---------------------------------------------------------------------------------------
    def token_string(self, tid):
        """hf simple_id_to_token, then the model vocab (doc §8.1)"""
        a = self.added.get(tid)
        if a is not None:
            return self.norm_cache.get(tid, a.content)
        return self.model.vocab_r.get(tid)

    def decode(self, ids, skip_special_tokens=False):
        toks = []
        for tid in ids:
            s = self.token_string(tid)
            if s is None:
                continue                                    # no string: skipped
            if skip_special_tokens and s in self.specials:
                continue
            toks.append(s)
        if self.decoder is None:
            return " ".join(toks)
        return "".join(decode_chain(self.decoder, toks))


def load(path):
    with open(path, "rb") as f:
        return SpmTokenizer(json.loads(f.read().decode("utf-8")))
