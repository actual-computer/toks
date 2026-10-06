"""python/tests/test_surface.py: the drop-in surface (hf's Encoding and its friends, tiktoken's Encoding) against its
oracles, through the wheel.

hf tokenizers 0.23.2, on every tokenizer.json under tests/data and in $TOKS_TOKENIZER_CACHE that both load:
  encode_ex          == Tokenizer.encode field by field: ids, type_ids, tokens, attention_mask, special_tokens_mask,
                        n_sequences, len, overflowing (hf's is [] unless it truncated the text: those are counted
                        apart, toks never computes the overflow), in every mode x add_special_tokens (NONSPECIAL =
                        encode_special_tokens, NONE = the file without its added tokens); .ids == encode's
  encode_batch(_ex)  == Tokenizer.encode_batch: the file's padding over the batch (BatchLongest: its longest member)
  num_special_tokens_to_add(False), get_added_tokens_decoder() (every field of every AddedToken)
  the tiktoken view where hf defines it: n_vocab = the highest id + 1; special_tokens_set = the contents the file
                        lists special; eot_token; encode_ordinary = NONSPECIAL without template, truncation or padding;
                        encode(allowed_special="all") = ALL without them; encode(allowed_special=set()) raises
                        ValueError where the text holds a special, else is ALL's; a partial allowed set with the check
                        off is tiktoken's split (the text between allowed specials encoded alone, ordinary);
                        decode_bytes (python/toks_oracle/primitives.py's raw reference); token_byte_values on ByteLevel
                        files (the byte table over the model's strings)
tiktoken 0.14.0 on the tiktoken models, every tiktoken name: kimik3 (tests/parity/oracle_tiktoken.py: the bare
engine; it needs transformers 5.18.0) and qwen1-72b (tests/parity/qwen1_check.py: the wrapper's own constants; the
wrapper encodes NFC(text), and so does toks).

Where toks refuses a value instead of guessing (Encoding.tokens: see refusal() for the three cases and where each is
allowed) the refusal is counted apart, never as equal; a refusal anywhere else fails. Counts go to TOKS_PY_REPORT.

    TOKS_PY_REPORT=report.json pytest python/tests/test_surface.py
"""
import json
import os
import random
import re
import sys
import unicodedata

import pytest

import toks
from conftest import ROOT, TOKENIZERS, report

tokenizers = pytest.importorskip("tokenizers")
from toks_oracle import oracle as O  # noqa: E402
from toks_oracle import primitives as P  # noqa: E402

FIELDS = ("ids", "type_ids", "tokens", "attention_mask", "special_tokens_mask")
MODES = (("ALL", "all"), ("NONSPECIAL", "nonspecial"), ("NONE", "none"))
T2 = ("The quick brown fox jumps over the lazy dog, twice: 1234567 and 3.14159!  Gr\u00fc\u00dfe aus K\u00f6ln, \u00e7a va?\n"
      "def f(x):\n\treturn x ** 2  # square\n"
      "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3001\u4e2d\u6587\u6587\u672c\u3002\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8. "
      "Emoji \U0001F600\U0001F44D\U0001F3FD and \U0001F468\u200d\U0001F469\u200d\U0001F467 families.\n"
      "    indented   runs   of   spaces\t\ttabs\r\nand CRLF, don't DON'T 'S 'll I'VE.\n")


def show(s) -> str:
    return ascii(s).replace("|", "\\x7c")[:200]


# A difference in the C core, not in the binding: hf's BPE appends the byte ids of a
# byte-fallback character before it writes the unk an earlier unknown character left pending (merge_word: the
# byte-fallback branch adds without flushing `unk`, and a later unknown character fuses into that same unk), so on
# these fixtures "\u00fce" is hf [<0x65>, <unk>] and toks [<unk>, <0x65>]. A character is unknown there only because a
# byte token is missing (no <0xC3>); no cached tokenizer lacks one a character needs. A case whose ids hold the unk
# (id 0) is counted apart on these two files when they differ, and nowhere else.
UNK_ORDER = {"tests/data/spm/unk_fused.json": 0, "tests/data/spm/unk_unfused.json": 0}


def same(name, got, want):
    if got == want:
        return True
    unk = UNK_ORDER.get(name)
    if unk is not None and (unk in got or unk in want):
        return "hf_unk_order"
    return False


# Encoding.tokens refuses (TOKS_E_UNSUPPORTED) instead of guessing in three cases, each allowed only where it holds;
# any other refusal is a mismatch:
# - a Unigram unknown piece (hf writes the normalized text it covers): a Unigram file whose hf ids hold its unk;
# - an id the model writes under one string and an added token under another, the text holding the token: the
#   fixture built for it (no tokenizer in the cache gives a model id to an added token of another string);
# - whitespace lstrip / rstrip tokens whose matches no piece separates: hf's rstrip lets the next match of a run it
#   swallowed overlap the previous one ("\t\t" is the tokens "\t\t" and "\t"), or one whitespace run could be either
#   of two whitespace tokens: the fixtures built for those.
TWO_SOURCES = {"tests/data/spm/holes_added.json"}
STRIP_OVERLAP = {"tests/data/hardening/ws_rstrip.json", "tests/data/hardening/ws_lrstrip.json",
                 "tests/data/breadth/added_opts.json"}


def refusal(name, j, ex, w):
    m = str(ex)
    if ex.code != -3:
        ok = False
    elif "Unigram model's unknown piece" in m:              # Unigram by shape: hf reads an untyped model so too
        ok = isinstance(j["model"].get("vocab"), list) and j["model"].get("unk_id") in w.ids
    elif "which the text holds" in m:
        ok = name in TWO_SOURCES
    elif "within whitespace for its" in m or "could be the match of any of" in m:
        ok = name in STRIP_OVERLAP
    else:
        ok = False
    return "refused" if ok else False


class Tally:
    def __init__(self):
        self.c, self.bad = {}, []

    def add(self, op, ok, why=""):
        d = self.c.setdefault(op, {"cases": 0, "equal": 0})
        d["cases"] += 1
        if ok is True:
            d["equal"] += 1
        elif ok:                                    # a named outcome counted apart (refused, hf_overflow, ...)
            d[ok] = d.get(ok, 0) + 1
        elif len(self.bad) < 12:
            self.bad.append(f"{op}: {why}")

    def check(self, key):
        report(key, self.c)
        assert not self.bad, f"{key}: mismatches against the oracle:\n" + "\n".join(self.bad)
        return self.c


def _files():
    out = []
    for d, _, fs in sorted(os.walk(os.path.join(ROOT, "tests", "data"))):
        out += [os.path.join(d, f) for f in sorted(fs) if f.endswith(".json")]
    if os.path.isdir(TOKENIZERS):
        out += [os.path.join(TOKENIZERS, f) for f in sorted(os.listdir(TOKENIZERS))]
    keep = []
    for p in out:
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        if isinstance(j, dict) and "model" in j:
            keep.append(p)
    return keep


FILES = _files()


def _name(p):
    return os.path.relpath(p, ROOT) if p.startswith(ROOT) else os.path.basename(p)


def texts_for(j):
    """plain text, a long one (truncation), the file's added tokens between whitespace runs (lstrip / rstrip take
    them), characters a small vocabulary lacks (unknown pieces)"""
    contents = [a["content"] for a in (j.get("added_tokens") or []) if a.get("content")]
    picks = contents[:3] + contents[-3:]
    around = "".join(f"a{c}b {c}  x\n{c}\n y" for c in picks) if picks else "no added tokens here"
    return ["", "Hello, world! It's 2026.", T2, T2 * 20, around, "a\U0001D11E\U0001D11Eb \u0fff\u0ffe z \u00e9t\u00e9"]


def _hf(fn, *a, **k):
    """an hf call, or None where hf raises or panics (pyo3 panics are BaseException)"""
    try:
        return fn(*a, **k)
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        return None


def _fresh(src, nonspecial=False):
    t = tokenizers.Tokenizer.from_str(src)
    t.no_truncation()
    t.no_padding()
    t.encode_special_tokens = nonspecial
    return t


def _raw_reference(hf, j, ids):
    """decode_bytes' reference on an hf file (primitives.py): ByteLevel decoders' bytes before their lossy step, a
    ByteFallback chain's decode with each U+FFFD of an invalid run back to its byte, else decode's utf-8"""
    kind = P.decoder_kind(j)
    if kind == "bytelevel":
        return b"".join(P.bytelevel_bytes(hf.id_to_token(i)) for i in ids if hf.id_to_token(i) is not None)
    if kind == "bytefallback":
        return P.hf_decode_raw(hf, ids, [hf.id_to_token(i) for i in ids])
    return hf.decode(ids, skip_special_tokens=False).encode("utf-8")


def _subset(view, text, allowed):
    """tiktoken's split with hf as the engine: the text between allowed specials encoded alone"""
    if not allowed:
        return view.encode(text, add_special_tokens=False).ids
    pat = re.compile("|".join(re.escape(k) for k in sorted(allowed, key=len, reverse=True)))
    out, pos = [], 0
    for m in pat.finditer(text):
        if m.start() > pos:
            out += view.encode(text[pos:m.start()], add_special_tokens=False).ids
        out.append(allowed[m.group()])
        pos = m.end()
    if pos < len(text):
        out += view.encode(text[pos:], add_special_tokens=False).ids
    return out


@pytest.mark.parametrize("path", FILES, ids=[_name(p) for p in FILES])
def test_hf_file(path):
    assert tokenizers.__version__ == "0.23.2"
    try:
        t = toks.Tokenizer.from_file(path)
    except toks.Error as e:
        pytest.skip(f"toks refuses it: {e}")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    j = json.loads(src)
    o = _hf(O.load, path)
    if o is None:
        pytest.skip("hf refuses it")
    hf = o.t
    name = _name(path)
    texts = texts_for(j)
    tl = Tally()

    # encode_ex == hf encode, field by field, every mode x add_special_tokens
    for mode, m in MODES:
        view = _hf(o.view, mode)
        if view is None:
            tl.add("encode_ex", "hf_refuses_view")
            continue
        for pp in (True, False):
            for x in texts:
                w = _hf(view.encode, x, add_special_tokens=pp)
                if w is None:
                    tl.add("encode_ex", "hf_raises")
                    continue
                e = t.encode_ex(x, add_special_tokens=pp, added_tokens=m)
                tl.add("encode_ex.ids==encode", e.ids == t.encode(x, add_special_tokens=pp, added_tokens=m),
                       f"{mode} pp={pp} {show(x)}")
                if same(name, e.ids, w.ids) == "hf_unk_order":
                    for fld in FIELDS + ("overflowing", "n_sequences+len"):
                        tl.add(fld, "hf_unk_order")
                    continue
                for fld in FIELDS:
                    try:
                        g = getattr(e, fld)
                    except toks.Error as ex:
                        tl.add(fld, refusal(name, j, ex, w), f"{mode} pp={pp} {show(x)}: {ex}")
                        continue
                    want = getattr(w, fld)
                    tl.add(fld, g == want, f"{mode} pp={pp} {show(x)}: want {show(want)} got {show(g)}")
                tl.add("overflowing", True if not w.overflowing else "hf_overflow",
                       f"{mode} pp={pp} {show(x)}") if e.overflowing == [] else tl.add("overflowing", False, show(x))
                tl.add("n_sequences+len", e.n_sequences == w.n_sequences == 1 and len(e) == len(w), show(x))

    # encode_batch / encode_batch_ex == hf encode_batch (the file's padding over the batch)
    for pp in (True, False):
        w = _hf(hf.encode_batch, texts, add_special_tokens=pp)
        if w is None:
            tl.add("encode_batch", "hf_raises")
            continue
        for g, we in zip(t.encode_batch(texts, add_special_tokens=pp), w):
            tl.add("encode_batch", same(name, g, we.ids), f"pp={pp}: want {show(we.ids)} got {show(g)}")
        unk_order = any(same(name, g.ids, we.ids) == "hf_unk_order" for g, we in
                        zip(t.encode_batch_ex(texts, add_special_tokens=pp), w))   # the batch's padding moves too
        for g, we in zip(t.encode_batch_ex(texts, add_special_tokens=pp), w):
            for fld in FIELDS:
                if unk_order:
                    tl.add("batch." + fld, "hf_unk_order")
                    continue
                try:
                    gv = getattr(g, fld)
                except toks.Error as ex:
                    tl.add("batch." + fld, refusal(name, j, ex, we), str(ex))
                    continue
                want = getattr(we, fld)
                tl.add("batch." + fld, gv == want, f"pp={pp} {fld}: want {show(want)} got {show(gv)}")

    # the template's count, the added tokens
    tl.add("num_special_tokens_to_add", t.num_special_tokens_to_add() == hf.num_special_tokens_to_add(False))
    want = [(i, a.content, a.single_word, a.lstrip, a.rstrip, a.normalized, a.special)
            for i, a in hf.get_added_tokens_decoder().items()]
    got = [(i, a.content, a.single_word, a.lstrip, a.rstrip, a.normalized, a.special)
           for i, a in t.get_added_tokens_decoder().items()]
    tl.add("get_added_tokens_decoder", sorted(want) == got, f"{len(want)} vs {len(got)}")

    # the tiktoken view, defined by hf: specials are the contents listed special, ordinary is NONSPECIAL
    vocab = hf.get_vocab(with_added_tokens=True)
    tl.add("n_vocab", t.n_vocab == max(vocab.values()) + 1 == t.max_token_value + 1, f"{t.n_vocab} vs {max(vocab.values()) + 1}")
    specials = {a["content"] for a in (j.get("added_tokens") or []) if a.get("special") and a.get("content")}
    tl.add("special_tokens_set", t.special_tokens_set == specials, f"{sorted(t.special_tokens_set ^ specials)[:5]}")
    try:
        eot = t.eot_token
    except KeyError:
        eot = None
    tl.add("eot_token", eot == (vocab["<|endoftext|>"] if "<|endoftext|>" in specials else None), str(eot))
    bare, ordinary = _hf(_fresh, src), _hf(_fresh, src, True)
    if bare is not None and ordinary is not None:
        sid = {c: vocab[c] for c in specials if c in vocab}
        for x in texts:
            w_all = _hf(bare.encode, x, add_special_tokens=False)
            w_ord = _hf(ordinary.encode, x, add_special_tokens=False)
            if w_all is None or w_ord is None:
                tl.add("tiktoken.encode", "hf_raises")
                continue
            tl.add("encode(allowed_special='all')", same(name, t.encode(x, allowed_special="all"), w_all.ids), show(x))
            tl.add("encode_ordinary", same(name, t.encode_ordinary(x), w_ord.ids), show(x))
            tl.add("encode(disallowed_special=())", same(name, t.encode(x, disallowed_special=()), w_ord.ids), show(x))
            # the specials the text holds, as hf matches them (single_word, normalized=true): ALL's ids carry them where
            # the text spells them (an unk id the model wrote is no special of the text), leftmost first
            present = sorted((s for s in specials if sid.get(s) in w_all.ids and s in x), key=lambda s: (x.find(s), -len(s)))
            try:
                g = t.encode(x, allowed_special=set())
                tl.add("encode(allowed_special=set())", not present and same(name, g, w_all.ids),
                       f"{show(x)}: {present[:2]}")
            except ValueError as ex:
                tl.add("encode(allowed_special=set())", bool(present) and repr(present[0]) in str(ex), f"{show(x)}: {ex}")
            if len(present) >= 2:                   # one allowed, the others text: tiktoken's split
                keep = {present[0]: sid[present[0]]} if present[0] in sid else {}
                w = _hf(_subset, ordinary, x, keep)
                g = t.encode(x, allowed_special=set(keep), disallowed_special=())
                tl.add("encode(partial allowed_special)", w is not None and same(name, g, w), show(x))
            raw = _hf(_raw_reference, hf, j, w_all.ids)
            if raw is not None:
                tl.add("decode_bytes", t.decode_bytes(w_all.ids) == raw, show(x))
        if P.decoder_kind(j) == "bytelevel":
            added = set(hf.get_added_tokens_decoder())
            want = sorted(P.bytelevel_bytes(s) for s, i in vocab.items() if i not in added)
            tl.add("token_byte_values", t.token_byte_values() == want, f"{len(want)}")
    c = tl.check(f"surface/{name}")
    assert c["ids"]["equal"] > 0


# ---- tiktoken 0.14.0 on the tiktoken models -----------------------------------------------------------------------

def _tiktoken_checks(t, enc, prep, name, rnd):
    tl = Tally()
    tl.add("n_vocab", t.n_vocab == enc.n_vocab and t.max_token_value == enc.max_token_value)
    tl.add("special_tokens_set", t.special_tokens_set == enc.special_tokens_set)
    try:
        want = enc.eot_token
    except KeyError:
        want = None
    try:
        got = t.eot_token
    except KeyError:
        got = None
    tl.add("eot_token", got == want, f"{got} vs {want}")
    tl.add("token_byte_values", t.token_byte_values() == enc.token_byte_values())
    sp = sorted(enc.special_tokens_set)
    pool = [rnd.choice(sp) for _ in range(12)] + ["<|", "|>", "[", "]", "\u00e9", "e\u0301", "A\u030a", " ", "  ", "\n",
                                                  "hello", " world", "\u4e2d\u6587", "\U0001F600", "1234567", "'s"]
    texts = ["", "Hello, world!", T2] + ["".join(rnd.choice(pool) for _ in range(rnd.randint(1, 16))) for _ in range(300)]
    for x in texts:
        xr = prep(x)
        present = [s for s in sp if s in xr]
        for kw in ({"allowed_special": "all"}, {"disallowed_special": ()}, {"allowed_special": set()},
                   {"allowed_special": set(present[:1])}, {"allowed_special": set(present[:1]), "disallowed_special": ()},
                   {"allowed_special": "all", "disallowed_special": set(present[:1])}):
            try:
                w = enc.encode(xr, **kw)
            except ValueError as ex:
                w = ("ValueError", str(ex))
            try:
                g = t.encode(x, **kw)
            except ValueError as ex:
                g = ("ValueError", str(ex))
            tl.add(f"encode({','.join(sorted(kw))})", g == w, f"{show(x)} {kw}: {show(w)} vs {show(g)}")
        tl.add("encode_ordinary", t.encode_ordinary(x) == enc.encode_ordinary(xr), show(x))
        ids = enc.encode(xr, allowed_special="all")
        tl.add("decode_bytes", t.decode_bytes(ids) == enc.decode_bytes(ids), show(x))
    for _ in range(300):
        ids = [rnd.randrange(enc.n_vocab + 3) for _ in range(rnd.randint(0, 12))]
        try:
            w = enc.decode_bytes(ids)
        except KeyError as ex:
            w = ("KeyError", str(ex))
        try:
            g = t.decode_bytes(ids)
        except KeyError as ex:
            g = ("KeyError", str(ex))
        tl.add("decode_bytes(random)", g == w, f"{ids}: {w} vs {g}")
    batch = [enc.encode(x, allowed_special="all") for x in texts[:20]]
    tl.add("decode_bytes_batch", t.decode_bytes_batch(batch) == enc.decode_bytes_batch(batch))
    return tl.check(f"surface/{name}")


def test_tiktoken_kimik3():
    pytest.importorskip("tiktoken")
    pytest.importorskip("transformers")
    d = os.path.expanduser(os.environ.get("TOKS_KIMI_DIR", "~/.cache/toks/kimik3"))
    if not os.path.isdir(d):
        pytest.skip(f"{d} is missing (tests/parity/oracle_tiktoken.py --fetch)")
    sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))
    import oracle_tiktoken as OT
    k = OT.Kimi(d)
    # short texts: the bare engine and Kimi's own tokenizer (toks) part only at its 25,000-character chunks
    _tiktoken_checks(toks.Tokenizer.from_file(d), k.enc, lambda x: x, "kimik3", random.Random(3))


def test_tiktoken_qwen1(tmp_path):
    pytest.importorskip("tiktoken")
    for src, dst in (("qwen1-72b.tiktoken", "qwen.tiktoken"), ("qwen1-72b_tokenization_qwen.py", "tokenization_qwen.py"),
                     ("qwen1-72b_tokenizer_config.json", "tokenizer_config.json")):
        p = os.path.join(TOKENIZERS, src)
        if not os.path.isfile(p):
            pytest.skip(f"{p} is missing (tools/corpora/fetch_tokenizers.py --targets)")
        os.symlink(p, tmp_path / dst)
    sys.path.insert(0, os.path.join(ROOT, "tests", "parity"))
    import qwen1_check
    enc, _ = qwen1_check.reference(str(tmp_path))
    _tiktoken_checks(toks.Tokenizer.from_file(tmp_path), enc, lambda x: unicodedata.normalize("NFC", x), "qwen1-72b",
                     random.Random(4))
