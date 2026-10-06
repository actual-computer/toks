"""python/tests/test_api.py: the toks package's api, error paths and memory behaviour (no hf needed).

gpt2's pinned file is the subject: ids below are hf 0.23.2's (checked by test_parity.py too).
"""
import array
import ctypes
import gc
import glob
import hashlib
import json
import mmap
import os
import pathlib
import pickle
import platform
import shutil
import sys
import tracemalloc

import pytest

import toks

E = {"OPEN": -1, "FORMAT": -2, "UNSUPPORTED": -3, "TIER": -5, "ID": -7, "LIMIT": -9}
EOT = "<|endoftext|>"                       # gpt2's one added token, id 50256 (special)
HELLO = [15496, 995]                        # "Hello world"


def raises(code, fn, *a, **k):
    with pytest.raises(toks.Error) as e:
        fn(*a, **k)
    assert e.value.code == E[code] and e.value.name == "TOKS_E_" + code, str(e.value)
    assert str(e.value).startswith("TOKS_E_" + code + ": ")
    return e.value


def test_version():
    import importlib.metadata
    import re

    with open(os.path.join(os.path.dirname(__file__), "..", "..", "include", "toks.h")) as f:
        want = re.search(r'#define TOKS_VERSION +"(.*)"', f.read()).group(1)
    assert toks.__version__ == want == importlib.metadata.version("toks")      # library == header == wheel


# ---- loading --------------------------------------------------------------------------------------------------

def test_load_forms(gpt2_path, tmp_path):
    data = pathlib.Path(gpt2_path).read_bytes()
    d = tmp_path / "model"
    d.mkdir()
    shutil.copy(gpt2_path, d / "tokenizer.json")
    loads = [toks.Tokenizer.from_file(gpt2_path), toks.Tokenizer.from_file(pathlib.Path(gpt2_path)),
             toks.Tokenizer.from_file(os.fsencode(gpt2_path)), toks.Tokenizer.from_file(d),
             toks.Tokenizer.from_str(data.decode()), toks.Tokenizer.from_buffer(data),
             toks.Tokenizer.from_buffer(bytearray(data)), toks.Tokenizer.from_buffer(memoryview(data))]
    sha = hashlib.sha256(data).hexdigest()
    for t in loads:
        assert t.encode("Hello world") == HELLO
        i = t.info()
        assert i["source_sha256"] == sha and i["n_ids"] == 50257 and i["algorithm"] == "bpe_bytelevel"
        assert i["abi"] == toks.ABI and i["max_text"] == toks.MAX_TEXT == 1 << 29
        assert i["tier"] in ("scalar", "neon", "avx2", "avx512") and i["image_sha256"] is None
    assert loads[0].info()["name"] == "gpt2" and loads[3].info()["name"] == "model"
    assert "gpt2" in repr(loads[0]) and "50257 ids" in repr(loads[0])
    assert loads[0].source == gpt2_path and loads[5].source is data


def test_load_errors(tmp_path, gpt2_path):
    e = raises("OPEN", toks.Tokenizer.from_file, tmp_path / "missing.json")
    assert "missing.json" in str(e)
    raises("OPEN", toks.Tokenizer.from_file, tmp_path)                      # a directory without tokenizer.json
    (tmp_path / "bad.json").write_text("{")
    raises("FORMAT", toks.Tokenizer.from_file, tmp_path / "bad.json")
    raises("FORMAT", toks.Tokenizer.from_str, '{"model": 1}')
    wl = {"version": "1.0", "added_tokens": [], "normalizer": None, "pre_tokenizer": None, "post_processor": None,
          "decoder": None, "truncation": None, "padding": None,
          "model": {"type": "WordLevel", "vocab": {"a": 0, "[UNK]": 1}, "unk_token": "[UNK]"}}
    raises("UNSUPPORTED", toks.Tokenizer.from_str, json.dumps(wl))
    with pytest.raises(ValueError):
        toks.Tokenizer.from_file(gpt2_path, tier="sse2")
    with pytest.raises(ValueError):
        toks.Tokenizer.from_file("a\0b")
    with pytest.raises(TypeError):
        toks.Tokenizer.from_file(42)
    with pytest.raises(TypeError):
        toks.Tokenizer.from_str(b"{}")
    with pytest.raises(TypeError):
        toks.Tokenizer()
    other = "avx2" if platform.machine().lower() in ("arm64", "aarch64") else "neon"
    raises("TIER", toks.Tokenizer.from_file, gpt2_path, tier=other)


def test_tiers_agree(gpt2_path):
    auto = toks.Tokenizer.from_file(gpt2_path)
    scalar = toks.Tokenizer.from_file(gpt2_path, tier="scalar")
    assert scalar.info()["tier"] == "scalar"
    text = "The quick brown fox \u00e9\u00e8 \u65e5\u672c <|endoftext|> 1234567 " * 300
    assert auto.encode(text) == scalar.encode(text) and auto.pieces(text) == scalar.pieces(text)


def test_cache_mib(gpt2_path):
    auto = toks.Tokenizer.from_file(gpt2_path)
    big = toks.Tokenizer.from_file(gpt2_path, cache_mib=8)
    text = "The quick brown fox \u00e9\u00e8\u00e9\u00e8 \u65e5\u672c\u8a9e   indentation_is_long_here 1234567 " * 400
    want = auto.encode(text)
    assert big.encode(text) == want and big.encode(text) == want   # the second call is warm: the long cache answers
    for bad in (1, 3, 129, 256, -4):
        with pytest.raises(ValueError):
            toks.Tokenizer.from_file(gpt2_path, cache_mib=bad)


# ---- encode ---------------------------------------------------------------------------------------------------

def test_encode_inputs(gpt2):
    s = "Hello world \u00e9 \U0001F600"
    want = gpt2.encode(s)
    b = s.encode()
    for x in (b, bytearray(b), memoryview(b), array.array("B", b)):
        assert gpt2.encode(x) == want
    assert gpt2.encode("") == [] and gpt2.encode(b"") == []
    raw = b"\xff\xfe abc \xc3"                       # byte input (SPEC 3.3): byte-level bpe keeps every byte
    assert gpt2.decode(gpt2.encode(raw)) == raw.decode("utf-8", "replace") and gpt2.pieces(raw)[-1] == len(raw)
    with pytest.raises(UnicodeEncodeError):
        gpt2.encode("\ud800")
    for bad in (None, 3, ["a"], 1.5):
        with pytest.raises(TypeError):
            gpt2.encode(bad)


def test_encode_arguments(gpt2):
    assert gpt2.encode(text="Hello world") == HELLO
    with pytest.raises(TypeError):
        gpt2.encode("x", True)                     # add_special_tokens is keyword-only (hf's 2nd is `pair`)
    with pytest.raises(TypeError):
        gpt2.encode("x", nope=1)
    with pytest.raises(TypeError):
        gpt2.encode("x", text="y")
    with pytest.raises(TypeError):
        gpt2.encode()
    with pytest.raises(ValueError):
        gpt2.encode("x", added_tokens="special")
    with pytest.raises(ValueError):
        gpt2.encode("x", added_tokens=1)


def test_modes(gpt2):
    s = "a" + EOT + "b"
    assert gpt2.encode(s) == [64, 50256, 65]
    assert gpt2.encode(s, added_tokens="all") == [64, 50256, 65]
    plain = gpt2.encode(s, added_tokens="none")
    assert 50256 not in plain and gpt2.encode(s, added_tokens="nonspecial") == plain
    assert gpt2.encode(s, add_special_tokens=False) == [64, 50256, 65]          # gpt2 has no post-processor ids
    assert gpt2.pieces(s) == [1, 14, 15] and gpt2.pieces(s, added_tokens="none")[-1] == 15
    assert gpt2.encode_special_tokens is False
    gpt2.encode_special_tokens = True               # hf's property: the default mode becomes NONSPECIAL
    try:
        assert gpt2.encode(s) == plain and gpt2.encode(s, added_tokens="all") == [64, 50256, 65]
        assert gpt2.encode_batch([s])[0] == plain
    finally:
        gpt2.encode_special_tokens = False
    assert gpt2.encode(s, continuation=True) == gpt2.encode(s)


def test_limit(gpt2):
    m = mmap.mmap(-1, toks.MAX_TEXT + 1)            # untouched pages: nothing is read or allocated
    try:
        mv = memoryview(m)
        raises("LIMIT", gpt2.encode, mv)
        raises("LIMIT", gpt2.pieces, mv)
        raises("LIMIT", gpt2.encode_into, mv, array.array("I", [0]))
        raises("LIMIT", gpt2.encode_batch, ["a", mv])
        mv.release()
    finally:
        m.close()


def test_encode_into(gpt2):
    s = "Hello world, this is a test of encode_into " * 20
    want = gpt2.encode(s)
    out = array.array("I", bytes(4 * (len(want) + 5)))
    assert gpt2.encode_into(s, out) == len(want) and out[:len(want)].tolist() == want
    short = array.array("i", [0] * 7)               # int32 works; a short buffer gets the exact prefix
    assert gpt2.encode_into(s, short) == len(want) and short.tolist() == want[:7]
    assert gpt2.encode_into(s, array.array("I")) == len(want)
    mv = memoryview(bytearray(4 * len(want))).cast("I")
    assert gpt2.encode_into(s, mv) == len(want) and mv.tolist() == want
    for bad in (array.array("q", [0] * 10), array.array("H", [0] * 10), bytearray(40), bytes(40), "x"):
        with pytest.raises((TypeError, BufferError)):
            gpt2.encode_into(s, bad)
    buf = bytearray(b"Hello world" + bytes(64))
    with pytest.raises(ValueError):                 # out must not overlap the text
        gpt2.encode_into(memoryview(buf)[:11], memoryview(buf).cast("B")[8:72].cast("I"))


def test_encode_batch(gpt2):
    texts = ["Hello world", "", "a" + EOT + "b", b"bytes \xff", "x" * 5000, "\u00e9" * 3000]
    want = [gpt2.encode(x) for x in texts]
    assert gpt2.encode_batch(texts) == want
    assert gpt2.encode_batch(tuple(texts)) == want
    assert gpt2.encode_batch(iter(texts)) == want
    assert gpt2.encode_batch(texts, add_special_tokens=False, added_tokens="none") == \
        [gpt2.encode(x, add_special_tokens=False, added_tokens="none") for x in texts]
    assert gpt2.encode_batch([]) == []
    with pytest.raises(TypeError, match="text 1"):
        gpt2.encode_batch(["ok", 5])
    with pytest.raises(TypeError):
        gpt2.encode_batch("one text")
    with pytest.raises(TypeError):
        gpt2.encode_batch(["a"], True)


# ---- decode ---------------------------------------------------------------------------------------------------

def test_decode(gpt2):
    assert gpt2.decode(HELLO) == "Hello world"
    assert gpt2.decode([50256]) == "" and gpt2.decode([50256], False) == EOT         # hf: skip by default
    assert gpt2.decode([50256], skip_special_tokens=False) == EOT
    for ids in ((15496, 995), array.array("I", HELLO), array.array("q", HELLO), array.array("i", HELLO),
                iter(HELLO), memoryview(array.array("I", HELLO)), [True, 995]):
        assert gpt2.decode(ids) in ("Hello world", '"' + " world")
    assert gpt2.decode(array.array("B", [72, 72])) == "ii"
    assert gpt2.decode([]) == ""
    assert gpt2.decode([31373] * 3000) == "hello" * 3000
    e = raises("ID", gpt2.decode, [15496, 50257])
    assert "index 1" in str(e) and "50257" in str(e)
    raises("ID", gpt2.decode, [2**32 - 1])
    for bad in ([-1], [2**32], array.array("q", [-5]), [2**70]):
        with pytest.raises(OverflowError):
            gpt2.decode(bad)
    for bad in (["a"], [1.0], "text", b"ab", 5, [None]):
        with pytest.raises(TypeError):
            gpt2.decode(bad)
    assert gpt2.decode([gpt2.token_to_id("\u00c3")]) == "\ufffd"             # byte c3 alone: U+FFFD
    assert gpt2.decode_batch([HELLO, [50256]]) == ["Hello world", ""]
    assert gpt2.decode_batch([HELLO, [50256]], skip_special_tokens=False) == ["Hello world", EOT]


def test_token_lookups(gpt2):
    assert gpt2.token_bytes(15496) == b"Hello" and gpt2.token_bytes(995) == b" world"
    assert gpt2.token_bytes(50256) == EOT.encode() and gpt2.token_bytes(50257) is None
    with pytest.raises(OverflowError):
        gpt2.token_bytes(-1)
    assert gpt2.token_to_id("Hello") == 15496 and gpt2.token_to_id("\u0120world") == 995
    assert gpt2.token_to_id(EOT) == 50256 and gpt2.token_to_id("nope nope") is None
    assert gpt2.id_to_token(995) == "\u0120world" and gpt2.id_to_token(50256) == EOT
    assert gpt2.id_to_token(50257) is None and gpt2.id_to_token(2**32 - 1) is None
    with pytest.raises(OverflowError):
        gpt2.id_to_token(-1)
    with pytest.raises(TypeError):
        gpt2.token_to_id(5)
    assert gpt2.get_vocab_size() == gpt2.get_vocab_size(False) == 50257 and len(gpt2.get_vocab()) == 50257


def test_vocab_source_changed(gpt2_path, tmp_path):
    p = tmp_path / "tok.json"
    shutil.copy(gpt2_path, p)
    t = toks.Tokenizer.from_file(p)
    blob = pickle.dumps(t)
    p.write_bytes(p.read_bytes() + b" ")
    raises("FORMAT", t.token_to_id, "Hello")         # the file is no longer what was loaded
    raises("FORMAT", pickle.loads, blob)


def test_decode_stream(gpt2):
    st = gpt2.decode_stream()
    assert "gpt2" in repr(st)
    try:
        parts = [st.push(15496), st.push([995, 50256]), st.flush()]
    except toks.Error as e:
        assert e.code == E["UNSUPPORTED"]           # the library build has no stream decode yet
        pytest.skip("toks_stream_* returns TOKS_E_UNSUPPORTED")
    assert "".join(parts) == gpt2.decode([15496, 995, 50256], skip_special_tokens=False)
    ids = gpt2.encode("caf\u00e9 \U0001F600")
    st = gpt2.decode_stream()
    assert "".join([st.push(i) for i in ids] + [st.flush()]) == "caf\u00e9 \U0001F600"
    st = gpt2.decode_stream(skip_special_tokens=True)
    assert st.push([50256, 15496]) + st.flush() == "Hello"
    raises("ID", gpt2.decode_stream().push, [50257])
    with pytest.raises(TypeError):
        st.push("x")


def test_lookups(gpt2):
    """token_to_id's two forms and id_flags (toks.h toks_token_to_id / toks_id_flags); the ids are hf 0.23.2's"""
    assert gpt2.token_to_id(b" hello") == 23748 and gpt2.token_to_id("\u0120hello") == 23748   # bytes / hf's form
    assert gpt2.token_to_id(b"hello") == 31373 and gpt2.token_to_id(b"<|endoftext|>") == 50256 == gpt2.token_to_id(EOT)
    assert gpt2.token_to_id("\u0120hello".encode()) is None       # bytes are what an id decodes to, not hf's string
    assert gpt2.token_to_id(bytearray(b"!")) == 0 and gpt2.token_to_id(memoryview(b"!")) == 0
    assert gpt2.token_to_id(b"") is None and gpt2.token_to_id(b"\xff\xfe\xfd no such token") is None
    assert gpt2.id_flags(50256) == toks.ID_ADDED | toks.ID_SPECIAL and gpt2.id_flags(31373) == 0
    assert gpt2.id_flags(0) == toks.ID_BYTE                    # '!': a byte-level one-byte token
    # the two forms can name different ids: hf's '\u00a2' spells the byte A2 (id 95); the bytes C2 A2 are another
    # token, the one hf spells '\u00c2\u00a2' (id 44359)
    assert gpt2.token_to_id("\u00a2") == 95 == gpt2.token_to_id(b"\xa2")
    assert gpt2.token_to_id("\u00a2".encode()) == 44359 == gpt2.token_to_id("\u00c2\u00a2")
    assert (toks.ID_ADDED, toks.ID_SPECIAL, toks.ID_BYTE) == (1, 2, 4)
    raises("ID", gpt2.id_flags, 50257)
    with pytest.raises(TypeError):
        gpt2.token_to_id(5)
    with pytest.raises(OverflowError):
        gpt2.id_flags(-1)


def test_encode_bound(gpt2):
    """toks_encode_bound (toks.h "capacity"): ceil(r n) + g. gpt2 is byte-level with no normalizer and no template,
    so r = 1, g = 0 (tests/c/test_bound.c pins the same terms on its gpt2style fixture): the bound is n itself"""
    for n in (0, 1, 7, 4096, toks.MAX_TEXT, toks.MAX_TEXT + 1, 2**64 - 1):
        assert gpt2.encode_bound(n) == n
    for text in ("", "Hello world", EOT * 3, " " * 1000, "\u00e9t\u00e9 \U0001F600" * 40, "a\nb" * 500):
        n = len(text.encode())
        ids = gpt2.encode(text)
        assert len(ids) <= gpt2.encode_bound(n)
        out = array.array("I", bytes(4 * gpt2.encode_bound(n)))   # exactly the bound: every id fits
        m = gpt2.encode_into(text, out)
        assert m == len(ids) and list(out[:m]) == ids
    assert gpt2.encode_bound(True) == 1                 # an index, as len() takes
    with pytest.raises(OverflowError):
        gpt2.encode_bound(-1)
    with pytest.raises(OverflowError):
        gpt2.encode_bound(2**64)
    with pytest.raises(TypeError):
        gpt2.encode_bound("12")
    with pytest.raises(TypeError):
        gpt2.encode_bound(12.0)


def _libtoks():
    """the C library of this checkout (make lib), or $TOKS_LIB: the binding is checked against it where present"""
    suffix = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")   # this platform's, never a foreign one
    paths = [os.environ["TOKS_LIB"]] if os.environ.get("TOKS_LIB") else sorted(
        glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "build", "*", "libtoks" + suffix)))
    if not paths:
        pytest.skip("no shared libtoks (make lib, or TOKS_LIB=<path>)")
    lib = ctypes.CDLL(paths[0])
    lib.toks_load.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p]
    lib.toks_load.restype = ctypes.c_int64
    lib.toks_encode_bound.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.toks_encode_bound.restype = ctypes.c_uint64
    lib.toks_unload.argtypes = [ctypes.c_void_p]
    lib.toks_unload.restype = None
    return lib


def test_encode_bound_vs_c():
    """Tokenizer.encode_bound(n) == toks_encode_bound(ctx, n) of the C library on the same file, for r 1 (gpt2,
    llama3: a template id), 3 (qwen38: NFC byte-level), 11 (dg-exaone35: NFKC), 18/3 and 198/3 (unigram charsmaps), at
    the lengths where ceil(r n) rounds, the limit, and the saturation at 2^64 - 1; and on each file encode stays within
    the bound on texts that make many ids a byte (decomposing and compatibility characters, spaces, digits)"""
    tokenizers = os.path.expanduser(os.environ.get("TOKS_TOKENIZER_CACHE", "~/.cache/toks/tokenizers"))
    lib = _libtoks()
    seen = 0
    for name in ("gpt2", "llama3", "qwen38", "dg-exaone35", "uni_t5base", "uni_albert", "gemma4"):
        path = os.path.join(tokenizers, name)
        if not os.path.isfile(path):
            continue
        tok = toks.Tokenizer.from_file(path)
        ctx = ctypes.c_void_p()
        assert lib.toks_load(ctypes.byref(ctx), path.encode(), None) == 0, name
        try:
            for n in list(range(0, 50)) + [255, 256, 4095, 4096, 4097, 10**6, toks.MAX_TEXT, toks.MAX_TEXT + 1,
                                           2**62, 2**63, 2**64 // 3, 2**64 - 2, 2**64 - 1]:
                assert tok.encode_bound(n) == lib.toks_encode_bound(ctx, n), (name, n)
            for text in ("\ufdfa" * 64, "\U0001d160" * 64, "\u00e9\u0301 \u2460\u2474" * 50, " " * 777 + "x", "1 2 3 " * 300):
                n = len(text.encode())
                ids = tok.encode(text)
                assert len(ids) <= tok.encode_bound(n), (name, text[:8], len(ids))
                out = array.array("I", bytes(4 * tok.encode_bound(n)))
                m = tok.encode_into(text, out)
                assert m == len(ids) and list(out[:m]) == ids, (name, text[:8])
        finally:
            lib.toks_unload(ctx)
        seen += 1
    if seen == 0:
        pytest.skip(f"none of the tokenizer files under {tokenizers}")


def test_lookups_llama2():
    p = os.path.join(os.path.expanduser(os.environ.get("TOKS_TOKENIZER_CACHE", "~/.cache/toks/tokenizers")), "llama2")
    if not os.path.isfile(p):
        pytest.skip(f"{p} is missing (tools/corpora/fetch_tokenizers.py)")
    t = toks.Tokenizer.from_file(p)
    # an added token's content either way; its decoded string (llama2 normalizes <s>: "\u2581<s>") as bytes
    assert t.token_to_id("<s>") == 1 == t.token_to_id(b"<s>") == t.token_to_id("\u2581<s>".encode())
    assert t.token_to_id("</s>") == 2 and t.token_to_id(b"<unk>") == 0
    assert t.token_to_id("\u2581hello") == 22172 == t.token_to_id("\u2581hello".encode())   # a piece as written
    assert t.token_to_id(b"<0x41>") == 68 and t.token_to_id(b"<0x00>") == 3
    assert t.id_flags(1) == toks.ID_ADDED | toks.ID_SPECIAL and t.id_flags(0) == toks.ID_ADDED | toks.ID_SPECIAL
    assert t.id_flags(68) == toks.ID_BYTE and t.id_flags(22172) == 0


def test_decode_stream_hold():
    """a byte-fallback run past the stream's own 44 bytes: the binding grows the hold (toks.h toks_stream_hold), so
    300 x U+13000 (1200 <0xHH> ids, one valid run) streams one id at a time to what decode gives"""
    p = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "spm", "llamalike.json")
    t = toks.Tokenizer.from_file(p)
    ids = t.encode("\U00013000" * 300, add_special_tokens=False)
    assert len(ids) >= 1200
    want = t.decode(ids, skip_special_tokens=False)
    st = t.decode_stream()
    try:
        parts = [st.push(i) for i in ids]
    except toks.Error as e:
        assert e.code == E["UNSUPPORTED"]
        pytest.skip("toks_stream_* returns TOKS_E_UNSUPPORTED")
    assert all(x == "" for x in parts[-1200:])      # the run is held whole until its end
    assert "".join(parts) + st.flush() == want
    st = t.decode_stream()
    assert st.push(ids[:700]) + st.push(ids[700:]) + st.flush() == want


# ---- hf's Encoding, the template, the added tokens (abi 0.4: toks_template, toks_added, toks_info) -------------

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data")


def test_encode_ex(gpt2):
    e = gpt2.encode_ex("Hello world")
    assert isinstance(e, toks.Encoding) and len(e) == 2 and repr(e).startswith("Encoding(num_tokens=2,")
    assert e.ids == HELLO and e.type_ids == [0, 0] and e.attention_mask == [1, 1] and e.special_tokens_mask == [0, 0]
    assert e.tokens == ["Hello", "\u0120world"] and e.overflowing == [] and e.n_sequences == 1
    assert e.ids is not e.ids                       # hf's getters: a new list on every access
    e.ids.append(7)
    e.tokens.append("x")
    assert e.ids == HELLO and e.tokens == ["Hello", "\u0120world"]
    s = "a" + EOT + "b"                             # a special token in the text is text: the mask marks the template
    e = gpt2.encode_ex(s)
    assert e.ids == [64, 50256, 65] and e.tokens == ["a", EOT, "b"] and e.special_tokens_mask == [0, 0, 0]
    for kw in ({"added_tokens": "none"}, {"added_tokens": "nonspecial"}, {"add_special_tokens": False},
               {"continuation": True}):
        assert gpt2.encode_ex(s, **kw).ids == gpt2.encode(s, **kw)
    assert gpt2.encode_ex("").ids == [] and gpt2.encode_ex(b"bytes \xff").ids == gpt2.encode(b"bytes \xff")
    buf = bytearray(b"Hello world")                 # a mutable buffer is copied: tokens read the text later
    e = gpt2.encode_ex(buf)
    buf[:] = b"xxxxx xxxxx"
    assert e.tokens == ["Hello", "\u0120world"]
    with pytest.raises(TypeError):
        gpt2.encode_ex("x", True)                   # keyword-only, as encode
    with pytest.raises(TypeError):
        gpt2.encode_ex("x", allowed_special="all")


def test_encode_ex_layout():
    """the template's type ids around the text's, Left padding to a multiple, pad_type_id: hf 0.23.2's fields on
    tests/data/primitives/types_left_pad.json ([CLS] t2 / $A t1 / [SEP] t3, Fixed 13 rounded to 16, truncation 8)"""
    t = toks.Tokenizer.from_file(os.path.join(DATA, "primitives", "types_left_pad.json"))
    e = t.encode_ex("hello world")
    assert e.ids == [0] * 12 + [2, 82, 83, 3] == t.encode("hello world")
    assert e.type_ids == [5] * 12 + [2, 1, 1, 3] and e.attention_mask == [0] * 12 + [1] * 4
    assert e.special_tokens_mask == [1] * 13 + [0, 0, 1]
    assert e.tokens == ["[PAD]"] * 12 + ["[CLS]", "hello", "world", "[SEP]"]
    e = t.encode_ex("hello world the dog fox. hello world the dog fox.")       # truncated to 8 with the template
    assert e.ids == [0] * 8 + [2, 82, 83, 84, 86, 85, 77, 3] and e.type_ids == [5] * 8 + [2] + [1] * 6 + [3]
    e = t.encode_ex("hello world", add_special_tokens=False)                  # $A's type id without the template
    assert e.ids == [0] * 14 + [82, 83] and e.type_ids == [5] * 14 + [1, 1] and e.special_tokens_mask == [1] * 14 + [0, 0]
    assert t.num_special_tokens_to_add() == t.num_special_tokens_to_add(False) == 2
    i = t.info()
    assert i["truncation"] == {"max_length": 8, "stride": 0} and i["template"] == {"prefix": 1, "suffix": 1, "seq_type_id": 1}
    assert i["padding"] == {"strategy": "fixed", "length": 13, "pad_to_multiple_of": 4, "pad_id": 0, "pad_type_id": 5,
                            "direction": "left"}
    raises("UNSUPPORTED", t.num_special_tokens_to_add, True)


def test_tokens_template_readings():
    """Encoding.tokens takes the template's strings from the readings of the post_processor that give the template's
    ids. hf takes the first variant of its untagged enum that accepts the object, so a TemplateProcessing that also
    carries cls and sep is read as Roberta (hf 0.23.2: '<c>', 'hello', 'world', '<e>'); there two readings give the
    same ids under other strings, and toks refuses rather than guess."""
    with open(os.path.join(DATA, "primitives", "types_left_pad.json"), encoding="utf-8") as f:
        j = dict(json.load(f), padding=None, truncation=None)
    assert toks.Tokenizer.from_str(json.dumps(j)).encode_ex("hello world").tokens == ["[CLS]", "hello", "world", "[SEP]"]
    j["post_processor"] = dict(j["post_processor"], cls=["<c>", 2], sep=["<e>", 3])
    e = toks.Tokenizer.from_str(json.dumps(j)).encode_ex("hello world")
    assert e.ids == [2, 82, 83, 3]
    raises("UNSUPPORTED", getattr, e, "tokens")


def test_encode_batch_longest():
    """BatchLongest pads a batch to its longest member (hf's encode_batch), never one text: hf 0.23.2's ids on
    tests/data/breadth/roberta.json (right, pad 1) and trunc_left.json (left, pad 295, a multiple of 4)"""
    t = toks.Tokenizer.from_file(os.path.join(DATA, "breadth", "roberta.json"))
    want = [[295, 259, 297] + [1] * 10, [295, 116, 256, 32, 100, 111, 103, 32, 102, 111, 120, 261, 297]]
    assert t.encode_batch(["hello", "the dog fox hello"]) == want
    b = t.encode_batch_ex(["hello", "the dog fox hello"])
    assert [e.ids for e in b] == want and b[0].attention_mask == [1, 1, 1] + [0] * 10
    assert b[0].special_tokens_mask == [1, 0, 1] + [1] * 10 and b[0].tokens == ["<s>", "hello", "</s>"] + ["<pad>"] * 10
    assert t.encode("hello") == [295, 259, 297] and t.encode_batch(["hello"]) == [[295, 259, 297]]
    assert t.encode_batch(["hello", "the dog fox hello"], continuation=True) == \
        [t.encode(x, continuation=True) for x in ("hello", "the dog fox hello")]      # a part: no padding
    t = toks.Tokenizer.from_file(os.path.join(DATA, "breadth", "trunc_left.json"))
    want = [[295] * 7 + [259], [295, 295, 295, 116, 256, 32, 100, 111]]
    assert t.encode_batch(["hello", "the dog fox hello"]) == want == [e.ids for e in t.encode_batch_ex(["hello", "the dog fox hello"])]
    assert t.encode_batch([]) == [] == t.encode_batch_ex([])


def test_added_tokens_decoder(gpt2):
    d = gpt2.get_added_tokens_decoder()
    assert list(d) == [50256] and d[50256] == toks.AddedToken(EOT, normalized=True, special=True)
    assert repr(d[50256]) == 'AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True)'
    assert str(d[50256]) == EOT and pickle.loads(pickle.dumps(d[50256])) == d[50256]
    a = toks.AddedToken("x")
    assert (a.normalized, a.special) == (True, False) and toks.AddedToken("x", special=True).normalized is False
    assert a != toks.AddedToken("x", lstrip=True) and hash(a) == hash(toks.AddedToken("x")) and a != "x"
    assert gpt2.num_special_tokens_to_add() == 0 and gpt2.num_special_tokens_to_add(is_pair=False) == 0
    t = toks.Tokenizer.from_file(os.path.join(DATA, "breadth", "added_opts.json"))
    flags = {i: (a.lstrip, a.rstrip, a.single_word, a.normalized, a.special) for i, a in t.get_added_tokens_decoder().items()}
    assert len(flags) == t.info()["n_added"] and any(f[0] for f in flags.values()) and any(f[1] for f in flags.values())


# ---- the tiktoken view (tiktoken 0.14.0's Encoding) -------------------------------------------------------------

def test_tiktoken_view(gpt2):
    assert (gpt2.n_vocab, gpt2.max_token_value, gpt2.eot_token, gpt2.name) == (50257, 50256, 50256, "gpt2")
    assert gpt2.special_tokens_set == {EOT} and gpt2.special_tokens_set is not gpt2.special_tokens_set
    s = "a" + EOT + "b"
    plain = gpt2.encode(s, added_tokens="none", add_special_tokens=False)
    assert gpt2.encode(s, allowed_special="all") == [64, 50256, 65]
    assert gpt2.encode(s, disallowed_special=()) == gpt2.encode_ordinary(s) == plain
    assert gpt2.encode(s, allowed_special={EOT}) == [64, 50256, 65]
    assert gpt2.encode(s, allowed_special=set(), disallowed_special=()) == plain
    with pytest.raises(ValueError, match="disallowed special token '<\\|endoftext\\|>'"):
        gpt2.encode(s, allowed_special=set())
    with pytest.raises(ValueError):
        gpt2.encode(s, disallowed_special="all")
    assert gpt2.encode("Hello world", allowed_special=set()) == HELLO
    with pytest.raises(ValueError, match="'world'"):                     # tiktoken looks for any disallowed string
        gpt2.encode("Hello world", disallowed_special={"world"})
    with pytest.raises(TypeError):
        gpt2.encode(s, allowed_special="all", add_special_tokens=False)
    # tiktoken's surrogate fix-up: a lone one becomes U+FFFD, a pair its character
    assert gpt2.encode_ordinary("a\ud800b") == gpt2.encode_ordinary("a\ufffdb")
    assert gpt2.encode("x\ud83d\ude00", allowed_special="all") == gpt2.encode("x\U0001F600")
    with pytest.raises(UnicodeEncodeError):
        gpt2.encode("a\ud800b")                                           # hf's encode keeps its error


def test_decode_bytes(gpt2):
    text = "caf\u00e9 \U0001F600 " + EOT
    ids = gpt2.encode(text)
    assert gpt2.decode_bytes(ids) == text.encode() and gpt2.decode_bytes([50256]) == EOT.encode()
    c3 = gpt2.token_to_id("\u00c3")                                       # the byte C3 alone: raw, where decode repairs
    assert gpt2.decode_bytes([c3]) == b"\xc3" and gpt2.decode([c3]) == "\ufffd"
    assert gpt2.decode_bytes([]) == b"" and gpt2.decode_bytes(array.array("I", HELLO)) == b"Hello world"
    assert gpt2.decode_bytes_batch([HELLO, [c3]], num_threads=2) == [b"Hello world", b"\xc3"]
    with pytest.raises(KeyError, match="Invalid token for decoding: 50257"):
        gpt2.decode_bytes([15496, 50257])
    with pytest.raises(OverflowError):
        gpt2.decode_bytes([-1])
    v = gpt2.token_byte_values()
    assert len(v) == 50256 and v == sorted(v) and len(set(v)) == 50256 and b" world" in v and EOT.encode() not in v


# ---- objects --------------------------------------------------------------------------------------------------

def test_pickle(gpt2, gpt2_path):
    for t in (gpt2, toks.Tokenizer.from_str(pathlib.Path(gpt2_path).read_text()),
              toks.Tokenizer.from_buffer(pathlib.Path(gpt2_path).read_bytes())):
        t.encode_special_tokens = True
        u = pickle.loads(pickle.dumps(t))
        t.encode_special_tokens = False
        assert u.encode_special_tokens is True and u.encode("a" + EOT) == t.encode("a" + EOT, added_tokens="nonspecial")
        assert u.info() == t.info()


def test_subclass(gpt2_path):
    class Mine(toks.Tokenizer):
        def extra(self):
            return len(self.encode("Hello world"))

    t = Mine.from_file(gpt2_path)
    assert isinstance(t, Mine) and t.extra() == 2


def test_no_leak(gpt2):
    text = "Hello world, \u00e9t\u00e9 \U0001F600 " * 50
    big = "word " * 2000

    def work():
        for _ in range(300):
            ids = gpt2.encode(text)
            gpt2.decode(ids)
            gpt2.pieces(text)
            gpt2.encode_batch([text, big, text])
            gpt2.encode_into(big, array.array("I", bytes(4 * 3000)))
            e = gpt2.encode_ex(text)
            e.ids, e.type_ids, e.attention_mask, e.special_tokens_mask, e.tokens
            gpt2.encode_batch_ex([text, big])
            gpt2.encode(text + EOT, allowed_special="all")
            gpt2.encode_ordinary(text)
            gpt2.decode_bytes(ids)
    work()                                          # warm: int objects, slots, utf-8 caches
    gc.collect()
    tracemalloc.start()
    a = tracemalloc.take_snapshot()
    work()
    gc.collect()
    b = tracemalloc.take_snapshot()
    tracemalloc.stop()
    grown = sum(s.size_diff for s in b.compare_to(a, "filename"))
    assert grown < 64 << 10, f"{grown} bytes left after 300 rounds"
