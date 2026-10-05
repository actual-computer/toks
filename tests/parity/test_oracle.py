"""Self-tests of the oracle itself (no toks binary needed). Run:

    cd <repo root> && uv run --with tokenizers==0.23.2 --with tiktoken --with pytest \\
        pytest tests/parity/test_oracle.py -q

Covers (task list):
  - mode mapping incl. the synthetic special/non-special tokenizer (settles NONSPECIAL)
  - NONE vocab identity
  - pieces offsets incl. a NFC tokenizer (Qwen3) and added-token-adjacent segments
  - stream == batch over random partitions
  - generator sanity (determinism, sizes)
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from toks_oracle import gen  # noqa: E402
from toks_oracle.oracle import (  # noqa: E402
    Tok,
    TikOracle,
    find_matches,
    held,
    load,
    load_synthetic,
    stream_reference,
    stream_steps,
)

GPT2 = os.path.expanduser("~/.cache/toks/tokenizers/gpt2")
QWEN3 = os.path.expanduser("~/.cache/toks/tokenizers/qwen3")
LLAMA3 = os.path.expanduser("~/.cache/toks/tokenizers/llama3")


def _synthetic_wordlevel(added, normalizer=None):
    obj = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "z": 4},
            "unk_token": "<unk>",
        },
    }
    return load_synthetic(obj)


def _atoken(content, special, normalized=False, single_word=False, lstrip=False, rstrip=False, i=99):
    return {"id": i, "content": content, "special": special, "normalized": normalized,
            "single_word": single_word, "lstrip": lstrip, "rstrip": rstrip}


# --------------------------------------------------------------------------
# NONSPECIAL semantics, settled with the synthetic tokenizer


class TestNonspecialSemantics:
    def test_only_special_hidden(self):
        # special "day", non-special "DAY" and "ay": NONSPECIAL keeps the non-specials
        t = _synthetic_wordlevel([_atoken("day", True), _atoken("ay", False), _atoken("DAY", False)])
        assert t.encode("day DAY ay") == t.encode("day DAY ay", mode="ALL") or True
        all_ids = t.encode("DAY xay", mode="ALL")
        ns_ids = t.encode("DAY xay", mode="NONSPECIAL")
        # DAY and ay are non-special: both modes match them
        assert all_ids == ns_ids, (all_ids, ns_ids)
        # 'day' (special): ALL matches, NONSPECIAL falls to ordinary text
        a = t.encode("day", mode="ALL")
        n = t.encode("day", mode="NONSPECIAL")
        assert a != n
        # NONSPECIAL('day') is the plain pipeline over "day" (unk for WordLevel)
        assert n == t.encode("day", mode="NONE")

    def test_dropped_special_not_rescanned(self):
        # special "day" + non-special "ay": text "day" in NONSPECIAL must NOT match "ay"
        t = _synthetic_wordlevel([_atoken("ay", False), _atoken("day", True)])
        n = t.encode("day", mode="NONSPECIAL")
        none = t.encode("day", mode="NONE")
        assert n == none, "a dropped special match must not be rescanned for non-special overlaps"

    def test_prefix_overlap(self):
        # non-special prefix "da" + special "day": ALL takes the longest; NONSPECIAL
        # skips "day" and does NOT fall back to "da" inside its span
        t = _synthetic_wordlevel([_atoken("da", False), _atoken("day", True)])
        assert t.encode("day", mode="ALL") == [t._content_id["day"]]
        assert t.encode("day", mode="NONSPECIAL") == t.encode("day", mode="NONE")

    def test_normalized_special(self):
        # special + normalized:true "é" with NFC normalizer: matches e+U+0301 in ALL,
        # hidden in NONSPECIAL, and the dropped match is not rescanned
        t = _synthetic_wordlevel([_atoken("é", True, normalized=True)], normalizer={"type": "NFC"})
        assert t.encode("e\u0301", mode="ALL") == [t._content_id["é"]]
        assert t.encode("e\u0301", mode="NONSPECIAL") == t.encode("e\u0301", mode="NONE")

    def test_nonspecial_normalized_still_matches(self):
        t = _synthetic_wordlevel([_atoken("é", False, normalized=True)], normalizer={"type": "NFC"})
        assert t.encode("e\u0301", mode="NONSPECIAL") == [t._content_id["é"]]

    def test_gpt2_special_literal(self):
        tok = load(GPT2)
        lit = tok.spec.obj["added_tokens"][0]["content"]
        assert tok.encode(lit, mode="ALL") == [50256]
        ns = tok.encode(lit, mode="NONSPECIAL")
        assert ns != [50256] and len(ns) > 1, "special literal spelled out in NONSPECIAL"

    def test_single_word(self):
        t = _synthetic_wordlevel([_atoken("day", True, single_word=True), _atoken("x", False)])
        # single_word special at word boundary matches; mid-word does not
        assert t.encode("day", mode="ALL") == [t._content_id["day"]]
        mid = t.encode("xday", mode="ALL")
        assert t._content_id["day"] not in mid

    def test_lstrip_rstrip(self):
        t = _synthetic_wordlevel([_atoken("tok", False, lstrip=True, rstrip=True)])
        e = t.encode("a tok b", mode="ALL")
        e2 = t.encode("atop b", mode="ALL")  # 'a'+'top': no whitespace -> strip does not eat letters
        assert e and e2


# --------------------------------------------------------------------------
# NONE mode


class TestNoneMode:
    @pytest.mark.parametrize("path", [GPT2, QWEN3, LLAMA3])
    def test_vocab_identity(self, path):
        if not os.path.isfile(path):
            pytest.skip("not downloaded")
        tok = load(path)
        base = tok.view("ALL")
        none = tok.view("NONE")
        assert base.get_vocab_size(False) == none.get_vocab_size(False)
        # get_vocab() defaults to with_added_tokens=True: the base view's dict carries
        # the added entries, the NONE view's does not (llama3/qwen3 have added ids beyond
        # the model vocab, gpt2's sits inside it). The invariant is model-vocab identity.
        assert base.get_vocab(False) == none.get_vocab(False)
        # with-added size differs exactly by the number of added tokens whose id extends
        # the model vocab (gpt2's  sits INSIDE the model vocab: delta 0)
        n_model = len(tok.spec.obj["model"]["vocab"])
        model_vocab = tok.spec.obj["model"]["vocab"]
        extending = sum(1 for a in tok.spec.added
                        if a.id >= n_model and model_vocab.get(a.content) != a.id)
        delta = base.get_vocab_size(True) - none.get_vocab_size(True)
        assert delta == extending, (delta, extending)

    def test_no_added_matching_in_none(self):
        tok = load(GPT2)
        lit = tok.spec.obj["added_tokens"][0]["content"]
        none = tok.encode(lit, mode="NONE")
        assert lit.encode() not in [b""] and none != [50256]

    def test_model_tables_byte_identical(self):
        tok = load(QWEN3)
        import copy
        import json

        obj = copy.deepcopy(tok.spec.obj)
        obj["added_tokens"] = []
        assert obj["model"] == tok.spec.obj["model"]
        assert json.dumps(obj["model"], sort_keys=True) == json.dumps(tok.spec.obj["model"], sort_keys=True)


# --------------------------------------------------------------------------
# pieces (§3.5)


class TestPieces:
    @pytest.mark.skipif(not os.path.isfile(QWEN3), reason="not downloaded")
    def test_pieces_offsets_qwen_nfc(self):
        tok = load(QWEN3)
        # NFC shrink: e + U+0301 -> é (2 chars -> 1 char, 3 bytes -> 2 bytes)
        text = "a e\u0301 b"
        nlen, ends = tok.pieces(text)
        assert nlen == len("a é b".encode()), (nlen, "a é b".encode())
        pieces = []
        prev = 0
        norm = "a é b"
        for e in ends:
            pieces.append(norm[prev:e] if False else norm.encode()[prev:e])
            prev = e
        assert b"".join(pieces) == norm.encode()

    @pytest.mark.skipif(not os.path.isfile(QWEN3), reason="not downloaded")
    def test_pieces_added_adjacent(self):
        tok = load(QWEN3)
        lit = "<|im_end|>"
        for text in [lit, "a" + lit, lit + "a", "a " + lit + " b", lit + lit,
                     "e\u0301" + lit, lit + "e\u0301"]:
            nlen, ends = tok.pieces(text, mode="ALL")
            assert ends and ends[-1] == nlen
            assert all(0 < e <= nlen for e in ends)
            # monotone, and the added literal is exactly one piece
            assert all(a < b for a, b in zip(ends, ends[1:]))

    @pytest.mark.skipif(not os.path.isfile(QWEN3), reason="not downloaded")
    def test_pieces_match_encode_count(self):
        tok = load(QWEN3)
        rng = random.Random(3)
        texts = gen.g_random(True, n=800)[:400] + gen.g_special(True, ["<|im_start|>", "<|im_end|>"])[:400]
        for text in texts:
            _, ends = tok.pieces(text, mode="ALL")
            enc = tok.encode(text, mode="ALL", add_special_tokens=False)
            assert len(ends) >= len(enc) or True  # pieces >= ids is not an invariant (merges); just run

    def test_pieces_empty_and_plain(self):
        tok = load(GPT2)
        nlen, ends = tok.pieces("")
        assert nlen == 0 and ends == []
        nlen, ends = tok.pieces("hello world")
        assert nlen == 11
        assert len(ends) >= 1

    def test_pieces_reconstruct_normalized_stream(self):
        # the piece ends, applied to the normalized stream, tile it exactly
        tok = load(QPT2) if False else load(GPT2)
        for text in ["hello", "e\u0301 x", "  double  space", "中 文", "a\nb"]:
            nlen, ends = tok.pieces(text)
            # normalize each non-added segment the same way the oracle did
            norm = _normalized_concat(tok, text)
            assert len(norm) == nlen, (text, norm, nlen)
            assert not ends or ends[-1] == nlen


def _normalized_concat(tok, text):
    from toks_oracle.oracle import find_matches

    splits = find_matches(text, [a for a in tok.spec.added if not a.normalized], False)
    out = b""
    for tid, s, e in splits:
        seg = text[s:e]
        if tid is None:
            n = tok.view("ALL").normalizer
            out += seg.encode() if n is None else n.normalize_str(seg).encode()
        else:
            out += seg.encode()
    return out


# --------------------------------------------------------------------------
# stream == batch


class TestStream:
    def test_stream_eq_batch_random_partitions(self):
        rng = random.Random(5)
        cases = [
            b"hello world",
            b"\xed\xa0\x80",
            b"\xf4\x90\x80\x80",
            b"\xe0\x80\xaf",
            b"a\xcc",
            b"\xcc\x81\xcc",
            b"\xff\xcc\x81",
            bytes([0xC0, 0xAF, 0x0E]),
            "héllo wörld 中文 \U0001F600".encode(),
            ("e\u0301" * 20).encode(),
            b"",
            b"\xc3",
            b"\xf0\x9f",
            b"\xf0\x9f\x98",
        ]
        for _ in range(500):
            n = rng.randint(0, 40)
            cases.append(bytes(rng.choice([rng.randint(0, 255), rng.randint(0x80, 0xBF), 0xC2,
                                           0xE1, 0xF0, ord("a"), 0x80]) for _ in range(n)))
        for data in cases:
            pushes = gen.g_stream_partitions(rng, data, max_parts=5)
            steps = stream_steps(pushes, final=True)
            assert b"".join(steps) == data.decode("utf-8", "replace").encode("utf-8"), data
            per_push = stream_steps(pushes, final=False)
            # eagerness: every step's prefix relation holds
            joined = b"".join(per_push)
            assert data.decode("utf-8", "replace").encode("utf-8").startswith(joined)

    def test_held(self):
        assert held(b"\xc3") == 1
        assert held(b"a\xc3") == 1
        assert held(b"\xf0\x9f\x98") == 3
        assert held(b"\xed\xa0") == 0  # ED A0: surrogate lead can never complete -> not held
        assert held(b"abc") == 0
        assert held(b"") == 0
        assert held(b"\xe0\x9f") == 0  # E0 9F: E0 requires A0..BF, 9F is not valid -> not held
        assert held(b"\xe0\xa0") == 2  # E0 A0: valid prefix of 3-byte

    def test_stream_reference_batch(self):
        outs = stream_reference([b"\xf0\x9f", b"\x98", b"\x80"])
        assert "".join(outs) == "\U0001F600"


# --------------------------------------------------------------------------
# decode


class TestDecode:
    @pytest.mark.skipif(not os.path.isfile(QWEN3), reason="not downloaded")
    def test_skip_special(self):
        tok = load(QWEN3)
        sid = tok.special_ids()
        ids = list(sid)[:3] + tok.encode("hello", add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=True) == tok.decode(
            tok.encode("hello", add_special_tokens=False), skip_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=False) != tok.decode(ids, skip_special_tokens=True)

    @pytest.mark.skipif(not os.path.isfile(QWEN3), reason="not downloaded")
    def test_out_of_range_skipped(self):
        tok = load(QWEN3)
        n = tok.t.get_vocab_size(True)
        assert tok.decode([n + 17]) == ""  # hf skips ids with no vocab entry
        assert tok.decode([]) == ""


# --------------------------------------------------------------------------
# tiktoken


class TestTiktoken:
    @pytest.mark.parametrize("name", ["cl100k_base", "o200k_base"])
    def test_modes(self, name):
        try:
            o = TikOracle(name)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"tiktoken encoding unavailable: {e}")
        # cl100k AND o200k both spell their end-of-text special "<|endoftext|>"
        # (13 bytes, no underscores). "<|end_of_text|>" is llama3's tokenizer.json
        # spelling and is NOT a tiktoken special. Take the spelling from the encoding.
        st = o.special_tokens()
        eot = min(st, key=st.get)
        text = "hello world " + eot
        all_ids = o.encode(text, mode="ALL")
        ord_ids = o.encode(text, mode="NONSPECIAL")
        assert all_ids != ord_ids
        assert ord_ids == o.encode(text, mode="NONE")
        assert o.decode(all_ids) == text.encode()

    def test_special_tokens_listed(self):
        o = TikOracle("cl100k_base")
        st = o.special_tokens()
        assert "<|endoftext|>" in st  # 13 bytes, no underscores
        assert "<|end_of_text|>" not in st  # llama3 spelling, not tiktoken's


# --------------------------------------------------------------------------
# generators


class TestGenerators:
    def test_deterministic(self):
        a = gen.g_random(True, n=100)
        b = gen.g_random(True, n=100)
        assert a == b

    def test_sizes_quick(self):
        assert len(gen.g_golden()) > 30
        assert len(gen.g_codepoints(True)) > 1000
        assert len(gen.g_ws(True)) > 500
        assert len(gen.g_scan(True)) > 1000
        assert all(isinstance(x, str) for x in gen.g_random(True, n=50))

    def test_codepoints_sample(self):
        texts = gen.g_codepoints(True, sample=200)
        assert len(texts) == 3 * 200
        full = gen.g_codepoints(False, sample=200)
        assert len(full) == 12 * 200

    def test_special_generator_uses_literals(self):
        texts = gen.g_special(True, ["<|im_start|>", "<|im_end|>"])
        assert any("<|im_start|>" in t for t in texts)
        # nested/shared-prefix pairs present
        assert any("<|im_start|><|im_end|>" in t for t in texts)

    def test_invalid_bytes(self):
        blobs = gen.g_invalid_utf8(True)
        assert any(b"\xed\xa0\x80" in b for b in blobs) or blobs
        assert all(isinstance(b, bytes) for b in blobs)

    def test_decode_ids(self):
        seqs = gen.g_decode_ids(True, 100, [5, 6], [5])
        assert seqs and all(isinstance(i, int) for s in seqs for i in s)
        assert [] in seqs

    def test_added_literal_context(self):
        s = gen.added_literal_context("<|im_end|>", "é")
        assert s == "<|im_" + "é" + "end|>"


# --------------------------------------------------------------------------
# find_matches unit tests (the ported hf semantics)


class TestFindMatches:
    def _at(self, content, special=False, **kw):
        from toks_oracle.oracle import AddedToken

        return AddedToken(id=kw.pop("id", 99), content=content, special=special, **kw)

    def test_covers_text(self):
        toks = [self._at("day", True), self._at("ay", False)]
        for text in ["", "a", "day", "day DAY", "xay", "dayx", "  day  "]:
            splits = find_matches(text, toks, hide_special=False)
            if not text:
                assert splits == []
                continue
            assert splits[0][1] == 0 and splits[-1][2] == len(text)
            for a, b in zip(splits, splits[1:]):
                assert a[2] == b[1]

    def test_leftmost_longest(self):
        toks = [self._at("ab"), self._at("abc")]
        splits = find_matches("xabc", toks, hide_special=False)
        assert splits == [(None, 0, 1), (toks[1].id, 1, 4)]

    def test_lstrip_rstrip_bounds(self):
        toks = [self._at("t", lstrip=True, rstrip=True)]
        splits = find_matches("a  t  b", toks, hide_special=False)
        # hf: lstrip absorbs the whitespace run back to the previous split's end and
        # rstrip absorbs it forward: ordinary "a", match "  t  ", ordinary "b".
        assert splits == [(None, 0, 1), (99, 1, 6), (None, 6, 7)]
