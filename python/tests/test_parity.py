"""python/tests/test_parity.py: the toks wheel == hf tokenizers 0.23.2, through the Python API.

Per pinned tokenizer (gpt2, llama3, glm53, qwen38, gemma4, o200k = gpt-oss; tests/c/test_e2e.py and
tests/data/targets pin their sha256): texts from python/toks_oracle/gen.py (every generator, sampled evenly;
TOKS_PY_N texts), every
mode x add_special_tokens for encode and encode_batch, pieces in every mode, decode both ways (of the
encodings and of g_decode_ids sequences), stream decode when the library has it, token strings over the
whole vocabulary, and a sample of the parity case set ($TOKS_CASES/<name>.jsonl, TOKS_PY_CASES lines) when
it is on the host. The reference is python/toks_oracle (hf 0.23.2; NONE = the file with added_tokens
removed; pieces = hf's normalizer + pre-tokenizer offsets in normalized bytes; stream = hf's decode of all
ids, with every push's text a prefix of it). Decode of an id beyond the table is TOKS_E_ID by SPEC 3.4
(hf skips it): counted as spec_e_id, never as equal.

    TOKS_PY_N=1500 TOKS_PY_CASES=3000 TOKS_PY_REPORT=report.json pytest python/tests/test_parity.py
"""
import json
import os
import random

import pytest

import toks
from conftest import ROOT, report, tok_path

tokenizers = pytest.importorskip("tokenizers")
from toks_oracle import gen as G  # noqa: E402
from toks_oracle import oracle as O  # noqa: E402

NAMES = ["gpt2", "llama3", "glm53", "qwen38", "gemma4", "o200k"]
MODES = (("ALL", "all"), ("NONSPECIAL", "nonspecial"), ("NONE", "none"))
N = int(os.environ.get("TOKS_PY_N", "1500"))
NCASES = int(os.environ.get("TOKS_PY_CASES", "3000"))
CASES = os.path.expanduser(os.environ.get("TOKS_CASES", "~/.cache/toks/cases"))
E_ID = -7
E_UNSUPPORTED = -3
E_LIMIT = -9


def show(s) -> str:
    """repr without raw '|' (tool output strips <|...|>): special tokens stay readable"""
    return ascii(s).replace("|", "\\x7c")[:300]


def test_hf_pin():
    assert tokenizers.__version__ == "0.23.2", "the semantic reference is hf tokenizers 0.23.2"


class Tally:
    def __init__(self):
        self.c = {}
        self.bad = []

    def add(self, op, what, ok=None, why=""):
        d = self.c.setdefault(op, {"cases": 0, "equal": 0, "spec_e_id": 0})
        d["cases"] += 1
        if what == "equal":
            d["equal"] += 1
        elif what in ("spec_e_id", "spec_limit"):
            d[what] = d.get(what, 0) + 1
        elif len(self.bad) < 12:
            self.bad.append(f"{op}: {why}")

    def check(self, key):
        report(key, self.c)
        assert not self.bad, f"{key}: mismatches vs hf 0.23.2:\n" + "\n".join(self.bad)
        return self.c


@pytest.fixture(scope="module", params=NAMES)
def pair(request):
    path = tok_path(request.param)
    return request.param, toks.Tokenizer.from_file(path), O.load(path)


def sample(xs, k):
    if len(xs) <= k:
        return list(xs)
    step = len(xs) / k
    return [xs[int(i * step)] for i in range(k)]


def gen_texts(o, n):
    added = [a.content for a in o.spec.added]
    pools = [G.g_codepoints(True, sample=4000), G.g_random(True, n=4000), G.g_ws(True), G.g_scan(True), G.g_cjk(True),
             G.g_emoji(True), G.g_marks(True), G.g_long(True)]
    if added:
        pools.append(G.g_special(True, added))
    k = max(1, n // len(pools))
    texts = list(G.g_golden())
    for p in pools:
        texts += sample(p, k)
    return texts


def pieces_equal(t, o, x, mode, m):
    """toks pieces == hf's. For sentencepiece-style bpe (bpe_spm) toks runs the normalizer as tables and
    reports offsets in the caller's bytes (docs/algorithms/spm_bpe.md 6.3: hf's pieces are toks' pieces with
    phi applied), hf in the normalized text: there the count, order and coverage must agree."""
    w = o.pieces(x, mode=mode)[1]
    g = t.pieces(x, added_tokens=m)
    if t.info()["algorithm"] != "bpe_spm":
        return g == w, w, g
    n = len(x.encode("utf-8")) if isinstance(x, str) else len(x)
    ok = len(g) == len(w) and all(a < b for a, b in zip(g, g[1:])) and (g[-1] == n if g else n == 0)
    return ok, w, g


def ids_ok(t, ids):
    return all(i < t.info()["n_ids"] for i in ids)


def test_encode_pieces_decode(pair):
    name, t, o = pair
    texts = gen_texts(o, N)
    n_ids = t.info()["n_ids"]
    tl = Tally()
    for mode, m in MODES:
        for pp in (True, False):
            got = t.encode_batch(texts, add_special_tokens=pp, added_tokens=m)
            for x, g in zip(texts, got):
                w = o.encode(x, mode=mode, add_special_tokens=pp)
                one = t.encode(x, add_special_tokens=pp, added_tokens=m)
                ok = g == w and one == w
                tl.add("encode", "equal" if ok else "diff",
                       why=f"{mode} pp={pp} {show(x)}: want {w[:16]} got {one[:16]} batch {g[:16]}")
                if mode == "ALL" and pp and ok:                 # decode of a real encoding, both ways
                    for skip in (False, True):
                        dw, dg = o.decode(w, skip_special_tokens=skip), t.decode(one, skip_special_tokens=skip)
                        tl.add("decode", "equal" if dw == dg else "diff", why=f"skip={skip} {show(x)}: {show(dw)} vs {show(dg)}")
        for x in texts:
            ok, w, g = pieces_equal(t, o, x, mode, m)
            tl.add("pieces", "equal" if ok else "diff", why=f"{mode} {show(x)}: want {w[:16]} got {g[:16]}")
    seqs = G.g_decode_ids(True, o._vocab_size, sorted(o.added_ids()), sorted(o.special_ids()))
    for ids in sample(seqs, max(200, N)):
        for skip in (False, True):
            if any(i >= n_ids for i in ids):
                with pytest.raises(toks.Error) as e:
                    t.decode(ids, skip_special_tokens=skip)
                tl.add("decode", "spec_e_id" if e.value.code == E_ID else "diff", why=f"{ids[:8]}: {e.value}")
                continue
            w, g = o.decode(ids, skip_special_tokens=skip), t.decode(ids, skip_special_tokens=skip)
            tl.add("decode", "equal" if w == g else "diff", why=f"skip={skip} {ids[:12]}: {show(w)} vs {show(g)}")
    c = tl.check(f"{name}/generated")
    assert c["encode"]["equal"] == 6 * len(texts)


def stream_supported(t) -> bool:
    try:
        st = t.decode_stream()
        st.push([0])
        st.flush()
        return True
    except toks.Error as e:
        if e.code != E_UNSUPPORTED:
            raise
        return False


def stream_check(t, o, pushes, skip):
    """'equal' | 'spec_limit' | a diff: DecodeStream over the pushes == hf decode of all ids (stream == batch,
    SPEC 3.4), and after every push the text so far is a prefix of it (a push emits only bytes no later id
    can change). The binding grows the stream's hold on TOKS_E_LIMIT (toks_stream_hold), so 'spec_limit' (a
    refusal) no longer happens; it is still counted apart if a library without the hold refuses."""
    full = o.decode([i for p in pushes for i in p], skip_special_tokens=skip)
    st = t.decode_stream(skip_special_tokens=skip)
    got = ""
    for p in pushes:
        try:
            got += st.push(p)
        except toks.Error as e:
            if e.code == E_LIMIT and t.info()["algorithm"] == "bpe_spm":
                return "spec_limit"
            raise
        if not full.startswith(got):
            return f"after push {p[:8]}: {show(got[-40:])} is not a prefix of hf's {show(full[:80])}"
    got += st.flush()
    return "equal" if got == full else f"stream {show(got[:80])} != hf decode {show(full[:80])}"


def test_stream(pair):
    name, t, o = pair
    if not stream_supported(t):
        pytest.skip("toks_stream_push returns TOKS_E_UNSUPPORTED in this library build")
    rng = random.Random(7)
    n_ids = t.info()["n_ids"]
    tl = Tally()
    seqs = [s for s in G.g_decode_ids(True, o._vocab_size, sorted(o.added_ids()), sorted(o.special_ids()))
            if all(i < n_ids for i in s)]
    seqs += [o.encode(x, add_special_tokens=True) for x in gen_texts(o, max(100, N // 5))]
    for ids in sample(seqs, max(300, N)):
        cut = sorted(rng.sample(range(len(ids) + 1), min(len(ids) + 1, rng.randint(1, 6))))
        pushes = [ids[a:b] for a, b in zip([0] + cut, cut + [len(ids)])]
        for skip in (False, True):
            r = stream_check(t, o, pushes, skip)
            tl.add("stream", r if r in ("equal", "spec_limit") else "diff", why=f"skip={skip} {r}")
    tl.check(f"{name}/stream")


def test_vocab(pair):
    name, t, o = pair
    h = o.t
    n_ids = t.info()["n_ids"]
    bad = [i for i in range(n_ids + 64) if t.id_to_token(i) != h.id_to_token(i)]
    assert not bad, f"id_to_token differs at {bad[:10]}: {[show(t.id_to_token(i)) for i in bad[:3]]}"
    for added in (True, False):
        v = h.get_vocab(with_added_tokens=added)
        assert t.get_vocab(with_added_tokens=added) == v
        assert t.get_vocab_size(with_added_tokens=added) == h.get_vocab_size(with_added_tokens=added)
    v = h.get_vocab(True)
    bad = [s for s, i in v.items() if t.token_to_id(s) != h.token_to_id(s)]
    assert not bad, f"token_to_id differs for {[show(s) for s in bad[:5]]}"
    for s in ("", " ", "not a token at all \u00e9", "<|nope|>", "\U0001F600" * 3):
        assert t.token_to_id(s) == h.token_to_id(s)
    report(f"{name}/vocab", {"ids": n_ids + 64, "tokens": len(v)})


# ---- the parity case set (tests/parity/gen_cases.py; on the lab hosts) ---------------------------------------

def read_cases(path, n):
    """n lines spread evenly over the file (seeks, so a 2 GB set costs n reads), all lines of a small one"""
    size = os.path.getsize(path)
    out = []
    with open(path, "rb") as f:
        if size <= (64 << 20):
            lines = f.read().splitlines()
            return [json.loads(x) for x in sample(lines, n) if x.strip()]
        for k in range(n):
            f.seek(k * size // n)
            if k:
                f.readline()
            line = f.readline()
            if line.strip():
                out.append(json.loads(line))
    return out


@pytest.mark.parametrize("name", NAMES)
def test_case_set(name):
    path = os.path.join(CASES, f"{name}.jsonl")
    if NCASES <= 0 or not os.path.isfile(path):
        pytest.skip(f"no parity case set at {path} (tests/parity/gen_cases.py)")
    tp = tok_path(name)
    t, o = toks.Tokenizer.from_file(tp), O.load(tp)
    n_ids = t.info()["n_ids"]
    stream = stream_supported(t)
    tl = Tally()
    for case in read_cases(path, NCASES):
        op, fl = case["op"], case["flags"]
        if op == "encode":
            mode, m = MODES[fl & 3]
            w = o.encode(case["text"], mode=mode, add_special_tokens=not fl & 4)
            g = t.encode(case["text"], added_tokens=m, add_special_tokens=not fl & 4)
            tl.add(op, "equal" if g == w else "diff", why=f"flags {fl} {show(case['text'])}: want {w[:16]} got {g[:16]}")
        elif op == "pieces":                         # the m1a sets carry mode NONE only: run every mode
            for mode, m in MODES:
                ok, w, g = pieces_equal(t, o, case["text"], mode, m)
                tl.add(op, "equal" if ok else "diff", why=f"{mode} {show(case['text'])}: want {w[:16]} got {g[:16]}")
        elif op == "decode":
            ids, skip = case["ids"], bool(fl & 1)
            if not ids_ok(t, ids):
                try:
                    t.decode(ids, skip_special_tokens=skip)
                    tl.add(op, "diff", why=f"{ids[:8]}: no TOKS_E_ID")
                except toks.Error as e:
                    tl.add(op, "spec_e_id" if e.code == E_ID else "diff", why=str(e))
                continue
            w, g = o.decode(ids, skip_special_tokens=skip), t.decode(ids, skip_special_tokens=skip)
            tl.add(op, "equal" if g == w else "diff", why=f"skip={skip} {ids[:12]}: {show(w)} vs {show(g)}")
        elif op == "stream" and stream:
            pushes, skip = case["pushes"], bool(fl & 1)
            if not all(ids_ok(t, p) for p in pushes):
                continue
            r = stream_check(t, o, pushes, skip)
            tl.add(op, r if r in ("equal", "spec_limit") else "diff", why=f"skip={skip} {r}")
    c = tl.check(f"{name}/cases")
    assert sum(d["cases"] for d in c.values()) > 0


# ---- the critical targets (tests/data/targets/ledger.txt) ------------------------------------------------------

def ledger():
    out = []
    with open(os.path.join(ROOT, "tests", "data", "targets", "ledger.txt")) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                target, file, sha, kind, *want = line.split()
                out.append((target, file, kind, " ".join(want)))
    return out


@pytest.mark.parametrize("target,file,kind,want", ledger())
def test_target(target, file, kind, want, tmp_path):
    path = tok_path(file)
    if file.endswith(".tiktoken"):                  # toks_load wants the model directory's own names
        stem = path[: -len(".tiktoken")]
        for src, dst in ((path, "tiktoken.model"), (stem + "_tokenizer_config.json", "tokenizer_config.json"),
                         (stem + "_tokenization_kimi.py", "tokenization_kimi.py")):
            if not os.path.isfile(src):
                pytest.skip(f"{src} is missing (tools/corpora/fetch_tokenizers.py --targets)")
            os.symlink(src, tmp_path / dst)
        path = str(tmp_path / "tiktoken.model")
        if kind == "compiles":
            toks.Tokenizer.from_file(path)
            pytest.skip(f"{target} loads; its reference is tiktoken, not hf tokenizers (the kimi lane's tests)")
    if kind == "compiles":
        t, o = toks.Tokenizer.from_file(path), O.load(path)
        texts = gen_texts(o, max(60, N // 10))
        bad = []
        for mode, m in MODES:
            for pp in (True, False):
                for x in texts:
                    w, g = o.encode(x, mode=mode, add_special_tokens=pp), t.encode(x, added_tokens=m, add_special_tokens=pp)
                    if w != g:
                        bad.append(f"{mode} pp={pp} {show(x)}: want {w[:12]} got {g[:12]}")
                    if mode == "ALL" and pp and t.decode(g) != o.decode(w, skip_special_tokens=True):
                        bad.append(f"decode {show(x)}")
        report(f"target/{target}", {"texts": len(texts), "encodes": 6 * len(texts), "diffs": len(bad)})
        assert not bad, "\n".join(bad[:10])
    elif kind == "refused":
        with pytest.raises(toks.Error) as e:
            toks.Tokenizer.from_file(path)
        assert want in str(e.value), str(e.value)
    else:
        pytest.skip(f"{target}: {kind} {want}")
