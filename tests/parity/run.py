#!/usr/bin/env python3
"""tests/parity/run.py — the toks parity runner (SPEC §9 parity loop).

    uv run --with tokenizers==0.23.2 --with pytest python tests/parity/run.py \
        --tokenizer ~/.cache/toks/tokenizers/gpt2 --cases ~/.cache/toks/cases/gpt2.jsonl \
        --toks ./build/toks_driver --out report.json

Compares the hf oracle (python/toks_oracle) against a toks binary speaking the
driver protocol (tests/driver/toks_driver.c), case by case; diffs are minimized
(a delta-debugging shrink that must preserve the mismatch, then a byte-level
1-minimal pass) and written to the report. Exit 0 iff no diffs.

Case file format — one JSON per line, either
    {"op": "encode", "text": "...", "flags": 0}
    {"op": "pieces", "text": "...", "flags": 0}
    {"op": "decode", "ids": [...], "flags": 0}
    {"op": "stream", "pushes": [[...], ...], "flags": 0}
flags: 0 = ALL, 1 = NONSPECIAL, 2 = NONE; +4 NO_POSTPROCESS (encode/pieces);
+1 SKIP_SPECIAL (decode/stream); +2 (stream) a caller's hold (toks_stream_hold) that grows on
TOKS_E_LIMIT. text is a str (encoded utf-8 on the wire).

Stream (SPEC §3.4, §4.7): hf has no stream decoder with SPEC's semantics (0.23.2's DecodeStream
holds every output that ends in U+FFFD, final or not), so the reference is SPEC's rule (a push emits
every byte no later id can change): ByteLevel, from hf's own token strings: B = the bytes of the ids
hf's decode keeps (id_to_token through the ByteLevel alphabet, specials dropped by content under
SKIP_SPECIAL), push k emits lossy(B_k minus its held utf-8 prefix) minus what was emitted before;
sentencepiece-style chains (Replace / Metaspace, ByteFallback, Fuse, Strip, or none), from hf's decode
of prefixes: after push k the stream has emitted hf.decode(ids[:cut]), cut = the ids so far, or the
start of a trailing run of kept <0xHH> tokens whose bytes are still a utf-8 prefix (its output waits on
its end); a run that must hold more than 44 bytes is TOKS_E_LIMIT (counted, not a diff) in st's own
bytes, and with flag 2 the driver answers that refusal as toks.h says (the run moved into a caller's
hold of at least the current size plus the push's n, the same ids pushed again), so nothing is refused
and every case is compared. The flush emits the rest. Every case also checks the reference itself: its
pushes plus flush must equal hf's decode of the whole sequence (else a "reference" diff). The driver
checks atomicity, the bound and the exact capacity on every call (toks_driver.c STREAM).

The two equalities a stream gives: (1) always, push_1 ++ ... ++ push_k ++ flush == hf.decode(ids);
(2) for runs that are valid utf-8 as a whole (12 x U+13000 is one 48-byte run), that concatenation
also equals the concatenation of hf's DecodeStream steps. DecodeStream emits a run's chars as soon as
the prefix decodes, so a later stray byte, which turns the whole run into U+FFFD in hf.decode, can
contradict what it already emitted; the stream emits only bytes no later id can change, so per push
it emits a run at the run's end (tests/parity/stream_hold.py prints both for the U+13000 cases).

Self-test (--self-test) needs no toks binary: the fake toks is the oracle
itself (must be 100% clean) and a deliberately corrupted copy (must be caught,
shrunk and reported). This proves the diff + minimizer before the library exists.

Minimization contract: shrink() only ever *reduces* the case (shorter text /
fewer ids / fewer pushes) and only keeps a candidate when the oracle-vs-toks
mismatch is still present; a case that stops misbehaving is reverted. Results
are therefore always a sub-case of the original failure.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from toks_oracle import oracle as O  # noqa: E402

# ---------------------------------------------------------------- protocol client


class DriverError(Exception):
    pass


class Driver:
    """Speaks the toks_driver.c stdin/stdout protocol. Lazy: one process reused."""

    def __init__(self, argv, cwd=None):
        self.argv = [str(a) for a in argv]
        self.cwd = cwd
        self.p = None

    def _start(self):
        if self.p is None:
            self.p = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.cwd)
        return self.p

    def _rd(self, n):
        b = self.p.stdout.read(n)
        if len(b) != n:
            raise DriverError("driver EOF")
        return b

    def _req(self, op, flags, path=b"", payload=b""):
        p = self._start()
        p.stdin.write(b"TKR1" + op.to_bytes(4, "little") + flags.to_bytes(4, "little")
                      + len(path).to_bytes(4, "little") + path + payload)
        p.stdin.flush()
        status = int.from_bytes(self._rd(8), "little", signed=True)
        n = int.from_bytes(self._rd(8), "little", signed=True)
        if status != 0 or n < 0:
            return status if status != 0 else n, b""
        out = self._rd(n) if n else b""
        return 0, out

    def load(self, path):
        st, _ = self._req(1, 0, path.encode())
        if st != 0:
            raise DriverError(f"load {path}: status {st}")

    def encode(self, text_bytes, flags):
        st, out = self._req(2, flags, b"", len(text_bytes).to_bytes(4, "little") + text_bytes)
        if st != 0:
            return st, None
        n = len(out) // 4
        return 0, [int.from_bytes(out[4 * i:4 * i + 4], "little") for i in range(n)]

    def pieces(self, text_bytes, flags):
        st, out = self._req(3, flags, b"", len(text_bytes).to_bytes(4, "little") + text_bytes)
        if st != 0:
            return st, None
        n = len(out) // 4
        return 0, [int.from_bytes(out[4 * i:4 * i + 4], "little") for i in range(n)]

    def decode(self, ids, flags):
        st, out = self._req(4, flags, b"", len(ids).to_bytes(4, "little")
                            + b"".join(i.to_bytes(4, "little") for i in ids))
        return st, out

    def stream(self, pushes, flags):
        payload = len(pushes).to_bytes(4, "little") + b"".join(
            len(p).to_bytes(4, "little") + b"".join(i.to_bytes(4, "little") for i in p) for p in pushes)
        st, out = self._req(6, flags, b"", payload)
        if st != 0:
            return st, None
        outs, at = [], 0
        while at < len(out):
            k = int.from_bytes(out[at:at + 4], "little")
            outs.append(out[at + 4:at + 4 + k])
            at += 4 + k
        return 0, outs

    def close(self):
        if self.p:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.p.kill()
            self.p = None


class FakeDriver:
    """A toks implementation from the oracle itself; `corrupt` flips outcomes.

    corrupt: {op: every_n} — an input whose zlib.crc32 key hits the class
    (crc32(key) % every_n == 0) gets a wrong answer (ids shifted by 1, the
    last piece end dropped, a U+FFFD appended). Input-keyed, NOT call-count
    keyed, so a mismatch reproduces exactly when the minimizer re-runs a
    candidate. Used by --self-test to prove the runner catches, shrinks and
    reports before the real library exists.
    """

    def __init__(self, tok: "O.Tok", corrupt: dict = None):
        self.tok = tok
        self.corrupt = corrupt or {}

    def _hit(self, op, key: bytes) -> bool:
        if op not in self.corrupt:
            return False
        import zlib

        return zlib.crc32(key) % self.corrupt[op] == 0

    def load(self, path):
        self.tok = O.load(path)

    def encode(self, text_bytes, flags):
        mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[flags & 3]
        ast = (flags & 4) == 0
        ids = self.tok.encode(text_bytes.decode("utf-8", "replace"), mode=mode, add_special_tokens=ast)
        if self._hit("encode", text_bytes):
            ids = [i + 1 for i in ids]
        return 0, ids

    def pieces(self, text_bytes, flags):
        mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[flags & 3]
        _nlen, ends = self.tok.pieces(text_bytes.decode("utf-8", "replace"), mode=mode)
        if self._hit("pieces", text_bytes) and ends:
            ends = ends[:-1]
        return 0, ends

    def decode(self, ids, flags):
        s = self.tok.decode(ids, skip_special_tokens=bool(flags & 1))
        out = s.encode("utf-8", "replace")
        if self._hit("decode", b"".join(i.to_bytes(4, "little") for i in ids)):
            out += b"\xef\xbf\xbd"
        return 0, out

    def stream(self, pushes, flags):
        outs = oracle_stream(self.tok, {"pushes": pushes, "flags": flags})
        if outs == "E_LIMIT":
            return -9, None
        outs = list(outs)
        key = b"".join(b"".join(i.to_bytes(4, "little") for i in p) for p in pushes)
        if self._hit("stream", key):
            outs = [o + b"\xef\xbf\xbd" for o in outs]
        return 0, outs

    def close(self):
        pass


# ---------------------------------------------------------------- oracle side


def oracle_encode(tok, case):
    mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[case["flags"] & 3]
    return tok.encode(case["text"], mode=mode, add_special_tokens=not (case["flags"] & 4))


def oracle_pieces(tok, case):
    mode = {0: "ALL", 1: "NONSPECIAL", 2: "NONE"}[case["flags"] & 3]
    _n, ends = tok.pieces(case["text"], mode=mode)
    return ends


def oracle_decode(tok, case):
    return tok.decode(case["ids"], skip_special_tokens=bool(case["flags"] & 1)).encode("utf-8", "replace")


STREAM_HOLD = 44       # stream.c SST_RUN: the longest open byte run a stream holds


def _byte_token(s):
    """hf ByteFallback: '<0x' h h '>' with u8::from_str_radix(.., 16), which also takes '+h'; else None"""
    if s is None or len(s) != 6 or not s.startswith("<0x") or not s.endswith(">"):
        return None
    h = s[3:5]
    if h[0] == "+":
        h = h[1:]
    if not h or any(c not in "0123456789abcdefABCDEF" for c in h):
        return None
    return int(h, 16)


def _valid_so_far(b):
    """whole valid utf-8, or valid up to a sequence cut by the end"""
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        return e.reason == "unexpected end of data" and e.end == len(b)


class SpmStreamRef:
    """SPEC §3.4's stream rule for a non-ByteLevel chain, from hf's decode of prefixes (hold: a caller's
    hold that grows, so no run is too long)."""

    def __init__(self, tok):
        self.tok = tok
        t = tok.t
        self.special = {a.content for a in t.get_added_tokens_decoder().values() if a.special}
        dec = json.dumps(tok.spec.obj.get("decoder"))
        self.bf = '"ByteFallback"' in dec
        self.cache = {}

    def _s(self, i):
        if i not in self.cache:
            self.cache[i] = self.tok.t.id_to_token(i)
        return self.cache[i]

    def outcome(self, pushes, skip, hold=False):
        """(per-push bytes + the flush, or "E_LIMIT")"""
        ids, steps, done = [], [], b""
        for p in pushes:
            ids += p
            cut = len(ids)
            if self.bf:
                run, j = [], len(ids)
                while j > 0:
                    s = self._s(ids[j - 1])
                    if s is None or (skip and s in self.special):
                        j -= 1
                        continue
                    b = _byte_token(s)
                    if b is None:
                        break
                    run.append(b)
                    j -= 1
                if run:
                    rb = bytes(reversed(run))
                    if _valid_so_far(rb):
                        if len(rb) > STREAM_HOLD and not hold:
                            return "E_LIMIT"
                        cut = j
            e = self.tok.decode(ids[:cut], skip_special_tokens=skip).encode("utf-8")
            if not e.startswith(done):
                return None
            steps.append(e[len(done):])
            done = e
        whole = self.tok.decode(ids, skip_special_tokens=skip).encode("utf-8")
        if not whole.startswith(done):
            return None
        steps.append(whole[len(done):])
        return steps


class StreamRef:
    """SPEC §3.4's stream rule from hf's own token strings (see the module docstring)."""

    def __init__(self, tok):
        import tokenizers

        self.t = tok.t
        if not isinstance(self.t.decoder, tokenizers.decoders.ByteLevel):
            raise ValueError("stream reference: not a ByteLevel decoder")
        self.special = {a.content for a in self.t.get_added_tokens_decoder().values() if a.special}
        bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.u2b = {chr(c): b for b, c in zip(bs, cs)}
        self.cache = {}

    def token(self, i):
        """(string or None, bytes) of id i: hf's ByteLevel decoder maps a token through the alphabet when
        every char is in it, else takes its utf-8 (an added token like '\u00abx\u00bb')."""
        r = self.cache.get(i)
        if r is None:
            s = self.t.id_to_token(i)
            if s is None:
                r = (None, b"")
            elif all(ch in self.u2b for ch in s):
                r = (s, bytes(self.u2b[ch] for ch in s))
            else:
                r = (s, s.encode("utf-8"))
            self.cache[i] = r
        return r

    def steps(self, pushes, skip):
        bp = []
        for p in pushes:
            b = []
            for i in p:
                s, tb = self.token(i)
                if s is None or (skip and s in self.special):
                    continue
                b.append(tb)
            bp.append(b"".join(b))
        return O.stream_steps(bp)


STREAM_REF_ERRORS = [0]
STREAM_E_LIMIT = [0]           # cases the stream must refuse (an open run over STREAM_HOLD bytes), and did


def oracle_stream(tok, case):
    """per push, then the flush ("E_LIMIT" when the stream must refuse); the concatenation must be hf's
    decode of the whole sequence."""
    ref = getattr(tok, "_stream_ref", None)
    if ref is None:
        import tokenizers

        bl = isinstance(tok.t.decoder, tokenizers.decoders.ByteLevel)
        ref = tok._stream_ref = StreamRef(tok) if bl else SpmStreamRef(tok)
    skip = bool(case["flags"] & 1)
    whole = tok.decode([i for p in case["pushes"] for i in p], skip_special_tokens=skip).encode("utf-8")
    if isinstance(ref, SpmStreamRef):
        steps = ref.outcome(case["pushes"], skip, bool(case["flags"] & 2))
        if steps == "E_LIMIT":
            return steps
    else:
        steps = ref.steps(case["pushes"], skip)
    if steps is None or b"".join(steps) != whole:
        STREAM_REF_ERRORS[0] += 1
        got = b"<a prefix that is not a prefix>" if steps is None else b"".join(steps)
        return ["REFERENCE != hf decode: " + repr(got[:80]) + " vs " + repr(whole[:80])]
    return steps


ORACLE = {"encode": oracle_encode, "pieces": oracle_pieces,
          "decode": oracle_decode, "stream": oracle_stream}


# ---------------------------------------------------------------- fake "driver" wire

# FakeDriver speaks python, not the wire; adapt: the runner calls DRIVER.encode(...)
# on the case's utf-8 bytes for both real and fake. Stream is compared per-push.


def run_case(tok, drv, case):
    """-> None (match) or dict(op=..., detail=...) describing the mismatch."""
    op = case["op"]
    want = ORACLE[op](tok, case)
    if op in ("encode", "pieces"):
        tb = case["text"].encode("utf-8")
        st, got = (drv.encode if op == "encode" else drv.pieces)(tb, case["flags"])
    elif op == "decode":
        st, got = drv.decode(case["ids"], case["flags"])
    else:
        if not hasattr(drv, "stream"):
            return {"op": op, "detail": "SKIP: driver has no stream op"}
        st, got = drv.stream(case["pushes"], case["flags"])
        if want == "E_LIMIT":
            STREAM_E_LIMIT[0] += st == -9
            return None if st == -9 else {"op": op, "detail": f"want TOKS_E_LIMIT (an open run over {STREAM_HOLD} "
                                                               f"bytes), driver status {st}"}
    if st != 0:
        return {"op": op, "detail": f"driver status {st}"}
    if got != want:
        return {"op": op, "detail": _fmt_diff(op, want, got)}
    return None


def _fmt_diff(op, want, got):
    if isinstance(want, list) and want and isinstance(want[0], bytes):  # stream
        i = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
        return (f"stream push {i}: want {want[i] if i < len(want) else '<none>'!r} "
                f"got {got[i] if i < len(got) else '<none>'!r}")
    at = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), min(len(want), len(got)))
    wa = want[at] if at < len(want) else None
    ga = got[at] if at < len(got) else None
    return f"first diff at [{at}]: want {wa!r} got {ga!r} (len {len(want)} vs {len(got)})"


# ---------------------------------------------------------------- minimization


def shrink(tok, drv, case, budget=2000):
    """Delta-debug the case down while the mismatch is preserved.

    Line strategy: try deleting each third of the sequence-ish payload (text
    chars, ids, pushes), accept if the case still fails; then a byte-safe
    1-minimal pass on text. `drv`-side determinism is required (the real toks
    is deterministic; the self-test corruptor is count-based, so we reset the
    fake's counter by reconstructing it — see _fresh_drv in self_test).
    """
    calls = 0

    def still_bad(c):
        nonlocal calls
        calls += 1
        if calls > budget:
            raise TimeoutError
        return run_case(tok, drv, c) is not None

    case = dict(case)
    op = case["op"]
    if op in ("encode", "pieces"):
        text = case["text"]
        # char-level chunk deletion (never splits a surrogate pair: python strs are atoms)
        chunk = max(1, len(text) // 3)
        while chunk >= 1:
            i = 0
            while i < len(text):
                cand = text[:i] + text[i + chunk:]
                if cand and cand != text:
                    try:
                        if still_bad(dict(case, text=cand)):
                            text = cand
                            continue  # accepted: retry same position (cand is shorter)
                    except TimeoutError:
                        case["text"] = text
                        return case
                i += chunk
            chunk //= 2
        chunk = max(1, chunk // 3)
        while chunk >= 1 and len(text) > 1:
            i = 0
            while i < len(text):
                cand = text[:i] + text[i + chunk:]
                if cand and cand != text:
                    c2 = dict(case, text=cand)
                    try:
                        if still_bad(c2):
                            text = cand
                            continue
                    except TimeoutError:
                        case["text"] = text
                        return case
                i += chunk
            chunk //= 2
        case["text"] = text
    else:
        key = "ids" if op == "decode" else "pushes"
        seq = case[key]
        chunk = max(1, len(seq) // 3) if seq else 0
        while chunk >= 1:
            i = 0
            while i < len(seq):
                cand = seq[:i] + seq[i + chunk:]
                if cand != seq:
                    c2 = dict(case, **{key: cand})
                    try:
                        if still_bad(c2):
                            seq = cand
                            continue
                    except TimeoutError:
                        case[key] = seq
                        return case
                i += chunk
            chunk //= 3 if chunk > 1 else 2
            if chunk == 0:
                break
        case[key] = seq
    return case


# ---------------------------------------------------------------- self-test


def self_test(tokenizer_path, verbose=False):
    """The fake toks = the oracle (must pass 100%) and a corrupted copy (must be caught)."""
    tok = O.load(tokenizer_path)
    ok = True

    # build a tiny deterministic case list covering every op and flag combo
    import random

    from toks_oracle import gen

    rng = random.Random(99)
    texts = gen.g_golden()[:20] + gen.g_random(True, n=60)[:60]
    added = [a.content for a in tok.spec.added][:8]
    if added:
        texts += gen.g_special(True, added)[:60]
    cases = []
    for t in texts:
        for fl in (0, 1, 2, 4, 5, 6):
            cases.append({"op": "encode", "text": t, "flags": fl})
        for fl in (0, 1, 2):
            cases.append({"op": "pieces", "text": t, "flags": fl})
    nv = tok.t.get_vocab_size(True)
    aid = sorted(tok.added_ids())[:8]
    sid = sorted(tok.special_ids())[:8]
    for ids in gen.g_decode_ids(True, nv, aid, sid)[:150]:
        for fl in (0, 1):
            cases.append({"op": "decode", "ids": ids, "flags": fl})
    # stream: partition a few texts' encoded ids
    for t in texts[:12]:
        ids = tok.encode(t, mode="ALL", add_special_tokens=False)
        if len(ids) > 40:
            ids = ids[:40]
        for _ in range(3):
            pushes = [ids[i:j] for i, j in _cuts(rng, len(ids))]
            cases.append({"op": "stream", "pushes": pushes, "flags": 1})

    # 1. clean fake: every case must match
    clean = FakeDriver(tok)
    bad = [c for c in cases if run_case(tok, clean, c) is not None]
    print(f"[self-test] clean fake over {len(cases)} cases: {len(bad)} mismatches (want 0)")
    if bad:
        ok = False
        for c in bad[:3]:
            print("  UNEXPECTED:", json.dumps(_esc(c))[:160], run_case(tok, clean, c))

    # 2. corrupted fakes: each must be caught, shrunk, and stay a sub-case
    corrupts = [
        {"encode": 7},       # crc-class inputs: every id +1
        {"pieces": 5},       # crc-class inputs: last piece end dropped
        {"decode": 6},       # crc-class inputs: U+FFFD appended
        {"stream": 4},       # crc-class inputs: U+FFFD appended to every push
    ]
    for cor in corrupts:
        d = FakeDriver(tok, cor)
        hits = [i for i, c in enumerate(cases) if (r := run_case(tok, d, c)) is not None
                and not r["detail"].startswith("SKIP")]
        print(f"[self-test] corrupt {list(cor)}: {len(hits)}/{len(cases)} cases flagged at first pass")
        if not hits:
            ok = False
            print("  FAIL: corruption not detected")
            continue
        shrunk_sizes = []
        for i in sorted(hits, key=lambda i: -_case_size(cases[i]))[:3]:  # the biggest first
            c = cases[i]
            r = run_case(tok, d, c)
            assert r is not None
            s = shrink(tok, d, dict(c))
            # the mismatch must survive the shrink on the minimized case...
            assert run_case(tok, d, s) is not None, "minimized case stopped failing?"
            # ...and the minimizer must have made progress on at least one
            shrunk_sizes.append((_case_size(c), _case_size(s)))
            print(f"  case #{i} op={c['op']}: {_case_size(c)} -> {_case_size(s)} units")
        if not any(a > b for a, b in shrunk_sizes):
            ok = False
            print(f"  FAIL: minimizer never shrank any case {shrunk_sizes}")
    return ok


def _cuts(rng, n):
    if n == 0:
        return [(0, 0)]
    k = rng.randint(1, min(5, n))
    cuts = sorted(rng.sample(range(1, n), k - 1)) if k > 1 else []
    b = [0] + cuts + [n]
    return list(zip(b, b[1:]))


def _case_size(c):
    if "text" in c:
        return len(c["text"])
    if "ids" in c:
        return len(c["ids"])
    return sum(len(p) for p in c["pushes"])


def _esc(c):
    d = dict(c)
    if "text" in d:
        d["text"] = d["text"].encode("unicode_escape").decode()[:80]
    return d


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    ap.add_argument("--cases", help="case file (.jsonl); omit with --self-test")
    ap.add_argument("--toks", help="toks driver binary argv (quoted) or 'fake'")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (of this shard)")
    ap.add_argument("--shard", default="0/1", help="I/N: the case lines whose index is I mod N")
    ap.add_argument("--pieces-modes", default="0,1,2",
                    help="run every pieces case once per listed mode (gen_cases.py's sets carry mode NONE only)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-shrink", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test(args.tokenizer) else 1)

    tok = O.load(args.tokenizer)
    if not args.toks or args.toks == "fake":
        drv = FakeDriver(tok)
    else:
        import shlex

        drv = Driver(shlex.split(args.toks))
        drv.load(args.tokenizer)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    pieces_modes = [int(x) for x in args.pieces_modes.split(",") if x != ""]
    n_ids = tok.t.get_vocab_size(True)

    t0 = time.time()
    diffs = []
    n_shrunk = 0
    counts: dict = {}          # "op flags" -> cases run
    unsupported: dict = {}     # op -> cases toks does not run (the driver's SKIP)
    spec_e_id = 0              # decode of an id beyond the table: TOKS_E_ID by SPEC §3.4 (hf skips it)
    n_run = 0

    def run_one(i, c):
        nonlocal n_shrunk, spec_e_id
        key = f"{c['op']} {c['flags']}"
        counts[key] = counts.get(key, 0) + 1
        r = run_case(tok, drv, c)
        if r is None:
            return
        if r["detail"].startswith("SKIP"):
            unsupported[c["op"]] = unsupported.get(c["op"], 0) + 1
            return
        flat = c["ids"] if c["op"] == "decode" else [x for p in c.get("pushes", []) for x in p]
        if c["op"] in ("decode", "stream") and r["detail"] == "driver status -7" and any(x >= n_ids for x in flat):
            spec_e_id += 1
            return
        entry = {"case": _esc(c), "raw_case_index": i, **r}
        if not args.no_shrink and len(diffs) < 200:
            try:
                entry["minimized"] = _esc(shrink(tok, drv, dict(c)))
                n_shrunk += 1
            except Exception as e:  # noqa: BLE001
                entry["minimize_error"] = repr(e)
        diffs.append(entry)
        if len(diffs) <= 5:
            print(f"[diff] case {i} op={c['op']} flags={c['flags']}: {r['detail'][:120]}", flush=True)

    with open(args.cases, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % shard_n != shard_i:
                continue
            line = line.strip()
            if not line:
                continue
            if args.limit and n_run >= args.limit:
                break
            n_run += 1
            c = json.loads(line)
            if c["op"] == "pieces":
                for m in pieces_modes:
                    run_one(i, dict(c, flags=m))
            else:
                run_one(i, c)
    drv.close()
    n_compared = sum(counts.values()) - sum(unsupported.values()) - spec_e_id   # the cases actually compared
    report = {
        "tokenizer": args.tokenizer,
        "cases": args.cases,
        "shard": args.shard,
        "n_lines": n_run,
        "n_cases": sum(counts.values()),
        "n_compared": n_compared,
        "counts": dict(sorted(counts.items())),
        "unsupported": unsupported,
        "spec_e_id": spec_e_id,
        "stream_ref_errors": STREAM_REF_ERRORS[0],
        "stream_e_limit": STREAM_E_LIMIT[0],
        "n_diffs": len(diffs),
        "n_minimized": n_shrunk,
        "elapsed_s": round(time.time() - t0, 2),
        "diffs": diffs[:200],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[run] {report['n_cases']} cases ({n_run} lines), {len(diffs)} diffs, {n_shrunk} minimized, "
          f"unsupported {unsupported}, spec E_ID {spec_e_id}, stream reference errors {STREAM_REF_ERRORS[0]}, stream E_LIMIT {STREAM_E_LIMIT[0]}, "
          f"{report['elapsed_s']}s -> {args.out}", flush=True)
    # the count line, "PASS|FAIL <n> compared <suite> (<detail>)": a run that compared nothing fails, as does one
    # with a diff or a reference error; unsupported (SKIP) and spec E_ID cases are not compared and are named in the
    # detail
    why = [w for w, bad in ((f"{len(diffs)} diffs", diffs), (f"{STREAM_REF_ERRORS[0]} stream reference errors",
                                                              STREAM_REF_ERRORS[0]),
                            ("nothing compared", n_compared <= 0)) if bad]
    suite = f"parity {os.path.basename(args.tokenizer)} {os.path.basename(args.cases)} shard {args.shard}"
    detail = f"{report['n_cases']} cases, {sum(unsupported.values())} unsupported, {spec_e_id} spec E_ID"
    print(f"{'FAIL' if why else 'PASS'} {max(n_compared, 0)} compared {suite} ({'; '.join(why) if why else detail})",
          flush=True)
    sys.exit(1 if why else 0)


if __name__ == "__main__":
    main()
