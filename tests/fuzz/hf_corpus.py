#!/usr/bin/env python3
"""tests/fuzz/hf_corpus.py: the corpus oracle (docs/fuzz.md section 6): the fuzz corpora's tokenizer files through hf
tokenizers 0.23.2 and through toks, outside the fuzzer.

The harnesses compare toks with itself (the asm tiers with their C twins, both reading the tables the compiler
built), so a wrong table looks the same on both sides; only hf's ids pair the compiler with the reference, and the
parity suites do that on the census files alone. This step does it on the hostile files the campaigns grew:

  - every file hf loads, toks loads (or refuses with a documented code: exact or refused), and where both load,
    toks's ids equal hf's on the load battery's texts: its probe texts, slices of the file, and runs of the
    tokenizer's own tokens (hf's decode of random ids), in ALL and NONSPECIAL, with and without post-processing;
  - a file toks accepts and hf refuses is a finding in itself;
  - a file hf loads and toks refuses is listed by toks's code (TOKS_E_NOMEM on a small file is a finding).

toks runs as tests/driver/toks_driver.c over the release library (toks_load on the file, as a caller would);
inputs of the load harness's tiktoken form ('TIKTOKEN' + ...) are counted and skipped. Texts that are not UTF-8
are skipped (hf takes str). Run from the repository root on a lab host:

    make lib && clang -std=c17 -O2 -Iinclude -o build/toks_driver tests/driver/toks_driver.c build/<os>-<isa>/libtoks.a
    uv run -q --with tokenizers==0.23.2 python tests/fuzz/hf_corpus.py --driver build/toks_driver \\
        --out build/fuzz/hf_corpus.json build/fuzz/corpus/load_json build/fuzz/seeds/load_json

Prints the counts; the JSON report lists every difference. Exit 1 on any finding (a mismatch, toks accepting a
file hf refuses, NOMEM, a driver failure), else 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

CODES = {-1: "OPEN", -2: "FORMAT", -3: "UNSUPPORTED", -5: "TIER", -6: "SCRATCH", -7: "ID", -8: "CAP", -9: "LIMIT",
         -10: "ARG", -11: "NOMEM"}

# tests/fuzz/load.h FZ_PROBE, as bytes
PROBE = [b"Hello, world! It's 2026.", b" leading", b"trailing ", b"\t\n\r\n x", b"e\xcc\x81 caf\xc3\xa9 \xe2\x84\xab",
         b"\xe4\xb8\xad\xe6\x96\x87 \xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xd8\xa7\xd9\x84", b"1234567 3.14",
         b"<|endoftext|>", b"\xe2\x96\x81\xe2\x96\x81\x61 b", b"\xff\xfe\x80 x",
         b"  \n\n  def f(x):\n    return x\n", b"[CLS] ##ing <s> </s>"]

# driver flags: 0 ALL, 1 NONSPECIAL; +4 no post-processing (hf add_special_tokens=False)
FLAGS = [(0, "ALL", True), (4, "ALL", False), (1, "NONSPECIAL", True), (5, "NONSPECIAL", False)]


class Driver:
    """tests/driver/toks_driver.c's protocol; the driver exits on a refused load, so it restarts on demand"""

    def __init__(self, path: str):
        self.path = path
        self.p = None

    def close(self) -> None:
        if self.p is not None:
            try:
                self.p.stdin.close()
            except OSError:
                pass
            self.p.wait()
            self.p = None

    def _req(self, op: int, flags: int, path: bytes, payload: bytes):
        if self.p is None:
            self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.p.stdin.write(b"TKR1" + op.to_bytes(4, "little") + flags.to_bytes(4, "little") +
                           len(path).to_bytes(4, "little") + path + payload)
        self.p.stdin.flush()
        st = self.p.stdout.read(16)                    # every reply: i64 status, i64 nbytes, payload
        if len(st) != 16:
            raise RuntimeError("driver EOF")
        status = int.from_bytes(st[:8], "little", signed=True)
        n = int.from_bytes(st[8:], "little", signed=True)
        out = self.p.stdout.read(n) if n > 0 else b""
        if len(out) != max(n, 0):
            raise RuntimeError("driver EOF")
        return status, out

    def load(self, path: str) -> int:
        self.close()                                   # one context per process: a fresh driver per file
        st, _ = self._req(1, 0, path.encode(), b"")
        if st != 0:
            self.close()
        return st

    def encode(self, text: bytes, flags: int):
        st, out = self._req(2, flags, b"", len(text).to_bytes(4, "little") + text)
        if st != 0:
            return st, None
        return 0, [int.from_bytes(out[i:i + 4], "little") for i in range(0, len(out), 4)]


def texts_of(data: bytes, hf, seed: int):
    """the load battery's kinds of text (load.h fz_battery): probes, slices of the file, runs of its own tokens"""
    rng = random.Random(seed)
    out = list(PROBE)
    for _ in range(4):
        if data:
            a = rng.randrange(len(data))
            out.append(data[a:a + 1 + rng.randrange(min(512, len(data) - a))])
    n = hf.get_vocab_size(with_added_tokens=True)
    for _ in range(3):
        if n:
            ids = [rng.randrange(n) for _ in range(1 + rng.randrange(24))]
            try:
                out.append("".join(hf.decode([i], skip_special_tokens=False) + (" " if rng.randrange(4) == 0 else "")
                                   for i in ids).encode("utf-8", "surrogatepass"))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:                      # an id hf cannot decode alone: skip the run
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="at most this many files (0: all)")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    import tokenizers

    files = sorted(os.path.join(d, f) for d in a.dirs for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    if a.limit:
        files = files[:a.limit]
    drv = Driver(a.driver)
    c = {"files": 0, "tiktoken_inputs": 0, "hf_loads": 0, "toks_loads": 0, "both": 0, "neither": 0,
         "toks_only": 0, "hf_only": 0, "texts_compared": 0, "texts_not_utf8": 0, "mismatch_files": 0,
         "toks_encode_errors": 0, "hf_encode_errors": 0, "driver_failures": 0}
    hf_only_codes: dict = {}
    rep = {"counts": c, "hf_only_by_code": hf_only_codes, "toks_only": [], "hf_only": [], "mismatches": [],
           "errors": []}
    seen = set()
    for path in files:
        with open(path, "rb") as f:
            data = f.read()
        h = hashlib.sha256(data).hexdigest()
        if h in seen:                                  # corpus and seeds overlap
            continue
        seen.add(h)
        c["files"] += 1
        if data.startswith(b"TIKTOKEN"):
            c["tiktoken_inputs"] += 1
            continue
        hf, hf_ns, hf_err = None, None, None
        try:
            hf = tokenizers.Tokenizer.from_str(data.decode("utf-8"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:                     # not UTF-8, not JSON, a tokenizer hf refuses, or a panic
            hf_err = ((type(e).__name__ + ": " + str(e)).splitlines() or [""])[0][:200]
        try:
            st = drv.load(os.path.abspath(path))
        except Exception as e:
            c["driver_failures"] += 1
            rep["errors"].append({"file": path, "sha256": h, "error": "load: " + str(e)})
            drv.close()
            continue
        if hf is not None:
            c["hf_loads"] += 1
        if st == 0:
            c["toks_loads"] += 1
        if hf is None and st != 0:
            c["neither"] += 1
            continue
        if hf is None:
            c["toks_only"] += 1
            rep["toks_only"].append({"file": path, "sha256": h, "bytes": len(data), "hf": hf_err})
            continue
        if st != 0:
            c["hf_only"] += 1
            name = CODES.get(st, str(st))
            hf_only_codes[name] = hf_only_codes.get(name, 0) + 1
            rep["hf_only"].append({"file": path, "sha256": h, "bytes": len(data), "toks": st, "code": name})
            continue
        c["both"] += 1
        bad = 0
        for t in texts_of(data, hf, int(h[:12], 16)):
            try:
                s = t.decode("utf-8")
            except UnicodeDecodeError:
                c["texts_not_utf8"] += 1
                continue
            for fl, mode, post in FLAGS:
                try:
                    hv = hf
                    if mode == "NONSPECIAL":
                        if hf_ns is None:                  # a fresh load (oracle.py Tok.view's reason)
                            hf_ns = tokenizers.Tokenizer.from_str(data.decode("utf-8"))
                            hf_ns.encode_special_tokens = True
                        hv = hf_ns
                    want = hv.encode(s, add_special_tokens=post).ids
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    c["hf_encode_errors"] += 1
                    rep["errors"].append({"file": path, "sha256": h, "error": "hf encode: " + str(e)[:200]})
                    continue
                try:
                    tst, got = drv.encode(t, fl)
                except Exception as e:
                    c["driver_failures"] += 1
                    rep["errors"].append({"file": path, "sha256": h, "error": "encode: " + str(e)})
                    drv.close()
                    break
                c["texts_compared"] += 1
                if tst != 0:
                    c["toks_encode_errors"] += 1
                    rep["errors"].append({"file": path, "sha256": h, "flags": fl, "text": t.hex(), "toks": tst})
                    continue
                if got != want:
                    bad += 1
                    if bad <= 3:
                        rep["mismatches"].append({"file": path, "sha256": h, "flags": fl, "text": t.hex(),
                                                  "hf": want[:64], "toks": got[:64], "n_hf": len(want),
                                                  "n_toks": len(got)})
            if drv.p is None:
                break
        if bad:
            c["mismatch_files"] += 1
    drv.close()
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("hf_corpus: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    print("hf_corpus: hf loads, toks refuses, by code: " +
          (", ".join(f"{k} {v}" for k, v in sorted(hf_only_codes.items())) or "none"))
    finding = (c["mismatch_files"] or c["toks_only"] or c["driver_failures"] or c["toks_encode_errors"] or
               hf_only_codes.get("NOMEM", 0))
    return 1 if finding else 0


if __name__ == "__main__":
    sys.exit(main())
