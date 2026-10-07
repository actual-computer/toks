# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""tests/fuzz/hostile/hf_arbiter.py: hf 0.23.2 against toks on the hostile files (docs/fuzz.md "hostile").

hf_corpus.py's method (the hardening lane's corpus oracle), on the hand-reasoned hostile corpus: for
every file, hf's verdict against toks's; where both load, toks's ids must equal hf's on the battery
texts; a file toks accepts and hf refuses is a finding; NOMEM on a small file is a finding.

    uv run --with tokenizers==0.23.2 python tests/fuzz/hostile/hf_arbiter.py \\
        --driver build/toks_driver --out build/hostile-hf.json <hostile-files-dir>...

Prints the counts; exit 1 on any finding. Texts that are not utf-8 are skipped (hf takes str).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python"))

CODES = {-1: "OPEN", -2: "FORMAT", -3: "UNSUPPORTED", -5: "TIER", -6: "SCRATCH", -7: "ID", -8: "CAP",
         -9: "LIMIT", -10: "ARG", -11: "NOMEM"}

# the hostile battery's probe texts (texts.py's, hex-decoded here): a slice of the hostile texts that
# are utf-8, plus the standard probes
PROBE = [b"Hello, world! It's 2026.", b" leading", b"trailing ", b"\t\n\r\n x",
         b"e\xcc\x81 caf\xc3\xa9 \xe2\x84\xab", b"\xe4\xb8\xad\xe6\x96\x87 \xed\x95\x9c\xea\xb5\xad\xec\x96\xb4",
         b"1234567 3.14", b"", b"\xe2\x96\x81\xe2\x96\x81a b", b"  \n\n  def f(x):\n    return x\n",
         b"[CLS] ##ing <s> </s>", b"'s 'S don't DON'T", b"\n\nA", b"\t\t1", b"<|im_start|>user\nhi<|im_end|>\n",
         b"\xef\xbf\xbd\x80", b"\xf0\x9f\x92\xa9", b"a\xcc\x81\xcc\x86b", b"\xed\xa0\x80", b"\xc0\x80"]

FLAGS = [(0, "ALL", True), (4, "ALL", False), (1, "NONSPECIAL", True), (5, "NONSPECIAL", False),
         (2, "NONE", True), (6, "NONE", False)]


class Driver:
    def __init__(self, path):
        self.path = path
        self.p = None

    def close(self):
        if self.p is not None:
            try:
                self.p.stdin.close()
            except OSError:
                pass
            self.p.wait()
            self.p = None

    def _req(self, op, flags, path, payload):
        if self.p is None:
            self.p = subprocess.Popen([self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.p.stdin.write(b"TKR1" + op.to_bytes(4, "little") + flags.to_bytes(4, "little") +
                           len(path).to_bytes(4, "little") + path + payload)
        self.p.stdin.flush()
        st = self.p.stdout.read(16)
        if len(st) != 16:
            raise RuntimeError("driver EOF")
        status = int.from_bytes(st[:8], "little", signed=True)
        n = int.from_bytes(st[8:], "little", signed=True)
        out = self.p.stdout.read(n) if n > 0 else b""
        if len(out) != max(n, 0):
            raise RuntimeError("driver EOF")
        return status, out

    def load(self, path):
        self.close()
        st, _ = self._req(1, 0, path.encode(), b"")
        if st != 0:
            self.close()
        return st

    def encode(self, text, flags):
        st, out = self._req(2, flags, b"", len(text).to_bytes(4, "little") + text)
        if st != 0:
            return st, None
        return 0, [int.from_bytes(out[i:i + 4], "little") for i in range(0, len(out), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    import tokenizers

    files = []
    for d in a.dirs:
        for root, _, names in os.walk(d):
            for f in sorted(names):
                files.append(os.path.join(root, f))
    files.sort()
    drv = Driver(a.driver)
    c = {"files": 0, "hf_loads": 0, "toks_loads": 0, "both": 0, "neither": 0, "toks_only": 0,
         "hf_only": 0, "texts_compared": 0, "texts_not_utf8": 0, "mismatch_files": 0,
         "toks_encode_errors": 0, "hf_encode_errors": 0, "driver_failures": 0}
    rep = {"counts": c, "toks_only": [], "hf_only": [], "mismatches": [], "errors": []}
    seen = set()
    for path in files:
        with open(path, "rb") as f:
            data = f.read()
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        c["files"] += 1
        hf, hf_ns, hf_err = None, None, None
        # catastrophic regexes hang hf's oniguruma (backtracking); toks refuses or answers them in
        # bounded time -- an asymmetry recorded in the report, not compared here
        catastrophic = b"(a+)+$" in data or b"(a*)*b" in data
        try:
            hf = None if catastrophic else tokenizers.Tokenizer.from_str(data.decode("utf-8"))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            hf_err = ((type(e).__name__ + ": " + str(e)).splitlines() or [""])[0][:200]
        try:
            st = drv.load(os.path.abspath(path))
        except Exception as e:
            c["driver_failures"] += 1
            rep["errors"].append({"file": path, "error": "load: " + str(e)})
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
            rep["toks_only"].append({"file": path, "bytes": len(data), "hf": hf_err})
            continue
        if st != 0:
            c["hf_only"] += 1
            rep["hf_only"].append({"file": path, "toks": st, "code": CODES.get(st, str(st))})
            continue
        c["both"] += 1
        bad = 0
        for t in PROBE:
            try:
                s = t.decode("utf-8")
            except UnicodeDecodeError:
                c["texts_not_utf8"] += 1
                continue
            for fl, mode, post in FLAGS:
                try:
                    hv = hf
                    if mode == "NONSPECIAL":
                        if hf_ns is None:
                            hf_ns = tokenizers.Tokenizer.from_str(data.decode("utf-8"))
                            hf_ns.encode_special_tokens = True
                        hv = hf_ns
                    want = hv.encode(s, add_special_tokens=post).ids
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    c["hf_encode_errors"] += 1
                    rep["errors"].append({"file": path, "error": "hf encode: " + str(e)[:180]})
                    continue
                try:
                    tst, got = drv.encode(t, fl)
                except Exception as e:
                    c["driver_failures"] += 1
                    rep["errors"].append({"file": path, "error": "encode: " + str(e)})
                    drv.close()
                    break
                c["texts_compared"] += 1
                if tst != 0:
                    c["toks_encode_errors"] += 1
                    rep["errors"].append({"file": path, "flags": fl, "toks": tst})
                    continue
                if got != want:
                    bad += 1
                    if bad <= 3:
                        rep["mismatches"].append({"file": path, "flags": fl, "text": t.hex(),
                                                  "hf": want[:64], "toks": got[:64],
                                                  "n_hf": len(want), "n_toks": len(got)})
            if drv.p is None:
                break
        if bad:
            c["mismatch_files"] += 1
    drv.close()
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("hf_arbiter: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    print("hf_arbiter: hf loads, toks refuses: " +
          (", ".join(f"{k} {v}" for k, v in
                     sorted({CODES.get(e["toks"], e["toks"]): 0 for e in rep["hf_only"]}.items())) or "none"))
    finding = (c["mismatch_files"] or c["toks_only"] or c["driver_failures"] or c["toks_encode_errors"] or
               c["hf_only"])
    print("hf_arbiter: " + ("FINDINGS" if finding else "clean"))
    return 1 if finding else 0


if __name__ == "__main__":
    sys.exit(main())
