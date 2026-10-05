#!/usr/bin/env python3
"""tests/hardening/crlf_proof.py BUILD_DIR [BIN_DIR]: the teeth of tests/c/lines.inc. Each line-oriented data file a
tests/c reader reads, rewritten with CRLF line ends (what a core.autocrlf=true checkout writes), must make its test FAIL
fast, naming the file, the line and the byte 0x0d, instead of spinning or comparing garbage. Not part of make test (it
rewrites tracked files): run from the repository root after `make`; every file gets its bytes back after its test.

  python3 tests/hardening/crlf_proof.py build/linux-x86_64"""
import subprocess, sys, time
B = sys.argv[1]
BIN = sys.argv[2] if len(sys.argv) > 2 else B + "/tests"
CASES = [("test_tiktoken", "tests/data/kimi/bpe_cases.txt"), ("test_tiktoken", "tests/data/kimi/expect.txt"),
         ("test_compile", "tests/data/compile/refuse.txt"), ("test_compile", "tests/data/compile/gpt2style.expect"),
         ("test_load", "tests/fuzz/regress/expect.txt"), ("test_norm", "tests/norm/nfc_golden.txt"),
         ("test_norm", "tests/norm/nfkc_golden.txt"), ("test_spm", "tests/data/spm/refuse.txt"),
         ("test_spm", "tests/data/spm/gemma4like.expect"), ("test_targets", "tests/data/targets/ledger.txt")]
bad = 0
for t, f in CASES:
    b = open(f, "rb").read()
    open(f, "wb").write(b.replace(b"\n", b"\r\n"))
    t0 = time.time()
    try:
        r = subprocess.run([f"{BIN}/{t}"], capture_output=True, text=True, errors="replace", timeout=60)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    dt = time.time() - t0
    open(f, "wb").write(b)
    line = next((l for l in out.splitlines() if l.startswith(f"FAIL {f}:")), "")
    ok = rc not in (0, "timeout") and "0x0d" in line
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'}  {t} on CRLF {f}: exit {rc} in {dt:.2f} s: {line or next((l for l in out.splitlines() if 'FAIL' in l), '')}")
sys.exit(1 if bad else 0)
