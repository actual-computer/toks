#!/usr/bin/env python3
"""tests/unigram/mutants.py: give the unigram differential teeth. Each mutant is a one-line change to
tests/model/unigram_model.py that breaks one rule of docs/algorithms/unigram.md; run_diff.py over the pins or
synth.py over synthetic tokenizers must report mismatches for it (a surviving mutant is a rule no generator
exercises, or an equivalent mutant: docs/algorithms/unigram.md §11 lists both). Mutated copies live in
build/unigram-mut/<name>/. Heavy: run on a lab host.

  uv run --with tokenizers==0.23.2 tests/unigram/mutants.py [--n 4000] [--workers 6] [names...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

MUTANTS = {
    # Viterbi
    "tie_ge": ("if frm[e] is None or cand > best[e]:\n                        best[e] = cand",
               "if frm[e] is None or cand >= best[e]:\n                        best[e] = cand"),
    "unk_pen9": ("self.unk_score = self.min_score - 10.0", "self.unk_score = self.min_score - 9.0"),
    "unk_single_nocheck": ("                    if e == s + 1:\n                        single = True",
                           "                    if True:\n                        single = True"),
    "no_fuse": ("while j + 1 < len(path) and path[j + 1][2] == self.unk_id:", "while False:"),
    "bytefb_off": ("            if self.byte_fallback:\n                bt", "            if False:\n                bt"),
    "dup_first": ("            self.tok2id[p] = i  # duplicates", "            self.tok2id.setdefault(p, i)  # duplicates"),
    "float_cr": ("    f = float(sig)\n", "    return float(text)\n    f = float(sig)\n"),
    # Precompiled + graphemes
    "grapheme_off": ("if len(gb) < 6:", "if False:"),
    "lt6_le6": ("if len(gb) < 6:", "if len(gb) <= 6:"),
    "longest_key": ("                res = a[node] & ((1 << 31) - 1)\n                break",
                    "                res = a[node] & ((1 << 31) - 1)"),
    "nul_nobreak": ("            if c == 0:\n                break\n            node ^= c", "            node ^= c"),
    "crlf_split": ("if b == \"CR\" and a == \"LF\":\n            return False", "if False:\n            return False"),
    "gb9a_off": ("if a == \"SpacingMark\":\n            return False", "if False:\n            return False"),
    "gb9b_off": ("if b == \"Prepend\":\n            return False", "if False:\n            return False"),
    "gb9c_off": ("if a == \"InCB_Consonant\":", "if False:"),
    "gb11_off": ("if b == \"ZWJ\" and a == \"Extended_Pictographic\":", "if False:"),
    "ri_parity": ("return n % 2 == 0", "return n % 2 == 1"),
    # other normalizers, pre-tokenizers, added tokens, decoders
    "collapse_off": ("return lambda s: re.sub(\" {2,}\", content, s)", "return lambda s: s"),
    "ws_nbsp": ("0x85, 0xA0, 0x1680", "0x85, 0x1680"),
    "prepend_never": ("if self.scheme == \"always\" or (self.scheme == \"first\" and at_original_start):", "if False:"),
    "lstrip_off": ("            a = max(_trailing_ws_start(text, a), start_offset)", "            pass"),
    "rstrip_off": ("            b = _leading_ws(text, b)", "            pass"),
    "resume_start1": ("        i = b  # resume at the raw match end", "        i = a + 1"),
    "meta_dec_first": ("(\"\" if (k == 0 and m.scheme != \"never\") else \" \")", "(\" \")"),
}

TOKS = "uni_bgem3,uni_pmminilm,uni_bgererank,uni_llmjp3,uni_t5base,uni_ruri3,uni_mxbaixs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tok", default=TOKS)
    ap.add_argument("--synth", type=int, default=600, help="synthetic tokenizers per mutant (synth.py)")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    src = open(os.path.join(ROOT, "tests", "model", "unigram_model.py"), encoding="utf-8").read()
    runner = open(os.path.join(HERE, "run_diff.py"), encoding="utf-8").read()
    synth = open(os.path.join(HERE, "synth.py"), encoding="utf-8").read()
    table = os.path.join(ROOT, "tests", "data", "unigram", "grapheme17.txt")
    survivors = []
    for name in a.names or list(MUTANTS):
        old, new = MUTANTS[name]
        if src.count(old) < 1:
            raise SystemExit(f"mutant {name}: anchor not found (the model changed)")
        d = os.path.join(ROOT, "build", "unigram-mut", name)
        os.makedirs(d, exist_ok=True)
        m = src.replace(old, new, 1).replace(
            'GRAPHEME_TABLE = os.path.join(HERE, "..", "data", "unigram", "grapheme17.txt")',
            f"GRAPHEME_TABLE = {table!r}")
        with open(os.path.join(d, "unigram_model.py"), "w", encoding="utf-8") as f:
            f.write(m)
        for fn in ("gen.py", "fetch.py", "pins.json"):
            shutil.copy(os.path.join(HERE, fn), os.path.join(d, fn))
        for fn, text in (("run_diff.py", runner), ("synth.py", synth)):
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                f.write(text.replace('sys.path.insert(0, os.path.join(HERE, "..", "model"))', "sys.path.insert(0, HERE)"))
        out = os.path.join(d, "mismatches.jsonl")
        if os.path.exists(out):
            os.remove(out)
        p = subprocess.run([sys.executable, os.path.join(d, "run_diff.py"), "--tok", a.tok, "--kind", "random",
                            "--n", str(a.n), "--workers", str(a.workers), "--out", out],
                           capture_output=True, text=True)
        per = {}
        for line in p.stdout.splitlines():
            if line.startswith("SUMMARY "):
                s = json.loads(line[8:])
                per[s["tok"]] = s["mismatch"]
        q = subprocess.run([sys.executable, os.path.join(d, "synth.py"), "--tokenizers", str(a.synth),
                            "--workers", str(a.workers)], capture_output=True, text=True)
        for line in q.stdout.splitlines():
            if line.startswith("SUMMARY "):
                s = json.loads(line[8:])
                per["synthetic"] = s["mismatch"] + s["load_mismatch"]
        if "synthetic" not in per:
            per["synthetic"] = 1  # the mutant crashed the synthetic run: detected
        killed = sum(per.values()) > 0
        if not killed:
            survivors.append(name)
        print(f"{name:20s} {'killed' if killed else 'SURVIVED'} {json.dumps(per)}", flush=True)
        if not per:
            print(p.stderr[-1500:], flush=True)
    print(f"mutants: {len(a.names or MUTANTS)}, survivors: {survivors}")


if __name__ == "__main__":
    main()
