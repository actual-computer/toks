#!/usr/bin/env python3
"""tests/model/mutants_kimi.py: give the kimi fuzz teeth. Each mutant is one small change to
tests/model/kimi_model.py (a rule of docs/models/kimi.md broken on purpose); run_kimi_fuzz.py must report
mismatches against the reference for every one of them.

    tools/oracle/py.sh tests/model/mutants_kimi.py --jobs 20
Mutated copies go to build/kimi/mut/<n>/kimi_model.py; the fuzz imports them first (TOKS_KIMI_MODEL_DIR).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

MUTANTS = [
    ("no Han run (K1)", "            if c in han:\n                e = i\n", "            if False:\n                e = i\n"),
    ("tail takes '/' (o200k)", "                while e < n and base[cps[e]] == NL:\n                    e += 1\n",
     "                while e < n and (base[cps[e]] == NL or cps[e] == 0x2F):\n                    e += 1\n"),
    ("Han in the case classes", "        self.up, self.lo, self.han = pr[\"UP\"], pr[\"LO\"], Hs",
     "        self.up, self.lo, self.han = pr[\"UP\"] | Hs, pr[\"LO\"] | Hs, Hs"),
    ("cuts use \\s, not str.isspace", "ISSPACE = frozenset([*range(0x09, 0x0E), *range(0x1C, 0x21),",
     "ISSPACE = frozenset([*range(0x09, 0x0E), *range(0x20, 0x21),"),
    ("run limit 25,001", "RUN_CHARS = 25_000", "RUN_CHARS = 25_001"),
    ("chunk 400,001", "CHUNK_CHARS = 400_000", "CHUNK_CHARS = 400_001"),
    ("serving without the trie", "            for is_tok, t in self.trie_split(s):",
     "            for is_tok, t in [(False, s)]:"),
    ("trie = all 256 specials", "        self._trie_re = re.compile(\"|\".join(map(re.escape, self.trie)))",
     "        self._trie_re = re.compile(\"|\".join(map(re.escape, self.specials)))\n        self.trie = dict(self.specials)"),
    ("bpe ties to the rightmost", "heapq.heappush(heap, (rk, i, end[j]))", "heapq.heappush(heap, (rk, -i, end[j]))"),
    # (L1(i) before L1(i + 1) is an equivalent mutant: when both apply they end at the same atom)
    ("L2(i+1) before L1(i)", "for f in (lambda: L1(i + 1), lambda: L1(i), lambda: L2(i + 1)):",
     "for f in (lambda: L1(i + 1), lambda: L2(i + 1), lambda: L1(i)):"),
    ("4 digits", "while e < n and e < i + 3 and base[cps[e]] == N:", "while e < n and e < i + 4 and base[cps[e]] == N:"),
    # (dropping the whole-piece lookup is equivalent for Kimi's file: every one of its 163,328 multi-byte
    # tokens is its own bpe, tools/oracle/selfmerge.py; test_tiktoken.c's hand vocabulary gives that rule teeth)
    ("specials not recognized in direct", "        if mode == \"direct\":\n            return self.wrapper(s, True)",
     "        if mode == \"direct\":\n            return self.wrapper(s, False)"),
    ("K6 keeps the last NL out", "                if t >= i:\n                    e = t + 1                   # K5",
     "                if t > i:\n                    e = t                       # K5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--short", type=int, default=20000)
    ap.add_argument("--long", type=int, default=40)
    args = ap.parse_args()
    src = open(os.path.join(HERE, "kimi_model.py")).read()
    survived = 0
    for k, (name, a, b) in enumerate(MUTANTS, 1):
        if src.count(a) != 1:
            print(f"m{k} {name}: pattern found {src.count(a)} times", flush=True)
            survived += 1
            continue
        d = os.path.join(ROOT, "build", "kimi", "mut", str(k))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "kimi_model.py"), "w") as f:
            f.write(src.replace(a, b, 1))
        out = os.path.join(d, "fuzz.json")
        env = dict(os.environ, TOKS_KIMI_MODEL_DIR=d,
                   TOKS_KIMI_PROBES=os.path.join(ROOT, "build", "kimi", "kimi_probes.pkl"))
        cmd = [sys.executable, os.path.join(HERE, "run_kimi_fuzz.py"), "--jobs", str(args.jobs), "--short", str(args.short),
               "--exh", "3", "--long", str(args.long), "--vocab", "2000", "--seed", f"mut{k}", "--out", out]
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            res = json.load(open(out))
            bad = res["mismatches"]
        except (OSError, ValueError, KeyError):
            bad = -1
        if bad and bad > 0:
            print(f"m{k} killed ({bad} mismatches): {name}", flush=True)
        else:
            survived += 1
            print(f"m{k} SURVIVED ({bad}): {name}", flush=True)
    print(f"{len(MUTANTS) - survived} of {len(MUTANTS)} killed")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
