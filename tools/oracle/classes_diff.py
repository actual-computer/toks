#!/usr/bin/env python3
"""tools/oracle/classes_diff.py: tiktoken's character classes (Kimi K3's pattern, probed through the tiktoken
0.14.0 wheel over every scalar value) against toks' class tables (src/core/classes.c, onig data of hf
tokenizers 0.23.2), plus the Han set the pattern needs and toks does not carry yet.

    uv run --python 3.12 --with tiktoken==0.14.0 python tools/oracle/classes_diff.py [--onig-han FILE]

Needs build/kimi/toks_classes.bin (tools/oracle/dump_classes.c's output); --onig-han FILE is a json list of
the code points hf 0.23.2's onig matches with [\\p{Han}] (made by --probe-onig-han under
uv run --with tokenizers==0.23.2). Prints every disagreement by code point and writes build/kimi/han.json
(the Han set as ranges) and build/kimi/classes_diff.json.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tests", "model"))

C_BASE, C_UPPER, C_LOWER, C_MARK, C_FOLD_S = 0x07, 0x08, 0x10, 0x20, 0x40   # layout.h
BASE_NAME = {0: "P", 1: "L", 2: "N", 3: "WS", 4: "NL"}


def ranges(cps):
    out, s = [], sorted(cps)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        out.append((s[i], s[j]))
        i = j + 1
    return out


def fmt_ranges(rs):
    return ", ".join(f"U+{a:04X}" if a == b else f"U+{a:04X}..U+{b:04X}" for a, b in rs)


def probe_onig_han(path):
    from tokenizers import Regex, pre_tokenizers
    sp = pre_tokenizers.Split(Regex(r"[\p{Han}]"), behavior="removed")
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    got = set()
    B = 4096
    for st in range(0, len(cps), B):
        sub = cps[st:st + B]
        s = "\x00".join(map(chr, sub))
        kept = set()
        for _t, (a, b) in sp.pre_tokenize_str(s):
            for k in range(a, b):
                kept.add(k)
        for idx, c in enumerate(sub):
            if 2 * idx not in kept:
                got.add(c)
    with open(path, "w") as f:
        json.dump(sorted(got), f)
    print(f"onig [\\p{{Han}}]: {len(got)} code points -> {path}")


def main():
    if "--probe-onig-han" in sys.argv:
        probe_onig_han(sys.argv[sys.argv.index("--probe-onig-han") + 1])
        return
    import kimi_model as km
    pr = km.run_probes()
    cl = km.Classes(pr)
    with open(os.path.join(ROOT, "build", "kimi", "toks_classes.bin"), "rb") as f:
        toks = f.read()
    assert len(toks) == 0x110000
    scalars = km.SCALARS
    lu, ll, lt, lm, lo, mk = pr["Lu"], pr["Ll"], pr["Lt"], pr["Lm"], pr["Lo"], pr["M"]
    up_all, lo_all = lu | lt | lm | lo | mk, ll | lm | lo | mk
    diffs = {"base": [], "upper": [], "lower": [], "mark": [], "fold_s": []}
    for cp in scalars:
        t = toks[cp]
        if (t & C_BASE) != cl.base[cp]:
            diffs["base"].append((cp, BASE_NAME[t & C_BASE], BASE_NAME[cl.base[cp]]))
        if bool(t & C_UPPER) != (cp in up_all):
            diffs["upper"].append((cp, bool(t & C_UPPER), cp in up_all))
        if bool(t & C_LOWER) != (cp in lo_all):
            diffs["lower"].append((cp, bool(t & C_LOWER), cp in lo_all))
        if bool(t & C_MARK) != (cp in mk):
            diffs["mark"].append((cp, bool(t & C_MARK), cp in mk))
        fs = cp in pr["fold_s"] and cp not in (0x73, 0x53)
        if bool(t & C_FOLD_S) != fs:
            diffs["fold_s"].append((cp, bool(t & C_FOLD_S), fs))
    print(f"tiktoken {pr['tiktoken']} (regex-syntax 0.8.11, unicode 16.0.0) vs toks classes (onig, hf 0.23.2):")
    for k, v in diffs.items():
        print(f"  {k:7s} {len(v)} code points differ" + (": " + ", ".join(
            f"U+{cp:04X} toks={a} tiktoken={b}" for cp, a, b in v[:40]) if v else ""))
    han = sorted(pr["H"])
    by_cat = {}
    for cp in han:
        by_cat.setdefault(unicodedata.category(chr(cp)), []).append(cp)
    print(f"Han (tiktoken [\\p{{Han}}]): {len(han)} code points in {len(ranges(han))} ranges")
    for cat, v in sorted(by_cat.items()):
        print(f"  {cat}: {len(v):6d}  {fmt_ranges(ranges(v))[:400]}")
    print(f"  (python {sys.version.split()[0]} unicodedata {unicodedata.unidata_version} for the categories)")
    base_of_han = {}
    for cp in han:
        base_of_han.setdefault(BASE_NAME[cl.base[cp]], 0)
        base_of_han[BASE_NAME[cl.base[cp]]] += 1
    print("  base class of the Han code points:", base_of_han)
    out = {"han_ranges": ranges(han), "diffs": {k: [list(x) for x in v] for k, v in diffs.items()}}
    if "--onig-han" in sys.argv:
        with open(sys.argv[sys.argv.index("--onig-han") + 1]) as f:
            onig = set(json.load(f))
        a, b = sorted(onig - set(han)), sorted(set(han) - onig)
        print(f"onig [\\p{{Han}}] {len(onig)} vs tiktoken {len(han)}: onig-only {len(a)} {fmt_ranges(ranges(a))[:300]}"
              f"; tiktoken-only {len(b)} {fmt_ranges(ranges(b))[:300]}")
        out["onig_han_only"], out["tiktoken_han_only"] = a, b
    os.makedirs(os.path.join(ROOT, "build", "kimi"), exist_ok=True)
    with open(os.path.join(ROOT, "build", "kimi", "classes_diff.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
