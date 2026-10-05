#!/usr/bin/env python3
"""tests/data/breadth/facts.py: what the census's byte-level files actually contain, for docs/breadth.md.

    uv run --with tokenizers==0.23.2 python tests/data/breadth/facts.py [--files DIR] [--json OUT]

reads targets.json (next to this file: census tokenizer id -> file sha-256, features, models) and the census's
file cache (default ~/.cache/toks/census/coverage/files, on the lab host that ran the census) and prints, per
tokenizer: the byte-level alphabet chars the vocab lacks, unk / byte_fallback / fuse_unk, the vocab strings
outside the alphabet, every added token with an option set (content escaped: tool output strips <|...|>),
the post-processor, padding, dropout and the pre-tokenizer chain. With --json the same facts go to OUT.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b)
        _cs.append(256 + _n)
        _n += 1
B2U = {b: chr(c) for b, c in zip(_bs, _cs)}
U2B = {u: b for b, u in B2U.items()}


def esc(s):
    return s.encode("unicode_escape").decode("ascii")


def facts(path):
    j = json.load(open(path, encoding="utf-8"))
    m = j["model"]
    vocab = m.get("vocab", {})
    have = {U2B[s] for s in vocab if len(s) == 1 and s in U2B}
    missing = sorted(set(range(256)) - have)
    outside = [s for s in vocab if not all(c in U2B for c in s)]
    opts = []
    for t in j.get("added_tokens") or []:
        if t.get("lstrip") or t.get("rstrip") or t.get("single_word") or t.get("normalized"):
            opts.append({k: t[k] for k in ("id", "special", "single_word", "lstrip", "rstrip", "normalized")}
                        | {"content": esc(t["content"]), "in_vocab": t["content"] in vocab})
    return {
        "missing_bytes": [f"{b:02x}" for b in missing],
        "unk_token": m.get("unk_token"), "byte_fallback": m.get("byte_fallback"), "fuse_unk": m.get("fuse_unk"),
        "dropout": m.get("dropout"), "ignore_merges": m.get("ignore_merges"),
        "n_vocab": len(vocab), "n_outside": len(outside), "outside_examples": [esc(s) for s in outside[:8]],
        "added_with_options": opts, "n_added": len(j.get("added_tokens") or []),
        "normalizer": j.get("normalizer"), "pre_tokenizer": j.get("pre_tokenizer"),
        "post_processor": j.get("post_processor"), "decoder": j.get("decoder"),
        "padding": j.get("padding"), "truncation": j.get("truncation"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=os.path.expanduser("~/.cache/toks/census/coverage/files"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma-separated tokenizer ids")
    a = ap.parse_args()
    targets = json.load(open(os.path.join(HERE, "targets.json")))
    out = {}
    for tid, t in sorted(targets.items(), key=lambda kv: -kv[1]["dl_b"]):
        if a.only and tid not in a.only.split(","):
            continue
        p = os.path.join(a.files, "sha256_" + t["file"])
        if not os.path.exists(p):
            print(f"{tid}: file missing ({p})")
            continue
        f = facts(p)
        out[tid] = f | {"features": t["features"], "models": t["models"], "file": t["file"]}
        print(f"== {tid} {t['models'][0]} {t['features']}")
        print(f"   missing {len(f['missing_bytes'])}: {' '.join(f['missing_bytes'])}")
        print(f"   unk {f['unk_token']!r} byte_fallback {f['byte_fallback']} fuse_unk {f['fuse_unk']} "
              f"dropout {f['dropout']} vocab {f['n_vocab']} outside {f['n_outside']} {f['outside_examples'][:4]}")
        for o in f["added_with_options"][:6]:
            print(f"   added {o}")
        if len(f["added_with_options"]) > 6:
            print(f"   ... {len(f['added_with_options'])} added tokens with options")
        pp = f["post_processor"]
        if pp and pp.get("type") != "ByteLevel":
            print(f"   post_processor {esc(json.dumps(pp)[:300])}")
        if f["padding"]:
            print(f"   padding {f['padding']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
