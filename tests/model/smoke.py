#!/usr/bin/env python3
"""Quick local smoke: verify pattern strings against real hub tokenizer.json files
(incl. Qwen 3.5), then small-scale A1-A7 comparison so the big fuzz is trusted."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toks_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

def report(pattern_name, expected, actual):
    same = expected == actual
    print(f"{pattern_name:28s} {'MATCH' if same else 'DIFFER'}")
    if not same:
        print("  expected:", expected)
        print("  actual  :", actual)
    return same

for repo, fname in [("Qwen/Qwen3.5-0.8B", "Qwen_Qwen3.5-0.8B.json"),
                    ("Qwen/Qwen3.5-4B", "Qwen_Qwen3.5-4B.json"),
                    ("Qwen/Qwen2.5-0.5B", "Qwen_Qwen2.5-0.5B.json"),
                    ("openai-community/gpt2", "openai-community_gpt2.json"),
                    ("unsloth/Llama-3.2-1B-Instruct", "unsloth_Llama-3.2-1B-Instruct.json")]:
    if not os.path.exists(fname):
        os.system(f"curl -sL -o {fname} https://huggingface.co/{repo}/resolve/main/tokenizer.json")
    try:
        j = json.load(open(fname))
    except Exception as e:
        print(f"{repo}: FETCH/PARSE FAIL {e}")
        continue
    pt = j.get("pre_tokenizer")
    seq = pt.get("pretokenizers") if pt and pt.get("type") == "Sequence" else ([pt] if pt else [])
    pats = [p.get("pattern", {}).get("Regex") for p in seq if p.get("type") == "Split"]
    print(f"===== {repo} (added {len(j.get('added_tokens') or [])})")
    for p in pats:
        report("  pattern", None, p)

# small A1-A7 sanity: 3000 random cases per variant
M.PROBE = M.run_probes()
rng = random.Random(42)
for variant in ["gpt2", "cl100k", "qwen2", "qwen35"]:
    bad = 0
    for i in range(3000):
        s = M.gen_case(rng) if i % 3 else M.gen_case(rng, "ws")
        want = M.hf_pieces(variant, s)
        got = M.k3_scan(s, variant)
        if want != got:
            bad += 1
            if bad <= 3:
                print(f"[{variant}] MISMATCH {s!r}\n  hf : {want}\n  doc: {got}")
    print(f"variant {variant}: {3000 - bad}/3000 ok, {bad} mismatches")

# apostrophe lookahead probe: does (?i:...) fold anything else onto the contraction set?
fs = M.fold_sets()
for k, v in fs.items():
    nz = sorted(c for c in v if c > 0x7F)
    print(f"(?i:{k}) matches {len(v)} cps; non-ascii: {[hex(c) for c in nz[:12]]}"
          + (" ..." if len(nz) > 12 else ""))
# byte-offset convention guard
demo = "é中\U0001F600a"
print("Split offsets demo:", M.pre_tokenizers.Split(M.Regex(r"\p{L}"), behavior="isolated").pre_tokenize_str(demo))
