#!/usr/bin/env python3
"""Big K3 fuzz: kernels.md §3 A1-A7 model vs hf Split/ByteLevel, per variant.
Run on a benchmark machine: uv run --with tokenizers==0.23.2 python run_k3_fuzz.py N_SHORT N_LONG"""
import json, multiprocessing as mp, os, random, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toks_model as M

VARIANTS = ["gpt2", "cl100k", "qwen2", "qwen35"]
N_SHORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
N_LONG = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
CHUNK = 4000

def worker(args):
    variant, seed, n, long_mode = args
    M.PROBE = M.run_probes()
    rng = random.Random(seed)
    bad = 0
    examples = []
    for i in range(n):
        s = M.gen_long(rng) if long_mode else M.gen_case(rng)
        want = M.hf_pieces(variant, s)
        got = M.k3_scan(s, variant)
        if want != got:
            bad += 1
            if len(examples) < 3:
                examples.append((s, want, got))
    return bad, n, examples

def main():
    M.PROBE = M.run_probes()  # also caches for workers
    # class-table cross-check: doc classes (onig data via probes) vs python unicodedata
    ud_delta = {"L": [], "N": [], "M": [], "s": []}
    for name, cats in [("L", ("Lu","Ll","Lt","Lm","Lo")), ("N", ("Nd","Nl","No")),
                       ("M", ("Mn","Mc","Me"))]:
        for cp in sorted(getattr(M.PROBE, name) if False else M.PROBE[name]):
            cat = unicodedata.category(chr(cp))
            if cat not in cats:
                ud_delta[name].append(cp)
    for cp in sorted(M.PROBE["s"]):
        if not (chr(cp).isspace() or cp in M.UNICODE_WS):
            ud_delta["s"].append(cp)
    ws_probe = sorted(M.PROBE["s"])
    print(f"probed \\s count={len(ws_probe)} == doc 25-list: {ws_probe == sorted(M.UNICODE_WS)}")
    print(f"unicodedata-vs-onig deltas: L={len(ud_delta['L'])} N={len(ud_delta['N'])} "
          f"M={len(ud_delta['M'])} s={len(ud_delta['s'])} "
          f"first L deltas: {[hex(c) for c in ud_delta['L'][:5]]}", flush=True)

    total_bad = 0
    total_n = 0
    for variant in VARIANTS:
        tasks = []
        seed0 = random.randrange(1 << 30)
        for st in range(0, N_SHORT, CHUNK):
            tasks.append((variant, seed0 + st, min(CHUNK, N_SHORT - st), False))
        for st in range(0, N_LONG, CHUNK):
            tasks.append((variant, seed0 + 777_000_000 + st, min(CHUNK, N_LONG - st), True))
        bad = n = 0
        shown = 0
        with mp.Pool(64) as pool:
            for b, cnt, ex in pool.imap_unordered(worker, tasks, chunksize=1):
                bad += b; n += cnt
                for (s, want, got) in ex:
                    if shown < 2:
                        print(f"[{variant}] MISMATCH {s!r}\n  hf : {want}\n  doc: {got}", flush=True)
                        shown += 1
        print(f"RESULT {variant}: {n - bad}/{n} ok, {bad} mismatches "
              f"({N_SHORT} gen_case + {N_LONG} gen_long)", flush=True)
        total_bad += bad; total_n += n
    print(f"TOTAL: {total_n - total_bad}/{total_n} ok, {total_bad} mismatches", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_fuzz_done.json"), "w") as f:
        json.dump({"n_short": N_SHORT, "n_long": N_LONG, "variants": VARIANTS,
                   "total": total_n, "bad": total_bad,
                   "ws_eq_doc25": ws_probe == sorted(M.UNICODE_WS),
                   "ud_delta": {k: len(v) for k, v in ud_delta.items()}}, f)

if __name__ == "__main__":
    main()
