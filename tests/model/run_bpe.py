#!/usr/bin/env python3
"""§5 K6 bpe model (linear lowest-rank-leftmost, ignore_merges via vocab, dup pairs last)
vs hf BPE on byte-level-mapped pieces, for gpt2 / llama3 (ignore_merges) / qwen3.
Pieces: real-text-shaped substrings + random byte strings up to ~300 bytes."""
import json, multiprocessing as mp, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toks_model as M
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
CHUNK = 2000

CORPUS = (
    "The quick brown fox jumps over the lazy dog. Isn't it wonderful? I'd say so; we've "
    "seen 12,345 instances since 2019-10-03. def tokenize(x): return x.split(' ')  # comment\n"
    "naïve café résumé Zürich Straße烟火 东京タワー 한성 Ελληνικά русский العربية עברית "
    "emoji 👍🏽🔥 and ZWJ-family 👨‍👩‍👧‍👦 done. <html><body class='x'>1 &amp; 2</body></html>\n"
    "\t\tindented\tcode\there\n\n\nparagraphs\r\nwindows\rline\u00a0nbsp\u3000ideographic"
    "01234567890123456789 sums 3.14159 * 2.71828 = 8.53973422267 foreign digits ٠١٢٣ ①②③"
)

TOKS = {}
def load(name, fname):
    tk, bpe = M.bpe_only_tokenizer(os.path.join(HERE, fname))
    return tk, bpe

def worker(args):
    which, seed, n = args
    rng = random.Random(seed)
    tk, bpe = TOKS[which]
    bad = 0
    examples = []
    for i in range(n):
        r = rng.random()
        if r < 0.55:
            # real-text-shaped piece: random slice of corpus (bytes), decoded lossless via latin1 trick
            a = rng.randrange(0, len(CORPUS))
            b_ = rng.randrange(a, min(a + 300, len(CORPUS)))
            piece = CORPUS[a:b_].encode("utf-8")
        else:
            # random bytes up to 300, skewed to ascii-heavy and to valid-utf8-looking
            n_ = rng.randint(1, 300)
            if rng.random() < 0.5:
                piece = bytes(rng.randint(0x20, 0x7E) if rng.random() < 0.85 else rng.randint(0, 255)
                              for _ in range(n_))
            else:
                piece = bytes(rng.randint(0, 255) for _ in range(n_))
        mapped = "".join(M.BYTES_CHAR[b] for b in piece)
        try:
            want = tk.encode(mapped, add_special_tokens=False).ids
        except Exception as e:
            continue
        got = bpe.tokenize_bytes(piece)
        if want != got:
            bad += 1
            if len(examples) < 3:
                examples.append((piece, want, got))
    return bad, n, examples

def main():
    for which, fname in [("gpt2", "openai-community_gpt2.json"),
                         ("llama3", "unsloth_Llama-3.2-1B-Instruct.json"),
                         ("qwen3", "Qwen_Qwen3-0.6B.json")]:
        TOKS[which] = load(which, fname)
    total_bad = total_n = 0
    for which in TOKS:
        tasks = [(which, random.randrange(1 << 30) + st, min(CHUNK, N - st))
                 for st in range(0, N, CHUNK)]
        bad = n = 0
        shown = 0
        with mp.Pool(64) as pool:
            for b, cnt, ex in pool.imap_unordered(worker, tasks, chunksize=1):
                bad += b; n += cnt
                for (piece, want, got) in ex:
                    if shown < 2:
                        print(f"[{which}] MISMATCH piece={piece!r}\n  hf : {want}\n  doc: {got}", flush=True)
                        shown += 1
        print(f"RESULT bpe {which}: {n - bad}/{n} ok, {bad} mismatches", flush=True)
        total_bad += bad; total_n += n
    print(f"TOTAL bpe: {total_n - total_bad}/{total_n} ok, {total_bad} mismatches", flush=True)
    with open(os.path.join(HERE, "bpe_done.json"), "w") as f:
        json.dump({"n_per_tokenizer": N, "total": total_n, "bad": total_bad}, f)

if __name__ == "__main__":
    main()
