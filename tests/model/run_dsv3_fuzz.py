#!/usr/bin/env python3
"""The dsv3 proof runs: tests/model/dsv3_model.py (docs/templates/dsv3.md transcribed) vs hf 0.23.2.

  uv run --python 3.12 --with tokenizers==0.23.2 python tests/model/run_dsv3_fuzz.py [options]

    --short N         gen_case strings                                   (default 3000)
    --long N          gen_long strings (60..1000 chars)                  (default 300)
    --exh L           every string of length 0..L over dsv3_model.REP    (default 3)
    --exh-small L     every string of length 0..L over REP_SMALL         (default 0)
    --vocab N         N strings of 1..30 vocabulary tokens of the pinned file
    --real DIR        every real-text file under DIR: each line, each paragraph, random chunks of 1..16384 chars
    --fetch DIR       first fetch a real-text corpus into DIR (FLORES-200, python's stdlib, clang's headers)
    --jobs J --seed S --seq-every K (also compare the three Splits built from the model's constants, without
                      ByteLevel, with the file's pre-tokenizer on every K-th case; default 7)

The oracle is the pinned tokenizer.json's own pre_tokenizer (dsv3_model.hf_pieces). Prints one RESULT line per
kind and a TOTAL; writes build/dsv3_fuzz_summary.json. Exit 1 on any mismatch. Mismatch examples are printed
escaped (tool output strips some sequences).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv3_model as M  # noqa: E402

CHUNK = 2000


def esc(s):
    return s.encode("unicode_escape").decode("ascii")


# ---------------------------------------------------------------- case sources (deterministic per task)
def cases(task):
    kind, a, b, seed = task
    if kind == "short":
        rng = random.Random(seed)
        for _ in range(b - a):
            yield M.gen_case(rng)
    elif kind == "long":
        rng = random.Random(seed)
        for _ in range(b - a):
            yield M.gen_long(rng)
    elif kind in ("exh", "exh_small"):
        alpha = M.REP if kind == "exh" else M.REP_SMALL
        length, lo, hi = seed, a, b                 # seed carries the length for exhaustive tasks
        k = len(alpha)
        for idx in range(lo, hi):
            out = []
            x = idx
            for _ in range(length):
                out.append(alpha[x % k])
                x //= k
            yield "".join(out)
    elif kind == "vocab":
        voc = M.vocab_strings()
        rng = random.Random(seed)
        for _ in range(b - a):
            yield "".join(rng.choice(voc) for _ in range(rng.randint(1, 30)))
    elif kind == "real":
        with open(a, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        yield from M.segments_of(text, random.Random(seed))
    else:
        raise ValueError(kind)


def worker(task):
    kind, seq_every = task[0], task[4]
    M.init()
    bad = n = sq_bad = sq_n = nbytes = npieces = 0
    examples = []
    for s in cases(task[:4]):
        want = M.hf_pieces(s)
        got = M.k3_scan(s)
        n += 1
        nbytes += want[-1][1] if want else 0
        npieces += len(want)
        if want != got:
            bad += 1
            if len(examples) < 3:
                examples.append((s, want, got))
        if seq_every and n % seq_every == 0:
            sq_n += 1
            if M.seq3_pieces(s) != want:
                sq_bad += 1
                if len(examples) < 3:
                    examples.append((s, want, "the model's three Splits differ from the file's pre-tokenizer"))
    return kind, bad, n, sq_bad, sq_n, nbytes, npieces, examples


# ---------------------------------------------------------------- real-text corpus
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"   # 204 languages, CC-BY-SA 4.0


def fetch(dirpath):
    """FLORES-200 dev + devtest (one file per language and split), python's stdlib, clang's headers."""
    import io
    import sysconfig
    import tarfile
    import urllib.request
    os.makedirs(dirpath, exist_ok=True)
    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "toks-dsv3-fuzz/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    out = os.path.join(dirpath, "flores")
    os.makedirs(out, exist_ok=True)
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith((".dev", ".devtest")):
                body = tf.extractfile(m).read()
                with open(os.path.join(out, os.path.basename(m.name) + ".txt"), "wb") as f:
                    f.write(body)
                total += len(body)
    for name, root, exts, cap in (("code-python.txt", sysconfig.get_paths()["stdlib"], (".py",), 24 << 20),
                                  ("code-c.txt", os.path.expanduser("~/.cache/toks-llvm/21.1.8/lib/clang"),
                                   (".h",), 8 << 20)):
        acc, size = [], 0
        for dp, _dn, fns in sorted(os.walk(root)):
            for fn in sorted(fns):
                if fn.endswith(exts) and size < cap:
                    try:
                        with open(os.path.join(dp, fn), "rb") as f:
                            acc.append(f.read())
                        size += len(acc[-1])
                    except OSError:
                        pass
        with open(os.path.join(dirpath, name), "wb") as f:
            f.write(b"\n".join(acc))
        total += size
    print("fetch: %d bytes in %s" % (total, dirpath), file=sys.stderr)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=int, default=3000)
    ap.add_argument("--long", type=int, default=300)
    ap.add_argument("--exh", type=int, default=3)
    ap.add_argument("--exh-small", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--real", default=None)
    ap.add_argument("--fetch", default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seq-every", type=int, default=7)
    args = ap.parse_args()

    import tokenizers
    print("oracle: tokenizers %s; python %s" % (tokenizers.__version__, sys.version.split()[0]), flush=True)
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("tokenizers %s != 0.23.2 (the pinned oracle)" % tokenizers.__version__)
    for line in M.check_tokenizer_file():
        print(line, flush=True)
    t0 = time.time()
    M.init()
    print("probes (%s); class claims of docs/templates/dsv3.md §2 hold on every scalar (%.1fs)"
          % (M.PROBE_PATH, time.time() - t0), flush=True)
    for line in M.FACTS:
        print("  " + line, flush=True)
    if args.fetch:
        fetch(args.fetch)
        args.real = args.real or args.fetch

    seed0 = args.seed if args.seed is not None else random.randrange(1 << 30)
    print("seed %d" % seed0, flush=True)
    tasks = []
    for st in range(0, args.short, CHUNK):
        tasks.append(("short", st, min(st + CHUNK, args.short), seed0 + st, args.seq_every))
    for st in range(0, args.long, CHUNK // 10):
        tasks.append(("long", st, min(st + CHUNK // 10, args.long), seed0 + 10**9 + st, args.seq_every))
    for kind, alpha, lmax in (("exh", M.REP, args.exh), ("exh_small", M.REP_SMALL, args.exh_small)):
        for length in range(0, lmax + 1):
            total = len(alpha) ** length
            for st in range(0, total, 20000):
                tasks.append((kind, st, min(st + 20000, total), length, args.seq_every))
    for st in range(0, args.vocab, CHUNK):
        tasks.append(("vocab", st, min(st + CHUNK, args.vocab), seed0 + 2 * 10**9 + st, args.seq_every))
    if args.real:
        for dp, _dn, fns in sorted(os.walk(args.real)):
            for fn in sorted(fns):
                tasks.append(("real", os.path.join(dp, fn), 0, seed0, args.seq_every))

    acc = {}
    shown = 0
    t1 = time.time()
    with mp.Pool(args.jobs) as pool:
        for kind, bad, n, sq_bad, sq_n, nbytes, npieces, ex in pool.imap_unordered(worker, tasks, chunksize=1):
            a = acc.setdefault(kind, [0] * 6)
            for idx, v in enumerate((bad, n, sq_bad, sq_n, nbytes, npieces)):
                a[idx] += v
            for s, want, got in ex:
                if shown < 6:
                    print("[%s] MISMATCH %s\n  hf   : %s\n  model: %s" % (kind, esc(s), want, got), flush=True)
                    shown += 1
    tot = [0] * 6
    summary = {"seed": seed0, "args": vars(args), "results": []}
    for kind, a in sorted(acc.items()):
        print("RESULT %-9s %10d cases (%d bytes, %d pieces), %d mismatches; three Splits == file on %d, %d differ"
              % (kind, a[1], a[4], a[5], a[0], a[3], a[2]), flush=True)
        summary["results"].append(dict(kind=kind, cases=a[1], bytes=a[4], pieces=a[5], mismatches=a[0],
                                       seq3_cases=a[3], seq3_mismatches=a[2]))
        tot = [x + y for x, y in zip(tot, a)]
    print("TOTAL %d cases (%d bytes, %d pieces), %d mismatches; three Splits %d cases, %d differ (%.0fs, %d jobs)"
          % (tot[1], tot[4], tot[5], tot[0], tot[3], tot[2], time.time() - t1, args.jobs), flush=True)
    summary.update(total=tot[1], bytes=tot[4], pieces=tot[5], mismatches=tot[0], seq3_total=tot[3],
                   seq3_mismatches=tot[2])
    os.makedirs(os.path.join(M.REPO, "build"), exist_ok=True)
    with open(os.path.join(M.REPO, "build", "dsv3_fuzz_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return 1 if (tot[0] or tot[2]) else 0


if __name__ == "__main__":
    sys.exit(main())
