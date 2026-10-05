#!/usr/bin/env python3
"""The o200k proof runs: tests/model/o200k_model.py (docs/templates/o200k.md transcribed) vs hf 0.23.2.

  uv run --python 3.12 --with tokenizers==0.23.2 python tests/model/run_o200k_fuzz.py [options]

    --short N         gen_case strings per variant                      (default 3000)
    --long N          gen_long strings (60..1000 chars) per variant     (default 300)
    --exh L           every string of length 0..L over o200k_model.REP  (default 3)
    --exh-small L     every string of length 0..L over REP_SMALL        (default 0)
    --vocab N         N strings of 1..30 o200k vocabulary tokens (needs ~/.cache/toks/tokenizers/o200k)
    --real DIR        every real-text file under DIR: each line, each paragraph, random chunks
    --fetch DIR       first fetch a real-text corpus into DIR (wikipedia in ~50 languages, python and c code)
    --variants LIST   default o200k,nemo; kimi (needs --with tiktoken==0.14.0) runs --kshort / --kexh / --kreal:
                      strings of <= 64 utf-8 bytes, the bound of its oracle (tiktoken itself, o200k_model.py)
    --jobs J --seed S --ri-every K (also compare hf Removed+invert with Isolated on every K-th case, default 5)

Prints one RESULT line per (variant, kind) and a TOTAL; writes build/o200k_fuzz_summary.json. Exit 1 on
any mismatch. Mismatch examples are printed escaped (tool output strips some sequences).
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o200k_model as M  # noqa: E402

CHUNK = 2000


def esc(s):
    return s.encode("unicode_escape").decode("ascii")


# ---------------------------------------------------------------- case sources (deterministic per task)
def _vocab_strings():
    path = M.PATTERN_FILES[0][1]
    with open(path, "rb") as f:
        tj = json.loads(f.read())
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    out = []
    for tok in tj["model"]["vocab"]:
        try:
            out.append(bytes(inv[ch] for ch in tok).decode("utf-8"))
        except (KeyError, UnicodeDecodeError):
            pass
    return out


_VOCAB = None


def cases(task):
    variant, kind, a, b, seed = task
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
        global _VOCAB
        if _VOCAB is None:
            _VOCAB = _vocab_strings()
        rng = random.Random(seed)
        for _ in range(b - a):
            yield "".join(rng.choice(_VOCAB) for _ in range(rng.randint(1, 30)))
    elif kind == "real":
        path = a
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        rng = random.Random(seed)
        yield from M.segments_of(text, rng)
    elif kind == "kshort":                       # kimi: <= KIMI_MAXB bytes (the tiktoken oracle's bound)
        rng = random.Random(seed)
        for _ in range(b - a):
            yield M.gen_kimi(rng)
    elif kind == "kexh":
        length, lo, hi = seed, a, b
        k = len(M.REP_KIMI)
        for idx in range(lo, hi):
            out = []
            x = idx
            for _ in range(length):
                out.append(M.REP_KIMI[x % k])
                x //= k
            yield "".join(out)
    elif kind == "kreal":
        with open(a, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        yield from M.kimi_real_slices(text, random.Random(seed), b)
    else:
        raise ValueError(kind)


def worker(task):
    variant, kind = task[0], task[1]
    hf = M.VARIANTS[variant]["engine"] == "hf"
    if hf:
        M.init()
    else:
        M.init_kimi()
    ri_every = task[5] if hf else 0
    bad = n = ri_bad = ri_n = 0
    examples = []
    batch = []

    def run(batch):
        nonlocal bad, n, ri_bad, ri_n
        for s, want in zip(batch, M.oracle_batch(variant, batch)):
            got = M.k3_scan(s, variant)
            n += 1
            if want != got:
                bad += 1
                if len(examples) < 3:
                    examples.append((s, want, got))
            if ri_every and n % ri_every == 0:
                ri_n += 1
                if M.hf_pieces(variant, s, removed_invert=True) != want:
                    ri_bad += 1
                    if len(examples) < 3:
                        examples.append((s, want, "removed+invert differs"))

    for s in cases(task[:5]):
        batch.append(s)
        if len(batch) == 500:
            run(batch)
            batch = []
    run(batch)
    return variant, kind, bad, n, ri_bad, ri_n, examples


# ---------------------------------------------------------------- real-text corpus
WIKI_LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "tr", "vi", "ru", "uk", "bg", "sr", "el",
              "ka", "hy", "he", "ar", "fa", "ur", "hi", "mr", "ne", "bn", "pa", "gu", "ta", "te", "kn", "ml",
              "si", "th", "lo", "my", "km", "bo", "zh", "ja", "ko", "am", "yo", "ha", "sw", "is", "cy", "eu",
              "hu", "fi", "lt", "ro", "az", "kk", "mn", "dv", "chr", "iu"]


def _get_json(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "toks-o200k-fuzz/0.1 (tokenizer test corpus)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"   # 204 languages, CC-BY-SA 4.0


def fetch_flores(dirpath):
    """FLORES-200 dev + devtest (~2000 sentences in each of 204 languages, every major script): one
    file per language and split under DIR/flores/."""
    import io
    import tarfile
    import urllib.request
    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "toks-o200k-fuzz/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    out = os.path.join(dirpath, "flores")
    os.makedirs(out, exist_ok=True)
    n = size = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith((".dev", ".devtest")):
                body = tf.extractfile(m).read()
                with open(os.path.join(out, os.path.basename(m.name) + ".txt"), "wb") as f:
                    f.write(body)
                n += 1
                size += len(body)
    print("fetch flores-200: %d files, %d bytes (tarball %d bytes)" % (n, size, len(data)), file=sys.stderr)
    return size


def fetch(dirpath, wiki=True):
    os.makedirs(dirpath, exist_ok=True)
    total = fetch_flores(dirpath)
    api = "https://%s.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0" \
          "&prop=extracts&explaintext=1"
    for lang in (WIKI_LANGS if wiki else []):
        texts = []
        for q in ["&grnlimit=20&exintro=1&exlimit=20"] * 3 + ["&grnlimit=1"] * 4:   # intros, articles
            try:
                d = _get_json(api % lang + q)
                texts += [p.get("extract", "") for p in d.get("query", {}).get("pages", {}).values()]
            except Exception as e:  # noqa: BLE001 (best effort corpus; failures are reported)
                print("fetch %s: %s" % (lang, e), file=sys.stderr)
                break
            time.sleep(1.0)          # wikimedia rate limits anonymous api clients
        body = "\n\n".join(t for t in texts if t)
        with open(os.path.join(dirpath, "wiki-%s.txt" % lang), "w", encoding="utf-8") as f:
            f.write(body)
        total += len(body.encode("utf-8"))
        print("fetch wiki %-4s %8d bytes" % (lang, len(body.encode("utf-8"))), file=sys.stderr, flush=True)
    # code: python's stdlib and the toolchain's c headers (local files, no network)
    import sysconfig
    for name, root, exts, cap in (("code-python.txt", sysconfig.get_paths()["stdlib"], (".py",), 24 << 20),
                                  ("code-c.txt", os.path.expanduser("~/.cache/toks-llvm/21.1.8/lib/clang"),
                                   (".h",), 8 << 20)):
        acc = []
        size = 0
        for dp, _dn, fns in sorted(os.walk(root)):
            for fn in sorted(fns):
                if fn.endswith(exts) and size < cap:
                    try:
                        with open(os.path.join(dp, fn), "rb") as f:
                            data = f.read()
                        acc.append(data)
                        size += len(data)
                    except OSError:
                        pass
        with open(os.path.join(dirpath, name), "wb") as f:
            f.write(b"\n".join(acc))
        total += size
        print("fetch %s %d bytes from %s" % (name, size, root), file=sys.stderr, flush=True)
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
    ap.add_argument("--variants", default="o200k,nemo")
    ap.add_argument("--kshort", type=int, default=3000, help="kimi: gen_kimi strings")
    ap.add_argument("--kexh", type=int, default=3, help="kimi: every string of 0..L atoms over REP_KIMI")
    ap.add_argument("--kreal", type=int, default=0, help="kimi: slices per real-text file")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ri-every", type=int, default=5)
    args = ap.parse_args()

    import tokenizers
    print("oracle: tokenizers %s; python %s" % (tokenizers.__version__, sys.version.split()[0]), flush=True)
    if tokenizers.__version__ != "0.23.2":
        raise SystemExit("tokenizers %s != 0.23.2 (the pinned oracle)" % tokenizers.__version__)
    for line in M.check_pattern_files():
        print(line, flush=True)
    t0 = time.time()
    M.init()
    pr = M.PROBE
    print("probes (%s): %s" % (M.PROBE_PATH, ", ".join("%s %d" % (k, len(v)) for k, v in sorted(pr.items()))),
          flush=True)
    print("class claims of docs/templates/o200k.md §2 hold on every scalar; folds {x,X}+U+017F; contraction "
          "probe == fold sets (%.1fs)" % (time.time() - t0), flush=True)
    if "kimi" in args.variants.split(","):
        M.init_kimi()
        kp = M.KPROBE
        print("kimi probes through tiktoken %s (%s): %s" % (kp["tiktoken"], M.KIMI_PROBE_PATH, ", ".join(
            "%s %d" % (k, len(v)) for k, v in sorted(kp.items()) if k != "tiktoken")), flush=True)
        diff = {k: len(kp[k] ^ pr[k]) for k in ("L", "N", "M", "S", "UP", "LO", "PFX", "P")}
        print("kimi: tiktoken's \\p{Han} == regex-syntax 0.8.11 Script=Han (%d code points); §6 class claims "
              "hold; sets that differ from onig's: %s" % (len(kp["HAN"]), diff), flush=True)
    if args.fetch:
        fetch(args.fetch)
        args.real = args.real or args.fetch

    seed0 = args.seed if args.seed is not None else random.randrange(1 << 30)
    print("seed %d" % seed0, flush=True)
    tasks = []
    for variant in args.variants.split(","):
        if M.VARIANTS[variant]["engine"] == "tiktoken":
            for st in range(0, args.kshort, CHUNK):
                tasks.append((variant, "kshort", st, min(st + CHUNK, args.kshort), seed0 + st, 0))
            for length in range(0, args.kexh + 1):
                total = len(M.REP_KIMI) ** length
                for st in range(0, total, 20000):
                    tasks.append((variant, "kexh", st, min(st + 20000, total), length, 0))
            if args.real and args.kreal:
                for dp, _dn, fns in sorted(os.walk(args.real)):
                    for fn in sorted(fns):
                        tasks.append((variant, "kreal", os.path.join(dp, fn), args.kreal, seed0, 0))
            continue
        for st in range(0, args.short, CHUNK):
            tasks.append((variant, "short", st, min(st + CHUNK, args.short), seed0 + st, args.ri_every))
        for st in range(0, args.long, CHUNK // 10):
            tasks.append((variant, "long", st, min(st + CHUNK // 10, args.long), seed0 + 10**9 + st,
                          args.ri_every))
        for kind, alpha, lmax in (("exh", M.REP, args.exh), ("exh_small", M.REP_SMALL, args.exh_small)):
            for length in range(0, lmax + 1):
                total = len(alpha) ** length
                for st in range(0, total, 20000):
                    tasks.append((variant, kind, st, min(st + 20000, total), length, args.ri_every))
        for st in range(0, args.vocab, CHUNK):
            tasks.append((variant, "vocab", st, min(st + CHUNK, args.vocab), seed0 + 2 * 10**9 + st,
                          args.ri_every))
        if args.real:
            for dp, _dn, fns in sorted(os.walk(args.real)):
                for fn in sorted(fns):
                    tasks.append((variant, "real", os.path.join(dp, fn), 0, seed0, args.ri_every))

    acc = {}
    shown = 0
    t1 = time.time()
    with mp.Pool(args.jobs) as pool:
        for variant, kind, bad, n, ri_bad, ri_n, ex in pool.imap_unordered(worker, tasks, chunksize=1):
            a = acc.setdefault((variant, kind), [0, 0, 0, 0])
            a[0] += bad
            a[1] += n
            a[2] += ri_bad
            a[3] += ri_n
            for s, want, got in ex:
                if shown < 6:
                    print("[%s/%s] MISMATCH %s\n  hf   : %s\n  model: %s" % (variant, kind, esc(s), want, got),
                          flush=True)
                    shown += 1
    tb = tn = trb = trn = 0
    summary = {"seed": seed0, "args": vars(args), "results": []}
    for (variant, kind), (bad, n, ri_bad, ri_n) in sorted(acc.items()):
        print("RESULT %-6s %-9s %10d cases, %d mismatches; removed+invert == isolated on %d, %d differ"
              % (variant, kind, n, bad, ri_n, ri_bad), flush=True)
        summary["results"].append(dict(variant=variant, kind=kind, cases=n, mismatches=bad, ri_cases=ri_n,
                                       ri_mismatches=ri_bad))
        tb += bad
        tn += n
        trb += ri_bad
        trn += ri_n
    print("TOTAL %d cases, %d mismatches; removed+invert %d cases, %d differ (%.0fs, %d jobs)"
          % (tn, tb, trn, trb, time.time() - t1, args.jobs), flush=True)
    summary.update(total=tn, mismatches=tb, ri_total=trn, ri_mismatches=trb)
    os.makedirs(os.path.join(M.REPO, "build"), exist_ok=True)
    with open(os.path.join(M.REPO, "build", "o200k_fuzz_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return 1 if (tb or trb) else 0


if __name__ == "__main__":
    sys.exit(main())
