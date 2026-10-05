# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2", "tiktoken==0.14.0"]
# ///
"""tools/bench/par_ref.py: tiktoken's and hf tokenizers' parallel batch encode on exactly the batches
tests/par/bench_par.c measures: the python comparators of the toks_par receipts (tests/par/run_par.sh REFS,
docs/bench/par.md). Never a dependency of the library.

    RAYON_NUM_THREADS=k taskset -c <cpus> uv run tools/bench/par_ref.py <tiktoken|hf> <tokenizer.json> <text> \\
        <doc bytes> <MiB,..> <k> [reps]

tiktoken (0.14.0): Encoding.encode_batch(docs, num_threads=k) (a thread pool per call; the rust core releases the GIL)
over the Encoding tools/bench/e2e_ref.py builds from the same tokenizer.json (its one Split pattern and byte-level
vocab, no special tokens: hf without the post-processor). hf (tokenizers 0.23.2): Tokenizer.encode_batch(docs), hf's
default call (post-processor applied) on rayon's global pool, which run_par.sh sizes with RAYON_NUM_THREADS = k. Both
are what people run. docs: the text's first S MiB cut after the first '\\n' at or past every <doc bytes> boundary
(bench_par.c's, e2e.c's and gigatoken par.rs's rule: e2e_ref.chunks_of). States, bench_par.c's and par.rs's: first =
the process's first call on the docs (hf: rayon's pool starts in it); pass = a call right after one call on the file's
last S MiB, cut the same way; warm = the same docs again (best and median of reps). pass_seen = the share of the timed
text the warm-up call held (> 0 when 2 S exceeds the file). ids: fnv-1a-64 over the flat id stream without the
post-processor (hf: the tokens its post-processor added, dropped by special_tokens_mask), bench_par.c's batch rows'
fnv_nopp: par_table.py checks every row against toks' serial row (else MISMATCH)."""
import os
import sys
import time
from importlib.metadata import version

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_ref import chunks_of, tiktoken_encoding  # noqa: E402


def fnv(id_lists):
    h, mask = 0xCBF29CE484222325, (1 << 64) - 1
    for ids in id_lists:
        for i in ids:
            h = ((h ^ i) * 0x100000001B3) & mask
    return h


def main():
    if len(sys.argv) < 7 or sys.argv[1] not in ("tiktoken", "hf"):
        sys.exit("usage: par_ref.py <tiktoken|hf> <tokenizer.json> <text> <doc bytes> <MiB,..> <k> [reps]")
    tool, path, text, doc, sizes, k = sys.argv[1:7]
    doc, k = int(doc), int(k)
    reps = max(1, min(64, int(sys.argv[7]) if len(sys.argv) > 7 else 7))
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if tool == "tiktoken":
        enc, why = tiktoken_encoding(path)
        if enc is None:
            print(f"PYREF tool=tiktoken na={why}")
            return

        call = lambda d: enc.encode_batch(d, num_threads=k)                       # noqa: E731
        nopp = lambda out: out                                                     # noqa: E731
        ver, threads = version("tiktoken"), str(k)
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(path)
        call = tok.encode_batch
        nopp = lambda out: [[i for i, m in zip(e.ids, e.special_tokens_mask) if not m] for e in out]   # noqa: E731
        ver, threads = version("tokenizers"), os.environ.get("RAYON_NUM_THREADS", "default")
    buf = open(text, "rb").read()
    for s in sizes.split(","):
        n = int(float(s) * 1048576)
        if n > len(buf):
            sys.exit(f"par_ref.py: text too short for {s} MiB")
        docs, odocs = chunks_of(buf[:n], doc), chunks_of(buf[len(buf) - n:], doc)

        def run(d):
            t0 = time.perf_counter()
            out = call(d)
            return time.perf_counter() - t0, out
        first, out = run(docs)
        ids = nopp(out)
        del out
        h, nids = fnv(ids), sum(map(len, ids))
        del ids
        run(odocs)
        pas, out = run(docs)
        if fnv(nopp(out)) != h:
            sys.exit(f"par_ref.py: {tool} pass ids differ from first ({s} MiB)")
        del out
        warm = sorted(run(docs)[0] for _ in range(reps))
        seen = (2 * n - len(buf)) / n if 2 * n > len(buf) else 0.0
        mb = n / 1e6
        print(f"PYREF tool={tool} version={ver} mode=batch tok={name} doc={doc} docs={len(docs)} bytes={n} threads={threads} "
              f"ids={nids} fnv_nopp={h:016x} first_ms={first * 1e3:.3f} pass_ms={pas * 1e3:.3f} "
              f"warm_best_ms={warm[0] * 1e3:.3f} warm_med_ms={warm[reps // 2] * 1e3:.3f} first_mbps={mb / first:.1f} "
              f"pass_mbps={mb / pas:.1f} warm_mbps={mb / warm[0]:.1f} pass_seen={seen:.2f}", flush=True)


if __name__ == "__main__":
    main()
