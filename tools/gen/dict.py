#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""toks: the piece dictionary (src/gen/dict.c) of the byte-level bpe words table (kernels.md §6 "Static table").

Pieces of 2..15 bytes that common text cuts and that are no single token of a byte-level bpe model: the words table
answers a model token on first sight, while a word the model splits (" Bingley" = " Bing" + "ley") costs a K6 run on
every first sight in a call. toks_bpe_build seats these pieces in the words table's free ways after the model's own
tokens, each valued by K6's c twin on the tables just built (SPEC §2.7: the list only picks WHICH pieces get a
certified entry, never what an entry says). Pieces are counted in public text held out from the bench corpora,
split by the pre-tokenizers of the reference models; a piece's score is its highest share of the pieces of one
(text set, model) pair, so prose and code rank on one scale; a piece is kept when at least one reference model
encodes it in 2..4 ids. The output is the top N in score order.

A corpus argument is a file, or PATH:GLOB: the files under a directory, or the members of a tarball, whose name
matches GLOB, in path order. The shipped list (THIRD_PARTY_NOTICES.md "The piece dictionary"):

    uv run tools/gen/dict.py --tok ~/.cache/toks/tokenizers/{llama3,gpt2,o200k,qwen38} \\
        --prose ~/.cache/toks/enwik8 \\
        --code ~/.cache/toks-llvm/21.1.8/include:*.h ~/.cache/toks/dict-src/{numpy-2.5.2,matplotlib-3.11.1,\\
               fonttools-4.63.0,huggingface_hub-1.22.0}.tar.gz:*.py
"""
import argparse, collections, concurrent.futures, fnmatch, hashlib, os, sys, tarfile

def b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

U2B = {v: k for k, v in b2u().items()}

def raw(s):
    try:
        return bytes(U2B[ch] for ch in s)
    except KeyError:
        return None

def texts(spec):
    """the texts of one corpus argument, in path order"""
    path, _, glob = spec.partition(":")
    path = os.path.expanduser(path)
    if not glob:
        yield open(path, "rb").read()
    elif os.path.isdir(path):
        names = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs if fnmatch.fnmatch(f, glob))
        for f in names:
            yield open(f, "rb").read()
    else:
        with tarfile.open(path) as t:
            for m in sorted((m for m in t.getmembers() if m.isfile() and fnmatch.fnmatch(os.path.basename(m.name), glob)),
                            key=lambda m: m.name):
                yield t.extractfile(m).read()

def provenance(spec):
    """sha256 of the file (a tarball's: the archive itself), or of a directory's selected files in path order"""
    path, _, glob = spec.partition(":")
    h, n, size = hashlib.sha256(), 0, 0
    if glob and os.path.isdir(os.path.expanduser(path)):
        for x in texts(spec):
            h.update(x); n += 1; size += len(x)
        what = f"{n} files, {size} bytes, sha256 of their concatenation"
    else:
        h.update(open(os.path.expanduser(path), "rb").read())
        if glob:
            for x in texts(spec):
                n += 1; size += len(x)
            what = f"{n} members, {size} bytes"
        else:
            what = f"{os.path.getsize(os.path.expanduser(path))} bytes"
    return f"{h.hexdigest()}  {name(path)}{':' + glob if glob else ''} ({what})"

def name(path):
    """a corpus path as the notices name it: a directory by its last two components, a file by its name"""
    p = os.path.expanduser(path).rstrip("/")
    return "/".join(p.split("/")[-2:]) if os.path.isdir(p) else os.path.basename(p)

def count(tok_path, specs):
    """pieces of 2..15 raw bytes cut by tok's pre-tokenizer: (Counter raw -> n, total pieces, those in 2..4 ids)"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    pre, model, vocab = tok.pre_tokenizer, tok.model, tok.get_vocab()
    c = collections.Counter()
    tot = 0
    for spec in specs:
        for data in texts(spec):
            p = 0
            while p < len(data):                           # newline-aligned blocks of ~1 MiB
                q = data.find(b"\n", p + (1 << 20))
                q = len(data) if q < 0 else q + 1
                ps = [s for s, _ in pre.pre_tokenize_str(data[p:q].decode("utf-8", "replace"))]
                tot += len(ps)
                c.update(ps)
                p = q
    out, multi = collections.Counter(), set()
    for s, n in c.items():
        r = raw(s)
        if r is None or not 2 <= len(r) <= 15:
            continue
        out[r] = n
        if s not in vocab and 2 <= len(model.tokenize(s)) <= 4:
            multi.add(r)
    return out, tot, multi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--prose", nargs="+", required=True)
    ap.add_argument("--code", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=131072)
    ap.add_argument("--tsv")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen", "dict.c"))
    a = ap.parse_args()
    jobs = [(t, name, specs) for t in a.tok for name, specs in (("prose", a.prose), ("code", a.code))]
    score, multi = collections.defaultdict(float), set()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for (t, name, _), (c, tot, m) in zip(jobs, ex.map(count, [j[0] for j in jobs], [j[2] for j in jobs])):
            print(f"{os.path.basename(t)} {name}: {tot} pieces, {len(c)} distinct of 2..15 B, {len(m)} in 2..4 ids",
                  file=sys.stderr)
            for r, n in c.items():
                score[r] = max(score[r], n / tot)
            multi |= m
    ranked = sorted(multi, key=lambda r: (-score[r], r))[: a.n]
    if a.tsv:
        with open(a.tsv, "w") as f:
            for r in ranked:
                f.write(f"{r.hex()}\t{score[r]:.3e}\n")
    blob = b"".join(bytes([len(r)]) + r for r in ranked)
    def lit(r):                                            # one piece as a c string literal: length, then its bytes
        out = []
        for b in bytes([len(r)]) + r:
            c = chr(b)
            out.append(c if 0x20 <= b < 0x7F and c not in '"\\?' else f"\\{b:03o}")
        return '"' + "".join(out) + '"'
    lines = [
        "/* toks: the piece dictionary of the words table (kernels.md §6).  GENERATED FILE -- DO NOT EDIT.",
        " * Regenerate with tools/gen/dict.py (its docstring has the command); provenance and licenses:",
        " * THIRD_PARTY_NOTICES.md \"The piece dictionary\".",
        f" * {len(ranked)} pieces of 2..15 bytes in score order, each one length byte then its bytes: {len(blob)} bytes,",
        f" * sha256 {hashlib.sha256(blob).hexdigest()}",
        " * reference models (their pre-tokenizers and vocabularies): " + " ".join(os.path.basename(t) for t in a.tok),
        " * prose:",
        *(f" *   {provenance(s)}" for s in a.prose),
        " * code:",
        *(f" *   {provenance(s)}" for s in a.code),
        " */",
        "",
        '#include "dict.h"',
        "",
        f"const uint32_t toks_dict_n = {len(ranked)}u;",
        f"const uint8_t toks_dict[{len(blob) + 1}] =",     # + the literal's terminating 0, never read
    ]
    cur = "   "
    for r in ranked:
        s = lit(r)
        if len(cur) + 1 + len(s) > 118:
            lines.append(cur)
            cur = "   "
        cur += " " + s
    lines.append(cur + ";")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{a.out}: {len(ranked)} pieces, {len(blob)} bytes", file=sys.stderr)

if __name__ == "__main__":
    main()
