# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2", "tiktoken==0.14.0", "transformers==5.18.0"]
# ///
"""tools/bench/e2e_ref.py: the python comparators on exactly the chunks tools/bench/e2e.c measures, single thread.

    uv run tools/bench/e2e_ref.py <tokenizer.json | kimi dir> <chunk_bytes|0> <reps> <file>... \
        [--ids toks_ids.u32] [--giga gigatoken_ids.u32]

tokenizer.json: hf = Tokenizer.encode(chunk) (hf's default call: added tokens recognized, post-processor
applied: the reference comparator). tiktoken = encode_ordinary over an Encoding built from the same vocab
(ranks = ids) and the same single Split pattern (byte-level bpe files only).
kimi dir (a tiktoken model, docs/models/kimi.md): the reference is the model's own tokenizer as served,
transformers' TikTokenTokenizer encode(chunk, add_special_tokens=False) (tests/parity/oracle_tiktoken.py
"serving" = toks ALL), reported as the hf column; tiktoken = the wrapper's own Encoding, encode_ordinary.
Outside the timers: the id stream of all chunks in order (u32 little-endian) -> count + sha-256, the digest
e2e.c prints for toks; --ids <file> (E2E_IDS_OUT of e2e.c) reports the first differing chunk and position.
tiktoken's ids and gigatoken's (--giga, tools/bench/gigatoken) are checked against the reference without the
post-processor (hf encode(chunk, add_special_tokens=False); kimi: the serving ids): a cell where they differ is
semantically different (n/a, never a forced ratio); the line says which.
Best of reps per tool; machine-readable REF lines plus a human line each."""
import hashlib
import json
import os
import sys
import time
from array import array
from importlib.metadata import version


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tiktoken_encoding(path):
    """(Encoding, None): tiktoken over a byte-level bpe tokenizer.json, its one Split pattern (or ByteLevel's own) and
    its vocab as the mergeable ranks (ranks = ids), no special tokens: hf without the post-processor. (None, why) for a
    file that is not one Split pattern plus ByteLevel. tools/bench/par_ref.py builds the same one."""
    cfg = json.load(open(path))
    pt = cfg.get("pre_tokenizer") or {}
    subs = pt.get("pretokenizers", [pt])
    pats = [p["pattern"]["Regex"] for p in subs if p.get("type") == "Split" and "Regex" in p.get("pattern", {})]
    if not pats and any(p.get("type") == "ByteLevel" and p.get("use_regex", True) for p in subs):
        pats = [r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""]
    if len(pats) != 1 or not any(p.get("type") == "ByteLevel" for p in subs):
        return None, "not_one_split_pattern_plus_bytelevel"
    import tiktoken
    dec = {c: b for b, c in bytes_to_unicode().items()}
    ranks = {}
    for s, i in cfg["model"]["vocab"].items():
        try:
            ranks[bytes(dec[ch] for ch in s)] = i
        except KeyError:
            pass
    return tiktoken.Encoding("ref", pat_str=pats[0], mergeable_ranks=ranks, special_tokens={}), None


def chunks_of(buf: bytes, chunk: int):
    out, p, n = [], 0, len(buf)
    while p < n:
        e = n
        if chunk and p + chunk < n:
            e = p + chunk
            while e < n and buf[e - 1] != 0x0A:
                e += 1
        out.append(buf[p:e].decode("utf-8"))
        p = e
    return out


def best(fn, chunks, reps):
    b = 1e30
    for _ in range(reps):
        a = time.perf_counter()
        for c in chunks:
            fn(c)
        b = min(b, time.perf_counter() - a)
    return b


def digest(streams):
    h, n = hashlib.sha256(), 0
    for ids in streams:
        a = array("I", ids)
        assert a.itemsize == 4 and sys.byteorder == "little"
        h.update(a.tobytes())
        n += len(ids)
    return n, h.hexdigest()


def read_u32(path):
    a = array("I")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def first_diff(stream, streams):
    """stream: one flat u32 array; streams: the reference's ids per chunk."""
    pos = 0
    for k, ids in enumerate(streams):
        mine = stream[pos:pos + len(ids)].tolist()
        if mine != list(ids):
            j = next((i for i, (a, b) in enumerate(zip(mine, ids)) if a != b), min(len(mine), len(ids)))
            return f"chunk_{k}_id_{j}_(stream_pos_{pos + j}):_got_{mine[max(0, j - 3):j + 4]}_ref_{list(ids[max(0, j - 3):j + 4])}".replace(" ", "")
        pos += len(ids)
    return "none" if pos == len(stream) else f"{len(stream) - pos}_extra_ids"


def report_giga(path, chunk, ref, label):
    """gigatoken's ids (an untimed pass of tools/bench/gigatoken) against the no-post-processor reference."""
    g = read_u32(path)
    n_g, sha_g = len(g), hashlib.sha256(g.tobytes()).hexdigest()
    n_r, sha_r = digest(ref)
    same = "yes" if (n_g, sha_g) == (n_r, sha_r) and n_g > 0 else "no"     # two empty streams are not a match
    fd = "none" if same == "yes" else first_diff(g, ref)
    print(f"REF tool=gigatoken chunk={chunk} ids={n_g} sha={sha_g[:16]} {label}_ids={n_r} {label}_sha={sha_r[:16]} same_as_{label}={same} first_diff={fd}")


def main():
    argv = sys.argv[1:]
    opts = {}
    for flag in ("--ids", "--giga"):
        if flag in argv:
            i = argv.index(flag)
            opts[flag] = argv[i + 1]
            del argv[i:i + 2]
    path, chunk, reps, files = argv[0], int(argv[1]), int(argv[2]), argv[3:]
    buf = b"".join(open(f, "rb").read() for f in files)
    chunks = chunks_of(buf, chunk)
    if os.path.isdir(path):
        return kimi(path, chunk, reps, buf, chunks, opts)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    hf_ids = [tok.encode(c).ids for c in chunks]
    n_hf, sha_hf = digest(hf_ids)
    t = best(tok.encode, chunks, reps)
    print(f"hf    chunk {chunk:7d}  {len(buf) / t / 1e6:8.2f} MB/s {n_hf / t / 1e6:7.3f} Mtok/s {t / len(chunks) * 1e9:11.0f} ns/call")
    print(f"REF tool=hf version={version('tokenizers')} chunk={chunk} bytes={len(buf)} calls={len(chunks)} ids={n_hf} sha={sha_hf[:16]} s={t:.6f} reps={reps}")
    if "--ids" in opts:
        print(f"FIRSTDIFF {first_diff(read_u32(opts['--ids']), hf_ids)}")
    nopp = [tok.encode(c, add_special_tokens=False).ids for c in chunks]
    if "--giga" in opts:
        report_giga(opts["--giga"], chunk, nopp, "hf_nopp")
    enc, why = tiktoken_encoding(path)
    if enc is None:
        print(f"REF tool=tiktoken chunk={chunk} na={why}")
        return
    tk_ids = [enc.encode_ordinary(c) for c in chunks]
    n_tk, sha_tk = digest(tk_ids)
    n_np, sha_np = digest(nopp)
    same = "yes" if (n_tk, sha_tk) == (n_np, sha_np) and n_tk > 0 else "no"
    t = best(enc.encode_ordinary, chunks, reps)
    print(f"tikt  chunk {chunk:7d}  {len(buf) / t / 1e6:8.2f} MB/s {n_tk / t / 1e6:7.3f} Mtok/s {t / len(chunks) * 1e9:11.0f} ns/call  ids==hf(no post-processor): {same}")
    print(f"REF tool=tiktoken version={version('tiktoken')} chunk={chunk} bytes={len(buf)} calls={len(chunks)} ids={n_tk} sha={sha_tk[:16]} s={t:.6f} reps={reps} hf_nopp_ids={n_np} hf_nopp_sha={sha_np[:16]} same_as_hf_nopp={same}")


def kimi(path, chunk, reps, buf, chunks, opts):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "parity"))
    from oracle_tiktoken import Kimi            # pins tiktoken 0.14.0 + transformers 5.18.0 and the model files
    k = Kimi(path)
    serve = lambda c: k.encode(c, "serving")
    ref = [serve(c) for c in chunks]
    n_r, sha_r = digest(ref)
    t = best(serve, chunks, reps)
    print(f"ref   chunk {chunk:7d}  {len(buf) / t / 1e6:8.2f} MB/s {n_r / t / 1e6:7.3f} Mtok/s {t / len(chunks) * 1e9:11.0f} ns/call  (transformers TikTokenTokenizer, serving)")
    print(f"REF tool=hf version=transformers-{version('transformers')}+tiktoken-{version('tiktoken')} impl=kimi-serving chunk={chunk} bytes={len(buf)} calls={len(chunks)} ids={n_r} sha={sha_r[:16]} s={t:.6f} reps={reps}")
    if "--ids" in opts:
        print(f"FIRSTDIFF {first_diff(read_u32(opts['--ids']), ref)}")
    if "--giga" in opts:
        report_giga(opts["--giga"], chunk, ref, "hf_nopp")
    tk_ids = [k.enc.encode_ordinary(c) for c in chunks]
    n_tk, sha_tk = digest(tk_ids)
    same = "yes" if (n_tk, sha_tk) == (n_r, sha_r) and n_tk > 0 else "no"
    t = best(k.enc.encode_ordinary, chunks, reps)
    print(f"tikt  chunk {chunk:7d}  {len(buf) / t / 1e6:8.2f} MB/s {n_tk / t / 1e6:7.3f} Mtok/s {t / len(chunks) * 1e9:11.0f} ns/call  ids==reference: {same}")
    print(f"REF tool=tiktoken version={version('tiktoken')} chunk={chunk} bytes={len(buf)} calls={len(chunks)} ids={n_tk} sha={sha_tk[:16]} s={t:.6f} reps={reps} hf_nopp_ids={n_r} hf_nopp_sha={sha_r[:16]} same_as_hf_nopp={same}")


if __name__ == "__main__":
    main()
