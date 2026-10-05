#!/usr/bin/env python3
"""§4 added-token policy: synthetic tokenizer (special + non-special with shared
prefixes, a non-special inside a special, one-byte tokens, lstrip/rstrip/single_word)
run in modes ALL and NONSPECIAL. hf encode vs (a) the hf mirror of find_matches
(resume at start+1 after a dropped match) and (b) kernels.md §4 as written
(resume at raw m_end). argv: n_cases n_workers (workers=1 -> serial, for mac)."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toks_model as M

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
W = int(sys.argv[2]) if len(sys.argv) > 2 else 64
CHUNK = 2000
HERE = os.path.dirname(os.path.abspath(__file__))

VOCAB = {}
for b in range(256):
    VOCAB[M.BYTES_CHAR[b]] = b
merges = []
def add_merge(a, b):
    VOCAB[a + b] = len(VOCAB)
    merges.append((a, b))
for a, b in [("a", "b"), ("a", "c"), ("<", "m"), ("m", "a"), ("ma", "s"), ("mas", "k"),
             ("a", "s"), ("s", "k"), ("a", "t"), ("a", "r"), ("a", "v"), ("e", ">")]:
    add_merge(a, b)
MODEL = {"type": "BPE", "vocab": VOCAB, "merges": [[a, b] for a, b in merges],
         "ignore_merges": False}

ADDED = [
    {"id": 1000, "content": "<mask>",       "single_word": True,  "special": False, "lstrip": True,  "rstrip": True,  "normalized": True},
    {"id": 1001, "content": "ask>",         "single_word": False, "special": False, "lstrip": False, "rstrip": False, "normalized": True},
    {"id": 1002, "content": "<pad>",        "single_word": False, "special": True,  "lstrip": False, "rstrip": False, "normalized": False},
    {"id": 1003, "content": "<|im_start|>", "single_word": False, "special": True,  "lstrip": False, "rstrip": False, "normalized": False},
    {"id": 1004, "content": "<|im",        "single_word": False, "special": True,  "lstrip": False, "rstrip": False, "normalized": False},
    {"id": 1005, "content": "im_end|>",     "single_word": False, "special": False, "lstrip": True,  "rstrip": True,  "normalized": True},
    {"id": 1006, "content": "s",            "single_word": True,  "special": False, "lstrip": False, "rstrip": False, "normalized": True},
    {"id": 1007, "content": "X",            "single_word": False, "special": False, "lstrip": True,  "rstrip": False, "normalized": True},
    {"id": 1008, "content": "<cls>",        "single_word": False, "special": False, "lstrip": True,  "rstrip": True,  "normalized": True},
    {"id": 1009, "content": "sk>",          "single_word": False, "special": False, "lstrip": False, "rstrip": False, "normalized": False},
    {"id": 1010, "content": "im",           "single_word": False, "special": False, "lstrip": False, "rstrip": False, "normalized": True},
]
TJ = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": ADDED,
      "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
      "trim_offsets": False, "use_regex": False}, "post_processor": None, "decoder": None,
      "model": MODEL}

# hf reassigns ids on load (the JSON `id` is only a warning-checked hint): reuse the
# model-vocab id when the content is a vocab token, else the next free id.
def _hf_ids():
    nxt = len(VOCAB)
    for t in ADDED:
        if t["content"] in VOCAB:
            t["id"] = VOCAB[t["content"]]
        else:
            t["id"] = nxt
            nxt += 1
_hf_ids()

GENS = ["<mask>", "<pad>", "<|im_start|>", "<|im_end|>", "<|im", "ask>", "X", "<cls>", "s",
        "as", "<mask>s", "A<mask>", "<mask>ony", "im_end|>", "ab", "ac", "<mask>,", "at"]
WS = [chr(c) for c in M.WS_CPS]

def gen(rng):
    n = rng.randint(1, 10)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            out.append(rng.choice(GENS))
        elif r < 0.65:
            out.append(rng.choice(WS) * rng.randint(1, 3))
        elif r < 0.85:
            out.append("".join(rng.choice("ab<cXsk>|_me") for _ in range(rng.randint(1, 8))))
        else:
            out.append("".join(rng.choice(M.ALPHABET) for _ in range(rng.randint(0, 4))))
    return "".join(out)

_TK = None
_PIPE_HF = None
_PIPE_DOC = None

def _init():
    global _TK, _PIPE_HF, _PIPE_DOC
    from tokenizers import Tokenizer
    _TK = Tokenizer.from_str(json.dumps(TJ))
    word = M.load_word_set()
    _PIPE_HF = M.Pipe(TJ, word, resume="start+1")
    _PIPE_DOC = M.Pipe(TJ, word, resume="end")

def one(s, drop):
    if drop:
        M.set_encode_special(_TK, True)
        want = _TK.encode(s, add_special_tokens=False).ids
        M.set_encode_special(_TK, False)
    else:
        want = _TK.encode(s, add_special_tokens=False).ids
    return (want, _PIPE_HF.encode(s, drop_specials=drop),
            _PIPE_DOC.encode(s, drop_specials=drop))

def worker(seed):
    if _TK is None:
        _init()
    rng = random.Random(seed)
    bad_hf = bad_doc = 0
    ex = []
    for _ in range(CHUNK):
        s = gen(rng)
        d = rng.random() < 0.5
        want, got_hf, got_doc = one(s, d)
        if want != got_hf:
            bad_hf += 1
            if len(ex) < 3: ex.append((s, d, want, got_hf, got_doc))
        elif want != got_doc:
            bad_doc += 1
            if len(ex) < 3: ex.append((s, d, want, got_hf, got_doc))
    return bad_hf, bad_doc, CHUNK, ex

def main():
    _init()
    print("--- hf rust unit-test replays ---", flush=True)
    for s, drop in [("Hi <mask> there\t<mask>\t<mask> ", True),
                    ("Hi <mask> there\t<mask>\t<mask> ", False)]:
        want, got_hf, got_doc = one(s, drop)
        print(f"input={s!r} NONSPECIAL={drop}")
        print(f"  hf  : {want}")
        print(f"  mir : {got_hf} {'OK' if want == got_hf else 'MISMATCH'}")
        print(f"  doc : {got_doc} {'OK' if want == got_doc else 'MISMATCH'}", flush=True)

    tasks = [random.randrange(1 << 30) for _ in range(max(1, N // CHUNK))]
    bh = bd = n = 0
    shown = 0
    def show(ex):
        nonlocal shown
        for (s, d, want, got_hf, got_doc) in ex:
            if shown < 4:
                print(f"FUZZ MISMATCH input={s!r} NONSPECIAL={d}\n  hf : {want}\n  mir: {got_hf}\n  doc: {got_doc}", flush=True)
                shown += 1
    if W > 1:
        import multiprocessing as mp
        with mp.Pool(W) as pool:
            for b1, b2, cnt, ex in pool.imap_unordered(worker, tasks):
                bh += b1; bd += b2; n += cnt
                show(ex)
    else:
        for t in tasks:
            b1, b2, cnt, ex = worker(t)
            bh += b1; bd += b2; n += cnt
            show(ex)
    print(f"RESULT added: n={n} hf-mirror mismatches={bh} doc-§4 mismatches={bd}", flush=True)
    with open(os.path.join(HERE, "added_done.json"), "w") as f:
        json.dump({"n": n, "bad_mirror": bh, "bad_doc": bd}, f)

if __name__ == "__main__":
    main()
