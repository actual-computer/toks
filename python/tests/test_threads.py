"""python/tests/test_threads.py: one Tokenizer shared by many threads (toks.h: a context is read-only and
shared; the package gives every concurrent call its own scratch slot). Texts of 2 KiB or more run without
the GIL, so their core calls overlap; the results must equal the serial ones exactly."""
import random
import threading

import pytest

import toks
from conftest import tok_path

THREADS = 8


def corpus():
    rng = random.Random(3)
    words = ["the", "quick", "brown", "fox", "\u00e9t\u00e9", "\u65e5\u672c\u8a9e", "\U0001F600", "<|endoftext|>",
             "1234567", "  ", "\n\n", "don't", "x" * 40]
    texts = []
    for n in (1, 5, 30, 300, 1500, 4000):                    # short (GIL held) and long (GIL released)
        for _ in range(6):
            texts.append(" ".join(rng.choice(words) for _ in range(n)))
    return texts


@pytest.mark.parametrize("name", ["gpt2", "llama3"])
def test_shared_tokenizer(name):
    t = toks.Tokenizer.from_file(tok_path(name))
    texts = corpus()
    want = {(x, m): t.encode(x, added_tokens=m) for x in texts for m in ("all", "none")}
    want_pieces = {x: t.pieces(x) for x in texts}
    want_dec = {x: t.decode(want[(x, "all")]) for x in texts}
    errors = []
    barrier = threading.Barrier(THREADS)

    def worker(k):
        rng = random.Random(k)
        try:
            barrier.wait()
            for _ in range(60):
                x = rng.choice(texts)
                m = rng.choice(("all", "none"))
                op = rng.randrange(5)
                if op == 0:
                    ok = t.encode(x, added_tokens=m) == want[(x, m)]
                elif op == 1:
                    batch = rng.sample(texts, 5)
                    ok = t.encode_batch(batch, added_tokens=m) == [want[(y, m)] for y in batch]
                elif op == 2:
                    ok = t.pieces(x) == want_pieces[x]
                elif op == 3:
                    ok = t.decode(want[(x, "all")]) == want_dec[x]
                else:
                    import array
                    out = array.array("I", bytes(4 * (len(want[(x, m)]) + 1)))
                    ok = t.encode_into(x, out, added_tokens=m) == len(want[(x, m)]) and \
                        out[:len(want[(x, m)])].tolist() == want[(x, m)]
                if not ok:
                    errors.append((k, op, x[:40]))
        except Exception as e:                               # noqa: BLE001 (reported below)
            errors.append((k, "exception", repr(e)))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors[:5]


def test_load_unload_in_threads(gpt2_path):
    errors = []

    def worker():
        try:
            for _ in range(5):
                t = toks.Tokenizer.from_file(gpt2_path)
                if t.encode("Hello world " * 500)[:2] != [15496, 995]:
                    errors.append("ids")
                del t
        except Exception as e:                               # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors[:3]
