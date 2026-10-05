#!/usr/bin/env python3
"""tests/wordpiece/fetch.py: download the pinned WordPiece tokenizer.json files (tests/wordpiece/pins.json) at their
hub revision, verify their sha256, and store them as $TOKS_TOKENIZER_CACHE/wp-<name> (default
~/.cache/toks/tokenizers/wp-<name>, the cache tests/c and tools/corpora use). Anonymous hub access only.

    uv run tests/wordpiece/fetch.py                 # every pin
    uv run tests/wordpiece/fetch.py minilm-l6 mpnet # some
"""
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def cache_dir():
    return os.environ.get("TOKS_TOKENIZER_CACHE") or os.path.expanduser("~/.cache/toks/tokenizers")


def pins():
    with open(os.path.join(HERE, "pins.json")) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def path_of(name):
    return os.path.join(cache_dir(), "wp-" + name)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, pin):
    dest = path_of(name)
    if os.path.isfile(dest) and sha256_file(dest) == pin["sha256"]:
        return dest
    url = "https://huggingface.co/%s/resolve/%s/%s" % (pin["repo"], pin["revision"], pin.get("path", "tokenizer.json"))
    req = urllib.request.Request(url, headers={"User-Agent": "toks-wordpiece-fetch"})
    last = None
    for _ in range(4):                                      # bound: 4 attempts
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
            got = hashlib.sha256(data).hexdigest()
            if got != pin["sha256"]:
                raise RuntimeError("%s: sha256 %s != pinned %s" % (name, got, pin["sha256"]))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError("%s: %s" % (name, last))


def main():
    ps = pins()
    names = sys.argv[1:] or list(ps)
    for n in names:
        p = fetch(n, ps[n])
        print("%s: ok %s %s" % (n, ps[n]["sha256"][:16], p))


if __name__ == "__main__":
    main()
