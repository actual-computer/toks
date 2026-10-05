#!/usr/bin/env python3
# tests/unigram/fetch.py: download the pinned unigram-family tokenizer.json files (tests/unigram/pins.json),
# anonymously, at a pinned hub revision, verified by sha256. Files land in ~/.cache/toks/tokenizers/<name>
# (TOKS_TOKENIZER_CACHE overrides), the same cache tools/corpora/fetch_tokenizers.py fills.
#
#   uv run tests/unigram/fetch.py                 # every pin
#   uv run tests/unigram/fetch.py uni_bgem3 ...   # some
#
# /// script
# requires-python = ">=3.10"
# ///

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def cache_root() -> str:
    return os.environ.get("TOKS_TOKENIZER_CACHE", os.path.expanduser("~/.cache/toks/tokenizers"))


def pins() -> dict:
    with open(os.path.join(HERE, "pins.json"), encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_of(name: str) -> str:
    return os.path.join(cache_root(), name)


def fetch_one(name: str, pin: dict) -> str:
    dest = path_of(name)
    if os.path.isfile(dest) and sha256_file(dest) == pin["sha256"]:
        return dest
    url = f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
    os.makedirs(cache_root(), exist_ok=True)
    for attempt in range(4):  # bounded retries
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                data = r.read()
            got = hashlib.sha256(data).hexdigest()
            if got != pin["sha256"]:
                raise RuntimeError(f"sha256 mismatch for {name}: got {got} want {pin['sha256']}")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"  retry {name}: {e}", file=sys.stderr, flush=True)
            time.sleep(2 + 3 * attempt)
    return dest


def main() -> None:
    p = pins()
    names = sys.argv[1:] or list(p)
    for n in names:
        d = fetch_one(n, p[n])
        print(f"{n}: ok {p[n]['sha256'][:16]} {d}")


if __name__ == "__main__":
    main()
