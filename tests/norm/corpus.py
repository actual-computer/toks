# /// script
# requires-python = ">=3.10"
# ///
"""A multilingual corpus for the NFC differential: wikipedia 20231101 articles (wikimedia/wikipedia on the hf
hub, cc by-sa 4.0) in 64 languages, through the hub's datasets-server rows api (100 rows per request, no
download of the dumps). Writes DIR/<lang>.jsonl, one json string per document. Not checked in; the bytes are
whatever the api serves for (config, offset), recorded with their sha256 in DIR/MANIFEST.

usage: uv run tests/norm/corpus.py [DIR] [MB_PER_LANGUAGE]      (default build/norm-corpus 3)
"""

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

LANGS = ("en de fr es pt it nl pl cs sk hu ro tr az vi yo ig ha sw zu ln wo ru uk bg sr mk el ka hy he yi ar fa "
         "ur ps ckb dv hi mr ne sa bn as or pa gu ta te kn ml si th lo km my bo am ti chr iu ko ja zh").split()
API = "https://datasets-server.huggingface.co/rows"


def rows(lang, offset):
    q = urllib.parse.urlencode({"dataset": "wikimedia/wikipedia", "config": "20231101." + lang, "split": "train",
                                "offset": offset, "length": 100})
    last = None
    for attempt in range(8):
        time.sleep(1.0)                               # the api rate-limits (429): stay slow
        try:
            with urllib.request.urlopen(API + "?" + q, timeout=120) as r:
                return [x["row"]["text"] for x in json.load(r)["rows"]]
        except Exception as e:  # noqa: BLE001 -- retry anything, report the last
            last = e
            time.sleep(15 * (attempt + 1) if "429" in str(e) else 3)
    sys.stderr.write("corpus: %s offset %d failed: %s\n" % (lang, offset, last))
    return []


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/norm-corpus")
    budget = int(float(sys.argv[2] if len(sys.argv) > 2 else 3) * 1e6)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lang in LANGS:
        path = out / (lang + ".jsonl")
        if not path.exists() or path.stat().st_size == 0:
            docs, size, offset = [], 0, 0
            while size < budget and offset < 20000:
                got = rows(lang, offset)
                if not got:
                    break
                docs += got
                size += sum(len(d.encode("utf-8")) for d in got)
                offset += 300 if len(got) == 100 else 100
            if not docs:
                continue                              # nothing served: retried by the next run
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
            tmp.rename(path)
        blob = path.read_bytes()
        manifest.append("%s %d %s" % (path.name, len(blob), hashlib.sha256(blob).hexdigest()))
        print(manifest[-1], flush=True)
    (out / "MANIFEST").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
