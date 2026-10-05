#!/usr/bin/env python3
"""tools/ci/fetch_tokenizers.py: every pinned tokenizer file, in the cache the tests read, verified by sha256.

No pin is repeated here. They are read where their owners keep them:
  tools/corpora/fetch_tokenizers.py   TOKENIZERS (tests/wordpiece/pins.json and tests/unigram/pins.json folded in)
  tests/spm/pins.json                 the sentencepiece-style files
  tests/parity/oracle_tiktoken.py     MODEL_FILES: the kimi k3 directory under the hub's own names
and tests/data/targets/ledger.txt (tests/c/test_targets.c) must agree with them. Files land in $TOKS_TOKENIZER_CACHE
(default ~/.cache/toks/tokenizers) and $TOKS_KIMI_DIR (default ~/.cache/toks/kimik3), where tests/c reads them. A file
already there with its pinned sha256 is kept; anything else is fetched again. HF_TOKEN, when set, goes to
huggingface.co only (gated repos; never forwarded on a redirect to another host); without it a gated file is
reported as gated. Python 3 standard library only.

  python3 tools/ci/fetch_tokenizers.py           fetch what is missing or wrong, then print one line per file
  python3 tools/ci/fetch_tokenizers.py --check   fetch nothing: verify what is there (a benchmark machine's or yours)
  python3 tools/ci/fetch_tokenizers.py --key     print the sha256 of the pin table (the CI cache key)

Exit 1 when a critical target's file is not there with its pin afterwards, when two sources pin one file to different
bytes, or when the ledger pins a file no source fetches. Other files that could not be had are listed: the tests
SKIP them and tools/ci/suites.py counts the SKIPs. docs/ci.md.
"""

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(ROOT, "tests", "data", "targets", "ledger.txt")
HF = "huggingface.co"

# The critical targets, SPEC §1.1 (d), by their ledger names: the first five (2026-10-03) and the last generations of
# llama, nemotron, deepseek and minimax (2026-10-04). Missing after the fetch = exit 1, never a SKIP.
CRITICAL = ("glm-5.3", "kimi-k3", "gpt-oss", "qwen-3.8", "gemma-4",
            "llama-3.3", "llama-3.1", "llama-4",
            "nemotron-3", "nemotron-3n30", "nemotron-3o", "nemo-nano2", "nemo-nano2vl",
            "deepseek-v3", "deepseek-r1", "deepseek-v3.1", "deepseek-v3.2", "deepseek-v4", "deepseek-v4.1",
            "minimax-t01", "minimax-m1", "minimax-m2", "minimax-m3")


def cache_dirs():
    tok = os.environ.get("TOKS_TOKENIZER_CACHE") or os.path.expanduser("~/.cache/toks/tokenizers")
    kimi = os.environ.get("TOKS_KIMI_DIR") or os.path.expanduser("~/.cache/toks/kimik3")
    return {"tokenizers": tok, "kimik3": kimi}


def corpora_pins():
    """tools/corpora/fetch_tokenizers.py's TOKENIZERS. That module downloads with requests (imported at its top);
    only its table is read here, so a stand-in module serves the import when requests is not installed."""
    path = os.path.join(ROOT, "tools", "corpora", "fetch_tokenizers.py")
    spec = importlib.util.spec_from_file_location("toks_corpora_pins", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    had = "requests" in sys.modules
    if not had:
        sys.modules["requests"] = types.ModuleType("requests")
    try:
        spec.loader.exec_module(mod)
    finally:
        if not had:
            del sys.modules["requests"]
    return {name: (p["url"], p["sha256"]) for name, p in mod.TOKENIZERS.items()}


def spm_pins():
    with open(os.path.join(ROOT, "tests", "spm", "pins.json"), encoding="utf-8") as f:
        pins = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    return {k: (f"https://{HF}/{v['repo']}/resolve/{v['revision']}/tokenizer.json", v["sha256"]) for k, v in pins.items()}


def kimi_dir_pins(corpora):
    """oracle_tiktoken.py's MODEL_FILES (read as literals: the module imports tiktoken and transformers), fetched
    from the hub revision tools/corpora pins kimik3.tiktoken at; the two pins must name the same revision."""
    path = os.path.join(ROOT, "tests", "parity", "oracle_tiktoken.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    lit = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("KIMI_REVISION", "MODEL_FILES"):
                lit[node.targets[0].id] = ast.literal_eval(node.value)
    base = corpora["kimik3.tiktoken"][0].rsplit("/", 1)[0]
    if not base.endswith("/" + lit["KIMI_REVISION"]):
        sys.exit(f"fetch_tokenizers: oracle_tiktoken.py pins Kimi-K3 at {lit['KIMI_REVISION']}, tools/corpora at {base}")
    return {name: (f"{base}/{name}", sha) for name, sha in lit["MODEL_FILES"].items()}


def ledger():
    """(target, file, sha256) per line, parsed as tests/c/test_targets.c parses it."""
    rows = []
    with open(LEDGER, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            target, file, sha, kind = line.split()[:4]
            if kind != "no-reader":
                rows.append((target, file, sha))
    return rows


def pin_table():
    """{(dir, name): (url, sha256)} and the conflicts: a file two sources pin to different bytes."""
    corpora = corpora_pins()
    table, errors = {}, []
    for d, src in (("tokenizers", corpora), ("tokenizers", spm_pins()), ("kimik3", kimi_dir_pins(corpora))):
        for name, (url, sha) in src.items():
            old = table.get((d, name))
            if old is not None and old[1] != sha:
                errors.append(f"{d}/{name}: pinned to {old[1]} and to {sha}")
            table.setdefault((d, name), (url, sha))
    return table, errors


def critical_files(table, rows):
    """{(dir, name): target} for every file a critical target is read from: its ledger file, a tiktoken file's
    companions (<stem>_*, as test_targets.c reads them) and, for kimi k3, the kimik3 directory."""
    missing = [t for t in CRITICAL if t not in {r[0] for r in rows}]
    if missing:
        sys.exit(f"fetch_tokenizers: CRITICAL names targets the ledger does not have: {' '.join(missing)}")
    out = {}
    for target, file, _ in rows:
        if target not in CRITICAL:
            continue
        out[("tokenizers", file)] = target
        if file.endswith(".tiktoken"):
            stem = file[:-len(".tiktoken")] + "_"
            for d, name in table:
                if (d == "tokenizers" and name.startswith(stem)) or (target == "kimi-k3" and d == "kimik3"):
                    out[(d, name)] = target
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _TokenStaysOnHub(urllib.request.HTTPRedirectHandler):
    """urllib copies every header onto a redirect; the hub redirects file bodies to its cdn: drop the token there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != HF:
            new.remove_header("Authorization")
        return new


OPENER = urllib.request.build_opener(_TokenStaysOnHub)


def fetch(url, dest, sha):
    """download url to dest through dest.part, verified; returns (status, detail)."""
    headers = {"User-Agent": "toks-ci-fetch/1"}
    token = os.environ.get("HF_TOKEN")
    if token and urllib.parse.urlsplit(url).hostname == HF:
        headers["Authorization"] = "Bearer " + token
    detail = ""
    for attempt in range(4):                                   # bound: 4 attempts, 1 + 2 + 4 s apart
        if attempt:
            time.sleep(1 << (attempt - 1))
        try:
            h, n = hashlib.sha256(), 0
            with OPENER.open(urllib.request.Request(url, headers=headers), timeout=120) as r, \
                    open(dest + ".part", "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if h.hexdigest() != sha:
                os.remove(dest + ".part")
                return "bad-sha", f"{n} bytes with sha256 {h.hexdigest()}"
            os.replace(dest + ".part", dest)
            return "fetched", f"{n} bytes"
        except urllib.error.HTTPError as e:                    # 401 / 403 / 404 are answers; 429 and 5xx are retried
            code = (e.headers.get("X-Error-Code") if e.headers else None) or ""
            detail = f"HTTP {e.code} {code}".strip()
            if code in ("RepoNotFound", "RevisionNotFound", "EntryNotFound") or e.code in (404, 410):
                return "missing", detail
            if code == "GatedRepo" or e.code in (401, 403):
                return "gated", detail + (" (HF_TOKEN set)" if token else " (no HF_TOKEN)")
        except (urllib.error.URLError, OSError) as e:          # a transient network failure: retried
            detail = str(e)
    if os.path.exists(dest + ".part"):
        os.remove(dest + ".part")
    return "error", detail


def main():
    ap = argparse.ArgumentParser(description="every pinned tokenizer file, fetched and verified (docs/ci.md)")
    ap.add_argument("--check", action="store_true", help="verify the cache, fetch nothing")
    ap.add_argument("--key", action="store_true", help="print the sha256 of the pin table and exit")
    ap.add_argument("-j", type=int, default=8, help="parallel downloads (default 8)")
    args = ap.parse_args()

    table, errors = pin_table()
    if args.key:
        print(hashlib.sha256("".join(f"{d}/{n} {u} {s}\n" for (d, n), (u, s) in sorted(table.items()))
                             .encode()).hexdigest())
        return 0 if not errors else 1
    rows = ledger()
    for target, file, sha in rows:
        pin = table.get(("tokenizers", file))
        if pin is None:
            errors.append(f"ledger {target}: {file} has no pin in any source (tools/corpora, tests/spm)")
        elif pin[1] != sha:
            errors.append(f"ledger {target}: {file} sha256 {sha}, the fetch pin is {pin[1]}")
    crit = critical_files(table, rows)
    dirs = cache_dirs()

    def one(key):
        d, name = key
        url, sha = table[key]
        dest = os.path.join(dirs[d], name)
        if os.path.isfile(dest):
            got = sha256_file(dest)
            if got == sha:
                return key, "ok", f"{os.path.getsize(dest)} bytes"
            if args.check:
                return key, "bad-sha", f"sha256 {got} on disk"
        elif args.check:
            return key, "absent", ""
        return (key,) + fetch(url, dest, sha)

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.j)) as ex:
        results = sorted(ex.map(one, table))
    secs = time.time() - t0

    counts, crit_bad, gated = {}, [], set()
    print(f"{'status':8} {'file':48} {'critical':14} detail")
    for (d, name), status, detail in results:
        counts[status] = counts.get(status, 0) + 1
        target = crit.get((d, name), "")
        print(f"{status:8} {d + '/' + name:48} {target:14} {detail}")
        if status not in ("ok", "fetched") and target:
            crit_bad.append(f"{d}/{name} ({target}: {status})")
        if status == "gated":
            gated.add(table[(d, name)][0].split("/resolve/")[0])
    print(f"fetch_tokenizers: {len(results)} pinned files ({len(crit)} for {len(CRITICAL)} critical targets), "
          f"ledger {len(rows)} targets; " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
          + f"; {secs:.1f} s; {dirs['tokenizers']}, {dirs['kimik3']}")
    for g in sorted(gated):
        print(f"fetch_tokenizers: gated: {g} (HF_TOKEN: a read token of an account that accepted the repo's terms)")
    for e in errors:
        print(f"fetch_tokenizers: ERROR {e}")
    if crit_bad:
        print("fetch_tokenizers: FAIL: critical targets without their pinned file: " + "; ".join(crit_bad))
    return 1 if errors or crit_bad else 0


if __name__ == "__main__":
    sys.exit(main())
