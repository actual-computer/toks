#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""tools/census/coverage.py -- the toks coverage census (SPEC 1.1, 1.3, 1.4; docs/coverage.md).

Population (SPEC 1.1):
  (b) the top 500 text-generation models and the top 100 embedding / reranker models on the hf hub by
      downloads (the hub's 30-day count at census time);
  (c) the named families, whatever their rank;
  (d) an extension slice, reported separately and never mixed into (b)'s percentages: the top 100
      image-text-to-text / any-to-any models (multimodal llms whose text side is an llm tokenizer).
For every model: the tokenizer it loads (tokenizer.json; else tekken.json / tiktoken / remote code /
sentencepiece-only / slow vocab files; a repo with none of them -- gguf, adapters -- resolves through its
base_model). Tokenizers are deduplicated by the sha-256 of their normalized configuration: hf tokenizers
0.23.2's own re-serialization (Tokenizer.from_str(file).to_str()), canonical json, "version" dropped. Each
distinct tokenizer gets its component signature (model + flags, normalizer chain, pre-tokenizer chain with the
exact regex strings, decoder chain, post-processor, added-token options), and the report ranks patterns and
components by models and summed downloads against what toks covers today.

Anonymous hub access only: a gated repo is resolved through an ungated repo whose file is byte-identical
(content hash from the hub's own blob listing); without one it is listed as gated.

Phases (each resumable, state under $TOKS_CENSUS_CACHE, default ~/.cache/toks/census/coverage):
  collect   hub listings + named families               -> population.json
  resolve   each model -> tokenizer source + content key -> resolved.json
  fetch     download each distinct file once             -> files/<key>
  analyze   normalized config, hash, signature           -> analyzed.json
  report    ranking, coverage, the ordered work list     -> census/coverage.json, docs/coverage.md

  uv run tools/census/coverage.py all              # network phases: run on a lab host (maintainer doctrine)
  uv run tools/census/coverage.py report --analyzed analyzed.json   # report only, from a copied state
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

TOOL = "tools/census/coverage.py"
VERSION = "coverage-1"
HUB = "https://huggingface.co"
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("TOKS_CENSUS_CACHE", str(Path.home() / ".cache/toks/census"))) / "coverage"
FILES = CACHE / "files"

N_TEXT = 500
N_EMBED = 100
N_VLM = 100
EMBED_TAGS = ("sentence-similarity", "feature-extraction", "text-ranking")
VLM_TAGS = ("image-text-to-text", "any-to-any")
EXPAND = ("downloads", "gated", "siblings", "tags", "pipeline_tag", "library_name", "cardData")

# SPEC 1.1(c). The first repo is the family's canonical one; a gated canonical repo is resolved through a
# byte-identical ungated copy. tiktoken: the openai encoding the hub conversion stands for.
FAMILIES = [
    ("gpt-2 / r50k", ["openai-community/gpt2"], "r50k_base"),
    ("cl100k", ["Xenova/gpt-4", "Xenova/text-embedding-ada-002"], "cl100k_base"),
    ("o200k", ["Xenova/gpt-4o"], "o200k_base"),
    ("gpt-oss (o200k harmony)", ["openai/gpt-oss-20b"], None),
    ("llama 2", ["meta-llama/Llama-2-7b-hf"], None),
    ("llama 3", ["meta-llama/Meta-Llama-3-8B"], None),
    ("llama 3.3", ["meta-llama/Llama-3.3-70B-Instruct"], None),
    ("llama 4", ["meta-llama/Llama-4-Scout-17B-16E-Instruct"], None),
    ("qwen 2", ["Qwen/Qwen2-7B-Instruct"], None),
    ("qwen 2.5", ["Qwen/Qwen2.5-7B-Instruct"], None),
    ("qwen 3", ["Qwen/Qwen3-8B"], None),
    ("qwen 3.5", ["Qwen/Qwen3.5-9B"], None),
    ("qwen 3.6", ["Qwen/Qwen3.6-35B-A3B"], None),
    ("qwen 3.8", ["Qwen/Qwen3.8-27B"], None),
    ("deepseek v3", ["deepseek-ai/DeepSeek-V3"], None),
    ("deepseek r1", ["deepseek-ai/DeepSeek-R1"], None),
    ("deepseek v4", ["deepseek-ai/DeepSeek-V4-Flash"], None),
    ("kimi k2", ["moonshotai/Kimi-K2-Instruct"], None),
    ("glm 4", ["zai-org/glm-4-9b-chat-hf", "THUDM/glm-4-9b-chat-hf"], None),
    ("glm 4.5-4.7", ["zai-org/GLM-4.7-Flash"], None),
    ("glm 5", ["zai-org/GLM-5"], None),
    ("nemotron 3", ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"], None),
    ("olmo 2", ["allenai/OLMo-2-1124-7B-Instruct"], None),
    ("olmo 3", ["allenai/Olmo-3-7B-Instruct"], None),
    ("phi-3", ["microsoft/Phi-3-mini-4k-instruct"], None),
    ("phi-4", ["microsoft/phi-4"], None),
    ("mistral v0.3", ["mistralai/Mistral-7B-Instruct-v0.3"], None),
    ("mistral tekken", ["mistralai/Mistral-Nemo-Instruct-2407"], None),
    ("gemma 1", ["google/gemma-7b"], None),
    ("gemma 2", ["google/gemma-2-9b"], None),
    ("gemma 3", ["google/gemma-3-4b-it"], None),
    ("gemma 4", ["google/gemma-4-26B-A4B-it"], None),
    ("modernbert", ["answerdotai/ModernBERT-base"], None),
    ("bert", ["google-bert/bert-base-uncased"], None),
]

# the critical targets (they lead every order): the exact repo, its family's other sizes, and the popular
# quantized / mirror repos, each fingerprinted and compared with the target's own tokenizer.
CRITICAL = [
    ("GLM 5.3", "zai-org/GLM-5.3", "GLM-5.3"),
    ("Kimi K3", "moonshotai/Kimi-K3", "Kimi-K3"),
    ("gpt-oss", "openai/gpt-oss-20b", "gpt-oss"),
    ("Qwen 3.8", "Qwen/Qwen3.8-27B", "Qwen3.8"),
    ("Gemma 4", "google/gemma-4-26B-A4B-it", "gemma-4"),
]
CRIT_ORGS = ("unsloth", "nvidia", "lmstudio-community", "mlx-community", "bartowski", "RedHatAI")
CRIT_TOP = 30

SPM_FILES = ("tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "sentencepiece.model",
             "spm.model", "source.spm")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gitsha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def load(name, default=None):
    p = CACHE / name
    return json.loads(p.read_text()) if p.exists() else default


def save(name, obj):
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(CACHE / name)


# ---------------------------------------------------------------------------------------------------------
# hub client: anonymous, paced by the hub's own ratelimit header ("api";r=<left>;t=<seconds to reset>)

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.wait_until = {"api": 0.0, "resolvers": 0.0}
        self.calls = {"api": 0, "resolvers": 0}
        self.noredir = urllib.request.build_opener(_NoRedirect)
        self.opener = urllib.request.build_opener()

    def _pace(self, bucket):
        while True:
            with self.lock:
                w = self.wait_until[bucket] - time.time()
            if w <= 0:
                return
            time.sleep(min(w, 5))

    def _note(self, bucket, headers):
        rl = headers.get("ratelimit") if headers else None
        if not rl:
            return
        m = re.search(r'"(\w+)";r=(\d+);t=(\d+)', rl)
        if m and int(m.group(2)) <= 3:
            with self.lock:
                self.wait_until[m.group(1) if m.group(1) in self.wait_until else bucket] = \
                    time.time() + int(m.group(3)) + 2
            log(f"hub: {m.group(1)} budget exhausted, pausing {m.group(3)}s")

    def request(self, url, method="GET", bucket="api", follow=True, tries=8, headers=None):
        for attempt in range(tries):
            self._pace(bucket)
            req = urllib.request.Request(url, method=method,
                                         headers={"User-Agent": "toks-census/1", **(headers or {})})
            with self.lock:
                self.calls[bucket] += 1
            try:
                op = self.opener if follow else self.noredir
                with op.open(req, timeout=120) as r:
                    body = r.read() if method == "GET" else b""
                    self._note(bucket, r.headers)
                    return r.status, r.headers, body
            except urllib.error.HTTPError as e:
                self._note(bucket, e.headers)
                if e.code in (301, 302, 303, 307, 308) and not follow:
                    return e.code, e.headers, b""
                if e.code == 429 or e.code >= 500:
                    t = 30 * (attempt + 1)
                    m = re.search(r";t=(\d+)", (e.headers or {}).get("ratelimit", "") or "")
                    if e.code == 429 and m:
                        t = int(m.group(1)) + 2
                    with self.lock:
                        self.wait_until[bucket] = max(self.wait_until[bucket], time.time() + t)
                    log(f"hub: {e.code} on {url[:120]}, retry in {t}s")
                    continue
                return e.code, e.headers, b""
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                log(f"hub: {type(e).__name__} {e} on {url[:120]}, retry")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"hub: giving up on {url}")

    def list_models(self, params, limit, expand=EXPAND):
        q = dict(params)
        q.update({"sort": "downloads", "direction": "-1", "limit": str(min(limit, 1000))})
        url = HUB + "/api/models?" + urllib.parse.urlencode(q) + "".join(f"&expand[]={e}" for e in expand)
        out = []
        while url and len(out) < limit:
            st, hd, body = self.request(url)
            if st != 200:
                raise RuntimeError(f"list {params}: http {st}")
            out += json.loads(body)
            m = re.search(r'<([^>]+)>;\s*rel="next"', hd.get("link", "") or "")
            url = m.group(1) if m else None
        return out[:limit]

    def model_info(self, repo, blobs=False):
        q = "?blobs=true" if blobs else "?" + "&".join(f"expand[]={e}" for e in EXPAND)
        st, _, body = self.request(f"{HUB}/api/models/{repo}{q}")
        if st != 200:
            return {"_status": st}
        return json.loads(body)

    def head_key(self, repo, path):
        """content key of a file from the resolve endpoint: sha256:<hex> (lfs / xet) or git:<sha1>."""
        url = f"{HUB}/{repo}/resolve/main/{urllib.parse.quote(path)}"
        for _hop in range(4):
            st, hd, _ = self.request(url, method="HEAD", bucket="resolvers", follow=False)
            loc = hd.get("location") if hd else None
            if st in (301, 302, 307, 308) and loc and not hd.get("x-linked-etag") and "/resolve/" in loc:
                url = urllib.parse.urljoin(url, loc)        # a renamed repo (THUDM -> zai-org, case)
                continue
            break
        if st in (401, 403):
            return None, "gated"
        if st == 404:
            return None, "missing"
        et = (hd.get("x-linked-etag") or hd.get("etag") or "").strip('"').replace("W/", "").strip('"')
        size = hd.get("x-linked-size") or hd.get("content-length")
        if re.fullmatch(r"[0-9a-f]{64}", et):
            return {"key": "sha256:" + et, "size": int(size) if size else None}, "ok"
        if re.fullmatch(r"[0-9a-f]{40}", et):
            return {"key": "git:" + et, "size": None}, "ok"
        return None, f"no-etag-{st}"

    def download(self, repo, path, dest: Path):
        url = f"{HUB}/{repo}/resolve/main/{urllib.parse.quote(path)}"
        st, _, body = self.request(url, bucket="resolvers")
        if st != 200:
            raise RuntimeError(f"download {repo}/{path}: http {st}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return body


def gguf_header(buf: bytes) -> dict:
    """the metadata of a gguf file's first bytes, up to the token list (no tensor data needed)."""
    import struct
    if buf[:4] != b"GGUF":
        return {"error": "not gguf"}
    out = {"version": struct.unpack_from("<I", buf, 4)[0]}
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rstr(off):
        n = struct.unpack_from("<Q", buf, off)[0]
        return buf[off + 8:off + 8 + n].decode("utf-8", "replace"), off + 8 + n

    def rval(t, off):
        if t in fmt:
            return struct.unpack_from(fmt[t], buf, off)[0], off + struct.calcsize(fmt[t])
        if t == 8:
            return rstr(off)
        if t == 9:
            et, n = struct.unpack_from("<IQ", buf, off)
            off += 12
            if et in fmt:
                return {"n": n}, off + struct.calcsize(fmt[et]) * n
            for _ in range(n):
                _, off = rval(et, off)
            return {"n": n}, off
        raise ValueError(f"gguf type {t}")
    try:
        n_kv = struct.unpack_from("<Q", buf, 16)[0]
        off = 24
        for _ in range(n_kv):
            k, off = rstr(off)
            t = struct.unpack_from("<I", buf, off)[0]
            off += 4
            if k == "tokenizer.ggml.tokens":
                out["n_tokens"] = struct.unpack_from("<IQ", buf, off)[1]
                break
            v, off = rval(t, off)
            if k.startswith(("general.base_model", "general.source", "tokenizer.ggml.")) or \
                    k in ("general.architecture", "general.name", "general.basename"):
                out[k] = v
    except (struct.error, ValueError, UnicodeError) as e:
        out["truncated"] = f"{type(e).__name__}"
    return out


NAME_SUFFIXES = re.compile(r"(?i)([-_.](i?mat|imatrix|gguf|ggml|dspark|dflash|eagle3?|mtp|nvfp4|fp8|awq|gptq|"
                           r"int4|int8|w4a16|w8a8|bnb|4bit|8bit|mlx|onnx|exl2|uncensored|abliterated))+$")


def slim(m: dict, slices=()):
    """the fields of a hub model record the census keeps."""
    card = m.get("cardData") or {}
    bases = card.get("base_model") or []
    if isinstance(bases, str):
        bases = [bases]
    for t in m.get("tags") or []:
        if t.startswith("base_model:"):
            b = t.split(":")[-1]
            if "/" in b and b not in bases:
                bases.append(b)
    return {
        "id": m["id"], "downloads": m.get("downloads") or 0, "gated": m.get("gated") or False,
        "pipeline_tag": m.get("pipeline_tag"), "library_name": m.get("library_name"),
        "siblings": sorted(s["rfilename"] for s in m.get("siblings") or []),
        "base_models": [b for b in bases if isinstance(b, str)],
        "tags": [t for t in m.get("tags") or [] if t in ("gguf", "mlx", "onnx", "peft", "transformers.js")],
        "slices": list(slices),
    }


# ---------------------------------------------------------------------------------------------------------
# phase: collect

def phase_collect(hub: Hub, args):
    pop = load("population.json")
    if pop and not args.force:
        log(f"collect: cached ({len(pop['models'])} models)")
        return pop
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = []
    models = {}

    def add(m, sl):
        r = models.get(m["id"])
        if r is None:
            models[m["id"]] = slim(m, [sl])
        elif sl not in r["slices"]:
            r["slices"].append(sl)

    log("collect: text-generation")
    q = {"pipeline_tag": "text-generation"}
    queries.append({"slice": "textgen", "params": q, "limit": args.text})
    for m in hub.list_models(q, args.text):
        add(m, "textgen")
    # embedding / reranker candidates: 3x the target so non-text repos can be skipped in resolve
    cand = {}
    for tag in EMBED_TAGS:
        log(f"collect: {tag}")
        q = {"pipeline_tag": tag}
        queries.append({"slice": "embed-candidates", "params": q, "limit": args.embed * 3})
        for m in hub.list_models(q, args.embed * 3):
            cand.setdefault(m["id"], m)
    log("collect: filter=reranker")
    queries.append({"slice": "embed-candidates", "params": {"filter": "reranker"}, "limit": args.embed})
    for m in hub.list_models({"filter": "reranker"}, args.embed):
        cand.setdefault(m["id"], m)
    embed_order = sorted(cand.values(), key=lambda m: -(m.get("downloads") or 0))
    for m in embed_order:
        r = slim(m, [])
        models.setdefault(m["id"], r)
    vlm = {}
    for tag in VLM_TAGS:
        log(f"collect: {tag}")
        q = {"pipeline_tag": tag}
        queries.append({"slice": "vlm", "params": q, "limit": args.vlm})
        for m in hub.list_models(q, args.vlm):
            vlm.setdefault(m["id"], m)
    for m in sorted(vlm.values(), key=lambda m: -(m.get("downloads") or 0))[: args.vlm]:
        add(m, "vlm")
    fams = []
    for name, repos, tk in FAMILIES:
        chosen = None
        for r in repos:
            info = models.get(r)
            if info is None:
                mi = hub.model_info(r)
                if "_status" in mi:
                    log(f"collect: family {name}: {r} -> http {mi['_status']}")
                    continue
                info = slim(mi, [])
                models[r] = info
            chosen = r
            break
        if chosen:
            if "family" not in models[chosen]["slices"]:
                models[chosen]["slices"].append("family")
        fams.append({"family": name, "repo": chosen, "tiktoken": tk})
    pop = {"date": when, "queries": queries, "models": models, "embed_order": [m["id"] for m in embed_order],
           "families": fams, "targets": {"textgen": args.text, "embed": args.embed, "vlm": args.vlm}}
    save("population.json", pop)
    log(f"collect: {len(models)} repos, api calls {hub.calls}")
    return pop


def phase_critical(hub: Hub, args):
    """adds the critical targets' repos (slice "crit") to the population: the target, the most downloaded repos
    whose id carries the family stem, and every such repo of the usual quantizers / mirrors."""
    pop = load("population.json")
    crit = []
    for name, repo, stem in CRITICAL:
        found = {}
        for m in hub.list_models({"search": stem}, 100):
            found.setdefault(m["id"], m)
        for org in CRIT_ORGS:
            for m in hub.list_models({"author": org, "search": stem}, 25):
                found.setdefault(m["id"], m)
        if repo not in found:
            mi = hub.model_info(repo)
            if "_status" not in mi:
                found[repo] = mi
        keep = sorted((m for m in found.values() if stem.lower() in m["id"].lower()),
                      key=lambda m: -(m.get("downloads") or 0))
        sel = {m["id"]: m for m in keep[:CRIT_TOP]}
        for m in keep[CRIT_TOP:]:
            if m["id"].split("/")[0] in CRIT_ORGS and len(sel) < CRIT_TOP + 25:
                sel[m["id"]] = m
        if repo in found:
            sel[repo] = found[repo]
        for rid, m in sel.items():
            r = pop["models"].get(rid)
            if r is None:
                pop["models"][rid] = slim(m, ["crit"])
            elif "crit" not in r["slices"]:
                r["slices"].append("crit")
        crit.append({"target": name, "repo": repo, "stem": stem,
                     "repos": sorted(sel, key=lambda x: (x != repo, -(sel[x].get("downloads") or 0)))})
        log(f"critical: {name}: {len(sel)} repos")
    pop["critical"] = crit
    pop["queries"].append({"slice": "crit", "params": {"search": [c[2] for c in CRITICAL],
                                                       "author": list(CRIT_ORGS)}, "limit": CRIT_TOP})
    save("population.json", pop)
    return pop


# ---------------------------------------------------------------------------------------------------------
# phase: resolve

def artifacts(sibs):
    root = [s for s in sibs if "/" not in s]
    a = {}
    if "tokenizer.json" in root:
        a["tokenizer.json"] = "tokenizer.json"
    else:
        sub = sorted((s for s in sibs if s.endswith("/tokenizer.json") and s.count("/") == 1),
                     key=lambda s: (not s.startswith(("tokenizer/", "onnx/")), s))
        if sub:
            a["tokenizer.json"] = sub[0]
    if "tekken.json" in root:
        a["tekken"] = "tekken.json"
    tk = [s for s in root if s.endswith(".tiktoken") or s == "tiktoken.model"]
    if tk:
        a["tiktoken"] = tk[0]
    py = [s for s in root if s.startswith("tokenization_") and s.endswith(".py")]
    if py:
        a["remote-code"] = py[0]
    spm = [s for s in root if s in SPM_FILES]
    if spm:
        a["sentencepiece-only"] = spm[0]
    if "vocab.json" in root and "merges.txt" in root:
        a["bpe-vocab-merges-only"] = "vocab.json"
    if "vocab.txt" in root:
        a["wordpiece-vocab-only"] = "vocab.txt"
    if any(s.endswith(".gguf") for s in sibs):
        a["gguf"] = True
    return a


KIND_ORDER = ("tokenizer.json", "tekken", "tiktoken", "remote-code", "sentencepiece-only",
              "bpe-vocab-merges-only", "wordpiece-vocab-only")


def primary_kind(a):
    for k in KIND_ORDER:
        if k in a:
            return k
    return None


def phase_resolve(hub: Hub, pop, args):
    res = load("resolved.json", {}) if not args.force else {}
    models = pop["models"]
    info_cache = load("info_cache.json", {})

    def info(repo):
        if repo in models:
            return models[repo]
        if repo not in info_cache:
            mi = hub.model_info(repo)
            info_cache[repo] = slim(mi, []) if "_status" not in mi else {"_status": mi["_status"]}
        return info_cache[repo]

    gguf_seen = {}

    def gguf_meta(repo, sibs):
        ggs = sorted((s for s in sibs if s.endswith(".gguf") and "mmproj" not in s.lower()), key=len)
        if not ggs:
            return {}
        url = f"{HUB}/{repo}/resolve/main/{urllib.parse.quote(ggs[0])}"
        st, _, body = hub.request(url, bucket="resolvers", headers={"Range": "bytes=0-4194303"})
        return gguf_header(body) if st in (200, 206) else {"error": f"http {st}"}

    def name_match(repo):
        """a repo with no tokenizer and no declared base: the same model name without a packaging suffix
        (-GGUF, -DSpark, -FP8, ...), same org first, then the most downloaded exact name."""
        org, name = repo.split("/")
        base = NAME_SUFFIXES.sub("", name)
        if base == name or len(base) < 4:
            return None
        found = hub.list_models({"search": base}, 30, expand=("downloads", "siblings"))
        exact = [c for c in found if c["id"].split("/")[1].lower() == base.lower()
                 and primary_kind(artifacts([x["rfilename"] for x in c.get("siblings") or []]))]
        exact.sort(key=lambda c: (c["id"].split("/")[0] != org, -(c.get("downloads") or 0)))
        return exact[0]["id"] if exact else None

    def resolve_one(repo):
        """-> {kind, repo (where the file lives), path, via [...], gated, key|None, status}"""
        via = []
        cur = repo
        seen = set()
        while cur and cur not in seen and len(via) < 5:
            seen.add(cur)
            m = info(cur)
            if "_status" in m:
                return {"kind": None, "status": f"base-missing-{m['_status']}", "via": via}
            a = artifacts(m["siblings"])
            k = primary_kind(a)
            if k:
                r = {"kind": k, "repo": cur, "path": a[k], "via": via, "gated": bool(m["gated"]),
                     "also": sorted(x for x in a if x not in (k, "gguf"))}
                if k == "tiktoken" and "remote-code" in a:
                    r["py"] = a["remote-code"]
                if repo in gguf_seen:
                    r["gguf"] = gguf_seen[repo]
                if k == "remote-code":
                    for f in ("tokenizer.model", "tiktoken.model"):
                        if f in m["siblings"]:
                            r["model_file"] = f
                return r
            nxt = next((b for b in m["base_models"] if b != cur and b not in seen), None)
            how = "base_model"
            gg = None
            if not nxt and a.get("gguf"):
                gg = gguf_meta(cur, m["siblings"])
                for k2, v2 in sorted(gg.items()):
                    if k2.startswith(("general.base_model.", "general.source.")) and k2.endswith(
                            ("repo_url", ".url")) and isinstance(v2, str) and "huggingface.co/" in v2:
                        cand = v2.split("huggingface.co/")[1].strip("/")
                        if cand.count("/") == 1 and cand != cur and cand not in seen:
                            nxt, how = cand, "gguf-header"
                            break
            if not nxt:
                nxt = name_match(cur)
                how = "name"
            if not nxt:
                return {"kind": None, "status": "no-tokenizer" + ("-gguf" if a.get("gguf") else ""),
                        "via": via, "gguf": gg}
            via.append(nxt if how == "base_model" else f"{nxt} ({how})")
            if gg:
                gguf_seen[repo] = gg
            cur = nxt
        return {"kind": None, "status": "base-chain-loop", "via": via}

    # the embedding slice: the top N_EMBED text models in download order
    todo = [r for r, m in models.items() if m["slices"]]
    embed_ids = []
    for rid in pop["embed_order"]:
        if len(embed_ids) >= pop["targets"]["embed"]:
            break
        r = res.get(rid)
        if not (r and r.get("kind")):
            r = resolve_one(rid)
        res[rid] = r
        if r.get("kind"):
            embed_ids.append(rid)
            if "embed" not in models[rid]["slices"]:
                models[rid]["slices"].append("embed")
        else:
            models[rid].setdefault("skipped", r.get("status"))
    pop["embed_ids"] = embed_ids
    for rid in todo + embed_ids:
        if rid not in res or not res[rid].get("kind"):
            res[rid] = resolve_one(rid)
    save("info_cache.json", info_cache)

    # content keys: HEAD for ungated files; the blob listing for gated ones
    def key_of(r):
        if not r.get("kind") or r.get("key"):
            return
        if not r["gated"]:
            kk, st = hub.head_key(r["repo"], r["path"])
            r["status"] = st
            if kk:
                r.update(kk)
            elif st == "gated":
                r["gated"] = True          # listed ungated, but the file answers 401: treat as gated
        if r["gated"]:
            mi = hub.model_info(r["repo"], blobs=True)
            for s in mi.get("siblings") or []:
                if s["rfilename"] == r["path"]:
                    lfs = s.get("lfs")
                    r["key"] = ("sha256:" + lfs["sha256"]) if lfs else ("git:" + s["blobId"])
                    r["size"] = s.get("size")
            r["status"] = "gated" if r.get("key") else "gated-no-blob"
            return
        if r["kind"] == "tiktoken" and r.get("py"):
            k2, _ = hub.head_key(r["repo"], r["py"])
            if k2:
                r["py_key"] = k2["key"]

    for c in pop.get("critical", []):
        for rid in c["repos"]:
            m, r = models.get(rid), res.get(rid)
            if m and r is not None and any(x.endswith(".gguf") for x in m["siblings"]) and not r.get("gguf"):
                r["gguf"] = gguf_meta(rid, m["siblings"])
    pend = [r for r in res.values() if r.get("kind") and not r.get("key")]
    log(f"resolve: {len(res)} repos, {len(pend)} content keys to fetch")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(key_of, pend))
    save("resolved.json", res)
    save("population.json", pop)
    log(f"resolve: done, calls {hub.calls}")
    return res


# ---------------------------------------------------------------------------------------------------------
# phase: fetch (+ gated mirrors)

def file_path(key):
    return FILES / key.replace(":", "_")


def phase_fetch(hub: Hub, pop, res, args):
    have = load("have.json", {})        # any key (sha256 / git) -> canonical sha256 key of the file on disk

    def got(key):
        return key in have and file_path(have[key]).exists()

    def fetch(repo, path, key=None):
        if key and got(key):
            return have[key]
        tmp = FILES / ("dl_" + sha256(f"{repo}/{path}".encode())[:16])
        body = hub.download(repo, path, tmp)
        s, g = "sha256:" + sha256(body), "git:" + gitsha1(body)
        if key and key not in (s, g):
            log(f"fetch: {repo}/{path}: content {s[:20]} / {g[:16]} != key {key[:24]} (file changed?)")
        dest = file_path(s)
        tmp.replace(dest)
        have[s] = s
        have[g] = s
        if key:
            have[key] = s
        return s

    jobs = {}
    for rid, r in res.items():
        if r.get("kind") and r.get("key") and not r["gated"] and not got(r["key"]):
            jobs.setdefault(r["key"], (r["repo"], r["path"]))
        if r.get("py_key") and not got(r["py_key"]):
            jobs.setdefault(r["py_key"], (r["repo"], r["py"]))
    log(f"fetch: {len(jobs)} distinct files")
    lock = threading.Lock()

    def job(item):
        key, (repo, path) = item
        try:
            fetch(repo, path, key)
        except Exception as e:  # noqa: BLE001 -- recorded, reported as a miss
            with lock:
                log(f"fetch: FAILED {repo}/{path}: {e}")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(job, jobs.items()))
    save("have.json", have)

    # slow-only and remote-code repos: the tokenizer class transformers would build (tokenizer_config.json)
    def tclass(r):
        if r.get("kind") in (None, "tokenizer.json", "tekken") or "tokenizer_class" in r:
            return
        url = f"{HUB}/{r['repo']}/resolve/main/tokenizer_config.json"
        st, _, body = hub.request(url, bucket="resolvers")
        try:
            tc = json.loads(body) if st == 200 else {}
        except Exception:  # noqa: BLE001
            tc = {}
        r["tokenizer_class"] = tc.get("tokenizer_class")
        if r.get("kind") == "tiktoken":
            atd = tc.get("added_tokens_decoder") or {}
            names = sorted((int(k), v.get("content"), bool(v.get("special"))) for k, v in atd.items())
            r["tk_config"] = {"sha256": sha256(body) if st == 200 else None, "named_specials": len(names),
                              "names_sha256": sha256(canon(names).encode())[:16] if names else None,
                              "bos": str(tc.get("bos_token")), "eos": str(tc.get("eos_token"))}
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(tclass, [r for r in res.values() if not r.get("gated")]))

    # gated: a byte-identical ungated copy (same content key), found in the census or by a hub search
    mirrors = load("mirrors.json", {})
    for rid, r in res.items():
        if not (r.get("kind") and r["gated"]):
            continue
        key = r.get("key")
        if not key:
            continue
        if got(key):
            r["mirror"] = mirrors.get(key, {}).get("repo", "(census)")
            r["status"] = "gated-mirrored"
            continue
        if key in mirrors and mirrors[key].get("repo") is None:
            r["status"] = "gated-no-mirror"
            if mirrors[key].get("standin"):
                r["standin"] = mirrors[key]["standin"]
            continue
        name = r["repo"].split("/")[1]
        cands = [{"id": f"{org}/{name}"} for org in ("unsloth", "NousResearch")] + \
            [c for c in hub.list_models({"search": name}, 40, expand=("downloads", "gated", "siblings"))
             if c["id"] != r["repo"] and not c.get("gated")
             and r["path"] in [s["rfilename"] for s in c.get("siblings") or []]]
        found = standin = None
        tried = 0
        for c in cands:
            if tried >= 12:
                break
            kk, st = hub.head_key(c["id"], r["path"])
            if not kk:
                continue
            tried += 1
            if kk["key"] == key:
                fetch(c["id"], r["path"], kk["key"])
                found = c["id"]
                break
            same_name = c["id"].split("/")[1].lower() == name.lower()
            if kk["key"].split(":")[0] == key.split(":")[0]:
                standin = standin or ((c["id"], kk["key"]) if same_name else None)
                continue                        # same hash kind, different value: different bytes
            if r.get("size") and kk.get("size") and kk["size"] != r["size"]:
                standin = standin or ((c["id"], kk["key"]) if same_name else None)
                continue
            try:                                # git sha1 vs sha256: compare the bytes
                s = fetch(c["id"], r["path"], kk["key"])
            except Exception:  # noqa: BLE001
                continue
            if got(key) and have[key] == s:
                found = c["id"]
                break
            standin = standin or ((c["id"], kk["key"]) if same_name else None)
        mirrors[key] = {"repo": found, "for": r["repo"]}
        r["mirror"] = found
        r["status"] = "gated-mirrored" if found else "gated-no-mirror"
        if not found and standin:
            # not byte-identical: the report may show it for a named family, labelled as a stand-in
            try:
                r["standin"] = {"repo": standin[0], "key": standin[1], "sha256": fetch(standin[0], r["path"],
                                                                                        standin[1])}
                mirrors[key]["standin"] = r["standin"]
            except Exception:  # noqa: BLE001
                pass
        log(f"fetch: gated {r['repo']} -> {found} (stand-in {standin and standin[0]})")
        save("mirrors.json", mirrors)
        save("have.json", have)
    save("resolved.json", res)
    save("have.json", have)
    log(f"fetch: done, calls {hub.calls}")


# ---------------------------------------------------------------------------------------------------------
# phase: analyze -- one distinct file -> normalized configuration, hash, component signature

BYTE_CHARS = None


def byte_chars():
    """gpt-2's bytes_to_unicode: the 256 chars a byte-level vocabulary is written in."""
    global BYTE_CHARS
    if BYTE_CHARS is None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
            list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        BYTE_CHARS = [chr(c) for c in cs]
    return BYTE_CHARS


def flat(node, key):
    if not node:
        return []
    if node.get("type") == "Sequence":
        out = []
        for x in node.get(key) or []:
            out += flat(x, key)
        return out
    return [node]


def pattern_of(p):
    if not isinstance(p, dict):
        return {"?": p}
    if "Regex" in p:
        return {"Regex": p["Regex"]}
    if "String" in p:
        return {"String": p["String"]}
    return p


def desc_norm(n):
    t = n.get("type")
    if t == "Precompiled":
        raw = base64.b64decode(n.get("precompiled_charsmap") or "")
        return {"type": t, "charsmap_sha256": sha256(raw)[:16], "charsmap_bytes": len(raw)}
    if t == "Replace":
        return {"type": t, "pattern": pattern_of(n.get("pattern")), "content": n.get("content")}
    return {k: v for k, v in n.items()}


def desc_pre(p):
    t = p.get("type")
    d = {k: v for k, v in p.items() if k != "trim_offsets"}   # trim_offsets: offsets only, never ids
    if t == "Split":
        d["pattern"] = pattern_of(p.get("pattern"))
    return d


def desc_dec(d):
    t = d.get("type")
    out = {k: v for k, v in d.items()}
    if t == "Replace":
        out["pattern"] = pattern_of(d.get("pattern"))
    return out


def desc_template(tpl):
    parts = []
    for piece in tpl or []:
        if "Sequence" in piece:
            parts.append(f"${piece['Sequence'].get('id')}:{piece['Sequence'].get('type_id', 0)}")
        elif "SpecialToken" in piece:
            parts.append(f"{piece['SpecialToken'].get('id')}:{piece['SpecialToken'].get('type_id', 0)}")
        else:
            parts.append(json.dumps(piece, ensure_ascii=False))
    return " ".join(parts)


def desc_pp(p):
    t = p.get("type")
    if t == "TemplateProcessing":
        return {"type": t, "single": desc_template(p.get("single")), "pair": desc_template(p.get("pair")),
                "special_tokens": {k: v.get("ids") for k, v in (p.get("special_tokens") or {}).items()}}
    return {k: v for k, v in p.items()}


def algorithm_of(model, pres, norms):
    t = model.get("type")
    if t == "BPE":
        bytelevel = any(x.get("type") == "ByteLevel" for x in pres) or \
            any(x.get("type") == "ByteLevel" for x in norms)
        if bytelevel:
            return "bpe-bytelevel"
        if model.get("byte_fallback"):
            return "bpe-byte-fallback"
        return "bpe-plain"
    return {"Unigram": "unigram", "WordPiece": "wordpiece", "WordLevel": "wordlevel"}.get(t, f"?{t}")


def model_type(m):
    if m.get("type"):
        return m["type"]
    if "merges" in m:
        return "BPE"
    if isinstance(m.get("vocab"), list):
        return "Unigram"
    if "max_input_chars_per_word" in m:
        return "WordPiece"
    return "WordLevel"


def signature_tokenizer_json(cfg):
    model = dict(cfg.get("model") or {})
    model["type"] = model_type(model)
    norms = flat(cfg.get("normalizer"), "normalizers")
    pres = flat(cfg.get("pre_tokenizer"), "pretokenizers")
    decs = flat(cfg.get("decoder"), "decoders")
    pps = flat(cfg.get("post_processor"), "processors")
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        vset = vocab
        vsize = len(vocab)
    elif isinstance(vocab, list):
        vset = {e[0]: i for i, e in enumerate(vocab)}
        vsize = len(vocab)
    else:
        vset, vsize = {}, 0
    m = {"type": model["type"], "vocab_size": vsize}
    if model["type"] == "BPE":
        for k in ("byte_fallback", "ignore_merges", "fuse_unk", "dropout", "unk_token",
                  "continuing_subword_prefix", "end_of_word_suffix"):
            m[k] = model.get(k)
        m["merges"] = len(model.get("merges") or [])
    elif model["type"] == "Unigram":
        m["unk_id"] = model.get("unk_id")
        m["byte_fallback"] = model.get("byte_fallback")
    elif model["type"] == "WordPiece":
        for k in ("unk_token", "continuing_subword_prefix", "max_input_chars_per_word"):
            m[k] = model.get(k)
    elif model["type"] == "WordLevel":
        m["unk_token"] = model.get("unk_token")
    algo = algorithm_of(model, pres, norms)
    if algo == "bpe-bytelevel":
        bc = byte_chars()
        m["byte_alphabet"] = sum(1 for c in bc if c in vset)
        alpha = set(bc)
        m["non_alphabet_tokens"] = sum(1 for k in vset if any(ch not in alpha for ch in k))
    if model["type"] in ("BPE", "Unigram") and (model.get("byte_fallback")):
        m["byte_tokens"] = sum(1 for b in range(256) if f"<0x{b:02X}>" in vset)
    added = cfg.get("added_tokens") or []
    a = {"n": len(added), "special": 0, "lstrip": 0, "rstrip": 0, "single_word": 0, "normalized": 0,
         "normalized_special": 0, "not_in_vocab": 0, "max_bytes": 0, "with_space": 0}
    for t in added:
        a["special"] += bool(t.get("special"))
        for k in ("lstrip", "rstrip", "single_word", "normalized"):
            a[k] += bool(t.get(k))
        a["normalized_special"] += bool(t.get("normalized") and t.get("special"))
        a["not_in_vocab"] += t.get("content") not in vset
        a["max_bytes"] = max(a["max_bytes"], len((t.get("content") or "").encode()))
        a["with_space"] += any(ch.isspace() for ch in t.get("content") or "")
    opt_tokens = [{"content": t.get("content"), "id": t.get("id"), "special": t.get("special"),
                   **{k: t.get(k) for k in ("lstrip", "rstrip", "single_word", "normalized") if t.get(k)}}
                  for t in added if t.get("lstrip") or t.get("rstrip") or t.get("single_word")]
    a["option_tokens"] = opt_tokens[:12]
    return {
        "algorithm": algo,
        "model": m,
        "normalizer": [desc_norm(n) for n in norms],
        "pre_tokenizer": [desc_pre(p) for p in pres],
        "decoder": [desc_dec(d) for d in decs],
        "post_processor": [desc_pp(p) for p in pps],
        "added_tokens": a,
        "truncation": cfg.get("truncation"),
        "padding": cfg.get("padding"),
    }


def kimi_pattern(src: str):
    """the regex of a tiktoken-based tokenization_*.py: pat_str = "|".join([r"...", ...]) or a literal."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            name = getattr(tgt, "id", None) or getattr(tgt, "attr", None)
            if name in ("pat_str", "pattern", "PAT_STR"):
                v = node.value
                try:
                    return ast.literal_eval(v)
                except Exception:  # noqa: BLE001
                    pass
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "join":
                    try:
                        sep = ast.literal_eval(v.func.value)
                        parts = ast.literal_eval(v.args[0])
                        return sep.join(parts)
                    except Exception:  # noqa: BLE001
                        pass
    return None


def spm_info(raw: bytes):
    """the few sentencepiece ModelProto fields that name the algorithm (hand-rolled protobuf walk)."""
    def varint(b, i):
        x = s = 0
        while True:
            c = b[i]
            i += 1
            x |= (c & 0x7F) << s
            s += 7
            if c < 0x80:
                return x, i

    def fields(b):
        i = 0
        while i < len(b):
            k, i = varint(b, i)
            f, w = k >> 3, k & 7
            if w == 0:
                v, i = varint(b, i)
            elif w == 1:
                v, i = b[i:i + 8], i + 8
            elif w == 2:
                n, i = varint(b, i)
                v, i = b[i:i + n], i + n
            elif w == 5:
                v, i = b[i:i + 4], i + 4
            else:
                raise ValueError("wire type")
            yield f, w, v
    out = {"pieces": 0}
    try:
        for f, w, v in fields(raw):
            if f == 1 and w == 2:
                out["pieces"] += 1
            elif f == 2 and w == 2:      # trainer_spec
                for f2, w2, v2 in fields(v):
                    if f2 == 3 and w2 == 0:
                        out["model_type"] = {1: "unigram", 2: "bpe", 3: "word", 4: "char"}.get(v2, v2)
                    elif f2 == 35 and w2 == 0:
                        out["byte_fallback"] = bool(v2)
            elif f == 3 and w == 2:      # normalizer_spec
                for f2, w2, v2 in fields(v):
                    if f2 == 1 and w2 == 2:
                        out["normalizer"] = v2.decode("utf-8", "replace")
                    elif f2 == 3 and w2 == 0:
                        out["add_dummy_prefix"] = bool(v2)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out.setdefault("model_type", "unigram")   # proto default
    return out


def analyze_one(job):
    """(kind, path_on_disk, extra) -> record; runs in a worker process."""
    kind, path, extra = job
    raw = Path(path).read_bytes()
    rec = {"kind": kind, "file_sha256": sha256(raw), "bytes": len(raw)}
    if kind == "tokenizer.json":
        try:
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"json: {e}"[:300])
            return rec
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_str(raw.decode("utf-8"))
            norm = json.loads(tk.to_str())
            rec["hf_loads"] = True
        except Exception as e:  # noqa: BLE001
            norm = obj
            rec["hf_loads"] = False
            rec["hf_error"] = f"{type(e).__name__}: {e}"[:300]
        norm.pop("version", None)
        rec["hash"] = sha256(canon(norm).encode())
        rec["parts"] = {k: sha256(canon(norm.get(k)).encode())[:12] for k in
                        ("model", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder",
                         "truncation", "padding")}
        rec["sig"] = signature_tokenizer_json(norm)
        if rec["hf_loads"]:
            # the raw file's own spelling of the pre-tokenizer (hf may re-spell; the census reports both)
            raw_pres = [desc_pre(p) for p in flat(obj.get("pre_tokenizer"), "pretokenizers")]
            if raw_pres != rec["sig"]["pre_tokenizer"]:
                rec["raw_pre_tokenizer"] = raw_pres
        return rec
    if kind == "tekken":
        obj = json.loads(raw)
        c = obj.get("config") or {}
        rec["hash"] = sha256(canon({k: v for k, v in obj.items()}).encode())
        rec["sig"] = {"algorithm": "bpe-bytelevel(tiktoken ranks)", "tekken": {
            "pattern": c.get("pattern"), "version": c.get("version"),
            "num_vocab_tokens": c.get("num_vocab_tokens"), "default_vocab_size": c.get("default_vocab_size"),
            "default_num_special_tokens": c.get("default_num_special_tokens"),
            "special_tokens": len(obj.get("special_tokens") or []), "vocab": len(obj.get("vocab") or [])}}
        return rec
    if kind == "tiktoken":
        pat = None
        if extra.get("py_path"):
            try:
                pat = kimi_pattern(Path(extra["py_path"]).read_text())
            except Exception as e:  # noqa: BLE001
                rec["py_error"] = str(e)[:200]
        lines = raw.count(b"\n")
        rec["hash"] = sha256((rec["file_sha256"] + "\0" + (pat or "")).encode())
        rec["sig"] = {"algorithm": "bpe-bytelevel(tiktoken ranks)", "tiktoken": {"pattern": pat, "ranks": lines}}
        return rec
    if kind == "sentencepiece-only":
        rec["hash"] = rec["file_sha256"]
        rec["sig"] = {"algorithm": "sentencepiece-proto", "spm": spm_info(raw)}
        return rec
    rec["hash"] = rec["file_sha256"]
    rec["sig"] = {"algorithm": kind}
    if kind == "remote-code":
        try:
            rec["sig"]["py_pattern"] = kimi_pattern(raw.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001
            pass
    return rec


def phase_analyze(pop, res, args):
    have = load("have.json", {})
    done = load("analyzed.json", {}) if not args.force else {}
    jobs = {}
    for rid, r in res.items():
        if r.get("standin") and r["standin"].get("sha256") and r["standin"]["sha256"] not in done:
            jobs[r["standin"]["sha256"]] = (r["kind"], str(file_path(r["standin"]["sha256"])), {})
        cv = (r.get("converted") or {}).get("sha256")
        if cv and cv not in done and file_path(cv).exists():
            jobs[cv] = ("tokenizer.json", str(file_path(cv)), {})
        k = r.get("key")
        if not (r.get("kind") and k and k in have):
            continue
        s = have[k]
        if s in done:
            continue
        extra = {}
        if r.get("py_key") and r["py_key"] in have:
            extra["py_path"] = str(file_path(have[r["py_key"]]))
        jobs[s] = (r["kind"], str(file_path(s)), extra)
    log(f"analyze: {len(jobs)} files")
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.jobs) as ex:
        for s, rec in zip(jobs, ex.map(analyze_one, jobs.values())):
            done[s] = rec
            log(f"analyze: {s[:20]} {rec.get('kind')} {rec.get('sig', {}).get('algorithm')} "
                f"{'' if rec.get('hf_loads', True) else 'HF-FAIL ' + rec.get('hf_error', '')[:80]}")
    save("analyzed.json", done)
    return done


# ---------------------------------------------------------------------------------------------------------
# phase: convert -- repos that ship only slow files (sentencepiece .model, vocab.json + merges.txt, vocab.txt):
# the fast tokenizer transformers builds from them (AutoTokenizer, no remote code), so their components are
# known. Needs: uv run --with transformers --with sentencepiece --with protobuf tools/census/coverage.py convert

SLOW_KINDS = ("sentencepiece-only", "bpe-vocab-merges-only", "wordpiece-vocab-only")


def phase_convert(res, args):
    os.environ.setdefault("HF_HOME", str(CACHE.parent / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import transformers
    from transformers import AutoTokenizer
    conv = load("converted.json", {}) if not args.force else {}
    FILES.mkdir(parents=True, exist_ok=True)
    for rid, r in res.items():
        if r.get("kind") not in SLOW_KINDS or r.get("gated"):
            continue
        repo = r["repo"]
        if repo not in conv:
            try:
                tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
                bt = getattr(tk, "backend_tokenizer", None) or getattr(tk, "_tokenizer", None)
                b = bt.to_str().encode()
                key = "sha256:" + sha256(b)
                file_path(key).write_bytes(b)
                conv[repo] = {"sha256": key, "transformers": transformers.__version__, "class": type(tk).__name__}
            except Exception as e:  # noqa: BLE001 -- recorded; the model stays a miss
                conv[repo] = {"error": f"{type(e).__name__}: {e}"[:300]}
            log(f"convert: {repo}: {conv[repo]}")
        r["converted"] = conv[repo]
    save("converted.json", conv)
    save("resolved.json", res)


# ---------------------------------------------------------------------------------------------------------
# phase: probe -- toks's own load path (toks_load; tools/census/probe.c) on every tokenizer the census read:
# the "today, in code" answer next to the census's design-scope rules. tokenizer.json files (and the files
# transformers converts slow inputs into: SPEC 1.5 keeps those inputs out, the probe says what their conversion
# would meet) load as files; a tiktoken model loads as a directory holding the repo's own files under their own
# names (ranks file, tokenization_*.py, tokenizer_config.json), as toks_load sees a downloaded model. The census
# keeps no tokenizer_config.json bytes, so the probe fetches one per distinct content (sha-256 recorded at fetch
# time; a file that changed since is not used) into $TOKS_CENSUS_CACHE/coverage/tkconfig/.

def probe_tiktoken_dirs(hub: Hub, res):
    """-> {dir: [resolved ids]}: one model directory per distinct (ranks, wrapper, tokenizer_config.json)."""
    tkc = CACHE / "tkconfig"
    tkc.mkdir(parents=True, exist_ok=True)
    have = load("have.json", {})
    groups = {}
    for rid, r in res.items():
        ranks = have.get(r.get("key"))
        if r.get("kind") != "tiktoken" or r.get("status") not in ("ok", "gated-mirrored") or not ranks:
            continue
        cs = (r.get("tk_config") or {}).get("sha256")
        key = (ranks, r.get("path"), r.get("py_key"), r.get("py"), cs)
        groups.setdefault(key, []).append(rid)
    dirs = {}
    for n, ((ranks, path, py_key, py, cs), rids) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        d = REPO_ROOT / "build" / "census-tiktoken" / f"{n:02d}"
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)
        (d / Path(path or "tiktoken.model").name).symlink_to(file_path(ranks))
        if py_key and py and py_key in have and file_path(have[py_key]).exists():
            (d / Path(py).name).symlink_to(file_path(have[py_key]))
        if cs:
            got = tkc / cs
            if not got.exists():
                repo = res[rids[0]].get("repo")
                st, _, body = hub.request(f"{HUB}/{repo}/resolve/main/tokenizer_config.json", bucket="resolvers")
                if st == 200 and sha256(body) == cs:
                    got.write_bytes(body)
                else:
                    log(f"probe: {repo}/tokenizer_config.json: status {st}, content differs from the census's")
            if got.exists():
                (d / "tokenizer_config.json").symlink_to(got)
        dirs[str(d)] = rids
    return dirs


def phase_probe(hub: Hub, res, args):
    import subprocess
    import platform
    isa = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x86_64"}[platform.machine()]
    osn = "macos" if sys.platform == "darwin" else "linux"
    exe = REPO_ROOT / "build" / "census-probe"
    lib = REPO_ROOT / "build" / f"{osn}-{isa}" / "libtoks.a"
    subprocess.run(["make", "-C", str(REPO_ROOT), "-j4", "lib"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([os.environ.get("CC", "clang"), "-std=c17", "-O2", "-Wall", "-Wextra", "-Werror",
                    f"-I{REPO_ROOT}/include", f"-I{REPO_ROOT}/src/core", f"-I{REPO_ROOT}/src/platform",
                    str(REPO_ROOT / "tools/census/probe.c"), str(lib), "-lpthread", "-o", str(exe)], check=True)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    analyzed = load("analyzed.json", {})

    def run(paths):
        got = {}
        for i in range(0, len(paths), 50):
            p = subprocess.run([str(exe)] + paths[i:i + 50], capture_output=True, text=True, check=True)
            for line in p.stdout.splitlines():
                path, stage, code, what = line.split("\t", 3)
                got[path] = {"stage": int(stage), "code": int(code), "what": what}
        return got

    files = {str(file_path(s)): s for s, a in analyzed.items()     # converted slow inputs analyze as tokenizer.json
             if a.get("kind") == "tokenizer.json" and file_path(s).exists()}
    out = {files[p]: v for p, v in run(list(files)).items()}
    for s, a in analyzed.items():
        a.pop("probe", None)
        if s in out:
            a["probe"] = out[s]
    save("analyzed.json", analyzed)
    dirs = probe_tiktoken_dirs(hub, res)
    tk = run(list(dirs))
    for d, rids in dirs.items():
        for rid in rids:
            res[rid]["probe"] = tk[d]
    save("resolved.json", res)
    src = os.environ.get("CENSUS_COMMIT") or (head.stdout.strip() if head.returncode == 0 else "unknown")
    stages = {}
    for v in list(out.values()) + list(tk.values()):
        stages[str(v["stage"])] = stages.get(str(v["stage"]), 0) + 1
    save("probe.json", {"commit": src, "host": platform.node(), "files": len(out), "tiktoken_dirs": len(tk),
                        "stages": stages, "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    log(f"probe: {len(out)} files + {len(tk)} tiktoken model directories, stages {stages} (0 loads; 8 = toks_load, "
        "toks_load_mem_copy and the config parse disagree)")


# ---------------------------------------------------------------------------------------------------------
# phase: tiktoken -- the openai encodings' pattern strings as tiktoken itself spells them (tiktoken_ext/
# openai_public.py; the rank files are not downloaded), so the report checks the census's strings against the
# source instead of a transcription. Needs: uv run --with tiktoken tools/census/coverage.py tiktoken

def phase_tiktoken():
    import importlib.metadata
    import tiktoken_ext.openai_public as op
    op.load_tiktoken_bpe = lambda *a, **k: {}          # the patterns only
    pats = {}
    for name in ("r50k_base", "p50k_base", "cl100k_base", "o200k_base", "o200k_harmony"):
        fn = getattr(op, name, None)
        if fn is not None:
            pats[name] = fn()["pat_str"]
    save("tiktoken.json", {"tiktoken": importlib.metadata.version("tiktoken"), "patterns": pats})
    log(f"tiktoken {importlib.metadata.version('tiktoken')}: {', '.join(pats)}")


# ---------------------------------------------------------------------------------------------------------
# phase: state -- everything the report needs, in one file (the report runs anywhere from it)

def phase_state(pop, res):
    have = load("have.json", {})
    analyzed = load("analyzed.json", {})
    models = {}
    for rid, m in pop["models"].items():
        if not m["slices"] and rid not in res:
            continue
        mm = {k: v for k, v in m.items() if k != "siblings"}
        mm["artifacts"] = sorted(k for k in artifacts(m["siblings"]))
        models[rid] = mm
    for r in res.values():
        if r.get("key") in have:
            r["sha256"] = have[r["key"]]
    st = {"tool": TOOL, "version": VERSION, "date": pop["date"], "queries": pop["queries"],
          "targets": pop["targets"], "families": pop["families"], "embed_ids": pop.get("embed_ids", []),
          "models": models, "resolved": res, "analyzed": analyzed, "mirrors": load("mirrors.json", {}),
          "critical": pop.get("critical", []), "probe": load("probe.json", {}), "tiktoken": load("tiktoken.json", {})}
    save("state.json", st)
    log(f"state: {CACHE / 'state.json'}")
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("collect", "critical", "resolve", "fetch", "convert", "analyze", "probe",
                                      "tiktoken", "state", "report", "all"))
    ap.add_argument("--text", type=int, default=N_TEXT)
    ap.add_argument("--embed", type=int, default=N_EMBED)
    ap.add_argument("--vlm", type=int, default=N_VLM)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="redo the phase instead of resuming")
    ap.add_argument("--state", default=None, help="report: state.json to read (default: the cache's)")
    ap.add_argument("--out-json", default=str(REPO_ROOT / "census/coverage.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "docs/coverage.md"))
    args = ap.parse_args()
    hub = Hub()
    ph = args.phase
    if ph in ("collect", "all"):
        pop = phase_collect(hub, args)
    if ph in ("critical", "all"):
        pop = phase_critical(hub, args)
    if ph in ("resolve", "fetch", "convert", "analyze", "probe", "state", "all"):
        pop = load("population.json")
        if pop is None:
            raise SystemExit("run collect first")
    if ph in ("resolve", "all"):
        phase_resolve(hub, pop, args)
    if ph in ("fetch", "convert", "analyze", "probe", "state", "all"):
        res = load("resolved.json")
        if res is None:
            raise SystemExit("run resolve first")
    if ph in ("fetch", "all"):
        phase_fetch(hub, pop, res, args)
        res = load("resolved.json")
    if ph == "convert":
        phase_convert(res, args)
        res = load("resolved.json")
    if ph in ("analyze", "all"):
        phase_analyze(pop, res, args)
    if ph in ("probe", "all"):
        phase_probe(hub, res, args)
    if ph == "tiktoken":
        phase_tiktoken()
    if ph in ("state", "all"):
        phase_state(load("population.json"), load("resolved.json"))
    if ph in ("report", "all"):
        from report import report   # tools/census/report.py
        st = json.loads(Path(args.state).read_text()) if args.state else load("state.json")
        report(st, Path(args.out_json), Path(args.out_md))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
