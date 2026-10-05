"""tools/census/report.py -- census/coverage.json and docs/coverage.md from the coverage state.

Called by `tools/census/coverage.py report`; reads only the state file (no network), so it runs anywhere.
Every rule that decides "covered" lives here, next to the strings it compares against.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------------------
# what toks covers today: the cl100k template family of docs/kernels.md §3, as exact regex strings (the
# strings the regex engine sees, i.e. tokenizer.json's after json decoding). qwen 3.5 is kernels.md's prose
# ("as qwen 2 with [\p{L}\p{M}]+ for the letter run and [^\s\p{L}\p{M}\p{N}] for punctuation") applied to
# the qwen 2 string.

P_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
P_CL100K = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|\s+(?!\S)|\s+")
P_QWEN2 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
           r"\s*[\r\n]+|\s+(?!\S)|\s+")
P_QWEN35 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
            r"\s*[\r\n]+|\s+(?!\S)|\s+")
KERNELS_MD = {P_GPT2: "gpt-2", P_CL100K: "cl100k / llama 3", P_QWEN2: "qwen 2 / 2.5 / 3",
              P_QWEN35: "qwen 3.5 / 3.6 / 3.8 (marks)"}

# the o200k family (TOKS_TMPL_O200K is reserved in layout.h)
_O_UP = r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"
_O_LO = r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]"
_O_CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
# the deepseek v3 multi-split (three Splits then ByteLevel)
DS_DIGITS = r"\p{N}{1,3}"
DS_CJK = "[一-龥぀-ゟ゠-ヿ]+"
DS_MAIN = ("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
           " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+")
# kimi k2 / k2.5 / k3 / linear (tiktoken, pattern from tokenization_kimi.py)
P_KIMI = ("[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
          "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?"
          "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
          "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

TESTING_ORGS = ("trl-internal-testing", "hf-internal-testing", "peft-internal-testing", "hf-tiny-model-private",
                "sentence-transformers-testing", "cross-encoder-testing", "optimum-internal-testing",
                "tiny-random", "yujiepan", "katuni4ka", "fxmarty")


def cl100k_family():
    """every string the cl100k template's parameters (layout.h TOKS_TP_*, marks folding) can express."""
    out = {}
    contr = {"none": "", "cs": "'s|'t|'re|'ve|'m|'ll|'d|", "ci": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"}
    digits = {"1_3": r"\p{N}{1,3}", "1": r"\p{N}", "sp_run": r" ?\p{N}+", "run": r"\p{N}+"}
    for marks in (False, True):
        lr = r"[\p{L}\p{M}]+" if marks else r"\p{L}+"
        letters = {"space": " ?" + lr, "any": r"[^\r\n\p{L}\p{N}]?" + lr}
        if marks:
            letters["any_m"] = r"[^\r\n\p{L}\p{M}\p{N}]?" + lr
        pc = r"[^\s\p{L}\p{M}\p{N}]+" if marks else r"[^\s\p{L}\p{N}]+"
        for c, cs in contr.items():
            for lk, ls in letters.items():
                for dk, ds in digits.items():
                    for pn in (False, True):
                        for wn in (False, True):
                            pat = (cs + ls + "|" + ds + "| ?" + pc + (r"[\r\n]*" if pn else "") + "|" +
                                   (r"\s*[\r\n]+|" if wn else "") + r"\s+(?!\S)|\s+")
                            out[pat] = {"contr": c, "letters": lk, "marks": marks, "digits": dk,
                                        "punct_nl": pn, "ws_nl": wn}
    return out


def o200k_family():
    out = {}
    for cn, cs in (("contractions", _O_CONTR), ("no contractions", "")):
        for dn, ds in (("1_3", r"\p{N}{1,3}"), ("1", r"\p{N}")):
            for pn, ps in (("[\\r\\n/]*", r"[\r\n/]*"), ("[\\r\\n]*", r"[\r\n]*")):
                pat = (r"[^\r\n\p{L}\p{N}]?" + _O_UP + "*" + _O_LO + "+" + cs + "|" +
                       r"[^\r\n\p{L}\p{N}]?" + _O_UP + "+" + _O_LO + "*" + cs + "|" + ds + "| ?" +
                       r"[^\s\p{L}\p{N}]+" + ps + r"|\s*[\r\n]+|\s+(?!\S)|\s+")
                out[pat] = {"contr": cn, "digits": dn, "punct_tail": pn}
    return out


CL100K = cl100k_family()
O200K = o200k_family()
O200K_EXACT = (r"[^\r\n\p{L}\p{N}]?" + _O_UP + "*" + _O_LO + "+" + _O_CONTR + "|" + r"[^\r\n\p{L}\p{N}]?" + _O_UP +
               "+" + _O_LO + "*" + _O_CONTR + r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
assert O200K_EXACT in O200K and P_CL100K in CL100K and P_QWEN2 in CL100K and P_QWEN35 in CL100K and P_GPT2 in CL100K


def config_c_patterns():
    """the exact strings src/core/config.c compiles today (TOKS_PATTERNS), c-unescaped."""
    p = ROOT / "src/core/config.c"
    if not p.exists():
        return []
    src = p.read_text()
    m = re.search(r"TOKS_PATTERNS\[\]\s*=\s*\{(.*?)\n\};", src, re.S)
    if not m:
        return []
    out = []
    for lit in re.findall(r'\{\s*"((?:[^"\\]|\\.)*)"', m.group(1)):
        out.append(re.sub(r"\\(.)", lambda x: {"n": "\n", "r": "\r", "t": "\t"}.get(x.group(1), x.group(1)), lit))
    return out


# ---------------------------------------------------------------------------------------------------------
# features: everything a tokenizer can need beyond today's coverage. category, short description.

FEATURES = {
    # templates (K3 scanners)
    "tmpl:cl100k-params": ("template", "cl100k template, a parameter combination outside the four kernels.md strings "
                                       "(compiler table entry + tests)"),
    "tmpl:o200k": ("template", "o200k template (TOKS_TMPL_O200K): case-split letter runs, optional contraction "
                               "suffix; variants gpt-oss / llama 4 / MiniMax and tekken / nemotron"),
    "tmpl:deepseek-v3": ("template", "deepseek v3 multi-split: Split(\\p{N}{1,3}) > Split(CJK run) > Split(ascii-punct "
                                     "word | letters+marks | punct) > ByteLevel"),
    "tmpl:kimi": ("template", "kimi (tiktoken): Han runs, then o200k-style letters with Han subtracted"),
    "tmpl:metaspace": ("template", "Metaspace(split=true): replace ' ' by U+2581, prepend per scheme, split before "
                                   "U+2581 (optionally after WhitespaceSplit): the sentencepiece unigram families"),
    "tmpl:bert": ("template", "BertPreTokenizer: split on whitespace, isolate every punctuation char"),
    "tmpl:digits-gpt2": ("template", "Digits(individual) then the gpt-2 regex: every digit a hard piece boundary"),
    # model
    "model:wordpiece": ("model", "WordPiece (greedy longest match, ## continuation, [UNK] over max_input_chars)"),
    "model:unigram": ("model", "Unigram (viterbi over piece scores, unk; byte_fallback variant)"),
    "model:bpe-byte-fallback": ("model", "sentencepiece-style BPE: chars, <0xXX> byte fallback, unk + fuse_unk"),
    "model:bpe-plain": ("model", "BPE over chars without byte-level or byte fallback (unk, end_of_word_suffix)"),
    "model:wordlevel": ("model", "WordLevel (whole-piece lookup)"),
    "bpe:partial-byte-alphabet": ("model", "byte-level BPE whose vocab lacks some of the 256 byte chars: hf drops "
                                           "those bytes silently (or emits unk when unk_token is set)"),
    "bpe:partial-byte-fallback": ("model", "byte_fallback with some <0xXX> tokens missing: those bytes become unk"),
    "bpe:end_of_word_suffix": ("model", "BPE end_of_word_suffix (</w>)"),
    "bpe:dropout": ("model", "BPE dropout (non-deterministic: refuse)"),
    "bpe:byte-fallback+bytelevel": ("model", "byte_fallback set on a byte-level BPE"),
    # normalizers
    "norm:BertNormalizer": ("normalizer", "BertNormalizer (clean text, CJK spacing, lowercase / strip accents)"),
    "norm:Precompiled": ("normalizer", "Precompiled charsmap (sentencepiece nmt_nfkc tables)"),
    "norm:Replace": ("normalizer", "Replace with a literal string"),
    "norm:Replace(regex)": ("normalizer", "Replace with a regex pattern"),
    "norm:Prepend": ("normalizer", "Prepend (U+2581)"),
    "norm:NFKC": ("normalizer", "NFKC"), "norm:NFKD": ("normalizer", "NFKD"), "norm:NFD": ("normalizer", "NFD"),
    "norm:Lowercase": ("normalizer", "Lowercase"), "norm:Strip": ("normalizer", "Strip"),
    "norm:StripAccents": ("normalizer", "StripAccents"), "norm:Nmt": ("normalizer", "Nmt"),
    "norm:ByteLevel": ("normalizer", "ByteLevel normalizer"),
    # pre-tokenizer pieces outside the template
    "pretok:ByteLevel.add_prefix_space": ("pre-tokenizer", "ByteLevel add_prefix_space=true"),
    "pretok:Metaspace(split=false)": ("pre-tokenizer", "Metaspace(split=false): replace ' ' by U+2581 and prepend it per "
                                      "scheme (first / always / never), no split, so each segment is one BPE piece and "
                                      "the substitution can live in the tables (mistral v0.1-0.3, yi coder)"),
    # added tokens
    "added:lstrip": ("added tokens", "lstrip (the match swallows the whitespace before it)"),
    "added:rstrip": ("added tokens", "rstrip (the match swallows the whitespace after it)"),
    "added:single_word": ("added tokens", "single_word (match only at word boundaries)"),
    # post-processors
    "pp:RobertaProcessing": ("post-processor", "RobertaProcessing (<s> $A </s>; maps onto prefix/suffix ids)"),
    "pp:BertProcessing": ("post-processor", "BertProcessing ([CLS] $A [SEP]; maps onto prefix/suffix ids)"),
    # decoders
    "dec:WordPiece": ("decoder", "WordPiece decoder (## joins, cleanup)"),
    "dec:Metaspace": ("decoder", "Metaspace decoder"),
    "dec:Replace": ("decoder", "Replace decoder (U+2581 -> ' ')"),
    "dec:ByteFallback": ("decoder", "ByteFallback decoder (<0xXX> -> bytes)"),
    "dec:Fuse": ("decoder", "Fuse decoder"),
    "dec:Strip": ("decoder", "Strip decoder (leading space)"),
    "dec:BPEDecoder": ("decoder", "BPEDecoder (</w> suffix)"),
    "dec:CTC": ("decoder", "CTC decoder"),
    # tokenizer.json fields hf applies inside encode()
    "file:truncation": ("encode()", "truncation set in tokenizer.json: hf encode() truncates to max_length"),
    "file:padding-fixed": ("encode()", "padding Fixed(n) (or pad_to_multiple_of) in tokenizer.json: hf encode() pads "
                                       "every output (BatchLongest alone is a no-op for one sequence)"),
    # compiler spellings (no new kernel: a semantically equal spelling the compiler must accept)
    "compile:split-removed-invert": ("compiler", "Split(behavior=Removed, invert=true) on a pattern whose matches "
                                                 "tile the text: the same pieces as Isolated"),
    "compile:noop-split": ("compiler", "Split(' ') after a normalizer that replaced every ' ' (gemma): no split"),
    "compile:inert-padding": ("compiler", "padding BatchLongest (no pad_to_multiple_of): a no-op for one sequence "
                                          "(hf encode_batch still pads a batch to its longest); config.c refuses any "
                                          "padding"),
    "compile:inert-dropout": ("compiler", "BPE dropout 0.0: never skips a merge; config.c refuses any non-null "
                                          "dropout"),
    "compile:non-alphabet-vocab": ("compiler", "byte-level vocab strings outside the byte-level alphabet: no encoding "
                                               "reaches them (decode uses hf's fallback); config.c refuses the file"),
    # inputs
    "input:tiktoken": ("input", "tiktoken ranks + pattern from the repo's tokenization_*.py (SPEC 0.3 input)"),
    "input:tekken": ("input", "mistral tekken.json (SPEC 0.3 input)"),
    "input:slow-wordpiece": ("input", "only vocab.txt (transformers builds the WordPiece tokenizer)"),
    "input:slow-bpe": ("input", "only vocab.json + merges.txt (transformers builds the byte-level BPE)"),
    "input:slow-sentencepiece": ("input", "only a sentencepiece .model (transformers converts it)"),
}
SLOW_INPUT = {"wordpiece-vocab-only": "input:slow-wordpiece", "bpe-vocab-merges-only": "input:slow-bpe",
              "sentencepiece-only": "input:slow-sentencepiece"}
DEC_COVERED = {"ByteLevel"}
PP_COVERED = {"TemplateProcessing", "ByteLevel"}
NORM_COVERED = {"NFC"}


def h8(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


def pat(e):
    p = e.get("pattern") or {}
    if "Regex" in p:
        return p["Regex"], True
    return p.get("String"), False


def classify_regex(rx):
    """-> (template, variant, status) for one regex used as the whole split."""
    if rx in KERNELS_MD:
        return "cl100k", KERNELS_MD[rx], "covered"
    if rx in CL100K:
        p = CL100K[rx]
        v = (f"contr={p['contr']} letters={p['letters']}{'+M' if p['marks'] else ''} digits={p['digits']} "
             f"punct_nl={int(p['punct_nl'])} ws_nl={int(p['ws_nl'])}")
        return "cl100k", v, "params"
    if rx in O200K:
        p = O200K[rx]
        v = "o200k (gpt-oss, llama 4)" if rx == O200K_EXACT else \
            f"o200k {p['contr']}, digits {p['digits']}, punct tail {p['punct_tail']}"
        return "o200k", v, "new"
    if rx == P_KIMI:
        return "kimi", "kimi k2 / k2.5 / k3 / linear", "new"
    return None, None, None


def classify_chain(pre, norms):
    """the pre-tokenizer chain of one tokenizer -> template, variant, status, features."""
    types = [e["type"] for e in pre]
    f = set()
    if not pre:
        return "none", "no pre-tokenizer (one piece per segment)", "covered", f
    if types == ["ByteLevel"]:
        if pre[0].get("add_prefix_space"):
            f.add("pretok:ByteLevel.add_prefix_space")
        if pre[0].get("use_regex", True):
            return "cl100k", "gpt-2 (ByteLevel use_regex)", "covered", f
        return "none", "ByteLevel without regex", "covered", f
    bl_last = types[-1] == "ByteLevel" and not pre[-1].get("use_regex", True)
    if bl_last and pre[-1].get("add_prefix_space"):
        f.add("pretok:ByteLevel.add_prefix_space")
    body = pre[:-1] if bl_last else pre
    btypes = [e["type"] for e in body]
    if btypes == ["Split"]:
        rx, is_re = pat(body[0])
        beh, inv = body[0].get("behavior"), body[0].get("invert")
        t, v, st = classify_regex(rx) if is_re else (None, None, None)
        if t:
            if (beh, inv) == ("Removed", True):
                f.add("compile:split-removed-invert")
                if st == "covered":
                    st = "compile"
            elif (beh, inv) != ("Isolated", False):
                return "other", f"Split {beh} invert={inv}", "new", f
            if t == "cl100k" and st == "params":
                f.add("tmpl:cl100k-params")
            if t == "o200k":
                f.add("tmpl:o200k")
            if not bl_last:
                return "other", f"{v} without ByteLevel", "new", f
            return t, v, st, f
        if rx == " " and beh == "MergedWithPrevious" and not inv and norms and norms[-1].get("type") == "Replace" \
                and pat(norms[-1])[0] == " " and norms[-1].get("content") != " " and not bl_last:
            f.add("compile:noop-split")
            return "none", "Split(' ') after Replace(' ' -> U+2581): no split", "compile", f
    if bl_last and btypes == ["Split", "Split", "Split"] and [pat(e)[0] for e in body] == [DS_DIGITS, DS_CJK, DS_MAIN] \
            and all((e.get("behavior"), e.get("invert")) == ("Isolated", False) for e in body):
        f.add("tmpl:deepseek-v3")
        return "deepseek-v3", "deepseek v3 / r1 / v3.1 / v3.2 / v4", "new", f
    if types == ["Digits", "ByteLevel"] and pre[0].get("individual_digits") and pre[1].get("use_regex", True):
        f.add("tmpl:digits-gpt2")
        return "digits-gpt2", "Digits(individual) > ByteLevel(gpt-2 regex)", "new", f
    if types in (["Metaspace"], ["WhitespaceSplit", "Metaspace"]):
        m = pre[-1]
        v = ("WhitespaceSplit > " if len(types) == 2 else "") + \
            f"Metaspace(prepend={m.get('prepend_scheme')}, split={str(m.get('split')).lower()})"
        if types == ["Metaspace"] and m.get("split") is False:
            # no split: one piece per segment (the sentencepiece-BPE spelling of Replace + Prepend), no template
            f.add("pretok:Metaspace(split=false)")
            return "none", v + ": no split", "compile", f
        f.add("tmpl:metaspace")
        return "metaspace", v, "new", f
    if types == ["BertPreTokenizer"]:
        f.add("tmpl:bert")
        return "bert", "BertPreTokenizer", "new", f
    return "other", None, "new", f


def chain_text(pre):
    """a short, exact-enough rendering of a chain for tables (patterns by id; exact strings in the appendix)."""
    out = []
    for e in pre:
        t = e["type"]
        if t == "Split":
            rx, is_re = pat(e)
            s = rx if len(rx) <= 24 else rx[:21] + "..."
            out.append(f"Split[{e.get('behavior')}{',invert' if e.get('invert') else ''}]({'re' if is_re else 'str'}:"
                       f"{json.dumps(s, ensure_ascii=False)})")
        elif t == "ByteLevel":
            out.append(f"ByteLevel(regex={int(bool(e.get('use_regex', True)))},prefix_space="
                       f"{int(bool(e.get('add_prefix_space')))})")
        elif t == "Metaspace":
            out.append(f"Metaspace({e.get('prepend_scheme')},split={int(bool(e.get('split')))})")
        elif t == "Digits":
            out.append(f"Digits(individual={int(bool(e.get('individual_digits')))})")
        elif t == "Punctuation":
            out.append(f"Punctuation({e.get('behavior')})")
        else:
            out.append(t)
    return " > ".join(out) or "(none)"


# ---------------------------------------------------------------------------------------------------------
# per tokenizer: features (what it needs beyond today)

def synth_chain(rx):
    return [{"type": "Split", "pattern": {"Regex": rx}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}] if rx else None


def chain_key(chain):
    return json.dumps(chain, sort_keys=True, ensure_ascii=False) if chain is not None else None


def tokenizer_chain(sig, kind):
    if kind == "tiktoken":
        return synth_chain((sig.get("tiktoken") or {}).get("pattern"))
    if kind == "tekken":
        return synth_chain((sig.get("tekken") or {}).get("pattern"))
    return sig.get("pre_tokenizer")


def tokenizer_features(sig, kind, pid=None):
    """-> dict(template, variant, status, features:set, chain:list|None). pid names an unnamed template."""
    other = f"tmpl:{pid}" if pid else "tmpl:other"
    if kind == "tiktoken":
        rx = (sig.get("tiktoken") or {}).get("pattern")
        t, v, st = classify_regex(rx) if rx else (None, None, None)
        f = {"input:tiktoken"}
        if t == "cl100k" and st == "params":
            f.add("tmpl:cl100k-params")
        elif t and t != "cl100k":
            f.add("tmpl:" + t)
        elif not t:
            f.add(other)
        return {"template": t or "other", "variant": v, "status": st or "new", "features": f,
                "chain": synth_chain(rx)}
    if kind == "tekken":
        rx = (sig.get("tekken") or {}).get("pattern")
        t, v, st = classify_regex(rx) if rx else (None, None, None)
        f = {"input:tekken"} | ({"tmpl:" + t} if t and t != "cl100k" else set()) | (set() if t else {other})
        return {"template": t or "other", "variant": v, "status": st or "new", "features": f,
                "chain": synth_chain(rx)}
    pre, norms, m = sig["pre_tokenizer"], sig["normalizer"], sig["model"]
    t, v, st, f = classify_chain(pre, norms)
    algo = sig["algorithm"]
    if t == "other":
        f.add(other)
    if algo != "bpe-bytelevel":
        f.add("model:" + algo)
    if algo == "bpe-bytelevel":
        if m.get("byte_alphabet", 256) < 256:
            f.add("bpe:partial-byte-alphabet")
        if m.get("byte_fallback"):
            f.add("bpe:byte-fallback+bytelevel")
        if m.get("non_alphabet_tokens"):
            f.add("compile:non-alphabet-vocab")
    if m.get("type") == "BPE":
        if m.get("end_of_word_suffix"):
            f.add("bpe:end_of_word_suffix")
        if m.get("dropout"):
            f.add("bpe:dropout")
        elif m.get("dropout") is not None:
            f.add("compile:inert-dropout")
        if algo == "bpe-byte-fallback" and m.get("byte_tokens", 256) < 256:
            f.add("bpe:partial-byte-fallback")
    for n in norms:
        tn = n["type"]
        if tn in NORM_COVERED:
            continue
        if tn == "Replace":
            f.add("norm:Replace(regex)" if "Regex" in (n.get("pattern") or {}) else "norm:Replace")
        else:
            f.add("norm:" + tn)
    for d in sig["decoder"]:
        if d["type"] not in DEC_COVERED:
            f.add("dec:" + d["type"])
    for p in sig["post_processor"]:
        if p["type"] not in PP_COVERED:
            f.add("pp:" + p["type"])
    a = sig["added_tokens"]
    for k in ("lstrip", "rstrip", "single_word"):
        if a.get(k):
            f.add("added:" + k)
    if sig.get("truncation"):
        f.add("file:truncation")
    pad = sig.get("padding") or {}
    if (isinstance(pad.get("strategy"), dict) and "Fixed" in pad["strategy"]) or pad.get("pad_to_multiple_of"):
        f.add("file:padding-fixed")
    elif pad:
        f.add("compile:inert-padding")
    for x in f:
        if x not in FEATURES and not x.startswith("tmpl:"):
            FEATURES[x] = (x.split(":")[0], x)
    return {"template": t, "variant": v, "status": st, "features": f, "chain": pre}


# ---------------------------------------------------------------------------------------------------------
# the census table: models -> tokenizers -> features

def build(st):
    res, an, models = st["resolved"], st["analyzed"], st["models"]
    toks = {}       # tok id -> record
    rows = []
    cfgc = config_c_patterns()
    for rid, m in models.items():
        if not m.get("slices"):
            continue
        r = res.get(rid) or {}
        kind = r.get("kind")
        row = {"id": rid, "slices": sorted(m["slices"]), "downloads": m["downloads"],
               "pipeline_tag": m.get("pipeline_tag"), "kind": kind, "status": r.get("status"),
               "via": r.get("via") or [], "source": r.get("mirror") or r.get("repo"),
               "testing": any(x in rid for x in TESTING_ORGS)}
        sha, extra_f, label = None, set(), kind
        if kind in ("tokenizer.json", "tiktoken", "tekken") and r.get("status") in ("ok", "gated-mirrored"):
            sha = r.get("sha256")
        elif kind in SLOW_INPUT:
            cv = r.get("converted") or {}
            sha = cv.get("sha256")
            extra_f.add(SLOW_INPUT[kind])
            label = f"{kind} (converted: transformers {cv.get('transformers')}, {cv.get('class')})" if sha else kind
            if not sha:
                row["miss"] = (f"{kind}, gated: not converted" if r.get("gated") else
                               f"{kind}; transformers builds no fast tokenizer from it "
                               f"({r.get('tokenizer_class') or '?'}: {cv.get('error', '?')[:60]})")
        elif kind == "remote-code":
            row["miss"] = "remote code only (tokenization_*.py, no tokenizer.json / tiktoken file)"
        elif r.get("status", "").startswith("gated"):
            row["miss"] = "gated, no byte-identical ungated copy found"
        else:
            row["miss"] = {"no-tokenizer": "no tokenizer files and no base model / name match",
                           "no-tokenizer-gguf": "gguf only, no base model / name match"}.get(
                               r.get("status"), r.get("status") or "unresolved")
        rec = an.get(sha) if sha else None
        if sha and not rec:
            row["miss"] = "file not analyzed"
        if rec:
            if rec.get("error") or ("sig" not in rec):
                row["miss"] = f"unreadable: {rec.get('error', '?')[:80]}"
            else:
                tid = rec["hash"][:12]
                row["tok"] = tid
                if tid not in toks:
                    fk = "tokenizer.json" if kind in SLOW_INPUT else kind
                    toks[tid] = {"id": tid, "hash": rec["hash"], "kind": label, "file_sha256": rec["file_sha256"],
                                 "hf_loads": rec.get("hf_loads"), "algorithm": rec["sig"].get("algorithm"),
                                 "fkind": fk, "extra": sorted(extra_f), "chain": tokenizer_chain(rec["sig"], fk),
                                 "sig": rec["sig"], "raw_pre_tokenizer": rec.get("raw_pre_tokenizer"),
                                 "parts": rec.get("parts"), "probe": rec.get("probe"),
                                 "models": [], "dl_b": 0, "dl_all": 0}
                toks[tid]["models"].append(rid)
                toks[tid]["dl_all"] += m["downloads"]
                if set(m["slices"]) & {"textgen", "embed"}:
                    toks[tid]["dl_b"] += m["downloads"]
        rows.append(row)
    # pattern ids: one per distinct exact chain, numbered by (b) downloads
    chains = {}
    for t in toks.values():
        if t["chain"] is None:
            continue
        chains.setdefault(chain_key(t["chain"]), {"chain": t["chain"], "toks": []})["toks"].append(t["id"])
    for t in toks.values():
        t["pid"] = None
    for i, (k, c) in enumerate(sorted(chains.items(), key=lambda kc: (-sum(toks[x]["dl_b"] for x in kc[1]["toks"]),
                                                                       -sum(toks[x]["dl_all"] for x in kc[1]["toks"]),
                                                                       kc[0]))):
        c["pid"] = f"P{i + 1:02d}"
        for x in c["toks"]:
            toks[x]["pid"] = c["pid"]
    for t in toks.values():
        tf = tokenizer_features(t["sig"], t["fkind"], t["pid"])
        t.update(template=tf["template"], variant=tf["variant"], tmpl_status=tf["status"],
                 features=sorted(tf["features"] | set(t["extra"])))
        if tf["template"] == "other" and t["pid"]:
            FEATURES[f"tmpl:{t['pid']}"] = ("template", f"a template for pattern {t['pid']} "
                                                        f"({chain_text(t['chain'])})")
    return rows, toks, chains, cfgc


def weights(rows, which):
    """downloads per model for a population: b = textgen + embed (SPEC 1.1(b)); b-real = b without testing repos."""
    w = {}
    for r in rows:
        s = set(r["slices"])
        if which == "b" and s & {"textgen", "embed"}:
            w[r["id"]] = r["downloads"]
        elif which == "b-real" and s & {"textgen", "embed"} and not r["testing"]:
            w[r["id"]] = r["downloads"]
        elif which in s:
            w[r["id"]] = r["downloads"]
    return w


LIGHT = ("dec:", "pp:", "compile:")

WORDPIECE = {"tmpl:bert", "model:wordpiece", "norm:BertNormalizer", "dec:WordPiece", "pp:BertProcessing"}
UNIGRAM = {"tmpl:metaspace", "model:unigram", "norm:Precompiled", "dec:Metaspace", "norm:Replace(regex)"}
# the no-op Split (gemma 2-4) and Metaspace(split=false) (mistral v0.1-0.3) are how sentencepiece-style BPE files
# spell their Replace / Prepend: they belong here, not to the Unigram stack or to the compiler-spelling package
SPM_BPE = {"model:bpe-byte-fallback", "norm:Replace", "norm:Prepend", "dec:Replace", "dec:ByteFallback", "dec:Fuse",
           "dec:Strip", "bpe:partial-byte-fallback", "compile:noop-split", "pretok:Metaspace(split=false)"}
PACKAGES = [
    ("WordPiece stack (BERT family)", lambda f: f in WORDPIECE),
    ("encode() options set in tokenizer.json (truncation, padding)", lambda f: f.startswith("file:")),
    ("Unigram stack (sentencepiece unigram: Metaspace split, Precompiled charsmap)", lambda f: f in UNIGRAM),
    ("added-token options (lstrip, rstrip, single_word)", lambda f: f.startswith("added:")),
    ("post-processor mappings (Roberta)", lambda f: f.startswith("pp:")),
    ("o200k template", lambda f: f == "tmpl:o200k"),
    ("compiler spellings (equal forms of covered templates)", lambda f: f.startswith("compile:") and f not in SPM_BPE),
    ("deepseek-v3 template", lambda f: f == "tmpl:deepseek-v3"),
    ("byte-level BPE with a partial byte alphabet", lambda f: f in ("bpe:partial-byte-alphabet",
                                                                    "bpe:byte-fallback+bytelevel")),
    ("sentencepiece-style BPE stack (llama 2, mistral v0.1-0.3, gemma 2-4, yi)", lambda f: f in SPM_BPE),
    ("slow-file inputs (vocab.json + merges.txt, vocab.txt, sentencepiece .model)",
     lambda f: f.startswith("input:slow")),
    ("kimi template + tiktoken input", lambda f: f in ("tmpl:kimi", "input:tiktoken", "input:tekken")),
    ("digits-gpt2 template (SmolLM)", lambda f: f == "tmpl:digits-gpt2"),
    ("cl100k template parameters beyond the four strings", lambda f: f == "tmpl:cl100k-params"),
    ("ByteLevel add_prefix_space", lambda f: f.startswith("pretok:")),
    ("other normalizers (NFKC, NFKD, StripAccents, Lowercase, Strip)", lambda f: f.startswith("norm:")),
    ("other algorithms (plain BPE with </w>, WordLevel)", lambda f: f.startswith(("model:", "bpe:", "dec:"))),
    ("one-off patterns (each its own template, or the generic engine)", lambda f: f.startswith("tmpl:")),
]
PID_EXAMPLE = {}     # pattern id -> its most downloaded model (set by report())


def package_of(f):
    if f.startswith("tmpl:P"):
        ex = PID_EXAMPLE.get(f[5:])
        return f"pattern {f[5:]} template" + (f" ({ex})" if ex else "")
    if f.startswith("code:"):
        return "load path: " + f[5:]
    for name, pred in PACKAGES:
        if pred(f):
            return name
    return "other"


def package_order(rows, toks, w, start=frozenset(), start_name=None):
    """the headline: whole work packages. each step adds the packages one tokenizer group still needs (so a
    package that only pays off together with another arrives with it), most downloads per unit of work first.
    start (the critical targets' features) is step 0, whatever it unlocks."""
    need = {}
    for r in rows:
        if r["id"] in w and r.get("tok"):
            k = frozenset(toks[r["tok"]]["features"])
            need[k] = need.get(k, 0) + w[r["id"]]
    allf = set().union(*need) if need else set()
    pk = collections.defaultdict(set)
    for f in allf:
        pk[package_of(f)].add(f)
    covered, out, cur = set(), [], sum(v for k, v in need.items() if not k)
    if start:
        g = sum(v for k, v in need.items() if k and k <= start)
        covered |= start
        cur += g
        names = sorted({package_of(f) for f in start}, key=lambda n: ([x[0] for x in PACKAGES] + [n]).index(n))
        out.append({"package": "critical targets: " + ", ".join(start_name or []), "packages": names,
                    "features": sorted(start), "gain": g, "cum": cur, "critical": list(start_name or [])})
    while True:
        cands = {frozenset(package_of(f) for f in k - covered) for k in need if k - covered}
        if not cands:
            break
        best = None
        for c in sorted(cands, key=lambda c: (len(c), sorted(c))):
            fs = set().union(*(pk[n] for n in c)) - covered
            g = sum(v for k, v in need.items() if k and not k <= covered and k <= covered | fs)
            key = (g / cost(fs), g)
            if best is None or key > best[0]:
                best = (key, c, fs, g)
        _, c, fs, g = best
        covered |= fs
        cur += g
        names = sorted(c, key=lambda n: ([x[0] for x in PACKAGES] + [n]).index(n))
        out.append({"package": " + ".join(names), "packages": names, "features": sorted(fs), "gain": g, "cum": cur})
    return out


def cost(features):
    """units of work: a template, algorithm, normalizer, encode option, input format or model flag is 1; a decoder,
    a post-processor mapped onto prefix / suffix ids, or a spelling the compiler must accept is 1/4."""
    return sum(0.25 if f.startswith(LIGHT) else 1.0 for f in features)


def critical_features(st, by_id, toks):
    """The critical targets (names, in census order) and every feature their own tokenizers need: step 0 of
    every build order, whatever it unlocks."""
    names, feats = [], set()
    for c in st.get("critical") or []:
        names.append(c["target"])
        tid = (by_id.get(c["repo"]) or {}).get("tok")
        if tid:
            feats |= set(toks[tid]["features"])
    return names, frozenset(feats)


def packages_alone(rows, toks, slices):
    """what each package unlocks on its own from today's coverage (no step 0, nothing else added), per slice: the
    attribution the orders cannot show (a package whose tokenizers also need another one's features unlocks
    little alone, which is why the orders add them together)."""
    pkf = collections.defaultdict(set)
    for t in toks.values():
        for f in t["features"]:
            pkf[package_of(f)].add(f)
    out = []
    for name, fs in pkf.items():
        row = {"package": name, "features": sorted(fs)}
        for sname, sw in slices:
            tw = sum(sw.values()) or 1
            row[sname] = sum(sw[r["id"]] for r in rows if r["id"] in sw and r.get("tok") and toks[r["tok"]]["features"]
                             and set(toks[r["tok"]]["features"]) <= fs) / tw
        out.append(row)
    return sorted(out, key=lambda x: (-x["textgen"], -x["b"], x["package"]))


def tiktoken_check(st, families, toks):
    """the census's pattern string for each openai encoding (the family's hub tokenizer.json) against tiktoken's own
    pat_str (coverage.py tiktoken). o200k_harmony is gpt-oss's encoding."""
    tk = st.get("tiktoken") or {}
    pats = tk.get("patterns") or {}
    if not pats:
        return {}
    fam_enc = {f["tiktoken"]: f for f in families if f.get("tiktoken")}
    fam_enc.setdefault("o200k_harmony", next((f for f in families if f["repo"] == "openai/gpt-oss-20b"), None))
    out = []
    for enc, p in pats.items():
        f = fam_enc.get(enc)
        if not f:
            continue                        # p50k_base: no family in the census stands for it
        t = toks.get(f.get("tok")) if f.get("tok") else None
        rx = None
        for e in (t or {}).get("chain") or []:
            if e.get("type") == "Split" and "Regex" in (e.get("pattern") or {}):
                rx = e["pattern"]["Regex"]
            elif e.get("type") == "ByteLevel" and e.get("use_regex", True) and len(t["chain"]) == 1:
                rx = P_GPT2
        out.append({"encoding": enc, "repo": f["repo"], "pid": t.get("pid") if t else None,
                    "census": rx, "tiktoken": p, "equal": rx == p if rx else None})
    return {"version": tk.get("tiktoken"), "checks": out}


def package_stats(pkgs, rows, toks, w, slices):
    """per package of an order weighted by w: the models it unlocks (in w), its share and the cumulative share of
    w, its cost, the cumulative share of distinct tokenizers (in w, misses counted), and the cumulative share of
    every slice."""
    tot = sum(w.values()) or 1
    wtoks = {r["tok"] for r in rows if r["id"] in w and r.get("tok")}
    nmiss = sum(1 for r in rows if r["id"] in w and not r.get("tok"))
    cov = set()
    for p in pkgs:
        before = set(cov)
        cov |= set(p["features"])
        p["models"] = [r["id"] for r in rows if r["id"] in w and r.get("tok")
                       and set(toks[r["tok"]]["features"]) <= cov and not set(toks[r["tok"]]["features"]) <= before]
        p["share"] = p["gain"] / tot
        p["cum_share"] = p["cum"] / tot
        p["cost"] = cost(p["features"])
        p["cum_count"] = sum(1 for t in wtoks if set(toks[t]["features"]) <= cov) / ((len(wtoks) + nmiss) or 1)
        for name, sw in slices:
            tw = sum(sw.values()) or 1
            p["cum_" + name] = sum(sw[r["id"]] for r in rows if r["id"] in sw and r.get("tok")
                                   and set(toks[r["tok"]]["features"]) <= cov) / tw
    return pkgs


def greedy(rows, toks, w, start=frozenset(), start_name=None):
    """the ordered work list. a model counts once its tokenizer needs nothing missing. each step adds the missing
    feature set of one tokenizer group: the set that unlocks the most downloads per unit of work (cost(), below),
    ties to the larger gain. the classic greedy for budgeted coverage: after any number of steps, few other
    choices of the same cost cover more. start (the critical targets' features) is step 0, whatever it
    unlocks."""
    need = {}
    for r in rows:
        if r["id"] in w and r.get("tok"):
            need.setdefault(frozenset(toks[r["tok"]]["features"]), 0)
            need[frozenset(toks[r["tok"]]["features"])] += w[r["id"]]
    total = sum(w.values())
    covered = set()
    base = sum(v for k, v in need.items() if not k)
    steps = []
    cur = base
    if start:
        g = sum(v for k, v in need.items() if k and k <= start)
        covered |= start
        cur += g
        steps.append({"add": sorted(start), "gain": g, "cum": cur, "cost": cost(start),
                      "critical": list(start_name or [])})
    def gain(c):
        return sum(v for k, v in need.items() if (k - covered) and (k - covered) <= c)
    while True:
        cands = {frozenset(k - covered) for k in need if k - covered}
        if not cands:
            break
        best, bkey = None, None
        for c in sorted(cands, key=lambda c: (len(c), sorted(c))):
            g = gain(c)
            key = (g / cost(c), g)
            if bkey is None or key > bkey:
                best, bkey = c, key
        covered |= best
        cur += bkey[1]
        steps.append({"add": sorted(best), "gain": bkey[1], "cum": cur, "cost": cost(best)})
    return {"total": total, "base": base, "steps": steps}


def share_after(rows, toks, w, steps):
    """cumulative share of population w after each step of a given order."""
    covered = set()
    out = []
    tot = sum(w.values()) or 1
    seq = [set()] + [set(s["add"]) for s in steps]
    for add in seq:
        covered |= add
        c = sum(w[r["id"]] for r in rows if r["id"] in w and r.get("tok")
                and set(toks[r["tok"]]["features"]) <= covered)
        out.append(c / tot)
    return out


def step_name(add):
    tm = [a[5:] for a in add if a.startswith("tmpl:")]
    md = [a[6:] for a in add if a.startswith("model:")]
    parts = [f"{', '.join(tm)} template" if tm else None, " + ".join(md) if md else None]
    parts = [p for p in parts if p]
    if parts:
        return " + ".join(parts)
    rank = ("input:", "file:", "bpe:", "norm:", "added:", "pp:", "compile:", "pretok:", "dec:")
    for r in rank:
        hit = [a for a in add if a.startswith(r)]
        if hit:
            return ", ".join(hit)
    return add[0]


def fmt_pct(x):
    return f"{100 * x:.2f}%" if x < 0.9995 or x >= 1 else f"{100 * x:.2f}%"


def fmt_n(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(n)


def examples(rows_by_id, ids, k=3):
    ids = sorted(ids, key=lambda i: -rows_by_id[i]["downloads"])
    return ", ".join(f"`{i}`" for i in ids[:k])


# ---------------------------------------------------------------------------------------------------------

def dump_lines(data):
    """json with one top-level key per block and one list item / dict entry per line: small, reviewable diffs."""
    out = ["{"]
    keys = list(data)
    for i, k in enumerate(keys):
        v = data[k]
        tail = "," if i + 1 < len(keys) else ""
        if isinstance(v, list) and v:
            out.append(f" {json.dumps(k)}: [")
            out += [f"  {json.dumps(x, ensure_ascii=False, separators=(',', ':'))}" + ("," if j + 1 < len(v) else "")
                    for j, x in enumerate(v)]
            out.append(" ]" + tail)
        elif isinstance(v, dict) and v and k in ("tokenizers", "mirrors"):
            out.append(f" {json.dumps(k)}: {{")
            items = list(v.items())
            out += [f"  {json.dumps(a)}: {json.dumps(b, ensure_ascii=False, separators=(',', ':'))}" +
                    ("," if j + 1 < len(items) else "") for j, (a, b) in enumerate(items)]
            out.append(" }" + tail)
        else:
            out.append(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False, separators=(',', ':'))}{tail}")
    out.append("}")
    text = "\n".join(out) + "\n"
    json.loads(text)
    return text


SPEC15_KINDS = {"wordpiece-vocab-only", "bpe-vocab-merges-only", "sentencepiece-only", "remote-code"}


def code_today(st, rows, toks, chains):
    """today = what toks's load path accepts (tools/census/probe.c on the census's files; coverage.py probe).
    Sets each model row's code verdict (loads / miss / out: an input SPEC 1.5 keeps out) and the reason that
    blocks it, gives each tiktoken tokenizer the verdict of its models' directories (split by verdict: the wrapper
    and tokenizer_config.json differ between repos sharing one ranks file), and rebases every tokenizer's features
    on the code: a feature some loading tokenizer uses is done; a refused tokenizer keeps its features nobody loads
    yet, or, when every one of them is done, the load path's own refusal as a `code:` feature."""
    res = st["resolved"]
    for r in rows:
        rr = res.get(r["id"]) or {}
        t = toks.get(r.get("tok")) if r.get("tok") else None
        if r["kind"] in SPEC15_KINDS:
            pr = (t or {}).get("probe")
            conv = ("its converted tokenizer.json loads" if pr and pr["stage"] == 0 else
                    f"its converted tokenizer.json is refused: {pr['what']}" if pr else "no converted tokenizer.json")
            r["code"], r["block"] = "out", (f"{r['kind']} (SPEC 1.5: input out of scope; {conv})"
                                            if r["kind"] != "remote-code" else
                                            "remote code only (SPEC 1.5: input out of scope)")
        elif r["kind"] == "tiktoken" and t:
            pr = rr.get("probe")
            r["code"] = "loads" if pr and pr["stage"] == 0 else "miss"
            r["block"] = None if r["code"] == "loads" else "tiktoken model: " + (pr["what"] if pr else "not probed")
            r["probe"] = pr
            r["generic"] = bool(pr and pr["stage"] == 0 and pr["what"] == "ok generic")
        elif t:
            pr = t.get("probe")
            r["code"] = "loads" if pr and pr["stage"] == 0 else "miss"
            r["block"] = None if r["code"] == "loads" else (pr["what"] if pr else "not probed")
            r["generic"] = bool(pr and pr["stage"] == 0 and pr["what"] == "ok generic")
        else:
            r["code"], r["block"] = "miss", "no readable tokenizer: " + str(r.get("miss"))
    # one tokenizer record per verdict: tiktoken repos sharing a ranks file differ in wrapper and config, and a
    # slow input whose conversion hashes like a shipped tokenizer.json is still an input SPEC 1.5 keeps out
    def move(tid, rs, nid, **over):
        t = toks[tid]
        if nid not in toks:
            toks[nid] = dict(t, id=nid, models=[], dl_all=0, dl_b=0, **over)
            for c in chains.values():
                if tid in c["toks"]:
                    c["toks"].append(nid)
        for r in rs:
            t["models"].remove(r["id"])
            t["dl_all"] -= r["downloads"]
            toks[nid]["models"].append(r["id"])
            toks[nid]["dl_all"] += r["downloads"]
            if set(r["slices"]) & {"textgen", "embed"}:
                t["dl_b"] -= r["downloads"]
                toks[nid]["dl_b"] += r["downloads"]
            r["tok"] = nid

    for tid in list(toks):
        t = toks[tid]
        rs = [r for r in rows if r.get("tok") == tid]
        if t["fkind"] == "tiktoken":
            groups = collections.defaultdict(list)
            for r in rs:
                groups[json.dumps(r.get("probe"), sort_keys=True)].append(r)
            order = sorted(groups, key=lambda k: (json.loads(k) or {}).get("stage", 9))
            t["probe"] = json.loads(order[0]) if order else None
            for i, k in enumerate(order[1:], 1):
                move(tid, groups[k], f"{tid}~{i}", probe=json.loads(k))
        for kind, f in SLOW_INPUT.items():
            slow = [r for r in rs if r["kind"] == kind]
            if slow and f not in t["features"]:
                move(tid, slow, f"{tid}~{kind}", features=sorted(set(t["features"]) | {f}), extra=[f], kind=kind)
    done = set()
    for t in toks.values():
        t["design_features"] = list(t["features"])
        if (t.get("probe") or {}).get("stage") == 0 and not any(f.startswith("input:slow") for f in t["features"]):
            done |= set(t["features"])
    for t in toks.values():
        pr = t.get("probe") or {}
        slow = [f for f in t["features"] if f.startswith("input:slow")]
        if pr.get("stage") == 0 and not slow:
            t["features"] = []
            continue
        rest = set(t["features"]) - done
        if not rest:
            what = pr.get("what") or "not probed"
            if t["fkind"] == "tiktoken":
                what = "tiktoken model: " + what
            rest = {"code:" + what}
            FEATURES["code:" + what] = ("load path", f"toks's load path refuses: {what}")
        t["features"] = sorted(rest)
    return done


def code_shares(rows, slices):
    """per slice: the share that loads, the share SPEC 1.5's out-of-scope inputs take, the share that loads
    once those leave the denominator, and the share that loads on compiled fast paths (SPEC 1.4's gate: not
    the generic engine)."""
    out = {}
    for name, w in slices:
        tot = sum(w.values()) or 1
        loads = sum(w[r["id"]] for r in rows if r["id"] in w and r.get("code") == "loads")
        gen = sum(w[r["id"]] for r in rows if r["id"] in w and r.get("code") == "loads" and r.get("generic"))
        oos = sum(w[r["id"]] for r in rows if r["id"] in w and r.get("code") == "out")
        out[name] = {"downloads": tot, "loads": loads / tot, "out_of_scope": oos / tot,
                     "loads_without_out_of_scope": loads / ((tot - oos) or 1), "generic": gen / tot,
                     "fast": (loads - gen) / tot, "fast_without_out_of_scope": (loads - gen) / ((tot - oos) or 1)}
    return out


def code_misses(rows, wb, tot_b):
    """every (b) model that does not load today, grouped by what blocks it, most downloads first."""
    g = collections.defaultdict(list)
    for r in rows:
        if r["id"] in wb and r.get("code") != "loads":
            g[(r["code"], r["block"])].append(r)
    out = []
    for (code, block), rs in sorted(g.items(), key=lambda kv: -sum(wb[r["id"]] for r in kv[1])):
        rs.sort(key=lambda r: -r["downloads"])
        out.append({"block": block, "out_of_scope": code == "out", "share_b": sum(wb[r["id"]] for r in rs) / tot_b,
                    "downloads_b": sum(wb[r["id"]] for r in rs),
                    "textgen": sum(r["downloads"] for r in rs if "textgen" in r["slices"]),
                    "embed": sum(r["downloads"] for r in rs if "embed" in r["slices"]),
                    "tokenizers": sorted({r["tok"] for r in rs if r.get("tok")}),
                    "models": [{"id": r["id"], "downloads": r["downloads"], "slices": r["slices"],
                                "tok": r.get("tok")} for r in rs]})
    return out


def report(st, out_json: Path, out_md: Path):
    rows, toks, chains, cfgc = build(st)
    done = code_today(st, rows, toks, chains)
    by_id = {r["id"]: r for r in rows}
    # a gated repo resolved through a file already in the census: name an ungated repo with the same bytes
    same = collections.defaultdict(list)
    for rid, rr in st["resolved"].items():
        if rr.get("sha256") and not rr.get("gated") and rr.get("status") == "ok" and rr.get("kind"):
            same[rr["sha256"]].append(rr.get("repo") or rid)
    for rid, rr in st["resolved"].items():
        if rr.get("gated") and rr.get("mirror") in (None, "(census)") and rr.get("status") == "gated-mirrored":
            cands = sorted(set(same.get(rr.get("sha256"), [])),
                           key=lambda x: (x.split("/")[1].lower() != rr["repo"].split("/")[1].lower(), x))
            rr["mirror"] = cands[0] if cands else "(an ungated repo in the census)"
        if rid in by_id and rr.get("gated") and rr.get("mirror"):
            by_id[rid]["source"] = rr["mirror"]
    wb, wreal = weights(rows, "b"), weights(rows, "b-real")
    wt, we, wv, wf = weights(rows, "textgen"), weights(rows, "embed"), weights(rows, "vlm"), weights(rows, "family")
    tot_b = sum(wb.values())
    crit_names, crit_f = critical_features(st, by_id, toks)
    order = greedy(rows, toks, wb, crit_f, crit_names)
    order_real = greedy(rows, toks, wreal, crit_f, crit_names)
    order_tg = greedy(rows, toks, wt, crit_f, crit_names)
    for c in chains.values():
        ms = [m for t in c["toks"] for m in toks[t]["models"]]
        PID_EXAMPLE[c["pid"]] = max(ms, key=lambda m: by_id[m]["downloads"]).split("/")[1] if ms else None
    slices = (("b", wb), ("textgen", wt), ("embed", we), ("vlm", wv), ("b-real", wreal), ("family", wf))
    pkgs = package_stats(package_order(rows, toks, wb, crit_f, crit_names), rows, toks, wb, slices)
    pkgs_tg = package_stats(package_order(rows, toks, wt, crit_f, crit_names), rows, toks, wt, slices)
    cols = {name: share_after(rows, toks, w, order["steps"])
            for name, w in (("b", wb), ("textgen", wt), ("embed", we), ("vlm", wv), ("b-real", wreal))}
    # which tokenizers / models each step unlocks (in b)
    covered = set()
    for i, s in enumerate(order["steps"]):
        before = set(covered)
        covered |= set(s["add"])
        newly = [r["id"] for r in rows if r["id"] in wb and r.get("tok")
                 and set(toks[r["tok"]]["features"]) <= covered and not set(toks[r["tok"]]["features"]) <= before]
        s["models"] = newly
        s["tokenizers"] = sorted({by_id[x]["tok"] for x in newly})
        s["share"] = s["gain"] / tot_b
        s["cum_share"] = s["cum"] / tot_b
        s["name"] = ("critical targets: " + ", ".join(s["critical"])) if s.get("critical") else step_name(s["add"])
        for k, v in cols.items():
            s["cum_" + k] = v[i + 1]
    tot_t = sum(wt.values()) or 1
    cum_tg = share_after(rows, toks, wt, order_tg["steps"])
    order_textgen = [{"add": s["add"], "gain": s["gain"], "share": s["gain"] / tot_t, "cum_share": cum_tg[i + 1],
                      **({"critical": s["critical"]} if s.get("critical") else {})}
                     for i, s in enumerate(order_tg["steps"])]
    # per-tokenizer b downloads already summed; per pattern / per feature aggregates over b
    pat_rows = []
    for key, c in chains.items():
        tl = [toks[t] for t in c["toks"]]
        ms = [m for t in tl for m in t["models"]]
        pat_rows.append({"chain": c["chain"], "toks": c["toks"], "models": ms,
                         "dl_b": sum(wb.get(m, 0) for m in ms), "dl_all": sum(by_id[m]["downloads"] for m in ms),
                         "template": tl[0]["template"], "variant": tl[0]["variant"],
                         "status": tl[0]["tmpl_status"], "n_b": sum(1 for m in ms if m in wb)})
    pat_rows.sort(key=lambda p: (-p["dl_b"], -p["dl_all"]))
    for i, p in enumerate(pat_rows):
        p["pid"] = f"P{i + 1:02d}"
        for t in p["toks"]:
            toks[t]["pid"] = p["pid"]
    feat_rows = {}
    for t in toks.values():
        for f in t["features"]:
            fr = feat_rows.setdefault(f, {"feature": f, "toks": [], "models": [], "dl_b": 0})
            fr["toks"].append(t["id"])
            fr["models"] += t["models"]
    for fr in feat_rows.values():
        fr["dl_b"] = sum(wb.get(m, 0) for m in fr["models"])
        fr["n_b"] = sum(1 for m in fr["models"] if m in wb)
    # covered-today components (for the component table): what the covered tokenizers use
    base_rows = []
    misses = [r for r in rows if r["id"] in wb and not r.get("tok")]
    data = {
        "census": {"tool": "tools/census/coverage.py + tools/census/report.py", "version": st.get("version"),
                   "date": st["date"], "hf_tokenizers": "0.23.2", "queries": st["queries"],
                   "targets": st["targets"], "downloads": "hub 30-day download count at census time"},
        "covered_today": {"kernels_md_strings": {v: k for k, v in KERNELS_MD.items()},
                          "config_c_strings": cfgc,
                          "components": sorted(["model:bpe-bytelevel (ignore_merges included)", "norm:NFC (m1b)",
                                                "pp:TemplateProcessing", "pp:ByteLevel", "dec:ByteLevel",
                                                "tmpl:none", "added tokens without lstrip/rstrip/single_word "
                                                "(both phases, m1b)"])},
        "totals": {"b_downloads": tot_b, "b_models": len(wb), "b_known_models": sum(1 for r in rows
                                                                                   if r["id"] in wb and r.get("tok")),
                   "b_tokenizers": len({r["tok"] for r in rows if r["id"] in wb and r.get("tok")}),
                   "covered_today_b": order["base"] / tot_b},
        "packages": pkgs,
        "order": order["steps"],
        "order_without_testing_repos": [{"add": s["add"], "share": s["gain"] / (order_real["total"] or 1)}
                                        for s in order_real["steps"]],
        "patterns": [{"pid": p["pid"], "template": p["template"], "variant": p["variant"], "status": p["status"],
                      "chain": p["chain"], "tokenizers": p["toks"], "models_b": p["n_b"], "models": len(p["models"]),
                      "downloads_b": p["dl_b"], "share_b": p["dl_b"] / tot_b,
                      "examples": sorted(p["models"], key=lambda m: -by_id[m]["downloads"])[:5]} for p in pat_rows],
        "features": [{"feature": f["feature"], "category": FEATURES.get(f["feature"], ("template", ""))[0],
                      "what": FEATURES.get(f["feature"], ("", f["feature"]))[1], "tokenizers": len(f["toks"]),
                      "models_b": f["n_b"], "downloads_b": f["dl_b"], "share_b": f["dl_b"] / tot_b,
                      "examples": sorted(set(f["models"]), key=lambda m: -by_id[m]["downloads"])[:5]}
                     for f in sorted(feat_rows.values(), key=lambda x: -x["dl_b"])],
        "tokenizers": {t["id"]: {k: v for k, v in t.items() if k not in ("chain",)}
                       for t in sorted(toks.values(), key=lambda t: -t["dl_b"])},
        "models": rows,
        "families": [],
        "misses_b": [{"id": r["id"], "downloads": r["downloads"], "miss": r.get("miss"), "slices": r["slices"]}
                     for r in sorted(misses, key=lambda r: -r["downloads"])],
        "mirrors": st.get("mirrors", {}),
    }
    # the text-generation-weighted orders sit next to their (b) twins; step 0 of every order is the critical targets
    reordered = {}
    for k, v in data.items():
        reordered[k] = v
        if k == "packages":
            reordered["packages_textgen"] = pkgs_tg
        elif k == "order":
            reordered["order_textgen"] = order_textgen
    data = reordered
    for x, s in zip(data["order_without_testing_repos"], order_real["steps"]):
        if s.get("critical"):
            x["critical"] = s["critical"]
    for fam in st["families"]:
        repo = fam["repo"]
        r = by_id.get(repo, {})
        rr = st["resolved"].get(repo, {})
        t = toks.get(r.get("tok")) if r.get("tok") else None
        stand = None
        if not t and rr.get("standin"):
            srec = st["analyzed"].get(rr["standin"].get("sha256"))
            if srec and "sig" in srec:
                tf = tokenizer_features(srec["sig"], "tokenizer.json")
                stand = {"repo": rr["standin"]["repo"], "hash": srec["hash"][:12], "template": tf["template"],
                         "variant": tf["variant"], "features": sorted(tf["features"])}
        data["families"].append({
            "family": fam["family"], "repo": repo, "tiktoken": fam.get("tiktoken"),
            "source": rr.get("mirror") if rr.get("gated") else None, "status": rr.get("status"),
            "tok": r.get("tok"), "pid": t.get("pid") if t else None,
            "template": t["template"] if t else None, "variant": t["variant"] if t else None,
            "missing": t["features"] if t else None, "standin": stand, "miss": r.get("miss")})
    data["today_slices"] = {k: share_after(rows, toks, w, [])[0] for k, w in
                            (("textgen", wt), ("embed", we), ("vlm", wv), ("family", wf))}
    btoks = {r["tok"] for r in rows if r["id"] in wb and r.get("tok")}
    nmiss = sum(1 for r in rows if r["id"] in wb and not r.get("tok"))
    data["today_slices"]["count"] = sum(1 for t in btoks if not toks[t]["features"]) / (len(btoks) + nmiss)
    slow = sum(wb[r["id"]] for r in rows if r["id"] in wb and r["kind"] in SLOW_INPUT)
    data["totals"]["ceiling_without_slow_inputs"] = 1 - (slow + sum(wb[r["id"]] for r in rows
                                                                    if r["id"] in wb and not r.get("tok"))) / tot_b
    use = collections.Counter()
    for t in toks.values():
        sg = t["sig"]
        if t["algorithm"] == "bpe-bytelevel":
            use["byte-level BPE"] += 1
            if (sg.get("model") or {}).get("ignore_merges"):
                use["ignore_merges"] += 1
        for n in sg.get("normalizer") or []:
            if n["type"] in NORM_COVERED:
                use["norm " + n["type"]] += 1
        for x in sg.get("post_processor") or []:
            if x["type"] in PP_COVERED:
                use["pp " + x["type"]] += 1
        a = sg.get("added_tokens") or {}
        if a.get("normalized"):
            use["added tokens with normalized=true"] += 1
    data["covered_components_in_use"] = dict(use.most_common())
    data["critical"] = critical(st, rows, toks, by_id, cfgc)
    data["packages_alone"] = packages_alone(rows, toks, (("b", wb), ("textgen", wt), ("embed", we), ("vlm", wv)))
    data["tiktoken"] = tiktoken_check(st, data["families"], toks)
    data["findings"] = findings(data, rows, toks, wb, cfgc)
    data["totals"]["loads_today_b"] = sum(wb[r["id"]] for r in rows if r["id"] in wb and r.get("code") == "loads") / tot_b
    data["code_today"] = {"shares": code_shares(rows, (("b", wb), ("textgen", wt), ("embed", we), ("vlm", wv),
                                                       ("family", wf), ("b-real", wreal))),
                          "misses_b": code_misses(rows, wb, tot_b), "done_features": sorted(done)}
    data["probe"] = probe_summary(st, rows, toks, wb, tot_b)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(dump_lines(data))
    write_md(data, out_md, by_id, toks)
    print(f"report: {out_json} {out_md}")


# ---------------------------------------------------------------------------------------------------------
# critical targets (the five): the target's tokenizer, and every family / quantizer / mirror repo against it

CRIT_ORGS_MD = ("unsloth", "nvidia", "lmstudio-community", "mlx-community", "bartowski", "RedHatAI")
KIMI_PY_READ = "git:5df9d462b43295245084fbab46fd956020d4b84e"   # the tokenization_kimi.py the notes below describe

TARGET_NOTES = {
    "Kimi K3": [
        "The tokenizer is `tokenization_kimi.py` (remote code) over `tiktoken.model`; there is no tokenizer.json, so "
        "hf tokenizers cannot load it and the model's own code is the reference (SPEC 0.3).",
        "`encode()` adds no BOS and recognizes every special token (`allowed_special=\"all\"`): 256 reserved ids "
        "after the ranks, named from tokenizer_config.json's `added_tokens_decoder`, else `<|reserved_token_i|>`.",
        "It cuts the text into 400,000-character chunks, then cuts each chunk wherever a run of whitespace or of "
        "non-whitespace (python `str.isspace`) passes 25,000 characters, and runs tiktoken on each piece: pieces "
        "never span those cuts, so toks must reproduce them to stay exact on long inputs.",
        "The pattern is tiktoken (fancy-regex) syntax: `\\p{Han}` and class intersection `&&[^\\p{Han}]`, "
        "which neither the cl100k nor the o200k template expresses.",
    ],
    "Qwen 3.8": [],     # computed in critical(): it depends on config.c's strings and on the probe
    "Gemma 4": [
        "Sentencepiece-style BPE: byte_fallback with `<0x00>`..`<0xFF>`, unk `<unk>` with fuse_unk. The normalizer "
        "turns every ' ' into U+2581 before the pre-tokenizer's `Split(' ', MergedWithPrevious)`, so the split "
        "never fires and each segment between added tokens is one BPE piece. The post-processor adds no BOS "
        "(single template `$A`).",
    ],
    "gpt-oss": [
        "o200k pattern (case-split letter runs with an optional contraction suffix, `\\p{N}{1,3}`, punctuation "
        "tail `[\\r\\n/]*`), ignore_merges, 21 special added tokens (harmony format).",
    ],
    "GLM 5.3": [],
}


def critical(st, rows, toks, by_id, cfgc=()):
    res, an, models = st["resolved"], st["analyzed"], st["models"]
    out = []
    for c in st.get("critical") or []:
        tgt = c["repo"]
        r0 = res.get(tgt) or {}
        t0 = toks.get(by_id.get(tgt, {}).get("tok")) if by_id.get(tgt, {}).get("tok") else None
        a0 = an.get(r0.get("sha256")) or {}
        entry = {"target": c["target"], "repo": tgt, "kind": r0.get("kind"), "tok": t0["id"] if t0 else None,
                 "file_sha256": a0.get("file_sha256"), "repos": [], "notes": list(TARGET_NOTES.get(c["target"], []))}
        if t0:
            entry.update(template=t0["template"], variant=t0["variant"], status=t0["tmpl_status"], pid=t0.get("pid"),
                         features=t0["features"], probe=t0.get("probe"), signature=signature_text(t0),
                         chain=t0["chain"])
        if r0.get("kind") == "tiktoken":
            entry["tiktoken"] = {"py": r0.get("py_key"), "tk_config": r0.get("tk_config")}
            if r0.get("py_key") and r0["py_key"] != KIMI_PY_READ and c["target"] == "Kimi K3":
                entry["notes"].append(f"tokenization_kimi.py is now `{r0['py_key']}`, not the `{KIMI_PY_READ}` these "
                                      "notes were written from: re-read it.")
        if t0 and any((x.get("pattern") or {}).get("Regex") == P_QWEN35 for x in (t0["chain"] or [])):
            pr = t0.get("probe") or {}
            if P_QWEN35 in cfgc:
                txt = ("The pattern is the hub's qwen 3.5 string (kernels.md section 3), which `src/core/config.c` "
                       "compiles (TOKS_PATTERNS).")
            else:
                txt = ("The pattern is the kernels.md section 3 qwen 3.5 string, but `src/core/config.c` compiles a "
                       "different spelling: the exact-string match refuses the hub file (section 9).")
            an_ = pr.get("after_nfc")
            if pr.get("what") == "normalizer NFC (m1b)" and an_:
                txt += (" The load path refuses the file for its NFC normalizer alone (m1b): with only the normalizer "
                        "removed it loads (section 9)." if an_.get("stage") == 0 else
                        f" NFC is m1b; with only the normalizer removed the load path still refuses it: "
                        f"{an_.get('what')}.")
            entry["notes"].append(txt)
        for rid in c["repos"]:
            r = res.get(rid) or {}
            m = models.get(rid) or {}
            a = an.get(r.get("sha256")) or {}
            row = {"repo": rid, "downloads": m.get("downloads"), "kind": r.get("kind"), "via": r.get("via") or [],
                   "status": r.get("status"), "tok": a.get("hash", "")[:12] or None, "gguf": None}
            gg = r.get("gguf") or {}
            if gg and not gg.get("error"):
                row["gguf"] = {k: gg.get(k) for k in ("tokenizer.ggml.model", "tokenizer.ggml.pre", "n_tokens",
                                                      "truncated")}
            if not r.get("kind"):
                row["same"] = "no tokenizer found"
            elif rid == tgt:
                row["same"] = "target"
            elif r.get("kind") != r0.get("kind") and "tiktoken" in (r.get("kind"), r0.get("kind")):
                row["same"] = f"differs: another tokenizer kind ({r.get('kind')})"
            elif r.get("kind") == "tiktoken" and r0.get("kind") == "tiktoken":
                diffs = [k for k, x, y in (("tiktoken.model", r.get("sha256"), r0.get("sha256")),
                                           ("tokenization_kimi.py", r.get("py_key"), r0.get("py_key")),
                                           ("special-token names", (r.get("tk_config") or {}).get("names_sha256"),
                                            (r0.get("tk_config") or {}).get("names_sha256"))) if x != y]
                row["same"] = "identical (ranks, code, special names)" if not diffs else "differs: " + ", ".join(diffs)
            elif a.get("file_sha256") and a.get("file_sha256") == a0.get("file_sha256"):
                row["same"] = "identical file"
            elif a.get("hash") and a.get("hash") == a0.get("hash"):
                row["same"] = "same tokenizer (file bytes differ)"
            elif a.get("parts") and a0.get("parts"):
                d = [k for k in a["parts"] if a["parts"].get(k) != a0["parts"].get(k)]
                row["same"] = "differs: " + ", ".join(d)
            else:
                row["same"] = "differs"
            if row["via"] and row["same"] not in ("target",):
                row["same"] += " (repo has no tokenizer; via " + row["via"][-1] + ")"
            entry["repos"].append(row)
        groups = collections.defaultdict(list)
        for row in entry["repos"]:
            if row["same"].startswith("differs") and row.get("tok"):
                groups[(row["tok"], row["same"].split(" (repo has")[0])].append(row)
        entry["variants"] = []
        for (tid, why), rs in sorted(groups.items(), key=lambda kv: -sum(r["downloads"] or 0 for r in kv[1])):
            t = toks.get(tid) or {}
            entry["variants"].append({"tok": tid, "why": why, "pid": t.get("pid"), "variant": t.get("variant"),
                                      "repos": [r["repo"] for r in rs],
                                      "downloads": sum(r["downloads"] or 0 for r in rs)})
        out.append(entry)
    return out


def signature_text(t):
    sg = t["sig"]
    if "tiktoken" in sg:
        return (f"tiktoken ranks ({sg['tiktoken'].get('ranks')}): byte-level BPE by rank; no normalizer; the "
                "pattern of the next column; 256 reserved special ids; decoder: bytes")
    m = sg["model"]
    bits = [t["algorithm"]]
    flags = [k for k in ("ignore_merges", "byte_fallback", "fuse_unk") if m.get(k)]
    if m.get("unk_token"):
        flags.append(f"unk {m['unk_token']}")
    if flags:
        bits[0] += " (" + ", ".join(flags) + ")"
    bits.append(f"vocab {m.get('vocab_size')}" + (f", {m.get('merges')} merges" if m.get("merges") else ""))
    bits.append("normalizer " + (" > ".join(n["type"] + (f"({json.dumps(pat(n)[0], ensure_ascii=False)}"
                                                         f"->{json.dumps(n.get('content'), ensure_ascii=False)})"
                                                         if n["type"] == "Replace" else "")
                                            for n in sg["normalizer"]) or "none"))
    bits.append("pre-tokenizer " + chain_text(sg["pre_tokenizer"]))
    bits.append("decoder " + (" > ".join(x["type"] for x in sg["decoder"]) or "none"))
    pp = []
    for x in sg["post_processor"]:
        pp.append(x["type"] + (f"[{x['single']}]" if x["type"] == "TemplateProcessing" else ""))
    bits.append("post-processor " + (" > ".join(pp) or "none"))
    a = sg["added_tokens"]
    opts = [f"{k} {a[k]}" for k in ("lstrip", "rstrip", "single_word", "normalized") if a.get(k)]
    bits.append(f"{a['n']} added tokens ({a['special']} special" + (", " + ", ".join(opts) if opts else "") + ")")
    if sg.get("truncation"):
        bits.append(f"truncation max_length {sg['truncation'].get('max_length')}")
    if sg.get("padding"):
        bits.append(f"padding {json.dumps(sg['padding'].get('strategy'))}")
    return "; ".join(bits)


STAGES = {1: "config refuses", 2: "load refuses", 8: "toks_load, toks_load_mem_copy and the config parse disagree",
          9: "unreadable"}   # tools/census/probe.c


def probe_summary(st, rows, toks, wb, tot_b):
    """what the library's load path does with each tokenizer today (tools/census/probe.c), against the census's
    rules (`census_covered`: needs no feature outside today's done set)."""
    sp = st.get("probe") or {}
    out = {"commit": sp.get("commit"), "host": sp.get("host"), "date": sp.get("date"), "stages": sp.get("stages"),
           "files": sp.get("files"), "tiktoken_dirs": sp.get("tiktoken_dirs"), "rows": []}
    try:
        import subprocess
        lib = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", out["commit"], "--", "src",
                              "include", "Makefile"], capture_output=True, text=True, timeout=30)
        out["library_commit"] = lib.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- a report without git still renders
        out["library_commit"] = None
    agg = collections.defaultdict(lambda: {"toks": 0, "dl_b": 0, "covered": 0})
    for t in toks.values():
        pr = t.get("probe")
        if not pr or any(f.startswith("input:slow") for f in t["design_features"]):
            continue
        key = "loads" if pr["stage"] == 0 else STAGES.get(pr["stage"], f"stage {pr['stage']}") + ": " + pr["what"]
        if t["fkind"] == "tiktoken":
            key = "tiktoken model directory: " + key
        a = agg[key]
        a["toks"] += 1
        a["dl_b"] += t["dl_b"]
        a["covered"] += 0 if t["features"] else 1
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["dl_b"]):
        out["rows"].append({"result": k, "tokenizers": a["toks"], "census_covered": a["covered"],
                            "share_b": a["dl_b"] / tot_b})
    return out


# ---------------------------------------------------------------------------------------------------------
# findings: facts the census turned up that other lanes act on (each computed, printed only when it holds)

def findings(d, rows, toks, wb, cfgc):
    tb = d["totals"]["b_downloads"]
    out = []

    def share(tids):
        return sum(wb.get(m, 0) for t in tids for m in toks[t]["models"]) / tb

    def pids(pred):
        return [x for x in d["patterns"] if pred(x)]
    q35 = pids(lambda x: x["chain"] and any(e.get("pattern", {}).get("Regex") == P_QWEN35 for e in x["chain"]))
    if cfgc and P_QWEN35 not in cfgc and q35:
        bad = [c for c in cfgc if c not in KERNELS_MD]
        tids = [t for x in q35 for t in x["tokenizers"]]
        out.append(f"**config.c refuses the qwen 3.5 family.** `src/core/config.c` TOKS_PATTERNS holds "
                   f"`{json.dumps(bad[0]) if bad else '?'}`; every hub file of qwen 3.5 / 3.6 / 3.8 "
                   f"({', '.join(x['pid'] for x in q35)}: {len(tids)} tokenizers, {fmt_pct(share(tids))} of (b)) "
                   f"spells the letter prefix `[^\\r\\n\\p{{L}}\\p{{N}}]?` (no `\\p{{M}}`), which is kernels.md "
                   f"section 3's prose applied to the qwen 2 string. The exact-string match fails, so the compile "
                   f"lane should replace the entry with `{json.dumps(P_QWEN35)}`.")
    ri = [t["id"] for t in toks.values() if "compile:split-removed-invert" in t["features"]]
    if ri:
        ex = sorted({m for t in ri for m in toks[t]["models"]}, key=lambda m: -next(
            r["downloads"] for r in rows if r["id"] == m))[:5]
        out.append(f"**Split(behavior=Removed, invert=true)** spells {len(ri)} tokenizers' split "
                   f"({fmt_pct(share(ri))} of (b); e.g. {', '.join('`' + e + '`' for e in ex)}). Every character "
                   "matches some alternative of these cl100k / o200k patterns, so the kept matches are the same "
                   "pieces Isolated gives; config.c refuses both fields today (`compile:split-removed-invert`).")
    tr = [t["id"] for t in toks.values() if "file:truncation" in t["features"]]
    pd = [t["id"] for t in toks.values() if "file:padding-fixed" in t["features"]]
    if tr or pd:
        out.append(f"**hf `encode()` applies tokenizer.json's truncation and padding.** {len(tr)} tokenizers "
                   f"({fmt_pct(share(tr))} of (b)) set truncation and {len(pd)} ({fmt_pct(share(pd))}) set Fixed "
                   "padding (sentence-transformers exports: `all-MiniLM-L6-v2` pads every encoding to 128). "
                   "tokenizers 0.23.2 `post_process` truncates to max_length minus the post-processor's added "
                   "tokens, adds them, then pads; a flags-0 `toks_encode` must do the same or refuse the file.")
    pb = [t["id"] for t in toks.values() if "bpe:partial-byte-alphabet" in t["features"]]
    if pb:
        sizes = collections.Counter(toks[t]["sig"]["model"].get("byte_alphabet") for t in pb)
        out.append(f"**Byte-level vocabularies with holes.** {len(pb)} byte-level BPE tokenizers "
                   f"({fmt_pct(share(pb))} of (b): pythia / gpt-neox, SmolLM, bloom, falcon, ModernBERT and "
                   f"granite-embedding, deepseek-coder) lack some of the 256 byte chars (alphabet sizes "
                   f"{', '.join(f'{k}: {v}' for k, v in sorted(sizes.items()))}). hf's BPE drops a char that has no "
                   "vocab entry when unk_token is null (models/bpe/model.rs `merge_word` has no else branch) and "
                   "emits unk otherwise; config.c refuses these files.")
    gm = pids(lambda x: x["variant"] and x["variant"].startswith("Split(' ') after Replace"))
    np_ = pids(lambda x: x["variant"] == "no pre-tokenizer (one piece per segment)")
    ms = pids(lambda x: x["variant"] and x["variant"].endswith("split=false): no split"))
    if gm or np_ or ms:
        tids = [t for x in gm + np_ + ms for t in x["tokenizers"]]
        mex = sorted({e for x in ms for e in x["examples"]}, key=lambda m: -next(
            r["downloads"] for r in rows if r["id"] == m))[:3]
        out.append(f"**Whole-segment BPE.** gemma 2 / 3 / 4 ({', '.join(x['pid'] for x in gm)}) normalize ' ' to "
                   "U+2581 before `Split(' ', MergedWithPrevious)`, so the split never fires; "
                   + (f"Metaspace(split=false) files ({', '.join(x['pid'] for x in ms)}: "
                      f"{', '.join('`' + e + '`' for e in mex)}) replace and prepend without splitting; " if ms else "")
                   + f"llama 2 / phi-3 / mistral v0.1 style files ({', '.join(x['pid'] for x in np_)}) have no "
                   f"pre-tokenizer at all. In all of them ({len(tids)} tokenizers, {fmt_pct(share(tids))} of (b)) each "
                   "segment between added tokens is a single BPE piece as long as the input: K6 needs its heap path "
                   "for long pieces and K5's piece caches never hit.")
    ig = [t["id"] for t in toks.values() if (t["sig"].get("model") or {}).get("ignore_merges")]
    if ig:
        out.append(f"`ignore_merges` is set on {len(ig)} tokenizers ({fmt_pct(share(ig))} of (b): llama 3.x, gpt-oss, "
                   "glm 4-5, nemotron, mistral nemo, ...); TOKS_TF_IGNORE_MERGES covers it.")
    ls = [t["id"] for t in toks.values() if "added:lstrip" in t["features"]]
    rs = [t["id"] for t in toks.values() if "added:rstrip" in t["features"]]
    if ls:
        out.append(f"`lstrip` is on {len(ls)} tokenizers ({fmt_pct(share(ls))} of (b)): the `<mask>` / `[MASK]` "
                   f"token of the xlm-r, mpnet and ModernBERT families, and every special token of phi-4; `rstrip` "
                   f"on {len(rs)} ({fmt_pct(share(rs))}: phi-3.5 / phi-4). No tokenizer in the census uses "
                   f"`single_word`." if not any("added:single_word" in t["features"] for t in toks.values())
                   else "")
    tkc = (d.get("tiktoken") or {}).get("checks") or []
    tkd = [x for x in tkc if x["equal"] is False]
    if tkd:
        out.append(f"**tiktoken spells {', '.join(x['encoding'] for x in tkd)} differently from the hub.** tiktoken "
                   f"{d['tiktoken']['version']}'s pattern strings for them are not the strings of their hub "
                   f"tokenizer.json ({', '.join((x['pid'] or '-') + ' `' + str(x['repo']) + '`' for x in tkd)}: "
                   "tiktoken's strings in section 7, the hub's in appendix A)"
                   + (f", while {', '.join(x['encoding'] for x in tkc if x['equal'])} match exactly"
                      if any(x["equal"] for x in tkc) else "") +
                   ". config.c compares pattern strings exactly, so a `.tiktoken` reader (SPEC 0.3) that hands it "
                   "tiktoken's string needs that spelling in the table too, with a proof (or an enumeration) that it "
                   "splits like the hub string.")
    km = [t for t in toks.values() if t["template"] == "kimi"]
    if km:
        out.append(f"kimi k2 / k2.5 / k3 / linear (and the RadixArk k3 draft) load one tiktoken file "
                   f"(`tiktoken.model`, {km[0]['sig']['tiktoken']['ranks']} ranks) with one pattern from "
                   "`tokenization_kimi.py`: `[\\p{Han}]+` first, then o200k-style letter runs written with class "
                   "intersection (`[\\p{Lu}...\\p{M}&&[^\\p{Han}]]`): the o200k template's kimi parameters; the special "
                   "tokens live in tokenizer_config.json, not in the tiktoken file, and the repos' wrappers and "
                   "configs differ (section 8 lists the ones the kimi reader refuses).")
    for e in d.get("critical") or []:
        for v in e.get("variants") or []:
            if v["downloads"] >= 1_000_000:
                parts = set(v["why"].replace("differs:", "").replace(" ", "").split(","))
                if parts & {"pre_tokenizer", "model", "normalizer"}:
                    eff = "they split or merge some text differently from the target"
                elif parts & {"added_tokens"}:
                    eff = "their added tokens differ from the target's"
                elif parts & {"post_processor"}:
                    eff = "`encode()` adds different special tokens around the text"
                else:
                    eff = "`encode()` truncates or pads differently"
                out.append(f"**{e['target']}: family repos with another tokenizer.** {len(v['repos'])} repos "
                           f"({fmt_n(v['downloads'])} downloads, e.g. {', '.join('`' + x + '`' for x in v['repos'][:3])}) "
                           f"ship tokenizer `{v['tok']}` ({v.get('pid')}: {v.get('variant')}); against `{e['repo']}` it "
                           f"{v['why']}, so {eff}. Parity suites must pin the exact file they test.")
    vb = [r for r in rows if r["id"] in wb and any("(" in v for v in r["via"])]
    vbase = [r for r in rows if r["id"] in wb and r["via"] and not any("(" in v for v in r["via"])]
    out.append(f"{len(vbase)} models in (b) ({fmt_pct(sum(wb[r['id']] for r in vbase) / tb)}) carry no tokenizer and "
               "resolve through their card's base_model (gguf quantizations, adapters); "
               f"{len(vb)} more ({fmt_pct(sum(wb[r['id']] for r in vb) / tb)}) name no base and resolve by the same "
               "model name without the packaging suffix (gguf repos also by the gguf header; their `n_tokens` is "
               "recorded in coverage.json). Without them the denominators shrink and the order does not change.")
    sl = [r for r in rows if r["id"] in wb and r["kind"] in SLOW_INPUT]
    if sl:
        out.append(f"{len(sl)} models in (b) ({fmt_pct(sum(wb[r['id']] for r in sl) / tb)}) ship only slow files "
                   "(facebook/opt, DialoGPT, rinna, t5 variants, vocab.txt-only BERTs). The census analyzed the fast "
                   "tokenizer transformers builds from them. SPEC 1.5 keeps these inputs out; section 8 says "
                   "whether each conversion would load.")
    return [x for x in out if x]


METHOD = [
    "- **Population.** Anonymous hub API, `GET /api/models?pipeline_tag=<tag>&sort=downloads&direction=-1`. (b): "
    "text-generation top 500; embedding / reranker: the union of sentence-similarity, feature-extraction, "
    "text-ranking and `filter=reranker` listings in download order, the first 100 that carry a text tokenizer (the "
    "non-text repos skipped on the way are in coverage.json `models` without a slice). (d): image-text-to-text and "
    "any-to-any, top 100 together. (c): one canonical repo per family (section 7). `downloads` is the hub's 30-day "
    "count when listed.",
    "- **Which tokenizer a model loads.** tokenizer.json at the repo root (else one directory down); else tekken.json, "
    "a tiktoken file (pattern parsed from tokenization_*.py), remote code only, or slow files. A repo with none of "
    "them (gguf quantizations, adapters, speculative-decoding drafts) follows its card's base_model chain, then the "
    "gguf header's `general.base_model.*.repo_url`, then the same model name without its packaging suffix (-GGUF, "
    "-DSpark, -FP8, ...); coverage.json `via` records the route. Slow-file repos get the fast tokenizer transformers "
    "5.18 builds from them (AutoTokenizer, no remote code).",
    "- **Gated repos.** The hub's blob listing gives the file's content key (git blob sha-1 or lfs sha-256) without "
    "access; the census uses an ungated repo whose file has that key (byte-identical, verified by hashing the "
    "download). Without one the model is listed as gated; for a named family a same-name ungated repo is shown as a "
    "labelled stand-in, never counted.",
    "- **Dedupe.** hf tokenizers 0.23.2 `Tokenizer.from_str(file).to_str()`, json with sorted keys, `version` "
    "dropped, sha-256. Files hf reads identically hash identically (old `\"a b\"` vs new `[\"a\", \"b\"]` merges, "
    "gpt-2's untyped model, Metaspace's `add_prefix_space` vs `prepend_scheme`). Every file in the census loads in "
    "hf 0.23.2 (coverage.json `hf_loads`).",
    "- **Signature.** From the normalized form: model type and flags (byte_fallback, ignore_merges, unk_token, "
    "fuse_unk, dropout, continuing_subword_prefix, end_of_word_suffix, byte-alphabet / byte-token completeness), "
    "normalizer chain (Precompiled charsmaps by hash), pre-tokenizer chain with the exact regex strings, decoder "
    "chain, post-processor (template shapes and ids), added-token option counts, truncation, padding.",
    "- **Covered today.** toks's own load path accepts the model's files (section 9). A census feature (template, "
    "algorithm, normalizer and other components, model flag, added-token or encode option, input format; rules in "
    "`tools/census/report.py`) counts as done when a loading tokenizer uses it; a refused tokenizer needs its "
    "features nobody loads yet, or, when every one is in use elsewhere, the load path's refusal (`code:`). The orders "
    "of sections 1-2 start from there.",
    "- **Critical targets.** For each of section 0's five: the exact repo, then hub searches for the family stem "
    "(the 30 most downloaded repos whose id carries it, plus every such repo of unsloth, nvidia, lmstudio-community, "
    "mlx-community, bartowski and RedHatAI). Each repo goes through the same resolution; gguf repos also get their "
    "header read (llama.cpp's `tokenizer.ggml.model` / `.pre` / token count). A repo matches the target when its "
    "file has the same bytes or hf's normalized configuration hashes the same; otherwise the differing top-level "
    "parts are named (model, added_tokens, normalizer, pre_tokenizer, post_processor, decoder, truncation, "
    "padding). tiktoken repos compare the ranks file, tokenization_*.py and tokenizer_config.json's special names.",
    "- **The code today.** `tools/census/probe.c` links the library and calls `toks_load` on every tokenizer.json "
    "the census read (slow-input conversions included) and on one model directory per distinct tiktoken model (the "
    "repo's ranks file, tokenization_*.py and tokenizer_config.json under their own names; coverage.py fetches each "
    "distinct tokenizer_config.json once and keeps it only when its sha-256 equals the one recorded at census time). "
    "Each file also loads through `toks_load_mem_copy` and its config parse alone names the stage; a disagreement "
    "is stage 8.",
    "- **Order.** Greedy budgeted coverage: step 0 is everything the critical targets' own tokenizers need, "
    "whatever it unlocks; then each step adds the missing features of one tokenizer group, the group whose features "
    "unlock the most downloads per unit of work (template, algorithm, normalizer, pre-tokenizer component, model flag, "
    "added-token option, encode option, input format: 1; decoder, post-processor mapping, compiler spelling: 1/4). "
    "Section 1 runs the same greedy over work packages (fixed groups of features, `PACKAGES` in report.py: a feature "
    "belongs to the package whose tokenizers spell with it, e.g. gemma's no-op Split and mistral's "
    "Metaspace(split=false) to the sentencepiece-style BPE stack), once weighted by (b) downloads and once by "
    "text-generation downloads alone, and shows what each package unlocks alone.",
    "- **Caveats.** Downloads include CI traffic (testing repos are flagged; section 1 shows the order without "
    "them). Name-matched gguf / draft repos are inferences (gguf `n_tokens` recorded for checking). The `compile` "
    "equivalences are argued from the regexes, not yet proven by enumeration (SPEC 5.3 style); the generic "
    "engine counts as coverage (it is exact) but not as a fast path (SPEC 1.4; section 1 gives both). The census sees each file as hf does; transformers-only "
    "behaviour (tokenizer_config.json's add_bos_token, chat templates) is outside it.",
    "- **An earlier, unreviewed census, corrected here.** It used `Xenova/gpt-4o` (o200k) for cl100k, "
    "reported kimi k2 and glm 4 as missing (both exist: tiktoken + remote code, and `zai-org/glm-4-9b-chat-hf`), "
    "called gpt-2's untyped model `unknown`, labelled the qwen 2 pattern `gpt2-family`, deduped by file bytes "
    "before hf's normalization, counted gguf-only repos (8 of the top text-generation models) as misses, and "
    "transcribed o200k_base's pattern without the '/' of `[\\r\\n/]*` (its snapshot.json `tiktoken_fallbacks`, found "
    "by the o200k template work; never merged). This census copies no pattern by hand: tokenizer.json strings are hf's own "
    "re-serialization, tiktoken strings come from tiktoken itself (section 7).",
    "",
    "Re-run (network phases on a lab host, maintainer doctrine): `tools/remote.sh <host> 'for p in collect critical resolve "
    "fetch; do uv run tools/census/coverage.py $p; done && uv run --with transformers --with sentencepiece --with "
    "protobuf tools/census/coverage.py convert && uv run tools/census/coverage.py analyze && uv run --with tiktoken "
    "tools/census/coverage.py tiktoken && CENSUS_COMMIT=<sha> uv run tools/census/coverage.py probe && uv run "
    "tools/census/coverage.py state'` (state under `$TOKS_CENSUS_CACHE`, default `~/.cache/toks/census`; "
    "CENSUS_COMMIT names the commit, remote.sh syncs without .git), copy `coverage/state.json` back, then `uv run "
    "tools/census/coverage.py report --state state.json` writes census/coverage.json and docs/coverage.md. A code-only "
    "refresh re-runs just `tiktoken`, `probe` and `state` over the same files.",
]


# ---------------------------------------------------------------------------------------------------------
# markdown

def esc(s):
    """plain table text: pipes escaped, angle brackets as entities (`<s>`, `<unk>` are not html)."""
    return str(s).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def esc_code(s):
    """text inside a code span in a table: only pipes (code spans show entities literally)."""
    return str(s).replace("|", "\\|")


def write_critical(d, w, tb):
    cr = d.get("critical") or []
    if not cr:
        return
    w("## 0. Critical targets")
    w("")
    w("The five critical targets: step 0 of every build order below (sections 1 and 2), whatever their rank. *toks today* is toks's "
      "own load path run on the model's files as it ships them (section 9). Exact patterns are json string literals "
      "of the regex.")
    w("")
    w("| target | repo / tokenizer | component signature | exact pattern | toks today | blocks it | family + mirrors |")
    w("|---|---|---|---|---|---|---|")
    for e in cr:
        if not e.get("tok"):
            w(f"| **{e['target']}** | `{e['repo']}` | - | - | - | no readable tokenizer ({e.get('kind')}) |")
            continue
        rx = [x["pattern"]["Regex"] for x in (e.get("chain") or []) if x.get("type") == "Split"
              and "Regex" in (x.get("pattern") or {})]
        if e.get("chain") and e["chain"][0].get("type") == "ByteLevel" and e["chain"][0].get("use_regex"):
            rx = [P_GPT2]
        pat_cell = "<br>".join("`" + esc_code(json.dumps(x, ensure_ascii=False)) + "`" for x in rx) or \
            esc(chain_text(e.get("chain") or []))
        pr = e.get("probe")
        code = "loads" if pr and pr["stage"] == 0 else f"refused: {pr['what']}" if pr else "not probed"
        blocks = ", ".join(f"`{f}`" for f in e["features"]) or "nothing"
        same = [r for r in e["repos"] if r["same"].startswith(("identical", "same tokenizer"))]
        diff = [r for r in e["repos"] if r["same"].startswith("differs")]
        fam = (f"{len(same)} of {len(e['repos']) - 1} other repos load it exactly"
               + (f"; {len(diff)} ship another tokenizer (below)" if diff else ""))
        w(f"| **{e['target']}** | `{e['repo']}`<br>`{e['tok']}` ({e.get('pid')}, {e['template']}: "
          f"{esc(e.get('variant') or '')}) | {esc(e['signature'])} | {pat_cell} | {esc(code)} | {blocks} | {fam} |")
    w("")
    for e in cr:
        w(f"**{e['target']}** (`{e['repo']}`)")
        w("")
        for n in e["notes"]:
            w(f"- {n}")
        if e.get("tiktoken"):
            tk = e["tiktoken"]
            c = tk.get("tk_config") or {}
            w(f"- tiktoken file `{e.get('file_sha256', '')[:16]}`, tokenization_kimi.py `{tk.get('py')}`, "
              f"{c.get('named_specials')} named special tokens in tokenizer_config.json (bos `{c.get('bos')}`, eos "
              f"`{c.get('eos')}`).")
        for v in e.get("variants") or []:
            ex = ", ".join(f"`{x}`" for x in v["repos"][:4]) + (", ..." if len(v["repos"]) > 4 else "")
            w(f"- {len(v['repos'])} repo{'s' if len(v['repos']) > 1 else ''} ({fmt_n(v['downloads'])} downloads; "
              f"{ex}): tokenizer `{v['tok']}` ({v.get('pid') or '-'}: {esc(v.get('variant') or '-')}); against the "
              f"target it {v['why']}.")
        if e["notes"] or e.get("tiktoken") or e.get("variants"):
            w("")
        same = sum(1 for r in e["repos"] if r["same"].startswith(("identical", "same tokenizer")))
        w(f"{len(e['repos']) - 1} other repos (the family's other sizes, the most downloaded repos whose id carries "
          f"the family name, and the quantizer / mirror orgs {', '.join(CRIT_ORGS_MD)}); {same} load the target's "
          "tokenizer exactly (`identical file` = same bytes; `same tokenizer` = hf reads them identically).")
        w("")
        w("| repo | downloads | ships | tokenizer | vs the target | gguf (llama.cpp model / pre / n_tokens) |")
        w("|---|---:|---|---|---|---|")
        for r in e["repos"]:
            g = r.get("gguf")
            gtxt = (f"{g.get('tokenizer.ggml.model')} / {g.get('tokenizer.ggml.pre') or '-'} / {g.get('n_tokens')}"
                    if g and g.get("n_tokens") else "token list past the first 4 MiB" if g and g.get("truncated")
                    else "-" if g else "")
            ships = r.get("kind") or "-"
            if r["via"]:
                ships = "gguf / weights only" if g else "no tokenizer files"
            w(f"| `{r['repo']}` | {fmt_n(r['downloads'] or 0)} | {esc(ships)} | `{r.get('tok') or '-'}` | "
              f"{esc(r['same'])} | {esc(gtxt)} |")
        w("")


def write_md(d, path: Path, by_id, toks):
    T = d["totals"]
    tb = T["b_downloads"]
    L = []
    w = L.append
    pk, steps = d["packages"], d["order"]
    n99 = next((i for i, x in enumerate(pk) if x["cum_share"] >= 0.99), None)
    s99 = next((i for i, x in enumerate(steps) if x["cum_share"] >= 0.99), None)
    c = d["census"]
    w("# toks coverage: the templates and components \"every model\" needs, in order")
    w("")
    w(f"Census of {c['date'][:10]} by `tools/census/coverage.py` (data: `census/coverage.json`; method: section 10). "
      f"Population, SPEC 1.1: **(b)** the top {c['targets']['textgen']} text-generation and the top "
      f"{c['targets']['embed']} embedding / reranker models on the hf hub by downloads (the hub's 30-day count), "
      f"**(c)** the named families, and **(d)** an extension slice, the top {c['targets']['vlm']} image-text-to-text / "
      f"any-to-any models (multimodal llms; their text side is an llm tokenizer), reported in its own column and never "
      f"mixed into (b). Shares are of (b)'s **{fmt_n(tb)} downloads** ({T['b_models']} models, {T['b_tokenizers']} "
      f"distinct tokenizers by normalized-configuration hash); a model whose tokenizer could not be read counts as a "
      f"miss. *Covered today* means toks's own load path (`toks_load` at "
      f"`{((d.get('probe') or {}).get('commit') or '?')[:10]}`, section 9) accepts the model's tokenizer as the model "
      f"ships it; toks refuses at load what it cannot reproduce exactly, and the pinned files of "
      f"tests/data/targets/ledger.txt carry the hf parity checks. The orders below start from there: a feature some "
      f"loading tokenizer uses counts as done.")
    w("")
    write_critical(d, w, tb)
    w("## 1. Headline")
    w("")
    pr = d.get("probe") or {}
    cs = (d.get("code_today") or {}).get("shares") or {}

    def sh(k, f="loads"):
        return fmt_pct((cs.get(k) or {}).get(f, 0))
    w(f"**Today toks loads {sh('b')} of (b) by downloads** (text-generation {sh('textgen')}, embedding / reranker "
      f"{sh('embed')}; (d) image-text {sh('vlm')}; (c) named families {sh('family')}) at "
      f"`{(pr.get('commit') or '?')[:10]}` (section 9). SPEC 1.5 keeps tokenizer inputs beyond SPEC 0.3's list "
      "out (slow files alone: a sentencepiece .model, vocab.txt, vocab.json + merges.txt; remote code without a "
      f"tiktoken file); they take {sh('b', 'out_of_scope')} of (b) ({sh('textgen', 'out_of_scope')} of text-gen, "
      f"{sh('embed', 'out_of_scope')} of embed) and count as misses here. With them out of the denominator (the maintainer's "
      f"call, not made yet) toks loads **{sh('b', 'loads_without_out_of_scope')} of (b)** (text-gen "
      f"{sh('textgen', 'loads_without_out_of_scope')}, embed {sh('embed', 'loads_without_out_of_scope')}). Section 8 "
      "lists every miss with what blocks it.")
    gm = sorted((r for r in by_id.values() if r.get("generic") and set(r["slices"]) & {"textgen", "embed"}),
                key=lambda r: -r["downloads"])
    w("")
    w(f"**On compiled fast paths (SPEC 1.4's gate: >= 99% of census downloads): {sh('b', 'fast')} of (b)** "
      f"({sh('b', 'fast_without_out_of_scope')} without SPEC 1.5's inputs; text-generation {sh('textgen', 'fast')}, "
      f"embedding / reranker {sh('embed', 'fast')}). The rest of what loads, {sh('b', 'generic')} of (b), runs its "
      "pre-tokenizer on the generic engine (docs/algorithms/generic.md: exact, onig's semantics for the census's "
      "constructs, no compiled template)" + (": " + ", ".join(f"`{esc(r['id'])}`" for r in gm[:12]) +
                                              (f" and {len(gm) - 12} more" if len(gm) > 12 else "") if gm else "") + ".")
    w("")
    crit = any(x.get("critical") for x in pk)
    w(("Step 0 of every order below is the critical targets (section 0): everything their own tokenizers still "
       "need. The" if crit else "Every critical target loads today (section 0), so no order needs a step 0. The") +
      " work packages still missing, added in this order from today, " +
      (f"reach **{fmt_pct(pk[n99]['cum_share'])} of (b) with package {n99}**" if n99 is not None
       else f"top out at {fmt_pct(pk[-1]['cum_share'])}" if pk else "add nothing") +
      ". Order: the package that unlocks the most downloads per unit of work first (a template, algorithm, "
      "normalizer, encode option or input format is one unit; a decoder, a post-processor mapping or a compiler "
      "spelling a quarter; a refusal of the load path in components other tokenizers already load (`code:`) one); a "
      "model counts only when every component its tokenizer uses is covered. (b) counts every download, so its order "
      "is led by the embedding models; the text-generation order follows this table.")
    w("")
    w("| # | package | adds | unlocks (b) | cumulative (b) | text-gen | embed | tokenizers (count) | (d) vlm | "
      "example models |")
    w("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    w(f"| today | toks `{(pr.get('commit') or '?')[:10]}` (loads) | | | {fmt_pct(T['covered_today_b'])} | "
      f"{fmt_pct(d['today_slices']['textgen'])} | {fmt_pct(d['today_slices']['embed'])} | "
      f"{fmt_pct(d['today_slices']['count'])} | {fmt_pct(d['today_slices']['vlm'])} | |")
    for i, x in enumerate(pk):
        if x["gain"] <= 0 and i > (n99 or 0) and not x.get("critical"):
            continue
        adds = ", ".join(f"`{f}`" for f in x["features"])
        name = f"**{esc(x['package'])}**" if n99 is None or i <= n99 else esc(x["package"])
        w(f"| {i} | {name} | {adds} | +{fmt_pct(x['share'])} | {fmt_pct(x['cum_share'])} | "
          f"{fmt_pct(x['cum_textgen'])} | {fmt_pct(x['cum_embed'])} | {fmt_pct(x['cum_count'])} | "
          f"{fmt_pct(x['cum_vlm'])} | {examples(by_id, x['models'])} |")
    w("")
    pt = d.get("packages_textgen") or []
    if pt:
        t99 = next((i for i, x in enumerate(pt) if x["cum_share"] >= 0.99), None)
        mt = [m for m in d["misses_b"] if "textgen" in m["slices"]]
        tg_ids = [m for m in by_id if "textgen" in by_id[m]["slices"]]
        tt = sum(by_id[m]["downloads"] for m in tg_ids) or 1
        k1 = next((i for i, x in enumerate(pk) if not x.get("critical")), None)
        prev = pk[k1 - 1]['cum_textgen'] if k1 else d['today_slices']['textgen']
        lead = (f"; the (b) order's first package, {esc(pk[k1]['package'])}, unlocks "
                f"+{fmt_pct(pk[k1]['share'])} of (b) but +{fmt_pct(pk[k1]['cum_textgen'] - prev)} "
                "of text-gen" if k1 is not None else "")
        w(f"**Text-generation order.** The same greedy over the same packages, weighted by the text-generation half "
          f"of (b) alone ({fmt_n(tt)} downloads, {len(tg_ids)} models){lead}. Today "
          f"{fmt_pct(d['today_slices']['textgen'])}; the packages " +
          (f"reach **{fmt_pct(pt[t99]['cum_share'])} of text-gen with package {t99}**" if t99 is not None
           else f"top out at {fmt_pct(pt[-1]['cum_share'])}") +
          f"; {len(mt)} unreadable text-gen models cap it at {fmt_pct(1 - sum(m['downloads'] for m in mt) / tt)}. "
          "Packages no text-generation model needs are left out; `census/coverage.json` `packages_textgen` has the "
          "rows.")
        w("")
        w("| # | package | adds | unlocks text-gen | cumulative text-gen | (d) vlm | (b) | example text-gen models |")
        w("|---:|---|---|---:|---:|---:|---:|---|")
        w(f"| today | toks `{(pr.get('commit') or '?')[:10]}` (loads) | | | {fmt_pct(d['today_slices']['textgen'])} | "
          f"{fmt_pct(d['today_slices']['vlm'])} | {fmt_pct(T['covered_today_b'])} | |")
        for i, x in enumerate(pt):
            if x["gain"] <= 0 and i > (t99 or 0) and not x.get("critical"):
                continue
            adds = ", ".join(f"`{f}`" for f in x["features"])
            name = f"**{esc(x['package'])}**" if t99 is None or i <= t99 else esc(x["package"])
            w(f"| {i} | {name} | {adds} | +{fmt_pct(x['share'])} | {fmt_pct(x['cum_share'])} | "
              f"{fmt_pct(x['cum_vlm'])} | {fmt_pct(x['cum_b'])} | {examples(by_id, x['models'])} |")
        w("")
    pa = [x for x in d.get("packages_alone") or [] if not x["package"].startswith("pattern ")]
    if pa:
        w("**Each package alone.** What one package unlocks from today's coverage with nothing else added (no step 0): "
          "the attribution the orders cannot show. A package whose tokenizers also need another package's features "
          "unlocks little alone, which is why the orders add them together; the one-off pattern templates are in "
          "`census/coverage.json` `packages_alone`.")
        w("")
        w("| package | features | (b) | text-gen | embed | (d) vlm |")
        w("|---|---|---:|---:|---:|---:|")
        for x in pa:
            w(f"| {esc(x['package'])} | {', '.join('`' + f + '`' for f in x['features'])} | +{fmt_pct(x['b'])} | "
              f"+{fmt_pct(x['textgen'])} | +{fmt_pct(x['embed'])} | +{fmt_pct(x['vlm'])} |")
        w("")
    w(f"Ceiling: {len(d['misses_b'])} unreadable models cap (b) at "
      f"{fmt_pct(1 - sum(m['downloads'] for m in d['misses_b']) / tb)}; without the slow-file package (outside "
      f"SPEC 0.3's input list today) the ceiling is {fmt_pct(T['ceiling_without_slow_inputs'])}, so **SPEC "
      f"1.1's >= 99% of (b) by downloads needs slow-file inputs in scope** (or those models out of the census). "
      "By count of distinct tokenizers (SPEC 1.1 asks for both) the tail is long: every one-off pattern is a "
      "tokenizer, so 99% by count needs the generic engine for certification while the fast-path gate (SPEC "
      "1.4) is by downloads.")
    w("")
    w("Feature names: `tmpl:*` a K3 scanner template (`tmpl:Pnn` a template for one pattern of section 3), `model:*` an "
      "algorithm, `bpe:*` a model flag, `norm:*` / `pretok:*` / `dec:*` / `pp:*` SPEC 1.3 components, `added:*` "
      "added-token options, `file:*` tokenizer.json fields hf applies inside `encode()`, `compile:*` spellings the "
      "compiler must accept as equal to a covered form (no new kernel), `input:*` a tokenizer source other than "
      "tokenizer.json. Section 5 defines each one and ranks it on its own.")
    w("")
    real = [x for x in d["order_without_testing_repos"] if not x.get("critical")]
    w(f"Robustness: the same order computed without CI / testing repos ({', '.join(TESTING_ORGS[:3])}, ...) starts "
      "with " + ", ".join(f"`{step_name(x['add'])}`" for x in real[:5]) +
      f"; (c)'s named families are {fmt_pct(pk[n99]['cum_family'] if n99 is not None else pk[-1]['cum_family'])} "
      f"covered by downloads after the packages that reach 99% (section 7 lists each family's needs).")
    w("")
    w("## 2. The same order, feature by feature")
    w("")
    w("The greedy behind section 1 at the finest grain: " +
      ("step 0 adds the critical targets' features, then " if any(x.get("critical") for x in steps) else "") +
      "each step adds the missing features of one tokenizer group (features that only ever occur together form one "
      "step), most downloads per unit of work first. " + (f"Step {s99} crosses 99%." if s99 is not None else ""))
    w("")
    w("| # | adds | unlocks (b) | cumulative (b) | text-gen | example models |")
    w("|---:|---|---:|---:|---:|---|")
    lim = max((s99 or 0) + 1, 13)
    for i, x in enumerate(steps[:lim]):
        label = f"**{esc(x['name'])}**: " if x.get("critical") else ""
        w(f"| {i} | {label}{', '.join('`' + f + '`' for f in x['add'])} | +{fmt_pct(x['share'])} | "
          f"{fmt_pct(x['cum_share'])} | {fmt_pct(x['cum_textgen'])} | {examples(by_id, x['models'], 2)} |")
    rest = steps[lim:]
    if rest:
        w("")
        w(f"{len(rest)} more steps cover the tail to {fmt_pct(rest[-1]['cum_share'])} (each <= "
          f"{fmt_pct(max(x['share'] for x in rest))}); `census/coverage.json` `order` has them all, and `order_textgen` "
          "the same greedy weighted by text-generation downloads.")
    w("")
    w("## 3. Pre-tokenizer patterns, ranked")
    w("")
    w("One row per distinct pre-tokenizer chain; the exact strings are in appendix A by id. status: `covered` = a "
      "kernels.md section 3 string in the spelling toks compiles; `compile` = covered semantics in another spelling "
      "the compiler must accept; `params` = the cl100k template's existing parameters express it; `new` = needs a new "
      "template (or the generic engine, which SPEC 1.4 does not count as a fast path).")
    w("")
    w("| id | template | variant | status | tokenizers | models (b) | share (b) | examples |")
    w("|---|---|---|---|---:|---:|---:|---|")
    for x in d["patterns"]:
        w(f"| {x['pid']} | {x['template']} | {esc(x['variant'] or chain_text(x['chain']))} | {x['status']} | "
          f"{len(x['tokenizers'])} | {x['models_b']} | {fmt_pct(x['share_b'])} | "
          f"{', '.join('`' + e + '`' for e in x['examples'][:3])} |")
    nochain = [t for t in toks.values() if t["chain"] is None]
    if nochain:
        w("")
        w("Without a readable pattern: " + ", ".join(f"`{t['id']}` ({t['kind']})" for t in nochain) + ".")
    w("")
    w("## 4. Templates, summed")
    w("")
    agg = collections.OrderedDict()
    for x in d["patterns"]:
        a = agg.setdefault(x["template"], {"n": 0, "dl": 0, "m": 0, "st": set(), "p": []})
        a["n"] += len(x["tokenizers"])
        a["dl"] += x["downloads_b"]
        a["m"] += x["models_b"]
        a["st"].add(x["status"])
        a["p"].append(x["pid"])
    w("| template | patterns | statuses | tokenizers | models (b) | share (b) |")
    w("|---|---|---|---:|---:|---:|")
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["dl"]):
        pl = ", ".join(a["p"][:8]) + (", ..." if len(a["p"]) > 8 else "")
        w(f"| {k} | {pl} | {', '.join(sorted(a['st']))} | {a['n']} | {a['m']} | {fmt_pct(a['dl'] / tb)} |")
    w("")
    w("## 5. Components, ranked")
    w("")
    w("Every feature some tokenizer needs beyond today's coverage, with the (b) share of the tokenizers using it (a "
      "tokenizer needs all of its features, so shares overlap; sections 1-2 give the order).")
    w("")
    w("| feature | category | what | tokenizers | models (b) | share (b) | examples |")
    w("|---|---|---|---:|---:|---:|---|")
    for f in d["features"]:
        w(f"| `{f['feature']}` | {f['category']} | {esc(f['what'])} | {f['tokenizers']} | {f['models_b']} | "
          f"{fmt_pct(f['share_b'])} | {', '.join('`' + e + '`' for e in f['examples'][:3])} |")
    w("")
    w("Components in use that today's coverage already handles (counts over all distinct tokenizers): " +
      "; ".join(f"{k} {v}" for k, v in d["covered_components_in_use"].items()) + ".")
    w("")
    w("## 6. Distinct tokenizers")
    w("")
    w(f"{len(d['tokenizers'])} distinct tokenizers across (b), (c), (d) and the critical-target repos ({T['b_tokenizers']} "
      "in (b); id = first 12 hex of the sha-256 of hf 0.23.2's normalized configuration). The 60 with the most (b) "
      "downloads; `census/coverage.json` `tokenizers` has every one with "
      "its full signature (model flags, normalizer / pre-tokenizer / decoder chains, post-processor, added-token "
      "option counts, truncation, padding).")
    w("")
    w("| tokenizer | algorithm | pattern | models | share (b) | needs beyond today | most downloaded model |")
    w("|---|---|---|---:|---:|---|---|")
    for t in list(d["tokenizers"].values())[:60]:
        w(f"| `{t['id']}` | {t['algorithm']} | {t.get('pid') or '-'} | {len(t['models'])} | "
          f"{fmt_pct(t['dl_b'] / tb)} | {esc(', '.join(t['features']) or 'nothing (covered)')} | "
          f"`{sorted(t['models'], key=lambda m: -by_id[m]['downloads'])[0]}` |")
    w("")
    w("## 7. Named families (SPEC 1.1(c))")
    w("")
    w("| family | repo | tokenizer | pattern | template / variant | needs beyond today |")
    w("|---|---|---|---|---|---|")
    for f in d["families"]:
        repo = f"`{f['repo']}`" + (f" (gated; byte-identical copy `{f['source']}`)" if f.get("source") else "")
        if f.get("tok"):
            w(f"| {f['family']} | {repo} | `{f['tok']}` | {f['pid']} | {f['template']} / {esc(f['variant'])} | "
              f"{esc(', '.join(f['missing']) or 'nothing (covered)')} |")
        elif f.get("standin"):
            x = f["standin"]
            w(f"| {f['family']} | `{f['repo']}`: gated, no byte-identical copy; stand-in `{x['repo']}` (not "
              f"verified identical) | `{x['hash']}` | - | {x['template']} / {esc(x['variant'])} | "
              f"{esc(', '.join(x['features']) or 'nothing (covered)')} |")
        else:
            w(f"| {f['family']} | {repo} | - | - | - | {esc(f.get('miss') or f.get('status'))} |")
    w("")
    tk = [f for f in d["families"] if f.get("tiktoken")]
    if tk:
        chk = (d.get("tiktoken") or {}).get("checks") or []
        same = [x["encoding"] for x in chk if x["equal"]]
        diff = [x["encoding"] for x in chk if x["equal"] is False]
        w("The openai encodings have no hf repo of their own: " + "; ".join(
            f"{f['tiktoken']} is represented by `{f['repo']}`'s tokenizer.json" for f in tk) +
          ". toks also takes the `.tiktoken` file with its pattern (SPEC 0.3). " +
          (f"Against tiktoken {d['tiktoken']['version']}'s own pattern strings (`tiktoken_ext/openai_public.py`, "
           "`coverage.py tiktoken`): " + (f"{', '.join(same)} equal the census string exactly" if same else "") +
           ("; " if same and diff else "") +
           (f"{', '.join(diff)} are spelled differently from their hub tokenizer.json, so a `.tiktoken` input "
            "with tiktoken's string needs its own exact-string entry (or a proof that both split alike)" if diff
            else "") + "." if chk else "The pattern strings were not checked against tiktoken (`coverage.py "
                                        "tiktoken`)."))
        w("")
        if chk:
            w("| encoding | census string (hub tokenizer.json) | tiktoken's string (json) |")
            w("|---|---|---|")
            for x in chk:
                w(f"| {x['encoding']} | {x['pid'] or '-'} (`{x['repo']}`) | " +
                  ("identical" if x["equal"] else "`" + esc_code(json.dumps(x["tiktoken"], ensure_ascii=False)) + "`")
                  + " |")
            w("")
    w("## 8. Misses")
    w("")
    ct = d.get("code_today") or {}
    cm = ct.get("misses_b") or []
    oos = [x for x in cm if x["out_of_scope"]]
    tt = (ct.get("shares") or {}).get("textgen", {}).get("downloads") or 1
    te = (ct.get("shares") or {}).get("embed", {}).get("downloads") or 1
    w(f"Every model in (b) that toks does not load today: {sum(len(x['models']) for x in cm)} models, "
      f"{fmt_pct(sum(x['share_b'] for x in cm))} of (b), grouped by what blocks them (the load path's own refusal: "
      "`toks_load`'s diag on the file, or on the model directory for a tiktoken model; else why no tokenizer could be "
      f"read), most downloads first. {sum(len(x['models']) for x in oos)} of them "
      f"({fmt_pct(sum(x['share_b'] for x in oos))} of (b)) ship an input SPEC 1.5 keeps out. `needs` = the "
      "census features of their tokenizers that no loading tokenizer uses yet (section 5), or `code:` the load "
      "path's refusal where every feature is already in use elsewhere. Shares of text-gen / embed are of those "
      "slices.")
    w("")
    w("| blocked by | (b) | text-gen | embed | needs | models (downloads) |")
    w("|---|---:|---:|---:|---|---|")
    for x in cm:
        needs = sorted({f for t in x["tokenizers"] for f in toks[t]["features"]})
        ms = ", ".join(f"`{m['id']}` ({fmt_n(m['downloads'])})" for m in x["models"])
        w(f"| {esc(x['block'])} | {fmt_pct(x['share_b'])} | {fmt_pct(x['textgen'] / tt)} | "
          f"{fmt_pct(x['embed'] / te)} | {esc(', '.join(needs)) or '-'} | {ms} |")
    w("")
    w("## 9. Findings for the rest of toks")
    w("")
    for line in d["findings"]:
        w(f"- {line}")
    w("")
    pr = d.get("probe") or {}
    if pr.get("rows"):
        w("")
        stg = pr.get("stages") or {}
        w(f"**The code today.** `tools/census/probe.c` runs `toks_load` (src/core/load.c: the config parse or the "
          f"tiktoken reader, compile, the split check, the algorithm's tables, the decode tables, the tier) at commit "
          f"`{(pr.get('commit') or '?')[:10]}` (library sources as of `{pr.get('library_commit') or '?'}`) on "
          f"{pr.get('host') or 'a lab host'}{' at ' + pr['date'] if pr.get('date') else ''} over every distinct "
          f"tokenizer.json ({pr.get('files')} files, the conversions of slow inputs included) and every distinct "
          f"tiktoken model directory ({pr.get('tiktoken_dirs')}: ranks file, tokenization_*.py and "
          "tokenizer_config.json under the repo's own names); each file also loads through `toks_load_mem_copy`, "
          "and its config parse alone names the stage" +
          (f" ({stg['8']} disagreements: stage 8)" if stg.get("8") else " (no disagreement)") +
          ". Slow-input conversions are in section 8.")
        w("")
        w("| load path result | tokenizers | share (b) |")
        w("|---|---:|---:|")
        for x in pr["rows"]:
            w(f"| {esc(x['result'])} | {x['tokenizers']} | {fmt_pct(x['share_b'])} |")
        w("")
    w("## 10. Method")
    w("")
    for line in METHOD:
        w(line)
    w("")
    w("## Appendix A: exact pre-tokenizer chains")
    w("")
    w("Each chain as hf 0.23.2 re-serializes it (json; `trim_offsets` dropped: it changes offsets, never ids). "
      "Regex strings are json string literals: the regex the engine sees is the decoded string.")
    w("")
    for x in d["patterns"]:
        w(f"**{x['pid']}** ({x['template']}, {x['status']}; {len(x['tokenizers'])} tokenizers, "
          f"{fmt_pct(x['share_b'])} of (b); e.g. `{x['examples'][0]}`)")
        w("")
        w("```json")
        for e in x["chain"]:
            w(json.dumps(e, ensure_ascii=False))
        w("```")
        w("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L).rstrip("\n") + "\n")
