#!/usr/bin/env python3
"""tests/model/spm_mutants.py: does run_spm_fuzz.py have teeth? Each mutant changes one rule of
docs/algorithms/spm_bpe.md inside spm_model.py (monkeypatched, never edited); the harness must report a
mismatch against hf 0.23.2 for every one of them. A surviving mutant means an untested rule.

    uv run --with tokenizers==0.23.2 tests/model/spm_mutants.py [--synthetic 300] [--n 300]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_spm_fuzz as R  # noqa: E402
import spm_model as S  # noqa: E402

ORIG = {}


def keep(obj, name):
    ORIG[(obj, name)] = getattr(obj, name)


def restore():
    for (obj, name), v in ORIG.items():
        setattr(obj, name, v)


# ---- mutants: (name, the doc rule it breaks, install) --------------------------------------------------
def m_unk_flush():
    f = ORIG[(S.Model, "symbols")]

    def symbols(self, piece):                               # flush a pending unk before byte-fallback ids
        out, unk = [], None
        for ch in piece:
            tid = self.vocab.get(ch)
            bids = [self.byte_id[b] for b in ch.encode("utf-8")] if tid is None and self.byte_fallback else None
            if tid is not None or (bids is not None and None not in bids):
                if unk is not None:
                    out.append(unk)
                    unk = None
                out.extend([tid] if tid is not None else bids)
                continue
            if self.unk_token is not None:
                if unk is not None and self.fuse_unk:
                    continue
                uid = self.vocab.get(self.unk_token)
                if uid is None:
                    raise S.HfError("UnkTokenOutOfVocabulary")
                if unk is not None:
                    out.append(unk)
                unk = uid
        if unk is not None:
            out.append(unk)
        return out
    S.Model.symbols = symbols
    return f


def m_no_fuse():
    f = ORIG[(S.Model, "symbols")]

    def symbols(self, piece):
        saved = self.fuse_unk
        self.fuse_unk = False
        try:
            return f(self, piece)
        finally:
            self.fuse_unk = saved
    S.Model.symbols = symbols


def m_ties_right():
    def merge(self, syms):
        mm, INF = self.merges, 1 << 62
        while len(syms) > 1:
            best, bi = INF, -1
            for i in range(len(syms) - 1):
                r = mm.get((syms[i], syms[i + 1]), (INF, 0))[0]
                if r <= best and r < INF:                    # ties to the RIGHTMOST
                    best, bi = r, i
            if bi < 0:
                break
            syms[bi:bi + 2] = [mm[(syms[bi], syms[bi + 1])][1]]
        return syms
    S.Model.merge = merge


def m_dup_first():
    init = ORIG[(S.Model, "__init__")]

    def __init__(self, m):
        init(self, m)
        seen = {}
        for rank, mg in enumerate(m.get("merges", [])):
            a, b = mg.split(" ") if isinstance(mg, str) else mg
            k = (self.vocab[a], self.vocab[b])
            if k not in seen:
                seen[k] = (rank, self.vocab[a + b])          # the FIRST duplicate wins
        self.merges = seen
    S.Model.__init__ = __init__


def m_ignore_merges_off():
    f = ORIG[(S.Model, "tokenize")]

    def tokenize(self, piece):
        saved = self.ignore_merges
        self.ignore_merges = False
        try:
            return f(self, piece)
        finally:
            self.ignore_merges = saved
    S.Model.tokenize = tokenize


def m_first_is_always():
    f = ORIG[(S, "pretokenize")]

    def pretokenize(ops, chars, orig):
        ops = [("metaspace", o[1], "always" if o[2] == "first" else o[2], o[3]) if o[0] == "metaspace" else o
               for o in ops]
        return f(ops, chars, orig)
    S.pretokenize = pretokenize


def m_prepend_checks():
    f = ORIG[(S, "normalize")]

    def normalize(ops, chars, orig):
        for op in ops:
            if op[0] == "prepend" and chars and "".join(chars).startswith(op[1]):
                continue                                     # Prepend that skips an existing prefix
            chars, orig = f([op], chars, orig)
        return chars, orig
    S.normalize = normalize


def m_split_after():
    f = ORIG[(S, "_split_spans")]

    def _split_spans(n, matches, behavior):
        if behavior == "merged_with_next":
            behavior = "merged_with_previous"
        return f(n, matches, behavior)
    S._split_spans = _split_spans


def m_bf_subparts():
    f = ORIG[(S, "decode_chain")]

    def decode_chain(ops, tokens):
        out = []
        for op in ops:
            if op[0] != "byte_fallback":
                tokens = f([op], tokens)
                continue
            out, run = [], []
            for t in tokens + [None]:
                v = S._byte_token(t) if t is not None else None
                if v is not None:
                    run.append(v)
                    continue
                if run:
                    out.append(bytes(run).decode("utf-8", "replace"))   # per maximal subpart
                    run = []
                if t is not None:
                    out.append(t)
            tokens = out
        return tokens
    S.decode_chain = decode_chain


def m_skip_by_content():
    def decode(self, ids, skip_special_tokens=False):
        toks = []
        for tid in ids:
            s = self.token_string(tid)
            a = self.added.get(tid)
            if s is None or (skip_special_tokens and a is not None and a.content in self.specials):
                continue
            toks.append(s)
        return " ".join(toks) if self.decoder is None else "".join(S.decode_chain(self.decoder, toks))
    S.SpmTokenizer.decode = decode


def m_match_content():
    f = ORIG[(S, "build_added")]

    def build_added(tj, model, norm_ops):
        by_id, specials, cache = f(tj, model, norm_ops)
        for a in by_id.values():
            a.match = a.content                              # normalized=true tokens by their raw content
        return by_id, specials, cache
    S.build_added = build_added


def m_keep_empty_token():
    def added_split(text, toks, drop, specials):            # the policy without the empty-span rule
        if not toks:
            return [("gap", 0, len(text))] if text else []
        by_first = {}
        for t in toks:
            by_first.setdefault(t.match[0], []).append(t)
        for k in by_first:
            by_first[k].sort(key=lambda t: -len(t.match))
        out, prev, n, s = [], 0, len(text), 0
        while s < n:
            m = next((t for t in by_first.get(text[s], []) if text.startswith(t.match, s)), None)
            if m is None:
                s += 1
                continue
            start, stop = s, s + len(m.match)
            s = stop
            if drop and m.content in specials:
                continue
            if m.single_word:
                w = S._word_set()
                if (start > 0 and ord(text[start - 1]) in w) or (stop < n and ord(text[stop]) in w):
                    continue
            if m.lstrip:
                j = start
                while j > 0 and text[j - 1] in S.WHITE_SPACE:
                    j -= 1
                start = max(j, prev)
            if m.rstrip:
                while stop < n and text[stop] in S.WHITE_SPACE:
                    stop += 1
            if prev < start:
                out.append(("gap", prev, start))
            out.append(("tok", start, max(start, stop), m.id))
            prev = max(start, stop)
        if prev < n:
            out.append(("gap", prev, n))
        return out
    S.added_split = added_split


def m_strip_all():
    f = ORIG[(S, "decode_chain")]

    def decode_chain(ops, tokens):
        ops = [("strip", o[1], 1 << 30, o[3]) if o[0] == "strip" else o for o in ops]
        return f(ops, tokens)
    S.decode_chain = decode_chain


def m_meta_dec_first_only():
    f = ORIG[(S, "decode_chain")]

    def decode_chain(ops, tokens):
        for op in ops:
            if op[0] == "metaspace":
                _, r, scheme = op
                new = []
                for i, t in enumerate(tokens):
                    if i == 0 and scheme != "never" and t.startswith(r):
                        t = t[1:]                            # only the leading replacement char dropped
                    new.append(t.replace(r, " "))
                tokens = new
            else:
                tokens = f([op], tokens)
        return tokens
    S.decode_chain = decode_chain


def m_unk_any_vocab():
    f = ORIG[(S.Model, "symbols")]

    def symbols(self, piece):                               # byte fallback per byte, unk only for missing bytes
        out = []
        for ch in piece:
            tid = self.vocab.get(ch)
            if tid is not None:
                out.append(tid)
                continue
            if self.byte_fallback:
                for b in ch.encode("utf-8"):
                    bid = self.byte_id[b]
                    if bid is not None:
                        out.append(bid)
                    elif self.unk_token is not None and self.unk_token in self.vocab:
                        out.append(self.vocab[self.unk_token])
                continue
            return f(self, piece)
        return out
    S.Model.symbols = symbols


def m_cut_everywhere():
    def words(self, piece):                                 # a cut between ANY two vocab chars
        if self.ignore_merges or not piece:
            return [piece] if piece else []
        out, start = [], 0
        for i in range(1, len(piece)):
            if piece[i - 1] in self.vocab and piece[i] in self.vocab:
                out.append(piece[start:i])
                start = i
        out.append(piece[start:])
        return out
    S.Model.words = words


def m_cut_before_repl():
    def words(self, piece):                                 # before every U+2581 not after U+2581 (no pair test)
        if self.ignore_merges or not piece:
            return [piece] if piece else []
        out, start = [], 0
        for i in range(1, len(piece)):
            if piece[i] == S.REPL and piece[i - 1] != S.REPL and piece[i - 1] in self.vocab:
                out.append(piece[start:i])
                start = i
        out.append(piece[start:])
        return out
    S.Model.words = words


MUTANTS = [
    ("unk-flush", "doc §5.2: a pending unk is not flushed before byte-fallback ids", m_unk_flush),
    ("no-fuse", "doc §5.2: fuse_unk fuses consecutive unks", m_no_fuse),
    ("unk-per-byte", "doc §5.2: byte fallback is all-or-nothing per char", m_unk_any_vocab),
    ("ties-right", "doc §5.3: equal ranks merge leftmost first", m_ties_right),
    ("dup-first", "doc §5.3: a duplicate merge pair keeps its last rank", m_dup_first),
    ("ignore-merges-off", "doc §5.1: ignore_merges answers whole vocab pieces", m_ignore_merges_off),
    ("first-is-always", "doc §4.1: first prepends only at original offset 0", m_first_is_always),
    ("prepend-checks", "doc §3: Prepend does not look at what is already there", m_prepend_checks),
    ("split-after", "doc §4.1: split=true cuts before every replacement char", m_split_after),
    ("bf-subparts", "doc §8.3: an invalid byte run decodes to one U+FFFD per byte", m_bf_subparts),
    ("skip-by-id", "doc §8.1: skip_special tests the token STRING (normalized form) against the special contents",
     m_skip_by_content),
    ("match-content", "doc §2.2: normalized=true tokens match by their normalized form", m_match_content),
    ("keep-empty-token", "doc §2.3: a token span emptied by lstrip vanishes (reversed: hf panics)", m_keep_empty_token),
    ("strip-all", "doc §8.4: Strip removes at most `start` leading chars", m_strip_all),
    ("cut-everywhere", "doc §5.5: a cut needs a pair no vocab string spans", m_cut_everywhere),
    ("cut-before-repl", "doc §5.5: gemma 3/4 span >▁ (the vocab string '>▁</'): no blanket cut before ▁", m_cut_before_repl),
    ("meta-dec-first", "doc §8.5: Metaspace decode drops EVERY replacement char of token 0", m_meta_dec_first_only),
]


def run_one(args):
    bad = 0
    for k in range(args.synthetic):
        c, _, _ = R.work_synth((k, 991 + k * 7919, args.synth_n))
        bad += sum(c.get(x, 0) for x in ("bad_encode", "bad_decode", "bad_pieces", "bad_cuts"))
    for name in args.real:
        R.init(name, R.F.fetch(name, R.F.pins()[name]))
        c, _, _ = R.work((424242, args.n))
        bad += sum(c.get(x, 0) for x in ("bad_encode", "bad_decode", "bad_pieces", "bad_cuts"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=300)
    ap.add_argument("--synth-n", type=int, default=40)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--real", nargs="*", default=["gemma4", "llama2", "mistral-v0.1"])
    args = ap.parse_args()
    for obj, name in [(S.Model, "symbols"), (S.Model, "merge"), (S.Model, "__init__"), (S.Model, "tokenize"),
                      (S, "pretokenize"), (S, "normalize"), (S, "_split_spans"), (S, "decode_chain"),
                      (S, "build_added"), (S, "added_split"), (S.SpmTokenizer, "decode"), (S.Model, "words")]:
        keep(obj, name)
    base = run_one(args)
    print(f"baseline (no mutant): {base} mismatches", flush=True)
    killed = 0
    for name, rule, install in MUTANTS:
        install()
        n = run_one(args)
        restore()
        killed += n > 0
        print(f"mutant {name:18} {'KILLED' if n else 'SURVIVED'} ({n} mismatches)  -- {rule}", flush=True)
    print(f"MUTANTS killed {killed}/{len(MUTANTS)}; baseline {base}")
    sys.exit(0 if base == 0 and killed == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
