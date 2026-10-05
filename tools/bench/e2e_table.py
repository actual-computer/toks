# /// script
# requires-python = ">=3.10"
# ///
"""tools/bench/e2e_table.py: tools/bench/e2e.sh logs -> docs/bench/e2e.md (the speed table), regenerated whole.

    uv run tools/bench/e2e_table.py <label>=<log> ... > docs/bench/e2e.md
    uv run tools/bench/e2e_table.py --ab <e2e_ab.sh log> ...      (the tier A/B table of tools/bench/e2e_ab.sh)

A label names one run: <host>-<tier>[-...] (e.g. tr9970x-avx2, tr9970x-scalar, m2ultra1-neon, gb10a-neon; <host> is a
chipset key, docs/machines.md). The same label given for several logs = the best of those runs per cell and state (the
same code repeated on a shared host). A label
starting with tokv1- names a tools/bench/tokv1.sh log: its cells make the incumbent section (toks vs tok v1).
Exactness, outside every timer: a toks cell is exact when its ids sha-256 equals the reference's (EXACT yes in its
own log, or, for a REF=0 run, the same ids sha as an exact cell of another log); otherwise it is VOID and in no
ratio. tiktoken / gigatoken count only where their ids equal the reference without the post-processor; elsewhere
they are n/a with the reason. Comparator times: the fastest of the same host over all logs listed."""
import re
import sys

STATES = ("cold", "pass", "warm")


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", line))


def secs(d, st):
    """a tool line's best seconds for state st; lang only where it ran (lang_s > 0), else None"""
    v = d.get(st + "_s")
    return float(v) if v not in (None, "na") and (st not in ("lang", "warmo", "coldo") or float(v) > 0) else None


def parse(path):
    head, cells, cur = {}, [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("== "):
            tk, corp, _, ch = line[3:].split()
            cur = {"tk": tk, "corp": corp, "chunk": int(ch), "load": []}
            cells.append(cur)
            continue
        tag = line.split(" ", 1)[0]
        if cur is None:
            if tag in ("HOST", "CPU", "OS", "GOVERNOR", "THP", "PIN", "GIT", "CC", "KERNELS", "GIGATOKEN"):
                head[tag] = line[len(tag) + 1:]
            elif tag in ("BIN", "TOKENIZER", "CORPUS", "UPTIME"):
                head.setdefault(tag, []).append(line[len(tag) + 1:])
            continue
        if tag == "LOAD":
            cur["load"].append(line.split()[2])
        elif tag == "BUSY":
            cur["busy"] = line[5:].strip()
        elif tag == "E2E":
            cur["toks"] = kv(line)
        elif tag == "GIGA":
            cur["giga"] = kv(line)
        elif tag == "REF":
            d = kv(line)
            cur[d["tool"]] = d
        elif tag == "EXACT":
            cur["exact"] = line.split()[1]
        elif tag == "FIRSTDIFF":
            cur["firstdiff"] = line.split(" ", 1)[1]
        elif tag == "UPTIME":
            head.setdefault("UPTIME", []).append(line[7:])
    return head, cells


def f(x, d=1):
    return f"{x:.{d}f}"


def ratio(a, b):
    return f"{a / b:.2f}x" if a and b else "-"


def chs(c):
    return str(c) if c else "whole"


class Cell:
    """one (run, tokenizer, corpus, chunk): MB/s per tool and state, the comparators' validity."""

    def __init__(self, run, host, c):
        t = c["toks"]
        self.run, self.host, self.tk, self.corp, self.chunk = run, host, c["tk"], c["corp"], c["chunk"]
        self.n = int(t["bytes"])
        self.tier = t["tier"]
        self.ids, self.calls, self.sha = int(t["ids"]), int(t["calls"]), t["sha"]
        self.s = {st: float(t[st + "_s"]) for st in STATES}
        for extra in ("lang", "warmo", "coldo"):    # logs from before these states have none
            v = secs(t, extra)
            if v:
                self.s[extra] = v
        self.floor = float(t["floor_s"])
        self.ghz = float(t["ghz0"])
        self.pass_after = t.get("pass_after", "same")   # logs before b3c92da: pass right after cold (pass-same)
        self.exact = c.get("exact")
        self.load = c["load"]
        self.busy = c.get("busy", "")
        self.raw = c

    def mbs(self, st):
        return self.n / self.s[st] / 1e6


def pass_name(after):
    return "pass" if after == "other" else "pass-same"


def comparators(c):
    """(hf s, tiktoken s or None, its n/a reason, gigatoken {state: s} or None, its n/a reason) of one raw cell"""
    hf = c.get("hf")
    hf_s = float(hf["s"]) if hf and "s" in hf else None
    tk = c.get("tiktoken")
    tk_s, tk_why = None, "-"
    if tk:
        if tk.get("same_as_hf_nopp") == "yes":
            tk_s = float(tk["s"])
        else:
            tk_why = "n/a (ids differ)" if "same_as_hf_nopp" in tk else "n/a"
    g, gr = c.get("giga"), c.get("gigatoken")
    g_s, g_why = None, "-"
    if g and "na" in g:
        g_why = "n/a (" + g["na"].split(":")[0].replace("_", " ") + ")"
    elif g and gr:
        if gr.get("same_as_hf_nopp") == "yes":
            g_s = {st: secs(g, st) for st in STATES + ("lang", "warmo", "coldo") if secs(g, st)}
            if g.get("pass_after", "same") != c.get("toks", {}).get("pass_after", "same"):
                g_s.pop("pass", None)                # the two passes are different states: no ratio
        else:
            g_why = "n/a (ids differ)"
    elif g:
        g_why = "unchecked"
    return hf_s, tk_s, tk_why, g_s, g_why


def parse_tokv1(path):
    """a tools/bench/tokv1.sh log: header + cells {tk, corp, chunk, load, busy, tokv1: TOKV1 kv, tv1: {(state, side): kv}}"""
    head, cells, cur = {}, [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("== "):
            tk, corp, _, ch = line[3:].split()
            cur = {"tk": tk, "corp": corp, "chunk": ch, "load": [], "tv1": {}}
            cells.append(cur)
            continue
        tag = line.split(" ", 1)[0]
        if cur is None:
            if tag in ("HOST", "CPU", "OS", "GOVERNOR", "THP", "PIN", "GIT", "EREV", "CC", "AS", "KERNELS"):
                head[tag] = line[len(tag) + 1:]
            elif tag in ("BIN", "TOKENIZER", "CORPUS", "UPTIME"):
                head.setdefault(tag, []).append(line[len(tag) + 1:])
        elif tag == "LOAD":
            cur["load"].append(line.split()[2])
        elif tag == "BUSY":
            cur["busy"] = line[5:].strip()
        elif tag == "TOKV1":
            cur["tokv1"] = kv(line)
        elif tag == "CLOCK":
            cur["clock"] = kv(line)
        elif tag == "TV1":
            d = kv(line)
            cur["tv1"][(d["state"], d["side"])] = d
        elif tag == "UPTIME":
            head.setdefault("UPTIME", []).append(line[7:])
    return head, cells


def incumbent(tlogs, w):
    """the incumbent section: toks vs tok v1 per cell (tools/bench/tokv1.sh logs)"""
    w("## Incumbent: tok v1\n")
    w("The incumbent bar: every cell tok v1 supports must be faster in toks. tok v1 = the incumbent tokenizer at the commit")
    w("in the receipts (x86-64 asm with avx-512; arm64 asm with sve2 + neon), its image packed by its own tool from the")
    w("same tokenizer.json (qwen38: image version 3; glm53, nemotron3-omni: version 4). Its source is not public:")
    w("`tools/bench/tokv1.sh` (E_DIR = a checkout of it: an optional bench input, never a build or test dependency) runs")
    w("`tools/bench/tokv1.c`: one binary, one pinned core, both libraries on the same chunks, the sides alternating from")
    w("rep to rep (paired), every call's ids compared, fresh and warm, outside the timers (a difference voids the cell).")
    w("MB/s = the median of reps; **x** = the median of the per-rep ratios tok v1 time / toks time with its 95% bootstrap")
    w("interval: > 1 = toks faster. **toks** = 2 MiB piece caches, memo off (flags 0 before 0.3.0,")
    w("`TOKS_SCRATCH_MEMO_MIB(0)` since), **toks32** = `TOKS_SCRATCH_CACHE_MIB(32)`, memo off (a long-lived worker")
    w("scratch), **toksm** = `TOKS_SCRATCH_MEMO_MIB(4)` (toks's 4 MiB segment memo: the incumbent's flags, and toks'")
    w("default since 0.3.0). States: **cold** and **pass** as above; **lang** =")
    w("initialized, warmed on the other corpora (never a measured file), then timed: the incumbent's \"new prompt in a warm worker\",")
    w("tok v1's own headline state; **warm** = the same text again, where tok v1's and toksm's 4 MiB segment memos")
    w("answer every chunk they recorded (**memo** = the share of tok v1's pieces it skipped): a replay, not a")
    w("piece-cache race. **chat** = one conversation of 24 turns")
    w("(1-4 KiB of the en text each, qwen's chat markup) encoded as its 24 growing prompts in order on one scratch")
    w("(the affinity case): from the second prompt on, tok v1's memo answers every earlier")
    w("turn in the pass state too (toksm's likewise), where the markup is added tokens (qwen38, nemotron3-omni: memo")
    w("0.90); on glm53 it is")
    w("plain text (one segment per prompt), so there it is not a replay. **zh** = wiki-zh + the Gutenberg Chinese novel")
    w("(part of cjk). nemotron3-omni is the public stand-in for the incumbent's Nemotron tokenizer.\n")
    gate = []
    for label, path in tlogs:
        head, cells = parse_tokv1(path)
        host = head.get("HOST", "?").split()[0]
        pin = head.get("PIN", "")
        tier = next((c["tokv1"].get("tier") for c in cells if "tokv1" in c), "?")
        tname = {"1": "scalar", "2": "neon", "3": "avx2", "4": "avx512"}.get(tier, tier)
        w(f"**{label}** ({host}; toks tier {tname} vs tok v1 {'avx-512' if 'x86_64' in head.get('HOST', '') else 'sve2 + neon'};"
          f" {pin.split(' TOKS_TIER')[0]}; e {head.get('EREV', '?').split()[0]}; toks {head.get('GIT', '?')[:12]})\n")
        w("| tokenizer | corpus | chunk | cold tok v1 / toks / toksm | x cold (toks; toksm) | pass tok v1 / toks / toksm |"
          " x pass (toks; toksm) | lang tok v1 / toks / toks32 / toksm | x lang (toks; toks32; toksm) |"
          " warm tok v1 / toks / toks32 / toksm (memo) | x warm (toks32; toksm) | exact | load | busy |")
        w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        for c in cells:
            t, d = c.get("tokv1"), c["tv1"]
            ch = {"0": "whole", "replay": "24 prompts"}.get(c["chunk"], c["chunk"])
            if not t or t.get("exact") != "yes" or ("cold", "toks") not in d:
                w(f"| {c['tk']} | {c['corp']} | {ch} | VOID | | | | | | | | {t.get('exact', 'failed') if t else 'failed'} | | |")
                continue

            def mb(st, side):
                return f(float(d[(st, side)]["mbs_med"])) if (st, side) in d else "-"

            def xs(st, side):
                e = d.get((st, side))
                return f"{float(e['x_med']):.2f} [{float(e['x_lo']):.2f}, {float(e['x_hi']):.2f}]" if e else "-"
            memo = d.get(("warm", "tokv1"), {}).get("memo_frac", "0")
            pmemo = float(d.get(("pass", "tokv1"), {}).get("memo_frac", "0"))
            w("| " + " | ".join([
                c["tk"], c["corp"], ch,
                f"{mb('cold', 'tokv1')} / {mb('cold', 'toks')} / {mb('cold', 'toksm')}",
                f"{xs('cold', 'toks')}; {xs('cold', 'toksm')}",
                f"{mb('pass', 'tokv1')} / {mb('pass', 'toks')} / {mb('pass', 'toksm')}" +
                (f" (memo {pmemo:.2f})" if pmemo > 0.01 else ""),
                f"{xs('pass', 'toks')}; {xs('pass', 'toksm')}",
                f"{mb('lang', 'tokv1')} / {mb('lang', 'toks')} / {mb('lang', 'toks32')} / {mb('lang', 'toksm')}",
                f"{xs('lang', 'toks')}; {xs('lang', 'toks32')}; {xs('lang', 'toksm')}",
                f"{mb('warm', 'tokv1')} / {mb('warm', 'toks')} / {mb('warm', 'toks32')} / {mb('warm', 'toksm')}"
                f" ({float(memo):.2f})",
                f"{xs('warm', 'toks32')}; {xs('warm', 'toksm')}",
                "yes (" + t["ids"] + " ids)", "→".join(c["load"][:1] + c["load"][-1:]), c.get("busy", "")]) + " |")
            for st in ("cold", "pass", "lang", "warm"):
                for side in ("toks", "toksm"):
                    e = d.get((st, side))
                    replay = c["chunk"] == "replay" or st == "warm"
                    if e and (side == "toksm" or not replay):
                        gate.append((label, c["tk"], c["corp"], ch, st, float(e["x_med"]), float(e["x_lo"]), side,
                                     replay))
        w("")
    for name, sel in (("toks default vs tok v1, cold / pass / lang of every chunked and whole cell",
                       lambda g: g[7] == "toks"),
                      ("toksm (4 MiB memo) vs tok v1, the same new-prompt cells", lambda g: g[7] == "toksm" and not g[8]),
                      ("toksm vs tok v1, the replays: warm of every cell + the chat cells", lambda g: g[7] == "toksm" and g[8])):
        gs = [g for g in gate if sel(g)]
        if not gs:
            continue
        slow = [g for g in gs if g[5] < 1.0]
        unsure = [g for g in gs if g[5] >= 1.0 and g[6] <= 1.0]
        w(f"**Incumbent gate** ({name}): toks faster in {len(gs) - len(slow)} / {len(gs)} (median x > 1),"
          f" {len(gs) - len(slow) - len(unsure)} with the interval's lower bound > 1" +
          (f"; SLOWER in: {', '.join(f'{g[0]} {g[1]} {g[2]} {g[3]} {g[4]} ({g[5]:.2f}x)' for g in slow)}" if slow else "") +
          ".\n")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--ab":
        print(ab_table(args[1:]))
        return
    logs = [a.split("=", 1) for a in args]
    tlogs = [x for x in logs if x[0].startswith("tokv1-")]
    logs = [x for x in logs if not x[0].startswith("tokv1-")]
    runs, heads = {}, {}
    exact_sha = {}                               # (tk, corp, chunk) -> ids sha of an exact cell
    comp = {}                                    # (host, tk, corp, chunk) -> the fastest comparator times
    for label, path in logs:
        head, cells = parse(path)
        heads.setdefault(label, []).append((path, head, cells))
        host = head.get("HOST", "?").split()[0]
        for c in cells:
            if "toks" in c and c.get("exact") == "yes":
                exact_sha[(c["tk"], c["corp"], c["chunk"])] = c["toks"]["sha"]
            hf_s, tk_s, tk_why, g_s, g_why = comparators(c)
            k = (host, c["tk"], c["corp"], c["chunk"])
            d = comp.setdefault(k, {"hf": None, "tk": None, "tk_why": "-", "g": None, "g_why": "-"})
            if hf_s and (d["hf"] is None or hf_s < d["hf"]):
                d["hf"] = hf_s
            if tk_s and (d["tk"] is None or tk_s < d["tk"]):
                d["tk"] = tk_s
            if d["tk"] is None and tk_why != "-":
                d["tk_why"] = tk_why
            if g_s:
                old = d["g"] or {}
                d["g"] = {st: min(v for v in (old.get(st), g_s.get(st)) if v is not None) for st in {**old, **g_s}}
            if d["g"] is None and g_why != "-":
                d["g_why"] = g_why
    for label, parsed in heads.items():
        bycell = {}
        for path, head, cells in parsed:
            host = head.get("HOST", "?").split()[0]
            for c in cells:
                if "toks" not in c:
                    continue
                x = Cell(label, host, c)
                if x.exact != "yes":
                    x.exact = "yes" if exact_sha.get((x.tk, x.corp, x.chunk)) == x.sha else (x.exact or "unchecked")
                k = (x.tk, x.corp, x.chunk)
                o = bycell.get(k)
                if o is None:
                    bycell[k] = x
                else:                            # best of runs per state
                    for st, v in x.s.items():
                        if v < o.s.get(st, float("inf")):
                            o.s[st] = v
                    if x.exact != "yes":
                        o.exact = x.exact
                    o.load = o.load + x.load
        runs[label] = list(bycell.values())

    out = []
    w = out.append
    w("# toks end-to-end speed table\n")
    w("Generated by `tools/bench/e2e_table.py` from the raw logs in `docs/bench/raw/` (`tools/bench/e2e.sh` on each host;")
    w("never edit the numbers by hand). One thread, single process, MB/s = decimal megabytes of input per second.")
    w("host names are chipset keys (see docs/machines.md); home directories were replaced by $HOME in the receipts.\n")
    w("- **cells**: tokenizer x corpus x chunk. Corpora (`tools/bench/corpus.sha256`): **en** two Gutenberg novels (2.05 MB),")
    w("  **code** cpython + toks C sources (0.61 MB), **ml** wikipedia ar/de/el/fr/he/hi/ka/ru/ta/th/uk/vi + Gutenberg de/es/fr")
    w("  (6.6 MB), **cjk** wikipedia zh/ja/ko + a Gutenberg Chinese novel (3.1 MB). Chunk 4096 = each call gets the text")
    w("  up to the first newline at or past the next 4 KiB boundary; whole = the corpus in one call.")
    w("- **toks states** (`tools/bench/e2e.c`; cpu-cache residency is declared with each): **cold** = the")
    w("  scratch (dynamic piece cache) initialized before EVERY call, outside the timer, each call timed alone; logs from")
    w("  commit b3c92da on run the cold reps first, back to back (each cold pass starts with this text's lines in the cpu caches).")
    w("  Cold reps are cpu-cache-hot (back to back on one text); pass after other text is the serving state (a")
    w("  measure on gb10c cpu 8, master d11f9d1: coldo, each cold rep after an untimed pass over the other corpora, is 11-40%")
    w("  slower on code 4096 with identical K5 counters: llama3 425.0 -> 292.4, o200k 460.1 -> 277.3, qwen38 338.5 -> 259.8,")
    w("  gpt2 310.0 -> 277.1 MB/s; the loss scales with the static words table, 2 MiB gpt2 .. 8 MiB o200k, against the")
    w("  X925's 2 MiB L2). **coldo** (logs that have it): those cold reps, both tools. **warmo** (logs that have it): init,")
    w("  an untimed pass over the text, one over the other corpora, the timed pass over the text, no init between (the")
    w("  serving replay; not a same-language warm state, which is memo off on a disjoint corpus).")
    w("  The 5f71528 logs had no lang state, so their cold also followed a pass over the same text (the previous rep's")
    w("  warm); runs that did have one (defaults.sh's E2E_WARM_ON) ran cold right after the lang pass's sweep, which cost")
    w("  cold 6-15% (gb10c llama3 en 4096: 241.8 vs 256.9 MB/s at flags 0, 213.9 vs 250.6 at CACHE_MIB(16)):")
    w("  those cold numbers are not comparable with these. **pass** = one scratch over the")
    w("  corpus after an untimed pass over OTHER text (every bench corpus file outside the measured corpus, cut at 4096")
    w("  bytes whatever the cell's chunk), the scratch")
    w("  re-initialized before the timed pass: an empty piece cache filling as the pass goes, the cpu caches holding the")
    w("  other text's lines; **pass-same** (the pass of the logs at 5f71528 and earlier) = the")
    w("  same pass right after the cold pass over the same text, its lines still in the caches (llama3 4096 gb10a: code")
    w("  1.37 ms pass-same vs 1.82 ms pass). **warm** = the pass right after the pass: an exact replay from each tool's")
    w("  default cache budget as measured (toks: a 2 MiB piece cache, and from 0.3.0 the 4 MiB segment memo; gigatoken: 512")
    w("  MiB per state, its unit memo for")
    w("  sentencepiece), so warm vs warm is an unmatched comparison; **lang-x** (logs that have it) = the timed pass")
    w("  after an untimed pass over the other corpora on the same scratch (tools/bench/tokv1.sh's rule; cross-language:")
    w("  not SPEC §12.3's same-language warm-lang bar). toks runs hf's default call (added tokens")
    w("  recognized, post-processor applied). Best of reps per state.")
    w("- **comparators** on exactly the same chunks: **hf** tokenizers 0.23.2 `Tokenizer.encode` (Kimi K3: its own")
    w("  reference, transformers' TikTokenTokenizer, serving path); **tiktoken** 0.14.0 `encode_ordinary` over an Encoding")
    w("  built from the file's vocab and Split pattern (byte-level files only); **gigatoken** (the bar to beat; pinned")
    w("  source, its Rust API, `tools/bench/gigatoken`) in the same states: cold = a fresh state (fork / new")
    w("  EncodeState) before every call, cold reps first; pass = a state encodes the other text untimed, then a fresh")
    w("  state times the pass; warm = replay; lang-x = a fresh state warmed on the other text, then timed. hf and")
    w("  tiktoken run as they ship (their own caches warm, Python call overhead included). All times are best of reps.")
    w("  Kimi K3's gigatoken column: gigatoken-bench loads the model with `load_tiktoken(.., PretokenizerType::Kimi, [])`,")
    w("  without the wrapper's cuts (400,000-char chunks, 25,000-char same-class runs) and without specials, so its ids")
    w("  can differ from the reference on texts over 400,000 chars or with a 25,000-char run: the kimik3 en / code / ml")
    w("  whole cells are \"n/a (ids differ)\" for that reason (tiktoken's `encode_ordinary` likewise), not a toks defect;")
    w("  cjk whole happens to agree; a 4 KiB chunk holds no cut, so the 4096 cells are exact.")
    w("- **exactness**, outside every timer: toks ids == the reference's ids (sha-256 of the whole id stream), else the")
    w("  cell is VOID and in no ratio; tiktoken and gigatoken count only where their ids equal hf's without the")
    w("  post-processor (else n/a: \"ids differ\", or plain n/a where the file is not one byte-level Split pattern, e.g.")
    w("  gemma 4's sentencepiece-style bpe). Ratios use the fastest comparator time of the same host.")
    w("- **receipts** (end of this file): host, cpu, os, pin + governor, commit, compiler, kernels, binary / tokenizer /")
    w("  corpus sha-256, load (1 min) before -> after every cell and, on linux, the busy share of the pinned cpu and its")
    w("  smt sibling over the cell. Not a certified cell: no abba pairs, no bootstrap intervals.\n")
    w("Reproduce: `tools/release/rc.sh` (all hosts) or, on one host, `sh tools/bench/gigatoken.sh` then")
    w("`PINCPU=12 GIT_SHA=<sha> tools/bench/e2e.sh \"taskset -c 12\" > log`, then this script.\n")

    # ---- headline -------------------------------------------------------------------------------------
    for chunk, title in ((4096, "4 KiB chunks"), (0, "whole corpus in one call")):
        w(f"## Headline: {title}\n")
        w("MB/s. toks / x = toks MB/s over the comparator's (cold vs hf and tiktoken; cold vs cold and warm vs warm for gigatoken).\n")
        for label, cells in runs.items():
            sel = [x for x in cells if x.chunk == chunk]
            if not sel:
                continue
            x0 = sel[0]
            w(f"**{label}** ({x0.host}, tier {x0.tier})\n")
            w("| tokenizer | corpus | toks cold | toks warm | hf | tiktoken | gigatoken cold | gigatoken warm | toks / hf | toks / tiktoken | toks / gigatoken cold | toks / gigatoken warm | load |")
            w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for x in sel:
                d = comp.get((x.host, x.tk, x.corp, x.chunk), {})
                void = x.exact != "yes"
                hf_m = x.n / d["hf"] / 1e6 if d.get("hf") else None
                tk_m = x.n / d["tk"] / 1e6 if d.get("tk") else None
                g = d.get("g")
                gc = x.n / g["cold"] / 1e6 if g else None
                gw = x.n / g["warm"] / 1e6 if g else None
                cold, warm = x.mbs("cold"), x.mbs("warm")
                w("| " + " | ".join([
                    x.tk, x.corp,
                    ("VOID " if void else "") + f(cold), f(warm),
                    f(hf_m, 2) if hf_m else "-",
                    f(tk_m) if tk_m else d.get("tk_why", "-"),
                    f(gc) if gc else d.get("g_why", "-"), f(gw) if gw else "-",
                    "-" if void else ratio(cold, hf_m), "-" if void else ratio(cold, tk_m),
                    "-" if void else ratio(cold, gc), "-" if void else ratio(warm, gw),
                    "→".join(x.load)]) + " |")   # every run's before → after: a re-measured cell shows both
            w("")

    # ---- gates ----------------------------------------------------------------------------------------
    w("## Gates\n")
    w("spm (gemma4) warm cells without the memo (flags 0 before 0.3.0: 2 MiB, no long region): no word-cache change moves them")
    w("(both tools measured in one session).\n")
    for label, cells in runs.items():
        voids = [x for x in cells if x.exact != "yes"]
        after = cells[0].pass_after if cells else "same"
        gstates = (("cold", "cold", "cold vs cold"),
                   ("coldo", "coldo", "coldo vs coldo: each cold rep after the other corpora (the cold reps above are cpu-cache-hot)"),
                   ("pass", pass_name(after), "pass vs pass" + ("" if after == "other" else ", each right after its own cold pass over the same text")),
                   ("warm", "warm", "warm vs warm, each tool's default cache: unmatched"),
                   ("lang", "lang-x", "lang-x vs lang-x, both warmed on the other corpora"),
                   ("warmo", "warmo", "warmo vs warmo: the 2nd sight of the text after the other corpora, no init between"))
        lose_tk, n_tk = [], 0
        lose_g, n_g = {st: [] for st, _, _ in gstates}, {st: 0 for st, _, _ in gstates}
        for x in cells:
            if x.exact != "yes":
                continue
            d = comp.get((x.host, x.tk, x.corp, x.chunk), {})
            if d.get("tk"):
                n_tk += 1
                if x.mbs("cold") <= x.n / d["tk"] / 1e6:
                    lose_tk.append(f"{x.tk} {x.corp} {chs(x.chunk)}")
            for st, _, _ in gstates:             # >=: a tie counts as a loss
                if d.get("g") and st in d["g"] and st in x.s:
                    n_g[st] += 1
                    if x.s[st] >= d["g"][st]:
                        lose_g[st].append(f"{x.tk} {x.corp} {chs(x.chunk)} ({x.mbs(st):.0f} vs {x.n / d['g'][st] / 1e6:.0f})")
        w(f"**{label}**: {len(cells)} cells, {len(cells) - len(voids)} exact"
          + (f", VOID: {', '.join(f'{x.tk} {x.corp} {chs(x.chunk)}' for x in voids)}" if voids else ""))
        w(f"- faster than tiktoken (toks cold vs tiktoken): {n_tk - len(lose_tk)} / {n_tk} cells"
          + (f"; NOT in: {', '.join(lose_tk)}" if lose_tk else ""))
        for st, name, what in gstates:
            if n_g[st]:
                w(f"- faster than gigatoken {name} ({what}): {n_g[st] - len(lose_g[st])} / {n_g[st]} cells"
                  + (f"; NOT in: {', '.join(lose_g[st])}" if lose_g[st] else ""))
        w("")
    # asm tier vs the c twin on the same host (rent rule: no cell slower than the c twin)
    pairs = []
    for la, ca in runs.items():
        for lb, cb in runs.items():
            if la == lb or not ca or not cb or ca[0].host != cb[0].host or cb[0].tier != "scalar" or ca[0].tier == "scalar":
                continue
            twin = {(x.tk, x.corp, x.chunk): x for x in cb}
            rows = []
            for x in ca:
                y = twin.get((x.tk, x.corp, x.chunk))
                if y is None:
                    continue
                rows.append((x.s["cold"] and y.s["cold"] / x.s["cold"], y.s["pass"] / x.s["pass"], x, y))
            if rows:
                pairs.append((la, lb, rows))
    for la, lb, rows in pairs:
        slow = [r for r in rows if r[0] < 1.0 or r[1] < 1.0]
        w(f"**asm vs c twin, {la} / {lb}** (same host, same cells; toks MB/s ratio, cold and pass): {len(rows) - len(slow)} / {len(rows)} cells faster in both states"
          + (": slower in " + ", ".join(f"{x.tk} {x.corp} {chs(x.chunk)} (cold {rc:.2f}x, pass {rp:.2f}x)" for rc, rp, x, _ in slow) if slow else "")
          + f"; median cold {sorted(r[0] for r in rows)[len(rows) // 2]:.2f}x\n")

    if tlogs:
        incumbent(tlogs, w)

    # ---- all cells ------------------------------------------------------------------------------------
    w("## All cells\n")
    afters = {x.pass_after for cells in runs.values() for x in cells}
    pn = pass_name(afters.pop()) if len(afters) == 1 else "pass / pass-same (per run)"
    w(f"| run | tokenizer | corpus | chunk | tier | toks cold | {pn} | warm | cold Mtok/s | cold ns/call | hf | tiktoken | gigatoken cold / {pn} / warm | x_floor ({pn}) | exact | GHz | load 1m | busy |")
    w("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|")
    for label, cells in runs.items():
        for x in cells:
            d = comp.get((x.host, x.tk, x.corp, x.chunk), {})
            g = d.get("g")
            w("| " + " | ".join([
                label, x.tk, x.corp, chs(x.chunk), x.tier,
                f(x.mbs("cold")), f(x.mbs("pass")), f(x.mbs("warm")),
                f(x.ids / x.s["cold"] / 1e6, 2), f(x.s["cold"] / x.calls * 1e9, 0),
                f(x.n / d["hf"] / 1e6, 2) if d.get("hf") else "-",
                f(x.n / d["tk"] / 1e6) if d.get("tk") else d.get("tk_why", "-"),
                " / ".join(f(x.n / g[st] / 1e6) if st in g else "-" for st in STATES) if g else d.get("g_why", "-"),
                f(x.s["pass"] / x.floor, 0), x.exact, f(x.ghz, 2), "→".join(x.load), x.busy]) + " |")
    w("")

    # ---- receipts -------------------------------------------------------------------------------------
    w("## Receipts\n")
    for label, parsed in heads.items():
        for path, head, cells in parsed:
            w(f"### {label}: `{path.split('/')[-1]}`\n")
            for k in ("HOST", "CPU", "OS", "GOVERNOR", "THP", "PIN", "GIT", "CC", "KERNELS", "GIGATOKEN"):
                if k in head:
                    w(f"- {k.lower()}: `{head[k].strip()}`")
            for k in ("BIN", "TOKENIZER", "CORPUS", "UPTIME"):
                for v in head.get(k, []):
                    w(f"- {k.lower()}: `{v.strip()}`")
            loads = [float(x) for c in cells for x in c["load"]]
            if loads:
                w(f"- load (1 min) over the cells: {min(loads):.2f} .. {max(loads):.2f}")
            diffs = [f"{c['tk']} {c['corp']} {chs(c['chunk'])}: {c['firstdiff']}" for c in cells
                     if c.get("firstdiff", "none") != "none"]
            for dline in diffs:
                w(f"- first diff: `{dline}`")
            w("")
    for label, path in tlogs:
        head, cells = parse_tokv1(path)
        w(f"### {label}: `{path.split('/')[-1]}`\n")
        for k in ("HOST", "CPU", "OS", "GOVERNOR", "THP", "PIN", "GIT", "EREV", "CC", "AS", "KERNELS"):
            if k in head:
                w(f"- {k.lower()}: `{head[k].strip()}`")
        for k in ("BIN", "TOKENIZER", "CORPUS", "UPTIME"):
            for v in head.get(k, []):
                w(f"- {k.lower()}: `{v.strip()}`")
        loads = [float(x) for c in cells for x in c["load"]]
        if loads:
            w(f"- load (1 min) over the cells: {min(loads):.2f} .. {max(loads):.2f}")
        clk = [float(c["clock"][k]) for c in cells if "clock" in c for k in ("ghz0", "ghz1")]
        if clk:
            w(f"- core clock over the cells: {min(clk):.2f} .. {max(clk):.2f} GHz")
        w("")
    print("\n".join(out))


def ab_table(paths):
    """tools/bench/e2e_ab.sh logs: B / A of the medians per cell (pass and cold)."""
    out = ["| tokenizer | corpus | chunk | A | B | pass B/A | cold B/A | load |", "|---|---|---|---|---|---|---|---|"]
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            if line.startswith("AB "):
                d = kv(line)
                out.append(f"| {d['tk']} | {d['corp']} | {d['chunk'] if d['chunk'] != '0' else 'whole'} | {d['A']} | {d['B']}"
                           f" | {float(d['pass_B_over_A']):.3f} | {float(d['cold_B_over_A']):.3f} | {d['load']} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
