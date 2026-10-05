# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""tools/readme_charts.py: the README's charts, drawn from the generated speed tables, light and dark.

    uv run --with matplotlib tools/readme_charts.py

Reads docs/bench/e2e.md ("Headline: 4 KiB chunks" and the receipts) and docs/bench/par.md (the "after" tables) and
writes docs/img/<name>-light.svg and docs/img/<name>-dark.svg. Never type a number in here: every value is parsed
from those files, which are themselves generated from the raw logs (tools/bench/e2e_table.py, tests/par/par_table.py).
Rerun after regenerating them. The ratio and scaling charts compare toks with hf tokenizers (the reference people
run) and tiktoken; gigatoken, the bar toks races internally, is a dot in the throughput plots and otherwise stays
in e2e.md's columns. Colors follow a validated categorical order (blue, orange, aqua; hf in
neutral gray) and a one-hue blue ramp for magnitude, and every series also has its own marker shape, so identity
never rests on color alone."""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E2E = ROOT / "docs/bench/e2e.md"
PAR = ROOT / "docs/bench/par.md"
OUT = ROOT / "docs/img"

NAMES = {
    "gpt2": "GPT-2", "llama3": "Llama 3", "glm53": "GLM 5.3", "qwen38": "Qwen 3.8", "o200k": "gpt-oss (o200k)",
    "gemma4": "Gemma 4", "nemotron3-4b": "Nemotron 3", "llama4": "Llama 4", "minimaxm2": "MiniMax M2",
    "dsv4": "DeepSeek V4", "kimik3": "Kimi K3",
}
CORPORA = ("en", "code", "ml", "cjk")
CORPUS_TITLE = {"en": "English prose", "code": "code", "ml": "multilingual", "cjk": "Chinese / Japanese / Korean"}
# e2e.md's <host>-<tier> labels; <host> is a chipset key (docs/machines.md)
HOST_DESC = {
    "gb10c-neon": "gb10c · NVIDIA GB10, one Cortex-X925 core · NEON",
    "tr9970x-avx2": "tr9970x · AMD Threadripper 9970X (Zen 5), one core · AVX2",
    "m2ultra2-neon": "m2ultra2 · Apple M2 Ultra, one core · NEON",
}
# par.md's hosts and the e2e.md table whose Llama 3 en row gives tiktoken's and hf's one-thread MB/s there. tr9970x is
# the same machine; gb10b has no e2e.md table, so gb10c's (the same GB10 Cortex-X925 core) is used and the chart says so.
REF_TABLE = {"gb10b": ("gb10c-neon", "gb10c, same X925 core"), "tr9970x": ("tr9970x-avx2", "")}

# ---- theme --------------------------------------------------------------------------------------------------------

THEMES = {
    "light": dict(
        ink="#0b0b0b", ink2="#52514e", muted="#6b6a65", grid="#e1e0d9", axis="#c3c2b7",
        toks="#2a78d6", giga="#eb6834", tik="#1baf7a", hf="#8f8d86",
        seq=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],   # blue 100 .. 700: light = low, dark = high
        surface="#ffffff",
    ),
    "dark": dict(
        ink="#f0f6fc", ink2="#c3c2b7", muted="#9d9b93", grid="#2c2c2a", axis="#45453f",
        toks="#3987e5", giga="#d95926", tik="#199e70", hf="#8a8880",
        seq=["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#b7d3f6"],   # the same ramp, low recedes into the dark
        surface="#0d1117",
    ),
}

plt.rcParams.update({
    "svg.fonttype": "none",            # real text: crisp at any width, selectable, system fonts
    "svg.hashsalt": "toks",            # stable ids: regenerating unchanged data gives the same bytes
    "font.family": "sans-serif",
    "font.sans-serif": ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.linewidth": 0.8,
})


def style_axes(ax, t, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_facecolor("none")


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-{theme}.svg"
    fig.savefig(path, format="svg", transparent=True, metadata={"Date": None, "Creator": "tools/readme_charts.py"})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


# ---- parsing ------------------------------------------------------------------------------------------------------

def num(cell):
    m = re.match(r"\s*([0-9.]+)", cell)
    return float(m.group(1)) if m else None


def parse_e2e():
    """{label: {(tokenizer, corpus): row}} from the 4 KiB headline, plus {label: commit} from the receipts."""
    text = E2E.read_text()
    head = text.split("## Headline: 4 KiB chunks", 1)[1].split("\n## ", 1)[0]
    tables, label, cols = {}, None, None
    for line in head.splitlines():
        m = re.match(r"\*\*([\w.-]+)\*\*", line)
        if m:
            label, cols = m.group(1), None
            tables[label] = {}
            continue
        if not line.startswith("|") or label is None or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "tokenizer":
            cols = cells
            continue
        row = dict(zip(cols, cells))
        tables[label][(row["tokenizer"], row["corpus"])] = {
            "toks_cold": num(row["toks cold"]), "toks_warm": num(row["toks warm"]), "hf": num(row["hf"]),
            "tiktoken": num(row["tiktoken"]), "giga_cold": num(row["gigatoken cold"]),
        }
    commits = {}
    for m in re.finditer(r"### ([\w.-]+): `[^`]+`\n\n(?:- [^\n]*\n)*?- git: `([0-9a-f]+)`", text):
        commits[m.group(1)] = m.group(2)[:7]
    gen = re.search(r"Generated by `tools/bench/e2e_table.py`", text)
    if not tables or not gen:
        sys.exit("docs/bench/e2e.md: headline tables not found")
    return tables, commits


def parse_par(input_name="16 MiB"):
    """{host: {"rows": [(k, toks pass MB/s)], "serial": MB/s, "commit": sha, "load": str}}"""
    text = PAR.read_text()
    out = {}
    for sec in re.split(r"\n## ", text)[1:]:
        m = re.match(r"(\w+), after \([^ ]+ ([0-9a-f]+)\)", sec)
        if not m:
            continue
        host, commit = m.group(1), m.group(2)
        load = re.search(r"1-minute load over the run ([0-9.]+ \.\. [0-9.]+)", sec)
        rows, serial = [], None
        for line in sec.splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 11 or cells[0] != input_name:
                continue
            if cells[1] == "serial":
                serial = num(cells[2])
            else:
                rows.append((int(cells[1].split()[0]), num(cells[2])))
        if rows:
            out[host] = {"rows": rows, "serial": serial, "commit": commit, "load": load.group(1) if load else "?"}
    if not out:
        sys.exit("docs/bench/par.md: no 'after' tables found")
    return out


# ---- chart 1/2: cold throughput, dot plot per corpus ----------------------------------------------------------------

def chart_throughput(tables, commits, label, name, theme):
    t = THEMES[theme]
    data = tables[label]
    toks_names = list(dict.fromkeys(k for k, _ in data))
    n = len(toks_names)
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 0.27 * n + 1.25), sharey=True)
    fig.subplots_adjust(left=0.135, right=0.99, top=1 - 0.62 / (0.27 * n + 1.25), bottom=0.42 / (0.27 * n + 1.25),
                        wspace=0.07)
    series = [  # drawn back to front; toks last so it sits on top
        ("hf", "hf tokenizers", t["hf"], "o", 5.5),
        ("tiktoken", "tiktoken", t["tik"], "s", 5.5),
        ("giga_cold", "gigatoken", t["giga"], "D", 5.5),
        ("toks_cold", "toks", t["toks"], "o", 7.5),
    ]
    for ax, corpus in zip(axes, CORPORA):
        style_axes(ax, t)
        ax.set_xscale("log")
        ax.set_xlim(2, 1000)
        ax.set_xticks([3, 10, 30, 100, 300, 1000])
        ax.set_xticklabels(["3", "10", "30", "100", "300", "1k"])
        ax.minorticks_off()
        ax.set_title(CORPUS_TITLE[corpus] if corpus != "cjk" else "CJK", fontsize=9.5, color=t["ink"], loc="left",
                     pad=4)
        for i, tk in enumerate(toks_names):
            y = n - 1 - i
            row = data[(tk, corpus)]
            vals = [row[key] for key, *_ in series if row[key]]
            ax.plot([min(vals), max(vals)], [y, y], color=t["axis"], linewidth=1, zorder=1, solid_capstyle="round")
            for key, _, color, marker, size in series:
                if row[key]:
                    ax.plot(row[key], y, marker=marker, markersize=size, color=color, markeredgecolor=t["surface"],
                            markeredgewidth=1.2, linestyle="none", zorder=3)
        ax.set_ylim(-0.7, n - 0.3)
        ax.tick_params(axis="y", labelsize=9.5, labelcolor=t["ink"])
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([NAMES.get(x, x) for x in reversed(toks_names)])
    fig.text(0.135, 0.012, "MB/s of input, one thread, log scale", fontsize=8.5, color=t["muted"])
    fig.text(0.99, 0.012, f"{HOST_DESC.get(label, label)} · cold, flags 0 · 4 KiB chunks · {commits.get(label, '?')}",
             fontsize=8.5, color=t["muted"], ha="right")
    handles = [plt.Line2D([], [], marker=m, markersize=s, color=c, markeredgecolor=t["surface"], linestyle="none")
               for _, _, c, m, s in reversed(series)]
    fig.legend(handles, [lbl for _, lbl, *_ in reversed(series)], loc="upper left", ncol=4, frameon=False,
               bbox_to_anchor=(0.125, 1.0), fontsize=9.5, labelcolor=t["ink"], handletextpad=0.3, columnspacing=1.6)
    save(fig, name, theme)


# ---- chart 3: toks / hf tokenizers, cold, on a log color scale -------------------------------------------------------
# cold only: warm (the same text a second time) is answered from the default scratch's 4 MiB memo where the replay
# fits it, a lookup at GB/s, not a tokenizer speed to set beside hf's; the README gives warm's MB/s in its own sentence.

def chart_vs_hf(tables, commits, labels, name, theme):
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    panels = [(lb, "cold") for lb in labels]
    toks_names = list(dict.fromkeys(k for k, _ in tables[labels[0]]))
    n = len(toks_names)
    ratios = {(lb, st, tk, c): tables[lb][(tk, c)][f"toks_{st}"] / tables[lb][(tk, c)]["hf"]
              for lb, st in panels for tk in toks_names for c in CORPORA}
    norm = LogNorm(vmin=min(ratios.values()), vmax=max(ratios.values()))   # one scale for every panel
    h = 0.25 * n + 1.5
    fig, axes = plt.subplots(1, len(panels), figsize=(8.4, h), sharey=True)
    fig.subplots_adjust(left=0.135, right=0.99, top=1 - 0.78 / h, bottom=0.42 / h, wspace=0.08)
    for ax, (lb, st) in zip(axes, panels):
        for i, tk in enumerate(toks_names):
            for j, corpus in enumerate(CORPORA):
                r = ratios[(lb, st, tk, corpus)]
                color = cmap(norm(r))
                ax.add_patch(plt.Rectangle((j + 0.04, n - 1 - i - 0.44), 0.92, 0.88, color=color, linewidth=0))
                lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                ax.text(j + 0.5, n - 1 - i, f"{r:.1f}", ha="center", va="center", fontsize=7.6,
                        color="#0b0b0b" if lum > 0.45 else "#ffffff")
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_facecolor("none")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, labelsize=8.5, labelcolor=t["ink2"])
        ax.set_xticks([j + 0.5 for j in range(4)])
        ax.set_xticklabels(CORPORA)
        host = lb.split("-")[0]
        ax.set_title(f"{host} · toks {st} ÷ hf", fontsize=9.5, color=t["ink"], loc="left", pad=4)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([NAMES.get(x, x) for x in reversed(toks_names)], fontsize=9.5, color=t["ink"])
    commit = commits.get(labels[0], "?")
    lo, hi = norm.vmin, norm.vmax
    fig.text(0.135, 1 - 0.3 / h, "toks MB/s ÷ hf tokenizers MB/s, same chunks, same core.   "
             f"every cell > 1 · darker is further ahead · log scale, {lo:.1f}x to {hi:.0f}x",
             fontsize=9, color=t["ink2"])
    fig.text(0.99, 0.012, f"flags 0 · cold = a fresh scratch every call · 4 KiB chunks · one thread · {', '.join(labels)} · {commit}", fontsize=8.5,
             color=t["muted"], ha="right")
    save(fig, name, theme)


# ---- chart 4: toks_par scaling --------------------------------------------------------------------------------------

def chart_par(par, tables, commits, name, theme):
    t = THEMES[theme]
    hosts = [h for h in ("gb10b", "tr9970x") if h in par]
    desc = {"gb10b": "gb10b · GB10 Cortex-X925 cores", "tr9970x": "tr9970x · Zen 5, one CCD"}
    fig, axes = plt.subplots(1, len(hosts), figsize=(8.4, 4.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.735, bottom=0.185, wspace=0.10)
    ymax = max(max(r[1], par[h]["serial"] * r[0]) for h in hosts for r in par[h]["rows"])
    for ax, h in zip(axes, hosts):
        style_axes(ax, t, grid_axis="y")
        rows = par[h]["rows"]
        ks = [r[0] for r in rows]
        ideal = [par[h]["serial"] * k for k in ks]
        ax.plot(ks, ideal, color=t["axis"], linewidth=1.2, zorder=1)
        ax.text(ks[-1], ideal[-1], " serial × k", color=t["muted"], fontsize=8.5, va="bottom", ha="left")
        ref_label, ref_from = REF_TABLE[h]
        ref = tables[ref_label][("llama3", "en")]       # tiktoken and hf, one thread, Llama 3 en, 4 KiB chunks
        for key, lbl, color, dy in (("tiktoken", "tiktoken", t["tik"], 1), ("hf", "hf tokenizers", t["hf"], 0)):
            ax.axhline(ref[key], color=color, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2 + dy,
                       dash_capstyle="round")
            # the two lines sit on the axis at this scale (that is the point); the labels stack to the right,
            # where the toks line is high above them
            ax.annotate(f"{lbl} {ref[key]:.2f} MB/s" + (f" ({ref_from})" if ref_from else ""), (10.9, ref[key]),
                        xytext=(0, 4 + 13 * dy), textcoords="offset points", color=t["ink2"], fontsize=8.5,
                        va="bottom", ha="right")
        ax.plot(ks, [r[1] for r in rows], color=t["toks"], linewidth=2, marker="o", markersize=7.5,
                markeredgecolor=t["surface"], markeredgewidth=1.2, zorder=3, solid_capstyle="round")
        last = rows[-1]
        ax.annotate(f"{last[1]:,.0f} MB/s", (last[0], last[1]), xytext=(7, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=9, color=t["ink"])
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlim(0.5, 11)
        ax.set_ylim(0, ymax * 1.08)
        ax.set_xlabel("threads (k)", fontsize=9, color=t["muted"])
        ax.set_title(f"{desc.get(h, h)} · load {par[h]['load']} · {par[h]['commit']}", fontsize=9, color=t["ink"],
                     loc="left", pad=4)
    axes[0].set_ylabel("MB/s", fontsize=9, color=t["muted"])
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    handles = [
        plt.Line2D([], [], color=t["toks"], linewidth=2, marker="o", markersize=7.5, markeredgecolor=t["surface"]),
        plt.Line2D([], [], color=t["tik"], linewidth=1.6, linestyle=(0, (4, 3))),
        plt.Line2D([], [], color=t["axis"], linewidth=1.2),
        plt.Line2D([], [], color=t["hf"], linewidth=1.6, linestyle=(0, (4, 3))),
    ]
    fig.legend(handles, ["toks_par_encode", "tiktoken, one thread (speed table)", "perfect scaling of serial toks",
                         "hf tokenizers, one thread (speed table)"],
               loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.07, 1.0), fontsize=9.5,
               labelcolor=t["ink"], columnspacing=2.0, labelspacing=0.35)
    fig.text(0.08, 0.815, "one 16 MiB input (enwik8), Llama 3, pass state (new text on a warm pool)", fontsize=9,
             color=t["ink2"])
    e2e_commit = commits.get(REF_TABLE[hosts[0]][0], "?")
    fig.text(0.985, 0.012, "reference lines: the speed table's Llama 3 en row, 4 KiB chunks, one thread, "
             f"{e2e_commit} · gb10b has no row there, so its panel uses gb10c's", fontsize=8.5, color=t["muted"],
             ha="right")
    save(fig, name, theme)


# ---- chart 5: two vocabularies people deploy today, up close --------------------------------------------------------

def chart_models(tables, commits, label, models, name, theme):
    """GLM 5.3 and Kimi K3 up close: linear MB/s so the gap reads as the gap (hf is a sliver, and says so)."""
    t = THEMES[theme]
    data = tables[label]
    series = [("toks_cold", "toks", t["toks"]), ("giga_cold", "gigatoken", t["giga"]),
              ("tiktoken", "tiktoken", t["tik"]), ("hf", "hf tokenizers", t["hf"])]
    fig, axes = plt.subplots(1, len(models), figsize=(8.4, 3.7), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.16, wspace=0.08)
    w = 0.2
    top = max(data[(m, c)]["toks_cold"] for m in models for c in CORPORA) * 1.30
    for ax, model in zip(axes, models):
        style_axes(ax, t, grid_axis="y")
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for j, corpus in enumerate(CORPORA):
            row = data[(model, corpus)]
            for s_, (key, _, color) in enumerate(series):
                x = j + (s_ - 1.5) * w
                v = row[key]
                if not v:
                    ax.text(x, top * 0.01, "n/a", ha="center", va="bottom", fontsize=6.5, color=t["muted"],
                            rotation=90)
                    continue
                ax.bar(x, v, width=w - 0.03, color=color, linewidth=0, zorder=3)
                big = key == "toks_cold"
                ax.text(x, v + top * 0.012, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.6 if big else 6.4, color=t["ink"] if big else t["ink2"],
                        fontweight="bold" if big else "normal")
            ax.text(j - 1.5 * w, row["toks_cold"] + top * 0.085, f"{row['toks_cold'] / row['hf']:.0f}x hf",
                    ha="center", va="bottom", fontsize=9.5, color=t["toks"], fontweight="bold")
        ax.set_xticks(range(len(CORPORA)))
        ax.set_xticklabels([CORPUS_TITLE[c] if c != "cjk" else "CJK" for c in CORPORA], fontsize=9,
                           color=t["ink2"])
        ax.set_xlim(-0.6, len(CORPORA) - 0.4)
        ax.set_title(NAMES[model], fontsize=10, color=t["ink"], loc="left", pad=4)
    axes[0].set_ylabel("MB/s of input", fontsize=9, color=t["muted"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for _, _, c in series]
    fig.legend(handles, [lbl for _, lbl, _ in series], loc="upper left", ncol=4, frameon=False,
               bbox_to_anchor=(0.065, 1.0), fontsize=9.5, labelcolor=t["ink"], handlelength=1.2,
               handleheight=0.9, columnspacing=1.6)
    fig.text(0.075, 0.885, "cold: a fresh scratch for every call, so no cache from earlier text helps toks · "
             "same text, same core, linear scale", fontsize=9, color=t["ink2"])
    fig.text(0.99, 0.012, f"{HOST_DESC.get(label, label)} · cold, flags 0 · 4 KiB chunks · {commits.get(label, '?')}",
             fontsize=8.5, color=t["muted"], ha="right")
    save(fig, name, theme)


def main():
    tables, commits = parse_e2e()
    par = parse_par()
    for theme in THEMES:
        chart_throughput(tables, commits, "gb10c-neon", "speed-gb10c", theme)
        chart_throughput(tables, commits, "tr9970x-avx2", "speed-tr9970x", theme)
        chart_models(tables, commits, "gb10c-neon", ["glm53", "kimik3"], "modern-models", theme)
        chart_vs_hf(tables, commits, ["gb10c-neon", "tr9970x-avx2"], "vs-hf", theme)
        chart_par(par, tables, commits, "par-scaling", theme)


if __name__ == "__main__":
    main()
