# toks

[![test](https://github.com/actual-computer/toks/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/actual-computer/toks/actions/workflows/test.yml) [![nightly parity](https://github.com/actual-computer/toks/actions/workflows/nightly.yml/badge.svg)](https://github.com/actual-computer/toks/actions/workflows/nightly.yml) [![license](https://img.shields.io/badge/license-BUSL--1.1-blue)](LICENSING.md) [![python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](python/README.md)

**The tokenizer that keeps up with your memory bus.** (ﾉ◕ヮ◕)ﾉ*:・ﾟ✧

toks is Actual Computer's tokenizer. Give it the `tokenizer.json` your model ships with and it returns exactly the
ids [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) would, 13-151x faster. It's a small C
library with hand-written asm kernels for arm64 (NEON) and x86-64 (AVX2), no runtime and no dependencies, plus a
Python package with an hf-style `Tokenizer` API.

- **Exact.** Same ids as hf tokenizers 0.23.2, checked over hundreds of thousands of cases per model on every
  CPU tier. A tokenizer toks can't reproduce exactly gets refused at load, with the missing feature named. You
  never get quietly different ids.
- **Fast.** On one core with fresh text, toks is 13-151x faster than hf tokenizers in every cell of the speed
  table, 2-24x faster than tiktoken in every cell tiktoken can run, and ahead of gigatoken, the fastest tokenizer
  we know of, in 252 of 255 cold cells ([receipts below](#how-fast)).
- **Tiny and embeddable.** A plain C ABI ([`include/toks.h`](include/toks.h)) with caller-owned buffers. Nothing
  is allocated after load, the core has no threads and no callbacks, and the input is read where it sits without
  being copied. It's built to live inside inference engines.

> **toks 0.3.0 is released** (tag `v0.3.0`): C bundles for linux arm64, linux x86-64 and macOS arm64, and Python
> wheels for CPython 3.10-3.14. The release report, [`docs/release/0.3.md`](docs/release/0.3.md), carries every gate
> with its receipt at the release commit `d56a5c1`, and an UNMET table for the ones still open, the fuzzing budget
> and the proof package among them. The speed numbers below were measured at `245cc5c`, the release candidate; the
> release commit differs from it in one Unigram bound that none of the table's tokenizers uses. Commit ids and PR
> numbers quoted in this README and under `docs/` from before the first public commit belong to the private history
> this tree was cut from; the receipts they name are in the tree.

## How fast

On one core with fresh text, toks runs 13-151x faster than hf tokenizers in every cell of the speed table on the
asm tiers: 88 cells per machine on gb10c, tr9970x and m2ultra2 (the chipset keys of
[`docs/machines.md`](docs/machines.md)), 4 KiB chunks and whole-corpus calls, all of them exact.
Every number here comes with its cell: machine, tier, state, chunk size and commit. All of them were measured at
commit `245cc5c`, the release candidate; the release commit `d56a5c1` differs from it in one id-buffer bound in the
Unigram model (`src/core/unigram.c`), which none of the table's eleven tokenizers uses (all are BPE), and two
comments in `include/toks.h`. The receipts (machine fingerprint, pinning, load before and after every cell, binary
and corpus sha-256) are in [`docs/bench/e2e.md`](docs/bench/e2e.md), which is generated from raw logs. These are
one-thread runs, and toks's ids are compared with the reference for every cell, outside the timer.

**Cold** means a fresh scratch for every call, so no piece cache or memo from earlier text helps toks; the CPU's own
caches may still hold the text ([`docs/bench/e2e.md`](docs/bench/e2e.md) says which runs ran cold first and what that
costs). **Pass** means one scratch kept across the stream of chunks, its caches starting empty after other text went
through, the way a serving worker runs. **Warm** means the same text a second time. Every state uses the default
scratch: a 2 MiB piece cache and a 4 MiB memo.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/speed-gb10c-dark.svg">
  <img alt="Dot plot: one-thread MB/s on GB10 (Cortex-X925, NEON), cold, 4 KiB chunks, for 11 tokenizers across English prose, code, multilingual and CJK text. toks is the rightmost dot in every row, ahead of gigatoken, tiktoken and hf tokenizers." src="docs/img/speed-gb10c-light.svg">
</picture>

*gb10c-neon (NVIDIA GB10, one Cortex-X925 core), cold, 4 KiB chunks, commit `245cc5c`. For example, Llama 3 on
English prose: toks 269.8 MB/s, gigatoken 65.7, tiktoken 30.3, hf 5.36. tiktoken has no dot where it can't run
the file (Gemma 4 and DeepSeek V4) or where its ids differ from hf's (Qwen 3.8 ml / cjk).*

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/speed-tr9970x-dark.svg">
  <img alt="The same dot plot on a Threadripper 9970X (Zen 5, AVX2), cold, 4 KiB chunks: toks is the rightmost dot in every row." src="docs/img/speed-tr9970x-light.svg">
</picture>

*tr9970x-avx2 (AMD Threadripper 9970X, Zen 5, one core), cold, 4 KiB chunks, commit `245cc5c`.*

Two of the vocabularies people deploy today, up close. GLM 5.3's tokenizer is from 2026, and Kimi K3 ships a
tiktoken-format file with its own pre-tokenizer cuts; neither is a GPT-2-era format that is easy to make fast.
( ⌐■_■)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/modern-models-dark.svg">
  <img alt="Grouped bar chart on a linear scale, two panels: GLM 5.3 and Kimi K3 on GB10 (Cortex-X925), cold, 4 KiB chunks. For each of English prose, code, multilingual and CJK text there are four bars, toks, gigatoken, tiktoken and hf tokenizers, in MB/s, every bar labelled with its number. toks towers over the group everywhere and hf tokenizers is a sliver at the bottom; toks's multiple over hf is printed above each group, from 21x to 124x." src="docs/img/modern-models-light.svg">
</picture>

*gb10c-neon, cold, 4 KiB chunks, one thread, commit `245cc5c`. GLM 5.3 on English prose and code: toks 278.8 and
424.5 MB/s, hf 5.41 and 5.30, tiktoken 30.2 and 26.9, gigatoken 56.5 and 87.5. Kimi K3 on English prose and code:
toks 280.1 and 474.3 MB/s, gigatoken 113.2 and 157.9, hf 3.69 and 3.84. On tr9970x-avx2 the same cells are toks 295.0
and 383.2 MB/s (GLM 5.3) and 286.9 and 432.3 MB/s (Kimi K3).*

And the whole table against hf tokenizers, the reference everyone runs, as ratios, cold:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/vs-hf-dark.svg">
  <img alt="Heatmaps of toks cold MB/s divided by hf tokenizers MB/s for 11 tokenizers by 4 corpora, one panel per host, GB10 (Cortex-X925) and Threadripper 9970X (Zen 5), on a log color scale with the ratio printed in each cell. Every cell is above 1, from 14x (Gemma 4, multilingual, GB10) to 124x (Kimi K3, code, GB10)." src="docs/img/vs-hf-light.svg">
</picture>

*toks cold ÷ hf tokenizers 0.23.2, 4 KiB chunks, one thread, gb10c-neon and tr9970x-avx2, commit `245cc5c`. Darker
is further ahead, and every cell prints its ratio. hf has no cache state (it runs as it ships, its own caches warm).
The smallest multiple is Gemma 4 on multilingual text on the GB10 (Cortex-X925): 14x. The largest is Kimi K3 on code
on the GB10: 124x.*

Warm, the same text a second time, isn't in this chart, because a replay that fits the default 4 MiB memo is
answered from the memo: a lookup at 5.1-19.2 GB/s on English prose and code (the 66 4 KiB-chunk cells, all three
machines), not tokenization. A replay that outgrows the memo (the multilingual and CJK corpora, their 66 4 KiB-chunk
cells) falls back to the piece cache, at 183-706 MB/s, where gigatoken's 512 MiB pretoken cache is faster (the warm column of the tally below).

gigatoken isn't in the ratio chart. It's the bar toks races internally ([the mission](#the-mission)), and its
cold, pass and warm columns sit next to every cell in [`docs/bench/e2e.md`](docs/bench/e2e.md); the tally below
counts them.

Across the whole table (4 KiB chunks and whole-corpus calls, 88 cells per machine, all 88 exact) at `245cc5c`
([Gates](docs/bench/e2e.md#gates)):

| machine · tier | faster than tiktoken, cold | faster than gigatoken, cold vs cold | faster than gigatoken, pass vs pass | faster than gigatoken, warm vs warm |
|---|---:|---:|---:|---:|
| gb10c · neon (NVIDIA GB10, Cortex-X925) | 65 / 65 | 85 / 85 | 83 / 85 | 43 / 85 |
| tr9970x · avx2 (AMD Threadripper 9970X, Zen 5) | 65 / 65 | 83 / 85 | 82 / 85 | 49 / 85 |
| m2ultra2 · neon (Apple M2 Ultra) | 65 / 65 | 84 / 85 | 84 / 85 | 50 / 85 |

The asm earns its place: with the same cells on the same machine, the asm tier beats toks's own portable C twin in
88 / 88 cells, by a median of 2.14x cold on gb10c and 2.63x on tr9970x.

And toks replaces tok v1, the asm tokenizer it grew out of: on gb10c and tr9970x it is faster in all 264 cell x state
medians with the incumbent's integration flags (`TOKS_SCRATCH_MEMO_MIB(4)`), the 180 new-prompt cells (cold / pass /
lang) and the 84 replays, and in all 180 new-prompt cells with the default flags too
([Incumbent](docs/bench/e2e.md#incumbent-tok-v1)).

### Where toks is behind, by cell

Every cell toks loses, by name. ᕙ(⇀‸↼‶)ᕗ

- **Cold against gigatoken: three whole-corpus GPT-2 calls.** tr9970x code 0.87x (381 vs 438 MB/s) and English
  prose 0.95x (410 vs 431), and m2ultra2 code 0.99x (381 vs 383). Every 4 KiB cell is ahead on every machine.
- **Pass against gigatoken: six cells.** gb10c MiniMax M2 code 4 KiB 0.96x and English 4 KiB 0.97x; tr9970x
  GPT-2 code whole 0.90x, code 4 KiB 0.90x and English whole 0.99x; m2ultra2 GPT-2 code 4 KiB 0.97x.
- **Warm replays the memo doesn't answer.** toks wins 43-50 of 85 warm cells per machine. The 35-42 it loses are
  replays that outgrow the default 4 MiB memo and fall back to the piece cache, where gigatoken's 512 MiB pretoken
  cache is faster: every multilingual whole-corpus call, every CJK one but DeepSeek V4's, 8-11 multilingual and 2
  CJK 4 KiB cells, and 5-9 English whole-corpus calls per machine. The worst is Gemma 4 on CJK whole, 0.19-0.24x. The
  memo keeps each record's text beside its ids; records that keep the ids alone are the next step.
  [Gates](docs/bench/e2e.md#gates) lists every cell.
- **The floor itself.** The speed table measures toks against other tokenizers. The number it will carry next is
  the gap to the machine's physics floor, bytes moved per input byte at measured bandwidth: that gap is the target.

### On a few cores: `toks_par`

`toks_par` encodes one big input, or a batch of documents, on a small worker pool and returns exactly what serial
`toks_encode` returns. It cuts the text only at certified split points, and it only goes wide when its measured
cost model says that beats one core.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/par-scaling-dark.svg">
  <img alt="Line chart: toks_par_encode MB/s vs threads (1, 2, 4, 8) for one 16 MiB input, Llama 3, on GB10 (Cortex-X925) and Threadripper 9970X (Zen 5), with perfect scaling of serial toks as a thin line and tiktoken's and hf tokenizers' one-thread speeds from the speed table as flat dashed lines along the bottom. At 8 threads toks reaches 1,608 MB/s on the GB10 and 1,804 MB/s on the Threadripper 9970X; on one thread tiktoken is 30.3 and 35.1 MB/s and hf tokenizers 5.36 and 6.46." src="docs/img/par-scaling-light.svg">
</picture>

*One 16 MiB input (enwik8), Llama 3, pass state, commit `b6eca5f`. gb10b uses GB10 X925 cores and tr9970x uses one
Zen 5 CCD, with load recorded per run. At 8 threads: 1,608 MB/s (7.19x serial) on gb10b and 1,804 MB/s (5.74x) on
tr9970x, against tiktoken's 30.3 / 35.1 MB/s and hf's 5.36 / 6.46 on one thread. Those two flat lines are the speed
table's Llama 3 English-prose row (4 KiB chunks, one thread, commit `245cc5c`): tr9970x's own row for the tr9970x panel,
and gb10c's for the gb10b panel, which has no row of its own (the same GB10 X925 core; the chart says so). par.md
also measures tiktoken's and hf's own `encode_batch` on the same machines and pools, beside `toks_par` and gigatoken
(its "tiktoken and hf beside toks_par" sections). At 64 MiB, enwik8
(100 MB) is too short for two disjoint 64 MiB spans, so about half of the timed text was already in the warm-up:
those rows are a half replay where gigatoken's 512 MiB cache counts. They are not a pass row. Receipts:
[`docs/bench/par.md`](docs/bench/par.md).*

## What it speaks, exactly

toks implements the pieces every hf tokenizer is built from: added tokens, normalizers, pre-tokenizers, the four
model algorithms, decoders and post-processors. A model's `tokenizer.json` is just data, so supporting a new
model means loading its file.

| algorithm | families |
|---|---|
| byte-level BPE | **GLM 5.3**, **Kimi K3** (tiktoken file), **gpt-oss** (o200k), **Qwen 3.8**, Llama 3.x and 4 (Llama 1 and 2 too), DeepSeek V3, R1, V3.1, V3.2, V4, V4.1, Nemotron 3 and Nano 2, MiniMax Text-01, M1, M2, M3, GPT-2, SmolLM2, StarCoder-style digit splitters, Qwen-1 tiktoken directories, and more on the generic engine (DeepSeek-Coder, bloom, falcon, MiniCPM5, ...) |
| SentencePiece-style BPE | **Gemma 4** (and Gemma 1-3, Mistral v0.1-v0.3, CodeLlama, TinyLlama on the same path) |
| Unigram | T5 / Flan-T5, XLM-R family embedders (BGE-M3, multilingual E5), ALBERT, XLNet, BGE reranker, Arctic Embed 2, ruri, llm-jp |
| WordPiece | BERT, MiniLM, MPNet, BGE (zh), GTE, LaBSE, Arctic Embed, Jina v2, SPECTER2, PubMedBERT, ... |

The release targets and the latest-generation rows are pinned by sha-256 in
[`tests/data/targets/ledger.txt`](tests/data/targets/ledger.txt) and checked by `make test`; the rest are covered
by the census ([`docs/coverage.md`](docs/coverage.md)) and the fixture suites under `tests/data/`.

The bold ones are the five release targets, plus the latest generations of Nemotron, DeepSeek, MiniMax and
Llama. Each of those was checked against hf over 217,388-221,783
cases (encode in every mode, pieces, decode, stream decode) on every tier of the release machines, with 0 diffs
([`docs/release/0.3.md`](docs/release/0.3.md)).
Kimi K3 was checked against its own reference over 166,284 texts in 3 modes.

Beyond the named families, a census of the Hugging Face Hub (the top 500 text-generation and top 100
embedding/reranker models by downloads) finds that **toks loads 98.40% of them by downloads**, and 98.00% run on
compiled fast paths ([`docs/coverage.md`](docs/coverage.md), census of 2026-10-04 at `d0b927a418`). Most of the
rest are repos that ship only slow-tokenizer files (a bare `vocab.txt` or a SentencePiece `.model`). They're
listed there by name, together with what's missing.

## Getting it

Releases are on GitHub, not on PyPI yet: the wheels and the C bundles are attached to each tagged release of
[actual-computer/toks](https://github.com/actual-computer/toks/releases).

**Python wheel** (CPython 3.10-3.14; linux x86-64 and arm64 as manylinux2014, macOS arm64):

```sh
gh release download v0.3.0 --repo actual-computer/toks --pattern 'toks-0.3.0-cp312-*'   # your python's tag
uv pip install ./toks-0.3.0-cp312-cp312-<your platform>.whl
```

**C library** (`include/toks.h`, `lib/libtoks.a`, `lib/libtoks.so` / `.dylib`, plus asm headers and a `MANIFEST`
with every file's sha-256):

```sh
gh release download v0.3.0 --repo actual-computer/toks --pattern 'toks-0.3.0-linux-x86_64.tar.gz'
#   also: toks-0.3.0-linux-arm64.tar.gz, toks-0.3.0-macos-arm64.tar.gz
tar xzf toks-0.3.0-linux-x86_64.tar.gz
cc -O2 app.c -Itoks-0.3.0-linux-x86_64/include toks-0.3.0-linux-x86_64/lib/libtoks.a -pthread
```

**From source** (needs clang 21, LLVM's or Apple's, and make; the Python side uses
[uv](https://github.com/astral-sh/uv)):

```sh
make                                              # libtoks (static + shared) for this machine
python3 tools/ci/fetch_tokenizers.py              # the pinned tokenizer files the tests read (~640 MB, sha256-checked)
make test                                         # build and run the unit tests
uv build --wheel python --out-dir build/wheels    # the Python wheel
uv pip install build/wheels/toks-*.whl            # into the active environment
```

The fetch puts the files in `~/.cache/toks` (`TOKS_TOKENIZER_CACHE` moves it). Without them `make test` still
passes, and the tests that read a real tokenizer file print SKIP.

On Windows x86-64, `tools/win/build.cmd` builds the same library (`libtoks.lib`, `toks.dll`) and
`tools/win/test.cmd` tests it. There's no prebuilt Windows bundle yet.

## Using it from Python

```python
import toks

tok = toks.Tokenizer.from_file("path/to/tokenizer.json")   # a model directory works too
ids = tok.encode("Hello world")                            # == hf tok.encode("Hello world").ids
tok.encode("Hello world", add_special_tokens=False)        # no bos / eos / template
tok.encode(text, added_tokens="nonspecial")                # == hf with encode_special_tokens=True
tok.encode(text, added_tokens="none")                      # no added token recognized
tok.encode_batch(texts)                                    # list of lists, without the GIL
n = tok.encode_into(text, out)                             # ids straight into a uint32 / int32 buffer
tok.decode(ids)                                            # == hf tok.decode(ids)
st = tok.decode_stream(); st.push(ids); st.flush()         # incremental decode, for streaming output
tok.token_to_id("<|endoftext|>"), tok.id_to_token(50256), tok.token_bytes(50256)
tok.info()                                                 # algorithm, tier, n_ids, sha256, ...
```

The API follows hf's `tokenizers.Tokenizer` and drops in for the calls inference engines and embedding servers
make. The full surface is in [`python/README.md`](python/README.md).

## Using it from C

```c
#include "toks.h"

toks_ctx *ctx;
if (toks_load(&ctx, "path/to/tokenizer.json", NULL) != 0) {
    /* pass a toks_load_opts with .diag to learn which feature or byte was the problem */
}

uint64_t bytes = toks_scratch_bytes(ctx, max_len, 0);   /* per thread, sized for texts up to max_len */
void *scr = malloc(bytes);
toks_scratch_init(ctx, scr, bytes, 0);

uint32_t ids[4096];
int64_t n = toks_encode(ctx, text, len, 0, ids, 4096, scr);   /* n = total ids; ids[0..min(n, 4096)) exact */

toks_unload(ctx);
```

A loaded context is read-only and any number of threads can share it. Each thread brings its own scratch and
reuses it across calls, which is where the piece caches and the segment memo pay off.

## What you get back

- **Exactly hf's ids.** With flags `0` you get what hf's `tokenizer.encode(text)` gives: added tokens recognized,
  post-processor applied. Invalid UTF-8 has defined behaviour instead of an error.
- **Three added-token modes.** `TOKS_ADDED_ALL` (the default), `TOKS_ADDED_NONSPECIAL` (hf's
  `encode_special_tokens=True`) and `TOKS_ADDED_NONE`. Add `TOKS_NO_POSTPROCESS` to skip BOS/EOS/templates.
- **The other direction.** `toks_decode`, a stream decoder for token-by-token output (`toks_stream_push` /
  `toks_stream_flush`), and `toks_token` for zero-copy bytes of one id. hf decides a SentencePiece byte-fallback
  run as a whole, so the stream holds one until it ends: up to 44 bytes in the stream itself, any length in a
  buffer you give it with `toks_stream_hold`, which also grows a live stream ([`docs/usage.md`](docs/usage.md)).
- **Lookups.** `toks_token_to_id` takes the bytes an id decodes to (what `toks_token` returns) and gives the id back,
  an added token's content included, as hf's `token_to_id` does. It takes bytes, not hf's vocabulary spelling: a
  byte-level `" hello"`, not `"Ġhello"`. `toks_id_flags` says whether an id is an added token, a special one or one
  raw byte ([`docs/usage.md`](docs/usage.md)).
- **Sizing out.** `toks_encode_bound(ctx, len)` is the most ids `toks_encode` can return for any text of `len` bytes,
  so an output buffer of that size is allocated once and never retried.
- **Pieces.** `toks_pieces` shows the pre-tokenizer's pieces as end offsets, for inspection.
- **Certified split points.** `toks_split_points` finds places where you can cut a text, encode the parts
  separately and concatenate the ids to get exactly the whole text's ids. Use it for segment caches, chunk stores
  and incremental documents. A piece boundary is not a cut ([`docs/usage.md`](docs/usage.md)).
- **A few cores.** `toks_par_encode` / `toks_par_encode_batch` give the same output as serial calls, on a small
  pool that only goes wide where it measurably pays.
- **Caches you can size.** The default scratch has a 2 MiB piece cache and a 4 MiB segment memo, which answers
  text it has seen before (a re-sent conversation) without encoding it again. `TOKS_SCRATCH_MEMO_MIB(0)` turns the
  memo off for batch jobs over text that never comes back; `TOKS_SCRATCH_CACHE_MIB(n)` sets a bigger piece cache
  for long-lived workers. The output is identical at every size, and [`docs/usage.md`](docs/usage.md) says which
  flags win where, by measurement.

## Platforms

| platform | asm tier | C library | Python wheel | tested on |
|---|---|---|---|---|
| linux arm64 | NEON | release bundle | cp310-cp314, manylinux2014 | NVIDIA GB10 (Cortex-X925) |
| linux x86-64 | AVX2 (AVX-512 machines run it) | release bundle | cp310-cp314, manylinux2014 | AMD Threadripper 9970X (Zen 5) |
| macOS arm64 | NEON | release bundle | cp310-cp314 | Apple M2 Ultra |
| windows x86-64 | AVX2 | build with `tools/win/build.cmd` | not yet | AMD Ryzen AI MAX+ 395 |

Every tier also has a portable C twin (`TOKS_TIER_SCALAR`) that gives identical output. The fastest tier your
CPU supports is picked at load.

## How it works, briefly

A tokenizer turns bytes into ids in five stages, and toks runs each one the way hf does:

1. **Added tokens.** Literal special strings (`<|im_start|>`, `<|eot_id|>`, ...) are found first and cut the
   text into segments.
2. **Normalizer.** Unicode rewriting (NFC, NFKC, lowercase, ...). Where a step only substitutes or prepends, it's
   compiled into the tables, so the text is never rewritten.
3. **Pre-tokenizer.** Each segment is cut into pieces. Instead of running a regex engine, toks maps the model's
   pattern onto a SIMD scanner template ([`docs/kernels.md`](docs/kernels.md)), falling back to an exact generic
   engine for rare patterns.
4. **Model.** Each piece becomes ids (byte-level BPE, SentencePiece BPE, Unigram or WordPiece). Common pieces
   are answered from caches whose every entry is computed by the real algorithm.
5. **Post-processor.** BOS/EOS and templates.

The text is read where it sits, and the ids go straight into your buffer.

## The mission

toks exists to run this problem as fast as the machine physically allows, exactly, every time, with nothing to
tune. It *rooflines* its machine: it works out the hardware's physical limit and runs as close to it as physics
allows. The target isn't another library. It's the **physics floor** of the machine you're on: the time it takes
to stream the text in and the ids out at that machine's measured memory bandwidth (internal spec §0.1).
Whatever separates toks from that floor is the backlog.

Exactness comes first, and speed is built on top of it. hf tokenizers is the **oracle**: whatever ids it
produces are the right answer by definition. tiktoken, [gigatoken](https://github.com/marcelroed/gigatoken) and
the asm tokenizer toks replaces (tok v1, [credits](#credits)) are **bars on the way to the floor**. We measure
against them in every cell and clear them in all but the handful listed above; none of them is the goal.

It's also meant to be simple (internal spec §0.3). You load what the model ships, and toks compiles it,
certifies it and picks the fastest exact path for your CPU. There's no profile to choose and no "compat mode"
that trades exactness for speed. The exact mode is the fast mode. ( ˘▽˘)っ♨

## Contributing and the docs

toks is held to a written contract, and every claim ships with its receipts. Issues, bug reports, receipts, ideas
and pull requests are welcome; for now every pull request that merges is run by an Actual Computer engineer, who
takes contributors' changes into it with their authorship kept. How a change lands (the tests, the receipts a
speed claim needs, the sign-off) is in [`CONTRIBUTING.md`](CONTRIBUTING.md). The docs sometimes cite a section of
the project's internal contract as SPEC §n; the contract itself is not in this tree.

- [`docs/`](docs): kernel semantics, design decisions, the census, the speed tables and release reports
- [`docs/machines.md`](docs/machines.md): the benchmark machines behind every receipt, by chipset, and the keys
  that name them in tables and logs
- [`tools/readme_charts.py`](tools/readme_charts.py): redraws this README's charts from the generated tables
  (`uv run --with matplotlib tools/readme_charts.py`)

## License

toks is source-available under the [Business Source License 1.1](LICENSE) (SPDX `BUSL-1.1`). It is free for any
organization that, together with its affiliates, processes under 10^15 tokens a year in total, and every version
becomes Apache-2.0 four years after its first public release. Organizations at or above that line get an inexpensive
commercial license from [Actual Computer](https://actual.inc); the plain-language summary is
[LICENSING.md](LICENSING.md).

## Credits

toks is made by [Actual Computer](https://actual.inc), standing on the shoulders of (´｡• ᵕ •｡`)

- **[Hugging Face tokenizers](https://github.com/huggingface/tokenizers).** It defines what every toks result
  must be. toks implements its pipeline (normalizer, pre-tokenizer, model, post-processor, decoder) and checks
  every result against it.
- **[gigatoken](https://github.com/marcelroed/gigatoken)** by [@marcelroed](https://github.com/marcelroed). Its
  SIMD pre-tokenizer scanners and pretoken cache showed how fast this problem can go, inspired toks's scanner and
  cache designs, and are still the bar on warm replays of more text than toks's memo holds.
- **[tiktoken](https://github.com/openai/tiktoken)** by OpenAI. It's the home of the cl100k and o200k encodings
  that toks's scanner templates follow, the reference for models that ship tiktoken files, and a speed bar toks
  clears in every cell it supports.
- **tok v1**, the tokenizer inside e, Actual Computer's inference engine: the asm tokenizer toks grew out of and
  now replaces. It lives in a private repository, so its cells in [`docs/bench/e2e.md`](docs/bench/e2e.md) are
  receipted there but cannot be reproduced from this tree.
