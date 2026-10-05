toks design
===========

This file records how toks is built and the decisions taken on the way. Section numbers such as SPEC §2.7 refer
to the project's internal contract, which is not in this tree.

1. shape
--------

  python (uv)      tooling only: parity vs hf, benches, census, table generators, poking at the library
       |
  include/toks.h   the abi (source of truth)
       |
  c core           readers (tokenizer.json, tiktoken, tekken), the compiler (config -> tables), loader and
  (src/core)       validation, tier selection, the driver that sequences kernels per segment and chunk,
                   post-processing, decode, info; plus the c twin of every kernel (the scalar tier)
       |   switch on the load-time tier at each call site, direct calls, no function pointers
       v
  kernels          K1 added-token find, K3 scan (one per template), K5 encode pieces, K6 bpe
  (src/asm/<isa>)  arm64: neon. x86-64: avx2 (avx-512 is detected; no avx-512 kernel is built). two
                   independent implementations, same contract.

The maintainer's direction for this repo: the core loops are hand-written asm, maintained as two parallel
implementations (arm64, x86-64); a little c (clang) manages the loops and makes the attachment surfaces easy;
python is for interactivity and prodding. SPEC's architecture (c core + per-isa asm kernels) is the same
picture: everything per-byte and per-piece is asm; c runs once per call, segment or 256-piece chunk.

The c twin of each kernel is not a fallback we hope to use: it is the scalar tier SPEC §2.1 asks for, the
oracle the asm is diffed against (T5), and the baseline of the rent rule (§10.5, maintainer doctrine). Every hot loop
gets asm on both isas; each asm kernel must beat its twin by the rent rule's margin. A kernel that does not is
reported to the maintainer with its numbers, not silently swapped.

2. contracts
------------

  include/toks.h      public abi
  src/core/layout.h   tables, argument structs, offsets, entry/hash formats (c + asm)
  docs/kernels.md     kernel semantics, geometry rules, the driver around them

Properties worth knowing before touching them:
  - every table format is isa-independent. the hashes are crc32c steps; the portable definition, arm64
    crc32cx and x86 sse4.2 crc32 were checked equal over 100k random words. one compiled table set serves
    every tier.
  - a kernel takes (const toks_tables *t, <k>_args *a): two pointer arguments on every abi, so the
    sysv / win64 / aapcs64 differences stop at the macro layer.
  - kernels never read outside [text, text + len); the tail is the kernel's own business (scalar or a padded
    copy on its stack). no caller padding, no over-read "because it is the same page".
  - shortcut entries (static table + dynamic cache) carry the exact output of the reference algorithm for
    their key (SPEC §2.7): a hit is never "the token whose string this is".

3. decisions
------------

  d1  toks_load takes an options struct (forced tier, cache flag, diagnostics) and returns the context
      through an out pointer; the blocking feature of TOKS_E_UNSUPPORTED is reported in toks_diag.what.
  d2  error codes TOKS_E_ARG (checkable precondition failures) and TOKS_E_NOMEM (load-time allocation) are
      added to SPEC §4.9's list. accepted (decided 2026-10-03; SPEC v3.1).
  d3  "caches are updated only by calls that return n <= cap" (SPEC §4.4) is read as the segment memo.
      shortcut caches (the dynamic piece cache) hold exact per-piece results whatever the call's capacity,
      so they are filled by every call. accepted (decided 2026-10-03; SPEC v3.1).
  d4  m1 compiles tokenizer files into in-memory tables at every load. the .toks image (serialization,
      SPEC §8.2) is p3; until it lands the API has no image loader and no image flag (0.3.0 removed the reserved ones).
  d5  added tokens: K1 finds raw leftmost-longest matches; the driver applies hf's per-match policy (mode,
      single_word, lstrip, rstrip) exactly as find_matches does, including "a dropped match is not rescanned".
  d6  the x86 scalar tier is the c twin compiled for x86-64-v2; there is no hand-written scalar x86 asm.
  d7  heavy work (parity at full size, fuzzing, benchmarks) runs on the benchmark machines (d9) or ci, never on
      a developer laptop (maintainer doctrine). x86 asm can be smoke-tested locally under rosetta: avx2 with
      ROSETTA_ADVERTISE_AVX=1, never avx-512.
  d8  kernels own their tail (a padded copy of the last partial block in their own <= 256 B frame, masked
      loads, or scalar code) instead of SPEC §10.3's "the core feeds the last partial block from a padded
      copy in scratch". the guarantee is the same (no access outside [p, p + len), callers never pad); the
      kernel-owned copy keeps a piece that straddles the last block boundary inside one buffer. accepted
      (decided 2026-10-03; SPEC v3.1 §10.3).
  d9  machines (decided 2026-10-03; SPEC v3.1 §11; named by chipset since the public release, 2026-10-05): one
      AMD Threadripper 9970X box (zen 5, 32c/64t, avx-512 + vbmi, 125 GB; shared, record its load), five NVIDIA
      GB10 boxes (10x cortex-x925 + 10x cortex-a725, 119 GB, 4 KiB pages), two Apple M2 Ultra Mac Studios
      (192 GB) and one AMD Ryzen AI MAX+ 395 Windows box. no other machine is a receipt host. llvm 21.1.8 and uv
      are installed under the home directory on each; keys, toolchains and etiquette in docs/machines.md.
  d10 the repository is public (2026-10-05): releases are GitHub releases, and receipts carry chipset keys
      (docs/machines.md), never a hostname, a home path or another person's name (maintainer doctrine).
  d11 the first code review's comments (commit 9fa466c) are folded in: one-line 64-byte buckets for both
      shortcut tables (a hit never touches a second line on 64-byte-line cores), TOKS_TF_IDS_AS_RANK (the merged
      id is the priority: no rank -> id load on bpe's serial chain), kernel-owned tails (d8), huge-page advice
      before first touch, no start-of-input flag in K3 (no cl100k-family pattern consumes one), the scratch binding
      that makes the dynamic cache safe. the contract rules themselves were confirmed against hf 0.23.2 with
      0 mismatches over 13.2M scanner cases, 1.2M bpe pieces and 400K added-token encodings.
  d13 the python wheel (python/toks: a thin binding over the c abi) is the first satellite and ships with v1,
      outside T10 (decided 2026-10-03).
  d14 ci: the exactness gate runs on GitHub Actions (docs/ci.md); speed receipts come only from the benchmark
      machines (d9).
  d15 the split (decided 2026-10-03): asm for everything per byte and per piece (K1, K2, K3, K5, K6, the decode
      loop), written once per isa and often twice in parallel by independent workers (the fastest correct one
      ships, disagreements find bugs); c fearlessly wherever it does not affect speed: readers, compiler, loader,
      validation, the driver that runs once per call / segment / 256-piece chunk, api, platform. "the best code
      is the code that never needs to be written."
  d16 scope (decided 2026-10-03): every model, production-grade stability, tiny package, easy to add models. toks is
      the generic machinery hf tokenizers are built from (added tokens, scanner templates, normalizers, the four
      algorithms, decoders, post-processors); every model-specific fact is data the compiler derives from the
      model's own files. a model toks cannot represent exactly is refused at load with the feature named.
  d17 first consumer: the in-house inference engine (c, asm, ptx, sass). it calls tok v1 from asm on both isas;
      toks replaces it behind the same seam with a plain c abi, caller-owned scratch and output (the engine's
      pinned staging memory), no runtime, and a small asm include of toks.h's constants for its .asm / .S
      callers. the incumbent's internals were studied in an internal brief; what toks took from it is recorded
      in docs/kernels.md.
  d18 T10's line budgets count CODE lines: blank lines and comment-only lines do not count (decided 2026-10-04;
      until then the budgets counted raw `wc -l` lines). Asm is <= 5,000 code lines per isa (SPEC §22). Long
      rationale still belongs in docs/ with a section pointer in the code (docs/notes/c-core.md); the budget is
      spent on code that buys speed or coverage, never on cramming. `make size` (tools/size.sh) prints every T10
      row and fails when one is over; it is part of `make test`.
  d19 decided 2026-10-04 (SPEC §22): speed first, per core; multi-threading sensible and graceful (a couple of
      workers + a load balancer, only where it beats one core); the consumer does not exist yet, so toks's usage
      and abi may change for speed, documented in docs/usage.md; slow-tokenizer sources are converted at load in c
      like transformers' converters.

4. milestones
-------------

  m1 (byte-level bpe on the cl100k template, nfc, tiktoken inputs), m2 (the templates the census ranks next,
  lstrip / rstrip / single_word, stream decode, the segment memo, split planning, toks_par) and m3
  (sentencepiece-style bpe, unigram, wordpiece) have shipped; docs/release/ has the release reports. the .toks
  image (d4) and tekken inputs have not.

5. ownership
------------

  the maintainer owns the contract (internal), include/toks.h, src/core/layout.h, docs/kernels.md, this file, the
  Makefile and tools/remote.sh. struct toks_ctx in core.h is append-only (new fields at the end, commented).
  changes land as pull requests that list the commands they ran and on which machines (CONTRIBUTING.md).
