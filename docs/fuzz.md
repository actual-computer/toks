fuzzing toks
============

SPEC §14.2 (sanitizer matrix), §14.4 (stateful, synthetic testing), T6 (libFuzzer per entry point, >= 24 cpu-h per
isa per entry point at release, 0 findings) and T9 (readers and compiler fuzzed on arbitrary and structurally valid
hostile tokenizer files). This file says what the harnesses check, how to run them, and what has been run.
Everything lives in tests/fuzz/; `make fuzz` builds it (never part of `make test`). A developer laptop runs no
fuzzing (maintainer doctrine): build and run on a lab host through tools/remote.sh.


1. harnesses
------------

One libFuzzer binary per entry point, ASan + UBSan (-fno-sanitize-recover=all) by default, MSan and source coverage
as build variants (tests/fuzz/Makefile). The library is compiled from source with coverage instrumentation; the asm
kernels of the isa are assembled and dispatched as in `make`, so every harness runs each input through TOKS_TIER_AUTO
(the asm tier: neon / avx2) and TOKS_TIER_SCALAR (the instrumented c twins) and compares them. Black box: toks.h only.

  harness     entry points                          input (header + payload)            mutator
  encode      toks_encode, toks_encode_bound        gen.h: 8 bytes + text               gen.h class-aware generator
  pieces      toks_pieces                           the same                            the same
  par         toks_par_encode, toks_split_points    the same                            the same
  decode      toks_decode, toks_token_to_id,        ids.h: 4 bytes + u32 ids            libFuzzer's own
              toks_id_flags
  stream      toks_stream_init / push / flush /     the same                            libFuzzer's own
              hold
  load        toks_load_mem_copy, toks_load (file,  arbitrary bytes; a TIKTOKEN input   libFuzzer's own + tokenizer.dict
              model directory, tiktoken model       (ranks, config, wrapper) is a
              directory), opts and diag;            tiktoken model directory
              every entry point above on each
              accepted file
  load_json   the same                              tokenizer.json text                 jsonmut.h structure edits +
                                                                                        crossover, synth.h tokenizers

The text and id harnesses run on 17 pinned tokenizers, 16 files in ~/.cache/toks/tokenizers (tools/corpora/
fetch_tokenizers.py) and the kimi k3 model directory beside it, chosen by header byte 0 (fuzz.h FZ_PIN_DEF):
byte-level bpe gpt2, llama3, glm53, o200k (gpt-oss), dsv3,
nemotron3-4b, dg-smollm2 (digits template), pythia (missing bytes), qwen38 and minimaxm2 (NFC), kimik3 (tiktoken, the
kimi wrapper's run cuts); sentencepiece-style
bpe gemma4, mistral-v0.3, tinyllama; wordpiece wp-bert-uncased; unigram uni_t5base, uni_bgem3. A missing file falls
through to the next. The header also picks the flags (every mode, post-processing on/off, continuation) and `sel`:
scratch alignments, capacities, rebinding, partitions. The text is copied to an exact-size heap block (ASan sees any
read past len).

The class-aware generator (gen.h) writes atoms of every class the scanners distinguish
(L, N, P, WS, NL, marks), contractions with case and U+017F variants, the 25 \s chars, invalid utf-8 of every §3.3
kind, the tokenizer's own markup tokens (whole, cut short, overlapping, doubled, with a char inside), runs straddling
16 / 32 / 64-byte blocks and the driver's 256-piece rounds, single pieces up to 6,000 letters, NFC changes (growth up
to 3x), sentencepiece texts; a quarter of the mutations are libFuzzer's byte-level ones.


2. oracles
----------

Invariants of the contract and second toks paths that must agree; never hf (the parity suites own hf):

  - tiers: encode / pieces / decode through the scalar context == through the auto (asm) context, every input;
  - state (T2): a cold scratch of exactly toks_scratch_bytes(len) at any start alignment, the same scratch again,
    scratches warm across every earlier input at 2 MiB and at TOKS_SCRATCH_CACHE_MIB(32), a scratch rebound to
    another tokenizer (TOKS_E_SCRATCH until rebound) and back: one answer. (A scratch binds to the tokenizer's source
    identity: both tiers of one file share it, by design);
  - capacity: any cap returns the same total and an exact prefix, nothing written at or past cap; count-only calls;
    unknown flags are TOKS_E_ARG; encode's count <= toks_encode_bound(len) under every flag set, the bound the same
    on both tiers, monotone in len, 0 for a NULL ctx and saturated at UINT64_MAX;
  - pieces: ends <= 3 len (NFC growth); for byte-level files without a normalizer the last end is len (ends may go
    back: an rstrip added token's span overlaps the matches after it, exactly as hf's offsets do);
  - decode: an id >= n_ids is TOKS_E_ID with nothing written, unknown flags TOKS_E_ARG; always valid utf-8; the
    capacity rule; for byte-level files without a normalizer (FZ_RT) decode(ids, 0) == an independent
    from_utf8_lossy (unicode §3.9 table 3-7) of the ids' toks_token bytes, and decode(encode(x, NONE |
    NO_POSTPROCESS)) == from_utf8_lossy(x);
  - stream: the ids pushed in random parts (1-64) + flush == batch decode, every push within toks_stream_bound
    (+ 3 x the cap of a caller's hold); a push or flush one byte short is TOKS_E_CAP with the state and the hold's
    bytes unchanged and out[0, cap) the exact prefix. The hold (toks_stream_hold): a caller's hold from the start
    now and then, and moves between calls to a larger or smaller hold or back into the stream's own 44 bytes (any
    alignment, each hold flush with its heap block; TOKS_E_LIMIT with the state and the hold unchanged when the run
    does not fit); a push that returns TOKS_E_LIMIT leaves the state and the hold unchanged and gets toks.h's
    recovery (a hold of the current limit plus the push's ids, then the same ids again), so a second TOKS_E_LIMIT
    on that push is a finding;
  - lookups: toks_id_flags of every id the harness holds: TOKS_ID_* bits only, SPECIAL only with ADDED, TOKS_E_ID at
    or past n_ids, both tiers agree; toks_token_to_id of each id's own string is found, and of that string cut,
    extended or with one byte changed, and of slices of a loaded file: TOKS_E_ID, or an id whose toks_token bytes
    are the query, or an added id (an added content of those bytes holds it); the empty string is never found;
    NULL with len > 0 and a NULL ctx are TOKS_E_ARG; every query sits flush with its heap block;
  - par: toks_par_encode == toks_encode; toks_split_points returns strictly increasing cuts in (0, len), and the
    parts (TOKS_CONTINUATION after the first) concatenate to the whole input's ids without post-processing (§5.2);
    its offs sits at any alignment, flush with its heap block, now and then with a cap below n_want - 1: at most
    min(cap, n_want - 1) cuts, nothing written before offs or at or past offs[cap];
  - load: both tiers give the same verdict; a refusal is a documented code with *out NULL and diag carrying it;
    TOKS_E_NOMEM on a source under 16 MiB is a finding (load work must be bounded by the source); an accepted file
    gets the battery: probe texts, slices of the file, its own token strings glued, through every text oracle above
    (1 in 16 also through toks_par), plus decode, stream and the lookups of random ids.

A violated invariant prints "FUZZ INVARIANT VIOLATED file:line [tokenizer]: ..." and aborts; libFuzzer saves the
input.


3. running
----------

On a lab host, from the repository root (tools/remote.sh syncs to ~/toks-ci/<branch>/ and keeps build/; record
load; <= 8 processes per GB10, A725 cores only; tr9970x CCD0 only, cpus 0-7 and 32-39, run.sh's defaults):

  tools/remote.sh <host> 'make fuzz'                                  # = tests/fuzz/run.sh build, build/fuzz-<os>-<isa>/
  tools/remote.sh <host> 'TOKS_BENCH_TEXT=<bench text dir> tests/fuzz/run.sh seeds'   # texts, ids, shrunk files
  tools/remote.sh <host> 'tests/fuzz/run.sh start 4800'               # 7 detached processes, 80 min each
  tools/remote.sh <host> 'tests/fuzz/run.sh status'                   # execs, exec/s, cov, corpus, cpu-s, findings
  tools/remote.sh <host> 'tests/fuzz/run.sh stop [harness...]'
  tools/remote.sh <host> 'tests/fuzz/run.sh merge encode'             # minimized corpus
  tools/remote.sh <host> 'tests/fuzz/run.sh cover'                    # llvm-cov of src/ by the corpora
  tools/remote.sh <host> 'FUZZ_CPUS=0-3 tests/fuzz/run.sh start 4800'
  tools/remote.sh <host> 'make -f tests/fuzz/Makefile SAN=memory -j8'  # msan, build/fuzz-msan-<os>-<isa>/ (linux)
  tools/remote.sh <host> 'uv run tests/fuzz/seeds.py sweep --out build/fuzz/seeds <tokenizer dir> tests/data/compile'
  tools/remote.sh <host> 'uv run tests/fuzz/seeds.py tiktoken --out build/fuzz/seeds <tokenizer dir>'   # in seeds
  tools/remote.sh <host> 'tests/fuzz/run.sh probe'                    # the arena probe over corpora, seeds, repros

A finding reproduces with `build/fuzz-<os>-<isa>/fuzz_<h> <file>`. Each one gets a repro in tests/fuzz/regress/<h>/,
and `tests/fuzz/run.sh regress` replays them all (each must pass once its owner's fix is in). The msan build has no
asm kernels (msan cannot see their stores, so TOKS_TIER_AUTO would report every read of a kernel's output): it fuzzes
the c twins, the tier oracle compares the scalar tier with itself there. The load harness writes a file load's input
into $TMPDIR/toks-fuzz-<pid>/ and removes the directory after each such load. Every build of tests/fuzz/Makefile
carries the arena probe (tests/fuzz/arprobe.h, force-included into its src/core objects only): `run.sh probe` prints
per toks_ar_alloc site the calls, the refusals and the least slack the load harnesses' inputs reached
(docs/hardening.md §6).

The cpu-hour ledger (`run.sh hours`, T6): each run's cpu-s are user + sys from /usr/bin/time (fork-mode jobs
included: the harness reaps them), plus the cpu-s so far of every run in flight (/proc), per harness and build sha.
`run.sh stop` stops the fuzzer and its jobs but not the run's time, so a stopped run keeps its cpu-s (less a
fork-mode job in flight, at most 300 s). Every run forces leak detection on (ASAN_OPTIONS ...:detect_leaks=1) and
writes the options as its log's first line and into its .meta: the hours are asan + ubsan + lsan hours.


4. budgets (T6, §14.5)
----------------------

  release    >= 24 cpu-h per isa per entry point (seven harnesses: 168 cpu-h per isa), 0 findings
  nightly    1 cpu-h per entry point per isa
  per change make fuzz builds (on a host with libFuzzer)


5. campaigns
------------

5.1 campaign 1 (2026-10-04, 80 min per process, ASan + UBSan, aligned caller arrays): master 8aacce4 + harness d8f3127
(encode restarted after the dg-smollm2 oracle fix: 72 min). gb10a + gb10b on A725 cores 0-4,10-14, tr9970x on cpus 24-31;
load (1 / 5 / 15 min) at the end, incl. the 7 fuzz
processes: gb10a 1.2 / 4.8 / 6.3, gb10b 2.3 / 5.8 / 7.2, tr9970x 10.1 / 13.8 / 15.2 (shared host). cpu-s from /usr/bin/time, execs from the logs:

  harness     gb10a execs / cpu-s     gb10b execs / cpu-s     tr9970x execs / cpu-s          cpu-h total
  encode         770,856 / 4,381          799,493 / 4,380        1,479,695 / 4,343          3.64
  pieces       1,411,312 / 4,799        1,475,882 / 4,799        2,442,705 / 4,791          4.00
  decode      13,992,569 / 4,799       11,718,019 / 4,800       28,884,379 / 4,799          4.00
  stream      17,681,121 / 4,799       17,831,800 / 4,800       27,898,752 / 4,799          4.00
  par          2,665,870 / 4,798        2,789,625 / 4,799        4,587,525 / 4,792          4.00
  load         2,119,356 / 4,813        2,314,422 / 4,884        4,435,098 / 4,874          4.05
  load_json      365,845 / 4,838          349,642 / 4,850          734,316 / 4,804          4.03

  findings: F1 (82 repros) only; plus three harness false positives, fixed in the harness (dg-smollm2 drops bytes:
  no roundtrip oracle there; rstrip overlaps make piece ends go back exactly as hf's offsets do; a scratch binds to
  the source identity, so the rebind check needs another tokenizer).

5.2 misaligned-array run (tr9970x cpus 32-39, ~53 min per process, pre-fix master): caller id arrays 0-3 bytes off their
allocation, alignment reports recoverable. decode 16.2M, stream 21.6M, pieces 1.6M, par 3.0M, encode 1.0M, load 2.8M,
load_json 0.45M execs: UBSan alignment reports only in wordpiece (F2: wp_api.c:57, wp.c:47 / 56 / 98 / 101), plus F1.

5.3 findings

  F1  config.c read_vocab / parse_unigram: an id-hole vocabulary (spm-style bpe, unigram) sized its arrays and string
      index by max id + 1 from the parse arena (32 B a source byte): 8-14 KB files with max ids 492 .. 197,375 were
      refused with TOKS_E_NOMEM "model.vocab arrays". Fixed in master ac1a5f5 (index by entries; TOKS_E_LIMIT when
      the holes still do not fit). Repros: regress/load/, regress/load_json/.
  F2  wp_api.c:57 + wp.c: wordpiece wrote the caller's out with typed stores / a memcpy through uint32_t *
      (SPEC 4.1: no alignment required). Fixed in master ac1a5f5 (toks_st32 / toks_cpy). Repro: regress/encode/.

  F3  the generic engine + small dense files: after the F1 / F2 fixes, TOKS_E_NOMEM was still reachable on 1-3.4 KB
      files through the parse arena ("generic pre-tokenizer" ~130 KiB of compile arrays; "model.vocab arrays" /
      "added_tokens arrays" on tiny files); hf loads 9 of the 19 repros. Fixed (the compiler's work leaves the
      arena; toks_config_arena_bound covers the program). Repros: regress/load*/nomem-generic-*, nomem-vocab-*.

  F4  unigram.c:571 uni_resolve, found by campaign 4 (the T6 hours, below): the load_json harness on tr9970x,
      2026-10-05 08:16Z, UBSan 'index 32 out of bounds for type uint32_t[32]' (uni_resolve <- piece_close <-
      toks_uni_encode_segment <- toks_encode, the load battery's text). A queued piece (<= 15 bytes plus the virtual
      U+2581) was resolved into a 32-entry id buffer; under byte fallback a remapped space is the three ids of U+2581's
      bytes, so a Unigram with byte_fallback, the ' ' -> U+2581 Replace and no U+2581 piece emits up to 48 ids for one
      piece, and a release build returned the values past the 32nd as ids (the count was right). Minimal case: vocab
      <unk>, <0xE2>, <0x96>, <0x81>, a, the Replace alone, 11 spaces: hf 0.23.2 gives 33 ids, (1 2 3) x 11; toks gave
      33 with the 33rd not an id of the vocab, both tiers, any isa (toks_encode_bound: 33, held); 15 spaces: 13 wrong.
      Reach: none of the 12 cached unigram files has the shape (the three with byte fallback, llm-jp-3, llm-jp-4 and
      ruri-3, have a U+2581 piece). Fixed in #248 (the buffer holds the 48; a piece that emits more is answered
      straight into the caller's emitter, uncached); test_load's test_uni_resolve_ids checks every id against hf on
      0..20 spaces in three shapes over tests/data/unigram/byte_fallback_spaces.json. Repro: regress/load_json/
      ubsan-uni-resolve-3fcbffd5.json, exit 1 before the fix on both isas, `run.sh regress` 29 / 29 at #248's head
      on gb10d and tr9970x. A second file of the same site (tr9970x, 08:44Z) passes on the fix as well.

5.4 campaign 2 (2026-10-04, 45 min per process, master ac1a5f5, with the F1 / F2 fixes; misaligned caller arrays
everywhere, alignment reports recoverable; same hosts and cores as 5.1):

  harness     gb10a execs / cpu-s     gb10b execs / cpu-s     tr9970x execs / cpu-s          cpu-h total
  encode         487,640 / 2,700          470,299 / 2,700          920,606 / 2,655          2.24
  pieces         775,348 / 2,699          778,029 / 2,700        1,363,225 / 2,654          2.24
  decode       6,482,676 / 2,700        5,115,133 / 2,701       15,987,083 / 2,656          2.24
  stream       9,465,306 / 2,700        8,949,556 / 2,701       16,354,172 / 2,657          2.24
  par          1,448,006 / 2,700        1,410,051 / 2,700        2,596,754 / 2,654          2.24
  load           853,462 / 2,730          928,763 / 2,726        1,941,425 / 2,717          2.27
  load_json      170,384 / 2,708          165,087 / 2,757          408,461 / 2,687          2.26

  findings: F3 only (95 repros: 15 + 16 + 64); no UBSan report at all (alignment clean after the F2 fix on both isas).
  On master 5f71528 (the F3 fix in) all 95 replay clean (gb10b, neon build) and tests/fuzz/regress passes (22 files).

5.5 totals per entry point (campaigns 1 + 2 + the tr9970x misaligned run, which lost its timer files to the stop: its
~53 min per process are counted as 0.88 cpu-h): 6.8-7.2 cpu-h each; arm64 3.9-4.2, x86-64 2.8-3.0. T6's release
budget is 24 cpu-h per isa per entry point: unmet, and stays listed as unmet.

With campaign 3 (5.7: load and load_json, 0.50 cpu-h each on arm64) and campaign 4 (the T6 hours, 2026-10-05: every
public entry point, ASan + UBSan + LSan; arm64 on gb10d + gb10b, one process per harness each, x86-64 on tr9970x, two
per harness), whose ledger was read in flight at 08:50Z (`run.sh hours` over its builds fe94cde, 28ba09a, e8b634a and
ee6efde), cpu-h per harness:

  harness     5.1-5.7               campaign 4 at 08:50Z    total
              arm64     x86-64      arm64     x86-64        arm64     x86-64
  encode       3.93      2.82        7.34      1.95         11.27      4.77
  pieces       4.17      2.95        7.34      1.95         11.51      4.90
  decode       4.17      2.95        7.34      1.95         11.51      4.90
  stream       4.17      2.95        7.34      1.95         11.51      4.90
  par          4.17      2.95        6.93      2.61         11.10      5.56
  load         4.71      2.99        7.36      2.04         12.07      5.03
  load_json    4.71      2.96        7.33      2.03         12.04      4.99

Campaign 4's findings: F4 (two files of one site, both from x86-64's load_json, 08:16Z and 08:44Z), fixed in #248.
Since 09:01Z (arm64) and 09:09Z (x86-64) every process runs 0.3.0's final code (d56a5c1) with #236's harness changes,
after a replay of each harness's corpus and seeds through the new binaries (no report); the arm64 legs run to 15:10Z
(~20 cpu-h per harness from campaign 4), the x86-64 leg to 19:52Z (~24). T6 stays unmet until both isas pass 24 with
no open finding.

5.6 the triage of every finding file (hardening lane, 2026-10-05)

The 395 files: every crash-, leak-, oom- and slow-unit- file the campaigns above and earlier campaigns left,
kept in ~/.cache/toks/fuzz-findings on five hosts when their build trees were cleaned (gb10a 96, gb10b 116, tr9970x
178, gb10d 3, gb10c 2); 263 distinct by sha256 (gb10b's c2x/ held copies of the other hosts'), 300 (harness, file)
pairs. Each pair was replayed under ASan + UBSan + LSan on master ac14d02 (arm64 gb10d and x86-64 tr9970x: 297 / 297
pass; the 3 others are the archived front-accept and smoke harnesses' inputs, judged as texts below) and on the build
its campaign ran: d8f3127 (5.1), 026e4a5 (5.4), 2ff7e11 (5.2: the misaligned harness on the pre-#110 library) and
497303d (the hardening lane's par harness), plus 22 more builds spanning 82cdeda .. d11f9d1 (13 master merges, the
wordpiece / unigram / K1 branch heads), both isas for five of them. No file was an out-of-bounds access.

  bucket               files (distinct)  harness                found in, hosts          fails on    verdict  fix
  F1 vocab-hole NOMEM  82 (82)           load 81, load_json 1   5.1, gb10a gb10b tr9970x  d8f3127     a        #110
  piece-ends oracle    2 (2)             load_json              5.1                       d8f3127     b        a62d8ce
  dg-smollm2 oracle    3 (3)             encode                 5.1, all three            d8f3127     b        5b1cc77
  F3 generic NOMEM     190 (95)          load_json 188, load 2  5.4, gb10a gb10b tr9970x  026e4a5     a        #116
  F1 again             35 (35)           load                   5.2, tr9970x              2ff7e11 [1] a        #110
  F2 misaligned store  74 (37)           load 36, load_json 36, 5.2, tr9970x              2ff7e11     a (UB)   #110
                                         encode 1, par 1 (the 36 load and load_json files are the same 36)
  par leak             3 (3)             par                    hardening, 3 hosts        497303d     b        e9845b6
  msan OOM             2 (2)             load, load_json        msan run, gb10c           none [2]    c        -
  slow unit            1 (1)             encode                 smoke, gb10a              none [3]    c        -
  slow unit            1 (1)             par                    smoke, gb10b              none [4]    c        -
  slow unit            2 (2)             front-accept glm53     archived branch, gb10d    none [5]    c        -
  total                395 (263)         300 (harness, file) pairs

  failures: F1 / F3 'TOKS_E_NOMEM on a N-byte source' (the harness's invariant: load work bounded by the source); the
  piece-ends oracle 'piece ends decrease' (rstrip overlaps make ends go back, exactly as hf's offsets do); dg-smollm2
  'decode(encode(x)) != from_utf8_lossy(x)' (that file drops bytes); F2 UBSan 'store to misaligned address' at
  wp_api.c:57 (62 pairs) and wp.c:56 (12); par 'LeakSanitizer: detected memory leaks' (the pool's plan arrays live in
  its mmap'd arena, which LSan does not scan). [1] also on d8f3127 and 026e4a5. [2] peak RSS 35-81 MB on every ASan
  build, 72-145 MB under MSan on master: a long fork-mode run's RSS blamed on its last input. [3] dsv3, a 10 KB run of
  e / t letters (one giant piece): 53-55 ns/B, flat from 11 KB to 710 KB, 7.5-9.2x the en text at the same size
  (docs/hardening.md §2). [4] uni_bgem3, 'trailing ' tiled to 16-144 KiB by the harness: 2.6 ns/B, 0.6x en; the par
  harness's battery under ASan. [5] 7.0-9.9 ns/B, 1.0-1.9x en, flat from 1.8 KB to 635 KB on master.

  regress files: F1 regress/load/nomem-vocab-holes-dup-keys.json, regress/load_json/nomem-vocab-holes.json; F3 the 19
  regress/load*/nomem-* files; F2 regress/encode/misaligned-out-wordpiece.bin, and new: regress/par/
  misaligned-out-wordpiece-b1b77507.bin (the same 19 bytes through toks_par), regress/load_json/
  misaligned-wp-out-58bf6732.json (wp_api.c:57 through the load battery), regress/load_json/
  misaligned-wp-ids-b4e0f0b6.json (wp.c:56); new for the harness fixes: regress/load_json/
  oracle-piece-ends-19a40bb8.json, regress/encode/oracle-dg-smollm2-446aec98.bin, regress/par/
  leak-pool-arena-09cdc04e.bin.

  (a) a real bug, fixed before the triage; (b) the harness's own assumption, fixed in the harness; (c) a cost or an
  artefact, nothing to fix. The 2 + 1 + 1 + 1 + 1 new regress files load (expect.txt) or pass their harness on both
  isas (tests/fuzz/run.sh regress, 28 files).

  MSan on master (TOKS_TIER=scalar, the 297 pairs): one site, api.c:52 toks_scratch_init reading the caller's 64-byte
  header before it is written (the warm re-init's binding test): an unspecified value in C17, sound by the epoch tags,
  but SPEC §14.2's msan row stays red until it is marked (docs/hardening.md §4).


5.7 campaign 3: the arena probe (hardening lane, 2026-10-05, harness 7d8fac3 = master 0d5fbad + the arena guard)

load and load_json, 1800 s each, gb10d A725 cpus 0-3, nice 10, ASan + UBSan, fork mode, seeded with the shrink seeds
and the sweep seeds (tests/fuzz/seeds.py sweep: 6,400 per harness from 22 kinds of tokenizer file) and with the arena
probe in every object (docs/hardening.md §6):

  harness    execs     exec/s  cov     ft      corpus  cpu-h  findings
  load       588,352   287     14,001  63,337  2,025   0.50   0
  load_json  106,366    72     14,262  54,069  1,794   0.50   0

No crash, leak, timeout or oom; no UBSan line. The probe over the seeds, the repros and the grown corpora: 35 of 51
toks_ar_alloc sites reached, least slack 478 B (wp.c:436) in the table arenas and 20,976 B in the parse arena;
refusals only at config.c:242 / :243, the id-hole vocabularies #110 refuses with TOKS_E_LIMIT. tr9970x (x86-64, CCD0),
seeds and repros only: the same sites and the same least slack per site as gb10d's seeds-only pass, regress 28 / 28. The guard's mutant
(tests/hardening/ar_mutant.sh) passes the same replays: the guard is defense; nothing here needed it.


5.8 campaign 4: the T6 hours (2026-10-05, every public entry point, ASan + UBSan + LSan; in progress)

Toward T6's release budget (24 cpu-h per isa per entry point): arm64 on gb10d and gb10b (seven processes each, cpus
5-19, nice 10), x86-64 on tr9970x (CCD2 + CCD3, cpus 16-31 and 48-63, two processes per harness). Every run's log
starts with its ASAN_OPTIONS (detect_leaks=1) and its .meta names the build; `run.sh hours` gives the ledger per
harness and build. Seeds: `run.sh seeds` plus the grown load / load_json corpora of campaign 3; the arm64 hosts'
corpora merged at the 06:07Z and 07:01Z switches and seeded the x86-64 leg; at the 09:01Z / 09:09Z switch each state
dir kept its own.

  build     what it is                                                    arm64 runs          x86-64 runs
  fe94cde   master (stream hold, lookups), the harnesses as of 5.7        05:10-06:07Z        -
  28ba09a   + the pool's root region (below), par only                    05:47-06:07Z        -
  e8b634a   0.3.0's FREEZE-2 (bound) + the harness additions below but    06:07-07:01Z        -
            the misaligned offs
  ee6efde   0.3.0's FREEZE-4 245cc5c + all the harness additions below    07:01-09:01Z        07:52-09:09Z
  f09600b   0.3.0's final code d56a5c1 (#248: F4's fix) + the same         09:01Z-             09:09Z-
            harnesses

The harness additions: toks_stream_hold in the stream oracle (a caller's hold from the start and moved between
calls, toks.h's TOKS_E_LIMIT recovery, the hold's bytes in the atomicity checks); toks_token_to_id / toks_id_flags
on the decode harness's ids and in the load battery; encode's count against toks_encode_bound; toks_load_opts'
refusals from an opts block of exactly its size bytes; toks_split_points' offs at any alignment and a cap below
n_want - 1; the tiktoken readers (a load input can be a kimi or qwen model directory, seeds.py tiktoken) and kimik3
as the 17th pin (the kimi wrapper's run cuts, which no harness reached: tiktoken.c and api.c's run_cuts were at 0%
in the corpora's coverage). Each new oracle was replayed over the grown corpora before it joined (no report), and
the ones that check a fixed bug were run against the code before the fix:

  - toks_load_opts with size != sizeof(toks_load_opts): the loader read o->diag past the caller's struct (ASan
    heap-buffer-overflow READ in fail <- check_opts, load.c). Fixed in master (#238); the opts oracle stops the load
    seeds replay in 9 s on the code before it, and replays clean after.
  - toks_split_points stored each cut through the caller's uint64_t pointer (UBSan 'store to misaligned address',
    split.c, found by the toks.h clause audit, fixed in #241); the misaligned-offs oracle reports it on the par seeds
    with the code before the fix, nothing after.

Findings of the campaign so far: the par harness's runs stopped in their first minute on both arm64 hosts with
LeakSanitizer 'Direct leak' reports of the plan arrays toks_par mallocs. (b) harness: a pool keeps its state, and its
pointers to those arrays, in mmap'd pages of its own, which LSan never scans, so libFuzzer's per-input leak check
saw them unreferenced (the exit-time fix e9845b6 did not cover it); toks_par_destroy frees all three. The harness
registers the pool's state as an LSan root region while the pool lives (ASan builds only): the par seeds with
per-input leak detection, exit 77 before, exit 0 after.

The load_json harness on tr9970x (build ee6efde, 08:16Z, 0.4 cpu-h into the x86-64 leg) found F4 (5.3): UBSan
'index 32 out of bounds for type uint32_t[32]' at unigram.c:571, wrong ids past the 32nd on a release build for a
byte-fallback Unigram without a U+2581 piece; fixed in #248, repro regress/load_json/ubsan-uni-resolve-3fcbffd5.json.
A second file of the same site came at 08:44Z (fuzz-a); both pass on the fix, and both are kept in the state dirs'
triaged/load_json. Every process switched to f09600b after `run.sh regress` (29 / 29) and a replay of each harness's
corpus and seeds through the new binaries (no report) on each host.
