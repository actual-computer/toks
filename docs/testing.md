toks testing
============

`make test` is the gate on every host: docs/ci.md says where CI runs it, docs/machines.md names the lab machines.
docs/hardening.md lists the safety nets beyond it and docs/fuzz.md the fuzzing. This file holds the test geometries
that need a build of their own.


1. the guard geometry: make test-guard
--------------------------------------

The kernel geometry (SPEC §14.3, docs/kernels.md §1) puts a text buffer's start and end flush against a no-access
page, so a kernel that reads one byte outside [text, text + len) faults. Everything else a call touches sits in
mapped memory with neighbours:

- a context's tables are carved one after another from one block per builder (toks_plat_arena);
- a scratch is one caller buffer carved into regions with nothing between them.

An overrun from one table into the next, or from work into the bounce, reads or writes mapped bytes. ASan does not
see it (the blocks are mmap'd, the scratch is the caller's), and neither does the kernel geometry. The guard
geometry applies that geometry to every table and every scratch region, and checks the tables' placement.

    make -j test-guard                     run 1 and run 2, the native tier (TOKS_TIER selects, as for make test)
    TOKS_TIER=scalar make -j test-guard    the c twins
    tools/remote.sh <host> 'taskset -c 10-14 make -j5 test-guard'      a gb10's A725 cores (tr9970x: CCD0, 0-7,32-39)
    sh tests/common/guard_mutant.sh -j8    the teeth: three planted bugs, each must be caught as 1.2 says

Each run builds the library and every test program in its own directory (build/<os>-<isa>-guard1, -guard2) with
-DTOKS_GUARD=1 or 2. It then runs every test program as make test does, and one line says how each run went. The
object audits and the size budget stay with make test, since they read the shipped objects. CI runs both tiers after
make test (.github/workflows/test.yml). Posix only (mmap, mprotect).

What changes in that build (src/core/core.h; the hooks are tests/common/guard.c):

  tables     Every table pointer a builder forms comes from toks_tab(block, o, n, x) or toks_tab_ar(arena, n,
             align, x).
             - In the guard build each table is a mapping of its own, with a no-access page on each side. It is kept
               under the block it came from, and freed with it (toks_tab_free).
             - The guard also records where the shipped build would have put it. Once its builder has taken every
               table, toks_tab_seal checks that placement: every table lies inside its block and no two overlap. A
               table's own pages cannot show an offset that runs past the block or into the next table; this can.
               Then it seals the block no-access, so a pointer that bypassed toks_tab faults on first use.
             - tok_bytes and the vocabulary's pool are sized by a bound. toks_tab_fit moves their contents to a table
               of their exact length, so the no-access page sits at the contents' end.
  regions    Every scratch region pointer is formed in its shipped form (base + off) and passed through
             toks_scr_p, which translates it.
             - In the guard build toks_scratch_init maps every region on its own pages: ends, the short cache, the
               long cache's buckets and arena, the memo, work, extra (the generic engine's lists), the bounce and the
               norm region. Wordpiece's work and extra are one region, as wp_api.c uses them; unigram's work, extra
               and bounce are one, as uni_api.c does.
             - Each region is exactly what its readers may touch: the bounce is the 4 (tmax + 4) bytes K5 may write,
               the norm region x max_len. init's zeroing goes through toks_scr_zero, which fails on a byte that is in
               no region.
             - The header stays in the caller's buffer, followed by the table of regions that toks_scr_p reads.
             - A re-init with the same layout keeps the regions, so the epoch path runs. Fresh regions start as 0xA5
               junk rather than zeros: init owes every zero a reader relies on.
  run 1      A table's end plus its declared pad, or a region's end, is flush against the no-access page after it. Its
             start is aligned to the declared alignment only.
  run 2      Its start is flush against the no-access page before it. A zero-length table or region still gets one
             body page, so in run 2 its pointer is readable for a page; in run 1 it sits on the no-access page.

So a byte read or written outside a table or a region faults on first touch, from C or from asm, in one run or the
other, and a table placed outside its block or over another stops the load.

The shipped build. Every hook is the expression it stands for (TOKS_GUARD_HOOK in core.h: block + o, toks_ar_alloc,
p, memset), and no production .c names TOKS_GUARD. The per-call code is the same machine code as without the hooks:
compare the disassembly of api.o, gen.o, uni_api.o, wp_api.o, stream.o and the kernels' objects with master's.

The extents. layout.h declares every table's extent next to toks_tables, TOKS_X_<TABLE> = { pad, align }: how far
past its end a reader may read (its builder reserves the pad) and the alignment a reader may assume. A kernel reads
a table only inside [p, p + n + pad) (docs/kernels.md §1). Every declaration starts at pad 0 and its element's
alignment. A read that leans on undeclared padding faults in run 1 and is a finding: fixed, or declared with its
reason, each in its own commit, listed in 1.2.

tests/c/test_guard.c checks the geometry itself. For a context of each family:

- every table pointer the context holds is the first byte of a guard table;
- every table and region is probed one byte past its end (run 1) or one byte before its start (run 2), in a child
  process that catches its own fault (no core, no crash reporter), and must fault;
- the kernels run at the tables' ends, on the tier the build binds: K1 on the added token whose bytes end add_bytes,
  whole and a byte short; K5 / K6 on the token whose bytes end tok_bytes; K5 on a key in words' last bucket;
- an unload unmaps the context's tables.

The parity driver under the geometry, on the light sample tools/release/rc_host.sh runs (its case files are
build/rc/cases/<target>-{ids,pieces}.jsonl):

    make GUARD=1 BUILD_DIR=build/linux-arm64-guard1 build/linux-arm64-guard1/libtoks.a
    clang -std=c17 -O2 -DTOKS_GUARD=1 -Iinclude -Isrc/core -Isrc/platform -Itests/common tests/driver/toks_driver.c \
        tests/common/guard.c build/linux-arm64-guard1/libtoks.a -lpthread -o build/linux-arm64-guard1/toks_driver
    sh tests/parity/shards.sh <tokenizer> <cases>.jsonl build/linux-arm64-guard1/toks_driver <out> <shards>

Not covered here:

- the generated tables compiled into the binary (src/gen), whose bounds ASan's global redzones see;
- the generic engine's program and the context struct, which are malloc'd and so ASan's;
- the text and the caller's output (the kernel geometry and the abi tests);
- the sub-areas a family carves inside one region (wordpiece's pieces, copy and norm; unigram's three areas; the
  memo's ring);
- the slack compile.c's added-token candidate list reserves past what it uses (a bound);
- the scratch header;
- test_spm's moves, a scratch moved between a byte-level and an spm context: the two families' caches alias in the
  shipped layout only, so the guard build skips it.


1.1 receipts
------------

At d0489c5 (master 6d45fa9 merged), untimed, each host on its gate cores. "pass" is every program exit 0 with no fault;
load is the 1-minute average before -> after:

  host       cores            step                                     auto            scalar          load
  gb10c      A725 10-14       make test (shipped)                      44 pass         44 pass         1.08 -> 6.17
                              make test-guard, run 1 / run 2           88 pass         88 pass         6.17 -> 7.50
                              tests/common/guard_mutant.sh             9 of 9 outcomes as 1.2 says     7.50 -> 2.92
                              light parity, run 1 / run 2              42 / 42 PASS    42 / 42 PASS    2.92 -> 4.26
  tr9970x    CCD0 0-7,32-39   make test (shipped)                      44 pass         44 pass         2.86 -> 10.67
                              make test-guard, run 1 / run 2           88 pass         88 pass         10.67 -> 13.95
                              tests/common/guard_mutant.sh             9 of 9 outcomes as 1.2 says     13.95 -> 13.42
                              light parity, run 1 / run 2              42 / 42 PASS    42 / 42 PASS    13.42 -> 9.52

test_guard is 541 checks with 0 failures in every run and tier. The developer laptop (macOS arm64, 16 KiB pages,
not a receipt host) passes make test in both tiers and make test-guard (neon) at the same tree. CI runs both tiers
of make test-guard on 4-vcpu runners: 4 min 8 s on linux x86-64 and 6 min 41 s on linux arm64, after make test.

The light parity sample is the 0.3.0 rc's case set (tools/release/rc_host.sh): gen_cases.py --quick's every 7th
encode / pieces case and every decode and stream case, plus gen_stream.py's adversarial streams, for gpt2 llama3
glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4, ids and pieces each; and kimik3 through run_kimi.py
(25,655 texts, 8,458,949 ids). Each (run, tier) makes 3,261,295 comparisons with hf: 13,045,180 per host, 0 failed.


1.2 findings
------------

None. make test, test_guard and the parity sample ran with every extent at pad 0 and its element's alignment, and no
read or write on either isa or tier reached past a table or region in run 1 or before one in run 2; every builder's
placement passed the seal.

The teeth (tests/common/guard_mutant.sh: each mutant on a copy of the tree, shipped and both runs; the same nine
outcomes on gb10c and tr9970x):

  extent     toks_compile_cls_flags reads cls_ascii[0..128], one byte past the 128-byte table. Shipped, test_e2e
             passes (163,731 checks). Run 1 faults on that read at load, in toks_compile_cls_flags. Run 2 passes,
             since the overrun is at the end.
  bound      decode's block one byte short of its two tables (stream.c dec_block_bytes), so dec_len's last byte lies
             in the page's slack. Shipped, test_stream passes. Both runs stop at load: "a table of 299 bytes at
             offset 4784 runs past its block of 5082".
  overlap    dec_len placed one byte early, over the last slot's 16th byte. Shipped, test_stream sees 3 wrong
             decodes. Both runs stop at load: "tables overlap in their block: [0, +4784) and [4783, +299)".
