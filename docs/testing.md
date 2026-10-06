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
object audits and the size budget stay with make test, since they read the shipped objects. Posix only (mmap,
mprotect).

CI: test.yml runs both tiers after make test on every pull request, on linux x86-64 and arm64. arm64's job has a
16-minute timeout: make test takes about 4 minutes and make test-guard about 8 (12 min 10 s at adb55f8). nightly.yml
runs each tier again after that tier's parity job, on both isas.

When it fails. A program that reads or writes outside a table or region dies on the fault: make test-guard prints its
section with the shell's message where the program's last line would be, its .rc holds 139 (SIGSEGV; 138, SIGBUS, on
macOS), and the run's summary line names it. This is make test-guard on aimax395b with guard_mutant.sh's scratch
mutant (a write past the work region into the bounce):

    == build/linux-x86_64-guard1/tests/test_e2e
    Segmentation fault (core dumped)
    ...
    make guard-run (auto): FAIL: test_alloc test_api test_bound test_breadth test_e2e test_guard test_kimi ...
    make guard-run (auto): FAIL: test_e2e test_state test_state_hash
    make test-guard (auto): run 1 FAIL, run 2 FAIL

The site is the top of the program's backtrace in that build directory: gdb -batch -ex run -ex bt <program> (lldb on
macOS). For the scratch mutant it is k5_run, called from toks_round. The fault address lies on the no-access page
right after the table or region in run 1, right before it in run 2. A placement the seal refuses stops the load
instead: the program prints the seal's line and aborts, and its .rc holds 134:

    guard: tables overlap in their block: [0, +4784) and [4783, +299)
    Aborted (core dumped)

What changes in that build (src/core/core.h; the hooks are tests/common/guard.c):

  tables     Every table pointer a builder forms comes from toks_tab(block, o, n, x) or toks_tab_ar(arena, n,
             align, x).
             - In the guard build each table is a mapping of its own, with a no-access page on each side. It is kept
               under the block it came from, and freed with it: toks_tab_free releases a block's tables, and so does
               mem.c's toks_plat_arena_free, since a test's stand-in free skips mem.c and toks_par's blocks skip
               toks_tab_free (the second release finds nothing).
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
- an unload unmaps the context's tables;
- the seal's own check, on blocks of its own: an empty table may share its offset with the next, in either order; a
  table past the block's end, across it, or over another stops the load.

The parity driver under the geometry, on the light sample tools/release/rc_host.sh runs (its case files are
build/rc/cases/<target>-{ids,pieces}.jsonl):

    make GUARD=1 BUILD_DIR=build/linux-arm64-guard1 build/linux-arm64-guard1/libtoks.a
    clang -std=c17 -O2 -DTOKS_GUARD=1 -Iinclude -Isrc/core -Isrc/platform -Itests/common tests/driver/toks_driver.c \
        tests/common/guard.c build/linux-arm64-guard1/libtoks.a -lpthread -o build/linux-arm64-guard1/toks_driver
    sh tests/parity/shards.sh <tokenizer> <cases>.jsonl build/linux-arm64-guard1/toks_driver <out> <shards>

Not covered here:

- the generated tables compiled into the binary (src/gen), whose bounds ASan's global redzones see;
- the generic engine's program and the context struct, which are malloc'd and so ASan's;
- toks_par's own blocks: the pool (its struct, slots and threads) and its stage array are toks_plat_arena blocks
  (src/par/par.c), not carved through toks_tab and never sealed, so neither the guard nor ASan sees an overrun inside
  them;
- the text and the caller's output (the kernel geometry and the abi tests);
- the sub-areas a family carves inside one region (wordpiece's pieces, copy and norm; unigram's three areas; the
  memo's ring);
- the slack compile.c's added-token candidate list reserves past what it uses (a bound);
- the scratch header;
- test_spm's moves, a scratch moved between a byte-level and an spm context: the two families' caches alias in the
  shipped layout only, so the guard build skips it.


1.1 receipts
------------

At adb55f8 (master 7a80008 merged), untimed, each host on its gate cores. "pass" is every program exit 0 with no
fault; load is the 1-minute average before -> after:

  host         cores            step                                   auto            scalar          load
  gb10c        A725 10-14       make test (shipped)                    45 pass         45 pass         0.25 -> 5.87
                                make test-guard, run 1 / run 2         90 pass         90 pass         5.87 -> 6.20
                                tests/common/guard_mutant.sh           9 of 9 outcomes as 1.2 says     6.20 -> 3.18
                                light parity, run 1 + run 2            21 + 21 PASS    21 + 21 PASS    3.18 -> 3.31
  aimax395b    CCD0 0-7,16-23   make test (shipped)                    45 pass         45 pass         0.00 -> 7.07
                                make test-guard, run 1 / run 2         90 pass         90 pass         7.07 -> 9.67
                                tests/common/guard_mutant.sh           9 of 9 outcomes as 1.2 says     9.67 -> 4.68
                                light parity, run 1 + run 2            21 + 21 PASS    21 + 21 PASS    4.68 -> 6.91

test_guard is 573 checks with 0 failures in every run and tier. tr9970x (CCD0 0-7,32-39) passed the same matrix at
6fa887d, before 7a80008. The developer laptop (macOS arm64, 16 KiB pages, not a receipt host) passed make test in both
tiers and make test-guard (neon) at db63557, and test_guard, test_primitives, test_load, test_vocab and test_stream in
runs 1 and 2 at adb55f8.

The light parity sample is the 0.3.0 rc's case set (tools/release/rc_host.sh): gen_cases.py --quick's every 7th
encode / pieces case and every decode and stream case, plus gen_stream.py's adversarial streams, for gpt2 llama3
glm53 qwen38 o200k gemma4 nemotron3-4b llama4 minimaxm2 dsv4, ids and pieces each; and kimik3 through run_kimi.py
(25,655 texts, 8,458,949 ids). That is 21 suites, each run under 2 guard runs x 2 tiers: 84 suite runs per host, all
PASS, 3,261,295 comparisons with hf per (run, tier), 13,045,180 per host, 0 failed.


1.2 findings
------------

None in the library. make test, test_guard and the parity sample ran with every extent at pad 0 and its element's
alignment, and no read or write on either isa or tier reached past a table or region in run 1 or before one in run 2;
every builder's placement passed the seal.

Two in the geometry itself, both fixed:

  bounds     tok_bytes and the vocabulary's pool were guard tables of their bound, not of their contents: 13 B
             (gpt2) to 12,876 B (nemotron3-4b) of mapped slack past the last token's bytes. toks_tab_fit.
  the seal   CI's linux x86-64 job, scalar tier, run 1: test_bound stopped at load with "tables overlap in their
             block: [2048, +40) and [2048, +0)". An empty table shares its offset with the next one, and the check
             compared neighbours in an order that test_bound's workers (loading and freeing in parallel) make
             arbitrary. Empty tables now take no part in the overlap check. The same reading showed a hole: a table
             wholly past its block's end passed. test_guard's seal cases fail twice with the old check, once for
             each.

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
