hardening toks
==============

SPEC §0.4 (safety: no input within §8.3's limits crashes toks or makes it touch memory outside §7.2's regions), §7.3
(measured adversarial ceilings), §9 (the strict c subset), §4.1 (nothing allocates, syscalls or recurses after load),
§14.2 (the sanitizer matrix), T6, T7, T9. This file says which safety nets exist, what each one checks, how to run it,
what it found, and what stays open. The fuzzing campaigns and their triage are in docs/fuzz.md; the Frama-C Eva
package is docs/proof.md.


1. the nets
-----------

  net                                what it checks                                           where it runs
  tests/abi/cf_audit.py (§3)         §9 / §4.1 on the built objects: no indirect call, no      make test (asmcheck)
                                     address-taken function, libc = memcpy / memset / memcmp,
                                     no recursion, no vla, nothing after load reaching libc or
                                     the platform layer
  tests/c/test_stall.c (§2)          no adversarial class grows superlinearly; each path       make test
                                     within its bound x the en text
  tests/fuzz/ (docs/fuzz.md)         7 libFuzzer harnesses, asan + ubsan (msan: c twins only)  lab hosts
  tests/fuzz/regress/ (28 files)     every finding's repro through its harness; the load       tests/fuzz/run.sh regress on lab
                                     ones also through test_load (expect.txt)                  hosts; test_load in make test
  tests/c/lines.inc (§5)             the tests' data readers fail by name on a byte they do    make test
                                     not expect, never loop
  tests/hardening/ab_ids.sh          ids / pieces digests of n random texts, old library vs    by hand, for every
    + idsdiff.c                      new: the receipt of a semantics-preserving fix            semantics-preserving fix
  tests/hardening/textcost.c         one text's cold cost vs the en text at the same size      by hand (slow units, §2)
  tests/hardening/crlf_proof.py      lines.inc's teeth: every guarded file in CRLF fails fast  by hand
  tests/fuzz/arprobe.{h,c} (§6)      per toks_ar_alloc site of the fuzz build: calls, refusals, tests/fuzz/run.sh probe
                                     least slack (bound - end)                                 on lab hosts
  tests/hardening/ar_mutant.sh (§6)  the arena guard taken out: the corpora still pass          by hand
  tests/hardening/bcmp_same.sh (§3)  -fno-builtin-bcmp moves no instruction                   by hand
  tests/proof/ (docs/proof.md)       Frama-C Eva over the readers and the table compiler       make proof (lab hosts)


2. the stall policy (SPEC §7.3, T7)
-----------------------------------

Every admitted input encodes exactly and there is no work budget; what bounds the time is the code. Three tools:

  - tools/bench/stall.c: the audit. Adversarial classes per path (K1 added tokens, K3 templates, K5 / K6 incl. the
    long-piece heap, whole-segment spm bpe, K7, unigram's Viterbi, wordpiece's greedy match, the normalizers, the
    generic engine), each at 4 KiB / 64 KiB / 1 MiB in a child with a timeout, against the en text.
  - tests/c/test_stall.c: the regression in make test. 16 classes at 8 / 32 / 128 KiB, best of 3 cold calls. A class
    fails when it grows x8 or more per 4x the bytes twice (exponent >= 1.5: linear is x4, the bpe heap's n log n
    ~x4.5-5, the quadratic walks it was written for x16), or costs more than its path's bound x the pseudo-en text's
    ns per byte at 128 KiB: byte-level bpe 60x, sentencepiece-style bpe 40x, unigram and wordpiece 20x (about 3x the
    worst class measured at 64 KiB: it catches a new stall, not noise).
  - tests/hardening/textcost.c: one text, e.g. a fuzz slow unit, tiled at 4x steps, against the en text tiled to the
    same size, cold, best of 5 on one pinned core.

SPEC §7.3's ceiling (for inputs <= 1 MiB the worst class within 10x the en-prose median; superlinear classes publish
their curve and an absolute ceiling) is NOT met by every class today, and test_stall's bounds were set to catch new
stalls, not to enforce it. What is measured (cold, gb10d Cortex-A725 cpu 3, master ac14d02, load 0.1-0.5; en = the
bench corpus's gut-en-2701.txt tiled to the same size; ids sha not compared: a cost, not a speed claim):

  class                                tokenizer  4 KiB          16 KiB         64 KiB         256 KiB        1 MiB
  one piece of random a-z letters      llama3     51.6 (8.8x)    65.2 (10.7x)   84.5 (11.4x)   129.7 (18.3x)  208.1 (31.0x)
  one piece of random a-z letters      dsv3       49.4 (7.0x)    64.2 (9.6x)    81.6 (10.7x)   120.0 (16.9x)  201.8 (30.4x)
  ns per byte (x en); time grows x5.1, x5.2, x6.1, x6.4 per 4x the bytes on llama3 (exponent ~1.25 over 4 KiB .. 1 MiB):
  the long-piece heap's n log n plus a working set that outgrows the core's L2. Under test_stall's exponent rule; over
  §7.3's 10x from 16 KiB. At 1 MiB the call takes 218 ms (llama3) / 212 ms (dsv3): the absolute ceiling §7.3 asks for
  at this size. One piece of 709,504 bytes: e / t only 98-128 ns/B (16-17x), all 'a' 116-136 ns/B (18-20x), a-z
  130-198 ns/B (21-27x) on dsv3, llama3, gpt2, o200k.

  the fuzz slow units (docs/fuzz.md 5.6), the same rig:
  dsv3, a 10 KB run of e / t letters with marks (gb10a smoke)   53-55 ns/B flat from 11 KB to 710 KB = 7.5-9.2x en
  glm53, the front-accept fuzzer's two units                    7.0-9.9 ns/B flat from 1.8 KB to 635 KB = 1.0-1.9x en
  uni_bgem3, 'trailing ' tiled (par harness)                    2.6 ns/B = 0.6x en
  wp-bert-uncased, '▁▁a▁b' tiled (the F2 file)                  2.8 ns/B = 0.6x en (one word over 100 chars: [UNK])

test_stall's own report on the control-plane mac (pseudo-en, not the bench text, so the ratios run high): worst 44x
llama3 ws_tab, 42x qwen38 ws_nl, 30x o200k '<', 18x gemma4 ws_tab, 16x gpt2 digits_rand, 10x mistral-v0.3 ws_space;
worst growth x7.3 per 4x (qwen38 cjk_rand). The giant piece belongs to K6's long path (cold-k6), the whitespace and
'<' floods to K3 / K1 and the driver; none of them is a stall by the exponent rule, and §7.3's 10x needs either those
paths or a measured exception per class in the release report.


3. SPEC §9 on the built objects (tests/abi/cf_audit.py)
-------------------------------------------------------

The source can be clean and the object not: clang 21's loop-idiom pass turned `while (key[kl] != 0) kl++`
(json.c toks_jv_get, json.h toks_jstr) into calls to libc's strlen in config.o, json.o and tiktoken.o, on Apple clang
and LLVM 21.1.8, both isas; -fno-builtin-strlen in CSTRICT removes them (and changes no other object but load.o's
frame layout, all load time). On linux the same compiler spells `memcmp(...) == 0` as bcmp, in 15 of the 49 objects
(the run-time c twins among them); -fno-builtin-bcmp keeps memcmp, and tests/hardening/bcmp_same.sh shows the flag
moves no instruction: the 49 objects built without and with it, disassembled, the first build's bcmp renamed memcmp,
are the same text (gb10d linux-arm64 and tr9970x linux-x86_64, LLVM 21.1.8; glibc's bcmp is memcmp's alias). Apple
clang never writes bcmp. So the check reads what links: every object under build/<os>-<isa>/obj (src/core,
src/gen, src/platform, src/par, the isa's asm) and the asmcheck objects of both isas in mach-o, elf and coff, through
llvm-objdump -d -r and llvm-nm -P, and fails make test on:

  R1  a call through a register or memory (no function pointers: §0.5, §9, §10.4). Darwin's stack probe (a frame of a
      page or more calls ___chkstk_darwin through x16) is the compiler's and is recognised. A jump through a table
      (a switch) is not a §9 item: counted per object, printed, not gated.
  R2  a function's address used other than by a branch: a pointer-sized relocation from data, or any relocation from
      a non-branch instruction, to a function's entry (§11: no address-taken function, no guard-cf table).
  R3  an external symbol of the c core (src/core, src/gen) other than memcpy, memset, memcmp, Darwin's memset
      lowerings (bzero, from the backend, which no flag removes; memset_pattern16) and the compiler's runtime (stack
      probe, stack protector). bcmp is not among them (-fno-builtin-bcmp).
  R4  a cycle in the c core's static call graph (no recursion; the json reader and the regex compiler use explicit
      bounded stacks).
  R5  a stack-pointer move by a register (a vla, alloca), other than a fixed frame's probe.
  R6  anything reachable from the after-load entry points (toks_encode, toks_pieces, toks_split_points, toks_decode,
      toks_token, toks_stream_*, toks_scratch_*, toks_get_info, toks_version) reaching an external symbol other than
      memcpy / memset / memcmp, or the platform layer (§4.1).
  R7  a byte of writable data in the c core (src/core, src/gen): a data, bss, common or thread-local section with
      anything in it (toks.h: no global is written; one loaded context serves every thread). Read-only data is not
      writable data: .rodata, mach-o's __const sections (__DATA,__const only by the loader's relocations) and elf's
      .data.rel.ro (a const table of pointers under -fPIC). A writable global in api.c is reported as R7 with its
      symbol and section (COMMON under Apple clang, __data when initialized).

The allow-list (in the script, each entry dated with its reason; an entry that matches nothing in a build is printed,
not failed):
  R2  thr_main (src/par): the thread entry toks_par hands pthread_create / CreateThread; toks_par is outside the proof
      boundary (§2.4) and the os calls it.
  R4  rx_node (compile.c, depth <= 255), parse_alt (gen.c, <= 32), cls_items (gen.c, <= 8), add <-> search (gen.c,
      2: no lookahead inside a lookahead), elem <-> piece (gen.c, <= 16 steps): bounded, and still violations of §9's
      letter (maintainers' ruling, 2026-10-05): the explicit-stack PR removes all five; until then each is a dated debt.
  R6  toks_scratch_init -> toks_plat_hint_huge (madvise): the first init of a scratch only, only when
      TOKS_SCRATCH_CACHE_MIB(n) != 0, never on the default scratch; a syscall after load, tied to decision 8.

Information it prints: jump tables per object (Apple clang 2, LLVM 21.1.8 arm64 2, x86-64 ~16: x86's threshold is
4 cases), and every frame of 4 KiB or more, marked when it is on the after-load path (§4).

Teeth: tests/abi/cf_teeth.c, compiled by the audit with the library's own flags, breaks each rule once (a call
through a pointer, an address in data, strlen, malloc, recursion, a vla, an after-load entry calling getenv, an
initialized and a zero-initialized writable global) and holds one jump table; the audit fails unless it reports
exactly those. It needs two LLVM tools, llvm-objdump and llvm-nm,
found beside CC, on PATH, or as the system's own when they are LLVM's (macos); CI's tools/ci/llvm.sh unpacks both
from the pinned LLVM 21.1.8. Without either it fails, never skips (asm_regs_audit.py's rule): a CI image that drops
one fails make test before any tier runs, as master's first run with the audit did until llvm.sh unpacked llvm-nm.

First run (master ac14d02 + -fno-builtin-strlen): 0 violations, 7 allowed, on macos-arm64 (Apple clang 21), gb10d
linux-arm64 and tr9970x linux-x86_64 (LLVM 21.1.8); 49 library objects, 274-275 functions, 135 reachable after load;
48 asmcheck objects (mach-o, elf, coff of both isas). The C objects in coff are built only on Windows
(tools/win/build.cmd); the script reads build\\windows-x86_64\\obj the same way.


4. open findings
----------------

  - §7.2 (the run-time chain <= 4 KiB): spm_c.c's model keeps a 5,280-byte frame (5,312 on x86-64) on the
    sentencepiece run path: toks_encode -> run -> run_seg -> run_text -> toks_spm_encode -> word_slow -> model. On
    macos every call to it goes through ___chkstk_darwin, which touches each page of the frame. No fix in this lane
    (families' file; the next spm card). Every other frame over 4 KiB is load time: toks_bpe_build 9.8 KiB,
    toks_pc_build 8.8 KiB, toks_spm_build 5.4 KiB, premerge_build 4.5 KiB, toks_uni_build 4.4 KiB, toks_pc_bytes
    4.4 KiB, toks_uni_load 4.1 KiB.
  - §14.2 msan: toks_scratch_init reads the caller's 64-byte header before writing it (api.c:52, the warm re-init's
    binding test): every first init of a fresh buffer reads indeterminate bytes. C17: an unspecified value; sound by
    the epoch tags; but no MSan user can run toks clean (open with the scratch code's owners: mark the read for MSan
    only, or a toks.h sentence that the header's bytes are zero on first use).
  - §9 recursion: the five cycles of §3, until the explicit-stack PR.
  - §7.3: the giant-piece curve of §2 (31x en at 1 MiB, exponent ~1.25), and test_stall's 16-44x classes.
  - §4.1: the madvise after load (decision 8).


5. the tests' data readers
--------------------------

A CRLF checkout (core.autocrlf=true) once made test_tiktoken spin 37 minutes on a CR its strtoul loop could not pass.
Every line-oriented data reader in tests/c now goes through tests/c/lines.inc: td_text / td_line refuse any control
byte but tab and newline, a missing final newline and a line longer than the reader's buffer; td_num fails where
strtoul does not move. Each prints `FAIL <file>:<line>: <what>, at byte 0xNN` and the reader stops. The files are read
in binary mode: the msvc ucrt's text mode would turn CRLF into LF and stop at 0x1A. tests/hardening/crlf_proof.py
rewrites each guarded file with CRLF ends and checks that its test fails in about a second, naming the byte 0x0d.


6. the arena guard and its probe
--------------------------------

Every load-time arena goes through one allocator, toks_ar_alloc (core.h): the parse arena (toks_plat_alloc'd;
toks_config_arena_bound(len) = 32 B a source byte + 65536 + toks_gen_max_bytes(), or toks_tiktoken_arena_bound for
a tiktoken model) and the table arenas (toks_plat_arena'd; each sized by its builder's *_bytes function). Since
2026-10-05 it refuses a request whose aligned start lies past the arena's end (p > len) before it takes len - p, so
every request lies inside its arena or returns NULL; before, such a start would have passed the size test through
the unsigned difference. Its 51 call sites map NULL to TOKS_E_NOMEM or TOKS_E_LIMIT, or skip an optional table
(bpe_build.c's premerge, the wordpiece tries).

The probe measures how close each site comes. tests/fuzz/arprobe.h is force-included into the fuzz build's src/core
objects (never the library's own build) and reports every request's site, bound, start and size to arprobe.c; per
site it keeps the calls, the refusals and the least slack seen, bound - (aligned start + size).
`tests/fuzz/run.sh probe` replays the load harnesses' corpora, seeds and regress repros and prints the table; a
refusal that ends as TOKS_E_NOMEM stops the replay (load.h's check). The seeds include tests/fuzz/seeds.py sweep: per
kind of tokenizer file, 64 consecutive vocabulary counts and an unused top-level key holding an array of 0, [], ""
or {} at 64 consecutive lengths up to the harnesses' 65536-byte limit (every residue of a 64-byte alignment).

The least slack per site on gb10d (linux-arm64) over the seeds, the regress repros and the corpora of the 30-minute
campaign of docs/fuzz.md §5.7 (load and load_json together); tr9970x (linux-x86_64), seeds and repros, gives the
same bytes wherever it reaches the same inputs. The nine smallest are all in the table arenas:

  site              what it allocates                bound     end    slack  the arena is sized by
  wp.c:436          wordpiece tries: terms         125,130  124,652     478  toks_wp_ctx_build: toks_wp_tables_bytes
  wp.c:435          wordpiece tries: cells          68,655   67,648   1,007    (the two hash tables, the keys, the
  wp.c:353          wordpiece whole-word table      60,380   50,304  10,076    whole-word table, 4 x 64) + the
  wp.c:299          wordpiece continuation table    60,380   33,920  26,460    tries' bytes (toks_wp_tries) +
  wp.c:298          wordpiece word table            60,380   17,536  42,844    sizeof(toks_wp_tables) + 128
  wp.c:296          wordpiece keys                  60,380    1,104  59,276
  wp.c:481          wordpiece tables struct         60,380      368  60,012
  bpe_build.c:451   bpe certified words            589,828  589,312     516  toks_bpe_tables_bytes: each table's
  bpe_build.c:208   bpe ascii premerge             551,888  535,040  16,848    bytes + 64 (bpe_build.c:49)

The parse arena's least: 20,976 B at config.c:242 (bound 456,856, end 435,880). Refusals: config.c:242 and :243
only (load 2, load_json 6), every one the id-hole vocabulary that #110 answers with TOKS_E_LIMIT, as designed
(regress/load_json/nomem-vocab-holes.json and campaign units like it). No site reached zero slack, in the seeds or in
the campaign: the guard is defense, not a fix. tests/hardening/ar_mutant.sh builds the harnesses with the guard taken
out and replays the same corpora and every regress repro: all pass (gb10d: both load harnesses' replays complete,
28 / 28 repros), so nothing on record depends on it.

The probe reaches 35 of the 51 sites. What it did not see, 16 sites, and why:

  tiktoken.c:95-97     toks_tiktoken_ranks: the rank file's bytes, its per-rank pointers and lengths
  tiktoken.c:129-131   toks_tiktoken_ranks: the vocabulary's strings, pointers and lengths
  tiktoken.c:155-157   toks_tiktoken_ranks: the merges' left / right / output arrays
  tiktoken.c:383-384   toks_tiktoken_kimi: the reserved specials and their names
  tiktoken.c:526-527   qwen: the specials and their names
  tiktoken.c:567       toks_tiktoken_parse: the added-token table
      the tiktoken reader runs only from toks_load on a model directory of three files (the ranks,
      tokenizer_config.json, the wrapper .py); the load harnesses feed one file or memory, and toks_load_mem_copy
      refuses a tiktoken source. No harness feeds that path.
  config.c:1248        the Precompiled normalizer's charsmap, base64-decoded (uni_norm_one)
  precompiled.c:295    toks_pc_build's tables (ar_take: stage1, stage2, the multi-char keys and values, the pool)
      every Precompiled charsmap on record is 316,720 bytes of base64, over the harnesses' 65,536-byte input limit,
      so no seed and no mutation carries one. No harness input reaches that path.
