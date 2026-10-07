proof toks
==========

SPEC §14.1 (the proof package, T4) asks for Frama-C Eva over the core's readers and table compiler, for every input
meeting the entry preconditions, with a declared boundary. Today the package covers the json reader; the other readers
and the table compiler are planned (1). This file says what is analyzed, under which assumptions, what is proven,
what is open and how to run it. The fuzzing campaigns are docs/fuzz.md; the other nets docs/hardening.md.


1. what is analyzed
-------------------

  job            entry (tests/proof/entries.c)          files                                   placements x models
  json-end       eva_json: toks_json_parse, toks_jv_get  src/core/json.c                         flush end   x lp64, llp64
  json-start     the same                                the same                                flush start x lp64, llp64
  config-end     eva_config: toks_config_parse           json.c config.c alloc.c precompiled.c   flush end   x lp64, llp64
                                                         segment.c norm.c gen.c api.c and the
                                                         generated tables they read
  config-start   the same                                the same                                flush start x lp64, llp64

make proof runs the gate: the json jobs (eva.sh's ALL), whose every open property the ledger classifies. The config
jobs run by name and are not in the gate yet (3: Eva stops at the regex compiler's recursion).

Each entry gives its function a source of len bytes, len anywhere in [1, 256 MiB] (SPEC §8.3), every byte unknown
(Frama_C_make_unknown), placed flush against the end (EVA_PLACE=0) or the start (EVA_PLACE=1) of a 256 MiB block: a
read past the source's end leaves the block in the first run, a read before its start in the second. The parse arena
is the one load.c gives (toks_config_arena_bound(len)). The data models are SPEC §14.1's two: lp64 (Frama-C's
gcc_x86_64) and llp64 (msvc_x86_64).

Not analyzed yet, in SPEC §14.1's scope for this package: the tiktoken reader (tiktoken.c), the table compiler
(compile.c, bpe_build.c, spm_build.c), the wordpiece and unigram builders (wp.c, unigram.c, uni_api.c), the
normalizer table builders beyond what toks_config_parse reaches, and load.c's driver around them. Each becomes a job
of eva.sh with its entry in entries.c.


2. assumptions (the declared boundary)
-------------------------------------

  - Frama-C's libc contracts for memcpy, memset, memcmp, malloc and free (its share/libc, ACSL): Eva checks every call
    against them; their postconditions are assumed ("Considered valid").
  - the platform layer (tests/proof/plat.c): each toks_plat_* function has exactly the contract core.h states, with
    every allowed outcome, failure included (an allocation may return NULL anywhere).
  - the arena (tests/proof/prelude.h): toks_ar_alloc is analyzed under its contract, NULL or a fresh block of n bytes,
    disjoint from every other; each block is its own exact-size allocation, so an access past a block's end is an
    alarm even where the real arena would hand it the next block's bytes (stricter than the arena). core.h's
    allocator meets the contract: it returns a + p with p >= pos and p + n <= len, or NULL (core.h:71-77; the
    `p > a->len` refusal, docs/hardening.md §6).
  - the entry preconditions of SPEC §4.1: data points to len readable bytes, len <= 256 MiB. len = 0 is outside the
    analyzed range: load.c refuses an empty source (load.c:324) and the file reader an empty file (file.c:121), so no
    load reaches the parser with one; tiktoken.c's json config path, which calls the parser too, is not analyzed yet.
  - the data models: both of SPEC §14.1's (lp64, llp64) are x86_64 machdeps, where char is signed; on aarch64 linux it
    is unsigned. json.c never reads a plain char as a value: it reads the source through const uint8_t * (json.c:197)
    and compares a key, its one char *, with memcmp (json.c:317), so the signedness reaches no property.
  - the compiler: C17 as Frama-C's kernel reads it, -fwrapv and -fno-strict-aliasing as CSTRICT builds it (signed
    overflow is still reported as an alarm), and the asm kernels' contracts of SPEC §10.3 (no job here reaches a
    kernel: the readers and the compiler are C only).


3. what is proven, what is open
-------------------------------

Every property of a report is Valid (proven by Eva), Considered valid (an assumption of §2), Dead (unreachable) or
open. Every open property is in tests/proof/alarms.md, in one of SPEC §14.1's two categories: discharged by a separate
proof (counted as proven) or accepted by review (an open, unproven obligation, with the argument the review checked).
tests/proof/ledger.py fails on any open property the ledger does not classify. The counts of the last run are below;
the ledger names each class and its argument.

  job.model          open  proven  review  unclassified  stale  time (gb10d A725 cores 0 and 10, Frama-C 33.0, the
                                                                        commit that wrote this table, load 8.2 -> 10.5
                                                                        beside the fuzz campaign)
  json-end.lp64        80       0      80             0      0  221 s
  json-end.llp64       80       0      80             0      0  222 s
  json-start.lp64      79       0      79             0      0  222 s
  json-start.llp64     79       0      79             0      0  218 s
  config-end.lp64      at bd80b7d: stopped after 2,492 s at gen.c's parse_alt (below); 1,194 alarms reported before
                       it, not classified; not re-run since

Every json job reaches 17 of the 69 functions its files define and 754 of those 17's 759 statements (Eva's summary,
quoted per job by eva.sh). Each json report's open properties include three
of Frama-C's share (memcmp's preconditions in its string.h, N3).

The config job reaches the generic regex compiler (gen.c), whose parse_alt calls itself once per group (depth <= 32
by r->depth): Eva stops there ("Recursive call to parse_alt without assigns clause"). parse_alt is the first of the
recursions cf_audit R4 allows (five: parse_alt, cls_items, add / search and elem / piece in gen.c, rx_node in
compile.c), each a debt of the explicit-stack change R4's list names. Eva needs a specification for each one it
reaches (an assigns clause at least) or -eva-unroll-recursive-calls past every depth the code allows; the
explicit-stack change removes the question. Until one of the two lands the config jobs stay out of the gate. (Frama-C 33.0's C parser also refused config.c's one universal character name, a \u2581 in
a diag string, which is C17; the diag now says U+2581.)

The json reader's open properties fall in three groups, none of them a defect found: the source's end (Eva cannot
add the offset of data and len back together: S1-S5), the decoded string's block (pass 2 writes at most what pass 1
measured: S6-S11), and the frame stack and node tree (an invariant of the parser's phases and of jv_new's zeroing:
F1-F5, N1-N2); and the number parse's range (S12). Earlier rounds of this package found and fixed two defects in the
json reader, each pinned by tests/c/test_proof.c: J1, a json integer past 2^63 - 1 (now num_float), and J2, tail checks
that formed a pointer past one-past-the-end (now length comparisons).


4. running it
-------------

On a lab host (never the control-plane mac), once per host: tests/proof/install.sh puts opam 2.6.0, OCaml 4.14.2 and
Frama-C 33.0 under ~/toks-ci/opam (no sudo: the gmp headers come from the distribution's libgmp-dev deb, unpacked).
Then, from the repository root:

  tools/remote.sh <gb10> 'nice -n 10 taskset -c 10-13 sh tests/proof/install.sh'
  tools/remote.sh <gb10> 'nice -n 10 taskset -c 10-13 make proof'                       the gate (eva.sh's ALL)
  tools/remote.sh <gb10> 'nice -n 10 taskset -c 10-13 bash tests/proof/eva.sh config-end' one job
  tools/remote.sh <gb10> 'python3 tests/proof/ledger.py build/proof/eva'                the ledger over a directory

Reports land in build/proof/eva/<job>.<model>.{log,csv}; each log starts with the Frama-C command line and EVA_EXTRA,
and eva.sh prints versions.txt (install.sh: the tools, the opam repository, every package of the switch) and, per
job, Eva's coverage lines. JOBS (default 4) bounds the jobs at once; MODELS picks the data models; EVA_OUT another
directory; EVA_EXTRA passes Frama-C options (a precision study, never a check removed). A job fails when Frama-C errs,
writes no report, runs past 2 h, or reports a NON TERMINATING FUNCTION (the properties after a bottom state are Dead,
which the ledger would close) or a degeneration; eva.sh exits 1 then, or when the ledger fails: an open property no
class takes, or a stale class (its jobs match a report it read and it matched nothing there). install.sh checks the
opam binary against the release's sha256; the opam repository is not pinned (the installs so far read
opam.ocaml.org's 2026-10-04 snapshot, and versions.txt lists every package at its version).
