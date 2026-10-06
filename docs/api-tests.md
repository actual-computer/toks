# toks.h, clause by clause: the test behind every promise (T11)

`include/toks.h` as this branch leaves it (master 245cc5c's FREEZE-2 header, plus two doc comments the audit asked
for: the capacity block's phase terms and unigram's 127-byte pieces), read sentence by sentence. A row is one promise
of behaviour: a return value, an error code, a limit, a lifetime rule, a no-op, a tightness or a flag's meaning. Its
test is the check that fails when the promise is broken (`file:line`, the CHECK), or `by inspection: <where>` where
no test can observe it, or GAP. A caller's obligations (preconditions) and the header's measurements (speed, bytes
per id) are listed under their section and not counted. Line numbers are this branch's (a script checks that every
quoted clause lies in its row's toks.h lines, and that every cited `file:line` exists).

abi 0.4 (PR#13, the hf and tiktoken primitives) added rows EN11-EN14, TP1-TP8, SP14, DE6-DE9, VO21, AD1-AD9 and
IN14-IN16, each against hf tokenizers 0.23.2 or tiktoken 0.14.0 through `tests/c/test_primitives.c`, and moved the
line numbers from toks.h:137 on to that header's.

**Count: 266 clauses: 251 tested, 13 by inspection, 2 gaps** (severity 1: 0, severity 2: 1, severity 3: 1).

Regenerate the count (a row is a table line whose first cell is an id such as `G1`, `SH12b`):

    awk -F'|' '$2 ~ /^ [A-Z]+[0-9]+[a-z]? $/ { n++; s = $3; gsub(/ /, "", s); c[s]++ }
        END { printf "%d clauses: %d TESTED, %d INSPECTION, %d GAP\n", n, c["TESTED"], c["INSPECTION"], c["GAP"] }' docs/api-tests.md
    grep -oE '^\| [A-Z]+[0-9]+[a-z]? \| GAP \|.*sev [123]' docs/api-tests.md | grep -oE 'sev [123]' | sort | uniq -c

Runners. **make test**: every `tests/c` program and the asmcheck audits (`tests/abi`); CI runs it on every pull
request (linux arm64 and x86-64, both tiers), on macOS daily and through `tools/win` on Windows. **nightly**: hf
tokenizers 0.23.2 parity (`tests/parity`, the critical targets, both tiers) and the wheel tests (`python/tests`).
**rc san**: `tools/release/rc_host.sh`'s san step, every test program under ASan + UBSan and `test_par` under TSan,
on every release candidate. **fuzz**: the libFuzzer campaigns (`tests/fuzz`, asan + ubsan + lsan). A row's test is
make test unless it names another runner.

Severity of a gap: **1** a wrong byte or id, or memory outside the caller's buffer, can reach the caller; **2** a
wrong error code, return value, flag or field, or a valid input refused; **3** a policy or a documentation-level
promise. Each gap names a one-line mutant that breaks the promise and that, by reading the suite, no check
catches (`survives`: built and run against the tests that reach that code, which stayed green), and the test that
closes it.

## G: rules for every function (toks.h:6-15)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| G1 | TESTED | 7-8 | "NULL is allowed exactly where a length or cap of 0 is passed" | encode: test_api.c:485 (`toks_encode(&CTX, NULL, 0u, ...) == 2`), :374 (out NULL, cap 0: the count); decode :811; split_points test_split.c:263, :265; token_to_id test_vocab.c:197 (NULL, 0: TOKS_E_ID, accepted and never found); stream push / flush test_stream.c:732-733, :323; hold :876; batch test_par.c:382; par_encode out :183 (cap 0 with out NULL, :177); load_mem_copy test_e2e.c:493 (TOKS_E_FORMAT: an empty source). pieces and par_encode reach the same check (api.c:618, par.c:1011) |
| G2 | TESTED | 8 | "no alignment is required of any buffer": out, ends, ids, scratch, hold, text | test_misalign.c:71 (encode out at +1..3, every cap), :74 (pieces ends), :77 (decode ids), :65 (scratch); test_api.c:412 (scratch start 0..63); test_state.inc:533 (out at any alignment); test_stream.c:951 (hold at h2 + 1, + 7, + 63); text: test_k1.c:176-180 (K1 on text flush against a guard page, starting at offsets 0..2, every tier against the reference); the encode texts of test_api and test_e2e sit at arbitrary addresses |
| G2b | TESTED | 8 | the same, for split_points' offs and load_mem_copy's data | split_points' offs: test_misalign.c:93 (offs at +1..7, since PR#241 fixed the audit's finding: split.c:301 stored `offs[c]` through a `uint64_t *`); load_mem_copy's data: test_misalign.c:119 (data at +0..7 for gpt2, wp-bert-uncased, uni_t5base: the same source_sha256 and ids as the aligned load). No one-line mutant: the source is copied with memcpy (load.c:329) and parsed by bytes; the rc san step's UBSan build checks typed reads |
| G3 | TESTED | 10 | "for arbitrary contents of valid buffers, nothing outside SPEC §7.2's regions is read or written" | guard pages: test_api.c:656 (encode / pieces out flush against a no-access page, every cap), :835 (decode out), :401 + :412 with :357-360 (the scratch's ends; every region touched), :447 (an unbound buffer's end); test_stream.c:319 (push out); rc san (ASan, every program); fuzz (every entry point) |
| G4 | TESTED | 12 | "a context is read-only after load" | rc san: test_par under TSan (a pool's workers encode on one context at once: a write is a race report) |
| G4b | TESTED | 12 | the same, in make test | test_alloc.c:320 (the hash of every block live after load, before and after a battery of every entry point: six contexts). Mutant receipt: api.c:613 storing into ctx->dec_max on every encode survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| G5 | TESTED | 12-13 | "may be shared by any number of threads" | test_par.c:349 (two caller threads on one pool, both equal serial), :183 (a pool's workers on one context); rc san (TSan); nightly python/tests/test_threads.py |
| G6 | GAP | 13-14 | "no global is written" | sev 2. Nothing checks it. Today the core and gen objects have no writable data at all (nm / size: no __data, __bss or __common bytes), so nothing can be written; a new writable global passes every test. Test: tests/abi/cf_audit.py, a rule "no writable data in src/core and src/gen objects" |
| G7 | TESTED | 13-14 | "after load there is no allocation, syscall, lock, recursion or callback" | asmcheck: tests/abi/cf_audit.py:73 (ENTRY_AFTER_LOAD) with R6 (nothing reaches libc but memcpy / memset / memcmp, nor the platform layer), R1 (no indirect call), R4 (no recursion), its teeth tests/abi/cf_teeth.c; test_api.c:897 (no cpu probe after load). Note: toks_par allocates, locks and starts threads by design (its own section); the rule is the core's |
| G8 | TESTED | 15 | "errors are stable negative codes" | nightly: python/tests/test_api.py:22, :30 pin OPEN -1, FORMAT -2, UNSUPPORTED -3, TIER -5, ID -7, LIMIT -9 as the library returns them |
| G8b | TESTED | 15 | the same, for SCRATCH -6, CAP -8, ARG -10, NOMEM -11 | test_abi.c:78, :80, :82, :83. Mutant receipt: toks.h:54 TOKS_E_SCRATCH made -16 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| G9 | TESTED | 15 | "a failure leaves the context unchanged" | test_alloc.c:320 over the battery's failing calls (mode 3, unknown bits, NULL with a length, no or an uninitialized scratch, an id beyond the table, get_info size 7, a stream initialized with flag 2). Mutant receipt: api.c:616 a store into tok_bytes before returning -10 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| G10 | TESTED | 15 | "and the scratch valid" | test_state.inc:563 (the errors cell: unknown flags, another context, a text longer than the scratch, NULL out with cap, a text over TOKS_MAX_TEXT, each then the same call with room on the same scratch, against the cold ids at :213); test_par.c:303 (batch items failing E_ARG / E_LIMIT on a worker's scratch, the next items exact) |

Caller obligations (not counted): 7 "pointers are valid for their stated lengths and lifetimes"; 8-9 "out must not
overlap text, the context or the scratch"; 13 "a scratch and a toks_stream belong to one thread at a time".
Definition: 11 "units: bytes, uint32_t ids, counts in ids".

## V: versions (toks.h:3-4, 27-35, 332-333)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| V1 | TESTED | 4 | "An abi change bumps TOKS_ABI_MAJOR" | test_abi.c:70-222 (abi 0.4 written out: every public struct's size, alignment and field offsets, every constant, every entry point's prototype at its written type, test_abi.c:59: a changed type does not compile). Mutant receipt: toks.h:304 control_isolation made uint64_t survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (sizeof(toks_info) 192, pinned 184 then; 240 since abi 0.4). **Finding** (stays): 0.x's additive changes bumped TOKS_ABI_MINOR (0.3, 0.4); the sentence names only MAJOR |
| V2 | TESTED | 30-31 | "tag v<TOKS_VERSION>, and the python wheel's version" | tools/release.sh:18 (a tag that is not v<TOKS_VERSION> stops the release); nightly python/tests/test_api.py:41 (`toks.__version__` == toks.h's TOKS_VERSION == the wheel's metadata) |
| V3 | INSPECTION | 31 | "a release that changes the abi bumps both" | by inspection: docs/release.md, the release checklist (a process rule) |
| V4 | TESTED | 32-35 | TOKS_VERSION is MAJOR.MINOR.PATCH of the three macros | test_version.c:14 |
| V5 | TESTED | 332 | toks_version: "TOKS_VERSION of the library the program runs with" | test_version.c:15 |

## E: error codes (toks.h:47-62)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| E1 | TESTED | 47 | "fixed for the life of the major version" | as G8 (six values pinned by the wheel tests); the other four are G8b's gap |
| E2 | TESTED | 49 | TOKS_E_OPEN "the file could not be opened or read" | test_e2e.c:473 (a missing path: -1, *out NULL, diag.code -1); test_tiktoken.c:1007 (ranks without their companion files); nightly python/tests/test_api.py:68, :70 |
| E3 | TESTED | 50 | TOKS_E_FORMAT "not a valid tokenizer file / image (syntax, structure, checksum)" | syntax: test_e2e.c:492 (`"{"`: -2, diag.code -2); structure: test_compile.c:564 over tests/data/compile/refuse.txt's -2 rows (added_tokens entry fields, ByteLevel fields: the config path toks_load uses); nightly python/tests/test_api.py:73. **Finding**: "image ... checksum" names the .toks image, which has no loader and left the header in 0.3.0 |
| E4 | TESTED | 51 | TOKS_E_UNSUPPORTED "... (named in diag)" | test_spm.c:297 (the 15 files of tests/data/spm/refuse.txt: no context, diag.code -3, diag.what the named feature) |
| E5 | INSPECTION | 52 | "-4 is not assigned" | by inspection: toks.h:49-62 (no macro is -4; every error return in src names a macro); the abi pin (V1) would make it a test |
| E6 | TESTED | 53 | TOKS_E_TIER "a forced tier this machine cannot run" | test_tier.c:250 (every tier the cpu or the build lacks: -5, *out NULL), test_e2e.c:336 (the other isa's tier, diag.code -5), :500 (TOKS_TIER=bogus) |
| E7 | TESTED | 54 | TOKS_E_SCRATCH "scratch NULL" | test_api.c:465; pieces reaches the same check (api.c:620) |
| E8 | TESTED | 54 | "too small" | test_api.c:462 (init one byte short), :473 (a text one byte over max_len), :467 (the default memo's 4 MiB missing) |
| E9 | TESTED | 54 | "not initialized" | test_api.c:447-448 (encode and pieces on never-initialized buffers of 127..65599 bytes of garbage), :464 (zeros) |
| E10 | TESTED | 54 | "or bound to another context" | test_api.c:476-477, test_e2e.c:460 (gpt2's scratch refused by llama3) |
| E11 | TESTED | 55 | TOKS_E_ID "an id beyond the table (decode)" | test_api.c:806 (id = n_ids: -7, nothing written); nightly python/tests/test_api.py:225 (2^32 - 1) |
| E12 | TESTED | 56 | TOKS_E_CAP "output capacity below what an atomic call needs" | test_stream.c:307 (push, every cap below the need), :331 (flush), :652 (hand cases) |
| E13 | TESTED | 57 | TOKS_E_LIMIT "text > 2^29 bytes" | test_api.c:489 (encode), test_split.c:266, test_par.c:365, :281 + :303 (a batch item), test_stream.c:777 (a push of 2^29 + 1 ids); nightly python/tests/test_api.py:164-170 (pieces too, through an untouched 2^29 + 1 mapping) |
| E14 | TESTED | 57-59 | TOKS_E_LIMIT "a byte-fallback run longer than a stream's hold" | test_stream.c:1207 (the 45th run byte in st's own 44), :1031 (12 x U+13000), :889-890 (a 64-byte hold) |
| E15 | TESTED | 60 | TOKS_E_ARG "NULL with a nonzero length" | encode text / out test_api.c:484, :486; decode :808-809; scratch_init :460; split_points test_split.c:262, :264; token_to_id test_vocab.c:196; push test_stream.c:727-728; flush :731; hold :871-872; par_encode test_par.c:363-364; batch :367; a batch item's text :280 + :303; toks_load's path test_e2e.c:475 |
| E15b | TESTED | 60 | the same, for toks_load_mem_copy(data NULL, len > 0) | test_load.c:321 (data NULL, len 5: -10, *out NULL, diag.code -10). Mutant receipt: load.c:321 removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (SIGSEGV: the sniff reads NULL) |
| E16 | TESTED | 60-61 | "unknown flag bits" | load opts.flags test_e2e.c:488, :490; scratch_init test_api.c:559 (bit 21); encode :487; pieces :490; split_points test_split.c:261; decode test_api.c:810; stream_init test_stream.c:744-745 (inited with flag 2: push and flush -10); par_encode test_par.c:362 |
| E16b | TESTED | 60-61 | the same, for toks_par_create's scratch_flags and toks_par_encode_batch's flags | test_par.c:410, :412 (scratch_flags bit 21, bit 31: -10, *out NULL), :378 (batch flags 16, 64, 1 << 31 and mode 3: -10, the items untouched). Mutant receipts: par.c:916's mask taking bit 21 in, par.c:981's taking bit 4 in, each survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| E17 | TESTED | 62 | TOKS_E_NOMEM "load-time allocation failed" | test_alloc.c:205 (every request of a load failing in turn: -11, diag.code -11, *out NULL, no block left live), :211 (a request with a fallback: the context encodes the probes as the clean one), :218 (any other code fails); gpt2style by toks_load and toks_load_mem_copy, nosplit (the generic engine), llamalike (1 of 8 requests has a fallback), bound_bf_meta (unigram), cached gpt2 and wp-minilm-l6 (12 of 18 requests are the optional tries, wp.c:473, :485). The program's own toks_plat_* win the link over mem.o. Mutant receipts: load.c:328 NOMEM made FORMAT, vocab.c:102 NOMEM made LIMIT, each survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |

## L: limits (toks.h:64-70)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| L1 | TESTED | 66 | TOKS_MAX_TEXT "bytes per encode call": 2^29 + 1 refused | E13's rows |
| L1b | TESTED | 66 | 2^29 itself accepted | test_load.c:355-356 (encode and pieces of 2^29 bytes on a no-access mapping with a 64-byte scratch: -6, the length check passed and no text byte read), :364 (split_points over 2^29 read-only zero pages: 0 or 1 cut). Mutant receipts: api.c:619 and split.c:278 `>` made `>=` each survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| L2 | TESTED | 67 | TOKS_MAX_TOKEN_BYTES 65535 "bytes of one vocabulary token" | test_tiktoken.c:458 (a 65536-byte tiktoken.model token: -2 naming TOKS_MAX_TOKEN_BYTES) |
| L2b | TESTED | 67 | the same, for a tokenizer.json vocabulary token (config.c:235, :1341: -9), and 65535 loading | test_load.c:381 (a 65535-byte vocab token loads: toks_token and toks_token_to_id find it), :383 (65536: -9 naming TOKS_MAX_TOKEN_BYTES), :389 (a 65536-byte unigram piece: -9). Mutant receipts: config.c:235 `>` made `>=` survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here; config.c:1341 removed likewise (unigram then says -3). unigram's own limit, 127, is L2c |
| L2c | TESTED | 67 | "unigram pieces: 127, -3 above" | test_load.c:404 (bound_bf_meta with a 127-byte piece spliced in loads, toks_token finds its 127 bytes), :406 (128 bytes: -3 naming the limit). Mutant receipt: unigram.c:380 `>` made `>=` (a 127-byte piece refused) survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (test_load.c:404) |
| L3 | TESTED | 68 | TOKS_MAX_ADDED_BYTES 255 "bytes of one added token" | test_compile.c:564 over tests/data/compile/refuse.txt:75 (-9 naming TOKS_MAX_ADDED_BYTES) |
| L3b | TESTED | 68 | a 255-byte added token loads | test_load.c:432 (a 255-byte added token loads, encode matches it once, TOKS_ID_ADDED), :436 (256: -9 naming TOKS_MAX_ADDED_BYTES). Mutant receipt: config.c:389 `>` made `>=` survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| L4 | TESTED | 69 | TOKS_MAX_IDS "every id is < TOKS_MAX_IDS" | test_tiktoken.c:416 (rank 2097151 refused) |
| L4b | TESTED | 69 | the same, for tokenizer.json ids (config.c:234 vocab, :400 added: -9) | test_load.c:445 (a vocab id of 2^21 - 1, dense: -9 naming TOKS_MAX_IDS), :447 (with holes). Audit correction: config.c:234 removed was already killed on master, by test_load.c:195's regress repro tests/fuzz/regress/load_json/nomem-generic-051fe215.json (-3 "not dense", want -9; gb10a receipt). config.c:400, an added token's next id: reachable only past 2^21 - 1 vocabulary strings (holes do not help: the next id counts strings): by inspection. the accepting side, an id of 2^21 - 2: L4c |
| L4c | GAP | 69 | the same: an id of 2^21 - 2 (< TOKS_MAX_IDS) loads | sev 3. 0.3.1: refuses one id early for sentencepiece-style and unigram. spm_build.c:273 and config.c:1326 refuse n_ids == TOKS_MAX_IDS (holes_added with a vocab id 2097150: -9 "ids or merges beyond the table widths"; 2097149 loads); bpe_build.c:325 and tiktoken.c:47 allow it. Ruled a 0.3.1 one-character fix (`>`), with this test: holes_added with a vocab id of 2^21 - 2 loads (no test pins the wrong boundary meanwhile) |
| L5 | TESTED | 70 | TOKS_MAX_SOURCE_BYTES "tokenizer file / image size" (256 MiB) | test_load.c:339 (toks_load_mem_copy of 256 MiB + 1 on a no-access mapping: -9, *out NULL, diag.code -9). Mutant receipt: load.c:322 removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (bus error: the sniff reads the mapping). toks_load's file-size check (file.c) and tiktoken.c:78: by inspection |

## T: types (toks.h:72-94)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| T2 | TESTED | 77 | toks_diag.code "the TOKS_E_* returned" | test_e2e.c:473 (-1), :492 (-2), :336 (-5); :196 (0 on success) |
| T3 | TESTED | 78 | toks_diag.what "NUL-terminated: the blocking feature or the reason" | test_spm.c:297 (strcmp with the named reason) |
| T3b | TESTED | 78 | a reason over 247 bytes is cut and NUL-terminated within 248 | test_load.c:221 (a 599-byte missing path, diag flush against a guard page: -1, what the path's first 247 bytes, terminated). Mutant receipt: load.c:31's `- 1u` dropped survives 4d5a92f's suite and dies here (exit 138: the write one past what lands on the guard page) |
| T4 | TESTED | 81 | "TOKS_TIER_AUTO picks the fastest the machine supports" | test_tier.c:241 |
| T5 | TESTED | 83-86 | each tier is that tier's kernels | test_tier.c:250 (a forced tier loads exactly when built and the cpu has it, else -5), :193 (its ids equal scalar's) |
| T5b | TESTED | 84-86 | the features each tier needs (neon + crc32; avx2, bmi1, bmi2, lzcnt, popcnt + sse4.2 crc32; avx-512 f, bw, vl, vbmi + bmi2) | test_abi.c:218-222 (cpu.h's three masks against the list). cpu.h's AVX512 mask is the avx2 tier's plus f, bw, vl, vbmi (bmi2 among them): toks.h names only the avx-512 bits and bmi2; every avx-512 cpu has the rest. Mutant receipt: cpu.h:35 lzcnt dropped from the avx2 mask survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (an arm host; on x86 test_tier computes "the cpu has it" from the same mask) |
| T6 | TESTED | 89 | size "sizeof(toks_load_opts)" (else -10, and diag is not read) | test_e2e.c:477 (size 8), :482-483 (a 16-byte block, toks_load and toks_load_mem_copy: -10 before the diag pointer past it is read) |
| T7 | TESTED | 90 | "AUTO also honours the TOKS_TIER environment variable" | test_tier.c:257, test_e2e.c:506 |
| T8 | TESTED | 91 | flags "a nonzero value is TOKS_E_ARG" | test_e2e.c:488, :490 |
| T9 | TESTED | 92 | rsv "0" (a nonzero rsv: -10, load.c:263) | test_load.c:317 (rsv 1: -10, *out NULL, diag.code -10). Mutant receipt: load.c:263's `o->rsv != 0u \|\|` removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| T10 | TESTED | 93 | diag "optional" | NULL: test_tier.c:238 (options without a diag load); set: test_e2e.c:473 |

Definition (not counted): 74 toks_ctx "opaque; created by toks_load*, freed by toks_unload".

## LD: loading (toks.h:96-106)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| LD1 | TESTED | 98 | path "a tokenizer.json" | test_e2e.c:196 (14 real files: 0, a context, diag.code 0) |
| LD2 | TESTED | 98-99 | "a tiktoken model's tiktoken.model ... (its other files beside it)" | test_tiktoken.c:993 (kimi's tiktoken.model by its path), :1007 (without its companions: -1 naming them) |
| LD2b | TESTED | 98-99 | "... or qwen.tiktoken" | test_e2e.c:406 (the cache's qwen1-72b files under their model names in a model directory: toks_load of the directory and of its qwen.tiktoken: 0), :412 (n_ids 151851, byte-level, the directory's name), :418, :420 (tiktoken 0.14.0's ids through the wrapper's own constants, tests/parity/qwen1_check.py's reference). Audit correction: test_targets.c:107 parses and compiles these files through the internal parser, not toks_load's path. Mutant receipts: load.c:287's and :305's "qwen.tiktoken" misspelled, each survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| LD3 | TESTED | 98-99 | "or a model directory holding one of them" | test_e2e.c:266 (a directory holding tokenizer.json; :270 its name), test_tiktoken.c:993 (kimi's directory) |
| LD4 | INSPECTION | 99 | "Compiles, validates and certifies" | a summary: its promises are rows E3, E4, L2-L5 (validation) and SP8 (certified cuts); by inspection: load.c load_src |
| LD5 | TESTED | 99-100 | "bounded work" | fuzz: fuzz_load and fuzz_load_json (libFuzzer's per-input timeout makes a slow load a finding); test_load.c:195 replays the campaigns' repros |
| LD6 | TESTED | 99 | "opts may be NULL" | test_load.c:111, test_e2e.c:266 |
| LD7 | TESTED | 99-100 | "On success *out is a new context and 0 is returned" | test_load.c:111 (0 and a context, 2000 cycles); test_e2e.c:460 (two loads are two contexts) |
| LD8 | TESTED | 100 | "Mistral tekken.json is not read" | test_load.c:326 (a directory holding only tests/data/tekken/tekken.json: -1), :329 (that file: -2, no "model"). Mutant receipt: load.c:287 looking for tekken.json where it looks for qwen.tiktoken survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (the directory gives -2) |
| LD9 | TESTED | 103 | toks_load_mem_copy: "copied into memory the context owns" | rc san: test_par.c:119 frees the source right after the load and keeps using the context (ASan reports a borrowed buffer); test_e2e.c:253 (the same ids from memory) |
| LD10 | TESTED | 103 | "(a tiktoken model: toks_load its directory)" | test_tiktoken.c:1012 (tiktoken.model's bytes: -3 naming the three files) |
| LD11 | TESTED | 106 | toks_unload "NULL-safe" | test_load.c:196 (toks_unload after every refused repro, whose context is NULL: a crash fails the program) |

## SC: scratch (toks.h:108-128)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| SC1 | TESTED | 110 | TOKS_SCRATCH_MEMO_MIB(m): "the segment memo (SPEC §6), m MiB" | test_api.c:619-620 (2 MiB), :306-310 (the formula's memo bytes) |
| SC2 | TESTED | 110-111 | "Flags without it give the default, 4 MiB" | test_api.c:604-605, :467-468 |
| SC3 | TESTED | 110-111 | "a text or conversation sent again is answered from it" | test_api.c:636-638 (the second encode: a memo hit, no K5 call, the same ids); test_state.inc:213 (the memo cell's conversations against cold ids) |
| SC4 | TESTED | 111-112 | "TOKS_SCRATCH_MEMO_MIB(0) is a budget of zero, no memo" | test_api.c:614-616 (no memo region), :637 (no hit, K5 runs) |
| SC5 | TESTED | 113 | "Wordpiece and unigram have no memo at any m" | test_api.c:319-321 (the formula), :624-626 (init) |
| SC6 | TESTED | 113-114 | "a 0.2.0 binary's TOKS_SCRATCH_MEMO_MIB(m > 0) means what it did" | test_api.c:611-612 (raw 4: the same budget), :310 (raw 3: + 3 MiB) |
| SC7 | TESTED | 116 | TOKS_SCRATCH_MEMO_SET "(alone: a budget of zero)" | test_api.c:617-618, :309 |
| SC8 | TESTED | 117 | CACHE_MIB "0 = the default (2 MiB, short pieces only)" | test_api.c:585-586 (cache_mib 0, no long cache), :568-569 (a fresh scratch: the 2 MiB mask, no long cache) |
| SC9 | TESTED | 117-118 | "else a power of two 4..128 (anything else: TOKS_E_ARG)" | test_api.c:557 (1, 2, 3, 5, 6, 12, 129, 192, 255: -10), :561, :583 (8, 4 accepted), :549 (128's bytes); nightly python/tests/test_api.py:106-108 |
| SC10 | TESTED | 118 | "n / 2 MiB for short pieces and n / 2 MiB for long ones" | test_api.c:564 (8: the long cache at + 4 MiB, work at + 12 MiB with the memo), :583-584 (4: at + 2 MiB) |
| SC11 | TESTED | 118-119 | "A memory budget only: outputs are identical at every size" | test_state.inc:213 (every call of the warm, warm32, memo, mode and par cells against the cold ids); test_e2e.c:241 (:103-:175's hf checks again, through CACHE_MIB(4)) |
| SC12 | INSPECTION | 122 | toks_scratch_bytes "pure" | by inspection: api.c toks_scratch_bytes reads only its arguments (and G6: the core has no writable global) |
| SC13 | TESTED | 122 | "the scratch size for texts of up to max_len bytes (the published formula)" | test_api.c:301-302 (the formula restated independently), :401-402 (exactly that many bytes, flush against a guard page, encode a max_len text); test_e2e.c:153 (a tight scratch per real text, NFC included) |
| SC14 | TESTED | 125 | toks_scratch_init "binds scr (any alignment, `bytes` long)" | test_api.c:412 (start 0..63), :401 (end flush, + 0..63 bytes), :345-360 (the regions inside, every byte touched) |
| SC15 | TESTED | 125 | "and clears its caches" | test_api.c:517-519 (re-init: the epoch moves, K5 gets it as the cache tag, the counters reset), :527 (the last epoch zeroes), :537 (a foreign header zeroes); test_state.inc:213 (the rebind and wrap cells against cold ids) |
| SC16 | TESTED | 125-126 | "A scratch bound to another context returns TOKS_E_SCRATCH from encode calls until it is initialized again" | test_api.c:476-477, :525, :480 (re-initialized: works) |
| SC17 | TESTED | 126-127 | "so does any buffer of at least 127 bytes that was never initialized" | test_api.c:447-448 |
| SC18 | TESTED | 127 | "(a call reads only its header before the binding is proven)" | test_api.c:447-448 (the buffer's end flush against a guard page from 127 bytes up) |

Measurements (not counted): 111 "at GB/s"; 112-113 "slows by 1-14%"; 119 "a scratch reused across calls (warm) gets
faster" (receipts: docs/usage.md "Scratch", docs/bench/e2e.md).

## EN: encode (toks.h:130-152)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| EN1 | TESTED | 132-133 | mode bits 0-1; TOKS_ADDED_ALL "special and non-special added tokens recognized (hf default)" | test_e2e.c:103 (hf's ids, 14 real files), test_breadth.c:102 (the breadth fixtures); nightly tests/parity/run.py:413 over gen_cases.py:33's flags. Note: mode 3 is -10 (test_api.c:488); the header does not say so |
| EN2 | TESTED | 134 | TOKS_ADDED_NONSPECIAL "(hf encode_special_tokens=True)" | as EN1, flags 1 |
| EN3 | TESTED | 135 | TOKS_ADDED_NONE "no added token recognized" | as EN1, flags 2 |
| EN4 | TESTED | 136 | "flags 0 returns exactly what hf encode(text) returns by default" | as EN1, flags 0 |
| EN5 | TESTED | 143 | TOKS_NO_POSTPROCESS "(hf add_special_tokens=False)" | as EN1, flags 4-6 |
| EN6 | TESTED | 144 | TOKS_CONTINUATION "no start-of-input behaviour" | test_split.c:143, :156 (a part with the flag concatenates to the whole, every family with rules, every mode); test_e2e.c:231 (a continuation part is not truncated) |
| EN7 | TESTED | 149 | "returns n >= 0, the total number of ids" | test_api.c:656 (every cap 0..n + 1), test_e2e.c:117 |
| EN8 | TESTED | 149 | "out[0 .. min(cap, n)) holds the exact prefix" | test_api.c:656, test_e2e.c:117 |
| EN9 | TESTED | 149-150 | "out may be NULL only with cap 0" | test_api.c:374 (NULL, 0: the count), :486 (NULL, 1: -10) |
| EN10 | TESTED | 150 | "Entries of out at index >= min(cap, n) and < cap may be overwritten" (and none at >= cap) | test_api.c:656 (out flush against a guard page), test_e2e.c:117 (out[cap] untouched), test_state.inc:533 |
| EN11 | TESTED | 136-138 | flags 0: "post-processor applied, then the file's truncation and padding" | test_primitives.c:151 (hf encode(), flags 0 and 4, five texts, on the 29 files that truncate or pad: tests/data and the cache, all-MiniLM-L6-v2's Fixed 128, types_left_pad's Fixed 13 on the Left with a multiple of 4) |
| EN12 | TESTED | 145 | TOKS_NO_TRUNCATE "the file's truncation is not applied (hf no_truncation())" | test_primitives.c:151 (hf encode() after no_truncation(), the flag sets with 16; one text past 8,196 ids), :171 (no change on a file without truncation or padding), :164 (toks_pieces takes it, unchanged) |
| EN13 | TESTED | 146 | TOKS_NO_PAD "the file's padding is not applied (hf no_padding())" | test_primitives.c:151 (after no_padding(), the flag sets with 32), :171, :164 |
| EN14 | TESTED | 146-147 | "With both the count is the whole text's, unpadded" | test_primitives.c:151 (flags 48 and 52 against hf with both off: 1,160 cases in all), :243 (encode("") with both: hf's, the template alone) |

## PC: pieces (toks.h:154-159)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| PC1 | TESTED | 154 | "the pieces the model sees, as end offsets into the caller's bytes" | test_e2e.c:135 (hf's ends, every mode, 14 real files), test_breadth.c:116; nightly parity (the pieces op) |
| PC2 | TESTED | 154-155 | "where a materializing normalizer (NFC) changed the text, into its normalized form" | test_e2e.c:135 with :128 (qwen38's NFC: the ends tile the normalized length) |
| PC3 | TESTED | 155 | "An added-token match is one piece" | test_breadth.c:116; test_api.c:656 (pieces against a reference that makes each match one end) |
| PC4 | TESTED | 155-156 | "same capacity rule as toks_encode" | test_api.c:656 (every cap, guard page), :379 (NULL, 0), :490 |

## TP: the template (toks.h:161-167)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| TP1 | TESTED | 161-162 | "the ids it puts before the text's ids (*n_prefix of them) and then after them, written to ids[0, count), and their type ids to type_ids" | test_primitives.c:226 (hf's post_process() of the empty encoding: ids and type ids, the prefix count from post_process() of a short text; 173 files: bert's [CLS] / [SEP], llama 3's one prefix id, xlnet's <sep> <cls> after the text with type ids 0 and 2, types_left_pad's 2 and 3; nothing written past the count) |
| TP2 | TESTED | 162 | "(NULL: not written)" | test_primitives.c:237 |
| TP3 | TESTED | 162-163 | "returns the count, hf's num_special_tokens_to_add(False)" | test_primitives.c:219, :226 (test_primitives.py refuses a file whose post_process() count is not num_special_tokens_to_add(False)) |
| TP4 | TESTED | 163 | "the text's own type id is toks_info's seq_type_id" | test_primitives.c:387 (hf's type id of the text's ids in post_process(): 1 for types_left_pad) |
| TP5 | TESTED | 163-164 | "ids NULL with cap 0 sizes it (the count, *n_prefix set)" | test_primitives.c:219 |
| TP6 | TESTED | 164 | "a cap below the count is TOKS_E_CAP and writes nothing" | test_primitives.c:233 (the ids, type ids and *n_prefix canaries kept) |
| TP7 | TESTED | 164-165 | "TOKS_E_ARG for ctx NULL or ids NULL with cap > 0" | test_primitives.c:239 |
| TP8 | TESTED | 165 | "n_prefix may be NULL" | test_primitives.c:237 |

## CA: capacity (toks.h:169-182)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| CA1 | TESTED | 171 | toks_encode_bound: "the most ids toks_encode can return for a text of len bytes, under any flags" | test_bound.c:184 (every fixture and cached file, all 12 flags: the corpora, runs of 0xFF / '\n' / ' ' to 4096, the scan's worst char, added tokens, random bytes and utf-8, the fuzz corpora) |
| CA2 | TESTED | 171-172 | "ceil(r len) + g, with r ... and g computed per context at load" | test_bound.c:527 (the arithmetic: ceil, unreduced r, + g, on constructed terms), :427 (the context's terms are the load's) |
| CA3 | TESTED | 172 | "(a context that pads: at least its padded length)" | test_bound.c:542 (Fixed / multiple rounding on constructed terms), :559, :564 (trunc_pad with a multiple of 8, through encodes), :446 |
| CA4 | TESTED | 172-173 | "so an out of this many ids is never short" | test_bound.c:184 (encodes into an out of the bound: n <= bound) |
| CA5 | TESTED | 173-174 | g: "the template's ids plus the ▁ a model prepends (1; 3 for unigram byte fallback with no ▁ piece)" | test_bound.c:184 and :448 (an r = 1 context must reach len + g exactly, so g cannot be short or long there) |
| CA6 | TESTED | 174-179 | r by normalizer is an upper bound: the scan's worst char never passes it | test_bound.c:251 (every char a normalizer maps, every byte, a sample of each utf-8 length: ids per byte <= r) |
| CA6b | TESTED | 174-179 | r's values: NFC byte-level 3, NFKC 11, nmt_nfkc 6 (11 under byte fallback), NFKD then the charsmap 66, x 3 for unigram byte fallback without ▁, "everything else 1" | test_bound.c:435 over PINNED (test_bound.c:72-77): byte-level gpt2style 1, NFC drop_nfc and qwen38 3, NFKC dg-exaone35 11, nmt_nfkc uni_t5base 18/3, NFKD + charsmap uni_albert 198/3, with PR#239's four (unigram byte fallback without ▁: f 3). nmt_nfkc under byte fallback (11): no fixture or cached file has it, by inspection (compile.c:580-586). Mutant receipt: compile.c:591 byte-level r 1 made 2 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (gpt2style r 2/1); compile.c:566 dies on bound_phase0's pin |
| CA7 | TESTED | 179-180 | "and tight: len + the template's ids (+ 1 for a prepended ▁), which a text of len >= 1 bytes that are one id each reaches" | test_bound.c:448 (every context with r = 1, and every padded one, reaches its bound on an input) |
| CA8 | TESTED | 180-181 | "Any len has a bound, but toks_encode returns TOKS_E_LIMIT above TOKS_MAX_TEXT" | test_bound.c:527 (lens to UINT64_MAX); toks_encode's -9: E13 |
| CA9 | TESTED | 181 | "Saturates at UINT64_MAX" | test_bound.c:527 (never a wrapped value) |
| CA10 | TESTED | 181 | "NULL gives 0" | test_bound.c:509 |

Measurements (not counted): 175-177 "the worst measured 3.67", "2.04", "2.4" (test_bound's printed census).

## SP: split planning (toks.h:184-196)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| SP1 | TESTED | 186 | "up to min(cap, n_want - 1) cuts" | test_split.c:247 (against a transcription, caps 0..4 and 4096), :276 |
| SP2 | TESTED | 186 | "strictly increasing offsets in (0, len)" | test_split.c:224 |
| SP3 | TESTED | 186-187 | "returns their count c" | test_split.c:247 |
| SP4 | TESTED | 187 | "the certified cut nearest its target len * i / n_want" | test_split.c:247, :273-274 |
| SP5 | TESTED | 187 | "(lower on a tie)" | test_split.c:275, :247 |
| SP6 | TESTED | 187-188 | "within D = min(4096, len / (4 n_want)) bytes" | test_split.c:247, :270 |
| SP7 | TESTED | 188 | "a target without one is skipped" | test_split.c:247, :270 |
| SP8 | TESTED | 188-190 | "encoding the parts ... and concatenating gives the whole input's ids" | test_split.c:143, :156 (every string of up to L symbols and random ones, every family with rules, real files); test_breadth.c:243 |
| SP9 | TESTED | 190-191 | "the added-token mode changes the valid cuts" | test_split.c:143 under each of :176's three modes |
| SP10 | TESTED | 192 | "n_want <= 1 returns 0" | test_split.c:267-268 |
| SP11 | TESTED | 191-193 | "a tokenizer family without certified rules returns 0" | test_split.c:277 |
| SP12 | TESTED | 193 | "Reads at most n_want x (2D + W) bytes" | test_split.c:444 (every real file plans a 32 MiB text whose pages beyond D + 512 bytes of each target are no-access: text with cuts near every target, then letters with none, whose whole windows are read: no fault, every cut within D). Mutant receipt: split.c:283's D cap removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (a bus error: the search leaves the window). The exact count, n_want x (2D + W): by inspection, split.c:289-298 (rings of D around each target) and :237 (win = the longest token + 16 a side) |
| SP13 | TESTED | 193-194 | "scr is reserved and may be NULL" | every call in test_split.c passes NULL (:245) |
| SP14 | TESTED | 191-192 | "a file's truncation or single-text padding (Fixed, a multiple) leaves none unless the flags turn it off (TOKS_NO_TRUNCATE, TOKS_NO_PAD)" | test_primitives.c:189 (flags 0: no cut on every file that truncates or pads one text), :192, :206 (both flags: the family's cuts, and the parts, each after the first with TOKS_CONTINUATION, concatenate to the whole: 147 cuts), :185 (a file that does neither, BatchLongest alone among them: the flags change no cut) |

## DE: decode (toks.h:198-212)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| DE1 | TESTED | 200 | TOKS_SKIP_SPECIAL "hf skip_special_tokens=True" (flags 0: False) | test_e2e.c:175 (hf's strings, both, real files), test_breadth.c:125, test_api.c:792-796, :860 (a compiled normalized special); nightly parity (decode, both) |
| DE2 | TESTED | 206 | "returns the total number of bytes; out[0 .. min(cap, n)) holds the exact prefix" | test_api.c:835 (every cap, guard page), :811 |
| DE3 | TESTED | 206-207 | "An id beyond the table returns TOKS_E_ID" | test_api.c:806; nightly python/tests/test_api.py:225 |
| DE4 | TESTED | 211 | toks_token: "zero-copy raw bytes of one id" | test_api.c:881 |
| DE5 | TESTED | 211 | "or NULL if the id has no string" | test_api.c:883 (and :884: beyond the table, NULL) |
| DE6 | TESTED | 201-203 | TOKS_DECODE_RAW: "where decode writes U+FFFD for bytes that are not utf-8 (a byte-level id's, a byte-fallback run's), those bytes stand instead (tiktoken's decode_bytes)" | test_primitives.c:336 (1,347 references: tiktoken decode_bytes on Kimi K3; hf's ByteLevel bytes before its lossy step; hf's ByteFallback decode with each U+FFFD of an invalid run put back to its byte, on the byte-fallback files), :324 (byte-level: exactly the tokens' bytes, every file's texts and random ids) |
| DE7 | TESTED | 203-204 | "every other step is decode's, so a result that is valid utf-8 is decode's" | test_primitives.c:311 (every file: its texts' ids and 64 random sequences, with and without TOKS_SKIP_SPECIAL: 23,688) |
| DE8 | TESTED | 204 | "Not a stream flag" | test_stream.c:743-745 (a stream inited with flag 2: push and flush -10) |
| DE9 | TESTED | 206 | under TOKS_DECODE_RAW: "out[0 .. min(cap, n)) holds the exact prefix" | test_primitives.c:366 (every cap 0..n + 1, nothing written at cap) |

## VO: vocabulary (toks.h:214-245)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| VO1 | TESTED | 216 | "the id whose string is the bytes s[0, len): toks' key is the bytes an id decodes to (toks_token's)" | test_vocab.c:169 (every id's string, 13 real files and 11 fixtures, gets exactly the rule's answer from a sorted copy of every key), :220 (hf's pins) |
| VO2 | TESTED | 216-218 | "NOT hf's vocabulary spelling ... "Ġhello" is TOKS_E_ID" | test_vocab.c:220 (gpt2 " hello" 23748, "Ġhello" -7) |
| VO3 | TESTED | 218-220 | the footgun: gpt2's C2 A2 is 44359, not 95 | test_vocab.c:220 |
| VO4 | TESTED | 220 | "A sentencepiece piece is as written ("▁hello", "<0x41>")" | test_vocab.c:220 (llama2 22172, 68) |
| VO5 | TESTED | 220-221 | "an added token's content finds it" | test_vocab.c:220 (llama2 <s> 1, gpt2 <\|endoftext\|> 50256) |
| VO6 | TESTED | 221-222 | "as does its normalized form where hf decodes one (llama2's "▁<s>")" | test_vocab.c:220 |
| VO7 | TESTED | 222-224 | "Where ids share a string ... an added token's content first" | test_vocab.c:220 over :333's content_first pins (tests/data/vocab/content_first.json: the content "▁<q>" 299 against "<q>" normalized, 300, whose string is "▁<q>" too: 299, hf 0.23.2's answer), :169 (the rule on every id of it). Mutant receipt: vocab.c:139 removed survives 4d5a92f's suite (every other shared content is the later id, where "the later one" agrees) and dies here ("▁<q>" -> 300, want 299) |
| VO8 | TESTED | 224 | "then the id the file writes as those very bytes" | test_vocab.c:220 (written_tie: a raw U+200D 295 against its alphabet form 296), :169. Mutant receipt: vocab.c:180 disabled survives 433a7a2's test_vocab and dies on this one ("\u200d" -> 296, want 295) |
| VO9 | TESTED | 224 | "then the later one" | test_vocab.c:264 (a hand-built context through vocab.c's own builder: two ids decode to "zz", the file spells neither so: the later). No loadable file reaches this rule (a byte-level string outside the alphabet is written as its own bytes; a piece is its key; a content wins first), hence the hand-built case. Mutant receipt: vocab.c:180 made "the earlier stays" survives 4d5a92f's suite and dies here ("zz" -> 1, want 2) |
| VO10 | TESTED | 224-225 | "The empty string is never found" | test_vocab.c:196-197 (NULL, 0 and "a", 0: -7) |
| VO11 | TESTED | 225-226 | "TOKS_E_ID when no id has that string" | test_vocab.c:180 (strings near every id's: exactly the rule's answer, -7 where no key has them), :201 (over the longest token), :220 |
| VO12 | TESTED | 226 | "TOKS_E_ARG for ctx NULL or s NULL with len > 0" | test_vocab.c:196 |
| VO13 | TESTED | 229 | toks_id_flags "0: a plain vocabulary id" | test_vocab.c:222 (gpt2 hello, llama3 hello, ...: 0) |
| VO13b | TESTED | 229 | "... or an id with no string" | test_vocab.c:165 (every id with no string that no added content holds, in every file: flags 0). Mutant receipt: vocab.c:174 setting ADDED on such an id survives 4d5a92f's suite and dies here (holes_added id 1) |
| VO14 | TESTED | 229 | "TOKS_E_ID beyond the table" | test_vocab.c:198-199 (n_ids, UINT32_MAX) |
| VO15 | TESTED | 230 | "TOKS_E_ARG for ctx NULL" | test_vocab.c:198 |
| VO16 | TESTED | 231 | TOKS_ID_ADDED "(hf's added_tokens_decoder has it)" | test_vocab.c:222 (hf's flags pinned on 7 files), :187 (every K1 entry ADDED) |
| VO17 | TESTED | 232-234 | TOKS_ID_SPECIAL: a content listed once | test_vocab.c:222 |
| VO17b | TESTED | 232-234 | "special if any listing of the content ... is special" (a content listed twice) | test_vocab.c:222 over :323's dup_added pins (296, 297: ADDED and SPECIAL; hf's added_tokens_decoder says False, the documented divergence) |
| VO18 | TESTED | 234-236 | "the token's own flag, not decode's skip rule (llama2's id 1 is special ...)" | test_vocab.c:222 (llama2 <s>: ADDED and SPECIAL) |
| VO19 | TESTED | 237-238 | TOKS_ID_BYTE "a byte-fallback <0xHH> ... that a ByteFallback chain decodes as one byte" | test_vocab.c:162 (every id of every file), :222 (llama2 <0x41>, <0x00>) |
| VO19b | TESTED | 237 | "(or hf's <0x+F>)" | test_vocab.c:285 (tests/data/vocab/byte_plus.json, llamalike's byte 0F spelled "<0x+F>": TOKS_ID_BYTE, the piece as written, decode 0F as hf 0.23.2's), :288 (encode is hf's: "a\x0fb" is 269 257 0 258, the byte fallback cannot reach "<0x+F>"). Mutant receipt: core.h:169 (the '+' branch) removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (flags 0) |
| VO20 | TESTED | 238 | "or a byte-level one-byte token" | test_vocab.c:162, :222 (gpt2 "!", "\xa2") |
| VO21 | TESTED | 239-244 | TOKS_ID_LSTRIP, RSTRIP, SINGLE_WORD, NORMALIZED: "an added id's options, as hf's added_tokens_decoder holds its AddedToken (the content listed last for the id, with that listing's options" | test_primitives.c:263 (on every added id toks_id_flags equals toks_added but for SPECIAL, and toks_added's are hf's: AD4), test_vocab.c:222 (the pins with NM and LR, test_vocab.c:231-232) |

Measurements (not counted): 220 "56 to 148 such strings per byte-level file" (tests/parity/vocab_sweep.py); 226
"5.5 to 9 bytes per id" (test_vocab's printed index size).

## AD: added tokens (toks.h:247-253)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| AD1 | TESTED | 247-248 | "the added tokens in id order, hf's get_added_tokens_decoder(): for i = 0, 1, ... the i-th id an added token holds (*id)" | test_primitives.c:260 (ids strictly increasing, below n_ids), :271 (the count and digest of every entry against hf's: 173 files, 40,436 entries), :275 (every id toks_id_flags calls ADDED is listed) |
| AD2 | TESTED | 248-249 | "the bytes of the content listed last for that id as the file writes them" | test_primitives.c:271 (hf's AddedToken.content in the digest: tests/data/spm/holes_added.json's id 42 is "hell", listed after "<D>"), :266 (the content finds the id) |
| AD3 | INSPECTION | 248-249 | "they live as long as ctx" | by inspection: vocab.c (the records and their contents are in the context's lookups block, freed by toks_unload) |
| AD4 | TESTED | 249-250 | "its flags as the return value: TOKS_ID_ADDED, the four above, TOKS_ID_SPECIAL as that listing says (hf's AddedToken.special" | test_primitives.c:271 (special, lstrip, rstrip, single_word and normalized in hf's digest), :260 (no other bit) |
| AD5 | TESTED | 250 | "differs from toks_id_flags' only for a content listed both special and not" | test_primitives.c:263 (every other bit equal on every added id; SPECIAL apart on 2 ids, tests/data/breadth/dup_added.json's) |
| AD6 | TESTED | 250-251 | "TOKS_ID_BYTE as toks_id_flags" | test_primitives.c:263 |
| AD7 | TESTED | 251 | "TOKS_E_ARG past the last one (i == their count) or for ctx NULL" | test_primitives.c:279 (ctx NULL, i == count, UINT32_MAX: -10, the outs untouched) |
| AD8 | TESTED | 251 | "content, len and id may be NULL" | test_primitives.c:282 |
| AD9 | TESTED | 252 | "A tiktoken file's specials: TOKS_ID_SPECIAL where its config names them, none of the four" | test_primitives.c:271 on Kimi K3 (its 256 specials against transformers' all_special_ids; the cache's directory, SKIP without it) |

## ST: stream decode (toks.h:255-264)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| ST1 | TESTED | 257 | `toks_stream { uint64_t opaque[8]; }`: 64 bytes | test_abi.c:181 (64 bytes, 8-aligned); that the state is those bytes: stream.c:23's _Static_assert (compile time) |
| ST2 | TESTED | 260 | toks_stream_bound "worst-case bytes of one push" | test_stream.c:314 (every push of every partition), :328 (flush), :675 (reached), :1139-1143 (its value per file), :816-821 |
| ST3 | TESTED | 261 | "push and flush are atomic: with cap below what the call needs they return TOKS_E_CAP" | test_stream.c:307 (push), :331 (flush), :652 |
| ST4 | TESTED | 261 | "and leave st as is" | test_stream.c:307, :331, :791-792 |

## SH: the stream's hold (toks.h:266-284)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| SH1 | TESTED | 266-267 | "hf decides a byte-fallback run ... as a whole: its chars when its bytes are valid utf-8, else one U+FFFD per byte" | test_stream.c:458 + :319 (decode and every push against the rule's reference), :1047 (12 x U+13000 then 0x80: 49 U+FFFD); nightly parity (the stream op, against hf's decode) |
| SH2 | TESTED | 267-268 | "a stream holds a run that is valid so far until a string or the flush ends it" | test_stream.c:319, :877 (30 run bytes: 0), :1226 (a run ended inside its push is not held) |
| SH3 | TESTED | 268 | "in st itself up to 44 bytes" | test_stream.c:1205, :1029 |
| SH4 | TESTED | 268-269 | "a push that would hold more returns TOKS_E_LIMIT" | test_stream.c:1207, :1212, :1031 |
| SH5 | TESTED | 269-270 | "the held bytes are copied into hold[0, cap) (any alignment)" | test_stream.c:882, :946, :951 (odd offsets) |
| SH6 | TESTED | 270 | "and the run's limit is then cap" | test_stream.c:469 (seq_walks through a hold of exactly the longest run: no refusal, its pushes checked at :319), :475 (one byte short: exactly one), :889 |
| SH7 | TESTED | 270-271 | "A run of c chars needs at most 4c bytes (1 KiB holds any run of 256 chars)" | test_stream.c:1037 (12 x U+13000 through 1 KiB: no refusal), :1043 (300 x: one), with SH6's exactness |
| SH8 | TESTED | 271 | "(a flush keeps the hold)" | test_stream.c:913 |
| SH9 | TESTED | 272-273 | "st and its hold are unchanged" after TOKS_E_LIMIT | test_stream.c:423, :889-890, :1031 |
| SH10 | TESTED | 273-275 | "a hold of at least the current one's size ... plus that push's n bytes takes it" | test_stream.c:428 + :437 (current + 1 for n = 1), :891 (64 + 5) |
| SH11 | TESTED | 274-275 | "the recovery is to grow the hold and push the same ids again" | test_stream.c:437, :893 |
| SH12 | TESTED | 275-277 | lifetime: after another toks_stream_hold moves the run out, the old hold is not read | test_stream.c:892 (the old hold overwritten) then :911 (the flush still equals decode) |
| SH12b | TESTED | 275-277 | ... nor written, and after toks_stream_init on st again neither | test_stream.c:992 (moved to hb, ha refilled by the caller: the pushes and the flush give decode's bytes, ha stays as filled), :998 (after toks_stream_init st equals a fresh stream), :1002 (hb refilled after the init: untouched, the output decode's). Mutant receipt: stream.c:442 keeping the old hold dies here and on master's own hold tests too (test_stream.c:438 on the mac, :897 on gb10a): no one-line mutant shows the gap; st is rebuilt from zero by inspection (stream.c:401, :408) |
| SH13 | TESTED | 277 | "(the current hold ... which copies from it: no realloc)" | test_stream.c:891 + :911, :951 (65 distinct bytes intact after each move) |
| SH14 | TESTED | 277-278 | "a copy of st shares its hold" | test_stream.c:921-922 |
| SH15 | TESTED | 278-279 | "hold NULL / cap 0 moves the run back into st's own 44 bytes" | test_stream.c:960 + :962 (equal to a stream that never left them), :968-971 (44), :976-977 (45: -9, unchanged), :886-887 |
| SH16 | TESTED | 279 | "Returns the bytes held (>= 0)" | test_stream.c:882, :891, :960, :428 |
| SH17 | TESTED | 279-280 | "TOKS_E_LIMIT when they exceed cap (44 for hold NULL / cap 0; st unchanged)" | test_stream.c:879-880 (a 29-byte hold for 30), :886-887 and :976-977 (NULL / 0 for 60 and for 45: -9, st and the hold unchanged) |
| SH18 | TESTED | 280 | "TOKS_E_ARG, TOKS_E_UNSUPPORTED as push" | test_stream.c:871-872, :874, :906-908, :789 |
| SH19 | TESTED | 281-282 | "ByteLevel and WordPiece streams ...: the call returns 0 and changes nothing" | test_stream.c:787-788, :847, :851-852 |
| SH20 | TESTED | 282-283 | "With a hold of cap bytes a push writes at most toks_stream_bound(ctx, n) + 3 cap bytes" | test_stream.c:314 (with :285's 3 cap), :410-437 |

Caller obligation (not counted): 280-281 "A push's out[0, cap) and the hold must not overlap".

## IN: info (toks.h:286-333)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| IN1 | TESTED | 288-291 | TOKS_ALGO_*: byte-level, sentencepiece-style, unigram, wordpiece | test_e2e.c:206 (byte-level, unigram), test_wp_e2e.c:66 (wordpiece), test_api.c:895; sentencepiece-style: test_spm.c:263 checks ctx->t.algo, which toks_get_info returns (api.c:772) |
| IN2 | TESTED | 294 | size "in: sizeof(toks_info)" (else -10) | test_api.c:891 |
| IN3 | TESTED | 296 | abi_major, abi_minor | test_api.c:894 (major); nightly python/tests/test_api.py:60 (both, against the header's) |
| IN4 | TESTED | 298 | tier "TOKS_TIER_* in use" | test_tier.c:241, :250, test_e2e.c:328 |
| IN5 | TESTED | 299 | n_ids "every id is < n_ids" | test_e2e.c:206, test_api.c:894, :806 (n_ids itself: -7) |
| IN6 | TESTED | 300 | n_added | test_e2e.c:206 |
| IN7 | TESTED | 301, 327 | TOKS_PATH_SCAN "pre-tokenizer on a compiled template (else the generic engine)" | test_breadth.c:53 |
| IN7b | TESTED | 328 | TOKS_PATH_NORMALIZE "normalizer on the compiled path" | test_e2e.c:211 (every pinned file: TOKS_PATH_NORMALIZE set; pinned so a change is a decision). Mutant receipt: api.c:776 without the bit survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| IN8 | TESTED | 302 | cpu_features "the platform layer's feature bits" | test_api.c:897 (the load-time word, no probe after load), test_load.c:166 |
| IN9 | TESTED | 303 | max_text "TOKS_MAX_TEXT" | test_api.c:896 |
| IN10 | TESTED | 304 | control_isolation "1 when SPEC §3.6 is certified for this tokenizer" | test_e2e.c:211 (0 for every pinned file: nothing is certified in 0.3). Mutant receipt: api.c:779 made 1 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here. **Finding** (stays): the header could say it is 0 in 0.3 |
| IN11 | TESTED | 306 | source_sha256 | test_e2e.c:207 (14 files' sha-256 pinned); nightly python/tests/test_api.py:59 |
| IN12 | TESTED | 307 | image_sha256 | test_api.c:896 (zero); nightly python/tests/test_api.py:61. **Finding**: the header does not say it is all zero (no image) |
| IN13 | TESTED | 308 | name "NUL-terminated" | test_e2e.c:207 (the file's name), :270 (a directory's name) |
| IN13b | TESTED | 308 | a name longer than 63 bytes is cut and terminated | test_e2e.c:291 (gpt2 from a model directory of a 100-byte name: info.name its first 63 bytes, terminated; posix hosts). Mutant receipt: load.c:311's 63 made 62 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| IN14 | TESTED | 294-295 | size "or abi 0.3's 184 (offsetof(toks_info, trunc_on)): then only the fields before trunc_on are written" | test_primitives.c:394 (184: the 180 bytes after size are the full call's, bytes 184 and 239 untouched), :403 (0, 4, 183, 185, 239, 241, 4096: -10) |
| IN15 | TESTED | 309-320 | trunc_on, trunc_max, trunc_stride, pad_on, pad_fixed, pad_id, pad_type_id, pad_len, pad_multiple, pad_left: "the file's truncation and padding as hf's Tokenizer.truncation / .padding report them" | test_primitives.c:385 (every field against hf's dicts, 173 files: BatchLongest alone, Fixed, multiples, Left, types_left_pad's pad_type_id 5) |
| IN16 | TESTED | 321-323 | n_template_prefix, n_template_suffix, seq_type_id | test_primitives.c:387 (hf's template: TP1, TP4) |

toks_version (toks.h:332) is row V5.

## PA: toks_par (toks.h:335-387)

| ID | status | toks.h | clause | the test, or the gap |
|---|---|---|---|---|
| PA1 | TESTED | 337-338 | "each with its own scratch, which the pool grows to the largest unit it meets" | test_par.c:183 (inputs up to 1.3 MB after small ones, through pools whose scratches start small, equal serial) |
| PA1b | INSPECTION | 338 | "and keeps" | by inspection: par.c scratch_fit (a scratch never shrinks) |
| PA2 | TESTED | 338 | "every result equals the serial toks_encode call's (T2)" | test_par.c:183 (one input), :303 (batches), every mode, fixtures and real files; test_state.inc:213 (the par cell) |
| PA3 | INSPECTION | 338-340 | "A call uses only as many participants as its work justifies, by a cost model the pool measures" | by inspection: par.c's cost model; its observable edges are PA4 and PA35 |
| PA4 | TESTED | 340-341 | "every call under 16 KiB ... runs on the caller alone" | test_par.c:186, :400 |
| PA5 | INSPECTION | 341-343 | "Work is cut into units ... that the participants claim in order from one queue" | by inspection: par.c (the units and the queue) |
| PA6 | INSPECTION | 343 | "Idle workers spin about as long as a wake costs, then sleep" | by inspection: par.c (spin_ns); test_par.c:354 checks the woken workers' results |
| PA7 | INSPECTION | 343-344 | "a call wakes exactly the workers it uses" | by inspection: par.c (the wake) |
| PA8 | TESTED | 344 | "no thread is created per call" | test_par.c:514 (an eager pool of 4 on 1 MiB: the process's thread count after its first wide call and after 20 more is the same, and at most threads - 1 above the count before create). No one-line mutant: a thread per call is not one line |
| PA9 | TESTED | 344-345 | "One call at a time per pool (concurrent calls wait for each other)" | test_par.c:349 |
| PA10 | TESTED | 349 | toks_par_create "a pool of at most n_threads participants ..., the caller included" | test_par.c:419 (pools of at most 1, 2, 3, 8: threads <= n_threads), :421 (a call's participants <= threads). Mutant receipt: par.c:922 `n = n_threads + 1u` survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| PA11 | TESTED | 349-350 | "(0: a couple ...)" | test_par.c:396 (1..4) |
| PA11b | TESTED | 349-350 | "min(4, the fast cores the process may run on)" | test_par.c:487 (the default pool: threads == min(4, fast)). Mutant receipt: par.c:923's 4 made 3 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here (master's check accepts 1..4) |
| PA12 | TESTED | 350 | "never more than the cpus it may run on" | test_par.c:492 (1024 asked: threads <= the online cpus, which bound the cpus the process may run on). Mutant receipt: par.c:924's clamp removed survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| PA13 | INSPECTION | 350-351 | "Workers prefer the fast cores" | by inspection: par.c topo_read and thr_start (the platform's placement) |
| PA14 | TESTED | 351-352 | scratch_flags "every participant's toks_scratch_init flags": checked like init's | test_par.c:408 (a bad cache size: -10) |
| PA14b | INSPECTION | 351-352 | ... and given to every participant's scratch | by inspection: par.c (scratch_fit initializes with p->scr_flags) |
| PA15 | TESTED | 352 | "0, TOKS_E_ARG or TOKS_E_NOMEM" | test_par.c:405-408 (-10: out NULL, ctx NULL, too many, flags), :78 (0); TOKS_E_NOMEM: PA26 |
| PA16 | TESTED | 354 | toks_par_destroy "NULL-safe" | test_par.c:413 |
| PA16b | TESTED | 354 | "joins the threads" | test_par.c:533 (after toks_par_destroy the process's thread count is back where it was before create, polled up to 1 s: a joined thread can stay counted for a moment, on macOS and linux alike). A destroy that neither wakes nor joins its threads also crashes master's create / destroy cycles (test_par.c:424); the join itself: by inspection (par.c:970) |
| PA17 | INSPECTION | 357 | item text "read in place" | by inspection: par.c:409 (toks_encode on it->text) |
| PA18 | TESTED | 359 | item out "this input's ids go straight here" | test_par.c:301 |
| PA19 | TESTED | 361 | item n "toks_encode(...)'s return value" | test_par.c:303 (counts, E_ARG and E_LIMIT items) |
| PA20 | TESTED | 364 | toks_par_encode_batch "every item encoded as toks_encode would" | test_par.c:303 |
| PA21 | TESTED | 364-365 | "0 (each item's result in its n)" | test_par.c:290, :383 |
| PA22 | TESTED | 365 | "or TOKS_E_ARG for a bad pool / flags / array" | test_par.c:366-368 |
| PA22b | TESTED | 365 | "(then no item is touched)" | test_par.c:378 (n and out canaries unchanged after mode 3, flags 16, 64, 1 << 31 and a NULL pool). Mutant receipt: par.c:997 resetting the items' n before the check survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| PA23 | TESTED | 368-369 | toks_par_encode "split at toks_split_points' cuts ... assembled in out" | test_par.c:183 |
| PA24 | TESTED | 369 | "Returns exactly what toks_encode returns, out[0, min(cap, n)) the same ids" | test_par.c:183 (caps 0, n / 2, n, n + 40), :360-365 (the errors) |
| PA25 | TESTED | 369-370 | entries in [min(cap, n), cap) "may be overwritten" (and none at >= cap) | test_par.c:182-183 (64 canaries past cap) |
| PA26 | TESTED | 370 | "TOKS_E_NOMEM when a worker's scratch cannot grow" | test_alloc.c:349 (a pool whose arena requests fail after create: toks_par_encode of 1 MiB returns -11), :351 (the allocator back: the same call equals serial). Mutant receipt: par.c:369 NOMEM made LIMIT survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| PA28 | TESTED | 377 | size "in: sizeof(toks_par_info)" | test_par.c:387 (sizeof - 1: -10). Note: par.c:1023 accepts a larger size; toks_get_info wants it exact |
| PA29 | TESTED | 378 | threads "participants at most, the caller included" | test_par.c:419 (PA10's) |
| PA30 | TESTED | 379 | fast "the fast cpus the process may run on" | test_par.c:486 (fast >= 1 and <= the online cpus). Mutant receipt: par.c:321 making fast n_cpu + 1 survives master 245cc5c's make test (gb10a: GB10, A725 cores) and dies here |
| PA31 | TESTED | 380 | last "participants of the last call" | test_par.c:186, :357 |
| PA32 | TESTED | 381 | ns_per_mib "the measured encode cost" | test_par.c:397 (measured before the first call: > 0) |
| PA33 | TESTED | 382 | wake_ns | test_par.c:397 |
| PA34 | TESTED | 383 | join_ns | test_par.c:397 |
| PA35 | TESTED | 384 | min_bytes "(0: never)" | test_par.c:398 (0 for a pool of one, else >= 16 KiB) |
| PA35b | TESTED | 384 | "the smallest call that would take a second participant now" | test_par.c:525 (an eager pool of 4, whose decision reads no clock: a call of min_bytes on text with cuts takes two or more participants, one byte less runs on the caller alone; 16384 on every host here). Mutant receipt: par.c:1043 `hi` made `hi + 1u` survives master 245cc5c's make test (gb10a: GB10, A725 cores; master's check, test_par.c:398, accepts 0 or >= 16 KiB) and dies here. A cost-model pool's min_bytes depends on its clock and measurements: not pinned |
| PA36 | TESTED | 387 | toks_par_get_info "0 or TOKS_E_ARG" | test_par.c:87, :387, :389 |

Caller obligations (not counted): 350 "ctx must outlive it"; 359 "no two items' out ranges may overlap".
Definition (not counted): 375 TOKS_PAR_HAS_INFO.

## Findings (code or header, not tests)

- split.c:304 wrote offs through a `uint64_t *`, undefined behaviour for offs at an odd address, which toks.h:8
  allows (G2b): fixed by PR#241 (2cfeb1c, toks_st64), with test_misalign.c:93.
- toks.h:174-179: r's list ended "everything else 1, and tight", but a sentencepiece context with a prepended ▁ and a
  one-byte added token has r 2 (the phase terms, compile.c:608-609): PR#239's bound_phase0 / bound_phase1, where hf
  0.23.2 gives 3 ids for the 2 bytes "#a". The code is right (hf attains more than 1 id per byte); the header now
  names the phase terms (this PR, toks.h:178-179: "at least (1 + p) / l where an added token of l bytes can be
  followed by a prepended ▁ of p ids"; CA6b).
- toks.h:67: TOKS_MAX_TOKEN_BYTES read as 65535 for every family, but unigram refuses a piece over 127 bytes (-3,
  named: unigram.c:381); the header now says so (this PR: "unigram pieces: 127, -3 above"; L2c).
- toks.h:4: in 0.x additive changes bumped TOKS_ABI_MINOR; the sentence names only TOKS_ABI_MAJOR (V1).
- toks.h:50: TOKS_E_FORMAT's comment still names the .toks image and its checksum (E3).
- toks.h:304, 307: control_isolation is always 0 and image_sha256 always zero in 0.3; the header does not say so.
- toks.h:132-135: mode 3 is TOKS_E_ARG (test_api.c:488, test_split.c:258, test_par.c:355); the header lists three
  modes and says nothing of the fourth value.
- toks.h:330: toks_get_info's return (0; TOKS_E_ARG for NULL or a wrong size) is not written in the header; the size
  rule differs from toks_par_get_info's (exact vs at least).
- toks.h:349-353: toks_par_create's TOKS_E_ARG for n_threads above 1024 (par.c:87, :915) is not written.
- toks.h:14: the after-load rule is the core's; toks_par allocates, locks and starts threads (G7).
- TOKS_MAX_IDS (toks.h:69, "every id is < TOKS_MAX_IDS"): spm_build.c:273 and config.c:1326 refuse n_ids ==
  TOKS_MAX_IDS, so an id of 2^21 - 2 is -9 for sentencepiece-style and unigram files (holes_added with a vocab id
  2097150: -9 "ids or merges beyond the table widths"; 2097149 loads); bpe_build.c:325 and tiktoken.c:47 allow it.
  Ruled: a 0.3.1 one-character fix (`>` in the two checks) with its boundary test; no shipped or cached file is near
  it (L4c, a severity-3 gap until then).
- Not a finding, for whoever reads load-time memory: a wordpiece load whose double-array tries cannot be allocated
  loads without them (wp.c:473, :485: the hash-probe path, the same ids): 12 of wp-minilm-l6's 18 load requests;
  1 of llamalike's 8 has a fallback too. test_alloc checks the ids; the speed of such a context is not checked.
- Audit corrections (made while closing the gaps, each with a master receipt): config.c:234 was already reached by
  test_load's regress repro nomem-generic-051fe215.json (L4b); test_targets.c parses and compiles the qwen1-72b
  files, but not through toks_load (LD2b).

## The gap tests, in order (tests/c and tests/data only)

1. **Severity 1** (done: VO7, VO9, VO13b, T3b).
2. **Severity 2** (done, but G6: a cf_audit rule in tests/abi, "no writable data in src/core and src/gen objects";
   a SEAM for the hardening lane, after the flip unless it is a one-line rule with a receipt).
3. **Severity 3** (done: G2b, SP12, SH12b, IN7b, IN10, IN13b, PA8, PA11b, PA12, PA16b, PA30, PA35b, VO19b), but L4c
   (0.3.1's one-character fix brings its test).
