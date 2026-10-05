# gigatoken encode path — a brief for toks

Source: https://github.com/marcelroed/gigatoken @ fac0114 (Rust, MIT). Read-only analysis; this doc cites file:line for every
claim. Numbers are quoted from gigatoken's own benchmarks/profiling notes unless labeled "reasoning".

Scope: how gigatoken encodes (single doc, batch, files), what it caches, how BPE runs, what loaders accept, how
it scales across cores, and what it measured — including its own negative results. Ends with what toks should do
about it.

## 1. Data flow of one encode

API surface (src/lib.rs:49-347): a `BPETokenizer` wraps one Rust `Tokenizer` (src/bpe/tiktoken.rs:339) plus a
`WorkerPool` (src/batch.rs:651) of lazily forked per-thread tokenizers. Four entry shapes:
- single doc: `encode`/`encode_into` (lib.rs:18) → pretokenize the whole slice, then BPE per pretoken;
- batch: `encode_batch` → `batch::encode_docs_ragged` (batch.rs:760) — rayon over byte chunks;
- files: `encode_files` → input/file_source (mmap, jsonl, parquet; `chunk_ranges`) → same ragged engine
  (batch.rs:934 `encode_files_docs`);
- serial twins of each (`*_serial`) for the `parallel=False` path.

The encode core is `Tokenizer::memoized_encode*` (tiktoken.rs:1292 `memoized_encode`,
tiktoken.rs:1339 `memoized_encode_flat`). Both consume a `PretokenSpans` iterator (pretokenize/mod.rs:389) NOT as an
`Iterator`: pulling happens in a dedicated out-of-line loop so the pretokenizer state stays register-allocated —
inlined into the register-starved emit loop it lived in stack slots at ~9 cycles/pretoken of spill traffic on Zen 2
(mod.rs:361-369). The engine loops in chunks of `PRETOKEN_CHUNK = 256` pretokens (mod.rs:53) through two phases
per chunk (tiktoken.rs:1332-1338):

**Phase A — fill + key + prefetch staging** (`fill_spans_keyed`, mod.rs:396). The fast path is
`MaskState::fill_spans_two_phase` (fast/mask.rs:1047), dispatched by x86 tier to `_avx512_vbmi2_crc` (Zen 4/5,
Ice Lake+), `_avx512_crc`, `_avx2_crc` (Haswell+; sse4.2 required for the CRC hash), or scalar (mask.rs:1074-1092).
Its body `fill_spans_two_phase_impl` (mask.rs:1189) runs the two-phase chunk pipeline:
- **Phase A1 — harvest boundaries into a u16 buffer.** Per 64-byte batch of input, the scheme's
  `batch_masks`/`batch_masks_x86` produce a `usable` (token-start) u64 mask and a `bad` (hard-zone) mask.
  Clean batches flatten set-bit positions to u16 offsets relative to the fill base (`flatten_bits`,
  mask.rs:880 — NEON `vld1q_u16(BIT_POS[b])` + `vaddq`; `flatten_bits_avx512` for VBMI2 = `vpcompressb` of iota,
  mask.rs:934, with a `flatten_bits_dispatch` tier switch at mask.rs:973). Dirty batches (any bad bit) re-walk the
  zone with the scalar `S::advance` ground truth, keeping prefix usable bits (mask.rs:1303-1358). A scalar end
  leaving the u16 window (`REL_LIMIT`, mask.rs:1020: `u16::MAX - 127`) is dropped and re-derived next fill; zero
  boundaries in the window means a >65 KB pretoken, emitted alone (mask.rs:1361-1381).
- **Phase A2 — flat key build with no data-dependent branch** (mask.rs:1383-1462). One hoisted check
  `fill_base + last_end + 16 <= len` proves every 16-byte key load in-bounds. Per span: plain 16-byte `read_unaligned`
  load (address depends only on `prev`, so it issues early), `PACK_MASK_TABLE` row lookup (a 16×2 u64 table replacing
  a 7-op ALU chain per half — mod.rs:55-71), a `keep` AND-mask routes long spans to key 0 without a branch
  (mask.rs:1427-1433), then the hash and the L2 **prefetch**: `prefetch(hv)` is `ShortPretokenCache::prefetch_l2`
  (pretoken_cache.rs:231) — `_mm_prefetch(T1)` / `prfm pldl2keep` — a whole chunk (hundreds of cycles) ahead,
  enough to cover DRAM, without evicting the span walker's L1 working set (pretoken_cache.rs:226-229). Recomputing
  the hash at the consumer instead of storing it measured 4% slower end-to-end (mod.rs:393-395).
  Entries are 32 B `BatchEntry { key: u128, ptr, meta }` records (mod.rs:294-300), written with two `stp`s; `meta`
  carries the hash for short spans (key≠0) or the byte length for long ones (key=0) — mod.rs:277-286.
  `SPAN_BATCH_SLACK = 16` readable entries past the chunk let the emit loop's `entries[i + D].meta` prefetch load
  skip a per-iteration index clamp (mod.rs:310-315).

**Phase B — probe/emit** (`Tokenizer::probe_emit_chunk`, tiktoken.rs:1375). L1 prefetch distance `D = 16`
(tiktoken.rs:1400, asserted ≤ SPAN_BATCH_SLACK): each iteration first promotes the pair's line L2→L1 via
`table.prefetch(batch.entries[i + D].meta)` (tiktoken.rs:1413), then `probe_pair(key, h)` (pretoken_cache.rs:432)
loads both slots of the home pair — one cache line — and folds the two key compares + register-value `cmov`s
(aarch64 `csel` asm at pretoken_cache.rs:446; x86 `cmovne` at pretoken_cache.rs:472; the asm exists because LLVM
canonicalizes every Rust spelling into an address-select feeding a second dependent load). The fast predicate
`found & !spilled & key != 0` (tiktoken.rs:1422) covers ~99% of pretokens: the four token lanes are stored
UNCONDITIONALLY at the write cursor as one fused 16 B store (tiktoken.rs:1427-1433), the cursor advancing by the
packed count only when fast — dead stores past the cursor get overwritten later or truncated by `set_len`
(tiktoken.rs:1358-1373). Everything else — displaced keys, arena spills, long pretokens, misses — takes the
`#[cold] probe_emit_slow` (tiktoken.rs:1466), which re-walks via `get_or_slot`, serves arena spills
(`token_arena` slice by `(val >> 32)` offset), or lands in `encode_pretoken_miss` (tiktoken.rs:1584): BPE-encode the
bytes, pack the result into the cache (`insert_at` — the slot from the failed probe walk, skipping a re-walk), spill
to `token_arena` if >4 tokens, and append. Slack invariant `out.capacity >= cursor + 4·(iterations left)` keeps the
two 8-byte stores in bounds (tiktoken.rs:1370-1373, re-established at tiktoken.rs:1515).

Per-pretoken delivery (`memoized_encode`) instead records `ends[i] = w` and slices `out` per pretoken
(tiktoken.rs:1307-1319).

## 2. Pretoken cache

`ShortPretokenCache` (bpe/pretoken_cache.rs:157): open-addressing linear probe over power-of-two slots for
pretokens ≤ 15 bytes.

- **Entry layout** (pretoken_cache.rs:26-48): exactly 32 B `Entry { key: u128, val: u64, ext: u64 }` — two slots per
  64 B line, never straddling one; `key` is the packed pretoken (bytes in low 15 lanes, length in the top byte —
  `pack_pretoken_key`, mod.rs:92; page-boundary spans take a plain ≤15-byte copy). Key 0 = empty sentinel.
- **Hash** (mod.rs:141 `pretoken_key_hash`): aarch64 `__crc32d`×2 (compile-time `crc` feature; Linux builds need
  `-C target-feature=+crc` or silently fall back); x86-64 CRC32C per process (`crc_hash_selected`, mod.rs:230);
  else a multiply fold `pretoken_key_hash_fold` (mod.rs:177). All arms map key 0 to hash 0 and may never mix in
  one process image (grow's rehash must agree — mod.rs:124-139).
- **Probe** (`probe_pair`, pretoken_cache.rs:432): a bucket is the slot pair `idx & mask & !1` — both slots share one
  64 B line; branch-free pair probe from that single line. On a miss the lanes are another entry's (discarded by the
  predicate); the slow path disambiguates displaced keys from real misses via `get_or_slot` (pretoken_cache.rs:255),
  which also returns the first empty slot for the insert.
- **Replacement** (`replace`, pretoken_cache.rs:331): insert-or-overwrite for the loader phase — an added token's
  content can duplicate a seeded vocab byte string and must take over its entry (used by `set_added_tokens` and
  `fork_sized`'s re-apply).
- **Insertion**: `insert_at` (pretoken_cache.rs:312) writes the slot the failed probe reported; growth doubles at
  3/4 load (`grow`, pretoken_cache.rs:357, rehashing through `pretoken_key_hash`); `clear` zeroes in place
  (generation wipe); `reset_to_capacity` reuses the allocation when the cap matches.
- **Seeding** (`seeded_pretoken_cache`, tiktoken.rs:468): every vocab entry of 1..=15 bytes pre-inserted with its
  **BPE result computed by the same `merge_short` the miss path runs** — bit-identical to a cold miss on those
  bytes. Without `ignore_merges` the seed value MUST be the merge result, not the entry's own id: vocabs contain
  merge-UNREACHABLE entries — qwen3_5 has ~200 (multi-char CJK phrases, " Japón", …) plus ~50 entries > 15 bytes —
  that HF encodes as their merge decomposition. Seeding `bytes -> [own id]` **was a measured divergence from HF**
  (tiktoken.rs:443-455, test `verify_vocab_seeded_cache_matches_merge_decomposition`). Merge-reachable entries — all
  of gpt2/olmo3/qwen2/deepseek_v3 — degenerate to `[own id]` anyway. WITH `ignore_merges` the rule flips: every
  seed value is `[vocab_inv[bytes]]`. 5+ token values (only possible for unreachable entries) spill to the arena.
- **Value packing** (`pack_val`/`pack_val_inline`, tiktoken.rs:196-215): `val` = count in bits 0-6, spill flag bit 7,
  tokens 1-2 in bits 8-63, `ext` = tokens 3-4; inline covers 1-4 tokens with one dependent load and no second random
  access. Spilled: arena offset in `val >> 32`. Rationale (pretoken_cache.rs:1-21, measured 1 GB OWT, Zen 2): the
  table holds ~1.3M unique pretokens (~99.4% hit rate), ~90% of pretokens encode to one token, ~98% to ≤2;
  228M output tokens / 208M pretokens.
- **Spill arena**: `token_arena` (a `Vec<TokenId>` shared by short spills and the long map) — long pretokens
  (> 15 bytes, rare) always spill and live in `pretoken_cache_long`, a plain `FxHashMap<Box<[u8]>, (u32 offset,
  u32 len)>` (tiktoken.rs:1503, insert at tiktoken.rs:1692).
- **Budget** (`CacheBudget`, tiktoken.rs:102-165, default `set_max_cache_bytes = 512 MiB` per worker,
  tiktoken.rs:343): short-table slot ceiling = largest power of two with slots·32 ≤ 70% of budget (seed requirement
  can push higher; headroom floor keeps seed ≤ 5/8 of ceiling so a generation admits ≥ cap/8 entries —
  tiktoken.rs:145-151), remaining split evenly between arena entries and long-map bytes (48 B/entry estimate).
  At the ceiling the 3/4 threshold triggers a **generation wipe** (`wipe_generation`, tiktoken.rs:1034) instead of
  growth; `max_encoding` slack lets one recurring giant pretoken not force a wipe per occurrence.
- **Sizing for workers** (`fork_sized`, tiktoken.rs:759): Heaps' law estimate distinct(n) ≈ 3.45·n^0.62 from
  ~1.3M distinct at 1 GB / ~5.5M at 10 GB OWT (tiktoken.rs:760-761), sized at 3/4 load with 1.4× headroom so a
  >2×-oversubscribed worker never rehashes; a default-sized worker rehashes through 6-7 doublings of scatter
  writes on every worker at once (tiktoken.rs:749-757).
- **Huge pages** (`Slots::new_zeroed`, pretoken_cache.rs:64-93): manual 2 MiB-aligned allocation,
  `madvise(MADV_HUGEPAGE)` **before** the zeroing memset — `alloc_zeroed`'s memset faults fresh pages in as 4 KiB
  first and the hint becomes a no-op for the run (khugepaged collapses too slowly); the table then walks the dTLB
  on every probe and Zen drops software prefetches that miss the dTLB — **+15% cold / +7% warm from this ordering
  alone** (pretoken_cache.rs:71-79, profiling/zen5_st_profile.md §3). Align = min(2 MiB, next_pow2(size)).max(64)
  so small tables stay modest. Note: `PR_SET_THP_DISABLE` sandboxes silently get 4 KiB pages anyway
  (pretoken_cache.rs:22-25).

## 3. BPE engine

All in src/bpe/mod.rs. Three merge cores, deliberately not unified ("unification was measured as a regression
risk to the tuned short-miss path", mod.rs:459-464, mod.rs:540-542):

- **short stack cores** for ≤ 15 symbols (`SHORT_MERGE_MAX = 16`, mod.rs:531): tiktoken-style — per-position rank
  array (merged token id of the pair starting there, or `u32::MAX`), find min by linear scan, merge, refresh only
  the two affected neighbor ranks over a stack-resident doubly-linked u8 list (mod.rs:548
  `bpe_merge_symbols_short_scalar`; NEON twin at mod.rs:647 `bpe_merge_symbols_short_neon` packs `(rank, position)`
  into one u32 per lane and finds the min with `vminq_u32`×2-3 + `vminvq_u32`, only the first half of the array for
  n ≤ 8 — mod.rs:640-680).
- **Vec-based small core** for 16..=32 symbols (`SMALL_MERGE_MAX = 32`, mod.rs:449, `bpe_merge_symbols_small`,
  mod.rs:465): same algorithm over `Vec<TokenId>` — the long-pretoken miss path.
- **heap core** for > 32 (`bpe_merge_symbols_by_rank`, mod.rs:343): BinaryHeap of packed `(merged_id:32, pos:32)`
  u64 entries (`pack_merge_entry`, mod.rs:304) + u32-index linked list, entries validated on pop against the
  current pair (lazy deletion), O(n log n).

Thresholds: the dispatch is `n <= SMALL_MERGE_MAX → small` (mod.rs:359-363); the tiktoken encode miss path routes
≤ 15-symbol pretokens straight to the short stack cores (`merge_short`, tiktoken.rs:645-695).

- **Merge table layout — `PairRankTable`** (mod.rs:153-220): two-level structure. (a) **Dense "byte-pair grid"**:
  `2^11 × 2^11 = 2^22` u32 cells covering every pair with both sides < 2^11 — the initial byte tokens (0..255 GPT-2,
  3..258 DeepSeek) AND the earliest ~1.8k merged IDs, which by merge order are the most frequent symbols. 16 MiB,
  L3-resident, Arc-shared across forks. Round-1 lookups plus most mid-merge lookups (hits and no-merge misses)
  become one shift-or index and one load (mod.rs:177-190). (b) **Flat open-addressing level**: all merges (superset
  of the dense subset, complete on its own) at ≤ 1/2 load; each u64 slot packs `key = (a << 21) | b` in the top 43
  bits and the merged id in the low 21 (`PAIR_ID_BITS = 21`, "covers any vocab < 2M IDs", mod.rs:150-151); the
  empty sentinel's key field exceeds every real key so it never false-hits (mod.rs:238-247). Probe: multiply
  `0x9E3779B97F4A7C15` + shift, linear walk (mod.rs:232-249). Build returns `None` (→ HashMap fallback) if
  vocab > 2^21 or a pathological displacement > 64 (mod.rs:166-212). `prefetch_rank` (mod.rs:264) issues `prefetcht0`
  on the refresh pairs before the list surgery — the refresh lookups' consumers are ~5-7% of cold cycles on Zen 5
  (profiling/zen5_st_profile.md §4.3).
- **id-as-rank detection**: the fast loops take the merged token's ID as merge priority, correct only when the merge
  list produces IDs in rank order (every tiktoken-style vocab: GPT-2, cl100k, o200k, Qwen, Llama-3). The loader
  checks `entries.is_sorted_by_key(merged)` (hf.rs:750) and otherwise builds a `RankedMerges`
  `HashMap<u64 ranked_merge_key, (merged, rank)>` (hf.rs:744-767) — Fairseq-heritage vocabularies (RoBERTa/OPT/
  DeBERTa) order IDs by corpus frequency and carry explicit list position as rank. In ranked mode every merge runs
  through the ranked loops and the id-as-rank fast paths stay off (tiktoken.rs:354-357, miss path
  `encode_pretoken_miss_ranked` tiktoken.rs:1524). The ranked pair key `ranked_merge_key(a,b) = (a<<32)|b`
  hashes with a single multiply (mod.rs:911-916).
- **`ignore_merges`** (HF BPE semantics): a pretoken whose whole byte string is a vocab entry emits that entry's id
  directly (hf.rs:102-106; applied in the seed at tiktoken.rs:457-460 and in the long miss path tiktoken.rs:1662-1665).
- **`byte_fallback`**: rejected in the ByteLevel loader (`ensure!(!tj.model.byte_fallback, "byte_fallback
  tokenizers should use load_hf_sentencepiece instead")`, hf.rs:698-701); implemented in the SentencePiece engine:
  `init_symbols` maps each char to its vocab piece, else falls back to the char's UTF-8 bytes through
  `byte_fallback_ids` (sentencepiece.rs:1468-1490).
- **SentencePiece path** (src/bpe/sentencepiece.rs:230+): a separate `SentencePieceTokenizer` mirroring SP-BPE
  with byte_fallback: normalizer ops (`NormOp`: Prepend, Replace, Strip, Precompiled charsmap via spm_precompiled,
  CollapseSpaces for `" {2,}"` — hf.rs:432-490), Metaspace handling (prepend_scheme, hf.rs:507-563), then
  ranked-merge BPE over pieces (`bpe_merge_symbols_ranked`, mod.rs:994). It has its own pretoken cache
  (`SentencePieceEncoder` with budget + wipe, sentencepiece.rs:918-967) keyed on the front of the piece
  (front_index, sentencepiece.rs:881).
- **Added tokens** (`set_added_tokens`, tiktoken.rs:1059): matched atomically in the RAW input before
  pretokenization, mirroring HF's AddedVocabulary, including lstrip/rstrip whitespace stripping
  (apply_added_token_overwrites tiktoken.rs:318, trim_ws_start/end tiktoken.rs:268/291). Added tokens containing a
  space block parallel-split cuts (batch.rs, see §5).

## 4. Loader coverage (hf.rs, tiktoken.rs)

- **Format probe** (hf.rs:203-262): `model.type` is probed BEFORE the BPE-shaped schema — WordPiece/Unigram are
  refused by name ("Unsupported model type ... gigatoken supports BPE tokenizers (byte-level, or SentencePiece-style
  with byte_fallback)"); untyped legacy files are classified by `unk_id` (Unigram, e.g. t5-small, xlm-roberta) and
  `max_input_chars_per_word` (WordPiece, e.g. bert-base-uncased) family markers. Parse errors inline the
  deserializer's own message (sonic_rs).
- **`load_hf_bpe` / `build_bpe`** (hf.rs:688-787): model.type must be `BPE`; `byte_fallback` must be false. Vocab
  built by ID with byte→unicode reversal (GPT-2 byte-level tables, hf.rs:654-687); added tokens extend the vocab
  (hf.rs:180-202). Merges resolved through the byte-decoded vocab strings, missing pieces skipped. Then:
  pretokenizer, NFC normalizer, add_prefix_space, ignore_merges, added tokens — each detected or refused
  explicitly.
- **Pretokenizer coverage** (hf.rs:596-638, options.rs:106-142): `Split` regexes are collected from the
  pre_tokenizer (including `Sequence` children) and matched against a fixed table — GPT2/r50k, GPT4/cl100k, Qwen2
  (Qwen3), Qwen35, Olmo3, DeepSeekV3 (exact three-Split sequence), O200k (GPT-4o, gpt-oss), Nemotron, Kimi
  (options.rs:10-20). A bare `ByteLevel` maps to GPT2; **no `Split` regex in the chain is an error** ("no fast
  pretokenizer for ..."); an unknown regex is an error, never silently mis-encoded (hf.rs:627-638).
- **Normalizers**: NFC detection accepts `NFC` / `Sequence(NFC + Lowercase + NFC...)`-style chains and refuses
  anything else by name (hf.rs:577-598). SentencePiece normalizers: Prepend, Replace (literal, or exactly the
  `" {2,}"` regex), Strip, Precompiled charsmap; anything else errors (hf.rs:430-490). Metaspace prepend_scheme
  beyond supported values errors (hf.rs:523).
- **Non-id-as-rank handling**: see §3 — `is_sorted_by_key(merged)` check at hf.rs:750; ranked path otherwise.
- **tiktoken loader** (load_tokenizer/tiktoken.rs:31-54): base64-per-line rank file; ranks must be DENSE (line i
  must have rank i, else error); merges are RECONSTRUCTED by running simple BPE over each vocab entry
  (`from_ranks`, tiktoken.rs:711-739 — every ≥2-byte token must decompose to exactly 2 symbols, asserted at
  tiktoken.rs:733). The pretokenizer scheme and special tokens MUST be supplied by the caller — "a .tiktoken file
  carries neither ... a wrong scheme silently changes every encode, so neither may be defaulted" (tiktoken.rs:23-30);
  special-token ids must sit above the mergeable ranks.
- **SentencePiece loader** (`load_hf_sentencepiece`, hf.rs:275, `build_sentencepiece` hf.rs:279-428): normalizer
  ops → Metaspace → vocab with `<0xXX>` byte-fallback pieces (hf.rs:158-166) → ranked merges.
- **hub.rs**: fetches tokenizer files from the HF hub (not examined in detail).

## 5. Threading and multi-core scaling

**Work split.** `batch.rs`: documents are grouped into coarse chunks — `MIN_CHUNK_BYTES = 1 MiB` (batch.rs:23, "a
chunk this size encodes for tens of milliseconds"), chunk target ≈ total/(16·threads) floored at 1 MiB so rayon
gets ~16 chunks/thread for work-stealing (batch.rs:30-33). `build_doc_chunks` orders chunks LPT-style:
~2×-target chunks over the first ~80% of bytes, quarter-target over the last ~20% (batch.rs:173-178). Chunks are
handed out by an **atomic-counter in-order pull loop** (one rayon task per thread), NOT `par_iter` — rayon's
recursive range splitting let a thread steal a big head chunk late, causing a 104 ms straggler spread on M4 Max;
in-order handout bounds it at one tail chunk (mt_round3_findings.md, fix verified in round 4: 104 → 23 ms).
`safe_split_ranges` (pretokenize/mod.rs:570-660) cuts documents at a space preceded by an ASCII alnum and followed
by an ASCII letter — a boundary under every supported regex; added tokens containing a space (memmem finders) and
rstrip tokens ending exactly at a cut block it.

**Per-thread tables.** Each rayon slot gets a lazily-forked tokenizer (`WorkerPool`, batch.rs:651) with its OWN
pretoken cache — `fork_sized(expected_bytes)` pre-sizes it from the Heaps' law estimate so workers skip the 6-7
doubling rehashes of a cold run (tiktoken.rs:749-764). Round-4 analysis: 16 default workers would zero 2 GiB of
tables (16 × 128 MB); Heaps sizing (2^21 slots) halved the memset CPU 289 → 143 ms and the ramp 32 → 19 ms wall
(mt_round4/5 findings). Cost: each worker re-derives the Zipf head independently — Σ per-worker distinct ≈ 16M
vs 5.5M single-thread at 10 GB; encode CPU 14.7 s (MT) vs ~11 s (ST) on M4 Max (mt_round4_findings (c)).

**Gather.** Per-chunk id buffers are gathered into one flat output. M4 Max 10 GB traces (GPT-2, 16 threads):
r3 = 1.462 s (6842 MB/s) with 114 ms serial munmap of chunk buffers on the main thread; r4 fused copy+drop
(7.2/16 busy — munmap write-lock convoy vs first-touch faults) 1.336 s (7487 MB/s); r5 deferred the drop to a
detached background task, 1.226 s = **8159 MB/s**, leaving only the parallel copy (170 ms, 12.2/16 busy, ~555k
16 KB zero-fill faults — no THP on Apple Silicon; >half the gather's CPU is concurrency-induced kernel
contention: 228 ms CPU/GB vs ~60-100 uncontended). Round 5 then implemented commit-the-prefix-during-encode
(reserve `flat` at total_bytes upper bound — token count ≤ input bytes; finishing workers try_lock a commit cursor
and copy completed prefix chunks during encode; expected ~8.6-8.9 GB/s).

**NUMA / huge pages.** Huge pages: the per-worker cache tables are 2 MiB-aligned + `MADV_HUGEPAGE` BEFORE the
zeroing memset — the ordering bug cost 15% cold / 7-8% warm (§2). Output buffers madvise'd before the encode's
stores fault them in (~2.5 MB/chunk, batch.rs:136-137). x86_port_plan.md §3 (Zen 2 EPYC, 255 threads, still
pending as of the doc): aggregate table memory = workers × table — 4.2 GB at 10 GB/255 share, clamping means the
128 MB/worker regime starts at ~200 GB total; rayon pool size (SMT2 128c vs 255) and `numactl --interleave=all`
vs first-touch are listed as open A/Bs; PRETOKEN_CHUNK × 64 B must stay ≲ 256 KB against Zen 2's 512 KB private
L2 (x86_port_plan.md:323-326).

**WHY per-core throughput drops.** README EPYC 9565 (144 cores): GPT-2 **24.53 GB/s** aggregate ≈ **170 MB/s/core**
vs ~900 MB/s single-thread (Zen 5). From gigatoken's own notes: (1) the memory wall — the encode is bound by random
DRAM/LLC probes; 144 workers each stream input + probe a 64-128 MB table + write output: the aggregate LLC/dRAM
bandwidth saturates long before 144 × single-thread rate (reasoning; consistent with zen5_st_profile.md §2's warm
L2-miss/L3-fraction picture and the mt findings' "16 cold caches pay this per worker"). (2) Sharded caches: every
worker re-derives the Zipf head (16M vs 5.5M distinct at 10 GB — mt_round4 (c)), so total work grows with workers.
(3) Kernel contention on the gather: 16 concurrent faulters more than double the copy's CPU (mt_round5 (b)); at 144
cores the vm-map/pmap locks scale worse. (4) At 10 GB/16 the per-worker table is 2^21-2^22 slots (64-128 MB) —
exceeding per-core L3 shares, so warm steady state per core degrades toward the cold rate. (5) EPYC 9565 = 2
sockets; cross-socket DRAM latency/bandwidth for the shared flat output buffer (x86_port_plan lists the NUMA A/B
as open).

## 6. Every number

**Multi-tokenizer file encode, owt_train.txt 11.9 GB (README benchmarks, `encode_files`, whole file un-split;
HF gets 100 MB presplit, tiktoken 1 GB — neither caches, so rates are uniform):**

EPYC 9565 72c ×2 (144 cores): GPT-2 24.53 GB/s (989× HF at 24.8 MB/s, 681× tiktoken at 36.0); Phi-4 24.00;
GPT-OSS 23.96 (482×/560×); OLMo 2/3 23.06; Nemotron 3 22.79; Qwen 3 22.16; Llama 3 22.15; GLM 5 20.97; Llama
3.3 20.82; Llama 4 20.77; GLM 4 20.61; Phi-4-mini 20.05; DeepSeek V3/R1/V4 19.69; Qwen 2/2.5 19.12; Kimi K2
18.85; Qwen 3.5/3.6 15.49; Gemma 4 4.82 (14×); ModernBERT 4.18 (155×); Mistral 7B v0.3 3.57 (10×); TinyLlama/
Phi-3 3.48 (11×); CodeLlama 3.47 (10.0×); Gemma 3 3.43 (9.6×); Gemma 1 2.51 (7.3×). (CLI run: 11.92 GB in
0.486 s = 24532 MB/s, 2.70 Gtok, 20401 documents validated identical to HF.)

M4 Max (16 cores): GPT-2 8.79 GB/s (1268× HF at 6.9 MB/s, 140× tiktoken at 62.8); Nemotron 3 7.82; Phi-4 7.76;
Llama 3 7.60; OLMo 2/3 7.56; Llama 3.3 7.50; Phi-4-mini 6.97; Kimi K2 6.88; Llama 4 6.81; Qwen 2/2.5 6.37;
Qwen 3 6.36; Qwen 3.5/3.6 6.31; GPT-OSS 6.20 (306×/71×); GLM 4 6.17; DeepSeek V3 5.68; GLM 5 5.55; ModernBERT
2.64; Mistral 1.99; Gemma 4 1.82; CodeLlama 1.73; TinyLlama 1.69; Gemma 1 1.42; Gemma 3 1.38.

Ryzen 9800X3D (8c/16t): GPT-2 6.27 GB/s (106× HF at 59.0, 68× tiktoken at 92.1); Phi-4 6.09; OLMo 6.06; GPT-OSS
5.68; Qwen 3 5.34; Qwen 2 5.30; Kimi 5.23; Qwen 3.5 5.22; Nemotron 5.20; GLM 5 5.05; DeepSeek V3 4.21;
ModernBERT 2.84; Gemma 4 1.45; CodeLlama 1.38; Gemma 3 1.12.

**Single-thread Zen 5 (9800X3D, profiling/zen5_st_profile.md, 1 GB OWT unless noted):** baseline reproduced
682-700 MB/s cold, 1008-1057 MB/s warm; 774 MB/s full-file cold (11.9 GB ref ~763). THP fixed: 802 cold mean /
1101 warm mean @ 1 GB (glibc A/B: +15.4% cold / +7.3% warm). IPC: cold 3.88, warm 4.83, warm+THP 5.23 (cold+THP
4.38). dTLB: ~28M page walks per pass unfixed → ~0 with THP (warm dTLB-loads 1.165G → 155K, walks 143M → 11K).
L2-miss traffic 38M lines cold vs 5.5M warm (warm table is ~64 MB). Hot loops: phase-B span emission 32.2% of
warm cycles (37 scalar instr/span, IPC-bound); probe/emit fast loop 36.5% warm (frontend-bound 5.3%,
branch-miss 0.7%); `merge_short` 17.8% of COLD cycles, dominated by rank probes (srcline `bpe/mod.rs:225` =
5.66% of cold; refresh-load consumers ~5-7%); cache insert/grow 0.65% + 0.67% cold.

**M4 Max multi-thread traces (samply, 10 GB cold GPT-2 16 threads):** r3 1.462 s = 6842 MB/s → r4 1.336 s =
7487 → r5 1.226 s = 8159 MB/s (bucket tables quoted in §5). Per-worker encode CPU 846-972 ms (r4) / 908-948 (r5),
E/P-core spread; straggler tail 104 → 23 → 11.7 ms; gather 277 → 214 → 170 ms; ramp 32 → 19 ms; fork memset
~74 GB/s aggregate (r3) / ~65 (r4).

**Corpus/cache stats (Zen 2, 1 GB OWT, pretoken_cache.rs:1-21):** ~1.3M unique pretokens, ~99.4% hit rate
steady-state; 228M output tokens / 208M pretokens; ~90% of pretokens → 1 token, ~98% → ≤2. Heaps: 5.5M distinct
@ 10 GB (mt_round4 (b)).

**Cold vs warm (their definition, zen5_st_profile.md:41-42):** cold = pass 0 (miss-path), `perf --delay 400`,
ENCODE_PASSES=1; warm = passes ≥1 (hit/emit path), delay 1900 with 6 passes so the window is pure warm. README's
file numbers are the whole-file un-split encode (cold-dominated; per-README note the first-run security scan on
macOS should be excluded by running twice).

**Mask-scanner/e2e micro-numbers:** mask scanner "~2× the serial scalar scanners" (mask.rs:5-7);
byte-64 lookahead cut hard-batch deferrals 18.6% → 1.36% of OWT batches (mask.rs:438-478, census :765-770);
vpcompressb harvest +3-8% (mask.rs:934-958); walker phase A + classification ≈ 22% warm (zen5 §8);
`extended_masks` 4.7% warm / 3.8% cold / 4.0% full-size; `batch_masks_avx512` 7.5% warm.

## 7. Measured negatives and pitfalls

- **AVX-512 masked key load in phase B**: `vmovdqu8 {k}{z}` under a tok_len-derived kmask + vpextrq into the CRC
  measured **−36% warm / −30% cold** on Zen 5 (5-round interleaved A/B, 1 GB gpt2, tokens identical; ~90% of loop
  samples on the masked load + dependents). The plain load's address depends only on `prev` so it issues early;
  the masked load's kmask waits on the whole boundary→tok_len chain and the extract adds a vector→GPR crossing
  before the CRC. "Do not re-try" (mask.rs:1396-1408).
- **AVX-512/AVX2 min-rank scan for short merges on x86**: kept as tested reference
  (`bpe_merge_symbols_short_avx512/_avx2`) but measured **~1% slower** on cold encode_st (Zen 5, gpt2, 100 MB and
  1 GB OWT, interleaved min-of-5): the x86 horizontal reduce is a 4-step dependent chain + vector→GPR transfer on
  the serial merge chain, `target_feature` boundary blocks inlining, and the scalar `rank < best` branches predict
  well (tiktoken.rs:672-680, x86_port_plan.md §6). The aarch64 NEON vminq scan STAYS.
- **THP ordering** (the pitfall, not the negative): madvise AFTER first-touch = no-op for the run → **−15% cold /
  −7-8% warm** (zen5_st_profile.md §3; +15.4%/+7.3% from the fix).
- **Prefetch distance D=32 vs 16 in the emit loop: neutral** (an earlier internal study, from gigatoken's notes).
- **`#[cold]` split of the unicode path out of the scanner: −30% on ARM** (an earlier internal study).
- **Recomputing the hash at the consumer instead of storing `meta`: 4% slower end-to-end** (mod.rs:393-395).
- **Parallel-array SpanBatch layout (3 arrays) vs AoS 32 B records**: 4 store µops across 3 streams, probe read
  touched 3 cache lines — replaced (mod.rs:270-275).
- **16-entry mask-value table for key packing**: a dependent L1 load (2.43% of process) on the n→key→store chain —
  replaced by per-half variable shifts / PACK_MASK_TABLE (mod.rs:55-63).
- **Parallel-split granularity / par_iter chunk distribution**: recursive range splitting broke LPT order (104 ms
  straggler, r3) — fixed with in-order atomic handout.
- **Gather copy+drop fused**: munmap write-lock convoy against first-touch faults — 7.2/16 busy; deferred drop
  instead (r4→r5). **mlock/madvise prefault of flat**: serializes at memset bandwidth (~300+ ms for 9.1 GB), worse
  (r5 rejected list). **COW-prototype table clone (mach_vm_remap)**: platform-specific, ≤19 ms upside — rejected.
- **encode CPU 14.7 s MT vs ~11 s ST** for the same 10 GB: 16 cold caches re-derive the Zipf head 16× — inherent
  to sharded caches (mt_round4 (c)), not scheduler-recoverable.
- **Per-core rate collapse on EPYC 9565**: 24.53 GB/s / 144 ≈ 170 MB/s/core vs ~900 ST — see §5.

## 8. Exactness caveats vs hf

- Stated contract (README.md:46): "outputs match exactly with what you would get with HuggingFace Tokenizers in
  this setting" — "this setting" = the benchmark setting (validated per-run: "validation OK: 20401 documents
  match"); at "a non-negligible cost to performance". The HF-compat (`from_hf`) path is way faster than HF but
  "not quite the 1000×".
- **Merge-unreachable vocab entries** are the sharpest divergence trap: seeding whole-word vocab entries as
  themselves was a MEASURED divergence (qwen3_5: ~200 unreachable ≤15 B entries + ~50 >15 B entries like multi-char
  CJK phrases, " Japón") — the seed must store the BPE merge decomposition (tiktoken.rs:443-460, test
  `verify_vocab_seeded_cache_matches_merge_decomposition`). The same rule forbids a whole-pretoken `vocab_inv`
  shortcut on the long miss path without `ignore_merges` — "Do not reintroduce it" (tiktoken.rs:1647-1659).
- **`ignore_merges` flips the rule** to `[own id]` — the seed and miss paths both branch on the flag.
- **Non-BPE families refused by name** (WordPiece, Unigram, ...) rather than mis-parsed (hf.rs:203-262); unknown
  pretokenizer regexes / normalizers error rather than silently diverge (hf.rs:430-490, 596-638). tiktoken files:
  dense-rank check; scheme and specials must be caller-supplied — a wrong scheme silently changes every encode
  (load_tokenizer/tiktoken.rs:23-30).
- **NFC**: only NFC (and NFC+Lowercase chains) accepted; invalid UTF-8 passes through unchanged — "HF only ever
  sees `str`, so there is no parity behavior to match" (tiktoken.rs:167-171).
- **Known-fragility areas** (reasoning from the code's own warnings): added-token matching is greedy
  leftmost-longest restart per chunk boundary, guarded by the space-blocker/rstrip rules (mod.rs:570-660);
  contraction edge cases per scheme are handled by per-apostrophe fixups with `'`+non-ASCII deferred
  (cl100k_family.rs:552-582); parallel `safe_split_ranges` boundaries are proven under every
  SUPPORTED scheme only.

## 9. Takeaways for toks

Ranked by expected gain per core (reasoning, grounded in the numbers above):

1. **Keep the pretoken cache; make the probe/emit loop asm.** Gigatoken's warm path is a two-load-per-pretoken
   loop (pair line + batch entry) at IPC 4.8-5.2 — near issue limits on Zen 5 already. The remaining per-core
   headroom on Apple M-series/Neoverse V2 is the phase-A walker (~22% warm) and phase-B span emission (32% warm,
   37 scalar instr/span): hand-written asm can fuse key pack + CRC + prefetch + record store tighter than LLVM
   (gigatoken needed hand-pinned `csel`/`cmov` asm to stop LLVM's address-select canonicalization — the same class
   of fix applies to our whole loop). Expected: gigatoken ST is 1074 MB/s warm / 780 cold on Zen 5 at 1 GB; a
   tighter loop plausibly buys 10-25% warm per core.
2. **Per-core ≠ aggregate: toks must beat 170 MB/s/core on big x86.** At 144 EPYC cores gigatoken collapses to
   ~170 MB/s/core from ~900 ST — memory-bound (probe traffic × workers + output writes + sharded caches
   re-deriving the Zipf head). toks should (a) share one read-only seeded table + per-worker DELTA caches or a
   shared lock-free cache (saves the 16×-distinct re-derivation: 14.7 s vs 11 s CPU at 16 threads), (b) keep
   per-worker tables under the L3 slice budget, (c) commit-prefix during encode (gigatoken's own r5 estimate
   8.6-8.9 GB/s at 16 threads — copy the design), (d) NUMA-interleave the output buffer on 2-socket parts.
3. **Copy the huge-page discipline exactly**: 2 MiB-aligned table, madvise BEFORE the zeroing touch, and note
   macOS has no THP at all (the M4 Max gather is pure 16 KB zero-fill faults — ~555k for 9.1 GB). On Neoverse V2
   check 64 KB-page dTLB behavior; on Apple nothing to do.
4. **The mask scanner approach transfers**: per-64 B class masks + u64 algebra + scalar ground-truth re-walk for
   bad zones + u16 boundary harvest + branch-free phase B. Gigatoken measured 2× the scalar scanner and the
   AVX-512 VBMI2 `vpcompressb` harvest +3-8%. BUT do NOT copy: masked key loads (−30-36%), AVX-512 min-rank scan
   on x86 (−1%), D=32 prefetch (neutral). On ARM keep the NEON vminq short-merge scan (it's their production
   path) — port it to our asm rather than redesigning.
5. **Byte-pair grid**: the 2^22-cell dense table for pairs with both sides < 2^11 is what makes cold merge cheap
   (one load, L3-resident, Arc-shared). toks should replicate with an asm-friendly layout (u32 cells, no packed
   key on the dense path) and keep the flat-level packed u64 (key 43 bits + merged 21) for the tail.
6. **Vocab-seed with BPE results, never own-ids** (unless ignore_merges) — the exactness trap that cost them a
   measured divergence. Port their test.
7. **Chunk sizing**: 1 MiB floor, ~16 chunks/thread, LPT-descending sizes with strict in-order handout;
   fork_sized with the Heaps estimate 3.45·n^0.62 (saves a 2 GiB→1 GiB memset at 16 workers and the rehash
   churn). Bound the aggregate: their 2^22-slot clamp means 128 MB/worker past ~0.78 GB share — flag RSS on
   long-lived pools.
8. **Exactness first**: refuse-by-name unsupported components (no silent normalizer skipping), explicit
   ignore_merges/byte_fallback handling, added-token atomic matching before pretokenization with lstrip/rstrip,
   and per-run validation vs HF — that is what makes their "989×" number honest.
