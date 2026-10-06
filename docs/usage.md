# Using toks

How toks is meant to be called for speed. SPEC §22: toks's usage and abi may change wherever that makes it
faster or simpler, and every such change is written down here.

## Cutting text: toks_split_points

To cut a text into parts that encode separately (a segment cache, a chunk store, an incremental document, a
parallel split), use **`toks_split_points`**, never `toks_pieces`. Its cuts are certified (SPEC §5,
docs/split.md): encoding the parts, every part after the first with `TOKS_CONTINUATION` and none post-processed,
and concatenating the ids gives exactly the whole text's ids; the offsets always index the caller's bytes; and a
cut is decided from a window of bytes around it only (the longest added token + 16 on each side), so it stays a
cut when the text grows past that window. Every algorithm has rules: byte-level bpe (cl100k, o200k, deepseek
templates), sentencepiece-style bpe, wordpiece and unigram; a tokenizer without rules (truncation, padding, kimi's
own chunking) returns no cut and encodes whole.

```c
uint64_t offs[64];
int64_t c = toks_split_points(ctx, text, len, flags, n_want, offs, 64, NULL);   /* cuts near len * i / n_want */
/* parts [0, offs[0]), [offs[0], offs[1]), ..., [offs[c - 1], len): the first encoded with flags | NO_POSTPROCESS,
   the rest with TOKS_CONTINUATION too; the post-processor's ids go around the whole once (toks_split_points'
   doc in toks.h). n_want = len asks for every certified cut. */
```

Why not `toks_pieces`: a piece boundary is not a cut. The pre-tokenizers look ahead at the end of their input, so
the left part, cut at a piece boundary, can tokenize differently from the same bytes inside the whole text
(hf 0.23.2: gpt-2 "\n\n" + "A" gives [628, 32], the whole [198, 198, 32]; llama 3 "\t\t1" gives [298, 16]
against [197, 197, 16]). And where a normalizer rewrites the text (NFC: qwen 3.8), piece ends are offsets into
the normalized form (SPEC §3.5), which no call maps back to the input:
a cache cut there cannot even find its bytes. Mapping them back would cost a per-call alignment table and would
still not make a piece boundary a cut, so toks_pieces stays what it is, a view of the model's input for
inspection, and cutting is one function (split.h's predicate, shared by toks_par and the incremental layer).

## Sizing out: toks_encode_bound

`toks_encode_bound(ctx, len)` is the most ids `toks_encode` can return for any text of `len` bytes under any flags,
so an `out` of that many ids is allocated once and never retried (the count `toks_encode` returns is never above it).
Where nothing can lengthen the text (r = 1) it is `len` + the template's ids, + 1 where the model prepends a ▁, and
exact: a text of `len` >= 1 bytes that are one id each reaches it (the empty text gets no ▁, so `len` 0 can be one
over). Elsewhere it is ceil(r x `len`) + g, r by normalizer: NFC byte-level 3, NFKC 11, a unigram nmt_nfkc charsmap 6
(11 under byte fallback), NFKD then the charsmap 66 (albert, xlnet: conservative; a scan of the two steps composed
tightens that class in 0.3.1); unigram byte fallback with no ▁ piece triples its r (3 with no normalizer) and a
prepended ▁ counts 3 in g, not 1 (a ▁ is then its 3 byte ids). A context that pads gets at least its padded length.
Any `len` has a bound, but `toks_encode` returns `TOKS_E_LIMIT` above `TOKS_MAX_TEXT` (2^29 bytes).

## Stream decode: the hold

hf decides a SentencePiece byte-fallback run (Gemma, Mistral v0.3, Llama 2: `<0xHH>` ids in a row) as a whole: its
characters when its bytes are valid UTF-8, else one U+FFFD per byte. A stream therefore emits nothing of a run that is
still valid until a string or the flush ends it, and holds its bytes meanwhile: in the `toks_stream` itself up to 44
bytes, where twelve 4-byte characters in a row (U+13000 x 12, 48 bytes) return `TOKS_E_LIMIT`. `toks_stream_hold`
gives the stream memory of the caller's instead, and then the only limit is that buffer:

```c
toks_stream st;
uint8_t hold[1024];                      /* any run of 256 characters: at most 4 bytes each */
toks_stream_init(ctx, &st, 0);
toks_stream_hold(ctx, &st, hold, sizeof hold);
```

A push that would hold more than the buffer returns `TOKS_E_LIMIT` with the stream and its buffer unchanged (`out`
may hold a prefix of what the push would have written; nothing of it counts). A buffer of at least the current one's
size plus that push's id count takes it, so the recovery is to grow and push the same ids again (the old buffer must
be alive during the call, which copies from it; free it afterwards, never `realloc` it in place):

```c
/* held_cap: the hold's size now (44, the stream's own, before any toks_stream_hold);
   old_buf: the caller's buffer behind it, or NULL */
int64_t r = toks_stream_push(ctx, &st, ids, n, out, cap);
if (r == TOKS_E_LIMIT) {
    uint64_t bigger = 2 * held_cap + n;
    uint8_t *nb = malloc(bigger);
    toks_stream_hold(ctx, &st, nb, bigger);   /* moves the held bytes; returns how many */
    free(old_buf);                            /* after the call, which copied from it */
    old_buf = nb, held_cap = bigger;
    r = toks_stream_push(ctx, &st, ids, n, out, cap);
}
```

The buffer belongs to the stream until `toks_stream_init` runs on it again, the stream is dropped, or
`toks_stream_hold(ctx, &st, NULL, 0)` moves the run back into the stream's own 44 bytes (the way to take the buffer
back). A push's output is at most `toks_stream_bound(ctx, n) + 3 * cap` bytes, `cap` being the buffer's size. Byte-level and WordPiece
streams hold at most 3 bytes and ignore the call. What the stream emits, concatenated, is always hf's `decode` of the
ids; for runs that are valid as a whole it is also what hf's `DecodeStream` emits, which writes a run's characters as
soon as they decode, so a stray byte after them (which makes the whole run U+FFFD in hf's own `decode`) can contradict
it. The stream emits only bytes no later id can change, which for a run means at its end.

## Vocabulary lookups

`toks_token_to_id(ctx, s, len)` takes the bytes an id decodes to, which is what `toks_token` returns for it. It does
not take hf's vocabulary spelling. A byte-level model's `" hello"` is gpt2's id 23748, which hf spells `"Ġhello"`;
`"Ġhello"` itself is `TOKS_E_ID` here.

The footgun: an hf spelling fed in as bytes can silently name ANOTHER valid id. gpt2's `"¢"` (bytes C2 A2) is hf's
spelling of the byte token A2 (id 95). Looked up as bytes, it is the token hf spells `"Â¢"` (id 44359). Each byte-level
file has 56 to 148 such strings. So convert an hf spelling to its bytes first; the Python binding's `token_to_id` does
that for a `str`.

What the lookup finds:
- A SentencePiece piece as written (`"▁hello"`, `"<0x41>"`).
- An added token by its content (`"<s>"`, `"<|eot_id|>"`), first, as hf does.
- An added token by its normalized form, where hf decodes one (Llama 2's `"▁<s>"`).
- `toks_token_to_id(toks_token(id))` gives the id back unless ids share bytes. Then the bytes name the id hf's
  `token_to_id` gives that text:
  1. the added token (Pythia's added `"  "` against the vocabulary's `"ĠĠ"`);
  2. else the id the file writes as those very bytes (a raw `"\u200d"` against its alphabet form `"âĢį"`);
  3. else the later id.

  The other id is reachable only by its number. The sweep of the cached files found 221 such ids in 14 files.
- Never the empty string: hf finds wp-specter2's empty token, id 30106; here it is `TOKS_E_ID`.

`toks_id_flags(ctx, id)` answers, per id:
- `TOKS_ID_ADDED`: an added token holds the id (hf's `added_tokens_decoder` has it).
- `TOKS_ID_SPECIAL`: some listing of that token's content is special (hf's `special_tokens_set`). This is the token's
  own flag, not the rule `TOKS_SKIP_SPECIAL` uses. Llama 2's `<s>` (id 1) is special, but decode with
  `TOKS_SKIP_SPECIAL` keeps it, as hf does, because its decoded string `"▁<s>"` is no special token's content.
- `TOKS_ID_BYTE`: the id stands for exactly one raw byte. That is a `<0xHH>` that the ByteFallback decoder reads as one
  byte, or a byte-level one-byte token.
- 0: a plain vocabulary id, or an id with no string.

Both are built at load, at 5.5 to 9 bytes per id and about 1% of the load time.

## Threads: toks_par

toks_par encodes one big input, or a batch of documents, on a few cores and returns exactly what serial
`toks_encode` calls return (SPEC §0.3, §2.4; T2). It is a small persistent worker pool with a load balancer,
and it goes wide only where that beats one core: toks never spreads out over a machine to do what one core
does faster.

### Calls

```c
toks_par *p;
toks_par_create(&p, ctx, 0, 0);          /* a couple of participants, default scratches */

int64_t n = toks_par_encode(p, text, len, 0, out, cap);      /* = toks_encode(ctx, text, len, 0, out, cap, scr) */

toks_par_item it[N];                     /* { text, len, out, cap, n }: each document has its own out */
toks_par_encode_batch(p, it, N, 0);      /* it[i].n = toks_encode's result for document i */

toks_par_destroy(p);
```

- `toks_par_create(&p, ctx, n_threads, scratch_flags)`: `n_threads` caps the participants, the calling thread
  included (it works too). 0 is the default: a couple, min(4, the fast cores the process may run on). The pool
  never has more participants than the cpus in the process's affinity mask. Workers prefer the fast cores: on
  linux a worker is confined to the highest `cpu_capacity` class (else the highest max frequency) when the
  process may run on several (a GB10's X925s, not its A725s); on apple it takes the caller's QoS class (p-cores
  for p-work). `scratch_flags` are every participant's `toks_scratch_init` flags (a 32 MiB cache costs 32 MiB per
  participant). create measures the pool's wake and join delays on this host (about 5 ms, once).
- `toks_par_encode`: one input under `toks_encode`'s contract (flags, cap, the count or an error).
- `toks_par_encode_batch`: many documents in one call. Each item's ids go straight into its own `out`; the call
  returns 0 (or `TOKS_E_ARG`) and each item's `n` is its `toks_encode` result.
- `toks_par_get_info(p, &info)` (`TOKS_PAR_HAS_INFO`): what the pool measured and decides by: `threads`, `fast`,
  `last` (the participants of the last call), `ns_per_mib`, `wake_ns`, `join_ns` and `min_bytes` (the smallest
  call that would take a second participant now; 0: none would).
- One call at a time per pool: concurrent calls on one pool wait for each other. A server whose own threads each
  tokenize their own requests keeps one scratch per thread and calls `toks_encode` (no pool); a pool is for a
  caller that wants one big input or a batch done sooner.

### What a call does

1. **Small calls stay on the caller.** A call of fewer than 16 KiB runs on the calling thread: no clock read, no
   thread touched.
2. **Participants from a measured cost model.** Otherwise the pool estimates the call's serial time (its bytes x
   the measured encode cost a byte: the lower of the pool's split units' and its serial calls', so the cache
   effects of splitting never argue for splitting more) and takes the most participants `k` that keep the
   modeled efficiency, serial time / (k x the call's time), at or above 75%. The model's terms are measured per
   pool on its own host and tokenizer: a worker's join delay by its state (spinning after its last call: about
   a microsecond; asleep: a kernel wake, 2-250 us measured, deep idle states being slow to leave; calls back to
   back within a spin count as spinning), the per-worker cost of short calls beyond the join (cold caches and
   clocks), a proportional cost of going wide (from calls of >= 2 ms), the idle tail and a unit's fixed cost.
   Every call of 16 KiB or more refines it. So a call one core finishes faster runs on one core, and a call
   that can use two well but not eight gets two.
3. **The load balancer.** The work is cut into units of about bytes / (8 k): whole documents grouped, a big
   document split at `toks_split_points`' certified cuts (SPEC §5), and units of a quarter of that size over
   the last quarter of the bytes, so whoever runs out of work last waits for a small unit (longest first). The
   participants claim units in order from one atomic cursor: dynamic balancing, no static partition (equal bytes
   are not equal time across scripts, caches and big / little cores).
4. **Ids in place.** A whole document is one `toks_encode` straight into its own `out`: zero copies. A split
   document's first part goes straight into `out`; each later part into the pool's staging, and the participant
   that encoded it copies it once to its place as soon as every earlier part is done (its ids still in that core's
   cache). Output equals the serial call's, error codes included (a part that fails sends its document back to a
   serial encode).
5. **Idle.** A worker spins about as long as a wake costs (10-200 us), then sleeps in the kernel on its own word
   (futex, ulock, WaitOnAddress). A call wakes exactly the workers it uses; the rest stay asleep. No thread is
   created per call.

### Where it pays

Receipts: docs/bench/par.md (generated from the raw logs in docs/bench/raw/par-*.log by tests/par/par_table.py;
every row's ids equal the serial call's; these two tables are `par_table.py --summary`). llama 3 on enwik8, flags
0; gb10b = a GB10's X925 cores, tr9970x = one Zen 5 CCD (8 cores); a cell of k runs on k pinned cores; load recorded
per cell (gb10b 1.1-1.7, tr9970x 1.5-5.6: shared hosts). Commit b6eca5f.

**Scaling** (pass state = new text on a warm pool; k = the pool's cap, and every call here used all k; latency =
the call's ms; gigatoken = its parallel api, `encode_docs_ragged` with `RAYON_NUM_THREADS` = k, same cores and
states):

| input | k | gb10b MB/s (x serial, eff) | ms | gigatoken | toks / giga | tr9970x MB/s (x serial, eff) | ms | gigatoken | toks / giga |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one input, 1 MiB | serial | 210 (serial) | 5.00 |  |  | 305 (serial) | 3.44 |  |  |
| one input, 1 MiB | 1 | 206 (0.98x, 98%) | 5.08 | 177 | 1.17 | 287 (0.94x, 94%) | 3.65 | 229 | 1.26 |
| one input, 1 MiB | 2 | 346 (1.65x, 82%) | 3.04 | 176 | 1.96 | 483 (1.58x, 79%) | 2.17 | 216 | 2.24 |
| one input, 1 MiB | 4 | 631 (3.01x, 75%) | 1.66 | 176 | 3.59 | 741 (2.43x, 61%) | 1.42 | 234 | 3.17 |
| one input, 1 MiB | 8 | 1243 (5.92x, 74%) | 0.84 | 172 | 7.24 | 1167 (3.83x, 48%) | 0.90 | 230 | 5.08 |
| one input, 4 MiB | serial | 216 (serial) | 19.38 |  |  | 318 (serial) | 13.21 |  |  |
| one input, 4 MiB | 1 | 215 (0.99x, 99%) | 19.52 | 165 | 1.30 | 318 (1.00x, 100%) | 13.19 | 210 | 1.52 |
| one input, 4 MiB | 2 | 363 (1.68x, 84%) | 11.55 | 329 | 1.10 | 555 (1.75x, 87%) | 7.56 | 452 | 1.23 |
| one input, 4 MiB | 4 | 662 (3.06x, 77%) | 6.33 | 306 | 2.16 | 920 (2.90x, 72%) | 4.56 | 421 | 2.19 |
| one input, 4 MiB | 8 | 1431 (6.61x, 83%) | 2.93 | 383 | 3.74 | 1504 (4.74x, 59%) | 2.79 | 429 | 3.51 |
| one input, 16 MiB | serial | 224 (serial) | 75.04 |  |  | 314 (serial) | 53.44 |  |  |
| one input, 16 MiB | 1 | 224 (1.00x, 100%) | 74.75 | 233 | 0.96 | 314 (1.00x, 100%) | 53.51 | 295 | 1.06 |
| one input, 16 MiB | 2 | 384 (1.72x, 86%) | 43.65 | 445 | 0.86 | 568 (1.81x, 90%) | 29.54 | 532 | 1.07 |
| one input, 16 MiB | 4 | 729 (3.26x, 81%) | 23.02 | 592 | 1.23 | 1075 (3.42x, 86%) | 15.60 | 745 | 1.44 |
| one input, 16 MiB | 8 | 1608 (7.19x, 90%) | 10.43 | 1331 | 1.21 | 1804 (5.74x, 72%) | 9.30 | 1403 | 1.29 |
| one input, 64 MiB | serial | 227 (serial) | 295.13 |  |  | 318 (serial) | 211.34 |  |  |
| one input, 64 MiB | 1 | 230 (1.01x, 101%) | 291.27 | 447 | 0.52 | 303 (0.96x, 96%) | 221.21 | 511 | 0.59 |
| one input, 64 MiB | 2 | 393 (1.73x, 86%) | 170.62 | 575 | 0.68 | 588 (1.85x, 93%) | 114.12 | 654 | 0.90 |
| one input, 64 MiB | 4 | 767 (3.37x, 84%) | 87.51 | 994 | 0.77 | 1075 (3.39x, 85%) | 62.41 | 1190 | 0.90 |
| one input, 64 MiB | 8 | 1751 (7.70x, 96%) | 38.32 | 1920 | 0.91 | 1914 (6.03x, 75%) | 35.06 | 1936 | 0.99 |
| 16 MiB of 4 KiB docs | serial | 217 (serial) | 77.31 |  |  | 316 (serial) | 53.14 |  |  |
| 16 MiB of 4 KiB docs | 1 | 222 (1.02x, 102%) | 75.66 | 231 | 0.96 | 313 (0.99x, 99%) | 53.62 | 291 | 1.07 |
| 16 MiB of 4 KiB docs | 2 | 424 (1.95x, 98%) | 39.62 | 444 | 0.95 | 622 (1.97x, 99%) | 26.95 | 488 | 1.27 |
| 16 MiB of 4 KiB docs | 4 | 795 (3.66x, 92%) | 21.11 | 592 | 1.34 | 1201 (3.80x, 95%) | 13.97 | 739 | 1.63 |
| 16 MiB of 4 KiB docs | 8 | 1790 (8.25x, 103%) | 9.38 | 1250 | 1.43 | 2084 (6.60x, 83%) | 8.05 | 1489 | 1.40 |

A pool splits one input nearly as well as it spreads a batch: 7.2-7.7x on 8 X925s and 5.7-6.1x on 8 Zen 5 cores
for 16-64 MiB, 8.3x / 6.6x for a batch of 4 KiB documents. gigatoken's parallel api cuts work into chunks of at
least 1 MiB (its src/batch.rs MIN_CHUNK_BYTES): a 1 MiB input stays at one core's speed and 4 MiB reaches about
2x; at 64 MiB its single core is the faster one in this state (447 vs 230 MB/s on gb10b, 511 vs 303 on tr9970x: a
per-core lead), so it ends 10% ahead on 8 X925s and 1% ahead on 8 Zen 5 cores.

**Where going wide starts** (`tests/par/bench_par.c` sweep: new text every call; the participants the pool chose
for a call of that size, and serial / the pool's call, medians of 5; at 1 the call ran on the caller exactly as
`toks_encode`, and its x is noise: each variant meets its own windows of the file, and after idle gb10b's cores
wake slowly from deep idle):

| size | gb10b, back to back | gb10b, 5 ms idle | tr9970x, back to back | tr9970x, 5 ms idle |
|---|---:|---:|---:|---:|
| 4 KiB | 1: 1.39x | 1: 0.41x | 1: 1.22x | 1: 0.89x |
| 8 KiB | 1: 1.14x | 1: 1.11x | 1: 1.20x | 1: 1.36x |
| 16 KiB | 1: 1.04x | 1: 0.36x | 1: 1.05x | 1: 0.86x |
| 32 KiB | 1: 0.93x | 1: 0.62x | 2: 1.77x | 1: 0.98x |
| 64 KiB | 1: 1.48x | 1: 0.57x | 3: 1.92x | 3: 1.85x |
| 128 KiB | 3: 2.32x | 2: 1.19x | 4: 2.07x | 4: 2.67x |
| 256 KiB | 3: 2.70x | 3: 1.46x | 8: 4.13x | 8: 3.84x |
| 512 KiB | 5: 3.10x | 6: 2.85x | 8: 5.12x | 8: 4.31x |
| 1 MiB | 8: 5.10x | 8: 4.90x | 8: 5.11x | 8: 4.49x |
| 2 MiB | 8: 5.38x | 8: 4.97x | 8: 5.89x | 8: 4.52x |
| 4 MiB | 8: 6.41x | 8: 6.30x | 8: 6.38x | 8: 5.67x |

The pool's own threshold after these runs (`toks_par_get_info`'s min_bytes, back to back / idle): gb10b 82 / 150
KB, tr9970x 29 / 41 KB. On the same text repeated (one core's caches hold all of it, each participant only its units:
the worst case for splitting) it waits longer (min_bytes gb10b 599 / 221 KB, tr9970x 420 / 244 KB): back to back it
keeps calls up to 512 KiB on one or two participants, where on gb10b 8 forced participants give 1.5-2.5x
(medians) and 2 forced 1.05-1.42x, but on tr9970x 2 forced give 1.45-2.07x at 128-512 KiB: left on the table (an open
issue: the pool's proportional cost is learned on wide calls). Near its threshold a pool explores before it knows:
on gb10b's repeated text it went two wide at 64 KiB (1.27x best, 0.75x median) and 128 KiB (1.26x / 1.00x)
before the measured cost kept 256-512 KiB on one. Before this pool (master b02ac89: every call it split went
n-wide), a 32 KiB call on 8 cores took 3.4x (gb10b) / 3.3x (tr9970x) as long as one core (best of 5; medians 5.7x /
31x); now such calls stay on the caller (gb10b) or take two (tr9970x: 1.77x on new text).

### Knobs

- `n_threads`: the most cores a pool may use, the caller included. 0 = min(4, the fast cores). Unpinned on
  gb10b (20 cpus), a pool of 4 reports `fast` = 10 and its workers ran on X925s only (cpus 8, 9, 15; the caller
  on 5).
- create takes about 4-5 ms (its calibration; gb10b, pools of 2-8); destroy well under a millisecond.
- `scratch_flags`: each participant's scratch (`TOKS_SCRATCH_CACHE_MIB`, `TOKS_SCRATCH_MEMO_MIB`).
- `TOKS_PAR_EAGER=1` in the environment at `toks_par_create`: skip the model, use every participant the units
  allow. A measurement knob for tests and scaling benches, not for production.

### Memory

Each participant owns a scratch it grows to the longest unit it meets and keeps (units hold at most 1 MiB where cuts
allow: about 42 MiB of address space per participant for llama 3 with the default 4 MiB memo, touched only as far as
a call needs). A split input also uses the pool's staging for its later parts' ids (4 bytes of address space per
input byte, kept for the next call). Measured on gb10b (llama 3, enwik8; resident memory after the call, the text
and the caller's out excluded): a 64 MiB input takes +68 MiB serially (`toks_encode`, a 2,306 MiB scratch of address
space), +270 MiB on a pool of 4 and +286 MiB on a pool of 8; 16 MiB: +19 MiB serially, +113 MiB on 8. A pool at
create holds no scratch yet (+0 MiB).

### Limits

- A big single document is split only where `toks_split_points` certifies cuts (docs/split.md): byte-level bpe
  with the cl100k-family, o200k-family (gpt-oss, llama 4, nemotron, minimax) and deepseek pre-tokenizers,
  sentencepiece-style bpe (gemma, mistral), wordpiece (after whitespace) and the unigram chains that cut at spaces
  (xlm-r / bge-m3 / e5, t5). Truncating or padding tokenizers, kimi (its own 400,000-code-point cuts) and the
  one-piece unigram chains (ruri, llm-jp) keep a single document on one core; batches of their documents still
  spread.
- Decisions are made per call from measurements, so the first calls of a pool decide from create's measures
  and the seeds (400 MB/s); a call's own participants are never more than its units (a unit holds >= 8 KiB).

## Scratch

A scratch is one caller buffer (`toks_scratch_bytes(ctx, max_len, flags)` bytes) that holds every per-call region:
the piece ends, the piece caches, the segment memo, K6's work and the bounce buffer (docs/kernels.md §7 lists them).
Keep one per thread and reuse it: the caches and the memo only help a scratch that lives across calls.

Size (the published formula, docs/kernels.md §7): `63 + 128 + 4 TOKS_CHUNK_PIECES + caches + memo +
align64(256 + 32 tmax) + align64(4 (tmax + 4)) [+ align64(3 max_len)]` bytes, tmax = max_len (3 max_len for NFC
tokenizers), at most 189 B of rounding. With the default flags: 6 MiB + 1,487 B + 36 B per byte of max_len (NFC:
111 B per byte), of which 2 MiB is the piece cache and 4 MiB the segment memo: 6.14 MiB a thread at max_len 4 KiB
(llama 3, gpt2, o200k, gemma 4; qwen 3.8 6.43 MiB), 42.0 MiB at 1 MiB. Wordpiece and unigram have no memo at any
flags (2.14-3.6 MiB at 4 KiB).

The default (flags 0, one default on every host; decided 2026-10-05): the 2 MiB piece cache and the 4 MiB segment memo
(SPEC §6). Replay is the serving workload (every chat request re-sends its conversation, a system prompt comes back
with every request), and the memo answers a segment it has seen (>= 256 B between added tokens) with a keyed check
and an id copy instead of an encode (docs/kernels.md §7). Text that never comes back pays for it: the memo records the
first ~2-4 MB a scratch sees (about 1 B stored per byte encoded: a record keeps the ids and a 16-byte check, not the
bytes; until a ring of records passes without a hit), and those writes slow the next pass as they are written back
(2-4% when a record also held the bytes, 2 B a byte). Measured (receipts below), MB/s ranges over llama 3 / o200k / qwen
3.8 x en / code, 4 KiB calls:

| host class (key) | cold | pass | lang | warm | warmo |
|---|---|---|---|---|---|
| NVIDIA GB10, X925 (gb10c) | 263..459 | 285..478 | 247..323 | 9,441..14,993 | 2,277..3,819 |
| AMD Threadripper 9970X, Zen 5 (tr9970x) | 291..431 | 390..482 | 396..487 | 12,942..19,209 | 3,703..7,330 |
| Apple M2 Ultra (m2ultra2, unpinned: shape) | 265..391 | 355..446 | 352..421 | 8,781..11,176 | 3,028..5,552 |

States: cold = a fresh scratch per call (cpu caches hot); pass = the first sight after an untimed pass over the other
corpora and a re-init; lang = the first sight after the other corpora, no re-init; warm = the same text again right
after; warmo = the same text again after the other corpora (the serving replay). Without the memo the same cells ran
warm 612..817 and warmo 286..511 MB/s; gigatoken's warmo (its 512 MiB cache and unit memo) was 577-868 MB/s on
tr9970x. The rows above were measured while a record held its segment's bytes. With the check record, gb10c (cpu 8,
tools/bench/e2e_commits.sh against the master before it, 3 abba rounds, ids equal; the same three tokenizers x en /
code at 4 KiB) reads cold 259..442, pass 235..307, lang 248..323, warm 11,383..11,598, warmo 3,903..7,574 MB/s, from
cold 260..446, pass 228..305, lang 245..318, warm 9,333..13,815, warmo 2,188..3,715 in the same runs; tr9970x and
m2ultra2 are to be measured again.

- `TOKS_SCRATCH_MEMO_MIB(0)`: a budget of zero, no memo, for batch jobs over text that never comes back. Against the
  default it runs first sights faster, most on the X925, whose 2 MiB L2 holds the piece cache the records evict:
  gb10c pass x1.05..1.17 (en x1.13..1.17), lang x1.03..1.09, cold x1.01..1.05; tr9970x and m2ultra2 pass
  x1.01..1.04, lang x1.00..1.02, cold x1.00..1.05. Every replay is then a full encode (warm x0.02..0.08 of the
  default's on en / code, x0.42..0.47 on ml / zh).
- `TOKS_SCRATCH_MEMO_MIB(m)`: a memo of m MiB (m < 4096): a bigger one holds more text to replay (records take ~1-2
  B per byte of text; at 4 MiB the 3 MB cjk corpus warms to 1,888 / 1,921 / 10,347 MB/s with llama 3 / o200k /
  qwen 3.8 at 4 KiB on gb10c, 380 / 419 / 443 when a record also held the bytes).
- `TOKS_SCRATCH_CACHE_MIB(n)`, the kept-scratch budget (n a power of two 4..128; the piece caches only, the memo
  stays): the long-piece cache holds what the memo does not replay: pieces over 15 bytes or 4 ids that repeat across
  different texts. On top of the default memo, `TOKS_SCRATCH_CACHE_MIB(8)` (12.14 MiB a thread at 4 KiB):
  - tr9970x and m2ultra2: ml warm x2.47 / x2.61, warmo x1.19 / x1.22, pass and lang x1.05..1.08; gpt2 code whole
    (long whitespace runs) cold, pass and lang x1.10..1.15; en first sights x0.93..0.98, en / code warm x0.93..1.03;
    zh x0.84..0.91 in every state but cold. For a worker whose text is code or multilingual, not en- or zh-heavy.
  - gb10c: ml warm x2.39, warmo x1.11, gpt2 code whole cold / pass x1.12..1.13, but every other first sight pays: en
    pass and lang x0.88..0.90, code lang x0.88..0.97, ml x0.95..0.96, zh x0.78..0.86. Keep the default unless ml
    replay dominates.
  - Larger budgets were measured only without the memo (docs/kernels.md §7: there warm kept gaining up to 16-32 MiB
    and every first sight of en and zh paid more).
- Memory, SPEC §7.2's line at max_len 4 KiB (llama 3): one thread 6.14 MiB (the default), 2.14 MiB
  (`TOKS_SCRATCH_MEMO_MIB(0)`), 12.14 MiB (`TOKS_SCRATCH_CACHE_MIB(8)`); 64 threads 393 / 137 / 777 MiB, 256 MiB of the
  default's being memo. An NFC tokenizer adds 75 B per byte of max_len (qwen 3.8: +0.29 MiB at 4 KiB).
- Round size: TOKS_CHUNK_PIECES (256) pieces a round (layout.h), not a flag.

Receipts: e2e.c on master dc9b0c4 (the memo rule and index of commit ca33993), the scratch flags as variants of one library
(`E2E_MEMO_MIB=4` stood for today's default, `E2E_MEMO_MIB` unset for `TOKS_SCRATCH_MEMO_MIB(0)`), 3 palindromic
rounds of 5 reps, each run's best, the median over rounds; ratios paired by round; ids equal in every cell and
variant; gb10c cpu 8 (load 1.0), tr9970x cpu 28 under its timing lock (load 2.5-4.9), m2ultra2 unpinned (load 2.0);
also llama 3 ml / zh at 4 KiB and gpt2 code whole. The per-host default question has one answer:
on all three, no piece-cache size beat the 2 MiB default on a first sight (median over the cells, within 0.2%); what
differs by host is the memo's price on first sights (the X925's is largest) and what `TOKS_SCRATCH_CACHE_MIB(8)` buys.
