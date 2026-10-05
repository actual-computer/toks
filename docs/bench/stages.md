# Cold stage profile (2026-10-04)

Where a COLD byte-level bpe encode spends its time, per stage, on the two timing hosts. Cold = a fresh scratch for
every call (toks_scratch_init before each call, outside the timer; default scratch flags), the asm tiers that
toks_load binds (neon on gb10e, avx2 on tr9970x). master 086ff0e + the bench tools below. Quick look, best of 5,
not a SPEC §12 cell (no abba pairs, no intervals). Receipts: raw/stages-cold-gb10e-086ff0e.log,
raw/stages-cold-tr9970x-086ff0e.log (hosts, loads, sibling busy, binary and corpus sha-256, every STAGES / CLASSIFY /
COUNT line).

    E2E_STATE=cold STAGES=1 COUNT=1 REF=0 GIGA=0 TOKS_LIST="llama3 o200k" CORPORA="en code zh" CHUNKS="4096 0" \
        PINCPU=7 sh tools/bench/e2e.sh "taskset -c 7"            # gb10e: X925 cpu 7; tr9970x: cpu 20 (sibling 52)
    uv run tools/bench/stages_table.py gb10e=<log> tr9970x=<log>

How the stages are measured (tools/bench/stages.c): the library is compiled a second time with the kernel call
sites renamed to timing wrappers (K1, K3 per template, K5, and K6 inside k5_long.c, so every tier's K6 is timed);
one timer pair per kernel call, its measured cost subtracted. K5's own time (K5 minus its K6 calls) is split over
its piece classes by a least-squares fit over the rounds (x = 1-byte pieces, static hits of 2..15 B, cache hits,
short misses, long pieces, 1 per round). Caveats: the timers serialize (isb / rdtscp): K6 alone over the same
pieces costs 135 ns a call on gb10e without timers and 168 ns with them, so the K6 columns are inflated by
~35 ns a call, and the 'driver' column (total minus the kernels) is mostly the timers' cost (instrumented minus
uninstrumented time = 0.7 ns/B on en; the driver itself is <= ~3%).

## 4 KiB chunks, cold (gb10e X925 cpu 7, load 0.2-0.4 / tr9970x Zen 5 cpu 20, sibling 2-5% busy, load 2.1-2.4)

| cell | MB/s uninstr. | ns / piece | K1+K3 | K5 static | K5 miss side | K5 other | K6 short | K6 long | driver + timers |
|---|---|---|---|---|---|---|---|---|---|
| llama3 en | 204 / 259 | 26.4 / 21.4 | 8% / 6% | 7% / 12% | 8% / 7% | 10% / 9% | 51% / 52% | 0% | 15% / 14% |
| llama3 code | 395 / 386 | 13.3 / 13.2 | 14% / 8% | 16% / 18% | 0% / 3% | 12% / 15% | 29% / 30% | 10% / 11% | 19% / 14% |
| llama3 zh | 109 / 161 | 146 / 94 | 8% / 8% | 2% / 2% | 6% / 0% | 8% / 10% | 16% / 18% | 44% / 46% | 16% / 18% |
| gpt-oss en | 191 / 268 | 34.7 / 20.3 | 7% / 7% | 5% / 12% | 14% / 6% | 9% / 11% | 54% / 50% | 0% | 11% / 14% |
| gpt-oss code | 427 / 425 | 12.1 / 12.0 | 17% / 11% | 17% / 21% | 0% / 3% | 14% / 15% | 28% / 29% | 5% / 6% | 20% / 15% |
| gpt-oss zh | 92 / 177 | 173 / 90 | 7% / 9% | 1% / 2% | 7% / 0% | 7% / 10% | 17% / 16% | 48% / 43% | 14% / 19% |

Whole files in one call (cold at the start only; the cache fills as the call goes): gb10e / tr9970x llama3 en 259 /
376, code 487 / 455, zh 115 / 171; gpt-oss en 241 / 384, code 528 / 500, zh 97 / 183 MB/s.

Per piece (llama3 en 4 KiB, gb10e / tr9970x): 10.6% one byte, 79.3% static hits, 1.5% cache hits, 8.5% short misses,
0.06% over 15 B. The fit's K5 cost a piece: one byte 5.5 / 4.5 ns, static hit 2.3 / 3.1 ns, cache hit 17.5 / 20.8
ns (a cold call reaches the cache after both static buckets), the K5 side of a short miss 24.5 / 16.9 ns (the
second words bucket, the cache line, the call, the fill), per round 42 / 30 ns. K6 a short call: 161 / 130 ns in
context (gpt-oss 229 / 125); K6 alone back to back on the same 37,060 pieces: 135 ns (gb10e neon), 168 ns with
the timers. gpt-oss on gb10e is the one in-context penalty (229 vs 174 timed alone): its ~31 MB of tables
(huge pages: 26 MB) outgrow the cache that llama3's ~19 MB still fits.

K6 calls by cause (llama3 en 4 KiB cold, 37,336): first occurrence in the call 16,300; seen in an earlier call
(a scratch kept across calls would hit) 18,628; vocab tokens the words table left out 2,071 (5.6%: two-choice
two-way placement in id order leaves 18,904 of llama3's 126,153 eligible keys out, 14,206 of gpt-oss's 194,250,
3,933 of gpt2's 49,871); > 4 ids 61; over 15 B 276. The short misses' answers: 1 id 5.6% (the left-out vocab),
2 ids 72%, 3 ids 19%, 4 ids 3%; mean 8.2 B, ~6 merges, ~22 ns a merge. Their pieces: space + letters 55%,
letters 25%, mixed 9%, symbols only 5% (en); code: mixed 34%, space + letters 24%, symbols 18%, letters 18%.

Bytes moved per input byte (the c twin, source-level loads + stores of data regions, derived for the asm tiers,
which are not instrumented): llama3 en 40 + 5 (tables 31, text 7 re-reads, ids 3 stored, cache 2.6, ends 1.7),
code 41 + 6, zh 58 + 4; gpt-oss en 43 + 5.

## Distance to SPEC T8's cold floors (gb10e, per core; 4 KiB rows)

en-prose >= 600 MB/s: llama3 204 (0.34x), gpt-oss 191 (0.32x). code >= 600: 395 / 427 (0.66x / 0.71x).
cjk >= 200: zh 109 / 92 (0.55x / 0.46x). On en, K6's short misses alone (~2.5 ns/B after the timer correction)
exceed the floor's whole budget (1.67 ns/B): the floor needs a ~4x faster exact bpe for short first-occurrence
pieces, or far fewer of them; K5, K3 and the driver together are < 2 ns/B.

## Measured and not taken

- A fresh scratch's dynamic cache confined to its first 16 / 64 / 256 KiB (L1 / L2): cold +0..3% on gb10e,
  within noise on tr9970x, pass -1..3% (fills before the warm switch land outside the warm buckets).
- Seating every eligible key in the words table: sizing for load <= 0.85 doubles llama3's table to 8 MiB and
  loses cold 1-3% (the hot entries spread over twice the lines); one-level relocation of any resident seats
  most keys but moves frequent tokens to their second bucket: cold -5% gpt2 / -4.5% gpt-oss (gb10e), -9 / -15%
  (tr9970x, loaded). Relocating only rare residents seats far fewer than the 5.6% of misses would need.
- Placing more words keys by moving only rare residents (first id >= max(8192, id / 4)) to their other bucket:
  llama3 107,249 -> 111,411 seated (the 0.85 cap binds), gpt-oss 180,044 -> 189,842, qwen38 208,726 -> 222,243;
  A/B neutral (gb10e 0.985-1.009, tr9970x 0.981-1.014). Halving the words table (more keys left out): cold -5..6%.
- A "second bucket holds keys of this one" bit in each words bucket (TOKS_WORDS_MORE in way 0's val), so a miss
  skips the second random line when the bit is clear (12-39% of buckets set): gb10e neon en cold +3.7% gpt-oss,
  +6.4% GLM 5.3, +2.7% qwen38, +1.8% llama3, +0.8% gpt2 but -4.2% nemotron 3 4B (10 abba rounds; same 39%
  flagged as llama3, unexplained); avx2 on tr9970x 0.98-1.00 (the line is L3-resident there). Mixed: not taken.
- K6's short path interleaved over 1..8 pieces (one merge step per piece, round robin, c): 183-191 ns a piece
  for every width, as one piece at a time: the short path is not limited by overlappable probe latency.
- A certified two-token split of a short miss (cut P = A | B with A, B self-encoding tokens; exact when no merge
  across the cut can fire: for every right-edge symbol x of A's merge-only run and left-edge symbol y of B's,
  rank(x, y) is absent or above one side's pending bound, the premerge argument of kernels.md 5.1 at token
  level): offline over llama3's en misses it accepts 99.6% of the 2-id answers and never a wrong one, but it
  needs ~9 words probes to find the cut plus ~7 merge probes and both tokens' edge chains a miss, about K6's own
  cost; a byte-folded certificate that fits the words entry accepts only 40% (two-byte fold 67%).
