tests/model: the executable model of the kernel contract
=========================================================

toks_model.py is a straightforward python transcription of docs/kernels.md (atoms and classes, the cl100k
template rules A1-A7 under layout.h's TOKS_TP_* parameters, the added-token driver policy, byte-level bpe with
ignore_merges and duplicate merges keeping the last). It was written by the review of the kernel contract (merged
in commit 9fa466c) and compared against hf tokenizers 0.23.2: 14.8M differential cases, 0 mismatches (REVIEW.md
has the counts and generators).

Use it as the reference while a c twin is young, and as a case generator; hf 0.23.2 stays the definition.

  uv run --with tokenizers==0.23.2 tests/model/smoke.py         # downloads the pinned tokenizer.json files here
  uv run --with tokenizers==0.23.2 tests/model/run_k3_fuzz.py   # K3 differential fuzz (big: run on a lab host)
  uv run --with tokenizers==0.23.2 tests/model/run_bpe.py       # K6 pieces vs hf's model
  uv run --with tokenizers==0.23.2 tests/model/run_added.py     # added-token policy vs hf

probes.pkl / wordset.pkl and the downloaded tokenizer files are caches, regenerated on first run, never committed.
