# Where toks is behind, by cell

The speed table ([e2e.md](e2e.md), generated from the raw logs under [raw/](raw)) compares toks with hf tokenizers,
tiktoken and gigatoken in every cell, and its [Gates](e2e.md#gates) section lists the result of every comparison.
This page names the cells toks loses, as of the table at commit `245cc5c` (the 0.3.0 release candidate), so the
README can carry the tally and this page the detail. It is rewritten with each regenerated table.

- **Cold against gigatoken: three whole-corpus GPT-2 calls.** tr9970x code 0.87x (381 vs 438 MB/s) and English
  prose 0.95x (410 vs 431), and m2ultra2 code 0.99x (381 vs 383). Every 4 KiB cell is ahead on every machine.
- **Pass against gigatoken: six cells.** gb10c MiniMax M2 code 4 KiB 0.96x and English 4 KiB 0.97x; tr9970x
  GPT-2 code whole 0.90x, code 4 KiB 0.90x and English whole 0.99x; m2ultra2 GPT-2 code 4 KiB 0.97x.
- **Warm replays the memo doesn't answer.** toks wins 43-50 of 85 warm cells per machine. The 35-42 it loses are
  replays that outgrow the default 4 MiB memo and fall back to the piece cache, where gigatoken's 512 MiB pretoken
  cache is faster: every multilingual whole-corpus call, every CJK one but DeepSeek V4's, 8-11 multilingual and 2
  CJK 4 KiB cells, and 5-9 English whole-corpus calls per machine. The worst is Gemma 4 on CJK whole, 0.19-0.24x. The
  memo keeps each record's text beside its ids; records that keep the ids alone are the next step.
  [Gates](e2e.md#gates) lists every cell.
- **The floor itself.** The speed table measures toks against other tokenizers. The number it will carry next is
  the gap to the machine's physics floor, bytes moved per input byte at measured bandwidth: that gap is the target.
