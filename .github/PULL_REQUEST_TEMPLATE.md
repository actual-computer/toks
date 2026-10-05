<!-- Thank you. CONTRIBUTING.md first: for now every pull request that merges is run by an Actual Computer
     engineer; yours is read and, if taken, cherry-picked with your authorship kept. -->

## What changes

## Why

The bug, the measurement or the idea; link the issue if there is one.

## Receipts

- Commands run and on what machine: `make test`, `TOKS_TIER=scalar make test`, and `python/build.sh` if the
  Python binding changed.
- Exactness: which tokenizers and cases were compared, ids byte for byte against hf tokenizers 0.23.2.
- Speed, if claimed: host and its load, tokenizer, corpus, chunk size, the exact command and its output, the
  comparator's number beside toks's.

## Sign-off

Every commit carries `Signed-off-by: Name <email>` (`git commit -s`): the DCO plus the grant in CONTRIBUTING.md.
