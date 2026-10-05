# Contributing to toks

toks is small on purpose, exact by contract and fast by measurement. Contributions that keep it that way are
welcome.

## What we merge, for now

Issues, bug reports, reproductions, benchmark receipts and ideas are welcome, and so are pull requests. For now
Actual Computer Inc. does not merge contributor pull requests directly: every change lands through a pull
request run by an Actual Computer engineer, who takes your change into it once it passes the exactness and
speed gates on the benchmark machines. Your commits are cherry-picked with their authorship kept; an idea or a
bug report that we implement is credited in the commit message. The reasons are practical: the gates need
machines and reference runs that live with us, and the license model below needs every line in the tree to
carry the grant. Code we take from a pull request, or from a patch posted in an issue, needs the sign-off below
all the same.

## How work lands

- One branch per change, a pull request to `master`. Keep the diff to the change; every line is read.
- `make test` and `TOKS_TIER=scalar make test` (the portable C twin of every kernel) must pass on your machine
  before you open the PR; `make test` also runs `asmcheck` (every .S for mach-o / elf / coff, then the abi lint)
  and the size budgets (`size`). Fetch the pinned tokenizer files first (`python3 tools/ci/fetch_tokenizers.py`,
  into `~/.cache/toks`): without them the tests that read a real tokenizer file print SKIP, and CI runs them all
  ([docs/ci.md](docs/ci.md)). Changes to the Python binding also run `python/build.sh` (or
  `uv build --wheel python --out-dir build/wheels` and `uv run --no-project --with build/wheels/toks-*-cp312-*.whl
  --with pytest --with tokenizers==0.23.2 pytest python/tests`, as [python/README.md](python/README.md) shows). Say
  in the PR which commands you ran and on what.
- A change that grows or changes the C ABI (`include/toks.h`) bumps `TOKS_ABI_MINOR` in the same PR
  ([docs/release.md](docs/release.md)).
- Exactness is the contract: for every tokenizer toks supports, its ids are hf tokenizers 0.23.2's ids, byte for
  byte. A change that moves an id is a bug unless the oracle under `python/toks_oracle` and `tests/` says
  otherwise, and then the PR explains why.
- A speed claim ships with its receipts, the way [docs/bench](docs/bench) does: the host and its load, the
  tokenizer, the corpus, the chunk size, the exact command and its output, and the comparator's number beside
  toks's. A number without them is not a claim.
- No third-party code in the C and assembly core (`src/`, `include/`). If a port would need an attribution,
  write an original solution instead; design lineage goes in a comment, not code.
- Simplicity wins ties. Families are data, not code; a failed experiment is deleted and its negative result
  recorded; there are no default-off switches.

## Sign-off and license grant

toks is licensed under the Business Source License 1.1 and converts to Apache-2.0 version by version
([LICENSING.md](LICENSING.md)); Actual Computer Inc. also sells commercial licenses. For that model to hold,
Actual Computer Inc. must be able to license every contribution under each of those terms. So every commit in
a pull request carries a sign-off line with your real name and an email address you answer at:

```
Signed-off-by: Ada Lovelace <ada@example.com>
```

`git commit -s` adds it. By adding it you certify the Developer Certificate of Origin 1.1
(https://developercertificate.org) and, in addition, you agree to the following, in plain language:

1. **License grant.** You grant Actual Computer Inc. a perpetual, worldwide, non-exclusive, royalty-free,
   irrevocable license to use, reproduce, modify, distribute, sublicense and relicense your contribution, in
   whole or in part, under any terms it chooses, including the Business Source License 1.1, the Apache License,
   Version 2.0, and commercial licenses, and to do the same with works that include your contribution.
2. **Patent grant.** For patent claims you can license that your contribution necessarily infringes, alone or in
   combination with toks, you grant Actual Computer Inc. and every recipient of toks a perpetual, worldwide,
   non-exclusive, royalty-free, irrevocable patent license to make, use, sell, offer for sale, import and
   otherwise transfer the contribution and toks.
3. **You have the right.** You certify that the contribution is your original work, or that you have the right
   to submit it under these terms, and that if you made it in the course of your employment your employer has
   agreed to these terms.
4. **You keep your copyright.** This is a license, not an assignment. You may use your contribution for anything
   else you like.

Pull requests without a sign-off on every commit are not taken.
