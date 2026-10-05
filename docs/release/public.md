# Going public: the flip runbook

How the toks repository goes from a private history to a public one, and what to check before the switch. The
public repository starts from **one orphan root commit of the scrubbed tree**: the private history (commit
messages, PR bodies and receipts that named lab machines, home directories and people) is not carried over.
Commit ids and PR numbers quoted in docs written before that root commit belong to the private history; the
receipts they name are in the tree (README status).

## Before the flip

1. The scrub PRs are merged on master and the tree passes the grep set ([Verify](#verify)) with 0 lines after
   its documented exceptions. The complete set, with the lab's own host aliases, home paths and names, lives with
   the maintainers' notes outside this tree; the one below is its public shape.
2. `LICENSE` is present at the root, `python/pyproject.toml` carries the license field, `include/toks.h` the SPDX
   line.
3. `make -j8 test` and `TOKS_TIER=scalar make -j8 test` are green on a developer laptop, and the CI gate is green on
   the commit that becomes the root commit.
4. Every relative link and every `#anchor` in the tracked markdown resolves: [the link check](#the-link-check)
   prints nothing.
5. The local-only files (the last block of `.gitignore`: the contract, the maintainer doctrine and two agent
   files) are untracked and ignored. Step 2 builds the root tree from the commit, never from a working directory,
   so nothing untracked can ship.
6. The release candidate's receipts were measured at the freeze commit (`docs/release/<major.minor>.md`), and from
   there to the commit that becomes the root only docs changed:
   `git diff --stat <freeze> master -- src include python Makefile tools/bench tools/release` prints nothing.

## The flip

1. The old repository: open PRs land or close, and nothing is pushed to it afterwards. Delete its stale workflow
   registrations (`gh api repos/actual-computer/toks/actions/workflows`) and prune the branches of the private
   history, then rename it (for example `toks-private`) and archive it, so its history stays reachable to the
   maintainers only.
2. Build the root commit from the frozen tree, in a clone with no remote, so nothing can be pushed by accident:

       git clone -q <the private repository> toks-public && cd toks-public && git remote remove origin
       export TZ=UTC GIT_AUTHOR_NAME="<github login>" GIT_COMMITTER_NAME="<github login>" \
              GIT_AUTHOR_EMAIL="<id>+<github login>@users.noreply.github.com" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
       git branch public "$(git commit-tree 'master^{tree}' -m 'toks <version>: first public commit')"
       git rev-parse 'public^{tree}' 'master^{tree}'      # the same tree twice
       git rev-list --count public                        # 1

   `git commit-tree` takes master's tree object, so the root commit's tree is master's by construction: no index,
   no working directory, no ignore rule takes part. The author is the GitHub noreply identity, not a work
   address, and `TZ=UTC` keeps the committer's local offset out of the commit.
3. Create the new repository `actual-computer/toks`. Recommended order (the maintainer's call): create it
   **private**, push, tag, attach the release, wait for CI, run [Verify](#verify), and only then switch it to
   public, so a visitor's first `gh release download` (README, Getting it) finds the release.
   - Before the first push: grant the Blacksmith GitHub App access to the new repository. The workflows run on
     Blacksmith runners, and their runs do not start without it.
   - Push `public` as `master`. The org setting "members can create public repositories" must allow the later
     switch.
4. The cut is a release: its tag is the root commit's tag, and its artifacts are built from the same code, so the
   tag, its source and its binaries agree.
   - Tag the root commit with the version its own `include/toks.h` states (`TOKS_VERSION`; 0.3.0 for the first
     public commit): `git tag -a v<version> -m "toks <version>" public` and `git push origin v<version>`.
   - The bundles and the wheels are built on the release machines at the freeze commit, after the rc
     (`tools/release.sh` with `TOKS_COMMIT` = the freeze sha, docs/release.md step 4), so every bundle's `MANIFEST`
     names the freeze sha, a build id of the private history; Before step 6 shows the root commit carries the
     same code. Attach them to a GitHub release of the tag with their sha256s in its notes (docs/release.md
     step 5). Rebuilding them from the public tag is optional: the same code, with the public sha in the
     `MANIFEST`.
   - Artifacts of the private history's tags (v0.1.0, v0.2.0) are never attached: their source is not in this
     repository.
5. Repository settings, right after the push:
   - description: "exact hf-tokenizers ids, as fast as the hardware allows: C + asm, Python wheel";
     homepage `https://actual.inc`; topics: tokenizer, bpe, sentencepiece, wordpiece, simd, avx2, neon, c,
     assembly, huggingface.
   - a ruleset on `master`: require a pull request, block force pushes and deletion; required status checks
     `linux-x86_64` and `linux-arm64` (test.yml) only. `windows-x86_64` and `macos-arm64` report but are not
     required (the maintainers' decision): Blacksmith's Windows pool held jobs for 5-10 minutes on 2026-10-05,
     and macos.yml runs only on filtered paths and a daily schedule, so a required macOS check would hold pull
     requests that never ran it. The two linux checks report on every pull request, docs only included:
     test.yml's pull_request trigger has no path filter, and its changes job skips both jobs on a docs-only PR,
     which a required check counts as success (docs/ci.md, The required checks); windows.yml works the same way,
     so `windows-x86_64` can be added as it is.
   - a ruleset on tags `v*`: block deletion and force pushes, so a release tag cannot move (recommended).
   - Actions: policy "selected", allowing GitHub-owned and verified actions plus the Blacksmith groups the
     workflows use; fork pull request workflows disabled (no workflow runs for pull requests from forks on our
     runners); "require approval for all outside collaborators" kept; workflow permissions read-only.
   - security: secret scanning and push protection on, Dependabot alerts on (free on public repositories), private
     vulnerability reporting on (recommended).
   - issues on, discussions off, wiki off, projects off unless used; automatically delete head branches
     (recommended); forking allowed if the org permits it.
6. [Verify](#verify), then the maintainer switches the repository to public.

## Verify

- The grep set returns 0 lines after the documented exceptions. Its public shape (the maintainers' copy fills
  the placeholders):

      git grep -nIP '\b(<host-alias>|<host-alias>|...)\b|/home/<user>|/Users/<user>|C:\\Users\\<user>|@<work-domain>|<card-id-pattern>|<person-names>|<lane-and-branch-names>|<private-repository-paths>|SPEC\.md|AGENTS\.md|GOAL\.md|CLAUDE\.md' -- . ':!tests/data'

  Documented exceptions in the 0.3.0 tree: `.gitignore`'s four local-only names; numbers that happen to spell a
  host alias (download counts in docs/coverage.md, token counts in docs/kernels.md); a C variable in
  `tests/c/test_memo.inc` and a model id (`Phi-3-mini-4k-instruct`) that do the same. `tests/data` holds Unicode
  data whose hex spells words.
- `LICENSE` is present. `gh api repos/actual-computer/toks --jq .license.spdx_id` reports `NOASSERTION`:
  GitHub's license detection does not know BUSL-1.1 (hashicorp/terraform, cockroachdb/cockroach and
  getsentry/sentry report the same), so the LICENSE file is the check.
- CI is green on `master` at the root commit.
- The link check prints nothing, the charts render, and the release page carries every bundle and wheel with the
  sha256s the notes list.
- `docs/machines.md` is the only place that describes the benchmark machines, and it names chipsets and keys
  only.

## The link check

Every relative link and `#anchor` in the tracked markdown, checked the way GitHub renders it (an anchor is the
heading lowercased, punctuation dropped, spaces as hyphens, `-1`, `-2` for repeats); external links are not
fetched. From the root of the checkout:

```python
# python3 linkcheck.py: prints each broken link; exit 1 if any
import os, re, subprocess, sys, unicodedata

def slugs(path, cache={}):
    if path not in cache:
        out, seen, fence = set(), {}, False
        for ln in open(path, encoding="utf-8").read().split("\n"):
            fence ^= ln.startswith("```")
            if fence or not re.match(r"#{1,6} ", ln):
                continue
            t = re.sub(r"`|\*\*?|_(?=\w)", "", ln.lstrip("#").strip()).lower()
            s = "".join(c for c in t if c in " -_" or unicodedata.category(c)[0] in "LN").replace(" ", "-")
            out.add(s if s not in seen else f"{s}-{seen[s]}")
            seen[s] = seen.get(s, 0) + 1
        cache[path] = out
    return cache[path]

bad = 0
for f in subprocess.run(["git", "ls-files", "*.md"], capture_output=True, text=True).stdout.split():
    text = re.sub(r"```.*?```", "", open(f, encoding="utf-8").read(), flags=re.S)
    for m in re.finditer(r"\]\(([^)\s]+)\)|(?:src|srcset)=\"([^\"]+)\"", text):
        link = m.group(1) or m.group(2)
        if re.match(r"[a-z]+:", link):
            continue
        target, _, anchor = link.partition("#")
        path = os.path.normpath(os.path.join(os.path.dirname(f), target)) if target else f
        if target and not os.path.exists(path):
            print(f"{f}: missing {link}"); bad += 1
        elif anchor and path.endswith(".md") and anchor not in slugs(path):
            print(f"{f}: no anchor #{anchor} in {path}"); bad += 1
sys.exit(bad > 0)
```

## After

- New receipts print the chipset key, never `hostname` or `$HOME` (tools/bench, tests/par, tools/release).
- The maintainer doctrine's public-repository rule applies to every PR: no secret, hostname, home path or other
  person's name in the tree.
- The first PRs after the flip: the source comments under `src/` that still carry private-history wording.
