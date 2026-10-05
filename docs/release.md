# Releasing toks

A toks release is a tag `v<MAJOR.MINOR.PATCH>` on master, the C library + headers per platform, and the Python
wheels. One number: `TOKS_VERSION` in `include/toks.h`. Everything else reads it (`toks_version()`, the wheel's
version through `python/setup.py`, `toks.__version__`, the bundle names, the generated asm include);
`tests/c/test_version.c` checks the macros agree with each other and with the library, `python/tests/test_api.py`
that the wheel, the module and the header agree.

Versions: `TOKS_VERSION_*` is the release (semver). `TOKS_ABI_MAJOR / MINOR` is the C ABI's own compatibility
(include/toks.h): a release that changes the ABI bumps both; an additive ABI change bumps `TOKS_ABI_MINOR`. A pull
request that grows or changes the ABI bumps `TOKS_ABI_MINOR` in the same pull request, so master's header never
claims a compatibility it no longer has (`toks_par_get_info` sat under ABI 0.2 until 0.3.0 fixed the number).

## Cutting a release (the maintainer)

1. master is green at the release commit: `make -j8 test` on a developer laptop, on a GB10 (linux arm64), on the
   9970X (linux x86-64, taskset) and `tools/win/build.cmd` + `tools/win/test.cmd` on the Ryzen AI MAX+ 395
   (docs/machines.md). `tools/release/rc.sh <ref>` runs all of it but the laptop, the critical targets' parity
   samples included, and writes docs/release/<major.minor>.md (from the staged `TOKS_VERSION`): every item met, or
   the reason it is not, including T8's incumbent cells (tools/bench/tokv1.sh against tok v1 on the linux
   machines; `E_SRC` names an optional checkout of the incumbent, never a build or test dependency, and has no
   default). The README's numbers come from that rc's tables: `uv run --with matplotlib tools/readme_charts.py`
   redraws the charts and `python3 tools/readme_numbers.py` prints every number the README quotes, each with its
   cell. The README pull request lists every changed number before -> after (the script on the old tables and on
   the new ones) and quotes none the script does not print.
2. A version PR: `TOKS_VERSION_MAJOR / MINOR / PATCH` and `TOKS_VERSION` in `include/toks.h`, and
   `TOKS_ABI_MINOR + 1` when the ABI grew since the last tag and no pull request bumped it (0.3.0: flags 0 turns
   the segment memo on, `toks_par_get_info`), and the LICENSE Change Date sentence: update it at every tag (each version states its own Change
   Date, four years out; LICENSING.md explains). Merge it, then run rc.sh on that merge commit.
3. Tag the rc'd commit and push the tag:

       git tag -a v<version> -m "toks <version>" <sha> && git push origin v<version>

4. Artifacts, from a worktree at the tag (`git worktree add ../toks-v<version> v<version>`), one run per
   platform:

       tools/remote.sh <m2ultra1> "TOKS_COMMIT=$(git rev-parse HEAD) sh tools/release.sh"
       tools/remote.sh <gb10a>    "TOKS_COMMIT=$(git rev-parse HEAD) sh tools/release.sh"
       tools/remote.sh <tr9970x>  "TASKSET='taskset -c 0-7' TOKS_COMMIT=$(git rev-parse HEAD) sh tools/release.sh"

   (`<key>` stands for whatever your ssh config calls that machine; docs/machines.md has the keys.)

   Each run builds and tests the library (`make lib test` into `build/release/obj-<os>-<isa>`), packs
   `build/release/toks-<v>-<os>-<isa>.tar.gz` (`include/toks.h`, `include/toks.inc` for NASM, `include/toks_asm.h`
   for GNU as / .S, `lib/libtoks.a`, `lib/libtoks.so|dylib`, `MANIFEST`: version, commit, compiler, the kernels
   built, sha256 of every file), then builds and tests the wheels for CPython 3.10..3.14 (`python/build.sh`).
   Linux wheels are retagged by `auditwheel repair` to the lowest policy the module allows: manylinux_2_17
   (manylinux2014) on x86_64 and aarch64, glibc 2.17+ (src/platform/file.c avoids stat / fstat, the only
   glibc >= 2.33 symbols the module used; the module does not link toks_par's pthread calls).
   The bundle's `libtoks.so` does link them: built on glibc 2.39 it needs the newest symbol version its MANIFEST
   names (`glibc` line; GLIBC_2.34 for pthread_create / pthread_join since glibc merged libpthread). `libtoks.a`
   has no floor of its own: its symbols bind when the consumer links it.
   No windows bundle: tools/release.sh is posix sh and windows has no make; tools/win/build.cmd builds the same
   library there (libtoks.lib, toks.dll) and rc.sh tests it.
   At a tag on the Mac the script refuses a tag that is not `v<TOKS_VERSION>` and a dirty tree.
5. Collect them (`scp <host>:toks-ci/v<version>/build/release/*.tar.gz` and `.../build/wheels/*.whl`) and attach
   them to the GitHub release, its notes carrying the consumer's lock lines (the bundles' sha256s, below) and the
   integration flags:

       gh release create v<version> --title "toks <version>" --notes-file <notes> toks-<version>-*.tar.gz toks-<version>-*.whl

   The wheels are not on PyPI yet, and nothing in the package stops an upload: do not publish them (`uv publish`,
   `twine`) until that decision is taken.

## How a consumer pins toks

The first consumer, an inference engine written in asm on both ISAs (gcc as the linker, no C of its own), consumes
the release bundle, not the source:

- its repository records the pin in one place, e.g. `third_party/toks.lock`:

      toks 0.3.0 commit <sha>
      linux-x86_64 sha256 <bundle sha256>
      linux-arm64  sha256 <bundle sha256>

- its build fetches `toks-0.3.0-linux-<isa>.tar.gz` from the GitHub release (`gh release download v0.3.0`),
  checks its sha256 against the lock and unpacks it under `build/`; the .asm / .S sources include `toks.inc` /
  `toks_asm.h` (constants, struct sizes and field offsets, the function names), and the link adds
  `lib/libtoks.a` (static: no runtime dependency, tier dispatch at `toks_load`).
- it checks the pin twice: at assembly time the include's `TOKS_ABI_MAJOR` / `TOKS_VERSION_*` against what it was
  written for (`%if TOKS_ABI_MAJOR != 0` -> `%error`), and at startup `toks_version()` against the
  `TOKS_VERSION` string it assembled with (a stale libtoks.a fails loudly).
- Upgrading is one PR on the consumer's side that changes the lock to the next tag's commit and sha256s and runs
  its tokenizer tests.
- the integration flags (measured with tools/bench/tokv1.sh, docs/bench/e2e.md "Incumbent"): one scratch per
  worker thread, kept across requests, initialized with flags 0. Since 0.3.0 that is the default scratch: the
  4 MiB segment memo (a conversation's earlier turns are answered, as tok v1's memo did) and the 2 MiB piece cache
  (32 MiB loses on new prompts); `TOKS_SCRATCH_MEMO_MIB(n)` resizes the memo and `TOKS_SCRATCH_MEMO_MIB(0)` turns it
  off. Encode calls with flags 0 (hf's default).

A from-source pin works the same way when a consumer wants one: a git submodule at the tag's commit and
`make -C third_party/toks BUILD_DIR=<build>/toks <build>/toks/libtoks.a` (needs clang 21 on its build boxes).
