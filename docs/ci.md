# CI: the exactness gate

CI answers one question on every pull request and every push to master: does `make test` still pass, in every
tier, on every platform we can rent, with every pinned tokenizer file actually present? It never answers "how
fast": the runners are shared, unpinned virtual machines, so a number from them is not a receipt (SPEC §12.8).
Speed receipts come only from the benchmark machines (SPEC §11, docs/machines.md).

GitHub Actions on Blacksmith's hosted runners (2026-10-04; the Blacksmith GitHub app is installed for the
organization). The workflows: `.github/workflows/test.yml` (linux), `macos.yml` (macOS), `windows.yml` (the Windows
build and tests) and `nightly.yml` (hf parity, the differentials, the wheels); their scripts are in `tools/ci/`.
Every action is pinned by commit, with its version in a comment.

## What runs where

| workflow | job | runner | toolchain | when |
|---|---|---|---|---|
| test.yml | linux-x86_64 | `blacksmith-4vcpu-ubuntu-2404` (AMD EPYC; some hosts have avx-512, some not) | LLVM 21.1.8, the benchmark machines' tarball | every master push that is not docs only; every PR (docs only: skipped, reported as success, The required checks) |
| test.yml | linux-arm64 | `blacksmith-4vcpu-ubuntu-2404-arm` (Ampere, neon) | LLVM 21.1.8 | the same |
| macos.yml | macos-arm64 | `blacksmith-6vcpu-macos-26` (Apple M4 Pro, virtual) | Apple clang 21.0.0 (clang-2100.1.1.101, the image's default Xcode; the m2ultra machines' Xcode carries the same) | PRs that touch `src/platform/**`, `src/asm/arm64/**`, `Makefile`, `tests/common/**`, `tools/ci/**` (not `tools/ci/win_*`) or macos.yml; master pushes that touch those, `src/core/**` or `include/**`; daily at 06:41 UTC on master |
| windows.yml | windows-x86_64 | `blacksmith-2vcpu-windows-2025` (AMD EPYC 4565P, Zen 5; Windows Server 2025, VS 2022 and the 10.0.26100 SDK) | clang 21.1.8 from LLVM's win64 installer | every master push that is not docs only; PRs that touch `src/platform/**`, `src/asm/x86_64/**`, `tools/win/**`, `tools/ci/**`, `Makefile`, `tests/common/**` or windows.yml (other PRs: skipped, reported as success); daily at 05:37 UTC on master; by hand (workflow_dispatch) |
| nightly.yml | parity-<isa>-<tier> (x86_64 and arm64, auto and scalar), norm, bpe, wheels-linux-x86_64, wheels-linux-arm64 | `blacksmith-4vcpu-ubuntu-2404` / `-2404-arm` | LLVM 21.1.8 | Monday-Saturday 07:23 UTC on master (the nightly set); Sundays 07:23 UTC (the weekly superset); by hand (workflow_dispatch, input `weekly`) |
| nightly.yml | wheels-macos-arm64 | `blacksmith-6vcpu-macos-26` | Apple clang 21.0.0 | Sundays; by hand with `weekly` |

Docs only (`**.md`, `docs/**`, `LICENSE*`) runs no test: no test reads them, and tools/size.sh counts source lines
only (a docs-only PR runs the two changes jobs alone: The required checks). Every job runs both tiers: native
(TOKS_TIER unset: what the cpu and the build pick: avx2 on x86_64, neon on arm64; no avx-512 kernel is built, so an
avx-512 host runs avx2 too) and scalar (`TOKS_TIER=scalar`, the c twins). `make test` includes `asmcheck` (every .S
for mach-o, elf and coff on both isas, then the abi lint and SPEC §9 on the built objects, tests/abi/cf_audit.py) and
`size` (SPEC T10).

macOS runs only where a mac build can break, because its minute costs twenty linux ones (Minutes, below) and a
night's merge train lands 20-30 PRs: on PRs that touch the platform layer, the arm64 asm (mach-o), the Makefile, the
test harness's own asm and guard pages, or tools/ci (its Windows-only scripts excepted); on master pushes that touch
those or the C core and its header (what Apple clang compiles differently); and once a day on master whatever
changed, so a break from anywhere else shows within a day. linux-arm64 runs neon and scalar on every PR and asmcheck
assembles every .S for mach-o on every job, so such a break is rare and is fixed forward. macos-15 is not used: its
newest Xcode ships Apple clang 17. The m2ultra machines' receipts build with LLVM's clang 21.1.8 (`tools/remote.sh`
puts it first on PATH); CI's mac job and a developer laptop build with Apple clang 21. Both are clang 21 (SPEC
§11).

## One job, step by step

1. start (the time, for the cost line), checkout (no credentials kept).
2. linux only: `tools/ci/llvm.sh`: the official `LLVM-21.1.8-Linux-{X64,ARM64}.tar.xz`, sha256 pinned in the
   script, only clang, llvm-objdump, llvm-ar / llvm-ranlib and clang's headers unpacked (~350 MB of ~2 GB) into
   `~/.cache/toks-llvm/21.1.8`, the benchmark machines' path, then first on PATH. Cached (key: os, arch, the
   script's hash); a miss costs 52 s on x86_64 and 78 s on arm64.
3. tokenizers: restore the cache, `python3 tools/ci/fetch_tokenizers.py` (its exit fails the step: test.yml's and
   macos.yml's `run:` steps are `bash -eo pipefail`, their `defaults`), save the cache when the restore missed.
4. `tools/ci/test.sh`: `make -j all asmcheck` once, then `make -j test` per tier, one tier after the other (make
   runs a tier's test binaries side by side on every cpu; the tiers do not overlap, so one tier's load never
   reaches the other's timing checks, test_stall's), each tier's whole output printed when it finishes, then
   `tools/ci/suites.py`.
5. the logs (`build/ci/fetch.log`, `build.log`, `native.log`, `scalar.log`) uploaded as the artifact
   `test-logs-<job>`, kept 7 days.
6. `tools/ci/cost.sh`: the job's cost so far, in the log and the job summary.

## The Windows job

windows.yml builds and tests the way the aimax395 machine does (docs/machines.md): `tools\win\build.cmd` (the
library as `libtoks.lib` and `toks.dll`, every tests/c binary, test_e2e against the dll, the K3 tiers direct and
through the abi canary harness), a check that its kernels line names the seven avx2 kernels (K1, K3 cl100k / dsv3 /
o200k, K5, K6, K7 spm), then `tools/ci/win_test.ps1`: `tools\win\test.cmd` once per tier, native and
`TOKS_TIER=scalar` side by side (the scalar tier is the c twins built for LLP64 and the msvc crt), with a per-suite
limit (a suite still running at 240 s is killed and its log ends `-- FAIL <suite> killed ...`; the slowest takes
~15 s), then `tools/ci/suites.py` over both logs as on linux. clang comes from `tools/ci/win_llvm.ps1`:
`LLVM-21.1.8-win64.exe`, sha256 pinned, pruned to clang.exe, llvm-lib.exe and clang's resource directory, under
build.cmd's default `%USERPROFILE%\.cache\toks-llvm\21.1.8` (key: os, arch, the script's hash; a miss costs ~30 s);
the image's own clang (20.1.8 on PATH) is never used, because build.cmd calls clang by its full path. The tokenizer
cache is the one scheme below. Two runner settings make the job see the source exactly: `core.autocrlf false` before
the checkout (Git for Windows converts LF to CRLF by default and the tests read their data as bytes: with CRLF,
test_compile fails 48 refusal names and test_tiktoken's case parser spins; .gitattributes pins every text file to LF,
and the setting keeps the runner's config out of it as well), and `HOME` set to `%USERPROFILE%` (test_k7 and the
tests' default paths read `$HOME/.cache/toks/...`; Windows sets only USERPROFILE).

It runs where a Windows build can break, for the same reason macOS does: its minute costs two linux ones, and test.yml
is 9 a PR push against a budget of 12. So PRs run it when they touch the platform layer, the x86_64 asm (coff and the
win64 abi), tools/win, tools/ci, the Makefile or the test harness's asm and guard pages; every master push that is not
docs only runs it (6 units: the proof that the merged core still builds and passes under LLP64 and the msvc crt); and
once a day on master. A maintainer can run it on any branch by hand (workflow_dispatch). It is reported, not required
(docs/release/public.md). Reading a Windows failure: the artifact `windows-x86_64-logs` holds fetch / build / native /
scalar logs for 7 days; `FAIL compile <file>` or `FAIL build <file>` in build.log is a build error; `build.cmd built no
K..._AVX2` means a kernel source is gone or renamed; `-- FAIL <suite> exit <code>` is a suite's nonzero exit (a crash
exits negative: -1073741819 is 0xC0000005); `-- FAIL <suite> killed after 240 s` is the per-suite limit. Reproduce on
aimax395 with a `git archive` of the branch (docs/machines.md), then `tools\win\build.cmd` and `tools\win\test.cmd`
from the source root.

## The nightly

nightly.yml runs what is too heavy for a push. It never runs on a pull request, and it uses test.yml's llvm and
tokenizer steps and keys, so it restores what test.yml saved. Only parity runs both tiers, a job for each; norm, bpe
(the c twins) and the wheels run once:

- parity-<isa>-<tier>: tools/release/rc_host.sh's parity step for one tier (`RC_STEPS=parity TIERS=auto` or
  `scalar`, one shard per vcpu): the release's own seeded sample (tests/parity/gen_cases.py --quick, every 7th
  encode / pieces case, every decode / stream case, gen_stream.py's adversarial streams) against hf tokenizers
  0.23.2 for gpt2, llama3, glm53, qwen38, o200k, gemma4, nemotron3-4b, llama4, minimaxm2 and dsv4, and kimik3
  against transformers + tiktoken (run_kimi.py: 166,284 texts, three modes and two decodes each). x86_64 runs avx2
  and scalar, arm64 neon and scalar: the only hf parity of the arm64 kernels in CI (`make test` does not compare
  against hf). The sample is seeded, so the counts equal the release reports': 217,388..221,783 ids cases and
  95,385..97,584 pieces cases per target and tier (kimik3's texts are generated per shard, run_kimi.py, so its
  sample follows the shard count). The four jobs run on 4 vcpus: the 16-vcpu labels queued 20 and 14 minutes on
  2026-10-05 while the 4-vcpu ones started in seconds, and a tier on 4 vcpus takes 3.5 minutes on x86_64 and 9.5 on
  arm64, against 6 and 10.5 for both tiers on 16.
- norm: `tests/norm/run.sh check`: 22 jobs, 313,357 records against hf 0.23.2's NFC and Qwen 3.8's own normalizer.
  The corpus (tests/norm/corpus.py: 64 languages, 225 MB, 61 MB in the cache, fetched from the hub's
  datasets-server) is cached under the hash of corpus.py: the first fetch takes ~24 min at the server's rate limit,
  later runs restore it in seconds.
- bpe: tests/bpe/run.sh for gpt2, llama3, qwen38 and o200k: K6 / K5's c twins against hf 0.23.2's BPE model on the
  vocabulary plus 1M generated pieces (5M on Sundays).
- wheels-<os>: python/build.sh for CPython 3.10-3.14 and python/tests on each. On linux x86_64 and macOS, 3.13 also
  runs the hf parity tests through the Python API (test_parity.py: six tokenizers and every ledger target, with the
  quick case sets generated first). linux arm64 skips them: neon's hf parity is parity-arm64-auto's, and the Ampere
  takes 10 minutes over them. Every wheel is installed into a fresh venv and imported, and once more with the stock
  interpreter's pip. The wheels are uploaded for 14 days, never published.
- The count rule (`tools/ci/assert_count.py`): every runner ends with `PASS <n> compared <suite> (...)`. A job fails
  on a FAIL line, on n = 0, on a runner that printed no line, on fewer lines than the job runs, and on a critical
  target that did not run (parity: the first five + Llama 3 by rc_host.sh's names, every target and tier its
  steps.txt and host.txt list, every kimi shard; wheels: `CRITICAL` in tools/ci/fetch_tokenizers.py). The lines also
  go to the job summary.
- Reading a nightly failure: the counts step's FAIL line names the suite, and the job's artifact holds the runner's
  records for 14 days (parity-<isa>-<tier>: build/rc, every shard's log and report.json; norm: every job's log; bpe:
  one log per tokenizer; wheels-<os>: the wheels and the python report). A parity diff reproduces on a benchmark
  machine with tests/parity/shards.sh on the same seeded case file.

## The required checks

`linux-x86_64` and `linux-arm64` are the checks a pull request must pass (docs/release/public.md), so they report on
every pull request, docs only included. A workflow that a path filter in `on:` skips reports nothing, and a required
check that never reports holds the PR; a job that its `if:` skips reports success. So test.yml's pull_request
trigger has no filter. Its `changes` job (`blacksmith-2vcpu-ubuntu-2404-arm`, one billed minute: 0.625 units) runs
`tools/ci/changes.sh`: the PR's merge commit against the base it merges into (a rename counts both paths), output
`code=false` when only docs changed. The two linux jobs need it and skip on `code == 'false'`; anything else runs
them: a change outside the docs, a push or a dispatch (changes does not run), a changes job that failed (no output).
windows.yml does the same with the script's `windows` output (the Windows paths above), so `windows-x86_64` can be
required as it is. macos.yml keeps its filter in `on:`: it is not required, and a filtered workflow costs nothing on
the PRs it skips. The two linux jobs are two jobs, not a matrix: a job's `if:` is evaluated before its matrix
expands, so a skipped matrix job would report one check under its unexpanded name and leave both names pending; they
share their steps through a YAML anchor.

## Reading a failure

- The step that failed says which kind it is.
  - **tokenizers**: one line per pinned file: `ok` (in the cache, sha256 equal), `fetched`, `gated`, `missing`,
    `bad-sha`, `error`, and a `critical` column. `FAIL: critical targets without their pinned file` lists what
    blocks the job. `gated` needs the `HF_TOKEN` secret (below); `missing` means the pinned hub revision is gone and
    the pin's owner re-pins; `bad-sha` means the hub served other bytes for a pinned revision; `error` is the
    network after 4 attempts (re-run the job).
  - **make test**: open the group of the failing tier, or `native.log` / `scalar.log` from the artifact. Every
    binary runs; their outputs follow in a fixed order, each under its `== build/<os>-<isa>/tests/<binary>` line,
    and the tier ends `make test (<tier>): FAIL: <binaries>` when any failed: the first `FAIL` line in a named
    binary's section is the cause. A compile error, an .S that does not assemble for one format, the abi lint or
    cf_audit (docs/hardening.md §3) stops the job before any tier runs (`build.log`).
  - **suites** (the last lines of the step): `suites: FAIL ...` says what: a binary that did not run, a binary
    whose counts were all 0, a SKIP that names a critical target's file, a cpu that binds none of the asm tiers
    the build has (no asm kernel ran), or a missing asmcheck / abi lint / cf_audit / size line. The same `tools/ci/suites.py`
    reads `tools\win\test.cmd`'s log (its headers carry no path): there a missing `test.cmd: FAIL=0` end or a
    `-- FAIL` line fails too, and test_e2e_dll and the K3 tier exes are counted beside tests/c.
- Reproduce on a benchmark machine (docs/machines.md) with the same scripts, its toolchain and its staged cache
  (verify it, never fetch into a shared machine's cache), on its build cpus; on a gb10:
  `tools/remote.sh <host> 'python3 tools/ci/fetch_tokenizers.py --check && taskset -c 0-4,10-14 tools/ci/test.sh'`
  (the A725 cores; on tr9970x `taskset -c 0-7,32-39`, one CCD; on an m2ultra no taskset). One tier:
  `tools/ci/test.sh scalar`. A developer laptop runs `make -j8 test` and `TOKS_TIER=scalar make -j8 test`,
  nothing heavier.
- While a job runs, its log prints an ssh line to the live runner (Blacksmith, your GitHub key).

## The SKIP policy

Tests SKIP a pinned tokenizer file that is not in the cache, so `make test` runs on any machine. CI does not accept
that silently:

- Every pinned file is fetched before the build. The critical targets, SPEC §1.1 (d) (the first five: GLM 5.3, Kimi
  K3, gpt-oss, Qwen 3.8, Gemma 4; and the last generations of llama, nemotron, deepseek and minimax: 23 ledger
  targets, `CRITICAL` in `tools/ci/fetch_tokenizers.py`), must be there with their pinned sha256 after the fetch,
  or the job fails before it builds.
- After the tests, a SKIP line that names a critical target's file fails the job (`suites.py`), whatever its cause.
- A binary that printed counts, all of them 0, fails the job. A binary that prints no count at all is listed
  (today: test_json_hostile, test_json_smoke, test_proof, test_version, test_wp).
- A native tier that ran no asm kernel fails the job: test_tier's `tier auto = scalar (...; built: avx2)` means the
  runner's cpu cannot run the asm tier the build has, so the "native" pass was the scalar one again.
- Every other SKIP is allowed and printed with its tier and binary, so a growing list is visible on every run.
  Today the linux and mac jobs print none, except test_k0's avx-512 frame on an x86_64 host without avx-512.

## The caches

- Tokenizers. The pins are read where their owners keep them, never copied: `tools/corpora/fetch_tokenizers.py`
  (TOKENIZERS, with `tests/wordpiece/pins.json` and `tests/unigram/pins.json`), `tests/spm/pins.json`, and
  `tests/parity/oracle_tiktoken.py` (MODEL_FILES: the kimi k3 directory). `tests/data/targets/ledger.txt` must agree
  with them; the fetcher checks it. Today: 122 files, 919 MB (~217 MB in the cache), ledger 91 targets.
  - Layout: `~/.cache/toks/tokenizers/<name>` and `~/.cache/toks/kimik3/<hub name>`, the tests' defaults (some tests
    read these `$HOME` paths directly, so CI does not move them).
  - Key: `tokenizers-v1-<runner.os>-<python3 tools/ci/fetch_tokenizers.py --key>`, the sha256 of the pin table. A pin
    change makes a new key and a full fetch (4-8 s on these runners); bump `v1` to force one.
  - `HF_TOKEN`: not needed (every pinned file answers without a token), so no workflow passes it. The fetcher sends
    `$HF_TOKEN` to huggingface.co only (never to the cdn a download redirects to); if a gated pin ever lands, a
    maintainer adds the repository secret (a read token of an account that accepted the repo's terms) and the fetch
    step passes it.
- LLVM: `~/.cache/toks-llvm/21.1.8`, key `llvm-21.1.8-<os>-<arch>-<hash of tools/ci/llvm.sh>` (on Windows, of
  `tools/ci/win_llvm.ps1`, the installer's recipe).
- The norm corpus (nightly.yml only): `build/norm-corpus`, key `norm-corpus-v1-<hash of tests/norm/corpus.py>`.
- One scheme for every workflow: the same scripts, paths and keys, so a run restores what any other run saved. Each
  os keeps its own entries (runner.os is in every key; a macOS run does not restore a linux entry even with the same
  paths), so no workflow sets `enableCrossOsArchive`. A PR run reads its own entries and master's; an entry unused for
  7 days is evicted (Blacksmith's colocated cache behind `actions/cache`).

## Minutes

Blacksmith bills in x64 2-vcpu minutes: an x64 runner of n vcpus costs n/2 a minute, arm 0.625 x n/2, windows
2 x n/2, a 6-vcpu mac 20. 3000 a month are free per organization, then about $0.004 each. Every job ends with a
`cost:` line (log and job summary): its seconds from the start step, rounded up to the billed minute, times its rate.
Measured 2026-10-05 (caches warm; `make -j test` per tier, the tiers one after the other, test_stall last in each):

| job | wall | billed |
|---|---|---|
| linux-x86_64, 4 vcpu | 1m17s (build and native tier 35 s, scalar 35 s) | 2 min x 2 = 4 |
| linux-arm64, 4 vcpu | 2m27s (69 s, 66 s) | 3 min x 1.25 = 3.75 |
| macos-arm64, 6 vcpu | 53 s (24 s, 24 s) | 1 min x 20 = 20 |
| windows-x86_64, 2 vcpu | 2m45s (build 28 s, both tiers 115 s) | 3 min x 2 = 6 |

So test.yml is 7.75 a push, a push that also runs windows.yml 13.75, and one that also runs macos.yml 20 more; a PR
push adds the two changes jobs (0.625 each), and a docs-only PR push costs those 1.25 alone. Shared runners vary: the
same linux-arm64 job took 127, 146, 147 and 254 s on four runs (with the test binaries one at a time and the two
tiers side by side it took 199-235 s and billed 5; linux-x86_64 103-107 s, 4; macos-arm64 102-104 s, 40). The Windows
job on 4 vcpus took 137 s for 12. Blacksmith's Windows runners are a public beta: they start in 15-50 s, but jobs
have queued 8-9 minutes at either size, and a macOS job queued 12 minutes. On cold caches with 2-vcpu linux and the
tiers one after the other the first run took 4m06s / 8m17s / 3m13s. The wall now is each tier's longest binaries
(test_e2e) and test_stall, which runs last and alone in its tier.

nightly.yml, from its jobs' cost lines (2026-10-05, caches warm, the weekly superset):

| job | wall | billed |
|---|---|---|
| parity-x86_64-auto / -scalar, 4 vcpu | 3m20s / 3m35s | 4 min x 2 = 8 each |
| parity-arm64-auto / -scalar, 4 vcpu | 9m35s / 9m22s | 10 min x 1.25 = 12.5 each |
| norm, 4 vcpu | 2m29s | 3 min x 2 = 6 |
| bpe, 4 vcpu, 5M pieces (Sundays) | 6m11s | 7 min x 2 = 14 |
| wheels-linux-x86_64, 4 vcpu | 4m06s | 5 min x 2 = 10 |
| wheels-linux-arm64, 4 vcpu | 55 s | 1 min x 1.25 = 1.25 |
| wheels-macos-arm64, 6 vcpu (Sundays) | 3m51s | 4 min x 20 = 80 |

The nightly bpe at 1M pieces is not yet measured on 4 vcpus (1m34s on 8 vcpus, caches cold); at 3 min x 2 = 6 a
night is ~64, a Sunday ~152 and a month ~2,300. The four parity jobs are 41 of a night's 64; on 16 vcpus the two took
103, since their wall barely shrank with the vcpus. Shared VMs vary (one 16-vcpu x86 VM ran the same parity in
27m21s, every step slower): a job that times out on a bad VM is re-run.

## Deliberately not in CI

- Benchmarks, the gate's speed tables (tools/bench), e2e A/B runs, the release speed tables (tools/release): they
  need pinned, quiet, recorded machines (SPEC §12.8). A shared VM's number is not a receipt.
- Fuzz campaigns (docs/fuzz.md): hours of cpu on the benchmark machines; `make fuzz` (the build) could join later.
- hf parity at scale, the norm and bpe differentials, the wheels: nightly.yml, not per push.
- The avx2 tier forced: the x86_64 runners pick avx2 themselves (no avx-512 kernel is built).
- Windows arm64: Blacksmith has no Windows arm64 runner, GitHub's windows-11-arm bills outside Blacksmith, and
  tools/win builds x86_64 only; it is not a release target today.
- The hf differential over tests/k3/gen.py case files on Windows (`tools\win\test.cmd` takes them): hf parity runs
  nightly on linux.
