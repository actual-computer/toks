# toks benchmark machines

The machines behind every speed receipt in this repository, by chipset. A receipt, a raw log, a generated table
or a chart names its machine by a **key** from the table below; this page is the legend. These machines
are the only sources of speed receipts, and every receipt records the machine's key, its load, the commit and the
corpus.

## Keys

| key | machine | what runs there |
|---|---|---|
| `tr9970x` | AMD Threadripper 9970X (Zen 5), linux x86-64 | the x86 timing box: the avx2 tier (avx-512 is detected; no avx-512 kernel is built), the x86 speed table, the x86 release bundle |
| `gb10a` .. `gb10e` | NVIDIA GB10 (Cortex-X925 + Cortex-A725), linux arm64; five identical units | the arm timing boxes: the neon tier. `gb10c` carries the speed table (`gb10a` did through 0.2), `gb10b` the `toks_par` table, the others parity, fuzzing and kernel A/Bs |
| `m2ultra1`, `m2ultra2` | Apple M2 Ultra 192 GB (Mac Studio), macOS arm64; two units | the macOS neon tier, the macOS speed table (`m2ultra2` from 0.3), the macOS release bundle and wheels |
| `aimax395` | AMD Ryzen AI MAX+ 395, Windows 11 x86-64 | the Windows build and tests (`tools/win`) |
| `aimax395b` | AMD Ryzen AI MAX+ 395 (Zen 5), linux x86-64 | the x86 exactness gate, guard-page runs and compiler differentials while `tr9970x` is unreachable (from 0.3.1); untimed: no speed cell names it |
| `r7950x3d` | AMD Ryzen 9 7950X3D (Zen 4), linux x86-64 | the x86 fuzzing leg (docs/fuzz.md) while `tr9970x` is unreachable; untimed: one CCD carries the 3D V-cache, so its two halves do not share an L3 size |

Two units of one chipset are the same hardware; the key says which box a number came from so that two runs can be
told apart, not because the boxes differ. A developer laptop is not a receipt host: compiles and unit tests
that finish in seconds run there, everything else runs on the machines above.

## Machine classes

### AMD Threadripper 9970X (Zen 5): `tr9970x`

- 32 cores / 64 threads, 125 GB, 4 KiB pages; avx-512 (f, bw, vl, vbmi) and bmi2. toks binds its avx2 tier there
  (no avx-512 kernel is built) and tests the scalar tier beside it; the speed table's x86 cells are the avx2 tier.
- Linux x86-64; clang 21.1.8 (the official LLVM release tarball under `~/.cache/toks-llvm/21.1.8`), uv, make, rsync;
  nasm 3.01 at `~/toks-ci/tools/nasm/bin/nasm` (tok v1's assembler: `tools/bench/tokv1.sh`, the release's tokv1 step).
- Timed runs pin one core with `taskset` and record its SMT sibling's busy share; the `toks_par` table uses one CCD
  (8 cores, siblings idle). The box is shared: its load is recorded with every number.

### NVIDIA GB10 (Cortex-X925 + Cortex-A725): `gb10a` .. `gb10e`

- 10x Cortex-X925 + 10x Cortex-A725, 20 cores, 119 GB, 4 KiB pages; armv8 neon.
- Linux arm64; clang 21.1.8 (same tarball), uv, make, rsync.
- Timed runs pin one X925 core. Fuzzing and the python oracle pools run on the A725 cores so the X925 timing cores
  stay free (`tests/fuzz/run.sh`, docs/fuzz.md).

### Apple M2 Ultra 192 GB: `m2ultra1`, `m2ultra2`

- 24 cores, 192 GB, 16 KiB pages; neon. Mac Studio.
- macOS arm64; clang 21.1.8 (LLVM's, not Apple's), uv, `SDKROOT=$(xcrun --show-sdk-path)`. macOS's system python
  is 3.9, so the oracle runs as `uv run --python 3.12 --with tokenizers==0.23.2 ...`.

### AMD Ryzen AI MAX+ 395: `aimax395` (Windows), `aimax395b` (linux)

- 16 cores / 32 threads, 64 GB, 4 KiB pages; avx2 and avx-512, so `test_k0` reports the same case count as
  `tr9970x` (toks binds avx2 there too).
- Windows 11 (10.0.26200), x86-64. clang 21.1.8 from LLVM's win64 installer at `%USERPROFILE%\.cache\toks-llvm\21.1.8`
  (target `x86_64-pc-windows-msvc`; VS 2022 and the Windows 10 SDK supply the CRT, no vcvars environment needed),
  uv. Point `TOKS_LLVM_BIN` at that `bin` when another clang is first on `PATH`.
- No make and no rsync: `tools/win/build.cmd` builds the library (`libtoks.lib`, `toks.dll`) and `tools/win/test.cmd`
  tests it, from the source root, with tokenizer files from `%TOKS_TOKENIZER_CACHE%`. Sync a `git archive` tarball
  by scp and unpack it with the bundled bsdtar. Over ssh the shell is cmd.exe: run a `.cmd` directly, because a
  nested `cmd /c "x.cmd arg"` hands the batch file a trailing quote and quotes away its redirections; powershell
  needs `-NoProfile -ExecutionPolicy Bypass`; download with `curl.exe` (BITS fails there).

- `aimax395b` is a second unit of the same chipset running Linux x86-64 (glibc 2.39), 4 KiB pages, 64 GB; clang 21.1.8
  (the same tarball), uv, make, rsync. It took over the x86 exactness gate, the guard-page runs and the compiler
  differentials when `tr9970x` went unreachable at the 0.3.1 cut. It is a Zen 5 like `tr9970x` (toks binds avx2
  there too), but it is not a timing box: no speed table cell names it, and the x86 speed table stays `tr9970x`'s.

### AMD Ryzen 9 7950X3D (Zen 4): `r7950x3d`

- 16 cores / 32 threads, 32 GB, 4 KiB pages; avx2 and avx-512 (Zen 4 double-pumps 512-bit vectors; toks binds avx2).
- Linux x86-64 (glibc 2.39); clang 21.1.8 (the same tarball), uv, make, rsync.
- The x86 fuzzing leg while `tr9970x` is unreachable (docs/fuzz.md). Untimed: one CCD carries the 3D V-cache (96 MB
  L3) and the other 32 MB, so a pinned core's cache depends on which half it landed on; nothing timed runs there.

## Etiquette on shared machines

- Record load: `uptime` before and after every timed run goes into the receipt, with the sibling's busy share where
  a core is pinned. A number without its load is not a claim (maintainer doctrine).
- Work only under `~/toks-ci/<branch>/` (`tools/remote.sh` puts you there); nothing outside your home directory, no
  sudo, never touch a process you did not start.
- One timed run per machine at a time: check `uptime` and the pinned core's neighbours first, and wait if another
  timed run is on the box. Background work (fuzzers, oracle pools) goes on the non-timing cores.
- Caches live under `~/.cache/toks/` (`TOKS_TOKENIZER_CACHE`, bench text); the toolchain under `~/.cache/toks-llvm/`.

## tools/remote.sh

```
tools/remote.sh <host> <command...>
```

It rsyncs the worktree (without `.git`, `build/`, `.venv/`) to `<host>:~/toks-ci/<branch>/` and runs the command
there with the pinned toolchain first on `PATH` (`~/.cache/toks-llvm/21.1.8/bin`, `~/.local/bin`; on macOS it also
sets `SDKROOT`). `<host>` is whatever your ssh config calls the machine. Each branch gets its own directory, so two
worktrees on one machine never collide, and `build/` stays on the host between runs. Example:
`tools/remote.sh <host> make -j20 test`. It has no Windows mode: on `aimax395` use the tarball path above and
`tools/win/*.cmd`.

Syncing by hand in a loop, write `${h}:dir`, never `$h:dir`: under zsh `$h:t` is a modifier (the tail of `$h`), so
`rsync ./ $h:toks-ci/x/` copies into a local directory named after the tail of `$h` and the host keeps its old tree.
