# tools/ci/win_llvm.ps1: clang 21.1.8 on a Windows CI runner (SPEC §11; docs/ci.md, Windows), where
# tools\win\build.cmd looks for it with TOKS_LLVM_BIN unset: %USERPROFILE%\.cache\toks-llvm\21.1.8\bin (the aimax395
# machine's layout, docs/machines.md). The official release installer, verified by its sha256 (the release's asset
# digest), installed silently into a temporary directory; only what build.cmd runs is kept (clang.exe, llvm-lib.exe,
# clang's resource directory). Nothing to do when the CI cache restored it. windows.yml caches the directory keyed on
# this file's hash, as test.yml does with tools/ci/llvm.sh: a change here is a new cache entry.
$ErrorActionPreference = 'Stop'
$v = '21.1.8'
$d = "$env:USERPROFILE\.cache\toks-llvm\$v"
if (-not (Test-Path "$d\bin\clang.exe")) {
    $t0 = Get-Date
    $tmp = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
    $exe = "$tmp\LLVM-$v-win64.exe"
    curl.exe -fsSL --retry 3 -o $exe "https://github.com/llvm/llvm-project/releases/download/llvmorg-$v/LLVM-$v-win64.exe"
    if ($LASTEXITCODE) { throw "win_llvm.ps1: download failed" }
    $sha = (Get-FileHash -Algorithm SHA256 $exe).Hash.ToLower()
    if ($sha -ne '7a5386c26497db1691f320121e5b113364dd0274b98e55f15f4dbc00c0450113') { throw "win_llvm.ps1: LLVM-$v-win64.exe sha256 $sha" }
    $full = "$tmp\llvm-$v-full"
    $p = Start-Process -Wait -PassThru -FilePath $exe -ArgumentList '/S', "/D=$full"
    if ($p.ExitCode) { throw "win_llvm.ps1: the installer exited $($p.ExitCode)" }
    New-Item -ItemType Directory -Force "$d\bin", "$d\lib" | Out-Null
    Copy-Item "$full\bin\clang.exe", "$full\bin\llvm-lib.exe" "$d\bin"
    Copy-Item -Recurse "$full\lib\clang" "$d\lib\clang"
    Remove-Item -Recurse -Force $full, $exe
    $mb = [math]::Round((Get-ChildItem -Recurse -File $d | Measure-Object Length -Sum).Sum / 1MB)
    "win_llvm.ps1: LLVM-$v-win64.exe: download, sha256, install and prune $([int]((Get-Date) - $t0).TotalSeconds) s, $mb MB kept"
}
$ver = & "$d\bin\clang.exe" --version
if ($ver[0] -notmatch "^clang version $([regex]::Escape($v))( |$)") { throw "win_llvm.ps1: $d\bin\clang.exe is not $v`: $($ver[0])" }
$ver[0]
