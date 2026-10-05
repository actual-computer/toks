# tools/ci/win_test.ps1 [-Limit s] [tier ...]: tools\win\test.cmd once per tier (default: native, i.e. TOKS_TIER unset,
# and scalar, the c twins), every tier at once (each test.cmd runs one suite at a time), from the source root, each
# into build\ci\<tier>.log (docs/ci.md, Windows). A per-suite watchdog reads each log's "== <suite>" headers: a suite
# that runs longer than -Limit seconds (default 240; the slowest takes ~15 s on these runners) is killed with its
# test.cmd, and "-- FAIL <suite> killed ..." ends that log, so a hang costs minutes, not the job's timeout (a CRLF
# checkout once kept one suite spinning for 37 minutes). Prints each log whole; exit 1 when a tier failed or was
# killed.
[CmdletBinding(PositionalBinding = $false)]                 # -Limit by name only; every positional word is a tier
param([int] $Limit = 240, [Parameter(ValueFromRemainingArguments)] [string[]] $Tiers)
$ErrorActionPreference = 'Stop'
if (-not $Tiers) { $Tiers = 'native', 'scalar' }
New-Item -ItemType Directory -Force build\ci | Out-Null
$runs = foreach ($t in $Tiers) {
    if ($t -eq 'native') { Remove-Item Env:TOKS_TIER -ErrorAction SilentlyContinue } else { $env:TOKS_TIER = $t }
    $log = "build\ci\$t.log"
    $p = Start-Process cmd -ArgumentList '/c', "tools\win\test.cmd > $log 2>&1" -NoNewWindow -PassThru
    $null = $p.Handle                                       # keeps ExitCode readable after the process is gone
    [pscustomobject]@{ tier = $t; log = $log; proc = $p; suite = ''; since = Get-Date; killed = '' }
}
Remove-Item Env:TOKS_TIER -ErrorAction SilentlyContinue
$t0 = Get-Date
while (@($runs | Where-Object { -not $_.proc.HasExited }).Count) {
    Start-Sleep -Seconds 2
    foreach ($r in @($runs | Where-Object { -not $_.proc.HasExited })) {
        $head = Get-Content $r.log -ErrorAction SilentlyContinue | Where-Object { $_ -like '== *' } | Select-Object -Last 1
        if ($head -ne $r.suite) { $r.suite = $head; $r.since = Get-Date; continue }
        if ($head -and ((Get-Date) - $r.since).TotalSeconds -gt $Limit) {
            $r.killed = "$($head.Substring(3).Trim()) killed after $Limit s (the per-suite limit)"
            taskkill /T /F /PID $r.proc.Id | Out-Null
            $r.proc.WaitForExit()
            Add-Content $r.log "-- FAIL $($r.killed)"
        }
    }
}
$fail = 0
foreach ($r in $runs) {
    "::group::test.cmd ($($r.tier))"
    Get-Content $r.log
    '::endgroup::'
    $what = if ($r.killed) { "FAILED: $($r.killed)" } elseif ($r.proc.ExitCode) { "FAILED, exit $($r.proc.ExitCode)" } else { 'passed' }
    if ($r.killed -or $r.proc.ExitCode) { $fail = 1 }
    "win_test.ps1: test.cmd ($($r.tier), TOKS_TIER=$(if ($r.tier -eq 'native') { 'unset' } else { $r.tier })): $what"
}
"win_test.ps1: $($Tiers.Count) tiers side by side, $([int]((Get-Date) - $t0).TotalSeconds) s"
exit $fail
