# Unit test for scripts/title-smoke.ps1: feeds its own summary function a synthetic log and
# checks the parsed values. No Pester: throws on the first wrong value, exits 1.
$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'title-smoke.ps1'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$null, [ref]$null)
$function = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Get-TitleSmokeSummary'
}, $true)
if (-not $function) { throw 'Get-TitleSmokeSummary not found in title-smoke.ps1.' }
. ([scriptblock]::Create($function.Extent.Text))

function Assert-Equal {
    param([string]$Name, $Actual, $Expected)
    $actualText = if ($null -eq $Actual) { '<null>' } else { [string]$Actual }
    $expectedText = if ($null -eq $Expected) { '<null>' } else { [string]$Expected }
    if ($actualText -ne $expectedText) { throw "FAIL $Name`: expected '$expectedText', got '$actualText'" }
}

try {
    $log = @'
synthetic start
D3D12.X: 300 frames presented, 29.5 fps over the last 300
sparse windows: 16 submits, 12 MB of CPU writes uploaded, 40 ms spent, 9 tiles mapped
sparse windows timing: write-watch 40 ms (12 dirty pages), lock wait 3 ms, chunk scan 5 ms, upload 8 ms (4 runs)
D3D12.X: console code so far: 64 draws with translated pipelines (2 skipped, 1 with PC pipelines), 32 dispatches (3 skipped); coherent 100/200 MB, sparse 50 MB, 4 blocks held for 2 allocators
D3D12.X: 600 frames presented, 30.0 fps over the last 300
D3D12.X: 900 frames presented, 31.5 fps over the last 300
D3D12.X: 1200 frames presented, 33.0 fps over the last 300
sparse windows: 64 submits, 22 MB of CPU writes uploaded, 90 ms spent, 11 tiles mapped
sparse windows timing: write-watch 192 ms (33 dirty pages), lock wait 7 ms, chunk scan 9 ms, upload 12 ms (5 runs)
D3D12.X: console code so far: 128 draws with translated pipelines (5 skipped, 1 with PC pipelines), 64 dispatches (3 skipped); coherent 100/200 MB, sparse 50 MB, 4 blocks held for 2 allocators
D3D12.X: XErrorReport 0x8007000e: synthetic
thread 'main' panicked at synthetic
unhandled exception: synthetic
crash: synthetic
D3D12 device removed: 0x887a0005 synthetic
device lost (0x887A0006) synthetic
'@

    $s = Get-TitleSmokeSummary -Text $log -ExitCode 3

    Assert-Equal 'exit_code' $s.exit_code 3
    Assert-Equal 'frames_presented' $s.frames_presented 1200
    Assert-Equal 'fps_recent' ($s.fps_recent -join ',') '30,31.5,33'
    Assert-Equal 'fps_average' $s.fps_average 31.5
    Assert-Equal 'sparse_submits' $s.sparse_submits 64
    Assert-Equal 'write_watch_ms' $s.write_watch_ms 192
    Assert-Equal 'dirty_pages' $s.dirty_pages 33
    Assert-Equal 'write_watch_ms_per_submit' $s.write_watch_ms_per_submit 3
    Assert-Equal 'lock_wait_ms' $s.lock_wait_ms 7
    Assert-Equal 'chunk_scan_ms' $s.chunk_scan_ms 9
    Assert-Equal 'upload_ms' $s.upload_ms 12
    Assert-Equal 'upload_runs' $s.upload_runs 5
    Assert-Equal 'draws' $s.draws 128
    Assert-Equal 'draws_skipped' $s.draws_skipped 5
    Assert-Equal 'draws_pc' $s.draws_pc 1
    Assert-Equal 'dispatches' $s.dispatches 64
    Assert-Equal 'dispatches_skipped' $s.dispatches_skipped 3
    Assert-Equal 'device_removal_lines' $s.device_removal_lines 2
    Assert-Equal 'panic_lines' $s.panic_lines 1
    Assert-Equal 'exception_lines' $s.exception_lines 1
    Assert-Equal 'crash_lines' $s.crash_lines 1
    Assert-Equal 'xerrorreport_lines' $s.xerrorreport_lines 1

    $empty = Get-TitleSmokeSummary -Text ''
    Assert-Equal 'empty exit_code' $empty.exit_code 0
    Assert-Equal 'empty frames' $empty.frames_presented $null
    Assert-Equal 'empty fps count' @($empty.fps_recent).Count 0
    Assert-Equal 'empty fps average' $empty.fps_average $null
    Assert-Equal 'empty submits' $empty.sparse_submits $null
    Assert-Equal 'empty per submit' $empty.write_watch_ms_per_submit $null
    Assert-Equal 'empty device removals' $empty.device_removal_lines 0
    Assert-Equal 'empty panics' $empty.panic_lines 0
    Assert-Equal 'empty exceptions' $empty.exception_lines 0
    Assert-Equal 'empty crashes' $empty.crash_lines 0
    Assert-Equal 'empty xerror' $empty.xerrorreport_lines 0

    'ok: title-smoke summary parsed'
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
