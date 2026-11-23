# Unattended smoke run: starts a title with the release build, ends it after a fixed time,
# keeps stdout+stderr in a timestamped log and prints a short summary parsed from that log.
# The summary is also written as summary.json in the log folder. Log lines are never printed:
# a title's log can contain its folder and file names, so it stays local.
#
#   powershell -ExecutionPolicy Bypass -File scripts\title-smoke.ps1 -Game D:\games\title
#   powershell -ExecutionPolicy Bypass -File scripts\title-smoke.ps1 -Game D:\games\title -Seconds 90 -PressAAt 20
#   powershell -ExecutionPolicy Bypass -File scripts\title-smoke.ps1 -Game D:\games\title -Env @{XCL_ACTIVITY='1'} -Label menus
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Exe = (Join-Path (Split-Path -Parent $PSScriptRoot) 'target\release\bathyal.exe'),
    [int]$Seconds = 120,
    [double]$PressAAt = 30,
    [string]$Logs = (Join-Path $env:LOCALAPPDATA 'bathyal\smoke'),
    [hashtable]$Env = @{},
    [string]$Label = ''
)
$ErrorActionPreference = 'Stop'

# Parses the text of one run's log into an ordered summary. Kept separate so a test can feed it
# a synthetic log without starting anything.
function Get-TitleSmokeSummary {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [int]$ExitCode = 0
    )
    $lines = @($Text -split "`r?`n")
    $frames = $null
    $fps = New-Object System.Collections.Generic.List[double]
    $submits = $null
    $watchMs = $null
    $dirtyPages = $null
    $lockWaitMs = $null
    $chunkScanMs = $null
    $uploadMs = $null
    $uploadRuns = $null
    $draws = $null
    $drawsSkipped = $null
    $drawsPc = $null
    $dispatches = $null
    $dispatchesSkipped = $null

    foreach ($line in $lines) {
        if ($line -match '(\d+) frames presented, ([\d.]+) fps over the last 300') {
            $frames = [int]$Matches[1]
            $fps.Add([double]$Matches[2])
            continue
        }
        if ($line -match 'sparse windows timing: write-watch (\d+) ms \((\d+) dirty pages\), lock wait (\d+) ms, chunk scan (\d+) ms, upload (\d+) ms \((\d+) runs\)') {
            $watchMs = [int]$Matches[1]
            $dirtyPages = [int]$Matches[2]
            $lockWaitMs = [int]$Matches[3]
            $chunkScanMs = [int]$Matches[4]
            $uploadMs = [int]$Matches[5]
            $uploadRuns = [int]$Matches[6]
            continue
        }
        if ($line -match 'sparse windows: (\d+) submits') {
            $submits = [int]$Matches[1]
            continue
        }
        if ($line -match 'console code so far: (\d+) draws with translated pipelines \((\d+) skipped, (\d+) with PC pipelines\), (\d+) dispatches \((\d+) skipped\)') {
            $draws = [int]$Matches[1]
            $drawsSkipped = [int]$Matches[2]
            $drawsPc = [int]$Matches[3]
            $dispatches = [int]$Matches[4]
            $dispatchesSkipped = [int]$Matches[5]
            continue
        }
    }

    $recent = @($fps | Select-Object -Last 3)
    $fpsAverage = $null
    if ($recent.Count -gt 0) {
        $fpsAverage = [math]::Round((($recent | Measure-Object -Sum).Sum / $recent.Count), 1)
    }
    $watchPerSubmit = $null
    if ($null -ne $submits -and $submits -gt 0 -and $null -ne $watchMs) {
        $watchPerSubmit = [math]::Round($watchMs / $submits, 3)
    }

    $countMatches = {
        param([string]$Pattern)
        $n = 0
        foreach ($line in $lines) { if ($line -match $Pattern) { $n++ } }
        $n
    }

    [ordered]@{
        exit_code = $ExitCode
        frames_presented = $frames
        fps_recent = $recent
        fps_average = $fpsAverage
        sparse_submits = $submits
        write_watch_ms = $watchMs
        dirty_pages = $dirtyPages
        write_watch_ms_per_submit = $watchPerSubmit
        lock_wait_ms = $lockWaitMs
        chunk_scan_ms = $chunkScanMs
        upload_ms = $uploadMs
        upload_runs = $uploadRuns
        draws = $draws
        draws_skipped = $drawsSkipped
        draws_pc = $drawsPc
        dispatches = $dispatches
        dispatches_skipped = $dispatchesSkipped
        device_removal_lines = (& $countMatches '0x887A')
        panic_lines = (& $countMatches 'panic')
        exception_lines = (& $countMatches 'exception')
        crash_lines = (& $countMatches 'crash')
        xerrorreport_lines = (& $countMatches 'XErrorReport')
    }
}

# One "key: value" line per summary entry, arrays joined by commas. No log text is included.
function Format-TitleSmokeSummary {
    param([Parameter(Mandatory = $true)]$Summary)
    foreach ($key in $Summary.Keys) {
        $value = $Summary[$key]
        if ($null -eq $value) {
            "${key}: n/a"
        } elseif ($value -is [array]) {
            if ($value.Count -eq 0) { "${key}: n/a" } else { "${key}: " + (($value | ForEach-Object { [string]$_ }) -join ', ') }
        } else {
            "${key}: $value"
        }
    }
}

if (-not (Test-Path -LiteralPath $Exe)) { throw "bathyal.exe not found: $Exe (cargo build --release --workspace)" }
if (-not (Test-Path -LiteralPath $Game)) { throw "game folder not found: $Game" }
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$settings = [ordered]@{}
if ($Seconds -gt 0) { $settings['XCL_RUN_SECONDS'] = "$Seconds" }
if ($PressAAt -gt 0) {
    $settings['XCL_PRESS_A_AT'] = $PressAAt.ToString([Globalization.CultureInfo]::InvariantCulture)
}
foreach ($key in $Env.Keys) { $settings[[string]$key] = [string]$Env[$key] }

$saved = @{}
foreach ($key in $settings.Keys) {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
}

$safeLabel = ''
if ($Label) { $safeLabel = '-' + ($Label -replace '[^A-Za-z0-9._-]', '_') }
$log = Join-Path $Logs ("smoke-{0}{1}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $safeLabel)

try {
    "running $Game; log $log"
    & $Exe run $Game 2>&1 | Tee-Object -FilePath $log -Encoding utf8
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
} finally {
    foreach ($key in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process')
    }
}

$text = Get-Content -LiteralPath $log -Raw
if ($null -eq $text) { $text = '' }
$summary = Get-TitleSmokeSummary -Text $text -ExitCode $code

$summaryPath = Join-Path (Split-Path -Parent $log) 'summary.json'
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding utf8

Format-TitleSmokeSummary -Summary $summary | Write-Output
"log: $log"
"summary_json: $summaryPath"
exit $code
