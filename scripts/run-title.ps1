# Runs a title with the release build in a visible window and keeps a log of each run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-title.ps1 -Game D:\games\title
#   powershell -ExecutionPolicy Bypass -File scripts\run-title.ps1 -Game D:\games\title -Seconds 120
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Xcl = (Join-Path (Split-Path -Parent $PSScriptRoot) 'target\release\bathyal.exe'),
    [string]$Logs = (Join-Path $env:LOCALAPPDATA 'bathyal\runs'),
    [int]$Seconds = 0
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Xcl)) { throw "bathyal.exe not found: $Xcl (cargo build --release --workspace)" }
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$log = Join-Path $Logs ("run-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
if ($Seconds -gt 0) { $env:XCL_RUN_SECONDS = "$Seconds" }
"running $Game; log $log"
& $Xcl run $Game 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
"bathyal exited with code $code; the run's report.json is written as XCL describes in README.md"
exit $code
