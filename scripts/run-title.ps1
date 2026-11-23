# Runs a title with the release build in a visible window and keeps a log of each run.
# Before the first run it fills XCL's shader caches, as HANDOFF.md advises for large titles: compute programs
# first (`--precompile --only cs`), then the compute pipelines (`--precompile --pipelines`). A marker file in the
# log folder records that this was done for this bathyal.exe; -Precompile forces it again.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-title.ps1 -Game D:\games\title
#   powershell -ExecutionPolicy Bypass -File scripts\run-title.ps1 -Game D:\games\title -Seconds 120
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Xcl = (Join-Path (Split-Path -Parent $PSScriptRoot) 'target\release\bathyal.exe'),
    [string]$Logs = (Join-Path $env:LOCALAPPDATA 'bathyal\runs'),
    [int]$Seconds = 0,
    [switch]$Precompile
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Xcl)) { throw "bathyal.exe not found: $Xcl (cargo build --release --workspace)" }
$shaders = Get-ChildItem -LiteralPath $Game -Filter 'shaders.rpf' -Recurse -Depth 2 -ErrorAction SilentlyContinue |
    Select-Object -First 1
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$stamp = (Get-FileHash -LiteralPath $Xcl -Algorithm SHA256).Hash.Substring(0, 16)
$marker = Join-Path $Logs "precompiled-$stamp.txt"
if ($shaders -and ($Precompile -or -not (Test-Path -LiteralPath $marker))) {
    "precompiling $($shaders.FullName) (compute programs, then pipelines); this can take a long time once"
    & $Xcl shaders $shaders.FullName --precompile --only cs
    if ($LASTEXITCODE -ne 0) { throw "precompile --only cs exited $LASTEXITCODE" }
    & $Xcl shaders $shaders.FullName --precompile --pipelines
    if ($LASTEXITCODE -ne 0) { throw "precompile --pipelines exited $LASTEXITCODE" }
    Get-Date -Format o | Out-File -LiteralPath $marker -Encoding utf8
}

$log = Join-Path $Logs ("run-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
if ($Seconds -gt 0) { $env:XCL_RUN_SECONDS = "$Seconds" }
"running $Game; log $log"
& $Xcl run $Game 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
"bathyal exited with code $code; the run's report.json is written as XCL describes in README.md"
exit $code
