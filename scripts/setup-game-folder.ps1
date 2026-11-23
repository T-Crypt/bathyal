# Builds a local game folder for `xcl run` without copying the game's content.
#
# A packaged title often arrives as a read-only folder (a NAS share, an unpacked package) with the executable
# stored somewhere else. `xcl run` canonicalizes the executable and takes the folder that holds
# MicrosoftGame.Config as the game root, so the executable and the config have to be real files side by side.
# This script makes -Dest hold:
#   - a copy of the executable (-Exe),
#   - copies of the small top-level metadata files (MicrosoftGame.Config, microsoft_gameconfig.bin,
#     appxmanifest.xml) when the source has them,
#   - one symbolic link per other top-level entry of -Source, so archives stay where they are.
# Nothing is written into -Source. XCL sends the title's writes to its overlay, not to the game folder.
#
# Symbolic links need Developer Mode or an elevated shell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup-game-folder.ps1 -Source \\nas\games\Title `
#       -Exe D:\quarantine\verified\title.exe -Dest D:\games\title
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Dest
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Source folder not found: $Source" }
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "Executable not found: $Exe" }
$config = Join-Path $Source 'MicrosoftGame.Config'
if (-not (Test-Path -LiteralPath $config)) { throw "No MicrosoftGame.Config in $Source" }

# The executable's name must be the one MicrosoftGame.Config names, or `xcl run` starts nothing.
$text = Get-Content -LiteralPath $config -Raw
$named = [regex]::Matches($text, '<Executable\s+Name="([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
$exeName = Split-Path -Leaf $Exe
if ($named -and ($named -notcontains $exeName)) {
    throw "MicrosoftGame.Config names $($named -join ', '), not $exeName"
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$copied = 0; $linked = 0; $kept = 0

$exeTarget = Join-Path $Dest $exeName
if (-not (Test-Path -LiteralPath $exeTarget)) { Copy-Item -LiteralPath $Exe -Destination $exeTarget; $copied++ } else { $kept++ }

$metadata = @('MicrosoftGame.Config', 'microsoft_gameconfig.bin', 'appxmanifest.xml')
foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
    $target = Join-Path $Dest $item.Name
    if (Test-Path -LiteralPath $target) { $kept++; continue }
    if ($metadata -contains $item.Name) {
        Copy-Item -LiteralPath $item.FullName -Destination $target; $copied++
    } else {
        New-Item -ItemType SymbolicLink -Path $target -Target $item.FullName | Out-Null; $linked++
    }
}

$src = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash
$dst = (Get-FileHash -LiteralPath $exeTarget -Algorithm SHA256).Hash
if ($src -ne $dst) { throw "Executable copy differs from $Exe" }
"game folder: $Dest"
"copied $copied, linked $linked, already present $kept; executable SHA-256 $dst"
