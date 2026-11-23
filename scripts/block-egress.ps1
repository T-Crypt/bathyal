# Adds a Windows Firewall rule that blocks all outbound traffic of one program (default: the release xcl.exe).
# A title's network calls go straight to the host's network stack, so a first run of an untrusted executable
# should have no egress. Needs an elevated shell. Running it again keeps the existing rule.
#
#   powershell -ExecutionPolicy Bypass -File scripts\block-egress.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\block-egress.ps1 -Program D:\other\bathyal.exe
param(
    [string]$Program = (Join-Path (Split-Path -Parent $PSScriptRoot) 'target\release\bathyal.exe'),
    [string]$Name = ''
)
$ErrorActionPreference = 'Stop'
$Program = [System.IO.Path]::GetFullPath($Program)
if (-not $Name) { $Name = "bathyal egress block: $Program" }

$existing = Get-NetFirewallRule -DisplayName $Name -ErrorAction SilentlyContinue
if (-not $existing) {
    New-NetFirewallRule -DisplayName $Name -Direction Outbound -Action Block -Program $Program -Profile Any | Out-Null
}
$rule = Get-NetFirewallRule -DisplayName $Name
$app = $rule | Get-NetFirewallApplicationFilter
"$($rule.DisplayName): Enabled=$($rule.Enabled) Direction=$($rule.Direction) Action=$($rule.Action) Program=$($app.Program)"
