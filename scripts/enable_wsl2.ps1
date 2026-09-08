[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script must be run from an elevated Administrator PowerShell session."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$statusDirectory = Join-Path $repoRoot "outputs\setup"
$statusPath = Join-Path $statusDirectory "wsl2-status.json"
New-Item -ItemType Directory -Path $statusDirectory -Force | Out-Null

$featureNames = @(
    "Microsoft-Windows-Subsystem-Linux",
    "VirtualMachinePlatform"
)
$featureResults = @()
$restartRequired = $false

foreach ($featureName in $featureNames) {
    $before = (Get-WindowsOptionalFeature -Online -FeatureName $featureName).State.ToString()
    $enableResult = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
    $after = (Get-WindowsOptionalFeature -Online -FeatureName $featureName).State.ToString()
    if ($enableResult.RestartNeeded) {
        $restartRequired = $true
    }
    $featureResults += [ordered]@{
        name = $featureName
        state_before = $before
        state_after = $after
        restart_needed = [bool]$enableResult.RestartNeeded
    }
}

$wslCommands = @()
if (-not $restartRequired) {
    foreach ($arguments in @(
        @("--update", "--web-download"),
        @("--set-default-version", "2")
    )) {
        $output = & "$env:WINDIR\System32\wsl.exe" @arguments 2>&1 | Out-String
        $wslCommands += [ordered]@{
            arguments = $arguments
            exit_code = $LASTEXITCODE
            output = $output.Trim()
        }
    }
}

$status = [ordered]@{
    schema_version = 1
    completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    features = $featureResults
    restart_required = $restartRequired
    wsl_commands = $wslCommands
}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8

Write-Host "WSL 2 prerequisite status written to $statusPath"
if ($restartRequired) {
    Write-Host "A Windows restart is required before WSL 2 and Docker Desktop can be started."
}
