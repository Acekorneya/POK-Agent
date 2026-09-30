<#
.SYNOPSIS
Exports a clean copy of POK-Agent source files and git repository to a destination folder,
excluding temporary build artifacts, compiler caches, and node_modules.

.PARAMETER Destination
The destination directory where the clean POK-Agent project should be copied.

.PARAMETER IncludeGit
Copies the .git directory preserving all git commit history, branches, and tags.
Defaults to $true. Use -IncludeGit:$false to exclude.

.PARAMETER IncludeDiagnostics
If specified, also copies all diagnostic sessions and traces. Defaults to $false.

.EXAMPLE
.\export-clean.ps1 -Destination "D:\Projects\POK-Agent"
Exports to D:\Projects\POK-Agent including git history.

.EXAMPLE
.\export-clean.ps1 -Destination "D:\Projects\POK-Agent" -IncludeGit:$false
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Destination,

    [switch]$IncludeGit = $true,

    [switch]$IncludeDiagnostics
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot

if (-not (Test-Path $ProjectDir)) {
    throw "Source directory not found: $ProjectDir"
}

$DestPath = [System.IO.Path]::GetFullPath($Destination)

# Confirm destination is not inside source
if ($DestPath.StartsWith($ProjectDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination cannot be inside the current project folder: $ProjectDir"
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " POK-Agent Clean Exporter" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Source:                       $ProjectDir" -ForegroundColor Gray
Write-Host "Destination:                  $DestPath" -ForegroundColor Gray
Write-Host "Include Git (.git):           $IncludeGit" -ForegroundColor Gray
Write-Host "Include Diagnostics Sessions: $IncludeDiagnostics" -ForegroundColor Gray
Write-Host ""

# Directories to exclude from copy
$ExcludeDirs = [System.Collections.Generic.List[string]]::new()
$ExcludeDirs.Add("target")
$ExcludeDirs.Add("target-*")
$ExcludeDirs.Add("node_modules")
$ExcludeDirs.Add("dist")
$ExcludeDirs.Add(".cargo-ok")

if (-not $IncludeGit) {
    $ExcludeDirs.Add(".git")
}

if (-not $IncludeDiagnostics) {
    $ExcludeDirs.Add("sessions")
}

# Files to exclude from copy
$ExcludeFiles = @(
    "*.db-shm",
    "*.db-wal",
    "*.log"
)

# Ensure destination exists
if (-not (Test-Path $DestPath)) {
    Write-Host "Creating destination directory: $DestPath" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
}

Write-Host "Copying repository files via Robocopy..." -ForegroundColor Cyan
if ($IncludeGit) {
    Write-Host "  -> Git repository (.git) is INCLUDED (preserving all history, branches, tags)" -ForegroundColor Green
} else {
    Write-Host "  -> Git repository (.git) is EXCLUDED" -ForegroundColor Yellow
}
Write-Host "  -> Excluding build artifacts ('target', 'node_modules', 'dist') to prevent copying corrupt or oversized caches" -ForegroundColor Gray

$RobocopyArgs = @(
    $ProjectDir,
    $DestPath,
    "/E",
    "/XD"
) + $ExcludeDirs + @(
    "/XF"
) + $ExcludeFiles + @(
    "/NJH",
    "/NJS",
    "/NDL",
    "/NP"
)

& robocopy @RobocopyArgs

# Robocopy exit codes: 0-7 are success codes (0 = no files copied, 1 = files copied, etc.), >= 8 is failure.
if ($LASTEXITCODE -ge 8) {
    throw "Robocopy failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Clean export completed successfully!" -ForegroundColor Green
Write-Host "Destination: $DestPath" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps on the SSD ($DestPath):" -ForegroundColor Cyan
Write-Host "  1. cd `"$DestPath`"" -ForegroundColor White
Write-Host "  2. git status (verify git history and branch)" -ForegroundColor White
Write-Host "  3. .\scripts\setup-windows.ps1 (if needed) or .\scripts\run-windows.ps1 -Mode Dev" -ForegroundColor White

