<#
.SYNOPSIS
Packages a built release into the files published on GitHub.

.DESCRIPTION
Expects `npm run tauri -- build` to have produced target/release/pok-ai-desktop.exe
and the NSIS installer. Writes to target/release/publish/v<version>:

  POK-Agent-windows-x64.exe        standalone executable
  POK-Agent-windows-x64-setup.exe  NSIS installer
  POK-Agent-windows-x64.zip        portable ZIP (exe, example config, README, LICENSE, NOTICE)
  SHA256SUMS.txt                checksums of the three files above

Prints the output folder. Used by run-windows.ps1 -Release and the release workflow.

.EXAMPLE
.\scripts\package-release.ps1 -Version 0.2.0
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$TargetDir = Join-Path $ProjectDir "target"
$ReleaseExe = Join-Path $TargetDir "release/pok-ai-desktop.exe"
$NsisDir = Join-Path $TargetDir "release/bundle/nsis"

if (-not (Test-Path $ReleaseExe)) {
    throw "Release executable not found at $ReleaseExe. Build first with: npm --prefix apps/desktop run tauri -- build"
}
$NsisInstallers = @(
    Get-ChildItem -Path $NsisDir -Filter "*-setup.exe" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*$Version*" }
)
if ($NsisInstallers.Count -ne 1) {
    throw "Expected exactly one NSIS installer for version $Version in $NsisDir, but found $($NsisInstallers.Count)."
}

$PublishDir = Join-Path $TargetDir "release/publish/v$Version"
$PortableDir = Join-Path $PublishDir "portable"
if (Test-Path $PublishDir) {
    Remove-Item -Path $PublishDir -Recurse -Force
}
New-Item -ItemType Directory -Path $PortableDir -Force | Out-Null

$ExecutableAsset = Join-Path $PublishDir "POK-Agent-windows-x64.exe"
$InstallerAsset = Join-Path $PublishDir "POK-Agent-windows-x64-setup.exe"
$ZipAsset = Join-Path $PublishDir "POK-Agent-windows-x64.zip"
$ChecksumAsset = Join-Path $PublishDir "SHA256SUMS.txt"

Copy-Item $ReleaseExe $ExecutableAsset -Force
Copy-Item $NsisInstallers[0].FullName $InstallerAsset -Force
Copy-Item $ReleaseExe (Join-Path $PortableDir "POK-Agent.exe") -Force
foreach ($File in @("LICENSE", "NOTICE", "README.md", "pok-ai.example.toml")) {
    Copy-Item (Join-Path $ProjectDir $File) (Join-Path $PortableDir $File) -Force
}
Compress-Archive -Path (Join-Path $PortableDir "*") -DestinationPath $ZipAsset -CompressionLevel Optimal -Force
Remove-Item -Path $PortableDir -Recurse -Force

$ChecksumLines = foreach ($Asset in @($ExecutableAsset, $InstallerAsset, $ZipAsset)) {
    $Hash = (Get-FileHash -Path $Asset -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $([System.IO.Path]::GetFileName($Asset))"
}
Set-Content -Path $ChecksumAsset -Value $ChecksumLines -Encoding ascii

Write-Output $PublishDir
