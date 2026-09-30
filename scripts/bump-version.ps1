<#
.SYNOPSIS
Sets the release version, commits it, and tags it for the release workflow.

.DESCRIPTION
Updates workspace.package.version in Cargo.toml and "version" in
apps/desktop/src-tauri/tauri.conf.json (the release checks they match),
refreshes Cargo.lock, commits "Release v<version>", and creates the tag
v<version>. Nothing is pushed; push when ready:

  git push origin main v<version>

Pushing the tag starts .github/workflows/release.yml, which builds the
installers and publishes the GitHub release.

.EXAMPLE
.\scripts\bump-version.ps1 -Version 0.2.0
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$')]
    [string]$Version
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$CargoToml = Join-Path $ProjectDir "Cargo.toml"
$TauriConf = Join-Path $ProjectDir "apps/desktop/src-tauri/tauri.conf.json"

function Assert-NativeSuccess([string]$CommandName) {
    if ($LASTEXITCODE -ne 0) { throw "$CommandName failed with exit code $LASTEXITCODE" }
}

Push-Location $ProjectDir
try {
    if (@(git status --porcelain).Count -gt 0) {
        throw "Commit or stash your changes first; the release commit should contain only the version bump."
    }
    if (@(git tag --list "v$Version").Count -gt 0) {
        throw "Tag v$Version already exists."
    }

    # First `version = "..."` in Cargo.toml is [workspace.package].
    $cargo = Get-Content $CargoToml -Raw
    $regex = [regex]'(?m)^version\s*=\s*"[^"]+"'
    if (-not $regex.IsMatch($cargo)) { throw "No version line found in Cargo.toml." }
    $cargo = $regex.Replace($cargo, "version = `"$Version`"", 1)
    Set-Content -Path $CargoToml -Value $cargo -NoNewline -Encoding utf8

    $tauri = Get-Content $TauriConf -Raw
    $tauriRegex = [regex]'"version"\s*:\s*"[^"]+"'
    if (-not $tauriRegex.IsMatch($tauri)) { throw "No version found in tauri.conf.json." }
    $tauri = $tauriRegex.Replace($tauri, "`"version`": `"$Version`"", 1)
    Set-Content -Path $TauriConf -Value $tauri -NoNewline -Encoding utf8

    cargo update --workspace --offline 2>$null
    if ($LASTEXITCODE -ne 0) { cargo update --workspace }
    Assert-NativeSuccess "cargo update --workspace"

    git add Cargo.toml Cargo.lock apps/desktop/src-tauri/tauri.conf.json
    Assert-NativeSuccess "git add"
    git commit -m "Release v$Version"
    Assert-NativeSuccess "git commit"
    git tag -a "v$Version" -m "POK-Ai v$Version"
    Assert-NativeSuccess "git tag"

    Write-Host "Tagged v$Version. Publish it with:" -ForegroundColor Green
    Write-Host "  git push origin main v$Version" -ForegroundColor Cyan
}
finally {
    Pop-Location
}
