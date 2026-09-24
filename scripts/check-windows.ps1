param(
    [switch]$SkipFormat
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot

function Assert-NativeSuccess([string]$CommandName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$CommandName failed with exit code $LASTEXITCODE"
    }
}

$DesktopDir = Join-Path $ProjectDir "apps/desktop"
& (Join-Path $PSScriptRoot "ensure-windows-frontend.ps1") -DesktopDir $DesktopDir

if (-not $SkipFormat) {
    cargo fmt --manifest-path (Join-Path $ProjectDir "Cargo.toml") --all -- --check
    Assert-NativeSuccess "cargo fmt"
}
cargo test --manifest-path (Join-Path $ProjectDir "Cargo.toml") --workspace
Assert-NativeSuccess "cargo test"
Push-Location $DesktopDir
try {
    npm run build
    Assert-NativeSuccess "npm run build"
}
finally {
    Pop-Location
}
cargo check --manifest-path (Join-Path $ProjectDir "Cargo.toml") -p pok-ai-desktop
Assert-NativeSuccess "cargo check -p pok-ai-desktop"

Write-Host "All native Windows checks passed."
