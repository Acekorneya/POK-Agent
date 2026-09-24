$ErrorActionPreference = "Stop"

function Assert-NativeSuccess([string]$CommandName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$CommandName failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget is required. Install or update App Installer from Microsoft Store."
}

if (-not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe")) {
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
}
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    winget install --id Rustlang.Rustup -e
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    winget install --id OpenJS.NodeJS.LTS -e
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
}
winget install --id Microsoft.EdgeWebView2Runtime -e --accept-package-agreements --accept-source-agreements

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $VsWhere) {
    $VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($VsPath) {
        Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
        Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
    }
}

rustup toolchain install 1.88.0-x86_64-pc-windows-msvc --profile minimal
Assert-NativeSuccess "rustup toolchain install"
rustup component add --toolchain 1.88.0-x86_64-pc-windows-msvc rustfmt clippy
Assert-NativeSuccess "rustup component add"
$ProjectDir = Split-Path -Parent $PSScriptRoot
cargo fetch --manifest-path (Join-Path $ProjectDir "Cargo.toml")
Assert-NativeSuccess "cargo fetch"
$DesktopDir = Join-Path $ProjectDir "apps/desktop"
& (Join-Path $PSScriptRoot "ensure-windows-frontend.ps1") -DesktopDir $DesktopDir
Write-Host "Windows environment ready. Run scripts/check-windows.ps1 next."
