# Builds the release pok-ai CLI for the arena VM into its own target folder,
# so it never collides with a running desktop build. Prints the exe path last.
param([string]$TargetDir = "$env:LOCALAPPDATA\POK-Ai\arena-target")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null

$env:CARGO_TARGET_DIR = $TargetDir
$ErrorActionPreference = "Continue"
cmd /c "cargo build --release --manifest-path `"$repo\Cargo.toml`" -p pok-ai-cli 2>&1" | Select-Object -Last 5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Join-Path $TargetDir "release\pok-ai.exe"
