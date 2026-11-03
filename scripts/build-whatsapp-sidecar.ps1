<#
.SYNOPSIS
Builds the WhatsApp component (sidecars/whatsapp) for Windows.

.DESCRIPTION
The unofficial WhatsApp client lives in its own crate because it needs a newer
Rust than the agent and must exist only on computers that opt in. It is not
part of the installer: the app downloads it on request from the release, and
package-release.ps1 publishes it as POK-Agent-whatsapp-windows-x64.exe.

rustup does not install a toolchain from rust-toolchain.toml by itself and
`--manifest-path` does not change which toolchain applies, so the toolchain is
installed and selected explicitly here.

Writes sidecars/whatsapp/target/release/pok-ai-whatsapp.exe and prints its path.

.EXAMPLE
.\scripts\build-whatsapp-sidecar.ps1
#>
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $ProjectDir "sidecars/whatsapp/Cargo.toml"

# The version pinned in the crate's own rust-toolchain.toml.
$Toolchain = (Select-String -Path (Join-Path $ProjectDir "sidecars/whatsapp/rust-toolchain.toml") -Pattern 'channel\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
if (-not $Toolchain) { throw "Could not read the toolchain from sidecars/whatsapp/rust-toolchain.toml." }

& rustup toolchain install $Toolchain --profile minimal
if ($LASTEXITCODE -ne 0) { throw "Could not install Rust $Toolchain." }

& cargo "+$Toolchain" build --release --locked --manifest-path $Manifest
if ($LASTEXITCODE -ne 0) { throw "The WhatsApp component failed to build." }

$Exe = Join-Path $ProjectDir "sidecars/whatsapp/target/release/pok-ai-whatsapp.exe"
if (-not (Test-Path $Exe)) { throw "Expected $Exe after the build." }
Write-Output $Exe
