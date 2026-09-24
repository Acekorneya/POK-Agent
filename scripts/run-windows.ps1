<#
.SYNOPSIS
Builds and runs POK-Ai in either development or shipped-user mode.

.EXAMPLE
.\scripts\run-windows.ps1 -Mode Dev

.EXAMPLE
.\scripts\run-windows.ps1 -Mode Shipped

.EXAMPLE
.\scripts\run-windows.ps1 -Mode Shipped -SkipChecks -NoLaunch

.EXAMPLE
.\scripts\run-windows.ps1 -Mode Shipped -Release

.EXAMPLE
.\scripts\run-windows.ps1 -Mode Dev -Clean
#>
param(
    [ValidateSet("Dev", "Shipped")]
    [string]$Mode = "Dev",
    [switch]$SkipChecks,
    [switch]$NoLaunch,
    [switch]$Clean,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$DesktopDir = Join-Path $ProjectDir "apps/desktop"
$TargetDir = Join-Path $ProjectDir "target"
$ReleaseExe = Join-Path $TargetDir "release/pok-ai-desktop.exe"
$BuildProfileStamp = Join-Path $TargetDir ".pok-build-profile"
$BuildProfileVersion = "context-generated-tools-v2"
$script:ReleaseMetadata = $null

if ($Release -and $Mode -ne "Shipped") {
    throw "-Release requires -Mode Shipped."
}

function Assert-NativeSuccess([string]$CommandName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$CommandName failed with exit code $LASTEXITCODE"
    }
}

function Get-AvailablePort([int]$StartPort = 1430) {
    for ($port = $StartPort; $port -lt ($StartPort + 100); $port++) {
        $inUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if (-not $inUse) {
            try {
                $listenerV4 = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
                $listenerV4.Start()
                $listenerV4.Stop()

                $listenerV6 = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Loopback, $port)
                $listenerV6.Start()
                $listenerV6.Stop()

                return $port
            }
            catch {
                # Port occupied or cannot bind
            }
        }
    }
    return $StartPort
}

function Enter-PokBuildEnvironment {
    foreach ($Command in @("cargo", "rustup", "node", "npm")) {
        if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
            throw "$Command is required. Run .\scripts\setup-windows.ps1 first."
        }
    }

    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $VsWhere) {
        $VsPath = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($VsPath) {
            Import-Module (Join-Path $VsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
            Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
        }
    }

    & (Join-Path $PSScriptRoot "ensure-windows-frontend.ps1") -DesktopDir $DesktopDir
}

function Get-PokVersion {
    $TauriConfigPath = Join-Path $DesktopDir "src-tauri/tauri.conf.json"
    $TauriConfig = Get-Content $TauriConfigPath -Raw | ConvertFrom-Json
    $TauriVersion = [string]$TauriConfig.version

    $WorkspaceManifest = Get-Content (Join-Path $ProjectDir "Cargo.toml")
    $VersionLine = $WorkspaceManifest | Select-String -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    if (-not $VersionLine) {
        throw "Could not read workspace.package.version from Cargo.toml."
    }
    $WorkspaceVersion = $VersionLine.Matches[0].Groups[1].Value

    if ($TauriVersion -ne $WorkspaceVersion) {
        throw "Release versions do not match: tauri.conf.json is $TauriVersion but Cargo.toml is $WorkspaceVersion."
    }
    if ($TauriVersion -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
        throw "Release version '$TauriVersion' is not a valid release version."
    }

    return $TauriVersion
}

function Assert-GitHubReleaseReady {
    foreach ($Command in @("git", "gh")) {
        if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
            throw "$Command is required for -Release. Install GitHub CLI and run 'gh auth login' before publishing."
        }
    }

    Push-Location $ProjectDir
    try {
        git rev-parse --is-inside-work-tree *> $null
        Assert-NativeSuccess "git rev-parse"

        $Changes = @(git status --porcelain)
        Assert-NativeSuccess "git status"
        if ($Changes.Count -gt 0) {
            throw "The working tree has uncommitted changes. Commit them before publishing a release."
        }

        gh auth status --active --hostname github.com *> $null
        Assert-NativeSuccess "gh auth status"

        $Repository = (& gh repo view --json nameWithOwner --jq '.nameWithOwner').Trim()
        Assert-NativeSuccess "gh repo view"
        if (-not $Repository) {
            throw "GitHub CLI could not determine the repository. Configure a GitHub remote first."
        }

        $Upstream = (& git rev-parse --abbrev-ref --symbolic-full-name '@{u}').Trim()
        Assert-NativeSuccess "git upstream lookup"
        git fetch --quiet
        Assert-NativeSuccess "git fetch"

        $HeadCommit = (& git rev-parse HEAD).Trim()
        Assert-NativeSuccess "git rev-parse HEAD"
        $UpstreamCommit = (& git rev-parse '@{u}').Trim()
        Assert-NativeSuccess "git rev-parse upstream"
        if ($HeadCommit -ne $UpstreamCommit) {
            throw "HEAD is not identical to its fetched upstream $Upstream. Push the release commit (and resolve any remote changes) first."
        }

        $Version = Get-PokVersion
        $Tag = "v$Version"
        & gh release view $Tag --repo $Repository *> $null
        if ($LASTEXITCODE -eq 0) {
            throw "GitHub release $Tag already exists. Bump the version before publishing again."
        }

        $ExistingTag = @(git ls-remote --tags "https://github.com/$Repository.git" "refs/tags/$Tag")
        Assert-NativeSuccess "git ls-remote"
        if ($ExistingTag.Count -gt 0) {
            throw "Git tag $Tag already exists in $Repository. Bump the version or publish that tag manually."
        }

        return [PSCustomObject]@{
            Version = $Version
            Tag = $Tag
            Repository = $Repository
            HeadCommit = $HeadCommit
        }
    }
    finally {
        Pop-Location
    }
}

function Publish-GitHubRelease($Metadata) {
    $BundleDir = Join-Path $TargetDir "release/bundle"
    $NsisDir = Join-Path $BundleDir "nsis"
    $NsisInstallers = @(
        Get-ChildItem -Path $NsisDir -Filter "*-setup.exe" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "*$($Metadata.Version)*" }
    )
    if ($NsisInstallers.Count -ne 1) {
        throw "Expected exactly one NSIS installer for version $($Metadata.Version) in $NsisDir, but found $($NsisInstallers.Count)."
    }

    $PublishDir = Join-Path $TargetDir "release/publish/$($Metadata.Tag)"
    $PortableDir = Join-Path $PublishDir "portable"
    if (Test-Path $PublishDir) {
        Remove-Item -Path $PublishDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PortableDir -Force | Out-Null

    $ExecutableAsset = Join-Path $PublishDir "POK-Ai-windows-x64.exe"
    $InstallerAsset = Join-Path $PublishDir "POK-Ai-windows-x64-setup.exe"
    $ZipAsset = Join-Path $PublishDir "POK-Ai-windows-x64.zip"
    $ChecksumAsset = Join-Path $PublishDir "SHA256SUMS.txt"

    Copy-Item $ReleaseExe $ExecutableAsset -Force
    Copy-Item $NsisInstallers[0].FullName $InstallerAsset -Force
    Copy-Item $ReleaseExe (Join-Path $PortableDir "POK-Ai.exe") -Force
    foreach ($File in @("LICENSE", "NOTICE", "README.md", "pok-ai.example.toml")) {
        Copy-Item (Join-Path $ProjectDir $File) (Join-Path $PortableDir $File) -Force
    }
    Compress-Archive -Path (Join-Path $PortableDir "*") -DestinationPath $ZipAsset -CompressionLevel Optimal -Force

    $ChecksumLines = foreach ($Asset in @($ExecutableAsset, $InstallerAsset, $ZipAsset)) {
        $Hash = (Get-FileHash -Path $Asset -Algorithm SHA256).Hash.ToLowerInvariant()
        "$Hash  $([System.IO.Path]::GetFileName($Asset))"
    }
    Set-Content -Path $ChecksumAsset -Value $ChecksumLines -Encoding ascii

    Write-Host "Publishing $($Metadata.Tag) to $($Metadata.Repository)..." -ForegroundColor Cyan
    & gh release create $Metadata.Tag $ExecutableAsset $InstallerAsset $ZipAsset $ChecksumAsset `
        --repo $Metadata.Repository `
        --target $Metadata.HeadCommit `
        --title "POK-Ai $($Metadata.Tag)" `
        --generate-notes `
        --latest `
        --fail-on-no-commits
    Assert-NativeSuccess "gh release create"

    $ReleaseUrl = (& gh release view $Metadata.Tag --repo $Metadata.Repository --json url --jq '.url').Trim()
    Assert-NativeSuccess "gh release view"
    Write-Host "Published release: $ReleaseUrl" -ForegroundColor Green
}

function Initialize-BuildProfileCache {
    $PreviousVersion = ""
    if (Test-Path $BuildProfileStamp) {
        $PreviousVersion = (Get-Content $BuildProfileStamp -Raw).Trim()
    }

    $HasBuildArtifacts = Test-Path (Join-Path $TargetDir "debug")
    $ProfileChanged = $HasBuildArtifacts -and ($PreviousVersion -ne $BuildProfileVersion)

    if ($Clean -or $ProfileChanged) {
        if ($ProfileChanged -and -not $Clean) {
            Write-Host "The Rust development profile changed; clearing incompatible Cargo artifacts once..." -ForegroundColor Yellow
        }
        else {
            Write-Host "Clearing generated Cargo artifacts..." -ForegroundColor Yellow
        }

        cargo clean --manifest-path (Join-Path $ProjectDir "Cargo.toml")
        Assert-NativeSuccess "cargo clean"
    }

    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    Set-Content -Path $BuildProfileStamp -Value $BuildProfileVersion -NoNewline
}

function Stop-RunningReleaseExecutable {
    if (-not (Test-Path $ReleaseExe)) {
        return
    }

    $ExpectedPath = [System.IO.Path]::GetFullPath($ReleaseExe)
    $RunningReleaseProcesses = @(
        Get-Process -Name "pok-ai-desktop" -ErrorAction SilentlyContinue | Where-Object {
            try {
                [string]::Equals(
                    [System.IO.Path]::GetFullPath($_.Path),
                    $ExpectedPath,
                    [System.StringComparison]::OrdinalIgnoreCase
                )
            }
            catch {
                $false
            }
        }
    )

    if ($RunningReleaseProcesses.Count -gt 0) {
        Write-Host "Closing the running release app before rebuilding it..." -ForegroundColor Yellow
    }

    foreach ($Process in $RunningReleaseProcesses) {
        if ($Process.MainWindowHandle -ne 0) {
            $null = $Process.CloseMainWindow()
            Wait-Process -Id $Process.Id -Timeout 5 -ErrorAction SilentlyContinue
        }

        if (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue) {
            Stop-Process -Id $Process.Id -Force
            Wait-Process -Id $Process.Id -Timeout 5 -ErrorAction SilentlyContinue
        }
    }

    # Confirm that Rust/Tauri can replace the existing binary. This also gives
    # a useful error when a different process (for example, a security scanner)
    # owns the lock and cannot be identified safely by executable path.
    for ($Attempt = 0; $Attempt -lt 10; $Attempt++) {
        try {
            $Stream = [System.IO.File]::Open(
                $ReleaseExe,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            $Stream.Dispose()
            return
        }
        catch {
            if ($Attempt -lt 9) {
                Start-Sleep -Milliseconds 200
            }
        }
    }

    throw "Cannot rebuild $ReleaseExe because another process still has it open. Close POK-Ai and any program inspecting the executable, then retry."
}

Enter-PokBuildEnvironment
if ($Release) {
    $script:ReleaseMetadata = Assert-GitHubReleaseReady
    $NoLaunch = $true
}
Initialize-BuildProfileCache

if (-not $SkipChecks) {
    Write-Host "Running native Windows checks before launch..." -ForegroundColor Cyan
    # Runtime verification is intentionally independent of repository-wide
    # formatting state so this launcher can test work-in-progress changes.
    & (Join-Path $PSScriptRoot "check-windows.ps1") -SkipFormat
}

if ($Mode -eq "Dev") {
    $DevPort = Get-AvailablePort 1430
    if ($DevPort -ne 1430) {
        Write-Host "Port 1430 is currently occupied; automatically selected available port $DevPort." -ForegroundColor Yellow
    }
    Write-Host "Starting optimized development mode on port $DevPort with hot reload..." -ForegroundColor Green

    $TauriConfPath = Join-Path $DesktopDir "src-tauri/tauri.conf.json"
    $TauriConf = Get-Content $TauriConfPath -Raw | ConvertFrom-Json
    if ($TauriConf.build.devUrl -ne "http://localhost:$DevPort") {
        $TauriConf.build.devUrl = "http://localhost:$DevPort"
        $TauriConf | ConvertTo-Json -Depth 10 | Set-Content $TauriConfPath
    }

    $env:VITE_PORT = "$DevPort"
    Push-Location $DesktopDir
    try {
        npm run tauri -- dev
        Assert-NativeSuccess "tauri dev"
    }
    finally {
        $env:VITE_PORT = $null
        Pop-Location
    }
    exit 0
}

Stop-RunningReleaseExecutable
Write-Host "Building the release configuration users would receive..." -ForegroundColor Green
Push-Location $DesktopDir
try {
    npm run tauri -- build
    Assert-NativeSuccess "tauri build"
}
finally {
    Pop-Location
}

if (-not (Test-Path $ReleaseExe)) {
    throw "Release build succeeded but $ReleaseExe was not found."
}

Write-Host "Release executable: $ReleaseExe" -ForegroundColor Green
$BundleDir = Join-Path $ProjectDir "target/release/bundle"
if (Test-Path $BundleDir) {
    Write-Host "Installer bundles: $BundleDir" -ForegroundColor Green
}

if ($Release) {
    Publish-GitHubRelease $script:ReleaseMetadata
}

if (-not $NoLaunch) {
    Start-Process -FilePath $ReleaseExe -WorkingDirectory $ProjectDir
}
