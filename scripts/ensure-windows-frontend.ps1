param(
    [Parameter(Mandatory = $true)]
    [string]$DesktopDir
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is required. Run .\scripts\setup-windows.ps1 first."
}

$PackageLock = Join-Path $DesktopDir "package-lock.json"
if (-not (Test-Path $PackageLock)) {
    throw "Frontend lockfile was not found at $PackageLock."
}

Push-Location $DesktopDir
try {
    if (-not (Test-Path (Join-Path $DesktopDir "node_modules"))) {
        Write-Host "Installing locked frontend dependencies..." -ForegroundColor Cyan
        npm ci --include=optional
        if ($LASTEXITCODE -ne 0) {
            throw "npm ci failed with exit code $LASTEXITCODE"
        }
    }

    # node_modules is shared when this checkout is used from both WSL and
    # native Windows. npm can retain Linux optional packages while omitting
    # their Windows counterparts (npm/cli#4828).
    $NativePackages = @(
        "@esbuild/win32-x64",
        "@rollup/rollup-win32-x64-msvc",
        "@tauri-apps/cli-win32-x64-msvc"
    )
    foreach ($PackageName in $NativePackages) {
        $NativeManifest = Join-Path $DesktopDir "node_modules/$PackageName/package.json"
        if (Test-Path $NativeManifest) {
            continue
        }

        $LockKey = "node_modules/$PackageName"
        $VersionExpression = "require('./package-lock.json').packages['$LockKey'].version"
        $PackageVersion = (& node -p $VersionExpression).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $PackageVersion) {
            throw "Could not determine the locked version of $PackageName from $PackageLock."
        }

        Write-Host "Installing the Windows native package $PackageName@$PackageVersion..." -ForegroundColor Cyan
        $PackageSpec = "$PackageName@$PackageVersion"
        $TempDir = Join-Path ([System.IO.Path]::GetTempPath()) "pok-ai-native-package-$([System.Guid]::NewGuid().ToString('N'))"
        $NativePackageDir = Split-Path -Parent $NativeManifest
        New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
        New-Item -ItemType Directory -Path $NativePackageDir -Force | Out-Null
        try {
            # npm install can fail while resolving a node_modules tree created
            # on another OS. npm pack only downloads the exact locked package,
            # so extracting it avoids that cross-platform dependency-tree bug.
            $ArchiveName = (& npm pack --silent --pack-destination $TempDir $PackageSpec | Select-Object -Last 1).Trim()
            if ($LASTEXITCODE -ne 0 -or -not $ArchiveName) {
                throw "Downloading $PackageSpec failed with exit code $LASTEXITCODE"
            }
            $ArchivePath = Join-Path $TempDir $ArchiveName
            tar -xzf $ArchivePath -C $NativePackageDir --strip-components=1
            if ($LASTEXITCODE -ne 0) {
                throw "Extracting $PackageSpec failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        if (-not (Test-Path $NativeManifest)) {
            throw "$PackageName is still missing after npm repair. Remove apps/desktop/node_modules and rerun this command from native Windows PowerShell."
        }
    }
}
finally {
    Pop-Location
}
