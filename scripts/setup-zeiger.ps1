param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$Python = "python",
    # HF repo id of the judge checkpoint to install. php-ai/zeiger-0.6b is the
    # default: offline+live testing (docs/decision-router-backends.md) found
    # it the most reliable cross-model judge of everything benchmarked --
    # 91.7% correlation judging Laya's picks, beating JEV's own self-judgment
    # (70%, including confidently re-endorsed wrong picks) at zero cost and
    # a fraction of the latency.
    [string]$Repo = "php-ai/zeiger-0.6b",
    # HF revision of the checkpoint to install. MUST be pinned: the Hub's
    # `main` was updated to "Round 15" on 2026-09-21 and that revision
    # answers the reversible/cheap/evidenced safety questions with
    # "unknown" on nearly every input (40% judge correlation, zero live
    # promotions), while the pre-Round-15 weights reproduce the validated
    # behavior (80-85% correlation). See docs/decision-router-backends.md.
    [string]$Revision = "441dcf64aef0",
    # git remote for the zeiger inference package itself: it is not published
    # to PyPI (see scripts/zeiger_sidecar.py's docstring), so it is cloned
    # onto PYTHONPATH the same way this repo's own README documents doing it
    # manually.
    [string]$PackageRepo = "https://github.com/PurHur/zeiger.git"
)

$ErrorActionPreference = "Stop"

$Root = Join-Path $DataDir "judge"
$Venv = Join-Path $Root "runtime"
$Model = Join-Path $Root "models\zeiger-0.6b"
$PackageDir = Join-Path $Root "zeiger"
New-Item -ItemType Directory -Force -Path $Root, $Model | Out-Null

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required for the initial judge-model bootstrap. Install Python, then retry."
}

if (-not (Test-Path (Join-Path $PackageDir "zeiger\__init__.py"))) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is required to fetch the zeiger inference package (it is not published to PyPI). Install git, then retry."
    }
    if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
    git clone --depth 1 $PackageRepo $PackageDir
    if ($LASTEXITCODE -ne 0) { throw "Could not clone the zeiger package from $PackageRepo." }
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the managed judge-model Python environment." }
}

$ManagedPython = Join-Path $Venv "Scripts\python.exe"
& $ManagedPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the managed judge-model environment." }
$HasGpu = [bool](
    (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or
    (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) -or
    (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" })
)

if ($HasGpu) {
    & $ManagedPython -m pip install --disable-pip-version-check --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "Could not install the CUDA-enabled PyTorch runtime." }
} else {
    & $ManagedPython -m pip install --disable-pip-version-check torch
    if ($LASTEXITCODE -ne 0) { throw "Could not install the PyTorch runtime." }
}
# zeiger pins transformers below 5: its engine registers a custom attention
# kernel through the 4.x AttentionInterface, and 5.x resolves unknown
# implementation names as flash-attn hub kernels instead (see
# scripts/zeiger_sidecar.py's docstring / the package's own requirements.txt).
& $ManagedPython -m pip install --disable-pip-version-check "transformers>=4.44,<5" "safetensors>=0.4" "numpy>=1.24" "fastapi==0.116.1" "uvicorn==0.35.0"
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned judge-model sidecar dependencies." }

$Download = @'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[1],
    allow_patterns=["model.safetensors", "rl_agent_config.json", "tokenizer/*", "encoder/*"],
)
'@
$Download | & $ManagedPython - $Model $Repo $Revision
if ($LASTEXITCODE -ne 0) { throw "Could not download the $Repo checkpoint (revision $Revision)." }

$Manifest = @{
    package = "zeiger"
    package_repo = $PackageRepo
    checkpoint = $Repo
    revision = $Revision
    installed_at = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json
Set-Content -Encoding UTF8 -Path (Join-Path $Root "manifest.json") -Value $Manifest
Write-Output $Manifest
