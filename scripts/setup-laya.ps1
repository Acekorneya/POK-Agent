param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$Python = "python",
    # HF repo id of the checkpoint to install. Defaults to the Apache-2.0
    # typed-decisions specialist checkpoint (convaiinnovations/laya-typed-decisions):
    # on the 20-case harness suite (bench_decision_models.py) it is the best
    # local picker measured to date -- 75% raw two-stage pick accuracy vs.
    # 60% for Luni/laya-grounded and 55% for the ModernBERT "english" fallback
    # (2026-09-22 replay) -- and with the pinned zeiger judge it reaches the
    # 90% cross-model cascade with zero false positives. It is also
    # permissively licensed, so it is safe for a commercial product, unlike
    # Luni/laya-grounded (CC-BY-NC-4.0). Whichever repo is installed here
    # lands in the same fixed `models\typed-decisions` directory the app
    # expects (laya_checkpoint() in apps/desktop/src-tauri/src/lib.rs), so
    # only one checkpoint is resident at a time. Update
    # `decision_router.laya.model` in pok-ai.toml to match whatever is
    # installed.
    [string]$Repo = "convaiinnovations/laya-typed-decisions",
    # HF revision of the checkpoint to install. Pinned so a Hub-side update
    # cannot silently change picker behavior between installs (the zeiger
    # judge's Hub `main` was updated mid-day on 2026-09-21 and regressed from
    # 90% to 40% judge correlation -- see docs/decision-router-backends.md).
    # All revisions of this repo currently share the same weights; the pin is
    # the measured fine-tune commit. Pass "main" to follow the latest
    # revision explicitly.
    [string]$Revision = "843893f92cf9"
)

$ErrorActionPreference = "Stop"

$Root = Join-Path $DataDir "laya"
$Venv = Join-Path $Root "runtime"
$Model = Join-Path $Root "models\typed-decisions"
New-Item -ItemType Directory -Force -Path $Root, $Model | Out-Null

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ is required for the initial Laya bootstrap. Install Python, then retry."
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the managed Laya Python environment." }
}

$ManagedPython = Join-Path $Venv "Scripts\python.exe"
& $ManagedPython -m pip install --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the managed Laya environment." }
$HasGpu = [bool](
    (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or
    (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) -or
    (Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" })
)

if ($HasGpu) {
    & $ManagedPython -m pip install --disable-pip-version-check --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "Could not install the CUDA-enabled PyTorch runtime." }
}
& $ManagedPython -m pip install --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cu126 "laya==0.3.4" "fastapi==0.116.1" "uvicorn==0.35.0"
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned Laya sidecar dependencies." }

$Download = @'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[1],
    allow_patterns=["model.safetensors", "rl_agent_config.json", "tokenizer/*", "encoder/config.json"],
)
'@
$Download | & $ManagedPython - $Model $Repo $Revision
if ($LASTEXITCODE -ne 0) { throw "Could not download the $Repo checkpoint (revision $Revision)." }

$Manifest = @{
    package = "laya"
    package_version = "0.3.4"
    checkpoint = $Repo
    revision = $Revision
    installed_at = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json
Set-Content -Encoding UTF8 -Path (Join-Path $Root "manifest.json") -Value $Manifest
Write-Output $Manifest
