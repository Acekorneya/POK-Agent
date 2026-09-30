#!/usr/bin/env bash
# Load POK-Agent credentials into the current shell without printing them.
# Reads provider keys from the Windows Credential Manager via WSL interop
# (scripts/read-credential.ps1) and sets the local decision-router tokens.
# Usage: source scripts/load-keys.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PS_EXE="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
PS_SCRIPT="$(wslpath -w "$SCRIPT_DIR/read-credential.ps1")"

read_cred() {
    "$PS_EXE" -NoProfile -NonInteractive -ExecutionPolicy Bypass \
        -File "$PS_SCRIPT" -Name "$1" 2>/dev/null | tr -d '\r'
}

export TYPESAFE_API_KEY="$(read_cred TYPESAFE_API_KEY)"
export OPENROUTER_API_KEY="$(read_cred OPENROUTER_API_KEY)"
export POK_LAYA_TOKEN="${POK_LAYA_TOKEN:-dev-laya-token}"
export POK_JUDGE_TOKEN="${POK_JUDGE_TOKEN:-dev-judge-token}"