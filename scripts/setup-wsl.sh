#!/usr/bin/env bash
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi

source "${CARGO_HOME:-$HOME/.cargo}/env"
rustup toolchain install 1.88.0 --profile minimal
rustup component add --toolchain 1.88.0 rustfmt clippy

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 20+ is required for the Tauri frontend." >&2
  exit 1
fi

cargo fetch --manifest-path "$(dirname "$0")/../Cargo.toml"
npm --prefix "$(dirname "$0")/../apps/desktop" install

echo "WSL environment ready. Run scripts/check-wsl.sh next."
