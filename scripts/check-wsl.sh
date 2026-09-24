#!/usr/bin/env bash
set -euo pipefail

source "${CARGO_HOME:-$HOME/.cargo}/env"
project_dir="$(cd "$(dirname "$0")/.." && pwd)"

cargo fmt --manifest-path "$project_dir/Cargo.toml" --all -- --check
cargo test --manifest-path "$project_dir/Cargo.toml" -p pok-ai-core -p pok-ai-cli
npm --prefix "$project_dir/apps/desktop" run build

