#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <name>"
  echo "Example: $0 u64"
  exit 1
fi

NAME="$1"
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" != "Linux" ]]; then
  echo "❌ Only works on Linux!"
  exit 1
fi

if [[ "$ARCH" == "aarch64" ]]; then
  echo "❌ Only works on x86_64!"
  exit 1
fi

echo "🔧 Building 'profile_${NAME}' w/ custom flags..."

RUSTFLAGS="-C force-frame-pointers=yes -C debuginfo=2 -C opt-level=2 -C codegen-units=1 -C panic=abort -C llvm-args=--inline-threshold=0" \
cargo build --example "profile_${NAME}" --profile profiling

echo "🚀 Running perf record..."
taskset -c 2 perf record --call-graph dwarf \
  --output perf.data ./target/profiling/examples/profile_"${NAME}"

echo "✅ Done!"
