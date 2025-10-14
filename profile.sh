#!/usr/bin/env bash
set -euo pipefail

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

echo "🔧 Building 'profile' w/ custom flags..."

RUSTFLAGS="-C force-frame-pointers=yes \
           -C debuginfo=2 \
           -C opt-level=2 \
           -C codegen-units=1 \
           -C panic=abort \
           -C llvm-args=--inline-threshold=0" \

cargo build --example profile --profile profiling

echo "🚀 Running perf record..."
taskset -c 2 perf record --call-graph dwarf ./target/profiling/examples/profile

echo "✅ Done! Now inspect w/"
echo "   perf report"
