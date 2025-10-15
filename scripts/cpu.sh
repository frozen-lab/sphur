#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_NAME=$1
OUT_DIR=".local"
OUT_FILE="${OUT_DIR}/stat_${EXAMPLE_NAME}.txt"
EXE="./target/profiling/examples/profile_${EXAMPLE_NAME}"

# Create output directory if missing
mkdir -p "${OUT_DIR}"

echo "🔧 Building 'profile_${EXAMPLE_NAME}'..."
cargo build --example "profile_${EXAMPLE_NAME}" --profile profiling

echo "🚀 Running perf stat..."

sudo chrt -f 99 taskset -c 2 perf stat \
  "${EXE}" 2>&1 | tee >(grep -E 'cycles|instructions|branches|branch-misses|time elapsed|task-clock' > "${OUT_FILE}")

echo "✅ Saved to ${OUT_FILE}"
