#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-/workspace/gdn-qsa-ppu-local}"
JOBS="${JOBS:-16}"

mkdir -p "$OUT"
cmake -S "$ROOT" -B "$OUT" -DBUILD_TESTING=ON
cmake --build "$OUT" -j"$JOBS"
ctest --test-dir "$OUT" --output-on-failure
"$OUT/l001_ppu_superchunk_schedule"
"$OUT/l002_ppu_warp_mma_ownership"
"$OUT/l004_ppu_affine_pipeline"
"$OUT/l005_ppu_backend_c_abi"

OUT="$OUT/l003" \
  bash "$ROOT/dev/ppu/run_l003_device_compile.sh"
