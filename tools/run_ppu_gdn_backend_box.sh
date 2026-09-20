#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPU_SDK_ROOT="${PPU_SDK:-/usr/local/PPU_SDK}"
JOBS="${JOBS:-16}"
DEVICE="${DEVICE:-0}"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-/workspace/gdn-qsa-ppu-${SHA}-${STAMP}}"
BUILD_DIR="$OUT/build"

mkdir -p "$OUT" "$BUILD_DIR"
git -C "$ROOT" submodule update --init --recursive

BUILD_DIR="$BUILD_DIR" PPU_SDK="$PPU_SDK_ROOT" JOBS="$JOBS" \
  bash "$ROOT/scripts/build_ppu.sh" 2>&1 | tee "$OUT/build.log"

LIB="$BUILD_DIR/libgdn_qsa_ppu.so"
sha256sum "$LIB" | tee "$OUT/library.sha256"
CUDA_VISIBLE_DEVICES="$DEVICE" \
LD_LIBRARY_PATH="$PPU_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python "$ROOT/tests/test_ppu_gdn_backend.py" \
    --library "$LIB" --device 0 2>&1 | tee "$OUT/correctness.log"

echo "[PPU GDN box] PASS: artifacts=$OUT"
