#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/workspace/gdn-qsa-ppu-build}"
JOBS="${JOBS:-16}"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"

mkdir -p "$BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DGDN_QSA_ENABLE_PPU=ON \
  -DPPU_SDK_ROOT="$PPU_SDK_ROOT" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target gdn_qsa_ppu -j"$JOBS"

LIB="$BUILD_DIR/libgdn_qsa_ppu.so"
HGOBJDUMP="$PPU_SDK_ROOT/bin/hgobjdump"
test -s "$LIB"
for symbol in \
  gdn_qsa_ppu_workspace_size_v1 \
  gdn_qsa_ppu_forward_bf16_v1 \
  gdn_qsa_ppu_scan_round_bf16; do
  if ! nm -D --defined-only "$LIB" | grep -q " $symbol$"; then
    echo "[PPU binary audit] FAIL: missing exported symbol $symbol" >&2
    exit 1
  fi
done

if [[ ! -x "$HGOBJDUMP" ]]; then
  echo "[PPU binary audit] FAIL: hgobjdump unavailable at $HGOBJDUMP" >&2
  exit 1
fi
"$HGOBJDUMP" --arch=ppu1.0 --dump-isa "$LIB" >"$BUILD_DIR/gdn_qsa_ppu.isa"
"$HGOBJDUMP" --dump-resource-usage=all "$LIB" \
  >"$BUILD_DIR/gdn_qsa_ppu.resources"
MMA_COUNT="$(grep -c 'v\.mma\.f32\.bf16\.m16n16k16' \
  "$BUILD_DIR/gdn_qsa_ppu.isa" || true)"
if [[ "$MMA_COUNT" -le 0 ]]; then
  echo "[PPU binary audit] FAIL: shipping binary contains no BF16 MMA" >&2
  exit 1
fi
if grep -Eq 'STACK SIZE:[[:space:]]*[1-9]' \
  "$BUILD_DIR/gdn_qsa_ppu.resources"; then
  echo "[PPU binary audit] FAIL: shipping kernel has a nonzero stack frame" >&2
  exit 1
fi

echo "[PPU binary audit] PASS mma_static_sites=$MMA_COUNT stack_bytes=0"
echo "library: $LIB"
