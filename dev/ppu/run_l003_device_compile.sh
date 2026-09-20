#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PPU_SDK_ROOT="${PPU_SDK:-${PPU_SDK_ROOT:-/usr/local/PPU_SDK}}"
if [[ ! -x "$PPU_SDK_ROOT/bin/hgcc" &&
      -x /root/autodl-tmp/root-relocated/ppu-sdk/2.1.1/bin/hgcc ]]; then
  PPU_SDK_ROOT=/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1
fi
OUT="${OUT:-/workspace/gdn-qsa-ppu-device-compile}"
HGCC="$PPU_SDK_ROOT/bin/hgcc"

mkdir -p "$OUT"
if [[ ! -x "$HGCC" ]]; then
  echo "[ppu device compile] SKIP: hgcc unavailable at $HGCC"
  exit 0
fi

common=(
  -std=c++17 -arch=ppu_10 -O3
  --expt-relaxed-constexpr --expt-extended-lambda
  -I"$ROOT/include"
  -I"$ROOT/third_party/actlize/include"
  -I"$PPU_SDK_ROOT/include"
  -c
)
"$HGCC" "${common[@]}" \
  "$ROOT/csrc/gdn_chunk_ppu/gdn_pipeline_ppu.cu" \
  -o "$OUT/gdn_pipeline_ppu.o"
"$HGCC" "${common[@]}" \
  "$ROOT/csrc/gdn_chunk_ppu/gdn_scan_stage2_ppu.cu" \
  -o "$OUT/gdn_scan_stage2_ppu.o"

for object in "$OUT/gdn_pipeline_ppu.o" "$OUT/gdn_scan_stage2_ppu.o"; do
  test -s "$object"
done
echo "[ppu device compile] PASS: exact pipeline + scan sources reached hgcc ppu_10 objects"
