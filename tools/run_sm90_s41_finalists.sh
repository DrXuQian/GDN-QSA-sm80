#!/usr/bin/env bash
# A bounded, sequential physical-H800 confirmation. No compiler/kernel changes.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WORK=/workspace/gdn-sm90-library-compare-20260926
export CUDA_EXTENSION=/workspace/gdn-sm90-value-split-20260926/s38-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export PPU_SOURCE_EXTENSION=/workspace/gdn-sm90-h800-20260926/ppu-source-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export CANDIDATE_EXTENSION=/workspace/gdn-sm90-value-loader-20260926/s41-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export CANDIDATE_RAW_BIT=1
base=/workspace/gdn-sm90-multishape-20260926/s41-nsys
[[ ! -e "$base" ]] || { echo "refuse existing evidence directory: $base"; exit 2; }
mkdir -p "$base"
failures=0
for pair in 'flashinfer seq2048' 'flashinfer seq8192' 'flashqla heads16'; do
  read -r FAMILY WORKLOAD <<< "$pair"
  export FAMILY WORKLOAD
  for GATE in -0.1 -1.0; do
    export GATE OUT="$base/$FAMILY-$WORKLOAD-$GATE"
    if bash "$ROOT/tools/run_sm90_library_nsys.sh" > "$OUT-driver.log" 2>&1; then
      echo "[S41 nsys] PASS $FAMILY $WORKLOAD $GATE"
    else
      failures=$((failures+1))
      echo "[S41 nsys] FAIL $FAMILY $WORKLOAD $GATE; retained $OUT-driver.log"
    fi
  done
done
echo "[S41 nsys] attempts=6 failures=$failures; performance verdicts in result.json"
test "$failures" -eq 0
