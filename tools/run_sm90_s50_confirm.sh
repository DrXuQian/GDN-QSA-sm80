#!/usr/bin/env bash
# Registered after S50's eight-cell graph screen; four full-call captures.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WORK=/workspace/gdn-sm90-library-compare-20260926
export WORKLOAD=batch2 CANDIDATE_RAW_BIT=1
export CUDA_EXTENSION=/workspace/gdn-sm90-win-20260926/relative-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export PPU_SOURCE_EXTENSION=/workspace/gdn-sm90-h800-20260926/ppu-source-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export CANDIDATE_EXTENSION=/workspace/gdn-sm90-paired-tail-20260927/s50-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
base=/workspace/gdn-sm90-paired-tail-20260927/s50-nsys
[[ ! -e "$base" ]] || { echo "refuse existing evidence: $base"; exit 2; }
mkdir -p "$base"
failures=0
for FAMILY in flashinfer flashqla; do
  for GATE in -0.1 -1.0; do
    export FAMILY GATE OUT="$base/$FAMILY-g$GATE"
    if bash "$ROOT/tools/run_sm90_library_nsys.sh" > "$OUT-driver.log" 2>&1; then
      echo "[S50 confirm] CAPTURE/PASS $FAMILY g=$GATE; speed verdict in result.json"
    else
      failures=$((failures+1))
      echo "[S50 confirm] FAIL $FAMILY g=$GATE; retained $OUT-driver.log"
    fi
  done
done
echo "[S50 confirm] attempts=4 failures=$failures; expanded goal checked separately"
test "$failures" -eq 0
