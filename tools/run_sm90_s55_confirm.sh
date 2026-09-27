#!/usr/bin/env bash
# Registered after the four-cell graph screen; complete calls, fixed references.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WORK=/workspace/gdn-sm90-library-compare-20260926 CANDIDATE_RAW_BIT=1
export CUDA_EXTENSION=/workspace/gdn-sm90-v64-paired-tail-20260927/s52-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export PPU_SOURCE_EXTENSION=/workspace/gdn-sm90-h800-20260926/ppu-source-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export CANDIDATE_EXTENSION=/workspace/gdn-sm90-v64-inverse-tail-20260927/s55-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
base=/workspace/gdn-sm90-v64-inverse-tail-20260927/s55-nsys
[[ ! -e "$base" ]] || { echo "refuse existing evidence: $base"; exit 2; }
mkdir -p "$base"
failures=0
for WORKLOAD in seq8192 heads16; do
  for FAMILY in flashinfer flashqla; do
    for GATE in -0.1 -1.0; do
      export WORKLOAD FAMILY GATE OUT="$base/$WORKLOAD-$FAMILY-g$GATE"
      if bash "$ROOT/tools/run_sm90_library_nsys.sh" > "$OUT-driver.log" 2>&1; then
        echo "[S55 confirm] CAPTURE/PASS $WORKLOAD $FAMILY g=$GATE; speed verdict in result.json"
      else
        failures=$((failures+1))
        echo "[S55 confirm] FAIL $WORKLOAD $FAMILY g=$GATE; retained $OUT-driver.log"
      fi
    done
  done
done
echo "[S55 confirm] attempts=8 failures=$failures; expanded goal checked separately"
test "$failures" -eq 0
