#!/usr/bin/env bash
# Two diagnostic captures after S47's B2 screen loss, not a promoted finalist.
# Count prepare+state, keep both immutable controls and pinned FI unchanged.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WORK=/workspace/gdn-sm90-library-compare-20260926
export FAMILY=flashinfer GATE=-0.1 CANDIDATE_RAW_BIT=1
export PPU_SOURCE_EXTENSION=/workspace/gdn-sm90-h800-20260926/ppu-source-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
export CANDIDATE_EXTENSION=/workspace/gdn-sm90-precomputed-aux-20260926/s47-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
base=/workspace/gdn-sm90-precomputed-aux-20260926/s47-nsys-diagnostic
[[ ! -e "$base" ]] || { echo "refuse existing evidence: $base"; exit 2; }
mkdir -p "$base"
failures=0
for WORKLOAD in seq2048 batch2; do
  export WORKLOAD OUT="$base/$WORKLOAD"
  if [[ "$WORKLOAD" == seq2048 ]]; then
    export CUDA_EXTENSION=/workspace/gdn-sm90-value-split-20260926/s38-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
  else
    export CUDA_EXTENSION=/workspace/gdn-sm90-win-20260926/relative-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so
  fi
  if bash "$ROOT/tools/run_sm90_library_nsys.sh" > "$OUT-driver.log" 2>&1; then
    echo "[S47 diagnostic] CAPTURE/PASS $WORKLOAD; performance in result.json"
  else
    failures=$((failures+1))
    echo "[S47 diagnostic] FAIL $WORKLOAD; retained $OUT-driver.log"
  fi
done
echo "[S47 diagnostic] attempts=2 failures=$failures; not general speed admission"
test "$failures" -eq 0
