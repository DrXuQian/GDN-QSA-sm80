# SM90 follow-up: producer ownership and independent state issue

Continuation of [the first SASS campaign](SM90_FLASHINFER_SASS_20260926.md).
Same physical H800, CUDA12.8, B1/T2048/Hqk16/Hv32/D128/C64, scalar gates
g=-0.1/-1.0, unchanged BF16/FP32 boundaries and independent 2% O/state gate.
Numbers below are **nsys sums of every GPU kernel in one full forward**,
12 interleaved/reversed-order calls per role. They are not API/event times.
No default routing or SM80/PPU1.0 changes. Native PPU1.7 remains unmeasured.

## S16 is the new confirmed experimental incumbent

S16 `e799c55`, branch `sm90-factor-owner-20260926`, combines cached identical
gate coefficients with publication by the existing alpha-last warp, leaving
the Q/K/V TMA warp free to issue copies. Relative to its cached S12 parent,
state/auxiliary arithmetic and storage are unchanged: no extra warp, no new
shared allocation, no fast-math flag. The producer publishes all four factor
channels before alpha commit, consumes/releases that same alpha stage while
publishing alpha-last, and retains all416 alpha consumers.

This is an interaction test, not a reclassification of the earlier cache-only
loss/overlap or producer-only overlap as wins. Native state coefficient work
is retained at the lower count. Delivery correctness passes14 independent
cases and parent input/output-state fingerprints/errors,8 stable repeats and
all captured outputs. Missing-channel/producer/callback/release negatives fail.

| Paired capture | S11 control | S16 | Fastest reference | Verdict |
|---|---:|---:|---:|---|
| FI, g=-0.1 |147.6165 |142.9600 |112.8005 no-CP |S16 improves; FI wins |
| FI, g=-1.0 |147.2800 |142.8805 |112.8800 no-CP |S16 improves; FI wins |
| QLA, g=-0.1 |146.3530 |140.6090 |165.3610 auto-CP |S16 wins |
| QLA, g=-1.0 |146.4320 |140.2405 |163.6005 auto-CP |S16 wins |

All stated wins/losses use disjoint observed envelopes in that SAME row.
Units:microseconds. S16 improves S11 by3.0–4.2% and beats QLA1.176x/1.167x;
it is still about1.267x as slow as FI. **The two-library target is NOT MET.**
Do not pair the best QLA-window candidate time with another window's FI time.

Measured binary SHA256:
`a88a07e8c48c53a860e5cc94a936737be508e487314acbc0d197cbbf34970b08`.
Remote binary:`/workspace/gdn-sm90-win-20260926/factor-owner-build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so`.

## Rejected or unresolved experiments remain visible

- **S14** `107c0de`: immediate predicated named-barrier IDs remove the intended
  runtime-ID spill path. Other spills grow; stack64→96B. Paired148.689 versus
  147.2805us overlaps:UNRESOLVED. Actual old S11 native image fails the new
  immediate-ID gate. No protocol waits were deleted.
- **S15** `5ab062b`: an empty-assembly H operand fence intended to force
  recomputation is optimized away. All29,768 normalized native instructions
  equal S11, SHA`96c0bf23a280c0cdf48d1a12368e2546bb3e125ad21303d6674806bbaa1e9ab5`.
  Rejected before device timing; do not retain this inert asm as a fix.
- **S17** `daabe62`: K,Q,V TMA order instead of Q,K,V, independently on S11.
  Actual native destinations change from Q(base0),K(base0x8000) to K,Q in all
  four bodies; six TMA load sites and descriptor/storage binding remain.
  All14 numeric/fingerprint cases pass.147.9845[147.041,148.961] versus
  146.865[145.792,148.384]us overlaps:UNRESOLVED, no composition/promotion.

## S18: remove issue ordering, not data dependencies

Experimental kernel`642d353`, native-gate follow-up`f529083`, parentS16.
Since inverse ownership moved to the auxiliary WG, state WG0 owns V0..63
and WG1 V64..127. The actual instantiated CuTe types prove disjoint ownership
of16,384 H elements,8,192 O elements and8,192 physical shared output elements
for each of four gate/initial-state specializations. A stale WG0 writer fails.

Only the two-WG alternating issue preference is removed, and only for the
aux-owned inverse specialization. The original state-owned inverse keeps its
ordered policy. TMA stage completion/reuse, all arrival counts, O publication,
async shared fences and every WGMMA fence/commit/wait remain. Source hashes
bind the state body and pipeline sources to S16; deleting a real completion
wait or release fails. The PTX contract scopes WGMMA groups/completion to the
executing warpgroup; that is necessary background, NOT by itself proof that
arbitrary inter-WG barriers may be removed. See
[NVIDIA PTX8.7 WGMMA completion](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-wait-group).

All four native bodies retain identical HGMMA/HMMA, TMA and SYNCS/WARPGROUP
families while256-thread issue-barrier sites33/41 become0. Old native code
and removed data-completion/arrival negatives fail. Local spills decrease,
but stack88/80B remains; this is NOT a spill-free implementation.37 host
tests pass in the Torch-capable environment. The local default Python lacks
Torch; its four import errors were not reported as passes.

Device14/14 CPU cases and S16 output/state fingerprints/errors pass.
First timing attempt`independent-fi-weak` is **INVALID**, stopped before
capture by the idle guard observing foreign PID143838. No timing is inferred
from that attempt; no foreign job was stopped. Fresh exclusive retry
`independent-fi-weak-r2` gives142.016[140.257,143.009]us versus S16
142.321[140.993,143.616]us: **UNRESOLVED, do not promote**. FI no-CP
114.4165[113.025,114.881]us remains faster. Eight-repeat/every-captured-call
checks passed. This rejects the claim that the inter-WG issue preference
alone explains the remaining gap; it does not price other dependency waits.

S16 also passes four actual bodies against the PPU CUTLASS3.6 dependency in
CUDA source-check mode. Native PPU1.7 compile/execution/performance are still
unavailable, not inferred from the H800 result.
