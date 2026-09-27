# Closing the remaining SM90 GDN losses

Checkpoint:2026-09-27 01:43:35 UTC. Physical H800 PCIe,114SMs,CUDA12.8.93.
The fixed14-workload inventory and both gate regimes are unchanged. This is
not a native PPU1.7 result. No default routing, SM80, reference, clock, power
or numerical-tolerance change. **Expanded speed goal remains unmet.**

## S52: V64 plus independently live O2/KV operands

Source2d5e78d (kernel549ee4c), parent2b51b2f/S38. Apply the
[S50 overlap](SM90_PAIRED_STATE_20260927.md) to one state WG, not the old
two-WG geometry. Keep loader24/state192/aux232,384threads,152576B shared.
Actual delta map4096/4096 and two negatives pass. Actual state/output map
16384/8192,96 shape-stride maps and four negatives pass. The one-state
data-ring checker terminates for every reachable reduced state at1..8chunks;
this is not a memory-order proof.

All four native bodies keep matrix/TMA/publication work and four paired
O2/KV epochs, with distinct live operand registers. Old-body/missing-matrix/
serialized-WGMMA negatives fail. Initial-state spills increase; the candidate
is not a blanket resource improvement. PPU CUTLASS3.6 CUDA source-check
passes; unavailable native PPU1.7 SDK/model is **SKIP**.

All14 independent CPU cases, parent fingerprints, two extreme raw-byte
stress pairs, eight direct repeats and every graph/profiler output pass.
Eight graph cells give7parent wins/1unresolved. B2 remains much slower than
the retained V128 path, so V64 must not become an unconditional selector.

Eight registered nsys captures,528 complete forwards, have all been
re-extracted locally from SQLite to byte-identical result JSON. Units are
microseconds, **sum of every GPU kernel in one complete forward**, not graph
screen time or Python/API latency:

| Workload | Gate | Reference | Paired S38 | S52 | Fastest reference | S52 vs fastest |
|---|---:|---|---:|---:|---:|---|
| B1/T8192/Hv32 | -.1 | FlashInfer |361.4865|351.5185|354.654 auto|UNRESOLVED|
| B1/T8192/Hv32 | -1 | FlashInfer |361.903|352.559|353.439 auto|UNRESOLVED|
| B1/T8192/Hv32 | -.1 | FlashQLA |362.160|351.839|382.4955 auto|WIN|
| B1/T8192/Hv32 | -1 | FlashQLA |361.9515|351.7275|375.024 auto|WIN|
| B1/T2048/Hv16 | -.1 | FlashInfer |94.544|91.904|93.360 auto|UNRESOLVED|
| B1/T2048/Hv16 | -1 | FlashInfer |94.752|91.888|92.752 auto|UNRESOLVED|
| B1/T2048/Hv16 | -.1 | FlashQLA |93.536|91.792|92.3195 no-CP|UNRESOLVED|
| B1/T2048/Hv16 | -1 | FlashQLA |93.696|91.712|92.096 no-CP|UNRESOLVED|

Median-only reference leads do **not** pass the unchanged disjoint-range
criterion. E.g. T8192 weak S52[348.958,352.606] overlaps fastest
FI[351.679,357.951]. Strong S52[346.079,353.024] overlaps
FI[346.368,356.671]. Keep these as UNRESOLVED, not wins.

Evidence:/workspace/gdn-sm90-v64-paired-tail-20260927/s52-nsys.
Source and binary identities are in each receipt. Local native image SHA:
553b77feb975bce16201a669e40847c0f390c52abd1827a8be2a5e80c55abc77.

Retaining old winners plus only these confirmed per-cell improvements gives
FI16wins/8losses/4unresolved; QLA25wins/0losses/2unresolved/1numeric-invalid.
This is a composite best-so-far ledger across bound epochs, **not** a newly
rerun full matrix or permission to ship an automatic selector. The remaining
FI losses are B2/B4/Hv64 in both gate regimes. The known QLA weak-GVA1
numeric failure remains visible under the unchanged2% gate.

## S51: last inverse partials local to two row-owning warps

Source36c007d, independent S24 parent. The same last32->64 inverse level
keeps its two K16 FP32 products, separate FP16 casts and FP16 high+low sum.
It does not contract them into one differently-rounded K32 product.
Two row-owning warps compute twice the per-warp last-level work; CTA useful
HMMA count remains32. One input-C alias barrier stays, the cross-warp partial
publication barrier and exchange disappear. Caller publication remains.

Actual DC/output1024/1024 maps and wrong-half/omitted-warp/contraction negatives
pass. Four native bodies preserve state work and async protocols, remove two
barrier/four partial-store sites across full/tail clones, with noC7512.
14CPU parent-raw+2stress and eight-screen repeated/captured outputs pass.
Three narrow S24 graph wins/five unresolved are not reference speed admission.
S50 remains the confirmed B2 improvement; no S51 reference win is claimed.

## Bounded followups, not presumed additive wins

- S53 source8e89c8a: exactly compose S51 with S50 on V128. Four-arm,
  six-cell B2/B4/Hv64-GVA4 screen includes both single-change controls.
  Native/map gates pass; device14CPUparentraw+2stress pass; screen in progress.
- S54 sourcefa0dfa7: only remove optional alternating state issue on S50;
  every S50 data/lifetime source is bound unchanged. Actual physical H/O
  owners are disjoint in all four types; old inverse-owned-by-state retains
  ordering. Native33/41 issue sites become zero with all matrix/TMA/async/data
  families preserved. Device admission is queued; no speed claim.
- S55 source6317ea4: exactly compose S51 with S52 V64/aux232. Native/map
  gates pass. Four-cell T8192/Hv16 screen will compare S52 and S38 after
  fixed numerical admission. Device performance NOT_RUN.

Each has a pre-edit plan, fixed deadline and independent worktree under
/workspace. GPU work is sequential and DeviceWatch-gated. H800 remains on.
