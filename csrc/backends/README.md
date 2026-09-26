# Execution-family boundary

`cuda_sm80/primitives.cuh` and `ppu_aiu/primitives.cuh` contain the unchanged
primitive implementations extracted from the legacy shared adapter.
`csrc/gdn_chunk/gdn_target.cuh` is the existing warp-MMA/AIU compatibility entry.
It is not the interface for a future Hopper pipeline.

The existing scan/reset source TUs remain shared where they were already shared.
WY/residual PPU TUs retain their distinct algorithms and private workspaces.
They can continue to reuse those already-proven legacy AIU primitives.
There is no benefit in copying those files merely to change a directory name.

A new SM90 algorithm gets a separate source graph here, with its own TMA/GMMA
pipeline and workspace, then a complete-forward registration. PPU1.7 uses the
selected PPU CUTLASS3.6.0 dependency, not these legacy AIU primitives. A validated
shared SM90 algorithm may select target-specific descriptor/copy/TF32/barrier
traits; no assumption of binary-equivalent NVIDIA and PPU hardware is made.

Targets not yet implemented are explicitly rejected by
`gdn_qsa_sm80/backends/targets.json` in both build and public dispatch. No empty
SM90 kernel or silent SM80/PPU1.0 fallback is provided. FP8/FP4 remain out of scope.
