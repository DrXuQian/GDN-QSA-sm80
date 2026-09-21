# FLA-aligned weak-path candidate

Source parent: `8837640119835b6c8b7953d2c61e24de9fe35000`.
Worktree: `/workspace/gdn-wy-align-20260921`; forward only, pure C++/CuTe/actlize.
No PPU is available locally. This iteration ends at locally verified code and
a device admission/comparison command, not at a claimed performance win.

## Contract and target

Priority: B1/S2048/Hk16/Hv32/K128/V128, BF16 q/k/v/g/beta, native GVA,
zero initial state, output and final state, scale=1/sqrt(128), no implicit
QK normalization. Test g=-0.1, g=0, g=-1, and varying nonpositive g separately.
Also cover S=1/16/63/64/65/129, nonzero initial state, Hk1/Hv2 and Hk16/Hv64.
The new explicit API may accept FP32 gates/state; old APIs remain unchanged.
Frozen numerical gate: BOTH output and state max-abs-error/reference-max <2%,
using the existing token-recurrent CPU oracle. Repeat and native/expanded GVA
must be bit-stable. Do not weaken this gate after observing a result.

## Work packages

1. Add a separate C64 WY path. Compute the triangular system/inverse and W/U
   per chunk, independently of incoming state. Recur only over W@H and
   K^T@Vnew. Give each CTA a V32 slice with FP32 state retained in registers.
   Compute output in a separate chunk-parallel kernel. No approximate resets
   in this path, no host decay reduction/readback, no repeated GVA Q/K copies.
2. Prove the factorization against token recurrence locally, including tail,
   state, GVA, decay, inverse sign and output causality negative controls.
   Anchor actual native fragment ownership to actlize traits; compile/link
   real PPU bodies and reject spills in the new recurrent kernel.
3. Add a three-arm same-input runner: unchanged original auto / explicit WY /
   installed FLA. Preserve raw alternating samples, exact binary/source/device
   identity, errors and fingerprints. Full-call timing includes all new stages
   and allocations; kernel-only ACU is a separate measurement. Overlapping
   sample envelopes are UNRESOLVED. Goal: WY <=1.10x FLA on the priority weak
   shape, without a numerical regression. This is a target, not evidence.

## Preserve, do not blindly replace

| Existing capability | Decision / evidence boundary |
|---|---|
| Admitted strong-decay B-only reset preparation | Keep original source and thresholds. Can avoid full transfer/scan work; approximate, not universally valid or universally faster. |
| Eight-warp fused register replay + double-buffer prefetch | Keep untouched. This is the strong-path advantage to retain in the candidate portfolio. |
| Six-product 16x16 Neumann inverse | Keep original. It is a possible later substitution for the new diagonal solve, not silently mixed into the FLA-alignment comparison. |
| Full affine transfer + Blelloch/Hillis-Steele fallback | Keep original, including its non-power-of-two support and admission checks. |
| Weak serial path | Keep as immutable numerical/performance counterfactual. New WY path is explicit opt-in until box evidence. |

FLA's equivalent algebra does not promise bit equality to the original BF16
chunk recurrence. It retains FP32 recurrent state; casts for MMA and saved
snapshots are explicit. V splitting must not split a reduction dimension.
The cost of extra saved states/Vnew and launches is included, not hidden.

## Evidence / promotion

The uploaded PPU ACU points to recurrence+output (720.421 us versus FLA
90.271+45.311 us), but another process was visible: diagnostic, not clean timing.
RTX5070 findings establish a same-upstream structural issue, not a PPU forecast.
Local validation: algebra + negative controls, native layouts, original-source
preservation, SDK compile/link/resource checks, Python contracts.
Device validation: unchanged original admission, new path vs recurrent oracle,
three-arm alternating unprofiled timing, optional direct `acu --set full`.
Never replace auto dispatch based only on host or compile evidence.
