# PPU port of the original optimized GDN

This port compiles the **original** `csrc/gdn_chunk/` kernels and
`gdn_ops.cu` host dispatcher for PPU. The simplified backend from
`a6ac2de` has been removed: it reproduced the broad affine-scan algorithm,
but did not retain upstream's important optimizations. It is not the path
described below.

The source authority is upstream `aa04271`. This is forward-only GDN,
BF16, D=128, C=16, zero initial state, optional final state, including GVA
(q/k heads divide value heads). QSA and output-gate kernels are **not** PPU
ported by this change. PPU timing and device numerical admission remain pending.

The auto API retains upstream's Hv=32 restriction on its serial path; the
explicit two-level API supports the other positive divisible head counts.

## What is retained

| Optimization | Actual shared source | PPU change |
|---|---|---|
| Strong-decay, B-only last-chunk reset preparation; no six-tensor chunk workspace on the accepted path | `gdn_scan_stage1_reset.cu` and original `gdn_ops.cu` | Native loads/MMA and shared layout |
| Eight-warp fused replay; `s_reg[kCPW][K_BLOCKS_TOTAL]` survives the entire group | `gdn_kernel.cu::gdn_recurrence_fused_kernel` | Native result-to-operand register permutation |
| `raw[2]` / `in[2]` asynchronous next-chunk prefetch, wait cadence, current cooked tiles | Same fused replay | Native PPU cp.async; 16-byte transfers remain contiguous |
| Six 16x16 MMA products forming L²/L⁴/L⁸ Neumann inverse | Original prepare and fused reset helpers | Native F16 MMA with explicit A/C register-order conversion |
| Full A/B transfer and packed `A2 @ [A1|B1]` scan | Original stage1/stage2 TUs | Target primitives only |
| Blelloch exclusive scan for power-of-two group counts; Hillis–Steele inclusive otherwise | Original `gdn_ops.cu` | No dispatch rewrite |
| Serial short/weak-decay path and existing sequence/decay thresholds | Original `gdn_ops.cu` | Same C++ host wrapper, compiled/linked for PPU |

Upstream's reset is an **approximation with numerical admission thresholds**,
not an order-independent/raw-bit guarantee. The gt metric and subsequent
max-|A| check are preserved; this port does not establish a universal error
bound for arbitrary q/k/g. Device tests use the existing repository's 2%
max-absolute-error / reference-max criterion. Repeat stability and GVA
expansion equality are separately checked bit-for-bit.

For S=2048, upstream auto dispatch uses GC=8 two-level/reset when
`mean(-g) >= 0.55`, otherwise serial. Those are original A800-tuned thresholds,
**not a PPU performance optimum**. The benchmark reports both decay domains
and includes host dispatch synchronization; it must not be presented as
kernel-only latency.

## Target seam

`csrc/gdn_chunk/gdn_target.cuh` selects NVIDIA or PPU primitives at compile
time. NVIDIA continues to use its original SM80 m16n8 operations and layouts.
PPU uses actlize's native m16n16 BF16/F16 operations, not scalar-gather GEMMs.

- `include/gdn_qsa/ppu/shared_copy.cuh`: independent swizzled 16x16 cubes,
  transpose views, hardware swizzled matrix loads and native accumulator stores.
- `include/gdn_qsa/ppu/fragment.cuh`: explicit native A/B/C coordinate maps
  and cross-lane register conversion, without a shared-memory state spill.
- All producer and consumer layouts change together. Global inputs and public
  outputs are unchanged; internal prepared workspace belongs to this backend.
- PPU's native result layout differs from its input layout. Treating the
  registers as interchangeable is not valid, even though both fragments have
  eight half elements per thread.

This is optimization-structure parity, **not identical ISA or a speed claim**.
PPU register shuffle cost and hardware load semantics still need device
measurement. An inherited SDK diagnostic says the optional `.LLC::128B`
cp.async hint is ignored; cp.async itself is emitted. No scalar fallback is
silently selected.

## Local evidence and limits

`scripts/verify_ppu_backend_local.sh` runs:

1. Independent affine-algebra tests: 55 cases with inverse-sign, scan-order
   and replay-prefix negatives. These are algebra tests, not device execution.
2. L006 actual CuTe native fragment/layout tests: every lane/slot in normal
   and transposed cubes across the selected 16/32/128/256 extents. Swizzle
   addresses are independently anchored to actlize's hardware-facing cube
   simulator, and 16-byte contiguity is exhaustive for those extents.
3. A six-product native-fragment Neumann chain versus independent forward
   substitution, on an exactly representable dyadic fixture. All 256 results
   match; leaving the result in C order corrupts 111 elements.
4. Original-source comparison: all 305 control expressions across the seven
   original TUs, launch geometries, wait/barrier cadence, and the complete
   host dispatch body remain upstream-identical. Removing a wait, a buffer,
   changing a loop bound, or moving state into the chunk loop must turn the gate red.
5. hgcc PPU0010 compile **and link** of all six original device TUs plus the
   original PyTorch host wrapper, then resource/ISA admission.

Local SDK 2.1.1 emits 15 kernel symbols including both reset and full-scan
paths. Default C16 kernels have zero stack; fused register replay uses
182 vector registers/thread in this build. The original experimental C32
recurrence has a 144-byte stack and is explicitly **not PPU performance
admitted**; auto never chooses it. The resource report prints it, rather than
hiding it or treating it as a C16 regression.

The source gate is tied to `aa04271`; a shallow checkout must contain that
commit to run it. These checks do not prove PPU runtime load/execute behavior,
rounding, race freedom or speed. This local host cannot import the SDK runtime
(GLIBC_2.38 / GLIBCXX_3.4.32 are unavailable), and it has no PPU. Compile/link
is PASS; local runtime import and device execution are **unavailable**, not PASS.
Use the matching PPU SDK/PyTorch container on the box.

The SDK's executable named `nvcc` is its PPU compatibility driver (a trial
`-arch=sm_80` invocation still reported `compiling for ppu001`), not independent
NVIDIA codegen validation. NVIDIA execution/rebuild is not claimed here.

## Build and device handoff

On the PPU box, from this repository's `ppu-backend` branch:

```bash
git pull --ff-only
git submodule update --init --recursive
PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 \
  bash tools/run_ppu_gdn_backend_box.sh
```

All artifacts go under `/workspace/gdn-qsa-ppu-<sha>-<UTC>`. The runner builds
against the box's installed PyTorch (no hard-coded torch 2.8 binary), preserves
source SHA/diff and binary hashes, then tests reset, Blelloch, Hillis–Steele,
serial, tails, GVA and S=2048. Any correctness/route failure stops timing.
The default final timing is B1/S2048/Hk16/Hv32/D128 at strong and weak decay.
Set `PERF=0` for correctness only.

To use the built original host wrapper directly:

```python
# Set GDN_QSA_PPU_EXTENSION to the build's _gdn_chunk_ppu*.so before first use.
from gdn_qsa_sm80 import gdn_chunk, gdn_chunk_twolevel
out, state = gdn_chunk(q, k, v, g, beta)
out, state, route_info = gdn_chunk_twolevel(q, k, v, g, beta, group_chunks=8)
```

The PPU extension is explicit opt-in. Without the environment variable, the
original NVIDIA extension remains the default. An invalid PPU path fails;
there is no silent fallback. All public inputs must be BF16 on one device.
