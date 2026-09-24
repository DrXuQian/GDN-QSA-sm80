# FLA-structured C++/actlize forward rewrite

Decision2026-09-24: stop making the next box experiment another address
micro-ablation. Build an independent execution path aligned to the actual
FLA forward dataflow and operand layouts. The current admitted mask1520 path
and original strong-decay implementation remain controls, not deleted or
silently replaced. This document is a design/feasibility checkpoint, not a
completed rewrite or speed result.

## What is wrong with the present implementation

The [verified uploaded profile](PPU_WY_SHARED_ACU_20260922.md) establishes
three gaps, not just state: prepare135.350 vs83.107 us, state145.466 vs89.490,
output66.476 vs43.385. BF16 MMA work agrees phase by phase. State/output
execute2.165x/2.352x as many native instructions; their HBM reads and global
write traffic agree. State occupancy is equal; output occupancy is higher
on our slower implementation. Thus a different grid or another few reused
exponents is not a sufficient explanation or an architectural plan.

We matched tensor formulas and gradually improved tiling, but retained a
generic shared-cube delivery layer. FLA compiles a tensor ownership/layout
schedule together with each dot. Its selected native code uses lane-address
NCOM loads; ours uses scalar-descriptor SWZL loads, generic pointer/layout
arithmetic and extra exchanges/materialization. This is a structural
implementation difference, not proof that the GDN/WY equations are wrong.
Nor does it prove the SWZL instruction itself accounts for every excess.

The reference is the uploaded, hash-verified FLA source, not a moving latest
checkout. Only forward is in scope. The relevant source-to-stage mapping is:

| Phase | Actual FLA forward source | Current C++ structure | Rewrite boundary |
|---|---|---|---|
| Prefix and solve | `utils/cumsum.py`, `gated_delta_rule/chunk_fwd.py` | Prefix, shared lower/inverse, solve, conditioning and W/U in one70 KiB CTA | Separate prefix and KKT+solve; tile-local solve ownership |
| W/U | `gated_delta_rule/wy_fast.py::recompute_w_u_fwd_kernel` | Scalar BF16 W/U publication; KVD interface16x useful output bytes | Dedicated tensor-core W/U with vector publication |
| State | `common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | FP32 register state already resident, but repeated generic shared delivery | Explicit resident-state/operand layouts, compile-time address increments |
| Output | `common/chunk_o.py::chunk_fwd_kernel_o` | QK first, then two V64 panels reloading Q for QH, shared output exchange each panel | Joint QH/QK schedule using loaded Q, whole selected V tile, direct/vector epilogue |

FLA itself still recurs over the32 chunks in the state kernel. Do not replace
this with a new parallel-scan approximation or claim FLA removed recurrence.
Also, our FP32 state already lives in registers; moving it there is not a
new optimization. The difference is operand conversion/delivery and surrounding
state publication, not simply where the main state variable is declared.

## A concrete adapter gap is locally reachable

Pinned actlize423253c0 `include/cute/arch/copy_ppu.hpp` deletes all six plain
PPU0010 LDSM entrypoints: old inline-PTX grammar was unproved. That is not a
hardware limitation. SDK2.1.1 public `hggc_mma.h` exposes `awmma::ldmatrix`.
The [compile-only probe](../dev/ppu/l014_plain_ldmatrix_compile.cu) uses that
API, not guessed PTX syntax or copied SDK internals.

Local real SDK compilation/disassembly produces:

| Public SDK operation | Native body load | v.mov.v2s | Registers / stack |
|---|---|---:|---|
| `ldmatrix<no_trans,4>` | one `tsm.ld.ncom.b32x4` | 0 | 32 /0 |
| `ldmatrix<trans_16x16b16,4>` | one `tsm.ld.ncom.mt1616.b32x4` | 0 | 32 /0 |

These match the instruction families used by FLA. Compile-only reachability
does **not** prove lane/register ordering, numerical behavior, bank behavior
or a complete-kernel speedup. Those are explicit next gates. The existing
actlize SWZL atom and disabled interfaces have not been changed.

Some unsupported PTX can lower through slow compatibility sequences. A
compiling source instruction is therefore not sufficient admission. Inspect
the actual target body, transfers, barriers, registers and spills; distinguish
compatibility expansion from generic address arithmetic. Prefer the verified
native SDK interface before restoring an unproved inline-PTX spelling.

## Chosen architecture

Use an independent `fla_aligned` namespace/path, not more conditionals inside
the current mixed experimental kernels. Proposed modules:

- `include/gdn_qsa/ppu/fla_aligned/`: ownership/layout plans, typed operand
  and accumulator views, vector copies and the proved native load adapter.
  No dynamic layout decomposition in the MMA inner loop. Share these maps
  with exhaustive tests anchored to native traits, not a parallel host model.
- `csrc/gdn_chunk/gdn_fla_prepare_ppu.cu`: prefix, KKT+solve and W/U kernels;
  each has an explicit resource/lifetime budget. Keeping KKT+solve fused
  avoids a large FP32 lower-matrix round trip. W/U separation permits its
  own warp/register schedule and coalesced publication.
- `csrc/gdn_chunk/gdn_fla_state_ppu.cu`: a recurrent state kernel with FP32
  resident state and tile-address invariants hoisted outside recurrence.
- `csrc/gdn_chunk/gdn_fla_output_ppu.cu`: QH and QK scheduled together,
  followed by gated causal PV, without the current two-panel output exchange.
- A thin C++ launcher/workspace contract: five math kernels in one stream,
  no Python between stages, no host synchronization or decay heuristic.
  Allocation-owning public API and prepared-workspace entry remain separately
  measurable. Defaults/original routing remain unchanged.

Internal workspace may use the reference's token-major W/U/Vnew and
`[B,NT,Hv,K,V]` snapshots; it is not an external weight artifact. Its strides,
units and allocation extents must be declared once and consumed by every
stage. All required upper-triangle/tail zeros must be written explicitly:
omitting FLA's fill launches is valid only if our producer does that work.
No alias may cross a live producer/consumer interval.

Implementation order is native-load/layout proof, state, output, then split
prepare. Each stage can be compared independently before composing the whole
new path. This is one structural rewrite delivered in verifiable stages,
not several untracked whole-kernel rewrites at once. Do not spend a box run
on the superseded address-only prototype.

## Arithmetic is not free to change

Retain BF16 boundaries, FP32 state, tails/GVA and initial/final-state semantics.
FLA source uses log2-prefix plus exp2; our admitted path uses natural-log
prefix plus expf. Its output scale expression and inverse block composition
also differ. Do not copy those expressions and call it a layout-only change.
Keep the admitted evaluation/reduction order for the structural delivery
port and require raw equality to its control plus the independent2% oracle.
Any necessary arithmetic change must be separated and declared, not hidden
under that raw-equality contract.

In particular, current inverse merges retain TF32 high/high plus two residual
terms (98,304 native MMA vs FLA32,768). Do not silently remove them to match
FLA's instruction count. This precision cost remains separately accounted.

## Performance goal and measurement contract

The first goal is to approach the reference's **GPU-stage execution**, not
declare victory from its Python dispatch cost. Current matched math anchors
sum to215.982 us; all seven FLA kernels including fills sum to222.482 us.
They are profiler anchors, not a promised public-API latency. The new path
must also beat the same-run current352-us-class public API; a faster kernel
with a slower complete call is not a production win.

Record both:

1. Balanced complete public-API event spans for both g=-0.1 and-1.0, with
   original/scalar/incumbent496/shared1520/FLA retained as applicable. No
   allocation or required kernel omitted from one side only.
2. Same-call kernel timeline: kernel durations and launch gaps separately.
   The current352.192-vs494.490 API result and347.292-vs222.482 ACU sums are
   different protocols. Their difference is NOT measured Python overhead.
   Graph/preallocated timing is additional evidence only when equally
   supported and applied to both sides; it must not replace public-API timing.

Native codegen must show the intended operand path, no spill, useful work
unchanged and materially reduced delivery overhead. Device gates keep16
cases, independent oracle, raw bits where arithmetic is unchanged,8 repeated
launches, GVA/tail/nonzero initial/output-only checks. Keep loss/UNRESOLVED
results. No selector promotion from compilation or static instruction counts.

## Saved checkpoints

The abandoned-as-next-handoff address-only prototype is preserved at branch
`wy-state-operands-20260924`, commit794b2e7, not merged into `ppu-backend`.
It compiled and passed its coordinate gate, but its Python family, complete
binary gate and regression seal are unfinished. It is not box-ready. Its
static footprint improvement does not certify a device speed result.

Rewrite feasibility evidence is under
`/workspace/gdn-fla-rewrite-evidence-20260924/`. The native probe is the only
new compiled component in this checkpoint; the five-stage path above remains
to be implemented. No PPU device code was executed locally.
