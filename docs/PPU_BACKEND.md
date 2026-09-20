# PPU backend

The PPU backend ports the algorithm and scheduling of this repository; it does
not reuse the older one-CTA-per-head implementation from `actlizeLA`.

The backend contract is:

1. per-chunk preparation remains parallel over `(batch, value-head, chunk)`;
2. each superchunk publishes one affine transfer `(A_g, B_g)`;
3. the transfer scan composes `(A_2, B_2) o (A_1, B_1)` as
   `(A_2 A_1, A_2 B_1 + B_2)`;
4. replay is parallel over `(batch, value-head, group)` and is the only stage
   that walks the chunks of one group serially.

`include/gdn_qsa/ppu/superchunk_schedule.hpp` is shared by the host oracle and
the device backend. The local oracle exhausts batches, tails, GVA ratios and
group sizes. It requires every chunk and every global group/lock identifier to
be visited exactly once. Dropping the final group and replacing the global ID
with a group-local ID are mandatory red controls.

The PPU device primitives come from the pinned `third_party/actlize` submodule
(`ppu-w4a16-dev`). NVIDIA SM80 remains the numerical and algorithmic reference;
the PPU backend must not change its sources, public results, or dispatch.

## Admission sequence

- host scheduler coverage and negative controls;
- exact PPU0010 operand/output coordinate ownership, independently anchored;
- hgcc compile/link of the exact shipping specialization;
- device correctness against the same inputs as the SM80 implementation;
- ACU verification of BF16 MMA opcodes, register spills and stage timings.

No PPU performance claim is admitted before all five steps close.

## Current implementation boundary

The PPU implementation now contains the complete forward path:

- `gdn_pipeline_ppu.cu`: per-chunk prepare, superchunk transfer construction,
  and parallel group replay;
- `gdn_scan_stage2_ppu.cu`: the inclusive affine scan over superchunks;
- `backend.h`: a C ABI with caller-owned workspace and no hidden allocation.

The triangular solve follows the repository's torch authority exactly: it
solves the unit-lower system `(I + L)x=b`.  This wording is intentional.  Some
comments in the historical SM80 source call the alternating Neumann expansion
`(I-L)^-1`, while the executable expansion and `torch.solve_triangular`
authority both require the negative lower-triangular recurrence.

PPU0010 has no admitted CUDA plain-`ldmatrix` compatibility contract, so this
first backend gathers logical operands directly into the real PPU MMA fragment
map.  The generated binary contains `v.mma.f32.bf16.m16n16k16`; it does not
pretend that the SM80 shared-memory byte map is portable.  This is a
correctness-first delivery seam.  ACU decides whether a later swizzled
shared-memory specialization is worthwhile.

The exact dynamic shared-memory ledger is host-visible and compile-time bound
to the shipping structs: prepare 18,048 bytes, group 127,488 bytes, and replay
95,232 bytes.  The local gate prints the same three values so drift is visible.

## Local proofs

`scripts/verify_ppu_backend_local.sh` establishes five independent facts:

1. exhaustive superchunk coverage, scan predecessor ownership, and global IDs;
2. real PPU0010 A/B/C fragment ownership, with swapped/rotated negative maps;
3. the complete direct-recurrence = prepare/transfer/scan/replay algebra over
   tails and group sizes, with inverse-sign, scan-order, and replay-prefix
   negative controls;
4. the public header is a real C ABI, with fixed field size/order and function
   signatures;
5. compilation of the exact shipping `.cu` sources with `hgcc -arch=ppu_10`.

Device execution remains a separate admission step and is run by
`tools/run_ppu_gdn_backend_box.sh`.

## Supported v1 surface

- forward only, BF16 inputs/outputs, `D=128`, `C=16`;
- zero initial recurrent state; optional BF16 final state output;
- grouped-value attention when `value_heads % qk_heads == 0`;
- runtime superchunk size (`group_chunks`) and arbitrary sequence tails;
- caller-owned workspace, queried before launch.

Initial-state input, other head dimensions/chunk sizes, backward, and a
performance-tuned PPU shared-memory delivery path are deliberately not implied
by the v1 symbol.
