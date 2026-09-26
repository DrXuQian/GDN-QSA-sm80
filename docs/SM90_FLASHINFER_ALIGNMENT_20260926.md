# Exact FlashInfer pipeline alignment, before another speed claim

Control: S24 `bc3c154` / `06b7471`, H800 binary `43d62a8...`. Reference:
FlashInfer `5d9f8c8d97fa53e22952ce8672f475d235f07478`, the actual loaded
no-CP SM90 cubin, not its slower auto-CP dispatch. Same B1/T2048/Hqk16/Hv32,
D128/C64. No configuration sweep has been completed: S1–S33 are implementation
experiments on one geometry, not an enumerated optimum.

## Source and native difference ledger

Pinned reference paths below are relative to
`flashinfer/gdn_kernels/delta_rule_dsl/`. C++ paths are relative to
`csrc/backends/sm90/`. The source proves scheduling dependencies; static SASS
counts do not prove their elapsed cost.

| Axis | S24 C++ | Fastest measured FlashInfer | Evidence / disposition |
|---|---|---|---|
| Work ownership | One CTA per B×Hv;512threads, loader+aux+2state WGs | Same | `delta_rule_sm90.py:51–60`; kernel launch/types match |
| Registers per role | 24/104/192/192 | Same | Actual USETMAXREG; lowering state to184 serialized WGMMA in S30, rejected |
| Chunk / operand / accumulator | C64,D128; BF16 products, FP32 accumulator/state | Same | Actual HMMA/HGMMA types, independent CPU O/state gates |
| Inverse | FP32 8×8 diagonal, FP16 blocked8→16→32→64 | Same | `collective_inverse_hmma.py`; no different mathematical algorithm established |
| Q/K/V/O stages | **2/2/1/1** | **2/3/2/2** | C++ mainloop traits versus `delta_rule_sm90.py:164–173`; not previously aligned or swept |
| alpha/beta stages | **2/2** | **5/5** | Same source comparison; changing issue order alone did not test this topology |
| Loader issue order | Q,K,V | K,Q,V | `load_qkv_tma:486–589`; S17 alone unresolved on older shallow stages |
| Auxiliary publication | Acquire KK+QK; store both; inverse; commit both | Inverse+commit KK, then acquire/store/commit QK | `run_aux_loop_body:1519–1607`; S20 alone lost on older shallow stages |
| Scalar alpha consumers | aux128+state256+own producer32=416 | aux128+state256=384 | Reference `kernel:2149`; S33 removal alone unresolved |
| Beta consumers | aux128+state256=384, although scalar state no longer reads beta | aux128 only | Reference `kernel:2154`; part of old S13, not a new hypothesis |
| State matrix schedule | O1→SK→NewV→O2→KV, ordered WGs, commit/wait0 each | Same | `compute_loop_body:862–1008`; do not call FlashInfer wait-free |
| First-chunk specialization | Separate single-chunk and multi-chunk first bodies;4state clones | One first body with B=min(T,64);3state clones | `run_state_math_role:1257–1378`; whole no-initial kernel112vs100HGMMA sites includes an unexecuted clone, not12extra dynamic MMAs |
| NewV FP32→BF16 operand | Scalar conversion during retile | Vector conversion before operand use | `helpers.py:400`; actual no-initial body128 scalar F2F sites vs0. Old S9 removed the sites but was unresolved on~180us S7; not proof it cannot matter after S24 |
| Output retirement | TMA commit then wait0 before slot release | Same | Both `collective_store_tma`; an O2-stage ring still allows producer lookahead, not unsafe reuse |
| Gate input/math | Natural-log g, standard exp2f; cached relative factor | alpha multiplier, log2(alpha+1e-10), fastmath exp2; relative factor in state | Deliberately retained numerical contract; direct port is not raw-bit equivalent. Adapter excluded and separately measured as already registered |
| Shared storage not consumed | Original vector-KDA scaled-Q/K and alpha-last allocations retained | No such duplicate scalar allocation | Capacity cost, not repeated traffic; do not claim byte traffic without accesses |
| Tail ABI | Fixed-batch guarded tail; four dtype/initial bodies | Varlen/index/checkpoint capable | Not the full-chunk target's arithmetic bottleneck; preserve admitted tail behavior |

The stage-depth mismatch should have been checked before repeated local inverse
edits. S31 lowered diagonal selectors but lost; S32 removed16 full-chunk sites
but was unresolved; S33 duplicated part of the earlier S13 direction in a new
S24 context and was unresolved. None establishes the remaining~8us cause.

## Next modification: one coherent reference-delivery profile (S34)

User explicitly requested implementation after closing this ledger. Match
Q/K/V/O=2/3/2/2, alpha/beta=5/5, K→Q→V loading, KK-ready-before-QK-acquire,
scalar alpha384 / beta128 consumers. This is a **coupled reference profile**,
not attribution of any speedup to one knob and not a configuration sweep.
Keep S24 arithmetic, prefix/exp2 semantics, FP16/BF16 boundaries, inverse,
state matrix order, register allocation, and output-store retirement. Keep
unused storage in this first comparison to avoid mixing allocation reclamation.

Add explicit O/alpha/beta stage options alongside existing Q/K/V options so
future enumeration instantiates actual types. Print actual stage counts,
shared bytes and participant counts from those types; compilation alone is not
proof of legal resource occupancy. Native matrix/math/completion work, all14
CPU/parent fingerprints, overflow byte checks,8repeats/everycapturedoutput and
complete-forward nsys remain required. Same immutable S24 and fastest library
paths are controls. No SM80/default-route/native-PPU performance change.

After this comparison, a bounded configuration sweep should enumerate only
implemented stage axes, report the entire legal/rejected denominator and price
shared/register occupancy together. Chunk size and state-WG count require new
layout/math admission first; they are not currently legal sweep axes.

13:49 result: S34 is raw-admitted but **loses**128.8165[127.425,129.601]us
versus S24120.704[120.129,121.761]; fastestFI113.089us. Deeper rings and
reference publication/order together are not a speed admission. No defaults
change. Whole no-initial footprint6656->6560, but spill bytes20->48; do not
ascribe the loss to any one part without an ablation. This complete profile
is retained for the future config sweep, not called the optimum.

The remaining state-loop clone and conversion differences above are native
code-generation differences in **writing the same computation**, not a new
GDN algorithm. The next bounded source-alignment arm S35 tests these together
on S24, without adopting losing S34's delivery profile. It must preserve each
chunk's valid rows (first min(T,64), final tail), RNE and exact parent output.
