# PPU original-structure port

updated-at: 2026-09-24 08:20:23 UTC
working-on: V16 ACU review closed; next design V32/8warps with separate mandatory BC work
blocked-on: no analysis blocker; next device implementation not started; routing unchanged
last-commit: 5a3ebc9 (captured source and local handoff)

Upload4e66b340028d2a91abd5e671ed4b7a4d7a09aabc2388041ec11120c8cb5fd92d.
603files/602hashes/127sources match5a3ebc9. Same-binary/fixture/device and
30 V16 RAW-BIT cases x8 verified; both gates numeric, g=-1 only captured.
Complete ACU sums residual226.02824 /V16 224.41118 /FLA225.67471us;
state124.68529->122.12588us, achievedwarps7.08->14.12. All15kernels1.700GHz.
Near-parity, no stable win/promotion;1.5x goal150.44981us remains NOT_MET.
KVD->TSM112->224MiB; logicalK/P96->192MiB explains96of112MiB increment.
Residual16MiB consistent with V16 transaction granularity, not per-plane proven.
ReadBC2.883584M->3.932160M, writeBC4.046848M->3.981312M; transposed shared
loads+50%, usefulMMA655360 unchanged. Per-PC opcode sum+5.90%, distinct PU
counter+8.98%; do not mix them. All15 per-PC sums close, omitted-PC plants red.
Extra1024 conversion is a zero-constant V16 prologue, not a rounding change.
No claim BC alone explains lost speedup; ratios are not wall-time shares.

Next preferred design V32/grid128 with256threads instead of128: preserve
shared45568B and112MiB logical input load while doubling launched warps.
Native-CLayout host probe proves40960output/K-atom cells perCTA equal and
three omitted-warp plants red. Device codegen/numerics/performance NOT_RUN.
BC is mandatory separately:4/8warps x old/BC layout bounded2x2; no automatic
promotion. Adjacent BF16 result columns occupy different lanes, so packed
stores need proved native exchange/paired layout, not a free reinterpret cast.
No kernel/default edits this turn. Full report docs/PPU_GDN_V16_ACU_20260924.md;
replay/evidence /workspace/gdn-residual-v16-acu-analysis-20260924.

## Previous local residual-delivery handoff

Three independent candidates from old residual: next-inverse prefetch,
shared-register operand prefetch, V16 geometry. No combination or math change.
V16 at B1/S2048/Hk16/Hv32/D128/C64 gives128->256CTAs, still128threads,
registers242->98 and shared45568->35328B, zero stack. K/P duplicate loads
increase; this is a compile resource result, NOT an occupancy or speed claim.
Inverse-prefetch244regs,16 UPDATE MMAs before next wait. Operand244regs;
control already pipelines shared loads, UPDATE lookahead unchanged, KH/PR
reordered. Three arms opt-in; no new default production selection.

Final complete local rerun PASS:15/15 compiled host tests,80 Python contracts,
7 HGCC dialect tests,45 WY+61 residual algebra,305 original controls,
26 WY images and15 original images. Old23/23 WY native instructions+operands
identical. Missing wait/retirement/fragment/grid/K atom, serialized-prefetch,
missing ACU and wrong selector negative controls all fail as required.
Local evidence /workspace/gdn-wy-residual-prefetch-evidence-20260924;
verify_local.sh and final-local-complete.log are the final authority.
Device library f0d918f03877d79806531f52258881ad21e8599459ecc9c438d702e22fed93cc.

Command from repo: DEVICE=0 JOBS=16 CANDIDATE=residual-v16
bash tools/run_ppu_residual_delivery_acu_box.sh. Matching installed SDK,
old gates then30 candidate cases x8/raw-byte equality, both-gate numerical
admission then sequential residual/V16/FLA site-ACU captures (4/4/7 expected).
API_TIMING=NOT_RUN. Upload printed /workspace/.../acu/acu.tar.gz.
Optional independent arms CANDIDATE=residual-operands/residual-prefetch.
BC elimination explicitly NEXT in docs/PPU_GDN_RESIDUAL_DELIVERY.md:
per-PC bank mapping, paired AIU/SWZL writer-reader proof, no layout change now.
Skills updated and pushed Quactlize d3f7b88; unrelated dirty files preserved.

## Prior residual ACU analysis (before current delivery candidates)

Latest upload d1f0c05aacf91bb01005475adcb0f8a334f72fe9644f6483a2e7a5843d001151:
591 files/590 checks/115 source files verify against1a843eb. Tracked sources
clean; one untracked wy-g-1.0.report.acurep is recorded, not a build input.
Same PPU-ZW81072CU, all16 kernels1.700GHz; capture uses requested site ACU.
Control/candidate/FLA inventories5/4/7, complete kernel sums255.22647 /
230.58412 /224.30883us. Candidate2.80% slower than FLA;1.5x requires149.53922us.
Thirty residual device cases x8 repeats PASS, both gates numerically admitted;
g=-1 profiled only. Native report reimport/per-PC sums close all16 kernels;
omitted-PC negative fails. Host parser is local SDK, not the site capture tool.
Solve52.69412 vsFLA44.88118us: BF16 MMA81920 equal, TF32 MMA98304 vs32768
(explicit x3 residual products vs single TF32). Diagonal native interval has
5,013,504 executed instructions,196608 indirect ivreg reads and466944 pipe_flush.
Not all moves are layout conversion; no blanket attribution of waits to barriers.
Residual state's129.64353us replaces WU+state, not only FLA's90.27529us state;
FLA combined126.47588us. Source/binary verified, no kernel edit or promotion.
State grid128 x128 gives7.09warps/CU (grid ceiling7.111; resource limit16),
same asFLA7.10; common underfill, not unique causal explanation. Shared-bank
conflicts6.93M vsFLA8.55M do not establish a conflict-dominated gap.1.31072M
BF16 converts close on H/residual/Vnew/scaledV, with no duplicate-value mystery.
Full report docs/PPU_GDN_RESIDUAL_ACU_20260924.md; retained artifacts and replay
/workspace/gdn-residual-acu-analysis-20260924. Old WY16 cases+32delivery gates
verified as well. Skill records precision-scope and indirect-register lessons.
Read-only kernel investigation complete; proposed optimizations remain unimplemented.

Worktree /workspace/gdn-wy-residual-20260924; evidence/plan under
/workspace/gdn-wy-residual-evidence-20260924. Explicit new BF16 rounding
contract P[beta*(V-exp(g)*K@H)], no CP/reset/output fusion. Existing gated
inverse and output native bodies reused. Independent inverse allocation
prevents cross-V-CTA snapshot overwrite; padded old solve pitch retained.
Old22 kernel bodies and RAW-BIT delivery gates must stay unchanged.
SDK2.1.1 compile/link PASS:23 images, all22 controls native-identical;
new242regs/stack0/45568B/40MMA static sites.61 CPU algebra cases PASS,
max output/state error0.008878/0.004194,4semantic negatives red. l020
real-trait ownership6144values/product163840cells/group-tail1064960cases,
6negatives PASS.9source+4native new negatives red. Three-arm ACU contract
explicitly distinguishes new rounding; old delivery RAW-BIT gate retained.
Final full rerun PASS:12/12 CTest,74 Python contracts,7 compiler dialect
tests,45 old WY algebra and61 new algebra,305 original source controls,
23 WY images/22 native-identical controls,15 original kernel images.
Runner: tools/run_ppu_residual_acu_box.sh; old RAW-BIT first,30 residual
device cases,8-repeat both-gate admission,then same-binary pipeline/residual/
FLA direct site ACU (default g=-1; GATE=-0.1 optional). API timing NOT_RUN.
Local device library69145cfe11997f1fb05ad4519111eadb9e50efa6b56a06b4da65e363aa2ea3c9.
Doc docs/PPU_GDN_RESIDUAL.md; reusable lesson published Quactlize65528be.
No performance or PPU numerical PASS inferred from local proofs.

Upload SHA256 a735ad2ad07a9390358664dbe88a1ad0036fb3961d8d7ae0007e0c3d3c30bc6f:
580 files/579 hashes verify; clean8eaaa56, all17 kernels at1.700GHz,
control/candidate/FLA5/5/7 kernels. Device16 cases x2 deliveries x8 repeats
RAW-BIT+2% oracle PASS; both gates numerical admission, g=-1 capture only.
ACU sum control271.48295 -> candidate252.19059 us, FLA224.77529 us.
State122.19647 ->105.33471 us; unchanged-stage variation will be separated.
Native re-import/per-PC totals close all17 kernels; omitted-PC plants red.
Uploaded native CFG retains12 PROJECT/16 UPDATE overlap; same-opcode wait
relocation is red. State instructions-0.91%, sync-stall avg-48.55%, commit
stall avg-55.53%; math/load/maintraffic/occupancy unchanged. Actual box232regs,
1956static instructions, not local230/1880. Unchanged-stage variation2.43060us
is not credited to new source. Goal requires149.85019us; another40.58% reduction.
Full report docs/PPU_WY_STATE_PIPELINE_ACU_20260924.md; evidence under
/workspace/gdn-wy-state-pipeline-acu-analysis-20260924. Keep opt-in62960;
no kernel/default/routing edits. Future box captures default to
/sim/eec/shared/junfu.qx/asight/bin/acu per user instruction, no implicit SDK/PATH
fallback. Explicit override remains recorded.69 host contracts PASS; actual
missing-site invocation returns rc1/INCOMPLETE before build/profile despite
available SDK ACU. Existing upload retains actual SDK ACU provenance. Local
report re-import only uses available local parser; no device execution.
Reusable lessons recorded via skill-creator in Quactlize88f4efe; skill validator
PASS. Code atom choice alone is not a tuned collective: WU equal MMA but1.716x
instructions. Next change must remove proven excess without altering numerics.

## Prior state-pipeline handoff (device-pending claims superseded above)

New worktree /workspace/gdn-wy-state-pipeline-20260924, parent5ec469f;
plan/ledger /workspace/gdn-wy-state-pipeline-evidence-20260924. Hypothesis:
stage current K/U under W@H and next W into its dead shared plane under
current state update. Same49408B storage, same arithmetic/FP32 state;
barrier changes require per-warp lifetime and native schedule proof. Keep
split30192/all21 old kernels; one new opt-in, no production routing change.
First real SDK compile/link: new state230 regs vs234 control, stack0,
same49408B.21/21 prior native instruction+operand sequences identical.
Actual CFG: K/U overlaps12 PROJECT MMAs before wait; next W crosses16 UPDATE
MMAs before next-loop wait.32 math sites retained; native barrier sites5->4
(4->3 per loop). Same-opcode-count early-wait negative red. Host lifetime
proof66 schedules/30,275,476 conflicting-pair candidates,270,532,608 tail
cells,9 plants red.11 source negatives red. Final complete regression PASS:
11/11 CTest,68 Python contracts,7 HGCC contracts,45 algebra cases,305
original structure controls;22 WY/15 original native images audited. New
same-binary capture contract tested, including missing/ignored delivery and
raw-bit drift negatives. Libraries/hash manifest and final-local.log preserved.
No device numeric/performance claim. One explicit62960 arm; default unchanged.

One command after pull: DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16
bash tools/run_ppu_wy_state_pipeline_acu_box.sh. Use the working SDK path.
Build+numeric admission precede direct sequential ACU split30192/new62960/FLA;
all kernels, API_TIMING=NOT_RUN. Upload printed /workspace/.../acu.tar.gz.
Docs: docs/PPU_WY_STATE_PIPELINE.md. Skill0c96847 retains CFG/async-lifetime
lesson, no unmeasured speed claim. Next separate candidate is W/U conditioning.

## Previous split-prepare device verdict

New upload /root/acu.tar.gz SHA256
9c1802c922dd18aa7262fb786e96b4d10b192484ee31ba3f9ead6931e541ad71:
575 files/574 hashes verify. Clean c2bdb5c, same device/input/loaded binaries;
all15 kernels at1.700 GHz, actual symbols/grids verified. Device16 cases x2
deliveries x8 repeats pass original2% oracle and scalar raw equality. Both
gates numerical admission PASS; performance capture g=-1 only, API not timed.
Native per-PC counts close all15 kernels; omitted-PC negative red in each.

ACU sums: incumbent304.99471 -> split269.30589 us (-11.7015%); FLA223.58294
us including two fills. Still20.4501% slower than FLA; same-run1.5x target
149.05529 us, another120.25060 us/44.652% reduction required. Prepare
136.34118 ->101.56589 us; WU KVD store amplification512->32 MiB removed.
Split adds inverse traffic; prepare DRAM reads24.27->40.91 MiB, not a claim
all traffic falls. State/output bodies and dynamic counts unchanged. State
120.93294 vs FLA90.40765 us is now58.5% of matched-math excess; next inspect
operand staging/wait schedule, then WU conditioning/shared roundtrip. No
kernel/default/routing edits; no local device execution or weak-gate speed claim.
Evidence: /workspace/gdn-wy-split-acu-analysis-20260924.
Report: docs/PPU_WY_SPLIT_ACU_20260924.md. All65 host contracts rerun PASS
with the prior task-local Python/Torch environment, GPUs hidden. Default
Python lacks Torch; no system installation changed. Phase/identity/resource
claims cross-checked with parsed native reports. Skill lesson6699c07 pushed
to Quactlize; unrelated dirty files untouched. Next implementation is not
started by that upload-analysis checkpoint. Its measured kernel SHA isc2bdb5c.

## Prior local handoff (device-pending statements superseded above)

User goal now1.5x FLA:222.86413/1.5=148.57609us. Not yet achieved.
State+output168.19412us already exceeds target; prepare split is a bounded
first step, not a promise to hit goal alone. Dedicated worktree
/workspace/gdn-wy-split-prepare-20260924 at8b64bd3; plan/ledger under
/workspace/gdn-wy-split-prepare-evidence-20260924. Preserve AIU13808 and
all old bodies; native paired loads, arithmetic/raw-bit and full-call gates.
First native compile21 images:18/18 incumbent instruction+operand sequences
identical. New prefix/solve/WU32/84/128 registers, stack0; WU256threads,
41472B shared, solve49664B. Local shipping-helper gate:32768 prefix rows,
4194304 conditioned values,16384 W/U outputs,49373184 inverse scratch cells;
6 fault plants red. Source solve segment matches control exactly;9 source
lifetime/order/selector negatives red. Final complete gate PASS:10/10 compiled
host tests,65 Python contracts,7 HGCC contracts,45 algebra cases,305 original
structure controls.21 WY images/15 original images linked; old18/18 native
bodies identical; new6 binary negatives + changed-control negative red.
This does not establish device correctness or any speedup. Opt-in30192 only.

One command: DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16
bash tools/run_ppu_wy_split_prepare_acu_box.sh. Builds, device16-case gate
with8 repeats/raw scalar equality, then both-gate FLA numeric admission;
API_TIMING=NOT_RUN. Three direct sequential ACU captures:AIU13808,split30192,
FLA once(g=-1). All five candidate kernels count toward total. Upload printed
/workspace/gdn-wy-split-prepare-<sha>-<UTC>-<pid>/acu.tar.gz. No PPUProfiler.
Current candidate only changes prepare; state/output pipeline remains the
next structural step. Report gains AND regressions; no1.5x claim from compile.

## Prior verified ACU result (before split-prepare)

New archive gdn-qsa-acu-20260924T031047Z-2637570.tar.gz:570 files/569 hashes
verify. Clean8340fc7 measured binary; b75b25d collector; all3 arms same UUID,
input, precision and1.700 GHz. Exact16 cases x4 deliveries,8 repeats and
2 gates x7 roles x14 API samples verified. Native per-PC sums close all13
kernels; dropping an executed PC fails all13.62 host contracts PASS.

User explicitly chooses ACU TIME ONLY for next optimization. Current
prepare/state/output135.854/121.071/47.123 us vs FLA83.428/90.198/42.748;
FLA fills6.491 us separately. Total304.048 vs222.864 us:36.4% slower.
Old control346.278 us; real state/output gains17.0%/27.5% with45.0%/46.1%
fewer dynamic instructions, equal MMA counts and similar traffic/warps.
Prepare is60% of remaining matched-phase gap. Next is structural prefix/
KKT-solve/WU resource separation and vector publication, not a claim one
store edit closes it; then state load/wait/operand schedule. Output only5%.
Keep native AIU pair, FP32 state and TF32 residual precision; no kernel or
route changed now. Full report docs/PPU_WY_AIU_ACU_20260924.md; revised
design docs/PPU_WY_FLA_REWRITE.md. Evidence /workspace/gdn-wy-aiu-acu-analysis-20260924.
Quactlize skill9a51415 records measured instruction/time and sync-counter
limits; unrelated dirty files were not staged. No local device execution.

## Prior collector/user-table checkpoint (superseded by verified upload)

Current helper adds optional --wy-control prepare-rows-shared alongside
--wy-delivery aiu-state-output --wy-run EXISTING_COMPARISON --gate -1.0.
No rebuild or sweep; three independent preflights then three sequential
captures; FLA once, one /workspace tar. Both WY DSOs/inputs/device/raw bits
must match, with separately checked delivery choices. No kernel/router edits.
Found old AIU comparison metadata delivery_ab=false; collector now binds
actual arm/mask rather than that summary flag. Old JSON remains immutable;
future benchmark summary fixes the AIU family flag. Missing/wrong masks fail.
62 Python contracts and7 HGGC contracts PASS;9 three-arm fault plants red,
old two-arm lifecycle preserved. Shell syntax/diff checks PASS. This is
host synthetic validation, not a profiler or device timing result. Evidence
/workspace/gdn-wy-aiu-acu-evidence-20260924; command docs/PPU_GDN_ACU.md.

User reports correctness PASS after the8340fc7 handoff. This is a user device
report, not an independently hash-verified receipt: no per-arm log, case
counts or loaded binary hash supplied yet. User now supplies API medians:
AIU both318.564/state332.100/output339.696 vs shared354.364 us;
original422.000, scalar WY716.770, FLA492.188 us. Both is10.103% lower
latency than shared and35.276% lower than FLA, DESCRIPTIVE_MEDIANS_ONLY.
Gate/shape/device/SHA and sample ranges are not included in the table;
do not infer both gates passed speed admission. The pasted ailu spelling is
treated as a label typo, not a verified runtime role. subject=wy/FLA-WINS
is scalar716.770 vs FLA492.188, not a verdict against the AIU candidate.
The runner separately emits WY delivery verdict for every candidate/control.
Next paired ACU of incumbent/AIU-both/FLA, without subtracting profile sums
from API spans. Default routing unchanged; no local device execution.

Worktree /workspace/gdn-wy-aiu-20260924, branch wy-aiu-pair-20260924.
Artifacts /workspace/gdn-wy-aiu-evidence-20260924, explicit plan/ledger.
New state/output delivery uses full-height32/64-column AIU cubes and matching
SWZL descriptors; no16x16 software base decomposition for input staging.
Current prepare shared-row control, original/scalar/old bodies and arithmetic
remain. Three opt-in cells (state/output/both), no default routing change.
Local gates:9/9 CTests;58 Python contracts;7 HGGC contracts;45 algebra cases;
305 original source controls;18 native images with16/16 old bodies unchanged.
Layout:28,416 values and17,633,280 descriptor/tail cells;6 negative plants red.
State static instructions2300->1875, v2s52->34, registers242->234; output
1725->1442, v2s68->52, registers98->94. Both stack0; not latency predictions.
Native writer l0 vs linear l1 confirmed with same-source compile control;
wrong writer/one wrong reader/missing image/link/control mutation all red.
Full Python3.12/Torch2.9 bindings for WY AND original now compile/link with
real SDK2.1.1 device libraries. Task-local CPython/dependencies reused the
existing CUDA Torch wheel; no global interpreter or installed Torch changed.
Ready command: AIU_AB=1 DEVICE=0 PPU_SDK=/usr/local/PPU_SDK JOBS=16
bash tools/run_ppu_wy_fla_box.sh.7 roles,14 balanced samples,2 gates;16-case
raw-bit/oracle/replay/GVA/output-only admission first. Device correctness
now USER-REPORTED/PASS; API medians supplied, no envelope or kernel-speed
admission yet. Detailed result scope is recorded above and in the AIU doc.
This closes the delivery candidate, not the entire five-stage FLA rewrite.

## Pre-implementation diagnosis

User distinction confirmed: AIU.swzl and matching ld.swzl avoid a software
unswizzle, but current WY instead uses per-thread async_copy16 with software
swizzled addresses. Our port1af3d5c introduced that choice; upstream aa04271
uses NVIDIA cp.async/LDSM. Not an upstream PPU design or a numerical failure.
Local l015 native control: SWZL fixed/warp-base v2s=0/1; NCOM=0/0. The
move's SGPR is the SWZL load base, not post-load data. Four negative plants
red (extra/missing move, wrong consumer, missing cell). Actual SDK compiled;
numeric/performance NOT_RUN. Ordinary v.mov remains separately classified.
Quactlize skill90d771c records matched AIU/SWZL producer/consumer contract,
explicit manual-copy alternatives, and register-born intermediate exception.
No production kernel, selector, numerical criterion or submodule modified.

## Structural rewrite feasibility checkpoint

User redirected from micro-tuning to structural alignment. Architecture:
prefix, KKT+solve, W/U, state, output; pure C++/actlize, forward only, retain
current mask1520/original and arithmetic contract. No routing changes.
Read docs/PPU_WY_FLA_REWRITE.md before implementing. Local exact l014 probe
compiles to one NCOM / mt1616 NCOM load,0 v.mov.v2s,32 vregs,0 stack per
body. Three validator negatives pass. This is not device mapping admission
or a speed result. Evidence /workspace/gdn-fla-rewrite-evidence-20260924.

Address-only prototype preserved in unmerged branch
wy-state-operands-20260924 at794b2e7. Native library and l013 host gate pass;
Python family, extended binary audit and full regression seal NOT CLOSED.
That WIP branch is not box-ready; do not run it as the next experiment.

Skill update pushed to Quactlize develop5cb8583: compatibility PTX lowering
must be checked in native code, API/kernel timing scopes kept separate,
checkpoint lessons retained with evidence and limits. No unrelated files
in that dirty worktree were staged.

New upload gdn-qsa-acu-20260922T225857Z-797428.tar.gz:553 files/552 hashes
verify; clean eed5ba2 binary origin, shared1520 actual kernel, same physical
UUID/input/runtime, all kernels1.700 GHz. Native reimport metrics agree;
per-PC sums close and omitted-PC negatives fail for all10 kernels.
Preceding16 cases/80 candidate admissions/8 repeats and2 gates x8 roles x16
samples verified. Weak shared354.666 vs496358.918 remains UNRESOLVED;
strong352.192 vs356.590 wins. Both shared/FLA API comparisons win. Raw role
is prepare-rows-warp, resolving the pasted singular spelling.

Profiled prepare/state/output135.350/145.466/66.476 us vs matched FLA
83.107/89.490/43.385; two FLA fills6.500 us separately. Profile sums are NOT
API spans or a host-overhead measurement. State has same grid128/block128,
7.09 active warps/CU,524288 BF16 MMA and98 MiB KVD write traffic as FLA,
but2.165x measured instructions. Output occupancy58.75% vs35.48% and2.352x
instructions. Prepare1.758x instructions, W/U scalar-write KVD16x
amplification; explicit TF32 residual precision3x MMA kept separate.
Next target state shared fragment/base-address overhead, then prepare W/U
publication, then output delivery; not another grid-only adjustment.
Report docs/PPU_WY_SHARED_ACU_20260922.md. All 24 host ACU contract tests
pass. No kernel/default/routing edits in the analysis checkpoint.

## Previous capture handoff (completed by the upload above)

Next command in docs/PPU_GDN_ACU.md. Existing collector already supports
prepare-rows-shared and binds its exact measured role/binding/device library;
no rebuild, default routing or runner change needed.24 host ACU contract
tests rerun PASS. Source expectation: shared prepare<1> grid1024/128 threads,
state_ab<true,true> grid128/128, old tiled output grid1024/256. Capture strong
g=-1.0 plus FLA sequentially, all stages, direct external acu and automatic
tar. Current helper SHA remains distinct from reused kernel provenance.
Actual profiling NOT_RUN locally. No inference from old profiles or API
sum subtraction. User can paste the exact artifacts path at command prompt.

User-pasted g=-1.0: incumbent496 356.590 [355.552,357.508] us; shared1520
352.192 [351.240,353.872]; warp2544 354.350 [352.660,357.468]; FLA494.490
[478.012,522.352]. All8 roles have16 finite samples and matching recomputed
printed medians. Shared vs496 CANDIDATE-WINS:4.398 us /1.233% lower latency,
1.680 us disjoint-envelope margin. Warp vs496 UNRESOLVED; shared vswarp also
UNRESOLVED, not a proven cache-mechanism ranking. Shared/FLA median ratio
1.404x is full-public-API, not kernel-only. Original426.348 (BF16 final state),
scalar WY705.266; final subject=wy verdict refers only to that old scalar.
The pasted warp role spells row (singular), unlike the source's rows; do not
treat this excerpt as a hash-verified raw JSON. No new kernel/default/route
changes. Shared is a strong-gate experimental incumbent only; weak unknown.

## Prior local prepare-row handoff

Task worktree /workspace/gdn-wy-prepare-rows-20260922; pre-edit plan/evidence
/workspace/gdn-wy-prepare-rows-evidence-20260922. User-reported strong medians:
prepare-address356.760 vs state-both379.088 us, disjoint envelopes; output
380.480 / stage-both357.674 add no established benefit. FLA487.472,
original425.572. Full identities/weak results not received. Keep old output.
Next caches only expf(prefix[row]), not beta*exp or pairwise gate differences;
shared vs warp modes independent, no change to TF32 solve or FP32 state.
Both native variants compile/link, shared/warp regs86/84 stack0. All14 old
native instruction+operand sequences IDENTICAL. Actual backward-branch CFG
proves row exponents moved outside conditioning; shared store+publication
must precede consumption. Same-opcode-count negatives move exponent back or
publication after consumption and fail. Complete post-edit seal rc0:
8/8 CTests,63 Python contracts,45 CPU algebra cases+5 negatives,305 original
controls preserved; original and16-image WY native libraries compile/link.
21 native negatives red, including same-count wrong-loop/wrong-barrier.
Evidence sealed-local-r1.log SHA256:
d85964093051c1dacd7b98f91e09b863c4be0cb747d9f95442a2bba5592d0ea5.
WY DSO SHA256:
0e55fc5236df8ea5b9ec728d839e64021ed10b07689dde7c8a46da6247abeee8.
No new device result. Eight-role same-binary command is ready; full binding
and raw-bit/replay/speed still require box. No routing/default changes.
Handoff plan: docs/PPU_WY_PREPARE_ROWS.md, PREPARE_ROWS_AB eight-role family.

## Prior stage-address handoff

Task worktree /workspace/gdn-wy-stage-address-20260922, registered plan and
resume /workspace/gdn-wy-stage-address-evidence-20260922. User-pasted strong
g=-1.0: mask48 436.296 us, address401.968, gates413.482, both378.696;
original424.998/FLA490.140. All three gains have disjoint envelopes against
mask48, both beats singles; weak gate and complete identities not supplied.
No routing promotion. Next only changes prepare/output address loops, retains
TF32 high/residual precision and all12 prior native instruction sequences.

Final local pass: 7/7 CTests,60 Python contracts,45 algebra cases+5 negatives;
new gate28,672 coordinates,
1,492,480 vector/tail conditions,12,288 prepare element iterations,96 complete
selector combinations and8 planted defects. All12 prior native instruction+
operand sequences IDENTICAL in the14-image real SDK build. New prepare/output
regs86/120, stack0; static copy sites16/14, unchanged BF16/TF32/exp body counts.
Output regs increased98->120: keep the losing possibility explicit, no speed
claim. Full unchanged Torch binding remains SKIP/environment locally (CPU
torch); box must build it and pass raw-bit/replay admission before timing.
STAGE_AB eight roles retain mask48/mask240 and all three ablations; old
experiment inventories/defaults/routing unchanged. Final post-edit seal rc=0:
/workspace/gdn-wy-stage-address-evidence-20260922/sealed-local-r2.log.
All14 native images linked,17 codegen negatives red as required. Final DSO
SHA256 8ef9494cbfc487e913738f4ba5b8c6f40b2ee3672e0e76a4fafd31fe9d63ad6f.
Command/criteria: docs/PPU_WY_STAGE_ADDRESS.md. No device speed admission.

## Prior local state-ablation handoff

Active worktree /workspace/gdn-wy-state-address-20260922, evidence/registered
plan /workspace/gdn-wy-state-address-evidence-20260922/docs/plan.md. New opt-in
mask112/176/240 arms retain mask48 and all prior controls; no auto routing,
prepare/output arithmetic, precision or tolerance changes.

Final local checks: 6/6 CTests, 57 Python contracts, 45 CPU algebra cases
and five algebra negatives PASS. New address gate checks 18,432 coordinates,
1,406,720 vectors and 131,072 gate-reuse values; six planted defects go red.
Real SDK2.1.1 compiles and links all12 native WY images; 12 binary negatives
go red. All9 old instruction+operand sequences are IDENTICAL to the same-SDK
parent build. New address/gates/both kernels: 238/234/242 registers, zero
stack, 18/3/18 static copy sites, 17/5/5 exponent sites; each retains32 BF16
MMA sites. Static footprint/regs increase for address is recorded, not called
a measured speed win. Native DSO SHA256:
d659a512fb3e194b66aa1bf91196e706e38268ce7c802ddcfc81ba56c0549a7a.
Post-edit rerun sealed in the evidence directory's sealed-local.log; rc=0.

Full Torch-binding local build: SKIP/environment (private torch2.9 CPU lacks
cuda_cmake_macros.h and CUDA torch libraries); no stubbed PASS. Box builds
that binding and must pass device raw-bit/8-repeat admission before timing.
No local device result. STATE_AB=1 keeps eight balanced roles including
original, scalar, mask48, all, three candidates and FLA at both gates.
Command and fixed verdict: docs/PPU_WY_STATE_ADDRESS.md. Waiting on device
measurements; do not promote a route from static instruction counts.

## Prior verified device evidence

New upload gdn-qsa-acu-20260922T080504Z-2605202.tar.gz: 542 regular
files, all 541 checksums and exact manifest denominator PASS. Source90eafeb,
empty source diff, exact loaded binding/library and preceding comparison
verified. Actual kernels scalar prepare<false> + tiled state + tiled output;
WY/FLA share inputs, UUID and 1.700 GHz measured CE frequency. No device run
or kernel/default/routing changes. Native reports imported locally using a
private compatible host runtime, without changing system libraries.

Weak API result now supplied/verified: pair443.810 us versus scalar720.480,
state458.826, all451.954, original952.442, FLA770.394. Pair wins against
scalar/state/original/FLA; pair versus all UNRESOLVED. Strong prior verdicts
confirmed. FLA API samples remain broad; median ratios are descriptive.
ACU prepare/state/output161.331/199.598/63.963 us versus FLA math stages
83.073/90.024/43.643 us (FLA fills6.504 us separate). All three still slower.
State has same128x128 launch, about7.1 active warps/CU, same524288 BF16 MMA
and98 MiB global-store request footprint, but34.386M versus10.040M total
instructions. Occupancy and store volume no longer explain the relative gap.
Measured per-PC opcode exports now close exactly for all ten kernels;
omitted-PC negatives fail for all ten. State W-copy loop alone is50 static /
6,537,216 dynamic instructions (19.11% of SASS sum), repeatedly decomposing
signed tile coordinates around one cp.async. State exp2 count278528 versus
49152 identifies a separate row-factor reuse target. Neither is a measured
microsecond attribution. Next: state copy/address-only ablation, then same-
expf row-factor reuse; preserve precision, recurrence and all stage controls.
Report: docs/PPU_WY_STATE_OUTPUT_ACU_20260922.md. All native-import/parser
evidence in /workspace/gdn-wy-state-output-acu-analysis-20260922. No API-minus-
ACU overhead subtraction. No implementation, criterion or routing changes.

## Prior pasted-result handoff (superseded by verified upload above)

User-reported /workspace/gdn-wy-fla-90eafeb-20260922T013928Z, g=-1.0:
state+output 439.486 us [437.684,444.076], scalar 710.210, state458.998,
all446.700, original478.898, FLA730.812 us. Pair wins versus scalar/state/
FLA under unchanged envelopes; pair versus all and original UNRESOLVED.
FLA range486.836–968.548 makes1.6629x descriptive, not stable speedup.
Keep all/original/default routing and precision unchanged. Full raw JSON/
binary/device identity not verified locally; weak not inferred from strong.

Existing capture supports --wy-run /workspace/gdn-wy-fla-90eafeb-20260922T013928Z
--wy-delivery tiled-state-output --gate -1.0, no rebuild. Expected kernels:
scalar prepare<false>, tiled state, tiled output. Both WY/FLA reports plus
full comparison JSON and identities join UPLOAD tar. All three stages must
be compared, not just state; no subtraction of ACU sum from API time.
Current local contract rerun ENVIRONMENT BLOCKED at import torch. Previous
23-contract admission remains historical; no new PASS claimed. No collector,
benchmark or device code changed. Report: docs/PPU_WY_STATE_OUTPUT_RESULT_20260922.md.

## Prior experiment handoff

Active worktree /workspace/gdn-wy-state-output-20260922, evidence and
registered plan /workspace/gdn-wy-state-output-evidence-20260922.
No C++/CUDA kernel, actlize, default or original routing edits. Mask48 uses
existing scalar prepare + tiled state/output. TILE_AB now has eight roles;
default 16 samples derives from inventory, old explicit14 rejected. The
pair compares directly to scalar, FLA, state-only, all-tiled and original.
Final 53 Python contracts (15 WY / 8 FLA / 23 ACU / 7 hgcc), 45 algebra
cases + five negatives PASS. Full CPU-mocked comparison visits all eight
roles, preserves raw-bit rejection even within 2%, and prints direct pair
comparisons. Wrong mask/omitted role/old14/cross-variant captures fail.
Unchanged five compiled host gates, 305 original controls and nine-image
SDK binary gate/seven negatives rechecked read-only. C++/CUDA/geometry/
actlize sources identical to a712a7d; no new device build or execution claimed.
Box: TILE_AB=1 DELIVERY_AB=0 SAMPLES=16 tools/run_ppu_wy_fla_box.sh.
Both gates -0.1/-1.0, same shape and raw-bit admission, sequential full-call
timing; new candidate wy-tiled-state-output. No new speed claim/promotion.

## Preceding user-reported strong result

User-reported run /workspace/gdn-wy-fla-df90c61-20260921T143917Z,
g=-1.0: scalar WY 705.644 us, tiled-all 443.362 us, FLA 485.896 us.
Tiled-all wins with disjoint observed envelopes against both controls:
1.5916x vs scalar, 1.0959x vs FLA (37.17% / 8.75% lower latency).
Original remains faster at 424.072 us, envelope [417.596,430.344] below
tiled-all [441.996,445.852]; keep strong/reset routing unchanged. Original
final state BF16 versus WY/FLA FP32 remains a comparison-scope distinction.

Prepare-only 765.836 us definitively loses to scalar; state-only 453.986 us
wins against scalar and FLA; output-only 689.352 us wins against scalar only.
All versus state-only improves by 10.624 us, also with disjoint envelopes.
One-stage deltas predict 497.886 us if additive, but all is 443.362 us:
the -54.524 us interaction is observed, its mechanism is not established.
Do not subtract prepare's isolated 60.192 us penalty from the all time.
Next discriminating cell is state+output-only (delivery mask 48); ABI admits
it but the Python benchmark currently has no named role for it. No new
kernel/selector/default edit made from this pasted result.

Reported raw-bit-versus-scalar checks PASS. Current raw JSON/identities have
not been received/hash-verified locally; weak g=-0.1 is NOT PROVIDED, not
inferred from strong. Legacy summary WY verdict refers to scalar only;
candidate verdicts refer to the tiled roles. Local evidence ledger:
/workspace/gdn-wy-tiles-evidence-20260921/DEVICE_RESULT.md.

## Prior local-only handoff (before the result above)

FINAL post-edit gate PASS: /workspace/gdn-wy-tiles-evidence-20260921/sealed-local-r2.log.
5/5 CTests, 48 Python contracts, 45 algebra cases + five negatives, original
305 controls preserved; original 15 and WY 9 real SDK device images linked.
9,516 vector ownership cases / 27 negatives; H exchange 8,192 reads,
34,816 per-output reduction traces (W/U separately counted), 256 selector
values / 27 valid combinations, eight new layout/order negatives. Tiled
prepare/state/output 160/232/98 registers, zero stack; state zero shuffle
opcodes. Seven binary negatives include missing tiled cross-TU launcher.
Only prepare controls change codegen after sharing their source-identical
arithmetic body; all four old state/output machine sequences remain identical
to the preceding local build. No claim of whole-control binary identity.
Keep same-new-binary scalar admission/timing, not historical timing subtraction.

Box: PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16 TILE_AB=1 DELIVERY_AB=0
bash tools/run_ppu_wy_fla_box.sh. Sixteen device cases and raw-bit versus
scalar plus 8 repeats before 7-role balanced timing at both gates. Defaults
and strong/reset auto routing unchanged. ACU reuse supports --wy-delivery
tiled-all and binds to that exact compared role. No local device result.
Design/limits/command: docs/PPU_WY_COMPUTE_TILES.md.

## Initial implementation checkpoint

Active sources: /workspace/gdn-wy-tiles-20260921, branch ppu-wy-tiles-20260921.
Evidence/registered plan: /workspace/gdn-wy-tiles-evidence-20260921.
New prepare/state/output kernels real-SDK compile and link: 160/232/98
registers, all zero stack. State has 4 useful warps, output 8; shared
state/output both 49,408 bytes. Native state shuffle opcodes removed (old
128 static shuffle instructions); independent FP32 state and reduction order
preserved. Prepare TF32 high/residual solve shared with old control, unchanged.
Scalar/packed six images retained, tiled three added, explicit stage switches.
No route promotion. Host native ownership/order and seven negative plants
pass; complete post-edit suite running before final handoff. No new device
numerics/performance claims.

## Prior verified device review

New upload gdn-qsa-acu-20260921T123213Z-3755721.tar.gz: all 537 files and
checksum denominator verified. Actual three <true> kernels, g=-1.0, loaded
binding/library match comparison origin 1b1f508; WY/FLA same inputs and GPU
UUID, all CE clocks 1.700 GHz. No scalar capture in this archive. Included
comparison.json supplies both gates; medians/envelopes recomputed, 16-case /
64-candidate / 8-repeat numerical denominator validated.

ACU prepare/state/output: all 164.106/443.367/86.580 us versus FLA
82.146/91.879/44.786 us (prepare includes prefix+solve+WU). Executed instruction
ratios 2.69x/3.88x/2.14x. State KVD write bytes now exactly FLA's 98 MiB,
store instructions both 102400, BF16 MMA both 524288; matching write traffic
did not close the 4.83x state time gap. Stage write-footprint predictions
128.25/98/64 MiB all confirmed. State still 64 vs FLA 128 threads, 3.55 vs
7.10 active warps/CU; register shuffle/repack consumer remains after the new
snapshot exchange. Prepare has 3x TF32 MMA (explicit high/residual precision).

Complete API medians weak: scalar/all/FLA 717.626/724.098/496.814 us; strong:
710.058/719.990/495.610 us. Every candidate UNRESOLVED vs scalar, loses to
FLA. Keep defaults/original routing, no candidate speed admission. Original
strong route wins under existing numerical gate, with BF16 final state vs
WY/FLA FP32 explicitly noted. No cross-protocol time subtraction and no
old-scalar/new-candidate latency delta: the old ACU gate/build differs.

Report: docs/PPU_WY_DELIVERY_VERDICT_20260921.md. Local analysis:
/workspace/gdn-wy-delivery-acu-20260921/analyze_bundle.py and parsed.json.
Next bounded direction is MMA producer/consumer tile layout and warp work
distribution across all three stages, not another global-store-only patch.
No implementation, numerical criterion, default or routing changes here.

## Earlier pasted-result checkpoint (raw files are now verified above)

User-reported run /workspace/gdn-wy-fla-1b1f508-20260921T111202Z,
g=-1.0 only: original/scalar-WY/FLA medians 425.240/710.058/495.610 us.
prepare/state/output/all candidates: 708.476/714.356/713.394/719.990 us.
All four candidate-versus-scalar observed envelopes overlap: UNRESOLVED,
not an admitted improvement. All four lose to FLA with disjoint envelopes.
Original beats FLA with disjoint envelopes; preserve its strong-decay path.
Reported raw-bit checks pass. Terminal PASS means numerics and measurement
completed, not speed. Raw comparison file and run bundle have not been
received/hash-verified locally. Weak g=-0.1 is NOT PROVIDED, not inferred.

Source audit: delivery 1/2/4/7 reaches prepare/state/output template selection;
benchmark lambdas bind each delivery correctly. This is not runtime proof
of selected device images. Reuse the existing --wy-run binary for separate
--wy-delivery scalar/all --gate -1.0 ACU captures, covering all three stages.
Check kernel identities, actual KVD traffic, added shared/barrier costs and
kernel durations; do not subtract unlike API-event and profiled protocols.
Predicted transaction reduction is NOT yet a measured candidate result and
the old traffic amplification does not establish latency causality.
Local record: /workspace/gdn-wy-delivery-20260921/DEVICE_RESULT.md.
No routing/default/kernel modification; no device execution on this host.

## Delivery handoff checkpoint (superseded by device report above)

Candidate sources: /workspace/gdn-wy-align-20260921. Local evidence:
/workspace/gdn-wy-delivery-20260921. Prepare/state/output packed variants
compile+link with real SDK: 78/240/60 vregs, zero stack; scalar controls
84/244/80, zero stack. All three packed variants have real b32x4 global
stores and no scalar BF16 global stores. Full post-edit gate PASS in
sealed-local.log: 4 CTests, 47 Python contracts, 45 algebra cases plus five
negatives, 305 preserved original controls, original 15 + WY 6 kernel images.
Host ownership: 4,080 cases and 12 negatives PASS. Controls and original
auto routing retained. All three stages are in scope; no device timing claim.
Ready same-binary
scalar / prepare / state / output / all + original / FLA paired box command.
DELIVERY_AB=1 PPU_SDK=/usr/local/PPU_SDK DEVICE=0 JOBS=16
bash tools/run_ppu_wy_fla_box.sh. Runs 16 cases across five WY deliveries;
raw equality to scalar + 8 repeats before seven-arm balanced timing.
For subsequent ACU reuse: --wy-delivery all, never silently profile scalar.
See docs/PPU_WY_DELIVERY.md for resources, lifetime barriers, commands,
and the explicit distinction between recompiled scalar and archived dd70e5d.

New upload gdn-qsa-acu-20260921T073350Z-900246.tar.gz: all 535 files and
the complete SHA256SUMS denominator verified. STATUS PASS, WY 3 and FLA 7
launches captured, identical inputs/device and preserved dd70e5d binary hashes.
Actual box ACU is v2.0.2_20260603-2c7c7f1/data15000, not the local SDK's
v2.1.1. The unchanged binary now captures successfully with SDK-first ACU;
the precise earlier map::at throw site remains unknown.

Profiled stage times: WY prepare/state/output 165.799/464.781/84.339 us;
FLA prefix+solve+WU/state/output 83.122/89.155/43.711 us, plus 6.575 us
for two fills. Kernel-duration sums 714.919 vs 222.563 us are NOT public-API
event spans. State explains 76.3% of the profiled gap, with identical BF16
MMA count 524,288. WY state KVD-interface global-store traffic 784 MiB vs
FLA 98 MiB; DRAM bytes are almost equal. Native scalar fragment stores and
warp mapping checked before any implementation change. Both state grids
are 128; WY 64 vs FLA 128 threads, achieved 3.55 vs 7.09 warps/CU.
All profiled kernels report 1.700 GHz. No new routing or performance verdict.
Local analysis: /workspace/gdn-wy-acu-analysis-20260921.
Report: docs/PPU_WY_ACU_20260921.md. Host program uses shipping coordinate
helper, independently checks native MMA traits and reproduces 802,816 store
instructions / 784 MiB KVD traffic exactly: snapshots 512 + Vnew 256 + final
16 MiB for only 50 MiB useful data. Omitted-slot negative fails the counter
anchor. Actual uploaded assembly has scalar b16/b32 stores. Production
unchanged; no local device execution, new timing or routing promotion.

## Superseded capture checkpoint

At the previous checkpoint capture was USER-REPORTED/FAIL, not NOT_RUN or numerical FAIL. The
site ACU path was selected before the SDK. Earlier saved site-version proof:
v2.0.0_20251231/data12006; actual local SDK ACU --version gives
v2.1.1_20260725/data15000 (host-only version invocation, no profiling).
Failed-run tool binary/version not yet received: version mismatch remains a
hypothesis for the exception, not a proved cause. Fix SDK-first precedence,
keep explicit overrides, print actual tool version/hash, and preserve loaded
profiler/parser identities even on failure. Never retry/suppress map::at.
Pre-fix SDK-vs-site negative failed on the old chooser; patched 44 Python
contracts and 3/3 host tests pass. Planted map::at preserves FAIL evidence,
throws the same exception once, and cannot pass comparison admission.
No production source or binary has been changed.

User-reported WY device gate: PASS, 16 cases; all comparison arms 8/8 repeat.
Weak original/WY/FLA median: 915.720/715.232/480.806 us. Strong:
437.670/715.126/507.562 us. Both WY-versus-FLA observed envelopes overlap,
so both formal verdicts remain UNRESOLVED; median ratios are descriptive.
Weak WY beats original, strong original beats WY under the existing envelope
rule. No routing change. FLA ~500 us is an engineering target, not a new
admission rule. Prior terminal PASS meant execution/numerics, not a speed win.
Next capture must explicitly select WY (old `ours` selected original), reuse
the existing binding AND device library, and keep profiled kernel durations
separate from unprofiled complete-API spans. Do not subtract unlike protocols.

Local follow-up PASS: 40 Python contracts (18 ACU + 7 WY + 8 FLA + 7 HGGC),
3/3 compiled host tests, 45 algebra cases plus five negatives; 305 original
control expressions unchanged. Fake-tool integration proves reuse never builds
or selects original; real missing-ACU invocation returns rc=1 + INCOMPLETE tar,
not PASS. Local counter execution remains NOT_RUN; user capture FAIL is above.
Logs: /workspace/gdn-wy-acu-followup-20260921.
Command: DEVICE=0 bash tools/run_ppu_gdn_fla_acu_box.sh --wy-run
/workspace/gdn-wy-fla-dd70e5d-20260921T041641Z.
No csrc/include/Python API/CMake/submodule differences from dd70e5d.

## Prior checkpoints (historical; current status is above)

Current plan: C64 parallel W/U preparation, V32 FP32 register-state recurrence,
chunk-parallel output. Keep original reset/replay/scan and auto routing intact.
This is an algorithm-structure candidate, not merely a backend replacement.
CPU/layout/codegen admission precedes device three-arm comparison. No new
performance claim; FLA proximity target is <=1.10x on the priority weak shape.

Local full gate PASS: 3 CTests, 45 arithmetic cases + five numeric negatives,
33 Python contracts, 305 unchanged original control expressions, all original
15 and new three device images linked. Final prepare/state/output: 84/244/80 vregs,
zero stack. Four native-map negatives and three binary negatives red.
No device execution or timing claimed. Handoff: tools/run_ppu_wy_fla_box.sh,
which admits original + WY before same-input alternating original/WY/FLA timing.
Final unchanged-code validation: /workspace/gdn-wy-align-20260921-build/admitted-local.log.

User's current build failure is before kernel compilation: inherited actlize
maps logical ppu0010 to -arch=ppu_10, but box HGGC lists ppu001/ppu0015/all.
GDN's local CMake seam now compile-probes ppu_10, retries ppu001 only on the
exact unsupported-spelling error, and retains logical ppu0010 plus every
non-architecture option. Wrong-target/all fallback is forbidden. No kernel
or actlize submodule changes. Architecture/compiler hash is an explicit
object dependency and joins the ACU bundle. Full local gate PASS: 7 compiler
dialect/negative tests (synthetic legacy interface, not real legacy SDK),
12 direct-ACU tests, 8 FLA tests, 2/2 compiled host tests, 55 algebra cases,
real CuTe delivery/retile and 305 unchanged source controls. SDK2.1.1 actually
recompiled all six device TUs and linked both libraries; 15 device images,
C16 zero stack, unchanged resources/opcode counts. Submodule stays 423253c0.

User-reported same-input full-API timing: g=-0.1 ours 925.020 us vs FLA
726.992 us (FLA-WINS); g=-1 ours 457.340 us vs FLA 746.904 us
(OURS-WINS). These are API event spans, not sums of kernel durations.
Current task: remove PPUProfiler entirely as requested. Reference is
quactlize/tools/run_dense_marlin_m8_acu_box.sh: separate correctness process,
then direct acu -f -o ... --set full against a subject-only process. No
kernel, dispatcher or tolerance changes. Exact runtime failure log was not
provided; no unsupported diagnosis of that error is claimed.

New handoff: tools/run_ppu_gdn_fla_acu_box.sh, default weak g=-0.1.
One subject API call per arm, no subject warmup; preflight in another process.
CPU-only verification, no profiler hooks/library. Whole-process capture can
include runtime initialization or fresh-process FLA autotuning, explicitly
not claimed to be a warmed steady-state kernel range. Same-input/device/binary receipts, native ACU
reports plus automatic plain-text metrics, source snapshots and binaries,
resource dumps and checksums go to a single /workspace tar.gz. Failures keep
an INCOMPLETE diagnostic tar. No CSV copy-paste or manual identity variables.
Revised local gate PASS: 12 direct-capture/bundle tests + 8 FLA tests, then
the complete host/codegen suite (2/2 CTest, 55 affine cases, CuTe delivery,
305 unchanged source controls and all 15 linked PPU kernel images). Negatives
cover accidental profiler hooks, subject warmups, device verification casts,
preflight/subject identity mismatch, wrong/missing reports and failed tools.
Actual ACU execution remains box-unverified. No production code changed.
Previous handoff local: 11 capture/bundle tests and 8 FLA tests; complete
host/codegen gate passes (55 algebra cases, real CuTe alias/delivery,
305 original control expressions, 15 linked PPU kernel images). Actual
capture is box-unverified, not locally PASS. End-to-end missing-ACU negative
returned rc=1 and preserved an INCOMPLETE tar under:
/workspace/gdn-acu-missing-tool-contract-20260920T230350Z.

Box FLA compilation failed in Triton ptx_get_version("13.0"). Reproduced
locally: the old function handles only CUDA 10/11/12, then emits its misleading
"only support CUDA 10.0 or higher" error. Process-local backport uses Triton
v3.5.0's 13.0 -> 90 entry. Does not replace packages, fake a CUDA12 version,
change assembler/backend selection, override accepted vendor mappings, or
swallow other errors. Compatibility decision and compiler hash are recorded.
Actual local parser passes 13.0 after the fix; 12.9 remains 88, LLVM +ptx86
cap unchanged. PPU SDK 2.1.1 ptxas reports release13.0 and compiles a minimal
PTX9.0 kernel; hgobjdump identifies PPU1.0 output. Compile-only, not device PASS.
Eight CPU contract tests and existing host/codegen suite PASS:
/workspace/gdn-triton-cuda13-compat/local.log. GDN/FLA kernels unchanged.

Device follow-up, 2026-09-20: user reports "pass" in response to the PERF=0
handoff. Correctness is now USER-REPORTED/PASS, not inferred from local tests.
The full new device log, binary hash and measured error values have not been
provided; no specific numerical errors or device timings are invented here.
Next: PERF=1, B1/S2048/Hk16/Hv32/D128, strong and weak decay reported separately.
The protocol includes allocations/preprocessing/host dispatch synchronization,
so it is full-public-API event span, not kernel-only latency.

FLA comparison opt-in: WITH_FLA=1 on tools/run_ppu_gdn_backend_box.sh.
Uses installed PPU FLA (optional explicit FLA_ROOT), same BF16 tensors and
CPU oracle, zero initial state and output+final-state on both sides. Default
native GVA; explicit FLA_HEADS=expanded does its expansion outside timing.
Both decay cases independently pass correctness and repeat gates before
their timing. JIT/autotune excluded; sequential AB/BA samples, no overlap.
Saves fla-comparison.log/json beside source and binary identities. No
ours-only fallback on FLA error; overlapping sample envelopes UNRESOLVED.
Original comparison-handoff verification: 5 CPU contract tests PASS (including missing
FLA, missing/wrong state, wrong/nonfinite output and both winner directions),
full existing host/codegen suite PASS. Log:
/workspace/gdn-qsa-retile-fix/fla-local.log. No new PPU timing is claimed.

Device failure: B2,S65,Hk1,Hv2,C16,GC2, g=0, Hillis-Steele fallback,
output/state error [0.9999995827674866, 1.0]. Root cause: retile_D returned
owning Tensor&, so `auto view = ...` copied register storage; load wrote the
copy and MMA read the original. retile_S had the same non-view contract.
Pre-fix actual CuTe alias test: 3072/3072 detached destinations and stale
sources. Fixed: 0/3072, exact old implementation remains a RED control.
Same fixture's zero-result error signature exactly matches both reported
numbers; actual device nonzero counts were not printed in the old runner.
Repaired runner now prints them on failure. No 2% tolerance or kernel control
flow has changed. Explicit adapter header dependencies prevent stale objects.

Post-fix local checkpoint: six original device TUs and original gdn_ops.cu compile/link
with hgcc ppu_10. All 305 original control expressions and host dispatch body
match; four structure negatives are red. Delivery/fragment proof and full
six-product Neumann vs forward-substitution pass with four mapping negatives.
The superseded scalar-gather implementation and its dead C ABI were removed.
Runner now checks the original public API on reset / Blelloch / Hillis-Steele /
serial, then measures B1,S2048,Hk16,Hv32,D128 for strong and weak decay.

Default C16 kernels: zero stack; serial/fused recurrence uses 176/204 vector
registers after the correctness repair (broken baseline: 148/182).
Original experimental C32 recurrence: 144-byte
stack, not on the production auto-dispatch path; must remain explicitly
unadmitted as a PPU performance configuration.

Fresh post-fix compile run PASS: /workspace/gdn-qsa-retile-fix/full-local.log.
Final complete local rerun PASS: /workspace/gdn-qsa-retile-fix/final-local.log.
2/2 compiled host tests; 55 algebra cases; four mapping, four preservation,
and exact legacy alias negatives; six device TUs plus original host wrapper
linked; 15 kernel images. CPU failure-signature check passes with the device
harness's own fixture/reference/comparator, hash d25bbac598263f63.
The pre-fix device result is FAIL; corrected admission is USER-REPORTED/PASS.
Corrected performance is pending. Local runtime import and device execution
remain unavailable; none is inferred from host/codegen PASS.
