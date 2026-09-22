# PPU original-structure port

updated-at: 2026-09-22 09:48:30 UTC
working-on: local admission complete; STATE_AB box handoff ready, defaults unchanged
blocked-on: device numerical/performance admission awaits box; local CPU torch cannot build the unchanged CUDA binding
last-commit: 9ab15f5 (state address/gate experiment and complete local admission)

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
