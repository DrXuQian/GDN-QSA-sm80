# PPU original-structure port

updated-at: 2026-09-21 07:24:03 UTC
working-on: matched-SDK ACU handoff ready; kernel bytes unchanged; waiting for capture evidence
blocked-on: user ACU capture throws map::at after preflight; precise native throw site unknown; no local PPU
last-commit: fe0c7be (SDK-first ACU and failure evidence; following commit is metadata only)

Current capture is USER-REPORTED/FAIL, not NOT_RUN or numerical FAIL. The
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
