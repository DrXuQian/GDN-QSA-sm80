# PPU original-structure port

updated-at: 2026-09-20 23:06:44 UTC
working-on: ACU capture/bundle complete; awaiting user's uploaded report tar
blocked-on: no local PPU; native profiler capture will run on user's box
last-commit: ba05c9f (same-input ACU capture, source/binary evidence and tar)

User-reported same-input full-API timing: g=-0.1 ours 925.020 us vs FLA
726.992 us (FLA-WINS); g=-1 ours 457.340 us vs FLA 746.904 us
(OURS-WINS). These are API event spans, not sums of kernel durations.
Current task: profile the weak path on both implementations, after JIT and
warmup, then automatically archive reports, logs, source/binary identities
and resource evidence. No changes to kernels, dispatcher or tolerances.

New handoff: tools/run_ppu_gdn_fla_acu_box.sh, default weak g=-0.1.
One warmed full API call per arm, native hggcProfilerStart/Stop from the
already-loaded PPU runtime; no new SDK runtime injection. Preflight/JIT/
warmup/oracle outside range. Same-input/device/binary receipts, native ACU
reports plus automatic plain-text metrics, source snapshots and binaries,
resource dumps and checksums go to a single /workspace tar.gz. Failures keep
an INCOMPLETE diagnostic tar. No CSV copy-paste or manual identity variables.
Local: 11 capture/bundle contract tests and 8 existing FLA tests; complete
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
