# PPU original-structure port

updated-at: 2026-09-20 13:30:24 UTC
working-on: same-input FLA comparison ready; WITH_FLA=1 handoff
blocked-on: no local PPU; actual A/B timings require the user's installed PPU FLA
last-commit: 44772bf (same-input PPU GDN/FLA comparison; kernels unchanged)

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
Local verification: 5 CPU benchmark contract tests PASS (including missing
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
