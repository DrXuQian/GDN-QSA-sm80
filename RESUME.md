# WY alignment checkpoint

updated-at: 2026-09-21 04:48:43 UTC
parent: 8837640119835b6c8b7953d2c61e24de9fe35000
working-on: capture-only follow-up for the admitted dd70e5d WY comparison
blocked-on: new per-stage ACU counters need PPU execution
last-commit: a0fe950 (capture-only follow-up)

User has now reported 16/16 device cases PASS and 8/8 repeats. Weak
original/WY/FLA median 915.720/715.232/480.806 us, strong
437.670/715.126/507.562 us. Both WY/FLA envelopes overlap: UNRESOLVED.
Do not discard slow FLA samples, switch the verdict statistic or promote
auto routing. The <=1.10x target is unchanged and not established.

Next command (no rebuild): DEVICE=0 bash tools/run_ppu_gdn_fla_acu_box.sh
--wy-run /workspace/gdn-wy-fla-dd70e5d-20260921T041641Z.
Explicit WY role, old binding AND device library hashes, preceding raw API
samples, direct ACU reports and FLA sources go to one /workspace tar.
No profiler hooks, CSV requirement or new speed claim. Do not equate profiled
kernel sums with complete-API event spans or subtract them to infer CPU gaps.
Old comparison properties lack physical UUID; that limitation remains visible.

Current local follow-up: 40 Python contracts, 3 compiled host gates, 45 algebra
cases and numeric negatives PASS; original 305 controls and every production
file unchanged. End-to-end missing-tool test rejects with INCOMPLETE archive.
Tests: /workspace/gdn-wy-acu-followup-20260921.

## Earlier implementation evidence

Original kernels/dispatch are immutable controls. New code belongs in
gdn_wy_* files with an explicit API and separate binding. No automatic route
promotion before device evidence. Do not repeat the completed RTX5070 study.

Local CPU arithmetic: 45 cases including S2048 Hv32/Hv64 pass original 2%
gate, five planted mistakes rejected. True double WY/token recurrence agree.
Native ownership passes 701,484 output rows and full state ownership; all four
mapping negatives rejected. Actual SDK build/link passes new three kernels;
final resources prepare/state/output=84/244/80 vregs, all zero stack. Initial
32-byte state spill rejected; narrowed native mapping removed it without
touching original fragment adapter. Full original rebuild passed all 15 images.
Final log: /workspace/gdn-wy-align-20260921-build/admitted-local.log. Three CTests,
33 Python contracts, 45 arithmetic cases and old structure preservation pass.
Device admission/timing were NOT_RUN at this local checkpoint. The user device
follow-up above supersedes that status; do not rebuild just to collect ACU.

Local candidate library SHA256:
`d704093f85ead3921ad216a47df47852f30271220017a03c1a3c8ea618630d09`.
Local binding SHA256:
`728bce6cf9f5e5e2a700eaf9284db850f7d13c732e25d1ee2e623ce0d8bc32c3`.
Compiler SHA256:
`fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e`.
These local SDK2.1.1 artifacts establish compile/link only. Box runner rebuilds
against its installed SDK/PyTorch and records that run's hashes separately.
