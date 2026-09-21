# WY alignment checkpoint

updated-at: 2026-09-21 04:05:02 UTC
parent: 8837640119835b6c8b7953d2c61e24de9fe35000
working-on: implementation complete locally; next is box admission/comparison
blocked-on: no local PPU execution; SDK compile/link available
last-commit: 5e9460a (implementation)

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
Device admission/timing NOT_RUN. tools/run_ppu_wy_fla_box.sh is the handoff.

Local candidate library SHA256:
`d704093f85ead3921ad216a47df47852f30271220017a03c1a3c8ea618630d09`.
Local binding SHA256:
`728bce6cf9f5e5e2a700eaf9284db850f7d13c732e25d1ee2e623ce0d8bc32c3`.
Compiler SHA256:
`fa62c590c67411c23fa4028f15fa562b39ce0cf830830d038a1ec04c59d8c76e`.
These local SDK2.1.1 artifacts establish compile/link only. Box runner rebuilds
against its installed SDK/PyTorch and records that run's hashes separately.
