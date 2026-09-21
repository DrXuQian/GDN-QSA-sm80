# WY alignment checkpoint

updated-at: 2026-09-21 04:05:02 UTC
parent: 8837640119835b6c8b7953d2c61e24de9fe35000
working-on: final full local gate PASS; commit/push and three-arm box handoff
blocked-on: no local PPU execution; SDK compile/link available
last-commit: b36290e

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
