# WY alignment checkpoint

updated-at: 2026-09-21 03:26:22 UTC
parent: 8837640119835b6c8b7953d2c61e24de9fe35000
working-on: C64 WY candidate; see docs/plan.md
blocked-on: no local PPU execution; SDK compile/link available
last-commit: 8837640

Original kernels/dispatch are immutable controls. New code belongs in
gdn_wy_* files with an explicit API and separate binding. No automatic route
promotion before device evidence. Do not repeat the completed RTX5070 study.
