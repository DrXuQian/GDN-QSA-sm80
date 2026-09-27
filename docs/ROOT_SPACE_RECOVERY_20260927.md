# Recoverable root-filesystem cleanup

2026-09-27 UTC, requested by the user while the SM90 experiment was active.
Root overlay30GiB had695MiB available. `/workspace` held18GiB of artifacts;
`/root` held11GiB including5GiB repositories and3.5GiB Codex programs/sessions.
The existing root cache was already a link onto the separate100GiB data disk.

Moved, not discarded: four historical evidence directories and eight root
report/archive files into `/root/autodl-tmp/root-space-recovery-20260927`.
Exact targets, pre-move SHA256 manifests, symbolic-link inventories and the
completed migration log are under `/workspace/root-space-recovery-20260927`.
All regular-file hashes and nested link names/targets matched after migration;
old paths remain links. No active worktree, current H800 control, code, session,
model or measurement was deleted. No recursive cleanup traversed nested links.

Root availability afterward4.1GiB, usage87% rather than98%. Data disk55/100GiB.
The move is reversible; restore an original real directory only after validating
and removing its exact compatibility link and checking root capacity. Do not
run generic cleanup over the relocated evidence or through these links.
