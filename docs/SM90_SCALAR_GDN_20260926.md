# Scalar GDN specialization campaign (in progress)

Parent6bebe1d, same-input library comparison registered beforehand in
[SM90_LIBRARY_NSYS_PLAN.md](SM90_LIBRARY_NSYS_PLAN.md). User requirement:
beat BOTH libraries' fastest admitted forward paths, not a slower selected
reference. Current target on H800 B1/T2048/Hqk16/Hv32/D128 is FlashInfer
CP-off112.014/111.279us and FlashQLA auto166.252/163.598us (g=-.1/-1).
Use full nsys kernel sums; require disjoint observed ranges, unchanged2%
independent recurrent CPU oracle on both O/state. No SM80 or routing change.

R1 changes only role register budgets24/152/168/168 ->24/104/192/192,
including aux `setmaxnreg.dec`. The reference source is pinned FlashInfer
5d9f8c8; this is a testable resource hypothesis, not a claim that resource
counts alone explain the gap. Four native bodies compile without C7512;
stack104/136B and spills remain. Full physical admission/timing is pending.

S1 adds `scalar_gdn_aux.cuh`, inheriting TMA, layouts, FP32 state recurrence,
inverse and pipeline contracts. It does NOT rewrite or remove the old KDA
collective. For scalar prefix p_i:

    sum_d ((q_id * exp(p_i-a)) * (k_jd * exp(a-p_j)))
      = exp(p_i-p_j) * sum_d(q_id*k_jd)

The gate is independent of d; this identity is NOT valid for vector-gated KDA.
The old path gates 16x16 sub-block operands before TF32 MMA; S1 issues whole
64x64 BF16 shared/shared WGMMA for QK/KK and applies decay to FP32 accumulators.
Positive KK lower input and beta[row] enter the unchanged FP16 inverse; inverse
then applies beta[column]. Causal and tail masks apply before exponential/cast.

Rounding changes explicitly: BF16 Q/K -> FP32 dot -> FP32 scalar decay ->
existing BF16 QK / FP16 KK, rather than TF32 of pre-gated operands. This is an
algebraic specialization, NOT a raw-bit-preserving delivery optimization.
Keep BF16 input/output, FP32 state and the existing inverse; do not weaken
tolerance or change workload. Expose numeric_schedule in the extension.

Local checks: all192 tail/gate combinations factor correctly in FP64; wrong
gate-row, beta-column and future-live negatives must fail. The existing72
full-recurrence algebra cases/four negatives remain. These are not device
rounding evidence. Real4body compile,14physical CPU-oracle cases,8repeat
stability and captured-output checks are required before performance admission.

The existing library nsys harness gains an optional separately hash-bound
candidate arm; it never replaces either incumbent. Its registered denominator
expands by12 calls. Removing the same candidate call from both trace and
receipt must still fail. R1 additionally requires raw equality with incumbent;
S1 is judged against CPU numerics, with rounding differences retained.

Experiment deadline checkpoint07:30UTC2026-09-26; success is not inferred from
that budget. Native PPU1.7 remains SKIP (no10700 SDK/model); H800 is physical
SM90 evidence, not PPU1.7 evidence. Task files/evidence:
`/workspace/gdn-sm90-win-20260926`.
