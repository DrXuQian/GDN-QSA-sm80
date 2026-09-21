# Weak-decay GDN versus FLA: one-command counter bundle

From the `ppu-backend` checkout on the PPU box:

```bash
git pull --ff-only
DEVICE=0 JOBS=16 bash tools/run_ppu_gdn_fla_acu_box.sh
```

Use the same physical `DEVICE` as the preceding latency comparison. The
default SDK is `/usr/local/PPU_SDK`; override `PPU_SDK` only if needed. ACU is
found at the site's `/sim/eec/shared/junfu.qx/asight/bin/acu`, then the SDK,
then PATH; `ACU=/absolute/path/to/acu` overrides discovery. Installed FLA is
used by default; optional `FLA_ROOT` selects an existing checkout. No clone,
pip install, SSH, clock setting, or mandatory identity form is involved.

The script builds locally **on the box**, against its installed PyTorch,
then captures ours and FLA sequentially. It does not modify kernels or run a
performance sweep. To reuse a particular build, set `EXTENSION` to its exact
`_gdn_chunk_ppu*.so` filename, with `libgdn_qsa_ppu.so` beside it. That mode is
explicitly recorded as operator-supplied: the current checkout is **not**
asserted to be the origin of an old binary. The actually loaded binary
hashes must match the archived files.

Artifacts use a new `/workspace/gdn-qsa-acu-<UTC>-<pid>` directory (`mkdir`,
not `mktemp`). `OUT` may specify another **new** directory. Existing output
directories are refused, not deleted. Upload the **single `.tar.gz` file**
printed after `UPLOAD=`. The tar contains native `.report[.acurep]` reports;
there is no GUI screenshot or CSV-copying requirement.

## Fixed subject and capture scope

- Weak decay g=-0.1, BF16, B1/S2048/Hk16/Hv32/K128/V128, native GVA.
- The same CPU fixture/seed, recurrent oracle and fixed 2% output **and**
  final-state criterion as `bench_ppu_gdn_fla.py`. Inputs/reference hashes
  must match across arms. Start with zero state; return final state; no QK
  normalization. FLA's Triton path is explicit, not FlashQLA dispatch.
- Follow `quactlize/tools/run_dense_marlin_m8_acu_box.sh`: first run a
  **separate unprofiled preflight process** (first call + five warmups with
  correctness/fingerprint checks), then run a **subject-only process** under
  `acu -f -o ... --set full python ... --phase subject`.
- There is no profiler API, dynamic profiler-library lookup, start/stop
  range, launch-skip/count guess, or kernel-name filter. The subject process
  calls forward exactly once, with zero warmups. Verification copies results
  to CPU **before** dtype casts/comparisons, so it adds no verification GPU
  kernels. Its output/state must match the separate preflight bit-for-bit.
- ACU profiles the **whole subject process**: runtime setup if any, gate
  reduction, head expansion, clears, transpose, preparation, recurrence,
  and FLA's complete pipeline. Preflight and subject inherit the same Triton
  disk-cache environment, but in-memory autotuner choices are not carried
  between processes. If the installed FLA repeats library-internal tuning
  in the fresh process, those launches remain in the report. Do not claim
  the report excludes all JIT/autotuning or contains only steady-state work.
  One public API call is not necessarily one kernel launch/replay.
- ACU's default profiling/cache policy is used. Profiled durations are
  **diagnostic**, not new winner timings. The profiler flushes caches and
  replays launches. Neither sum of profiled kernel times nor wall time of
  capture is the preceding unprofiled full-API event span. Host dispatch
  synchronization/gaps require separate timeline measurement if needed.

Use `bash tools/run_ppu_gdn_fla_acu_box.sh --gate -1.0` for the strong-decay
control in a separate bundle; it is not mixed into the default weak capture.

## What is archived

Both native reports, exact commands and complete ACU logs; automatic plain
text `details`/`raw` metric exports; per-arm correctness/identity JSON;
SHA/diff/submodule pins and relevant C++/Python source snapshots; actual
extension and device library, their hashes and PPU ISA/resource dumps;
imported FLA source files and its Triton compiler source/hash; torch/Python
versions/build information; SDK/ACU versions/help; read-only `ppu-smi -q`
before and after, if available. ACU's in-kernel frequency counters, not an
idle pre/post clock sample, determine the achieved frequency.

`STATUS.json` separates optional probe/export unavailability from capture
success. `SHA256SUMS` covers every other archived file. The archive's own
checksum is saved beside it. No full build directory, Triton cache, model
weights, tensors, credentials or environment dump is archived. A failed
build/import/capture still produces an **INCOMPLETE** diagnostic tar, never
a one-arm comparison PASS. The original numerical benchmark is unchanged.

## Questions this capture must answer (not preselected conclusions)

The user's unprofiled same-input results are 925.020 vs 726.992 us at g=-0.1
(FLA wins), and 457.340 vs 746.904 us at g=-1 (ours wins). Their complete
timing logs/binary identities are not yet in this repository.

On the current weak-path source, auto dispatch expands Q/K from 16 to 32
heads, builds per-chunk workspace, then runs serial recurrence with 32 CTAs,
128 C16 chunks per CTA. Only 128 of its 256 threads execute the C16 MMA
body. Those are **source facts**, not measured causal attributions. Determine:

1. Which kernels dominate: preparation/clears/expansion, or recurrence?
2. For recurrence: grid, resources/occupancy, active warps, tensor issue,
   memory dependency, synchronization, instruction fetch, traffic and spills.
3. For FLA: its actual kernel split, launch grids, resources and the same
   counters on the identical input/device. Compare work as well as duration;
   the two implementations need not use the same chunk size/rounding points.
4. Whether the weak-path gap is primarily insufficient independent work,
   state/fragment delivery, or public API preprocessing. Low occupancy alone
   does not establish which optimization will win.

Local tests validate separate preflight/subject call counts, CPU-only
verification, absence of profiler APIs, identity binding, report
selection, failure propagation and tar/checksum contracts. No local PPU is
available: actual ACU interception/counters remain **box-unverified** until
this command completes successfully. No new device speedup is claimed.
