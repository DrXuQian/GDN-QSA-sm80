# SM90 GDN: multi-workload admission

Status: **in progress; the expanded performance target is not met**. The H800
remains on. A primary-shape win is not permission to declare all shapes faster.
No production routing, legacy SM80 code, clock or power settings were changed.
Native PPU1.7 remains **SKIP: SDK/model unavailable**; these are H800 controls.

## Registered coverage

The inventory was registered before its measurements, following the user's
request not to rely on a single test case. It is fourteen explicit workloads,
not a full Cartesian sweep. Every row has both natural-log gate regimes
`g=-0.1` and `g=-1.0`, and both pinned reference libraries, for **56 captures**.
K/V dimensions128, chunk64, BF16 Q/K/V/beta and FP32 final state stay fixed.

| Workload axis | Cases |
|---|---|
| Sequence length, B1/Hq16/Hv32 | 512,1024,2048,4096,8192 |
| Incomplete final chunk | T2051 |
| Independent batch sequences, T2048/Hq16/Hv32 | B2,B4 |
| Value/query head mapping, B1/T2048 | 16/8,32/32,64/16,64/32 |
| Gate format/pattern | token/head-distinct FP32 gate, zero initial state |
| Initial state | same varying gate with nonzero, nonsymmetric FP32 state |

The last two use B1/T2048/Hq16/Hv32. The nominal fixture is preserved byte for
byte. Batch sequences are never concatenated into one recurrent history.
FlashInfer's packed-token/VK-state API gets explicit CPU-side adapters;
FlashQLA uses fixed-batch/KV-state inputs. Cross-library input, initial-state
and independent CPU-reference hashes must match per scenario.

Authority: experiment branch `sm90-multishape-20260926`,
`tools/sm90_workloads.py`. Every capture uses12 interleaved/reversed complete
forward calls and sums all GPU kernels from nsys, including reference helper
kernels. Both auto and no-CP routes are measured; each scenario uses its own
fastest reference. The original2% numerical criterion,8 repeated bit-stable
launches, candidate/parent raw equality, per-capture output checks and disjoint
observed timing envelopes are unchanged. Overlap remains UNRESOLVED.

## Counterexamples already established

The tested candidate is S38 (V128 split into two independent V64 CTAs,
aux232/state192). Control is the immutable unsplit S24. Representative
same-capture weak-gate medians in microseconds:

| Workload | S24 | S38 | Fastest FlashInfer | Verdict for S38 |
|---|---:|---:|---:|---|
| B1/T2048/Hq16/Hv32 | 120.513 | 101.505 | 112.769 | wins |
| B1/T8192/Hq16/Hv32 | 447.938 | 358.753 | 353.425 | loses |
| B2/T2048/Hq16/Hv32 | 125.089 | 194.386 | 114.289 | loses |
| B4/T2048/Hq16/Hv32 | 249.409 | 288.786 | 227.329 | loses |
| B1/T2048/Hq16/Hv64 | 125.185 | 195.026 | 114.689 | loses |
| B1/T2048/Hq32/Hv64 | 126.304 | 196.801 | 115.232 | loses |

The same losses also occur at the strong gate in the completed cells. Do not
infer a universal selector from the first row. V64 doubles the physical grid
and duplicates QK/KK/inverse work. At64 batch-heads it creates128 CTAs on114
SMs while resources still allow only one such resident CTA per SM. The
unsplit control is substantially faster there, but still loses FlashInfer.

T8192 is a different gap: the fastest FlashInfer route is CP. Its pinned
`gdn_prefill.py`/`delta_rule_dsl/varlen_helper.py` chooses CP for insufficient
parallel batch-heads, then bounds preprocessing chunks against the SM count.
Calling these two gaps the same scheduler bottleneck would miss that change
of algorithm. No averaging across rows may hide either counterexample.

## Evidence repair, not a timing-rule change

A later, independent failure remains visible: on B1/T2048/Hq32/Hv32,
g=-0.1, FlashQLA auto has output/state relative errors
0.021621605/0.009716575. It fails the registered2% gate before timing. Our
same-input S24/S38 pass. This is a reference admission failure, not evidence
of a regression in our kernel and not permission to loosen the criterion.
The paired strong-gate capture passed. Isolated auto/no-CP diagnosis is
pending; unrelated workloads continue in fresh processes.

At the34th original capture, first-JIT FlashQLA passed all five arms' numerics
but failed before timing: the old identity collector required a mapped
`executable.so`. Fresh TVM executables can instead run live LLVM/CUDA modules.
Three previous partially fresh-JIT captures also had incomplete image sets.
All four attempts and the original matrix are preserved, not overwritten.

The repaired collector serializes each actual live CUDA module, validates
its complete CUDA ELF extent and hashes it. The pinned TVM JSON exporter
cannot describe TMA descriptor dtype30, so binary serialization is used; no
reference recompilation or disk-cache guess substitutes for the live image.
Real fresh-JIT and cache-hit runs of the failing workload each bind all four
CUDA images; their four hashes are identical. Both processes also pass all
five arms' CPU oracles and8 repeats. No timings from that proof are ranked.

Two source epochs are stored. An AST proof permits changes only inside
`reference_binaries`; input, numeric, capture and timing changes fail the
negative test. Retried cells use new `-identity-r2` directories. Negative
controls reject a missing module, wrong ELF, truncated extent, omitted
workload, failed capture or a performance loss being counted as goal closure.

`tools/record_sm90_workloads.py` re-extracts every completed SQLite, validates
images/epochs/device/input hashes and reports unfinished cells explicitly.
Run it with Python3.12 like the capture host: Python3.10's naive float `sum`
changes last bits of descriptive per-symbol totals. The integer-nanosecond
forward sums and verdicts are unaffected; no tolerance was added to disguise
that reanalysis mismatch. Python3.12 local re-extraction is exact.

Artifacts: `/workspace/gdn-sm90-multishape-20260926/{captures,jit-fresh-cached-proof-r2}`.
Final full-inventory results and the verified archive will be added after
all56 captures close. S39/S40/S41 are separately registered followups, not
changes to the running S24/S38 comparison, and not admitted performance wins.
