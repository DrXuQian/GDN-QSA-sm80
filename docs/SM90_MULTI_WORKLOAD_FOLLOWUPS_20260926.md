# Expanded-case followups: keep negative results and distinct protocols

Physical H800 PCIe, unchanged scalar-GDN contract and immutable S24/S38
parents. No default routing or SM80 edits. Native PPU1.7 is still SKIP.
The expanded goal is **not met**, so the H800 remains on.

## S39/S40: output stash rejected by screening

Both add a32KiB private FP32 O1 stash; S40 also moves register budgets from
state192/aux104 to state160/aux168. All14 CPU/parent cases and2 direct-byte
stress pairs pass, all4 native bodies remain asynchronous. Neither wins:

| Candidate / workload | Parent weak / candidate | Parent strong / candidate |
|---|---:|---:|
| S39 B1/T2048 | 115.686 /126.848 | 117.088 /127.772 |
| S39 B2/T2048 | 116.546 /127.956 | 116.366 /127.772 |
| S40 B1/T2048 | 115.296 /122.992 | 117.300 /125.060 |
| S40 B2/T2048 | 115.314 /123.688 | 117.112 /125.594 |

Microseconds, paired graph screening, **not nsys admission**. Every cell
has disjoint ranges favoring the parent. Extra shared traffic does not become
free just because it removes accumulator liveness. No promotion.

## S41: eliminating loader spill is not a universal speed improvement

Loader24→32 on S38, state192/aux232 unchanged; all4 actual bodies have zero
spill with unchanged matrix/retirement counts. All numerical gates pass.
Six registered nsys captures/408 full forwards were re-extracted exactly
locally from SQLite. Same12 interleaved samples and disjoint-range rule.

| Workload / gate | S38 | S41 | Parent comparison | Fastest reference | S41 vs reference |
|---|---:|---:|---|---:|---|
| T2048 / -.1 | 99.985 | 101.872 | UNRESOLVED | FI113.169 | wins |
| T2048 / -1 | 100.145 | 101.488 | UNRESOLVED | FI112.721 | wins |
| T8192 / -.1 | 360.729 | 358.729 | S41 wins | FI354.376 | UNRESOLVED |
| T8192 / -1 | 359.354 | 355.674 | UNRESOLVED | FI352.987 | UNRESOLVED |
| Hv16 / -.1 | 94.111 | 95.071 | UNRESOLVED | QLA92.255 | loses |
| Hv16 / -1 | 93.375 | 94.063 | UNRESOLVED | QLA91.503 | loses |

The T8192 weak-gate parent envelopes are separated by only0.001us:
S38[360.025,362.232], S41[357.752,360.024]. Report the registered verdict,
but do not turn that edge into a general selector or claim a reference win.
The graph screen's tiny T2048 gain was not reproduced as an nsys parent win.
Different protocols remain separate. S38 stays the admitted primary path.

## S42: one state WG rejected before GPU

V128/one state WG and shared BF16 H passed the actual CuTe STSM→SS reader
map:16384/16384 cells, transpose/alias/missing-owner negatives red. It does
not pass native execution-structure admission: all4 bodies emit C7512,
stack920/1016B, spill stores2580/2852B and loads2630/3310B (without/with
initial state). Actual state248/aux232/loader24 options are emitted.

Removing the converted H operand did not remove H128+O64+SK64 FP32 live
accumulators: already256 registers before control. No timing was run.
Retain this failed branch, do not describe its generic compile PASS as a
valid asynchronous candidate. The bounded S43 followup retains two state
WGs, shared H and state160/aux168; at this checkpoint it is **compile-only
in progress**, not a performance result.

## Evidence

Frozen56-attempt matrix archive is local and remote, SHA256
`7940961bd67962ca6cca1db3c0b9f27551cc397e82d3838fa0a7c05f14835792`:
`/workspace/gdn-sm90-multishape-evidence-20260926T1712Z.tar.gz`.
Contains55 valid captures plus3 superseded captures, original failure
receipts, exact reference images, numerical diagnosis, binaries and sources.
The failed reference remains failed; it is not silently removed from coverage.

Followup evidence root `/workspace/gdn-sm90-multishape-20260926/`:
`followup-admission-r2`, `followup-screen`, `s41-nsys` (all locally copied).
S42 `/workspace/gdn-sm90-shared-h-20260926/`, source branch
`sm90-shared-h-20260926` commit`bbddadd`. No performance data for S42.
