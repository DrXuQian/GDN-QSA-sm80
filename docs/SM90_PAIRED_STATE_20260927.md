# S49/S50: independent matrix overlap, no default promotion

S49 is rejected. S50 is a separate candidate with native/host admission only,
device build in progress. All performance gates use the existing14-workload
inventory, not only the successful primary shape. H800 remains on.

## S49: shared rounded H, paired O1/SK retirement

Sourcefbdea2d, parent06b7471(kernelbc3c154/S24). O1 andSK have disjoint
accumulators and identical BF16 H operand layout. Their real16384-cell CuTe
map and wrong-value/missing-thread negatives pass. Source-bound1..8chunk
pipeline progress and5source negatives pass, not a memory-order proof.

Allfour native bodies preserve matrix/TMA work and emit paired16-instruction
register-source HGMMA groups before one full retirement. Six native negatives
include the unchanged old body, a restored intervening wait and a wrong H
register. Retirements18->16(noinitial)/22->18(initial); noC7512.
The noinitial BF16 conversion counts barely change: packed384->382, scalar
128 unchanged. The compiler had already reused much of the old conversion;
do not claim a halved conversion cost from C++ source count. Initial-state
stack32->48B and local-load/store growth are retained as costs.

Local/remote native SHA:
f708b17c615b3cd7ba1439751a2bbd2debe57c984d37faee347158c24c1fb018.
Device14fixedCPUparentraw cases+2overflow stressesPASS, all8direct repeats
and48captured outputs/cell rawPASS. CPU harness69testsPASS withCUDAhidden.
PPU CUTLASS3.6 CUDA source-checkPASS; nativePPU17SKIP(unavailable).

Nine alternating/reversed complete-forward graph samples/arm,16launches/sample.
These are screening spans, **not** replacements for frozen nsys sums:

| Workload/gate | S24 us | S49 us | S38 us | S49 vs S24 |
|---|---:|---:|---:|---|
| B1/T2048,-.1 |115.750|115.820|90.894|UNRESOLVED|
| B1/T2048,-1 |116.124|116.406|91.170|UNRESOLVED|
| B2/T2048,-.1 |114.962|115.412|186.408|UNRESOLVED|
| B2/T2048,-1 |116.478|117.240|189.556|UNRESOLVED|
| B1/T8192,-.1 |448.614|448.528|356.082|UNRESOLVED|
| B1/T8192,-1 |449.058|448.906|357.130|UNRESOLVED|
| B1/T2048/Hv16,-.1 |113.652|114.782|89.564|PARENT-WINS|
| B1/T2048/Hv16,-1 |113.770|114.778|89.572|UNRESOLVED|

No new incumbent winner; reject rather than run a full reference matrix for
a nonfinalist. This falsifies the claim that removing these completion points
alone closes the high-head gap, not every possible async schedule.
Root:/workspace/gdn-sm90-paired-state-20260926.

## S50: retain both live delta operands, overlap O2 and KV

Source170f34f, independent S24 parent. O2 starts, then unchanged H decay and
delta scaling can execute while O2 is pending. Scaled delta uses a separate
buffer; original delta remains live. KV is issued, both groups retire, then
output is published. Ordered-WG turns and all data barriers remain.

Actual8192delta coordinates and two negativesPASS. Four native bodies show
four O2/KV paired epochs with disjoint live operand registers, four fewer
full completion boundaries and unchanged matrix/TMA/publication work.
NoC7512. Noinitial stack24->16B, but initial-state stack32->104B and spills
grow sharply: this is not a universal resource improvement.
The progress checker initially assumed two output stages; its source binding
correctly failed. Actual S24 StagesO=1; after using that source authority,
all reachable reduced data-ring states1..8chunks pass. This is not CUDA
memory-order proof; fixed device numerics/replay and eight-cell screen are
still required. No measured S50 speed yet.

Prior S47/S48 evidence archive, local and remote:
/workspace/gdn-sm90-prepared-aux-evidence-20260927T0038Z.tar.gz
SHA25658d6ecdfac941bee2092c646ee37210b58e2b491e2f505ea03b56a19d6f4dc51.
No clock/power/reference/tolerance/default/SM80 changes.
