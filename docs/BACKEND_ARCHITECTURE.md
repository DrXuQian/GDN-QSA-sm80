# One repository, independent algorithms and execution families

Status: design only, 2026-09-26 UTC. No kernel, numerical threshold, build target
or default routing is changed by this document. The published legacy PPU
`residual-solve-static` experiment remains pending device admission.

## Decision

Integrate **complete implementations behind a common operator contract**, not
different algorithms inside a common mainloop. Sharing a repository does not
require sharing a chunk size, kernel count, workspace layout or synchronization
protocol. Keep the current SM80 / legacy PPU implementation independently usable
and testable when an SM90 implementation is added.

There are three independent choices:

| Choice | Examples | Owns |
|---|---|---|
| Algorithm and numerical contract | original scan/reset; materialized W/U; inverse-residual; future fused chunk algorithm | Recurrence realization, rounding boundaries, approximation eligibility, complete execution graph |
| Execution family | SM80 warp MMA; legacy PPU AIU; SM90 TMA/GMMA | Copy/MMA atoms, operand layouts, pipeline, barriers, resource allocation |
| Target and toolchain | CUDA SM80; CUDA SM90; PPU1.0; PPU1.5; PPU1.7 | Compiler, dependency revision, runtime, native feature admission |

Only implement useful, proven combinations. **No requirement to port every
algorithm to every target.** PPU1.7 and NVIDIA SM90 can share algorithm/pipeline
source where verified; they retain different target identities and admission.
PPU1.0/1.5 are not called SM80 hardware just because some source originated there.

## What is shared, and what is private

Share:

- Forward operator semantics: Q/K/V, log-decay convention, beta, normalization
  and scale, head/GVA mapping, optional initial/final state, shapes/strides,
  dtypes, and the associated numerical-policy identity.
- Independent recurrent reference, fixture generation, numerical evaluation,
  input validation, benchmark protocol and result format.
- A registry of available complete implementations and admitted configurations.
- Pure index/math helpers only when their semantic contracts actually match.

Keep private to each implementation:

- Chunk and superchunk sizes, stage boundaries and number of launches.
- Materialized W/U versus residual realization; intermediate dtype and rounding.
- State representation, scratch layouts, TMA descriptors, shared-memory swizzle,
  producer/consumer roles, barriers, persistent scheduling and native atoms.
- Workspace allocation plan and lifetime. No SM80 workspace may be handed to an
  SM90 plan merely because both implement GDN.

For example, the old path may use `prepare -> solve -> state -> output`, while
a new SM90 path fuses several of those operations or uses a different scan.
Both expose one forward operation. Do not make the new path materialize W/U just
to satisfy the old stage interface. Do not feed old prepare output into a new
state kernel without an explicitly versioned, proved common intermediate ABI.

The current original public entry has narrower features than the experimental
WY/residual entries, including initial-state handling. A common API must report
these capabilities, not silently drop state or imply all features already work.

## Repository layout and interface

The following is a proposed destination, not an immediate mass rename:

```text
gdn_qsa_sm80/                 existing public imports, retained for compatibility
include/gdn/                 plain operator/plan/identity contracts
csrc/gdn/
  dispatch/                  host selection of a complete implementation
  algorithms/
    scan_reset/              current scan/reset mathematics and implementations
    wy_residual/             existing W/U and residual contracts, kept distinct
    fused_chunk/             future independently admitted algorithm
  targets/
    cuda_sm80/
    ppu_aiu/                 explicit 1.0/1.5 capability sets
    cuda_sm90/
    ppu17/
third_party/
  actlize/                   legacy PPU authority
  cutlass_cuda/              pinned CUDA dependency, as needed
  cutlass_ppu36/             pinned PPU1.7 fork
tests/                       common semantic suite + implementation-specific tests
benchmarks/reference_adapters/  optional cuLA / FlashInfer / FLA comparisons
```

Within an algorithm, execution-family files remain separate; an algorithm may
have only one implementation. Do not pre-create empty copies for all families.
No new repository or package rename is necessary to introduce this boundary.

A complete implementation provides the conceptual operations
`supports(problem, policy)`, `make_plan`, `workspace_requirements`, `launch`,
and `identity`. `launch` executes its whole graph, however many kernels it needs.
This is host dispatch once per plan/call, **not a virtual call inside the GPU
mainloop**. The native interface carries plain data and an explicitly tagged
runtime context, never CuTe layouts, CUTLASS templates or unvalidated stream casts.

Plan identity includes operator, algorithm, numerical contract, execution family,
target/device, dependency revision, configuration and private workspace ABI.
Reusing a plan on a different device, contract or workspace layout fails before
launch. A user-selected implementation is explicit and cannot silently fall back.
`auto` may choose only from admitted capabilities; it reports the actual selection.

## Dependency/build isolation

User-required policy:

| Target | Dependency / execution path |
|---|---|
| PPU1.0 and PPU1.5 | actlize, legacy AIU path |
| PPU1.7 | supplied PPU CUTLASS3.6.0 fork, explicit ACOMPUTE10700 / SM90a path |
| NVIDIA SM80 / SM90 | compatible pinned CUDA CUTLASS revision(s), native CUDA path |

These are build policies, not claims that every cell already compiles or runs.
Use separate target build directories and target-specific native extensions.
Load only the appropriate extension. Each translation unit sees exactly one
CUTLASS/CuTe dependency authority; do not put actlize and PPU CUTLASS3.6.0 on the
same global include path. Isolate exported symbols and template instantiations
with hidden visibility and distinct entry names to prevent cross-DSO interposition.
Compilers and global CMake cache options must also remain isolated.

The current GDN `GDN_QSA_PPU` branch selects legacy atoms, and its CMake selects
`third_party/actlize`. It is **not** a generic selector for all PPU generations.
Introduce an explicit PPU1.7 target instead of replacing that dependency in place.
Do not fake compiler architecture macros or compile an empty guarded device body.

Local source authority inspected on 2026-09-26:

- `/root/cutlass3-3.6.0`, commit
  `023e82d03e80b4d5982f664925e15499033df314`.
- `build.sh` maps `ACOMPUTE_VERSION=10700` to `CUTLASS_NVCC_ARCHS=90a`;
  CMake otherwise defaults ACOMPUTE_VERSION to 10000.
- Standard SM90 BF16/FP16 GMMA and TMA exist separately from PPU-specific
  block-scaled/FP8/FP4 atoms. Start with BF16/FP16 forward; ignore those low-bit
  extensions for this integration.

This inspection proves a source route, not a successful PPU1.7 build or run.
Native compile, simulator correctness/performance and physical-device evidence
remain separate. Missing target capability is a reasoned SKIP, not PASS and not
a fallback to the legacy architecture.

## Reference reuse is not forced algorithm equivalence

[cuLA's repository layout](https://github.com/inclusionAI/cuLA/blob/main/REPO_LAYOUT.md)
and [KDA API](https://github.com/inclusionAI/cuLA/blob/main/cula/kda/README.md)
describe CUDA C++ SM90 fused KDA as well as a separate CuTeDSL FlashKDA path.
Its public auto entry need not select the C++ implementation. Bind comparisons
to an exact entry/revision, not just the repository name.

KDA with a per-channel gate is not automatically scalar-gate GDN. Reuse a proven
compatible recurrence specialization or the delivery/pipeline techniques; do not
register a different mathematical operator as a GDN implementation. If KDA is
ever added as a feature, give it a separate operator contract.

Current [FlashInfer GDN](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py)
uses CuTeDSL paths; [PR #3613](https://github.com/flashinfer-ai/flashinfer/pull/3613)
removed the older GDN C++ implementation in favor of that replacement. A chosen
historical C++ implementation must be revision-pinned. A DSL implementation is
an algorithm/schedule reference or optional comparison dependency, **not a new
shipping Triton/CuTeDSL dependency**. Our shipping requirement remains C++/CUDA
with the target's CUTLASS/CuTe backend.

For any adapted source, record origin revision, selected files, license/notices,
unchanged algorithm properties and local backend adaptations. Do not maintain
two full copied upstream repositories inside two algorithm directories.

## Maintenance and admission

1. **Preserve first.** Wrap/register existing entries with no math, layout or
   dispatch changes. Keep existing imports and explicit experimental entries.
   Check numerical replay and generated-body preservation before moving files;
   separate mechanical moves from algorithm changes in review.
2. **Add SM90 opt-in.** A new complete implementation gets its own workspace and
   numerical contract. Shared same-math helpers are extracted only after both
   users have matching tests, not in anticipation of hypothetical reuse.
3. **Test semantics across implementations.** Use identical Q/K/V/g/beta,
   normalization, state and output conventions; include tails, GVA, nonzero
   initial state, weak/strong/mixed decay and long sequences. Unsupported cells
   reject explicitly. An approximation such as reset/truncation has an explicit
   eligibility and error policy, never an unreported performance shortcut.
4. **Keep two kinds of equality distinct.** A delivery-only rewrite must retain
   its existing RAW-BIT oracle. A genuinely different association/algorithm must
   pass the preregistered independent recurrence and numerical gate, including
   state, plus its own replay checks. Cross-algorithm BF16 bit equality is not
   assumed. This design does not change existing thresholds.
5. **Select per target and workload.** Compare whole-forward work, including
   required adapters, initialization and all launches under the same measurement
   protocol. Keep API timing and ACU kernel sums separately named. Preserve the
   original strong-decay winner where it wins; SM90 does not inherit a legacy
   timing table or replace a path by age/name alone. Input-dependent selection
   costs and actual reset eligibility must be included.
6. **Keep regressions visible.** Common semantic fixtures apply to every supported
   implementation; pipeline/layout/coverage/native-code checks remain local to
   the relevant implementation. Wrong target, missing kernel, mismatched workspace
   ABI, omitted state and wrong gate convention must fail, not select another path.
   Compile unavailable targets as SKIP with reasons; do not advertise a device
   admission from host tests or another architecture's result.

Only admitted winners enter automatic dispatch. Unsuccessful experiments remain
explicit controls/evidence and need not bloat the default binary. A correctness
fix to a shared contract is evaluated against all registered implementations; a
TMA pipeline fix does not force editing the legacy AIU implementation.

The integration deliverable is therefore **one maintained interface and evidence
system with several deliberately independent implementations**, not a universal
kernel whose inner loop branches between every algorithm and architecture.
