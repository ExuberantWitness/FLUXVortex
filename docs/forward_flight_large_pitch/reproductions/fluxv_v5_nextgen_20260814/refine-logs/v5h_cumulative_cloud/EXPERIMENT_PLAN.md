# Experiment Plan: FluxV v5h cumulative-cloud v2

**Problem:** The audited v1 path transports only one released rVPM cloud and
intentionally blocks step 2 to step 3; resetting the mapper or advancing only
the newest release would give a false multi-release result.
**Method thesis:** preserve Ptera/FluxV as the only load owner while DVM supplies
source facts and rVPM advances an exact, release-ordered cumulative cloud in one
shared LSRK3 velocity field.
**Date:** 2026-08-15

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1. Releases 1..k can be appended exactly and advanced continuously in one rVPM field. | This closes the current step-2-to-step-3 blocker without resetting state, dropping old particles, or double-owning topology. | v2 single-release is bitwise equal to v1; at least four releases complete with additive counts, immutable old prefixes, exact lineage, combined-field stage replay, rollback and exact-once. | B1, B2 |
| C2. A fixed SI birth core and independently refined deposition spacing/time step have a common bounded limit. | It rules out the previous false stability caused by changing the regularized model with h. | Straight/taper/twist all pass the frozen h/time refinement family with the same birth sigma and no core selection. | B3 |

**Anti-claim to rule out:** the apparent success comes from advancing only the
latest release, transporting old and new clouds independently, welding or
cancelling particles across releases, changing sigma with h, or reading the
DVM continuous absolute birth instead of the attested rVPM frontier.

## Paper Storyline

- Main evidence must prove exact cumulative state ownership and a common
  fixed-core mechanical limit.
- Appendix evidence can include attack matrices, core-family diagnostics and a
  TEV-only smoke.
- Ptera loads, target-paper accuracy, viscous corrections and parameter search
  are intentionally cut from this experiment.

## Experiment Blocks

### B1. v1 exact reduction and schema attacks

- **Claim tested:** C1.
- **Why this block exists:** a new cumulative container must not silently alter
  the already audited single-release result.
- **Task:** generic straight two-cell ribbon, one active release.
- **Compared systems:** frozen passive-frontier v1 versus cumulative v2.
- **Metrics:** bitwise positions, circulation vectors, sigma, IDs, lineage and
  frontier facts; exact-off disabled reduction.
- **Setup:** `U=2 m/s`, release `dt=0.02 s`, `sigma_birth=0.085 m`, `h=0.04 m`.
- **Success criterion:** all physical arrays are `array_equal`; copied,
  reordered, 1-ULP-tampered, stale or cross-wing objects fail before commit;
  failure permits a clean retry.
- **Failure interpretation:** STOP v2; the container or attestation changes the
  frozen v1 model.
- **Target:** mechanical gate table, not a paper-performance table.
- **Priority:** MUST-RUN.

### B2. Multi-release lifecycle and one-field ownership

- **Claim tested:** C1.
- **Why this block exists:** it directly closes the current third-release
  blocker.
- **Task:** straight two-cell three/four-release sequences plus
  `active -> inactive -> restart -> continuous`.
- **Compared systems:** cumulative one-field transport; independent old/new
  transport is a negative control and must not match accidentally.
- **Metrics:** particle counts per release slice, immutable old prefix, IDs,
  lineage/time continuity, one-third continuous birth, stage-pre velocity
  replay, exact-once, rollback and deterministic fresh replay.
- **Success criterion:** counts are exactly additive; no sort, welding,
  cancellation, deletion or remeshing; a direct independent replay of the
  combined LSRK3 stages is bitwise equal.
- **Failure interpretation:** STOP before geometry expansion; a multi-release
  result would be phantom state continuity.
- **Target:** lifecycle ledger and failure-analysis figure.
- **Priority:** MUST-RUN.

### B3. Fixed-core time and spacing refinement

- **Claims tested:** C1 and C2.
- **Why this block exists:** topological correctness alone does not establish a
  stable transport limit.
- **Task:** straight/taper/twist, three active releases,
  `h=(0.04,0.02,0.01,0.005) m`, substeps `(1,2,4)`.
- **Metrics:** frontier and fixed-probe relative L2, error-reduction ratios,
  positive finite sigma, particle invariants, edge/vector conservation and
  semantic replay hash.
- **Setup:** `sigma_birth=0.085 m` for every release and every h; core multipliers
  `(0.5,1,2)` are diagnostic-only and cannot select a result.
- **Success criterion:** fine time error `<=1e-6`, fine h error `<=1%`, adjacent
  error ratio `>=1.5`, and every mechanical/integrity gate passes in all three
  geometries.
- **Failure interpretation:** STOP cumulative promotion; diagnose topology,
  stage-field stiffness or inadequate quadrature without tuning on target data.
- **Target:** main mechanical convergence table.
- **Priority:** MUST-RUN.

### B4. Artifact and ownership integrity

- **Claims tested:** C1 and C2.
- **Why this block exists:** a GO must be independently reconstructible rather
  than derived from hard-coded counters.
- **Task:** two fresh full runs to new directories plus an adversarial audit.
- **Metrics:** raw-array recomputation, semantic digest equality, per-run
  provenance, strict JSON, source/result SHA closure, runtime guarded call
  counts and STOP-artifact behavior.
- **Success criterion:** semantic payloads match; UUID/time/path differ; no
  target/Ptera solver/load call; tampered arrays/manifests fail closed.
- **Failure interpretation:** retain only a development result; no promotion.
- **Target:** audit appendix.
- **Priority:** MUST-RUN.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Schema and exact reduction | v1-v2 straight one-release parity plus attacks | Bitwise physical equality and transactional failure | local CPU, minutes | circular attestation or copied live objects |
| M1 | Close the third-release blocker | straight three-release, then fourth-release smoke | additive slices, immutable prefix, combined-field replay | local CPU, minutes | old/new field double count or latest-layer-only advance |
| M2 | Exercise lifecycle | active/inactive/restart/continuous and partial activity | exact frontier ownership and no phantom release | local CPU, minutes | restart consumes the wrong frontier fact |
| M3 | Full non-target gate | 3 geometries x 4 h x 3 time resolutions | every preregistered convergence/mechanical gate passes | bounded local CPU, target <30 min | O(N^2) cloud cost or stretching stiffness |
| M4 | Integrity and audit | two fresh artifact runs and fresh reviewer | raw recomputation, semantic equality, SHA closure | local CPU, minutes | provenance contaminates deterministic payload |

## Compute and Data Budget

- GPU-hours: `0`.
- Data preparation: none; generic synthetic geometry and source-only DVM facts.
- Human evaluation: none.
- Runtime budget: first three smoke runs before the 36-configuration gate;
  full local CPU budget capped at 30 minutes unless the user explicitly expands
  it.
- Biggest bottleneck: direct rVPM interaction cost grows approximately
  quadratically with cumulative particle count.

## Risks and Mitigations

- **Cross-release double ownership:** exact release slices, immutable prefixes,
  one combined stage field and negative independent-cloud control.
- **Attestation cycle/replay:** live handoff tokens, parent digests, role-aware
  global exact-once and transactional registries.
- **False convergence from changing core:** freeze `sigma_birth` in SI units and
  vary h independently.
- **Memory/runtime explosion:** preregister a particle-count/resource ceiling;
  crossing it is STOP, not permission to merge or delete particles.
- **Scope creep:** no Ptera solver/load, target observations or paper scoring in
  this experiment.

## Final Checklist

- [x] v2 single release is bitwise equal to v1
- [x] three and four release exact-append paths pass
- [x] lifecycle and partial-activity gates pass
- [x] 36/36 configuration-mechanics, 108/108 release rows and every time family
  pass
- [ ] straight/taper/twist fixed-core spatial refinement passes — FAIL at the
  first fixed-probe ratio for straight (`1.4507852205964775`) and taper
  (`1.4964923945744306`), both below `1.5`; twist passes
- [x] two fresh semantic artifact replays agree and the fresh audit passes with
  an eager-import/runtime-closure warning
- [x] zero target observations are read and zero guarded Ptera/target solver,
  load or builder calls occur
- [ ] nice-to-have TEV/joint-family tests do not delay the core gate

## Frozen Result and Claim Decision

The full gate is durably recorded at
`runs/20260815_fluxv_v5h_cumulative_cloud_gate`.  Its frozen SHA-256 identities
are:

| Item | SHA-256 |
|---|---|
| cumulative runner | `155e1ea8c704c5f58e077eca59f67b54b908be54e552c1b40bd595a78750a02b` |
| runner tests | `86b0ae5b6d9d7b8f739878fc9d6b3e46c05dd28589a3d28d1387e94557095256` |
| cumulative transport core | `8b4c3efb19293952a854308508d5d76d55f95268a0ee03acd916eb13287f3d49` |
| semantic result | `cc8666ad43ea291aae03a2b1b7e549e8dd6711b2e1c3006a07e293d5437a4759` |

M0--M2 pass.  In M3 all 36/36 configuration-mechanics rows, all 108/108
release rows and all time-refinement families pass.  Twist also passes spatial
refinement.  Straight and taper remain monotone and have fine errors below the
frozen limit, but their first fixed-probe reduction ratios are respectively
`1.4507852205964775` and `1.4964923945744306`, short of the preregistered `1.5`.
The result is therefore strict `STOP`: cumulative ownership claim C1 is
supported; common-limit claim C2 is not supported.  No threshold or grid was
retuned.  Runtime evidence establishes zero target-data reads and zero guarded
Ptera/target solver, load or builder calls.  The audit remains
`PASS-with-WARN` because package initialisation eagerly loads definitions and
the declared manifest does not claim complete runtime import closure.

## Preregistered Next Candidate: v5h2 Dyadic Panels

### Independence and hypothesis

v5h2 is a new deposition API, not an edit to the frozen v2 evidence or a
post-hoc relaxation of C2.  Its hypothesis is that the two near-threshold first
ratios reflect non-nested per-edge `ceil(L/h)` panel families.  The test removes
that one confound by making every edge's spatial family exactly nested; the
cumulative ownership/transport implementation, physical core, observables and
all acceptance thresholds remain unchanged.

### Frozen construction

For every physical edge of length `L`, define

`n0 = max(1, ceil(L / 0.04 m))`

and

`n(level) = n0 * 2**level`, for `level in {0, 1, 2, 3}`.

The realized per-edge spacing is `L/n(level)`.  Birth smoothing is fixed at
`sigma_birth=0.085 m` for every level, geometry and release.  The new API must
expose the edge key, `L`, `n0`, level, realized `n`, realized spacing and the
parent attestation; it must not infer a dyadic level from global particle
counts or modify the frozen v2 API in place.

### Unchanged matrix and gates

- geometries: straight, taper and twist;
- spatial levels: `0, 1, 2, 3`;
- transport substeps: `1, 2, 4`;
- active releases per configuration: exactly `3`;
- total: 36 configurations and 108 release rows;
- time fine-error limit `<=1e-6`, spatial fine-error limit `<=1%`, adjacent
  time and spatial error-reduction ratio `>=1.5`;
- all topology, exact-once, lineage, conservation, lifecycle, resource,
  runtime-boundary and strict-artifact gates from v2 remain mandatory.

### Additional v5h2 gates

1. Level 0 physical arrays and release ledgers are bitwise equal to the frozen
   v2 `h=0.04 m` result for every geometry and time resolution.
2. For every edge key independently, `n(level+1) == 2*n(level)` for all three
   transitions; missing, split, welded or cross-edge count evidence fails.
3. Two fresh full runs have identical semantic payload digests while UUID,
   clocks and output paths remain distinct.

### STOP conditions

Immediate `STOP` is required for any level-0 parity difference, per-edge count
that is not exact dyadic doubling, change to `sigma_birth`, geometry/time/
release matrix, observables or the `1.5` threshold, any failed mechanical,
time, spatial or resource gate, non-identical fresh semantic digests, manifest
or raw-recomputation failure, target-observation read, or guarded Ptera/target
solver/load/builder call.  A v5h2 `GO` would establish only a non-target
mechanical limit and would not authorize paper scoring or Ptera feedback by
itself.

## v5h2 Executed Result

The candidate was executed exactly as preregistered.  All 9/9 level-0
geometry/time rows reduce bitwise to the frozen v2 physical cloud, frontier,
probe and release ownership ledger.  Per-edge panel
counts obey exact dyadic doubling at all three transitions, while
`sigma_birth=0.085 m`, the three geometries, time grid, three-release history,
observables and thresholds remain unchanged.

All 36/36 configurations, 108/108 release ledgers and 12/12 time-refinement
families pass.  At finest time resolution, adjacent spatial-error reduction
ratios are:

| geometry | frontier ratios | fixed-probe ratios | finest-level particles |
|---|---:|---:|---:|
| straight | `4.007144`, `4.001804` | `4.001871`, `4.000472` | `816` |
| taper | `4.007393`, `4.001866` | `4.002685`, `4.000675` | `888` |
| twist | `4.007177`, `4.001812` | `4.002384`, `4.000600` | `816` |

Every ratio exceeds the frozen `1.5` minimum and every fine difference is
below its preregistered limit.  Two independent full processes produced
byte-identical semantic payloads with SHA-256
`dece97d1d708056659bcaeb5cd82dd189ba834b0f97665b7de7271f7413245e4`;
their UUID, UTC time and output directories differ.  Raw arrays, per-release
dyadic panel ledgers, disk gate recomputation, source closure and result hashes
all verify.  A third independent execution published the formal bundle at
`runs/20260815_fluxv_v5h2_dyadic_cumulative_cloud_gate` with the same semantic
digest.

The claim decision is therefore: C1 **yes** and dyadic-C2 **yes**, limited to
the frozen non-target mechanical matrix.  The original non-dyadic v2 `STOP`
remains historical evidence and is not relabelled.  v5h2 does not establish
surface-load accuracy, general three-dimensional stability, or agreement with
Yang/Izraelevitz/Baik; it does not authorize Ptera feedback or paper scoring.

## Preregistered successor: v5h3 native Ptera feedback vertical slice

- **Run tier:** auxiliary/dev; non-target mechanics only.
- **Research question:** can the live v5h2 cloud enter the existing FluxV
  `UVPMHybridSolver` collocation and native load-velocity paths exactly once,
  without creating a second circulation, TE-wake, or force owner?
- **Null:** the additive field has a time-layer, identity, count, residual, or
  exact-reduction failure.
- **Alternative:** hard-off and empty-cloud cases reduce bitwise to FluxV, and
  a step-aligned nonzero cloud changes only the Ptera-native bound solution and
  KJ load velocities through one independently replayable Gaussian-erf field.
- **Baseline:** unmodified `fluxvortex.solver.UVPMHybridSolver` with its
  prescribed Ptera ring wake and one-way legacy VPM path.
- **Minimal run:** generic straight wing, steps 0--2, prebuilt live v5h2
  reports at `t_1` and `t_2`, with the frozen one-based DVM step equal to the
  zero-based Ptera step plus one; both circulation signs, no target data.
- **Primary keys:** off/empty bitwise equality; collocation call count `1`;
  load-leg call count `4`; no-penetration residual `<=1e-12`; direct-field
  replay residual `<=1e-13 m/s`; Ptera load-owner count `1`; extension
  force/write count `0`; wake/legacy-VPM unchanged.
- **Abandonment condition:** any mandatory STOP in PLAN section 14. No retry
  may change core, report, target geometry, load formula, or tolerance.
- **Next boundary after a pass:** preregister bound/wake-to-rVPM transport
  feedback. Paper scoring remains forbidden until that separate gate passes.

### v5h3 vertical-slice outcome

The preregistered straight three-step slice passed. Hard-off and enabled-empty
reduce bitwise to FluxV; both signed clouds and the two-report time chain pass;
the measured per-active-step ledger is exactly `1` collocation evaluation,
`4` ordered Ptera load-leg evaluations, and `1` native load-processor call.
The maximum no-penetration residual is `2.7755575615628914e-17`, with zero
extension force/moment/load writes. The result is
`go_one_way_native_feedback_mechanics_only` at
`runs/20260815_fluxv_v5h3_native_feedback_vertical_slice`. Two-way transport
feedback and all paper scoring remain blocked.

## Preregistered successor: v5h4 reverse transport velocity

Test one frozen Ptera step as a read-only external field for one live v5h2
cloud update. The baseline is the exact v5h2 self+freestream LSRK3 step; the
active alternative adds parent-only Ptera bound/ring-wake/freestream velocity
and its central-difference spatial Jacobian. Freeze relative perturbations
`2^-8`, `2^-10`, and `2^-12` against `min(sigma_min,c_ref)`, with `2^-10`
nominal. Required evidence is exact disabled/zero reduction, no parent writes,
full stage-pre replay, finite positive state, and a common second-order
Jacobian/transport limit. Any double-counted self/freestream term, parent
mutation, convergence failure, or target access is STOP.

### v5h4 frozen-field outcome

The one-step slice passes with verdict
`go_frozen_parent_transport_mechanics_only`.  Disabled and zero-external
reductions are bitwise exact; the nominal live row advances 208 particles in
three stages with 3 parent center calls, 18 finite-difference calls, zero
parent/load writes, and bitwise independent stage replay.  Parent state is
unchanged.  The parent-Jacobian and full-state epsilon-family ratios are
`15.2037`, `15.9794`, `15.7699`, and `15.9421`, close to the theoretical
factor 16 for a fourfold epsilon refinement.  The formal-run operational floor
is 12; the preregistration required a common second-order limit without a
numeric ratio floor.  Artifact:
`runs/20260815_fluxv_v5h4_frozen_ptera_rvpm_transport`.  This remains a frozen
one-step result and does not authorize target scoring.

## Preregistered successor: v5h5 synchronized time march

Use one ordered partitioned layer: append DVM release at `t_n`; solve Ptera
with that stationary post-release cloud; advance the entire cloud once using
the frozen solved Ptera field plus the rVPM self field; publish the same-stage
frontier at `t_(n+1)`.  Do not call the old v5h2 self+freestream transport and
then v5h4, because that would double the transport/freestream owner.  The
minimal gate is a generic straight wing with first/continuous/continuous
releases over three Ptera layers.  It requires additive slices, immutable old
prefixes before each advance, exact step/time lineage, one Ptera solve/load
owner, one combined LSRK3 update, and hard-off reduction to each frozen
one-way parent.  Any reset, replay, doubled term, mismatched frontier stage,
mutation, non-finite value, target access, or failed residual is immediate
STOP.  Taper/twist refinement and paper scoring remain out of scope.

### v5h5 synchronized outcome

The preregistered generic straight-wing slice passes. Three attested
node-local DVM layers produce first/continuous/continuous placement and append
102 particles per layer, yielding 102/204/306 particles. The post-release
cloud is used by the native Ptera collocation and four load-velocity batches
before one combined LSRK3 update; no v5h2 pre-transport is called. Old-prefix
and frontier replay residuals are bitwise zero, the maximum no-penetration
residual is `2.7755575615628914e-17`, and all extension/parent/load write
counters are zero. Independent complete reruns are deterministic. The
feedback-only and transport-only reductions are bitwise exact against their
full/native Ptera comparators. Artifact:
`runs/20260815_fluxv_v5h5_synchronized_smoke`; verdict:
`go_three_layer_synchronized_mechanics_only`.

This supports only the ordered three-layer mechanical claim. It does not
support taper/twist refinement, general three-dimensional stability,
aerodynamic accuracy, or Yang/Izraelevitz/Baik scoring. Those remain explicit
future gates.

## Preregistered successor: v5h6 generic geometry/refinement

Hold the v5h5 release/Ptera/combined-transport/frontier owner order fixed and
test straight, linearly tapered, and linearly twisted four-cell generic wings.
Ptera and both node/cell DVM source layers must use one shared geometry law.
At a fixed `0.06 s` horizon, `sigma_birth=0.085 m`, and base spacing `0.04 m`,
run dyadic levels 0/1/2 at `dt=0.02 s`, then time steps
0.02/0.01/0.005 s at level 0.  Gate the final frontier displacement and five
fixed-probe induced velocities independently: fine relative difference
`<=0.02`, consecutive-difference ratio `>=1.25`, finite and nondegenerate for
every geometry.  Preserve all v5h5 mechanics, residual, ownership, resource,
and no-target gates.  Missing rows, post-hoc parameter selection, or any gate
failure is STOP.  Even a pass remains a non-target mechanical refinement
result and does not authorize paper scoring.

Source-only executability is frozen at constant `60 deg` incidence.  Before
any coupled result was read, `35 deg` failed the required all-active source
contract and the standard feasibility set `(45,60,75,90) deg` identified
`60 deg` as its lowest value that preserves `first -> continuous` across all
preregistered chord/time-step combinations.  This is an activity-only check,
not a coupled-response or target-data optimization; changing it after the
first coupled rerun is STOP.

## v5h6 frozen result and next decision

The 15 unique configurations all pass mechanical ownership and residual
checks, but the full gate is `STOP`.  Straight spatial frontier/probe and
taper/twist spatial frontier converge; taper/twist fixed-probe fields fail.
All frontier and probe time families fail.  The formal summary SHA-256 is
`55ac0c4e68df13559393861d48d70f37f11e620628af30fbb8758c2861f5fff8`.
A second fresh run is byte-identical and independent disk recomputation agrees
with every stored ratio and boolean.  No target data were read.

The newest-frontier observable is not a fixed-age time-refinement quantity:
it is advanced only for the current `dt`.  More importantly, fixed probes
also fail, so metric repair alone is insufficient.  Before any further
geometry or target run, v5h7 must use a manufactured temporal-topology oracle
to distinguish independent shrinking closed rings from a connected sheet.
Freeze fixed-age tracer, vorticity impulse and fixed-probe observables; require
one temporal shared-edge owner and an explicit old-particle update/remesh
contract.  No threshold/core adjustment or paper scoring is allowed.

### v5h7 frozen manufactured matrix

Use `T=.08 s`, `U=2 m/s`, span `.60 m`, constant increment rate
`.8 m2/s2`, `sigma=.085 m`, spacing `.02 m`, and
`dt=(.02,.01,.005,.0025) s`.  Compare topologically independent rings of
strength `DeltaGamma=.8 dt` with a shared-node temporal sheet whose panel
strength is cumulative, `(i+1)DeltaGamma`.  Gate exact conservation and edge
incidence, the analytic impulse
`Iz=-.8*2*.6*(T^2+T*dt)/2`, independent field/impulse halving in
`[.49,.51]`, connected impulse first-order convergence, connected fixed-probe
difference ratios in `[1.8,2.2]`, and agreement of successive first-order
Richardson probe extrapolants to `<=.01` relative with nonzero limit.  Cap the
cloud at 5,000 particles.  This oracle contains no Ptera solve, DVM target
case, load, feedback or paper observation and cannot itself authorize a
production temporal-sheet owner.

## v5h7 frozen result and v5h8 decision

The manufactured v5h7 oracle passes.  Independent closed rings halve their
impulse and fixed-probe norm under every `dt` halving, while the connected
cumulative-strength sheet matches its analytic impulse and approaches a
nonzero probe field at first order.  Connected probe difference ratios are
`2.00525` and `2.00281`; Richardson extrapolants differ by `7.50e-4`
relative.  The deterministic summary SHA-256 is
`5b3c925fa9065726c4e2c4cd24ff89acd50aea377bf00875bbb5d4d40210bd0b`.

Production promotion remains blocked.  v5h8 must compare a causal incremental
construction with a direct full connected graph for one-to-four releases.
The new panel's upstream contribution must reuse the exact live transported
boundary facts so that old `-Gamma_prev` plus new `+Gamma_current` yields the
source increment without mutating the old prefix.  Test zero transport first,
then uniform/affine deformation including sigma/gamma evolution.  If exact
boundary reuse fails, choose and audit an explicit conservative remesh/update
owner; do not preserve the wording "exact append" by hiding particle edits.

Freeze the v5h8 manufactured inputs at span `.60 m`, panel displacement
`.04 m`, increment `.016 m2/s`, birth sigma `.085 m`, spacing `.02 m`, four
releases and the v5h7 probes.  The affine material map per release is
`A=[[1,.015,0],[0,1,.01],[.02,0,.99]]`, translation
`(.04,-.002,.003) m`, gamma map `A`, and sigma factor `1.01`.  Exact append is
tested only through particlewise reuse of the old live downstream boundary;
fresh geometry redeposition is intentionally a negative control.  Required
residuals are `2e-14` for field/impulse parity and `1e-14` for clone relations,
with a 1,000-particle cap.  A pass supports only inherited-material-basis
algebra under this affine family, not generic rVPM/non-affine physical parity.

## v5h8 frozen result and production decision

The v5h8 oracle passes with status
`go_v5h8_bounded_affine_live_basis_mechanics_only`.  Across releases one
through four, zero-transport velocity-field, Jacobian and impulse residuals
are at most `6.938893903907228e-18`, `1.249000902703301e-16`, and `0`.
The bounded affine live-basis clone versus algebraically collapsed comparator
closes to `6.938893903907228e-18`, `2.7755575615628914e-17`, and
`4.336808689942018e-19`.  Exact old-prefix, clone relation, finite sigma,
rollback, cap and no-target/Ptera/load gates all pass.

The ordinary fresh-geometry append is not an admissible substitute.  Its
relative field differences at affine releases two through four are
`0.0011467742630084674`, `0.0014549564075497143`, and
`0.0013029774819262976`, all well above the frozen `1e-8` negative-control
floor.  The result therefore supports a **GO** only for the restricted
inherited live-material-basis clone and a **NO-GO** for ordinary fresh append.
It does not establish generic rVPM or non-affine support compatibility.

The formal bundle is `runs/20260815_fluxv_v5h8_incremental_sheet_oracle`;
its summary SHA-256 is
`af38f29d8da8298d9ca9444e6210791bac2a4a65ca3920578e8ba9c9066dfb91`.
The module/module-test/runner/runner-test SHA-256 values are
`f30b5fbf6d2f1718bbbecec669a47dfb1c3c001942d2212ce814098966a610e2`,
`db0ee997688e3f65802ab4345721f07337f1c9f2622548d600f9843c8c2ca710`,
`f2079cb457b883fd35892434745d8a3e96c6a6587ba23f1caf710193723f6e98`,
and `1f402bca2944c1f4e87e0d0f975d3713cb0d092b2c94e3bb876338f24b0be953`.
Thirty-eight module, nine runner and 131 joint tests pass.  The fresh audit
passes with a `cap-after-allocation` warning, which must be resolved by
rejecting oversized candidates before allocation.

Production promotion remains blocked.  The next mandatory experiment block
must implement and audit an explicit conservative boundary update/remesh
owner for general transported sheets.  It must own support changes rather
than hide them as exact append, conserve circulation and particle moment,
bound long-time cancellation, enforce caps before allocation, roll back
transactionally, and remain observation-free until its mechanics gate closes.
