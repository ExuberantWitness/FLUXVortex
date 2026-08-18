# FluxV v5a → v5b Main Experiment Plan

## 1. Objective

- run ids: `20260814_fluxv_v5a_threepaper`, then `20260814_fluxv_v5b_threepaper`
- selected idea: v5a replaces v4b global load blending with a strip-local equilibrium residual plus a convectively high-passed paired-LDVM transient residual. v5b is attempted only after v5a is frozen and adds a shared TE/LE material wake to the UVLM AIC, with LDVM reduced to a shedding law.
- user requirements: implement v5a first, then v5b, then report both together.
- non-negotiable constraints: retain UVLM; no case-ID or observation-residual fit; preserve Yang/Figure14/Baik metric contracts; do not hide failed gates; do not mix LDVM force discrepancy with v5b shared-wake force.
- research question: can the smallest ownership repair produce a three-paper Pareto improvement, and does a later shared-wake solver improve beyond it?
- null hypothesis: neither v5a nor v5b improves every frozen paper gate relative to v4b.
- alternative hypothesis: v5a passes all ownership gates; v5b then reduces residual wake-memory error without regressing the other papers.

## 2. Baseline And Comparability

- baseline id: canonical FluxV v4b full artifacts at commit ancestry `8611fa5` plus canonical Baik full artifacts at `52e979f`.
- baseline variant: same geometry, kinematics, source parameters, filters, resolution and scoring for each paper.
- datasets:
  - Yang 2025 six cycle-mean lift/drag points;
  - Scherer/Izraelevitz Figure14 14 observations and 12 unique conditions;
  - Baik W1–W4 400 unique phase samples, raw and 1 Hz filtered.
- primary metrics:
  - Yang L/D MAE [gf];
  - Figure14 CT RMSE;
  - Baik filtered CL/CD macro RMSE.
- required metric keys: aggregate, per-AoA/per-condition/per-case, Figure14 theta subgroups, Baik Q1–Q4, attached Figure11 regression, numerical sensitivities.
- comparability risks: Yang nominal four-bar vs unavailable LDS; Figure14 Cd0 source conflict; Baik free-tip surrogate vs endplate/wall apparatus; LDVM Lcrit transfer; current Baik wake non-convergence.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/forward_flight_benchmarks/fluxv_v5a.py` | absent | equilibrium/transient ledger and convective state | v5a core | qS/unit double count |
| `platform/forward_flight_benchmarks/run_fluxv_v5a_crosspaper.py` | absent | three-paper smoke/full runner | comparable evidence | source-schema drift |
| `platform/tests/test_fluxv_v5a.py` | absent | exact limits and ledger tests | guard physical contract | insufficient integration coverage |
| `src/fluxvortex/...v5b...` or dedicated benchmark solver | absent | shared TE/LE wake after v5a freeze | v5b | topology/time-layer instability |
| `platform/tests/test_fluxv_v5b.py` | absent | Kelvin/time-layer/exact reduction tests | prevent double count | Ptera internal coupling |

Old v4b code and result directories are read-only baselines.

## 4. Execution Design

- minimal experiment: formula/unit tests, then frozen-history representative smoke.
- v5a smoke:
  - Yang 5°,15°,25°;
  - Figure14 15°/45°,25°/45°,25°/90°,25°/105°;
  - Baik W2/W3.
- v5a full: all three frozen matrices plus equilibrium-only/transient-only/full ablation and one-factor sensitivity.
- v5b smoke: Ramesh 2-D golden → no-LEV exact reduction → one-strip finite wing → same representative paper cases.
- v5b full: only after smoke conservation and finite outputs; same full matrix and metric contract.
- expected outputs: source/test hashes, commands/logs, manifests, phase/mean CSV, metrics JSON/CSV/MD, PNG/PDF comparison figures, summary and audit.
- v5a stop condition: any two papers fail their primary direction at representative smoke, or ledger/exact-limit tests fail.
- v5a abandonment condition: full result cannot meet all three hard gates without selecting lambda/Lcrit from observed target errors.
- v5b stop condition: no-LEV exact reduction, Kelvin/time-layer, or force-owner contract fails; numerical instability repeats after two discriminative fixes.
- strongest alternative hypothesis: v4b gains are mostly paper-specific; a universal low-order ownership model cannot improve all three without section-specific calibration.

## 5. Runtime Strategy

- environment: local CPU, `/home/exuber/anaconda3/envs/fluxvortex/bin/python`.
- smoke command: frozen in runner `--quality smoke`; exact command recorded after CLI lands.
- main command: frozen in runner `--quality full --output-dir <new-empty-dir>`.
- expected budget: v5a smoke hours/sub-hour depending on history reuse; v5a full tens of CPU-hours; v5b smoke/full potentially several days.
- artifact root: `docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/runs/`.
- safe efficiency: reuse frozen UVLM histories when mathematically valid; serialize cases; cache only source-stable kinematics/GT, never scores.
- monitoring cadence: 60 s initially, then 120/300/600 s; no blocking wait above 60 s.
- kill/relaunch: non-finite load, singular solve without new diagnostic evidence, artifact/source-hash mismatch, or process no-progress beyond expected case runtime.

Current tool limitation: the experiment skill's `bash_exec/artifact` interface is unavailable. Unified exec sessions and durable local logs/manifests are used instead.

## 6. Fallbacks And Recovery

- if full runner is too slow: freeze/reuse baseline UVLM phase histories and recompute only v5 residuals, with provenance linking the base hash.
- if v5a path is wrong after smoke: stop and document failure; do not tune against target curves.
- if v5b shared wake is unstable: reduce to one strip/Ramesh golden case, isolate AIC vs birth vs convection, then retry once per concrete fix.
- if full run becomes non-comparable: mark invalid, create a new empty output directory and rerun from the frozen command.

## 7. Checklist Link

- checklist: `CHECKLIST.md`
- execution outcome: v5a stopped after two-paper regression; v5b shared wake and
  single-pressure ledger were implemented, but the current-FluxV no-LEV exact
  reduction gate failed before any cross-paper scoring.
- next action: implement the shared TE/LE wake inside the Ptera-native solver and
  re-run G1 before Ramesh force parity or paper cases.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-14 | split v5a ownership repair from v5b shared wake | independent review found double-count/complexity risk | sequential evidence; no loss of comparability |
| 2026-08-14 | use existing isolated worktree before creating `run/*` full-run branch | v5a agents already started in `/tmp/fluxv-v5-nextgen` | full run will be moved/frozen to a dedicated run branch before launch |
| 2026-08-14 | stop v5a after frozen cache-adapter smoke | Figure 14 RMSE rose to 0.09470 and Baik macro CL/CD RMSE rose to 0.80165/0.40441; two papers failed the preregistered direction rule | no v5a full run or target-driven lambda tuning; preserve the failed result and proceed to v5b |
| 2026-08-14 | implement v5b as an isolated shared-wake/pressure prototype | avoid mixing v4b/v5a force discrepancies with a material LE wake | topology, time layer and one-force ledger are directly testable |
| 2026-08-14 | stop v5b before paper scoring | LEV-disabled standalone v5b differed from current FluxV by max phase coefficient 0.556435, violating the `1e-12` exact-reduction gate | Ramesh force parity and Yang/Figure14/Baik scoring remain not run; v4b remains the evidence-backed version |

## 9. FluxV v5h DVM-to-rVPM Experiment Line

- run id: `20260815_fluxv_v5h_dvm_node_birth_fixed_core_gate`
- tier: `auxiliary/dev`
- selected idea: retain Ptera as FluxV's UVLM and unique load owner; use the
  source-parity Ramesh LDVM only for LE/TE event, circulation, and relative
  birth-placement facts; use a node-owned ribbon for three-dimensional
  incidence; use the FLOWVPM-parity rVPM path only for post-birth transport.
- research question: can attested node-local DVM placement remove the old
  strip-seam ambiguity while a fixed physical smoothing radius and an
  independently refined deposition spacing produce a finite, deterministic,
  observation-free two-release mechanical limit?
- null hypothesis: the node-local handoff or fixed-core refinement family
  fails topology, conservation, time-layer, integrity, or convergence gates.
- alternative hypothesis: first/restart placement maps exactly to GP1 shared
  nodes, continuous placement consumes only an attested transported frontier,
  and the fixed-core quadrature family converges without changing the physical
  smoothing model.
- strongest alternative explanation: apparent convergence is only excessive
  smoothing at one coarse core, or the three-dimensional mapping silently
  interpolates cell-centre DVM births and recreates a seam.

### 9.1 Frozen ownership and comparability contract

- Ptera/FluxV remains the only surface-load owner; this run exports no force,
  lift, drag, pressure, polar residual, or target-paper score.
- cell-centre DVM sources own circulation; node-local DVM sources are
  geometry-only and their circulation must never enter the ribbon ledger.
- first/restart use the attested relative DVM placement; continuous release
  must use the already transported rVPM frontier and the one-third placement
  rule.  A DVM absolute continuous birth is audit-only.
- adjacent cells must consume the same attested shared-node fact.  Averaging,
  welding, and interpolation between cell-centre sources are forbidden.
- the smoothing radius is fixed in SI units for a refinement family;
  deposition target spacing is refined independently.  The prior coupled
  `sigma=2.125*h` result remains a historical STOP and is not re-labelled.
- no Yang, Izraelevitz, or Baik observations are read in this run.

### 9.2 Code and output contract

Expected code touchpoints:

- `platform/forward_flight_benchmarks/v5h_dvm_source.py` and its tests:
  frozen placement schema and producer attestation;
- `platform/forward_flight_benchmarks/v5h_dvm_node_placement.py` and its tests:
  new node-local DVM-to-GP1 adapter;
- `platform/forward_flight_benchmarks/v5h_passive_frontier_transport.py`:
  stable conservation recomputation compatible with the frozen edge bridge;
- a new isolated non-target runner and tests for the fixed-core gate.

Baseline references:

- source parity: exact LEV event mask at output rows `116..289`, 174 LEVs and
  499 TEVs in the clean-room Ramesh replay;
- topology/transport baseline: audited single-release passive-frontier v1;
- negative baseline: coupled `h,sigma` vertical gate, which honestly stopped.

Expected durable outputs under
`runs/20260815_fluxv_v5h_dvm_node_birth_fixed_core_gate/`:

- strict JSON summary with `allow_nan=False`;
- source and result SHA-256 manifests;
- per-geometry/per-refinement mechanical metrics;
- exact commands, Python/package environment, and run log;
- a concise GO/STOP report with no target-performance claim.

### 9.3 Gates and stop conditions

Minimal smoke: one straight two-cell strip, first release only, with positive
and negative incidence, disabled input-blind reduction, and direct event
attestation.  It must pass before taper/twist or transport is run.

The full auxiliary gate covers straight, taper, and twist geometries and must
simultaneously satisfy:

1. node-frame, time-layer, lineage, and event-attestation closure;
2. exact shared-node identity and zero seam/non-manifold residual;
3. Kelvin and edge-vector conservation at roundoff;
4. first/restart half-step placement equality and continuous frontier-only
   ownership;
5. fixed-core quadrature refinement with finite monotone error reduction;
6. time refinement and transported-state invariants;
7. deterministic replay and complete source/result hash closure.

Immediate STOP conditions: any source-placement tamper is accepted; node-local
circulation enters the physical ring; mixed activity is silently interpolated;
continuous placement uses the DVM absolute point; non-finite state; topology or
Kelvin failure; fixed-core quadrature lacks a common limit; or any target data
is accessed.  If stopped, cumulative-cloud v2 and all paper scoring remain
blocked.

Runtime budget: bounded local CPU smoke first, then the non-target gate only;
no target-paper matrix and no parameter search.  A cumulative-cloud v2 is a
separate later experiment after this run is durably closed.

## 10. v5h Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-15 | split physical smoothing radius from deposition spacing | coupled `sigma=2.125*h` refinement changed the regularized model while refining quadrature | new gate holds SI sigma fixed and refines h independently |
| 2026-08-15 | require node-local DVM placement facts | cell-centre interpolation would recreate the v5f span seam | circulation and geometry sources are explicitly separated |
| 2026-08-15 | keep continuous DVM birth audit-only | the transported rVPM frontier already owns three-dimensional wake position | prevents DVM/rVPM transport double ownership |
| 2026-08-15 | treat requested spacings with identical per-edge `ceil` counts as observationally equivalent | the frozen deposited state contains counts and realized particles, not redundant preimage metadata | runner records requested and realized spacing; transport rejects every state-changing mismatch |
| 2026-08-15 | accept the fixed-core auxiliary gate as `go_v1_only` | 36 configurations, raw recomputation, three fresh semantic replays and the global-exact-once repair passed | cumulative-cloud v2 is eligible; Ptera feedback and paper scoring remain blocked |
| 2026-08-15 | separate semantic reproducibility from per-run provenance | UUID, UTC, argv and output path must differ between executions while physical arrays and semantic hashes remain equal | deterministic claims no longer require hiding run identity |

## 11. FluxV v5h cumulative-cloud v2 line and outcome

The next isolated experiment is specified in
`refine-logs/v5h_cumulative_cloud/EXPERIMENT_PLAN.md`.  Its two claims are:

1. release-ordered exact append plus one shared LSRK3 field can carry the full
   old+new cloud through at least four releases without reset, deletion,
   welding, cancellation or replay;
2. the same fixed SI birth core and independently refined deposition spacing
   have a common time/space limit on straight, taper and twist geometries.

The first three mandatory runs are v1/v2 bitwise single-release parity, a
straight three-release blocker-closure smoke, and
`active -> inactive -> restart -> continuous`.  The 36-configuration full gate
is forbidden until all three pass.  Ptera solver/load calls, target observations
and Yang/Izraelevitz/Baik scoring remain forbidden throughout v2.

The pre-gates passed and the full non-target gate was therefore executed.  Its
formal artifact is
`runs/20260815_fluxv_v5h_cumulative_cloud_gate`.  Frozen implementation hashes
are:

- cumulative runner:
  `155e1ea8c704c5f58e077eca59f67b54b908be54e552c1b40bd595a78750a02b`;
- runner tests:
  `86b0ae5b6d9d7b8f739878fc9d6b3e46c05dd28589a3d28d1387e94557095256`;
- cumulative transport core:
  `8b4c3efb19293952a854308508d5d76d55f95268a0ee03acd916eb13287f3d49`.

The artifact closes at semantic SHA-256
`cc8666ad43ea291aae03a2b1b7e549e8dd6711b2e1c3006a07e293d5437a4759`.
All 36/36 configuration-mechanics rows and all 108/108 release rows passed;
every frozen time-refinement family also passed.  The twist spatial family
passed, but the first fixed-probe spatial error-reduction ratios were only
`1.4507852205964775` for straight and `1.4964923945744306` for taper, both below
the preregistered `1.5` minimum.  The honest verdict is therefore strict
`STOP`: C1 is supported, C2 is not supported, and Ptera feedback plus all paper
scoring remain blocked.  Runtime evidence records zero target-observation reads
and zero guarded Ptera/target solver or builder calls; package initialisation did
eagerly load some definitions, so this is a call/read boundary rather than a
claim of a complete import-free runtime.

## 12. FluxV v5h2 dyadic-panel next candidate (preregistered)

This is a new, isolated candidate and does not retroactively alter or rescue the
v5h cumulative-cloud v2 `STOP`.  The existing cumulative container, ownership
rules, physical core and all thresholds remain frozen.  A new deposition API
will assign an independent dyadic panel family to every physical edge of length
`L`:

- `n0 = max(1, ceil(L / 0.04 m))`;
- `n(level) = n0 * 2**level`, for levels `0, 1, 2, 3`;
- `sigma_birth = 0.085 m` for every edge, level and release.

The matrix remains three geometries (straight/taper/twist) by four spatial
levels by three time resolutions, with exactly three releases per
configuration.  The existing fine-error limits and the adjacent-error ratio
gate `>=1.5` are unchanged.  Before the full matrix, level 0 must be bitwise
equal to the frozen v2 `h=0.04 m` physical state, and every individual edge must
exactly double its subdivision count at each level.  Promotion additionally
requires 36/36 mechanics, 108/108 release ledgers, every time/space gate, and
two fresh artifacts with identical semantic digests but distinct provenance.

Immediate `STOP` conditions are any level-0 physical-array difference, any edge
that fails exact dyadic doubling, any change in `sigma_birth`, the geometry,
time or release matrix, or the `1.5` threshold, any mechanical/time/space gate
failure, unequal fresh semantic digests, artifact verification failure, or any
target-data read or guarded Ptera/target solver/load/builder call.  No target
scoring or Ptera feedback is authorized by this preregistration.

## 13. FluxV v5h2 dyadic-panel result (mechanical GO)

The preregistered v5h2 matrix completed without changing the physical birth
core, geometry set, release count, time resolutions, observables, fine-error
limits or the adjacent-error ratio threshold.  The final implementation hashes
are:

- dyadic edge bridge / tests:
  `26a101c3a4de5cf53d2f3aef92af3a6973a3c65ca1ff9423e6a8d973af94f622` /
  `72410c52fca6fd26722b5a9f6ae7deca772f8bb771456e98c917d5895f215efe`;
- dyadic cumulative core / tests:
  `db9332b72b2b13f0f83c0b7f7d66a3672bb6cd7e40dd102feefa877e474c4688` /
  `c1b163d4954e11346e092eeedbdae0225e9b4a3ca2b5ae7118d57ddfe7726171`;
- gate runner / tests:
  `b7ae44481c01a09b4eabd20c890a530d73c87f2ca4a36dea3bafe76e42362262` /
  `9913db7723d137e4fb2f9b43b1f1eaf2874477c18b687aaee7c5d406dc70355e`.

All 9/9 level-0 geometry/time rows are bitwise equal to the frozen v2 cloud,
particle IDs, lineage, release ledger, frontier and probe observables.  All
36/36 configurations and 108/108 releases passed; every
edge independently satisfies exact `n(level)=n0*2**level`; all 12 time families
passed.  The spatial frontier/probe reduction ratios are approximately
`4.007/4.002` and `4.002/4.000` for straight, `4.007/4.002` and
`4.003/4.001` for taper, and `4.007/4.002` and `4.002/4.001` for twist.  These
are comfortably above the frozen `1.5` gate.  Final finest-time particle counts
are `102/204/408/816` for straight and twist and `111/222/444/888` for taper.

Two independent full runs at
`/tmp/fluxv-v5h2-dyadic-full-20260815-{a,b}` passed disk recomputation and
SHA closure.  Their six semantic files are byte-identical at semantic SHA-256
`dece97d1d708056659bcaeb5cd82dd189ba834b0f97665b7de7271f7413245e4`,
while UUID, UTC time and output path are distinct.  The v5h2 conclusion is
therefore `go_v5h2_mechanics_only`: the non-nested `ceil(L/h)` confound is
closed and the fixed-core, cumulative transport has a common non-target
mechanical limit on this frozen matrix.  This does **not** validate aerodynamic
loads, authorize Ptera feedback, or support Yang/Izraelevitz/Baik scoring.
The third independently recomputed, project-local bundle is
`runs/20260815_fluxv_v5h2_dyadic_cumulative_cloud_gate` and closes at the same
semantic digest.

## 14. FluxV v5h3 native Ptera feedback gate (preregistered)

This is a new non-target mechanical experiment. It does not modify the v5h2
transport result and it does not authorize paper scoring. FluxV remains the
public solver; its existing `UVPMHybridSolver` continues to inherit and reuse
Ptera's native AIC, bound-circulation solve, prescribed TE ring wake, and
Kutta--Joukowski plus unsteady load calculation. The only new mechanism is an
additive velocity induced by an already transported, live-attested v5h2 rVPM
cloud.

### 14.1 First-stage coupling boundary

- Ptera uses zero-based step `n`, while the DVM/ribbon ledger uses one-based
  source steps. At Ptera step `n`, only a v5h2 report with
  `for_source_step_index == n+1` and `transport_end_time_s == n*dt` may be
  staged. Step zero therefore has no particle feedback.
- The frozen cloud at `t_n` is evaluated once at Ptera collocation points; its
  normal component is added once to the parent wake-side RHS before the native
  bound solve.
- The same frozen cloud is evaluated by native `calculate_solution_velocity`
  at the four Ptera LineVortex-centre batches used by `_calculate_loads`.
  Ptera remains the sole force, moment, panel, and airplane-load owner;
  DVM/rVPM exports no additive force term.
- Ptera's prescribed TE ring-wake algorithm and strength ownership remain
  unchanged in this first gate. Active feedback may change native bound
  circulation and therefore the parent's next TE strength and legacy
  trailing-particle shedding; those downstream values must equal a replay of
  the unchanged parent formulas. The v5h2 cloud is already transported and is
  not advanced again by the solver. Bound/wake-to-rVPM transport feedback is a
  separate later experiment.
- Disabled construction must return the exact original `UVPMHybridSolver`.
  Enabled construction with no report at a step must execute the same parent
  calls and return parent arrays without adding numeric zeros or reprocessing
  loads.

### 14.2 Mandatory mechanical gates

1. Hard-off and enabled-empty runs are stepwise bitwise equal to native FluxV
   for bound strengths, ring-wake vertices/strengths/ages, panel and airplane
   loads, and the legacy VPM state. In the active case, TE/legacy-VPM changes
   must be attributable only to the feedback-modified native bound solution.
2. A nonzero cloud has one collocation-RHS evaluation and exactly four native
   load-leg evaluations per scored step; recorded target arrays must be
   byte-equal to Ptera's live collocation/leg-centre arrays.
3. The active solve must close
   `A*Gamma + wake_parent + freestream + normal(v_rVPM) = 0` to `1e-12`, and
   an independent Gaussian-erf replay must reproduce every injected velocity
   to `1e-13 m/s`.
4. Final forces and moments remain the unchanged Ptera KJ-plus-unsteady
   formula evaluated with feedback-augmented native velocities. No second load
   processor, DVM force, pressure residual, or impulse correction is allowed.
5. Wrong step/time/wing/family, copied or stale report, non-finite cloud,
   report replay, point-order mismatch, callable/source drift, or partial
   mutation must fail closed before publishing a feedback step. A failed
   extension poisons the solver and cannot be resumed as though Ptera parent
   state had been transactionally rolled back.
6. A bounded non-target straight-wing smoke must exercise steps 0--2 and both
   feedback signs. Taper/twist and refinement remain blocked until this
   vertical slice passes. No Yang, Izraelevitz, or Baik observation may be
   read, and no target-case branch may select a parameter.

Immediate `STOP`: any hard-off/empty mismatch, feedback at step zero, a cloud
counted twice in the RHS or loads, any non-Ptera surface/load write, time-layer
or report-identity failure, non-finite state, residual above tolerance, or
target-data access. A pass establishes only native rVPM-to-Ptera velocity
plumbing, not two-way wake coupling or aerodynamic accuracy.

## 15. FluxV v5h3 vertical-slice result

The one-way native feedback slice passed without changing Ptera's AIC, load
formula, prescribed wake algorithm, v5h2 cloud, DVM threshold, or any target
case. The frozen implementation/test SHA-256 values are
`b7c938ee53a68614de515a88948a01fcb737e8fdbd1920598bb7f4a24076f6b2`
and
`65720e860337ef755dc89c8be8cfa9b2d24cf39cc281552bc1d340bea37f7f4d`.

Hard-off returns the exact native FluxV solver and the enabled-empty three-step
run is bitwise equal. For each active step the ledger records one collocation
field evaluation, four ordered native LineVortex-centre evaluations, one
Ptera load-processor call, and zero extension force/moment/load-processor
writes. Both DVM signs pass; a live step-1/step-2 cloud parent chain maps DVM
source steps 2/3 to Ptera steps 1/2. The maximum no-penetration residual is
`2.7755575615628914e-17`; direct Gaussian-erf replay closes to the frozen
`1e-13 m/s` tolerance. Eleven focused and 117 related tests pass.

The auxiliary artifact is
`runs/20260815_fluxv_v5h3_native_feedback_vertical_slice`. Verdict:
`go_one_way_native_feedback_mechanics_only`. This authorizes preregistration
of bound/ring-wake-to-rVPM transport feedback, not taper/twist performance,
paper loads, or Yang/Izraelevitz/Baik scoring.

## 16. FluxV v5h4 frozen-Ptera-to-rVPM transport gate (preregistered)

The next auxiliary gate closes the reverse velocity direction without changing
the successful v5h3 surface/load path. At Ptera step `n`, after the native
bound solve and loads have closed, the step-aligned rVPM cloud is advanced from
`t_n` to `t_(n+1)` in a single partitioned update. Ptera's bound and current
ring-wake geometry/strength state is frozen during that update.

- Particle position rate is the sum of the Gaussian-erf self field and the
  native parent-only Ptera velocity (bound + current ring wake + freestream).
  The v5h3 subclass override must be bypassed explicitly so the rVPM cloud is
  not added to itself twice.
- Particle stretching uses the sum of the analytic rVPM Jacobian and a central
  finite-difference Jacobian of that frozen parent-only Ptera field. The three
  preregistered perturbations are
  `epsilon/min(sigma_min,c_ref) = 2^-8, 2^-10, 2^-12`; `2^-10` is the nominal
  row and cannot be selected after seeing a target result.
- The complete merged cloud is advanced by one shared LSRK3 field. Ptera state
  is read-only; no bound, wake, panel, load, or legacy-VPM value may change
  during this transport call.
- Disabled external-field transport must be bitwise equal to the frozen v5h2
  self+freestream step. Zero external velocity/Jacobian is a second exact
  reduction. The active path must have exactly three parent velocity batches
  per LSRK3 substep plus the finite-difference batches declared by the ledger.
- The first run is one straight-wing, one-cloud-step mechanical slice. It must
  independently replay all stage-pre velocity/Jacobian arrays, preserve
  positive finite sigma, and show second-order finite-difference convergence
  across the three frozen perturbations before any multi-release integration.

Immediate `STOP`: parent-only bypass fails, any self/freestream term is counted
twice, a Ptera array/object changes, finite-difference rows lack a common
limit, stage replay or exact reduction fails, state becomes non-finite, or any
target observation/load comparison is accessed. Passing this gate would still
be a partitioned mechanical coupling result, not aerodynamic validation.

## 17. FluxV v5h4 frozen-field result

The preregistered one-step slice passes.  Disabled external transport is
bitwise equal to the existing rVPM self-plus-freestream LSRK3 step, and an
enabled zero external velocity/Jacobian is bitwise equal to the self-field
step.  The active nominal row advances 208 live v5h2 particles at Ptera step 2
from DVM source step 3 in exactly three LSRK3 stages.  It records three
parent-only center evaluations and 18 centered-difference evaluations, with
zero feedback, parent or surface-load writes.  Independent use of the recorded
RHS and low-storage arrays reproduces all three stage outputs bitwise.

The relative epsilon family `2^-8`, `2^-10`, `2^-12` gives adjacent-difference
ratios `15.203654906666083` for the independently sampled parent Jacobian and
`15.979367984336829`, `15.769888894626797`, and
`15.94211049620808` for final position, circulation vector, and smoothing
radius.  The values are close to the theoretical centered-difference factor of
16.  The formal-run test uses a conservative operational floor of 12; section
16 preregistered a common second-order limit but no numeric ratio floor.  The
full parent field-input, load and legacy-VPM state hashes identically before and after at
`004692b31c65fab32a52dd056b72041398aa1279fc2edafae5a8d0df222bd2ad`.
Two fresh three-epsilon executions have the same combined transported-state
SHA-256 `dfeeb8967c44577ea1d25f6a7b68f93bb32f78e197e503cf39e1a9ae0035279f`.

The frozen implementation/test hashes are
`d784fdab354c9bd55676eb1402bd0444fa1b9c5f6bd54544a4c9de7912d511d7`
and `746c7fc3cd798e77d85f52e671a46daff6c08621daea4e40756fbf1ff9cf0945`.
Fourteen focused and 319 related tests pass.  The formal auxiliary bundle is
`runs/20260815_fluxv_v5h4_frozen_ptera_rvpm_transport`; verdict:
`go_frozen_parent_transport_mechanics_only`.  This is not a synchronized time
march and does not authorize target scoring.

## 18. FluxV v5h5 synchronized partitioned time-march gate (preregistered)

The next vertical slice must replace, not stack, the previously separate
transport paths.  For a generic straight wing and one LEV family, each time
layer has exactly this order:

1. append the attested DVM release to the old cloud at `t_n` without moving the
   old prefix;
2. use that post-release/pre-transport cloud exactly once in the v5h3 native
   Ptera collocation and load-velocity channels;
3. freeze the solved Ptera bound/current-ring-wake/freestream field and advance
   the complete cloud once to `t_(n+1)` with the v5h4 combined LSRK3 field;
4. publish frontier facts from those same stage-pre fields for the next
   continuous DVM placement.

The orchestrator may reuse the frozen v5h2 release ledger and cloud containers,
but it must not call the v5h2 self-plus-freestream transport and then call v5h4
again.  One freestream, one self field, one frozen Ptera parent field and one
transport update are allowed per layer.  The first gate is three Ptera layers
with first/continuous/continuous DVM releases, exact additive particle slices,
bitwise immutable old prefixes before each transport, step/time continuity,
Ptera residual/load ownership, and full stage replay.  Hard-off reductions
must separately reproduce v5h3 and v5h4 one-way parents.

Immediate `STOP`: any report reset or reused live object, a cloud moved before
the Ptera solve, two transport calls in one layer, a freestream/self/parent
term counted twice, frontier facts from a different stage field, any Ptera or
particle owner mismatch, non-finite state, failed residual/replay, or any
target observation access.  Even a pass remains a non-target straight-wing
mechanical result; taper/twist refinement and paper scoring require later
gates.

### v5h5 synchronized-gate outcome (2026-08-15)

The generic three-layer vertical slice passes with verdict
`go_three_layer_synchronized_mechanics_only`.  The attested node-local DVM
placements are first/continuous/continuous; each layer deposits 102 new
particles, so the cumulative cloud is 102/204/306 without moving the old
prefix before the Ptera solve.  Each layer uses one native Ptera feedback
ledger, one combined three-stage transport, 3 parent center calls plus 18
finite-difference calls for the particle field, and separate 3+3 parent calls
for frontier transport and its read-only replay.  Old-prefix and frontier
replay residuals are exactly zero; the largest Ptera no-penetration residual
is `2.7755575615628914e-17`; parent/write counters are zero.

Two fresh complete executions have identical report and native-state hashes.
The feedback-only one-layer reduction is bitwise identical to the full-mode
Ptera state/load ledger, and the three-layer transport-only reduction is
bitwise identical to native `UVPMHybridSolver`.  Ten focused and 176 related
tests pass.  Frozen implementation/runner/test SHA-256 values are
`257f8fd3ff2cdf63a9637b5258e54d4b069a54d0c83128101bd2905fc685abfa`,
`0c3712a62dbdc75787411c96d80b057bbb58116f101d71ad33c4c37c053a0d76`,
and `c4298a23230b656f062f7c31ff1dc70fe362425c07b7b051ad3579ce58855665`.
Artifact: `runs/20260815_fluxv_v5h5_synchronized_smoke` (summary SHA-256
`81e54da53e0772a19f5b23dff79a66dc131c5c5d7a97a7e9d4d7576568765269`).
This remains a straight-wing, LEV-only mechanical result; taper/twist
refinement, general 3-D stability, and all target-paper scoring remain blocked.

## 19. FluxV v5h6 geometry/refinement gate (preregistered)

The next auxiliary gate keeps the frozen v5h5 ownership order and changes only
generic geometry or resolution.  It uses three non-target four-cell wings:
straight (`c=0.20 m`), linearly tapered (`0.20 -> 0.12 m`), and linearly
twisted (`0 -> 20 deg`, constant `c=0.20 m`).  Ptera cross-sections,
node-local DVM facts, cell-centre DVM sources, and GP1 plane maps must share
the same leading-edge nodes, chord law, and twist law.

The fixed physical horizon is `0.06 s`, speed is `2 m/s`, birth core is
`0.085 m`, and base deposition spacing is `0.04 m`.  Spatial refinement uses
dyadic levels `0,1,2` at `dt=0.02 s`; temporal refinement uses
`dt=(0.02,0.01,0.005) s` at dyadic level 0, with respectively 3, 6, and 12
release/Ptera/transport layers.  The compared observables are the final
node-frontier displacement from its physical leading-edge anchor and induced
velocity at five fixed GP1 probes.  Particle arrays of unequal size are never
compared directly.

The isolated DVM source incidence is frozen at a constant `60 deg`.  This was
selected before any coupled refinement result was observed, using only a
source-activity feasibility gate: the inherited `35 deg` v5h5 input becomes
inactive on the second layer at `c=0.20 m`, while the lowest member of the
standard check set `(45,60,75,90) deg` that remains active for all
`c=(0.12,0.16,0.20) m` and `dt=(0.02,0.01,0.005) s` is `60 deg`.  No
Ptera/rVPM result or target observation entered this choice, and the incidence
must not change after coupled execution begins.

For both observables and all three geometries, the preregistered convergence
gate is: finite/nondegenerate fields, last-level relative difference `<=0.02`,
and consecutive-difference ratio `>=1.25`.  Every configuration must also
retain v5h5 exact append/replay, Ptera residual `<=1e-12`, positive finite
sigma, zero extension/parent/load writes, deterministic report lineage, and no
target access.  Runtime budget is 30 minutes for the full matrix; resource cap
is 20,000 cumulative particles per configuration.

Immediate `STOP`: any mismatch between Ptera and DVM geometry, altered v5h5
owner/order, post-hoc threshold/core/spacing selection, missing refinement
row, non-finite state, particle-cap breach, failed mechanics, failed spatial
or temporal gate, or target/paper scoring access.  A pass would support only
generic geometry/refinement mechanics, not aerodynamic accuracy or general
three-dimensional stability.

### v5h6 executed outcome (2026-08-15)

The full matrix completed but the preregistered decision is `STOP`.  All 15
unique configurations pass the geometry, finite-state, ownership, exact
append, exact frontier replay, resource and Ptera residual gates; the largest
no-penetration residual is below `1.7e-16`.  Spatial frontier convergence
passes for straight, taper and twist, and the straight fixed-probe field also
passes.  Taper and twist probe fields fail with fine differences
`0.0578303` and `0.119875`; the taper ratio is `0.811703` and the twist fine
difference exceeds `0.02`.  All six temporal observable gates fail.  Their
fine relative differences range from `0.845478` to `1.136064`, far outside
the frozen limit.

The failure is deterministic rather than a solver crash.  A second fresh run
is byte-identical for README, source/result manifests, checksums and summary;
an independent disk recomputation reproduces every difference, ratio and gate
boolean.  The formal artifact is
`runs/20260815_fluxv_v5h6_geometry_refinement`; runner/test/summary SHA-256
values are `84dcf58673eb58f8fd34f183f9fdf7188a24e4005a346a4036940d5ef1a5a246`,
`48ee580bd3cde22cce7df97b6eb52f2cc16cc707646b02b7080b9cd65376e9df`,
and `55ac0c4e68df13559393861d48d70f37f11e620628af30fbb8758c2861f5fff8`.
No target observation or paper scoring path was used.

The temporal frontier metric also exposes a contract error: it describes the
newest release advanced for one step, so its physical age is `dt`, not the
fixed `0.06 s` horizon.  Its norm divided by `dt` is approximately stable even
while the unscaled norm tends to zero.  The fixed probes are valid same-time
observables and fail independently, so correcting this metric cannot reverse
the STOP.  Source-only ledgers show finite cumulative released circulation,
while the coupled probe field decreases with `dt`; this is consistent with an
independent-closed-ring temporal topology whose dipole area collapses as the
birth displacement shrinks.  It is a diagnosis to test, not yet a promoted
physical claim.

### Preregistered successor: v5h7 temporal-topology oracle

Do not add another target or refinement run.  First isolate a manufactured
straight ribbon with fixed physical horizon/core/span and compare the current
independent closed-ring release against a temporally connected sheet/edge
ledger.  The primary observables are fixed-age frontier tracers, physical
vorticity impulse, and fixed-probe velocity.  The gate must prove the temporal
shared edge is owned exactly once, no leading-edge/pseudovortex edge is
duplicated per release, the same physical-age object is compared across time
steps, and the field has a common finite limit.  It must also specify whether
old particle strengths are updated/remeshed; silently retaining v5h5 exact
append while double-owning a shared temporal edge is forbidden.  Any owner
ambiguity, field tending spuriously to zero, loss of circulation/impulse,
non-convergence or target access is immediate STOP.

The frozen manufactured contract uses `T=0.08 s`, `U=2 m/s`, span `0.60 m`,
constant shed increment rate `dGamma/dt=0.8 m2/s2`, `sigma=0.085 m`, particle
spacing `0.02 m`, and `dt=(0.02,0.01,0.005,0.0025) s`.  Three fixed GP1 probes
are `(0.05,0.30,0.30)`, `(0.10,0.15,0.40)`, and `(0.20,0.45,0.50) m`.
The independent family uses one topologically distinct small closed ring per
increment with strength `DeltaGamma=(dGamma/dt)dt`.  The connected family uses
shared temporal nodes and panel strength
`Gamma_panel[i]=(i+1)DeltaGamma`, so each internal shared span edge reconstructs
one `DeltaGamma` filament.

Required oracle gates are: exact ring/particle conservation; zero shared edges
for the independent family; exactly `N-1` two-incidence temporal edges for the
connected family, each with net `DeltaGamma`; independent impulse and probe
norms halve within `[0.49,0.51]`; connected impulse matches
`-(dGamma/dt) U span (T^2+T dt)/2` in GP1 z to `1e-14`; its error to the
analytic `dt->0` limit halves; both connected probe difference ratios lie in
`[1.8,2.2]`; two first-order Richardson probe extrapolants differ by at most
`0.01` relative and the extrapolated norm exceeds `1e-4`.  Total particles are
capped at 5,000.  These thresholds are frozen from topology/order analysis
and a non-target implementation feasibility probe, not target observations.

### v5h7 executed outcome (2026-08-15)

The oracle passes with verdict
`go_temporal_connected_topology_oracle_mechanics_only`.  Across four time
levels, independent-ring impulse ratios are `0.5` to roundoff and probe-norm
ratios are `0.499981--0.499993`, proving that this representation vanishes
linearly as `dt -> 0`.  The connected cumulative-strength sheet has exactly
`N-1` shared temporal edges, each reconstructing one `DeltaGamma`.  Its
analytic impulse error ratios are `2.0`, fixed-probe difference ratios are
`2.00525` and `2.00281`, and successive first-order Richardson extrapolants
differ by `0.0007501` relative at nonzero norm `0.0101992`.  All edge,
particle, incidence and conservation ledgers pass; the largest cloud has
1,984 particles.

Two fresh artifacts are byte-identical.  The project artifact is
`runs/20260815_fluxv_v5h7_temporal_topology_oracle`; runner/test/summary
SHA-256 values are `22a90cc17cd8e47d212ebcc2d7f71bdb8108cd3a0e61a8a3d2310bede9127ac9`,
`1b16ee6cae370c4d493c3af042350f9c283169e6b875c330325253bc63b4e023`,
and `5b3c925fa9065726c4e2c4cd24ff89acd50aea377bf00875bbb5d4d40210bd0b`.
Eighteen focused and 84 bridge/reference joint tests pass.  This confirms the
manufactured topology mechanism only; it does not authorize replacing the
production cloud or scoring target cases.

### Preregistered successor: v5h8 causal incremental-sheet oracle

The next gate must show how a time-connected sheet is built causally, without
silently replacing v5h5 exact append.  At release `n`, retain the transported
old downstream boundary contribution `-Gamma_panel[n-1]` and add an upstream
contribution `+Gamma_panel[n]` on exactly the same live boundary
particle/frontier facts, giving net `DeltaGamma[n]`; add the new downstream
and tip edges once.  Freeze one-to-four manufactured releases and compare the
incremental particle field, impulse and edge ledger against a direct full
connected-graph rebuild.  First test zero transport, then uniform/affine
transport with boundary sigma/gamma changes.  Exact old-prefix immutability,
live-boundary identity, causal cumulative strength, field/impulse parity,
finite positive sigma, rollback and no target/Ptera/load access are mandatory.
If the current boundary cannot be reused exactly after rVPM stretching, the
result is STOP and the production design must adopt an explicit conservative
remesh/update owner rather than claim exact append.

The v5h8 manufactured values are frozen as `span=.60 m`, streamwise panel
step `.04 m`, `DeltaGamma=.016 m2/s`, `sigma_birth=.085 m`, spacing `.02 m`,
and one-to-four releases at the three v5h7 probes.  The affine live-material
step is
`A=[[1,.015,0],[0,1,.01],[.02,0,.99]]`,
`b=(.04,-.002,.003) m`, with `gamma<-A gamma` and
`sigma<-1.01 sigma`.  Before each append, all old arrays and panel endpoints
receive this same map.  The new upstream incidence must clone the previous
downstream boundary's particle positions, sigma and weights exactly and scale
its gamma by `-Gamma_current/Gamma_previous`; geometry redeposition is a
required negative control, not an allowed exactness path.

Zero-transport incremental versus direct full-graph field/impulse residuals
must be `<=2e-14`.  Under affine transport, the incremental cloud must equal
an algebraically consolidated live-boundary comparator to `<=2e-14`, retain
the transformed old prefix bitwise during append, and close the clone
position/sigma/gamma relation to `<=1e-14`.  A fresh fixed-core geometry
redeposition must remain measurably different (`>1e-8` field relative after
two or more releases), proving that it is not being mistaken for the live
material comparator.  Particle cap is 1,000.  Non-affine midpoint/tangent
incompatibility and long-time cancellation conditioning remain explicit later
STOP gates even if this bounded affine oracle passes.

### v5h8 executed outcome (2026-08-15)

The frozen oracle passes with verdict
`go_v5h8_bounded_affine_live_basis_mechanics_only`.  For one-to-four releases,
the zero-transport incremental cloud agrees with its direct connected-graph
comparator with maximum velocity-field, Jacobian and impulse residuals of
`6.938893903907228e-18`, `1.249000902703301e-16`, and `0`, respectively.
Under the bounded affine family, the exact live-material-basis clone agrees
with the algebraically collapsed comparator to `6.938893903907228e-18`,
`2.7755575615628914e-17`, and `4.336808689942018e-19`.  Old prefixes remain
bitwise unchanged during append, clone position/sigma/gamma relations close,
and rollback, finite-positive-sigma, particle-cap and no-target/Ptera/load
gates pass.

The fresh-geometry redeposition negative control differs from the live-basis
field by `0.0011467742630084674`, `0.0014549564075497143`, and
`0.0013029774819262976` relative at releases two through four.  Therefore an
ordinary fresh append is **NO-GO** after material deformation; only the
preregistered inherited live-basis clone is **GO** for this bounded affine
mechanics claim.  This is not evidence for a generic non-affine or production
rVPM sheet.

The formal artifact is
`runs/20260815_fluxv_v5h8_incremental_sheet_oracle`, whose `summary.json`
SHA-256 is `af38f29d8da8298d9ca9444e6210791bac2a4a65ca3920578e8ba9c9066dfb91`.
The sheet module/module-test/runner/runner-test SHA-256 values are
`f30b5fbf6d2f1718bbbecec669a47dfb1c3c001942d2212ce814098966a610e2`,
`db0ee997688e3f65802ab4345721f07337f1c9f2622548d600f9843c8c2ca710`,
`f2079cb457b883fd35892434745d8a3e96c6a6587ba23f1caf710193723f6e98`,
and `1f402bca2944c1f4e87e0d0f975d3713cb0d092b2c94e3bb876338f24b0be953`.
Thirty-eight module tests, nine runner tests and 131 joint tests pass.  A fresh
audit passes with one `cap-after-allocation` warning; pre-allocation cap
enforcement remains required before production use.

Production promotion remains blocked.  The next owner-level task is an
explicit conservative boundary update/remesh owner for general transported
sheets, including non-affine support compatibility, long-time cancellation,
pre-allocation resource rejection and transactional rollback.  No target
scoring or production coupling is authorized before that owner closes.
