# FluxV v5a → v5b Experiment Checklist

## Identity

- run ids: `20260814_fluxv_v5a_threepaper`, `20260814_fluxv_v5b_threepaper`
- idea: FluxV v5a exclusive equilibrium/transient ledger, followed by conditional v5b shared wake
- stage: v5a stopped by frozen smoke gate; v5b force-promotion NO-GO

## Planning

- [x] idea and Problem Anchor frozen
- [x] baseline and metric contract confirmed
- [x] code touchpoints listed
- [x] v5a smoke/full plan written
- [x] v5b sequential stop/go rule written
- [x] fallbacks recorded

## v5a Implementation

- [x] core module implemented
- [x] qS/unit convention unique
- [x] equilibrium path bypasses LDVM projection
- [x] transient path uses frozen component projection
- [x] exact-limit and ledger tests pass
- [x] unrelated baseline files unchanged

## v5a Pilot / Smoke

- [x] representative command executed
- [x] all outputs finite and phase aligned
- [x] no target fitting or parameter selection
- [x] two-or-more-paper direction gate checked
- [x] implementation/proposal divergence recorded
- [x] v5a stopped after Figure 14 and Baik failed the frozen direction gate

## v5a Main Run

- [ ] dedicated `run/*` branch/worktree frozen
- [ ] full command and environment recorded
- [ ] all three paper matrices completed
- [ ] equilibrium/transient/full ablations completed
- [ ] one-factor sensitivities completed
- [x] v5a verdict recorded: cache-adapter hypothesis rejected; full intentionally not run

## v5b Implementation

- [x] v5a frozen before v5b edits
- [x] shared-wake equations/topology/time layer documented
- [x] LDVM force discrepancy, polar residual, impulse and second force owners disabled
- [x] internal N1 no-LEV pressure baseline reduces exactly
- [ ] standalone v5b reduces to **current FluxV** with LEV disabled — FAIL, max phase error `0.556435`
- [x] topology/Kelvin/LESP/material-circulation gates pass
- [x] one-strip single-pressure-ledger and smooth-birth development gates pass
- [ ] high-AR Ramesh force parity — NOT RUN after exact-reduction stop
- [ ] representative paper force smoke eligible — BLOCKED by G1

## v5b Main Run

- [x] dedicated force-gate result regenerated after source freeze
- [ ] all three matrices completed
- [ ] paper-level numerical sensitivities completed
- [ ] comparison against old/v4b/v5a complete
- [x] cross-paper scoring deliberately stopped before invalid promotion

## Validation and Closeout

- [x] v5a and v5b gate metrics independently recomputed
- [x] declared source/result/figure hashes verified
- [x] main claims classified supported/refuted/inconclusive
- [x] PNG/PDF figures rendered and inspected
- [x] final combined report written
- [x] exact next action stated
- [x] fresh same-family experiment audit incorporated (`WARN / provisional`)

## v5h DVM-to-rVPM Auxiliary Gate

### Contract and preflight

- [x] Ptera retained as FluxV UVLM and unique surface-load owner
- [x] DVM restricted to source/event/circulation/placement facts
- [x] rVPM restricted to post-birth three-dimensional transport
- [x] no target-paper GT/force/load access frozen
- [x] fixed physical smoothing radius separated from deposition spacing
- [x] run id, outputs, minimal smoke, and STOP conditions written in `PLAN.md`

### Source and topology implementation

- [x] source-parity DVM event mask recovers 174 LEVs and 499 TEVs
- [x] DVM source event/Kelvin/history-chain attestation passes
- [x] placement dataclasses resist extra-attribute and method-shadow injection
- [x] placement schema and tests frozen by SHA-256 (`fe973845...` / `ef7d89f2...`)
- [x] node-local DVM placement to GP1 adapter implemented and audited (`2829e8c...`)
- [x] first/restart half-step placement identity passes for both signs
- [x] continuous DVM absolute point remains audit-only
- [x] shared-node identity, wing/patch boundary, exact-once, and mixed-activity gates pass

### Fixed-core experiment

- [x] prescribed-sigma/independent-spacing edge deposition API passes audit
- [x] componentwise stable conservation sum passes high-count q48/q96/q128 cases
- [x] passive-frontier producer updated to stable-sum and prescribed-spacing contracts
- [x] passive/source/ribbon/frontier full regression frozen (`190 passed`; producer `5229b165...`)
- [x] straight first-release smoke passes
- [x] straight/taper/twist two-release fixed-core gate completes
- [x] quadrature, time, topology, conservation, and deterministic-replay gates pass
- [x] strict JSON artifact, raw arrays, independent recomputation, logs, environment, metrics, and hashes written
- [x] cross-mapper ribbon global exact-once and live placement/patch binding pass fresh attack audit
- [x] fixed-core v1 formal artifact closes at semantic SHA `c18f0f07...`

### Promotion boundary

- [x] cumulative-cloud v2 eligible after the fixed-core gate passes
- [ ] Yang/Izraelevitz/Baik scoring eligible only after cumulative transport and Ptera feedback gates pass
- [x] FluxV4b remains the current evidence-backed paper baseline

### Cumulative-cloud v2

- [x] v2 single-release physical state is bitwise equal to v1
- [x] copied/reordered/1-ULP/cross-wing handoff attacks fail transactionally
- [x] straight three-release and fourth-release smoke close the current blocker
- [x] active/inactive/restart/continuous lifecycle passes
- [x] cumulative slices are additive and old prefixes remain immutable
- [x] old and newborn particles use one combined LSRK3 stage field
- [x] fixed `sigma_birth` remains independent of h across every release
- [x] 36/36 configuration-mechanics and 108/108 release rows pass
- [x] every straight/taper/twist cumulative time-refinement family passes
- [ ] spatial fixed-probe gate passes all geometries — FAIL: straight
  `1.4507852205964775`, taper `1.4964923945744306`, both `<1.5`; twist passes
- [x] two fresh semantic artifacts agree and the integrity audit passes with
  the documented eager-import/runtime-closure warning

Formal artifact:
`runs/20260815_fluxv_v5h_cumulative_cloud_gate`; semantic SHA-256:
`cc8666ad43ea291aae03a2b1b7e549e8dd6711b2e1c3006a07e293d5437a4759`.
Runner/test/core SHAs are `155e1ea8...`, `86b0ae5b...`, and `8b4c3efb...`.
Verdict: C1 **yes**, C2 **no**, strict `STOP`; zero target-observation reads
and zero guarded Ptera/target solver, load or builder calls.  Ptera feedback
and Yang/Izraelevitz/Baik scoring remain blocked; FluxV4b remains the current
evidence-backed paper baseline.

### v5h2 dyadic-panel next candidate

- [x] candidate preregistered as a new API; frozen v2 evidence remains unchanged
- [x] level 0 is bitwise equal to frozen v2 at `h=0.04 m`
- [x] each edge uses `n0=max(1,ceil(L/0.04 m))` and exactly doubles panels at
  levels 1--3
- [x] `sigma_birth=0.085 m`, three geometries, three time resolutions, three
  releases and the `1.5` gate remain unchanged
- [x] 36/36 mechanics, 108/108 release rows and all time/space gates pass
- [x] two fresh semantic artifacts agree with distinct provenance
- [x] no target observation or surface-load/scoring path is used by the v5h2
  numerical chain

v5h2 verdict: `go_v5h2_mechanics_only`.  Straight/taper/twist spatial
frontier and fixed-probe families reduce by approximately four per dyadic
level; all time families pass.  Independent A/B semantic SHA-256:
`dece97d1d708056659bcaeb5cd82dd189ba834b0f97665b7de7271f7413245e4`.
Formal bundle: `runs/20260815_fluxv_v5h2_dyadic_cumulative_cloud_gate`.
This closes the deposition-refinement confound only; Ptera feedback and target
paper scoring remain blocked pending a separately preregistered coupling gate.

### v5h3 native Ptera feedback gate

- [x] Ptera/FluxV ownership and call order audited
- [x] one-way first-stage feedback boundary and STOP conditions preregistered
- [x] hard-off factory returns the exact native `UVPMHybridSolver`
- [x] enabled-empty path is stepwise bitwise equal to native FluxV
- [x] report step/time/wing/family and live-object identity gates pass
- [x] rVPM collocation normal enters the native AIC RHS exactly once
- [x] four native load-leg velocity batches receive the same cloud exactly once
- [x] Ptera remains the only panel/airplane force and moment owner
- [x] prescribed TE ring-wake and legacy VPM algorithms have no direct
  extension write; active changes reduce to the feedback-modified parent bound
  state
- [x] non-target steps 0--2 smoke, signs, poison, and direct-field replay pass
- [ ] taper/twist/refinement gate preregistered only after the vertical slice
- [ ] Yang/Izraelevitz/Baik scoring remains blocked until this gate and a later
  bound/wake-to-rVPM transport gate both pass

Vertical-slice result: `go_one_way_native_feedback_mechanics_only`. The formal
auxiliary bundle is
`runs/20260815_fluxv_v5h3_native_feedback_vertical_slice`; 11 focused and 117
related tests pass, and the maximum active native no-penetration residual is
`2.7755575615628914e-17`. This does not close two-way transport feedback.

### v5h4 frozen-Ptera-to-rVPM transport gate

- [x] reverse-direction ownership and time layer preregistered
- [x] parent-only velocity must bypass v5h3 feedback to prevent self double count
- [x] finite-difference Jacobian epsilon family frozen before execution
- [x] disabled and zero-external-field bitwise reductions pass
- [x] active LSRK3 stage velocity/Jacobian call ledger and independent replay pass
- [x] Ptera bound/wake/load/legacy-VPM state is byte-identical before/after transport
- [x] epsilon family has a common second-order Jacobian/transport limit
- [x] straight one-step slice passes before cumulative integration
- [x] target-paper scoring remains blocked

Formal v5h4 bundle:
`runs/20260815_fluxv_v5h4_frozen_ptera_rvpm_transport`.  Verdict:
`go_frozen_parent_transport_mechanics_only`; 14 focused and 319 related tests
pass.  This closes only a frozen one-step reverse-velocity slice.

### v5h5 synchronized partitioned time march

- [x] exact release/Ptera/transport/frontier order preregistered
- [x] post-release cloud enters native Ptera exactly once before transport
- [x] complete cloud receives one combined self+parent LSRK3 update per layer
- [x] v5h2 self+freestream transport is not stacked with v5h4
- [x] first/continuous/continuous three-layer lineage and frontier facts pass
- [x] hard-off reductions reproduce the two frozen one-way parents
- [x] generic straight non-target slice passes before taper/twist refinement
- [x] Yang/Izraelevitz/Baik scoring remains blocked

Formal v5h5 bundle:
`runs/20260815_fluxv_v5h5_synchronized_smoke`. Verdict:
`go_three_layer_synchronized_mechanics_only`; 10 focused and 176 related tests
pass. This closes the straight/LEV-only mechanical gate, not taper/twist
refinement, general 3-D stability, or paper performance.

### v5h6 geometry/refinement gate

- [x] straight/taper/twist geometry laws and common GP1 ownership preregistered
- [x] fixed horizon/core/base-spacing and resolution matrix preregistered
- [x] convergence observables and thresholds frozen before execution
- [x] constant `60 deg` source incidence frozen by source-only activity gate
- [x] bounded straight baseline smoke passes
- [x] 9 spatial rows (3 geometries x 3 dyadic levels) complete
- [x] 9 temporal rows (3 geometries x 3 time steps) complete
- [ ] **FAIL:** all geometry, mechanics, finite, resource, and convergence gates pass
- [x] two fresh semantic executions agree
- [x] Yang/Izraelevitz/Baik data and scoring remain unaccessed
- [x] deterministic `STOP` artifact and independent disk recomputation close
- [ ] v5h7 fixed-age/temporal-connected-sheet oracle is implemented

### v5h7 temporal-topology oracle

- [x] manufactured geometry, strengths, probes, matrix and analytic impulse frozen
- [x] independent and connected topology claims separated
- [x] convergence/halving thresholds and 5,000-particle cap frozen
- [x] shared-edge incidence and conservation tests pass
- [x] independent closed-ring field is confirmed to vanish linearly
- [x] connected cumulative-strength sheet has a finite first-order limit
- [x] deterministic no-target artifact closes

### v5h8 causal incremental-sheet oracle

- [x] direct connected-graph comparator and causal boundary-update claim frozen
- [x] zero-transport incremental append matches the direct full graph
- [x] bounded affine live-basis identity and strength update close
- [x] exact old-prefix, field/Jacobian/impulse parity and rollback gates pass
- [x] ordinary fresh-geometry append is rejected as the required negative control
- [x] formal artifact, 38 module tests, 9 runner tests and 131 joint tests close
- [x] fresh audit passes with one `cap-after-allocation` warning
- [x] conservative boundary update/remesh is selected as the next owner task
- [ ] cap rejection occurs before allocation and the general owner is audited
- [x] production promotion and target scoring remain blocked
