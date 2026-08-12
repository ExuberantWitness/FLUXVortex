# FluxV v4 LDVM/LEV integration and Stevens validation plan

## 1. Objective

- run id: `20260812_fluxv_v4_ldvm_stevens`
- selected idea: retain the finite-wing UVLM circulation, wake and non-circulatory
  load ledger, but replace the post-hoc periodic load owner with a causal,
  source-traceable LEV shedding/suction-loss state derived from Ramesh's LDVM.
  The mechanism must use local kinematics/Reynolds/geometry only and must not
  branch on paper or case identity.
- user's core requirements:
  1. improve Figure-14 accuracy without sacrificing Yang 2025 or
     Izraelevitz Figure 11;
  2. study the LDVM v2.5 source before integrating any mechanism;
  3. determine whether the Ramesh and Martínez-Carmena theses include genuine
     3-D computation cases;
  4. reproduce Stevens & Babinsky 2017 and evaluate improved-FluxV lift.
- non-negotiable constraints: preserve UVLM; no CFD/CSD replacement; no
  case-id residual table; no use of Stevens force observations for parameter
  selection; retain frozen v1/v2/v3 outputs as read-only baselines.
- research question: can one causal LEV/shedding mechanism improve the joint
  Yang/Figure-11/Figure-14 envelope and transfer to Stevens pitch-up lift?
- null hypothesis: a source-derived LDVM/LEV state does not improve the joint
  error envelope over the best existing v2/v3 variants.
- alternative hypothesis: the source-derived state reduces separated-flow
  load error while retaining attached-flow ULLT/UVLM accuracy and improves
  Stevens lift without observation fitting.

## 2. Baseline And Comparability

- baseline id: `20260812_periodic_v2_ullt_full` and
  `20260812_periodic_v3_persistent_full`.
- baseline variant: use the best existing *single recorded* variant per table;
  do not splice per-case outputs into a claimed model.
- dataset / split:
  - development/audit: Yang 2025 six wind-tunnel means; Izraelevitz Figure 11
    numerical traces; Scherer Figure 14 14 experimental observations;
  - development transfer: Stevens 2017 LE-axis and mid-chord-axis experimental
    `CL(s/c)` traces and LEV diagnostics. Stevens was initially intended as
    held-out, but was reclassified after the first architecture smoke result.
- primary metrics:
  - Yang lift/drag MAE [gf];
  - Figure 11 lift/drag raw-phase RMSE;
  - Figure 14 mean-CT RMSE;
  - Stevens lift trace RMSE, range-NRMSE, correlation, peak magnitude and phase.
- required metric keys: all four channels above; no pooled metric may hide a
  regressing task.
- minimum gate: v4 must beat old FluxV on every available channel.
- solid gate: Yang `L/D MAE <= 3.6/2.5 gf`; Figure 11 `L/D RMSE <= 0.17/0.27`;
  Figure 14 `CT RMSE <= 0.0465`; Stevens lift RMSE improves by at least 10%
  over a frozen old-FluxV baseline on both axes, or the transfer claim fails.
- comparability risks: Figure 11 is numerical rather than experimental;
  Figure 14 has mean thrust only; Stevens publishes lift but not drag; LDVM is
  two-dimensional; historical v2/v3 share evolving source files; none of the
  curves constitutes an independent sample ensemble for significance testing.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/ldvm_fourier.py` | existing partial LDVM | expose component-resolved force ledgers and retain source-time semantics | establish auditable paired discrepancy | partial source parity |
| `platform/forward_flight_benchmarks/ldvm_uvlm_correction.py` | absent | paired separated-minus-attached stripwise LDVM discrepancy | retain UVLM while adding separation response | no conservative common wake |
| `platform/forward_flight_benchmarks/causal_incidence_owner.py` | absent | causal signed/absolute incidence persistence state | replace post-hoc periodic owner | heuristic ownership law |
| `platform/forward_flight_benchmarks/stevens2017.py` | absent | geometry, Wang-Eldredge motion, digitized observations and old/v4 adapters | development-transfer experiment | missing drag and exact LE radius |
| `platform/forward_flight_benchmarks/run_v4_crosspaper.py` | absent | common four-benchmark runner and manifests | one fair evaluation surface | runtime/provenance |
| `platform/tests/test_ramesh_ldvm_reference.py` | absent | reference landmarks, Kelvin, sign and LESP tests | source fidelity | partial rather than full parity |
| `platform/tests/test_ldvm_uvlm_correction.py` | absent | no-onset exact reduction and component ledger tests | protect UVLM | does not prove conservative coupling |
| `platform/tests/test_stevens2017.py` | absent | geometry/motion/digitization tests | reproducibility | vector digitization uncertainty |

## 4. Execution Design

- minimal experiment: reproduce one author LDVM reference case and demonstrate
  module-off exact reduction to old FluxV on a smoke movement.
- smoke plan: coarse Yang 15/25 deg, Figure 11, two Figure-14 conditions, and
  one Stevens axis; verify finite states, Kelvin/sign/load ledgers and outputs.
- full run plan: six Yang angles, full Figure 11 phase, all 14 Figure-14
  observations, both Stevens axes; then one-factor time/grid checks.
- expected outputs: source/thesis audit, digitized Stevens CSV, phase/mean CSVs,
  metrics, manifests, ablation table, lift/drag figures and a Chinese report.
- stop condition: all solid gates pass or a documented mechanism-level failure
  shows the source-derived LDVM state cannot satisfy the joint envelope.
- abandonment condition: source reference case cannot be reproduced; coupling
  requires case-specific parameters; module-off does not exactly recover UVLM;
  or any task degrades beyond its minimum gate after two discriminative fixes.
- strongest alternative hypothesis: current errors arise mainly from geometry/
  kinematics/profile drag rather than LEV shedding; then LDVM integration will
  not jointly improve the tasks.

## 5. Runtime Strategy

- smoke command: to be frozen after the LDVM source interface and Stevens data
  parser are implemented.
- main command: `python -m forward_flight_benchmarks.run_v4_crosspaper
  --steps-per-cycle 256 --yang-strips 12 --output
  docs/forward_flight_large_pitch/reproductions/
  unified_fluxv_v4_ldvm_stevens_20260812/runs/
  20260812_fluxv_v4b_crosspaper_full` (shown wrapped for readability).
- expected runtime / budget: CPU-only; smoke under 5 minutes, full matrix under
  60 minutes using frozen old histories when mathematically valid.
- log / artifact locations: this directory under `runs/` with distinct
  `smoke` and `full` ids.
- safe efficiency levers: reuse immutable kinematics/reference histories;
  vectorize strip states; never reduce fidelity of the primary full run.
- existing tooling: Ptera movement builders, ULLT/UVLM ledgers and existing
  metric/plot helpers remain shared; no duplicated metric implementation.

Monitoring: check after 60/120/300/600 s; kill on NaN/Inf, force-ledger failure,
wrong condition count, or output provenance mismatch.

## 6. Fallbacks And Recovery

- if LDVM Fortran cannot compile: validate equations against bundled reference
  outputs and implement a minimal Python port with byte-frozen test fixtures.
- if Stevens ramp formula remains ambiguous: implement both source-consistent
  interpretations as a preregistered kinematic sensitivity, not force fitting.
- if coupling is wrong after smoke: return to standalone LDVM and explicit
  UVLM load decomposition before another full run.
- if a full run becomes non-comparable: preserve it as failed evidence and
  relaunch under a new run id; never overwrite a frozen baseline.

## 7. Checklist Link

- checklist path: `CHECKLIST.md`
- next unchecked item: finish LDVM source/thesis audit and freeze the v4
  executable mechanism contract.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-12 | opened isolated v4 line from published v3 branch | user clarified that a safe pass-through is insufficient | v1/v2/v3 become immutable baselines; Stevens becomes held-out confirmation |
| 2026-08-12 | completed LDVM v2.5 source audit and compiler parity | the author mechanism and its defects had to be known before transfer | use LESP/Kelvin/material-LEV form; do not copy GPL Fortran or its array/truncation bugs |
| 2026-08-12 | rejected standalone 2-D LDVM replacement | Stevens smoke improved the LE axis but degraded the mid-chord axis | retain UVLM and add only separated-minus-attached LDVM increments |
| 2026-08-12 | split finite-wing normal and suction scaling | Ramesh axial suction is quadratic in LESP/circulation | use `g` for delta-CN and `g^2` for delta-CS instead of one common Prandtl factor |
| 2026-08-12 | replaced v3 two-pass persistence by a causal two-pole state | v3 used future cycle statistics and could not transfer to a pitch-up transient | use Izraelevitz circulation-state poles -0.30 and -0.045; preserve zero-future prefix invariance |
| 2026-08-12 | Stevens reclassified from untouched held-out to development transfer | the additive architecture was inspected after the first old/standalone-LDVM smoke result | report it honestly as a new experimental transfer test, not independent confirmation |
| 2026-08-12 | replaced cycle-mean ownership with phase-resolved blending, split normal-force ledgers, fixed Yang force projection and fixed `r_c/c=0.02` | independent v4a review found covariance, scaling, projection and coupled-discretization errors | all formal v4b results use the corrected source hashes; earlier v4/v4a outputs remain failed evidence |
