# Izraelevitz Figure 14 / Scherer experiment integrity audit

Status: **WARN, no numerical-integrity FAIL; v3 post-hoc repair passed this known gate**  
Evaluation type: `real_gt_digitized_experimental_cycle_mean`

## Passed checks

- `mean_thrust_vs_phase.csv` contains 14 experimental observations at 12
  unique conditions.  The duplicate observations at `(15 deg, 15 deg)` and
  `(15 deg, 75 deg)` are retained separately; aggregate metrics therefore
  weight 14 observations rather than 12 condition means.
- Every row of `accuracy_metrics.csv` was independently recomputed from the
  prediction/observation CSV.  The maximum discrepancy was below `2e-15`.
- The source-specified profile drag is added exactly once to each local
  inviscid history.  The main `Cd0=0.057` values exactly equal the matching
  rows in `profile_drag_sensitivity.csv`.
- All Figure-14 result hashes match their frozen artifacts, and the v3
  manifest's current source/base/result hashes all match.  The older frozen
  source hashes are retained as historical execution records; later shared
  source changes mean they are not a complete current-tree source snapshot.
- The conclusion that current v1/v2 fails the experimental gate is unchanged
  for `Cd0=0`, `0.027`, and `0.057`.
- The isolated v3 full run contains 12 unique predicted conditions and is
  scored against all 14 observations, retaining both duplicate experimental
  markers.  Its reported metrics were independently recomputed from
  `izraelevitz2017_fig14_v3_mean_thrust.csv` with the same 14-observation
  weighting.
- v3 computes the Figure-14 source `Cd0=0.057` load from a separate kinematic
  ledger at the paper's actual `3/4c` pitch-axis velocity and adds it once.
  The nonlinear-polar proxy and constant-profile-drag ledger are constructed
  separately even though they share the same geometry.

## Key metrics (14 observations, `Cd0=0.057`)

| Model | MAE CT | RMSE CT | Bias CT |
|---|---:|---:|---:|
| Local one-state ULLT | 0.033879 | 0.046361 | 0.015025 |
| Old FluxV | 0.034841 | 0.051147 | 0.026983 |
| Authors' one-state ULLT | 0.045836 | 0.063212 | 0.045392 |
| Authors' six-state ULLT | 0.050136 | 0.067037 | 0.050136 |
| Authors' QS + added mass | 0.097814 | 0.112079 | 0.097814 |
| FluxV v1 = v2 cycle mean | 0.183314 | 0.222598 | -0.167865 |

## v3 full repair metrics (14 observations, `Cd0=0.057`)

| Model | MAE CT | RMSE CT | Bias CT |
|---|---:|---:|---:|
| **FluxV v3 persistent owner** | **0.033951** | **0.047185** | **0.016340** |
| v3 mean pass-through ablation | 0.034946 | 0.052044 | 0.028298 |
| Frozen old FluxV comparator | 0.034841 | 0.051147 | 0.026983 |

The main v3 is better than the frozen old comparator by `0.003961` RMSE CT,
whereas the pass-through ablation is slightly worse.  This supports the load-
ownership diagnosis on this already-seen gate: at every Scherer condition the
periodic persistent-incidence fraction is exactly zero at the registered
numerical floor, so v3 assigns both the cycle mean and alternating load to the
one-state ULLT owner.  It does not retain a recentered static-polar drag
history.

The development smoke-to-full comparison changes mesh, time step, and cycle
count together and is therefore not a formal convergence study.  Its maximum
paired Figure-14 change is `0.00555 CT`, slightly above the predeclared
`0.005 CT` development target.  The full result remains the only reported
primary result, but numerical convergence is not claimed.

## Warnings and claim limits

- Old FluxV is the baseline UVLM channel extracted from the same augmented
  run, not an independently executed second solver.  This is appropriate for
  exact reduction but is not independent-model agreement.
- v2 inherits the v1 cycle mean, so v1/v2 are exactly identical for the
  published Figure-14 observable.  The experiment cannot validate v2's phase
  mechanism because it contains no experimental instantaneous load history.
- The source figure does not define the statistical meaning of its error
  bars.  The reported `RMS_uncertainty_units` is only a scale-normalized
  residual and must not be called a z-score, standard-deviation unit, or
  significance test.
- The listed manifest hashes are internally consistent but are not a complete
  transitive environment snapshot; the Ptera installation and original PDF
  are described by provenance rather than vendored into the repository.
- The small numerical advantage of local ULLT over old FluxV is not evidence
  of statistical significance.
- v3 was designed after inspecting the v1/v2 Figure-14 failure.  Its pass is
  therefore post-hoc exploratory evidence, not a held-out, preregistered, or
  independent confirmation.
- v3 and the frozen old/v1/v2 metrics do not use the same profile-drag
  velocity reference: v3 fixes the source geometry to `3/4c`; frozen models
  used the earlier quarter-chord proxy.  They remain in the audit table as
  historical failure controls, but the final main plot instead uses the
  same-ledger `3/4c` pass-through curve as its local ablation.
- Figure 14 observes cycle-mean thrust only.  Because v3 selects ULLT at
  `p=0`, this gate cannot validate v3's instantaneous AC blend, LEV suction,
  vortex shedding, or dynamic-stall physics.

Supported claims: (1) the frozen exploratory v1/v2 that improved Yang 2025 and
the Izraelevitz Figure-11 numerical task does **not** generalize to the Figure-
14 Scherer experimental mean-thrust benchmark; and (2) the post-hoc v3
persistent-owner repair removes that known failure and slightly outperforms
the frozen old FluxV comparator on mean thrust.  No independent-generalization
or LEV-physics claim is supported.
