# Q16 Real FSI Long-Horizon Plan

## 1. Objective

- run id: `20260822_q16_real_fsi_long_horizon`
- selected idea: establish the high-load four-step boundary, then advance a
  declared damped Q16 fixture to eight coordinates on one live
  structural/aerodynamic owner with a CUDA discrete work-balance ledger.
- user's core requirement: continue toward long-time or multi-cycle FSI.
- non-negotiable constraints: Q16 only; CUDA float64 scientific numerics;
  separated LEV always enabled; joint TEV and free wake always active; no Q9,
  toy/reduced aerodynamics, CPU fallback, GT, scorer, paper or formal matrix.
- research question: can the real coupled owner cross at least one chord
  convective time without history loss, non-finite state, coupling failure or
  discrete work-balance drift?
- null hypothesis: later steps expose finite-horizon assumptions, lose
  LEV/free-wake continuity, fail coupling, or accumulate unexplained work.
- alternative hypothesis: eight consecutive damped-fixture steps converge,
  advance each history once and satisfy the augmented damping/work ledger.

This static-inflow case has no imposed oscillation frequency. Eight steps are
therefore an L1 long-horizon integration gate, not a multi-cycle claim.

## 2. Baseline And Comparability

- baseline id: `20260822_q16_real_fsi_short_trajectory` supplies the control
  configuration only. Its numerical outputs are invalidated for comparison by
  the Newmark-predictor alias defect found in this run and must be regenerated.
- baseline variant: pitched one-element Q16, 2x3 Ptera panels, `dt=0.04`,
  `E=1e8`, coupling tolerance `2e-7`, relaxation `0.7`, mandatory separated
  flow and material-Jacobi GPU-PCG.
- dataset / split: deterministic synthetic integration fixture; no paper data.
- primary metric: eight accepted steps on one owner.
- required metric keys: completed steps, physical/convective duration,
  structural/aero generations, state/velocity/acceleration norms, load norm,
  coupling/aero/Newton/PCG counts and residuals, LEV count, wake convection,
  kinetic-energy change, trapezoidal internal/external work, normalized balance
  closure, prefix/resume chain hashes and failed-next-coordinate integrity.
- comparability risks: Ptera clone digests contain instance identity; compare
  exact hashes only within one lineage and compare semantic metrics across
  clones. Longer wake/LEV history legitimately increases cost and load.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why this is needed | Risk |
|---|---|---|---|---|
| `src/fluxvortex/warp_fsi/q16_structural_solver.py` | GPU Newmark/Newton/PCG | CUDA endpoint work-balance audit | detect cross-step transfer/integration drift | naming an endpoint identity as exact stored energy |
| `platform/warp_vpm/q16_real_fsi_coupling.py` | one-step atomic coupling | attach work evidence before publication | keep audit in the same transaction | post-solve audit could become a failure point |
| `platform/warp_vpm/q16_real_fsi_trajectory.py` | bounded sequential owner | retain work/state metrics and resume an existing chain | enable segmented long runs | duplicated or discontinuous prefix |
| `platform/warp_vpm/test_q16_real_fsi_long_horizon.py` | absent | 4-step pilot, 8-step/resume and horizon-failure gates | exercise real accumulated history | runtime and later-step convergence |

## 4. Execution Design

- minimal experiment: four accepted steps with an extended five-coordinate
  Ptera problem and work ledger enabled.
- smoke / pilot plan: run 4 steps with GPU Aitken relaxation and an explicitly
  revised outer-iteration budget of 64, verify finite metrics, exact generation and
  history increments, coupling residual `<=2e-7`, normalized work closure
  `<=1e-6`, and particle capacity headroom.
- full run plan: use a nine-coordinate Ptera problem with `E=1e9 Pa`,
  `alpha_M=20 s^-1` and structural tolerance `5e-8`; advance 4 steps, resume the
  same hash chain for 4 more, then request the exhausted ninth continuation and
  prove the completed eight-step parent is unchanged.  This is distinct from
  the undamped `E=1e8 Pa` high-load stress fixture.
- expected outputs: production audit/resume code, CUDA tests, exact per-step
  metrics, logs, claim map, manifests and next-route decision.
- stop condition: non-finite CUDA state, mandatory-mode loss, work closure
  above `1e-6`, particle capacity exhaustion, non-monotone solver coordinate,
  coupling failure at the 64-iteration diagnostic budget, or failed-step mutation.
- abandonment condition: reaching eight steps requires disabling separated
  flow, reducing Q16/Ptera topology, loosening `2e-7`, or changing `dt`.
- strongest alternative hypothesis: the two-step result was correct, but the
  present fixed-point relaxation is not robust to accumulated wake loading.

## 5. Runtime Strategy

- command for smoke: focused 4-step CUDA long-horizon pytest.
- command for main run: focused 8-step/resume test, then selected
  Q16/LEV/TEV/FSI regressions.
- expected runtime / budget: pilot under 4 minutes; full focused gate under
  8 minutes; selected regression under 4 minutes on RTX 4090 D.
- log / artifact location: this directory; command summaries in
  `runlog.summary.md`.
- safe efficiency levers: cached Warp kernels and the accepted GPU-PCG only;
  never reduce the model or skip corrector evaluations.
- tooling fallback: managed experiment shell/artifact/memory interfaces are
  unavailable, so local non-interactive execution plus versioned artifacts is
  used and disclosed.

Monitoring: poll every 30--60 seconds for focused runs. Continue while step
markers advance and residuals remain finite; terminate a run on NaN/Inf,
mandatory-mode loss, GPU failure or owner-corruption evidence.

## 6. Fallbacks And Recovery

- if the four-step pilot fails, preserve the last committed prefix and diagnose
  residual history before changing one variable.
- if memory is tighter than expected, shorten retained host evidence, never
  particle physics or Q16/Ptera topology.
- if work closure fails while endpoint equilibrium passes, audit the exact
  Newmark endpoint identity before modifying the solver.
- if eight steps fail only at the iteration cap with monotone residual decay,
  record it as coupling-robustness evidence; do not silently loosen the cap.

## 7. Checklist Link

- checklist: `artifacts/experiment/20260822_q16_real_fsi_long_horizon/CHECKLIST.md`
- next experiment: define a frequency-bearing periodic FSI validation case and
  freeze cycle-count and cycle-to-cycle convergence gates before execution.

## 8. Revision Log

| Time | What changed | Why it changed | Impact |
|---|---|---|---|
| 2026-08-22 | initial freeze | move from two steps to an L1 long-horizon gate | physics and convergence gates unchanged |
| 2026-08-22 | freeze Newmark predictor before Newton | work audit proved the first correction mutated the predictor, violating exact kinematics | prior FSI numbers are not a valid numerical baseline; all affected gates will be rerun |
| 2026-08-22 | outer iteration cap 30 -> 48 | corrected step 2 decreased strictly from `6.81e-3` to `1.31e-6` in 30 iterations and projected to cross the unchanged tolerance in 7--9 more | runtime budget only; `2e-7`, `dt`, relaxation and physics unchanged |
| 2026-08-22 | fixed relaxation -> bounded CUDA Aitken; pilot cap restored to 30 | the 48-iteration retry entered a near-tolerance predictor/formal-corrector cycle instead of closing | coupling algorithm changes; acceptance tolerance and physical discretization remain frozen |
| 2026-08-22 | Aitken pilot cap 30 -> 64 | step 2 converged in 11 iterations; step 3 fell from `1.80e-2` to `1.20e-6` by iteration 30 | runtime budget only; Aitken, tolerance and physics unchanged |
| 2026-08-22 | add matrix-free CUDA GMRES fallback | step 4 effective tangent had non-positive curvature, where PCG is mathematically invalid | SPD steps retain PCG; indefinite Newton systems use separately counted GMRES |
| 2026-08-22 | left-precondition GMRES and retain up to 128 Krylov vectors | unpreconditioned/restarted variants stalled at `5.84e-7`/`1.02e-8`; the 96-DOF full-space solve converged in 72 iterations | step 4 now passes the unchanged `2e-10` linear gate; no physics change |
| 2026-08-22 | add geometry-guarded Newton initialization and correction backtracking | the step-5 Newmark extrapolation was orientation reversing although the committed step-4 state was valid | the predictor remains frozen for exact kinematics; it is no longer assumed to be an admissible Newton state |
| 2026-08-22 | long-gate Newton cap 15 -> 30 | guarded step 5 no longer inverted but exhausted 15 Newton iterations; step 4 already required 12 | diagnostic iteration budget only; structural tolerance, `dt`, material and loads unchanged |
| 2026-08-22 | split high-load boundary from damped history gate | the undamped `E=1e8` fixture approached Q16 geometric instability at step 5; making it stiffer alone did not remove dynamic energy growth | four-step result remains the high-load stress claim; eight-step history uses declared `E=1e9`, `alpha_M=20 s^-1` and `5e-8` structural tolerance |
| 2026-08-22 | add mass-proportional damping and damping work | a long-time structural equation without any dissipation was physically incomplete | damping enters residual, consistent tangent, preconditioner, trajectory identity and endpoint work balance |
| 2026-08-22 | close joint TEV/LEV active set | the step-5 coupled solution drove an initially inactive strip above the G3 margin | re-solve on CUDA until all violating strips are pinned; no G3 threshold was loosened |
