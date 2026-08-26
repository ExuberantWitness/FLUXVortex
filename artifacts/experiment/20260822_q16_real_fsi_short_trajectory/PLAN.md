# Q16 Real FSI Short-Trajectory Plan

## 1. Objective

- run id: `20260822_q16_real_fsi_short_trajectory`
- selected idea: advance the existing real Q16 strong-FSI owner through a
  bounded sequence of committed time steps, preserving the coupled structural,
  separated-LEV, joint-TEV and free-wake history rather than rebuilding a
  one-step case at every time coordinate.
- user's core requirement: continue the actual Q16 + predictor/corrector + FSI
  path; separated LEV is mandatory and may never be disabled.
- non-negotiable constraints: Q16 only, CUDA float64 structural/load numerics,
  joint TEV and free wake active, no Q9/toy/reduced aero, no GT/scorer/paper run.
- research question: can two or more consecutive real coupled steps advance
  the same live owner with exact generation/history continuity and a durable
  per-step hash chain?
- null hypothesis: the second step exposes stale one-step assumptions, loses
  wake/LEV history, or cannot fail without corrupting the last good prefix.
- alternative hypothesis: every successful step advances both owners exactly
  once, consumes the previous committed state, and a failed next step leaves
  the completed prefix and live owner unchanged.

## 2. Baseline And Comparability

- baseline id: `20260822_q16_real_strong_fsi_transaction` plus
  `20260822_q16_gpu_pcg`.
- baseline variant: pitched one-element Q16, 2x3 Ptera panels, `dt=0.04`,
  `E=1e8`, mandatory separated flow and real strong coupling.
- dataset / split: synthetic bounded integration fixture; no experimental data.
- primary metrics: completed step count, structural/aero generation continuity,
  `_steps_done`, wake convection count, LEV particle/history continuity,
  per-step coupling residual and trajectory hash-chain validity.
- required metric keys: `completed_step_count`, `trajectory_sha256`, per-step
  result SHA, parent/result state SHA, solver step, LEV count, wake count,
  coupling/aero/CG counts and residuals.
- comparability risks: the existing test solver has a finite prebuilt problem
  horizon; trajectory length must remain inside it. Timing is diagnostic only.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/q16_real_fsi_trajectory.py` | absent | bounded sequential owner, immutable step records, chain and stopped prefix | make time continuity explicit and auditable | wrapper could accidentally hide a failed step |
| `platform/warp_vpm/test_q16_real_fsi_trajectory.py` | absent | consecutive-step, history, failure-prefix and determinism gates | expose stale one-step assumptions | real Ptera test runtime |
| existing one-step coupling/solver | trusted baseline | no algorithm change unless a concrete multi-step bug is found | preserve comparability | hidden horizon/state assumption |

## 4. Execution Design

- minimal experiment: call the real one-step owner twice on the same committed
  parent and inspect all coupled histories.
- smoke / pilot: two successful steps within the existing three-step fixture.
- full run: bounded two-step trajectory with `max_coupling_iterations=30`
  (the original tolerance remains exactly unchanged), plus a horizon-exhaustion failure on
  the next coordinate; verify exact completed prefix and unchanged failed-step
  owner. Repeat a fresh trajectory for deterministic hashes where supported.
- expected outputs: plan/checklist, production trajectory module, tests,
  metrics, run summary, claims and exact artifact hashes.
- stop condition: second step cannot converge, loses mandatory separated-flow
  state, or mutates the owner on a failed step.
- abandonment condition: continuity requires changing the frozen aerodynamic
  or structural method rather than fixing a trajectory ownership defect.
- strongest alternative hypothesis: a solver-horizon failure, not a coupling
  defect, explains inability to advance beyond the bounded fixture.

## 5. Runtime Strategy

- smoke command: focused two-step real Ptera/Q16 test.
- main command: trajectory tests followed by selected Q16/LEV/TEV/FSI suite.
- expected runtime / budget: under 4 minutes on the assigned RTX 4090 D.
- log / artifact location: this directory and pytest output summarized in
  `runlog.summary.md`.
- safe efficiency levers: reuse the frozen GPU-PCG stepper; do not loosen any
  tolerance or reduce the separated-flow model.
- tooling: managed experiment shell/artifact/memory interfaces are unavailable;
  local non-interactive commands and repository artifacts are the disclosed
  fallback.

Monitoring: commands under 60 seconds are awaited directly; longer commands
are polled every 30-60 seconds. Any non-finite state, generation drift,
mandatory-mode loss or owner mutation kills the route before a broader run.

## 6. Fallbacks And Recovery

- if the prebuilt horizon is exhausted, classify it as an exact failed
  coordinate and do not reinterpret it as numerical instability.
- if memory is tighter than expected, reduce only retained evidence, never the
  Q16/aero model.
- if the second step reveals a real ownership bug, repair only that boundary
  and rerun the two-step red case before broader testing.
- if fresh repetitions are not bitwise deterministic due to external Ptera
  objects, compare frozen semantic digests and explicitly downgrade the claim.

## 7. Checklist Link

- checklist: `artifacts/experiment/20260822_q16_real_fsi_short_trajectory/CHECKLIST.md`
- completion state: all frozen implementation, execution and evidence gates
  are closed; the selected next action is a longer-horizon history gate.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-22 | initial freeze | move from one committed step to a bounded trajectory | none; one-step physics/tolerances unchanged |
| 2026-08-22 | outer budget 10 -> 30 for trajectory fixture | second-step residual decreased monotonically but needed 22 iterations | runtime budget increases; coupling tolerance and accepted solution gate unchanged |
| 2026-08-22 | exact cloned-solver hashes -> semantic replay | Ptera instance digests include clone identity | exact hash chains remain valid within each run; cross-clone comparison is limited to physical/iteration fields |
