# Q16 Ptera Resolved-Load Transfer Plan

## 1. Run contract

- run id: `20260822_q16_ptera_resolved_load_transfer`
- tier: `auxiliary/dev` integration checkpoint
- selected idea: represent each real Ptera vortex-leg / unsteady-pressure load
  point by a same-panel, current-configuration algebraic stencil over the
  lower and upper Q16 support surfaces, then apply the exact transpose of that
  frozen stencil and the existing Q16 surface map.
- research question: can all five real Ptera load-point blocks per panel be
  transferred to Q16 coordinates while preserving force, moment and virtual
  work for arbitrary valid current geometry, including the trailing-vortex
  velocity/freestream offset?
- null hypothesis: a general current-geometry transfer cannot close all three
  invariants without inverse-fitting a Q16 parametric coordinate or smearing a
  net load.
- alternative hypothesis: eight local Q16 support points (four panel vertices
  on each of the two shell faces) span one conservative affine stencil for the
  current Ptera point and give an exact transpose load map.

## 2. Non-negotiable constraints

- Q16 only; no Q9 or low-order structural fallback.
- separated LEV, joint TEV and free wake remain mandatory and enabled.
- scientific tensors and the stencil solve remain CUDA float64; CPU is limited
  to immutable topology setup, orchestration, hashes and test evidence.
- no inverse fit of `(xi, eta, zeta)` and no equal-node force split.
- every load uses only the eight supports of its owning Ptera panel.
- the frozen per-trial map must preserve total force, total moment and virtual
  work; rank loss, excessive extrapolation, geometry drift or content drift
  stops before a structural/aerodynamic commit.
- this checkpoint must compose the resolved load with the already audited
  source-owned LEV impulse load; it must not relabel a partial load as complete.

## 3. Scientific abstraction

For load point `p`, let its owning panel expose eight Q16 support points
`x_pa`: the four panel vertices at shell coordinates `zeta=-1,+1`.  On CUDA,
solve the local minimum-norm affine system

`sum_a w_pa = 1`, `sum_a w_pa x_pa = x_p`.

The frozen current-trial transfer is

`f_a = sum_p w_pa f_p`, followed by the existing exact Q16 surface transpose.
Consequently, for the frozen stencil,

- `sum_a f_a = sum_p f_p`,
- `sum_a x_a cross f_a = sum_p x_p cross f_p`, and
- `delta q dot Q = sum_p delta x_p dot f_p`, where
  `delta x_p = sum_a w_pa delta x_a`.

This is a conservative local wrench embedding of the current Ptera point.  It
does not claim that a vortex-center force is a uniquely recovered surface
traction, and its weights are recomputed for every predictor/corrector trial.

## 4. Code and evidence map

| Path | Planned role |
|---|---|
| `src/fluxvortex/warp_fsi/q16_ptera_resolved_transfer.py` | CUDA local affine solve, deterministic transpose, resolved + LEV composition |
| `tests/test_q16_ptera_resolved_transfer_gpu.py` | synthetic independent oracle, real incremental branch, hostile gates |
| existing `q16_aero_load_packet.py` | immutable real point/force packet; no scientific formula change |
| existing `q16_lev_impulse_transfer.py` | source-owned LEV impulse transfer; no model change |

Expected durable outputs: this plan/checklist, test logs, metrics, claim
validation, summary and exact file hashes.

## 5. Acceptance and stop conditions

- synthetic curved Q16 geometry: force/moment error `<= 1e-11`, relative
  virtual-work error `<= 1e-11`.
- real incremental separated-LEV/joint-TEV/free-wake branch: all `5P` points
  reconstruct inside the frozen bound and resolved plus LEV generalized load
  is finite and nonzero.
- actual packet order must be five complete panel-major blocks.
- fail before output for host/float32/mixed-device tensors, topology mismatch,
  singular local support, nonfinite values, point mutation, source mismatch or
  packet/LEV impulse mismatch.
- focused and selected Q16/LEV/TEV/FSI suites remain green; Black, Ruff,
  py_compile and whitespace checks pass.
- no paper/full-matrix run in this checkpoint.

## 6. Runtime and tooling

- target runtime: under three minutes on the assigned CUDA device.
- bounded pilot first; no formal paper case.
- managed `bash_exec`, artifact and memory tools required by the experiment
  skill are not exposed in this session. Local non-interactive commands and
  repository evidence files are the disclosed fallback.

## 7. Claim boundary and next step

Passing this run supports a complete instantaneous Q16 generalized load
`Q_resolved + Q_LEV`. It does not yet prove a converged strong-coupling time
step or atomic structure+aerodynamic commit. The next checkpoint must install
this combined load in the Newmark predictor/corrector loop and publish both
owners only after convergence.

## 8. Revision log

| Date | Revision | Reason |
|---|---|---|
| 2026-08-22 | Initial freeze | Continue from the causal LEV impulse-work checkpoint into the general real-Ptera load path |
