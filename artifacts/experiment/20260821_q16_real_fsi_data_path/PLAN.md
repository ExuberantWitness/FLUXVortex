# Q16 Real FSI Data Path Plan

## 1. Objective

- run id: `20260821_q16_real_fsi_data_path`
- selected idea: expose the real CUDA LEV/TEV solver's resolved aerodynamic
  point loads, preserve their physical application points, and apply the exact
  transpose of the Q16 surface interpolation map.  A non-zero LEV impulse
  without a defensible application-point model remains an explicit unresolved
  load and must stop a coupled commit.
- user's core requirement: continue the actual Q16 + mandatory separated-LEV +
  free-wake + predictor/corrector FSI data path.
- non-negotiable constraints: CUDA float64 scientific path; separated LEV and
  joint TEV always enabled; free wake always enabled; no Q9/toy structure; no
  arbitrary surface smearing of a net force; parent aerodynamic state changes
  only on an accepted transaction.
- research question: can the real solver's spatially resolved aerodynamic
  loads be transferred to shared-node Q16 generalized coordinates without
  violating force, moment, virtual-work, or transaction invariants?
- null hypothesis: the existing solver output is insufficient to construct a
  work-conjugate Q16 load vector.
- alternative hypothesis: the four bound-vortex leg forces and unsteady panel
  force are sufficient for an exact resolved-load path, while the particle
  impulse can be isolated and rejected until a separate localization contract
  is supplied.

## 2. Baseline And Comparability

- baseline id: commit `dc43d4cc0c2290ee34df942e51cbb05e13afbb0d`
  plus the untracked Q16 implementation recorded in
  `20260821_q16_review_bugfix`.
- baseline variant: real `CudaJointLEVTEVSolver`, Q16 MITC16/EAS shared mesh,
  exact transpose surface map.
- dataset / split: no paper-data run; deterministic synthetic transfer oracles
  plus a bounded real two-step free-wake LEV/TEV solver pilot.
- primary metrics: relative virtual-work residual, resolved force residual,
  resolved moment residual, packet mutation rejection, unresolved impulse
  rejection, parent-state hash preservation.
- required metric keys: `virtual_work_relative_error`,
  `resolved_force_max_abs_error`, `resolved_moment_max_abs_error`,
  `real_packet_point_count`, `unresolved_impulse_norm`, `focused_test_count`.
- comparability risks: correcting a dimensionally invalid moment transform may
  alter moment outputs; it must be recorded as a scientific bug fix rather than
  hidden as a refactor.  No paper-accuracy claim is permitted in this run.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why this is needed | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/bing_joint_ptera_gpu.py` | real CUDA Ptera/LEV/TEV loads | retain CUDA world-frame application points, forces, unresolved impulse, and correct wrench translation | the current public result loses the point-load decomposition | affects reported moments; requires rigid-transform oracle |
| `src/fluxvortex/warp_fsi/q16_aero_load_packet.py` | absent | exact CUDA packet, mutation seal, resolved Q16 transfer, geometry/impulse gates | creates the audited Ptera-to-Q16 boundary | Torch/Warp ownership and device semantics |
| `tests/test_q16_aero_load_packet_gpu.py` | absent | synthetic virtual-work, geometry, mutation and impulse gates | independent numerical oracle | must not become a toy substitute for real pilot |
| `platform/warp_vpm/test_q16_real_aero_branch_transaction.py` | real branch pilot | assert real packet force/moment closure and mandatory-mode behavior | exercises production solver output | bounded GPU/Ptera runtime |

## 4. Execution Design

- minimal experiment: synthetic Q16 packet whose points are produced by the
  exact CUDA interpolator; confirm transpose virtual work and net wrench.
- smoke / pilot plan: run one real two-step free-wake, joint-TEV,
  separated-LEV solver and validate its retained packet without committing an
  FSI step.
- full run plan: joint focused Q16 + real branch + active-LEV regression.  This
  is an integration run, not a paper matrix.
- expected outputs: code/tests, durable results summary, exact commands and
  explicit unresolved-load frontier.
- stop condition: any host numerical fallback, mixed device/dtype, force or
  moment mismatch, geometry mismatch, packet drift, parent mutation, or
  non-zero unresolved impulse presented as a completed generalized load.
- abandonment condition: the real solver cannot expose spatial point loads
  without changing its numerical force calculation.
- strongest alternative hypothesis: a panel-result-only transfer is adequate;
  rejected because it loses the four vortex-leg application points and cannot
  independently establish moment/virtual-work closure.

## 5. Runtime Strategy

- command for smoke: focused `pytest` on the new packet test and real branch
  test with plugin autoload disabled.
- command for main run: the existing 102-test Q16/LEV/TEV joint surface plus
  the new tests.
- expected runtime / budget: under 3 minutes on the assigned CUDA device.
- log / artifact locations: this directory; command output is reported in
  `RESULTS.md` because managed `bash_exec`/artifact/memory tools are not exposed
  in this session.
- safe efficiency levers: retain and concatenate already-computed CUDA tensors;
  use Torch-to-Warp shared CUDA memory; no recomputation of aerodynamic kernels.
- tooling: local non-interactive terminal is the disclosed fallback for the
  unavailable managed experiment shell.

Monitoring: focused commands are short; report after red test, green pilot, and
joint verification.  Kill on CUDA errors, non-finite values, or unexpected
paper/full-matrix execution.

## 6. Fallbacks And Recovery

- if Torch-to-Warp aliasing is unsupported, use an explicit device-to-device
  CUDA copy and record it; never stage scientific arrays through CPU.
- if the real packet has a non-zero unresolved impulse, validate the resolved
  portion but reject completed FSI transfer.
- if the moment transform correction changes existing force metrics, preserve
  force baselines and report the moment delta separately.
- if the route becomes non-comparable, stop before a paper run and retain the
  last 102/102 integration baseline.

## 7. Checklist Link

- checklist path: `artifacts/experiment/20260821_q16_real_fsi_data_path/CHECKLIST.md`
- next unchecked item: add tests-first red gates.

## 8. Revision Log

| Time | What changed | Why it changed | Impact |
|---|---|---|---|
| 2026-08-21 | Initial contract | Start actual load path after transaction/Q16 map baseline passed | No paper metrics or model parameters changed |
