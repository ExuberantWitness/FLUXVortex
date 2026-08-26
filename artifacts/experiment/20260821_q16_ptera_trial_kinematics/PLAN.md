# Q16-to-Ptera Two-State Trial Kinematics Plan

## 1. Objective

- run id: `20260821_q16_ptera_trial_kinematics`
- selected idea: make one real aerodynamic trial consume Q16 `q_trial` and
  `dq_trial` by constructing the two Ptera geometry states
  `q_previous = q_trial - dt*dq_trial` and `q_current = q_trial` through the
  exact CUDA Q16 surface interpolator.
- user requirement: continue the actual Q16 + mandatory separated-LEV + joint
  TEV + free-wake predictor/corrector FSI path.
- non-negotiable constraints: Q16 only; CUDA float64 interpolation and frame
  transform; no attached-flow/prescribed-wake fallback; no arbitrary LEV
  impulse distribution; no claim of repeatable multi-step FSI until the solver
  has a genuine incremental-step owner.
- research question: does changing Q16 `q_trial/dq_trial` change the exact Ptera
  panel geometry and the real aerodynamic result while leaving the live parent
  transaction state unchanged?
- null hypothesis: the current trial inputs remain disconnected from Ptera.
- alternative hypothesis: an exact two-state geometry envelope is sufficient
  to demonstrate one real predictor interval and expose the remaining
  incremental-owner blocker.

## 2. Baseline And Comparability

- baseline: the 109/109 real-load-path checkpoint under commit
  `dc43d4cc0c2290ee34df942e51cbb05e13afbb0d` plus untracked Q16 sources.
- dataset: bounded 2-chordwise by 3-spanwise flat-wing Ptera fixture; no paper
  data, GT, or scorer.
- metrics: panel-vertex geometry error, inferred vertex-velocity error,
  aerodynamic packet SHA change, parent solver hash preservation, test count.
- comparability: both comparison branches use identical mandatory aerodynamic
  modes, `lesp_crit=0.001`, dt and panel topology; only `dq_trial` changes.  The
  explicit low LESP threshold makes separated-LEV activity observable in this
  bounded two-state mechanism test and is not a paper-calibration claim.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/q16_ptera_trial_kinematics.py` | absent | exact panel-vertex topology plus two-state CUDA adapter | connects q/dq to real Ptera geometry | Ptera Panel objects are host orchestration containers |
| `platform/warp_vpm/test_q16_ptera_trial_kinematics.py` | absent | geometry, velocity, real-load, drift, mode and lifecycle gates | tests the production boundary | must not overclaim incremental time marching |

## 4. Execution Design

- minimal experiment: map one Q16 surface to 12 panel vertices and reconstruct
  both Ptera states; independently recover velocity from the two states.
- pilot: run two fresh real two-step solvers with equal final q and different
  dq; require different sealed load packet while all mandatory modes remain on.
- full run: previous 109-test joint surface plus new tests.
- stop: mixed device/dtype, topology mismatch, nonfinite/degenerate panel,
  mutation of q/dq, already-run solver, reduced aero mode, or unchanged loads
  under a discriminative dq perturbation.
- abandonment: Ptera cannot accept reconstructed immutable Panel objects without
  private cache mutation or a host numerical solve.

## 5. Runtime Strategy

- GPU: assigned `cuda:0`, float64.
- expected runtime: under 3 minutes.
- managed experiment shell/memory/artifact interfaces are unavailable; local
  non-interactive terminal plus durable files are the disclosed fallback.
- no long run or monitoring loop is required.

## 6. Fallbacks And Recovery

- if Ptera panel replacement fails, stop rather than mutate individual
  set-once Panel geometry fields or their derived caches in place.  The only
  admitted private seam is atomic replacement of the whole `_panels` owner on
  the isolated branch because upstream Ptera has no post-construction remesh
  API.
- if the two-state runner cannot preserve the parent, keep only the kinematic
  packet and do not connect it to branch commit.
- if actual loads do not change, diagnose geometry collapse and motion terms;
  do not relax the assertion.

## 7. Checklist Link

- `artifacts/experiment/20260821_q16_ptera_trial_kinematics/CHECKLIST.md`

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-21 | Initial two-state scope | smallest honest bridge before incremental solver refactor | narrows claim to one predictor interval |
| 2026-08-21 | Permit atomic `_panels` owner replacement on isolated branch | public Ptera setter is construction-only; replacing the owner avoids mutating set-once Panel coordinates/caches | private integration seam is explicit and must be removed if Ptera gains a remeshing API |
| 2026-08-21 | Add active-LEV transaction STOP gate | predictor must not treat unresolved impulse as a completed Q16 force | proves parent rollback and records the exact remaining scientific blocker |
