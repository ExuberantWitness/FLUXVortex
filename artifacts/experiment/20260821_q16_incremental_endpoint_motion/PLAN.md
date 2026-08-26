# Q16 Incremental Endpoint Motion Plan

## 1. Objective

- run id: `20260821_q16_incremental_endpoint_motion`
- selected idea: bind the exact Q16 Newmark endpoint pair `(q_trial,
  dq_trial)` to one isolated incremental aerodynamic branch.  Keep the
  committed geometry history immutable and replace only the finite-difference
  motion operands used by the current Ptera step with a sealed endpoint-motion
  shadow.
- user's core requirement: continue the actual Q16 + mandatory separated LEV
  + joint TEV + free-wake predictor/FSI data path.
- non-negotiable constraints: no Q9/toy fallback, no attached-flow fallback,
  CUDA float64 Q16 interpolation, no silent host input upload, no parent wake
  mutation during predictor evaluation.
- research question: at identical parent state and identical `q_trial`, does a
  changed `dq_trial` causally change the real boundary condition, LEV/TEV/wake
  and load while leaving the parent bitwise unchanged?
- null hypothesis: `dq_trial` remains decorative and the branch result is
  independent of it.
- alternative hypothesis: an exact sealed endpoint-motion binding makes the
  branch result velocity-sensitive without changing the position owner or
  committed history.

## 2. Baseline And Comparability

- baseline id: `20260821_q16_incremental_trial_geometry`.
- baseline variant: Q16 `q_trial` controls real next-step panels; Ptera derives
  movement from committed/current position difference.
- dataset/split: bounded synthetic 2x3 Ptera panel / Q16 surface-map mechanism
  fixture; no paper data.
- primary metrics: same-q/different-dq load and wake deltas, parent SHA
  invariance, exact motion reconstruction error.
- required metric keys: `velocity_sha_equal`, `packet_sha_equal`,
  `force_delta_norm`, `wake_max_abs_delta`, `parent_unchanged`,
  `lev_particle_count`, focused and joint test counts.
- comparability risk: an endpoint velocity is not an interval-average velocity.
  It is used only as the current Newmark boundary velocity; the committed
  geometry and circulation histories remain the actual prior state.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `platform/warp_vpm/q16_incremental_ptera_owner.py` | real incremental Ptera lifecycle | add sealed vertex-velocity owner and current-step motion-shadow application | make every current Ptera motion operand consume the same endpoint velocity | incomplete shadow coverage |
| `platform/warp_vpm/q16_ptera_trial_kinematics.py` | CUDA Q16-to-panel geometry | add exact `q_trial,dq_trial` binder and evidence | carry structural endpoint velocity into aero branch | frame/sign error |
| `platform/warp_vpm/test_q16_incremental_endpoint_motion.py` | new tests-first boundary | deterministic, discriminative and hostile gates | independently reject decorative velocity | fixture-only false positive |

## 4. Execution Design

- minimal experiment: commit a zero-velocity first Q16 state, fork twice, bind
  the same next `q_trial` with two different `dq_trial` states, then advance one
  real separated-flow step.
- smoke: exact same `(q,dq)` branches must match; changed `dq` must change
  motion evidence and real load/wake state.
- full run: focused new tests followed by the existing Q16/LEV/TEV/transaction
  joint regression surface.
- expected outputs: code/tests plus a hashed auxiliary experiment package.
- stop condition: any parent mutation, reduced mode, non-finite/mixed-device
  input, missing motion operand, or inconsistent shadow rejects before a
  scientific step is issued.
- abandonment condition: if the real Ptera step cannot consume one consistent
  velocity field at collocation, LE/TE, vortex initialization and load points,
  record STOP rather than claim `dq_trial` integration.
- strongest alternative: retain causal displacement-only motion and postpone
  endpoint velocity until an aerodynamic time integrator is redesigned.

## 5. Runtime Strategy

- smoke command: focused CUDA pytest for the new endpoint-motion test.
- main command: existing bounded Q16 joint test list plus the new test.
- budget: under two minutes on the assigned RTX 4090 D.
- artifacts: `artifacts/experiment/20260821_q16_incremental_endpoint_motion/`.
- efficiency: reuse one committed parent and pickle forks; no paper matrix.
- managed `bash_exec`, artifact and memory services are unavailable in this
  environment; local terminal commands and repository artifacts are the
  disclosed fallback.

## 6. Fallbacks And Recovery

- environment/GPU failure: record blocked; never upload structural inputs from
  host.
- failed motion coverage: keep the prior geometry checkpoint as last known
  good and do not alter the four-paper result path.
- non-comparable result: classify as partial and retain the exact counterexample.

## 7. Checklist Link

- checklist: `CHECKLIST.md`
- next item: write RED tests for identical-q/different-dq real branches.

## 8. Revision Log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-21 | Initial endpoint-motion contract | next gate after real Q16 position binding | auxiliary mechanism only; no paper claim |
