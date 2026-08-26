# Q16 Real Strong-FSI Transaction Plan

## 1. Run contract

- run id: `20260822_q16_real_strong_fsi_transaction`
- tier: `auxiliary/dev` integration checkpoint
- hypothesis: one predictor/corrector step can repeatedly branch the same
  committed separated-LEV/joint-TEV/free-wake parent, evaluate the complete
  resolved+LEV Q16 load, solve the Q16 Newmark step from one structural
  pre-state, and publish both owners exactly once only after convergence.
- baseline: the audited Q16 Newmark solver, incremental Ptera owner, endpoint
  motion binder, conservative resolved transfer and source-owned LEV impulse
  transfer on branch `run/q16-lev-tev-pc-fsi-20260821`.

## 2. Frozen algorithm

1. Freeze one structural `(q_n,dq_n,ddq_n)` and one incremental aerodynamic
   parent `(bound, LEV, TEV, free wake)`.
2. Form the Newmark displacement/velocity predictor on CUDA.
3. For each outer coupling iteration, clone the same aero parent, bind the Q16
   trial `q,dq`, advance exactly one real aerodynamic step, construct the
   complete Q16 generalized load, and solve the structure again from the same
   pre-state.
4. Measure displacement and `dt*velocity` fixed-point residuals on CUDA.
5. Once the trial meets the tolerance, perform one final aero/structure replay
   at that trial. Publish only if the replay also meets the same tolerance.
6. Commit the latest issued aero branch and the validated structural result in
   one non-callback publication section. Any earlier or failed branch is
   discarded.

## 3. Non-negotiable constraints

- separated LEV, joint TEV and free wake remain mandatory; no reduced mode.
- Q16 only; no Q9/toy structure.
- all scientific arrays and relaxation updates remain CUDA float64.
- Ptera host objects are orchestration owners only; no host structural/load
  numerical fallback.
- every predictor evaluation reads the same pre-step aero/structural state.
- failure, nonconvergence or drift commits neither owner; clean retry must match
  a fresh owner.
- this checkpoint is a bounded real 2x3-panel integration pilot, not a paper
  case or an FSI validation claim.

## 4. Code and tests

| Path | Role |
|---|---|
| `src/fluxvortex/warp_fsi/q16_structural_solver.py` | public CUDA Newmark predictor used by the coupling owner |
| `platform/warp_vpm/q16_real_fsi_coupling.py` | structural+aero owner, fixed-point solve and joint commit |
| `platform/warp_vpm/test_q16_real_fsi_coupling.py` | real active-LEV success, rollback, clean retry and reduced-mode gates |

## 5. Acceptance

- a real next aerodynamic step evaluates nonzero separated-LEV, joint-TEV and
  free-wake state and reaches the Q16 Newton solve through the complete load.
- converged success increments aero and structural generations exactly once;
  predictor evaluation count may exceed one but live wake convection advances
  once.
- injected second-evaluation failure and forced nonconvergence leave both
  owners bitwise/hash unchanged; clean retry equals fresh execution.
- focused and selected integration suites plus static/whitespace gates pass.
- no GT, scorer, paper data or formal matrix is accessed.

## 6. Tooling and outputs

Managed experiment shell, artifact and memory tools are unavailable; local
non-interactive commands and this repository artifact directory are the
disclosed fallback. Expected outputs are plan/checklist, metrics, claim map,
summary, run log and exact hashes.

## 7. Claim boundary

Passing supports one bounded, real, transactionally committed Q16 strong-FSI
step. It does not prove time-marching stability, mesh/time convergence,
experimental accuracy, structural constitutive range or production scaling.

## 8. Revision log

| Date | Revision | Reason |
|---|---|---|
| 2026-08-22 | Initial freeze | Install the complete load in the actual predictor/corrector transaction |
