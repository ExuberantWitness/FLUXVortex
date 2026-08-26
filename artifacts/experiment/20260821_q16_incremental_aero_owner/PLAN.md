# Q16 Incremental Aerodynamic Owner Plan

## 1. Objective

- run id: `20260821_q16_incremental_aero_owner`
- selected idea: split the existing real CUDA Ptera trajectory orchestration
  into an explicit initialize / advance-one-step / finalize state machine while
  reusing the exact existing AIC, free-wake, joint-TEV, separated-LEV, load and
  wake-shedding methods.
- user requirement: continue the actual Q16 predictor/corrector FSI path with
  separated LEV always integrated.
- non-negotiable constraints: no attached-flow mode, no prescribed wake, no
  reduced TEV path, no LEV impulse smearing, no duplicated aerodynamic
  equations, CUDA float64 numerical work, and no claim of completed structural
  coupling from an execution-lifecycle refactor alone.
- research question: can a solver be advanced and branch-committed one state at
  a time while remaining numerically identical to its existing monolithic
  trajectory for the same frozen geometry sequence?
- null hypothesis: hidden initialization/finalization state makes stepwise
  execution drift from `run()`.
- alternative hypothesis: an exact orchestration state machine preserves all
  wake, LEV, TEV, load, ledger and CUDA-counter evidence.

## 2. Baseline And Comparability

- baseline commit: `dc43d4cc0c2290ee34df942e51cbb05e13afbb0d`
  plus the current untracked Q16 implementation.
- reference oracle: a fresh `CudaJointLEVTEVSolver.run()` using identical
  problem, configuration and free-wake setting.
- comparison fixture: bounded real CUDA separated-LEV/joint-TEV/free-wake case;
  no paper data, GT or scorer.
- exact comparison objects: per-step bound strength, TEV/LEV history, particle
  state, wake vortex locations/strengths, panel loads, retained Q16 packet,
  ledger/diagnostics and CUDA counters.

## 3. Code Translation Plan

| Path | Change | Reason |
|---|---|---|
| `platform/warp_vpm/q16_incremental_ptera_owner.py` | strict session state machine around exact CUDA solver | adds transactional single-step lifecycle without copying numerical kernels |
| `platform/warp_vpm/test_q16_incremental_ptera_owner.py` | red lifecycle, monolithic parity, branch isolation and hostile-order tests | independent integration oracle |

## 4. Execution Contract

- `begin()` accepts only a pristine exact real solver with mandatory
  separated-LEV + joint-TEV + free-wake configuration.
- `advance_one_step(n)` first initializes geometry `n`; when `n>0`, it then
  completes the pending wake birth from solved state `n-1` using geometry `n`,
  and finally runs the exact existing numerical method order for state `n`.
  This is algebraically the same ordering as monolithic Ptera, but makes wake
  birth causal: each predictor/corrector branch owns the wake created by its
  candidate geometry.
- `finalize()` is legal only after exactly `num_steps` advances and sets
  `ran=True` after the existing load finalizer.
- repeated/out-of-order begin, advance, finalize and monolithic `run()` mixing
  fail closed.
- the solver is pickle-clonable after every completed step; advancing a clone
  must not mutate the parent.

## 5. Stop And Scope

- STOP on any monolithic parity drift, parent mutation, lifecycle ambiguity,
  nonfinite state, reduced aerodynamic mode, or CPU numerical fallback.
- This run does not yet alter panel geometry between steps and therefore does
  not resolve the Newmark endpoint-velocity versus Ptera finite-difference
  kinematics contract.
- This run does not close the active-LEV impulse work-conjugacy blocker.

## 6. Runtime

- GPU: `cuda:0`, float64, NVIDIA GeForce RTX 4090 D.
- bounded tests only; expected under five minutes.
- managed `bash_exec`, artifact and memory interfaces are unavailable; local
  non-interactive commands and this durable directory are the disclosed
  fallback.

## 7. Revision Log

| Time | Change | Reason |
|---|---|---|
| 2026-08-21 | Initial incremental-owner contract | required before Q16 structural and aerodynamic state can be jointly committed |
| 2026-08-21 | Defer wake birth to arrival of the next trial geometry | preserves Ptera algebra while allowing distinct predictor/corrector branches to own distinct real wakes |
