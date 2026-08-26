# Q16 + LEV/TEV/free-wake predictor–corrector FSI: H0/H1 implementation plan

## 1. Objective

- run id: `20260821_q16_lev_tev_pc_fsi_h0_h1`
- tier: `auxiliary/dev`
- selected idea: implement the first production-facing slice of the Q16-only structural/FSI line: fixed Q16 kinematics plus a fail-closed aerodynamic `snapshot -> trial -> commit/abort` protocol suitable for separated LEV, newborn TEV and free-wake state.
- user's core requirements: Q16 only; separated LEV is mandatory; TEV/free wake participate in predictor–corrector; FSI must advance the real wake exactly once after convergence; GPU float64 is the eventual production path.
- non-negotiable constraints: no Q9; no attached-only production switch; no wake-off production switch; no CPU fallback claim; no mutation of live aerodynamic state during a trial evaluation.
- research question: can the existing V5M aerodynamic and FSI primitives be placed behind an exact transaction boundary while establishing the fixed Q16 mathematical state required by the later coupled solver?
- null hypothesis: the current mutable solver ownership prevents a safe PC integration without changing aerodynamic semantics.
- alternative hypothesis: a narrow immutable protocol plus fixed Q16 kinematics can make trial evaluation and single commit mechanically testable without altering current V5M baselines.

## 2. Baseline and comparability

- baseline commit: `dc43d4c` (`perf: parallelize FLUX-V5M GPU iteration`).
- baseline branch: `run/v5m-full-gpu-20260820`.
- active branch: `run/q16-lev-tev-pc-fsi-20260821`.
- baseline assets remain read-only: current Q4 ANCF, `GpuFluidSolve`, `CudaJointLEVTEVSolver`, existing GPU reports and profiles.
- primary metrics for this slice:
  - Q16 shape identities and cubic reproduction;
  - exact 16-node/96-DOF contract;
  - trial evaluation causes zero live-state mutation;
  - commit succeeds exactly once and rejects stale/foreign/double commit;
  - abort causes zero mutation;
  - failed trial leaves state unchanged and clean retry matches fresh execution.
- comparability risk: this slice does not yet produce a full Q16 structural force or a coupled physical trajectory; it must not be reported as completed FSI.

## 3. Code translation plan

| Path | Current role | Planned change | Why | Risk |
|---|---|---|---|---|
| `src/fluxvortex/q16_ancf_shell.py` | absent | fixed Q16 nodes, cubic basis/derivatives, 96-DOF geometry/state validation and independent reference operations | establish the real element contract without Q9 | basis/node ordering drift |
| `src/fluxvortex/warp_fsi/aero_step_transaction.py` | absent | exact snapshot/proposal/commit/abort protocol and immutable ledger | prevent PC trials from advancing live LEV/TEV/wake | false security if adapter bypasses protocol |
| `tests/test_q16_ancf_element.py` | absent | shape, derivative, polynomial, rigid transform and input attacks | tests-first Q16 evidence | tolerances hiding algebra errors |
| `tests/test_aero_step_transaction.py` | absent | zero mutation, stale/double/foreign proposal, injected failure and clean retry | establish transaction semantics | mock evidence is auxiliary only |

## 4. Execution design

- minimal experiment: Q16 mathematical tests plus an adversarial stateful fake aerodynamic engine whose snapshot contains separated-LEV, newborn-TEV and free-wake arrays.
- smoke: focused tests for the two new modules.
- joint pilot: focused tests plus existing V5M FSI contract and active-LEV tests when CUDA is available.
- expected outputs: source/tests, pytest log, static-quality results, summary and exact hashes.
- stop condition: any Q16 identity failure, any live mutation during trial/abort/failure, or any double/stale/foreign proposal accepted.
- abandonment condition: if immutable transaction semantics require changing the frozen V5M numerical equations in this slice, stop and redesign the adapter boundary before editing the solvers.
- strongest alternative hypothesis: the transaction should live inside `CudaJointLEVTEVSolver` rather than as a reusable protocol; this will be decided only after the protocol tests expose the exact state surface.

## 5. Runtime strategy

- focused command: `PYTHONPATH=src:platform pytest -q tests/test_q16_ancf_element.py tests/test_aero_step_transaction.py`
- joint command: `PYTHONPATH=src:platform pytest -q tests/test_q16_ancf_element.py tests/test_aero_step_transaction.py tests/test_flux_v5m_fsi_gpu_contract.py platform/warp_vpm/test_ptera_gpu_active_lev.py`
- static: `python -m py_compile ...`, `black --check ...`, `ruff check ...`, `git diff --check`.
- budget: under 2 GPU-hours; most H0/H1 protocol tests are CPU-only and do not claim GPU execution.
- durable output: this directory.
- tool fallback: the required `bash_exec/artifact/memory` interfaces are unavailable in this session; commands use the available terminal and logs are summarized locally.

## 6. Fallback and recovery

- if CUDA tests are unavailable: retain focused protocol results, record joint CUDA gate as blocked, and do not claim GPU integration.
- if current baseline tests fail independently: separate baseline failure from the new focused result; do not patch unrelated legacy code.
- if the transaction mock is insufficient: next slice must bind the exact `CudaJointLEVTEVSolver` mutable state tree before PC integration.
- no fallback to Q9, attached-only, wake-off or CPU production execution.

## 7. Checklist

- checklist: `artifacts/experiment/20260821_q16_lev_tev_pc_fsi_h0_h1/CHECKLIST.md`
- next item: freeze exact public schemas and write red tests.

## 8. Revision log

| Time | Change | Reason | Impact |
|---|---|---|---|
| 2026-08-21 | initial H0/H1 plan | user authorized continued implementation | establishes an auxiliary, non-claiming checkpoint |

