# v5h9 M0/M1 Implementation and Run Plan

## 1. Objective

- run id: `fluxv-v5h9-live-boundary-m0-m1-20260815`
- tier: `auxiliary/dev` mechanism gate; it cannot carry production or paper-accuracy claims.
- selected idea: replace the v5h8 gross clone/counter physical cloud with one canonical live-boundary net state.  Preserve an immutable event ledger as the audit control plane, but never transport ledger contributions as particles.
- user's core requirement: keep advancing the three-dimensional FluxV research with explicit evidence and honest stop/go decisions.
- non-negotiable constraints: no target observation, no Ptera solver/load, no feedback writes, no threshold/core tuning, no remesh implementation in M0/M1.
- research question: can a unique live-boundary owner reproduce v5h8 clone-then-collapse exactly while retaining zero counter-pair particles and full transactional rollback?
- null hypothesis: the direct gamma update changes the physical state or requires hidden clone/remesh operations.
- alternative hypothesis: under exact material-basis compatibility, direct live-slice gamma update is the canonical reduction of v5h8 and leaves all non-boundary state bitwise unchanged.

## 2. Baseline and Comparability

- baseline id: `v5h8-incremental-sheet-f30b5fbf`.
- baseline variant: gross upstream clone followed immediately by canonical collapse, releases 1--4.
- data / split: deterministic manufactured zero and affine histories only; no external data.
- primary metrics: physical-array parity, induced velocity/Jacobian, impulse and rVPM RHS residual.
- required metric keys: `positions_equal`, `sigma_equal`, `ids_equal`, `lineage_equal`, `changed_indices_exact`, `gamma_residual`, `velocity_residual`, `jacobian_residual`, `impulse_residual`, `rhs_residual`, `clone_count`, `fresh_upstream_count`, `parent_unchanged`, `clean_retry_equal`.
- tolerance: exact fields use bitwise equality; floating field reductions use `tau=64*eps*max(1, ||reference||inf)` and must also remain below `2e-14`.
- comparability risks: using a fresh upstream discretization, changing particle count/core, modifying birth lineage, or allowing a gross pair into transport invalidates the comparison.

## 3. Code Translation Plan

| Path | Current role | Planned change | Why this is needed | Risk |
|---|---|---|---|---|
| `platform/forward_flight_benchmarks/fluxv_v5h8_incremental_sheet.py` | frozen oracle/baseline | read-only | defines clone-collapse reference and manufactured state | accidental baseline drift |
| `platform/forward_flight_benchmarks/fluxv_v5h9_live_boundary_owner.py` | absent | add immutable plan/result/state/event-ledger API | canonical net state and unique writer | circular attestation or incomplete validation |
| `platform/tests/test_fluxv_v5h9_live_boundary_owner.py` | absent | add M0/M1 positive/negative/rollback gates | executable claim-to-metric evidence | self-referential oracle |
| `docs/.../v5h9_live_boundary_owner/CHECKLIST.md` | absent | maintain execution frontier | auditable state | stale checklist |

No existing production, v5h8, bridge, transport, Ptera or load file may be changed by M0/M1.

## 4. Execution Design

- minimal experiment: construct one compatible boundary, plan/commit one release, compare to explicit v5h8 clone-collapse, then replay after one injected failure.
- smoke plan: focused tests for schema, one update, one mismatch, one cap failure and clean retry.
- full M0/M1 run: releases 1--4 for zero/affine states plus the complete 1-ULP/type/finite/identity/replay mismatch matrix.
- expected outputs: module, focused tests, test log, source SHA, claim verdict and updated tracker/checklist.
- stop condition: any output clone/counter, fresh upstream deposition, non-boundary mutation, support mismatch commit, parent mutation on failure, or residual above gate.
- abandonment condition: exact reduction requires changing v5h8 baseline or weakening compatibility/tolerance.
- strongest alternative hypothesis: v5h8 parity appears only because the test reuses the same collapse implementation; mitigate with independent array/field/impulse/RHS reconstruction.

## 5. Runtime Strategy

- smoke command: isolated focused pytest with `PYTHONPATH=src:platform`, plugin autoload disabled and task-specific Numba cache.
- main command: the same isolated environment over focused v5h9 plus frozen v5h8/rvpm reference tests.
- expected runtime / budget: under 15 minutes local CPU; no GPU.
- artifact location: preliminary logs under `/tmp`; durable bounded artifact only after M0/M1 passes and fresh audit agrees.
- safe efficiency levers: reuse frozen fixtures and vectorized reference kernels; do not change physics or tolerances.
- execution-tool limitation: the selected skill's `bash_exec`, memory and artifact control-plane tools are not exposed in this session.  Repository files, task-scoped commands and explicit logs are the fallback truth source; this limitation must be reported and no claim of managed-session provenance is allowed.
- workspace limitation: the shared repository is intentionally dirty with user-owned research work, so no branch/worktree mutation is attempted.  Changes are limited to the two new v5h9 files and this control directory.

Monitoring plan: tests are bounded to seconds/minutes; inspect immediately at completion.  Kill/revise on non-finite output, allocation beyond preregistered cap, unexpected import of Ptera/target modules, or repeated failure without a new hypothesis.

## 6. Fallbacks and Recovery

- environment/import failure: use isolated namespace loading already proven by v5h8; do not edit package `__init__` in this run.
- memory tighter than expected: reduce only attack batching, never the preregistered cases; cap is checked before allocation.
- wrong code path after smoke: stop, record first failing invariant, restore last-known-good candidate through forward patching only.
- non-comparable full run: mark partial/blocked and do not launch M2/remesh.

## 7. Checklist Link

- checklist: `docs/forward_flight_large_pitch/reproductions/fluxv_v5_nextgen_20260814/refine-logs/v5h9_live_boundary_owner/CHECKLIST.md`
- next unchecked item: freeze public M0/M1 API between implementation and independent tests.

## 8. Revision Log

| Time | What changed | Why it changed | Impact on comparability or runtime |
|---|---|---|---|
| 2026-08-15 21:46 | Initial M0/M1 plan frozen | v5h8 passed; next owner mechanism is now concrete | none; baseline remains read-only |
| 2026-08-15 21:46 | Pedrizzetti order added as mandatory M2 follow-up | gross/collapse does not commute with nonlinear relaxation | narrows M0/M1 claim; prevents premature production use |
