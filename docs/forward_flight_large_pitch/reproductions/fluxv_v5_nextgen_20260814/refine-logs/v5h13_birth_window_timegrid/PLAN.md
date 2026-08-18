# V5H13 birth-window graded time grid plan

Status: preregistered before implementation or execution.  Approved at
research-pipeline Gate 1 (Idea B) on 2026-08-17; parameters below are frozen
before the first V5H13 number.

## Run contract

- Run ID: `v5h13-birth-window-timegrid`.
- Research question: can the preregistered convective accuracy gate
  `dt_eval * max|U-U_gal| / sigma <= 0.5` be passed for the full Baik-W2
  formal matrix by refining ONLY the time grid inside the birth window after
  each frozen release, with the physics contract, thresholds, and release
  schedule unchanged?
- Null hypothesis: birth-window refinement does not clear the gate at N=32
  without changing the physics contract.
- Baseline: immutable V5H12 formal-A STOP
  `/tmp/fluxv-v5h12-formal-A-20260817` (stop at N32/layer3/substep1/stage1,
  convective birth values 0.46-0.7, layer-2 window peak 0.4972).
- Observation access: `none`.  GT/scorer remain sealed.

## Selected change (Idea B, hostile-reviewed, falsifier-backed)

Deterministic graded time grid: for nominal level N in (32, 64, 128), each
release layer integrates

    schedule(N) = [T/(N*r)] * (k*r)  +  [T/N] * (N-k)

i.e. the first `k` sub-steps after every frozen release time are each replaced
by `r` sub-steps of width `T/(N*r)`; the remaining `N-k` sub-steps keep `T/N`.
Effective sub-steps per layer: `N_eff(N) = N - k + k*r`; total covered time
remains exactly `T`.  The SAME (k, r) policy applies to all N.

**Frozen parameters: r = 4, k = 5.**  (Approved at Gate 1; changing them later
requires a new branch.)

## Gate-accounting amendment (explicit, part of this preregistration)

The convective and Jacobian stage gates are evaluated with the sub-step delta
time ACTUALLY integrated for that stage (`dt_eval = dt_j` of sub-step j).
Thresholds stay 0.5 / 1.5.  This is honest accounting of the real
discretization, not a relaxation: in the refined window `dt_j = T/(N*r)`, and
the gate number scales accordingly.  STOP semantics are unchanged.

## Pre-declared quantitative predictions (frozen)

1. Layer-2 birth-window peak convective number ~= 0.4972 / r = 0.1243.
2. Layer-3 sub-step-1 (birth) value ~= v3 / r where v3 in (0.5, 0.7].
3. Measured values deviating from prediction/r by more than +/-20% => branch
   FAILS the integrator-delivery test (STOP; no parameter change).
4. Replay determinism: two runs produce byte-identical record hashes and call
   counts.
5. Convergence guard: at 32/64/128 the observed temporal order is >= 2, and
   N32 observables move TOWARD the N64 reference
   (|Q32^V5H13 - Q64| <= |Q32^V5H12 - Q64| on the shared prefix channels).

## Non-negotiable boundary

- Frozen V5H11/V5H12 files below are immutable controls; implementation is a
  mechanical fork into the V5H13 namespace with only the admitted deltas.
- Admitted deltas ONLY: (a) per-sub-step delta-time schedule in the forked
  stream macro; (b) schedule construction + per-substep gate evaluation in the
  forked coupling; (c) per-record actual `substep_delta_time` in executor
  rows; (d) runner expected-key/counter tables as functions of
  `N_eff(N) = N - k + k*r`; (e) V5H13 namespace labels and freeze paths.
- NO change to: W2 case, kinematics, release times (4/5/6), RK coefficients,
  thresholds 0.5/1.5, sigma birth values, ownership/Kelvin ledger semantics,
  artifact schema field sets (values of `substep_delta_time` become
  per-substep; the field set itself is unchanged), A/B rules.
- No result-driven promotion, no threshold relaxation, no sigma evolution
  (V5H12 brief ban carries over; Idea A is rejected for this lineage).

## Allowed change map (8 mechanical forks)

| New V5H13 path | Mechanical source | Only admitted delta |
|---|---|---|
| `src/fluxvortex/rvpm_ir_wrk3_v5h13_stream.py` | frozen `rvpm_ir_wrk3_stream.py` | `substep_delta_times` schedule parameter for the macro; validator binds per-record actual dt; counters keyed to `3*N_eff` |
| `platform/forward_flight_benchmarks/fluxv_v5h13_baik_coupling.py` | frozen `fluxv_v5h11_baik_coupling.py` | build `schedule(N)`, pass to forked stream; per-substep composer dt; V5H13 schema/leaf labels |
| `platform/forward_flight_benchmarks/fluxv_v5h13_baik_w2_executor.py` | frozen V5H12 executor | read per-record dt; stage-order count `3*N_eff`; V5H13 namespace |
| `platform/forward_flight_benchmarks/run_fluxv_v5h13_baik_w2.py` | frozen V5H12 runner | `EXPECTED_*` tables via `N_eff`; counter tables via `N_eff`; gate amendment text; V5H13 namespace |
| `platform/tests/test_fluxv_v5h13_baik_coupling.py` | frozen coupling test | schedule/order tests |
| `platform/tests/test_fluxv_v5h13_baik_w2_executor.py` | frozen V5H12 executor test | per-dt expectations |
| `platform/tests/test_run_fluxv_v5h13_baik_w2.py` | frozen V5H12 runner test | `N_eff` table tests |
| `tests/test_rvpm_ir_wrk3_v5h13_stream.py` | frozen stream test | schedule-parameter tests |

## Validation order

1. Rehash every frozen control before editing (STOP on drift).
2. Tests-first: schedule unit tests (grid sums to T, deterministic, N_eff
   counts), per-record dt binding, replay determinism — RED on the forked
   intermediate, then implement.
3. Analytic oracle (B0-class): graded grid on an analytic velocity field
   preserves order >= 2 and matches the uniform-grid solution within
   truncation error.
4. Frozen small-N focused tests for stream/coupling.
5. Fresh disposable N32/layer1 smoke with the V5H12 H8 checklist plus
   birth-window gate-margin printout; verify prediction 1 on the layer-2
   window via a diagnostic run of layers 1-2 only if admissible under the
   smoke scope (no formal artifact).
6. Fresh formal A (full matrix) -> if PASS -> fresh formal B -> 9-file
   semantic byte parity -> fresh read-only audit (inherit V5H12 H9-H10).
7. Failure at any step: archive exact-prefix STOP; no parameter rescue.

## Runtime and monitoring

Inherited from V5H12: formal A+B <= 2 CPU hours total; monitor 60/120/300/600/
1800 s; fresh destinations; RENAME_NOREPLACE; no reuse of any existing
artifact directory.

## Success and abandonment

Success = full-matrix formal A/B PASS under the amended honest accounting,
byte-identical A/B semantics, unchanged conservation audits.  Success does
NOT imply paper accuracy.  Any conservation-ledger drift, replay
non-determinism, prediction deviation > 20%, or gate failure is a strict STOP.

## Control surface and revision log

| Time | Change | Reason |
|---|---|---|
| 2026-08-17 | initial preregistration | Gate-1 approval of Idea B after literature survey, hostile review, and the sigma-resize falsifier |
