# FluxV v5c experiment plan

## Selected idea

Preserve the corrected v4b/UVLM result as the exact baseline and test one small
new mechanism: a strip-local, causal and bounded rate-sensitive leading-edge
suction-loss discrepancy. The first implementation does not introduce a
material LEV wake and does not replace the native TE wake or force owner.

## Non-negotiable requirements

1. With every v5c contribution disabled, phase histories and means reduce to the
   corrected v4b baseline within `1e-12`; any failure stops paper scoring.
2. The Figure 14 source `Cd0=0.057` is evaluated at the published `0.75c`
   reference and added exactly once. Baseline and candidate use the same ledger.
3. No paper name, case ID, observed residual table, phase fit, amplitude fit or
   offset fit may enter the physical model.
4. v5a full-equilibrium residual and standalone v5b wake/load replacement remain
   archived failures and are not fused into v5c.
5. The 22 previously inspected conditions are development/confirmation evidence,
   not held-out generalization.

## Baseline contract

| Dataset/channel | v4b primary metric |
|---|---:|
| Yang lift MAE | `4.554509817 gf` |
| Yang drag MAE | `2.643997474 gf` |
| Figure 14 CT RMSE, 14 markers (v5c0 corrected reference) | `0.024250786` |
| Figure 14 CT RMSE, 12 unique conditions (v5c0 corrected reference) | `0.025700353` |
| Baik filtered macro CL RMSE | `0.657541867` |
| Baik filtered macro CD RMSE | `0.345152419` |

Yang and Baik retain the frozen v4b baseline. Figure 14 uses the audited v5c0
reference-corrected baseline; the legacy `0.025949167/0.027508176` rows remain
reported but are no longer the v5c1 exact-reduction target. Metrics retain their
original units and are never averaged across datasets.

## Candidate mechanism

Let the existing v4b phase load be `F_4b`. The first candidate produces

`F_5c = F_4b + w_rate * Delta F_suction`,

where `w_rate` is causal and strip-local, and `Delta F_suction` acts only in the
section chordwise/axial suction ledger. It vanishes in attached flow and disabled
mode. At zero local flux-rate, a cold state remains zero while a previously
excited state decays analytically as `exp[-0.5 Delta tau]`; it is not reset
instantaneously. The state and discrepancy are bounded; the candidate may only
reduce the declared axial-suction magnitude and may not add a second
normal-force or full LDVM owner.

The canonical convective increment is frozen as
`Delta tau=|V_rel,3/4c,perp span| dt/c_local`. This is half of Izraelevitz's
published `Delta t_tilde=(2/c) integral(v_perp dt)`, hence the state pole is
`2|b|=0.5` for `b=-0.25`. A zero/ill-defined convective increment fails closed;
no adapter may insert an undocumented velocity floor. A cache proxy using an
LDVM constant reference speed must declare that substitution and is ineligible
for canonical promotion.

## Implementation map

- New core: `platform/forward_flight_benchmarks/fluxv_v5c_suction.py`.
- New tests: `platform/tests/test_fluxv_v5c_suction.py`.
- New shadow runner only after unit/mechanical gates pass.
- Existing v4b source and result artifacts remain read-only.

## Run order

### M0 — v5c0 ledger and identity

- Freeze source-defined reference-point API and single-add Cd0 contract.
- Test disabled-module identity, causality, boundedness, sign and unit scaling.
- Stop if any identity residual exceeds `1e-12`.

### M1 — v5c1 mechanical shadow smoke

- Run analytic section histories spanning attached, persistent and fast-rotation
  regimes without loading experimental target values.
- Require attached/cold-zero-rate exact limits, analytic hot-state decay at zero
  rate, a non-increasing/non-reversing axial suction ledger, finite state and
  stable refinement. This is a sign/ownership gate, not an independent
  aerodynamic-work balance.
- Stop if any parameter must be selected from the three target curves.

### M2 — frozen 22-condition development sweep

- Run exactly the existing metric code and condition set.
- Require non-inferiority in all six primary aggregate rows, both Figure 14
  amplitude subgroups and all eight Baik case/channel rows.
- Require at least one preregistered improvement larger than the corresponding
  numerical/digitization uncertainty.
- Stop on any material primary-channel regression; do not average it away.

### M3 — robustness and external transfer

- Time/grid/cycle/wake-retention and Baik geometry/thickness sensitivity.
- Freeze code and parameters before a genuinely unseen experiment.
- Only an unseen transfer can support a case-agnostic generalization claim.

## Compute budget

M0/M1 are CPU unit/synthetic tests. M2 begins with cache-compatible shadow
evaluation only if the required local kinematic fields are available; otherwise
it performs a bounded representative rerun before any full matrix. No broad
parameter sweep is authorized.

## Revision log

- 2026-08-14: initial v5c plan created after v5a rejection, v5b G1 failure and
  the complete 22-condition result-to-claim review.
- 2026-08-14: v5c0 replaced both the polar and source-Cd0 local reference from
  0.25c to the published 0.75c. Legacy v4b replay was exact; Figure-14 all-14
  RMSE changed from 0.0259492 to 0.0242508 without observation fitting.
- 2026-08-14: v5c1-RSLS was frozen before paper scoring. The observation-free
  mechanical suite passed exact-off, attached, zero-rate, boundedness,
  causality, axial-suction-sign, periodic-state and refinement gates. An
  independent audit corrected the nondimensional state pole from 0.25 to 0.5:
  the published b=-0.25 enters exp[2*b*(U/c)*dt], while this implementation uses
  delta_tau=(U/c)*dt. It remains unscored until a clearly labelled proxy run is
  complete.
- 2026-08-14: a second mechanical audit replaced BDF2 with a first-order causal
  backward flux-rate because absolute-valued BDF2 rectified a post-step
  overshoot into a false excitation. Non-finite states, ill-scaled flux ratios
  and overflowed rates now fail closed before any paper proxy is evaluated.
- 2026-08-14: canonical mechanical evidence is generated by the current default
  run `20260814_fluxv_v5c1_mechanical_pole05_rate1_reproducible`. Earlier directories named
  `mechanical`, `mechanical_audited`, `pole05_audited`, and `pole05_rate1_*`
  are superseded development snapshots and must not be cited as current-code
  evidence.
- 2026-08-14: the reproducible all-22 negative artifact is
  `20260814_fluxv_v5c1_proxy_all22_pole05_rate1_reproducible`. It records the
  unit rate-excitation scale, direct dependency and raw Yang-source hashes,
  package versions, and previous-cycle state needed to replay the periodicity
  gate. Its predictions are numerically unchanged from the frozen NO-GO run.
- 2026-08-14: the frozen non-canonical all-22 proxy was stopped. Yang drag and
  Baik macro CL/CD improved slightly, but Yang lift changed
  `4.554510 -> 4.555308 gf`, Figure-14 all-14 RMSE changed
  `0.0242508 -> 0.0242981`, and Baik W2/W4 drag failed strict channel gates.
  The candidate is archived without tuning. The next route is a literature and
  ownership audit of subcritical rotation-driven suction loss versus an
  exclusive attached/LEV/persistent owner.
- 2026-08-14: the source audit rejected a second pure axial subcritical state.
  Martínez-Carmena supplies a material circulation rate but not a validated,
  parameter-free causal onset/recovery law for the three target sections.
  The owner-capacity audit also showed that reweighting the existing UVLM,
  polar and paired-LDVM total-force vertices cannot satisfy the Figure-14
  25-degree gate even with a target-leaking oracle. A parameter-free v5d0
  shadow ledger is authorized solely to expose exclusive residual ownership;
  the next accuracy candidate is blocked on an explicit ULLT-to-UVLM line-item
  replacement or source-parity material-LEV implementation.
- 2026-08-14: audit of the existing causal-incidence owner found that the
  published Jones poles had been advanced with `U_inf*dt/c` although the
  source defines half-chord travel `d(t_tilde)=2|V_perp|dt/c`.  The isolated
  v5d1 correctness run is preregistered in `V5D1_SOURCE_CLOCK_PLAN.md`; it
  changes only this clock and the source reference point, has no target-tuned
  parameter, and remains non-canonical until same-layer induced strip velocity
  is exported.
- 2026-08-14: the reproducible v5d1 all-22 run passed every mechanical gate but
  failed the joint paper gate.  Baik improved in all eight CL/CD rows and Yang
  drag improved, while Yang lift and all four Figure-14 score groups regressed.
  The source-clock result is archived without tuning; the frozen follow-up is
  the Yang Eq. (11)--(12) A/L/P region shadow in
  `V5D2_REGION_OWNER_PLAN.md`.
