# V5H15 birth-sigma scheduling plan

Status: preregistered before implementation.  Approved continuation of the
Option-1-vs-Option-2 decision; Option 1 (longer window) was falsified by the
k=12 diagnostic probe, Option 2 (KAPPA birth-sigma scaling) passed it.

## Selected change

Fork V5H13 (8 files, graded grid K=5/R=4 inherited unchanged) into the V5H15
namespace with exactly one behavioral delta: the birth smoothing radius

    ROW_SMOOTHING_RADIUS_M: 0.00152 -> 0.00266  (= 0.00152 * KAPPA, KAPPA = 1.75)

applied uniformly at particle deposit (whole-layer birth discretization, a
priori, position-independent; no runtime or a-posteriori sigma change).

## Evidence chain

- V5H12 formal-A STOP (uniform grid, frozen sigma): layer-3 birth > 0.5.
- V5H13 formal-A STOP (graded k=5): prediction 0.1243 exact-hit; layer-3 first
  coarse step 0.547-equivalent.
- Probe k=12: FAIL; window-end value ROSE to 0.664 — the layer-3 transient
  grows through coarse-time 12; extending the window is unbounded.
- Probe KAPPA=1.75 on the V5H13 chain: FULL PASS, coarse peaks
  0.0325/0.2018/0.2355 (margin >53%), better than the conservative 1/KAPPA
  scaling because larger sigma also smooths the induced field.

## Pre-declared predictions (frozen)

1. N32 per-layer coarse peaks within probe values +/-20%:
   L1 0.0325, L2 0.2018, L3 0.2355; every stage gate value < 0.35 at ALL
   levels.
2. N79/N143 coarse peaks approx N32 values * 47/79 and * 47/143 (+/-20%).
3. Replay determinism byte-identical; conservation audits unchanged.

## Boundary

- Physics contract unchanged (kinematics, release times, RK, thresholds
  0.5/1.5, N roles, ownership/Kelvin ledger semantics).
- KAPPA frozen after the first formal number; failure => STOP, no rescue.
- GT/scorer sealed until inherited outer gate + unlock.

## Validation order

Fork + constant + directed tests -> static gates -> focused suites ->
compact hostile review -> dependency re-sign (old tokens fail closed) ->
graded+kappa disposable smoke -> formal A -> (if PASS) formal B -> 9-file
byte parity -> read-only audit.
