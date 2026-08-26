# Q16 Real FSI Long-Horizon Results

## Outcome

The bounded long-horizon development gate passed on CUDA.  One live Q16 FSI
owner advanced four steps and resumed for four more without resetting its
structural, LEV, TEV, free-wake or trajectory-hash history.  The accepted
duration is `0.32 s`, or `1.28 c/U` for `c=1 m` and `U=4 m/s`.

This is an L1 long-horizon integration result, not a multi-cycle validation:
the fixture has static inflow and no imposed periodic frequency.

## Frozen Long-Horizon Fixture

- one Q16 MITC16/EAS macro element; no Q9 or reduced structural model;
- 2 chordwise x 3 spanwise aerodynamic panels;
- mandatory separated LEV, joint TEV and free wake enabled for every step;
- CUDA float64 numerical paths only;
- `dt=0.04 s`, `E=1e9 Pa`, density `950 kg/m^3`;
- mass-proportional damping `alpha_M=20 s^-1`;
- structural/coupling tolerances `5e-8` / `2e-7`;
- bounded CUDA Aitken coupling, maximum 64 iterations.

The earlier undamped `E=1e8 Pa` stress fixture remains a separate boundary
result: it passed four steps, including one indefinite-tangent GMRES fallback,
but its fifth-step predictor approached element inversion.  It was not relabeled
as an eight-step success.

## Primary Evidence

| Gate | Result |
|---|---:|
| accepted steps | 8 / 8 |
| owner / aero generations | 8 / 8 |
| wake convection count | 1 through 8, exactly once per accepted step |
| LEV particle count | 24, 36, 36, 36, 48, 60, 72, 84 |
| maximum coupling residual | `1.1875092024682785e-7` |
| maximum structural residual | `3.0716056356639344e-8` |
| maximum endpoint-work relative residual | `2.4513724383723456e-12` |
| indefinite fallbacks / GMRES iterations | 3 / 216 |
| exhausted ninth coordinate | rejected; eight-step owner unchanged |
| focused long test | `1 passed in 83.38 s` |

The step-5 LEV increase from 36 to 48 is the regression evidence for the new
joint TEV/LEV active-set closure: an initially inactive strip crossed the G3
margin after the coupled solve, was activated, pinned and re-solved rather than
silently ignored or causing history loss.

## Bugs Exposed And Repaired

1. The frozen Newmark predictor aliased the mutable Newton state.  The first
   correction therefore corrupted all later acceleration reconstructions.
2. PCG was used after the effective tangent became indefinite.  A separately
   counted, left-preconditioned, matrix-free CUDA GMRES path now handles that
   branch and verifies the original residual before acceptance.
3. Predictor extrapolation could be geometrically inadmissible.  The predictor
   remains the exact Newmark kinematic reference, while the Newton initial guess
   and corrections are geometry guarded.
4. The structural model had no damping term, so a long static separated-flow
   run accumulated unbounded dynamic energy.  Configurable mass damping now
   enters the residual, consistent tangent and work ledger.
5. The joint TEV/LEV solve classified strips only before solving.  A coupled
   inactive strip could cross the LESP margin and fail G3.  The CUDA active set
   now closes and the final active set owns shedding and diagnostics.

## Boundaries

- No paper or experiment accuracy claim follows from this synthetic fixture.
- `alpha_M=20 s^-1` is a declared development parameter, not a calibrated
  material damping value.
- Eight static-inflow steps do not establish periodic steady state, flutter
  onset, limit-cycle oscillation accuracy or multi-cycle durability.
- The single-element high-load fixture still has a demonstrated geometric
  horizon at its fifth undamped step; that failure is transactional and remains
  useful as a stress gate.

