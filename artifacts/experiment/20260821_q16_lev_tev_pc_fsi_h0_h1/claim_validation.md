# Claim validation

## Supported at this checkpoint

- The fixed structural interpolation is Q16 only: sixteen 4x4 tensor-product
  Gauss--Lobatto nodes, six coordinates per node and 96 coordinates per element.
- The independent NumPy reference reproduces tensor cubics, finite rigid motion,
  thickness stretch and consistent reference mass for the registered cases.
- A trial/commit control-plane state includes bound circulation, separated LEV,
  newborn TEV and free-wake geometry/circulation.
- Repeated predictor trials do not advance the live parent; abort is non-mutating;
  only the latest exact issued proposal can commit; mutation followed by failure
  restores the parent and a clean retry matches a fresh execution.
- Existing V5M GPU FSI contract and active-LEV regression tests remain green in
  the registered joint suite.

## Not supported yet

- No CUDA Q16 residual, mass action, tangent/Jv or nonlinear structural solve has
  been implemented.
- ANS/EAS locking control, work-conjugate aerodynamic transfer and real solver
  adaptation are not yet implemented.
- The generic transaction oracle is not yet connected to
  `CudaJointLEVTEVSolver`; therefore no real separated-LEV/TEV/free-wake FSI time
  step or one-commit ledger has been demonstrated.
- No structural benchmark, flexible-wing experiment, performance result or
  co-design claim is unlocked.

## Decision

`GO` only to the next implementation slice: Q16 continuum residual/mass/Jv and
work-conjugate transfer, followed by an exact adapter around the real GPU
LEV/TEV/free-wake owner.  `STOP` for any full-FSI or co-design claim.
