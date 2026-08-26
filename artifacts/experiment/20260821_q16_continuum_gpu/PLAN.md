# Fixed Q16 continuum residual/mass/Jv CUDA slice

## Objective

Implement a matrix-free, total-Lagrangian Q16 continuum baseline with full
Green--Lagrange strain, compressible St. Venant--Kirchhoff material response,
consistent mass action and analytic tangent action.  Establish independent
NumPy and CUDA float64 parity before adding ANS/EAS locking projection or a
nonlinear time integrator.

## Fixed equations

- `x = sum N_a (r_a + zeta g_a)` with 16 Q16 nodes and 96 coordinates.
- `F = dx/dX`, `E = 0.5 (F^T F - I)`.
- `S = lambda tr(E) I + 2 mu E`, `P = F S`.
- `Q_int = integral P grad_X(phi) dV0`.
- `Jv` uses the analytic linearization
  `dE=sym(F^T dF)`, `dS=lambda tr(dE)I+2mu dE`,
  `dP=dF S+F dS`.
- consistent mass action uses the same Q16 position/director interpolation.
- quadrature is fixed at 6x6 in-plane and 3 through thickness.

## Gates

1. finite rigid transform has zero strain energy and internal force;
2. analytic energy directional derivative matches `Q_int dot dq`;
3. analytic `Jv` matches centered finite difference;
4. mass action is symmetric/positive and reproduces rigid translational mass;
5. CUDA residual/mass/Jv match the independent NumPy oracle to `1e-10`;
6. repeated CUDA output is bitwise deterministic;
7. host/wrong-dtype/nonfinite state fails closed.

## Boundary

This slice is the unprojected continuum baseline.  It does not satisfy the
thin-shell locking claim until the frozen ANS/EAS projection and thickness-ratio
benchmarks pass.  It also does not integrate the real LEV/TEV/free-wake solver.
