# Q16 work-conjugate CUDA transfer plan

## Objective

Implement the fixed-Q16 aerodynamic surface kinematics and the exact transpose
load map used by predictor--corrector FSI.  The same interpolation must map Q16
position/director velocities to aerodynamic points and map aerodynamic forces
back to all 96 element coordinates.

## Frozen contract

- Q16 only: 16 nodes, six coordinates per node, 96 per element.
- Surface coordinate is `(xi, eta, zeta)`; nonzero `zeta` must load director
  coordinates and therefore retain the aerodynamic moment arm.
- For every registered state increment and force, `dq dot Q == dx dot f`.
- Total force and total moment must be preserved by the transpose map.
- Production wrapper accepts only Warp float64 arrays already resident on CUDA;
  there is no host numerical fallback or implicit state/force upload.
- Parametric point maps are immutable setup data and may be uploaded once when
  constructing the operator.
- No Q9, equal-four-node transfer, attached-only or wake-off alternative is
  introduced.

## Tests-first gates

1. CPU-independent Q16 oracle: virtual-work identity, force and moment balance.
2. A nonzero surface offset must produce nonzero director generalized force;
   the old equal-four-node distribution is an explicit negative control.
3. CUDA batched interpolation and transpose transfer agree with the oracle to
   `1e-11` in float64.
4. Host Warp arrays, wrong dtype/shape, out-of-range coordinates and nonfinite
   inputs fail before kernel launch.
5. Existing Q16, transaction, V5M GPU FSI and active-LEV suites remain green.

## Claim boundary

Passing this slice proves the Q16 geometry/load transfer operator and its CUDA
execution.  It does not yet prove the Q16 structural residual/Jv, nonlinear
solve, real LEV/wake adapter or coupled trajectory.
