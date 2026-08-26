# Q16 MITC16+ANS/EAS CUDA translation plan

## Scope

Translate the verified NumPy projected-strain and one-parameter EAS-condensed
operator to resident CUDA float64 arrays.  The GPU path must use the identical
fixed projection stencil, quadrature, enhanced mode, local condensation,
residual and analytic Jv.  It may not call the NumPy oracle in production.

## Parallel decomposition

- precompute immutable projection stencils and reference metric samples;
- one CUDA thread per `(batch,element,quadrature)` for projected strains and
  directional projected strains;
- one CUDA thread per `(batch,element)` for the bounded 108-point scalar EAS
  condensation reduction;
- one CUDA thread per `(batch,element,dof)` for deterministic quadrature
  gathering of residual and Jv;
- no floating-point atomic scatter and no host numerical fallback.

## Acceptance

- alpha, residual and analytic Jv match the independent NumPy oracle;
- repeated residual is bitwise identical;
- host, float32, wrong-shape and nonfinite inputs fail closed;
- legacy Q16, shared mesh, transfer, LEV transaction and V5M tests do not
  regress.

