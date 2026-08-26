# Q16 CUDA Newmark--Newton--CG structural-step plan

- run id: `20260821_q16_newmark_newton_cg`
- tier: auxiliary/dev structural integrator gate
- baseline: frozen Q16 MITC16+ANS/EAS shared-mesh CUDA operator and boundary owner

## Research question

Can one nonlinear Q16 structural trial be advanced with all vector, mass,
internal-force, tangent-action and CG numerical work on CUDA while preserving
the exact clamped boundary and failing transactionally on nonconvergence?

## Frozen method

- Newmark average acceleration: `beta=1/4`, `gamma=1/2`;
- residual `M a(q) + f_int(q) - f_external`;
- matrix-free Newton action `M/(beta dt^2) + K_t(q)`;
- batched unpreconditioned CG for this first correctness slice;
- projected free-DOF solve and constrained reaction extraction;
- host activity limited to launch/control and bounded convergence scalars;
- no dense 96x96/global tangent, CPU numerical fallback, implicit upload, Q9,
  or changes to separated-LEV/free-wake policy.

## Acceptance metrics

- zero-load reference state is bitwise stationary;
- batched small-load final free residual meets the registered tolerance;
- clamped state/velocity/acceleration remain exact;
- all outputs finite and CUDA float64;
- input arrays remain bitwise unchanged after success and forced failure;
- max-iteration failure raises and clean retry matches a fresh solve;
- existing Q16/V5M joint tests and static checks remain green.

## Stop conditions

- non-positive CG curvature, NaN/Inf, nonconvergence, boundary drift, host
  input, float32 or mismatched shapes raise without returning a step result;
- failure blocks the real aero adapter and complete FSI.

## Claim boundary

This slice establishes one structural trial only. It is not yet a committed
FSI step, a structural benchmark, a scaled performance result, or evidence of
LEV/TEV/free-wake coupling.
