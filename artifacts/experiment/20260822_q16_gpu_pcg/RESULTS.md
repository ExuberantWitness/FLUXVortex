# Q16 GPU PCG Results

## Outcome

PASS for the bounded solver-performance checkpoint. The Q16 nonlinear
equilibrium and mandatory separated-flow FSI path are unchanged, while the
linear solve now reuses one CUDA linearization per Newton correction and uses
a CUDA condensed-material/consistent-mass Jacobi preconditioner.

## Controlled structural comparison

The comparison used the same high-modulus nonzero Q16 step, the same nonlinear
and linear tolerances, synchronized CUDA timing, one warm-up of each path and
three measured repetitions.

| Quantity | Unpreconditioned | GPU PCG |
|---|---:|---:|
| Newton iterations | 3 | 3 |
| total Krylov iterations | 336 | 176 |
| median synchronized time | 0.349053 s | 0.190826 s |
| final nonlinear relative residual | 7.131e-9 | 7.052e-9 |
| displacement norm | 5.912990182637e-5 | 5.912990182633e-5 |

This is a 47.62% Krylov reduction and a 45.33% synchronized wall-time
reduction in the controlled one-element workload. The test compares state,
velocity and acceleration against the unpreconditioned solve under the frozen
nonlinear tolerance.

## Independent diagonal oracle

At the stress-free reference state, where the geometric tangent is zero, all
96 CUDA condensed-material diagonal entries were compared with independent
CPU basis-vector applications of the projected Q16 tangent. The maximum
relative discrepancy was `1.52e-15`. The consistent-mass diagonal discrepancy
was `2.15e-16`. CPU is used only as a test oracle; the production PCG path
constructs and applies both diagonals on CUDA float64.

## Real FSI integration

The nonzero five-degree Q16 / real Ptera step still used separated LEV, joint
TEV and a convected free wake. It converged in 9 coupling iterations / 10
aerodynamic evaluations, retained 24 LEV particles, advanced wake convection
once, and published the two owners once. The final structural replay used 944
PCG iterations across 6 Newton corrections with residual `1.43e-9`.

Rollback, injected second aerodynamic-evaluation failure, structural-owner
drift and time-step drift all remained failure-closed. The complete selected
suite was `169 passed`; the focused post-format solver/FSI suite was
`18 passed`.

## Interpretation

The speedup comes from two complementary changes: the same Newton
linearization is no longer rebuilt every Krylov iteration, and the PCG basis
is scaled by a physically derived positive material/mass diagonal. The full
nonlinear geometric tangent is still evaluated in every matrix-vector product;
it was not dropped from the equation.

## Limits

This does not yet demonstrate multi-element scaling, multi-step stability,
mesh/time convergence, constitutive-range generality or agreement with an FSI
experiment. No paper data, GT, scorer or formal matrix was accessed.
