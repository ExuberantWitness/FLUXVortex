# N2.6e1-S0 actual-surface foundation result

Date: 2026-07-30  
Verdict: `S0_STATIC_ATTACHED_FOUNDATION_VALIDATED`  
Parent `N2.6e1` verdict: unchanged, `open`.

## Scope

This result validates only the equation and type foundation needed before an
unsteady strong viscous--inviscid implementation:

- a closed, clockwise, two-sided NACA four-digit wall;
- a constant-source plus uniform-circulation Hess--Smith attached solve;
- explicit canonical IBL order
  `[upper, lower] x [stagnation cut -> trailing edge]` and its signed map to
  contour order
  `[lower TE -> LE, upper LE -> TE]`;
- typed laminar `n` versus turbulent `C_tau` inventory;
- typed TE and separation wake branches;
- pressure/shear traction, panel force, and reference-point moment.

It does **not** validate a moving wall, unsteady Bernoulli pressure, active
IBL residual, strong interaction, a second separation wake, or Figure 12.

## Independent guards

The implementation was hardened after an independent audit found that the
first draft could accept forged geometry and did not have a unique
upper/lower orientation map.

The final guards include:

- every stored surface derivative is recomputed from the two side traces;
- reversed normals/tangents, translated midpoints, changed lengths, forged
  signed area, or inconsistent side traces fail closed;
- a circle fixture reproduces
  `Cp = 1 - 4 sin(theta)^2` to below `3e-13`;
- pressure lift converges to the Kutta--Joukowski circulation ledger;
- inviscid pressure drag converges toward zero;
- `+alpha/-alpha` lift and circulation are antisymmetric and drag symmetric;
- a near-coincident two-sided wall fails a condition-number guard instead of
  being mislabelled as the thin-sheet limit;
- a non-symmetric canonical IBL field round-trips through the contour map,
  including the lower-side tangent sign;
- a nonzero edge velocity without deficit is not misclassified as active
  viscous coupling;
- zero-length wake segments and inconsistent `n/C_tau` regimes fail closed;
- panel force and reference-point moment are derived from the same pressure
  and shear arrays.

## Numerical record

For NACA0015 at `alpha=4 deg`, `U=11 m/s`, with `n` panels per side:

| n | CL | CD | normal residual | source quadrature | trace-circulation quadrature |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.4903649221 | 2.58523e-3 | 4.34e-16 | 5.498e-3 | -5.023e-3 |
| 32 | 0.4922343583 | 9.37579e-4 | 3.33e-16 | 2.788e-3 | -2.708e-3 |
| 40 | 0.4925907189 | 6.90581e-4 | 4.57e-16 | 2.235e-3 | -2.200e-3 |
| 64 | 0.4930967783 | 3.77667e-4 | 7.37e-16 | 1.402e-3 | -1.407e-3 |
| 96 | 0.4933610604 | 2.32353e-4 | 9.75e-16 | 9.372e-4 | -9.500e-4 |
| 128 | 0.4934876583 | 1.67058e-4 | 2.10e-15 | 7.037e-4 | -7.170e-4 |
| 192 | 0.4936097856 | 1.06579e-4 | 1.32e-15 | 4.697e-4 | -4.811e-4 |

The normal and Kutta equations close to roundoff. Integrated source flux and
surface-trace circulation are collocation quadrature errors and converge
under refinement; they are not renamed as the later algebraic Kelvin ledger.

Validation commands:

```bash
PYTHONPATH=platform python -m unittest discover \
  -s platform/tests -p 'test_svi_dw*.py' -v
PYTHONPATH=platform python -m unittest discover \
  -s platform/tests -p 'test_claim_runtime*.py' -v
```

Observed: `15/15` SVI-DW tests and `11/11` claim-runtime regression tests
passed. `git diff --check` also passed.

## Claim writeback

`N2.6e1a` may be frozen as a validated numerical foundation. The parent
source-response claim remains open because all of the following are still
missing:

1. moving-body normal condition and unsteady potential history;
2. TE newborn emission and material wake Kelvin evolution;
3. active regime-aware Eq. (9)/(10) boundary-layer residuals;
4. simultaneous transpiration coupling;
5. separation topology, second wake, remeshing, and state projection;
6. one total unsteady Bernoulli pressure;
7. all eight Figure 12 response gates.
