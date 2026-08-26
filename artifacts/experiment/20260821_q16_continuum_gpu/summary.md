# Fixed Q16 continuum CUDA checkpoint

## Result

`PASS` for the registered unprojected continuum baseline.  The implementation
uses fixed 6x6x3 quadrature, full Green--Lagrange strain, StVK first Piola stress,
consistent mass action and the analytic matrix-free tangent action.  CUDA work
is parallel over batch, Q16 element, quadrature point and output coordinate;
no 96x96 tangent is assembled.

## Numerical evidence

| Gate | Result |
|---|---:|
| Focused continuum tests | 6/6 PASS |
| Full relevant joint suite | 51/51 PASS |
| Energy directional derivative relative error | `2.107712699851824e-10` |
| Analytic Jv vs centered-FD relative error | `1.405727182475283e-10` |
| CUDA force vs NumPy relative L2 | `1.937886705540085e-15` |
| CUDA Jv vs NumPy relative L2 | `3.286196128524644e-16` |
| CUDA mass vs NumPy relative L2 | `1.905362862908455e-16` |
| Rigid-motion energy | `1.056611592473532e-23` |
| Rigid-motion force L2 | `3.395128576005271e-08` |
| Black/Ruff/py_compile/diff-check | PASS |

Absolute CUDA differences reflect force scale; relative errors are at or near
float64 accumulation precision.  Repeated registered CUDA residual output is
bitwise identical.

## SHA-256

| File | SHA-256 |
|---|---|
| `src/fluxvortex/q16_ancf_continuum.py` | `4ef5d5ddd014f5373e46c4e1543372c843e2f0f800d6c0721c771da6f5c7929f` |
| `src/fluxvortex/warp_fsi/kernels_q16_ancf.py` | `4a19280865386295805ae2730f3c89b2f8184d0e73f98514a4fca033f352af52` |
| `tests/test_q16_ancf_continuum_gpu.py` | `0c88ff023c0ee9c572d9525374cc797e5aab06cc0291577cd675fa8292e3a297` |

## Remaining hard gates

- Freeze and implement ANS/EAS projection; run thickness-ratio locking tests.
- Add shared-node Q16 mesh assembly, boundary constraints and nonlinear
  generalized-alpha/Newton--Krylov time integration.
- Bind the real CUDA joint LEV/TEV/free-wake state owner to the transaction and
  combine it with the Q16 transfer/structure operators.
- Run structural benchmarks, coupled convergence, rollback and GPU scaling.

Therefore this checkpoint does not yet support a thin-shell, complete FSI or
co-design claim.
