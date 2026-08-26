# Q16 + LEV/TEV/free-wake PC-FSI H0/H1 checkpoint

## Outcome

The first implementation slice is green.  It establishes the fixed Q16
mathematical state and a fail-closed aerodynamic trial transaction without
changing the existing V5M aerodynamic equations.  This is a prerequisite, not
the completed Q16 FSI solver.

## Files and SHA-256

| File | SHA-256 |
|---|---|
| `src/fluxvortex/q16_ancf_shell.py` | `8d7f386fb0e41ae79cc2fe027ea149fb2d8aef6acb11b5ce8dcc7c62bfd3ed51` |
| `src/fluxvortex/warp_fsi/aero_step_transaction.py` | `006a0303dc1a7a9837b05aa76d5a4b14478619b16b7012cc0e5593eeda20bf4a` |
| `tests/test_q16_ancf_element.py` | `2cf1acd5eaefc818a80210189bbed38ebec212a2e25a27be57596c58f1d5cc46` |
| `tests/test_aero_step_transaction.py` | `1865e957f48a673f32d8cde357ec4451237e354a91ed1940a115f34b79a26e13` |

## Verification

- Focused: 25 passed.
- Joint with existing V5M FSI contract and active separated-LEV test: 36 passed.
- Black, Ruff, py_compile and whitespace checks: passed.
- No Q9 implementation, configuration or test was introduced.

## Next action

Implement the fixed Q16 continuum force/mass/Jv operators and their CUDA
float64 parity tests, then the Q16 work-conjugate surface transfer.  Only after
those gates pass should the generic transaction be adapted to the real
`CudaJointLEVTEVSolver` state tree and used by a strong predictor--corrector FSI
step with exactly one real wake commit.
