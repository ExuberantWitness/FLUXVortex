# Run Log Summary

| Command / slice | Result |
|---|---|
| Black, Ruff, `py_compile`, `git diff --check` on affected files | pass |
| `tests/test_q16_structural_step_gpu.py` after damping ledger addition | 7 passed in 3.44 s |
| structural step + PCG regression before damping-specific test | 13 passed in 14.06 s |
| real coupling + trajectory regression | 12 passed in 53.58 s |
| joint CUDA LEV reference regression after active-set repair | 2 passed in 2.30 s |
| final 8-step 4+4 long-horizon test | 1 passed in 83.38 s |
| deterministic evidence rerun | same prefix/final hashes; exhausted step 9 left owner unchanged |
| final combined affected regression | 29 passed in 148.69 s |

Important failed diagnostics retained in the plan revision log:

- stale Newmark predictor alias;
- fixed relaxation and capped Aitken convergence limits;
- unpreconditioned/restarted GMRES stalls;
- undamped fifth-step geometry/nonlinear failure;
- joint-solve inactive LESP G3 failure before active-set closure.
