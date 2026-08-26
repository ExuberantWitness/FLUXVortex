# Run log summary

- 2026-08-22: froze Q16 GPU-PCG hypothesis and acceptance gates.
- RED: new PCG tests failed because the production stepper had no preconditioner contract.
- Implemented a reusable per-Newton CUDA MITC16/ANS/EAS linearization.
- Implemented CUDA consistent-mass and condensed-material diagonals.
- Replaced unpreconditioned CG recurrence with PCG; retained `none` as an audit baseline.
- Controlled warm benchmark: 336 to 176 Krylov iterations; 0.349053 s to 0.190826 s median.
- Independent 96-DOF diagonal oracle passed.
- Real mandatory-separated-flow FSI: 5/5 focused gates passed.
- Selected Q16/LEV/TEV/FSI suite: 169/169 passed.
- Post-format focused solver/FSI suite: 18/18 passed.
- Black, Ruff, py_compile and whitespace checks passed.
- No GT, scorer, paper data or formal matrix accessed.
