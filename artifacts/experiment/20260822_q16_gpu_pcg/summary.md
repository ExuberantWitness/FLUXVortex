# Summary

The Q16 structural corrector now freezes one CUDA ANS/EAS linearization per
Newton step and solves the unchanged effective tangent with a CUDA float64
material/mass Jacobi PCG. In the controlled nonzero high-stiffness case,
Krylov work fell 47.62% and synchronized median time fell 45.33%. The real
separated-LEV/joint-TEV/free-wake one-step FSI transaction and all rollback
gates remained green. This is a solver and one-step integration milestone, not
a multi-step or experimental FSI validation result.
