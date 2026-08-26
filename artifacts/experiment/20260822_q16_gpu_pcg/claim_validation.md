# Claim validation

| Claim | Evidence | Verdict |
|---|---|---|
| The Q16 effective solve is GPU-preconditioned | CUDA diagonal kernels, PCG vector kernels and device/dtype gates | supported |
| The diagonal represents the implemented Q16 material and mass operators | all 96 entries agree with an independent basis oracle at the reference state | supported |
| PCG reduces Krylov work on the frozen nonzero step | 336 to 176 iterations | supported |
| PCG accelerates the controlled workload | median 0.349053 s to 0.190826 s | supported |
| The accepted nonlinear solution is materially unchanged | equal Newton count, residual gate, state/velocity/acceleration comparison | supported |
| Mandatory separated LEV, joint TEV and free wake remain integrated | real FSI test: 24 LEV particles, solved TEV, non-prescribed wake, one convection | supported |
| Failure remains transactional | nonconvergence, injected evaluation failure and drift tests | supported |
| The implementation scales efficiently to large Q16 meshes | no multi-element scaling campaign | not tested |
| Multi-step FSI is stable and experimentally accurate | no trajectory or experimental case run | not tested |
