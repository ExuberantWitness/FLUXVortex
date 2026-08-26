# Claim validation

| Claim | Evidence | Verdict |
|---|---|---|
| One Q16 nonlinear structural trial executes its numerical operators on CUDA | CUDA-resident inputs/results; mass/internal/Jv/vector/CG implementation and 4 focused tests | supported |
| The clamped boundary remains exact | 24 constrained single-element DOFs exact in state, velocity and acceleration | supported |
| Nonconvergence is transactional | forced max-iteration failure leaves all inputs bitwise unchanged; clean retry=fresh | supported |
| The solver is production-scale optimized | only 1/8/32 development diagnostic, unpreconditioned CG, host scalar checks | inconclusive |
| Complete Q16 + separated LEV/TEV/free-wake FSI exists | real aerodynamic adapter and joint commit are absent | refuted for current checkpoint |
