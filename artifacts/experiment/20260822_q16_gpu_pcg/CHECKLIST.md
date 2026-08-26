# Q16 GPU PCG Checklist

## Planning

- [x] numerical hypothesis and baseline frozen
- [x] Q16/CUDA-float64/mandatory-separated-flow boundaries frozen
- [x] governing residual and transactional semantics frozen
- [x] baseline CG count and synchronized time recorded

## Tests first

- [x] unpreconditioned versus PCG nonlinear-solution comparison
- [x] strict Krylov iteration reduction
- [x] invalid preconditioner fail-closed gate
- [x] real strong-FSI success and rollback gates

## Implementation

- [x] reusable per-Newton CUDA linearization workspace
- [x] exact mass plus condensed-material diagonal on CUDA
- [x] left-preconditioned CG with original residual oracle
- [x] no scientific host array conversion or CPU fallback

## Verification

- [x] focused numerical/performance test passes
- [x] real nonzero FSI integration passes
- [x] selected broad regression passes
- [x] Black/Ruff/pycompile/whitespace gates pass
- [x] metrics, claims, hashes and run summary written

## Next frontier

- [ ] multi-element GPU profiling and scaling
- [ ] multi-step flexible-wing trajectory
- [ ] time/Q16/aerodynamic-grid convergence
- [ ] classic experimental FSI validation case
