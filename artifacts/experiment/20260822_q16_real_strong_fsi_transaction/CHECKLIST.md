# Q16 Real Strong-FSI Transaction Checklist

## Planning

- [x] same-parent predictor/corrector algorithm frozen
- [x] complete resolved+LEV load required
- [x] CUDA float64 / Q16 / mandatory separated flow frozen
- [x] atomic success and rollback semantics frozen
- [x] bounded claim boundary recorded

## Tests first

- [x] successful real Q16 + LEV/TEV/free-wake step
- [x] repeated predictor branches leave live parent unchanged
- [x] injected evaluation failure commits neither owner
- [x] coupling nonconvergence commits neither owner
- [x] clean retry matches fresh result and counters

## Implementation

- [x] public CUDA Newmark predictor
- [x] sealed structural/aero joint owner
- [x] complete-load aero trial callback
- [x] CUDA fixed-point residual and relaxation
- [x] final replay and prepared joint publication

## Verification

- [x] focused tests pass
- [x] selected Q16/LEV/TEV/FSI joint suite passes
- [x] static and whitespace gates pass
- [x] durable metrics, claims, summary and hashes written
- [x] no paper/full-matrix/GT/scorer access

## Next frontier

- [ ] multi-step flexible-wing trajectory
- [ ] time/Q16/aerodynamic-grid convergence
- [ ] classic experimental FSI validation case
- [ ] GPU profiling and batch/mesh scaling
