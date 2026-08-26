# Q16 Ptera Resolved-Load Transfer Checklist

## Planning

- [x] Q16 / mandatory LEV / joint TEV / free-wake scope frozen
- [x] current-configuration local algebraic map defined
- [x] force, moment and virtual-work gates defined
- [x] GPU-only scientific boundary recorded
- [x] unresolved claim boundary recorded

## Tests first

- [x] synthetic same-panel affine reconstruction and exact block order
- [x] independent force/moment/virtual-work oracle
- [x] real multi-step incremental Ptera packet conversion
- [x] resolved + source-owned LEV composition
- [x] point/packet/source/topology/rank/device/dtype attacks

## Implementation

- [x] immutable panel/support topology
- [x] CUDA float64 batched local affine solve
- [x] deterministic CUDA support-force transpose
- [x] exact Q16 surface transpose composition
- [x] combined resolved + LEV generalized force gate

## Verification

- [x] focused tests pass
- [x] selected Q16/LEV/TEV/FSI joint suite passes
- [x] Black, Ruff, py_compile and whitespace checks pass
- [x] metrics, hashes, runlog, summary and claim validation written
- [x] no paper data, GT/scorer or formal matrix accessed

## Next frontier

- [x] combined load installed inside Newmark predictor/corrector evaluation
- [x] convergence failure leaves Q16 and aero owners unchanged
- [x] accepted step advances structure and LEV/TEV/free wake exactly once
