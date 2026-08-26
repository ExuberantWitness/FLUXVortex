# Q16 Incremental Aerodynamic Owner Checklist

## Planning

- [x] baseline and monolithic oracle frozen
- [x] mandatory separated-LEV/joint-TEV/free-wake mode frozen
- [x] impulse and structural-kinematics non-claims frozen

## Tests First

- [x] lifecycle red tests
- [x] per-step parity red test
- [x] pickle-branch isolation red test
- [x] reduced-mode and mutation rejection red tests

## Implementation

- [x] exact wake-array preallocation
- [x] per-step Panel ring initialization
- [x] causal pending-wake completion on arrival of candidate geometry
- [x] exact existing numerical call order
- [x] sealed per-step receipt
- [x] exact finalize gate

## Validation

- [x] focused tests pass: 7/7
- [x] previous 116-test surface plus new tests passes: 123/123
- [x] Black/Ruff/py_compile/whitespace pass
- [x] durable metrics and next blocker recorded
