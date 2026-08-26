# Q16 shared-node mesh checklist

## Planning

- [x] Q16-only shared ownership contract frozen
- [x] deterministic CSR assembly selected
- [x] claim boundary and stop conditions recorded

## Implementation

- [x] tests-first RED checkpoint
- [x] exact global node/connectivity schema implemented
- [x] rectangular 1xN and NxM Q16 construction implemented
- [x] CPU energy/force/mass/Jv assembly implemented
- [x] CUDA gather/stable assembly implemented
- [x] host/nonfinite/connectivity attacks fail closed

## Validation

- [x] focused shared-mesh suite passes
- [x] all Q16 + relevant baseline joint suite passes
- [x] Black/Ruff/py_compile/diff-check pass
- [x] hashes, metrics and next action recorded
