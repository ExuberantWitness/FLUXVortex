# Q16 LEV Impulse Source Ledger Checklist

## Planning

- [x] mandatory separated LEV / joint TEV / free-wake scope frozen
- [x] last-known-good hashes and regression counts recorded
- [x] global-force comparability and non-smearing rule frozen
- [x] next work-conjugate gate explicitly excluded from this checkpoint

## Tests First

- [x] ring legs retain exact source-strip IDs
- [x] forward/reverse ring traversal matches four adjacent-edge oracle
- [x] source IDs survive convection and removal/reordering
- [x] invalid host/dtype/shape/range source input rejects without mutation
- [x] real two-step strip force closes global impulse force
- [x] source-ledger mutation invalidates incremental state and clean retry works

## Implementation

- [x] persistent CUDA int64 source-strip field
- [x] ring index propagation through active-leg mask
- [x] per-strip free and bound impulse decomposition
- [x] per-strip force and current leading-edge endpoints retained
- [x] new state included in incremental scientific digest
- [x] frozen global impulse result unchanged

## Validation

- [x] focused tests pass: `5/5`, zero warnings
- [x] existing particle/active-LEV tests pass
- [x] incremental owner/geometry/motion tests pass
- [x] bounded Q16 joint regression passes: `146/146`
- [x] Black/Ruff/py_compile/diff-check pass
- [x] artifact hashes and metrics verify

## Closeout

- [x] claim boundary recorded
- [x] next Q16 impulse-work operator specified
