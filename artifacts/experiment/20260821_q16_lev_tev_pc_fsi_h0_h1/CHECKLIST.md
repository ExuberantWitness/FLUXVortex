# Q16 + LEV/TEV/free-wake PC-FSI H0/H1 checklist

## Identity

- run id: `20260821_q16_lev_tev_pc_fsi_h0_h1`
- branch: `run/q16-lev-tev-pc-fsi-20260821`
- stage: `auxiliary/dev`

## Planning

- [x] Q16-only idea and claim boundary recorded
- [x] baseline commit and read-only comparability boundary recorded
- [x] code touchpoints listed
- [x] smoke and joint pilot commands written
- [x] fallback and STOP conditions written

## Implementation

- [x] exact Q16 node and 96-DOF schema tests written
- [x] Q16 cubic basis and derivative tests RED on absent implementation
- [x] transaction tests RED on absent implementation
- [x] fixed Q16 reference implementation added
- [x] immutable snapshot/trial/commit/abort protocol added
- [x] stale, double, foreign and failed proposals fail closed
- [x] no Q9 files/configurations/tests added
- [x] unrelated baseline files remain unchanged

## Pilot / smoke

- [x] focused tests pass: 25/25
- [x] existing V5M FSI contract remains green
- [x] existing active-LEV CUDA test remains green
- [x] joint focused/baseline suite passes: 36/36
- [x] Black/Ruff/py_compile/diff-check pass

## Validation

- [x] exact source/test hashes recorded
- [x] result classified as partial/supported for H0/H1 only
- [x] remaining Q16 residual/Jv/GPU and real solver adapter gaps stated

## Closeout

- [x] checkpoint summary written
- [x] next action explicitly selected
