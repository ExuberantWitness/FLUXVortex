# Q16 review bug-fix checklist

- [x] Review findings and baseline scope frozen.
- [x] RED: shared two-element transfer exposes `192 != 168`.
- [x] GREEN: transfer is keyed by shared `Q16Mesh` and returns 168 DOFs.
- [x] Virtual-work/force/moment/shared-edge regression passes on 2x1 and 2x2.
- [x] Explicit stable-node root owner is rotation invariant.
- [x] Per-element homogenized stiffness/mass ownership is available for Q16 macro design.
- [x] Mandatory separated-LEV/joint-TEV/free-wake configuration rejects all off modes.
- [x] Real solver transaction branches the complete serialized CUDA/Ptera owner graph.
- [x] Ordinary `deepcopy` loss of Ptera panel-vortex state is covered by a real-run regression.
- [x] Predictor failure, hostile parent mutation and clean retry tests pass.
- [x] Existing Q16 focused suite passes: 102/102 in the registered joint command.
- [x] Black/Ruff/py_compile checks pass on the 13 touched source/test files.
- [x] Final scope and unresolved blockers recorded in `RESULTS.md`.
