# Q16-to-Ptera Trial Kinematics Checklist

## Identity

- run id: `20260821_q16_ptera_trial_kinematics`
- stage: implementation / real GPU pilot

## Planning

- [x] exact two-state contract fixed
- [x] mandatory LEV/TEV/free-wake and impulse boundaries fixed
- [x] red-test and real-pilot route fixed

## Implementation

- [x] exact panel-grid topology owner
- [x] CUDA Q16 position/velocity interpolation
- [x] CUDA W-to-GP frame conversion
- [x] fresh immutable Ptera Panel reconstruction and atomic branch owner swap
- [x] input/runtime/mode/lifecycle gates

## Pilot

- [x] q state appears exactly in current Ptera panel vertices
- [x] `(current-previous)/dt` matches interpolated dq
- [x] real loads change under discriminative dq
- [x] separated LEV, joint TEV and free wake remain active
- [x] active-LEV unresolved work stops without committing the parent

## Validation

- [x] focused tests pass: 7/7
- [x] previous joint surface plus new tests passes: 116/116
- [x] Black/Ruff/py_compile/whitespace pass
- [x] scope and next blocker recorded
