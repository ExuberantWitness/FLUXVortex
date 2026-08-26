# Q16 Real FSI Data Path Checklist

## Identity

- run id: `20260821_q16_real_fsi_data_path`
- idea id: `q16-resolved-load-transfer-with-unresolved-impulse-gate`
- stage: implementation / integration pilot

## Planning

- [x] selected idea and scientific ownership boundary stated
- [x] baseline and comparability contract confirmed
- [x] code touchpoints listed
- [x] bounded real-solver smoke plan written
- [x] joint regression plan written
- [x] fail-closed fallback written

## Implementation

- [x] rigid force/moment transform red test added
- [x] exact CUDA load packet and mutation seal added
- [x] resolved Q16 transpose path added
- [x] non-zero unresolved impulse blocks completed transfer
- [x] real solver retains exact resolved points/forces without recomputation
- [x] unrelated files remain untouched

## Pilot / Smoke

- [x] synthetic virtual-work/force/moment oracle passes
- [x] real one-step packet reaches Q16 generalized coordinates
- [x] real two-step LEV/TEV packet closes force and moment
- [x] parent solver transaction behavior remains exact
- [x] no paper/full-matrix run is triggered

## Validation

- [x] focused tests pass
- [x] previous joint surface plus new tests passes: 109/109
- [x] Black, Ruff, py_compile and whitespace gates pass
- [x] claims classified as supported / blocked
- [x] results and next frontier recorded durably

## Closeout

- [x] summarize what is now a real data path
- [x] state why full FSI commit remains blocked or is enabled
- [x] identify the exact next scientific implementation step
