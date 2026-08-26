# Q16 work-conjugate CUDA transfer checklist

- [x] plan and claim boundary frozen
- [x] tests written and RED on absent implementation
- [x] immutable multi-element surface map implemented
- [x] CPU-independent oracle virtual-work/force/moment gates pass
- [x] CUDA float64 interpolation and transpose transfer pass
- [x] host/wrong-dtype/wrong-shape/nonfinite inputs fail closed
- [x] deterministic per-DOF CSR gather replaces atomic load scatter
- [x] no Q9 or equal-four-node production path added
- [x] focused 9/9 and relevant joint 45/45 suites pass
- [x] Black/Ruff/py_compile/diff-check pass
- [x] hashes and remaining gaps recorded
