# H0/H1 metrics

| Metric | Result |
|---|---:|
| Focused Q16 + transaction tests | 25/25 PASS |
| Joint Q16 + transaction + V5M FSI + active LEV tests | 36/36 PASS |
| Q16 nodes / DOF per node / DOF per element | 16 / 6 / 96 |
| Black / Ruff / py_compile / diff-check | PASS |
| New Q9 implementation or test | none |
| Full Q16 CUDA FSI claim | not supported yet |

The joint command required `PYTHONPATH=src:platform:platform/warp_vpm`; the first
attempt omitted the final path and stopped during collection without executing
the joint tests.  The corrected command passed all 36 tests.
