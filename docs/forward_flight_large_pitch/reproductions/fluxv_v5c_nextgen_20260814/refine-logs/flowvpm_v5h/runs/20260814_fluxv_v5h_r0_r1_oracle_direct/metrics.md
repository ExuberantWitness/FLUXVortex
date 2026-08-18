# v5h R0–R1 parity metrics

| Gate | Frozen limit | Result | Status |
|---|---:|---:|---|
| Direct velocity relative L2 | `1e-12` | `1.3149583542588946e-16` | PASS |
| Direct Jacobian relative L2 | `1e-11` | `8.135639459129776e-17` | PASS |
| RK position relative L2, worst stage | `1e-11` | `4.422880509680879e-19` | PASS |
| RK Gamma relative L2, worst stage | `1e-11` | `0` | PASS |
| RK sigma relative L2, worst stage | `1e-11` | `0` | PASS |
| RK low-storage state relative L2 | `1e-11` | `2.455653689598918e-16` | PASS |
| RK RHS velocity relative L2 | `1e-11` | `1.4344867445833976e-16` | PASS |
| RK RHS Jacobian relative L2 | `1e-11` | `2.437361909695629e-16` | PASS |
| Corrected Pedrizzetti Gamma parity | `1e-12` | `0` | PASS |
| Corrected Pedrizzetti norm error | `1e-14` | `0` | PASS |
| Nonfinite count | `0` | `0` | PASS |
| Clip/replacement count | `0` | `0` | PASS |

Additional evidence:

- Pinned FLOWVPM official suite: 14/14 testsets passed (10 single-ring and 4 leapfrog variants).
- Local Python suite: 24/24 tests passed.
- A fresh current-code metric evaluation was byte-identical to `metrics.json`.
- A fresh same-basename Julia export produced a byte-identical JSON mirror; `h5diff` found no semantic HDF5 differences.

These metrics establish direct numerical transport parity only. They do not validate a TE/LEV release bridge, Ptera coupling, force prediction, or experimental accuracy.
