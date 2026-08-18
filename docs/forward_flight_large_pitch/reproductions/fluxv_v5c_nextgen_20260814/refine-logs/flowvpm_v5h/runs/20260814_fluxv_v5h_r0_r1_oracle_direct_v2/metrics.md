# v5h R0–R1 schema-v2 parity metrics

| Gate | Frozen limit | Result | Status |
|---|---:|---:|---|
| Full direct velocity relative L2 | `1e-12` | `1.3149583542588946e-16` | PASS |
| Full direct Jacobian relative L2 | `1e-11` | `8.135639459129776e-17` | PASS |
| Independent-probe velocity relative L2 | `1e-12` | `2.0417328424520753e-16` | PASS |
| Independent-probe Jacobian relative L2 | `1e-11` | `8.781466983618569e-17` | PASS |
| Nearfield probe velocity row relative L2 max | `1e-9` | `1.9005804868259926e-15` | PASS |
| Nearfield probe Jacobian row relative L2 max | `1e-9` | `2.9430882510679194e-15` | PASS |
| Nearfield geometry `r/sigma` absolute error | `1e-14` | `1.1102230246251565e-16` | PASS |
| Fixed-Uinf RK state/storage worst relative L2 | `1e-11` | `2.455653689598918e-16` | PASS |
| Fixed-Uinf RK RHS worst relative L2 | `1e-11` | `2.437361909695629e-16` | PASS |
| Affine step-time Uinf RK state/RHS worst relative L2 | `1e-11` | `0` | PASS |
| RK clock/freestream contract error | `1e-15` | `0` | PASS |
| Corrected Pedrizzetti Gamma/norm error | `1e-12 / 1e-14` | `0 / 0` | PASS |
| Schema/config contract | all true | all true | PASS |
| Nonfinite / clip count | `0 / 0` | `0 / 0` | PASS |

The nearfield fixture contains nine zero-strength probes from `r/sigma=1e-4` through `2.0`. State-only `pre/post` nodes contain no U/J; only each stage's `rhs` node contains a field evaluated at the stage-pre state.

Upstream pinned suite: 14/14 testsets passed. Local schema/config/numerical suite: 29/29 tests passed. This remains transport-operator evidence only.
