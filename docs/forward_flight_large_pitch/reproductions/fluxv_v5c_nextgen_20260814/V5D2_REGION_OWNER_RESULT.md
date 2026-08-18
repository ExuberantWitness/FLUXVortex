# FluxV v5d2 source-region owner result

## Decision

The frozen non-canonical all-22 shadow is rejected by the preregistered
cross-paper no-regression gate.  No `C_alpha`, separation angle, phase,
amplitude, offset, or branch coefficient was tuned after the result.

Run: `runs/20260814_fluxv_v5d2_region_owner_all22`

Status: `stopped_region_owner_crosspaper_gate_failure`.

## Mechanical result

The A/L/P weights were finite, bounded and summed to one; disabled mode
returned the attached branch exactly.  These checks validate the region-owner
bookkeeping only.  The adapter used kinematic `0.75c` incidence and integrated
branch loads, so every row remains non-canonical.

## Accuracy result

| Benchmark | Reference | v5d2 | Decision |
|---|---:|---:|---|
| Yang lift MAE, gf | 4.554510 | 8.709806 | FAIL |
| Yang drag MAE, gf | 2.643997 | 2.887963 | FAIL |
| Figure 14 all-14 CT RMSE | 0.024251 | 0.027769 | FAIL |
| Figure 14 15-degree RMSE | 0.020759 | 0.024509 | FAIL |
| Figure 14 25-degree RMSE | 0.029512 | 0.032833 | FAIL |
| Figure 14 unique-12 RMSE | 0.025700 | 0.029594 | FAIL |
| Baik macro CL RMSE | 0.657542 | 0.573853 | PASS |
| Baik macro CD RMSE | 0.345152 | 0.283350 | PASS |

Baik W1, W2 and W4 improved in both channels.  W3 drag improved, but W3 lift
regressed from `0.374319` to `0.384613`, so Baik also fails its strict
per-channel gate.

## Interpretation

The Yang A/L/P region taxonomy is useful descriptive source evidence, but an
area-weighted selector over three integrated load vertices is not a unified
force model.  Its failure also shows that simply routing more Yang phases to
the full-angle branch does not repair the retained UVLM/LDVM pressure ledger.
The next candidate therefore changes an explicit UVLM line item instead of
reweighting these existing total-load branches.

