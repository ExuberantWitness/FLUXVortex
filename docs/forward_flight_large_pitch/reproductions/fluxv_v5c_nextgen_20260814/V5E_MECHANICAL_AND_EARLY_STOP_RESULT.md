# FluxV v5e line-item replacement result

## Decision

The ULLT-to-UVLM line-item shadow passed its mechanical contract but failed the
pre-registered representative accuracy screen.  It is therefore archived as a
mechanically valid negative result and is not run over all 22 development
conditions.

Run: `runs/20260814_fluxv_v5e_mechanical_smoke`

Status: `mechanical_gates_passed; accuracy_early_stop_no_go`.

No coefficient, pole, added-mass factor, phase, offset, or branch threshold was
selected from the three experimental data sets after this result.

## Mechanical result

The disabled path replayed the parent FluxV loads exactly.  The panel, strip
and airplane ledgers closed, and the enabled force satisfied

`F_new = F_KJ + Delta F_(phi-Gamma) + F_AM,kin`

after removing the original `F_dGamma` in full.  The one-state history closed
over its periodic warm-up.  Forty-two targeted tests passed; all 9 declared
source hashes and all 3 result hashes match the frozen files.

Independent CSV recomputation gave the following maximum residuals:

| Identity | Maximum absolute residual |
|---|---:|
| `F_old-F_KJ-F_dGamma` | `5.684e-14 N` |
| `F_new-F_KJ-DeltaF-F_AM` | `7.816e-14 N` |
| replacement ledger identity | `4.263e-14 N` |
| state recursion | `5.55e-17` |

This establishes bookkeeping and implementation consistency only.  The smoke
uses low-resolution representative movements and is not a full-paper accuracy
result.

## Representative accuracy early stop

The line-item increment was added to each frozen, same-ledger baseline.  In
Figure 14 this means the corrected v5c0 baseline, which already contains the
published `Cd0=0.057` exactly once; profile drag was not added again.

| Condition/channel | Experiment | Frozen baseline | v5e representative proxy | Absolute error change | Decision |
|---|---:|---:|---:|---:|---|
| Yang AoA 15 deg lift, gf | 38.7 | 34.183114 | 34.397540 | 4.516886 -> 4.302460 | improve |
| Yang AoA 15 deg drag, gf | 14.1 | 9.074158 | 8.597739 | 5.025842 -> 5.502261 | **regress** |
| Figure 14 theta=15 deg, psi=60 deg, CT | 0.230641 | 0.218594 | 0.373088 | 0.012047 -> 0.142447 | **severe regress** |
| Baik W2 filtered CL RMSE | -- | 1.032306 | 1.542894 | +49.5% | **regress** |
| Baik W2 filtered CD RMSE | -- | 0.725678 | 0.589109 | -18.8% | improve |

The Figure-14 line-item increment is `Delta CT=+0.154494`, about 28 times the
smoke-to-full change in the original UVLM result.  Even restricting the
increment to the existing transient branch leaves an absolute error near
`0.140`.  This provides ample margin for the early-stop decision.

## Why the candidate fails

The failure is structural rather than an added-mass calibration issue:

- in Figure 14, removing the original `F_dGamma` and adding the
  phi--Gamma mismatch each contribute roughly half of the excessive thrust;
- in Yang, the drag regression is almost entirely the phi--Gamma term;
- in Baik W2, the improved drag is produced by cancellation between two large
  terms, while removal of `F_dGamma` substantially reduces lift;
- the mean kinematic added-mass contribution is negligible in all three
  representative cases.

Consequently, changing the AR interpolation of `K_AM` cannot repair this
candidate.  The result also rejects only this particular KJ-retaining,
`F_dGamma`-replacement mapping; it does not prove that every ULLT/UVLM coupling
is invalid.

## Next route

Do not tune or partially scale the line-item correction on these observations.
The next candidate must place a material LEV in the same UVLM AIC, wake and
pressure ledger while preserving exact parent behavior whenever the LEV state
is pristine.  It must pass the no-LEV exact-reduction gate that blocked the
standalone v5b model before any paper scoring is allowed.

