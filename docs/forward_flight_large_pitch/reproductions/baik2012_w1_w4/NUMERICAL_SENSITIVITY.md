# Baik W2 numerical sensitivity

The production reference is UVLM `(4 chord, 8 span, 128 steps/cycle,
3 cycles, 2 retained wake cycles)` plus LDVM `(512 steps/cycle,
256 retained material-wake steps, ndiv=32, naterm=14, rc/c=0.02)`.
All reported RMSE values below use the same W2 experimental history and the
same source-matched 1 Hz Fourier cutoff.

## UVLM one-factor checks

| Change from reference | old CL RMSE | v4b CL RMSE | old CD RMSE | v4b CD RMSE |
|---|---:|---:|---:|---:|
| 64 instead of 128 steps/cycle | 1.10110 | 1.07088 | 0.72805 | 0.68435 |
| reference | 1.07838 | 1.03231 | 0.78905 | 0.72568 |
| 12 instead of 8 span panels | 1.08510 | 1.03820 | 0.78981 | 0.72826 |
| 4 instead of 3 simulated cycles | 1.07762 | 1.03152 | 0.78903 | 0.72546 |

The 8-to-12 span change is small: v4b CL/CD RMSE changes by +0.00589/+0.00258.
The 3-to-4-cycle change is smaller still: -0.00078/-0.00022.  The 64-to-128
time change is larger and non-monotonic in accuracy because it changes the
resolved UVLM force history, not merely post-processing.  It changes v4b
CL/CD RMSE by -0.03858/+0.04133.

At the reference discretization, old-to-v4b improves W2 CL/CD RMSE by
0.04607/0.06337.  The CL improvement is comparable to the 64-to-128 temporal
shift, while the CD improvement is about 1.53 times that shift.  Therefore the
W2 direction is supported at the reference resolution, but the size of the
improvement is not fully time-converged.

## Controlled LDVM time and wake checks

For the LDVM time check, retained material-wake duration is kept at half a
cycle and `rc/c=0.02` is fixed:

| LDVM steps/cycle / wake steps | CL RMSE | CD RMSE |
|---|---:|---:|
| 256 / 128 | 1.02863 | 0.73844 |
| 512 / 256 reference | 1.03231 | 0.72568 |
| 1024 / 512 | 1.03776 | 0.72976 |

Reference-to-1024 changes are +0.00546 CL and +0.00408 CD, much smaller than
the production old-to-v4b differences 0.04607 and 0.06337.  The LDVM temporal
increment is therefore stable enough for the direction of the W2 comparison,
although the three points are not strictly monotonic and are not a formal
order-of-accuracy study.

At fixed 512 steps/cycle, material-wake retention remains important:

| retained material wake | CL RMSE | CD RMSE |
|---|---:|---:|
| 0.25 cycle | 1.04068 | 0.77081 |
| 0.50 cycle reference | 1.03231 | 0.72568 |
| 0.75 cycle | 0.96989 | 0.70030 |

Extending the retained LDVM wake from 0.50 to 0.75 cycle changes CL/CD RMSE by
-0.06241/-0.02537.  The CL shift is larger than the reference old-to-v4b CL
gain, so the LDVM wake is **not converged**.  The 0.75-cycle result happens to
fit the observed W2 lift better, but it must not be selected on that basis.

## Supported numerical statement

- span and simulated-cycle sensitivity at the reference grid are small;
- controlled LDVM time refinement is small compared with the old-to-v4b
  differences;
- UVLM 64-to-128 temporal sensitivity is material;
- LDVM material-wake retention is material and not converged;
- consequently, the full-run observation that v4b improves every W1--W4
  channel is valid for the frozen production settings, but the precise
  improvement percentages are provisional and not mesh/time/wake-converged.

Raw data and manifests:

- `sensitivity/20260813_w2_one_factor_reproducible/`
- `sensitivity/20260813_w2_ldvm_controlled_reproducible/`

The earlier directories without the `_reproducible` suffix are deprecated
development artifacts.  They reproduce the same numerical values, but their
manifests predate the final dependency/environment provenance collector.
