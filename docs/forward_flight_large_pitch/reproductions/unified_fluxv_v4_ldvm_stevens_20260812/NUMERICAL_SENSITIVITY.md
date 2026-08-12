# v4b numerical sensitivity

## Frozen comparison

This check changes only the LDVM temporal resolution from 256 to 512 samples
per cycle. The strip count remains 12 and the material-vortex core remains
fixed at `r_core/c=0.02`. This corrects the earlier development scan in which
the core size was changed together with the time step.

| Metric | 256 samples/cycle | 512 samples/cycle | Relative change |
|---|---:|---:|---:|
| Yang lift MAE [gf] | 4.554510 | 4.581258 | +0.587% |
| Yang drag MAE [gf] | 2.643997 | 2.653210 | +0.348% |
| Figure 14 mean-thrust RMSE | 0.025949 | 0.025694 | -0.982% |

Across individual Yang conditions, the largest 256-to-512 difference is
`0.2282 gf` in lift and `0.0513 gf` in drag. Across the 12 unique Figure-14
conditions, the largest prediction change is `0.01142` in mean thrust
coefficient. The 512-sample run uses the same source hashes as the 256-sample
run and records its own result hashes in `20260812_fluxv_v4b_crosspaper_spc512`.

## Interpretation boundary

The fixed-core temporal result is stable at the aggregate-metric level. It is
not a complete convergence proof: the Fourier resolution, strip count, UVLM
grid, wake retention, and vortex-core model were not independently refined in
this final comparison. Earlier variable-core development runs are not used as
headline evidence because they confounded numerical resolution with a physical
regularization parameter.
