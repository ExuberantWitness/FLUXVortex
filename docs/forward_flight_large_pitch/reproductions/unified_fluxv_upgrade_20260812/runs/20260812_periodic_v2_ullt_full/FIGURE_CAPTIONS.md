# Figure captions and interpretation limits

Across all local-model figures, plotted drag follows `D=-T=-Fx_W`: positive
drag is resistance and negative drag denotes net thrust.

## yang2025_mean_lift_drag_vs_aoa

Cycle-mean single-wing lift and drag versus installation angle for the Yang
2025 rigid-wing wind-tunnel cases.  The wind-tunnel and authors' modified-UVLM
points are vector/raster-digitized cycle means.  The assigned ±0.4 gf source
uncertainty is digitization uncertainty only, not an experimental error bar,
and it is not drawn as an error bar.  FluxV v2 is exploratory periodic
two-pass output.

## yang2025_15deg_phase_lift_drag

Local-model diagnostic at installation angle `alpha_0=15 deg`.  Yang et al. did not publish
phase-resolved test or proposed-model loads, so this figure compares predicted
shape only and is not a phase-accuracy validation.  Shared-gate transition
features are not smoothed or hidden; the movement derivative is closed on the
selected coherent cycle rather than across a duplicated movement endpoint.

## izraelevitz2017_fig11_lift_drag_phase

Paper-scaled lift and drag for Figure 11.  Author curves were recovered from
the source PDF vector paths without amplitude fitting or phase alignment.  The
authors' UVLM is a numerical reference rather than experimental ground truth.
The dimensionless scaling is `C_Lalpha=C_L/sin(15 deg)` and
`C_Dalpha=C_D/sin(15 deg)`; negative `C_Dalpha` denotes net thrust.

## crosspaper_accuracy_summary

Yang bars are six-angle MAE in gf against wind-tunnel cycle means.
Izraelevitz bars are raw-phase RMSE in paper-scaled coefficients against the
authors' UVLM; no optimal cyclic phase shift is applied.  Units therefore
differ between panels and values must not be pooled into a single score.
