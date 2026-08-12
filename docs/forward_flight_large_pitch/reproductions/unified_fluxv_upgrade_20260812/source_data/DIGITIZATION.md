# Izraelevitz et al. (2017) Figure 11(a) vector-curve recovery

## Source and scope

- Paper: J. S. Izraelevitz, Q. Zhu, and M. S. Triantafyllou, *State-Space Adaptation of Unsteady Lifting Line Theory: Twisting/Flapping Wings of Finite Span* (2017).
- Local source PDF: `FLUXV/docs/forward_flight_large_pitch/literature/candidates_20260807/izraelevitz_zhu_triantafyllou_2017_state_space_ullt.pdf`.
- Source SHA-256: `68d1a5fca17479eb857d327b632ab6762a7cf6b363633a9157d5b50400077304`.
- Source location: PDF page 11 (printed page 11), Figure 11(a), “Force and Mid-span Circulation.”
- The reference is the authors' numerical UVLM curve, not experimental data.
- This file recovers the plotted force curves only. The orange mid-span circulation curves in Figure 11(a) and the spanwise distributions in Figure 11(b) are outside this dataset.

## Output schema

`izraelevitz2017_fig11_digitized.csv` contains 128 uniformly spaced samples over the final plotted period, $2 \le t/T < 3$, shifted to `phase = t/T - 2`:

- `paper_uvlm_CLalpha`, `paper_uvlm_CDalpha`: authors' UVLM reference.
- `6_state_CLalpha`, `6_state_CDalpha`: authors' six-state state-space ULLT.
- `1_state_CLalpha`, `1_state_CDalpha`: authors' one-state state-space ULLT.
- `qs_added_mass_CLalpha`, `qs_added_mass_CDalpha`: quasi-steady plus added-mass model.

The coefficient names reproduce the paper axes, $C_{L\alpha}$ and $C_{D\alpha}$. For this case the paper normalizes forces by $\sin \alpha_{\max}$, with $\alpha_{\max}=15^\circ$.

## Extraction method

1. Poppler `pdftocairo 22.02.0` converted only PDF page 11 to SVG:

   ```bash
   pdftocairo -f 11 -l 11 -svg \
     FLUXV/docs/forward_flight_large_pitch/literature/candidates_20260807/izraelevitz_zhu_triantafyllou_2017_state_space_ullt.pdf \
     /tmp/izraelevitz2017_page11.svg
   ```

2. Curve centerlines were read from SVG `path` elements before rasterization. The raw SVG path coordinates were calibrated by least-squares fits to all seven visible ticks on each axis:

   ```text
   x_raw = 1209.088877142857 * (t/T) + 358.2968670000005
   y_raw =  167.745752500000 * coefficient + 1491.433742857143
   ```

3. The four models were distinguished by the paper's SVG line/marker encoding and drawing order:

   - UVLM: solid, width 20.
   - 6-State: thin dotted connector, `x` markers; first such curve.
   - 1-State: thin dotted connector, circle markers; second such curve.
   - QS,AM: long dashed, width 20.

   Lift is the purple path family and drag is the light-orange path family.

4. The complete final plotted cycle was selected to avoid the legend obscuring part of the first cycle. Native vector vertices were sorted by phase, duplicate vertices were averaged, and piecewise-linear interpolation generated the common 128-point grid. No smoothing, phase shift, amplitude fit, or model calibration was applied.

## Integrity checks

- CSV shape: 128 data rows, 9 columns.
- Phase grid: starts at `0.00000000`, ends at `0.99218750`, and has constant spacing `1/128`.
- Sampled extrema and means:

| model | CLalpha min | CLalpha max | mean CLalpha | CDalpha min | CDalpha max | mean CDalpha |
|---|---:|---:|---:|---:|---:|---:|
| paper UVLM | -5.68637 | 5.68807 | 0.00250 | -3.49709 | 0.06521 | -1.62152 |
| 6-State | -5.84935 | 5.85995 | 0.00279 | -3.78329 | -0.20660 | -1.90097 |
| 1-State | -5.74942 | 5.75962 | 0.00093 | -3.66704 | -0.20660 | -1.84003 |
| QS + added mass | -6.72494 | 6.72612 | 0.00030 | -4.75621 | -0.13505 | -2.41526 |

## Error limits and permitted use

This is a recovery of plotted centerlines, not the authors' raw solver output. It is appropriate for curve overlays and plot-level RMSE, peak, mean, and phase comparisons, but not for claiming more precision than the figure supports.

- The least-squares tick calibration has maximum residuals of approximately `0.0011 T` horizontally and `0.0064` coefficient units vertically.
- The thick UVLM/QS trace half-width corresponds conservatively to about `±0.009 T` and `±0.060` coefficient units; the thin state-space traces correspond to about half those values.
- PDF vector centerlines avoid raster pixel error, but publication rendering, line overlap, and interpolation remain.
- Do not silently optimize phase before computing the primary accuracy metric. If a cyclic phase-aligned diagnostic is reported, label it separately and retain the raw-phase metric.
- Do not describe agreement with `paper_uvlm` as experimental validation.
