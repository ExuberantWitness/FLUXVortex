# Riziotis & Voutsinas (2008), Figure 12 vector digitization

This directory is a deterministic vector extraction of Figure 12 from:

V. A. Riziotis and S. G. Voutsinas, “Dynamic stall modelling on airfoils
based on strong viscous–inviscid interaction coupling,” *International
Journal for Numerical Methods in Fluids* 56 (2008), 185–208.
[doi:10.1002/fld.1525](https://doi.org/10.1002/fld.1525)

The row schema follows the
[frozen N2.6e1 source-response contract](../n26e1_source_response_contract_20260730.md).

No raster tracing, force fitting, or curve smoothing is used. The script
`platform/digitize_riziotis_fig12.py` reads the publication's
`stroke-width:0.284` vector objects, identifies one 75–79-point published
double-wake polyline per panel, maps through the vector axis frames, and
validates two experimental diamonds at each of the 15 published pressure-tap
stations on both surfaces. The one legend diamond in each panel is excluded.
Ambiguous topology or counts terminate extraction.

The Figure 12 caption defines the source case as NACA0015,
`Re = 1.5e6`, `Ma = 0.12`, mean pitch `alpha0 = 11°`, pitch amplitude
`alpha1 = 8°`, and reduced frequency `k = 0.05`. The 15 pressure-tap
stations resolved from the common vector abscissae are:
`0.00025, 0.0025, 0.01, 0.025, 0.05, 0.1, 0.17, 0.26, 0.37, 0.5, 0.59, 0.7, 0.83, 0.95, 0.98` x/c.
On the upper surface these are taps `X15` through `X1` from leading to
trailing edge; on the lower surface they are `X16` through `X30`.

## Files

- `fig12_digitized.csv`: model curve and experimental pressure taps. It stores
  both `minus_cp` (the plotted ordinate) and `cp`, plus surface, motion phase,
  series, the complete frozen source-contract provenance, and unsnapped
  source-vector coordinates. Experimental `x_over_c` is the Report 9221
  nominal tap coordinate; `vector_x_over_c` retains the extracted coordinate.
- `panel_summary.csv`: counts, frame calibration, pressure-tap residual, and
  persisted-CSV round-trip errors for each panel.
- `roundtrip_overlay.svg`: all eight panels. Black curves are source vectors,
  red samples are coordinates after CSV formatting and inverse mapping, and
  blue diamonds are the extracted experimental taps.

`phase_rad` follows `alpha = 11° + 8° sin(phase_rad)` away from reversal.
It is intentionally empty for both 19° panels: angular velocity is zero at
the maximum angle, so an upstroke/downstroke signed phase is not uniquely
defined at that instant.

The contract uncertainty columns are the preregistered digitization values:
`digitization_sigma_xc=0.002`, and `digitization_sigma_cp=0.10` for
`x/c<0.01` or `0.03` elsewhere. The additional `vector_half_stroke_*`
columns report the smaller mechanical coordinate-resolution estimate from
half the published vector stroke. Neither quantity is experimental pressure
uncertainty.

## Provenance

- Audited source PDF SHA-256: `cc4970b38b3586affc4805a84e526fcb0049ba2dfa42219c01379e2a8f48fa84`
- Derived/source SVG page 17 SHA-256: `cf1bb762c4e7c6cc794420c01fbbe06c2a5df5819ef7d891886a4150e3b824b1`
- Derived/source SVG page 18 SHA-256: `50f241446fb1ecf4253619e39e608ff1b727663e5e3c144b5946a5a7801c8dd0`
- SVG generator used for PDF input: `pdftocairo -svg`, one page per call.
- Maximum absolute tap-location residual from the 15 nominal stations:
  `0.000108361` x/c.
- Maximum persisted-value round-trip error:
  `8.06e-11` SVG x units and `7.71e-11` SVG y units.
- Extracted rows: `859` total =
  `619` model +
  `240` experiment.

Reproduce from a legally obtained source PDF:

```bash
python platform/digitize_riziotis_fig12.py /path/to/riziotis2008_full.pdf
```

Or from the two vector pages in either order:

```bash
python platform/digitize_riziotis_fig12.py page17.svg page18.svg
```
