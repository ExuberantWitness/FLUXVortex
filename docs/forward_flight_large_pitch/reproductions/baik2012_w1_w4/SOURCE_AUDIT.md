# Baik 2012 W1--W4 source and reconstruction audit

## Source identity

The force ground truth is frozen from Yeon Sik Baik's 2011 University of
Michigan dissertation, *Unsteady Force Generation and Vortex Dynamics of
Pitching and Plunging Airfoils at Low Reynolds Number*.  The Deep Blue item is
`e1cec17c-27e7-46c6-8956-704134cb257a`; bitstream UUID
`1ef3aed8-ca97-48ec-b992-85acc3be814a`.  The 259-page PDF has SHA-256
`2efbf3becd339df61cc9275e2e933700ef75216504580d5f4e5cca1e80eadc0a`.

The associated journal paper is Y. S. Baik, L. P. Bernal, K. Granlund and
M. V. Ol, “Unsteady force generation and vortex dynamics of pitching and
plunging aerofoils,” *Journal of Fluid Mechanics* 709 (2012), 37--68,
DOI `10.1017/jfm.2012.318`.  The paywalled journal PDF was not used to invent
missing data.

The scored histories are dissertation Figures 5.24--5.27 (physical PDF pages
204--207, printed pages 179--182).  They are direct-force, corrected-total
histories.  The earlier AIAA precursor used a pre-trigger-relative datum,
omitted W4, and must not be substituted.

## Frozen geometry and boundary condition

- chord: 0.076 m;
- span: 0.600 m;
- thickness ratio: 6.25%, or 0.00475 m;
- semicircular leading- and trailing-edge radii: 0.002375 m;
- pitch axis: quarter chord;
- Reynolds number: 5000;
- channel width about 0.61 m, approximately 1 mm bottom gap, and a free-surface
  endplate.

The apparatus is therefore a wall-to-wall/end-plated quasi-two-dimensional
experiment.  It is **not** a freely shedding finite wing with aspect ratio
7.895.  Ptera currently has no wall-image/endplate boundary, so the retained
UVLM path uses the physical 600 mm span as a clearly labelled free-tip adapter.
It also represents only the zero-camber mean surface and does not resolve the
physical 6.25% thickness, rounded edges or viscous boundary layers.

## W1--W4 matrix

| Case | k | h0/c | printed St | printed pitch amplitude | period |
|---|---:|---:|---:|---:|---:|
| W1 | 0.5 | 0.50 | 0.16 | 13.16 deg | 7.13 s |
| W2 | 1.0 | 0.50 | 0.32 | 33.73 deg | 3.56 s |
| W3 | 1.0 | 0.25 | 0.16 | 13.16 deg | 3.56 s |
| W4 | 0.5 | 1.00 | 0.32 | 33.73 deg | 7.13 s |

The AIAA table's W3 `k=0.5` is a typo.  Its period, the surrounding text, and
the dissertation all require `k=1.0`.  The tabulated Strouhal numbers are
rounded: the internally consistent values from `St=2 k h0/pi` are 0.159155 and
0.318310.  Freestream is computed from `U=pi f c/k`, about 0.067 m/s; mixing
the measured period with the approximate prose value 0.06 m/s would violate
the stated reduced frequency.

## Non-harmonic motion

With `tau=t/T`, the source prescribes

```text
h_dot/U = -tan(alpha_pl,max sin(2 pi tau))
alpha_g = 8 deg - theta0 sin(2 pi tau)
alpha_eff = 8 deg + 14 deg sin(2 pi tau)
```

The plunge displacement is the periodic integral of the first equation, not a
sinusoid.  Solving the quarter-stroke displacement constraint gives
`alpha_pl,max=27.182110 deg` for W1/W3 and `47.755954 deg` for W2/W4.  The
implemented pitch amplitudes are consequently 13.182110 and 33.755954 degrees;
the 0.02--0.03 degree difference from the table is rounding, not a load fit.
Phase zero is the upper plunge limit and the first half-cycle is downstroke.
At phases 0, 0.25, 0.5 and 0.75, effective incidence is exactly 8, 22, 8 and
-6 degrees.

## Force datum and filtering

The source coordinate conversion is

```text
CL = Fx* sin(alpha_g) + Fy* cos(alpha_g)
CD = -Fx* cos(alpha_g) + Fy* sin(alpha_g)
```

Negative `CD` therefore means thrust.  The experiment sampled at 2000 Hz,
removed in-air inertial tare, removed pre-trigger sensor bias, restored the
separately measured steady hydrodynamic force, discarded five cycles at each
end, phase averaged about 500 realizations, and applied a sharp 1 Hz Fourier
low-pass.  Primary model scoring applies the same ideal cutoff: harmonics 0--7
for W1/W4 and 0--3 for W2/W3.  Raw numerical histories remain diagnostics.

The source reports approximately +/-0.02 coefficient uncertainty.  Raster
digitization uncertainty is separate and documented per panel in
`source_data/DIGITIZATION_AND_PROVENANCE.md`.

## Model-comparison contract

- no phase shift, amplitude multiplier or offset correction is fitted;
- the duplicate phase-1 endpoint is excluded from metrics, leaving 400 unique
  equally weighted phase samples per case;
- metrics are reported per case before equal-case macro aggregation;
- published standard-Theodorsen open-circle markers from Figures 5.28--5.31
  are a secondary lift-only reference;
- FluxV v4b's primary `Lcrit=0.11` is transferred from the detailed Ramesh
  flat-plate Re=1000 text and plots, not calibrated to Baik;
- Ramesh Table 4.1 prints 0.19 for the same flat-plate cases.  This source
  conflict is retained as a labelled sensitivity, not selected by looking at
  W1--W4 accuracy;
- the Baik plate is thicker and at a different Reynolds number, so neither
  LESP value is a validated Baik parameter;
- W1--W4 are development-transfer tests, not held-out confirmation.
