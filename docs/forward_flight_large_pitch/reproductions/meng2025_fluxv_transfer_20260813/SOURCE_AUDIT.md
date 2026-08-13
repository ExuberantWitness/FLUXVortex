# Meng 2025 source and observable audit

Date: 2026-08-13

Paper: Rui Meng, Bifeng Song, Jianlin Xuan and Yugang Zhang, *Design and Experimental Study of a Flapping–Twist Coupled Biomimetic Flapping-Wing Mechanism*, Drones 9 (2025) 535.

Local source: `researchpaper/Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf`

SHA-256: `eccaf750707a693fd58c0e38476a2b8ce2c694bfbf40b910f3bdf10017aa0a66`

## Executable reconstruction contract

| Item | Reconstructed value | Evidence status |
|---|---:|---|
| One-wing semispan | 800 mm | printed in Figure 11(b) |
| Root chord | 287 mm | printed in Figure 11(b) |
| Inner planform length | 340 mm | printed in Figure 11; used as the constant-chord break |
| Outer planform | rounded trailing edge, straight leading edge | quarter-ellipse adapter digitized from Figure 11(b); no coordinates published |
| Main-spar axis | 0.27 root chord from leading edge | drawing digitization; no numerical coordinate published |
| Section/thickness | custom zero-camber numerically thin surface | adapter only; the paper publishes neither an airfoil nor thickness |
| Flap amplitude | 45 deg peak-to-peak = +/-22.5 deg | Sections 2.2--2.3 and Figure 10 |
| Twist amplitude | tabulated value is peak-to-peak = `2 beta` | Section 2.2, Equation context, nomenclature and Figure 10 |
| Phase | twist is zero at stroke limits and peaks at midstroke | Section 2.1 and Figure 10 |
| Sign | downstroke is nose-down; upstroke is nose-up | Sections 2.2 and 4.2 |
| Figure-16 condition | U=8 m/s, alpha=5 deg, f=2 Hz; twist=0/22.5/45 deg peak-to-peak | Section 4.2 |
| Air density | 1.225 kg/m3 | assumption; density, pressure and temperature are not reported |

The nominal implementation fixes phase zero at the upper stroke limit:

```text
phi(t)   = +(45 deg / 2) cos(2 pi t/T)
theta(t) = -(twist_pp / 2) sin(2 pi t/T)
```

Positive `theta` is aerodynamic nose-up, so the negative sign implements the
paper's downstroke nose-down convention.  The mechanism is rigid and the
paper defines the motion as rotation about the main spar.  There is no
published spanwise twist distribution to impose; treating the tabulated
amplitude as a tip-only linear twist would be a different geometry.

## Equation (11): gravity and axis transform

The printed parentheses matter.  The equation is

```text
[L, Tnet]^T = M(alpha) ([Fx, Fz]^T + G [sin(alpha), cos(alpha)]^T)
```

with

```text
M = [[sin(alpha), cos(alpha)], [cos(alpha), -sin(alpha)]].
```

After expansion, gravity contributes exactly `+G` to lift and exactly zero to
net thrust for every installation angle.  It is therefore incorrect to read
the typography as `M F + G[sin(alpha),cos(alpha)]`.  The executable form and a
reduction test live in `meng2025_case.py`.

## What `net thrust` does and does not establish

The paper reports the balance output after gravity correction and calls the
wind-axis streamwise component `net thrust`.  Positive is forward thrust and
negative is net drag.  At the Figure-16 condition all three digitized means
are negative.

The methods do **not** report any of the following:

- a wind-off flapping run or an inertial-load subtraction;
- an empty-support / mechanism tare in wind;
- subtraction of fuselage, drive, sting, or support parasite drag;
- a static-wing force baseline used to define the zero of `Tnet`;
- load-cell sampling rate, anti-alias treatment, repeat count, error bars, or
  run-to-run uncertainty;
- test-section air density.

The fifth-order 8 Hz Butterworth filter and four displayed cycles are stated
only for Figure 16.  At 2 Hz, the filter retains harmonics through the fourth
flapping harmonic.  Instantaneous curves can therefore contain structural
and inertial reaction loads.  Their exact periodic mean should cancel ideal
inertia, but offsets, mechanism/support aerodynamics, non-identical cycles and
filter/transient handling remain in the reported balance-level observable.

Accordingly:

- Figure-16/17--19 mean `Tnet` is an **instrument-level system observable**;
- a pure-wing FluxV streamwise force is not definition-identical to it;
- absolute thrust RMSE must not be described as pure aerodynamic-model error
  until a support/mechanism tare is available;
- none of these omissions is evidence of fabrication.  They are limitations
  of what can be independently checked from the paper.

## Figure 16 internal color ambiguity

Panel (a) maps black/red/green solid curves to 45/22.5/0 deg.  Panel (b)'s
solid-curve legend reverses green and black (green/red/black = 45/22.5/0), but
the dashed mean levels only agree with the Section-4.2 statement—0 and 22.5
have similar mean lift and 45 has the lowest mean lift—when dashed green/red/
black are mapped to 0/22.5/45 deg.  The digitized mean file uses that
text-consistent mapping and records it in every row.

Digitization calibration at 300 dpi:

- thrust axes: y=138 px -> +200 gf and y=688 px -> -700 gf;
- lift major ticks: y=171/256/341/426/511/596/681 px ->
  2000/1500/1000/500/0/-500/-1000 gf.

The resulting Figure-16 dashed means are:

| Twist peak-to-peak | Mean net thrust | Mean lift |
|---:|---:|---:|
| 0 deg | -139 +/-3 gf | 824 +/-10 gf |
| 22.5 deg | -80 +/-3 gf | 794 +/-10 gf |
| 45 deg | -230 +/-3 gf | 620 +/-10 gf |

These uncertainties describe plot-reading resolution only, not experimental
uncertainty.

## Frozen-transfer policy for FluxV v4b

No Meng force datum may set a model parameter.  A first transfer run must:

1. retain the prescribed-wake UVLM baseline;
2. use the already frozen full-angle-polar residual and causal incidence
   owner unchanged;
3. if the LDVM branch is exercised, label the Yang thin-plate `Lcrit=sin(5
   deg)` threshold as a cross-section transfer hypothesis, not a Meng-derived
   parameter;
4. add no fitted profile-drag constant, tare, scale, phase shift, or offset;
5. report old FluxV and v4b histories before looking at residual trends;
6. label absolute `Tnet` comparison as non-definition-identical.

## Reproducibility decision

- Geometry: **conditionally reconstructable**.  Overall dimensions are clear;
  outer outline and spar coordinate are drawing-derived.
- Nominal kinematics: **conditionally reconstructable**.  Amplitude, phase and
  sign are clear; per-condition encoder histories are not published.
- Mean lift: **usable as a balance-level diagnostic** with missing density and
  uncertainty metadata.
- Mean net thrust: **usable only as a system-level diagnostic**, not a clean
  pure-wing drag/thrust validation target.
- Instantaneous force: **not suitable for aerodynamic validation** without
  wind-off inertial histories and an unambiguous Figure-16(b) series mapping.
