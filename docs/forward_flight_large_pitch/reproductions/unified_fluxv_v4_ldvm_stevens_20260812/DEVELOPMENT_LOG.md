# FluxV v4 development log

This file records rejected variants so the final model is not presented as if
it were the first or only attempted formulation.

## Frozen baselines

- Yang 2025 old FluxV: lift/drag MAE `6.85487 / 12.92166 gf`.
- Yang 2025 exploratory v2: `3.95162 / 2.06249 gf`.
- Figure 14 old FluxV: mean-thrust RMSE `0.0511466`.
- Figure 14 one-state ULLT: RMSE `0.0463608`.
- Figure 14 failed v1/v2 polar owner: RMSE `0.222598`.
- Izraelevitz Figure 11 old FluxV: lift/drag RMSE `2.43099 / 0.95313`.
- Figure 11 attached one-state ULLT: `0.154624 / 0.314200`.

## Rejected or limited variants

### Standalone two-dimensional LDVM

The first Stevens smoke comparison used the 2-D LDVM as the complete model.
It improved the leading-edge-axis lift RMSE from `1.11750` to `1.02344` but
degraded the mid-chord-axis result from `0.57629` to `1.06325`.  This falsified
the idea that LDVM should replace the finite-wing UVLM.

### Common finite-wing gain for all LDVM force components

Applying one Prandtl slope factor to both normal force and axial suction was
rejected. Ramesh's `Cs=2*pi*A0^2` is quadratic in LESP/circulation. The final
diagnostic separates circulatory, non-circulatory, wake-nonlinear and axial-
suction ledgers; it applies `g`, the AR-dependent added-mass factor, `g^2`, and
`g^2`, respectively.

### First causal-owner run with dimensional inconsistency

Run `20260812_fluxv_v4_crosspaper_spc128` incorrectly combined a convective-
time angular rate with dimensional strip radii and dimensional freestream.
This reduced the flap-induced incidence below one degree and forced the Yang
persistence to one for all installation angles above zero.  It is retained as
failed evidence and must never be cited.  A regression test now requires the
zero-installation Yang effective-incidence range to exceed +/-20 degrees.

### Shedding-fraction gate applied to the persistent branch

Run `20260812_fluxv_v4_crosspaper_spc128_v3` multiplied even the persistent
stall owner by the instantaneous new-LEV shedding fraction.  At high Yang
installation angles this incorrectly returned most load ownership to attached
ULLT, producing lift/drag MAE `7.658 / 9.795 gf`.  The persistent owner must
remain active after onset even when a new discrete vortex is not born at every
sample.  This variant was rejected.

## Current developmental variant

The current model uses:

1. retained UVLM plus separated-minus-attached LDVM increments for transient
   shedding;
2. retained UVLM plus full-angle polar for persistent incidence;
3. a causal two-pole incidence state using Izraelevitz circulation-state poles
   `0.30` and `0.045` per convective time;
4. section/Re-specific LESP threshold hypotheses with explicit provenance.

The ULLT solution is not mixed into the transient v4b load. It is retained as
the separate Figure-11 attached/no-onset reference. This removes the v4a
mean-level three-owner ambiguity but does not turn the additive discrepancy
into a conservative common-wake coupling.

This remains a development diagnostic because its material LEV wakes are
independent two-dimensional sections.  They do not yet feed a globally coupled
spanwise LEV row into the UVLM AIC.
