# FluxV v4b mechanism report

## Problem anchor

The retained prescribed-wake UVLM load channel is useful in attached flow but
misses intermittent leading-edge separation, axial-suction loss and separated
pressure drag.  A periodic post-processing gate improved Yang 2025 but failed
the Scherer experiment in Izraelevitz Figure 14.  The replacement must therefore
be causal and must not remove the finite-wing UVLM geometry/wake solution.

## Implemented mechanism

The v4b development model uses two exclusive phase-resolved branches:

1. transient/reversing flow: retained UVLM plus a Ramesh LDVM
   `separated - attached` discrepancy;
2. persistent installed incidence: retained UVLM plus the existing full-angle
   polar discrepancy.

A causal two-pole signed/absolute-incidence state selects the branches.  Loads
are blended at each phase sample and only then cycle-averaged.  This avoids the
discarded v4a error of multiplying cycle-mean gates by cycle-mean loads.

The LDVM discrepancy is split into four auditable ledgers:

- circulatory normal force, scaled by the finite-wing lift-slope ratio `g`;
- non-circulatory force, scaled by the Izraelevitz AR-dependent added-mass
  factor (`0.85` at AR=3, `0.95` at AR=6);
- wake/bound nonlinear normal force, scaled by `g^2`;
- Ramesh axial suction, scaled by `g^2`.

The material-vortex core is fixed at the source value `r_c/c=0.02`; it no
longer changes when the time step changes.  Yang strip lift increments are
projected to the global vertical direction with the instantaneous flapping
angle.

## Exact reductions and ownership

- If the LESP threshold is not crossed, separated and attached LDVM histories
  are identical and the LDVM discrepancy is exactly zero.
- UVLM remains the sole transient baseline load owner.
- ULLT is an external attached-flow reference and the Figure-11 no-LEV limit;
  it is not added simultaneously to the UVLM load.
- Profile drag is retained once in the Scherer source ledger.

This is still an additive discrepancy model.  It is not yet a conservative,
globally coupled 3-D LEV wake in the UVLM AIC.  The next production step is a
spanwise coupled material-LEV row with a shared Kelvin ledger.

## Parameter provenance

- Ramesh reference: source-faithful LESP/Kelvin form and `r_c/c=0.02`.
- Yang: `Lcrit=sin(5 deg)` is an explicit mapping of the published Yang
  separation angle, not a force fit.
- Scherer Figure 14: `Lcrit=sin(CLmax/CLa)=0.239...`, using Scherer's static
  `CLmax=0.90` and `CLa=0.065/deg`; this is a hypothesis and Figure 14 is a
  development set.
- Izraelevitz Figure 11: NACA0012 `Lcrit=0.29`; the threshold is not crossed,
  so the model exactly uses the frozen v2 ULLT attached limit.
- Stevens: the pre-existing thin-plate LDVM value `0.11`; no Stevens force
  sweep selected it, although Stevens is conservatively classified as a
  development-transfer test rather than untouched holdout.

## Mechanism-level limitations

- The LDVM strips do not mutually induce one another and do not feed a 3-D LEV
  row back into the UVLM AIC.
- The threshold remains section/Re dependent; a universal numerical `Lcrit`
  is neither claimed nor supported by the sources.
- The causal persistence ratio is a new source-assisted heuristic; the two
  poles are traceable to Izraelevitz, but their signed/absolute ratio is not a
  uniquely derived paper equation.
- Figure 14 supplies cycle-mean thrust only; it cannot validate the predicted
  phase-resolved correction.
- Stevens publishes lift only, so no experimental drag score is possible.
