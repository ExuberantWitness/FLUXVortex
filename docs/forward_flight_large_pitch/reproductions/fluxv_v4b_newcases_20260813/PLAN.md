# FluxV v4b new-case confirmation plan

## Scope frozen before scoring

This campaign evaluates the committed FluxV v4b implementation from
`8611fa573d02bd4d3d1918540fe3eff0766a26a6` on two additional forward-flight
datasets.  No aerodynamic-force observation may be used to tune a threshold,
gain, phase, geometry scale, or sign convention.

### Razak and Dimitriadis (2014), as reproduced by Lambert et al. (2017)

- Geometry: two rigid rectangular NACA 6409 wings; each wing has chord
  0.160 m, aerodynamic span 0.400 m, and a 0.150 m hinge-to-root offset.
- Nominal motion: root flap `gamma=30 sin(2 pi f t)` degrees.  Pitch-leading
  cases use `theta=theta0+theta_a cos(2 pi f t)`; pitch-lagging cases use
  `theta=theta0-theta_a cos(2 pi f t)`.
- The nominal sinusoids are an explicit reconstruction.  The unpublished
  measured cycle-averaged flap/pitch histories used by Lambert's simulations
  are not replaced by a fitted phase shift.
- Primary observations: experimental phase-resolved CL and CD in Figures
  9, 10, 11, 13, 14, and 15.  Figure 12 is excluded from absolute scoring
  because each drag trace was re-centred by subtracting its maximum.
- Primary scores: raw-phase CL/CD RMSE and MAE per figure, followed by an
  equal-case macro average.  No phase alignment or amplitude fit is allowed.
- NACA 6409 is retained because its thickness ratio is 9%, below the current
  exclusion threshold for 10%-plus thick sections.

### Meng et al. (2025)

- Use the paper planform, root flapping, and its rigid-wing rotation about the
  main spar. The paper calls this degree of freedom `twist`, but it does not
  publish a spanwise twist distribution; the executable adapter therefore
  treats it as the uniform rigid pitch shown by the mechanism and Figure 10.
- Freeze the present thin-wing v4b physics before comparing with Figure 16
  phase loads or Figures 17--19 mean loads.
- Audit the complete force ledger separately: sensor axes, coordinate
  transform, gravity term, aerodynamic tare/support drag, and wind-off
  inertial subtraction.  `Net thrust` is not silently converted to profile
  drag, and a missing subtraction is reported rather than inferred.
- The campaign tests consistency; it does not presume or claim misconduct.

## Exclusions requested for this campaign

- New NACA 0012/NACA 0013 cases are excluded because their 12%/13% thickness
  is outside the current thin-wing validation domain.
- Thielicke (2011) and Heathcote (2008) are not pursued.
- Yang (2023) and Baik (2012) receive source/geometry/data-completeness audits
  only; they are not used to modify v4b in this campaign.

## Model identities

- `FluxV old`: the current `UVPMHybridSolver` load channel, which is the
  prescribed-wake Ptera ring-UVLM load.  VPM particles do not feed back.
- `FluxV v4b frozen transfer`: old FluxV plus the committed source-derived
  separated-minus-attached LDVM discrepancy and causal persistent-incidence
  polar ownership.  It remains a development model, not a conservative 3-D
  material-LEV coupling.

## Acceptance and reporting

- Report every scored case, including regressions.
- A result is a transfer improvement only when both the force definition and
  the nominal motion are sufficiently closed.  Otherwise it is diagnostic.
- Run smoke first, then one-factor time/grid sensitivity before a full claim.
- After results are produced, a fresh reviewer recomputes headline metrics
  from the saved CSVs and audits hashes, missing observations, and exclusions.
