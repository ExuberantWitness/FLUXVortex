# Baik 2012 W1--W4 reproduction plan

## Objective

Reconstruct the W1--W4 pitching--plunging flat-plate experiments reported in
Baik's dissertation and the associated 2012 JFM study, then compare the
published corrected-total-load histories against:

1. the unchanged FluxV prescribed-wake UVLM load channel (`FluxV old`);
2. the already frozen FluxV v4b mechanism, without fitting phase, amplitude,
   offsets, or LESP parameters to W1--W4;
3. published Theodorsen references when they can be extracted without
   confusing them with experimental data.

## Frozen source and case contract

- Primary source for phase loads: Baik dissertation, Figures 5.24--5.27,
  printed pages 179--182 (physical PDF pages 204--207).
- Use corrected total hydrodynamic loads. Do not mix the AIAA precursor's
  pre-trigger-relative curves with the dissertation totals.
- Geometry: 76 mm chord, 600 mm span, 6.25%-thick stainless flat plate with
  rounded leading and trailing edges, quarter-chord pitch axis, end-plated
  water-channel configuration, nominal `Re=5000`.
- Cases: W1--W4 exactly as Table 1.4. In particular W3 has `k=1.0`, not the
  erroneous `k=0.5` printed in the AIAA precursor.
- Motion must enforce sinusoidal plunge-induced incidence through
  `h_dot=-U*tan(alpha_pl,max*sin(2*pi*f*t))`; a harmonic displacement is not an
  acceptable replacement at these Strouhal numbers.
- Experimental phase is used as published. No phase shift, amplitude scaling,
  mean-offset fitting, or model-derived ground truth is allowed.

## Route

1. Freeze and audit the experimental Cl/Cd curves with a reproducible vector
   or raster extraction and source hashes.
2. Implement a case/kinematics module with analytic/numerical invariants.
3. Add a runner that produces old/v4b histories, per-case metrics, aggregate
   metrics, manifests, and phase figures.
4. Run targeted tests, one bounded smoke matrix, then the full W1--W4 matrix.
5. Independently recompute metrics and audit ground-truth provenance, result
   existence, scope, numerical sensitivity, and claim strength.

## Comparison and metric contract

- Primary channels: phase-resolved `Cl` and `Cd`.
- Metrics are computed at the published experimental sample phases with
  periodic interpolation only: MAE, RMSE, signed bias, maximum absolute error,
  and range-normalized RMSE.
- Aggregate values are macro averages over four cases, so different
  digitization point counts cannot silently reweight a case.
- Report the paper-printed cycle means as independent consistency landmarks;
  do not force digitized curves or predictions to those means.
- Both per-case and aggregate results must be reported. A macro improvement
  cannot support a robustness claim if one or more cases materially regress.

## Frozen model-transfer rules

- FluxV old is the existing `UVPMHybridSolver` load channel; its VPM particles
  remain one-way and do not feed back into loads.
- Keep the v4b architecture unchanged: retained UVLM plus paired clean-room
  Ramesh-style LDVM discrepancy and causal persistent polar ownership.
- No W1--W4 force observation may select a threshold or blend weight.
- A section/Re-specific LESP threshold must have source provenance. If no
  defensible threshold exists, v4b must fail closed or be labelled an explicit
  transfer hypothesis; it may not be silently tuned.
- The physical 6.25% thickness is documented, while the UVLM mean-surface
  adapter cannot resolve thickness or viscous rounded-edge effects.

## Quality settings and acceptance

- Smoke: small UVLM grid/time history sufficient to validate signs, motion,
  finite outputs, data plumbing, and plots.
- Full: a declared production grid/time history plus a separate LDVM history;
  exact values are frozen in the implementation before scoring.
- Required hard checks: W1--W4 geometry/parameter invariants, periodic motion,
  correct effective-incidence range `[-6, 22] deg`, W3 `k=1`, coefficient sign
  convention, zero phase fitting, finite outputs, and result/source hashes.
- Scientific outcome may be `verified_improved`, `mixed`, `diverged`, or
  `blocked`; completion does not require v4b to outperform old FluxV.

## Main risks and fallback

- Figure curves may be raster-only; digitization uncertainty must be explicit.
- The water-channel end plates make the experiment quasi-2D, whereas the
  finite-wing UVLM model is only an adapter. Results therefore validate a
  section-level unsteady-load mechanism, not 3D bird-wing generalization.
- A published LESP threshold for this exact 6.25%-thick rounded flat plate at
  `Re=5000` may not exist. The fallback is a provenance-labelled flat-plate
  transfer sensitivity, never a W1--W4 fitted value.
- Smoke-to-full changes are diagnostic, not a one-factor convergence proof.
