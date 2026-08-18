# FluxV v5a cross-paper runner contract

## Scope

This contract keeps the Yang 2025, Scherer/Izraelevitz Figure 14, and Baik
2012 W1--W4 scoring rules frozen while the v5a force ledger is implemented.
It separates a cheap cache-reuse development run from a canonical run.  The
two must not be given the same evidence status.

## Incidence evidence levels

| `incidence_source` | Meaning | Permitted claim |
|---|---|---|
| `kinematic_proxy` | Local relative flow is computed from freestream and prescribed surface motion only. | Auxiliary/development result. |
| `uvlm_induced` | Local relative flow also contains the induced velocity from the same UVLM time layer that owns the baseline load. | Canonical v5a candidate. |

The existing `movement_polar_residual()` and all frozen polar histories use
the first definition.  It computes `freestream - panel_velocity` and does not
read the UVLM induced velocity.  The frozen phase CSVs store integrated loads,
not strip-induced velocities.  Consequently, a cache-only run is always
`kinematic_proxy`, even if the variable has historically been called local or
effective incidence.  A canonical gate must fail closed until an induced-
velocity strip history is exported and consumed.

## Frozen inputs

### Yang 2025

- Baseline and equilibrium-provider history:
  `unified_fluxv_upgrade_20260812/runs/20260812_periodic_v2_ullt_full/yang2025_phase_histories.csv`.
- Use `fluxv_uvpm` as the retained UVLM history.
- The cache-reuse equilibrium residual is
  `fluxv_periodic_v1 - fluxv_uvpm` at each of the 128 frozen phase samples.
- There is no separate profile-drag coefficient in the Yang ledger.
- The paired-LDVM residual must be recomputed because v4b froze its means but
  did not freeze the full v4b/LDVM phase history.
- Yang publishes only six cycle means.  Internal phase histories must never be
  scored or used to select a parameter.

### Scherer/Izraelevitz Figure 14

- Retained baseline history:
  `unified_fluxv_v4_ldvm_stevens_20260812/source_data/izraelevitz2017_fig14_local_phase_cache.csv`, model `fluxv_uvpm`.
- `Cd0=0.057` is already present in the retained baseline and must not be
  added again.  The equilibrium residual excludes this common baseline.
- The old cached `fluxv_periodic_v1` residual was evaluated at the default
  quarter-chord velocity reference.  It is not a valid canonical v5a
  equilibrium provider for a three-quarter-chord pivot.  A development runner
  may retain it only with `incidence_source=kinematic_proxy` and an explicit
  `quarter_chord_cache` warning.  The preferred development fix is to rebuild
  the motion and evaluate the geometry-only polar residual at `0.75c`; this is
  still a kinematic proxy, not an induced-incidence history.
- Figure 14 scores cycle-mean `CT=-CD` only.  The 14 plotted observations are
  the primary score; the 12 unique conditions are a secondary equal-condition
  score.

### Baik 2012 W1--W4

- Baseline, v4b and paired-LDVM histories:
  `baik2012_w1_w4/runs/20260813_baik2012_w1_w4_full_reproducible/model_phase_histories.csv`.
- Use unfiltered `fluxv_old` as the retained UVLM input and the unfiltered v4b
  rows for the stored `persistence`, `ldvm_delta_CL`, and `ldvm_delta_CD`.
- The old runner did not store the equilibrium-provider history directly.  It
  can be recovered exactly on this frozen run where `p>0`:

  `polar = (v4b - (1-p)*(old + ldvm_delta))/p`.

  The recovered cache must be labelled `derived_from_v4b_ledger`; it is a
  reproducibility bridge, not a new physical provider.
- Baseline section profile drag is zero in this adapter.  The equilibrium
  provider owns only its declared nonlinear section-drag residual; UVLM
  induced drag remains in the baseline.
- Score the 400 unique experimental phase samples (drop the duplicate endpoint)
  without phase, amplitude or offset fitting.  Report raw and the source's
  ideal 1 Hz Fourier-filtered view.  Retained harmonics are W1/W4: 7 and
  W2/W3: 3.

## Core ledger interface

The runner passes aligned coefficient histories to a pure v5a core:

```text
baseline:       CL/CD, shape (n,)
equilibrium:    delta_CL_eq/delta_CD_eq, shape (n,)
ldvm_delta:     delta_CL/delta_CD, shape (n,)
delta_chi:      scalar or shape (n,), local convective increment
lambda_tau:     1.0 primary; 0.5 and 2.0 sensitivity only
mode:           full | equilibrium_only | transient_only
```

The core returns final `CL/CD` plus equilibrium, raw LDVM, low-pass state,
high-pass transient and ledger-closure histories.  Dynamic pressure and area
must not be multiplied inside both runner and core.  The cross-paper runner
uses dimensionless histories and converts to Yang gram-force only after the
ledger closes.

## Output schema

Every run directory must contain:

- `phase_histories.csv`: benchmark, case, model, phase, retained baseline,
  equilibrium residual, raw LDVM residual, low-pass state, transient residual,
  final coefficient, incidence source and provider provenance.
- `case_metrics.csv`: benchmark, case, model, quantity, raw/filtered/mean view,
  observation count, MAE, RMSE, bias and maximum absolute error.
- `gate_results.csv`: gate ID, measured value, comparator/threshold, relation,
  pass/fail, canonical eligibility and failure reason.
- `summary.json`: exact CLI, source/result hashes, parameter manifest, metric
  contract, canonical/partial status and all gate outcomes.

No output may be labelled canonical when any selected case uses
`kinematic_proxy`, an integrated rather than strip-local ledger, or a
non-converged state history.

## Frozen gates

### Yang

- Lift/drag MAE at most `4.327/2.512 gf`.
- No installation angle worsens by more than `0.4 gf` relative to v4b.
- 5 and 10 degree lift and 15 degree drag improve materially.
- The 20 and 25 degree saturation benefit is retained.

### Figure 14

- 14-marker RMSE at most `0.0230`.
- 15/25 degree subgroup RMSE at most `0.02268/0.02284`.
- 12-condition RMSE below `0.02751`.
- Maximum error at most `0.050`; single-observation regression at most
  `0.0112` relative to v4b.

### Baik

- Filtered macro-case CL/CD RMSE each improve by at least 5% over v4b.
- All eight case/channel pairs are non-regressive.
- W2 Q1 and W3 Q1--Q2 lift improve; no phase-quadrant RMSE worsens by more
  than 5%.
- Raw and filtered metrics are both emitted.

### Cross-paper eligibility

- Figure 11 attached regression is within 2% and proves `r=m=R_eq=0`.
- Module-off identity and relative ledger closure below `1e-12` pass.
- The state is periodic to `1e-4` and headline time/span/chord/wake
  sensitivities are below 5%.
- All selected cases use `incidence_source=uvlm_induced` for a canonical v5a
  promotion.  Otherwise numeric gates may be shown, but the overall verdict is
  `partial_development_only`.

## Commands

The intended commands are:

```bash
python -m forward_flight_benchmarks.run_fluxv_v5a_crosspaper \
  --quality smoke --incidence-source kinematic-proxy \
  --output-dir <new-empty-smoke-dir>

python -m forward_flight_benchmarks.run_fluxv_v5a_crosspaper \
  --quality full --incidence-source uvlm-induced --require-canonical \
  --output-dir <new-empty-full-dir>
```

The first command is a mechanism and integration smoke test.  The second must
refuse canonical promotion if the UVLM-induced strip-incidence exporter is not
available; it must never silently fall back to the first definition.
