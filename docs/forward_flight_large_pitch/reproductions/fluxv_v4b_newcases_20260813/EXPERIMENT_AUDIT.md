# Independent experiment audit

Review status: `WARN / qualified only`

Reviewer role: fresh same-family reviewer; no result-file edits.

Audit date: 2026-08-13.

## Integrity checks that passed

- The reviewer independently recomputed all Razak full per-case and macro
  RMSE/MAE/bias values from `scored_phase_samples.csv`; maximum discrepancy
  from saved metrics was `1.11e-16`.
- All six independent full case directories and the merged full directory had
  matching declared source/input/result hashes at review time.
- The merged histories, scored samples and metrics matched the six isolated
  case outputs row for row; all values were finite.
- Experimental phases were used directly. No phase shift, cross-correlation
  alignment, amplitude fit or observation-derived aerodynamic parameter was
  found.
- Figure 12 was excluded from absolute drag scoring as preregistered.
- The Meng runner stores executable old/polar outputs and records the forced
  Yang-threshold branch as failed/rejected rather than clipping or scoring it.
- The Meng documents explicitly reject fabrication inferences from missing
  tare metadata.

## Warnings

1. Razak full macro CL RMSE improves 5.78% and CD RMSE improves 2.28%, but CD
   MAE worsens 4.23%. Figure 13 regresses and Figure 15 CD regresses severely;
   a robust across-case/across-load improvement claim fails.
2. Smoke and full simultaneously change UVLM grid, step count, cycle count and
   wake retention. Per-figure v4b RMSE changes by more than 20% in some cases.
   There is no one-factor convergence proof.
3. The Razak model uses an extrema-matched nominal sinusoid and assumed
   quarter-chord pitch axis because measured 64-point motions and the rig axis
   are not public. Results are transfer diagnostics, not exact self-validation.
4. The v4b bridge remains independent 2-D LDVM strips added to a finite-wing
   load. Figure 13's direct 128-step strip solve is singular; 512 steps avoids
   that failure but is not a convergence demonstration.
5. Meng `net thrust` is an instrument-level balance observable without a
   published support/mechanism/wind-off tare. It is not definition-identical
   to pure-wing FluxV thrust.
6. Yang/Baik audit notes contain stable URLs and key figure/page references,
   but the Baik source PDF is not vendored in this branch; source-package
   provenance remains weaker than the Razak vector package.

## Allowed claims

- On the frozen Razak six-case full reconstruction, v4b reduces macro CL RMSE
  from 0.24689 to 0.23262 and macro CD RMSE from 0.05474 to 0.05349.
- The improvement is mixed: the lift correction is useful in Figure 15 and
  the drag correction in Figure 14, while other cases regress or do not change.
- Meng Figure 16 old/polar predictions reproduce the broad decreasing trend
  with increasing rigid-pitch amplitude, but full v4b transfer is currently
  blocked and absolute thrust accuracy is not identifiable from the paper.

## Disallowed claims

- FluxV v4b is uniformly or robustly more accurate on Razak.
- Razak full results are grid/time/wake converged.
- Meng validates or falsifies full FluxV v4b.
- Missing Meng tare metadata proves incorrect definitions, poor experimental
  practice, fabrication, or data falsification.
